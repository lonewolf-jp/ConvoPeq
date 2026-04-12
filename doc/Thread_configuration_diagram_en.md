Here is the English translation of the provided document detailing the thread architecture and data access rules for ConvoPeq.

## 1. Thread Architecture Diagram

```mermaid
graph TD
    subgraph "Hard Real-Time Constraints (Hard RT)"
        AT[Audio Thread<br>getNextAudioBlock()]
    end

    subgraph "Soft Real-Time / Asynchronous Processing (Non-RT)"
        MT[Message Thread<br>JUCE Main / UI]
        WT[Worker Thread<br>Snapshot Debounce]
        RT[Rebuild Thread<br>DSP Graph Builder]
        LT[Loader Thread<br>IR File I/O & Resample]
        UT[ProgressiveUpgrade Thread<br>IR Cache Upgrade]
        LMT[Learner Main Thread<br>CMA-ES Optimizer]
        LET[Learner Eval Thread(s)<br>Parallel Cost Evaluation]
        DFT[DeferredFree Thread<br>RCU Garbage Collector]
        RNGT[RNG Producer Thread<br>Dither Noise Prefill]
    end

    AT <-->|Atomic / RCU / SPSC| MT
    MT -->|CommandBuffer (SPSC)| WT
    MT -->|Task Start| RT
    MT -->|Task Start| LT
    MT -->|Task Start| UT
    MT -->|Task Start| LMT
    WT -->|Snapshot Atomic Swap| AT
    RT -->|DSPCore Atomic Swap| AT
    LT -->|IRState RCU Swap| AT
    UT -->|PreparedIRState RCU Swap| AT
    LMT -->|CoeffSet RCU Swap| AT
    DFT -->|Delayed free| AT
    RNGT -->|RingBuffer (SPSC)| AT
```

## 2. Thread Roles and Constraints

| Thread Name | Priority/Affinity | Primary Responsibility | Forbidden Operations |
| :--- | :--- | :--- | :--- |
| **Audio Thread** | `MMCSS Pro Audio` / `Worker` Core | Executes `getNextAudioBlock()`. Drives the entire DSP chain (Convolver, EQ, NS). | **Strictly Forbidden**: `malloc`/`new`/`mkl_malloc`, `std::mutex`, File I/O, libm calls (e.g., `std::exp`), `DftiCommitDescriptor`, `try`/`catch`, `std::shared_ptr` operations. |
| **Message Thread** | `UI` Core | JUCE main loop. Handles UI events, initiates all asynchronous tasks, saves device settings. | Long-duration locks that block the Audio Thread. |
| **Worker Thread** | `Worker` Core | Dedicated to `SnapshotCoordinator`. Debounces parameter change commands from the UI and generates `GlobalSnapshot` asynchronously. | Direct manipulation of the Audio Thread. |
| **Rebuild Thread** | `HeavyBackground` Core | Reconstructs the entire DSP graph (`DSPCore`). Handles IR reuse decisions and delegates NUC engine initialization. | Execution on the Audio Thread. |
| **Loader Thread** | `HeavyBackground` Core | Reads IR files, performs `r8brain` resampling, phase conversion. Generates `PreparedIRState` and saves to cache. | Execution on the Audio Thread. |
| **Progressive Upgrade Thread** | `HeavyBackground` Core | Performs stepwise upgrade from low-resolution IR cache to high-resolution IR. | Execution on the Audio Thread. |
| **Learner Main Thread** | `LearnerMain` Core | Runs the CMA-ES optimization loop for noise shaper coefficients. Manages evaluation workers. | Execution on the Audio Thread. |
| **Learner Eval Thread(s)** | `LearnerEval` Core(s) | Executes parallel cost function evaluation for candidate coefficients. | Execution on the Audio Thread. |
| **DeferredFree Thread** | `LightBackground` | Monitors the `SafeStateSwapper` retired queue and `delete`s `ConvolverState` objects once safe. | `delete`/`mkl_free` on the Audio Thread. |
| **RNG Producer Thread** | `Lowest` | Refills the random number ring buffer used by `PsychoacousticDither` in the background. | Interfering with Audio Thread real-time performance. |

## 3. Formalization of Data Access Rules

### 3.1 Core Principles

| Data Type | Synchronization Mechanism | Implementation Pattern | Memory Lifecycle |
| :--- | :--- | :--- | :--- |
| **Scalar Settings** | `std::atomic<T>` | Non-RT thread `store(release)`, Audio Thread `load(acquire)`. | N/A (value type). |
| **Structured Parameters** | **RCU (Read-Copy-Update)** | `SafeStateSwapper` + `DeletionQueue`. `swap()` publishes new state, `tryReclaim()` delays deletion. | Producer (Non-RT) allocates with `new`/`mkl_malloc`. Audio Thread reads only. Dedicated thread handles deletion. |
| **Large Data Buffers** | **SPSC FIFO + RCU** | Pointer handoff via `LockFreeRingBuffer`. | Producer allocates. Consumer (Audio) returns pointer via **Return FIFO** to Non-RT for safe deallocation. |
| **Commands/Messages** | **SPSC FIFO** | `CommandBuffer`, `LearningCommand` queue. | Value copy. |

### 3.2 Detailed Rules per Key Data Structure

#### A. GlobalSnapshot (All DSP Parameters)
- **Path**: `UI (Message)` → `CommandBuffer` → `Worker Thread` → `SnapshotCoordinator`
- **Audio Thread Access**: `m_coordinator.getCurrent()` (atomic load)
- **Update Method**: `SnapshotCoordinator::startFade()` performs atomic swap.
- **Deallocation**: Via `DeletionQueue`, `SnapshotFactory::destroy()` called after all readers have exited.

#### B. ConvolverState (IR Frequency Domain Data)
- **Path**: `Loader Thread` → `Message Thread` (callAsync) → `SafeStateSwapper::swap()`
- **Audio Thread Access**: `ConvolverProcessor::acquireIRState()` calls `enterReader()` → reads → `exitReader()`
- **Update Method**: `rcuSwapper.swap()`
- **Deallocation**: `DeferredFreeThread` retrieves via `tryReclaim()` and calls `delete`.

#### C. EQCoeffCache (EQ Coefficient Cache)
- **Path**: `Worker Thread` (`createSnapshotFromCurrentState`) → `EQCacheManager::getOrCreate()`
- **Audio Thread Access**: `EQProcessor::process(..., cache)` reads coefficients from cache.
- **Update Method**: `CacheMap` is atomically swapped (`cacheMapPtr`). Old map is queued for delayed deletion via `g_deletionQueue`.

#### D. Adaptive Noise Shaper Coefficients (Learned Coefficients)
- **Path**: `Learner Main Thread` → `MessageManager::callAsync` → `AudioEngine::publishCoeffsToBank()`
- **Audio Thread Access**: `getActiveCoeffSet()` (RCU) inside `DSPCore::processOutput()`
- **Update Method**: `CoeffSetWriteLockGuard` atomically swaps the double buffer.

#### E. Audio Capture Queue (Training Signal)
- **Path**: `Audio Thread` → `LockFreeRingBuffer<AudioBlock>` → `Learner Main Thread`
- **Direction**: Audio (Producer) → Non-RT (Consumer)
- **Note**: Value type (`AudioBlock`), no deallocation needed. Drop-oldest policy on overflow.

### 3.3 Concrete Examples of Thread Interaction

#### Example 1: IR File Loading
1. **Message Thread**: `ConvolverProcessor::loadIR()` → Creates and starts `LoaderThread`.
2. **Loader Thread**: Reads file → Resamples → Generates `PreparedIRState` → `MessageManager::callAsync()`.
3. **Message Thread**: `applyPreparedIRState()` → `updateConvolverState()` → `rcuSwapper.swap()`.
4. **Audio Thread**: On next `process()` call, acquires new IR state via `acquireIRState()` and begins convolution.
5. **DeferredFree Thread**: After confirming no active readers reference the old `ConvolverState`, it is `delete`d.

#### Example 2: EQ Parameter Change
1. **Message Thread**: `EQEditProcessor::setBandGain()` → `scheduleDebounce()` (50ms timer).
2. **Message Thread (Timer)**: `enqueueSnapshotCommand()` → `CommandBuffer::push()`.
3. **Worker Thread**: `CommandBuffer::pop()` → `createSnapshotFromCurrentState()` → `SnapshotFactory::create()` → `SnapshotCoordinator::startFade()`.
4. **Audio Thread**: On next `getNextAudioBlock()`, `updateFade()` is called, crossfading between old and new snapshots during processing.