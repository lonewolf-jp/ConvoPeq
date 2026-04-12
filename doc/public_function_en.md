Here is the English translation of the public API documentation in Doxygen format.

---

# ConvoPeq Public API Documentation

## 1. AudioEngine

`AudioEngine` is the central class that orchestrates all audio processing for the application.

### 1.1 Audio I/O Lifecycle

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void prepareToPlay(int samplesPerBlockExpected, double sampleRate)` | Prepares the audio playback. Requests a rebuild of the internal DSP graph. | **Message Thread** only. |
| `void releaseResources()` | Releases resources when the audio device is stopped. | **Message Thread** only. |
| `void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)` | Audio callback. Executes DSP processing. | **Audio Thread** only. Subject to real-time constraints. |

### 1.2 Parameter Control

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void setInputHeadroomDb(float db)` | Sets the input headroom gain (dB). | **Message Thread**. Updates internal atomic variables. |
| `float getInputHeadroomDb() const` | Returns the current input headroom gain (dB). | Thread-safe (atomic load). |
| `void setOutputMakeupDb(float db)` | Sets the output makeup gain (dB). | **Message Thread**. |
| `float getOutputMakeupDb() const` | Returns the current output makeup gain (dB). | Thread-safe (atomic load). |
| `void setProcessingOrder(ProcessingOrder order)` | Sets the processing order (`ConvolverThenEQ` or `EQThenConvolver`). | **Message Thread**. |
| `ProcessingOrder getProcessingOrder() const` | Returns the current processing order. | Thread-safe (atomic load). |
| `void setEqBypassRequested(bool shouldBypass)` | Requests EQ bypass state. | **Message Thread**. |
| `bool isEqBypassRequested() const` | Returns the requested EQ bypass state. | Thread-safe (atomic load). |
| `void setConvolverBypassRequested(bool shouldBypass)` | Requests convolver bypass state. | **Message Thread**. |
| `bool isConvolverBypassRequested() const` | Returns the requested convolver bypass state. | Thread-safe (atomic load). |

### 1.3 Convolver / EQ Accessors

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `ConvolverProcessor& getConvolverProcessor()` | Returns a reference to the UI `ConvolverProcessor` instance. | **Message Thread** only. |
| `EQEditProcessor& getEQProcessor()` | Returns a reference to the UI `EQEditProcessor` instance. | **Message Thread** only. |

### 1.4 Noise Shaper Learning

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume)` | Starts noise shaper learning. | **Message Thread**. Launches a worker thread as an asynchronous task. |
| `void stopNoiseShaperLearning()` | Stops noise shaper learning. | **Message Thread**. |
| `bool isNoiseShaperLearning() const` | Returns whether learning is currently in progress. | Thread-safe (atomic load). |
| `const NoiseShaperLearner::Progress& getNoiseShaperLearningProgress() const` | Returns the learning progress. | Thread-safe. The returned reference contains atomic variables. |

---

## 2. ConvolverProcessor

`ConvolverProcessor` handles IR file loading, management, and convolution processing.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void prepareToPlay(double sampleRate, int samplesPerBlock)` | Prepares the convolution engine. | **Message Thread**. |
| `void releaseResources()` | Releases engine resources. | **Message Thread**. |
| `void process(juce::dsp::AudioBlock<double>& block)` | Performs convolution on the input block. | **Audio Thread** only. Real-time constraints apply. |
| `bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false)` | Initiates asynchronous loading of the specified IR file. | **Message Thread**. |
| `void setMix(float mixAmount)` | Sets the dry/wet mix ratio (0.0–1.0). | **Message Thread**. |
| `float getMix() const` | Returns the current mix ratio. | Thread-safe (atomic load). |
| `void setPhaseMode(PhaseMode mode)` | Sets the phase mode (AsIs, Mixed, Minimum). | **Message Thread**. |
| `PhaseMode getPhaseMode() const` | Returns the current phase mode. | Thread-safe (atomic load). |
| `void setSmoothingTime(float timeSec)` | Sets the mix smoothing time (seconds). | **Message Thread**. |
| `float getSmoothingTime() const` | Returns the current smoothing time. | Thread-safe (atomic load). |
| `bool isIRLoaded() const` | Returns whether an IR is currently loaded. | Thread-safe (RCU). |
| `juce::String getIRName() const` | Returns the filename of the currently loaded IR. | **Message Thread**. |
| `int getLatencySamples() const` | Returns the current algorithmic latency in samples. | Thread-safe (returns cached value). |

---

## 3. EQProcessor

`EQProcessor` provides DSP processing for the 20-band parametric EQ.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void prepareToPlay(double sampleRate, int maxBlockSize)` | Sets the sample rate and maximum block size, and resets internal state. | **Message Thread**. |
| `void process(juce::dsp::AudioBlock<double>& block)` | Performs EQ processing on the input block. | **Audio Thread** only. |
| `void setBandFrequency(int band, float freq)` | Sets the frequency (Hz) for the specified band. | **Message Thread**. |
| `void setBandGain(int band, float gainDb)` | Sets the gain (dB) for the specified band. | **Message Thread**. |
| `void setBandQ(int band, float q)` | Sets the Q factor for the specified band. | **Message Thread**. |
| `void setBandEnabled(int band, bool enabled)` | Enables or disables the specified band. | **Message Thread**. |
| `EQBandParams getBandParams(int band) const` | Returns the current parameters for the specified band. | **Message Thread**. |
| `void setTotalGain(float gainDb)` | Sets the total gain (dB). | **Message Thread**. |
| `float getTotalGain() const` | Returns the current total gain. | **Message Thread**. |
| `void setAGCEnabled(bool enabled)` | Enables or disables Automatic Gain Control (AGC). | **Message Thread**. |
| `bool getAGCEnabled() const` | Returns the current AGC enabled state. | Thread-safe (atomic load). |

---

## 4. MKLNonUniformConvolver

`MKLNonUniformConvolver` is an Intel IPP-based non-uniform partitioned convolution engine.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `bool SetImpulse(const double* impulse, int irLen, int blockSize, double scale, bool enableDirectHead, const FilterSpec* filterSpec)` | Configures the IR data and builds internal FFT structures. | **Message Thread** only. |
| `void Add(const double* input, int numSamples)` | Feeds input samples into the engine. | **Audio Thread** only. |
| `int Get(double* output, int numSamples)` | Writes convolution results to the output buffer. | **Audio Thread** only. |
| `void Reset()` | Clears internal buffers and state to zero. | May be called from **Message Thread** or `releaseResources()`. |
| `bool isReady() const` | Returns whether the engine is in a usable state. | Thread-safe (atomic load). |
| `int getLatency() const` | Returns the engine's upfront latency in samples. | Thread-safe. |

---

## 5. SnapshotCoordinator

`SnapshotCoordinator` manages DSP parameter snapshots in a lock-free manner.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `const GlobalSnapshot* getCurrent() const` | Returns a pointer to the currently active snapshot. | **Audio Thread** safe (atomic load). |
| `void switchImmediate(const GlobalSnapshot* newSnap)` | Switches to a new snapshot immediately. | **Message Thread** / **Worker Thread**. |
| `void startFade(const GlobalSnapshot* target, int fadeSamples)` | Initiates a crossfade over the specified number of samples. | **Message Thread** / **Worker Thread**. |
| `bool updateFade(float& outAlpha, const GlobalSnapshot*& outCurrent, const GlobalSnapshot*& outTarget)` | Updates the fade state and returns the current alpha and snapshots. | **Audio Thread** only. |
| `void advanceFade(int numSamples)` | Advances the fade progress. | **Audio Thread** only. |
| `bool tryCompleteFade()` | Attempts to complete the fade and updates state upon success. | **Message Thread** (timer callback). |

---

## 6. WorkerThread

`WorkerThread` is a dedicated thread that debounces parameter change commands and requests snapshot creation.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void start()` | Starts the worker thread. | **Message Thread**. |
| `void stop()` | Stops the worker thread. | **Message Thread**. |
| `void flush()` | Requests immediate processing of pending commands. | **Message Thread**. |
| `void setSnapshotCreator(SnapshotCreatorCallback callback, void* userData)` | Sets the snapshot creation callback. | **Message Thread**. |

---

## 7. SafeStateSwapper (RCU)

`SafeStateSwapper` provides an epoch-based RCU (Read-Copy-Update) mechanism.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void swap(ConvolverState* newState)` | Publishes a new state and places the old state into the retired queue. | **Message Thread** only. |
| `void enterReader(int readerIndex)` | Declares that a reader is entering a critical section. | **Audio Thread** only. |
| `void exitReader(int readerIndex)` | Declares that a reader is exiting a critical section. | **Audio Thread** only. |
| `ConvolverState* getState() const` | Returns a pointer to the currently active state. | **Audio Thread** only (called between `enterReader()`/`exitReader()`). |
| `ConvolverState* tryReclaim(uint64_t minReaderEpoch)` | Retrieves one reclaimable old state object. | **DeferredFree Thread** only. |
| `uint64_t getMinReaderEpoch() const` | Returns the minimum epoch among all active readers. | **DeferredFree Thread** only. |

---

## 8. NoiseShaperLearner

`NoiseShaperLearner` optimizes lattice noise shaper coefficients using CMA-ES.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `void startLearning(bool resume = false)` | Starts (or resumes) learning. | **Message Thread**. |
| `void stopLearning()` | Stops learning. | **Message Thread**. |
| `bool isRunning() const` | Returns whether learning is currently in progress. | Thread-safe. |
| `void setLearningMode(LearningMode mode)` | Sets the learning mode (Short, Middle, Long, Ultra, Continuous). | **Message Thread**. |
| `const Progress& getProgress() const` | Returns the learning progress. | Thread-safe. |
| `int copyBestScoreHistory(double* outScores, int maxPoints) const` | Copies the best score history. | **Message Thread**. |

---

## 9. EQCoeffCache

`EQCoeffCache` is an immutable cache that holds EQ coefficients and parallel-mode work buffers.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `static uint64_t computeParamsHash(const convo::EQParameters& params)` | Computes a hash value for the given EQ parameters. | Thread-safe (pure function). |
| `static EQCoeffCache* createCoeffCache(const convo::EQParameters& eqParams, double sampleRate, int maxBlockSize, uint64_t generation)` | Creates a new cache instance. | **Message Thread** / **Worker Thread**. |

---

## 10. AllpassDesigner

`AllpassDesigner` designs allpass filters that approximate a target group delay.

| Function Signature | Description | Thread Safety |
| :--- | :--- | :--- |
| `bool design(double sampleRate, const std::vector<double>& freq_hz, const std::vector<double>& target_group_delay_samples, const Config& config, std::vector<SecondOrderAllpass>& sections, const std::function<bool()>& shouldExit, std::function<void(float)> progressCallback)` | Designs an allpass filter (using Greedy+AdaGrad or CMA-ES). | May be called from **Loader Thread**. |
| `static juce::AudioBuffer<double> applyAllpassToIR(const juce::AudioBuffer<double>& linearIR, const std::vector<SecondOrderAllpass>& sections, double sampleRate, const std::vector<double>& freq_hz, int fftSize, const std::function<bool()>& shouldExit, std::function<void(float)> progressCallback)` | Applies a designed allpass filter to an IR. | May be called from **Loader Thread**. |