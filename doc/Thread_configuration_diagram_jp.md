以下は、ConvoPeq ソースコードに基づく**完全なスレッド構成図**と**データアクセス規則の明文化**です。

## 1. スレッド構成図 (Thread Architecture)

```mermaid
graph TD
    subgraph "ハードリアルタイム制約 (Hard RT)"
        AT[Audio Thread<br>getNextAudioBlock()]
    end

    subgraph "ソフトリアルタイム / 非同期処理 (Non-RT)"
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

## 2. スレッド別役割と制約詳細

| スレッド名 | 優先度/アフィニティ | 主要責務 | 禁止事項 |
| :--- | :--- | :--- | :--- |
| **Audio Thread** | `MMCSS Pro Audio` / `Worker` Core | `getNextAudioBlock()` を実行。DSPチェーン全体（Convolver, EQ, NS）を駆動。 | **絶対禁止**: `malloc`/`new`/`mkl_malloc`、`std::mutex`、ファイルI/O、`std::exp`等のlibm呼び出し、`DftiCommitDescriptor`、`try`/`catch`、`std::shared_ptr`操作。 |
| **Message Thread** | `UI` Core | JUCEメインループ。UIイベント処理、全非同期タスクの起動、デバイス設定の保存。 | Audio Threadをブロックする長時間ロック。 |
| **Worker Thread** | `Worker` Core | `SnapshotCoordinator` 専用。UIからのパラメータ変更コマンドをデバウンスし、`GlobalSnapshot` を非同期生成。 | Audio Threadの直接操作。 |
| **Rebuild Thread** | `HeavyBackground` Core | DSPグラフ (`DSPCore`) 全体の再構築。IR再利用判定、NUCエンジンの初期化委譲。 | Audio Threadでの実行。 |
| **Loader Thread** | `HeavyBackground` Core | IRファイル読み込み、`r8brain`リサンプリング、位相変換。`PreparedIRState` 生成とキャッシュ保存。 | Audio Threadでの実行。 |
| **Progressive Upgrade Thread** | `HeavyBackground` Core | 低解像度IRキャッシュから高解像度IRへの段階的アップグレード。 | Audio Threadでの実行。 |
| **Learner Main Thread** | `LearnerMain` Core | ノイズシェーパー係数のCMA-ES最適化ループ。評価ワーカーの管理。 | Audio Threadでの実行。 |
| **Learner Eval Thread(s)** | `LearnerEval` Core(s) | 各候補係数に対するコスト関数評価を並列実行。 | Audio Threadでの実行。 |
| **DeferredFree Thread** | `LightBackground` | `SafeStateSwapper` の retired キューを監視し、安全になった `ConvolverState` を `delete`。 | Audio Threadでの `delete`/`mkl_free`。 |
| **RNG Producer Thread** | `Lowest` | `PsychoacousticDither` が使用する乱数リングバッファをバックグラウンドで補充。 | Audio Threadのリアルタイム性阻害。 |

## 3. データアクセス規則の明文化

### 3.1 基本原則

| データ種別 | 同期機構 | 実装パターン | メモリライフサイクル |
| :--- | :--- | :--- | :--- |
| **スカラー設定値** | `std::atomic<T>` | Non-RTスレッドが `store(release)`、Audio Threadが `load(acquire)`。 | 値型のため不要。 |
| **構造化パラメータ** | **RCU (Read-Copy-Update)** | `SafeStateSwapper` + `DeletionQueue`。`swap()` で新状態公開、`tryReclaim()` で遅延解放。 | 生成側（Non-RT）が `new`/`mkl_malloc`。Audio Threadは読み取り専用。解放は専用スレッド。 |
| **大容量データバッファ** | **SPSC FIFO + RCU** | `LockFreeRingBuffer` でポインタ受け渡し。 | 生成側が確保。受信側（Audio）は消費後、**返却用FIFO** でNon-RTへ戻し、Non-RTが解放。 |
| **コマンド/メッセージ** | **SPSC FIFO** | `CommandBuffer`、`LearningCommand` キュー。 | 値型コピー。 |

### 3.2 主要データ構造ごとの詳細ルール

#### A. GlobalSnapshot (全DSPパラメータスナップショット)
- **経路**: `UI (Message)` → `CommandBuffer` → `Worker Thread` → `SnapshotCoordinator`
- **Audio Thread アクセス**: `m_coordinator.getCurrent()` (atomic load)
- **更新方法**: `SnapshotCoordinator::startFade()` が atomic swap を実行。
- **解放**: `DeletionQueue` を介して、全 Reader の退出後に `SnapshotFactory::destroy()`。

#### B. ConvolverState (IR周波数領域データ)
- **経路**: `Loader Thread` → `Message Thread` (callAsync) → `SafeStateSwapper::swap()`
- **Audio Thread アクセス**: `ConvolverProcessor::acquireIRState()` で `enterReader()` → 読み取り → `exitReader()`
- **更新方法**: `rcuSwapper.swap()`
- **解放**: `DeferredFreeThread` が `tryReclaim()` で取得し `delete`。

#### C. EQCoeffCache (EQ係数キャッシュ)
- **経路**: `Worker Thread` (`createSnapshotFromCurrentState`) → `EQCacheManager::getOrCreate()`
- **Audio Thread アクセス**: `EQProcessor::process(..., cache)` でキャッシュ内の係数を読み取り。
- **更新方法**: `CacheMap` を atomic swap (`cacheMapPtr`)。古いマップは `g_deletionQueue` で遅延解放。

#### D. Adaptive Noise Shaper Coefficients (学習済み係数)
- **経路**: `Learner Main Thread` → `MessageManager::callAsync` → `AudioEngine::publishCoeffsToBank()`
- **Audio Thread アクセス**: `DSPCore::processOutput()` 内で `getActiveCoeffSet()` (RCU)
- **更新方法**: `CoeffSetWriteLockGuard` でダブルバッファを atomic swap。

#### E. Audio Capture Queue (学習用信号)
- **経路**: `Audio Thread` → `LockFreeRingBuffer<AudioBlock>` → `Learner Main Thread`
- **方向**: Audio (Producer) → Non-RT (Consumer)
- **特徴**: 値型 (`AudioBlock`) のため解放不要。オーバーフロー時は最新データ優先でドロップ。

### 3.3 スレッド間相互作用の具体例

#### 例1: IRファイルロード
1. **Message Thread**: `ConvolverProcessor::loadIR()` → `LoaderThread` 生成・開始。
2. **Loader Thread**: ファイル読み込み → リサンプリング → `PreparedIRState` 生成 → `MessageManager::callAsync()`。
3. **Message Thread**: `applyPreparedIRState()` → `updateConvolverState()` → `rcuSwapper.swap()`。
4. **Audio Thread**: 次回 `process()` で `acquireIRState()` し、新しいIRで畳み込み開始。
5. **DeferredFree Thread**: 古い `ConvolverState` への参照が無くなったことを確認後、`delete`。

#### 例2: EQパラメータ変更
1. **Message Thread**: `EQEditProcessor::setBandGain()` → `scheduleDebounce()` (50msタイマー)
2. **Message Thread (Timer)**: `enqueueSnapshotCommand()` → `CommandBuffer::push()`
3. **Worker Thread**: `CommandBuffer::pop()` → `createSnapshotFromCurrentState()` → `SnapshotFactory::create()` → `SnapshotCoordinator::startFade()`
4. **Audio Thread**: 次回 `getNextAudioBlock()` で `updateFade()` → 新旧スナップショットをクロスフェードしながら処理。