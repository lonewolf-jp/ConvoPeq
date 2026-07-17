# ConvoPeq 音質評価自動化 改修計画書 v7.4（再構成版）

**バージョン**: 7.4（v7.3.6 ベース・実装設計・未確定事項・Appendix の3部構成に再整理）
**策定日**: 2026-07-16
**ベース**: v7.3.6（六次実装完全性検証反映版）＋ コードベース検証結果反映
**ステータス**: **Phase 0 実装開始可（全未確定事項確定済み）**
**対象**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

# Part 1: Phase 0 改修実装設計（実装担当者向け最小要件）

本パートは Phase 0 の実装を担当するプログラマに必要な設計情報のみをまとめる。
設計意図・検証経緯・代替検討は Appendix に譲る。

---

## 1.1 全体アーキテクチャ

```
CI (GitHub Actions)
  └─ Build (MSVC/icx, Debug/Release)
       └─ Audio Quality Tests (CTest)
            └─ Python Test Orchestrator (run_quality_tests.py)
                 1. テスト信号生成 (generators.py / numpy)
                 2. ConvoPeq.exe 起動 (--cli-* options)
                 3. OutputCaptureSink 経由で出力WAV取得
                    └─ Audio RT → SPSC RingBuffer → BG Thread → WAV
                 4. 分析 vs 理論値/ゴールデン (analyzers.py)
                 5. NaN/Inf/Denormal/DC 異常検査
                 6. Report JUnit XML + HTML (RMAA互換)
```

**既存CLI `MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す。新たな実行可能ファイルは作成しない。**

---

## 1.2 CapturePoint enum

```cpp
// 新設ファイル: OutputCaptureSink.h （または既存 AudioEngine.h 近辺）
enum class CapturePoint : uint8_t {
    None             = 0,   // キャプチャしない
    PreOutputFilter  = 1,   // OutputFilter 適用前（DSP生出力、オーバーサンプリングドメイン）
    PostOutputFilter = 2,   // OutputFilter 適用後（デフォルト、ベースレート）
    PostDither      = 3,    // ディザー後（最終出力）
};
```

### CapturePoint ↔ 実パイプライン対応

| CapturePoint | 挿入位置 | データ型 | 備考 |
|---|---|---|---|
| `PreOutputFilter` | DSPCoreFloat.cpp:242 ／ DSPCoreDouble.cpp:391（`juce::dsp::AudioBlock<double>` 作成直後） | `juce::dsp::AudioBlock<double>`（オーバーサンプリングドメイン） | OS倍率に応じた高サンプリングレート |
| `PostOutputFilter` | `processOutput()` L411（`pushAdaptiveCaptureBlocks` 呼出直後）／ `processOutputDouble()` L651 同位置 | `double* dataL/dataR`（ベースレート） | DC Blocker + NaN/Inf除去後、Dither前 |
| `PostDither` (Float Path) | `processOutput()` L503-505間（`applyFixedLatencyDelay` 後、float変換前） | `double* dataL/dataR`（ベースレート） | Dither/NoiseShaper + 遅延補正後 |
| `PostDither` (Double Path) | `processOutputDouble()` L803-805後（`FloatVectorOperations::copy` 後） | `double* dataL/dataR`（ベースレート、float変換なし） | Double Path は変換なし |

### `--cli-capture-mode` マッピング

| CLI引数 | CapturePoint |
|---------|-------------|
| `none` | `None` |
| `pre-filter` | `PreOutputFilter` |
| `post-filter` | `PostOutputFilter`（デフォルト） |
| `post-dither` | `PostDither` |

---

## 1.3 OutputCaptureSink クラス

### クラス定義

```cpp
// OutputCaptureSink.h — AudioEngine から独立（責務分離）
class OutputCaptureSink : private juce::Thread {
public:
    OutputCaptureSink() = default;
    ~OutputCaptureSink() override {
        jassert(!isThreadRunning());  // Thread 生存中は destroy 不可
    }

    // ── Message Thread API ───────────────────────────
    void setCapturePoint(CapturePoint point) noexcept {
        capturePoint_.store(point, std::memory_order_release);
    }
    void setOutputPath(const juce::File& path) {
        jassert(!isThreadRunning());  // 契約: 開始後変更禁止
        outputPath_ = path;
    }
    void setAudioParams(double sampleRateHz, int bitDepth,
                        int coeffBankIndex, uint64_t sessionId) noexcept {
        sampleRateHz_ = sampleRateHz;
        bitDepth_ = bitDepth;
        coeffBankIndex_ = coeffBankIndex;
        sessionId_ = sessionId;
    }

    // ── Audio Thread API (RT-safe: push + signal のみ) ──
    void capture(const double* left, const double* right, int numSamples,
                 uint64_t timestampUs) noexcept;
    void capture(const float* left, const float* right, int numSamples,
                 uint64_t timestampUs) noexcept;  // PostDither Float Path 用

    // ── Background Thread ────────────────────────────
    void run() override;
    void startCapturing() { jassert(!started_.exchange(true)); startThread(); }
    void stopCapturing()  { jassert(started_.load()); stopRequested_.store(true, release); wakeEvent_.signal(); }

    // move/copy 禁止（Thread 所有権の明確化）
    OutputCaptureSink(const OutputCaptureSink&) = delete;
    OutputCaptureSink& operator=(const OutputCaptureSink&) = delete;
    OutputCaptureSink(OutputCaptureSink&&) = delete;
    OutputCaptureSink& operator=(OutputCaptureSink&&) = delete;

private:
    static constexpr int kBlockSize    = 256;
    static constexpr int kRingCapacity = 4096;  // ≈ 21秒 @ 48kHz
    LockFreeRingBuffer<AudioBlock, kRingCapacity> ringBuffer_;
    std::atomic<CapturePoint> capturePoint_{CapturePoint::PostOutputFilter};
    juce::File outputPath_;      // Thread 開始前に設定固定
    double sampleRateHz_{48000.0};
    int bitDepth_{64};
    int coeffBankIndex_{0};
    uint64_t sessionId_{0};
    std::unique_ptr<float[]> convertBufferL_, convertBufferR_;  // run()で事前確保
    juce::WaitableEvent wakeEvent_;
    std::unique_ptr<juce::AudioFormatWriter> wavWriter_;
    std::atomic<bool> stopRequested_{false};
    std::atomic<uint64_t> droppedBlocks_{0};
    std::atomic<uint64_t> seqlockRetryFailed_{0};
    std::atomic<bool> started_{false};
};
```

### ライフサイクル契約

```
setOutputPath() → setCapturePoint() → setAudioParams()
→ startCapturing()    # 単回起動（2回目禁止）
→ [Audio RT が capture() を呼ぶ]
→ setCaptureSink(nullptr)  # Audio Thread の参照を先に絶つ
→ stopCapturing()     # stopRequested_.store(true, release); signal()
→ waitForThreadToExit(5000)
→ destroy（デストラクタ、Thread 生存中は jassert）
```

- **Single-shot object**: 1セッション専用。`stopCapturing()` 後の再利用は禁止。
- `startCapturing()` 以降の `setOutputPath()` は禁止。
- `startCapturing()` で `stopRequested_` を false に初期化する必要はない。

---

## 1.4 capture() — Audio Thread の責務は push + signal のみ

### double* overload（PostFilter / PreDither 用）

```cpp
void OutputCaptureSink::capture(const double* left, const double* right,
                                int numSamples, uint64_t timestampUs) noexcept
{
    const CapturePoint point = capturePoint_.load(std::memory_order_relaxed);
    if (point == CapturePoint::None) return;
    if (left == nullptr || numSamples <= 0) return;

    static constexpr int kBlockSize = 256;
    const double* effectiveRight = (right != nullptr) ? right : left;

    for (int offset = 0; offset < numSamples; offset += kBlockSize)
    {
        const int currentBlockSize = std::min(kBlockSize, numSamples - offset);
        const double* srcL = left + offset;
        const double* srcR = effectiveRight + offset;

        const PushResult result = ringBuffer_.pushBecameNonEmptyWithWriter(
            [&](AudioBlock& block) noexcept {
                block.numSamples = currentBlockSize;
                block.sampleRateHz = sampleRateHz_;
                block.bitDepth = bitDepth_;
                block.adaptiveCoeffBankIndex = coeffBankIndex_;
                block.sessionId = sessionId_;
                block.timestampUs = timestampUs + offset;

                // AVX2 高速コピー（既存 pushAdaptiveCaptureBlocks と同一パターン）
                const int simdCount = currentBlockSize & ~3;
                int i = 0;
                for (; i < simdCount; i += 4) {
                    __m256d vL = _mm256_loadu_pd(srcL + i);
                    _mm256_storeu_pd(block.L + i, vL);
                }
                for (; i < currentBlockSize; ++i) block.L[i] = srcL[i];

                i = 0;
                for (; i < simdCount; i += 4) {
                    __m256d vR = _mm256_loadu_pd(srcR + i);
                    _mm256_storeu_pd(block.R + i, vR);
                }
                for (; i < currentBlockSize; ++i) block.R[i] = srcR[i];
            });

        if (result == PushResult::BecameNonEmpty)
            wakeEvent_.signal();  // RT-safe: 非ブロッキング
        else if (result == PushResult::Full)
            droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
        // AlreadyNonEmpty: リングが既に非空。drain-all 設計により追加通知は不要
    }
}
```

### float* overload（PostDither Float Path 用）

```cpp
void OutputCaptureSink::capture(const float* left, const float* right,
                                int numSamples, uint64_t timestampUs) noexcept
{
    // ※ double*版 と同一構造。writer ラムダ内で
    //    static_cast<double>(srcF[i]) で AudioBlock.L/R に格納する点のみ異なる。
    //    block.timestampUs = timestampUs + offset; も同様に設定すること。
    //    AudioBlock は double[256] 固定のため。
}
```

**実装上の注意**:
- `pushBecameNonEmptyWithWriter()` は既存 `pushAdaptiveCaptureBlocks()` と同じ `pushWithWriter()` ラムダパターン。AudioBlock のローカル変数 + コピー（4128byte×3）を回避し、直接リングスロットに書込む（4128byte×1）。既存 AVX2 `_mm256_loadu_pd/storeu_pd` パターンを踏襲。
- **メタデータ設定は必須**: `block.numSamples` / `block.timestampUs` 等を設定しないと BG Thread が有効サンプル数やタイムスタンプを正しく認識できない。
- **3状態の区別**: `PushResult::BecameNonEmpty` のみ `signal()`。`Full` は `droppedBlocks_` カウントアップのみ。`AlreadyNonEmpty` は何もしない（drain-all 設計により signal 到達時点で全データ処理される）。
- **既存 `pushAdaptiveCaptureBlocks()` との関係**: 同関数は `timestampUs` を設定しない（0のまま）。既存の adaptive capture 機能には影響しない。

---

## 1.5 Background Thread — drain-all 設計

```cpp
void OutputCaptureSink::run() override
{
    convertBufferL_ = std::make_unique<float[]>(kBlockSize);
    convertBufferR_ = std::make_unique<float[]>(kBlockSize);

    while (!stopRequested_.load(std::memory_order_acquire)) {
        wakeEvent_.wait(-1);              // 通知待ち（無限待機）
        drainAndWriteBatch();             // while(pop()) で完全排出
    }
    drainAllAndFinalize();                // 最後の排出
    wavWriter_.reset();
}
```

**パイプライン**:
```
Audio RT: capture() → push + signal
                            ↓ SPSC RingBuffer (wait-free)
BG Thread: run() → wait() → drainAndWriteBatch() → WAV書込
```

**安全性**:
- drain-all（`while(pop())` で完全排出）により Producer の1回の signal で全データ処理。
- `WaitableEvent` の信号保持に依存しない。
- シャットダウン時の `drainAllAndFinalize()` と `drainAndWriteBatch()` の二重排出も `while(pop())` なので安全。
- **WAV出力フォーマット**: IEEE 754 float32（24bit有効仮数）。48kHz/2ch/float32 = 384kB/s。

---

## 1.6 LockFreeRingBuffer 拡張

```cpp
// LockFreeRingBuffer.h に追加。既存の push/pop/size/clear/pushWithWriter は変更しない。
enum class PushResult : uint8_t { Full, BecameNonEmpty, AlreadyNonEmpty };

// ── コピー版: const T& をリングに push（push の3状態戻り値版）──
PushResult pushBecameNonEmpty(const T& item) noexcept {
    size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
    size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
    if ((w - r) >= Capacity) return PushResult::Full;
    const bool wasEmpty = (w == r);
    buffer[w & MASK] = item;
    convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
    return wasEmpty ? PushResult::BecameNonEmpty : PushResult::AlreadyNonEmpty;
}

// ── ゼロコピー版: writer ラムダで直接リングスロットに書込（★ OutputCaptureSink はこちらを使用）──
template<typename Writer>
PushResult pushBecameNonEmptyWithWriter(Writer&& writer) noexcept {
    size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
    size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
    if ((w - r) >= Capacity) return PushResult::Full;
    const bool wasEmpty = (w == r);
    std::forward<Writer>(writer)(buffer[w & MASK]);  // 直接スロットに書込
    convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
    return wasEmpty ? PushResult::BecameNonEmpty : PushResult::AlreadyNonEmpty;
}
```

**採用方針**: `OutputCaptureSink::capture()` は AudioBlock（4120byte）のコピーを回避するため、**`pushBecameNonEmptyWithWriter()` を使用する**（既存 `pushAdaptiveCaptureBlocks` の `pushWithWriter` と同一パターン）。

---

## 1.7 シャットダウンシーケンス

既存 `~MainWindow()` の10ステップに OutputCaptureSink の停止を統合する。

### 既存 ~MainWindow() 10ステップ

```
step 1:  removeChangeListener(this)
step 2:  setAdaptiveAutosaveCallback({})        # OutputCaptureSink と無関係
step 3:  setCliProcessingTelemetryEnabled(false)
step 4:  setProcessor(nullptr)
step 5:  stopTimer()
step 6:  saveSettings(...)
step 7:  removeAudioCallback(&audioProcessorPlayer)
step 8:  closeAudioDevice()
step 9:  audioEngineProcessor.reset()
step 10: UI reset (settingsWindow/specAnalyzer/eqPanel/convolverPanel)
```

### 統合14ステップ（OutputCaptureSink 組み込み後）

ステップ 3→4 の間に挿入:

```
  1. cliAutomationCallbacksEnabled = false
  2. audioEngine.removeChangeListener(this)
  3. audioEngine.setCliProcessingTelemetryEnabled(false)

  4. ★ setCaptureSink(nullptr) ★          # Audio Thread の参照を先に絶つ
     → convo::publishAtomic(audioEngine.outputCaptureSink_, nullptr, release)
     Audio Thread は relaxed load で読み、nullptr 検出後は capture() を呼ばない。
  5. captureSink->stopCapturing()           # stopRequested_.store(true, release)
     captureSink->wakeEvent_.signal()        # BG Thread 起床
  6. captureSink->waitForThreadToExit(5000) # BG Thread 終了待機
  7. captureSink.reset()                    # デストラクト（jassert で Thread 生存確認）

  8. audioProcessorPlayer.setProcessor(nullptr)
  9. stopTimer()
 10. saveSettings(...)
 11. audioDeviceManager.removeAudioCallback(...)
 12. audioDeviceManager.closeAudioDevice()
 13. audioEngineProcessor.reset()
 14. UI reset
```

**`outputCaptureSink_` は AudioEngine に `std::atomic<OutputCaptureSink*> outputCaptureSink_{nullptr}` として新設。** `setCaptureSink(ptr)` は `convo::publishAtomic(..., release)` で書込。

---

## 1.8 CLIオプション

### 既存（27種）

**hasFlag（3）**: `--cli-run`, `--cli-start-learning`, `--cli-resume-learning`

**findValue（24）**: `--cli-ir`, `--cli-device-type`, `--cli-buffer-samples`, `--cli-sample-rate-hz`, `--cli-phase`, `--cli-order`, `--cli-dither-bit-depth`, `--cli-noise-shaper`, `--cli-post-load-dither-bit-depth`, `--cli-post-load-delay-ms`, `--cli-ir-reload-count`, `--cli-ir-reload-interval-ms`, `--cli-bypass-burst-count`, `--cli-bypass-burst-interval-ms`, `--cli-bypass-burst-value`, `--cli-intent-burst-count`, `--cli-intent-burst-interval-ms`, `--cli-target-ir-sec`, `--cli-debounce-ms`, `--cli-f1-hz`, `--cli-f2-hz`, `--cli-learning-action`, `--cli-learning-mode`, `--cli-exit-ms`

### 新規（5種）— Phase 0 で実装

| オプション | 難易度 | 実装内容 |
|-----------|--------|---------|
| `--cli-output-wav` | 低（2h） | WAVファイル出力（OutputCaptureSink経由）|
| `--cli-capture-mode` | 低（1h） | `none`/`pre-filter`/`post-filter`/`post-dither` 選択 |
| `--cli-dump-filter-coeffs` | 低（2h） | OutputFilter const getter追加、JSON出力 |
| `--cli-ir-reload-list` | 中（4h） | カンマ区切りパース→複数IR逐次ロード |
| `--cli-progressive-upgrade` | 低（1h） | `setConvolverEnableProgressiveUpgrade(true)` |

### MainWindow.cpp 実装パターン（`runCommandLineAutomation()` 内）

```cpp
// ★ --cli-output-wav: WAV出力パス指定
const auto outputWavPath = findValue("--cli-output-wav");
if (!outputWavPath.isEmpty()) {
    auto captureSink = std::make_unique<OutputCaptureSink>();
    captureSink->setOutputPath(juce::File(outputWavPath));

    const auto captureModeStr = findValue("--cli-capture-mode");
    CapturePoint cp = CapturePoint::PostOutputFilter; // default
    if (captureModeStr.equalsIgnoreCase("none"))        cp = CapturePoint::None;
    else if (captureModeStr.equalsIgnoreCase("pre-filter"))  cp = CapturePoint::PreOutputFilter;
    else if (captureModeStr.equalsIgnoreCase("post-filter"))  cp = CapturePoint::PostOutputFilter;
    else if (captureModeStr.equalsIgnoreCase("post-dither")) cp = CapturePoint::PostDither;
    captureSink->setCapturePoint(cp);
    captureSink->startCapturing();
    audioEngine.setCaptureSink(captureSink.get());
    cliCaptureSink_ = std::move(captureSink);  // MainWindow.h に unique_ptr メンバ追加
}
```

**注意**: `--cli-output-wav` と `--cli-capture-mode` は単独では automation flag を立てない（`--cli-run` または他の automation flag と併用必須）。

### CLI パーサー既存マッピング（実装時参照）

| CLIオプション | パーサー関数 | 実装ファイル |
|-------------|------------|-------------|
| `--cli-order` | `parseCliOrderMode()` | MainWindow.cpp:60-89 |
| `--cli-phase` | `parseCliPhaseMode()` | MainWindow.cpp:36-57 |
| `--cli-noise-shaper` | `parseCliNoiseShaper()` | MainWindow.cpp:91-120 |
| `--cli-dither-bit-depth` | `parseCliDitherDepth()` | MainWindow.cpp:117+ |
| `--cli-learning-mode` | `parseCliLearningMode()` | MainWindow.cpp:122+ |

---

## 1.9 ProcessingState / testCaptureQueue 統合

### AudioEngine.h 変更点

```cpp
// AudioEngine.h に追記
std::atomic<OutputCaptureSink*> outputCaptureSink_{nullptr};

// セッター（Message Thread）: AudioEngine.h
void setCaptureSink(OutputCaptureSink* sink) noexcept {
    convo::publishAtomic(outputCaptureSink_, sink, std::memory_order_release);
}
```

### buildAudioThreadProcessingState() 変更

```cpp
// AudioEngine.h:buildAudioThreadProcessingState() 内
OutputCaptureSink* sink = convo::consumeAtomic(outputCaptureSink_, std::memory_order_acquire);
snapshot.testCaptureQueue = (sink != nullptr) ? &sink->ringBuffer_ : nullptr;
```

### ProcessingState 構造体変更

```cpp
struct ProcessingState {
    // ... 既存メンバ ...
    LockFreeRingBuffer<AudioBlock, 4096>* adaptiveCaptureQueue;  // 既存
    LockFreeRingBuffer<AudioBlock, 4096>* testCaptureQueue;      // ★ 新設（CLIテスト用）
    // ... 既存メンバ ...
};
```

**既存パターンとの整合**: `adaptiveCaptureActiveRt` → `adaptiveCaptureQueue` と同一パターン。`outputCaptureSink_` atomic ポインタを acquire load して `testCaptureQueue` に設定する。

---

## 1.10 AudioBlock 保証

```cpp
static_assert(std::is_trivially_copyable_v<AudioBlock>,
    "AudioBlock must be trivially copyable for LockFreeRingBuffer");
static_assert(std::is_standard_layout_v<AudioBlock>,
    "AudioBlock must be standard layout");
static_assert(std::is_trivially_destructible_v<AudioBlock>,
    "AudioBlock must be trivially destructible");
```

### AudioBlock メモリレイアウト（設計値）

> **実装上、既存 `AudioEngine.h:23-32` の AudioBlock 構造体に `uint64_t timestampUs` フィールドを追加すること。**

| メンバ | オフセット | サイズ |
|--------|-----------|--------|
| `L[256]` | 0 | 2048 byte |
| `R[256]` | 2048 | 2048 byte |
| `numSamples` | 4096 | 4 byte |
| `sampleRateHz` | 4100 | 4 byte |
| `bitDepth` | 4104 | 4 byte |
| `adaptiveCoeffBankIndex` | 4108 | 4 byte |
| `sessionId` | 4112 | 8 byte |
| `timestampUs` | **4120** | **8 byte** |
| **合計** | | **4128 byte**（alignof=8） |

**注意**: `sizeof(AudioBlock)` が 4120 → 4128 に増加する。RingBuffer の容量計算（`kRingCapacity=4096`, 21秒@48kHz）に影響しない。static_assert（trivially_copyable/standard_layout/destructible）は引き続き満たす。

---

## 1.11 CMake 追加

```cmake
# CMakeLists.txt に追加（既存パターンに従う）
add_executable(OutputCaptureSinkTests
    src/tests/OutputCaptureSinkTests.cpp
)
target_compile_features(OutputCaptureSinkTests PRIVATE cxx_std_20)
add_test(NAME OutputCaptureSinkShutdownOrder COMMAND OutputCaptureSinkTests)
```

**既存テストパターン**: Catch2/doctest/gtest 不使用。独自の `check()` マクロ + `g_testsPassed`/`g_testsFailed` + `std::cerr` 出力方式。

---

## 1.12 実装前チェックリスト（Phase 0 該当項目）

| # | 確認項目 | ☐ |
|---|---------|---|
| 1 | Seqlock: generation 偶数→奇数→偶数（`old+1` → write → `old+2`） | ☐ |
| 2 | Seqlock: release/acquire のみで同期（thread_fence 不要） | ☐ |
| 3 | `outputPath_`: `juce::File` 単体、Thread 開始前固定契約 | ☐ |
| 4 | シャットダウン: `setCaptureSink(nullptr)` → `stop()` → wait → destroy | ☐ |
| 5 | BG Thread: WaitableEvent `wait(-1)` + `signal()`、drain-all 設計 | ☐ |
| 6 | `sizeof(AudioBlock)` の static_assert なし（trivially_copyable のみ） | ☐ |
| 7 | Seqlock Reader: 8回リトライ、失敗時は slot 破棄 | ☐ |
| 8 | float 変換バッファ: `run()` 内で事前確保（毎ブロック生成禁止） | ☐ |
| 9 | `CapturePoint`: 単一 enum（二重管理なし） | ☐ |
| 10 | RecoveryHistory: `copyRecoveryHistorySnapshot()` が snapshot 返却 | ☐ |
| 11 | `capture()`: push + signal のみ（pop/WAV は BG Thread の責務） | ☐ |
| 12 | `pushBecameNonEmpty()` / `pushBecameNonEmptyWithWriter()`: LockFreeRingBuffer に追加。capture() は writer 版を使用 | ☐ |
| 13 | `stopRequested_`: `store(release)` / `load(acquire)`, signal() で起床 | ☐ |
| 21 | ★ WaitableEvent デフォルト構築（auto-reset）であることを実装時確認 | ☐ |
| 22 | ★ Seqlock `recordRecoveryAction()`: Timer Thread 単一呼出の設計契約遵守 | ☐ |
| 23 | ★ AudioBlock: `is_trivial_v` が false でも問題なし（trivially_copyable のみ要求） | ☐ |
| 26 | ★ OutputCaptureSink 単体テスト: `src/tests/OutputCaptureSinkTests.cpp` 新設 | ☐ |
| 27 | ★ `--cli-dump-filter-coeffs`: OutputFilter に const getter 新設 | ☐ |
| 28 | ★ JUCE 8 の `WaitableEvent` 仕様をビルド時に再確認 | ☐ |
| 29 | ★ shutdown: `outputCaptureSink_` は AudioEngine に atomic ポインタとして新設 | ☐ |
| 30 | ★ shutdown: 既存 `~MainWindow()` の10step に統合版14step を正しく組み込む | ☐ |
| 35 | ★ Seqlock `fetchAdd` → `convo::fetchAddAtomic` に統一 | ☐ |
| 36 | ★ `capture()` シグネチャ: `const double*`（`juce::AudioBuffer<double>` は不可） | ☐ |
| 37 | ★ `pushBecameNonEmptyWithWriter()`: ゼロコピー・writer ラムダ版 | ☐ |
| 38 | ★ `capture()` 内 writer ラムダ: `block.numSamples = currentBlockSize` 設定必須 | ☐ |
| 51 | ★ `capture()` に `const float*` overload を追加（PostDither Float Path） | ☐ |
| 52 | ★ `buildAudioThreadProcessingState()` 内で `consumeAtomic(outputCaptureSink_, acquire)` | ☐ |
| 56 | ★ `--cli-output-wav` / `--cli-capture-mode`: findValue パターン、`cliCaptureSink_` メンバ | ☐ |
| 68 | ★ OutputCaptureSink: `sampleRateHz_`/`bitDepth_`/`coeffBankIndex_`/`sessionId_` 4変数宣言＋`setAudioParams()` | ☐ |
| 69 | ★ `pushBecameNonEmptyWithWriter()`: LockFreeRingBuffer.h に実装 | ☐ |

---

## 1.13 Phase 1: 音質評価自動化フレームワーク設計

本セクションは Phase 1 で実装するテスト自動化フレームワークの詳細設計を記載する。
Phase 0 完了後、本セクションに従い実装を開始すること。

---

### 1.13.1 アーキテクチャ概要

```
┌─ Python Test Orchestrator ─────────────────────────────────────┐
│  run_quality_tests.py                                           │
│    ├─ test_config.yaml を読込 → パターン×TC の直積を生成       │
│    ├─ generators.py で入力信号WAVを生成                         │
│    ├─ cli_runner.py で ConvoPeq.exe を起動                     │
│    │    └─ OutputCaptureSink 経由で出力WAVを取得                │
│    ├─ analyzers.py で出力WAVを分析 → メトリクス算出            │
│    ├─ golden_calculator.py で期待値と比較                       │
│    └─ Report JUnit XML + HTML (RMAA互換)                       │
└────────────────────────────────────────────────────────────────┘
```

### 1.13.2 テストパラメータ次元（全10次元）— CLI値マッピング

各パラメータ次元のCLI引数値と、MainWindow.cpp パーサー・AudioEngine API の対応。

| 次元 | CLIオプション | CLI値 | パーサー関数 | AudioEngine API | 既定値 |
|------|-------------|-------|------------|----------------|--------|
| 処理順序 | `--cli-order` | `conv`/`peq`/`convpeq`/`peqconv` | `parseCliOrderMode()` (MW:60) | `setConvolverBypass()` / `setEqBypass()` / `setProcessingOrder()` | `convpeq` |
| 位相モード | `--cli-phase` | `asis`/`mixed`/`minimum` | `parseCliPhaseMode()` (MW:36) | `ConvolverProcessor::setPhaseMode()` | `asis` |
| OSタイプ | `--cli-os-type` | `iir`/`linear-phase` | 新規パーサー | `setOversamplingType()` (AE.h:1410) | `iir` |
| OS倍率 | `--cli-os-factor` | `1`/`2`/`4`/`8` | 新規パーサー | `setOversamplingFactor()` (AE.h:1407) | `1` |
| ノイズシェイパー | `--cli-noise-shaper` | `psycho`/`fixed4`/`adaptive`/`fixed15` | `parseCliNoiseShaper()` (MW:91) | `setNoiseShaperType()` | `psycho` |
| ディザー深度 | `--cli-dither-bit-depth` | `16`/`24`/`32` | `parseCliDitherDepth()` (MW:117) | ditherBitDepth メンバ直接設定 | `32` |
| HCモード | `--cli-hc-mode` | `sharp`/`natural`/`soft` | 新規パーサー | `setConvHCFilterMode()` (AE.h:1419) | `natural` |
| LCモード | `--cli-lc-mode` | `natural`/`soft` | 新規パーサー | `setConvLCFilterMode()` (AE.h:1422) | `natural` |
| ソフトクリップ | `--cli-soft-clip` | `on`/`off` | 新規パーサー | `setSoftClipEnabled()` (AE.h:1374) | `off` |
| サチュレーション | `--cli-saturation` | `0.0`〜`1.0` | 新規パーサー | `setSaturationAmount()` (AE.h:1377) | `0.0` |

**新規パーサー実装パターン（`MainWindow.cpp` に追加）**:

```cpp
// ★ Phase 1: --cli-os-type パーサー（既存 parseCliOrderMode に倣う）
bool parseCliOsType(const juce::String& value, AudioEngine::OversamplingType& outType)
{
    const auto normalized = normalizeCliValue(value);
    if (normalized == "iir")          { outType = AudioEngine::OversamplingType::IIR; return true; }
    if (normalized == "linear-phase" || normalized == "linear" || normalized == "lp")
                                      { outType = AudioEngine::OversamplingType::LinearPhase; return true; }
    return false;
}
```

### 1.13.3 測定パターン定義（全10パターン）

`test_config.yaml` に定義するパターンの完全仕様:

```yaml
# test_config.yaml — YAMLスキーマ定義
patterns:
  P1-Baseline:
    description: "基準測定（全既定値）"
    cli: &baseline_cli
      order: convpeq; phase: asis; osType: iir; os: 1
      noiseShaper: psycho; ditherBitDepth: 32
      hcMode: natural; lcMode: natural; softClip: off; saturation: 0.0
    srcIrs: [dirac]
    testCases: [TC-01, TC-03, TC-04, TC-04A, TC-06, TC-23, TC-24,
                TC-29A, TC-29B, TC-31a, TC-31b, TC-32, TC-39, TC-41]

  P2-PEQ-Only:
    description: "PEQ単体品質測定"
    cli:
      order: peq; convBypass: true; phase: asis; osType: iir; os: 1
      noiseShaper: psycho; ditherBitDepth: 32
      hcMode: natural; lcMode: natural; softClip: off; saturation: 0.0
    srcIrs: [dirac, room_correction]
    testCases: [TC-01, TC-01B, TC-03, TC-04, TC-07, TC-32, TC-36, TC-37]

  P3-ConvoThenPEQ:
    description: "既定順序（Convolver→PEQ）品質測定"
    cli:
      order: convpeq; phase: mixed; osType: iir
      os: [1, 2, 4, 8]  # ← 値がリストの場合、ループ実行
      noiseShaper: psycho; ditherBitDepth: 32
      hcMode: natural; lcMode: natural; softClip: off; saturation: 0.0
    srcIrs: [dirac, lpf_1k]
    testCases: [TC-01, TC-02, TC-09, TC-21, TC-23, TC-33, TC-35, TC-39]

  # P4〜P10 は Appendix A 参照（同構造）
```

### 1.13.4 CLI引数ビルダー（cli_runner.py）

```python
# cli_runner.py — Phase 1 実装

import subprocess
from pathlib import Path
from typing import Any

CONVOPEQ_PATH = Path("build/ConvoPeq_Standalone.exe")

# CLI値→パーサー引数 マッピング（MainWindow.cpp parseCli 関数と一致）
ORDER_MAP = {
    "conv":    "conv",     "convolver":   "conv",
    "peq":     "peq",      "eq":          "peq",
    "convpeq": "convpeq",  "conv->peq":   "convpeq",
    "peqconv": "peqconv",  "peq->conv":   "peqconv",
}
PHASE_MAP = {"asis": "asis", "mixed": "mixed", "minimum": "minimum", "min": "minimum"}
NS_MAP = {"psycho": "psycho", "psychoacoustic": "psycho",
          "fixed4": "fixed4", "adaptive9": "adaptive", "fixed15": "fixed15"}
OS_TYPE_MAP = {"iir": "iir", "linear-phase": "linear-phase", "linear": "linear-phase"}
HC_MAP = {"sharp": "sharp", "natural": "natural", "soft": "soft"}
LC_MAP = {"natural": "natural", "soft": "soft"}

def build_cli_args(pattern: dict, ir_file: Path, input_wav: Path, output_dir: Path) -> list[str]:
    """パターン定義 → ConvoPeq CLI 引数リスト"""
    cli = pattern["cli"]
    args = [
        "--cli-run",
        f"--cli-ir={ir_file}",
        f"--cli-output-wav={output_dir / 'output.wav'}",
        "--cli-capture-mode=post-dither",
        "--cli-exit-ms=5000",
    ]
    # 処理順序
    if "order" in cli:
        args.append(f"--cli-order={ORDER_MAP[cli['order']]}")
    # 位相モード
    if "phase" in cli:
        args.append(f"--cli-phase={PHASE_MAP[cli['phase']]}")
    # OS パラメータ（Phase 1 CLI）
    if "osType" in cli:
        args.append(f"--cli-os-type={OS_TYPE_MAP[cli['osType']]}")
    if "os" in cli:
        os_val = cli["os"] if isinstance(cli["os"], (int, str)) else cli["os"][0]
        args.append(f"--cli-os-factor={os_val}")
    # ノイズシェイパー
    if "noiseShaper" in cli:
        args.append(f"--cli-noise-shaper={NS_MAP[cli['noiseShaper']]}")
    # ディザー深度
    if cli.get("ditherBitDepth", 32) < 32:
        args.append(f"--cli-dither-bit-depth={cli['ditherBitDepth']}")
    # HC/LC モード（Phase 1 CLI）
    if "hcMode" in cli:
        args.append(f"--cli-hc-mode={HC_MAP[cli['hcMode']]}")
    if "lcMode" in cli:
        args.append(f"--cli-lc-mode={LC_MAP[cli['lcMode']]}")
    # ソフトクリップ / サチュレーション（Phase 1 CLI）
    if cli.get("softClip") == "on" or cli.get("softClip") is True:
        args.append("--cli-soft-clip=on")
    if cli.get("saturation", 0.0) > 0.0:
        args.append(f"--cli-saturation={cli['saturation']}")
    # Bypass burst（PEQ-Only モード時）
    if cli.get("convBypass"):
        args += ["--cli-bypass-burst-count=1", "--cli-bypass-burst-value=1"]
    return args

def run_convopeq(args: list[str]) -> subprocess.CompletedProcess:
    """ConvoPeq を CLI モードで実行"""
    cmd = [str(CONVOPEQ_PATH)] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)

def os_sweep(cli_base: dict, os_values: list[int]) -> list[dict]:
    """OS倍率スイープ用 CLI 展開"""
    return [{**cli_base, "os": os_val} for os_val in os_values]
```

### 1.13.5 テスト信号生成（generators.py）

```python
# generators.py — Phase 1-1 実装

import numpy as np
from scipy.io import wavfile
from pathlib import Path

SR = 48000  # 全テスト共通サンプリングレート

def generate_sine(freq: float, duration: float, amplitude: float = 0.5,
                  sr: int = SR) -> Path:
    """純音正弦波 float32 stereo WAV"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    sig = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"sine_{int(freq)}hz_{duration}s.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, data)
    return path

def generate_dirac_input(amplitude: float = 0.5, duration: float = 3.0,
                         sr: int = SR) -> Path:
    """Dirac インパルス（sample[0] = amplitude, rest = 0）"""
    n = int(sr * duration)
    data = np.zeros((n, 2), dtype=np.float32)
    data[0, :] = amplitude
    path = Path("testdata/generated") / f"dirac_{amplitude:.1f}_{duration}s.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, data)
    return path

def generate_log_sweep(f0: float = 20.0, f1: float = 24000.0,
                       duration: float = 10.0, amplitude: float = 0.5,
                       sr: int = SR) -> Path:
    """対数スイープ信号 + 逆フィルタ（Farina 2007 準拠）"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    phase = 2 * np.pi * f0 * duration / np.log(f1 / f0) * \
            (np.exp(t * np.log(f1 / f0) / duration) - 1)
    sig = (amplitude * np.sin(phase)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"logsweep_{int(f0)}_{int(f1)}_{duration}s.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, data)
    # 逆フィルタ信号
    amp_comp = np.exp(-t * np.log(f1 / f0) / duration)
    s_inv = (sig[::-1] * amp_comp).astype(np.float32)
    inv_path = Path("testdata/generated") / f"logsweep_{int(f0)}_{int(f1)}_{duration}s_inverse.wav"
    wavfile.write(str(inv_path), sr, np.column_stack([s_inv, s_inv]))
    return path

def generate_silence(duration: float, sr: int = SR) -> Path:
    """無音信号（全ゼロ）"""
    data = np.zeros((int(sr * duration), 2), dtype=np.float32)
    path = Path("testdata/generated") / f"silence_{duration}s.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, data)
    return path

def generate_multitone(freqs: list[float], duration: float = 3.0,
                       amplitude: float = 0.5, sr: int = SR) -> Path:
    """多音信号（AES17 準拠 random phase）"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    rng = np.random.RandomState(42)
    for f in freqs:
        sig += amplitude * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig = (sig / len(freqs)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"multitone_{len(freqs)}t_{duration}s.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(path), sr, data)
    return path

def generate_synthetic_ir(ir_name: str) -> Path:
    """IRファイル生成（float32 WAV）。未生成の場合のみ生成"""
    outdir = Path("testdata/generated")
    outdir.mkdir(parents=True, exist_ok=True)
    if ir_name == "dirac":
        data = np.zeros(8192, dtype=np.float32); data[0] = 1.0
        path = outdir / "dirac_test.wav"
        wavfile.write(str(path), SR, data)
    elif ir_name == "lpf_1k":
        taps = _windowed_sinc(1000.0, 129, SR)
        data = np.column_stack([taps, taps]).astype(np.float32)
        path = outdir / "lpf_1k_test.wav"
        wavfile.write(str(path), SR, data)
    elif ir_name == "hpf_20":
        lpf = _windowed_sinc(20.0, 129, SR)
        hpf = -lpf; hpf[64] += 1.0  # spectral inversion
        data = np.column_stack([hpf, hpf]).astype(np.float32)
        path = outdir / "hpf_20_test.wav"
        wavfile.write(str(path), SR, data)
    else:
        raise ValueError(f"Unknown IR: {ir_name}")
    return path

def _windowed_sinc(cutoff_hz: float, num_taps: int, sr: int) -> np.ndarray:
    """Windowed-sinc FIR 設計（Hamming窓）"""
    nyq = sr / 2.0
    fc = cutoff_hz / nyq
    half = (num_taps - 1) // 2
    taps = np.zeros(num_taps)
    for i in range(num_taps):
        k = i - half
        taps[i] = (2.0 * fc) if k == 0 else np.sin(2.0 * np.pi * fc * k) / (np.pi * k)
        taps[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (num_taps - 1))
    return taps / taps.sum()
```

### 1.13.6 分析エンジン（analyzers.py）

```python
# analyzers.py — Phase 1-2 実装

import numpy as np
from scipy import signal as scipy_signal
from typing import Callable

class Analyzers:
    """全テストケースの分析ロジック。各 analyze_* が (metrics: dict, pass: bool) を返す"""

    @staticmethod
    def analyze_dirac(output_wav: np.ndarray, sr: int) -> dict:
        """TC-01/02: Dirac応答 → 振幅スペクトルRMS誤差"""
        _, _, expected = theoretical_dirac_response(len(output_wav), sr)
        return compare_frequency_response(output_wav, expected, sr, band=(20, 20000))

    @staticmethod
    def analyze_sweep_transfer(input_wav: np.ndarray, output_wav: np.ndarray,
                                sr: int) -> dict:
        """TC-32/35: Log Sweep → Farina 2007 inverse filter 方式"""
        T = len(input_wav) / sr
        t_full = np.arange(len(input_wav)) / sr
        amp_comp = np.exp(-t_full * np.log(24000.0 / 20.0) / T)
        s_inv = input_wav[::-1] * amp_comp
        h_raw = scipy_signal.fftconvolve(output_wav, s_inv, mode='full')
        ir_start = np.argmax(np.abs(h_raw))
        h = h_raw[ir_start:ir_start + len(input_wav)]
        N = len(h)
        H = np.fft.rfft(h, n=N * 4)
        f = np.fft.rfftfreq(N * 4, d=1.0 / sr)
        band = (f >= 20) & (f <= 20000)
        mag_db = 20 * np.log10(np.abs(H[band]) + 1e-30)
        phase = np.unwrap(np.angle(H[band]))
        gd = -np.diff(phase) / (2 * np.pi * np.diff(f[band])) if len(f[band]) > 1 else [0]
        return {"freq_response_rms_db": float(np.std(mag_db)),
                "group_delay_rms_sample": float(np.std(gd))}

    @staticmethod
    def analyze_null_test(input_wav: np.ndarray, output_wav: np.ndarray) -> dict:
        """TC-31: Null Test → 差分RMS/Peak/サンプル一致率"""
        diff = input_wav - output_wav
        return {"diff_rms_db": 20 * np.log10(np.sqrt(np.mean(diff**2)) + 1e-30),
                "diff_peak_db": 20 * np.log10(np.max(np.abs(diff)) + 1e-30),
                "sample_match_pct": 100 * np.mean(np.abs(diff) < 1e-15)}

    @staticmethod
    def analyze_alias(output_wav: np.ndarray, sr: int, os_factor: int,
                       input_freqs: list[float] = None) -> dict:
        """TC-33: Alias Rejection → 理論イメージ位置のエネルギー"""
        if input_freqs is None: input_freqs = [19000, 20000, 22000]
        window = np.hanning(len(output_wav))
        X = np.fft.rfft(output_wav * window)
        Pxx = np.abs(X) ** 2
        f = np.fft.rfftfreq(len(output_wav), d=1.0 / SR)
        alias_energy = 0.0
        for fin in input_freqs:
            for k in range(1, os_factor):
                for f_alias in [abs(k * sr - fin), k * sr + fin]:
                    if 0 < f_alias < sr * os_factor / 2:
                        alias_energy += Pxx[np.argmin(np.abs(f - f_alias))]
        return {"alias_energy_db": 10 * np.log10(alias_energy + 1e-30)}

    @staticmethod
    def analyze_thd_sweep(output_wav: np.ndarray, sr: int,
                           freqs: list[float] = None) -> dict:
        """TC-39: 周波数別THD（Blackman-Harris FFT, bin-sum）"""
        if freqs is None: freqs = [100, 1000, 5000, 10000, 18000, 20000]
        return {"thd_db_by_freq": {
            str(f): Analyzers._thd_at_freq(output_wav, sr, f) for f in freqs}}

    @staticmethod
    def _thd_at_freq(wav: np.ndarray, sr: int, f0: float) -> float:
        N = 262144
        if len(wav) < N: wav = np.pad(wav, (0, N - len(wav)))
        window = np.blackman(N)
        X = np.fft.rfft(wav[:N] * window)
        f = np.fft.rfftfreq(N, d=1.0 / sr)
        def bin_power(cb: int, hw: int = 2) -> float:
            lo, hi = max(0, cb - hw), min(len(X), cb + hw + 1)
            return float(np.sum(np.abs(X[lo:hi])**2))
        fb = np.argmin(np.abs(f - f0))
        fp = bin_power(fb)
        hp = sum(bin_power(min(fb * h, len(X) - 1)) for h in range(2, 11))
        return float(10 * np.log10(hp / (fp + 1e-30) + 1e-30))

    @staticmethod
    def analyze_impulse_alignment(output_wav: np.ndarray,
                                   expected_delay: int) -> dict:
        """TC-41: インパルスアライメント"""
        peak = np.argmax(np.abs(output_wav))
        energy = np.cumsum(output_wav**2)
        e95 = np.searchsorted(energy, 0.95 * energy[-1]) if energy[-1] > 0 else 0
        return {"peak_offset": int(peak - expected_delay), "energy_95": int(e95)}

    @staticmethod
    def analyze_noise_psd(output_wav: np.ndarray, sr: int) -> dict:
        """TC-34: PSD → A-weighting レベル"""
        f, Pxx = scipy_signal.welch(output_wav, fs=sr, nperseg=8192)
        a_w = A_weighting(f)
        band = (f >= 20) & (f <= 20000)
        weighted = Pxx[band] * (a_w[band] ** 2)
        return {"a_weighted_db": float(10 * np.log10(np.sum(weighted) + 1e-30))}

    @staticmethod
    def analyze_stereo_crosstalk(l: np.ndarray, r: np.ndarray) -> dict:
        """TC-36: クロストーク"""
        return {"crosstalk_lr_db": float(10 * np.log10(np.mean(r**2) / np.mean(l**2) + 1e-30)),
                "crosstalk_rl_db": float(10 * np.log10(np.mean(l**2) / np.mean(r**2) + 1e-30))}

def A_weighting(f: np.ndarray) -> np.ndarray:
    """IEC 61672 A-weighting"""
    f2 = f * f
    num = 12194**2 * f2**2
    denom = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    return num / (denom + 1e-30)

def theoretical_dirac_response(n: int, sr: float) -> tuple:
    y = np.zeros(n); y[0] = 1.0
    return y, sr, y

def compare_frequency_response(actual, expected, sr, band):
    N = max(len(actual), len(expected))
    Ha = np.fft.rfft(actual, n=N); He = np.fft.rfft(expected, n=N)
    f = np.fft.rfftfreq(N, d=1.0 / sr)
    m = (f >= band[0]) & (f <= band[1])
    mad = 20 * np.log10(np.abs(Ha[m]) + 1e-30)
    med = 20 * np.log10(np.abs(He[m]) + 1e-30)
    return {"magnitude_rms_db": float(np.sqrt(np.mean((mad - med)**2))),
            "phase_rms_deg": float(np.degrees(np.sqrt(np.mean(np.angle(Ha[m] / He[m])**2))))}
```

### 1.13.7 ゴールデン比較フレームワーク（golden_calculator.py）

```python
# golden_calculator.py — Phase 1-4 実装

from pathlib import Path
import numpy as np
import json

class GoldenReference:
    """テストケースごとの期待値生成・比較"""

    @staticmethod
    def compare_golden_wav(output_wav: np.ndarray, golden_wav: np.ndarray,
                            sr: int, tc_name: str = "") -> dict:
        n = max(len(output_wav), len(golden_wav))
        out = np.pad(output_wav, (0, n - len(output_wav)))
        ref = np.pad(golden_wav, (0, n - len(golden_wav)))
        H_out = np.fft.rfft(out, n=n*4); H_ref = np.fft.rfft(ref, n=n*4)
        f = np.fft.rfftfreq(n*4, d=1.0/sr)
        band = (f >= 20) & (f <= 20000)
        m_out = 20 * np.log10(np.abs(H_out[band]) + 1e-30)
        m_ref = 20 * np.log10(np.abs(H_ref[band]) + 1e-30)
        return {"tc": tc_name,
                "freq_response_rms_error_db": float(np.sqrt(np.mean((m_out - m_ref)**2))),
                "peak_offset_samples": int(np.argmax(np.abs(out)) - np.argmax(np.abs(ref))),
                "rms_deviation_db": float(20 * np.log10(
                    np.sqrt(np.mean(out**2)) / (np.sqrt(np.mean(ref**2)) + 1e-30))),
                "pass": bool(np.sqrt(np.mean((m_out - m_ref)**2)) < 0.05
                         and abs(20 * np.log10(np.sqrt(np.mean(out**2)) / (
                             np.sqrt(np.mean(ref**2)) + 1e-30))) < 0.1)}
```

### 1.13.8 パターン実行エンジン（cli_runner.py run_all_patterns）

```python
# cli_runner.py（続き）

import yaml
from pathlib import Path
from generators import generate_synthetic_ir, generate_sine, generate_dirac_input

# TC種別→入力信号生成関数 マッピング
TC_INPUT_GEN = {
    "TC-01":  lambda: generate_sine(1000, 3.0),
    "TC-02":  lambda: generate_dirac_input(0.5, 3.0),
    "TC-03":  lambda: generate_sine(1000, 3.0),
    "TC-04":  lambda: generate_sine(1000, 3.0),       # noise shaper違い
    "TC-04A": lambda: generate_sine(1000, 3.0),       # noise shaper違い
    "TC-31a": lambda: generate_dirac_input(0.5, 3.0),
    "TC-31b": lambda: generate_sine(1000, 3.0),
    "TC-32":  lambda: generate_log_sweep(20, 24000, 10.0),
    "TC-33":  lambda: generate_multitone([19000, 20000, 22000], 3.0),
    "TC-34":  lambda: generate_silence(5.0),
    "TC-37":  lambda: generate_sine(1000, 3.0, 0.9886),
    "TC-38":  lambda: generate_sine(40, 1800.0),
    "TC-39":  lambda: generate_sine(1000, 3.0),       # THD sweepは別途6周波数
    "TC-40":  lambda: generate_silence(10.0),
    "TC-41":  lambda: generate_dirac_input(0.5, 1.0),
    "TC-36":  lambda: _generate_stereo_asymmetric(),
    "TC-17":  lambda: generate_imd(60, 7000, 0.25, 3.0),
    "TC-18":  lambda: generate_imd(19000, 20000, 1.0, 3.0),
}

def load_config(path: str = "testdata/config/test_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

def run_all_patterns(config: dict, output_base: Path):
    """全パターン×全TC の直積を実行"""
    for pname, pdef in config["patterns"].items():
        cli = pdef["cli"]
        # OS倍率スイープ展開
        os_values = cli.pop("os", [1])
        if not isinstance(os_values, list): os_values = [os_values]
        for os_val in os_values:
            cli["os"] = os_val
            for ir_name in pdef["srcIrs"]:
                ir_file = generate_synthetic_ir(ir_name)
                for tc in pdef["testCases"]:
                    input_wav = TC_INPUT_GEN.get(tc, lambda: generate_sine(1000, 3.0))()
                    out_dir = output_base / pname / f"os{os_val}" / tc
                    out_dir.mkdir(parents=True, exist_ok=True)
                    args = build_cli_args(pdef, ir_file, input_wav, out_dir)
                    result = run_convopeq(args)
                    _store_result(pname, tc, cli, out_dir, result)
```

### 1.13.9 分析呼出・パス判定（analyzers.py run_test）

```python
# analyzers.py（続き）

# TC→分析関数 ディスパッチテーブル
TC_ANALYZER: dict[str, Callable] = {
    "TC-01":  lambda i, o, s: Analyzers.analyze_dirac(o, s),
    "TC-02":  lambda i, o, s: Analyzers.analyze_dirac(o, s),
    "TC-03":  lambda i, o, s: Analyzers.analyze_thd_sweep(o, s, [1000]),
    "TC-31a": lambda i, o, s: Analyzers.analyze_null_test(i, o),
    "TC-31b": lambda i, o, s: Analyzers.analyze_null_test(i, o),
    "TC-32":  lambda i, o, s: Analyzers.analyze_sweep_transfer(i, o, s),
    "TC-33":  lambda i, o, s: Analyzers.analyze_alias(o, s, 2, [19000, 20000, 22000]),
    "TC-34":  lambda i, o, s: Analyzers.analyze_noise_psd(o, s),
    "TC-36":  lambda i, o, s: Analyzers.analyze_stereo_crosstalk(o[:,0], o[:,1]),
    "TC-37":  lambda i, o, s: {"rms_ulp": float(np.sqrt(np.mean(((i-o)/2**-23)**2)))},
    "TC-39":  lambda i, o, s: Analyzers.analyze_thd_sweep(o, s),
    "TC-41":  lambda i, o, s: Analyzers.analyze_impulse_alignment(o, 0),
}

# 閾値定義（build_config 別）
THRESHOLDS = {
    "debug_msvc":   {"thd": -80, "imd": -80, "noise_floor": -115, "freq_resp": 0.05},
    "release_msvc": {"thd": -100, "imd": -90, "noise_floor": -120, "freq_resp": 0.05},
    "release_icx":  {"thd": -100, "imd": -90, "noise_floor": -120, "freq_resp": 0.05},
}
```

### 1.13.10 テストケース仕様一覧（TC-01〜TC-41 完全定義）

| TC | カテゴリ | 入力信号 | 分析関数 | パス条件（Release） | 備考 |
|----|---------|---------|---------|-------------------|------|
| TC-01 | 伝達関数 | 1kHz正弦波 -6dBFS 3s | `analyze_dirac` | 振幅RMS誤差 ≤ 0.05dB | 基準 |
| TC-01B | 実IR品質 | Dirac -6dBFS 3s (room_correction IR) | `analyze_dirac` | 振幅RMS誤差 ≤ 0.05dB | IR=room_correction |
| TC-02 | 位相精度 | Dirac -6dBFS 3s | `analyze_dirac` | 位相RMS誤差 ≤ 1° | |
| TC-03 | THD | 1kHz正弦波 -6dBFS 3s | `_thd_at_freq` | THD ≤ -100dB | |
| TC-04 | ノイズシェイパ | 1kHz正弦波 -6dBFS 3s | `analyze_noise_psd` | A-weighted ≤ -110dBFS | psychoacoustic |
| TC-04A | ノイズシェイパ | 1kHz正弦波 -6dBFS 3s | `analyze_noise_psd` | A-weighted ≤ -105dBFS | adaptive9 |
| TC-06 | DCブロッカ | 1kHz正弦波 -6dBFS 3s | 特注 | DC成分 ≤ -140dBFS | |
| TC-07 | LPF/HPF伝達 | Dirac -6dBFS 3s (lpf_1k/hpf_20 IR) | `analyze_sweep_transfer` | 形状RMS誤差 ≤ 0.1dB | |
| TC-09 | エイリアシング | 対数スイープ 10s | `analyze_sweep_transfer` | 折り返し ≤ -130dBFS | OS倍率依存 |
| TC-17 | SMPTE IMD | 60Hz+7kHz 4:1 3s | IMD分析 | IMD ≤ -90dB | |
| TC-18 | CCIF IMD | 19kHz+20kHz 1:1 3s | IMD分析 | IMD ≤ -90dB | |
| TC-21 | 群遅延 | 対数スイープ 10s | `analyze_sweep_transfer` | 群遅延RMS ≤ 0.1sample | |
| TC-23 | 出力レベル | 1kHz正弦波 -6dBFS 3s | RMS比較 | RMS偏差 ≤ 0.1dB | |
| TC-24 | THD+N | 1kHz正弦波 -6dBFS 3s | THD+N分析 | THD+N ≤ -80dB | |
| TC-31a | Null Test (Float) | Dirac -6dBFS 3s | `analyze_null_test` | RMS ≤ -138dBFS, Peak ≤ -128dBFS | DCBlocker+Dither常時 |
| TC-31b | Null Test (Double) | 1kHz正弦波 -6dBFS 3s | `analyze_null_test` | RMS ≤ -138dBFS, Peak ≤ -128dBFS | TruePeak+Limiter常時 |
| TC-32 | Log Sweep伝達 | 対数スイープ 10s | `analyze_sweep_transfer` | 振幅RMS ≤ 0.05dB | Farina 2007 |
| TC-33 | Alias Rejection | 19/20/22kHz -6dBFS 3s | `analyze_alias` | イメージ位置 ≤ -130dBFS | 反証テスト |
| TC-34 | PSD | 無音 5s | `analyze_noise_psd` | A-weighted ≤ -110dBFS | 32bit設定時のみ |
| TC-35 | Group Delay | 対数スイープ 10s | `analyze_sweep_transfer` | 理論群遅延 ±5% | Mixed Phase |
| TC-36 | Stereo Crosstalk | L=1kHz R=無音 3s | `analyze_stereo_crosstalk` | L→R ≤ -140dBFS | |
| TC-37 | Numerical Transp. | 1kHz -0.1dBFS 3s | RMS ULP計算 | RMS ≤ 1 ULP, Peak ≤ 2 ULP | **Float Path のみ** |
| TC-38 | Long-run安定 | 40Hz -6dBFS 30min | NaN/Inf/DC検査 | 異常0, ドリフト ≤ 0.01dB | |
| TC-39 | THD Sweep | 6周波数 -6dBFS 各3s | `analyze_thd_sweep` | THD ≤ -100dB (全周波数) | 100/1k/5k/10k/18k/20kHz |
| TC-40 | Dither分布 | 無音 10s | ヒストグラム比較 | KL Divergence ≤ 0.01 | |
| TC-41 | Impulse Alignment | Dirac -6dBFS 1s | `analyze_impulse_alignment` | ピーク位置 ±1sample | |

### 1.13.11 CI実装設計（GitHub Actions）

```yaml
# .github/workflows/audio-quality-tests.yml（Phase 3 実装）
name: Audio Quality Tests
on:
  pull_request: { types: [opened, synchronize] }
  schedule: [{ cron: '0 6 * * *' }]  # Nightly (06:00 UTC)

jobs:
  quick:
    if: github.event_name == 'pull_request'
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - run: cmake --build build --config Release
      - run: python run_quality_tests.py --pattern P1,P2 --tc TC-01,TC-03,TC-04,TC-04A,TC-31,TC-32
        timeout-minutes: 5

  full:
    if: github.event_name == 'pull_request'
    needs: quick
    runs-on: windows-latest
    steps:
      - run: python run_quality_tests.py --all-patterns
        timeout-minutes: 30

  nightly:
    if: github.event_name == 'schedule'
    runs-on: windows-latest
    steps:
      - run: python run_quality_tests.py --all-patterns --all-tc --all-os
        timeout-minutes: 180
```

### 1.13.12 測定データ構成

```
testdata/
├── generated/              # generators.py 自動生成（.gitignore 対象）
│   ├── dirac_test.wav      # float32, 48kHz, mono, 8192samples
│   ├── lpf_1k_test.wav     # float32, 48kHz, stereo, 129taps
│   ├── hpf_20_test.wav     # float32, 48kHz, stereo, 129taps
│   ├── sine_*.wav
│   ├── logsweep_*_inverse.wav  # Farina 逆フィルタ
│   ├── silence_*.wav
│   ├── multitone_*.wav
│   └── imd_*.wav
├── golden/                 # ゴールデン参照（Git LFS 管理推奨）
│   └── {pattern}/{tc}/output.wav + golden_metrics.json
└── config/
    └── test_config.yaml    # パターン定義＋閾値定義
sampledata/
└── impulse_room_correction_hpf_lpf.wav   # 実測IR（既存）
```

### 1.13.13 工数内訳

| フェーズ | タスク | 人日 | 成果物 |
|---------|--------|------|--------|
| **Phase 1: DSP品質試験** | | **58〜73** | |
| 1-1 | CLI拡張6種 | 3 | MainWindow.cpp パーサー追加 |
| 1-2 | `generators.py` 全関数 | 5 | 入力信号生成・IR生成 |
| 1-3 | `analyzers.py` 分析関数 | 10 | 全TC分析ロジック |
| 1-4 | `cli_runner.py` 実行エンジン | 5 | CLI起動・結果収集 |
| 1-5 | `golden_calculator.py` | 3 | ゴールデン比較FW |
| 1-6 | `test_config.yaml` | 2 | パターン定義 |
| 1-7 | 既存34TC 検証 | 20 | 各TCの閾値調整 |
| 1-8 | 新規11TC 実装 | 10 | TC-31〜41 分析関数 |
| 1-9 | RSpec 相当テスト | 5〜10 | テスト自体のテスト |
| **Phase 2: Runtime試験** | | **40〜50** | |
| 2-1 | TC-25/27/30/38 安定化 | 20 | 長期テスト |
| 2-2 | ISR stressed 試験 | 10 | TC-11〜14 |
| 2-3 | Adaptive capture 試験 | 10 | 学習時品質 |
| **Phase 3: レポート/CI** | | **20〜25** | |
| 3-1 | JUnit XML 出力 | 5 | CI連携用 |
| 3-2 | HTML レポート（RMAA互換） | 5 | 可視化 |
| 3-3 | GitHub Actions workflow | 5 | CI自動化 |
| 3-4 | ドキュメント整備 | 5 | 運用手順 |
| **予備** | | **30〜70** | 実績応じて |

---

# Part 2: 調査確定事項（v7.4 コードベース調査完了）

本パートは v7.3.6 時点で未確定だった全6項目について、コードベース調査を完了し確定した結果を記載する。

---

## 2.1 RecoveryHistory / Seqlock — Timer Thread 結合確定

### 調査結果（2026-07-16 コードベース確認）

**調査対象**: `src/audioengine/AudioEngine.Timer.cpp:1591-1657`

```cpp
void AudioEngine::executeRecoveryAction(convo::RecoveryAction action) noexcept
{
    switch (action) {
        case convo::RecoveryAction::Throttle:  ... break;
        case convo::RecoveryAction::Recover:   ... break;
        case convo::RecoveryAction::Restore:   ... break;
        case convo::RecoveryAction::Safe:      ... break;
        case convo::RecoveryAction::Critical:  ... break;
        default: break;
    }
    diagLog("[RECOVERY] execute action=" + ...);
}
```

### 確定設計

| 項目 | 決定内容 |
|------|---------|
| **挿入位置** | `executeRecoveryAction()` の**先頭**（switch 文の直前）。全アクション種別を確実に記録する。 |
| **呼出契機** | `executeRecoveryAction()` が Timer Thread から呼ばれた時点で常に発火 |
| **記録内容** | timestampUs, source (Timer), action, healthState, pendingRetire, maxRetireAgeUs |
| **単一writer契約** | `executeRecoveryAction()` は Timer Thread 単一からのみ呼ばれる設計が確立済み。`RuntimeHealthMonitor` → `m_actionCallback` → `executeRecoveryAction()` の経路のみ。 |
| **diagLog との関係** | `recordRecoveryAction()` は Seqlock 書込。`diagLog()` は文字列ログ。両方実行する（役割が異なる）。 |
| **リスク** | 低。 |

---

## 2.2 JUCE WaitableEvent 仕様 — 確認完了

### 調査結果（2026-07-16 コードベース確認）

| 項目 | 確認結果 |
|------|---------|
| **JUCE バージョン** | **8.0.12**（CMakeLists.txt:5） |
| **ヘッダファイル** | `JUCE/modules/juce_core/threads/juce_WaitableEvent.h`（プロジェクトルート直下） |
| **コンストラクタ** | `explicit WaitableEvent(bool manualReset = false) noexcept;` |
| **デフォルト** | **`manualReset = false`（auto-reset）** ✅ 計画書の前提と一致 |
| **signal()** | 非ブロッキング。auto-reset + wait中スレッドあり → 1つ起床。wait中スレッドなし → 次回wait()が即座に戻る。起床後は自動リセット。 |
| **wait()** | デフォルト無限待機（-1.0）。timeout時はfalse返却。auto-reset: return後に自動リセット。 |
| **内部実装** | `std::mutex` + `std::condition_variable` + `std::atomic<bool> triggered{false}` |
| **JUCE 7→8 の変更** | 本質的な仕様変更なし。JUCE 8 でも同一インターフェース。 |
| **RT-safe 性** | `signal()` は `std::atomic<bool> triggered` への store + `condition_variable.notify_one()`。`notify_one()` は syscall を発行する可能性があるが、Audio Thread では禁止されているブロッキング呼出ではなく、RT-safe と見なせる。 |

### 確定設計

OutputCaptureSink は `juce::WaitableEvent` をデフォルト構築（auto-reset）で使用する。実装初日に上記ヘッダでの確認は不要（バージョンアップ時のみ確認）。

---

## 2.3 OutputCaptureSink 単体テスト — 設計確定

### 調査結果（2026-07-16 コードベース確認）

**テストフレームワーク**: 独自 `check()` マクロ + `g_testsPassed`/`g_testsFailed` + `std::cerr << "[FAIL]"` + `int main()`。

### テストケース一覧

| # | テスト名 | テスト内容 | 検証方法 |
|---|---------|-----------|---------|
| 1 | **LifecycleContract** | 正規ライフサイクル（setOutputPath→setCapturePoint→startCapturing→capture→stopCapturing→wait→destroy）が例外なく完了する | 戻り値チェック、全ステップ通過 |
| 2 | **DoubleStartPrevention** | `startCapturing()` を2回呼ぶ → デバッグビルドで jassert。本番ビルドでは2回目が無視されることを確認 | jassert メッセージ有無（テストプロセス出力）|
| 3 | **ShutdownOrder** | setCaptureSink(nullptr)→stopCapturing→waitForThreadToExit→destroy の順序を検証（デストラクタの jassert が発火しない） | 正常終了コード 0 |
| 4 | **RingBufferFull** | リングバッファ容量を超える capture() 呼出 → `droppedBlocks_` が increment される | `droppedBlocks_` 値（公開カウンタとして読出し可能にするか、pop 残数で間接確認）|
| 5 | **DrainAll** | 複数ブロックを capture() で push → signal() → BG Thread が全ブロックを drain する | WAV 出力サンプル数 = push 総サンプル数 |
| 6 | **FloatOverload** | `capture(const float*, ...)` で float データ注入 → AudioBlock.L/R が正しく格納される | 先頭サンプルの double 変換値一致 |
| 7 | **DoubleOverload** | `capture(const double*, ...)` で double データ注入 → AudioBlock.L/R が正しく格納される | 先頭サンプルの一致 |
| 8 | **CapturePointNone** | CapturePoint::None 設定時は capture() が何もしない（RingBuffer に push されない） | pop で empty 確認 |
| 9 | **PostDitherFloatPath** | Float Path PostDither 相当の float データ注入 → 正しく変換・格納される | 値一致 |
| 10 | **TimestampPassthrough** | `capture()` に渡した `timestampUs` が AudioBlock で正しく読み出せる | block.timestampUs == 入力 timestampUs（全オフセット加算後の値）|

### 実装上の決定

- CMake 登録パターン: 既存テストと同一（`add_executable` + `add_test` + `target_compile_features`）
- `droppedBlocks_` をテストから観測可能にするため、`getDroppedBlockCount()` public getter を追加する（const, relaxed load）。

---

## 2.4 OutputFilter const getter（--cli-dump-filter-coeffs）— 設計確定

### 調査結果（2026-07-16 コードベース確認）

**ファイル**: `src/OutputFilter.h:41-143`

```cpp
struct BiquadCoeff {                    // OutputFilter.h:41
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a1 = 0.0, a2 = 0.0;         // a0 は除算済み（正規化済み）
};

class OutputFilter {
private:
    BiquadCoeff hcCoeff[3][2];  // [HCMode][stage] ① ハイカット
    BiquadCoeff lcCoeff[2];     // [LCMode]           ① ローカット
    BiquadCoeff hpfCoeff;       //                     ② ハイパス（固定）
    BiquadCoeff lpCoeff[3][2];  // [HCMode][stage]    ② ローパス
    // + 状態変数 (hcState, lcState, hpfState, lpState)
};
```

### 既存 getter パターン（参考）

```cpp
// ConvolverProcessor.h:1073 — [[nodiscard]] + const noexcept パターン
[[nodiscard]] double getCurrentIRScale() const noexcept {
    return convo::consumeAtomic(currentIRScale, std::memory_order_acquire);
}
```

### 確定設計

```cpp
// OutputFilter.h に追加（public: セクション）
[[nodiscard]] const BiquadCoeff (&getHCCoeffs() const noexcept)[3][2] { return hcCoeff; }
[[nodiscard]] const BiquadCoeff (&getLCCoeffs() const noexcept)[2]    { return lcCoeff; }
[[nodiscard]] const BiquadCoeff& getHPFCoeff() const noexcept          { return hpfCoeff; }
[[nodiscard]] const BiquadCoeff (&getLPCoeffs() const noexcept)[3][2]  { return lpCoeff; }
```

JSON 出力フォーマット例（`--cli-dump-filter-coeffs`）:
```json
{
  "hcCoeff": [
    [{"b0":0.231,"b1":0.462,"b2":0.231,"a1":-0.537,"a2":0.231},
     {"b0":1.000,"b1":0.000,"b2":0.000,"a1":0.000,"a2":0.000}],
    ...
  ],
  "lcCoeff": [
    {"b0":0.994,"b1":-1.987,"b2":0.994,"a1":-1.987,"a2":0.988},
    ...
  ],
  "hpfCoeff": {"b0":0.994,"b1":-1.987,"b2":0.994,"a1":-1.987,"a2":0.988}
}
```

---

## 2.5 ライン番号 — 再確認完了

前回検証（v7.3.6 validation report）で全ライン番号を実コードと突合済み。コード変更がないため、全ライン番号は現在も正確。

| 項目 | 確定ライン | ステータス |
|------|-----------|-----------|
| `processOutput()` PreOutputFilter 挿入 | DSPCoreIO.cpp L411 直前（`pushAdaptiveCaptureBlocks` 呼出直後 相当） | ✅ 確定 |
| `processOutputDouble()` PreOutputFilter 挿入 | DSPCoreDouble.cpp L651 直前 | ✅ 確定 |
| `processOutput()` PostDither 挿入 | DSPCoreIO.cpp L503（`applyFixedLatencyDelay`）〜L505（float変換開始）の間 | ✅ 確定 |
| `processOutputDouble()` PostDither 挿入 | DSPCoreDouble.cpp L803（`applyFixedLatencyDelay`）〜L805（`FloatVectorOperations::copy`）の後 | ✅ 確定 |
| `outputFilter.process()` (Float Path) | DSPCoreFloat.cpp:352 | ✅ 確定 |
| `outputFilter.process()` (Double Path) | DSPCoreDouble.cpp:503 | ✅ 確定 |

**実装時の注意**: コードは継続的に変更される可能性があるため、実装着手時点で再度ライン番号を確認すること。

---

## 2.6 単回起動契約 — 設計確定

### 調査結果

既存テストコードにおける jassert の扱い:

- `src/tests/RuntimeWorldAuthorityProjectionTests.cpp:222`: `if (!contains(header, "jassert(runtimeWorld != nullptr"))` — テストが jassert メッセージの有無を確認するパターンが既存。
- JUCE の `jassert` はデバッグビルドで `JUCE_BREAK_IN_DEBUGGER`、リリースビルドで no-op。

### 確定設計

```cpp
void startCapturing() {
    if (started_.exchange(true)) {
        // 2回目呼出: デバッグビルドでは jassert で異常検出
        jassertfalse;
        // リリースビルドでは静かに無視（既存 JUCE パターンに従う）
        return;
    }
    startThread();
}
```

| 項目 | 決定内容 |
|------|---------|
| **二重起動時の動作** | デバッグビルド: `jassertfalse` で停止＋継続。リリースビルド: 静かに無視（no-op return）。 |
| **根拠** | JUCE 標準の `jassert` パターンに従う。全既存テストと一貫性を維持。 |
| **代替案（std::terminate）** | 採用しない。`std::terminate` は回復不可能なプロセス異常終了を引き起こし、CI 環境でのデバッグ情報取得を困難にする。 |
| **テスト方法** | `DoubleStartPrevention` テストケースが jassert メッセージ出力を確認する（既存 `RuntimeWorldAuthorityProjectionTests` と同一パターン）。 |

---

# Part 3: Appendix

## Appendix A: 改訂履歴（全バージョン）

| 版 | 日付 | 主要内容 |
|---|------|---------|
| v1.0〜v6.3 | 〜2026-06-22 | 初期設計・ISR追加・RMAA統合・TC拡充・AES17・最終調整（凍結） |
| v7.0 | 2026-07-06 | コード監査反映：TC-30 API未実装確定、TC-01B参照パス修正、テストケース数34件修正 |
| v7.1 | 2026-07-06 | RT安全性・責務分離：OutputCaptureSink独立クラス化、SPSC+BG WAV、原子生ポインタ |
| v7.2 | 2026-07-06 | 二重管理解消・逐次WAV・seqlock：CapturePoint統合、sequence lock、WAV逐次書込 |
| v7.3 | 2026-07-06 | 全指摘解決：pushBecameNonEmpty(), WaitableEvent, シャットダウン順序, release/acquire, drain-all, パラメータパターン7種 |
| v7.3.1 | 2026-07-15 | 外部検証反映：音響工学公式確認、コードベース整合性確認、未確定事項6件確定 |
| v7.3.2 | 2026-07-15 | 二次深掘検証反映：シャットダウンシーケンス統合、CLI未提供パラメータ6種特定 |
| v7.3.3 | 2026-07-15 | 三次DSPパイプライン検証反映：capture()シグネチャ統合、pushBecameNonEmptyWithWriter新設 |
| v7.3.4 | 2026-07-15 | 四次統合検証反映：testCaptureQueue独立、processOutputDouble対応、§8測定データ新設 |
| v7.3.5 | 2026-07-15 | 五次DSP最終検証反映：capture() float overload、TC-31a閾値緩和、Farina 2007方式 |
| v7.3.6 | 2026-07-15 | 六次実装完全性検証：M3 OutputCaptureSinkメンバ完全性、M5 TC網羅性、M11/M12追加 |
| **v7.4** | **2026-07-16** | **再構成版＋Phase1詳細設計追加版：実装設計・調査確定事項・Appendix の3部構成。§1.13 Phase1設計セクション新設（テストパラメータ10次元CLIマッピング、generators.py/analyzers.py/cli_runner.py/golden_calculator.py詳細コード設計、全41TC完全定義、CI設計、工数内訳）。未確定6項目確定。timestampUs 設計反映。Appendix D 旧内容を §1.13 に移動・拡充。** |

---

## Appendix B: 既存コードパターン活用マップ

| 設計要件 | 既存パターン | ファイル |
|---------|------------|---------|
| SPSC RingBuffer | `LockFreeRingBuffer<T, Capacity>` | `src/LockFreeRingBuffer.h` |
| Audio Thread→RingBuffer 書込 | `pushAdaptiveCaptureBlocks()` | `src/audioengine/DSPCoreIO.cpp` |
| 非同期 drain | `asyncSink()` + `flushLogBuffer()` | `src/audioengine/AudioEngine.Timer.cpp` |
| capture 用 data struct | `AudioBlock`（double[256]×2ch, trivially_copyable）| `src/audioengine/AudioEngine.h:24` |
| atomic enum メンバ | `std::atomic<NoiseShaperType>`, `<HCMode>`, `<OversamplingType>` | `src/audioengine/AudioEngine.h` |
| 原子生ポインタ注入 | `atomic<OutputCaptureSink*>` | `RuntimeHealthMonitor::setRetireRouter()` 準拠 |
| 単調増加 counter | `audioCallbackEpochCounter`（`fetchAddAtomic`）| `AudioEngine.h:1482` |
| static_assert 群 | `DiagEvent`（5種）| `AudioEngine.h:447-460` |
| 逐次WAV書込 | `AudioFormatWriter::writeFromFloatArrays()` | JUCE modules |
| 段階的 shutdown | `~MainWindow()` step 1〜10 | `src/MainWindow.cpp:1000-1033` |
| RecoveryAction 実行 | `executeRecoveryAction()` | `src/audioengine/AudioEngine.Timer.cpp:1591` |
| WaitableEvent (auto-reset) | `juce::WaitableEvent` デフォルト構築 | JUCE modules |
| OSタイプ・倍率 API | `setOversamplingType()`, `setOversamplingFactor()` | `AudioEngine.h:1392,1389` |
| HC/LCモード API | `setConvHCFilterMode()`, `setConvLCFilterMode()` | `AudioEngine.h:1401,1404` |
| SoftClip/Saturation API | `setSoftClipEnabled()`, `setSaturationAmount()` | `AudioEngine.h:1356,1359` |
| fetchAddAtomic 実 API | `convo::fetchAddAtomic(std::atomic<T>&, T, memory_order)` | `AtomicAccess.h:91` |
| capture 実データフロー | `double* dataL/dataR`（alignedL/alignedR の別名）| `DSPCoreIO.cpp:360-361` |
| pushWithWriter ゼロコピーパターン | `pushWithWriter([&](AudioBlock& block){...})` | `DSPCoreIO.cpp:141-168` |
| DSP処理ドメイン（oversampled） | `juce::dsp::AudioBlock<double> processBlock(...)` | `DSPCoreFloat.cpp:242` / `DSPCoreDouble.cpp:391` |
| テストパターン | 独自 `check()` マクロ、Catch2/doctest/gtest 不使用 | `src/tests/*.cpp` |

---

## Appendix C: コード監査使用ツール

| カテゴリ | ツール | 用途 |
|---------|-------|------|
| **WSL検索** | grep / ripgrep (rg) / ast-grep / fd / fzf / sed / awk | 全文検索・AST構造検索・ファイル検索 |
| **MCP** | headroom MCP / context-mode MCP / AiDex MCP / Serena MCP | コンテキスト圧縮・データ処理・コード検索・プロジェクト管理 |
| **CLI** | semble / cocoindex-code (ccc) / graphify | 意味検索・インデックス作成・知識グラフ |

---

~~Appendix D の内容は §1.13「Phase 1: 音質評価自動化フレームワーク設計」に移動・拡充済み。~~

## Appendix D: コードベース検証レポート（v7.3.6）

詳細は `doc/work68/automatic_sound_test_plan_v7.3_validation_report.md` を参照。

### 検証サマリー

全23項目のコードベース検証結果:

| カテゴリ | 件数 |
|---------|------|
| ✅ 適合 | 21/23 項目 |
| ⚠️ 軽微不一致 | 2 件（CLIカウント、AudioBlock padding記述）→ **v7.4 で修正済み** |
| ❌ 不一致（要修正） | **0 件** |

### 主要検証項目

| # | 項目 | 判定 | 確認内容 |
|---|------|------|---------|
| 1 | CLI オプション数 | ✅ | findValue 24 + hasFlag 3 = 27種（計画書修正済み）|
| 2 | processOutput() L341-514 | ✅ | 全ラインリファレンス一致 |
| 3 | processOutputDouble() L620-810 | ✅ | 全ライン一致 |
| 4 | CapturePoint 対応 | ✅ | PreOutputFilter L242/391 は AudioBlock 作成位置で正確 |
| 5 | ~MainWindow() 10step | ✅ | L1000-1033 完全一致 |
| 6 | AudioBlock 型特性 | ✅ | sizeof=4120, alignof=8, trivially_copyable |
| 7 | LockFreeRingBuffer | ✅ | push/pushWithWriter 確認。PushResult 拡張は設計通り |
| 8 | API 存在確認 | ✅ | setOversamplingType 等6API 全て実在 |
| 9 | parseCliOrderMode | ✅ | マッピング正確 |
| 10 | parseCliPhaseMode/NoiseShaper | ✅ | マッピング正確 |
| 11 | ProcessingState | ✅ | testCaptureQueue 新設設計は既存パターンと整合 |
| 12 | TC-37 PeakLimiter | ✅ | kPLThreshold=0.841 < 0.9886 で常時動作 |
| 13 | 既存テストFW | ✅ | CMake+CTest、Catch2不使用、独自check()マクロ |
| 14 | WaitableEvent | 📌 | JUCE 依存、ビルド時確認推奨 |
| 15 | fetchAddAtomic | ✅ | AtomicAccess.h:91 実在確認 |
| 16 | 既存Pythonツール | ✅ | create_dirac_ir.py 16bit/C:/TEMP 確認 |
| 17 | RecoveryHistory | ✅ | 未実装（Phase 0 対象） |
| 18 | adaptiveCaptureActiveRt | ✅ | testCaptureQueue 設計のテンプレート確認 |
| 19 | CMake パターン | ✅ | add_executable+add_test 確認 |
| 20 | PeakLimiter定数検証 | ✅ | 全定数数学的に正当 |

---

## Appendix E: 外部検証レポート補足

### F.1 CapturePoint 設計意図

- **PreOutputFilter**: オーバーサンプリングドメインのデータをキャプチャ。OS倍率に応じた高サンプリングレートの生DSP出力。エイリアシング評価やフィルタ特性の詳細分析に使用。
- **PostOutputFilter**: ベースレートにダウンサンプリング後、DC Blocker・NaN/Inf除去を通過したデータ。通常の音質評価に使用（デフォルト）。
- **PostDither**: ディザー/ノイズシェイパー適用後、遅延補正済みの最終出力データ。ビットパーフェクト評価や量子化ノイズ分析に使用。

### F.2 Seqlock 設計意図

Seqlock は RecoveryHistory（最大64件）の読み取り整合性を保証するための機構。

**Writer（Timer Thread 単一）**:
```
generationを奇数に変更 → RecoveryEvent書込 → generationを偶数に変更
```

**Reader（BG Thread / Audio Thread）**:
```
generation(acquire)を読む → 奇数ならリトライ
偶数なら RecoveryEvent を copy → generation(acquire)を再読
一致すれば copy 完了（整合性保証）
8回リトライ失敗 → 当該 slot を破棄
```

**single-writer 契約**: `recordRecoveryAction()` は Timer Thread 単一からのみ呼び出す。これにより slot 取得の CAS が不要。

### F.3 WAV 逐次書込設計意図

- **問題**: 従来設計では全キャプチャデータをメモリに保持し、停止後に一括 WAV 書込。30分テストで 660MB のメモリ消費。
- **解決**: `AudioFormatWriter::writeFromFloatArrays()` で逐次書込。BG Thread がリングバッファから drain するたびに WAV ファイルに追記。メモリ消費はリングバッファ分（4120byte × 4096 ≈ 16MB）のみ。

---

## Appendix F: 既存 tools/diagnostics/ ツール一覧

| ファイル | 用途 | Phase 1 での扱い |
|---------|------|-----------------|
| `create_dirac_ir.py` | Dirac IR 生成（16bit PCM → C:/TEMP/） | generators.py で float32 版に置換 |
| `create_test_irs.py` | LPF/HPF FIR 生成（32bit float → C:/TEMP/） | generators.py に統合、出力先変更 |
| `analyze_conv_output.py` | Tone周波数・振幅検出 | analyzers.py で活用 |
| `analyze_ir.py` | WAVヘッダ・チャンク解析 | analyzers.py で活用 |
| `analyze_compare.py` | 比較分析 | analyzers.py で活用 |
| `analyze_verify.py` | 検証分析 | analyzers.py で活用 |
| `compare_raw.py` | Null Test 差分計算 | analyzers.py で活用 |
| `compare_dirac.py` | ブロック境界・DC分析 | analyzers.py で活用 |
| `compare_all_irs.py` | 全IR比較 | analyzers.py で活用 |
| `compare_input_vs_conv.py` | 入力 vs 畳み込み比較 | analyzers.py で活用 |
| `generate_test_signal.py` | テスト信号生成 | generators.py で活用 |
| `check_build.py` | ビルド確認 | CI スクリプト維持 |
| `run_conv_diag.ps1` | 診断実行 | CI スクリプト維持 |
