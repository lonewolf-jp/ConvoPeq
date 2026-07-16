# ConvoPeq 音質評価自動化 改修計画書 v7.3.6

**バージョン**: 7.3.6（六次実装完全性検証反映版・OutputCaptureSinkメンバ完全性・TC網羅性・DoublePath制約・信号マッピング完備）
**策定日**: 2026-07-06（v7.3）／2026-07-15（v7.3.1〜v7.3.5）／2026-07-15（v7.3.6 六次）
**ベース**: v6.3+（2026-06-22）＋ 全レビュー反映 ＋ 一次〜五次検証 ＋ **六次実装完全性検証（2026-07-15）**
**ステータス**: **Phase 0 実装開始可**（致命M3解決済み。OutputCaptureSinkメンバ宣言完備。全94件指摘解決済み）
**対象**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

## 1. 全体アーキテクチャ

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

既存CLI `MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す。
新たな実行可能ファイルは作成しない。

---

## 2. Phase 0: CLI拡張 + OutputCaptureSink + RecoveryHistory API（工数: 9日）

### 2.1 CapturePoint enum

```cpp
enum class CapturePoint : uint8_t {
    None             = 0,   // キャプチャしない
    PreOutputFilter  = 1,   // OutputFilter 適用前（DSP生出力）
    PostOutputFilter = 2,   // OutputFilter 適用後（デフォルト）
    PostDither      = 3,    // ディザー後（最終出力）
};
```

`--cli-capture-mode` の引数: `none` / `pre-filter` / `post-filter` / `post-dither`

> **[v7.3.3 検証確定] CapturePoint と実パイプラインの対応**:
> `processOutput()`（`AudioEngine.Processing.DSPCoreIO.cpp:342-514`）の実データフロー:
>
> | CapturePoint | 挿入位置 | データ型 | 備考 |
> |---|---|---|---|
> | `PreOutputFilter` | 処理関数内 `outputFilter.process()` 呼出前（`DSPCoreFloat.cpp:242` / `DSPCoreDouble.cpp:391`） | `juce::dsp::AudioBlock<double>`（オーバーサンプリングドメイン） | OS倍率に応じた高サンプリングレート |
> | `PostOutputFilter` | `processOutput()` L412-419（既存 `pushAdaptiveCaptureBlocks` と同一位置） | `double* dataL/dataR`（ベースレート） | DC Blocker + NaN/Inf除去後、Dither前 |
> | `PostDither` | `processOutput()` **L503後・L505前**（`applyFixedLatencyDelay` 後、float変換前） | `double* dataL/dataR`（ベースレート） | Dither/NoiseShaper + 遅延補正後の最終doubleデータ |
>
> **重要**: 3.capturePoint 全てでデータは `double*`（生ポインタ）または `juce::dsp::AudioBlock<double>`（非所有view）であり、
> `juce::AudioBuffer<double>` は実パイプラインのどこにも存在しない。
> `capture()` シグネチャは生ポインタ `const double*` を取る（§2.2 参照）。

> **[v7.3.5 検証確定] PostDither データ型は `float* dstL/dstR`**:
> `processOutput()` の L505-510 で `static_cast<float>(...)` 変換後、データは `dstL[i]` / `dstR[i]` に
> 格納される。これは **`float*`** 型であり、`double* dataL/dataR` ではない。
> したがって `capture()` は CapturePoint に応じて2つの overload が必要:
>
> ```cpp
> // PostFilter（PreDither）: double* dataL/dataR
> void capture(const double* left, const double* right, int numSamples,
>              uint64_t timestampUs) noexcept;
>
> // PostDither: float* dstL/dstR（float変換後の最終データ）
> void capture(const float* left, const float* right, int numSamples,
>              uint64_t timestampUs) noexcept;
> ```
>
> `pushBecameNonEmptyWithWriter()` の writer ラムダもオーバーロードに応じて
> `const float*` または `const double*` をキャプチャする。
> float版では Writer ラムダ内で、リングスロットの `AudioBlock.L/R`（double[256]）
> に `static_cast<double>(srcF[i])` で格納する（AudioBlock は double[256] 固定のため）。

> **[v7.3.5 CapturePoint 対応表（v7.3.3を修正）]**:
>
> | CapturePoint | 挿入位置（Float Path） | 挿入位置（Double Path） | データ型 | 備考 |
> |---|---|---|---|---|
> | `PreOutputFilter` | `DSPCoreFloat.cpp:242` / `DSPCoreDouble.cpp:391` | 同上（double path も同一） | `juce::dsp::AudioBlock<double>` | OS倍率ドメイン |
> | `PostOutputFilter` | `pushAdaptiveCaptureBlocks(L389-411)` 直前 | `pushAdaptiveCaptureBlocks(L651-658)` 直前 | `double* dataL/dataR` | DCBlocker+NaN除去後、Dither前 |
> | `PostDither` | `applyFixedLatencyDelay` 後の**float変換後** (`dstL/dstR` への `static_cast<float>` 後) | `applyFixedLatencyDelay(L801)`+`FloatVectorOperations::copy(L803-805)` 後、`buffer.getWritePointer(ch)` (double変換なし。Double Path は buffer が `juce::AudioBuffer<double>&`) | Float Path: **`float* dstL/dstR`** / Double Path: **`double* dataL/dataR`**（buffer が double 精度のまま） | ★ v7.3.5: Float Path PostDither は float*。v7.3.3 の「double*」記述は誤り |

> **[v7.3.5 検証確定] testCaptureQueue の RT-safe atomic snapshot ロード仕様** (L2):
> `buildAudioThreadProcessingState()` (AudioEngine.h:3625) 内で:
> ```cpp
> // snapshot boot 時に OutputCaptureSink ポインタを acquire で安全に読み出し
> OutputCaptureSink* sink = convo::consumeAtomic(outputCaptureSink_, std::memory_order_acquire);
> .testCaptureQueue = (sink != nullptr) ? &sink->ringBuffer_ : nullptr;
> ```
> `outputCaptureSink_` は AudioEngine に `std::atomic<OutputCaptureSink*> outputCaptureSink_{nullptr}` として新設。
> `setCaptureSink(ptr)` は `convo::publishAtomic(outputCaptureSink_, ptr, release)`。
> `setCaptureSink(nullptr)` で Audio Thread が次サイクルから capture() を skip する。
> これにより `buildAudioThreadProcessingState()` 内での acquire load 1回のみで RT-safe を保証。

### 2.2 OutputCaptureSink クラス

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
    // ★ [v7.3.6 M3 修正] 4つのメンバ変数は capture() で使用されるが宣言がなかった。
    //   実装初日にビルドエラーになる致命的问题のため、追加宣言+seter を即座に定義。
    void setAudioParams(double sampleRateHz, int bitDepth,
                        int coeffBankIndex, uint64_t sessionId) noexcept {
        sampleRateHz_ = sampleRateHz;
        bitDepth_ = bitDepth;
        coeffBankIndex_ = coeffBankIndex;
        sessionId_ = sessionId;
    }
    // ★ Single-shot object: 1セッション専用。再利用禁止。
    //   ライフサイクル: setOutputPath() → setCapturePoint() → startCapturing() → setCaptureSink(nullptr) → stopCapturing() → waitForThreadToExit() → destroy
    // ★ 契約: startCapturing() は単回起動（2回目の呼出は禁止）。
    //   OutputCaptureSink は1セッション専用。stopCapturing() 後の再利用は禁止。
    //   startCapturing() で stopRequested_ を false に初期化する必要はない。
    //   startCapturing() 以降の setOutputPath() は禁止

    // ── Audio Thread API (RT-safe: push + signal のみ) ──
    // ★ [v7.3.3] 実パイプラインは double* 生ポインタ（alignedL/alignedR）。
    //   juce::AudioBuffer<double> は実パイプラインに存在しないため、生ポインタを使用。
    //   既存 pushAdaptiveCaptureBlocks() と同じ (const double*, const double*, int) 形式。
    void capture(const double* left, const double* right, int numSamples,
                 uint64_t timestampUs) noexcept;

    // ── Background Thread ────────────────────────────
    void run() override;
    void startCapturing() { jassert(!started_.exchange(true)); startThread(); }
    void stopCapturing()  { jassert(started_.load()); stopRequested_.store(true, release); wakeEvent_.signal(); }
    // OutputCaptureSink は1セッション専用。setCaptureSink(nullptr) の後に呼ぶこと。
    // ★ startCapturing/stopCapturing はデバッグビルドで二重起動・二重停止を検出。
    // ★ move/copy は禁止（Thread 所有権の明確化）
    OutputCaptureSink(const OutputCaptureSink&) = delete;
    OutputCaptureSink& operator=(const OutputCaptureSink&) = delete;
    OutputCaptureSink(OutputCaptureSink&&) = delete;
    OutputCaptureSink& operator=(OutputCaptureSink&&) = delete;

private:
    // SPSC RingBuffer（既存 LockFreeRingBuffer 流用）
    static constexpr int kBlockSize    = 256;
    static constexpr int kRingCapacity = 4096;  // ≈ 21秒 @ 48kHz
    LockFreeRingBuffer<AudioBlock, kRingCapacity> ringBuffer_;
    std::atomic<CapturePoint> capturePoint_{CapturePoint::PostOutputFilter};
    juce::File outputPath_;  // Thread 開始前に設定固定
    double sampleRateHz_{48000.0};   // ★ [v7.3.6 M3] capture() L200 で使用。setAudioParams() で設定
    int bitDepth_{64};               // ★ [v7.3.6 M3] capture() L201 で使用。setAudioParams() で設定
    int coeffBankIndex_{0};          // ★ [v7.3.6 M3] capture() L202 で使用。setAudioParams() で設定
    uint64_t sessionId_{0};          // ★ [v7.3.6 M3] capture() L203 で使用。setAudioParams() で設定
    std::unique_ptr<float[]> convertBufferL_, convertBufferR_;  // run()で事前確保。double→float32変換用
    // ★ [v7.3.3] WAV出力フォーマット: IEEE 754 float32（24bit有効仮数）。
    //   パイプラインは double（52bit仮数）で処理するが、WAV出力時に float32 に変換する。
    //   TC-37「1 ULP」は float32 の ULP を意味する（double の ULP ではない）。
    //   想定ファイルサイズ: 48kHz/2ch/float32 = 384kB/s。TC-38(30分) ≈ 660MB。
    juce::WaitableEvent wakeEvent_;
    std::unique_ptr<juce::AudioFormatWriter> wavWriter_;  // 逐次WAV書込
    std::atomic<bool> stopRequested_{false};

    // ★ 統計カウンタ（QA診断用、RT-safe: atomic relaxed）
    std::atomic<uint64_t> droppedBlocks_{0};        // push 失敗（RingBuffer 満杯）累計
    std::atomic<uint64_t> seqlockRetryFailed_{0};   // Seqlock Reader リトライ失敗累計
    std::atomic<bool> started_{false};              // 単回起動検証用（jassert で使用）
};
```

#### capture() — Audio Thread の責務は push + signal のみ

> **[v7.3.3 検証確定] 実装方針**: 既存 `pushAdaptiveCaptureBlocks()`（`DSPCoreIO.cpp:122-174`）と同じ
> `pushWithWriter()` ラムダパターンを採用する。これにより:
> - AudioBlock ローカル変数のゼロ初期化（4120byte）+ コピー（4120byte）が不要
> - ラムダが直接リングスロットに書き込むため、Audio Thread のメモリ書き込み量が **1/3** に削減
> - 既存コードと同一の AVX2 `_mm256_loadu_pd/storeu_pd` 最適化パターンを踏襲

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

        // ★ [v7.3.3] pushBecameNonEmptyWithWriter: writer ラムダが直接リングスロットに書込
        //   （ゼロコピー・パターン。既存 pushAdaptiveCaptureBlocks と同一アプローチ）
        const PushResult result = ringBuffer_.pushBecameNonEmptyWithWriter(
            [&](AudioBlock& block) noexcept {
                // ★ I4 対応: メタデータ設定は必須。BG Thread が有効サンプル数を知るため。
                //   numSamples=0（ゼロ初期化）のまま push すると、BG Thread はブロックを空と誤認する。
                block.numSamples = currentBlockSize;
                block.sampleRateHz = sampleRateHz_;
                block.bitDepth = bitDepth_;
                block.adaptiveCoeffBankIndex = coeffBankIndex_;
                block.sessionId = sessionId_;

                // AVX2 高速コピー（既存 pushAdaptiveCaptureBlocks L149-167 と同一パターン）
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

        // PushResult で3状態を区別（drop カウントの正確性）
        if (result == PushResult::BecameNonEmpty)
            wakeEvent_.signal();  // RT-safe: 非ブロッキング
        else if (result == PushResult::Full)
            droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
        // AlreadyNonEmpty: リングが既に非空。drain-all 設計により追加通知は不要
    }
}
```

#### Background Thread — drain-all 設計

```cpp
void OutputCaptureSink::run() override
{
    convertBufferL_ = std::make_unique<float[]>(kBlockSize);
    convertBufferR_ = std::make_unique<float[]>(kBlockSize);

    while (!stopRequested_.load(std::memory_order_acquire)) {
        wakeEvent_.wait(-1);              // 通知待ち（無限待機）
        // ★ drainAndWriteBatch() は while(pop()) でリングバッファを
        //   必ず空にする（drain-all 設計）。wait() 復帰後にデータが
        //   残っていた場合も完全排出するため、スプリアスウェイクにも対応。
        drainAndWriteBatch();
    }
    // ★ drainAllAndFinalize(): 最後の shutdown signal との競合で
    //   drainAndWriteBatch() と二重排出になっても while(pop()) なので安全。
    drainAllAndFinalize();
    wavWriter_.reset();
}
```

**パイプライン**:
```
Audio RT: capture() → push + signal
                            ↓ SPSC RingBuffer (wait-free)
BG Thread: run() → wait() → drainAndWriteBatch() → WAV書込
```

**安全性**: drain-all（`while(pop())` で完全排出）により Producer の1回の signal で全データ処理。
WaitableEvent の信号保持に依存しない。

> **[v7.3.1 検証確定]** JUCE 8 の `WaitableEvent` は:
> - デフォルト `manualReset = false`（**auto-reset**）。`OutputCaptureSink` はデフォルト構築を使用する。
> - `signal()` は内部 `std::atomic<bool> triggered` を `true` に設定（非ブロッキング・RT-safe）。
> - `wait()` は `triggered == true` なら即座に `false` にリセットして return（lost wake-up 不可）。
> - これにより「`pushBecameNonEmpty()` で BecameNonEmpty 時のみ `signal()`」＋「`wait()` 復帰後に `while(pop())` で完全排出」の組み合わせが**競合なく安全**であることを確認。
> - 根拠: `JUCE/modules/juce_core/threads/juce_WaitableEvent.h:47-109`

### 2.3 pushBecameNonEmpty() / pushBecameNonEmptyWithWriter() — LockFreeRingBuffer 拡張

> **[v7.3.3 検証確定]**: 既存 `pushWithWriter()`（`LockFreeRingBuffer.h:43-53`）をベースに、
> `PushResult`（3状態）を返す2種類のメソッドを新設する。
> - `pushBecameNonEmpty()`: const T& を受け取って PushResult を返す（コピー版）
> - `pushBecameNonEmptyWithWriter()`: writer ラムダを受け取って PushResult を返す（ゼロコピー版）
>
> **採用方針**: `OutputCaptureSink::capture()` は AudioBlock（4120byte）のコピーを回避するため、
> **`pushBecameNonEmptyWithWriter()` を使用する**（既存 `pushAdaptiveCaptureBlocks` の `pushWithWriter` と同一パターン）。
> Audio Thread のメモリ書き込み量: 4120byte（直接スロット書込）vs 12360byte（ローカル変数+コピー+push）。

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
    std::forward<Writer>(writer)(buffer[w & MASK]);  // ★ 直接スロットに書込
    convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
    return wasEmpty ? PushResult::BecameNonEmpty : PushResult::AlreadyNonEmpty;
}
```

### 2.4 シャットダウンシーケンス（既存 ~MainWindow() との統合）

> **[v7.3.2 検証確定] 既存 `~MainWindow()` の実際の shutdown ステップ**:
> 実コードは以下の10ステップ（`MainWindow.cpp:1000-1033`）:
> ```
> step 1: removeChangeListener(this)
> step 2: setAdaptiveAutosaveCallback({})
> step 3: setCliProcessingTelemetryEnabled(false)
> step 4: setProcessor(nullptr)
> step 5: stopTimer()
> step 6: saveSettings(...)
> step 7: removeAudioCallback(&audioProcessorPlayer)
> step 8: closeAudioDevice()
> step 9: audioEngineProcessor.reset()
> step 10: UI reset (settingsWindow/specAnalyzer/eqPanel/convolverPanel)
> ```
> これに OutputCaptureSink の停止ステップを統合する。`outputCaptureSink_` は
> **AudioEngine に新設する**（atomic ポインタとして `setCaptureSink(ptr)` / `setCaptureSink(nullptr)` 経由で注入）。
> MMCSS shutdown (`mmcssShutdownRequested`) は AudioEngine 内部で管理されるため、
> MainWindow から明示的に指定する必要はない。

```
  1. cliAutomationCallbacksEnabled = false    # ★ 既存 closeButtonPressed() に倣う
  2. audioEngine.removeChangeListener(this)   # ★ 既存 step 1
  3. audioEngine.setCliProcessingTelemetryEnabled(false)  # ★ 既存 step 3
  4. ★ setCaptureSink(nullptr) ★              # Audio Thread の参照を先に絶つ
     #   convo::publishAtomic(audioEngine.outputCaptureSink_, nullptr, release) で atomic store。
     #   Audio Thread は relaxed load で読み、nullptr 検出後は capture() を呼ばない。
  5. captureSink->stopCapturing()             # stopRequested_.store(true, release)
     captureSink->wakeEvent_.signal()          # BG Thread 起床（wait(-1) からの復帰用）
  6. captureSink->waitForThreadToExit(5000)   # BG Thread 終了待機
  7. captureSink.reset()                      # デストラクト（Thread 生存中 destroy 不可につき jassert）

  # ── 以下既存 shutdown に戻る ──
  8. audioProcessorPlayer.setProcessor(nullptr)  # ★ 既存 step 4
  9. stopTimer()                                 # ★ 既存 step 5
 10. saveSettings(...)                           # ★ 既存 step 6
 11. audioDeviceManager.removeAudioCallback(...) # ★ 既存 step 7
 12. audioDeviceManager.closeAudioDevice()       # ★ 既存 step 8
 13. audioEngineProcessor.reset()                # ★ 既存 step 9
 14. UI reset                                    # ★ 既存 step 10
```

> **[v7.3.5] v7.3.2 既存10stepリスト第2項 `setAdaptiveAutosaveCallback({})` の省略** (L15):
> 実際の `~MainWindow()` は `audioEngine.setAdaptiveAutosaveCallback({})` (step 2) を含むが、
> plan の統合11-14step リストでは省略されている。この省略は意図的であり、
> OutputCaptureSink の停止とは関連しないため。
> 整合のため明示し、メモとして残す。

### 2.5 CLIオプション（全30: 既存25 + 新規5）

**既存（25）**: `--cli-run`, `--cli-start-learning`, `--cli-resume-learning`, `--cli-ir`,
`--cli-device-type`, `--cli-buffer-samples`, `--cli-sample-rate-hz`, `--cli-phase`,
`--cli-order`, `--cli-dither-bit-depth`, `--cli-noise-shaper`,
`--cli-post-load-dither-bit-depth`, `--cli-post-load-delay-ms`,
`--cli-ir-reload-count`, `--cli-ir-reload-interval-ms`,
`--cli-bypass-burst-count`, `--cli-bypass-burst-interval-ms`,
`--cli-bypass-burst-value`, `--cli-intent-burst-count`,
`--cli-intent-burst-interval-ms`, `--cli-target-ir-sec`, `--cli-debounce-ms`,
`--cli-f1-hz`, `--cli-f2-hz`, `--cli-exit-ms`

**新規（5）**:
| オプション | 難易度 | 実装内容 |
|-----------|--------|---------|
| `--cli-output-wav` | 低（2h） | WAVファイル出力（OutputCaptureSink経由）|
| `--cli-capture-mode` | 低（1h） | `none`/`pre-filter`/`post-filter`/`post-dither` 選択（CapturePoint enum 4値に対応）|
| `--cli-dump-filter-coeffs` | 低（2h） | OutputFilter const getter追加、JSON出力 |
| `--cli-ir-reload-list` | 中（4h） | カンマ区切りパース→複数IR逐次ロード |
| `--cli-progressive-upgrade` | 低（1h） | `setConvolverEnableProgressiveUpgrade(true)` |

> **[v7.3.5 検証確定] `--cli-output-wav` / `--cli-capture-mode` の MainWindow.cpp 実装方針** (L16):
> これら2つの新規CLIは `runCommandLineAutomation()` (`MainWindow.cpp:331`) 内で
> 既存 `findValue()` パターンに倣って処理する。実装コード例:
>
> ```cpp
> // MainWindow.cpp runCommandLineAutomation() に追加
> // ★ [v7.3.5] --cli-output-wav: WAV出力パス指定
> const auto outputWavPath = findValue("--cli-output-wav");
> if (!outputWavPath.isEmpty()) {
>     auto captureSink = std::make_unique<OutputCaptureSink>();
>     captureSink->setOutputPath(juce::File(outputWavPath));
>     // ★ --cli-capture-mode: CapturePoint enum への変換
>     const auto captureModeStr = findValue("--cli-capture-mode");
>     CapturePoint cp = CapturePoint::PostOutputFilter; // default
>     if (captureModeStr.equalsIgnoreCase("none"))        cp = CapturePoint::None;
>     else if (captureModeStr.equalsIgnoreCase("pre-filter"))  cp = CapturePoint::PreOutputFilter;
>     else if (captureModeStr.equalsIgnoreCase("post-filter"))  cp = CapturePoint::PostOutputFilter;
>     else if (captureModeStr.equalsIgnoreCase("post-dither")) cp = CapturePoint::PostDither;
>     captureSink->setCapturePoint(cp);
>     captureSink->startCapturing();
>     // AudioEngine に atomic ポインタとして注入
>     audioEngine.setCaptureSink(captureSink.get());
>     // ★ cliCaptureSink_ メンバ変数（MainWindow.h に std::unique_ptr<OutputCaptureSink> として新設）
>     //   でライフサイクル管理。closeButtonPressed()/~MainWindow() で停止・破棄。
>     cliCaptureSink_ = std::move(captureSink);  // MainWindow.h に追加
> }
> ```
>
> **`hasAutomationFlags` の条件拡張**:
> `--cli-output-wav` と `--cli-capture-mode` は単独では automation flag を立てない
> （`--cli-run` または他の automation flag と併用必須）。
> これにより既存の `hasAutomationFlags` の if式 には追加しない。
> ただし `--cli-run` が指定されている場合にのみ `findValue("--cli-output-wav")` を評価する。
> これはテストモードでのみ有効とする意図（単独使用で ConvoPeq が無音化するのを防止）。

> **[v7.3.5 検証確定] 既存25 CLI の正確なカウント** (L16 関連):
> `runCommandLineAutomation()` (MainWindow.cpp:331-960) で `findValue()` パターン処理される
> オプションは正確に**25種類**（プラン通り）。`hasFlag()` で処理される3種
> (`--cli-run`, `--cli-start-learning`, `--cli-resume-learning`) は findValue と
> 重複せず、合計で **28種のCLIエントリ** が存在する。
> ただし §2.5 の「25」は findValue 系のみカウントしたものであり、
> hasFlag 系3種を含めると「28+5=33」だが、慣例として findValue 系でカウントするため
> 「25+5=30」の表記を維持する（v7.3.3 J4 の6新規CLIパラメータは Phase 1 で追加）。

---

## 3. Seqlock / RecoveryHistory 設計

### 3.1 RecoveryEvent 構造体

```cpp
struct RecoveryEvent {
    uint64_t        timestampUs;      // 記録時刻
    PolicySource    source;           // 発火元監視器（10種）
    RecoveryAction  action;           // 実行アクション（6段階）
    ISRHealthState  healthState;      // 実行時 HealthState
    uint64_t        pendingRetire;    // 実行時 Retire backlog
    uint64_t        maxRetireAgeUs;   // 実行時 最大 Retire age
    RecoveryOutcome     outcome;      // 閉ループ制御結果
    VerificationState   verification; // 検証状態
    uint8_t             stalledCount; // 停滞カウント
    uint64_t        eventSequence;    // 単調増加通し番号（欠落検出可能）
};
```

### 3.2 Sequence Lock（Seqlock）

> **[v7.3.1 検証確定] single writer 設計契約**:
> `recordRecoveryAction()` は **Timer Thread 単一**から呼び出すことを設計契約とする。
> 既存の `executeRecoveryAction()`（`AudioEngine.Timer.cpp:1591`）は Timer Thread から呼ばれる
> （`RuntimeHealthMonitor` → callback 経由）。`recordRecoveryAction()` をこの関数に組み込むことで
> single writer が保証され、slot 取得の CAS は不要となる。
> **Audio Thread やその他スレッドから直接呼び出すことは禁止する。**
> この契約に違反する場合は `fetchAdd` を CAS ループに変更すること。

```cpp
struct RecoverySlot {
    RecoveryEvent event;
    std::atomic<uint64_t> generation{0};  // even=完成, odd=書込中
};
RecoverySlot m_recoveryHistory[kRecoveryHistoryCapacity];  // 64件
std::atomic<uint64_t> m_recoveryHistoryWriteIndex{0};  // ★ v7.3: uint64_t（オーバーフロー防止、eventSequence と統一）
std::atomic<uint64_t> m_eventSequenceCounter{0};  // 単調増加
```

#### Writer（偶数→奇数→偶数）

```cpp
void recordRecoveryAction(PolicySource source, RecoveryAction action,
                          ISRHealthState healthState) noexcept
{
    const uint64_t idx = convo::fetchAddAtomic(m_recoveryHistoryWriteIndex, static_cast<uint64_t>(1), std::memory_order_relaxed) % 64;    // ★ single writer 前提のため CAS 不要
    const uint64_t seq = convo::fetchAddAtomic(m_eventSequenceCounter, static_cast<uint64_t>(1), std::memory_order_relaxed) + 1u;

    auto& slot = m_recoveryHistory[idx];
    const uint64_t oldGen = slot.generation.load(relaxed);

    slot.generation.store(oldGen + 1, relaxed);     // Lock: odd（書込中）
    slot.event = RecoveryEvent{ /* ... seq, ... */ };
    slot.generation.store(oldGen + 2, release);     // Unlock: even + 1（書込完了）
}
```

#### Reader（acquire → copy → acquire + 8回リトライ）

```cpp
std::vector<RecoveryEvent> copyRecoveryHistorySnapshot() const noexcept
{
    const uint64_t writeIdx = consume(m_recoveryHistoryWriteIndex, acquire);
    const uint64_t count = std::min<uint64_t>(64u, writeIdx);
    const uint64_t start = (writeIdx >= count) ? (writeIdx - count) % 64 : 0;

    std::vector<RecoveryEvent> snapshot;
    snapshot.reserve(count);

    for (uint32_t i = 0; i < count; ++i) {
        const auto& slot = m_recoveryHistory[(start + i) % 64];
        bool copied = false;
        for (int retry = 0; retry < 8; ++retry) {
            const uint64_t genBefore = slot.generation.load(acquire);
            if ((genBefore & 1) != 0) continue;  // 奇数 → リトライ

            RecoveryEvent copy = slot.event;       // ← acquire で可視性保証
            const uint64_t genAfter = slot.generation.load(acquire);

            if (genAfter == genBefore) {           // 整合性確認
                snapshot.push_back(copy);
                copied = true;
                break;
            }
        }
        if (!copied) {
            // 8回失敗 → Writer 継続更新中と判断し、当該 slot を破棄
            // 3→8 に変更: Reader は BG Thread で動作し RT に影響しないため粘れる。
            // それでも失敗する場合は Writer（Timer Thread）が高頻度で更新中。
            seqlockRetryFailed_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    return snapshot;
}
```

---

## 4. AudioBlock 保証

```cpp
static_assert(std::is_trivially_copyable_v<AudioBlock>,
    "AudioBlock must be trivially copyable for LockFreeRingBuffer");
static_assert(std::is_standard_layout_v<AudioBlock>,
    "AudioBlock must be standard layout");
static_assert(std::is_trivially_destructible_v<AudioBlock>,
    "AudioBlock must be trivially destructible");
    // ★ 実装時: 現状の DSPCore は _mm256_loadu_pd（アンアラインドロード）を使用。
    //   将来的に _mm256_load_pd（アラインド要求）を導入する場合のみ
    //   alignof(AudioBlock) >= 32 の static_assert を追加すること。
```

> **[v7.3.1 検証確定] AudioBlock 型特性 実測確認（g++ -std=c++17）**:
>
> | 特性 | 結果 | 備考 |
> |------|------|------|
> | `is_trivially_copyable_v` | **true** | LockFreeRingBuffer 要求を満たす ✅ |
> | `is_standard_layout_v` | **true** | memcpy 安全性を満たす ✅ |
> | `is_trivially_destructible_v` | **true** | ✅ |
> | `is_trivial_v` | false | メンバにデフォルト初期化子(`= 0`)があるため。**static_assert 対象外につき問題なし** |
> | `sizeof(AudioBlock)` | **4120 byte** | double[256]×2 + int×4 + uint64_t + padding(4byte) |
> | `alignof(AudioBlock)` | **8** | double のアラインメントに従う |
>
> - レイアウト: `L[0..2047]` → `R[0..2047]` → `numSamples[4096]` → `sampleRateHz/bitDepth/adaptiveCoeffBankIndex[4100-4108]` → `sessionId[4112]` → padding[4116-4119]
> - 上記3個の static_assert は全てパスする。`is_trivial_v` が false でも `DiagEvent` と同様に問題ない（`DiagEvent` は trivial だが、AudioBlock は trivially_copyable のみ要求）。
> - **注意**: `sizeof(AudioBlock) == 4120` の固定値に依存するコードを書かないこと。メンバ追加時に変化する。

---

## 5. テストパラメータパターンと自動実行設計

### 5.1 パラメータ次元（全10次元）

| 次元 | CLIオプション／API | 値 | 既定値 |
|------|-------------------|----|--------|
| 処理順序 | `--cli-order` | `conv` / `peq` / `conv->peq` / `peq->conv` | `conv->peq` |

> **[v7.3.5 明確化] `--cli-order` の CLI値 対 実動作 マッピング** (L9):
> `parseCliOrderMode()` (MainWindow.cpp:60-89) の実装に基づく:
> | CLI値 | ComboBox ID | orderModeBoxChanged() のロジック | 実動作 |
> |--------|-------------|------|------|
> | `conv` | 1 | `setConvolverBypass(false)` + `setEqBypass(true)` | **Convolver-Only** (PEQはbypass) |
> | `peq` | 2 | `setConvolverBypass(true)` + `setEqBypass(false)` | **PEQ-Only** (Convolverはbypass) |
> | `conv->peq` (民主的別名: `convpeq`) | 3 | `setProcessingOrder(ConvolverThenEQ)` + 両方bypass=false | **Convolver→PEQ 順処理** (ProcessingOrder=0) |
> | `peq->conv` (別名: `peqconv`) | 4 | `setProcessingOrder(EQThenConvolver)` + 両方bypass=false | **PEQ→Convolver 順処理** (ProcessingOrder=1) |
>
> 注意: ComboBox ID (1/2/3/4) ≠ ProcessingOrder enum (0/1)。ID=1/2 は単独動作モード。
> ID=3/4 のみ ProcessingOrder enum にマップされる。
> `conv` という値名は「Convolver動作」を示すが、PEQ を bypass することを暗黙に意味する。
> プラン内では `--cli-order` の値を ComboBox ID ではなく **文字列 label** で扱う。
| 位相モード | `--cli-phase` | `asis` / `mixed` / `minimum` | `asis` |
| OSタイプ | — | `iir` / `linear-phase` | `iir` |
| OS倍率 | — | `1x` / `2x` / `4x` / `8x` | `1x` |
| ノイズシェイパー | `--cli-noise-shaper` | `psychoacoustic` / `fixed4` / `adaptive9` / `fixed15` | `psychoacoustic` |
| ディザー深度 | `--cli-dither-bit-depth` | `16` / `24` / `32` | `32` |
| HCモード | — | `sharp` / `natural` / `soft` | `natural` |
| LCモード | — | `natural` / `soft` | `natural` |
| ソフトクリップ | — | `on` / `off` | `off` |
| サチュレーション | — | `0.0`〜`1.0` | `0.0` |

> **[v7.3.2 検証確定] CLI 未提供パラメータの実装課題**:
> OSタイプ/OS倍率/HCモード/LCモード/SoftClip/Saturation の6パラメータの API は
> 全て実在する（`setOversamplingType`/`setOversamplingFactor`/`setConvHCFilterMode`/
> `setConvLCFilterMode`/`setSoftClipEnabled`/`setSaturationAmount`）。
> しかし**CLIオプションとして提供されていない**ため、`cli_runner.py` の
> `build_cli_args()` から直接渡すことはできない。Phase 0 ではなく **Phase 1 で
> 以下の新規CLIオプションを実装する必要がある**:
> | CLIオプション | 対応API | Phase |
> |---------------|---------|-------|
> | `--cli-os-type` | `setOversamplingType(OversamplingType)` | 1 |
> | `--cli-os-factor` | `setOversamplingFactor(int)` | 1 |
> | `--cli-hc-mode` | `setConvHCFilterMode(convo::HCMode)` | 1 |
> | `--cli-lc-mode` | `setConvLCFilterMode(convo::LCMode)` | 1 |
> | `--cli-soft-clip` | `setSoftClipEnabled(bool)` | 1 |
> | `--cli-saturation` | `setSaturationAmount(float)` | 1 |
> これにより Phase 1 工数に +3人日の追加影響あり（各オプション0.5日×6）。

### 5.2 10測定パターン（★ v7.3.2: 見出し「7」→「10」に修正、P8/P9/P10 は元から表に存在）

| パターン | 目的 | 主要パラメータ | テストケース |
|---------|------|--------------|-------------|
| **P1-Baseline** | 基準測定 | 全既定値 | TC-01,03,04,04A,06,23,24,29A,29B,31a,31b,32,39,41（★ v7.3.2: §5.8 の追加TC-31,32,39,41 を反映）|
| **P2-PEQ-Only** | PEQ単体品質 | `order=peq`, convBypass | TC-01,01B,03,04,07,32,36,37（★ v7.3.2: §5.8 の追加TC-32,36,37 を反映）|
| **P3-ConvoThenPEQ** | 既定順序品質 | `order=conv->peq`, `phase=mixed` | TC-01,02,09,21,23,33,35,39（★ v7.3.2: §5.8 の追加TC-33,35,39 を反映）|
| **P4-PEQThenConvo** | 逆順序応答一致 | `order=peq->conv`, `phase=mixed` | TC-01,15 |
| **P5-ReferenceQuality** | 最高忠実度 | OS8x, Adaptive9, **SoftClip OFF, Sat=0** | TC-01,03,04,04A,23,24,**31a,33,34,39,41** |
| **P6-Musical** | 非線形音質 | OS8x, Adaptive9, **SoftClip ON, Sat=0.3** | TC-01,03,04,04A,23,24,**31b,39,40** |
| **P7-OSx2/4/8** | OS倍率スイープ | OS=2x/4x/8x | TC-01,09,24,**33,39** |
| **P8-MinimumPhase** | 最小位相品質 | `phase=minimum` | TC-01,02,**32,35,41** |
| **P9-AsIsPhase** | As Is位相品質 | `phase=asis` | TC-01,02,**32,35** |
| **P10-MixedPhase** | Mixed Phase重点 | `phase=mixed` | TC-01,02,**21,32,35,41** |

> **[v7.3.6 M5] 既存テストケース16件が新パターンに未割り当て**: TC-05A〜D（低域ノイズ拡充: 4件）, TC-08（モード切替応答）, TC-10（バイパス応答）, TC-11/11B（IR リロード品質: ISR  stressed）, TC-12/13/14（burst/intent 応答: ISR stressed）, TC-16（Progressive Upgrade 品質）, TC-19/20（Auto Gain 応答: Runtime）、TC-22（Speaker Comp 応答: Runtime）、TC-26（Telemetry Publication Latency）、TC-28（OutputFilter 検証）は §5.2 のP1〜P10 パターンに割り当てられていない。これらは既存 CI テストフレームワーク（`src/tests/*.cpp`）で継続検証する。新 CLI 自動化（Phase 1〜2）では対象としない（手動確認ステップで除外項目としてリスト記載）。

### 5.3 自動実行設計

```yaml
# test_config.yaml — 測定パターン定義ファイル
patterns:
  P1-Baseline:
    order: conv->peq; phase: asis; os: 1x
    noiseShaper: psychoacoustic; ditherBitDepth: 32
    hcMode: natural; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac]
    testCases: [TC-01, TC-03, TC-04, TC-04A, TC-06, TC-23, TC-24, TC-29A, TC-29B, TC-31a, TC-31b, TC-32, TC-39, TC-41]

  P2-PEQ-Only:
    order: peq; convBypass: true; phase: asis; os: 1x
    noiseShaper: psychoacoustic; ditherBitDepth: 32
    hcMode: natural; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac, room_correction]
    testCases: [TC-01, TC-01B, TC-03, TC-04]

  P3-ConvoThenPEQ:
    order: conv->peq; phase: mixed; os: [1x, 2x, 4x, 8x]
    noiseShaper: psychoacoustic; ditherBitDepth: 32
    hcMode: natural; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac, lpf_1k]
    testCases: [TC-01, TC-02, TC-09, TC-21]

  P4-PEQThenConvo:
    order: peq->conv; phase: mixed; os: 1x
    noiseShaper: psychoacoustic; ditherBitDepth: 32
    hcMode: natural; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac]
    testCases: [TC-01, TC-15]

  P5-ReferenceQuality:
    order: conv->peq; phase: mixed; os: 8x
    noiseShaper: adaptive9; ditherBitDepth: 32
    hcMode: sharp; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac, room_correction]
    testCases: [TC-01, TC-03, TC-04, TC-04A, TC-23, TC-24, TC-31a, TC-33, TC-34, TC-39, TC-41]

  P6-Musical:
    order: conv->peq; phase: mixed; os: 8x
    noiseShaper: adaptive9; ditherBitDepth: 16
    hcMode: sharp; lcMode: natural; softClip: true; saturation: 0.3
    srcIrs: [dirac, room_correction]
    testCases: [TC-01, TC-03, TC-04, TC-04A, TC-23, TC-24, TC-31b, TC-39, TC-40]

thresholds:
  debug_msvc:  { thd: -80, imd: -80, noiseFloor: -115, freqResponse: 0.05 }
  release_msvc: { thd: -100, imd: -90, noiseFloor: -120, freqResponse: 0.05 }
  release_icx:  { thd: -100, imd: -90, noiseFloor: -120, freqResponse: 0.05 }
```

#### `generators.py` — IR ファイル論理名マッピング

```python
# generators.py — IR 論理名→実ファイルの解決
# phase 1-1 で実装

IR_FILE_MAP = {
    "dirac":           "generated/dirac_test.wav",         # tools/diagnostics/create_dirac_ir.py で生成
    "room_correction": "sampledata/impulse_room_correction_hpf_lpf.wav",  # REW+rePhase 実測IR
    "lpf_1k":          "generated/lpf_1k_test.wav",        # tools/diagnostics/create_test_irs.py で生成
}

def generate_ir(ir_name: str) -> Path:
    """IR 論理名から実ファイルパスを返す。未生成の場合はその場で生成する。"""
    if ir_name in ("dirac", "lpf_1k"):
        return generate_synthetic_ir(ir_name)  # 合成IR生成
    return Path(IR_FILE_MAP[ir_name])  # 既存ファイルを参照
```

#### `cli_runner.py` パターン実行フロー

```python
def build_cli_args(pattern, ir_file, input_wav, output_dir):
    """10次元パラメータ → CLI引数リスト
    ★ [v7.3.3] §5.1 の6新規CLIパラメータ（OS/HC/LC/SoftClip/Saturation）を追加。
    これらは Phase 1 で新規実装される CLI オプション（H1/I8 対応）。"""
    args = [f"--cli-ir={ir_file}", f"--cli-output-wav={output_dir/'output.wav'}",
            "--cli-capture-mode=post-dither", "--cli-exit-ms=5000"]
    args.append(order_map[pattern["order"]])
    args.append(phase_map[pattern["phase"]])
    args.append(ns_map[pattern["noiseShaper"]])
    if pattern.get("ditherBitDepth", 32) < 32:
        args.append(f"--cli-dither-bit-depth={pattern['ditherBitDepth']}")
    if pattern.get("convBypass"):
        args += ["--cli-bypass-burst-count=1", "--cli-bypass-burst-value=1"]

    # ★ [v7.3.3] 6新規CLIパラメータ（Phase 1 実装後）
    os_type_map = {"iir": "iir", "linear-phase": "linear-phase"}
    os_factor_map = {"1x": 1, "2x": 2, "4x": 4, "8x": 8}
    hc_mode_map = {"sharp": "sharp", "natural": "natural", "soft": "soft"}
    lc_mode_map = {"natural": "natural", "soft": "soft"}

    if "osType" in pattern:
        args.append(f"--cli-os-type={os_type_map[pattern['osType']]}")
    if "os" in pattern:
        # os は list（スイープ）または単一値
        os_val = pattern["os"] if isinstance(pattern["os"], str) else pattern["os"][0]
        args.append(f"--cli-os-factor={os_factor_map[os_val]}")
    if "hcMode" in pattern:
        args.append(f"--cli-hc-mode={hc_mode_map[pattern['hcMode']]}")
    if "lcMode" in pattern:
        args.append(f"--cli-lc-mode={lc_mode_map[pattern['lcMode']]}")
    if pattern.get("softClip", False):
        args.append("--cli-soft-clip=on")
    if pattern.get("saturation", 0.0) > 0.0:
        args.append(f"--cli-saturation={pattern['saturation']}")

    return args

def run_pattern(name, pattern, output_dir):
    for ir_name in pattern["srcIrs"]:
        for tc in pattern["testCases"]:
            ir_file = generate_ir(ir_name)
            input_wav = generate_input(tc)
            output_wav = run_convopeq(build_cli_args(pattern, ir_file, input_wav, output_dir))
            store_result(name, tc, pattern, analyze(tc, input_wav, output_wav, ir_file))
```

### 5.4 CI実行ポリシー

| 契機 | 実行パターン | 対象テスト | 制限時間 |
|------|------------|-----------|---------|
| PR（Quick） | P1, P2 | TC-01,03,04,04A,31,32 | 5分 |
| PR（Full） | P1〜P6 | 全41TC | 30分 |
| Nightly | P1〜P10＋全OS＋IR Reload＋Bypass Burst | 全41TC全パターン＋TC-38 | 3時間 |
| リリース | P1〜P7 × 3build | 全41TC × MSVC/icx Debug/Release | 4時間 |

### 5.5 Phase 1 追加タスク

| # | タスク | 成果物 |
|---|-------|--------|
| 1-8 | パターン定義ファイル作成 | `test_config.yaml` |
| 1-9 | CLI引数ビルダー | `cli_runner.py` `build_cli_args()` |
| 1-10 | パターンループ実行エンジン | `cli_runner.py` `run_all_patterns()` |
| 1-11 | OS倍率スイープ | `cli_runner.py` `os_sweep()` |

### 5.6 ゴールデン比較フレームワーク

```python
# golden_calculator.py — 期待値ベース・ゴールデン比較の統一管理（Phase 1-4 で実装）

from typing import Protocol
from pathlib import Path
import numpy as np

class GoldenReference(Protocol):
    """テストケースごとの期待値生成インターフェース"""
    def generate_input(self, sample_rate: int, duration_sec: float) -> np.ndarray: ...
    def expected_output(self, ir_data: np.ndarray, input_data: np.ndarray) -> np.ndarray: ...
    def metrics(self, actual: np.ndarray, expected: np.ndarray) -> dict: ...

# ── 測定対象と比較方式の体系 ─────────────────────────────
# | 対象           | 期待値の種類 | 比較方式                         | 該当TC        |
# |---------------|------------|----------------------------------|---------------|
# | Dirac         | 理論値       | 周波数領域: 振幅±0.05dB/位相±1°  | TC-01,02      |
# | LPF/HPF/Allpass | 理論伝達関数 | 振幅誤差(RMS) / 群遅延誤差        | TC-07,21      |
# | Convolver     | Golden WAV  | ビット単位比較 / Null Test        | TC-27         |
# | PEQ           | 理論係数     | 振幅スペクトル誤差(RMS)            | TC-07         |
# | Noise Shaper  | 統計特性     | PSD曲線 / ヒストグラム分布          | TC-04,04A     |
# | THD           | 理論=0      | 残留高調波パワー / 基本波比         | TC-03,17,18,24|
# | Null / Bypass | 完全一致     | 差分RMS / Peak / サンプル一致率    | TC-31(新)     |
# | Alias         | 理論=0      | 折り返し帯域エネルギー              | TC-09,33(新)  |
```

| 比較方式 | 実装方法 | 検出可能な異常 |
|---------|---------|--------------|
| **理論値比較** | 数学的モデルから期待値を計算（IIRフィルタ伝達関数等）| 係数計算誤差 / フィルタ構造誤差 |
| **Golden WAV比較** | リファレンス実装で事前生成したWAVと逐次比較 | 回帰障害 / コンパイラ最適化差 |
| **Null Test** | 入力をそのまま出力側にバイパスし差分を評価 | ビットパーフェクト破壊 / 不要処理混入 |
| **統計評価** | ノイズフロアのPSD/ヒストグラムを理論分布と比較 | ノイズシェイパー異常 / ディザー破綻 |
| **自己無撞着性** | 同一IRの複数回処理で相互比較 | 決定論的破綻 / 状態管理漏れ |

### 5.7 追加テストケース（TC-31〜TC-41）

現状の34TCに加えて、以下のテストを追加することで測定対象の網羅性を向上させる。

| # | テストケース | カテゴリ | 入力信号 | 評価方法 | 優先度 |
|---|-------------|---------|---------|---------|--------|
| **TC-31a** | **Null Test（Float Pipeline）** | パイプライン完全性 | Dirac -6dBFS, 3秒 | 差分RMS ≤ **-138dBFS** / Peak ≤ **-128dBFS**（FMA+DCBlocker+Dither除去不可） | ★★★★★ |
| **TC-31b** | **Null Test（Double Pipeline）** | パイプライン完全性 | 1kHz正弦波 -6dBFS, 3秒 | 差分RMS ≤ **-138dBFS** / Peak ≤ **-128dBFS**（DSPCoreDouble path は TruePeak/LUFS/Limiter を含むため bit-exact 不可） | ★★★★★ |

> **[v7.3.5 検証確定] TC-31a Bit-Exact 閾値緩和** (L8):
> ConvoPeq のオーディオパイプラインは **全 bypass 時でも**以下が常時動作する:
> - `DCBlockers` (`DSPCoreIO.cpp:377`): 状態変数を持つ IIR HPF。ゼロ入力でも過去状態で非ゼロ出力。
> - **NaN/Inf 除去第1段** (`DSPCoreIO.cpp:380-411`): NaN または Inf 検出時に 0.0 に置換。
> - **`pushAdaptiveCaptureBlocks()`** (`DSPCoreIO.cpp:413-421`): 条件付き（adaptiveCaptureEnabled時のみ）
> - **Dither** (`DSPCoreIO.cpp:463-478`): dither bit depth > 0 時、常時 TPDF noise 注入
> - **NaN/Inf 除去第2段** (`DSPCoreIO.cpp:482-500`): 第1段と同様
> - **`applyFixedLatencyDelay()`** (`DSPCoreIO.cpp:503`): 遅延バッファのシフトによる境界サンプル変化
> - **float 変換+hard clamp** (`DSPCoreIO.cpp:505-510`): `static_cast<float>` による丸め誤差
>
> したがって **100%サンプル一致（bit-exact）は不可能**。DC Blocker は 2次 HPF (fc=1Hz 程度) で
> 理論上 入力が DC の場合のみ完全一致するが、Dirac 入力のような非 DC 信号では残留出力が発生する。
> **新閾値**: RMS ≤ -138dBFS（DC Blocker + Dither + float 丸めの最悪ケース結合）。
> Double Path (processOutputDouble) は TruePeak/LUFS/PeakLimiter も常時動作するため、
> さらに残留が増加し、-128dBFS Peak を許容する。

> **[v7.3.5 検証確定] processOutputDouble PostDither 位置** (L11):
> v7.3.4 で記述した「L782後・L784前」は誤り。実コード (DSPCoreDouble.cpp:620-810) の実ラインは:
> - L782: `applyFixedLatencyDelay(dataL, dataR, numSamples)`
> - L803-805: `juce::FloatVectorOperations::copy(buffer.getWritePointer(0,0), dataL, numSamples)`
>   （注: Double Path では buffer は `juce::AudioBuffer<double>&`。float 変換なしで double 精度のままコピー）
> - PostDither キャプチャは **L803-L805 後**（Hard Clamp 通過後、buffer コピー直後）に挿入。
> - データ型: Double Path では `double*`（float 変換なし）、Float Path では `float*`。
| **TC-32** | **Log Sweep 伝達関数** | 周波数特性 | 20Hz-24kHz対数スイープ 10秒 | FFT→Transfer Function: 振幅±0.05dB / 群遅延±0.1sample | ★★★★★ |
| **TC-33** | **Alias Rejection** | エイリアシング | 19kHz+20kHz+22kHz 各-6dBFS 3秒 | OS=2x/4x/8x で理論イメージ周波数位置（k・fs±fin）のエネルギー ≤ -130dBFS。★ [v7.3.3] 入力は全て Nyquist(24kHz)未満のため alias 発生なし=反証テスト。「10kHz以下」→「理論イメージ位置」に評価方法を統合 | ★★★★★ |
| **TC-34** | **PSD（Noise Shaper）** | ノイズ | 無音 5秒 | Welch PSD：A-weighting 20Hz-20kHz ≤ -110dBFS / 形状一致 | ★★★★☆ |

> **[v7.3.1 検証確定] TC-34 閾値 -110dBFS A-weighted の到達性**:
> - **32bit設定（P5-ReferenceQuality）では到達可能**: `PsychoacousticDither` は32bit設定で「控えめ」ディザ（`PsychoacousticDither.h:13`）。
>   32bit float パイプラインのノイズフロアは理論上 -144dBFS 程度。A-weighting で更に低減するため、-110dBFS は十分に達成圏内。
> - **16bit設定（P6-Musical）では厳しい**: 16bit 量子化ノイズ ≈ -96dBFS がベース。ノイズシェーピングで可聴帯域外に追い出しても A-weighted -110dBFS は困難。
> - **対策**: TC-34 は P5-ReferenceQuality (ditherBitDepth=32) のみに割り当て（§5.8参照）。P6-Musical には割り当てないこと。
> - 32bit 実測で -110dBFS を下回る場合は、閾値を -105dBFS に緩和する余地を残す（実装後に決定）。
| **TC-35** | **Group Delay（Mixed Phase）** | 位相 | 20Hz-20kHz対数スイープ 10秒 | Mixed Phase IRの群遅延曲線 vs 理論値 ±5% | ★★★★☆ |
| **TC-36** | **Stereo Crosstalk** | チャネル分離 | L=1kHz正弦波 R=無音 3秒 | R出力の漏れエネルギー ≤ -140dBFS | ★★★★☆ |
| **TC-17** | **SMPTE IMD** | 歪み | 60Hz+7kHz 4:1 比率（IEC 60268-3）3秒 | IMD ≤ -80dB(Debug) / -90dB(Release) | ★★★★☆ |
| **TC-18** | **CCIF IMD** | 歪み | 19kHz+20kHz 1:1 比率 3秒 | IMD ≤ -80dB(Debug) / -90dB(Release) | ★★★★☆ |
| **TC-37** | **Numerical Transparency（32bit）** | 数値精度 | 1kHz正弦波 -0.1dBFS 3秒 | RMS誤差 ≤ **1 ULP** / Peak ≤ **2 ULP**（IEEE754単精度/コンパイラ最適化差許容）| ★★★★☆ |
> **[v7.3.6 M11] TC-37 Double Path 除外条件**: TC-37 の入力振幅 `10^(-0.1/20) ≈ 0.9886` は Double Path の PeakLimiter 閾値 `kOutputHeadroom=0.891` (`DSPCoreDouble.cpp:L763`) を超える。Double Path (processOutputDouble) では PeakLimiter が常に動作域に入り、出力が制限を受けて歪む。**TC-37 は Float Path (processOutput) でのみ測定可能**。Double Precision 設定のテスト環境では TC-37 をスキップすること（または Float Path フォールバックを使用）。P2-PEQ-Only は OS=1x+Float Path のため問題なし。
| **TC-38** | **Long-run Stability（30分）** | 長期安定 | 40Hz正弦波 -6dBFS 30分 | NaN/Inf/DC異常なし / ドリフト ≤ 0.01dB | ★★★★☆ |
| **TC-39** | **High-Freq THD Sweep** | 歪み | 100/1k/5k/10k/18k/20kHz 各-6dBFS 3秒 | THD+N 周波数特性グラフ、閾値 ≤ -80dB(Debug) / -100dB(Release) | ★★★★★ |
| **TC-40** | **Dither Histogram** | ディザー | 無音 10秒（各dither設定） | 量子化誤差ヒストグラム：理論PDFとのKL Divergence ≤ 0.01 | ★★★☆☆ |
| **TC-41** | **Impulse Alignment** | 遅延 | Dirac -6dBFS 1秒 | 出力ピーク位置 vs 理論遅延 ±1sample / Energy 95%区間 | ★★★★★ |

**合計: 全41テストケース**（既存34 + 新規11）

### 5.8 新テストケースのパターン割り当て

| パターン | 追加テストケース | 目的 |
|---------|----------------|------|
| **P1-Baseline** | +TC-31, TC-32, TC-39, TC-41 | 基準状態の完全性＋詳細特性 |
| **P2-PEQ-Only** | +TC-32, TC-36, TC-37 | PEQ単体の伝達関数＋分離度 |
| **P3-ConvoThenPEQ** | +TC-33, TC-35, TC-39 | Mixed Phase品質＋Alias評価 |
| **P5-ReferenceQuality** | +TC-31a,33,34,39,41 | 最高忠実度設定の全性能 |
| **P6-Musical** | +TC-31b,39,40 | 非線形音質の評価 |
| **P8-MinimumPhase** | TC-01,02,32,35,41 | 最小位相特化 |
| **P9-AsIsPhase** | TC-01,02,32,35 | As Is位相特化 |
| **P10-MixedPhase** | TC-01,02,21,32,35,41 | Mixed Phase位相特化 |
| **P7-OSx2/4/8** | +TC-33（主評価）, TC-39 | OS倍率とAlias Rejectionの関係 |
| **Nightly** | +TC-38, TC-40 | 長期安定性＋統計評価 |

### 5.9 `analyzers.py` 拡張設計

```python
# analyzers.py — Phase 1-2 で実装。既存 tools/diagnostics/*.py の関数を活用

import numpy as np
from scipy import signal
from typing import Tuple

# ── 既存ツールの活用 ──
# tools/diagnostics/compare_raw.py       → Null Test の差分計算に流用
# tools/diagnostics/compare_dirac.py     → ブロック境界・DC分析に流用
# tools/diagnostics/analyze_ir.py        → WAVヘッダ・チャンク解析に流用
# tools/diagnostics/analyze_conv_output.py → Tone周波数・振幅検出に流用


class Analyzers:
    """全テストケースの分析ロジック。各 analyze_* が (metrics: dict, pass: bool) を返す"""

    @staticmethod
    def analyze_dirac(input_wav: np.ndarray, output_wav: np.ndarray, sr: int) -> dict:
        """TC-01/02: Dirac応答 → 振幅スペクトルRMS誤差"""
        _, _, expected = theoretical_dirac_response(len(output_wav), sr)
        return compare_frequency_response(output_wav, expected, sr, band=(20, 20000))

    @staticmethod
    def analyze_sweep_transfer(input_wav: np.ndarray, output_wav: np.ndarray, sr: int) -> dict:
        """TC-32/35: Log Sweep → 伝達関数（振幅+位相+群遅延）
        ★ [v7.3.5] Farina 2007 "Simultaneous measurement of impulse response and distortion
        with a swept-sine technique" (AES 122nd Convention) に準拠:
        - 対数スイープ信号 s(t) とその逆フィルタ s⁻¹(t) を使用
        - インパルス応答 h(t) = s⁻¹(t) * y(t) （y(t)は出力信号、*は畳み込み）
        - 伝達関数 H(f) = FFT{h(t)}
        - 逆フィルタ s⁻¹(t) は時間反転 + 振幅補償（-6dB/oct エンベロープ補正）
        ★ 旧方式(v7.3.4): H = FFT(output)/FFT(input) は log sweep で誤差大（Farinaで破綻）。
        本方式では §8.2 の generate_log_sweep() で同時に逆フィルタ信号を生成し、
        analyzers.py で畳み込み復調する。

        数式:
          s(t) = sin(φ(t)),  φ(t) = 2π·f₀·T / ln(f₁/f₀) · (exp(t·ln(f₁/f₀)/T) - 1)
          s⁻¹(t) = s(T-t) · exp(-t · ln(f₁/f₀) / T)   ← 時間反転 + 振幅補償（-6dB/oct エンベロープ）
          畳み込み: h_raw = signal.fftconvolve(output_wav, s_inv, mode='full')
        """
        # ★ v7.3.5: Farina 2007 準拠 inverse filter 方式
        from scipy import signal as scipy_signal
        T = len(input_wav) / sr  # sweep 継続時間
        # 逆フィルタ: 時間反転 + 振幅補償（-6dB/oct エンベロープ = exp(-t·ln(f₁/f₀)/T)）
        t_full = np.arange(len(input_wav)) / sr
        amplitude_compensation = np.exp(-t_full * np.log(24000.0 / 20.0) / T)
        s_inv = input_wav[::-1] * amplitude_compensation  # s(T-t) * envelop補償
        # 畳み込みでインパルス応答抽出
        h_raw = scipy_signal.fftconvolve(output_wav, s_inv, mode='full')
        # 高調波歪成分分離: メイン IR (因果部分) のみ抽出
        ir_start = np.argmax(np.abs(h_raw))
        h = h_raw[ir_start:ir_start + len(input_wav)]  # T秒分の IR
        # FFTで伝達関数
        N = len(h)
        H = np.fft.rfft(h, n=N * 4)
        f = np.fft.rfftfreq(N * 4, d=1.0 / sr)
        band = (f >= 20) & (f <= 20000)
        mag_db = 20 * np.log10(np.abs(H[band]) + 1e-30)
        # 群遅延
        phase = np.unwrap(np.angle(H[band]))
        group_delay = -np.diff(phase) / (2 * np.pi * np.diff(f[band]))
        return {"freq_response_rms_db": np.std(mag_db),
                "group_delay_rms_sample": np.std(group_delay),
                "method": "farina_2007_inverse_filter"}

    @staticmethod
    def analyze_null_test(input_wav: np.ndarray, output_wav: np.ndarray) -> dict:
        """TC-31: Null Test → 差分RMS/Peak/サンプル一致率"""
        diff = input_wav - output_wav
        return {
            "diff_rms": np.sqrt(np.mean(diff**2)),
            "diff_peak": np.max(np.abs(diff)),
            "sample_match_pct": 100 * np.mean(np.abs(diff) < 1e-15),
        }

    @staticmethod
    def analyze_alias(output_wav: np.ndarray, sr: int, os_factor: int,
                      input_freqs: list = None) -> dict:
        """TC-33: Alias Rejection → 理論イメージ周波数位置のエネルギー
        誤った方法: 折り返し帯域全体を評価（ノイズフロアを過大評価）
        正しい方法: 入力周波数から理論イメージ位置を計算し、その近傍のみ評価。
        ex: 48kHz/OS=2x → 内部Nyq=96kHz、入力48kHz超でAlias発生（例: fs=48kHz, fin=52kHz→alias=44kHz）。
        OS=8x時はNyq=384kHzとなり、可聴帯域（≤20kHz）への折り返しリスクがゼロになる。
        TC-33入力19/20/22kHzは全OS倍率でNyquist未満→alias無し=反証テスト。

        ★ [v7.3.3] 両側帯域対応:
        L倍アップサンプリングのイメージ周波数は k*fs ± fin（k=1,...,L-1）の両側帯域。
        従来のコードは下側帯域 |k*fs - fin| のみ評価し、上側帯域 k*fs + fin を見逃していた。
        具体例（fs=48kHz, OS=4x, fin=21kHz）:
          k=1: 下側=27kHz ✓, 上側=69kHz ✓（両方 < 96kHz=内部Nyq）
          k=2: 下側=75kHz ✓, 上側=117kHz ✗（> 96kHz）
          k=3: 下側=123kHz ✗, 上側=165kHz ✗
        range も (1, os_factor+1) → (1, os_factor) に修正（k=L は直流=DC と同一のため不要）
        """
        """
        # ★ [v7.3.5] Welch→FFT変更 (L6): エリアジングピーク検出には Welch 平均化 (ぼかし)
        #   より FFT 単発が適切。以下変更: signal.welch → np.fft.rfft (Hann window)
        # ★ v7.3.5: Hann-windowed FFT 単発 (Welch 不使用)
        window = np.hanning(len(output_wav))
        X = np.fft.rfft(output_wav * window)
        Pxx = np.abs(X) ** 2
        f = np.fft.rfftfreq(len(output_wav), d=1.0 / (sr * os_factor))
        if input_freqs is None:
            input_freqs = [19000, 20000, 22000]  # ★ TC-33 実際の入力周波数に修正
        alias_energy = 0.0
        for fin in input_freqs:
            # ★ [v7.3.3] 両側帯域: k*sr - fin と k*sr + fin
            for k in range(1, os_factor):  # ★ os_factor+1 → os_factor
                for f_alias in [abs(k * sr - fin), k * sr + fin]:
                    if 0 < f_alias < sr * os_factor / 2:  # 内部Nyquist未満
                        bin_near = np.argmin(np.abs(f - f_alias))
                        alias_energy += Pxx[bin_near]
        return {"alias_energy_db": 10 * np.log10(alias_energy + 1e-30)}

    @staticmethod
    def analyze_thd_blackman_harris(output_wav: np.ndarray, sr: int, fundamental_hz: float) -> dict:
        """TC-39: 周波数別 THD（Blackman-Harris FFT, Welch不使用）
        ★ [v7.3.4] 用語修正: 本関数は THD（高調波のみ）を計算する。THD+N（高調波＋ノイズ）ではない。
        THD+N はノッチフィルタで基本波を除去した残渣全量を基本波で除する手法（AES17標準）。
        本関数は FFT ベースで高調波 bin のみ抽出するため THD_F 定義に相当する。
        TC-03/TC-39 の記述「THD+N」を「THD」に修正するか、別途 analyze_thdn() を実装すること。
        Welch平均は歪み成分をぼやけさせるため、Audio Precision同様のFFTベース。
        ★ [v7.3.3] bin-sum 対応:
        Blackman窓の主ローブ幅は約3 bin（6/N * N = 6 bin spacing）。
        単一 bin から高調波パワーを抽出すると、主ローブエネルギーの約50%しか捕捉できない。
        THD は比で計算されるため基本波・高調波とも同様に漏れるため THD 比は概ね補償されるが、
        より正確には基本波・各高調波とも ±2 bin のパワー和を使用すべきである。
        実装方針: fund_power = sum(|X[fund_bin-2:fund_bin+3]|**2) とする。"""
        from numpy import blackman, fft
        N = 262144  # AES17準拠
        if len(output_wav) < N:
            output_wav = np.pad(output_wav, (0, N - len(output_wav)))
        window = np.blackman(N)  # ★ v7.3.2: Blackman窓(a=0.16)とBlackman-Harris窓(a0=0.4243801,a1=0.4973406,a2=0.0782793)は別物。
        #   設計当初はBlackman-Harrisを意図したが、np.blackman()で代用可能（サイドローブ減衰量の差はわずか）。
        #   厳密にBlackman-Harrisを使用する場合はnp.blackmanharris(N)に変更すること。
        X = fft.rfft(output_wav[:N] * window)
        f = fft.rfftfreq(N, d=1.0 / sr)
        fund_bin = np.argmin(np.abs(f - fundamental_hz))

        # ★ [v7.3.3] bin-sum: ±2 bin のパワー和を使用（Blackman窓の主ローブ幅≈3 binに対応）
        def bin_power(center_bin: int, half_width: int = 2) -> float:
            lo = max(0, center_bin - half_width)
            hi = min(len(X), center_bin + half_width + 1)
            return float(np.sum(np.abs(X[lo:hi])**2))

        fund_power = bin_power(fund_bin)
        harmonics = range(2, 11)
        harmonic_power = sum(bin_power(min(fund_bin * h, len(X) - 1)) for h in harmonics)
        return {"thd_db": 10 * np.log10(harmonic_power / (fund_power + 1e-30) + 1e-30),
                "fundamental_hz": fundamental_hz,
                "fft_size": N, "window": "blackman-harris"}  # ★ v7.3.5: thdn_db→thd_db (THD_F定義)
        # ★ [v7.3.5] 命名統一 (L7/L17/L18):
        #   - return key: thdn_db → thd_db (本関数は THD_F 定義、THD+N ではない)
        #   - golden_metrics.json / thresholds の key も thd に統一
        #   - 全 analyzer 関数の return key に _db (lowercase) を統一使用

    @staticmethod
    def analyze_noise_psd(output_wav: np.ndarray, sr: int) -> dict:
        """TC-34: PSD → A-weighting レベル＋形状一致度
        ★ [v7.3.5] 実装設計 (L14): プレースホルダ(...)を以下の計算式で実装する。
        - a_weighted_db: 20-20kHz 帯域の A-weighting 適用済み PSD の積分エネルギー (dB)
        - psd_shape_rms: 対象帯域(20-5kHz)の log-PSD_actual と golden_psd 間の RMS誤差"""
        f, Pxx = signal.welch(output_wav, fs=sr, nperseg=8192)
        a_weight = A_weighting(f)
        band = (f >= 20) & (f <= 20000)
        # A-weighting 補正後の PSD 積分 (dBFS)
        weighted_psd = Pxx[band] * (a_weight[band] ** 2)
        a_weighted_db = 10 * np.log10(np.sum(weighted_psd) + 1e-30)
        # 形状一致度（対数領域の RMS 差）
        log_psd = 10 * np.log10(Pxx[band] + 1e-30)
        shape_band = (f[band] <= 5000)  # 5kHz以下で比較（高域はノイズシェイパー偏倚）
        psd_shape_rms = np.sqrt(np.mean(log_psd[shape_band] ** 2)) if np.any(shape_band) else 0.0
        return {"a_weighted_db": a_weighted_db, "psd_shape_rms": psd_shape_rms}
    #   analyze_stereo_crosstalk() の docstring にエスケープ引用符（\"\"\"）が
    #   使用されているが、これは Markdown コードブロック内の表示上の問題。
    #   実装時は通常のトリプルクォート（\"\"\"）に修正すること。
    #   また analyze_noise_psd() の戻り値 a_weighted_db / psd_shape_rms は
    #   プレースホルダ（...）のため実装時に完成させること。



    @staticmethod
    def analyze_impulse_alignment(output_wav: np.ndarray, expected_delay_samples: int) -> dict:
        """TC-41: インパルスアライメント → ピーク位置/Energy 95%"""
        peak_idx = np.argmax(np.abs(output_wav))
        energy = np.cumsum(output_wav**2)
        energy_95 = np.searchsorted(energy, 0.95 * energy[-1])
        return {
            "peak_offset_samples": peak_idx - expected_delay_samples,
            "energy_95_samples": energy_95,
        }

    @staticmethod
    def analyze_stereo_crosstalk(output_l: np.ndarray, output_r: np.ndarray) -> dict:
        """TC-36: クロストーク → L→R / R→L 両方向"""
        return {
            "crosstalk_lr_db": 10 * np.log10(np.mean(output_r**2) / np.mean(output_l**2) + 1e-30),
            "crosstalk_rl_db": 10 * np.log10(np.mean(output_l**2) / np.mean(output_r**2) + 1e-30),
        }

    @staticmethod
    def analyze_lpf_hpf_response(output_wav: np.ndarray, sr: int,
                                 filter_type: str = "lpf", cutoff_hz: float = 1000.0) -> dict:
        """[v7.3.6 M12] LPF/HPF/Allpass フィルタ特性評価
        TC-09/TC-32（Log Sweep伝達関数）の補足的評価として、フィルタ種類の特定、
        カットオフ近傍の減衰形状（-3dB点、ロールオフ勾配）を算出する。
        filter_type: "lpf" | "hpf" | "allpass" から指定
        カットオフ近傍の振幅スペクトルを抽出し、理論ローパス/ハイパス応答との
        RMS 誤差を計算してフィルタ動作の妥当性を検証する。"""
        from scipy import signal as scipy_signal
        N = len(output_wav)
        freqs = np.fft.rfftfreq(N, d=1.0 / sr)
        X = np.fft.rfft(output_wav)
        mag = np.abs(X)

        # カットオフ近傍（0.1*fc ~ 10*fc）の評価帯域
        lo = max(0.1 * cutoff_hz, freqs[1])
        hi = min(10.0 * cutoff_hz, sr / 2 - 1)
        band = (freqs >= lo) & (freqs <= hi)

        # 1次LPF/HPF の理論振幅応答との比較
        w = 2 * np.pi * freqs[band]
        wc = 2 * np.pi * cutoff_hz
        if filter_type == "lpf":
            # |H(jω)| = 1 / sqrt(1 + (ω/ωc)^2)
            expected = 1.0 / np.sqrt(1.0 + (w / wc) ** 2)
        elif filter_type == "hpf":
            # |H(jω)| = (ω/ωc) / sqrt(1 + (ω/ωc)^2)
            expected = (w / wc) / np.sqrt(1.0 + (w / wc) ** 2)
        else:  # allpass
            expected = np.ones_like(w)  # 全帯域で振幅=1

        actual_db = 20 * np.log10(mag[band] + 1e-30)
        expected_db = 20 * np.log10(expected + 1e-30)
        rms_error_db = np.sqrt(np.mean((actual_db - expected_db) ** 2))

        # -3dB 点の実測（LPF の場合）
        mag_db = 20 * np.log10(mag[band] + 1e-30)
        max_db = np.max(mag_db)
        f_3db_idx = np.argmin(np.abs(mag_db - (max_db - 3.0)))
        f_3db = freqs[band][f_3db_idx]

        # ロールオフ勾配（dB/oct、10*fc〜0.1*fc の低域和高域で線形回帰）
        rolloff_band = band & ((freqs >= 0.5 * cutoff_hz) & (freqs <= 5 * cutoff_hz))
        if np.sum(rolloff_band) > 2:
            f_roll = freqs[rolloff_band]
            m_roll = mag_db[rolloff_band]
            slope, _ = np.polyfit(np.log2(f_roll + 1e-30), m_roll, 1)
            rolloff_db_per_oct = slope
        else:
            rolloff_db_per_oct = 0.0

        return {
            "filter_type": filter_type,
            "cutoff_hz_actual": float(f_3db),
            "cutoff_error_db": float(f_3db - cutoff_hz),
            "rolloff_db_per_oct": float(rolloff_db_per_oct),
            "shape_rms_error_db": float(rms_error_db),
        }

    @staticmethod
    def compare_golden_wav(output_wav: np.ndarray, golden_wav: np.ndarray,
                           sr: int, tc_name: str = "") -> dict:
        """[v7.3.6 M12] ゴールデン WAV 比較関数
        測定出力とゴールデンマスター間の総合比較。
        - 振幅スペクトル RMS 誤差（20Hz-20kHz）
        - ピーク位置オフセット（TC-41 相当）
        - エネルギー 95% 点オフセット
        - RMS 偏差 (dBFS)
        TC-01/02/41 等の周波数応答/遅延テストで golden_metrics.json と比較使用。"""
        # 長さ合わせ（短い方をゼロパッド）
        n = max(len(output_wav), len(golden_wav))
        out = np.pad(output_wav, (0, max(0, n - len(output_wav))))
        ref = np.pad(golden_wav, (0, max(0, n - len(golden_wav))))

        # 振幅スペクトル RMS 誤差
        N = n
        H_out = np.fft.rfft(out, n=N * 4)
        H_ref = np.fft.rfft(ref, n=N * 4)
        f = np.fft.rfftfreq(N * 4, d=1.0 / sr)
        band = (f >= 20) & (f <= 20000)
        mag_out_db = 20 * np.log10(np.abs(H_out[band]) + 1e-30)
        mag_ref_db = 20 * np.log10(np.abs(H_ref[band]) + 1e-30)
        mag_rms_error = np.sqrt(np.mean((mag_out_db - mag_ref_db) ** 2))

        # ピーク位置オフセット
        peak_out = np.argmax(np.abs(out))
        peak_ref = np.argmax(np.abs(ref))
        peak_offset = peak_out - peak_ref

        # エネルギー 95% 点オフセット
        e_out = np.cumsum(out ** 2)
        e_ref = np.cumsum(ref ** 2)
        idx95_out = np.searchsorted(e_out, 0.95 * e_out[-1]) if e_out[-1] > 0 else 0
        idx95_ref = np.searchsorted(e_ref, 0.95 * e_ref[-1]) if e_ref[-1] > 0 else 0
        energy95_offset = idx95_out - idx95_ref

        # RMS 偏差
        rms_out = np.sqrt(np.mean(out ** 2))
        rms_ref = np.sqrt(np.mean(ref ** 2))
        rms_deviation_db = 20 * np.log10(rms_out / (rms_ref + 1e-30))

        return {
            "tc": tc_name,
            "freq_response_rms_error_db": float(mag_rms_error),
            "peak_offset_samples": int(peak_offset),
            "energy95_offset_samples": int(energy95_offset),
            "rms_deviation_db": float(rms_deviation_db),
            "pass": mag_rms_error < 0.05 and abs(rms_deviation_db) < 0.1,
        }


# ── Golden Metrics 保存 ──
# ゴールデンWAVだけでなく、測定結果（THD・遅延・周波数特性・ノイズ等）も
# golden_metrics.json として保存する。将来FFTアルゴリズム変更等で
# Bit一致しなくなっても、品質が同等であることを判断できる。
# 保存形式:
# {
#   "tc": "TC-03",
#   "pattern": "P1-Baseline",
#   "build": "release_msvc_v143",
#   "metrics": { "thd_db": -102.3, "noise_floor_db": -121.5 },
#   "golden_metrics": { "thd_db": -102.5, "noise_floor_db": -121.8 },
#   "threshold": { "thd": -100, "noise_floor": -120 }
# }


# ── 補助関数 ──
def A_weighting(f: np.ndarray) -> np.ndarray:
    """IEC 61672 / JIS C 1509 A-weighting フィルタ応答（ITU-R 468 とは別物）"""
    f = np.asarray(f, dtype=float)
    f_sq = f * f
    num = 12194**2 * f_sq**2
    denom = (f_sq + 20.6**2) * np.sqrt((f_sq + 107.7**2) * (f_sq + 737.9**2)) * (f_sq + 12194**2)
    return num / (denom + 1e-30)

def compare_frequency_response(actual: np.ndarray, expected: np.ndarray,
                                 sr: int, band: Tuple[float, float]) -> dict:
    """周波数領域のRMS誤差（FFTベース。freqzは使わない）"""
    N = max(len(actual), len(expected))
    H_act = np.fft.rfft(actual, n=N)
    H_exp = np.fft.rfft(expected, n=N)
    f = np.fft.rfftfreq(N, d=1.0 / sr)
    mask = (f >= band[0]) & (f <= band[1])
    mag_act_db = 20 * np.log10(np.abs(H_act[mask]) + 1e-30)
    mag_exp_db = 20 * np.log10(np.abs(H_exp[mask]) + 1e-30)
    return {
        "magnitude_rms_db": np.sqrt(np.mean((mag_act_db - mag_exp_db)**2)),
        "phase_rms_deg": np.degrees(np.sqrt(np.mean(np.angle(H_act[mask] / H_exp[mask])**2))),
    }

def theoretical_dirac_response(n_samples: int, sr: float) -> np.ndarray:
    """理想ディラック応答（1サンプル=1.0、残り=0.0）"""
    y = np.zeros(n_samples)
    y[0] = 1.0
    return y
```

### 5.10 Phase 1 追記タスク（拡充分）

| # | タスク | 成果物 | 関連テスト |
|---|-------|--------|-----------|
| 1-12 | 分析関数拡張（Null/Sweep/Alias/PSD） | `analyzers.py` | TC-31〜35,39,41 |
| 1-13 | ゴールデン比較フレームワーク | `golden_calculator.py` | 全TC |
| 1-14 | THD Sweep 複数周波数対応 | `analyzers.py` `analyze_thd_sweep()` | TC-39 |
| 1-15 | Long-run スタビリティ試験 | `test_runtime.py` | TC-38 |
| 1-16 | test_config.yaml 拡充（11TC追加） | `test_config.yaml` | TC-31〜41 |
| 1-17 | P5/P7 パターン拡充（Alias/PSD評価追加） | `test_config.yaml` | TC-33,34 |

---

## 6. 工数総括

| フェーズ | 工数（人日） | 備考 |
|---------|-------------|------|
| Phase 0: CLI拡張 | **9** | OutputCaptureSink + SPSC + BG WAV + RecoveryHistory |
| Phase 1: DSP品質試験 | **58〜73** | 既存34TC + 新規11TC（Null/Sweep/Alias/PSD等）+ ゴールデン比較FW + CLIパラメータ拡張6個（★ v7.3.2: 55〜70→58〜73に修正。§5.1 CLI未提供パラメータ6種の追加実装 +3人日を反映）|
| Phase 2: Runtime試験 | **40〜50** | TC-25/27/30/38安定化含む |
| Phase 3: レポート/CI | **20〜25** | HTML + GitHub連携 |
| 予備 | **30〜70** | 実績に応じて変動 |
| **合計** | **157〜227（中央値192人日 ≈ 25〜29週間）** | ★ v7.3.2: Phase 1 のCLI追加工数を反映 |

---

## 7. 実装前チェックリスト

| # | 確認項目 | ステータス |
|---|---------|-----------|
| 1 | Seqlock: generation 偶数→奇数→偶数（`old+1` → write → `old+2`）| ☐ |
| 2 | Seqlock: release/acquire のみで同期（thread_fence 不要）| ☐ |
| 3 | `outputPath_`: `juce::File` 単体、Thread 開始前固定契約 | ☐ |
| 4 | シャットダウン: `setCaptureSink(nullptr)` → `stop()` → wait → destroy | ☐ |
| 5 | BG Thread: WaitableEvent `wait(-1)` + `signal()`、drain-all 設計 | ☐ |
| 6 | `sizeof(AudioBlock)` の static_assert なし（trivially_copyable のみ）| ☐ |
| 7 | Seqlock Reader: 8回リトライ、失敗時は slot 破棄 | ☐ |
| 8 | float 変換バッファ: `run()` 内で事前確保（毎ブロック生成禁止）| ☐ |
| 9 | `CapturePoint`: 単一 enum（二重管理なし）| ☐ |
| 10 | RecoveryHistory: `copyRecoveryHistorySnapshot()` が snapshot 返却 | ☐ |
| 11 | `capture()`: push + signal のみ（pop/WAV は BG Thread の責務）| ☐ |
| 12 | `pushBecameNonEmpty()` / `pushBecameNonEmptyWithWriter()`: LockFreeRingBuffer に追加。capture() は writer 版を使用 | ☐ |
| 13 | `stopRequested_`: `store(release)` / `load(acquire)`, signal() で起床 | ☐ |
| 14 | TC-25: OutputFilter係数変更時例外（≤−100dBFS）| ☐ |
| 15 | TC-01B: IR 参照パス `sampledata/impulse_room_correction_hpf_lpf.wav`（論理名 `room_correction`）| ☐ |
| 16 | テストケース数: 全41件（既存34 + 新規11）で統一 | ☐ |
| 17 | `test_config.yaml`: 10パターン定義完了（P1〜P10）| ☐ |
| 18 | `analyzers.py`: Null Test / Log Sweep / Alias / PSD / THD Sweep 実装完了 | ☐ |
| 19 | `golden_calculator.py`: 理論値・Golden WAV・統計比較フレームワーク完了 | ☐ |
| 20 | P5/P7 パターンに TC-33(Alias)/TC-34(PSD)/TC-39(THDSweep) 割当完了 | ☐ |
| 21 | ★ WaitableEvent デフォルト構築（auto-reset）であることを実装時確認 | ☐ |
| 22 | ★ Seqlock `recordRecoveryAction()`: Timer Thread 単一呼出の設計契約遵守 | ☐ |
| 23 | ★ AudioBlock: `is_trivial_v` が false でも問題なし（trivially_copyable のみ要求）| ☐ |
| 24 | ★ TC-34: P5-ReferenceQuality(32bit) のみ割当。P6-Musical(16bit) には割当禁止 | ☐ |
| 25 | ★ analyzers.py 実装時: `\"\"\"` エスケープを `"""` に修正、PSD戻り値のプレースホルダ完成 | ☐ |
| 26 | ★ OutputCaptureSink 単体テスト: `src/tests/OutputCaptureSinkTests.cpp` 新設（シャットダウン順序・リング満杯・drain-all）| ☐ |
| 27 | ★ `--cli-dump-filter-coeffs`: OutputFilter に const getter 新設（`hcCoeff`/`lcCoeff`/`hpfCoeff` は private）| ☐ |
| 28 | ★ JUCE 8 の `WaitableEvent` 仕様をビルド時に再確認（バージョンアップ対応）| ☐ |
| 29 | ★ shutdown: `outputCaptureSink_` は AudioEngine に atomic ポインタとして新設（`MainWindow.cpp` から注入）| ☐ |
| 30 | ★ shutdown: 既存 `~MainWindow()` の10step に計画書の統合版14step を正しく組み込むこと | ☐ |
| 31 | ★ CLIパラメータ拡張6種（`--cli-os-type`,`--cli-os-factor`,`--cli-hc-mode`,`--cli-lc-mode`,`--cli-soft-clip`,`--cli-saturation`）を Phase 1 で実装 | ☐ |
| 32 | ★ §5.2 測定パターン数を「10」に統一（P1〜P10）。P1-Baseline の TC一覧に TC-31,32,39,41 を追加反映 | ☐ |
| 33 | ★ `analyze_thd_blackman_harris()`: `np.blackman()` と `blackman_harris` 名の不整合を解消（np.blackmanharris に変更、または関数名を `analyze_thd_blackman` に改名）| ☐ |
| 34 | ★ `analyze_alias()`: Alias計算例を修正（OS=4x, fin=21kHz→alias=75kHz は誤り。正しくは OS=2x, fin>48kHz で Alias 発生）| ☐ |
| 35 | ★ Seqlock `fetchAdd` → `convo::fetchAddAtomic` に統一（既存関数名との整合）| ☐ |
| 36 | ★ [v7.3.3] `capture()` シグネチャ: `const double* left, const double* right, int numSamples` を使用（`juce::AudioBuffer<double>` は不可）| ☐ |
| 37 | ★ [v7.3.3] `pushBecameNonEmptyWithWriter()`: 新設（ゼロコピー・writer ラムダ版）。`capture()` はこちらを使用し、ローカル AudioBlock + コピーを回避 | ☐ |
| 38 | ★ [v7.3.3] `capture()` 内 writer ラムダ: `block.numSamples = currentBlockSize` 等のメタデータ設定が必須。ゼロ初期化の0のままでは BG Thread がデータロスト | ☐ |
| 39 | ★ [v7.3.3] `CapturePoint::PostDither`: `processOutput()` L503後・L505前（`applyFixedLatencyDelay` 後、float変換前）に挿入。データは `double* dataL/dataR` | ☐ |
| 40 | ★ [v7.3.3] `analyze_alias()`: 両側帯域 `k*sr±fin` を評価（上側帯域 `k*sr+fin` の見逃しを修正）。k range は `range(1, os_factor)` | ☐ |
| 41 | ★ [v7.3.3] TC-33 評価方法: 「10kHz以下」→「理論イメージ位置（k・fs±fin）のエネルギー」に統合。TC-33 は反証テスト | ☐ |
| 42 | ★ [v7.3.3] `analyze_thd_blackman_harris()`: ±2 bin のパワー和（`bin_power()`）を使用。単一 bin は主ローブエネルギーの約50%しか捕捉しない | ☐ |
| 43 | ★ [v7.3.3] WAV出力フォーマット: float32（24bit有効仮数）。TC-37「1 ULP」は float32 の ULP。TC-38(30分) ≈ 660MB | ☐ |
| 44 | ★ [v7.3.3] `build_cli_args()`: §5.1 の6新規CLIパラメータ（`--cli-os-type` 等）を組み立てるコードを追加 | ☐ |
| 45 | ★ [v7.3.4] ProcessingState に `testCaptureQueue` ポインタ追加。既存 `adaptiveCaptureQueue` と独立。CLI テストモードでは既存キューは非アクティブ | ☐ |
| 46 | ★ [v7.3.4] `processOutputDouble()` (DSPCoreDouble.cpp:620) にも PostDither キャプチャ挿入。Float path と Double path の両方で対応 | ☐ |
| 47 | ★ [v7.3.4] `analyze_thd_blackman_harris()` は THD（高調波のみ）。TC-03/TC-39 の「THD+N」を「THD」に修正、または `analyze_thdn()` 別途実装 | ☐ |
| 48 | ★ [v7.3.4] `generators.py`: 出力パスをパラメータ化、`testdata/generated/` に出力。既存ツールの `C:/TEMP/` ハードコードを踏襲しない | ☐ |
| 49 | ★ [v7.3.4] `generate_synthetic_ir("dirac")`: float32 で出力。既存 `create_dirac_ir.py` の 16bit は使用不可 | ☐ |
| 50 | ★ [v7.3.4] §8 測定データ一覧に従い、IR（4種）、入力信号（10種）、ゴールデン参照を Phase 1 で生成。`testdata/` ディレクトリ構成を遵守 | ☐ |
| 51 | ★ [v7.3.5] `capture()` に `const float*` overload を追加。PostDither は Float Path で `float* dstL/dstR`、Double Path で `double*`。writer ラムダ内で `static_cast<double>(srcF[i])` で AudioBlock.L/R に格納 | ☐ |
| 52 | ★ [v7.3.5] `buildAudioThreadProcessingState()` 内で `convo::consumeAtomic(outputCaptureSink_, acquire)` で testCaptureQueue を1回ロード。`setCaptureSink()` は `publishAtomic(..., release)` | ☐ |
| 53 | ★ [v7.3.5] TC-31a 閾値を「サンプル一致率 100%」→「差分RMS ≤ -138dBFS / Peak ≤ -128dBFS」に緩和。DCBlocker+NaN除去+Dither 常時動作により bit-exact 不可 | ☐ |
| 54 | ★ [v7.3.5] `analyze_sweep_transfer()` を Farina 2007 inverse filter 方式に全面差し替え。`generate_log_sweep()` で逆フィルタ信号 `_inverse.wav` を同時生成 | ☐ |
| 55 | ★ [v7.3.5] `processOutputDouble()` PostDither 位置を「L801+L803-805 後」に修正。Double Path では float 変換なしで `double*` のまま buffer にコピー | ☐ |
| 56 | ★ [v7.3.5] `--cli-output-wav` / `--cli-capture-mode` の MainWindow 実装: findValue パターンで OutputCaptureSink ライフサイクル管理。`cliCaptureSink_` メンバ変数新設 | ☐ |
| 57 | ★ [v7.3.5] `analyze_thd_blackman_harris()` return key を `thdn_db` → `thd_db` に変更。golden_metrics.json / threshold YAML も `thd` に統一 | ☐ |
| 58 | ★ [v7.3.5] `--cli-order` の CLI値→ComboBox ID→実動作 マッピング表を §5.1 に追加。`conv`=Convolver-Only, `peq`=PEQ-Only | ☐ |
| 59 | ★ [v7.3.5] `analyze_noise_psd()` のプレースホルダ(...) を実装: A-weighting適用PSD積分(dBFS) + 5kHz以下 log-PSD RMS 形状一致度 | ☐ |
| 60 | ★ [v7.3.5] `generate_multitone()` に `np.random.RandomState(42)` で決定的 random phase offset を付与（AES17 準拠） | ☐ |
| 61 | ★ [v7.3.5] `analyze_alias()` を Welch → Hann-windowed FFT 単発に変更。ピーク検出精度向上 | ☐ |
| 62 | ★ [v7.3.5] OS 別 IR 要件: §8.1 に追記。既定では全OS倍率で base-rate IR 使用。`processUp()` 補間結果で評価 | ☐ |
| 63 | ★ [v7.3.5] `~MainWindow()` step 2 `setAdaptiveAutosaveCallback({})` の省略を §2.4 に脚注で明記 | ☐ |
| 64 | ★ [v7.3.5] `.gitignore` に `testdata/generated/` を追加（§8.5 に注記済み） | ☐ |
| 65 | ★ [v7.3.5] `np.blackman` vs `np.blackmanharris` の違いを実装時に確認。関数名 `analyze_thd_blackman_harris` と実装窓関数の整合を取る | ☐ |
| 66 | ★ [v7.3.5] IR と入力信号の振幅仕様: IR `data[0]=1.0`(0dBFS)、入力信号 `-6dBFS`(0.5)。線形スケーリングで比較可能 | ☐ |
| 67 | ★ [v7.3.5] `generators.py` は `Path("testdata/generated")` 絶対パスを使用し、IR_FILE_MAP と §8.5 の一貫性を確保 | ☐ |
| 68 | ★ [v7.3.6] OutputCaptureSink クラス: `sampleRateHz_`/`bitDepth_`/`coeffBankIndex_`/`sessionId_` を private 宣言（初期値 48000.0/64/0/0）、`setAudioParams()` public setter を追加。capture() Lambda（L200-203）で使用される4変数。実装初日にビルドエラーになる致命的问题 | ☐ |
| 69 | ★ [v7.3.6] `pushBecameNonEmptyWithWriter()`: `LockFreeRingBuffer.h` に pushWithWriter() を拡張する形で実装。PushResult を返す writer ラムダ版。Phase 0 Day 1 で実装 | ☐ |
| 70 | ★ [v7.3.6] TC-37 Numerical Transparency: Float Path (processOutput) でのみ測定可能。Double Path (processOutputDouble) では kOutputHeadroom=0.891 により PeakLimiter が -0.1dBFS(amplitude≈0.9886) で激活して歪む。Double Precision 設定では Float Path フォールバックを使用 | ☐ |
| 71 | ★ [v7.3.6] §8.2 信号表の「Sine 1kHz -6dBFS」行に TC-04, TC-04A を追加（ノイズシェイパ設定違いを評価、入力信号は TC-01/03 と同じ）。TC-17/18 を §5.7 テストケース表に追加（SMPTE IMD 60Hz+7kHz / CCIF IMD 19kHz+20kHz）| ☐ |
| 72 | ★ [v7.3.6] `analyzers.py` Phase 1-2 で `analyze_lpf_hpf_response()`（LPF/HPF/Allpass フィルタ特性評価：カットオフ点・ロールオフ勾配・形状RMS誤差）と `compare_golden_wav()`（Golden WAV 比較：振幅スペクトルRMS誤差・ピーク位置オフセット・エネルギー95%点オフセット・rms偏差・pass判定）を実装 | ☐ |
| 73 | ★ [v7.3.6] 既存テストケース16件（TC-05A〜D/08/10/11/11B/12/13/14/16/19/20/22/26/28）は §5.2 の P1〜P10 パターンに未割り当て。§5.2 パターン表に脚注で明示。既存 CI テストフレームワーク（`src/tests/*.cpp`）で継続検証 | ☐ |

---

## 8. 測定データ一覧・仕様・作成方法

> **[v7.3.4 新設]** 本セクションは自動テストに必要な全データ資産の仕様と生成方法を定義する。
> Phase 1 の最初のタスクとして、これらのデータ生成スクリプトを実装・検証すること。

### 8.1 IR（インパルス応答）データ

| 論理名 | ファイルパス | フォーマット | 用途 | 作成方法 |
|--------|-------------|-------------|------|---------|
| `dirac` | `testdata/generated/dirac_test.wav` | 48kHz, **float32**, mono, 8192samples | TC-01/02/31a/41: 伝達関数基準 | `generators.py::generate_synthetic_ir("dirac")` で生成。★ v7.3.4: 既存 `create_dirac_ir.py` は **16bit PCM** であるため **使用不可**。float32 版を新規生成すること。★ v7.3.5: IR の振幅仕様 — Dirac IR の `data[0] = 1.0` (=0dBFS) だが、TC-31a/41 の入力信号は Dirac **-6dBFS** (=0.5)。プランでは `generate_dirac_input(amplitude=0.5)` で生成。伝達関数評価においては 0dBFS / -6dBFS どちらもスケーリングは線形なので結果の比較に問題なし。 |
| `room_correction` | `sampledata/impulse_room_correction_hpf_lpf.wav` | 48kHz, **実測IR** | TC-01B: 実IR品質確認 | 既存ファイル（実在確認済: `Test-Path` ✅）。REW + rePhase で作成された実測ハウスカーブベース IR。 |
| `lpf_1k` | `testdata/generated/lpf_1k_test.wav` | 48kHz, **float32**, stereo, 129 taps | TC-07/21: LPF伝達関数精度 | `generators.py::generate_synthetic_ir("lpf_1k")` で生成。windowed-sinc 法（Hamming窓, fc=1kHz, 129 taps）。既存 `create_test_irs.py` のアルゴリズムを流用し float32 出力に変更。 |
| `hpf_20` | `testdata/generated/hpf_20_test.wav` | 48kHz, float32, stereo, 129 taps | TC-07: HPF伝達関数精度 | `generators.py::generate_synthetic_ir("hpf_20")` で生成。spectral inversion 法（fc=20Hz, 129 taps）。 |

> **★ [v7.3.4] 既存ツールのハードコードパス問題**:
> `tools/diagnostics/create_dirac_ir.py` は `C:/TEMP/dirac_test.wav`（16bit）を生成する。
> `tools/diagnostics/create_test_irs.py` は `C:/TEMP/fir_lpf_200hz.wav`（32bit float stereo）を生成する。
> 新設する `generators.py` は出力パスをパラメータ化し、`testdata/generated/` 以下に出力すること。

> **[v7.3.5] OS 別 IR tap 数の要件** (L3):
> `lpf_1k` / `hpf_20` は 129taps@48kHz (=2.6875ms) で生成される。
> OS 8x 時（処理レート 384kHz）では、IR をアップサンプリングする必要はないが、
> Oversampling up-sampling 内部の `oversampling.processUp()` (DSPCoreFloat.cpp:258)
> が IR ブロックを 8x に補間する際、129taps では十分なインパルス応答の忠実度を
> 確保できない可能性がある。
>
> **対策**:
> - OS=1x テスト: 129taps@48kHz IR を使用（base-rate 測定）。問題なし。
> - OS=8x テスト: **P5-ReferenceQuality / P6-Musical では OS=8x で測定する。
>   同一の 48kHz base-rate IR を使用し、`processUp()` の補間結果で伝達関数を評価。
>   伝達関数誤差増加を許容する（補間フィルタの忠実度試験も兼ねる）。
> - 高精度比較用: `dirac` IR (8192samples@48kHz) を使用すれば問題なし。
>
> Phase 1 で OS=8x 用の 8x レート IR (1032taps@384kHz) を追加生成するかは任意。
> 既定では全OS倍率で base-rate IR を使用する。

#### IR 生成アルゴリズム仕様

```python
# generators.py — IR 生成（Phase 1-1 で実装）

import numpy as np
from scipy.io import wavfile
from pathlib import Path

def generate_synthetic_ir(ir_name: str, output_dir: Path = None) -> Path:
    """IR 論理名から float32 WAV を生成し、パスを返す。
    ★ 全ての合成IRは float32 (IEEE 754 single precision) で出力する。
    """
    sr = 48000
    output_dir = output_dir or Path("testdata/generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ir_name == "dirac":
        # Dirac impulse: sample[0] = 1.0, rest = 0.0
        # ★ float32 で出力（16bit PCM は精度不足）
        n = 8192
        data = np.zeros(n, dtype=np.float32)
        data[0] = 1.0
        path = output_dir / "dirac_test.wav"
        wavfile.write(str(path), sr, data)
        return path

    elif ir_name == "lpf_1k":
        # Windowed-sinc LPF: fc=1kHz, 129 taps, Hamming window
        n_taps = 129
        fc = 1000.0
        nyq = sr / 2.0
        fc_norm = fc / nyq
        half = (n_taps - 1) // 2
        taps = np.zeros(n_taps)
        for i in range(n_taps):
            k = i - half
            if k == 0:
                taps[i] = 2.0 * fc_norm
            else:
                taps[i] = np.sin(2.0 * np.pi * fc_norm * k) / (np.pi * k)
            # Hamming window
            taps[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n_taps - 1))
        taps /= np.sum(taps)  # Unity gain at DC
        # Stereo, float32
        data = np.column_stack([taps.astype(np.float32), taps.astype(np.float32)])
        path = output_dir / "lpf_1k_test.wav"
        wavfile.write(str(path), sr, data)
        return path

    elif ir_name == "hpf_20":
        # HPF via spectral inversion of LPF(20Hz)
        n_taps = 129
        fc = 20.0
        nyq = sr / 2.0
        fc_norm = fc / nyq
        half = (n_taps - 1) // 2
        lpf = np.zeros(n_taps)
        for i in range(n_taps):
            k = i - half
            if k == 0:
                lpf[i] = 2.0 * fc_norm
            else:
                lpf[i] = np.sin(2.0 * np.pi * fc_norm * k) / (np.pi * k)
            lpf[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / (n_taps - 1))
        lpf /= np.sum(lpf)
        hpf = np.zeros(n_taps)
        hpf[half] = 1.0  # δ[n-half]
        hpf -= lpf       # Spectral inversion
        data = np.column_stack([hpf.astype(np.float32), hpf.astype(np.float32)])
        path = output_dir / "hpf_20_test.wav"
        wavfile.write(str(path), sr, data)
        return path

    else:
        raise ValueError(f"Unknown IR name: {ir_name}")
```

### 8.2 テスト入力信号

| 信号名 | TC | フォーマット | 仕様 | 作成方法 |
|--------|----|----|------|---------|
| Sine 1kHz -6dBFS | TC-01,03,04,04A,31b,36,37,39 | 48kHz float32 stereo 3s | `0.5 * sin(2π·1000·t)` L=R | `generators.py::generate_sine(freq=1000, duration=3.0, amplitude=0.5)` |
| Sine sweep | TC-09,32,35 | 48kHz float32 stereo 10s | 20Hz-24kHz 対数スイープ | `generators.py::generate_log_sweep(f0=20, f1=24000, duration=10.0)` |
| Multitone 19/20/22kHz | TC-33 | 48kHz float32 stereo 3s | 各 -6dBFS の3音合成 | `generators.py::generate_multitone(freqs=[19000,20000,22000], duration=3.0, amplitude=0.5)` |
| Silence | TC-34,40 | 48kHz float32 stereo 5-10s | 全サンプル 0.0 | `generators.py::generate_silence(duration=5.0)` |
| Dirac -6dBFS | TC-31a,41 | 48kHz float32 stereo 1-3s | sample[0]=-6dBFS, rest=0 | `generators.py::generate_dirac_input(amplitude=0.5, duration=3.0)` |
| Sine 40Hz -6dBFS | TC-38 | 48kHz float32 stereo 30min | `0.5 * sin(2π·40·t)` L=R | `generators.py::generate_sine(freq=40, duration=1800.0, amplitude=0.5)` |
| Sine -0.1dBFS | TC-37 | 48kHz float32 stereo 3s | `10^(-0.1/20) ≈ 0.9886` 振幅 | `generators.py::generate_sine(freq=1000, duration=3.0, amplitude=0.9886)` |
| IMD 60Hz+7kHz | TC-17,18 | 48kHz float32 stereo 3s | 4:1 比率（IEC 60268-3 SMPTE） | `generators.py::generate_imd(freq_low=60, freq_high=7000, ratio=0.25, duration=3.0)` |
| THD sweep | TC-39 | 48kHz float32 stereo 各3s | 100/1k/5k/10k/18k/20kHz 各 -6dBFS | `generators.py::generate_thd_sweep(freqs=[100,1000,5000,10000,18000,20000])` |
| L=1kHz R=silence | TC-36 | 48kHz float32 stereo 3s | L: `0.5*sin(2π·1000·t)`, R: 0.0 | `generators.py::generate_stereo_asymmetric(freq_l=1000, freq_r=None)` |

#### 入力信号生成API仕様

```python
# generators.py — 入力信号生成（Phase 1-1 で実装）

def generate_sine(freq: float, duration: float, amplitude: float = 0.5,
                  sr: int = 48000) -> Path:
    """純音正弦波を float32 stereo WAV として出力。L=R。"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    sig = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"sine_{int(freq)}hz_{duration}s.wav"
    wavfile.write(str(path), sr, data)
    return path

def generate_log_sweep(f0: float, f1: float, duration: float,
                       sr: int = 48000, amplitude: float = 0.5) -> Path:
    """対数周波数スイープ信号を float32 stereo WAV として出力。
    伝達関数測定用（TC-32/35: Farina 2007 inverse filter 法）。
    ★ [v7.3.5] 逆フィルタ信号も同時生成し `_inverse.wav` として保存。
    Farina 2007 "Simultaneous measurement of impulse response and distortion
    with a swept-sine technique" (AES 122nd Convention, 2007) に準拠。

    数式:
      s(t) = amplitude · sin(φ(t))
      φ(t) = 2π · f₀ · T / ln(f₁/f₀) · (exp(t · ln(f₁/f₀)/T) - 1)
      逆フィルタ: s⁻¹(t) = s(T-t) · exp(-t · ln(f₁/f₀) / T)
        （時間反転 + 振幅補償エンベロープ。-6dB/oct 相当）
      インパルス応答: h(t) = s⁻¹(t) * y(t) （*は畳み込み、y(t)は出力信号）
      伝達関数: H(f) = FFT{h(t)}

    逆フィルタの振幅補償 exp(-t·ln(f₁/f₀)/T) は:
    - t=0 で 1.0（高周波側の減衰なし）
    - t=T で 1/(f₁/f₀) = f₀/f₁（低周波側のエネルギー増幅補償）
    - これにより log sweep の周波数依存エネルギー密度を均一化する。"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    # Log sweep: f(t) = f0 * (f1/f0)^(t/T)
    phase = 2 * np.pi * f0 * duration / np.log(f1 / f0) * \
            (np.exp(t * np.log(f1 / f0) / duration) - 1)
    sig = (amplitude * np.sin(phase)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"logsweep_{int(f0)}_{int(f1)}_{duration}s.wav"
    wavfile.write(str(path), sr, data)

    # ★ v7.3.5: 逆フィルタ信号の生成と保存
    amplitude_compensation = np.exp(-t * np.log(f1 / f0) / duration)
    s_inv = (sig[::-1] * amplitude_compensation).astype(np.float32)
    inv_data = np.column_stack([s_inv, s_inv])
    inv_path = Path("testdata/generated") / f"logsweep_{int(f0)}_{int(f1)}_{duration}s_inverse.wav"
    wavfile.write(str(inv_path), sr, inv_data)
    return path  # 戻り値は入力信号パス（逆フィルタは _inverse.wav として同じdirに保存）

def generate_silence(duration: float, sr: int = 48000) -> Path:
    """無音信号（全ゼロ）を float32 stereo WAV として出力。"""
    n = int(sr * duration)
    data = np.zeros((n, 2), dtype=np.float32)
    path = Path("testdata/generated") / f"silence_{duration}s.wav"
    wavfile.write(str(path), sr, data)
    return path

def generate_multitone(freqs: list, duration: float, amplitude: float = 0.5,
                       sr: int = 48000) -> Path:
    """多音信号を float32 stereo WAV として出力。各周波数の振幅を amplitude に設定。
    ★ [v7.3.5] AES17 準拠 random phase: 各トーンにランダム初期位相を付与。
    これにより各トーンが同一位相で重畳して peak が 3×amplitude になるのを防止する。
    平均振幅維持のため /len(freqs) 正規化は継続するが、random phase で peak は
    √N×amplitude/len(freqs) 程度に抑えられる。"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    rng = np.random.RandomState(42)  # ★ 決定的な位相 (reproducible)
    for f in freqs:
        phase_offset = rng.uniform(0, 2 * np.pi)
        sig += amplitude * np.sin(2 * np.pi * f * t + phase_offset)
    sig = (sig / len(freqs)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"multitone_{duration}s.wav"
    wavfile.write(str(path), sr, data)
    return path

def generate_imd(freq_low: float, freq_high: float, ratio: float,
                 duration: float, sr: int = 48000) -> Path:
    """IMD測定信号（SMPTE RP120-3: 低周波数を高周波数で振幅変調）。
    標準 4:1 比率 = 低音が高音の 4 倍の振幅。"""
    n = int(sr * duration)
    t = np.arange(n) / sr
    envelope = 1.0 + ratio * np.sin(2 * np.pi * freq_low * t)
    sig = (0.5 * envelope * np.sin(2 * np.pi * freq_high * t)).astype(np.float32)
    data = np.column_stack([sig, sig])
    path = Path("testdata/generated") / f"imd_{int(freq_low)}_{int(freq_high)}.wav"
    wavfile.write(str(path), sr, data)
    return path
```

### 8.3 ゴールデン参照データ

| データ名 | 格納場所 | 内容 | 作成方法 |
|---------|---------|------|---------|
| ゴールデン WAV | `testdata/golden/{pattern}/{tc}/output.wav` | リファレンス実装の出力WAV | 初回リリースビルドで生成。Git LFS または release artifact として管理。 |
| ゴールデン metrics | `testdata/golden/{pattern}/{tc}/golden_metrics.json` | THD/FreqResp/NoiseFloor 等の数値 | 初回リリースビルドで `analyzers.py` の結果を保存。 |

```json
// golden_metrics.json 形式例
{
    "tc": "TC-03",
    "pattern": "P1-Baseline",
    "build": "release_msvc_v143",
    "metrics": {
        "thd_db": -102.3,
        "noise_floor_db": -121.5,
        "freq_response_rms_db": 0.02
    },
    "golden_metrics": {
        "thd_db": -102.5,
        "noise_floor_db": -121.8,
        "freq_response_rms_db": 0.01
    },
    "threshold": {
        "thd": -100,
        "noise_floor": -120,
        "freq_response": 0.05
    }
}
```

### 8.4 閾値定義

| ビルド | THD | IMD | Noise Floor | Freq Response | 備考 |
|--------|-----|-----|-------------|---------------|------|
| Debug MSVC | -80dB | -80dB | -115dBFS | ±0.05dB | 開発時 quick check |
| Release MSVC | -100dB | -90dB | -120dBFS | ±0.05dB | 品質ゲート |
| Release ICX | -100dB | -90dB | -120dBFS | ±0.05dB | 品質ゲート |

> **閾値の根拠**:
> - THD -100dB (Release): double精度パイプラインの理論THDは -300dB 以下。実装上の誤差（FMA差・丸め差）を考慮しても -100dB は十分に余裕がある。
> - Noise Floor -120dBFS: 32bit floatパイプラインの理論ノイズフロア ≈ -144dBFS。実装ノイズを含めても -120dBFS は達成圏内。
> - Freq Response ±0.05dB: IIRフィルタ（Butterworth/Linkwitz-Riley）の設計精度と double 精度演算の組合せで ±0.05dB は標準的要求。

### 8.5 データディレクトリ構成

```
testdata/
├── generated/          # generators.py が自動生成（★v7.3.5: .gitignore に testdata/generated/ 追加必要）
│   ├── dirac_test.wav
│   ├── lpf_1k_test.wav
│   ├── hpf_20_test.wav
│   ├── sine_*.wav
│   ├── logsweep_*.wav
│   ├── silence_*.wav
│   └── multitone_*.wav
├── golden/             # ゴールデン参照（Git LFS 管理推奨）
│   └── {pattern}/{tc}/
│       ├── output.wav
│       └── golden_metrics.json
└── config/
    └── test_config.yaml

sampledata/
└── impulse_room_correction_hpf_lpf.wav   # 実測IR（既存・実在確認済）
```

### 8.6 既存テストフレームワークとの統合

> **[v7.3.4 検証確定]** ConvoPeq のテストは **CMake + CTest** で構成される。
> Catch2 / doctest / gtest は不使用。各テストは独立した C++ 実行ファイルとしてコンパイルされ、
> `add_test(NAME ... COMMAND ...)` で CTest に登録される（`CMakeLists.txt:236-254`）。
>
> OutputCaptureSinkTests.cpp もこのパターンに従う:
> ```cmake
> # CMakeLists.txt に追加
> add_executable(OutputCaptureSinkTests src/tests/OutputCaptureSinkTests.cpp)
> target_link_libraries(OutputCaptureSinkTests ${COMMON_LIBS})
> add_test(NAME OutputCaptureSinkShutdownOrder COMMAND OutputCaptureSinkTests)
> ```
>
> Python テスト（analyzers.py / cli_runner.py 等）は別途 `pytest` で実行するか、
> CTest から `add_test(NAME ... COMMAND python3 -m pytest ...)` で呼び出す。

---

## Appendix A: 改訂履歴

| 版 | 日付 | 主要内容 | 備考 |
|---|------|---------|------|
| v1.0〜v6.3 | 2026-06-22 まで | 初期設計・ISR追加・RMAA統合・TC拡充・AES17・最終調整 | 凍結 |
| v6.3+ | 2026-06-22 | コード監査完了、Phase 0 着手確定 | 凍結 |
| **v7.0** | **2026-07-06** | **コード監査反映**: TC-30 API未実装確定、TC-01B参照パス修正、テストケース数34件修正 | 9件解決 |
| **v7.1** | **2026-07-06** | **RT安全性・責務分離**: OutputCaptureSink独立クラス化、SPSC+BG WAV、原子生ポインタ | +10件 |
| **v7.2** | **2026-07-06** | **二重管理解消・逐次WAV・seqlock**: CapturePoint統合、sequence lock、WAV逐次書込 | +9件 |
| **v7.3** | **2026-07-06** | **全指摘解決**: pushBecameNonEmpty(), WaitableEvent, シャットダウン順序, release/acquire, drain-all, パラメータパターン7種 | **全件解決** |
| **v7.3.1** | **2026-07-15** | **外部検証反映**: 音響工学公式確認（IEC 61672/AES17）、コードベース整合性確認（16項目）、未確定事項6件確定（AudioBlock特性実測/Seqlock single writer/WaitableEvent auto-reset/TC-34到達性）、工数修正（Phase 1: 45〜55→55〜70人日）、チェックリスト8項目追加 | **外部検証完了** |
| **v7.3.2** | **2026-07-15** | **二次深掘検証反映**: シャットダウンシーケンスの既存実コード統合（実際の`~MainWindow()` 10stepとの整合）、CLI未提供パラメータ6種の実装課題特定（OS/HC/LC/SoftClip/Saturation）、測定パターン数「7」→「10」修正、P1-Baseline TC一覧拡充、Blackman/Blackman-Harris窓混同の修正指示、alias計算例修正、`fetchAdd`→`fetchAddAtomic`統一、shutdown原子ポインタ新設場所の明記 | **二次検証完了・全指摘解決** |
| **v7.3.3** | **2026-07-15** | **三次DSPパイプライン検証反映**: `capture()`シグネチャを実コード（`const double*`）に統合、`pushBecameNonEmptyWithWriter()` ゼロコピー版の新設、`CapturePoint` と実パイプラインの対応表（PreFilter=oversampled `dsp::AudioBlock<double>`, PostFilter/PostDither=`double*`）、`analyze_alias()` 両側帯域(`k*sr±fin`)修正、TC-33評価方法統一、`analyze_thd_blackman_harris()` bin-sum化、WAV出力float32明記、`build_cli_args()` 6新規CLI追加 | **三次検証完了・全指摘解決（計56件）** |
| **v7.3.4** | **2026-07-15** | **四次統合検証反映**: `audioCaptureQueue` との統合戦略確定（独立キュー + ProcessingState 第二ポインタ）、`processOutputDouble()` パス対応明記、THD/THD+N 用語修正、§8 測定データ一覧・仕様・作成方法 新設（IR/入力信号/ゴールデン/閾値の完全仕様）、既存テストツールのハードコードパス問題（K4/K5）記載（v7.3.4 は計62件解決） | **四次検証完了** |
| **v7.3.5** | **2026-07-15** | **五次DSP最終検証反映**: capture() float overload（PostDither float*）、TC-31a 閾値緩和（RMS ≤ -138dBFS、DCBlocker常時動作理由）、analyze_sweep_transfer Farina 2007 inverse filter 方式（§8.2逆フィルタ信号生成＋§5.9数式完全定式化）、processOutputDouble PostDither 位置修正（L801/L803後）、--cli-output-wav/capture-mode MainWindow実装パターン詳細、testCaptureQueue RT-safe atomicロード仕様、thd_db 命名統一、--cli-order=conv ComboBoxマッピング明記、OS別IR要件、analyze_noise_psd実装完了、multitone AES17 random phase、analyze_alias FFT単発化、~MainWindow step 2 省略明記 | **五次検証完了・全指摘解決（計88件）** |
| **v7.3.6** | **2026-07-16** | **六次実装完全性検証反映**: M3 致命的OutputCaptureSinkメンバ完全性修正（sampleRateHz_/bitDepth_/coeffBankIndex_/sessionId_ 4変数の宣言+seter実装）、M5 既存TC16件のパターン未割り当て明示化（§5.2脚注）、M6 TC-04/04Aノイズシェイパ信号を§8.2信号表に追加、TC-17/18を§5.7テストケース表に追加、M11 TC-37 DoublePath PeakLimiter閾値超過问题（-0.1dBFS入力 > kOutputHeadroom=0.891、Float Path限定測定を明記）、M12 analyzers.pyにLPF/HPF/Allpass評価関数（analyze_lpf_hpf_response）+ Golden WAV比較関数（compare_golden_wav）を新規実装、analyze_alias docstringのstray comment修正、Appendix A/B/E.13/チェックリスト/件数更新（88→94件）| **六次検証完了・全指摘解決（計94件）** |

---

## Appendix B: 全指摘解決履歴

| # | 指摘 | 解決内容 | 版 |
|---|------|---------|----|
| C1 | OutputCapture RT安全性 | `std::function` → 原子生ポインタ | v7.0 |
| C2 | Audio Thread で WAV I/O | SPSC RingBuffer + BG Thread 分離 | v7.0 |
| C3 | 生 span 返却の競合 | `copyRecoveryHistorySnapshot()` | v7.0 |
| C4 | 行番号 L470 固定 | 処理段階 Stage で規定 | v7.0 |
| C5 | AudioEngine 肥大化 | `OutputCaptureSink` 独立クラス化 | v7.0 |
| C6 | CapturePoint 拡張性不足 | 3点 enum（Pre/Post/PostDither）| v7.0 |
| C7 | TC-25 例外条件不足 | OutputFilter 係数変更時追加 | v7.0 |
| C8 | Python 工数楽観的 | 35〜45日に修正 | v7.0 |
| C9 | RecoveryEvent 情報不足 | 8フィールド＋eventSequence | v7.0 |
| D1 | CaptureMode non-atomic | `std::atomic<CapturePoint>` 統合 | v7.1 |
| D2 | `juce::File` data race | 単純 `juce::File`＋固定契約 | v7.1→v7.3 |
| D3 | Timer drain 冗長 | BG Thread 直接 drain | v7.1 |
| D4 | CapturePoint/Mode 二重管理 | 単一 enum 統合 | v7.1 |
| D5 | RecoveryHistory acquire 不完全 | Sequence Lock 導入 | v7.1 |
| D6 | eventSequence = idx（0〜63） | 単調増加 counter（`fetchAddAtomic`）| v7.1 |
| D7 | WAV メモリ肥大化 | `AudioFormatWriter::writeFromFloatArrays()` 逐次書込 | v7.1 |
| D8 | AudioBlock 保証不足 | `static_assert` 追加 | v7.1 |
| D9 | Sink 寿命管理未明文化 | シャットダウン手順明文化 | v7.1 |
| E1 | Seqlock 常に奇数 | 偶数→奇数→偶数 の正規パターン | v7.2 |
| E2 | shared_ptr store/load 違反 | `juce::File` + 固定契約に単純化 | v7.2 |
| E3 | シャットダウン順序逆 | `setCaptureSink(nullptr)` を先に | v7.2 |
| E4 | signal_fence 能なし | → F1 で thread_fence も削除 | v7.2→v7.3 |
| E5 | wait(10ms) ポーリング | WaitableEvent + signal() 通知 | v7.2 |
| E6 | sizeof 固定は脆弱 | 削除（trivially_copyable のみ）| v7.2 |
| E7 | snapshot リトライなし | 8回リトライループ（Reader は BG Thread のため RT 非影響）| v7.2→v7.3 |
| E8 | float 変換バッファ毎回確保 | Thread 開始時に事前確保 | v7.2 |
| **F1** | **Seqlock thread_fence 不要** | **1st store(relaxed) → write → 2nd store(release)** | **v7.3** |
| **F2** | **ringBufferWasEmpty_ lost wake-up** | **`pushBecameNonEmpty()` 新設で TOCTOU 排除** | **v7.3** |
| **F3** | **memory order 説明不備** | **release/acquire + シャットダウン signal 明記** | **v7.3** |
| **F4** | **outputPath 契約曖昧** | **`startCapturing()` 以降の `setOutputPath()` 禁止** | **v7.3** |
| **G1** | **WaitableEvent auto-reset 未明記** | **JUCE 8 デフォルト `manualReset=false`（auto-reset）を明記。lost wake-up 不可を確認** | **v7.3.1** |
| **G2** | **Seqlock single writer 前提未明文化** | **`recordRecoveryAction()` を Timer Thread 単一呼出に制限。CAS 不要の根拠明確化** | **v7.3.1** |
| **G3** | **AudioBlock 特性未実測** | **g++ で実測: trivially_copyable/standard_layout/destructible 全て true。sizeof=4120, alignof=8** | **v7.3.1** |
| **G4** | **Reader コメント「Audio Thread」誤り** | **「Timer Thread」に修正（Writer は Timer Thread）** | **v7.3.1** |
| **G5** | **TC-34 閾値 -110dBFS 到達性未検証** | **32bit設定（P5）なら到達可能。16bit（P6）には割当禁止。実測で -105dBFS に緩和余地** | **v7.3.1** |
| **G6** | **analyzers.py エスケープ・未完成** | **`\"\"\"` → `"""` 修正指示、`analyze_noise_psd()` 戻り値プレースホルダ完成指示** | **v7.3.1** |
| **G7** | **Phase 1 工数楽観的** | **45〜55 → 55〜70人日に修正（45TC×1.2日/TC）** | **v7.3.1** |
| **G8** | **OutputCaptureSink 単体テスト未計画** | **`src/tests/OutputCaptureSinkTests.cpp` 新設をチェックリスト追加** | **v7.3.1** |
| **G9** | **`--cli-dump-filter-coeffs` getter 要件** | **OutputFilter の `hcCoeff`/`lcCoeff`/`hpfCoeff` は private。const getter 新設が必要** | **v7.3.1** |
| **H1** | **CLI未提供パラメータ6種（OS/HC/LC/SoftClip/Saturation）** | **Phase 1 で `--cli-os-type`/`--cli-os-factor`/`--cli-hc-mode`/`--cli-lc-mode`/`--cli-soft-clip`/`--cli-saturation` を実装。工数 +3人日** | **v7.3.2** |
| **H2** | **shutdown 13step が既存実コードと不整合** | **既存 `~MainWindow()` の実際の10step に OutputCaptureSink 停止段階を統合し 14step に修正。`outputCaptureSink_` は AudioEngine に新設** | **v7.3.2** |
| **H3** | **「7測定パターン」が実際に10パターン** | **見出し・表を「10測定パターン」に修正。P1/P2/P3 の TC一覧を §5.8 の追加TC と一致させる** | **v7.3.2** |
| **H4** | **Blackman vs Blackman-Harris 窓混同** | **`np.blackman()`(3項) と Blackman-Harris(3項/4項) の違いを注記。関数名と実装の整合指示** | **v7.3.2** |
| **H5** | **alias 計算例の誤り** | **OS=4x, fin=21kHz, alias=75kHz → OS=2x, fin=52kHz, alias=44kHz に修正。TC-33 は全OS倍率で Nyquist未満により alias無し=反証テストであることを明記** | **v7.3.2** |
| **H6** | **`fetchAdd` → `fetchAddAtomic` 未統一** | **`convo::fetchAddAtomic(std::atomic<T>&, T, memory_order)` に統一（`AtomicAccess.h:91` 既存 API との整合）** | **v7.3.2** |
| **H7** | **`--cli-capture-mode` 引数漏れ** | **PreOutputFilter (`pre-filter`) を追加。4値の enum と引数を一致** | **v7.3.2** |
| **H8** | **Phase 1 工数再修正** | **55〜70 → 58〜73人日に修正（CLIパラメータ拡張6種の +3人日を反映）** | **v7.3.2** |
| **I1** | **`capture()` シグネチャが実パイプラインと不整合** | **`juce::AudioBuffer<double>` → `const double* left, const double* right, int numSamples` に変更。実パイプラインは全て `double*`（alignedL/alignedR）または `juce::dsp::AudioBlock<double>`（非所有view）であり、`AudioBuffer<double>` は存在しない** | **v7.3.3** |
| **I2** | **PostDither キャプチャポイントの実装位置が未定義** | **`processOutput()` L503後・L505前（`applyFixedLatencyDelay` 後、float変換前）に挿入。§2.1 に CapturePoint ↔ 実パイプライン対応表を新設** | **v7.3.3** |
| **I3** | **`capture()` が3倍のメモリコピーを発生** | **`pushBecameNonEmptyWithWriter()` を新設（writer ラムダ版）。既存 `pushAdaptiveCaptureBlocks` の `pushWithWriter` と同一パターン。Audio Thread のメモリ書き込み量 4120byte×3 → 4120byte×1** | **v7.3.3** |
| **I4** | **`capture()` プレースホルダが `numSamples` 設定を欠落** | **writer ラムダ内で `block.numSamples = currentBlockSize` 等のメタデータ設定を必須化。未設定の場合、BG Thread が有効サンプル数 0 と誤認しデータロスト** | **v7.3.3** |
| **I5** | **`analyze_alias()` が上側帯域を評価漏れ** | **`abs(k*sr - fin)` のみ → `abs(k*sr - fin)` と `k*sr + fin` の両側帯域に修正。k range も `range(1, os_factor+1)` → `range(1, os_factor)`（k=L は DC と同一）。具体例: OS=4x, fin=21kHz で k=1 上側 69kHz が漏れていた** | **v7.3.3** |
| **J1** | **TC-33 記述と実装の不整合** | **「10kHz以下の折り返しエネルギー」→「理論イメージ位置（k・fs±fin）のエネルギー」に統合。TC-33 入力(19/20/22kHz)は全て Nyquist 未満のため alias 不発生=反証テスト** | **v7.3.3** |
| **J2** | **`analyze_thd_blackman_harris()` 単一bin抽出** | **±2 bin のパワー和（`bin_power()` 関数）を使用。Blackman窓の主ローブ幅≈3 bin で、単一 bin は約50%のエネルギーしか捕捉しない。THD 比は概ね補儮されるが精度向上のため** | **v7.3.3** |
| **J3** | **WAV出力フォーマット（float32）未明記** | **`convertBufferL_/convertBufferR_` は `float[]` であり、double→float32 変換で出力。TC-37「1 ULP」は float32 の ULP。TC-38(30分)のファイルサイズ ≈ 660MB** | **v7.3.3** |
| **J4** | **`build_cli_args()` に6新規CLIパラメータが未反映** | **§5.1 の6パラメータ（`--cli-os-type`, `--cli-os-factor`, `--cli-hc-mode`, `--cli-lc-mode`, `--cli-soft-clip`, `--cli-saturation`）を組み立てるコードを `build_cli_args()` に追加** | **v7.3.3** |
| **K1** | **OutputCaptureSink と既存 `audioCaptureQueue` の統合未定義** | **ProcessingState に `testCaptureQueue` ポインタを追加し、OutputCaptureSink のリングバッファを独立キューとして並列動作させる。CLI テストモードでは `adaptiveCaptureActiveRt=false` のため既存キューは非アクティブ** | **v7.3.4** |
| **K2** | **`processOutputDouble()` パスが計画書で未対応** | **`DSPCoreDouble.cpp:620` の `processOutputDouble()` も `pushAdaptiveCaptureBlocks` (L651) を呼ぶ。PostDither 挿入位置を Float path (L503後) と Double path (L782後) の両方に明記** | **v7.3.4** |
| **K3** | **THD と THD+N の用語混同** | **`analyze_thd_blackman_harris()` は高調波のみ抽出=THD。TC-03/TC-39 の「THD+N」を「THD」に修正、または別途 `analyze_thdn()` を実装** | **v7.3.4** |
| **K4** | **既存テストツールのハードコードパス** | **`create_dirac_ir.py` は `C:/TEMP/` に出力。`generators.py` は出力パスをパラメータ化し `testdata/generated/` に出力** | **v7.3.4** |
| **K5** | **`create_dirac_ir.py` が16bit PCM生成** | **テスト要件として float32 Dirac IR が必要。`generators.py::generate_synthetic_ir("dirac")` で float32 WAV を新規生成** | **v7.3.4** |
| **K6** | **測定データ一覧・仕様・作成方法が不在** | **§8 を新設: IR（4種）、入力信号（10種）、ゴールデン参照、閾値定義、ディレクトリ構成、テストフレームワーク統合を完全仕様化** | **v7.3.4** |
| **L1** | **PostDither データ型が float\* なのに plan は double\* と誤記** | **capture() に float\* overload を追加。§2.1 の CapturePoint 対応表を v7.3.3 から v7.3.5 に修正。PostDither は Float Path で `float* dstL/dstR`（float変換後）、Double Path で `double*`（変換なし）** | **v7.3.5** |
| **L2** | **testCaptureQueue RT-safe atomic ロード未定義** | **`buildAudioThreadProcessingState()` 内で `convo::consumeAtomic(outputCaptureSink_, acquire)` で 1回ロード。`setCaptureSink(ptr)` は `publishAtomic(..., release)` で書込** | **v7.3.5** |
| **L3** | **OS 別 IR tap 数が固定（129taps@48kHz）で OS=8x 時分解能不足の懸念** | **§8.1 に OS別 IR 要件を追記。既定では全OS倍率で base-rate IR を使用し、`processUp()` の補間結果で評価。Phase 1 で 8x レート IR 追加は任意** | **v7.3.5** |
| **L4+L23** | **analyze_sweep_transfer が log sweep に不適（FFT 比方式）+ Farina 逆フィルタ数学未定義** | **§5.9 を Farina 2007 inverse filter 方式に全面差し替え。§8.2 generate_log_sweep で逆フィルタ信号 `_inverse.wav` を同時生成。数式（時間反転 + 振幅補償エンベロープ exp(-t·ln(f₁/f₀)/T)）を完全定式化** | **v7.3.5** |
| **L5** | **multitone の random phase 未使用、AES17 非準拠** | **generate_multitone() に `np.random.RandomState(42)` で決定的ランダム位相を付与。AES17 準拠 random phase offset** | **v7.3.5** |
| **L6** | **analyze_alias が Welch 使用（ピークぼかし）** | **Welch → Hann-windowed FFT 単発に変更。エイリアスピーク検出精度向上** | **v7.3.5** |
| **L7+L17+L18** | **THD return key `thdn_db` が THD+N と混同、golden_metrics/threshold key 不一致** | **`thdn_db` → `thd_db` に統一。golden_metrics.json 形式例と threshold YAML の key も `thd` に統一** | **v7.3.5** |
| **L8** | **TC-31a 100% bit-exact は不可能（DCBlocker+NaN除去+Dither 常時動作）** | **TC-31a 閾値を「サンプル一致率 100%」→「差分RMS ≤ -138dBFS / Peak ≤ -128dBFS」に緩和。理由（DCBlocker IIR 状態残留 + Dither TPDF注入 + float丸め）を明記** | **v7.3.5** |
| **L9** | **--cli-order=conv が「Convolver-Only」なのに混乱しやすい** | **§5.1 に parseCliOrderMode() の CLI値→ComboBox ID→実動作 マッピング表を追加。`conv`=Convolver-Only(PEQ bypass)、`peq`=PEQ-Only(Conv bypass)、`conv->peq`=両方動作** | **v7.3.5** |
| **L10** | **analyze_sweep_transfer FFT 比方式は log sweep で誤差大** | **L4/L23 と同一対応。Farina 2007 inverse filter 方式に全面差し替え** | **v7.3.5** |
| **L11** | **processOutputDouble PostDither 位置「L782後」は誤り** | **§2.1 に正確位置を修正: applyFixedLatencyDelay(L801) + FloatVectorOperations::copy(L803-805) 後。データ型も Double Path では `double*`（float変換なし）と明記** | **v7.3.5** |
| **L12** | **Dirac IR amplitude 1.0 vs テスト信号 -6dBFS(0.5) の不一致** | **§8.1 に脚注追加。IR の `data[0]=1.0` は0dBFS、テスト入力信号は `amplitude=0.5`(-6dBFS)。線形スケーリングなので比較に問題なし** | **v7.3.5** |
| **L13** | **Blackman 窓 vs Blackman-Harris 窓名混同（v7.3.2 既知）** | **既に v7.3.2 で注記済み。`np.blackman(N)` と `np.blackmanharris(N)` は別物。実装時に `np.blackmanharris` に変更するか関数名を `analyze_thd_blackman` に改名。v7.3.5 で再確認** | **v7.3.5** |
| **L14** | **analyze_noise_psd がプレースホルダ(...)のまま実装未定義** | **§5.9 に実装設計を追加: A-weighting適用済みPSD積分(dBFS) + 5kHz以下帯域のlog-PSD RMS形状一致度。`weighted_psd = Pxx * a_weight**2`** | **v7.3.5** |
| **L15** | **§2.4 の14step に step 2(setAdaptiveAutosaveCallback) が省略** | **§2.4 に脚注追加。省略は意図的（OutputCaptureSink 停止と無関係のため）。明示的にメモとして残す** | **v7.3.5** |
| **L16** | **--cli-output-wav / --cli-capture-mode の MainWindow 実装方針不在** | **§2.5 に `runCommandLineAutomation()` 内の findValue パターン実装コード例を追加。setOutputPath→setCapturePoint→startCapturing→setCaptureSink のライフサイクル。`cliCaptureSink_` メンバ変数新設** | **v7.3.5** |
| **L19/L22** | **IR_FILE_MAP 相対パスと §8.5 ディレクトリ構成の不一致** | **軽微。generators.py で `Path("testdata/generated")` 絶対パスを使用し、IR_FILE_MAP も同様に修正可能** | **v7.3.5** |
| **L25** | **.gitignore に testdata/ 未設定** | **§8.5 ディレクトリ構成の `generated/` コメントに「.gitignore に testdata/generated/ 追加必要」と明記** | **v7.3.5** |
| **M1** | **build_cli_args の CLI 値マッピングが parser と不一致の可能性** | **parseCliPhaseMode()（36行）："asis"/"mixed"/"minimum" を確認、parseCliOrderMode()（60行）："conv"/"peq"/"convpeq"/"peqconv" を確認、parseCliNoiseShaper()（91行）："psycho"/"fixed4"/"adaptive"/"fixed15" を確認、parseCliDitherDepth()（117行）："16"/"24"/"32" を確認。全て既存実装とマッピング確認済み** | **v7.3.6** |
| **M2** | **Phase 1 追加タスク6種（OS/HC/LC/SoftClip/Saturation）の実装可否未検証** | **6パラメータ全てが既存 CLI オプションのパターン（findValue + 原子変数の setXXX）に従う。新規 API 要件なし。実装は技術的に可能** | **v7.3.6** |
| **M3** | **OutputCaptureSink メンバ変数 sampleRateHz_/bitDepth_/coeffBankIndex_/sessionId_ が capture() Lambda（L200-203）で使用されるが、クラス定義に宣言がない** | **OutputCaptureSink クラス定義に4変数（sampleRateHz_{48000.0}/bitDepth_{64}/coeffBankIndex_{0}/sessionId_{0}）を private 宣言に追加し、setAudioParams() public setter を追加。setAudioParams() は startCapturing() 前に呼ぶ。実装初日にビルドエラーとなる致命的问题** | **v7.3.6** |
| **M5** | **既存テストケース16件（TC-05A〜D, TC-08, TC-10, TC-11/11B/12/13/14, TC-16, TC-19/20, TC-22, TC-26, TC-28）が P1〜P10 パターンに割り当てられていない** | **§5.2 パターン表に脚注を追加: 16件は既存 CI テストフレームワーク（`src/tests/*.cpp`）で継続検証。新 CLI 自動化（Phase 1〜2）では対象としない** | **v7.3.6** |
| **M6** | **TC-04/04A ノイズシェイパテストの入力信号が §8.2 信号表に未定義。TC-17/18 が §5.7 テストケース表に未记载** | **§8.2 信号表の「Sine 1kHz -6dBFS」行の TC 欄に TC-04,TC-04A を追加（入力信号は TC-01/03 と同じ 1kHz -6dBFS。ノイズシェイパ設定違いをテストする）。§5.7 テストケース表に TC-17（SMPTE IMD 60Hz+7kHz 4:1）/TC-18（CCIF IMD 19kHz+20kHz 1:1）を追加。IMD 60Hz+7kHz 信号は §8.2 に定義済み** | **v7.3.6** |
| **M7** | **pushBecameNonEmptyWithWriter() が未実装** | **LockFreeRingBuffer.h に pushBecameNonEmptyWithWriter() を pushWithWriter() を拡張する形で実装可能（PushResult を返す writer ラムダ版）。§2.2 設計済み、実装は Phase 0 Day 1** | **v7.3.6** |
| **M8** | **Seqlock single-writer contract が文書上没有明確** | **Timer Thread だけが seqlock フィールド（RecoveryHistory 内の）に書込。Audio Thread（R/W）と BG Thread（R）の reader は atomic read のみで contract 成立。§2.1 と Appendix B D5 領域に single-writer 明記** | **v7.3.6** |
| **M9** | **Phase 0 9日工数が乐观的との指摘** | **OutputCaptureSink（SPSC + WAV 逐次書込）、CLI 拡張（10 新オプション）、ロックフリースレッドセーフ API の実装は DSP 专业知识が必要だが JUCE 既存 API 活用で実現可能。9日は DSP 経験ある開発者であれば妥当** | **v7.3.6** |
| **M10** | **§8.4 の閾値（thd: -80dB、imd: -80dB、noise_floor: -115dBFS、freq_response: 0.05dB）が物理的に到達可能か未検証** | **PsychoacousticDither 32bit設定のノイズフロア ≈ -144dBFS（float 量子化限界）、A-weighting 適用で -110dBFS 到達可能。THD -80dB は Audio Precision 測定器の標準性能。閾値は物理的に妥当** | **v7.3.6** |
| **M11** | **TC-37 入力 -0.1dBFS (amplitude ≈ 0.9886) が Double Path の kOutputHeadroom=0.891 を超過。PeakLimiter が常に動作して歪む** | **TC-37 行に注記追加: Double Path (processOutputDouble) では PeakLimiter が -0.1dBFS 入力で常に激活するため TC-37 は Float Path (processOutput) でのみ測定可能。Double Precision 設定では Float Path フォールバックを使用すること。P2-PEQ-Only は OS=1x+Float Path のため問題なし** | **v7.3.6** |
| **M12** | **LPF/HPF/Allpass 評価関数（analyze_lpf_hpf_response）および Golden WAV 比較関数（compare_golden_wav）が analyzers.py に未実装** | **analyzers.py の Analyzers クラスに analyze_lpf_hpf_response()（フィルタ種類判定・カットオフ点実測・ロールオフ勾配計算・形状RMS誤差算出）と compare_golden_wav()（振幅スペクトルRMS誤差・ピーク位置オフセット・エネルギー95%点オフセット・rms偏差dBFS・pass判定）を新規実装。§5.9 の Phase 1-2 実装対象として明記** | **v7.3.6** |

---

## Appendix C: 既存コードパターン活用マップ

| 設計要件 | 既存パターン | ファイル |
|---------|------------|---------|
| SPSC RingBuffer | `LockFreeRingBuffer<T, Capacity>` | `src/LockFreeRingBuffer.h` |
| Audio Thread→RingBuffer 書込 | `pushAdaptiveCaptureBlocks()` | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` |
| 非同期 drain | `asyncSink()` + `flushLogBuffer()` | `src/audioengine/AudioEngine.Timer.cpp` |
| capture 用 data struct | `AudioBlock`（double[256]×2ch, trivially_copyable）| `src/audioengine/AudioEngine.h:24` |
| atomic enum メンバ | `std::atomic<NoiseShaperType>`, `<HCMode>`, `<OversamplingType>` | `src/audioengine/AudioEngine.h` |
| 原子生ポインタ注入 | `atomic<OutputCaptureSink*>` | `RuntimeHealthMonitor::setRetireRouter()` 準拠 |
| 単調増加 counter | `audioCallbackEpochCounter`（`fetchAddAtomic`）| `AudioEngine.h:1482` |
| static_assert 群 | `DiagEvent`（5種）| `AudioEngine.h:447-460` |
| 逐次WAV書込 | `AudioFormatWriter::writeFromFloatArrays()` | JUCE modules |
| 段階的 shutdown | `~MainWindow()` step 1〜10 | `src/MainWindow.cpp` |
| RecoveryAction 実行（Timer Thread）| `executeRecoveryAction()` | `src/audioengine/AudioEngine.Timer.cpp:1591` ★ v7.3.1 追記 |
| WaitableEvent (auto-reset) | `juce::WaitableEvent` デフォルト構築 | `JUCE/modules/juce_core/threads/juce_WaitableEvent.h:59` ★ v7.3.1 追記 |
| shutdown 実コード | `~MainWindow()` step 1〜10 | `src/MainWindow.cpp:1000-1033` ★ v7.3.2 追記 |
| OSタイプ・倍率 API | `setOversamplingType()`, `setOversamplingFactor()` | `src/audioengine/AudioEngine.h:1392,1389` ★ v7.3.2 追記 |
| HC/LCモード API | `setConvHCFilterMode()`, `setConvLCFilterMode()` | `src/audioengine/AudioEngine.h:1401,1404` ★ v7.3.2 追記 |
| SoftClip/Saturation API | `setSoftClipEnabled()`, `setSaturationAmount()` | `src/audioengine/AudioEngine.h:1356,1359` ★ v7.3.2 追記 |
| fetchAddAtomic 実 API | `convo::fetchAddAtomic(std::atomic<T>&, T, memory_order)` | `src/audioengine/AtomicAccess.h:91` ★ v7.3.2 追記 |
| capture 実データフロー | `double* dataL/dataR`（alignedL/alignedR の別名）| `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:360-361` ★ v7.3.3 追記 |
| pushWithWriter ゼロコピーパターン | `pushWithWriter([&](AudioBlock& block){...})` | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:141-168` ★ v7.3.3 追記 |
| DSP処理ドメイン（oversampled） | `juce::dsp::AudioBlock<double> processBlock(channels, 2, numSamples)` | `DSPCoreFloat.cpp:242` / `DSPCoreDouble.cpp:391` ★ v7.3.3 追記 |
| AudioBlock メンバ | `L[256], R[256], numSamples, sampleRateHz, bitDepth, adaptiveCoeffBankIndex, sessionId` | `src/audioengine/AudioEngine.h:24-32` ★ v7.3.3 追記 |

---

## Appendix D: コード監査使用ツール

| カテゴリ | ツール | バージョン | 用途 |
|---------|-------|-----------|------|
| **WSL検索** | grep (GNU) | — | 全文検索 |
| | ripgrep (rg) | 15.1.0 | 高速パターン検索 |
| | ast-grep | 0.44.0 | AST構造検索 |
| | fd (fdfind) | — | ファイル検索 |
| | fzf | — | ファジーファインダー |
| | sed / awk | 4.9 / 5.3.2 | テキスト処理 |
| **MCP** | AiDex MCP | 2.2.2 | コードベース検索・セマンティック検索 |
| | Serena MCP | 1.5.3 | プロジェクト管理・コード探索 |
| **CLI** | semble | — | 意味検索 |
| | cocoindex-code (ccc) | — | インデックス作成・意味検索 |
| | graphify | 0.9.7 | 知識グラフ操作 |

---

## Appendix E: 外部検証レポート（v7.3.1: 2026-07-15）

### E.1 検証手法

| 項目 | 手法 |
|------|------|
| コードベース整合性 | 実ソースコードの読み込み・grep 検索（16項目） |
| 型特性 | g++ -std=c++17 によるコンパイル時実測（AudioBlock 型特性） |
| 音響工学文献 | IEC 61672-1:2013（A-weighting）、AES17（THD+N/FFT）、Wikipedia 英語版による公式照合 |
| JUCE API 仕様 | `JUCE/modules/juce_core/threads/juce_WaitableEvent.h` ソース直接確認 |

### E.2 科学的・技術的妥当性（音響工学文献に基づく）

| 項目 | 評価 | 根拠 |
|---|---|---|
| **A-weighting 数式** | ✅ 正確 | IEC 61672-1:2013 の R_A(f) 式と完全一致。12194²・20.6・107.7・737.9 の定数も正規（§5.9 `A_weighting()` は IEC 公式と同一） |
| **THD+N 定義** | ✅ 正確 | THD_N = (高調波+ノイズ)/基本波。§5.9 `analyze_thd_blackman_harris()` は高調波2〜10次のパワー和/基本波パワーで計算 — THD_F 定義に適合 |
| **Blackman-Harris FFT vs Welch** | ✅ 適切 | Welch平均は高調波をぼやかす。AP System One/AES17 準拠の単一FFT + BH窓は業界標準。N=262144 (4.4s@48k) は AES17 推奨に準拠 |
| **Null Test** | ✅ 妥当 | 入力をバイパスし差分評価は業界必須試験。RMS ≤ -145dBFS は FMA 最適化差を許容する適切な閾値 |
| **Alias Rejection** | ✅ 設計思想は妥当 | 入力周波数から理論Alias位置を計算し近傍のみ評価 — 折り返し帯域全体評価の陥穽を正しく回避 |
| **TC-34 A-weighted ≤ -110dBFS** | ✅ 到達可能（32bit設定時）| PsychoacousticDither 32bit設定で控えめディザ。float パイプラインのノイズフロア ≈ -144dBFS。A-weighting で更に低減 |
| **RMAA 互換レポート** | ✅ 妥当 | 主要7測定項目（THD/IMD/FreqResp/Noise/Dynamic/Crosstalk/Stereo）は RMAA suite の体系と整合 |

### E.3 コードベース整合性確認（16項目）

| 計画書記載 | 実在確認 | 備考 |
|---|---|---|
| `LockFreeRingBuffer<T, Capacity>` | ✅ `src/LockFreeRingBuffer.h:24` | SPSC・alignas(64)・acquire/release。`pushBecameNonEmpty()` 追加は既存 `push()` と整合 |
| `AudioBlock` (double[256]×2ch) | ✅ `src/audioengine/AudioEngine.h:24-32` | trivially_copyable/standard_layout 実測確認（§4 参照） |
| `DiagEvent` static_assert 群 | ✅ `AudioEngine.h:447-461` | 5種の static_assert + sizeof==88 保証 |
| `pushAdaptiveCaptureBlocks()` | ✅ `AudioEngine.Processing.DSPCoreIO.cpp:122`, `DSPCoreDouble.cpp:269` | 既存 capture 実装箇所 |
| `asyncSink()` + `flushLogBuffer()` | ✅ `AudioEngine.Timer.cpp:80,98` | BG Thread 直接 drain の既存パターン |
| `setConvolverEnableProgressiveUpgrade(bool)` | ✅ `AudioEngine.h:1305`, `Parameters.cpp:713` | `--cli-progressive-upgrade` は既存APIを呼ぶ |
| `requestConvolverPreset(File)` | ✅ `AudioEngine.h:1234`, `Parameters.cpp:197` | `--cli-ir` 既存実装の土台 |
| `setCliProcessingTelemetryEnabled()` | ✅ `AudioEngine.h:1713` | シャットダウン順序ステップ3に対応 |
| `audioCallbackEpochCounter` (fetchAdd) | ✅ `AudioEngine.h:1544` | eventSequence 単調増加 counter のパターン踏襲 |
| `OutputFilter` (private BiquadCoeff) | ✅ `OutputFilter.h:94-145` | `hcCoeff`/`lcCoeff`/`hpfCoeff` は private → **const getter 新設が必要** |
| `sampledata/impulse_room_correction_hpf_lpf.wav` | ✅ 実在 | TC-01B 参照パス正確 |
| `tools/diagnostics/create_dirac_ir.py` | ✅ 実在 | 16bit/48k/mono。`generate_synthetic_ir` が活用可能 |
| `compare_dirac/compare_raw/analyze_conv_output` | ✅ 全て実在 | §5.9 の「既存ツール活用」方針は妥当 |
| `executeRecoveryAction()` | ✅ `AudioEngine.Timer.cpp:1591` | Timer Thread から呼出 → Seqlock single writer 成立 |
| `PolicySource` enum (10種) | ✅ `RuntimePolicyEngine.h:28-40` | 10分類確認。RetireStall〜SafeModeState |
| `setCaptureSink`/`OutputCaptureSink` | ⚠️ 未実装（Phase 0 新規） | `MainWindow.cpp` に0件。新規実装対象として整合 |

### E.4 未確定事項の確定結論

| # | 未確定事項 | 確定結論 | 根拠 |
|---|---|---|---|
| 1 | AudioBlock standard_layout | **✅ 確定: 全て true** | g++ 実測: `is_trivially_copyable=1`, `is_standard_layout=1`, `is_trivially_destructible=1`。`is_trivial=0`（デフォルト初期化子）は問題なし |
| 2 | Seqlock single writer | **✅ 確定: Timer Thread 単一** | `executeRecoveryAction()` は `AudioEngine.Timer.cpp:1591` に存在。Timer Thread から呼出される構造。`recordRecoveryAction()` をこの関数に組み込むことで single writer 成立。CAS 不要の主張妥当 |
| 3 | JUCE WaitableEvent 仕様 | **✅ 確定: auto-reset（デフォルト）** | JUCE 8 `juce_WaitableEvent.h:59`: `explicit WaitableEvent(bool manualReset = false)`。signal() は `triggered=true`、wait() は即 return して `triggered=false`。lost wake-up 不可。drain-all 設計と完全整合 |
| 4 | OutputCaptureSink 単体テスト | **✅ 確定: `src/tests/OutputCaptureSinkTests.cpp` 新設** | 既存16テストファイル（`src/tests/*.cpp`）と同等の構成。シャットダウン順序・リング満杯・drain-all を検証 |
| 5 | TC-34 -110dBFS 到達性 | **✅ 確定: 32bit設定で到達可能** | PsychoacousticDither 32bit設定は控えめディザ。float パイプライン ≈ -144dBFS。TC-34 は P5-ReferenceQuality(32bit) のみに割当。実測で -105dBFS に緩和余地 |
| 6 | analyzers.py エスケープ | **✅ 確定: 実装時修正事項** | `\"\"\"` → `"""` 修正、`analyze_noise_psd()` 戻り値プレースホルダ完成。設計意図は明確 |

### E.5 実装前必須対応事項サマリ

1. **WaitableEvent**: デフォルト構築（auto-reset）で使用することをコードコメントで明記（§2.2）
2. **Seqlock**: `recordRecoveryAction()` を `executeRecoveryAction()`（Timer.cpp:1591）に組み込み、Timer Thread からのみ呼ぶこと（§3.2）
3. **AudioBlock**: `sizeof==4120` の固定値に依存するコードを書かないこと（§4）
4. **TC-34**: P5-ReferenceQuality(32bit) のみに割当てること（§5.7）
5. **analyzers.py**: エスケープ修正・PSD戻り値完成（§5.9）
6. **OutputCaptureSink 単体テスト**: シャットダウン順序・リング満杯・drain-all を検証すること（§7 #26）
7. **`--cli-dump-filter-coeffs`**: OutputFilter に const getter を新設すること（§7 #27）

### E.6 検証ツール・文献

| カテゴリ | 名称 | 版/URL | 用途 |
|---------|------|--------|------|
| 規格 | IEC 61672-1:2013 | — | A-weighting 周波数補正曲線の公式定義 |
| 規格 | AES17-2020 | — | デジタルオーディオ測定法（THD+N, FFT窓） |
| 文献 | Wikipedia "A-weighting" | 2026-04-26 UTC revision | R_A(f) 公式の照合 |
| 文献 | Wikipedia "Total harmonic distortion" | 2026-04-16 UTC revision | THD+N 定義の照合 |
| JUCE API | `juce_WaitableEvent.h` | JUCE 8（juce-8-licence） | WaitableEvent auto-reset 仕様確認 |
| コンパイラ | g++ (GCC) | -std=c++17 | AudioBlock 型特性の実測 |
| 検索 | ripgrep / ast-grep / fd | 15.1.0 / 0.44.0 / — | コードベース統合性確認 |

---

### E.7 二次深掘検証で新たに発見された問題（v7.3.2: 2026-07-15）

| # | 問題 | 詳細 | 影響 | 解決 |
|---|---|---|---|---|
| 1 | shutdown 13step が既存実コードと不整合 | `~MainWindow()` 実際は10step（`MainWindow.cpp:1000-1033`）。計画書のstep順序（removeChangeListener→setTelemetry→cliautomation）が実装（removeChangeListener→setAdaptiveAutosave→setTelemetry→setProcessor→...）と異なる | OutputCaptureSink 停止の挿入位置がずれる。重大 | 既存の実際の10stepに統合し14stepに修正。`mmcssShutdownRequested` はAudioEngine内部管理のため削除 |
| 2 | CLI未提供パラメータ6種 | OSタイプ/OS倍率/HC/LC/SoftClip/Saturation のAPIは実在するが CLIオプション無し。`cli_runner.py` の `build_cli_args()` から渡せない | P1〜P10全パターンでデフォルト値以外をテスト不可。重大 | Phase 1 で6種の新規CLI追加（+3人日）。Phase 0 ではデフォルト値のみテスト |
| 3 | 「7測定パターン」が実際には10パターン | P8-MinimumPhase/P9-AsIsPhase/P10-MixedPhase が表に存在するが見出しが「7」 | ドキュメントの一貫性欠如。中程度 | 見出し修正、P1/P2/P3 の TC一覧を §5.8 と一致させる |
| 4 | Blackman vs Blackman-Harris 窓混同 | `np.blackman(N)`(3項, a=0.16) を使用するが関数名が `blackman_harris`。Blackman-Harris 4項(a0=0.35875,a1=0.48829,a2=0.14128,a3=0.01168) とは別物 | THD+N 測定値にわずかな差（数dB以内）。低 | 注記追加。`np.blackmanharris(N)` への変更を実装時判断 |
| 5 | alias 計算例の誤り | 「OS=4x, fin=21kHz→alias=75kHz」は誤り。4x では Nyquist=96kHz であり 21kHz は Nyquist 未満で alias 不発生 | 理解の混乱。低 | 修正例: OS=2x, fin=52kHz→alias=44kHz。TC-33 は反証テストであることを明記 |
| 6 | `fetchAdd` → `fetchAddAtomic` 未統一 | `AtomicAccess.h:91` の既存APIは `convo::fetchAddAtomic(std::atomic<T>&, T, memory_order)`。計画書では単に `fetchAdd` と表記 | 実装時の名前解決混乱。低 | `convo::fetchAddAtomic` に統一 |
| 7 | `--cli-capture-mode` 引数漏れ | `CapturePoint::PreOutputFilter` (=1) に対応する `pre-filter` 引数が記載漏れ | 軽微 | 引数リストに `pre-filter` 追加 |
| 8 | `outputCaptureSink_` の新設場所未指定 | 計画書では `setCaptureSink(nullptr)` と呼ぶが、この atomic ポインタをどこに持つか未指定 | AudioEngine か MainWindow。中程度 | AudioEngine に `std::atomic<OutputCaptureSink*> outputCaptureSink_{nullptr}` を新設。`setCaptureSink(ptr)` 経由で注入 |

### E.8 六次実装完全性検証 総合評定（v7.3.6: 2026-07-16）

**Phase 0 実装開始可**。全未確定事項は確定済み、全指摘（C1〜M12, 合計94件）は解決済み。

v7.3.6 で発見・解決した6件の M-series 指摘:
- **M3 [致命的]**: OutputCaptureSink 缺少 `sampleRateHz_`/`bitDepth_`/`coeffBankIndex_`/`sessionId_` の4変数宣言+seter。実装初日にビルドエラーになる問題。**即座に解決（§2.2 に追加宣言+seter実装済み）**
- **M11 [中]**: TC-37 の -0.1dBFS 入力が Double Path の PeakLimiter 閾値(kOutputHeadroom=0.891)を超過。Float Path 限定測定を明記
- **M12 [中]**: `analyzers.py` に `analyze_lpf_hpf_response()` + `compare_golden_wav()` を新規実装
- **M5/M6 [中]**: 既存TC16件のパターン未割り当てを脚注明示、TC-04/04A 信号追加、TC-17/18 を §5.7 に追加

Phase 1 の CLI パラメータ拡張6種（OS/HC/LC/SoftClip/Saturation）を実施するまでは、
P1-Baseline 以外のパターンでこれらのパラメータを既定値から変更できない制約がある。
Phase 0 のテストは全パラメータ既定値の P1-Baseline と、既存CLIで設定可能なパラメータのみの
P2〜P4 に限定する必要がある。

### E.9 三次深掘検証で新たに発見された問題（v7.3.3: 2026-07-15）

> 三次検証は DSP パイプラインの**実データフロー**（`processOutput()` の全行程）を追跡し、
> 計画書の API シグネチャ・データ型・メモリコピーパターンが実コードと整合するかを検証した。

| # | 問題 | 詳細 | 影響 | 解決 |
|---|---|---|---|---|
| I1 | `capture()` シグネチャが実パイプラインと不整合 | 計画書は `const juce::AudioBuffer<double>&` だが、実パイプラインの全CapturePoint でデータは `double*`（生ポインタ）または `juce::dsp::AudioBlock<double>`（非所有view）。`AudioBuffer<double>` はどこにも存在しない | **致命的**: 実装着手直後に型不整合で壁にぶつかる | `const double* left, const double* right, int numSamples` に変更（§2.2） |
| I2 | PostDither キャプチャポイントの挿入位置が未定義 | PostDither は `applyFixedLatencyDelay` 後・float変換前（L503-505間）だが、その時点のデータは `double* dataL/dataR`（生ポインタ）。§2.1 に対応表なし | **致命的**: 実装時にどこに挿入すべきか不明 | §2.1 に CapturePoint ↔ 実パイプライン対応表を新設（§2.1） |
| I3 | `capture()` が3倍のメモリコピーを発生 | 計画書は `AudioBlock block{}` ローカル変数（4120B ゼロ初期化）+ コピー（4120B）+ `push()` コピー（4120B）= 12360B 書込。既存コードは `pushWithWriter()` でスロット直接書込（4120B×1）= **3倍の差** | **高**: Audio Thread の RT 安全性に悪影響 | `pushBecameNonEmptyWithWriter()` を新設し writer ラムダで直接スロット書込（§2.2, §2.3） |
| I4 | `capture()` プレースホルダが `numSamples` 設定を欠落 | `// ... buffer データを block に詰める ...` は `block.numSamples = currentBlockSize` の設定を含まない。既存コードは L143 で明示的に設定。未設定の場合 `numSamples` はゼロ初期化の 0 のままで、BG Thread が有効サンプル数 0 と誤認 | **高**: 最終ブロック（256未満）でデータロスト | writer ラムダ内でのメタデータ設定を必須化（§2.2） |
| I5 | `analyze_alias()` が上側帯域を評価漏れ | `abs(k*sr - fin)` は下側帯域のみ。上側帯域 `k*sr + fin`（例: OS=4x, fin=21kHz, k=1 → 69kHz）を見逃す。また `range(1, os_factor+1)` は k=L を含むが DC と同一のため不要 | **高**: イメージエネルギーを過小評価 | 両側帯域 `for f_alias in [abs(k*sr-fin), k*sr+fin]`、`range(1, os_factor)` に修正（§5.9） |
| J1 | TC-33 記述と `analyze_alias()` 実装の不整合 | 記述は「10kHz以下の折り返しエネルギー」、実装は「理論Alias位置のエネルギー」。入力19/20/22kHzは全てNyquist未満でalias不発生=反証テスト | **中**: テスト意図の混同 | TC-33 評価方法を「理論イメージ位置（k・fs±fin）のエネルギー」に統合（§5.7） |
| J2 | `analyze_thd_blackman_harris()` 単一bin抽出 | `np.abs(X[fund_bin * h])**2` で単一 FFT bin からパワー抽出。Blackman窓の主ローブ幅は約3 bin で、単一 bin は約50%のエネルギーしか捕捉しない | **中**: THD 比は概ね補償されるが精度不足 | ±2 bin のパワー和 `bin_power()` 関数を導入（§5.9） |
| J3 | WAV出力フォーマット（float32）未明記 | `convertBufferL_/convertBufferR_` は `float[]` で double→float32 変換するが、出力WAVの精度が文書化されていない。TC-37「1 ULP」が double か float32 か不明 | **中**: TC-37 の解釈に曖昧さ | float32（24bit有効）を明記。TC-37「1 ULP」= float32 ULP。TC-38(30分)≈660MB（§2.2） |
| J4 | `build_cli_args()` に6新規CLIパラメータが未反映 | §5.1 で追加された6パラメータ（`--cli-os-type` 等）の組み立てコードが `build_cli_args()` にない。実行時にこれらのパラメータが ConvoPeq に渡らない | **中**: P5/P6/P7 パターンでOS倍率等が変更できない | 6パラメータの組み立てコードを `build_cli_args()` に追加（§5.3） |

### E.10 確認済み項目（三次検証・問題なし）

| 項目 | 結果 | 根拠 |
|---|---|---|
| `pushAdaptiveCaptureBlocks` の既存パターン | ✅ 統合済 | `DSPCoreIO.cpp:122-174`: `pushWithWriter()` + AVX2 `_mm256_loadu_pd/storeu_pd` |
| `processOutput()` のデータフロー | ✅ 全行程追跡完了 | L342-514: DC Blocker → NaN sanitization → pushAdaptiveCaptureBlocks → Dither → NaN → fixedLatencyDelay → float変換 |
| `dsp::AudioBlock<double>` 使用箇所 | ✅ 処理ドメインと一致 | `DSPCoreFloat.cpp:242`, `DSPCoreDouble.cpp:391`（oversampled domain） |
| AudioBlock 全メンバ | ✅ 7フィールド確認 | `AudioEngine.h:24-32`: L[256], R[256], numSamples, sampleRateHz, bitDepth, adaptiveCoeffBankIndex, sessionId |
| LockFreeRingBuffer API | ✅ push/pushWithWriter/pop/size/clear | `LockFreeRingBuffer.h:33-88`: SPSC, acquire/release, alignas(64) |
| Seqlock release/acquire 妥当性 | ✅ 正しい | 1st store(relaxed)→write→2nd store(release) は single-writer seqlock として正規 |
| kRingCapacity=4096 の容量 | ✅ 妥当 | 21.8秒のバックログ。連続排出設計のため30分録音も可能（660MB WAV） |

### E.11 四次統合検証で新たに発見された問題（v7.3.4: 2026-07-15）

> 四次検証は**既存キャプチャ機構との統合**（`audioCaptureQueue` / NoiseShaperLearner）、
> **double 精度パイプライン**（`processOutputDouble()`）、**測定データ資産の完全仕様化**に焦点を当てた。

| # | 問題 | 詳細 | 影響 | 解決 |
|---|---|---|---|---|
| K1 | OutputCaptureSink と既存 `audioCaptureQueue` の統合未定義 | `AudioEngine.h:2446` に既存 `LockFreeRingBuffer<AudioBlock, 4096> audioCaptureQueue` が存在。NoiseShaperLearner が消費 (`Parameters.cpp:423`)。`ProcessingState.adaptiveCaptureQueue` が条件付きで参照 (`L3657`)。OutputCaptureSink とこの既存機構の関係が未定義 | **致命的**: 二重プッシュ・競合・データ汚染のリスク | ProcessingState に独立した `testCaptureQueue` ポインタを追加。CLI テストモードでは `adaptiveCaptureActiveRt=false` で既存キューは非アクティブ。並列干渉なし（§2.2） |
| K2 | `processOutputDouble()` パスが計画書で未対応 | `DSPCoreDouble.cpp:620-789` に double 精度出力パスが存在。`pushAdaptiveCaptureBlocks` (L651) を呼出。TruePeak/LUFS/PeakLimiter を含む (`L741-753`)。計画書は `processOutput()` (DSPCoreIO.cpp) のみ対象 | **高**: Double 精度ビルドでキャプチャが動作しない | Float path と Double path の両方に PostDither キャプチャ挿入を明記（§2.1） |
| K3 | THD と THD+N の用語混同 | `analyze_thd_blackman_harris()` は高調波のみ（THD_F）を計算するが、TC-03/TC-39 の記述は「THD+N」。THD+N は基本波をノッチで除去した残渣全量であり、FFT bin 抽出とは異なる手法 | **中**: 測定値の解釈にズレ | `analyze_thd_blackman_harris()` の docstring を「THD」に修正。THD+N が必要な場合は `analyze_thdn()` 別途実装（§5.9） |
| K4 | 既存テストツールのハードコードパス | `create_dirac_ir.py` は `C:/TEMP/` に出力。`create_test_irs.py` も `C:/TEMP/`。`generators.py` がこれらを踏襲するとポータビリティなし | **低**: CI 環境でパス問題 | `generators.py` は出力パスをパラメータ化、`testdata/generated/` に出力（§8.1） |
| K5 | `create_dirac_ir.py` が 16bit PCM 生成 | Dirac IR が 16bit 整数。テスト要件は float32（24bit 有効仮数）。16bit では量子化ノイズフロアが -96dB で TC-34/TC-37 の閾値に到達不可 | **低**: 既存ツールの直接流用不可 | `generators.py::generate_synthetic_ir("dirac")` で float32 WAV を新規生成（§8.1） |
| K6 | ProcessingOrder enum と CLI ComboBox IDs の非一致 | `Types.h`: ProcessingOrder{ConvolverThenEQ=0, EQThenConvolver=1}（2値）。`parseCliOrderMode()`: "conv"→1, "peq"→2, "convpeq"→3, "peqconv"→4（ComboBox IDs、4値）。マッピング自体は正しいが計画書に未記載 | **情報**: 実装時の混同リスク | 本件で文書化完了。`--cli-order` の引数仕様は enum 値ではなく文字列（conv/peq/convpeq/peqconv）であることを明記 |

### E.12 五次DSP最終検証で新たに発見された問題（v7.3.5: 2026-07-15）

> 五次検証は**DSPパイプラインの最終データ型確認**（PostDither float\* vs double\*）、
> **bit-exact可能性の物理的検証**（DCBlocker常時動作）、**対数スイープ伝達関数の数学的妥当性**
> （Farina 2007 inverse filter）、**CLI実装パターンの完全定義**に焦点を当てた。

| # | 問題 | 詳細 | 影響 | 解決 |
|---|---|---|---|---|
| L1 | PostDither データ型が plan と不一致 | `processOutput()` L505-510 で `static_cast<float>` 変換後、データは `dstL[i]/dstR[i]` (`float*`) に格納。plan §2.1 は `double* dataL/dataR` と誤記 | **致命的**: 実装初日のビルドエラー。capture() が float\* を受け取れない | capture() に `const float*` overload を追加（§2.1）。PostDither は Float Path で `float*`、Double Path で `double*` |
| L2 | testCaptureQueue RT-safe atomic ロード未定義 | K1 で `testCaptureQueue` ポインタを ProcessingState に追加したが、`buildAudioThreadProcessingState()` での読み出し方式（acquire/release）が未定義 | **高**: Audio Thread での data race リスク | `convo::consumeAtomic(outputCaptureSink_, acquire)` で1回ロード。`setCaptureSink()` は `publishAtomic(..., release)`（§2.1） |
| L3 | OS 別 IR tap 数が固定 | `lpf_1k`/`hpf_20` は 129taps@48kHz。OS=8x 時（384kHz処理）に `processUp()` の補間で IR 忠実度不足の懸念 | **中**: OS=8x での伝達関数精度 | 既定では全OS倍率で base-rate IR を使用。高精度比較時は `dirac` IR (8192samples) を使用（§8.1） |
| L4+L23 | analyze_sweep_transfer が log sweep に不適 + Farina 数式未定義 | `H = FFT(output)/FFT(input)` は log sweep で誤差大。Farina 2007 では逆フィルタ畳み込みで IR を抽出後 FFT で H(f) を計算。逆フィルタの振幅補償エンベロープ `exp(-t·ln(f₁/f₀)/T)` が未定義 | **致命的**: 全 golden 基準が破綻 | §5.9 を Farina 方式に全面差し替え。§8.2 で逆フィルタ信号 `_inverse.wav` を同時生成（§5.9, §8.2） |
| L5 | multitone random phase 未使用 | `generate_multitone()` は全トーン同位相で重畳。AES17 は random phase offset を推奨 | **低**: peak が N×amplitude になりクリップリスク | `np.random.RandomState(42)` で決定的ランダム位相を付与（§8.2） |
| L6 | analyze_alias の Welch 使用 | `signal.welch()` の平均化でエイリアスピークがぼける。ピーク検出には FFT 単発が適切 | **低**: エイリアスエネルギー過小評価 | Hann-windowed FFT 単発に変更（§5.9） |
| L7+L17+L18 | THD return key `thdn_db` と golden/threshold key 不一致 | `analyze_thd_blackman_harris()` は THD_F（高調波のみ）を計算するが、return key が `thdn_db`（THD+N と混同）。golden_metrics.json と threshold YAML の key も `thdn` | **中**: metrics 比較時の key 衝突・混乱 | `thdn_db` → `thd_db` に統一。golden/threshold key も `thd` に統一（§5.9） |
| L8 | TC-31a 100% bit-exact 不可能 | ConvoPeq パイプラインは全 bypass 時でも DCBlocker（IIR HPF 状態残留）+ NaN除去 + Dither（TPDF注入）+ float丸め が常時動作。100%サンプル一致は物理的不可能 | **致命的**: 最初のテスト実行で失敗、設計ミスと誤認 | TC-31a 閾値を「サンプル一致率 100%」→「差分RMS ≤ -138dBFS / Peak ≤ -128dBFS」に緩和（§5.7） |
| L9 | --cli-order=conv の意味が混同しやすい | `conv` = ComboBox ID 1 = `setConvolverBypass(false) + setEqBypass(true)` = **Convolver-Only** (PEQ bypass)。plan にこの意味が未明記 | **中**: P2-PEQ-Only テストで `--cli-order=peq` を使うべきところを `conv` と間違えるリスク | §5.1 に parseCliOrderMode() の CLI値→ComboBox ID→実動作 マッピング表を追加 |
| L10 | analyze_sweep_transfer FFT 比方式 | L4/L23 と同一問題。log sweep における FFT 比は周波数依存エネルギー密度ムラで誤差大 | **致命的**: TC-32/35 の伝達関数測定が不正確 | L4/L23 と同一対応（Farina 2007 inverse filter） |
| L11 | processOutputDouble PostDither 位置「L782後」は誤り | v7.3.4 で「L782後・L784前」と記述したが、実コードは `applyFixedLatencyDelay` (L801) + `FloatVectorOperations::copy` (L803-805) 後。Double Path では float 変換なしで `double*` のまま buffer にコピー | **高**: Double Path で PostDither キャプチャ位置が間違っている | §2.1 に正確位置を修正。Double Path では `double*`（float変換なし）と明記 |
| L12 | Dirac IR amplitude 1.0 vs テスト信号 -6dBFS(0.5) | IR の `data[0]=1.0` (0dBFS) だが TC-31a/41 の入力は -6dBFS (0.5)。振幅スケールが異なる | **低**: 線形スケーリングなので比較に問題なし | §8.1 に脚注追加。IR と入力信号の振幅仕様を明記 |
| L13 | Blackman 窓 vs Blackman-Harris 窓名混同 | `np.blackman(N)` (3項, a=0.16) と `np.blackmanharris(N)` (4項) は別物。関数名 `analyze_thd_blackman_harris` だが実装は `np.blackman` | **低**: サイドローブ減衰量の差（数dB以内） | v7.3.2 で既注記。実装時に `np.blackmanharris` に変更するか関数名を `analyze_thd_blackman` に改名 |
| L14 | analyze_noise_psd プレースホルダ(...) | `a_weighted_db: ..., psd_shape_rms: ...` の実装が未定義 | **中**: TC-34 ノイズ測定が実行不可 | §5.9 に実装設計を追加: A-weighting適用PSD積分(dBFS) + 5kHz以下 log-PSD RMS 形状一致度 |
| L15 | §2.4 の14step に step 2(setAdaptiveAutosaveCallback) が省略 | 実コード ~MainWindow() は10step。plan 統合版は14step だが step 2 が省略されている | **低**: 省略は意図的だが明記なし | §2.4 に脚注追加。省略理由（OutputCaptureSink 停止と無関係）を明記 |
| L16 | --cli-output-wav/capture-mode の MainWindow 実装方針不在 | §2.5 に5新規CLIを列挙するのみで、`runCommandLineAutomation()` 内の実装コードパターンが未定義 | **致命的**: Phase 0 実装開始時に手が止まる | §2.5 に findValue パターン実装コード例を追加。setOutputPath→setCapturePoint→startCapturing→setCaptureSink のライフサイクル |
| L19/L22 | IR_FILE_MAP 相対パスと §8.5 ディレクトリ構成の不一致 | IR_FILE_MAP は `"generated/dirac_test.wav"`（プレフィックスなし）、§8.5 は `testdata/generated/` | **低**: generators.py で絶対パス解決可能 | generators.py で `Path("testdata/generated")` を使用し一貫性確保 |
| L25 | .gitignore に testdata/ 未設定 | §8.5 は `generated/` を .gitignore 対象と推奨するが .gitignore には未設定 | **低**: 生成ファイルが誤って commit されるリスク | §8.5 の `generated/` コメントに「.gitignore に testdata/generated/ 追加必要」と明記 |

---

### E.13 六次実装完全性検証で新たに発見された問題（v7.3.6: 2026-07-16）

> 六次検証は**実装初日のビルドエラーを引き起こす致命的欠陥（M3）の発見**、
> **TC-37 Double Path 制約（M11）**、**既存 TC のパターン未割り当て（M5）**、
> **TC-04/04A 信号定義漏落（M6）**、**analyzers.py 缺失関数（M12）** に焦点を当てた。
> M3 は実装初日に必ず発見される欠陥であり、v7.3.6 で事前に計画書に完全反映した。

| # | 問題 | 詳細 | 影響 | 解決 |
|---|---|---|---|---|
| M1 | build_cli_args CLI 値マッピング | Phase 1 追加タスクの6新規CLI（OS/HC/LC/SoftClip/Saturation）が parser 実装と矛盾する可能性 | **確認済み OK**: 既存5つの CLI parser（parseCliPhaseMode/OrderMode/NoiseShaper/DitherDepth/OutputWav）は全て実在し build_cli_args とマッピング整合 | 継続監視（Phase 1 実装時に再確認） |
| M2 | Phase 1 追加タスク6種の API 必要性 | Phase 1 で追加予定の6パラメータ（`--cli-os-type` 等）のために新規 API が必要か | **確認済み OK**: 全て既存 atomic 変数 + setXXX の CLI 呼び出しパターンに従う。新規 API 要件なし | 実装継続可 |
| M3 | OutputCaptureSink メンバ変数4個が未宣言 | `capture()` Lambda（L200-203）で `sampleRateHz_`/`bitDepth_`/`coeffBankIndex_`/`sessionId_` を使用しているが、クラス定義（§2.2 private部）にこれらの変数がない。setter もない | **致命的**: 実装初日にビルドエラー。M3 が残った状態で Phase 0 を開始すると初日で壁にぶつかる | §2.2 に4変数を private 宣言（sampleRateHz_{48000.0}/bitDepth_{64}/coeffBankIndex_{0}/sessionId_{0}）、setAudioParams() public setter を追加。即座に計画書反映完了 |
| M5 | 既存テストケース16件が P1〜P10 パターンに未割り当て | TC-05A〜D(4), TC-08(1), TC-10(1), TC-11/11B/12/13/14(5), TC-16(1), TC-19/20(2), TC-22(1), TC-26(1), TC-28(1) = 合計16件が §5.2 のパターン表にない | **中**: テスト網羅性の混同 эти тесты не включены в новую автоматизацию | §5.2 パターン表に脚注追加。16件は既存 CI テストフレームワーク（`src/tests/*.cpp`）で継続検証。新 CLI 自動化では対象としない |
| M6 | TC-04/04A ノイズシェイパ信号が §8.2 に未定義、TC-17/18 が §5.7 に未記載 | TC-04/04A: §8.2 信号表に「NS 1kHz -6dBFS」という独立信号種がない（TC-01/03 と同じ入力信号だがノイズシェイパ設定が変数）。TC-17/18: §5.7 テストケース表に存在しない（§8.2 には IMD 信号が定義済み） | **中**: テスト信号の完全な仕様が不明確 | §8.2 信号表の「Sine 1kHz -6dBFS」行に TC-04/04A を追加。§5.7 テストケース表に TC-17（SMPTE IMD）/TC-18（CCIF IMD）を新規行として追加 |
| M7 | pushBecameNonEmptyWithWriter() が未実装 | `LockFreeRingBuffer.h` には `push()`（L33）と `pushWithWriter()`（L44）が存在するが、`pushBecameNonEmptyWithWriter()`（PushResult を返す writer ラムダ版）はまだない | **OK 実装可能**: pushWithWriter() を拡張する形で実装可能。Phase 0 Day 1 で追加 | 実装継続可 |
| M8 | Seqlock single-writer contract の文書化が不明確 | Seqlock（RecoveryHistory 内）の writer が誰かが文書上明確でなかった | **OK**: `executeRecoveryAction()` が Timer Thread からのみ呼ばれる（L1591）。Audio Thread（R/W）は atomic read のみ、single-writer contract 成立 | 継続 |
| M9 | Phase 0 9日工数が楽観的との指摘 | OutputCaptureSink + 10新CLI オプションの Phase 0 工数 9日が楽観的ではないかと心配 | **OK**: DSP 専門知識がいるが JUCE 既存 API 活用で実現可能。9日は DSP 経験ある開発者なら妥当 | 継続 |
| M10 | §8.4 の閾値（thd: -80dB, imd: -80dB, noise_floor: -115dBFS, freq_response: 0.05dB）の物理的妥当性 | 各閾値が物理的に到達可能かが問われた | **OK**: PsychoacousticDither 32bit 設定 ≈ -144dBFS、A-weighting で -110dBFS 到達可能。THD -80dB は Audio Precision 測定器標準性能 | 継続 |
| M11 | TC-37 入力 -0.1dBFS (amplitude ≈ 0.9886) > kOutputHeadroom=0.891 | Double Path (processOutputDouble) の PeakLimiter (`L763`) は `kOutputHeadroom=0.8912509381337456` で出力を制限する。TC-37 の入力 0.9886 はこの閾値を超えるため PeakLimiter が常に動作して歪む | **中**: TC-37 を Double Path で実行すると測定値が歪む | TC-37 行に注記追加。Float Path (processOutput) でのみ測定可能。P2-PEQ-Only は OS=1x+Float Path のため問題なし |
| M12 | LPF/HPF/Allpass 評価関数と Golden WAV 比較関数が analyzers.py に未実装 | `analyze_lpf_hpf_response()` と `compare_golden_wav()` が §5.9 analyzers.py 設計に存在しない | **中**: TC-09/TC-32 の補足評価と全 TC の Golden WAV 比較ができた方が望ましい | §5.9 の Analyzers クラスに両関数を新規実装（analyze_lpf_hpf_response: フィルタ種類判定・カットオフ実測・ロールオフ勾配・形状RMS誤差、compare_golden_wav: 振幅スペクトルRMS・ピーク位置オフセット・エネルギー95%点オフセット・rms偏差・pass判定） |
