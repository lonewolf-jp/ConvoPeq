# ConvoPeq 音質評価自動化 改修計画書 v7.3

**バージョン**: 7.3（最終設計確定版）
**策定日**: 2026-07-06
**ベース**: v6.3+（2026-06-22）＋ 全レビュー反映
**ステータス**: **Phase 0 実装開始可**
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
    // ★ Single-shot object: 1セッション専用。再利用禁止。
    //   ライフサイクル: setOutputPath() → setCapturePoint() → startCapturing() → setCaptureSink(nullptr) → stopCapturing() → waitForThreadToExit() → destroy
    // ★ 契約: startCapturing() は単回起動（2回目の呼出は禁止）。
    //   OutputCaptureSink は1セッション専用。stopCapturing() 後の再利用は禁止。
    //   startCapturing() で stopRequested_ を false に初期化する必要はない。
    //   startCapturing() 以降の setOutputPath() は禁止

    // ── Audio Thread API (RT-safe: push + signal のみ) ──
    void capture(const juce::AudioBuffer<double>& buffer,
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
    std::unique_ptr<float[]> convertBufferL_, convertBufferR_;  // run()で事前確保
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

```cpp
void capture(const juce::AudioBuffer<double>& buffer,
             uint64_t timestampUs) noexcept
{
    const CapturePoint point = capturePoint_.load(std::memory_order_relaxed);
    if (point == CapturePoint::None) return;

    AudioBlock block{};  // ★ ゼロ初期化（将来の padding/reserved にも安全）
    // ... buffer データを block に詰める ...

    // ★ PushResult で 3 状態を区別（drop カウントの正確性）
    switch (ringBuffer_.pushBecameNonEmpty(block)) {
    case PushResult::BecameNonEmpty:
        wakeEvent_.signal();  // RT-safe: 非ブロッキング
        break;
    case PushResult::AlreadyNonEmpty:
        // リングが既に非空であるため、drain-all設計により追加通知は不要
        break;
    case PushResult::Full:
        droppedBlocks_.fetch_add(1, std::memory_order_relaxed);
        break;
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

### 2.3 pushBecameNonEmpty() — LockFreeRingBuffer 拡張

```cpp
// LockFreeRingBuffer.h に追加。既存の push/pop/size/clear は変更しない。
enum class PushResult : uint8_t { Full, BecameNonEmpty, AlreadyNonEmpty };

PushResult pushBecameNonEmpty(const T& item) noexcept {
    size_t w = convo::consumeAtomic(writeIndex, std::memory_order_acquire);
    size_t r = convo::consumeAtomic(readIndex, std::memory_order_acquire);
    if ((w - r) >= Capacity) return PushResult::Full;
    const bool wasEmpty = (w == r);
    buffer[w & MASK] = item;
    convo::publishAtomic(writeIndex, w + 1, std::memory_order_release);
    return wasEmpty ? PushResult::BecameNonEmpty : PushResult::AlreadyNonEmpty;
}
```

### 2.4 シャットダウンシーケンス

```
  1. cliAutomationCallbacksEnabled = false
  2. audioEngine.removeChangeListener(this)
  3. audioEngine.setCliProcessingTelemetryEnabled(false)
  4. ★ setCaptureSink(nullptr) ★              # Audio Thread の参照を先に絶つ
     #   convo::publishAtomic(outputCaptureSink_, nullptr, release) で atomic store。
     #   Audio Thread は relaxed load で読み、nullptr 検出後は capture() を呼ばない。
  5. captureSink->stop()                       # stopRequested_.store(true, release)
     captureSink->wakeEvent_.signal()          # BG Thread 起床（wait(-1) からの復帰用）
  6. captureSink->waitForThreadToExit(5000)    # BG Thread 終了待機
  7. captureSink.reset()                       # デストラクト
  8. mmcssShutdownRequested = true
  9. setProcessor(nullptr)
 10. stopTimer()
 11. saveSettings(...)
 12. removeAudioCallback(...)
 13. closeAudioDevice()
```

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
| `--cli-capture-mode` | 低（1h） | `none`/`post-dither` 選択 |
| `--cli-dump-filter-coeffs` | 低（2h） | OutputFilter const getter追加、JSON出力 |
| `--cli-ir-reload-list` | 中（4h） | カンマ区切りパース→複数IR逐次ロード |
| `--cli-progressive-upgrade` | 低（1h） | `setConvolverEnableProgressiveUpgrade(true)` |

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
    const uint64_t idx = fetchAdd(m_recoveryHistoryWriteIndex, 1, relaxed) % 64;    // ★ single writer 前提のため CAS 不要
    const uint64_t seq = fetchAdd(m_eventSequenceCounter, 1, relaxed) + 1u;

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
            // それでも失敗する場合は Writer（Audio Thread）が高頻度で更新中。
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

---

## 5. テストパラメータパターンと自動実行設計

### 5.1 パラメータ次元（全10次元）

| 次元 | CLIオプション／API | 値 | 既定値 |
|------|-------------------|----|--------|
| 処理順序 | `--cli-order` | `conv` / `peq` / `conv->peq` / `peq->conv` | `conv->peq` |
| 位相モード | `--cli-phase` | `asis` / `mixed` / `minimum` | `asis` |
| OSタイプ | — | `iir` / `linear-phase` | `iir` |
| OS倍率 | — | `1x` / `2x` / `4x` / `8x` | `1x` |
| ノイズシェイパー | `--cli-noise-shaper` | `psychoacoustic` / `fixed4` / `adaptive9` / `fixed15` | `psychoacoustic` |
| ディザー深度 | `--cli-dither-bit-depth` | `16` / `24` / `32` | `32` |
| HCモード | — | `sharp` / `natural` / `soft` | `natural` |
| LCモード | — | `natural` / `soft` | `natural` |
| ソフトクリップ | — | `on` / `off` | `off` |
| サチュレーション | — | `0.0`〜`1.0` | `0.0` |

### 5.2 7測定パターン

| パターン | 目的 | 主要パラメータ | テストケース |
|---------|------|--------------|-------------|
| **P1-Baseline** | 基準測定 | 全既定値 | TC-01,03,04,04A,06,23,24,29A,29B |
| **P2-PEQ-Only** | PEQ単体品質 | `order=peq`, convBypass | TC-01,01B,03,04,07 |
| **P3-ConvoThenPEQ** | 既定順序品質 | `order=conv->peq`, `phase=mixed` | TC-01,02,09,21,23 |
| **P4-PEQThenConvo** | 逆順序応答一致 | `order=peq->conv`, `phase=mixed` | TC-01,15 |
| **P5-ReferenceQuality** | 最高忠実度 | OS8x, Adaptive9, **SoftClip OFF, Sat=0** | TC-01,03,04,04A,23,24,**31a,33,34,39,41** |
| **P6-Musical** | 非線形音質 | OS8x, Adaptive9, **SoftClip ON, Sat=0.3** | TC-01,03,04,04A,23,24,**31b,39,40** |
| **P7-OSx2/4/8** | OS倍率スイープ | OS=2x/4x/8x | TC-01,09,24,**33,39** |
| **P8-MinimumPhase** | 最小位相品質 | `phase=minimum` | TC-01,02,**32,35,41** |
| **P9-AsIsPhase** | As Is位相品質 | `phase=asis` | TC-01,02,**32,35** |
| **P10-MixedPhase** | Mixed Phase重点 | `phase=mixed` | TC-01,02,**21,32,35,41** |

### 5.3 自動実行設計

```yaml
# test_config.yaml — 測定パターン定義ファイル
patterns:
  P1-Baseline:
    order: conv->peq; phase: asis; os: 1x
    noiseShaper: psychoacoustic; ditherBitDepth: 32
    hcMode: natural; lcMode: natural; softClip: false; saturation: 0.0
    srcIrs: [dirac]
    testCases: [TC-01, TC-03, TC-04, TC-04A, TC-06, TC-23, TC-24, TC-29A, TC-29B]

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
  debug_msvc:  { thdn: -80, imd: -80, noiseFloor: -115, freqResponse: 0.05 }
  release_msvc: { thdn: -100, imd: -90, noiseFloor: -120, freqResponse: 0.05 }
  release_icx:  { thdn: -100, imd: -90, noiseFloor: -120, freqResponse: 0.05 }
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
    """10次元パラメータ → CLI引数リスト"""
    args = [f"--cli-ir={ir_file}", f"--cli-output-wav={output_dir/'output.wav'}",
            "--cli-capture-mode=post-dither", "--cli-exit-ms=5000"]
    args.append(order_map[pattern["order"]])
    args.append(phase_map[pattern["phase"]])
    args.append(ns_map[pattern["noiseShaper"]])
    if pattern.get("ditherBitDepth", 32) < 32:
        args.append(f"--cli-dither-bit-depth={pattern['ditherBitDepth']}")
    if pattern.get("convBypass"):
        args += ["--cli-bypass-burst-count=1", "--cli-bypass-burst-value=1"]
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
| **TC-31a** | **Null Test（Bit Exact）** | バイパス完全性 | Dirac -6dBFS, 3秒 | **サンプル一致率 = 100%**（全サンプルビット完全一致）| ★★★★★ |
| **TC-31b** | **Null Test（Float Pipeline）** | パイプライン完全性 | 1kHz正弦波 -6dBFS, 3秒 | 差分RMS ≤ **-145dBFS** / Peak ≤ **-135dBFS**（FMA/コンパイラ差許容）| ★★★★★ |
| **TC-32** | **Log Sweep 伝達関数** | 周波数特性 | 20Hz-24kHz対数スイープ 10秒 | FFT→Transfer Function: 振幅±0.05dB / 群遅延±0.1sample | ★★★★★ |
| **TC-33** | **Alias Rejection** | エイリアシング | 19kHz+20kHz+22kHz 各-6dBFS 3秒 | OS=2x/4x/8x で10kHz以下の折り返しエネルギー ≤ -130dBFS | ★★★★★ |
| **TC-34** | **PSD（Noise Shaper）** | ノイズ | 無音 5秒 | Welch PSD：A-weighting 20Hz-20kHz ≤ -110dBFS / 形状一致 | ★★★★☆ |
| **TC-35** | **Group Delay（Mixed Phase）** | 位相 | 20Hz-20kHz対数スイープ 10秒 | Mixed Phase IRの群遅延曲線 vs 理論値 ±5% | ★★★★☆ |
| **TC-36** | **Stereo Crosstalk** | チャネル分離 | L=1kHz正弦波 R=無音 3秒 | R出力の漏れエネルギー ≤ -140dBFS | ★★★★☆ |
| **TC-37** | **Numerical Transparency（32bit）** | 数値精度 | 1kHz正弦波 -0.1dBFS 3秒 | RMS誤差 ≤ **1 ULP** / Peak ≤ **2 ULP**（IEEE754単精度/コンパイラ最適化差許容）| ★★★★☆ |
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
        """TC-32: Log Sweep → 伝達関数（振幅+位相+群遅延）
        注意: signal.freqz() はフィルタ係数用API。測定データには使わない。
        正しい方法: H = FFT(output) / FFT(input) で伝達関数を計算。"""
        N = len(output_wav)
        # ゼロパディングで周波数分解能向上
        f = np.fft.rfftfreq(N * 4, d=1.0 / sr)
        H = np.fft.rfft(output_wav, n=N * 4) / (np.fft.rfft(input_wav, n=N * 4) + 1e-30)
        # Amplitude response
        band = (f >= 20) & (f <= 20000)
        mag_db = 20 * np.log10(np.abs(H[band]) + 1e-30)
        # Group delay from unwrapped phase derivative
        phase = np.unwrap(np.angle(H[band]))
        group_delay = -np.diff(phase) / (2 * np.pi * np.diff(f[band]))
        return {"freq_response_rms_db": np.std(mag_db), "group_delay_rms_sample": np.std(group_delay)}

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
        """TC-33: Alias Rejection → 理論Alias周波数近傍のエネルギー
        誤った方法: 折り返し帯域全体を評価（ノイズフロアを過大評価）
        正しい方法: 入力周波数から理論Alias位置を計算し、その近傍のみ評価。
        ex: 48kHz/OS=4x → nyq=96kHz、入力21kHzのAliasは 75kHz に発生"""
        f, Pxx = signal.welch(output_wav, fs=sr * os_factor, nperseg=16384)
        if input_freqs is None:
            input_freqs = [21000, 22000, 23000]  # デフォルト入力
        alias_energy = 0.0
        for fin in input_freqs:
            # 理論Alias位置: f_alias = |k * fs_in - fin| (k=1,2,...)
            for k in range(1, os_factor + 1):
                f_alias = abs(k * sr - fin)
                if f_alias < sr * os_factor / 2:
                    bin_near = np.argmin(np.abs(f - f_alias))
                    alias_energy += Pxx[bin_near]
        return {"alias_energy_db": 10 * np.log10(alias_energy + 1e-30)}

    @staticmethod
    def analyze_thd_blackman_harris(output_wav: np.ndarray, sr: int, fundamental_hz: float) -> dict:
        \"\"\"TC-39: 周波数別 THD+N（Blackman-Harris FFT, Welch不使用）
        Welch平均は歪み成分をぼやけさせるため、Audio Precision同様のFFTベース。\"\"\"
        from numpy import blackman, fft
        N = 262144  # AES17準拠
        if len(output_wav) < N:
            output_wav = np.pad(output_wav, (0, N - len(output_wav)))
        window = np.blackman(N)
        X = fft.rfft(output_wav[:N] * window)
        f = fft.rfftfreq(N, d=1.0 / sr)
        fund_bin = np.argmin(np.abs(f - fundamental_hz))
        fund_power = np.abs(X[fund_bin])**2
        harmonics = range(2, 11)
        harmonic_power = sum(np.abs(X[min(fund_bin * h, len(X)-1)])**2 for h in harmonics)
        return {"thdn_db": 10 * np.log10(harmonic_power / (fund_power + 1e-30) + 1e-30),
                "fundamental_hz": fundamental_hz,
                "fft_size": N, "window": "blackman-harris"}

    @staticmethod
    def analyze_noise_psd(output_wav: np.ndarray, sr: int) -> dict:
        """TC-34: PSD → A-weighting レベル＋形状一致度"""
        f, Pxx = signal.welch(output_wav, fs=sr, nperseg=8192)
        # A-weighting フィルタ適用
        a_weight = A_weighting(f)
        return {"a_weighted_db": ..., "psd_shape_rms": ...}



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
        \"\"\"TC-36: クロストーク → L→R / R→L 両方向\"\"\"
        return {
            \"crosstalk_lr_db\": 10 * np.log10(np.mean(output_r**2) / np.mean(output_l**2) + 1e-30),
            \"crosstalk_rl_db\": 10 * np.log10(np.mean(output_l**2) / np.mean(output_r**2) + 1e-30),
        }


# ── Golden Metrics 保存 ──
# ゴールデンWAVだけでなく、測定結果（THD・遅延・周波数特性・ノイズ等）も
# golden_metrics.json として保存する。将来FFTアルゴリズム変更等で
# Bit一致しなくなっても、品質が同等であることを判断できる。
# 保存形式:
# {
#   \"tc\": \"TC-03\",
#   \"pattern\": \"P1-Baseline\",
#   \"build\": \"release_msvc_v143\",
#   \"metrics\": { \"thdn_db\": -102.3, \"noise_floor_db\": -121.5 },
#   \"golden_metrics\": { \"thdn_db\": -102.5, \"noise_floor_db\": -121.8 },
#   \"threshold\": { \"thdn\": -100, \"noise_floor\": -120 }
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
| Phase 1: DSP品質試験 | **45〜55** | 既存34TC + 新規11TC（Null/Sweep/Alias/PSD等）+ ゴールデン比較FW |
| Phase 2: Runtime試験 | **40〜50** | TC-25/27/30/38安定化含む |
| Phase 3: レポート/CI | **20〜25** | HTML + GitHub連携 |
| 予備 | **30〜70** | 実績に応じて変動 |
| **合計** | **144〜209（中央値176人日 ≈ 22〜26週間）** | |

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
| 12 | `pushBecameNonEmpty()`: LockFreeRingBuffer に追加 | ☐ |
| 13 | `stopRequested_`: `store(release)` / `load(acquire)`, signal() で起床 | ☐ |
| 14 | TC-25: OutputFilter係数変更時例外（≤−100dBFS）| ☐ |
| 15 | TC-01B: IR 参照パス `sampledata/impulse_room_correction_hpf_lpf.wav`（論理名 `room_correction`）| ☐ |
| 16 | テストケース数: 全41件（既存34 + 新規11）で統一 | ☐ |
| 17 | `test_config.yaml`: 7パターン定義完了 | ☐ |
| 18 | `analyzers.py`: Null Test / Log Sweep / Alias / PSD / THD Sweep 実装完了 | ☐ |
| 19 | `golden_calculator.py`: 理論値・Golden WAV・統計比較フレームワーク完了 | ☐ |
| 20 | P5/P7 パターンに TC-33(Alias)/TC-34(PSD)/TC-39(THDSweep) 割当完了 | ☐ |

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
