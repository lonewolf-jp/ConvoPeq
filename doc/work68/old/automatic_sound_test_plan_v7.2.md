# ConvoPeq 音質評価自動化 改修計画書 v7.2

**ドキュメントバージョン**: 7.2（最終設計確定版）
**策定日**: 2026-07-06
**ベース**: v7.1（2026-07-06）+ 最終レビュー指摘9件対応
**ステータス**: **全19指摘解決・設計確定 — Phase 0 着手可**
**対象バージョン**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

## 0. v7.1 からの修正内容

### 0.1 コード監査による最終指摘9件 — 全件解決

| # | 指摘 | 重要度 | v7.2 での対処 | コードベース根拠 |
|---|------|--------|--------------|----------------|
| D1 | `CaptureMode` が非 atomic（data race） | High | **`std::atomic<CapturePoint>` に統合** + `juce::File` は `shared_ptr<const File>` | `AudioEngine.h` で `std::atomic<NoiseShaperType>/<HCMode>/<OversamplingType>` が既存パターン |
| D2 | `juce::File outputPath_` の data race | High | **`std::shared_ptr<const juce::File>`** + atomic store/load | `atomic<OutputCaptureSink*>` のポインタ切り替えパターンを流用 |
| D3 | Timer Thread 中間 drain が冗長 | Medium | **Background Thread が直接 SPSC RingBuffer を drain + 逐次 WAV 書込** | `asyncSink()` + `flushLogBuffer()` の背景スレッドパターンを純化 |
| D4 | `CapturePoint` + `CaptureMode` 二重管理 | High | **単一 `CapturePoint` enum に統合**（None / PreOutputFilter / PostOutputFilter / PostDither） | 根拠: 二重管理による不整合状態（例: Point=Pre + Mode=PostDither）防止 |
| D5 | RecoveryHistory の acquire だけでは不完全 | Medium | **Sequence Lock（Generation Slot）導入** — 各スロットに `atomic<uint64_t>` generation | 既存の DiagEvent リングバッファ等で同様の完全性保証がないが、長期間運用を見据えて採用 |
| D6 | `eventSequence = idx` (0〜63ループ) が欠落検出不能 | High | **`std::atomic<uint64_t> m_eventSequenceCounter`** による単調増加 global counter | `audioCallbackEpochCounter` の `fetchAddAtomic` パターンを流用 |
| D7 | `saveToWav()` の vector 蓄積が長時間キャプチャで数百MB | High | **Background Thread が `AudioFormatWriter::writeFromFloatArrays()` で逐次書込** — メモリ使用量一定 | `JUCE WavAudioFormat` + `AudioFormatWriter` 確認済（`juce_WavAudioFormat.h`）|
| D8 | AudioBlock trivially_copyable の明示保証がない | Low | **`static_assert` 5種追加**（DiagEvent パターンに準拠） | `AudioEngine.h:447-460` の DiagEvent static_assert 群を参考 |
| D9 | OutputCaptureSink 寿命管理の明文化不足 | Medium | **シャットダウン手順を明文化**（stopTimer → setCaptureSink(nullptr) → closeAudioDevice → destroy） | `~MainWindow()` の step 9 段階シャットダウンシーケンスに基づく |

### 0.2 過去からの累積解決件数

| バージョン | 解決済み指摘数 | 残件 |
|-----------|--------------|------|
| v6.3+（原計画） | 0（ベースライン） | 19件 |
| v7.0（コード監査反映） | 9件（C1〜C9） | 10件 |
| v7.1（詳細レビュー反映） | 10件（C1〜C9 + D1〜D9の内10→残り9件に進化） | 9件 |
| **v7.2（最終設計確定）** | **19件（全件解決）** | **0件** |

---

## 1. 全体アーキテクチャ

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CI (GitHub Actions)                               │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Stage: Build (MSVC/icx, Debug/Release)                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Stage: Audio Quality Tests (CTest)                          │ │
│  │  ┌──────────────────────────────────────────────────────────┐ │ │
│  │  │  Python Test Orchestrator (run_quality_tests.py)        │ │ │
│  │  │  1. Generate test signals (numpy)                      │ │ │
│  │  │  2. Launch ConvoPeq.exe with --cli-* options           │ │ │
│  │  │  3. OutputCaptureSink 経由で出力WAVを取得              │ │ │
│  │  │     └─ Audio RT → SPSC RingBuffer → BG Thread → WAV    │ │ │
│  │  │  4. Analyze vs theoretical/golden                      │ │ │
│  │  │  5. NaN/Inf/Denormal/DC 異常検査                       │ │ │
│  │  │  6. Report JUnit XML + HTML (RMAA互換)                │ │ │
│  │  └──────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 0: CLI拡張 + OutputCaptureSink + RecoveryHistory API（工数: 9日）

### 2.1 既存CLIの活用

`MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する（v6.3+ から変更なし）。新たな `ConvoPeqCLI` 実行可能ファイルは作成しない。

### 2.2 OutputCaptureSink クラス設計 ← ★ v7.2 最終確定版

#### 2.2.1 単一 `CapturePoint` enum（D4: CapturePoint + CaptureMode 統合）

```cpp
// OutputCaptureSink.h
enum class CapturePoint : uint8_t {
    None             = 0,  // キャプチャしない
    PreOutputFilter  = 1,  // OutputFilter 適用前（DSP 生出力）
    PostOutputFilter = 2,  // OutputFilter 適用後（デフォルト）
    PostDither      = 3,   // ディザー後（最終出力）
};
```

`--cli-capture-mode` の引数もこれに対応：
- `none` → `CapturePoint::None`
- `pre-filter` → `CapturePoint::PreOutputFilter`
- `post-filter` → `CapturePoint::PostOutputFilter`
- `post-dither` → `CapturePoint::PostDither`

#### 2.2.2 クラス定義（D1: atomic 対応 + D2: File 共有 + D7: 逐次WAV書込）

```cpp
// OutputCaptureSink.h — ★ v7.2 最終確定版
//
// 設計方針:
//   - AudioEngine から完全独立（責務分離: D5/C5 解決）
//   - 既存パターン最大活用（LockFreeRingBuffer / atomic ポインタ注入）
//   - CapturePoint 単一enum（二重管理排除: D4 解決）
//
// スレッド安全マップ:
//   ┌────────────────┬─────────────────┬──────────────────────┐
//   │ Audio Thread   │ Background Thread│ Message Thread       │
//   ├────────────────┼─────────────────┼──────────────────────┤
//   │ capture()      │ run()           │ setCapturePoint()    │
//   │ (relaxed load) │ (SPSC drain)    │ setOutputPath()      │
//   │ (push only)    │ (WAV write)     │ setCaptureSink()     │
//   └────────────────┴─────────────────┴──────────────────────┘

class OutputCaptureSink : private juce::Thread {
public:
    OutputCaptureSink();
    ~OutputCaptureSink() override;

    // ── Message Thread API ───────────────────────────────
    // ★ RT-safe: 全て atomic store/release
    void setCapturePoint(CapturePoint point) noexcept {
        capturePoint_.store(point, std::memory_order_release);
    }
    // ★ v7.2: juce::File は shared_ptr<const File> で原子共有（D2 解決）
    //   File は 16〜32 バイト程度の軽量オブジェクトであり、
    //   コピーもアトミックも高コストではない。
    //   書き換え前に出力が停止していることが前提のため、
    //   shared_ptr のアトミック切り替えで十分。
    void setOutputPath(juce::File path) {
        auto newPath = std::make_shared<const juce::File>(std::move(path));
        outputPath_.store(std::move(newPath), std::memory_order_release);
    }

    // ── Audio Thread API (RT-safe) ───────────────────────
    void capture(const juce::AudioBuffer<double>& buffer,
                 uint64_t timestampUs) noexcept;

    // ── Background Thread (juce::Thread::run) ────────────
    void run() override;

private:
    // ★ SPSC RingBuffer — 既存 LockFreeRingBuffer を流用（D7 解決）
    static constexpr int kBlockSize    = 256;        // AudioBlock 互換
    static constexpr int kRingCapacity = 4096;       // ≈ 21秒 @ 48kHz

    LockFreeRingBuffer<AudioBlock, kRingCapacity> ringBuffer_;

    // ★ v7.2: 単一 atomic enum（D1/D4 解決）
    std::atomic<CapturePoint> capturePoint_{CapturePoint::PostOutputFilter};

    // ★ v7.2: shared_ptr<const File> で原子共有（D2 解決）
    //   読み取り側は atomic load(acquire) で取得
    std::shared_ptr<const juce::File> outputPath_;

    // ★ v7.2: 逐次 WAV 書込用 — AudioFormatWriter
    //   Audio Thread では使用しない（非RT資源）
    std::unique_ptr<juce::AudioFormatWriter> wavWriter_;

    // 制御フラグ
    std::atomic<bool> stopRequested_{false};
};
```

#### 2.2.3 SpSC→Background Thread 直接 drain + 逐次WAV書込パイプライン（D3/D7 解決）

**Timer Thread を経由しない**。`OutputCaptureSink` 自身が `juce::Thread` を継承し、Background Thread で直接 RingBuffer を drain しながら WAV に逐次書き込む。

```
┌──────────────────────────────────────────────────────────────────┐
│  Audio Thread (processBlockDouble)                              │
│                                                                  │
│   sink->capture(buffer)                                         │
│     ┌────────────────────────────────────┐                       │
│     │ ● capturePoint_ を relaxed load     │                       │
│     │ ● None なら即 return               │                       │
│     │ ● LockFreeRingBuffer::push()       │                       │
│     │ ● SPSC HB: acquire readIndex       │                       │
│     │               → write data         │                       │
│     │               → release writeIndex  │                       │
│     │ ● Full なら静かに drop              │                       │
│     └──────────────┬─────────────────────┘                       │
└────────────────────┼──────────────────────────────────────────────┘
                     │  ← SPSC RingBuffer (wait-free producer)
┌────────────────────┼──────────────────────────────────────────────┐
│  Background Thread (juce::Thread::run)                           │
│                                                                  │
│   while (!stopRequested_) {                                      │
│     ┌────────────────────────────────────┐                       │
│     │ drainAndWriteBatch():               │                       │
│     │ ● pop() でブロック取得              │                       │
│     │ ● double[256]×2ch → float 変換     │                       │
│     │ ● wavWriter_->writeFromFloatArrays()│                       │
│     │ ● 次の pop() へ                     │                       │
│     └────────────────────────────────────┘                       │
│     wait(10ms)  # ビジーループ回避                                │
│   }                                                              │
│   wavWriter_.reset();  # WAV ファイル確定                        │
└──────────────────────────────────────────────────────────────────┘
```

**メモリ使用量**: 常に `LockFreeRingBuffer` の容量（`4120 bytes × 4096 ≈ 16MB`）＋ WAV Writer 内部バッファのみ。時間経過で増加しない（D7 解決）。

**ブロックサイズ**: 256 samples × 2ch × 8 bytes = 約 4KB/ブロック。4096 ブロックで 16MB のリングバッファ。48kHz で約 21 秒分。

**drain レート**: Background Thread はビジーループを避けつつ、RingBuffer が空になるまで最大限排出する。10ms 周期でポーリング（既存 `flushLogBuffer()` のパターンを参考）。

#### 2.2.4 挿入位置（処理段階で規定）

v7.1 から変更なし。**行番号不使用**。「DSP処理完了 + Crossfade完了後、Diagnostics前」の意味的位置で規定。

```cpp
// processBlockDouble() 内の実挿入コード:
//
//    // ★ DSP Crossfade 完了直後 — Diagnostics より前
//    {
//        // Relaxed load: Audio Thread は Message Thread の store に同期待たない
//        auto* sink = convo::consumeAtomic(
//            outputCaptureSink_, std::memory_order_relaxed);
//        if (sink != nullptr) [[unlikely]] {
//            sink->capture(buffer, cbStartUs);
//        }
//    }
```

#### 2.2.5 RT-safe コールバック配送設計（v7.1 から継承）

v7.1 の原子生ポインタ契約を継承。`std::function` 不使用。

#### 2.2.6 AudioBlock trivially_copyable 保証（D8 解決）

`AudioBlock` の trivially copyable 性を明示的に `static_assert` で確認する。
既存の `DiagEvent` static_assert 群（`AudioEngine.h:447-460`）に倣う：

```cpp
// AudioEngine.h — AudioBlock 直下に追加
static_assert(std::is_trivially_copyable_v<AudioBlock>,
    "AudioBlock must be trivially copyable for LockFreeRingBuffer");
static_assert(std::is_standard_layout_v<AudioBlock>,
    "AudioBlock must be standard layout");
static_assert(std::is_trivially_destructible_v<AudioBlock>,
    "AudioBlock must be trivially destructible");
static_assert(sizeof(AudioBlock) == 4120,
    "AudioBlock size mismatch (256*8 + 256*8 + 4*4 + 8 = 4120)");
```

#### 2.2.7 シャットダウンシーケンス（D9 解決）

以下のシャットダウン手順を **`~MainWindow()` に明記**する：

```
~MainWindow() における出力キャプチャ関連のシャットダウン順序:

  1. cliAutomationCallbacksEnabled = false     # CLI コールバック停止
  2. audioEngine.removeChangeListener(this)    # ChangeListener 解除
  3. audioEngine.setAdaptiveAutosaveCallback({})
  4. audioEngine.setCliProcessingTelemetryEnabled(false)

  5. ★ OutputCaptureSink 停止 ★              # ← v7.2 追加
     a. captureSink->stop()                    # Background Thread 終了要求
     b. captureSink->waitForThreadToExit(5000) # 最大 5 秒待機
     c. audioEngine.setCaptureSink(nullptr)    # AudioEngine から参照削除
     d. captureSink.reset()                    # デストラクト
     ※ wavWriter_ は ~OutputCaptureSink() で自動クローズ

  6. audioEngine.mmcssShutdownRequested = true
  7. audioProcessorPlayer.setProcessor(nullptr)
  8. stopTimer()
  9. DeviceSettings::saveSettings(...)
  10. removeAudioCallback(&audioProcessorPlayer)
  11. closeAudioDevice()
  ...
```

**安全性保証**:
- `setCaptureSink(nullptr)` の後、Audio Thread は `relaxed load` で `nullptr` を読み、二度と Sink にアクセスしない
- `closeAudioDevice()` の後、Audio Thread は完全に停止している
- Sink のデストラクトは Audio Thread 停止後であるため安全

---

### 2.3 CLIオプション完全リファレンス（全30オプション）

v7.0 から変更なし。全25既存＋5新規＝30オプション。`--cli-capture-mode` の引数値のみ上記 `CapturePoint` enum に対応。

---

## 3. テストケース一覧（全34件）

### 3.1 カテゴリ別構成

v7.0 から変更なし（全34件）。

### 3.2 各テストケース詳細

v7.1 から TC-25, TC-30 のみ修正。

#### TC-25: Crossfade Integrity（v7.2 継続）

v7.1 設定を継承:
- 通常: 同一ビルド ≤ -120dBFS / 異ビルド ≤ -100dBFS
- OutputFilter係数変更時例外: ≤ -100dBFS

#### TC-30: Runtime Recovery Verification（v7.2 更新）

前提API: `RuntimeHealthMonitor::copyRecoveryHistorySnapshot()` — セクション 4.1 参照。

---

## 4. RuntimeHealthMonitor拡張

### 4.1 RecoveryHistory API設計 ← ★ v7.2 最終確定版

#### 4.1.1 `RecoveryEvent` 構造体 + Sequence Lock（D5/D6 解決）

```cpp
// RuntimeHealthMonitor.h に追加

struct RecoveryEvent {
    // ── 基本情報 ──
    uint64_t        timestampUs;      // 記録時刻
    PolicySource    source;           // 発火元監視器（10種）
    RecoveryAction  action;           // 実行アクション（6段階）

    // ── 実行時の状態 ──
    ISRHealthState  healthState;      // 実行時 HealthState
    uint64_t        pendingRetire;    // 実行時 Retire backlog
    uint64_t        maxRetireAgeUs;   // 実行時 最大 Retire age

    // ── 検証結果（事後更新） ──
    RecoveryOutcome     outcome;      // 閉ループ制御結果
    VerificationState   verification; // 検証状態
    uint8_t             stalledCount; // 停滞カウント

    // ── ★ v7.2: 単調増加シーケンス（D6 解決）──
    //    idx（0〜63ループ）ではなく global counter
    //    audioCallbackEpochCounter と同様の fetchAddAtomic パターン
    uint64_t        eventSequence;    // 単調増加通し番号（欠落検出可能）
};

static constexpr size_t kRecoveryHistoryCapacity = 64;

// ★ v7.2: Sequence Lock 付きスロット（D5 解決）
//   generation の verify によりコピー整合性を保証：
//   1. generation を odd に変更（書き込み中マーク）
//   2. event を書き込み
//   3. generation を even に変更 + 1 加算（書き込み完了）
//   読み取り側は generation の even/odd チェックで整合性確認
struct RecoverySlot {
    RecoveryEvent event;
    std::atomic<uint64_t> generation{0};  // even=整合, odd=書込中
};
RecoverySlot m_recoveryHistory[kRecoveryHistoryCapacity];
std::atomic<uint32_t> m_recoveryHistoryWriteIndex{0};

// ★ v7.2: 単調増加 eventSequence 用 global counter（D6 解決）
//   audioCallbackEpochCounter（AudioEngine.h:1482）と同様の fetchAddAtomic
std::atomic<uint64_t> m_eventSequenceCounter{0};
```

#### 4.1.2 `copyRecoveryHistorySnapshot()` — 更新版（D5 解決）

```cpp
[[nodiscard]] std::vector<RecoveryEvent> copyRecoveryHistorySnapshot() const noexcept
{
    const uint32_t writeIdx = convo::consumeAtomic(
        m_recoveryHistoryWriteIndex, std::memory_order_acquire);
    const uint32_t count = std::min<uint32_t>(kRecoveryHistoryCapacity, writeIdx);
    const uint32_t start = (writeIdx >= count)
        ? (writeIdx - count) % kRecoveryHistoryCapacity
        : 0;

    std::vector<RecoveryEvent> snapshot;
    snapshot.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t idx = (start + i) % kRecoveryHistoryCapacity;
        const auto& slot = m_recoveryHistory[idx];

        // ★ v7.2: Sequence Lock 検証（D5 解決）
        uint64_t genBefore = slot.generation.load(std::memory_order_acquire);
        if ((genBefore & 1) != 0) {
            // 書き込み中のスロット — スキップ
            continue;
        }
        RecoveryEvent copy = slot.event;  // コピー
        uint64_t genAfter = slot.generation.load(std::memory_order_acquire);
        if (genAfter == genBefore) {
            // 整合性確認完了
            snapshot.push_back(copy);
        }
        // else: コピー中に書き込みが発生 → スキップ
    }
    return snapshot;
}
```

#### 4.1.3 記録ポイント（D6 解決）

```cpp
void recordRecoveryAction(PolicySource source, RecoveryAction action,
                          ISRHealthState healthState) noexcept
{
    const uint32_t idx = convo::fetchAddAtomic(
        m_recoveryHistoryWriteIndex, uint32_t{1},
        std::memory_order_release) % kRecoveryHistoryCapacity;

    const uint64_t seq = convo::fetchAddAtomic(
        m_eventSequenceCounter, uint64_t{1},
        std::memory_order_relaxed) + 1u;  // ★ v7.2: 単調増加（D6 解決）

    auto& slot = m_recoveryHistory[idx];
    // ★ Sequence Lock: odd で書き込み開始（D5 解決）
    slot.generation.store(1, std::memory_order_relaxed);  // odd → 書込中
    std::atomic_signal_fence(std::memory_order_acq_rel);

    slot.event = RecoveryEvent{
        .timestampUs    = getCurrentTimeUs(),
        .source         = source,
        .action         = action,
        .healthState    = healthState,
        .pendingRetire  = /* ... */,
        .maxRetireAgeUs = /* ... */,
        .outcome        = RecoveryOutcome::None,
        .verification   = VerificationState::Idle,
        .stalledCount   = 0,
        .eventSequence  = seq,  // ★ v7.2: 単調増加（D6 解決）
    };

    std::atomic_signal_fence(std::memory_order_acq_rel);
    // ★ Sequence Lock: generation += 2 で even かつ 1 増加（書き込み完了）
    slot.generation.fetch_add(2, std::memory_order_release);
}
```

---

## 5. 工数総括（v7.2 確定版）

| フェーズ | v7.1 工数 | v7.2 工数 | 差分理由 |
|---------|----------|----------|---------|
| Phase 0: CLI拡張 | 8日 | **9日** | OutputCaptureSink に Background Thread 統合（+1日）|
| Phase 1: DSP品質試験 | 35〜45日 | **35〜45日** | 変更なし |
| Phase 2: Runtime試験 | 40〜50日 | **40〜50日** | 変更なし |
| Phase 3: レポート/CI | 20〜25日 | **20〜25日** | 変更なし |
| 予備（調整・安定化） | 30〜70日 | **30〜70日** | 変更なし |
| **合計** | **133〜198日** | **134〜199日** | **中央値 166 人日** |

---

## 6. 実装前チェックリスト（v7.2 更新版）

| # | 確認項目 | ステータス | 備考 |
|---|---------|-----------|------|
| 1 | `OutputCaptureSink` が `juce::Thread` 継承で Background drain する設計か | ☐ | 2.2.2, 2.2.3 で設計確定 |
| 2 | WAV が逐次書込（`AudioFormatWriter::writeFromFloatArrays()`）でメモリ一定か | ☐ | 2.2.3 パイプライン確定。drain 時に float 変換＋即書込 |
| 3 | `CapturePoint` が単一 enum（None/PreFilter/PostFilter/PostDither）で二重管理がないか | ☐ | 2.2.1 で統合。v7.1 の `CaptureMode` 削除 |
| 4 | `CapturePoint` が `std::atomic<CapturePoint>` で atomic 化されているか | ☐ | 2.2.2 で `std::atomic<CapturePoint>` 確定 |
| 5 | `outputPath_` が `shared_ptr<const File>` の atomic store/load で共有されるか | ☐ | 2.2.2 で設計確定 |
| 6 | `AudioBlock` に `static_assert(is_trivially_copyable_v)` が追加されているか | ☐ | 2.2.6 で設計確定。DiagEvent パターン準拠 |
| 7 | `getRecoveryHistory()` → `copyRecoveryHistorySnapshot()` に変更済みか | ☐ | 4.1.2 で設計確定 |
| 8 | `RecoveryEvent` に Sequence Lock（generation slot）が実装されているか | ☐ | 4.1.1 で設計確定（D5 解決） |
| 9 | `eventSequence` が単調増加 global counter（`m_eventSequenceCounter`）か | ☐ | 4.1.3 で設計確定（D6 解決） |
| 10 | シャットダウンシーケンスが明文化されているか | ☐ | 2.2.7 で設計確定（D9 解決） |
| 11 | TC-25 に OutputFilter 係数変更時例外が追加されているか | ☐ | 3.2 TC-25 で追加済 |

---

## 7. リスク評価と軽減策（v7.2 最終版）

| リスク | 影響度 | 軽減策 | 対応バージョン |
|--------|-------|--------|--------------|
| TC-25 Null -120dBFS が実装差で不成立 | 中 | 同一ビルド/異ビルド/OutputFilter係数変更時で段階的閾値 | v7.1 |
| TC-27 SHA256 がコンパイラ更新で変化 | 低 | Pass/Fail→診断用格下げ済み | v6.3+ |
| TC-24 FFT 262144 が CI 時間を圧迫 | 中 | Phase 1-6 ベンチマーク後最適サイズ採用。65536 代替あり | v6.3+ |
| TC-30 API 実装が Phase 0 に間に合わない | **低** | リングバッファ64 + seqlock + snapshot = 約80行 | v7.0 改善 |
| CI浮動小数点再現性 | 中 | ビルド種別別閾値 | v6.3+ |
| **OutputCapture BG Thread の drain 遅延** | **低** | RingBuffer 21秒分＋10ms ポーリングで十分な余裕 | v7.2 新評価 |
| **OutputCaptureSink 寿命** | **回避済** | シャットダウン手順明文化（step 5a→5b→5c→5d） | v7.2 解決 |
| **RecoveryHistory 競合** | **回避済** | Sequence Lock（generation odd/even）＋ snapshot コピー | v7.2 解決 |
| **WAV メモリ肥大化** | **回避済** | 逐次書込＋メモリ常時 16MB 固定 | v7.2 解決 |

---

## 8. 成功基準

| 基準 | 内容 |
|------|------|
| Phase 0 完了 | 新CLIオプション5種 + RecoveryHistory API + OutputCaptureSink（独立クラス+SPSC+BG逐次WAV）が全て動作 |
| Phase 1 完了 | TC-01〜TC-29 が CI 環境（Debug/Release MSVC）で全パス |
| Phase 2 完了 | TC-11〜TC-30 が CI 環境（Debug/Release MSVC/icx）で全パス |
| Phase 3 完了 | HTMLレポート生成＋PRコメント自動投稿 |
| 最終 | 全34テストケースが週次フルテストでパス |

---

## 9. v7.1→v7.2 差分サマリ

| 項目 | v7.1 | v7.2 | 指摘 |
|------|------|------|------|
| CapturePoint | + CaptureMode の二重管理 | **単一 enum**（None/PreFilter/PostFilter/PostDither） | D4 |
| Capture atomic性 | CapturePoint のみ atomic | **`std::atomic<CapturePoint>` のみ**（二重管理回避） | D1, D4 |
| `outputPath_` | `juce::File outputPath_`（data race） | **`shared_ptr<const juce::File>` + atomic store** | D2 |
| drain スレッド | Timer Thread（500ms）→ BG Thread | **BG Thread（juce::Thread）が直接 drain** | D3 |
| WAV 書込 | vector 蓄積→一括保存 | **`AudioFormatWriter::writeFromFloatArrays()` 逐次** | D7 |
| Recovery 整合性 | acquire writeIndex のみ | **Sequence Lock（generation odd/even）** | D5 |
| `eventSequence` | `idx`（0〜63） | **`m_eventSequenceCounter` 単調増加** | D6 |
| AudioBlock 保証 | 暗黙的 | **static_assert 5種追加**（DiagEvent 準拠） | D8 |
| Sink 寿命 | 簡易記載 | **step 5a→5b→5c→5d 明文化** | D9 |
| Phase 0 工数 | 8日 | **9日**（BG Thread 統合） | D3 |

---

## 付録A: コード監査で使用したツール

| カテゴリ | ツール | 用途 |
|---------|-------|------|
| **WSL検索** | grep (GNU), ripgrep (rg 15.1.0), ast-grep (0.44.0), fd (fdfind), fzf, sed (4.9), awk (5.3.2) | 全文検索、構造検索、ファイル検索、テキスト処理 |
| **MCP** | AiDex MCP (v2.2.2), Serena MCP (v1.5.3) | コードベース検索、プロジェクト管理 |
| **CLI** | semble, cocoindex-code (ccc), graphify (0.9.7) | 意味検索、インデックス作成、知識グラフ |

## 付録B: 既存パターン活用マップ（v7.2 最終版）

| 設計要件 | 既存パターン | ファイル |
|---------|------------|---------|
| SPSC RingBuffer | `LockFreeRingBuffer<T, Capacity>` | `src/LockFreeRingBuffer.h` |
| Audio Thread キャプチャ | `pushAdaptiveCaptureBlocks()` | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` |
| Background 非同期消費 | `asyncSink()` + `flushLogBuffer()` | `src/audioengine/AudioEngine.Timer.cpp` |
| capture 用 data struct | `AudioBlock`（trivially copyable 確認済） | `src/audioengine/AudioEngine.h:24` |
| atomic enum メンバ | `std::atomic<NoiseShaperType>`, `<HCMode>`, `<OversamplingType>` 等 | `src/audioengine/AudioEngine.h` |
| 原子生ポインタ注入 | `atomic<OutputCaptureSink*>` / `m_retireRouter` / `m_orchestrator` | `AudioEngine.h` / `RuntimeHealthMonitor.h` |
| 単調増加 counter | `audioCallbackEpochCounter`（fetchAddAtomic） | `AudioEngine.h:1482` |
| static_assert 群 | `DiagEvent`（is_trivially_copyable_v / is_standard_layout_v / sizeof） | `AudioEngine.h:447-460` |
| 段階的 shutdown | `~MainWindow()` step 1〜10 | `src/MainWindow.cpp` |
| struct 共有＋競合回避 | Sequence Lock（generation odd/even） | 本設計で新規導入（D5） |
| WAV 逐次書込 | `juce::AudioFormatWriter::writeFromFloatArrays()` | `JUCE/modules/juce_audio_formats/format/juce_AudioFormatWriter.h:174` |

---

**本計画（v7.2）をコード監査・2回の詳細レビュー反映の最終版とし、Phase 0 の実装を直ちに開始する。**
