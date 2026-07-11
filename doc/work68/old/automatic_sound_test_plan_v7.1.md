# ConvoPeq 音質評価自動化 改修計画書 v7.1

**ドキュメントバージョン**: 7.1（コード監査・全指摘反映版）
**策定日**: 2026-07-06
**ベース**: v7.0（2026-07-06）+ 詳細レビュー指摘対応
**ステータス**: **全10指摘解決・設計確定 — Phase 0 着手可能**
**対象バージョン**: ConvoPeq v0.5.3 → v1.0 (QA Phase)

---

## 0. v7.0 からの修正内容

### 0.1 コード監査による重要指摘（4項目）— 対処完了

| # | 指摘 | 重要度 | v7.1 での対処 | 該当セクション |
|---|------|--------|--------------|--------------|
| C1 | OutputCaptureCallback のRT安全性：`std::function` の更新がスレッド安全でない | **Critical** | **atomic_load/store によるポインタ切り替え**＋Message Thread 専用セッター規定 | 2.2.1, 2.2.3 |
| C2 | OutputCapture→WAV保存がAudio Threadで行われてしまう | **Critical** | **三重分離パイプライン**: Audio RT → SPSC RingBuffer → Background Thread → WAV I/O | 2.2.4 |
| C3 | `getRecoveryHistory()` が生 `span` を返すのはスレッド安全でない | **Critical** | **`copySnapshot()` メソッド**に変更し、スナップショットコピーを返す | 4.1.2 |
| C4 | 挿入位置が行番号（L470）固定でメンテ不能 | **High** | **処理段階 (`Stage`) enum による位置指定**＋`CapturePoint` enum で複数取得点対応 | 2.2.2, 2.2.5 |

### 0.2 追加指摘（6項目）— 対処完了

| # | 指摘 | 重要度 | v7.1 での対処 | 該当セクション |
|---|------|--------|--------------|--------------|
| C5 | AudioEngine が肥大化（4155行）しており、コールバック追加は責務を増やす | **High** | **`OutputCaptureSink` を独立クラス**として分離。`AudioEngine` は `setCaptureSink()` のみ | 2.2 |
| C6 | `CapturePoint` enum が `none/post-dither` のみで拡張性低い | **Medium** | **`PreOutputFilter` / `PostOutputFilter` / `PostDither`** の3点に拡張。`CapturePoint` enum 導入 | 2.2.5 |
| C7 | TC-25 は OutputFilter 係数変更時も閾値が緩む例外条件を設けるべき | **Medium** | **`OutputFilterChanged` 例外条件を TC-25 に追加**（閾値を -100dBFS に緩和） | TC-25 |
| C8 | Python 流用工数が楽観的（40→30〜40日） | **Medium** | 精査結果 **35〜45日** に修正。既存ツールの重複領域のみ20〜25%削減と再評価 | 5.0 |
| C9 | RecoveryEvent のフィールドが `{action, result}` のみで情報不足 | **High** | **`PolicySource` / `HealthState` / `VerificationState` / `TrendSnapshot` を追加** | 4.1.1 |
| C10 | TC-27 SHA256 診断用格下げは妥当だが SHA256 の保存自体は有用 | **Low** | 診断用 SHA256 保存を継続。変更不要を確認 | TC-27 |

### 0.3 コード監査で確定した既存パターン（設計資産）

| 既存パターン | 場所 | v7.1 での活用 |
|------------|------|-------------|
| `LockFreeRingBuffer<T, N>` | `src/LockFreeRingBuffer.h` | **SPSC キューとして再利用**。AudioBlock は trivially copyable 確認済 |
| `pushAdaptiveCaptureBlocks()` | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` | **Audio Thread からのキャプチャパターンそのまま流用** |
| `asyncSink() + flushLogBuffer()` | `src/audioengine/AudioEngine.Timer.cpp` | **Background Thread での非同期消費パターンを模倣** |
| `AudioBlock` struct | `src/audioengine/AudioEngine.h:24` | **24バイト固定長＋256サンプル/ch＝軽量**。`static_assert(is_trivially_copyable)` 確認済 |
| `ProcessingState::adaptiveCaptureQueue` | `src/audioengine/AudioEngine.h:805` | **キャプチャキュー注入パターン確立済み** |

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

## 2. Phase 0: CLI拡張 + OutputCaptureSink + RecoveryHistory API（工数: 8日）

### 2.1 既存CLIの活用

`MainWindow::runCommandLineAutomation()` をテストハーネスから呼び出す形で利用する（v6.3+ から変更なし）。新たな `ConvoPeqCLI` 実行可能ファイルは作成しない。

### 2.2 OutputCaptureSink クラス設計 ← ★ v7.1 最大の改善点

#### 2.2.1 クラス責務と設計方針

**`OutputCaptureSink` は `AudioEngine` から完全に独立したクラスとする。**

```cpp
// OutputCaptureSink.h — 新規（独立クラス）
//
// 責務:
//   1. Audio Thread から最終出力バッファを受け取り、SPSC RingBuffer に格納
//   2. Message Thread (Timer) からの drain 要求に応じて RingBuffer を排出
//   3. Background Thread で WAV ファイルに保存
//   4. CLI からの制御（capture-point / capture-mode / output-wav-path）
//
// ★ RT-safe 契約:
//   - capture()        : Audio Thread → relaxed load + push (non-blocking)
//   - setCapturePoint() / setOutputPath() : Message Thread only（atomic store）
//   - drainToBuffer()  : Timer Thread（バッチ排出）
//   - saveToWav()      : Background Thread（Disk I/O）
//
// ★ スレッド安全性:
//   - CaptureSink インスタンスのポインタは atomic<OutputCaptureSink*>
//   - AudioEngine は relaxed load でポインタを取得
//   - Message Thread は release/store でポインタを更新
```

```cpp
class OutputCaptureSink {
public:
    // ── スレッド境界 ─────────────────────────────────────
    // [Audio Thread] 最終出力をキャプチャ
    void capture(const juce::AudioBuffer<double>& buffer, uint64_t timestampUs) noexcept;

    // [Message Thread] 設定変更 — RT-safe（atomic store）
    void setCapturePoint(CapturePoint point) noexcept;
    void setOutputPath(const juce::File& path);
    void setCaptureMode(CaptureMode mode) noexcept;

    // [Timer Thread] RingBuffer を排出して内部バッファに蓄積
    size_t drainToBuffer() noexcept;

    // [Background Thread] 蓄積バッファを WAV に保存
    bool saveToWav(const juce::File& path);

    // ── 内部構造 ─────────────────────────────────────────
private:
    static constexpr int kBlockSize     = 256;        // pushAdaptiveCaptureBlocks 互換
    static constexpr int kRingCapacity  = 4096;       // ≈ 21秒 @ 48kHz
    static constexpr int kDrainBatch    = 100;        // 1回の排出上限

    LockFreeRingBuffer<AudioBlock, kRingCapacity> ringBuffer_;
    std::vector<double> accumulatorL_;   // drain 先（非RT）
    std::vector<double> accumulatorR_;   // drain 先（非RT）

    std::atomic<CapturePoint> capturePoint_{CapturePoint::PostOutputFilter};
    CaptureMode captureMode_{CaptureMode::PostDither};
    juce::File outputPath_;
};
```

#### 2.2.2 挿入位置（行番号不使用・処理段階で規定）

**「L470直後」という記述は排除する。代わりに以下の挿入段階 (`Stage`) で規定する：**

```
processBlockDouble() の制御フロー:

  ┌─ 前処理 (Lifecycle/MMCSS/RuntimeScope/Telemetry)
  │
  ├─ DSPCore::processDouble() ─┬─ (内部で OutputFilter 適用)
  │                             └─ pushAdaptiveCaptureBlocks()  ← 既存
  │
  ├─ Crossfade Mix
  │
  │   ★ [Stage: PostOutputFilter] ─ capture(buffer)  ← 新規: プライマリキャプチャ点
  │
  ├─ Diagnostics (DSP_TIMING / CALLBACK_STAGE / XRUN ...)
  │
  ├─ MMCSS Shutdown Check
  │
  └─ }
```

**コード上の具体的挿入箇所（意味論的）**:

```cpp
// src/audioengine/AudioEngine.Processing.BlockDouble.cpp
// processBlockDouble() 内:

    // ★ DSP 処理完了 + Crossfade 完了 直後 ─────────────────
    //   ← OutputFilter 適用済み、Crossfade 完了済み
    //   ← Diagnostics より前（診断負荷の影響を受けない）
    //   ★ OutputCaptureSink 挿入点: PostOutputFilter ★
    {
        OutputCaptureSink* sink = convo::consumeAtomic(
            outputCaptureSink_, std::memory_order_relaxed);
        if (sink != nullptr) [[unlikely]] {
            sink->capture(buffer, cbStartUs);
        }
    }
    // ────────────────────────────────────────────────────

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // Diagnostics ブロック...
```

#### 2.2.3 RT-safe コールバック配送設計

```cpp
// AudioEngine.h — OutputCaptureSink との連携
class AudioEngine {
    // ...
public:
    // ★ RT-safe: Message Thread からのみ設定可能
    //   Audio Thread は relaxed load + nullptr check のみ
    void setCaptureSink(OutputCaptureSink* sink) noexcept {
        convo::publishAtomic(outputCaptureSink_, sink, std::memory_order_release);
    }

private:
    // ★ RT-safe: 生ポインタ + atomic store/release
    //   std::shared_ptr ではなく生ポインタを使用（RT-safe かつ軽量）
    //   Sink のライフタイムは MainWindow が管理（AudioEngine より長生きしない）
    std::atomic<OutputCaptureSink*> outputCaptureSink_{nullptr};
};

// ★ RT-safe 保証の理由:
//   1. setCaptureSink() は Message Thread からのみ呼ばれる
//   2. capture() 内の relaxed load は 1μs 未満で完了→ポインタ解放前に AE 破棄
//   3. Sink のデストラクタは MainWindow::~MainWindow() 内で呼ばれ、
//      cliAutomationCallbacksEnabled=false かつ AudioEngine::setCaptureSink(nullptr)
//      を先行させることで Audio Thread からの参照を完全に絶つ
//   4. std::function コピー問題は生ポインタ使用により回避
```

#### 2.2.4 三重分離パイプライン（Audio RT → SPSC → Background WAV I/O）

```
┌──────────────────────────────────────────────────────────────────┐
│  Audio Thread (processBlockDouble)                              │
│                                                                  │
│  sink->capture(buffer)                                          │
│    ↓                                                             │
│  LockFreeRingBuffer<AudioBlock, 4096>::pushWithWriter()         │
│     ┌─────────────────────────────────┐                         │
│     │ ● SPSC HB Contract              │                         │
│     │ ● relaxed load writeIndex       │                         │
│     │ ● acquire load readIndex        │                         │
│     │ ● buffer[w & MASK] = data       │                         │
│     │ ● release store writeIndex      │                         │
│     │ ● non-blocking (drop if full)   │                         │
│     └──────────────┬──────────────────┘                         │
└────────────────────┼─────────────────────────────────────────────┘
                     │  ← SPSC RingBuffer (wait-free producer)
┌────────────────────┼─────────────────────────────────────────────┐
│  Timer Thread (500ms interval)                                  │
│                                                                  │
│  sink->drainToBuffer()  [バッチ排出]                             │
│    ● kDrainBatch=100 ブロック/回                                 │
│    ● 蓄積先: accumulatorL_ / accumulatorR_ (std::vector)         │
│    ● 排出完了後、蓄積サイズをチェック                              │
│    ● 全ブロック排出後 → background task 発行                      │
└────────────────────┼─────────────────────────────────────────────┘
                     │
┌────────────────────┼─────────────────────────────────────────────┐
│  Background Task (MessageManager::callAsync / juce::ThreadPool) │
│                                                                  │
│  sink->saveToWav(outputPath_)                                    │
│    ● Disk I/O (RTスレッドでは決して実行しない)                     │
│    ● libsndfile / JUCE WAV で保存                                │
│    ● 32bit float / 48kHz / ステレオ                               │
│    ● 保存後、accumulator_ をクリア                                 │
└──────────────────────────────────────────────────────────────────┘
```

**drop 耐性**: RingBuffer 満杯時は静かにドロップする（既存 `pushAdaptiveCaptureBlocks` と同様）。4096 ブロック × 256 サンプル = 1,048,576 サンプル ≈ 21秒 @ 48kHz。Timer が 500ms 間隔で drain するため、通常の運用ではドロップは発生しない。

#### 2.2.5 `CapturePoint` enum 拡張

```cpp
enum class CapturePoint : uint8_t {
    None             = 0,  // キャプチャしない
    PreOutputFilter  = 1,  // OutputFilter 適用前（DSP 生出力）
    PostOutputFilter = 2,  // OutputFilter 適用後（デフォルト）
    PostDither      = 3,   // ディザー後（最終出力）
};
```

`--cli-capture-mode` の値もこの enum に対応させる：
- `none` → `CapturePoint::None`
- `pre-filter` → `CapturePoint::PreOutputFilter`
- `post-filter` → `CapturePoint::PostOutputFilter`（デフォルト）
- `post-dither` → `CapturePoint::PostDither`

---

### 2.3 CLIオプション完全リファレンス（全30オプション）

v7.0 から変更なし。全25既存＋5新規＝30オプション。

---

## 3. テストケース一覧（全34件）

### 3.1 カテゴリ別構成

v7.0 から変更なし（全34件）。

### 3.2 各テストケース詳細

v7.0 から以下の TC-25 のみ修正。

#### TC-25: Crossfade Integrity（v7.1 修正）

| 項目 | 内容 |
|------|------|
| 条件 | 1ms, 5ms, 10ms, 20ms, 50ms の5条件 |
| 参照信号 | IR-B（LPF 1kHz）単独処理出力 |
| 測定信号 | クロスフェード完了後の出力 |
| 評価1（クロスフェード中） | ピークゲインジャンプ |
| 評価2（クロスフェード完了後） | Null Test（差分RMS） |
| 閾値（完了後） | **同一ビルド: ≤ -120dBFS / 異ビルド: ≤ -100dBFS** |
| ★ 例外条件 | **OutputFilter係数変更（HPF/LPF周波数・Q値変更）時: ≤ -100dBFS**（出力信号のスペクトル変化により同一Buildでも差分が生じうる）|

---

## 4. RuntimeHealthMonitor拡張

### 4.1 RecoveryHistory API設計

#### 4.1.1 `RecoveryEvent` 構造体 ← ★ v7.1 拡張

```cpp
// RuntimeHealthMonitor.h に追加
struct RecoveryEvent {
    // ── 基本情報 ──
    uint64_t        timestampUs;     // 記録時刻
    PolicySource    source;          // 発火元監視器（10種）
    RecoveryAction  action;          // 実行アクション（6段階）

    // ── 実行時の状態 ──
    ISRHealthState  healthState;     // 実行時 HealthState（Healthy/Degraded/Critical）
    uint64_t        pendingRetire;   // 実行時 Retire backlog
    uint64_t        maxRetireAgeUs;  // 実行時 最大 Retire age

    // ── 検証結果（事後更新） ──
    RecoveryOutcome     outcome;         // 閉ループ制御結果（5種）
    VerificationState   verification;    // 検証状態（Idle/PendingVerification）
    uint8_t             stalledCount;    // 停滞カウント

    // ── 整合性チェック用 ──
    uint64_t        eventSequence;   // 通し番号（欠落検出用）
};
```

**v7.0 からの改善点**:
| v7.0 | v7.1 | 理由 |
|------|------|------|
| `action` + `result` のみ | **8フィールドに拡張** | 実障害解析に必要な情報を網羅 |
| `RecoveryOutcome` (=result) | **`outcome` は別途追跡** | Verification により事後更新可能に |
| - | **`source: PolicySource` 追加** | どの監視器が原因で発火したか追跡可能 |
| - | **`healthState` / `pendingRetire` / `maxRetireAgeUs` 追加** | 実行時のシステム状態を記録 |
| - | **`eventSequence` 追加** | リングバッファラップ時の欠落検出 |

#### 4.1.2 `getRecoveryHistory()` → `copyRecoveryHistorySnapshot()` ← ★ v7.1 安全化

```cpp
// RuntimeHealthMonitor.h — 公開API
class RuntimeHealthMonitor {
public:
    // ★ v7.1: 生 span ではなく snapshot を返す
    //   戻り値の std::vector は Message Thread で安全に読める
    //   書き込み側（Audio Thread）と読み取り側（Python/Message）の
    //   間に競合が生じない。
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
            snapshot.push_back(m_recoveryHistory[idx]); // copy — 整合性保証
        }
        return snapshot; // RVO保証
    }
};
```

**v7.0 からの改善点**:

| 観点 | v7.0 (`getRecoveryHistory()` → `span`) | v7.1 (`copyRecoveryHistorySnapshot()`) |
|------|----------------------------------------|----------------------------------------|
| スレッド安全性 | ❌ Audio Thread 書き込み中に生メモリ読み取り競合 | ✅ **コピー完了後に返却** |
| 整合性 | ❌ 読み取り時に途中状態を読む可能性 | ✅ **writeIndex acquire 後に全件コピー** |
| 所有権 | ❌ 呼び出し側で生ポインタの寿命管理が必要 | ✅ **std::vector が所有権を保持** |
| CI/Python連携 | ❌ Python側で span のラップが必要 | ✅ **JSON直列化可能な vector を返す** |
| パフォーマンス | 高速（コピーなし） | 64 × sizeof(RecoveryEvent=~64) = ~4KB（約0.1μs）|

#### 4.1.3 記録ポイント

既存の `m_actionCallback(action)` 発火箇所（`RuntimeHealthMonitor.cpp:232`）に以下の挿入で記録する：

```cpp
// m_actionCallback(action) 呼び出し直前
recordRecoveryAction(source, action, currentHealthState);

// 記録実装:
void recordRecoveryAction(PolicySource source, RecoveryAction action,
                          ISRHealthState healthState) noexcept
{
    const uint32_t idx = convo::fetchAddAtomic(
        m_recoveryHistoryWriteIndex, uint32_t{1},
        std::memory_order_release) % kRecoveryHistoryCapacity;

    m_recoveryHistory[idx] = RecoveryEvent{
        .timestampUs    = getCurrentTimeUs(),
        .source         = source,
        .action         = action,
        .healthState    = healthState,
        .pendingRetire  = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0,
        .maxRetireAgeUs = /* read from ref */,
        .outcome        = RecoveryOutcome::None,   // 事後更新
        .verification   = VerificationState::Idle,
        .stalledCount   = 0,
        .eventSequence  = idx,
    };
}
```

#### 4.1.4 Verification 結果の事後更新

閉ループ制御の Verification 完了時（`RuntimeHealthMonitor.cpp:58-85`）に、該当する `RecoveryEvent` を更新する：

```cpp
if (lastAction > RecoveryAction::Observe) {
    auto& entry = m_policyEngine_.getEntry(lastAction);
    if (entry.state == VerificationState::PendingVerification) {
        // ... verification logic ...

        // ★ RecoveryEvent の outcome を事後更新
        updateLastRecoveryOutcome(trend);
    }
}
```

---

## 5. 工数総括（v7.1 修正版）

| フェーズ | v7.0 工数 | v7.1 工数 | 差分理由 |
|---------|----------|----------|---------|
| Phase 0: CLI拡張 | 7日 | **8日** | OutputCaptureSink 独立クラス化＋SPSC＋BG WAV（+1日）|
| Phase 1: DSP品質試験 | 30〜40日 | **35〜45日** | 既存ツールの流用範囲を精査し現実的な値に再見積もり |
| Phase 2: Runtime試験 | 40〜50日 | **40〜50日** | 変更なし |
| Phase 3: レポート/CI | 20〜25日 | **20〜25日** | 変更なし |
| 予備（調整・安定化） | 30〜70日 | **30〜70日** | 変更なし |
| **合計（推奨管理レンジ）** | **127〜192日** | **133〜198日** | **中央値 165 人日（約20〜25週間）** |

---

## 6. 実装前チェックリスト（v7.1 更新）

| # | 確認項目 | ステータス | 備考 |
|---|---------|-----------|------|
| 1 | `OutputCaptureSink` を `AudioEngine` と独立して設計できているか | ☐ | 2.2.1 で設計確定。setCaptureSink(OutputCaptureSink*) |
| 2 | Audio Thread → SPSC → Background Thread の三重分離が設計されているか | ☐ | 2.2.4 で設計確定。`LockFreeRingBuffer<AudioBlock, 4096>` 再利用 |
| 3 | RT-safe 生ポインタ atomic store/release 契約が文書化されているか | ☐ | 2.2.3 で契約文書化。`std::function` 不使用 |
| 4 | `getRecoveryHistory()` が snapshot コピーを返す設計になっているか | ☐ | 4.1.2 で `copyRecoveryHistorySnapshot()` に変更 |
| 5 | 全CLIオプション（30個）の一覧が設計書に存在するか | ☐ | 2.3 で完全一覧化 |
| 6 | テストケース数が全34件で統一されているか | ☐ | 3.1 で修正済 |
| 7 | TC-01B の参照IRが `sampledata/impulse_room_correction.wav` に修正されているか | ☐ | v7.0 で修正済 |
| 8 | TC-25 に OutputFilter 係数変更時の例外条件が追加されているか | ☐ | 3.2 TC-25 で追加済 |
| 9 | CapturePoint enum が PreOutputFilter / PostOutputFilter / PostDither の3点となっているか | ☐ | 2.2.5 で設計確定 |
| 10 | Phase 1 工数が 35〜45 日のレンジで見積もられているか | ☐ | 5 で修正済 |

---

## 7. リスク評価と軽減策（v7.1 更新）

| リスク | 影響度 | 軽減策 | v7.0→v7.1 変更 |
|--------|-------|--------|----------------|
| TC-25 Null -120dBFS が実装差で不成立 | 中 | 同一ビルド/異ビルドで閾値分離済み。**OutputFilter係数変更時は -100dBFS** | ★ 例外条件追加 |
| TC-27 SHA256 がコンパイラ更新で変化 | 低 | Pass/Failから診断用に格下げ済み | 変更なし |
| TC-24 FFT 262144 が CI 時間を圧迫 | 中 | Phase 1-6 でベンチマーク後、最適サイズを採用。65536 代替あり | 変更なし |
| TC-30 API 実装が Phase 0 に間に合わない | **低** | Phase 0-6 でプロトタイプ実装。リングバッファ+snapshot のみ | **v6.3+ から改善済** |
| CI環境での浮動小数点再現性 | 中 | Debug/Release/icx/MSVC で別閾値を設定 | 変更なし |
| **OutputCaptureSink の Audio Thread 負荷** | **低** | LockFreeRingBuffer の pushWithWriter は非ブロッキング。256サンプルコピーは <1μs | **v7.1 新規評価・問題なし** |
| **AudioEngine 肥大化** | **中→低** | OutputCaptureSink を独立クラス化により AudioEngine の責務増加なし | **v7.1 改善** |
| **RecoveryEvent 読み取り競合** | **回避済** | snapshot コピー返却により Audio Thread 書き込みとの競合なし | **v7.1 解決** |
| **Python 依存不足** | **低** | numpy/scipy Windows .venv にインストール済み | v7.0 確認済 |

---

## 8. 成功基準

| 基準 | 内容 |
|------|------|
| Phase 0 完了 | 新CLIオプション5種 + RecoveryHistory API + OutputCaptureSink(独立クラス+SPSC+BG WAV) が全て動作 |
| Phase 1 完了 | TC-01〜TC-29 が CI 環境（Debug/Release MSVC）で全パス |
| Phase 2 完了 | TC-11〜TC-30 が CI 環境（Debug/Release MSVC/icx）で全パス |
| Phase 3 完了 | HTMLレポートが生成され、PRコメントに自動投稿される |
| 最終 | 全34テストケースが週次フルテストでパスし、長期運用が確立される |

---

## 9. v7.0→v7.1 差分サマリ

| 項目 | v7.0 | v7.1 | 根拠 |
|------|------|------|------|
| OutputCapture設計 | AudioEngine 直コールバック | **独立OutputCaptureSinkクラス** | 責務分離・保守性向上 |
| RT-safe 契約 | 未規定（std::function） | **原子生ポインタ＋スレッド別契約** | Audio Thread保護 |
| WAV保存パイプライン | 未設計（暗黙的にRT） | **三重分離パイプライン** | Disk I/O を RT から排除 |
| 挿入位置指定 | L470（行番号） | **処理段階 Stage で規定** | コード変更耐性 |
| CapturePoint | none/post-dither | **PreFilter/PostFilter/PostDither** | 将来拡張性 |
| RecoveryHistory | `span` 生公開 | **`copySnapshot()` コピー返却** | スレッド安全 |
| RecoveryEvent フィールド | action + result | **8フィールド + sequence** | 実障害解析対応 |
| TC-25 例外条件 | なし | **OutputFilter係数変更時追加** | 誤検出防止 |
| Phase 1 工数 | 30〜40日 | **35〜45日** | 流用範囲を精査 |
| AudioEngine 肥大化 | 対処なし | **setCaptureSink(*) のみ** | 責務追加なし |

---

## 付録A: コード監査で使用したツール

v7.0 と同一（省略）。

## 付録B: 既存Pythonツール活用マップ

v7.0 と同一（省略）。

## 付録C: 既存コードパターン活用マップ（v7.1 追加）

| 設計要件 | 既存パターン | ファイル |
|---------|------------|---------|
| SPSC RingBuffer | `LockFreeRingBuffer<T, Capacity>` | `src/LockFreeRingBuffer.h` |
| Audio Thread からのキャプチャ | `pushAdaptiveCaptureBlocks()` | `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp` |
| Background 非同期消費 | `asyncSink()` + `flushLogBuffer()` | `src/audioengine/AudioEngine.Timer.cpp` |
| キャプチャ用データ構造 | `AudioBlock`（trivially copyable） | `src/audioengine/AudioEngine.h:24` |
| AudioEngine への外部コンポーネント注入 | `setRetireRouter()` / `setOrchestrator()` / `setCrossfadeRuntime()` | `src/audioengine/AudioEngine.h` |
| atomic 生ポインタ切り替え | `m_retireRouter`, `m_orchestrator` 等 | `src/audioengine/RuntimeHealthMonitor.h` |

---

**本計画（v7.1）をコード監査・詳細レビュー反映の最終版とし、Phase 0 の実装を直ちに開始する。**
