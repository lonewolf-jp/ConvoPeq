# D101-9 Step 5-VI-C — Change Inventory (pre-implementation)

> 作成時点: 実装前。5-VI-B 設計を実コードへ落込む変更点の完全インベントリ。
> **このドキュメント作成時点で production code は未変更。**

## 0. Baseline status (C-1)

| 項目 | 結果 |
| --- | --- |
| `git diff` RuntimeHealthMonitor.h/.cpp, RuntimePolicyEngine.h | **0 diff** ✅ |
| Baseline Debug build (build/ MSVC cl + oneAPI) | BUILD_EXIT=0 |
| Baseline Debug ctest | **34/34 全 PASS** (`_vi_c_base-ctest.log`) |
| Baseline Release build + ctest | 実行中（完了後に記録） |

備考:

- `build-icx` (Ninja/icx) は既知の `ConvoPeq.exe` icx-Debug リンク失敗 (`_CrtDbgReport`) を持ち、
  9 テストターゲットが `add_dependencies(<Test> ConvoPeq)` により連鎖スキップされるため、
  検証は **MSVC `build/` ディレクトリ**（過去タスクと同じ手法）を使用する。
- テスト #30 HeadlessAudioPathVerification は PowerShell スクリプトテスト（ビルド対象外）。
- テスト実行可能ターゲット名は ninja phony より確定済み（例: テスト名 OwnerChannel →
  ターゲット `OwnerChannelTests`）。

## 1. 変更ファイル一覧（4 ファイル・全て設計書 §Handoff 対応）

| # | File | 変更種別 |
| --- | --- | --- |
| 1 | `src/audioengine/RuntimePolicyEngine.h` | TrendSnapshot +5 fields |
| 2 | `src/audioengine/RuntimeHealthMonitor.h` | イベントコード4種・メソッド宣言2種・状態メンバ群 |
| 3 | `src/audioengine/RuntimeHealthMonitor.cpp` | takeSnapshot 拡張 / tick() 呼出 / evaluateRetireChainTiers 実装 / emitTerminalChainEvent 実装 / reset() 拡張 |
| 4 | `src/audioengine/AudioEngine.Timer.cpp` | onHealthEvent へ Terminal 系コードの evidence ハンドラ追加（既存 handler 責務に乗る・C16 の文書化された例外） |

変更禁止リスト（5-VI-C 指示）との照合: computeTrend / updateHealthState 既存判定 /
evaluateRetirePressureLevelNoRt / PublicationAdmission / Timer tick state /
ISRRetireRouter reclaim/storage semantics / quarantineReader / EVENT_READER_STUCK recovery /
terminalReclaim ownership contract / terminalPeakResident / K_terminal / S=2 / T_stall_design
— **全て不変**。

## 2. Symbol-level changes

### 2.1 `RuntimePolicyEngine.h` — `struct TrendSnapshot` (L71)

末尾に追加:

```cpp
// ★ D101-9 Step 5-VI-C: retire-chain raw observations (5-VI-B §B-2).
//   Raw values ONLY — deltas are computed solely by
//   RuntimeHealthMonitor::evaluateRetireChainTiers().
std::uint64_t terminalStoreCount{0};
std::uint64_t terminalReclaimResidentCount{0};
std::uint64_t emergencyQuarantineResidentCount{0};
std::uint64_t quarantineOverflowCount{0};
std::uint64_t minReaderEpoch{0};
```

### 2.2 `RuntimeHealthMonitor.h`

(a) イベントコード定数 — `EVENT_OVERFLOW_RATE_CRITICAL` (L63) の後に追加:

```cpp
// ★ D101-9 Step 5-VI-C: Terminal/quarantine chain evidence codes
//   (1xxx retire-chain family continuation; free numbers verified in 5-VI-B §B-8)
static constexpr uint32_t EVENT_EMERGENCY_Q_ENGAGED          = 1014;  // Tier 2 Warning
static constexpr uint32_t EVENT_QUARANTINE_OVERFLOW_DETECTED = 1015;  // Tier 3 Warning (Q+E aggregate)
static constexpr uint32_t EVENT_TERMINAL_ADMISSION           = 1016;  // Tier 4 Error
static constexpr uint32_t EVENT_TERMINAL_GROWTH_SUSTAINED    = 1017;  // Tier 5 Error
```

(b) メソッド宣言 — `computeTrend` 宣言 (L283-284) の後に追加:

```cpp
// ★ D101-9 Step 5-VI-C: sole delta-evaluation site for the retire spill chain
void evaluateRetireChainTiers(const TrendSnapshot& now,
                              const TrendSnapshot& prev) noexcept;
void emitTerminalChainEvent(uint32_t eventCode, uint64_t value) noexcept;
```

(c) private members — `m_prevRetireAgeState` (L302) の後に追加:

```cpp
// ★ D101-9 Step 5-VI-C: retire-chain tier state (Message Thread only — non-atomic)
MonitorState m_prevEmergencyQState_{MonitorState::Normal};
MonitorState m_prevQuarantineOverflowState_{MonitorState::Normal};
TrendSnapshot m_prevTickSnapshot_{};          // authoritative previous sample (B-3)
bool m_prevTickSnapshotValid_{false};         // bootstrap guard (Tier 3-5 skip on first tick)
bool m_terminalAdmissionLatched_{false};      // Tier 4 episode latch
bool m_terminalGrowthSustainedLatched_{false};// Tier 5 latch
std::uint8_t m_terminalGrowthTicks_{0};       // consecutive Δresident>0 count (cap 2)
std::uint64_t m_lastTerminalEvidenceUs_{0};   // 10 s periodic evidence timer (episode)
// Correlation cache — filled by takeSnapshot() (sole raw-read site), consumed on the rare
// event-emission path. mutable because takeSnapshot() is const.
struct CachedStuckDiagnosis {
    bool isStuck{false};
    int32_t readerIndex{-1};
    std::uint64_t readerEpoch{0};
    std::uint64_t residencyTimeUs{0};
};
mutable CachedStuckDiagnosis m_lastStuckDiagnosis_{};
```

注: `m_prevMinReaderEpoch_` は **削減**（指示 C-5）— `prev.minReaderEpoch` /
`now.minReaderEpoch` のスナップショットペアで相関判定可能なため保持しない。

### 2.3 `RuntimeHealthMonitor.cpp`

(a) `takeSnapshot()` — `if (m_retireRouter)` ブロック内 (`stuckInfo` 取得の直後、L621 相当)
に追加:

```cpp
snap.terminalStoreCount = m_retireRouter->terminalStoreCount();
snap.terminalReclaimResidentCount = m_retireRouter->terminalReclaimResidentCount();
snap.emergencyQuarantineResidentCount = m_retireRouter->emergencyQuarantineResidentCount();
snap.quarantineOverflowCount = m_retireRouter->quarantineOverflowCount();
snap.minReaderEpoch = m_retireRouter->minReaderEpoch();
// correlation cache — same-tick diagnosis reuse (no second detection pass)
m_lastStuckDiagnosis_.isStuck = stuckInfo.isStuck;
m_lastStuckDiagnosis_.readerIndex = stuckInfo.readerIndex;
m_lastStuckDiagnosis_.readerEpoch = stuckInfo.readerEpoch;
m_lastStuckDiagnosis_.residencyTimeUs = stuckInfo.residencyTimeUs;
```

(b) `tick()` — `checkRetireStall();` (L20) の直後に挿入:

```cpp
// ★ D101-9 Step 5-VI-C: retire-chain tier evaluation (sole raw-read via takeSnapshot)
{
    const TrendSnapshot chainNow = takeSnapshot();
    evaluateRetireChainTiers(chainNow, m_prevTickSnapshot_);
    m_prevTickSnapshot_ = chainNow;
    m_prevTickSnapshotValid_ = true;
}
```

(c) 新関数実装 — `takeSnapshot()` 定義の後 (L650 付近) に追加。内容:

- Tier 2: `emitOnTransition(m_prevEmergencyQState_, E>0 ? Warning : Normal, Warning,
  EVENT_EMERGENCY_Q_ENGAGED, now.emergencyQuarantineResidentCount)` — bootstrap tick でも評価
- bootstrap guard: `if (!m_prevTickSnapshotValid_) return;`
- signed delta ×3（`int64_t(now) − int64_t(prev)` — unsigned 減算禁止）
- Tier 3: `emitOnTransition(m_prevQuarantineOverflowState_, dOverflow>0 ? Warning : Normal,
  Warning, EVENT_QUARANTINE_OVERFLOW_DETECTED, now.quarantineOverflowCount)`
- Tier 4: `dStore>0 && !latched` → latch + `emitTerminalChainEvent(1016,
  now.terminalStoreCount)` + evidence timer start / latched 中は 10s 周期で同イベント再送
  (`kStuckEvidenceIntervalUs`)
- Tier 5: `dResident>0 → growthTicks=min(+1,2)` / else reset。`growthTicks==2 && !latched`
  → latch + `emitTerminalChainEvent(1017, now.terminalReclaimResidentCount)`
- Episode exit (N=1 暫定): latched ∧ `resident==0 ∧ dStore==0` → 3 state clear（無音・1018 不実装）

(d) `emitTerminalChainEvent` 実装: Severity=Error 固定、`m_lastStuckDiagnosis_.isStuck`
なら readerIndex/readerEpoch/residencyTimeUs を充填（correlated）、callback 発火。

(e) `reset()` — 末尾（`m_criticalExitStableStartUs_ = 0;` の後）に追加:

```cpp
// ★ D101-9 Step 5-VI-C: retire-chain tier state reset (no cross-episode carryover)
m_prevEmergencyQState_ = MonitorState::Normal;
m_prevQuarantineOverflowState_ = MonitorState::Normal;
m_prevTickSnapshot_ = TrendSnapshot{};
m_prevTickSnapshotValid_ = false;
m_terminalAdmissionLatched_ = false;
m_terminalGrowthSustainedLatched_ = false;
m_terminalGrowthTicks_ = 0;
m_lastTerminalEvidenceUs_ = 0;
m_lastStuckDiagnosis_ = CachedStuckDiagnosis{};
```

### 2.4 `AudioEngine.Timer.cpp` — `onHealthEvent()` (L1586)

冒頭の汎用 `[HEALTH]` diagLog の直後に挿入（既存 handler 責務パターンに一致）:

```cpp
// ★ D101-9 Step 5-VI-C: Terminal/quarantine chain evidence.
//   Evidence-only — NO recovery action here; recovery stays exclusively on the
//   reader-stuck path (EVENT_READER_STUCK → quarantineReader).
if (event.eventCode == convo::EVENT_EMERGENCY_Q_ENGAGED
    || event.eventCode == convo::EVENT_QUARANTINE_OVERFLOW_DETECTED
    || event.eventCode == convo::EVENT_TERMINAL_ADMISSION
    || event.eventCode == convo::EVENT_TERMINAL_GROWTH_SUSTAINED)
{
    const auto tp = getRuntimeBackpressureTelemetry();  // single same-tick snapshot read
    diagLog("[TERMINAL_EVIDENCE] code=" + juce::String(static_cast<int>(event.eventCode))
        + " T_store=" + juce::String(static_cast<juce::int64>(tp.terminalStoreCount))
        + " T_resident=" + juce::String(static_cast<juce::int64>(tp.terminalResident))
        + " pend=" + juce::String(static_cast<juce::int64>(tp.pendingRetireCount))
        + " E_resident=" + juce::String(static_cast<juce::int64>(tp.emergencyQuarantineResident))
        + " readers=" + juce::String(static_cast<int>(tp.activeReaderCount))
        + " minEpoch=" + juce::String(static_cast<juce::int64>(tp.minReaderEpoch))
        + " readerIdx=" + juce::String(static_cast<int>(event.readerIndex)));
    return;
}
```

使用する telemetry フィールド名は `AudioEngine.h:1561` の
`RuntimeBackpressureTelemetry` 実定義から確認済み（pendingRetireCount /
emergencyQuarantineResident / activeReaderCount / minReaderEpoch / terminalStoreCount /
terminalResident）。

## 3. ISRHealthState wiring — 調査結果と判断

`updateHealthState(const PolicyDecision&)` (cpp:414-457) は特定の `m_prev*State_`
（retire/publication/overflowRate/readerSlot/retireAge）のみを列挙して Critical/Degraded
を決定する。**新規 Terminal tier 状態はこの列挙に含めない** → Error イベントが自動的に
Critical になることはない（指示どおり仮定で接線しない）。今回の wiring は見送り、
open item として 5-VI-E 後の判断に残す。

## 4. Gate C1–C16 に対する設計上の適合

C1✓(5 fieldsのみ) C2✓(raw read は takeSnapshot のみ) C3✓(delta は evaluateRetireChainTiers
のみ) C4✓(bootstrap guard) C5✓(dStore>0 のみ) C6✓(×2 consecutive) C7✓(==0/<0 reset)
C8✓(peak 未読取) C9✓(K 比較なし) C10✓(直接 quarantineReader 呼びなし) C11✓(verdict は
detectStuckReaders(10) のみ) C12✓(reset 全項目) C13/C14→実装後検証 C15✓(diagnostics macro
非依存) C16△→RuntimeHealthMonitor.* + PolicyEngine.h + Timer.cpp handler 1 ブロック
（文書化された例外・既存 handler 責務内）
