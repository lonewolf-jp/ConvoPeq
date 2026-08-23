# D101-9 Step 5-VI-D — HealthMonitor Threshold Contract Tests

> **Status**: COMPLETE — 17 test functions / 72 checks, **0 failures**; full suite
> **35/35 PASS on Debug and Release** (34 pre-existing + 1 new contract-test executable).
> **New files**: `src/tests/RuntimeHealthMonitorTierTests.cpp`,
> `tools/vi_c_msbuild_test.bat` (verification helper).
> **Production change**: ONE line in `RuntimeHealthMonitor.h`
> (`friend struct RuntimeHealthMonitorTierTestAccess;` — the minimal test-only seam,
> priority 3 of Step 5-VI-D §D-14). No behavior change.
> **Raw output**: `evidence/_vi_c_d8-tier-direct.txt` (72 PASS lines).

---

## 1. 実コード再監査結果 (D-0)

5-VI-C 報告書と実ソースを marker 照合（29 項目）→ **差異ゼロ**。
`evaluateRetireChainTiers()` 本体を全文再読し、以下を確認:

- Tier2 → bootstrap guard → signed delta → Tier3 → Tier4 episode latch + 10s periodic
  evidence (`kStuckEvidenceIntervalUs`) → Tier5 3-case machine (cap 2) → N=1 silent exit の順序
- `emitTerminalChainEvent()` は Severity=Error 固定・stuck cache 時のみ reader 3 フィールド充填

## 2. Test fixture 設計 (D-14)

### 2.1 Seam 選定

| 優先度 | 手段 | 判定 |
| --- | --- | --- |
| 1 既存 public seam | `setEventCallback` / `reset` / ctor は存在。ただし tier 機械は private | 不十分 |
| 2 既存 fake router | `setRetireRouter(isr::ISRRetireRouter*)` は**具象型ポインタ**、かつ terminal accessor 群 (`terminalStoreCount()` 等) は**非 virtual** header-inline → fake で intercept 不能 | 不可能 |
| **3 最小 test-only seam** | **`friend struct RuntimeHealthMonitorTierTestAccess;` 1 行** | ✅ 採用 |
| 4 production API 追加 | — | 見送り |

### 2.2 Access struct が提供する操作

`feedTick(now)`（= evaluateRetireChainTiers(now, m_prevTickSnapshot_) + prev roll +
valid flag set — tick() 内呼出と同一経路）/ `primePrev` / `invalidatePrev` /
latch・ticks・timer の状態読み書き / stuck diagnosis cache 操作。

### 2.3 Link stubs

テストは `tick()` を一切呼ばないため、`RuntimeHealthMonitor.cpp` 内の無関係関数
（tick/checkPublicationStall/checkRetireStall）が参照する engine 外部シンボル
（`RuntimePolicyEngine` ×12、`RecoveryBudget` ×6、`isr::LifetimeState::pendingIntentCount`、
`isr::RuntimeIntentCoordinator::getPublicationBacklogCount`、
`isr::RuntimePublicationOrchestrator::getMaxDeferredAgeMs`）をテスト TU 内で空実装スタブ。
production は無変更。完全型のため `ISRRetire.h` / `ISRRuntimePublicationCoordinator.h` /
`RuntimePublicationOrchestrator.h` を include。

## 3. テストケース一覧（17 関数 / 72 checks）

| # | 関数 | 対象 Gate | 主アサーション |
| --- | --- | --- | --- |
| 1 | `testBootstrapSkip` | D1 | 初回 tick: E>0・全 delta 正でも発火は 1014 のみ。1015/1016/1017 無し・非 latch |
| 2 | `testTier2EngageClearReengage` | D2 | E 0→1 で 1014 Warning 1 回。1→1 で再発火なし。1→0 で無音。0→1 で再発火 |
| 3 | `testTier3OverflowDelta` | D3 | overflow 10→11 で 1015（value=累積 11）。11→12 連続増加では Warning 状態維持により再発火なし。12→12 無音 |
| 4 | `testTier4SingleAdmissionAndLatch` | D4/D5 | store 100→101 で 1016 Error ×1 + latch。101→102 で再発火なし（episode entry） |
| 5 | `testTier4PeriodicEvidence` | D6 | 遷移 tick 発火後 <10s 無音 → ≥10s で evidence 再送 → 直後 tick 抑制（double-fire suppression） |
| 6 | `testTier5PositiveX2` | D7 | res 0→1: ticks=1 無イベント。1→2: ticks=2 で 1017 latch。以降再発火なし |
| 7 | `testTier5PlateauReset` | D8 | 1→2(ticks=1)→2(**reset**)→3(ticks=1)。1017 不発火 |
| 8 | `testTier5DrainReset` | D8 | +1(ticks=1)→−1(**reset**, 正常ドレイン)→+1(ticks=1)。1017 不発火 |
| 9 | `testIndependenceAdmissionWithoutGrowth` | D9 | Δstore>0 & res flat → 1016 のみ・ticks=0 |
| 10 | `testIndependenceGrowthWithoutAdmission` | D9 | store flat & res 0→1→2 → 1017 のみ・1016 無し |
| 11 | `testEpisodeExitAndRearm` | D10 | latch 中に resident==0 ∧ dStore==0 → 両 latch+ticks clear（無音）。次 admission で 1016 再発火（計 2 回） |
| 12 | `testCorrelationCorrelated` | D11 | stuck cache 有 → readerIndex=7 / readerEpoch=555 / residencyTimeUs=123456 を伝播 |
| 13 | `testCorrelationSuspected` | D11 | cache 無・readers>0・minEpoch停滞 → 1016 発火するが reader フィールド未設定（独自 verdict を作らない） |
| 14 | `testCorrelationUncorrelated` | D11 | readers=0・epoch 変動 → 発火（cause open）・フィールド未設定 |
| 15 | `testResetHygiene` | D13 | 全状態構築後 reset() → latch/ticks/cache 清掃。続行 tick は bootstrap 扱い（carry-over delta で 1015-1017 不発火）。新 episode は正常開始 |
| 16 | `testPayloadValueSemantics` | D14 | 1014=E resident / 1015=累積 overflow / 1016=累積 store / 1017=latch 時点の resident |
| 17 | `testSignedDeltaLargeValues` | （追加） | u64 巨大値（2^62）でも符号付き delta は正しく正方向判定 |

静的検証（コード grep）: Terminal event から `quarantineReader()` 呼び出しゼロ（D12）、
`terminalPeakResident` / `4092` / `8192` / S=2 / T_stall_design の出現ゼロ（§D-15）、
1014–1017 は `onHealthEvent()` の `[TERMINAL_EVIDENCE]` handler へ routing 済み
（5-VI-C 実装・Timer.cpp、evidence 項目 T_store/T_resident/pend/E_resident/readers/minEpoch/readerIdx）。

## 4. 実装した test files

- `src/tests/RuntimeHealthMonitorTierTests.cpp`（新規・~620 行）
- `src/audioengine/RuntimeHealthMonitor.h` — friend 宣言 1 行追加（唯一の production 差分）
- `CMakeLists.txt` — `RuntimeHealthMonitorTierTests` ターゲット（sources: test cpp +
  RuntimeHealthMonitor.cpp + ISRRetireRouter.cpp、JUCE link、NOMINMAX/WIN32_LEAN_AND_MEAN 定義、
  `src/eqprocessor` include path、**add_dependencies(ConvoPeq) は意図的に付けない**）
- `tools/vi_c_msbuild_test.bat` — MSVC build/ 検証ヘルパー

## 5. Debug 結果

```text
BUILD_EXIT=0
CTEST_EXIT=0
100% tests passed out of 35
Tier tests: 72 checks, 0 failures   (_vi_c_d8-tier-direct.txt)
```

## 6. Release 結果

```text
BUILD_EXIT=0
CTEST_EXIT=0
100% tests passed out of 35
```

## 7. 全既存テスト結果

既存 34 テスト + 新規 1 = **35/35 PASS（Debug・Release 両方）**。
ベースライン（5-VI-C 直後）も 34/34 PASS であり、回帰ゼロ。

## 8. D1–D16 判定

| Gate | 条件 | 判定 | 根拠テスト |
| --- | --- | --- | --- |
| D1 | Bootstrap で Tier3–5 不発火 | **PASS** | #1 |
| D2 | Tier2 absolute gauge transition | **PASS** | #2 |
| D3 | Tier3 delta は Q+E aggregate 扱い | **PASS** | #3（テスト名・コメントとも aggregate 明記） |
| D4 | Tier4 Δstore>0 で episode entry | **PASS** | #4 |
| D5 | episode 中再発火なし | **PASS** | #4/#5 |
| D6 | 10s periodic double-fire なし | **PASS** | #5 |
| D7 | positive ×2 のみで latch | **PASS** | #6 |
| D8 | plateau/drain で reset | **PASS** | #7/#8 |
| D9 | Tier4/Tier5 独立 | **PASS** | #9/#10 |
| D10 | exit/re-arm | **PASS** | #11 |
| D11 | correlation 3 分類 | **PASS** | #12/#13/#14 |
| D12 | Terminal event から quarantine されない | **PASS** | 静的 grep（呼び出しゼロ）+#12–14（event は evidence のみ） |
| D13 | reset 全 state 初期化 | **PASS** | #15 |
| D14 | evidence routing 1014–1017 | **PASS** | #16（payload 契約）+ 静的確認（handler block 存在） |
| D15 | 既存 34/34 維持 | **PASS** | §5/§7 |
| D16 | Debug/Release build PASS | **PASS** | BUILD_EXIT=0 × 2 |

## 9. 残存 OPEN 項目

1. Episode-exit 保持期間 N: N=1 暫定実装のまま。5-VI-E の T1–T4 回帰データで妥当性確認。
2. `ISRHealthState::Critical` wiring: 未接線（Error イベントは自動 Critical にならないことを
   cpp:414-457 の列挙で確認済み）。5-VI-E 後の独立判断。
3. `EVENT_TERMINAL_EPISODE_CLEARED` (1018): 未実装（silent exit）。
4. `[TERMINAL_EVIDENCE]` handler（AudioEngine.Timer.cpp）の runtime test は本ステップ対象外
   （AudioEngine 丸ごと必要になるため）。静的確認のみ。5-VI-E の T3/T4 ログで実働を確認する。
5. `quarantineResidentCount()` の Q-store/auxiliary 分離（Step 5-V §5 提案）は未着手。
