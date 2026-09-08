# REPAIR_PLAN2-dash2 / I4_DESIGN_CONTRACT 未実装項目 詳細確認報告（read-only）

```text
Date: 2026-09-01 / Type: read-only 調査（Production source 変更: 0 / Test: 0 / CMake: 0 / Build: 0 / CTest: 0）
baseline: ConvoPeq.md Generated 2026-09-01 15:30:13（D161 で確定した現行派生 snapshot）
対象: doc/work88/REPAIR_PLAN2-dash2.md（5391 行）/ doc/work88/I4_DESIGN_CONTRACT.md（7952 行）
方法: 全見出しインベントリ + 明示的未実装マーカー全数抽出（未実装 / not yet implemented / CONFIRMED UNIMPLEMENTED / 実装不在 / 未導入）+ 現行 src 実測照合
位置づけ: 通常開発でのユーザー要求による調査。D159 freeze register との整合判定を含む

─── D175-0 更新記録（2026-09-08・doc-only）──────────────────────────────
本 inventory の 1-C（新規 CR 候補 2 系統）は後続調査で STALE 化が確定したため反映した。
根拠: D163（CR-α/CR-β REJECT・2026-09-06）/ D171-1（OPEN items re-audit・2026-09-07）/
      D174（OPEN candidate triage・call-chain 実測・2026-09-08）。
authority stamp: ConvoPeq.md Generated 2026-09-08 07:15:27（実装 commit 54ba7b40 = D172-3）。
現行結論: genuine OPEN implementation item = 0 件。
  - CR-α 本体（Site 3 warmup backoff）= 実装済み CLOSED（CR-α-1..6・commit 0aeb22ca）
  - Site 2 retry 適用 = DEFER（非 defect・dash2 §1.8 Phase D 仕様通り・将来拡張コメント現役）
  - buildErrorCount_ = DEFER / monitoring（freeze register 補助 trigger 登録済み → §3 更新分参照）
  - CR-β / CW-8 = ALREADY COVERED / STALE（実装済み・production caller 0 は保守的休止）
─────────────────────────────────────────────────────────────────────────
```

## 総合判定

> **【2026-09-01 時点の原文】両文書から、D159 freeze register でカバーされていない未実装項目が 2 系統確認された。**
> 1. **dash2 §1.8 BuildError Phase-2 残部（A-4/A-5 / Amendment 1.8 criteria）** — exponential backoff / retry count / retry telemetry。ソース内コメントで将来拡張として明示記録済み・非 blocking。
> 2. **CW-8 PublishedWorldObservation atomic snapshot contract** — 設計のみ・src 0 件。
>
> これ以外の全項目は「実装済み」「STALE（後続実装で陳腐化）」「D159 DEFER 凍結（Phase-II）」のいずれかに分類され、新たに着手すべき OPEN 項目は上記 2 件のみ。

> **【D175-0 更新・2026-09-08 現行】上記 2 系統はいずれも後続実装で解決済み（STALE 化）:**
> 1. **CR-α 本体（Site 3 warmup backoff / retry count / disposition→delay mapping）= 実装済み CLOSED** — `RetryBackoffPolicy{kDefaultWarmupRetryBackoff={10,80,2}}`・`warmupRetryDecision()` 純関数・唯一の production call site `schedule(req, decision.delayMs)`（RebuildDispatch.cpp:1311）に接続済み・T-CRα-1..4 テスト完備（CR-α-1..6 CLOSED・commit 0aeb22ca・D163 REJECT / D174 call-chain 実測）。**Site 2 retry 適用は DEFER**（非 defect・dash2 §1.8 Phase D 仕様通り・将来拡張コメント現役）。**buildErrorCount_ telemetry は DEFER / monitoring**（§3 更新分の trigger 登録参照）。
> 2. **CR-β / CW-8 = ALREADY COVERED / STALE** — `PublishedWorldObservation` 型（private ctor + friend 構造遮断）+ factory `observePublishedObservation()`（単一 acquire load から {world, &world->publication} 同時確定）+ T-CW8-1..7 テスト + harness 登録が実装済み（commit 0aeb22ca ND-01..04・D163 REJECT / D174 確認）。production caller 0 件は保守的休止。
>
> **現行結論（2026-09-08）: genuine OPEN implementation item = 0 件。** Phase-II 項目は D159 freeze register（D1〜D6 + 補助 trigger）待ち。実装系新規 track は RuntimeBuilder.h:118-124 の trigger 条件成立（設計確定イベント）まで起票禁止（D174 Final Decision）。

---

## 1. REPAIR_PLAN2-dash2 — 項目別照合結果

### 1-A. 実装済み（現行ソースにアンカー実測あり）

| dash2 項目 | 現行ソースの証拠（本日実測） |
|---|---|
| §1.3 LinearRamp RT violation | `resetRT()` + generation handshake（D156-A PASS・実測済み） |
| §1.4 isFullyDrained 上書き全廃 | `Threading.cpp:121` 廃止コメント + `setPendingIntentCount` は TEST-ONLY（h:176）・他 setter 削除済み（h:171 comment） |
| §1.7 currentWorld_ 廃止 | CW-3c 実装済み（RuntimeStore::current 単一 source・INV-ISR-06） |
| §1.8 BuildError 分離（本体） | `BuildErrorPolicy.h`: `FailureClassification` / `RetryDisposition` / **constexpr descriptor table（§1.8.10.3 採用・"★ §1.8.10.3" コメント実在）** + static_assert 2 件 / caller-side policy 配線（RebuildDispatch.cpp:1178・1247）/ `BuildErrorClassificationTests.cpp` 実在 |
| §1.8.9 item 14（無限リトライ防止） | `kMaxRecoveryConsecutiveFailures = 4`（RebuildDispatch.cpp:1067）現役 |
| §1.9 quarantine wake | E-1.9-B event-driven wake 実装済み |
| §2.1 R4 retire 順序 | INV-EPOCH-1/2 保証・FIFO secondary（Retire.cpp「FIFO 強化は実装しない」） |
| §2.2 / A-3 / H.6.3 / H.11.3（A2 Proof/Permit） | **実装済み** — `ISRLifetimeProof.h`（ReclaimPermit/Proof/ShutdownRuntimeIdentity 30 hits）・旧 `bool reclaim(...)` API は削除済み（compile guard・h:817-818）・`tryMakeQuiescenceProof` → Proof → Permit → `reclaimShutdownQuiescent`（AudioEngine.h:4410-4414） |
| §2.2 G14 / H.6.3 后続（identity authority） | `ReclaimIdentity{handle, retireEpoch}` 登録（AudioEngine.h:4391-4403）・`pendingReclaimHandles_.empty()` が isFullyDrained Layer 1 条件（INV-X3-5・Threading.cpp:158-161） |
| §2.2.2 postStopEnqueue tracking | 19 hits 実装済み |
| §2.5 / H.11.4 AdmissionState FSM | **実装済み** — `ISRShutdown.h:171` `enum class AdmissionState` + D101-31-B `AdmissionPackedState`（Open/Closing/Closed/Faulted + version ABA + reservationCount、単一 32-bit atomic CAS） |
| H.6.2 Path B PublicationIntent Gap | `enqueuePublicationIntent` は push 前 fetchAdd の reservation-before-push（h:959-963）で解決済み |
| A-1 mixSmoother RT violation | 解消済み（resetRT、D156-A） |
| §5.16 catch 分類 | `RuntimeBuilder.cpp:441-449`: `bad_alloc → ResourceUnavailable` / `... → InternalError` 現役 |

### 1-B. STALE — dash2 の記述が後続実装で陳腐化（対応する現行実装あり）

| dash2 記述 | 現行状態 |
|---|---|
| §1.2 Recovery coalesce「四次レビュー NO-GO / 🔴 Do not implement」 | **G-4.1→G-4.3 で Phase-I 実装・監査・ST-1 実証済み**（CoalesceIdentity={handle, target}・coalesce CAS・T1/T2/T7/T8/T10 回帰）。旧記述は STALE（S1） |
| §1.8.10.1 static_assert chain「未実装」 | より強い §1.8.10.3 descriptor table が実装済み → 検討案としては役目を終えた（STALE） |
| H.6.3a Build Identity Semantic Comparison「CONFIRMED UNIMPLEMENTED」 | **G-4.1/G-4.2 で実装済み** — SemanticRecoveryTarget 6-field（irIdentityHash / convolutionConfigHash / dspParameterHash が実 hash 値 + buildInputHash = FNV-1a 実計算）。IR hash 比較欠落の指摘は解消。残る `isSemanticSuperset`（包含比較）は Phase-II Supersession = D159 D2 凍結 |
| H.11.27.1 RC-11（Building 中 supersession） | 核心懸念（Building 中の lease 上書き）は **I-HS2（D144/D146）で解決済み**（same-oblId overwrite 禁止・Building 中上書き経路消滅）。pending supersession の full 実装は Phase-II = D2 凍結 |
| GO/NO-GO 表「1.2 coalesce 🔴 Do not implement」 | Phase-I coalesce 実装済みのため STALE（§1.2 と同じ） |
| §5.14「ConvolverFailure/PrepareFailure がどの経路からも生成されない」 | 現在も生成経路なし（下記 1-C-2 として未実装項目として継続管理・解消されてはいない） |

### 1-C. 未実装 — D159 freeze register でカバーされていない項目（新規 CR 候補）

#### 1-C-1. dash2 §1.8 Phase-2 残部 / A-4 / A-5 / Amendment 1.8 Acceptance Criteria（retry backoff・count・telemetry）

> **【D175-0 判定更新・2026-09-08】本項目は STALE — CR-α 本体（Site 3 warmup backoff / count / disposition→delay mapping）は実装済み CLOSED。**
> 根拠: CR-α-1..6 CLOSED（commit 0aeb22ca・CRALPHA6_CLOSURE_REPORT）・D163 REJECT / D174-1 call-chain 実測（classifyBuildError 2 sites・schedule production call site 1 箇所 = RebuildDispatch.cpp:1311 `schedule(req, decision.delayMs)`）。
> **分離後の現行状態**: ① Site 3 backoff = 実装済み（kDefaultWarmupRetryBackoff={10,80,2}・T-CRα-1..4 テスト）② Site 2 retry 適用 = **DEFER**（非 defect・dash2 §1.8 Phase D 仕様通り・将来拡張コメント現役）③ buildErrorCount_ telemetry = **DEFER / monitoring**（§3 更新分 trigger 登録参照）。

**以下は 2026-09-01 時点の原文実測（historical）:**

- `classifyBuildError()` は 2 call site（build 失敗 :1178 / warmup 失敗 :1247）で分類・ログ出力するが、**retry 方針の実適用（backoff 等）は行っていない**。ソース自身が明示: 「retry 方針の実適用（backoff 等）は D-5 RetryBackoffPolicy tuning と併せて将来拡張 — 1.8.9 実装手順」（RebuildDispatch.cpp:1175-1177）
- `retryScheduler_->schedule(req, std::chrono::milliseconds(0))` — **delay 0 固定**（:1256、唯一の call site）。Amendment 1.8 の「exponential backoff（min 1ms, max 100ms）」未実装
- retry count / backoff tuning 構造体 `RetryBackoffPolicy`（H.11.27.6 設計）— src 0 件
- retry telemetry（`retryCount / retryLatency / retryStormDetected`）— src 0 件
- `buildErrorCount_` telemetry — コメント言及のみ（RuntimeBuilder.h:124）、実装 0 件
- **MKLFailure / ConvolverFailure / PrepareFailure は production のどの経路からも生成されない**（catch 拡張 `mkl::exception → MKLFailure` 未実装・convolver `init()` bool は `WarmupFailed` に集約・`DSPCore::prepare()` は依然 `void` で PrepareResult status 型未導入）。enum・table・test には存在するが未使用

**緩和要因（現行安全性）:** recovery warmup retry は `kMaxRecoveryConsecutiveFailures=4` で bounded、scheduled retry は `RetryScheduler`（capacity 8・NonRT worker・NonRT 判定のみ = AC-ISR-1 / BE-8 準拠）、recovery deferred retry は D135-8/9 の budget（kMax=2・dormant）+ event wake で別系統として実装済み。**非 blocking・現行動作に影響なし**。

**D159 freeze register での扱い:** 未登録（D6 は P2/G2/W1 static bound のみ）。→ 新規 CR または freeze register への追記が必要。

**【D175-0 追記】上記「未登録」は解消済み**: buildErrorCount_ の補助 trigger は D175-1 で登録（§3 更新分）。Site 3 backoff 実装は CR-α として独立 closure 済み。

#### 1-C-2. dash2 H.11.27.4 CW-8 — PublishedWorldObservation atomic snapshot contract

> **【D175-0 判定更新・2026-09-08】本項目は STALE / ALREADY COVERED — 実装済み。実装禁止。**
> 根拠: D163 CR-β REJECT / D174-4 確認。実装 anchor（commit 0aeb22ca ND-01..04）: `PublishedWorldObservation` 型（RuntimeWorldAuthority.h — private ctor + friend 構造遮断・trivially copyable・独立構築不能）+ factory `observePublishedObservation(const ReadToken&)`（**単一 acquire load から {world, &world->publication} 同時確定**・未 publish 時 {nullptr,nullptr}）+ `testCW8_PublishedWorldObservation`（T-CW8-1/2/3/4/6/7）+ harness 登録。inventory「src 0 hits」記述は stale（19 hits 実測）。production caller 0 件は保守的休止。

**以下は 2026-09-01 時点の原文実測（historical）:**

- 「`{world, identity}` が同一 publication transaction 由来であることを read-contract として保証」する設計（第十八者 #37-C）。**src 0 hits** — 設計のみ。
- 現行は CW-5（`RuntimeStore::current.identity == RuntimeState::publication.identity`）+ INV-ISR-06（CW-3c 実装済み）で部分カバーされるが、単一 acquire load でのペア取得保証は未実装。
- **D159 freeze register 未登録** → 新規 CR 候補（低リスク・強化系）。

### 1-D. DEFER — D159 freeze register で正式凍結済み（本調査でも trigger 非発生を再確認）

| dash2 項目 | freeze register 対応 | 本日再実測 |
|---|---|---|
| §1.1 R1 MPSC（1.1.1 の `pendingRecoveryAdmission_` MPSC 化含む — D155 §2.3 単独不要） | **D1**（第 2 producer 出現） | Timer/Processor からの recovery API 呼び出し 0 件（D158 実測のまま） |
| §1.5 sparse completion | **D4**（MPSC completion 許容） | `completedOutOfOrder` 未導入 |
| §1.6 X2 wraparound / out-of-order テスト | **D5**（D4 と同時） | INV-X2-6 アンカー維持 |
| H.6.3a の `isSemanticSuperset`（包含比較 supersession） | **D2**（Supersession 要件化 → Phase-II） | equality-only containment（G-4.3-T 回帰）維持 |
| RC-11 の pending supersession full 実装 | **D2** に含む | 同上 |
| §1.8.9 item 15 系の `buildErrorCount_` を含む retry telemetry 強化 | （1-C-1 として新規扱い。実装時は D2 と独立） | — |

---

## 2. I4_DESIGN_CONTRACT — Phase-I / Phase-II 境界照合

規範は I4 §7（D105-R23 収束・2026-08-28）と D159。§7.3 amendment 表の Phase-I status に沿って現行ソースを検証した。

### 2-A. Phase-I production contract — 実装アンカー実測（全在）

| I4 項目 | 現行ソースの証拠 |
|---|---|
| D12.2 SemanticRecoveryTarget 6-field | `ISRRuntimePublicationCoordinator.h:300-315`（operator== = 5 semantic values・domainCoverage は必要条件 metadata 分離） |
| D18.7 CoalesceIdentity = {handle, target} | h:320-327（RecoveryEpisodeId 成分なし） |
| D18.8 Snapshot drift（buildInputHash 実計算） | h:309 + computeBuildInputHash FNV-1a（G-4.2 実装） |
| D17.5 / D22.3 capacity 分離 | `quarantineActiveFlags_[256]`（Q_max）・`kRecoveryIntentQueueCapacity = 256` + durable 1（L_residency_max = 257）・`kMaxLogicalRecoveryObligations = 32`（L_logical_max・INV-CAP-7・h:395） |
| D18.3 / D15.2 disappearance set（R21） | ObligationState: ResolvedSuccess / ResolvedFailed / ResolvedStaleSuperseded / ResolvedSuperseded（dormant）+ RetryExhaustedDiscard（Orchestrator 側 dormant guard・kMax=2） |
| D14.3 / D29.8 MARK-TRANSIENT-FAILURE non-terminal | markTransientFailure → delivery=None + redrive（G-4.4-P1/P2/P3 実装・wake 配線済み `redriveWakePending_` h:1113） |
| D21 / D24 backpressure liveness | D137 P2 wake プロトコル + rearmRecoveryRetry（h:599）実装 |
| D22.2 / D26.2 単一 counter model | `liveCount_ < kCapacity` ゲート（tryInsert・G-4.x 監査済み） |
| D27.2 COALESCE vs terminal linearization | coalesce CAS（同一 slots_[i].state atomic・G-4.3-R 監査 PASS） |
| D35/D38 の code-side anchor | capacity 定数上記のとおり実在（proof 自体は文書・実装対象外） |

### 2-B. Phase-II deferred — §7.5 の全項目が production から不在（D159 D2/D3 凍結と一致）

| §7.5 項目 | 本日実測 |
|---|---|
| `RecoveryEpisodeId` 型 + counter | production 0 件（design-only コメントのみ） |
| `GlobalRecoveryBudget` API / acquireTentative | 0 件 |
| Recovery Episode `EpisodeAdmissionState`（episode 単位 OPEN/CLOSED） | 0 件 — ※ `ISRShutdown.h` の `AdmissionState` は shutdown admission の別物（D101-31-B 実装済み）であり混同しないこと |
| E_max / O_max / E×O ≤ 32 | NOT A PHASE-I INVARIANT（§7.2 確定どおり） |
| D19.1 / D20 / D23 closure linearization（episode 層） | episode 抽象自体が不在のため構造的に対象外 |
| T12 / T25-T27 / T35-T37 等 episode test matrix | 未実装（Phase-II。Phase-I の T13/T17/T21 相当は G-4.3-T T1/T2/T7/T8/T10 として実装済み） |

→ **§7.5 の Phase-II deferred はすべて D159 freeze register（D2: Supersession 要件化 / D3: Phase-II 実装開始 / D13 補助 trigger）でカバー済み。着手は不適。**

### 2-C. I4 telemetry counter 名（設計のみ・同等観測は実装あり）

I4 設計名の `terminalDispositionCount` / `retryExhaustedCount` / `recoverClosedEpisodeRejectCount_` 等は src 0 件。ただし R23 §7.1 自身が「episode 系 telemetry は設計のみ」と認定しており、 obligation レベルの同等観測は実装済み: `recoveryCoalescedCount_` / `recoveryCapacityExhaustedCount_` / `recoveryObligationShutdownDiscardCount_` / `recoveryRetryDeferredCount_` / `recoveryIntentDropCount_` / `recoveryShutdownDiscardCount_`（h:196-240・626-636 実測）。→ **非 blocking・設計名の違いのみ。** 完全一致が必要になった時点で別途判断（Phase-II 着手時に episode 系とまとめて処理）。

---

## 3. 結論と推奨処理

> **【D175-0 更新・2026-09-08 現行】**
>
> | 分類 | 件数 | 処理 |
> |---|---|---|
> | 実装済み（dash2 本体 + A 系 + H 系 findings + **CR-α Site 3 backoff + CW-8**） | 19 項目 | なし（closure 済み） |
> | STALE（後続実装で陳腐化した記述。**1-C-1 本体・1-C-2 を含む**） | 8 項目 | なし（historical。D159 規則どおり） |
> | DEFER（D159 凍結と一致 + **Site 2 retry 適用** + **buildErrorCount_ telemetry**） | 8 項目 | なし（trigger 待ち・monitoring） |
> | **未実装・未登録（新規 CR 候補）** | **0 系統** | **— genuine OPEN implementation item = 0 件（D174 Final Decision）** |
>
> **buildErrorCount_ の freeze register 補助 trigger（D175-1 登録・2026-09-08）:**
>
> ```text
> buildErrorCount_ telemetry（集約 build failure counter）
>     ↓ trigger 未発生（現行）
> DEFER / monitoring — 追加実装は行わない
>     ↓ trigger 発生時のみ Phase-II 再評価
> ```
>
> **trigger 条件**: convolver / prepare の実 failure が production 経路から実際に観測可能になり、かつ subsystem 別 retry policy が必要であることが設計として確定した場合（RuntimeBuilder.h:118-124 の仕様記述と同一）。現状 MKLFailure / ConvolverFailure / PrepareFailure は生成経路なしの休眠分類であり、観測ギャップは既存 telemetry（REBUILD_TELEMETRY + classifyBuildError ログ + Site 3 Exhausted terminal）で埋まっている。
>
> **D174 判定（2026-09-08）のまま**: BuildError Phase-II / CW-8 / Site 2 retry の実装系新規 track 起票は禁止。doc-only maintenance（本更新）が着手可能な作業のすべて。

**以下は 2026-09-01 時点の原文結論（historical — 1-C は上記更新により STALE/DEFER 化済み）:**

| 分類 | 件数 | 処理 |
|---|---|---|
| 実装済み（dash2 本体 + A 系 + H 系 findings） | 17 項目 | なし（closure 済み） |
| STALE（後続実装で陳腐化した dash2 記述） | 6 項目 | なし（historical。dash2 を authority としない D159 規則どおり） |
| DEFER（D159 凍結と一致） | 6 項目 | なし（trigger 待ち） |
| **未実装・未登録（新規 CR 候補）** | **2 系統** | **ユーザーの判断を待つ** |

**新規 CR 候補（いずれも非 blocking・現行安全性は緩和済み）:**

1. **CR 候補 α: BuildError retry backoff / count / telemetry**（dash2 §1.8.9 Phase-2 残部 + A-4/A-5 + Amendment 1.8 criteria + H.11.27.6 RetryBackoffPolicy）。現行は即時 retry（delay=0）・bounded（kMax=4 / capacity 8）で動作。実装する場合は H.11.27.6 の tuning parameter 設計（RT で待たせない・non-blocking）と Amendment 1.8 の AC を仕様として使用可能。ソース内に将来拡張コメントあり（RebuildDispatch.cpp:1175-1177）。
2. **CR 候補 β: CW-8 PublishedWorldObservation atomic snapshot contract**（H.11.27.4）。`{world, identity}` ペアの単一 acquire load 保証。INV-ISR-06 の上強化・影響範囲は RuntimeWorldAuthority / RuntimeStore の read path。

上記を着手する場合は D159 通常開発サイクル（scope 確認 → 境界確認 → Invariant 影響 → 実装 → targeted test）に従う。着手しない場合は、D159 freeze register の補助 trigger として登録する（記録・監視のみ）ことを推奨する。

---

## 4. 本調査で使用した実測コマンド系譜

- grep: `RecoveryEpisodeId`(0) / `GlobalRecoveryBudget|EpisodeAdmissionState|acquireTentative`(0) / `postStopEnqueue`(19) / `PublishedWorldObservation`(0) / `retryStormDetected|retryLatency`(0) / `buildErrorCount_`(1: コメント) / `BuildError::MKLFailure`(1: table のみ)
- アンカー確認: BuildErrorPolicy.h（descriptor table + static_assert）/ RetryScheduler.h（kCapacity=8・delay 引数あり）/ RebuildDispatch.cpp:1067（kMax=4）:1178・1247（classifyBuildError）:1256（schedule 0ms）/ ISRShutdown.h:171・365（AdmissionState + packedState）/ ISRLifetimeProof.h（Permit/Proof 30 hits）/ AudioEngine.h:4391-4414（ReclaimIdentity・Proof→Permit）/ ISRRuntimePublicationCoordinator.h:300-327・395・626-636（D12.2 / D18.7 / capacity / telemetry）
