# D156 — Residual Open-Item Reconciliation Audit（read-only）

**Date:** 2026-09-01 (+09:00)
**Type:** 完全 read-only 統合監査。**Production source: 0 / Test source: 0 / CMake: 0 / build: 0 / CTest: 0。**
**Baseline:** `ConvoPeq.md` `Generated: 2026-08-31 23:39:12`（T3c Close Audit / D155 と同一 — ソース変更なし）。
**目的:** REPAIR_PLAN2-dash2 等に残る「将来対応・未実装・修正 GO」記述が、**現行 baseline の実コードにも本当に存在するのか**を再判定する。原則「設計書に未実装と書いてある ≠ 現行ソースで未実装」。

---

## 0. 総合判定（先出し）

> ## **OPEN = 0 件 — 残存項目はすべて DEFER（trigger 待ち）・CLOSED・STALE のいずれか**

着手すべき実装は **0 件**。次工程は「Project Open Items = 0 確認 → 全体統合 Close Audit」への進行が妥当。

---

## 1. 項目別分類表（指示の表 + 完全版）

| Item | REPAIR_PLAN-dash2 status | Current source（baseline 実測） | Classification | Next action |
|---|---|---|---|---|
| **1.3 LinearRamp / mixSmoother RT violation** | 対象外・文書化（本文書内 8/14 再検証で「解消済み」を既に記載） | `Runtime.cpp:360` = `activeMixSmoother.resetRT()`（DspNumericPolicy.h:341 `ASSERT_AUDIO_THREAD` 付き RT-safe 版）+ `mixSmootherResetPendingGen` generation handshake（Lifecycle.cpp:489 / Runtime.cpp:341-344 HB 結線）。`reset()` は prepareToPlay 内で規約どおり NonRT | **CLOSED（STALE 清掃対象）** | なし（dash2 記述は既に自己訂正済み） |
| **1.4 isFullyDrained 実測上書き全廃** | 🟡 設計先行必須（16-condition drain semantic） | **実装済み** — `AudioEngine.Threading.cpp:118` は実測値直接判定（retireDepth / lifetimeRetireIntentPending / ring / quarantine×3 / terminal / hasDeferredCommit / `pendingReclaimHandles_.empty()` / coordinator isFullyDrained）。絶対値上書き（setPendingIntentCount 等）は P2-1 §1.1.5 で廃止・TEST-ONLY 化 | **CLOSED** | なし（D107/D108 が 16-condition 全網羅を確認済み） |
| **1.1 R1 MPSC 化** | 🟢 条件付き GO — Phase 5 将来拡張 | D155 実測: SPSC invariant 成立（producer = CL のみ / consumer = RebuildThread のみ）・Timer からの recovery API 呼び出し 0 件 | **DEFER** | trigger 待ち（第 2 producer 出現時に R1 設計開始 — D155 §2.5 に変更量記録済み） |
| **1.2 Recovery coalesce** | 四次レビュー NO-GO → 別タスク | **Phase-I 実装・監査・stress 検証済み**（G-4.1→G-4.3 チェーン + ST-1）。旧 P3 設計（lastRecoveryHandle_）残存 0 件 | **STALE（dash2 記述）** | なし。Phase-II Supersession は D18 凍結中 |
| **1.5 PublishReceiptWaiter sparse completion** | 🟢 将来保留（FIFO invariant 維持） | 未導入（sparse 0 件）+ H-0 事前監査（2026-08-19）で **NO-GO 判定済み**（現行 O(1) watermark で十分） | **DEFER（文書化済み NO-GO）** | sparse 化要件化時のみ再検討 |
| **1.6 X2 wraparound テスト** | 現状維持（1.5 と同時に） | INV-X2-6 architectural test 維持・SequenceArithmetic.h に sparse 拡張コメント | **DEFER** | 1.5 と同時（トリガなし） |
| **1.7 currentWorld_ 廃止（X4-B 案2）** | 高リスク・将来タスク（dual-pointer 暫定許容） | **実装済み** — `currentWorld_` は削除済み（CW-3c）・残存は歴史コメントのみ。RuntimeStore::current 単一 source（INV-ISR-06） | **CLOSED（STALE）** | なし |
| **1.8 BuildError wiring** | 🔴 現案 NO-GO → 1.8.5.2 分離が解決策 | **実装済み** — `BuildErrorPolicy.h` に `FailureClassification` / `RetryDisposition` 分離実装・RebuildDispatch.cpp:1244（D101-24 Step 3 caller-side policy）・RetryScheduler は 4-field のみ | **CLOSED** | なし |
| **1.9 初回 publish 前 quarantine 無駄な起床** | 🟡 条件付き GO — Phase 5 最適化候補 | **実装済み** — E-1.9-B event-driven wake（ISRCoordinatorLoop.cpp「Event-driven wake with 1ms fallback timeout」・Non-RT 明記） | **CLOSED** | なし |
| **2.1 R4 retire 順序逆転完全解消** | 🟡 条件付き GO — epoch safety と FIFO を分離 | **分離済み・意図的完了** — INV-EPOCH-1/2 が UAF を保証・INV-FIFO-1 は secondary optimization・Retire.cpp:88「FIFO 強化は実装しない」 | **CLOSED（設計どおり）** | なし |
| **2.2 shutdown lifetime contract（A2）** | 🟢 強く GO（A2-G01〜G23 PASS まで production reclaim 接続は NO-GO） | **D109→D110→D111 で解決済み** — D109 が D108 の「GAP」を訂正（tryShutdownQuiescentReclaim production caller 3 件実在）・D110 が GO 認可・D111 が Debug/Release 40/40 実証で **IMPLEMENTATION CLOSED / ACCEPTED**。ReclaimPermit（move-only / single-use / identity-bound ISRLifetimeProof.h:126）実装済み | **CLOSED** | なし |
| CacheMap destructor reclaim | A2 の一部 | tryShutdownQuiescentReclaim caller 3 件の 1 つ（AudioEngine.h:2106）として運用中 | **CLOSED** | なし |

## 2. 分類サマリ

| Classification | 件数 | 項目 |
|---|---|---|
| **CLOSED**（現行 baseline で実装・検証済み） | 6 | 1.3 LinearRamp、1.4 isFullyDrained、1.7 currentWorld_、1.8 BuildError、1.9 wake 最適化、2.1 R4 分離、2.2 A2（CacheMap 含む） |
| **DEFER**（trigger 待ち・意図的保留） | 3 | 1.1 R1 MPSC（trigger 待ち）、1.5 sparse completion（NO-GO 判定済み）、1.6 wraparound テスト（1.5 と同時） |
| **STALE**（dash2 記述のみが古い） | 1 | 1.2 coalesce（Phase-I 完了済み） |
| **OPEN**（着手根拠あり） | **0** | — |

## 3. 重点確認項目の詳細（指示 1〜3）

### 3.1 LinearRamp / ConvolverProcessor（D156-A）

- **現行 topology**: `reset()` / `setCurrentAndTargetValue()` → prepareToPlay 等 NonRT（Lifecycle.cpp:370-388、規約どおり）。RT 側の smoothingTime 変更は generation handshake（`mixSmootherResetPendingGen` fetchAdd acq_rel ← Lifecycle / acquire + `resetRT` ← Runtime.cpp:360）で RT への reset 侵入を排除（§1.3-B 解消設計）。
- `resetRT()` は `ASSERT_AUDIO_THREAD()` 付き RT-safe 純演算（DspNumericPolicy.h:341-347）。
- `getNextValue()` / `setTargetValue()` は Audio Thread 専用（クラス規約 DspNumericPolicy.h:323-327）。
- **判定: D156-A = CLOSED**。旧記述の「mixSmoother.reset() RT violation」は dash2 本文内の 8/14 再検証で既に自己訂正済みであり、baseline でも解消状態を確認。

### 3.2 isFullyDrained / shutdown drain

- `AudioEngine::isFullyDrained()`（Threading.cpp:118-175）は 9 条件の実測直接判定: `!hasDeferredCommit && pendingReclaimHandles_.empty() && retireDepth==0 && lifetimeRetireIntentPending==0 && ringResident==0 && dspQuarantineResident==0 && retireQuarantineResident==0 && terminalReclaimResident==0 && runtimePublicationBridge_.isFullyDrained()`。
- Coordinator 側 isFullyDrained は `recoveryIntentQueue_.size()==0 && pendingIntentCount_==0` を含む（cpp:525/532）。
- `waitForDrain` は ShutdownPhase 前提 jassert（AudioStopped 以降）+ bounded timeout。recoveryAdmissionPending は shutdown close（RebuildDispatch:814）で false 化。
- 旧設計書の「未完了扱い」のうち、絶対値上書き廃止・queue emptiness 追加・pendingReclaimHandles identity authority（INV-X3-5 / 二十六次レビュー必須修正2）は**すべて実装済み**。**OPEN 残存なし。**

### 3.3 A2 / Permit / Proof 系

- 「NO-GO 記録」（dash2 の A2-G PASS まで接続 NO-GO）は **D108 時点の判断**であり、その後 **D109（D108 の GAP 判定を訂正 — production caller 3 件実在）→ D110（Case A GO 認可）→ D111（Debug/Release 40/40 実証・IMPLEMENTATION CLOSED / ACCEPTED）** で完結。
- `ReclaimPermit` は move-only / single-use / identity-bound（ISRLifetimeProof.h:126-130、INV-LIFE-4）で実装済み。`pendingReclaimHandles_` は `ReclaimIdentity`（handle + retireSequence）の identity authority（Retire.cpp:92）。
- 責務分離: `isFullyDrained()` は観測的 predicate（G03 PASS）、Proof/Permit は ShutdownRuntime 専権（INV-LIFE-4）— 分離は成立済み。
- **判定: 実装不足ではなく「旧文書が残っている」。CLOSED。**

## 4. 指示の制約と成果物の確認

- 本監査でコードは 1 行も変更していない（production / test / CMake すべて 0）。
- 成果物: 本文書（`evidence/D156_RESIDUAL_OPEN_ITEM_RECONCILIATION.md`）+ work88 報告。
- 「設計書に未実装 ≠ 現行ソースで未実装」の原則で dash2 全項目を実コード照合した結果、**OPEN = 0 件**。

## 5. 次工程の提言

OPEN = 0 件のため、指示の判定ルールに従い次は実装ではなく:

> **「Project Open Items = 0 確認 → 全体統合 Close Audit」**

へ進行する。全体統合 Close Audit では、T3c Close Audit（evidence/T3C_CLOSE_AUDIT.md）と本 D156 の分類表を統合し、REPAIR_PLAN2-dash2 由来の全項目の最終状態（CLOSED / DEFER / STALE）を 1 枚の表に確定させる。
