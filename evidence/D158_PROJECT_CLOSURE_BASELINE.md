# D158 — Project Closure Baseline / Deferred-Item Freeze Audit（read-only）

**Date:** 2026-09-01 (+09:00)
**Type:** 完全 read-only 最終ガバナンス監査。**Production source: 0 / Test source: 0 / CMake: 0 / build: 0 / CTest: 0 / 新規実装: 0 / 新規 stress: 0 / 既存テスト再実行: 0。**
**Baseline（必須参照・本監査の第 1 基準）:** `ConvoPeq.md` `Generated: 2026-08-31 23:39:12` — **本会話で提示済みの D157 使用 baseline**。再実測: mtime 23:39:16 と stamp 一致・**src より新しいファイル 0 件（NEWER_SRC_COUNT=0）**。リポジトリ内の別スナップショット（`doc/ConvoPeq_元データ_20260602195821.md` 等の旧版・File Library の 07:12:57 / 21:03:41）は**最新扱いしない**。
**性格:** 「問題を探す監査」ではなく、**閉鎖状態を壊さずに将来項目を凍結する最終ガバナンス監査**。新規の安全性証明は作らない。
**参照:** D157 / D156 / D155 / T3c Close Audit / D152-R2 / D154-R2 / D154-F1 / D154-F2 / ST-1 / RT Affinity / REPAIR_PLAN2-dash2（historical/design reference・現行ソースより下位）。

---

## 0. 総合判定（先出し）

```text
OPEN     = 0
BLOCKED  = 0
CLOSED   = 15  （fixed — 再作業禁止境界 §3 で固定）
DEFER    = 6   （trigger-based — §1 で各 trigger を再確認・凍結）
STALE    = 3   （historical only — current work item に数えない）
```

> ## **D158 PASS — Project Closure Baseline 成立。** OPEN = 0 && BLOCKED = 0 であり、DEFER は trigger 待ちとして正式凍結、STALE は historical のみ。**次は Project Closure Record → 通常開発へ移行。**

---

## 1. A: DEFER 6 件の個別再確認（非 OPEN 理由 + 再開 trigger、1 行ずつ）

各項目について baseline で trigger 特性を実測再確認した（本日実測値を併記）:

| # | Item | 現在 OPEN ではない理由（1 行） | 将来再開する trigger（1 行） | 本日実測 |
|---|---|---|---|---|
| D1 | R1 MPSC（recoveryIntentQueue_） | producer = CoordinatorLoop 単一・consumer = RebuildThread 単一で SPSC invariant が構造的に成立しているため（D155 実測） | recovery transport への**第 2 producer 出現**（Timer 等が submit/pop を直接呼ぶ） | Timer/Processor の recovery API 呼び出し = **0 件** |
| D2 | Phase-II Supersession | D9 blocking gate は G-4.1 の equality-conservative 固定（6-field target）で解決済み・D18 により意図的凍結・equality-only containment は ST-1 で stress 実証済み | **Supersession（A→B 上書き）が製品要件化**した時点で Phase-II 設計監査（D103 readiness + D18 規範を出発点） | `ResolvedSuperseded` 遷移 = **0 件**（凍結維持） |
| D3 | D102-C3/C4 remaining gates | Phase-II 実装時の前提条件 inventory であり、現行ソースに C3/C4 ラベルの実装対象は存在しない | **Phase-II Supersession 実装開始**（D2 と同時） | baseline/src に C3/C4 ラベル = **0 件** |
| D4 | PublishReceiptWaiter sparse completion（1.5） | 単一 completion writer（INV-X2-5）+ FIFO（INV-X2-6）の O(1) watermark で十分・H-0 事前監査（2026-08-19）が **NO-GO 判定済み** | **MPSC completion / parallel publish を許容する設計変更**が発生した時点 | `completedOutOfOrder` 未導入（defer anchor コメント確認） |
| D5 | X2 wraparound / out-of-order テスト（1.6） | INV-X2-6 contiguous completion が現行アーキテクチャ不変条件として維持されており、テスト対象の状態が存在しない | **sparse completion（D4）実装と同時**（isBefore 等の modular arithmetic 定義後にテスト追加） | `INV-X2-6` アンカー + SequenceArithmetic.h「現状は実装不要・将来保留」確認 |
| D6 | P2/G2/W1 static bound | D102 は O_denom 実測（C2-2/C2-3 campaign）で代替済みのため非 blocking・measure vs constrain の選択（D40.5）未要求 | **D40 追補で static bound を要求**（「measure or constrain」の constrain 選択）時 | `N_retired_world` / `G_max/T_min` の実装 = **0 件** |

**判定: DEFER 6 件はすべて trigger-based として妥当 — 変更なし。** いずれも「未完了の作業項目」ではなく「trigger 待ちの意図的凍結」であり、Project 未完了とは数えない（判定ルール適用）。

## 2. B: STALE 3 件の OPEN 除外（historical → correction → current 再確認）

| # | Item | historical status | correction / subsequent implementation | current status | current action |
|---|---|---|---|---|---|
| S1 | dash2「1.2 Recovery coalesce 将来対応（別タスク）」 | Future・四次レビュー NO-GO（2026-08 前半） | 旧 P3 設計（lastRecoveryHandle_）は廃止 → **G-4.1→G-4.3 で Phase-I coalesce 実装・監査（12/12 + CTest 40/40）・ST-1 T5 storm 実証** | **STALE** — dash2 記述は現行状態を反映していない | **none** |
| S2 | dash2「1.7 currentWorld_ 廃止 = 将来タスク（高リスク）」 | 将来タスク（dual-pointer 暫定許容） | **CW-3c で currentWorld_ 削除済み**（RuntimeStore::current 単一 source・INV-ISR-06）。残存は歴史コメントのみ | **STALE** | **none** |
| S3 | D108「A2 production reclaim wiring = GAP / NO-GO」 | D108 判定（2026-08 下旬） | **D109 が D108 の GAP 判定を訂正**（tryShutdownQuiescentReclaim production caller 3 件実在）→ **D110 GO 認可** → **D111 Debug/Release 40/40 実証で IMPLEMENTATION CLOSED / ACCEPTED** | **STALE** — D108 記述は supersede 済み | **none** |

dash2 のその他の過去時点記述（A2 `reclaim()` bool 問題等）も同様に、後続監査（D102-C2-5-D3/D4 Bool-Wrapper/Sink-Caller Closure で閉鎖済み）により OPEN から除外。**STALE 3 件 + dash2 由来の全過去記述は current work item に数えない（判定ルール適用）。**

## 3. C: CLOSED 項目の「再作業禁止境界」の確認

以下は baseline で既存 close の実在確認（アンカー実測）を行い、**今後不用意に再設計しない領域**として固定する。新規の安全性証明は本監査では作っていない（境界確認のみ）:

| 固定領域 | close 根拠 | baseline アンカー実測 |
|---|---|---|
| T3c lifecycle CAS（full-word CAS 9 サイト） | T3c Close Audit（ST-1 + RT affinity PASS） | `compare_exchange_strong` lifecycle CAS 4 + durable 4 + test = 実測 9 |
| `RecoveryLifecycleWord`（16B 構造） | T3c Close Audit + D152-R2 規範 | 宣言 1 件（h:386）・static_assert 5 件 |
| 16B full-word CAS semantics | D152-R2（lock-pool 下でも原子的相互排他 + フルバリア） | lock-pool 記述 11 件（ConvoPeq.md） |
| DSPHandle atomic backend | D154-F2（comment-only 訂正済み） | numstat 6/6・11/11・実コード 0 変更 |
| RT affinity boundary | RT Affinity Audit PASS | RT パス接触 0 件（T3c Close Audit 実測） |
| A2 ReclaimPermit / Proof | D109→D110→D111 ACCEPTED | `ReclaimPermit` 記述 11 件（ISRLifetimeProof.h） |
| isFullyDrained semantics | D107 16 条件 + D108 G03 PASS | 実装アンカー 5 件（Threading.cpp） |
| Phase-I coalesce | G-4.3-R/T-R PASS + ST-1 | coalesce identity CAS（cpp:942）実在 |

**変質例外（Trigger Matrix §4 の closed exception 2 行）に該当しない限り、上記領域への再設計・再監査・stress 再実行は行わない。**

## 4. D: Trigger Matrix（D158 の重要成果物）

| Trigger | 再開する工程 | 現在 |
|---|---|---|
| recovery transport の第 2 producer 出現（Timer 等が直接 submit/pop） | R1 MPSC 実装前設計（D155 §2.5 の変更対象 5 項目を含む） | **DEFER** |
| Supersession が製品要件化 | Phase-II 設計監査（D103 readiness + D18 規範・D9 固定内容を出発点） | **DEFER** |
| sparse completion が必要になる（MPSC completion / parallel publish 許容） | 1.5 sparse 実装 + 1.6 X2 wraparound/out-of-order/duplicate/wraparound テスト（modular isBefore 定義後） | **DEFER** |
| D40 が static bound を要求（constrain 選択） | P2/G2/W1 静的 bound 導出（T_min/G_max のコード保証または制約追加） | **DEFER** |
| RT callback → lifecycle W 接触（RT パスからの recovery API 呼び出し発生） | **T3c close invalidation → D152 再評価**（D152 §2 wrapper 案＝真の lock-free への backend 切替必須） | closed exception |
| `retireCoordinator_` の RT dereference（EQ/Convolver が RT callback 内から coordinator を呼ぶ） | **T3c close invalidation → 再監査** | closed exception |

補助 trigger（記録・監視のみ）: RecoveryEpisodeId 必須化（D13）は Phase-II 内で処理。`pendingRecoveryAdmission_` の MPSC 化は R1 と同時判断（D155 §2.3 — 単独では不要）。

## 5. 判定ルールの適用

```text
OPEN >= 1                → Project Close 不可        … 不適用（OPEN = 0）
OPEN = 0 && BLOCKED = 0  → Project Closure Baseline  … **適用**
DEFER > 0                → Project 未完了とは扱わない … 適用（6 件すべて trigger-based と再確認）
STALE > 0                → current work item に数えない… 適用（3 件すべて historical only と再確認）
```

**D157 の数値をそのままコピーしたものではなく、DEFER 6 件は本日実測で trigger 非発生を確認・STALE 3 件は correction チェーンで OPEN 除外を再確認済み。**

## 6. 監査手順の記録

- baseline 再照合: Generated 23:39:12・NEWER_SRC_COUNT=0・旧版（元データ 20260602 / File Library 07:12:57・21:03:41）を最新扱いしない旨を明記
- DEFER trigger 実測: Timer/Processor の recovery API 0 件・ResolvedSuperseded 遷移 0 件・C3/C4 ラベル 0 件・completedOutOfOrder 未導入（defer anchor コメント確認）・N_retired_world 0 件
- STALE: correction チェーン（G-4.x / CW-3c / D109-D111）で OPEN 除外を 3 件とも再確認
- CLOSED 境界: 8 領域の baseline アンカー実測（CAS 9 サイト・ReclaimPermit 11 件・isFullyDrained 5 件等）
- コード変更: 0（production / test / CMake すべて）

---

> ## **D158 PASS — Project Closure Baseline 成立。次は Project Closure Record を作成し、通常開発へ移行。**
