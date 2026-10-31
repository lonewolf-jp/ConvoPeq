# STG-5 Concrete Defect Discovery — CLOSED（Concrete Defect なし）

```text
Authority HEAD : 901a6c336bbd9ec83db546028e749cc868f1a26a
Authority file : ConvoPeq.md 5,580,623 B
SHA-256        : 69963A9BF920EAEDDD73A85EF7D57BC36E55A29DB349ADFB6A42EF352BF6FB7B
Generated      : 2026-09-28 10:43:22
--check        : FRESH（NEWER_SRC_COUNT 0）
前回 authority  : 4FF62E07…（10:02:13）→ size 同一・Generated のみ更新（同一 sources）
Phase          : Discovery only（production 0 / test 0 / commit 禁止）
worktree       : 901a6c33 clean（pre-existing M AGENTS.md / headroom-proxy-start.ps1 のみ・不干渉）
```

## A. 非同期・世代管理・ownership（STG-4-1 以外）

| 対象 | 追跡結果 | 判定 |
|---|---|---|
| ProgressiveUpgradeThread | entry／convert 後／apply 前の isGenerationValid＋callAsync 内再照合＋applyComputedIR 照合の三重 guard。cancel＋stopUpgradeThread 併用 | guarded・clear |
| loaderTrashBin | cleanup() が終了済み除去＋size>2 cap。無界蓄積なし | clear |
| rebuildPendingAfterLoad | bool フラグの coalesce は設計。set（ring-overflow 経路）→consume（timer／commit）の lifecycle 完結 | clear |
| deferred Structural（convolverParamsChanged→Timer） | hash-dedup＋200ms window＋dueTicks 再駆動。二重 defer は later-wins、余剰時は Replaceable が吸収。欠落・逆転の経路なし | clear |
| updateConvolverState（RCU） | 生成時 gen 刻印＋適用時照合。STG-4-1 bump は旧要求の適用を正しく stale 化する方向 | clear |
| isCacheEntrySafeToDelete × bump | active engine 基準は不変。bump は superseded entry の削除許容を増やす安全方向 | clear |
| handleLoadError | engine 保持＋3 flag 整理＋通知。ownership 移動なし | clear |
| IncrementalRebuildJob | job.shouldCancel が currency 担保。STG-4-1 で現世代刻印（動作不変・全 suite green） | clear |

## B. State／metadata／runtime 三者不整合

```text
STG-4-1 discard：swap・currentIrFile・metadata・publish のいずれも不変（B 維持を動的確認済み）。
applyComputedIR（B）：irName・irLength・IRState・RCU を一括更新。save/load は同値を往復。
save→load→save 非冪等・mode/bypass 順序差・分散 clamp：STG-3 で棚卸し済み、新規なし。
```

## C. Failure／retry／partial completion

```text
LoaderThread FlagResetter／error dispatch／callAsync 失敗時 atomic 整合：読取り済み。
RetryScheduler／deferred admission：既存テスト PASS。D105 再開なし。
runSynchronously／incremental エラー経路：job.lastError＋stage 遷移で完結。
failure 由来の stale commit：executePendingCommit guard が phase 共通で捕捉。
concrete 経路なし。
```

## 判定

```text
STG-5 CLOSED — Concrete Defect なし
理由：STG-4-1 型の「guard のない completion 関口」は executePendingCommit 以外に
  存在しないことを swap 経路全数（commit／prepare re-init／share-clone／teardown）
  で確認。非同期・defer・retry の各 state machine は currency／ownership の
  照合・cap・coalesce を備え、deterministic な invariant violation を構成できない。
  残るのは設計内挙動（coalesce・defer・flag 精度）のみ。
```

## Phase gate

```text
Production changes: NONE
Test-only changes: NONE
commit: NOT AUTHORIZED
push: NOT AUTHORIZED
STG-1／STG-2／STG-4-1：CLOSED のまま変更なし
```
