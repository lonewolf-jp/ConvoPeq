# D162-2-I1-R0 Work Report — Deferred-slot Terminal Disposition / Shutdown Ordering Re-audit

- Work item: D162-2-I1-R0（B-1 STOP の修復設計確定。**read-only / production 変更 0** / B-2〜B-6 実施禁止）
- Date: 2026-09-04
- Baseline: ConvoPeq.md `Generated: 2026-09-04 20:28:50`
- 判定: **GO（Candidate A′）** — 詳細: evidence/D162-2-I1-R0_DEFERRED_DISPOSITION_REAUDIT.md

## 0. Executive Summary

B-1 residual=1 の原因を source 構造として確定した: **deferred slot の terminal disposition が
3 経路すべて条件付き trigger に依存し（EmergencyDrain request / Timer midrun / C1 fallback）、
shutdown pipeline に無条件の最終処分が存在しない**。短時間 profile（15s・健康 trigger なし）で
slot 保持 DSP（149MB）が無処分で shutdown に到達した。修復は
**Candidate A′ = releaseResources VerifyDrained の world clear 後に
`clearDeferredForShutdown()` を無条件呼出する 1 行追加**と決定。Candidate C（dtor）は
member 破壊順により UAF で明確に却下。

## 1. R0-3: 「published + deferred 残留」の意味論（重要訂正を含む）

- gen7 DSP（DE2D0080）は **publish 待ち（DeferredFadingActive retention）個体**として
  deferred slot に残留した。V-D の active-final が gen6 であった = shutdown 時の active world
  に gen7 は Admission されていない。**「published DSP の deferred 二重残留」という
  ownership 不整合は成立していない。** B-1 leak は disposition の呼出欠落が原因。
- churn 69 回の駆動源: `[XFADE] start expected=0.010s` が completed に到達せず
  fading が残存 → `hasFadingRuntimeInWorld()` が true のまま Admission が常に defer を返した
  （D135-8 F3 re-defer churn・TTL bounded）。**crossfade 非完了の root cause は DEFER 登録**
  （leak 修復と独立）。
- 留保: `[PUBLISH] seq=7 gen=7` タグ（:848）は publication-log bookkeeping の採番ずれであり
  DSP 個体の publish 証拠には採用しない（gen7 BUILD_PHASE より前に出現・gen6 publish と
  整合）。修復実装時に 1 回確認する。

## 2. R0-1: shutdown ordering（確定点）

- `stopRebuildThread()`（releaseResources :202・join 済み）は **EmergencyDrain(:349)/
  VerifyDrained(:441) より前** → これらの phase では RebuildThread が停止しており
  deferred slot の MessageThread 処分が単一 writer 契約上安全（EmergencyDrain での既存
  呼出 + C1 fallback が契約の先例・D135-9 Gate F VALID）。
- EBR digest は dtor body D5/D8 drain（G4 実績）で消化され E-3 assert が保証。
- member 破壊順（宣言の逆順）: `dspHandleRuntime_`(h:5034) → `shutdownRuntime_`(h:5000) →
  `m_retireRouter`(h:4813) → **`runtimeOrchestrator_`(h:3671)** → … — dtor 側処分は
  resolve/map/router が全て破壊済みで **UAF**。

## 3. R0-2: 候補比較

| 候補 | 判定 | 理由 |
| --- | --- | --- |
| **A′: VerifyDrained world clear 後に無条件 `clearDeferredForShutdown()`** | **採用** | 既存 S3 block（G1）をそのまま実行・冪等（hasDeferred_ check）・INV-D162-1〜9 全適合・dtor UAF なし・E-1（CacheMap 事前 drain）と同型の設計判断に整合 |
| B: EmergencyDrain 常時化 | 却下 | PolicyEngine 契約変更 + tryReclaim/crossfade recovery 強制という別副作用を normal shutdown に巻き込む（単なる if 外し禁止どおり） |
| C: dtor 側 | **明確に却下** | Orchestrator(3671) より先に m_retireRouter(4813)/dspHandleRuntime_(5034) が破壊される → resolve/erase/enqueue が UAF。E-1 と同型の設計判断（teardown 中 authority 触碰禁止） |
| D: 別の既存経路 | 該当なし | pendingTask retire は pendingTask のみ・drainDeferredRetireQueues は EBR queue の drain（slot 処分ではない）— 既存に最終処分 site がないこと自体が B-1 の原因 |
| E: 解釈誤り | 否定 | E-4 会計不成立 + `[ISR][Shutdown]` deferred=1 観測 + gen7 destroy 0 件 + 3 trigger 不成立が source 構造と突合済み |

## 4. R0-4: 最小修復単位（実装仕様）

```cpp
// releaseResources / VerifyDrained phase・world clear + V-D retire block の直後:
if (runtimeOrchestrator_)
    runtimeOrchestrator_->clearDeferredForShutdown();   // ★ D162-2-I1 修復
```

- Thread: MessageThread（post-join）・Authority: retireRegisteredDSP（G1 S3 同一経路）→ EBR。
- 二重防止: hasDeferred_ check（EmergencyDrain で先に clear されていれば no-op）+ map erase guard。
- INV-D162-1〜9: 全適合（§4 の表参照）。
- **副次効果**: S3 standalone `retired=1`（G1-G4 未観測事項）が B profile で実測可能になる。

## 5. GO 条件判定（9/9 満たす）

B-1 原因の構造再確認 ✓ / 責任箇所一意 ✓ / Thread ownership 明確 ✓ / retireRegisteredDSP 使用可 ✓ /
EBR 順序保証 ✓ / V-D 二重処分防止 ✓ / published/deferred 意味論説明 ✓ / dtor UAF なし ✓ /
測定契約変更不要 ✓ → **R0 = GO**。

## 6. DEFER / 残置

1. crossfade 非完了 root cause（tryCompleteFade 条件・B-1 churn 69 回の起点）— leak 修復と独立。
2. `[PUBLISH] seq/gen` タグ意味論（bookkeeping 採番ずれ）— 実装時に 1 回確認。
3. V-D fading-final dangling resolve（B-1 実測・V-D-b が防止済み）— residual register 記録済み。

## 7. 次工程

**修復実装（releaseResources 1 行 + 注記）→ Build/CTest → B-1 再試験（residual 0・
S3 retired=1 実測の確認）→ B-2〜B-6 → Profile C → D → I1 final。**
実装はユーザーの go 待ち。
