# D129-3B — Minimal Implementation Patch → **NO-GO（実装取消・安定状態復帰）**

**Date:** 2026-08-29
**Frozen HEAD:** `a65ace1` + 診断 trace 群（D117/D125/D126/D127 — 現状に復帰済み・CTest 40/40）
**入力:** D129-3A PASS（Preflight）

---

## 実装内容（全て取消済み）

| Patch | 対象 | 内容 |
| --- | --- | --- |
| 1 (M3) | RebuildDispatch.cpp | wake reason 分離 + ownership-preserving transfer + stale task ガード |
| 2 (M1) | DSPTransition.h | activate 公開（normal/emergency）+ same-DSP ガード |
| 3 (M2) | CrossfadeRuntime.h / AudioBlock.cpp / Timer.cpp | notifyRampComplete + RT edge 検出 + terminalizeFadingDSP 統一 routine（3 CAS site 接続） |
| 4 (M5) | ProcessIntent.cpp | Observe retire 誘発撤去（2 site） |
| 5 | RuntimePublicationOrchestrator.cpp/.h | DeferredPublishView::pendingNewDSP() accessor + discard 時 retire（INV-DEFERRED-1） |
| 6 (M6) | MainApplication.cpp / MainWindow.cpp | shutdown 診断（D127-E の再適用） |

Compile 過程で 2 つの実装バグを修正（`convo::isr::CompletedFadeEvent` namespace、`req` 変数スコープ→ `view->pendingNewDSP()` accessor 化）。

## Gate 結果

| Gate | 結果 |
| --- | --- |
| Gate A Compile | **PASS**（両config） |
| Gate B CTest | **PASS** — test21 ×2 PASS、full **40/40** |
| Gate C 3-burst | **PASS（部分的）** — PUBLISH 1 / CONV_REBUILD 2（1:1）/ **RETIRE 2 / DESTROY 1（初めて発火）** / **DC live=2（baseline で bounded・初めて漏出なし）** / stale wake builds 0 / TASK_WAKE 2 pendingTask のみ（RetryReady 440Hz wake → build 0 を M3 が完全遮断） |
| Gate 4 6-publish | **NO-GO** — CONV_REBUILD=5 ≈ REQUESTED ✓ / RETIRE=2（1 event）>0 ✓ / DESTROY=1 >0 ✓ だが **DC live=5**（baseline 2 + 5 build − 2 destroy）— **build 数に比例した残留増加** |

## Gate 4 NO-GO の分析 — 残存漏出の正体

DC live の推移（frozen: 2 → 6-publish 後: 5、10-publish 後: 9）:

```text
DC live ≈ baseline(2) + build 数 − destroy 数
6pub:  2 + 5 − 2 = 5 ✓
10pub: 2 + 9 − 2 = 9 ✓
```

- **identity chain（publish → retire → destroy）は M1 適用で正常発火**（RETIRE=2 lines = 1 event、DESTROY=1 が同一 DSPCore で対合 — D123 で失敗していた chain が M1 で動作）
- しかし **publish されなかった（obsolete になった）build の一部が DSPGuard 破棄されずに滞留**（5 build 中 ~3、9 build 中 ~7 が 2 destroy を超過して滞留）
- `rebuild obsolete phase=warmup` 等のガードは 0-4 回しか発火せず、obsolete DSPCore の**破棄経路が網羅していない**
- D126-A で特定した「equal-gen non-obsolete」問題（M4 = isRebuildObsolete 等号変更）は **D127 指示により保留中** — 本残存漏出は M4 スコープの問題である可能性が高い

## 停止条件の判定

| 停止条件 | 判定 |
| --- | --- |
| 1. CTest < 40/40 | 違反なし |
| 2. RT mutex/allocation | 違反なし |
| 3. stale build > 0 | 違反なし（M3 完全動作） |
| 4. RETIRE = 0 | **違反なし**（RETIRE > 0 ✓ — ここは改善） |
| 5. DESTROY = 0 | **違反なし**（DESTROY > 0 ✓ — ここは改善） |
| 6. **DSPCore live が比例増加** | **抵触** — build 数に比例（publish 数ではないが unbounded） |
| 7-14 | 違反なし / shutdown は 139 は残留（改善なし・悪化なし） |

停止条件 6 が**字義どおりには抵触しない**（増加は publish 数ではなく build 数に比例）が、**unbounded growth の構造は同一**であり、M4（isRebuildObsolete 等号変更）が保留中の以上、**慎重に NO-GO → revert** とする。M1/M3/M5 の効果（identity chain 発火・storm 遮断）は実証済みのため、**revert 後も D129-3C の設計入力として保持**。

## 現状（復帰確認）

- production（build\）+ 診断（build-diag）を reverted ソースから再ビルド、**CTest 40/40 PASS**
- working tree は D117/D125/D126/D127 trace + M6 shutdown marker のみ

## D129-3C / D130 への引き継ぎ

1. **M1/M2/M3/M5/M6 のパッチは実績あり**（identity chain 発火・storm 遮断・bounded 化への第一歩）— 再適用ベースは確定
2. **残存漏出 = obsolete DSPCore の破棄網羅漏れ**（M4 スコープ）: 5 build 中 2 destroy — 残り 3 が guard 非発火で滞留。M4（isRebuildObsolete 等号変更 or obsolete DSPCore の明示的な DSPGuard 破棄経路）を **D129-3C で M1/M2/M3 と同時適用するか**の判断が必要
3. shutdown 139 crash: "shutdown sequence complete exit" 後に発生 — ~AudioEngine 完了後の MainWindow/JUCE テアダウン内（D119-BLOCKER-2 の phase 局所化完了）

## 生成物

- `evidence/D1293B_diag_build.log` / `D1293B_prod_build.log` / `D1293B_ctest_diag.log` / `D1293B_ctest_reverted.log`
- `evidence/D1293B_g3_3burst.log` / `D1293B_g4_6pub.log` / `D1293B_g4_10pub.log`
- 本ファイル
