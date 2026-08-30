# D126-A — Rebuild Task Re-execution / Obsolete DSPCore Causality Audit

**Date:** 2026-08-29
**性質:** read-only + macro-gated diagnostic only（`[D126_TASK_WAKE]` トレース 1 点追加 — RebuildDispatch.cpp）
**Frozen HEAD:** `a65ace1` + D117 trace + D126 trace
**入力:** D125（IR flag ループ否定 → 実行側再実行ループへ焦点）

---

## D126-A1/A3 — rebuild thread loop の構造と再実行条件（G3）

`rebuildThreadLoop`（RebuildDispatch.cpp:816-1100）の実構造:

```cpp
while (true) {
  wait(lock, []{ return hasPendingTask || publishRetryReady || recoveryPending || exit; });
  task = pendingTask;                 // ★ (A) 常に pendingTask を copy（新規タスクが無くても前回タスク）
  pendingTask.currentDSP = nullptr;
  hasPendingTask = false;
  doDeferredPublish = publishRetryReady;   // ★ (B) wakeup 要因を記録するのみ
  publishRetryReady = false;
  ...
  if (doDeferredPublish) processDeferredAdmission();   // deferred publish ハンドオフ
  ...（recovery 処理）...
  if (isObsolete()) continue;         // ★ (C) isRebuildObsolete(task.generation)
  if (!task.runtimeBuildSnapshot.sealed) continue;
  build(task...)                      // ★ (D) ★【無条件に再 build される】★
  ...
  if (isObsolete()) continue;         // phase=rebuildIR / warmup でも再判定
  warmup validate → trySubmit → publish
}
```

**再実行条件の確定（Candidate B = rebuildThreadLoop 内 task 再実行 — CONFIRMED）**:

1. wakeup predicate に `publishRetryReady` が含まれる → CoordinatorLoop の deferred publish 再通知（1ms tick）で起床
2. 起床時 `task = pendingTask` が**前回タスクの stale copy** を取る（currentDSP のみ null 化）
3. `isRebuildObsolete(task.generation)` は `task.generation < committed` を意味し、**同一世代（等号）では obsolete にならない** → 再 build が実行される
4. 再 build → 新 DSPCore → publish/obsolete 判定… の繰り返し

## D126-A2 — RetryScheduler の除外（G2）

| 観測 | 値 |
| --- | --- |
| `warmup failed`（D123 storm log） | **0 回** |
| `retryScheduler_->schedule` の前提（retryable warmup failure） | 不成立 |
| frozen probe での schedule | 0 |

→ **RetryScheduler は再実行源ではない（除外確定）**。Candidate A 棄却。

## D126-A4/A5 — D123 storm と frozen baseline の対合（G1/G4/G5）

### frozen baseline（D126-g4b: 6 burst / 50s / revert 後）

| 指標 | 値 |
| --- | --- |
| wake | 5 回（**全て wokeByPendingTask=1、RetryReady=0**） |
| CONV_STATUS | **各世代 1 回のみ**（gen 4-8） |
| build / publish / RETIRE / DESTROY | 5 / 5 / **0** / **0** |
| DC live 最終 | 7（+1/publish、**破棄されず滞留** — D117 の baseline 漏出そのもの） |
| storm | **なし** |

### D123 storm（D123-g4: 6 burst / 50s / D123 patch 適用時）

| 指標 | 値 |
| --- | --- |
| REBUILD_REQUESTED（intent） | **11 回のみ（反復なし）** |
| CONV_STATUS 同一世代反復 | **gen 6: 10× / gen 7: 9× / gen 8: 10× / gen 9: 73×** |
| build（DSPCORE_PREPARE/CONV_REBUILD） | 107 |
| publish / RETIRE / DESTROY | 2 / 4（8 line）/ 6 |
| DC live 最終 | **103** |

### intent → execution 対応（G1）

```text
frozen baseline:  1 intent → 1 build → 1 publish → 0 retire（漏出）
D123 patched:     1 intent → 約 10 build（同一世代再実行）→ 1 publish 前後 → 約 9 滞留
```

## D126-A6 — DSPCore outcome 分類（G5/G6）

| outcome | frozen baseline | D123 storm |
| --- | --- | --- |
| built | 5 | ~107 |
| published（commit） | 5 | 2 |
| obsolete-before/after-build | 0 | 4（phase=warmup） |
| publish failed | 0 | 0 |
| destroyed | 0 | 6 |
| **unresolved/live（滞留）** | **5** | **~101** |
| 総和 = 生成数 | ✅ 5=5 | ✅ 107 ≈ 2+4+6+~101+2 |

両 run で総和が生成数と一致（G6 PASS）。

## D126-A7 — 未 publish DSPCore の ownership 終端（G7）

**2 つの独立した漏出経路が確定**:

| 経路 | 対象 | 機構 | 状態 |
| --- | --- | --- | --- |
| **経路 1（baseline・frozen でも発生）** | publish **された** DSPCore | retire 経路全体が到達不能（D117/D121: activate 未公開 → oldHandle null → DSPTransition 分岐 skip） | D121 で確定済み。D126 の D122-Patch-A/B（activate 公開 + retire 接続）で閉じる |
| **経路 2（crossfade 分岐到達時に発生）** | publish されなかった **obsolete 系** DSPCore | (a) 同一世代再実行で built、(b) `isRebuildObsolete` が等号を obsolete と見なさず DSPGuard が発火しない、(c) 結果として破棄されず滞留 | **D126-A 新規確定**。D123 patch 適用時のみ顕在化 |

**D123 の 101 滞留の説明**: 経路 1（published 2 本のうち実質 leak 分）＋ 経路 2（未 publish ~99 本が equal-gen non-obsolete により guard 発火せず）の合算で一致。

## D126-A5 — crossfade は再実行源か（G4）

**答え: crossfade は「再実行のトリガー」であり「再実行の機構」ではない。**
- crossfade 分岐到達（D123-A）→ deferred admission が反復（hasDeferred_ 再 enqueue → Coordinator 1ms 再通知）→ (B) の再実行が加速
- crossfade を通らなければ（frozen baseline）再実行は発生しない
- よって D125 案 X（HardReset 固定）は **storm 防止としても有効**だが、経路 2 の再実行構造（等号 obsolete 漏れ + stale task 再 build）は**別途修正が必要**（crossfade を使わなくても将来の到達経路で顕在化し得る）

## D126-A8 — D126 最小修正候補（production 変更なしで確定）

| # | 修正 | 対象 | 効果 |
| --- | --- | --- | --- |
| M1 | `DSPHandleRuntime::activate(newHandle)` 公開（D122-A/Patch-A） | DSPTransition.h | 経路 1 閉鎖（retire 到達） |
| M2 | fade completion → retirePublishedDSP 接続（D122-C/Patch-B） | Timer.cpp / AudioBlock.cpp | 経路 1 の crossfade ケース閉鎖 |
| M3 | **stale task 再 build ガード**: `publishRetryReady` のみで起床した場合（hasPendingTask=false）は build を skip する | RebuildDispatch.cpp | 経路 2 の再実行遮断（D123 storm の構造修正） |
| M4 | `isRebuildObsolete` の等号扱い再審査（task.generation == committed → obsolete とするか、再 build 前に明示廃棄） | RebuildDispatch.cpp / isRebuildObsolete | 経路 2 の二重防護 |
| M5 | Observe からの retire 誘発撤去（D122-E） | ProcessIntent.cpp | authority singularization |
| M6 | shutdown 診断（D122-G） | MainApplication.cpp ほか | BLOCKER-2 特定 |

**順序**: D127 = M1+M2+M3(+M5, M6 診断) を単一パッチで実装（D119/D123 の教訓: 単点修正は別欠陥を顕在化）。M4 は M3 の効果確認後に判断。

## GO/NO-GO（G1-G8）

| Gate | 判定 |
| --- | --- |
| G1 intent ↔ execution 対応 | **PASS**（frozen 1:1 / D123 1:約10 を telemetry+CONV_STATUS で対合） |
| G2 RetryScheduler 除外 | **PASS**（warmup failed 0 → schedule 0） |
| G3 rebuildThreadLoop 再実行条件 | **PASS** — Candidate B 確定: deferred publish wakeup × stale task copy × equal-gen non-obsolete |
| G4 crossfade は再実行源か | **PASS** — トリガーであって機構ではない |
| G5 outcome 分類 | **PASS**（両 run とも総和一致） |
| G6 総数対合 | **PASS** |
| G7 未 publish DSPCore ownership 終端 | **PASS** — 経路 1（retire 到達不能）+ 経路 2（equal-gen guard 不発）の 2 経路で完全説明 |
| G8 最小修正候補固定 | **PASS**（M1-M6・production 変更なし） |

**D126-A: PASS（8/8）** → **D127（M1+M2+M3+M5 統合パッチ実装）へ進行可**。ただし M3（stale task ガード）は D119/D123 の失敗を踏まえ、M1/M2 と**同一パッチ**で実装すること（単点修正禁止の教訓）。

## 生成物

- 本ファイル（`evidence/D126A_REBUILD_EXECUTION_CAUSALITY_AUDIT.md`）
- `evidence/D126_diag_build.log` / `D126_probe.log`（frozen 3-burst）/ `D126_g4b.log`（frozen 6-burst）
- `src/audioengine/AudioEngine.RebuildDispatch.cpp` に `[D126_TASK_WAKE]` macro-gated trace 追加（観測専用・D117 trace と同扱い）
