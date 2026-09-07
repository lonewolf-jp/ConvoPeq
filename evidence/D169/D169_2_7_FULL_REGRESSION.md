# D169-2-7 — Full Regression（Work Record）

```text
D169-2-7 — Full Regression（D169-2 close gate）
Date:      2026-09-07
Contract:  evidence/D169/D169_2_2_REPAIR_CONTRACT.md（RC-D169-2-1〜7）
Prior:     D169-2-4 IMPLEMENTED / D169-2-5 PASS / D169-2-6 PASS
Verdict:   **PASS → D169-2 CLOSED**
```

---

## P1 — Source / diff boundary audit **PASS**

### Production boundary

`git diff -- src`（base = 最終 commit・working tree に D167/D170/D169 の未 commit 分を含む）
を全走査し、各変更を track に帰属させた:

| File | diff | 帰属 | D169 該当 |
| --- | --- | --- | --- |
| `AudioEngine.Processing.PrepareToPlay.cpp` | +17/-0 | **D169-2-4 のみ**（marker `D169-2-4` 1 件・他 track marker 0） | ✓ 許可された唯一の production change |
| `AudioEngine.Processing.ReleaseResources.cpp` | +83/-14 | D170（D169-1R marker 4）+ D167 分 | D170 track（別 track・D169-2 とは無関係） |
| `AudioEngine.h` / `RebuildDispatch.cpp` / `RetrySchedulerTypes.h` / `MainApplication.cpp` | D167 報告 §2 と行数一致 | D167 | 非対象 |
| **`ISRLifecycle.cpp` / `ISRLifecycle.h`** | **0 行** | — | ✓ 無変更（FSM/rollback/placeholder/world/publish/rebuild/shutdown FSM 全て無変更） |

### Test infrastructure boundary

`PublishPipelineIntegrationTests.cpp` の diff を解析:

- 新規 test 関数の追加 = **D167 ×2（既存未 commit 分）+ `testD169DuplicatePrepareCollapseNoop`
  + `testD169DeviceRestartCollapseStress`（documented deviations ×2）のみ**。
- 削除 4 行は T-I2-1 の FAIL message 言い回し変更（D167 時点の既存未 commit 分・
  D167 報告 §3.3 記録の T-I2-1 更新に含まれる）— **D169 による既存テスト改変 0**。
- `AudioEngineHarness.{h,cpp}`: `startAudioOnly(int)` seam のみ（D169-2-6 deviation）+
  D167/D162-2-I2 既存分。
- **unexpected diff: 0 → STOP 条件不発。**

## P2 — Binary freshness gate **PASS**

D169-2-6 で発見した LNK1285 取りこぼし問題に対し、build log の `[OK]` 表示に依存せず
failure pattern 直接走査 + mtime 照合を実施。3 config を全 rebuild:

| Config | `[OK]` | FAILED/fatal/ninja-stopped | exe mtime (UTC) | harness test obj | PrepareToPlay obj | freshness |
| --- | --- | --- | --- | --- | --- | --- |
| Debug | ✓ | **0** | 12:04:55 | src より新 | src より新 | exe > objs ✓ |
| Release | ✓ | **0** | 12:08:12 | src より新 | src より新 | exe > objs ✓ |
| RelWithDebInfo | ✓ | **0** | 12:11:49 | src より新 | src より新 | exe > objs ✓ |

3 config 全て: build 成功 + failure pattern 0 + obj が source より新 + **exe が両 obj より新**
（stale binary 0 — D169-2-6 の教訓 gate を通過）。

## P3 — Targeted D169 regression（consistency 再実行）**PASS**

fresh binary で harness suite ×3 config を再実行:

| Config | D169-2-5（collapse ×4 no-op） | D169-2-6（restart stress 50 cycles） | FAIL 行 | 全テスト |
| --- | --- | --- | --- | --- |
| RelWithDebInfo | PASS | **PASS（50 cycles OK）** | 0 | all PASS |
| Debug | PASS | **PASS（50 cycles OK）** | 0 | all PASS |
| Release | PASS | **PASS（50 cycles OK）** | 0 | all PASS |

D169-2-5/2-6 の実証結果と一致（consistency 確認・テスト内部 assert により
collapse count 一致 / gen・seq・telemetry・slot 不変 / Prepared・admission Open /
ShutdownComplete を含む）。

## P4 — Full CTest **PASS**

| Config | 結果 | 時間 |
| --- | --- | --- |
| Debug | **100% tests passed（40/40）** | 49.85s |
| Release | **100% tests passed（40/40）** | 45.57s |
| RelWithDebInfo | **100% tests passed（40/40）** | 51.59s |

baseline（Debug 40/40・Release 40/40）を維持 + RWDI full test を追加達成。
既存系統（Uninitialized→Preparing→Prepared / Released→Preparing→Prepared /
Prepared+SR change→Preparing→Prepared / device restart / publication / rebuild /
shutdown）は D167 テスト・T-I2-1・D169 targeted tests が 40 suite 内で網羅し、
collapse が誤って通常 prepare に介入していないこと（negative path）も含めて PASS。

## P5 — Lifecycle regression **PASS**

- **Positive**: Prepared + same SR/BS → expectedPhase == Prepared → collapse →
  return → Prepared 維持（D169-2-5 ×4 観測 + D169-2-6 ×50 cycles ×3 config で実証）。
- **Negative**: Prepared + SR/BS 変更 → expectedPhase != Prepared → 通常 prepare →
  Preparing → Prepared（D169-2-5 SR 変更観測 + D167 テスト + CTest 40/40）。
- **Other paths**: Uninitialized → Preparing → Prepared（全 harness テストの start 経路）/
  Released → Preparing → Prepared（T-I2-1 phase3）— D169 branch は
  `expectedPhase == Prepared` のみで分岐するため他経路に侵入なし
  （D169-2-3 P1 一意性証明 + CTest 全 PASS で回帰なし）。

## P6 — Side-effect regression **PASS**

D169-2-3 P4 の 25 side effects に対する **25/25 NOT EXECUTED** を最終記録:

| 主要 side effect | collapse 時 |
| --- | --- |
| rebuildRequestGeneration | 不変（D169-2-5/2-6 stress 前後比較） |
| publication / RuntimeWorld | sequenceId 不変・進行なし |
| placeholder | slot pointer 不変 |
| latency resources / analyzerFifo / crossfade state / uiConvolverProcessor | body 非実行のため不発 |
| submitRebuildIntent | REBUILD_TELEMETRY 0 行 |
| leavePrepare | collapse 経路から到達不能（return :36 → leavePrepare :344） |

duplicate prepare が完全な no-op であることを Full Regression の主要 acceptance
criterion として成立させた。

## P7 — Device restart regression **PASS**

D169-2-6 の 50 cycles ×3 config（計 150 cycles）を baseline として参照し、
Full CTest（同じ binary）が同一変更下で PASS することを確認（P4）。
restart chain（audioDeviceStopped → releaseResources → about-to-start → same SR/BS
prepare → collapse → audio resume）の維持は D169-2-6 evidence による。

## P8 — Shutdown integrity **PASS**

- targeted tests 内で `h.stop()` → admission Closed + `ShutdownComplete` 到達を
  D169-2-5（collapse ×4 後）・D169-2-6（50 cycles 後）で確認済み。
- full suite の shutdown 系テスト（TeardownPublish 等）も 40/40 内で PASS。
- `pending collapse → shutdown` の未処理状態なし（collapse は no-op であり
  state を一切残さない — RC-2 の構造的性質）。

## P9 — Known hazard の扱い **PASS（STOP 条件不発）**

`timerCallback → MEM_SNAP → getActiveRuntimeDSP() → collectTrackedMemoryStatistics()`
の dangling 参照 risk は D169-2-7 では修正しない（指示どおり）。

- 分類: **D169 defect: NO / D169 regression: NO / pre-existing hazard: YES / separate track: YES**
- Full Regression 実行中の新規 AV/crash: **0 件**
  （CrashDumps の AudioEngineHarness dump は調査時の既存 1 件のみ・
  Full Regression 中の新規生成なし）
- **STOP 条件不発** → PASS 判定可能。

## P10 — 最終判定

```text
Production diff boundary       PASS
Test deviation boundary        PASS
Binary freshness               PASS
Debug CTest                    PASS (40/40)
Release CTest                  PASS (40/40)
RWDI full regression           PASS (40/40)
D169 targeted collapse         PASS (×4 ×3 config)
D169 device restart            PASS (×50 ×3 config)
same SR/BS collapse            PASS
SR/BS change normal prepare    PASS
Lifecycle integrity            PASS
Publication integrity          PASS
Rebuild integrity              PASS
Shutdown integrity             PASS
Abort                          0
AV                             0
Unexpected exception           0
Unexpected publication         0
Unexpected rebuild             0
────────────────────────────────
D169-2-7 = **PASS**
```

## Evidence

- freshness: evidence/D169/d169_2_7_build_{Debug,Release,RWDI}.log
- targeted: evidence/D169/d169_2_7_harness_{rwdi,debug,release}.log
- CTest: evidence/D169/d169_2_7_ctest_{debug,release,rwdi}.log

## Verdict

**D169-2-7 = PASS → D169-2 CLOSED。**
