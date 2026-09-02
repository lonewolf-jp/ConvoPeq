# CR-α-4 — CTest 40/40 × Debug/Release + Regression（Work Report）

```text
CR-α-4 — CTest 40/40 × Debug/Release + Regression

Date: 2026-09-01
baseline: ConvoPeq.md Generated 2026-09-01 21:47:45（--check FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0 実測）
Production source: 0（本工程は完全 read-only）
Test source: 0
CMake: 0
Build: 0（既存 exe 使用 — CR-α-3R Debug 21:29 / Release 21:37 生成物）
CTest: RUN（Debug + Release / --output-on-failure）
stress: 0
```

## 総合判定

> ## **CR-α-4 = PASS** — Debug 40/40・Release 40/40。CR-α-5（retry-specific source audit）へ進行可

---

## 1. 実施前 baseline 確認（実測）

```text
ConvoPeq.md --check: Generated 2026-09-01 21:47:45 / NEWER_SRC_COUNT=0 / STATUS FRESH → CTest 開始条件充足
累積 diff 再確認: BuildErrorClassificationTests.cpp +88/−2（CR-α-1 +87/−2 + CR-α-3R +1/−0 = 意図どおり）
RetrySchedulerTypes.h / AudioEngine.h / CMakeLists.txt: diff 0（name-only 不在）
```

## 2. Debug CTest（実測）

```text
ctest --test-dir build -C Debug --output-on-failure
CRA4_DBG_CTEST_EXIT = 0
Total tests: 40 / Passed: 40 / Failed: 0（"100% tests passed out of 40" 実測）
Log: evidence/cra4_ctest_debug.log
```

全 40 test 明細（実 CTest 出力を正とする・全 Passed）:

```text
 1/40 PublicationAdmissionTests           0.12s   21/40 ISRSemanticValidationRejects      0.22s
 2/40 D8_1_WrapperCacheTests              0.03s   22/40 InvariantINV3INV5                 0.14s
 3/40 D8_2_B_2_Tests                      0.16s   23/40 AdmissionPackedState              0.36s
 4/40 TerminalTelemetryContract           0.10s   24/40 RetireGraceSemantics              0.11s
 5/40 RuntimeHealthMonitorTierTests       0.13s   25/40 ShutdownRetireIntentDrain         0.09s
 6/40 ISRSoakTests                        0.44s   26/40 StuckReaderFallbackDrain          0.12s
 7/40 OwnerChannel                        0.07s   27/40 NormalRetireDSPHandleCompare      0.06s
 8/40 BuildErrorClassificationTests       0.03s   28/40 RuntimeSemanticSchemaValidation   0.05s
 9/40 RetrySchedulerTests                 0.65s   29/40 ObservePathSingleSource           0.02s
10/40 DeferredDeletionQueueReclaimTests   3.05s   30/40 OverlapAuthoritySingular          0.03s
11/40 MpscBoundedRingTests                0.04s   31/40 ShadowCompareContract             0.03s
12/40 SequenceArithmeticTests             0.05s   32/40 CrossfadeExecutorLocalContract    0.07s
13/40 DSPHandleTableTests                 0.06s   33/40 RuntimeWorldAuthorityProjection   0.32s
14/40 GainStagingContractTests            0.05s   34/40 PartialPublicationReject          0.03s
15/40 EQProcessorMaxGainTests             0.04s   35/40 RebuildAdmissionRegression        0.02s
16/40 EQAnalysisUnitTests                 0.04s   36/40 HeadlessAudioPathVerification    11.15s
17/40 FFTBackendTests                     0.06s   37/40 BuildInputSemanticContract        0.04s
18/40 EQBoundExcessBenchmark              0.23s   38/40 PriorityIntegration               0.10s
19/40 ISRRuntimeIdentityGenerators        0.03s   39/40 MTNUPCMeasurement                 0.44s
20/40 RuntimePublicationCoordinatorRej.   0.03s   40/40 AudioEngineHarness               16.14s
```

### CR-α テストの実行対象化確認

T-CRα-1〜4 は CTest 項目ではなく **#8 BuildErrorClassificationTests 内の runTestF() 内部 CHECK**
（test 名を仮定した追加実行は行わない — 指示どおり）。内部 CHECK の実際の実行証明として
standalone 実行（ND-04 で確立した targeted 実行惯例）を実施:

```text
Debug standalone:   [PASS] TestF warmup retry policy (delay table + saturation + decision + mapping)
                    checks=86 fails=0 PASS / DBG_RUN_EXIT=0
Release standalone: [PASS] TestF warmup retry policy (delay table + saturation + decision + mapping)
                    checks=86 fails=0 PASS / REL_RUN_EXIT=0
→ T-CRα-1（backoff table + saturation）/ T-CRα-2（{10,80,2} + K=3）/ T-CRα-3（decision truth table）
  / T-CRα-4（disposition→delay mapping）が両 config で実行・全 PASS を実測
```

## 3. Release CTest（Debug 40/40 PASS 後に実施・実測）

```text
ctest --test-dir build -C Release --output-on-failure
CRA4_REL_CTEST_EXIT = 0
Total tests: 40 / Passed: 40 / Failed: 0（"100% tests passed out of 40" 実測）
Log: evidence/cra4_ctest_release.log
```

全 40 test 明細: Debug と同一 test 名構成で全 Passed（所要時間は最速 #29 0.04s〜最遅
#40 AudioEngineHarness 15.82s）。個別明細は Log 参照（#8 BuildErrorClassificationTests 0.03s /
#9 RetrySchedulerTests 0.55s / #20 RuntimePublicationCoordinatorRejects 0.03s /
#35 RebuildAdmissionRegression 0.04s を含め全 40/40 Passed 実測）。

## 4. CR-α-1 回帰対象（指示 §4 の合格条件別確認）

| 確認対象 | Debug | Release |
|---|---|---|
| BuildErrorClassificationTests（#8） | PASS | PASS |
| RetrySchedulerTests（#9） | PASS | PASS |
| RebuildAdmissionRegression（#35） | PASS | PASS |
| RuntimePublicationCoordinatorRejects（#20） | PASS | PASS |
| PublicationAdmissionTests（#1） | PASS | PASS |
| ISR 系既存 tests（#6 ISRSoak・#19 ISRRuntimeIdentity・#21 ISRSemanticValidation） | PASS | PASS |
| **全 CTest** | **40/40** | **40/40** |

## 5. source 変更禁止の遵守

```text
Production source = 0 / Test source = 0 / CMake = 0（本工程）
CTest 実行中の failure 発生 = 0（修正機会も含め発生せず）
```

## 6. CTest 後の diff / forbidden check（実測）

```text
RetrySchedulerTypes.h = diff 0（name-only 不在）
AudioEngine.h         = diff 0（name-only 不在）
CMakeLists.txt        = diff 0（name-only 不在）

CTest 実施後の累積 diff（CTest 開始前の numstat と完全一致 → CR-α-3R 以降 source 増加 0）:
  .gitignore                                      1  0   （ND 運用・CR-α 分離対象外）
  ConvoPeq.md                                   462 20   （派生 snapshot・21:47:45）
  output_sourcecode_markdown.py                  84  0   （ND 運用・分離対象外）
  src/audioengine/AudioEngine.RebuildDispatch.cpp  67 17   ★ CR-α-1
  src/audioengine/BuildErrorPolicy.h               81  0   ★ CR-α-1
  src/audioengine/RuntimeWorldAuthority.h          63  0   （ND-01〜04 残留・分離対象外）
  src/tests/BuildErrorClassificationTests.cpp      88  2   ★ CR-α-1 +87/−2 + CR-α-3R +1/−0
  src/tests/ISRSemanticValidationTests.cpp        162  0   （ND-01〜04 残留・分離対象外）
★ = CR-α 新規差分（ND 残留と明確分離済み）
```

## 7. Artifacts（source/test/CMake 以外）

```text
tools/cra4_ctest.bat          : CR-α-4 CTest helper（vcvarsall x64 + oneAPI quoted CL include）
evidence/cra4_ctest_debug.log : Debug CTest log（40/40・全明細）
evidence/cra4_ctest_release.log: Release CTest log（40/40・全明細）
```

## 8. Overall

```text
Overall:
CR-α-4 = PASS

Debug:   Total 40 / Passed 40 / Failed 0 / EXIT 0
Release: Total 40 / Passed 40 / Failed 0 / EXIT 0
T-CRα-1〜4: 両 config TestF PASS（checks=86 fails=0・standalone 実行実測）
source/CMake 無変更: 確認済み（numstat 事後一致）
```

## 9. Next

```text
CR-α-4 = PASS → CR-α-5 進行可

CR-α-5 — retry-specific source audit（read-only・最新 ConvoPeq.md 21:47:45 基準）
  ①〜⑫ の 12 項目（counter ownership / generation rebind / #4 境界 / 再発行なし /
  NoRetry 経路 / Immediate boundedness / reject rollback / one-shot exhaustion /
  Exhausted→submitRebuildIntent なし / diagLog 追跡性 / K=4≠K=3 分離 /
  forbidden footprint）を「テスト通過」と別の証明として実施
```

## 遷移

```text
CR-α-1   Implementation             PASS
CR-α-2   Read-only verification      PASS
CR-α-3   Build                       FAIL（using 漏れ）
CR-α-3R  Minimal test repair         PASS（Debug/Release build PASS）
CR-α-4   CTest 40/40 × 2             ← NOW = PASS
CR-α-5   Retry-specific audit        解禁（待機）
Closure                              待機
```
