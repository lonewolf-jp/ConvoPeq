# D101-25 Release Full Gate Audit

Date: 2026-08-24
Baseline: ConvoPeq.md 再生成 2026-08-24 00:42:10 (output_sourcecode_markdown.py TARGET src/build.bat/CMakeLists.txt)
Prior baseline: 2026-08-23 23:29:45 (D101-24 正本) → 本 Gate は新規生成版で検証、コード変更 0
Toolchain: MSVC 19.51.36256 + Intel oneAPI MKL 2026.1 / IPP, Ninja Multi-Config, JUCE 8.x, CMAKE_CXX_FLAGS_RELEASE PGO/LTCG

## Step 1 — Baseline 固定

- `python output_sourcecode_markdown.py` 実行 → `ConvoPeq.md` 再生成 (`# Project Extract & Source Code: ConvoPeq / Generated: 2026-08-24 00:42:10`)
- `git status --short / git log -1 / git diff --stat / git diff --check` 実行
  - 期待: D101-24 wiring 以外の意図しない差分なし、`git diff --check` 0
  - 実績: D101-24 差分のみ残存、ConvoPeq.md 更新は本 Step の正本更新として許容、他差分なし → 継続判定
- 中断条件（意図しない差分）に該当せず Step 2 へ進捗

## Step 2 — Release Build

- `cmake -S . -B build -G "Ninja Multi-Config"` configure 成功 (MKL static intel_lp64 sequential, IPP found, juceaide ok)
- `cmake --build build --config Release --parallel 8` 成功
  - 警告: `premature end of file; recovering` (ninja 既知の一過性、recovery 後完走)
  - 成果: `[89/90] AudioEngineHarness.exe` まで到達、`ConvoPeq / RetrySchedulerTests` 含む全 target link 成功（Debug 37/37 と同一 invariant）

## Step 3 — Release CTest 37/37

```
Test project C:/VSC_Project/ConvoPeq/build
  1  PublicationAdmissionTests .................   Passed    0.03 sec
  2  TerminalTelemetryContract .................   Passed    0.03 sec
  3  RuntimeHealthMonitorTierTests .............   Passed    0.03 sec
  4  ISRSoakTests ..............................   Passed    0.21 sec
  5  OwnerChannel ..............................   Passed    0.04 sec
  6  BuildErrorClassificationTests .............   Passed    0.18 sec
  7  RetrySchedulerTests .......................   Passed    0.70 sec
  8  DeferredDeletionQueueReclaimTests .........   Passed    3.16 sec
  9  MpscBoundedRingTests ......................   Passed    0.15 sec
 10  SequenceArithmeticTests ...................   Passed    0.15 sec
 11  DSPHandleTableTests .......................   Passed    0.15 sec
 12  GainStagingContractTests ..................   Passed    0.14 sec
 13  EQProcessorMaxGainTests ...................   Passed    0.15 sec
 14  EQAnalysisUnitTests .......................   Passed    0.27 sec
 15  FFTBackendTests ...........................   Passed    0.19 sec
 16  EQBoundExcessBenchmark ....................   Passed    0.17 sec
 17  ISRRuntimeIdentityGenerators ..............   Passed    0.03 sec
 18  RuntimePublicationCoordinatorRejects ......   Passed    0.03 sec
 19  ISRSemanticValidationRejects ..............   Passed    0.04 sec
 20  InvariantINV3INV5 .........................   Passed    0.04 sec
 21  RetireGraceSemantics ......................   Passed    0.03 sec
 22  ShutdownRetireIntentDrain .................   Passed    0.03 sec
 23  StuckReaderFallbackDrain ..................   Passed    0.18 sec
 24  NormalRetireDSPHandleCompare ..............   Passed    0.04 sec
 25  RuntimeSemanticSchemaValidation ...........   Passed    0.15 sec
 26  ObservePathSingleSource ...................   Passed    0.15 sec
 27  OverlapAuthoritySingular ..................   Passed    0.18 sec
 28  ShadowCompareContract .....................   Passed    0.14 sec
 29  CrossfadeExecutorLocalContract ............   Passed    0.15 sec
 30  RuntimeWorldAuthorityProjectionContract ...   Passed    0.37 sec
 31  PartialPublicationReject ..................   Passed    0.15 sec
 32  RebuildAdmissionRegression ................   Passed    0.15 sec
 33  HeadlessAudioPathVerification .............   Passed   12.30 sec
 34  BuildInputSemanticContract ................   Passed    0.15 sec
 35  PriorityIntegration .......................   Passed    0.15 sec
 36  MTNUPCMeasurement .........................   Passed    0.28 sec
 37  AudioEngineHarness ........................   Passed   15.74 sec
100% tests passed, 0 tests failed out of 37 (36.39 sec)
```

重点9件 明示確認: `RetrySchedulerTests / BuildErrorClassificationTests / RebuildAdmissionRegression / ShutdownRetireIntentDrain / StuckReaderFallbackDrain / RuntimePublicationCoordinatorRejects / PublicationAdmissionTests / ISRSemanticValidationRejects / InvariantINV3INV5` 全 PASS。RetryScheduler wiring の recovery/publication 漏洩なし。

Debug 37/37 (前回 34.99s) と Release 37/37 (36.39s) 同一 invariant、構成差異による分岐なし。

## Step 4 — Release 環境で Gates A-D 再確認（コード変更 0）

### Gate A — production caller = 1

- `grep -rn 'retryScheduler_->schedule' src --include='*.h' --include='*.cpp'`
  - `src/audioengine/AudioEngine.RebuildDispatch.cpp:1178: retryScheduler_->schedule(req, std::chrono::milliseconds(0));` のみ → 1
  - test 側 `.schedule` は `RetrySchedulerTests.cpp` に分離、production 混入 0

### Gate B — scheduler への semantic 侵入 0

- `grep -rn 'BuildError|RetryDisposition|RecoveryGeneration|epoch|RuntimeWorld|PublicationAdmission|RecoveryEpisode|supersession|obligation' src/audioengine/RetryScheduler.h src/audioengine/RetryScheduler.cpp` → 0 (EXIT 0 / no output)

### Gate C — scheduler 側に直接 API 0 ＋ boundary 維持

- `grep -rn 'submitRebuildIntent|requestRebuild|rebuildRequestGeneration' src/audioengine/RetryScheduler.h src/audioengine/RetryScheduler.cpp` → 0
- Boundary チェーン確認:
  - `AudioEngine.CtorDtor.cpp:98-99` `make_unique<RetryScheduler>([this](req){ this->submitRebuildIntent(req.kind,req.reason,req.rebuildClass,req.collapsePolicy); })` (DispatchFn)
  - `AudioEngine.RebuildDispatch.cpp:151` `submitRebuildIntent(kind,reason,class,policy)` → `334/418` `requestRebuild(sr,bs,…)` → `459` `requestRebuild(sr,bs)` 内 `rebuildRequestGeneration++`（既存 admission 境界）
  - `RetryScheduler → DispatchFn → submitRebuildIntent → requestRebuild → rebuildRequestGeneration++` 単一境界 維持

### Gate D — shutdown order

- `grep -n 'retryScheduler_|RetryScheduler::shutdown|shutdownCoordinatorLoop|stopRebuildThread' src/audioengine/AudioEngine.CtorDtor.cpp`
  ```
  98: retryScheduler_ = make_unique<RetryScheduler>(...)
  123: if (retryScheduler_) retryScheduler_->shutdown();
  126: shutdownCoordinatorLoop();
  127: stopRebuildThread();
  ```
  - `RetryScheduler::shutdown() → Coordinator停止 → RebuildThread停止` 固定、member destruction 依存なし、idempotent

### 全ツール横断 検証

- WSL: `rg / ag / fdfind / fzf / sed / awk` — Gate A 1 / B 0 / C 0 / D 順序固定で一致
- `ast-grep sg run -p 'RetryScheduler' / 'retryScheduler_' / 'BuildError'` — 各 2 / 5 / 0 hits で rg と一致
- `serena .serena/project.yml` language_servers [cpp,python,bash] 正常
- `cocoindex ccc status` 133k chunks / `ccc grep RetryScheduler` hits=RetryScheduler.h/.cpp/CtorDtor/AudioEngine.h/RebuildDispatch のみ / `ccc grep BuildErrorPolicy` 集約確認
- `graphify query RetryScheduler` 1 node（重複なし）
- `semble search RetryScheduler` top RetryScheduler.h / `search BuildError` policy 側のみ
- `AiDex` `.aidex` index 維持、rg 代替で contamination 0 確認（規約: 毎回 ConvoPeq.md 基準）

## Step 5 — Close / Freeze

- 本 Gate でコード変更 0 を維持（上記 Gates 再確認は read-only）
- W1-W6 formal wiring tests は `追加可能` に留め、現行 37/37 を崩して直ちに追加しない（D101-24 結論踏襲）
- よって `D101-24 → D101-25 Release Full Gate PASS → RetryScheduler wiring freeze` とする

## 次フェーズ ロードマップ（本 Gate 後に着手）

REPAIR_PLAN2-dash2 Tier1 優先度に従い、RetryScheduler 拡張ではなく lifetime/shutdown 系へ復帰:

```
[D101-24] RetryScheduler Production Wiring — close
[D101-25] Release Full Gate — PASS (本書)
RetryScheduler wiring freeze
  ↓
[次] Shutdown / Lifetime Proof — ShutdownQuiescenceProof Q0-Q7/C1-C7/ShutdownCompletionAuthority 現行コード監査
  ↓
isFullyDrained semantic
  ↓
Retire ordering
  ↓
必要なら currentWorld_ read-source singularization
```

D14/D15 obligation/backpressure、Practical Stable ISR Bridge Runtime の `Shutdown が完全 Drain を保証` 条件は本 wiring と非交差のため、本フェーズで変更しない。

## 結論

- ConvoPeq.md を 2026-08-24 00:42:10 版に更新し baseline 固定
- Release Build 成功、Release CTest 37/37 PASS（重点9件含む）
- Gates A-D 全 PASS（Release 環境で再確認、コード変更 0）
- D101-25 close、RetryScheduler wiring freeze、次は Shutdown/Lifetime Proof 監査へ
