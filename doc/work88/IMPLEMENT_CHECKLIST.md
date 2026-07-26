# ConvoPeq BUG修正 実装チェックリスト

- **作成日**: 2026-07-26
- **ベース**: `doc/work88/REPAIR_PLAN.md` v24
- **凡例**: ✅ 完了 / 🔧 作業中 / ⏳ 未着手 / ❌ 保留

---

## グループA: 即時実施可能（13件）— 実装可

| ID | BUG | ファイル | 修正内容 | 状態 | 担当 |
|----|-----|----------|----------|------|------|
| A-1 | BUG-038 | `SpectrumAnalyzerComponent.h:74` | FFT_MAGNITUDE_SCALE 4→2 | ✅ | |
| A-2 | BUG-035 | `ConvolverProcessor.LoadPipeline.cpp` | RAII ApplyComputedIRLoadingGuard 導入 | ✅ | |
| A-3 | BUG-036 | `ConvolverProcessor.LoadPipeline.cpp:616` | irL/irR release()→get()+成功時release | ✅ | |
| A-4 | BUG-034 | `MKLNonUniformConvolver.cpp` (6箇所) | IPP FFT戻り値チェック＋clearFFTOutputOnError | ✅ | |
| A-5a | BUG-011 | `CmaEsOptimizer.h:79` | deserializeFrom sigma std::clamp | ✅ | |
| A-5b | BUG-012 | `CmaEsOptimizerDynamic.h:29` | setSigma std::clamp | ✅ | |
| A-5c | BUG-013 | `CmaEsOptimizerDynamic.cpp:204` | deserializeFrom sigma std::clamp | ✅ | |
| A-6 | BUG-029 | `DSPTransition.h:54-74` | Emergency Override exchangeFadingRuntimeDSP追加 | ✅ | |
| A-7 | BUG-028 | `CrossfadeRuntime.h:93-105` | complete()にフラグリセット3行追加 | ✅ | |
| A-8a | BUG-015 | `ISRRetireRouter.cpp:154` | enqueueWithRetry戻り値チェック | ✅ | |
| A-8b | BUG-015 | `SnapshotCoordinator.cpp:37` | enqueueWithRetry戻り値チェック | ✅ | |
| A-8c | BUG-015 | `SnapshotCoordinator.cpp:89` | enqueueWithRetry戻り値チェック | ✅ | |
| A-9 | BUG-016 | `CmaEsOptimizer*.h` (2箇所) | sanitizeにisfinite()追加 | ✅ | |
| A-10a | BUG-042 | `CmaEsOptimizer.h` | Rule of Five =delete追加 | ✅ | |
| A-10b | BUG-044 | `MklFftEvaluator.h` | Rule of Five =delete追加 | ✅ | |
| A-10c | BUG-046 | `PsychoacousticDither.h` | Rule of Five =default move追加 | ✅ | |
| A-11 | BUG-045 | `IRConverter.cpp:271` | actualSampleRate = sourceRate | ✅ | |
| A-12 | BUG-039 | `CustomInputOversampler.cpp:836-841` | memcpy長をminで制限 | ✅ | |
| A-13 | BUG-040 | `NoiseShaperLearner.cpp:1164-1168` | 1→48000 fallback | ✅ | |

---

## グループB: 設計確定済み（6件）

| ID | BUG | ファイル | 修正内容 | 状態 | 担当 |
|----|-----|----------|----------|------|------|
| B-1 | BUG-030 | `DSPTransition.h`, `AudioEngine.Timer.cpp`, `AudioEngine.h` | claimFadingRuntimeDSP CAS-only | ⏳ | |
| B-2 | BUG-023 | `SafeStateSwapper.h` | ⚠️ **未確定・実装保留** | ❌ | — |
| B-3 | BUG-031 | `AudioEngine.h:3696-3706`, `BlockDouble.cpp` | updateAudioThreadSnapshotFade実装 | ⏳ | |
| B-4 | BUG-032 | `AudioEngine.Snapshot.cpp:28-53` | GlobalSnapshot一括取得 | ⏳ | |
| B-5 | BUG-024 | `SnapshotFadeState.h:41-67` | fadeGeneration追加＋state再確認 | ⏳ | |
| B-6 | BUG-037 | `ConvolverProcessor.h`, `LoadPipeline.cpp` | loaderGenerationカウンタ | ⏳ | |

---

## グループC: 計画的対応（7件）

| ID | BUG | ファイル | 修正内容 | 状態 | 担当 |
|----|-----|----------|----------|------|------|
| C-1 | BUG-033 | `BlockDouble.cpp:400-427` | dryScaleラムダキャプチャ追加 | ⏳ | |
| C-2 | BUG-025 | `SnapshotCoordinator.cpp:57-72` | switchImmediate enqueueWithRetry化 | ⏳ | |
| C-3 | BUG-018 | 3ファイル | `!=1.0`→`std::abs(x-1.0)>1e-12` | ⏳ | |
| C-4 | BUG-019 | `TruePeakDetector.cpp:102-111` | int→size_t | ⏳ | |
| C-5 | BUG-020 | `LoaderThread.cpp:198` | targetLength<=0ガード | ⏳ | |
| C-6 | BUG-021/022 | `Lifecycle.cpp` | RCU GlobalGuard追加 | ⏳ | |
| C-7 | BUG-026 | `ObservedRuntime.h:42-49` | rootEnterSucceeded確認 | ⏳ | |

---

## グループD: 余裕時（4件）

| ID | BUG | ファイル | 修正内容 | 状態 | 担当 |
|----|-----|----------|----------|------|------|
| D-1 | BUG-041 | `NoiseShaperLearner.cpp:643` | VLA→ヒープ | ⏳ | |
| D-2 | BUG-043 | `IRConverter` | パラメータ名修正 | ⏳ | |
| D-3 | BUG-027 | `SnapshotCoordinator` | target==null時state再確認 | ⏳ | |
| D-4 | BUG-046 | `PsychoacousticDither.h` | A-10に含む | ⏳ | |
