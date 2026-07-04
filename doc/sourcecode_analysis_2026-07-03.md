# ConvoPeq ソースコード構造解析 (2026-07-03)

## プロジェクト概要

**ConvoPeq v0.6.8** — IR畳み込み + 20バンド パラメトリックEQ + リアルタイム アナライザ を統合したWindows 11 x64 スタンドアロン音声処理アプリ

**スタック**: JUCE 8.0.12 + Intel oneMKL (sequential) + Intel IPP + AVX2 / C++20

## 1. リポジトリ全体構成

```
C:\VSC_Project\ConvoPeq\
├── src/           ← 主要ソース (246ファイル / ~2.78 MB)
├── config/        ← JSON 権威マニフェスト (4ファイル)
├── JUCE/          ← JUCE 8.0.12 (in-tree, 3,056ファイル / 44 MB)
├── r8brain-free-src/ ← IR リサンプラ (49ファイル / 14.5 MB)
├── sampledata/    ← テスト IR/音声
├── manual/        ← EN+JP ユーザーマニュアル
├── tools/         ← Python/PS CI 検証スクリプト
├── doc/           ← アーキテクチャ作業ドキュメント
├── CMakeLists.txt (985行) / CMakePresets.json / build.bat
└── README.md / ARCHITECTURE.md / SOUND_PROCESSING.md / BUILD_GUIDE_WINDOWS.md
```

## 2. src/ 階層

| ディレクトリ | ファイル数 | サイズ | 役割 |
|---|---|---|---|
| `src/` ルート | 77 | ~966 KB | トップレベル DSP/UI/IF |
| `src/audioengine/` | 101 | ~1.19 MB | ISR ランタイム ガバナンス層 |
| `src/convolver/` | 10 | ~251 KB | 畳み込みスプリット実装 |
| `src/core/` | 37 | ~118 KB | RCU スナップショット基盤 |
| `src/eqprocessor/` | 6 | ~163 KB | 20バンド EQ スプリット実装 |
| `src/tests/` | 15 | ~153 KB | CTest 回帰テスト |

### 2.1 `src/` ルート — 主要ファイル

**アプリエントリ / 起動・終了**
- `MainApplication.{h,cpp}` (171行) — JUCEApplication サブクラス。シングルトン、`MainWindow` 保持
- `MainWindow.{h,cpp}` (62.2 KB) — JUCE `DocumentWindow`。`AudioEngine`/`EQControlPanel`/`SpectrumAnalyzerComponent`/`DeviceSettings` を保持
- `MKLRealTimeSetup.{h,cpp}` — MKL 逐次実行セットアップ
- `AsioBlacklist.h` — 既知の壊れた ASIO ドライバ除外

**オーディオI/O アダプタ**
- `audioengine/AudioEngineProcessor.{h,cpp}` — `juce::AudioProcessor` ダブル/フロート対応。`AudioProcessorPlayer` 経由でデバイス コールバックにブリッジ
- `DeviceSettings.{h,cpp}` (51.7 KB) — ASIO/WASAPI 永続化 (`device_settings.xml`)

**畳み込み トップ**
- `ConvolverProcessor.h` (1,180行, 60 KB) — 公開 API
- `MKLNonUniformConvolver.{h,cpp}` (65.2 KB) — MKL NUC (Non-Uniform Partitioned Convolution)
- `IRConverter.{h,cpp}` / `IRDSP.{h,cpp}` / `InputBitDepthTransform.h` / `AllpassDesigner.{h,cpp}` (26.8 KB) / `ConvolverState.{h,cpp}` / `ConvolverSettingsComponent.{h,cpp}` / `ConvolverRuntimeCompatAliases.h` / `IRAdvancedSettings` 等

**ノイズシェイパ (複数実装)**
- `FixedNoiseShaper.h` (13.6 KB), `Fixed15TapNoiseShaper.h` (19.4 KB), `LatticeNoiseShaper.h` (10.3 KB) — SIMD フォールバック
- `NoiseShaperLearner.{h,cpp}` (**68.4 KB** — 最大 TU) — CMA-ES 駆動 9次IIR 適応学習
- `NoiseShaperLearnerTypes.h` / `NoiseShaperLearningComponent.cpp` (22.8 KB) / `PsychoacousticDither.h`

**DSP ユーティリティ**
- `CustomInputOversampler.{h,cpp}` (33.5 KB) — AVX2 多段 FIR/IIR アップサンプラ (2x/4x/8x)
- `OutputFilter.{h,cpp}` (20.2 KB) — 出力段フィルタ (HCMode/LCMode切替) + SoftClip
- `UltraHighRateDCBlocker.h` / `TruePeakDetector.{h,cpp}` / `LoudnessMeter.{h,cpp}` / `DftiHandle.h`

**EQ 編集 (UI/Worker 側)**
- `EQProcessor.h` / `EQProcessor.cpp` (4.2 KB — 薄い API)
- `EQEditProcessor.{h,cpp}` (4.2 KB) — UI/ワーカー側 EQ 編集

**DSP/インフラ**
- `AlignedAllocation.h` (6.0 KB) — 64B SIMD 整列 `aligned_malloc` + `ScopedAlignedPtr`
- `DspNumericPolicy.h` (13.5 KB) — dsp 数値定数/型の単一ソース (`kDenormThresholdAudioState` 等)
- `LockFreeRingBuffer.h` (4.8 KB) / `LockFreeAudioRingBuffer.h` (7.6 KB) — オーディオスレッド安全リング
- `DeferredDeletionQueue.h` (12.4 KB) / `DeferredFreeThread.h` (8.6 KB) / `RefCountedDeferred.h` (2.8 KB) — 非同期再回収
- `SafeStateSwapper.h` (19.7 KB) — RAII 状態スワップ
- `MixedPhasePersistentCache.{h,cpp}` / `CacheManager.{h,cpp}` (16.2 KB) / `CmaEsOptimizer{,.Dynamic}.{h,cpp}` / `GenerationManager.h` / `StateKey.h` / `ProgressiveUpgradeThread.{h,cpp}` (7.3 KB)

**UI**
- `EQControlPanel.{h,cpp}` (26.0 KB) — 20バンド EQ UI
- `SpectrumAnalyzerComponent.{h,cpp}` (52.3 KB) — FFT + EQ オーバーレイ、ピークホールド/スムーシング
- `MixedPhaseOptimizationComponent.{h,cpp}` (3.7 KB)

### 2.2 `src/audioengine/` — ISR ランタイム (101 ファイル, ~1.19 MB)

このプロジェクトの**アーキテクチャ的中核**。`AudioEngine.h` 単独で **5,600+ 行** (`AudioEngine.h` 全体)。

**AudioEngine スプリット TU**
- `AudioEngine.h` — 全型定義 + `RuntimeState` (`BuilderToken` で封印)
- `AudioEngine.CtorDtor.cpp` (11.9 KB) — コンストラクタ/デストラクタ
- `AudioEngine.Init.cpp` (4.7 KB) / `AudioEngine.Globals.cpp` (0.2 KB)
- `AudioEngine.Parameters.cpp` (32.5 KB) — 高レベル UI パラメータ

**Processing 系列** (オーディオスレッド コア)
- `AudioEngine.Processing.AudioBlock.cpp` (32.8 KB) — オーディオスレッド入口 (float 経路)
- `AudioEngine.Processing.BlockDouble.cpp` (28.7 KB) — ダブル精度経路
- `AudioEngine.Processing.DSPCoreFloat.cpp` (15.3 KB) / `DSPCoreDouble.cpp` (36.0 KB) — DSP コアの float/double
- `AudioEngine.Processing.DSPCoreLifecycle.cpp` (17.2 KB) / `.DSPCoreIO.cpp` (18.8 KB) / `.DSPCoreToBuffer.cpp` (1.4 KB)
- `AudioEngine.Processing.PrepareToPlay.cpp` (16.2 KB)
- `AudioEngine.Processing.ReleaseResources.cpp` (23.6 KB)
- `AudioEngine.Processing.Latency.cpp` (5.8 KB) / `.Snapshot.cpp` (2.1 KB)

**Publishing / Rebuild パイプライン**
- `AudioEngine.Commit.cpp` (32.2 KB) — 原子 commit/publish `RuntimeState`
- `AudioEngine.RebuildDispatch.cpp` (46.3 KB) — デバウンス再ビルドディスパッチャ
- `AudioEngine.Timer.cpp` (**75.6 KB** — 最大 TU) — UI タイマーポーリング
- `AudioEngine.Threading.cpp` (7.5 KB) / `AudioEngine.Transition.cpp` (1.2 KB)
- `AudioEngine.Learning.cpp` (26.0 KB) — 適応ノイズシェイパー統合
- `AudioEngine.EQResponse.cpp` (10.5 KB) / `AudioEngine.UIEvents.cpp` (9.3 KB)
- `AudioEngine.StateIO.cpp` (10.7 KB) / `AudioEngine.Publication.cpp` (2.0 KB)
- `AudioEngine.Retire.cpp` (18.5 KB) — 旧 `RuntimeState` retire-router ロジック

**ISR サブシステム (ガバナンス層)**
- `ISRAuthorityClass.h` (1.5 KB) — `Authoritative/Derived/Diagnostic/ExecutorLocal` 列挙
- `ISRLifecycle.{h,cpp}` (9.7 KB) — ライフサイクル スケジューラ
- `ISRRTExecution.{h,cpp}` (4.5 KB) — RT 実行契約
- `ISRShutdown.{h,cpp}` (17.4 KB) — `ShutdownPhase` FSM (`Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → EmergencyDrain → VerifyDrained → TimedOut | Failed → ShutdownComplete`), `alignas(64) BlockingReasonStats`
- `ISRDSPHandle.{h,cpp}` (9.3 KB) — ハンドルベース DSP レジストリ (`DSPHandleRuntime::MAX_DSP_SLOTS`)
- `ISRDSPQuarantine.{h,cpp}` (2.4 KB) — 隔離セマンティクス
- `ISRClosure.{h,cpp}` — 反射クロージャグラフ / `ClosureNodeRef` / `PayloadClosureDescriptor`
- `ISRClosureGraphWalker.{h,cpp}` (3.0 KB) — グラフ トラバーサル (`validateGraph`)
- `ISRPayloadTier.{h,cpp}` (2.6 KB) — ペイロード階層化 (`PayloadTier::InlineImmutable` / `ImmutableShared`)
- `ISRHB.{h,cpp}` (9.2 KB) — ハートビート/ハザードバリア
- `ISRRetire.{h,cpp}` (9.8 KB) — `RuntimeState` リタイア
- `ISRRetireLane.h` (0.2 KB) / `ISRRetireOverflowRing.h` (4.8 KB) / `ISRRetireRouter.{h,cpp}` (5.3 KB) / `ISRRetireRuntimeEx.{h,cpp}` (21.3 KB)
- `ISRRuntimePublicationCoordinator.{h,cpp}` (23.3 KB) — **公開の要**
- `ISRRuntimeSemanticSchema.h` (19.6 KB) — スキーマ v9: 権威クラスの単一ソース (`kRuntimeSemanticSchemaVersion`)
- `ISRRuntimeIdentityGenerators.h` (1.0 KB) — ランタイム/遷移 UUID
- `ISRSealedObject.h` (2.9 KB) — RAII シールラッパ (Builder/Engine のみ構築可)
- `ISRDebugRuntime.{h,cpp}` (6.0 KB)

**Runtime Publication Center**
- `RuntimeHealthMonitor.{h,cpp}` (57.9 KB) — 連続テレメトリ/ヘルスモニタ
- `RuntimePolicyEngine.{h,cpp}` (10.4 KB) — リビルド入場ポリシー
- `RuntimePublicationOrchestrator.{h,cpp}` (18.6 KB) — 公開コレオグラフィ
- `RuntimePublicationValidator.{h,cpp}` (7.8 KB) — 静的検証 (Dither/NS/トポロジ)
- `RuntimePublicationCoordinator.{h,cpp}` (5.1 KB) — テンプレート化された公開実行
- `RuntimePublicationState.h` (7.0 KB)
- `PublicationAdmission.{h,cpp}` (2.4 KB) / `PublicationExecutor.{h,cpp}` (3.3 KB)
- `CrossfadeAuthority.{h,cpp}` (2.3 KB) / `CrossfadeRuntime.h` (9.0 KB) — クロスフェード実行ガバナンス
- `RuntimeBuilder.{h,cpp}` (28.0 KB) — `BuilderToken` 経由でのみ `RuntimeState` を構築
- `RuntimeBuildTypes.h` (3.7 KB) — Build スナップショット
- `RuntimeGraph.h` (5.1 KB) — グラフ基底
- `RuntimeTransition.h` (2.3 KB) — 状態遷移記述
- `RuntimeStore.h` (core 内)
- `FrozenRuntimeWorld.{h,cpp}` (4.4 KB) — フェーズ 4 不変 World 概念
- `WorldLifecycleAudit.{h,cpp}` (4.1 KB) — World 寿命監査
- `TelemetryRecorder.{h,cpp}` (4.3 KB)
- `RuntimeDrainAudit.h` — Drain 監査
- `ISREvidenceExporter.h` — 証拠エクスポータ
- `AtomicAccess.h` (5.8 KB) — `consumeAtomic`/`publishAtomic`/`fetchAddAtomic`/`compareExchangeAtomic` API (モジュール横断統一)
- `DSPLifetimeManager.h` (3.1 KB) / `DSPTransition.h` (6.8 KB)

### 2.3 `src/convolver/` — 畳み込みスプリット (10 ファイル, ~251 KB)

8個の `CONVOPEQ_ENABLE_CONVOLVER_SPLIT_*` フラグで分割:

- `ConvolverProcessor.Internal.h` (5.5 KB) — 分割専用ヘルパー (`unwrapPhaseRadians`, `nextPow2`, `resampleIR`, `convertToMinimumPhase`)
- `.Lifecycle.cpp` (22.1 KB) — ライフサイクル/RCU 統合
- `.Rebuild.cpp` (12.4 KB) — 再ビルド判定
- `.LoaderThread.cpp` (29.5 KB) + `LoaderThreadInline.h` (3.6 KB) — IR ロードスレッド
- `.LoadPipeline.cpp` (32.1 KB) — パイプライン処理
- `.MixedPhase.cpp` (38.9 KB) — As-Is/Mixed/Minimum 位相
- `.ResampleAndFallback.cpp` (18.0 KB) — r8brain フォールバック
- `.Runtime.cpp` (47.8 KB) — オーディオスレッドランタイム
- `.StateAndUI.cpp` (47.5 KB) — プリセット/UI

> **注**: `ConvolverProcessor.h` 自体は `src/` 直下に存在 (1180行)。
> **レガシー単一 TU** `MKLNonUniformConvolver.cpp` (~65 KB) は `#ifdef` でコンパイル保持 (後方互換)。

### 2.4 `src/eqprocessor/` — 20バンド EQ (6 ファイル, ~163 KB)

- `EQProcessor.h` (32.3 KB) — `EQBandType` (LowShelf/Peaking/HighShelf/LowPass/HighPass)、`EQChannelMode` (Stereo/Left/Right/Mid/Side)、TPT SVF 係数、Biquad 解析用係数
- `EQProcessor.Core.cpp` (42.4 KB)
- `EQProcessor.Coefficients.cpp` (19.3 KB) — SVF/Biquad 計算
- `EQProcessor.Parameters.cpp` (12.7 KB)
- `EQProcessor.Processing.cpp` (**57.2 KB** — EQ 最大 TU) — TPT SVF 処理 (AVX2 FMA 利用)
- `EQProcessor.ProcessingCache.cpp` (2.7 KB)

**RCU パターン + uintptr_t-backed atomic handle** でパラメータ更新。AGC (attack/release/smooth 係数テーブル)、シリアル/パラレル構造、M/S 処理、SoftSaturation (fastTanh) を実装。

### 2.5 `src/core/` — RCU 基盤 (37 ファイル, ~118 KB)

**スナップショット/RCU**
- `RCUReader.h` (8.7 KB) — RAII リーダー epoch 入場/退場
- `GlobalSnapshot.{h,cpp}` — 不変スナップショット基底
- `SnapshotFactory.{h,cpp}` (5.9 KB) / `SnapshotAssembler.{h,cpp}` / `SnapshotCoordinator.{h,cpp}` (7.9 KB) / `SnapshotParams.h` / `SnapshotFadeState.h` / `SnapshotSlotStore.h` / `SnapshotRetireManager.h`
- `RuntimeStore.h` (内部 Store) / `RuntimeReaderContext.h`

**Epoch**
- `EpochDomain.h` (**26.0 KB**) — 64名前付きリーダースロット、`globalEpoch` 管理
- `IEpochProvider.h` / `IReaderEpochProvider.h` / `IRetireProvider.h` / `IPublicationProvider.h`
- `ObserveChannel.h` / `ObservedRuntime.h` — 観測抽象
- `RebuildTypes.h` / `Types.h` / `TimeUtils.h`

**非同期削除**
- `DeletionQueue.{h,cpp}` / `DeferredRetireFallbackQueue.h`
- `WorkerThread.{h,cpp}` (4.5 KB) / `ThreadAffinityManager.h` / `CommandBuffer.h` / `FadeEngine.h`

**Telemetry**
- `RetireBoundaryTelemetry.h`

**Compat**
- `ConvolverRuntimeCompatTypes.h` / `EQParameters.h`

CMake コメント: "Phase 1: RCU Snapshot Foundation (v13.0)" → "Phase 3: Debounced snapshot worker" まで納品済み。

### 2.6 `src/tests/` — CTest 回帰 (15 ファイル, ~153 KB)

すべて `add_test(...)` バインド。一部は JUCE 非依存。主なファイル:

- `ISRRuntimeIdentityGeneratorsTests.cpp` (1.5 KB) — UUID 生成器
- `RuntimePublicationCoordinatorTests.cpp` (5.3 KB) — テンプレート実装
- `ISRSemanticValidationTests.cpp` (19.2 KB) — 意味検証
- `RetireGraceSemanticsTests.cpp` (12.7 KB) — retire grace
- `RuntimeSemanticSchemaValidationTests.cpp` (26.7 KB) — スキーマ検証
- `ObservePathSingleSourceTests.cpp` (2.3 KB)
- `OverlapAuthoritySingularTests.cpp` (2.1 KB)
- `ShadowCompareContractTests.cpp` (1.3 KB)
- `CrossfadeExecutorLocalContractTests.cpp` (2.5 KB)
- `RuntimeWorldAuthorityProjectionTests.cpp` (13.1 KB)
- `PartialPublicationRejectTests.cpp` (21.2 KB, MSVC で MKL リンク)
- `RebuildAdmissionRegressionTests.cpp` (3.4 KB)
- `BuildInputSemanticContractTests.cpp` (16.5 KB, `/STACK:8388608`)
- `PriorityIntegrationTests.cpp` (7.6 KB)
- `PublicationValidatorIsolationTests.cpp` (21.2 KB)

+ 外部 CI: `HeadlessAudioPathVerification` (PowerShell、$CONVO_CI_BUILD でゲート)

## 3. config/ (JSON マニフェスト)

| ファイル | サイズ | 役割 |
|---|---|---|
| `runtime_graph_baseline.json` | 97 B | ベースライン トポロジ参照 |
| `publication_manifest.json` | 2.0 KB | 機械可読公開インベントリ |
| `authority_inventory.json` | 10.5 KB | 各ランタイムフィールドの権威クラス宣言 (生成物) |
| `pub_boundary_registry.json` | 2.3 KB | 公開境界レジストリ |

**`tools/*.py` 検証スクリプト** がソースを JSON と照合し、権威ドリフトを検出 (=ビルド時契約保証)。

## 4. JUCE 統合 (in-tree, 8.0.12)

使用モジュール:

| モジュール | 用途 |
|---|---|
| `juce_core` / `juce_events` | ベース / MessageManager/Timer |
| `juce_audio_basics` / `juce_audio_devices` | AudioBuffer / **AudioDeviceManager (ASIO/WASAPI/DirectSound)** |
| `juce_audio_formats` | WAV/AIFF (IR ロード) |
| `juce_audio_processors` / `_headless` | AudioProcessor 基底 |
| `juce_audio_utils` | AudioProcessorPlayer ブリッジ |
| `juce_dsp` | DSP ユーティリティ + JUCE FFT (`JUCE_DSP_USE_INTEL_MKL=1`) |
| `juce_gui_basics` / `_extra` / `juce_graphics` | UI |
| `juce_data_structures` | ValueTree (XML 状態保存) |
| (`juce_opengl` 他多数は未使用) | |

主要結合点:
- `juce_add_gui_app(ConvoPeq ...)` — スタンドアロンターゲット
- `JUCE_DISABLE_ACCESSIBILITY=1` / `JUCE_WIN_PER_MONITOR_DPI_AWARE=0` / `JUCE_USE_CURL=0`
- `JuceLibraryCode/` が `ConvoPeq_artefacts/` 下に生成され、テストからも参照

## 5. r8brain-free-src 統合

**IR リサンプリング専用** (畳み込みは MKL)。MIT ライセンス。`INTERFACE` ライブラリとしてヘッダのみ使用。

```cmake
target_compile_definitions(r8brain INTERFACE R8B_FASTTIMING=1 R8B_EXTFILTERS=0)
```

`IppsFFTSpec_R_64f` の IPP バージョン間 不具合のため `R8B_IPP=1` は使用せず、組込 FFT (PFFFT) を使用。

主要消費ヘッダ: `CDSPResampler` / `CDSPHBDownsampler` / `CDSPHBUpsampler` / `CDSPBlockConvolver` / `CDSPFracInterpolator` / `fft/pffft`。

## 6. CMake ビルド構造

```
CMakeLists.txt (985行)
├── ProjectMetadata.cmake (include)
├── CMakePresets.json (3 configure + 2 build)
├── CMakeSettings.json (VS Code Tools cache)
└── build.bat (vcvarsall + setvars + cmake)
```

### 6.1 構成 / フラグ
- C++20
- `CONVOPEQ_ENABLE_CLANG_TIDY` / `_ENABLE_ISR_TESTS` (default ON) / `_ENABLE_RUNTIME_DIAGNOSTICS`
- `ENABLE_ASAN` (Debug のみ、CRT を `/MDd` に強制)
- `CONVOPEQ_PGO_INSTRUMENT` / `_PGO_USE` (3 モード)
- 環境 `CONVO_CI_BUILD` → `NUC_DEBUG_GUARDS` 有効化

### 6.2 ツールチェイン
- **MSVC 19.44+ (VS2022 17.11+)** デフォルト
- **Intel icx (oneAPI 2026.0)** 代替
- ジェネレータ: **Ninja Multi-Config**
- 出力: `build/` (vs2026) / `build-icx/` (icx)

### 6.3 MKL / IPP リンク
- **MSVC**: `MKL::MKL` (sequential + lp64、**静的リンク — RT スレッド競合回避**)
- **icx**: `/Qmkl:sequential` で .obj 直結
- **IPP**: `IPPROOT` が設定されていれば使用、なければスキップ
- `JUCE_DSP_USE_INTEL_MKL=1` で JUCE DSP モジュール経由でも MKL 配線

### 6.4 プラットフォーム
- Windows x64 のみ (`CMAKE_SIZEOF_VOID_P EQUAL 4` をガード)
- AVX2 強制 (`/arch:AVX2` / `/QxCORE-AVX2`)
- 静的 CRT (`/MT` Release, `/MTd` Debug) — ASan は例外
- Release で LTCG (`INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE`)
- `ole32` / `avrt` (WASAPI MMCSS) / `winmm` (timeBeginPeriod) / `psapi` (Diagnostics) リンク

### 6.5 Convolver Split コンパイル定義

```cmake
target_compile_definitions(ConvoPeq PRIVATE
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME=1
    CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI=1
)
```

## 7. 主要アーキテクチャ パターン

### 7.1 JUCE アプリ所有権ツリー
```
JUCEApplication (MainApplication)        [main() は START_JUCE_APPLICATION マクロ]
  └─ MainWindow (DocumentWindow, Timer, ChangeListener, Label::Listener)
        ├─ AsioBlacklist asioBlacklist
        ├─ AudioEngine audioEngine                          ← オーケストレータ
        │     ├─ EQEditProcessor uiEqEditor                  [UI/Worker 用独立インスタンス]
        │     │     └─ EQProcessor EQEditProcessor::eq
        │     ├─ ConvolverProcessor uiConvolverProcessor     [UI/Worker 用独立インスタンス]
        │     ├─ ISRRetireRouter m_retireRouter
        │     ├─ RuntimePublicationOrchestrator runtimeOrchestrator_
        │     ├─ RuntimeHealthMonitor m_healthMonitor
        │     ├─ RuntimePublicationBridge runtimePublicationBridge_
        │     ├─ CrossfadeRuntime crossfadeRuntime_
        │     ├─ WorkerThread m_workerThread
        │     └─ EQCacheManager eqCacheManager
        ├─ AudioEngineProcessor (juce::AudioProcessor)       [浮動小]
        │     └─ AudioProcessorPlayer audioProcessorPlayer → デバイス コールバック
        │              └─ audioEngine.getNextAudioBlock()
        ├─ EQControlPanel eqPanel
        ├─ ConvolverControlPanel convolverPanel
        ├─ SpectrumAnalyzerComponent specAnalyzer
        └─ DeviceSettings deviceSettings
```

### 7.2 ISR ランタイム ガバナンス
全フィールドが `Authoritative / Derived / Diagnostic / ExecutorLocal` のいずれかに分類 (`ISRAuthorityClass.h`)。

`ISRShutdown.h` の 10 層:
1. `RuntimeGraph` 2. `RuntimeState` (封印) 3. `RuntimeBuilder` (Token 経由のみ構築可)
4. `RuntimePublicationCoordinator` 5. `Validator`
6. `PublicationAdmission` / `Executor`
7. `CrossfadeAuthority`
8. **Shutdown FSM**: `Running → AudioStopped → ObserverDrained → RetireClosed → EpochSettled → ReclaimComplete → EmergencyDrain → VerifyDrained → TimedOut | Failed → ShutdownComplete`
9. `ISRRetire` / `ISRRetireRouter`
10. `RuntimeHealthMonitor` + `TelemetryRecorder`

### 7.3 RCU パターン (オーディオスレッド安全性)
- `EpochDomain` (64 名前付きリーダースロット、quiescent state ベース)
- `RCUReader` (RAII 入場/退場、`m_retireRouter` 経由)
- `SnapshotCoordinator` + `SnapshotAssembler` (不変スナップショットの原子公開)
- `DeferredDeletionQueue` + `DeferredFreeThread` (猶予後オフスレッド解放)
- `RefCountedDeferred` (共有資源の参照カウント)

EQ パラメータ更新、IR リロード、適応ノイズシェイパー係数更新など**全てで使用**。**オーディオスレッドは確保/ブロックを行わない**。

### 7.4 プラグイン互換 Processor 表面
スタンドアロンだが `AudioEngineProcessor` は完全な `juce::AudioProcessor` を実装 (`processBlock(float)` と `processBlock(double)`)。将来 VST/AU への展開を意図した設計。

### 7.5 スキーマ権威ビルド検証
`config/authority_inventory.json` と `ISRRuntimeSemanticSchema.h` + `RuntimeGraph.h` + `AudioEngine.h` の `kFieldDescriptors[21]` / `kRuntimeAuthorityInventory[21]` / `kRuntimeReadAuthorityInventory[10]` が対応。`tools/*.py` により、ソース内の権威宣言と JSON の整合を**ビルド時/コミット時に検証**。

---

## 8. 主要クラス早見表

| クラス/ファイル | 役割 (一行) |
|---|---|
| `MainApplication` | JUCE app シングルトン、`initialise/shutdown` |
| `MainWindow` | トップレベル UI 合成、`Timer`/`ChangeListener` |
| `AudioEngine` | ランタイム オーケストレータ (29 TU スプリット) |
| `AudioEngineProcessor` | `juce::AudioProcessor` アダプタ |
| `EQProcessor` | 20 バンド パラ EQ (TPT SVF + RCU) |
| `EQEditProcessor` | UI 側 EQ 編集 |
| `ConvolverProcessor` | IR 畳み込み (MKL NUC + 8 TU スプリット) |
| `MKLNonUniformConvolver` | MKL 駆動 分割畳み込み |
| `CustomInputOversampler` | AVX2 多段 UPS/DC ブロッカ |
| `OutputFilter` | 出力段フィルタ (HC/LC mode 切替) + SoftClip |
| `NoiseShaperLearner` | CMA-ES 駆動 9 次 IIR 学習 |
| `RuntimeBuilder` | `RuntimeState` 唯一の構築者 |
| `RuntimePublicationOrchestrator` | 検証 + 公開コレオグラフィ |
| `RuntimePublicationCoordinator` | 検証 + 公開実行 (テンプレート) |
| `ISRRuntimePublicationCoordinator` | ISR ガバナンス付き公開 |
| `ISRShutdown` | FSM (alignas(64) stats) |
| `RuntimeHealthMonitor` (57.9 KB) | 連続テレメトリ/HealthState |
| `RuntimePolicyEngine` | リビルド入場ポリシー |
| `EpochDomain` (26.0 KB) | RCU エポック/QS 状態 (64 slots) |
| `SnapshotCoordinator` | スレッド安全スナップショット |
| `DeferredDeletionQueue` | 非同期再回収 |
| `IRConverter` | 最小位相/混合位相変換 |
| `AllpassDesigner` | 混合位相タップ設計 |
| `CmaEsOptimizerDynamic` | CMA-ES 最適化 |
| `CacheManager` | IR 再ビルド キャッシュ |
| `TruePeakDetector` / `LoudnessMeter` | ITU-R BS.1770-4/5 + EBU R128 |
| `ProgressiveUpgradeThread` | 背景 FFT アップグレード |

---

## 10. 詳細データ処理フロー

### 10.1 起動からオーディオ処理開始まで

```
1. main()
   └─ START_JUCE_APPLICATION(MainApplication)  [JUCE マクロが main() を生成]

2. MainApplication::initialise(commandLine)
   ├─ FileLogger 初期化 → ConvoPeq.log
   ├─ Windows: timeBeginPeriod(1)  [タイマー精度 1ms]
   ├─ Windows: ProcessPowerThrottling 無効化 (EcoQoS)
   ├─ Windows: SetPriorityClass(HIGH_PRIORITY_CLASS)
   ├─ MKLRealTime::setup()  [MKL シングルスレッド + FTZ/DAZ]
   ├─ ippInit()  [IPP CPU ディスパッチ確定]
   ├─ _MM_SET_FLUSH_ZERO_ON / _MM_DENORMALS_ZERO_ON  [メインスレッド]
   ├─ vmlSetMode(VML_FTZDAZ_ON | VML_ERRMODE_IGNORE)
   ├─ MainWindow::showMainWindowAsync()
   │   └─ MainWindow コンストラクタ
   │       ├─ AudioEngine 構築 (RCU/EpoochDomain/ISR 初期化)
   │       ├─ AudioEngineProcessor 生成 (juce::AudioProcessor)
   │       ├─ AudioProcessorPlayer::setProcessor(audioEngineProcessor)
   │       ├─ AudioDeviceManager::initialise() → ASIO/WASAPI 列挙
   │       │   └─ AudioProcessorPlayer::start()  [デバイス I/O 開始]
   │       ├─ UI コンポーネント生成 (EQControlPanel 等)
   │       └─ startTimer(100)  [Timer 100ms 周期でポーリング]
   └─ mainWindow->runCommandLineAutomation(commandLine)
       └─ CLI モード時: プリセット自動適用、テレメトリ出力

3. オーディオ デバイス コールバック登録済み (AudioProcessorPlayer)
   ※実際の getNextAudioBlock() はデバイス開始後に呼ばれる
```

### 10.2 prepareToPlay パス (Message Thread、デバイス I/O 開始前)

```
AudioEngine::prepareToPlay(samplesPerBlockExpected, sampleRate)

Step 1: ライフサイクル状態遷移
  Unprepared → Preparing
  └─ setShutdownPhase(Running)
  └─ rebuildThread.start()  [如果未启动]

Step 2: パラメータ初期化
  ├─ publishAtomic(currentSampleRate, safeSampleRate)
  ├─ publishAtomic(maxSamplesPerBlock, bufferSize)
  ├─ m_irFadeTimeSec 計算・公開
  └─ crossfadeRuntime_.reset()

Step 3: レイテンシ整合バッファ確保
  ├─ maxDelay = min(kMaxLatencySamples, sr * 2.0)
  ├─ requiredBufSize = maxDelay + bufferSize + 2
  └─ latencyBufOldL/R, latencyBufNewL/R 確保 (aligned_free / makeAlignedArray)

Step 4: 既存 RuntimeWorld が存在すれば publishWorld()
  ├─ resolveActiveRuntimeDSPFromRuntimeWorldOnly()
  ├─ RuntimeBuilder.buildRuntimePublishWorld()
  └─ RuntimePublicationCoordinator.publishWorld()

Step 5: DSP 未存在時 → placeholderDSP 生成・publish
  ├─ DSPCore::prepare()        [コン-vol/EQ/OS/DCBlocker 等確保]
  ├─ setActiveRuntimeDSP(dsp)
  └─ RuntimeBuilder.buildRuntimePublishWorld(HardReset)

Step 6: Convolver/EQ prepareToPlay 同期呼び出し
  ├─ uiConvolverProcessor.prepareToPlay()
  └─ uiEqEditor.prepareToPlay()

Step 7: サンプルレート/ブロックサイズ 変更時
  └─ submitRebuildIntent(Structural)

Step 8: lifecycleState = Prepared  [これ以降 Audio Thread 起動]
```

### 10.3  аудио callback データフロー (Audio Thread、getNextAudioBlock)

```
AudioSource::getNextAudioBlock(bufferToFill)

┌─ 【Step 0: 前処理】────────────────────────────────────────────────────────
│ AudioCallbackRuntimeScope (RAII) 構築
│   ├─ lifecycleRuntime_.enterAudioCallback()
│   ├─ rtCapabilityFirewall_.enter()
│   └─ audioCallbackActiveCount++.fetch_add(1)
│
│ ASSERT_AUDIO_THREAD()
│ 事前チェック: lifecycleState == Prepared, isShutdownInProgress()
│
│ 診断収集 (CONVO_DIAG_SAMPLE_MASK 符合)
│   ├─ CPU番号 ::GetCurrentProcessorNumber()
│   ├─ callback間隔・処理時間測定
│   ├─ publicationSequence 変化検出
│   └─ XRUN 判定 (interval > 1.5x expected OR callback > 1.5x expected)
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 1: RuntimeWorld 読取 (RCU)】─────────────────────────────────────
│ auto runtimeReadHandle = readAudioRuntimeView()
│   └─ makeRuntimeReadHandle(audioCtx)  [audioThreadRcuReader]
│
│ const RuntimePublishWorld* runtimeWorld =
│     getRuntimeWorldFromReadHandle(runtimeReadHandle)
│   └─ RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)
│
│ DSPCore* dsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)
│ DSPCore* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle)
│
│ AudioCallbackAuthorityView authority =
│     makeCrossfadePreparedSnapshotFromWorld(*runtimeWorld)
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 2: パラメータ スナップショット取得】────────────────────────────
│ EngineParameterSnapshot parameterSnapshot =
│     captureAudioThreadParameterSnapshot(runtimeWorld)
│
│ DSPCore::ProcessingState procState =
│     buildAudioThreadProcessingState(dsp, parameterSnapshot)
│
│ procState に含まれるもの:
│   - eqBypassed / convBypassed / ProcessingOrder / AnalyzerSource
│   - softClipEnabled / saturationAmount
│   - inputHeadroomGain / outputMakeupGain / convolverInputTrimGain
│   - convHCMode / convLCMode / eqLPFMode
│   - adaptiveCoeffBankIndex / adaptiveCoeffSet (RCU 経由)
│   - eqParams / eqCache / eqCoeffHash (RCU 経由)
│   - adaptiveCaptureQueue
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 3: クロスフェード ディレイ ゲート】────────────────────────────
│ processCrossfadeDelayGateIfPending(fading, useDryAsOld, preparedCrossfade, [&]{
│     // 旧 DSP (fading) の出力を dspCrossfadeFloatBuffer に溜める
│     fading->process(bufferToFill, analyzerFifo, nullptr, nullptr, fadingState)
│ })
│   └─ 遅延整合後に true を返した場合、早期 return (このブロックは完了)
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 4: 通常処理 OR クロスフェード処理】──────────────────────────────
│
│ ■ クロスフェード可 (canCrossfade = true) の場合:
│   ├─ dspCrossfadeFloatBuffer.clear()
│   ├─ fading->processToBuffer(...) → dspCrossfadeFloatBuffer [旧出力待避]
│   │   (useDryAsOld == true の場合は dry 入力を待避)
│   ├─ dsp->process(bufferToFill, ...)  [新 DSP 処理]
│   └─ runLatencyAlignedCrossfadeMixLoop()
│       └─ 遅延整合後、等電力フェード適用
│       └─ 例: outL[i] = newL[i] * gNew + dryScaledL[i] * (1-gNew)
│       └─ finalizeCrossfadeMixPath(dsp, fading, true)
│
│ ■ 通常処理 (canCrossfade = false) の場合:
│   └─ dsp->process(bufferToFill, analyzerFifo, &inputLevelLinear, &outputLevelLinear, procState)
│       └─ cleanupCrossfadeDirectPath(dsp, fading)
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 5: DSPCore::process() 内部 詳細】───────────────────────────────
│
│ // 10.4 章で詳細説明
│
└──────────────────────────────────────────────────────────────────────────

┌─ 【Step 6: 後処理】──────────────────────────────────────────────────────
│ 診断ログ収集 (CONVOOPEQ_ENABLE_RUNTIME_DIAGNOSTICS)
│   ├─ CBSUMMARY: callback処理時間/間隔最大値を 1秒周期で収集
│   ├─ DSP_STAGE: getCurrentTimeUs() で INPUT/DSP/OUTPUT 区間測定
│   ├─ XRUN/ACTIVATE イベント → xRunBuffer.push()
│   └─ CallbackTimingHistory リングバッファ書込
│
│ AudioCallbackRuntimeScope 解体 (RAII)
│   ├─ audioCallbackActiveCount--.fetch_sub(1)
│   ├─ RTAllocatorFirewall::markRTContext(false)
│   └─ lifecycleRuntime_.leaveAudioCallback(lifecycleToken)
└──────────────────────────────────────────────────────────────────────────
```

### 10.4 DSPCore::process() 内部 データフロー

```
DSPCore::process(bufferToFill, analyzerFifo, inputLevelLinear, outputLevelLinear, state)

Input Stage (processInput):
┌─ 【A】rawInputLinear = processInput(bufferToFill, numSamples, headroomGain, analyzerInputTap, analyzerFifo)
│   ├─ buffer から float を読取、headroomGain でスケーリング
│   ├─ inputLevelLinear に publishAtomic (release)
│   └─ analyzerInputTap が true の場合: analyzerFifo に raw 入力を pushToFifo()
└──────────────────────────────────────────────────────────────────────

Oversampling Stage (processUp):
┌─ 【B】if (oversamplingFactor > 1)
│   ├─ processBlock = oversampling.processUp(originalBlock)
│   │   └─ AVX2 多段 FIR/IIR アップサンプリング
│   ├─ dcBlockers.oversampledL/R.process()  [ハイレート DC ブロッカ]
│   └─ numProcSamples = processBlock.getNumSamples() [元block × OS比]
│   ※maxInternalBlockSize を超える場合は無音出力して return
└──────────────────────────────────────────────────────────────────────

Processing Order Routing:
┌─ 【C】if (order == ConvolverThenEQ)
│
│   Conv-only (if !convBypassed):
│   ┌─ convolverRt().process(processBlock)
│   │   ├─ MKL NUC 分割畳み込み
│   │   └─ 内部で irFinalized == false なら bypass
│   └─ DC除去: 必要に応じて dcBlockers.inputL/R 適用
│
│   EQ (if !eqBypassed):
│   ┌─ eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse)
│   │   └─ TPT SVF 20バンド パラメトリック EQ (AVX2)
│   └─ eqRt().process(processBlock)  [パラメータ未設定時]
│
└──────────────────────────────────────────────────────────────────────

┌─ 【D】if (order == EQThenConvolver)
│
│   EQ (if !eqBypassed):
│   ┌─ eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse)
│   └─ DC除去: 必要に応じて
│
│   Conv (if !convBypassed):
│   ┌─ scaleBlockFallback(ptr, numSamples, convolverInputTrimGain)  [もし gain != 1.0]
│   └─ convolverRt().process(processBlock)
│
└──────────────────────────────────────────────────────────────────────

Output Filter Stage (HCMode/LCMode):
┌─ 【E】if (convActive || eqActive)
│   ├─ convIsLast = (convActive && (!eqActive || order == EQThenConvolver))
│   └─ outputFilter.process(processBlock, convIsLast, convHCMode, convLCMode, eqLPFMode)
│       ├─ ① convIsLast == true: HCフィルター (Butterworth 4次) + LCフィルター (Butterworth 2次 18Hz)
│       └─ ② convIsLast == false: HPF 20Hz + LPFs (Sharp/Natural/Soft × 2段)
└──────────────────────────────────────────────────────────────────────

Makeup Gain:
┌─ 【F】scaleBlockFallback(ptr, numSamples, outputMakeupGain)  [各ch]
│   └─ AVX2 _mm256_mul_pd vGain
└──────────────────────────────────────────────────────────────────────

Soft Clip:
┌─ 【G】if (state.softClipEnabled)
│   ├─ clipThreshold = 0.95 - 0.45 * sat
│   ├─ clipKnee = 0.05 + 0.35 * sat
│   ├─ clipAsymmetry = 0.10 * sat
│   │
│   ├─ if (oversamplingFactor > 1):
│   │   └─ 。元信号 그대로 AVX2 で processBlock に対し softClipBlockAVX2()
│   │       └─ musicalSoftClipScalar(): fastTanh ベースの軟饱和
│   │
│   └─ else:
│       ├─ softClipOS.prepareSingleStage()  [2倍 OS のみ]
│       ├─ osBlock = softClipOS.processUp(originalBlock)
│       ├─ softClipBlockAVX2(osBlock)
│       └─ softClipOS.processDown(osBlock, originalBlock)
└──────────────────────────────────────────────────────────────────────

Oversampling Downsample:
┌─ 【H】if (oversamplingFactor > 1)
│   ├─ oversampling.processDown(processBlock, originalBlock)
│   └─ processBlock = originalBlock  [元ブロック参照に戻す]
└──────────────────────────────────────────────────────────────────────

Analyzer FIFO (Output tap):
┌─ 【I】if (analyzerEnabled && analyzerSource == Output)
│   └─ pushToFifo(processBlock, analyzerFifo)  [LockFreeAudioRingBuffer]
└──────────────────────────────────────────────────────────────────────

Output Level:
┌─ 【J】outputLinear = measureLevel(originalBlock)
│   └─ outputLevelLinear に publishAtomic (release)
└──────────────────────────────────────────────────────────────────────

Output Stage (processOutput):
┌─ 【K】processOutput(bufferToFill, numSamples, state)
│   ├─ DC除去: dcBlockers.outputL/R 適用
│   ├─ 固定レイテンシ ディレイ適用 (applyFixedLatencyDelay)
│   ├─ applyGainRamp (FADE_IN_SAMPLES = 2048 でフェードイン)
│   └─ 結果を書込: buffer->getWritePointer(ch, startSample)
└──────────────────────────────────────────────────────────────────────
```

### 10.5 EQ 処理 (EQProcessor::process) 詳細

```
EQProcessor::process(block, [eqParams, eqCache])

Audio Thread 入口。**ロックなし・メモリ確保なし・libm呼び出しなし (fastTanh 使用)**

Step 1: バイパス・サイレント チェック
  ├─ m_rtBypassShadow = bypassRequested (atomic) を shadow にコピー
  ├─ rtBypassedShadow = m_rtBypassShadow
  └─ isAudioBlockSilent() → 全バンド無効・バイパス時高速パス

Step 2: 係数キャッシュ lookup (RCU)
  ├─ paramsHash = computeParamsHash(eqParams)
  ├─ if (coeffCache && cache->paramsHash == paramsHash && cache->sampleRate == sr)
  │   └─ coeffs = cache->coeffs (事前計算済み SVF 係数)
  └─ else:
      ├─ EQCoeffCache::createCoeffCache(eqParams, sr, maxBlockSize, generation)
      └─ 各バンド: calcSVFCoeffs(bandType, freq, gain, q, sr)
          ├─ LowShelf:  係数 g,k,a1,a2,a3,m0,m1,m2 計算
          ├─ Peaking:   同上
          ├─ HighShelf: 同上
          ├─ LowPass:   同上
          └─ HighPass:  同上

Step 3: bands 状态リセット (bandResetPacked)
  ├─ rtDeferredBandResetMask = bandResetMaskFromPacked(bandResetPacked)
  ├─ rtSeenBandResetSerial = bandResetSerialFromPacked(bandResetPacked)
  └─ 対象バンドの filterState[ch][band][0/1] を {0,0} にリセット

Step 4: M/S 処理分岐 (ChannelMode による)
  ├─ L/R → Mid/Side 変換 (msWorkBuffer 使用)
  ├─ Mid = (L + R) * 0.5, Side = (L - R) * 0.5
  └─ filterState は kFilterChannels=4 (L/R/Mid/Side) で独立保持

Step 5: シリアル/パラレル構造分岐

  【Serial (DEFAULT)】
  ┌─ for ch in {0,1}:
  │   for band in 0..19:
  │       if (!bandActive[band]) continue;
  │       if (bandChannelMode != Stereo && bandChannelMode != ch) continue;
  │       auto& coeffs = (channelMode==Mid) ? cache->coeffs[band] : ...
  │       processBandStereo() / processBand()
  │           ├─ TPT SVF 計算 (ic1eq, ic2eq 状態変数更新)
  │           ├─ saturation 適用 (fastTanhScalarOutput)
  │           ├─ NaN/Inf クランプ (-100..+100)
  │           └─ denormal フラッシュ
  │
  │   // バンド間 Serial 接続: 各バンドの出力が次バンドの入力に
  │
  └─ 最後: totalGainTarget (linear) で全局スケール

  【Parallel】
  ┌─ for ch in {0,1}:
  │   // 全バンド並列処理 → parallelAccumBuffer に累積
  │   for band in 0..19:
  │       if (!bandActive) continue;
  │       processBand() → parallelWorkBuffer
  │       for i in 0..numSamples: parallelAccumBuffer[i] += parallelWorkBuffer[i]
  │   // Input copy → parallelInputBuffer
  │   for i: parallelOutput[i] = parallelInputBuffer[i] + parallelAccumBuffer[i] * totalGain
  └─ structureXfadeBufferCapacity でシリアル/パラレル クロスフェード可能

Step 6: AGC 適用 (processAGC)
  ├─ RMS 計算 (AVX2使用)
  ├─ エンベロープ追踪 (attackCoeff / releaseCoeff)
  ├─ 目標ゲイン計算 → agcCurrentGain 更新
  └─ 全出力に乗算 (Smooth 係数で補間)

Step 7: Nonlinear Saturation 適用
  ├─ fastTanhScalarOutput / fastTanhV128Output 適用
  └─ Saturation パラメータ (0..1) で線形/非線形ブレンディング

Step 8: 出力クランプ & 書込
  ├─ NaN/Inf クランプ (-100..+100)
  └─ block にコピー (parallelOutput → 元block)

### 10.6 Convolver 処理 (MKL NUC) 詳細

```
ConvolverProcessor::process(block)  [Audio Thread]

Step 1: irFinalized チェック (atomic, acquire)
  ├─ if (!irFinalized) {
  │     // IR 未確定 (ロード中/最適化中)
  │     if (hasValidPreviousEngine()) {
  │         processBypassWithLatencyCompensation()  // 遅延整合のみ
  │     }
  │     return;
  │ }

Step 2: レイテンシ バッファ書込 (DELAY_BUFFER_SIZE = 2^22)
  ├─ writePos を更新し、入力を delayBuffer[ch] に循環書込

Step 3: activeEngine.load(acquire) → StereoConvolver*
  ├─ RCU 保護ポインタ、if (conv==nullptr) → bypass

Step 4: Partitioned Convolution (MKL)
  ├─ 各パーティション: FFT → 複素乗算 (input_fft * filter_fft) → IFFT
  └─ waitTime ベースの時刻整列

Step 5: Latency Reset 検出
  ├─ latencyChangeRequestedGen 変化で pendingLatencyValue 更新
  └─ latencyWritePos = 0 (バッファ リセット)

Step 6: バイパス時
  ├─ processBypassWithLatencyCompensation()
  │   ├─ 入力を delayBuffer に保存
  │   ├─ 遅延整合後、dry 信号を返す
  │   └─ crossfadeGain 適用
  └─ return

Step 7: 出力の後処理
  ├─ delayBuffer に保存 (遅延補償用)
  ├─ convolverInputTrimGain 適用
  └─ DC除去: UltraHighRateDCBlocker (if irFinalized)
```

### 10.7 IR ロード パイプライン (Message Thread → Worker Thread)

```
UI: ConvolverControlPanel → setImpulseResponse(file)

Step 1: setImpulseResponse() [Message Thread]
  ├─ pendingOverride = BuildSnapshot { irFile, mix, phaseMode, ... }
  └─ submitRebuildIntent(Structural)

Step 2: rebuildThreadLoop() → RebuildTask 実行 [Dedicated Thread]
  ├─ loadImpulseResponsePreview() → IRLoadPreview (長さ検出)
  ├─ 位相モード適用 (AsIs / Mixed / Minimum)
  │   ├─ Mixed: AllpassDesigner で混合位相分解
  │   └─ Minimum: convertToMinimumPhase()
  ├─ r8b::CDSPResampler24IR でターゲット SR にリサンプリング
  │   └─ R8B_IPP=0 → 組込 PFFFT 使用
  ├─ IRConverter::convert() — フェード適用
  ├─ AutoGain 計算 (dBFS RMS)
  └─ loadImpulseResponseComplete()

Step 3: MKLNonUniformConvolver::init() [Worker Thread]
  ├─ パーティション分割サイズ計算
  ├─ filter の FFT Plans 生成 (MKL DFTI)
  ├─ filterSpectrum[partition] 事前計算
  └─ irFinalized = true, latency 更新

Step 4: DSPCore 更新 [Message Thread]
  ├─ DSPCore::prepare() 再呼び出し
  └─ publishWorld() → RuntimeWorld 更新
```

### 10.8 ISR Runtime Publication パイプライン

```
【Non-RT パス】

submitRebuildIntent(kind, ...)
  └─ RebuildDispatch に enqueueCommand()

rebuildThreadLoop() → dequeueCommand()
  ├─ DSPCore* newDSP = buildNewDSP(task)
  └─ enqueuePublicationIntentForRuntimeCommit(newDSP, gen, sealedSnapshot)

enqueuePublicationIntentForRuntimeCommit(newDSP, gen, sealedSnapshot)
  ├─ DSPHandle handle = registerDSPHandleForRuntime(newDSP)
  ├─ PublicationAdmission::PublishRequest req { handle, gen, sealedSnapshot }
  └─ runtimeOrchestrator_->submitPublishRequest(req)

RuntimePublicationOrchestrator::submitPublishRequest(req)
  ├─ admission.evaluate(healthState, req) → Accepted / Rejected
  └─ if (Accepted) → キューに追加

orchestrator.tick() [Timer Thread 100ms]
  ├─ RuntimeBuilder.buildRuntimePublishWorld()
  │   ├─ validateSemanticCompleteness()
  │   ├─ validateRuntimeGraphAuthorityContract()
  │   ├─ world.generation = runtimeGenerationGenerator_.next()
  │   ├─ world.publicationSequence = publicationSequenceCounter_++
  │   └─ world.isFrozen() = true
  └─ coordinator.publishWorld(std::move(worldOwner))

RuntimePublicationCoordinator::publishWorld(worldOwner)
  ├─ precheckRuntimePublication(closure, descriptor)
  ├─ onRuntimePublishedNonRt(world)
  │   ├─ worldLifecycleAudit_.onWorldPublished()
  │   ├─ runtimePublicationBridge_.commit()
  │   ├─ Atomic: lastCommittedRuntimeGeneration_ = world.generation
  │   └─ emitEvidenceTickNonRt()
  └─ bridge_.commit() → RuntimeStore に publish + Retire 通知

【RT パス: Audio Thread — 読み出しのみ】

readAudioRuntimeView()
  ├─ audioThreadRcuReader.enter()
  └─ makeRuntimeReadHandle(audioCtx)

getRuntimeWorldFromReadHandle(handle)
  └─ RuntimePublicationCoordinator::consumeWorldHandle(runtimeStore)
      └─ store.observe() → RuntimeState* (atomic 読み取り)

resolveActiveRuntimeDSPFromRuntimeWorldOnly(handle)
  ├─ world->graph.activeNode->dspHandle
  └─ dspHandleRuntime_.getDSP(activeNodeId)
```

### 10.9 Retire / Reclaim パイプライン

```
World 退役時 (onRuntimeRetiredNonRt):
  ├─ RetireIntent { dspSlot, generation, retireEpoch }
  ├─ retireRuntime_.emitRetireIntentRT(intent)
  ├─ ISRRetireRouter::enqueueRetire(slot, epoch)
  │   ├─ ISRRetireRouter::publishEpoch()
  │   └─ ISRRetireRouter::tryReclaim()
  │
  ├─ retireRuntimeEx_.enqueueRetire(slot)
  │   ├─ RetireLane 分類 (Active / Pending / Free / Quarantine)
  │   └─ exceedDeferralThresholds → quarantineSlot()
  │
  └─ quarantineSlot(slot, gen, reason)
      ├─ dspQuarantineManager_.quarantine(slot)
      ├─ dspHandleRuntime_.quarantine(slot)
      └─ DeferredDeletionQueue に登録

DeferredDeletionQueue (Worker Thread):
  ├─ DeferredFreeThread がバックグラウンド実行
  ├─ EpochDomain::globalEpoch が進捗 → 該当 epoch 資源を解放
  └─ aligned_free() / delete で実体解放

Grace Period (ISRRetireRuntimeEx):
  ├─ isGracePeriodCompleted(pendingGen, maxObservedGen, callbackCount)
  │   └─ (maxObservedGen > pendingGen) || (callbackCount == 0)
  └─ canReclaimAfterEscalation(noReader, noExecutorRef, noPendingTransition)
```

### 10.10 シャットダウン シーケンス

```
~AudioEngine()
  ├─ setShutdownPhase(StopAcceptingWork)
  ├─ lifecycleState = Releasing
  ├─ runtimePublicationBridge_.requestShutdown()
  │
  ├─ setShutdownPhase(StopAudio) → stopTimer()
  ├─ setShutdownPhase(StopWorkers) → stopRebuildThread()
  │
  ├─ DSPLifetimeManager::retire() × active/fading
  ├─ shutdownWorkerThread()
  ├─ setShutdownPhase(ForceEpochAdvance) → m_retire_router->publishEpoch()
  │
  ├─ setShutdownPhase(DrainRetire) [最大 5秒待機]
  │   └─ while (pendingRetireCount>0 || activeReaderCount>0)
  │         m_retire_router->publishEpoch() / tryReclaim()
  │
  ├─ runtimePublicationCoordinator.requestShutdownClearNonRt()
  ├─ runtimePublicationBridge_.markShutdownComplete()
  ├─ drainDeferredRetireQueues(true)
  ├─ m_epochDomain.drainAll()
  └─ lifecycleState = Destroyed
```

### 10.11 Noise Shaper Learner 適応フロー

```
UI: startNoiseShaperLearning(mode, resume)  [Message Thread]
  ├─ LearningCommand { Start, resume, mode, irGeneration }
  ├─ enqueueLearningCommand(cmd)
  └─ submitRebuildIntent(DeferredStructural)

Audio Thread: processLearningCommands()
  ├─ dequeue cmd → learningCommandBuffer
  ├─ cmd.type == Start: adaptiveCaptureActiveRt=true, state=WaitingForDSP
  ├─ cmd.type == DSPReady: state=Running → learner->start()
  └─ cmd.type == Stop: adaptiveCaptureActiveRt=false, state=Idle

Audio Thread: DSPCore::process 内 Adaptive Capture
  └─ adaptiveCaptureQueue->push(AudioBlock{L,R,numSamples,sr,bitDepth,bankIdx,sessionId})

Worker Thread: NoiseShaperLearner
  ├─ AdaptiveCaptureReader が pop
  ├─ CMA-ES 最適化 (9次 IIR係数 k[9])
  ├─ 完了 → storeLearnedCoeffs(coeffs)
  │   ├─ CoeffSet bank[bankIndex] に保存
  │   ├─ adaptiveCoeffGeneration++ (atomic)
  │   └─ submitRebuildIntent(DeferredFinalizeAware)
```

---

## 11. ISR Runtime ガバナンス 詳細

### 11.1 フィールド権威クラス

| Authority | Field | Visibility |
|---|---|---|
| Authoritative | generation, topology, routing, execution, publication, overlap, metadata, retire, timing, latency | PublicationBoundary |
| Derived | generationSemantic, graph, engine, resource, automation, coefficient, dspProjection | ObserveBoundary |
| Diagnostic | worldId, affinity, projectionFreshness, semanticHash | DiagnosticBoundary |

### 11.2 Semantic Transaction State

```
Idle → Building → Validated → Committed → Published
              ↘ Rejected ↗
```

### 11.3 Crossfade Prepared Snapshot

```
makeCrossfadePreparedSnapshotFromWorld():
  latencyDelayOld/New, fadeTimeSec, startDelayBlocks,
  dryHoldSamples, useDryAsOld, firstIrDryCrossfadePending
```

---

## 12. メモリ / スレッド管理まとめ

### 12.1 アロケータ

| アロケータ | 用途 | 整列 |
|---|---|---|
| `convo::aligned_malloc(size, 64)` | オーディオバッファ (SIMD) | 64B |
| `convo::makeAlignedArray<T>(n)` | `ScopedAlignedPtr` 経由 | 64B |
| `MKL DftiMalloc` | FFT ワークスペース | 64B |
| `r8b::PFFFT` | r8brain 組込 FFT | 16B |

### 12.2 Lock-Free 構造

| 構造 | 種類 | 用途 |
|---|---|---|
| `LockFreeAudioRingBuffer<AudioBlock, 4096>` | SPSC | 解析 FIFO |
| `LockFreeRingBuffer<XRunEvent, ...>` | SPSC | XRUN イベント |
| `LockFreeRingBuffer<DiagEvent, 512>` | SPSC | 診断イベント |
| `EpochDomain::readers[64]` | RCU reader slot | 64スレッド対応 |

### 12.3 スレッド別責務

| スレッド | 責任 |
|---|---|
| **Message Thread** | UI イベント、パラメータ変更、prepareToPlay |
| **Audio Thread** | getNextAudioBlock / DSPCore::process 全 DSP |
| **Timer Thread (100ms)** | rebuild dispatch、テレメトリ、健康監視 |
| **Worker Thread** | IR ロード、rebuild 実行 |
| **DeferredFree Thread** | 非同期 delete/reclaim |
| **NoiseShaperLearner Thread** | CMA-ES 最適化 |

---

## 13. テスト階層

| テスト | 内容 |
|---|---|
| ISRRuntimeIdentityTests | UUID/Generation 生成器 |
| RuntimePublicationCoordinatorTests | Coordinator テンプレート |
| ISRSemanticValidationTests | 意味検証 (スキーマ v9) |
| RetireGraceSemanticsTests | Grace period セマンティクス |
| RuntimeSemanticSchemaValidationTests | フィールド/権威不変式 |
| RuntimeWorldAuthorityProjectionTests | World 射影不変式 |
| PartialPublicationRejectTests | 部分公開却下 (MKL 依存) |
| BuildInputSemanticContractTests | Build 入力契約 |
| PriorityIntegrationTests | 優先度統合 (RetireRuntime) |
| PublicationValidatorIsolationTests | Validator 隔離 |
| HeadlessAudioPathVerification.ps1 | CI ヘッドレステスト |

---

## 14. 主要定数一覧

```cpp
kAdaptiveNoiseShaperOrder         = 9
kAdaptiveNoiseShaperSampleRateBankCount = 10
kNumAdaptiveCoeffBanks            = 10*3*6 = 180
FIFO_SIZE                         = 1048576
SAFE_MAX_BLOCK_SIZE                = 65536
MAX_IR_LATENCY                    = 2097152  // 2^21
DELAY_BUFFER_SIZE                 = 4194304  // 2^22
NUM_BANDS                        = 20
kFilterChannels                  = 4   // L,R,Mid,Side
DiagRuntimeLimits::BufferCapacity = 512
RTLocalState::kPublishTimingSlots = 16
RTLocalState::kCallbackTimingSlots = 32
```

---

*生成日: 2026-07-03*
*ソース解析対象: ConvoPeq v0.6.8 — JUCE 8.0.12 / Intel oneMKL/IPP / AVX2 / C++20 / Windows 11 x64*

---

## 15. Runtime Orchestrator 内部詳細

### 15.1 Orchestrator ポリシー パイプライン

```
RuntimePublicationOrchestrator 所有コンポーネント:
  ├─ RuntimePublicationStateOwner stateOwner_
  │     └─ submitted → built → activated → retired 順序管理
  ├─ TelemetryRecorder telemetryRecorder_ (進捗・失敗記録)
  ├─ PublicationAdmission admission_ (HealthState参照 → Accepted/Deferred/Rejected)
  ├─ PublicationExecutor executor_ (Coordinator への最終 commit)
  ├─ DSPTransition transition_ (active slot切替・crossfade登録)
  └─ DSPLifetimeManager lifetime_ (DSPCore 退役)

trySubmit(req) パイプライン:
  Phase 1: admission_.evaluate(req, engine_, pubCtx)
    ├─ Rejected → return (caller 処理)
    └─ Accepted → Phase 2 続行

  Phase 2a: RuntimeBuilder.buildRuntimePublishWorld(newDSP, oldDSP)
    ├─ CrossfadeAuthority.evaluate(oldWorld, newWorld, policy)
    └─ executor_.executePendingPublish() → Coordinator.commit()

  Phase 2b: DSPTransition (クロスフェード後の activate)
    ├─ transition_.activate(newDSP) → activeRuntimeDSPSlot更新
    └─ クロスフェード arm → crossfadeRuntime_.getGain().setTargetValue(1.0)

Deferred Publish:
  ├─ deferredSlot_ に PublishRequest 保存
  └─ kDeferredPublishTTLUs = 30秒 → notifyTransitionComplete() で re-evaluate

Publication Stalling 監視:
  ├─ isPublicationStalled(): elapsed >= 30s
  └─ HealthState → Critical
```

---

## 16. RuntimeHealthMonitor 監視体系

### 16.1 監視カテゴリ別イベントコード

| コード | 名称 | Severity | 閾値 |
|---|---|---|---|
| 1001 | RETIRE_STALL | Error | pendingRetire > 0 × 5 tick |
| 1002 | RETIRE_STALL_WARNING | Warning | pendingRetire > 0 × 1 tick |
| 2001 | PUBLICATION_STALL | Error | isPublicationStalled() |
| 2002 | PUBLICATION_WARNING | Warning | deferredAge > 15s |
| 3001 | READER_STUCK | Error | readerDepth > 0 × 30s |
| 3010 | READER_SLOT_USAGE | Warning | activeReaders > 75% |
| 4001 | CROSSFADE_TIMEOUT | Error | fade active > 30s |
| 4002 | CROSSFADE_EVENT_DROP | Error | diff > 10 |
| 4003 | CROSSFADE_ABORTED_EMERGENCY | Error | Emergency Override |
| 5001 | LEARNER_BACKPRESSURE_WARNING | Warning | FIFO > 85% |
| 5002 | LEARNER_BACKPRESSURE_ERROR | Error | FIFO > 95% |
| 6000 | VALIDATION_FAILURE | Error | validator reject |
| 1009 | RETIRE_AGE_NORMAL | Info | (正常復帰通知) |
| 1010 | RETIRE_AGE_WARNING | Warning | 5s age |
| 1011 | RETIRE_AGE_CRITICAL | Error | 30s age |

### 16.2 HealthState 3状態 と Critical Exit Blocker

```
HealthState:
  Healthy ──(stall/drop/stuck)──→ Degraded ──(継続)──→ Critical

CriticalExitCondition:
  ├─ allMonitorsNormal
  ├─ suppressionInactive (admission strict非発動中)
  ├─ noRecoveryActionRunning (PolicyEngine 恢復未完了)
  └─ stableDuration (30s以上正常後)
CriticalExitBlocker: MonitorNotNormal / SuppressionActive / RecoveryRunning /
                     StableDurationInsufficient / PendingRetireExceeded / RetireAgeExceeded
```

---

## 17. RuntimePolicyEngine 回復行動階層

### 17.1 RecoveryAction 6段

```
Level 0: Observe    → HealthEvent 記録のみ
Level 1: Throttle   → admissionStrict, PauseLearner
Level 2: Recover    → ForceRetireDrain, ForceSnapshotPublish
Level 3: Restore    → RollbackToLastHealthyWorld, LearnerRollback, CheckpointRestore
                      └─ Phase: EpochRecoveryIssued → LearnerRollbackDone → IdleWorldPublished
Level 4: Safe       → SoftSafeMode (ConvByPass+LearnerStop), HardSafeMode (1xOS+FlatEQ)
Level 5: Critical   → RejectNewPublication, EmergencyDrain, Shutdown
```

### 17.2 PolicySource 分類 (統合版)

```
PolicySource: RetireStall / PublicationStall / ReaderStuck / CrossfadeTimeout /
              LearnerAnomaly / WorldConsistency / AudioOutputAnomaly /
              EmergencyCondition / RecoveryOutcome / SafeModeState
              計 10カテゴリ (HealthMonitor 全イベントを網羅)
```

---

## 18. CustomInputOversampler 内部構造

```
prepare(maxInputBlockSize, ratio, preset):
  ├─ sanitizeRatio → 2x / 4x / 8x
  ├─ ステージ数 = log2(ratio)
  │    IIRLike:  taps {511, 127, 31}, atten {140, 110, 90} dB
  │    LinearPhase: taps {1023, 255, 63}, atten {160, 140, 120} dB
  ├─ convCoeffs: BesselI0-based alt-halfband FIR (meets linear phase spec)
  └─ workCapacity = maxUpsampledBlockSize (64B aligned)

processUp(inputBlock, numChannels):
  ├─ AVX2 FMA upsample: upHistory + convCoeffs → inline FMA
  └─ corruptionChecker: NaN/Inf → auto clear (3 consecutive) / hard fallback (4+ consecutive)

processDown(upsampledBlock, outputBlock):
  ├─ 逆順に downHistory × coeffsReversed
  └─ 最終ダウンサンプルブロックをoutputBlockに戻す

prepareSingleStage(taps, attenDb, stageInputMax):
  ├─ 局所2xOS (SoftClip専用)
  └─ coeffs/係数のみ設計、historyは Audio Blockサイズでalloc
```

---

## 19. SpectrumAnalyzerComponent 描画パイプライン

```
timerCallback (~60fps):
  ├─ engine.readFromFifo(buf, NUM_FFT_POINTS=4096) [LockFreeAudioRingBuffer]
  ├─ MKL DFTI FFT: TimeDomain → FrequencyDomain (NUM_FFT_BINS = 2049)
  ├─ rawBuffer[i] = 20*log10(mag) [FFT_MAGNITUDE_SCALE]
  ├─ smoothedBuffer[i] = α*raw + (1-α)*smoothed (α=0.15 new, 85% old retain)
  ├─ peakBuffer[i] = max(raw, peak) + 1秒 hold → decay
  ├─ updateEQData() → engine.calcEQResponseCurve(magnitudes, zCache, 128 point, fnEQCoeffBiquad)
  │     └─ individualBandCurves L/R/Mid/Side 別計算
  └─ paint(): Grid → Spectrum (green→yellow→red) → Peak hold (white) → EQ curve (white)
```

---

## 20. Convolver Mixed Phase 内部詳細

```
convertToMixedPhase(owner, fileHash, linearIR, minimumIR, sr, f1, f2, tau):
  概念: 低域 = linearIR phase, 高域 = minimumIR phase, 遷移帯 = allpass filter分解

  Step 1: キャッシュ lookup (memory + disk)
  Step 2: convertToMixedPhaseAllpass()
    ├─ AllpassDesigner: targetGroupDelay = linPhase - minPhase
    │     └─ SecondOrderAllpass × 16 sections × CMA-ES/AdaGrad最適化
    ├─ success → state "Completed"
    └─ fail → step 3 fallback
  Step 3: convertToMixedPhaseFallback()
    └─ IIR x-over + linearPhase (FFT IFFT phase zeroing)
  Step 4: キャッシュ保存
```

---

## 21. ISR 10層 Architecture Layer 詳細 Map

```
Layer 1    RuntimeGraph                  src/core/RuntimeGraph.h
Layer 2    RuntimeState                  AudioEngine.h L133-299 (21 field descriptors)
Layer 3    RuntimeBuilder                src/audioengine/RuntimeBuilder.h (28KB, only constructor)
Layer 4   RuntimePublicationCoordinator src/core/RuntimePublicationCoordinator.h (template<World,Handle,Bridge>)
Layer 5   RuntimePublicationValidator    src/audioengine/RuntimePublicationValidator.h
Layer 6   PublicationAdmission         PublicationAdmission.h (HealthState依存)
Layer 7   CrossfadeAuthority           CrossfadeAuthority.h (dspProjection参照)
Layer 8   ISRShutdown                  ISRShutdown.h (10-state FSM)
Layer 9   ISRRetire / ISRRetireRouter / ISRRetireRuntimeEx / ISRRetireOverflowRing
Layer 10  RuntimeHealthMonitor + TelemetryRecorder + RuntimePolicyEngine

Cross-layer: RuntimePublicationBridge
  ├─ commit(PublishAuthority, RuntimeBoundary, newWorld, ver, seq, epoch, mappedGen)
  ├─ retire(RetireAuthority, RuntimeBoundary, oldWorld)
  └─ NonRTWorld のみ受入 (RTWorld → Faulted)
```

---

## 22. AudioEngine DSPCore サブ State 構造体

```cpp
DSPCore:
    ConvolverProcessor convolver;  // MKL NUC
    EQProcessor eq;               // 20-band TPT SVF
    PsychoacousticDither dither; FixedNoiseShaper fixedNoiseShaper;
    Fixed15TapNoiseShaper fixed15TapNoiseShaper;
    LatticeNoiseShaper adaptiveNoiseShaper;
    OutputFilter outputFilter; TruePeakDetector truePeakDetector;
    LoudnessMeter loudnessMeter;

    CustomInputOversampler oversampling; CustomInputOversampler softClipOS;
    DCBlockerRuntimeState* dcBlockerState: { UltraHighRateDCBlocker ×6 }
    ConvolverRuntimeState* convolverState; EQRuntimeState* eqState;
    RampRuntimeState* rampState: { LinearRamp bypassFadeGainDouble, fadeInSamplesLeft }
    HistoryRuntimeState* historyState: { fixedLatencyBuffer L/R, softClipPrevSample[2] }
```

---

## 23. Processing Order 経路分岐表

| order | convBypass | eqBypass | DSP 経路 |
|---|---|---|---|
| ConvolverThenEQ | 両方 false | — | conv→eq(eqParams,eqCache) |
| ConvolverThenEQ | false | true | conv→eq(passthrough) |
| ConvolverThenEQ | true | false | bypass conv→eq(eqParams,eqCache) |
| EQThenConvolver | false | false | eq(eqParams,eqCache)→convTrimGain→conv |
| EQThenConvolver | true | — | eq(passthrough)→conv (bypass) |

共通: outputFilter 常に適用、makeup gain 適用、softClip 適用 (saturation amount 依存)

---

## 24. ConvoPeq パラメータ 種別 全一覧

### AudioEngine スレッド間伝達パラメータ (atomic)
- `manualOversamplingFactor` (int) / `ditherBitDepth` (int)
- `currentProcessingOrder` (ProcessingOrder) / `currentAnalyzerSource` (AnalyzerSource)
- `analyzerEnabled` / `eqBypassRequested` / `convBypassRequested` / `softClipEnabled` (bool)
- `saturationAmount` (float)
- `inputHeadroomGain` / `outputMakeupGain` / `convolverInputTrimGain` (double)
- `convHCMode` (HCMode) / `convLCMode` (LCMode) / `eqLPFMode` (HCMode)
- `m_irFadeTimeSec` / `irLengthFadeTimeSec` / `phaseFadeTimeSec` / `directHeadFadeTimeSec` / `nucFilterFadeTimeSec` / `tailFadeTimeSec` / `osFadeTimeSec` (double)
- `m_crossfadeStartDelayBlocks` (int) / `m_irFadeSamples` / `m_eqFadeSamples` (int)
- `m_pendingIRChange` (bool) / `noiseShaperType` (NoiseShaperType)

### Convolver Build パラメータ (BuildSnapshot)
- mix / bypassed / phaseMode / resamplingPhaseMode / smoothingTimeSec / targetIRLengthSec / autoDetectedIRLengthSec / irLengthManualOverride / mixedTransitionStartHz/EndHz / rebuildDebounceMs / experimentalDirectHeadEnabled / tailMode / tailStartSec / tailStrength / tailL1L2Multiplier / targetUpgradeFFTSize / maxCacheEntries / nucHCMode / nucLCMode

### EQ パラメータ (EQState, 20バンド)
- EQBandParams[20]: { frequency(Hz), gain(dB), Q, enabled }
- EQBandType[20]: LowShelf/Peaking/HighShelf/LowPass/HighPass
- EQChannelMode[20]: Stereo/Left/Right/Mid/Side
- totalGainDb / agcEnabled / nonlinearSaturation (0..1)
- FilterStructure (Serial / Parallel)

### NoiseShaperLearner パラメータ
- NoiseShaperLearningMode { Short, Medium, Long, Broadcast, Tonal, Custom }
- NoiseShaperLearnerSettings: { cmaesRestarts, coeffSafetyMargin, enableStabilityCheck }

---

## 25. 宣言的最大規模チューニング値

```cpp
AudioEngine:            FIFO_SIZE = 1048576 (1<<20) / SAFE_MAX_BLOCK_SIZE = 65536
latency crossfade:      MAX_LATENCY_ALIGN_SAMPLES = 192000 (2s@48kHz)
Convolver:              DELAY_BUFFER_SIZE = 4194304 (1<<22) / MAX_IR_LATENCY = 2097152 (1<<21)
                         MAX_BLOCK_SIZE = 524288 (1<<19) / MAX_TOTAL_DELAY ≈ 2.6M
Diagnostics:            kDiagEventSizeMax = 96 bytes / BufferCapacity = 512 (≈ 49KB)
                         MaxDrainPerTick = 64 (Timer thread per-cycle)
Crossfade:              kMaxLatencySamples = 1536000 (2s@768kHz)
```

---

*生成日: 2026-07-03*
*解析対象: ConvoPeq v0.6.8 — JUCE 8.0.12 / Intel oneMKL+IPP / MSVC 19.44+ icx 2026.0 / AVX2 / C++20 / Windows 11 x64*







