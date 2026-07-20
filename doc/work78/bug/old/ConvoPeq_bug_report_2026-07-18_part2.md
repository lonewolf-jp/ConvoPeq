# ConvoPeq ソースコード調査報告書（Part 2）

Part 1（確定バグ2件 + パッチ）の続きです。本パートは「疑わしいが検証の結果問題なしと判断した箇所」と「調査範囲マップ」です。他のAIへ引き継ぐ際、ここに記載した項目は再調査不要です。

---

## 1. 検証したが「問題なし」と判断した箇所

静的解析の仮説を実コードで裏取りし、誤検知と判明したもの／設計として妥当と確認できたものです。（メモリの「検証重視」方針に基づき、根拠のない指摘をしないためにここに明記します。）

### 1.1 メモリ削減パッチ（AoS/SoA）は適用済みで健全

`MKLNonUniformConvolver::Layer` の `irFreqDomain`/`fdlBuf`（AoS、スクラッチ用途）と `irFreqReal/Imag`・`fdlReal/Imag`（SoA、永続データ）を確認。以前のセッションで指摘・修正された「AoSとSoAの二重保持」は解消済みで、AoS側は `partStride`（1パーティション分）/`2*partStride`（2スロット分）のスクラッチサイズに縮小されています（`[Mem-Fix]` コメントあり）。再発なし。

### 1.2 `applySpectrumFilter` とAudio Threadの競合

`applySpectrumFilter()`（周波数ゲイン適用、`irFreqReal/Imag`を直接書き換え）は `SetImpulse()` 内からのみ呼ばれており（`grep`で他の呼び出し元がないことを確認）、`SetImpulse()` は冒頭で `m_ready=false`、末尾で `m_ready=true` を発行するブロックの内側に収まっています。Audio Thread（`Add()`/`Get()`）は `m_ready` を確認してから当該バッファへアクセスするため、単体では競合しません。単独で外部から呼べる構造であれば重大なデータ競合でしたが、実際にはそうなっていません。

### 1.3 `m_ready` フラグとRCU設計

`SetImpulse()` は既存の「新規DSPCoreをバックグラウンドで構築→ウォームアップ→RCU publish」というアーキテクチャ（メモリに記載の active/fading dual DSPCore パターン）の中で、**未公開（audio threadからまだ参照されていない）インスタンス**に対してのみ呼ばれる設計と判断しました。ライブ中の変換器を直接書き換える経路は見当たりません。

### 1.4 ロックフリー基盤（`LockFreeRingBuffer.h`, `LockFreeAudioRingBuffer.h`, `AtomicAccess.h`）

SPSC前提のインデックス管理・メモリオーダー（読み手はacquire、書き手はrelease）を確認し、教科書的に正しい実装でした。指摘事項なし。

### 1.5 `IppFFTPlanCache`（`MKLNonUniformConvolver.cpp`内）のmutex使用

`std::lock_guard<std::mutex>` を使用していますが、直前に `ASSERT_NON_RT_THREAD()` があり、Audio Threadからの誤用はデバッグビルドで検出される設計。使用箇所（`getOrCreate`）もIR設定時のプラン取得用途のみで、Audio Thread規約違反ではありません。

### 1.6 `CacheManager.cpp` の `mkl_malloc`/`mkl_free` 非対称（誤検知）

`copyFromMmapToAligned()`/`loadPreparedState()` で `DIAG_MKL_MALLOC` により確保した `double*` を `PreparedIRState::partitionData` に格納しますが、同ファイル内に対応する解放コードが見当たらず、一見リークに見えます。実際には `PreparedIRState`（`PreparedIRState.h`）が `~PreparedIRState()` 内で `convo::aligned_free(partitionData)` を呼んでおり、リークではありません（関数名が異なるため機械的な grep 突合では見逃す典型例でした）。

### 1.7 `MKLNonUniformConvolver.h` の `mkl_malloc` 大量言及（誤検知）

ヘッダ内の16箇所の `mkl_malloc(...)` はすべてメンバ変数宣言に添えられた**説明コメント**であり、実際の確保コードではありません（実装は `.cpp` 側）。

### 1.8 Bypassクロスフェード（線形）とMixクロスフェード（等電力）の相違

`bypassFadeGainDouble/Float` によるwet/dry合成は線形補間（`gWet + gDry(=1-gWet)`）、一方でMixノブ等は `equalPowerSin` による等電力補間を使用しており、一見メモリに記載の「クロスフェードは等電力補間必須」という学習内容と矛盾するように見えました。

調査の結果、Bypassフェードは `BYPASS_FADE_TIME_SEC = 0.005`（5ms固定）の**瞬間的な遷移**専用であり、任意の値で長時間保持されうるMixノブとは用途が異なります。5ms程度の遷移では等電力/線形の違いによる−3dBディップは実用上問題にならず、クリックノイズ回避が主目的である短時間Bypassフェードでは線形補間で妥当という설계判断は技術的に筋が通っています。**バグとしては報告しません**が、意図的な設計差である旨をコメントで明記しておくと今後の混乱防止になります（提案レベルであり、パッチは作成していません）。

### 1.9 Fixed15TapNoiseShaper / PsychoacousticDither のFIRシフトレジスタ

両者とも12〜16タップのエラーフィードバック型ノイズシェーパーで、シフトレジスタの更新順序・係数インデックス対応・デノーマル処理を確認し、正しく実装されていました。

### 1.10 `OutputFilter.cpp` のSSE2/FMA ステレオBiquad

`_mm_set_pd(R,L)` によるパッキングと `_mm_store_sd`/`_mm_storeh_pd` によるアンパッキングの対応関係、Direct Form II TransposedのFMA実装を検証し、正しいことを確認しました。

### 1.11 `FastTanhApprox.h` の高次Padé近似係数

`SoftClipPadéPolicy`（10395係数版）について、コメントに記載された「x=4.5で約0.99927に収束」という主張を実際に数値計算で検算し、一致することを確認しました（≈0.999266）。

### 1.12 `TruePeakDetector.cpp` のワークバッファレイアウト

4倍オーバーサンプリング用のオフセット計算（`kStage0LOffset`等）と確保サイズ（`maxBlockSize * 12`）の整合性を検算し、`numSamples <= maxBlockSize` の前提下で境界外アクセスがないことを確認しました。

---

## 2. 調査範囲マップ（265ファイル中の内訳）

全量76,677行を全行精読することは現実的な時間内では不可能なため、優先度付けを行いました。優先順位は「Audio Threadで実行される／RT規約違反が起きやすい／今回新規追加された」ファイルです。

### 2.1 精読・詳細検証済み（20ファイル前後）

```
src/AlignedAllocation.h                                  src/AllpassDesigner.cpp
src/AutoGainPlanner.cpp/.h (audioengine)                  src/CacheManager.cpp
src/CustomInputOversampler.cpp（一部）                     src/DftiHandle.h
src/Fixed15TapNoiseShaper.h                               src/IRAnalyzer.cpp/.h
src/LockFreeAudioRingBuffer.h                             src/LockFreeRingBuffer.h
src/MKLNonUniformConvolver.cpp（ほぼ全域）                  src/MKLNonUniformConvolver.h
src/OutputFilter.cpp                                      src/PsychoacousticDither.h/.cpp（一部）
src/TruePeakDetector.cpp（一部）                            src/UltraHighRateDCBlocker.h
src/audioengine/AtomicAccess.h                             src/audioengine/AudioEngine.Processing.AudioBlock.cpp
src/audioengine/AudioEngine.Processing.BlockDouble.cpp（一部）
src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp（冒頭のみ）
src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp（冒頭のみ）
src/audioengine/AudioEngine.Timer.cpp（一部）
src/core/IRetireRouter.h                                  src/core/ScopedMXCSR.h
src/dsp/math/FastTanhApprox.h
src/eqprocessor/EQProcessor.Coefficients.cpp（一部）         src/eqprocessor/EQProcessor.Parameters.cpp
```

### 2.2 全体スイープ（grep等でパターン検証、全265ファイル対象）

- `try`/`catch` の使用箇所と所属ファイル（Audio Thread規約違反なし）
- `std::shared_ptr` 使用箇所（コメント1件のみ、実使用なし）
- `std::mutex`/`lock_guard`/`CriticalSection` 使用箇所
- `TODO`/`FIXME`/`XXX`/`HACK` コメント（0件）
- `mkl_malloc`/`mkl_free`・`_aligned_malloc`/`_aligned_free` の粗い出現数比較
- デノーマル対策（`ScopedNoDenormals`/`ScopedMXCSR`/FTZ・DAZ設定）の適用箇所一覧

### 2.3 未調査・調査が浅い領域（優先度順）

**最優先で読むべき（RT/並行処理の中核でありながら未着手）:**

- `src/audioengine/` 配下の `ISR*` 系ファイル群（約40ファイル: `ISRRuntimePublicationCoordinator`, `ISRRetireRouter`, `ISRClosure`, `ISRDSPHandle`, `ISRDSPQuarantine`, `ISRLifecycle`, `ISRRetireLane`, `ISRRetireOverflowRing` 等）— RCU publish/retireパイプラインの本体。今回はEQ側の一部（`retireEQStateDeferred`）としてしか触れておらず、汎用ルーター実装そのものは未検証。
- `src/audioengine/RuntimeBuilder.cpp`, `RuntimePublicationOrchestrator.cpp`, `RuntimePolicyEngine.cpp`, `PublicationAdmission.cpp`, `PublicationExecutor.cpp` — 新規DSPCore構築〜公開の実処理。
- `src/audioengine/AudioEngine.Cache.cpp`, `AudioEngine.RebuildDispatch.cpp`, `AudioEngine.Commit.cpp`, `AudioEngine.Retire.cpp`, `AudioEngine.Transition.cpp` — 状態遷移・リビルド判断ロジック。
- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`, `DSPCoreIO.cpp`, `DSPCoreToBuffer.cpp` — Doubleパスの対になるFloat/IOパス。DSPCoreDoubleとの実装差異（コピペミス等）がないか要確認。
- `src/convolver/ConvolverProcessor.*`（9ファイル）— `LoadPipeline`/`LoaderThread`/`Rebuild`/`ResampleAndFallback`/`Runtime`/`StateAndUI` はtry/catchの所属確認のみで中身は未読。
- `src/eqprocessor/EQProcessor.Processing.cpp`, `EQProcessor.Core.cpp`, `EQProcessor.ProcessingCache.cpp` — EQ本体のサンプル処理ループ（Audio Thread最重要パス）。今回は冒頭のヘルパー関数のみ確認。

**中優先度:**

- `src/NoiseShaperLearner.cpp`（スレッド構造のみ確認、CMA-ESアルゴリズム本体は未検証）、`src/FixedNoiseShaper.h`, `src/LatticeNoiseShaper.h`（未読）
- `src/IRConverter.cpp`（`analyzeIR`関数のみ確認、ファイル全体は未読。IRロードパイプラインの中核）
- `src/core/` 配下のRCUプリミティブ本体: `EpochDomain.h`, `RCUReader.h`, `DeletionQueue.cpp/.h`, `SnapshotCoordinator.cpp/.h`, `SnapshotAssembler.cpp/.h`, `SnapshotFactory.cpp/.h`, `GlobalSnapshot.cpp/.h`, `ObserveChannel.h`
- `src/DeferredDeletionQueue.h`, `src/DeferredFreeThread.h`, `src/RefCountedDeferred.h`, `src/SafeStateSwapper.h`
- `src/ProgressiveUpgradeThread.cpp/.h`（メモリに記載の60〜70秒ウォームアップ窓の実装）
- `src/audioengine/CrossfadeAuthority.cpp/.h`, `CrossfadeRuntime.h`（Bypass/Mixフェードの実発生源。1.8節の設計判断を裏付けるなら本来ここを読むべきでした）

**低優先度（GUIおよびテスト）:**

- GUIコンポーネント一式: `MainWindow`, `MainApplication`, `EQControlPanel`, `EQEditProcessor`, `ConvolverControlPanel`, `ConvolverSettingsComponent`, `NoiseShaperLearningComponent`, `MixedPhaseOptimizationComponent`, `SpectrumAnalyzerComponent`, `LoudnessMeter`（UI描画部分）, `DeviceSettings`
- `src/tests/` 配下18ファイル（契約テスト群。バグ探しというよりは「意図された不変条件」の一次情報源として読む価値はあります）
- `CMakeLists.txt`, `build.bat`（ビルド構成。構造のみ確認、詳細な設定値は未検証）

---

## 3. 次のステップ（提案）

「つづけて」で以下の優先順で継続できます:

1. `EQProcessor.Processing.cpp`（Audio Thread最重要パスの一つ、今回未読）
2. `ISR*` 系（RCU publish/retireの本体、40ファイル規模なので複数ターンに分割推奨)
3. `DSPCoreFloat.cpp`/`DSPCoreIO.cpp`/`DSPCoreToBuffer.cpp`（Doubleパスとの実装差異チェック）
4. `ConvolverProcessor.*` 一式
5. `CrossfadeAuthority.cpp/.h`（1.8節の設計判断の裏取り）

ご希望の順番があればご指定ください。
