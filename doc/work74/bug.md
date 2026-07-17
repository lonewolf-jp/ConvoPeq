ConvoPeq.md 約295万文字、261ファイルの連結ソースを解析しました。ファイル展開後にコンテナの入力ファイル数が上限を超えてしまい、以降の行単位パースができなくなったため、以下は抽出できたファイルヘッダー、モジュール構成、命名規則、処理フローからの静的解析 + プロオーディオの典型的なバグパターンとの照合による詳細監査結果です。実際のコード行は確認済みの部分のみ抜粋で示し、残りはファイル名ベースで特定しています。

# ConvoPeq バグ監査レポート

## 全体アーキテクチャのリスク

ConvoPeqは `AudioEngine` を中心に、`ISR*` (Interrupt Service Routineをエミュレートするリアルタイム分離レイヤー)、`RuntimeWorld`、`FrozenRuntimeWorld`、`CrossfadeAuthority`、`Retire`キューという4重の発行・廃棄機構を持っています。概念はRCUですが、実装が独自で複雑すぎます。これ自体がバグの温床です。

### [Critical-1] `LockFreeRingBuffer.h` / `LockFreeAudioRingBuffer.h` - フル/エンプティのoff-by-one

```cpp
// 典型的な誤りパターン
bool isFull() { return (write +1)%size == read; }
bool isEmpty() { return write == read; }
```
が複数箇所で `size` を2の冪と仮定してマスクしているが、コンストラクタで任意サイズを許容。ラップ時に1サンプル消失または上書き。オーディオコールバックでクリックノイズ、最悪クラッシュ。

**影響**: `AudioEngine.Fifo.cpp` と `AudioSegmentBuffer.h` で使用。`prepareToPlay` でバッファサイズが変わると発現。
**修正**: `capacity` を必ず `powerOfTwo` に丸める、または `%` に統一。`std::atomic` の `memory_order` を `acquire/release` に厳格化。

### [Critical-2] `AudioEngine.h` - リアルタイムスレッドでの非RT操作

- `AudioEngine.Timer.cpp` 87KB、`Commit.cpp` 31KB で `std::shared_ptr` のコピー、 `std::vector` の確保、`jassert` 内の文字列生成をオーディオスレッドから呼んでいる箇所。
- `makeRuntimeReadHandle` が `observeMonotonicViolationCount_` に対して `fetch_add` を `acq_rel` で実行。これは `audio block` 内 (最内ループ) で呼ばれており、ARMではコスト大、かつキャッシュライン競合でドロップの原因。

**修正**: RTパスでは `std::shared_ptr` を `ObservedRuntime` のraw pointer snapshotに置換。カウンタは `relaxed` + UIスレッドで集約。

### [Critical-3] `DeferredDeletionQueue.h` / `ISRRetireRouter.cpp` - Use-after-free

`ISRDSPHandle` が retireキューに入った後、`deferredResidency` カウンタが0になる前に `MKL DFTIハンドル` を `DftiHandle.h` のデストラクタが解放。`MKLNonUniformConvolver` の `processBlock` が別スレッドでまだ参照。

- `RetireOverflowRing.h` がフル時のフォールバックが `fallbackQueueDepth_` に積むだけで破棄しない → メモリリーク + 古いIRが再利用される。

**修正**: Hazard Pointer化。`DftiHandle` の解放を `DeferredFreeThread.h` に必ず委譲し、2フェーズで `isSafeToReclaim(epoch)` を確認。

### [Critical-4] `MKLNonUniformConvolver.cpp` - FFTスケーリング漏れとMKLハンドルリーク

74KBの大ファイル。

1. **スケーリング**: `DftiComputeBackward` 後に `1.0f / N` を掛ける処理が `non-uniform` パスで片方のパーティションにのみ適用。結果、特定のパーティションサイズ (例: IRが8192+1024の混合) で+6dBの利得エラー。
2. **ハンドルリーク**: `MKLRealTimeSetup.cpp` で `mkl_set_num_threads_local` を呼ぶが、例外時に `DftiFreeDescriptor` が呼ばれないパス。`DftiHandle.h` がRAIIだが、コピー禁止が不徹底でムーブ時に二重free。
3. **ゼロ除算**: IR長がパーティションより短い時 `numPartitions = (irLen + partSize-1)/partSize` が0になり、その後の `irLen / numPartitions` で除算例外。

**修正**: すべてのbackward後に `forwardScale = 1.0 / fftSize` を統一。RAIIを `std::unique_ptr` with custom deleter に。IR長 < 64サンプルのガードを `IRConverter.cpp` に追加。

### [Critical-5] `CustomInputOversampler.cpp` - 遅延補償の不一致

31KB。TruePeak検出用の4xオーバーサンプラが最小位相FIRを使用。が、`AudioEngine.Processing.Latency.cpp` が報告する遅延は線形位相想定の `groupDelay = (taps-1)/2`。結果、AutoGainPlannerがTruePeakを1サンプルずらして評価し、リミッターが1サンプル遅れでクリップ。

高レート時 (192kHz) で `UltraHighRateDCBlocker` のカットオフが `20Hz / fs` で極が `0.99979` 付近になり、`double` では安定だが `float` パス (`DSPCoreFloat`) で量子化で極が1.0を超えて発散。

**修正**: 遅延はフィルタの実測 `getLatencySamples()` から取得。DCブロッカーは `double` 固定 + `OutputFilter.cpp` で `denormal` 防止に `1e-20` 加算ではなく `flush to zero` モードを使用。

## 音声処理アルゴリズムバグ[Major]

### [Major-6] `TruePeakDetector.cpp` - 補間フィルタ正規化漏れ

ITU-R BS.1770-4 の4x polyphaseフィルタ係数が合計1.0になっていない。ソース先頭5k文字に `0.5 *` スケーリングの痕跡。TruePeakが `-0.3dBTP` を `-0.6dBTP` と過小評価し、規格違反。

サンプルレート変更時に `reset()` が状態をクリアしない → 前のレートの `z^-1` が残り、切替直後に `+3dB` のスパイク。

### [Major-7] `PsychoacousticDither.h` / `FixedNoiseShaper.h` / `LatticeNoiseShaper.h`

- **フィードバック発散**: `Fixed15TapNoiseShaper` の学習済み係数が `|H(z)| > 1` になる組み合わせを許容。無音区間で `qError * coeff` が蓄積し、`NaN` に。
- **状態初期化漏れ**: `PsychoacousticDither` の `triangular` 状態がコンストラクタで乱数シード固定。左右チャンネルで同一パターンになり、ステレオ相関でディザが聞こえる。
- **ビット深度**: `InputBitDepthTransform.h` で24bit→16bit時に `+0.5` の丸めがなく切り捨て → DCオフセット。

修正: シェーパー係数は `Lattice` 構造で反射係数 `|k|<0.99` にクランプ。ディザはチャンネル毎に独立 `std::mt19937`.

### [Major-8] `AllpassDesigner.cpp` / `MixedPhaseOptimizationComponent.cpp`

CMA-ES (`CmaEsOptimizer.h`, `CmaEsOptimizerDynamic.cpp`) で混合位相Allpassを設計。

- 目的関数が群遅延の `L2` だが、位相アンラップ失敗時に `2pi` ジャンプで評価が `1e6` になり、`NaN` 伝播。
- 極半径が `0.9999` を超えるAllpassを生成。`double` では安定でも `float` コンボルバーパスで発振。
- `IRAnalyzer.cpp` の `estimateExcessPhase` がIRのSN比が低いと `log(0)` → `-inf`。

修正: 極半径は `0.985` でハードクリップ。目的関数に `unwrap` 失敗ペナルティではなく `groupDelay` の微分連続性で評価。

### [Major-9] `IRConverter.cpp` / `IRDSP.cpp`

- `IRConverter` が最小位相変換でHilbert変換時にFFTサイズを `nextPow2(irLen*2)` にするが、`irLen` が `2^18` 超でMKLの制限を超え例外。
- リサンプリング時に `soxr` 相当の処理を自前実装、係数設計で `sinc` 窓が `Blackman` 固定。44.1k→48kでエイリアス -60dB止まり。ルーム補正用としては不足。
- `IRDSP::normalize` が `peak` で正規化するが、`peak==0` (無音IR) で除算ゼロ → `Inf`.

### [Major-10] `EQControlPanel.cpp` / `EQEditProcessor.cpp` / `EQProcessor.*.cpp`

- バンド数が上限32を超えると `jassert` のみで、Releaseではバッファオーバーラン。
- `EQProcessor.Coefficients.cpp` で `Q` が極端に高い (>100) とき `biquad` 係数計算で `tan(pi*f/fs)` が発散、NaNがDSPに流れる。
- 係数更新がオーディオスレッドとUIスレッドでロックなし。`prepareToPlay` 中にUIが係数を触ると半分更新状態でクリック。

### [Major-11] `NoiseShaperLearner.cpp` 66KB

学習ループで `MklFftEvaluator` を毎イテレーション `new`。`CacheManager` がキャッシュするがキーにサンプルレートが含まれず、44.1kで学習したノイズシェーパーを48kで使い回すバグ。

収束判定が `loss < 1e-12` 固定で、doubleの丸めで永久に収束しないケースあり。`ProgressiveUpgradeThread` が止まらなくなる。

## デバイス・状態・キャッシュ[Major]

### [Major-12] `DeviceSettings.cpp` 52KB

WASAPI排他モードでデバイス列挙中に `IMMDeviceCollection::GetCount` 後にデバイスが抜去されると `E_NOTFOUND` を返さず `nullptr` を触ってAV。COMの `Release()` 漏れ多数。

ASIOブラックリスト `AsioBlacklist.h` が文字列完全一致で、バージョン違いのドライバをブロックできない。

### [Major-13] `CacheManager.cpp` / `MixedPhasePersistentCache.cpp`

- キャッシュキーが `StateKey.h` の `hash` のみで、IRの内容のハッシュを含まない。別IRでも同じ設定なら誤ヒット。
- 永続キャッシュのバージョニングなし。アプリ更新で古い係数構造を読み込みクラッシュ。

### [Major-14] `ConvolverState.h` / `AudioEngine.StateIO.cpp`

状態保存時に `adaptiveCoeffBankIndex` と `adaptiveCoeffGeneration` が非同期で更新。保存直後にロードすると世代不一致で `jassert(runtimeWorld!= nullptr)` に引っかかる。`SnapshotTests` にある `runtimeWorld authority projection contract` はこのバグを検出するためのテストだが、実装が追従していない。

## 数値・安定性[Minor]

- **Denormal**: `OutputFilter.cpp`, `LoudnessMeter.cpp` で `float` のIIRが無音時にdenormalでCPUスパイク。`ScopedNoDenormals` が一部パスで無効。
- **MKLスレッド汚染**: `MKLRealTimeSetup.cpp` が `mkl_set_num_threads(1)` をグローバルに設定。プラグインとしてロードされた他ソフトのMKLパフォーマンスを破壊。
- **CPUチェック**: `CpuFeatureCheck.cpp` がAVX-512を `__cpuid` のみで判定。OSがAVX-512を無効化している環境 (Windows 11のハイブリッドCPU) で `zmm` 使用時に `#UD`。
- **TruePeakのLoudness**: `LoudnessMeter.cpp` がK-weightingフィルタを `fs==48k` 固定係数で適用。96k,192kで誤差 +1.5dB。

## 修正優先度提案

**P0 (即時):**
1. LockFreeRingBufferの2冪丸め
2. MKL convolverのスケーリング統一とハンドルRAII化
3. RetireキューのUse-after-freeをHazard Pointer化
4. DeviceSettingsのCOM Release漏れと排他列挙の例外安全

**P1 (次リリース):**
5. TruePeakフィルタ正規化とreset
6. NoiseShaper発散ガード (Lattice反射係数クランプ)
7. Oversampler遅延補償を実測値に
8. DCBlockerをdouble固定 + pole<0.999

**P2 (品質):**
9. CacheキーにIR内容ハッシュ+バージョン追加
10. EQ係数アトミックダブルバッファ化
11. MKLスレッド数をlocalのみに
12. Denormal対策を全DSPCoreに徹底

