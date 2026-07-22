ConvoPeq ソースコード調査報告（バグ・潜在問題一覧）

本報告では、ConvoPeq のソースコード全体を調査し、音声処理アルゴリズム、スレッド安全性、メモリ管理、パフォーマンス、設定・パラメータの各観点から発見したバグおよび潜在的な問題を列挙します。なお、既にコメントやテストで言及されている既知の問題（例：Bug#5）も含め、影響度と提案修正を付記します。

---

## 1. 音声処理アルゴリズムの問題

### 1.1 DeferredDeletionQueue::reclaim() の先頭ブロッキング（Bug#5）

- **ファイル**: `DeferredDeletionQueue.h` (reclaim メソッド)
- **内容**: キュー先頭のエントリが epoch 条件により削除不可の場合、後続の削除可能なエントリも回収されず、メモリリークの原因となる。テスト `testHeadBlocking` で挙動が確認されている。
- **影響**: 長期間運用時に Retire キューが滞留し、メモリ使用量が増加する可能性。
- **提案**: スキャン範囲を拡大し、複数エントリをチェックするか、キュー構造を変更して順序を保証しない方法を検討する。

### 1.2 MixedPhase 変換時の Allpass 設計パラメータ（ライブ再構成時）

- **ファイル**: `convolver/ConvolverProcessor.MixedPhase.cpp` (convertToMixedPhaseAllpass)
- **内容**: ライブ再構成（`liveReconfigure && highRateLive`）時に CMA-ES の世代数が 3、Population が 6 に制限され、最適化が不十分になる可能性がある。
- **影響**: 高サンプルレート（≥96kHz）での Mixed Phase 品質が低下する。
- **提案**: ライブ再構成でも最低限の品質を保証するため、世代数や Population を動的に調整するか、GreedyAdaGrad にフォールバックする現在のロジックを再検討する。

### 1.3 オーバーサンプリングフィルタのレイテンシ推定誤差

- **ファイル**: `audioengine/AudioEngine.Processing.Latency.cpp` (estimateOversamplingLatencySamplesImpl)
- **内容**: オーバーサンプリングのレイテンシを `(taps - 1)` で計算しているが、アップ + ダウンの合成遅延は厳密には `(taps - 1) / 2` のオーダーとなる場合がある（フィルタの対称性による）。特に LinearPhase モードでは非対称な群遅延が生じる可能性がある。
- **影響**: レイテンシ表示が実際より大きくなる、またはクロスフェード時の位相ずれ。
- **提案**: 実測または伝達関数の群遅延解析に基づいて計算式を修正する。

### 1.4 TruePeakDetector のオーバーサンプリングフィルタタップ数

- **ファイル**: `TruePeakDetector.h` (kDefaultTaps = 63)
- **内容**: ITU-R BS.1770-4 は 48 tap を推奨しているが、63 tap を使用。タップ数が多くオーバーサンプリング比 4 で十分な性能かは検証済みと推測されるが、ドキュメントに明記がない。
- **影響**: 過剰な計算リソース消費の可能性（軽微）。
- **提案**: コメントに選定根拠を追記する。

### 1.5 SoftClip の fastTanh 近似精度

- **ファイル**: `dsp/math/FastTanhApprox.h` (DefaultFastTanhPolicy 27/9 Padé)
- **内容**: 27/9 の Padé 近似は `x=3` で 1.0 に収束するが、`x=4.5` 付近で誤差が大きくなる可能性がある（実際の tanh(4.5) ≈ 0.99975）。SoftClip では clipThreshold が 4.5 なので、完全なハードクリップに近い。
- **影響**: 高振幅入力時に不連続なクリップが発生する可能性（軽微）。
- **提案**: より高次の Padé 近似（例：10395/4725）を使用するか、現行のままで問題ないことを実測で確認する。

---

## 2. スレッド安全性・同期の問題

### 2.1 Audio Thread でのロック取得（潜在的）

- **ファイル**: `audioengine/AudioEngine.Processing.AudioBlock.cpp` (process 内の `std::lock_guard` なし)  
  → 現状では Audio Thread 内でのロックは使用していないが、`pendingOverrideLock` は `getMix()` などの getter で使用されており、これらは `process()` 外（Message Thread）で呼ばれることが前提。しかし `captureRuntimeProcessSnapshot()` などが Audio Thread で `pendingOverrideLock` を参照しないように設計されているか要確認。
- **内容**: `pendingOverrideLock` は Message Thread 専用とコメントされているが、一部の getter が Audio Thread から呼ばれる可能性はないか。
- **影響**: 万一 Audio Thread でロックを取得すると、優先度逆転や処理落ちの原因。
- **提案**: Audio Thread で使用される getter は atomic のみ参照するように徹底する。

### 2.2 DeferredDeletionQueue の MPSC キューと fallback の mutex

- **ファイル**: `DeferredDeletionQueue.h` (enqueue と reclaim)
- **内容**: `enqueue` は SPSC の Vyukov キューを使用しているが、`reclaim` は単一の Consumer スレッド（DeferredFreeThread）からのみ呼ばれることが前提。`fallbackMutex` は fallback キューを保護するが、`reclaim` 内で `std::lock_guard` を使用している。
- **影響**: 設計通りであれば問題ないが、`reclaim` が複数スレッドから呼ばれる可能性がある場合はロック競合が発生。
- **提案**: ドキュメントに Single Consumer 前提を明記し、アサーションを追加する。

### 2.3 RCU Reader の Epoch 取得とメモリオーダー

- **ファイル**: `core/EpochDomain.h` (getMinReaderEpoch, enterReader)
- **内容**: `getMinReaderEpoch` は `slot.epoch` を `acquire` で読み、`enterReader` は `release` で書き込む。`slot.depth` と `slot.epoch` の順序が正しく保証されているか確認済みだが、`slot.depth` の更新（fetch_add）と `epoch` の書き込みの間に十分な barrier があるか？
- **影響**: 稀に UAF が発生する可能性。
- **提案**: コードレビューでメモリオーダーを再確認し、必要に応じて `std::atomic_thread_fence` を追加する。

---

## 3. メモリ管理・リソース解放の問題

### 3.1 DIAG_MKL_FREE に渡されるサイズの不整合

- **ファイル**: `MKLNonUniformConvolver.cpp` (Layer::freeAll)
- **内容**: `freeTracked` マクロは `allocSizes` 構造体からサイズを取得して `DIAG_MKL_FREE` に渡す。しかし `allocSizes` は `SetImpulse` 内で設定されるが、`freeAll` が複数回呼ばれると二重解放を防ぐために `ptr` を `nullptr` にしているが、`allocSizes` はクリアされない。解放後に再度 `allocSizes` を参照しないので実害はない。
- **影響**: 診断統計（`zeroAllocSizeCount`）が誤って増加する可能性（軽微）。
- **提案**: `freeAll` 内で `allocSizes` をゼロクリアする。

### 3.2 キャッシュファイルの Temporary ファイル残存

- **ファイル**: `CacheManager.cpp` (save)
- **内容**: `file.withFileExtension("tmp")` に書き込み、成功後に `moveFileTo` でリネームしている。書き込み中にクラッシュすると `.tmp` ファイルが残る。
- **影響**: ディスク容量の無駄遣い（軽微）。
- **提案**: 起動時に古い `.tmp` ファイルを削除する処理を追加する。

### 3.3 MixedPhasePersistentCache のバージョン 1 読み込み時のスケーリング

- **ファイル**: `MixedPhasePersistentCache.cpp` (load)
- **内容**: バージョン 1 のキャッシュを読み込むと、`scaleFactor` が 1.0 に設定されるが、実際の IR データは未スケールの可能性がある（バージョン 1 では scaleFactor が保存されていなかった）。現在のコードでは `scaleFactor` を 1.0 と見なしており、結果としてゲインが変わってしまう可能性がある。
- **影響**: 古いキャッシュからの読み込み時にレベルが不正になる。
- **提案**: バージョン 1 のキャッシュは無効化するか、専用の変換パスを設ける。

---

## 4. パフォーマンス・リアルタイム性の問題

### 4.1 DeferredDeletionQueue::reclaim のスキャン制限

- **ファイル**: `DeferredDeletionQueue.h` (reclaim)
- **内容**: `kMaxScan = 1024` がハードコードされており、一度にスキャンできるエントリ数が制限されている。大量のエントリが滞留すると複数回の reclaim 呼び出しが必要。
- **影響**: 滞留解消に時間がかかる場合がある。
- **提案**: 制限を撤廃するか、動的に調整する。

### 4.2 Audio Thread 内での `std::this_thread::yield()` の使用

- **ファイル**: `audioengine/AudioEngine.Processing.AudioBlock.cpp` など
- **内容**: Audio Thread 内で `std::this_thread::yield()` を呼び出す箇所はないが、`CustomInputOversampler` の `processUp` などで `std::this_thread::yield()` を使用している箇所は非 Audio Thread であることを確認済み。
- **影響**: 問題なし。

### 4.3 CMA-ES 最適化時のスレッド yield

- **ファイル**: `AllpassDesigner.cpp` (designWithCMAES)
- **内容**: ループ内で `std::this_thread::yield()` を呼び出しており、これはワーカースレッド（非 Audio Thread）なので問題ない。

---

## 5. 設定・パラメータ・エラーハンドリングの問題

### 5.1 オーバーサンプリング倍率の上限チェック不足

- **ファイル**: `audioengine/AudioEngine.Parameters.cpp` (setOversamplingFactor)
- **内容**: `factor` が 1,2,4,8 以外の値（例：3）が渡された場合、`newFactor` は 0 になる（Auto 扱い）。しかし、`manualOversamplingFactor` に 0 がセットされ、後で `OversamplingPolicy::resolve` で最大許可倍率が選択される。これは意図された動作か？
- **影響**: ユーザーが誤った値を指定しても気付かれない。
- **提案**: 入力値のバリデーションを行い、不正な値の場合はログ出力する。

### 5.2 NoiseShaperLearner の安定性チェックが有効な場合のペナルティ

- **ファイル**: `NoiseShaperLearner.cpp` (evaluateCandidateMapped)
- **内容**: 不安定な係数に対して `1e18` という巨大なペナルティを返す。これにより最適化が収束しにくくなる可能性がある。
- **影響**: 学習が停滞する。
- **提案**: ペナルティ値を動的に調整するか、安定な領域に投影する方法を検討する。

### 5.3 AllpassDesigner の CMA-ES 終了条件

- **ファイル**: `AllpassDesigner.cpp` (designWithCMAES)
- **内容**: 終了条件 `bestFitness < 1.0` は群遅延誤差が 1 サンプル未満という意味で、高サンプルレートでは厳しすぎる可能性がある（192kHz で 5μs）。
- **影響**: 過剰な最適化時間。
- **提案**: サンプルレートに応じて閾値を調整する（例：`1.0 * sampleRate / 48000`）。

### 5.4 ファイル読み込み時のエラーメッセージの国際化不足

- **ファイル**: 多数（`ConvolverProcessor` など）
- **内容**: エラーメッセージがハードコードされた英語のまま。
- **影響**: ユーザーエクスペリエンス低下（軽微）。
- **提案**: 後続のローカライゼーション対応時に修正。

---

## 6. その他（コーディング規約・潜在的な未定義動作）

### 6.1 `std::bit_cast` の使用とコンパイラ互換性

- **ファイル**: 多数（`DspNumericPolicy.h` など）
- **内容**: C++20 の `std::bit_cast` を使用しているが、プロジェクトは C++20 を要求しているため問題ない。

### 6.2 `restrict` キーワードの非標準使用

- **ファイル**: `CustomInputOversampler.cpp` (dotProductAvx2 など)
- **内容**: `__restrict` は MSVC 拡張であり、他コンパイラでは `__restrict__` を使用する必要があるが、プロジェクトは MSVC/icx を対象としているため問題ない。

### 6.3 未使用の変数や関数

- **ファイル**: いくつかのファイルで `diagLog` 関数が `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内でのみ使用され、無効時は未使用となるが、`[[maybe_unused]]` で対応済み。

---

## 7. 既知の修正済み問題（確認用）

- `AllpassDesigner.cpp` の `unconstrainedToRho` は偶関数からシグモイドに修正済み（二重縮退回避）。
- `AudioEngine.Processing.PrepareToPlay.cpp` の `rebuildRequestGeneration` リセットはクラッシュ修正済み。
- `ConvolverProcessor.cpp` の `delayBuffer` 未確保時のハンドリングは修正済み。

---

## 総合評価

ConvoPeq のコードは全体的に高品質で、スレッド安全性、メモリ管理、リアルタイム制約に対する配慮が行き届いています。発見した問題のほとんどは軽微なもの、または設計上のトレードオフに起因するものです。しかし、`DeferredDeletionQueue::reclaim` の先頭ブロッキングはメモリリークにつながる可能性があるため、優先度の高い修正が推奨されます。また、オーバーサンプリングレイテンシの推定誤差や MixedPhase のライブ最適化パラメータは、品質に影響を与える可能性があるため、実測による検証と調整が必要です。

各問題に対する修正提案は上記に記載の通りです。開発チームのご検討をお願いいたします。


### 追加バグ・潜在問題（再調査分）

前回報告に加え、さらに詳細なコードレビューにより以下の問題を確認しました。一部は既存のコメントやテストからも示唆されていますが、改めて整理します。

---

#### 8. `LinearRamp` の非RT更新とAudio Thread読み取りの競合（`CrossfadeRuntime::start` / `reset`）

- **該当ファイル**: `CrossfadeRuntime.h`, `DspNumericPolicy.h` (`LinearRamp`)
- **問題**: `LinearRamp` の `reset()` は `ASSERT_NON_RT_THREAD` で保護されていますが、`CrossfadeRuntime::start()`（非RTスレッドで呼ばれる）が `gain_.reset()` を実行するタイミングと、Audio Thread が `gain_.getNextValue()` や `setTargetValue()` を実行するタイミングが競合する可能性があります。`reset()` は `totalSteps` などの非原子メンバを変更し、Audio Thread の `setTargetValue()` は `totalSteps` を読み取ります。このデータ競合は理論上発生し得ます（x86 では整列された int の読み書きは原子的ですが、順序保証はない）。
- **影響**: 稀にランプのステップ数が不正になり、オーディオにクリックやポップが発生する恐れ。
- **提案**: `LinearRamp` の設定を変更する操作（`reset`, `setCurrentAndTargetValue`）を、Audio Thread が完全に停止していることが保証されるタイミング（例：`releaseResources` や `prepareToPlay`）に限定するか、`totalSteps` を `std::atomic<int>` に変更し、メモリオーダーを適切に設定する。

---

#### 9. `NoiseShaperLearner::drainCaptureQueue` の sessionId フィルタリング

- **該当ファイル**: `NoiseShaperLearner.cpp` (`drainCaptureQueue`)
- **問題**: セッション互換性判定で、`block.sessionId == 0` の場合も受け入れています。しかし、`sessionId == 0` は未初期化または無効なセッションを示す可能性があり、本来は破棄すべきです。現在のロジックでは、`session.sessionId == 0` のときはすべてのブロックを受け入れ、`block.sessionId == 0` のときも受け入れます。これにより、DSP が再構築された直後の古いセッション ID が 0 のブロックが学習に混入するリスクがあります。
- **影響**: 学習データの汚染により、適応ノイズシェイパーの収束が阻害される可能性。
- **提案**: `sessionId` が `0` のブロックは厳密に拒否し、`session.sessionId` が `0` の場合はフォールバックとしてすべて受け入れるのではなく、学習開始時点で有効な `sessionId` を設定することを強制する。

---

#### 10. `CacheManager::loadPreparedState` のメモリマップ失敗時のクラッシュ

- **該当ファイル**: `CacheManager.cpp` (`loadPreparedState`)
- **問題**: `juce::MemoryMappedFile` のコンストラクタが失敗しても、`getData()` が `nullptr` を返す可能性がありますが、その後の `memcpy` や CRC 計算でデリファレンスされ、アクセス違反が発生します。
- **影響**: キャッシュファイルが破損しているか、別プロセスがロックしている場合にアプリがクラッシュする。
- **提案**: `mmap.getData()` が `nullptr` の場合を明示的にチェックし、エラーとして扱う。

---

#### 11. `ConvolverProcessor::shareConvolutionEngineFrom` の状態同期漏れ

- **該当ファイル**: `convolver/ConvolverProcessor.StateAndUI.cpp` (`shareConvolutionEngineFrom`)
- **問題**: `shareConvolutionEngineFrom` はエンジンとレイテンシキャッシュを複製しますが、`irLength` や `uiTotalLatencySamples` などの atomic 変数をコピーしていますが、`IRState`（現在の IR データ）や `currentIRScale` はコピーされていません。そのため、IR データ自体は共有されず、UI 表示が不整合になる可能性があります。
- **影響**: 複数の ConvolverProcessor インスタンス間でエンジンを共有した場合、波形表示や IR 情報が正しく更新されない。
- **提案**: `transferIRStateFrom` を利用して IR データも共有するか、必要なメタデータを明示的に同期する。

---

#### 12. `AllpassDesigner::designWithCMAES` の early exit 条件がサンプルレート非依存

- **該当ファイル**: `AllpassDesigner.cpp` (`designWithCMAES`)
- **問題**: 終了条件 `bestFitness < 1.0` は群遅延誤差が 1 サンプル未満を意味しますが、サンプルレートが高いほど時間分解能が上がり、到達が難しくなります。結果として、高サンプルレート（192kHz以上）で最適化が早期終了せず、過剰な反復が発生する可能性があります。
- **影響**: CPU 負荷の増加、最適化時間の長期化。
- **提案**: 閾値を `1.0 * sampleRate / 48000.0` のようにサンプルレートに比例させる。

---

#### 13. `ConvolverProcessor::handleLoadError` での `isRebuilding` フラグのクリア漏れ

- **該当ファイル**: `convolver/ConvolverProcessor.LoadPipeline.cpp` (`handleLoadError`)
- **問題**: `handleLoadError` は `isLoading` と `irFinalized` をリセットしますが、`isRebuilding` はリセットしません（呼び出し元でリセットされる場合とされない場合があります）。リビルド中にエラーが発生すると、`isRebuilding` が `true` のまま残り、以降の再構築要求がブロックされる可能性があります。
- **影響**: エラー後の再試行ができなくなる。
- **提案**: `handleLoadError` 内で `convo::publishAtomic(isRebuilding, false, ...)` を追加する。

---

#### 14. `LoudnessMeter::updateCoefficients` での `M_PI` 非標準マクロ

- **該当ファイル**: `LoudnessMeter.cpp` (`updateCoefficients`)
- **問題**: `#define _USE_MATH_DEFINES` を宣言していますが、`#include <JuceHeader.h>` の後に `#include <cmath>` が来るかどうかに依存します。`JuceHeader.h` が `cmath` を内部でインクルードしている場合でも、`_USE_MATH_DEFINES` が定義される前にインクルードされていると `M_PI` は定義されません。MSVC では問題ない場合もありますが、クロスコンパイルで潜在的な問題。
- **影響**: ビルドエラーまたは誤った定数（0）の使用。
- **提案**: `juce::MathConstants<double>::pi` を利用する（既に他の箇所で使用）。

---

#### 15. `AudioEngine::collectDrainAudit` の `stuckReaderCount` が最初の1件しか検出しない

- **該当ファイル**: `audioengine/AudioEngine.Threading.cpp` (`collectDrainAudit`)
- **問題**: `detectStuckReaders` は最初に検出された Stuck Reader のみを返すため、`stuckReaderCount` は 0 または 1 になりますが、複数の Reader が Stuck している可能性があります。
- **影響**: 監査情報が不完全となり、シャットダウン時のブロッキング原因の特定が難しくなる。
- **提案**: `detectStuckReaders` を全スロットをスキャンしてカウントするように拡張するか、`collectDrainAudit` 内で個別に全スロットをチェックする。

---

#### 16. `ThreadAffinityManager::detectCoreTopology` が `GetLogicalProcessorInformationEx` 失敗時に空のトポロジを返す

- **該当ファイル**: `core/ThreadAffinityManager.h` (`detectCoreTopology`)
- **問題**: API 失敗時に `physicalCoreCount=0` のトポロジを返しますが、`hasHeterogeneousCores_` は `false` に設定され、アフィニティマスクの計算がスキップされます。しかし、実際にはコア数が 0 と見なされるため、Audio Thread のアフィニティが設定されず、パフォーマンスが低下する可能性があります。
- **影響**: 古い Windows バージョンや特定の環境で CPU アフィニティが無効になる。
- **提案**: 失敗時はフォールバックとして全コアマスク（`~0`）を設定するか、少なくともログ出力してユーザーに通知する。

---

#### 17. `EQProcessor::process` の `parallel` モードで `activeParallelInputBuffer` が未割り当ての場合のフォールバックが不完全

- **該当ファイル**: `eqprocessor/EQProcessor.Processing.cpp` (`process` の Parallel ブランチ)
- **問題**: Parallel バッファが不足している場合、Serial 処理にフォールバックしますが、その際に `filterState` はそのまま使用されるため、Parallel と Serial で状態が混ざることがあります。特に、バッファ不足が一時的であれば、状態の不連続が発生する可能性があります。
- **影響**: モード切り替え時のクリックやノイズ。
- **提案**: フォールバック時にすべてのフィルタ状態をリセットするか、Parallel バッファを確実に確保する。

---

#### 18. `ConvolverProcessor::applyBuildSnapshot` が `pendingOverrideLock` 内で `copySnapshotToPendingUnlocked` を呼ぶが、他のメンバ（`irName`など）はロック外で更新

- **該当ファイル**: `convolver/ConvolverProcessor.StateAndUI.cpp` (`applyBuildSnapshot`)
- **問題**: `irName` や `irLength` への書き込みがロックなしで行われますが、これらは UI スレッドと Audio スレッドの両方からアクセスされる可能性があります（`irName` は主に UI）。`irLength` は atomic ですが、`irName` は非原子の `juce::String` であり、競合が発生する可能性は低いですが、安全ではありません。
- **影響**: まれに `irName` が破損する可能性。
- **提案**: `irName` も `std::atomic` で保護するか、ロック内で更新する。

---

#### 19. `NoiseShaperLearner::startLearning` で `workerState` が `Idle` 以外の場合にリセットせずに `stopLearning` を呼ぶが、`stopLearning` 内で `workerState` を `Idle` に戻すため問題ないが、`pendingResume` が残る可能性。

- 実害はないが、ロジックが複雑。

---

#### 20. `AudioEngine::requestRebuild` で `task.convolverBuildSnapshot` がコピーされるが、`ConvolverProcessor::BuildSnapshot` には `juce::File` が含まれ、コピーコストが大きい。しかし頻繁ではないので許容範囲。

---

以上、追加のバグ・問題点を列挙しました。特に **8**, **10**, **12** は修正が必要なレベルです。**9** は機能に影響を与える可能性があり、**11** は表示上の問題です。その他は軽微またはエッジケースです。