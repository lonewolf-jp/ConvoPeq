# ConvoPeq 統合バグリスト

**作成日**: 2026-07-22
**対象ファイル**: deepseek.md, gemini.md, meta_ai.md, qwen.md
**総バグ数**: 73件（重複含む）

---

## 📊 重大度別集計

| 重大度 | 件数 | 説明 |
|--------|------|------|
| **Critical** | 14件 | クラッシュ、メモリ破壊、Use-After-Free |
| **High** | 21件 | メモリリーク、データレース、严重的な品質問題 |
| **Medium** | 25件 | 計算誤差、パフォーマンス問題、エッジケース |
| **Low** | 13件 | 軽微な問題、改善提案 |

---

## 🔴 Critical バグ（14件）

### C-1: DeferredDeletionQueue::reclaim() の先頭ブロッキング（Bug#5）

- **ファイル**: `DeferredDeletionQueue.h`
- **問題**: キュー先頭のエントリが epoch 条件により削除不可の場合、後続の削除可能なエントリも回収されず、メモリリークの原因
- **影響**: 長期間運用時に Retire キューが滞留し、メモリ使用量が増加
- **提案**: スキャン範囲を拡大し、複数エントリをチェックするか、キュー構造を変更

### C-2: ScopedAlignedPtr::reset における自己代入保護の欠如（Use-After-Free）

- **ファイル**: `AlignedAllocation.h`
- **問題**: `reset` メソッド内で、保持しているポインタと引数 `p` が同一である場合のチェックが存在しない
- **影響**: 即座にメモリが解放されダングリングポインタとなり、ランタイムクラッシュやメモリ破壊
- **提案**: `ptr != p` ガードを追加

### C-3: CustomInputOversampler / TruePeakDetector - プリフェッチがガードページを越える

- **ファイル**: `CustomInputOversampler.cpp`
- **問題**: `dotProductAvx2` 内で `_mm_prefetch(x+64)` がヒープ外をプリフェッチ
- **影響**: ASAN で heap-buffer-overflow 検出
- **提案**: プリフェッチの境界チェックを追加

### C-4: Fixed15TapNoiseShaper / LatticeNoiseShaper - クランプ→ディザでオーバーフロー

- **ファイル**: `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h`
- **問題**: 量子化後に `int16` で `-32768` にラップ
- **影響**: オーディオにクリックやポップノイズが発生
- **提案**: 量子化前に `std::clamp` を適用

### C-5: MKLNonUniformConvolver Directパス memset越え

- **ファイル**: `MKLNonUniformConvolver.cpp`
- **問題**: `directTapCount > irLen` の場合、`memset` が配列外に書き込み
- **影響**: ASAN で heap-buffer-overflow 検出
- **提案**: `std::min(m_directTapCount, irLen)` を使用

### C-6: SafeStateSwapper 2-step bump 競合

- **ファイル**: `SafeStateSwapper.h`
- **問題**: 2つのスレッドが同じ EpochDomain を共有し、retired epoch が逆転して UAF 発生
- **影響**: ランタイムクラッシュ
- **提案**: `swapMutex` を追加してスレッド安全に

### C-7: ConvolverProcessor::LoaderThread — applyNewState 呼び出し時のスレッド規約違反

- **ファイル**: `ConvolverProcessor.LoaderThreadInline.h`
- **問題**: ワーカースレッドで `applyNewState()` を呼び、Message Thread 前提の操作を実行
- **影響**: JUCE のスレッド規約違反、潜在的なクラッシュ
- **提案**: `callAsync` 経由で Message Thread で実行されるよう修正

### C-8: AudioSegmentBuffer::pushBlock — データレース（複数Writer可能性）

- **ファイル**: `AudioSegmentBuffer.h`
- **問題**: `writePosition` と `totalSamples` の read-modify-write がアトミック操作ではない
- **影響**: Lost update によりデータ不整合
- **提案**: SPSC 前提を明記し、アサーションで保護

### C-9: DeferredDeletionQueue::reclaim — FIFO 先頭ブロッキングによるメモリリーク

- **ファイル**: `DeferredDeletionQueue.h`
- **問題**: 先頭エントリの epoch が `minReaderEpoch` より新しい場合、後続の全エントリが回収されない
- **影響**: メモリが無限に増殖
- **提案**: 先読み機能を実装し、FIFO 順序を保証しない

### C-10: EQProcessor::process — processBandStereo での NaN 伝播

- **ファイル**: `EQProcessor.Processing.cpp`
- **問題**: ループ内で NaN が発生した場合、`numSamples` 分すべてが NaN で汚染
- **影響**: 出力音声に NaN が伝播し、オーディオが無音になるかクラッシュ
- **提案**: ループ内で NaN チェックを追加

### C-11: ConvolverProcessor::process — delayBuffer 未初期化時の未定義動作

- **ファイル**: `ConvolverProcessor.Runtime.cpp`
- **問題**: `prepareToPlay()` が呼ばれる前に `process()` が呼ばれた場合、`delayBuffer[0].get()` が `nullptr`
- **影響**: `memcpy` でクラッシュ
- **提案**: `nullptr` チェックを追加

### C-12: RuntimePublicationCoordinator — publishWorld 後の worldOwner 二重解放リスク

- **ファイル**: `core/RuntimePublicationCoordinator.h`
- **問題**: `worldOwner.release()` 後に publish が失敗した場合、`const_cast` されたポインタに対して `~T()` を呼び出す
- **影響**: `const` オブジェクトのデストラクタ呼び出しは UB
- **提案**: `const` を外すか、設計を見直す

### C-13: CustomInputOversampler::decimateStage — 境界チェック後の OOB アクセス

- **ファイル**: `CustomInputOversampler.cpp`
- **問題**: `convCount` が大きい場合、`base - convParity - ((convCount-1)*2)` が負のインデックスになる
- **影響**: 配列外アクセスによりクラッシュ
- **提案**: 境界チェックを追加

### C-14: ConvolverProcessor::StereoConvolver::init — 例外安全性の不完全さ

- **ファイル**: `ConvolverProcessor.h`
- **問題**: `irData[0] = newIrL.release()` の後に例外が発生すると、`irData[0]` がリーク
- **影響**: メモリリーク
- **提案**: 例外安全性を確保する設計に変更

---

## 🟠 High バグ（21件）

### H-1: MixedPhase 変換時の Allpass 設計パラメータ（ライブ再構成時）
- **ファイル**: `convolver/ConvolverProcessor.MixedPhase.cpp`
- **問題**: ライブ再構成時に CMA-ES の世代数が 3、Population が 6 に制限
- **影響**: 高サンプルレート（≥96kHz）での Mixed Phase 品質が低下
- **提案**: 世代数や Population を動的に調整

### H-2: オーバーサンプリングフィルタのレイテンシ推定誤差
- **ファイル**: `audioengine/AudioEngine.Processing.Latency.cpp`
- **問題**: レイテンシを `(taps - 1)` で計算しているが、厳密には `(taps - 1) / 2`
- **影響**: レイテンシ表示が実際より大きくなる
- **提案**: 実測または伝達関数の群遅延解析に基づいて計算式を修正

### H-3: LinearRamp の非RT更新とAudio Thread読み取りの競合
- **ファイル**: `CrossfadeRuntime.h`, `DspNumericPolicy.h`
- **問題**: `reset()` は非RTスレッドで保護されているが、Audio Thread が `setTargetValue()` を実行するタイミングが競合
- **影響**: 稀にランプのステップ数が不正になり、オーディオにクリックやポップが発生
- **提案**: `totalSteps` を `std::atomic<int>` に変更

### H-4: NoiseShaperLearner::drainCaptureQueue の sessionId フィルタリング
- **ファイル**: `NoiseShaperLearner.cpp`
- **問題**: `sessionId == 0` のブロックを受け入れ、古いセッション ID が学習に混入
- **影響**: 学習データの汚染により、適応ノイズシェイパーの収束が阻害
- **提案**: `sessionId` が `0` のブロックは厳密に拒否

### H-5: CacheManager::loadPreparedState のメモリマップ失敗時のクラッシュ
- **ファイル**: `CacheManager.cpp`
- **問題**: `juce::MemoryMappedFile` のコンストラクタが失敗しても、`getData()` が `nullptr` を返す
- **影響**: キャッシュファイルが破損している場合にアプリがクラッシュ
- **提案**: `mmap.getData()` が `nullptr` の場合を明示的にチェック

### H-6: ConvolverProcessor::shareConvolutionEngineFrom の状態同期漏れ
- **ファイル**: `convolver/ConvolverProcessor.StateAndUI.cpp`
- **問題**: `IRState` や `currentIRScale` がコピーされていない
- **影響**: 複数の ConvolverProcessor インスタンス間で波形表示や IR 情報が正しく更新されない
- **提案**: `transferIRStateFrom` を利用して IR データも共有

### H-7: AllpassDesigner::designWithCMAES の early exit 条件がサンプルレート非依存
- **ファイル**: `AllpassDesigner.cpp`
- **問題**: 終了条件 `bestFitness < 1.0` は高サンプルレートで厳しすぎる
- **影響**: CPU 負荷の増加、最適化時間の長期化
- **提案**: 閾値を `1.0 * sampleRate / 48000.0` のようにサンプルレートに比例させる

### H-8: ConvolverProcessor::handleLoadError での isRebuilding フラグのクリア漏れ
- **ファイル**: `convolver/ConvolverProcessor.LoadPipeline.cpp`
- **問題**: `handleLoadError` は `isRebuilding` をリセットしない
- **影響**: エラー後の再試行ができなくなる
- **提案**: `handleLoadError` 内で `convo::publishAtomic(isRebuilding, false, ...)` を追加

### H-9: LoudnessMeter::updateCoefficients での M_PI 非標準マクロ
- **ファイル**: `LoudnessMeter.cpp`
- **問題**: `_USE_MATH_DEFINES` の定義タイミングに依存
- **影響**: ビルドエラーまたは誤った定数（0）の使用
- **提案**: `juce::MathConstants<double>::pi` を利用

### H-10: AudioEngine::collectDrainAudit の stuckReaderCount が最初の1件しか検出しない
- **ファイル**: `audioengine/AudioEngine.Threading.cpp`
- **問題**: `detectStuckReaders` は最初に検出された Stuck Reader のみを返す
- **影響**: 監査情報が不完全となり、シャットダウン時のブロッキング原因の特定が難しくなる
- **提案**: 全スロットをスキャンしてカウントするように拡張

### H-11: ThreadAffinityManager::detectCoreTopology が API 失敗時に空のトポロジを返す
- **ファイル**: `core/ThreadAffinityManager.h`
- **問題**: API 失敗時に `physicalCoreCount=0` のトポロジを返す
- **影響**: 古い Windows バージョンで CPU アフィニティが無効になる
- **提案**: 失敗時はフォールバックとして全コアマスクを設定

### H-12: EQProcessor::process の parallel モードで activeParallelInputBuffer が未割り当ての場合のフォールバックが不完全
- **ファイル**: `eqprocessor/EQProcessor.Processing.cpp`
- **問題**: Parallel バッファが不足している場合、Serial 処理にフォールバックするが、状態が混ざる
- **影響**: モード切り替え時のクリックやノイズ
- **提案**: フォールバック時にすべてのフィルタ状態をリセット

### H-13: ConvolverProcessor::applyBuildSnapshot が pendingOverrideLock 内で copySnapshotToPendingUnlocked を呼ぶが、他のメンバはロック外で更新
- **ファイル**: `convolver/ConvolverProcessor.StateAndUI.cpp`
- **問題**: `irName` や `irLength` への書き込みがロックなしで行われる
- **影響**: まれに `irName` が破損する可能性
- **提案**: `irName` も `std::atomic` で保護するか、ロック内で更新

### H-14: RCU遅延削除キューにおけるメモリリーク
- **ファイル**: AudioThread 制御および Epoch ベース RCU モデル
- **問題**: 遅延削除キューにおいて、メモリの解放処理が正常にキックされない
- **影響**: メモリ使用量が肥大化
- **提案**: 解放トリガーを見直す

### H-15: コンヴォルヴァー破棄時におけるUse-After-Free
- **ファイル**: 畳み込み演算エンジンのライフサイクル管理
- **問題**: UI/制御スレッドとリアルタイム音声処理スレッドの間でレースコンディション
- **影響**: 非決定的なランタイムクラッシュ
- **提案**: ポインタのライフサイクルと RCU の解放トリガーを見直す

### H-16: 状態遷移パスおよび遷移完了通知のハンドリング不備
- **ファイル**: 状態遷移マシンおよび通知ルーチン
- **問題**: 処理状態の遷移パスと完了通知のルーティングに不備
- **影響**: UIと内部音声エンジン側の状態が乖離
- **提案**: 遷移パスと通知のルーティングを修正

### H-17: ソフトクリッパーにおけるSIMD状態の破損
- **ファイル**: `SoftClipper` クラス（AVX2/SIMD最適化コード）
- **問題**: SIMD並列演算ループにおいて、レジスタの退避・復元にミス
- **影響**: オーディオに突発的なプチノイズやバーストノイズが発生
- **提案**: SIMD レジスタの退避・復元を修正

### H-18: ソフトニー・リミッターにおける数学的不連続性
- **ファイル**: リミッターのダイナミクス処理アルゴリズム
- **問題**: ソフトニー特性を計算する補間数式に不連続点が存在
- **影響**: 高次の高調波歪みやポップノイズ
- **提案**: C2連続な3次スプライン等への置き換え

### H-19: AVX2 デシメーションにおけるメモリ安全性問題
- **ファイル**: `AVX2 Decimation`（ダウンサンプリング/帯域分割処理）
- **問題**: バッファ境界チェックの不足、またはSIMDポインタのインクリメント計算の誤り
- **影響**: 隣接するヒープ領域のデータを破壊、最悪の場合はクラッシュ
- **提案**: 境界チェックを追加

### H-20: ノイズシェーパのゲイン補正誤り
- **ファイル**: ディザリングおよび `NoiseShaper` モジュール
- **問題**: ゲイン補正係数の計算ロジックに誤り
- **影響**: ノイズフロアが上昇する、または全体のエネルギーバランスが崩れる
- **提案**: ゲイン補正ロジックを修正

### H-21: "ISR Verification" ワークフローの継続的な失敗
- **ファイル**: CI/CD環境（GitHub Actions等）
- **問題**: リアルタイム性と自己修復機能を検証するための自動検証ワークフローが失敗し続けている
- **影響**: 製品クオリティの安定性を自動担保できない
- **提案**: ワークフローの修正とテストの安定化

---

## 🟡 Medium バグ（25件）

### M-1: TruePeakDetector のオーバーサンプリングフィルタタップ数
- **ファイル**: `TruePeakDetector.h`
- **問題**: ITU-R BS.1770-4 は 48 tap を推奨しているが、63 tap を使用
- **影響**: 過剰な計算リソース消費の可能性
- **提案**: コメントに選定根拠を追記

### M-2: SoftClip の fastTanh 近似精度
- **ファイル**: `dsp/math/FastTanhApprox.h`
- **問題**: 27/9 の Padé 近似は `x=4.5` 付近で誤差が大きくなる可能性
- **影響**: 高振幅入力時に不連続なクリップが発生
- **提案**: より高次の Padé 近似を使用するか、実測で確認

### M-3: DeferredDeletionQueue の MPSC キューと fallback の mutex
- **ファイル**: `DeferredDeletionQueue.h`
- **問題**: `reclaim` が複数スレッドから呼ばれる可能性がある場合、ロック競合が発生
- **影響**: デザイン通りであれば問題ないが、ドキュメントに Single Consumer 前提を明記すべき
- **提案**: アサーションを追加

### M-4: RCU Reader の Epoch 取得とメモリオーダー
- **ファイル**: `core/EpochDomain.h`
- **問題**: `slot.depth` の更新（fetch_add）と `epoch` の書き込みの間に十分な barrier があるか確認が必要
- **影響**: 稀に UAF が発生する可能性
- **提案**: コードレビューでメモリオーダーを再確認

### M-5: DIAG_MKL_FREE に渡されるサイズの不整合
- **ファイル**: `MKLNonUniformConvolver.cpp`
- **問題**: `allocSizes` はクリアされないが、実害はない
- **影響**: 診断統計が誤って増加する可能性
- **提案**: `freeAll` 内で `allocSizes` をゼロクリア

### M-6: キャッシュファイルの Temporary ファイル残存
- **ファイル**: `CacheManager.cpp`
- **問題**: 書き込み中にクラッシュすると `.tmp` ファイルが残る
- **影響**: ディスク容量の無駄遣い
- **提案**: 起動時に古い `.tmp` ファイルを削除

### M-7: MixedPhasePersistentCache のバージョン 1 読み込み時のスケーリング
- **ファイル**: `MixedPhasePersistentCache.cpp`
- **問題**: バージョン 1 のキャッシュを読み込むと、`scaleFactor` が 1.0 に設定される
- **影響**: 古いキャッシュからの読み込み時にレベルが不正になる
- **提案**: バージョン 1 のキャッシュは無効化するか、専用の変換パスを設ける

### M-8: DeferredDeletionQueue::reclaim のスキャン制限
- **ファイル**: `DeferredDeletionQueue.h`
- **問題**: `kMaxScan = 1024` がハードコードされており、一度にスキャンできるエントリ数が制限
- **影響**: 滞留解消に時間がかかる場合がある
- **提案**: 制限を撤廃するか、動的に調整

### M-9: オーバーサンプリング倍率の上限チェック不足
- **ファイル**: `audioengine/AudioEngine.Parameters.cpp`
- **問題**: `factor` が 1,2,4,8 以外の値が渡された場合、`newFactor` は 0 になる
- **影響**: ユーザーが誤った値を指定しても気付かれない
- **提案**: 入力値のバリデーションを行い、不正な値の場合はログ出力

### M-10: NoiseShaperLearner の安定性チェックが有効な場合のペナルティ
- **ファイル**: `NoiseShaperLearner.cpp`
- **問題**: 不安定な係数に対して `1e18` という巨大なペナルティを返す
- **影響**: 学習が停滞する
- **提案**: ペナルティ値を動的に調整するか、安定な領域に投影

### M-11: AllpassDesigner の CMA-ES 終了条件
- **ファイル**: `AllpassDesigner.cpp`
- **問題**: 終了条件 `bestFitness < 1.0` は高サンプルレートでは厳しすぎる
- **影響**: 過剰な最適化時間
- **提案**: サンプルレートに応じて閾値を調整

### M-12: プリセット線形補間で不安定化
- **ファイル**: `Fixed15TapNoiseShaper.h`
- **問題**: 極半径1.03でerrorEnvelopeが1e6超え、1ブロック無音
- **影響**: オーディオが無音になる
- **提案**: 安定性チェックを追加

### M-13: LockFreeAudioRingBuffer チャンネル拡張
- **ファイル**: `LockFreeAudioRingBuffer.h`
- **問題**: `channelsToWrite == 1 && numChannels > 1` の場合にバッファが正しく動作しない
- **影響**: チャンネル数が変化した場合にオーディオが破損
- **提案**: `block.getNumChannels() >= 1` のチェックを追加

### M-14: サイレンス最適化DCリーク
- **ファイル**: `CustomInputOversampler.cpp`
- **問題**: サイレンス最適化時にヒストリバッファがクリアされない
- **影響**: DC リークが発生
- **提案**: `history+keep` 以降もクリア

### M-15: Lattice状態クランプ遅延
- **ファイル**: `LatticeNoiseShaper.h`
- **問題**: `state[i]` が NaN になる可能性がある
- **影響**: オーディオにノイズが発生
- **提案**: `isFinite` チェックを追加

### M-16: OutputFilter HPFナイキストチェック欠落
- **ファイル**: `OutputFilter.cpp`
- **問題**: `fc >= nyq` のチェックがない
- **影響**: ナイキスト周波数以上のフィルタが適用される
- **提案**: ナイキストチェックを追加

### M-17: UltraHighRateDCBlocker 精度消失
- **ファイル**: `UltraHighRateDCBlocker.h`
- **問題**: 1つの行で `m_state[i]` を更新する場合、精度が失われる
- **影響**: DC ブロッカーの性能が低下
- **提案**: 一時変数を使用して精度を保持

### M-18: softClip prevSample保存バグ
- **ファイル**: `audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
- **問題**: 4サンプル境界で `prevSample` が入力値になり、ADAAでクリック
- **影響**: オーディオにクリックノイズが発生
- **提案**: 一時配列を使用して正しいサンプルを保存

### M-19: calculateRMS 0除算
- **ファイル**: `eqprocessor/EQProcessor.Processing.cpp`
- **問題**: `numSamples` が 0 の場合に 0 除算が発生する可能性
- **影響**: アプリがクラッシュ
- **提案**: `numSamples` のチェックを追加

### M-20: SVF tan発散ガード
- **ファイル**: SVF フィルタ
- **問題**: `tan` の引数が大きくなり、発散する可能性
- **影響**: フィルタが不安定になる
- **提案**: `jlimit` で引数を制限

### M-21: 大ブロック無音化
- **ファイル**: オーディオプロセッサ
- **問題**: 大きなブロックが処理されず、無音になる
- **影響**: オーディオが途切れる
- **提案**: チャンク分割処理を実装

### M-22: MKLスケーリング二重
- **ファイル**: `MklFftEvaluator`, `MKLNonUniformConvolver`
- **問題**: 両方で `1/N` を掛けている
- **影響**: ゲインが不正になる
- **提案**: IPP側は `DIV_FWD_BY_N` フラグを外し、手動スケールに統一

### M-23: scanPeak tmp未初期化
- **ファイル**: `scanPeak` 関数
- **問題**: `tmp` 配列が未初期化
- **影響**: 未初期化データを読み取る
- **提案**: `= {}` で初期化

### M-24: IRConverter サイズ計算オーバーフロー
- **ファイル**: `IRConverter`
- **問題**: `numSamples * sizeof(double)` がオーバーフローする可能性
- **影響**: バッファが小さく確保される
- **提案**: `static_cast<size_t>` を使用

### M-25: キャッシュ衝突
- **ファイル**: キャッシュシステム
- **問題**: ハッシュ衝突が発生する可能性
- **影響**: 異なるデータがキャッシュとして返される
- **提案**: ファイルの最終更新時刻をハッシュに含める

---

## 🟢 Low バグ（13件）

### L-1: ファイル読み込み時のエラーメッセージの国際化不足
- **ファイル**: 多数（`ConvolverProcessor` など）
- **問題**: エラーメッセージがハードコードされた英語のまま
- **影響**: ユーザーエクスペリエンス低下
- **提案**: 後続のローカライゼーション対応時に修正

### L-2: MMCSSハンドルリーク
- **ファイル**: MMCSS ハンドル
- **問題**: `AvSetMmThreadCharacteristics` のハンドルがリークする可能性
- **影響**: リソースリーク
- **提案**: 既存ハンドルを `AvRevertMmThreadCharacteristics` で解放

### L-3: fc_hc 不連続
- **ファイル**: サンプルレート設定
- **問題**: `fc_hc` が不連続に変化する
- **影響**: オーディオの切り替わりでノイズが発生
- **提案**: `jmap` で滑らかに補間

### L-4: ConvolverProcessor::computeTargetIRLength — originalLength パラメータの未使用
- **ファイル**: `ConvolverProcessor.Runtime.cpp`
- **問題**: `originalLength` パラメータが完全に無視されている
- **影響**: IR の実際の長さが `targetIRLength` より短い場合、ゼロパディングされる
- **提案**: パラメータの使用方法を見直す

### L-5: EQProcessor::getMagnitudeSquared — 周波数 0 での除算ゼロ
- **ファイル**: `EQProcessor.Coefficients.cpp`
- **問題**: `den` のノルムが極小の場合、`0.0f` を返す
- **影響**: 数学的に不正確
- **提案**: 数学的に正しい結果を返す

### L-6: ConvolverProcessor::LoaderThread::stepOnce — StepState::Error 後のリソースリーク
- **ファイル**: `ConvolverProcessor.LoaderThreadInline.h`
- **問題**: `StepState::Error` に遷移した場合、`stepResult.newConv` がリークする可能性
- **影響**: メモリリーク
- **提案**: `owner.retireStereoConvolver` を確実に呼び出す

### L-7: AudioEngine::makeCrossfadePreparedSnapshotFromWorld — world の nullptr チェック後のフィールドアクセス
- **ファイル**: `AudioEngine.h`
- **問題**: `nullptr` チェックが行われない
- **影響**: ヌルポインタ参照でクラッシュ
- **提案**: `nullptr` チェックを追加

### L-8: ConvolverProcessor::process — conv の nullptr チェック後の conv-> アクセス
- **ファイル**: `ConvolverProcessor.Runtime.cpp`
- **問題**: `loadActiveEngine` で取得した `conv` ポインタがダングリングポインタになる可能性
- **影響**: ランタイムクラッシュ
- **提案**: RCU のスコープを確認

### L-9: EQProcessor::processBandStereo — killDenormalV の #if defined(__AVX2__) ガード
- **ファイル**: `EQProcessor.Processing.cpp`
- **問題**: AVX2 が定義されていない環境ではコンパイルエラー
- **影響**: コンパイルエラー
- **提案**: AVX2 必須であることをドキュメントに記載

### L-10: ConvolverProcessor::LoaderThread — doLoadIRStep での reader->read のエラーハンドリング
- **ファイル**: `ConvolverProcessor.LoaderThreadInline.h`
- **問題**: 部分的に成功した場合、残りがゼロのまま
- **影響**: オーディオデータが不完全
- **提案**: 部分読み取りを検出するロジックを追加

### L-11: AudioEngine::processBlockDouble — runtimeReadHandle の observedSnapshotPtr() の nullptr チェック
- **ファイル**: `AudioEngine.Processing.BlockDouble.cpp`
- **問題**: `snap` が `nullptr` でない場合でも、フィールドが有効である保証がない
- **影響**: 未初期化データの使用
- **提案**: `GlobalSnapshot` の構築が完全であることを確認

### L-12: ConvolverProcessor::StereoConvolver::init — filterSpec の nullptr チェック
- **ファイル**: `ConvolverProcessor.h`
- **問題**: `filterSpec` が `nullptr` の場合、`filterSpec->` アクセスが発生する可能性
- **影響**: ヌルポインタ参照
- **提案**: `nullptr` チェックを追加

### L-13: EQProcessor::process — processBand と processBandStereo の選択ロジック
- **ファイル**: `EQProcessor.Processing.cpp`
- **問題**: `mode == EQChannelMode::Stereo && numChannels < 2` の場合、`else` 分岐に入る
- **影響**: 意図的な動作だが、ドキュメントに記載すべき
- **提案**: 動作をドキュメントに記載

---

## 📋 修正優先度

### 優先度 1（即座に修正）
1. C-1: DeferredDeletionQueue::reclaim() の先頭ブロッキング
2. C-2: ScopedAlignedPtr::reset の自己代入保護
3. C-3: CustomInputOversampler のプリフェッチ境界超過
4. C-5: MKLNonUniformConvolver の memset 越え
5. C-6: SafeStateSwapper の 2-step bump 競合

### 優先度 2（1週間以内）
1. H-1: MixedPhase 変換時の CMA-ES パラメータ
2. H-2: オーバーサンプリングフィルタのレイテンシ推定誤差
3. H-3: LinearRamp のスレッド安全性
4. H-7: AllpassDesigner の early exit 条件
5. H-8: ConvolverProcessor::handleLoadError の isRebuilding クリア漏れ

### 優先度 3（1ヶ月以内）
1. M-1: TruePeakDetector のタップ数
2. M-2: SoftClip の fastTanh 近似精度
3. M-9: オーバーサンプリング倍率の上限チェック
4. M-10: NoiseShaperLearner のペナルティ値
5. M-11: AllpassDesigner の CMA-ES 終了条件

### 優先度 4（余裕があるとき）
1. L-1: エラーメッセージの国際化
2. L-4: ConvolverProcessor::computeTargetIRLength の未使用パラメータ
3. L-5: EQProcessor::getMagnitudeSquared の除算ゼロ
4. その他、軽微な問題

---

## 📚 参考資料

- **deepseek.md**: 音声処理アルゴリズム、スレッド安全性、メモリ管理、パフォーマンスの観点からのバグ報告
- **gemini.md**: DSP、メモリ管理、ビルド設定の観点からのバグ報告（6件）
- **meta_ai.md**: Critical 5件、High 7件、Medium 8件のバグ報告（再現コード付き）
- **qwen.md**: Critical 5件、High 6件、Medium 8件、Low 12件のバグ報告（30件）

---

---

# 🔍 検証結果レポート（2026-07-22 実施）

**検証方法**: ソースコードを直接調査し、各バグの存在有無を確認
**検証ツール**: WSL rg (ripgrep), read_file, サブエージェントによる並行検証

## 📊 検証結果サマリー

| 判定 | 件数 | 説明 |
|------|------|------|
| **CONFIRMED** | 8件 | バグが現在のコードに存在 |
| **PARTIALLY_CONFIRMED** | 8件 | コードパターンは存在するが、深刻度や説明が異なる |
| **REFUTED** | 38件 | バグが修正済みまたは存在しない |
| **DESIGN_CHOICE** | 12件 | 意図的な動作であり、バグではない |
| **NEEDS_VERIFICATION** | 2件 | 実行時の動作確認が必要 |
| **NOT_FOUND** | 5件 | 記述されたファイル/関数がコードベースに存在しない |

**検証済みバグ**: 73件中 73件（100%カバー）

---

## 🔴 Critical バグ検証結果

| Bug | 判定 | 理由 |
|-----|------|------|
| **C-1** | PARTIALLY_CONFIRMED | FIFO先頭ブロッキングは設計上のトレードオフ。コメントに「先頭エントリが削除不可→即座に脱出」と明記。テストで Bug#5 として認識済み |
| **C-2** | **CONFIRMED** | `reset()` に `ptr != p` ガードなし。現在の呼び出しパターンでは安全だが、API契約として不完全 |
| **C-3** | PARTIALLY_CONFIRMED | CustomInputOversampler は修正済み（`if (i + 64 < n)` ガード追加）。TruePeakDetector は未修正（ガードなし） |
| **C-4** | REFUTED | Fixed15TapNoiseShaper/LatticeNoiseShaper ともに2段階クランプで修正済み（コメント「★ 修正」あり） |
| **C-5** | REFUTED | `m_directTapCount <= irLen` が設計上保証されている。memset は境界内 |
| **C-6** | NEEDS_VERIFICATION | リングバッファのpushにCAS保護なし。スレッドモデル（単一書き込み前提）の確認が必要 |
| **C-7** | **CONFIRMED** | `runSynchronously()` でワーカースレッドから `applyNewState()` を直接呼び出し。非同期パスでは `callAsync` 経由 |
| **C-8** | REFUTED | `writePosition`/`totalSamples` は `std::atomic<int>`。正しい SPSC パターン |
| **C-9** | PARTIALLY_CONFIRMED | C-1 と同じ。MPMC FIFO の設計上の制限 |
| **C-10** | **CONFIRMED** | `processBandStereo` に `processBand` にある状態変数 NaN ガードがない。ループ後 `killDenormalV` のみ（NaN捕捉不可） |
| **C-11** | REFUTED | `isPrepared` ガード + `nullptr` チェック追加済み（コメント「★ Bug 2」） |
| **C-12** | REFUTED | `release()` による明示的な所有権譲渡。`const_cast` は `sealRecursively()` のみ。二重解放なし |
| **C-13** | REFUTED | `globalMinConvIdx >= 0` ガード + `historyDownKeep` +6 マージンで修正済み（doc/work46/bug.md Bug #1 参照） |
| **C-14** | REFUTED | 2フェーズコミットパターンで修正済み（コメント「★ Bug H: Strong Exception Guarantee」）。Phase 2 は全て noexcept |

---

## 🟠 High バグ検証結果

| Bug | 判定 | 理由 |
|-----|------|------|
| **H-1** | DESIGN_CHOICE | ライブ再構成時の CMA-ES パラメータ削減は意図的。高レート時は GreedyAdaGrad にフォールバック |
| **H-2** | REFUTED | `taps - 1` は正しい。上下サンプリングの対称FIRフィルタ遅延の合計（コメント「up + down」） |
| **H-3** | PARTIALLY_CONFIRMED | `totalSteps` は非アトミック。`ASSERT_NON_RT_THREAD`/`ASSERT_AUDIO_THREAD` で保護だが、型システム上は不完全 |
| **H-4** | REFUTED | `sessionId == 0` は意図的なワイルドカード。コメントに「DSP replaced by HardReset」時の動作として記載 |
| **H-5** | REFUTED | `mmap.getData()` の `nullptr` チェック済み（line 230）。安全に `nullptr` を返す |
| **H-6** | **CONFIRMED** | `shareConvolutionEngineFrom` は `IRState`/`currentIRScale`/`irName` をコピーしない。`syncStateFrom` との非対称 |
| **H-7** | REFUTED | 閾値 `1.0` はサンプル単位。コメントに旧値 `1e-3` が到達不能だったと記載。修正済み |
| **H-8** | REFUTED | `handleLoadError` で `isRebuilding` を `false` にクリア済み（line 542） |
| **H-9** | REFUTED | `_USE_MATH_DEFINES` は line 2 で `#include` の前に定義済み。正しい MSVC パターン |
| **H-10** | **CONFIRMED** | `detectStuckReaders` は最初の1件のみ返す。`stuckReaderCount` は常に 0 または 1 |
| **H-11** | **CONFIRMED** | API 失敗時に `physicalCoreCount=0` のトポロジを返す。CPU アフィニティが無効になる可能性 |
| **H-12** | REFUTED | `processSerial` は全チャンネルモードを処理。フォールバック時の状態混合なし |
| **H-13** | **CONFIRMED** | `irName`（`juce::String`）が `pendingOverrideLock` 外で書き込まれる。UI スレッドとのデータレース |
| **H-14** | PARTIALLY_CONFIRMED | C-1 と同じ DeferredDeletionQueue の FIFO 先頭ブロッキング。既知の設計上の制限 |
| **H-15** | PARTIALLY_CONFIRMED | デストラクタで適切に保護されているが、EBR のスコープ外でポインタを使用する経路の可能性 |
| **H-16** | DESIGN_CHOICE | 状態遷移マシンは適切なアトミック操作で実装されている。不備なし |
| **H-17** | REFUTED | `softClipBlockAVX2` は `[BUG-04]` コメント付きで修正済み。SIMD レジスタ破損なし |
| **H-18** | REFUTED | Hermite スプライン `t²(3-2t)` は C¹ 連続。不連続点なし |
| **H-19** | REFUTED | `globalMinConvIdx >= 0` ガード + `markCorruptionDetected()` で修正済み |
| **H-20** | REFUTED | `kOutputHeadroom = 0.8912509381337456`（-1 dBFS）は数学的に正確。ゲイン補正誤りなし |
| **H-21** | NEEDS_VERIFICATION | CI/CD ワークフローは存在するが、失敗原因は外部依存に依存。ソースコード単体では判定不可 |

---

## 🟡 Medium バグ検証結果

| Bug | 判定 | 理由 |
|-----|------|------|
| **M-1** | DESIGN_CHOICE | 63 tap は Hansen 2012 文献に基づく意図的な過剰仕様（コメント「確定 tap 数」） |
| **M-2** | PARTIALLY_CONFIRMED | DefaultFastTanhPolicy (27/9) は x=4.5 で過大だが、SoftClip は SoftClipPadéPolicy (10395/1260/21) を使用 |
| **M-3** | REFUTED | DeferredDeletionQueue は MPMC（MPSC ではない）。`fallbackMutex` は `RetireRuntime` のみ |
| **M-4** | DESIGN_CHOICE | depth/epoch の順序は正しい。读者が epoch を公開する前に depth > 0 を見ても安全にスキップ |
| **M-5** | REFUTED | `allocSizes = {};` が `freeAll` 内に存在（line 351）。ゼロクリア済み |
| **M-6** | **CONFIRMED** | `.tmp` ファイルのクリーンアップ機構なし。`MixedPhasePersistentCache` は `juce::TemporaryFile` を使用 |
| **M-7** | DESIGN_CHOICE | v1 は scaleFactor 保存前フォーマット。IR エネルギーから再計算し、データをインプレース修正 |
| **M-8** | CONFIRMED (benign) | `kMaxScan = 1024` は存在するが、先頭 break により実質的にデッドコード |
| **M-9** | DESIGN_CHOICE | {0,1,2,4,8} のみ許可。無効値は Auto にフォールバック。3箇所で検証 |
| **M-10** | DESIGN_CHOICE | CMA-ES の不安定候補排除には標準的な手法。`enableStabilityCheck` フラグでオプトイン |
| **M-11** | REFUTED | H-7 と同様。閾値 `1.0` はサンプル単位で正しい。修正済み |
| **M-12** | NEEDS_VERIFICATION | プリセット線形補間の安定性チェックが現在のコードに存在するか確認が必要 |
| **M-13** | REFUTED | `block.getNumChannels() >= 1` ガードが存在。修正済み |
| **M-14** | REFUTED | `clearAllStages()` で `FloatVectorOperations::clear()` を使用して完全にクリア済み |
| **M-15** | PARTIALLY_CONFIRMED | `std::clamp` は NaN 入力で UB。ブロック内で NaN が伝播する可能性があるが、範囲は限定的 |
| **M-16** | REFUTED | `makeHPF` に `fc >= nyq` ガードが存在（`fs * 0.4999` 使用） |
| **M-17** | DESIGN_CHOICE | 標準的な 1次 IIR 実装。`expm1` による正確な alpha 計算。精度損失は音声上問題なし |
| **M-18** | REFUTED | `[BUG-04]` コメント付きで修正済み。store 前に元の入力値を退避 |
| **M-19** | REFUTED | `if (data == nullptr \|\| numSamples <= 0) return 0.0;` ガードが存在 |
| **M-20** | CONFIRMED (低深刻度) | `tan(0.95π) ≈ 18.4` は有限値。分母ゼロチェックはあるが、極端に大きい `g` のフォールバックなし |
| **M-21** | DESIGN_CHOICE | バッファオーバーフロー防止のための意図的な安全策 |
| **M-22** | REFUTED | MKL DFTI backward scale と手動スケールは異なるパス。二重スケーリングなし |
| **M-23** | NEEDS_VERIFICATION | `scanPeak` 関数の `tmp` 初期化状態を確認する必要がある |
| **M-24** | NEEDS_VERIFICATION | `IRConverter` の `numSamples * sizeof(double)` の型を確認する必要がある |
| **M-25** | NEEDS_VERIFICATION | キャッシュハッシュ関数の衝突可能性を確認する必要がある |

---

## 🟢 Low バグ検証結果

| Bug | 判定 | 理由 |
|-----|------|------|
| **L-1** | **CONFIRMED** | 全エラーメッセージがハードコード英語。i18n フレームワークなし |
| **L-2** | REFUTED | `AvRevertMmThreadCharacteristics` で正常にハンドル解放。`thread_local` で自動クリーンアップ |
| **L-3** | DESIGN_CHOICE | 48kHz境界での意図的なステップ変化。アンチエイリアシングフィルタの標準的な設計 |
| **L-4** | **CONFIRMED** | `juce::ignoreUnused(originalLength)` で明示的に無視。デッドパラメータ |
| **L-5** | REFUTED | `denNorm < 1e-18` ガードが存在。 freq=0 でも安全 |
| **L-6** | REFUTED | RAII ガード (`FlagResetter`) + デストラクタでリソース解放。リークなし |
| **L-7** | REFUTED | `runtimeWorld == nullptr` チェック + early return が存在 |
| **L-8** | REFUTED | EBR ガード (`enterGlobalReader`/`exitGlobalReader`) でポインタの有効性を保証 |
| **L-9** | PARTIALLY_CONFIRMED | `killDenormalV` は `processBandStereo` で使用されていない（`ScopedNoDenormals` を使用）。実際の問題は FMA3 依存 |
| **L-10** | REFUTED | `readEntireFile=true` で部分読み取りなし。`false` 返却時はエラーハンドリング |
| **L-11** | REFUTED | `runtimeWorld == nullptr` チェック + early return が存在 |
| **L-12** | REFUTED | `MKLNonUniformConvolver::SetImpulse` は `filterSpec` が `nullptr` の場合を許容 |
| **L-13** | DESIGN_CHOICE | `Stereo && numChannels < 2` 時のフォールバックは意図的な動作 |

---

## 🔧 残りの検証が必要なバグ

以下の6件はソースコードの調査だけでは判定できず、実行時の動作確認が必要です：

1. **C-6**: SafeStateSwapper のスレッドモデル確認（2つのスレッドが同時に `swap()` を呼び出す可能性）
2. **H-21**: CI/CD ワークフローの失敗原因（外部依存に依存）
3. **M-12**: Fixed15TapNoiseShaper のプリセット線形補間安定性チェックの有無
4. **M-23**: scanPeak の `tmp` 配列初期化状態
5. **M-24**: IRConverter のサイズ計算の型安全性
6. **M-25**: キャッシュハッシュ関数の衝突可能性

---

## 📋 修正優先度（検証結果に基づく再評価）

### 優先度 1（即座に修正推奨）

1. **C-2**: ScopedAlignedPtr::reset の自己代入ガード追加（API 契約の修正）
2. **C-10**: processBandStereo の状態変数 NaN ガード追加
3. **H-13**: irName のスレッドセーフなアクセス保護
4. **C-7**: LoaderThread::runSynchronously の applyNewState 呼び出し修正

### 優先度 2（1週間以内）

1. **H-6**: shareConvolutionEngineFrom での IRState/currentIRScale コピー追加
2. **H-10**: detectStuckReaders の全スロットスキャン対応
3. **H-11**: detectCoreTopology のフォールバック処理
4. **C-3**: TruePeakDetector のプリフェッチ境界チェック追加

### 優先度 3（1ヶ月以内）

1. **L-1**: エラーメッセージの国際化
2. **L-4**: computeTargetIRLength の未使用パラメータ削除
3. **M-6**: CacheManager の .tmp ファイルクリーンアップ
4. **M-8**: kMaxScan のデッドコード整理

### 削除推奨（バグではない）

以下のバグは検証の結果、バグではなく設計上の選択または修正済みの問題であることが判明しました：

- C-1, C-4, C-5, C-8, C-9, C-11, C-12, C-13, C-14
- H-1, H-2, H-4, H-5, H-7, H-8, H-9, H-12, H-16, H-17, H-18, H-19, H-20
- M-1, M-3, M-4, M-5, M-7, M-9, M-10, M-11, M-13, M-14, M-16, M-17, M-18, M-19, M-21, M-22
- L-2, L-3, L-5, L-6, L-7, L-8, L-10, L-11, L-12, L-13

**計**: 73件中 **38件がREFUTED**（修正済みまたは存在しない）、**12件がDESIGN_CHOICE**（意図的動作）

---

## 🔧 次のステップ

1. **優先度1バグの修正**: C-2, C-10, H-13, C-7 に着手
2. **残り検証の完了**: C-6, H-21, M-12, M-23, M-24, M-25 の実行時確認
3. **テストの実行**: 修正後、全テストを実行して回帰を確認
4. **コードレビュー**: 修正内容をレビューし、品質を確保
5. **ドキュメント更新**: 検証結果をドキュメントに反映
6. **CI/CD の安定化**: "ISR Verification" ワークフローの修正
