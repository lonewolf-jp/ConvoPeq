# ConvoPeq — Intel MKL → Intel IPP 完全移行 改修計画書

**Rev.2**
**対象リポジトリ**: ConvoPeq (JUCE 8.0.12 / C++20 / Windows 11 x64 / MSVC 19.44+ or Intel icx oneAPI 2026.0)
**対象ソース**: `ConvoPeq.md`(Rev.2時点: 252ファイル、72,052行、連結ソース)の静的解析に基づく
**調査範囲**: `mkl` を含む全参照(36ファイル)を個別に追跡し、呼び出しスレッド(Audio Thread / Message Thread / Loader Thread / 非RT Worker Thread)、確保・解放の対応関係、ビルド設定への影響まで検証済み

### 改訂履歴

- **Rev.1**: 初版(249ファイル、71,819行時点)。
- **Rev.2**(本版): 更新されたソース(252ファイル、72,052行、新規+3ファイル・変更22ファイル)に合わせて内容を全面照合。差分は新旧ソースをファイル単位で機械的にdiffして検出し、目視では見落としのない形で反映した。主な変更点:
  - `MKLNonUniformConvolver.cpp/.h` が **AoS(`irFreqDomain`/`fdlBuf`)を使い捨てスクラッチへ縮小し、SoA(`irFreqReal/Imag`・`fdlReal/Imag`)を唯一の永続データ表現へ一本化する大規模リファクタ([Mem-Fix])** を受けた。MKL API自体の使用箇所(`mkl_malloc`/`mkl_free`/`vdMul`/`cblas_dscal`)は変わらないが、呼び出しパターンとバッファサイズが変化したため §1.2(B)・§5 Phase 2・§9.1 を更新した(詳細後述)。
  - `AlignedAllocation.h` に `aligned_malloc_nothrow`/`makeAlignedArray_nothrow`(いずれも内部は引き続き`mkl_malloc`)が追加された。Phase 1の対象に追加。
  - `CMakeLists.txt` に `src/CpuFeatureCheck.cpp` の追加とclang-tidy CI強制化が入ったが、**MKL/IPP関連の設定変更は無い**ため、本計画のPhase 0/5の記述に変更は不要。
  - 新規ファイル `src/CpuFeatureCheck.cpp/.h`(AVX2ランタイム検出)・`src/audioengine/SimplePeakLimiter.h`(ピークリミッター)は、内容を確認した上で**MKL/IPP参照が皆無であることを確認済み**。本移行とは無関係な並行作業のため計画本体には含めないが、§1.3に文脈として記録する。
  - `DspNumericPolicy.h`/`LatticeNoiseShaper.h`(`std::isnan()`をビットパターン判定`isFinite()`へ統合)、`AudioEngine.*`各ファイル・`EQProcessor.Processing.cpp`・`ConvolverProcessor.Rebuild/Runtime.cpp`の変更は、確認の結果**MKL/IPPを一切参照しないため本計画のスコープ外**と判断した。
  - `ConvolverProcessor.Lifecycle.cpp`(§2.1のmkl_free×3箇所)・`ConvolverProcessor.LoadPipeline.cpp`(cblas_dscal×2箇所)・`ConvolverProcessor.MixedPhase.cpp`・`ConvolverProcessor.ResampleAndFallback.cpp`は、MKL関連行そのものに変更が無いことを確認済み(前者2ファイルは無関係な安全ガードの追加のみ、後者2ファイルはバイト単位で無変更)。**§2.1のリスク分析、§4のAPI対応表、§6のR2C/C2R最適化提案は全てそのまま有効。**

---

## 0. エグゼクティブサマリー

### 0.1 最初に共有すべき事実

**現在の Audio Thread(`processLayerBlock` / `processDirectBlock` / `Add`)内には MKL 呼び出しが一切存在しない。** 2026年4月の先行移行(`MKLNonUniformConvolver`・`MklFftEvaluator` の FFT バックエンド換装)により、リアルタイム経路の FFT は既に `ippsFFTFwd_RToCCS_64f` / `ippsFFTInv_CCSToR_64f` へ移行済みで、これは `MKLNonUniformConvolver.cpp` 内のコメントでも明示されている(`applySpectrumFilter ─ Message Thread のみ` 等の注記が全ての残存 MKL 呼び出しに付与されている)。

したがって本移行の性質は「オーディオスレッドを壊さないための綱渡り」ではなく、以下の4点に整理される。

1. **残存する非RT MKL依存の完全除去**(IR ロード・パイプライン、UI操作、学習/最適化バックグラウンド処理、起動時初期化)
2. **ビルド依存の除去**(`find_package(MKL REQUIRED)` を必須から解放し、CI/開発機のセットアップコストとライセンス制約を軽減)
3. **潜在バグの是正**(調査中に発見した、アロケータとデアロケータがファイルをまたいで不整合を起こしているケース — 詳細は §2.1)
4. **性能の最大化**(非RTとはいえ IR ロード時間はユーザー体感に直結するため、対称性を活用したアルゴリズム最適化を提案 — 詳細は §6)

### 0.2 スコープの数値

| 分類 | ファイル数 | 代表例 |
|---|---|---|
| 既に IPP 移行済み(要クリーンアップのみ) | 2 | `MklFftEvaluator.h`, `MKLNonUniformConvolver.cpp/h`(FFT部) |
| 非RT・IRロード/UI経路のMKL直接呼び出し | 9 | `CacheManager.cpp`, `IRConverter.cpp`, `ConvolverProcessor.MixedPhase.cpp`, `ConvolverProcessor.ResampleAndFallback.cpp`, `ConvolverProcessor.StateAndUI.cpp`, `ConvolverProcessor.LoadPipeline.cpp`, `ConvolverProcessor.Lifecycle.cpp`, `SpectrumAnalyzerComponent.cpp/h` |
| メモリ確保基盤(全体に影響) | 1 | `AlignedAllocation.h` |
| RNG(非RT) | 1 | `PsychoacousticDither.h` |
| 起動時/スレッド初期化 | 3 | `MainApplication.cpp`, `MKLRealTimeSetup.cpp/h`, `AudioEngine.RebuildDispatch.cpp` |
| 学習・最適化(オフライン) | 3 | `NoiseShaperLearner.cpp/h`, `AllpassDesigner.cpp/h` |
| ビルド設定 | 2 | `CMakeLists.txt`, `build.bat` |
| 型宣言・コメントのみ(実処理なし) | 15 | `DftiHandle.h`, `ConvolverProcessor.h`, `ConvolverProcessor.Internal.h`, `ConvolverProcessor.LoaderThread.cpp`, `DeferredFreeThread.h`, `RuntimeBuilder.cpp/h`, `ConvolverState.h`, `DspNumericPolicy.h`, `UltraHighRateDCBlocker.h`, `AudioEngine.h`, `FrozenRuntimeWorld.h` 等 |

### 0.3 最重要リスク(3行)

1. **アロケータ/デアロケータ不整合**: `IRConverter.cpp`・`CacheManager.cpp` が `mkl_malloc` で直接確保したバッファが `PreparedIRState::~PreparedIRState()` の `convo::aligned_free()` で解放される、というファイルをまたいだ非対称なペアが3箇所存在する。今日は両者とも実体が `mkl_malloc`/`mkl_free` なので問題が顕在化していないだけで、`AlignedAllocation.h` だけを先に IPP 化すると **ヒープ破壊** に直結する。
2. **IPP の 64byte アライメント保証は版とドキュメントで揺れがある**(§2.2)。`mkl_malloc(size, 64)` のように明示的にアライメントを指定できる API が IPP の `ippsMalloc_*` には存在しない。
3. **`ippvm`(Ln/Exp/Tanh 等)は IPP の CMake Config パッケージで component 定義漏れの既知バグ報告がある**(§2.3)。ビルド設定変更は本番コードに着手する前に単独で検証すべき。

---

## 1. 現状分析(As-Is)

### 1.1 既に完了している移行の実装パターン(踏襲すべき手本)

`MklFftEvaluator.h` と `MKLNonUniformConvolver.cpp` の Audio Thread 経路は、以下のパターンで既に IPP 化されており、これは本計画の全ての新規 FFT 移行が踏襲すべき「実績のある型」である。

```cpp
// 1) サイズ問い合わせ → 2) スペック/ワークバッファ確保 → 3) 初期化 → 4) 実行 → 5) 解放
int sizeSpec=0, sizeInit=0, sizeWork=0;
ippsFFTGetSize_R_64f(order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast, &sizeSpec, &sizeInit, &sizeWork);
fftSpecBuf = ippsMalloc_8u(sizeSpec);
Ipp8u* initBuf = (sizeInit > 0) ? ippsMalloc_8u(sizeInit) : nullptr;
ippsFFTInit_R_64f(&fftSpec, order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast, fftSpecBuf, initBuf);
if (initBuf) ippsFree(initBuf);
// Audio Thread:
ippsFFTFwd_RToCCS_64f(timeBuf, ccsBuf, fftSpec, fftWorkBuf);
ippsFFTInv_CCSToR_64f(ccsBuf, timeBuf, fftSpec, fftWorkBuf);
// 解放:
ippsFree(fftSpecBuf); ippsFree(fftWorkBuf);
```

加えて `MKLNonUniformConvolver.cpp` 内の `IppFFTPlanCache`(FFTサイズ=`order`をキーに `IppsFFTSpec_R_64f*` を `std::unordered_map` でキャッシュし、`ASSERT_NON_RT_THREAD()` で非RTスレッドからのみ生成を許可する設計)は、複数の畳み込みパーティションサイズで同一プランを共有する優れた設計であり、本計画で新たに IPP 化する `MixedPhase.cpp` / `ResampleAndFallback.cpp` / `SpectrumAnalyzerComponent` の各 FFT 経路でも **同一クラスを共有ヘッダへ切り出して再利用する**ことを提案する(§6.4)。

また `MklFftEvaluator.h` のコンストラクタ・`evaluate()` に付されたコメント「IPP は完全シングルスレッド設計のため、mkl_set_num_threads_local(1) の呼び出しは不要」は、IPP 移行後にスレッド数制御コードそのものが不要になることを示す先例であり、`MKLNonUniformConvolver` コンストラクタの `mkl_set_num_threads(1)` も同様に撤去対象となる(§5 Phase 2)。

### 1.2 MKL依存ファイルの全体マップ

以下、36ファイルを実行コンテキスト別に分類する。**「RTスレッド」列が Yes の行は存在しない**(§0.1参照)。

#### (A) メモリ確保基盤 ― 全体に波及

| ファイル | 内容 | RTスレッド |
|---|---|---|
| `src/AlignedAllocation.h` | `mkl_malloc`/`mkl_free` を包む `aligned_malloc`/`aligned_free`、`MKLAllocator`(STLアロケータ)、`ScopedAlignedPtr`、`aligned_unique_ptr`、`makeAlignedArray` 等、**コードベース全体が経由する唯一の確保口**。**[Rev.2で追加]** 例外を投げない `aligned_malloc_nothrow`/`makeAlignedArray_nothrow`(内部は同じく`mkl_malloc`、失敗時は`nullptr`を返す契約)も追加された。呼び出し元は現時点のソースにはまだ存在しない | No(呼び出し元は全て非RT。RTスレッドは確保済みバッファを読み書きするのみ) |

#### (B) 既にIPP移行済み・残作業はクリーンアップのみ

| ファイル | 残存するMKL要素 | 対応 |
|---|---|---|
| `src/MklFftEvaluator.h` | `#include <mkl.h>`(死んだinclude。実処理は全て`ippsFFT*`/`ippsMalloc_8u`/`ippsFree`) | includeを削除するのみ |
| `src/MKLNonUniformConvolver.h/.cpp` | FFT(`DftiComputeForward/Backward`)は既に`ippsFFTFwd/Inv_*CCSToR_64f`へ移行済み。残りは `mkl_malloc`/`mkl_free`(全て `SetImpulse()` 内、IRロード時)、`vdMul`(`applySpectrumFilter`と`SetImpulse`のテール処理、Message Thread限定とコメントで明示)、`cblas_dscal`(`SetImpulse`内のIRスケーリング)、`mkl_set_num_threads(1)`(コンストラクタで1回のみ)。**[Rev.2で変更]** `irFreqDomain`/`fdlBuf`(AoS)が全パーティション分の永続データから1パーティション分/2スロット分の使い捨てスクラッチへ縮小され、`irFreqReal/Imag`・`fdlReal/Imag`(SoA)が唯一の永続データ表現に一本化された([Mem-Fix])。これに伴い`applySpectrumFilter`・テール減衰ゲイン適用の`vdMul`は「interleaved配列1本×1回」から「re/im各配列に対しループ2回」の形に変わり、事前のinterleave/事後のdeinterleave処理そのものが不要になった(コード単純化)。パーティション逆順並び替え時にAoS用一時バッファ(`swapDomain`)を確保していた`mkl_malloc`/`mkl_free`のペアは、AoSが永続データでなくなったため**削除された**(移行対象が1組減った)。またAVX2非対応環境向けのスカラーフォールバック分岐(`kEnableSplitComplexKernel`)自体が撤去され、SoA/AVX2経路のみの単一実装になった(§1.3参照)。API自体の対応関係(§4)に変更は無い | §5 Phase 1/2 |

#### (C) 非RT・IRロード/UI経路のMKL直接呼び出し

| ファイル | 関数 | 実行コンテキスト | MKL API |
|---|---|---|---|
| `src/convolver/ConvolverProcessor.MixedPhase.cpp` | `convertToMixedPhase`/`convertToMixedPhaseAllpass` | Loaderスレッド(`shouldExit`キャンセル対応あり) | `DftiCreateDescriptor`系(`DFTI_DOUBLE, DFTI_COMPLEX`)、`MKL_Complex16`、`MKLAllocator` |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | `convertToMinimumPhase` | Loaderスレッド | 同上 + `vzAbs`/`vdLn`/`vzExp`(実ケプストラム法によるミニマムフェーズ再構成、§6.1で詳述) |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | `createFrequencyResponseSnapshot` | Message Thread(`applyComputedIR`から呼び出し、`JUCE_ASSERT_MESSAGE_THREAD`で保証) | `DftiCreateDescriptor`(`DFTI_SINGLE, DFTI_COMPLEX`)、`vcAbs`、`MKL_Complex8` |
| `src/SpectrumAnalyzerComponent.cpp/.h` | `prepareFFT`/`updateSpectrum`系 | UIタイマー(`startTimerHz`、Message Thread) | 同上パターン(単精度複素in-place FFT) |
| `src/CacheManager.cpp` | `copyFromMmapToAligned`(未使用/§9.2参照)、`loadPreparedState` | Loaderスレッド(キャッシュファイル読込) | `mkl_malloc`、`cblas_ddot`、`cblas_dscal` |
| `src/IRConverter.cpp` | `computeScaleFactor`、`convertFile` | Loaderスレッド | `cblas_ddot`、`mkl_malloc`/`mkl_free` |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | `applyComputedIR` | Message Thread(`JUCE_ASSERT_MESSAGE_THREAD`で保証) | `cblas_dscal` ×2 |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp` | IRState破棄用ラムダ、`releaseResources`等 | 非RT(prepareToPlay/releaseResources内) | `mkl_free` ×3(§2.1でクロスファイル不整合を指摘) |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | エラーメッセージ文字列のみ | ― | 実処理なし(`#include <mkl.h>`のみ) |

#### (D) RNG(乱数)

| ファイル | 内容 | RTスレッド |
|---|---|---|
| `src/PsychoacousticDither.h` | `vslNewStream(VSL_BRNG_SFMT19937)`、`vdRngUniform` によるTPDFディザ生成。**ただしAudio Threadは`popUniformFromRing()`でリングバッファから読み出すのみ**。実際のMKL呼び出しは`fillChunkForChannel`/`refillRandomRingNonRt()`という非RTワーカー専用関数の中にのみ存在する、SPSCリングバッファ設計 | No |

#### (E) 起動時/スレッド初期化

| ファイル | 内容 |
|---|---|
| `src/MKLRealTimeSetup.cpp/.h` | `mkl_set_num_threads(1)`、`mkl_set_dynamic(0)`、環境変数`MKL_NUM_THREADS`強制。`MainApplication::initialise()`から一度だけ呼ばれる |
| `src/MainApplication.cpp` | `MKLRealTime::setup()` と `ippInit()` を両方呼び出し、`vmlSetMode(VML_FTZDAZ_ON \| VML_ERRMODE_IGNORE)` をメインスレッドに設定 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | `rebuildThreadLoop()` 冒頭で `vmlSetMode(...)` を再設定(スレッドローカルのため) |

#### (F) 学習・最適化(オフライン、ユーザートリガー)

| ファイル | 内容 | 備考 |
|---|---|---|
| `src/NoiseShaperLearner.cpp/.h` | `vdTanh`(CMA-ES候補集団のtanh写像)、`MklFftEvaluator`利用(既にIPP化済み、§1.1(B)参照) | ノイズシェーパー係数学習ワーカースレッド |
| `src/AllpassDesigner.cpp/.h` | `MKLAllocator`のみ使用(FFT/VML/BLAS直呼び出しなし) | ミックスフェーズ用オールパス設計のグリッドサーチ/勾配降下。`AlignedAllocation.h`移行のみで自動的に追従 |

#### (G) ビルド設定

| ファイル | 内容 |
|---|---|
| `CMakeLists.txt` | `find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)`(**REQUIRED**=MKL無しではビルド不能)、`MKL::MKL`リンク、`JUCE_DSP_USE_INTEL_MKL=1`定義、`RuntimePublicationCoordinatorTests`/`PartialPublicationRejectTests`への個別MKLリンク、既存の`find_package(IPP QUIET ...)`(ippcore+ippsのみ、ippvm未指定) |
| `build.bat` | `setvars.bat`実行(MKL/IPP共通のoneAPI環境変数設定)、未検出時のエラーメッセージがMKL名指しの文言 |

#### (H) 型宣言・コメントのみ(コード変更不要、または軽微)

`DftiHandle.h`(RAIIラッパー、§5 Phase3で置換対象)、`ConvolverProcessor.h`(型宣言・include文)、`ConvolverProcessor.Internal.h`(`<mkl.h>`/`<mkl_vml.h>`のinclude文のみ、実呼び出しなし)、`DeferredFreeThread.h`(設計思想コメントのみ)、`RuntimeBuilder.cpp/.h`(`BuildError::MKLFailure`という列挙子名)、`ConvolverState.h`・`DspNumericPolicy.h`・`UltraHighRateDCBlocker.h`・`audioengine/AudioEngine.h`・`audioengine/FrozenRuntimeWorld.h`(いずれもコメント内言及のみ)。

### 1.3 Rev.2で確認した、本移行とは無関係な変更(参考情報)

Rev.2のソースには本移行のスコープ外の変更も含まれていたため、内容を確認した上で「関係が無いこと」を明示的に記録しておく(単に見落としたのではないことを示すため)。

- **AVX2必須化の強化**: `MKLNonUniformConvolver.h`に`#ifndef __AVX2__ #error ...`というコンパイル時ガードが追加され、`src/CpuFeatureCheck.cpp/.h`(新規)が`MainApplication::initialise()`冒頭でランタイムのAVX2/FMA対応チェックを行い、非対応CPUでは起動時にメッセージボックスを表示して終了するようになった。これは以前から把握していた「AVX2/FMA3の無条件要求」という課題に対する対応と見られるが、**MKL/IPPのいずれも参照しない**独立した変更であり、本計画には影響しない。むしろ「対象ハードウェアはAVX2が前提でAVX-512は前提外」という§2.2のアライメント議論の前提を裏付ける材料になっている。
- **`src/audioengine/SimplePeakLimiter.h`(新規)**: シンプルなピークリミッター機能。MKL/IPP/メモリ確保関連の参照は皆無。
- **`std::isnan()`の統合**: `DspNumericPolicy.h`に`isFinite`/`absNoLibm`等がビットパターン判定として追加され、`LatticeNoiseShaper.h`がこれを使うよう変更された(`/fp:fast`環境での`std::isnan()`不整合への対応)。両ファイルともMKL/IPP参照は無く、本計画のスコープ外。
- **`AudioEngine.*`各ファイル・`EQProcessor.Processing.cpp`・`ConvolverProcessor.Rebuild.cpp`/`Runtime.cpp`**: いずれも差分を確認したが、ピークリミッター統合・安全ガード追加等でありMKL/IPP参照は皆無。
- **`ConvolverProcessor.Lifecycle.cpp`/`LoadPipeline.cpp`**: コメント追加・安全ガード追加のみで、§2.1で指摘した`mkl_free`(3箇所)・`cblas_dscal`(2箇所)の行そのものに変更は無い。
- **`ConvolverProcessor.MixedPhase.cpp`/`ResampleAndFallback.cpp`**: Rev.1からバイト単位で無変更。§6.1のR2C/C2R最適化提案はそのまま有効。

---

## 2. 重大リスクと設計判断(着手前に必ず共有する事項)

### 2.1 [最重要] アロケータ/デアロケータのクロスファイル不整合

調査の過程で、`mkl_malloc`/`mkl_free` の**生呼び出し**(`convo::aligned_malloc`/`aligned_free` ラッパーを経由しない箇所)を全数洗い出したところ、以下の対応表が判明した。

| # | 確保箇所(生 `mkl_malloc`) | 解放箇所 | 解放方法 | 状態 |
|---|---|---|---|---|
| 1 | `IRConverter::convertFile()` の `data`(→ `prepared->partitionData`) | `PreparedIRState::~PreparedIRState()` / `operator=` | `convo::aligned_free(partitionData)` | **不整合**(確保=生mkl_malloc、解放=ラッパー経由) |
| 2 | `CacheManager::loadPreparedState()` の `copied`(→ `prepared->partitionData`) | 同上 | 同上 | **不整合** |
| 3 | `CacheManager::copyFromMmapToAligned()` の戻り値 `dst` | 呼び出し元なし(§9.2, デッドコード) | ― | 実害なし(未使用関数) |
| 4 | `convo::aligned_make_unique<IRState>()`(→ `aligned_malloc`経由で確保) | `ConvolverProcessor.Lifecycle.cpp` 内のカスタムdeleterラムダ、および2箇所の直接呼び出し | 生 `mkl_free(state)` / `mkl_free(oldIrState)` ×2 | **不整合(逆方向)**(確保=ラッパー経由、解放=生mkl_free) |

**なぜ今日は壊れていないのか**: `convo::aligned_free()` は今のところ `mkl_free()` を呼ぶだけの薄いラッパーであり、実体が同一だからである。

**なぜ危険か**: `AlignedAllocation.h` だけを IPP(または `_aligned_malloc`)へ先行して切り替え、上記4箇所の生呼び出しをそのまま放置すると、「`ippsMalloc_8u` で確保した領域を `mkl_free` で解放する」「`_aligned_malloc` で確保した領域を `mkl_free` で解放する」といった**ヒープマネージャ不一致**が発生する。Windows上でのヒープ破壊は即座にクラッシュするとは限らず、別の無関係な確保操作が失敗する形で遅延発現することが多く、原因特定が極めて困難になる。

**是正方針**: 本移行の第一歩として、上記4箇所の生 `mkl_malloc`/`mkl_free` 呼び出しを **全て `convo::aligned_malloc`/`convo::aligned_free` 経由に統一する**(関数シグネチャ自体は変えず、内部実装を差し替えるだけで済むようにする)。これにより「実装を1箇所(`AlignedAllocation.h`)だけ変更すればよい」という状態を先に作ってから、その1箇所の中身をIPP(または後述の`_aligned_malloc`)に差し替える、という順序を厳守する。詳細は §5 Phase 1。

### 2.2 [重要] 64byteアライメント保証の非対称性

`mkl_malloc(size, alignment)` は**呼び出し側が任意のアライメント値を明示指定できる**API である。ConvoPeqは一貫して `64` を指定しており(`AlignedAllocation.h`のデフォルトテンプレート引数も`Alignment = 64`)、コメントには「MKL/AVX-512用に64byteアライメントを保証する」という記述が複数箇所にある。

一方、IPPの `ippMalloc`/`ippsMalloc_<type>` には**アライメント値を指定する引数が存在しない**。実際のアライメント保証値についても資料間で食い違いが確認された。

- IPP 7〜8世代のリファレンスマニュアル(`ipps_manual`系)には「32-byte boundary」と明記されているものが複数存在する。
- 一方でIntel公式コミュニティ回答(2016年)では「ドキュメントは古く、AVX-512(512bit=64byte)対応のため現在は64-byteに変更済み」との説明がある。
- 2017年版Developer Guideには「AVX非対応=16byte、AVX/AVX2=32byte、MIC(AVX-512)=64byte」という**CPU依存の可変長**という記述もある。

つまり「ConvoPeqが実際にビルド・実行される環境(AVX2/FMA3が前提、AVX-512は前提外)において、リンクしたIPPのバージョンとビルド設定で本当に64byte保証が得られるか」は、ドキュメントだけからは断定できない。`ippsMalloc`はアライメント不足時にエラーを返す仕組みも無いため、不足していても検出されずに「たまたま動く」状態になりかねない。

**推奨(デフォルト)**: `AlignedAllocation.h`の確保・解放を、IPPではなく **Windows/MSVC・Intel icx共通のCRT関数 `_aligned_malloc(size, alignment)` / `_aligned_free(ptr)`**(`<malloc.h>`)に置き換える。理由:
1. アライメント値を`mkl_malloc`と全く同じ引数で明示指定でき、意味論が完全に一致する(ドロップイン置換)。
2. OS/CRTレベルの契約であり、IPPのバージョンや将来のアップデートに左右されない。
3. MKLへの依存は完全になくなる(「MKLをIPPへ移行する」という要件は、周辺のBLAS/VML/FFT相当機能をIPPへ寄せることで満たされ、アライメント確保という一機能だけは「ツールキット非依存のOS標準機構」に切り出す、という判断)。

**代替案(IPP純化を優先する場合)**: `ippsMalloc_8u`/`ippsFree`を採用してもよいが、その場合は起動時に以下のような検証コードを追加し、実際のアライメント値をログ出力または`jassert`で保証すること。

```cpp
void* p = ippsMalloc_8u(4096);
jassert((reinterpret_cast<uintptr_t>(p) & 63) == 0); // 64byte境界を実測検証
ippsFree(p);
```

どちらを選ぶにせよ、**この決定は`AlignedAllocation.h`一箇所に閉じる**ため、後から差し替えることも比較的容易である。

### 2.3 [重要] `ippvm` コンポーネントのCMake既知バグ

`ippsLn`/`ippsExp`/`ippsTanh`等の超越関数(MKL VMLに相当する機能)は、IPPでは `ippvm.h` / `ippvm.lib` という**`ipps`とは別の追加コンポーネント**に属する。現在の`CMakeLists.txt`は

```cmake
find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)
target_link_libraries(ConvoPeq PRIVATE IPP::ippcore IPP::ipps)
```

と`ippvm`を含めていない(現状FFTのみ利用のため不要だった)。Intel Community上には、IPP 2021.4.0〜2022.1世代のCMake Config パッケージにおいて、`ippvm`をCOMPONENTSに明示しないと正しく構成されず、`IPP::ippvm`をリンクしようとしてもターゲット未定義でビルドが失敗する、という既知の不具合報告が複数存在する。

**対応**: `find_package(IPP CONFIG COMPONENTS ippcore ipps ippvm)` + `target_link_libraries(... IPP::ippcore IPP::ipps IPP::ippvm)` への変更を、**DSPコードに着手する前の単独ステップ(Phase 0)として検証する**。具体的には、`ippsExp_64f`を1要素配列に対して呼ぶだけの最小限のスモークテスト(専用のCTestターゲット、または既存テスト実行ファイルへの一時的な診断コード)を用意し、静的リンクが解決することを最初に確認してから本実装に進む。ここでリンクエラーが出た場合は、`ippvm.lib`のパスを`target_link_directories`で明示するか、使用しているoneAPIバージョンのCMake Configの既知不具合を踏まえてバージョンアップを検討する。

### 2.4 [中] RNGアルゴリズムの変更

`PsychoacousticDither.h`は`VSL_BRNG_SFMT19937`(SIMD指向高速メルセンヌツイスタ)を使用しているが、IPPの`ippsRandUniform_64f`(状態は`ippsRandUniformInit_64f`で初期化)は、内部で用いる生成アルゴリズムをIntelが前面公開していない。統計的性質(周期長・相関特性)がSFMT19937と同一である保証はない。

ただし前述の通り、この呼び出しは**非RTワーカースレッド(`refillRandomRingNonRt`)にのみ存在し、Audio Threadは値を消費するだけ**なので、置き換えの実装難度・リスクそのものは低い。むしろこのファイルのクラス冒頭コメントには設計意図として「Xoshiro256\*\*(L/R独立jump)」と明記されているにもかかわらず、実装は一貫してMKL VSLを使っている、というドキュメントと実装の乖離が既に存在する。この点を踏まえ、§6.3で「IPPへの1:1差し替え」ではなく「コメントに記載された設計意図を自前実装で実現する」案を性能最適化の提案として示す。

### 2.5 [中] `JUCE_DSP_USE_INTEL_MKL` の扱い

`CMakeLists.txt`は`target_compile_definitions(ConvoPeq PRIVATE JUCE_DSP_USE_INTEL_MKL=1)`を設定している。JUCE本体(`juce_dsp.h`)のドキュメントコメントによれば、このフラグは**`juce::dsp::FFT`と`juce::dsp::Convolution`クラスの内部実装のみ**に影響し、それ以外のDSPモジュールの挙動(`WindowingFunction`等)には影響しない。

ConvoPeqのソース全体を検索した結果、`juce::dsp::FFT`・`juce::dsp::Convolution`のいずれも**一度も使用されていない**(独自の`MKLNonUniformConvolver`/`MklFftEvaluator`/直接DFTI呼び出しで完結している)。したがってこのフラグ自体は現状のランタイム挙動には無関係だが、**このフラグを`1`のまま残すと、JUCEモジュール自体のコンパイル単位が`<mkl_dfti.h>`を要求し続けるため、MKLのfind_package/リンクを削除した時点でビルドが失敗する**。よってこのフラグの削除(または`0`への変更)は「クリーンアップ」ではなく**必須の対応項目**である。

### 2.6 VML関数のIPP対応の確認結果

`mkl_vml.h`系の全呼び出しについて、IPP `ippvm.h` 側の対応関数の実在をリファレンスドキュメントで確認した結果を示す(§4に完全な対応表)。特筆すべき点:

- **`vzExp`(複素数の指数関数)には直接対応する`ippsExp_64fc`が存在する**。当初「IPPには複素数の超越関数は無く実部・虚部に分解する必要があるのでは」と想定していたが、これは誤りで、IPPの`ippvm`ドメインには`ippsExp_32fc/64fc`(精度別に`_A11/_A21/_A24`または`_A26/_A50/_A53`サフィックス違いあり)が存在するため、手動分解は不要である。
- `vzAbs`→`ippsMagnitude_64fc`、`vdLn`→`ippsLn_64f`、`vdTanh`→`ippsTanh_64f`(32f版の実在は文献で確認、64f版は命名規則の一貫性から存在を強く推定。**Phase 0のスモークテストで実存を確認すること**)。
- `vdMul`→`ippsMul_64f`/`ippsMul_64f_I`(こちらは`ippvm`ではなく基本の`ipps`ライブラリに属するため、追加コンポーネントは不要)。
- `cblas_dscal`→`ippsMulC_64f_I`(スカラー倍、in-place)。`cblas_ddot`→`ippsDotProd_64f`。いずれも`ipps`本体に属する。

---

## 3. 移行方針・原則

1. **既存の実績パターンを踏襲する**: FFTは`MKLNonUniformConvolver`/`MklFftEvaluator`で既に確立された「GetSize→Malloc→Init→(Fwd/Inv)→Free」の型を再利用する。新しい設計判断が必要なのは、VML相当(§2.6)・RNG(§2.4)・アライメント確保(§2.2)の3点のみに限定する。
2. **アロケータ不整合の是正を全ての先頭に置く**(§2.1)。これはIPP移行そのものとは独立した既存バグの是正であり、後続の全フェーズの安全性の前提になる。
3. **RTスレッドに影響する変更は無い**ことを各フェーズで明示的に再確認する(既存の`JUCE_ASSERT_MESSAGE_THREAD` / `ASSERT_NON_RT_THREAD()`等のアサーションが付いている箇所は、そのアサーションが引き続き成立することをコードレビューで確認する)。
4. **ビルド設定の検証(Phase 0)をコード変更より先に完了させる**(§2.3のippvmリスクのため)。
5. **1:1の直接移植(Phase 1〜5)と、アルゴリズム構造の最適化(Phase 6 / §6)を明確に分離する**。前者はレビュー・検証コストが低く先に本流へ統合し、後者(R2C/C2R化等)は数値的な検証項目が増えるため、別ブランチ・別PRとして独立にレビューする。
6. **MKLとIPPの完全な数値ビット一致は目指さない**。異なるベンダーライブラリ間で浮動小数点演算の丸め結果が完全一致することは一般に保証されないため、検証は「ビット一致」ではなく「聴感上・統計的に有意な差がないこと」を基準にする(§7.2)。

---

## 4. API対応表(MKL → IPP 完全対応表)

| # | MKL API | シグネチャ(簡略) | 使用箇所 | IPP代替 | 必要コンポーネント | 備考 |
|---|---|---|---|---|---|---|
| 1 | `mkl_malloc` | `void* mkl_malloc(size_t, int align)` | `AlignedAllocation.h`ほか多数 | `_aligned_malloc(size, align)`(推奨) / `ippsMalloc_8u(len)`(要検証) | `<malloc.h>`(CRT) / `ippcore`+`ipps` | §2.2 |
| 2 | `mkl_free` | `void mkl_free(void*)` | 同上 | `_aligned_free(ptr)` / `ippsFree(ptr)` | 同上 | 上と対で変更 |
| 3 | `mkl_set_num_threads` | `void mkl_set_num_threads(int)` | `MKLRealTimeSetup.cpp`、`MKLNonUniformConvolver`コンストラクタ | 呼び出し自体を削除 | ― | IPPは非スレッド設計のため不要 |
| 4 | `mkl_set_dynamic` | `void mkl_set_dynamic(int)` | `MKLRealTimeSetup.cpp` | 呼び出し自体を削除 | ― | 同上 |
| 5 | `vmlSetMode` | `int vmlSetMode(unsigned int)` | `MainApplication.cpp`、`AudioEngine.RebuildDispatch.cpp` | 呼び出し自体を削除(`_MM_SET_FLUSH_ZERO_MODE`/`_MM_SET_DENORMALS_ZERO_MODE`は残す) | ― | VML関数が無くなるため不要 |
| 6 | `DftiCreateDescriptor`/`DftiSetValue`/`DftiCommitDescriptor`(`DFTI_COMPLEX`, in-place) | ― | `SpectrumAnalyzerComponent`、`StateAndUI.cpp` | `ippsFFTGetSize_C_32fc`+`ippsFFTInit_C_32fc`(直接移植) または `ippsFFTGetSize_R_32f`+`ippsFFTInit_R_32f`(最適化版、§6.2) | `ipps` | out-of-place推奨 |
| 7 | `DftiCreateDescriptor`(`DFTI_DOUBLE, DFTI_COMPLEX`、実質実数入力) | ― | `MixedPhase.cpp`、`ResampleAndFallback.cpp` | `ippsFFTGetSize_R_64f`+`ippsFFTInit_R_64f`(既存パターンを共有、§6.1) | `ipps` | 対称性活用で2倍高速化 |
| 8 | `DftiComputeForward`/`DftiComputeBackward` | ― | 上記全て | `ippsFFTFwd_CToC_*`/`ippsFFTInv_CToC_*`(直接移植) または `ippsFFTFwd_RToCCS_*`/`ippsFFTInv_CCSToR_*`(最適化版) | `ipps` | ― |
| 9 | `DftiFreeDescriptor` | ― | `DftiHandle.h`(RAII) | `ippsFree(specBuf)` | `ipps` | RAIIラッパーを`ScopedIppFFTSpec`として再設計(§5 Phase3) |
| 10 | `vdMul` | `void vdMul(int n, const double* a, const double* b, double* y)` | `MKLNonUniformConvolver.cpp`(`applySpectrumFilter`、`SetImpulse`テール処理) | `ippsMul_64f_I(pSrc, pSrcDst, len)` | `ipps`(追加リンク不要) | in-place版で意味論一致 |
| 11 | `cblas_dscal` | `void cblas_dscal(int n, double a, double* x, int incx)` | `CacheManager.cpp`、`ConvolverProcessor.LoadPipeline.cpp`、`MKLNonUniformConvolver.cpp` | `ippsMulC_64f_I(val, pSrcDst, len)` | `ipps` | incx=1前提のため直接対応 |
| 12 | `cblas_ddot` | `double cblas_ddot(int n, const double* x, int incx, const double* y, int incy)` | `CacheManager.cpp`、`IRConverter.cpp`(エネルギー計算 `x·x`) | `ippsDotProd_64f(pSrc1, pSrc2, len, &result)` | `ipps` | incx=incy=1前提 |
| 13 | `vzAbs` | `void vzAbs(int n, const MKL_Complex16*, double*)` | `ResampleAndFallback.cpp`(実ケプストラム法) | `ippsMagnitude_64fc(pSrc, pDst, len)` | `ippvm` | §2.3の検証必須 |
| 14 | `vdLn` | `void vdLn(int n, const double*, double*)` | 同上 | `ippsLn_64f(pSrc, pDst, len)` | `ippvm` | 同上 |
| 15 | `vzExp` | `void vzExp(int n, const MKL_Complex16*, MKL_Complex16*)` | 同上 | `ippsExp_64fc(pSrc, pDst, len)` | `ippvm` | 分解不要、直接対応あり(§2.6) |
| 16 | `vcAbs` | `void vcAbs(int n, const MKL_Complex8*, float*)` | `StateAndUI.cpp` | `ippsMagnitude_32fc(pSrc, pDst, len)` | `ippvm` | 単精度版 |
| 17 | `vdTanh` | `void vdTanh(int n, const double*, double*)` | `NoiseShaperLearner.cpp` | `ippsTanh_64f(pSrc, pDst, len)` | `ippvm` | Phase 0で実在確認 |
| 18 | `vslNewStream`(`VSL_BRNG_SFMT19937`) | ― | `PsychoacousticDither.h` | `ippsRandUniformGetSize_64f`+`ippsRandUniformInit_64f` | `ipps` | §2.4、§6.3で自前実装案も提示 |
| 19 | `vdRngUniform` | ― | 同上 | `ippsRandUniform_64f(pDst, len, pState)` | `ipps` | 非推奨版`_Direct_`は使わない |
| 20 | `vslDeleteStream` | ― | 同上 | `ippsFree(pState)` | `ipps` | ― |
| 21 | `MKL_INT`/`MKL_LONG`/`MKL_Complex8`/`MKL_Complex16` | 型定義 | 複数 | `int`/`Ipp32s`、`Ipp32fc`/`Ipp64fc` | `ipps.h` | メモリレイアウトは`{re;im}`で共通 |
| 22 | `ippInit()`(参考: MKL側に直接対応物は無いが現状MainApplication.cppで既に呼ばれている) | ― | ― | 変更なし(静的リンクではIPP 9.0以降必須ではないが害もないため残置) | ― | 参考情報 |

---

## 5. ファイル別詳細移行手順

### Phase 0 ── ビルド基盤の検証(コード変更前の前提確認)

**目的**: §2.3のippvmリスクを本実装前に潰す。

1. `CMakeLists.txt`の`find_package(IPP QUIET CONFIG COMPONENTS ippcore ipps)`を
   ```cmake
   find_package(IPP CONFIG COMPONENTS ippcore ipps ippvm)
   ```
   に変更(`QUIET`を外し、失敗時にCMake自体がエラーで停止するようにして問題を早期に顕在化させる)。
2. `target_link_libraries(ConvoPeq PRIVATE ... IPP::ippcore IPP::ipps IPP::ippvm ...)`を追加。
3. 一時的な診断コード(例: `MainApplication::initialise()`冒頭、または専用の使い捨てテスト実行ファイル)で以下を実行し、リンク・実行の両方が成功することを確認する。
   ```cpp
   {
       double src[1] = { 2.718281828 };
       double dst[1] = { 0.0 };
       IppStatus st = ippsLn_64f(src, dst, 1);
       jassert(st == ippStsNoErr);
       st = ippsExp_64f(src, dst, 1);
       jassert(st == ippStsNoErr);
       st = ippsTanh_64f(src, dst, 1);
       jassert(st == ippStsNoErr);
   }
   ```
4. §2.2で選択したアライメント確保方式(`_aligned_malloc`推奨)についても、`reinterpret_cast<uintptr_t>(p) & 63 == 0`を実測し、64byte境界が実際に得られることを確認する。
5. ここで問題が出た場合(リンクエラー、アライメント不足)、後続フェーズには進まず、oneAPIのバージョン確認・パス設定の見直しを先に行う。

### Phase 1 ── メモリ確保基盤の統一と不整合是正(最優先・最重要)

**対象**: `AlignedAllocation.h`、および§2.1で特定した4つの生呼び出し箇所。

1. **先に生呼び出しを解消する**(実装本体を変える前に):
   - `IRConverter.cpp::convertFile()`: `mkl_malloc(bytes, 64)` → `convo::aligned_malloc(bytes)`(または既存のヘルパー関数シグネチャに合わせる)、対になる`mkl_free(data)`(キャンセル時の早期return分)→ `convo::aligned_free(data)`。
   - `CacheManager.cpp::copyFromMmapToAligned()`・`loadPreparedState()`: 同様に`mkl_malloc`呼び出し2箇所を`convo::aligned_malloc`に統一。
   - `ConvolverProcessor.Lifecycle.cpp`: 3箇所の生`mkl_free(state)`/`mkl_free(oldIrState)`を`convo::aligned_free(state)`/`convo::aligned_free(oldIrState)`に統一。
   - この時点でコードベース内の`mkl_malloc`/`mkl_free`の生呼び出しは、`AlignedAllocation.h`自身の実装と`MKLNonUniformConvolver.cpp`(自己完結、Phase 2で扱う)のみになっていることをコードベース全体の`grep`で再確認する。
2. **`AlignedAllocation.h`本体を書き換える**:
   ```cpp
   // Before:
   inline void* aligned_malloc(size_t size, size_t alignment = 64) {
       return mkl_malloc(size, static_cast<int>(alignment));
   }
   inline void aligned_free(void* p) { mkl_free(p); }
   inline void* aligned_malloc_nothrow(size_t size, size_t alignment) noexcept {
       return mkl_malloc(size, (int)alignment);
   }

   // After (推奨: _aligned_malloc):
   inline void* aligned_malloc(size_t size, size_t alignment = 64) {
       return _aligned_malloc(size, alignment);
   }
   inline void aligned_free(void* p) { _aligned_free(p); }
   inline void* aligned_malloc_nothrow(size_t size, size_t alignment) noexcept {
       return _aligned_malloc(size, alignment); // _aligned_malloc自体は失敗時nullptrを返すため、noexcept契約はそのまま満たせる
   }
   ```
   `MKLAllocator`(STLアロケータ)内部の`allocate`/`deallocate`も同様に書き換える。クラス名`MKLAllocator`自体は、`AllpassDesigner.h`等7ファイルが型名として参照しているため、**本フェーズでは変更しない**(§9.2でリネームを将来の任意対応として提案)。`aligned_malloc_nothrow`/`makeAlignedArray_nothrow`(Rev.2で追加、§0改訂履歴参照)も同様に書き換えの対象に含める。現時点では呼び出し元が存在しないため挙動検証はビルド確認のみでよいが、将来これらが実際に使われ始めた際に備え、書き換え漏れが無いようにする。
3. `#include <mkl.h>`をこのファイルから削除し、`_aligned_malloc`推奨の場合は`#include <malloc.h>`を追加(IPP採用の場合は`#include <ipp.h>`)。

### Phase 2 ── `MKLNonUniformConvolver.cpp/.h`の残存MKL除去

**対象**: 既に自己完結している同ファイル内の`mkl_malloc`/`mkl_free`(§2.1のクロスファイル問題とは独立)、`vdMul`、`cblas_dscal`、`mkl_set_num_threads`。

**[Rev.2注記]** Rev.1時点の記述はAoS(`irFreqDomain`が全パーティション分の永続バッファ)を前提にしていたが、Rev.2で当該バッファはSoA(`irFreqReal`/`irFreqImag`)へ一本化され、AoSは1パーティション分の使い捨てスクラッチに縮小された([Mem-Fix])。以下は現行のSoA構造に合わせた手順である。

1. Phase 1で`convo::aligned_malloc`/`aligned_free`が非MKL実装に切り替わっているため、このファイル内の`mkl_malloc`/`mkl_free`呼び出しは**そのまま`convo::aligned_malloc`/`convo::aligned_free`に置き換えるだけでよい**(このファイルはこれまで独自に生呼び出ししていたため、Phase 1の対象には含めなかった箇所)。対象は`Layer::freeAll`、`SetImpulse`内の各種スクラッチバッファ(`irFreqDomain`・`fdlBuf`・`tempTime`・`tempFreq`等、いずれもサイズはRev.2で縮小済み)。なお、パーティション逆順並び替え時にAoS用一時バッファを確保していた`mkl_malloc`/`mkl_free`のペア(旧称`swapDomain`)はRev.2で**コード自体が削除された**ため、本フェーズでの対応は不要になった(移行対象が1組減った)。
2. `applySpectrumFilter`内、および`SetImpulse`のテール減衰ゲイン適用内の`vdMul`呼び出しは、Rev.2でいずれも「実数値ゲイン配列(`gain`/`gainReal`、要素数`cSize`または`complexSize`)を、SoAの実部配列・虚部配列それぞれに個別に掛ける」という形へ変わっている(interleave/deinterleave手順そのものが不要になった)。具体的には:
   ```cpp
   // Before (MKL VML, いずれの箇所も):
   vdMul(cSize, re, gain, re);
   vdMul(cSize, im, gain, im);

   // After (IPP、in-place版):
   ippsMul_64f_I(gain, re, cSize);
   ippsMul_64f_I(gain, im, cSize);
   ```
   1箇所につき呼び出しが2回に増えるが、各回の処理要素数は旧AoS版(`cSize*2`要素×1回)の半分(`cSize`要素)なので、実効演算量は変わらない。
3. `SetImpulse`内の`cblas_dscal(l.complexSize * 2, scale, l.irFreqDomain, 1)`(Rev.2ではオフセットが`+ p * l.partStride`から固定オフセット0のスクラッチ単体に変わっている点に注意)を`ippsMulC_64f_I(scale, l.irFreqDomain, l.complexSize * 2)`に置換。
4. コンストラクタの`mkl_set_num_threads(1)`呼び出しと、それに付随するコメントを削除。
5. ヘッダのコメントブロック(`// [FFT換装] ...`および`// [Mem-Fix] ...`)を更新し、「VML/BLASもIPP化完了」の旨を追記する(このコメントは今後のメンテナ・他AIインスタンスへの引き継ぎ精度に直結するため、削除ではなく更新を推奨)。
6. `#include <mkl.h>`を削除し`#include <ipp.h>`に置換。
7. 併せて`#ifndef __AVX2__ #error ...`ガード(Rev.2で追加、本移行とは無関係)はそのまま維持してよい。IPP化後もSoA/AVX2の単一実装という前提は変わらない。

### Phase 3 ── 残存DFTI直接呼び出しのIPP化

**対象**: `DftiHandle.h`、`SpectrumAnalyzerComponent.cpp/.h`、`ConvolverProcessor.StateAndUI.cpp`、`ConvolverProcessor.MixedPhase.cpp`、`ConvolverProcessor.ResampleAndFallback.cpp`。

1. `DftiHandle.h`を置き換える新ヘッダ(仮称`IppFftSpecHandle.h`)を作成し、`DFTI_DESCRIPTOR_HANDLE`の代わりに`Ipp8u*`(スペックバッファ)を管理するRAIIラッパーを実装する。`MKLNonUniformConvolver.cpp`内の`IppFFTPlanCache`と設計思想を共有すること(§6.4で共有ヘッダへの統合を提案)。
   ```cpp
   class ScopedIppFFTSpecR64 {
   public:
       bool init(int order) noexcept {
           int sizeSpec=0, sizeInit=0, sizeWork=0;
           if (ippsFFTGetSize_R_64f(order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast,
                                     &sizeSpec, &sizeInit, &sizeWork) != ippStsNoErr) return false;
           specBuf = static_cast<Ipp8u*>(convo::aligned_malloc(sizeSpec));
           workBuf = static_cast<Ipp8u*>(convo::aligned_malloc(sizeWork));
           Ipp8u* initBuf = sizeInit > 0 ? static_cast<Ipp8u*>(convo::aligned_malloc(sizeInit)) : nullptr;
           IppStatus st = ippsFFTInit_R_64f(&spec, order, IPP_FFT_DIV_INV_BY_N, ippAlgHintFast, specBuf, initBuf);
           if (initBuf) convo::aligned_free(initBuf);
           return st == ippStsNoErr;
       }
       ~ScopedIppFFTSpecR64() { convo::aligned_free(specBuf); convo::aligned_free(workBuf); }
       IppsFFTSpec_R_64f* spec = nullptr;
       Ipp8u* workBuf = nullptr;
   private:
       Ipp8u* specBuf = nullptr;
   };
   ```
   (単精度・複素版は`_32f`/`_C_64fc`等をテンプレートまたは並行クラスで用意)
2. **`SpectrumAnalyzerComponent`**: 現行は実信号を虚部ゼロで複素配列に詰めてからC2Cで in-place FFT している。Phase 3(直接移植)では`ippsFFTGetSize_C_32fc`+`ippsFFTFwd_CToC_32fc`への1:1移植にとどめ、R2C化(§6.2、約2倍高速)は別PRとして切り出す。
3. **`ConvolverProcessor.StateAndUI.cpp`**(`createFrequencyResponseSnapshot`): 同様にC2C単精度への直接移植を基本とし、`vcAbs`は`ippsMagnitude_32fc`に置換。
4. **`ConvolverProcessor.MixedPhase.cpp`/`ResampleAndFallback.cpp`**(`convertToMixedPhase`系・`convertToMinimumPhase`): Phase 3では`ippsFFTGetSize_C_64fc`+`ippsFFTFwd/Inv_CToC_64fc`への直接移植(現行のフルコンプレックス構造を変えない)にとどめる。`vzAbs`→`ippsMagnitude_64fc`、`vdLn`→`ippsLn_64f`、`vzExp`→`ippsExp_64fc`への置換もこの段階で行う。R2C/C2R化による2倍高速化は§6.1でPhase 6として独立提案する。
5. `ConvolverProcessor.h`の`convo::ScopedDftiDescriptor fftHandle;`宣言を新クラス(`ScopedIppFFTSpecC32`等、用途に応じた型)に置換。

### Phase 4 ── RNG(`PsychoacousticDither.h`)

1. まず直接移植版: `vslNewStream`→`ippsRandUniformGetSize_64f`+`ippsRandUniformInit_64f`、`vdRngUniform`→`ippsRandUniform_64f`、`vslDeleteStream`→`ippsFree`。状態オブジェクト(`IppsRandUniState_64f*`)は`convo::aligned_malloc`で確保したバッファに配置する。
2. `refillRandomRingNonRt()`内の呼び出し1点のみが変更対象であり、Audio Threadの`popUniformFromRing`/`fallbackUniform`ロジックには一切手を入れない。
3. 移行後、ディザのノイズフロア(周波数特性・ヒストグラム分布)をSFMT19937版と比較し、可聴域で有意な相関ノイズ(周期性アーティファクト)が生じていないことを確認する(§7.2)。
4. より高い性能を狙う場合は§6.3の自前Xoshiro256\*\*実装を検討する(このクラスの冒頭コメントに既に設計意図として記載されている)。

### Phase 5 ── ビルド設定の最終クリーンアップ

1. `CMakeLists.txt`から`find_package(MKL REQUIRED ...)`ブロック全体、`MKL::MKL`のリンク、icx向け`/Qmkl:sequential`オプション、`MKLROOT`関連のinclude/link path設定を削除。
2. `target_compile_definitions(... JUCE_DSP_USE_INTEL_MKL=1)`を削除(§2.5により必須)。
3. `RuntimePublicationCoordinatorTests`・`PartialPublicationRejectTests`への個別`MKL::MKL`リンクを削除(Phase 1〜4完了後、これらのテストが`AlignedAllocation.h`等を通じて要求するのはIPPのみになっているはずなので、必要であれば`IPP::ippcore IPP::ipps`を代わりにリンクする)。
4. `build.bat`のエラーメッセージ「Intel oneAPI MKL not found!」を「Intel oneAPI environment not found!」等、MKL固有ではない文言に更新(oneAPI自体はIPPのためにも引き続き必要なため、`setvars.bat`呼び出しロジック自体は変更不要)。
5. `RuntimeBuilder.h`の`BuildError::MKLFailure`列挙子は、後方互換性よりも正確性を優先するなら`DspLibraryInitFailure`等へのリネームを検討(呼び出し元3ファイル程度の影響、任意対応)。

---

## 6. パフォーマンス最大化の提案(移行を超えた改善)

いずれもAudio Thread外(IRロード時間・バックグラウンド処理時間の短縮)に効く提案であり、Phase 1〜5の直接移植が完了し回帰検証が済んだ後に、**独立したフェーズ(Phase 6)**として着手することを推奨する。

### 6.1 [最大の効果] `convertToMinimumPhase`のR2C/C2R化による計算量半減

現行の実ケプストラム法によるミニマムフェーズ再構成(`ResampleAndFallback.cpp`)は、以下の4回のFFT(順・逆・順・逆)を**すべてフルコンプレックス(`DFTI_COMPLEX`、N点複素データ)**で実行している。

1. `X[k] = FFT(x[n])` — `x[n]`は実数(虚部ゼロ埋め)
2. `c[n] = IFFT(ln|X[k]|)` — `ln|X[k]|`は実数かつ偶対称(実信号のFFTはエルミート対称なので`|X[k]|=|X[N-k]|`)
3. ケプストラム領域での因果窓掛け(`c_min[n] = 2c[n]` for `1≤n<N/2`、`c_min[n]=0` for `n>N/2`)— 実数配列に対する処理でFFTと無関係
4. `C_min[k] = FFT(c_min[n])` — `c_min[n]`は実数
5. `X_min[k] = exp(C_min[k])`
6. `x_min[n] = IFFT(X_min[k])` — `X_min[k]`は(2)の対称性を`exp`が保つため再びエルミート対称、よって`x_min[n]`は実数

**ステップ1・2・4・6は全て「実数入力・実数出力」の性質を持つ**(実信号のFFTが持つエルミート対称性を`log`/`exp`の各段階が保存するため)。にもかかわらず現行実装はN点の冗長な複素データとして扱っており、理論上必要な計算量のほぼ2倍を消費している。

**改修案**: 全ての`DftiComputeForward/Backward`(N点複素)を、`ippsFFTFwd_RToCCS_64f`/`ippsFFTInv_CCSToR_64f`(N点実数 ⇔ (N/2+1)点CCS複素)に置き換える。これは`MklFftEvaluator.h`・`MKLNonUniformConvolver.cpp`で既に実績のある型であり、新規パターンの導入ではない。特に有用なのは、IPPのCCS形式が「DC・ナイキスト成分の虚部を0とした通常の複素配列」として扱える点で、これは`MklFftEvaluator.h`が`CcsComplex{double real, imag;}`型でCCS出力をそのままインデックスアクセスしていることからも裏付けられる。つまり:

- ステップ2の`vzAbs`/`vdLn`は(N/2+1)要素に対してのみ実行すればよい(要素数半減)。
- ステップ3のケプストラム窓掛けは、実数配列(`double[N]`)に対する処理のまま変更不要(むしろ「虚部を明示的にゼロクリアする」現行コードの後処理が不要になり、コードは単純化する)。
- ステップ5の`vzExp`→`ippsExp_64fc`も(N/2+1)要素のみでよい。

**期待効果**: 大規模IR(長尺リバーブ等でFFTサイズが数十万点に達するケース)ほど効果が大きく、当該関数のCPU時間はおおむね40〜50%程度の削減が見込める(FFT自体の理論計算量がO(N log N)からO((N/2)log(N/2))相当に、要素毎演算がN→N/2+1に減るため)。これはIRロード〜ミニマムフェーズ変換完了までのユーザー体感待ち時間の短縮に直結する。

**検証上の注意**: アルゴリズムの数学的等価性は上記の通り厳密に成立するが、実装変更はAPI置換にとどまらずデータ構造(生配列の意味論)を変えるため、既存のPhase 3(直接移植)とは切り離してレビューし、`convertToMinimumPhase`単体のゴールデンデータ回帰テスト(既知の入力IRに対する出力波形の差分がフロアノイズレベル以下であること)を追加してから統合すること。

### 6.2 `SpectrumAnalyzerComponent`のR2C化

現行は実信号を虚部ゼロで複素配列に展開してからC2C FFTを行っている(§5 Phase 3の直接移植ではこの構造を維持する想定)。UIタイマー駆動とはいえ、表示更新レート(`TIMER_HZ_ACTIVE`相当)によっては定常的なCPU負荷源になる。§6.1と同じ理由により、`ippsFFTFwd_RToCCS_32f`への置換で「実→複素展開ループ」自体を丸ごと削除でき、FFT計算量も半減する。表示に使う振幅スペクトルは元々前半(N/2+1)ビンのみのため、CCS出力はまさに過不足のない形式であり、後段の`vcAbs`→`ippsMagnitude_32fc`のロジックもそのまま(N/2+1)要素に対して適用できる。

### 6.3 RNGの自前実装(Xoshiro256\*\*)によるライブラリコール排除

`PsychoacousticDither.h`冒頭のクラスコメントには設計意図として「Xoshiro256\*\*(L/R独立jump)」と明記されているが、実装は一貫してMKL VSL(SFMT19937)を使用しており、**ドキュメントと実装が乖離した状態(いわゆる意図と実装のドリフト)**にある。

IPPへの1:1差し替え(§5 Phase 4)は最小変更として妥当だが、性能を最大化する観点では、コメントに既に明記されている設計を実際に実装することを提案する。Xoshiro256\*\*は数行の整数シフト演算のみで1つの64bit一様乱数を生成でき、`refillRandomRingNonRt()`のバッチ生成ループ内でインライン展開すれば、ライブラリ呼び出しのディスパッチオーバーヘッドを排除できる。L/R独立ストリームは`jump()`関数(固定的な多項式によるジャンプ)で2^128周期分だけ離れた状態を生成すれば、チャンネル間の統計的独立性も確保できる。この案は同時に「IPPのRNGアルゴリズムがSFMT19937と異なることによる統計的性質の不確実性」(§2.4)そのものを解消する(自前実装なので統計的性質を完全に把握・検証できる)という副次的な利点もある。

### 6.4 `IppFFTPlanCache`の共有ヘッダへの切り出し

`MKLNonUniformConvolver.cpp`内に private に実装されている`IppFFTPlanCache`(FFTオーダーをキーにスペックを共有する仕組み)は、§5 Phase 3で新設する`SpectrumAnalyzerComponent`・`MixedPhase.cpp`・`ResampleAndFallback.cpp`・`StateAndUI.cpp`のFFTスペック管理にもそのまま適用可能な汎用設計である。共有ヘッダ(例: `src/IppFftPlanCache.h`)に切り出し、精度・ドメイン(`_R_64f`/`_C_64fc`/`_R_32f`/`_C_32fc`)をテンプレートパラメータ化して各所から再利用することで、(a) コード重複の削減、(b) 同一セッション内で複数箇所が同じFFTサイズを使う場合のスペック再生成コスト削減、の両方が得られる。これはMKL→IPP移行そのものの必須要件ではないが、移行作業と同時に着手するコストが最も低い(いずれにせよ各ファイルのFFT初期化コードを書き直すため)ため、本計画に含めることを推奨する。

### 6.5 その他の軽微な最適化機会

- **`ippAlgHintFast`の一貫使用**: 既存の`MKLNonUniformConvolver`は`ippAlgHintFast`(精度よりも速度を優先するアルゴリズムヒント)を使用している。Phase 3で新規に初期化する全FFTスペックでも同じヒントを使用し、意図せず`ippAlgHintAccurate`(デフォルト)相当の遅いパスを選んでしまわないよう明示的に指定すること。
- **`mkl_set_num_threads(1)`削除に伴うプロセス全体のスレッド数設定の見直し**: 現行は起動時に環境変数`MKL_NUM_THREADS=1`・`OMP_NUM_THREADS=1`を強制している。MKLが完全に除去されれば`MKL_NUM_THREADS`の設定自体が無意味になるが、`OMP_NUM_THREADS=1`はIPPが内部でOpenMPを使う可能性のあるスレッド版ライブラリを誤ってリンクした場合の保険として残す価値がある(現状`IPP_THREADING=sequential`でこれを回避しているため通常は不要だが、多層防御として維持を推奨)。

---

## 7. 検証・テスト計画

### 7.1 ビルド検証

- Phase 0のスモークテスト(§5参照)をCI(またはローカル)の最初のゲートとする。
- MSVCビルドとicxビルドの両方で、Phase 0〜5それぞれの完了時点でクリーンビルドが通ることを確認する(現行`CMakeLists.txt`がコンパイラごとに異なるMKLリンク方法を持つため、IPP側でも同様の分岐が必要かを都度確認する)。
- `find_package(MKL ...)`削除後、CI環境からMKLパッケージ自体をアンインストールした状態でビルドが通ることを確認する(依存の完全除去を証明する最も確実な方法)。

### 7.2 数値等価性検証

異なるベンダーのFFT/VML実装は浮動小数点演算の内部順序が異なるため、ビット完全一致は前提にしない。代わりに以下を用いる。

- **IR畳み込み出力の差分**: 移行前後で同一IR・同一入力信号に対する`processLayerBlock`出力の差分RMSを測定し、-120dBFS(倍精度演算のノイズフロア相当)以下であることを確認する。§0.1の通りAudio Thread自体は既にIPP化済みで無変更のため、ここで差分が出るとすれば主にIRロード時のスケーリング計算(`cblas_dscal`→`ippsMulC_64f_I`等)由来であり、差分の発生源を切り分けやすい。
- **`convertToMinimumPhase`/`convertToMixedPhase`の出力波形**: 既知のテストIR(短尺・長尺・無音・インパルスのみ等のエッジケースを含む)に対する処理前後の波形差分をゴールデンデータとして保存し、Phase 3(直接移植)・Phase 6(§6.1のR2C化)それぞれの段階で比較する。
- **ディザのノイズフロア**: `PsychoacousticDither`移行前後で、無音入力に対する出力のパワースペクトル密度を比較し、周期性アーティファクト(特定周波数のスパイク)が新たに出現していないことを確認する。

### 7.3 リアルタイム性検証(既存の診断基盤の活用)

現在調査中の`CB_HIST`/`CPU_MIG`/`CB_SEQ`/`DSP_TIMING`のRT違反ログ基盤を活用し、本移行の前後でこれらの計測値に有意な変化がないことを確認する。§0.1の通りAudio Thread自体は無変更のため理論上は差分ゼロのはずだが、ビルドオプション変更(MKLリンク除去によるバイナリレイアウト変化、Phase 5でのコンパイルフラグ変更)が間接的にコード生成に影響する可能性はゼロではないため、念のため実測することを推奨する。

### 7.4 リグレッションテスト

- 既存のテストスイート(`RuntimePublicationCoordinatorTests`、`PartialPublicationRejectTests`等)をPhase 5完了後に再実行し、MKLリンク除去後もパスすることを確認する。
- Phase 1で是正するアロケータ不整合(§2.1)については、`PreparedIRState`の生成・破棄を繰り返すユニットテスト(理想的には[AddressSanitizer](https://github.com/google/sanitizers)またはWindows環境であれば[Application Verifier](https://learn.microsoft.com/en-us/windows-hardware/drivers/devtest/application-verifier)のヒープ検証を有効化した状態での実行)を追加し、Phase 1適用前は問題が顕在化しない(または稀にしか顕在化しない)一方、適用後の是正によりヒープ検証が確実にクリーンであることを積極的に確認する。

---

## 8. ロールアウト順序とロールバック戦略

推奨する統合順序は本計画の記載順(Phase 0 → 1 → 2 → 3 → 4 → 5 → 6)である。理由:

- Phase 0(ビルド検証)は他の全てのコード変更の前提であり、ここで問題が出れば後続作業は着手しない。
- Phase 1(アロケータ統一)は独立した既存バグ修正であり、単独でPRとしてマージ可能。これによりPhase 2以降のどの段階で問題が起きても「アロケータ起因のヒープ破壊」という最悪のクラスの障害は既に排除された状態になる。
- Phase 2〜4は互いに独立なファイル群を対象とするため、並行して着手・レビューしてよい。
- Phase 5(ビルド設定の最終クリーンアップ、MKL依存の完全撤去)は、Phase 1〜4が全て完了し検証を終えた後の最後のステップとする。**Phase 5より前の段階ではMKLの`find_package`/リンクをCMakeに残したままにしておき**、Phase 1〜4の各変更がIPP経路とMKL経路の両方で共存可能な状態(MKLは単にリンクされているだけで、コードからは呼ばれなくなっている状態)を保つことで、問題が出た場合に当該ファイルの変更のみを個別にrevertできる。
- Phase 6(§6のパフォーマンス最適化)は、Phase 1〜5の直接移植が本流にマージされ、最低1回のリリースサイクルで実運用検証が済んだ後に着手することを推奨する。理由はアルゴリズム構造そのものを変える変更であり、直接移植(挙動同一を目指す変更)とは検証の質が異なるため。

**ロールバック戦略**: 各Phaseを独立したコミット(またはPR)単位にすることで、問題発生時に該当Phaseのみをrevertできるようにする。特にPhase 1はコードベース全体に影響するため、revert時の影響範囲(Phase 2以降が既にマージされている場合はそちらも連動してrevertが必要か)を事前に整理しておくこと。

---

## 9. 付録

### 9.1 `mkl_malloc`/`mkl_free` 生呼び出しの完全リスト(§2.1の詳細根拠)

調査時点でコードベース全体から検出した生呼び出し(`convo::aligned_malloc`/`aligned_free`ラッパーを経由しないもの)は以下の通り。`MKLNonUniformConvolver.cpp`内の呼び出しは全て同一ファイル内で確保・解放が完結しており、他のクラス/ファイルの生存期間管理と交差しないことを個別に確認済み。

| ファイル | 関数 | 変数 | 確保/解放 |
|---|---|---|---|
| `MKLNonUniformConvolver.cpp` | `Layer::freeAll` | `irFreqDomain`(Rev.2で1パーティション分のスクラッチに縮小), `fdlBuf`(同、2スロット分に縮小), `fftTimeBuf`, `fftOutBuf`, `prevInputBuf`, `accumBuf`, `inputAccBuf`, `tailOutputBuf` 等 | 解放(自己完結) |
| `MKLNonUniformConvolver.cpp` | `applySpectrumFilter` | `reusableGain`(Rev.2で`reusableGainInterleaved`から改称、サイズも`cSize*2`→`cSize`へ半減。実部・虚部で共有する実数ゲイン配列) | 確保(メンバとして永続化、呼び出しを跨いで再利用。解放は`Layer::freeAll`/デストラクタ側) |
| `MKLNonUniformConvolver.cpp` | `SetImpulse`(テール減衰ゲイン適用箇所) | `gainReal`(Rev.2で`gainInterleaved`から改称、サイズも`complexSize*2`→`complexSize`へ半減) | 確保・解放(自己完結、`ScopedAlignedPtr`で管理) |
| `MKLNonUniformConvolver.cpp` | `SetImpulse` | `irFreqReal`, `irFreqImag`, `tempTime`, `tempFreq`ほか多数 | 確保(自己完結、Layer解放時に対で解放) |
| ~~`MKLNonUniformConvolver.cpp`~~ | ~~パーティション逆順並び替え(`swapDomain`)~~ | ~~AoS用一時バッファ~~ | **Rev.2でコード自体が削除され、対応不要になった** |
| `CacheManager.cpp` | `copyFromMmapToAligned` | `dst` | 確保(**呼び出し元が存在しないデッドコード**、§9.2) |
| `CacheManager.cpp` | `loadPreparedState` | `copied` | 確保 → `prepared->partitionData`へ代入 → **`PreparedIRState`デストラクタで`convo::aligned_free`により解放**(要Phase1是正) |
| `IRConverter.cpp` | `convertFile` | `data` | 確保 → `prepared->partitionData`へ代入 → 同上(要Phase1是正)。キャンセル時の早期return分は同関数内で`mkl_free`により解放 |
| `ConvolverProcessor.Lifecycle.cpp` | IRState破棄まわり(3箇所) | `state`, `oldIrState` | 解放。対応する確保は`convo::aligned_make_unique<IRState>()`(要Phase1是正) |

### 9.2 副次的に発見した事項(本移行のスコープ外だが記録に値する)

- **`CacheManager::copyFromMmapToAligned()`はデッドコードである**: 宣言・定義は存在するが、コードベース全体を検索してもどこからも呼び出されていない。本移行(Phase 1)ではこの関数もアロケータ統一の対象に含めるが、機能自体を削除するかどうかは本計画のスコープ外として別途判断を委ねる。
- **`RuntimeBuilder.h`の`BuildError::MKLFailure`という列挙子名**は、移行完了後は実態と乖離した名称になる。§5 Phase 5で任意対応として言及した通り、リネームは本移行の必須要件ではない。
- **`MklFftEvaluator`・`MKLNonUniformConvolver`というクラス名自体**も移行完了後は「実装はIPPだがMKLの名を冠したまま」という状態になる。両クラスとも参照箇所が7〜10ファイルに及ぶため、リネームは本移行とは別の独立したリファクタリングPRとして扱うことを推奨する(本計画には含めない)。
- **`PsychoacousticDither.h`のクラス冒頭コメントと実装の乖離**(Xoshiro256\*\*と明記されているが実装はMKL VSL)は§6.3で積極的な改善提案として扱ったが、これ自体は本移行が発見した「既存のドキュメント/実装ドリフト」の一例であり、他のコメントについても同様の乖離がないか、移行作業中に併せて目視確認することを推奨する。

### 9.3 本計画作成にあたり参照した外部情報

本計画のIPP API対応(§2.2〜2.6、§4)は公開されているIntel oneAPI IPPのリファレンスドキュメント・開発者コミュニティの情報を参照して検証した。特に以下の点は版・世代によって記述が異なるため、Phase 0のスモークテストでの実機検証を強く推奨する。

- `ippsMalloc`系のアライメント保証値(32byte/64byteの記述の版差)
- `ippvm`コンポーネントのCMake Config Package上での既知の構成漏れ
- `ippsTanh_64f`の実在(32f版は文献で確認、64f版は命名規則からの推定)

