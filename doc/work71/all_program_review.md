はい。可能です。

ただし、この依頼は **ConvoPeq全体ではなく、コンボルバーだけでも数万～十数万行規模** の解析になります。現状の1回の応答では全文を解析することはできません。

また、先ほど更新された **最新版 ConvoPeq.md** を確認すると、コンボルバーは以下のように分割された実装になっています。

* ConvolverProcessor.Runtime.cpp
* ConvolverProcessor.LoadPipeline.cpp
* ConvolverProcessor.LoaderThread.cpp
* ConvolverProcessor.MixedPhase.cpp
* ConvolverProcessor.Rebuild.cpp
* ConvolverProcessor.ResampleAndFallback.cpp
* ConvolverProcessor.Lifecycle.cpp
* ConvolverProcessor.StateAndUI.cpp
* ConvolverProcessor.Internal.h
* MKLNonUniformConvolver.cpp
* MklFftEvaluator.h
* IRDSP.cpp
* DftiHandle.h

さらに確認したところ、

* Intel MKL DFTI
* Intel IPP FFT
* AVX2
* FTZ/DAZ
* MKL VML
* 独自NUCコンボルバー

が混在する構成になっています。

---

## 私が実施する解析

単なるコードレビューではなく、DSP実装レビューとして行います。

### ① アルゴリズム

* Overlap Save
* Overlap Add
* Non Uniform Convolution
* Partition設計
* Direct Head
* Tail処理
* latency算出
* Mixed Phase
* Minimum Phase
* Resampling

---

### ② Intel MKL

* DFTI使用方法
* Descriptor生成
* Commitタイミング
* destroy漏れ
* stride設定
* placement設定
* scaling設定
* workspace使用
* cache locality
* vectorization

---

### ③ Intel IPP

かなり重点的に見ます。

例えば

* ippsMalloc使用方法
* ippsFree対応
* Alignment
* CCS形式
* FFTSpec
* WorkBuffer
* FFT初期化
* FFT再利用
* FFT order
* 実数FFT
* Complex FFT
* inverse scaling
* inplace/outplace

などです。

最新版を見る限りでも、

MklFftEvaluatorでは

* FFTSpec初期化失敗
* WorkBuffer確保失敗

のガードが追加されています。

これは改善点ですが、さらに

* IPP APIの使い方そのもの
* 最新IPPに対して最適か
* 不要なコピー
* unnecessary malloc

まで確認します。

---

### ④ RT安全性

現在ConvoPeqはISR Runtime化されていますので

* Audio Thread
* LoaderThread
* Publication
* Retire
* Crossfade

との境界も確認します。

---

### ⑤ SIMD

* AVX2
* AVX512への移行余地
* gather
* cache line
* prefetch
* false sharing
* alignment

---

### ⑥ メモリ

* コピー回数
* move漏れ
* unique_ptr
* aligned allocation
* キャッシュ効率
* NUMA影響

---

### ⑦ 並列性

* LoaderThread
* MessageThread
* AudioThread

の競合

---

### ⑧ 数値解析

* 正規化
* NaN
* denormal
* clipping
* scaling
* FFT leakage
* phase wrap

---

### ⑨ Intel oneAPI観点

Intel推奨の

* IPP
* MKL
* icx

のベストプラクティスとの差異

---

## 現時点で見えているレビュー候補

最新版だけでも既に気になる箇所があります。

### 1. MKLとIPP FFTが混在

MixedPhaseではMKL DFTIを使用しています。

一方でMklFftEvaluatorではIPP FFTを使用しています。

これは

* FFT実装が二重
* 保守性低下
* キャッシュ特性の違い
* API差異

を生みやすい構造です。

---

### 2. FFTサイズ

MixedPhaseでは

```
fftSize = nextPowerOfTwo(numSamples * 4)
```

となっています。

4倍固定が妥当か

* メモリ効率
* FFT効率
* leakage
* convolution精度

を確認します。

---

### 3. LoaderThread

LoaderThreadで

```
VML_FTZDAZ_ON
```

を設定しています。

これは良い方向ですが

Audio Threadとの整合

MKL内部スレッド

IPP内部

まで確認します。

---

### 4. キャッシュ

MixedPhasePersistentCache

IR Cache

Persistent Cache

Memory Cache

の整合も見ます。

---

### 5. StereoConvolver

```
Add()

Get()
```

モデルになっています。

ここは

* リングバッファ
* XRUN
* copy
* branch prediction

まで解析します。

---

## 推奨する進め方

この規模では、一括レビューよりも体系的なレビューの方が品質が高くなります。

1. **ConvolverProcessor全体アーキテクチャ**
2. **MKLNonUniformConvolver詳細解析**
3. **Intel IPP使用箇所の完全監査**
4. **Mixed Phaseアルゴリズム**
5. **リアルタイム処理・ISR安全性**
6. **性能改善（CPU・メモリ・SIMD）**
7. **潜在バグ・境界条件・数値安定性**
8. **Intel oneAPIベストプラクティスとの差分評価**
9. **改善案（優先順位付き）**

という順番でレビューするのが最も効果的です。

この方法であれば、単なる指摘ではなく、**実装バグ・設計上の問題・Intel IPP/MKLの非効率な使い方・リアルタイム性・性能改善余地**まで含めて、かなり深いレベルまで洗い出せます。



了解しました。

まず **第1回** として、

> **① ConvolverProcessor 全体アーキテクチャレビュー**

を行います。

---

# 総合評価

現行実装は以前よりかなり改善されており、

* Audio Thread
* Loader Thread
* Message Thread

が明確に分離されています。さらに

* RCU
* Runtime Publication
* Split Compilation
* BuildSnapshot

なども導入されており、構造としてはかなり成熟しています。

ただし、アーキテクチャ上まだ改善余地があります。

---

# ① 良い点

## 1. RuntimeとBuildが完全分離

現在は

* Runtime.cpp
* LoaderThread.cpp
* LoadPipeline.cpp
* Rebuild.cpp
* MixedPhase.cpp

に役割分離されています。

これは以前の巨大ConvolverProcessor.cppより遥かに保守しやすい構成です。

評価

★★★★★

---

## 2. Audio Threadが極めて軽い

Audio Thread側は

* 新規確保なし
* FFT生成なし
* IRロードなし
* Buildなし

となっています。Runtime側でもリアルタイム経路で libm を避ける実装や有限値サニタイズが導入されています。

ISR設計として非常に良いです。

評価

★★★★★

---

## 3. StereoConvolverの抽象化

StereoConvolverが

* 左右NUC
* latency
* IR
* callQuantum

などを一元管理しています。

ConvolverProcessorが

```
StereoConvolver
    ↓
MKLNonUniformConvolver
```

だけを扱えばよくなっています。

これは良い設計です。

---

## 4. BuildSnapshot導入

BuildSnapshotをLoaderへ渡しているため

Build条件が固定されています。

これにより

* UI変更
* パラメータ変更

による途中変更を避けられます。

非常に良い設計です。

---

# ② 気になる点

ここから重要です。

---

## 問題1

ConvolverProcessorの責務がまだ多い

現状、

ConvolverProcessorは

* Loader管理
* MixedPhase
* Resample
* Publication
* Cache
* Runtime
* UI通知
* Waveform生成

を持っています。

つまり

DSPエンジン

ではなく

巨大オーケストレーター

になっています。

これは保守性を落とします。

---

### 理想

```
ConvolverProcessor

    ↓

LoaderManager

    ↓

IRBuilder

    ↓

PublicationManager

    ↓

RuntimeEngine
```

くらいまで分離するとさらに良くなります。

---

## 問題2

StereoConvolverが状態を持ち過ぎ

StereoConvolverには

* IR
* latency
* scale
* block size
* stored parameters

まで保存されています。

実質

Engine + Metadata

になっています。

本来は

```
StereoConvolver

↓

Engine only
```

が望ましいです。

Metadataは別構造体にした方が責務が明確になります。

---

## 問題3

IR処理が複数箇所に分散

IR変換が

* LoaderThread
* MixedPhase
* Resample
* Rebuild

に散っています。

現在でも理解可能ですが、

IR Builder

という一つのクラスへ集約した方が

保守性がさらに向上します。

---

## 問題4

Publication経路がやや複雑

現在

```
Loader

↓

commitNewConvolver

↓

applyNewState

↓

Publish

↓

Notify
```

という段階があります。

安全性は高い一方、

処理経路が長く

追跡が難しくなっています。

設計としては悪くありませんが、

状態遷移図が無いと理解しづらいです。

---

## 問題5

キャッシュ責務

ConvolverProcessorが

```
IR Cache
```

を直接持っています。

将来的には

```
IRCacheManager
```

として独立した方が拡張性があります。

---

# ③ Intel観点

Intel oneAPI設計として見ると

かなり良いです。

理由

* FFTはEngine側
* BuildはLoader
* Audio ThreadでFFT生成なし
* aligned allocation
* MKL使用

と基本方針が整理されています。

Intelが推奨する

> Build once
>
> Execute many

の考え方に近づいています。

---

# ④ 改善優先順位

現時点での優先順位は以下です。

| 項目                     |   重要度 |
| ---------------------- | ----: |
| ConvolverProcessor責務分離 | ★★★★★ |
| IR Builder独立           | ★★★★★ |
| StereoConvolver軽量化     | ★★★★☆ |
| Cache Manager独立        | ★★★★☆ |
| Publication経路簡素化       | ★★★☆☆ |

---

# 結論

現行アーキテクチャは、リアルタイムオーディオ処理としては堅実に設計されており、**Audio Threadとバックグラウンド処理の分離、RCUによる公開、BuildSnapshotによる一貫性確保**など、重要な設計要件は概ね満たされています。

一方で、**ConvolverProcessorに責務が集中している点**が最も大きな設計上の課題です。これは直ちに性能や安定性へ悪影響を与える問題ではありませんが、今後の機能追加や保守性を考えると、IR構築・キャッシュ・Publicationなどを段階的に独立コンポーネント化する余地があります。

次回は **② MKLNonUniformConvolver の詳細解析** を行い、パーティション構成、FFT処理、リングバッファ、Direct Head、Tail処理、計算量、メモリアクセス、潜在的な性能・実装上の問題点まで踏み込んでレビューします。



## 第2回: MKLNonUniformConvolver 詳細解析

今回は **コンボルバーの中核である `MKLNonUniformConvolver`** を中心に解析します。

現時点での評価は次のとおりです。

| 項目          |    評価 |
| ----------- | ----: |
| アルゴリズム      | ★★★★★ |
| リアルタイム性     | ★★★★★ |
| メモリ設計       | ★★★★☆ |
| Intel IPP利用 | ★★★★☆ |
| SIMD効率      | ★★★★☆ |
| 保守性         | ★★★☆☆ |

DSPエンジンとしてはかなり完成度が高い一方で、いくつか改善余地があります。

---

# 1. Non-Uniform Partition設計

実装は典型的な

```
Layer0
Layer1
Layer2
```

構造になっています。各レイヤーが個別のFFTサイズ、FDL、FFTワークバッファ、累積バッファなどを保持しています。`Layer::freeAll()`からもレイヤー単位でリソースを所有していることが確認できます。

さらに、

* partition毎のFDL
* tailOutput
* input accumulator
* overlap

もレイヤー単位になっています。

これはIntelや研究論文で一般的な **Hierarchical Non-Uniform Partitioned Convolution** に近い構成です。

---

# 2. FFT Plan管理

これは非常に良い実装です。

現在は

```
IppFFTPlan

↓

IppFFTPlanCache

↓

Layer
```

という構成になっています。

つまり

FFTSpecを毎回生成せず

```
FFT Size

↓

共有Plan
```

になっています。

これはIPP推奨実装です。

以前よりかなり改善されています。

---

# 3. Audio ThreadでFFT生成なし

SetImpulseで

* FFT Plan
* WorkBuffer
* IR FFT

まで構築しています。Audio Thread側は `Add()` / `Get()` のみを実行する前提です。

これはRT設計として理想的です。

---

# 4. メモリ配置

ほぼ全バッファが

```
mkl_malloc(...,64)
```

で確保されています。`Layer::freeAll()` や `releaseAllLayers()` の解放処理からも64バイトアライン前提の設計が読み取れます。

64Byte Alignmentなので

AVX2

だけでなく

AVX512

にも対応可能です。

これは高評価です。

---

# 5. Layer所有データ

Layerは

かなり多くの状態を持っています。

例えば

* FFT
* FDL
* IR
* overlap
* work buffer
* accumulator
* tail

などです。

DSP的には正しいですが

Layerクラスが巨大化しています。

保守性としては少し気になります。

---

# 6. Ring Buffer

releaseAllLayersを見ると

```
m_ringBuf
```

が

Engine全体に1つあります。

これは

Layer毎ではありません。

つまり

Direct Head

↓

Layer0

↓

Layer1

↓

Layer2

を

一つの入力リングで駆動しています。

これはメモリ効率として非常に良いです。

---

# 7. Direct Head

現在

```
m_directIRRev

m_directHistory

m_directWindow

m_directOutBuf
```

が独立しています。

つまり

短IR

↓

FFTを使わない

という設計です。

これは

Intel

JUCE

REAPER

でも採用される方式です。

---

# 気になる点①

## Layerが大きすぎる

Layerに

```
FFT

FDL

Overlap

Tail

Accumulator

Input

Output

State
```

が全部あります。

DSPとしては成立していますが

L1 Cache

の観点では

構造体が巨大になります。

---

### 推奨

```
LayerFFT

LayerBuffers

LayerState
```

程度には分割できます。

---

# 気になる点②

## releaseAllLayers()

```
for Layer

↓

freeAll()
```

しています。

安全ですが

大量IR切替時

大量free

になります。

Message ThreadなのでRT影響はありません。

ただし

OS Heap

断片化

には多少影響します。

---

### 改善案

Pool化

または

Reusable Layer

を持つと

IR切替高速化できます。

---

# 気になる点③

## Layer毎に大量malloc

Layerには

10個以上のバッファがあります。

例えば

```
FFT

FDL

Accum

Tail

Prev

Real

Imag
```

などです。

IRロード時は

かなりmallocします。

---

改善案

```
One Large Arena
```

方式です。

Intel oneAPIも

Arena Allocation

を推奨しています。

---

# 気になる点④

## Real/Imag分離

現在

```
irFreqDomain

irFreqReal

irFreqImag
```

を保持しています。

つまり

AoS

SoA

両方あります。

これは

計算高速化

とのトレードオフです。

---

しかし

保持メモリは

約2倍近くになります。

---

改善案

必要時のみ

Split

する方法もあります。

ただし

AVX2では現在方式にも利点があります。

---

# 気になる点⑤

## MKL + IPP混在

現在

FFT

↓

IPP

フィルタ

↓

MKL VML

スケーリング

↓

CBLAS

という構成です。

これは

悪い実装ではありません。

しかし

Intelライブラリが

2種類混在しています。

---

例えば

FFT

IPP

↓

Complex Multiply

自前

↓

VML

↓

CBLAS

↓

IPP

となります。

ライブラリ切替が多く

保守性は少し落ちます。

---

# 気になる点⑥

## Complex Multiply

コードを見る限り

複素乗算は

自前AVX実装があります。

Intel IPPには

複素ベクトル積

があります。

現在実装との性能比較は

行う価値があります。

必ずしもIPPの方が速いとは限りませんが、

CPU世代によってはIPP実装が有利になるケースがあります。

---

# Intel oneAPI観点

Intelの推奨と比較すると

非常に近い構成です。

良い点

* FFT Plan共有
* WorkBuffer共有
* Audio Thread mallocゼロ
* 64Byte Alignment
* Single Thread FFT
* MKL Thread=1

はいずれもIntelのベストプラクティスと整合しています。

改善余地としては

* Layer Arena Allocation
* Complex演算のIPPとの比較評価
* Layer構造体の軽量化

程度です。

---

## 総合評価

`MKLNonUniformConvolver` は、**リアルタイム性を最優先に設計された成熟度の高い実装**です。FFTプランの共有、64バイトアラインメント、Audio Threadでの動的メモリ確保排除、レイヤー化された非一様パーティションなど、基本設計は適切です。

現時点で重大な設計上の欠陥は見当たりません。一方で、Layerの責務集中、IRロード時の多数のメモリ確保、複数ライブラリの混在、複素演算の実装方式については、今後さらに最適化できる余地があります。

次回は **③ Intel IPP使用箇所の完全監査** を行い、`ippsFFTGetSize_R_64f`、`ippsFFTInit_R_64f`、`ippsFFTFwd_RToCCS_64f`、`ippsFFTInv_CCSToR_64f` を含め、IPP APIの使用方法がIntelの推奨に沿っているか、性能・安全性・メモリ効率の観点から詳細に検証します。



## 第3回: Intel IPP使用箇所の完全監査

今回は **Intel IPP の使用方法そのもの** をレビューします。

結論から言うと、

> **IPP API の使い方は概ね正しく、以前見られた危険な実装はかなり改善されています。**

ただし、**まだ改善できる箇所**が複数あります。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| FFT初期化  | ★★★★★ |
| FFT利用方法 | ★★★★★ |
| CCS利用   | ★★★★★ |
| エラー処理   | ★★★★☆ |
| メモリ効率   | ★★★★☆ |
| IPP最適化  | ★★★☆☆ |

---

# 1. FFT初期化

現在

```cpp
ippsFFTGetSize_R_64f()

↓

ippsFFTInit_R_64f()
```

という流れになっています。

これは

Intel推奨そのままです。

さらに

```
sizeSpec

sizeInit

sizeWork
```

を取得しています。

これも正しいです。

---

## 良い点

以前は

```
FFTInit

↓

そのまま使用
```

だったようですが

現在は

* 戻り値

* fftSpec==nullptr

まで確認しています。

これは非常に良い改善です。

---

# 2. WorkBuffer

現在

```
sizeWork

↓

ippsMalloc_8u()
```

しています。

さらに

nullptr

も検査しています。

これはIntel推奨です。

---

### ただし改善点

現状

```cpp
fftSpec=nullptr;
```

へ落としています。

つまり

FFT自体を使わなくなります。

安全ですが

理由が

OOM

なのか

FFTInit失敗

なのか

区別できません。

---

改善案

```cpp
enum FFTStatus

Ready

OutOfMemory

InitError

Unsupported
```

くらいは持った方が

ログ解析が容易になります。

---

# 3. FFTSpec寿命

現在

Constructor

↓

一回生成

↓

Destructor

↓

Free

です。

これはIPPとして理想です。

FFTSpecは

何度も生成すべきではありません。

---

# 4. FFT Forward

現在

```
ippsFFTFwd_RToCCS_64f()
```

を使っています。

これは

Real FFT

として正しいです。

---

しかも

```cpp
double

↓

CCS
```

なので

無駄がありません。

---

# 5. CCS利用

これはかなり良いです。

現在

```
CcsComplex
```

を定義しています。

さらに

```
static_assert
```

があります。

つまり

IPP CCS

↓

reinterpret_cast

↓

CcsComplex

が

安全になっています。

これは高く評価できます。

---

# 6. MKL_Complex16廃止

これは改善です。

以前

```
MKL_Complex16
```

依存でした。

現在

```
CcsComplex
```

です。

つまり

MKL依存が減りました。

---

# 7. Audio Thread

Audio Threadでは

```
FFTInit

無し
```

です。

Forwardのみです。

理想です。

---

# 気になる点①

## CCS→SoA変換

現在

```
Forward FFT

↓

CCS

↓

deinterleaveComplex()
```

しています。

つまり

```
Interleaved

↓

Split
```

です。

---

これは

コピー

が発生します。

---

さらに

IRでも

同じことをしています。

つまり

IRロード時も

毎回

Splitしています。

---

改善候補

AVX実装が

AoS対応なら

Split不要になります。

---

# 気になる点②

## AoSとSoA両保持

現在

Layerには

```
irFreqDomain

irFreqReal

irFreqImag
```

があります。

つまり

同じFFT結果を

2形式

保持しています。

---

メリット

AVX高速

---

デメリット

IRメモリ約2倍

---

IRが長いほど

効いてきます。

---

# 気になる点③

## memset

processLayerBlockでは

```
memset

accumBuf

accumReal

accumImag
```

しています。

IPPには

```
ippsZero_64f()
```

があります。

---

### ただし

ここは

必ずしも

IPPへ変えるべきではありません。

近年CPUでは

libcの

memset

は非常に高速です。

ここは

ベンチマークして決めるべきです。

---

# 気になる点④

## memcpy

Mirror Write

では

```
memcpy()
```

しています。

一方

別経路では

```
FloatVectorOperations::copy
```

もあります。

実装が統一されていません。

---

どちらかに統一した方が

保守性は上がります。

---

# 気になる点⑤

## IPP演算API未利用

現在

FFTだけ

IPPです。

しかし

Intel IPPには

例えば

* 複素乗算
* 複素加算
* Magnitude
* Power
* Vector Multiply

があります。

現在は

AVX自前実装が中心です。

---

これは

悪いことではありません。

最近のIntel CPUでは

自前AVX2が

IPPより速いケースもあります。

---

しかし

一度

性能比較

は実施すべきです。

---

# 気になる点⑥

## FFT Order固定

現在

```
4096

↓

Order12
```

固定です。

MklFftEvaluatorなので

問題ありません。

しかし

将来

8192

16384

へ拡張するなら

テンプレート化

または

動的生成

も検討できます。

---

# Intel Best Practice比較

Intel資料と比較すると

かなり近いです。

実施済み

* FFTSpec再利用
* WorkBuffer再利用
* FFTInit一回
* CCS利用
* Real FFT
* 64Byte Alignment
* Audio Thread生成無し

はいずれも適切です。

---

## 私が最も気になる点

今回の監査で最も気になったのは、

**FFT API の使い方ではなく、FFT後のデータレイアウト変換**です。

現在は

```
FFT(CCS AoS)

↓

deinterleave()

↓

Real[]

Imag[]
```

という流れになっています。

この変換は、各レイヤー・各パーティションで繰り返されるため、CPU時間だけでなくキャッシュ帯域も消費します。

ただし、これを即座に削除すべきとは言えません。後段のAVX2複素演算がSoA（Real/Imag分離）を前提として最適化されているなら、この変換コスト以上の性能向上を得ている可能性があります。そのため、

1. **AoS(CCS)のまま複素演算**
2. **現状のAoS→SoA変換＋SoA演算**

の2方式を実測比較して判断するのが適切です。現時点では、この点がIntel IPP利用に関する最も重要な性能検証項目です。

---

### 現時点での優先順位

| 改善項目                                        |   優先度 |
| ------------------------------------------- | ----: |
| AoS→SoA変換コストの実測評価                           | ★★★★★ |
| IPP複素演算との性能比較                               | ★★★★☆ |
| AoS/SoA二重保持の必要性再評価                          | ★★★★☆ |
| FFT状態管理の細分化                                 | ★★★☆☆ |
| `memcpy`/`FloatVectorOperations::copy`の方針統一 | ★★★☆☆ |

次回は **④ Mixed Phase アルゴリズムの詳細解析** を行い、位相再構成、最小位相化、全域通過補正、キャッシュ設計、数値安定性、FFT/IFFT処理、アルゴリズムの妥当性を中心にレビューします。



## 第4回: Mixed Phase アルゴリズム詳細解析

今回は **`ConvolverProcessor.MixedPhase.cpp`** を中心に、Mixed Phase IR生成アルゴリズムを解析します。

結論から述べると、

> **DSPアルゴリズム自体はかなり高度で、実用レベルとして完成度が高い実装です。**

一方で、**計算量・FFT構成・キャッシュ戦略・最適化アルゴリズムの切り替え**には改善余地があります。

---

# 総合評価

| 項目        |    評価 |
| --------- | ----: |
| DSPアルゴリズム | ★★★★★ |
| 位相設計      | ★★★★★ |
| 数値安定性     | ★★★★☆ |
| キャッシュ設計   | ★★★★★ |
| 計算効率      | ★★★★☆ |
| 保守性       | ★★★☆☆ |

---

# 1. キャッシュ設計

これは非常に良い設計です。

Mixed Phase生成では

1. メモリキャッシュ確認
2. 永続キャッシュ確認
3. 最適化実行
4. メモリキャッシュ保存
5. ディスクキャッシュ保存

という多段構成になっています。

さらにディスクキャッシュは

```
fileHash
sampleRate
phaseMode
f1
f2
targetLength
```

をキーにしています。

これは十分妥当です。

---

# 2. FFT構成

Mixed Phaseでは

```
fftSize = nextPowerOfTwo(numSamples * 4)
```

を採用しています。

### 良い点

ゼロパディングを十分確保できるため、

* 周波数分解能
* IFFT折り返し
* Allpass設計

には有利です。

---

### 気になる点

4倍固定です。

IR長によっては

```
65536

↓

262144 FFT
```

になります。

非常に大きなFFTです。

---

改善候補

例えば

```
×2

×4

×8
```

を

IR長で切り替える方法です。

特に長いIRではメモリ消費と計算時間を削減できます。

---

# 3. MKL DFTI

FFTは

```
DftiCreateDescriptor()

↓

Commit

↓

Forward

↓

Backward
```

となっています。

Backward Scaleも設定されています。

これはMKL推奨です。

---

# 4. Allpass設計

現在

```
Linear IR

↓

Group Delay

↓

AllpassDesigner

↓

Response

↓

LinearSpec × Allpass

↓

IFFT
```

という流れです。

DSP的に非常に綺麗です。

Magnitudeを変えず

Phaseだけ変更しています。

---

# 5. 最適化アルゴリズム

ここは面白い設計です。

現在

通常

```
CMA-ES
```

ライブ

```
GreedyAdaGrad
```

となっています。

さらに

```
CMAES失敗

↓

GreedyAdaGrad
```

へフォールバックしています。

非常に堅牢です。

---

# 6. 永続キャッシュ

さらに

```
CMAES成功時のみ

↓

保存
```

しています。

これは非常に合理的です。

Greedy結果は

局所解

なので保存しない判断は妥当です。

---

# 7. Allpass Response

現在

```
computeResponse()

↓

Mirror

↓

Conjugate
```

しています。

Hermitian対称を維持しています。

DSP的に正しいです。

---

# 8. RMS補正

IFFT後

RMS

Peak

を補正しています。

さらに

```
peak > 0.99

↓

0.98
```

へ抑えています。

これはクリッピング防止として適切です。

---

# 気になる点①

## FFTサイズ固定

これが一番気になります。

例えば

```
IR 32768

↓

FFT131072
```

になります。

---

Mixed Phase設計では

そこまで巨大FFTが

本当に必要か

検証価値があります。

---

改善候補

```
FFT =

max(
nextPow2(IR×2),

8192)
```

程度でも

十分なケースがあります。

---

# 気になる点②

## 周波数ベクトル生成

現在

毎回

```
freq_hz[]

```

を生成しています。

FFTサイズが同じなら

毎回同じです。

---

改善

FFTサイズごと

Static Cache

できます。

---

# 気になる点③

## computeResponse()

毎回

```
std::complex
```

を大量生成しています。

これも

SIMD化余地があります。

---

例えば

```
Real[]

Imag[]
```

で保持すれば

AVX2

AVX512

が使えます。

---

# 気になる点④

## std::complex

MixedPhaseでは

かなり

```
std::complex<double>
```

を使っています。

---

MKL

IPP

AVX

を使うなら

これは

最適ではありません。

---

Intel CPUでは

SoA

の方が

SIMD化しやすいです。

---

# 気になる点⑤

## Progress

現在

```
0.1

↓

0.9

↓

1.0
```

です。

しかし

CMAES160世代

Greedy50反復

では

途中進捗がありません。

UI的には

止まって見えます。

---

改善

Generation毎

Iteration毎

Progress更新

---

# 気になる点⑥

## メモリキャッシュ

現在

```
AudioBuffer

+

Allpass Sections
```

保存しています。

しかし

```
computeResponse
```

結果は保存していません。

---

これは

再利用余地があります。

---

# 気になる点⑦

## 例外処理

最後は

```
catch(...)
```

です。

安全ですが

原因解析には弱いです。

---

可能なら

* MKLエラー
* メモリ不足
* 設計失敗

を分類すると保守性が向上します。

---

# 私が最も評価した点

今回の解析で最も完成度が高いと感じたのは、

**Mixed Phase生成処理が完全にAudio Threadの外へ隔離されていること**です。

最適化（CMA-ES／GreedyAdaGrad）、FFT、キャッシュI/Oはいずれもローダー側で完結し、Audio Threadには生成済みIRのみが公開されます。さらにメモリキャッシュ・永続キャッシュ・フォールバック戦略まで備えており、**高負荷なオフライン最適化とリアルタイムDSPが明確に分離**されています。

---

## 私が最も改善効果が大きいと考える点

優先順位は以下です。

| 改善項目                      |   優先度 |
| ------------------------- | ----: |
| FFTサイズをIR長に応じて可変化         | ★★★★★ |
| `freq_hz`テーブルのキャッシュ化      | ★★★★★ |
| `computeResponse()`のSIMD化 | ★★★★☆ |
| `std::complex`依存の削減       | ★★★★☆ |
| 進捗更新の高粒度化                 | ★★★☆☆ |
| 例外原因の細分化                  | ★★★☆☆ |

このうち、**FFTサイズの見直し**は最も効果が大きい可能性があります。ただし、Mixed Phase設計精度とのトレードオフがあるため、変更前に位相誤差・群遅延誤差・最適化収束性を含めた比較評価を行うことを推奨します。



## 第5回: リアルタイム処理（Audio Thread / ISR安全性）詳細レビュー

今回は **`ConvolverProcessor.Runtime.cpp`** を中心に、リアルタイムオーディオ処理の安全性を解析します。

結論から言うと、

> **ConvoPeq のコンボルバーは、以前の実装と比較すると ISR（Interrupt Service Routine）を非常に強く意識した設計へ進化しています。**

特に

* Audio Thread
* Loader Thread
* Message Thread

の責務分離はかなり完成度が高くなっています。

---

# 総合評価

| 項目           |    評価 |
| ------------ | ----: |
| ISR安全性       | ★★★★★ |
| RTメモリアロケーション | ★★★★★ |
| ロック回避        | ★★★★★ |
| Atomic設計     | ★★★★☆ |
| キャッシュ効率      | ★★★★☆ |
| 将来拡張性        | ★★★★☆ |

---

# 1. Audio Threadでlibmを排除

`Runtime.cpp` 冒頭を見ると

```cpp
equalPowerSin()
floorNoLibm()
isFiniteAndAbsBelowNoLibm()
```

などが独自実装されています。

これは非常に良い判断です。

通常

```cpp
std::sin()
std::floor()
std::isfinite()
```

は

libm

↓

OS DLL

↓

CPU dispatch

を経由する可能性があります。

現在は

Audio Thread専用実装になっています。

これはプロ用DSPとして理想です。

---

# 2. NaNサニタイズ

Runtimeでは

```cpp
sanitizeFiniteChunk()
```

があります。

つまり

```
NaN

↓

Inf

↓

巨大値
```

を

0

へ戻しています。

---

これは

コンボルバーでは非常に重要です。

FFTは

NaN

一つで

全部壊れます。

---

# 3. Atomic Memory Order

最新版では

かなり丁寧です。

例えば

```
publishAtomic(..., release)

↓

consumeAtomic(..., acquire)
```

が多数あります。

以前のような

```
memory_order_seq_cst
```

乱用ではありません。

これは良い改善です。

---

# 4. Loader Thread

Loaderでは

```
_MM_SET_FLUSH_ZERO_MODE

_MM_SET_DENORMALS_ZERO_MODE

VML_FTZDAZ_ON
```

を設定しています。

これはIntel CPUでは重要です。

Denormalが大量発生すると

CPUが極端に遅くなります。

---

# 5. Audio Threadではmalloc無し

ロード側で

```
StereoConvolver::init()

↓

IR FFT

↓

Layer

↓

WorkBuffer
```

まで生成しています。

Runtimeでは

生成処理はありません。

理想です。

---

# 6. Latency更新

現在

```
latencyResetPendingGen

↓

Audio Thread
```

方式です。

つまり

Message Threadが

SmoothedValue

へ触っていません。

以前よりかなり安全です。

---

# 7. IR Publication

現在

```
Loader

↓

applyNewState()

↓

publishAtomic()

↓

Audio Thread
```

です。

RCUとして綺麗です。

---

# 8. Placeholder DSP

prepareToPlayでは

Placeholder DSP

を先にPublishしています。

つまり

初回IRロード前でも

Runtimeが存在します。

これはXRUN防止に効きます。

---

# 気になる点①

## Atomic変数が非常に多い

ConvolverProcessorだけでも

かなり大量です。

例えば

```
isLoading

isRebuilding

irFinalized

loadProgress

latencyChange

uiLatency

...
```

などがあります。

---

一つ一つは軽いですが

Cache Line

が増えます。

---

改善案

更新頻度が近い状態は

構造体へまとめることです。

---

# 気になる点②

## UI Atomic

例えば

```
uiAlgorithmLatency

uiPeakLatency

uiTotalLatency
```

です。

これらは

UIしか見ません。

Audio Threadとは

更新周期が違います。

---

Snapshot化

してまとめてPublishすると

Atomic回数を減らせます。

---

# 気になる点③

## sanitizeFiniteChunk()

毎回

```
isfinite

↓

abs

↓

if
```

しています。

AVX2なら

8サンプル

まとめて判定できます。

---

改善余地があります。

---

# 気になる点④

## Crossfade

現在

Equal Power

近似です。

これは悪くありません。

---

ただし

AVX2

なら

8サンプルずつ

処理できます。

---

# 気になる点⑤

## Loader終了

現在

```
stopThread(500)
```

です。

Message Threadなので

RT影響はありません。

しかし

500ms

待つ設計は

終了時に

UI停止が起こる可能性があります。

---

# 気になる点⑥

## cleanup()

LoaderThreadを

```
waitForThreadToExit(0)
```

で掃除しています。

安全ですが

大量IR切替時

Threadオブジェクトが

少し残ります。

---

ThreadPool

方式も検討できます。

---

# 気になる点⑦

## Progress通知

現在

```
publishAtomic(progress)

↓

postCoalescedChangeNotification()
```

です。

Mixed Phase中は

通知頻度が

やや粗い印象です。

UI体験改善の余地があります。

---

# Intel CPU観点

Intel CPUでは

現在実装はかなり良いです。

実施済み

* FTZ
* DAZ
* VML_FTZDAZ_ON
* malloc無し
* release/acquire
* RCU
* Placeholder Runtime

はいずれもリアルタイム設計として適切です。

---

## 私が最も評価した点

今回最も評価したのは、

**Audio Threadから「重い処理」を徹底的に排除していること**です。

Audio Threadでは

* FFTプラン生成なし
* IRロードなし
* メモリ確保なし
* Mixed Phase計算なし
* `libm`依存を避けた演算
* RCUによる状態取得

という方針が一貫しています。これはリアルタイムオーディオソフトウェアとして非常に重要であり、現在の設計はその原則をよく守っています。

---

## 私が最も改善効果が大きいと考える点

ここで最も改善余地があるのは**アルゴリズムではなく、マイクロアーキテクチャ**です。

優先順位は次のとおりです。

| 改善項目                                    |   優先度 |
| --------------------------------------- | ----: |
| `sanitizeFiniteChunk()` のAVX2ベクトル化      | ★★★★★ |
| UI向けAtomic状態の集約・Snapshot化               | ★★★★☆ |
| Crossfade処理のSIMD化                       | ★★★★☆ |
| Atomic変数のキャッシュライン配置最適化（false sharing対策） | ★★★★☆ |
| LoaderThreadの再利用（ThreadPool化）           | ★★★☆☆ |

### 特に重要な指摘: false sharing の可能性

今回の解析で性能面から最も気になったのは、**多数の `std::atomic` が同一オブジェクト内に密集している点**です。

正しさには問題ありませんが、Audio Thread・Loader Thread・Message Thread が異なるAtomicを書き換えていても、それらが同じキャッシュライン上に配置されると、**キャッシュラインの所有権がCPUコア間で頻繁に移動し、不要なコヒーレンシトラフィック**が発生する可能性があります。

この問題はコード上だけでは断定できませんが、Intel VTune の **False Sharing** や **Memory Access** 解析で確認する価値があります。もし頻繁に更新されるAtomicが同一キャッシュラインに載っている場合は、`alignas(64)` や更新頻度別の構造体分離により改善できる可能性があります。

次回は **第6回: CPU性能・SIMD・キャッシュ効率の詳細解析** を行い、AVX2実装、メモリアクセスパターン、Intel CPUでの実行効率、さらなる高速化余地を重点的にレビューします。



## 第6回: CPU性能・SIMD・キャッシュ効率 詳細解析

今回は **Intel CPU（AVX2/MKL/IPP）を前提**として、コンボルバー全体のCPU効率を解析します。

結論から述べると、

> **アルゴリズムよりも、CPUマイクロアーキテクチャ最適化に改善余地があります。**

DSPとしては成熟していますが、Intel CPUを100%使い切っているとはまだ言えません。

---

# 総合評価

| 項目           |    評価 |
| ------------ | ----: |
| AVX利用        | ★★★★☆ |
| キャッシュ効率      | ★★★★☆ |
| SIMD化        | ★★★★☆ |
| メモリ帯域        | ★★★★☆ |
| 分岐予測         | ★★★★★ |
| Intel CPU最適化 | ★★★★☆ |

---

# 1. 64Byte Alignment

全体を確認すると

```cpp
mkl_malloc(...,64)
```

が徹底されています。`MKLNonUniformConvolver` の各レイヤーやFFT関連バッファも64バイトアラインメントで確保されています。

これは非常に良いです。

Intel CPUでは

* AVX2
* AVX-512

どちらでも十分なアラインメントです。

---

# 2. FFT Plan共有

FFTSpecが

```
Plan Cache

↓

Layer共有
```

になっています。

FFT生成は高コストなので

これは正しい設計です。

---

# 3. Layer単位処理

現在

Layerごとに

```
入力

↓

FFT

↓

Multiply

↓

IFFT

↓

Overlap
```

となっています。

CPUから見ると

局所性は悪くありません。

---

# 4. Direct Head

短IRを

時間領域

長IRを

FFT

へ分離しています。

Intel CPUでは

非常に効率的です。

---

# 5. FFT WorkBuffer再利用

毎回

```
malloc

↓

free
```

していません。

これは

L3 Cache

にも優しいです。

---

# 気になる点①

## Layer構造体が巨大

Layerには

* FFT
* IR
* FDL
* overlap
* accum
* tail
* work

など多数のメンバーがあります。

---

CPUから見ると

Layerを読むだけで

L1 Cache

L2 Cache

へ大量ロードします。

---

改善案

```
LayerState

LayerBuffers

LayerFFT
```

へ分離。

---

# 気になる点②

## SoA/AoS変換

現在

```
CCS

↓

Real[]

Imag[]
```

へ展開しています。

これは

コピーが発生します。

---

Intel CPUでは

このコピーが

L2帯域

を消費します。

---

改善案

AoSのまま

AVX実装

との比較ベンチマークを行うことです。

---

# 気になる点③

## std::complex

Mixed Phaseでは

```
std::complex<double>
```

を多用しています。

---

Intel Compilerでも

自動SIMD化される場合はありますが

完全ではありません。

---

SoA

へ変更すると

AVX2化しやすくなります。

---

# 気になる点④

## gather命令

現在のコードを見る限り

複素演算は

```
Real[]

Imag[]
```

別配列です。

これは

Gather不要

です。

---

これは

非常に良い設計です。

---

# 気になる点⑤

## memset

多数あります。

```
memset

↓

Accum

↓

Tail
```

などです。

---

Intel CPUでは

memset自体は速いですが

メモリ帯域を消費します。

---

改善候補

Dirty Flag

方式です。

---

# 気になる点⑥

## memcpy

Mirror Buffer

では

```
memcpy
```

しています。

---

長IRでは

かなり帯域を使います。

---

RingBuffer

だけで

Mirror不要

になるか検討できます。

---

# 気になる点⑦

## Branch

DSP処理は

if

が非常に少ないです。

これは

Branch Predictor

に優しいです。

---

# 気になる点⑧

## False Sharing

Audio Thread

Loader

UI

が

Atomicを共有しています。

---

64Byte境界へ

分離すると

改善可能です。

---

# 気になる点⑨

## Prefetch

現在

```
_prefetch()
```

系は

ありません。

---

長IRでは

FDL

IR

Overlap

の読み込みがあります。

---

Intel CPUでは

```
_mm_prefetch()
```

が効く場合があります。

ただし、**近年のIntel CPUではハードウェアプリフェッチャが十分に機能するケースが多く、手動プリフェッチが常に有利とは限りません。** 特にアクセスパターンが連続的で予測しやすい場合は、逆効果になることもあります。導入するならVTune等でL2/L3ミス率を確認したうえで判断すべきです。

---

# 気になる点⑩

## NUMA

現状

NUMA対応は

ありません。

---

通常の

デスクトップ

では不要です。

---

Xeon

Dual Socket

なら

改善余地があります。

---

# Intel VTuneで確認したい項目

現状コードで最も確認したいのは次の5項目です。

| 項目                       |   優先度 |
| ------------------------ | ----: |
| L1 Data Cache Miss       | ★★★★★ |
| False Sharing            | ★★★★★ |
| Memory Bandwidth         | ★★★★★ |
| Vectorization Efficiency | ★★★★☆ |
| Branch Mispredict        | ★★★☆☆ |

---

# Intel oneAPIとの比較

Intelが推奨するFFTアプリケーション設計と比較すると、

実装済みなのは

* FFT Plan再利用
* WorkBuffer再利用
* 64Byte Alignment
* Single Thread FFT
* Audio Thread mallocゼロ
* Layer分割

です。

---

不足している可能性があるもの

* VTuneベースのキャッシュ最適化
* メモリアリーナ化
* 頻繁に更新するAtomicのキャッシュライン分離
* SoA/AoS変換コストの実測評価

---

## 今回最も重要だと考える指摘

性能改善候補の中で、**最も効果が大きい可能性があるのは「メモリ帯域の削減」**です。

現状は演算性能よりも、

* AoS→SoA変換
* `memcpy`
* `memset`
* FDLやオーバーラップバッファの読み書き

といった**データ移動**が支配的になっている可能性があります。Intel CPUでは、AVX2演算器そのものよりもL1/L2/L3キャッシュやメモリ帯域がボトルネックになるケースは少なくありません。

そのため、今後は演算命令の最適化よりも、**「どれだけデータ移動を減らせるか」**という視点でVTuneによる実測評価を行うことを推奨します。

---

次回は **第7回: 潜在バグ・境界条件・数値安定性の徹底レビュー** を行い、クラッシュ要因、NaN伝播、FFT境界条件、IR長の極端なケース、例外処理、オーバーフロー、IPP/MKL APIのエラーパスまで含めて詳細に解析します。


## 第7回: 潜在バグ・境界条件・数値安定性の徹底レビュー

今回は最新版の `ConvoPeq.md` を基に、**実際にクラッシュや音飛び・数値異常につながる可能性がある箇所**を重点的にレビューします。

最初に結論です。

> **以前指摘した重大な不具合の多くは修正されています。**
>
> 現在残っている課題は、「設計バグ」よりも「極端条件」「数値安定性」「保守性」に関するものが中心です。

---

# 総合評価

| 項目      |    評価 |
| ------- | ----: |
| クラッシュ耐性 | ★★★★★ |
| 境界条件    | ★★★★☆ |
| 数値安定性   | ★★★★☆ |
| OOM耐性   | ★★★★☆ |
| 例外安全性   | ★★★★☆ |
| 異常入力耐性  | ★★★★☆ |

---

# ① リングバッファ Overflow

これは改善されています。

以前レビューした

```
ringWrite()

overflow

↓

writePos更新
```

の問題は

```
m_ringWrite
```

を二重更新しないよう修正されています。ソースにも「overflow 時の追加更新は不要」と明記されています。

これは完全に正しい修正です。

---

# ② Audio Threadの境界チェック

`Add()`を見ると

```
if (!ready)
    return;
```

さらに

```
numSamples<=0
```

も処理されています。

その後

```
consumed

inputPos

partSize
```

も細かく管理されています。

さらに

```
consumed > numSamples
```

まで安全ガードがあります。

以前よりかなり堅牢です。

---

# ③ FFTサイズ上限

MixedPhaseでは

```
MAX_MIXED_FFT_SIZE
```

が導入されています。

巨大IRでも

無限FFT

にはなりません。

これは非常に良い改善です。

---

# ④ 入力検証

MixedPhaseでは

```
numSamples

numChannels

sampleRate

transition
```

をすべて確認しています。

つまり

異常IR

では

即終了します。

---

# ⑤ Runtime Build

RuntimeBuilderでは

```
sampleRate<=0

blockSize<=0
```

を拒否しています。

さらに

```
bad_alloc

catch(...)
```

まであります。

Buildとして十分です。

---

# ⑥ StereoConvolver

現在

```
retired
```

Atomicがあります。

二重Retire防止です。

これはRCU設計では重要です。

---

# ⑦ destroy

さらに

```
destroyStereoConvolver()
```

で

```
NUC

IR

aligned_free
```

まで一括です。

リーク耐性は高いです。

---

# 気になる点①

## applySpectrumFilter()

ここは少し気になります。

```
denom =
kEnd-kStart
```

です。

通常は

```
kEnd>kStart
```

ですが

極端なFFTサイズでは

丸め誤差で

```
kEnd==kStart
```

になる可能性があります。

すると

```
x = /0
```

になります。

---

### 改善

```
if(kEnd<=kStart)
```

を追加した方が安全です。

---

# 気になる点②

## MixedPhase例外

最後は

```
catch(...)
```

です。

状態は戻しています。

これは良いです。

しかし

原因は失われます。

---

例えば

```
MKL

OOM

CMAES

FFT
```

を分類できれば

保守性は上がります。

---

# 気になる点③

## sanitizeFiniteChunk()

現在

```
threshold=1e300
```

です。

これは

Inf

NaN

巨大値

を除去します。

---

しかし

```
1e250
```

などは

通ります。

FFT途中では

十分危険です。

---

改善候補

FFT前だけ

```
1e100
```

程度へ

下げることも検討できます。

ただし、しきい値を下げると**大きなダイナミックレンジを持つ正当な信号まで異常扱いする可能性**があります。変更する場合は、どの処理段階（FFT入力前・IFFT後など）に限定するかを含めて評価すべきです。

---

# 気になる点④

## applySpectrumFilter()

OOM時

```
continue;
```

です。

つまり

Layerだけ

処理されません。

---

安全ですが

左右Layerで

処理差

が出る可能性があります。

---

改善案

全部失敗

にした方が

音響的には

自然です。

---

# 気になる点⑤

## memcpy

Audio Threadでは

```
inputAccBuf
```

へ

毎回

```
memcpy
```

しています。

これは

設計上必要ですが

CPU負荷になります。

---

# 気になる点⑥

## warmupCompleted

Debugだけ

確認しています。

Releaseでは

ありません。

---

将来

Warmup失敗時

Releaseでも

検出できる

統計は

あると便利です。

---

# 気になる点⑦

## MixedPhase Cache

Memory Cache

Persistent Cache

があります。

しかし

途中生成物

```
Frequency Response

FFT

```

は

保存しません。

---

これは

再最適化時

少し損です。

---

# 私が今回最も気になった点

今回の最新版では、**以前のような「致命的なバグ」はほぼ見当たりません**。

一方で、唯一コード上から直接懸念できるのは **`applySpectrumFilter()` の除算**です。

```cpp
denom = kEnd - kStart;
x = (k - kStart) / denom;
```

という計算は、通常のFFTサイズでは問題ありませんが、**非常に小さいFFTサイズや境界条件では `denom == 0` になる可能性**があります。

実運用では発生しないケースかもしれませんが、このような境界条件は

```cpp
if (kEnd <= kStart)
{
    // ゲインを1.0のまま維持する等
}
```

のように明示的に防御しておく方が、安全性・保守性の両面で望ましいと考えます。

---

# 現時点での重大度ランキング

| 項目                               |   重大度 |
| -------------------------------- | ----: |
| `applySpectrumFilter()` のゼロ除算防御  | ★★★★★ |
| OOM時のLayer単独スキップ動作               | ★★★★☆ |
| 例外原因の分類                          | ★★★☆☆ |
| `sanitizeFiniteChunk()` のしきい値再評価 | ★★★☆☆ |
| Warmup異常のRelease診断               | ★★★☆☆ |

## 現時点の総括

ここまで **7回** にわたって解析した範囲では、以前レビューした重大な不具合（リングバッファ更新、RT安全性、FFT初期化など）は概ね改善されています。

今後さらに品質を高めるのであれば、**アルゴリズムの変更よりも、境界条件の防御・キャッシュ効率・CPUマイクロアーキテクチャ最適化・診断性の向上**に注力するのが効果的です。



## 第8回: Intel oneAPI・MKL・IPPの観点から見た「根本改善点」

ここまでの解析を踏まえ、今回は **Intel oneAPI / MKL / IPP のベストプラクティス** と比較した総合レビューを行います。

結論から言うと、

> **ConvoPeq の現在のコンボルバーは「アルゴリズム」よりも「実装レベルの最適化」が改善余地です。**

DSPアルゴリズム自体は十分成熟しています。

---

# 総合評価

| 項目           |    評価 |
| ------------ | ----: |
| アルゴリズム設計     | ★★★★★ |
| MKL利用方法      | ★★★★★ |
| IPP利用方法      | ★★★★☆ |
| Intel CPU最適化 | ★★★★☆ |
| oneAPI活用度    | ★★★★☆ |
| 将来性          | ★★★★★ |

---

# ① MKLの使い方

かなり良いです。

現在

* MKL
* IPP
* icx

を考慮したCMakeになっています。さらに `MKL_THREADING=sequential` と `IPP_THREADING=sequential` を採用しており、リアルタイムDSPに適した構成です。

これはAudio Threadでは理想です。

---

# ② IPP利用

IPPは

現在

主に

```text
FFT

Memory

Vector
```

で利用されています。

---

しかし

IPPには

まだ

* Complex Vector
* Magnitude
* Dot Product
* Window
* Convert

など

高速APIがあります。

---

全部置き換える必要はありません。

ただし

比較ベンチは

行う価値があります。

---

# ③ PGO

これは高評価です。

CMakeを見ると

```text
GENPROFILE

↓

USEPROFILE
```

両方あります。

プロ用DSPでは

PGO効果は非常に大きいです。

---

# ④ icx対応

Intel LLVM

まで考慮されています。

これは

珍しいです。

---

# ⑤ R8Brain

IPP FFTを使わない理由も

コメントされています。

非常に保守性が高いです。

---

# 私が一番評価した点

## Threading

現在

```text
MKL_THREADING=sequential

IPP_THREADING=sequential
```

です。

これは

Audio DSPでは

正しいです。

---

OpenMP

TBB

を使うべきではありません。

---

# 一番気になる点①

## Arena Allocation

Layer毎に

かなりの

```text
malloc

free
```

があります。

これは以前から指摘しています。

---

Intel資料では

Arena

Pool

が推奨です。

---

IR切替頻度が高い環境では

効果があります。

---

# 一番気になる点②

## NUMA未考慮

普通は不要です。

しかし

Xeon

Dual Socket

では

Layer毎

NUMA固定

も可能です。

---

一般ユーザー向けなら

不要です。

---

# 一番気になる点③

## PGO

設定はあります。

しかし

重要なのは

**プロファイル取得方法**です。

---

例えば

IR

256

↓

4096

↓

65536

全部

学習しないと

意味がありません。

---

# 一番気になる点④

## VTune

現状

まだ

VTune前提

設計ではありません。

---

私は

最低でも

以下は

取得します。

* Top Down Analysis
* Memory Access
* False Sharing
* Hotspot
* Vectorization

---

# 一番気になる点⑤

## AVX512

64Byte Alignment

なので

対応できます。

---

しかし

AVX512コードは

ありません。

---

今後

Arrow Lake

Xeon

では

価値があります。

---

# 一番気になる点⑥

## huge page

現在

通常mallocです。

---

FFT

FDL

IR

巨大です。

---

Windowsなら

Large Page

も検討できます。

ただし

権限

互換性

が悪く

一般用途では

優先度は低いです。

---

# Intelが勧める最適化順位

Intelのチューニング資料と、今回解析したConvoPeqの構造を照らし合わせると、優先順位は次のようになります。

| 順位 | 改善項目                    | 期待効果  |
| -: | ----------------------- | ----- |
|  1 | VTuneで実測しHotspotを特定     | ★★★★★ |
|  2 | Arena / Pool Allocation | ★★★★★ |
|  3 | AoS⇔SoA変換コスト削減          | ★★★★★ |
|  4 | PGOプロファイル最適化            | ★★★★☆ |
|  5 | False Sharing対策         | ★★★★☆ |
|  6 | AVX512対応（対応CPUのみ）       | ★★★☆☆ |
|  7 | Large Page利用            | ★★☆☆☆ |

---

# 今回の結論

Intel oneAPIの観点から見ると、

**現在のConvoPeqは「設計」はかなり完成しています。**

その一方で、**Intel CPUの性能をさらに引き出すには、ライブラリの追加導入よりも「計測ベースの最適化」が重要**です。

特に、

* VTuneでのHotspot解析
* メモリ帯域・キャッシュミス解析
* PGOプロファイルの品質向上

は、ソースコードを大きく変更せずに性能改善できる可能性があります。

---

## ここまで8回の解析を通じた総合順位

**改善効果が最も大きいと考える項目を、総合的に順位付けすると次のようになります。**

| 順位 | 改善項目                                           |   重要度 |
| -: | ---------------------------------------------- | ----: |
|  1 | VTuneで実測し、真のHotspotを特定する                       | ★★★★★ |
|  2 | LayerのArena/Pool Allocation化                   | ★★★★★ |
|  3 | AoS⇔SoA変換の必要性を実測で再評価する                         | ★★★★★ |
|  4 | `sanitizeFiniteChunk()` やCrossfadeなどRT処理のSIMD化 | ★★★★☆ |
|  5 | false sharing対策（Atomic配置の最適化）                  | ★★★★☆ |
|  6 | Mixed Phase FFTサイズの適応化                         | ★★★★☆ |
|  7 | IPP複素演算APIとのベンチマーク比較                           | ★★★☆☆ |
|  8 | AVX-512専用経路の追加                                 | ★★★☆☆ |

### 総括

ここまでのレビューでは、**リアルタイム安全性やDSPアルゴリズムに関する重大な問題は見当たりませんでした**。むしろ現在の実装は、`Practical Stable ISR Bridge Runtime` の「RTスレッドは実行のみ」「Publish後はImmutable」「atomicはWrapper経由」といった設計原則とも整合しています。

今後の改善は、**「正しさ」を変えるのではなく、「同じ処理をより少ないCPU時間・メモリ帯域で実行する」ための実測ベースの最適化**が中心になると考えます。



## 第9回: 「もし私が ConvoPeq のコンボルバーをゼロから設計するなら」

ここまで最新版ソースを解析した結果を踏まえ、「Intel CPU専用・Windows 11 x64・JUCE・ISR Runtime」という条件であれば、私ならどのような設計を採用するかをまとめます。

結論から言うと、

> **アルゴリズムは現状をほぼ維持します。**
>
> **変更するのはメモリ構造とデータフローです。**

これは非常に重要な違いです。

---

# 総合評価

現在の実装は

* Non Uniform Partition
* Overlap Save
* Direct Head
* RCU Publish
* Loader Thread

という基本設計は非常に優秀です。

この部分は変更しません。

---

# 私なら変更するもの

## 第1位

## Layerを全面的に作り直す

現在Layerは

```text
Layer

↓

FFT

FDL

IR

Tail

Overlap

Accum

State

WorkBuffer
```

となっています。

かなり巨大です。

---

私なら

```text
LayerConfig
    partSize
    fftSize
    numParts

LayerBuffers
    FDL
    FFT
    Tail

LayerRuntime
    writeIndex
    overlapIndex

LayerFFT
    FFTSpec
```

まで分割します。

---

理由

CPU Cacheです。

Layer一個読むだけで

大量Cache Missが起きます。

---

# 第2位

Arena Allocation

これは最重要です。

現在

IRロード時

Layer毎に

かなり多くの

```cpp
mkl_malloc()
```

があります。これは以前の解析でも確認したとおりです。

---

私なら

Layer毎ではなく

```text
Arena

↓

Pointerだけ分配
```

にします。

例えば

```text
Arena

----------------------------------

IR

FDL

FFT

Overlap

Tail

Accum

----------------------------------
```

---

こうすると

freeは

1回

です。

---

# 第3位

AoS完全廃止

これは今回一番言いたい点です。

最新版を見る限り

現在

IR

FDL

とも

AoS

SoA

両方保持しています。

以前提出されたメモリ解析でも、この二重保持が大きなメモリ消費要因になっていることが整理されています。

---

つまり

```
AoS

↓

Split

↓

SoA保存
```

しています。

---

私なら

最初から

```
SoA生成
```

します。

AoSは一瞬だけ。

保持しません。

---

これは

192kHz

巨大IR

で非常に効きます。

---

# 第4位

FDL Layout変更

現在

```
FDL

Part0

Part1

Part2
```

です。

---

私なら

```
Real

Part0

Part1

...

Imag

Part0

Part1
```

にします。

つまり

完全SoA

です。

---

理由

AVX2

AVX512

が使いやすい。

---

# 第5位

FFT Wrapper

現在

Layerが

FFTを知っています。

---

私は

```text
Layer

↓

FFT Interface

↓

IPP

↓

MKL

```

にします。

すると

将来

FFTW

PocketFFT

へ

差し替えできます。

---

# 第6位

SIMD Layer

現在

AVXコードは

DSPへ混ざっています。

例えば Direct Head や `processLayerBlock()` には AVX2/FMA が直接書かれています。

---

私なら

```
Convolver

↓

SIMDKernel

↓

AVX2

AVX512

Scalar
```

へ分離します。

---

保守性がかなり上がります。

---

# 第7位

Thread Local Scratch

現在

Layer毎

WorkBufferです。

---

私なら

Loader Thread

Audio Thread

毎に

Scratch Arena

を持ちます。

---

malloc

完全ゼロになります。

---

# 第8位

Publication

ここは

変更しません。

ISR Runtime

RCU

は非常に綺麗です。現在の Runtime/Publication 系は、責務分離も進んでいます。

---

# 第9位

Intel IPP

私は

FFTだけ

IPP

でも構いません。

理由

自前AVXが

かなり速いからです。

IPPの複素演算へ全面移行する前に、必ずベンチマークします。

---

# 第10位

VTune First

ここが重要です。

私は

コードを書く前に

VTuneを回します。

Intel CPU最適化は

**推測ではなく計測**

です。

---

# 私が絶対変更しないもの

以下は現状維持です。

✅ Direct Head

✅ Non Uniform Partition

✅ Overlap Save

✅ Placeholder Runtime

✅ Loader Thread

✅ Runtime Publish

✅ Build Snapshot

これらは十分に成熟しています。

---

# 私なら絶対採用するもの

| 機能                   | 採用    |
| -------------------- | ----- |
| Arena Allocation     | ★★★★★ |
| 完全SoA                | ★★★★★ |
| SIMD Layer分離         | ★★★★★ |
| FFT Wrapper          | ★★★★★ |
| Thread Scratch Arena | ★★★★★ |
| VTune First          | ★★★★★ |

---

# ここまで9回解析した総合結論

約9回にわたりコンボルバーを解析した結果、現在の ConvoPeq は **DSPアルゴリズム・リアルタイム安全性・ISR設計**については高い完成度に達しています。

一方で、**最も大きな改善余地はメモリ構造**にあります。

特に重要なのは次の3点です。

1. **AoSとSoAの二重保持の解消**（メモリ使用量・キャッシュ効率）
2. **LayerのArena Allocation化**（IR切替時の確保・解放コスト削減）
3. **Layer構造の責務分離**（キャッシュ局所性・保守性の向上）

これらはアルゴリズムを変更せずに、メモリ消費・CPUキャッシュ効率・保守性を同時に改善できる可能性があり、現状の実装で最も投資対効果の高い改善候補だと考えます。
