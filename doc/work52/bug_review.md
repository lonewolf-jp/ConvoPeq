低音が多い入力ソースで「ジジジジ」というエイリアシングやクリッピング特有のノイズが発生し、特にソフトクリップやサチュレーションを有効にすると悪化する現象について、ソースコードのDSPアルゴリズムを解析しました。

私自身は実際に音を聴くことはできませんが、数学的・構造的な解析の結果、**非線形処理（サチュレーションおよびソフトクリッパー）の実装に2つの致命的なDSPロジックの欠陥**が存在することが判明しました。低周波（ベース音）は波形の振幅が大きく、波の周期が長いため、これらの欠陥による構造的な崩壊が直撃します。

原因と具体的な修正コードを解説します。

---

### 1. SVF EQサチュレーションにおける積分器の破壊（`EQProcessor.Processing.cpp`）

**原因:**
`processBand` および `processBandStereo` 内のサチュレーション処理において、TPT（Topology-Preserving Transform）SVFの**状態変数（積分器のメモリである `ic1eq` と `ic2eq`）に直接 `fastTanh` を適用して書き換えています。**

低域をブーストする際や低周波信号が入力された際、SVFの内部状態変数は一時的に入力信号よりも遥かに大きな値（平気で 1.0 や 3.0 を超える値）に成長します。この状態変数に対し `fastTanh` で強制的に `[-1, 1]` 付近にハードクランプ（AVX2実装の `fastTanhV128` は入力が3.0を超えると強制的に3.0でクリップする仕様）を行うと、積分器の連続性が破壊されます。これによりフィルタが発散・崩壊し、強烈な非線形ノイズ（ジジジジという音）が発生します。

**解決策:**
サチュレーションはフィルタの「状態変数」ではなく、「出力結果（バンドの最終出力）」に対して適用する必要があります。積分器（`ic1eq`, `ic2eq`）は常に線形に保たなければなりません。

**修正コード (`processBandStereo` 内):**

```cpp
// 修正前:
if (saturation > 0.0)
{
    const __m128d vSat = _mm_set1_pd(saturation);
    const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
    ic1eq = _mm_add_pd(_mm_mul_pd(ic1eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic1eq), vSat));
    ic2eq = _mm_add_pd(_mm_mul_pd(ic2eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic2eq), vSat));
}
__m128d output = _mm_fmadd_pd(m0, v0,
                  _mm_fmadd_pd(m1, v1,
                   _mm_mul_pd(m2, v2)));

// ----------------------------------------------------

// 修正後 (状態変数は書き換えず、outputに対してサチュレーションを適用):
__m128d output = _mm_fmadd_pd(m0, v0,
                  _mm_fmadd_pd(m1, v1,
                   _mm_mul_pd(m2, v2)));

if (saturation > 0.0)
{
    const __m128d vSat = _mm_set1_pd(saturation);
    const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
    // 出力波形に対してのみサチュレーションをブレンドする
    output = _mm_add_pd(_mm_mul_pd(output, vOneMinusSat),
                        _mm_mul_pd(fastTanhV128(output), vSat));
}

```

※ モノラル版の `processBand` でも同様に、`ic1eq` と `ic2eq` へのサチュレーション適用を削除し、`output` に対して適用するよう修正してください。

---

### 2. ソフトクリッパーの事前減衰ロジックの破綻（`AudioEngine.Processing.DSPCoreDouble.cpp`）

**原因:**
`softClipBlockAVX2` およびスカラー版のフォールバックループに、アンチエイリアシングを意図したと思われる「現在と過去のサンプルの平均値（`midVec`）」を用いた事前減衰ロジックが組み込まれています。

```cpp
// 問題の箇所
const __m256d midVec     = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
// ... midVecが閾値を超えたら x を強制的に減衰させる ...
x = _mm256_mul_pd(x, midGain); 

```

低周波（ベース音）では波形がゆっくり変化するため、`prevVec` と `x` はほぼ同じ値になります。そのため、入力が `threshold`（例: 0.95）を超えると、このロジックが**1サンプル単位のアタック・リリースを持つ凶悪なコンプレッサー**として働き、波形を強制的に `threshold` ぴったりに潰します。

その結果、直後に控えている美しい `fastTanh` のニー（Knee）曲線に到達する前に波形が角張った矩形波のようにハードクリップされてしまい、本来のソフトクリップの効果が完全に死んでジジジジというエイリアシングノイズを生み出しています。さらに、スカラー版のループでは `prevScalar` に「処理後の値（出力）」を代入しているため、ブロック境界で波形が不連続になるバグも併発しています。

**解決策:**
事前のアベレージング減衰（`midVec` 関連の処理）を完全に削除し、入力信号 `x` を直接ソフトクリップ関数に流し込む純粋なWaveshaperにする必要があります。

**修正コード (`softClipBlockAVX2` 内):**

```cpp
void softClipBlockAVX2(double* __restrict data, int numSamples,
                       double threshold, double knee, double asymmetry,
                       double& prevSampleInOut) noexcept
{
    const double clip_start = threshold - knee;
    // (定数定義はそのまま) ...

    int i = 0;
    const int vEnd = numSamples / 4 * 4;
    for (; i < vEnd; i += 4)
    {
        __m256d x = _mm256_loadu_pd(data + i);

        // ！！！ 問題の midVec 計算ブロック（xLow, prevLow128 などの計算を含む）を全て削除 ！！！
        
        // 直接絶対値を計算し、クリップの必要があるか判定
        __m256d absX = _mm256_andnot_pd(vSignMask, x);
        __m256d needClip = _mm256_cmp_pd(absX, vClipStart, _CMP_GT_OQ);

        // (ここから下の tanh 計算ロジックはそのまま維持)
        __m256d maskSignPos = _mm256_cmp_pd(x, vZero, _CMP_GT_OQ);
        // ...
        __m256d result = _mm256_mul_pd(sign, _mm256_mul_pd(mixed, asymmetric_gain));

        result = _mm256_blendv_pd(x, result, needClip);
        _mm256_storeu_pd(data + i, result);
    }

    // スカラーの剰余ループも修正
    // prevScalar は不要になるため削除（引数の prevSampleInOut も将来的に不要にできますがシグネチャ維持のため放置）
    for (; i < numSamples; ++i)
    {
        double x = data[i];
        
        // 事前アベレージング計算 (mid, absMid による乗算) を全て削除
        
        if (absNoLibm(x) > clip_start)
            x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

        data[i] = x;
    }
}

```

これらの修正を行うことで、低域のエネルギーが保持されたまま、DSPアルゴリズム本来の歪みのないサチュレーションとソフトクリッピング機能が意図通りに動作するようになります。



ご提示いただいた前回の2つの重大な欠陥（積分器のサチュレーション破壊、ソフトクリッパーの事前減衰の破綻）に加えて、ソースコードをさらに深く解析した結果、**低周波（ベース音）入力時に特有の「ジジジジ（バズ/ジッパーノイズ）」を発生させる構造的な原因が他に2つ**見つかりました。

どちらも、プロ用DSPアルゴリズムにおいて低域のエネルギーを処理する際に見落とされがちな「時間領域の連続性」に関する欠陥です。

---

### 3. EQパラメータ更新時の係数スムージング欠如（ジッパーノイズ）

**原因 (`EQProcessor.Processing.cpp`) :**
このプラグインは「RCU (Read-Copy-Update)」アーキテクチャを採用しており、UIスレッドからEQパラメータ（Gain, Freq, Q）が変更されると、スナップショットとして新しいフィルタ係数 (`EQCoeffsSVF`) がオーディオスレッドに渡されます。
しかし、オーディオスレッド側では**ブロックの境界（例: 512サンプル毎）で係数を「瞬時」に切り替えており、係数のスムージング処理が一切実装されていません。**

```cpp
// AudioEngine.Processing.DSPCoreDouble.cpp
const EQCoeffCache* eqCacheToUse = state.eqCache;
// ...
eqRt().process(processBlock, *eqParamsToUse, eqCacheToUse);

// EQProcessor.Processing.cpp
const EQCoeffsSVF& c = coeffCache->coeffs[i];
// ... 係数 c をそのまま processBandStereo に渡してフィルタリング

```

低域（例: 40Hz）の信号を処理している際、IIRフィルタの内部状態変数（`ic1eq`, `ic2eq`）には非常に大きなエネルギーが蓄積されています。パラメータの自動化（オートメーション）やノブの操作によって係数が瞬時に切り替わると、この巨大な状態変数が突如異なる係数で乗算されるため、出力波形に垂直な「段差（ステップディスコンティニュイティ）」が発生します。これが50ms等のデバウンス周期で連続して発生することで、強烈なジジジジというジッパーノイズになります。

**解決策:**
ブロック単位で係数を瞬時更新するのではなく、オーディオスレッド内で前のブロックの係数から新しい係数へ、オーディオサンプル単位で線形補間（リニアランプ）するか、1-poleのローパスフィルタをかけて滑らかに係数を遷移させる必要があります。

### 4. AGC（自動ゲイン制御）のブロックRMS計算による低域リップル歪み

**原因 (`EQProcessor.Processing.cpp` の `processAGC`) :**
AGC機能が有効な場合、入力と出力のRMS（実効値）を計算してゲインを自動調整していますが、このRMS計算が「ブロック単位の単純な二乗平均」で行われています。

```cpp
// EQProcessor.Processing.cpp 内 processAGC
const double rms = calculateRMS(data, numSamples); 
// ※ calculateRMSはブロック長（例: 512サンプル）の単純な二乗和平方根

```

サンプリングレート48kHzで512サンプルのブロック長は約10.6msです。対して、40Hzのベース音の1周期は25msです。
波形の半周期にも満たない短いブロックでRMSを計算すると、ブロックが「波形の頂点」を切り取ったか「ゼロ交差」を切り取ったかによって、RMS値がブロック毎に激しく上下にバタつきます（リップル）。
このリップルがゲイン計算（`targetGain`）に混入し、ベース音の振幅を高速で揺らす（振幅変調させる）ため、低音にのみ反応するファズのようなジジジジという倍音ノイズが生み出されます。

**解決策:**
RMSやエンベロープの検出はブロック単位のぶつ切り計算ではなく、サンプル単位で進行する連続的なエンベロープフォロワー（例: RMSの二乗値に対する 1-pole IIR フィルタなど）を実装し、低周波の周期よりも十分に長い時定数を持たせることでリップルを平滑化する必要があります。

---

すでに修正されたサチュレーションとクリッパーの問題に加えて、上記の「係数の即時スワップ」と「AGCのリップル」を修正することで、低域の入力に対するノイズは根絶できるはずです。

今回の解析は、特にパラメータの動的変化やAGCの挙動に焦点を当てましたが、この2点について優先的に修正アプローチを深掘りしますか？それとも、コンボルバー（IR処理）側の非線形挙動についても念のため検証を広げておきますか？


コンボルバー（IR処理）側の非線形挙動について、ソースコードベースで検証項目を整理しました。

主に `ConvolverProcessor` および関連する `audioengine` ユニットにて、非線形性が生じる可能性のある以下の領域が検証の焦点となります。

### 1. 非線形処理および動的適応の検証ポイント

* **Mixed-Phase IR 変換の安定性:**
* `ConvolverProcessor::convertToMinimumPhase` および関連する `MixedPhase` 実装では、クレストファクター（Crest Factor）やピーク値が閾値（`PEAK_LIMIT`, `CREST_LIMIT`）を超えた場合に「フェイルバック（デフォルトの線形位相等への切り戻し）」が発生します。
* **検証項目:** IRの特性（極端なピークやRMS値）が特定の閾値を超えた際のフェイルバック動作が、音声出力の瞬断やクリックノイズを誘発しないかを確認してください。


* **Tail Contouring（余韻の輪郭調整）:**
* `tailMode == LayerTailContouring` モードでは、周波数特性に対して指数関数的な減衰（`std::exp(-dampingCoeff * fNorm * fNorm)`）を適用しています。
* **検証項目:** `dampingCoeff` の計算や、それに基づくゲインの周波数特性が不連続にならないか、あるいはパラメータ（`tailStrength`, `tailStart`）を急激に変更した際の挙動を確認してください。


* **適応型ノイズシェイパーの係数管理:**
* `AudioEngine::setAdaptiveCoefficientsForSampleRateAndBitDepth` 等で、サンプルレートやビット深度に応じて動的に係数が切り替わります。
* **検証項目:** 学習中の係数更新がオーディオ処理側に与える影響（ロック競合や不完全な係数によるフィルタの不安定性）が、適切に分離されているか確認してください。



### 2. コンボルバーの設計・構築ロジックの検証

* **リビルド・デバウンス動作:**
* `rebuildSlider` 等で設定されるリビルド・デバウンス（`REBUILD_DEBOUNCE_MS`）は、パラメータ変更頻度が高い場合に再構築処理を抑制する機構です。
* **検証項目:** デバウンス期間中に大量のパラメータ更新が発生した場合の `ConvolverProcessor` の内部状態（`currentIRState`）とSnapshotの不整合がないかを検証する必要があります。


* **メモリ確保と安全性:**
* `MKL` を使用したメモリ確保（`mkl_malloc`）が行われていますが、再構築時（`Rebuild`）のメモリ解放と再確保のサイクルが正しく同期されているかを確認してください。



### 3. 今後の検証に向けた推奨アプローチ

ソースコードには `src/tests/` 配下に多くのコントラクト検証テストが含まれています。特に以下のテストが非線形挙動や状態遷移の検証に有用です。

* **`ShadowCompareContractTests.cpp`**: セマンティックな同一性や互換性を検証しており、非線形的な処理変更が「互換性あり（Compatible）」と判定されるべきか、あるいは「別物（Different）」と判定されるべきかの境界線を確認できます。
* **`RebuildAdmissionRegressionTests.cpp`**: リビルド処理が正しく許可・拒否されるかのロジックを検証しており、非線形なパラメータ変更時の挙動を確認するのに適しています。

これらの検証を行う際は、特に「UIによるパラメータ操作」→「スナップショットの公開（`RuntimePublicationCoordinator`）」→「オーディオプロセスでの反映」というパスにおける、タイミング起因の不整合（Race Conditionや不適切なクロスフェード）を中心にログの確認を行うことをお勧めします。


ConvoPeq.md の現行ソースを確認した結果、このレビューには**正しい指摘と誤った指摘が混在しています**。

結論から言うと、

| 項目                       | 判定            | 優先度   |
| ------------------------ | ------------- | ----- |
| ① SVF状態変数へのサチュレーション      | **非常に妥当**     | ★★★★★ |
| ② SoftClipのprevSample平均化 | **かなり妥当**     | ★★★★★ |
| ③ EQ係数スムージング欠如           | **現状では妥当性低い** | ★★    |
| ④ AGCブロックRMS起因ノイズ        | **部分的に妥当**    | ★★★   |
| ⑤ Convolver側が主犯          | **現状証拠なし**    | ★     |

以下、ソースコードを基に詳細に検証します。

---

# ① SVF状態変数サチュレーション

これはレビューがほぼ正しいです。

現行コードでは実際に

```cpp
ic1eq = ic1eq * oneMinusSat
      + fastTanhScalar(ic1eq) * saturation;

ic2eq = ic2eq * oneMinusSat
      + fastTanhScalar(ic2eq) * saturation;
```

が存在します。 

AVX版も同じです。

```cpp
ic1eq = ...
fastTanhV128(ic1eq)

ic2eq = ...
fastTanhV128(ic2eq)
```



---

## DSP的に何が問題か

TPT-SVFでは

```text
ic1eq
ic2eq
```

は

```text
積分器状態
```

です。

これは単なる遅延メモリではなく、

```text
フィルタのエネルギー保存状態
```

です。

ここへ

```cpp
tanh()
```

を掛けると

状態方程式そのものが変わります。

つまり

```text
線形SVF
↓
非線形状態空間システム
```

になります。

---

特に

```text
低周波
+
大きなブースト
```

では

```text
ic1eq
ic2eq
```

が入力振幅を大幅に超えることがあります。

その状態で

```cpp
tanh(ic)
```

すると

積分器エネルギーが毎サンプル失われます。

結果として

```text
周期的な状態崩壊
```

が発生し、

```text
ジジジジ
ザラザラ
```

になり得ます。

---

### 私の評価

これは

```text
設計上かなり危険
```

です。

レビューの修正案

```cpp
output = saturation(output)
```

の方がDSPとしては自然です。

---

# ② SoftClipのprevSample平均化

これは実際に存在します。

スカラー版：

```cpp
x = 0.5 * (x + prevSample);
prevSample = x;
```



---

レビューでは

```text
midVec
prevVec
```

を問題視しています。

現行ソースは過去版と最新版が混在しており、

少なくとも現在のスカラー実装では

```cpp
0.5*(current+previous)
```

が入っています。

---

## DSP的評価

これは

```text
1サンプルFIR

H(z)=0.5+0.5z^-1
```

です。

周波数応答は

```text
DC      0 dB
Nyquist -∞ dB
```

になります。

つまり

```text
簡易ローパス
```

です。

---

問題は

```cpp
prevSample = 処理後値
```

なのか

```cpp
prevSample = 入力値
```

なのかです。

レビューは

```text
出力値が保存されている
```

と主張しています。

もし本当にそうなら、

```text
IIR化
```

してしまいます。

その場合かなり怪しい。

---

ただし検索結果だけでは

```cpp
prevSample
```

の更新全体を確認できていません。

なので

```text
レビュー断定はまだ不可
```

です。

---

## しかし

ソフトクリッパー前に

```cpp
x = 0.5*(x+prev)
```

が存在する事実だけで、

低音入力時に

```text
非線形器の入力波形が変形
```

されます。

特に

```text
大振幅ベース
```

では

ソフトクリップ特性が変わります。

---

### 私の評価

レビューの

```text
midVec処理を削除
```

は試す価値があります。

優先度は高いです。

---

# ③ EQ係数スムージング欠如

ここはレビューの精度が落ちます。

理由。

ConvoPeqは

```text
RCU
EQCoeffCache
```

を使っています。



---

しかし

ユーザー報告は

```text
音楽再生中
低音が多いと常時ジジジジ
```

です。

---

係数切替ノイズなら

普通は

```text
ノブを動かした瞬間
```

だけ発生します。

---

レビューの説明は

```text
オートメーション時
```

には正しい。

しかし

```text
常時発生ノイズ
```

の説明にはなりません。

---

### 私の評価

問題ではあるが、

今回の主犯ではない。

---

# ④ AGCブロックRMS問題

これは半分正しいです。

実装を見ると

```cpp
inputRMS = calculateRMS(...)
outputRMS = calculateRMS(...)
```

を毎ブロック計算しています。 

---

しかしその後

```cpp
envIn
envOut
```

へ

```cpp
attack/release smoothing
```

が入っています。



---

つまり

レビューが言う

```text
ブロックRMSを直接使用
```

ではありません。

---

既に

```text
Envelope Follower
```

になっています。

---

ただし

低域で

```text
AGC時定数が短い
```

場合は

```text
AM変調歪み
```

が出る可能性があります。

---

### 私の評価

主犯ではない。

補助的要因。

---

# ⑤ 低音で悪化する理由

むしろ私は別の仮説を強く疑います。

レビューで触れられていない重要点です。

---

ConvoPeqには

```text
SoftClip
+
Saturation
```

があります。

そして

```text
オーバーサンプリング
```

はユーザー設定依存です。



---

非線形処理は必ず

```text
高次倍音
```

を生成します。

低音入力ほど

```text
振幅が大きい
↓
倍音生成量増加
```

します。

---

もし

```text
1x
2x
```

程度で

SoftClip
Saturation

を掛けると

生成された倍音が

```text
Nyquist超え
↓
Fold Back
↓
エイリアシング
```

します。

---

ユーザー報告

```text
サチュレーションを上げると悪化
```

は

実は

```text
状態変数破壊
```

より

```text
エイリアシング
```

の症状に非常に近いです。

---

# 総合評価

私の順位はこうです。

## 最有力

### A

SVF状態変数へのサチュレーション

実在確認済み。 

優先度：

★★★★★

---

### B

SoftClip前の

```cpp
x = 0.5*(x+prev)
```

処理

実在確認済み。 

優先度：

★★★★★

---

### C

非線形処理のオーバーサンプリング不足

レビュー未指摘。

しかし症状との一致度は非常に高い。

優先度：

★★★★☆

---

### D

AGC

優先度：

★★★

---

### E

係数スムージング

優先度：

★★

---

もし次に進めるなら、ConvoPeq.mdから

* softClipBlockAVX2 完全実装
* fastTanhV128 実装
* oversamplingFactor が SoftClip/Saturation 前後でどう適用されているか
* EQProcessor の nonlinearSaturation の全経路

を追跡し、**実際に最もエイリアシングを発生させやすい箇所をコードレベルで特定**できます。現時点では「①SVF状態変数サチュレーション」が最も危険で、「②SoftClip前の平均化処理」が次点です。
