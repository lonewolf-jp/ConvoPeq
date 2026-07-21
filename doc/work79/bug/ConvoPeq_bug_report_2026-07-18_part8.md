# ConvoPeq ソースコード調査報告書（Part 8）

`EQProcessor.Coefficients.cpp`の係数計算式そのものを、参照文献と照合して数式検証しました。**今回は新規バグなし**（良好な結果）です。

---

## 1. 検証内容

### 1.1 Biquad係数（`calcLowShelfBiquad`/`calcPeakingBiquad`/`calcHighShelfBiquad`/`calcLowPassBiquad`/`calcHighPassBiquad`）

Audio EQ Cookbook（Robert Bristow-Johnson）の標準式と、`b0`/`b1`/`b2`/`a0`/`a1`/`a2`全項を1項ずつ突き合わせました。**5種類全てのフィルタタイプで、符号・係数とも完全に一致**することを確認しました。

### 1.2 SVF係数（`calcLowShelfSVF`/`calcPeakingSVF`/`calcHighShelfSVF`/`calcLowPassSVF`/`calcHighPassSVF`）

Cytomic（Andrew Simper）のTPT SVF方程式（`a1=1/(1+g(g+k))`, `a2=g*a1`, `a3=g*a2`という共通形、および各フィルタタイプ固有の`m0`/`m1`/`m2`）と突き合わせました。**5種類全てで一致**を確認しました。

唯一、`calcPeakingSVF`の`c.m1 = (A - 1.0/A) / q;`は、Cytomic式の標準形`m1 = k*(A²-1)`（`k=1/(QA)`）とは異なる書き方でしたが、実際に代数的に展開すると

```
k*(A²-1) = (A²-1)/(QA) = A/Q - 1/(QA) = (A - 1/A)/Q
```

となり、**完全に数学的に同一の式**であることを検算で確認しました（単に既に計算済みの`k`を再利用せず、別の等価な形で書き直しているだけで、バグではありません）。

### 1.3 周波数応答計算（`getMagnitudeSquared`）

`num = b0*z² + b1*z + b2`, `den = a0*z² + a1*z + a2`という正べき乗の形で実装されていますが、標準的な負べき乗形`H(z)=(b0+b1*z⁻¹+b2*z⁻²)/(a0+a1*z⁻¹+a2*z⁻²)`の分子分母に`z²`を掛けただけの等価な代数変形であり、`z≠0`（単位円上の周波数応答評価では常に成立）で数学的に同一の値を与えることを確認しました。

---

## 2. 結論

**係数計算式そのものに関するバグは見つかりませんでした。** RBJ CookbookとCytomic SVF方程式という、プロジェクト自身が参照文献に指定している2つのソースに対して、実装が数式レベルで完全に一致していることを確認できました。これは調査範囲マップで長らく「未検証」としていた項目の解消です。

なお、これらの関数自体（`calcXxxSVF`/`calcXxxBiquad`）はいずれも`noexcept`かつ`std::pow`/`std::tan`/`std::sqrt`等のlibm呼び出しを含みますが、呼び出し元（`createBandNode`/`updateBandNode`）のコメントで明記されている通りMessage Thread専用であり、Audio Thread規約への抵触はありません。

---

## 3. 調査範囲の更新

- `EQProcessor.Coefficients.cpp`（686行）: **精読完了**。これでEQProcessor関連5ファイル（Coefficients/Core/Parameters/Processing/ProcessingCache）全てが精読完了となりました。

---

## 4. 次のステップ（提案）

1. `ISRRuntimePublicationCoordinator.cpp`の残り（precheckPublish, PriorityScheduler, ShutdownScheduler等）
2. `ISRRetireRuntimeEx.cpp`（444行）
3. `DSPCoreFloat.cpp`/`DSPCoreIO.cpp`/`DSPCoreToBuffer.cpp`
4. `ConvolverProcessor.*`一式
