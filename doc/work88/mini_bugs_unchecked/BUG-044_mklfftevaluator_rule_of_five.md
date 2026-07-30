# BUG-044: MklFftEvaluator Rule of Five 違反 — IPP リソース+生ポインタ所有

## 重要度
中 (Medium) — `NoiseShaperLearner` が `MklFftEvaluator` をメンバとして直接保持しており、
暗黙のコピー/ムーブが発動すると IPP リソースの多重解放やリークを引き起こす。

## 対象ファイル
- `src/MklFftEvaluator.h` (全830行)

## 概要
`MklFftEvaluator` はコンストラクタで以下のリソースを確保し、デストラクタで解放するが、
コピー/ムーブ制御が一切宣言されていない:

- 4 つの `convo::makeAlignedArray<>().release()` → `convo::aligned_free` (生ポインタ)
  - `inputLeft`, `inputRight` (double*)
  - `spectrumLeft`, `spectrumRight` (CcsComplex*)
- 2 つの `ippsMalloc_8u` → `ippsFree` (IPP 管理メモリ)
  - `fftSpecBuf` (Ipp8u*)
  - `fftWorkBuf` (Ipp8u*)
- 間接的に `fftSpec` (IppsFFTSpec_R_64f*, `fftSpecBuf` 内を指す非所有ポインタ)

## 問題点
```cpp
class MklFftEvaluator
{
public:
    MklFftEvaluator()
    {
        inputLeft  = convo::makeAlignedArray<double>(kFftLength).release();
        inputRight = convo::makeAlignedArray<double>(kFftLength).release();
        spectrumLeft  = convo::makeAlignedArray<CcsComplex>(kSpectrumBins).release();
        spectrumRight = convo::makeAlignedArray<CcsComplex>(kSpectrumBins).release();
        // IPP FFT スペック初期化...
        fftSpecBuf = ippsMalloc_8u(sizeSpec);
        // ...
    }

    ~MklFftEvaluator()
    {
        if (fftSpecBuf) { ippsFree(fftSpecBuf); ... }
        if (fftWorkBuf) { ippsFree(fftWorkBuf); ... }
        if (inputLeft)  convo::aligned_free(inputLeft);
        // ...
    }
    // コピー/ムーブ制御なし！
private:
    double*     inputLeft     = nullptr;
    double*     inputRight    = nullptr;
    CcsComplex* spectrumLeft  = nullptr;
    CcsComplex* spectrumRight = nullptr;
    IppsFFTSpec_R_64f* fftSpec    = nullptr;
    Ipp8u*             fftSpecBuf = nullptr;
    Ipp8u*             fftWorkBuf = nullptr;
};
```

暗黙のコピーが発動すると:
1. 全ポインタがシャローコピーされる
2. コピー元とコピー先の両方のデストラクタで `ippsFree` + `aligned_free` が同一アドレスに対して呼ばれる
3. **二重解放 (double-free) → UB / ヒープ破壊 / IPP 内部状態破損**

## 修正案
```cpp
class MklFftEvaluator
{
public:
    MklFftEvaluator();
    ~MklFftEvaluator();

    MklFftEvaluator(const MklFftEvaluator&) = delete;
    MklFftEvaluator& operator=(const MklFftEvaluator&) = delete;
    MklFftEvaluator(MklFftEvaluator&&) = delete;
    MklFftEvaluator& operator=(MklFftEvaluator&&) = delete;

    // または JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MklFftEvaluator)
    ...
};
```

もしくは全リソースを RAII ラッパー (`ScopedAlignedPtr`, `IppScopedPtr` など) で包み、
デストラクタとコピー制御の手動管理を排除する。
