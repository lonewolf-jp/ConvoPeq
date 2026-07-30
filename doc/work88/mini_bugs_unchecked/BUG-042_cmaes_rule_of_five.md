# BUG-042: CmaEsOptimizer Rule of Five 違反 — 生ポインタ所有による二重解放リスク

## 重要度
中 (Medium) — 現在使用パターンではコピー/ムーブが発生しないが、将来的なリファクタリングで発症する可能性が高い。

## 対象ファイル
- `src/CmaEsOptimizer.h` (全244行)

## 概要
`CmaEsOptimizer` はコンストラクタで `convo::makeAlignedArray<double>(kDim).release()` により
2 つの生ポインタ (`double* mean`, `double* covariance`) を確保し、
デストラクタで `convo::aligned_free` により解放している。

しかし、コピーコンストラクタ、コピー代入演算子、ムーブコンストラクタ、ムーブ代入演算子の
いずれも宣言されておらず、`JUCE_DECLARE_NON_COPYABLE` マクロも使用されていない。

## 問題点
```cpp
class CmaEsOptimizer
{
public:
    CmaEsOptimizer()
    {
        mean = convo::makeAlignedArray<double>(kDim).release();
        covariance = convo::makeAlignedArray<double>(kDim * kDim).release();
        ...
    }

    ~CmaEsOptimizer()
    {
        convo::aligned_free(mean);
        convo::aligned_free(covariance);
    }
    // コピー/ムーブ制御が一切なし！
    ...
private:
    double* mean = nullptr;
    double* covariance = nullptr;
    ...
};
```

暗黙のコピーコンストラクタが発動すると:
1. `mean` / `covariance` のポインタ値がビット単位でコピーされる (シャローコピー)
2. コピー元とコピー先の両方がデストラクタで同じポインタを `aligned_free` する
3. **二重解放 (double-free) → UB / ヒープ破壊**

暗黙のムーブコンストラクタについても、C++11 以降はユーザー宣言デストラクタが
ムーブ生成を抑制するため自動生成されないが、削除もされていないため
コンパイルエラーとして検出されないコードパスが存在しうる。

## 修正案
```cpp
class CmaEsOptimizer
{
public:
    CmaEsOptimizer();
    ~CmaEsOptimizer();

    CmaEsOptimizer(const CmaEsOptimizer&) = delete;
    CmaEsOptimizer& operator=(const CmaEsOptimizer&) = delete;
    CmaEsOptimizer(CmaEsOptimizer&&) = delete;
    CmaEsOptimizer& operator=(CmaEsOptimizer&&) = delete;

    // または JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(CmaEsOptimizer)
    ...
};
```

もしくは `std::unique_ptr` に移行して RAII 化し、暗黙のムーブを安全に有効化する。
