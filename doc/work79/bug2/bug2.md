# ConvoPeq 追加バグ詳細解析報告

前回の報告に加え、コードベース全体をコーディング規約・ISR Bridge Runtime設計指針・数値安定性の観点から詳細に解析した結果、以下の追加バグを特定しました。

---

## Bug A: ノイズシェイパーの NaN/Inf 伝播 【重大】

**対象:** `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h`

**問題:**
`killDenormal()` は Release ビルドで **no-op** であり、NaN/Inf を一切処理しない。

```cpp
// DspNumericPolicy.h
inline double killDenormal(double x) noexcept
{
#if !defined(JUCE_DEBUG) && !defined(_DEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
    static_cast<void>(x);
    return x;  // ← Release では何もしない
#else
    // ... denormal のみ処理。NaN/Inf は処理しない
#endif
}
```

フィードバックループが不安定化した場合、`fb` が NaN/Inf になり、以下のように伝播する：

```cpp
// Fixed15TapNoiseShaper.h - processSample
double fb = 0.0;
for (int i = 0; i < ORDER; ++i)
    fb += coeffs[i] * get(channelErrors, idx, i);
fb = killDenormal(fb);  // ← Release では NaN/Inf を通過させる
const double y = x - fb;  // ← NaN/Inf が伝播
const double yq = quantize(y, ...);  // ← NaN が quantize を通過
```

**影響:** フィルタ発散時に NaN/Inf が出力に伝播し、**音声出力が完全に破綻**する。

**修正提案:**
```cpp
// killDenormal を NaN/Inf 対応に強化
inline double killDenormal(double x) noexcept
{
    const auto bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits >> 52) & 0x7FFu) == 0u && (bits & 0x000FFFFFFFFFFFFFULL) != 0u;
    const bool isNanOrInf = ((bits >> 52) & 0x7FFu) == 0x7FFu;
    return (isSubnormal || isNanOrInf) ? 0.0 : x;
}
```

---

## Bug B: `quantize()` の NaN 通過 【重大】

**対象:** `FixedNoiseShaper.h`, `Fixed15TapNoiseShaper.h`, `LatticeNoiseShaper.h` の `quantize()`

**問題:**
NaN は比較演算が常に `false` を返すため、clamp を通過してしまう：

```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    const double minV = -1.0;
    const double maxV = 1.0 - (1.0 / invScale);
    if (v < minV)       // ← NaN < minV は false
        v = minV;
    else if (v > maxV)  // ← NaN > maxV も false
        v = maxV;
    // v は NaN のまま通過
    v += (u1 + u2 - 1.0) * scale;  // ← NaN + anything = NaN
    __m128d d = _mm_set_sd(v * invScale);  // ← NaN * anything = NaN
    d = _mm_round_sd(d, d, ...);  // ← round(NaN) = NaN
    return q * scale;  // ← NaN が出力に伝播
}
```

**影響:** Bug A と連鎖し、NaN が量子化器を通過して出力に伝播する。

**修正提案:**
```cpp
inline double quantize(double v, Xoshiro256State& rng) const noexcept
{
    // NaN/Inf ガードを最前面に追加
    if (!std::isfinite(v)) v = 0.0;
    // ... 既存の clamp 処理
}
```

---

## Bug C: `AudioSegmentBuffer::pushBlock` の境界チェック欠如 【重大】

**対象:** `AudioSegmentBuffer.h`

**問題:**
`numSamples > kCapacity` の場合、リングバッファのラップアラウンド計算が破綻し、**バッファオーバーフロー**が発生する：

```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    // numSamples > kCapacity のチェックがない
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire);
    int first = std::min(numSamples, kCapacity - currentWritePos);
    // ...
    if (first < numSamples)
    {
        int second = numSamples - first;
        // second > kCapacity の場合、バッファオーバーフロー
        juce::FloatVectorOperations::copy(leftSamples, left + first, second);
        // ...
    }
}
```

`kCapacity = 5 * 768000 = 3,840,000` であり、通常のブロックサイズでは問題ないが、**呼び出し元が `numSamples > kCapacity` を渡した場合にバッファオーバーフロー**が発生する。

**修正提案:**
```cpp
void pushBlock(const double* left, const double* right, int numSamples) noexcept
{
    if (left == nullptr || right == nullptr || numSamples <= 0)
        return;
    numSamples = std::min(numSamples, kCapacity);  // ← 追加
    // ...
}
```

---

## Bug D: `Fixed15TapNoiseShaper::processSample` の `fb` NaN/Inf 未チェック 【中】

**対象:** `Fixed15TapNoiseShaper.h`

**問題:**
フィードバック値 `fb` は `killDenormal()` のみで処理され、NaN/Inf がチェックされない：

```cpp
double fb = 0.0;
for (int i = 0; i < ORDER; ++i)
    fb += coeffs[i] * get(channelErrors, idx, i);
fb = killDenormal(fb);  // ← NaN/Inf は通過（Release では no-op）
const double y = x - fb;  // ← NaN/Inf が伝播
```

`FixedNoiseShaper.h` の `processSample` も同様の問題がある。

**影響:** Bug A と同一の伝播経路。15次フィルタは4次より発散リスクが高い。

---

## Bug E: `AudioSegmentBuffer::copyLatest` の TOCTOU 【中】

**対象:** `AudioSegmentBuffer.h`

**問題:**
`totalSamples` と `writePosition` を別々に読み取るため、読み取り間に書き込みが発生すると不整合なデータを読み取る可能性がある：

```cpp
int copyLatest(double* outLeft, double* outRight, int requestedSamples) const noexcept
{
    const int currentTotal = convo::consumeAtomic(totalSamples, std::memory_order_acquire);   // ①
    const int currentWritePos = convo::consumeAtomic(writePosition, std::memory_order_acquire); // ②
    // ①と②の間に pushBlock が実行されると、
    // currentTotal は古く、currentWritePos は新しい値になる
    const int availableSamples = std::min(requestedSamples,
        currentTotal >= kCapacity ? kCapacity : currentTotal);
    const int start = (currentWritePos - availableSamples + kCapacity) % kCapacity;
    // start が不正確になる可能性
}
```

**影響:** 実用上は `totalSamples` が古い値（小さい値）になるため、読み取り量が少なくなるだけで安全側に倒れる。ただし、`currentTotal >= kCapacity` の場合に `start = currentWritePos` となり、**書き込み中のデータを読み取る可能性**がある。

---

## Bug F: `StereoConvolver::init` の空ブロック 【低】

**対象:** `ConvolverProcessor.h`

**問題:**
`ownerProcessor` が非nullの場合の処理ブロックが**空**である：

```cpp
if (ownerProcessor != nullptr)
{
    // ← 空。何らかの処理が意図されていた可能性がある
}
```

**影響:** 意図された処理（例: `ownerProcessor` へのレイテンシ通知）が欠落している可能性。

---

## Bug G: `processBypassWithLatencyCompensation` の冗長な負値チェック 【低】

**対象:** `ConvolverProcessor.h`

**問題:**
C++20 では二の補数が義務付けられているため、`(writePos - delaySamples) & DELAY_BUFFER_MASK` は常に非負になる。`if (readPos < 0)` チェックは冗長：

```cpp
int readPos = (writePos - delaySamples) & DELAY_BUFFER_MASK;
if (readPos < 0)       // ← C++20 では常に false
    readPos += DELAY_BUFFER_SIZE;
```

**影響:** 機能的な問題はないが、C++20 以前のコンパイラでコンパイルした場合に問題になる可能性。

---

## Bug H: `StereoConvolver::init` の例外安全性 【低】

**対象:** `ConvolverProcessor.h`

**問題:**
`try-catch` が `std::bad_alloc` のみキャッチする。`SetImpulse` が他の例外を投げた場合、`irData` が解放されない：

```cpp
try
{
    auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    if (nuc0->SetImpulse(...) && nuc1->SetImpulse(...))
    {
        // ...
        return true;
    }
}
catch (const std::bad_alloc&)  // ← std::bad_alloc のみキャッチ
{
    // Fall through to cleanup
}
// cleanup: irData を解放
```

**影響:** `SetImpulse` が `std::bad_alloc` 以外の例外を投げた場合、`irData` がリークする。実用上は発生しにくいが、規約上「メモリリークが起きないようメモリ開放に注意」に抵触。

---

## 重要度まとめ

| # | バグ | 重要度 | カテゴリ |
|---|------|--------|----------|
| A | ノイズシェイパー NaN/Inf 伝播 | **重大** | 数値安定性 |
| B | `quantize()` NaN 通過 | **重大** | 数値安定性 |
| C | `pushBlock` 境界チェック欠如 | **重大** | メモリ安全性 |
| D | `fb` NaN/Inf 未チェック | **中** | 数値安定性 |
| E | `copyLatest` TOCTOU | **中** | スレッド安全性 |
| F | `init` 空ブロック | **低** | コード品質 |
| G | 冗長な負値チェック | **低** | コード品質 |
| H | 例外安全性 | **低** | メモリ安全性 |

Bug A/B は連鎖的に発生し、**フィルタ発散時に音声出力が完全に破綻**するリスクがあるため、最優先での修正を推奨します。