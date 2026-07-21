# ConvoPeq 追加バグ報告（第3回）

前回までの報告でカバーされていないバグを、コードを精査して報告します。

---

## Bug 1: `ConvolverProcessor::StereoConvolver::init` — `SetImpulse` が `bad_alloc` 以外の例外を投げた場合の `irData` メモリリーク 【重大】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::init`)

**問題:**
`irData[0]` / `irData[1]` は `try` ブロックの**前**で代入されるが、`catch` は `bad_alloc` しか捕捉しない。`SetImpulse` がそれ以外の例外を投げた場合、例外が伝播し `irData` が解放されない。

```cpp
irData[0] = irL;  // try の前で所有権移譲
irData[1] = irR;
// ...
try {
    auto nuc0 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    auto nuc1 = convo::aligned_make_unique<convo::MKLNonUniformConvolver>();
    if (nuc0->SetImpulse(irData[0], ...) && nuc1->SetImpulse(irData[1], ...))
    { /* success */ return true; }
}
catch (const std::bad_alloc&) { }  // ← bad_alloc しか捕捉しない
// cleanup（bad_alloc の場合のみ到達）
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
```

`SetImpulse` が `std::runtime_error` 等を投げた場合:
- `nuc0` / `nuc1` の `unique_ptr` は正しく破棄される
- しかし `irData[0]` / `irData[1]` は**解放されない**（`try` の前で代入済み、`catch` を通過しない）
- `StereoConvolver` のデストラクタは `jassert` のみで解放しない
- **結果: `irData` のメモリリーク**

**修正:**
```cpp
try {
    // ...
}
catch (const std::bad_alloc&) {
    // bad_alloc 固有の処理
}
catch (...) {
    // 全例外を捕捉して irData を解放
}
// cleanup（全経路で到達）
```

---

## Bug 2: `ConvolverProcessor::StereoConvolver::clone` — `init` 失敗時の `newConv` リーク 【中】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::clone`)

**問題:**
`init` が `l.release()` / `r.release()` の**後**で失敗した場合、`newConv` の `unique_ptr` デストラクタが `~StereoConvolver()` を呼ぶが、デストラクタは `jassert` のみで `nucConvolvers` / `irData` を解放しない。

```cpp
auto newConv = convo::aligned_make_unique<StereoConvolver>();
// ...
if (!newConv->init(l.release(), r.release(), ...))
    return nullptr;  // newConv のデストラクタが走るが、irData は init 内で解放済み
return newConv.release();
```

`init` 失敗時に `init` 内部で `irData` は解放されるが、`nucConvolvers` が設定されている場合（`SetImpulse` 成功後に `nuc1` が失敗した場合）、`nucConvolvers` は `init` の cleanup で `nullptr` に設定されるため、デストラクタの `jassert` はパスする。

しかし、`init` が `bad_alloc` 以外の例外を投げた場合（Bug 1 と同じ経路）、`irData` が解放されず、`newConv` のデストラクタが `jassert` で停止する（Debug）またはリークする（Release）。

**修正:** Bug 1 の修正と同一（`catch (...)` で全例外を捕捉）。

---

## Bug 3: `ConvolverProcessor::StereoConvolver::process` — `numSamples <= 0` の未チェック 【中】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::process`)

**問題:**
`numSamples` が 0 または負の場合のチェックがない。`numSamples` が負の場合、`numSamples * sizeof(double)` が `size_t` に変換されて巨大な値になり、`std::memset` でバッファオーバーフローが発生する。

```cpp
void process(int channel, const double* in, double* out, int numSamples)
{
    if (channel < 0 || channel >= 2 || !nucConvolvers[channel])
    {
        std::memset(out, 0, numSamples * sizeof(double));  // numSamples < 0 でオーバーフロー
        return;
    }
    nucConvolvers[channel]->Add(in, numSamples);  // numSamples < 0 で未定義動作
    const int got = nucConvolvers[channel]->Get(out, numSamples);
    if (got < numSamples)
        std::memset(out + got, 0, (numSamples - got) * sizeof(double));
}
```

**修正:**
```cpp
if (channel < 0 || channel >= 2 || !nucConvolvers[channel] || numSamples <= 0)
{
    if (numSamples > 0)
        std::memset(out, 0, numSamples * sizeof(double));
    return;
}
```

---

## Bug 4: `ConvolverProcessor::StereoConvolver::init` — 失敗時に `storedFilterSpec` / `hasStoredFilterSpec` がリセットされない 【低】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::init`)

**問題:**
`init` 失敗時の cleanup で `storedFilterSpec` / `hasStoredFilterSpec` がリセットされない。`init` が再呼び出しされた場合、前回の `storedFilterSpec` が使用される可能性がある。

```cpp
// 失敗時の cleanup
destroyNUCConvolver(nucConvolvers[0]);
destroyNUCConvolver(nucConvolvers[1]);
if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
irDataLength = 0;
latency = 0;
this->irLatency = 0;
// storedFilterSpec / hasStoredFilterSpec はリセットされない
```

**修正:**
```cpp
// cleanup に追加
hasStoredFilterSpec = false;
storedFilterSpec = convo::FilterSpec{};
```

---

## Bug 5: `ConvolverProcessor::StereoConvolver::init` — 失敗時に `callQuantumSamples` / `storedSampleRate` 等がリセットされない 【低】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::init`)

**問題:**
`init` 失敗時の cleanup で `callQuantumSamples` / `storedSampleRate` / `storedKnownBlockSize` / `storedScale` / `storedDirectHeadEnabled` がリセットされない。`init` が再呼び出しされた場合、前回の値が使用される可能性がある。

**修正:**
```cpp
// cleanup に追加
callQuantumSamples = 0;
storedSampleRate = 0.0;
storedKnownBlockSize = 0;
storedScale = 1.0;
storedDirectHeadEnabled = false;
```

---

## Bug 6: `ConvolverProcessor::processBypassWithLatencyCompensation` — `delayWritePos` の非アトミックアクセス 【低】

**対象:** `ConvolverProcessor.h` (`processBypassWithLatencyCompensation`)

**問題:**
`delayWritePos` は非アトミックのメンバ変数。`processBypassWithLatencyCompensation` は Audio Thread から呼ばれるが、`reset()` も `delayWritePos` を書き込む。`reset()` が Audio Thread 以外から呼ばれた場合、データレースが発生する。

```cpp
int delayWritePos = 0;  // 非アトミック
// ...
void processBypassWithLatencyCompensation(...) noexcept
{
    const int writePos = delayWritePos;  // Audio Thread から読み取り
    // ...
    delayWritePos = nextWritePos;  // Audio Thread から書き込み
}
void reset()
{
    delayWritePos = 0;  // reset() からも書き込み
}
```

**修正:**
`delayWritePos` を `std::atomic<int>` に変更するか、`reset()` が Audio Thread からのみ呼ばれることを保証する。

---

## Bug 7: `ConvolverProcessor::StereoConvolver::clone` — `irDataLength` が負の場合の `makeAlignedArray` オーバーフロー 【低】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::clone`)

**問題:**
`irDataLength` が負の場合、`static_cast<size_t>(irDataLength)` が巨大な値になり、`makeAlignedArray` が巨大なアロケーションを試みる。

```cpp
auto l = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));
```

`irDataLength` は `init` で設定されるため通常は正の値だが、防御的チェックがない。

**修正:**
```cpp
if (irDataLength <= 0 || !irData[0] || !irData[1])
    return newConv.release();  // IR なしでクローン
```

---

## Bug 8: `ConvolverProcessor::StereoConvolver::process` — `got > numSamples` の未チェック 【低】

**対象:** `ConvolverProcessor.h` (`StereoConvolver::process`)

**問題:**
`Get()` が `numSamples` より大きい値を返した場合、`got < numSamples` のチェックがスキップされ、`got > numSamples` の場合にバッファオーバーフローが発生する。

```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
// got > numSamples の場合、何もチェックされない
```

**修正:**
```cpp
const int got = nucConvolvers[channel]->Get(out, numSamples);
if (got < numSamples)
    std::memset(out + got, 0, (numSamples - got) * sizeof(double));
// got > numSamples は Get() の契約違反だが、防御的にチェック
jassert(got <= numSamples);
```

---

## 重要度まとめ

| # | バグ | 重要度 | 対象 |
|---|------|--------|------|
| 1 | `init` — `SetImpulse` 非 `bad_alloc` 例外で `irData` リーク | **重大** | `StereoConvolver::init` |
| 2 | `clone` — `init` 失敗時の `newConv` リーク | **中** | `StereoConvolver::clone` |
| 3 | `process` — `numSamples <= 0` 未チェック | **中** | `StereoConvolver::process` |
| 4 | `init` 失敗時に `storedFilterSpec` 未リセット | **低** | `StereoConvolver::init` |
| 5 | `init` 失敗時に `callQuantumSamples` 等未リセット | **低** | `StereoConvolver::init` |
| 6 | `delayWritePos` の非アトミックアクセス | **低** | `processBypassWithLatencyCompensation` |
| 7 | `clone` — `irDataLength` 負値のオーバーフロー | **低** | `StereoConvolver::clone` |
| 8 | `process` — `got > numSamples` 未チェック | **低** | `StereoConvolver::process` |

Bug 1 は前回報告と重複する部分がありますが、`catch (...)` による全例外捕捉が根本的な修正であり、Bug 2・3 も連鎖的に修正されます。最優先での修正を推奨します。