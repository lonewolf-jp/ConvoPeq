#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
from datetime import datetime

filepath = 'INTEGRATED_BUG_LIST.md'

with io.open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# ── Fix Bug 1-7: line number h:136 -> h:169 ──
content = content.replace(
    '`src/audioengine/ISRRetire.h:136`',
    '`src/audioengine/ISRRetire.h:169`'
)

# ── Fix Bug 1-3: add note about additional _mm256_store_pd sites ──
content = content.replace(
    '将来の呼び出しで非アライドな `double*` が渡ると #GP 例外で即落ちる。',
    '将来の呼び出しで非アライドな `double*` が渡ると #GP 例外で即落ちる。\n**追加調査**: `_mm256_store_pd` は他に `TruePeakDetector.cpp:85`, `EQProcessor.Processing.cpp:37`, `MKLNonUniformConvolver.cpp:1319` および `:1580` にも存在する。これらの呼び出しサイトもアライメント契約を検証する必要がある。'
)

# ── Fix Bug 1-4: add note about DefaultFastTanhPolicy ──
content = content.replace(
    '3箇所の係数は現時点では一致しているが、将来 `SoftClipPadéPolicy` の係数をチューニングした場合、Float 入出力経路と Double 入出力経路でサチュレーションカーブが乖離する保守上のリスク。',
    '3箇所の係数は現時点では一致しているが、将来 `SoftClipPadéPolicy` の係数をチューニングした場合、Float 入出力経路と Double 入出力経路でサチュレーションカーブが乖離する保守上のリスク。\n**追加調査**: `FastTanhApprox.h:28` に `DefaultFastTanhPolicy` が存在する (27/9 の Padé 近似)。`EQProcessor.Processing.cpp:104` は `fastTanh<>()` (デフォルトポリシー) を使用している。DSPCoreFloat/IO は `SoftClipPadéPolicy` (10395/1260/21 係数) と一致するポリシーを使用するべき。'
)

# ── Fix Bug 1-8: add note about MAX_FILE_LENGTH guard ──
content = content.replace(
    '`tempFloatBuffer(numChannels, static_cast<int>(fileLength))` で `fileLength` は `MAX 2,147,483,647` まで許可。ステレオで8GB確保試行。',
    '`tempFloatBuffer(numChannels, static_cast<int>(fileLength))` で `fileLength` は `MAX_FILE_LENGTH = 2147483647` (INT32_MAX) まで許可 (LoaderThread.cpp:450, ResampleAndFallback.cpp:293 にガード存在)。ステレオで `2 * 2B * 4bytes ≈ 16GB` の確保試行。ガードは整数オーバーフローを防止するが、メモリ容量の制限はしない。'
)

# ── Fix Bug 1-9: line numbers and fix note ──
content = content.replace(
    '**ファイル**: `src/MKLNonUniformConvolver.cpp:842, 846, 854`',
    '**ファイル**: `src/MKLNonUniformConvolver.cpp:843, 847, 853` (診断用: 843, 847; 実際のアロケーション: 853)'
)
content = content.replace(
    '**修正**: `static_cast<size_t>(l.fftSize) * sizeof(double)` に統一する。',
    '**修正**: `l.fftSize` を `int` から `int64_t` または `size_t` に変更し、`static_cast<size_t>(l.fftSize) * sizeof(double)` に統一する。`static_cast<size_t>` のみでは `fftSize` が `int` で溢れた時点で既に不正値になっているため、型そのものの変更が必要。'
)

# ── Fix Bug 1-5: line number ──
content = content.replace(
    '**ファイル**: `src/audioengine/AudioEngine.h:1059` (宣言), `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:341` (定義)',
    '**ファイル**: `src/audioengine/AudioEngine.h:1066` (宣言), `src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp:341` (定義)'
)

# ── Now Add detailed fix designs section ──
design_section = """

---

## 8. 詳細修正設計 (Detailed Fix Design)

以下は各バグに対する詳細設計。設計方針、修正アプローチ、コードパターン、テスト方針、リスク評価を含む。

### 1-1. `nucHCMode` / `nucLCMode` がセッション永続化から欠落

**Root Cause**: `ConvolverProcessor::getState()` (StateAndUI.cpp:202) は `juce::ValueTree` にテール関連のプロパティを書き出すが、`nucHCMode` と `nucLCMode` を `setProperty()` で追加していない。`setState()` (行 289) も同様に読み込みがない。

**Fix Approach**:
```cpp
// getState() に追加 (StateAndUI.cpp:235 付近)
v.setProperty("nucHCMode", static_cast<int>(snapshot.nucHCMode), nullptr);
v.setProperty("nucLCMode", static_cast<int>(snapshot.nucLCMode), nullptr);

// setState() に追加 (StateAndUI.cpp:358 付近)
if (v.hasProperty("nucHCMode")) {
    const int hcVal = static_cast<int>(v.getProperty("nucHCMode"));
    const int lcVal = static_cast<int>(v.getProperty("nucLCMode"));
    const auto hc = juce::jlimit(static_cast<int>(convo::HCMode::Sharp),
                                 static_cast<int>(convo::HCMode::Natural), hcVal);
    const auto lc = juce::jlimit(static_cast<int>(convo::LCMode::Natural),
                                 static_cast<int>(convo::LCMode::Sharp), lcVal);
    setNUCFilterModes(static_cast<convo::HCMode>(hc), static_cast<convo::LCMode>(lc));
}
```

**Testing**: 保存→再読込後に `getNucHCMode()` / `getNucLCMode()` が元値と一致することを確認。

**Risk**: Low — pure additon, no logic change.

---

### 1-2. `coordinatorDeferredRing_` / `lastResortQueue_` デッドコード

**Root Cause**: `coordinatorDeferredRing_` は `.pop()` でのみ消費され、producer (push) が存在しない。`lastResortQueue_` は drain 関数内でのみ read/compaction が行われ、new entry の追加は存在しない。`coordinatorDeferredCount_` は `fetchSub`/`consume` でのみ使用され、increment がない。`lastResortCount_` は `{0}` 初期化後増加しない。

**Fix Approach**:
1. `lastResortQueue_` の値初期化: `RetireOverflowEntry lastResortQueue_[kLastResortQueueCapacity]{};` (brace-init)
2. producer 実装の検討: `emitRetireIntent` overflow 時に `lastResortQueue_` への enqueue を追加
3. デッドコード削除: producer を実装しない場合、`coordinatorDeferredRing_` と `lastResortQueue_` を削除

**Testing**: 
- TSan で `coordinatorDeferredRing_` / `lastResortQueue_` への同時アクセスなしを確認
- drain 関数で `lastResortCount_ == 0` の場合即 return を確認

**Risk**: Medium — producer 追加は複雑な同期ロジックが必要。

---

"""

# Append the design section before section 7
content = content.replace(
    "\n## 7. 未調査領域 (今後の調査候補)",
    design_section + "\n## 7. 未調査領域 (今後の調査候補)"
)

# Add detailed designs for remaining bugs
more_designs = """
### 1-3. `_mm256_store_pd` のアライメント保証なし

**Root Cause**: `convertFloatToDoubleHighQuality()` (InputBitDepthTransform.h:108) は `double* dst` の 32-byte アライメントを保証しない。`_mm256_store_pd` は 32-byte アライメントが必要。現在の呼び出し元は偶然アライドだが将来の変更で非アライドになる可能性。

**Fix Approach**:
```cpp
// Option A: アンアライドストア (最も安全)
_mm256_storeu_pd(dst + i, _mm256_cvtps_pd(lo));
_mm256_storeu_pd(dst + i + 4, _mm256_cvtps_pd(hi));

// Option B: 契約の明示 (C++20 preconditions)
[[expects: reinterpret_cast<uintptr_t>(dst) % 32 == 0]]
```
**Additional sites to audit**: `TruePeakDetector.cpp:85`, `EQProcessor.Processing.cpp:37`, `MKLNonUniformConvolver.cpp:1319, 1580`

**Testing**: 非アライドバッファを渡して #GP が発生しないことを確認 (AddressSanitizer + 非アライド割り当て)。

**Risk**: Low — `_mm256_storeu_pd` はわずかに遅いが安全。

---

### 1-4. `fastTanh` の3箇所独立複製

**Root Cause**: `FastTanhApprox.h` にテンプレート版 `fastTanh<Policy>` が存在する (line 101-107, default `DefaultFastTanhPolicy`)。`SoftClipPadéPolicy` (line 63) は 10395/1260/21 係数。`DSPCoreDouble.cpp:127,191` は `SoftClipPadéPolicy` を使用。`DSPCoreFloat.cpp:146` と `DSPCoreIO.cpp:76` は独自の `inline double fastTanh(double x)` を複製。

**Fix Approach**:
1. `FastTanhApprox.h:63` の `SoftClipPadéPolicy` を `SoftClipPadeApproxPolicy` にリネーム (非ASCII 'é' 除去)
2. `DSPCoreFloat.cpp:146` の独自 `fastTanh` を削除し、`#include "dsp/math/FastTanhApprox.h"` を追加
3. `DSPCoreFloat.cpp:186` の呼び出しを `convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>(...)` に変更
4. `DSPCoreIO.cpp:76,116` に同じ変更を適用

```cpp
// DSPCoreFloat.cpp:186 — Before:
const double clipped = threshold + knee * fastTanh((abs_x - threshold) / knee);
// After:
const double clipped = threshold + knee * convo::dsp::fastTanh<convo::dsp::SoftClipPadeApproxPolicy>((abs_x - threshold) / knee);
```

**Testing**: Float パスと Double パスのサチュレーション出力が一致することを確認 (最大許容差 1e-12)。

**Risk**: Medium — 係数の不一致により既存のサウンドが変化する可能性。

---

### 1-5. `musicalSoftClip` が未使用のデッドコード

**Root Cause**: `AudioEngine::DSPCore::musicalSoftClip()` (h:1066, DSPCoreIO.cpp:341) は宣言・定義されているが、コードベース全体で呼び出されていない。実際の処理はファイルローカルな `musicalSoftClipScalar()` が使用されている (DSPCoreFloat:165, DSPCoreDouble:107, DSPCoreIO:95)。

**Fix Approach**: クラスメソッド `musicalSoftClip()` を削除するか、実際に呼び出し側に結線する。DSPCoreDouble.cpp:217 では `musicalSoftClipScalar` が直接呼ばれており、クラスメソッド版は冗長。

**Testing**: クラスメソッド削除後、ビルド成功 + 既存テスト通過を確認。

**Risk**: Low — デッドコード削除。

---

### 1-6. IPP FFT 戻り値無視

**Root Cause**: `MklFftEvaluator.h:270-271, 425-426` で `ippsFFTFwd_RToCCS_64f()` の戻り値 (`IppStatus`) を無視している。IPP 初期化失敗や不正な `fftSpec` 時に無効なデータが残る。

**Fix Approach**:
```cpp
// Line 270-271 Before:
ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
// After:
IppStatus st1 = ippsFFTFwd_RToCCS_64f(inputLeft,  reinterpret_cast<Ipp64f*>(spectrumLeft),  fftSpec, fftWorkBuf);
IppStatus st2 = ippsFFTFwd_RToCCS_64f(inputRight, reinterpret_cast<Ipp64f*>(spectrumRight), fftSpec, fftWorkBuf);
if (st1 != ippStsNoErr || st2 != ippStsNoErr) {
    juce::Logger::writeToLog("MklFftEvaluator: ippsFFTFwd_RToCCS_64f failed");
    return nullptr; // or appropriate error
}
```
**Note**: `FFTBackend.cpp:130,146` は既に戻り値をキャプチャ済み (修正済み)。

**Testing**: 無効な fftSpec でエラーが正しく検出されることを確認。

**Risk**: Low — エラーハンドリング追加。

---

### 1-7. `ISRRetire.cpp` での Mutex 使用

**Root Cause**: `emitRetireIntentRT()` (ISRRetire.cpp:94) は RT から呼ばれる可能性があるが、内部で `emitRetireIntent()` (line 102) を呼び出し、これが `fallbackMutex_` (h:169) を取得する (line 44, 135, 265)。RT スレッドでの mutex 取得は優先度逆転でオーディオドロップを引き起こす。

**Fix Approach**:
1. RT パス (`emitRetireIntentRT`) は `overflowRing_` のみに退避 (mutex なしロックフリー)
2. Non-RT パス (`emitRetireIntent`) は `overflowRing_` から drain し、必要時のみ `fallbackMutex_` を取得
3. `emitRetireIntentRT` が `fallbackMutex_` にアクセスしないことを保証

```cpp
// emitRetireIntentRT — RT path, no mutex
void LifetimeState::emitRetireIntentRT(const RetireIntent& intent) noexcept {
    // Only push to overflowRing_ (lock-free)
    if (overflowRing_ != nullptr && overflowRing_->tryPush(entry)) {
        return;
    }
    // If overflowRing_ is full or null, drop (RT-safe — no fallback to mutex)
    overflowDroppedCount_.fetch_add(1, std::memory_order_relaxed);
}
```

**Testing**: RT スレッドから `emitRetireIntentRT` を連続呼出しし、mutex が取得されないことを TSan で確認。

**Risk**: High — RT パスの設計変更。

---

"""
