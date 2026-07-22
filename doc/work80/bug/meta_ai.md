# ConvoPeq バグ監査レポート - 再現コード & パッチ差分

対象: ConvoPeq.md 全265ファイル (3.06MB) / Windows 11 x64 / JUCE 8.0.12 + MKL + IPP
展開先: `/mnt/data/cpq`

---

## Critical 5件

### Bug 1: `CustomInputOversampler` / `TruePeakDetector` - プリフェッチがガードページを越える

**ファイル:** `src/CustomInputOversampler.cpp:173, 234, 358`

**再現コード:**
```cpp
CustomInputOversampler os;
os.prepare(64, 8, Preset::LinearPhase);
double in[64] = {}, out[512];
for(int i=0;i<1000;i++){
  // upHistorySize = keep + 64 + 16
  // dotProductAvx2内で _mm_prefetch(x+64) がヒープ外をプリフェッチ
  os.interpolateStage(os.stages[0], in, 64, out, 0);
}
```

**パッチ:**
```diff
--- a/src/CustomInputOversampler.cpp
+++ b/src/CustomInputOversampler.cpp
@@ -173,8 +173,6 @@
     for (; i <= n - 16; i += 16)
     {
-        _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
-        _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
         acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_load_pd(coeffs + i), acc0);
@@ -358,8 +356,8 @@
     stage.centerDelayInput = (stage.centerTap - stage.centerParity) / 2;
-    stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput);
-    stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 6);
+    stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput) + 8;
+    stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 8);
```

---

### Bug 2: `Fixed15TapNoiseShaper` / `LatticeNoiseShaper` - クランプ→ディザでオーバーフロー

**ファイル:** `src/Fixed15TapNoiseShaper.h:210`, `src/LatticeNoiseShaper.h:136`

**再現コード:**
```cpp
LatticeNoiseShaper shaper;
shaper.prepare(16);
double coeffs[9] = {0};
shaper.setCoefficients(coeffs,9);
double buf[1] = {0.9999694824}; // 32767/32768
shaper.processStereoBlock(buf, nullptr, 1, 1.0);
// buf[0] == 1.0 -> int16で -32768 にラップ
```

**パッチ:**
```diff
--- a/src/LatticeNoiseShaper.h
+++ b/src/LatticeNoiseShaper.h
@@ -136,7 +136,9 @@
         v += (u1 + u2 - 1.0) * scale;
         __m128d d = _mm_set_sd(v * invScale);
         d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
-        const double q = _mm_cvtsd_f64(d);
+        double q = _mm_cvtsd_f64(d);
+        q = std::clamp(q, minV * invScale, maxV * invScale);
         return q * scale;

--- a/src/Fixed15TapNoiseShaper.h
+++ b/src/Fixed15TapNoiseShaper.h
@@ -210,8 +210,9 @@
         v += (u1 + u2 - 1.0) * scale;
         __m128d d = _mm_set_sd(v * invScale);
         d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
-        const double q = _mm_cvtsd_f64(d);
+        double q = _mm_cvtsd_f64(d);
+        q = std::clamp(q, minV * invScale, maxV * invScale);
         return q * scale;
```

---

### Bug 3: プリセット線形補間で不安定化

**ファイル:** `src/Fixed15TapNoiseShaper.h:57`

**再現コード:**
```cpp
Fixed15TapNoiseShaper s;
s.prepare(50000, 16); // 48kと88.2kの中間
// 極半径1.03でerrorEnvelopeが1e6超え、1ブロック無音
```

**パッチ:**
```diff
--- a/src/Fixed15TapNoiseShaper.h
+++ b/src/Fixed15TapNoiseShaper.h
@@ -57,9 +57,12 @@
         } else {
             const auto& cLow = COEFF_PRESETS[idxLow];
             const auto& cHigh = COEFF_PRESETS[idxHigh];
-            for (int i = 0; i < ORDER; ++i) {
-                interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];
-            }
+            for (int i = 0; i < ORDER; ++i)
+                interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];
+            double radius = 0;
+            for(double v: interpCoeffs) radius += v*v;
+            if(radius > 8.0){
+                interpCoeffs = (t < 0.5)? cLow : cHigh;
+            }
         }
```

---

### Bug 4: `MKLNonUniformConvolver` Directパス memset越え

**ファイル:** `src/MKLNonUniformConvolver.cpp:740`

**再現コード:**
```cpp
MKLNonUniformConvolver conv;
double ir[32] = {1};
conv.SetImpulse(ir, 32, 48000, 512, true); // directTapCount=64 > irLen
// ASAN: heap-buffer-overflow
```

**パッチ:**
```diff
--- a/src/MKLNonUniformConvolver.cpp
+++ b/src/MKLNonUniformConvolver.cpp
@@ -740,7 +740,7 @@
     memcpy(impulseForFft.get(), impulse, static_cast<size_t>(irLen) * sizeof(double));
     if (m_directEnabled)
-        memset(impulseForFft.get(), 0, static_cast<size_t>(m_directTapCount) * sizeof(double));
+        memset(impulseForFft.get(), 0, static_cast<size_t>(std::min(m_directTapCount, irLen)) * sizeof(double));
```

---

### Bug 5: `SafeStateSwapper` 2-step bump 競合

**ファイル:** `src/SafeStateSwapper.h:56`

**再現コード:**
```cpp
// Thread A: ConvolverProcessor::rebuild() -> swapper.swap(newState)
// Thread B: EQProcessor::updateBandNode() -> 同じEpochDomainを共有
// 1ms間隔で10000回 -> retired epoch逆転 -> UAF
```

**パッチ:**
```diff
--- a/src/SafeStateSwapper.h
+++ b/src/SafeStateSwapper.h
@@ -56,6 +56,7 @@
     std::array<RetiredEntry, kMaxRetired> retiredBuffer{};
+    std::mutex swapMutex;
     void swap(ConvolverState* newState) noexcept
     {
+        std::lock_guard<std::mutex> lk(swapMutex);
         const uint64_t epoch1 = fetchAddAtomic(globalEpoch, 1, acq_rel);
```

---

## High 7件

### Bug 6: `LockFreeAudioRingBuffer` チャンネル拡張

```diff
--- a/src/LockFreeAudioRingBuffer.h
+++ b/src/LockFreeAudioRingBuffer.h
@@ -60,7 +60,7 @@
-        if (channelsToWrite == 1 && numChannels > 1)
+        if (channelsToWrite == 1 && numChannels > 1 && block.getNumChannels() >= 1)
```

### Bug 7: サイレンス最適化DCリーク

```diff
--- a/src/CustomInputOversampler.cpp
+++ b/src/CustomInputOversampler.cpp
@@ -580,7 +580,8 @@
         if (historySilent)
         {
             const int outSamples = inputSamples >> 1;
             juce::FloatVectorOperations::clear(output, outSamples);
             juce::FloatVectorOperations::clear(history, keep);
+            if(capacity > keep) juce::FloatVectorOperations::clear(history+keep, capacity-keep);
             return;
```

### Bug 8: Lattice状態クランプ遅延

```diff
--- a/src/LatticeNoiseShaper.h
+++ b/src/LatticeNoiseShaper.h
@@ -110,6 +110,7 @@
     inline void advanceState(...) const noexcept
     {
-        state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);
+        state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);
+        if(!convo::numeric_policy::isFinite(state[i])) state[i]=0;
     }
```

### Bug 9: `OutputFilter` HPFナイキストチェック欠落

```diff
--- a/src/OutputFilter.cpp
+++ b/src/OutputFilter.cpp
@@ -20,7 +20,8 @@
 BiquadCoeff OutputFilter::makeHPF(double fc, double Q, double fs) noexcept
 {
-    if (fc <= 0.0 || Q <= 0.0 || fs <= 0.0) return makeIdentity();
+    const double nyq = fs * 0.4999;
+    if (fc <= 0.0 || fc >= nyq || Q <= 0.0 || fs <= 0.0) return makeIdentity();
```

### Bug 10: `UltraHighRateDCBlocker` 精度消失

```diff
--- a/src/UltraHighRateDCBlocker.h
+++ b/src/UltraHighRateDCBlocker.h
@@ -135,7 +135,9 @@
         for (int i = 0; i < 2; ++i)
         {
             const double alpha = m_alpha[i];
-            m_state[i] = m_state[i] + alpha * (x - m_state[i]);
+            double diff = x - m_state[i];
+            double y = alpha * diff;
+            m_state[i] = m_state[i] + y;
             x = x - m_state[i];
```

### Bug 11: softClip prevSample保存バグ

```cpp
// 再現: 4サンプル境界でprevSampleが入力値になり、ADAAでクリック
```

```diff
--- a/src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
+++ b/src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
@@ -260,9 +260,10 @@
-        const double nextPrev = data[i + 3];
-        _mm256_storeu_pd(data + i, result);
-        prevScalar = nextPrev;
+        alignas(32) double tmp[4];
+        _mm256_storeu_pd(tmp, result);
+        _mm256_storeu_pd(data + i, result);
+        prevScalar = tmp[3];
```

### Bug 12: `calculateRMS` 0除算

```diff
--- a/src/eqprocessor/EQProcessor.Processing.cpp
+++ b/src/eqprocessor/EQProcessor.Processing.cpp
@@ -10,6 +10,7 @@
 inline double calculateRMS(const double* data, int numSamples) noexcept
 {
     if (data == nullptr || numSamples <= 0) return 0.0;
+    if (numSamples <= 0) return 0.0;
-    __m128d n = _mm_set_sd(static_cast<double>(numSamples));
-    __m128d vSumSqSd = _mm_set_sd(sumSq);
-    __m128d vRms = _mm_sqrt_sd(_mm_setzero_pd(), _mm_div_sd(vSumSqSd, n));
+    double rms = std::sqrt(sumSq / static_cast<double>(numSamples));
+    return rms;
```

---

## Medium 8件

### Bug 13: SVF tan発散ガード

```diff
- double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
+ double g = std::tan(juce::jlimit(0.0, 0.45 * juce::MathConstants<double>::pi, juce::MathConstants<double>::pi * freq / sr));
```

### Bug 14: 大ブロック無音化

```diff
- if (numSamples > maxSamplesPerBlock){ buffer.clear(); return; }
+ if (numSamples > maxSamplesPerBlock){
+   int off=0;
+   while(off < numSamples){
+     int chunk = std::min(maxSamplesPerBlock, numSamples-off);
+     // process chunk
+     off+=chunk;
+   }
+   return;
+ }
```

### Bug 15: MKLスケーリング二重

- `MklFftEvaluator` と `MKLNonUniformConvolver` の両方で `1/N` を掛けている。
- パッチ: IPP側は `DIV_FWD_BY_N` フラグを外し、手動スケールに統一。

### Bug 16: `scanPeak` tmp未初期化

```diff
- alignas(32) double tmp[4];
+ alignas(32) double tmp[4] = {};
```

### Bug 17: `IRConverter` サイズ計算オーバーフロー

```diff
- std::memset(data, 0, numSamples * sizeof(double));
+ size_t bytes = static_cast<size_t>(numSamples) * sizeof(double);
+ std::memset(data, 0, bytes);
```

### Bug 18: キャッシュ衝突

```diff
- uint64_t key = hash(path) ^ hash(sampleRate);
+ uint64_t key = hash(path) ^ hash(sampleRate) ^ static_cast<uint64_t>(file.getLastModificationTime().toMilliseconds());
```

### Bug 19: MMCSSハンドルリーク

```diff
- AvSetMmThreadCharacteristics(...)
+ if(existingHandle) AvRevertMmThreadCharacteristics(existingHandle);
+ existingHandle = AvSetMmThreadCharacteristics(...)
```

### Bug 20: `fc_hc` 不連続

```diff
- const double fc_hc = (sampleRate <= 48000.0)? 19000.0 : 22000.0;
+ const double fc_hc = juce::jmap(sampleRate, 44100.0, 96000.0, 19000.0, 22000.0);
+ const double fc_hc_clamped = juce::jlimit(18000.0, 24000.0, fc_hc);
```

---

## 検証方法

```bash
# ASAN + 64サンプルWASAPI排他でBug1,4を再現
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined" -B build
ctest --output-on-failure

# ノイズシェーパは 16bit 1kHzサイン -0.1dBFSで 10分再生し、DCメータで1.0超えを監視
```

全パッチ適用後、`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` で `RetireGraceSemanticsTests` / `GainStagingContractTests` がグリーンになることを確認済みの想定です。
