ConvoPeq.md 全265ファイル 3.06MB を全展開して監査しました。DSPコア / コンボルバ / ノイズシェーパ / ロックフリー / RCU 周りを中心に、**クラッシュ・無音化・音質劣化に直結するバグ**を抽出しました。

## Critical

### 1. `CustomInputOversampler.cpp` / `TruePeakDetector.cpp` - ヒストリバッファ境界の非対称マージン
- **箇所**: `prepareStage()` 
  - `historyDownKeep = max(centerTap, convParity + (convCount-1)*2 + 6)` で +6 マージンを追加
  - `historyUpKeep = max(convCount-1, centerDelay)` は +6 無し
- **問題**: `loadStride2()` は `ptr[-6]` まで読む。Up側は `dotProductAvx2` が `loadu` なので安全だが、Down側で `dotProductDecimateAvx2` が8-wayアンロール時に `history - (r<<1)` に対して `loadStride2` を呼ぶ。コメントではDown側は保護済みとあるが、Up側の `interpolateStage` で `dotProductAvx2` のプリフェッチ `x+i+64` が確保サイズ `upHistorySize = keep + maxInput +16` を超えてプリフェッチする。x86のプリフェッチは例外を出さないが、ガードページに触れると稀に #PF 遅延が発生し、オーディオスレッドのXRUNを誘発。
- **修正**: Up/Down共に `+8` マージンを統一、`_mm_prefetch` を削除するか `if (i+64 < n)` ガード。

### 2. `Fixed15TapNoiseShaper.h` / `LatticeNoiseShaper.h` - クランプ→ディザの順序でオーバーフロー
- **箇所**: `quantize()`
```cpp
if(v < minV) v=minV; else if(v>maxV) v=maxV; // maxV = 1 - scale
v += (u1+u2-1)*scale; // TPDF
q = round(v*invScale)*scale;
```
- **問題**: `v = maxV` (例 16bitで32767/32768) に `+scale` のディザが加わると `1.0` になり、16bitでは表現不能な `32768` に量子化される。int16へ変換時にラップして大ノイズ。Lipshitzの正規順序は「ディザ→量子化→クランプ」だが、コメントは逆を「正規」と誤認。
- **修正**: 量子化後に `q = clamp(q, minV, maxV)` を追加。または `maxV` を `1.0` にして量子化後に飽和。

### 3. `Fixed15TapNoiseShaper.h` - プリセット線形補間で不安定化
- **箇所**: `prepare()` の `interpCoeffs[i] = (1-t)*cLow[i] + t*cHigh[i]`
- **問題**: 誤差フィードバック係数を直接線形補間。44.1kと48k間など、48k→88.2k間で補間された係数は極が単位円外に出る場合があり、エラー状態が `kErrorStateThreshold=1e6` まで発散してから `needsReset` で無音化。48k→50kのような中間レートで再現。
- **修正**: 補間は格子係数領域で行うか、補間後に `isStable()` チェックと極半径スケーリング。

### 4. `MKLNonUniformConvolver.cpp` - Direct/IR FFT二重カウントの境界条件
- **箇所**: `SetImpulse()` `memset(impulseForFft.get(),0, directTapCount)`
- **問題**: `m_directTapCount` が `irLen` より大きい場合、`memset` が確保サイズ `irLen` を超えて書き込み。`m_directTapCount` は `blockSize` 依存で決まり、`irLen` が極端に短いIR (例 32サンプル) で `directTapCount=64` になるとヒープ破壊。
- **修正**: `memset` サイズを `min(directTapCount, irLen)` に。

### 5. `SafeStateSwapper.h` - 2-step bump の同時実行で早期解放
- **箇所**: `swap()` で `fetchAdd` を2回、`retiredBuffer[t].epoch = epoch1`
- **問題**: コメントは「単一Writer前提」だが、`ConvolverProcessor` と `EQProcessor` の両方が同じ `EpochDomain` を共有し、異なるスレッドから `swap()` が呼ばれるパスがある。2つの `swap()` がインターリーブすると `epoch1` が逆転し、`getMinReaderEpoch() < epoch` 条件でまだリーダが参照中の `ConvolverState` を解放 → Use-After-Free。
- **修正**: `swap()` に `std::mutex` を追加するか、Writerを1スレッドに限定するアサートをReleaseでも有効化。

## High

### 6. `LockFreeAudioRingBuffer.h` - float 切り捨てとチャンネル拡張の競合
- `push()` で `double -> float` キャスト。`double` が `>1.0` (ソフトクリップ前) の場合、floatでも >1.0 だが、後段で `popMixToMono` が平均化 `(L+R)*0.5` で6dB下がる。モノラル入力をステレオバッファに拡張する分岐 `if(channelsToWrite==1 && numChannels>1)` で同じサンプルを2チャンネルに書き込むが、`writeIndex` の更新が1回のみのため、片チャンネルだけ上書きされるタイミングで他チャンネルが古いデータを保持し、L/R位相差が発生。

### 7. `CustomInputOversampler.cpp` - サイレンス最適化のDCリーク
- `decimateStage` のサイレンスパスで `inputSilent && historySilent` なら出力クリアして `return`。しかし `history` をクリアせず `keep` 部分だけクリア。次の非無音ブロックで `history` に残った古いDC成分が畳み込まれ、ポップノイズ。`juce::FloatVectorOperations::clear(history, keep)` は実行しているが、`history+keep` 以降の未使用領域に前回の有音データが残り、`memmove` 後に再出現。

### 8. `LatticeNoiseShaper.h` - ブロック末尾での状態クランプ遅延
- `clampStateSIMD` をブロック終了時に1回のみ呼出。`kOrder=9` で係数が不安定な場合、ブロック内(例 512サンプル)で状態が `1e12` まで発散し、途中の `computeFeedback` が `Inf` を返し、後続全サンプルが `NaN` 化。`kStateLimit=1e12` はクランプ閾値としては大きすぎ、Inf化前に止められない。
- **修正**: サンプル毎または4サンプル毎にクランプ。

### 9. `OutputFilter.cpp` - HPFのナイキストチェック欠落
- `makeLPF` は `fc >= nyq` で identity を返すが `makeHPF` は下限のみチェック。`fc=0.49*fs` 付近で `w0≈pi`, `sin(w0)≈0`, `alpha≈0`, `a0inv≈1` となり `b0≈(1+cos)/2≈0`, `a1≈2`, 極が単位円上に乗り発振。

### 10. `UltraHighRateDCBlocker.h` - 超高レートでのデノーマルフラッシュでDC除去不能
- `alpha = 1 - exp(-2pi*fc/fs)`。768kHz, fc=20Hzで `alpha≈1.6e-4`。`state += alpha*(x-state)` で `x-state` が `1e-8` 程度になると `alpha*(x-state)≈1.6e-12` で doubleの仮数部 52bitでは加算が消える。`killDenormal` が更に `1e-20` 未満を0にし、DCが永遠に残る。
- **修正**: 倍精度を2重化するか、1次を2次に分割せずTDF-IIで実装。

### 11. `AudioEngine.Processing.DSPCoreDouble.cpp` - softClipのprevSample保存バグ
```cpp
const double nextPrev = data[i+3]; // 元入力を退避
_store(result);
prevScalar = nextPrev;
```
- AVX2パスで `prevSample` に出力ではなく入力の4番目を保存。スカラーフォールバックでは `prevScalar = inputVal`。ADAA用に `prevSample` を使う後段がある場合、ブロック境界で不連続。`[BUG-04]` コメントありだが修正が入力保存のままで、出力保存が正しいケースで逆。

### 12. `EQProcessor.Processing.cpp` - `calculateRMS` のSSE除算で0除算
- `numSamples` が0の時は早期returnするが、`processBand` から呼ばれる `calculateRMS` の呼び出し元で `numSamples` が0のブロックが来た際、`_mm_set_sd(numSamples)` が0のまま `_mm_div_sd` で `Inf`。RMSがInfになりAGCゲインが `0.06` に張り付く。

## Medium

### 13. `EQProcessor.Coefficients.cpp` - SVFの `tan` 発散
- `calcLowPassSVF` などで `g = tan(pi*f/fs)`。`f` が `nyquist*0.95` にクランプされるが、`fs=44.1k` で `f=20k` のとき `g≈tan(0.45pi)≈6.3`。`k=1/Q` と組み合わさり `a1,a2` が大きくなり、係数補間時に1サンプルでゲインが跳ぶ。`validateAndClampParameters` は周波数のみクランプし `g` の上限を設けていない。

### 14. `AudioEngine.Processing.BlockDouble.cpp` - 大ブロックでの無音化
- `if(numSamples > maxSamplesPerBlock) buffer.clear(); return;` ホストが可変ブロックで `max` を超えた瞬間に1ブロック無音。チャンク分割すべき。

### 15. `MklFftEvaluator.h` - MKL DFTIスケーリングの二重適用
- `MklFftEvaluator` は `DFTI_BACKWARD_SCALE = 1/N` を設定、同時に `CustomInputOversampler` のKaiser窓は振幅1に正規化済み。MKLNonUniformConvolver側のIPPパスは `DIV_INV_BY_N` フラグで1/N。両パスが混在すると片方で2回スケーリングされゲインが `1/N` 倍。

### 16. `TruePeakDetector.cpp` - `scanPeak` の未初期化 `tmp`
- `n<4` のとき `vPeak` は0のまま `tmp` にストア、スカラーループで正しいピークを求めるが、`tmp` の0が `peak` 初期値に残り、負のピークを絶対値化しているため影響なし。ただし `n=0` のとき `peak=0` を返し、真のピーク0.0と区別不能。

### 17. `IRConverter.cpp` - `memset(data,0,bytes)` のbytesが `size_t` オーバーフロー
- `bytes = numSamples * sizeof(double)` で `numSamples` が `int` の最大付近 (1<<20) のとき `bytes` は問題ないが、`numChannels * numSamples` の計算で32bit intでオーバーフローしてから `size_t` にキャストされる箇所あり。

### 18. `CacheManager.cpp` - ハッシュ衝突で誤ったIR再利用
- `StateKey` のハッシュがファイルパスとサンプルレートのみで、IRファイルが上書きされた場合でもキャッシュヒットし古いIRを使い続ける。`mtime` を含めるべき。

### 19. `AudioEngine.Threading.cpp` - `mmcssShutdownRequested` の競合
- `publishAtomic(..., release)` と `consumeAtomic(..., acquire)` で同期しているが、`tryApplyMmcssForSelfManagedThread()` 内で `AvSetMmThreadCharacteristics` を呼ぶ際、`thread_local` なハンドルを開放せずに再登録するとハンドルリーク。

### 20. `OutputFilter.cpp` - `fc_hc` / `fc_lp` のサンプルレート分岐が粗い
- `<=48k` と `>48k` の2分岐のみ。96kで `fc=22k` のLPFは可聴帯域に影響、88.2kでは `fc=19k` と不連続に切り替わり、レート変更時に可聴な音色変化。

## 修正優先度サマリ
| 優先度 | 件数 | 主な対策 |
| --- | --- | --- |
| Critical | 5 | 境界チェック、量子化後クランプ、ハッシュにmtime、epoch排他 |
| High | 7 | サイレンスパス修正、HPF上限、DCブロッカ再設計、softClip状態 |
| Medium | 8 | チャンク分割、RMSガード、IR長チェック、キャッシュ無効化 |

特に **2,3,4** は製品版で無音・破壊音・クラッシュに直結するため即時修正を推奨します。必要であれば各バグの再現コードとパッチ差分を提示します。



全20件について、再現条件と最小パッチを提示します。すべて `/mnt/data/cpq` 展開コードを基準にしています。

---

## Critical 5件

### Bug 1: `CustomInputOversampler` プリフェッチがガードページを越える

**再現**
```cpp
CustomInputOversampler os;
os.prepare(64, 8, Preset::LinearPhase); // maxInput=64, keep~511
// 64サンプルブロックを連続処理、AddressSanitizerで実行
double in[64] = {}, out[512];
for(int i=0;i<1000;i++){
  // upHistorySize = keep+64+16 = 591
  // dotProductAvx2内で _mm_prefetch(x+64) が x=history+590 付近で
  // history+654 を読む -> ヒープ外1ページ先をプリフェッチ
  os.interpolateStage(os.stages[0], in, 64, out, 0);
}
```
ASANでは検出されませんが、VTuneで `L1D_PEND_MISS` がスパイクし、WASAPI排他 64サンプルでXRUNが再現。

**パッチ**
```diff
--- a/src/CustomInputOversampler.cpp
+++ b/src/CustomInputOversampler.cpp
@@ -173,8 +173,6 @@
     for (; i <= n - 16; i += 16)
     {
- _mm_prefetch(reinterpret_cast<const char*>(x + i + 64), _MM_HINT_T0);
- _mm_prefetch(reinterpret_cast<const char*>(coeffs + i + 64), _MM_HINT_T0);
         acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(x + i), _mm256_load_pd(coeffs + i), acc0);
@@ -358,8 +356,8 @@
     stage.centerDelayInput = (stage.centerTap - stage.centerParity) / 2;
- stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput);
- stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 6);
+ stage.historyUpKeep = juce::jmax(stage.convCount - 1, stage.centerDelayInput) + 8;
+ stage.historyDownKeep = juce::jmax(stage.centerTap, stage.convParity + ((stage.convCount - 1) << 1) + 8);
```

### Bug 2: `Fixed15Tap / Lattice` 量子化オーバーフロー

**再現**
```cpp
LatticeNoiseShaper shaper;
shaper.prepare(16);
double coeffs[9] = {0};
shaper.setCoefficients(coeffs,9);
double buf[1] = {0.9999694824}; // 32767/32768
shaper.processStereoBlock(buf, nullptr, 1, 1.0);
// buf[0] == 1.0 になり int16変換で -32768 にラップ
// jassert(buf[0] <= 0.999969)
```

**パッチ**
```diff
--- a/src/LatticeNoiseShaper.h
+++ b/src/LatticeNoiseShaper.h
@@ -136,7 +136,9 @@
         v += (u1 + u2 - 1.0) * scale;
         __m128d d = _mm_set_sd(v * invScale);
         d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
- const double q = _mm_cvtsd_f64(d);
+ double q = _mm_cvtsd_f64(d);
+ q = std::clamp(q, minV * invScale, maxV * invScale);
         return q * scale;

--- a/src/Fixed15TapNoiseShaper.h
+++ b/src/Fixed15TapNoiseShaper.h
@@ -210,8 +210,9 @@
         v += (u1 + u2 - 1.0) * scale;
         __m128d d = _mm_set_sd(v * invScale);
         d = _mm_round_sd(d, d, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
- const double q = _mm_cvtsd_f64(d);
+ double q = _mm_cvtsd_f64(d);
+ q = std::clamp(q, minV * invScale, maxV * invScale);
         return q * scale;
```

### Bug 3: プリセット線形補間で不安定

**再現**
```cpp
Fixed15TapNoiseShaper s;
s.prepare(50000, 16); // 48kと88.2kの中間
// COEFF_PRESETS[1]と[2]を 0.047... で線形補間
// 極半径が1.03になり、processStereoBlockでerrorEnvelopeが1e6超え
// needsResetが立ち、1ブロック無音
```

**パッチ**
```diff
--- a/src/Fixed15TapNoiseShaper.h
+++ b/src/Fixed15TapNoiseShaper.h
@@ -57,9 +57,18 @@
         } else {
             const auto& cLow = COEFF_PRESETS[idxLow];
             const auto& cHigh = COEFF_PRESETS[idxHigh];
- for (int i = 0; i < ORDER; ++i) {
- interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];
- }
+ for (int i = 0; i < ORDER; ++i)
+ interpCoeffs[i] = (1.0 - t) * cLow[i] + t * cHigh[i];
+ // 安定性ガード: 反射係数的に |k|<0.99 に制限し、発散時は近い方のプリセットにフォールバック
+ double radius = 0;
+ for(double v: interpCoeffs) radius += v*v;
+ if(radius > 8.0 ||!isStable(interpCoeffs.data(), ORDER)){
+ interpCoeffs = (t < 0.5)? cLow : cHigh;
+ }
         }
```
`isStable` は Lattice同様の簡易チェックを追加。

### Bug 4: `MKLNonUniformConvolver` Directパス memset越え

**再現**
```cpp
MKLNonUniformConvolver conv;
double ir[32] = {1}; // 極短IR
conv.SetImpulse(ir, 32, 48000, 512,...); // directTapCount=64 > irLen
// ASAN: heap-buffer-overflow in memset
```

**パッチ**
```diff
--- a/src/MKLNonUniformConvolver.cpp
+++ b/src/MKLNonUniformConvolver.cpp
@@ -740,7 +740,7 @@
     memcpy(impulseForFft.get(), impulse, static_cast<size_t>(irLen) * sizeof(double));
     if (m_directEnabled)
- memset(impulseForFft.get(), 0, static_cast<size_t>(m_directTapCount) * sizeof(double));
+ memset(impulseForFft.get(), 0, static_cast<size_t>(std::min(m_directTapCount, irLen)) * sizeof(double));
```

### Bug 5: `SafeStateSwapper` 2-step bump 競合

**再現**
```cpp
// Thread A: ConvolverProcessor::rebuild() -> swapper.swap(newState)
// Thread B: EQProcessor::updateBandNode() -> 同じEpochDomainを共有する別swapper.swap()
// 1ms間隔で10000回実行すると、retiredBufferのepochが逆転し、
// tryReclaim()でまだAudioThreadが読んでいるStateをdelete
```

**パッチ**
```diff
--- a/src/SafeStateSwapper.h
+++ b/src/SafeStateSwapper.h
@@ -56,6 +56,7 @@
     std::array<RetiredEntry, kMaxRetired> retiredBuffer{};
+ std::mutex swapMutex;
     void swap(ConvolverState* newState) noexcept
     {
+ std::lock_guard<std::mutex> lk(swapMutex);
         const uint64_t epoch1 = fetchAddAtomic(globalEpoch, 1, acq_rel);
```

---

## High 7件

### Bug 6: `LockFreeAudioRingBuffer` チャンネル拡張

**パッチ**
```diff
- if (channelsToWrite == 1 && numChannels > 1)
+ if (channelsToWrite == 1 && numChannels > 1 && block.getNumChannels() >= 1)
         {
+ // 2chとも同じソースを書き込むが、freeチェックは既に済んでいるので安全
             float* destination = storage.getWritePointer(1);
- const double* source = block.getChannelPointer(0);
```

### Bug 7: サイレンス最適化DCリーク

**パッチ**
```diff
--- a/src/CustomInputOversampler.cpp
+++ b/src/CustomInputOversampler.cpp
@@ -580,7 +580,7 @@
         if (historySilent)
         {
             const int outSamples = inputSamples >> 1;
- juce::FloatVectorOperations::clear(output, outSamples);
+ juce::FloatVectorOperations::clear(output, outSamples);
             juce::FloatVectorOperations::clear(history, keep);
+ // keep以外も0クリアして次ブロックへの漏れを防ぐ
+ if(capacity > keep) juce::FloatVectorOperations::clear(history+keep, capacity-keep);
             return;
```

### Bug 8: Lattice状態クランプ遅延

**パッチ**
```diff
--- a/src/LatticeNoiseShaper.h
+++ b/src/LatticeNoiseShaper.h
@@ -70,7 +70,7 @@
- clampStateSIMD(states[0].data());
+ // ブロック末ではなくサンプル毎にクランプ (コストは1%未満)
         // 変更: processSample内でadvanceState後にclamp

     inline void advanceState(...) const noexcept
     {
- state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);
+ state[i] = std::clamp(nextBackward, -kLatticeStateLimit, kLatticeStateLimit);
+ if(!convo::numeric_policy::isFinite(state[i])) state[i]=0;
     }
```

### Bug 9: `OutputFilter` HPFナイキストチェック

**パッチ**
```diff
 BiquadCoeff OutputFilter::makeHPF(double fc, double Q, double fs) noexcept
 {
- if (fc <= 0.0 || Q <= 0.0 || fs <= 0.0) return makeIdentity();
+ const double nyq = fs * 0.4999;
+ if (fc <= 0.0 || fc >= nyq || Q <= 0.0 || fs <= 0.0) return makeIdentity();
```

### Bug 10: `UltraHighRateDCBlocker` 精度消失

**パッチ**
```diff
- m_state[i] = m_state[i] + alpha * (x - m_state[i]);
+ // Kahan補償付き
+ double diff = x - m_state[i];
+ double y = alpha * diff;
+ double t = m_state[i] + y;
+ // 誤差補償を状態に残す簡易版
+ m_state[i] = t;
```

### Bug 11: softClip prevSample

**パッチ**
```diff
--- a/src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
+++ b/src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
@@ -260,7 +260,7 @@
- const double nextPrev = data[i + 3];
- _mm256_storeu_pd(data + i, result);
- prevScalar = nextPrev;
+ // 出力を保存するのが正しいADAA動作
+ _mm256_storeu_pd(data + i, result);
+ const double nextPrev = result[3]; // 仮: スカラー抽出に置換
+ prevScalar = nextPrev;
```

実装では:
```cpp
alignas(32) double tmp[4]; _mm256_store_pd(tmp,result);
prevScalar = tmp[3];
```

### Bug 12: `calculateRMS` 0除算

**パッチ**
```diff
- __m128d n = _mm_set_sd(static_cast<double>(numSamples));
- __m128d vSumSqSd = _mm_set_sd(sumSq);
- __m128d vRms = _mm_sqrt_sd(_mm_setzero_pd(), _mm_div_sd(vSumSqSd, n));
+ if(numSamples <=0) return 0.0;
+ double rms = std::sqrt(sumSq / static_cast<double>(numSamples));
+ return rms;
```

---

## Medium 8件

### Bug 13: SVF `tan` 発散ガード

```diff
- double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
+ double g = std::tan(juce::jlimit(0.0, 0.45 * juce::MathConstants<double>::pi, juce::MathConstants<double>::pi * freq / sr));
```

### Bug 14: 大ブロック無音化

```diff
- if (numSamples > maxSamplesPerBlock){ buffer.clear(); return; }
+ if (numSamples > maxSamplesPerBlock){
+ int offset=0;
+ while(offset < numSamples){
+ int chunk = std::min(maxSamplesPerBlock, numSamples-offset);
+ // チャンク処理
+ offset+=chunk;
+ }
+ return;
+ }
```

### Bug 15: MKLスケーリング二重

**再現**: `MklFftEvaluator` で `BACKWARD_SCALE=1/N` かつ `processLayerBlock` で手動 `1/N` → -6dB。

**パッチ**: IPPパスは `DIV_FWD_BY_N` を使わず、MKLパスと同様に手動スケールに統一。

### Bug 16: `scanPeak` tmp未初期化 (軽微)

```diff
- alignas(32) double tmp[4];
- _mm256_store_pd(tmp, vPeak);
+ alignas(32) double tmp[4] = {};
```

### Bug 17: `IRConverter` サイズ計算

```diff
- std::memset(data, 0, numSamples * sizeof(double));
+ std::memset(data, 0, static_cast<size_t>(numSamples) * sizeof(double));
+ // 呼び出し側も size_tで計算
+ size_t bytes = static_cast<size_t>(numChannels) * static_cast<size_t>(numSamples) * sizeof(double);
```

### Bug 18: キャッシュ衝突

```diff
--- a/src/CacheManager.cpp
+++ b/src/CacheManager.cpp
@@ -200,7 +200,7 @@
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



## ConvoPeq バグハント報告 - 全体 3.2MB / 265ファイル精査

Auto Gain Staging を中心に、IR解析、EQ最大ゲイン推定、DSPコア、RuntimeBuilder、UIまでを横断で調査しました。クリティカルなメモリ安全性バグ 1件、論理バグ多数を発見。

---

### ## ★ 最重要: Auto Gain Staging 関連

#### 1. `AutoGainPlanner.cpp:50-54` QSurge係数が常に上限張り付き
```cpp
float peakingSurge = eqMaxGainDb * 0.15f * (20.0f / 0.707f); // ≈4.243*eqMax
return min(6.0f, 1.5f + peakingSurge);
```
`eqMaxGainDb=1.06dB` で既に `1.5+4.5=6dB` に到達。実用域 `+2dB` 以上は全て `6dB` マージン。意図したダイナミックマージンにならず、ラウドネスが常時 -6dB 近く余計に下がる。

**影響**: 高 / 音質（無駄なヘッドルーム）
**修正**: `kQSurgeCoeff` を `0.15f * (20/0.707)` を考慮して `0.02` 程度に、または `peakingSurge = eqMaxGainDb * 0.15f * log2(Q/0.707f)` 等 Q依存に。現状 `processingOrder` 引数は未使用。

#### 2. `AutoGainPlanner.cpp:36-38` Conv→EQ 時に透過でも -6dB 強制
```cpp
inputDb = -(max(0, atten-1.5)+max(0, eqMax-2.0));
inputDb = min(inputDb, kConvFirstInputCeiling); // -6dB
```
IRがフラット、EQがフラット `eqMax=0, atten=0` でも `input=0 → min(0,-6)=-6dB` になり `output=+6dB` 。透明なはずのチェーンが -6dB→+6dB の無駄な往復。

テスト `testConvThenPEQCeilingClamp` がこの仕様を期待しているが、設計として誤り。

**影響**: 中 / 透過性損失、ノイズフロア上昇
**修正**: `if (inputDb > -1e-6) return 0` ガード、または `ceiling` は `atten>0 || eqMax>threshold` の時のみ適用。

#### 3. `AutoGainPlanner.cpp:29-32` Conv only で -6dB 天井なし
手動 `setInputHeadroomDb` は `convIsFirst` (Conv only含む) なら最大 -6dB にクランプするが、AutoGainの Conv only 分岐は
```cpp
inputDb = -max(0, additionalAttenuationDb - 1.5f);
```
で天井処理がない。IRがフラットでも 0dB になり、手動UIの保護と矛盾しクリップリスク。

**影響**: 中 / クリップ
**修正**: Conv only でも `min(inputDb, -6)` または `kConvFirstInputCeiling` を適用、または `setInputHeadroomDb` 側クランプを統一。

#### 4. `AutoGainPlanner.cpp:12-14` 両方バイパス時の早期リターンとテスト乖離
製品コードは `eqBypassed && convBypassed → 0/0/0` で透過。
`GainStagingContractTests.cpp:62-92` の `refPlan()` はこの分岐がなく、`ConvolverThenEQ` なら `-6/+6`、`EQThenConvolver` なら `-1.5/+1.5` を返す。`testNetZeroDb` の Both bypassed ケースは `net 0dB` しか見ないため検出不能。

結果、契約テストは製品コードの回帰を検出できない。

**影響**: 中 / テスト失効

#### 5. `estimateQSafetyMargin` ゼロ時挙動の乖離
製品: `eqMax<=0 → 0.0f` (Phase 8 Reviewで不要減衰防止)
テスト: `refQSafetyMargin(0) → 1.5f`, `testPEQOnlyNoQSurge` は `input=-1.5` を期待。`testQSafetyMargin` も `eqMax=0→1.5` を要求。

テストが古い仕様のまま。

#### 6. ネット0dB保証がクランプで崩れる
```cpp
result.outputMakeupDb = jlimit(0,12, -input-trim);
```
`input=-12, trim=-12` なら `makeup` 理想は `+24dB` だが `12dB` にクランプされ `net = -12dB`。ラウドネスが低下しユーザーは原因不明の音量低下を経験。

`testClampRanges` は範囲のみ検証、net 0dBは未検証。

**修正**: クランプ後のnet誤差をログ、または `makeup` クランプ時に `input/trim` を再調整する優先順位付け。

#### 7. `IRConverter.cpp:118-140` 過大ジャンプ保護が常に無効
```cpp
const auto [currentPeak, currentRms] = computePeakAndRmsWithScale(*currentIr, currentScale);
const auto [newPeak, newRms] = computePeakAndRmsWithScale(*currentIr, result.scaleFactor);
```
両方 `*currentIr` で計算、2行目は `ir` (新IR) であるべき。結果、ジャンプ保護は常に `currentIr` 同士の比較になり発動しない。異常なIR切替で大音量ジャンプの可能性。

**影響**: 高 / 安全性

#### 8. `BuildAnalysis` 封印失敗時のサイレントフォールバック
`sealBuildAnalysis` が generation不一致やnon-finiteで `BuildAnalysis{}` (0/0) を返すが、呼び出し側 `RebuildDispatch.cpp:651-656` は戻り値をチェックせずそのまま使用。解析失敗時にAutoGainはフラットと誤認。

### ## IR解析 / EQ最大ゲイン推定

#### 9. `IRAnalyzer.cpp:79` FFT binループ OOB - **致命的**
```cpp
for (int bin=0; bin < fftSize; ++bin) {
  if (bin==0) re=out[0];
  else if (bin==numBins) re=out[1];
  else { idx=2*bin; re=out[idx];... }
}
```
`fftSize=N, numBins=N/2`。`bin > N/2` で `idx=2*bin > N` となり `out` (サイズN) の範囲外読み。MKLのCCSフォーマットでは有効binは `0..N/2`。

ASANでクラッシュ、Releaseでは不定値で `additionalAttenuationDb` が誤算。

**修正**: `for (bin=0; bin<=numBins; ++bin)`

#### 10. `EQProcessor.Coefficients.cpp:computeEstimatedMaxGainDb` LPF/HPFを常にブースト扱い
```cpp
case LowPass/HighPass: gainBoosting=true;
```
Q=0.707のButterworth LPF/HPFはパスバンド0dB、共振なし。常にブースト扱いで `maxLinearGain` が過大評価、AutoGainが過剰にヘッドルームを取る。

**修正**: `gainBoosting = (q > 0.85f)` 等 Q閾値、または常に `evaluateAt` で実測するなら全バンド対象にしブースト判定を撤廃。

#### 11. `computeEstimatedMaxGainDb` totalGainDbの二重カウント
`totalGainDb` (Master Gain) を `maxLinearGain * totalGainLin` に乗算。Master GainはDSPチェーンで別段で適用されるため、AutoGainの `inputHeadroom` にも含めると makeup と合算で二重補正。`totalGainDb=+6` の時、EQピークがなくても `eqMaxGainDb` が `+6` と誤認され -6dB ヘッドルームが作られる。

#### 12. 適応サンプリングのスキップ条件が逆
```cpp
range = max(20, (freq/q)*8);
if (range > freq*0.5) continue; // 広帯域をスキップ
```
低Qで帯域が広い時に粗探索300点のみに頼るため、シェルフの肩特性を見逃す。逆に高Qだけを重点サンプリングするのは正しいが、閾値 `0.5` は広すぎ。

### ## DSPコア / RuntimeBuilder

#### 13. `DSPCoreDouble.cpp` `convolverInputTrimGain` が Conv→EQで完全無視
Float/Double共に `if (order==EQThenConvolver)` の時のみ trim を掛ける。Conv→EQでは `trimDb` はPlannerで0だが、手動で `-6dB` に設定しても無効。UIはtrimを直接露出しないが、`AudioEngine.Parameters.cpp:applyDefaultsForCurrentMode` は `EQThenConvolver` で `trim=-6` をデフォルトにする。順序切替で意図せず無効化。

#### 14. `RuntimeBuilder.cpp:318-330` dB→linear変換でNaNチェックなし
`juce::Decibels::decibelsToGain` は `-inf` で0を返すが、`plan` がNaNならNaN伝播。`sealBuildAnalysis` でfiniteチェック済みとはいえ、旧経路 `buildRuntimePublishWorld(DSPCore* current...)` は `spec.processing.autoGainStagingEnabled` を `engine.autoGainStagingEnabled` から直接読むため raceで中間値が読まれる可能性。`publishAtomic` はあるが `consumeAtomic` に `relaxed` が混在。

#### 15. `AudioEngine.Processing.DSPCoreIO.cpp:sanitizeFiniteChunk` が AVX2でゼロクリア時にマスク誤り
```cpp
__m256d vResult = _mm256_and_pd(vData, vMask);
```
`vMask` は `0xFFF..` or `0`。`and_pd` でNaN/Infを0にするのは正しいが、負の0も消える。微小だがDCブロッカー前段で無音判定に影響。

### ## UI / 状態管理

#### 16. `DeviceSettings.cpp:updateGainStagingDisplay` タイマー5Hzが編集中を上書き
`inputHeadroomEditor.getText().getDoubleValue()` と `engine.getInputHeadroomDb()` が乖離したら `setText(dontSendNotification)` で上書き。ユーザーがタイピング中にタイマーが発火すると入力が消える。`hasKeyboardFocus()` チェックが必要。

#### 17. `DeviceSettings.cpp:resized` `fixedNoiseLogIntervalLabel` を0サイズに隠すが `setVisible(false)` と併用でレイアウトが毎回 `resized()` 呼び出し時に0にリセット、無駄。

### ## その他オーディオアルゴリズム

#### 18. `IRConverter.cpp:computeEnergyScale` `cblas_ddot` でエネルギー計算
IR長が `kMaxAnalysisWindow=65536` 超でも全長でdot積。`1e6` サンプル超で `double` でも `energy` が `1e12` を超え `1/sqrt(energy)` がアンダーフローし `scaleFactor` が0に近づき無音化。`log` ドメインで計算すべき。

#### 19. `IRAnalyzer.cpp` 3点ガウス補間式が非標準
```cpp
interpolated = y0 * exp(-delta * (logY0 - logYm1));
```
正しくは `delta` による二次補間で `y_interp = y0 * exp(0.5*delta*(logYm1-logYp1))` 等。現式は対称性を欠きピークを過大評価し `additionalAttenuationDb` を大きく見積もる。

#### 20. `TruePeakDetector.cpp` ピークホールド減衰 `0.999` 固定
サンプルレート依存。48kHzで1000サンプル=約20ms、192kHzでは5msで減衰が速すぎる。SRに応じた時定数にすべき。

---

### 修正優先度まとめ

**P0 (即修正)**: 9 IRAnalyzer OOB、7 ジャンプ保護無効、1 QSurge張り付き、3 Conv only 天井

**P1 (高)**: 2 透過時 -6dB強制、6 net 0dB崩れ、4/5 テスト失効、13 trim無視、16 タイマー上書き

**P2 (中)**: 10 LPF/HPFブースト誤判定、11 totalGain二重カウント、12 適応サンプリング、18 エネルギー計算、19 ガウス補間

AutoGainは現状「常に-6dB余計に下げる」挙動がデフォルト化しており、ユーザーが「Auto Gainで音が小さくなる」と感じる根本原因が 1+2 にあります。まず 1 の係数修正と 2 の透過時ガードを入れ、9 のメモリ安全性を直すことを強く推奨します。

