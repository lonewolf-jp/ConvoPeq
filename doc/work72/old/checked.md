# `gain_design_spec.md` 詳細検証結果

コードベース 254ファイル照合の結果、設計書は大筋で正しいですが **7件の重大な論理バグと10件以上の実装不整合** があります。放置するとビルドエラー、ゲイン計算の符号反転、プリセット復元時の無効化、リビルドストームが発生します。

## 0. 重大バグ Critical - 必ず修正

### C-1. `DspNumericPolicy.h` 循環 include
**現状:** `EQProcessor.h` は `DspNumericPolicy.h` を include。設計書は `DspNumericPolicy.h` に `getComplexResponse()` 宣言を追加し、`DspNumericPolicy.cpp` で `#include "eqprocessor/EQProcessor.h"` すると書いています。ヘッダ側で `EQCoeffsBiquad` を使うために `EQProcessor.h` を include すると循環になります。

**修正:**
```cpp
// DspNumericPolicy.h には include しない
struct EQCoeffsBiquad; // 前方宣言
namespace convo::numeric_policy {
  [[nodiscard]] std::complex<double> getComplexResponse(const EQCoeffsBiquad& c, double omega) noexcept;
}
```
cpp側でのみ `#include "eqprocessor/EQProcessor.h"` する。`EQCoeffsBiquad` はグローバル名前空間なので `::EQCoeffsBiquad` でも可。

### C-2. `residualRiskDb` の符号反転
`IRConverter.cpp` の抜粋:
```cpp
double peakClipDb = 0.0; // 実装では常に0
result.residualRiskDb = peakClipDb + rmsClipDb + freqClipDb; // 負値
```
`recomputeAutoGainStaging()` では
```cpp
newInputDb = -max(0, irResidualDb - 1.5f);
```
`irResidualDb` が -6dB なら `max(0,-7.5)=0` で保護が働かない。**安全側に倒れるはずが無防備になる。**

**修正:** クランプ量を正の減衰量として保存する。
```cpp
double peakAttenDb = 0.0, rmsAttenDb = 0.0, freqAttenDb = 0.0;
if (freqRespGain > kMaxEffectiveFreqResponse) {
  double clip = kMaxEffectiveFreqResponse / freqRespGain; // <1
  freqAttenDb = -20.0*log10(clip); // 正値
  result.scaleFactor *= clip;
}
// peak/rmsも同様に
result.residualRiskDb = peakAttenDb + rmsAttenDb + freqAttenDb; // 正値 0..+inf
```
あるいは保存時に `-` を付ける: `residualRiskDb = -(peakClipDb+...)`。

### C-3. `computeScaleFactor` の peak/rms クランプ量が記録されていない
既存 `computeScaleFactor` は `absoluteClamp` を `min(peakClamp, rmsClamp)` で一括計算しているため、どちらが効いたか分からない。設計書の `peakClipDb=0` はプレースホルダで、実際には未実装。

**修正:** 分離して計算
```cpp
double peakClamp = 1.0, rmsClamp = 1.0;
if (irPeak > 1e-12) peakClamp = kMaxEffectivePeak / (irPeak*scale);
if (irRms > 1e-12) rmsClamp = kMaxEffectiveRms / (irRms*scale);
double finalClamp = std::min({1.0, peakClamp, rmsClamp});
peakAttenDb = (peakClamp < 1.0)? -20*log10(peakClamp) : 0;
rmsAttenDb = (rmsClamp < 1.0)? -20*log10(rmsClamp) : 0;
```

### C-4. `estimateMaxFrequencyResponseGain` の窓による振幅過小評価
Tukey α=0.5 の平均値は約0.75。矩形窓でないため FFT の DC 成分は `0.75*N` になり、周波数応答ピークを **-2.5dB 過小評価**。ゲイン補正で安全側を狙う設計なのに逆効果。

MKL DFTI の前方変換は無規格化なので、窓を掛けるとそのまま減衰します。側波帯 -40dB を満たすための窓と、ピーク推定は目的が違います。

**修正案2つ:**
1. **推定には矩形窓を使う** (最も安全、過大評価になる)
2. 窓を使うならコヒーレントゲインで補正:
```cpp
double windowMean = 0;
for(n) windowMean += tukey[n];
windowMean /= N;
maxMag /= windowMean; // 補正
```
設計書の UT-05 は窓関数自体のテストであり、ゲイン推定と分離すべきです。

また `sampleRate` 引数は不要。`AudioBuffer<double>` の周波数応答は `sampleRate` に依存しないため、引数を削除するか `[[maybe_unused]]` に。

### C-5. `DeviceSettings::loadSettings` で `recompute` が永久に無効
現コード:
```cpp
engine.beginBulkParameterRestore();
BulkRestoreGuard guard{engine};
if(xml) {... engine.setConvolverStateTree(...); return; } // guard破棄前にreturn
// guardのデストラクタで m_isRestoringState=false になるのは return 後
```
設計書通り `return` 直前に `engine.recomputeAutoGainStaging()` を呼んでも `m_isRestoringState==true` で早期リターンします。

**修正:** スコープを分ける
```cpp
void DeviceSettings::loadSettings(...) {
  loadNoiseShaperState(engine);
  {
    BulkRestoreGuard g{engine};
    if(file.exists()) { /* load */ }
    else { /* default */ }
  } // ここで m_isRestoringState=false
  engine.recomputeAutoGainStaging(); // 有効
}
```
早期 `return` を廃止し、if-else化が必要です。

### C-6. `setProcessingOrder` と `recompute` によるリビルドストーム
`setProcessingOrder` 現状: `submit` → `applyDefaults(内部でsubmit)` で2回。
新設計: `publish` → `applyDefaults(submit)` → `recompute→setInputHeadroomDb(submit)→setOutputMakeupDb(submit)→setTrim(submit)` で最大4回 submit。

`submitRebuildIntent` はワーカーにキューイングされ、短時間に4回発火すると CPU スパイクと診断ログ汚染。

**修正:** `recomputeAutoGainStaging()` は public setter を呼ばず、直接 atomic に publish し最後に1回だけ submit。
```cpp
void AudioEngine::recomputeAutoGainStaging() {
  //...計算...
  float clampedInput = juce::jlimit(-12.f, maxDb, newInputDb);
  // publishAtomic(inputHeadroomDb, clampedInput,...)
  // publishAtomic(outputMakeupDb,...)
  //...
  submitRebuildIntent(...); // 1回のみ
  sendChangeMessage();
}
```

### C-7. `Convolver first` 時の net -6dB ドロップ
`input` 上限 -6dB クランプを `setInputHeadroomDb` が行うが、`recompute` はクランプ前の値で `makeup = -input` を計算。結果 `input=-6, makeup=0` で net -6dB。設計書は「安全側なので許容」とするが、ユーザーは「Autoで音が小さくなった」と錯覚します。

**修正:** クランプ後の値で makeup を再計算
```cpp
float rawInput = -max(0,ir-1.5)-max(0,eq-2);
float clampedInput = std::min(rawInput, convIsFirst? -6.f : 0.f);
clampedInput = std::max(clampedInput, -12.f);
float makeup = -clampedInput - clampedTrim; // クランプ後で計算
```

## 1. 設計不整合 Major

### M-1. `ConvolverProcessor::IRState` と `currentResidualRiskDb`
`IRState` は `std::atomic<IRState*>` で RCU 管理されています。`double residualRiskDb` を追加すると trivially copyable ではなくなる恐れはありませんが、`IRState` 自体は `new` されるため問題なし。ただし `std::atomic<double>` は MSVC では lock-free ですが、`std::atomic<double>::is_always_lock_free` は保証されません。`static_assert` を追加するか `std::atomic<float>` を使用してください。

また `applyComputedIR` は `ConvolverIRPayload = PreparedIRState` の別名です。`prepared->residualRiskDb` を保存するなら `currentIRState` の更新と同時に `currentResidualRiskDb` も更新する必要がありますが、2つの atomic 間に一貫性がありません。`IRState` に直接持たせて `acquireIRState()` 経由で読む方が整合します。

### M-2. `EQProcessor::computeMaxGainDb` の M/S 評価誤り
設計書: `L = M+S, R = M-S` で評価。実装コードの Mid/Side 処理は `M = (L+R)*0.5` でエンコードし、Mid のみ処理して `L=M+Sorig` にデコードします。単純な `L=M+S` は振幅を2倍に過大評価します。

正しい最大利得は `max(|Hmid|, |Hside|)` です。詳細はコード解析で確認済み。Stereo/Left/Right とのカスケードも考慮すると:

```cpp
auto Hst = prod(Stereo); auto Hl = prod(Left); auto Hr = prod(Right);
auto Hm = prod(Mid); auto Hs = prod(Side);
float Lgain = std::abs(Hst*Hl) * (hasMS? std::max(std::abs(Hm), std::abs(Hs)) : 1.f);
float Rgain = std::abs(Hst*Hr) * (hasMS? std::max(std::abs(Hm), std::abs(Hs)) : 1.f);
maxGain = std::max(Lgain,Rgain);
```

Parallel 構造では `H = 1 + Σ(Hi-1)` で計算すべきです。設計書は Serial のみを想定しており、Parallel 時の過大評価/過小評価が未検討です。安全側に倒すなら Parallel でも Serial 積を上限として使えますが、コメントで明記が必要です。

### M-3. Q Surge Margin の単位
`gain × 0.15 × (Q/0.707)` で `gain` は dB 値。dB に無次元係数を掛けるのは次元的には許容されますが、`gain=0.5dB, Q=10` でも `0.5*0.15*14.1=1.06dB` と小さく、実際の共振は Q が高いほど +10dB 以上跳ねることがあります。ヒューリスティックであることを明記し、Phase 8 で実測較正する方針は妥当ですが、コードに `// TODO: empirical calibration` を残してください。

### M-4. UI `onTextChange` 上書き
`DeviceSettings.cpp` 既存:
```cpp
inputHeadroomEditor.onTextChange = [this]{ double v=...; engine.setInputHeadroomDb(v); };
```
設計書は
```cpp
inputHeadroomEditor.onTextChange = [this]{ if(auto) disableAuto(); };
```
と上書きし、元の `setInputHeadroomDb` 呼び出しが消えます。結果手動入力が反映されなくなります。

**修正:** チェーンする
```cpp
auto old = std::move(inputHeadroomEditor.onTextChange);
inputHeadroomEditor.onTextChange = [this, old=std::move(old)]{
  if(audioEngine.isAutoGainStagingEnabled()){
    audioEngine.setAutoGainStagingEnabled(false);
    autoGainToggle.setToggleState(false, dontSendNotification);
  }
  if(old) old();
};
```

### M-5. 永続化漏れ
`autoGainStagingEnabled` を `device_settings.xml` に保存/復元する記述がありません。`saveSettings`/`loadSettings` に
```cpp
xml->setAttribute("autoGainStagingEnabled", (int)engine.isAutoGainStagingEnabled());
```
を追加しないと再起動でリセットされます。

## 2. 軽微・ビルド関連 Minor

- **CMake:** `GLOB_RECURSE src/*.cpp` なので `DspNumericPolicy.cpp` は自動で本体内に含まれます。テスト用 `DspNumericPolicyTests` は `src/DspNumericPolicy.cpp` を直接列挙すると二重定義になりませんが、本体とテストで ODR 違反しないようテストは `add_library` ではなく `add_executable` で独立させるのは正しいです。ただし `GainStagingContractTests` は `AudioEngine` をリンクしていないため `computeMaxGainDb` を呼べません。`target_link_libraries(... MKL::MKL)` と `juce` 依存を追加する必要があります。

- **Tukey窓実装:** `α=0.5` の定義 `n < α(N-1)/2` で `cos` 計算時 `2πn/(α(N-1))` は分母が 0 になりませんが、`N=65536` で `α(N-1)/2=16383.75` と非整数のため境界処理に注意。`std::lround` で処理してください。

- **例外安全:** `estimateMaxFrequencyResponseGain` は `noexcept` 指定ですが MKL `DftiCreateDescriptor` は例外を投げずステータスコードを返します。`noexcept` は維持可能ですが、内部で `new` を使うと `std::bad_alloc` が `noexcept` 違反になります。`try/catch` で `return 1.0` にフォールバックするか `noexcept(false)` にするか、ヒープ確保を避けてスタック配列を使う設計にしてください。

- **`FADE_IN_SAMPLES = 2048` (42ms) とゲイン急変:** Auto ON で `input` が 0→-10dB に変わると、新 DSPCore は `FADE_IN` で 0→1 にフェードしますが、旧コアは即停止します。結果 42ms の無音に近いディップが生じます。`CrossfadeAuthority` が `processingOrder` 変化を検出しないのは既知ですが、ゲイン変化を伴う場合はクロスフェードを強制するか、ゲインランプを `LinearRamp` で別途掛けるべきです。将来対応として Appendix B に記載済みなのは良い判断です。

## 3. 推奨修正順序

1. **C-1, C-2, C-3** を先に修正しないとビルドもテストも通りません。
2. **C-5** の `loadSettings` スコープ修正は 10行程度で影響大。
3. **C-6, C-7** は `recomputeAutoGainStaging` の実装を一度書き直すことで同時解決。
4. **M-2** の M/S 最大利得計算は `computeMaxGainDb` 実装時に上記式を採用。
5. UI 系は最後に。

この順序で Phase 1→3→5→7 と進めると、途中でテストが常にグリーンになります。

---

必要であれば、修正済みの `DspNumericPolicy.h/.cpp`, `IRConverter.cpp` の `computeScaleFactor` 完全版、`AudioEngine::recomputeAutoGainStaging` の安全版を提示できます。