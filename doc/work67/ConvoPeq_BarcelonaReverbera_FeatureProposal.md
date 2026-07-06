# ConvoPeq × BarcelonaReverbera 比較分析と機能導入提案書

**作成日**: 2026年7月6日  
**対象**: ConvoPeq への BarcelonaReverbera 由来機能の導入検討

---

## 1. 2 プロジェクトの技術的位置づけ

| 観点 | ConvoPeq | BarcelonaReverbera |
|---|---|---|
| **製品形態** | スタンドアローン Application | VST3/AU プラグイン |
| **畳み込み方式** | Intel MKL/IPP による 3層 NUC（Non-Uniform Convolution） | テンプレートベース非一様分割 FFT + Direct Stage |
| **精度** | 64-bit double 全面 | 32-bit float（内部一部 double） |
| **IRT 処理系** | Phase 変換（Tukey フェード、DC ブロッキング、リサンプル、Phase モード）强大 | IR デコード＋ポスト処理（decay envelope + LPF/HPF）简单だが実用的 |
| **ドライ/ウェット** | equal-power sin 近似平滑（AVX2） | LUT（1024点）＋ smoothParameter |
| **出力フィルタ** | OutputFilter（HC/LC）— Convolver の後段 | LPF+HPF — wet 信号に適用（IR に対し in-place） |
| **IR decay/tail 制御** | Air Absorption（周波数領域 HF 減衰）のみ | **時間領域 envelope shaping**（カット点＋指数平滑テーパ） |
| **パラメータ補間** | LinearRamp + generation counter | **LUT（1024点）＋ expSmoothing** |
| **スレッド分担** | オーディオコールバックスレッド主体（ワーカースレッドは IR ローディングのみ） | FFT 大ステージは DspThread にオフロード |
| **ロックフリープリミティブ** | RCU（EpochDomain）＋ atomic + SPSC LockFreeRingBuffer | `std::atomic` のみ（simple but effective） |
| **SIMD 利用** | AVX2（Mix 平滑）に限定 | PFFFT（SSE/NEON）による FFT 全体 |

---

## 2. ConvoPeq の现有強み（機能充足済み領域）

以下は ConvoPeq がすでに優秀であり、導入不要の領域：

- **3層 NUC エンジン**（L0/L1/L2）— BarcelonaReverbera より高機能
- **Minimum/Mixed/Linear Phase 変換**（AllpassDesigner + CMA-ES）— BarcelonaReverbera にはなし
- **r8brain による高质量リサンプル** — BarcelonaReverbera の JUCE ResamplingAudioSource より高精度
- **AVX2 最適化 Mix 平滑**（equal-power sin + headroom）— BarcelonaReverbera の Taylor 近似より高精度
- **RCU パターンと EpochDomain** — 複雑な並行処理がすでに実装済み
- **DC ブロッキング・Tukey fade-out** — すでに存在（Tukey は対称のみ）
- **Air Absorption（周波数領域 HF 減衰）** — すでに存在
- **UltraHighRateDCBlocker、Noise Shaper、True Peak Detector** — BarcelonaReverbera には存在しない高機能
- **ISR Runtime Governance（101ファイル）** — 运营の坚実性は段違い

---

## 3. ConvoPeq に導入が望ましい機能（優先度順）

---

### 第 1 優先度：IR Decay Envelope（残響エンベロープ整形）

**BarcelonaReverbera の実装**（`ConvolutionReverb::updateIr()`）：

```cpp
// decay カット点を計算
const uint32_t decayCutPointSamples = irLen * m_decayCurrent;
// カット点以降を指数平滑でテーパダウン
for (uint32_t i = decayCutPointSamples; i < irLen; i++) {
    const float decayGainTarget = (i < decayCutPointSamples) ? 1.0f : 0.0f;
    decayGainCurrent = DspUtils::expSmoothing(decayGainTarget, decayGainCurrent, decayEnvSmoothingFactor);
    irPostProcessed[ch][i] = m_irPreProcessed[ch][i] * decayGainCurrent;
}
```

**ConvoPeq への導入意義**：

- ConvoPeq の Air Absorption は**周波数領域**の HF 減衰のみ。時間領域のエンベロープ整形機能はない。
- Decay ノブは残響の「長さ主観」を直接制御するため、ユーザーにとって最も直感的なパラメータ。
- BarcelonaReverbera の手法は実装が简单（時間領域乗算のみ）であり、IR ローディングパイプラインに追加しやすい。

**ConvoPeq への実装場所**：`ConvolverProcessor.LoadPipeline.cpp` — `doTransformStep()` 内に追加ステップとして実装。

**実装メモ（ConvoPeq 出发点）**：

```cpp
// IRConverter への追加メソッド案
void IRConverter::applyDecayEnvelope(double* irLeft, double* irRight,
                                      uint32_t irLen, double samplerate,
                                      float decayControl /* 0.0~1.0 */)
{
    const double decayMin = 0.015;        // BarcelonaReverbera: DECAY_MIN
    const double decayKnobDecades = 2.15; // DECAY_KNOB_DECADES（非線形カーブ用）
    const double envelopePct = 2.3;       // DECAY_ENVELOPE_PERCENTAGE

    // decayControl -> decayRatio 変換（LUT 化推奨）
    const double decayRatio = decayMin
        + (std::pow(10.0, decayKnobDecades * decayControl) - 1.0)
          / (std::pow(10.0, decayKnobDecades) - 1.0) * (1.0 - decayMin);

    const uint32_t decayCutPoint = static_cast<uint32_t>(irLen * decayRatio);

    // テーパダウン時定数（1.5 秒相当を上限）
    const double smoothingTimeSamples = std::min(
        decayCutPoint * envelopePct,
        1.5 * samplerate);
    const double factor = std::exp(-2.2 / smoothingTimeSamples);

    double gainCurrent = 1.0;
    for (uint32_t i = decayCutPoint; i < irLen; ++i) {
        gainCurrent = 0.0 - 0.0 * factor + gainCurrent * factor; // expSmoothing(0, current, factor)
        irLeft[i]  *= gainCurrent;
        irRight[i] *= gainCurrent;
    }
}
```

**期待効果**：IR の残響長をパラメータとして制御可能。カット点より先は指数関数的に減衰し、残響の「自然的衰减」を再現。

---

### 第 2 優先度：Color フィルタ（Wet 信号への LPF/HPF 制御）

**BarcelonaReverbera の実装**：

```cpp
// color <= 0 → LPF mode, color > 0 → HPF mode
const float filterFc = filterIsLowPass
    ? expf((1.0f + m_colorControl) * COLOR_LPF_FREQ_RANGE + COLOR_LPF_FREQ_LOGMIN)
    : expf(m_colorControl * COLOR_HPF_FREQ_RANGE + COLOR_HPF_FREQ_LOGMIN);

// カットオフ平滑（80ms 時定数）
m_filterLPF[ch].clearState();
m_filterLPF[ch].setTargetFreq(filterFc, smoothingFactor, samplerate);
m_filterLPF[ch].process(irPostProcessed[ch], irPostProcessed[ch], irLen);

m_filterHPF[ch].clearState();
m_filterHPF[ch].setTargetFreq(filterFc, smoothingFactor, samplerate);
m_filterHPF[ch].process(irPostProcessed[ch], irPostProcessed[ch], irLen);
```

**ConvoPeq との差別化**：

- ConvoPeq の OutputFilter は Dry+Wet 混合後の出力段에만適用される。**Wet 信号单独的**にフィルターを掛ける機能はない。
- BarcelonaReverbera の Color は IR に対して前処理として適用するため、IR の特性を可变でき、殘響の「明るさ・暗さ」を直接制御可能。
- LPF 範囲：220 Hz〜20 kHz、HPF 範囲：20 Hz〜3 kHz（指数曲線マッピング）。

**ConvoPeq への実装場所**：`ConvolverProcessor.LoadPipeline.cpp` — `applyDecayEnvelope()` の後に適用。

**実装メモ**：

```cpp
// ConvoPeq の FilterBiquad（DF2T、double 精度）を再用
void IRConverter::applyColorFilter(double* irLeft, double* irRight,
                                    uint32_t irLen, double samplerate,
                                    float colorControl /* -1.0~1.0 */)
{
    FilterBiquad lpf, hpf;
    lpf.init(true);    // lowPass = true
    hpf.init(false);   // lowPass = false (HPF)

    const float lpfFreqMin = 220.0f;
    const float lpfFreqMax = 20000.0f;
    const float hpfFreqMin = 20.0f;
    const float hpfFreqMax = 3000.0f;

    const float filterFc = (colorControl <= 0.0f)
        ? std::exp((1.0f + colorControl)
            * (std::log(lpfFreqMax) - std::log(lpfFreqMin)) + std::log(lpfFreqMin))
        : std::exp(colorControl
            * (std::log(hpfFreqMax) - std::log(hpfFreqMin)) + std::log(hpfFreqMin));

    // 時定数 80ms で平滑化（BarcelonaReverbera と同じ）
    const float smoothingFactor = DspUtils::getTimeConstantMs(80.0f, static_cast<float>(samplerate));

    lpf.setTargetFreq(filterFc, smoothingFactor, samplerate);
    hpf.setTargetFreq(filterFc, smoothingFactor, samplerate);

    lpf.process(irLeft,  irLeft,  irLen);  // in-place
    lpf.process(irRight, irRight, irLen);
    hpf.process(irLeft,  irLeft,  irLen);
    hpf.process(irRight, irRight, irLen);
}
```

**期待効果**：IR の周波数特性を変化させることで、殘響のbrightness/darknessを制御。LPF dominant（color<0）なら暗く、Hpf dominant（color>0）なら明るい的感觉に。

---

### 第 3 優先度：LUT ベース・パラメータ補間と SmoothParameter 統一化

**BarcelonaReverbera の手法**：

```cpp
// 1024点 LUT による高速補間（log/exp の計算を回避）
static float m_arrayColorLpfFcInterp[BCNRVRB_PARAM_INTERPOL_ARRAY_LEN];
const float index = posLinear * (1024 - 1);
const uint32_t indexInt = uint32_t(index);
const float mu = index - indexInt;
return m_arrayColorLpfFcInterp[indexInt] * (1.0f - mu)
     + m_arrayColorLpfFcInterp[indexInt + 1] * mu;

// パラメータ平滑
static float expSmoothing(float target, float current, float rate) {
    return target - target * rate + current * rate;
    // = current + (target - current) * (1.0f - rate)
}
```

**ConvoPeq への導入意義**：

- ConvoPeq の dry/wet 平滑は `LinearRamp`（区間線形補間）のみ。BarcelonaReverbera の `expSmoothing`（指数平滑）異なり、急激なパラメータ変化にじぐせが発生しやすい。
- `equalPowerSin` は Taylor 近似を使っているが、LUT 化すれば计算量を削减可能。
- 全パラメータ（mix、decay、color、latency）に統一的な平滑を適用できるように。

**実装場所**：`DspUtils`（新規追加） + `ConvolverProcessor.h`（LUT 配列追加）

**実装メモ**：

```cpp
// DspUtils への追加
namespace DspUtils {
    // BarcelonaReverbera 方式の指数平滑
    template<typename T>
    static inline T expSmoothing(T target, T current, T rate) noexcept {
        return target - target * rate + current * rate;
    }

    // パラメータ平滑（sub-block 内の ramp 生成）
    template<typename T>
    static void smoothParameter(T target, T& futureCurrent, T& current, T& incr,
                                 T smoothingFactor, uint32_t blockSize) noexcept {
        current = futureCurrent;
        futureCurrent = expSmoothing(target, current, smoothingFactor);
        if (futureCurrent == current)
            futureCurrent = target;
        incr = (futureCurrent - current) / static_cast<T>(blockSize);
    }

    // LUT 補間（1024点）
    template<typename T>
    static T interpolateLUT(const T* lut, T pos, int lutSize) noexcept {
        const T index = pos * static_cast<T>(lutSize - 1);
        const uint32_t indexInt = static_cast<uint32_t>(index);
        const T mu = index - static_cast<T>(indexInt);
        return lut[indexInt] * (1.0 - mu) + lut[indexInt + 1] * mu;
    }
}
```

**期待効果**：パラメータ変更時の click/pop を消除。BarcelonaReverbera 方式是めらかなパラメータ变化を実現。

---

### 第 4 優先度：DSP スレッドへの FFT 処理オフロード

**BarcelonaReverbera の実装**：

```cpp
// ConvolutionEngineFftStage::init() 内
m_processInThread = (!m_replacesDirectStage && (m_blockSize > audioProcessingBlockSize));
if (m_processInThread) {
    m_thread.startThread(juce::Thread::Priority::high);
}
```

**ConvoPeq への導入意義**：

- ConvoPeq は现在すべての DSP をオーディオコールバックスレッドで処理。IR が長く（L2 が大きい）、サンプルレートが极高（192kHz+）の場合、FFT 処理负荷が増大。
- 大きい L1/L2 ステージをワーカースレッドにオフロード하면、オーディオリン sens がよくなる场合がある。

**導入上の注意**：ConvoPeq は RT 要件が厳しく、`DspThread` 程度の简单なスレッドでも RT 安全证明が复杂になる。実装には充分なテストが必要。

**実装アプローチ案**：

```cpp
// MKLNonUniformConvolver 内の L1/L2 処理 부분을別のスレッドに分离
// オーディオスレッドは直近の L0 のみを処理し、
// L1/L2 はバックグラウンドで accumulation
// ロックフリー：オーディオ側の缓冲は double-buffer（2面）で、DspThread と共用しない
```

> **注意**：この導入は設計変更が大がかりになる。建议は「導入希望あり（要評価）」として優先度を下げ、実際の负荷テスト结果是ってから判断することを推奨。

---

### 第 5 優先度：VST3/AU プラグインラッパー

**BarcelonaReverbera の構成**：

```
BarcelonaReveraAudioProcessor : juce::AudioProcessor
  ├─ m_params (APVTS)
  ├─ m_convolutionReverb (ConvolutionReverb)
  └─ processBlock() → m_convolutionReverb.process()

BarcelonaReverberaAudioProcessorEditor : juce::AudioProcessorEditor
  ├─ Decay/Color/DryWet スライダー × 3
  ├─ IR 選択 ComboBox
  └─ paint() で IR 画像を背景に表示
```

**導入意義**：

- ConvoPeq は现時点でスタンドアローン专用。VST3/AU 対応すれば Reason、Ableton、Logic、Cubase 等のホストで使用可能。
- ConvoPeq の DSP 引擎（MKL NUC、20-band EQ、高精度 noise shaping）はすでに優秀であり、BarcelonaReverbera の DSP 部分よりも JUCE プラグインラッパーが欲しい功能。

**導入方法**：

```cpp
// 新規ファイル案：ConvoPeqPluginProcessor.h
class ConvoPeqAudioProcessor : public juce::AudioProcessor {
    juce::AudioProcessorValueTreeState m_params;
    ConvolverProcessor m_convolver;    // 既存引擎を再利用
    EQProcessor m_eq;                  // 20-band EQ
    // APVTS バインド: Decay, Color, Dry/Wet, IR 選択, Phase mode
    void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override;
};
```

> **注意**：これは ConvoPeq の既存代码に対して大がかりなアーキテクチャ変更。将来的な DSP 引擎の插件化を検討するプロジェクトにとっては有用な参考情報。

---

### 第 6 優先度：Asymmetric Tukey Fade-out + Pre-delay

**BarcelonaReverbera の Decay Envelope** は非对称（前半 full gain、後半指数减少）。ConvoPeq の現在の Tukey fade-out は对称。

**追加すべき機能**：

| 機能 | 説明 | ConvoPeq での実装箇所 |
|---|---|---|
| **Asymmetric fade-out** | 前半は平坦、後半は指数減少の非对称包絡 | `LoaderThread::doTrimStep()` の Tukey 部分を改変 |
| **Pre-delay / IR start offset** | IR の開始位置を可变（最初の反射面の達延時間を制御） | `MKLNonUniformConvolver::SetImpulse()` の前処理 |

**Pre-delay 実装例**：

```cpp
// IRConverter への追加
void IRConverter::applyPreDelay(double* irLeft, double* irRight,
                                uint32_t& irLen, uint32_t maxIrLen,
                                uint32_t preDelaySamples) noexcept
{
    if (preDelaySamples == 0 || preDelaySamples >= irLen)
        return;

    const uint32_t newLen = irLen - preDelaySamples;
    if (newLen > maxIrLen)
        return;

    // オリジナル IR を preDelay 分だけ後ろへシフト
    for (uint32_t i = 0; i < newLen; ++i) {
        irLeft[i]  = irLeft[preDelaySamples + i];
        irRight[i] = irRight[preDelaySamples + i];
    }
    irLen = newLen;
}
```

---

## 4. 導入優先度まとめ

| 優先度 | 機能 | ConvoPeq での実装場所 | 期待効果 | 実装難度 |
|---|---|---|---|---|
| **1** | IR Decay Envelope | `ConvolverProcessor.LoadPipeline.cpp` / `IRConverter` | 残響長の直感的制御（RT60 的） | 低 |
| **2** | Color フィルタ（Wet 用 LPF+HPF） | `ConvolverProcessor` 内部 + `FilterBiquad` 再用 | 殘響のbrightness/darkness制御 | 低〜中 |
| **3** | LUT 補間 + SmoothParameter 統一化 | `DspUtils` / `ConvolverProcessor.h` | パラメータ変更時の click/pop 消除 | 中 |
| **4** | DSP スレッドへの L1/L2 オフロード | `MKLNonUniformConvolver` + 新規スレッドクラス | 高サンプルレートでの RT 安全性維持 | 高 |
| **5** | VST3/AU プラグインラッパー | 新規 `AudioProcessor` サブクラス | _plugin形式での汎用ホスト対応 | 中 |
| **6** | Asymmetric fade-out + Pre-delay | `ConvolverProcessor.LoadPipeline.cpp` | IR 末端の的处理強化・反射特性の精细制御 | 中 |

---

## 5. 実装チェックリスト

### Phase 1（即時導入可）

- [ ] `IRConverter::applyDecayEnvelope()` の追加（`ConvolutionReverb::updateIr()` の算法をポート）
- [ ] `IRConverter::applyColorFilter()` の追加（`FilterBiquad` 再用）
- [ ] `DspUtils::expSmoothing()` + `DspUtils::smoothParameter()` の追加
- [ ] `DspUtils::interpolateLUT()` の追加
- [ ] `ConvolverProcessor.LoadPipeline.cpp` の `doTransformStep()` に上記 2 つの呼出を追加
- [ ] LUT 配列（`m_arrayDecayInterp`, `m_arrayColorLpfFcInterp`, `m_arrayColorHpfFcInterp`）の追加と事前計算

### Phase 2（中期的導入）

- [ ] `applyPreDelay()` の追加
- [ ] Asymmetric fade-out への対応
- [ ] Color/decay パラメータの APVTS バインディング
- [ ] `smoothParameter` による Color/decay/mix 全パラメータの統一平滑

### Phase 3（要評価）

- [ ] L1/L2 の DSP スレッドオフロード（负荷テスト後判断）
- [ ] VST3/AU プラグインラッパー

---

*本提案は BarcelonaReverbera の public コード（Custom Non-Commercial ライセンス）と ConvoPeq のアーキテクチャを比較したものであり、ConvoPeq の既存ライセンスや設計思想を尊重した形での機能採用を前提としています。具体的な実装には ConvoPeq のコードオーナーとの確認が推奨されます。*