# ConvoPeq × BarcelonaReverbera コードレベル詳細比較報告書

**作成日**: 2026年7月6日  
**対象範囲**: ConvoPeq（Intel MKL/IPP NUC）vs BarcelonaReverbera（PFFFT テンプレート分割 FFT）  
**比較軸**: エンジン構造、フィルタリング、IR パイプライン、スレッド、メモリ、エラーハンドリングの 12 軸  

---

## 第 I 部：畳み込みエンジン比較

### 1. パーティショニング戦略

**BarcelonaReverbera** — ブロックサイズ倍々方式（Doubling Block Sizes）:

```
Direct Stage:      H0-H1  (時間領域, ゼロレイテンシ)
FftStage<64, 2>:   H2-H3  (blockSize=64,  blockCount=2, OLA)
FftStage<128, 2>:  H4-H7  (blockSize=128, blockCount=2, OLA)
...
FftStage<16384, N>: 最終段 (blockSize=16384, blockCount = IR長から算出)
```

テンプレートメタプログラミングで `GenerateFftStages<64, 16384, false>` によりコンパイル時に全ステージが生成される。中間ステージは全て `blockCount=2`（ダブルバッファリング循環処理）。最終段のみ可変ブロック数。欠点として、IR が短い場合でも全ての中間ステージが起動する。

**ConvoPeq** — 3層適応型 NUC（MKLNonUniformConvolver）:

```
Layer 0 (即時):  partSize = nextPow2(max(blockSize, 64)), max 32 partitions
                 → 全パーティションを毎コールバック内でインライン処理
                 レイテンシ = l0PartSize（最小 64 サンプル）

Layer 1 (遅延):  partSize = L0 * 8 (default), max 64 partitions
                 → partsPerCallback で分散処理

Layer 2 (遅延):  partSize = L1 * 8 (default), 残り IR テール
                 → 同分散処理
```

ソース引用（`MKLNonUniformConvolver.cpp:615-634`）:

```cpp
const int l0Part = juce::nextPowerOfTwo(std::max(blockSize, 64));
const int l1Part = l0Part * tailL1L2Mult;   // デフォルト ×8
const int l2Part = l1Part * tailL1L2Mult;   // デフォルト ×8

// tailStartSec により L0 の coverage を動的に決定
const int l0LenTarget = static_cast<int>(std::llround(tailStartSec * sampleRateForTail));
const int l1Len = tailEnabled ? std::max(0, std::min(irLen - l0Len, kL1MaxParts * l1Part)) : 0;
const int l2Len = tailEnabled ? std::max(0, irLen - l0Len - l1Len) : 0;
```

**勝敗: ConvoPeq**。BarcelonaReverbera は全ての中間サイズで必ず 2 ブロック処理するため IR が短い場合でも 9 段階の FFT ステージがフル稼働する。ConvoPeq は `tailStartSec` で L0 coverage を動的調整し、L1/L2 の `partsPerCallback` 分散でピーク負荷を平坦化する。真の意味で「非一様分割」を計算スケジュールにまで拡張している。

---

### 2. FFT 実装詳細

**BarcelonaReverbera** — PFFFT（Pretty Fast FFT by Julien Pommier）:

```cpp
// Fft.h テンプレートラッパ
PFFFT_Setup* m_setup = pffft_new_setup(fftSize, PFFFT_REAL);  // 実数→ハーフコンプレックス
// 単精度 float のみ、16-byte 境界、split-radix
// unordered 時: pffft_zconvolve_accumulate() で畳み込み
// ordered 時: 手動複素乗算ループ（要素ごと）
```

- **精度**: float（32-bit）。FilterBiquad 内部のみ double
- **配置**: `alignas(16)` スタック配列（SSE 相当）
- **FFT サイズ**: 時間領域 `2×blockSize` → 周波数領域 `blockSize+1` 複素ビン
- **窓処理**: なし。OLA（Overlap-Add）で自然なブロック境界処理

**ConvoPeq** — Intel IPP（Integrated Performance Primitives）:

```cpp
// MKLNonUniformConvolver.cpp:665-713
int order = 0;
{ int tmp = l.fftSize; while (tmp > 1) { tmp >>= 1; ++order; } }

// 実数→CCS（Conjugate Complex Symmetric）フォワード
ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, l.fftSpec, l.fftWorkBuf);

// CCS→実数インバース（自動 1/N 正規化）
ippsFFTInv_CCSToR_64f(l.accumBuf, l.fftOutBuf, l.fftSpec, l.fftWorkBuf);
```

- **精度**: double（64-bit）。`R_64f` 系関数で統一
- **配置**: `mkl_malloc(size, 64)` — 64-byte 境界（AVX-512 準拠）
- **FFT サイズ**: `2×partSize` 時間領域 → `2×(partSize+1)` double（CCS インタリーブ）
- **キャッシュ**: `IppFFTPlanCache`（`std::unordered_map<order, IppFFTPlan>`）で FFT プランを Message Thread 上でキャッシュ。Audio Thread ではゼロアロケーション
- **形式**: IPP_FFT_DIV_INV_BY_N フラグにより逆変換で自動正規化。手動スケーリング不要

**FFT サイズとバッファ管理の比較**:

| 項目 | ConvoPeq（IPP） | BarcelonaReverbera（PFFFT） |
|---|---|---|
| メモリアロケータ | `mkl_malloc(size, 64)`（Intel TBB heap） | スタック配列 + `alignas(16)` |
| アライメント | 64-byte（AVX-512 ready） | 16-byte（SSE ready） |
| FDL ミラーリング | 線形化リング: 書き込み時に `index + numParts` へ mirror してから走査 | 循環 FDL: `(writePtr - b + blockCount) % blockCount` |
| メモリ表現 | デュアル: AoS（IPP 互換）+ SoA（AVX2 MAC 用 split-complex） | AoS のみ（`cplx_f32[2]` インタリーブ） |
| ワークバッファ | 全層に事前割当（`mkl_malloc`、SetImpulse 時に固定） | 各 Fft インスタンスが保持、init() 時に外部から受領 |
| FFT プラン再使用 | プランキャッシュ（order→Spec マップ） | PFFFT_Setup を Fft インスタンスごとに１つ |

**勝敗: ConvoPeq**。倍精度、64-byte アライメント、デュアル AoS+SoA 表現、プランキャッシュ、ゼロアロケーション。BarcelonaReverbera の PFFFT は SIMD 化された split-radix で FFT 単体の計算コストは低いが、倍精度でない点とインタリーブ形式のみの制約がパフォーマンス上限を決める。

---

### 3. オーバーラップ処理

**ConvoPeq** — Overlap-Save（OLS）:

```cpp
// processLayerBlock(): 1058-1159
// 1. [prevInput | currentInput] で OLS ブロック構築
FloatVectorOperations::copy(l.fftTimeBuf,              l.prevInputBuf, l.partSize);
FloatVectorOperations::copy(l.fftTimeBuf + l.partSize, l.inputAccBuf,  l.partSize);
FloatVectorOperations::copy(l.prevInputBuf, l.inputAccBuf, l.partSize);

// 2. フォワード FFT → 周波数領域乗算 → バックワード FFT
ippsFFTFwd_RToCCS_64f(l.fftTimeBuf, currentFDLSlot, ...);
// 全パーティション × IR 周波数領域の複素乗算積
ippsFFTInv_CCSToR_64f(l.accumBuf, l.fftOutBuf, ...);

// 3. Overlap-Save: 後半半分（partSize サンプル）のみ有効出力
ringWrite(l.fftOutBuf + l.partSize, l.partSize);  // 前半は循環畳み込みアーティファクト
```

- 出力が各コールバックで確定する（前回の overlap 状態に依存しない）
- `m_overlap[2][blockSize]` のような永続状態不要 ⇒ 浮動小数点誤差の蓄積なし

**BarcelonaReverbera** — Overlap-Add（OLA）:

```cpp
// ゼロパッド: 入力ブロック後半をゼロ埋め（自動的に 2×blockSize）
// IFFT 後: 前半 + 前回 overlap → 出力、後半 → 次回 overlap に保存
for i in [0, blockSize):
    out[i] = m_conv[i] + m_overlap[ch][i];
memcpy(m_overlap[ch], &m_conv[blockSize], blockSize * sizeof(float));
```

- IFFT 結果の前半と後半両方を使い切る（効率は良い）
- `m_overlap` への FP 誤差蓄積が長期的ドリフトにつながる可能性

**FDL（過去オーディオ FFT ブロック）リングバッファ**:

ConvoPeq は mirror write で線形化:
```cpp
// ミラーリング: index と index+numParts の両方に書き込む
double* mirrorFDLSlot = l.fdlBuf + (l.fdlIndex + l.numParts) * l.partStride;
memcpy(mirrorFDLSlot, currentFDLSlot, l.partStride * sizeof(double));
// 読み出し時は線形索引（% 演算不要）
const int linStart = baseFdlIdx - l.numPartsIR + 1 + l.numParts;
const double* fdlLin = fdlBase + linStart * l.partStride;
```

BarcelonaReverbera は循環:
```cpp
for b in [0, blockCount):
    readPtr = (writePtr - b + blockCount) % blockCount;
```

**勝敗: ConvoPeq（OLS）**。OLS は (1) 永続状態ドリフトなし、(2) 出力が決定的、(3) 線形化リングで % 演算不要。

---

### 4. IR スペクトルフィルタリング（HC/LC vs Color）

**ConvoPeq** — `applySpectrumFilter()`、IR 周波数領域へのベイク（静的）:

`MKLNonUniformConvolver.cpp:313-443`:

```cpp
void MKLNonUniformConvolver::applySpectrumFilter(const FilterSpec& spec) noexcept {
    for (int li = 0; li < m_numActiveLayers; ++li) {
        // HC（High-Cut）: Sharp（Butterworth 4次）/ Natural（cosine crossfade）/ Soft（Gaussian falloff）
        // LC（Low-Cut）: cosine roll-on（fc≈18Hz or 15Hz）
        // ゲイン曲線を周波数ドメインで直接計算
        double* gain = reusableGain.get();

        // 全パーティションの irFreqDomain に vdMul でベイク
        for (int p = 0; p < l.numParts; ++p) {
            double* slot = l.irFreqDomain + p * stride;
            vdMul(cSize * 2, slot, gainIL, slot);  // MKL VML 並列乗算
        }
    }
}
```

HC モードの具体的形状:
- **Sharp**: `1 / sqrt(1 + x^8)` — Butterworth 4次相当
- **Natural**: `0.5 * (1 + cos(pi * x))` — cosine crossfade（デフォルト）
- **Soft**: `exp(-4.60517 * x^2)` — Gaussian roll-off

**BarcelonaReverbera** — FilterBiquad、IR 時間領域への in-place 適用（動的）:

`updateIr()` 内:
```cpp
const float filterFc = filterIsLowPass
    ? std::exp((1.0f + m_colorControl) * COLOR_LPF_FREQ_RANGE + COLOR_LPF_FREQ_LOGMIN)
    : std::exp(m_colorControl * COLOR_HPF_FREQ_RANGE + COLOR_HPF_FREQ_LOGMIN);

m_filterLPF[ch].clearState();
m_filterLPF[ch].setTargetFreq(filterFc, smoothingFactor, samplerate);
m_filterLPF[ch].process(irPostProcessed[ch], irPostProcessed[ch], irLen);
m_filterHPF[ch].process(irPostProcessed[ch], irPostProcessed[ch], irLen);
```

| 項目 | ConvoPeq（周波数領域ベイク） | BarcelonaReverbera（時間領域 IIR） |
|---|---|---|
| オーディオスレッドコスト | **ゼロ**（プリベイク済） | ゼロ（IR 側で前処理済） |
| パラメータ変更コスト | `SetImpulse()` 再実行（全 IR FFT 再計算） | バックグラウンドスレッドで in-place フィルタ→atomic swap |
| フィルタ品質 | カスタム曲線（Butterworth/cosine/Gaussian）直接スペクトル乗算 | 2次バターワースのみ。IIR の過渡応答あり |
| IR 再計算 | 全パーティションの FFT 再実行が必要 | 時間領域フィルタのみ（FFT の再計算不要） |
| 変更適用レイテンシ | SetImpulse 完了後即座 | バックグラウンド処理後、atomic swap |

**勝敗: 用途による（引き分け）**。静的フィルタリング（一度設定して固定）には ConvoPeq のベイク方式が圧倒的に優位（ゼロ RT コスト＋高品質フィルタ形状）。動的フィルタリング（ユーザーがリアルタイムでノブを回す）には BarcelonaReverbera の IIR 前処理方式が優位（FFT 再計算不要＋パラメータ平滑化可能）。理想は ConvoPeq の HC/LC ベイク＋BarcelonaReverbera の Color ノブの**ハイブリッド**導入。

---

### 5. テール処理（残響減衰）

**ConvoPeq** — Air Absorption（周波数領域 HF チルト）＋ Layer Tail Contouring:

`SetImpulse():508-561` + `909-947`:

```cpp
// TailMode 0: Air Absorption — 周波数依存の HF 減衰を L1/L2 の irFreqDomain にベイク
const double dampingCoeff = dampingBase * layerWeight;  // layerWeight: L1=1.0, L2=1.6
for (int k = 0; k < l.complexSize; ++k) {
    const double fNorm = static_cast<double>(k) / denom;
    const double hfTilt = std::exp(-dampingCoeff * fNorm * fNorm);
    gainInterleaved[2*k] = gainInterleaved[2*k+1] = hfTilt;
}
vdMul(l.complexSize * 2, slot, gainInterleaved, slot);  // 全パーティションに適用

// TailMode 1: Layer Tail Contouring — レイヤーゲイン成形
layer1Gain = tailStrength * (1.05 + 0.20 * strength01);  // L1: 最大〜1.25x（強調可能）
layer2Gain = tailStrength * (0.82 + 0.12 * strength01);  // L2: 最大〜0.94x（減衰）
// Get() 時に addScaledFallback で適用
```

**BarcelonaReverbera** — Decay envelope（時間領域振幅エンベロープ）:

```cpp
// updateIr(): バックグラウンドスレッドで IR に in-place
const uint32_t decayCutPointSamples = irLen * m_decayCurrent;
for (uint32_t i = decayCutPointSamples; i < irLen; i++) {
    decayGainCurrent = DspUtils::expSmoothing(0.0f, decayGainCurrent, decayEnvSmoothingFactor);
    irPostProcessed[ch][i] = m_irPreProcessed[ch][i] * decayGainCurrent;
}
```

| 項目 | ConvoPeq Air Absorption | ConvoPeq Layer Contouring | BarcelonaReverbera Decay |
|---|---|---|---|
| 作用ドメイン | 周波数領域（HF 減衰） | レイヤー出力ゲイン | 時間領域（サンプル振幅） |
| 効果 | 高域が時間とともに減衰（物理モデル） | 低域層 vs 高域層の相対強度調整 | 単純な振幅フェード |
| 物理モデル | 空気の周波数依存吸収を近似 | 知覚的テールプロミネンス | 単純振幅操作 |
| L1 挙動 | dampCoeff=0.35〜1.45（可変） | gain=1.05〜1.25x（強調可能） | 一様フェード |
| L2 挙動 | dampCoeff=0.56〜2.32（L1 の 1.6倍） | gain=0.82〜0.94x（減衰） | 一様フェード（L1 と同一カーブ） |
| レイヤー間差別化 | あり（異なる減衰係数） | あり（異なるゲイン係数） | **なし** |

**勝敗: ConvoPeq**。周波数依存の HF 減衰（Air Absorption）は実際の音響現象（高域ほど距離で減衰）をモデル化しており、BarcelonaReverbera の単純な時間域一様フェードより物理的に正確。ただし、**ユーザーが減衰特性を直感的に制御できるパラメータ**（Decay ノブ）は BarcelonaReverbera にあって ConvoPeq にはない。これは導入が望まれる主要機能の一つ。

---

### 6. レイテンシ管理

**ConvoPeq** — リングバッファオフセット + オプション Direct Head:

```cpp
// レイテンシ = L0.partSize（最小 64 サンプル）
m_latency = m_layers[0].partSize;
// L0 → リングバッファ（ringWrite → ringRead で遅延）
// Direct convolution（オプション）: enableDirectHead=true → 最大 32 taps は即時出力
```

**BarcelonaReverbera** — Direct Stage による真のゼロレイテンシ:

```cpp
// DirectStage: H0-H1 を時間領域畳み込み（BCNRVRB_DIRECT_STAGE_MAX_BLOCK_SIZE=128）
// 128 タップまでゼロレイテンシ。その後 FFT ステージの自然遅延が加算。
```

| 項目 | ConvoPeq | BarcelonaReverbera |
|---|---|---|
| 最小レイテンシ | `l0PartSize`（設定次第、64〜128） | **0**（Direct Stage カバレッジ内） |
| ゼロレイテンシ経路 | `enableDirectHead`（オプション、最大32 taps） | DirectStage 必須（最大128 taps） |
| 遅延補償 | リングバッファ固定オフセット | ブロックスケジューリングに組込み済み |
| 直接畳み込み範囲 | 32 taps 最大（`kMaxDirectTaps=32`、AVX2） | 128 taps 最大 |

**勝敗: BarcelonaReverbera**。128 taps のゼロレイテンシ経路は ConvoPeq の 32 taps を大きく上回る。ConvoPeq への導入が望まれる機能。ただし、ConvoPeq の `enableDirectHead` は AVX2 で最適化済みであり、BarcelonaReverbera の Direct Stage は O(N²) の単純実装。ConvoPeq 実装時は N=128 taps への拡張＋AVX2 ベクトル化維持が望ましい。

---

## 第 II 部：フィルタリングとパラメータシステム比較

### 7. フィルタ設計・配置比較

**BarcelonaReverbera** — FilterBiquad：バターワース 2 次 / DF2T / 内部 double / `alignas(16)`

係数設計（双一次変換）:

```cpp
void FilterBiquad::computeCoefficientsButterworth2ndOrder(double samplerate) {
    const double omega_c = 2π * m_cutoffFreq_Current / samplerate;
    const double omega   = std::tan(omega_c / 2.0);  // pre-warping
    const double omega2  = omega * omega;

    const double a0 = 1.0 + sqrt2_omega + omega2;  // 正規化係数

    // a1, a2: LPF/HPF 共通
    a1 = 2.0 * (omega2 - 1.0) / a0;
    a2 = (1.0 - sqrt2_omega + omega2) / a0;

    if (m_lowPass) {          // LPF
        b0 = omega2 / a0; b1 = 2.0 * b0; b2 = b0;
    } else {                  // HPF
        b0 = 1.0 / a0; b1 = -2.0 * b0; b2 = b0;
    }
}
```

**ConvoPeq** — OutputFilter：RBJ Audio EQ Cookbook / 4 次（2 biquad cascade）/ DF2T / SSE2/FMA

係数設計（LPF 例、`OutputFilter.cpp:25-64`）:

```cpp
BiquadCoeff OutputFilter::makeLPF(double fc, double Q, double fs) noexcept {
    const double w0    = 2.0 * pi * fc / fs;
    const double sn    = std::sin(w0);       // libm call（Message Thread のみ）
    const double cs    = std::cos(w0);
    const double alpha = sn / (2.0 * Q);

    BiquadCoeff c;
    c.b0 = (1.0 - cs) * 0.5 / (1.0 + alpha);
    c.b1 = (1.0 - cs) / (1.0 + alpha);
    c.b2 = (1.0 - cs) * 0.5 / (1.0 + alpha);
    c.a1 = -2.0 * cs / (1.0 + alpha);
    c.a2 = (1.0 - alpha) / (1.0 + alpha);
    return c;
}
```

**配置の違い**:

| 項目 | BarcelonaReverbera | ConvoPeq |
|---|---|---|
| **フィルタ位置** | **PRE-convolution**（IR に in-place 適用、前処理段） | **POST-convolution**（畳み込み出力段に別途 OutputFilter 適用） |
| **フィルタ対象** | **Wet 信号のみ**（IR そのもののスペクトル変更） | **Wet+Dry 混合後**の最終出力全体 |
| **フィルタタイプ** | 1 つの Biquad で LPF または HPF（選択式） | HC（3 モード×4 次）+ LC（2 モード×2 次）+ fixed HPF + 可変 LP（3 モード） |
| **カットオフ変更** | `setTargetFreq()` → expSmoothing 連続変化可能 | `prepare()` で全モードの係数テーブルを事前計算（固定） |
| **ユーザー制御** | Color ノブ（-1〜+1）で連続可変 | モード選択（Sharp/Natural/Soft）のみ、周波数固定 |

ConvoPeq の SSE2/FMA 実装（`OutputFilter.cpp:152-180`）:

```cpp
inline __m128d biquadStep128_FMA(
    const __m128d x, const __m128d b0, b1, b2, a1, a2,
    const __m128d kDenThresh, __m128d& w1, __m128d& w2) noexcept
{
    const __m128d y   = _mm_fmadd_pd(b0, x, w1);                            // y = b0*x + w1
    __m128d new_w1 = _mm_fmadd_pd(b1, x, _mm_fnmadd_pd(a1, y, w2));         // w1 = b1*x - a1*y + w2
    __m128d new_w2 = _mm_fnmadd_pd(a2, y, _mm_mul_pd(b2, x));                // w2 = b2*x - a2*y
    // 非正規化数除去（ブランチレス）
    w1 = _mm_andnot_pd(_mm_cmplt_pd(abs_w1, kDenThresh), new_w1);
    w2 = _mm_andnot_pd(_mm_cmplt_pd(abs_w2, kDenThresh), new_w2);
    return y;
}
```

L/R チャンネルを `__m128d` にパックして処理。3 Biquad カスケード（HC+HC+LC）全体が FMA 命令で **2× スカラースループット**。

**勝敗: 比較不能（設計思想の違い）**。ConvoPeq は固定高品質フィルタ（一度設定して変更しない）、BarcelonaReverbera は連続可変 Color ノブ。両方の存在意義がある。ConvoPeq に **Wet 信号のみ**の連続可変フィルタ（Color ノブ）を導入する提案が第2優先度で有効。

---

### 8. パラメータ平滑化比較

**BarcelonaReverbera** — expSmoothing + smoothParameter + LUT 配列:

```cpp
// expSmoothing (指数平滑)
static inline float expSmoothing(float target, float current, float rate) {
    return target - target * rate + current * rate;
}

// smoothParameter（sub-block ramp 生成）
static inline void smoothParameter(float target, float& futureCurrent,
    float& current, float& incr, float smoothingFactor, uint32_t blockSize) {
    current = futureCurrent;
    futureCurrent = expSmoothing(target, current, smoothingFactor);
    if (futureCurrent == current) futureCurrent = target;
    incr = (futureCurrent - current) / blockSize;
}
```

パラメータ到着は漸近的（`τ = 5ms` で約 3τ 後に 95% 到達）。per-sample ramp は `incr` で線形補間。

**ConvoPeq** — LinearRamp + Generation Counter:

```cpp
struct LinearRamp {
    double current    = 0.0;
    double target     = 0.0;
    double step       = 0.0;
    int    remaining  = 0;
    int    totalSteps = 1;

    void setTargetValue(double v) noexcept {
        const int steps = (remaining > 0) ? remaining : totalSteps;
        step = (target - current) / static_cast<double>(steps);
        remaining = steps;
    }

    inline double getNextValue() noexcept {
        current += step;
        if (--remaining <= 0) current = target;   // 最終ステップでスナップ
        return current;
    }
};
```

パラメータ到着は決定的（「ちょうど totalSteps サンプル後に目標到達」）。Message→Audio スレッド間は Generation Counter で同期:

```cpp
// Message スレッド: 書き込み
convo::fetchAddAtomic(smoothingTimeChangePendingGen, 1, std::memory_order_acq_rel);

// Audio スレッド: 読み取り（Atomic 書き込みなし！）
const uint64_t curGen = consumeAtomic(smoothingTimeChangePendingGen, acquire);
if (curGen != m_smoothingTimeChangePendingGenSeen) {
    m_smoothingTimeChangePendingGenSeen = curGen;  // ローカル非アトミック更新
    activeMixSmoother.reset(sampleRate, newTime);
}
```

| 項目 | BarcelonaReverbera | ConvoPeq |
|---|---|---|
| 平滑化アルゴリズム | **指数平滑**（IIR 1-pole、漸近的） | **線形補間**（区間 ramp、決定的） |
| 収束挙動 | 3τ 後に 95%（理論上は無限に漸近） | 正確に totalSteps サンプル後に完全収束 |
| 単位 | 時定数 τ（秒 or サンプル数） | 平滑時間（秒）→整数ステップ数 |
| CPU コスト | per-sample: 1 FMA + 1 分岐止め | per-sample: 1 add + 1 conditional |
| Thread Safety | `std::atomic<float>` 直接書き込み（テアリング可能性あり） | Generation Counter（Message は atomic RMW、Audio は read-only） |
| 適用パラメータ | mix, color, decay（全パラメータに統一） | mix, latency, crossfade gain（3 つの専用 smoother） |

**勝敗: 実装依存**。ConvoPeq の Generation Counter パターンは **Audio Thread が絶対に atomic 書き込みをしない**設計で RT 安全性の点で優れる。LinearRamp は決定的収束が利点で、Instrument 的な操作に適する。BarcelonaReverbera の expSmoothing はリバーブ的な「ゆるやかな変化」に適する。ConvoPeq への導入提案としては、**Color/Decay パラメータには expSmoothing + smoothParameter、Mix/Latency には従来の LinearRamp** というハイブリッドが最適解。

---

### 9. SIMD 比較

**BarcelonaReverbera**: `alignas(16)` ヒントのみ。FilterBiquad・畳み込みに明示的な SIMD 組み込み命令なし。コンパイラの自動ベクトル化に依存。

**ConvoPeq**: 全面的な SIMD 最適化。

| コンポーネント | SIMD 幅 | 命令 | 詳細 |
|---|---|---|---|
| OutputFilter | 128-bit (`__m128d`) | `_mm_fmadd_pd`, `_mm_fnmadd_pd` | L/R チャンネルパック、2並列 Biquad、ブランチレス非正規化除去 |
| Mix 平滑 | 256-bit (`__m256d`) | `_mm256_fmadd_pd` | 4並列 Wet+Dry、2つのコードパス（平滑時／定常時） |
| Delay 補間 | 256-bit (`__m256d`) | `_mm256_fmadd_pd` | cubic 補間を 4サンプル並列 |
| Fade crossfade | 256-bit (`__m256d`) | `_mm256_fmadd_pd` | new*fade + old*(1-fade) の 4並列 |
| Denormal 除去 | 256-bit (`__m256d`) | `_mm256_andnot_pd`, `_mm256_cmp_pd` | ブランチレスマスク（masked zeroing） |
| Saturation | 128-bit (`__m128d`) | `_mm_max_sd`, `_mm_min_sd` | スカラー値の clamp に SIMD |
| ミラーリング | なし（`memcpy`） | — | FDL mirror write に fallback |

**勝敗: ConvoPeq**。Audio 処理の全段で明示的 SIMD。BarcelonaReverbera は `alignas(16)` のみでコンパイラ任せ。

---

## 第 III 部：IR パイプライン比較

### 10. IR ローディングパイプライン段階的比較

| 段階 | BarcelonaReverbera | ConvoPeq | 勝敗 |
|---|---|---|---|
| **0: ソース** | Python スクリプトでコンパイル時配列埋込み | Runtime `AudioFormatReader`（WAV/AIFF/FLAC/Ogg） | **ConvoPeq**（形式自由） |
| **1: 無音トリム** | Python build script でのみ実行（runtime では実行されない） | **AVX2 後方スキャン**（`_CMP_GT_OQ`、閾値 `1e-15`） | **ConvoPeq**（runtime、全 IR に確実） |
| **2: DC ブロック** | **なし** | **UltraHighRateDCBlocker**（fc=1Hz、2段カスケード IIR） | **ConvoPeq**（必須機能） |
| **3: Fade-out** | **Decay Envelope**（ユーザー可変、カット点以降を指数減衰） | **2層フェード**: Asymmetric Tukey（ピーク中心、adaptive post-ramp）+ 線形 2% ramp | 用途で異なる。ConvoPeq の **形状品質は高いがユーザー制御不可**。BarcelonaReverbera の **Decay ノブのユーザー制御性**が導入提案 |
| **4: リサンプリング** | JUCE ResamplingAudioSource（品質指定なし、windowed-sinc FIR） | **r8brain CDSPResampler24IR**（`transBand=2.0 Hz`, `stopBandAtten=140 dB`）＋位相モード選択可能 | **ConvoPeq**（品質桁違い）。r8brain の 140dB 減衰は JUCE 標準リサンプラを圧倒 |
| **5: 正規化** | `factor = 0.65 / sqrt(sumSquares)`（固定定数） | `computeScaleFactor()`（異なる IR 間のラウドネス整合 + safety margin） | **ConvoPeq**（適応的、cross-IR 音量一致） |
| **6: Phase 変換** | **なし**（IR は Linear Phase のみ） | **Minimum Phase**（Hilbert 変換、4×FFT）＋**Mixed Phase**（CMA-ES オールパス設計、160世代）。3 段 fallback chain。メモリ＋ディスクキャッシュ | **ConvoPeq 独占機能**。BarcelonaReverbera に全くない |
| **7: ポストプロセス** | **Color**（LPF/HPF on IR, 動的） | **HC/LC**（engine 出力段に bake, 静的）＋**Air Absorption**（周波数領域 HF tilt） | **ハイブリッド導入**が望ましい |

ConvoPeq の Floating-point 変換（`LoaderThread.cpp:453-476`）:
```cpp
// float → double: SIMD 最適化（独自実装、DspNumericPolicy 参照）
auto tempAligned = convo::makeAlignedArray<double>(static_cast<size_t>(fileLength));
convo::input_transform::convertFloatToDoubleHighQuality(src, tempAligned.get(), fileLength);
```

BarcelonaReverbera は float のまま処理。ConvoPeq は double 変換で精度を確保。

---

### 11. IR 交換メカニズム比較

**ConvoPeq** — RCU（Read-Copy-Update）によるエンジン丸ごと交換:

```cpp
// LoadPipeline.cpp:675-688
void ConvolverProcessor::switchEngineOnMessageThread(StereoConvolver* newEngine) noexcept {
    auto* oldEngine = exchangeActiveEngine(newEngine, std::memory_order_acq_rel);
    // EpochDomain を通じてリタイア進行
    if (auto* provider = getRcuProvider())
        provider->advanceRetireEpoch();
    if (oldEngine)
        retireStereoConvolver(oldEngine, 0);  // 遅延破棄
}
```

- **交換粒度**: エンジン全体（FFT プラン、IR データ、全段構成）
- **Thread Safety**: `epoch_domain.h` に基づく 3 相リタイア（allocation → use → free）
- **手続き**: 新エンジンを Loader Thread で完全構築 → 原子的ポインタ交換 → in-flight reader 完了後に旧エンジン破棄

**BarcelonaReverbera** — `std::atomic<uint8_t>` による IR バッファインデックス交換:

```cpp
std::atomic<uint8_t> m_irIndex;  // 0 or 1（2面バッファインジケータ）

// process() 内（Atomic store で瞬時に交換）
if (canUpdateIr()) {
    m_convolutionEngine.updateIr(m_irUpdateIndex);
    m_irUpdateIndex = (m_irUpdateIndex == 0) ? 1 : 0;
}
```

- **交換粒度**: IR データポインタのみ（エンジン構造は不変）
- **Thread Safety**: `std::atomic<uint8_t>`（lock-free guarantee 静的に checked）
- **IR 更新は非同期**: DspThread で IR を周波数領域変換中は旧 IR で継続 → 準備完了後 atomic swap

| 項目 | ConvoPeq RCU | BarcelonaReverbera Atomic |
|---|---|---|
| 交換可能なもの | エンジン全体（FFT サイズ、パーティション数、Phase mode、HC/LC 設定すべて） | IR データのみ（エンジン構造はテンプレート固定） |
| 複雑性 | 极高（EpochDomain、reader guard、retire queue、generation tracking） | 极低（1 つの atomic store） |
| 交換レイテンシ | エンジン構築に 500ms〜10s（Loader Thread 上）、ポインタ交換はインスタント | IR 前処理に 10〜500ms（DspThread 上）、atomic index store は 1 cycle |
| コード量 | ~1000 行（RCU infrastructure + retire routing） | ~10 行（atomic store + canUpdateIr チェック） |

**勝敗: アーキテクチャによる。** ConvoPeq の RCU はより強力（エンジン全体交換可能）だが複雑。BarcelonaReverbera の原子交換は IR 交換に特化し简单で十分。ConvoPeq はすでに RCU 方式で実装済みであり、導入提案は不要。

---

### 12. エラーハンドリング比較

**BarcelonaReverbera**: `DEBUG_ASSERT` のみ。本番ビルドではアサーションが除去される。

**ConvoPeq** — 8層エラー耐性システム:

| 層 | エラー種別 | ハンドリング方法 | コード |
|---|---|---|---|
| 1 | ファイル読み込み | 6 種のエラーメッセージ + エラーコード | `doLoadIRStep():422-476` |
| 2 | リサンプル失敗 | `ResampleResult::Cancelled/SilentIR/Error` に分岐 | `doTrimStep():542-576` |
| 3 | Phase 変換失敗 | CMA-ES → GreedyAdaGrad → Phase1（3 段 fallback） | `MixedPhase.cpp:39-62` |
| 4 | IR バリデーション | `isfinite` + 振幅制限 + 急峻ジャンプ検出 | `applyComputedIR():355-420` |
| 5 | NUC 初期化失敗 | `catch(bad_alloc)`, `catch(std::exception)`, `catch(...)` | `finalizeNUCEngineOnMessageThread():569-618` |
| 6 | メッセージキュー溢れ | `aligned_free` + error dispatch + retire | `queueFinalizeOnMessageThread():331-350` |
| 7 | UI 通知 | `handleLoadError()` でローディング/リビルド状態リセット + UI notification | `handleLoadError():507-515` |
| 8 | Loader Thread GC | 2 超の Trash Bin で force-delete | `cleanup():517-551` |

**勝敗: ConvoPeq** — 圧倒的に堅牢。BarcelonaReverbera の `DEBUG_ASSERT` 方式は軽量だが、本番品質には ConvoPeq の多層防御が必須。

---

## 第 IV 部：導入提案まとめ

### 12 軸総合スコア

| # | 比較軸 | ConvoPeq | BarcelonaReverbera | 同点 |
|---|---|---|---|---|
| 1 | パーティショニング | ★ | | |
| 2 | FFT 実装 | ★ | | |
| 3 | オーバーラップ | ★ | | |
| 4 | IR スペクトルフィルタリング | | | ★ |
| 5 | テール処理 | ★ | | |
| 6 | レイテンシ | | ★ | |
| 7 | フィルタ設計 | | | ★ |
| 8 | パラメータ平滑化 | | | ★ |
| 9 | SIMD | ★ | | |
| 10 | IR パイプライン | ★ | | |
| 11 | IR 交換 | | | ★ |
| 12 | エラーハンドリング | ★ | | |

### ConvoPeq に導入が望まれる機能（優先度順）

| 優先度 | 機能 | BarcelonaReverbera 由来 | 既存 ConvoPeq 相当 | 導入の根拠 |
|---|---|---|---|---|
| **1** | ユーザー制御可能な Decay Envelope | `updateIr()`: 時間域 exponential taper | Air Absorption（周波数域 HF tilt）のみ | ユーザーの残響長制御はリバーブの根幹機能。Air Absorption は周波数域のみで時間域エンベロープには対応不可 |
| **2** | Wet 信号への Color フィルタ（連続可変 LPF+HPF） | FilterBiquad + setTargetFreq + expSmoothing | OutputFilter（固定周波数、混合後出力全体に適用） | Wet 信号個別の輝度制御が可能に。既存の OutputFilter の高品質実装をベースにバターワース連続可変を追加する形で最小実装可能 |
| **3** | expSmoothing + smoothParameter の統一導入 | ConvolutionReverbCommon.h: DspUtils | LinearRamp（Mix/Latency 用） | Color/Decay パラメータには指数的変化が適切。LinearRamp は保持したまま新規パラメータに exp 平滑を追加 |
| **4** | LUT ベース非線形パラメータマッピング | 1024点 LUT: dB 音量、log 周波数、対数 Decay | 全パラメータが線形 or identity マッピング | ユーザー操作性向上。ノブ操作に自然なカーブ（dB スケール、log スケール）を提供 |
| **5** | Direct Head を 128 taps に拡張 | DirectStage 必須（128 taps） | enableDirectHead（オプション、32 taps） | 真のゼロレイテンシ経路延長。既存の AVX2 最適化を維持した拡張が必要 |
| **6** | コンパイル時 IR 埋め込み（Factory IR） | ConvertWavsToCArray.py + IrBuffersAutoGenerated.h | すべて Runtime | 起動時即座に利用可能なビルトイン IR。Floating-point 埋め込みの課題あり（float→double 変換 or pre-converted double） |

### 実装難度評価

| # | 機能 | 難度 | 既存コードへの影響範囲 | 推定期間 |
|---|---|---|---|---|
| 1 | Decay Envelope | 低 | `ConvolverProcessor.LoadPipeline.cpp` + `IRConverter` | 1〜2 日 |
| 2 | Color Filter（Wet 用） | 低〜中 | `ConvolverProcessor` 内部 + `FilterBiquad`（新規 or OutputFilter 拡張） | 2〜3 日 |
| 3 | expSmoothing + smoothParameter | 低 | `DspNumericPolicy.h` に追加 + 各パラメータ適用 | 1 日 |
| 4 | LUT マッピング | 中 | `ConvolverProcessor.h` + `prepare()` で LUT 生成 + 各パラメータセッター | 2 日 |
| 5 | Direct Head 128 taps | 中 | `MKLNonUniformConvolver` + `ConvolverProcessor.Runtime.cpp`（latency 再計算） | 3〜5 日 |
| 6 | Factory IR 埋め込み | 中〜高 | 新規 build script + `IrBuffersEmbedded.h` + `doLoadIRStep()` に fast path | 3〜5 日 |

---

*本報告書は BarcelonaReverbera（Custom Non-Commercial License）の公開コードと ConvoPeq のプライベートコードを比較したものです。具体的な実装には各プロジェクトのライセンス条件の確認が推奨されます。*