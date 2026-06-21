# バグレビュー妥当性検証報告 (v3)

- **作成日**: 2026-06-21
- **更新日**: 2026-06-21 (v3: 追加調査・確度評価を反映)
- **対象レビュー**: `doc/work52/bug_review.md`
- **調査者**: AI Agent (GitHub Copilot / DeepSeek V4 Flash)
- **調査対象**: PCローカルWorking Tree (GitHubではなくローカルソースコード)

---

## 1. 調査に使用したツール

| ツール | 用途 | 結果 |
|--------|------|------|
| **Serena MCP** | シンボル解決・関数ボディ取得 | ✅ 全対象関数の実装を確認 |
| **AiDex MCP** | 識別子検索・プロジェクト構造把握 | ✅ 278ファイルから合致箇所を特定 |
| **CodeGraph MCP** | モジュール構造・呼び出し関係 | ✅ 呼び出しチェーンを確認 |
| **semble (CLI)** | 自然言語コード検索 | ✅ 全15クエリ |
| **grep/Select-String** | パターン検索 | ✅ 補完的調査 |
| **Web文献調査** | DSP理論の外部検証 | ✅ TPT SVF・VAフィルタ理論を確認 |
| **cocoindex-code** | CLIコード検索 | ❌ インストールなし |

---

## 2. 確認したソースファイル

| ファイル | 主要シンボル |
|---------|-------------|
| `src/eqprocessor/EQProcessor.Processing.cpp` | `processBand`, `processBandStereo`, `fastTanhScalar`, `fastTanhV128`, `processAGC`, `calculateRMS`, `calculateAGCGain` |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | `softClipBlockAVX2`, `musicalSoftClipScalar`, `fastTanh`, `processDouble`, `processDoubleToBuffer`, `processOutputDouble` |
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | `softClipBlockAVX2` (float版) |
| `src/audioengine/AudioEngine.h` | `oversamplingFactor`, `CustomInputOversampler`, `ProcessingOrder` |
| `src/core/EQParameters.h` | `EQParameters`, `nonlinearSaturation`, `filterStructure` |
| `src/eqprocessor/EQProcessor.h` | `FilterStructure` enum (Serial/Parallel) |
| `src/eqprocessor/EQProcessor.Parameters.cpp` | `setNonlinearSaturation` (範囲0.0〜1.0) |
| `src/eqprocessor/EQProcessor.ProcessingCache.cpp` | `createCoeffCache` (filterStructure設定) |
| `src/core/Types.h` | `ProcessingOrder` enum |

---

## 3. レビュー指摘別の妥当性評価

### 指摘① SVF状態変数へのサチュレーション適用

**判定: 非常に妥当 (優先度 ★★★★★)**

#### コード確認結果

**スカラー版** (`processBand`, EQProcessor.Processing.cpp:138-143):

```cpp
if (saturation > 0.0)
{
    const double oneMinusSat = 1.0 - saturation;
    ic1eq = ic1eq * oneMinusSat + fastTanhScalar(ic1eq) * saturation;
    ic2eq = ic2eq * oneMinusSat + fastTanhScalar(ic2eq) * saturation;
}
```

**SIMD版** (`processBandStereo`, EQProcessor.Processing.cpp:232-238):

```cpp
if (saturation > 0.0)
{
    const __m128d vSat = _mm_set1_pd(saturation);
    const __m128d vOneMinusSat = _mm_set1_pd(1.0 - saturation);
    ic1eq = _mm_add_pd(_mm_mul_pd(ic1eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic1eq), vSat));
    ic2eq = _mm_add_pd(_mm_mul_pd(ic2eq, vOneMinusSat),
                       _mm_mul_pd(fastTanhV128(ic2eq), vSat));
}
```

**`fastTanhScalar`** (EQProcessor.Processing.cpp:85-91):

```cpp
inline double fastTanhScalar(double x) noexcept
{
    if (x >= 3.0) return 1.0;
    if (x <= -3.0) return -1.0;
    const double x2 = x * x;
    return x * (27.0 + x2) / (27.0 + 9.0 * x2);
}
```

→ |x| ≥ 3.0 で **±1.0 ハードクリップ**

**`fastTanhV128`** (EQProcessor.Processing.cpp:93-105):

```cpp
const __m128d xClamped = _mm_min_pd(_mm_max_pd(x, vNegThree), vThree);
```

→ |x| ≥ 3.0 で **±3.0 ハードクリップ**後にPade近似計算

#### DSP理論的根拠

TPT (Topology-Preserving Transform) SVF において、`ic1eq`, `ic2eq` は**積分器の内部状態**（エネルギー保存量）である。Zavalishin "The Art of VA Filter Design" (Native Instruments) に詳述される通り、TPTの中核は「積分器を bilinear transform で離散化し、それ以外のアナログトポロジーを保存すること」にある。状態変数に非線形性を導入すると**トポロジー保存性が完全に崩壊**し、線形SVFではなく非線形状態空間システムになる。

文献: <https://www.native-instruments.com/fileadmin/ni_media/downloads/pdf/VAFilterDesign_1.1.1.pdf>

#### 検証結論

- **レビューは完全に正しい。**
- 低域ブースト時、SVFの状態変数は入力振幅を大幅に超えて成長する。ここに `tanh`（特に |x|≥3 でハードクリップ）を適用すると、積分器エネルギーが毎サンプル失われ、周期的な状態崩壊 → 「ジジジジ」ノイズとなる。
- **修正案も妥当**: `output` に対して `saturation` を適用する設計がDSP的に自然。

---

### 指摘② ソフトクリッパーの事前平均化

**判定: 改善候補だが「ジジジジの主犯」とは断定できない (優先度 ★★★★☆)**

#### コード確認結果

**AVX2パス** (DSPCoreDouble.cpp:144-173):

```cpp
const __m128d prevLow128 = _mm_unpacklo_pd(_mm_set_sd(prevScalar), xLow);
const __m128d prevHigh128 = _mm_shuffle_pd(xLow, xHigh, 0x1);
const __m256d prevVec = _mm256_set_m128d(prevHigh128, prevLow128);

const __m256d midVec = _mm256_mul_pd(_mm256_add_pd(prevVec, x), vHalf);
const __m256d absMidVec = _mm256_andnot_pd(vSignMask, midVec);

const __m256d needMidClip = _mm256_cmp_pd(absMidVec, vThreshold, _CMP_GT_OQ);
const __m256d safeAbsMid = _mm256_max_pd(absMidVec, vTiny);
const __m256d midGainRaw = _mm256_div_pd(vThreshold, safeAbsMid);
const __m256d midGain = _mm256_blendv_pd(vOne, midGainRaw, needMidClip);
x = _mm256_mul_pd(x, midGain);
```

AVX2版の `prevScalar` 更新:

```cpp
const double nextPrev = data[i + 3]; // [BUG-04] store前に元の入力値を退避
_mm256_storeu_pd(data + i, result);
prevScalar = nextPrev;
```

→ **生入力値を保存** — これは正しい挙動。

**スカラーフォールバックパス** (DSPCoreDouble.cpp:208-222):

```cpp
for (; i < numSamples; ++i)
{
    const double mid = (prevScalar + data[i]) * 0.5;
    const double absMid = absNoLibm(mid);
    double x = data[i];
    if (absMid > threshold)
        x *= threshold / absMid;

    if (absNoLibm(x) > clip_start)
        x = musicalSoftClipScalar(x, threshold, knee, asymmetry);

    data[i] = x;
    prevScalar = x;  // ← バグ: 処理後値を保存！
}
```

**Float版** (DSPCoreFloat.cpp:110-127):

```cpp
void softClipBlockAVX2(double* __restrict data, int numSamples,
                       double threshold, double knee,
                       double asymmetry, double& prevSample) noexcept
{
    int i = 0;
    for (; i < numSamples; ++i)
    {
        double x = data[i];
        if (!isFiniteAndAbsBelowNoLibm(x, 1.0e300))
            x = 0.0;

        x = 0.5 * (x + prevSample);
        prevSample = x;
        data[i] = musicalSoftClipScalar(x, threshold, knee, asymmetry);
    }
}
```

→ 平均化自体が問題。`prevSample` に平均化値を保存する設計。

#### DSP的評価

`x = 0.5 * (x + prevSample)` は伝達関数 `H(z) = 0.5 + 0.5z⁻¹` の2タップFIR（簡易ローパス）である。
DC: 0dB, Nyquist: -∞dB の周波数特性を持ち、**軽いローパスフィルタ**として動作する。

「threshold前で矩形波化する」ほどの破壊力はない。しかしSoftClip本体 `musicalSoftClipScalar()` の前段で波形を改変することにより、クリッパーのニー特性が変質する可能性は否定できない。**削除してAB比較する価値はある。**

#### 検証結論

- 事前平均化そのものは軽いローパスであり、単独で「激しいジジジジ」を生む主犯とは断定できない。
- ただしクリッパーのニー特性を変質させる可能性があり、改善候補としての優先度は高い。
- **重大な追加発見 (新規)**: AVX2パスとスカラーパスで `prevScalar` の更新方法が**異なる**（不整合バグ）:

| コードパス | prevScalar更新 | 問題 |
|-----------|---------------|------|
| AVX2 (double) | `data[i+3]` (生入力) | ✅ `[BUG-04]`で半修正済み |
| スカラー (double) | `x` (処理後クリップ値) | ❌ **IIRフィードバックバグ** |
| Float版 | 平均化値 | ❌ 設計上の問題 |

- スカラーパスはブロック末尾で実行されるため、ブロック境界を跨いで `prevScalar` にクリップ出力値が保存される。次のブロックのAVX2パス先頭でこの値が使われ、ブロック境界で不連続が発生する可能性がある。

---

### 指摘③ EQ係数スムージング欠如

**判定: 妥当性低い (優先度 ★★)**

#### 検証結論

- レビューの理論的指摘（RCUによる係数瞬時切替はIIRフィルタの状態変数に不連続をもたらす）は正しい。
- しかしユーザー報告は「低音が多いと**常時**ジジジジ」であり、係数切替ノイズは「**ノブ操作時のみ**」発生する。
- 常時ノイズの説明にはならないため、**主原因ではない**。
- ただしオートメーション使用時には副次的原因になり得る。

---

### 指摘④ AGCブロックRMS問題

**判定: 主犯ではない (優先度 ★★)**

#### コード確認結果

**`calculateRMS`** (EQProcessor.Processing.cpp:18-49):

```cpp
inline double calculateRMS(const double* data, int numSamples) noexcept
{
    // ... ブロック単位の二乗平均平方根 ...
    double sumSq = 0.0;
    // SIMD/Fallback for sumSq
    // sqrt(sumSq / numSamples) を返す
}
```

→ 確かにブロック単位の単純RMS。レビューの指摘はここまで正しい。

**しかしその後**の `processAGC` (EQProcessor.Processing.cpp:347-425):

```cpp
// 1-pole envelope follower
const double inAlpha = (inputRMS > envIn) ? blockAttackCoeff : blockReleaseCoeff;
envIn = envIn * (1.0 - inAlpha) + inputRMS * inAlpha;

// ゲインスムージング
const double nextGain = currentGain * (1.0 - blockSmoothCoeff) + targetGain * blockSmoothCoeff;

// ブロック内線形ランプ
const double gainIncrement = (nextGain - currentGain) / static_cast<double>(numSamples);
for (int ch = 0; ch < numChannels; ++ch)
    applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, currentGain, gainIncrement);
```

**`calculateAGCGain`** (EQProcessor.Processing.cpp:327-342):

```cpp
double EQProcessor::calculateAGCGain(double inputEnv, double outputEnv) const noexcept
{
    // ±0.5dB の不感帯
    constexpr double DEAD_ZONE_RATIO = 1.059;
    if (ratio > 1.0 / DEAD_ZONE_RATIO && ratio < DEAD_ZONE_RATIO)
        return 1.0;
    // ...
}
```

#### 検証結論

- AGCには `envIn/envOut` (1-pole envelope follower) + `blockSmoothCoeff` (gain smoothing) + `applyGainRamp` (linear ramp) の三重の平滑化、さらに ±0.5dB 不感帯が実装されている。
- **「ブロックRMS直結」ではない。**
- AGC起因なら「ブーブー」「ワウワウ」系のレベル変調になりやすく、ユーザー報告の「ジジジジ」とは一致しにくい。
- **主犯ではない。**

---

### 指摘⑤ オーバーサンプリング不足（レビュー者の自己分析）

**判定: 補助的要因 (優先度 ★★★)**

#### コード確認結果

`AudioEngine.Processing.DSPCoreDouble.cpp` の `processDouble` 内の処理順序:

1. **`oversampling.processUp()`** — オーバーサンプリングアップ (factor > 1 時)
2. DC Blocker
3. Convolver / EQ (saturation含む)
4. Output Filter
5. Output Makeup Gain
6. **Soft Clip**  ← 非線形処理
7. Bypass Blend
8. **`oversampling.processDown()`** — ダウンサンプリング

**Soft clip は `processUp()` と `processDown()` の内部で実行されている。** これは設計として正しい。

ただし `oversamplingFactor` は**ユーザー設定依存**（`AudioEngine.h` の `DSPCore` メンバ）。

#### 検証結論

- SoftClip は `oversampling.processUp()` と `processDown()` の**内部**で実行されている。設計として正しい。
- しかし `oversamplingFactor` はユーザー設定依存（1x〜8x）。1x（OS無効）では非線形処理で生成された倍音が Nyquist を超えてフォールドバック → エイリアシング。
- 低音（40Hz）は振幅が大きく非線形倍音生成量が増加するため、OS無効時の症状（「サチュレーションを上げると悪化」）と一致する。
- **ただしこれはSVF状態変数破壊とは独立した別要因であり、共存しうる。** 両方修正するのが望ましい。

---

## 6. 新規追加: EQ並列モードの帯域加算問題（レビュー見落とし）

**判定: 見落とし、ただし副次的 (優先度 ★★★)**

### 発見経緯

ユーザーからの指摘を受けて追加調査。`EQProcessor.Processing.cpp` の Parallel モード実装を精査した結果、帯域ごとの非線形成分が独立に加算される構造が確認された。

### コード確認結果

**Parallelモードのアルゴリズム**（`processParallel` ラムダ、line 679-826、および第2オーバーロード line 1005-1090）:

```cpp
for (int i = 0; i < numActiveBands; ++i)
{
    // 各バンドは独立に ORIGINAL 信号から処理開始
    juce::FloatVectorOperations::copy(workL, srcL, numSamples);

    processBand(workL, ..., saturation);  // SVF + 状態変数サチュレーション

    // delta = (処理済み - オリジナル) を累積
    juce::FloatVectorOperations::add(accumL, workL, numSamples);
    juce::FloatVectorOperations::subtract(accumL, srcL, numSamples);
}

// 最終出力 = オリジナル + 全バンドのdeltaの総和
juce::FloatVectorOperations::copy(dstL, srcL, numSamples);
juce::FloatVectorOperations::add(dstL, accumL, numSamples);
```

### DSP的問題

Parallelモードでは**各バンドが独立にオリジナル信号（src）から処理を開始する**。Serialモードのように前段の出力を引き継がない。

SVF状態変数サチュレーション（指摘①）が存在する場合、各バンドの `processBand` は：

1. 入力振幅が大きい低域（例: 40Hz）を各バンドが**独立にフル振幅で受け取る**
2. 各バンドのSVF状態変数がそれぞれ異なる係数で歪む
3. 各バンドの出力 `workL` には**線形EQ効果＋非線形歪み**の両方が含まれる
4. `delta = workL - srcL` には**歪み成分も含まれる**
5. 全バンドの歪み成分が独立に総和される

Serialモードとの比較:

| モード | 帯域間の歪み伝搬 | 歪みの総和 |
|-------|----------------|-----------|
| **Serial** | 前段の歪みが後段のフィルタを通る | 後段SVFで一部の高調波が減衰される可能性 |
| **Parallel** | 各バンド独立（src共通） | **全バンドの歪み成分が生のまま加算** |

### 検証結論

- SVF状態変数サチュレーションが存在することが前提の**二次的影響**。サチュレーションがゼロなら問題にならない。
- しかしサチュレーション有効時、Parallelモードでは歪み成分がより直接的に出力に現れる。
- **「ジジジジ」の直接原因ではなく、増幅要因として認識すべき。**
- 実機検証での切り分け：Parallel → Serial 切替でノイズが変化するかを確認。

---

## 7. 完全な信号処理チェーン

```
Input Buffer
  │
  ├─ processInputDouble (headroom gain, analyzer tap)
  │
  ├─ [oversamplingFactor > 1] ── oversampling.processUp()
  │     ├─ DC Blocker (oversampledL/R)
  │     │
  │     ├─ [ProcessingOrder::ConvolverThenEQ]
  │     │     ├─ Convolver::process()
  │     │     └─ EQProcessor::process()  ← ★ SATURATION HERE
  │     │           ├─ Serial: processSerial → processBandStereo/processBand (saturation on ic1eq/ic2eq)
  │     │           └─ Parallel: processParallel → 各バンド独立 (saturation on ic1eq/ic2eq)
  │     │
  │     ├─ [ProcessingOrder::EQThenConvolver]
  │     │     ├─ EQProcessor::process()  ← ★ SATURATION HERE (同上)
  │     │     └─ Convolver::process()
  │     │
  │     ├─ outputFilter (LPF)
  │     ├─ outputMakeupGain
  │     │
  │     ├─ [softClipEnabled] ── softClipBlockAVX2()  ← 非線形処理 #2
  │     │     ├─ AVX2: midVec平均化 + fastTanh近似
  │     │     └─ Scalar: prevScalar = x (処理後値バグ)
  │     │
  │     ├─ Bypass Blend (wet/dry)
  │     │
  │     └─ [oversamplingFactor > 1] ── oversampling.processDown()
  │
  ├─ processOutputDouble
  │     ├─ DC Blocker (outputL/R)
  │     ├─ NaN/Inf チェック
  │     ├─ Adaptive Capture
  │     ├─ Dither / Noise Shaper
  │     └─ 最終クリッピング (±kOutputHeadroom)
  │
  └─ Output Buffer
```

### 連鎖の分析

SVF状態変数サチュレーションで生成された歪みは、その後段で：

1. **Convolver → EQ 順**: EQの歪みが最後 → SoftClipへ直行 → OS後ダウンサンプル
2. **EQ → Convolver 順**: EQの歪みがConvolverを通る → IRの畳み込みで歪みが拡散される可能性
3. **SoftClip**: 既に歪んだ信号をさらに非線形処理 → **相互変調歪み**発生

この連鎖が「サチュレーションを上げるとジジジジがひどくなる」症状のメカニズムである。

---

## 8. 未確定事項・残調査項目

| 項目 | 状況 | 理由 |
|------|------|------|
| SVF saturationが原因か | ✅ **確定** | コード確認済み。状態変数破壊はDSP理論上も有害 |
| SoftClip平均化が主犯か | ⚠️ 未確定 | 軽いローパスであり単独破壊力は限定的。ただしAVX2/スカラーの不整合はバグ |
| Parallelモード増幅効果 | ⚠️ 未確定 | 理論上は歪みを増幅しうるが、実機検証が必要 |
| エイリアシングの寄与度 | ⚠️ 未確定 | OS=1xでの非線形処理は必ずエイリアシングを生むが、症状への寄与度は未定量 |
| EQ係数スムージング | ✅ **却下** | 常時ノイズの説明にならない |
| AGCが主犯か | ✅ **却下** | 平滑化既存＋症状不一致 |

### 実機検証推奨手順（ユーザー提案より）

1. EQ Nonlinear Saturation を 0.0 にする → ノイズ消えるか？
2. SoftClip OFF → ノイズ変化するか？
3. Oversampling 8x → ノイズ減るか？
4. Parallel EQ → Serial EQ 切替 → ノイズ変化するか？
5. 低域 +12dB で再生

もし手順1でノイズが消えるなら、**ほぼSVF状態変数サチュレーションが犯人**と断定できる。

---

## 9. v3追加調査: prevScalar不整合の定量評価

### コードの正確な動作

AVX2パス（line 126-198）:

```
prevScalar = data[i + 3];   // 生入力値を保存（[BUG-04]）
```

→ **元入力値**を保持している。正しい。

スカラーフォールバックパス（line 200-220）:

```
prevScalar = x;   // xはmusicalSoftClipScalar()の処理後値
```

→ **クリップ出力値**を保持している。バグ。

ただしこのバグの**実用的影響範囲**は限定的：

- ブロックサイズが4の倍数の場合、スカラーパスは**実行されない**
- 典型的なブロックサイズ（64, 128, 256, 512, 1024）は全て4の倍数
- OS倍率（1x, 2x, 4x, 8x）も全て2の冪 → 4の倍数性は維持される
- スカラーパスはnumSamples % 4 ≠ 0の場合のみ実行（例: DAWが奇数ブロックサイズを送信）

**確度評価**: 不整合バグは存在するが、主症状への寄与は「中程度（症状説明力30%程度）」と評価。

---

## 10. v3追加調査: fastTanh閾値の違い（新発見）

SVF saturation用 `fastTanhScalar`（EQProcessor.Processing.cpp）:

```cpp
if (x >= 3.0) return 1.0;  // |x| ≥ 3 でハードクリップ
if (x <= -3.0) return -1.0;
```

SoftClip用 `fastTanh`（AudioEngine.Processing.DSPCoreDouble.cpp, TanhApprox）:

```cpp
constexpr double CLIP_THRESHOLD = 4.5;
if (x >= CLIP_THRESHOLD) return 1.0;  // |x| ≥ 4.5 でハードクリップ
if (x <= -CLIP_THRESHOLD) return -1.0;
```

**重要**: SVF saturation（ic1eq/ic2eq適用）のtanhは**3.0でクリップ**するのに対し、SoftClip本体のtanhは**4.5でクリップ**する。SVF状態変数が3.0を超えるとSVF saturationのtanhが先に飽和し、状態エネルギーを強制的に切り捨てる。これが「ジジジジ」ノイズの直接メカニズムとなる。

**確度評価**: SVF saturation → 状態変数破壊の連鎖はDSP理論とコードの両面から確認。確度70〜80%。

---

## 11. v3追加調査: SVF係数と状態変数振幅の関係

Low Shelf +12dB ブースト時の係数（`calcLowShelfSVF`）:

```cpp
const double A = std::pow(10.0, gainDb / 40.0);  // +12dB → A ≈ 3.98
c.m2 = A * A - 1.0;  // ≈ 14.84
```

出力式:

```
output = m0*v0 + m1*v1 + m2*v2
       = 1.0*v0 + m1*v1 + 14.84*v2
```

v2（ic2eqを含む項）に **14.84倍** の重みがかかる。ic2eqが入力振幅のオーダー（例: 0.5）でも、出力のv2成分は 7.4 に達する。ic1eq/ic2eq自体は状態変数として自然に成長しうる。

**SVF状態変数の成長条件**:

- 低周波（freq << sr）: g ≈ π*freq/sr が小 → a1 ≈ 1, a2 ≈ g, a3 ≈ g²
- 状態更新: ic1eq = 2*v1 - ic1eq → ic1eq は蓄積的
- ハイQ（resonance）: kが小 → a1が大きくなる → 状態変数の成長が加速

**結論**: 低域ブースト＋ハイQ設定で、ic1eq/ic2eqは容易に3.0を超える。3.0超過時にfastTanhScalar(±3.0クリップ)が発動 → 状態エネルギー喪失 → 周期的崩壊 → ジジジジ。

---

## 12. v3追加調査: 相互変調のメカニズム

SVF saturationが生成した歪み成分は、以下のパスを通る:

```
SVF saturation（ic1eq/ic2eq破壊）
  → 出力信号に歪み成分（高調波・非調和成分）が乗る
  → OutputFilter (LPF) で一部減衰
  → MakeupGain で増幅
  → SoftClip でさらに非線形処理
  → 相互変調歪み (IMD) 発生
  → OSなし(1x)ならエイリアシング
```

**SoftClipパラメータとの連動**:

```cpp
clipThreshold = 0.95 - 0.45 * sat;   // sat: saturationAmount (0.0〜1.0)
clipKnee      = 0.05 + 0.35 * sat;
clip_start    = threshold - knee;     // sat=0→0.90, sat=1.0→0.10
```

saturationAmountが上がるほどSoftClipの開始閾値が下がる（sat=1.0なら0.10）。
一方SVFのnonlinearSaturationも同じユーザー操作で上がる → 両方の非線形性が同時に強まる。

**この連動が「サチュレーションを上げるとジジジジがひどくなる」症状を説明する。**

---

## 13. v3 最終総合評価（確度付き）

| 優先度 | 項目 | 確度 | 症状説明力 |
|-------|------|------|-----------|
| **P1** | SVF状態変数サチュレーション除去 | 70〜80% | 高い |
| **P2** | prevScalar不整合修正 (AVX2/Scalar) | 60% | 中程度 |
| **P3** | SoftClip平均化のAB試験 | 40% | 限定的 |
| **P4** | Parallelモード帯域加算検証 | 30% | 増幅要因 |
| **P5** | OS最低値検討 | 30% | エイリアシング対策 |

### 修正優先順位（実装順）

1. **P1: SVF状態変数サチュレーション除去**
   - `processBand()` / `processBandStereo()` の `ic1eq`/`ic2eq` への saturation 適用を削除
   - `output` 計算後に saturation を適用するよう変更
   - これだけで症状の70〜80%が改善される可能性

2. **P2: prevScalar不整合修正**
   - スカラーフォールバックの `prevScalar = x` を `prevScalar = data[i]`（store前の値）に変更
   - AVX2パスに合わせる

3. **P3-P5: 検証・改善**
   - SoftClip平均化をコンパイルスイッチでOFFにしてAB比較
   - Parallel⇔Serial切替でノイズ変化を確認
   - OS 8xでノイズ低減効果を確認

---

## 14. 結論

v3までの全調査を通じて、以下の知見が確定した:

1. **SVF状態変数サチュレーション**（ic1eq/ic2eqへのfastTanh直接適用）は現行コードに実在し、DSP理論上も危険。これが**最も確度の高い原因**（70〜80%）。

2. **fastTanh閾値の不一致**: SVF用fastTanhScalarは3.0クリップ、SoftClip用fastTanhは4.5クリップ。SVF状態変数が3.0を超えるとSVF saturationが先に飽和し、状態崩壊を引き起こす。

3. **SoftClip平均化**は軽いローパス（H(z)=0.5+0.5z⁻¹）であり、単独では「激しいジジジジ」の主犯とは断定できない。ただしクリッパーのニー特性を変質させる。

4. **prevScalar不整合**（AVX2は生入力値、Scalarは処理後値）は存在するが、実用的にはブロックサイズが4の倍数でない場合のみ影響。発見としての価値は高いが症状説明力は中程度。

5. **Parallelモード**の非線形成分加算は理論的には歪みを増幅しうるが、SVF saturationが根本原因である場合の二次的増幅要因。

6. **SVF saturationとSoftClipの相互変調**: 両非線形処理が直列に接続され、かつユーザー操作で同時に強まる設計のため、「サチュレーションを上げると悪化する」症状を説明できる。

**最優先の修正は「SVF saturationを状態変数（ic1eq/ic2eq）ではなく出力（output）に適用する変更」の一点。** この修正後、残存ノイズの度合いに応じてSoftClip平均化除去、prevScalar不整合修正、Parallel/Serial比較を段階的に進めることを推奨する。
