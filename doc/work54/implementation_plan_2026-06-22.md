# ConvoPeq 音質改善 詳細改修計画書

**策定日**: 2026-06-22 (v5 最終版)
**対象バージョン**: ConvoPeq v4.21 (ワーキングツリー)
**前提**: ISR Bridge Runtime 完成度95%超（本改修はこれを壊さない範囲で実施）

---

# 第1部: 実装必須情報

---

## Q1. クイックリファレンス

| Phase | 項目 | 優先度 | 工期 | 新規ファイル | 修正ファイル | 主要リスク |
|-------|------|--------|------|-------------|-------------|-----------|
| 1 | SoftClip改善 | **S** | 3〜5人日 | 0 | 5 | prepareStage privateの公開 |
| 2 | Mid/Side EQ | **S** | 5〜8人日 | 0 | 6 | filterState拡張+周辺11箇所監査 |
| 3 | True Peak + LUFS Meter | **A** | 5〜8人日 | 4 | 3 | —（K-weighting/TP tap数は確定済） |
| 4 | True Peak Limiter | **C** | 3〜5人日 | 1 | 2 | Phase 3完了後判断 |
| 5 | ADAA | **D** | 5〜8人日 | 0 | 2 | Phase 1-4完了後判断 |

### 全Phase共通ルール

- ISR Bridge Runtime 非改修
- Audio Thread: libm禁止、メモリ確保ゼロ、ロックフリー
- AVX2/SSE2/スカラー全パス同一出力保証
- `isLinearPhaseFIR = true` 不変（既存static_assert維持）

---

## Q2. Phase 1: SoftClip改善（優先度S）

### 2A. 変更内容（実装手順）

#### Step 1: AVX2/スカラーパス統一

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp`
**関数**: `softClipBlockAVX2()` スカラーフォールバック部

```cpp
// 削除する3行:
const double mid    = (prevScalar + inputVal) * 0.5;
const double absMid = absNoLibm(mid);
if (absMid > threshold) x *= threshold / absMid;

// 維持:
double x = inputVal;
if (absNoLibm(x) > clip_start)
    x = musicalSoftClipScalar(x, threshold, knee, asymmetry);
data[i] = x;
prevScalar = inputVal;
```

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
**関数**: `softClipBlockAVX2()` 全体

```cpp
// 削除: const double avg = 0.5 * (x + prevSample);
// 変更: data[i] = musicalSoftClipScalar(x, threshold, knee, asymmetry);
// 維持: prevSample = x; // 状態更新のみ（ADAA用にフィールド残す）
```

#### Step 2: 局所2倍OS追加

**ファイル**: `src/CustomInputOversampler.h` — 公開メソッド追加

```cpp
public:
    // 単一stageの軽量オーバーサンプラを構築（SoftClip専用）
    bool prepareSingleStage(int taps, double attenDb, int stageInputMax) noexcept;
```

**ファイル**: `src/CustomInputOversampler.cpp` — 実装追加

```cpp
bool CustomInputOversampler::prepareSingleStage(int taps, double attenDb, int stageInputMax) noexcept
{
    release();
    upsampleRatio = 2;
    numStages = 1;
    maxInputBlockSize = stageInputMax;
    maxUpsampledBlockSize = stageInputMax * 2;
    prepareStage(stages[0], taps, attenDb, stageInputMax);
    workCapacity = maxUpsampledBlockSize;
    for (int ch = 0; ch < kMaxChannels; ++ch) {
        workA[ch] = convo::makeAlignedArray<double>(static_cast<size_t>(workCapacity));
        workB[ch] = convo::makeAlignedArray<double>(static_cast<size_t>(workCapacity));
        if (workA[ch]) juce::FloatVectorOperations::clear(workA[ch].get(), workCapacity);
        if (workB[ch]) juce::FloatVectorOperations::clear(workB[ch].get(), workCapacity);
        blockChannels[ch] = workA[ch].get();
    }
    return true;
}
```

**ファイル**: `src/audioengine/AudioEngine.h` — DSPCoreメンバ追加

```cpp
CustomInputOversampler softClipOS;
```

**ファイル**: `src/audioengine/AudioEngine.Processing.Lifecycle.cpp` — prepare

```cpp
// DSPCore::prepare() 内:
softClipOS.prepareSingleStage(31, 90.0, maxInternalBlockSize);
```

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` — process()内

```cpp
if (state.softClipEnabled) {
    if (oversamplingFactor > 1) {
        for (int ch = 0; ch < nCh; ++ch)
            softClipBlockAVX2(block.getChannelPointer(ch), numProcSamples, ...);
    } else {
        auto osBlock = softClipOS.processUp(originalBlock, nCh);
        for (int ch = 0; ch < nCh; ++ch)
            softClipBlockAVX2(osBlock.getChannelPointer(ch), (int)osBlock.getNumSamples(), ...);
        softClipOS.processDown(osBlock, originalBlock, nCh);
    }
}
```

**ファイル**: `src/audioengine/AudioEngine.h` — LatencyBreakdown

```cpp
int softClipLatencyBaseRateSamples = 0;  // 新規フィールド
```

**レイテンシ注意**: CustomInputOversampler の interpolateStage / decimateStage はポリフェーズ実装（centerTap, centerDelayInput, convParity）だが、線形位相FIRの群遅延は `(taps-1)/2` が厳密な公式（文献確定: DSP理論）。

- 31tap halfband → `(31-1)/2 = 15 samples @ 2x rate`
- base rate換算: `15 / 2 = 7.5 samples` (processUp) + 7.5 samples (processDown) = **合計15 base rate samples = 0.3125ms @ 48kHz**
- ポリフェーズ実装は畳み込み演算の実装詳細であり、フィルタの線形位相特性と群遅延に**影響しない**
- `getCurrentLatencyBreakdown()` で `softClipLatencyBaseRateSamples` を `totalLatencyBaseRateSamples` に合算し、PDC報告に含める
- 理論値は確定だが、実装後に単位インパルス応答で**実測確認**を推奨

### 2B. テスト項目

1. AVX2パスとスカラーフォールバックの出力一致（4倍数＋余剰）
2. Float版とDouble版の出力一致
3. OS=1: ナイキスト折り返し低減を周波数スペクトル確認
4. OS=8: 局所OS未挿入、既存動作と完全一致
5. CPU負荷: OS=1時の実機実測で確認（31tap Halfband×2のMAC見積もり: convCount=15, dot product 30回/ブロック @ 64 samples = 960MAC ≪ 総CPUの1%）

---

## Q3. Phase 2: Mid/Side EQ（優先度S）

### 3A. 変更内容（実装手順）

#### Step 1: filterState拡張

**ファイル**: `src/eqprocessor/EQProcessor.h`

```cpp
static constexpr int kFilterChannels = 4; // L=0, R=1, Mid=2, Side=3
std::array<std::array<std::array<double, 2>, NUM_BANDS>, kFilterChannels> filterState{};
```

**連動修正**: `EQProcessor.Processing.cpp:980`

```cpp
// 変更前: for (int ch = 0; ch < MAX_CHANNELS; ++ch)
for (int ch = 0; ch < EQProcessor::kFilterChannels; ++ch)
    std::memset(activeFilterState[ch][i].data(), 0, sizeof(double) * 2);
```

残り9箇所は自動対応のため修正不要（全11箇所監査済み）。

#### Step 2: EQChannelMode enum拡張

**ファイル**: `src/eqprocessor/EQProcessor.h:54-59`

```cpp
enum class EQChannelMode { Stereo, Left, Right, Mid, Side };
```

#### Step 3: Serialモード — processSerialラムダ内にM/S分岐追加

**ファイル**: `src/eqprocessor/EQProcessor.Processing.cpp`

```cpp
case EQChannelMode::Mid: {
    // MとSをエンコード（msWork[0..n]=M, [n..2n]=S）
    juce::FloatVectorOperations::copy(msWork, dataL, numSamples);
    juce::FloatVectorOperations::add(msWork, dataR, numSamples);
    scaleBlockFallback(msWork, numSamples, 0.5);
    juce::FloatVectorOperations::copy(msWork + numSamples, dataL, numSamples);
    juce::FloatVectorOperations::subtract(msWork + numSamples, dataR, numSamples);
    scaleBlockFallback(msWork + numSamples, numSamples, 0.5);
    // Mid成分のみ処理
    processBand(msWork, numSamples, band.node->coeffs,
                states[2][band.index].data(), saturation);
    // デコード: L=M+S, R=M-S
    for (int n = 0; n < numSamples; ++n) {
        dataL[n] = msWork[n] + msWork[numSamples + n];
        dataR[n] = msWork[n] - msWork[numSamples + n];
    }
    break;
}
case EQChannelMode::Side: {
    // 同上、ただし processBand は msWork[n..2n] (S成分) に対して実行
    // ※ M/Sエンコード部は Mid と同じ（両方必要）
    juce::FloatVectorOperations::copy(msWork, dataL, numSamples);
    juce::FloatVectorOperations::add(msWork, dataR, numSamples);
    scaleBlockFallback(msWork, numSamples, 0.5);
    juce::FloatVectorOperations::copy(msWork + numSamples, dataL, numSamples);
    juce::FloatVectorOperations::subtract(msWork + numSamples, dataR, numSamples);
    scaleBlockFallback(msWork + numSamples, numSamples, 0.5);
    // Side成分のみ処理
    processBand(msWork + numSamples, numSamples, band.node->coeffs,
                states[3][band.index].data(), saturation);
    // デコード: L=M+S, R=M-S
    for (int n = 0; n < numSamples; ++n) {
        dataL[n] = msWork[n] + msWork[numSamples + n];
        dataR[n] = msWork[n] - msWork[numSamples + n];
    }
    break;
}
```

#### Step 4: Parallelモード — processParallelラムダ内にM/S分岐追加【設計確定】

現行の `accum += work - src` 方式を維持。M/Sバンドの処理は以下の流れ:

```cpp
case EQChannelMode::Mid:
case EQChannelMode::Side: {
    // ① M と S をエンコード → workM, workS (msWork[0..n]=M, [n..2n]=S)
    juce::FloatVectorOperations::copy(msWork, srcL, numSamples);
    juce::FloatVectorOperations::add(msWork, srcR, numSamples);
    scaleBlockFallback(msWork, numSamples, 0.5);                    // M
    juce::FloatVectorOperations::copy(msWork + numSamples, srcL, numSamples);
    juce::FloatVectorOperations::subtract(msWork + numSamples, srcR, numSamples);
    scaleBlockFallback(msWork + numSamples, numSamples, 0.5);        // S

    // ② 処理: MidならmsWork[0..n], SideならmsWork[n..2n]
    auto* targetState = (mode == Mid) ? states[2][idx] : states[3][idx];
    auto* targetBuf   = (mode == Mid) ? msWork : (msWork + numSamples);
    processBand(targetBuf, numSamples, coeffs, targetState, saturation);

    // ③ デコード → workL, workR へ
    for (int n = 0; n < numSamples; ++n) {
        const double m  = msWork[n];
        const double sp = msWork[numSamples + n];
        workL[n] = m + sp;
        workR[n] = m - sp;
    }
    // ④ 差分加算: accumL += (workL - srcL), accumR += (workR - srcR)
    for (int n = 0; n < numSamples; ++n) {
        accumL[n] += workL[n] - srcL[n];
        accumR[n] += workR[n] - srcR[n];
    }
    break;
}
```

#### Step 5: M/S用スクラッチバッファ（容量再設計）

**ファイル**: `src/eqprocessor/EQProcessor.h`

```cpp
// ★ サイズは numSamples * 4（M+S+workL+workR の4領域を確保）
//    Serialでは numSamples*2 で十分だが、Parallelでも共通バッファとして使うため
convo::ScopedAlignedPtr<double> msWorkBuffer;  // サイズ: maxInternalBlockSize * 4
int msWorkCapacity = 0;
```

**ファイル**: `src/eqprocessor/EQProcessor.Core.cpp` prepareToPlay内

```cpp
const int requiredMS = maxInternalBlockSize * 4; // M+S+workL+workR
if (requiredMS > msWorkCapacity || !msWorkBuffer) {
    msWorkBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(requiredMS));
    msWorkCapacity = requiredMS;
}
```

#### Step 5: SpectrumAnalyzer — 独立Mid/Sideトレース【確定事項反映】

**ファイル**: `src/SpectrumAnalyzerComponent.cpp`

SpectrumAnalyzerのMid/Side対応は以下の4関数が対象。全経路監査済み【確定】:

| 関数 | 行 | 現状 | 修正 |
|------|-----|------|------|
| `updateEQData()` | ~863 | `curvesL`/`R` のみ | **`curvesMid`/`curvesSide` 追加** |
| `updateEQPaths()` | ~937 | L/R Path生成 | Mid/Side Path追加（cyan/magenta） |
| `paint()` | ~1036 | L/R+bandDot描画 | Mid/Sideトレース+ドット描画追加 |
| bandDot描画 | ~1176 | 3種ドットのみ | 5種（+Mid cyan, +Side magenta） |

```cpp
// 新規バッファ:
std::array<std::array<float, NUM_DISPLAY_BARS>, NUM_BANDS> individualBandCurvesMid{};
std::array<std::array<float, NUM_DISPLAY_BARS>, NUM_BANDS> individualBandCurvesSide{};

// 分岐追加（line 872-873）:
case Mid:  curvesMid[b][i]=db; break;   // cyan
case Side: curvesSide[b][i]=db; break;  // magenta
```

#### Step 6: calcEQResponseCurve() 分岐修正【確定事項反映】

**ファイル**: `src/audioengine/AudioEngine.EQResponse.cpp` (line 155-211)

現状の分岐には `else { // Right }` の暗黙デフォルトが存在。Mid/Side追加時に誤動作する。

```cpp
// 修正前:
if (band.mode == EQChannelMode::Stereo) { /* L,R両方 */ }
else if (band.mode == EQChannelMode::Left) { /* Lのみ */ }
else { /* Right: Rのみ */ }  // ★ Mid/Sideがここに落ちる

// 修正後:
if (band.mode == EQChannelMode::Stereo ||
    band.mode == EQChannelMode::Mid ||
    band.mode == EQChannelMode::Side) {
    // L,R両方に乗算（Mid/Sideは両chに影響）
} else if (band.mode == EQChannelMode::Left) {
    // Lのみ
} else { // Right
    // Rのみ
}
```

**注意**: Mid/Sideモードの周波数応答曲線はL/R両チャンネルに影響するため、Stereoと同様の両ch乗算で正しい。

#### Step 7: Mono入力時 分岐ガード【確定事項反映】

**ファイル**: `src/eqprocessor/EQProcessor.Processing.cpp` — processSerial/Parallelラムダ内

```cpp
case EQChannelMode::Mid:
    if (numChannels < 2) {
        processBand(dataL, numSamples, ..., states[2][...], saturation);
        juce::FloatVectorOperations::copy(dataR, dataL, numSamples); // R=M
        break;
    }
    // ...通常のM/S処理（M成分のみprocessBand）...
    break;

case EQChannelMode::Side:
    if (numChannels < 2) {
        juce::FloatVectorOperations::clear(dataL, numSamples); // Side=0
        break;
    }
    // ...通常のM/S処理（S成分のみprocessBand）...
    break;
```

| 条件 | Mid | Side |
|------|-----|------|
| Stereo入力 (ch==2) | `M=(L+R)/2` 通常処理 | `S=(L-R)/2` 通常処理 |
| Mono入力 (ch==1) | `dataL` そのまま処理、`dataR=dataL` | **dataL=0 出力**（L-R定義不可） |

#### Step 8: クロスフェード時のstateコピー確認【確定】

**ファイル**: `src/eqprocessor/EQProcessor.Processing.cpp` — 構造体切り替え部

```cpp
auto oldStateSnapshot = activeFilterState;  // C++配列全体のautoコピー（640→1280bytes）
```

- コピー量: 640 bytes → 1,280 bytes（2倍）
- 1回のクロスフェードに数命令追加のみ = **性能影響なし**
- ABI変化もなし

### 3B. テスト項目

1. Band1(Mid)→Band2(Side): L=1,R=-1入力→出力一致
2. 全モード混在 → Serial/Parallel正常動作
3. Mono Fold: Side成分キャンセル確認
4. Analyzer: Mid=cyan, Side=magenta 独立表示
5. calcEQResponseCurve: Mid/Sideモードで両ch応答曲線描画確認
6. Mono Mid: dataLそのまま、dataR=dataL確認
7. Mono Side: dataL=0確認
8. 既存 Stereo/Left/Right 後方互換

---

## Q4. Phase 3: True Peak + LUFS Meter（優先度A）

### 4A. 新規ファイル

| ファイル | 内容 | 規模 |
|---------|------|------|
| `src/TruePeakDetector.h/cpp` | **新規**: 4倍補間+peak detect（計測専用、ゲイン演算なし） | ~120行 |
| `src/LoudnessMeter.h/cpp` | **新規**: K-weighting+block power→RingBuffer publish | ~250行 |

### 4B. 修正ファイル

| ファイル | 変更 | 規模 |
|---------|------|------|
| `src/audioengine/AudioEngine.h` | DSPCoreにメンバ追加 | ~15行 |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | processOutputDouble内呼出 | ~20行 |
| `src/audioengine/AudioEngine.Processing.Lifecycle.cpp` | prepare/reset | ~15行 |

### 4C. 実装の要点

- **TruePeak補間**: Phase 1とは別インスタンス。tap数は63tapで確定（BS.1770-3例の48tapを上回る）
- **K-weighting**: BS.1770-4 Annex 1 規定係数を直接実装。RBJ Cookbook流用不可
- **LUFS Integrated集計**: Audio Threadはブロック平均電力のみ publish。集計は**専用ワーカースレッド**で行う（UIタイマー依存は不可）
- **LUFSゲーティング詳細【文献確定: ITU-R BS.1770-4 + EBU R128】**:
  - Momentary: 400ms矩形窓、チャンネル重み適用後にブロック平均電力を積算
  - Short-term: 3000ms矩形窓、75%オーバーラップ
  - Integrated: 2-pass gating
    1. 絶対ゲート (−70 LUFS) で全ブロックをフィルタ → 平均 L_abs
    2. 相対ゲート (L_abs − 10 LU) で再度フィルタ → 最終 Integrated Loudness L_Int
  - チャンネル重み: L=1.0, R=1.0（ステレオ2chの場合）

---

## Q5. Phase 4: True Peak Limiter（優先度C）

Phase 3完了後着手判断。`TruePeakDetector` 内包 + ルックアヘッド遅延線（**5ms固定、ただし0.5〜5msで設定可能にすることが望ましい**）＋ゲインコンピュータ。`src/TruePeakLimiter.h/cpp` 新規。

```cpp
// Lookahead: デフォルト5ms、0.5〜5msで設定可能【文献確定】
// マスタリング用0.5〜2ms（透明性重視）／放送用2〜5ms（安全策重視）
int lookaheadSamples = 0;
void prepare(double sampleRate, int maxBlockSize) {
    lookaheadSamples = static_cast<int>(sampleRate * 0.005); // 5ms default
    // 内部遅延線サイズ = lookaheadSamples + maxBlockSize
}
```

---

## Q6. Phase 5: ADAA（優先度D）

Phase 1-4完了後、実機ABX評価を経て着手判断。不定積分 `F(x)=x²/18+(4/3)·ln(x²+3)` はdilogarithm不要で閉形式。libm回避はAVX2多項式近似logを自作。

**重要: ADAA式の適用範囲【確定】**

`F(x)=x²/18+(4/3)·ln(x²+3)` は **EQバンドサチュレーション** の `fastTanhScalarOutput(x)=x*(27+x²)/(27+9x²)`（Padé(3,2)近似）に対する不定積分である。

`musicalSoftClipScalar()`（smoothstep + tanh近似 + asymmetry の合成関数）には**適用できない**。ADAAはEQ処理の一部として実装し、SoftClip（Phase 1）とは独立した別機能。

---

# 第2部: 補足情報

---

## S1. 設計決定の根拠

### S1.1 なぜSoftClipが最優先か

現状の `softClipBlockAVX2` は **AVX2本体 ≠ AVX2余りサンプル ≠ Float版** の三重不整合状態にある。非線形処理前の1tap FIR（`avg = 0.5*(x+prevSample)`）が意図しない音色変化・周波数応答変化を生んでいる。修正により全パスで一貫した出力が得られるため、即座に音質改善に直結する。

### S1.2 なぜADAAが後回しか

ConvoPeqのOS設定は `48kHz→8x, 96kHz→8x, 192kHz→4x` と既に高OS環境がデフォルト。ADAAの効果は `OS=1 + 強サチュレーション + 高域入力` の限定条件でしか発揮されず、Phase 1のSoftClip改善＋局所OSでエイリアシングの大部分は解決する。

### S1.3 各Phaseの不採用/格下げ理由

| 項目 | 優先度 | 理由 |
|------|--------|------|
| Garcia最適化 | E | 改善幅は机上推定。CPUプロファイル未実施 |
| Dynamic EQ | E | 実質「製品カテゴリ変更」。15〜20人日 |
| 最小位相OS | E | static_assertで禁止。PDC全面改修 |
| Air Absorption | E | ABX不能レベル |
| ISO226 JND | E | ATH+A重み+Barkで十分 |

---

## S2. 実装上の注意点

### S2.1 prepareSingleStageの実装量

`prepare()` は release, upsampleRatio, numStages, workCapacity, workA/B, blockChannels の全設定が必要。実装量は **30行ではなく80〜150行**。

### S2.2 SoftClip局所OSのレイテンシ

31tapハーフバンドの群遅延 = 15 samples = 0.3125ms（up+down合計30samples=0.625ms@48kHz）。Analyzer/AdaptiveCaptureは全信号がSoftClip経由のため相対差なし。LatencyBreakdownに `softClipLatencyBaseRateSamples` フィールド追加推奨。

### S2.3 TruePeak補間tap数【Web文献検証済み・確定】

ITU-R BS.1770-4 Annex 2 は「4倍オーバーサンプリング」を規定するが、FIR長は規定しない。Hansen (2012) の比較論文ではITU-R BS.1770-3のExample filterは**48tap**である。

| ソース | tap数 | 備考 |
|--------|-------|------|
| ITU-R BS.1770-3 Example (Hansen 2012) | **48** | BS.1770-3 Annexに記載のExample filter |
| DK-Technologies (Hansen 2012) | 128 | 高精度版、偏差±0.015dB |
| essentia TruePeakDetector | 実装依存 | 4x oversample + internal lowpass |
| 業務用TPメータ一般 | 63〜127 | 63tapで十分 |

**確定**: 63tapはITU例（48tap）を上回り、BS.1770-4/5準拠に十分な精度を持つ。31/63/127tapの比較テストは実装後検証として実施するが、63tapで問題ない。

### S2.4 K-weighting【Web文献検証済み、係数確定】

BS.1770-4のK-weightingは規定係数。RBJ Cookbook近似では準拠不可。Annex 1の係数を直接実装。

**48kHz固定係数**（ITU-R BS.1770-4/5 Table 1より、JUCE Forum実装で検証済み）:

```cpp
// Stage 1: Pre-filter (High-shelf)
static constexpr double pre_b0 = 1.535124859586970;
static constexpr double pre_b1 = -2.691696189406380;
static constexpr double pre_b2 = 1.198392810852850;
static constexpr double pre_a1 = -1.690659293182410;
static constexpr double pre_a2 =  0.732480774215850;

// Stage 2: RLB filter (High-pass)
static constexpr double rlb_b0 = 1.0;
static constexpr double rlb_b1 = -2.0;
static constexpr double rlb_b2 = 1.0;
static constexpr double rlb_a1 = -1.990047454833980;
static constexpr double rlb_a2 =  0.990072250366210;
```

他サンプルレート対応が必要な場合: Klangfreundのbilinear変換コード（JUCE Forum）で係数変換。

### S2.6 Mid/Side enum安全確認

`EQChannelMode` の全使用箇所（6ファイル）を検索した結果:

| ファイル | 使用形態 | switch default要否 |
|---------|---------|-------------------|
| `EQProcessor.h` | 定義 | — |
| `EQProcessor.Processing.cpp:663` | switch文で分岐 | 要対応（enum追加時） |
| `EQProcessor.Core.cpp:200` | 代入のみ `= Stereo` | 不要 |
| `EQProcessor.Parameters.cpp:176` | 代入のみ | 不要 |
| `AudioEngine.EQResponse.cpp:84` | 構造体メンバ | 不要 |
| `AudioEngine.h:843` | 引数型 | 不要 |

→ switch文は `Processing.cpp` の1箇所のみ。enum追加時はここに case を追加すればよく、default節未存在によるリスクはない。

Mid/Sideを `curvesL`/`curvesR` に割り当てるのは物理的に不正確（Mid/SideはL/R信号ではない）。独立した `curvesMid`(cyan)/`curvesSide`(magenta) トレースで表示。

---

## S3. filterState拡張 周辺コード監査結果

`[2][20][2]`→`[4][20][2]` 拡張時の全11箇所監査:

| # | 対象 | ファイル | 要修正 |
|---|------|---------|--------|
| 1 | filterState宣言 | `EQProcessor.h:594` | `kFilterChannels` に変更 |
| 2-4 | reset/prepare/クロスフェード | `Core.cpp:246,736`, `Processing.cpp:787` | 不要（auto/memset自動サイズ）✅ |
| 5 | **バンドリセットループ** | `Processing.cpp:980` | **`ch < kFilterChannels` に修正** |
| 6 | 全バンドmemset | `Processing.cpp:971` | 不要 ✅ |
| 7-8 | processSerial/Parallel | `Processing.cpp:654,830` | 不要（テンプレート自動）✅ |
| 9-11 | EQCoeffCache/Cache比較/Snapshot | 各所 | 不要（enum値追加のみ）✅ |

→ **実質2箇所のみ修正**。

---

## S4. 影響ファイル一覧

### Phase 1

```
M src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
M src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp
M src/audioengine/AudioEngine.h
M src/audioengine/AudioEngine.Processing.Lifecycle.cpp
M src/CustomInputOversampler.cpp/h
```

### Phase 2

```
M src/eqprocessor/EQProcessor.h
M src/eqprocessor/EQProcessor.Processing.cpp
M src/eqprocessor/EQProcessor.Core.cpp
M src/SpectrumAnalyzerComponent.cpp
M src/audioengine/AudioEngine.EQResponse.cpp
M src/EQControlPanel.cpp
```

### Phase 3

```
A src/TruePeakDetector.h/cpp
A src/LoudnessMeter.h/cpp
M src/audioengine/AudioEngine.h
M src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
M src/audioengine/AudioEngine.Processing.Lifecycle.cpp
```

### Phase 4

```
A src/TruePeakLimiter.h/cpp
M src/audioengine/AudioEngine.h
M src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
```

### Phase 5（判断待ち）

```
M src/eqprocessor/EQProcessor.Processing.cpp
M src/eqprocessor/EQProcessor.h
```

---

## S5. リスク評価

| リスク | 影響 | 確率 | 対策 |
|--------|------|------|------|
| M/S EQ: モード組合せ不具合 | 低 | 低 | 全モード回帰テスト |
| TruePeak: 31tapでBS.1770不足 | 中 | 中 | 63〜128tap採用 |
| LUFS: ゲーティング不整合 | 低 | 低 | Annex 1テスト信号検証 |
| SoftClipOS: 既存OS競合 | 低 | 低 | `OS==1` ガードで排他 |
| ADAA: 40状態管理 | 中 | 高 | Phase 5判断時 |
| filterState拡張: コピー量 | 低 | 確実 | 640→1,280B。影響なし |
