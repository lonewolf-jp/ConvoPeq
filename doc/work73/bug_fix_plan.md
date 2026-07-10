# ConvoPeq バグ修正計画案

**作成日:** 2026-07-09  
**改訂日:** 2026-07-09 （#1のバッファレイアウト計算誤りを修正）  
**対象:** doc/work73/bug_verification_report.md で確認された3件のTrue Positiveバグ

---

## #1 TruePeakDetectorのRチャンネル計測欠落

**優先度:** 高  
**対象ファイル:** `src/TruePeakDetector.cpp`（278行）  
**修正必要箇所:** 2箇所（processBlock: Stage 1 R追加 + ピーク検出範囲拡張）  
※ `prepare()` のバッファ拡張は不要（後述）

### ソースコード詳細調査

**`processBlock`(59-108行目)のバッファレイアウト分析:**

N = numSamples として計算する。

```cpp
double* work = upsampleBuffer.get();

// Step 1: 各チャンネルを 1x→2x にアップサンプル
interpolateStage(stages[0], dataL, numSamples, work, 0);                 // L → work[0..up1-1]
interpolateStage(stages[0], dataR, numSamples, work + up1Samples, 1);    // R → work[up1..up1*2-1]
const int up1Samples = numSamples * 2;    // up1 = 2N

// Step 2: Lチャンネルのみ 2x→4x にアップサンプル
interpolateStage(stages[1], work, up1Samples, work + up1Samples * 2, 0); // L only → work[up1*2..]
const int up2Samples = up1Samples * 2;   // up2 = 4N

// Step 3: ピーク検出 ← work + up1Samples * 2 (Lチャンネルの4xデータのみ)
for (int i = 0; i < vEnd; i += 4)
    __m256d v = _mm256_loadu_pd(work + up1Samples * 2 + i);  // ← 問題箇所
```

**`interpolateStage` の出力サイズ（重要）:**

`interpolateStage`（253-278行目）は入力 `inputSamples` 個に対して **`inputSamples * 2` 個**の出力を生成する（275-276行目の `output[n * 2]` / `output[n * 2 + 1]`）。

| Stage | 入力サイズ | 出力サイズ |
|-------|-----------|-----------|
| Stage 0 | N | N × 2 = up1 (2N) |
| Stage 1 | up1 (2N) | up1 × 2 = up2 (4N) |

**現在のバッファレイアウト:**
```
work バッファ:
  [0 ............... up1-1]         = Lチャンネル 2xデータ        (2N samples)
  [up1 .......... up1*2-1]          = Rチャンネル 2xデータ        (2N samples)
  [up1*2 ....... up1*2+up2-1]      = Lチャンネル 4xデータ        (4N samples = up2)
    ※ up1*2+up2 = up1*2+up1*2 = up1*4 = 8N が終端
  [up1*4 ........]                  = 未使用
```

**修正後のバッファレイアウト:**
```
work バッファ:
  [0 ............... up1-1]         = Lチャンネル 2xデータ        (2N samples)
  [up1 .......... up1*2-1]          = Rチャンネル 2xデータ        (2N samples)
  [up1*2 ....... up1*4-1]           = Lチャンネル 4xデータ        (4N samples = up2)
  [up1*4 ....... up1*6-1]           = Rチャンネル 4xデータ        (4N samples = up2)  ← 新規追加
    ※ 終端 = up1*6 = 12N
```

**バッファ容量の検証 — 拡張は不要:**

`prepare()` の呼び出し元（`DSPCoreLifecycle.cpp`）を確認すると:
```cpp
truePeakDetector.prepare(newSampleRate, maxInternalBlockSize);
// maxInternalBlock = inputMaxBlock * MAX_OS_FACTOR = hostBlockSize * 8
```

一方、`processBlock` の呼び出し元（`DSPCoreDouble.cpp:741`）は:
```cpp
truePeakDetector.processBlock(dataL, dataR, numSamples);
// numSamples = ホストブロックサイズ (≤ hostBlockSize)
```

したがって:
- バッファ実サイズ = `maxInternalBlockSize * 4` = `hostBlockSize * 32`
- 修正後の最大使用量 = `12N` ≤ `12 * hostBlockSize`
- `hostBlockSize * 12 << hostBlockSize * 32`

**結論: 現在のバッファサイズで十分。`prepare()` の変更は不要。**

### 修正手順

#### 1-1. `processBlock()` 内 Stage 1 Rチャンネル処理追加 (TruePeakDetector.cpp:76-78行目)

```cpp
// 修正前 (76-78行目)
// Stage 1: 2x → 4x (L/Rはworkにインターリーブ)
interpolateStage(stages[1], work, up1Samples, work + up1Samples * 2, 0);
const int up2Samples = up1Samples * 2;

// 修正後
// Stage 1: 2x → 4x (L)
interpolateStage(stages[1], work, up1Samples, work + up1Samples * 2, 0);

// ★ 追加: Stage 1: 2x → 4x (R)
// 出力先を up1*4 に設定（up1*3 ではない — up1*3 では L 4x [up1*2,up1*4) と
// 重複領域 [up1*3,up1*4) が発生し L 4x データが破壊される）
if (dataR != nullptr)
    interpolateStage(stages[1], work + up1Samples, up1Samples, work + up1Samples * 4, 1);
else
    interpolateStage(stages[1], work, up1Samples, work + up1Samples * 4, 1);

const int up2Samples = up1Samples * 2;
```

**バッファオフセットの意味:**

| 引数 | L出力 | R出力 | 意味 |
|------|-------|-------|------|
| 入力元 | `work` (L 2x) | `work + up1Samples` (R 2x) | Stage 0 の 2x データ |
| 入力サンプル数 | `up1Samples` (2N) | `up1Samples` (2N) | 2x データ長 |
| 出力先 | `work + up1Samples * 2` | `work + up1Samples * 4` | Stage 1 の 4x データ |
| 出力サイズ | up2 (4N) | up2 (4N) | 各チャンネル 4x |
| 占有区間 | `[up1*2, up1*4)` = `[4N, 8N)` | `[up1*4, up1*6)` = `[8N, 12N)` | **重複なし** ✅ |

#### 1-2. ピーク検出範囲をL+R全体に拡張 (TruePeakDetector.cpp:80-99行目)

```cpp
// 修正前 (80-99行目)
double peak = 0.0;
const int vEnd = up2Samples / 4 * 4;
for (int i = 0; i < vEnd; i += 4)
{
    __m256d v = _mm256_loadu_pd(work + up1Samples * 2 + i);
    // ...
}
for (int i = vEnd; i < up2Samples; ++i)
{
    const double v = std::abs(work[up1Samples * 2 + i]);
    if (v > peak) peak = v;
}

// 修正後
double peak = 0.0;
// ★ L+R両チャンネルの4xデータ全体をスキャン
//   L 4x: [up1*2, up1*4)  = work[4N .. 8N-1]
//   R 4x: [up1*4, up1*6)  = work[8N .. 12N-1]
//   合計: up2Samples * 2 = 8N サンプルを連続スキャン
const int total4xSamples = up2Samples * 2;
const int vEnd = total4xSamples / 4 * 4;
for (int i = 0; i < vEnd; i += 4)
{
    __m256d v = _mm256_loadu_pd(work + up1Samples * 2 + i);
    __m256d absV = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
    __m128d lo = _mm256_castpd256_pd128(absV);
    __m128d hi = _mm256_extractf128_pd(absV, 1);
    __m128d max01 = _mm_max_pd(lo, hi);
    __m128d max0 = _mm_max_sd(max01, _mm_unpackhi_pd(max01, max01));
    double m;
    _mm_store_sd(&m, max0);
    if (m > peak) peak = m;
}
for (int i = vEnd; i < total4xSamples; ++i)
{
    const double v = std::abs(work[up1Samples * 2 + i]);
    if (v > peak) peak = v;
}
```

### 修正影響

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| `prepare()` バッファサイズ | `maxBlockSize * 4` | **変更なし**（十分な余裕あり） |
| Stage 1処理 | Lチャンネルのみ | L+R両チャンネル |
| ピーク検出範囲 | `[up1*2, up1*2+up2)` = L 4xのみ (4N) | `[up1*2, up1*2+up2*2)` = L+R 4x (8N) |
| バッファ最大使用量 | 8N | 12N（`hostBlockSize*12 ≤ hostBlockSize*32` のため安全） |
| 修正行数 | — | ~15行（`processBlock` 内のみ） |

---

## #2 SimplePeakLimiterのKnee補間境界エラー

**優先度:** 高  
**対象ファイル:** `src/audioengine/SimplePeakLimiter.h`（87行）  
**修正必要箇所:** 1箇所（1行の条件式修正）

### ソースコード詳細調査

**`processBlock`(36-80行目)のKnee処理分析:**

```cpp
void processBlock(double* dataL, double* dataR, int numSamples,
                  double thresholdLinear, double kneeLinear) noexcept
{
    const double clipStart = thresholdLinear - kneeLinear * 0.5;

    for (int i = 0; i < numSamples; ++i)
    {
        const double peak = juce::jmax(std::abs(dataL[i]),
                                       hasR ? std::abs(dataR[i]) : absL);

        double desiredGain = 1.0;
        if (peak > clipStart)
        {
            if (peak <= thresholdLinear)  // ★ バグ: 右半分のKneeが欠落
            {
                const double t = (peak - clipStart) / kneeLinear;
                const double kneeShape = t * t * (3.0 - 2.0 * t);
                desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
            }
            else
            {
                desiredGain = thresholdLinear / peak;
            }
        }
        // ...
    }
}
```

**Knee範囲と分岐の分析:**

```text
         clipStart                     threshold     threshold+knee/2
            |                             |                 |
 t=0.0       |----------- t=0.5 ---------|---- t=1.0 -----|
            |      スプライン補間        |  ★ここで切断     |
            |  (なめらかに gain 低下)    |  →ハードリミッティングへジャンプ
```

| `peak` の位置 | 期待される `t` | 期待される `kneeShape` | 現在の実装 | 問題 |
|--------------|---------------|----------------------|-----------|------|
| `clipStart` | 0.0 | 0.0 (unity) | スプライン ✅ | - |
| `clipStart + knee/4` | 0.25 | ~0.156 | スプライン ✅ | - |
| `threshold` | 0.5 | 0.5 | ★ `else` 側へ | 傾きが不連続に |
| `threshold + knee/4` | 0.75 | ~0.844 | ★ `else` 側へ | 微係数ジャンプでクリックノイズ |
| `threshold + knee/2` | 1.0 | 1.0 | ★ `else` 側へ | Knee右半分全体が欠落 |

**音響への影響:**
- `peak`が`threshold`を超えた瞬間に、3次スプラインのなめらかなカーブからハードリミッティング（`threshold/peak`）へ**微係数が不連続に切り替わる**
- これにより**高調波歪み（クリックノイズ）**が発生する
- 特にトランジェントの多い素材（ドラム、パーカッション）で顕著

### 修正手順

#### 2-1. 条件式の修正 (SimplePeakLimiter.h:55行目)

```cpp
// 修正前
if (peak <= thresholdLinear)

// 修正後
if (peak <= thresholdLinear + kneeLinear * 0.5)
```

**修正後のKnee範囲:**

```text
         clipStart                     threshold     threshold+knee/2
            |                             |                 |
 t=0.0       |----------- t=0.5 ---------|---- t=1.0 -----|
            |          スプライン補間    |  (なめらかにgain低下)  |
            |  t=0.0→1.0 全区間で連続かつ滑らかなゲインリダクション
```

| `peak` の位置 | `t` | `kneeShape` | 修正後の実装 |
|--------------|-----|-------------|------------|
| `clipStart` | 0.0 | 0.0 (unity) | スプライン ✅ |
| `threshold` | 0.5 | 0.5 | スプライン ✅（連続かつ滑らか） |
| `threshold + knee/2` | 1.0 | 1.0 (full clip) | スプライン ✅ |
| `> threshold + knee/2` | - | - | ハードリミッティング ✅ |

**境界（t=1）での連続性の数学的検証:**

smoothstep関数 `s(t) = t²(3-2t)`:
- `s(0) = 0`, `s(1) = 1`
- `s'(t) = 6t(1-t)`, `s'(0) = 0`, `s'(1) = 0`

| 項目 | スプライン側 (t=1) | ハードリミッティング側 |
|------|-------------------|----------------------|
| 値 | `1 - (1 - thr/peak) * 1 = thr/peak` | `thr/peak` |
| 微係数 dG/d(peak) | `-thr/peak²` | `-thr/peak²` |

値も微係数も一致。**C1連続性が保証される。** ✅

**修正後の完全な関数:**

```cpp
void processBlock(double* dataL, double* dataR, int numSamples,
                  double thresholdLinear, double kneeLinear) noexcept
{
    if (dataL == nullptr || numSamples <= 0)
        return;

    const double clipStart = thresholdLinear - kneeLinear * 0.5;
    const bool hasR = (dataR != nullptr);

    for (int i = 0; i < numSamples; ++i)
    {
        const double peak = juce::jmax(std::abs(dataL[i]),
                                       hasR ? std::abs(dataR[i]) :
                                              std::abs(dataL[i]));

        double desiredGain = 1.0;
        if (peak > clipStart)
        {
            // ★ 修正: Knee上端までスプライン補間を適用
            if (peak <= thresholdLinear + kneeLinear * 0.5)
            {
                const double t = (peak - clipStart) / kneeLinear;
                const double kneeShape = t * t * (3.0 - 2.0 * t);
                desiredGain = 1.0 - (1.0 - thresholdLinear / peak) * kneeShape;
            }
            else
            {
                desiredGain = thresholdLinear / peak;
            }
        }

        if (desiredGain < envelope)
            envelope = desiredGain;
        else
            envelope = 1.0 + (envelope - 1.0) * releaseCoeff;

        dataL[i] *= envelope;
        if (hasR)
            dataR[i] *= envelope;
    }
}
```

### 修正影響

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| Knee適用範囲 | `clipStart` ～ `threshold`（t=0.0～0.5） | `clipStart` ～ `threshold+knee/2`（t=0.0～1.0） |
| Knee領域のt最大値 | 0.5 | 1.0（全区間） |
| スプラインとハードリミッターの境界 | t=0.5で不連続ジャンプ | t=1.0でC1連続遷移 |
| 修正行数 | - | 1行 |

---

## #3 StereoConvolver::clone()のプロファイル消失

**優先度:** 中  
**対象ファイル:** `src/ConvolverProcessor.h`（1194行, StereoConvolver内部クラス）  
**修正必要箇所:** 3箇所（メンバ追加 + init保存 + clone引数追加）

### ソースコード詳細調査

**StereoConvolverのinit関数 (ConvolverProcessor.h:705-744行目) とclone (764-787行目):**

```cpp
// initの保存処理 (720-723行目)
storedSampleRate = sr;
storedKnownBlockSize = knownBlockSize;
storedScale = scale;
storedDirectHeadEnabled = enableDirectHead;
// ★ storedFilterSpec は保存されていない！

// cloneの呼び出し (778行目)
if (!newConv->init(l.release(), r.release(),
                   irDataLength,
                   storedSampleRate,         // ✅ 復元される
                   irLatency,
                   storedKnownBlockSize,     // ✅ 復元される
                   callQuantumSamples,
                   storedScale,              // ✅ 復元される
                   storedDirectHeadEnabled)) // ✅ 復元される
    return nullptr;
// ★ storedFilterSpec は復元されない → nullptr → NUCでフィルター無効化！
```

**他のinit()呼び出し元との比較:**

| 呼び出し元 | filterSpec引数 |
|-----------|---------------|
| `ConvolverProcessor.Lifecycle.cpp` | `&tailSpec` ✅ 渡している |
| `ConvolverProcessor.LoaderThread.cpp` | `&spec` ✅ 渡している |
| `ConvolverProcessor.LoadPipeline.cpp` | `&spec` ✅ 渡している |
| `StereoConvolver::clone()` | **省略（nullptr）** ❌ |

**filterSpecがnullptrの場合のNUCの挙動:**
- ハイカットモード: `convo::HCMode::Natural`（フィルターなし）
- ローカットモード: `convo::LCMode::Natural`（フィルターなし）
- テールコンタリング: デフォルト値
- **結果:** cloneされたインスタンスではユーザー設定のフィルター特性がすべて消失

**FilterSpecの実定義（MKLNonUniformConvolver.h:124-135行目）:**

`namespace convo` 内に定義された POD 構造体。全メンバがデフォルト初期化子を持つため、デフォルト構築およびコピー代入が安全。

```cpp
struct FilterSpec
{
    double sampleRate = 48000.0;
    HCMode hcMode     = HCMode::Natural;
    LCMode lcMode     = LCMode::Natural;
    int tailMode = 1;
    bool tailEnabled  = true;
    double tailStartSeconds = 0.085;
    double tailStrength = 1.0;
    int tailL1L2Multiplier = 8;
};
```

### 修正手順

#### 3-1. `storedFilterSpec` メンバの追加 (ConvolverProcessor.h:644-648行目付近)

```cpp
struct StereoConvolver
{
    // ... 既存メンバ ...

    double* irData[2] = { nullptr, nullptr };
    std::array<convo::MKLNonUniformConvolver*, 2> nucConvolvers { nullptr, nullptr };
    int irDataLength = 0;
    int latency = 0;
    int irLatency = 0;
    int callQuantumSamples = 0;

    // Clone用に初期化パラメータを保存
    double storedSampleRate = 0.0;
    int storedKnownBlockSize = 0;
    double storedScale = 1.0;
    bool storedDirectHeadEnabled = false;

    // ★ 追加: 初期化時のFilterSpecを保存してclone時に復元する
    convo::FilterSpec storedFilterSpec{};

    // ... 以降のメンバ ...
};
```

#### 3-2. `init()` 内に保存処理追加 (ConvolverProcessor.h:720-723行目付近)

```cpp
bool init(double* irL, double* irR, int length, double sr, int peakDelay,
          int knownBlockSize, int preferredCallSize, double scale = 1.0,
          bool enableDirectHead = false,
          const convo::FilterSpec* filterSpec = nullptr,
          ConvolverProcessor* ownerProcessor = nullptr)
{
    // ... 既存の保存処理 (720-723行目) ...

    storedSampleRate = sr;
    storedKnownBlockSize = knownBlockSize;
    storedScale = scale;
    storedDirectHeadEnabled = enableDirectHead;

    // ★ 追加: filterSpecを保存（clone時に復元するため）
    if (filterSpec != nullptr)
        storedFilterSpec = *filterSpec;
    else
        storedFilterSpec = convo::FilterSpec{};

    // ... 以降のNUC初期化処理は変更なし ...
}
```

#### 3-3. `clone()` 内に引数追加 (ConvolverProcessor.h:778行目)

```cpp
[[nodiscard]] StereoConvolver* clone() const
{
    try
    {
        auto newConv = convo::aligned_make_unique<StereoConvolver>();

        if (irDataLength > 0 && irData[0] && irData[1])
        {
            auto l = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));
            auto r = convo::makeAlignedArray<double>(static_cast<size_t>(irDataLength));

            std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
            std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));

            // ★ 修正: storedFilterSpec を引数に追加
            if (!newConv->init(l.release(),
                               r.release(),
                               irDataLength,
                               storedSampleRate,
                               irLatency,
                               storedKnownBlockSize,
                               callQuantumSamples,
                               storedScale,
                               storedDirectHeadEnabled,
                               &storedFilterSpec))   // ← 追加
                return nullptr;
        }
        return newConv.release();
    }
    catch (const std::bad_alloc&)
    {
        return nullptr;
    }
}
```

### 修正影響

| 項目 | 修正前 | 修正後 |
|------|--------|--------|
| clone時のfilterSpec | nullptr（常にフィルターなし） | 元インスタンスのstoredFilterSpecを復元 |
| clone時のハイカット | Natural（無効） | ユーザー設定値を維持 |
| clone時のローカット | Natural（無効） | ユーザー設定値を維持 |
| clone時のテール処理 | デフォルト | ユーザー設定値を維持 |

**保存するパラメータと保存しないパラメータの一貫性:**

| initパラメータ | 保存 (storedXxx) | clone時復元 |
|---------------|-----------------|------------|
| `sr` | ✅ `storedSampleRate` | ✅ |
| `knownBlockSize` | ✅ `storedKnownBlockSize` | ✅ |
| `scale` | ✅ `storedScale` | ✅ |
| `enableDirectHead` | ✅ `storedDirectHeadEnabled` | ✅ |
| `filterSpec` | **修正前:** ❌ **修正後:** ✅ `storedFilterSpec` | **修正後:** ✅ |

---

## 修正影響まとめ

| バグ | ファイル | 修正箇所 | 修正行数 | リスク | テスト方針 |
|-----|---------|---------|---------|--------|-----------|
| #1 TruePeakDetector | `TruePeakDetector.cpp` | processBlock: Stage 1 R追加 + ピーク検出範囲 | ~15行 | 低（バッファ拡張なし、processBlock内のみ） | L-only, R-only, Stereo信号でのTruePeak値比較 |
| #2 SimplePeakLimiter | `SimplePeakLimiter.h` | processBlock内の条件式 | 1行 | 極小 | 異なるpeak値でのgainカーブ連続性確認 |
| #3 StereoConvolver::clone | `ConvolverProcessor.h` | StereoConvolver::init + clone + メンバ | ~8行 | 極小 | clone前後でフィルター特性が同一であることを確認 |

---

**報告作成完了日時:** 2026-07-09  
**改訂内容:** #1 TruePeakDetector のバッファレイアウト計算誤りを修正（R 4xオフセット `up1*3`→`up1*4`、バッファ拡張不要の証明追加）  
**修正計画対象バグ数:** 3件（すべて True Positive）