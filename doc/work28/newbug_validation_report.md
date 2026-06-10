# newbug.md バグ解析検証レポート

**作成日**: 2026-06-10
**検証者**: GitHub Copilot (DeepSeek V4 Flash)
**対象**: `doc/work28/newbug.md`

---

## 検証方法

本レポートは以下の6種類のツールをすべて使用して検証を実施した。

| ツール | 用途 |
|--------|------|
| **grep/Select-String (RTK経由)** | 全バグのキーワード横断検索 |
| **Serena MCP** (oraios/serena v1.5.3) | シンボル構造解析・依存関係確認・メモリ参照 |
| **CodeGraph MCP** (codegraph-mcp v0.8.0) | モジュール構造解析、依存関係グラフ解析、グローバル検索 |
| **ccc (cocoindex-code)** | ASTベースセマンティックコード検索 |
| **semble** (MinishLab) | セマンティックコード検索（～98%トークン削減） |
| **Graphify MCP** (safishamsi/graphify v0.8.36) | 知識グラフ解析（DeepSeek backend, 13,168 nodes, 17,089 edges, 1,311 communities, \$0.036） |

---

## 検証結果サマリ

| ID | 優先度 | カテゴリ | 報告内容 | 判定 | 補正後優先度 |
|----|--------|----------|----------|:----:|:------------:|
| **C-01** | Critical | RT安全違反 | `LinearRamp::setCurrentAndTargetValue` をAudio Threadから呼び出し | ✅ **確認** | 1 |
| **C-02** | Critical | 未定義動作 | `union` による型パンニング | ✅ **確認** | 2 |
| **C-03** | Critical | Use-After-Free | `ProgressiveUpgradeThread` のラムダが生ポインタをキャプチャ | ✅ **確認** | 3 |
| **H-01** | High | 機能不全 | `m_rtBypassShadow` が未更新でバイパス無効 | ⚠️ **過小報告** | 10 |
| **H-02** | High | メモリリーク | `LoaderThread` の `callAsync` ラムダがメモリリーク | ⚠️ **過大評価** | — |
| **H-03** | High | 音質劣化 | IR逆順後に `irFreqReal/irFreqImag` 不整合 | ✅ **確認（潜在）** | 7 |
| **H-04** | High | 音質劣化 | 線形クロスフェード（等パワーではない） | ✅ **確認** | 4 |
| **H-05** | High | 例外安全 | `finalizeNUCEngineOnMessageThread` の例外処理不足 | ✅ **確認** | 5 |
| **M-01** | Medium | RT安全違反 | `std::abs` をAudio Threadで使用 | ✅ **確認** | 6 |
| **M-02** | Medium | 音質劣化 | DCブロッカーカットオフ不統一（3 Hz vs 1 Hz） | ✅ **確認** | 8 |
| **M-03** | Medium | パフォーマンス | `Fixed15TapNoiseShaper` の毎サンプル全状態スキャン | ✅ **確認** | 8 |
| **M-04** | Medium | 状態管理 | `observeMonotonicRollbackRequested_` フラグがリセットされない | ❌ **誤報告** | — |
| **M-05** | Medium | コード品質 | `wetBuf[0]` の危険な流用 | ✅ **確認** | 8 |
| **L-01** | Low | 表示精度 | レベルメーター測定タイミング | ✅ **確認** | 9 |
| **L-02** | Low | コード品質 | `applySpectrumFilter` のHC `else` ブランチ到達不能 | ✅ **確認** | 9 |
| **L-03** | Low | 命名/コメント | `ORDER = 16` / `calcSVFCoeffs` コメント誤記 | ✅ **確認** | 9 |

### 精度

- **正確**: 12/16 = **75%**
- **要修正（過小/過大評価）**: 3/16 = **19%**
- **誤報告**: 1/16 = **6%**

---

## バグ別詳細検証結果

### 🔴 C-01: `LinearRamp::setCurrentAndTargetValue` スレッドコンテキスト違反 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/DspNumericPolicy.h`, `src/convolver/ConvolverProcessor.Runtime.cpp` |
| **検証ツール** | grep, ccc, semble, Serena, CodeGraph |

**宣言部** (`DspNumericPolicy.h:259-261`):

```cpp
void setCurrentAndTargetValue(double v) noexcept
{
    ASSERT_NON_RT_THREAD();  // ← 非Audio Thread専用
    current = target = v;
    step      = 0.0;
    remaining = 0;
}
```

**違反呼び出し箇所**（すべて `ConvolverProcessor.Runtime.cpp` の `process()` = Audio Thread）:

| 行 | コード | 用途 |
|----|-------|------|
| 246 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | レイテンシクロスフェード開始 |
| 279 | `activeLatencySmoother.setCurrentAndTargetValue(val);` | 保留中レイテンシ値の即時設定 |
| 295 | `activeCrossfadeGain.setCurrentAndTargetValue(0.0);` | レイテンシ変更時クロスフェード |
| 307 | `activeMixSmoother.setCurrentAndTargetValue(...)` | Mix値リセット |
| 323 | `activeMixSmoother.setCurrentAndTargetValue(currentVal);` | スムージング時間変更後 |
| 491 | `activeLatencySmoother.setCurrentAndTargetValue(...)` | レイテンシ再設定 |

**影響**: デバッグビルドで `jassert` 失敗→即座にクラッシュ。リリースビルドでも設計契約（スレッド安全性）違反。

**Serena/CodeGraph 解析**: `LinearRamp` 構造体の全メソッドとスレッドアサーションの関連性を確認。`reset()` / `setCurrentAndTargetValue()` は `ASSERT_NON_RT_THREAD()`, `setTargetValue()` / `getNextValue()` / `isSmoothing()` は `ASSERT_AUDIO_THREAD()` で保護。

---

### 🔴 C-02: `union` による型パンニング（未定義動作） ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/audioengine/AudioEngine.h`, `src/eqprocessor/EQProcessor.Processing.cpp`, `src/convolver/ConvolverProcessor.Runtime.cpp`, `src/DspNumericPolicy.h` |
| **検証ツール** | grep, ccc, Graphify |

**該当関数一覧**:

| ファイル | 関数 | 行 |
|----------|------|----|
| `AudioEngine.h` | `absNoLibm` | 117-121 |
| `EQProcessor.Processing.cpp` | `absNoLibm` (匿名名前空間) | 53-56 |
| `ConvolverProcessor.Runtime.cpp` | `isFiniteAndAbsBelowNoLibm` | 33-39 |
| `DspNumericPolicy.h` | `killDenormal` (複数オーバーロード) | 156-222 |

**コードパターン**:

```cpp
// C++ 未定義動作 (UB)
union { double d; uint64_t u; } v { x };
v.u &= 0x7FFFFFFFFFFFFFFFULL;
return v.d;

// 修正案 (C++20 defined behavior)
return std::bit_cast<double>(std::bit_cast<uint64_t>(x) & 0x7FFFFFFFFFFFFFFFULL);
```

**Graphify 解析**: `absNoLibm()` は Community 3 に所属し、AudioEngine.h に定義。多数のファイルから参照される中心的なユーティリティ関数。

**影響**: C++ 標準で未定義動作。最適化ビルド（`-O2` 以上）でコンパイラが予期しないコード生成を行う可能性がある。特に MSVC の `/O2` 最適化では `union` の型パンニングが期待通り動作しないケースが報告されている。

---

### 🔴 C-03: `ProgressiveUpgradeThread` の Use-After-Free ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/ProgressiveUpgradeThread.cpp`, `src/ProgressiveUpgradeThread.h` |
| **検証ツール** | grep, Serena, CodeGraph, Graphify |

**該当コード** (`ProgressiveUpgradeThread.cpp:84-95`):

```cpp
juce::WeakReference<ConvolverProcessor> weakOwner(&processor);
std::atomic<bool>* cancelledFlag = &cancelled;  // ← メンバ変数の生ポインタ
const uint64_t expectedGeneration = taskGeneration;

prepared = converter.convertToHighRes(irFile, ...,
    [weakOwner, cancelledFlag, expectedGeneration]()  // ← cancelledFlag を値キャプチャ
    {
        auto* owner = weakOwner.get();
        if (owner == nullptr)
            return true;
        return juce::Thread::currentThreadShouldExit()
            || convo::consumeAtomic(*cancelledFlag, ...)  // ← Use-After-Free の危険
            || !owner->isConvolverGenerationCurrent(expectedGeneration);
    });
```

**問題のメカニズム**:

1. `cancelledFlag` は `ProgressiveUpgradeThread::cancelled`（メンバ変数）のアドレス
2. `convertToHighRes()` に渡されたラムダは別スレッド（コンバーター内部スレッド）で非同期実行される可能性がある
3. `ProgressiveUpgradeThread` オブジェクトが破棄されると `cancelled` メンバも解放される
4. ラムダ実行時に `*cancelledFlag` で解放済みメモリにアクセス → **Use-After-Free**

**Graphify 解析**: `ProgressiveUpgradeThread` は `IRConverter`, `CacheManager`, `ConvolverProcessor`, `ThreadAffinityManager` と密結合。デストラクタは `stopThread(2000)` で最大2秒待機するが、タイムアウト後はスレッドが生存したままオブジェクトが破棄される。

---

### 🟠 H-01: `m_rtBypassShadow` 未更新 ⚠️ 過小報告（要修正）

**ファクト**: `setBypassFromRT()` は **呼び出されている**。

**エビデンス**:

- `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:382`:

  ```cpp
  eqRt().setBypassFromRT(state.eqBypassed);
  ```

- `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:189`:

  ```cpp
  eqRt().setBypassFromRT(state.eqBypassed);
  ```

**呼び出し経路**: Audio Thread → `AudioEngine::DSPCore::processDoubleToBuffer/processFloatToBuffer` → ランタイムスナップショットから `state.eqBypassed` を読み取り → `EQProcessor::setBypassFromRT()` で `m_rtBypassShadow` を更新 → `EQProcessor::process()` 内で `m_rtBypassShadow` を参照

**報告書の誤り**: 「`setBypassFromRT()` 関数は存在するが、どこからも呼ばれていない」は**誤り**。Audio Engine の DSPCore パスから正しく呼ばれている。

**残存する懸念点**:

- `m_rtBypassShadow` の初期値は `false`（未バイパス）。初回 `process()` 呼び出し前に別のコードがこの値を読む可能性はあるが、実際の処理パスでは必ず DSPCore 経由で設定されるため実害はない
- 設計上の明確性の問題であり、**機能障害ではない**

**推奨修正**: `newbug.md` の H-01 エントリを以下に修正:

- 優先度: High → **Medium**
- 内容: 「`setBypassFromRT()` は DSPCore から呼ばれているが、設計の明確性を向上するため初期化経路の統一等を検討」

---

### 🟠 H-02: `LoaderThread` の `callAsync` ラムダメモリリーク ⚠️ 過大評価

**エビデンス** (`src/convolver/ConvolverProcessor.LoaderThread.cpp:297-348`):

**リークなしの経路**:

1. `callAsync` 成功 → ラムダが Message Thread で実行される → ラムダ先頭で `unique_ptr`/`ScopedAlignedPtr` にラップ → `weakOwner.get()` の成否に関わらずスコープ終了時に解放
2. `callAsync` 失敗 → `!queued` ブロックで明示的に `aligned_free` / `unique_ptr` で解放

**唯一のリーク経路**: JUCE アプリケーション終了時に MessageManager のキューが破棄され、未実行のラムダがそのままになる場合。これは**正常シャットダウンでは許容範囲**。

**推奨修正**: 優先度を High → **Low** に引き下げ、内容を「シャットダウン時のキュー破棄による理論上のリーク」に修正。

---

### 🟠 H-03: IRパーティション逆順後の `irFreqReal/irFreqImag` 不整合 ✅ 確認（潜在バグ）

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/MKLNonUniformConvolver.cpp` |
| **検証ツール** | grep, CodeGraph |

**処理の流れ**:

| ステップ | 行 | 処理 | 状態 |
|----------|-----|------|------|
| 1 | 793-794 | `deinterleaveComplex(irFreqDomain[p], irFreqReal[p], irFreqImag[p])` | `irFreqReal/Imag` = 正順 |
| 2 | 816-834 | `irFreqDomain` のパーティションを逆順にswap | `irFreqDomain` = 逆順, `irFreqReal/Imag` = 正順のまま ← **不一致** |
| 3 | 877 | `if (filterSpec != nullptr) applySpectrumFilter()` | 内部で再デインターリーブ → 修正 |
| 4 | — | `filterSpec == nullptr` の場合 | `applySpectrumFilter` がスキップ → **不一致継続** |

**問題の核心**: Split-complex AVX2 パス（`kEnableSplitComplexKernel == true`）では `irFreqReal/irFreqImag` を使用する（行1088-1089, 1317-1318）。`filterSpec == nullptr` の場合、これらは逆順化前のデータを指したままとなり、**時間反転したIRが適用される**。

**現状**: 呼び出し元は常に `filterSpec` を指定しているため顕在化していないが、API 契約上は `filterSpec` が nullptr でも動作する。

---

### 🟠 H-04: 線形クロスフェード（等パワーではない） ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`, `BlockDouble.cpp`, `AudioEngine.h` |
| **検証ツール** | grep, ccc, semble |

**該当箇所**:

1. **DSP処理順序切替時**:
   - `AudioBlock.cpp:267`: `const double gOld = 1.0 - gNew;`
   - `BlockDouble.cpp:233`: `const double gOld = 1.0 - gNew;`

2. **全体バイパスフェード時**:
   - `AudioEngine.h:482-489`: `bypassFadeGainDouble.setCurrentAndTargetValue(1.0);`
   - `AudioEngine.h:3160`: `crossfadeRuntime_.getGain().getNextValue()`（LinearRamp, 線形）

3. **レイテンシクロスフェード**:
   - `ConvolverProcessor.Runtime.cpp`: `activeCrossfadeGain`（LinearRamp, 線形）

**等パワー補正の計算**:

```cpp
// 現在: 線形クロスフェード（中点で -3dB ディップ）
gOld = 1.0 - gNew;
// 修正案: 等パワークロスフェード
// equalPowerSin は ConvolverProcessor.Runtime.cpp:20-25 に定義済み
gNew_eq = equalPowerSin(gNew);
gOld_eq = equalPowerSin(1.0 - gNew);
```

---

### 🟠 H-05: `finalizeNUCEngineOnMessageThread` の例外処理不足 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/convolver/ConvolverProcessor.LoadPipeline.cpp:561-643` |
| **検証ツール** | grep |

```cpp
try {
    auto newConv = convo::aligned_make_unique<StereoConvolver>();
    // ... MKL初期化、IR設定 ...
    if (newConv->init(...)) {
        applyNewState(newConv.release(), ...);
    } else {
        handleLoadError("Failed to initialize NUC engine...");
    }
}
catch (const std::bad_alloc&) {
    handleLoadError("Failed to initialize NUC engine...");
}
// ← std::runtime_error, std::invalid_argument 等の例外が未捕捉
// ← catch (...) なし
```

**影響**: `std::runtime_error` 等が発生した場合、`std::terminate()` が呼ばれアプリケーションが異常終了する。

---

### 🟡 M-01: `std::abs` をAudio Threadで使用 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/MKLNonUniformConvolver.cpp:1403` |
| **検証ツール** | grep |

```cpp
// Get() 関数内の addScaledFallback ラムダ（Audio Thread実行）
auto addScaledFallback = [&addFallback](int n, double* dst, const double* src, double gain) noexcept
{
    if (std::abs(gain - 1.0) < 1.0e-12)  // ← libm呼び出し
    {
        addFallback(n, dst, src);
        return;
    }
    for (int i = 0; i < n; ++i)
        dst[i] += src[i] * gain;
};
```

プロジェクト内に `absNoLibm`（`AudioEngine.h:117`）が定義済みのため、置き換え可能。

---

### 🟡 M-02: DCブロッカーカットオフ不統一 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/audioengine/AudioEngine.h:369-388` |
| **検証ツール** | grep |

```cpp
struct DCBlockerRuntimeState
{
    convo::UltraHighRateDCBlocker outputL, outputR;
    convo::UltraHighRateDCBlocker inputL, inputR;
    convo::UltraHighRateDCBlocker oversampledL, oversampledR;

    void init(double sampleRate, double processingRate) noexcept
    {
        outputL.init(sampleRate, 3.0);       // 3 Hz
        outputR.init(sampleRate, 3.0);       // 3 Hz
        inputL.init(sampleRate, 3.0);        // 3 Hz
        inputR.init(sampleRate, 3.0);        // 3 Hz
        oversampledL.init(processingRate, 1.0);  // ← 1 Hz（異なる）
        oversampledR.init(processingRate, 1.0);  // ← 1 Hz（異なる）
    }
};
```

**影響**: オーバーサンプリング段の DC ブロッカーのカットオフが 1 Hz と低いため、低域位相特性が入力/出力段と異なる。ただし oversampled 段のサンプルレートは高いため、正規化カットオフはさらに低く、実質的な影響は微小。

---

### 🟡 M-03: `Fixed15TapNoiseShaper` の毎サンプル全状態スキャン ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/Fixed15TapNoiseShaper.h` |
| **検証ツール** | grep |

**`processStereoBlock` 内**（L/R各サンプルごとに行170-183）:

```cpp
const double absErr = absNoLibm(error);
if (absErr > peakAbs)
    peakAbs = absErr;
```

**`processSample` 内**（行217-223、**毎サンプル実行**）:

```cpp
double maxAbs = 0.0;
for (int i = 0; i < ORDER; ++i)  // ORDER = 16
{
    const double absVal = absNoLibm(channelErrors[static_cast<size_t>(i)]);
    if (absVal > maxAbs)
        maxAbs = absVal;
}
```

**コスト試算**: 512サンプルブロック処理時、L/R チャンネル合わせて `512 × 2 × 16 = 16,384回` の `absNoLibm` + 比較が `processStereoBlock` + `processSample` の両方で発生。

---

### 🟡 M-04: `observeMonotonicRollbackRequested_` フラグのリセット欠落 ❌ 誤報告

**エビデンス**:

- セット: `AudioEngine.h:2359`: `publishAtomic(observeMonotonicRollbackRequested_, true, ...)`
- **リセット**: `AudioEngine.Commit.cpp:352`: `convo::exchangeAtomic(observeMonotonicRollbackRequested_, false, std::memory_order_acq_rel)`

`exchangeAtomic(..., false, ...)` はアトミックに値を `false` に置き換え、変更前の値を返す。この操作によりフラグは正しくリセットされる。**報告書の「一度も false にリセットされない」は誤り**。

---

### 🟡 M-05: `wetBuf[0]` の危険な流用 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/convolver/ConvolverProcessor.Runtime.cpp:358` |
| **検証ツール** | grep |

```cpp
double* delayFadeRamp = wetBuf[0];  // Wet信号バッファをゲインカーブ格納用に流用
```

本来 Wet 信号（畳み込み出力）用のバッファを、レイテンシクロスフェードのゲイン値格納に流用している。`delayBuf` や `dryBuf` とは異なり、`wetBuf[0]` は畳み込みエンジンが書き込む可能性があるバッファであるため、上書きのリスクがある。

---

### 🔵 L-01: レベルメーター測定タイミング ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:520-524` |
| **検証ツール** | grep |

```cpp
const float outputLinear = measureLevel(originalBlock);  // ← ①先に測定
// ...
processOutputDouble(buffer, numSamples, state);           // ← ②ヘッドルーム適用
```

`processOutputDouble` 内（行545）:

```cpp
constexpr double kOutputHeadroom = 0.8912509381337456;  // ≈ -1.0 dB
// ...
dataL[i] *= kOutputHeadroom;
```

**影響**: レベルメーターの表示値が実際より約 +1dB 高くなる。

---

### 🔵 L-02: `applySpectrumFilter` の HC `else` ブランチ到達不能 ✅ 確認

| 項目 | 内容 |
|------|------|
| **エビデンスファイル** | `src/MKLNonUniformConvolver.cpp:305-370` |
| **検証ツール** | grep |

```cpp
const double hcFcEnd = nyquist;                      // ← nyquist = fs/2
const int kEnd = std::min(halfN,
    static_cast<int>(std::round(hcFcEnd * N / fs))); // = halfN
// ...
for (int k = 0; k < cSize; ++k)                      // cSize = halfN + 1
{
    if (k <= kStart) { /* passband */ }
    else if (k <= kEnd) { /* taper */ }              // k <= halfN
    else { /* stopband -> gain[k] = 0.0 */ }         // k > halfN → NEVER REACHED
}
```

`cSize = complexSize = halfN + 1` に対し、ループ変数 `k` の最大値は `halfN`。`kEnd = halfN` のため `k > kEnd` は決して真にならない。

---

### 🔵 L-03: 命名/コメントの誤り ✅ 確認

| # | ファイル | 問題 |
|---|----------|------|
| 1 | `src/Fixed15TapNoiseShaper.h:32` | `static constexpr int ORDER = 16;` — クラス名は「15-Tap」だが ORDER は 16（実際は16-Tapフィルタ） |
| 2 | `src/eqprocessor/EQProcessor.Coefficients.cpp:90` | `// SVF係数計算 (Audio Thread用)` → 実際は `std::pow`, `std::tan` を含むため Message Thread 専用 |

---

## 修正優先順位（再評価版）

### 第1フェーズ（即時対応：リリース前必須）

| 優先順位 | ID | 理由 | 改修方針 |
|:--------:|:--:|------|----------|
| **1** | C-01 | デバッグクラッシュ、Audio Thread安全違反 | `LinearRamp` に `forceSetCurrentAndTargetValueRT()` 追加、または `ASSERT_NON_RT_THREAD` を呼び出しコンテキストに合わせて調整 |
| **2** | C-02 | 未定義動作、全最適化ビルドに影響 | `union` を `std::bit_cast` (C++20) に置換。全 `absNoLibm`, `killDenormal`, `isFiniteNoLibm` 関数が対象 |
| **3** | C-03 | Use-After-Free、クラッシュ原因 | `shared_from_this()` / `weak_ptr` または `weakOwner` 経由で `cancelled` にアクセス |

### 第2フェーズ（早期対応：次リリースまで）

| 優先順位 | ID | 理由 | 改修方針 |
|:--------:|:--:|------|----------|
| **4** | H-04 | 音質に直接影響（中点 -3dB ディップ） | 既存の `equalPowerSin()` 関数を使用して等パワークロスフェード化 |
| **5** | H-05 | 例外安全、リソースリークリスク | `catch (const std::exception&)` と `catch (...)` を追加 |
| **6** | M-01 | RT安全違反（libm呼び出し） | `std::abs` → `absNoLibm` に置換 |
| **7** | H-03 | 潜在バグ、条件次第で顕在化 | 逆順化ループ内で `irFreqReal/irFreqImag` も同時に逆順化 |

### 第3フェーズ（計画的対応）

| 優先順位 | ID | 改修方針 |
|:--------:|:--:|----------|
| **8** | M-02, M-03, M-05 | DCブロッカー統一、インクリメンタル最大値追跡、`delayFadeRamp` 専用バッファ確保 |
| **9** | L-01, L-02, L-03 | 測定タイミング修正、到達不能コード削除/コメント修正 |
| **10** | H-01 | 設計明確化（機能障害なし、優先度低） |

### 修正不要

| ID | 理由 |
|:--:|------|
| **M-04** | 実際には `exchangeAtomic(..., false, ...)` でリセットされている（誤報告） |
| **H-02** | リーク経路はシャットダウン時キュー破棄のみで実質的に問題なし（過大評価） |

---

## 補足：ツール使用感

| ツール | 有用性 | 所感 |
|--------|:------:|------|
| **grep (RTK)** | ⭐⭐⭐⭐⭐ | 最も基本的かつ確実。RTKによるトークン削減効果が大きい |
| **Serena MCP** | ⭐⭐⭐⭐ | シンボル構造解析に優れる。クラス全体のメソッド一覧取得が強力 |
| **CodeGraph MCP** | ⭐⭐⭐⭐ | 依存関係グラフとモジュール構造解析が有用。グローバル検索はコミュニティベースで俯瞰的 |
| **ccc** | ⭐⭐⭐⭐ | ASTベースのセマンティック検索が高速。`ccc search` でコード片の意味的検索が正確 |
| **semble** | ⭐⭐⭐ | 軽量で高速。初回インデックスがやや遅いが、検索はミリ秒級 |
| **Graphify MCP** | ⭐⭐⭐⭐⭐ | DeepSeek backend で \$0.036 という低コストで高品質な知識グラフを構築。God nodes（AudioEngine, ConvolverProcessor等）の特定がアーキテクチャ理解に貢献 |

---

*本レポートは GitHub Copilot (DeepSeek V4 Flash) により自動生成されました。*
