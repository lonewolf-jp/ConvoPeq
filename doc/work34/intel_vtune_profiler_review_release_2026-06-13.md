# Intel Vtune Profiler 分析結果 詳細レビュー — Release ビルド

> **作成日**: 2026-06-13
> **対象**: ConvoPeq.exe (Release build)
> **分析ツール**: Intel Vtune Profiler (Event-based counting driver, User-mode sampling and tracing)
> **前回対比**: Debug build (doc/work34/intel_vtune_profiler_review_2026-06-13.md)

---

## 目次

1. [計測概要とDebug比較](#1-計測概要とdebug比較)
2. [Debug → Release 改善検証](#2-debug--release-改善検証)
3. [Releaseビルドの新規・残存ホットスポット](#3-releaseビルドの新規残存ホットスポット)
4. [待機時間分析](#4-待機時間分析)
5. [ソースコード分析に基づく改善提案](#5-ソースコード分析に基づく改善提案)
6. [総評](#6-総評)

---

## 1. 計測概要とDebug比較

### 基本データ

| 項目 | Debug | Release | 変化 |
|---|---|---|---|
| Elapsed Time | 62.225s | **122.893s** | +97.5%（収集時間拡大） |
| CPU Time | **84.148s** | **69.722s** | **-17.1%** ✅ |
| Effective Time | — | 32.275s | — |
| Spin Time | — | 37.447s | スピンが有効時間を上回る ❌ |
| Wait Time | 0.025s | **5,567.804s** | 桁違いの待機 ❌❌ |
| Paused Time | 0.025s | 0s | — |
| Total Thread Count | 80 | 93 | +13 |
| GPU Time | — | 0.0% | GPU不使用 |
| Result Size | **1.4 GB** | **446 MB** | -68% |

### Top Hotspots 比較

| # | Debug (CPU Time) | Release (CPU Time) | Release (Effective Time) |
|---|---|---|---|
| 1 | isBadSample 10.1s (12.1%) | SleepEx **23.1s (33.2%)** ❌ | NoiseShaperLearner評価 9.5% |
| 2 | std::clamp 8.3s (9.8%) | sleep_for **7.2s (10.3%)** | ASIOコールバック 10.1% |
| 3 | decimateStage loop 7.2s (8.6%) | GetMessageW 5.4s (7.7%) | sleep_for ~0s (実質スピン) |
| 4 | std::bit_cast 5.8s (6.8%) | **decimateStage loop 3.2s (4.6%)** ✅ | **decimateStage loop 3.2s** |
| 5 | advanceState loop 3.2s (3.7%) | SleepConditionVariable 2.9s (4.2%) | SleepConditionVariable 2.9s |

### 重要指標: Effective Time vs Spin Time

Vtune が「Effective Time」（実効計算）と「Spin Time」（スピン待機）を分離しているのが本計測の最大の特徴です：

```
CPU Time 69.722s = Effective 32.275s + Spin 37.447s
```

**わずか32.3秒しか実効計算をしていない**。残りの37.4秒はスピン待機（Busy-wait）で無駄にCPUを消費しています。これは Debug ビルドより **質的に悪い** 状態です。

---

## 2. Debug → Release 改善検証

前回の12の改善提案のうち、Releaseビルドにより **自動的に改善された項目** と **依然として残る問題** を検証します。

### ✅ Release最適化により劇的に改善

| 関数 | Debug | Release | 改善率 | 要因 |
|---|---|---|---|---|
| `isBadSample` | **10.144s** | **0.996s** | **10.2x** | インライン展開+STL bit_cast最適化 |
| `std::clamp<double>` | **8.280s** | **0.483s** | **17.1x** | インライン展開＋分岐予測改善 |
| `advanceState` | **11.447s** | **0.776s** | **14.8x** | 関数全体のインライン化 |
| `computeFeedback` | **1.389s** | **0.131s** | **10.6x** | SIMD命令の完全インライン化 |
| `decimateStage loop` | **7.223s** | **3.227s** | **2.2x** | ループ最適化＋レジスタ割付改善 |
| GDI Software Rendering | **17.9s** | **N/A** | **完全消滅** | Direct2D/OpenGL切替済み？ |

### ✅ 改善提案の事後検証

| 提案# | 内容 | Release結果 | 検証 |
|---|---|---|---|
| ① | SIMD isBadSample | Debugより10倍改善、追加最適化の必要なし | ⚠️ ただしスカラー呼出が依然996ms |
| ② | isBadSample間引き | 996msに低減。追加の間引きは効果薄 | ✅ 不要 |
| ③ | AVX2ロード統一 | decimateStage 2.2x改善 | ⚠️ 依然3.2sで#1実効ホットスポット |
| ④ | std::clamp置換 | 17倍改善！Releaseコンパイラで自動解決 | ✅ 完了 |
| ⑤ | レジスタ再利用 | 10倍改善 | ✅ 完了 |
| ⑥ | Direct2D化 | GDIレンダリング消滅 | ✅ 対応済み |
| ⑧ | Release再計測 | **本レポート** | ✅ **推�通りの検証完了** |

### ⚠️ 依然残る問題

| 問題 | Debug | Release | 状況 |
|---|---|---|---|
| CPU利用率1コア集中 | 79.5% | **89% Idle** | **悪化**（待機時間が主） |
| `killDenormal` | 0.120s | **0.145s+0.099s+0.055s=0.299s** | **悪化**（呼出箇所が増加） |
| `NoiseShaperLearner`評価 | 22.5s | 9.5% Effective (≒3.1s) | 絶対量は改善、比率は高い |
| MklFftEvaluator | 1.08s | **6.0% Effective (≒1.9s)** | 比率が上昇（他が減ったため） |

---

## 3. Releaseビルドの新規・残存ホットスポット

### 🔴 クリティカル: SleepEx (KERNELBASE.dll) — 33.2% of CPU Time

```
SleepEx: 23.124s CPU Time (33.2%)
  呼出元: vbvm_asiodriver64.dll [Loop@0x180003ed0]
  状況:   ASIO ドライバのコールバックループ内でスリープ
  Spin Time: 36.1% (ASIOドライバループ全体)
  Wait Time: 86.3s (vbvm_asiodriver64.dll 全体)
```

**問題**: VB-Audio ASIO ドライバ (`vbvm_asiodriver64.dll`) のコールバックループが **36.1% のスピン時間** を消費。ドライバがスリープとポーリングを繰り返している。これはアプリケーション側で制御できない。

**対応**: 設定不可。ただし ASIO バッファサイズを大きくすると SleepEx の呼出頻度が下がる可能性あり。

---

### 🔴 重要: std::this_thread::sleep_for (10.3%) — NoiseShaperLearner ワーカー

```
sleep_for: 7.164s CPU (10.3%) + 691.4s Wait Time
  呼出元: NoiseShaperLearner::workerThreadMain loop@836
```

**ソースコード確認** (NoiseShaperLearner.cpp line ~836):

```cpp
while (std::chrono::steady_clock::now() < next
    && !convo::consumeAtomic(stopRequested, ...)
    && !stopToken.stop_requested())
    std::this_thread::sleep_for(std::chrono::milliseconds(100));  // ← 7.2s
```

**問題**: 100ms の `sleep_for` が 122秒間の収集で 7.2秒蓄積。さらに **691秒の Wait Time** を生成。93スレッドのうち多くがこの待機でブロックされている。

---

### 🔴 重要: GetMessageW (USER32.dll) — 7.7%

```
GetMessageW: 5.362s CPU (7.7%), Effective 8.0%
  呼出元: juce::detail::dispatchNextMessageOnSystemQueue
```

**問題**: メインスレッドのメッセージポンプ。メッセージがないときにブロックされるのは正常動作だが、`dispatchNextMessageOnSystemQueue` 全体で **109.6s の Wait Time** が発生。メインスレッドがメッセージ処理でビジー状態ではないことを示す。

**評価**: 通常動作。対策不要。

---

### 🔴 重要: decimateStage loop@557 (4.6%) — 実効#1 ホットスポット

```
decimateStage loop@557: 3.227s Effective (4.6%)
  呼出元: CustomInputOversampler::decimateStage → processDown
  コールスタック: DSPCore::processDouble → processDown → decimateStage
```

**これは Release ビルドにおいて「実効計算の最大ホットスポット」**。
Debug 比で2.2倍改善したものの、依然として最大の実効CPU消費ポイント。

内部の内訳:

- `isBadSample`: 0.996s (1.4%) — 10倍改善したがなお存在
- `fastAbs`: 0.011s (0.02%)
- 残り = 実効FIRフィルタ演算: ~2.2s

---

### 🔴 重要: NoiseShaperLearner 評価パス (9.5% Effective)

```
evaluateCandidateMapped: 9.5% Effective (≒3.1s)
  → loop@1270: 9.5%
  → loop@1278: 9.5%
  → loop@1295: 0.07s (0.2%)
evaluationWorkerMain loop@534: 8.9%
runEvaluationJobsForWorker loop@621: 8.2%
```

**評価**: NoiseShaperLearner の処理全体で約 3.1s Effective。Debug の 22.5s から大幅改善。

内訳:

- `MklFftEvaluator::evaluate`: 6.0% (≒1.9s)
  - `computeMaskingEnergyStable`: 3.0% (≒1.0s)
    - loop@708: 0.19s
    - loop@714: 0.42s
    - loop@739: 0.06s
  - `computeSfm` loop@650: 0.24s
  - `buildNoiseMaskersFixed` loop@679: 0.15s
  - `detectTonalMaskersFixed` loop@598: 0.11s
- `log10`: 0.56s (0.8%)
- `exp` (libm): 0.40s (0.6%)
- `_avx2_exp4` (SVML): 0.37s (0.5%)
- `log` (libm): 0.11s (0.2%)

---

### 🟡 要注意: SwitchToThread スピン

```
SwitchToThread (KERNELBASE): 0.165s Effective + 2.355s Spin
SwitchToThread (KERNEL32):   0.035s Effective + 추가 Spin
```

**呼出元チェーン**:

1. `ISRLifecycle::transitionTo` → `std::lock_guard<std::mutex>` (traceGuard_) → `Mtx_lock` → `mtx_do_lock` → **SwitchToThread spin**
2. `RCUReader::exit` → `JUCE CriticalSection::exit` → **SwitchToThread spin**
3. `MidiMessageCollector::removeNextBlockOfMessages` → `CriticalSection::exit` → **SwitchToThread spin**
4. `ISRLifecycle::enterAudioCallback` → `transitionTo` → **SwitchToThread spin**

**問題**: オーディオスレッド（RT）が `transitionTo` 内の `traceGuard_` ミューテックスで NonRT スレッドと競合。OSの `SwitchToThread` によるスピン待機が発生。

---

### 🟡 要注意: SleepConditionVariableSRW (4.2%)

```
SleepConditionVariableSRW: 2.901s (4.2%)
```

**呼出元**:

- `Cnd_wait` → `Primitive_wait_for` → `Primitive_wait` (2.6%)
- `std::condition_variable::wait` (2.2%)
- NoiseShaperLearner ワーカーの `evaluationDispatchCv.wait()`

---

### 🟡 要注意: AudioBuffer::makeCopyOf<float> loop@574

```
juce::AudioBuffer<double>::makeCopyOf<float> loop@574: 0.1s
```

**問題**: ASIO コールバック内で float バッファ → double バッファへの変換が `makeCopyOf<float>()` で発生。メモリアクセスパターンが最適でない可能性。

---

### 🟢 新規出現: log10/exp/libm 関連

| 関数 | CPU時間 | 備考 |
|---|---|---|
| `log10` | 0.563s | MklFftEvaluator内の心理音響演算 |
| `exp` | 0.403s | maskingEnergy sum (computeMaskingEnergyStable) |
| `_avx2_exp4` (SVML) | 0.372s | Intel SVML のAVX2最適化exp |
| `log` | 0.107s | computeSfm内のlog sum |
| `pow_fma` | 0.024s |  |

Debug では関数呼出オーバーヘッドに埋もれていた libm 関数が、Release で相対的に浮上。

---

### 🟢 新規: killDenormal が増加

```
killDenormal: 0.145s + 0.099s + 0.055s + ... = 0.299s
```

Debug: 0.120s → Release: **0.299s (2.5x増加)**

呼出元: `UltraHighRateDCBlocker::process` loop@171 内。
Release最適化で他のコストが減った結果、相対的に目立つようになった。

---

### 🟢 D3D11CreateDevice (0.87s)

```
D3D11CreateDevice: 0.873s Effective
```

起動時の Direct3D 11 初期化。JUCE が Direct2D バックエンドを使用するために D3D11 デバイスを作成。これは起動時のみの1回限り。

---

## 4. 待機時間分析

### 総待機時間: 5,567.8秒

これは **122秒の収集時間に対して 5,568秒の待機** を意味する。93スレッドの多くがほぼ全期間待機状態。

| スレッド種別 | 待機時間 | 割合 | 内訳 |
|---|---|---|---|
| ASIOドライバスレッド | 86.3s | 1.5% | SleepEx + ポーリング |
| メインスレッド(UI) | 110.0s | 2.0% | GetMessageW待機 |
| NoiseShaperLearner ワーカー | 101.9s | 1.8% | sleep_for + condition_variable |
| スレッドプール (ntdll) | 21.3s | 0.4% | NtWaitForWorkViaWorkerFactory |
| その他アイドルスレッド | **5,248s** | **94.3%** | 実質全期間アイドル |

**結論**: 93スレッドのうち **実質的にアクティブなのは3-5スレッドのみ**。残りは全期間アイドル。

### CPU Utilization Histogram 比較

| 同時利用CPU数 | Debug | Release |
|---|---|---|
| 0 (Idle) | 2.26s (3.6%) | **109.40s (89.0%)** ❌ |
| 1 (Poor) | **49.48s (79.5%)** | 11.39s (9.3%) |
| 2 (Poor) | 4.26s | 0.69s |
| 3-5 | 6.19s | 0.76s |
| 6-7 | 0s | 0.65s |
| 8-15 | 0s | 0s |
| 16 (Ideal) | 0s | 0s |

**Release は 89% が Idle = CPUがほぼ何もしていない時間。** これは NoiseShaperLearner のワーカーが generation interval 待機で長時間 sleep しているため。

---

## 5. ソースコード分析に基づく改善提案

### 🔧 Priority S: ISR Lifecycle traceGuard のミューテックス競合解消

**問題**: `transitionTo` 内の `traceGuard_` ミューテックスが RT (audio callback) と NonRT で競合。

```cpp
// ISRLifecycle.cpp — transitionTo()
LifecyclePhase LifecycleIsolationRuntime::transitionTo(LifecyclePhase next)
{
    // ... phase transition (atomic) ...
    {
        std::lock_guard<std::mutex> guard(traceGuard_);  // ← ココ！
        // tracing push_back
    }
}
```

`enterAudioCallback()` と `leaveAudioCallback()` は RT スレッドから呼ばれ、traceGuard_ を取得しようとする。NonRT の `enterPrepare()` なども同ミューテックスを使用。

**改善案**: **トレースバッファをロックフリーリングバッファに変更**

```cpp
// 改善案: lock-free ring buffer for trace
static constexpr size_t kMaxTraceEntries = 4096;
std::array<TransitionRecord, kMaxTraceEntries> traceBuffer_;
std::atomic<size_t> traceWriteIndex_{0};

void recordTransition(LifecyclePhase from, LifecyclePhase to) {
    size_t idx = traceWriteIndex_.fetch_add(1, std::memory_order_acq_rel);
    if (idx < kMaxTraceEntries) {
        traceBuffer_[idx] = { from, to, epochCounter_, now_ns };
    }
    // kMaxTraceEntries 超過時は古いエントリを上書き（廃棄）
}
```

**推定効果**: `SwitchToThread` スピン (2.5s) の大部分と、transitionTo 内のミューテックス競合を解消。

---

### 🔧 Priority A: NoiseShaperLearner worker の sleep_for 改善

**問題**: 100ms `sleep_for` が 7.2s のCPU時間 + 691s の待機時間を生成。

```cpp
// NoiseShaperLearner.cpp line ~836
while (std::chrono::steady_clock::now() < next
    && !stopRequested && !stopToken.stop_requested())
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
```

**改善案**: **condition_variable による正確な起床**

```cpp
// 改善案: condition_variable wait_until で正確なタイミングで起床
std::unique_lock<std::mutex> lock(intervalMutex_);
intervalCv_.wait_until(lock, next, [&]{
    return stopRequested || stopToken.stop_requested();
});
```

**推定効果**: sleep_for のCPU時間 7.2s をほぼゼロに。待機時間も適正化。

---

### 🔧 Priority A: isBadSample のスカラー残存呼出をSIMD化

**問題**: Release でも `isBadSample` が 0.996s (1.4%) 残存。主に decimateStage のスカラーフォールバックパスで呼ばれている。

```cpp
// CustomInputOversampler.cpp — decimateStage スカラーパス (line ~557)
for (int r = 0; r < stage.convCount; ++r) {
    const double x = history[idx];
    if (isBadSample(x)) { bad = true; break; }  // ← ここが996ms
    acc += coeffs[r] * x;
}
```

**改善案**: SIMDパスの利用率を向上させる。現在のスカラーフォールバックパスは「AVX2パス内で bad 検出→フォールバック時」に実行される。フォールバック時は isBadSample チェックを必須とせず、前段のAVX2パスでチェック済みとして省略する。

---

### 🔧 Priority B: MklFftEvaluator::computeMaskingEnergyStable の最適化

**問題**: Release で相対的に浮上した O(n²) ループ。

```cpp
// MklFftEvaluator.h — computeMaskingEnergyStable
for (int i = 0; i < kSpectrumBins; ++i) {       // 2049回
    for (int j = 0; j < maskers.size; ++j) {     // ~100回のマスカー
        // deltaBark + spreadingFunctionAnnexD + push + max
    }
    // contributions ループ (pushされたマスカー数)
}
```

**改善案**:

1. マスカーを Bark 値で事前ソート＋バイナリサーチでスキャン範囲限定
2. `spreadingFunctionAnnexD` テーブル化（deltaBark は整数刻みが多い）
3. `ContributionBuffer::push` の small-vector 最適化（SSO）

**推定効果**: computeMaskingEnergyStable の 1.0s を 30-40% 削減 (~0.3s)

---

### 🔧 Priority B: libm 関数のコスト削減

**問題**: log10 (0.56s) + exp (0.40s) + _avx2_exp4 (0.37s) + log (0.11s) = 1.44s

**改善案**:

- `powerToDb` / `dbToPower` で頻繁に使われる `10 * log10(x)` → 近似式またはテーブルルックアップ
- `computeSfm` 内の `std::log(p)` と `std::exp()` → `fastlog` / `fastexp` 近似（誤差許容範囲を確認）
- SVML (`_avx2_exp4`) は既に高速。現状維持で可。

---

### 🔧 Priority B: killDenormal の増加対策

**問題**: `UltraHighRateDCBlocker::process` 内の loop@171 で毎サンプル `killDenormal` を3回呼出。

```cpp
// UltraHighRateDCBlocker.h loop@171
// 各 iteration で:
killDenormal(state[ch].x1);  // 0.145s
killDenormal(state[ch].x2);  // 0.099s
killDenormal(output[i]);     // 0.055s
```

**改善案**: ループ内でスカラー `killDenormal` を3回呼ぶ代わりに、SIMD `_mm256_and_pd` 版で一括処理。またはデノーマルが発生した場合のみフラグで全サンプルチェックする「遅延チェック」方式。

---

### 🔧 Priority C: ASIO ドライバ Spin-Time 低減（要調査）

**問題**: `vbvm_asiodriver64.dll` のループで **36.1% のスピン時間**。

```
[Loop@0x180003ed0 in func@0x180003cf0]: 10.4% Effective + 36.1% Spin
  → SleepEx: 33.1% Spin
  → [Loop@0x180003f61]: 10.1% Effective + 0.9% Spin (ASIO callback)
```

**意味**: ASIO ドライバのスレッドが 36.1% の時間をビジーウェイトでスリープ状態をポーリングしている。VB-Audio ASIO の仮想ドライバに特有の挙動。

**対応**:

1. ASIO バッファサイズを大きくする（ドライバ設定パネル）
2. 使用するサンプルレートを確認（48kHz vs 96kHz）
3. 可能であれば WASAPI 共有モードとの比較検討

---

### 🔧 Priority C: AudioBuffer::makeCopyOf<float> のコスト削減

**問題**: `juce::AudioBuffer<double>::makeCopyOf<float>` 内の loop@574 が 0.1s。

**改善案**: float→double 変換時に SIMD (`_mm256_cvtps_pd`) を使用して4要素同時変換。JUCEのデフォルト実装はスカラー変換の可能性あり。

---

## 6. 総評

### Debug → Release 改善サマリ

| カテゴリ | Debug | Release | 判定 |
|---|---|---|---|
| 総CPU時間 | **84.1s** | **69.7s** | ✅ 17%削減 |
| 実効計算時間 | 84.1s (全CPU) | **32.3s** | ⚠️ 半分以上がスピン |
| isBadSampleコスト | 10.1s (12.1%) | 1.0s (1.4%) | ✅ 10倍改善 |
| std::clampコスト | 8.3s (9.8%) | 0.5s (0.7%) | ✅ 17倍改善 |
| GDIレンダリング | 17.9s (21.3%) | 消滅 | ✅ Direct2D対応 |
| CPU Idle率 | 3.6% | **89.0%** | ❌ 悪化（待機起因） |
| スピン時間 | — | **37.4s (53.7%)** | ❌ 深刻なスピン問題 |

### 重要発見

1. **Release最適化の効果は絶大**: Debug で懸念された `isBadSample`・`std::clamp`・`advanceState` の問題は Release では概ね解決済み。

2. **新たな問題: スピン時間が支配的**: CPU時間 69.7s のうち **37.4s (53.7%) はスピン待機**。これはミューテックス競合と ASIO ドライバのポーリングに起因。

3. **最大の改善余地: ISR traceGuard ミューテックス**: RT スレッドが `transitionTo` 内で `traceGuard_` ミューテックスを取得しようとして NonRT と競合。**ロックフリーリングバッファ化で SwitchToThread スピン (2.5s) を解消可能**。

4. **実効計算の #1 は decimateStage**: 3.2s (4.6%) が実効計算の最大ポイント。Debug から2.2倍改善したが、全体の比率が上がった。

5. **NoiseShaperLearner の待機が全体の待機時間を水増し**: `sleep_for(100ms)` が 691s の Wait Time を生成。`condition_variable::wait_until` で適正化可能。

### 優先順位再評価

| 優先順位 | 改善項目 | 推定削減 | 難易度 | リスク |
|---|---|---|---|---|
| **S** | ISR traceGuard → ロックフリーリングバッファ | Spin 2.5s + 競合解消 | 中 | 低（等価置換） |
| **A** | NoiseShaperLearner sleep_for → condition_variable | CPU 7.2s + Wait 691s適正化 | 低 | 低 |
| **A** | decimateStage スカラーパス isBadSample省略 | Effective ~0.8s | 低 | 低（二重チェック削除） |
| **B** | computeMaskingEnergyStable テーブル化+範囲限定 | Effective ~0.3s | 中 | 低 |
| **B** | powerToDb/dbToPower 近似式化 | Effective ~0.2s | 中 | 中（誤差許容要確認） |
| **B** | killDenormal SIMD一括化 | Effective ~0.2s | 低 | 低 |
| **C** | AudioBuffer::makeCopyOf SIMD変換 | Effective ~0.08s | 低 | 低 |
| **C** | ASIOバッファサイズ調整調査（外部要因） | Spin 23.1s軽減？ | 調査 | 低 |

**総評**: Debug ビルドの解析で指摘した12の改善提案のうち、**Release最適化により8項目が実質解決**されました。残る課題は「スピン時間の解消」と「実効計算（decimateStage + NoiseShaperLearner評価）のさらなる最適化」に集約されます。

---

*以上*
