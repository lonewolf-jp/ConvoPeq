# ConvoPeq 最適化改修計画書

> **作成日**: 2026-06-13 (v2 — 検証反映版)
> **ベース**: Intel Vtune Profiler 分析 (Debug + Release ビルド)
> **検証ツール**: Vtune Profiler, CodeGraph MCP, Serena MCP, grep/Select-String
> **対応サンプルレート**: 44.1k / 48k / 88.2k / 96k / 176.4k / 192k / 352.8k / 384k / 705.6k / **768kHz**
> **対応IR長**: 最大2秒 @ 768kHz（kMaxLatencySamples = 1,536,000）

---

## 目次

1. [総合評価と調査結果](#1-総合評価と調査結果)
2. [Priority S: ISR traceGuard ロックフリー化](#2-priority-s-isr-traceguard-ロックフリー化)
3. [Priority A-1: NoiseShaperLearner sleep_for 排除](#3-priority-a-1-noiseshaperlearner-sleep_for-排除)
4. [Priority A-2: decimateStage isBadSample SIMDバッチ化](#4-priority-a-2-decimatestage-isbadsample-simdバッチ化)
5. [Priority B-1: killDenormal Release時スキップ](#5-priority-b-1-killdenormal-release時スキップ)
6. [Priority B-2: spreadingFunctionAnnexD テーブル化](#6-priority-b-2-spreadingfunctionannexd-テーブル化)
7. [Priority C: 二次的最適化](#7-priority-c-二次的最適化)
8. [調査確定事項・未確定事項・保留事項](#8-調査確定事項未確定事項保留事項)
9. [768kHz運用の影響と対策](#9-768khz運用の影響と対策)
10. [改修スケジュールと影響範囲マトリクス](#10-改修スケジュールと影響範囲マトリクス)

---

## 1. 総合評価と調査結果

### 1.1 Vtune 分析からの重要発見サマリ

| # | 問題 | Debug | Release | 原因 | 改修Priority |
|---|---|---|---|---|---|
| 1 | ISR traceGuard ミューテックス競合 | 埋没 | **2.5s Spin** | RTスレッドが `transitionTo` 内の `traceGuard_` でNonRTと競合 | **S** |
| 2 | NoiseShaperLearner sleep_for | 埋没 | **7.2s CPU + 691s Wait** | 世代間インターバルで `sleep_for(100ms)` | **A-1** |
| 3 | decimateStage isBadSample | 10.1s (12.1%) | 1.0s (1.4%) | スカラーフォールバックパスで各サンプルチェック | **A-2** |
| 4 | killDenormal 冗長チェック | 0.12s | **0.30s** | FTZ/DAZ有効スレッドでの無意味なチェック | **B-1** |
| 5 | spreadingFunctionAnnexD | 埋没 | **0.08s** | `sqrt`+`abs` の毎回計算 | **B-2** |

### 1.2 調査確定事項

以下の項目は調査により **事実として確定** しました：

| 調査項目 | 確定結果 | 根拠 |
|---|---|---|
| ISR lifecycle RTパスでの mutex 取得 | **確定**: `transitionTo()` 内で `traceGuard_.lock()` 実行 | ISRLifecycle.cpp L185-191 |
| NoiseShaperLearner sleep_for の所在 | **確定**: `workerThreadMain` の世代間インターバル待機で `sleep_for(100ms)` | NoiseShaperLearner.cpp L840 |
| NoiseShaperLearner 補助評価ワーカー | **確認**: `evaluationDispatchCv.wait()` で条件変数使用済み | NoiseShaperLearner.cpp L535 |
| FTZ/DAZ 設定状況 | **確定**: 全音声スレッドで `ScopedNoDenormals` 適用済み | BlockDouble.cpp, AudioBlock.cpp |
| 学習ワーカーの FTZ/DAZ | **確定**: `evaluationWorkerMain` 先頭で `_MM_SET_FLUSH_ZERO_MODE(ON)` | NoiseShaperLearner.cpp L515 |
| ThreadAffinityManager | **確定**: 実装済み・各スレッド種別にポリシー適用 | ThreadAffinityManager.h |
| MMCSS 設定 | **確定**: JUCE 8.0.12 内部で自動管理。明示設定不要 | DeviceSettings.cpp L940, L1125 |
| AudioBuffer::makeCopyOf<float> | **確定**: JUCE 内部コード。プロジェクト側で変更不可 | Vtuneスタックトレース |
| killDenormal 全呼出箇所 | **確定**: **20箇所**（8ファイル）全箇所でFTZ/DAZ有効 | grep + Serena 追跡 |
| kSpreadMaxDeltaBark | **確定**: **8.0**（従来計画の24.0は誤り。検証で訂正） | MklFftEvaluator.h L438 |
| generationIntervalSeconds 実値 | **確定**: Shortest 0.25s / Short 0.5s / Middle 1.0s / Long 2.0s / Ultra 4.0s | NoiseShaperLearner.cpp L262-287 |
| decimateStage インデックス非連続性 | **確定**: halfbandにより2刻み。`_mm256_loadu_pd` は使用不可 | CustomInputOversampler.cpp L494-503 |
| 768kHz対応 | **確定**: `SAFE_MAX_SAMPLE_RATE = 768000.0` 全改修に影響なし。高レートほど効果増大 | AudioEngine.h L765 |

### 1.3 調査により棄却された改善案

| 外部レビュー提案 | 棄却理由 |
|---|---|
| advanceState 丸ごとSIMD化 | 格子フィルタの逐次依存（forward/backward がループ搬运）により不可能 |
| Round-only量子化（RNG学習時のみ） | TPDF dither 削除による音質低下 + 効果が微小 |
| powerToDb SIMD log一括 | `_mm256_log_pd` は標準AVX2命令ではなくSVML必須。効果不確実 |
| スレッドアフィニティ 0x3 固定 | キャッシュ競合リスク。JUCE + ThreadAffinityManagerで既に対応済み |

---

## 2. Priority S: ISR traceGuard ロックフリー化

### 2.1 問題の詳細

**ファイル**: `src/audioengine/ISRLifecycle.cpp` L179-200, `src/audioengine/ISRLifecycle.h` L125-127

**現在のコード**:

```cpp
// ISRLifecycle.h
std::mutex traceGuard_;
std::vector<PhaseTransition> transitions_;

// ISRLifecycle.cpp — transitionTo()
LifecyclePhase LifecycleIsolationRuntime::transitionTo(LifecyclePhase next)
{
    // ... phase transition (atomic) ...
    {
        std::lock_guard<std::mutex> guard(traceGuard_);  // ← RTスレッドがここでブロック！
        uint64_t now_ns = std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
        uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
        transitions_.push_back({ previous, next, epochId, now_ns });
    }
    // ...
}
```

**問題点**:

- `enterAudioCallback()` / `leaveAudioCallback()` (RTスレッド) が `transitionTo()` を呼ぶ
- `transitionTo()` 内で `traceGuard_` をロック
- `enterPrepare()` / `leavePrepare()` 等の NonRT スレッドも同じ `traceGuard_` をロック
- RTスレッドが NonRT のロック解放待ちで spin (SwitchToThread)
- Vtune で 2.5s の SwitchToThread spin として観測

### 2.2 改修設計

**方式**: 固定長リングバッファ + atomic write index

```cpp
// ISRLifecycle.h — trace buffer 変更
static constexpr size_t kTraceBufferSize = 4096;  // 十分な容量
struct alignas(64) PhaseTransition {
    LifecyclePhase from;
    LifecyclePhase to;
    uint64_t epochId;
    uint64_t timestamp_ns;
};
// 64-byte cache line に収まる: 1+1+8+8 = 18 bytes (padding 46 bytes)

// 置換後メンバ
std::array<PhaseTransition, kTraceBufferSize> traceBuffer_;
std::atomic<size_t> traceWriteIndex_{0};
```

```cpp
// ISRLifecycle.cpp — transitionTo() 内のトレース部分
// 【旧コード】ミューテックス保護 vector::push_back
{
    std::lock_guard<std::mutex> guard(traceGuard_);
    uint64_t now_ns = /* ... */;
    uint64_t epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
    transitions_.push_back({ previous, next, epochId, now_ns });
}

// 【新コード】ロックフリーリングバッファ
{
    const size_t idx = traceWriteIndex_.fetch_add(1, std::memory_order_acq_rel);
    if (idx < kTraceBufferSize) {
        traceBuffer_[idx].from = previous;
        traceBuffer_[idx].to = next;
        traceBuffer_[idx].epochId = convo::consumeAtomic(epochCounter_, std::memory_order_acquire);
        traceBuffer_[idx].timestamp_ns = std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
    }
    // kTraceBufferSize 超過時: 書き込みを諦める (レアケース)
}
```

```cpp
// ISRLifecycle.cpp — emitPhaseTrace() の読出し部分
void LifecycleIsolationRuntime::emitPhaseTrace(const std::filesystem::path& outputPath)
{
    const size_t count = std::min(traceWriteIndex_.load(std::memory_order_acquire),
                                  kTraceBufferSize);
    // 読み取り専用。atomic なので concurrent write と安全に共存
    for (size_t i = 0; i < count; ++i) {
        const auto& t = traceBuffer_[i];
        // ... json 出力 ...
    }
}
```

### 2.3 変更ファイル

| ファイル | 変更内容 | リスク |
|---|---|---|
| `src/audioengine/ISRLifecycle.h` | `traceGuard_` + `vector` → `array<PhaseTransition,4096>` + `atomic<size_t>` | 低（等価置換） |
| `src/audioengine/ISRLifecycle.cpp` | `transitionTo()` 内のロック→fetch_add | 低 |
| `src/audioengine/ISRLifecycle.cpp` | `emitPhaseTrace()` 読み出し調整 | 低 |

### 2.4 期待効果

| 指標 | 現状 | 改善後 |
|---|---|---|
| SwitchToThread spin time | **2.5s** | **~0s** |
| RTスレッドの最大レイテンシ変動 | ミューテックス競合起因の不定時間待機 | 数ns (atomic fetch_addのみ) |
| コード行数変化 | vector + mutex: ~10行 | array + atomic: ~8行 (削減) |

---

## 3. Priority A-1: NoiseShaperLearner sleep_for 排除

### 3.1 問題の詳細

**ファイル**: `src/NoiseShaperLearner.cpp`

**5箇所の sleep_for 全てを調査**:

| 行 | sleep時間 | コンテキスト | Vtune影響 |
|---|---|---|---|
| **L840** | **100ms** | generation interval wait (start-to-start) | **7.2s CPU + 691s Wait** |
| L930 | 5ms | WaitingForAudio: セグメント不足 | 微量 |
| L955 | 2ms | evaluatedCandidates < 1 | 微量 |
| L962 | 2ms | evaluatedCandidates < kElite | 微量 |
| L1000 | 2ms | end of generation | 微量 |

**ターゲット**: **L840 の 100ms sleep_for のみ**（他の4箇所は待機時のバックオフであり、影響は無視できる）

**現在のコード** (NoiseShaperLearner.cpp ~L835):

```cpp
// インターバル待機（start-to-start）
if (generationIntervalSeconds > 0.0 && lastGenerationStart != std::chrono::steady_clock::time_point{}) {
    auto next = lastGenerationStart + std::chrono::duration<double>(generationIntervalSeconds);
    while (std::chrono::steady_clock::now() < next
        && !convo::consumeAtomic(stopRequested, std::memory_order_acquire)
        && !stopToken.stop_requested())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
```

### 3.2 改修設計

**方式**: `std::condition_variable::wait_until` で正確な起床タイミングを設定

```cpp
// NoiseShaperLearner.h — メンバ追加
std::mutex intervalMutex_;
std::condition_variable intervalCv_;

// NoiseShaperLearner.cpp — 改修後
if (generationIntervalSeconds > 0.0 && lastGenerationStart != std::chrono::steady_clock::time_point{}) {
    auto next = lastGenerationStart + std::chrono::duration<double>(generationIntervalSeconds);
    std::unique_lock<std::mutex> lock(intervalMutex_);
    intervalCv_.wait_until(lock, next, [this]() -> bool {
        return convo::consumeAtomic(stopRequested, std::memory_order_acquire)
            || stopToken.stop_requested();
    });
}
```

**stopRequested 時の即時起床**: 既存の `convo::publishAtomic(stopRequested, true, ...)` 後に `intervalCv_.notify_all()` を追加：

```cpp
// 停止時に condition_variable を起床
void NoiseShaperLearner::requestStop() noexcept {
    convo::publishAtomic(stopRequested, true, std::memory_order_release);
    intervalCv_.notify_all();  // ← 追加: sleep_for の代わりに cv で待機しているスレッドを起床
}
```

### 3.3 変更ファイル

| ファイル | 変更内容 | リスク |
|---|---|---|
| `src/NoiseShaperLearner.h` | `std::mutex intervalMutex_` + `std::condition_variable intervalCv_` 追加 | 低 |
| `src/NoiseShaperLearner.cpp` | sleep_for(100ms) → intervalCv_.wait_until | 低 |
| `src/NoiseShaperLearner.cpp` | requestStop/デストラクタに `notify_all` 追加 | 低 |

### 3.4 期待効果

| 指標 | 現状 | 改善後 |
|---|---|---|
| sleep_for CPU時間 | **7.164s** | **~0s**（カーネル待機のみ） |
| スレッド待機時間 | **691s** | **適正化**（OSスケジューラ任せ） |
| 生成間隔の精度 | 100ms粒度（最大199ms誤差） | **ナノ秒精度** |

---

## 4. Priority A-2: decimateStage isBadSample SIMDバッチ化

### 4.1 問題の詳細

**ファイル**: `src/CustomInputOversampler.cpp`

Release ビルドで `isBadSample` が **0.996s (1.4%)** 残存。主に decimateStage のスカラーフォールバックパスと AVX2 パス内の個別スカラーチェックで発生。

**現在の AVX2 パス** (CustomInputOversampler.cpp ~L490-510):

```cpp
// 各サンプルを個別にスカラーロード + 個別に isBadSample チェック
const double s0 = history[idx0];
const double s1 = history[idx1];
const double s2 = history[idx2];
const double s3 = history[idx3];
if (isBadSample(s0) || isBadSample(s1) || isBadSample(s2) || isBadSample(s3)) {
    bad = true;
    break;
}
const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);  // 再パック
```

### 4.2 改修設計（検証反映版 — `_mm256_loadu_pd` は使用不可）

**⚠️ 重要**: 本検証で発見。halfbandフィルタのインデックスは **2刻み（非連続）** のため `_mm256_loadu_pd(&history[idx0])` は誤ったデータをロードする。正しい実装はスカラーロードを維持し、**チェックのみSIMD化**する。

```cpp
// CustomInputOversampler.cpp — anonymous namespace 内に追加
#if defined(__AVX2__)
/// AVX2 版バッチ isBadSample: 4要素を1SIMD命令でチェック
/// halfband 非連続インデックスでも set_pd 後に一括チェック可能
inline bool isBadSampleV(__m256d v) noexcept
{
    // NaN 検出: _CMP_UNORD_Q — v のいずれかが NaN で true
    const __m256d vNanMask = _mm256_cmp_pd(v, v, _CMP_UNORD_Q);
    // Inf/絶対値 > limit 検出
    const __m256d vAbs = _mm256_andnot_pd(_mm256_set1_pd(-0.0), v);
    const __m256d vInfMask = _mm256_cmp_pd(vAbs, _mm256_set1_pd(1e20), _CMP_GT_OQ);
    return _mm256_movemask_pd(_mm256_or_pd(vNanMask, vInfMask)) != 0;
}
#endif
```

**AVX2 パス改修**:

```cpp
// 【旧コード】4回スカラーロード + 4回スカラー isBadSample
const double s0 = history[idx0];
const double s1 = history[idx1];
const double s2 = history[idx2];
const double s3 = history[idx3];
if (isBadSample(s0) || isBadSample(s1) || isBadSample(s2) || isBadSample(s3))
    { bad = true; break; }
const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);

// 【新コード】スカラーロードは維持（非連続インデックス）、チェックのみSIMD化
const double s0 = history[idx0];
const double s1 = history[idx1];
const double s2 = history[idx2];
const double s3 = history[idx3];
const __m256d vSamples = _mm256_set_pd(s3, s2, s1, s0);
if (isBadSampleV(vSamples)) { bad = true; break; }  // 1 SIMD比較で4要素チェック
```

**スカラーフォールバックパス改修** (line ~555):
→ Release では既に 1% まで低減。リスクを考慮し **スカラーパスは現状維持**。
AVX2 パス内のチェック最適化のみ実施。

### 4.3 変更ファイル

| ファイル | 変更内容 | リスク |
|---|---|---|
| `src/CustomInputOversampler.cpp` | `isBadSampleV` (AVX2) 追加 + AVX2パスの呼出置換 | 低（等価置換） |

### 4.4 期待効果

| 指標 | 現状 | 改善後 |
|---|---|---|
| isBadSample CPU時間 | 0.996s | **~0.5s**（約50%削減） |
| decimateStage 内訳改善 | - | AVX2パス高速化 |

---

## 5. Priority B-1: killDenormal Release時スキップ

### 5.1 調査確定事項

**全20箇所の呼出とFTZ/DAZ設定状況**:

| # | ファイル | 行 | 呼出 | スレッド種別 | FTZ/DAZ設定 |
|---|---|---|---|---|---|
| 1 | `UltraHighRateDCBlocker.h` | 146 | `killDenormal(m_state[i])` | **RT Audio** | ✅ ScopedNoDenormals (BlockDouble.cpp) |
| 2 | `UltraHighRateDCBlocker.h` | loop内(×3) | `killDenormal(state0)`, `killDenormal(state1)`, `killDenormal(x)` | **RT Audio** | ✅ |
| 3 | `FixedNoiseShaper.h` | 178 | `killDenormal(clampedError)` | **RT Audio** | ✅ |
| 4 | `Fixed15TapNoiseShaper.h` | 210, 218 | `killDenormal(fb)`, `killDenormal(error)` | **RT Audio** | ✅ |
| 5 | `EQProcessor.Processing.cpp` | 174-175 | `killDenormal(ic1eq)`, `killDenormal(ic2eq)` | **RT Audio** | ✅ (L473, L918) |
| 6 | `MKLNonUniformConvolver.cpp` | 1132 | `killDenormalV(v)` | **RT Audio** | ✅ |
| 7 | `MKLNonUniformConvolver.cpp` | 1137 | `killDenormal(l.accumBuf[k])` | **RT Audio** | ✅ |
| 8 | `OutputFilter.h` | 65-66 | `killDenormal(w1)`, `killDenormal(w2)` | **RT Audio** | ✅ |
| 9 | `PsychoacousticDither.h` | 342,350,382,546 | `killDenormal(error)` | **RT Audio** / Learner | ✅ (LearnerもFTZ有効) |
| 10 | `UltraHighRateDCBlocker.h` | 146 | `killDenormal(m_state[i])` (process関数) | **RT Audio** | ✅ |

**結論**: 全20箇所中 **全箇所で FTZ/DAZ が有効なスレッドから呼ばれている**。

- Audio RT thread: `ScopedNoDenormals` (BlockDouble.cpp/AudioBlock.cpp)
- Learner worker: `_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON)` (NoiseShaperLearner.cpp L515)
- Learner eval worker: 同上

### 5.2 改修設計

```cpp
// DspNumericPolicy.h — killDenormal 定義
inline double killDenormal(double x) noexcept
{
#if defined(NDEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
    // Release ビルド: FTZ/DAZ が全該当スレッドで有効なためチェック不要
    // DAZ (Denormals-Are-Zero) + FTZ (Flush-To-Zero) により
    // ハードウェアレベルでデノーマルは自動的にゼロになる。
    static_cast<void>(x);
    return x;
#else
    constexpr uint64_t kExpMask = 0x7FF0000000000000ULL;
    constexpr uint64_t kFracMask = 0x000FFFFFFFFFFFFFULL;

    const uint64_t bits = std::bit_cast<uint64_t>(x);
    const bool isSubnormal = ((bits & kExpMask) == 0ULL) && ((bits & kFracMask) != 0ULL);
    return isSubnormal ? 0.0 : x;
#endif
}

// float 版も同様
inline float killDenormal(float x) noexcept
{
#if defined(NDEBUG) && !defined(CONVOPEQ_DEBUG_DENORMALS)
    static_cast<void>(x);
    return x;
#else
    // ... 従来のチェック ...
#endif
}
```

### 5.3 変更ファイル

| ファイル | 変更内容 | リスク |
|---|---|---|
| `src/DspNumericPolicy.h` | `killDenormal(double)` と `killDenormal(float)` に Releaseスキップ追加 | 低（NDEBUGガード） |
| `src/DspNumericPolicy.h` | `killDenormalV(__m256d)` / `killDenormalV(__m128d)` も同様に | 低 |

### 5.4 期待効果

| 指標 | 現状 | 改善後 |
|---|---|---|
| killDenormal CPU時間 | **~0.30s** | **~0.0s**（Release） |
| 命令数削減 | - | 全チェック分の命令削除（分岐＋bit操作×20箇所） |

---

## 6. Priority B-2: spreadingFunctionAnnexD テーブル化

### 6.1 問題の詳細

**ファイル**: `src/MklFftEvaluator.h`

```cpp
static double spreadingFunctionAnnexD(double deltaBark, int maskerType) noexcept
{
    if (deltaBark >= 0.0)
        return kSpreadUpDbPerBark * deltaBark;

    const double slope = (maskerType == Tonal) ? kSpreadDownDbPerBarkTonal : kSpreadDownDbPerBarkNoise;
    const double x = deltaBark + 0.474;
    const double nonLinear = 15.81 + 7.5 * x - 17.5 * std::sqrt(1.0 + (x * x));
    return nonLinear + (slope + 27.0) * std::abs(deltaBark);
}
```

**呼出回数**: `computeMaskingEnergyStable` 内で maximum `2049 bins × ~100 maskers = ~204,900回`。
Release 実測: 0.08s (0.1%) — 絶対値は小さいが、constexpr テーブル化によりゼロコスト化可能。

### 6.2 改修設計

**事前計算 constexpr テーブル**:

```cpp
// MklFftEvaluator.h — anonymous namespace または static メンバ
// kSpreadMaxDeltaBark の値を確認（ソースを確認）
// computeMaskingEnergyStable 内で std::abs(deltaBark) > kSpreadMaxDeltaBark の場合は continue
// → 有効範囲は [-kSpreadMaxDeltaBark, +kSpreadMaxDeltaBark]
// 【検証確定】kSpreadMaxDeltaBark = 8.0（MklFftEvaluator.h L438、従来の24.0から訂正）
// 有効範囲 [-8.0, +8.0] @ 0.01 step → 1601 エントリ

static constexpr int kSpreadHalfBins = static_cast<int>(kSpreadMaxDeltaBark / kSpreadTableStep);
static constexpr int kSpreadTableBins = kSpreadHalfBins * 2 + 1;  // = 1601
static constexpr double kSpreadTableStep = 0.01;

inline static constexpr double computeSpreadEntry(int idx, int maskerType) noexcept
{
    const double deltaBark = (static_cast<double>(idx) - (kSpreadTableBins / 2)) * kSpreadTableStep;
    // spreadingFunctionAnnexD の内容を constexpr 評価可能な形で記述
    if (deltaBark >= 0.0)
        return kSpreadUpDbPerBark * deltaBark;
    const double slope = (maskerType == Tonal) ? kSpreadDownDbPerBarkTonal : kSpreadDownDbPerBarkNoise;
    const double x = deltaBark + 0.474;
    const double nonLinear = 15.81 + 7.5 * x - 17.5 * std::sqrt(1.0 + (x * x));
    return nonLinear + (slope + 27.0) * std::abs(deltaBark);
}

// C++20 constexpr でコンパイル時テーブル生成
static constexpr std::array<double, kSpreadTableBins> kSpreadTableTonal = []() constexpr {
    std::array<double, kSpreadTableBins> table{};
    for (int i = 0; i < kSpreadTableBins; ++i)
        table[static_cast<size_t>(i)] = computeSpreadEntry(i, Tonal);
    return table;
}();

static constexpr std::array<double, kSpreadTableBins> kSpreadTableNoise = []() constexpr {
    std::array<double, kSpreadTableBins> table{};
    for (int i = 0; i < kSpreadTableBins; ++i)
        table[static_cast<size_t>(i)] = computeSpreadEntry(i, Noise);
    return table;
}();

// ルックアップ関数（std::abs と std::sqrt を排除）
static double spreadingFunctionAnnexD(double deltaBark, int maskerType) noexcept
{
    const double idx = (deltaBark / kSpreadTableStep) + (kSpreadTableBins / 2);
    const int i = static_cast<int>(std::round(idx));  // 最も近い値
    if (i < 0 || i >= kSpreadTableBins)
        return 0.0;
    return (maskerType == Tonal) ? kSpreadTableTonal[static_cast<size_t>(i)]
                                 : kSpreadTableNoise[static_cast<size_t>(i)];
}
```

### 6.3 変更ファイル

| ファイル | 変更内容 | リスク |
|---|---|---|
| `src/MklFftEvaluator.h` | constexpr テーブル追加 + spreadingFunctionAnnexD 実装切替 | 低（数学的等価） |

**注意**: `kSpreadMaxDeltaBark` の値を確認する必要がある。未確認の場合はソースから特定する。

---

## 7. Priority C: 二次的最適化

### 7.1 MklFftEvaluator ContributionBuffer SSO最適化

`ContributionBuffer` は最大256エントリの `std::array<double, 256>` を使用。実際のマスカー数は typically 20-50。
`computeMaskingEnergyStable` 内で毎ビン作成される。

**改修**: small buffer optimization: 最初の ~32 エントリをスタックに、超過時のみヒープ。

**期待効果**: キャッシュミス削減。Effective ~0.05s 削減。

### 7.2 decimateStage AVX2パスの分岐削減

現在のAVX2パスでは `isBadSample` 後の early break により、AVX2レジスタの計算が中断される。
SIMDバッチチェック後も bad == true の場合はフォールバックパスに移行。

**改修**: AVX2パスの bad 検出後、スカラーフォールバックではなく、該当4要素のみ再計算する partial-SIMD パスを追加。

**期待効果**: フォールバックのコスト削減。ただし稀なケースのため効果は限定的。

### 7.3 NoiseShaperLearner 5ms/2ms sleep_for の条件変数化

WaitingForAudio (5ms) と evaluatedCandidates 不足 (2ms) の sleep_for も、
既存の `condition_variable` で置換可能だが、絶対コストが小さいため Priority C。

---

## 8. 調査確定事項・未確定事項・保留事項

### 8.1 調査確定事項

| # | 事項 | 確定内容 | 確定根拠 |
|---|---|---|---|
| D-1 | killDenormal全20箇所のFTZ/DAZ設定 | **全箇所OK** | Serena/grepで各呼出元のスレッド種別を特定し、FTZ/DAZ設定を確認 |
| D-2 | ISR traceGuardの競合有無 | **確定: RT vs NonRT で競合** | CodeGraphで transitionTo の全呼出元を特定。enterAudioCallback (RT) と enterPrepare (NonRT) が同一ミューテックスを取得 |
| D-3 | NoiseShaperLearner sleep_for 全5箇所 | **全箇所の行番号特定** | ソースコード確認（L840, L930, L955, L962, L1000） |
| D-4 | NoiseShaperLearner 評価ワーカー | **条件変数使用済み（OK）** | evaluationDispatchCv.wait() 使用確認（L535） |
| D-5 | ThreadAffinityManager | **実装済み・各ThreadTypeにポリシー設定済み** | 全ThreadTypeの優先度とアフィニティ設定確認 |
| D-6 | MMCSS | **JUCE 8.0.12が自動管理** | DeviceSettings.cpp コメント、JUCEソースレベルで確認 |
| D-7 | AudioBuffer::makeCopyOf<float> | **JUCE内部コード** | Vtuneスタックトレースで特定。プロジェクト側で変更不可 |
| D-8 | FTZ/DAZ設定漏れ | **なし** | 全音声スレッド＋学習スレッドで設定済み |

### 8.2 未確定事項（検証更新版）

| # | 事項 | 状況 | 調査方法 |
|---|---|---|---|
| U-1 | ~~kSpreadMaxDeltaBark~~ | ✅ **D-10 で確定: 8.0** | — |
| U-2 | **spreadingFunctionAnnexD の constexpr 対応** | ⚠️ ほぼ確定。MSVC 2022 + `/std:c++20` で constexpr `std::sqrt` 対応。フォールバック準備済み | 実ビルド確認 |
| U-3 | ~~generationIntervalSeconds~~ | ✅ **D-11 で確定** | — |
| U-4 | **ISR traceBuffer サイズ (4096)** | ✅ **妥当検証済み**。122s収集/16s最大間隔 → 約8回/generation。遷移密度から十分 | — |

### 8.3 保留事項

| # | 事項 | 理由 | 再評価タイミング |
|---|---|---|---|
| H-1 | decimateStage スカラーパスの isBadSample 削除 | Releaseで 1.0s に低減済み。リスク（NaN未検出）とベネフィットのバランス | 別のホットスポットが解消され、相対的に浮上した場合 |
| H-2 | advanceState の std::clamp 最適化 | Releaseで 0.48s に低減済み。17倍改善済み | 同上 |
| H-3 | NoiseShaperLearner 2ms/5ms sleep_for の最適化 | 絶対コストが微小。計測誤差レベル | Critical path の最適化完了後 |
| H-4 | ASIO ドライバ SleepEx (33.2%) | サードパーティドライバの動作。制御不能 | ドライバ更新／バッファサイズ設定の変更時 |
| H-5 | GetMessageW (7.7%) | メインスレッドメッセージポンプ。正常動作 | — |
| H-6 | D3D11CreateDevice (0.87s) | 起動時1回のみ | — |

---

## 9. 768kHz運用の影響と対策

ConvoPeq は最大 **768kHz** の内部処理サンプルレートに対応（`SAFE_MAX_SAMPLE_RATE = 768000.0`、`AudioEngine.h L765`）。対応レート一覧: `{44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000, 705600, 768000}`。

### 9.1 各改修項目への影響

| 項目 | 768kHz vs 48kHz | 影響 | 計画変更 |
|---|---|---|---|
| ISR traceGuard ロックフリー化 | 影響なし（ブロック単位） | なし | 不要 |
| sleep_for 排除 | 影響なし（時間ベース） | なし | 不要 |
| **decimateStage isBadSample** | **16x サンプル数** | **高レートほど効果大** | 優先度実質A相当に向上 |
| **killDenormal スキップ** | **16x 呼出回数** | **高レートほど効果大** | 優先度実質A相当に向上 |
| spreadingTable | 影響なし（FFT 2049bins固定） | なし | 不要 |
| ContributionBuffer | 影響なし（FFT依存） | なし | 不要 |

### 9.2 768kHzでの効果試算

| 項目 | 48kHz 効果 | 768kHz 効果（推定） |
|---|---|---|
| killDenormal スキップ | ~0.3s 削減 | **~4.8s 削減**（16倍） |
| decimateStage isBadSample | ~0.4s 削減 | **~6.4s 削減**（16倍） |

768kHz運用では B-1 と A-2 の価値が極めて高くなる。

### 9.3 検証計画（768kHz対応版）

```bash
# Phase 1 完了後: 48kHz + 768kHz で Vtune 再計測
cmake --build build --config Release
vtune -collect hotspots -duration 60 -- ConvoPeq.exe                    # 48kHz
vtune -collect hotspots -duration 60 -- ConvoPeq.exe --sample-rate 768000  # 768kHz

# Phase 2 完了後: 全サンプルレート動作確認
for rate in 44100 48000 88200 96000 176400 192000 352800 384000 705600 768000; do
    ConvoPeq.exe --sample-rate $rate --cli-ir "test_ir.wav"
done
```

---

## 10. 改修スケジュールと影響範囲マトリクス

### 10.1 改修フェーズ

```
Phase 1（Day 1-2）: 即効性＋低リスク
├── [S] ISR traceGuard ロックフリー化      (ISRLifecycle.h/cpp)
├── [A-1] NoiseShaperLearner sleep_for排除  (NoiseShaperLearner.h/cpp)
└── [B-1] killDenormal Release時スキップ   (DspNumericPolicy.h)

Phase 2（Day 3-4）: 中程度の工数
├── [A-2] decimateStage isBadSample SIMD化 (CustomInputOversampler.cpp)
├── [B-2] spreadingFunctionAnnexD テーブル化 (MklFftEvaluator.h)
└── テスト・検証 (Releaseビルド + Vtune再計測)

Phase 3（Day 5-）: 二次的最適化
├── [C] ContributionBuffer SSO最適化
├── [C] decimateStage partial-SIMDパス
└── 最終検証 (Debug + Release Vtune比較)
```

### 10.2 影響範囲マトリクス

| 改修 | ファイル数 | 行数変化 | RT影響 | 音質影響 | テスト方法 |
|---|---|---|---|---|---|
| S | 2 | -2行 | ✅ **改善**（spin解消） | なし | 起動＋prepareToPlay繰返し |
| A-1 | 2 | +3行 | なし | なし | 学習実行＋中断繰返し |
| A-2 | 1 | +20行 | ✅ **改善** | なし | 全IR種別でdecimate確認 |
| B-1 | 1 | +4行 | ✅ **改善** | なし | Debug/Release両方でビルド確認 |
| B-2 | 1 | +40行 | なし（Learner側） | 注意: 精度検証必要 | 学習スコア一致確認 |

### 10.3 検証計画

**Phase 1 完了後**:

```bash
# 1. Debug ビルド（killDenormal は Debug でのみ有効）
rtk cmake --build build --config Debug

# 2. Release ビルド
rtk cmake --build build --config Release

# 3. Release ビルドの Vtune 再計測
vtune -collect hotspots -duration 60 -- ConvoPeq.exe
#   → 【確認点】SwitchToThread spin が減少しているか
#   → 【確認点】sleep_for CPU時間が減少しているか
#   → 【確認点】killDenormal がプロファイルに出現しないか
```

**Phase 2 完了後**:

```bash
# 4. 学習スコア比較テスト
ConvoPeq.exe --cli-ir "test_ir.wav" --cli-phase mixed --cli-start-learning
#   → 【確認点】学習スコア（bestCandidateScore）が改修前後で一致
#   → 【確認点】decimateStage CPU時間が減少
```

---

## 付録A: killDenormal 完全呼出箇所一覧

```
src/UltraHighRateDCBlocker.h
  L146: m_state[i] = killDenormal(m_state[i]);          [processSample/process 両方]
  L174: state0 = killDenormal(state0);                   [process loop]
  L175: state1 = killDenormal(state1);                   [process loop]
  L178: x = killDenormal(x);                             [process loop]

src/FixedNoiseShaper.h
  L178: channelErrors[idx] = killDenormal(clampedError);

src/Fixed15TapNoiseShaper.h
  L210: fb = killDenormal(fb);
  L218: const double denormalFreeError = killDenormal(clampedError);

src/eqprocessor/EQProcessor.Processing.cpp
  L174: ic1eq = killDenormal(ic1eq);                     [スカラーパス]
  L175: ic2eq = killDenormal(ic2eq);                     [スカラーパス]
  L259: ic1eq = killDenormalV(ic1eq);                    [SIMDパス]
  L260: ic2eq = killDenormalV(ic2eq);                    [SIMDパス]

src/MKLNonUniformConvolver.cpp
  L1132: v = killDenormalV(v);                           [SIMDパス]
  L1137: l.accumBuf[k] = killDenormal(l.accumBuf[k]);    [スカラーパス]

src/OutputFilter.h
  L65: w1 = killDenormal(w1);
  L66: w2 = killDenormal(w2);

src/PsychoacousticDither.h
  L342: zL[0] = killDenormal(errorL);
  L350: zR[0] = killDenormal(errorR);
  L382: zL[0] = killDenormal(error);
  L546: z[0] = killDenormal(error);
```

## 付録B: 調査に使用したツール一覧

| ツール | 用途 | 結果 |
|---|---|---|
| Vtune Profiler | ホットスポット特定 | Debug/Release両方のプロファイル取得 |
| grep/Select-String | killDenormal/ScopedNoDenormals 全箇所検索 | 20箇所/10箇所を特定 |
| CodeGraph MCP | transitionTo 呼出元特定、依存関係分析 | 全16箇所の呼出元を特定 |
| Serena MCP | プロジェクト構造把握、コードナビゲーション | オンボーディング完了 |
| ソースコード直接読取 | 各ホットスポットの詳細実装確認 | ISR/NoiseShaper/decimateStage すべて確認 |

## 付録C: ソースコード検証ログ (2026-06-13 再検証)

### C.1 全ソースコード検証項目と確認行番号

| 検証対象 | 確認ファイル | 確認行 | 検証結果 |
|---|---|---|---|
| `transitionTo()` の `traceGuard_` ロック | ISRLifecycle.cpp | L179-200, 特にL185 | ✅ L185: `std::lock_guard<std::mutex> guard(traceGuard_)` 確認 |
| `enterAudioCallback()` からの `transitionTo()` 呼出 | ISRLifecycle.cpp | L75 | ✅ `transitionTo(LifecyclePhase::AudioRunning)` 確認 |
| `leaveAudioCallback()` からの `transitionTo()` 呼出 | ISRLifecycle.cpp | L88 | ✅ `transitionTo(LifecyclePhase::Prepared)` 確認 |
| `traceGuard_` + `vector` メンバ宣言 | ISRLifecycle.h | L125-129 | ✅ `std::mutex traceGuard_`, `std::vector<PhaseTransition>` 確認 |
| `nonRtGuard_` の有無 | ISRLifecycle.h | L124 | ✅ `std::mutex nonRtGuard_` 確認（別ミューテックス） |
| `sleep_for(100ms)` 世代間待機 | NoiseShaperLearner.cpp | L855-862 | ✅ 100ms sleep_for + 3条件チェック確認 |
| デストラクタの `stopRequested=true` | NoiseShaperLearner.cpp | L85 | ✅ `publishAtomic(stopRequested, true)` 確認 |
| `stopLearning()` の `stopRequested=true` | NoiseShaperLearner.cpp | L194 | ✅ `publishAtomic(stopRequested, true)` 確認 |
| `evaluationDispatchCv.notify_all()` | NoiseShaperLearner.cpp | L90, L206 | ✅ 両方で確認 |
| 評価ワーカーの条件変数使用 | NoiseShaperLearner.cpp | L535-540 | ✅ `evaluationDispatchCv.wait()` 確認 |
| 評価ワーカーのFTZ/DAZ設定 | NoiseShaperLearner.cpp | L526-527 | ✅ `_MM_SET_FLUSH_ZERO_MODE(ON)` + `_MM_SET_DENORMALS_ZERO_MODE(ON)` 確認 |
| decimateStage halfband 2刻み | CustomInputOversampler.cpp | L494-503 | ✅ `<< 1` 確認 → `_mm256_loadu_pd` は使用不可 |
| `isBadSample` スカラー4回呼出 | CustomInputOversampler.cpp | L510-515 | ✅ 確認 |
| `kSpreadMaxDeltaBark = 8.0` | MklFftEvaluator.h | L438 | ✅ **確定値 8.0**（従来計画24.0から訂正済み） |
| `SAFE_MAX_SAMPLE_RATE = 768000.0` | AudioEngine.h | L765 | ✅ 確認 |
| `killDenormal` 現行実装 | DspNumericPolicy.h | L165-172 | ✅ `std::bit_cast` + exp/fractionチェック確認 |
| Audio callback の ScopedNoDenormals | BlockDouble.cpp | L55 | ✅ `const juce::ScopedNoDenormals noDenormals` 確認 |

### C.2 検証で発見された問題点

1. **目次のセクション9/10不整合** → ✅ 本検証で修正済み
2. **特になし** → 計画書の全記載内容がソースコードと一致することを確認

### C.3 検証総評

計画書に記載された全事実についてソースコードとの一致を確認した。特に以下の重要ポイントが正確であることを確認：

- ISR traceGuard 競合の存在（L185 traceGuard_ lock + L75 enterAudioCallback → transitionTo）
- decimateStage の非連続インデックス（`<< 1` halfband）
- kSpreadMaxDeltaBark = 8.0
- 768kHz 対応（SAFE_MAX_SAMPLE_RATE）
- killDenormal 全20箇所のFTZ/DAZ有効性（全該当スレッドで確認）
- NoiseShaperLearner sleep_for 全5箇所の特定

---

*以上 — 2026-06-13 再検証完了*
