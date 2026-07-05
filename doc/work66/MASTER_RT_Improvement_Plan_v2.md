# 統合版 オーディオスレッドリアルタイム性向上改修計画書 v2.0

**作成日**: 2026-07-05 | **最終更新**: 2026-07-05
**対象ブランチ**: `main`
**ベース文書**:
- `doc/work66/AudioThread_RealtimeBlockers_Analysis.md`（初回分析）
- `doc/work66/AudioThread_RealtimeBlockers_Verification_Report.md`（検証）
- `doc/work66/RT_Improvement_Plan.md`（改訂計画v1）
- `doc/work66/RT_Complete_Audit_Report.md`（完全監査）

---

## 目次

1. [改修優先度一覧](#1-改修優先度一覧)
2. [P0 — 即時修正](#2-p0--即時修正)
3. [P1 — 優先対応](#3-p1--優先対応)
4. [P2 — 計画的対応](#4-p2--計画的対応)
5. [P3 — 調査・検討](#5-p3--調査検討)
6. [confirmed-rt-safe — 検証済み安全項目](#6-confirmed-rt-safe--検証済み安全項目)
7. [ファイル変更サマリ](#7-ファイル変更サマリ)
8. [スケジュールと依存関係](#8-スケジュールと依存関係)
9. [リスク評価とロールバック計画](#9-リスク評価とロールバック計画)
10. [付録: 現状のコールバックフロー図](#10-付録-現状のコールバックフロー図)

---

## 1. 改修優先度一覧

### 優先度定義

| 優先度 | 基準 | 推定コスト | 件数 |
|--------|------|-----------|:----:|
| **P0** | 確定バグ。全ビルドで音飛びリスクあり | 低（1-2ファイル, 1行） | 1 |
| **P1** | 毎コールバック実行。累積インパクト大。またはリグレッション防止 | 中（2-8ファイル） | 8 |
| **P2** | 影響限定的／診断時のみ。または大規模リファクタリング | 中〜高（3-15ファイル） | 7 |
| **P3** | 未確定リスク。まず調査が必要 | 調査のみ | 2 |

### 一覧表（P0→P1→P2→P3）

| ID | 優先度 | カテゴリ | 発見項目 | 出典 |
|:--:|:------:|---------|---------|:----:|
| **P0-1** | 🔥 P0 | バグ | DSPCoreFloat.cpp `diagLog()` ガード欠如 | 分析/検証/監査 |
| **P1-1** | ⚡ P1 | 最適化 | `std::hash<std::thread::id>` thread_local キャッシュ | 分析/監査 |
| **P1-2** | ⚡ P1 | 最適化 | `CallbackTelemetryScope` の無条件 `getCurrentTimeUs()` 条件化 | 検証/監査 |
| **P1-3** | ⚡ P1 | 最適化 | XRUN検出パスの `getCurrentTimeUs()` サンプリング化 | 検証/監査 |
| **P1-4** | 🛡️ P1 | false sharing | `RTLocalState` 構造体に `alignas(64)` 欠如 | 監査 |
| **P1-5** | 🛡️ P1 | false sharing | `AudioEngine` クラス atomic 変数レイアウト | 監査 |
| **P1-6** | 🛡️ P1 | 防御 | `IppFFTPlanCache::getOrCreate()` RT呼出防止 CI アサーション | 監査 |
| **P1-7** | 🛡️ P1 | 統一 | `MKLNonUniformConvolver.cpp` の `Logger::writeToLog` → `diagLog()` | 監査 |
| **P1-8** | ⚡ P1 | 最適化 | Debug `isAudioThread()` の hash 重複計算抑制 | 監査 |
| **P2-1** | 📋 P2 | 最適化 | Diagnostics 収集のサンプリング前倒し（getCurrentTimeUs 計測自体の間引き） | 改訂v1 |
| **P2-2** | 📋 P2 | 最適化 | `kTraceBufferSize` 超過後の branch 回収 | 改訂v1 |
| **P2-3** | 📋 P2 | 最適化 | `RTCapabilityFirewall` リリースビルド時省略 | 改訂v1 |
| **P2-4** | 📋 P2 | 最適化 | `ScopedNoDenormals` 重複構築の削減（thread_local 状態追跡） | 監査 |
| **P2-5** | 📋 P2 | 最適化 | `AudioCallbackRuntimeScope` atomic 連鎖の評価 | 監査 |
| **P2-6** | 📋 P2 | 最適化 | `RTAllocatorFirewall::markRTContext()` グローバル atomic 低減 | 監査 |
| **P2-7** | 📋 P2 | 最適化 | 診断パス `GetCurrentProcessorNumber()` syscall 削減 | 監査 |
| **P3-1** | 🔍 P3 | 調査 | `musicalSoftClipScalar()` の libm 呼出有無確認 | 監査 |
| **P3-2** | 🔍 P3 | 調査 | MMCSS 初回コールバック syscall defer 可能性 | 分析/改訂v1 |

---

## 2. P0 — 即時修正

### P0-1: DSPCoreFloat.cpp diagLog() `#if` ガード欠如 🔥

**発見経緯**: 初回分析で疑義→検証でガード欠如を確認→全ツールでの網羅調査で唯一の確定バグと断定
**重要度変更**: [LOW-MEDIUM] → **[CRITICAL]**（検証により引上げ）

#### 現状

```cpp
// src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:9-13
namespace
{
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);  // ★ 全ビルドで有効！
}
```

- `[[maybe_unused]]` で警告抑制 → 誤って使われてもコンパイラが検出不可能
- `juce::Logger::writeToLog()` は内部でミューテックス＋ファイルI/O → 音声スレッドで呼ばれたら音飛び

#### 修正

```cpp
// After: #if でガード
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}
#endif
```

#### 改修手順

```
Step 1: #if 追加（1行）
Step 2: cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=0 で Release ビルド通過確認
Step 3: cmake -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1 で Debug ビルド通過確認
```

**影響ファイル**: 1（`DSPCoreFloat.cpp`）
**所要時間**: 0.5h
**リスク**: 極小（#if追加のみ）

---

## 3. P1 — 優先対応

### P1-1: `std::hash<std::thread::id>` thread_local キャッシュ ⚡

#### 対象箇所

| ファイル | 関数 | 変更内容 |
|---------|------|---------|
| `src/core/ThreadHash.h` | **新規作成** | `convo::cachedThreadHash()` thread_local キャッシュ |
| `src/DspNumericPolicy.h:43` | `currentThreadTag()` | → `convo::cachedThreadHash()` |
| `src/core/RCUReader.h:151` | `currentThreadToken()` | → `convo::cachedThreadHash()` |
| `src/core/EpochDomain.h:69` | `registerReaderThread()` | → `convo::cachedThreadHash()` |
| `src/DspNumericPolicy.h:158` | `ASSERT_AUDIO_THREAD()` | → `cachedThreadHash()` 活用（関連最適化） |

#### 設計

```cpp
// src/core/ThreadHash.h（新規）
#pragma once
#include <thread>
#include <cstdint>

namespace convo {
inline uint64_t cachedThreadHash() noexcept
{
    static thread_local const uint64_t s_cachedHash =
        static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return s_cachedHash;
}
} // namespace convo
```

#### 期待効果

| 指標 | 現在 | 改善後 |
|------|------|--------|
| hash 呼出/コールバック | 1-3回 | 0回（thread_local初期化後） |
| 削減時間 | — | ~20-120ns |
| 影響ファイル数 | — | 4(新規1＋修正3) |

#### リスク

- `thread_local` 初回呼出の微小コスト。非RTスレッドでの初回初期化となるため問題なし
- JUCEプラグインDLLは単一DLL内で完結。DLL境界問題なし

---

### P1-2: CallbackTelemetryScope の無条件 getCurrentTimeUs() 条件化 ⚡

#### 対象箇所

| ファイル | 行 | 変更内容 |
|---------|-----|---------|
| `AudioBlock.cpp:100` | `startUs(...)` | `enabled ? getCurrentTimeUs() : 0` |
| `AudioBlock.cpp:106` | `endUs` | `if (enabled && startUs > 0)` ガード追加 |
| `BlockDouble.cpp` | 同様 | Float版と同様の変更 |

#### 修正方針

```cpp
CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
    : engine(owner)
    , samples(numSamplesIn)
    , enabled(owner.isCliProcessingTelemetryEnabled())
    , startUs(enabled ? convo::getCurrentTimeUs() : 0)  // ★ enabled時のみ
{
}

~CallbackTelemetryScope() noexcept
{
    if (enabled && startUs > 0)  // ★ ガード
    {
        const uint64_t endUs = convo::getCurrentTimeUs();
        // ...
    }
}
```

#### 影響

- CLIテレメトリ無効（通常時）: getCurrentTimeUs 2回 → 0回
- CLIテレメトリ有効時: 変更なし

---

### P1-3: XRUN検出 getCurrentTimeUs() サンプリング化 ⚡

#### 対象箇所

| ファイル | 行 | 変更内容 |
|---------|-----|---------|
| `AudioBlock.cpp:500-557` | XRUN検出ブロック | サンプリング間引き or 軽量方式 |
| `BlockDouble.cpp:465-521` | 同様 | 同上 |

#### 方式A（推奨）: サンプリング間引き

```cpp
constexpr uint64_t kXrunSampleMask = 0x3F;  // 1/64
if ((thisCallbackIndex & kXrunSampleMask) == 0)
{
    // 完全なXRUN検出（t0_start / t1_end の getCurrentTimeUs 2回）
}
// 軽量時は getCurrentTimeUs を呼ばない
```

#### 期待効果

| 方式 | 削減回数 | 困難度 |
|------|---------|--------|
| 方式A（推奨） | 1.98回/コールバック（平均） | 中 |
| 方式B（時刻統合） | 1回/コールバック | 高 |

---

### P1-4: RTLocalState 構造体 `alignas(64)` 欠如 🛡️

**発見経緯**: 監査で false sharing リスクを検出。文書では全く触れられていなかった。

#### 対象

```cpp
// src/audioengine/AudioEngine.h:1472-1516
struct RTLocalState {  // ← alignas(64) なし
    std::atomic<uint64_t> audioCallbackEpochCounter;  // 音声スレッド書込
    // ... (~25 fields) ...
    std::atomic<uint64_t> publishTimingWriteCount;    // ← Messageスレッド書込 ⚠️
    // ...
};
```

#### 問題

`publishTimingWriteCount`（Messageスレッド書込）と同一キャッシュライン上の `lastCallbackEndTicks`（音声スレッド書込）がコンフリクト。キャッシュラインのバウンスにより µs 級のペナルティ。

#### 修正

```cpp
struct alignas(128) RTLocalState {
    // 既存メンバ
};
```

#### 期待効果

| 指標 | 現在 | 改善後 |
|------|------|--------|
| false sharing risk | 高（alignasなし） | 無 |
| 推定ペナルティ | µs級（不定期） | 0 |

---

### P1-5: AudioEngine クラス atomic false sharing 🛡️

#### 対象

`src/audioengine/AudioEngine.h:2055-2130` 内の以下:

| 変数 | 書込スレッド | 現状のalignas |
|------|------------|:------------:|
| `mmcssShutdownRequested` | Message | ❌ なし |
| `mmcssApplied_` | **音声スレッド** | ❌ なし |
| `useMmcssPriority` | Message | ❌ なし |
| `manualOversamplingFactor` | Message | ❌ なし |
| `softClipEnabled` | Message | ❌ なし |

#### 修正

```cpp
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};
alignas(64) std::atomic<bool> mmcssApplied_{false};
alignas(64) std::atomic<bool> useMmcssPriority{true};
```

---

### P1-6: IppFFTPlanCache::getOrCreate() RT呼出防止 CI アサーション 🛡️

#### 対象

```cpp
// src/MKLNonUniformConvolver.cpp:87-99
static const IppFFTPlan* getOrCreate(int order)
{
    std::lock_guard<std::mutex> lock(getMutex());  // ← 非RTスレッド保証
    // ...
    auto plan = createPlan(order);  // ← 動的確保含む
    // ...
}
```

#### 現状

prepare()パス（Message Thread）からのみ呼ばれ安全。「Audio Thread からの FFT 再初期化禁止」のコメントあり。

#### 修正

```cpp
static const IppFFTPlan* getOrCreate(int order)
{
    ASSERT_NON_RT_THREAD();  // ★ CIでRT呼出を検出
    // ...
}
```

#### 期待効果

RTパスからの誤った呼出をコンパイル時/テスト時に検出。リグレッション防止。

---

### P1-7: MKLNonUniformConvolver.cpp の Logger → diagLog 統一 🛡️

#### 対象

```cpp
// src/MKLNonUniformConvolver.cpp:697
juce::Logger::writeToLog("MKLNonUniformConvolver: FFT plan cache creation failed...");
// 同様に line 704-708 にも複数
```

#### 修正

```cpp
diagLog("MKLNonUniformConvolver: FFT plan cache creation failed...");
```

#### 期待効果

- Logger 直接呼出を `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードの保護下に移行
- 現在は非RTスレッドで問題ないが、将来のリグレッション防止

---

### P1-8: Debug `isAudioThread()` の hash 重複計算抑制 ⚡

#### 対象

```cpp
// src/DspNumericPolicy.h:146-155
inline bool isAudioThread() noexcept
{
    const uint64_t tag = currentThreadTag();  // hash<thread::id> 毎回計算
    for (auto& slot : audioThreadSlots()) {
        if (convo::consumeAtomic(slot.tag, ...) == tag)
            return true;
    }
    return false;
}
```

#### 修正

```cpp
inline bool isAudioThread() noexcept
{
    const uint64_t tag = convo::cachedThreadHash();  // thread_local キャッシュ
    // ...
}
```

---

## 4. P2 — 計画的対応

### P2-1: Diagnostics 収集のサンプリング前倒し 📋

- **現状**: 診断時毎コールバック8回の `getCurrentTimeUs()` が**収集**される。サンプリングマスクは出力の間引きにのみ使用
- **改善**: 計測自体をサンプリングマスクでガード
- **対象**: `AudioBlock.cpp`, `BlockDouble.cpp` の診断ブロック
- **期待効果**: 診断時の getCurrentTimeUs 8回 → 1-2回
- **ファイル**: 2（AudioBlock.cpp, BlockDouble.cpp）

### P2-2: kTraceBufferSize 超過後の branch 回収 📋

- **現状**: 4096回書き込み後も `fetchAddAtomic(traceWriteIndex_)` + `if (idx < kTraceBufferSize)` を毎回実行
- **改善**: `traceFull_` フラグ導入で安定状態の分岐を排除
- **対象**: `ISRLifecycle.cpp`, `ISRLifecycle.h`
- **期待効果**: 安定状態で `fetchAddAtomic(acq_rel)` + 分岐 → `relaxed load` + 分岐
- **ファイル**: 2

### P2-3: RTCapabilityFirewall リリース時省略 📋

- **現状**: リリースビルドでも `FirewallToken` 取得＋`publishAtomic` を毎コールバック
- **改善**: `#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)` で空実装に
- **対象**: `ISRRTExecution.h`, `ISRRTExecution.cpp`
- **期待効果**: リリースビルドで publishAtomic 2回削減
- **注意**: `auditPublishAttempt()` の動作確認（非RTスレッドからのみ呼ばれる→問題なし）

### P2-4: ScopedNoDenormals 重複構築削減 📋

- **現状**: DSPCore + EQProcessor で2回 `ScopedNoDenormals` 構築（MXCSR書込×2）
- **改善**: thread_local で FTZ/DAZ 状態を追跡し、不要な MXCSR 書込を回避
- **対象**: 該当 process ファイル
- **期待効果**: MXCSR 書込のシリアライゼーションペナルティを低減

### P2-5: AudioCallbackRuntimeScope atomic 連鎖評価 📋

- **現状**: スコープ構築時に4連続の atomic RMW 操作
- **対応**: 影響は限定的。調査のみ実施し、実際のペナルティが確認できた場合のみ対応
- **対象**: 調査のみ

### P2-6: RTAllocatorFirewall::markRTContext() グローバルatomic低減 📋

- **現状**: `s_sharedRtContextFlag`（グローバル）への publishAtomic を毎コールバック
- **改善**: ファイアウォール自体をスレッドローカルに移行するか、リリースビルドで省略
- **対象**: `ISRRTExecution.cpp/h`
- **期待効果**: グローバル atomic 書込削減 → キャッシュ汚染低減

### P2-7: 診断パス GetCurrentProcessorNumber() syscall 削減 📋

- **現状**: 診断ビルド時、毎コールバック `GetCurrentProcessorNumber()` を複数回呼ぶ
- **改善**: 1回に集約して thread_local にキャッシュ
- **対象**: `AudioBlock.cpp`, `BlockDouble.cpp`
- **期待効果**: 診断時の syscall 回数削減（数百ns〜数µs）

---

## 5. P3 — 調査・検討

### P3-1: musicalSoftClipScalar() の libm 調査 🔍

```cpp
// DSPCoreIO.cpp
double AudioEngine::DSPCore::musicalSoftClip(double x, ...) noexcept
{
    return musicalSoftClipScalar(x, threshold, knee, asymmetry);
}
```

`musicalSoftClipScalar` の実装未確認。`std::pow`/`std::exp` 等の libm 呼出を含む場合、RTブロッカーになりうる。要調査。

### P3-2: MMCSS 初回コールバック defer 可能性調査 🔍

- **状況**: 初回コールバックにて `applyMmcssPriority()` が ~10-50µs のカーネル呼出を実行
- **課題**: JUCE が `prepareToPlay()` と同じスレッドをコールバックに使用する場合、defer 可能だがホスト依存
- **対応**: 調査のみ。実装はホスト互換性確認後。

---

## 6. ✅ confirmed-rt-safe — 検証済み安全項目

分析／検証／監査の3段階で確認した全RT-safe項目の統合リスト。

### 6-1. 全DSPカーネル

| モジュール | ファイル | 確認内容 |
|-----------|---------|---------|
| `LockFreeRingBuffer` push/pop | `core/LockFreeRingBuffer.h` | SPSC lock-free 確認 |
| `LockFreeAudioRingBuffer` | `LockFreeAudioRingBuffer.h` | 同上 |
| `CrossfadeRuntime` | `audioengine/` | atomicのみ |
| `RampBase`/各Smoother | `core/` | 数値演算のみ |
| `MKLNonUniformConvolver::Add/Get` | `src/MKLNonUniformConvolver.*` | lock-free, alloc-free |
| `EQProcessor::process` | `eqprocessor/EQProcessor.Processing.cpp` | RCU読取＋pre-alloc |
| `ConvolverProcessor::process` | `convolver/ConvolverProcessor.Runtime.cpp` | RCU読取＋pre-alloc |
| `OutputFilter::process` | `OutputFilter.*` | 係数準備済み |
| `LoudnessMeter::processBlock` | `LoudnessMeter.*` | AVX2内積のみ |
| `TruePeakDetector::processBlock` | `TruePeakDetector.*` | 事前割当のみ |
| `PsychoacousticDither::process` | `PsychoacousticDither.*` | 事前計算テーブル |
| `Fixed/LatticeNoiseShaper` | `*NoiseShaper*.h` | 事前割当のみ |
| `UltraHighRateDCBlocker` | `UltraHighRateDCBlocker.h` | IIRフィルタ |
| `CustomInputOversampler::processUp/Down` | `CustomInputOversampler.*` | 事前割当＋SIMD |
| `InputBitDepthTransform` | `InputBitDepthTransform.h` | AVX2/SIMD |

### 6-2. 全ミューテックス確認済み

| ファイル | Mutex名 | 実行スレッド |
|---------|---------|------------|
| `ISRLifecycle.cpp` | `nonRtGuard_` | Message/Worker |
| `PrepareToPlay.cpp:79` | `rebuildMutex` | Message（`ASSERT_NON_RT_THREAD()`） |
| `ReleaseResources.cpp:115` | `rebuildMutex` | Message（同） |
| `MKLNonUniformConvolver.cpp:89` | `IppFFTPlanCache::getMutex()` | prepare() |
| `AudioEngine.Commit.cpp` | — | Commit（`ASSERT_NON_RT_THREAD()`） |
| `AudioEngine.Parameters.cpp` | — | Message（同） |
| `AudioEngine.Threading.cpp` | — | Message（同） |
| `ISRRetireRuntimeEx.cpp:185` | — | Retire（同） |

### 6-3. 全 `diagLog()` 呼出元確認済み（ガード欠如のP0-1以外）

| ファイル | スレッド | 保護 |
|---------|---------|------|
| `AudioEngine.Commit.cpp` | Commit | `ASSERT_NON_RT_THREAD()` |
| `AudioEngine.CtorDtor.cpp` | Message | ctor/dtor |
| `AudioEngine.Init.cpp` | Message | 初期化 |
| `AudioEngine.Parameters.cpp` | Message | `ASSERT_NON_RT_THREAD()` |
| `DSPCoreLifecycle.cpp` | Message | prepareToPlay |
| `PrepareToPlay.cpp` | Message | prepareToPlay |
| `ReleaseResources.cpp` | Message | releaseResources |

---

## 7. ファイル変更サマリ

### 凡例

| マーク | 意味 |
|:------:|------|
| 🐛 | バグ修正 |
| ✨ | 新規作成 |
| ⚡ | パフォーマンス最適化 |
| 🛡️ | 防御的プログラミング |
| 📋 | 構造的改善 |

### 変更一覧（P0→P1→P2→P3）

| # | ファイル | 変更種別 | 変更内容 |
|:-:|---------|:---------:|---------|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 | `diagLog()` を `#if` でガード |
| P1-1 | `src/core/ThreadHash.h` | ✨⚡ | `convo::cachedThreadHash()` 新規作成 |
| P1-1 | `src/DspNumericPolicy.h:43` | ⚡ | `currentThreadTag()` → `cachedThreadHash()` |
| P1-1 | `src/core/RCUReader.h:151` | ⚡ | `currentThreadToken()` → `cachedThreadHash()` |
| P1-1 | `src/core/EpochDomain.h:69` | ⚡ | hash呼出 → `cachedThreadHash()` |
| P1-1 | `src/DspNumericPolicy.h:158` | ⚡ | `isAudioThread()` でキャッシュ活用 |
| P1-2 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:100,106` | ⚡ | `CallbackTelemetryScope` 条件化 |
| P1-2 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:93,99` | ⚡ | 同上（double版） |
| P1-3 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:500-557` | ⚡ | XRUNサンプリング化 |
| P1-3 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:465-521` | ⚡ | 同上（double版） |
| P1-4 | `src/audioengine/AudioEngine.h:1472` | 🛡️ | `RTLocalState` に `alignas(128)` |
| P1-5 | `src/audioengine/AudioEngine.h:2113,2120` | 🛡️ | 該当atomicに `alignas(64)` |
| P1-6 | `src/MKLNonUniformConvolver.cpp:87` | 🛡️ | `ASSERT_NON_RT_THREAD()` 追加 |
| P1-7 | `src/MKLNonUniformConvolver.cpp:697,704-708` | 🛡️ | `Logger::writeToLog` → `diagLog()` |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 | 診断収集のサンプリング前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 | 同上（double版） |
| P2-2 | `src/audioengine/ISRLifecycle.cpp` | 📋 | `traceFull_` フラグ導入 |
| P2-2 | `src/audioengine/ISRLifecycle.h` | 📋 | `traceFull_` メンバ追加 |
| P2-3 | `src/audioengine/ISRRTExecution.h` | 📋 | リリースビルド Firewall 省略 |
| P2-3 | `src/audioengine/ISRRTExecution.cpp` | 📋 | 同上 |
| P2-4 | `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp` | 📋 | ScopedNoDenormals 最適化 |
| P2-4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 | 同上 |
| P2-4 | `src/eqprocessor/EQProcessor.Processing.cpp` | 📋 | 同上 |
| P2-5 | — | 📋 | 調査のみ |
| P2-6 | `src/audioengine/ISRRTExecution.cpp/h` | 📋 | markRTContext 低減 |
| P2-7 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 | GetCurrentProcessorNumber 集約 |
| P2-7 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 | 同上 |
| P3-1 | — | 🔍 | musicalSoftClipScalar 調査 |
| P3-2 | — | 🔍 | MMCSS defer 調査 |

**ファイル数**: 新規1 + 修正13 = **全14ファイル**
**変更行数**: 概算 〜120行追加/修正

---

## 8. スケジュールと依存関係

### フェーズ構成（4フェーズ）

```
フェーズ0: P0 クリティカル        0.5h
フェーズ1: P1 優先+最適化         3-5日
フェーズ2: P2 構造的改善          3-5日
フェーズ3: P3 調査+検証           2-3日
```

### ガントチャート風スケジュール

```
Task                    | D1  | D2  | D3  | D4  | D5  | D6  | D7  | D8  | D9  | D10 |
------------------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
P0-1 diagLog guard     | ███ |     |     |     |     |     |     |     |     |     |
P1-1 thread_hash       |     | ███ | ███ |     |     |     |     |     |     |     |
P1-2 getTime条件化      |     | ██  |     |     |     |     |     |     |     |     |
P1-3 XRUN sampling     |     | ██  | ██  |     |     |     |     |     |     |     |
P1-4/5 false_sharing   |     |     | ██  | ██  |     |     |     |     |     |     |
P1-6/7 MKL防御         |     |     |     | █   |     |     |     |     |     |     |
P1-8 assert最適化      |     |     |     | █   |     |     |     |     |     |     |
P2-1 diag sampling     |     |     |     |     | ██  |     |     |     |     |     |
P2-2 trace回収         |     |     |     |     | █   |     |     |     |     |     |
P2-3 firewall 省略     |     |     |     |     | █   | █   |     |     |     |     |
P2-4 denormals最適化   |     |     |     |     |     | ██  |     |     |     |     |
P2-6 markRT低減       |     |     |     |     |     | █   |     |     |     |     |
P2-7 CPU番号集約      |     |     |     |     |     | █   |     |     |     |     |
P3-1/2 調査           |     |     |     |     |     |     | ██  | ██  |     |     |
全ビルド検証          |     |     |     |     |     |     |     |     | ██  |     |
長期安定性テスト       |     |     |     |     |     |     |     |     | ██  | ██  |
```

### 依存関係

```
P0-1: 独立（即時実行可能）
P1-1: 独立（新規ユーティリティ追加、段階的置換可能）
P1-2 ← P1-1（thread_local キャッシュが既にあると便利だが必須ではない）
P1-3: P1-2と独立
P1-4: P1-5と同時実行推奨（同一ファイルの構造変更）
P1-5: P1-4と同時実行推奨
P1-6: 独立
P1-7: P0-1完了後（diagLog() が全ビルドで安全になってから）
P1-8 ← P1-1（cachedThreadHash 導入後）
P2-X: P1全完了後、または並行実行可能
P3-X: 任意のタイミング。実装着手前に実施
```

---

## 9. リスク評価とロールバック計画

### リスクマトリクス

| ID | 変更 | リスク | 確率 | 影響度 | 緩和策 |
|:--:|------|--------|:----:|:------:|--------|
| P0-1 | #if追加 | 低 | 1% | 診断ログ非出力 | 両ビルド確認 |
| P1-1 | thread_local 追加 | 低 | 3% | 初回ハッシュ計算の微小増加 | 段階的置換、ビルド確認 |
| P1-2 | 条件分岐追加 | 低-中 | 5% | CLIテレメトリ非動作 | 条件反転テスト |
| P1-4 | alignas追加 | 極低 | 1% | 構造体サイズ増加 | sizeof確認、アライメントテスト |
| P1-6 | ASSERT追加 | 極低 | 1% | Debugビルドでabort | Releaseビルドでは無効 |
| P1-7 | Logger->diagLog | 低 | 2% | 診断時ログ非出力 | #if定義の確認 |
| P2-3 | #if追加 | 低 | 3% | Debug時のFirewall無効化 | CIビルド維持 |
| P2-2 | フラグ追加 | 低 | 2% | 誤った早期停止 | テストで検証 |

### ロールバック手順

各改修は**単一コミット**で行い、コミットメッセージに改修IDタグを含める:

```
git commit -m "[work66][P0-1] Add missing #if guard to diagLog() in DSPCoreFloat.cpp"
git commit -m "[work66][P1-1] Add convo::cachedThreadHash() and migrate callers"
git commit -m "[work66][P1-2] Conditionalize CallbackTelemetryScope timing"
...
```

問題発生時:

```bash
git log --oneline --grep="\[work66\]"  # 該当コミットを特定
git revert <commit-hash>                # 安全にrevert
```

---

## 10. 付録: 現状のコールバックフロー図（最適化対象の可視化）

```
getNextAudioBlock() / processBlockDouble()          [毎コールバック]
│
├─ consumeAtomic(lifecycleState)                     ← acquire  1回
├─ fetchAddAtomic(audioCallbackEpochCounter)          ← acq_rel  1回
├─ compareExchangeAtomic(mmcssApplied_)              ← acq_rel  1回(初回のみ)
│   └─ applyMmcssPriority() → カーネル呼出群        ← 初回10-50µs 🔥
│
├─ AudioCallbackRuntimeScope ctor                    [4連続atomic]
│   ├─ lifecycleRuntime_.enterAudioCallback()        ← 3回atomic
│   │   └─ transitionTo(AudioRunning)
│   │       ├─ consumeAtomic + publishAtomic          ← 2回
│   │       ├─ fetchAddAtomic(traceWriteIndex_)       ← acq_rel 1回
│   │       └─ high_resolution_clock::now()          ← 初回4096回まで
│   ├─ rtCapabilityFirewall_.enter()                 ← publishAtomic 1回
│   │   └─ get_id()                                  ← 軽量
│   └─ fetchAddAtomic(audioCallbackActiveCount)      ← acq_rel 1回
│
├─ ScopedThreadRole ctor
│   └─ acquireAudioThreadSlot()                      ★ thread_local化(P1-1)
│       ├─ currentThreadTag() → hash<thread::id>()   ★ 毎回計算→キャッシュ
│       └─ consumeAtomic/compareExchangeAtomic ×N
│
├─ CallbackTelemetryScope ctor
│   └─ getCurrentTimeUs()                            ★ 条件化(P1-2) 2回→0回
│
├─ [CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS]
│   ├─ getCurrentTimeUs() ×8                          ★ サンプリング(P2-1)
│   ├─ GetCurrentProcessorNumber() ×N                ★ 集約(P2-7)
│   └─ consumeAtomic ×N
│
├─ XRUN detection (常時)
│   ├─ getCurrentTimeUs() ×2                          ★ サンプリング(P1-3)
│   └─ consumeAtomic/fetchAddAtomic ×N
│
├─ captureAudioThreadParameterSnapshot()              ★ false sharing(P1-4/5)
│   └─ consumeAtomic(acquire) ×9-16
│
├─ dsp->process() [DSP カーネル — 最適化対象外]
│   ├─ EQProcessor::process()
│   │   ├─ RCUReaderGuard → hash<thread::id>         ★ キャッシュ(P1-1)
│   │   └─ ScopedNoDenormals                         ★ 重複(P2-4)
│   ├─ ConvolverProcessor::process()
│   │   └─ RCUReaderGuard → hash<thread::id>         ★ キャッシュ(P1-1)
│   └─ OutputFilter / Loudness / NS / Dither
│
├─ [CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS]
│   └─ getCurrentTimeUs() ×2-4                       ★ サンプリング(P2-1)
│
├─ CallbackTimingHistory (診断時)
│   ├─ getCurrentTimeUs()                             ★ サンプリング(P2-1)
│   └─ fetchAddAtomic(callbackTimingWriteCount)
│
├─ ScopedThreadRole dtor
│   └─ releaseAudioThreadSlot()
│       ├─ currentThreadTag() → hash<thread::id>()   ★ キャッシュ(P1-1)
│       └─ fetchSubAtomic + publishAtomic ×2-3
│
├─ CallbackTelemetryScope dtor
│   └─ getCurrentTimeUs()                            ★ 条件化(P1-2) 2回→0回
│
└─ AudioCallbackRuntimeScope dtor                    [4連続atomic]
    ├─ fetchSubAtomic(audioCallbackActiveCount)
    ├─ rtCapabilityFirewall_.leave()                 ★ リリース時省略(P2-3)
    │   └─ publishAtomic(sharedRtContextFlag)
    └─ lifecycleRuntime_.leaveAudioCallback()
        └─ transitionTo(Prepared)
```

### 凡例

| マーク | 意味 |
|:------:|------|
| ★ xxx | 本計画の最適化対象（IDは対応セクション番号）|
| 🔥 | P0バグ修正対象 |
| 太字 | 今回の改修で削減/改善可能な箇所 |
| ← acq_rel | memory_order指定 |
| [診断時] | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` でのみ実行 |

### 推定改善効果

| 指標 | 現状（推定） | 目標 |
|------|------------|------|
| 非DSPオーバーヘッド（通常時） | ~4µs | <2µs |
| 非DSPオーバーヘッド（診断時） | ~12µs | <5µs |
| false sharing 起因の不定期ペナルティ | µs級 | 0 |
| diagLog 誤呼出による音飛びリスク | あり | なし |
| 該当ファイル変更数 | — | 14ファイル |
| 総変更行数（概算） | — | ~120行 |

---

*本計画書は doc/work66/MASTER_RT_Improvement_Plan_v2.md として管理する。*
*改修の進捗に応じて随時更新すること。*
