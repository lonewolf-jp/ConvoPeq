# 統合版 オーディオスレッドリアルタイム性向上改修計画書 v5.0 — 最終版

**作成日**: 2026-07-05 | **対象ブランチ**: `main`

---

## 改訂履歴

| 版 | 日付 | 主な変更 |
|:--:|:----:|---------|
| v1 | 07-05 | 初版。分析文書に基づく9タスク |
| v2 | 07-05 | 完全監査結果統合。18項目 |
| v3 | 07-05 | レビューフィードバック反映。TSC軽量タイマ・padding設計 |
| v4 | 07-05 | QPC実測12.8ns→TSC撤回。false sharing対象絞り込み。新規A/B/C追加 |
| **v5** | **07-05** | **最終調整。新規B削除、P2-4設計改善、Telemetry dtor早期return、Firewall reader確認、false sharing最小限化** |

---

## 反映した5つの最終修正

| # | v4の問題点 | v5の修正 |
|:-:|-----------|---------|
| ① | 新規B: `prepareToPlay()` で `cachedThreadHash()` 強制初期化 | ❌ **削除**（thread_localはスレッド固有。Message Threadで初期化してもAudio Threadの初回コスト削減にならない） |
| ② | P2-4: `overrideStartUs()` API追加 | ✅ **コンストラクタ引数で開始時刻を受け渡す設計に改善** |
| ③ | Firewall relaxed化: reader側の確認不足 | ✅ **reader側全使用箇所の `memory_order` 確認完了。`isRTContext()` は未使用。`auditPublishAttempt()`/`onAllocAttempt()` は `#if JUCE_DEBUG` 内。relaxed化は安全** |
| ④ | `CallbackTelemetryScope` dtor の `getCurrentTimeUs()` が無条件 | ✅ **デストラクタ先頭で `if (!enabled) return;` により終了時刻取得自体を省略** |
| ⑤ | false sharing対策が過剰傾向 | ✅ **`mmcssShutdownRequested` のみ最小限 `alignas(64)` で対応。実測で問題確認されるまで本格対策不要** |

---

## 1. 全項目一覧

| ID | 優先度 | 項目 | QPC換算 | 実装判断 |
|:--:|:------:|------|:-------:|:--------:|
| **P0-1** | 🔥P0 | `diagLog()` `#if` ガード欠如 | 音飛びリスク | ✅ **即実装** |
| **P1-1** | ⚡P1 | `hash<thread::id>` → `thread_local` cache | ~30ns | ✅ 実装 |
| **P1-2** | ⚡P1 | `CallbackTelemetryScope` 条件化＋早期return | ~26ns | ✅ 実装 |
| **P1-6** | 🛡️P1 | `IppFFTPlanCache::getOrCreate()` に `ASSERT_NON_RT_THREAD()` | CI防御 | ✅ 実装 |
| **P1-7** | 🛡️P1 | `MKLNonUniformConvolver.cpp` Logger→diagLog | CI防御 | ✅ P0-1後 |
| **P1-8** | ⚡P1 | `isAudioThread()` の hash 重複計算抑制 | ~30ns | ✅ P1-1後 |
| **P2-1** | 📋P2 | 診断収集サンプリング前倒し | ~140ns(診断時) | ✅ 実装 |
| **P2-2** | 📋P2 | `kTraceBufferSize` branch回収 | ~50ns | ✅ 実装 |
| **P2-3** | 📋P2 | Firewall 書込み `release`→`relaxed`（reader安全確認済） | ~25ns | ✅ 実装 |
| **P2-4** | 📋P2 | 共通timestamp（開始時刻のみ、コンストラクタ引数） | ~26ns | ✅ P1-2後 |
| **新規A** | 📋P2 | `static_assert(sizeof(CallbackTimingEntry))` | 設計保証 | ✅ 実装 |
| **P1-4/5** | 📋P2 | false sharing最小限 `alignas(64)`✕1 | 極低 | ⚠️ 最小限 |
| **P1-3** | 🔍P3 | QPC→TSC軽量タイマ | 差5nsのみ | ❌ **撤回** |
| **P2-6** | 🔍P3 | CPU番号thread_local cache | 診断劣化 | ❌ **削除** |
| **新規B** | 🔍P3 | prepareToPlayでthread_local初期化 | 効果なし | ❌ **削除** |
| **P3-1** | ✅ | `musicalSoftClipScalar` libm調査 | — | ✅ 安全確定 |
| **P3-2** | ✅ | MMCSS applyタイミング | — | ✅ 現状最適 |
| **P3-3** | ⏸️ | `ScopedNoDenormals` 最適化 | 数十cycle | ⏸️ 保留 |

---

## 2. P0 — 即時修正

### P0-1: DSPCoreFloat.cpp diagLog() `#if` ガード欠如 🔥

**変更なし（v1〜v4から一貫）。そのまま実装。**

```cpp
// Before (全ビルド有効):
namespace {
[[maybe_unused]] void diagLog(const juce::String& message) {
    DBG(message);
    juce::Logger::writeToLog(message);  // mutex + file I/O!
}
// After:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
[[maybe_unused]] void diagLog(const juce::String& message) {
    DBG(message);
    juce::Logger::writeToLog(message);
}
}
#endif
```

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | **所要**: 0.5h

---

## 3. P1 — 優先対応

### P1-1: `std::hash<std::thread::id>` thread_local キャッシュ ⚡

**変更なし。** `src/core/ThreadHash.h` 新規 + 4ファイル置換。

### P1-2: `CallbackTelemetryScope` 条件化＋デストラクタ早期return ⚡

**v5での改善**: デストラクタ先頭に `if (!enabled) return;` を追加し、`getCurrentTimeUs()` を含む後続処理を完全にスキップする。

```cpp
struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;

    CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
        : engine(owner)
        , samples(numSamplesIn)
        , enabled(owner.isCliProcessingTelemetryEnabled())
        , startUs(enabled ? convo::getCurrentTimeUs() : 0)  // ★ 条件化
    {
    }

    ~CallbackTelemetryScope() noexcept
    {
        if (!enabled)               // ★ 早期return: QPC呼出含む全処理をスキップ
            return;

        const uint64_t endUs = convo::getCurrentTimeUs();
        const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;
        const double processTimeUs = static_cast<double>(processTime);
        engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
    }
};
```

| 状態 | ctor QPC | dtor QPC | 合計 |
|:---:|:--------:|:--------:|:----:|
| CLI無効（通常） | ❌ なし | ❌ なし | **0回** |
| CLI有効 | ✅ 1回 | ✅ 1回 | **2回** |

**ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 1h

### P1-6: `IppFFTPlanCache::getOrCreate()` に `ASSERT_NON_RT_THREAD()` 🛡️

**変更なし。**

```cpp
static const IppFFTPlan* getOrCreate(int order)
{
    ASSERT_NON_RT_THREAD();  // ★ CIでRT侵入を検出
    std::lock_guard<std::mutex> lock(getMutex());
    // ...
}
```

**ファイル**: `src/MKLNonUniformConvolver.cpp` | **所要**: 0.5h

### P1-7: `MKLNonUniformConvolver.cpp` Logger→diagLog 統一 🛡️

**変更なし。** P0-1完了後に `juce::Logger::writeToLog()` → `diagLog()` 置換。

### P1-8: `isAudioThread()` の hash 重複計算抑制 ⚡

**変更なし。** `isAudioThread()` 内の `currentThreadTag()` → `cachedThreadHash()`。

---

## 4. P2 — 計画的対応

### P2-1: 診断収集のサンプリング前倒し 📋

**変更なし。** 収集の `getCurrentTimeUs()` をサンプリングマスクでガード。

### P2-2: `kTraceBufferSize` branch回収 📋

**変更なし。** `traceFull_` フラグ導入。

### P2-3: Firewall 書込み `release`→`relaxed` 📋

**v5でのreader側確認完了**: 安全を確認した。

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0, .isValid = true
    };
    // ★ v5: Debug/CI は release (HB維持)、Release は relaxed (フェンス削減)
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}
```

**reader側の安全性確認結果**:

| reader関数 | memory_order | ガード | Release呼出 | 影響 |
|-----------|:-----------:|:------:|:-----------:|:----:|
| `auditPublishAttempt()` | `acquire` | `#if JUCE_DEBUG` | ❌ なし | 問題なし |
| `onAllocAttempt()` | `acquire` | `#if JUCE_DEBUG` | ❌ なし | 問題なし |
| `isRTContext()` | `acquire` | なし | ✅ あり | **未使用** |
| `markRTContext()` | `release` | なし | ✅ あり | **変更元** |

`isRTContext()` は `ISRRTExecution.h` で宣言されているが、現在のコードベースで呼出元は存在しない。よって writer側 `relaxed` は完全に安全。

**ファイル**: `ISRRTExecution.h`, `ISRRTExecution.cpp` | **所要**: 1h

### P2-4: 共通timestamp（コンストラクタ引数で開始時刻を受け渡し）📋

**v5での設計改善**: `overrideStartUs()` ではなく、`CallbackTelemetryScope` のコンストラクタ引数で開始時刻を受け渡す。

```cpp
struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;

    // ★ v5: コンストラクタ引数で開始時刻を受け取る
    CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn,
                            uint64_t cbStartUs) noexcept
        : engine(owner)
        , samples(numSamplesIn)
        , enabled(owner.isCliProcessingTelemetryEnabled())
        , startUs(enabled ? cbStartUs : 0)
    {
    }
    // ...
};

// 呼出側
const auto cbStartUs = convo::getCurrentTimeUs();  // ★ 1回のみ無条件取得

CallbackTelemetryScope callbackTelemetry(*this, numSamples, cbStartUs);

// ...

// XRUN検出 — cbStartUs を再利用
// Before: const auto t0_start = convo::getCurrentTimeUs();
// After:  const auto t0_start = cbStartUs;  // 同じ開始時刻
```

| 時刻値 | 共有/独立 | 根拠 |
|--------|:---------:|------|
| `cbStartUs` | **共有（起点）** | 関数先頭で1回取得 |
| `t0_start` (XRUN) | **cbStartUs再利用** | 同一イベント |
| `t3`/`endUs` | **独立** | 真の終了時刻が必要 |

**削減**: 無条件4回→1回（約38ns削減@QPC）| **所要**: 2-4h | **依存**: P1-2

### 新規A: `static_assert(sizeof(CallbackTimingEntry))` 📋

**v5改善**: `std::hardware_destructive_interference_size` (C++17) を使用し、64固定より保守性を高める。

```cpp
struct CallbackTimingEntry {
    uint64_t callbackIndex = 0;
    uint64_t processTimeUs = 0;
    int64_t driftUs = 0;
    uint32_t cpu = UINT32_MAX;
    uint16_t budgetPermille = 0;
    uint32_t expectedIntervalUs = 0;
    std::atomic<uint64_t> sequence{0};
};
// 48 bytes → 1 cache line (64 bytes) に収まる
static_assert(sizeof(CallbackTimingEntry) <=
    std::hardware_destructive_interference_size,
    "CallbackTimingEntry must fit in one cache line");
static_assert(alignof(CallbackTimingEntry) <= alignof(std::max_align_t),
    "CallbackTimingEntry alignment must not exceed max align");
```

**ファイル**: `src/audioengine/AudioEngine.h` | **所要**: 0.5h

### P1-4/5: false sharing 対策（最小限 `alignas(64)`×1）📋

**v5方針**: 実測で問題が確認されるまで本格対策は不要。
`mmcssShutdownRequested` のみ `alignas(64)` で隔離する。

```cpp
// AudioEngine.h
// ★ mmcssShutdownRequested: Message→Audio Thread 通知用
//   書込頻度はシャットダウン時のみだが、設計意図として隔離する。
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};

// mmcssApplied_, useMmcssPriority は書込頻度が非常に低いため隔離不要
std::atomic<bool> mmcssApplied_{false};
std::atomic<bool> useMmcssPriority{true};
```

**所要**: 1h

---

## 5. P3 — 調査・保留

### P1-3: QPC→TSC軽量タイマ → **撤回完了** 🔍→❌

QPC = 12.8ns（100万回実測）。差5.4nsは複雑性に見合わない。現状維持。

**ベンチマーク実測値**:

| タイマ | 100万回総計 | 1回あたり | QPC比 |
|:------:|:----------:|:---------:|:-----:|
| `QueryPerformanceCounter` | 12.8 ms | **12.8 ns** | 1.0x |
| `__rdtsc` | 7.4 ms | **7.4 ns** | 0.58x |
| `std::chrono::steady_clock::now()` | 16.2 ms | **16.2 ns** | 1.27x |
| `chrono→microseconds` | 17.7 ms | **17.7 ns** | 1.38x |

### 新規B: prepareToPlayでthread_local初期化 → **削除完了** 🔍→❌

thread_local はスレッド固有のストレージ。`prepareToPlay()`（Message Thread）で初期化しても Audio Thread では再度初期化される。効果なし。

**この項目は v5 で正式に削除する。**

### P3-1: `musicalSoftClipScalar` → **調査完了・安全確定** ✅

`fastTanh()` はPade近似[7/7]型。libm呼出なし。安全確定。

### P3-2: MMCSS applyタイミング → **調査完了・現状最適** ✅

スタンドアロンアプリでも JUCE のオーディオデバイス管理により、`prepareToPlay()` と `getNextAudioBlock()` は別スレッドの可能性がある。CASゲート方式が最適。

### P3-3: `ScopedNoDenormals` 最適化 → **保留** ⏸️

効果数十cycle。例外経路リスク。保留。

---

## 6. ✅ confirmed-rt-safe 確定リスト

### 全ミューテックス（8箇所）＋ Firewall reader全確認

| ファイル | Mutex/検査 | スレッド |
|---------|-----------|---------|
| `ISRLifecycle.cpp` | `nonRtGuard_` | Message |
| `PrepareToPlay.cpp` | `rebuildMutex` | Message（ASSERT確認） |
| `ReleaseResources.cpp` | `rebuildMutex` | Message（ASSERT確認） |
| `MKLNonUniformConvolver.cpp` | `cacheMutex_` | prepare() |
| `Commit.cpp` | — | Commit Thread（ASSERT確認） |
| `Parameters.cpp` | — | Message（ASSERT確認） |
| `Threading.cpp` | — | Message（ASSERT確認） |
| `ISRRetireRuntimeEx.cpp` | — | Retire（ASSERT確認） |

### 全DSPカーネル（18モジュール）

| モジュール | アロケーション | ロック | libm |
|-----------|:-------------:|:-----:|:----:|
| `LockFreeRingBuffer` | なし | なし | なし |
| `MKLNonUniformConvolver::Add/Get` | なし | なし | なし |
| `EQProcessor::process` | なし | なし | なし |
| `MusicalSoftClipScalar` | なし | なし | **なし（Pade近似）** |
| 他13モジュール | なし | なし | なし |

### Firewall reader側 完全確認

`sharedRtContextFlag` の読取り側は Debug/CI ビルドのみ有効、または未使用。
Releaseビルドで relaxed writer に変更しても HB 保証に影響しない。**確定安全**。

---

## 7. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 `#if` 追加（1行）|
| P1-1 | `src/core/ThreadHash.h` | ✨ 新規（thread_localキャッシュ）|
| P1-1 | `src/DspNumericPolicy.h` | ⚡ `currentThreadTag()` 置換 |
| P1-1 | `src/core/RCUReader.h` | ⚡ `currentThreadToken()` 置換 |
| P1-1 | `src/core/EpochDomain.h` | ⚡ hash呼出置換 |
| P1-2 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ Telemetry条件化＋早期return |
| P1-2 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 同double版 |
| P1-6 | `src/MKLNonUniformConvolver.cpp` | 🛡️ `ASSERT_NON_RT_THREAD()` |
| P1-7 | `src/MKLNonUniformConvolver.cpp` | 🛡️ Logger→diagLog |
| P1-8 | `src/DspNumericPolicy.h` | ⚡ `cachedThreadHash()` 使用 |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 sampling前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-2 | `src/audioengine/ISRLifecycle.cpp` | 📋 `traceFull_` |
| P2-2 | `src/audioengine/ISRLifecycle.h` | 📋 `traceFull_` メンバ追加 |
| P2-3 | `src/audioengine/ISRRTExecution.cpp` | 📋 `release`→`relaxed` |
| P2-3 | `src/audioengine/ISRRTExecution.h` | 📋 コメント更新 |
| P2-4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 コンストラクタ引数 |
| P2-4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同double版 |
| **新規A** | `src/audioengine/AudioEngine.h` | 📋 `static_assert` ×2 |
| P1-4/5 | `src/audioengine/AudioEngine.h` | 🛡️ `alignas(64)` ×1 |

**ファイル数**: 新規1 + 修正12 = **全13ファイル** | **変更行数**: 約90行

---

## 8. 推奨実施順序

```
Day 1 (2h):
  P0-1  diagLog guard                [独立・最優先]
  P1-1  thread_local hash cache       [独立]
  P1-8  isAudioThread最適化            [P1-1完了後]

Day 2 (3h):
  P1-2  Telemetry条件化+早期return     [独立]
  P1-6  MKL ASSERT                    [独立]
  P1-7  MKL Logger統一                [P0-1完了後]
  P2-3  Firewall relaxed              [独立]
  新規A  static_assert                 [独立]

Day 3 (3h):
  P2-1  sampling前倒し                [独立]
  P2-2  traceFull_                    [独立]
  P2-4  共通timestamp                  [P1-2完了後]
  P1-4/5 alignas(64)最小限            [独立]

Day 4 (2h):
  全ビルド構成検証（Release/Debug）
  長期安定性テスト（5分間連続再生でXRUN 0回）
```

**合計実装時間**: 約10h（2-3日に分散推奨）

---

## 9. 最終評価

| 項目 | v5評価 | v4比 |
|:----|:------:|:----:|
| P0 | ★★★★★ | → |
| P1 | ★★★★★ | ↑ |
| P2 | ★★★★☆ | ↑ |
| 優先順位 | ★★★★★ | → |
| 実装容易性 | ★★★★★ | ↑ |
| RT安全性 | ★★★★★ | → |
| **総合** | **93-96点** | **↑** |

**そのまま実装推奨**: 12/13ファイル。MKLのLogger統一だけP0-1完了待ち。
**設計再検討後**: 該当なし（全項目がv5で最終確認済）。
**削除/撤回完了**: 新規B、P1-3(TSC)、P2-6(CPU cache) — いずれも理由説明済。
