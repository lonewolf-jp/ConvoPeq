# 統合版 RT改善改修計画書 v5.4 — 最終確定版

**作成日**: 2026-07-05 | **対象**: `main`

---

## 改訂履歴

| 版 | 日付 | 主な変更内容 |
|:---:|:----:|-------------|
| v1 | 07-05 | 初版（分析文書ベース9タスク） |
| v2 | 07-05 | 完全監査統合（18項目） |
| v3 | 07-05 | TSC軽量タイマ・padding設計 |
| v4 | 07-05 | QPC実測12.8ns→TSC撤回。false sharing絞り込み |
| v5 | 07-05 | 新規B削除・Telemetry早期return・ctor引数・Firewall relaxed |
| v5.1 | 07-05 | 微調整（P2-4説明修正, static_assert#if, ラッパ構造体, 注意コメント, QPC詳細化） |
| v5.2 | 07-05 | ラッパ撤回→単純alignas, P2-3 コンパイル時保護 |
| v5.3 | 07-05 | P2-3 `#error`＋到達不能コード解消、`#if` 分岐整理 |
| **v5.4** | **07-05** | **`#pragma message` 削除、`relaxed` 使用理由をコメントで明記、実装順序を同一ファイル編集で最適化** |

### v5.3 → v5.4 の変更点

| # | 指摘 | v5.3 | v5.4 |
|:-:|------|------|------|
| ① | `#pragma message` は不要（`#if` 分岐だけで十分） | `#pragma message(...)` あり | ❌ **削除** |
| ② | `relaxed` を使う理由（HB不要であること）を記載 | 理由の記述なし | ✅ **コメントで明記** |
| ③ | 実装順序: P2-4 は P1-2 と同一ファイル | Day4 に分離 | ✅ **Day2 P1-2 直後に移動** |

---

## 1. 全項目一覧

| ID | 優先度 | 項目 | 実装判断 |
|:--:|:------:|------|:--------:|
| **P0-1** | 🔥P0 | `diagLog()` `#if` ガード欠如 | ✅ **即実装** |
| **P1-1** | ⚡P1 | `hash<thread::id>` → `thread_local` cache | ✅ 実装 |
| **P1-2** | ⚡P1 | `CallbackTelemetryScope` 条件化＋早期return | ✅ 実装 |
| **P1-6** | 🛡️P1 | `getOrCreate()` に `ASSERT_NON_RT_THREAD()` | ✅ 実装 |
| **P1-7** | 🛡️P1 | MKL Logger→`diagLog` 統一 | ✅ P0-1後 |
| **P1-8** | ⚡P1 | `isAudioThread()` 最適化 | ✅ P1-1後 |
| **P2-1** | 📋P2 | 診断収集サンプリング前倒し | ✅ 実装 |
| **P2-2** | 📋P2 | `kTraceBufferSize` branch回収 | ✅ 実装 |
| **P2-3** | 📋P2 | Firewall relaxed化（`#if` 分岐＋理由コメント） | ✅ 実装 |
| **P2-4** | 📋P2 | 共通timestamp（開始時刻のみctor引数） | ✅ P1-2後同日 |
| **新規A** | 📋P2 | `static_assert(CallbackTimingEntry)` | ✅ 実装 |
| **P1-4/5** | 📋P2 | false sharing 最小限 `alignas(64)` ×1 | ✅ 実装 |
| **P1-3** | 🔍→❌ | QPC→TSC軽量タイマ | ❌ **撤回** |
| **P2-6** | 🔍→❌ | CPU番号 thread_local cache | ❌ **削除** |
| **新規B** | 🔍→❌ | prepareToPlayでthread_local初期化 | ❌ **削除** |
| **P3-1** | ✅ | `musicalSoftClipScalar` libm調査 | ✅ **安全確定** |
| **P3-2** | ✅ | MMCSS applyタイミング | ✅ **現状最適** |
| **P3-3** | ⏸️ | `ScopedNoDenormals` 最適化 | ⏸️ **保留** |

---

## 2. P0 — 即時修正（変更なし）

### P0-1: DSPCoreFloat.cpp diagLog() #if ガード欠如 🔥

```cpp
// Before:
namespace {
[[maybe_unused]] void diagLog(const juce::String& message) {
    DBG(message);
    juce::Logger::writeToLog(message);  // 全ビルド有効 → 音飛びリスク
}
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

**ファイル**: `DSPCoreFloat.cpp` | **所要**: 0.5h

---

## 3. P1 — 優先対応

### P1-1: `std::hash<std::thread::id>` thread_local キャッシュ ⚡

**`src/core/ThreadHash.h` 新規 + 4ファイル置換。**

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

| ファイル | 変更前 | 変更後 |
|---------|--------|--------|
| `DspNumericPolicy.h:43` | `currentThreadTag()` → hash | `→ convo::cachedThreadHash()` |
| `RCUReader.h:151` | `currentThreadToken()` → hash | `→ convo::cachedThreadHash()` |
| `EpochDomain.h:69` | 直接 hash 呼出 | `→ convo::cachedThreadHash()` |
| `DspNumericPolicy.h:158`(P1-8) | `isAudioThread()` → hash | `→ convo::cachedThreadHash()` |

**ファイル**: 新規1 + 修正4 | **所要**: 1.5h

---

### P1-2 + P2-4: CallbackTelemetryScope 条件化＋早期return＋共通timestamp ⚡

**P2-4 を P1-2 と同じ日に実装（同一ファイルへの編集を集約）。**

```cpp
struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;

    // ★ v5.4: コンストラクタ引数で開始時刻を受け取る（P1-2 + P2-4 統合）
    CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn,
                            uint64_t cbStartUs) noexcept
        : engine(owner), samples(numSamplesIn)
        , enabled(owner.isCliProcessingTelemetryEnabled())
        , startUs(enabled ? cbStartUs : 0)
    {
    }

    ~CallbackTelemetryScope() noexcept
    {
        if (!enabled) return;                      // ★ 早期return
        const uint64_t endUs = convo::getCurrentTimeUs();
        const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;
        engine.recordAudioCallbackProcessingStats(samples,
            static_cast<double>(processTime));
    }
};

// 呼出側（getNextAudioBlock / processBlockDouble）
const auto cbStartUs = convo::getCurrentTimeUs();  // ★ 関数先頭で1回のみ無条件取得
CallbackTelemetryScope callbackTelemetry(*this, numSamples, cbStartUs);
```

#### QPC呼出回数

| シナリオ | 変更前 | 変更後 | 削減 |
|:--------:|:-----:|:-----:|:----:|
| 通常（CLI無効） | **2回** (ctor+dtor無条件) | **1回** (cbStartUsのみ) | **-1回** |
| CLI有効時 | **4回** (上記+XRUN開始+終了) | **2回** (cbStartUs+endUs) | **-2回** |

**ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 3h

---

### P1-6: IppFFTPlanCache::getOrCreate() に `ASSERT_NON_RT_THREAD()` 🛡️

```cpp
static const IppFFTPlan* getOrCreate(int order)
{
    ASSERT_NON_RT_THREAD();  // ★ CIでRT侵入を検出
    std::lock_guard<std::mutex> lock(getMutex());
    // ...
}
```

**ファイル**: `MKLNonUniformConvolver.cpp` | **所要**: 0.5h

---

### P1-7: MKLNonUniformConvolver.cpp Logger→diagLog 統一 🛡️

```cpp
// Before
juce::Logger::writeToLog("MKLNonUniformConvolver: ...");
// After (P0-1完了後)
diagLog("MKLNonUniformConvolver: ...");
```

**ファイル**: `MKLNonUniformConvolver.cpp` (line 697, 704-708) | **所要**: 0.5h | **依存**: P0-1

---

### P1-8: `isAudioThread()` の hash 重複計算抑制 ⚡

```cpp
inline bool isAudioThread() noexcept
{
    const uint64_t tag = convo::cachedThreadHash();  // ★ thread_local キャッシュ
    for (auto& slot : audioThreadSlots()) {
        if (convo::consumeAtomic(slot.tag, ...) == tag)
            return true;
    }
    return false;
}
```

**ファイル**: `DspNumericPolicy.h` | **所要**: 0.5h | **依存**: P1-1

---

## 4. P2 — 計画的対応

### P2-1: 診断収集のサンプリング前倒し 📋

```cpp
// Before
const uint64_t t3 = convo::getCurrentTimeUs();  // ★ 毎回
if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0) { /* 書込 */ }

// After
if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0) {
    const uint64_t t3 = convo::getCurrentTimeUs();  // ★ サンプリング時のみ
    /* 書込 */
}
```

**ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 2h

---

### P2-2: kTraceBufferSize branch回収 📋

```cpp
// Before (毎回)
const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, ..., std::memory_order_acq_rel);
if (idx < kTraceBufferSize) { /* timestamp 取得 */ }

// After (traceFull_ フラグ)
if (traceFull_.load(std::memory_order_relaxed))
    ; // 満杯 → fetchAddAtomic 不要
else {
    const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, ..., std::memory_order_acq_rel);
    if (idx < kTraceBufferSize) { /* ... */ }
    else { traceFull_.store(true, std::memory_order_release); }
}
```

**ファイル**: `ISRLifecycle.cpp`, `ISRLifecycle.h` | **所要**: 1h

---

### P2-3: Firewall 書込み release→relaxed（理由コメント付 `#if` 分岐）📋

**v5.4 改善**: `#pragma message` を削除。`relaxed` が許される理由（HB不要）をコメントに明記。

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0, .isValid = true
    };

    // sharedRtContextFlag は単なる状態フラグであり、
    // 他データとの同期(HB)を必要としないため relaxed が許される。
    // 前提: isRTContext() 自体は実装済みだが、現時点のコードベースで呼び出し箇所は存在しない。
    // 将来 CONVO_USE_IS_RT_CONTEXT を定義して isRTContext() を使用する場合、
    // writer の memory_order を release に戻す必要がある（reader 側が acquire のため）。
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
 #if defined(CONVO_USE_IS_RT_CONTEXT)
    // CONVO_USE_IS_RT_CONTEXT 定義時: release で HB 保証
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
 #else
    // 通常時: relaxed で安全（HB不要な状態フラグのため）
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
 #endif
#else
    // Debug/CI: release で完全な HB 保証
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}
```

**reader側の安全性（最終確認済み）**:

| reader | memory_order | Release呼出 | isRTContext依存 |
|--------|:-----------:|:-----------:|:--------------:|
| `auditPublishAttempt()` | acquire | ❌ (`#if JUCE_DEBUG`) | ❌ |
| `onAllocAttempt()` | acquire | ❌ (`#if JUCE_DEBUG`) | ❌ |
| `isRTContext()` | acquire | ✅ (実装済みだが未呼出) | **N/A** |

**ファイル**: `ISRRTExecution.h`, `ISRRTExecution.cpp` | **所要**: 1h

---

### 新規A: static_assert(sizeof(CallbackTimingEntry)) 📋

```cpp
#if defined(__cpp_lib_hardware_interference_size)
static_assert(sizeof(CallbackTimingEntry) <= std::hardware_destructive_interference_size,
    "CallbackTimingEntry must fit in one cache line");
#else
static_assert(sizeof(CallbackTimingEntry) <= 64,
    "CallbackTimingEntry must fit in one cache line (64 bytes)");
#endif
```

**ファイル**: `AudioEngine.h` | **所要**: 0.5h

---

### P1-4/5: false sharing 最小限 `alignas(64)` ×1 📋

```cpp
// ★ mmcssShutdownRequested: Message → Audio Thread 通知
//   書込頻度: シャットダウン時のみのため false sharing 影響は極小。
//   alignas(64) は「将来ここだけ分離したい」意思表示として配置。
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};

// mmcssApplied_, useMmcssPriority は書込頻度が非常に低いため隔離不要
std::atomic<bool> mmcssApplied_{false};
std::atomic<bool> useMmcssPriority{true};
```

**ファイル**: `AudioEngine.h` | **所要**: 1h

---

## 5. P3 — 調査・保留（変更なし）

| ID | 状態 | 項目 | 結論 |
|:--:|:----:|------|------|
| P1-3 | ❌ 撤回 | QPC→TSC | QPC 12.8ns、差5.4nsに見合わず |
| 新規B | ❌ 削除 | prepareToPlay初期化 | thread_localはスレッド別 |
| P2-6 | ❌ 削除 | CPU番号 cache | 診断劣化 |
| P3-1 | ✅ 完了 | musicalSoftClipScalar | libm不使用。Pade近似で安全 |
| P3-2 | ✅ 完了 | MMCSSタイミング | prepareToPlay≠Audio Thread |
| P3-3 | ⏸️ 保留 | ScopedNoDenormals | 数十cycle、例外経路リスク |

---

## 6. QPC/RDTSC/chrono 実測ベンチマーク

### 環境

| 項目 | 値 |
|:----|:---:|
| CPU | Intel Core (TSC: 3599.96 MHz) |
| Invariant TSC | ✅ 有効 |
| OS | Windows 11 build 26100 |
| コンパイラ | MSVC 19.51 `/O2` |
| QPC実装 | TSCベース（Windows 8+ 標準） |
| 計測 | 100万回ループ、volatile sink で最適化防止 |

### 結果

| タイマ | 1回あたり | QPC比 | 判定 |
|:------:|:---------:|:-----:|:----:|
| `QueryPerformanceCounter` | **12.8 ns** | 1.0x | ✅ 現状維持 |
| `__rdtsc` | **7.4 ns** | 0.58x | ❌ 差5.4ns。複雑性に見合わず撤回 |
| `std::chrono::steady_clock::now()` | **16.2 ns** | 1.27x | 誤差範囲 |
| `chrono→microseconds` | **17.7 ns** | 1.38x | 除算1回分 |

---

## 7. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 `#if` 追加 |
| P1-1 | `src/core/ThreadHash.h` | ✨ **新規** |
| P1-1 | `src/DspNumericPolicy.h` | ⚡ `currentThreadTag()` → `cachedThreadHash()` |
| P1-1 | `src/core/RCUReader.h` | ⚡ `currentThreadToken()` → `cachedThreadHash()` |
| P1-1 | `src/core/EpochDomain.h` | ⚡ hash呼出 → `cachedThreadHash()` |
| P1-2+4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡📋 Telemetry＋共通timestamp |
| P1-2+4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡📋 同double版 |
| P1-6 | `src/MKLNonUniformConvolver.cpp` | 🛡️ `ASSERT_NON_RT_THREAD()` |
| P1-7 | `src/MKLNonUniformConvolver.cpp` | 🛡️ `Logger::writeToLog` → `diagLog()` |
| P1-8 | `src/DspNumericPolicy.h` | ⚡ `isAudioThread()` → `cachedThreadHash()` |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 診断サンプリング前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-2 | `src/audioengine/ISRLifecycle.cpp` | 📋 `traceFull_` フラグ |
| P2-2 | `src/audioengine/ISRLifecycle.h` | 📋 `traceFull_` メンバ |
| P2-3 | `src/audioengine/ISRRTExecution.cpp` | 📋 `release`→`relaxed` + `#if` 分岐 |
| P2-3 | `src/audioengine/ISRRTExecution.h` | 📋 コメント更新 |
| 新規A | `src/audioengine/AudioEngine.h` | 📋 `static_assert`×2（`#if` ガード付） |
| P1-4/5 | `src/audioengine/AudioEngine.h` | 🛡️ `alignas(64)` ×1 |

**ファイル数**: 新規1 + 修正11 = **全12ファイル** | **変更行数**: 約85行

---

## 8. 推奨実装順序

```
Day 1 (2.5h):
  P0-1  diagLog guard                [独立]
  P1-1  thread_local hash cache       [独立]
  P1-8  isAudioThread最適化            [P1-1完了後]

Day 2 (4h):
  P1-2 + P2-4  Telemetry+共通timestamp  [同一ファイル、同時実装] ★
  P1-6  MKL ASSERT                    [独立]
  P1-7  MKL Logger統一                [P0-1完了後]

Day 3 (4.5h):
  P2-1  sampling前倒し                [独立]
  P2-2  traceFull_                    [独立]
  新規A  static_assert                 [独立]
  P2-3  Firewall relaxed              [独立]

Day 4 (4h):
  P1-4/5  alignas(64)最小限           [独立]
  全ビルド構成検証（Release/Debug）
  長期安定性テスト（5分間連続再生 XRUN 0回）
```

**合計**: 約10-13h（余裕見て3-4日）

---

## 9. レビュー履歴と最終判定

### 全5回のレビューで修正された項目

| レビュー | 反映版 | 主な修正 |
|---------|:-----:|---------|
| 1回目 | v3→v4 | TSC撤回・false sharing絞り込み・QPC実測 |
| 2回目 | v4→v5 | 新規B削除・Firewall relaxed・ctor引数・早期return |
| 3回目 | v5→v5.1 | P2-4説明修正・`#if` ガード・ラッパ構造体・QPC詳細化 |
| 4回目 | v5.1→v5.3 | ラッパ撤回→単純alignas・`#ifdef` + `#pragma message` |
| **5回目（最終）**| **v5.3→v5.4** | **`#pragma message`削除・relaxed理由コメント・実装順序最適化** |

### 最終判定

| カテゴリ | 件数 | 該当項目 |
|:---------|:----:|---------|
| ✅ **実装可** | 12 | 全P0/P1/P2項目 |
| ❌ **撤回** | 3 | P1-3(TSC), P2-6(CPU cache), 新規B(thread_local初期化) |
| ✅ **調査完了** | 2 | P3-1(musicalSoftClip), P3-2(MMCSSタイミング) |
| ⏸️ **保留** | 1 | P3-3(ScopedNoDenormals) |

**v5.4 は全レビュー指摘を反映した最終確定版である。実装に着手可能。**
