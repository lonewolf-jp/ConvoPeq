# 統合版 RT改善改修計画書 v5.3 — 最終確定版

**作成日**: 2026-07-05 | **対象**: `main`

---

## 改訂履歴

| 版 | 日付 | 主な変更内容 |
|:---:|:----:|-------------|
| v1 | 07-05 | 初版（分析文書ベース9タスク） |
| v2 | 07-05 | 完全監査統合（18項目） |
| v3 | 07-05 | TSC軽量タイマ・padding設計 |
| v4 | 07-05 | QPC実測12.8ns→TSC撤回。false sharing絞り込み |
| v5 | 07-05 | 最終確定版（新規B削除・Telemetry早期return・ctor引数・Firewall relaxed）|
| v5.1 | 07-05 | 微調整（P2-4説明修正, static_assert#if, ラッパ構造体, 注意コメント, QPC詳細化） |
| v5.2 | 07-05 | 最終確定: P1-4/5 ラッパ撤回→単純alignas, P2-3 コンパイル時保護 |
| **v5.3** | **07-05** | **P2-3 `#error`+到達不能コードの矛盾を解消。`#if` 分岐を整理し `#pragma message` に変更** |

### v5.2 → v5.3 の変更点

| # | 指摘 | v5.2 | v5.3 |
|:-:|------|------|------|
| ① | P2-3: `#error` の後ろに到達不能な `publishAtomic(release)` | ❌ `#error` + 到達不能コード | ✅ **`#if` 分岐を整理。`CONVO_USE_IS_RT_CONTEXT` 成立時は直接 `release` を使用し、到達不能コードを排除** |

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
| **P2-3** | 📋P2 | Firewall relaxed化（**整理された `#if` 分岐**）| ✅ 実装 |
| **P2-4** | 📋P2 | 共通timestamp（開始時刻のみctor引数） | ✅ P1-2後 |
| **新規A** | 📋P2 | `static_assert(CallbackTimingEntry)`（`#if` ガード付）| ✅ 実装 |
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

## 3. P1 — 優先対応（変更なし）

### P1-1: thread_local hash cache ⚡
### P1-2: Telemetry条件化＋早期return ⚡
### P1-6: MKL ASSERT 🛡️
### P1-7: MKL Logger統一 🛡️
### P1-8: isAudioThread() 最適化 ⚡

---

## 4. P2 — 計画的対応

### P2-1: 診断収集サンプリング前倒し（変更なし）📋
### P2-2: kTraceBufferSize branch回収（変更なし）📋

---

### P2-3: Firewall 書込み release→relaxed + 整理された `#if` 分岐 📋

**v5.3 改善点**: `#error` + 到達不能コードの問題を解消。`CONVO_USE_IS_RT_CONTEXT` が定義された場合は直接 `release` を使用し、同時に `#pragma message` で注意を促す。

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0, .isValid = true
    };

    // ★ v5.3: Release ビルドでは relaxed（フェンス削減）
    //   前提: isRTContext() は現在コードベースで未使用（宣言のみ）。
    //   将来 CONVO_USE_IS_RT_CONTEXT を定義して isRTContext() を使用する場合、
    //   writer の memory_order を release に戻す必要がある（reader 側が acquire のため）。
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
 #if defined(CONVO_USE_IS_RT_CONTEXT)
    // ★ CONVO_USE_IS_RT_CONTEXT 定義時: release で HB 保証
    #pragma message("CONVO_USE_IS_RT_CONTEXT defined: using memory_order_release instead of relaxed")
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
 #else
    // ★ 通常時: relaxed で安全（isRTContext() は未使用）
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
| `isRTContext()` | acquire | ✅ (宣言のみ・未使用) | **N/A** |

**ファイル**: `ISRRTExecution.h`, `ISRRTExecution.cpp` | **所要**: 1h

---

### P2-4: 共通timestamp（コンストラクタ引数で開始時刻を受け渡し）📋

```cpp
struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;

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

// 呼出側
const auto cbStartUs = convo::getCurrentTimeUs();  // ★ 関数先頭で1回のみ
CallbackTelemetryScope callbackTelemetry(*this, numSamples, cbStartUs);
```

#### QPC呼出回数（正確な説明）

| シナリオ | 変更前 | 変更後 | 削減 |
|:--------:|:-----:|:-----:|:----:|
| 通常（CLI無効） | **2回** (ctor+dtor無条件) | **1回** (cbStartUsのみ) | **-1回** |
| CLI有効時 | **4回** (上記+XRUN開始+終了) | **2回** (cbStartUs+endUs) | **-2回** |

**所要**: 2-4h | **依存**: P1-2

---

### 新規A: static_assert(sizeof(CallbackTimingEntry))（#ifガード付）📋

```cpp
#if defined(__cpp_lib_hardware_interference_size)
static_assert(sizeof(CallbackTimingEntry) <= std::hardware_destructive_interference_size,
    "CallbackTimingEntry must fit in one cache line");
#else
static_assert(sizeof(CallbackTimingEntry) <= 64,
    "CallbackTimingEntry must fit in one cache line (64 bytes)");
#endif
```

**所要**: 0.5h

---

### P1-4/5: false sharing 最小限（単純 alignas(64)）📋

```cpp
// ★ mmcssShutdownRequested: Message → Audio Thread 通知
//   書込頻度: シャットダウン時のみのため false sharing 影響は極小。
//   alignas(64) は「将来ここだけ分離したい」意思表示として配置。
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};

// mmcssApplied_, useMmcssPriority は書込頻度が非常に低いため隔離不要
std::atomic<bool> mmcssApplied_{false};
std::atomic<bool> useMmcssPriority{true};
```

**所要**: 1h

---

## 5. P3 — 調査・保留（変更なし）

### P1-3: QPC→TSC → 撤回（QPC実測12.8ns、差5.4nsに見合わず）
### 新規B: prepareToPlayでthread_local初期化 → 削除（スレッド別インスタンス）
### P3-1: musicalSoftClipScalar → 調査完了・安全確定（Pade近似、libm不使用）
### P3-2: MMCSSタイミング → 調査完了・現状最適（prepareToPlay≠Audio Thread）
### P3-3: ScopedNoDenormals → 保留（効果数十cycleに対し例外経路リスク）

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

## 7. CallbackTimingEntry レイアウト

| メンバ | 型 | サイズ | オフセット |
|--------|:---:|:-----:|:---------:|
| callbackIndex | uint64_t | 8 | 0 |
| processTimeUs | uint64_t | 8 | 8 |
| driftUs | int64_t | 8 | 16 |
| cpu | uint32_t | 4 | 24 |
| budgetPermille | uint16_t | 2 | 28 |
| expectedIntervalUs | uint32_t | 4 | 32 |
| sequence | atomic<uint64_t> | 8 | 40 |
| **合計** | | **48 bytes** | 1 cache line収容 |

`static_assert(<=64)` で設計意図を永続化（`__cpp_lib_hardware_interference_size` 分岐付）。

---

## 8. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 `#if` 追加 |
| P1-1 | `src/core/ThreadHash.h` | ✨ **新規** |
| P1-1 | `src/DspNumericPolicy.h` | ⚡ `currentThreadTag()` → `cachedThreadHash()` |
| P1-1 | `src/core/RCUReader.h` | ⚡ `currentThreadToken()` → `cachedThreadHash()` |
| P1-1 | `src/core/EpochDomain.h` | ⚡ hash呼出 → `cachedThreadHash()` |
| P1-2 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ Telemetry条件化+早期return+ctor引数+cbStartUs |
| P1-2 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 同double版 |
| P1-6 | `src/MKLNonUniformConvolver.cpp` | 🛡️ `ASSERT_NON_RT_THREAD()` |
| P1-7 | `src/MKLNonUniformConvolver.cpp` | 🛡️ `Logger::writeToLog` → `diagLog()` |
| P1-8 | `src/DspNumericPolicy.h` | ⚡ `isAudioThread()` → `cachedThreadHash()` |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 診断サンプリング前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-2 | `src/audioengine/ISRLifecycle.cpp` | 📋 `traceFull_` フラグ |
| P2-2 | `src/audioengine/ISRLifecycle.h` | 📋 `traceFull_` メンバ |
| P2-3 | `src/audioengine/ISRRTExecution.cpp` | 📋 `release`→`relaxed` + `#if` 分岐整理 |
| P2-3 | `src/audioengine/ISRRTExecution.h` | 📋 コメント更新 |
| P2-4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 共通timestamp ctor引数（P1-2と一体化） |
| P2-4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| 新規A | `src/audioengine/AudioEngine.h` | 📋 `static_assert`×2（`#if` ガード付） |
| P1-4/5 | `src/audioengine/AudioEngine.h` | 🛡️ `alignas(64)` ×1 |

**ファイル数**: 新規1 + 修正11 = **全12ファイル** | **変更行数**: 約85行

---

## 9. 推奨実装順序

```
Day 1:  P0-1(0.5h) → P1-1(1.5h) → P1-8(0.5h)                     計 2.5h
Day 2:  P1-2(1h) → P1-6(0.5h) → P1-7(0.5h) → 新規A(0.5h)         計 2.5h
Day 3:  P2-1(2h) → P2-2(1h) → P2-3(1h) → P1-4/5(1h)              計 5h
Day 4:  P2-4(2-4h) → 全ビルド検証(2h) → 長期安定性(2h)             計 6-8h
```

**合計**: 約10-13h（余裕見て3-4日）

---

## 10. レビュー履歴と最終判定

### 4回のレビューで修正された全項目

| レビュー | 反映版 | 主な修正 |
|---------|:-----:|---------|
| 1回目 | v3→v4 | TSC撤回・false sharing絞り込み・QPC実測 |
| 2回目 | v4→v5 | 新規B削除・Firewall relaxed・ctor引数・早期return |
| 3回目 | v5→v5.1 | P2-4説明修正・`#if` ガード・ラッパ構造体・QPC詳細化 |
| 4回目（最終）| v5.1→v5.3 | ラッパ撤回→単純alignas・`#ifdef` + `#pragma message` に整理 |

### 最終判定

| カテゴリ | 件数 | 該当項目 |
|:---------|:----:|---------|
| ✅ **そのまま実装可** | 8 | P0-1, P1-1, P1-2, P1-6, P1-7, P1-8, P2-1, P2-2 |
| ✅ **清掃後実装** | 4 | P2-3(整理済), P2-4(正確化済), 新規A(`#if`済), P1-4/5(alignas済) |
| ❌ **撤回/削除** | 3 | P1-3(TSC), P2-6(CPU cache), 新規B(thread_local) |
| ✅ **調査完了** | 2 | P3-1(musicalSoftClip), P3-2(MMCSSタイミング) |
| ⏸️ **保留** | 1 | P3-3(ScopedNoDenormals) |

**全レビュー指摘を反映し、P2-3の `#error` + 到達不能コードの問題も解消。**
**v5.3 は実装着手可能な最終確定版である。**
