# 統合版 RT改善改修計画書 v5.1 — 最終微調整版

**作成日**: 2026-07-05 | **対象**: `main`

---

## 改訂履歴

| 版    | 日付      | 主な変更内容                                       |
|:----:|:--------:|---------------------------------------------------|
| v1   | 07-05    | 初版（分析文書に基づく9タスク）                      |
| v2   | 07-05    | 完全監査統合（18項目）                              |
| v3   | 07-05    | TSC軽量タイマ・padding設計反映                      |
| v4   | 07-05    | QPC実測12.8ns→TSC撤回。false sharing絞り込み         |
| v5   | 07-05    | 最終確定版（新規B削除・Telemetry早期return・ctor引数・Firewall relaxed）|
| v5.1 | 07-05    | **最終微調整**: P2-4説明修正, static_assert#ifガード, alignasラッパ, relaxed注意コメント, QPCベンチ詳細化 |

---

## v5.1 の変更点（最終レビュー対応）

| # | 指摘 | v5.1 の対応 |
|:-:|------|-----------|
| ① | P2-4「無条件4回→1回」は誤り | ✅ **通常2→1, CLI時4→2 と正確に修正** |
| ② | `hardware_destructive_interference_size` は `#if` でガードすべき | ✅ **`__cpp_lib_hardware_interference_size` で分岐追加** |
| ③ | `alignas(64)` のみでは後続メンバが同一cachelineに入りうる | ✅ **専用ラッパ構造体で完全分離。ただし現状の対策不要も併記** |
| ④ | P2-3 relaxed: 将来 `isRTContext()` 使用時の注意が不足 | ✅ **コメントとして注意文を記載** |
| ⑤ | QPCベンチに Invariant TSC の記載がない | ✅ **ベンチマーク環境詳細に追記** |

---

## 1. 全項目一覧

| ID | 優先度 | 項目 | 実装判断 |
|:--:|:------:|------|:--------:|
| **P0-1** | 🔥P0 | diagLog() #if ガード欠如 | ✅ **即実装** |
| **P1-1** | ⚡P1 | hash<thread::id> → thread_local cache | ✅ 実装 |
| **P1-2** | ⚡P1 | CallbackTelemetryScope 条件化＋早期return | ✅ 実装 |
| **P1-6** | 🛡️P1 | IppFFTPlanCache::getOrCreate() に ASSERT_NON_RT_THREAD() | ✅ 実装 |
| **P1-7** | 🛡️P1 | MKLNonUniformConvolver.cpp Logger→diagLog | ✅ P0-1後 |
| **P1-8** | ⚡P1 | isAudioThread() の hash 重複計算抑制 | ✅ P1-1後 |
| **P2-1** | 📋P2 | 診断収集サンプリング前倒し | ✅ 実装 |
| **P2-2** | 📋P2 | kTraceBufferSize branch回収 | ✅ 実装 |
| **P2-3** | 📋P2 | Firewall 書込み release→relaxed（将来注意文付） | ✅ 実装 |
| **P2-4** | 📋P2 | 共通timestamp（開始時刻のみctor引数方式） | ✅ P1-2後 |
| **新規A** | 📋P2 | static_assert(CallbackTimingEntry)（#ifガード付）| ✅ 実装 |
| **P1-4/5** | 📋P2 | false sharing 最小限（専用ラッパ構造体） | ⚠️ 軽微修正後 |
| **P1-3** | 🔍→❌ | QPC→TSC軽量タイマ | ❌ **撤回** |
| **P2-6** | 🔍→❌ | CPU番号 thread_local cache | ❌ **削除** |
| **新規B** | 🔍→❌ | prepareToPlayで thread_local 初期化 | ❌ **削除** |
| **P3-1** | ✅ | musicalSoftClipScalar libm調査 | ✅ **安全確定** |
| **P3-2** | ✅ | MMCSS applyタイミング | ✅ **現状最適** |
| **P3-3** | ⏸️ | ScopedNoDenormals 最適化 | ⏸️ **保留** |

---

## 2. P0 — 即時修正

### P0-1: DSPCoreFloat.cpp diagLog() #if ガード欠如 🔥

**変更なし。即実装。**

---

## 3. P1 — 優先対応

### P1-1: std::hash<std::thread::id> thread_local キャッシュ ⚡
### P1-2: CallbackTelemetryScope 条件化＋デストラクタ早期return ⚡
### P1-6: IppFFTPlanCache::getOrCreate() に ASSERT_NON_RT_THREAD() 🛡️
### P1-7: MKLNonUniformConvolver.cpp Logger→diagLog 統一 🛡️
### P1-8: isAudioThread() の hash 重複計算抑制 ⚡

**以上5項目は変更なし。そのまま実装。**

---

## 4. P2 — 計画的対応

### P2-1: 診断収集のサンプリング前倒し 📋（変更なし）
### P2-2: kTraceBufferSize branch回収 📋（変更なし）

---

### P2-3: Firewall 書込み release→relaxed 📋

**v5.1 での改善**: 将来 `isRTContext()` 呼出に対する注意コメントを追加。

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0, .isValid = true
    };
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
    // ★ v5.1: Release ビルドでは relaxed（フェンス削減）
    //   NOTE: 将来 isRTContext() を利用する場合は、
    //   writer の memory_order を release に戻すか、
    //   reader 側の memory_order を relaxed に変更すること。
    //   現在 isRTContext() は未使用のため relaxed で安全。
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}
```

**reader側の安全性（最終確認済み）**:

| reader関数 | memory_order | ガード | Release呼出 | isRTContext使用 |
|-----------|:-----------:|:------:|:-----------:|:--------------:|
| auditPublishAttempt() | acquire | #if JUCE_DEBUG | ❌ | ❌ |
| onAllocAttempt() | acquire | #if JUCE_DEBUG | ❌ | ❌ |
| isRTContext() | acquire | なし | ✅ | **定義のみ未使用** |

**ファイル**: ISRRTExecution.h, ISRRTExecution.cpp | **所要**: 1h

---

### P2-4: 共通timestamp（コンストラクタ引数で開始時刻を受け渡し）📋

**v5.1 での改善**: QPC呼出回数の説明を実装内容と正確に一致させた。

```cpp
struct CallbackTelemetryScope final
{
    AudioEngine& engine;
    int samples;
    bool enabled;
    uint64_t startUs;

    CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn,
                            uint64_t cbStartUs) noexcept
        : engine(owner)
        , samples(numSamplesIn)
        , enabled(owner.isCliProcessingTelemetryEnabled())
        , startUs(enabled ? cbStartUs : 0)
    {
    }

    ~CallbackTelemetryScope() noexcept
    {
        if (!enabled)  // ★ 早期return
            return;

        const uint64_t endUs = convo::getCurrentTimeUs();
        const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;
        const double processTimeUs = static_cast<double>(processTime);
        engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
    }
};

// 呼出側
const auto cbStartUs = convo::getCurrentTimeUs();  // ★ 関数先頭で1回のみ
CallbackTelemetryScope callbackTelemetry(*this, numSamples, cbStartUs);
```

#### QPC呼出回数の正確な変化（v5.1で修正）

| シナリオ | 変更前 | 変更後 | 削減 |
|:--------:|:-----:|:-----:|:----:|
| 通常時（CLI無効） | **2回**（ctor+dtor無条件） | **1回**（cbStartUsのみ） | **-1回** |
| CLI有効時 | **4回**（ctor+dtor+XRUN開始+XRUN終了） | **2回**（cbStartUs+endUs） | **-2回** |

- 削減幅の説明を「無条件4回→1回」から **「通常2→1, CLI時4→2」** に修正

**所要**: 2-4h | **依存**: P1-2

---

### 新規A: static_assert(sizeof(CallbackTimingEntry))（#ifガード付）📋

**v5.1 での改善**: `hardware_destructive_interference_size` が未定義の環境でも安全なフォールバックを追加。

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
#if defined(__cpp_lib_hardware_interference_size)
static_assert(sizeof(CallbackTimingEntry) <= std::hardware_destructive_interference_size,
    "CallbackTimingEntry must fit in one cache line");
#else
static_assert(sizeof(CallbackTimingEntry) <= 64,
    "CallbackTimingEntry must fit in one cache line (64 bytes)");
#endif
static_assert(alignof(CallbackTimingEntry) <= alignof(std::max_align_t),
    "CallbackTimingEntry alignment must not exceed max_align_t");
```

**ファイル**: src/audioengine/AudioEngine.h | **所要**: 0.5h

---

### P1-4/5: false sharing 対策（専用ラッパ構造体化も検討）🛡️

**v5.1 での改善**: 完全なキャッシュライン分離を保証するため、専用ラッパ構造体の方式も併記。

```cpp
// ★ v5.1: 専用ラッパ構造体（alignas(64) のみより確実）
//   後続のメンバ変数が同一キャッシュラインに配置されるのを防ぐ。
struct alignas(64) AtomicBoolCacheLine {
    std::atomic<bool> value{false};
};

// 使用側
AtomicBoolCacheLine mmcssShutdownRequested;  // Message→Audio Thread 通知
```

ただし、`mmcssShutdownRequested` の書込頻度はシャットダウン時のみで極めて低い。
実運用で false sharing による問題が確認された場合のみ本格対策を実施することで十分。

**代替案（現状維持）**: 書込頻度が低いため対策不要という判断も成立する。

**所要**: 1h

---

## 5. P3 — 調査・保留

### P1-3: QPC→TSC軽量タイマ → **撤回完了**（変更なし）

### 新規B: prepareToPlayでthread_local初期化 → **削除完了**（変更なし）

### P3-1: musicalSoftClipScalar → 調査完了・安全確定 ✅

### P3-2: MMCSS applyタイミング → 調査完了・現状最適 ✅

### P3-3: ScopedNoDenormals 最適化 → **保留** ⏸️（変更なし）

---

## 6. QPC実測ベンチマーク詳細（v5.1追記）

### 実行環境（再現性のための詳細）

| 項目 | 値 |
|:----|:---:|
| **CPU** | Intel(R) Core(TM) (TSC: 3599.96 MHz) |
| **TSC特性** | **Invariant TSC** ✅（Intel Turbo/Speed Shift に影響されない） |
| **OS** | Windows 11, build 26100 |
| **コンパイラ** | MSVC 19.51 `/O2`（Release最適化） |
| **QPCカーネル** | TSCベース（Windows 8+ 標準） |
| **計測方法** | 100万回ループ, QPC実時間を基準に1回あたりns算出 |
| **パイプライン影響** | volatile sink で最適化防止済み |

### 結果

| タイマ | 1回あたり | 前提と注意 |
|:------:|:---------:|----------|
| **QPC** | **12.8 ns** | Windows 8+ ではTSCベースで高速。約50 CPU cycles |
| **RDTSC** | **7.4 ns** | 差5.4ns。Invariant TSCなら安全だが、キャリブレーション・仮想化・保守性のコストに見合わない |
| **chrono** | **16.2 ns** | 内部的にはQPC同様の経路。誤差範囲 |
| **chrono→us** | **17.7 ns** | 除算1回（1/1000）の追加コストのみ |

**結論**: QPC 12.8ns は RT用途として十分高速。TSC移行は **複雑性に対して改善量が見合わない**。

---

## 7. CallbackTimingEntry レイアウト分析（最終版）

| メンバ | 型 | サイズ | オフセット |
|--------|:---:|:-----:|:---------:|
| callbackIndex | uint64_t | 8 | 0 |
| processTimeUs | uint64_t | 8 | 8 |
| driftUs | int64_t | 8 | 16 |
| cpu | uint32_t | 4 | 24 |
| budgetPermille | uint16_t | 2 | 28 |
| expectedIntervalUs | uint32_t | 4 | 32 (自然アライン) |
| sequence | atomic<uint64_t> | 8 | 40 |
| **合計** | | **48 bytes** | padding 0, 1 cache line収容 |

**static_assert(<=64)** で設計意図を永続化。（`__cpp_lib_hardware_interference_size` で分岐）

---

## 8. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp | 🐛 #if追加 |
| P1-1 | src/core/ThreadHash.h | ✨ **新規** |
| P1-1 | src/DspNumericPolicy.h | ⚡ currentThreadTag()→cachedThreadHash() |
| P1-1 | src/core/RCUReader.h | ⚡ currentThreadToken()→cachedThreadHash() |
| P1-1 | src/core/EpochDomain.h | ⚡ hash呼出→cachedThreadHash() |
| P1-2 | src/audioengine/AudioEngine.Processing.AudioBlock.cpp | ⚡ Telemetry条件化+早期return+ctor引数 |
| P1-2 | src/audioengine/AudioEngine.Processing.BlockDouble.cpp | ⚡ 同double版 |
| P1-6 | src/MKLNonUniformConvolver.cpp | 🛡️ ASSERT_NON_RT_THREAD() |
| P1-7 | src/MKLNonUniformConvolver.cpp | 🛡️ Logger→diagLog |
| P1-8 | src/DspNumericPolicy.h | ⚡ isAudioThread()最適化 |
| P2-1 | src/audioengine/AudioEngine.Processing.AudioBlock.cpp | 📋 sampling前倒し |
| P2-1 | src/audioengine/AudioEngine.Processing.BlockDouble.cpp | 📋 同 |
| P2-2 | src/audioengine/ISRLifecycle.cpp/h | 📋 traceFull_ |
| P2-3 | src/audioengine/ISRRTExecution.cpp/h | 📋 relaxed化＋注意コメント |
| P2-4 | src/audioengine/AudioEngine.Processing.AudioBlock.cpp | 📋 ctor引数方式＋cbStartUs統合 |
| P2-4 | src/audioengine/AudioEngine.Processing.BlockDouble.cpp | 📋 同 |
| 新規A | src/audioengine/AudioEngine.h | 📋 static_assert×2（#ifガード付） |
| P1-4/5 | src/audioengine/AudioEngine.h | 🛡️ alignas(64)ラッパ構造体 |

**ファイル数**: 新規1 + 修正11 = **全12ファイル** | **変更行数**: 約90行

---

## 9. 最終実装推奨順序（4日）

```
Day 1:  P0-1(0.5h) → P1-1(1.5h) → P1-8(0.5h)                   計 2.5h
Day 2:  P1-2(1h) → P1-6(0.5h) → P1-7(0.5h) → P2-3(1h) → 新規A(0.5h)  計 3.5h
Day 3:  P2-1(2h) → P2-2(1h) → P1-4/5(1h)                         計 4h
Day 4:  P2-4(2-4h, P1-2完了後) → 全ビルド検証(2h) → 長期安定性(2h)   計 6-8h
```

**合計**: 約10-13h（余裕見て3-4日）

---

## 10. 最終判定サマリー

| カテゴリ | v5.1判定 | 該当項目 |
|:---------|:--------:|---------|
| ✅ **そのまま実装可** | 8項目 | P0-1, P1-1, P1-2, P1-6, P1-7, P1-8, P2-1, P2-2 |
| ⚠️ **軽微修正後に実装** | 4項目 | P2-3(注意コメント), P2-4(説明修正), 新規A(#ifガード), P1-4/5(ラッパ構造体) |
| ❌ **撤回/削除** | 3項目 | P1-3(TSC), P2-6(CPU cache), 新規B(thread_local初期化) |
| ✅ **調査完了** | 2項目 | P3-1(musicalSoftClip), P3-2(MMCSSタイミング) |
| ⏸️ **保留** | 1項目 | P3-3(ScopedNoDenormals) |
