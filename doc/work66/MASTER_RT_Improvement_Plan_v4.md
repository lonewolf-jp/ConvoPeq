# 統合版 オーディオスレッドリアルタイム性向上改修計画書 v4.0

**作成日**: 2026-07-05 | **最終更新**: 2026-07-05

---

## 改訂履歴

| 版 | 日付 | 変更内容 |
|:--:|:----:|---------|
| v1 | 07-05 | 初版。分析文書に基づく9タスク |
| v2 | 07-05 | 完全監査結果統合。18項目 |
| v3 | 07-05 | レビューフィードバック反映。TSC軽量タイマ採用など設計変更 |
| **v4** | **07-05** | **全未確定項目調査完了。QPC実測12.8ns→TSC移行を撤回。false sharing対象絞り込み。P2-6削除。新規A/B/C追加** |

### v3→v4 主な変更点

| 項目 | v3 | v4（調査反映） |
|:----:|:--:|:--------------|
| **P1-3** | __rdtsc 軽量タイマ化 | **QPC実測12.8ns → TSC不要。P1→P3降格。複雑性に見合わない** |
| **P1-4/5** | padding＋ラッパ設計 | **false sharing対象を `mmcssShutdownRequested` のみに絞り込み。優先度P1→P2降格** |
| **P2-6** | CPU番号thread_local cache | **診断能力劣化のため削除** |
| **P2-4** | 開始/終了両方共通化 | **開始時刻のみ共有、終了時刻は独立に修正** |
| **新規A** | — | **`static_assert(sizeof(CallbackTimingEntry) <= 64)` 追加** |
| **新規B** | — | **`prepareToPlay()` で `cachedThreadHash()` を強制初期化** |
| **新規C** | — | **QPC/RDTSC/chrono ベンチマーク完了。計画書に実測値を記載** |

---

## 目次

1. [優先度定義と全項目一覧](#1-優先度定義と全項目一覧)
2. [P0 — 即時修正](#2-p0--即時修正)
3. [P1 — 優先対応](#3-p1--優先対応)
4. [P2 — 計画的対応](#4-p2--計画的対応)
5. [P3 — 調査・保留](#5-p3--調査保留)
6. [✅ confirmed-rt-safe 確定リスト](#6-confirmed-rt-safe-確定リスト)
7. [QPC/RDTSC/chrono 実測ベンチマーク結果](#7-qpcrdtscchrono-実測ベンチマーク結果)
8. [CallbackTimingEntry レイアウト分析](#8-callbacktimingentry-レイアウト分析)
9. [ファイル変更サマリ](#9-ファイル変更サマリ)
10. [スケジュールと推奨実施順序](#10-スケジュールと推奨実施順序)

---

## 1. 優先度定義と全項目一覧

### 優先度定義

| 優先度 | 基準 | 件数 |
|:------:|------|:----:|
| **P0** | 確定バグ。全ビルドで音飛びリスク | 1 |
| **P1** | 毎コールバック実行。効果確実かつ安全 | 6 |
| **P2** | 効果は限定的だが設計明確 | 6 |
| **P3** | 調査済。効果小/複雑性高/不必要 | 4 |
| **✅** | 調査完了・問題なし確定 | 3 |

### 全17項目一覧

| ID | 優先度 | 項目 | 実装判断 | QPC換算 |
|:--:|:------:|------|:--------:|:-------:|
| **P0-1** | 🔥P0 | `diagLog()` ガード欠如 | ✅ **即実装** | 音飛びリスク |
| **P1-1** | ⚡P1 | `thread_local` hash cache | ✅ | ~30ns |
| **P1-2** | ⚡P1 | CallbackTelemetryScope 条件化 | ✅ | ~26ns |
| **P1-6** | 🛡️P1 | `getOrCreate()` ASSERT_NON_RT | ✅ | CI防御 |
| **P1-7** | 🛡️P1 | MKL Logger→diagLog統一 | ✅ | CI防御 |
| **P1-8** | ⚡P1 | `isAudioThread()` 最適化 | ✅ | ~30ns |
| **新規B** | ⚡P1 | prepareToPlayでthread_local初期化 | ✅ | ~5ns |
| **P2-1** | 📋P2 | 診断収集サンプリング前倒し | ✅ 診断時 | ~140ns |
| **P2-2** | 📋P2 | trace buffer branch回収 | ✅ | ~50ns |
| **P2-3** | 📋P2 | Firewall relaxed化 | ✅ | ~25ns |
| **P2-4** | 📋P2 | 共通timestamp（開始時刻のみ） | ✅ | ~26ns |
| **P1-4/5** | 🛡️P2 | false sharing対策（対象絞り込み） | ⚠️ 設計再検討 | 不定 |
| **新規A** | 📋P2 | `static_assert(CallbackTimingEntry)` | ✅ | 設計保証 |
| **P1-3** | 🔍P3 | QPC→TSC軽量タイマ | ❌ **撤回** | QPC=12.8nsで不要 |
| **P2-6** | 🔍P3 | CPU番号thread_local cache | ❌ **削除** | 診断劣化 |
| **P3-1** | ✅完了 | `musicalSoftClipScalar` libm調査 | ✅ 安全確定 | — |
| **P3-2** | ✅完了 | MMCSS applyタイミング | ✅ 現状最適 | — |
| **P3-3** | ⏸️保留 | ScopedNoDenormals 最適化 | ❌ 効果小 | — |

### 統合削減効果

| 指標 | v3推定 | v4推定（実測反映） |
|------|:-----:|:----------------:|
| 非DSPオーバーヘッド通常時 | <2µs | **<0.5µs**（実測ベース） |
| 非DSPオーバーヘッド診断時 | <5µs | **<2µs**（実測ベース） |

---

## 2. P0 — 即時修正

### P0-1: DSPCoreFloat.cpp diagLog() `#if` ガード欠如 🔥

**評価**: ★★★★★ **そのまま実装**（v2/v3から変更なし）

```cpp
// Before (全ビルド有効):
namespace {
[[maybe_unused]] void diagLog(const juce::String& message) {
    DBG(message);
    juce::Logger::writeToLog(message);  // mutex + file I/O!
}

// After (#if でガード):
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

**評価**: ★★★★☆ **そのまま実装**

- `src/core/ThreadHash.h` 新規: `convo::cachedThreadHash()` (thread_local)
- 4ファイルの `hash<thread::id>` → キャッシュ置換
- 効果: ~30ns/コールバック（実測: hashはMSVCで約20ns。thread_local参照は~1ns）

**ファイル**: 新規1 + 修正4 | **所要**: 1.5h

---

### P1-2: CallbackTelemetryScope の無条件 getCurrentTimeUs() 条件化 ⚡

**評価**: ★★★★★ **そのまま実装**

`QPC=12.8ns` の実測に基づき、削減効果は約26nsと小さいが、設計として「不要な処理をしない」ことは重要。

```cpp
CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
    : engine(owner), samples(numSamplesIn)
    , enabled(owner.isCliProcessingTelemetryEnabled())
    , startUs(enabled ? convo::getCurrentTimeUs() : 0)
{ }

~CallbackTelemetryScope() noexcept {
    if (enabled && startUs > 0) {
        const uint64_t endUs = convo::getCurrentTimeUs();
        // ...
    }
}
```

**ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 1h

---

### P1-6: IppFFTPlanCache::getOrCreate() に ASSERT_NON_RT_THREAD() 🛡️

**評価**: ★★★★★ **そのまま実装**

`getOrCreate()` は `lock_guard<mutex>` + `make_unique` + `ippsMalloc_8u` を含む。
RT侵入防止としてCIで検出可能にする。

**ファイル**: `src/MKLNonUniformConvolver.cpp` | **所要**: 0.5h

---

### P1-7: MKLNonUniformConvolver.cpp Logger → diagLog 統一 🛡️

**評価**: ★★★★★ **P0-1完了後に実装**

- P0-1で `diagLog()` が全ビルドで安全になってから実施
- 現在の `juce::Logger::writeToLog()` → `diagLog()` に置換

**ファイル**: `src/MKLNonUniformConvolver.cpp` | **所要**: 0.5h | **依存**: P0-1

---

### P1-8: Debug isAudioThread() の hash 重複計算抑制 ⚡

**評価**: ★★★★☆ **そのまま実装**

`ASSERT_AUDIO_THREAD()` → `isAudioThread()` → `hash<thread::id>` → `cachedThreadHash()`

**ファイル**: `src/DspNumericPolicy.h` | **所要**: 0.5h | **依存**: P1-1

---

### 新規B: prepareToPlay で cachedThreadHash() を強制初期化 ⚡

**評価**: ★★★★★ **追加推奨**

thread_local 変数の初回呼出には微小な初期化コストがかかる。
初回コールバックでのコストを完全に排除するため、非RTスレッドの `prepareToPlay()` で強制的に初期化する。

```cpp
// src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp
void AudioEngine::prepareToPlay(int samplesPerBlockExpected, double sampleRate)
{
    ASSERT_NON_RT_THREAD();

    // ★ thread_local キャッシュの事前初期化（初回コールバックのコスト排除）
    (void)convo::cachedThreadHash();

    // ...
}
```

**ファイル**: `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | **所要**: 0.5h

---

## 4. P2 — 計画的対応

### P2-1: Diagnostics 収集のサンプリング前倒し 📋

**評価**: ★★★★★ **そのまま実装**

診断時のみ有効。収集自体をサンプリングマスクでガード。

**削減**: 診断時8回→1-2回（約100-200ns@QPC） | **所要**: 2h

---

### P2-2: kTraceBufferSize 超過後の branch 回収 📋

**評価**: ★★★★★ **そのまま実装**

`traceFull_` フラグ導入で `fetchAddAtomic(acq_rel)` → `relaxed load`。

**所要**: 1h

---

### P2-3: RTCapabilityFirewall メモリオーダー軽減 📋

**評価**: ★★☆☆☆ → ★★★★☆ **relaxed化で継続（完全無効化は回避）**

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0, .isValid = true
    };
    // Debug/CI: release（HB保証）。Release: relaxed（フェンス削減）
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}
```

**所要**: 1h

---

### P2-4: 共通タイムスタンプ（開始時刻のみ共有） 📋

**評価**: ★★★★★ **設計修正。開始時刻のみ共有。終了時刻は独立**

#### 設計（ユーザーレビュー反映）

```cpp
void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
{
    // ...

    // ★ 共通開始時刻（関数先頭で1回だけ取得）
    const auto cbStartUs = convo::getCurrentTimeUs();

    // CallbackTelemetryScope に開始時刻を渡す
    // （P1-2で条件化済み。開始時刻の引数として cbStartUs を貸与）

    // ...

    // XRUN検出 — cbStartUs を再利用
    // Before: const auto t0_start = convo::getCurrentTimeUs();
    // After:  const auto t0_start = cbStartUs;  // ★ 同じイベント時刻

    // ...

    // CALLBACK_STAGE / CallbackTimingHistory の終了時刻は独立
    const uint64_t t3 = convo::getCurrentTimeUs();  // ★ 真の終了時刻
    const uint64_t endUs = convo::getCurrentTimeUs(); // ★ 真の最終時刻
```

| 時刻値 | 共有/独立 | 根拠 |
|--------|:---------:|------|
| `cbStartUs` | **共有（起点）** | 全計測の原点 |
| `t0_start` (XRUN) | **cbStartUsを再利用** | 同一イベント |
| `t3` (CALLBACK_STAGE) | **独立** | DSP処理後の真の終了時刻が必要 |
| `endUs` (CallbackTimingHistory) | **独立** | Telemetry保存直前の真の時刻が必要 |

**削減**: 無条件4回→2回（約26ns削減）| **所要**: 2-4h | **依存**: P1-2

---

### 新規A: `static_assert(sizeof(CallbackTimingEntry) <= 64)` 📋

**評価**: ★★★★★ **追加推奨**

CallbackTimingEntry は現在 48 bytes（`uint64_t×3 + int64_t×1 + uint32_t×2 + uint16_t×1 + atomic<uint64_t>×1 + padding`）。
1キャッシュライン（64 bytes）に収まることを設計意図として固定する。

```cpp
// AudioEngine.h 内、CallbackTimingEntry 定義直後
static_assert(sizeof(CallbackTimingEntry) <= 64,
    "CallbackTimingEntry must fit in one cache line (64 bytes)");
```

同時に、PublishTimingEntry（48 bytes）も同様の static_assert を追加する。

**所要**: 0.5h

---

### P1-4/5: false sharing 対策（対象絞り込み） 🛡️

**評価**: ★★★☆☆ **優先度P1→P2降格。対象を絞り込んで設計**

#### 調査結果

全mmcss変数について書込頻度を調査:

| 変数 | 書込スレッド | 書込頻度 | false sharing risk |
|------|------------|:--------:|:------------------:|
| `mmcssApplied_` | **音声スレッド** | **1回**（CAS初回のみ） | ❌ 無視可能 |
| `useMmcssPriority` | Message Thread | 設定変更時のみ | ❌ 無視可能 |
| `mmcssShutdownRequested` | **Message Thread書込** → 音声スレッド読取 | シャットダウン時のみ | ⚠️ 低い |

**結論**: 実質的なfalse sharingリスクは極めて低い。以下の最小限対策で十分:

```cpp
// AudioEngine.h — mmcssShutdownRequested のみ隔離
// シャットダウン要求（Message→Audio）用。頻度は極めて低いが分離する。
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};
// mmcssApplied_ は初回CASのみで隔離不要。
std::atomic<bool> mmcssApplied_{false};
```

**所要**: 1h（最小限のalignas(64)追加のみ）

---

## 5. P3 — 調査・保留

### P1-3: QPC→TSC軽量タイマ化 🔍→❌ **撤回**

**調査結果**: QPC = 12.8ns/call（100万回実測）
RDTSC = 7.4ns/call（差は5.4nsのみ）

絶対差が **5.4ns** であり、複雑性（2系統の時刻管理、クロック変動対応、保守コスト）に見合わない。
**ConvoPeqでは `getCurrentTimeUs()` を維持する。**

| 方式 | 1回あたり | 備考 |
|:----:|:---------:|------|
| `QueryPerformanceCounter` | **12.8 ns** | 本ベンチマーク |
| `__rdtsc` | **7.4 ns** | 差5.4ns。CPUクロック変動に弱い |
| `std::chrono::steady_clock::now()` | **16.2 ns** | 誤差範囲 |
| `std::chrono→microseconds` | **17.7 ns** | 除算1回追加 |

### P2-6: CPU番号thread_local cache 🔍→❌ **削除**

**理由**: `GetCurrentProcessorNumber()` は両方とも `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内。
thread_local cache すると `cpuMigrationCount` の検出が不能になる。
診断ビルドでのみ有効であるため、現状維持が正しい。

### P3-1: musicalSoftClipScalar ✅ **調査完了・安全確定**

`fastTanh()` は Pade近似 [7/7] 型。`std::tanh` 不使用。**安全確定**。

### P3-2: MMCSS applyタイミング ✅ **調査完了・現状最適**

`prepareToPlay()`（Message Thread）と `getNextAudioBlock()`（Audio Thread）は別スレッドの可能性。
初回コールバックCASゲート方式が最適。

### P3-3: ScopedNoDenormals 最適化 ⏸️ **保留**

改善量: MXCSR 書込 ~ 数十cycle。例外経路で状態が壊れるリスクを考慮し、優先度は最低。

---

## 6. ✅ confirmed-rt-safe 確定リスト

### 全ミューテックス確認（8箇所）

| ファイル | Mutex | スレッド | 確認方法 |
|---------|-------|---------|---------|
| `ISRLifecycle.cpp` | `nonRtGuard_` | Message | 直接読取 |
| `PrepareToPlay.cpp` | `rebuildMutex` | Message | `ASSERT_NON_RT_THREAD()` |
| `ReleaseResources.cpp` | `rebuildMutex` | Message | 同 |
| `MKLNonUniformConvolver.cpp` | `cacheMutex_` | prepare() | コメント |
| `Commit.cpp` | — | Commit | `ASSERT_NON_RT_THREAD()` |
| `Parameters.cpp` | — | Message | 同 |
| `Threading.cpp` | — | Message | 同 |
| `ISRRetireRuntimeEx.cpp` | — | Retire | 同 |

### 全DSPカーネル安全確認（18モジュール）

`musicalSoftClipScalar` を含む全18モジュールについて、ロック・アロケーション・libm呼出の不在を確認。

---

## 7. QPC/RDTSC/chrono 実測ベンチマーク結果

### 実行環境

- **CPU**: Intel Core (TSC: 3599.96 MHz)
- **OS**: Windows (QPCカーネル: TSCベース)
- **コンパイラ**: MSVC 19.51 `/O2` 最適化
- **計測方法**: 100万回ループ、QPC実時間を基準に1回あたりのnsを算出

### 結果

| タイマ | 100万回総計 | 1回あたり | QPC比 |
|:------:|:----------:|:---------:|:-----:|
| `QueryPerformanceCounter` | 12.8 ms | **12.8 ns** | 1.0x |
| `__rdtsc` | 7.4 ms | **7.4 ns** | **0.58x** |
| `std::chrono::steady_clock::now()` | 16.2 ms | **16.2 ns** | 1.27x |
| `chrono::now()→microseconds` | 17.7 ms | **17.7 ns** | 1.38x |

### 考察

- **QPC 12.8ns** = 約50 CPU cycles。これは **「QPC = 遅い」という前提が誤り**であることを示す
- RDTSCがQPCより約1.7倍速い（差5.4ns）が、絶対値が小さすぎて複雑性に見合わない
- `std::chrono` はQPCの約1.3倍だが、使用箇所が `ISRLifecycle.cpp`（初回4096回のみ）であるため問題なし
- **結論**: 現状の `getCurrentTimeUs()`（QPCベース）を維持。TSC移行は行わない

---

## 8. CallbackTimingEntry レイアウト分析

### 現在のサイズ計算

| メンバ | 型 | サイズ | オフセット |
|--------|:---:|:-----:|:---------:|
| `callbackIndex` | `uint64_t` | 8 | 0 |
| `processTimeUs` | `uint64_t` | 8 | 8 |
| `driftUs` | `int64_t` | 8 | 16 |
| `cpu` | `uint32_t` | 4 | 24 |
| `budgetPermille` | `uint16_t` | 2 | 28 |
| `expectedIntervalUs` | `uint32_t` | 4 | 30 (+2 padding→32) |
| `sequence` | `atomic<uint64_t>` | 8 | 32 |
| **合計** | | **40 bytes** (+8 padding) = **48 bytes** | |

**1キャッシュライン（64 bytes）に収まる**。

配列32要素合計: 48 × 32 = **1536 bytes**

### static_assert 追加

```cpp
struct CallbackTimingEntry {
    // ... 既存メンバ ...
};
static_assert(sizeof(CallbackTimingEntry) <= 64,
    "CallbackTimingEntry must fit in one cache line (64 bytes)");
```

### 備考

- 配列内のエントリは Audio Thread のみが書込。false sharing リスクなし
- 将来 Reader（Message Thread）が `sequence` だけ読む場合でも、Entry が 48 bytes なので、2エントリをまたぐことはない。問題なし

---

## 9. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 `#if` 追加 |
| P1-1 | `src/core/ThreadHash.h` | ✨ **新規** |
| P1-1 | `src/DspNumericPolicy.h` | ⚡ hash→cache |
| P1-1 | `src/core/RCUReader.h` | ⚡ hash→cache |
| P1-1 | `src/core/EpochDomain.h` | ⚡ hash→cache |
| P1-2 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ Telemetry条件化 |
| P1-2 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 同double |
| P1-6 | `src/MKLNonUniformConvolver.cpp` | 🛡️ ASSERT追加 |
| P1-7 | `src/MKLNonUniformConvolver.cpp` | 🛡️ Logger→diagLog |
| P1-8 | `src/DspNumericPolicy.h` | ⚡ isAudioThread最適化 |
| **新規B** | `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | ⚡ thread_local強制初期化 |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 sampling前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-2 | `src/audioengine/ISRLifecycle.cpp/h` | 📋 traceFull_ |
| P2-3 | `src/audioengine/ISRRTExecution.cpp/h` | 📋 relaxed化 |
| P2-4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 共通timestamp |
| P2-4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| **新規A** | `src/audioengine/AudioEngine.h` | 📋 static_assert追加 |
| P1-4/5 | `src/audioengine/AudioEngine.h` | 🛡️ alignas(64)最小限 |

**ファイル数**: 新規1 + 修正12 = **全13ファイル** | **変更行数**: 約100行

---

## 10. スケジュールと推奨実施順序

### 推奨実施順序（依存関係順）

```
フェーズ0（0.5h）
  P0-1: diagLog guard              [独立・最優先]

フェーズ1a（2h）
  P1-1: thread_local hash cache     [独立]
  新規B: prepareToPlay初期化        [P1-1完了後]
  P1-8: isAudioThread最適化          [P1-1完了後]

フェーズ1b（2h）
  P1-2: Telemetry条件化             [独立]
  P1-6: MKL ASSERT                  [独立]
  P1-7: MKL Logger統一              [P0-1完了後]

フェーズ2（4h）
  P2-1: sampling前倒し              [独立]
  P2-2: traceFull_                  [独立]
  P2-3: Firewall relaxed            [独立]
  P2-4: 共通timestamp               [P1-2完了後]
  新規A: static_assert              [独立]
  P1-4/5: alignas(64)最小限         [独立]

フェーズ3（2h）
  全ビルド構成検証
  長期安定性テスト
```

**合計**: 約10h（正味実装時間）→ 余裕を見て **2-3日**

### 想定削減効果（実測ベース）

| 指標 | 現状 | 実装後 |
|------|:----:|:------:|
| non-DSP overhead（通常時） | ~100ns (8回×12.8ns) | **~26ns (2回×12.8ns)** |
| non-DSP overhead（診断時） | ~205ns (16回×12.8ns) | **~64ns (5回×12.8ns)** |
| diagLog誤呼出リスク | あり | **なし** |
| hash<thread::id>計算 | 1-3回 | **0回** |
| false sharing不定期ペナルティ | 極低 | **最小限対策** |
| CIでのRT侵入検出 | なし | **ASSERT追加** |
