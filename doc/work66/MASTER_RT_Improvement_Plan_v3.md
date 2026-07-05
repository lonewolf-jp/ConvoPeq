# 統合版 オーディオスレッドリアルタイム性向上改修計画書 v3.0

**作成日**: 2026-07-05 | **最終更新**: 2026-07-05
**対象ブランチ**: `main`
**ベース**: v2計画書 + 全ツール網羅調査 + 実装レビューフィードバック反映

---

## 改訂履歴

| 版 | 日付 | 変更内容 |
|:--:|:----:|---------|
| v1 | 07-05 | 初版。分析文書に基づく9タスク |
| v2 | 07-05 | 完全監査結果統合。18項目に拡充 |
| **v3** | **07-05** | **レビューフィードバック反映。false sharing対策再設計。P1-3/4/5, P2-3/4を修正** |

### 主な v2→v3 変更点

| 項目 | v2 | v3 |
|:----:|:--:|:--:|
| P1-3 XRUN | `getCurrentTimeUs` サンプリング間引き | **RDTSC軽量タイマで毎回計測→必要時のみ `us` 変換** |
| P1-4 RTLocalState | `alignas(128)` 追加 | **padding でスレッド間共有変数を分離した設計に全面再設計** |
| P1-5 atomic align | `alignas(64)` 個別付与 | **専用ラッパ構造体 `alignas(64)` で各atomicを分離** |
| P2-3 Firewall | Release ビルドで完全無効化 | **`relaxed` メモリオーダーに軽減。診断品質維持** |
| P2-4 ScopedNoDenormals | thread_local最適化 | **優先度を P3 に降格。複雑性リスクを再評価** |
| P3-1 musicalSoftClip | 「要調査」 | **調査完了: libm呼出なし → 安全確定** |

---

## 目次

1. [優先度定義](#1-優先度定義)
2. [P0 — 即時修正](#2-p0--即時修正)
3. [P1 — 優先対応](#3-p1--優先対応)
4. [P2 — 計画的対応](#4-p2--計画的対応)
5. [P3 — 調査・検討](#5-p3--調査検討)
6. [confirmed-rt-safe — 検証済み安全項目](#6-confirmed-rt-safe--検証済み安全項目)
7. [全18項目一覧](#7-全18項目一覧)
8. [ファイル変更サマリ](#8-ファイル変更サマリ)
9. [スケジュールと依存関係](#9-スケジュールと依存関係)
10. [リスク評価](#10-リスク評価)

---

## 1. 優先度定義

| 優先度 | 基準 | 件数 |
|:------:|------|:----:|
| **P0** | 確定バグ。全ビルドで音飛びリスクあり | 1 |
| **P1** | 毎コールバック実行。累積インパクト大 | 8 |
| **P2** | 影響限定的／診断時のみ。中規模リファクタリング | 5 |
| **P3** | 調査済・効果小・複雑性高。優先度低 | 4 |
| **確認済** | 検証の結果、問題なしと確定した項目 | 2 |

---

## 2. P0 — 即時修正

### P0-1: DSPCoreFloat.cpp diagLog() `#if` ガード欠如 🔥

**評価**: ★★★★★ **そのまま実装**

`diagLog()` が `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガードされていない。
誤ってホットパスから呼ばれた場合、`juce::Logger::writeToLog()`（ミューテックス＋ファイルI/O）が全ビルドで有効。

**修正**: 定義全体を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS ... #endif` で囲む

```cpp
// Before
namespace {
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

// After
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

**ファイル**: `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp`
**所要**: 0.5h | **リスク**: 極小

---

## 3. P1 — 優先対応

### P1-1: `std::hash<std::thread::id>` thread_local キャッシュ ⚡

**評価**: ★★★★☆ **そのまま実装。ただし効果は過大評価気味**

| 指標 | 推定値 |
|------|--------|
| hash 1回あたりの削減時間 | MSVC: **5〜30ns**（v2の20-120nsより控えめ） |
| コールバックあたり削減 | **15〜90ns**（v2の20-120nsから修正） |
| ScopedThreadRole ctor含む | ✅ 確実に削減 |

#### 設計（v2から変更なし）

- `src/core/ThreadHash.h` 新規: `convo::cachedThreadHash()` (thread_local)
- `DspNumericPolicy.h`: `currentThreadTag()` → `cachedThreadHash()`
- `RCUReader.h`: `currentThreadToken()` → `cachedThreadHash()`
- `EpochDomain.h`: hash呼出 → `cachedThreadHash()`
- `DspNumericPolicy.h:isAudioThread()`: → `cachedThreadHash()`

**リスク**: 低。DLL境界の影響なし（単一DLL）

---

### P1-2: CallbackTelemetryScope の無条件 getCurrentTimeUs() 条件化 ⚡

**評価**: ★★★★★ **そのまま実装**

`CallbackTelemetryScope` のctor/dtorで毎回 `getCurrentTimeUs()` を呼んでいるが、
CLI telemetry が無効（通常時）なら値は全く使用されない。

#### 設計

```cpp
CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
    : engine(owner)
    , samples(numSamplesIn)
    , enabled(owner.isCliProcessingTelemetryEnabled())
    , startUs(enabled ? convo::getCurrentTimeUs() : 0)   // ★ 条件化
{
}

~CallbackTelemetryScope() noexcept
{
    if (enabled && startUs > 0)                           // ★ ガード
    {
        const uint64_t endUs = convo::getCurrentTimeUs();
        // ...
    }
}
```

**削減**: 無条件2回 → CLI有効時のみ2回、通常0回 | **所要**: 1h

---

### P1-3: XRUN検出パスの軽量タイマ利用 ⚡

**評価**: ★★★☆☆ **設計変更。サンプリング間引き → TSC軽量タイマで毎回計測**

#### なぜサンプリングでは不十分か

ユーザーレビュー指摘: XRUNを1/64にサンプリングすると、発生しても63回見逃す。
診断として価値が大幅に低下する。

#### 改善設計（RDTSC採用）

```cpp
// core/TimeUtils.h に追加
inline uint64_t getCurrentTicks() noexcept
{
    return __rdtsc();  // ~10-20cycles, ユーザー空間のみ
}

// 変換係数（#if 外で初期化）
// static double g_ticksPerUs = ...;  // prepareToPlayで設定

// 使用側
// 毎コールバック: TSC で tick 取得 (10-20 cycles = ~3-5ns)
const uint64_t t0_ticks = convo::getCurrentTicks();
// ... 処理 ...
const uint64_t t1_ticks = convo::getCurrentTicks();

// XRUN検出時のみ us 変換（まれ）
if (xrunDetected) {
    const double usPerTick = 1.0 / g_ticksPerUs;
    const double elapsedUs = static_cast<double>(t1_ticks - t0_ticks) * usPerTick;
}
```

#### 期待効果

| 方式 | 通常時コスト | XRUN検出能力 |
|------|:----------:|:-----------:|
| 現状: QPC毎回 | ~100-200ns | ✅ 100% |
| v2案: サンプリング | ~3-6ns(平均) | ❌ 1/64劣化 |
| **v3案: TSC毎回** | **~5ns** | **✅ 100%** |

**QPC (`QueryPerformanceCounter`) を毎回呼ぶ必要はなく、TSC (`__rdtsc`) で代用する。**
TSCはユーザー空間のみで完結し、パイプラインへの影響もQPCより小さい。
絶対時刻（µs）が必要なのはXRUN発生時のみ。

**ファイル**: `src/core/TimeUtils.h`（TSC関数追加）, `AudioBlock.cpp`, `BlockDouble.cpp`
**所要**: 2h | **リスク**: 低（RDTSCは全x64 CPUで利用可能）

---

### P1-4: RTLocalState false sharing 対策（padding設計）🛡️

**評価**: ★★☆☆☆ **→ v3で全面再設計**

#### なぜ `alignas(128)` だけでは不十分か

ユーザーレビュー指摘: `alignas` は構造体の**開始アドレス**しか保証しない。
構造体内のメンバ間のキャッシュライン競合は防げない。

**例**: RTLocalState 内で `publishTimingWriteCount`（Messageスレッド書込）と
`lastCallbackEndTicks`（音声スレッド書込）が同一キャッシュライン上にある可能性。

#### 正しい設計: padding による分離

```cpp
struct alignas(128) RTLocalState
{
    // ── Group 1: Audio Thread 書込（ホット） ──
    alignas(64) std::atomic<uint64_t> audioCallbackEpochCounter { 0 };
    alignas(64) std::atomic<uint64_t> audioSampleCursorCounter { 0 };
    alignas(64) std::atomic<uint32_t> audioCallbackActiveCount { 0 };
    alignas(64) std::atomic<uint64_t> audioThreadRetireEnqueueDropped { 0 };
    alignas(64) std::atomic<uint64_t> lastCallbackEndTicks { 0 };
    alignas(64) std::atomic<uint64_t> xrunSequenceCounter { 0 };
    alignas(64) std::atomic<uint64_t> lastActivatedGeneration { 0 };

    // ── padding: 非Audio Thread書込との分離 ──
    // publishTimingWriteCount は Message Thread が書込
    // → 前後の Audio Thread 書込変数とは別キャッシュラインに隔離
    std::atomic<uint64_t> publishTimingWriteCount { 0 };  // Message Thread
    PublishTimingEntry publishTimingHistory[16];

    // ── Group 2: Audio Thread 診断読取 ──
    alignas(64) std::atomic<uint64_t> lastCallbackEntryUs { 0 };
    alignas(64) std::atomic<int64_t>  lastCallbackDriftUs { 0 };
    alignas(64) std::atomic<uint32_t> lastCallbackProcessor { UINT32_MAX };
    alignas(64) std::atomic<uint64_t> cpuMigrationCount { 0 };
    alignas(64) std::atomic<uint64_t> lastCallbackPublicationSeq { 0 };

    // ── Group 3: CallbackTimingHistory（Audio Thread 書込） ──
    // 配列32要素 + その他 → キャッシュライン汚染注意
    alignas(64) CallbackTimingEntry callbackTimingHistory[32];
    std::atomic<uint64_t> callbackTimingWriteCount{0};

    // ── Group 4: 非atomic読み取り専用 ──
    uint64_t expectedCallbackIntervalUs { 0 };
    uint32_t cachedThreadId { 0 };
};
```

**代替案（より安全）**: 構造体をスレッド所有権単位に分割

```cpp
struct alignas(64) RTAudioThreadState { /* Audio Threadのみ書込 */ };
struct alignas(64) RTMessageThreadState { /* Message Threadのみ書込 */ };
struct alignas(64) RTDiagnosticState { /* 診断読取専用 */ };
struct RTLocalState {
    RTAudioThreadState audio;
    RTMessageThreadState message;
    RTDiagnosticState diagnostic;
};
```

**ファイル**: `src/audioengine/AudioEngine.h`
**所要**: 2-4h | **リスク**: 中（構造体分割は参照箇所が多い。コンパイルエラーで検出可能）

---

### P1-5: AudioEngine atomic false sharing 対策 🛡️

**評価**: ★★★☆☆ **v3で専用ラッパ構造体による分離に変更**

#### なぜ `alignas(64) atomic<bool>` だけでは不十分か

コンパイラは `alignas(64)` の後に後続変数を詰めて配置する可能性がある。
確実に分離するには、各atomicを専用構造体でラップする。

#### 設計

```cpp
// ラッパ構造体
struct alignas(64) AlignedAtomicBool {
    std::atomic<bool> value { false };
};

// 使用側
AlignedAtomicBool mmcssShutdownRequested;    // Message Thread書込
AlignedAtomicBool mmcssApplied_;              // 音声スレッド書込 → 別cacheline確保
AlignedAtomicBool useMmcssPriority;           // Message Thread書込
```

または、対象変数をグループ化してpadding:

```cpp
// Message Thread 書き込みグループ
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};
char _pad1[48];  // 残り48byte padding（64 - sizeof(atomic<bool>)）

// Audio Thread 書き込みグループ
alignas(64) std::atomic<bool> mmcssApplied_{false};
char _pad2[48];
```

**ファイル**: `src/audioengine/AudioEngine.h`
**所要**: 1-2h | **リスク**: 低（padding追加のみ）

---

### P1-6: IppFFTPlanCache::getOrCreate() に ASSERT_NON_RT_THREAD() 🛡️

**評価**: ★★★★★ **そのまま実装**

`getOrCreate()` は `std::lock_guard<std::mutex>` + `make_unique` + `ippsMalloc_8u` を含む。
将来RTパスから誤呼出された場合、音飛び確実。CIで検出可能にする。

```cpp
static const IppFFTPlan* getOrCreate(int order)
{
    ASSERT_NON_RT_THREAD();  // ★ 追加
    std::lock_guard<std::mutex> lock(getMutex());
    // ...
}
```

**ファイル**: `src/MKLNonUniformConvolver.cpp` | **所要**: 0.5h

---

### P1-7: MKLNonUniformConvolver.cpp Logger → diagLog 統一 🛡️

**評価**: ★★★★★ **そのまま実装**

```cpp
// Before
juce::Logger::writeToLog("MKLNonUniformConvolver: FFT plan cache creation failed...");

// After (P0-1完了後)
diagLog("MKLNonUniformConvolver: FFT plan cache creation failed...");
```

**ファイル**: `src/MKLNonUniformConvolver.cpp`（line 697, 704-708）
**所要**: 0.5h | **依存**: P0-1完了後（diagLogが全ビルドで安全になってから）

---

### P1-8: Debug isAudioThread() の hash 重複計算抑制 ⚡

**評価**: ★★★★☆ **そのまま実装**

`ASSERT_AUDIO_THREAD()` 内の `isAudioThread()` が毎回 `hash<thread::id>` を呼ぶ。
`cachedThreadHash()` で置換。

**ファイル**: `src/DspNumericPolicy.h` | **所要**: 0.5h | **依存**: P1-1

---

## 4. P2 — 計画的対応

### P2-1: Diagnostics 収集のサンプリング前倒し 📋

**評価**: ★★★★★ **そのまま実装**

診断ビルド時、`getCurrentTimeUs()` 収集自体をサンプリングマスクでガードする。
現在は出力のみ間引かれ、収集は毎回実行されている。

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

**削減**: 診断時8回→1-2回 | **ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 2h

---

### P2-2: kTraceBufferSize 超過後の branch 回収 📋

**評価**: ★★★★★ **そのまま実装**

`ISRLifecycle.cpp:transitionTo()` で毎回 `fetchAddAtomic(traceWriteIndex_)` と
`if (idx < kTraceBufferSize)` を実行。安定後は `traceFull_` フラグでスキップ。

```cpp
if (traceFull_.load(std::memory_order_relaxed))
    ; // 満杯 → fetchAddAtomic不要
else {
    const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, ...);
    if (idx < kTraceBufferSize) { /* ... */ }
    else { traceFull_.store(true, std::memory_order_release); }
}
```

**削減**: 安定状態で `fetchAddAtomic(acq_rel)` → `relaxed load` | **所要**: 1h

---

### P2-3: RTCapabilityFirewall メモリオーダー軽減（完全無効化→relaxed）📋

**評価**: ★★☆☆☆ **→ 完全無効化ではなく relaxed 化に変更**

#### なぜ完全無効化が危険か

ユーザーレビュー指摘: Firewall はRT侵入検知そのもの。
Releaseで消すと現場でRT違反を発見できなくなる。

#### 修正設計

```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),
        .epochId = 0,
        .isValid = true
    };

    // Release: relaxed（診断品質維持。フェンスコスト削減）
    // Debug/CI: release（HB 保証）
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_relaxed);
#else
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);
#endif
    return token;
}
```

| 方式 | Release fence | Debug fence | 診断品質 |
|:----:|:------------:|:-----------:|:--------:|
| v2: 完全無効化 | ❌ 呼出なし | 同 | ❌ 現場で検出不能 |
| **v3: relaxed** | **relaxed** | **release** | **✅ 維持** |

**ファイル**: `ISRRTExecution.h`, `ISRRTExecution.cpp` | **所要**: 1h

---

### P2-4: 共通タイムスタンプ化（getCurrentTimeUs 統合）📋

**評価**: ★★★★★ **新規追加項目。ユーザーレビューより**

現在、同一コールバック内で複数の独立した `getCurrentTimeUs()` が呼ばれている。
「同じイベント時刻」を共有できる箇所は再利用する。

#### 対象箇所

| 呼出 | 現在の行 | 再利用可能な値 |
|------|---------|--------------|
| `startUs` (line 100) | CallbackTelemetry ctor | —（起点） |
| 〜〜〜 DSP実行 〜〜〜 | | |
| `t0_start` (line 500) | XRUN 開始時刻 | **startUs を再利用**（同じ瞬間） |
| `t1_end` (line 515) | XRUN 終了時刻 | `t3` と同じ瞬間 |
| `t3` (line 654) | CALLBACK_STAGE | — |
| `endUs` (line 697) | CallbackTimingHistory | **t3 を再利用**（近接） |

#### 修正イメージ

```cpp
// 関数先頭で共通タイムスタンプ取得
const uint64_t cbStartUs = convo::getCurrentTimeUs();  // 1回のみ

// 以降は cbStartUs を流用
CallbackTelemetryScope callbackTelemetry(*this, numSamples);
callbackTelemetry.overrideStartUs(cbStartUs);  // ★ メンバ追加 or 引数追加

// ...

// XRUN検出でも cbStartUs を流用
// Before: const auto t0_start = convo::getCurrentTimeUs();
// After:  const auto t0_start = cbStartUs;  // ★ 同じ値
```

**削減**: 無条件4回→1回（3回削減、約150-300ns） | **所要**: 2-4h

---

### P2-5: AudioCallbackRuntimeScope atomic 連鎖評価 📋

**評価**: ★★★★★ **まず測定。方針維持**

コールバック開始時に4連続のatomic RMW操作。実測で有意なら P1 へ繰上げ。
本計画では調査のみ。

---

### P2-6: GetCurrentProcessorNumber() 診断時 syscall 削減 📋

**評価**: ★★★★★ **診断ビルド限定で改善**

診断時のみだが毎コールバック複数回 `GetCurrentProcessorNumber()` を呼ぶ。
1回の結果を thread_local にキャッシュする。

```cpp
// thread_local cache
static thread_local uint32_t s_lastCpu = UINT32_MAX;

// 関数内
uint32_t cpu = s_lastCpu;
if (cpu == UINT32_MAX) {
    cpu = static_cast<uint32_t>(::GetCurrentProcessorNumber());
    s_lastCpu = cpu;
}
```

**ファイル**: `AudioBlock.cpp`, `BlockDouble.cpp` | **所要**: 1h

---

## 5. P3 — 調査・検討

### P3-1: musicalSoftClipScalar — 調査完了 ✅ 問題なし

**評価**: ★★★★★ **コード確認完了。libm呼出なし。安全。**

`musicalSoftClipScalar()` は以下を使用:
- `absNoLibm()` — ビット演算による絶対値
- `fastTanh()` — **Pade近似（多項式除算）**。libm の `std::tanh` 不使用
- 四則演算のみ（加算/減算/乗算/除算）

`fastTanh` の実装:
```cpp
inline double fastTanh(double x) noexcept
{
    constexpr double numA = 10395.0, numB = 1260.0, numC = 21.0;
    constexpr double denA = 10395.0, denB = 4725.0, denC = 210.0;
    constexpr double clipThreshold = 4.5;
    const double x2 = x * x;
    const double num = x * (numA + x2 * (numB + x2 * numC));
    const double den = denA + x2 * (denB + x2 * (denC + x2));
    return num / den;
}
```

Pade approximant [7/7] 型。高精度かつlibm不使用でRTセーフ。

**結論: 対応不要。P3-1 はクローズ。**

---

### P3-2: MMCSS applyタイミング → 現状設計が最適

**評価**: ★★★★★ **調査完了。初回コールバック適用が適切。**

ConvoPeq はスタンドアロンアプリだが、JUCE のオーディオデバイス管理により、
`prepareToPlay()`（Message Thread）と `getNextAudioBlock()`（Audio Thread）は
別スレッドで実行される可能性がある。スレッドIDの一致はホスト（ASIO/WASAPI等）
とデバイス設定に依存し、確約できない。

したがって:
- `prepareToPlay()` で MMCSS 適用 → 間違ったスレッドに設定するリスク
- **初回コールバックの CAS ゲート方式** → 正しいスレッドで確実に適用

→ 現状の設計が最適。P3-2 はクローズ。

---

### P3-3: ScopedNoDenormals 最適化（旧P2-4）— 優先度降格

**評価**: ★★☆☆☆ **効果小・複雑性高。優先度を P2→P3 に降格**

`ScopedNoDenormals` はMXCSRレジスタのFTZ/DAZビットを設定する。
thread_local で状態追跡すると:
- 例外経路や再入で状態が壊れるリスク
- 改善量は数十cycle程度
- 現状の重複設定（DSPCore + EQProcessor）でも実害は極めて小さい

**結論: 対応不要。P3-3 は積極的に対応しない。**

---

## 6. ✅ confirmed-rt-safe — 検証済み安全項目

### 全DSPカーネル安全確認（17モジュール）

全モジュールについて「ロックなし・アロケーションなし・libmなし」を確認。

| モジュール | 確認方法 |
|-----------|---------|
| `LockFreeRingBuffer` | 直接読取 |
| `MKLNonUniformConvolver::Add/Get` | 直接読取 |
| `EQProcessor::process` | 直接読取 |
| `OutputFilter::process` | 直接読取 |
| `LoudnessMeter::processBlock` | 直接読取 |
| `TruePeakDetector::processBlock` | 直接読取 |
| `PsychoacousticDither` | 直接読取 |
| `Fixed/LatticeNoiseShaper` | 直接読取 |
| `UltraHighRateDCBlocker` | 直接読取 |
| `CustomInputOversampler::processUp/Down` | 直接読取 |
| `InputBitDepthTransform` | 直接読取 |
| `musicalSoftClipScalar` | **本調査で確定** ✅ |

### 全ミューテックス確認済み（8箇所）

| ファイル | Mutex | 実行スレッド | 確認方法 |
|---------|-------|------------|---------|
| `ISRLifecycle.cpp` | nonRtGuard_ | Message | 直接読取 |
| `PrepareToPlay.cpp` | rebuildMutex | Message | `ASSERT_NON_RT_THREAD()` |
| `ReleaseResources.cpp` | rebuildMutex | Message | 同 |
| `MKLNonUniformConvolver.cpp` | cacheMutex_ | prepare() | コメント確認 |
| `Commit.cpp` | — | Commit | `ASSERT_NON_RT_THREAD()` |
| `Parameters.cpp` | — | Message | 同 |
| `Threading.cpp` | — | Message | 同 |
| `ISRRetireRuntimeEx.cpp` | — | Retire | 同 |

### 全 `diagLog()` 呼出元確認済み（8ファイル中7ファイル安全）

P0-1 修正により全ファイル安全になる。

---

## 7. 全18項目一覧

| ID | 優先度 | カテゴリ | 項目 | 評価 | 対応 |
|:--:|:------:|:--------:|------|:----:|:----:|
| P0-1 | 🔥P0 | バグ | DSPCoreFloat.cpp diagLog() ガード欠如 | ★★★★★ | ✅ 実装 |
| P1-1 | ⚡P1 | 最適化 | thread_local キャッシュ | ★★★★☆ | ✅ 実装 |
| P1-2 | ⚡P1 | 最適化 | CallbackTelemetryScope 条件化 | ★★★★★ | ✅ 実装 |
| P1-3 | ⚡P1 | 最適化 | XRUN検出 TSC軽量タイマ化 | ★★★☆☆ | ⚠️ 設計変更 |
| P1-4 | 🛡️P1 | false sharing | RTLocalState padding分離 | ★★☆☆☆ | ⚠️ 全面再設計 |
| P1-5 | 🛡️P1 | false sharing | AudioEngine atomic ラッパ分離 | ★★★☆☆ | ⚠️ 設計変更 |
| P1-6 | 🛡️P1 | 防御 | getOrCreate() ASSERT追加 | ★★★★★ | ✅ 実装 |
| P1-7 | 🛡️P1 | 統一 | MKL Logger→diagLog | ★★★★★ | ✅ 実装 |
| P1-8 | ⚡P1 | 最適化 | Debug isAudioThread() 最適化 | ★★★★☆ | ✅ 実装 |
| P2-1 | 📋P2 | 最適化 | 診断収集サンプリング前倒し | ★★★★★ | ✅ 実装 |
| P2-2 | 📋P2 | 最適化 | trace buffer branch回収 | ★★★★★ | ✅ 実装 |
| P2-3 | 📋P2 | 最適化 | Firewall relaxed化 | ★★☆☆☆ | ⚠️ 設計変更 |
| P2-4 | 📋P2 | 最適化 | 共通タイムスタンプ統合 | ★★★★★ | **新規** ✅ |
| P2-5 | 📋P2 | 評価 | RuntimeScope atomic連鎖評価 | ★★★★★ | 調査のみ |
| P2-6 | 📋P2 | 最適化 | GetCurrentProcessorNumber 集約 | ★★★★★ | ✅ 実装 |
| P3-1 | 🔍P3 | 調査 | musicalSoftClipScalar | ★★★★★ | ✅ **完了・安全** |
| P3-2 | 🔍P3 | 調査 | MMCSS applyタイミング | ★★★★★ | ✅ **現状最適** |
| P3-3 | 🔍P3 | 最適化 | ScopedNoDenormals 最適化 | ★★☆☆☆ | ⏸️ **保留** |

---

## 8. ファイル変更サマリ

| ID | ファイル | 変更種別 |
|:--:|---------|:--------:|
| P0-1 | `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 `#if` 追加 |
| P1-1 | `src/core/ThreadHash.h` | ✨ 新規作成 |
| P1-1 | `src/DspNumericPolicy.h` | ⚡ hash→cache |
| P1-1 | `src/core/RCUReader.h` | ⚡ hash→cache |
| P1-1 | `src/core/EpochDomain.h` | ⚡ hash→cache |
| P1-2 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ Telemetry条件化 |
| P1-2 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 同double版 |
| P1-3 | `src/core/TimeUtils.h` | ✨ `getCurrentTicks()` 追加 |
| P1-3 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ XRUN→TSC |
| P1-3 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 同double版 |
| P1-4 | `src/audioengine/AudioEngine.h` | 🛡️ padding設計 |
| P1-5 | `src/audioengine/AudioEngine.h` | 🛡️ atomicラッパ |
| P1-6 | `src/MKLNonUniformConvolver.cpp` | 🛡️ ASSERT追加 |
| P1-7 | `src/MKLNonUniformConvolver.cpp` | 🛡️ Logger→diagLog |
| P2-1 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 sampling前倒し |
| P2-1 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-2 | `src/audioengine/ISRLifecycle.cpp/h` | 📋 traceFull_追加 |
| P2-3 | `src/audioengine/ISRRTExecution.cpp/h` | 📋 relaxed化 |
| P2-4 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 共通timestamp |
| P2-4 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |
| P2-6 | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | 📋 CPU番号cache |
| P2-6 | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | 📋 同 |

**ファイル数**: 新規2 + 修正13 = **全15ファイル** | **変更行数**: 約150行

---

## 9. スケジュールと依存関係

### ガントチャート

```
Task                   | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10|
------------------------|----|----|----|----|----|----|----|----|----|----|
P0-1 diagLog guard     | ██ |    |    |    |    |    |    |    |    |    |
P1-1 thread_hash       |    | ██ | ██ |    |    |    |    |    |    |    |
P1-2 Telemetry条件     |    | █  |    |    |    |    |    |    |    |    |
P1-3 TSC timer         |    |    | ██ |    |    |    |    |    |    |    |
P1-4/5 false_sharing   |    |    | ██ | ██ | ██ |    |    |    |    |    |
P1-6/7 MKL ASSERT      |    |    |    | █  |    |    |    |    |    |    |
P1-8 assert最適化      |    |    |    | █  |    |    |    |    |    |    |
P2-1 diag sampling     |    |    |    |    | ██ |    |    |    |    |    |
P2-2 trace回収         |    |    |    |    | █  |    |    |    |    |    |
P2-3 Firewall relaxed  |    |    |    |    | █  | █  |    |    |    |    |
P2-4 共通timestamp     |    |    |    |    |    | █  | ██ |    |    |    |
P2-6 CPU番号cache      |    |    |    |    |    | █  |    |    |    |    |
全ビルド検証           |    |    |    |    |    |    |    | ██ | ██ |    |
長期安定性テスト       |    |    |    |    |    |    |    |    | ██ | ██ |
```

**総所要**: 約10日 | **並行可能**: P1-2/P1-6/P1-7/P1-8 は独立して同時進行可

### 依存関係

```
P1-1 → P1-8 (cachedThreadHash が必要)
P0-1 → P1-7 (diagLog ガード完了後に Logger 置換)
P1-3 → TimeUtils.h 拡張
P1-4 ↔ P1-5 (同一ファイルの構造変更。同時実施推奨)
P2-4 → P1-2 (CallbackTelemetryScope 改修と共通timestampは連動)
```

---

## 10. リスク評価

### false sharing 対策（P1-4, P1-5）

**リスク**: 構造体変更による全参照箇所の再コンパイル必要。
`RTLocalState` は `AudioEngine.h` 内で定義されており、インクルードする全ファイルに影響。

**緩和策**:
- まず `alignas(64)` ＋ padding 方式で実施（構造体分割より影響範囲が小さい）
- 構造体分割は別コミットで行う
- コンパイルエラーで全参照箇所を検出可能

### Firewall relaxed化（P2-3）

**リスク**: `auditPublishAttempt()` の動作が relaxed でも正しいか要確認。
現在の呼出元は `Commit.cpp`（非RTスレッド）のみ。relaxed でも `assert` 動作に影響なし。

### 共通タイムスタンプ（P2-4）

**リスク**: 開始時刻を共有することで、各計測ポイント間の独立性が失われる。
XRUN検出の `t0_start` と `cbStartUs` が同一時刻を指すことは正しい（同一イベント）
が、`t1_end` と `endUs`／`t3` は実際には異なる瞬間を指す可能性がある。

**緩和策**: 「真に同じ瞬間」のものだけ統合する。`t1_end` は残す。

---

## 付録: 調査で確定した3項目

### 確定1: musicalSoftClipScalar ✅ libm呼出なし

```cpp
inline double fastTanh(double x) noexcept { /* Pade近似 [7/7] */ }
```
→ 多項式除算のみ。`std::tanh`, `std::exp`, `std::pow` 不使用。

### 確定2: MMCSS適用タイミング ✅ 初回コールバックが最適

`prepareToPlay()` と `getNextAudioBlock()` は別スレッドの可能性がある。
スレッドID一致は未確認。CASゲート方式を維持。

### 確定3: RDTSC による軽量計測 ✅ QPC代替として有効

x64では全CPUで `__rdtsc` 利用可能。コア間同期は不必要（差分計算のみ利用）。
`QueryPerformanceCounter` のユーザー空間代替として最適。
