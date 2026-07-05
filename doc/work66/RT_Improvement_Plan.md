# オーディオスレッドリアルタイム性向上 改修計画書

**作成日**: 2026-07-05
**ベース文書**: `doc/work66/AudioThread_RealtimeBlockers_Analysis.md`
**検証文書**: `doc/work66/AudioThread_RealtimeBlockers_Verification_Report.md`
**対象ブランチ**: `main`
**優先度**: P0（クリティカル）= 音飛びリスク、P1（重要）= レイテンシ/ジッタ改善、P2（計画的）= 構造的改善

---

## 目次

1. [フェーズ定義と優先順位基準](#1-フェーズ定義と優先順位基準)
2. [P0: 即時修正 — DSPCoreFloat.cpp diagLog() ガード欠如](#2-p0-即時修正)
3. [P1: std::hash<thread::id> thread_local キャッシュ](#3-p1-thread_local-キャッシュ)
4. [P1: 無条件 getCurrentTimeUs() の削減](#4-p1-無条件-getcurrenttimeus-削減)
5. [P1: MMCSS/Affinity system call のタイミング改善](#5-p1-mMcssaffinity-タイミング改善)
6. [P2: アトミック操作密度の低減](#6-p2-アトミック操作密度低減)
7. [P2: Diagnostics サンプリング戦略の改善](#7-p2-診断サンプリング改善)
8. [P2: RTCapabilityFirewall の軽量化](#8-p2-firewall-軽量化)
9. [P2: kTraceBufferSize 超過後の branch 回収](#9-p2-tracebuffer-回収)
10. [リスク評価とロールバック計画](#10-リスク評価)
11. [スケジュールとタスク割当](#11-スケジュール)

---

## 1. フェーズ定義と優先順位基準

### 優先度定義

| 優先度 | 基準 | コスト | 期待効果 |
|--------|------|--------|---------|
| **P0: Critical** | 現在のコードに明確なバグ。全ビルドで音飛びリスク | 修正コスト低（1-2ファイル） | リスク排除 |
| **P1: High** | 毎コールバック実行される。累積インパクト大。コード変更範囲限定 | 修正コスト中（3-8ファイル） | ジッタ低減〜数百ns |
| **P2: Medium** | 影響は限定的／診断時のみ。コード変更範囲広い | 修正コスト高（8-20+ファイル） | 長期的堅牢性向上 |

### 現状のコールバック内オーバーヘッド推定

| 項目 | 単位コスト | 1コールバックあたり回数 | 推定時間 |
|------|-----------|----------------------|---------|
| `getCurrentTimeUs()` (QPC) | ~50-100ns | 4-12回 | ~200-1200ns |
| `std::hash<std::thread::id>` | ~20-40ns | 1-3回 | ~20-120ns |
| `consumeAtomic` (acquire) | ~20-50ns | ~9-30回 | ~180-1500ns |
| `publishAtomic` (release) | ~20-50ns | ~3-8回 | ~60-400ns |
| `fetchAddAtomic` (acq_rel) | ~30-80ns | ~4-10回 | ~120-800ns |
| `compareExchangeAtomic` (acq_rel) | ~50-150ns | 1回 | ~50-150ns |
| **MMCSS API** (初回のみ) | ~10000-50000ns | 1回（計測不能） | ~10-50µs |

**48kHz/1024サンプル = 21.3ms バジェット内で**: これらのオーバーヘッドは総計せいぜい~4µsであり、絶対値としては問題にならない。**しかし、ジッターとしての影響は無視できない**（特に低レイテンシモニタリングや、32サンプルブロックのような小ブロック運用時）。

---

## 2. P0: 即時修正 — DSPCoreFloat.cpp diagLog() ガード欠如 🔥

### 現状の問題

`src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` の `diagLog()` が `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガードされていない。

```cpp
// ★ ガードなし！全ビルドでコンパイルされる
namespace
{
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);  // ミューテックス＋ファイルI/O
}
```

`[[maybe_unused]]` によりコンパイラ警告も抑制されており、誤ってホットパスから呼ばれた場合、音飛び（XRUN）が発生する。

### 修正対象

| ファイル | 行 | 修正内容 |
|---------|-----|---------|
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 6-13 | `diagLog()` 定義全体を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS ... #endif` で囲む |

### 修正方針

```cpp
// Before:
namespace
{
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}

// After:
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

### リスク評価

- **影響範囲**: DSPCoreFloat.cpp のみ。同ファイル内の `setEqDiagBuffer()` は `#if` 外で常時コンパイルする必要があるため、影響を受けない（確認済み）。
- **後方互換性**: ビルドフラグ変更のみ。ABI変更なし。
- **テスト**: `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` と `=0` の両方でビルド確認。

### 改修手順

```
Step 1: DSPCoreFloat.cpp の diagLog() 定義を #if でガード
Step 2: リリースビルド（CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=0）が正常にビルドできることを確認
Step 3: 診断有効ビルド（=1）も正常にビルドできることを確認
```

---

## 3. P1: std::hash<std::thread::id> thread_local キャッシュ

### 現状の問題

`std::hash<std::thread::id>` が毎コールバック1-3回実行される。スレッドIDは不変であるため、一度計算したハッシュ値を `thread_local` でキャッシュできる。

### 該当コード

| ファイル | 関数 | コールバック呼出 |
|---------|------|----------------|
| `src/DspNumericPolicy.h:41-43` | `currentThreadTag()` | ✅ 毎回（ScopedThreadRole経由） |
| `src/core/RCUReader.h:149-151` | `currentThreadToken()` | ✅ EQ/Convolver使用時 |
| `src/core/EpochDomain.h:69` | `registerReaderThread()` | ❌ 初回のみ |

### 修正設計

**アプローチ**: 共通の `thread_local` キャッシュユーティリティを `core/` に追加し、各呼出元から使用する。

#### 新規ファイル: `src/core/ThreadHash.h`

```cpp
#pragma once
#include <thread>
#include <cstdint>

namespace convo {

/// Audioスレッドに最適化された thread::id -> uint64_t キャッシュ。
/// 一度計算したハッシュ値を thread_local に保持する。
inline uint64_t cachedThreadHash() noexcept
{
    static thread_local const uint64_t s_cachedHash =
        static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return s_cachedHash;
}

} // namespace convo
```

#### 修正箇所

| ファイル | 変更前 | 変更後 |
|---------|--------|--------|
| `DspNumericPolicy.h:41-43` | `return static_cast<uint64_t>(std::hash<...>{}...(get_id()));` | `return convo::cachedThreadHash();` |
| `RCUReader.h:149-151` | `return static_cast<uint64_t>(std::hash<...>{}...(get_id()));` | `return convo::cachedThreadHash();` |
| `EpochDomain.h:69` | `std::hash<std::thread::id>{}(...get_id())` | `convo::cachedThreadHash()` |
| `RTCapabilityFirewall::enter()` | `std::this_thread::get_id()` (2回) | 1回に削減可能（後述） |

`std::this_thread::get_id()` 単体は非常に軽量（内部整数値のラッパ）であり、こちらはキャッシュ不要。

### 期待効果

| 項目 | 現在 | 改善後 |
|------|------|--------|
| hash<thread::id> 呼出/コールバック | 1-3回 | 0回（thread_local初期化後） |
| 1回あたりの時間 | ~20-40ns | ~0-1ns（変数読取のみ） |
| コールバックあたり削減時間 | — | ~20-120ns |

### リスク

- `thread_local` の初期化は初回アクセス時に行われる。初回コールバックのレイテンシに微小な影響を与える可能性があるが、MMCSSの初回適用と比較すれば無視できるレベル。
- DLL境界をまたぐ `thread_local` の挙動は実装依存だが、JUCEプラグインは単一DLL内で完結しており問題なし。

---

## 4. P1: 無条件 getCurrentTimeUs() の削減

### 現状の問題

`getCurrentTimeUs()` (= `QPC` / `steady_clock::now()`) が毎コールバック4回、診断時には12回実行される。うち4回は **完全に無条件** で実行される。

| 無条件呼出 | ファイル | 行 | 用途 |
|-----------|---------|-----|------|
| `CallbackTelemetryScope` ctor | `AudioBlock.cpp:100` | 開始時刻記録 |
| `CallbackTelemetryScope` dtor | `AudioBlock.cpp:106` | 終了時刻記録 |
| XRUN t0_start | `AudioBlock.cpp:500` | 開始時刻（interval計測） |
| XRUN t1_end | `AudioBlock.cpp:515` | 終了時刻（callback時間計測） |

BlockDouble.cppも同様。

### 4-A: CallbackTelemetryScope のサンプリング化

`CallbackTelemetryScope` の時間計測をサンプリングベースに変更する。

#### 修正方針

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
        , startUs(enabled ? convo::getCurrentTimeUs() : 0)  // ★ enabled時のみ取得
    {
    }

    ~CallbackTelemetryScope() noexcept
    {
        if (enabled && startUs > 0)  // ★ ガード追加
        {
            const uint64_t endUs = convo::getCurrentTimeUs();
            const uint64_t processTime = (endUs > startUs) ? (endUs - startUs) : 0;
            const double processTimeUs = static_cast<double>(processTime);
            engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
        }
    }
} callbackTelemetry(*this, numSamples);
```

#### 影響分析

| 側面 | 現在 | 変更後 |
|------|------|--------|
| 無条件 getCurrentTimeUs() | 2回（ctor+dtor） | CLI有効時のみ2回、通常時0回 |
| CLI連携 | 常時 `processTime` 記録 | CLI有効時のみ記録。通常時はデータ供給停止 |
| `isCliProcessingTelemetryEnabled()` | 通常時 false | 変更なし |

### 4-B: XRUN検出のサンプリング化

XRUN検出パスは、コールバック時間計測のために `getCurrentTimeUs()` を2回実行している。以下のいずれかの方式で削減する。

#### 方式A（推奨）: XRUN検出のサンプリング間引き

```cpp
// 毎回の完全なXRUN検出をやめ、サンプリング間引きを導入
constexpr uint64_t kXrunSampleMask = 0x3F;  // 1/64
if ((thisCallbackIndex & kXrunSampleMask) == 0)
{
    // 完全なXRUN検出（t0_start / t1_end の getCurrentTimeUs 2回）
    const auto t0_start = convo::getCurrentTimeUs();
    // ... interval/callback時間計測 ...
    const auto t1_end = convo::getCurrentTimeUs();
    // ...
}
else
{
    // 軽量XRUN検出：トリガー方式（atomicカウンタのみ）
    // 何か異常を検出した場合のみカウンタをインクリメント
}
```

#### 方式B: `rtLocalState_` の時刻キャッシュ再利用

`CallbackTelemetryScope` の開始時刻とXRUN開始時刻はほぼ同一タイミングで取得される。両者を統合する:

```cpp
// ★ 開始時刻を一箇所で取得し、複数箇所で再利用
const uint64_t cbStartUs = convo::getCurrentTimeUs();  // 1回のみ無条件
callbackTelemetry.startUs = cbStartUs;                  // Telemetryに貸与
cbPrevEndUs = ...;                                      // XRUNで使用
```

### 期待効果

| 最適化 | 削減回数 | 削減時間 | 困難度 |
|--------|---------|---------|--------|
| Telemetry 条件化 | 2回 | ~100-200ns | 低 |
| XRUNサンプリング | 1-2回 | ~50-200ns | 中 |
| 開始時刻統合 | 1回 | ~50-100ns | 高（構造変更） |

---

## 5. P1: MMCSS/Affinity system call のタイミング改善

### 現状の問題

`applyMmcssPriority()` は初回コールバック内で実行され、複数のカーネルモード呼出（`AvSetMmThreadCharacteristicsA`, `SetPriorityClass`, `SetThreadPriority`, `SetThreadAffinityMask`）を含む。それぞれ10-50µsのオーダーであり、初回コールバックの時間予算を完全に消費する可能性がある。

### 改修オプション

#### オプションA（推奨不可）: prepareToPlay() で実行

```cpp
// ★ 推奨しない — 理由は文書の通り
void AudioEngine::prepareToPlay(...)
{
    ASSERT_NON_RT_THREAD();  // メッセージスレッド
    applyMmcssPriority();   // ここで実行すればRTブロッカーにならない
}
```

**却下理由**: 文書の指摘通り、JUCEホストが `prepareToPlay()` → スレッド破棄 → 別スレッド生成 → `getNextAudioBlock()` のパターンを取る場合、設定が失われる。コントラクト違反ではないが、現実的に発生する。

#### オプションB（推奨）: 初回コールバックの MMCSS適用を defer 可能にする

JUCEのホストによっては `prepareToPlay()` と同じスレッドがコールバックを処理する。その場合は初回コールバックでのシステムコールを回避できる。

```cpp
// AudioEngine.h
std::thread::id prepareToPlayThreadId_;  // prepareToPlay 実行スレッドID

// PrepareToPlay.cpp に追加
void AudioEngine::prepareToPlay(...)
{
    prepareToPlayThreadId_ = std::this_thread::get_id();  // メッセージスレッド
    // ...
}

// MMCSS適用時にスレッド一致チェック
void AudioEngine::applyMmcssPriority() noexcept
{
    if (std::this_thread::get_id() == prepareToPlayThreadId_)
    {
        // ★ prepareToPlay と同じスレッド → 既に非同期待ち不要
        // ただし CAS ゲートはそのまま
    }
    // ...通常のapply...
}
```

#### オプションC（将来構想）: AVX2命令発行とMMCSSの同一周期での実行回避

MMSCSシステムコールの後に最初のAVX2命令が発行されるまでに、OSのスケジューラがコンテキストスイッチを行う可能性がある。これを避けるために、MMCSS適用後最初の数サンプルはスカラー処理にフォールバックする方法もあるが、過剰設計のリスクあり。

### 推奨: オプションB は効果が限定的。本タスクでは対応不要と判断する。

**理由**: MMCSS適用は初回のみであり、CASゲートにより2回目以降は実行されない。初回コールバックがたとえ遅延しても、それがXRUNとして記録されるのはバッファリングのおかげで聴感上の問題になることは稀。本タスクでは優先度を下げる。

---

## 6. P2: アトミック操作密度の低減

### 現状の問題

AudioBlock.cpp だけで毎コールバック19回のアトミック操作が実行される。全体では30回以上に及ぶ。内訳:

| 種類 | 操作 | 回数/コールバック | cost/回 |
|------|------|-----------------|---------|
| `consumeAtomic` (acquire) | 読取+フェンス | ~9-16回 | ~20-50ns |
| `fetchAddAtomic` (acq_rel) | RMW+フェンス | ~4-6回 | ~30-80ns |
| `publishAtomic` (release) | 書込+フェンス | ~3-5回 | ~20-50ns |
| `compareExchangeAtomic` (acq_rel) | CAS+フェンス | 1回 | ~50-150ns |
| `.load(std::memory_order_relaxed)` | relaxed読取 | ~8-12回 | ~5-15ns |
| `.store(std::memory_order_relaxed)` | relaxed書込 | ~3-5回 | ~5-15ns |

### 6-A: acquire → relaxed へのダウングレード（安全な箇所のみ）

AudioBlock.cpp 内の以下の `consumeAtomic` は relaxed にダウングレード可能:

| 行 | 現状のオーダー | ダウングレード可否 | 根拠 |
|----|-------------|----------------|------|
| 352 | `memory_order_relaxed`（既に） | ✅ 変更不要 | — |
| 476 | `memory_order_relaxed`（既に） | ✅ 変更不要 | — |
| 502 | `memory_order_relaxed`（既に） | ✅ 変更不要 | — |
| 544 | `memory_order_relaxed`（既に） | ✅ 変更不要 | — |
| 565 | 要確認 | — | — |
| 659 | `memory_order_relaxed`（既に） | ✅ 変更不要 | — |

→ 多くのconsumeAtomicは既にrelaxedになっている。acquireになっているものは正当な理由があるため、変更は推奨しない。

### 6-B: 複数consumeAtomicのバッチ化

`captureAudioThreadParameterSnapshot()` 内の複数 `consumeAtomic` を1回の構造体ロードにできないか検討する。

ただし、現状の設計では各atomic変数が独立しているため、これは大規模なリファクタリングを要する。本タスクでは **調査のみ** とし、実装は将来課題とする。

### 6-C: fetchAddAtomic の削減

`fetchAddAtomic` はRMW命令であり、acquire/consumeより高価。特に `acq_rel` オーダーはfull memory barrierと同等。

**減少可能なfetchAddAtomic**:
- `audioCallbackActiveCount` (AudioBlock.cpp:65, ~dtor) — デバッグカウンタ。削除または削減可能だが、現状維持を推奨（テレメトリの根幹）
- `xrunSequenceCounter` (AudioBlock.cpp:545) — XRUN毎のみ。問題なし
- `callbackTimingWriteCount` (AudioBlock.cpp:701) — 診断ガード内。問題なし

---

## 7. P2: Diagnostics サンプリング戦略の改善

### 現状

`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=1` 時に、毎コールバック8回の `getCurrentTimeUs()` が診断パスで実行される。サンプリングマスク `CONVOPEQ_DIAG_SAMPLE_MASK` は **出力** の間引きにのみ使用されており、**収集（計測）自体は毎回行われている。**

例えば `CallbackTimingHistory` リングバッファ書込では `getCurrentTimeUs()` は毎回実行されるが、実際に書き込まれるのはサンプリング条件を満たした場合のみ。

### 改善案

#### 7-A: 収集のサンプリング化

```cpp
// Before: 毎回 getCurrentTimeUs() を呼び、後でサンプリング判定
{
    const uint64_t t0 = callbackTelemetry.startUs;
    const uint64_t t3 = convo::getCurrentTimeUs();  // ★ 毎回

    if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    {
        // ... リングバッファ書込 ...
    }
}

// After: サンプリング判定後にのみ計測
{
    if ((thisCallbackIndex & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    {
        const uint64_t t3 = convo::getCurrentTimeUs();  // ★ サンプリング時のみ
        const uint64_t t0 = callbackTelemetry.startUs;
        // ... リングバッファ書込 ...
    }
}
```

#### 期待効果

| 項目 | 現在 | 改善後 |
|------|------|--------|
| 診断時の getCurrentTimeUs() | 8回/コールバック | ~1-2回/コールバック（サンプリング判定1回＋当選時追加） |
| 診断時の削減時間 | — | ~350-700ns |
| データ密度 | 毎回収集→間引いて出力 | 収集自体を間引き |
| 問題点 | 時間粒度が荒くなる | サンプリングマスク次第で十分（現在1/64〜1/256） |

---

## 8. P2: RTCapabilityFirewall の軽量化

### 現状の問題

`RTCapabilityFirewall::enter()` が毎コールバック実行され、`get_id()` と `publishAtomic`（release）を呼ぶ。`leave()` でも同様に `get_id()`（アサーション用）。

### 改善案

#### 8-A: エントリ/終了の省略（安全な場合のみ）

`ASSERT_AUDIO_THREAD()` と `ScopedThreadRole` が既にスレッドの同一性を検証している。`Firewall` は実質的にアサーション用であり、デバッグビルドでのみ有効でリリースビルドでは無効化する選択肢もある。

```cpp
// ISRRTExecution.h
class RTCapabilityFirewall
{
public:
    FirewallToken enter() noexcept
    {
#if defined(NDEBUG) && !defined(CONVO_CI_BUILD)
        return FirewallToken{ .isValid = true };  // リリースビルド: 空
#else
        // 既存の処理
#endif
    }
};
```

これによりリリースビルドでは `Firewall::enter()/leave()` が完全に除去される。`sharedRtContextFlag` の `publishAtomic` も不要になる。

#### リスク

`auditPublishAttempt()` は `sharedRtContextFlag` に依存している。現状の使用箇所は非RTスレッド（`Commit.cpp` 等）であり、リリースビルドではそもそも `assert` が無効化されているため問題なし。

### 期待効果

| 項目 | リリースビルド現在 | リリースビルド改善後 |
|------|-----------------|-------------------|
| RTCapabilityFirewall::enter | publishAtomic 1回 | ゼロ（最適化で消去） |
| leave | publishAtomic 1回 | ゼロ |
| 削減アトミック操作 | — | 2回/コールバック |
| 削減時間 | — | ~40-100ns |

---

## 9. P2: kTraceBufferSize 超過後の branch 回収

### 現状の問題

`ISRLifecycle.cpp:179-205` の `transitionTo()` は、初回4096エントリ書き込み後も毎回 `fetchAddAtomic(traceWriteIndex_)` と `if (idx < kTraceBufferSize)` の分岐を実行する。この分岐は4096回以降は**不発命中**となる。

### 改善案

```cpp
// Before: 毎回実行
void LifecycleIsolationRuntime::transitionTo(LifecyclePhase next)
{
    // ...
    const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, size_t{1}, std::memory_order_acq_rel);
    if (idx < kTraceBufferSize) { /* タイムスタンプ取得 */ }
}

// After: 一度諦めたら二度と試行しない
void LifecycleIsolationRuntime::transitionTo(LifecyclePhase next)
{
    // ...
    if (traceFull_.load(std::memory_order_relaxed))
    {
        // ★ traceBuffer は既に満杯 — fetchAddAtomic も timestamp も不要
    }
    else
    {
        const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, size_t{1}, std::memory_order_acq_rel);
        if (idx < kTraceBufferSize)
        {
            traceBuffer_[idx].timestamp_ns = std::chrono::high_resolution_clock::now()...
        }
        else
        {
            traceFull_.store(true, std::memory_order_release);  // 以後はスキップ
        }
    }
}
```

### 期待効果

| 項目 | 現在 | 改善後 |
|------|------|--------|
| 安定状態でのfetchAddAtomic | 毎回 | なし |
| 安定状態でのclock呼出 | なし（idx判定でガード） | 同左 |
| 安定状態での分岐cost | fetchAddAtomic(acq_rel) + 分岐 | relaxed load + 分岐 |
| 削減時間 | — | ~30-80ns/コールバック |

---

## 10. リスク評価とロールバック計画

### 全改修のリスクマトリクス

| 改修項目 | リスク | 確率 | 影響 | 緩和策 |
|---------|--------|------|------|--------|
| P0: DSPCoreFloat diagLogガード | 非常に低い（#if追加のみ） | 1% | 診断ログ非出力 | ビルド確認で発見可能 |
| P1: thread_localキャッシュ | 低い（新規ユーティリティ追加） | 5% | リンクエラー | 単一ファイル変更、ビルド確認 |
| P1: getCurrentTimeUs削減 | 中（条件分岐の追加） | 10% | CLIテレメトリ非動作 | ロジック変更あり、テスト必須 |
| P2: アトミック操作削減 | 低〜中（個別判断） | 15% | スレッド間可視性違反 | 個別レビュー必須 |
| P2: Firewall軽量化 | 低（#if追加のみ） | 5% | デバッグ時アサーション無効 | デバッグ/リリース両方でビルド確認 |

### ロールバック手順

各改修は**単一コミット**で行い、コミットメッセージに改修タグを含める（例: `[work66-P0] Add missing #if guard to diagLog()`）。問題発生時は該当コミットを `git revert` する。

---

## 11. スケジュールとタスク割当

### フェーズ1: P0 + P1 即効改善（見積: 2-3日）

| # | タスク | 見積 | 依存 |
|---|--------|------|------|
| 1 | DSPCoreFloat.cpp diagLog() ガード修正 | 0.5h | なし |
| 2 | `src/core/ThreadHash.h` 新規作成 | 0.5h | なし |
| 3 | DspNumericPolicy.h: currentThreadTag() 置換 | 0.5h | #2 |
| 4 | RCUReader.h: currentThreadToken() 置換 | 0.5h | #2 |
| 5 | EpochDomain.h: hash呼出置換 | 0.5h | #2 |
| 6 | CallbackTelemetryScope 条件化 | 1.5h | なし |
| 7 | リリース＋デバッグビルド確認 | 1h | #1-6 |
| 8 | コールバック時間の回帰テスト | 2h | #7 |

### フェーズ2: P2 構造的改善（見積: 3-5日）

| # | タスク | 見積 | 依存 |
|---|--------|------|------|
| 9 | Diagnostics サンプリング収集最適化 | 2h | なし |
| 10 | XRUN検出のサンプリング化（方式A） | 3h | なし |
| 11 | kTraceBufferSize branch回収 | 1h | なし |
| 12 | RTCapabilityFirewall リリース時省略 | 1h | なし |
| 13 | 全ビルド構成での検証 | 2h | #9-12 |
| 14 | 長期安定性テスト | 4h | #13 |

### フェーズ3: 計測とチューニング（見積: 2-4日）

| # | タスク | 見積 |
|---|--------|------|
| 15 | Intel VTune Profiler によるコールバック内ホットスポット計測 | 4h |
| 16 | 改修前後の XRUN 発生率比較 | 2h |
| 17 | 小ブロック（32/64/128 samples）での動作検証 | 2h |
| 18 | パフォーマンスリグレッション CI ゲート追加 | 2h |

### 合計見積: 7-12日

---

## Appendix A: 計測方法と合格基準

### XRUN発生率の計測

```cpp
// 現在のXRunBuffer を活用し、あらかじめ定めた期間のXRUN発生数を計測する
// 例: 5分間の連続再生で XRUN 0回
```

### 合格基準

| 指標 | 現状（推定） | 目標 |
|------|------------|------|
| XRUN発生率（通常負荷, 1024spl, 48kHz） | 0回/h | 0回/h（維持） |
| XRUN発生率（高負荷, 64spl, 96kHz） | 測定中 | 0回/h |
| 1コールバックあたりの非DSPオーバーヘッド | ~4µs（推定） | <2µs |
| ジッタ幅（p-p） | 測定中 | <500ns |

---

## Appendix B: コード変更サマリ

| ファイル | 変更種別 | 変更内容 |
|---------|---------|---------|
| `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp` | 🐛 バグ修正 | `diagLog()` を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード |
| `src/core/ThreadHash.h` | ✨ 新規作成 | `convo::cachedThreadHash()` thread_local キャッシュ |
| `src/DspNumericPolicy.h` | ⚡ 最適化 | `currentThreadTag()` → `convo::cachedThreadHash()` |
| `src/core/RCUReader.h` | ⚡ 最適化 | `currentThreadToken()` → `convo::cachedThreadHash()` |
| `src/core/EpochDomain.h` | ⚡ 最適化 | hash呼出 → `convo::cachedThreadHash()` |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ 最適化 | `CallbackTelemetryScope` 条件化, XRUNサンプリング |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp` | ⚡ 最適化 | 同上（float版と同様の変更） |
| `src/audioengine/ISRLifecycle.cpp` | ⚡ 最適化 | `traceFull_` フラグによる分岐回収 |
| `src/audioengine/ISRRTExecution.h` | ⚡ 最適化 | リリースビルド時 `enter()/leave()` 省略 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | ⚡ 最適化 | 診断サンプリング収集の前倒し判定 |

---

## Appendix C: 現状のコールバックフロー（最適化対象の可視化）

```
getNextAudioBlock() / processBlockDouble()
│
├─ consumeAtomic(lifecycleState)                    ← acquire
├─ fetchAddAtomic(audioCallbackEpochCounter)         ← acq_rel
├─ compareExchangeAtomic(mmcssApplied_)              ← acq_rel (初回のみ)
│   └─ applyMmcssPriority() → カーネル呼出群        ← 初回10-50µs
│
├─ AudioCallbackRuntimeScope ctor
│   ├─ lifecycleRuntime_.enterAudioCallback()
│   │   └─ transitionTo(AudioRunning)
│   │       ├─ consumeAtomic(phase_)                 ← acquire
│   │       ├─ publishAtomic(phase_)                 ← release
│   │       ├─ fetchAddAtomic(traceWriteIndex_)      ← acq_rel
│   │       ├─ high_resolution_clock::now()          ← 初回4096回のみ
│   │       └─ consumeAtomic(epochCounter_)          ← acquire
│   ├─ rtCapabilityFirewall_.enter()
│   │   ├─ get_id()                                  ← 軽量
│   │   └─ publishAtomic(sharedRtContextFlag)        ← release
│   └─ fetchAddAtomic(audioCallbackActiveCount)      ← acq_rel
│
├─ ScopedThreadRole ctor
│   └─ acquireAudioThreadSlot()
│       ├─ currentThreadTag() → hash<thread::id>()   ★ thread_local化
│       └─ consumeAtomic + compareExchangeAtomic ×N
│
├─ CallbackTelemetryScope ctor
│   └─ getCurrentTimeUs()                            ★ 条件化
│
├─ [diagnostics] getCurrentTimeUs() ×1-8             ★ サンプリング改善
├─ [diagnostics] consumeAtomic ×2-6
│
├─ XRUN検出ブロック
│   ├─ getCurrentTimeUs() ×2                         ★ サンプリング改善
│   └─ consumeAtomic + fetchAddAtomic ×3-5
│
├─ dsp->process() [DSPカーネル — 最適化対象外]
│
├─ [diagnostics] getCurrentTimeUs() ×2-4             ★ サンプリング改善
│
├─ CallbackTimingHistory (診断時)
│   ├─ getCurrentTimeUs()                            ★ サンプリング改善
│   └─ fetchAddAtomic(callbackTimingWriteCount)       ← acq_rel (診断時)
│
├─ CallbackTelemetryScope dtor
│   └─ getCurrentTimeUs()                            ★ 条件化
│
├─ ScopedThreadRole dtor
│   └─ releaseAudioThreadSlot()
│       ├─ currentThreadTag() → hash<thread::id>()   ★ thread_local化
│       └─ fetchSubAtomic + publishAtomic ×2-3
│
└─ AudioCallbackRuntimeScope dtor
    ├─ fetchSubAtomic(audioCallbackActiveCount)       ← acq_rel
    ├─ rtCapabilityFirewall_.leave()
    │   ├─ get_id() (assert)
    │   └─ publishAtomic(sharedRtContextFlag)        ← release
    └─ lifecycleRuntime_.leaveAudioCallback()
        └─ transitionTo(Prepared)
```

凡例:
- ★ = 本計画で最適化対象
- カーネル呼出（初回MMCSS）= 10-50µsで最大のブロッカーだが、初回限定につき許容
- 最適化によりコールバックあたりのオーバーヘッドを **~4µs → <2µs（推定）** に削減可能
