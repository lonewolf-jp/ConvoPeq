# Audio Thread Real-time Blockers 分析ドキュメント 検証報告書

**検証日**: 2026-07-05
**検証対象**: `doc/work66/AudioThread_RealtimeBlockers_Analysis.md`
**使用ツール**: context-mode MCP, RTK(WSL版), cocoindex-code(ccc.exe), semble, graphify, AiDex MCP, WSL(grep/awk/sed)

---

## 検証サマリー

| セクション | 文書の正確性 | 重要度 |
|-----------|------------|-------|
| [CRITICAL] MMCSS system calls | ✅ **正確** | 変更なし |
| [HIGH] high_resolution_clock | ⚠️ **一部不正確** (過大評価) | [HIGH→MEDIUM] に軽減推奨 |
| [HIGH] RTCapabilityFirewall::enter() | ❌ **不正確** (publishAtomic回数) | [HIGH→MEDIUM] に軽減 |
| [MEDIUM] thread::id hash | ⚠️ **追加の発見あり** (範囲拡大) | 変更なし |
| [MEDIUM] 高atomic密度 | ⚠️ **一部不正確** (数値過大) | 変更なし |
| [LOW-MEDIUM] diagLog定義 | ❌ **重大な不正確** (ガード欠如) | **[LOW-MEDIUM→MEDIUM]** に引上げ推奨 |
| [LOW] getCurrentTimeUs | ⚠️ **過小評価** (無条件呼出あり) | 変更なし |
| [INFO] RT-safe確認 | ✅ **正確** | 変更なし |

---

## 1. [CRITICAL] MMCSS/Affinity/Priority system calls — ✅ 正確

### 文書の主張
> `applyMmcssPriority()` が初回コールバックで実行される。`AvSetMmThreadCharacteristicsA`, `SetPriorityClass(REALTIME_PRIORITY_CLASS)`, `SetThreadAffinityMask` 等のカーネルモード呼出を含む。

### 検証結果 — 正確

- **ファイル**: `src/audioengine/AudioEngine.Timer.cpp:219-306`
- **呼出箇所**:
  - `AudioBlock.cpp:44-48` — `compareExchangeAtomic(mmcssApplied_, false, true)` でガード
  - `BlockDouble.cpp:47-51` — 同上
- **CASゲート**: ✅ `std::atomic<bool> mmcssApplied_{false}` (AudioEngine.h:2113) で正しく制御
- **リセット**: ✅ `prepareToPlay()` (PrepareToPlay.cpp:27) で `publishAtomic(mmcssApplied_, false)` によりリセット
- **該当API**: `AvSetMmThreadCharacteristicsA("Pro Audio")`, `AvSetMmThreadPriority(AVRT_PRIORITY_CRITICAL)`, `SetPriorityClass(REALTIME_PRIORITY_CLASS)`, `SetThreadPriority(THREAD_PRIORITY_TIME_CRITICAL)`, `SetThreadAffinityMask`

**追加発見**: 同一のシステムコール群は `src/core/ThreadAffinityManager.h` と `src/MainApplication.cpp` でも使用されているが、これらは非RTスレッド上での呼出であり問題なし。

**結論**: 文書の記述は完全に正確。初回コールバックでのカーネルモード呼出が最大の再現性あるRTブロッカーである点も正しい。

---

## 2. [HIGH] `high_resolution_clock::now()` — ⚠️ 一部不正確（過大評価）

### 文書の主張
> `ISRLifecycle.cpp:179-205` の `transitionTo()` が毎コールバック `high_resolution_clock::now()` を呼ぶ。enter+leaveで計2回/コールバック。

### 検証結果 — 一部不正確

- **ファイル**: `src/audioengine/ISRLifecycle.cpp:179-205`
- **実際の動作**:
  ```cpp
  const size_t idx = convo::fetchAddAtomic(traceWriteIndex_, ...);
  if (idx < kTraceBufferSize)  // ★ このガードが重要
  {
      // ... clock呼出はここでのみ実行 ...
      traceBuffer_[idx].timestamp_ns = std::chrono::high_resolution_clock::now()
          .time_since_epoch().count();
  }
  ```
- **kTraceBufferSize = 4096** (ISRLifecycle.h:109)

**評価**: `high_resolution_clock::now()` は **最初の4096エントリ（≒2048コールバックのenter+leave）が書き込まれるまで**のみ呼ばれる。これは48kHz/2048サンプルブロックで約85msに相当。実用的にはスタートアップ時の問題であり、永続的なブロッカーではない。

| 側面 | 文書の記述 | 実際 |
|------|-----------|------|
| 毎コールバック呼ばれる | ✅ 正しい（2048回まで） | ✅ |
| 永続的に呼ばれる | ❌ 暗黙の前提 | ❌ 4096回で停止 |
| セマンティクス | 典型的な[HIGH] | 実際は起動時限定 → **[MEDIUM]相当**

**推奨**: 文書に `kTraceBufferSize` の制限を追記し、重要度を [HIGH] から [MEDIUM]（起動時限定）に修正することを推奨。

---

## 3. [HIGH] `RTCapabilityFirewall::enter()` — ❌ 不正確

### 文書の主張
> `ISRRTExecution.cpp:90-108` — `std::this_thread::get_id()` と **2回の `publishAtomic`** を毎コールバック呼ぶ。

### 検証結果 — 不正確（publishAtomicの回数）

実際のコード（ISRRTExecution.cpp:90-108）:
```cpp
FirewallToken RTCapabilityFirewall::enter() noexcept
{
    FirewallToken token{
        .threadId = std::this_thread::get_id(),  // 1回
        .epochId = 0,
        .isValid = true
    };
    convo::publishAtomic(detail::sharedRtContextFlag(), true, std::memory_order_release);  // ★ 1回のみ
    return token;
}
```

| 側面 | 文書の記述 | 実際 |
|------|-----------|------|
| get_id() | ✅ 1回 | ✅ 1回 |
| publishAtomic | ❌ 「2回」 | ✅ 正確には**1回** |

`std::this_thread::get_id()` 自体は軽量（MSVCでは内部スレッドIDのラッパ）。`std::hash<thread::id>` とは異なり計算コストはほぼゼロ。

**推奨**: publishAtomic回数を「2回」から「1回」に修正。また、本質的には [LOW] 相当のコストであり、[HIGH] は過剰。

---

## 4. [MEDIUM] `std::hash<std::thread::id>` — ⚠️ 追加の発見あり

### 文書の主張
> `RCUReader.h:149-152` の `currentThreadToken()` が毎コールバック `std::hash<thread::id>` を計算。`EQProcessor.Processing.cpp:473` と `ConvolverProcessor.Runtime.cpp:199` で使用。

### 検証結果 — 文書の範囲は正確だが、見落としあり

✅ `RCUReader.h:149-151` — 確認済み
```cpp
static uint64_t currentThreadToken() noexcept
{
    return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
}
```

### 文書が見落とした追加の hash<thread::id> 使用箇所

| ファイル | 行 | 関数 | コールバック呼出頻度 |
|---------|-----|------|-------------------|
| `src/core/RCUReader.h` | 149-151 | `currentThreadToken()` | コールバック毎（EQ/Convolverで使用時） |
| `src/DspNumericPolicy.h` | 43 | `currentThreadTag()` | **毎コールバック** — `ScopedThreadRole`経由 |
| `src/core/EpochDomain.h` | 69 | `registerReaderThread()` | 初回登録時のみ — ホットパス外 |
| `src/DspNumericPolicy.h` | 43 | `currentThreadTag()` (in `releaseAudioThreadSlot`) | コールバック終了時 |

`ScopedThreadRole` は AudioBlock.cpp:78 と BlockDouble.cpp:79 で構築され、`acquireAudioThreadSlot()` 内で `currentThreadTag()` (= hash<thread::id> + get_id) を毎コールバック呼ぶ。さらに `RTCapabilityFirewall::enter` での `get_id()` も毎回実行される。

**1コールバックあたりの合計**:
- `ScopedThreadRole` ctor → `acquireAudioThreadSlot()`: 1x hash<thread::id>
- `RCUReader::enter()` (EQ/Convolver使用時のみ): 1x hash<thread::id>
- `RTCapabilityFirewall::enter()`: 1x get_id()（hashではない）

**スレッドIDの不変性に基づく最適化機会**: `thread_local` キャッシュで hash<thread::id> の再計算を完全に排除可能だが、現在コード中に `thread_local` の使用は**皆無**（cocoindex-code grepで確認済み）。

---

## 5. [MEDIUM] High consumeAtomic密度 — ⚠️ 一部不正確

### 文書の主張
> `captureAudioThreadParameterSnapshot()` が「20+」の `consumeAtomic (acquire)` を毎コールバック実行。

### 検証結果 — 数値は過大

**Worldパス**（AudioEngine.h:~3379）: 約9回のconsumeAtomic
- worldからの直接読み出し（routing, automation, coefficient）はアトミックではない（RCU保護済みの安定スナップショットからの読み出し）
- 共通末尾で約8回のconsumeAtomic（analyzerSource, convHCMode等）

**非Worldパス**（AudioEngine.h:~3345）: 約16回のconsumeAtomic
- 8回（個別変数）+ 8回（共通末尾）

| パス | 文書の記述 | 実際の回数 |
|------|-----------|-----------|
| World有り | 「20+」 | 約9回 |
| World無し | 「20+」 | 約16回 |

**ただし**: コールバック全体のアトミック操作総数はAudioBlock.cppだけで19回（consumeAtomic/fetchAddAtomic/publishAtomic/compareExchangeAtomic）。ISRRTExecution.cppやISRLifecycle.cppを含めると**合計30回以上**のアトミック操作が毎コールバック実行されている。文書がこの点について触れていないのは不十分。

---

## 6. [LOW-MEDIUM] diagLog()定義 — ❌ 重大な不正確（ガード欠如）

### 文書の主張
> 両ファイルとも `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 内部で `diagLog()` を定義。誤用時に `juce::Logger` のミューテックス＋ファイルI/O。

### 検証結果 — 重大な不正確

**DSPCoreFloat.cpp:9-13** — ❌ **ガードなし！**
```cpp
namespace
{
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);  // ★ 全ビルドで有効！
}
```

**DSPCoreDouble.cpp:12-16** — ✅ 正しくガードあり
```cpp
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
namespace {
[[maybe_unused]] void diagLog(const juce::String& message)
{
    DBG(message);
    juce::Logger::writeToLog(message);
}
}
```

| ファイル | 文書の記述 | 実際 | リスク |
|---------|-----------|------|-------|
| DSPCoreFloat.cpp | 「`#if`内」 | ❌ **ガードなし** | Releaseビルドでも呼出可能 |
| DSPCoreDouble.cpp | 「`#if`内」 | ✅ 正しくガードあり | 診断時のみ有効 |

**リスク評価**: DSPCoreFloat.cppの `diagLog()` はすべてのビルド構成で `juce::Logger::writeToLog(message)` を実行可能な状態にある（`juce::Logger::writeToLog` は内部でミューテックス＋ファイルI/Oを実行）。現在ホットパスから呼ばれていないが、`[[maybe_unused]]` 属性によりコンパイラ警告も抑制されているため、誤って使用された場合に検出が困難。

**推奨重要度**: [LOW-MEDIUM] では不十分 → **[MEDIUM]** に引上げを推奨。`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガードする修正が望ましい。

---

## 7. [LOW] getCurrentTimeUs() — ⚠️ 過小評価

### 文書の主張
> `AudioBlock.cpp` と `BlockDouble.cpp` の診断パスで5-10+回/ブロック。すべて `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 下。

### 検証結果 — 過小評価（無条件呼出あり）

**実際の呼出回数（AudioBlock.cpp）**:

| 行 | コード | ガード有無 | 分類 |
|----|-------|-----------|------|
| 100 | `startUs(convo::getCurrentTimeUs())` | ❌ **無条件** | CallbackTelemetryScope ctor |
| 106 | `endUs = convo::getCurrentTimeUs()` | ❌ **無条件** | CallbackTelemetryScope dtor |
| 171 | `nowUs = convo::getCurrentTimeUs()` | ✅ `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | A/G/H診断 |
| 356 | `t1_dspStartUs = ...` | ✅ 同上 | DSP stage timing |
| 448 | `t2_dspEndUs = ...` | ✅ 同上 | DSP stage timing |
| 468 | `observeUs = ...` | ✅ 同上 | DSP observe |
| 500 | `t0_start = ...` | ❌ **無条件** | XRUN検出 |
| 515 | `t1_end = ...` | ❌ **無条件** | XRUN検出 |
| 574 | `ev.timestampTicks = ...` | ✅ `#if`内 | ACTIVATE検出 |
| 589 | `nowUs = ...` | ✅ `#if`内 | Drift |
| 654 | `t3 = ...` | ✅ `#if`内 | CBSUMMARY |
| 697 | `endUs = ...` | ✅ `#if`内 | CallbackTiming |

**合計**:
- **無条件**: 4回/コールバック（CallbackTelemetryScope: 2 + XRUN検出: 2）
- **診断時追加**: 8回/コールバック
- **総計**: 12回/コールバック（診断時）

BlockDouble.cppも同様のパターンで12回/コールバック。

`getCurrentTimeUs()` は `std::chrono::steady_clock::now()` ベース（TimeUtils.h:14）で、Windowsでは `QueryPerformanceCounter` と同等のコスト（通常~50-100ns）。4回の無条件呼出はせいぜい~400ns程度だが、診断時12回では~1.2µs。

**結論**: 文書は「診断ガード内のみ」としているが、実際は4回の無条件呼出が存在する。重要度[LOW]は妥当だが、記述の正確性の観点から修正が必要。

---

## 8. [INFO] RT-safe確認項目 — ✅ 正確

### 文書の主張
すべてのDSPカーネル、ロックフリーキュー、アロケーション、I/Oパスが非RTスレッド上で使用されている。

### 検証結果 — 正確

以下の確認を実施:
- ✅ `ASSERT_AUDIO_THREAD()` / `ASSERT_NON_RT_THREAD()` の全使用箇所を確認（grep）
- ✅ 全 `std::lock_guard` / `std::unique_lock` が非RTスレッド上（PrepareToPlay / ReleaseResources / Commit / Parameters）— `ASSERT_NON_RT_THREAD()` で保護
- ✅ AudioBlock.cpp:78, BlockDouble.cpp:79 で `ScopedThreadRole(AudioRealtime)` によるスレッドロール明示
- ✅ DSPCoreホットパスにロック/アロケーションなし
- ✅ XRunBuffer は `LockFreeRingBuffer`（SPSCロックフリー）
- ✅ `diagLog()` 呼出はすべて非RTスレッド（Commit / CtorDtor / Init / Parameters / DSPCoreLifecycle / PrepareToPlay / ReleaseResources）

**スレッドアサーション使用状況**:
| アサーションタイプ | 使用ファイル数 | 合計使用回数 |
|------------------|--------------|------------|
| `ASSERT_AUDIO_THREAD()` | 3 | 毎コールバック実行 |
| `ASSERT_NON_RT_THREAD()` | 7 | 非RT操作毎 |

---

## 総合評価

### 文書の正確性スコア

| カテゴリ | 正確 | 一部不正確 | 不正確 | 重大な不正確 |
|---------|:----:|:---------:|:------:|:-----------:|
| [CRITICAL] | ✅ | | | |
| [HIGH] #1 (clock) | | ⚠️ | | |
| [HIGH] #2 (firewall) | | | ❌ | |
| [MEDIUM] #1 (hash) | | ⚠️ | | |
| [MEDIUM] #2 (atomic) | | ⚠️ | | |
| [LOW-MEDIUM] (diagLog) | | | | ❌ |
| [LOW] (getCurrentTimeUs) | | ⚠️ | | |
| [INFO] (RT-safe) | ✅ | | | |

### 発見された修正推奨事項（優先順位）

1. **[HIGH] DSPCoreFloat.cpp diagLog() のガード欠如** — `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` で囲む修正が必要。`juce::Logger::writeToLog()` が全ビルドで有効な状態はリスキー。

2. **[MEDIUM] `std::hash<std::thread::id>` の `thread_local` キャッシュ** — 現在コード中に `thread_local` 使用が皆無。`currentThreadToken()` や `currentThreadTag()` の結果を `thread_local` でキャッシュすれば、1コールバックあたり1-3回の hash<thread::id> 計算（数十ns〜数百ns）を削減可能。

3. **[INFO] `getCurrentTimeUs()` の無条件呼出削減** — CallbackTelemetryScope と XRUN検出パスで4回/コールバックの無条件 `getCurrentTimeUs()` が実行されている。これらを診断ガード下に移動するか、サンプリング間引きを導入する余地がある。

### 使用ツール一覧

| ツール | 用途 | 有効性 |
|-------|------|-------|
| **WSL(grep/awk/sed)** | ソースコードのパターン検索・集計 | ✅ 基本ツールとして有効 |
| **RTK (WSL版)** | CLI出力の圧縮フィルタ | ✅ 出力90%削減。必須 |
| **context-mode MCP** | ファイル分析の仮想化 | ✅ 生データをコンテキストに入れず解析 |
| **cocoindex-code (ccc.exe)** | 構造的grep、セマンティック検索 | ✅ grep: 言語認識パターンマッチ。search: 意味的関連の発見 |
| **semble (semble.exe)** | セマンティックコード検索 | ✅ 自然言語クエリで該当箇所を発見 |
| **graphify (graphify.exe)** | 知識グラフ管理 | ⚠️ graphify-out/manifest.json は存在するが graph.json は未生成。今回の検証では間接的に利用 |
| **AiDex MCP** | コードベースファイル一覧・シグネチャ | ✅ プロジェクト全体（288ファイル）の構造把握に有効 |
| **Serena MCP** | ファイル管理 | ⚠️ 今回の検証では直接使用せず（Read/cocoindexで代替可能な範囲だった） |
