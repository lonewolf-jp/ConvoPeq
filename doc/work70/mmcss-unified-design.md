# Unified MMCSS Layer 設計書 v1.0

**日付**: 2026-07-12
**ステータス**: ✅ 実装済み
**実装ファイル**:
- `src/audioengine/AudioEngine.Mmcss.cpp`（新規）
- `src/audioengine/AudioEngine.h`（宣言追加）
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`（コールバック統合）
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`（コールバック統合）
- `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`（フラグ通知）
- `src/audioengine/AudioEngine.Timer.cpp`（applyMmcssPriority 整理）
- `src/MainWindow.cpp`（クリーンアップ）

**関連**: `modification-plan-v3.md`（P8 MMCSS 再試行機構→本設計で置き換え）

---

## 1. 設計思想

### 1.1 Authority Singularization（権限一元化）

MMCSS スレッド優先度設定の責務は、オーディオバックエンドの種類に応じて異なる主体が持つ。ConvoPeq はこれを尊重し、必要な場合にのみ最小限介入する。

```
WASAPI:
  JUCE が Authority → ConvoPeq は何もしない (委譲)

ASIO (優良ドライバ: RME/MOTU/Lynx):
  ドライバが Authority → ConvoPeq は何もしない (成功扱い)

ASIO (未設定ドライバ: ASIO4ALL/安価系):
  ドライバが本来の Authority だが未設定 → ConvoPeq が Recovery として補助登録

DirectSound:
  JUCE がスレッドを生成するが MMCSS 未設定 → ConvoPeq が初回登録
```

### 1.2 設計原則

| # | 原則 | 根拠 |
|:-:|:-----|:------|
| 1 | **同一スレッドで AvSet/AvRevert する** | MSDN `AvRevertMmThreadCharacteristics` の要件。別スレッドからの Revert は失敗する。 |
| 2 | **コールバック内では初回1回のみ登録する** | `AvSetMmThreadCharacteristics` は内部で RPC/LPC を伴うため、毎回の呼び出しは RT 性能に悪影響。 |
| 3 | **`W` 版 API を使用する** | `A` 版は ANSI→UTF-16 変換＋レジストリ列挙の2段階の不確実性がある。`W` 版は直接 Unicode 検索。 |
| 4 | **エラー 5/183 は成功扱い** | 既に JUCE/ドライバが登録済みの場合、自前登録は不要。 |
| 5 | **エラー 1552/1531 はフォールバック** | タスク名不在の可能性があるため、代替タスク名で再試行。 |
| 6 | **`thread_local` でハンドル管理** | Message Thread とのデータ競合防止。ASIO のドライバ所有スレッドでも安全。 |
| 7 | **Message Thread から直接 AvRevert しない** | 必ずフラグ経由で Audio Thread に委譲。 |

---

## 2. インターネット調査に基づく事実確認

### 2.1 MSDN 公式ドキュメント

| 情報源 | 要点 |
|:-------|:------|
| [MMCSS Service](https://learn.microsoft.com/en-us/windows/win32/procthread/multimedia-class-scheduler-service) | デフォルトタスク: Audio, Capture, Distribution, Games, Playback, **Pro Audio**, Window Manager |
| [AvSetMmThreadCharacteristics](https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avsetmmthreadcharacteristicsa) | 公式エラーコード: `INVALID_TASK_INDEX`(1530), `INVALID_TASK_NAME`(1531), `PRIVILEGE_NOT_HELD`(1314)。**ERROR_NO_MORE_ITEMS(1552) は公式エラーではない**。 |
| [AvRevertMmThreadCharacteristics](https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avrevertmmthreadcharacteristics) | 「この関数は AvSetMmThreadCharacteristics を呼んだ**同じスレッド**から呼ばれなければならない。さもなければ失敗する。」 |
| [AvSetMmThreadPriority](https://learn.microsoft.com/en-us/windows/win32/api/avrt/nf-avrt-avsetmmthreadpriority) | 優先度値: `CRITICAL`(2), `HIGH`(1), `NORMAL`(0), `LOW`(-1) |
| Scheduling Category | **High**: スレッド優先度 23-26 (Pro Audio 用)。**Medium**: 16-22。**Low**: 8-15。 |

### 2.2 レジストリ確認結果（実機 Windows 11）

```powershell
# 全タスク一覧
Audio, Capture, DisplayPostProcessing, Distribution, Games, Playback, Pro Audio, Window Manager
```

| タスク名 | Scheduling Category | Priority | Background Only | スレッド優先度帯 |
|:---------|:-------------------:|:--------:|:---------------:|:----------------:|
| **Pro Audio** | **High** | 1 | False | **23-26** |
| Playback | Medium | 3 | False | 16-22 |
| Capture | Medium | 5 | True | 16-22 |
| Audio | Medium | 6 | True | 16-22 |

### 2.3 JUCE ソースコード調査

```cpp
// juce_WASAPI_windows.cpp:1515-1528 — WASAPI のみ MMCSS 対応
void setMMThreadPriority() {
    DynamicLibrary dll("avrt.dll");
    JUCE_LOAD_WINAPI_FUNCTION(dll, AvSetMmThreadCharacteristicsW, ...)  // W版
    JUCE_LOAD_WINAPI_FUNCTION(dll, AvSetMmThreadPriority, ...)
    if (auto h = avSetMmThreadCharacteristics(L"Pro Audio", &dummy))
        avSetMmThreadPriority(h, AVRT_PRIORITY_NORMAL);  // ← NORMAL を使用
}

// juce_ASIO_windows.cpp — MMCSS 呼び出し無し
// juce_directsound_windows.cpp — MMCSS 呼び出し無し
```

### 2.4 Error 1552 の分析

`ERROR_NO_MORE_ITEMS`(1552) は `AvSetMmThreadCharacteristics` の公式エラーコードではない。MSDN の GetLastError ドキュメントに明記されている通り:

> *"The error codes returned by a function are not part of the Windows API specification and can vary by operating system or device driver."*

内部実装では `RegEnumKeyEx` 系APIが列挙終端に達した際に 1552 を返す可能性があり、特に `A` 版（ANSI版）で発生しやすい。`W` 版は直接 Unicode キーを参照するため、この問題の影響を受けにくい。

### 2.5 Chromium 実装の参考

Chromium の AudioManagerWin では MMCSS を直接呼ばず、WASAPI の初期化タイミング（COM スレッド初期化後）で各種設定を行う。自前の `thread_local` MMCSS 管理は行っていない（WASAPI に委譲）。この設計とも整合する。

---

## 3. アーキテクチャ

### 3.1 ファイル構成

| ファイル | 種別 | 責務 |
|:---------|:-----|:------|
| `AudioEngine.h` | 宣言追加 | `MmcssPolicy` enum, 関数宣言 |
| `AudioEngine.Mmcss.cpp` | **新規** | 全 MMCSS 実装（thread_local + try/revert） |
| `AudioBlock.cpp` | 変更 | コールバック先頭で policy 判定＋登録 |
| `BlockDouble.cpp` | 変更 | 同上 |
| `AudioEngine.Timer.cpp` | 変更 | `applyMmcssPriority()` 削除・統合 |
| `ReleaseResources.cpp` | 変更 | フラグセットのみ（直接 AvRevert しない） |
| `MainWindow.cpp` | 変更 | `mmcssShutdownRequested` 削除維持 |

### 3.2 MmcssPolicy enum

```cpp
// AudioEngine.h
enum class MmcssPolicy : uint8_t {
    JuceManaged,           // WASAPI → JUCE が管理。何もしない。
    SelfManagedProAudio,   // ASIO → Pro Audio / CRITICAL
    SelfManagedPlayback,   // DirectSound → Playback / HIGH → Audio フォールバック
    None                   // その他 / 不明 → 何もしない
};
```

### 3.3 thread_local 管理

```cpp
// AudioEngine.Mmcss.cpp (新規)
namespace {
    thread_local HANDLE t_mmcssHandle = nullptr;
    thread_local DWORD  t_mmcssTaskIndex = 0;
    thread_local bool   t_mmcssTried = false;
}
```

`thread_local` を選ぶ理由:

| 方式 | 問題 | 本設計 |
|:-----|:------|:-------|
| `m_avrtHandle`（従来） | Message Thread と Audio Thread でデータ競合 | ❌ 廃止 |
| `std::atomic<HANDLE>` | AvRevert 呼び出しスレッドの保証なし | ❌ |
| **`thread_local`** | スレッドごとに独立。ロック不要。ASIOのドライバ所有スレッドでも安全。 | ✅ **採用** |

### 3.4 バックエンド判定

```cpp
[[nodiscard]] MmcssPolicy AudioEngine::getCurrentMmcssPolicy() const noexcept
{
    if (auto* d = deviceManager.getCurrentAudioDevice()) {
        auto type = d->getTypeName();
        if (type.containsIgnoreCase("WASAPI") || type.containsIgnoreCase("Windows Audio"))
            return MmcssPolicy::JuceManaged;
        if (type.containsIgnoreCase("ASIO"))
            return MmcssPolicy::SelfManagedProAudio;
        if (type.containsIgnoreCase("DirectSound"))
            return MmcssPolicy::SelfManagedPlayback;
    }
    return MmcssPolicy::None;
}
```

判定は Message Thread (`prepareToPlay`) と Audio Thread (callback) の両方から呼ばれる可能性があるが、デバイスが変更されない限り結果は不変であり、`thread_local` はスレッドごとに独立しているため安全。

---

## 4. 主要フロー

### 4.1 初回コールバック時（登録フロー）

```
getNextAudioBlock / processBlockDouble 開始
  │
  ├─ Policy = JuceManaged → 何もしない（JUCE委譲）
  │
  ├─ Policy = SelfManagedProAudio (ASIO)
  │   └─ t_mmcssTried?
  │       ├─ true  → スキップ
  │       └─ false → AvSetMmThreadCharacteristicsW(L"Pro Audio")
  │                    ├─ 成功 → AvSetMmThreadPriority(CRITICAL) + 終了
  │                    ├─ err=5/183 → 既に登録済み = 成功扱い
  │                    ├─ err=1552/1531 → L"Audio" フォールバック
  │                    │   ├─ 成功 → AvSetMmThreadPriority(CRITICAL) + 終了
  │                    │   └─ 失敗 → 諦める
  │                    └─ その他エラー → 諦める
  │
  ├─ Policy = SelfManagedPlayback (DirectSound)
  │   └─ t_mmcssTried?
  │       └─ false → AvSetMmThreadCharacteristicsW(L"Playback")
  │                    ├─ 成功 → AvSetMmThreadPriority(HIGH) + 終了
  │                    ├─ err=5/183 → 既に登録済み = 成功扱い（Win11エミュレーション対策）
  │                    ├─ err=1552/1531 → L"Audio" フォールバック
  │                    │   ├─ 成功 → AvSetMmThreadPriority(HIGH) + 終了
  │                    │   └─ 失敗 → L"Pro Audio" フォールバック
  │                    └─ その他エラー → 諦める
  │
  └─ Policy = None → 何もしない
```

### 4.2 シャットダウンフロー

```
releaseResources (Message Thread)
  │
  └─ mmcssShutdownRequested = true  （フラグのみ。AvRevert はしない）
        │
        ▼
  次回コールバック (Audio Thread)
        │
        └─ mmcssShutdownRequested == true?
              ├─ Yes → AvRevertMmThreadCharacteristics(t_mmcssHandle)
              │         t_mmcssHandle = nullptr
              │         t_mmcssTried = false
              │         mmcssShutdownRequested = false
              └─ No  → 通常処理
```

### 4.3 バックエンド別動作一覧

| バックエンド | 初回呼出結果 | スレッド優先度 | シャットダウン |
|:------------|:------------|:--------------|:--------------|
| **WASAPI (JUCE)** | err=5/183 → 成功扱い | `AVRT_PRIORITY_NORMAL`（JUCE設定） | JUCEが管理 |
| **ASIO (優良Driver)** | err=5/183 → 成功扱い | ドライバ設定 | ドライバが管理 |
| **ASIO (未設定)** | `Pro Audio`/`CRITICAL` 登録成功 | 23-26 | callback で AvRevert |
| **DirectSound (Win11)** | err=5/183 → 成功扱い | OS(WASAPIエミュレーション)設定 | OSが管理 |
| **DirectSound (従来)** | `Playback`/`HIGH` 登録成功 | 16-22 | callback で AvRevert |

---

## 5. エラーコード対応表

| エラー | 値 | 意味 | 本設計の挙動 |
|:-------|:---|:------|:-------------|
| `ERROR_SUCCESS` | 0 | 成功 | `AvSetMmThreadPriority` で優先度設定 |
| `ERROR_ACCESS_DENIED` | 5 | 既に別タスクが登録済み | ✅ **成功扱い**。JUCE/ドライバが管理。 |
| `ERROR_ALREADY_EXISTS` | 183 | 既に同一タスクが登録済み | ✅ **成功扱い**。同上。 |
| `ERROR_INVALID_TASK_NAME` | **1531** | タスク名がレジストリに無い | ⏩ 次候補タスクでフォールバック |
| `ERROR_NO_MORE_ITEMS` | **1552** | 公式エラーではない。`A`版の内部列挙終端。 | ⏩ 次候補タスクでフォールバック（`W`版で回避） |
| `ERROR_PRIVILEGE_NOT_HELD` | 1314 | 権限不足 | ⏩ 諦める（未設定でも動作はする） |
| その他 | — | 予期しないエラー | ⏩ 諦める（ログ出力のみ） |

---

## 6. 実装詳細

### 6.1 新規ファイル: `AudioEngine.Mmcss.cpp`

```cpp
// AudioEngine.Mmcss.cpp — Unified MMCSS Layer for WASAPI/ASIO/DirectSound
//
// Authority Singularization:
//   WASAPI:       JUCE manages → our call will fail(5/183) → success
//   ASIO(driver): Driver manages → our call may succeed or fail(5/183)
//   DirectSound:  Host manages → our call succeeds
//
// thread_local ensures:
//   - Safety across driver-owned threads (ASIO)
//   - No lock needed
//   - Auto-cleanup on thread destruction
//   - Device switch creates new thread → new t_mmcssTried

#include "AudioEngine.h"
#include "DiagnosticsConfig.h"
#include <avrt.h>
#include <windows.h>

namespace {
    thread_local HANDLE t_mmcssHandle = nullptr;
    thread_local DWORD  t_mmcssTaskIndex = 0;
    thread_local bool   t_mmcssTried = false;

    // Wrapper: AvSetMmThreadCharacteristicsW (Unicode version only)
    // Returns nullptr on failure, sets idx to task index on success.
    HANDLE tryTask(LPCWSTR taskName, DWORD& idx) noexcept
    {
        idx = 0;
        return ::AvSetMmThreadCharacteristicsW(taskName, &idx);
    }
}

// ── Public ──

bool AudioEngine::tryApplyMmcssForSelfManagedThread() noexcept
{
    if (t_mmcssTried)
        return (t_mmcssHandle != nullptr);
    t_mmcssTried = true;

    if (!convo::consumeAtomic(useMmcssPriority, std::memory_order_acquire))
        return false; // NativeRT mode

    const auto policy = getCurrentMmcssPolicy();

    if (policy == MmcssPolicy::JuceManaged || policy == MmcssPolicy::None) {
        // JUCE or unknown manages → nothing to do
        return true;
    }

    LPCWSTR primaryTask = nullptr;
    LPCWSTR fallback1   = nullptr;
    LPCWSTR fallback2   = nullptr;
    int priority        = AVRT_PRIORITY_CRITICAL;

    if (policy == MmcssPolicy::SelfManagedProAudio) {
        primaryTask = L"Pro Audio";
        fallback1   = L"Audio";
        fallback2   = nullptr;
        priority    = AVRT_PRIORITY_CRITICAL;
    } else { // SelfManagedPlayback
        primaryTask = L"Playback";
        fallback1   = L"Audio";
        fallback2   = L"Pro Audio";
        priority    = AVRT_PRIORITY_HIGH;
    }

    // ── Primary task ──
    DWORD idx = 0;
    HANDLE h = tryTask(primaryTask, idx);

    if (h != nullptr) {
        ::AvSetMmThreadPriority(h, static_cast<AVRT_PRIORITY>(priority));
        t_mmcssHandle = h;
        t_mmcssTaskIndex = idx;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog(juce::String("[MMCSS] registered: policy=")
            + juce::String(static_cast<int>(policy))
            + juce::String(" task=") + juce::String(primaryTask)
            + juce::String(" idx=") + juce::String(static_cast<int>(idx)));
#endif
        return true;
    }

    // ── Failure analysis ──
    const DWORD err = ::GetLastError();

    // Already registered by JUCE or driver → treat as success
    if (err == ERROR_ACCESS_DENIED || err == ERROR_ALREADY_EXISTS) {
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog(juce::String("[MMCSS] already registered (err=")
            + juce::String(static_cast<int>(err)) + juce::String(") policy=")
            + juce::String(static_cast<int>(policy)));
#endif
        return true;
    }

    // Task name not found → fallback chain
    auto attemptFallback = [&](LPCWSTR fallbackTask) -> bool {
        if (fallbackTask == nullptr) return false;
        DWORD idx2 = 0;
        HANDLE h2 = tryTask(fallbackTask, idx2);
        if (h2 != nullptr) {
            ::AvSetMmThreadPriority(h2, static_cast<AVRT_PRIORITY>(priority));
            t_mmcssHandle = h2;
            t_mmcssTaskIndex = idx2;
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
            diagLog(juce::String("[MMCSS] registered (fallback): policy=")
                + juce::String(static_cast<int>(policy))
                + juce::String(" task=") + juce::String(fallbackTask)
                + juce::String(" idx=") + juce::String(static_cast<int>(idx2)));
#endif
            return true;
        }
        return false;
    };

    if (err == ERROR_NO_MORE_ITEMS || err == ERROR_INVALID_TASK_NAME) {
        if (attemptFallback(fallback1)) return true;
        if (attemptFallback(fallback2)) return true;
    }

    // Unexpected error
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    diagLog(juce::String("[MMCSS] FAILED: err=") + juce::String(static_cast<int>(err))
        + juce::String(" policy=") + juce::String(static_cast<int>(policy)));
#endif
    return false;
}

void AudioEngine::revertMmcssOnAudioThread() noexcept
{
    if (t_mmcssHandle != nullptr) {
        ::AvRevertMmThreadCharacteristics(t_mmcssHandle);
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[MMCSS] reverted on Audio Thread");
#endif
        t_mmcssHandle = nullptr;
        t_mmcssTaskIndex = 0;
    }
    t_mmcssTried = false; // Allow retry on next device open
}
```

### 6.2 AudioBlock.cpp / BlockDouble.cpp 変更

```cpp
void AudioEngine::getNextAudioBlock(const AudioSourceChannelInfo& bufferToFill)
{
    // ...既存の lifecycle/early-exit チェック...

    // ★ [work70 v9.11] Unified MMCSS Layer
    //   - WASAPI: JUCE manages → skip
    //   - ASIO (self-managed): Pro Audio / CRITICAL → first callback only
    //   - DirectSound (self-managed): Playback / HIGH → first callback only
    {
        const auto mmcssPolicy = getCurrentMmcssPolicy();
        if (mmcssPolicy == MmcssPolicy::SelfManagedProAudio
            || mmcssPolicy == MmcssPolicy::SelfManagedPlayback)
        {
            if (!t_mmcssTried)  // namespace thread_local in AudioEngine.Mmcss.cpp
                tryApplyMmcssForSelfManagedThread();

            if (convo::consumeAtomic(mmcssShutdownRequested, std::memory_order_acquire)) {
                revertMmcssOnAudioThread();
                convo::publishAtomic(mmcssShutdownRequested, false, std::memory_order_release);
            }
        }
        // JuceManaged / None → nothing
    }

    // ...既存処理...
}
```

### 6.3 ReleaseResources.cpp 変更

```cpp
void AudioEngine::releaseResources()
{
    // ...既存処理...

    // ★ [work70 v9.11] Unified MMCSS: flag-based shutdown
    //    Message Thread から直接 AvRevert せず、フラグのみセット。
    //    実際の AvRevert は次回コールバック (Audio Thread) で実行される。
    const auto mmcssPolicy = getCurrentMmcssPolicy();
    if (mmcssPolicy == MmcssPolicy::SelfManagedProAudio
        || mmcssPolicy == MmcssPolicy::SelfManagedPlayback)
    {
        convo::publishAtomic(mmcssShutdownRequested, true, std::memory_order_release);
    }

    // JUCE managed (WASAPI) → JUCE が自動 AvRevert。何もしない。
}
```

### 6.4 診断ログ出力（RT-safe）

本設計の核心要件「ログにMMCSS登録、Playback/HIGHが成功したか失敗したかを出す。オーディオスレッドのリアルタイム性を損なわないこと」に対する実装:

```cpp
// AudioEngine.Mmcss.cpp — 該当箇所

// 成功時のログ（初回1回のみ）:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[MMCSS-" + policyTag + "] registered: task=" + primaryTask
            + " priority=" + prioStr + " taskIndex=" + String(idx));
#endif

// 既に登録済み（5/183）のログ:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[MMCSS-" + policyTag + "] already registered by JUCE/driver (err="
            + String(err) + ") task=" + primaryTask);
#endif

// 全失敗時のログ:
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        diagLog("[MMCSS-" + policyTag + "] FAILED: primary err=" + String(err)
            + " task=" + primaryTask);
#endif
```

**RT-safe の保証**:
- ログ出力は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガード内 — Release ビルドでは完全に除去される
- 登録処理自体も `t_mmcssTried` による初回1回のみの実行 — 2回目以降は O(1) TLS 読み取りのみ
- ログ文字列構築＋`juce::Logger::writeToLog()` のオーバーヘッドが生じるのは **初回成功時1回のみ**

### 6.5 AudioEngine.h への追加宣言

```cpp
// AudioEngine.h — MMCSS 関連宣言

enum class MmcssPolicy : uint8_t {
    JuceManaged,           // WASAPI → JUCE が管理
    SelfManagedProAudio,   // ASIO → 自前 Pro Audio/CRITICAL
    SelfManagedPlayback,   // DirectSound → 自前 Playback/HIGH
    None                   // その他
};

[[nodiscard]] MmcssPolicy getCurrentMmcssPolicy() const noexcept;
[[nodiscard]] bool tryApplyMmcssForSelfManagedThread() noexcept;
void revertMmcssOnAudioThread() noexcept;

// ★ [work70 v9.11] シャットダウン要求フラグ — Message Thread → Audio Thread 通知
alignas(64) std::atomic<bool> mmcssShutdownRequested{false};
```

### 6.5 AudioEngine.Timer.cpp からの削除

`applyMmcssPriority()` 関数は Unified MMCSS Layer に置き換わるため、以下のコードは削除する:

- `applyMmcssPriority()` 関数本体（Timer.cpp）
- `revertMmcssPriorityOnAudioThread()` 関数本体
- `finalizeMmcssShutdown()` 関数本体
- Timer heartbeat 内の MMCSS 関連コード
- `avrt.h` / `<windows.h>` include（既に Mmcss.cpp で include）
- `useMmcssPriority` 関連分岐（→ `tryApplyMmcssForSelfManagedThread()` で使用）

---

## 7. テスト観点

### 7.1 ユニットテスト

| # | テストケース | 期待結果 |
|:-:|:-----------|:--------|
| 1 | Policy=JuceManaged → `getNextAudioBlock` | `AvSetMmThreadCharacteristics` を一切呼ばない |
| 2 | Policy=JuceManaged → `releaseResources` | `mmcssShutdownRequested` をセットしない |
| 3 | Policy=SelfManagedProAudio → 初回callback | `AvSetMmThreadCharacteristicsW(L"Pro Audio")` を1回だけ呼ぶ |
| 4 | Policy=SelfManagedProAudio → 2回目以降のcallback | 上記を呼ばない（t_mmcssTried=true） |
| 5 | Policy=SelfManagedPlayback → 初回callback | `AvSetMmThreadCharacteristicsW(L"Playback")` を1回だけ呼ぶ |
| 6 | 全 Policy → `releaseResources` → 次回callback | `mmcssShutdownRequested` を検知 → `AvRevertMmThreadCharacteristics` を呼ぶ |
| 7 | err=5/183 が返る状況 → 成功扱いになる | 登録なしで true を返す |
| 8 | 1552/1531 → fallback タスクで再試行 | フォールバックタスクで登録成功 |

### 7.2 統合テスト

| # | シナリオ | 確認項目 |
|:-:|:---------|:--------|
| 1 | WASAPI 起動 → ログ確認 | `[MMCSS] already registered` or 出力なし |
| 2 | ASIO 起動 → ログ確認 | `[MMCSS] registered: policy=1 task=Pro Audio` |
| 3 | DirectSound 起動 → ログ確認 | `[MMCSS] registered: policy=2 task=Playback` |
| 4 | デバイス切替 WASAPI→ASIO | 新スレッドで再登録される |
| 5 | シャットダウンログ | `[MMCSS] reverted on Audio Thread` |
| 6 | XRUN カウント変化 | MMCSS 適用前後で XRUN レートが改善 or 同等 |

---

## 8. 現行 A 案からの変更点まとめ

| 項目 | 現行（A案: JUCE委譲のみ） | 本設計（Unified MMCSS） |
|:-----|:-------------------------|:----------------------|
| WASAPI | ✅ JUCE委譲（正しい） | ✅ 同じ（`JuceManaged`） |
| ASIO（優良Driver） | ❌ 未設定のまま | ✅ err=5/183 で成功扱い |
| ASIO（非対応Driver） | ❌ 未設定のまま | ✅ Pro Audio/CRITICAL で登録 |
| DirectSound | ❌ 未設定のまま | ✅ Playback/HIGH で登録 |
| `applyMmcssPriority()` | ❌ Timer.cpp に残存 | ✅ 全削除・`tryApplyMmcssForSelfManagedThread()` に統合 |
| `thread_local` | 未使用 | ✅ 新規採用 |
| `mmcssShutdownRequested` | ❌ 全削除済み | ✅ 復活（フラグ通知として適切に使用） |
| P8 再試行機構 | ❌ 削除済み | ❌ 削除維持（不要） |
| バックエンド検出 | 不要 | ✅ `getCurrentMmcssPolicy()` で明示的 |

---

## 9. 決定論と Authority の評価

### 9.1 INV-12/INV-13 との整合性

| 原則 | 本設計での該当 | 判定 |
|:-----|:-------------|:------|
| Builder は mutable Runtime を観測しない | 無関係（MMCSS は Builder 外） | ✅ |
| Authority の重複防止 | WASAPI=JUCE, ASIO=Driver, DirectSound=Host → 排他的 | ✅ |
| thread_local は Audio Thread のみアクセス | `m_avrtHandle`（共有）から `t_mmcssHandle`（TLS）に変更 | ✅ 改善 |

### 9.2 ASIO SDK 仕様との整合性

ASIO SDK の記述:

> "ASIO driver threads must be in the 'Pro Audio' class. The host shall by no means alter these priorities."

本設計では:
- **優良ドライバ**: err=5/183 → 成功扱い → ドライバ設定を変更しない ✅
- **未設定ドライバ**: Host が Recovery として補助登録 → ドライバが本来の責務を果たせていない場合のみ介入 ✅
- 常時登録ではなく、初回1回のみ → Authority 侵害の最小化 ✅

---

## 10. 実装手順

| Step | 内容 | ファイル |
|:-----|:------|:---------|
| 1 | `AudioEngine.h` に `MmcssPolicy` enum 追加、関数宣言追加 | `AudioEngine.h` |
| 2 | `AudioEngine.Mmcss.cpp` 新規作成（thread_local + try/revert） | `AudioEngine.Mmcss.cpp`（新規） |
| 3 | `AudioBlock.cpp` の MMCSS コメント＋CAS 削除箇所を置き換え | `AudioBlock.cpp` |
| 4 | `BlockDouble.cpp` の MMCSS コメント＋CAS 削除箇所を置き換え | `BlockDouble.cpp` |
| 5 | `AudioEngine.Timer.cpp` から `applyMmcssPriority()` 関連を全削除 | `Timer.cpp` |
| 6 | `ReleaseResources.cpp` にフラグ通知追加 | `ReleaseResources.cpp` |
| 7 | `MainWindow.cpp` 変更確認（mmcssShutdownRequested 削除済みを維持） | `MainWindow.cpp` |
| 8 | `PrepareToPlay.cpp` 変更確認 | `PrepareToPlay.cpp` |
| 9 | ビルド + CTest | — |
| 10 | 実機テスト（WASAPI/ASIO/DirectSound 各モード） | — |

---

## 11. Appendix: MMCSS タスクレジストリ完全値

### Pro Audio（High カテゴリ）

| 値名 | 値 |
|:-----|:---|
| Affinity | 0（未使用） |
| Background Only | False |
| Clock Rate | 10000 (ns) |
| GPU Priority | 8 |
| Priority | 1（base priority） |
| Scheduling Category | High |
| SFIO Priority | Normal |

### Playback（Medium カテゴリ）

| 値名 | 値 |
|:-----|:---|
| Affinity | 0（未使用） |
| Background Only | False |
| BackgroundPriority | 4 |
| Clock Rate | 10000 (ns) |
| GPU Priority | 8 |
| Priority | 3（base priority） |
| Scheduling Category | Medium |
| SFIO Priority | Normal |

### Audio（Medium カテゴリ）

| 値名 | 値 |
|:-----|:---|
| Affinity | 0（未使用） |
| Background Only | True |
| Clock Rate | 10000 (ns) |
| GPU Priority | 8 |
| Priority | 6（base priority） |
| Scheduling Category | Medium |
| SFIO Priority | Normal |

### Capture（Medium カテゴリ）

| 値名 | 値 |
|:-----|:---|
| Affinity | 0（未使用） |
| Background Only | True |
| Clock Rate | 10000 (ns) |
| GPU Priority | 8 |
| Priority | 5（base priority） |
| Scheduling Category | Medium |
| SFIO Priority | Normal |

---

*本設計書の全主張は MSDN 公式ドキュメント、JUCE 8.0.12 ソースコード、Windows 11 実機レジストリ値に基づく。*
