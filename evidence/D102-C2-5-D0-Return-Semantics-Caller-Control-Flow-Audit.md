# D102-C2-5-D0 — Return Semantics / 14 Caller Control-Flow 再確認（read-only, コード変更なし）

- **実施日**: 2026-08-26 00:45 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines）
- **目的**: `RetireEnqueueResult::QueueFull` を `tstored==false` 時の caller-retain 値として採用できるかを、production source 上で D0 として確定する。`enqueueDeferredDeleteNonRtWithResult` の QueueFull 処理、14 caller の `QueueFull` 到達時 control-flow、`bool` caller の `false=caller retains` 成立を静的に検証する
- **判定**: **CONDITIONAL PASS** — `QueueFull` は概念的には caller-retain と整合するが、**現行 `AudioEngine::enqueueDeferredDeleteNonRt` の `bool` ラッパーが `QueueFull` を `true`（transfer済み）と誤認する**ため、単純な `return QueueFull` だけでは 3 caller で D (ambiguous) が残る。`QueueFull` 採用にはラッパー側の `bool` 判定拡張が併せて必要

---

## 1. `RetireEnqueueResult` 意味確認（D0-A）

### 定義（`src/audioengine/ISRAuthorityClass.h:28`）

```cpp
enum class RetireEnqueueResult : uint8_t {
    Success = 0,         // D が所有（lock-free 成功）
    QueuePressure,       // Q/E が所有（quarantine store が所有、D 満杯の背圧通知）
    QueueFull,           // 旧: D 満杯の失敗（現行 ISRRetireRouter::enqueueRetire では未使用、QueuePressure に統合）
    Shutdown,            // shutdown 中の early return（caller が所有を保持、shutdownReclaim が失敗）
    TerminalReclaim      // T が所有（growable store が所有、または epoch-safe 同期破棄で所有終了）
};
```

### 各値の現行意味論（`src/audioengine/ISRRetireRouter.cpp:239` `enqueueRetire`）

| 値 | 意味 | Ownership |
|---|---|---|
| `Success` | `provider_->enqueueRetireTyped` 成功、または tryReclaim 後の再試行成功 | **Transferred** (D owns) |
| `QueuePressure` | `enqueueRetireTyped` が満杯で `m_overflowCount_++` 後に返す。`enqueueWithRetry` では Q/E への移送成功時に `QueuePressure` を返す（`ISRRetireRouter.cpp:346,356` `Q/E owns ptr`） | **Transferred** (Q/E owns) |
| `QueueFull` | 現行 `ISRRetireRouter::enqueueRetire` では **返さない**（`QueuePressure` に統合）。`SnapshotCoordinator::enqueueWithRetry` (旧) では `false` に対応 | **Caller retains**（historical） |
| `Shutdown` | `enqueueDeferredDeleteNonRtWithResult:4205` の shutdown early return で `shutdownReclaim` が `false` の場合のみ | **Caller retains** |
| `TerminalReclaim` | `enqueueWithRetry:368` の T 到達時、または `terminalReclaim` の epoch-safe 同期破棄時 | **Transferred** (T owns または破棄完了) |

**結論:** `QueueFull` は現行 `ISRRetireRouter::enqueueRetire` で未使用のため、**新たに `tstored==false` 用に割り当てる自由がある**。概念的には `QueuePressure`（transfer）と対照的に **caller-retain** として使用できるが、**各 caller の既存分岐が `QueueFull` をどう扱うかが鍵**。

---

## 2. `enqueueDeferredDeleteNonRtWithResult()` の QueueFull 処理（D0-B）

### 現行コード（`src/audioengine/AudioEngine.h:4198`）

```cpp
inline bool enqueueDeferredDeleteNonRt(void* ptr, void (*deleter)(void*),
                                       DeletionEntryType type) noexcept
{
    const auto result = enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type);
    return result != RetireEnqueueResult::Shutdown; // Shutdown のみ false
}

inline RetireEnqueueResult enqueueDeferredDeleteNonRtWithResult(void* ptr, ...) noexcept
{
    if (ptr==nullptr||deleter==nullptr) return Success;
    if (isShutdownInProgress()) {
        const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
        return transferred ? Success : Shutdown;
    }
    const uint64_t epoch = markRetireEpoch();
    auto result = m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, type);
    if (result == Success) return Success;
    drainDeferredRetireQueues(false);
    const uint64_t retireDepth = m_retireRouter->pendingRetireCount();
    publishAtomic(retireQueueDepth_, retireDepth, ...);
    return result; // ← QueuePressure/TerminalReclaim/QueueFull/Shutdown をそのまま返す
}
```

### `QueueFull` 到達時の挙動（仮に `tstored==false` で `QueueFull` を返した場合）

```text
enqueueWithRetry() → QueueFull
  ↓
enqueueDeferredDeleteNonRtWithResult() → return QueueFull
  ↓
enqueueDeferredDeleteNonRt() → result != Shutdown → true
```

- `enqueueDeferredDeleteNonRt` は `Shutdown` 以外を **全て `true`（transfer済み）とみなす**。
- したがって `QueueFull` を返しても **`bool` ラッパーは `true` を返し、caller は所有を手放したと誤認する（D）**。
- `drainDeferredRetireQueues(false)` は呼ばれるが、これは `QueueFull` を再試行するものではなく、**所有を移送するものではない**。
- **判定: 現行 wrapper では `QueueFull` は caller-retain として機能しない — 要修正**

### 必要な修正（D0 で確定、D で実装）

```cpp
// 現行: return result != Shutdown;
return result != RetireEnqueueResult::Shutdown
    && result != RetireEnqueueResult::QueueFull; // QueueFull も caller-retain として false
```

または `QueueFull` を返さず `Shutdown` として扱うか、`bool` ではなく `RetireEnqueueResult` を caller が直接見るようにする。**D14.3 の backpressure として扱うなら `QueueFull == caller retains` が自然**。

---

## 3. 14 Caller の `QueueFull` 到達時 Control-Flow（D0-C）

### 3.1 Caller 分類（`bool` vs `RetireEnqueueResult`）

| # | Caller | 戻り値型 | 現行 `QueueFull` 時の挙動（仮） | 正しい挙動 | 問題 |
|---|---|---|---|---|---|
| 1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | `RetireEnqueueResult` | `QueueFull` をそのまま return | **caller-retain** として `QueueFull` を返すことは正しい（caller が `QueueFull` を見て backpressure できる） | **PASS**（ただし `bool` wrapper が誤認） |
| 2 | `AudioEngine::enqueueDeferredDeleteNonRt` (bool) | `bool` | `QueueFull != Shutdown` → `true` | `false`（caller retains） | **D** |
| 3 | `SnapshotCoordinator::switchImmediate` (`h:94`) | `RetireEnqueueResult` | `result == Shutdown` のみ分岐 | `QueueFull` も `Shutdown` と同様に caller-retain として扱う必要あり | **要確認** |
| 4 | `SnapshotCoordinator::discardSnapshot` (`h:107` `if (!enqueueWithRetry)`) | `bool` | `bool==false` → `quarantineRetireSink` へ（caller retains → quarantine） | `QueueFull → false` なら正しい（quarantine へ） | **PASS**（`QueueFull` を `false` にすれば） |
| 5-6 | `SnapshotCoordinator::retireSnapshot` (`h:176,181`) | `bool` | 同上 | 同上 | **PASS** |
| 7-8 | `SnapshotCoordinator.cpp:57,114` (`startFade/completeFade`) | `RetireEnqueueResult` | 未使用（`result` 未チェック） | 所有を保持したまま retry しない → **leak** | **D**（戻り値無視） |
| 9-10 | `DSPLifetimeManager.cpp:49,96` | `RetireEnqueueResult` | `ignoreUnused(result)` | 同上、戻り値無視のため `QueueFull` でも caller は何もしない → **所有は保持されるが処理されない** | **D**（戻り値無視） |
| 11 | `EQProcessor.Core.cpp:61` | `RetireEnqueueResult` | `result==Success\|\|QueuePressure\|\|TerminalReclaim` が `true` | `QueueFull` は `false` → caller-retain として正しい | **PASS** |
| 12 | `ISRRuntimePublicationCoordinator.cpp:162` | `RetireEnqueueResult` | `result != Success` で return | `QueueFull != Success` → caller は失敗として扱う | **PASS** |
| 13 | `RetireGraceSemanticsTests.cpp:679` | `RetireEnqueueResult` | Test が `QueueFull` を検出 | Test が `QueueFull` を failure として扱う | **PASS** |
| 14 | `AudioEngine.Cache.cpp` / `ConvolverProcessor` (indirect via `enqueueDeferredDeleteNonRt`) | `void`/`bool` | `bool==true` | 同 #2 の問題 | **D** |

### 3.2 Summary

| 分類 | 件数 | `QueueFull` → caller-retain として正しく機能 |
|---|---|---|
| `RetireEnqueueResult` を直接見る caller (1,3,11,12,13) | 5 | **PASS**（`QueueFull != TerminalReclaim` として区別できる） |
| `bool` を見る caller (2,4-6) | 4 | **PASS**（`false` として扱える、ただし AudioEngine wrapper 要修正） |
| 戻り値を無視する caller (7-10,14) | 5 | **D**（`QueueFull` を返しても caller は何もしない → 所有は保持されるが **最終的に処理されない**） |

**結論:** `return QueueFull` だけでは **戻り値を無視する 5 caller で D が残る**。しかしこれら caller についても、`QueueFull` を返すことで **少なくとも `TerminalReclaim` と誤認して所有を手放すことは防げる**（D よりはマシ）。完全に `ownership loss = 0` を保証するには、**戻り値を無視する caller についても、所有を保持したまま適切に処理する経路（quarantine または retry）が必要**だが、現行でも `enqueueWithRetry` 内部で Q/E への移送を試みており、Terminal failure 時のみ `QueueFull` を返すため、**所有は caller に残り、caller が leak させなければ UAF は生じない**。caller が戻り値を無視する場合、**所有は保持されるが処理されない**状態となり、**leak** となる可能性がある。

### 3.3 `bool` caller の `false = caller retains` 成立確認

- `SnapshotCoordinator::enqueueWithRetry` (`h:151`) は `provider.enqueueRetire` が 2回失敗したら `false` を返し、呼び出し元 `retireCurrentAndTarget` は `if (!enqueueWithRetry) quarantineRetireSink` により **caller-retain → quarantine へ退避**する。`QueueFull → false` はこの経路と整合する。
- `AudioEngine::enqueueDeferredDeleteNonRt` の `bool` は現行 `Shutdown` のみ `false` だが、`QueueFull` も `false` とすべき（上記 D0-B の修正）。
- `EQProcessor` の `bool` 的判定 (`Success||QueuePressure||TerminalReclaim`) では `QueueFull` は `false` → caller-retain として正しい。

---

## 4. D102-C2-5-D0 判定

| Gate | 判定 | 理由 |
|---|---|---|
| D0-A `QueueFull` 意味 | **CONDITIONAL PASS** | `QueueFull` は現行未使用で caller-retain として割り当て可能。`QueuePressure` と対照的に transfer ではない |
| D0-B `enqueueDeferredDeleteNonRtWithResult` | **CONDITIONAL PASS（要修正）** | 現行 `bool` wrapper が `QueueFull` を `true` と誤認するため、`QueueFull` 採用には `result != Shutdown && result != QueueFull` への修正が必須 |
| D0-C 14 caller control-flow | **CONDITIONAL PASS（要認識）** | 戻り値を見る 9 caller は `QueueFull` で正しく caller-retain になる。戻り値を無視する 5 caller は所有を保持するが処理されないため **leak 可能性** が残るが、少なくとも `TerminalReclaim` 誤認による **UAF/leak よりは安全**。完全な保証には caller 側の `quarantine` または retry が必要だが、現行でも `QueueFull` は `TerminalReclaim` よりはまし |
| D0 全体 | **CONDITIONAL GO** | `QueueFull` が caller-retain semantics と整合することは確認できた。ただし `AudioEngine::enqueueDeferredDeleteNonRt` の `bool` 判定拡張が **必須前提**。5 caller の戻り値無視は次工程で `quarantine` 経路の追加検討を推奨 |

### Implementation 前提条件（D で実施）

```text
1. ISRRetireRouter::enqueueWithRetry の terminal fallback を
     if (!tstored) return QueueFull;
   に変更（jassert(tstored) 併記）

2. AudioEngine::enqueueDeferredDeleteNonRt の bool 判定を
     return result != Shutdown && result != QueueFull;
   に拡張（QueueFull も caller-retain として false）

3. 戻り値を無視する 5 caller については、所有を保持したまま
   leak しないことを次工程の T4/T6 test で検証し、必要に応じて
   quarantine への退避を追加検討
```

*本監査は read-only であり、production source 変更 0、D14/D15 改訂なし、Terminal bounded 化なしで実施された。*
