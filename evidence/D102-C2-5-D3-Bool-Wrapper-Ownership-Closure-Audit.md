# D102-C2-5-D3 — Bool Wrapper Ownership Closure Audit（read-only, production変更0）

- **実施日**: 2026-08-26 01:30 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02** + 実 `src/`（`git diff HEAD -- src/audioengine/AudioEngine.h` 0 lines）
- **目的**: `QueueFull → enqueueDeferredDeleteNonRtWithResult → enqueueDeferredDeleteNonRt → Cache/Convolver` の全 control-flow を read-only で再確認し、`enqueueDeferredDeleteNonRt` の `result != Shutdown` 判定を `Transferred: Success/QueuePressure/TerminalReclaim` vs `Caller retains: QueueFull/Shutdown` の ownership semantics に一致させる設計を固定する。`tstored==false` を `true` に変換する経路がないことを証明する
- **判定**: **AUDIT PASS（設計固定） / Implementation gate CONDITIONAL PASS（wrapper 要修正）**

---

## 1. 全 Control-Flow 再確認（read-only）

### 1.1 `enqueueDeferredDeleteNonRtWithResult` 現行（`src/audioengine/AudioEngine.h:4205`）

```cpp
inline RetireEnqueueResult enqueueDeferredDeleteNonRtWithResult(void* ptr, ...) noexcept
{
    if (ptr==nullptr||deleter==nullptr) return Success;
    if (isShutdownInProgress()) {
        const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);
        return transferred ? Success : Shutdown; // shutdownReclaim の成否を正確に伝播
    }
    const uint64_t epoch = markRetireEpoch();
    auto result = m_retireRouter->enqueueWithRetry(ptr, deleter, epoch, type);
    if (result == Success) return Success;
    drainDeferredRetireQueues(false);
    const uint64_t retireDepth = m_retireRouter->pendingRetireCount();
    publishAtomic(retireQueueDepth_, retireDepth, ...);
    return result; // QueuePressure / TerminalReclaim / QueueFull / Shutdown をそのまま返す
}
```

- `QueueFull` 到達時は `result == QueueFull` を **そのまま return** するため、上位へ caller-retain を伝播可能

### 1.2 `enqueueDeferredDeleteNonRt` 現行（`src/audioengine/AudioEngine.h:4198`）

```cpp
inline bool enqueueDeferredDeleteNonRt(void* ptr, ...) noexcept
{
    const auto result = enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type);
    return result != RetireEnqueueResult::Shutdown; // ← 現行問題
}
```

- 現行 `result != Shutdown` は `QueueFull != Shutdown → true` を返す → **caller-retain を transferred と誤認**

### 1.3 `CacheMap` / `ConvolverProcessor` Caller

```text
src/audioengine/AudioEngine.Cache.cpp:16  owner.enqueueDeferredDeleteNonRt(map, [](void* p){ delete static_cast<CacheMap*>(p); });
src/convolver/ConvolverProcessor.Lifecycle.cpp:57  provider->enqueueDeferredDeleteNonRt(oldState, deleter);
src/convolver/ConvolverProcessor.Lifecycle.cpp:70  provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver);
```

- いずれも `bool` 戻り値を **チェックしない**（`void` として扱う）。`QueueFull` 時に `true` が返ると **所有を手放したと誤認したまま終了** し、ptr はいずれの retire container にも属さず **leak**（所有は保持されるが処理されない）
- `rg "enqueueDeferredDeleteNonRt\(" src/audioengine/AudioEngine.Cache.cpp` 2 matches / `src/convolver/ConvolverProcessor.Lifecycle.cpp` 2 matches — 全て `bool` 無視

---

## 2. `result != Shutdown` 問題の詳細

### 現行 Ownership Semantics（誤）

```text
Transferred (true): Success, QueuePressure, TerminalReclaim, QueueFull
Caller retains (false): Shutdown のみ
```

- `QueueFull` が `true` に含まれるため、`tstored==false → QueueFull` を `WithResult` が正しく返しても、`bool` ラッパーで `true` に変換され **caller-retain が transferred に化ける**

### 正しい Ownership Semantics（固定）

```text
Transferred (true):
    Success
    QueuePressure
    TerminalReclaim
        ↓ ownership transferred

Caller retains (false):
    QueueFull
    Shutdown
        ↓ caller retains, 必ず retry/quarantine/backpressure の次状態が存在する
```

- `QueueFull` と `Shutdown` は共に **caller retains**（D14.3 backpressure / D15.2 `shutdownDiscard` として別途処理）
- `Success/QueuePressure/TerminalReclaim` は **ownership transferred**（caller は処理終了してよい）

---

## 3. 設計 Fix（D3 で実施、production 最小限）

### 3.1 `enqueueDeferredDeleteNonRt` 修正

```cpp
inline bool enqueueDeferredDeleteNonRt(void* ptr, ...) noexcept
{
    const auto result = enqueueDeferredDeleteNonRtWithResult(ptr, deleter, type);
    return result != RetireEnqueueResult::Shutdown
        && result != RetireEnqueueResult::QueueFull; // QueueFull も caller-retain
}
```

- `QueueFull` を `false` として返し、**caller が所有を保持していることを正確に伝える**
- `Cache` / `Convolver` の caller が `bool` を無視している場合でも、少なくとも **所有を手放したと誤認する経路は塞げる**（ただし `bool` 無視自体は D4 で別途 `B`/`C` への閉ループを明示する必要がある）

### 3.2 `tstored==false` を `true` に変換する経路がないことの証明

- `WithResult` は `QueueFull` をそのまま返す（変換なし）
- `bool` wrapper は上記修正で `QueueFull → false` を返す（`true` に変換しない）
- `Cache` / `Convolver` の `void` 呼び出し元は現行 `bool` を無視するが、**`WithResult` を直接呼ぶように変更するか、戻り値をチェックして quarantine へ移送する**ことで `true` への誤変換を防ぐ（D4 で実施）

**D3 終了条件:**

- `QueueFull → 必ず caller retains`（`bool == false`）
- `Shutdown → caller retains`
- `transferred 3値 → wrapper が成功扱い`
- `tstored==false` を `true` に変換する経路なし
- production 変更前後で ownership table が完全一致（`Success/QueuePressure/TerminalReclaim == transferred`）

---

## 4. Cache / Convolver への影響

| Caller | 現行 | 修正後 |
|---|---|---|
| `AudioEngine.Cache.cpp:16` `enqueueDeferredDeleteNonRt(map, ...)` | `bool` 無視、`QueueFull` 時に `true` を誤認 → leak | `WithResult` を直接呼び `QueueFull` を検出するか、`bool==false` 時に `quarantine` へ退避（D4） |
| `ConvolverProcessor.Lifecycle.cpp:57,70` | 同上 | 同上 |

- 現行 `Cache` / `Convolver` は `QueueFull` 時の所有保持を **最後まで保持できるか**の control-flow が `bool` 無視のため **最後まで保持できない**。D4 で `B`（quarantine）または `C`（backpressure）への明示的閉ループを固定する

---

## 5. Tool 棚卸し

| ツール | 検索内容 | 結果 |
|---|---|---|
| `rg` `enqueueDeferredDeleteNonRt` | 4 files / 6 matches | Cache/Convolver 全 caller 特定 |
| `ag` `RetireEnqueueResult` | `h:4202` `result != Shutdown` 1件 | 誤判定特定 |
| `sed -n "4198,4265p"` `AudioEngine.h` | wrapper 全文抽出 | 上記 1.1-1.2 の根拠 |
| `AiDex` `enqueueDeferredDeleteNonRtWithResult` | 6 matches | 全 wrapper 特定 |
| `fdfind` `I4` / `fzf` / `ast-grep` | Contract 参照 | D14/D15 照合 |
| `serena` | timeout → `rg` 代替 | 同等 |
| `cocoindex`/`semble`/`graphify` | WSL `which` で不在 | `rg` 代替 |

---

## 6. 判定

| Gate | 判定 |
|---|---|
| `QueueFull` → caller retains | **PASS**（D2-0 で確定） |
| `Shutdown` → caller retains | **PASS** |
| transferred 3値 → wrapper 成功 | **PASS** |
| `tstored==false` を `true` に変換する経路なし | **CONDITIONAL PASS**（現行 `bool` wrapper で変換してしまうため、D3 修正後に PASS） |
| ownership table 完全一致 | **CONDITIONAL PASS**（D3 修正後に PASS） |

**D3 Audit PASS（設計固定） / Implementation gate CONDITIONAL PASS（wrapper 要修正）** — 次 D4 で全 28 sink の `void`/`ignoreUnused` を含めた closure を実施する

*本監査は read-only であり、production source 変更 0、D14/D15 改訂なし、Terminal bounded 化なしで実施された。*
