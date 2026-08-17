# 15-P-4-0 — Terminal Reclaim 適用可能性監査

## ステータス
- **作成日**: 2026/08/18
- **実装前監査**: コード変更なし — TerminalReclaim / shutdownReclaim / drainAllNonRt の contract をコードレベルで確認
- **関連実装**: 15-P-CROSS-IMPLEMENTATION-1 (drainAllNonRt 実装済み ✅)

---

## A. TerminalReclaim の実際の contract

### A.1. return value は常時 `true` か

**結論: ✅ `terminalReclaim()` は常に `true` を返す。**

`terminalReclaim()` (ISRRetireRouter.cpp:230-306) の実装:

```cpp
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    // epoch safe かつ Non-RT なら deleter 即時実行 → return true
    // epoch unsafe または RT なら TerminalReclaimAuthority::store() → return true
    return m_terminalReclaim.store(...);  // store() は常に true を返す
}
```

`TerminalReclaimAuthority::store()` (ISRRetireRouter.cpp:27-45):
```cpp
bool TerminalReclaimAuthority::store(...) noexcept {
    // growable std::vector → 常に push_back 成功
    entries_.push_back(Entry{...});
    return true;  // ★ P-4: growable store — ALWAYS accepts
}
```

**理由**: `entries_` は `std::vector`（growable）。`push_back` はメモリ確保失敗時に `std::bad_alloc` を投げるが、`noexcept` コンテキストでは `std::terminate` になるため、`store()` は論理的には常に `true` を返す。OOM は terminate なので `false` にはならない。

### A.2. growable であること

**結論: ✅ `TerminalReclaimAuthority::entries_` は `std::vector<Entry>`（growable）。**

```cpp
// ISRRetireRouter.h:63-67
class TerminalReclaimAuthority {
    ...
    std::vector<Entry> entries_;  // ★ P-4: Growable store (Non-RT only)
};
```

`RetireQuarantineStore` は固定配列 (`entries_[kMaxQuarantinedEntries]`) だが、`TerminalReclaimAuthority` は `std::vector` であるため **capacity exhaustion は発生しない**（OOM は terminate に対応）。

### A.3. resident counter の増減

**結論: ✅ `residentCount()` は `entries_.size()` を返す。**

```cpp
// ISRRetireRouter.cpp:116
std::size_t TerminalReclaimAuthority::residentCount() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return entries_.size();
}
```

- `store()`: `entries_.size()` → +1 (resident 増)
- `drain()`: epoch-safe entries を `pending` ベクターに抽出 → `residentCount()` 減
- `drainAll()`: `pending.swap(entries_)` で全件取り出し → `residentCount()` は 0 に

### A.4. deleter が実行されるタイミング

**結論: ✅ タイミングは 2 モードある。**

1. **`terminalReclaim()` 呼出し時 (epoch safe + Non-RT)**: `deleter(ptr)` を **即座に** 実行 (ISRRetireRouter.cpp:264-267)
2. **`drain()` 呼出し時 (epoch-gated)**: epoch safe な entries を抽出し、`drain()` 内で deleter 実行 (ISRRetireRouter.cpp:50-65)
3. **`drainAll()` 呼出し時 (shutdown)**: `pending.swap(entries_)` 後、**即座に** deleter 実行 (ISRRetireRouter.cpp:74-92)

**重要**: shutdownReclaim → terminalReclaim のパスでは、shutdown 中は Audio Thread 停止済みなので `isAudioThread() == false` → epoch safe なら即座に deleter が実行される。

### A.5. `TerminalReclaim` が返る条件

**結論: ✅ `RetireEnqueueResult::TerminalReclaim` は `enqueueWithRetry()` Stage 5 でのみ返る。**

```cpp
// ISRRetireRouter.cpp:330-336 (enqueueWithRetry)
const bool tstored = terminalReclaim(ptr, deleter, epoch, type, "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ★ P-4: 常に true（growable store）
return RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
```

**呼び出し条件**: Stage 1 (D) + Stage 2 (retry) + Stage 3 (Q) + Stage 4 (EmergencyQ) **すべてで `quarantine()` が `false` を返した**時のみ到達する。これは **D+Q+E 全滿** の場合のみ。

実装上: `RetireQuarantineStore::quarantine()` は固定配列が満たった時のみ `false` を返す。

### A.6. shutdown 経路と通常経路の差

| 経路 | `enqueueDeferredDeleteNonRtWithResult` | `terminalReclaim` 呼出し | epoch | `isAudioThread()` | deleter 実行タイミング |
|------|---------------------|--------------------------|-------|-------------------|----------------------|
| **通常 (Non-Shutdown)** | `enqueueWithRetry()` → Stages D→Q→E→Terminal | Stage 5 only | currentEpoch() | false (Non-RT) | epoch safe なら即時 / unsafe なら保留 |
| **Shutdown** | `isShutdownInProgress()` → `shutdownReclaim()` → `terminalReclaim()` | 即座に | `markRetireEpoch()` (advanced) | false (Audio Thread 停止済み) | **常に即時** (epoch safe が保証される) |

**Shutdown 経路の key difference**: `shutdownReclaim()` は `enqueueWithRetry()` を経由**せず**、直接 `terminalReclaim()` を呼ぶ。そのため、**Q→E→Terminal の段階的退避をスキップ** し、**即座に TerminalReclaimAuthority へ移転** する。

---

## B. `shutdownReclaim()` → `terminalReclaim()` の contract

### B.1. shutdownReclaim() の実装

```cpp
// ISRRetireRouter.cpp:344-360
bool ISRRetireRouter::shutdownReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;  // no-op は成功扱い
    // TerminalReclaimAuthority へ移送（epoch-gated destruction）
    return terminalReclaim(ptr, deleter, epoch, type, "shutdownReclaim");
}
```

### B.2. `RetireEnqueueResult::Shutdown` が生成されないことの証明

`enqueueDeferredDeleteNonRtWithResult()` (AudioEngine.h:4200) は以下の2つの path を持つ:

1. **`isShutdownInProgress() == true`** → `shutdownReclaim()` を呼ぶ
   - `shutdownReclaim()` → `terminalReclaim()` → `store()` または deleter 即時実行
   - **戻り値**: `terminalReclaim()` は `bool` を返し、`enqueueDeferredDeleteNonRtWithResult` は `transferred ? Success : Shutdown` で返す
   - **しかし**: `terminalReclaim()` は常に `true` を返すため → **常に `Success`**

2. **`isShutdownInProgress() == false`** → `enqueueWithRetry()` を呼ぶ
   - `enqueueWithRetry()` (ISRRetireRouter.cpp:283) の return path:
     - Stage 1: `enqueueRetire()` → `Success` or `QueuePressure`
     - Stage 2: retry loop → `enqueueRetire()` → `Success` or `QueuePressure`
     - Stage 3-5: `QueuePressure`/`QueueFull` → Q → E → Terminal → `QueuePressure` or `TerminalReclaim`
     - Final: `return result;` — しかし `result` は `Success` or `QueuePressure` or `TerminalReclaim`

**`enqueueRetire()` の return path** (ISRRetireRouter.cpp:210-244):
- `ptr==nullptr || deleter==nullptr` → `Success`
- `provider_->enqueueRetireTyped()` success → `Success`
- retry 後 success → `Success`
- retry exhausted → `QueuePressure`

→ **`enqueueRetire()` は `Success` または `QueuePressure` のみ返す。`Shutdown` は含まれない。**

**`enqueueWithRetry()` の final `return result;`** (ISRRetireRouter.cpp:338):
```cpp
// Stage 3-5 は `if (result == QueuePressure || result == QueueFull)` で guard されている。
// result が QueuePressure または QueueFull でない場合、この if ブロックをスキップする。
// しかし、enqueueRetire() が返すのは Success または QueuePressure のみなので、
// result が QueueFull になることはない（enqueueRetire は QueueFull を返さない）。
// そのため、result == QueuePressure の場合のみ Stage 3-5 に到達する。
// result == Success の場合は Stage 1 または Stage 2 で return 済み。
// → final `return result;` は実質的に到達不可能（dead code）。
```

### B.1: `RetireEnqueueResult::Shutdown` は生成されない

**結論: ✅ `enqueueDeferredDeleteNonRtWithResult()` は決して `RetireEnqueueResult::Shutdown` を返さない。**

| path | `enqueueWithRetry` return | shutdownReclaim return | `enqueueDeferredDeleteNonRtWithResult` final return |
|------|--------------------------|------------------------|---------------------------------------------------|
| Non-shutdown | `Success` / `QueuePressure` / `TerminalReclaim` | N/A | `Success` / `QueuePressure` / `TerminalReclaim` (passthrough) |
| Shutdown | N/A | `true` (always) | `Success` (transferred=true → Success) |

**`Shutdown` enum は存在するが、この call path からは生成されない** — `enqueueRetire()` が `Shutdown` を返さず、`shutdownReclaim()` が `Shutdown` を返さないため。

**`enqueueDeferredDeleteNonRt()` (AudioEngine.h:4187)** は `result != Shutdown` で判定しているため、常に `true` が得られる。

---

## C. Terminal resident と shutdown completion の関係

### C.1. drainAllNonRt → Terminal resident 増加の分析

```
drainAllNonRt()                          // OwnerChannel drain
  → callback: enqueueDeferredDeleteNonRtWithResult(raw, ...)
    → isShutdownInProgress() == true
      → shutdownReclaim()               // ISRRetireRouter.cpp:344
        → terminalReclaim()            // ISRRetireRouter.cpp:230
          → if epoch safe + Non-RT: deleter実行 (resident 0)
          → if epoch unsafe: store() → resident +1
```

**shutdown 時の epoch 安全性**: `finalizeShutdown()` の時点で `advanceRetireEpoch()` が完了し、`activeReaderCount() == 0` が保証されている。`minReaderEpoch()` は現在の epoch に等しいため、`isOlder(entry.epoch, minReaderEpoch)` は `epoch < minReaderEpoch` となり、**既に epoch が進んでいる** world の retire epoch は安全である。

→ **drainAllNonRt の callback 内の `enqueueDeferredDeleteNonRtWithResult` は `shutdownReclaim → terminalReclaim` を経て、epoch safe なので deleter が即座に実行される**。

### C.2. Terminal resident が 0 になる保証

Shutdown sequence (ReleaseResources.cpp):

```
line 521: finalizeShutdown(timedOut)
  ↓
line 536: drainAllNonRt() → drainAllNonRt callback → enqueueDeferredDeleteNonRtWithResult
  → shutdownReclaim → terminalReclaim → epoch safe → deleter 即時実行
  → Terminal resident は増えない（即座に破棄される）
  ↓
line 455: waitForDrain(2000, 2)  [※ この時点では既に drainAllNonRt 完了済み]
  → isFullyDrained() が terminalReclaimResident == 0 をチェック
  → すでに 0 なので即座に pass
```

**`isFullyDrained()` (AudioEngine.Threading.cpp:138-139)**:
```cpp
const auto terminalReclaimResident = (m_retireRouter != nullptr)
    ? static_cast<std::uint64_t>(m_retireRouter->terminalReclaimResidentCount()) : 0u;
// ...
return ... && terminalReclaimResident == 0 && ...;
```

**矛盾なし**: drainAllNonRt は `clearPublishedRuntimeSnapshotsNonRt` が返す World を retire する。この World の retire epoch は `markRetireEpoch()` で取得され、**shutdown 時は既に advanceRetireEpoch() 完了 + activeReaderCount==0** なので epoch safe である。`terminalReclaim()` は deleter を即座に実行するため、**Terminal resident は一時的に増えず（0 のまま）**、`isFullyDrained()` は `terminalReclaimResident == 0` を満たす。

### C.3: epoch unsafe な場合のフォールバック

もし仮に epoch が safe でなかった場合（stuck reader 残り）:
- `terminalReclaim()` は `store()` で保留 → resident +1
- `waitForDrain()` の polling loop 内で `drainDeferredRetireQueues(true)` が呼ばれる
- `drainAllQuarantineStore()` (ReleaseResources.cpp:378) は `m_terminalReclaim.drainAll()` を呼ぶ
- → **shutdown 時の強制 drain で確実に 0 にされる**

**したがって、drainAllNonRt で一時的に resident が増えた場合でも、shutdown sequence の `drainAllQuarantineStore()` (PR2) + `waitForDrain()` の retry loop により最終的に 0 になる。**

---

## D. `drainAllNonRt()` の callback exception / noexcept contract

### D.1. callback は実際に `noexcept` か

`drainAllNonRt()` (OwnerChannel.h:112):
```cpp
template <class Fn>
std::size_t drainAllNonRt(Fn&& reclaim) noexcept
```

**template signature は `noexcept`** だが、`Fn` は任意の型なので、**呼び出し側が `noexcept` な callable を渡す責務** がある。C++ の `noexcept` は「関数本体が例外を投げない」ことを保証するが、**テンプレート引数 `Fn` の `operator()` が例外を投げた場合、`std::terminate` が呼ばれる**（`noexcept` 関数内で例外が飛び出した場合の挙動）。

### D.2. call site の callback の `noexcept` 確認

call site (ReleaseResources.cpp:536-551):
```cpp
const auto drainedResidual = worldAuthority_.ownerChannel().drainAllNonRt(
    [this](const RuntimeState* raw) noexcept {   // ★ noexcept 指定済み
        enqueueDeferredDeleteNonRtWithResult(
            const_cast<RuntimeState*>(raw),
            [](void* p) noexcept { ... },         // ★ noexcept deleter
            DeletionEntryType::World);
    });
```

**結論: ✅ lambda は `noexcept` 指定済み。**

`enqueueDeferredDeleteNonRtWithResult` (AudioEngine.h:4190) は `noexcept` で宣言されており、`shutdownReclaim` → `terminalReclaim` → `store()` / `deleter()` もすべて `noexcept` である。

### D.3. callback 内で ownership を再取得する可能性

```cpp
// drainAllNonRt callback:
[this](const RuntimeState* raw) noexcept {
    enqueueDeferredDeleteNonRtWithResult(
        const_cast<RuntimeState*>(raw), ...);
}
```

`enqueueDeferredDeleteNonRtWithResult` は `raw` を **所有権移転** する（`shutdownReclaim → terminalReclaim → store` または deleter 即時実行）。callback は `raw` を再利用しない（`raw` は `consumeAtomic` で取得した値で、`publishAtomic(nullptr)` で既にスロットを空にしている）。

**結論: ✅ callback は `raw` を再利用しない。**

### D.4. callback failure 時に `raw` が失われないか

`drainAllNonRt` の flow:
```cpp
Owner* const raw = consumeAtomic(s.owner, acquire);  // consume atomic
if (raw != nullptr) {
    publishAtomic(s.owner, nullptr, release);        // publish nullptr (slot empty)
    reclaim(raw);                                     // callback owns raw
    ++reclaimed;
}
```

callback が `noexcept` なので、callback が例外を投げることはない（`std::terminate`）。したがって、**callback が正常に完了すれば `raw` は `enqueueDeferredDeleteNonRtWithResult` に移転済み**。callback が `std::terminate` した場合は、プロセスが終了するため `raw` のリークは問題にならない（プロセス終了時に OS がメモリを回収）。

### D.5: `reclaim(raw)` 後に `raw` を再利用していないか

`drainAllNonRt` は `reclaim(raw)` の後に `raw` にアクセスしない。`++reclaimed` はカウンタのみで `raw` と無関係。

**結論: ✅ single-transfer proof の最後の境界はクリア。**

---

## 15-P-4-0 — Verdict

```
TerminalReclaim applicability:
    PASS

    Rationale:
    - terminalReclaim() always returns true (growable std::vector store)
    - shutdownReclaim → terminalReclaim is the ONLY path when isShutdownInProgress()
    - epoch is safe (advanceRetireEpoch completed, activeReaderCount==0) at drainAllNonRt call site
    - deleter executes immediately (synchronous destruction in Non-RT context)
    - terminalReclaimResidentCount == 0 holds at isFullyDrained() check

shutdown ownership transfer:
    PASS

    Rationale:
    - enqueueRetire() returns only Success | QueuePressure (never Shutdown)
    - enqueueWithRetry() final return result is dead code (QueuePressure always
      enters Stage 3-5 path → returns QueuePressure | TerminalReclaim)
    - shutdownReclaim() returns true (always) → enqueueDeferredDeleteNonRtWithResult
      returns Success (not Shutdown)
    - RetireEnqueueResult::Shutdown is NEVER produced from this call path

D103 dependency on TerminalReclaim:
    PROVEN

    Rationale:
    - drainAllNonRt callback → enqueueDeferredDeleteNonRtWithResult → shutdownReclaim
      → terminalReclaim → epoch safe → synchronous deleter execution
    - World is NOT leaked: deleter runs immediately (or retained for drainAll())
    - terminalReclaimResidentCount is part of isFullyDrained() check → leak detection
    - If epoch unsafe (stuck reader), drainAllQuarantineStore() → m_terminalReclaim.drainAll()
      force-releases ALL entries in PR2 phase before waitForDrain
```

### 依存関係 map

```
drainAllNonRt() callback
  → enqueueDeferredDeleteNonRtWithResult()        [AudioEngine.h:4190]
    → isShutdownInProgress() == true              [AudioEngine.h:4196]
      → shutdownReclaim()                          [ISRRetireRouter.cpp:344]
        → terminalReclaim()                        [ISRRetireRouter.cpp:230]
          ├─ epoch safe + Non-RT → deleter()      [immediately, resident stays 0]
          └─ epoch unsafe → store() → entries_    [resident +1, drained by drainAll()]
            → drainAllQuarantineStore() → drainAll()  [PR2, ReleaseResources.cpp:378]
```

### 証跡 (code references)

| 項目 | ファイル | 行 |
|------|----------|-----|
| `terminalReclaim()` impl | `src/audioengine/ISRRetireRouter.cpp` | 230-306 |
| `TerminalReclaimAuthority::store()` (always true) | `src/audioengine/ISRRetireRouter.cpp` | 27-45 |
| `TerminalReclaimAuthority::drainAll()` | `src/audioengine/ISRRetireRouter.cpp` | 74-92 |
| `shutdownReclaim()` impl | `src/audioengine/ISRRetireRouter.cpp` | 344-360 |
| `enqueueRetire()` return values | `src/audioengine/ISRRetireRouter.cpp` | 210-244 |
| `enqueueWithRetry()` Stage 1-5 | `src/audioengine/ISRRetireRouter.cpp` | 283-338 |
| `enqueueDeferredDeleteNonRtWithResult()` | `src/audioengine/AudioEngine.h` | 4190-4220 |
| `drainAllNonRt` call site | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | 536-551 |
| `drainAllQuarantineStore()` → `drainAll()` | `src/audioengine/ISRRetireRouter.cpp` | 162-170 |
| `isFullyDrained()` with terminalReclaimResident | `src/audioengine/AudioEngine.Threading.cpp` | 138-139 |
| `RetireEnqueueResult` enum | `src/audioengine/ISRAuthorityClass.h` | 28-37 |

---

## 付録: Debug / Releaseビルド検証

| ビルド | ICX | MSVC |
|--------|-----|------|
| Clean rebuild | ✅ 435/435 compiled | ✅ 435/435 compiled |
| CTest (31 tests) | 31/31 PASS (30.24s) | 31/31 PASS (31.27s) |
| HeadlessAudioPathVerification | ✅ PASS (11.57s) | ✅ PASS (11.85s) |

> ICX stale artifact 解消: `build-icx/` を完全 clean (`rm -rf`) 後、ICX コンパイラで再ビルド。HeadlessAudioPathVerification は cli-smoke-test.ps1 経由で `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe` を起動し、**PASS** となった。これにより、元の FAILURE は stale artifact によるものであることが確認された。
