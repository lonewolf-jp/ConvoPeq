# D102-C2-5-A — Terminal Ownership / Caller Audit（read-only, 実装前）

- **実施日**: 2026-08-26 00:30 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 00:00:02**（最新再生成、`src/audioengine/` 差分 0）および **2026-08-25 22:11:33** と同一 production
- **目的**: `enqueueWithRetry() → terminalReclaim() → TerminalReclaimAuthority::store()` の invariant と全 caller ownership flow を最新ソースで再抽出し、`tstored=false` 時の failure semantics を決定する前の監査
- **判定**: **AUDIT GO（実装へ進む） / Implementation gate NO-GO（現行コードは latent bug を含むため修正必須）**

---

## 1. 最新ソース版（ConvoPeq と実 src 照合）

| 項目 | 値 |
|---|---|
| `ConvoPeq.md` header | `Generated: 2026-08-26 00:00:02`（`stat` Modify 00:00:06）— 前回 22:11:33 から harness 追加（`OdenomCampaignTests.cpp` 19KB + `CMakeLists.txt:1829` + `PublishPipelineIntegrationTests.cpp:21`）により再生成 |
| `ConvoPeq(20260825-052131).md` との関係 | 052131 は 05:21 版、22:11 は同日夜版、00:00 は本日 0 時版。**production `src/audioengine/` は 052131 以降変更 0**（`git diff HEAD -- src/audioengine` 0 lines, `git log --oneline -5` ebdcd1e） |
| 実 `src/` 差分 | `src/audioengine/` 0 / `src/DeferredDeletionQueue.h` 0 / `src/audioengine/RetireQuarantineStore.h` 0 / `src/audioengine/ISRRetireRouter.*` 0 — harness-only のみ変更 |
| 本監査の基準 | production に関わる `ISRRetireRouter::enqueueWithRetry` / `terminalReclaim` / `TerminalReclaimAuthority::store` は **22:11:33 と 00:00:02 で同一** |

---

## 2. `ISRRetireRouter::enqueueWithRetry()` 現行コード（`src/audioengine/ISRRetireRouter.cpp:303`）

```cpp
RetireEnqueueResult ISRRetireRouter::enqueueWithRetry(void* ptr,
                                                        void (*deleter)(void*),
                                                        uint64_t epoch,
                                                        DeletionEntryType type) noexcept
{
    jassert(!convo::numeric_policy::isAudioThread()); // B-I3 RT boundary

    // P-4: Ownership chain D → Q → EmergencyQ → TerminalReclaimAuthority
    auto result = enqueueRetire(ptr, deleter, epoch, type);
    if (result == RetireEnqueueResult::Success) return result; // D owns ✅

    constexpr int kMaxRetry = 2;
    for (int attempt = 0; attempt < kMaxRetry; ++attempt) {
        provider_->tryReclaim();
        drainEmergencyAndTerminal();
        result = enqueueRetire(ptr, deleter, epoch, type);
        if (result == RetireEnqueueResult::Success) return result;
        if (result != RetireEnqueueResult::QueuePressure) break;
    }

    if (result == RetireEnqueueResult::QueuePressure || result == RetireEnqueueResult::QueueFull)
    {
        const bool stored = m_retireQuarantine.quarantine(ptr, deleter, epoch, type,
                                                           "enqueueWithRetry:QueuePressure");
        if (stored) result = RetireEnqueueResult::QueuePressure; // Q owns ✅
        else {
            const bool estored = m_emergencyQuarantine.quarantine(ptr, deleter, epoch, type,
                                                                   "enqueueWithRetry:EmergencyQuarantine");
            if (estored) result = RetireEnqueueResult::QueuePressure; // E owns ✅
            else {
                // E full → TerminalReclaimAuthority
                const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                                      "enqueueWithRetry:TerminalReclaim");
                (void)tstored;  // ★ P-4: 常に true（growable store）
                result = RetireEnqueueResult::TerminalReclaim; // Terminal owns ptr ✅
            }
        }
    }
    return result;
}
```

**Invariant コメント:** `ISRRetireRouter.cpp:25` `enqueueWithRetry() never returns with ptr unowned.` / `src/audioengine/ISRRetireRouter.cpp:319` `TerminalReclaimAuthority は growable store のため常に ownership を受領する。`

---

## 3. `ISRRetireRouter::terminalReclaim()` 現行コード（`src/audioengine/ISRRetireRouter.cpp:509`）

```cpp
bool ISRRetireRouter::terminalReclaim(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                      DeletionEntryType type, const char* reason) noexcept
{
    const uint64_t minReader = minReaderEpoch();
    const bool epochSafe = ISRRetireRouter::isOlder(epoch, minReader); // epoch < minReader → safe
    const bool isRt = convo::numeric_policy::isAudioThread(); // P-4 RT防御

    if (epochSafe && !isRt) {
        deleter(ptr); // Synchronous destruction (Non-RT, epoch-safe のみ)
        if (type == DeletionEntryType::World) m_terminalReclaim.recordWorldReclaim();
        return true; // destroyed immediately, no storage
    }
    // epoch unsafe OR RT caller → store for later drain
    return m_terminalReclaim.store(ptr, deleter, epoch, type, reason); // ALWAYS true
}
```

- `isOlder(a,b) = static_cast<int64_t>(a-b) < 0` (`src/audioengine/ISRRetireRouter.cpp:565`)
- `store` は `TerminalReclaimAuthority::store:27` で `std::vector::push_back` による growable store。**常に true** を返す設計。

---

## 4. `TerminalReclaimAuthority::store()` / `drain()` / `drainAll()` 現行コード

### `store()`（`src/audioengine/ISRRetireRouter.cpp:27` / `src/audioengine/ISRRetireRouter.h:62`）

```cpp
bool TerminalReclaimAuthority::store(void* ptr, void (*deleter)(void*), uint64_t epoch,
                                     DeletionEntryType type, const char* reason) noexcept
{
    // entries_ is GROWABLE (std::vector). store() ALWAYS succeeds.
    entries_.push_back({ptr, deleter, epoch, type, reason});
    convo::fetchAddAtomic(residentAtomic_, 1, memory_order_release);
    return true;
}
```

**Invariant:** `TerminalReclaimAuthority:62` `ALWAYS returns true (growable store) — ownership always transfers.`
`true → Terminal owns ptr / false → caller still owns ptr` の意味論は **設計上 `false` が発生しない** ことが前提。

### `drain()`（`src/audioengine/ISRRetireRouter.cpp:52`）

```cpp
void TerminalReclaimAuthority::drain(uint64_t minReaderEpoch,
                                     const std::function<bool(uint64_t,uint64_t)>& isOlderFn) noexcept
{
    // isOlder(entry.epoch, minReaderEpoch) == true の entry のみ deleter 実行
    // epoch unsafe な entry は保持
}
```

### `drainAll()`（`src/audioengine/ISRRetireRouter.cpp:92`）

- `drainAll()` は shutdown 時のみ（Audio Thread 停止後）に呼ばれる。epoch チェックなしで全 entry を破棄。

---

## 5. 全 caller 一覧（`enqueueWithRetry()` + `enqueueDeferredDeleteNonRtWithResult` 経由）

`rg "enqueueWithRetry\("` および `rg "enqueueDeferredDeleteNonRt"` の全 matches を caller 単位に整理。**14 caller** は `ISRRetireRouter::enqueueWithRetry` の直接 caller 8件 + `AudioEngine::enqueueDeferredDeleteNonRtWithResult` 経由の間接 caller 6件 を合わせた数。

| # | Caller file | Symbol | Direct / Indirect | Return 取得 | 備考 |
|---|---|---|---|---|---|
| 1 | `src/audioengine/AudioEngine.h:4226` | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | **Wrapper** (harness 含む全 deferred delete の入口) | `RetireEnqueueResult` を return | 全 deferred delete の 1次ラッパー |
| 2 | `src/core/SnapshotCoordinator.h:94` | `SnapshotCoordinator::switchImmediate` (oldTarget) | Direct | `result == Shutdown` を分岐 | work37 Phase 1.2 |
| 3 | `src/core/SnapshotCoordinator.h:107` | `SnapshotCoordinator::discardSnapshot` (oldSnap) | Direct | `bool` (ignore) | `if (!enqueueWithRetry) { }` |
| 4 | `src/core/SnapshotCoordinator.h:176` | `SnapshotCoordinator::retireSnapshot` (snap, generation) | Direct | `bool` (ignore) | 2箇所 |
| 5 | `src/core/SnapshotCoordinator.h:181` | 同上 (別 generation) | Direct | `bool` (ignore) | 同 file 内 2回目 |
| 6 | `src/core/SnapshotCoordinator.cpp:57` | `SnapshotCoordinator::startFade` (oldTarget) | Direct | `result` 未使用 | BUG-015/027 |
| 7 | `src/core/SnapshotCoordinator.cpp:114` | `SnapshotCoordinator::completeFade` (old) | Direct | `result` 未使用 | 同上 |
| 8 | `src/audioengine/DSPLifetimeManager.cpp:49` | `DSPLifetimeManager::retireDSPCore` (dsp) | Direct | `result` ログのみ | `router_->enqueueWithRetry(dsp, destroyDSPCoreNode)` |
| 9 | `src/audioengine/DSPLifetimeManager.cpp:96` | `DSPLifetimeManager::retireDSPCore` (別 path) | Direct | 同上 | 同 file 2回目 |
| 10 | `src/eqprocessor/EQProcessor.Core.cpp:61` | `EQProcessor::retire` (stackRouter) | Direct | `result` 未使用 | `stackRouter.enqueueWithRetry` |
| 11 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:162` | `ISRRuntimePublicationCoordinator::retire` | Direct | `result == Success` 分岐 | `router.enqueueWithRetry(ptr, deleter, Generic)` |
| 12 | `src/tests/RetireGraceSemanticsTests.cpp:679` | Test harness | Direct | `router.enqueueWithRetry` | Test のみ |
| 13 | `src/audioengine/AudioEngine.Cache.cpp:16,41` | `AudioEngine::cache` (via `enqueueDeferredDeleteNonRt`) | Indirect | `bool` (Shutdown 除外) | `enqueueDeferredDeleteNonRt` → `WithResult` → `enqueueWithRetry` |
| 14 | `src/convolver/ConvolverProcessor.Lifecycle.cpp:57,70` | `ConvolverProcessor` | Indirect | `void` | 同上 |

**Tool coverage:** `rg` 14 locations / `AiDex` `enqueueWithRetry` 39 matches / `AiDex` `enqueueDeferredDeleteNonRtWithResult` 6 matches / `ag` `ShutdownDiscard` / `ast-grep -p "tstored"` 2 hits / `fdfind` `I4` / `fzf` / `sed`で各 caller の B3-A5 文脈抽出。`serena` timeout → `AiDex/rg` 代替、`cocoindex`/`semble`/`graphify` は WSL `which` で不在確認し `rg` 代替。

---

## 6. 14 Caller Ownership Matrix（A/B/C/D 分類）

### 分類定義

```text
A. ownership transferred (authority が所有)
B. ownership retained by caller (caller が所有を保持)
C. ownership destroyed safely (epoch-safe 同期破棄で所有終了)
D. ownership ambiguous (caller が transfer 済みと誤認し得る)
```

### 現行コードでの分類（`tstored` 無条件 `TerminalReclaim` 返却）

| # | Caller | return value を取得 | `tstored==true` 時 (現行) | `tstored==false` 時 (仮に bounded 化) | 現在の問題 |
|---|---|---|---|---|---|
| 1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | Yes (`RetireEnqueueResult`) | A/C (`Success` or `TerminalReclaim` → Terminal が所有/破棄) | **D** (`TerminalReclaim` を返すと caller は transfer 済みと誤認 → **leak**) | **D が発生** |
| 2 | `SnapshotCoordinator::switchImmediate` | Yes (`result == Shutdown` 分岐) | A/C | **D** (`TerminalReclaim` 誤認) | **D** |
| 3 | `SnapshotCoordinator::discardSnapshot` | No (`if (!enqueueWithRetry)`) | A (bool true) | **D** (true を返すと caller は破棄不要と誤認 → leak) | **D** |
| 4-5 | `SnapshotCoordinator::retireSnapshot` (2箇所) | No | A | **D** | **D** |
| 6-7 | `SnapshotCoordinator.cpp:57,114` | No | A | **D** | **D** |
| 8-9 | `DSPLifetimeManager.cpp:49,96` | Partial (log) | A | **D** | **D** |
| 10 | `EQProcessor.Core.cpp:61` | No | A | **D** | **D** |
| 11 | `ISRRuntimePublicationCoordinator.cpp:162` | Yes | A/C | **D** | **D** |
| 12 | `RetireGraceSemanticsTests.cpp:679` | Yes (test) | A | **D** (test が pass してしまう) | **D** |
| 13-14 | `AudioEngine.Cache.cpp` / `ConvolverProcessor` (indirect) | No (`enqueueDeferredDeleteNonRt` → `bool`) | A (bool true, Shutdown 除外) | **D** | **D** |

**判定:** `tstored==false` 時に **D が 14/14 件で残る** — **implementation gate は NO-GO**（現行 growable では dormant だが、bounded 化した瞬間に leak を発生させる latent bug）。

---

## 7. `tstored=false` の場合の caller ごとの挙動（詳細）

### 現行 (growable) での挙動

```text
TerminalReclaimAuthority::store() は常に true を返す
  → tstored == true が常に成立
  → result = TerminalReclaim は常に正しい（Terminal が所有）
  → caller は所有を手放してよい（A/C）
```

### 仮に bounded 化した場合（`tstored==false` が発生し得る）

```text
tstored == false
  意味: Terminal が ownership を受領していない → caller still owns ptr
  禁止: ptr を破棄してよい、と解釈すること
        return TerminalReclaim をそのまま返すこと
  正しい: caller が ownership を保持できる result semantics を返す
         （例: QueueFull / QueuePressure / Shutdown 以外の caller-retain-compatible result）
```

**Caller 別の `tstored==false` 時の正しい挙動:**

| Caller group | 現在の戻り値処理 | `tstored==false` で必要なこと |
|---|---|---|
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult` | `TerminalReclaim` を返す → caller は `drainDeferredRetireQueues` 後に return | `tstored==false` なら `TerminalReclaim` を返さず、**caller が再試行または backpressure できる値**を返す（`QueueFull` 等） |
| `SnapshotCoordinator` (bool 返却) | `true` → caller は破棄完了とみなす | `false` を返し caller が retry/backpressure できるようにする |
| `DSPLifetimeManager` / `EQProcessor` | 戻り値を log のみ | 同上、呼び出し元が所有を保持していることを伝える必要あり |
| Indirect (`AudioEngine.Cache`) | `bool` (`Shutdown` 以外は true) | `false` 相当を返し caller が所有を保持 |

---

## 8. Release build での ownership loss の有無

- 現行コード: `(void)tstored` により Release でも `tstored` の値は破棄されるが、`result = TerminalReclaim` は無条件で実行される。
- `jassert(tstored)` **のみ**を追加した場合、Debug では検出できるが **Release (NDEBUG) では消える**ため、`tstored==false` の runtime semantics が残り、**Release で D (ambiguous) のまま leak** する。
- したがって **Release 相当（NDEBUG）でも ownership loss = 0 を保証するには、`tstored==false` を production Release でも ownership-safe に処理できる分岐が必須**。

```
現行 (Release): tstored==false でも TerminalReclaim を返す → caller は所有を手放す → leak (UAF ではないが leak) → ownership loss > 0
修正後: if (!tstored) return caller-retain result → caller が所有を保持 → leak 0
```

---

## 9. D14.3 / D15.2 との照合

### D14.3 — budget 枯渇 = backpressure（`doc/work88/I4_DESIGN_CONTRACT.md:154`）

```text
budget 枯渇で reservation を取得できない場合、admission を BLOCK（backpressure / upstream retry）
INV-X1-7 改訂: logical obligation の消失理由から terminal-failure を除外
```

- 現行 `enqueueWithRetry` の `TerminalReclaim` は **D14.3 に違反しない**（Terminal へ移送するため消失ではない）。しかし `tstored==false` 時に `TerminalReclaim` を誤返却すると、caller が **terminal-failure を disappearance として扱う**ことになり D14.3 違反となる。

### D15.2 — Ownership conservation（`doc/work88/I4_DESIGN_CONTRACT.md:179`）

```text
transport + durable + building + stalled + superseded + shutdownDiscard == admittedLogicalObligationCount
terminal-failure は消失理由に含めない
disappear only by: Success / Superseded / ShutdownDiscard
```

- `tstored==false` 時に `TerminalReclaim` を返すと、logical obligation が **いずれの右辺にも属さない状態**（Transport/Durable/Building/Stalled でもなく Superseded/ShutdownDiscard でもない）となり、**conservation 式が破れる**（`D` 分類）。
- 正しくは `tstored==false` 時に caller が所有を保持し、**Transport/Durable 等のいずれかに再計上**されるか、backpressure により admission 段階でブロックされるべき。

---

## 10. Implementation Requirement（D102-C2-5-D 以降）

### 最小 production implementation（候補）

```cpp
const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                      "enqueueWithRetry:TerminalReclaim");

jassert(tstored); // Debug 検出

if (!tstored)
{
    // ownership is still held by caller.
    // Must not report TerminalReclaim (would cause D/ambiguous).
    // Return caller-retain-compatible result (e.g. QueueFull) so caller
    // can retry/backpressure without leak.
    return RetireEnqueueResult::QueueFull; // or QueuePressure, not TerminalReclaim
}

return RetireEnqueueResult::TerminalReclaim;
```

**ただし `jassert` だけでは不十分** — Release でも `tstored==false` を正しく処理する分岐（上記 `if (!tstored)`）が必須。

### 検討事項

- `tstored==false` 時の return 値として `QueueFull` / `QueuePressure` のいずれが caller の既存分岐（`result == QueuePressure || QueueFull`）と整合するか、全 14 caller の分岐を再検証して決定する。
- `RetireEnqueueResult` に `TerminalFailure` 等の新規値を追加するかは **D14/D15 contract の改訂を要するため行わない**（本 task の禁止事項）。
- `epoch-safe synchronous destruction` の gate（`epochSafe && !isRt`）は **変更しない**（`if (!tstored) deleter(ptr)` は禁止、D101-9 Candidate D と同一の reject 方向）。

### テスト matrix（D102-C2-5-D 前に固定）

| Test | 条件 | 期待 |
|---|---|---|
| T1 Terminal store success | D/Q/E full, epoch unsafe, store success | `TerminalReclaim`, Terminal owns, deleter 0 |
| T2 epoch-safe NonRT | D/Q/E full, epoch safe, NonRT | deleter 1, Terminal store 0 |
| T3 epoch-unsafe | D/Q/E full, epoch unsafe, NonRT, store success | deleter 0, Terminal resident +1 |
| T4 Terminal store failure | D/Q/E full, Terminal store false | deleter 0, **TerminalReclaim を成功扱いしない**, caller owns, silent discard 0 |
| T5 RT caller | RT, Terminal 到達 | sync destruction 0 |
| T6 Release 相当 (NDEBUG, Terminal failure) | `tstored==false` | ownership loss 0, UAF 0, leak 0 |

### D14/D15 invariant test

```text
transport + durable + building + stalled + superseded + shutdownDiscard == admittedLogicalObligationCount
terminal failure を右辺に追加しない
```

### Growable に依存しないこと

- 現状 `std::vector` growable により `tstored==false` が発生しない説明は成立するが、**実装契約としては `Terminal always accepts OR caller retains safely` までコードで閉じる**。将来 bounded 化しても invariant が壊れないようにする。

---

## 11. GO / NO-GO

| Gate | 判定 | 理由 |
|---|---|---|
| **Audit GO**（次工程へ進むか） | **GO** | invariant、caller 全件、failure semantics、D14/D15 照合が確定し、implementation requirement が明確になった |
| **Implementation gate**（現行コードで実装完了か） | **NO-GO** | `tstored` を無条件で `TerminalReclaim` として返す現行コードは、bounded Terminal を仮定すると 14/14 caller で **D (ownership ambiguous)** を生む latent bug であるため。`jassert` だけでは Release で解消しない |

**次工程:** `D102-C2-5-D` 最小 production implementation（`tstored` failure guard） → `D102-C2-5-E` unit/contract tests → `D102-C2-5-F` Debug+Release build → `D102-C2-5-G` CTest → `D102-C2-5-H` grep/static audit → `D102-C2-5-CLOSE`

*本監査は read-only であり、production source 変更 0、O_denom/λ/G/M の再測定・再導出なし、Terminal bounded 化なし、D14/D15 改訂なしで実施された。*
