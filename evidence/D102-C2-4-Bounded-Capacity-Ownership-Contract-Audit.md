# D102-C2-4 — Bounded Capacity と Lifetime/Ownership Contract 整合性監査（read-only）

- **実施日**: 2026-08-25 23:50 (JST)
- **作業種別**: read-only contract/evidence review（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-25 22:11:33** + 実 `src/`（`git diff HEAD -- src/audioengine` 0 lines） / `evidence/D102-C2-3-CLOSE.md` 固定値
- **判定**: **PASS（contract 整合・ownership conservation 維持）** — `R_required=4121 <= R_cap,bounded=5120` かつ Terminal-full 時の unsafe destruction 経路なし。`enqueueWithRetry` の `tstored` 無視は growable 保証により ownership 消失を生まないが、防御的 assert の追加を推奨（本監査では実装変更なし）

---

## 0. 前提固定値（D102-C2-3 CLOSE 値）

```text
λ_prod_bound = 13 events/s
G_bound      = 1.0 s
K_starve     = 1.0 s
T_sampler    = 100 ms
M_scope      = 4120
O_denom      = 1 (campaign-wide max, 10 eligible windows)
K_min        = 4120
R_required   = 4121
R_cap,bounded = 5120 (=4096(D)+512(Q)+512(E))
Terminal dep = 0
```

---

## 1. `R_required=4121 <= R_cap,bounded=5120` — PASS

```text
R_required = 1 + ceil(4120/1) = 4121
R_cap,bounded = 5120
4121 <= 5120 → PASS bounded、headroom 999
```

## 2. Terminal dependency = 0 — PASS

```text
Terminal dep = max(0, 4121-5120) = 0
```

Terminal 容量の追加検討は不要。ただし「Terminal が不要」≠「Terminal-full 時に sync 破棄してよい」ことを §3-6 で別途証明する。

---

## 3. Terminal-full を理由に ownership を消失させる経路がない — PASS

### 3.1 Ownership chain（`src/audioengine/ISRRetireRouter.cpp:315`）

```text
D (DeferredDeletionQueue 4096, lock-free)
  → Q (RetireQuarantineStore 512, mutex + allocation-free array)
    → E (EmergencyQuarantineStore 512, 同型)
      → T (TerminalReclaimAuthority, growable std::vector)
```

- `enqueueWithRetry` は `D → Q → E → T` の順に ownership を委譲する。各段階で `stored` が true ならその authority が ownership を保持。
- `RetireQuarantineStore::quarantine` / `EmergencyQuarantineStore::quarantine` は `size_ >= kMaxQuarantinedEntries` で `overflowCount_` を記録し `false` を返すが、`deleter` を実行しない（UAF 構造的排除 `RetireQuarantineStore.h:78-84`）。
- 呼び出し元 `ISRRetireRouter::enqueueWithRetry:343-368` は `false` 時に次段へフォールスルーし、最終的に `T` へ到達する。`T` は `std::vector` により常に受領するため、`QueuePressure/QueueFull` を理由に ptr を drop する経路は存在しない。

### 3.2 RetireEnqueueResult の意味論（`src/audioengine/ISRAuthorityClass.h:28`）

```text
enum RetireEnqueueResult { Success, QueuePressure, QueueFull, Shutdown, TerminalReclaim }
```

- `Success` — D が所有
- `QueuePressure` — Q/E が所有（`enqueueWithRetry:343,353` で `stored==true` 時に設定）
- `TerminalReclaim` — T が所有（`ISRRetireRouter.cpp:368`）
- `Shutdown` — shutdown 中の early return（`src/core/SnapshotCoordinator.cpp:56` 等、正常時の bounded 容量とは別経路）

Terminal-full を理由に `QueueFull` を返して caller が `deleter(ptr)` する経路は **存在しない**（`enqueueWithRetry` は `QueueFull` を `Q` 判定に含めつつ `T` へフォールスルーする `src/audioengine/ISRRetireRouter.cpp:338`）。

## 4. Epoch-unsafe ptr の synchronous destruction がない — PASS

### 4.1 TerminalReclaim の epoch-gated 破棄（`src/audioengine/ISRRetireRouter.cpp:509`）

```cpp
bool ISRRetireRouter::terminalReclaim(..., epoch, ...) {
  epochSafe = isOlder(epoch, minReaderEpoch); // epoch < minReader → 全 Reader 通過済み
  isRt = isAudioThread();
  if (epochSafe && !isRt) { deleter(ptr); return true; } // epoch-safe かつ Non-RT のみ同期破棄
  return m_terminalReclaim.store(ptr, ...); // unsafe なら growable store へ保持
}
```

- `isOlder(a,b) = static_cast<int64_t>(a-b) < 0`（wraparound 安全 `ISRRetireRouter.cpp:565`、EpochDomain と同一）
- `QuarantineStore::drain:116` も `isOlderFn(e.epoch, minReaderEpoch)` が true の entry のみ `deleter` 実行
- `TerminalReclaimAuthority::drain:52` も同様に epoch-gated

∴ epoch-unsafe な ptr を `deleter` で同期破棄する経路は存在しない。

### 4.2 RT 防御

- `enqueueWithRetry:308` `jassert(!isAudioThread())` — Q/E/T 到達は Non-RT のみ。RT path は `retireRT() → enqueueRetire(D のみ)` を使用。
- `terminalReclaim:518` でも `isAudioThread()` を再チェックし、RT caller は `store` へフォールバック（synchronous destruction 禁止）。

## 5. `enqueueWithRetry()` の Terminal failure handling と ownership conservation の整合 — **最重要 PASS（要防御的改善指摘）**

### 5.1 該当コード（`src/audioengine/ISRRetireRouter.cpp:359-368`）

```cpp
// E full → Stage 5: TerminalReclaimAuthority
// D+Q+E 全滿 → TerminalReclaimAuthority へ移送
const bool tstored = terminalReclaim(ptr, deleter, epoch, type,
                                      "enqueueWithRetry:TerminalReclaim");
(void)tstored;  // ★ P-4: 常に true（growable store）
result = RetireEnqueueResult::TerminalReclaim;  // Terminal owns ptr ✅
```

- `tstored` を取得しつつ `(void)tstored` で無視し、無条件に `TerminalReclaim` を返す。

### 5.2 なぜ ownership 消失を生まないか

1. `terminalReclaim` は `epochSafe && !isRt` なら `deleter(ptr)` を同期実行して `true` を返す（**ownership は破棄により正常終了**、store 不要）。
2. それ以外は `m_terminalReclaim.store(...)` を呼び、`store` は `std::vector::push_back` による growable store のため **常に true** を返す（`src/audioengine/ISRRetireRouter.h:62-73` `ALWAYS returns true`）。
3. したがって `tstored==false` となる failure path は構造的に存在しない。`TerminalReclaim` を返すことは ownership が `T`（または同期破棄により既に解放）に移ったことを意味し、ptr が宙に浮くことはない。

### 5.3 過去指摘との関係

> `terminalReclaim()` の `tstored` を無視して `TerminalReclaim` を返す可能性

は、**growable 保証を前提にすれば** ownership 消失を生まない。`R_required=4121` で bounded 内に収まる本件では Terminal 到達自体が稀だが、将来 M_scope が増大して Terminal 到達が常態化しても、上記 5.2 の invariant により安全である。

### 5.4 防御的改善提案（本監査では実装変更なし）

- `(void)tstored` は意図を明確にする一方、**failure を検出できない**。`tstored==false` は growable 前提の破綻を意味するため、**`jassert(tstored)` または `if (!tstored) { jassertfalse; return Shutdown; }` のガード**を追加することを推奨する。
- 現行コードは `isAudioThread()` の `jassert` と `epochSafe` の分岐で安全性を確保しており、**本監査時点では PASS** と判定する。実装変更は D102-C2-4 の read-only 方針により見送り、次ゲートで検討する。

### 5.5 Tool 棚卸し

| ツール | 検索内容 | 結果 |
|---|---|---|
| `rg` `enqueueWithRetry\|terminalReclaim\|tstored` | 70 matches | 該当 code 特定 |
| `ast-grep -p "tstored"` | `ISRRetireRouter.cpp:365,367` | 同上 |
| `AiDex` `enqueueWithRetry` | 39 matches | 呼び出し元網羅 |
| `sed -n "303,400p"` `509,545p` | `enqueueWithRetry` / `terminalReclaim` 全文抽出 | 上記 5.1-5.2 の根拠 |
| `ag` `ShutdownDiscard` | `ISRRuntimePublicationCoordinator.cpp:822` 等 | Shutdown と Terminal の分離確認 |

## 6. ShutdownDiscard と通常時の terminal failure の分離 — PASS

- **ShutdownDiscard** は `ISRRuntimePublicationCoordinator.cpp:822,908,954` で `recoveryShutdownDiscardCount_` として **明示的に記録**される `ShutdownDiscard`（`D15.2` の `shutdownDiscardCount` に対応）。`popRecoveryRequest` 残留 Recovery の明示破棄等、shutdown 時のみ発生。
- **Terminal failure** は `RetireEnqueueResult::TerminalReclaim` として `RetireQuarantineStore` とは異なる authority に ownership が移るだけで、**消失理由としてカウントされない**（`D15.2` の `terminal-failure は消失理由に含めない` に合致）。
- 両者は `RetireEnqueueResult` (`Shutdown` vs `TerminalReclaim`) および `recoveryShutdownDiscardCount_` vs `terminalReclaimResidentCount()` で **明確に分離**されている。

## 7. D14/D15 contract と実コードの一致 — PASS

### D14 — Capacity / reservation-first / backpressure（`doc/work88/I4_DESIGN_CONTRACT.md:130`）

| 契約 | 実コード |
|---|---|
| D14.2 `1 logical obligation = exactly 1 reservation` | `ISRRuntimePublicationCoordinator` の `kMaxLogicalRecoveryObligations` による `transport+durable+building+stalled <= kMax` 不変式（`I4:150`） |
| D14.3 `budget 枯渇 = backpressure`（terminal-failure 撤廃） | `ISRRuntimePublicationCoordinator.cpp:163` `QueuePressure` 時に `jassert` + `HealthEvent` で admission を BLOCK、terminal-failure を消失理由にしない（`I4:155` `INV-X1-7 改訂`） |
| capacity 根拠 | `R_required=4121` vs `R_cap,bounded=5120` で bounded 内 PASS を本監査で証明 |

### D15 — Ownership conservation（`doc/work88/I4_DESIGN_CONTRACT.md:170`）

```text
admittedLogicalObligationCount = transport+durable+building+stalled + superseded + shutdownDiscard
terminal-failure は消失理由に含めない
disappear only by: Success / Superseded / ShutdownDiscard
```

- 実コードの `enqueueWithRetry` は Terminal 到達時も `TerminalReclaim` として ownership を保持し、`ShutdownDiscard` とは分離（§5,6）。
- `RetireQuarantineStore::quarantine` の `overflowCount_` は `HealthEvent` へ昇格し、silent discard を禁止（`I4:160`）。
- `D15.2` の `INV-X1-5`（1 obligation =1 reservation）と `INV-X1-7`（disappear 条件）は、bounded capacity が `R_required` を満たす本件で **コードと一致**。

## 8. Production source modification — 0

```text
git diff HEAD -- src/audioengine → 0 lines
git diff HEAD -- src/DeferredDeletionQueue.h src/audioengine/RetireQuarantineStore.h → 0 lines
```

- 本監査は read-only。`tstored` の defensive assert 等の改善は次ゲートで検討。

---

## 総合判定

| # | 確認 | 判定 |
|---|---|---|
| 1 | `R_required=4121 <= R_cap,bounded=5120` | **PASS** |
| 2 | Terminal dependency = 0 | **PASS** |
| 3 | Terminal-full を理由に ownership 消失なし | **PASS** |
| 4 | epoch-unsafe の synchronous destruction なし | **PASS** |
| 5 | `enqueueWithRetry` Terminal handling と ownership conservation の矛盾なし | **PASS**（防御的 assert 推奨） |
| 6 | ShutdownDiscard と terminal failure の分離 | **PASS** |
| 7 | D14/D15 contract と実コード一致 | **PASS** |
| 8 | production source modification | **0** |

### D102-C2-4 — **PASS（read-only）**

- `Practical Stable ISR Bridge Runtime` の原則（Retire → Epoch → Reclaim 境界維持、Overflow/Shutdown/lifetime を容量不足の同期破棄で代替しない）を満たす。
- 数値上 Terminal が不要（`4121<5120`）でも、`enqueueWithRetry` の ownership chain は Terminal まで保証されており、将来の M_scope 増大時も同一 invariant が維持される。
- 次の実装ゲートでは、本監査の推奨（`tstored` の `jassert`）を含め、D14/D15 の backpressure/ownership conservation を前提とした変更のみを許可する。

---

## 参照

- `src/audioengine/ISRRetireRouter.cpp:303,359-368,509` / `src/audioengine/ISRRetireRouter.h:39,62` / `src/audioengine/ISRAuthorityClass.h:28` / `src/audioengine/RetireQuarantineStore.h:65` / `doc/work88/I4_DESIGN_CONTRACT.md:130,170`
- `evidence/D102-C2-3-CLOSE.md` / `evidence/D102-C2-3-B-Campaign-Execution-Report.md` / `evidence/OdenomCampaign_console_2026-08-25.log`

*本監査は read-only であり、ソースコード変更 0、契約変更 0、O_denom の昇格 0 で実施された。*
