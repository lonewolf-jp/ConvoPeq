# D102-C2-5-D8-2-C — Disposition Contract Audit (read-only)

- **実施日**: 2026-08-26
- **基準版**: ConvoPeq.md `Generated: 2026-08-26 20:06:47`（= 2026-08-26 11:07 UTC 相当。ローカルソースと突合済み）
- **作業種別**: **read-only audit**（production source 変更 **0** / テスト追加 **0**）
- **総合判定**: **PASS（9/9 条件成立）** — patch 不要。FAIL/要修正項目は 0 件。

---

## C1. D → Q → E → T の ownership conservation

### 実コード検証（`ISRRetireRouter.cpp:303-384`）

```text
Stage 1 (L322): enqueueRetire(ptr,deleter,epoch,type)
    ├─ provider_->enqueueRetireTyped() == true  → return Success            … D owns ✅
    └─ false → QueuePressure を返す（4-arg enqueueRetire は Success|QueuePressure の
                2 値しか返さない — ISRRetireRouter.cpp:239-273 で確認）

Stage 2 (L327-336): retry ×2
    provider_->tryReclaim() → drainEmergencyAndTerminal() → 再 enqueue
    ├─ Success → return Success                                             … D owns ✅
    └─ QueuePressure 継続 → Stage 3 へ

Stage 3 (L339-346): if (result == QueuePressure || result == QueueFull)
    m_retireQuarantine.quarantine(...)          ← stored
    ├─ stored == true  → result = QueuePressure; signalDrainWakeup(); return   … Q owns ✅
    └─ stored == false → Stage 4

Stage 4 (L352-356):
    m_emergencyQuarantine.quarantine(...)       ← estored
    ├─ estored == true → result = QueuePressure                                … E owns ✅
    └─ estored == false → Stage 5

Stage 5 (L365-368):
    terminalReclaim(...)                        ← tstored（常に true、growable）
    result = TerminalReclaim                                                   … T owns / 即時削除 ✅
```

### Q/E store full 時の契約（`RetireQuarantineStore.h:70-84`）

```cpp
if (size_ >= kMaxQuarantinedEntries) {   // 512
    ++overflowCount_;
    return false;  // store full — caller must NOT delete
}
```
→ 満杯時は **deleter を実行せず** false を返す。ownership は移転しない（次 authority へ委譲）。

### TerminalReclaim の二分岐（`ISRRetireRouter.cpp:509-533`）

```text
epochSafe && !isRt  → deleter(ptr) 実行（synchronous destruction）、storage 不要 → true
otherwise           → m_terminalReclaim.store(...)（growable・常に受領）        → true
```
→ **いずれの分岐でも ownership は Router 内で完結し、true 固定**。
`tstored` が false になる経路は存在しない（store() は `entries_.push_back()` 常成功、`ISRRetireRouter.cpp:27-50`）。

### Result ⇔ ownership 対応表（確定版）

| Result | ownership | 根拠 |
|---|---|---|
| `Success` | **D** | L322/L332 のみで return |
| `QueuePressure` | **Q または E** | stored==true → Q / estored==true → E（どちらか排他的に 1 authority） |
| `TerminalReclaim` | **T または immediate delete** | epochSafe+NonRT → delete 済 / otherwise → T store |
| `QueueFull` | **現行 production 到達不能** | D の enqueueRetire は QueueFull を返さない（L272 は QueuePressure のみ）。Stage 3 条件式に名前があるのみ |
| `Shutdown` | **enqueueWithRetry の disposition ではない** | 全 return point を走査（C2）した結果、Shutdown を返す経路なし |

**判定: PASS** — 各遷移で before/after とも exactly one owner。

---

## C2. 「caller ownership 残存」経路の再検証

`enqueueWithRetry()` の全 return point 走査（L324, L333, L379, L383）:

| Return point | 到達条件 | ownership 状態 |
|---|---|---|
| L324 `return result(Success)` | D 初回成功 | D owns |
| L333 `return result(Success)` | retry 成功 | D owns |
| L379 `return result` | Q/E/T 受入後 | Q / E / T-or-deleted owns |
| **L383 `return result`** | `result` が QueuePressure/QueueFull **以外**で Stage 3 未突入 | — |

L383 は「Shutdown 等」を想定したフォールバックだが、Stage 1 の `enqueueRetire`(4-arg) は
**Success / QueuePressure の 2 値しか返さない**ため、L383 は**到達不能（dead path）**。

さらに外部 caller 側も確認:
- `AudioEngine.h:4206-4221` `enqueueDeferredDeleteNonRtWithResult`: shutdown 中は `shutdownReclaim()` で
  **先に移送してから** return（transferred ? Success : Shutdown）。ptr を捨てない。
- `EQProcessor.Core.cpp:61-67`: Success/QueuePressure/TerminalReclaim を全て transfer 成立と扱う。
- `ISRRuntimePublicationCoordinator.cpp:162-164`: result をそのまま転送するのみ（追加 quarantine なし）。

> **現行コードにおいて enqueueWithRetry() が caller ownership を残して return する経路は存在しない。**

**判定: C2 = PASS / QueueFull deferred**（bounded Terminal 化時の設計は D2-1 別 gate）

---

## C3. SnapshotCoordinator closure audit

### fallback チェーン（source-level corroboration）

| 呼び出し箇所 | enqueueWithRetry 失敗時の処理 |
|---|---|
| `switchImmediate` oldTarget (`SnapshotCoordinator.h:94-98`) | `quarantineRetireSink(oldTarget, ...)` |
| `switchImmediate` oldSnap (`SnapshotCoordinator.h:107-108`) | `quarantineRetireSink(oldSnap, ...)` |
| `startFade` oldTarget (`SnapshotCoordinator.cpp:57-61`) | `quarantineRetireSink(oldTarget, ...)` |
| `completeFade` old (`SnapshotCoordinator.cpp:114-118`) | `quarantineRetireSink(old, ...)` |
| destructor `retireCurrentAndTarget` (`SnapshotCoordinator.h:176,181`) | `quarantineRetireSink(snap, ...)` |

### `quarantineRetireSink` 実体（`SnapshotCoordinator.cpp:21-30`）

```cpp
void SnapshotCoordinator::quarantineRetireSink(void* ptr, void (*deleter)(void*),
                                               uint64_t epoch, const char* reason) noexcept
{
    if (m_retireSink == nullptr || ptr == nullptr || deleter == nullptr)
        return;
    const bool stored = m_retireSink->quarantineRetire(
        ptr, deleter, epoch, DeletionEntryType::Generic, reason);
    if (!stored)
        assert(false && "RetireQuarantineStore capacity exhaustion ...");
}
```

確認結果:
- **direct delete: なし**（SnapshotCoordinator は SnapshotFactory::destroy を直接呼ばない）
- **second quarantine: なし**（quarantineRetire 1 回のみ呼出し。Q full 時は assert で異常検出 —
  E/T への追加移送も行わない = 二重移送構造的に不可能）
- **ownership duplication: なし**（exchange で取り外した ptr を 1 度だけ enqueueWithRetry → 失敗なら
  1 度だけ quarantineRetire。B-2 の T2/T7 runtime 実証と一致）

**判定: PASS**

---

## C4. DSP `ignoreUnused(result)` の最終判定

`DSPLifetimeManager.cpp`:
- `retire()` (L33-63): `engine_.retireDSPHandleForRuntime()` 成功後 → `router_->enqueueWithRetry(...)` → `juce::ignoreUnused(result)`
- `retireByHandle()` (L65-107): handle map から取出し → `retire(handle)` + `requestReclaimHandle(handle)` → `router_->enqueueWithRetry(toDelete, ...)` → `juce::ignoreUnused(result)`

### 判定表 — result 返却時点での DSP pointer ownership

| Router result | DSP ownership | orphan? |
|---|---|---|
| `Success` | D（DeferredDeletionQueue） | No |
| `QueuePressure` | Q または E（enqueueWithRetry 内部で移送完結） | No |
| `TerminalReclaim` | T（growable store）または immediate delete | No |
| `QueueFull` | 現行 contract 上発生しない（C1/C2 確認） | — |

→ **result が返る時点で ownership は必ず Router 内 authority のいずれかに存在する。**
`ignoreUnused(result)` は orphan を意味しない。コメント（L53-57, L100-101）も
「二重移送（double-quarantine → double-free）を避けるため追加の quarantineRetire を呼ばない」と明示しており、
caller 側の追加 quarantine を行わない設計は正しい。

**判定: PASS** — `ignoreUnused(result)` は削除しない（維持）。DSP は Router 内部移送後に追加 quarantine されない。

---

## C5. RT boundary 確定

`resetFadeStateAndRetireTarget()`（`SnapshotCoordinator.cpp:81-96`）の呼出し系統:

```text
exchangeTarget(nullptr) → publishEpoch() → enqueueRetire(target, snapshotDeleter, retireEpoch) → resetToIdle()
呼び出し元: 0 件（rg 全走査 — 定義 h:138 / cpp:81 とコメント h:150 のみ = dead code 確定）
```

禁止呼出しの有無: `quarantineRetire` / `emergencyQuarantine` / `terminalReclaim` / `enqueueWithRetry`
— **すべて不使用**（grep 確認済み）。

加えて RT 防御の二重化を確認:
- `ISRRetireRouter::enqueueWithRetry` 先頭に `jassert(!convo::numeric_policy::isAudioThread())`（L313）
- `terminalReclaim` は RT caller を検出すると即時削除せず store に回す（L519-532）
- RT caller の正式経路は `retireRT()` → `enqueueRetire()`（D のみ、lock-free）（L282-288）

```text
確定:
  RT     ── D only（enqueueRetire / retireRT）
  NonRT  ── D → Q → E → T（enqueueWithRetry）
```

**判定: PASS**（patch 対象外遵守 — friend/public 化なし）

---

## C6. Shutdown の分離確認

shutdown ownership は通常 retire path とは別の**明示的経路**で conserved される:

```text
[通常 path]
  enqueueDeferredDeleteNonRtWithResult
      └─ !isShutdownInProgress() → markRetireEpoch → router->enqueueWithRetry()
           （enum Shutdown は返らない — C2 確認）

[shutdown path]
  isShutdownInProgress() == true
      └─ markRetireEpoch() → router->shutdownReclaim(ptr, deleter, epoch, type)   (AudioEngine.h:4212-4221)
           └─ terminalReclaim(..., "shutdownReclaim")                              (ISRRetireRouter.cpp:574-584)
                ├─ epoch safe（Audio Thread 停止済みで通常こちら）→ 即時破棄
                └─ epoch unsafe → T store（drainAll()/drainAllQuarantineStore() で shutdown 時強制解放）
```

- shutdownReclaim の production caller は `AudioEngine.h:4219` の 1 箇所のみ（rg 確認）
- `Shutdown` enum は「shutdown 中に enqueue が来た場合の wrapper 戻り値」であり、
  `enqueueWithRetry()` の state machine には混入しない
- shutdown drain は `drainAll()`（L593-601）が provider drain + `drainAllQuarantineStore()`（Q+E+T 全強制解放）で完結

**判定: PASS** — shutdown ownership は別経路で conserved、混同なし。

---

## C7. 最終 Disposition Matrix（確定版）

```text
Object admitted (exactly 1 owner at every step)
      │
      ├─ [D] enqueueRetire Success
      │    └─ D owns .......................................... owner count = 1
      │
      ├─ [D pressure] enqueueRetire == QueuePressure
      │    ├─ retry Success（×2 ループ内）
      │    │    └─ D owns ..................................... owner count = 1
      │    │
      │    └─ retry exhausted
      │         └─ Q (RetireQuarantineStore, cap 512)
      │              ├─ stored == true → Q owns ................ owner count = 1
      │              └─ full（deleter 実行せず false）
      │                   └─ E (EmergencyQuarantineStore, cap 512)
      │                        ├─ estored == true → E owns ..... owner count = 1
      │                        └─ full（deleter 実行せず false）
      │                             └─ T (TerminalReclaimAuthority, growable)
      │                                  ├─ epoch safe && NonRT
      │                                  │    └─ deleted ....... owner count = 0 (destroyed)
      │                                  └─ otherwise
      │                                       └─ T owns ........ owner count = 1
      │
      └─ [shutdown] isShutdownInProgress()
           └─ shutdownReclaim → terminalReclaim
                ├─ epoch safe → deleted ........................ owner count = 0 (destroyed)
                └─ epoch unsafe → T owns → shutdown drain ...... owner count = 1 → 0
```

補足:
- `QueueFull`: 上記 matrix に現れない（実効的な terminal disposition ではない — C8 条件 8）
- `Snapshot fallback`: exchange → enqueueWithRetry fail → quarantineRetire **1 回のみ**（T2/T7 実証 + C3 source 裏付け）
- 各遷移で直前 owner は ptr を手放し（exchange / move / push_back）、二重所属は構造的に発生しない

---

## C8. 判定基準マッピング

| # | PASS 条件 | 判定 | 根拠 |
|---|---|---|---|
| 1 | D/Q/E/T 遷移で ownership 消失なし | ✅ | C1（stored/estored/tstored 実コード確認） |
| 2 | 同一 ptr の同時二重所属なし | ✅ | C1/C3/C4（exchange→単一移送、B-2 T7/T8 実証） |
| 3 | Snapshot fallback の Q 単回移送 | ✅ | C3（quarantineRetireSink 実体、B-2 T2 実証） |
| 4 | DSP の追加 quarantine なし | ✅ | C4（ignoreUnused 維持、コメント契約と一致） |
| 5 | Terminal が最終受け皿 | ✅ | C1 Stage 5（growable store 常時受領） |
| 6 | RT caller が Q/E/T に到達しない | ✅ | C5（jassert + retireRT D-only、resetFade dead code） |
| 7 | shutdown path の分離 | ✅ | C6（shutdownReclaim 明示経路 1 caller） |
| 8 | QueueFull は実効 disposition ではない | ✅ | C1/C2（D が QueueFull を返さず、T が常に受領） |
| 9 | production source 変更不要 | ✅ | 本 audit は read-only、変更 0 |

**総合判定: PASS（9/9）**

FAIL/要修正項目: **0 件**
（enqueueWithRetry の無所有権 return / 二重移送 / quarantine 重複 / Terminal 移送後の caller 保持 /
RT からの mutex・allocation 経路到達 — すべて検出されず）

---

## DEFERRED（変更なし・確定事項）

| 項目 | 状態 |
|---|---|
| `QueueFull` enum 値 | dead code のまま維持。D2-1 bounded Terminal 設計時に再評価 |
| `Shutdown` enum in enqueueWithRetry | L383 フォールバックとして存在するが到達不能。削除も追加も本次期は行わない |
| `resetFadeStateAndRetireTarget()` | dead code（呼出し 0）のまま維持。patch 対象外 |
| DSPLifetimeManager 抽象化 | 引き続き禁止。T9 は source audit 方式へ分解済み（本報告 C4 がその一部） |

---

## 次ステップ

D8-2-C 完了。disposition contract は最新ソースで実証・裏付け完了。
以降は D2-1（bounded Terminal 設計 gate）または T9 残部（DSPLifetimeManager source audit の詳細化）へ進行可能。
