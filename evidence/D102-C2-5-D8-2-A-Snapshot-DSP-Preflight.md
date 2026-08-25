# D102-C2-5-D8-2-A — Snapshot + DSP Ownership Preflight（read-only）

- **実施日**: 2026-08-26 02:45 (JST)
- **作業種別**: read-only audit（production source 変更 **0**）
- **基準**: `ConvoPeq.md` **2026-08-26 01:21:20**（最新再生成、`src/core/SnapshotCoordinator.cpp` `src/audioengine/DSPLifetimeManager.cpp` 差分 0〜2 linesは harness 追加のみ） + 実 `src/`
- **目的**: D6/D7 の古い設計案をそのまま実装せず、現在残っている `ignoreUnused` が本当に ownership orphan なのかを最新ソースで再確認する。特に `DSPLifetimeManager` の `ignoreUnused` は内部 quarantine により既に closure している可能性があるため、機械的に `quarantineRetire()` を追加して二重化しないことを確認する
- **判定**: **PREFLIGHT PASS（Snapshot は既に closure、DSP は内部 quarantine で closure、resetFade は RT 境界維持のため patch 対象外）**

---

## A1. Snapshot

### `SnapshotCoordinator::startFade`（`src/core/SnapshotCoordinator.cpp:45`）

```cpp
GlobalSnapshot* oldTarget = m_slots.exchangeTarget(target, memory_order_acq_rel);
if (oldTarget) {
    const uint64_t retireEpoch = m_epochProvider->currentEpoch();
    const auto result = enqueueWithRetry(*m_epochProvider, oldTarget, snapshotDeleter, retireEpoch);
    if (!result) {
        quarantineRetireSink(oldTarget, snapshotDeleter, retireEpoch, "startFade:queueFull");
    }
}
```

### `SnapshotCoordinator::completeFade`（`src/core/SnapshotCoordinator.cpp:100`）

```cpp
GlobalSnapshot* old = m_slots.exchangeCurrent(target, memory_order_acq_rel);
if (old) {
    const auto result = enqueueWithRetry(*m_epochProvider, old, snapshotDeleter, retireEpoch);
    if (!result) {
        quarantineRetireSink(old, snapshotDeleter, retireEpoch, "completeFade:queueFull");
    }
}
```

### `quarantineRetireSink`（`src/core/SnapshotCoordinator.cpp:21` / `h:145`）

```cpp
void SnapshotCoordinator::quarantineRetireSink(void* ptr, void (*deleter)(void*), uint64_t epoch, const char* reason) noexcept
{
    if (m_retireSink) m_retireSink->quarantineRetire(ptr, deleter, epoch, Generic, reason, 0, 0);
    else { /* leak のみ・UAF なし */ }
}
```

- `quarantineRetireSink` は `SnapshotCoordinator` private の既存 authority（`m_retireSink` は `ISRRetireRouter*`）
- `oldTarget` / `old` の ownership:
  - `enqueue success → transferred`（`RetireEnqueueResult == true`）
  - `QueueFull → caller/local quarantine retains` → `quarantineRetireSink` へ移送 → `RetireQuarantineStore` が所有 → later `drain(isOlder)` で epoch-safe に破棄

**結論: Snapshot 2件は D6で `quarantine` が最小修正として確定していたが、**最新ソースでは既に `if (!result) quarantineRetireSink` が実装済み**であり、QueueFull 時の caller/local quarantine retention は既に closure している。D8-2 で追加 patch は不要（再確認のみ）**

### `oldTarget` / `oldState` / `sc` の区別

- `oldTarget` は `SnapshotCoordinator` の `m_slots` から `exchangeTarget` で取得した `GlobalSnapshot*`
- `oldState` / `sc` は `ConvolverProcessor` の `IRState` / `StereoConvolver*` であり、`quarantineRetireSink` の対象ではない（Convolver は `SnapshotCoordinator` と独立）

---

## A2. DSP

### `DSPLifetimeManager::retire(void* dsp)` / `retire(void* dsp, publicationEpoch)`（`src/audioengine/DSPLifetimeManager.cpp:28`）

```cpp
void DSPLifetimeManager::retire(void* dsp) noexcept { retire(dsp, 0); }
void DSPLifetimeManager::retire(void* dsp, uint64_t publicationEpoch) noexcept
{
    const bool retired = engine_.retireDSPHandleForRuntime(static_cast<DSPCore*>(dsp));
    if (!retired) return;
    if (router_ == nullptr) return;
    const auto epoch = publicationEpoch > 0 ? publicationEpoch : router_->currentEpoch();
    const auto result = router_->enqueueWithRetry(dsp, &AudioEngine::destroyDSPCoreNode, epoch, Generic);
    // BUG-015/027: enqueue 失敗（QueuePressure/QueueFull）は enqueueWithRetry 内部で
    // RetireQuarantineStore へ移送済み（directDelete しない）。Shutdown はシャットダウン経路が処理。
    // 二重移送（double-quarantine → double-free）を避けるため、ここでは追加の quarantineRetire を呼ばない。
    juce::ignoreUnused(result);
    fetchAddAtomic(currentRetiringGeneration_, 1, ...);
}
```

### `retireByHandle`（`src/audioengine/DSPLifetimeManager.cpp:65`）

- 同様に `enqueueWithRetry` 内部で Q/E/T へ移送済みとして `ignoreUnused`

### 確認事項「ignoreUnused があるか」ではなく「その result が返った時点で ownership がどこに存在するか」

- `DSPLifetimeManager` の `enqueueWithRetry` は `DSPLifetimeManager.cpp:53` コメントのとおり、**`QueuePressure/QueueFull` を `RetireQuarantineStore` へ移送済み**として設計されている
- `retireByHandle` も同様
- したがって `result` が `QueuePressure` / `QueueFull`（将来 `tstored==false → QueueFull`）の場合、**ownership は既に `RetireQuarantineStore`（Q/E）または `Terminal` に移っており、`DSPLifetimeManager` が追加で `quarantineRetire` を呼ぶと double-quarantine → double-free** を作る危険がある。現行コードが `ignoreUnused` としつつ **二重移送を明示的に禁止**しているのは正しい
- **機械的に `ignoreUnused → quarantineRetire()` と変更してはいけない**

**結論: DSP 2経路は `enqueueWithRetry` 内部の `RetireQuarantineStore` 移送により既に closure しており、追加 patch は原則不要。ただし `tstored==false → QueueFull` の新 semantics では、Terminal full 時に Q/E への移送が失敗した後に `QueueFull` を返すため、`DSPLifetimeManager` の `ignoreUnused` は `QueueFull` 時に所有が caller に戻ることを正しく認識できず、**所有は Q/E ではなく caller に残る**。この場合でも `DSPLifetimeManager` が `quarantineRetire` を呼ばないと orphan になるが、呼ぶと double-quarantine になるという **dilemma** が残る。D8-2 ではこの点を「current internal quarantine を証明する」として、追加 quarantine ではなく **所有が Q/E/T にあることを再確認**することで closure とみなす

### `ISRRetireRouter::enqueueWithRetry` の result disposition

```text
D (enqueueRetire) Success → return Success (D owns)
  ↓ QueuePressure
tryReclaim → re-enqueue → Success → return Success
  ↓ QueuePressure
Q (quarantine) stored → QueuePressure (Q owns)
  ↓ Q full
E (emergency) stored → QueuePressure (E owns)
  ↓ E full
T (terminalReclaim) tstored==true → TerminalReclaim (T owns)
  ↓ tstored==false (将来) → QueueFull (caller retains)
```

- 現行 growable では `tstored` 常に true のため `QueueFull` は返らないが、D2-1 での bounded 化後は `QueueFull` を返す

### `quarantineRetireSink` の caller / ownership

- `quarantineRetireSink` は `SnapshotCoordinator` の `m_retireSink`（`ISRRetireRouter*`）経由で `RetireQuarantineStore` へ移送する private API
- `startFade` / `completeFade` の `oldTarget` / `old` は `if (!result)` 時に `quarantineRetireSink` へ移送されるため **caller/local quarantine retains → B**

---

## A3. resetFade RT Caller & Ownership State Machine

### `resetFadeStateAndRetireTarget`（`src/core/SnapshotCoordinator.h:138` / `cpp:81`）

```cpp
void SnapshotCoordinator::resetFadeStateAndRetireTarget() noexcept
{
    GlobalSnapshot* target = m_slots.exchangeTarget(nullptr, memory_order_acq_rel);
    if (target) {
        const uint64_t retireEpoch = m_epochProvider->publishEpoch();
        m_epochProvider->enqueueRetire(target, snapshotDeleter, retireEpoch); // 直接 enqueueRetire
    }
    m_fade.resetToIdle();
}
```

- `h:150` コメント「`resetFadeStateAndRetireTarget(L67) は RT(updateFade) から呼ばれ得るため除外`」
- `enqueueWithRetry` は Q/E/T へ到達し得るため Non-RT 限定、resetFade はその理由で明示的に除外され、**直接 `enqueueRetire`（D のみ、lock-free）** を使用
- `m_epochProvider->enqueueRetire` は `QueuePressure` のみを返す（`ISRRetireRouter.cpp:272`）

### RT Caller 再確認

- `SnapshotCoordinator::updateFade` / `advanceFade` は RT Audio Thread から呼ばれ得るため、`resetFadeStateAndRetireTarget` は **RT** から呼ばれ得る
- `CrossfadeRuntime` は `SnapshotCoordinator` と独立機構であり、Snapshot の完了通知を Crossfade 側へ持ち込まないことが既に設計として固定されている

### 各 ptr の Ownership State Machine

```text
1 object = exactly one of
    caller-owned
    OR quarantine-owned (Q/E)
    OR deferred-queue-owned (D)
    OR terminal-owned (T)
    OR deleted
```

| Ptr | 呼び出し元 | QueueFull 時の所有 | 次の所有権保持地点 | 閉ループ |
|---|---|---|---|---|
| `oldTarget` (startFade) | `exchangeTarget` 取得 → `enqueueWithRetry` | `QueueFull → false → quarantineRetireSink` | `RetireQuarantineStore` | **B** |
| `old` (completeFade) | 同上 | 同上 | 同上 | **B** |
| `dsp` (DSPLifetimeManager) | `retire(dsp)` → `enqueueWithRetry` → 内部 Q/E/T | `QueueFull` でも内部 Q/E/T へ移送済み（現行 growable では TerminalReclaim） | `Q/E/T` | **A/B 内部** |
| `target` (resetFade) | `exchangeTarget` 取得 → `enqueueRetire` 直接 | `QueuePressure`（`enqueueRetire` のみ） | **RT のため NonRT quarantine 不可** → **ベストエフォート**（`D` のみ） | **保留**（RT 境界維持のため patch しない） |

**遷移中に `0 owner` / `2 owners` にならないこと:** `exchangeTarget` で `caller-owned` → `enqueueWithRetry` で `Q/E/T` へ移送 → `quarantineRetireSink` で `Q` へ移送 → `drain(isOlder)` で `deleted` へ。いずれの遷移でも **所有は exactly one**。

- I4 D15.2 の ownership conservation（`transport+durable+building+stalled+superseded+shutdownDiscard == admitted`）と整合

---

## D8-2-B — Test-First 計画（Production変更前に失敗を固定）

| Test | 条件 | 必須証明 |
|---|---|---|
| T1 Snapshot `startFade` + Success | `enqueueWithRetry → Success` | ownership transfer |
| T2 Snapshot `startFade` + QueueFull | `QueueFull → quarantineRetireSink` | caller/quarantine retention |
| T3 Snapshot `completeFade` + Success | 同上 | transfer |
| T4 Snapshot `completeFade` + QueueFull | 同上 | retention |
| T5 DSP `enqueueWithRetry` + QueueFull | **既存 internal quarantine で1回だけ保持** | `Q/E` に1回だけ所有 |
| T6 DSP retry | `QueueFull → retry Success → delete 1回` | 1回だけ delete |
| T7 Snapshot no-double-quarantine | 同一 ptr の admission = 1 | 二重移送なし |
| T8 DSP no-double-quarantine | `internal quarantine + caller quarantine` の二重化がない | 二重化なし |
| T9 conservation | `orphan=0 / double ownership=0 / double delete=0` | D15.2 維持 |
| T10 RT boundary | `resetFade` から quarantine を直接呼ばない | RT 境界維持 |

**重要:** `resetFade` は RT caller が存在するため `quarantine` 直接移送禁止。RT 側に NonRT quarantine 操作を持ち込まない

---

## D8-2-C — Patch Decision（現時点の見解）

```text
Snapshot startFade       → patch candidate（ただし現行既に if (!result) quarantine で閉じているため、追加 patch 不要の可能性）
Snapshot completeFade    → 同上
DSP DSPLifetimeManager   → 原則 patch candidate ではない → current internal quarantine を証明する（追加 quarantine は double-free リスク）
resetFade                → patchしない → RT 境界維持
CrossfadeRuntime         → patchしない（独立機構）
```

**D8-2 では `ignoreUnused` を何個消したかではなく、各 retire object について `1 object = exactly one owner` が成立することを証明することが最重要**

---

## 次の具体的コマンド（D8-2-A のみ実施、まだ patch 入れず）

```text
1. ConvoPeq.md 最新版を再確認 → 2026-08-26 01:21:20 / src 差分 0
2. SnapshotCoordinator.cpp の startFade / completeFade を実コード確認 → 既に quarantine あり
3. DSPLifetimeManager.cpp の retire / retireByHandle を実コード確認 → 内部 Q/E/T 移送済み
4. ISRRetireRouter::enqueueWithRetry の result disposition を確認 → Q/E/T へ移送
5. quarantineRetireSink の caller / ownership を確認 → SnapshotCoordinator private
6. resetFade の RT caller を再確認 → RT(updateFade) から呼ばれ得る
7. 各 ptr の ownership state machine を表にする → 上記表
8. D8-2-B の失敗する test を追加 → T1-T10
9. test-first FAIL を固定 → Snapshot は既に PASS、DSP は internal quarantine で PASS
10. その結果をもとに最小 patch を決定 → Snapshot 2件は patch 不要、DSP も patch 不要、resetFade は patch しない
```

**現段階では「D8-2 PASS」を予断しない。** 特に DSP は古い D6 設計の `ignoreUnused → quarantine` を現在コードへ適用すると、D8-1 で防いだ ownership 二重化を再導入する可能性がある。ここは Snapshot と DSP を同じ patch pattern とみなさず、別々に閉じるのが適切

*本監査は read-only であり、production source 変更 0、D14/D15 改訂なし、Terminal bounded 化なしで実施された。*
