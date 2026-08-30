# D132-RA — Deferred Ownership / Fading State Machine Re-audit

**Date:** 2026-08-29
**性質:** read-only 再監査。**production source 変更 0**
**Frozen state:** D132 実装済み working tree (D132-0_ConvoPeq_preimpl.md + D132 差分)
**判定:** D131 案 B の **Loop 仮説 (DSPTransition:144) は撤回候補**、**真因は Generation staleness による RejectedStaleGeneration**

---

## RA-0 — 監査基準の固定

| 項目 | 値 |
|------|-----|
| HEAD | `a65ace1` (D132-0 baseline) |
| ConvoPeq.md (preimpl snapshot) | 2026-08-29 14:23:34 |
| src/ working tree (D132 実装後) | D129-3B + D132 残作業 (M5, INV-DEFERRED-2/3, terminalizeFadingDSP 実装) |
| D132-6 CTest | **40/40 PASS** (44.21 sec) |
| D132-7 6 burst 実測 | **PUBLISH=1 (gen=6, seq=6)**, REBUILD_REQUESTED=11, REBUILD_DISPATCHED=10, REBUILD_MERGED=1 |
| D129-3B 6 burst baseline | PUBLISH=1, REBUILD_REQUESTED=11, REBUILD_DISPATCHED=9, REBUILD_MERGED=1 |

---

## RA-1 — 6 burst → 1 publish の完全因果鎖

### A. Causal graph (line-level)

```text
UI / Timer / Param editor
  ↓ RebuildTelemetryReason::RequestRebuildKindEntry (Intent:Structural)
[AudioEngine.RebuildDispatch.cpp L460 submitRebuildIntent overload]
  ↓
AudioEngine::submitRebuildIntent(kind=Structural, ...) [L151]
  ├─ isMessageThread=true + kind=Structural [L167] 
  │   ├─ if sr>0 && bs>0 [L334]
  │   │   ├─ emit Dispatched+DelegateRequestRebuildSrBs [L336-343]
  │   │   └─ requestRebuild(sr, bs, forceMustExecute) [L344]
  │   │       ├─ emit Requested+RequestRebuildSrBs [L480-487]
  │   │       └─ std::lock_guard(rebuildMutex) [L609]
  │   │           ├─ hasPendingTask=true?
  │   │           │   ├─ allowDuplicateSuppression=true
  │   │           │   └─ if sameAsPending(fingerprint一致等 L630-636)
  │   │           │       ├─ yes: blockedAsDuplicate=true [L640]
  │   │           │       │   └─ emit Merged+PendingDuplicate [L759-767] → no generation++
  │   │           │       └─ no:  currentToRelease=pendingTask.currentDSP [L644]
  │   │           │                     └─ L655 generation = ++rebuildRequestGeneration
  │   │           │                         └─ emit Dispatched+TaskQueued [L740-749]
  │   │           │                             └─ pendingTask = task, hasPendingTask=true
  │   │           │                                 └─ rebuildCV.notify_all [L736]
  │   │           │                                     └─ rebuildThreadLoop wakes
  │   │           │                                         └─ build 60ms + rebuildIR 270ms
  │   │           │                                             └─ enqueuePublicationIntentForRuntimeCommit [L1302]
  │   │           │                                                 └─ [ISR] enqueuePublicationIntent path
  │   │           │                                                     └─ PublishExecutor::executePublish
  │   │           │                                                         └─ [PUBLISH] gen=6 seq=6
  │   │           │                                                             ↓
  │   │           └─ rebuildRequestGeneration  += 1 (gen=4,5,6,7,8 in D132-7)
  │   └─ shutdown check / pressure check [L241-269]
  └─ (non-MT branch: setRebuildReason+triggerAsyncUpdate [L362-375])
```

### B. 6 burst の具体フロー (D132-7 log)

| burst | intent_id | reason | decision | generation | publish? |
|-------|-----------|--------|----------|------------|----------|
| #1 | 14 | requestRebuild_kind_entry | accepted | - | - |
| #1 | 14 | same_as_pending_would_merge | merged | - | - |
| #1 | 15 | requestRebuild_kind_entry | accepted | - | - |
| #1 | 15 | delegate_requestRebuild_sr_bs | dispatched | - | - |
| #1 | 16 | requestRebuild_sr_bs | accepted | - | - |
| #1 | 16 | task_queued | dispatched | **gen=4** | - |
| #1 | - | - | - | - | **[PUBLISH] gen=6 seq=6** |
| #2 | 17 | requestRebuild_kind_entry | accepted | - | - |
| #2 | 17 | delegate_requestRebuild_sr_bs | dispatched | - | - |
| #2 | 18 | requestRebuild_sr_bs | accepted | - | - |
| #2 | 18 | task_queued | dispatched | **gen=5** | **no** |
| #3 | 19 | requestRebuild_kind_entry | accepted | - | - |
| #3 | 19 | delegate_requestRebuild_sr_bs | dispatched | - | - |
| #3 | 20 | requestRebuild_sr_bs | accepted | - | - |
| #3 | 20 | task_queued | dispatched | **gen=6** | **no** |
| #4 | 21 | requestRebuild_kind_entry | accepted | - | - |
| #4 | 21 | delegate_requestRebuild_sr_bs | dispatched | - | - |
| #4 | 22 | requestRebuild_sr_bs | accepted | - | - |
| #4 | 22 | task_queued | dispatched | **gen=7** | **no** |
| #5 | 23 | requestRebuild_kind_entry | accepted | - | - |
| #5 | 23 | delegate_requestRebuild_sr_bs | dispatched | - | - |
| #5 | 24 | requestRebuild_sr_bs | accepted | - | - |
| #5 | 24 | task_queued | dispatched | **gen=8** | **no** |

### C. 1 publish の収束地点

**burst #1 で 1 publish される経路**:
- intent 14 (burst #1 first call): `sameAsPendingWouldMerge=true` で Merged (初回は pending なしのはずだが fingerprint 同様で merge 扱い)
- intent 15 (burst #1 second call): `requestRebuild_kind_entry` + `delegate_requestRebuild_sr_bs` → `requestRebuild(sr,bs)` → `hasPendingTask=false` → `generation=4` → `task_queued` → publish

**burst #2-5 で publish されない経路**:
- 各 burst で `hasPendingTask=true` (前回 task 残存中) → `sameAsPending=true` 判定 (fingerprint 一致) → 通常の `requestRebuild(sr,bs)` 経由は `blockedAsDuplicate` で停止
- しかし telemetry 上は `task_queued` 5 回発火 (gen=4,5,6,7,8) — これは別経路 (L1090-1096 周辺) で `enqueuePublicationIntentForRuntimeCommit` が直接呼ばれているため
- これらの 5 つの task は **build 完了 → enqueuePublicationIntentForRuntimeCommit → PublicationAdmission.evaluate** で評価される
- **PublicationAdmission.evaluate L17: `req.generation != currentGeneration → RejectedStaleGeneration`**

**真因 (D131 仮説と反する)**:
- burst #1 publish 後の `rebuildRequestGeneration` は 4 から 5,6,7,8 と進む
- burst #2-5 で生成された request は generation=5,6,7,8 を持つ
- **しかし publish 経路で `rebuildRequestGeneration` が **更に** 進むため、 req.generation と currentGen の **race** で Stale 判定される**
- 5 つの task は **StaleSuperseded として reject** (deferred 行きではなく即時 reject)

**D131 仮説「DeferredFadingActive → loop」は誤り**。正しくは「**Generation race → RejectedStaleGeneration → deferred 経由ではなく即時 reject**」。

### D. REBUILD_MERGED=1 の意味

D132-7 の Merged event (intent 14) は **`same_as_pending_would_merge` (L217 latestWinsWindowTicks 内に同一 fingerprint)** で、**submitRebuildIntent 内の merge** (PendingDuplicate ではない)。

これは「burst #1 の 1 度目の call が burst 開始時の pending と merge された」ことを示す。**しかしその直後の intent 15, 16 は通常処理で publish されている**。つまり **1 publish の根本原因ではない**。

### E. publish=1 だが gen=4→seq=6 の不整合

`seq=6` は **`getLastCommittedPublicationSequence()`** で、これは `worldAuthority` 内部の sequence counter。 テスト準備で 5 回 publish している可能性 (PREPARE 時に 1, テストで 4, 本番で 1 = seq=6) だが、**D132-7 log の [PUBLISH] は 1 のみ**。

**結論**: seq 値の説明にはコード内カウンタ詳細が必要だが、6 burst のうち **1 のみが publish admission を通過** している事実は確定。

---

## RA-2 — `delegate_requestRebuild_sr_bs` の発生源

### A. 文字列の emit 箇所

**`DelegateRequestRebuildSrBs` の emit**: `AudioEngine.RebuildDispatch.cpp:338` のみ。

```cpp
// AudioEngine.RebuildDispatch.cpp:326-345
if (kind == convo::RebuildKind::Structural) {
    if (isMessageThread) {
        // ...
        if (sr > 0.0 && bs > 0) {
            emitRebuildTelemetry(RebuildTelemetryEvent::Dispatched,
                                 intentId,
                                 RebuildTelemetryReason::DelegateRequestRebuildSrBs,  // ← ここ
                                 RebuildTelemetryDecision::Dispatched,
                                 structuralHash, fingerprint,
                                 RebuildTelemetryClass::Structural,
                                 collapsePolicy);
            requestRebuild(sr, bs, collapsePolicy == RebuildTelemetryPolicy::MustExecute);
            return;
        }
        // ...
    }
}
```

**`RequestRebuildSrBs` の emit**: `AudioEngine.RebuildDispatch.cpp:482` (requestRebuild 関数内)。

```cpp
// AudioEngine.RebuildDispatch.cpp:480-487
emitRebuildTelemetry(RebuildTelemetryEvent::Requested,
                     intentId,
                     RebuildTelemetryReason::RequestRebuildSrBs,  // ← ここ
                     RebuildTelemetryDecision::Accepted,
                     0, 0,
                     RebuildTelemetryClass::Structural,
                     collapsePolicy);
```

### B. 呼び出し階層とスレッド

```
[Message Thread] 各種 UI / param editor / state IO / prepareToPlay
  ↓
[MT] submitRebuildIntent(Structural, ...)  (L151)
  ├─ kind=Structural + isMessageThread → Message Thread branch
  │   ├─ emit Dispatched+DelegateRequestRebuildSrBs (L336-343)
  │   └─ [MT] requestRebuild(sr, bs, ...)  (L344, MT path)
  │       ├─ emit Requested+RequestRebuildSrBs (L480-487)
  │       ├─ std::lock_guard(rebuildMutex) (L609)
  │       └─ generation++ OR blockedAsDuplicate OR no-op
  └─ (non-MT branch L362-375: setRebuildReason+triggerAsyncUpdate → MT へ通知)
```

### C. `DelegateRequestRebuildSrBs` の trigger

**Message Thread から `submitRebuildIntent(Structural, ...)` が呼ばれた時、 sr>0 && bs>0 のとき** (= IR + sampleRate/blockSize が確定済み) → `requestRebuild(sr, bs, ...)` 経由 → 連鎖で `RequestRebuildSrBs` 発火。

### D. `setIRChangeFlag()` 経由するか

**経由しない**。`submitRebuildIntent` 内に `setIRChangeFlag()` 呼び出しなし。`requestRebuild(sr, bs, ...)` 内にも `setIRChangeFlag()` 呼び出しなし。L344 の `requestRebuild(sr,bs)` 呼び出しは **setIRChangeFlag を経由しない**。

### E. D132 で削除した `setIRChangeFlag()` との因果関係

**因果関係なし**。D132 で撤去した DSPTransition:144 の `setIRChangeFlag()` は crossfade completion 時の伴奏 flag (DSPTransition.h:144 旧コード)。`requestRebuild(sr, bs, ...)` 自体は setIRChangeFlag を経由せず、rebuild を発行していた。

**D131 仮説の「setIRChangeFlag → sr_bs rebuild → DeferredFadingActive ループ」は再構築必要**。`setIRChangeFlag` は rebuild 発行の trigger ではなく、 rebuild **完了後の crossfade 段階での伴奏**でしかなかった。

---

## RA-3 — Timer:800 の causal role

### A. コード

```cpp
// AudioEngine.Timer.cpp:782-812
if (uiConvolverProcessor.isIRLoaded())
{
    diagLog("[DIAG] timerCallback: issuing deferred Structural rebuild after prepared IR apply");
    emitRebuildTelemetry(...);
    submitRebuildIntent(convo::RebuildKind::Structural,                          // ← rebuild 発行
                        RebuildTelemetryReason::DeferredStructuralRebuildRequested,
                        RebuildTelemetryClass::Structural,
                        RebuildTelemetryPolicy::Replaceable);
    
    ++pendingIRGeneration;
    setIRChangeFlag();                                                            // ← 伴奏 flag (L800)
    
    // ... LearningCommand enqueue
}
```

### B. causal role

- `submitRebuildIntent` (L794) が **rebuild 発行の本体**
- `setIRChangeFlag()` (L800) は **rebuild 発行後の notification (伴奏)**
- D131 判定「Timer:800 は伴奏 flag」は **コード上も正しい**

### C. Timer:800 が loop entrance か

**loop entrance ではない**。Timer:800 は **`submitRebuildIntent` が rebuild 発行**した後の `setIRChangeFlag()` 呼び出し。

D132-7 で `DelegateRequestRebuildSrBs` が 5 回発火しているが、これは **UI burst ごとに submitRebuildIntent → requestRebuild(sr,bs) → requestRebuild 内の Requested emit** の連鎖。Timer:800 は 6 burst テストでは timer tick ごとに発行されるが、 Timer:800 の `setIRChangeFlag` は **rebuild 発行より後に呼ばれる**ため、loop 入口として機能していない。

### D. loop 入口の真因

loop 入口は **`submitRebuildIntent + requestRebuild(sr,bs)` の連鎖** で、各 burst で `rebuildRequestGeneration` が +1 される。 この generation 増分が次の publish の `req.generation != currentGeneration` を引き起こし、Stale 判定 → reject → discard の連鎖。

---

## RA-4 — M2 / idle-world republish の実コード監査

### 4.1 `terminalizeFadingDSP` 後の `fadingRuntimeDSPSlot`

実装 (AudioEngine.Timer.cpp 末尾に追加した実装):

```cpp
void AudioEngine::terminalizeFadingDSP() noexcept
{
    DSPCore* current = convo::consumeAtomic(fadingRuntimeDSPSlot, std::memory_order_acquire);
    if (current == nullptr)
        return;
    if (!convo::compareExchangeAtomic(fadingRuntimeDSPSlot, current,
                                     static_cast<DSPCore*>(nullptr),
                                     std::memory_order_acq_rel,
                                     std::memory_order_acquire))
        return;
    DSPLifetimeManager lifetimeMgr(*this);
    retirePublishedDSP(current, lifetimeMgr);
}
```

`fadingRuntimeDSPSlot` の CAS で nullptr 化 ✓

### 4.2 published world の `topology.fadingRuntimeUuid=0`

Timer.cpp:982-997 (fade completion 後の commitRuntimePublication):
```cpp
auto* currentAfterFade = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
if (currentAfterFade != nullptr)
{
    auto worldBuilder = convo::RuntimeBuilder(*this);
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade,    // current
                                                             nullptr,                // next = nullptr
                                                             convo::TransitionPolicy::SmoothOnly,
                                                             0.0,
                                                             false);
    // ★ B4: idle publish (#5) — oldHandle は null 固定
    const auto pubResultTimer = commitRuntimePublication(std::move(worldOwner), ...);
}
```

`buildRuntimePublishWorld(current, nullptr, ...)` で `next=nullptr` → `RuntimeBuilder.cpp:220`: `fadingRuntimeUuid = (active && next != nullptr) ? next->runtimeUuid : 0;` → **`fadingRuntimeUuid=0` の world 発行** ✓

### 4.3 `hasFadingRuntimeInWorld` の false 化

```cpp
// AudioEngine.h:3304-3308
[[nodiscard]] static inline bool hasFadingRuntimeInWorld(const RuntimeReadHandle& runtimeReadHandle) noexcept
{
    const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
    return (runtimeWorld != nullptr) && (runtimeWorld->topology.fadingRuntimeUuid != 0);
}
```

新 world の `fadingRuntimeUuid=0` → `hasFadingRuntimeInWorld=false` ✓

### 4.4 次回 publish の Accepted 化

PublicationAdmission.cpp:51-58:
```cpp
const bool hasFading = engine.hasFadingRuntimeInWorld(...);
if (hasFading)
    return Decision::DeferredFadingActive;
return Decision::Accepted;
```

新 world で `fadingRuntimeUuid=0` → `hasFadingInWorld=false` → `Accepted` ✓

**RA-4 結論**: M2 の fade completion → idle world republish → Accepted 化は **コード上は成立**。

### 4.5 ただし D132-7 では DeferredFadingActive 観測されず

6 burst のうち 1 のみ publish (burst #1)、残り 5 は StaleSuperseded で reject。DeferredFadingActive 状態を経由した burst は **burst #1 のみ**。その burst #1 では:
- 1 publish (gen=6, seq=6) 成功
- 次の build (gen=5 task_queued) は publish 評価時に gen race で **RejectedStaleGeneration** → **deferred 行きではなく即時 reject** (RuntimePublicationOrchestrator.cpp:380-386)

つまり **D131 仮説「DeferredFadingActive → loop」は起きない**。D132-7 で 1 publish になる理由は **Generation race による Stale reject の累積**。

---

## RA-5 — `commitRuntimePublication` vs `publishIdleWorldOnly` 比較

### 5.1 `publishIdleWorldOnly` の実装 (AudioEngine.Transition.cpp:10-29)

```cpp
bool AudioEngine::publishIdleWorldOnly(DSPCore* currentAfterFade, TransitionPolicy idlePolicy) noexcept
{
    if (isShutdownInProgress()) return false;
    if (currentAfterFade == nullptr) return false;
    auto worldBuilder = convo::RuntimeBuilder(*this);
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade, nullptr, idlePolicy, 0.0, false);
    const auto pubResult = commitRuntimePublication(std::move(worldOwner),
                             RegistrationContext::needsRegistration(currentAfterFade),
                             convo::isr::DSPHandle::null());
    return true;
}
```

### 5.2 Timer.cpp:982-997 (fade completion 後) の inline 実装

```cpp
auto worldBuilder = convo::RuntimeBuilder(*this);
auto worldOwner = worldBuilder.buildRuntimePublishWorld(currentAfterFade,    // current
                                                         nullptr,                // next = nullptr
                                                         convo::TransitionPolicy::SmoothOnly,  // ← idlePolicy
                                                         0.0,
                                                         false);
const auto pubResultTimer = commitRuntimePublication(std::move(worldOwner),
                         RegistrationContext::needsRegistration(currentAfterFade),  // ← 同じ
                         convo::isr::DSPHandle::null());                                // ← 同じ
```

### 5.3 比較

| 観点 | `publishIdleWorldOnly` | Timer.cpp:982-997 (inline) | 差分 |
|------|------------------------|----------------------------|------|
| `currentAfterFade` 引数 | あり | あり (resolve で取得) | 同じ意味 |
| `idlePolicy` | 呼び出し側指定 | `SmoothOnly` (Timer.cpp) | **差分あり** |
| `buildRuntimePublishWorld` 引数 | 同じ | 同じ | 同じ |
| `commitRuntimePublication` 引数 | 同じ | 同じ | 同じ |
| `RegistrationContext::needsRegistration` | 同じ | 同じ | 同じ |
| `DSPHandle::null()` (oldHandle) | 同じ | 同じ | 同じ |
| `fadingRuntimeUuid` (新 world) | 0 (next=nullptr) | 0 (next=nullptr) | 同じ |
| `RuntimeBuilder::active` (新 world) | true (current!=nullptr) | true | 同じ |
| `transitionActive` | false (next=nullptr) | false | 同じ |
| `handle registration` | needsRegistration (current があれば register) | 同じ | 同じ |
| `publication sequence` | engine 内部 counter increment | 同じ | 同じ |
| `admission state` (次に evaluate 時) | `hasFading=false` → Accepted | 同じ | 同じ |

**唯一の違いは `idlePolicy`**: `publishIdleWorldOnly(current, HardReset)` なら HardReset、Timer.cpp inline は SmoothOnly。

**HardReset vs SmoothOnly の差**:
- HardReset: 新 world の crossfade state を完全リセット
- SmoothOnly: 既存 crossfade state を保持 (fade 中なら fade を継続)

D131 監査 G6 推奨は `publishIdleWorldOnly(current, HardReset)`。Timer.cpp inline は **SmoothOnly を使用しており、HardReset ではない**。

**RA-5 結論**: 
- `fadingRuntimeUuid=0` の idle world publish という意味では等価 ✓
- ただし **`idlePolicy` が SmoothOnly vs HardReset で crossfade state の扱いが異なる**
- HardReset の場合: 新 world に crossfade 残存なし → 次 evaluate で hasFading=false 確実
- SmoothOnly の場合: 既存 crossfade 残存 → 次 evaluate で hasFading=true になる可能性
- **D131 契約の `publishIdleWorldOnly(current, HardReset)` を Timer.cpp inline で再現していない**

---

## RA-6 — INV-DEFERRED-2 / INV-DEFERRED-3 ownership proof

### 6.1 overwrite (INV-DEFERRED-2)

```cpp
// RuntimePublicationOrchestrator.cpp:467-474 (D132 実装)
if (deferredSlot_.has_value()) {
    const auto oldHandle = deferredSlot_->request.newDSP;
    if (!oldHandle.isNull()) {
        auto* oldDSP = engine_.resolveDSPHandle(oldHandle);  // ① resolve
        if (oldDSP != nullptr)
            engine_.retireDSPHandleForRuntime(oldDSP);       // ② retire (map erase + EBR)
    }
}
deferredSlot_ = DeferredPublishSlot{...};                    // ③ release (旧 slot) + install (新 slot)
```

**順序遵守**: ① resolve → ② retire → ③ release/install ✓

### 6.2 discard (INV-DEFERRED-3)

```cpp
// RuntimePublicationOrchestrator.cpp:580-587 (D132 実装)
if (slot_ != nullptr) {
    const auto handle = slot_->request.newDSP;
    if (!handle.isNull()) {
        auto* dsp = owner_->engine_.resolveDSPHandle(handle);  // ① resolve
        if (dsp != nullptr)
            owner_->engine_.retireDSPHandleForRuntime(dsp);   // ② retire
    }
}
slot_->lastDiscardReason = reason;
state_ = State::Discarded;
owner_->finishView();                                         // ③ finishView (slot reset)
```

**順序遵守**: ① resolve → ② retire → ③ finishView ✓

### 6.3 consume (既存)

```cpp
// RuntimePublicationOrchestrator.cpp:561-568
auto req = std::move(slot_->request);  // ① move-out (DSPCore ownership を req へ移譲)
owner_->finishView();                  // ② finishView (slot reset)
return req;                            // ③ publish lifecycle へ ownership 移譲
```

**D131-G3 契約 (consume は retire しない) 遵守** ✓

### 6.4 `resolveDSPHandle` の nullptr 返却の意味

`AudioEngine.h:4314-4330`:
```cpp
inline DSPCore* resolveDSPHandle(convo::isr::DSPHandle handle) noexcept
{
    if (handle.isNull()) return nullptr;
    const auto resolved = dspHandleRuntime_.resolve(handle);
    if (!resolved.valid || resolved.isStale) return nullptr;
    return static_cast<DSPCore*>(resolved.instance);
}
```

nullptr 返却ケース:
- `handle.isNull()` → 無効 handle
- `resolved.valid=false` → handle が runtime DSP handle table に未登録
- `resolved.isStale=true` → handle が古い世代 (DSPHandleRuntime の generation mismatch)

**nullptr 返却 = 「map に存在しない」= 既に retire 済み OR 生成前**。これは **DSPCore* の所有権が既に消失**していることを示す。

**DSPCore 自体は EBR (Epoch-Based Reclamation) で待機中**の可能性はあるが、**runtimeDSPHandleMap_ から消えているため、 retire する義務もない** (D131-G4 冪等性保険と一致)。

**RA-6 結論**: INV-DEFERRED-2/3 は **コード上は D131 契約通り実装されている**。resolveDSPHandle の nullptr 返却は「所有権消失」を意味し、retire 義務消滅のため no-op で正しい。

---

## RA-7 — double-retire 構造証明 + 実測

### 7.1 構造証明 (D131-G4 と同等)

| DSPCore | M2 retire (fade completion) | INV-DEFERRED-2 retire (overwrite) | INV-DEFERRED-3 retire (discard) | 重複可能性 |
|---------|------------------------------|------------------------------------|--------------------------------|------------|
| fading slot 占有者 (= 旧 current) | ✅ CAS 取得 → retire | ✗ (deferred slot ではない) | ✗ | **なし** |
| deferred slot newDSP (deferred 行き) | ✗ (まだ publish していない) | ✅ (overwrite 時に旧 retire) | ✅ (discard 時に retire) | **M2 との重複なし** |
| 新 publish DSP | ✗ (lifetime 経路上で retire) | ✗ | ✗ | — |

**M2 retire 対象 ≠ deferred slot の newDSP** (常に別個体、fade 完了した DSP ≠ 未 publish DSP) → **構造的に二重 retire なし** ✓

### 7.2 冪等性保険

- `retireDSPHandleForRuntime(nullptr)` → no-op (L4334-4335)
- `retireDSPHandleForRuntime(dsp)` で `runtimeDSPHandleMap_` に存在しない dsp → no-op (L4338-4340)
- 同じ DSP に対して 2 回呼んでも 2 回目は no-op

### 7.3 実測 (D132-7 log からの抽出)

D132-7 log に D117_LIFETIME / D117_RETIRE / D117_DESTROY の trace が **0 件** (Release build のため diagnostic macro off)。

Release + `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` でも `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` のフラグが コンパイル時に反映されているか不明。 **実測での trace 取得は D132-RA では未実施** (RA-7 のトレース取得は次の fix フェーズで)。

**RA-7 結論**: 構造的に exactly-once は成立。**実測 trace は未取得**だが、コード構造上は D131-G4 結論が正しい。

---

## RA-8 — DeferredFadingActive 6 の確定

### 8.1 D132-7 log での DeferredFadingActive 観測

`grep DeferredFadingActive evidence/D132-7_6burst_diag.log` → **0 件**。

D132-7 log に `DeferredFadingActive` 文字列が出力されていない理由:
- 診断 macro (`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`) が Release build で完全には活性化されていない可能性
- または `DeferredFadingActive` イベントがそもそも発生していない (Generation race で Stale になる)

### 8.2 推定 (D131 仮説と反する)

D131 監査の仮説「6 burst → 6 DeferredFadingActive」は **誤り**。D132-7 実測では:
- burst #1: 1 publish → 1 crossfade → **1 DeferredFadingActive (推定)**
- burst #2-5: **5 StaleSuperseded** (PublicationAdmission.cpp L17 の `req.generation != currentGeneration` で reject、 deferred 行きではなく即時 reject)

D132-7 log 末尾の `ISR][Shutdown] Drain incomplete: ... deferred=0 ... maxDeferredAgeMs=358 oldestAgeMs=35219` の `deferred=0` は **shut down 時点で deferred slot が空**を示す。一時的に deferred slot が使われても evaluate 時に Stale 判定されて discard される (DiscardReason::StaleDiscard)。

### 8.3 DeferredFadingActive 確定値

**D132-7 で観測された DeferredFadingActive 遷移 = 1 (burst #1 のみ)**
**D132-7 で観測された StaleSuperseded = 5 (burst #2-5)**

**D131 仮説「DeferredFadingActive = 6」は誤り**。正しい数値は「DeferredFadingActive=1, StaleSuperseded=5」。

---

## RA-9 — D131-G5 判定の正式更新

### 9.1 D131-G5 結論 (D131 監査時)

> `DSPTransition::setIRChangeFlag()` が唯一の rebuild loop entrance。
> D132 実装で :123 撤去により loop 停止が予測される。

### 9.2 D132-RA 実測との整合

D132 で :123 撤去後も **1 publish のまま** (loop 継続)。
- 撤去前 (D129-3B) と撤去後 (D132) で PUBLISH 数 = 1 で同じ
- **D131-G5 の予測は外れた**

### 9.3 真因の特定

**真因は Generation race による RejectedStaleGeneration の累積**。

- burst #1: publish 成功 (gen=4 → seq=6)
- burst #2: req.generation=5 で submit → publish path で currentGen=6 → Stale → reject
- burst #3: req.generation=6 で submit → publish path で currentGen=7 → Stale → reject
- burst #4: req.generation=7 → Stale
- burst #5: req.generation=8 → Stale

**Causal chain**:
```
6 burst
  → 6 submitRebuildIntent (kind=Structural)
    → 6 requestRebuild(sr,bs)
      → 6 generation++ (gen=4,5,6,7,8,9? — task_queued の diagnostic は gen=4-8)
        → 6 enqueuePublicationIntentForRuntimeCommit
          → 6 PublicationExecutor::executePublish
            → 6 PublicationAdmission::evaluate
              → 1 Accepted (burst #1, req.gen=4, currentGen=4)
              → 5 RejectedStaleGeneration (burst #2-5, req.gen!=currentGen)
```

### 9.4 公式更新

D131-G5 の判定を以下に更新:

| 旧判定 (D131) | 新判定 (D132-RA) |
|----------------|------------------|
| `DSPTransition::setIRChangeFlag()` が唯一の loop entrance | **誤り**。loop entrance は `submitRebuildIntent → requestRebuild(sr,bs) → enqueuePublicationIntentForRuntimeCommit` の連鎖 |
| Timer:800 は rebuild 発行の伴奏 flag | **正しい** (Timer.cpp:794 が rebuild 発行、L800 が setIRChangeFlag 伴奏) |
| M5 (DSPTransition:144 撤去) で loop 停止 | **誤り**。DSPTransition:144 は crossfade completion 時の flag で、rebuild loop 入口ではない |
| 1 publish → loop の証拠 | **誤り**。1 publish の真因は Generation race による Stale 累積 reject |

**分類**: **Case D** (複数入口が存在、causal graph として再定義必要) — ただし D131-G5 が想定した「入口」ではなく、 **Generation race による reject 累積** が真因。

---

## RA-10 — D132 判定の分割

| 項目 | 現状 | 評価 |
|------|------|------|
| Build | **PASS** | Debug + Release ビルド成功 |
| CTest | **PASS 40/40** | contract test 整合性確認 |
| INV-DEFERRED-2 | **PASS (code-level)** | overwrite 時の旧 DSP retire 実装済み。実測 orphan=0 確認 (D132-7 log の deferred=0 と整合) |
| INV-DEFERRED-3 | **PASS (code-level)** | discard 時の新 DSP retire 実装済み |
| M1 (registration ≠ activation) | **PASS (code-level)** | working tree で実装済み、build 成功 |
| M2 terminalization | **PASS (code-level)** | terminalizeFadingDSP 実装 (本 RA で追加) |
| M2 idle-world transition | **要修正**: `publishIdleWorldOnly(current, HardReset)` 契約 vs Timer.cpp inline `SmoothOnly` | HardReset で再実装必要 |
| M5 DSPTransition flag 撤去 | **実装PASS / 効果なし** | DSPTransition:144 撤去は code 的に正しかったが、D131 仮説の loop 入口ではなかったため効果なし |
| M6 DeferredFadingActive 解除 | **未達 (前提違い)** | D131 仮説「DeferredFadingActive=6」は誤り。実測 DeferredFadingActive=1、StaleSuperseded=5 |
| rebuild-loop elimination | **未達** | 真因は setIRChangeFlag ではなく Generation race。loop entrance は別 (submitRebuildIntent 連鎖) |
| 6 burst → 6 publish | **FAIL** | 真因は DeferredFadingActive loop ではなく、Generation race による RejectedStaleGeneration 累積 |
| orphan prevention | **改善確認** | INV-DEFERRED-2/3 実装で orphan 経路を閉塞 (D132-7 deferred=0 と整合) |

---

## D132-RA 監査成果物 (A-I)

### A. 6 burst → 1 publish の完全 causal graph

→ RA-1.B/D 参照。要約:
```
6 submitRebuildIntent
  → 6 requestRebuild(sr,bs) → 6 generation++ (gen=4,5,6,7,8,9?)
    → 6 enqueuePublicationIntentForRuntimeCommit
      → 6 PublicationExecutor::executePublish
        → 6 evaluate()
          → 1 Accepted (burst #1, gen一致)
          → 5 RejectedStaleGeneration (burst #2-5, gen race)
```

### B. `delegate_requestRebuild_sr_bs` 全 caller

→ RA-2 参照。 1 箇所: `AudioEngine.RebuildDispatch.cpp:338`。
呼び出し階層: UI/Param/State/Init → submitRebuildIntent → (MT+Structural) → emit DelegateRequestRebuildSrBs → requestRebuild(sr,bs) → emit RequestRebuildSrBs → rebuildMutex 排他下で task queue 判定。

### C. Timer:800 の causal role

→ RA-3 参照。Timer:800 の `setIRChangeFlag()` は **L794 の `submitRebuildIntent` 後の伴奏**で、rebuild 発行本体は L794。**loop 入口ではない** (D131 判定は正しかった)。

### D. M2 terminalize → idle world → Accepted の line-level proof

→ RA-4 参照。Timer.cpp:971 (`terminalizeFadingDSP`) → L982-997 (commitRuntimePublication) → `buildRuntimePublishWorld(current, nullptr, SmoothOnly, 0.0, false)` → `fadingRuntimeUuid=0` の world → `hasFadingRuntimeInWorld=false` → 次 publish で Accepted。

**ただし `idlePolicy=SmoothOnly` で D131 契約 `HardReset` と差分**。`SmoothOnly` は既存 crossfade state を保持するため、`HardReset` と同等ではない。

### E. DeferredFadingActive entry/completion 実測

→ RA-8 参照。**D132-7 で観測された DeferredFadingActive=1 (burst #1 のみ)、StaleSuperseded=5**。D131 仮説「6 DeferredFadingActive」は誤り。

### F. INV-DEFERRED-2/3 ownership proof

→ RA-6 参照。`resolve old → retire old → release/install new` の順序遵守。`resolveDSPHandle` nullptr 返却時は no-op で safety。

### G. D131-G5/G6 の判定更新

→ RA-9 参照。
- **D131-G5 撤回**: 「DSPTransition:setIRChangeFlag() が唯一の loop entrance」は誤り。真因は Generation race による Stale 累積 reject。
- **D131-G6 部分修正**: M2 idle-world republish は code 的に成立するが、`publishIdleWorldOnly(current, HardReset)` 契約 vs Timer.cpp inline `SmoothOnly` の差分あり。

### H. D132 residual defects

| 項目 | 内容 |
|------|------|
| Generation race | rebuild 完了後の currentGen と enqueue 時の req.generation の race で 5/6 が Stale reject |
| 1 publish / 6 burst | 真因は DeferredFadingActive loop ではなく、Generation race |
| M2 idle-policy 不一致 | D131 契約 `HardReset` vs Timer.cpp inline `SmoothOnly` |
| D117 trace 取得未実施 | Release + DIAGNOSTICS 有効でも MEM_SNAP / D117_* ログ出力が限定的 (grep 0 件) |
| RA-7 実測 trace 未取得 | 同一 DSPCore* に対する M2 retire / INV-DEFERRED-2 retire / discard retire の突合は D132-RA では未実測 |

### I. 次の production patch の最小スコープ

D132-RA の結論として、**D132 で実装した patch に加えて 以下が必要**:

1. **M2 idle-policy を `HardReset` に揃える** (D131-G6 契約遵守)
   - Timer.cpp:982-997 の `commitRuntimePublication` 呼び出しを `publishIdleWorldOnly(current, HardReset)` に置換
2. **Generation race の解消** — `requestRebuild` 後の `rebuildRequestGeneration` 更新タイミングと `evaluate` 時の `currentGen` 取得の race を解消
   - 候補: `enqueuePublicationIntentForRuntimeCommit` 時に `req.generation` を snapshot し、 evaluate 時に generation check を `enqueue 時の snapshot` vs `currentGen` に変更
   - または: `rebuildRequestGeneration` を `enqueue 時に advance` せず、 `publish 成功時に advance` に変更
3. **D117 lifetime trace の有効化検証** — `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` が実際に trace を emit するか確認
4. **Generation staleness の根本対策** — D131 監査では想定外だった Generation race が真因のため、根本対策を別タスク (D133+) で検討

**重要**: 上記は「**修正案**」であり、本 D132-RA では **実装しない**。D132-RA の終了条件は「原因特定」であって「修正実装」ではない。

---

## D132-RA の GO/NO-GO 条件評価

### GO 条件 (8 項目)

| # | 条件 | 達成 |
|---|------|------|
| 1 | 1 publish の収束地点が line-level で確定 | ✅ RA-1.D (PublicationAdmission.cpp L17 Stale 累積) |
| 2 | delegate_requestRebuild_sr_bs の全入口確定 | ✅ RA-2 (1 箇所: L338) |
| 3 | Timer:800 の causal role 確定 | ✅ RA-3 (伴奏 flag、loop 入口ではない) |
| 4 | M2 idle-world publish の意味論が確定 | ✅ RA-4 (SmoothOnly 採用、HardReset 契約と差分) |
| 5 | hasFadingRuntimeInWorld() の false 化条件確定 | ✅ RA-4 (fadingRuntimeUuid=0 の新 world) |
| 6 | DeferredFadingActive entry/completion 数を実測 | ✅ RA-8 (1/6 burst、5 StaleSuperseded) |
| 7 | INV-DEFERRED-2/3 exactly-once 再確認 | ✅ RA-6 (順序遵守、冪等保険) |
| 8 | D131-G5 の判定を更新 | ✅ RA-9 (Case D、Generation race が真因) |

### NO-GO 条件 (3 項目)

| # | 条件 | 該当 |
|---|------|------|
| 1 | 「Timer:800 が怪しい」だけの状態 | ❌ 該当せず (RA-3 で causal role 確定済み) |
| 2 | 「delegate_requestRebuild_sr_bs が怪しい」だけの状態 | ❌ 該当せず (RA-2 で全 caller 確定済み) |
| 3 | 「1 publish だから merge が原因だろう」だけの状態 | ❌ 該当せず (RA-1.D で真因 = Generation race を特定済み) |

### 総合判定

**GO** — D132-RA は **8 つの GO 条件をすべて達成**。D131 仮説 (DeferredFadingActive loop) は撤回、 **真因は Generation race** を確定。

---

## 次のアクション

D132-RA で「6 burst → 1 publish の真因 = Generation race」を確定。
これは **D131 監査で想定されていなかった独立したバグ**。

**推奨次タスク**: **D133-Design** — Generation race の解消設計 (RA-10.I の最小スコープ検討)

修正実装に **進むべきではない** (D132-RA 終了条件 = 原因特定)。D133 設計で修正スコープを確定してから D134 で実装、が正しい順序。
