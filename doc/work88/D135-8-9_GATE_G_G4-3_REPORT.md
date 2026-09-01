# D135-8/9 Gate G-4.3 — Phase-I Coalesce Admission（Work Report）

**Status:** implemented. **STOP after this.**
**Type:** implementation. Base: G-4.2-RF Audit PASS / D105-R23 Phase-I model.
**Files changed:** `src/audioengine/ISRRuntimePublicationCoordinator.cpp`（G-4.3 変更箇所は `submitRecoveryRequest` の coalesce 分岐のみ）+（.h は G-4.2-RF から不変; cumulative diff vs HEAD = 2ファイルのみ）。
**Tests:** 0変更（指示どおり、実装段階ではテストを触らない）。

## What was implemented
`submitRecoveryRequest` の Phase-I admission path は既に coalesce skeleton（`findByKey` → 既存 Live obligation 再利用・ΔL=0 / `tryInsert` → 新規 +1）を持っていた。G-4.3 は**不足していた V10（D27.2 linearization / terminal race）を閉じた**：

```cpp
const std::size_t existing = recoveryAdmissions_.findByKey(cid);
bool coalesceOnLive = false;
if (existing != recoveryAdmissions_.kCapacity) {
    auto& slot = recoveryAdmissions_.slot(existing);
    ObligationState expectLive = ObligationState::Live;
    coalesceOnLive = slot.state.compare_exchange_strong(expectLive, ObligationState::Live,
                                                        std::memory_order_acq_rel, std::memory_order_acquire);
}
if (coalesceOnLive) {
    // COALESCE: reuse existing Live obligation (D18.7 — ΔL = 0; no new oblId)
    oblId = slot(existing).id.load(acquire); slotIdx = existing;
    slot(existing).intentId = intent.intentId;   // diagnostic refresh only
    fetchAdd(recoveryCoalescedCount_);
    if (wasDeferredBefore && slot(existing).delivery != None) return true;  // no second delivery
} else {
    const auto ins = tryInsert(cid);   // NEW obligation (+1); capacity full → reject
    ...
}
```

- **CAS Live→Live を linearization fence に**: `findByKey` はスナップショットのため、その間に `resolveRecoveryObligation`（ISR/RebuildThread）が Live→terminal する競合があり得た。CAS で「この瞬间 Live」を原子的に再検証。**敗者（既に terminal）は coalesce せず tryInsert（新規・+1）へ** → terminal 後の coalesce mutation を構造的に禁止（D27.2）。
- coalesce での mutation は**診断用 intentId の refresh のみ**（identity/state/liveCount/delivery/recoveryGeneration は不変）。既存 oblId を再利用（V8）。
- redrive ordering / `wasDeferredBefore` 二重 delivery 防止は**維持**（R10-3/C16）。

## V1-V15 verification

| V | verify | expected | result |
|---|---|---|---|
| V1 | same handle + same target | COALESCE | ✅ findByKey match → coalesceOnLive(Uue→Live) → reuse |
| V2 | same handle + different target | NEW | ✅ findByKey npos → tryInsert (+1) |
| V3 | different handle + same target | NEW | ✅ CoalesceIdentity の handle 不一致 → npos → tryInsert |
| V4 | COALESCE liveCount_ | ΔL=0 | ✅ coalesce 分岐は tryInsert を呼ばない |
| V5 | NEW obligation | ΔL=+1 | ✅ tryInsert |
| V6 | capacity full + matching Live | COALESCE 成功 | ✅ findByKey で Live 検出 → coalesce（capacity チェック無し） |
| V7 | capacity full + no matching | reject | ✅ npos → tryInsert → liveCount_≥32 → nullopt → return false |
| V8 | COALESCE new oblId | なし | ✅ 既存 `slot(existing).id` 再利用 |
| V9 | duplicate delivery | なし | ✅ `wasDeferredBefore && delivery!=None → return true`（redrive ordering 維持） |
| V10 | terminal race | terminal 後の coalesce mutation 不可 | ✅ CAS Live→Live; 敗者は tryInsert（terminal へ mutation しない） |
| V11 | RecoveryGeneration regen on coalesce | しない | ✅ coalesce 分岐は recoveryGeneration を触らない（`intent.recoveryGeneration = slot.recoveryGeneration` で既存値維持。新規のみ `++nextRecoveryGeneration_`） |
| V12 | intentId separation | 混同なし | ✅ `recoveryGeneration = intent.intentId` = 0 |
| V13 | supersession | 未実装 | ✅ canSupersede/isSemanticSuperset/isDomainSuperset/isSemanticTargetSuperset はコメントのみ(production wiring なし)。ResolvedSuperseded dormant のまま |
| V14 | EpisodeId production ref | 0 | ✅ |
| V15 | diff boundary | G-4.3 scope 外なし | ✅ 2ファイルのみ（.cpp + .h）。tests/CMake/reclaim/shutdown/publish 未変更 |

## Nothing changed
- CoalesceIdentity = { quarantinedHandle, SemanticRecoveryTarget }（canonical key 不変。handle-only / handle+generation / handle+intentId / handle+epoch / handle+RecoveryEpisodeId は不使用）。
- RecoveryEpisodeId: production ref 0（Phase-II deferred 維持）。
- durable-table 置換 / blind-overwrite 再設計 / reservation 変更 / canSupersede / SUPERSEDE / ResolvedSuperseded production / stalled / retry K=4 / publish / reclaim / shutdown — **全て未着手**。
- `pendingRecoveryAdmission_`（single-slot durable fallback）は別レイヤのため触らない（coalesce と分離）。

## STOP
G-4.3 implementation complete（V1-V15 構造確認）. **STOP — durable overwrite / coalesce-追加 / その他はしない。** Await G-4.3 Audit（read-only）→ PASS →（必要なら）dedicated regression tests. No test files changed.
