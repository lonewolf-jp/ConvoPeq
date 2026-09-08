# D135-8/9 Gate G-4.3-RF — Coalesce Post-CAS Mutation Removal（Work Report）

**Status: PASS**
**Production source changes: 1 line deletion**（`submitRecoveryRequest()` の COALESCE 分岐から `recoveryAdmissions_.slot(existing).intentId = intent.intentId; // diagnostic refresh` を削除）。**G-4.3-R Audit は実施しない（指示どおり提示のみ）。**

## Removed
```cpp
recoveryAdmissions_.slot(existing).intentId = intent.intentId; // diagnostic refresh
```
COALESCE 分岐の**1行のみ**。NEW 分岐の `slot.intentId = intent.intentId;`（cpp:954）と durable の `pendingRecoveryAdmission_.intentId`（cpp:1002/1163）は**維持**。

## V1-V8

| V | 検証 | 期待 | 結果 |
|---|---|---|---|
| V1 | COALESCE 分岐の `slot(existing).intentId` 除去 | 0 | ✅ 0 件（`slot.intentId`/`pendingRecoveryAdmission_.intentId` は残存 = 他経路維持） |
| V2 | CAS 維持 | 変更なし | ✅ `compare_exchange_strong(expectLive, ObligationState::Live, ...)`（cpp:927）不変 |
| V3 | COALESCE ΔL=0 | tryInsert を呼ばない | ✅ `coalesceOnLive` 分岐（cpp:931-940）は tryInsert を呼ばず、`else`（cpp:942）でのみ tryInsert |
| V4 | identity/recoveryGeneration/buildSource 不変 | coalesce で変更なし | ✅ coalesce 分岐は oblId 読み取り + delivery 判定のみ。identity/handle/target/recoveryGeneration/buildSource を書かない |
| V5 | terminal race 維持 | 同一 atomic | ✅ coalesce CAS（cpp:927）と terminal CAS（h:428 `slots_[i].state`）は同一 `std::atomic<ObligationState> state` 上で競合 |
| V6 | EpisodeId prod ref | 0 | ✅ 0 |
| V7 | supersession 非導入 | production path なし | ✅ canSupersede/isSemanticSuperset 等 production 0（コメントのみ） |
| V8 | diff boundary | 当該1行削除のみ | ✅ 変更は `ISRRuntimePublicationCoordinator.cpp`（1行削除）+ `.h`（G-4.2-RF から不変）。tests/CMake/AudioEngine/reclaim/shutdown/publish 未変更 |

## 収束
修正後の COALESCE は実質的に：
```
findByKey(cid) → candidate → CAS Live→Live → authorized
  → 既存 oblId 取得（read）
  → ΔL = 0
  → delivery ordering 確認（read）
```
**CAS 成功後の obligation-state mutation = ゼロ**（`recoveryCoalescedCount_` fetchAdd は coordinator telemetry カウンタであり、`LogicalRecoveryObligation` slot の state/lifecycle は非改変）。A5 は「CAS-first だが診断 mutation 残存」から「**CAS が唯一の lifecycle authorization point、COALESCE は post-CAS mutation-free**」へ収束。

## 禁止事項（未着手）
G-4.3-R Audit 再実施 / regression tests / durable overwrite / durable table / admission redesign / reservation redesign / canSupersede / ResolvedSuperseded / stalled-retry redesign / publish / reclaim / shutdown wiring / `.h` 変更 — **全て未着手**。

## STOP
G-4.3-RF = PASS（V1-V8）。**G-4.3-R Audit は実施せず、本結果を提示。** 次の **G-4.3-R Audit → PASS → 専用 regression tests** の判定はお任せします。
