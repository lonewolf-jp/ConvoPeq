# Practical Stable ISR Bridge Runtime 実装TODO

対象設計:

- `doc/work11/Practical_Stable_ISR_Bridge_Runtime_完全実装_詳細設計_2026-05-31.md`
- `doc/work11/Practical_Stable_ISR_Bridge_Runtime_AI実装統治規約_v1.2_2026-05-31.md`

方針:

- fail-closed
- RuntimeWorld単一authority
- partial semantic update禁止
- publish/observe/retire契約分離

---

## 1. スキーマ/authority基盤

- [x] `RuntimeState::kFieldDescriptors` が必須authority（topology/routing/execution/publication/overlap/retire/generation/worldId）を包含
- [x] `RuntimeState::validateDescriptorSet()` が schema + publication descriptor を検証
- [x] `RuntimeSemanticSchema` の version 整合（`schemaVersion == kRuntimeSemanticSchemaVersion`）
- [x] `SemanticCategory/Ownership/Mutability/Visibility/Lifetime` 契約の重複・空名を拒否

## 2. Publication契約

- [x] `runPublicationPrecheckNonRt` で completeness/validity/admission を fail-closed
- [x] `generation` strict monotonic（rollback reject）
- [x] `publication.sequenceId` strict monotonic（rollback reject）
- [x] `publication.previousSequenceId < publication.sequenceId` を強制
- [x] `projectionFreshness` と generation 整合を強制
- [x] frozen/sealed world 条件を公開前必須化
- [x] replacement atomicity（new visible → old retire）を維持
- [x] queue ordering（enqueue/generation/publish）逆転禁止

## 3. Observe契約

- [x] AudioThread の authority observe source を RuntimeWorld のみに固定
- [x] observer-side mutation/lazy update を禁止
- [x] 禁止型参照（RuntimeGraph/BuildSnapshot/PublicationIntent/TransitionState）契約の検証を追加

## 4. Overlap契約

- [x] overlap authority を `world.overlap` のみに固定
- [x] crossfade/snapshot は ExecutorLocal としてのみ許可
- [x] overlap leakage 検出テストを追加

## 5. Retire契約

- [x] retire safety（No Reader/No Executor Ref/No Pending Transition）条件を検証
- [x] bounded starvation（epoch + wall clock）を検証
- [x] escalation 後 safety成立時のみ reclaim を検証
- [x] destruction ordering（Published→Retiring→Retired→Destroyed）を検証

## 6. 追加verifier（統治規約24項目対応）

- [x] schema completeness verifier
- [x] semantic validity verifier
- [x] runtime admission verifier
- [x] self-contained world verifier
- [x] semantic dependency graph verifier
- [x] publication queue ordering verifier
- [x] replacement atomicity verifier
- [x] generation monotonicity verifier
- [x] visibility monotonicity verifier
- [x] observe path verifier
- [x] observe forbidden-type verifier
- [x] overlap authority verifier
- [x] retire safety verifier
- [x] retire escalation verifier
- [x] hidden authority verifier
- [x] authority exhaustiveness verifier
- [x] semantic alias verifier
- [x] multi-writer prohibition verifier
- [x] memory ordering contract verifier
- [x] ownership transfer verifier
- [x] ABA hazard verifier
- [x] deterministic build verifier
- [x] semantic conflict verifier
- [x] semantic equivalence verifier

## 7. テスト/ゲート

- [x] `src/tests/RuntimeSemanticSchemaValidationTests.cpp` を契約テスト拡張
- [x] `src/tests/RuntimePublicationCoordinatorTests.cpp` を queue/monotonic契約で拡張
- [x] `src/tests/RetireGraceSemanticsTests.cpp` を escalation/safety契約で拡張
- [x] `src/tests/ISRSemanticValidationTests.cpp` を fail-closed reject理由まで拡張
- [x] CTest登録を確認（`CMakeLists.txt`）
- [x] Debugビルド成功
- [x] 全CTests成功（Green）

## 8. 完了条件

- [x] TODO全項目 [x]
- [x] ビルドとテストがGreen
- [x] 統治規約v1.2違反がないことを自己検証
