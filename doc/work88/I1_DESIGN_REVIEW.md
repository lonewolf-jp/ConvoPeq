# I1 Design Review — Recovery coalesce（§1.2）verify-before-implement

- 日付: 2026-08-15
- 判定: **Phase I は設計レビュー完了・実装は未 GO**（ユーザー10項目 × コード/D2 突合の結果）
- 前提: Phase H GO・H-1〜H-3 HOLD（2026-08-15 確定）。本レビューは REPAIR_PLAN2-dash2.md §1.2（§1.2.1〜§1.2.3 R1〜R17）と現行実装の突合。
- 参照: D2_IMPL_CHECKLIST Phase I（I-1〜I-5）/ P0-7 / INV-X1-1〜6

---

## 現行実装の状態（コード照合済み）

| 構成要素 | 現行コード | 状態 |
|---|---|---|
| `RecoveryIntent` | `ISRRuntimePublicationCoordinator.h:217`（handle + epoch + intentId + buildSource 値コピー・trivially copyable） | 実装済み |
| `recoveryIntentQueue_` | `h:637`（LockFreeRingBuffer SPSC 256・transport-only） | 実装済み |
| `PendingRecoveryAdmission` | `h:655`（State{NoAdmission, DurablePending, Building} + handle/recoveryGeneration/buildSource/reservationOwned/epoch/intentId） | 実装済み（**single slot**） |
| lease 状態遷移 | `cpp:927` take（DurablePending→Building・destructive でない）/ `cpp:968` settle(retry)（Building→DurablePending / →NoAdmission） | 実装済み |
| durable 上書き | `cpp:905-915` — **無条件 latest-wins 上書き**（コメント明記: "単一スロット — 既存 durable があれば最新で上書き"） | 🔴 NO-GO パターン |
| `LogicalRecoveryIdentity` | 未定義 | ❌ 未実装 |
| `RecoveryProvenance` | 未定義 | ❌ 未実装 |
| `SupersessionDecision` / `canSupersede()` | 未定義 | ❌ 未実装 |
| bounded durable table | 未定義（single slot のまま） | ❌ 未実装 |

---

## 10項目の突合・判定

### 1. Recovery の logical identity の定義
- **プラン（§1.2.1）**: `LogicalRecoveryIdentity` = handle + generation（epoch-consistent）+ semantic build identity（IR hash / sample rate / channel count）+ activation/publication epoch。**admission source は含めない**。
- **コード**: `PendingRecoveryAdmission` に handle / recoveryGeneration / buildSource（`RuntimeBuildSnapshot`: generation + buildInput + convolverFingerprint + rebuildFingerprint + sealed + PR-2 projection）/ epoch。**`LogicalRecoveryIdentity` 型なし**。
- **ギャップ**:
  - R1 未充足（型の定義自体がない）。
  - `recoveryGeneration` は `intent.intentId` を代用（`cpp:910`）— プランは「intentId は identity ではない」「generation は epoch-consistent」を要求。**intentId を generation に使うのは設計不一致**（R1/R9 の要整理）。
  - semantic build identity（IR hash / sample rate / channel count）の合成関数が未定義（buildSource の convolverFingerprint / rebuildFingerprint / buildInput から導出する設計が必要）。
- **判定**: ❌ 未実装・**型と生成規則の設計決定が必要**。

### 2. Compatibility と Supersession の完全な分離
- **プラン（§1.2.2 / R3）**: `SupersessionDecision` enum（CanSupersede / DifferentHandle / DifferentSemanticTarget / NotSameGenerationDomain / NotSemanticSuperset）+ `canSupersede(newer, older)`（handle 同一 + generation strict 増加 + `isSemanticSuperset`）。「generation 増加 ≠ semantic superset」を明示。
- **コード**: `SupersessionDecision` / `canSupersede()` なし。現行は「latest buildSource で上書き」— **プランが NO-GO と明示した単純 latest-wins パターン**（`cpp:905` コメントが自認）。
- **ギャップ**: R3/R7/R8/R9/R10 未充足。Compatibility（両立可能）と Supersession（完全代替）の区別が存在しない。
- **判定**: ❌ 未実装・**現行 durable 上書きは NO-GO パターンであり、実装前に修正必須**。

### 3. Building 中の request の状態遷移
- **プラン（§1.2.2 手順3 / R4 / RC-11・H.11.27.3）**: `NoAdmission → DurablePending → Building → [TransientFail → DurablePending] → [Success → NoAdmission]`。**Building 中の supersession は pending 扱い**（I-4）。
- **コード**: 状態遷移の骨格は実装済み（State enum・take=state transition・settle(retry)）。INV-X1-1。ただし **Building 中に新しい logical recovery が来た場合の supersession pending は single slot のため保持不能**（I-4 未実装）。
- **判定**: 🟡 骨格は実装済み（R4 部分）・**Building 中 supersession（RC-11 / I-4）は未実装**。

### 4. lease の所有 authority
- **プラン（§1.2.3 R4）**: lease 型 state machine。take は destructive dequeue でなく state transition。
- **コード**: lease 方式は実装済み — take = DurablePending→Building（コピー返却・クリアしない）、settle = Builder が build 結果で settle。SPSC（Producer=CoordinatorLoop / Consumer=Builder Loop）。**Coordinator が durable state を所有・Builder が lease を取得**という所有権モデルは整合。
- **判定**: ✅ 実装済み・整合（bounded table 化時の lease 拡張は未実装）。

### 5. durable table の容量・eviction semantics
- **プラン（§1.2.3）**: `pendingRecoveryAdmission_`（single slot）→ **bounded durable table**（例: `kMaxDurableRecoveryAdmissions`）。CanSupersede==false の複数 logical obligation を保持。「coalesce できないから捨てる」は**禁止**。
- **コード**: **single slot のみ**。`cpp:905-915` の無条件上書きにより、第二の非 supersedable Recovery が slot 占有中（DurablePending または Building）に来ると**前の obligation が喪失**（**P0-7 リスクが現行で実在**）。eviction semantics 未定義。
- **判定**: ❌ 未実装・**P0-7 が現行コードに顕在**（容量 1・上書きで obligation 喪失）。

### 6. supersession と stale-discard の境界
- **プラン（§1.2.2）**: obsolete policy（A(G10),B(G10),C(G11) → G11 のみ保持）は `canSupersede(C,A) && canSupersede(C,B)` 成立時のみ。stale（obsolete）は Discarded（settle(false)）。
- **コード**: semantic supersession 判定なし（上書きのみ）。Discarded（settle(false)）と shutdown discard（recoveryShutdownDiscardCount_）はあるが、**supersession との区別なし**。
- **判定**: ❌ 未実装。

### 7. RT/non-RT ownership
- **プラン（§1.2.3）**: coalesce は Audio Thread で行わない。producer = CoordinatorLoop（NonRT）のみ。Queue は transport のまま、semantic admission/coalesce は RecoveryAdmission 側。
- **コード**: Producer = CoordinatorLoop のみ（submitRecoveryRequest ← QuarantineIntentHandler via processIntent）。Consumer = Builder Loop。coalesce を NonRT（RecoveryAdmission）側に置く前提が成立。
- **判定**: ✅ 整合・**実装前提クリア**。

### 8. existing RecoveryIntent queue との整合
- **プラン（§1.2.3）**: Queue は transport のまま（recoveryIntentQueue_）。semantic admission は別。
- **コード**: transport（recoveryIntentQueue_ SPSC 256）と durable（pendingRecoveryAdmission_）の分離は設計と一致。INV-X1-6（durable は queue residency と二重計上しない）文書化済み。
- **判定**: ✅ 整合（coalesce 導入時も transport はそのまま・durable 側で semantic 処理）。

### 9. wraparound / generation / intentId の関係
- **プラン（§1.2.3）**: intentId は identity ではない（sequence/diagnostic）。coalesce は LogicalRecoveryIdentity の等価性で判定。generation domain mismatch は coalesce 不能（R9）。
- **コード**: intentId は診断用（RecoveryIntent.intentId・nextRecoveryIntentId_）で identity には使用していない — **プランと整合**。ただし **recoveryGeneration に intentId を代用**（cpp:910）— generation domain の意味が不明確（R1/R9 で要整理）。PublicationSequenceId/Epoch の modular 比較は Phase H の `SequenceArithmetic.h` を適用可能。
- **判定**: 🟡 部分整合（intentId≠identity は OK・recoveryGeneration=intentId の代用は要見直し）。

### 10. rollback point と invariant
- **プラン（R5/R14/R15 / D2 I-5）**: R5（exactly-one-logical-obligation）/ R14（coalesce で reservation 増加なし・INV-X1-5）/ R15（非 superseded recovery を削除しない）。D2 I-5 = build + ctest rollback point。
- **コード**: INV-X1-5/6 は文書化・実装済み（reservationOwned・1 admission = 1 reservation）。ただし **coalesce 後の logicalAdmissionCount == 1 検証テストは未実装**。
- **判定**: 🟡 骨格実装済み・coalesce 後の invariant テスト未実装。

---

## 総合判定

**Phase I は実装先行すべきでない。** 現行アーキテクチャは良い基盤を持つが、GO 条件（R1〜R17）のうち複数が未充足であり、**特に `cpp:905-915` の無条件 latest-wins 上書きはプランが NO-GO と明示したパターンが現行コードに残存**（P0-7 が顕在）。

### 実装済み・整合（基盤）
- lease 型 state machine の骨格（R4 部分・take/settle）
- NonRT-only producer（R7 前提）
- transport/durable 分離 + INV-X1-6（R13 部分）
- intentId ≠ identity（R の一部）
- INV-X1-5/6 の文書化

### 未実装（GO 条件）
- R1 `LogicalRecoveryIdentity`（型 + generation 規則 + semantic build identity 合成）
- R2 `RecoveryProvenance` enum（Transport/Durable/Retry/Quarantine）
- R3 `SupersessionDecision` + `canSupersede()`（R7〜R10）
- I-3 bounded durable table（容量・eviction・P0-7 解消）
- I-4 Building 中 supersession pending（RC-11）
- R5/R12/R15 の invariant（logicalAdmissionCount == 1 テスト）

### 実装前に必要な設計決定（ユーザー合意事項）
1. **LogicalRecoveryIdentity の生成規則**: generation を intentId から分離し「epoch-consistent」にする定義。semantic build identity（IR hash / sample rate / channel count）の合成元（buildSource の convolverFingerprint / rebuildFingerprint / buildInput から）。
2. **bounded durable table の容量**（kMaxDurableRecoveryAdmissions の値）と **full 時の eviction semantics**（coalesce 不能・table full 時の挙動 — drop 禁止のため admission gate か、oldest からの supersession 判定）。
3. **canSupersede() の semantic superset 判定**（isSemanticSuperset の実装 — EQ change vs IR change の区別）。
4. **現行の無条件上書き経路（cpp:905-915）の修正方針**（supersession 判定導入 or 一時的に Building 中上書きを拒否）。

### 推奨実装順序（GO 後のみ）
R1（LogicalRecoveryIdentity）→ R2（RecoveryProvenance）→ R3（SupersessionDecision + canSupersede）→ 状態遷移拡張（RC-11）→ bounded durable table（I-3）→ invariant テスト（logicalAdmissionCount == 1）→ I-5（build + ctest 全 PASS）。

### D2 Phase I チェックリスト状態
| # | 項目 | 状態 |
|---|------|------|
| I-1 | LogicalRecoveryIdentity + RecoveryProvenance 導入 | [ ] 未実装（設計レビュー完了・要設計決定） |
| I-2 | SupersessionDecision（Compatibility ≠ Supersession） | [ ] 未実装 |
| I-3 | lease state machine + bounded durable table | [-] 骨格は実装済み・table 化は未実装 |
| I-4 | Building 中の supersession は pending 扱い（RC-11） | [ ] 未実装 |
| I-5 | ビルド + ctest 全 PASS | [ ] rollback point（GO 後） |
