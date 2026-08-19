# Phase F-0 事前監査 — recoveryIntentQueue_ の MPSC 化

- **日付**: 2026-08-19
- **対象**: `ISRRuntimePublicationCoordinator::recoveryIntentQueue_`（`LockFreeRingBuffer<RecoveryIntent, 256>`）
- **目的**: Phase F（MPSC 化）の実装 GO/NO-GO を判定するための producer / consumer / ownership / dataflow 完全列挙
- **判定**: **NO-GO（現時点では実装しない）** — 条件付き GO（将来の第2 Non-RT producer 追加時）

---

## 1. 宣言・型・capacity

| 項目 | 値 | 出典 |
| --- | --- | --- |
| 型 | `LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity>` | `ISRRuntimePublicationCoordinator.h:637` |
| capacity | `kRecoveryIntentQueueCapacity = 256` | `ISRRuntimePublicationCoordinator.h:636` |
| 要素型 | `RecoveryIntent { DSPHandle handle; PublicationEpoch epoch; uint64_t intentId; RuntimeBuildSnapshot buildSource; }` | `ISRRuntimePublicationCoordinator.h:217-230` |
| 要素型保証 | trivially copyable / standard layout（static_assert 済み） | `ISRRuntimePublicationCoordinator.h:231-236` |
| リング実装 | SPSC 前提（writeIndex 非 CAS 更新 → 複数 producer 不可） | `LockFreeRingBuffer.h:25,41-48,60-75` |

## 2. 全 production producer（enqueue 経路）

```text
AudioEngine::submitRecoveryIntent (AudioEngine.h:4403-4430)
  └─ bridge.submitRecoveryRequest (ISRRuntimePublicationCoordinator.cpp:855)
       └─ recoveryIntentQueue_.push (cpp:894)   ← 唯一の enqueue 点
```

`submitRecoveryIntent` の呼び出し元（全列挙）:

| 呼び出し元 | 実行スレッド | 状態 |
| --- | --- | --- |
| `QuarantineIntentHandler::handle`（`ProcessIntent.cpp:140`） | CoordinatorLoop（Non-RT） | **PRIMARY 経路**（quarantine 成功 → recovery） |
| `RecoveryIntentHandler::handle`（`ProcessIntent.cpp:169`） | CoordinatorLoop（Non-RT） | **DEAD CODE**（誰も intentQueue_ に Recovery を push しない） |

- `submitRecoveryRequest` は shutdown gate（state==ShuttingDown → discard）→ reservation → push の順。
- **producer は CoordinatorLoop 単一スレッドのみ**。

## 3. consumer（dequeue 経路）

| 呼び出し元 | 実行スレッド | 内容 |
| --- | --- | --- |
| `AudioEngine::rebuildThreadLoop`（`RebuildDispatch.cpp:927`） | Builder Loop（RebuildThread） | `popRecoveryRequest()` → validate → build → enqueuePublicationIntentForRuntimeCommit |
| `discardRecoveryRequestsOnShutdown`（`cpp:1004`） | shutdown 時 | `popRecoveryRequest()` を drain（discard カウント） |
| `takePendingRecoveryAdmission`（`cpp:927`） | Builder Loop（`RebuildDispatch.cpp:1005`） | durable admission の lease（DurablePending → Building） |

- **consumer は Builder Loop 単一スレッドのみ**（SPSC 成立）。

## 4. 複数 producer の有無

- **無し**。単一 producer（CoordinatorLoop）が 2026-08-14 / 2026-08-15 の2回にわたり検証・文書化済み。
- `RecoveryIntentHandler` は dead code であり、現状 producer 集合に寄与しない。
- ソースコメント（`h:629-635`）も「MPSC 化は現時点で不要」と明記。

## 5. RT スレッドからの enqueue

- **無し**。両 handler とも CoordinatorLoop（Non-RT）経由。
- Timer / Audio Thread / 直接 `submitRecoveryRequest` 呼び出し経路は存在しない。
- INV-R1-MPSC-4（MPSC でも producer は NonRT 限定）は現状満たされている。

## 6. queue full 時の挙動

`submitRecoveryRequest`（`cpp:855-923`）:

1. shutdown gate → `recoveryShutdownDiscardCount_++` → return false
2. `pendingIntentCount_.fetch_add(1, release)`（reservation-before-push）
3. `push` 成功 → return true
4. `push` 失敗 → `fetch_sub(1)` rollback + `recoveryIntentDropCount_++`
5. `pendingRecoveryAdmission_` へ durable 記録（DurablePending, reservationOwned=true）+ `recoveryAdmissionPending_=true`（release）→ return true

- **INV-X1-2: queue full ≠ Recovery lost**（durable admission で obligation 保持）。

## 7. memory ordering

- `LockFreeRingBuffer` SPSC HB: push は buffer 書込 → writeIndex release / pop は writeIndex acquire → buffer 読込。
- reservation 計数: `pendingIntentCount_` fetchAdd(release) → push / 失敗時 fetchSub / pop 成功時 fetchSub。
- INV-R1-1（pendingIntentCount_ = Σreservations − Σpops）は現行実装で成立。

## 8. 既存 MPSC 実装との重複

| キュー | 型 | 状態 |
| --- | --- | --- |
| `intentQueue_` | `MpscBoundedRing<Intent, 4096>`（`h:688`） | 既に MPSC |
| `quarantineFallbackQueue_` | `MpscBoundedRing<Intent, 1024>`（`h:696`） | 既に MPSC |
| `recoveryIntentQueue_` | `LockFreeRingBuffer<RecoveryIntent, 256>`（`h:637`） | SPSC（本監査対象） |

- `MpscBoundedRing`（Vyukov bounded, CAS reservation + seq publication, single consumer）は P2-3 でテスト固定済み（`MpscBoundedRingTests.cpp`、ctest #5 PASS）。

## 9. 現行テスト

| テスト | 対象 |
| --- | --- |
| `invariant_INV3_INV5.cpp` INV-5-1 | 256 full → 257th drop + pendingIntentCount invariant |
| `invariant_INV3_INV5.cpp` INV-5-4 | discardRecoveryRequestsOnShutdown 残存 |
| `ISRSemanticValidationTests.cpp` | 1-hop transport / R7 epoch 伝播=42 / durable admission fallback / settle retry |

- 全 ctest 33/33 PASS（2026-08-19 確認）。

## 10. MPSC 化で変わる API / semantic contract

| 項目 | 現行（SPSC） | MPSC 化後 | 変更量 |
| --- | --- | --- | --- |
| 型 | `LockFreeRingBuffer<RecoveryIntent, 256>` | `MpscBoundedRing<RecoveryIntent, 256>` | 型置換のみ（API 互換） |
| reservation→push→rollback | 実装済み（X1/P2-1） | 変更不要 | **0** |
| pop 時 fetchSub | 実装済み | 変更不要 | **0** |
| `pendingRecoveryAdmission_` | SPSC plain struct（atomic 不要） | **mutex / atomic 保護が必要**（plan §1.1.1） | **要変更** |
| producer 制約 | CoordinatorLoop のみ | NonRT 限定（INV-R1-MPSC-4） | 不変 |
| テスト | SPSC 前提 | 2-producer 同時 enqueue / full→rollback / no-dup / no-underflow 追加 | **要追加** |

---

## 判定: NO-GO（現時点では実装しない）

### 根拠

1. **単一 producer が検証済みの不変条件**。CoordinatorLoop のみが enqueue し、2回の監査で確認済み。MPSC 化の動機（第2 producer）が現状コードに存在しない。
2. **将来拡張の引き金が未発生**。Timer 等からの直接 `submitRecoveryRequest` 経路は存在しない。plan §1.1 自身も「現状は SPSC 成立のため緊急性なし（Phase 5 将来拡張）」と明記。
3. **reservation/rollback 機構は既に実装済み**。MPSC 化の実質的な差分は「型置換 + pendingRecoveryAdmission_ の保護 + 2-producer テスト」のみで、現時点の機能利益はゼロ。
4. **リスク対効果が悪い**。`pendingRecoveryAdmission_` の mutex 保護は新たなロック経路を導入し、実 producer 不在のまま未使用の並行性コードを追加することは投機的複雑性（speculative complexity）に該当。
5. **INV-R1-1 / INV-R1-2 / INV-X1-2 は現行 SPSC 実装で既に成立**。MPSC 化は不変条件を「維持」するだけで「新規に満たす」ものではない。

### 条件付き GO（将来のトリガー）

以下のいずれかが発生した時点で Phase F を実施する:

- **第2の Non-RT producer**（例: Timer からの直接 `submitRecoveryRequest`）を実際に追加する設計が確定したとき。
- その際の実施内容（plan §1.1 準拠）:
  1. `LockFreeRingBuffer<RecoveryIntent, 256>` → `MpscBoundedRing<RecoveryIntent, 256>` 型置換
  2. `pendingRecoveryAdmission_` を mutex 保護（または単一 NonRT admission authority / single-writer 不変条件）
  3. 2-producer 同時 enqueue / full→rollback / no-dup / no-underflow テスト追加
  4. INV-R1-1 / INV-R1-2 / INV-R1-MPSC-4 / AC-ISR-1（Audio Thread 非 producer）の検証

### 推奨アクション

- 本監査結果を `ISRRuntimePublicationCoordinator.h:629-635` のコメントに追記し、判定（NO-GO + 条件付き GO トリガー）を文書化する。
- Phase F は backlog に「将来拡張（Phase 5）」として維持し、実装は開始しない。
