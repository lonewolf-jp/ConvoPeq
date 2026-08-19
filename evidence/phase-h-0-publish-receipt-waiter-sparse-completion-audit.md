# Phase H-0 事前監査 — PublishReceiptWaiter の sparse completion 化

- **日付**: 2026-08-19
- **対象**: `AudioEngine::PublishReceiptWaiter`（`AudioEngine.h:3684-3711`、メンバ `publishReceiptWaiter_` :3713）
- **目的**: sparse completion 化（plan §1.5: `completedThrough_` + `completedOutOfOrder_`）の実装 GO/NO-GO を判定するための completion-path 完全監査
- **判定**: **NO-GO（現時点では実装しない）** — 条件付き GO（第2 completion writer / parallel publish の設計確定時）

---

## 1. PublishReceiptWaiter の宣言・内部状態

| 項目 | 値 | 出典 |
| --- | --- | --- |
| 型 | `struct PublishReceiptWaiter`（AudioEngine の private ネスト型） | `AudioEngine.h:3684` |
| 内部状態 | `std::mutex mutex_` / `std::condition_variable cv_` / `PublicationSequenceId lastCompleted_ = 0`（plain uint64_t・mutex 保護） | `AudioEngine.h:3709-3711` |
| メンバ | `PublishReceiptWaiter publishReceiptWaiter_` | `AudioEngine.h:3713` |
| 公開ラッパ | `notifyPublishReceipt(seqId)` → `complete(seqId)` / `waitForPublishReceipt(seqId, timeoutMs)` → `waitFor` | `AudioEngine.h:3714-3719` |

- **high-water mark（monotonic watermark）単一**。sparse set は存在しない。

## 2. completion の記録方法

`complete(seqId)`（`AudioEngine.h:3687-3697`）:

```cpp
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (convo::isr::isAfter(seqId, lastCompleted_)) { lastCompleted_ = seqId; }
}
cv_.notify_all();
```

- mutex 保護下の **monotonic watermark 更新**（INV-X2-4: stale completion は上書きしない — `isAfter` 判定）。
- 比較は modular（`SequenceArithmetic.h` §1.6.1、RFC 1982 `< 2^63`、wraparound-safe）。
- **O(1)**。スキャン・集合操作なし。

## 3. waiter の登録・解除経路

`waitFor(seqId, timeoutMs)`（`AudioEngine.h:3699-3707`）:

```cpp
std::unique_lock<std::mutex> lock(mutex_);
const auto deadline = ... + milliseconds(timeoutMs < 0 ? 0 : timeoutMs);
cv_.wait_until(lock, deadline, [&] { return convo::isr::isCompleted(seqId, lastCompleted_); });
return convo::isr::isCompleted(seqId, lastCompleted_);
```

- **明示的な登録・解除は存在しない**。共有 watermark + 単一 CV の predicate-and-signal パターン。
- lost-wakeup 安全（`wait_until` の predicate 再評価 — 既存設計が正しい）。

## 4. completion を消費する全 caller

`waitForPublishReceipt` → `commitRuntimePublication`（`AudioEngine.h:4596`、`kPublishReceiptWaitTimeoutMs = 250`）:

| # | 呼び出し元 | スレッド |
| --- | --- | --- |
| 1 | `AudioEngine.Processing.PrepareToPlay.cpp:155` | Non-RT（device setup） |
| 2 | `AudioEngine.Processing.ReleaseResources.cpp:175` | Non-RT（release） |
| 3 | `AudioEngine.Timer.cpp:949` | Non-RT（MessageThread Timer） |
| 4 | `AudioEngine.Transition.cpp:25` | Non-RT（transition） |
| 5 | `PublicationExecutor.cpp:53`（`publishImpl(waitForReceipt=true)` ← `executor_.publish` ← `trySubmitImpl`） | Producer（publish admission 経路） |

- **waitFor の reader は複数スレッド**（単一 writer / 複数 reader パターン）。
- `enqueueRuntimePublicationFireAndForget`（CoordinatorLoop 上の deferred resubmit）は **wait しない**（自己待ち防止 — `AudioEngine.h:4500-4506`）。

## 5. dense scan / full-range scan の有無

- **無し**。現行は単一 watermark の **O(1) 判定**（`isCompleted(seq, lastCompleted_)`）。
- 削減対象となる dense scan / full-range scan は**存在しない**。
- sparse completion 化は `completedOutOfOrder_`（sparse set）を**追加**するため、状態・計算量は**増加**する（削減ではない）。

## 6. completion ID の生成・wraparound semantics

- seqId は `publicationSequenceCounter_`（`AudioEngine.h:2225`、`fetchAddAtomic` :3412）で採番 → world の `publication.sequenceId` → intent.sequenceId → `onPublishCommitted(intent.sequenceId)` → `complete(seqId)`（`SequenceArithmetic.h:33-35` の dataflow 確認済み）。
- 比較は modular（`isAfter` / `isCompleted`、RFC 1982 `< 2^63`）。wraparound は 2^64 値到達時（1ns 間隔で約5840億年）で**実質不可能**（plan §1.6.1 / Appendix E）。

## 7. timeout / cancellation / shutdown 時の挙動

| 状況 | 挙動 |
| --- | --- |
| timeout（250ms） | `waitFor` は false を返す。`commitRuntimePublication` は **timeout ≠ publish failure**（所有権は enqueue 時点で Transferred）→ Transferred 扱いで継続（二十三次レビュー確定） |
| deferred publish cancel | 古い seqId の receipt は来ない → timeout → Transferred（`REPAIR_PLAN2-dash.md:1771`） |
| shutdown | CoordinatorLoop 停止 → `complete()` が来ない → timeout で終了（既存設計が正しい — X2 §6.2） |

- **sparse completion 化は timeout / shutdown 挙動を変更しない**（waitFor の戻り値契約は不変）。

## 8. CoordinatorLoop / RebuildThread との thread boundary

- **writer（complete()）**: CoordinatorLoop 単一スレッドのみ。`PublishExecutor::executePublish` → `onPublishCommitted`（`RuntimePublicationOrchestrator.cpp:305-321`）→ `notifyPublishReceipt` → `complete()`。**INV-X2-5: sole completion writer**（`notifyPublishReceipt` の呼び出し元はこの1箇所のみ — grep 確認）。
- **reader（waitFor()）**: 複数 Producer スレッド（§4 の5箇所）。
- **単一 writer / 複数 reader** を mutex + CV で正しく同期。SPSC ではなく、**mutex 保護の共有 watermark**。

## 9. mutex / atomic / CV の ownership

| 変数 | 型 | 保護 | 役割 |
| --- | --- | --- | --- |
| `PublishReceiptWaiter::mutex_` | `std::mutex` | — | watermark 保護 |
| `PublishReceiptWaiter::cv_` | `std::condition_variable` | mutex_ | waitFor の wake |
| `PublishReceiptWaiter::lastCompleted_` | plain `uint64_t` | mutex_ | **Completion watermark** |
| `m_lastObservedSequence`（Orchestrator.h:246） | `std::atomic` | — | P1-6 stall observer（別 semantic） |
| `lastCommittedPublicationSequence_`（AudioEngine.h:2228） | `std::atomic` | — | **Committed**（≠ Completed — INV-ISR-05） |

- **atomic は不要**（mutex 保護のため）。sparse 化しても mutex 保護は維持される。

## 10. E-1.9-B（event-driven quarantine wakeup）との相互作用

- **E-1.9-B は実装済み**（2026-08-15 監査 → GO → 実装）:
  - `ISRRetireRouter.h:309,321,363` — drain wake primitive（Non-RT only）
  - `ISRRetireRouter.cpp:351,432,455` — 単一 signal point（Q/E/T 受信時）/ CoordinatorLoop への signal / drain predicate wait
  - `ISRCoordinatorLoop.cpp:41` — event-driven wake + 1ms fallback timeout
  - `AudioEngine.Threading.cpp:215,292` — Phase2: Deferred retire drain（Q/E/T）
  - `RetireGraceSemanticsTests.cpp:401` — wake protocol tests
- **相互作用なし**: E-1.9-B の wake primitive は **quarantine drain 専用**（`ISRRetireRouter` 所有）。`PublishReceiptWaiter::cv_` は **publication completion 専用**（publish/complete）。両者は独立した CV で、E-1.9-B 監査レポート（`evidence/phase-e-1-9-b-event-driven-quarantine-wake-audit.md:37-39`）も「unrelated purposes」と明記。
- sparse completion 化が E-1.9-B に影響を与える経路は**無い**（逆も同様）。

## 11. 現在のテストと未カバーな interleaving

| テスト | 対象 |
| --- | --- |
| `PublishPipelineIntegrationTests.cpp` `testPublishCompletionMonotonicity`（:250） | contiguous FIFO completion（8 publishes、INV-X2-6） |
| 同 `testRebuildPublishCompletes` / `testIdlePublishViaFacade` / `testTransitionPublish` / `testTeardownPublish` | 統合パイプライン |
| `SequenceArithmeticTests.cpp`（34 assertions） | 正常8 / out-of-order4 / duplicate5 / wraparound8 / antipode5 / seqDistance4 — **primitive のみ** |
| `SoakPublishIntegrationTests.cpp` / `WorldRetirementMeasurementTests.cpp` | soak / measurement |

- **未カバーな interleaving**: PublishReceiptWaiter **統合レベル**の out-of-order（11→10）/ duplicate（10→10）/ wraparound。これらは INV-X2-6（FIFO）下で**architecturally impossible** のため統合テスト対象外。primitive 群（`SequenceArithmetic.h`）は SequenceArithmeticTests でカバー済み。

## 12. sparse completion 化で実際に削減される計算量・状態

- **削減されるものは無い**。現行は O(1) watermark（スキャン・集合なし）。
- sparse 化は `completedThrough_`（contiguous frontier）+ `completedOutOfOrder_`（sparse set）を**追加**し、`waitFor` に集合メンバーシップ判定を**追加**する → **状態・計算量とも増加**。
- sparse completion は**性能最適化ではなく、out-of-order completion 許容時の正しさ担保**のための構造。

## 13. API / semantic change の必要性

| 項目 | 現行 | sparse 化後 | 変更量 |
| --- | --- | --- | --- |
| `complete(seqId)` シグネチャ | 不変 | 不変 | 0 |
| `waitFor(seqId, timeoutMs)` シグネチャ | 不変 | 不変 | 0 |
| 内部状態 | `lastCompleted_` 単一 | `completedThrough_` + `completedOutOfOrder_` | **要変更** |
| `waitFor` 完了判定 | `isCompleted(seq, watermark)` | frontier 到達 OR sparse set 所属 | **要変更** |
| timeout / shutdown 契約 | Transferred 扱い | 不変 | 0 |
| リスク | — | **高**（completion 意味論の変更。contiguous 前提が architecture に埋め込み） | plan §1.5 明記 |

## 14. 現行実装のままで invariant が成立しているか

| Invariant | 内容 | 現行で成立 |
| --- | --- | --- |
| INV-X2-4 | stale completion は newer を上書きしない（`isAfter` 判定） | ✅ |
| INV-X2-5 | sole completion writer（`onPublishCommitted` のみ） | ✅（grep 確認） |
| INV-X2-6 | completion order == publication sequence order（FIFO） | ✅（PublishExecutor sole gateway + intentQueue_ FIFO + 単一 CoordinatorLoop consumer） |
| INV-ISR-05 | committed ≠ completed（`lastCommittedPublicationSequence_` と分離） | ✅ |
| INV-X2-1 | seqId 採番は単調増加 | ✅（`publicationSequenceCounter_` fetchAdd） |

- **全 invariant が現行実装で成立**。FIFO 前提は PublishExecutor sole gateway + MpscBoundedRing（reservation order FIFO）+ 単一 CoordinatorLoop consumer で構造的に保証。

---

## 判定: NO-GO（現時点では実装しない）

### 根拠

1. **削減対象が存在しない**。現行は O(1) watermark で dense scan / full-range scan は無い。sparse 化は状態（sparse set）と計算量を**増加**させるだけで、性能 benefit はゼロ。
2. **plan §1.5 自身が「現状は実装不要」と明記**。リスクは「**高**（completion 意味論の変更。contiguous 前提が architecture に埋め込まれている）」。
3. **問題が存在しない**。単一 completion writer（INV-X2-5）+ FIFO completion（INV-X2-6）が構造的に保証され、全 invariant が現行で成立。scalability / latency 問題は確認できない。
4. **sparse completion は性能最適化ではなく正しさ担保**。MPSC completion / parallel publish / async completion を許容する**将来のアーキテクチャ変更**が唯一の動機であり、現状その引き金は存在しない（Phase F の MPSC 化と同型）。
5. **「計画上の記述だけを根拠に実装しない」**（ユーザー指示）。§1.5 は「将来保留」項目であり、実装トリガーは未発生。

### 条件付き GO（将来のトリガー）

以下のいずれかが発生した時点で sparse completion 化を実施する:

- **第2の completion writer**（例: MPSC completion / parallel publish / 複数スレッドからの async completion）の設計が確定したとき。
- その際の実施内容（plan §1.5 準拠）:
  1. `completedThrough_`（contiguous frontier）+ `completedOutOfOrder_`（sparse set）を導入
  2. `complete()` を frontier + sparse set 更新に変更（INV-X2-5 維持）
  3. `waitFor()` を frontier + sparse 併用に変更
  4. 統合テスト追加: out-of-order（11→10）/ duplicate（10→10）/ wraparound（UINT64_MAX 近傍）
  5. 既存 contiguous テスト（PublishPipelineIntegrationTests / PublishCompletionMonotonicity）維持 + INV-X2-6 architectural test 維持

### 推奨アクション

- 本監査結果を `AudioEngine.h:3679-3683`（PublishReceiptWaiter のコメント）に追記し、判定（NO-GO + 条件付き GO トリガー）を文書化する。
- §1.5 は backlog の「将来保留」として維持し、実装は開始しない。
- 現行の mutex + CV + monotonic watermark は**変更しない**（正しく、O(1)、全 invariant 成立）。
