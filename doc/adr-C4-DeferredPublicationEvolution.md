# ADR-C4 — Deferred Publication Evolution (notifyTransitionComplete 撤去と stale discard の位置づけ)

**Status:** Accepted (implemented)
**Date:** 2026-08-05
**Context:** ISR publish 経路の同期一本化リファクタリング。旧設計の `RuntimePublicationOrchestrator::notifyTransitionComplete` はクロスフェード完了時に保留中の deferred publish を再送する統合フックだったが、Builder/Coordinator 分離によりその責務はすべて別層へ移譲され、関数は dead code 化した。

## Decision
- **`notifyTransitionComplete()` は削除する。** 呼び出し元ゼロ (grep 確認済み)。
- **関数を置換した経路を唯一の正式経路とする:**
  ```
  CoordinatorLoop (Decision/Notify)
      ↓ publishRetryReady (rebuildMutex 保護 bool)
  RebuildThread (Builder)
      ↓ consumeDeferredRequest() → submitPublishRequest()
  ```
- **stale discard の設計知見はコードコメントではなく本 ADR に移管する。**
  - 半年後の読者にとって「なぜ dead code があるか」より「なぜこの層にあるべきか」の方が価値がある。
- **stale discard (generation + publication sequence guard) は Transition Completion の責務ではなく Admission Policy の責務として整理する。**

## 旧責務の分解と移譲先

`notifyTransitionComplete` が担っていた 4 責務の行き先:

| # | 旧責務 | 移譲先 | 状態 |
|---|--------|--------|------|
| 1 | Transition Completion (`transition_.onTransitionComplete`) | `AudioEngine::publishIdleWorldOnly()` (Timer が直接呼ぶ) | 移譲済み |
| 2 | Shutdown Guard (deferred キャンセル) | `clearDeferredForShutdown()` | 分離済み |
| 3 | Stale Discard (generation + sequence 二重ガード) | → **Admission Policy へ再配置 (下記)** | アルゴリズムのみ再設計対象 |
| 4 | Deferred Publish Submit | `consumeDeferredRequest()` → `submitPublishRequest()` (RebuildThread) | 完全移行済み |

## stale discard の設計知見（資産）

以降の Layer 2/3 統合では関数 API ではなく下記アルゴリズムを再利用する。

### 判定フロー（旧 `notifyTransitionComplete` 内）
TTL 超過 → generation 検査 → publication sequence 検査 → 有効なら submit。

**TTL 超過 (最優先):**
```
deferred.enqueueTimestampUs != 0
&& (nowUs - enqueueTimestampUs) > kDeferredPublishTTLUs
→ DiscardReason::Expired
```

**Generation Guard:**
```
deferred.guard.generation != 0
&& deferred.guard.generation != currentGen (engine_.rebuildRequestGeneration)
→ DiscardReason::StaleDiscard
```

**Publication Sequence Guard:**
```
deferred.guard.sequence < currentPubSeq (getLastCommittedPublicationSequence)
→ DiscardReason::StaleDiscard
```

### 設計指針（本 ADR による確定）
- ISR では `Transition` / `Admission` / `Publication` は**別 Layer**。
- したがって stale discard を Transition Completion に置くのは設計上の偶然であり、Layer 分離後は不適切。
- generation + sequence による拒否判定は **`submitPublishRequest()` の Admission (`PublicationAdmission::evaluate`) 層**に属する。

## Consequences
- `notifyTransitionComplete` / `DSPTransition::onTransitionComplete` は呼び出し元ゼロ (dead code 化; `onTransitionComplete` は今後の Layer 2/3 統合で整理予定)。
- `RuntimePublicationOrchestrator::transition_` / `transition()` は ProcessIntent から使用継続 (削除しない)。
- stale discard 実装を Admission 層へ統合する際は、本 ADR の判定フローを再実装の仕様とする。