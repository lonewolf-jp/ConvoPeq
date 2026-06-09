# Practical Stable ISR Bridge Runtime レビュー検証レポート

**作成日**: 2026-06-09
**検証対象**: 10件のレビュークレーム
**検証方法**: ソースコード実装との突合

---

## 全体評価

| 項目 | 判定 | 正確性 |
| --- | --- | --- |
| 全10件中 8件 完全一致 | ✅ 正確 | 100% |
| 全10件中 2件 部分的に正確 | ⚠️ 部分的 | 問題の本質は妥当だが事実認識に一部誤り |

---

## Claim 1: EpochDomain依存がまだ完全除去されていない — ✅ **正確**

### エビデンス

`AudioEngine.CtorDtor.cpp:21`:

```cpp
, m_coordinator(m_epochDomain)
#pragma warning(push)
#pragma warning(disable : 4996)  // [[deprecated]] — transitional
```

`SnapshotCoordinator` のコンストラクタは `IEpochProvider&` を受け付けるため `ISRRetireRouter`（同じく `IEpochProvider`）が渡せるが、現状は `EpochDomain` を直接注入している。

### 追加所見

`EQProcessor.Core.cpp` でも EpochDomain 直接呼び出しが多数残存:

- `m_epochDomain.enqueueRetire(...)`
- `m_epochDomain.publishEpoch()`
- `m_epochDomain.currentEpoch()`
- `m_epochDomain.tryReclaim()`
- `m_epochDomain.drainAll()`

EpochDomain 依存は AudioEngine 層に限定されず、他のコンポーネントにも広がっている。

---

## Claim 2: Router が単なる薄いラッパのまま — ✅ **正確**

### エビデンス

`ISRRetireRouter.h`: 全メソッドが EpochDomain への直接転送:

```cpp
uint64_t publishEpoch() noexcept override {
    return epochDomain_->advanceEpoch();
}
void enterReader(int readerIndex) noexcept override {
    epochDomain_->enterReader(readerIndex);
}
void exitReader(int readerIndex) noexcept override {
    epochDomain_->exitReader(readerIndex);
}
```

コード内に Future コメントあり:

```cpp
// [work21 P0-1] Future: delegate to DSPRetirePolicy / SnapshotRetirePolicy / DeferredRetirePolicy
```

`DSPRetirePolicy`, `SnapshotRetirePolicy`, `DeferredRetirePolicy` は **前方宣言のみで実装なし**。Policy 委譲構造は未構築。

---

## Claim 3: PublicationExecutor が未完成 — ✅ **正確**

### エビデンス

`PublicationExecutor.cpp`:

```cpp
PublishResult PublicationExecutor::publish(...) noexcept
{
    // Phase 1: Validate (via bridge)
    {
        // Use existing bridge through coordinator's publishWorld logic
        // We extract validation by attempting publish and catching failure
        // For PR-1, we use the existing publishWorld path
    }
    // ↑ 検証ブロック: コメントのみで何も実行されない

    // Phase 2: PublishAndSwap (use existing coordinator)
    coordinator.publishWorld(std::move(worldOwner));
    // ↑ coordinator.publishWorld() に丸投げ

    // NOTE: For PR-1, we delegate to the existing coordinator.publishWorld().
    // In PR-3, this will be replaced with direct store/bridge access.
    return PublishResult::Success;
}
```

`Validate → Admission → Authority Check → Publish → Retire → Evidence` のパイプラインのうち、実装されているのは Publish のみ。

---

## Claim 4: Deprecated経路がまだ生存 — ✅ **正確**

### エビデンス

`EpochDomain.h` の `[[deprecated]]` API 一覧:

| API | 属性 | 代替先 |
| --- | --- | --- |
| `enterReader(int)` | `[[deprecated]]` | `RCUReader::enter()` |
| `exitReader(int)` | `[[deprecated]]` | `RCUReader::exit()` |
| `advanceEpoch()` | `[[deprecated]]` | `Router::publishEpoch()` |
| `enqueueRetire(...)` | `[[deprecated]]` | `ISR RuntimePublicationCoordinator::enqueueRetire` |
| `reclaimRetired()` | `[[deprecated]]` | `tryReclaim()` |

`ISRRetireRouter.h` 上での抑制:

```cpp
#pragma warning(push)
#pragma warning(disable : 4996)  // [[deprecated]] — transitional, Router wraps EpochDomain
    return epochDomain_->advanceEpoch();
#pragma warning(pop)
```

---

## Claim 5: RuntimeReaderContext が型安全ではない — ✅ **正確**

### エビデンス

`RuntimeReaderContext.h` 自身が認める:

```cpp
// ■ 型安全性の限界
// C++ の型システムでは reader と channel の組み合わせの正当性は保証できない。
// 例えば以下の誤った組み合わせがコンパイルを通ってしまう:
//   RuntimeReaderContext{ messageThreadRcuReader, ObserveChannel::Audio }; // 誤りだがコンパイル可能
```

ヘルパー関数（`makeAudioReaderContext()` 等）は存在するが、コンパイル時型強制はされず運用依存。

---

## Claim 6: Readerスロット固定64 — ✅ **正確**

### エビデンス

`EpochDomain.h:53`:

```cpp
static constexpr int kMaxReaders = 64;
```

動的スロット確保機構（`ReaderSlotPool` 等）は未実装。Worker 増加・テスト拡張・将来機能追加で枯渇リスクあり。

---

## Claim 7: Retire Queue に mutex ベース経路が残る — ⚠️ **部分的に正確**

### 不正確な点

レビューは「EpochDomain: fallbackMutex / fallbackQueue」と主張するが、`EpochDomain` は **`DeferredDeletionQueue`（ロックフリー MPMC）** のみを使用。`fallbackMutex` / `fallbackQueue` は EpochDomain に存在しない。

### 妥当な点

コードベースに mutex ベースの retire 経路が存在するのは事実:

| ファイル | 該当箇所 |
| --- | --- |
| `SafeStateSwapper.h:364` | `std::mutex fallbackMutex` + `std::priority_queue<FallbackEntry> fallbackQueue` |
| `DeletionQueue.h:23` | `std::mutex mutex;`（旧実装・使用状況要確認） |
| `DeferredRetireFallbackQueue.h` | `std::mutex mutex_;`（定義のみ・未使用） |

Queue Pressure → Fallback → 大量蓄積 → Shutdown 遅延のリスク評価自体は妥当。

---

## Claim 8: ObserveToken が実質 Snapshot ポインタ保持のみ — ✅ **正確**

### エビデンス

`ObservedRuntime.h` の実メンバ:

```cpp
struct ObservedRuntime {
    RCUReaderGuard guard;
    const GlobalSnapshot* ptr = nullptr;
#ifndef NDEBUG
    std::thread::id ownerThreadId;
#endif
};
`generation`, `publicationId`, `worldId` は一切保持していない。Reader が「何を観測したか」の追跡能力は不足。

---

## Claim 9: Closure Graph が Runtime 強制に使われていない — ⚠️ **部分的に正確**

### 不正確な点

Closure Validation Failure は実際に publish を拒否する経路が存在する:

- `AudioEngine.Commit.cpp:301-304`:

  ```cpp
  const bool closureValid = closureGraphWalker_.validateGraph(closure);
  if (!closureValid || !precheckValid) {
      return rejectWithEvidence("closure_or_precheck_invalid");
  }
  ```

- `ISRRuntimePublicationCoordinator.cpp:28-31`:

  ```cpp
  ClosureValidator closureValidator;
  if (!closureValidator.validateClosureGraph(closure)) {
      publishAtomic(lastRejectCode_, RejectCode::InvalidClosure, ...);
      return false;
  }
  ```

### 妥当な点

`PublicationAdmission::evaluate()` 内では Closure validation が**呼ばれていない**。Admission 層と Closure validation が分離しており、統合は未完了。

---

## Claim 10: Authority単一化が未完了 — ✅ **正確**

### エビデンス

Authority 的責務を持つクラスが複数並立:

| クラス | ファイル | 役割 |
| --- | --- | --- |
| `RuntimePublicationCoordinator` (template) | `core/RuntimePublicationCoordinator.h` | テンプレート版 Coordinator |
| `RuntimePublicationOrchestrator` | `audioengine/RuntimePublicationOrchestrator.h` | 上位オーケストレータ |
| `ISRRuntimePublicationCoordinator` | `audioengine/ISRRuntimePublicationCoordinator.h` | ISR 版 Coordinator |
| `PublicationExecutor` | `audioengine/PublicationExecutor.h` | 実行 |
| `PublicationAdmission` | `audioengine/PublicationAdmission.h` | 判定 |
| `ISRRetireRouter` | `audioengine/ISRRetireRouter.h` | Retire 入口 |

レビューの図示する単一パス:

```text
RuntimePublicationOrchestrator → PublicationAdmission → RuntimePublicationCoordinator → RuntimeStore
```

は現時点で完全に確立していない。

---

## レビュー達成度評価の検証

レビューの総評「**達成度およそ 80〜90%**」は **概ね妥当**。

実際のコード確認所見:

- ✅ RCUReader 導入
- ✅ IEpochProvider 分離設計
- ✅ ObserveToken 導入
- ✅ RuntimePublicationCoordinator 導入
- ✅ Closure Graph 導入（検証は一部機能）
- ✅ RetireRouter 導入（ラッパー状態だが入口は統一）
- ❌ 移行用暫定経路（deprecated/transitional path）が残存
- ❌ Authority 単一化未完了
- ❌ PublicationExecutor 未完成

**最高優先度4項目**（EpochDomain直接依存の除去、Deprecated API全廃、PublicationExecutor完成、Authority単一化）の選定も妥当であり、これらが完了すれば実運用に耐える構造に到達する。

---

## 補足: レビューにない観点

- `EQProcessor.Core.cpp` では EpochDomain 直接呼び出しが多数残っており、AudioEngine 以外のコンポーネント移行も必要
- `RuntimePublicationOrchestrator` は既に Admission + Executor + Transition を統合しており、Authority 単一化の基盤はできつつある
- `ISRRuntimePublicationCoordinator`（audioengine/）と `RuntimePublicationCoordinator`（core/ テンプレート版）の責務分担が不明確
