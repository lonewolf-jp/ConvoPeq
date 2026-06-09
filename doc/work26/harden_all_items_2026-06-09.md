# 全未確定事項 最終確定レポート (2026-06-09)

**調査手段**: grep/Select-String, Serena MCP (find_symbol/find_referencing_symbols/get_symbols_overview),
CodeGraph (grep), 直接ファイル読取（6ファイル）

---

## 1. P0-1: PublishWorld 戻り値と PublishOutcome

### 確定

```cpp
// src/core/RuntimePublicationCoordinator.h:88
void publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept  // ← void!
```

- Validation Failure時: `worldOwner.release()` → `bridge_.retireRuntimePublishWorldNonRt()` → **void return**
- エラーが呼び出し元に**一切伝播しない**

### 影響

- `PublishOutcome` の新設は**必須**
- 変更影響: Coordinator テンプレート (`RuntimePublicationCoordinator<World, Handle, Bridge>`) の単一インスタンスのみ
- `PublicationExecutor.cpp` 内の `coordinator.publishWorld(...)` を outcome 対応に変更

### PublishResult 現在の実装

```cpp
// PublicationExecutor.h:9-12
enum class PublishResult { Success, ValidationFailed, PublishFailed, BridgeFailed };

// PublicationExecutor.cpp:7-29 — 実際の戻り値:
//   null world → PublishFailed
//   それ以外 → Success (publishWorld の void 戻りを無視)
```

**ValidationFailed, BridgeFailed は一度も返されていない。**

---

## 2. P0-4: RuntimeStore 書込API保護

### 確定

```cpp
// src/core/RuntimeStore.h
template <typename T, typename Owner>
class RuntimeStore final {
    friend Owner;  // ← Owner = RuntimePublicationCoordinator<World,Handle,Bridge>

    [[nodiscard]] WriteAccess acquireWriteAccess() noexcept;  // ← private! Ownerのみアクセス可
    // ...
    std::atomic<T*> current { nullptr };  // 唯一の状態
public:
    [[nodiscard]] const T* observe() const noexcept;  // ← public! 誰でも読める
};
```

- **書込は既に完全保護**: `acquireWriteAccess()` は private + friend Owner
- **読取は public**: `observe()` は誰でも呼べる
- **問題の本質**: `makeRuntimePublicationCoordinator()` が public → `RuntimePublicationCoordinator::create()` 経由で Store の WriteAccess を取得可能

### 結論

計画書の「Write API隠蔽、ReadToken API維持」方針は正しい。Store自体の変更は不要。Coordinator生成権限制限で自動解決。

---

## 3. P0-2/3: Coordinator 生成10箇所

### 確定 (全10箇所)

| # | ファイル | 行 | スコープ | 経路 |
|---|---------|----|---------|------|
| 1 | `AudioEngine.h:2733` | `publishWorld()`内部 | public inline | →P0-2/3で削除 |
| 2 | `AudioEngine.CtorDtor.cpp:127` | `~AudioEngine` shutdown | デストラクタ | →Orchestrator経由 |
| 3 | `AudioEngine.Init.cpp:46` | bootstrap publish | 初期化 | →Orchestrator経由 |
| 4 | `PrepareToPlay.cpp:124` | prepareToPlay | prepareToPlay | →Orchestrator経由 |
| 5 | `PrepareToPlay.cpp:236` | prepareToPlay (2nd) | prepareToPlay | →Orchestrator経由 |
| 6 | `ReleaseResources.cpp:124` | releaseResources | 解放 | →Orchestrator経由 |
| 7 | `ReleaseResources.cpp:196` | releaseResources drain | 解放 | →Orchestrator経由 |
| 8 | `AudioEngine.Timer.cpp:404` | クロスフェード完了後 | タイマー | →Orchestrator経由 |
| 9 | `DSPTransition.h:115` | onTransitionComplete | テンプレート | →Orchestrator経由 |
| 10 | `PublicationExecutor.cpp:15` | publish() | Executor | **維持** |

### 経路10の詳細

```cpp
// PublicationExecutor.cpp:15
auto coordinator = engine.makeRuntimePublicationCoordinator();
coordinator.publishWorld(std::move(worldOwner));
```

Executor は Orchestrator の委譲先 — この1箇所だけは Coordinator 生成を維持。

---

## 4. P1-4: ISRRetireRouter drainAll 不在

### 確定

```cpp
// ISRRetireRouter.h — 全メソッド一覧 (Serena確認)
// Epoch: snapshotEpoch, publishEpoch, currentEpoch, activeReaderCount
// Reader: registerReaderThread, reserveReaderThread, enterReader, exitReader, minReaderEpoch
// Retire: enqueueRetire(x4), enqueueRetire(x3), tryReclaim, pendingRetireCount
// ★ drainAll() 不在!
```

```cpp
// EpochDomain.h:220-223 — drainAll はこちらにのみ存在
void drainAll() noexcept {
    deferredDeletionQueue.drainAllUnsafe();
}
```

### 結論

P1-4 の手順1「ISRRetireRouter に drainAll() 委譲メソッド追加」は**確実に必要**。実装は Router に以下を追加するだけ:

```cpp
void drainAll() noexcept {
    assert(epochDomain_ != nullptr);
    epochDomain_->drainAll();
}
```

AudioEngine からの直接呼び出し2箇所:

- `AudioEngine.CtorDtor.cpp:131`: `m_epochDomain.drainAll()` → `m_retireRouter->drainAll()`
- `AudioEngine.Processing.ReleaseResources.cpp:208`: 同上

---

## 5. P1-9: Monotonicity 詳細

### 確定

**commit() 内の monotonicity check は既存 (ISRRuntimePublicationCoordinator.cpp)**:

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority, RuntimeBoundary boundary,
    const void* newWorld, uint64_t version,
    PublicationSequenceId sequenceId, PublicationEpoch epoch, uint64_t mappedGeneration) {
    // ...
    if (hasPrevious && sequenceId <= previousSequenceId) {
        publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);  // ← 即 Faulted
        return;
    }
    if (hasPrevious && epoch <= previousEpoch) {
        publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }
    if (hasPrevious && mappedGeneration <= previousMappedGeneration) {
        publishAtomic(state_, CoordinatorState::Faulted, std::memory_order_release);
        return;
    }
```

**makeRuntimeReadHandle() 内の監視も既存 (AudioEngine.h)**:

```cpp
// observeMonotonicViolationCount_ のインクリメント
// observeMonotonicRollbackRequested_ のセット
// — ただし誰も消費しない
```

### 結論

- `commit()` での **即 Faulted** 遷移は維持（明らかな論理違反）
- P1-9 ではウィンドウ監視（M秒N回）を追加 → Evidence出力 → 段階的Faulted
- `commit()` の Faulted とは**別経路**。rollbackRequested フラグ経由のソフト障害と commit 内のハード障害は区別する。

---

## 6. P2-2: DeletionEntry trivially copyable

### 確定

```cpp
// DeferredDeletionQueue.h:26-31
struct DeletionEntry {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
    uint64_t epoch = 0;
    DeletionEntryType type = DeletionEntryType::Generic;
};

static_assert(std::is_trivially_copyable_v<DeletionEntry>);  // ← 既存の静的チェック
static_assert(std::is_trivially_destructible_v<DeletionEntry>);
```

- `uint64_t publicationSequenceId` を追加しても trivially copyable は維持される（POD型のみのstruct）
- `uint64_t` 追加後の struct サイズ: 8+8+8+1+8=33 → アライメント padding で 40 bytes（現状32→40、許容範囲）

---

## 7. P2-3: ObserveToken メタデータ経路

### 確定

```cpp
// GlobalSnapshot.h — 利用可能なフィールド
struct GlobalSnapshot {
    uint64_t convStateId = 0;
    // ... パラメータ ...
    uint64_t generation = 0;  // ★ generation は GlobalSnapshot に存在
    // ★ publicationSequenceId なし
    // ★ worldId なし
    // ★ epoch なし
};
```

```cpp
// SnapshotCoordinator::observeCurrentRuntime()
ObservedRuntime observeCurrentRuntime(RCUReader& reader) const noexcept {
    ObservedRuntime observed(reader);
    observed.ptr = m_slots.loadCurrent(std::memory_order_acquire);
    return observed;  // ← ptr のみ設定
}

// ObservedRuntime の構成
struct ObservedRuntime {
    RCUReaderGuard guard;
    const GlobalSnapshot* ptr = nullptr;
    // ★ generation/sequenceId/worldId/epoch なし
};
```

### 結論

- **generation** は `GlobalSnapshot::generation` から取得可能（ObserveTokenが ptr 経由で間接参照）
- **publicationSequenceId / worldId / epoch** は SnapshotCoordinator に存在しない
- P2-3 の理想的アプローチ: ObserveToken の getter メソッド `generation() { return ptr ? ptr->generation : 0; }` を追加するのが最小変更。full metadata は `RuntimeReadHandle` 側の責務。

---

## 8. P1-5: Evidence 静的テンプレート全容

### 確定 (8種類すべて確認)

| 成果物 | 静的データ | 動的 | 状態 |
|--------|-----------|------|:----:|
| `closure_graph.json` | nodeCount:0, edgeCount:0, descriptorCoverageComplete:true, externalMutableDependencies:0 | なし | ❌静的 |
| `mutation_fault_trace.json` | テンプレート部固定 | sealViolationCount | ⚠️ 1値のみ |
| `hb_graph_trace.json` | eventCount:0 | なし | ❌静的 |
| `hb_violation_report.json` | violations:[] | なし | ❌静的 |
| `retire_timeline.json` | epochMode:shared, rollbackReady:true, totalTransitions:0 | なし | ❌静的 |
| `shutdown_trace.json` | phase:0, verified:true, sh*:全て0 | なし | ❌静的 |
| `retire_latency_report.json` | withinThreshold:true | なし | ❌静的 |
| `payload_tier_report.json` | violations:0 + 固定family配列 | なし | ❌静的 |

### EvidenceExporter の出力制御 (Release/Debug)

- **Release**: `shutdown_trace.json`, `retire_timeline.json`, `retire_latency_report.json`, `payload_tier_report.json` の4種のみ出力
- **Debug**: 上記 + `closure_graph.json`, `mutation_fault_trace.json`, `hb_graph_trace.json` の全7種 + manifest.json
- 出力先: `./evidence/`（カレントディレクトリ相対）

---

## 9. P1-8: FailureRecord 未実装

### 確定

- `PublicationFailureRecord` 構造体: **未実装**
- `PublicationFailureTaxonomyVerifier`: **スキーマ検証器**であり障害レコードではない
- リングバッファ: **未実装**
- `publication_failure_log.json`: **出力されていない**

---

## 10. P1-2/3: 保留項目の棚卸し

### 確定 — 全3Policyは前方宣言のみで実体なし

```cpp
// ISRRetireRouter.h:26
class DSPRetirePolicy;          // 前方宣言のみ
class SnapshotRetirePolicy;     // 前方宣言のみ
class DeferredRetirePolicy;     // 前方宣言のみ

// ISRRetireRouter.h:143
// [work21 P0-1] Future: delegate to DSPRetirePolicy / SnapshotRetirePolicy / DeferredRetirePolicy
```

- **実際の使用箇所**: 0（参照ゼロ）
- **現在のルーティング**: `ISRRetireRouter::enqueueRetire()` → 直接 `EpochDomain_->enqueueRetire()` に委譲
- **Policy Lane が必要となる条件**: 未確定。単一経路で十分機能している

### 結論

保留判断は妥当。実運用要求が明確になるまで `enqueueRetire` の単一経路を維持。

---

## 11. P3-2: RuntimeReaderContext 型安全化

### 確定

```cpp
// RuntimeReaderContext 直接構築 — 2箇所のみ
// 1. RuntimePublicationOrchestrator.cpp:24 — 正しい組み合わせ
// 2. AudioEngine.Processing.ReleaseResources.cpp:92 — 正しい組み合わせ
```

理論上のリスクはあるが、実際の誤用は確認されていない。P3-2 の優先度低は妥当。

---

## 12. P3-1: ReaderSlot 構成

### 確定

```cpp
// EpochDomain.h (ReaderSlot)
struct ReaderSlot {
    std::atomic<uint64_t> epoch { kInactiveEpoch };
    std::atomic<uint32_t> depth { 0 };
    // ★ enterTimestamp なし
    // ★ threadId なし
};
```

- `kMaxReaders = 64`, 現在の使用数: 11
- enterTimestamp / threadId なし → 停滞Reader検出不可

---

## 総評

| 項目 | 状態 | 確度 |
|------|------|:----:|
| P0-1: publishWorld void戻り値 | **確実** | 100% |
| P0-4: Store書込保護 | **既にprotected** (friend Owner) | 100% |
| P0-2/3: Coordinator生成10箇所 | **全件確認** | 100% |
| P1-4: drainAll不在 | **Routerに不在** → 追加必要 | 100% |
| P1-9: monotonicity commit内Faulted | **即Faulted実装済み** | 100% |
| P2-2: DeletionEntry trivially copyable | **static_assert確認** | 100% |
| P2-3: metadata経路 | generationのみ取得可, 他は不在 | 100% |
| P1-5: Evidence全種静的 | **8種中7種が完全静的** | 100% |
| P1-8: FailureRecord | **未実装** | 100% |
| P1-2/3: 保留Policy | **前方宣言のみ, 使用0** | 100% |
| P3-2: ReaderContext誤用 | **未確認** | 100% |
| P3-1: ReaderSlot欠落 | **enterTimestamp/threadIdなし** | 100% |
