# notfinished7.md 検証レポート (2026-06-06)

## 検証方法

- Serena MCP: シンボル検索・参照関係・実装解析
- CodeGraph MCP (CodeGraphContext v0.4.13): 11397 entities indexed
- cocoindex-code (ccc): ASTベースコード検索
- graphify (DeepSeek backend): 11633 nodes, 14188 edges, 1207 communities
- 手動コードリーディング: SnapshotCoordinator/SnapshotFactory/EpochDomain/ObservedRuntime/ISRRetire/ISRRetireRuntimeEx/ISRDSPQuarantine/DeferredDeletionQueue/RuntimePublicationCoordinator
- **semble (コード検索ツール)**: `enqueueRetire`/`reclaimRetired`/`SnapshotFactory::destroy`/`ownerThreadId`/`CrossfadeRuntime`/`SnapshotRetireManager` の全出現箇所を追跡

## 検証結果：各クレームの評価

### ① Epoch/RCUとSnapshotライフサイクルの「責務分散」

**評価: 部分的に正しいが、状況はnotfinished7.mdの記述より改善・複雑化している**

**正しい点**:

- SnapshotCoordinator → `EpochDomain::enqueueRetire()`（deprecated）経由で retire 発行
- EpochDomain → `DeferredDeletionQueue` で実際の deferred delete 管理
- ISRRetireRuntimeEx → 独自の retireIntent/reclaim/quarantine 経路を持つ
- ISRDSPQuarantine → 別個の quarantine 管理
- DSPQuarantineManager → さらに別個の隔離管理

**semble追認結果: enqueueRetireの呼び出し経路は3系統**

| 呼び出し元 | 呼び出し先 | 備考 |
|---|---|---|
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` | `m_epochDomain.enqueueRetire()` (deprecated) | 2回試行、間に reclaimRetired() |
| `ISRRuntimePublicationCoordinator::enqueueRetire()` | `m_epochDomain.enqueueRetire()` (deprecated) | authorized caller |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | `m_epochDomain.enqueueRetire()` (deprecated) | retry 4回＋reclaimRetired |

**notfinished7.mdに見落とし**:

- **`SnapshotRetireManager` は既に抽出済み**（Phase 5 で retire authority 分離のため別ファイル化）だが、**参照元がゼロ**（semble確認）。未統合の orphan クラス。notfinished7.md はこの存在に言及していない。
- `EpochDomain::enqueueRetire()` には `[[deprecated]]` が付いており、`ISRRuntimePublicationCoordinator::enqueueRetire()` が authorized caller として移行中。
- **Reclaim 経路は以下の6箇所**から呼ばれている（semble追認済み）：
  1. `AudioEngine::tryReclaimResources()` (AudioEngine.Threading.cpp:75) — **RTタイマーコールバックから呼ばれうる**
  2. `AudioEngine::drainDeferredRetireQueues()` (88, 221行) — シャットダウン drain
  3. `SnapshotCoordinator::~SnapshotCoordinator()` (67行) — デストラクタ
  4. `SnapshotCoordinator::reclaim()` (103行) — coordinator経由の委譲
  5. `EQProcessor::enqueueDeferredDeleteWithFallback()` (44, 63, 118行) — EQ側の独自回収
  6. **`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` (3170-3206行) — fallback回収** ★sembleで確認

### ② SnapshotCoordinator が「状態機械＋メモリ管理＋フェード制御」を兼務（God Object）

**評価: 正しい**

**実際の構成**:

- `SnapshotSlotStore` は既に別クラスに抽出済み → `SnapshotCoordinator::m_slots` フィールド
- `SnapshotFadeState` は既に別クラスに抽出済み → `SnapshotCoordinator::m_fade` フィールド
- しかし `SnapshotCoordinator` は依然として以下を同一クラス内で直列実行：
  - slot swap: `m_slots.exchangeCurrent()`, `m_slots.exchangeTarget()`
  - fade制御: `m_fade.start()`, `m_fade.resetToIdle()`
  - RCU retire: `m_epochDomain->enqueueRetire()`, `m_epochDomain->publish()`
  - lifecycle transition: destructor内の解放ロジック

**notfinished7.mdに見落とし**:

- 提案例の「SnapshotSlotManager」に相当する `SnapshotSlotStore` は **既に存在する**
- 提案例の「FadeController」に相当する `SnapshotFadeState` は **既に存在する**
- 提案例の「RetireScheduler」に相当する `SnapshotRetireManager` は **存在するが未使用（orphan）**
- `SnapshotCoordinator` が直接 `EpochDomain` の deprecated API を呼んでいる点は、移行途中であり設計上の認識はある

### ③ atomic ordering の意味が局所的で全体整合性が保証されていない

**評価: 正しい**

**根拠**:

- 各atomic操作には詳細な HB (happens-before) コメントが記述されている
- しかし `ISREpochMemoryModel.h` のようなグローバルな ordering model は存在しない
- `publication_ordering_matrix.md` に部分的な整序モデルは存在するが、コードとして実装されていない
- `std::atomic_thread_fence` は数箇所で使用されているが（ConvolverProcessor.Lifecycle.cpp:39, RuntimePublicationCoordinator.h:99）、体系化されていない

**notfinished7.mdに見落とし**:

- HBコメントは非常に詳細に記述されており、局所的には正確。notfinished7.md はこの精緻さを過小評価している。
- 実際のリスクは「局所的HBがグローバル順序を保証しない」こと自体より、「異なるretire経路間の順序保証がない」ことにある。

### ④ Snapshotの生成と比較ロジックが二重化されている

**評価: 設計上の意図があるため、notfinished7.mdの主張は過剰**

**実際の役割**:

- `computeContentHash()`: ビット厳密な高速ハッシュ。不一致の高速判定用（最適化目的）
- `areSnapshotsEquivalent()`: 許容誤差を含む実質的等価判定。ハッシュ衝突時のフォールバック
- `createImpl()`: 両者を組み合わせて「ハッシュ一致時のみ重い等価判定を実施（衝突対策）」という意図的な階層化

**リスク評価**:

- 機能の二重化というより **意図された2層設計**（ハッシュ＝否定用フィルタ、等価判定＝確定判定）
- ただし `areSnapshotsEquivalent()` と `computeContentHash()` で同じパラメータを走査しているため、メンテナンス時に両者の同期がずれるリスクは存在する

**notfinished7.mdに見落とし**:

- `create()` と `createImpl()` は責務が異なる（生生成 vs キャッシュ判定付き生成）
- 文書化された設計判断である（SnapshotFactory.h コメント参照）

### ⑤ Retire系処理がリアルタイム境界を跨いでいる

**評価: 正しい。ただしグラデーションがある**

**実態**:

- ✅ `RetireRuntime::emitRetireIntentRT()` — 明示的に RT-safe（lock-free queue, 名前で表明）
- ✅ `EpochDomain::enqueueRetire()` — RT-safe（lock-free MPMC queue）
- ⚠️ `EpochDomain::reclaimRetired()` → `DeferredDeletionQueue::reclaim()` — **Timerコールバック（AudioEngine::tryReclaimResources）**から呼ばれ、これは Audio Thread のタイマー上で動作しうる
- ❌ `ISRRetireRuntimeEx::reclaim()` — NonRT専用だが `AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` から呼ばれる（正しいNonRT文脈）
- ⚠️ `DSPQuarantineManager::reclaimSlot()` — reclaim権限が分散

**notfinished7.mdに見落とし**:

- `tryReclaimResources()` が `m_epochDomain.reclaimRetired()` を呼ぶ経路は、Audio Thread（Timer）上で動作するため、**本当のRT境界違反**になりうる
- `RetireRuntime` は RT/NonRT の意図的な分離設計（`emitRetireIntentRT` / `acknowledgeRetireCoordination`）を既に持っている

### ⑥ ObservedRuntime の安全性が「スレッドID依存」で脆弱

**評価: 正しい。ただし影響範囲は限定的**

**実態**:

- `ObservedRuntime::ownerThreadId` フィールドが存在
- `get()` で `ownerThreadId != std::this_thread::get_id()` チェック
- `operator bool()` でも同様のチェック
- このチェックは move semantics によりトークンが別スレッドに渡った場合の安全網

**リスク評価**:

- 実際には `EpochDomainReaderGuard` が epoch-based reader protection を提供しており、スレッドIDチェックは **二重の安全策（defense-in-depth）**
- しかし、std::thread::id の再利用問題（スレッド終了後、新スレッドが同じIDを得る可能性）に対しては脆弱
- coroutine / task system との互換性は確かに問題

**notfinished7.mdに見落とし**:

- `EpochDomainReaderGuard` （RAII epoch reader）が既に存在し、こちらが実質的な保護を提供している
- スレッドIDチェックは補助的な安全策であり、文書で主張されるほど「脆弱」ではない
- ただしコード品質としては改善が望ましい

### ⑦ fade state が snapshot lifecycle と密結合

**評価: 正しい**

**実態**:

- `SnapshotCoordinator::startFade()` 内で以下の処理が直列実行：
  1. `m_slots.exchangeTarget()` — snapshot slot swap
  2. `m_epochDomain->enqueueRetire()` — RCU retire
  3. `m_fade.start()` — fade開始
- `SnapshotCoordinator::completeFade()` 内でも同様に直列：
  1. `m_slots.exchangeTarget()`
  2. `m_slots.exchangeCurrent()`
  3. `m_epochDomain->enqueueRetire()`
  4. `m_fade.resetToIdle()`
- `resetFadeStateAndRetireTarget()` も同様に fade + retire が結合

**notfinished7.mdに見落とし**:

- fadeは `SnapshotFadeState` として既に分離抽出済みではない。これは `SnapshotCoordinator.h` のフィールド定義で確認済み。
  - 訂正: `SnapshotFadeState m_fade;` として既に独立クラス。notfinished7.mdはこれを認識している（コード引用内に `m_fade.start()` がある）。
  - 問題は「クラスが分離されていても、呼び出し側で直列に結合されている」こと。

## 総合評価

### notfinished7.md の正確性

| クレーム | 正確性 | 補足 |
|---------|--------|------|
| ① 責務分散 | ⚠ 部分的に正確 | SnapshotRetireManager の存在・deprecated移行状況を未記載 |
| ② God Object | ✅ 正確 | ただし抽出作業の進捗（SlotStore/FadeState）を過小評価 |
| ③ Atomic局所最適 | ✅ 正確 | ただしHBコメントの精緻さは認識すべき |
| ④ 二重化 | ⚠ 過剰主張 | 意図された2層設計。リスクはあるがnotfinished7の表現ほどではない |
| ⑤ RT境界跨ぎ | ✅ 正確 | tryReclaimResourcesのRT呼び出しの指摘追加価値あり |
| ⑥ Thread ID依存 | ✅ 正確（影響限定） | EpochDomainReaderGuardが実質的保護であり、過剰な警告の側面 |
| ⑦ Fade密結合 | ✅ 正確 | 直列実行の問題は的確 |

### 改修漏れの重要な発見（notfinished7.mdにないもの）

1. **`SnapshotRetireManager` 未統合（最重要）**
   - クラスは抽出済みだが、全ソースコードで参照ゼロ
   - `SnapshotCoordinator` は未だに `EpochDomain::enqueueRetire()` (deprecated) を直接使用
   - migrate 先（`ISRRuntimePublicationCoordinator::enqueueRetire()`）は authorized caller だが、SnapshotCoordinator はこれを使っていない

2. **Reclaim が Audio Thread Timer から呼ばれる**
   - `AudioEngine::tryReclaimResources()` (AudioEngine.Threading.cpp:75) が `m_epochDomain.reclaimRetired()` を呼ぶ
   - これは Timer コールバック（Audio Thread 上で動作）から呼ばれうる
   - `DeferredDeletionQueue::reclaim()` は lock-free だが、delete（デストラクタ）を実行するため、Audio Thread 上での破棄処理になりうる

3. **`RuntimePublicationOrchestrator` と `SnapshotCoordinator` の二重 publish 経路**
   - `RuntimePublicationOrchestrator` は publish world 用の orchestrator
   - `SnapshotCoordinator` は snapshot publish 用の coordinator
   - 両者のライフサイクル関係がnotfinished7.mdで触れられていない

4. **`ISRDSPHandle` に別の retire/reclaim 経路が存在**
   - `DSPHandleRuntime::retire()` / `reclaim()` / `quarantine()` が独立した経路として存在
   - これらは `ISRRetireRuntimeEx` とは異なるレイヤー

5. **`crossfadeRuntime_` にも別個の fade state が存在**
   - `AudioEngine::crossfadeRuntime_` (convo::isr::CrossfadeRuntime) が独立した fade 管理を持つ
   - SnapshotCoordinator の fade とは別次元の fade
