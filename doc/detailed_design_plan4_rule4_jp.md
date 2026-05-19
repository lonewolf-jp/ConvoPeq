# ConvoPeq 詳細設計書（plan4.md準拠 / rule4.md厳守）

## 0. 目的

本書は `doc/plan4.md` の改訂ロードマップを、`doc/rule4.md`（AI専用実装統制規約 v1.0）に厳密整合する実装可能粒度へ分解した詳細設計である。
最優先はコンパイル成功ではなく、Immutable Snapshot Runtime（ISR）の不変条件維持である。

---

## 1. 最上位設計原則（rule4準拠）

### 1.1 単一 publish unit

- Runtime 可視状態の唯一同期点は `RuntimeStore::current`（`std::atomic<RuntimeState*>`）のみ。
- `RuntimeState` 外に runtime coherence を形成する mutable state を置かない。
- side-channel atomic（`atomic<bool>`, `atomic<int>`, split param publication）を新設しない。

### 1.2 RuntimeState の完全不変

- publish 後の `RuntimeState` および内包 sub-snapshot は変更禁止。
- 更新は必ず clone-and-publish（新インスタンス構築 → publish → retire）で行う。

### 1.3 RT責務の限定

- Audio Thread は「snapshot observe + DSP execute + fade進行 + RT-local更新」のみ。
- RTでの ownership mutation（shared_ptr操作/heap alloc/free/lock/IO/logger/例外）禁止。
- RTが publish/retire しない（consume only）。

### 1.4 retire/reclaim 権限の一元化

- retire は `PublicationCoordinator -> RetireManager -> EpochDomain` 経路のみ許可。
- direct delete/free/destroy path を禁止。

---

## 2. 適用範囲

### 2.1 設計対象コンポーネント

- `core/RuntimeStore.*`
- `core/PublicationCoordinator.*`
- `core/TransitionPlanner.*`
- `core/FadeEngine.*`
- `core/RetireManager.*`
- `core/EpochDomain.*`
- `audioengine/AudioEngine.*`（publish呼び出し側）
- `eqprocessor/*`（runtime side-channel除去対象）

### 2.2 非対象

- `JUCE/` 配下
- `r8brain-free-src/` 配下
- 本設計書ではUI仕様変更・機能追加は扱わない（ISR整合性のみ対象）

---

## 3. アーキテクチャ完成像（plan4の最終形）

### 3.1 構成要素

1. **RuntimeStore**
   - `atomic<RuntimeState*> current` のみを保持
   - 公開APIは `observe()` のみ（acquire固定）
   - `publish()` は `PublicationCoordinator` のみ実行可能（friendで閉鎖）

2. **PublicationCoordinator**
   - commit linearization（単一飛行保証）
   - `PublicationLog` 消費
   - transition supersession 管理
   - retire要求を `RetireManager` に集約

3. **TransitionPlanner**
   - immutable `Transition` 生成
   - duration正規化、fade legality検証、overlap拒否、policy適用
   - 進行状態は持たない

4. **FadeEngine**
   - 補間計算のみ（純関数）
   - state/ownershipを持たない

5. **RetireManager**
   - runtime policyに基づく retire batching / drain sequencing
   - retire対象種別（RuntimeState / PublicationLog node / その他runtime object）の分類
   - `EpochDomain` 以外の解放経路を排除

6. **EpochDomain**
   - reader登録/退出、retire node enqueue、安全判定reclaim
   - epoch authority の唯一実体

### 3.2 データモデル

#### RuntimeState（唯一 publish 単位）

- `DSPTopologySnapshot topology`
- `RuntimeParameterSnapshot params`
- `TransitionSnapshot transition`
- `uint64_t runtimeVersion`
- `uint64_t transitionId`

> Phase 1〜3 は **inline embedding（完全所有モデル）** を固定採用する。
> sub-snapshot の部分共有・個別retireは禁止（最適化はPhase 6以降の別設計でのみ検討可）。

const graph 強制方針:

- `RuntimeState` は deep-const graph を原則とし、publish後可変フィールドを持たない。
- `mutable` / `const_cast` / lazy init による publish後書込みを禁止する。
- 可能な限り `const` メンバ構成（または同等の不変性保証）を採用する。

#### RuntimeParameterSnapshot（例）

- `bool eqBypass`
- `double mix`
- `double outputGain`
- `const EQCoeffCache* eqCache`

#### TransitionSnapshot（immutable）

- `uint64_t beginSample`
- `uint32_t durationSamples`
- `int startDelayBlocks`
- `int dryHoldSamples`
- `bool useDryAsOld`

#### ActiveTransition（RT-local）

- `Transition transition`（RT thread local copy）
- `int remainingSamples`（RT-local mutable）
- `bool completionLatched`（0到達時の一回発火制御、RT-local）

### 3.3 RuntimeState 物理寿命モデル（Critical-1対応）

- allocate authority: `PublicationCoordinator`（実体生成は `RuntimeStateFactory`）
- publish authority: `PublicationCoordinator` のみ
- retire authority: `PublicationCoordinator -> RetireManager`
- reclaim authority: `RetireManager -> EpochDomain`
- destroy authority: `RetireManager` が `RuntimeState` ルートのみ破棄（inline sub-snapshotは同時破棄）
- **禁止**: sub-snapshot 単体 retire / 部分再利用 / 外部寿命管理

このモデルにより、`RuntimeState` と sub-snapshot の寿命は常に一致し、UAF経路を閉塞する。

### 3.3.1 RuntimeState 構築完了契約（追加Critical）

- `RuntimeState` は **publish前に fully-constructed** でなければならない。
- publish後の subfield 書き換えを禁止する。
- constructor escape（構築途中オブジェクトの外部可視化）を禁止する。
- partially initialized `RuntimeState` の publish を禁止する。
- `RuntimeStateFactory` は fully-initialized object のみ返却可能とする。

### 3.4 PublicationLog ライフタイムモデル（Critical-2対応）

`PublicationLog` は append-only だが、無限成長を許可しない。以下を必須とする。

- `head`（append位置）
- `consumedTail`（PublicationCoordinator が消費完了した末尾）
- `retiredHead`（retire済み先頭）

GC方針:

1. PublicationCoordinator が intent を消費し `consumedTail` を進める。
2. 消費済みintent node を RetireManager に retire 委譲する。
3. EpochDomain が安全時点で reclaim する。

**唯一の log reclaim authority は PublicationCoordinator** とし、他経路のtruncate/destroyを禁止する。

### 3.4.1 PublicationLog 線形化仕様（Critical-6対応）

`PublicationLog` は **MPSC linearizable append queue / single-consumer** として定義する。

- producer: commit発行側スレッド（複数可、非RTのみ）
- consumer: `PublicationCoordinator` のみ（単一）
- queue順序: append成功順（線形化順）を `consume` 順序とする（FIFO）

線形化点:

1. append linearization point:
   - `tail->next` への CAS 成功時点（失敗時は再試行）
2. consume linearization point:
   - consumer が `head->next` を acquire load で観測し、`consumedTail` を進めた時点

順序保証:

- commit A の append CAS 成功が commit B より先なら、consumer は必ず A を先に消費する。
- consumer は `next` ポインタの単方向走査順以外で intent を再順序化しない。
- `PublicationCoordinator` 以外による consume / truncate / reclaim は禁止。

### 3.4.2 PublicationLog MPSC lock-free アルゴリズム契約（Critical-A対応）

`PublicationLog` は Michael-Scott 系 MPSC queue の制約を満たす実装に固定する。

append（producer）擬似手順:

```cpp
for (;;) {
   Node* tail = tailPtr.load(std::memory_order_acquire);
   Node* next = tail->next.load(std::memory_order_acquire);

   if (next == nullptr) {
      if (tail->next.compare_exchange_weak(
            next,
            newNode,
            std::memory_order_release,
            std::memory_order_acquire)) {
         tailPtr.compare_exchange_strong(
            tail,
            newNode,
            std::memory_order_release,
            std::memory_order_relaxed);
         break;
      }
   } else {
      tailPtr.compare_exchange_strong(
         tail,
         next,
         std::memory_order_release,
         std::memory_order_relaxed);
   }
}
```

必須ルール:

- linearization point は `tail->next CAS success` に固定。
- helping rule: `next != nullptr` を観測した producer は `tailPtr` 補正を試みる。
- stale tail correction を省略してはならない。
- CAS失敗時は lock-free retry とし、失敗ノードを書き換えない。
- `newNode->next` は append 前に `nullptr` 初期化し、再利用ノードを禁止。
- detach/reclaim は consumer 側 retire 以外で実行禁止。
- node ABA 回避は「単一割当・単一retire・epoch安全回収・再利用抑止」で担保する。

Log node publish visibility:

- producer は link CAS 前に payload と node全体初期化を完了する。
- `tail->next` の release CAS を payload visibility の publish point とする。
- consumer の acquire load は payload 完全初期化を観測可能でなければならない。

### 3.5 RuntimeStore 権限制御（Critical-4対応）

- `RuntimeStore` は observe専用窓口とする。
- publish入口は `PublicationCoordinator` に閉じる（friend または private publisher token）。
- `AudioEngine` を含む他コンポーネントから `RuntimeStore::publish` を直接呼ぶ設計は禁止。

### 3.6 RetireManager / EpochDomain 責務境界（Critical-5対応）

#### RetireManager（高レベル policy）

- retire対象分類（RuntimeState / Transition / PublicationLog node / DSP関連）
- retire batching
- shutdown時のdrain順序制御
- 非同期reclaim進行監視
- backpressure policy（`maxRetiredBytes`, `maxRetiredNodes`, `forcedReclaimCadence`）

#### EpochDomain（低レベル mechanism）

- reader enter/leave
- retire node enqueue
- safe epoch判定
- reclaim可能nodeの回収

重複責務を禁止し、RetireManager は policy、EpochDomain は mechanism のみに限定する。

### 3.6.1 Epoch reader protocol（Critical-7対応）

`EpochDomain` の reader は以下の token 契約で管理する。

```cpp
class EpochReaderToken {
      uint64_t localEpoch;
      uint32_t recursionDepth;
};
```

契約:

- thread registration は明示 (`registerReaderThread`) / 解除 (`unregisterReaderThread`) を必須化。
- `enterEpoch()` で `recursionDepth++`（0→1 遷移時のみ global epoch を `localEpoch` に取得、acquire）。
- `leaveEpoch()` で `recursionDepth--`（1→0 遷移時のみ quiescent 状態を publish、release）。
- nested/reentrant observe を許可する（depthで管理）。
- recursionDepth underflow は致命エラー（debug assert + fail-fast）。
- recursionDepth 上限は `kMaxEpochRecursionDepth` を持ち、超過時は fail-fast。
- stalled reader 検出のため、`lastProgressTimestamp` を監視対象に含める。

Audio callback の固定シーケンス:

1. callback entry: `enterEpoch()`
2. `observe()` + DSP処理
3. callback exit: `leaveEpoch()`

この entry/exit 以外での reader 保持は禁止する。

ReaderSlot lifetime contract:

- `ReaderSlot` は `EpochDomain` 所有とする。
- unregister 後の immediate free を禁止する。
- slot reclaim も epoch-safe retire 経路を必須とする。
- reader table scan 中に slot address を再利用してはならない。

### 3.6.2 global epoch advancement / reclaim 条件（Critical-B対応）

global epoch は以下のいずれかで advance 試行する。

1. retire enqueue 時（通常経路）
2. `forcedReclaimCadence` の周期tick
3. retired bytes / nodes が閾値超過時

safe reclaim 条件（厳密）:

- retired node の `retiredEpoch = E` に対し、**全active reader の `localEpoch > E`** を満たす場合のみ reclaim 可。
- `>=` 判定は不許可（同一epoch reader残存時の誤回収を防止）。

quiescent publish 対応:

- `leaveEpoch()` の release は reclaim 側の reader table acquire 走査と対応し、
   quiescent 遷移の可視化を保証する。

Permanent stalled reader policy:

- reclaim safety を破る強制reclaimは禁止する。
- stalled reader timeout 超過時は以下を実施する。
   1. 新規commit停止
   2. diagnostic dump
   3. degraded mode移行
   4. shutdown誘導
- memory safety より reclaim progress を優先してはならない。

### 3.7 識別子セマンティクス（High-2対応）

- `runtimeVersion`: publish成功ごとに単調増加（global publish sequence）
- `transitionId`: transition publishごとに単調増加（transition sequence）
- 一意性領域: 単一AudioEngine instance 内での64bit単調列
- wraparound policy: 64bit自然周回は実運用上到達不能とみなし、デバッグ時に残量監視アサートを置く

ID generation authority:

- `runtimeVersion` 採番権限は `PublicationCoordinator` のみ。
- `transitionId` 採番権限も `PublicationCoordinator` のみ。
- `RuntimeStateFactory` は ID生成を行わない。
- speculative build 中の provisional ID を禁止する。
- monotonic wrap は実装不変条件として禁止（運用中の周回を許容しない）。

### 3.8 RT-local 定義（High-3対応）

本書での RT-local は以下に限定する。

- **Audio Thread single-owner state**
- callback間で保持されても、他スレッド共有・公開をしない状態

thread_local ストレージ一般や Message Thread 可視な共有状態は RT-local に含めない。

### 3.9 RuntimeState pointer の ABA耐性（Critical-8対応）

`RuntimeStore::current` は `atomic<RuntimeState*>` を維持するが、以下の **reuse suppression policy** を必須化する。

- retire された `RuntimeState` のメモリは即時再利用しない。
- `EpochDomain` reclaim 後も、`RetireManager` の quarantine ring に一時保持してから allocator 返却する。
- quarantine期間は `minReuseEpochGap >= 2` を必須とし、同一アドレスの短周期再利用を禁止する。
- `RuntimeStateFactory` は quarantine を経由しない再利用経路を持ってはならない。

`RuntimeStateFactory` allocator 契約:

- thread-safe allocator を必須化（複数producer同時割当を許可）。
- slab/pool 再利用は quarantine 要件を満たす場合のみ許可。
- retired burst 耐性として `maxRetiredBytes` 上限超過時に emergency reclaim cadence を起動。
- allocator pressure telemetry（allocated bytes / retired bytes / quarantine depth）を常時記録。
- shutdown では allocator flush は `drainAll()` 完了後のみ許可。
- allocator は quarantine 検証を経ないメモリを RuntimeState 構築へ直接再投入してはならない。

RuntimeStateFactory failure atomicity:

- `RuntimeState` build failure 時は publish を実行しない。
- 部分構築 state の retire/register を禁止する。
- failure 時は current runtime を保持する。
- failed build object は publish前に局所破棄する。

将来最適化として tagged pointer 方式を採用する場合は、本ポリシーと同等以上の ABA 抑止証明を必須とする。

### 3.10 observe lifetime 契約（Critical-9対応）

`observe()` 単体返却を禁止し、寿命保証を guard と不可分にする。

```cpp
struct ObservedRuntime {
   EpochReaderGuard guard;
   const RuntimeState* ptr;
};
```

型制約:

```cpp
ObservedRuntime(const ObservedRuntime&) = delete;
ObservedRuntime& operator=(const ObservedRuntime&) = delete;
ObservedRuntime(ObservedRuntime&&) noexcept = default;
ObservedRuntime& operator=(ObservedRuntime&&) noexcept = default;
static_assert(!std::is_copy_constructible_v<ObservedRuntime>);
```

契約:

- `ptr` の有効期間は `guard` 生存期間と厳密一致。
- guard なし pointer 保存、他スレッド受け渡し、callback外持ち越しを禁止。
- RT/非RTともに `observe()` は `ObservedRuntime` API経由のみ許可。
- `ObservedRuntime` の thread handoff（move先スレッドでの利用）を禁止。

### 3.11 PublicationCoordinator single-flight 実装方式（Medium-3対応）

- single-flight は `atomic_flag`（またはCASゲート）で実装する。
- 先行取得者のみが `PublicationLog` drain と publish/retire を実行する。
- 非取得者は intent append のみ実行し、drain処理には入らない。
- 実装は非RT専用であり、RTからのsingle-flight参加は禁止。
- fairness は append順 + drainループで担保し、intentの飢餓を許可しない。

starvation防止ルール:

- drain実行者は `maxDrainBatch` ごとに gate 再判定し、長時間占有を禁止。
- gate再取得は FIFO waiters 優先（ticket/CAS sequence）で偏りを抑制。
- 一定時間未処理 intent がある場合は monitor warning を発火。

Drain completion rule:

- drain loop は「drain開始時点で存在したnode」ではなく、queue が空になるまで継続する。
- drain中 append された node も同一drainの消費対象に含める。
- gate release 前に head/tail の空状態を再確認する。

Drain non-reentrancy rule:

- `PublicationCoordinator` の drain 実行は非再入（non-reentrant）とする。
- nested drain invocation を禁止する。
- drain ownership は「queue empty 再確認 + publish/retire 完了」まで保持する。

---

## 4. 因果関係とメモリオーダー（Phase 0成果物）

### 4.1 HBチェーン（必須）

1. `publish(newState)` : `store-release(current, newState)`
2. `observe()` : `load-acquire(current)`
3. `retire(oldState)` : retire queue enqueue（release store）
4. `reclaim()` : epoch参照を acquire load で比較し安全判定

追加HB（PublicationLog）:

1. `appendIntent()` : release store で log へ連結
2. `consumeIntent()` : acquire load で可視化
3. `retireIntentNode()` : release store enqueue
4. `reclaimIntentNode()` : acquire compare で安全回収

追加HB（Epoch advancement）:

1. `retire(node@E)` : retired epoch 記録（release）
2. `reader leave` : quiescent publish（release）
3. `reclaim scan` : reader table load（acquire）
4. `all readerEpoch > E` 成立後のみ reclaim 実行

### 4.2 操作別 memory order 契約

- RuntimeState publish: **release 固定**
- RuntimeState observe: **acquire 固定**
- Reader register/leave: **release store 固定**
- Retire enqueue: **release store 固定**
- Reclaim epoch compare: **acquire load 固定**
- RT-local非共有カウンタ: **relaxed 許可**
- PublicationLog `tail->next` link CAS: **success=release / failure=acquire**
- PublicationLog tail correction CAS: **success=release / failure=relaxed**
- reader table publish (`leaveEpoch`): **release 固定**
- reader table scan（reclaimer）: **acquire 固定**

### 4.3 禁止事項

- `seq_cst` の便宜的導入
- publication path への relaxed 適用
- atomic直接dot-call乱立（helper API未経由）

---

## 5. Ownership設計

### 5.1 所有権ルール

- `RuntimeState` は sub-snapshot を **物理所有**（inline embedding）する。
- snapshotは publish 後immutable。
- retire までは `EpochDomain` が寿命保証する。

### 5.2 解放ルール

- direct `delete/free/destroy` 禁止。
- 解放は必ず retire 経由。
- shutdown時のみ `EpochDomain::drainAll()` 後に同期解放を許可。

Retire ordering semantics:

- retire queue は enqueue順 FIFO reclaim を基本とする。
- dependent object を含む retire batch は publish順を保持する。
- `RuntimeState` reclaim は内包snapshotより先に分離destroyされない。

### 5.3 superseded transition

- 中断ではなく supersede（新遷移 publish）で処理。
- 旧 transition は `RuntimeState` と同様に retire。

### 5.4 ActiveTransition 寿命保証（Critical-3対応）

- `ActiveTransition` は pointer参照を保持しない。
- 遷移開始時に `Transition` 値を RT-local へコピーし、そのコピーのみを進行管理する。
- supersede / retire / reclaim は RT-local copy に影響しない。

transition adoption protocol（High-A対応）:

- Audio callback entry で `runtime.transitionId` と `active.transitionId` を比較。
- 不一致時は `ActiveTransition` を `runtime.transition` で置換し、
   `remainingSamples` と `completionLatched=false` を初期化。
- 一致時のみ継続進行し、旧transitionの完了イベント再発火を禁止。

Superseded completion semantics:

- superseded transition は completion event を発火しない。
- transition completion は次を同時に満たす場合のみ成立。
   1. active transition が `remainingSamples == 0` に到達
   2. `runtime.transitionId == active.transitionId`

### 5.5 superseded transition の publish/retire 順序（Critical-10対応）

supersede時の順序を以下で固定する。

1. `new RuntimeState(new Transition)` を構築。
2. `publish(current = newState)` を release で実行。
3. reader が acquire observe で `newState` 可視化可能であることを成立させる。
4. 旧 `RuntimeState`（旧transition内包）を retire queue へ enqueue（release）。
5. `EpochDomain` が safe 判定後に reclaim。

禁止事項:

- publish 前に旧transitionを retire すること。
- 旧transition単体を direct retire/reclaim すること。

---

## 6. Phase別詳細設計（plan4準拠）

### 6.1 Phase 0: 因果仕様の凍結

#### 6.1 目的

- 実装前に HB と memory_order を固定化し、後続フェーズの評価基準を明文化する。

#### 6.1 設計タスク

- `doc/runtime_causality.md` を新規作成。
- publish/observe/retire/reclaim の順序図を定義。
- operation別 memory_order テーブルを固定。
- PublicationLog MPSC append/helping/tail-correction の擬似コードを固定。
- global epoch advancement 条件と safe reclaim 不等号（`>`）を固定。

#### 6.1 完了条件

- 以降の実装レビューで「仕様未定義」を理由にしたatomic変更が発生しない。

### 6.2 Phase 1: Epoch一元化

#### 6.2 目的

- epoch authority を `EpochDomain` 単一へ統合。

#### 6.2 設計タスク

1. `EpochDomain` API定義（reader guard, retire, reclaim, drainAll）。
2. `EpochManager`, `EpochCore`, `g_deletionQueue` を廃止方針に定義。
3. `RefCountedDeferred::release(EpochDomain&)` 注入設計。
4. `SafeStateSwapper` を `EpochDomain&` 依存に置換。
5. `drainAll()` 実行条件を「all epoch participants quiesced」に明文化。
6. `recursionDepth` 上限・overflow動作を定義。
7. shutdown timeout時のホスト通知方針（standalone/plugin別）を定義。

#### 6.2 完了条件

- dual epoch path = 0
- reclaim比較の無意味化パス = 0

### 6.3 Phase 2: Transition線形化

#### 6.3 目的

- lost wakeup / overlap / abort即時破棄を排除。

#### 6.3 設計タスク

1. `Transition` immutable化。
2. `ActiveTransition`（RT-local remaining）導入。
3. `abortFade` 削除、supersede方式へ統一。
4. completion flag（`atomic<bool>`) 廃止。
5. `PublicationLog`（append-only）導入。
6. `PublicationCoordinator` が単一飛行でlog消費。
7. `remainingSamples` は飽和減算（0未満にしない）とし、`completionLatched` で完了一回発火を保証。

#### 6.3 完了条件

- overlapping transition = 0
- completion side-channel flag = 0
- abort direct destroy path = 0

### 6.4 Phase 3: コントロールプレーンのsnapshot包含

#### 6.4 目的

- mutable side-channel / split publication を排除。

#### 6.4 設計タスク

1. `RuntimeState` 階層化（topology/params/transition）。
2. EQ bypass を `RuntimeParameterSnapshot` に移設。
3. IR fade関連 atomics を `TransitionSnapshot` に移設。
4. latency/mix関連の分割公開経路を削除。

#### 6.4 完了条件

- Runtime parameter side-channel = 0
- split publication = 0

### 6.5 Phase 4: キャッシュ寿命のsnapshot従属化

#### 6.5 目的

- RTでの `shared_ptr` 参照カウント経路を排除し、UAFリスクを抑止。

#### 6.5 設計タスク

1. `EQCacheManager::get()` 廃止、`getOrCreate()` のみ。
2. `EQCoeffCache` を snapshot所有へ移行。
3. RT は snapshot経由生ポインタ参照のみ（epoch寿命保証）。

#### 6.5 完了条件

- RT shared_ptr operation = 0
- cache lifetime outside snapshot = 0

### 6.6 Phase 5: SnapshotCoordinator解体

#### 6.6 目的

- God object解体と責務境界明確化。

#### 6.6 設計タスク

1. RuntimeStore / PublicationCoordinator / TransitionPlanner / FadeEngine / RetireManager へ責務分割。
2. publish authority を PublicationCoordinator に限定。
3. retire authority を RetireManager 経路に限定。

#### 6.6 完了条件

- SnapshotCoordinator が単一責務違反を持たない（最終的に削除可能）。
- RuntimeStore直接publishの外部呼び出し = 0

### 6.7 Phase 6: 正当性検証後の最適化（High-5対応）

#### 6.7 目的

- inline embedding の正当性を保持したまま、巨大snapshotコピーの性能劣化リスクを管理する。

#### 6.7 設計タスク

1. ISR correctness（Phase 0〜5）完了後のみ最適化検討を開始。
2. immutable structural sharing 候補（topology/IR graph）を抽出。
3. sharing導入時も publish unit 一貫性と retire一元化を維持する証明を提出。
4. 部分共有導入の可否をベンチマーク（rebuild latency / peak retired bytes）で判定。
5. Phase 1〜5 correctness mode 中は `maxSnapshotBytes` / copy時間テレメトリを必須収集。

#### 6.7 完了条件

- correctness退行 = 0
- publish coherence violation = 0
- 性能改善が定量確認できること

---

## 7. スレッド責務契約

### 7.1 Message Thread

- commit intent生成
- immutable snapshot構築
- PublicationCoordinator呼び出し
- 非RT通知

### 7.2 Audio Thread

- RuntimeStore observe（acquire）
- immutable graph traversal
- `ActiveTransition.remainingSamples` 更新（RT-local）
- 更新規約: `remainingSamples = max(0, remainingSamples - numSamples)`
- `completionLatched == false && remainingSamples == 0` の瞬間のみ完了通知を発火
- 非RTへ完了イベントをポーリング可能な形で引き渡し

Single observe per callback:

- Audio callback は entry で observe を1回のみ実行する。
- callback 中の再observeを禁止する。
- Audio callback scope での nested observe を禁止する。
- callback期間中は単一 `RuntimeState` を使用する。

Completion event publication path:

- RT completion event は **SPSC event queue** のみへ書き込む。
- completion side-channel atomic flag の新設を禁止する。
- completion通知の可視化経路は SPSC queue 消費側に一本化する。

### 7.3 Worker Thread

- 非RT計算のみ
- publish/retire不可

### 7.4 shutdown quiescence 契約（High-4対応）

`EpochDomain::drainAll()` の前提は「Audio Thread停止」のみでは不十分。
以下の **全epoch参加者のquiesced** を必須条件とする。

- Audio Thread
- rebuild/commit実行スレッド
- worker thread（例: learner / analyzer）
- epoch readerとして登録される補助スレッド

quiesced未達でのdrain開始を禁止する。

### 7.5 shutdown 停止保証プロトコル（High-6対応）

`drainAll()` 実行前に、以下の順序を必須化する。

1. 新規 reader 受付停止（registration gate close）
2. worker/rebuild/補助スレッドへ停止要求
3. join with timeout（ハング検知）
4. active reader 数 = 0 をアサート
5. `EpochDomain::drainAll()` 実行
6. drain完了後に最終解放フェーズへ遷移

timeout超過時は fail-safe として「drain延期 + エラー終了（強制解放禁止）」を採用する。

実行形態別 fail-safe:

- standalone: エラーログ出力後に安全停止（プロセス継続可）。
- plugin: unload拒否は行わず、ホストへエラー通知を返し以降の新規commitを停止。
- いずれも強制free/destroyは禁止し、診断ダンプを優先。

### 7.6 forced reclaim cadence 実行隔離（High-B対応）

- reclaim cadence は非RT専用管理スレッド（またはMessage Thread timer）でのみ実行。
- Audio Thread から cadence trigger / reclaim scan / free を呼ばない。
- cadence tick は lock-free queue 長と retired bytes を参照し、RT同期を発生させない。

---

## 8. API設計ガイド（実装拘束）

### 8.1 atomic helper（例）

- `publishRuntimeState(RuntimeState*)`
- `observeRuntimeState() -> ObservedRuntime`
- `enqueueRetire(RuntimeState*)`
- `enqueueTransitionCompletionEvent(Event)`（RT->非RT SPSC）

### 8.2 設計拘束

- helper内で memory_order を固定し、呼び出し側に裁量を持たせない。
- publish helper は非RTアサートを持つ。
- retire helper は PublicationCoordinator 経路のみ利用可。
- observe helper は `EpochReaderGuard` を内部確保し、裸ポインタ返却を禁止する。
- `ObservedRuntime` は move-only 強制（copy禁止）をAPI境界で静的検証する。
- `observeRuntimeState()` 返却値のスレッド越境 move を lint/レビューで禁止する。
- completion通知は SPSC queue 経路のみ許可し、atomic flag 経路を禁止する。

---

## 9. 検証設計

### 9.1 静的検証

- `src` 配下で direct atomic dot-call をゼロ運用（helper経由へ統一）。
- CI lintで `.store(` / `.load(` / `.exchange(` / `.compare_exchange` を検出し、helper外利用を失敗扱いにする。
- RT経路に以下がないことを検証:

  - lock/wait/condition_variable
  - malloc/free/new/delete
  - shared_ptr copy/reset/lock
  - logger/printf/file IO
  - completion side-channel flags
  - `std::function` の暗黙heap capture
  - JUCE `MessageManagerLock`
  - exception throw

### 9.1.1 RT禁止APIブラックリスト（Medium-2対応）

- `std::mutex`, `std::lock_guard`, `std::condition_variable`
- `malloc/free`, `operator new/delete`
- `std::shared_ptr`, `std::weak_ptr`
- `std::function`（heap captureを伴う利用）
- `juce::MessageManagerLock`
- `Logger`, `printf`, filesystem I/O
- exception throw / catch に依存する制御

### 9.2 振る舞い検証

- 連続 commit + crossfade 競合シナリオで overlapが起きない。
- supersede連打時に lost wakeup が起きない。
- shutdown時に `drainAll()` 後の未回収オブジェクトが残らない。
- PublicationLog で append順と consume順の不一致が発生しない（FIFO線形化）。
- reader stalled 注入時に backpressure / forced reclaim cadence が発火し、無限成長しない。
- high contention producer 条件で single-flight 偏りが許容閾値内であること。
- transitionId 切替境界で completion の重複発火/取りこぼしがないこと。
- supersede直後の旧transition completion event が発火しないこと。
- drain loop が race append node を取りこぼさないこと（queue empty まで継続）。
- callback中 multiple observe が発生しないこと（single observe invariant）。
- drain 実行中に nested drain が起きないこと（non-reentrant invariant）。
- completion event が SPSC queue 以外へ流出しないこと。

### 9.3 メモリ順序検証

- publish/observe が release/acquire を保持。
- reclaim比較が acquire load を保持。
- RT-local以外の relaxed を禁止。
- supersede時に「publish(new) HB retire(old)」が保持される。
- observe lifetime が guard scope 外へ漏れない。
- epoch safe reclaim 判定が `readerEpoch > retiredEpoch` で実装される。
- leave/reclaim の release-acquire 対応が崩れていない。
- RuntimeState publish 前に full initialization が完了している（partially initialized publish = 0）。
- Log payload 書込みが link CAS release 以前に完了している。

### 9.3.1 ABA / starvation 検証

- `A -> B -> A` アドレス再利用シナリオを疑似注入し、quarantine policy が再利用を抑止すること。
- retire queue 高水位（bytes/nodes）到達時に reclaim cadence が強制実行されること。
- 長時間reader存在時に shutdown protocol が timeout検知し、強制解放へ進まないこと。
- allocator pressure 急騰時に emergency reclaim cadence が作動し、上限超過が継続しないこと。
- recursionDepth 上限超過時に fail-fast が有効で未定義動作に入らないこと。
- ReaderSlot unregister 後に slot address 再利用由来の stale scan が発生しないこと。
- stalled reader timeout 超過時に commit停止/診断/degraded/shutdown誘導が発動すること。

### 9.4 受入判定（rule4 第12章に対応）

- RuntimeStore の atomic pointer 以外の同期点 = 0
- runtime side-channel = 0
- direct destroy path = 0
- dual epoch = 0
- RT ownership mutation = 0
- transition overlap = 0
- split publication = 0
- publish coherence violation = 0
- ObservedRuntime copy = 0
- unsafe epoch reclaim inequality (`>=`) = 0
- partially initialized RuntimeState publish = 0
- callback内 multiple observe = 0
- lost intent during drain reentrancy = 0
- nested drain invocation = 0
- completion side-channel atomic flag = 0

---

## 10. 実装前必須提出物テンプレート（rule4 8-2）

各タスク着手前に以下9点を必須提出する。

1. 変更理由
2. ISR不変条件への影響
3. happens-before 変化
4. ownership 変化
5. retire/reclaim 変化
6. RT-safe 判定
7. publish coherence 判定
8. supersession 安全性
9. shutdown 安全性

---

## 11. 実装後必須提出物テンプレート（rule4 8-3）

各タスク完了時に以下を必須提出する。

- 新HB図
- 新ownership図
- retire flow
- reclaim flow
- RT safety summary
- added atomics一覧
- memory_order justification

---

## 12. フェーズ管理規約

- フェーズ跨ぎの同時実装禁止（例: Phase 1未完でPhase 3に入らない）。
- 各Phase完了報告で以下を列挙する:

  - removed mutable states
  - removed atomics
  - removed ownership paths
  - removed destruction paths

---

## 13. 既知リスクと対策

1. **部分修正による再崩壊**
   - 対策: 変更単位ごとに HB/ownership/retire をセットでレビュー。

2. **RT経路への意図しない書込み混入**
   - 対策: publish helper 非RTアサート + RT lint で検知。

3. **旧経路の残骸（dead flags, legacy atomics）**
   - 対策: 機能移設と同時に旧状態を削除し、二重系を残さない。

4. **移行工程での mutable 旧経路残留**
   - 対策: legacy atomic / cached mutable state / RT shared_ptr / direct publish / compatibility path を各Phase完了時にゼロ確認する。

---

## 14. まとめ

本設計は `plan4.md` のロードマップを、`rule4.md` の厳格不変条件（単一publish、immutable state、単一epoch、retire一元化、RT責務限定）へ直接マッピングした。
実装は本書のフェーズ順で進め、各変更で HB / ownership / retire / RT-safe を同時に満たすことを合格条件とする。

加えて本改訂で、未閉塞だった以下を明示的に閉塞した。

1. RuntimeState/sub-snapshot の物理寿命（inline完全所有）
2. ActiveTransition の寿命保証（RT-local copy）
3. PublicationLog reclaimモデル（consumedTail/retire/reclaim）
4. RuntimeStore publish authority の閉鎖
5. RetireManager/EpochDomain 境界と shutdown quiescence 契約

---

## 15. 現行乖離監査セクション（2026-05-17）

本章は、上記「完成像」と現行ソース実装の差分を監査した結果を凍結する。
判定は **Critical / High / Medium** の3段階とし、各項目に主要証跡（file:line）を付す。

### 15.1 サマリ

| 区分 | 件数 | 意味 |
| --- | --- | --- |
| Critical | 8 | ISR不変条件（単一publish/単一epoch/retire一元化等）へ直接抵触 |
| High | 7 | フェーズ進行を阻害し、最終形への遷移経路が不安定 |
| Medium | 5 | 契約未定義・未導入・監査粒度不足 |

### 15.2 Critical 乖離

| ID | 乖離内容 | 主要証跡 | 設計参照 |
| --- | --- | --- | --- |
| C-01 | 単一publish unit未達（複数同期面が併存） | `src/audioengine/AudioEngine.h:1010,1012,1084` / `src/core/SnapshotCoordinator.h:121,122,127` | 1.1, 3.1, 9.4 |
| C-02 | 目標コンポーネント群未実装（RuntimeStore等） | `src/**` に `RuntimeStore*`, `PublicationCoordinator*`, `TransitionPlanner*`, `FadeEngine*`, `RetireManager*`, `EpochDomain*`, `PublicationLog*`, `RuntimeState*` なし | 2.1, 3.1 |
| C-03 | Epoch一元化未達（EpochManager/EpochCore/SafeStateSwapper併存） | `src/core/EpochManager.h`, `src/core/EpochCore.h`, `src/SafeStateSwapper.h`, `src/audioengine/AudioEngine.h:2476` | 3.6, 6.2 |
| C-04 | completion side-channel atomic残存 | `src/core/SnapshotCoordinator.h:127`, `src/core/SnapshotCoordinator.cpp:74,79` | 6.3, 7.2, 9.4 |
| C-05 | `abortFade` + direct destroy経路残存 | `src/core/SnapshotCoordinator.cpp:90,94` / `src/core/SnapshotCoordinator.h:49,52` | 5.2, 6.3 |
| C-06 | PublicationLog線形化未導入（queue+mutex経路） | `src/audioengine/AudioEngine.h:1361`, `src/audioengine/AudioEngine.Commit.cpp:43,58,218` | 3.4, 3.11, 6.3 |
| C-07 | observe lifetime契約（ObservedRuntime）未導入 | `src/core/SnapshotCoordinator.h:55`, `src/audioengine/AudioEngine.h:1534`, `src/eqprocessor/EQProcessor.h:338` | 3.10, 8.2 |
| C-08 | shutdownでepoch無視回収が残存 | `src/audioengine/AudioEngine.CtorDtor.cpp:124` (`reclaimAllIgnoringEpoch`) | 3.6.2, 7.5 |

### 15.3 High 乖離

| ID | 乖離内容 | 主要証跡 | 設計参照 |
| --- | --- | --- | --- |
| H-01 | `seq_cst` 禁止方針と実装不一致 | `memory_order_seq_cst` 使用24件（`src/audioengine/AtomicAccess.h`, `src/audioengine/AudioEngine.h`, `src/LockFreeRingBuffer.h`, `src/NoiseShaperLearner.cpp`） | 4.3 |
| H-02 | single observe per callback 不一致（観測面が複数） | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:55,90`, `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:35,52` | 7.2 |
| H-03 | retire/reclaim authority 集約未達 | `src/audioengine/AudioEngine.h:2388,2391,2392`, `src/audioengine/AudioEngine.Threading.cpp:41,108`, `src/eqprocessor/EQProcessor.Core.cpp:25,32` | 1.4, 3.6 |
| H-04 | SnapshotCoordinator解体未着手 | `src/audioengine/AudioEngine.h:2483`, `src/core/WorkerThread.cpp`, `src/SpectrumAnalyzerComponent.cpp:277` | 6.6 |
| H-05 | EQ cache寿命のsnapshot従属化未達 | `src/eqprocessor/EQProcessor.h:113`, `src/RefCountedDeferred.h` | 6.5 |
| H-06 | RuntimeState inline完全所有モデル未導入 | `src/audioengine/RuntimeTransition.h`, `src/audioengine/RuntimeGraph.h` | 3.2, 3.3 |
| H-07 | completion通知のSPSC一本化未達 | `src/core/SnapshotCoordinator.cpp:74,79`, `src/audioengine/AudioEngine.Timer.cpp:368` | 7.2, 8.2 |

### 15.4 Medium 乖離

| ID | 乖離内容 | 状態 |
| --- | --- | --- |
| M-01 | PublicationLog single-flight関連の検証項目が実装前提 | 検証不能（実装待ち） |
| M-02 | `EpochReaderToken` 契約（depth上限/underflow/fail-fast） | `EpochDomain` 未実装 |
| M-03 | ABA quarantine policy | 実装形跡なし |
| M-04 | shutdown quiescence の全参加者ゼロ確認 | 形式化不足 |
| M-05 | helper外atomicゼロ運用の段階移行計画 | 未定義 |

### 15.5 判定

- 現行コードは **Phase 0（因果仕様化）先行状態**。
- **Phase 1〜5 は未完了** であり、本書3章の完成像とは構造的に乖離している。
- よって、現時点の評価は「**設計妥当 / 実装未収束（移行中）**」とする。

### 15.10 Stage B 実装反映（2026-05-17 更新）

- `C-03` は **是正済み**。
  - 根拠: `EpochManager::instance()` / `EpochCoreReaderGuard` / `g_deletionQueue` の参照を `src/**` でゼロ化し、`EpochDomain` へ統合。
  - 主な変更: `src/core/EpochDomain.h` 導入、`AudioEngine`/`EQProcessor`/`ConvolverProcessor`/`RCUReader`/`SnapshotCoordinator` の epoch 経路統一。
- `C-08` は **是正済み**。
  - 根拠: shutdown 経路で `reclaimAllIgnoringEpoch` 呼び出しを撤去し、`EpochDomain::drainAll()` 経路へ統一。
  - 主な変更: `src/audioengine/AudioEngine.CtorDtor.cpp`, `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`, `src/audioengine/AudioEngine.Threading.cpp`。
- `C-07` は **Stage C-1 として着手済み（部分是正）**。
  - 根拠: `SnapshotCoordinator::getCurrent()` を廃止し、`ObservedSnapshot`（move-only guard + pointer）を導入。
  - 主な変更: `src/core/SnapshotCoordinator.h`, `src/core/EpochDomain.h`, `src/audioengine/AudioEngine.*`, `src/SpectrumAnalyzerComponent.cpp`。
   Stage C-2 進捗: `ObservedSnapshot` に owner thread runtime guard を追加し、`observeCurrent(0)` 直値呼び出しをゼロ化。
   残課題: thread handoff 禁止の静的検証（lint）および RuntimeState 系 observe API への同契約拡張。
- 検証結果: `Strict Atomic Dot-Call Scan` pass。grep は `EpochManager::instance(` / `EpochCoreReaderGuard` / `g_deletionQueue` / `reclaimAllIgnoringEpoch` / `getCurrent(` / `observeCurrent(0)` がすべて 0。`Debug Build (cmd env retry)` 成功。
- 注記: `src/core/EpochCore.h` は互換 shim（`#include "EpochDomain.h"`）として残置し、実体実装は除去。Stage D-1 として `executeCommit()` に single-flight（non-reentrant）drain gate を導入し、重複 drain 経路を抑止。

### 15.6 是正優先順位（実装順）

1. Phase 1: `EpochDomain` 導入と dual epoch解消
2. Phase 2: `abortFade` / `m_fadeCompleted` / `deferredCommitQueue` の撤去方針確定
3. Phase 3: Runtime parameter side-channel の snapshot 包含
4. Phase 4: EQ cache寿命を snapshot 従属化
5. Phase 5: SnapshotCoordinator解体（責務分割完了）

### 15.7 運用注記

- 本章は **監査時点のスナップショット** である。
- 各Phase完了時に、`15.2`〜`15.4` の該当行を更新し、件数と証跡を差し替えること。
- 受入判定は `9.4` のゼロ条件を最終判定軸とする。

### 15.8 再調査差分（精密化）

2026-05-17 再調査で、以下を追加確認した。

1. C-05 の direct destroy は `abortFade()` に加えて `~SnapshotCoordinator()` の `SnapshotFactory::destroy(...)` 経路にも存在。
2. H-01 は定性的指摘ではなく、`memory_order_seq_cst` 使用が `src/**` で24件であることを確認。
3. `7.3 Worker Thread` は **部分適合**（`WorkerThread` 本体は publish/retire を直接実行せず、主に command dequeue + callback dispatch）。
   - ただしアーキテクチャ全体としては `SnapshotCoordinator` 依存が残るため、Phase 5 要件（責務分割完了）は未達。
4. `3.6.2` の strict inequality（`readerEpoch > retiredEpoch`）相当は、現行 `DeferredDeletionQueue::reclaim()` が
   `EpochCore::isOlder(entry.epoch, minReaderEpoch)` を用いることで **部分適合**。
   - ただし `EpochDomain` 不在のため、設計上要求される authority 一元化は未達。

### 15.9 移行統制の再設計（レビュー反映）

本節は、最新レビューを受けて「個別不具合修正」から「移行統制」へ方針を再定義する。

#### 15.9.1 基本認識

- 現状は「部分的不整合」ではなく、**旧 mutable runtime 系と ISR 最終形の中間状態が長期化**している。
- したがって対処は、局所パッチではなく **Phase 境界を強制する移行戦略**として扱う。
- 優先順位は「新実装追加」より **旧経路削除を先行**する。

#### 15.9.2 Critical の相互依存（同一因果面）

以下は独立問題ではなく、同一因果面の崩壊として扱う。

- `C-01` 単一 publish unit 未達
- `C-03` dual epoch
- `C-06` PublicationLog 未導入
- `C-07` observe lifetime 未導入

運用規則:

- **1件ずつ潰さない**。
- **Phase 完了 + 旧経路削除**を同一マイルストーンで閉じる。

#### 15.9.3 強制移行順（上書き優先）

`15.6` の実装順は本節で上書きする。以後、次の順のみ許可する。

1. Phase 0 固定化（仕様凍結の再確認）
2. Phase 1 Epoch 統合
3. Phase 2 Publication / Transition 線形化
4. 旧 runtime 同期面の強制削除

禁止事項:

- 新設計経路と旧 mutable runtime 経路の長期併存
- 旧 epoch を残したまま RuntimeStore 導入
- `mutex queue` と `PublicationLog` の併存運用

#### 15.9.4 Critical 別の実施ルール

##### C-01 単一 publish unit 未達

順序を固定する。

1. `RuntimeState` 型を導入
2. runtime coherence 対象（bypass/mix/latency/fade/transition/graph/IR/EQ cache）を列挙
3. 既存 state を `RuntimeState` か RT-local へ分類
4. side-channel atomic を削除

実施原則:

- 「移設のみ」は禁止。
- **移設 + 旧削除を同一パッチで実施**。

##### C-02 目標コンポーネント未実装

導入順を固定する。

`EpochDomain -> RuntimeStore -> PublicationLog -> PublicationCoordinator -> TransitionPlanner -> RetireManager`

逆順導入は禁止。

##### C-03 dual epoch

最優先削除対象とする。

実施手順:

1. `EpochDomain` 実装（最適化しない）
2. 旧 API adapter で callsite 移行
3. 全 callsite 置換
4. 旧 epoch 実装削除

##### C-04 completion side-channel

単独修正を禁止。

- `ActiveTransition`
- `transitionId`
- `completionLatched`

を同時導入後に `m_fadeCompleted` を削除。

##### C-05 abortFade 残存

`abort -> supersede publish -> old retire` に全面移行し、`abortFade()` は deprecated ではなく削除する。

##### C-06 PublicationLog 未導入

移行順を固定する。

1. `PublicationIntent` 定義
2. MPSC append-only log 導入（まだ publish しない）
3. single-consumer drain 導入
4. queue+mutex 経路削除
5. single-flight 導入

##### C-07 ObservedRuntime 未導入

observe API を破壊的変更する。

- 旧: raw pointer observe
- 新: `ObservedRuntime` observe

原則:

- 長期 adapter 期間を持たない。
- guard 無し pointer 保存経路を残さない。

##### C-08 shutdown ignoring epoch

`reclaimAllIgnoringEpoch()` を削除し、

`registration close -> reader stop -> join -> drainAll`

の順で停止保証する。

#### 15.9.5 High 群の着手制約

- High は **Critical 解消後**に着手する。
- 特に `H-01`（`seq_cst` 整理）は HB 固定化後とする。
- `H-02` は `C-07` 完了後に機械検出で収束させる。
- `H-03` は RetireManager 導入後に実施する。
- `H-04` は最終段（責務分割完了後）で実施する。
- `H-05` は Phase 4 まで保留する。

#### 15.9.6 Stage 制御（運用必須）

- Stage A: 凍結（新機能/UI変更/DSP最適化禁止、ISR移行専用ブランチ）
- Stage B: Epoch統合（完了条件: `dual epoch = 0`）
- Stage C: observe寿命強制（完了条件: `裸 RuntimeState pointer = 0`）
- Stage D: Publication線形化（完了条件: `transition overlap = 0`）
- Stage E: RuntimeState統合（完了条件: `single publish unit 成立`）
- Stage F: cache lifetime（EQ cache snapshot ownership）

#### 15.9.7 運用上の最重要原則

- ISR は中間状態が最も危険である。
- 必ず

   1. `1 Phase 完了`
   2. `旧経路削除`
   3. `次 Phase 着手`

   の順を厳守する。
- **未完成 ISR + 旧 mutable runtime の長期共存を禁止**する。

### 15.11 15章監査表の最新版ドラフト（C/H/M再採番, 2026-05-19）

本節は、`15.1`〜`15.10` を履歴として保持したまま、**現行ソース実測を優先**して再採番したドラフトである。
正式版ではなく、Phase 進行に伴い随時更新する。

#### 15.11.1 再採番サマリ（Open項目のみ）

| 区分 | 件数 | 意味 |
| --- | --- | --- |
| Critical | 1 | plan4 最終形（単一 publish unit / 最終責務分割）に未到達 |
| High | 0 | 最終形へ到達するための主要な構造・統制ギャップはクローズ済み |
| Medium | 0 | 追加検証・監査更新で収束可能な未確定事項はクローズ済み |

#### 15.11.2 Critical（再採番）

| ID | 乖離内容 | 主要証跡（現行） | 判定理由 |
| --- | --- | --- | --- |
| C-03 | SnapshotCoordinator 解体未完 | `src/audioengine/AudioEngine.h:73,791,2661`, `src/core/SnapshotCoordinator.h` | Phase 5 要件（責務分割完了）に未到達 |

#### 15.11.3 High（再採番）

| ID | 乖離内容 | 主要証跡（現行） | 判定理由 |
| --- | --- | --- | --- |

#### 15.11.4 Medium（再採番）

| ID | 乖離内容 | 現状 | 次の検証 |
| --- | --- | --- | --- |

#### 15.11.5 解消済み（旧IDの明示クローズ）

以下は現行 `src/**` 実測で解消確認済み（再発監視は継続）。

- 旧 `C-03`（dual epoch）: `EpochManager::instance(` / `EpochCoreReaderGuard` / `g_deletionQueue` ヒット 0
- 旧 `C-08`（shutdown ignoring epoch）: `reclaimAllIgnoringEpoch` ヒット 0
- 旧 `C-04`（completion side-channel）: `m_fadeCompleted` ヒット 0
- 旧 `C-05`（abort direct destroy）: `abortFade` ヒット 0
- 旧 `C-01`（単一 publish unit 未達）: `currentDSPBits` / `fadingOutDSPBits` を削除し、NonRT 側保有へ統合
- 旧 `C-02`（`RuntimeState` 未導入）: `src/audioengine/AudioEngine.h` で `struct RuntimeState` を導入（`RuntimePublishWorld` は互換 alias）
- 旧 `H-01`（`seq_cst`）: `memory_order_seq_cst` ヒット 0
- 旧 `H-02`（observe lifetime 契約未収束）: `observeCurrentRuntime` を主契約へ統一し、`observeCurrentSnapshot` / `getSnapshotCoordinator` 公開面を撤去
- 新 `H-01`（PublicationCoordinator core 独立型未確立）: `src/core/RuntimePublicationCoordinator.h` を導入し、`AudioEngine` の nested 実装を `RuntimePublicationBridge` + 外部 coordinator 型へ置換
- 新 `H-03`（RuntimeStore 書込み権限の最終閉鎖未完）: `RuntimeStore<RuntimePublishWorld, convo::RuntimePublicationCoordinator<...>>` へ固定し、`acquireWriteAccess()` を外部 owner 型の static factory 経由に限定
- 旧 `H-04`（監査章と実装進捗の時差）: 15章/10章/task の C/H/M 再採番を同期反映
- 旧 `H-05`（Stage 運用チェック更新遅延）: `doc/task.md` 未チェック項目を再採番ベースで正規化
- 旧 `M-01`（EQ cache snapshot ownership 最終判定）: `AudioEngine.Cache.cpp` の `CacheMap` RAII + `enqueueDeferredDeleteNonRt` 経路で所有権/寿命一元化を再確認
- 旧 `M-02`（completion通知の SPSC 一本化最終判定）: `SnapshotCoordinator.cpp` の `tryCompleteFade` CAS 一回性 + `m_fadeCompleted` 非依存化を再確認
- 旧 `M-03`（thread handoff 禁止の静的検証）: `check-src-atomic-dotcall.ps1` に ObservedSnapshot handoff 検知ルールを追加し、Strict scan pass
- 旧 `M-04`（PublicationLog 線形化の受入証跡形式）: `doc/task.md` に受入証跡テンプレートを追加

#### 15.11.6 ドラフト運用注記

- 本ドラフトは **2026-05-19 時点**の再採番である。
- 正式版へ反映する際は、`doc/runtime_causality.md` 10章および `doc/task.md` の未チェック項目を同時更新し、文書間の時差を解消する。
