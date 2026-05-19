## ConvoPeq ISRアーキテクチャ違反リスト及び改訂ロードマップ（厳密改訂版）

これまでの議論を総合し、**実装レベルで再崩壊しないISR** を定義する。以下の指摘を反映した。

- `shared_ptr` 中心化の危険（RTスレッドでの参照カウント操作禁止）
- publish coherence の未定義問題
- `pendingCommit` 単一化の不十分性
- `currentSample >= endSample` 判定の危険性
- `PublicationCoordinator` 責務不足
- `superseded transition` のライフタイム未定義
- `RetireManager` 権限の曖昧さ
- **runtime causality と memory_order の仕様化必須**

---

# 0. 前提：ISRの基本不変条件（Phase 0 で文書化必須）

```text
[Publish]   store-release(current, newState)
   HB
[Observe]   load-acquire(current)
   HB
[Retire]    enqueue(oldState, retireEpoch)   (release store)
   HB
[Reclaim]   reclaim(epochDomain)             (acquire compare)
```

さらに、Transition は以下の因果関係を持つ。

```text
[PublishTransition]   store-release(transition)
   HB
[StartFade]           observe transition
   HB
[CompleteFade]        transition.endSample ≤ currentSample
   HB
[RetireTransition]    enqueue(transition)
```

**memory_order ガイドライン**:

| 操作                       | 要求メモリオーダー                 | 理由                                   |
| -------------------------- | ---------------------------------- | -------------------------------------- |
| `RuntimeState` の公開        | `release`                          | publish HB observe                     |
| `RuntimeState` の参照取得    | `acquire`                          | observe after publish                  |
| Reader epoch 登録           | `release` (store epoch)            | 参加表明                               |
| Reader epoch 退出           | `release` (store `kIdleEpoch`)     | 退出表明                               |
| Retire キュー登録           | `release` (store state, epoch)     | enqueue HB reclaim                     |
| Reclaim 時の epoch 比較     | `acquire` (load epoch)             | 比較前に古い epoch が見えることを保証   |
| Audio Thread 内部状態更新   | `relaxed` (同一スレッドのみ)       | 他スレッドと共有しない                 |

この仕様を満たさない実装は ISR ではない。

---

# 🔴 ISR-P0（不変条件の直接破壊）

## ISR-P0-1 Epoch二元性（因果関係の崩壊）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `EpochManager::instance()` | `core/EpochManager.h` | 全行 | グローバルシングルトン |
| `EpochCore` クラス | `core/EpochCore.h` | 全行 | ローカルエポック |
| `m_epochCore` | `audioengine/AudioEngine.h` | 1560 | メンバ変数 |
| `EpochCoreReaderGuard` 使用箇所 | `audioengine/AudioEngine.Processing.*.cpp` | 各所 | ローカルエポックのリーダー |
| `EpochManager::instance()` 使用箇所 | `eqprocessor/*.cpp`, `NoiseShaperLearner.cpp` | 各所 | グローバルエポック |
| `RefCountedDeferred::release()` | `RefCountedDeferred.h` | 20-26 | グローバル epoch で retire |
| `AudioEngine::tryReclaimResources()` | `audioengine/AudioEngine.Threading.cpp` | 30 | `m_epochCore` で reclaim |
| `DeletionQueue::reclaim()` | `core/DeletionQueue.h` | 50-70 | 比較が無意味 |
| `SafeStateSwapper` の内部エポック | `SafeStateSwapper.h` | 80-120 | 第3のエポック |
| `SnapshotCoordinator::m_deletionQueue` | `core/SnapshotCoordinator.h` | 50 | ローカル DeletionQueue |

**改修**:
- `EpochDomain` クラス新設（`core/EpochDomain.h`）。グローバル singleton ではなく、`RuntimeStore` が所有する。
- `EpochManager`, `EpochCore` を削除。全 epoch 操作を `EpochDomain` に移譲。
- `RefCountedDeferred::release(EpochDomain&)` に変更。`EpochDomain` 外部から注入。
- `g_deletionQueue` を廃止し、`EpochDomain` 内の `DeletionQueue` に統一。
- `SafeStateSwapper` も `EpochDomain&` を受け取るコンストラクタを追加。
- `SnapshotCoordinator` は `EpochDomain` を持たず、`RetireManager` に委譲。

---

## ISR-P0-2 スナップショット即時破棄（abortFade）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `SnapshotCoordinator::abortFade()` | `core/SnapshotCoordinator.cpp` | 70-78 | 関数 |
| `SnapshotCoordinator::switchImmediate()` | 同ファイル | 40-50 | 呼び出し元 |
| `SnapshotCoordinator::updateFade()` | 同ファイル | 90-100 | 呼び出し元 |

**改修**:
- `abortFade()` を完全に削除。
- 遷移キャンセルは `superseded transition` として処理。新しい遷移を発行し、古い遷移は `RuntimeState` とともに retire。
- `switchImmediate()` は遅延解放パイプラインを通す（既存の `enqueueDeferredDeleteNonRt` を流用）。

---

## ISR-P0-3 トランザクション可視性の欠如（lost wakeup）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `SnapshotCoordinator::m_fadeCompleted` | `core/SnapshotCoordinator.h` | 90 | メンバ変数 |
| `advanceFade()` / `requestFadeCompletion()` / `tryCompleteFade()` | `core/SnapshotCoordinator.cpp` | 110-155 | 関数群 |

**改修**:
- `m_fadeCompleted` フラグを廃止。
- `Transition` を不変オブジェクトとして定義（`from, to, beginSample, duration`）。
- 完了判定はサンプル位置ではなく **`remainingSamples` を Audio Thread が管理する**。
  ```cpp
  struct ActiveTransition {
      const Transition* immutable;  // epoch-managed
      int remainingSamples;         // RT-local only
  };
  ```
- Audio Thread は `advanceFade(numSamples)` で `remainingSamples` を減算し、ゼロになったら `transitionComplete` フラグを立てる（これも RT-local）。
- Message Thread は `tryCompleteFade()` の代わりに `pollCompletedTransition()` で完了した遷移を取得し、`RuntimeState` の更新処理を行う。

---

## ISR-P0-4 トランザクションの重複許可（crossfade overlap）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `AudioEngine::commitNewDSP()` 内の重複チェック | `audioengine/AudioEngine.Commit.cpp` | 150-170 | ロジック |
| `deferredCommitQueue` | `audioengine/AudioEngine.h` | 800 | キュー |
| `prepareCommit()` / `executeCommit()` | `audioengine/AudioEngine.Commit.cpp` | 20-60 | 関数 |

**改修**:
- `deferredCommitQueue` を廃止。代わりに **append-only publication log** を導入。
  ```cpp
  struct PublicationLog {
      std::atomic<CommitIntent*> head;  // 単方向リスト
  };
  struct CommitIntent {
      CommitId id;
      RuntimeState* state;
      CommitIntent* next;
  };
  ```
- `PublicationCoordinator` が log を消費し、**単一飛行** で遷移を開始する。同時実行される commit は log に積まれるだけ。
- Audio Thread は log を読まず、単に `RuntimeState` のアトミックポインタを参照する。

---

# 🟠 ISR-P1（不変条件の弱体化）

## ISR-P1-1 mutable side channel（EQ bypass）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `EQProcessor::m_rtBypassShadow` | `eqprocessor/EQProcessor.h` | 500 | メンバ変数 |
| `setBypassFromRT()` | 同ファイル | 501 | 未使用関数 |
| `bypassRequested` | 同ファイル | 498 | アトミック変数 |
| `EQProcessor::process()` 内参照 | `eqprocessor/EQProcessor.Processing.cpp` | 270 | 読み取り |

**改修**:
- `bypassRequested` と `m_rtBypassShadow` を削除。
- `GlobalSnapshot`（または `RuntimeParameterSnapshot`）に `eqBypass` フィールドを追加。
- UI でのバイパス変更 → 新 `RuntimeState` を publish。
- Audio Thread はスナップショットから `eqBypass` を読み取る。

---

## ISR-P1-2 関連パラメータの分割公開（IRフェード時間）

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `m_irFadeSamples`, `m_irFadeTimeSec` | `audioengine/AudioEngine.h` | 650-651 | アトミック変数 |
| `setIRFadeSamples()` | 同ファイル | inline | 関数 |
| `prepareToPlay()` 内の更新 | `AudioEngine.Processing.PrepareToPlay.cpp` | 150 | 書き込み |
| `commitNewDSP()`, `armDryAsOldCrossfadeForCurrentDSP()` 内の読み取り | `AudioEngine.Commit.cpp` 他 | 各所 | 読み取り |

**改修**:
- これらのアトミック変数を廃止。フェードパラメータは `TransitionSnapshot` に含める。
- `setIRFadeTimeSec(double sec)` に変更。サンプル数は必要時に計算する。

---

## ISR-P1-3 キャッシュライフタイムの外部所有

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `EQCacheManager::get()` | `audioengine/AudioEngine.Cache.cpp` | 70 | 関数 |
| `processDouble()` 他での使用 | `AudioEngine.Processing.*.cpp` | 各所 | 呼び出し元 |

**改修**:
- **shared_ptr を使用しない**。代わりにキャッシュのライフタイムを `RuntimeState` に完全に依存させる。
  - `RuntimeParameterSnapshot` が `EQCoeffCache*` を持つ（所有権はスナップショットが持つ）。
  - `EQCoeffCache` は `RefCountedDeferred` をやめ、スナップショットとともに epoch 管理される。
  - Audio Thread はスナップショット経由でキャッシュを参照する（生ポインタだが、スナップショットの epoch が生存を保証）。

---

## ISR-P1-4 shutdown pathの逸脱

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `~SnapshotCoordinator()` | `core/SnapshotCoordinator.cpp` | 15-25 | デストラクタ |

**改修**:
- シャットダウン時は Audio Thread 完全停止後に `EpochDomain::drainAll()` を呼び、全リソースを同期的に解放する。
- デストラクタでの即時破棄は許容するが、その前に `drainAll()` が呼ばれていることをアサートする。

---

# 🟡 ISR-P2（設計の複雑性）

## ISR-P2-1 God object（SnapshotCoordinator）

**改修**: 以下の5コンポーネントに分割。

| コンポーネント | 責務 | ファイル |
|--------------|------|----------|
| `RuntimeStore` | `atomic<RuntimeState*> current` の管理 | `core/RuntimeStore.h` |
| `PublicationCoordinator` | 公開の linearization、遷移開始、retire 権限 | `core/PublicationCoordinator.h` |
| `TransitionPlanner` | 不変 `Transition` オブジェクトの生成 | `core/TransitionPlanner.h` |
| `FadeEngine` | 補間計算のみ（状態なし） | `core/FadeEngine.h` |
| `RetireManager` | `EpochDomain` と連携した遅延解放 | `core/RetireManager.h` |

**PublicationCoordinator の必須責務**:
- 単一飛行保証
- 遷移の supersession 処理
- コミット log の消費
- retire 権限の一元化（`RetireManager` を通じて）

---

## ISR-P2-2 非階層的スナップショット（publish coherence 問題）

**問題**: 複数の sub-snapshot を個別に publish すると、関連する値が異なるバージョンで観測される恐れ。

**解決**: `RuntimeState` が **唯一の publish unit** であること。

```cpp
struct RuntimeState {
    DSPTopologySnapshot* topology;      // owned by this state
    RuntimeParameterSnapshot* params;   // owned by this state
    TransitionSnapshot* transition;     // owned by this state
};
```

- 内部のポインタは全て `const` で不変オブジェクトを指す。
- `RuntimeState` 全体を `aligned_malloc` で確保し、`atomic<RuntimeState*>` で publish。
- 古い `RuntimeState` は epoch 管理で解放。

**階層化のメリット**:
- トポロジー変更時のみ `topology` を新しくし、`params` は再利用可能。
- `RuntimeState` の内部ポインタはコピー時に参照カウントを増やさない（生ポインタ、epoch で寿命保証）。

---

## ISR-P2-3 カスタムメモリ管理の不統一

| 対象 | ファイル | 行 | 種別 |
|------|----------|----|------|
| `IppFFTPlanCache` の永続キャッシュ | `MKLNonUniformConvolver.cpp` | 30-50 | 静的マップ |
| `mkl_malloc` / `ippsMalloc` / `new` 混在 | 多数 | 各所 | メモリ確保 |

**改修**:
- 長期キャッシュ（FFTプラン）は `shared_ptr` でも良い（非RTスレッドのみアクセス）。
- ランタイムデータ（`RuntimeState`, `DSPCore`, `ConvolverState`, `Transition`）はすべて `mkl_malloc` + epoch 管理。
- 解放は `RetireManager` に一元化。

---

# 🟢 ISR-P3（軽微・最適化）

## P3-1 精度・パフォーマンス

| 対象 | 改修 |
|------|------|
| `equalPowerSinApprox` | 入力 `std::clamp(x, 0.0f, 1.0f)` 追加 |
| `SpectrumAnalyzerComponent` FFT | `DFTI_REAL` に変更 |
| `LinearRamp::skip` | `ConvolverProcessor::process()` で活用 |
| `applyAllpassToIR` | デッドコードのため削除、または `ASSERT_NON_RT_THREAD()` |

---

# 改訂ロードマップ（ISR完成への段階的実装計画）

## Phase 0: 因果関係仕様化（最優先・必須）

- **成果物**: `docs/runtime_causality.md`
  - happens-before 関係を文書化
  - memory_order ガイドラインを明示
  - 各 operation の要求オーダーを表形式で定義
- **コード変更**: なし（ただし、以降のフェーズはこの仕様に従う）

---

## Phase 1: Epoch一元化（ISR-P0-1, P2-3）

| ステップ | 作業 |
|----------|------|
| 1.1 | `EpochDomain` クラス新設（`core/EpochDomain.h`） |
| 1.2 | `EpochManager`, `EpochCore` を削除、全参照を `EpochDomain` に置換 |
| 1.3 | `AudioEngine::m_epochCore` → `EpochDomain domain_` に変更 |
| 1.4 | `RefCountedDeferred::release(EpochDomain&)` に変更、呼び出し元修正 |
| 1.5 | `g_deletionQueue` 廃止、`EpochDomain` 内の `DeletionQueue` に移行 |
| 1.6 | `SafeStateSwapper` を `EpochDomain&` 受け取りに変更 |
| 1.7 | `SnapshotCoordinator` のローカル `DeletionQueue` を廃止し、`RetireManager` へ移譲 |

---

## Phase 2: トランザクションのLinearizable化（ISR-P0-2, P0-3, P0-4）

| ステップ | 作業 |
|----------|------|
| 2.1 | `Transition` 不変構造体定義（`from, to, beginSample, duration`） |
| 2.2 | `ActiveTransition` 構造体定義（RT-local `remainingSamples`） |
| 2.3 | `abortFade()` 削除、代わりに superseded transition を発行 |
| 2.4 | `m_fadeCompleted` と関連関数削除 |
| 2.5 | `PublicationCoordinator` 新設（commit log 消費、単一飛行、supersede処理） |
| 2.6 | `deferredCommitQueue` 廃止、代わりに `PublicationLog`（append-only）導入 |
| 2.7 | `RuntimeState` に `transition` フィールド追加（`TransitionSnapshot*`） |

---

## Phase 3: コントロールプレーンのスナップショット包含（ISR-P1-1, P1-2, P2-2）

| ステップ | 作業 |
|----------|------|
| 3.1 | `RuntimeState` 階層化（`DSPTopologySnapshot`, `RuntimeParameterSnapshot`, `TransitionSnapshot`） |
| 3.2 | バイパス状態を `RuntimeParameterSnapshot` に移動、`bypassRequested` 削除 |
| 3.3 | IRフェードパラメータを `TransitionSnapshot` に移動、split atomics 削除 |
| 3.4 | `latencyDelayOld` / `latencyDelayNew` アトミック削除、`RuntimeState` から読み取り |
| 3.5 | `mixSmoother` リセットをスナップショット変更時のみに（専用フラグ削除） |

---

## Phase 4: キャッシュライフタイムの明確化（ISR-P1-3）

| ステップ | 作業 |
|----------|------|
| 4.1 | `EQCacheManager::get()` 削除、`getOrCreate()` のみに |
| 4.2 | `EQCoeffCache` を `RefCountedDeferred` から外し、スナップショットで所有 |
| 4.3 | `RuntimeParameterSnapshot` が `EQCoeffCache*` を保持 |
| 4.4 | Audio Thread はスナップショット経由でキャッシュを参照（epoch で寿命保証） |

---

## Phase 5: SnapshotCoordinator解体（ISR-P2-1）

| ステップ | 作業 |
|----------|------|
| 5.1 | `RuntimeStore` 実装（`atomic<RuntimeState*> current`） |
| 5.2 | `PublicationCoordinator` 実装（上記 Phase 2 で一部実装済み） |
| 5.3 | `TransitionPlanner` 実装（`Transition` オブジェクト生成） |
| 5.4 | `FadeEngine` 実装（静的補間関数） |
| 5.5 | `RetireManager` 実装（`EpochDomain` と連携） |
| 5.6 | 元の `SnapshotCoordinator` を削除し、上記コンポーネントに置換 |

---

# 最終的な ISR 完成形の要点

1. **唯一の同期ポイント**: `RuntimeStore::current` の `atomic<RuntimeState*>` のみ。
2. **すべてのランタイムパラメータはスナップショットに含まれる**: バイパス、フェード、レイテンシ、ミックスなど。
3. **Transition は不変オブジェクト**: 進行状態（`remainingSamples`）は RT-local のみ。
4. **Epoch は単一ドメイン**: `EpochDomain` が全てのオブジェクトのライフタイムを管理。
5. **retire 権限は `PublicationCoordinator` に一元化**: 誰でも自由に retire できない。
6. **shared_ptr を RT スレッドで使用しない**: epoch 管理の生ポインタ + スナップショット所有で代用。
7. **happens-before と memory_order がドキュメント化されている**: Phase 0 の成果物。

この改訂版を実装することで、ConvoPeq は現在の「mutable coordination plane」を完全に排除し、**真の Immutable Snapshot Runtime** へと進化する。これにより、これまで指摘された大半のバグ（epoch 二元性、abortFade、lost wakeup、重複遷移、mutable side channel、split atomics、キャッシュUAF）はアーキテクチャレベルで解消される。