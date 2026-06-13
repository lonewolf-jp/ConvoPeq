# A-4 / C-2 設計修正案 v2（レビュー反映版）

**バージョン**: 2.1
**日付**: 2026-06-13
**ベース**: a4_c2_design_proposal.md v2.0 + 最終レビュー指摘3点
**状態**: 設計提案（実装未着手）

---

## 0. 改訂履歴

| 版 | 日付 | 変更内容 |
|:--:|------|----------|
| 1.0 | 2026-06-13 | 初版: `completeCrossfadeTransition()` 全統合案 |
| 2.0 | 2026-06-13 | retire/unregister/complete 共通化取下げ。`publishIdleWorldOnly()` のみ抽出。line64 reset削除取下げ |
| 2.1 | 2026-06-13 | **最終調整**: 引数を `DSPCore* + TransitionPolicy` に変更。`isShutdownInProgress()` ガード追加。デフォルト引数削除 |

---

## 1. 現状分析

### 1.1 3経路の責務比較

| 責務 | Path 1 Timer | Path 2 DSPTransition | Path 3 Timeout |
|------|:---:|:---:|:---:|
| `exchangeFadingDSP→retire` | ✅ inline | ✅ inline | ✅ inline |
| `unregisterCrossfadeId` | ✅ inline | ❌ (設計上不要) | ✅ inline |
| `crossfadeRuntime_.complete()` | ✅ inline | ❌ (publish時は進行中) | ✅ inline |
| `setDryHoldSamples(0) + refreshSnapshot` | ✅ inline | ✅ inline | **❌** |
| Idle world publish | ✅ (SmoothOnly) | ✅ (HardReset) | **❌** |
| `sendChangeMessage()` | ✅ | ❌ | ❌ |

### 1.2 3経路の差異は設計上の意図

| 差異 | 理由 |
|------|------|
| Path 2 が unregister/complete を行わない | DSPTransition の publish 時点では crossfade がまだ進行中であり、unregister も complete もできない。Timer 経路でのみ complete 可能 |
| Path 1 のみ `setStartDelayBlocks(0)` を呼ぶ | DSPTransition は publish 後に fading が開始される。Timer 経路は fade 完了後に呼ばれるため start delay をリセットする意味がある |
| Path 1 のみ `sendChangeMessage()` を呼ぶ | UI 変更通知は Timer の責務。DSPTransition は publish サイクル内で完結 |
| Path 1 → SmoothOnly, Path 2/3 → HardReset | Timer 経路の正常完了は SmoothOnly (既存 fade 継続)。強制完了系 (DSPTransition の硬結合切替・Timeout) は HardReset |

### 1.3 Idle world publish 前の共通準備

Path 1 (Timer) と Path 2 (DSPTransition) は publish 直前に以下を行っている:

```cpp
crossfadeRuntime_.setDryHoldSamples(0);
refreshCrossfadePreparedSnapshotFromAtomics();
```

Path 3 (Timeout Recovery) ではこれらの準備と呼び出しの**両方が欠落**している。
これが A-4 で修正する唯一の実質的欠落である。

### 1.4 デッドコード: `notifyTransitionComplete()`

`RuntimePublicationOrchestrator::notifyTransitionComplete()` は以下4責務を定義するが、
**呼び出し元が存在しない**（grep 確認済み、確度100%）:

```cpp
// src/audioengine/RuntimePublicationOrchestrator.cpp:243  (定義)
// src/audioengine/RuntimePublicationOrchestrator.h:58     (宣言)
// (呼び出し元: なし)
```

4責務:

1. **Transition Completion**: `transition_.onTransitionComplete(currentAfterFade)`
2. **Shutdown Guard**: `isShutdownInProgress()` 時は deferred をキャンセル
3. **Stale Discard**: Generation Guard + Publication Sequence Guard
4. **Deferred Publish Submit**: 有効な deferred を `submitPublishRequest(req)`

### 1.5 C-2 crossfade recovery デッドコード問題

`crossfadeRuntime_.reset()` が `releaseResources()` の line 64 で早急に呼ばれており、
EmergencyDrain フェーズ（line 199）では `isPending() == false` が確定する。

確認済みの経路:

- `crossfadeRuntime_.start()` → DSPTransition.h:103 のみ。publish 経路で `isShutdownInProgress()` ガード
- Timer 経路: 全回復経路を `!isShutdownInProgress()` でガード（L201, L240）
- rebuild thread: L145 で停止

**結論**: EmergencyDrain 内の crossfade recovery は絶対に実行されない（確度100%）。

---

## 2. 設計方針（v2）

### 基本原則

1. **publish のみ共通化**: retire/unregister/complete は各経路の責務として維持
2. **既存の正常系を変更しない**: Path 1 (Timer) と Path 2 (DSPTransition) には手を加えない
3. **最小修正で欠落を補う**: Path 3 (Timeout Recovery) に publish を追加するのみ
4. **notifyTransitionComplete() は存続**: 将来の統合 entry point として責務定義を保持
5. **C-2 line 64 reset は削除しない**: shutdown 決定性を最優先。EmergencyDrain crossfade recovery は dead code のまま維持

### なぜ retire/unregister/complete を共通化しないのか

`completeCrossfadeTransition()` で retire/unregister/complete/publish を一括共通化すると、
DSPTransition 側の意味論を変更してしまう。

| 処理 | Path 1 Timer | Path 2 DSPTransition | 統合すると |
|------|:---:|:---:|:---:|
| unregister | ✅ する | ❌ しない | DSPTransition で不要な unregister が発生 |
| complete | ✅ する | ❌ しない | DSPTransition で crossfade 進行中の complete が発生 |
| setStartDelayBlocks(0) | ✅ する | ❌ しない | DSPTransition で不要な start delay リセットが発生 |

これらの差異は設計上の意図であり、共通化は設計境界を壊す。

---

## 3. A-4 実装案: `publishIdleWorldOnly()` 抽出

### 3.1 抽出範囲（v2.3 最終調整: publish only に限定）

v2.1 の3点修正からさらに、`setDryHoldSamples(0)` と `refreshCrossfadePreparedSnapshotFromAtomics()`
を helper から外す。helper の責務を pure publish に限定する。

理由: `startDelayBlocks_` も snapshot に含まれるが、これは Path 1 固有の管理項目。
helper が publish 前準備を担うと、将来の snapshot 構造変化に対して Timeout Recovery だけ
取り残されるリスクがある。

```cpp
// AudioEngine.h — 宣言
// ★ A-4: Idle world publish 統一関数
//   責務は publishWorld のみ。setDryHoldSamples/resfreshSnapshot は含まない。
//   currentAfterFade: 呼び出し側で解決して渡す
//   idlePolicy: 呼び出し側で明示指定。デフォルト値なし。
[[nodiscard]] bool publishIdleWorldOnly(
    AudioEngine::DSPCore* currentAfterFade,
    convo::TransitionPolicy idlePolicy) noexcept;
```

```cpp
// AudioEngine.Transition.cpp
// Returns: true=publish 実行, false=shutdown guard または nullptr で skip
bool AudioEngine::publishIdleWorldOnly(
    AudioEngine::DSPCore* currentAfterFade,
    convo::TransitionPolicy idlePolicy) noexcept
{
    // Shutdown guard（publishWorld パスに明示的な guard がないため）
    // bool 返却により呼び出し側が publish 成否を認識可能
    if (isShutdownInProgress())
        return false;
    if (currentAfterFade == nullptr)
        return false;

    // ★ Idle world 発行のみ — 前準備は呼び出し側の責務
    auto coordinator = makeRuntimePublicationCoordinator();
    auto worldBuilder = convo::RuntimeBuilder(*this);
    worldBuilder.setHealthStateRef(getHealthStateRef());
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(
        currentAfterFade, nullptr, idlePolicy, 0.0, false);
    coordinator.publishWorld(std::move(worldOwner));
}
```

**この関数が含まないもの**:

- ❌ `exchangeFadingRuntimeDSP → retire` — 各経路の責務
- ❌ `unregisterCrossfade` — 各経路の責務
- ❌ `crossfadeRuntime_.complete()` — 各経路の責務
- ❌ `crossfadeRuntime_.setStartDelayBlocks(0)` — Path 1 固有
- ❌ `crossfadeRuntime_.setDryHoldSamples(0)` — 呼び出し側で実行
- ❌ `refreshCrossfadePreparedSnapshotFromAtomics()` — 呼び出し側で実行
- ❌ `sendChangeMessage()` — 呼び出し元の責務

**`bool` 戻り値の設計意図**:
`false` 返却は「publish が実行されなかった」ことを呼び出し側に通知する。
現状の呼び出し側（Path 3）は戻り値をチェックしないが、将来の統合時に
publish 成否に応じた処理が必要になった場合に備える。
`jassert(!isShutdownInProgress())` は採用しなかった。理由: shutdown 中の
publish 試行は異常状態ではなく、防御的ガードとして静かに skip するのが
Practical Stable の思想に合致する。

### 3.2 Path 3 (Timeout Recovery) の変更 — ★ 唯一の実質的変更

**変更前** (`AudioEngine.Timer.cpp:600-624`):

```cpp
if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
{
    diagLog("[HEALTH] Crossfade timeout detected, initiating recovery");

    // 1. retire（既存）
    auto* doneRaw = exchangeFadingRuntimeDSP(nullptr);
    if (doneRaw != nullptr
        && reinterpret_cast<uintptr_t>(doneRaw) != (~static_cast<uintptr_t>(0)))
    {
        DSPLifetimeManager lifetime(*this);
        lifetime.retire(doneRaw);
    }

    // 2. unregister（既存）
    const auto activeId = convo::consumeAtomic(
        activeCrossfadeId_, std::memory_order_acquire);
    if (activeId != 0u)
    {
        crossfadeAuthorityRuntime_.unregisterCrossfade(activeId);
        convo::publishAtomic(activeCrossfadeId_,
            uint64_t{0}, std::memory_order_release);
    }

    // 3. complete（既存）
    crossfadeRuntime_.complete();

    diagLog("[HEALTH] Crossfade timeout recovery completed");
    // ※ idle publish なし — これが唯一の欠落
}
```

**変更後**:

```cpp
if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
{
    diagLog("[HEALTH] Crossfade timeout detected, initiating recovery");

    // 1. retire（既存: 変更なし）
    auto* doneRaw = exchangeFadingRuntimeDSP(nullptr);
    if (doneRaw != nullptr
        && reinterpret_cast<uintptr_t>(doneRaw) != (~static_cast<uintptr_t>(0)))
    {
        DSPLifetimeManager lifetime(*this);
        lifetime.retire(doneRaw);
    }

    // 2. unregister（既存: 変更なし）
    const auto activeId = convo::consumeAtomic(
        activeCrossfadeId_, std::memory_order_acquire);
    if (activeId != 0u)
    {
        crossfadeAuthorityRuntime_.unregisterCrossfade(activeId);
        convo::publishAtomic(activeCrossfadeId_,
            uint64_t{0}, std::memory_order_release);
    }

    // 3. complete（既存: 変更なし）
    crossfadeRuntime_.complete();

    // 4. publish 前準備（publishIdleWorldOnly は前準備を含まない）
    //    setDryHoldSamples/refreshSnapshot は呼び出し側で明示的に実行する。
    crossfadeRuntime_.setDryHoldSamples(0);
    refreshCrossfadePreparedSnapshotFromAtomics();

    // ★ A-4: Idle world publish（追加）
    //   Timeout Recovery の完了後、AudioThread が正しく idle 状態を観測できるよう
    //   publishIdleWorldOnly で RuntimePublishWorld を発行する。
    //   強制完了のため HardReset policy を使用（DSPTransition と同一基準）。
    {
        const convo::RuntimeReaderContext messageCtx{
            messageThreadRcuReader, convo::ObserveChannel::Message };
        const auto runtimeReadHandle = makeRuntimeReadHandle(messageCtx);
        auto* currentAfterFade =
            resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        publishIdleWorldOnly(currentAfterFade,
            convo::TransitionPolicy::HardReset);
    }

    diagLog("[HEALTH] Crossfade timeout recovery completed");
}
```

### 3.3 Path 1 (Timer) と Path 2 (DSPTransition) — 変更なし

両方とも既に正しく動作している。変更の必要はない。

Path 1 の publish 部分（Timer.cpp:404-440）は `publishIdleWorldOnly()` に置き換え可能だが、
**現時点では変更しない**。将来のリファクタリング候補として位置づける。

Path 2 (`DSPTransition::onTransitionComplete()`, DSPTransition.h:119-133) も同様。

### 3.5 コード検証結果（最終確認 2点）

v2.1 の設計検証として、以下の2点を実コードで確認した。

#### 確認①: `setDryHoldSamples(0)` は本当に Path1/Path2 共通責務か

実コードで確認:

| 経路 | ファイル | 行 | `setDryHoldSamples(0)` |
|:----:|---------|:--:|:----------------------:|
| Path 1 (Timer) | `AudioEngine.Timer.cpp` | 407 | ✅ 呼んでいる |
| Path 2 (DSPTransition) | `DSPTransition.h` | 120 | ✅ `engine_.crossfadeRuntime_.setDryHoldSamples(0)` |
| Path 3 (Timeout) | `AudioEngine.Timer.cpp` | 600-624 | ❌ 欠落（publish 自体がない） |

**結論**: Path 1 と Path 2 の両方で publish 直前に `setDryHoldSamples(0)` を呼んでいる。
`publishIdleWorldOnly()` 内に含めるのは正しい共通化。

#### 確認②: `complete()` 後に `startDelayBlocks_` 残留は起きないか

`CrossfadeRuntime::complete()` の実装（CrossfadeRuntime.h:88-92）:

```cpp
void complete() noexcept
{
    convo::publishAtomic(pending_, false, ...);
    convo::publishAtomic(queuedFadeTimeSec_, 0.030, ...);
    convo::publishAtomic(fadeStartTimestampUs_, 0, ...);
    // ★ startDelayBlocks_ と dryHoldSamples_ はリセットしない
}
```

`startDelayBlocks_` の設定箇所:

| 操作 | 設定値 | 箇所 |
|------|:------:|------|
| コンストラクタ | `0` | CrossfadeRuntime.h:160 |
| `start()` | `0` | CrossfadeRuntime.h:46 |
| `reset()` | `0` | CrossfadeRuntime.h:107 |
| `setStartDelayBlocks(0)` | `0` | Timer.cpp:406 (Path 1 のみ) |

非零を設定するコードは**存在しない**。

**結論**: `complete()` は `startDelayBlocks_` を明示的にリセットしないが、
`start()` が常に 0 に設定しており、非零にする経路がない。
残留リスクは理論上存在するが実質的な影響はゼロ。
Path 1 の `setStartDelayBlocks(0)` は Path 1 固有の防御的リセットであり、
Timeout Recovery に追加する必要はない。

### 3.6 追加検証結果（レビュー指摘対応）

以下の3点を実コードで追加検証した。

#### 検証③: Timeout Recovery 未publish のリスク分析

AudioThread の DSP 解決経路と retire タイミングを実コードで検証した。

**AudioThread の fading 解決** (`AudioEngine.Processing.AudioBlock.cpp:171`):

```cpp
DSPCore* fading = resolveFadingRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandleRef);
```

**`resolveFadingRuntimeDSPFromRuntimeWorldOnly()`** (`AudioEngine.h:2499-2505`):

```cpp
return static_cast<DSPCore*>(runtimeWorld->engine.fading);
```

**`EngineRuntime::fading` のデータ型**: `void*` — 生ポインタ。参照カウントなし。
`RuntimePublishWorld` 構築時 (`buildRuntimePublishWorld()`, `RuntimeBuilder.cpp:188`):

```cpp
worldOwner->engine = engineState;  // EngineRuntime 全体のコピー代入
// engineState.fading = next (DSPCore* next パラメータ)
```

**`DSPLifetimeManager::retire()` の実体** (`DSPLifetimeManager.h:56-72`):

```cpp
void retire(AudioEngine::DSPCore* dsp) noexcept
{
    // 1. Release DSP handle
    if (!engine_.retireDSPHandleForRuntime(dsp))
        return;
    // 2. enqueueRetire — 即時 delete ではなく、epoch 管理下に置く
    const uint64_t epoch = router_->currentEpoch();
    router_->enqueueRetire(static_cast<void*>(dsp),
                           &AudioEngine::destroyDSPCoreNode,
                           epoch);
    // destroyDSPCoreNode は epoch 回収時に後日実行される
}
```

**タイムライン（A-4 未適用）:**

```
T+0ms:   Timeout Recovery
         → exchangeFadingRuntimeDSP(nullptr)  // atomic slot = nullptr
         → DSPLifetimeManager::retire(done)   // enqueueRetire (即時 delete しない)
         → crossfadeRuntime_.complete()       // pending = false
         ⇒ DSPCore はまだ存活（epoch 未回収）

T+0ms:   AudioThread getNextAudioBlock
         → resolveFadingRuntimeDSPFromRuntimeWorldOnly()
           → runtimeWorld->engine.fading = 旧DSP へのポインタ
           → DSPCore は存活中 → まだダングリングではない
         → processCrossfadeDelayGateIfPending(fading, ...)
           → prepared.pending == false によりスキップ
         → canCrossfade = (fading!=nullptr && isSmoothing())
           → isSmoothing() == false (complete() 後) → false
         ⇒ 直ちにクラッシュする経路は blocked

T+30ms:  Epoch 回収 → destroyDSPCoreNode 実行 → DSPCore 破棄
         ⇒ runtimeWorld->engine.fading = ダングリングポインタに変化

T+31ms+: AudioThread が古い world を読み続ける限り
         ダングリングポインタが残る（UAF ウィンドウ）
```

**UAF リスク評価:**

| 要素 | 判定 | 根拠 |
|------|:----:|------|
| `engine.fading` の型 | ✅ 生ポインタ (`void*`) | 参照カウントなし、直接所有 |
| `retire()` の即時性 | ❌ 即時 delete しない | `enqueueRetire` — epoch 管理 |
| UAF ウィンドウ存在 | ✅ 存在する | epoch 回収後、次回 publish 前の間 |
| 保護機構 | ✅ あり | `pending=false` + `isSmoothing()=false` で大部分の経路は blocked |
| 将来のリスク | ⚠ 低い | 既存経路の保護は堅牢だが、新規コードが fading を直接 dereference する可能性 |

**A-4 の効果:**
新しい idle world 発行により `hasFadingRuntime=false, fading=nullptr` が設定され、
UAF ウィンドウが epoch 回収前に閉じられる。
即時性のあるバグ修正ではなく「UAF 可能性を排除する予防的修正」として位置づける。

`resolveActiveRuntimeDSPFromRuntimeWorldOnly()` の実体:

```cpp
// AudioEngine.h:2509-2514
const auto* runtimeWorld = getRuntimeWorldFromReadHandle(runtimeReadHandle);
return (runtimeWorld != nullptr)
    ? static_cast<DSPCore*>(runtimeWorld->engine.current)
    : nullptr;
```

`runtimeWorld->engine.current` は最後に publish された RuntimeWorld に記録された
アクティブ DSP を指す。Timeout Recovery の `exchangeFadingRuntimeDSP(nullptr)` は
**アトミック変数の fading スロットをクリアするのみ**で、RuntimeWorld の
`engine.current` には影響しない。

したがって `resolveActiveRuntimeDSPFromRuntimeWorldOnly()` が返す DSP は
Timeout 後も有効。Path 3 の publish で新しい idle world が発行されることで、
AudioThread が正しく idle 状態を観測できるようになる。

**結論**: ✅ Timeout Recovery の publish は意味論的に正当。RuntimeWorld 整合性を
改善する方向に働く。

#### 検証④: generation/publication sequence guard は不要

`notifyTransitionComplete()` の持つ guard は以下の2つ:

- **Generation Guard**: `deferred.guard.generation != currentGen` → stale discard
- **Publication Sequence Guard**: `deferred.guard.sequence < currentPubSeq` → stale discard

これらは両方とも **deferred publish の stale 検出** のためのものであり、
`runtimeOrchestrator_->deferredSlot_` に保留された publish request が
transition 完了後に submit される際の二重検査として機能する。

Path 3 (Timeout Recovery) は:

- `notifyTransitionComplete()` を経由しない（即時 publish）
- `deferredSlot_` を使用しない
- 新しい world を現在の DSP 状態から直接構築する

したがって generation guard / publication sequence guard は不要。

**結論**: ✅ `publishIdleWorldOnly()` に generation guard や sequence guard を
追加する必要はない。

#### 検証⑤: `buildRuntimePublishWorld()` の nullptr 安全性

`buildRuntimePublishWorld()` は常に `RuntimePublishWorld::createForBuilder()`
で世界を生成し、nullptr を返さない。

`RuntimePublicationCoordinator::publishWorld()` にも防御的チェックがある:

```cpp
// RuntimePublicationCoordinator.h:97-98
void publishWorld(convo::aligned_unique_ptr<World> worldOwner) noexcept
{
    if (!worldOwner)  // ← nullptr ガード
        return;
```

加えて `publishIdleWorldOnly()` 内で `currentAfterFade == nullptr` をチェックしており、
三重の防御が効いている。

**結論**: ✅ nullptr リスクはない。`if (worldOwner)` チェックは不要。

### 3.4 Path 1 将来置き換え案（参考）

```cpp
// 現状（約40行のinline処理）を以下のように置き換え可能:
if (fadeCompleted)
{
    // SPSC handoff（AudioThread 移行に備えた将来設計: 変更なし）
    const auto completedId = ...;
    if (completedId != 0u) { /* consumeCompletedFade → endCrossfade */ }

    // retire（変更なし）
    auto* doneRaw = exchangeFadingRuntimeDSP(nullptr);
    if (auto* done = ptr_unwrap(doneRaw))
    { DSPLifetimeManager lifetimeMgr(*this); lifetimeMgr.retire(done); }

    // complete（変更なし）
    crossfadeRuntime_.complete();
    crossfadeRuntime_.setStartDelayBlocks(0);

    // ★ publishIdleWorldOnly に置換
    publishIdleWorldOnly(convo::TransitionPolicy::SmoothOnly);

    sendChangeMessage();
}
```

---

## 4. C-2 修正案: 現状維持

### 4.1 line 64 reset 削除は非推奨

| 検討 | 判定 | 理由 |
|------|:----:|------|
| line 64 `crossfadeRuntime_.reset()` 削除 | ❌ **非推奨** | shutdown 決定性を損なうリスク > EmergencyDrain 回復の価値 |
| EmergencyDrain 内 `publishIdleWorldOnly()` 呼び出し | ❌ 効果小 | shutdown 中は publish スキップされる。retire/unregister/complete は既存処理で代替済み |
| 現状維持（dead code だが無害） | ✅ **推奨** | CompileFlag 未定義時は影響なし。定義時も crossfade recovery のみ dead code |

### 4.2 理由

1. **Practical Stable の優先順位**: shutdown 決定性 > EmergencyDrain 回復力
2. **既存の graceful drain で十分**: releaseResources は graceful drain → tryReclaim → drainDeferredRetireQueues → VerifyDrained の流れを持ち、crossfade が残存しても適切に処理される
3. **crossfade 残存リスクは極小**: releaseResources は Timer 停止後に実行されるため、タイマー経由の新規 crossfade は発生しない

### 4.3 EmergencyDrain の価値

crossfade recovery ブロックは dead code だが、EmergencyDrain 全体としては以下の価値がある:

| 構成要素 | 実行可否 | 価値 |
|----------|:--------:|------|
| `clearDeferredForShutdown()` | ✅ 実行可能 | 保留 publish を強制クリア（既存の graceful drain と重複） |
| `tryReclaim()` | ✅ 実行可能 | 安全な tryReclaim（VerifyDrained 前の追加試行） |
| `crossfadeRuntime_.isPending()` → recovery | ❌ 常に false | dead code。削除候補だが優先度低 |
| DiagnosticMode `collectDrainAudit()` | ✅ 実行可能 | VerifyDrained より前の段階で drain 状態を診断 |

---

## 5. 影響ファイル一覧

| ファイル | A-4 | C-2 | 変更内容 |
|---------|:---:|:---:|----------|
| `AudioEngine.h` | ✅ | - | `publishIdleWorldOnly()` 宣言追加 |
| `AudioEngine.Transition.cpp` | ✅ | - | **新規ファイル**: 実装 |
| `AudioEngine.Timer.cpp` | ✅ | - | Path 3 (Timeout) に `publishIdleWorldOnly()` 呼び出し追加（3行） |
| `RuntimePublicationOrchestrator.cpp` | ✅ | - | `notifyTransitionComplete()` コメント追記（dead code 明示） |
| `DSPTransition.h` | ✅ | - | `onTransitionComplete()` コメント追記（設計継続） |
| `ISRShutdown.h` | - | ✅ | 変更なし |
| `ISRShutdown.cpp` | - | ✅ | 変更なし |
| `AudioEngine.Processing.ReleaseResources.cpp` | - | ✅ | 変更なし |

---

## 6. 設計判断一覧

| 判断 | 根拠 |
|------|------|
| retire/unregister/complete は共通化しない | 3経路で意図的に差異がある。DSPTransition は publish 時点で complete できない。共通化は設計境界を壊す |
| `publishIdleWorldOnly()` のみ抽出 | publish 前準備 (setDryHoldSamples/resfreshSnapshot) + publishWorld は全経路で同一パターン。共通化メリットが大きく、リスクが小さい |
| Path 1 (Timer) は変更しない | 既に正しく動作している。変更によるリグレッションリスク > 共通化のメリット |
| Path 3 (Timeout Recovery) のみ変更 | 唯一の明確な欠落（idle publish なし）。追加によるリスクは極小 |
| line 64 reset を削除しない | shutdown 決定性を最優先。EmergencyDrain は optional フェーズであり、既存動作を変更する価値がない |
| `notifyTransitionComplete()` は存続 | 将来の統合 entry point として責務定義を保持。ただし現状 dead code であることを明記 |

---

## 7. 実装手順

| 手順 | 内容 | ファイル | 変更量 |
|:----:|------|---------|:------:|
| 1 | `publishIdleWorldOnly()` 宣言追加 | `AudioEngine.h` | 2行 |
| 2 | `publishIdleWorldOnly()` 実装（新規ファイル） | `AudioEngine.Transition.cpp` | 〜25行 |
| 3 | `CMakeLists.txt` に新規ファイル追加 | `CMakeLists.txt` | 1行 |
| 4 | Path 3 に `publishIdleWorldOnly()` 呼び出し追加 | `AudioEngine.Timer.cpp` | 2行 + include |
| 5 | コメント更新 | `DSPTransition.h`, `RuntimePublicationOrchestrator.cpp` | コメントのみ |

### 実装後の期待状態

Path 3 (Timeout Recovery) 完全形:

```
EVENT_CROSSFADE_TIMEOUT
  ↓
exchangeFadingRuntimeDSP → retire          (既存)
  ↓
unregisterCrossfade → activeCrossfadeId_=0 (既存)
  ↓
crossfadeRuntime_.complete()               (既存)
  ↓
publishIdleWorldOnly(HardReset)            ★ NEW
  ↓
AudioThread が正しく idle 状態を観測可能
```

完成形責務マトリックス:

| 責務 | Path 1 Timer | Path 2 DSPTransition | Path 3 Timeout |
|------|:---:|:---:|:---:|
| retire | ✅ inline | ✅ inline | ✅ inline |
| unregister | ✅ inline | ❌ (設計上不要) | ✅ inline |
| complete | ✅ inline | ❌ (設計上不要) | ✅ inline |
| setDryHoldSamples+resfreshSnapshot | ✅ inline | ✅ inline | ✅ **publishIdleWorldOnly** |
| publish | ✅ inline | ✅ inline | ✅ **publishIdleWorldOnly** |

---

## 8. 将来の統合可能性

`notifyTransitionComplete()` の責務整理が完了した時点で、
以下のような段階的統合を検討できる:

```
Layer 1: publishIdleWorldOnly()         ← publish のみ（今回実装）
Layer 2: completeCrossfadeTransition()  ← Layer1 + retire+unregister+complete（将来）
Layer 3: notifyTransitionComplete()     ← Layer2 + stale discard + deferred submit（将来）
```

各レイヤーは独立した責務を持ち、上位レイヤーが下位を呼び出す構成。
現時点では Layer 1 のみ実装し、Layer 2/3 は `notifyTransitionComplete()` の
責務整理が完了するまで延期する。
