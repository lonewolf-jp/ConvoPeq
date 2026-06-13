# Publish Helper Adoption Plan — A-4 最終設計（v3.0）

**Version**: 3.0
**Date**: 2026-06-13
**Base**: publish_path_unification_plan.md v2.0 + Practical Stable ISR Bridge Runtime 評価結果
**Status**: **Phase 1 のみ実装 — Phase 2 全面却下、Phase 3 保留継続**

---

## 0. 改訂履歴

| 版 | 日付 | 変更内容 |
|:--:|------|----------|
| 1.0 | 2026-06-13 | 初版: 3層アーキテクチャ + 5フェーズ |
| 2.0 | 2026-06-13 | 全未確定事項を調査確定: Phase 3 判断、3層の責務再定義 |
| **3.0** | **2026-06-13** | **Practical Stable ISR Bridge Runtime 評価で Phase 2 全面却下。Layer 2 を廃止し、Layer 1（publishIdleWorldOnly）のみのアーキテクチャに変更。Phase 1 のみ実装対象。** |

---

## 1. 評価結果サマリー

### 最終判定

| 項目 | 評価 | アクション |
|------|:----:|-----------|
| **Phase 1** (DSPTransition → publishIdleWorldOnly) | ✅ **推奨** | 実装する |
| **Phase 2a** (completeCrossfadeTransition 新規関数) | ❌ **却下** | 責務過大・正常系/異常系混在 |
| **Phase 2b** (Path 3 → completeCrossfadeTransition) | ❌ **却下** | 可読性低下メリットなし |
| **Phase 2c** (Path 1 → completeCrossfadeTransition) | ❌ **却下** | 最も危険・SPSC経路複雑 |
| **Phase 3** (notifyTransitionComplete 再有効化) | ⏸️ **保留継続** | v2.0 判断を維持 |

---

## 2. コード検証結果

### 2.1 3経路の実体確認（Timer.cpp / DSPTransition.h）

#### Path 1: Crossfade 正常完了（Timer.cpp:379-427）

```cpp
// ★ fadeCompleted: 通常のクロスフェード完了経路
fadeCompleted = m_coordinator.tryCompleteFade();
if (fadeCompleted) {
    // --- SPSC handoff（AudioThread→MessageThread）---
    const auto completedId = consumeAtomic(activeCrossfadeId_, acquire);
    if (completedId != 0u) {
        crossfadeRuntime_.notifyFadeComplete(completedId);
        if (crossfadeRuntime_.consumeCompletedFade(ev)) {
            dspHandleRuntime_.endCrossfade(ev.id);           // ★ AudioThread完了通知
            crossfadeAuthorityRuntime_.unregisterCrossfade(ev.id); // ★ CrossfadeID解放
        }
        publishAtomic(activeCrossfadeId_, 0u, release);
    }

    // --- retire + complete + publish ---
    auto* doneRaw1 = exchangeFadingRuntimeDSP(nullptr);
    if (auto* done = ptr_unwrap(doneRaw1))
    {
        DSPLifetimeManager lifetimeMgr(*this);
        lifetimeMgr.retire(done);                           // ★ Fading DSP退役
    }
    crossfadeRuntime_.complete();                            // ★ CrossfadeRuntime完了
    crossfadeRuntime_.setStartDelayBlocks(0);                // ★ Path1 のみ呼ぶ
    crossfadeRuntime_.setDryHoldSamples(0);                  // ★ publish前準備
    refreshCrossfadePreparedSnapshotFromAtomics();            // ★ publish前準備

    auto* currentAfterFade = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
    if (currentAfterFade != nullptr) {
        auto coordinator = makeRuntimePublicationCoordinator();
        auto worldBuilder = convo::RuntimeBuilder(*this);
        worldBuilder.setHealthStateRef(getHealthStateRef());
        auto worldOwner = worldBuilder.buildRuntimePublishWorld(
            currentAfterFade, nullptr,
            convo::TransitionPolicy::SmoothOnly, 0.0, false);
        coordinator.publishWorld(std::move(worldOwner));      // ★ Idle world publish
    }
    sendChangeMessage();                                     // ★ Path1 のみ
}
```

**特徴**:

- 正常系: AudioThread が完了通知済み
- SPSC handoff で 3 コンポーネントが連携（CrossfadeRuntime → DSPHandleRuntime → CrossfadeAuthority）
- `setStartDelayBlocks(0)` と `sendChangeMessage()` は Path 1 のみ必要
- publish は `SmoothOnly`

#### Path 2: DSPTransition 経由の publish 完了後処理（DSPTransition.h:onTransitionComplete）

```cpp
void onTransitionComplete(DSPCore* currentAfterFade) noexcept
{
    if (currentAfterFade == nullptr) return;

    auto* doneRaw = engine_.exchangeFadingRuntimeDSP(nullptr);
    if (auto* done = ptr_unwrap(doneRaw))
    {
        DSPLifetimeManager lifetime(engine_);
        lifetime.retire(done);                               // ★ Fading DSP退役
    }

    engine_.crossfadeRuntime_.setDryHoldSamples(0);         // ★ publish前準備
    engine_.refreshCrossfadePreparedSnapshotFromAtomics();   // ★ publish前準備

    // publish idling world
    auto coordinator = engine_.makeRuntimePublicationCoordinator();
    auto worldBuilder = convo::RuntimeBuilder(engine_);
    worldBuilder.setHealthStateRef(engine_.getHealthStateRef());
    auto worldOwner = worldBuilder.buildRuntimePublishWorld(
        currentAfterFade, nullptr,
        convo::TransitionPolicy::HardReset, 0.0, false);
    if (worldOwner) {
        coordinator.publishWorld(std::move(worldOwner));     // ★ Idle world publish
    }
}
```

**特徴**:

- 呼び出し元なし（dormant integration hook）。`RuntimePublicationOrchestrator::notifyTransitionComplete()` 内からのみ到達可能。
- retire + publish 前準備 + publish のシンプルな構成
- unregisterCrossfade, complete(), setStartDelayBlocks(0) は不要
- publish は HardReset

**確認済み呼び出し関係**（2026-06-13 実コード検証）:

```
[nobody calls]                                          ← 呼び出し元なし
  → RuntimePublicationOrchestrator::notifyTransitionComplete()  [Orchestrator.h:58, cpp:251]
    → transition_.onTransitionComplete()                [DSPTransition.h:117]
      → exchangeFadingRuntimeDSP → retire
      → setDryHoldSamples(0)
      → refreshCrossfadePreparedSnapshotFromAtomics()
      → buildRuntimePublishWorld + publishWorld (HardReset)   ← ★ ここを publishIdleWorldOnly に置換（Phase 1）
```

**確認**: `DSPTransition::onTransitionComplete()` は最新版でも存在し、
`buildRuntimePublishWorld + publishWorld` のインライン publish を維持している。
Phase 1 の置換対象はこの publish ブロック（8行）である。

#### Path 3: Crossfade Timeout 回復（Timer.cpp:600-650）

```cpp
// ★ EVENT_CROSSFADE_TIMEOUT: 異常回復経路
if (event.eventCode == convo::EVENT_CROSSFADE_TIMEOUT)
{
    // 1. 滞留中の fading DSP を強制退役
    auto* doneRaw = exchangeFadingRuntimeDSP(nullptr);
    if (doneRaw != nullptr && ptr_unwrap(doneRaw))
    {
        DSPLifetimeManager lifetime(*this);
        lifetime.retire(doneRaw);
    }

    // 2. アクティブな crossfade ID を取得して unregister
    const auto activeId = consumeAtomic(activeCrossfadeId_, acquire);
    if (activeId != 0u) {
        crossfadeAuthorityRuntime_.unregisterCrossfade(activeId);
        publishAtomic(activeCrossfadeId_, uint64_t{0}, release);
    }

    // 3. CrossfadeRuntime を complete 状態に戻す
    crossfadeRuntime_.complete();

    // ★ A-4: publish 前準備
    crossfadeRuntime_.setDryHoldSamples(0);
    refreshCrossfadePreparedSnapshotFromAtomics();

    // ★ A-4: Idle world publish（HardReset）
    {
        auto* currentAfterFade =
            resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle);
        (void)publishIdleWorldOnly(currentAfterFade,
            convo::TransitionPolicy::HardReset);
    }
}
```

**特徴**:

- 異常系: 強制終了・回復
- retire → unregister → complete → publish の明快な異常回復シーケンス
- A-4 導入後は publishIdleWorldOnly() を使用して責務明確
- `setStartDelayBlocks(0)` は不要
- publish は HardReset

### 2.2 検証: no-op 依存の有無

| 操作 | Path 1 時点 | Path 3 時点 | completeCrossfadeTransition 内 |
|------|:-----------:|:-----------:|:-----------------------------:|
| `exchangeFadingRuntimeDSP → retire` | 未実行 → 必要 | 未実行 → 必要 | ✅ 必要 |
| `unregisterCrossfade` | **済み** → no-op | 未実行 → 必要 | ⚠️ Path 1 では no-op |
| `crossfadeRuntime_.complete()` | 未実行 → 必要 | 未実行 → 必要 | ✅ 必要 |
| `setStartDelayBlocks(0)` | **必要** | **不要** | ⚠️ Path 3 で無意味 |

**結論**: no-op 依存が2箇所。将来の実装変更で壊れやすい。

---

## 3. 評価詳細

### 3.1 Phase 1（DSPTransition → publishIdleWorldOnly）: ✅ 推奨

`DSPTransition::onTransitionComplete()` のインライン publish ブロックを
`publishIdleWorldOnly()` に置換する。置換範囲:

```cpp
// 変更前:
setDryHoldSamples(0)
refreshSnapshot()
buildRuntimePublishWorld(currentAfterFade, nullptr, HardReset, 0.0, false)
publishWorld(worldOwner)

// 変更後:
setDryHoldSamples(0)
refreshSnapshot()
publishIdleWorldOnly(currentAfterFade, HardReset)
```

**根拠**:

- `publishIdleWorldOnly()` は publish 専用（retire/complete を含まない）
- DSPTransition の retire + publish 前準備 + publish の責務と一致
- 変更量: 3行、リスク極低

### 3.2 Phase 2a（completeCrossfadeTransition 新規関数）: ❌ 却下

**問題点**:

1. **正常系と異常系の混在**
   - Path 1: 正常完了（AudioThread 完了通知済み、endCrossfade 済み、unregister 済み）
   - Path 3: 異常回復（強制終了、完全リセット）
   - Practical Stable 原則: 「正常系と回復系はできるだけ分離」

2. **no-op 依存設計**
   - `unregisterCrossfade` が Path 1 で no-op になる前提
   - 将来の実装変更で壊れやすい

3. **setStartDelayBlocks(0) の不適切な共通化**
   - Path 1 のみ必要（最新コード確認済み）
   - Path 3 および DSPTransition では不要
   - 共通化のための押し込みは避けるべき

### 3.3 Phase 2b（Path 3 → completeCrossfadeTransition）: ❌ 却下

**問題点**:

- 現在の Timeout Recovery は retire → unregister → complete → publishIdleWorldOnly の明快な異常回復シーケンス
- A-4 導入後は責務が明確。completeCrossfadeTransition() に隠すと「何をしているか」が読みにくくなる
- 実装メリットが小さい

### 3.4 Phase 2c（Path 1 → completeCrossfadeTransition）: ❌ 却下（最も危険）

**問題点**:

- Path 1 は SPSC handoff 経由で CrossfadeRuntime / DSPHandleRuntime / CrossfadeAuthority の3者が絡む最も複雑な経路
- AudioThread → MessageThread の特殊経路を考慮する必要がある
- retire / complete / publish をまとめるより、現状のインライン実装を維持する方が安全
- Practical Stable 原則: 「複雑な経路は隠さない」

### 3.5 Phase 3（notifyTransitionComplete 再有効化）: ⏸️ 保留継続

v2.0 の判断を維持:

- 呼び出し元が存在しない
- transition_.onTransitionComplete が二重 retire リスク
- Layer B（Deferred管理）は deferred 複数スロット化まで価値限定的
- Path 1 の deferred submit は既存の triggerAsyncUpdate() で十分

---

## 4. 推奨する完成形アーキテクチャ

### Layer 1（publishIdleWorldOnly）

```
責務: publish のみ
  - buildRuntimePublishWorld + publishWorld
  - shutdown guard
  - nullptr guard
  → setDryHoldSamples / refreshSnapshot / retire / complete は含まない
```

変更不要。A-4 実装済み。`AudioEngine.Transition.cpp:10-39`。

### Path 1（Timer.cpp:379-427）— 現状維持

```
fadeCompleted
  → SPSC handoff（crossfadeRuntime → dspHandleRuntime → crossfadeAuthority）
  → exchangeFading → retire
  → complete()
  → setStartDelayBlocks(0)          ← Path 1 のみ必要
  → setDryHoldSamples(0)            ← publish 前準備
  → refreshSnapshot()               ← publish 前準備
  → buildRuntimePublishWorld + publishWorld  ★ publishIdleWorldOnly 未使用
  → sendChangeMessage()             ← Path 1 のみ必要
```

**注記**: Path 1 の publish 部分は `publishIdleWorldOnly()` へ置換可能だが、
SmoothOnly policy の指定が必要。変更は Phase 2c 却下により任意（低優先度）。

### Path 2（DSPTransition.h:onTransitionComplete）— Phase 1 適用

```
fadingDSP → retire
→ setDryHoldSamples(0)
→ refreshSnapshot()
→ publishIdleWorldOnly(HardReset)   ★ Phase 1 で置換
```

### Path 3（Timer.cpp:600-650）— 現状維持

```
exchangeFading → retire
→ unregisterCrossfade
→ complete()
→ setDryHoldSamples(0)              ← publish 前準備
→ refreshSnapshot()                 ← publish 前準備
→ publishIdleWorldOnly(HardReset)   ← A-4 実装済み
```

### 4.1 Path 1 が publishIdleWorldOnly を使わない理由

Path 1 の publish 部分（`buildRuntimePublishWorld → publishWorld`）は**技術的には** `publishIdleWorldOnly()` に置換可能です。
しかし意図的に置換していません。

**理由**:

| 観点 | 説明 |
|------|------|
| **安定性優先** | Path 1 は正常系の中核経路。SPSC handoff + retire + complete + setStartDelayBlocks + publish 前準備 + publish + sendChangeMessage の一連の流れは現状の安定動作が確認されている。変更によるリスク（3行削減の利益を上回る）を回避するため、あえてインライン維持。将来 Path 1 に別改修が発生した際に helper 化を再評価する |
| **SmoothOnly 指定** | Path 1 の publish は `SmoothOnly`。publishIdleWorldOnly は policy を引数で受け取れるが、呼び出し元で `SmoothOnly` を明示する必要があり、結局インラインと情報量は変わらない |
| **将来の AudioThread 主導** | SPSC handoff は将来的に AudioThread 主導の完了検出に移行する可能性がある。その場合、Path 1 の publish 部分はさらに変更されるため、現時点での共通化は premature |

**結論**: 技術的に可能だが、可読性・安定性を優先して意図的にインライン維持する。

### 4.2 Phase 1 実施後の重複コード状況

Phase 1 実施後、publish 実装は以下の状態になる:

| 経路 | publish 実装 | 備考 |
|:----:|:------------:|------|
| **Path 1** | `buildRuntimePublishWorld()` + `publishWorld()`（インライン） | 変更なし。インライン維持 |
| **Path 2** | `publishIdleWorldOnly()` | ✅ Phase 1 で置換 |
| **Path 3** | `publishIdleWorldOnly()` | ✅ A-4 で置換済み |

このように **完全統一は達成されない**。

- Path 2 と Path 3 は `publishIdleWorldOnly()` に統一される
- Path 1 のみインライン publish が残る

したがって「Publish Path Unification（統一路径）」という名称は実態より強く、
**「Publish Helper Partial Adoption（部分的採用）」** が正確である。

### 4.3 実態に即したプロジェクト名称

文書全体の名称を「Publish Path Unification Plan」から「Publish Helper Adoption Plan」に変更した。

| 旧名称 | 新名称 | 理由 |
|--------|--------|------|
| Publish Path **Unification**（統一路径） | Publish Helper **Adoption**（部分的採用） | 3経路中2経路のみ置換。Path 1 はインライン維持のため「統一」は不正確 |
| — | — | Practical Stable 原則: 「完全統一より責務分離の明確化」を優先 |

---

## 5. 実装計画（修正版）

| 順位 | 項目 | 内容 | 変更量 | リスク | 状態 |
|:----:|------|------|:------:|:------:|:----:|
| **1** | **Phase 1** | DSPTransition: publish → publishIdleWorldOnly | 小（3行） | 極低 | ✅ **実装完了** ✅ |
| — | Phase 2a | completeCrossfadeTransition 新規関数 | — | — | ❌ 却下 |
| — | Phase 2b | Path 3 → completeCrossfadeTransition | — | — | ❌ 却下 |
| — | Phase 2c | Path 1 → completeCrossfadeTransition | — | — | ❌ 却下（危険） |
| — | Phase 3 | notifyTransitionComplete 再有効化 | — | — | ⏸️ 保留継続 |

### Phase 1 実装詳細

**ファイル**: `DSPTransition.h`（`onTransitionComplete` メソッド内）

```cpp
// 変更前（現在のインライン publish）:
auto coordinator = engine_.makeRuntimePublicationCoordinator();
auto worldBuilder = convo::RuntimeBuilder(engine_);
worldBuilder.setHealthStateRef(engine_.getHealthStateRef());
auto worldOwner = worldBuilder.buildRuntimePublishWorld(
    currentAfterFade, nullptr,
    convo::TransitionPolicy::HardReset, 0.0, false);
if (worldOwner) {
    coordinator.publishWorld(std::move(worldOwner));
}

// 変更後（publishIdleWorldOnly に置換）:
// setDryHoldSamples / refreshSnapshot は既存コードで publish 前に実行済みのため、
// 置換対象は publish ブロックのみ。
engine_.publishIdleWorldOnly(currentAfterFade,
    convo::TransitionPolicy::HardReset);
```

---

## 6. ファイル影響一覧（確定）

| ファイル | Ph1 | 変更内容 |
|---------|:---:|----------|
| `DSPTransition.h` | ✅ | publish → `publishIdleWorldOnly()` 置換（3行） |
| `AudioEngine.Transition.cpp` | - | 変更なし（A-4 実装済み） |
| `AudioEngine.Timer.cpp` | - | **変更なし** — Path 1/3 は現状維持 |
| `AudioEngine.h` | - | **変更なし** — 宣言済み |
| `RuntimePublicationOrchestrator.cpp` | - | **変更なし** — dormant hook 維持 |
| 他の全ファイル | - | 変更なし |

---

## 7. 完成形責務マトリックス（v3.0 確定版）

| 処理 | Path 1 Timer | Path 2 DSPTransition | Path 3 Timeout |
|------|:---:|:---:|:---:|
| SPSC handoff（`endCrossfade`） | ✅ inline | ❌ | ❌ |
| `exchangeFadingDSP → retire` | ✅ inline | ✅ inline | ✅ inline |
| `unregisterCrossfade` | ✅ inline | ❌ 設計上不要 | ✅ inline |
| `crossfadeRuntime_.complete()` | ✅ inline | ❌ 設計上不要 | ✅ inline |
| `setStartDelayBlocks(0)` | ✅ inline（Path1 のみ） | ❌ | ❌ |
| `setDryHoldSamples(0)` | ✅ inline | ✅ inline | ✅ inline |
| `refreshSnapshot()` | ✅ inline | ✅ inline | ✅ inline |
| `publishIdleWorldOnly()` | 未使用（任意） | ✅ **Phase1適用** | ✅ A-4 済み |
| `sendChangeMessage()` | ✅ inline（Path1 のみ） | ❌ | ❌ |
| `notifyTransitionComplete()` | ⏸️（dormant hook） | ⏸️（dormant hook） | ⏸️（dormant hook） |

### 設計原則

1. **Layer 1 のみ**: publishIdleWorldOnly() の publish 専用責務を維持。Layer 2（completeCrossfadeTransition）は作らない。
2. **正常系と異常系は分離**: Path 1（正常完了）と Path 3（異常回復）を同じ関数に統合しない。
3. **no-op に依存しない**: 「呼ぶ必要がないなら呼ばない」。既に完了済みの操作を no-op 前提で再度呼ばない。
4. **複雑経路は隠さない**: Path 1 の SPSC handoff はインライン維持。関数に隠蔽して可読性を下げない。
5. **dormant hook 維持**: notifyTransitionComplete() / onTransitionComplete は将来の統合ポイントとして保持。削除も再有効化もしない。
   現在は Coordinator::notifyTransitionComplete 経由でのみ到達可能な dormant state であり、
   「呼び出し元がない完全な dead code」ではなく「いつでも呼び出せるが今は呼ばない」設計。
