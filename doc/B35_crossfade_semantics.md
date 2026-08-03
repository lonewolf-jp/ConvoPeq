# B3.5 — Crossfade 意味論の読解記録（読解専用セッション）

> 日付: 2026-08-03
> スコープ: B3（技術的基盤）完了後の**読解専用**。Producer は一行も変更しない。
> 目的: B4（Producer flip）の突入条件を満たすため、`decision` / `boundary` / `version` の
> 生成者・生成時期・ライフタイム・PublishIntent フィールド対応・Producer 責任を確定する。

---

## 0. 結論サマリ

- `PublishIntent` の `decision/boundary/version` は **publish の情報ではなく Crossfade 意味論**。
- 現在の実装で **decision を生成しているのは rebuild 経路（Orchestrator::trySubmit）の 1 箇所のみ**。
  残り 6 つの同期 Producer は decision を生成していない（idle/後フェード publish、または既存 world から読取）。
- `PublishDecisionSnapshot` は**定義のみで、現在どこからも生成されない**（`enqueuePublicationIntent` が休眠のため）。
- `boundary` は strict gate: coordinator の metadata commit は **NonRTWorld のみ受理**（RTWorld は Faulted）。
- `version` は 7-arg commit では**未使用**（レガシー互換フィールド）。意味上の基準は `mappedRuntimeGeneration`。
- **★ Invariant**: Decision は publish の結果ではなく **world 生成時点の意味論を凍結した Snapshot**。
  enqueue 時点で凍結し、executePublish は絶対に再判定しない（詳細 §4.0）。

---

## 1. 意味論マッピング（6 項目）

### 1.1 `CrossfadeAuthority::Decision`

```cpp
// CrossfadeAuthority.h:25-30
struct Decision {
    bool needsCrossfade = false;
    bool oldHasIR       = false;
    bool newHasIR       = false;
    double fadeTimeSec  = 0.0;
};
```

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | `CrossfadeAuthority::evaluate(oldWorld, newWorld, policy)`（CrossfadeAuthority.h:37）。現在の呼び出しは **Orchestrator::trySubmit の 1 箇所**（RuntimePublicationOrchestrator.cpp:187-192）。 |
| いつ生成するか | リビルド/再発行要求の submit 時（Non-RT）。oldWorld == nullptr の場合は手動フォールバック（cpp:196-200）、`ISRHealthState::Critical` 時は強制抑制（cpp:207-210）。 |
| どのライフタイムで保持されるか | **ローカル変数（一時）**。decision は `spec.execution.{transitionPolicy, transitionActive, fadeTimeSec}` に焼き込まれ、`worldBuilder.buildRuntimePublishWorld`（cpp:226）で世界に反映後は破棄。 |
| PublishIntent のどのフィールドへ対応 | `payload.publish.decision`（PublishDecisionSnapshot の先頭 4 フィールドが同一）。executePublish が再構築（RuntimePublishExecutor.h:68-73）。 |
| 7 Producer のうち誰が責任 | **現在は誰も負っていない**（decision 生成は rebuild 経路のみ）。B4 で commitRuntimePublication enqueue 化時に導入が必要。 |

### 1.2 `PublishDecisionSnapshot`

```cpp
// ISRRuntimePublicationCoordinator.h:186-193
struct PublishDecisionSnapshot {
    bool needsCrossfade;
    bool oldHasIR;
    bool newHasIR;
    double fadeTimeSec;
    DSPHandle newHandle;
    DSPHandle oldHandle;
};
```

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | **現在はどこにも生成されない**（定義のみ。`enqueuePublicationIntent` が休眠のため）。 |
| いつ生成するか | 設計上は enqueue 時（A3 Step 5-3: "fixed at enqueue"）。trivially copyable / standard layout を static_assert 済み（194-197）= LockFreeRingBuffer 輸送用。 |
| どのライフタイムで保持されるか | Intent payload 内に固定保持。HANDLER-1（executePublish）が read-only で消費（"never re-decides", RuntimePublishExecutor.h:16）。 |
| PublishIntent のどのフィールドへ対応 | `payload.publish.decision`（そのもの）。 |
| 7 Producer のうち誰が責任 | **B4 で生成導入が必要**。newHandle/oldHandle は登録済み DSPHandle（executePublish が `dspHandleRuntime().resolve` で DSPCore* に復元, h:66-67）。 |

### 1.3 `makeCrossfadePolicy()`

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | `AudioEngine::makeCrossfadePolicy()`（AudioEngine.Publication.cpp:32）。静的フェード時間設定 7 種（os/ir/irLength/phase/directHead/nucFilter/tail FadeTimeSec）を acquire で読んで immutable POD `CrossfadePolicy` を生成。 |
| いつ生成するか | 判定の直前に毎回。現在は trySubmit のみが呼ぶ（cpp:190）。 |
| どのライフタイムで保持されるか | 一時。`evaluate()` の入力として渡されるだけ。 |
| PublishIntent のどのフィールドへ対応 | 直接対応なし（Decision 生成の入力）。 |
| 7 Producer のうち誰が責任 | B4 では decision を計算する全経路が呼ぶ。Non-RT 安全（MessageThread 書き込み atomic の acquire 読取）。 |

### 1.4 `cfDecision`（trySubmit ローカル）

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | Orchestrator::trySubmit 内のローカル（RuntimePublicationOrchestrator.cpp:187）。3 ステップ生成ロジック: (a) `evaluate(old,new,policy)` (b) oldWorld null フォールバック (c) Critical 抑制。 |
| いつ生成するか | リビルド publish 要求の受理後・world 構築前。 |
| どのライフタイムで保持されるか | ローカル。`spec.execution.*` への焼き込みと world 再構築に消費。 |
| PublishIntent のどのフィールドへ対応 | `PublishDecisionSnapshot` の元（4 フィールドが直接対応）。 |
| 7 Producer のうち誰が責任 | 将来は Producer 側（共通ヘルパー）が同一の 3 ステップを実行する。**B4 の推奨実装 = このロジックを共通ヘルパー化**。 |

### 1.5 `boundary`（`RuntimeBoundary{RTWorld, NonRTWorld}`）

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | Producer（enqueue 時）。意味: **publish コミットのスレッド文脈**。 |
| いつ生成するか | enqueue 時に固定（"fixed at enqueue"）。 |
| どのライフタイムで保持されるか | `PublishPayload.boundary` に固定 → `authority.commit(..., p.boundary, ...)` に渡る。 |
| PublishIntent のどのフィールドへ対応 | `payload.publish.boundary`。 |
| **strict gate** | coordinator の metadata commit は **NonRTWorld のみ受理**（ISRRuntimePublicationCoordinator.cpp:82-85。RTWorld は `CoordinatorState::Faulted`）。`MultiStagePublisher` デフォルトは NonRTWorld（h:414）。RTWorld は「RT スレッドから coordinator metadata を触らせない」ための設計意図。 |
| 7 Producer のうち誰が責任 | **全 7 Producer は Non-RT 文脈なので NonRTWorld を設定**。RTWorld を設定すべきケースは現状存在しない。 |

### 1.6 `version`

| 軸 | 回答 |
| --- | --- |
| 誰が生成するか | 現在は**未生成**（`PublishPayload.version` は休眠）。 |
| 意味 | "publish version"。coordinator の `getVersion()` は**現在 world の `publication.mappedRuntimeGeneration`**（ISRRuntimePublicationCoordinator.cpp:173-178）。4-arg レガシー commit は version を seq/epoch/mappedGen に流用（cpp:66-72）。 |
| いつ生成するか | enqueue 時に固定（"fixed at enqueue"）。 |
| どのライフタイムで保持されるか | `PublishPayload.version` に固定。7-arg commit では**無視される**（`std::uint64_t /*version*/`, cpp:78）。 |
| PublishIntent のどのフィールドへ対応 | `payload.publish.version`。 |
| 7 Producer のうち誰が責任 | B4 では**既存 world の `publication.mappedRuntimeGeneration` を渡すのが一貫**（getVersion の指針と一致）。ただし 7-arg commit が受けないため、**実装上は任意値で足りる**（FUTURE-4 で統合予定）。 |

---

## 2. 7 Producer 一覧と責務

呼び出し元（`commitRuntimePublication` の全 7 箇所）:

| # | Producer | file:line | スレッド | boundary | decision（現状） | crossfade 文脈 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Bootstrap | AudioEngine.Init.cpp:51 | ctor (Message) | NonRT | なし（初回 world） | なし（first world） |
| 2 | PrepareToPlay #1 | AudioEngine.Processing.PrepareToPlay.cpp:154 | Message | NonRT | **既存 world から読取**（policy/fadeTime/transitionActive, 142-144） | 再開時（fading 継続） |
| 3 | PrepareToPlay #2 | AudioEngine.Processing.PrepareToPlay.cpp:275 | Message | NonRT | なし（placeholder） | 初回 publish |
| 4 | ReleaseResources | AudioEngine.Processing.ReleaseResources.cpp:173 | Message | NonRT | なし（teardown） | なし |
| 5 | Timer（post-fade idle 同期） | AudioEngine.Timer.cpp:915 | Timer (Message) | NonRT | なし（fade 完了後） | 完了済み |
| 6 | Transition idle | AudioEngine.Transition.cpp:25 | Non-RT (DSPTransition) | NonRT | なし（fade 完了後） | 完了済み |
| 7 | Rebuild / Orchestrator | PublicationExecutor.cpp:41 | Non-RT worker | NonRT | **有効**（trySubmit が cfDecision 計算→world 焼き込み） | 遷移あり |

- decision 生成は **#7（rebuild 経路）のみ**。#2 は decision を生成せず、既存 world の
  `transitionPolicy / fadeTime / transitionActive`（AudioEngine.h:3126/3135/3145）を読んで
  `buildRuntimePublishWorld(current, fading, policy, fadeTime, transitionActive)` に渡している。
- #5/#6 は crossfade 完了後の idle 同期 publish。`crossfadeRuntime_.complete()` / 呼び出し側で
  前準備完了の前提（Transition.cpp:6-8）。decision は needsCrossfade=false。

---

## 3. ライフタイムとデータフロー（現在の rebuild 経路）

```text
enqueuePublicationIntentForRuntimeCommit   // AudioEngine.Commit.cpp:689 (Producer)
  └─ submitPublishRequest                  // → Orchestrator
       └─ trySubmit                        // RuntimePublicationOrchestrator.cpp
            ├─ cfDecision = CrossfadeAuthority::evaluate(old, new, policy)   // 187-192
            │     (null-old fallback / Critical 抑制)
            ├─ spec.execution.{transitionPolicy, transitionActive, fadeTimeSec} = 焼き込み  // 215-221
            ├─ worldOwner = buildRuntimePublishWorld(sealedSnapshot, spec)   // 226 (再構築)
            └─ executor_.publish → PublicationExecutor.cpp:41 → commitRuntimePublication
                 └─ coordinator.publishWorld (core store-swap; seq/epoch/mappedGen bake)
```

**ISR 経路（B4 目標・休眠中）**:

```text
commitRuntimePublication (B4 で enqueue 化)
  ├─ OwnerChannel.enqueue(key{seq,epoch,mappedGen}, owner)
  ├─ enqueuePublicationIntent(Intent)          // payload.{version,boundary,decision} を Producer が設定
  └─ waitForPublishReceipt(seqId)
        ↑ ISR CoordinatorLoop → executePublish（HANDLER-1）
             ├─ take(owner) → publishWorld(std::move(owner))（唯一の store-swap）
             ├─ authority.commit(auth, p.boundary, newWorld, p.version, seq, epoch, mappedGen)
             ├─ p.decision から Decision 再構築 → ctx.transition.onPublishCompleted(new, old, decision, lifetimeMgr)
             └─ onPublishCommitted(seqId) → notifyPublishReceipt（B3 基盤で結線済み）
```

---

## 4. B4 への推奨（decision 導出戦略）

### 4.0 ★ 設計原則（Invariant）— decision は「enqueue 時点で凍結した意味論」

> **Decision は「publish の結果」ではなく「world を生成した瞬間の意味論」を固定した Snapshot である。**
> enqueue 時点で意味論を凍結し、**executePublish（HANDLER-1）は絶対に再判定しない**。

```text
newWorld
   ↓
evaluate(oldWorld, newWorld, policy)     ← Producer / 共通ヘルパー（凍結点）
   ↓
PublishDecisionSnapshot                  ← enqueue で固定（"fixed at enqueue"）
   ↓
CoordinatorLoop → executePublish         ← read-only 消費
   ↓
Decision 再構築（p.decision → Decision）
   ↓
onPublishCompleted(new, old, decision, lifetimeMgr)
```

**退行ガード**: executePublish 内で `CrossfadeAuthority::evaluate` を呼び直す・
`makeCrossfadePolicy()` を再生成するコードを**追加しない**こと（コードコメントにも明記:
"Reads ONLY the publish payload fixed at enqueue ... never re-decides", RuntimePublishExecutor.h:16）。
判定は必ず Producer 側の共通ヘルパー（§4.1）のみ。

### 4.1 decision 生成 = 共通ヘルパー化（trySubmit の 3 ステップを移植）

`AudioEngine` に 1 つのヘルパーを置き、全 Producer が enqueue 時に呼ぶ（trySubmit のロジックと同一に保つ）:

```cpp
// 案: AudioEngine::makePublishDecisionSnapshot(newWorld, newHandle, oldHandle)
//  1. oldWorld = observePublishedWorld()
//  2. policy  = makeCrossfadePolicy()
//  3. cf = CrossfadeAuthority::evaluate(old, new, policy)   // old==null は手動 fallback
//  4. Critical 抑制（getHealthStateRef()）
//  5. PublishDecisionSnapshot{cf.*, newHandle, oldHandle}
```

- **根拠**: cfDecision の生成ロジック（Orchestrator.cpp:187-212）が唯一の正実装。それを Producer 側へ
  共通化すれば、rebuild 経路と 6 同期 Producer の挙動を一致させられる。
- **代替案（world 導出）**: decision は world 自体が `execution.transitionActive` + `dspProjection` を
  保持しているため、newWorld から導出も可能（needsCrossfade ≈ hasFadingRuntime/transitionActive、
  oldHasIR/newHasIR = dspProjection.irLoaded）。ただし fadeTimeSec は world に直接ない（execution には
  crossfadeStartDelayBlocks/crossfadeDryHoldSamples のみ）ため、取得元の確定が必要。
  → **enqueue 時 evaluate（共通ヘルパー）を推奨**（A3 設計の Option A と一致）。

### 4.2 各 Producer のパラメータ

| Producer | boundary | version | decision |
| --- | --- | --- | --- |
| #1 Bootstrap | NonRTWorld | 0（初回） | needsCrossfade=false, newHandle/oldHandle=null |
| #2 PrepareToPlay#1 | NonRTWorld | currentWorld.mappedGen | 共通ヘルパー（既存 world から読取値と一致させる） |
| #3 PrepareToPlay#2 | NonRTWorld | currentWorld.mappedGen | needsCrossfade=false（初回） |
| #4 ReleaseResources | NonRTWorld | currentWorld.mappedGen | needsCrossfade=false |
| #5 Timer | NonRTWorld | currentWorld.mappedGen | needsCrossfade=false（fade 完了後） |
| #6 Transition | NonRTWorld | currentWorld.mappedGen | needsCrossfade=false（fade 完了後） |
| #7 Rebuild | NonRTWorld | currentWorld.mappedGen | 共通ヘルパー（= 現 trySubmit と同一結果） |

> newHandle/oldHandle は executePublish が `dspHandleRuntime().resolve` する前提。
> #2 は fading 継続ケースで decision.needsCrossfade が true になり得る点に注意（再開時）。

### 4.3 B4 突入条件の充足状況

| 条件 | 状況 |
| --- | --- |
| 7 Producer 全てで decision/boundary/version の導出元が明確 | ✅ 本ドキュメント §2/§4.2 |
| PublishDecisionSnapshot の生成責任が整理されている | ✅ §1.2/§4.1（共通ヘルパー案） |
| commitRuntimePublication enqueue 化時も waitForPublishReceipt 同期契約が維持される | ✅ B3 基盤（PublishReceiptWaiter 結線済み）。B4 で waitForPublishReceipt 呼び出しを追加するのみ |

---

## 5. 参照ファイル / 行

- `src/audioengine/CrossfadeAuthority.h` — Decision / CrossfadePolicy / evaluate / kEvaluateRelevantFieldNames
- `src/audioengine/CrossfadeAuthority.cpp` — evaluate 実装（dspProjection 3 フィールド参照）
- `src/audioengine/AudioEngine.Publication.cpp:32` — makeCrossfadePolicy 実装
- `src/audioengine/RuntimePublicationOrchestrator.cpp:187-226` — cfDecision 生成・適用（唯一の正実装）
- `src/audioengine/ISRRuntimePublicationCoordinator.h:60-63` — RuntimeBoundary 定義 / 186-207 — Snapshot/Payload
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp:75-115` — 7-arg commit（NonRT のみ / bake semantics）/ 173-178 — getVersion
- `src/audioengine/RuntimePublishExecutor.h:30-83` — executePublish（take→commit→store-swap→decision→onPublishCompleted）
- `src/audioengine/RuntimeWorldAuthority.h:100-123` — getVersion / commit 委譲
- `src/audioengine/RuntimeBuilder.h:135-141` — buildRuntimePublishWorld オーバーロード
- `src/audioengine/AudioEngine.h:3126/3135/3145` — world crossfade 読取ヘルパー
- `src/audioengine/AudioEngine.Commit.cpp:689` — enqueuePublicationIntentForRuntimeCommit（現 Producer）
- `src/audioengine/PublicationExecutor.cpp:41` — rebuild 経路の commitRuntimePublication
