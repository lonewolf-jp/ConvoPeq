# ND-01 — CW-8 PublishedWorldObservation 実装前 read-only 契約監査

```text
Date: 2026-09-01 / Type: 完全 read-only（Production source: 0 / Test: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0）
コード基準: ConvoPeq.md Generated 2026-09-01 16:28:11（指示どおり・--check FRESH / NEWER_SRC_COUNT=0 を実測）
対象: dash2 H.11.27.4 CW-8（第十八者 #37-C）を現行実装に対して再定義・再検証する
```

## 総合判定

> ## **ND-01 = GO（Case A — 最小修正）**
>
> **CW-8 の本質（world N + identity N+1 の混在不可）は、現行 topology で構造的に成立済み。**
> identity は独立 atomic ではなく **RuntimeState::publication として world 内部に bake 済み**であり、
> world pointer 1 回の acquire load がそのまま `{world, identity}` ペアの単一 linearized 読取になる。
> したがって CW-8 に必要なのは authority 新設ではなく **read-side 型強化（PublishedWorldObservation 導入）**
> のみ。X4 ownership topology 変更（Case B）には踏み込まない。

---

## 1. 現在の world pointer source（ND-01-A-1）

**唯一の物理 source: `RuntimeStore<RuntimeState, RuntimeWorldAuthority>::current`**（`std::atomic<RuntimeState*>`・`src/core/RuntimeStore.h:93`）。

read API は `RuntimeWorldAuthority`（src/audioengine/RuntimeWorldAuthority.h）に集約:

| API | 実装 | 呼び出し元（実測） |
|---|---|---|
| `observePublishedWorld()` | `runtimeStore_.observe()` | `AudioEngine.h:1147`（currentPublicationEpoch・RT 到達可）・`AudioEngine.h:3654`（委譲） |
| `consumeWorldHandle(ReadToken)` / `consumeWorldHandle()` | 同上 | `AudioEngine.h:1355/2246/3210/3477/3818`（RT/NonRT read paths）・`Commit.cpp:587` |
| `acquireReadToken()` | 空の opaque token | 上記 read path の precondition 表現のみ（**実効的な epoch 保護は持たない** — 詳細 §5） |

`observe()` は `consumeAtomic(current, acquire)` の 1 回 load（RuntimeStore.h:77-82）。**INV-X4-7/A/B（独立 read source 複数禁止・旧 getCurrent()/currentWorld_ は CW-3c で削除済み）により、替代 read source は存在しない。**

## 2. 現在の identity source（ND-01-A-2）

**identity は独立 atomic として存在しない。** 実測:

- `RuntimeState::publication`（`AudioEngine.h:204`、型 `convo::isr::PublicationSemantic`）= identity の実体。field: `sequenceId / epoch / mappedRuntimeGeneration / previousSequenceId`（ISRRuntimeSemanticSchema.h:252-258・全て `MutablePrePublish`）。
- `RuntimePublicationIdentity`（`AudioEngine.h:3494`: generation + worldId + publicationSequence）は **reserve 時の一時ハンドル**（`reserveRuntimePublicationIdentity()`）であり、永続 identity store ではない。reserve → builder bake → `RuntimeWorldAuthority::publish()` 内 `coordinator_.commit(...)` で **RuntimeState::publication に bake**（RuntimeWorldAuthority.h:231-237「commit metadata（publication bake + monotonicity check）」）。
- 導出 accessor は `world->publication.epoch` 等の直接読取のみ（`currentPublicationEpoch()` AudioEngine.h:1147、Commit.cpp:35-209 の validation）。

## 3. atomic object の数（ND-01-A-4）

| 対象 | atomic 数 | 内容 |
|---|---|---|
| world + identity | **1** | `RuntimeStore::current`（`atomic<RuntimeState*>`）のみ。identity は **world 内部 field** で、world pointer と同一オブジェクトに属する |
| その他の関連 atomic | 0 | world/identity を別 object で持つ atomic は存在しない（`RuntimePublicationState` は telemetry ledger・`PublicationLedger` であり identity source ではない。`PendingPublishRegistry` は enqueue→commit gap の一時 registry・別用途） |

**結論: 「別 atomic から読む torn-read 窓」は現行に存在しない。**

## 4. publication transaction の実体（ND-01-A-3 / A-4）

```text
[Producer・NonRT]
reserveRuntimePublicationIdentity()          ← generation/worldId/sequence 採番
  → builder が RuntimeState を構築（buildInput 等）
[ISR PublishExecutor::executePublish — RuntimePublishExecutor.h]
  sealRecursively()                          ← PR-5: publish 前 immutable 化（SealedObject）
  → authority.publish(owner, PublishMetadata{...})
      ├─ coordinator_.commit(...)            ← publication bake（world->publication へ書込）+ monotonicity check
      │    ＋ Faulted なら swap しない（transaction 原子性・WorldAuthority.h:241-245）
      ├─ release fence
      └─ writeAccess_.publishAndSwap(next)   ← 唯一の物理 swap（acq_rel exchange・RuntimeStore.h:49-50）
  → bridge.didPublish / willRetire / retire(oldWorld)
  → registry().unregister(seqId)
```

- **linearization point = `publishAndSwap` の acq_rel exchange（単一点）**。bake（commit）は swap より前に完了し（commit-before-swap ordering）、swap の release で identity 書込が可視化される。reader の acquire load は bake 済み identity を持つ world を見る。
- oldWorld は swap の戻り値として 1 度だけ取得され、PublishExecutor が retire（deferred delete）する。

## 5. Reader / Epoch protection の境界（ND-01-C）

実測した lifetime チェーン:

```text
reader: observe() [acquire, 非所有 borrow]
   ↓ （swap で置き換えられた旧 world）
PublishExecutor: bridge.retirePublishedRuntimeWorldNonRt(oldWorld)
   → enqueueDeferredDeleteNonRt(world, deleter, DeletionEntryType::World)   ← ただちに free しない
   → DeferredDeletionQueue（epoch 安全判定: currentEpoch() < minReaderEpoch() で reclaim）
   → drainDeferredRetireQueues が minReaderEpoch 進行後に deleter 実行
```

- **epoch 保護の本体は EBR**（`advanceRetireEpoch` / `m_retireRouter->minReaderEpoch()` / `isOlder(entry.epoch, minReaderEpoch)` — AudioEngine.h:4372-4384・5063、Retire.cpp:76-81 INV-EPOCH-2「reader が grace period 内に居る限り reclaim 禁止」）。
- `ReadToken` は **opaque 空トークン**（RuntimeWorldAuthority.h:164-171）であり epoch を取らない。protection は reader 側でなく **retire 側の EBR grace** で担保される（RT callback は同一スレッド逐次実行のため、callback 内で取得した world は次 callback まで物理 free されない構造）。
- `PublishedWorldObservation` を導入しても **ownership を持たない borrow 値**である限り、この境界は変化しない（observation の lifetime ≤ epoch protection の lifetime を構造として強制するには、observation を「1 callback 内消費」の慣約 + observe() 単一入口に限定する。型は pointer pair の copy で非所有）。
- **CW-8 実装は retire/reclaim authority を一切変更しない**（Case A の範囲では DeferredDeletionQueue / EpochDomain / LifetimeState に触れない）。

## 6. CW-8 を満たす最小データ構造（ND-01-B 方式比較）

| 方式 | 評価 | 判定 |
|---|---|---|
| **A: immutable publication record + 単一 acquire load** | **現行構造がまさにこれを既に実現している**: identity は publish 前 bake・sealed 後不変（MutablePrePublish + sealRecursively）、`atomic<RuntimeState*>` 1 回 load で `{world, identity}` が同時確定 | **採用（既に成立）** |
| B: world / identity を別 atomic で読む | torn-read 窓を**新規に導入**する（現行は別 atomic が存在しない） | 不採用（退化） |
| C: seqlock / double-check | 別 source が必要になるため topology 変更。現行はレコード単位 swap で retry 不要 | 不採用（Case B 化） |

**確定仕様（CW-8 の現行実装への再定義）:**

```text
PublishedWorldObservation（read-side 型・非所有）
{
    const RuntimeState* world;            // = RuntimeStore::current の 1 回 acquire load の結果
    const PublicationSemantic* identity;  // = &world->publication（同一オブジェクト由来・生成不能）
}
```

- **identity を world と別に構築できない型にする**（identity は `&world->publication` からのみ得る — pointer pair を外部から捏造できないよう、観測は authority の factory 関数経由のみ）。
- `RuntimeWorldAuthority` に `[[nodiscard]] PublishedWorldObservation observePublishedObservation(const ReadToken&) const noexcept` を 1 つ追加（内部は現行 `runtimeStore_.observe()` 1 回 load + identity pointer 導出）。既存 `observePublishedWorld()` は維持（後方互換・委譲先は同一）。
- **新 authority / new atomic / topology 変更は一切ない。** INV-X4-3/5（二階層化禁止）・AC-ISR-1・BE-8 に反しない。

## 7. X4 ownership topology に踏み込むか（ND-01-E）

**踏み込まない（Case A 確定）。** 判定根拠:

- RuntimeWorldAuthority は既に Store を物理所有し（X4-B-2/3 完了済み・h:110-116）、WriteAccess/publishAndSwap は単一。read 側も INV-X4-7/A/B で単一 source 済み。**「既存 publication state → atomic snapshot → PublishedWorldObservation」の read-side strengthening だけで CW-8 が成立する。**
- RuntimeStore ownership / Owner template / WriteAccess / publication topology の変更は一切不要 → 過去の X4 topology change（Case B・大規模変更）を再開する根拠がない。

## 8. INV-ISR-06 / CW-5 / CW-8 の関係（ND-01-D）

```text
INV-ISR-06（退役・ownership identity source は publish() oldWorld / Lifetime — Coordinator.h:113）
    ↑ 前提を共有（identity は world 内部・単一 source）
CW-5: RuntimeStore::current.identity == RuntimeState::publication.identity（dash2:5012）
    ↑ 現行は「別 store が存在しない」ため構造的に恒真（CW-3c で metadata alias 削除済み）
CW-8: {world, identity} が同一 publication transaction 由来（単一 linearized 読取）
    ↑ 現行は 1 atomic load で両方得られるため意味的に成立。未確定なのは「型がこれを表明していない」ことだけ
```

**判定: CW-8 は「新しい publication semantic」ではなく既存 invariant 群の read-side 型による強化（明示化）である。** 新 authority・新 ownership の追加は発生しない → NO-GO 条件（新 ownership authority の追加）には該当しない。X4-7/X4-8（独立 read source 複数禁止・current = physical source）は CW-8 によってむしろ強化される（観測結果の型が単一 source 由来であることを表明）。

## 9. 実装対象ファイル（ND-03 最小実装の範囲・Case A）

| ファイル | 変更内容 | 規模見込み |
|---|---|---|
| `src/audioengine/RuntimeWorldAuthority.h` | `PublishedWorldObservation` struct（公開型・identity pointer は world 由来のみ）+ `observePublishedObservation(ReadToken)` factory 1 件 | +30 行前後 |
| `src/tests/RuntimeWorldAuthorityObservationTests.cpp`（新規）または既存 test への追記 | T-CW8-1〜7（§10） | +150 行前後 |
| その他 | **変更しない**（AudioEngine / RuntimeStore / Coordinator / retire / CMake 追記のみで production diff 1 ファイル） | 0 |

禁止事項（指示どおり）: RuntimeStore ownership 変更 / Publish authority 変更 / Retire authority 変更 / Recovery 変更 / Crossfade 変更 / Admission 変更 / `RuntimeStore::current` の型変更。

## 10. targeted test ケース（ND-02 設計・実装は次工程）

| ID | 検証 | 方法（型強化を活かす） |
|---|---|---|
| T-CW8-1 | single publication → world/identity 一致 | `observePublishedObservation()` が返した `obs.world->publication.sequenceId` と `obs.identity->sequenceId` の一致（同一 object 由来の表明） |
| T-CW8-2 | publication N → N+1 transition | 2 回 publish → 各 observation の identity.sequenceId が単調・旧 observation は identity N のまま凍結 |
| T-CW8-3 | rapid consecutive publications | 連続 publish 後も取得済み observation が過去 identity を保持（swap は観測を無効化しない） |
| T-CW8-4 | observation during transition | bake 前 identity（0）を持つ world が観測されないこと（commit-before-swap / Faulted 不 swap の帰結） |
| T-CW8-5 | observation + reader protection | observation 保持中に retire → deferred delete 経路が observation の world を即時破棄しない（EBR grace）・既存 retire テストと共存 |
| T-CW8-6 | observation does not extend ownership | observation が非所有（trivially copyable・deleter を持たない）ことを static_assert + retire カウンタ不変で検証 |
| **T-CW8-7** | **identity mismatch が構造的に表現不能** | **(a)** `PublishedWorldObservation` の identity member が `const PublicationSemantic*` であり、`world` と独立に構築する public 途径が無いこと（factory 以外の ctor を delete / private、`static_assert(!std::is_constructible_v<...>)` 系）。**(b)** テストでは observation 生成を authority factory に限定し、「world と identity が別 publication になる状態」を型システム上作れないことをコンパイル時表明する。EXPECT による値比較に頼らない |

T-CW8-7 の補足: 混在（world N + identity N+1）は現行では物理的に発生し得ない（identity は world 内部 field）。テストの目的は「将来 topology をいじったときに混在可能な構造に変わったらコンパイル/静的検査で落ちる」ガードを作ることである。既存 `ObservePathSingleSourceTests` / `RuntimeWorldAuthorityProjectionTests` はソース文字列検査型（contains ベース）のため、T-CW8-7 は **真の型レベル static_assert** で実装する点が既存テストとの差別化になる（`ISRSemanticValidationTests` は動作系テストで共存可）。

---

## 参照実測一覧

- RuntimeWorldAuthority.h 全文（291 行）・RuntimeStore.h 全文（97 行）・RuntimePublishExecutor.h（executePublish tail）・ISRSealedObject.h（seal 機構）・AudioEngine.h（147-215 RuntimeState / 3494 RuntimePublicationIdentity / 4372-4384・4881 EBR / 3583 retirePublishedRuntimeWorldNonRt = enqueueDeferredDeleteNonRt）/ ISRRuntimeSemanticSchema.h:252-264（PublicationSemantic + MutablePrePublish）/ Retire.cpp:76-81（INV-EPOCH-2）
- grep 実測: `observePublishedWorld|consumeWorldHandle` caller 12 件 / `->publication.` read path 8 件 / `acquireReadToken` 7 件 / `PublishedWorldObservation` 0 件（未実装の確認）

## 次工程（この監査の判定に基づく）

1. **ND-02**: T-CW8-1〜7 のテスト契約確定（§10 の設計を踏襲・T-CW8-7 は型レベル）→ 実装はまだ
2. **ND-03**: RuntimeWorldAuthority.h への最小実装（Case A・production diff 1 ファイル）
3. **ND-04**: targeted CW-8 tests + 既存 3 test suite + 全体 CTest（stress は判断保留）
4. **ND-06**: CR-α（BuildError retry contract audit）へ進む — CW-8 closure 後
