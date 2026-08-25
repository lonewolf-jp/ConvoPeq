# D101-34-A — World Published-Domain Boundary / DeletionEntryType::World Producer Provenance Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0**）
- **判定**: **PASS**（全条件表で違反ゼロ。ただし重大な前提更新あり — §1）
- **基準**: ConvoPeq.md **2026-08-25 15:15 再生成版**（ローカル実ソースから）＋ 契約基準 `doc/work88/I4_DESIGN_CONTRACT.md` I4.D101 節

---

## 1. 最重要の前提更新（監査の結論に先行する事実）

**D101-34-B の対象だった「API separation」は、現行ソースに既に実装済みである。**

| 項目 | I4 契約記載（2026-08-16 時点） | 現行ソース実測（15:15 版） |
|---|---|---|
| `retireRuntimePublishWorldNonRt` | AudioEngine.h:3534 に存在（唯一の World producer を含む） | **参照ゼロ — 削除済み** |
| `retirePublishedRuntimeWorldNonRt(W, resetRevision)` | Step 2 設計のみ（NOT implemented） | **実装済み**（AudioEngine.h:3533）。PRECONDITION コメント「W ∈ PublishedDomain (must have passed publishAndSwap LP)」付き |
| `retireRejectedRuntimeWorldNonRt(W)` | 同上 | **実装済み**（h:3549）。「W ∉ PublishedDomain (rejected/unpublished)」+ `DeletionEntryType::Generic` 使用 |
| Init.cpp:67 の反例 | rejectedWorld → World → onRelease | **解消済み** — 現在は `bootstrapBridge.retireRejectedRuntimeWorldNonRt(rejectedWorld)`（Init.cpp:67）→ Generic |

→ I4 契約の「Step 2 = NOT STARTED」「INV-PUB-3 = DISPROVEN (direct counterexample)」は**陳腐化**。
D101 #1 の残作業は「実装」ではなく「**契約更新 + 形式証明**」となる。

---

## 2. World producer 全列挙（100%）

| # | site | 内容 |
|---|---|---|
| 1 | `AudioEngine.h:3545` | `enqueueDeferredDeleteNonRt(world, deleter, DeletionEntryType::World)` — **`retirePublishedRuntimeWorldNonRt` 内の唯一生成箇所** |

`grep 'DeletionEntryType::World'` の他のヒットは全て consumer 側の型比較
（ISRRetireRouter.cpp:82/:109/:525、RetireQuarantineStore.h:140/:177）。
✅ producer = 1 site（100% 列挙）。

---

## 3. caller 分類（Published / Rejected / Bootstrap / Shutdown）

### 3.1 `retirePublishedRuntimeWorldNonRt` — 全 caller（production 6 site）

| # | caller | 取得経路 | PublishedDomain 根拠 |
|---|---|---|---|
| P-1 | CtorDtor.cpp:246 | `clearedWorld` ← `clearBridge.clearPublishedRuntimeSnapshotsNonRt()` | `publishAndSwap(nullptr)` で swap-out された旧 current ✅ |
| P-2 | AudioEngine.Init.cpp:88 | bootstrap `oldWorld` ← `worldAuthority_.publish(...)` committed=true | publish LP 通過後の旧 current ✅ |
| P-3 | Processing.ReleaseResources.cpp:492 | `clearedWorld` ← clear path | 同 P-1 ✅ |
| P-4 | RuntimePublishExecutor.h:76 | `oldWorld` ← commit 成功後 | publish LP 通過 ✅ |
| P-5 | core/RuntimePublicationCoordinator.h:98 | `clearPublishedRuntimeSnapshotsNonRt()` 内 `writeAccess_.publishAndSwap(nullptr)` | ✅ |
| P-6 | core/RuntimePublicationCoordinator.h:147 | `oldWorld = writeAccess_.publishAndSwap(newWorld)` — **LP 直後** | ✅ |

テスト 2件（PartialPublicationRejectTests:113 / RuntimePublicationCoordinatorTests:56）は Test Bridge stub。✅

### 3.2 `retireRejectedRuntimeWorldNonRt` — 全 caller（production 2 site）

| # | caller | 状況 | 非 Published 根拠 |
|---|---|---|---|
| R-1 | AudioEngine.Init.cpp:67 | bootstrap `validatePublicationNonRt` 失敗 | `publishAndSwap` **前に**分岐 → 未公開 ✅ |
| R-2 | core/RuntimePublicationCoordinator.h:123 | `validatePublicationNonRt` 失敗 → `PublishStageResult::Rejected` | 同上 ✅ |

テスト stub 2件。✅

### 3.3 両関数を通らない unpublished World（直接破棄系）

| 経路 | 破棄方法 | World entry 化 | onRelease |
|---|---|---|---|
| facade enqueue 失敗（admission reject / ownerCh full / queue full） | `aligned_unique_ptr` デストラクタ / CallerDestroy | ❌ | ❌ |
| `ownerChannel().take(key)`（rollback） | caller 破棄 | ❌ | ❌ |
| `RuntimeWorldAuthority::publish()` 内部失敗 | publish 内で破棄（Init.cpp:96-99 コメント・dangling deref 防止確認済み） | ❌ | ❌ |

✅ いずれも Generic/直接破棄であり `DeletionEntryType::World` に入らない。

---

## 4. 条件表（指示の必須判定）

| 条件 | 必須判定 | 実測 |
|---|---|---|
| World producer 全列挙 | 100% | ✅ 1 site（h:3545） |
| Published World の全 caller | 100% | ✅ 6 site（P-1〜P-6） |
| Rejected World の全 caller | 100% | ✅ 2 site（R-1/R-2） |
| `DeletionEntryType::World` bypass | 0 | ✅ producer が単一関数内に閉じるため構造的に不可能 |
| unpublished → `World` entry | 0 | ✅ R-1/R-2 は Generic。直接破棄系も entry 化なし |
| published → Generic 誤分類 | 0 | ✅ P-1〜P-6 は全て World entry（Generic への誤経路なし） |
| `onRelease()` hidden path | 0 | ✅ World branch 付き consumer は 5 site のみ（§5）。type!=World では発火しない |
| `publishAndSwap()` 前の World が World entry 化 | 0 | ✅ pre-LP World は R-1/R-2（Generic）または直接破壊のみ |
| RuntimeStore destructor 時 current | shutdown contract 確認 | ⚠️ **構造確認まで**（§6）— 形式証明は未完（契約 Tier 4 🔴 のまま） |
| queue failure / validation rejection | PublishedDomain exclusion 確認 | ✅ facade CallerDestroy / R-1/R-2 Generic |

---

## 5. onAcquire / onRelease exactly-once 境界

| event | site 数 | 位置 |
|---|---|---|
| `onAcquire()` | **1** | AudioEngine.Commit.cpp:409（`worldRetirementReference_.onAcquire()` — publish LP 一致） |
| `onRelease()` | **5** | ①ISRRetireRouter.cpp:87（drain）②同:114（TerminalReclaimAuthority::drainAll）③ISRRetireRouter.h:128（recordWorldReclaim — tryReclaim の同期破壊経路）④RetireQuarantineStore.h:145（reclaimBatch）⑤同:182（drainAllUnsafe） |

⚠️ **契約値の陳腐化**: I4 記載「3 terminal sites」に対し現行は **5 sites**
（同期破壊経路 h:128 と quarantine shutdown drain :182 の追加）。exactly-once 証明
（INV-PUB-4）の基準値を **5 に更新する必要がある**。
各 World entry は単一 queue/store にのみ存在するため exactly-once の構造は維持されるが、
形式証明は未完（OPEN）。

---

## 6. `publishAndSwap()` と RuntimeStore destructor

- **LP 確認**: `WriteAccess::publishAndSwap` = `exchangeAtomic(store_->current, next, acq_rel)`
  （RuntimeStore.h）。current への atomic exchange が PublishedDomain membership の
  linearization point として機能（swap-in で加入、swap-out で離脱→retirePublished）。
- **Destructor**: `RuntimeStore` に user-declared destructor は存在せず、
  `std::atomic<T*> current` のデストラクタはポインタを delete しない。
  → 「RuntimeStore destructor は current を delete しない」は**構造的に確認**。
- **shutdown contract（current == nullptr at destruction）**: 
  ReleaseResources / CtorDtor の両 clear 経路が `clearPublishedRuntimeSnapshotsNonRt()` →
  `publishAndSwap(nullptr)` で current を null 化してから retire する構造を実測確認。
  ただし「全終了経路で nullptr 到達」の形式証明は契約 Tier 4 🔴 のまま（本監査では構造確認まで）。

---

## 7. INV-PUB / INV-WORLD-TYPE 判定

| Invariant | 契約状態 | 本監査後の状態 |
|---|---|---|
| INV-PUB-1（単一直列化 domain） | OPEN | 変更なし（本監査 scope 外） |
| INV-PUB-2 | OPEN | 変更なし |
| INV-PUB-3（W ∉ PD ⇒ R_ref=0） | **DISPROVEN**（反例記録） | ✅ **反例解消をコードレベルで確認**。形式証明は未作成 → 「code-fixed / formal proof pending」へ更新推奨 |
| INV-PUB-4（acquire/release exactly-once） | OPEN | 基準値修正が必要（release site 3→5）。証明は OPEN 継続 |
| INV-WORLD-TYPE | 提案段階 | ✅ **成立可能**。producer が単一関数に閉じ、PRECONDITION コメント・caller 分類完了のため、domain 违反はコード検査で機械的に検出可能。将来の debug assertion / type-state 強化と親和 |

---

## 8. 実装案の位置づけ（B 以降への引き継ぎ・実装はしない）

**D101-34-B は「API separation の新規設計」から「契約・証明の確定タスク」へ再定義すべき**:

1. `I4_DESIGN_CONTRACT.md` I4.D101 節の更新（Step 2 = IMPLEMENTED、反例解消の記録、
   onRelease site 数 3→5、INV-PUB-3 状態更新）
2. INV-PUB-4 の exactly-once 形式証明（5 terminal sites + 1 acquire site）
3. RuntimeStore shutdown contract（current==nullptr at destruction）の全終了経路列挙証明
4. （任意）retirePublished/retireRejected への debug assertion 追加検討

---

## 9. PASS 条件表（指示どおり）

* [x] 最新 ConvoPeq.md 再生成（15:15 版）
* [x] `DeletionEntryType::World` producer / caller / consumer / terminal release 全追跡
* [x] Published / Rejected / Bootstrap / Shutdown / failure-retry 分類（§3）
* [x] INV-PUB-1〜4 個別判定（§7）
* [x] INV-WORLD-TYPE 成立可否判定（§7 — 成立可能）
* [x] `publishAndSwap()` と PublishedDomain membership の関係確認（§6）
* [x] `onRelease()` hidden path 再監査（§5 — 5 site で網羅、hidden なし）
* [x] ソース変更 0
* [x] 実装案は記載のみで実装に進まない

# VERDICT: D101-34-A = **PASS**

（注: 「Step 2 未実装」前提での API separation 実装設計は不要 — 既存実装の検証と
契約更新が D101-34-B の実質内容となる）

## 次ステップ（D101-34-B 実施条件）

1. I4_DESIGN_CONTRACT.md の I4.D101 節を現状に合わせ更新（Step 1 CLOSED / Step 2 IMPLEMENTED /
   反例解消 / onRelease 5 sites / INV-PUB-3 状態）
2. INV-PUB-4 exactly-once 形式証明の基準値を 5 sites に修正
3. RuntimeStore shutdown contract の全終了経路列挙（Tier 4 🔴 解消への第一歩）
4. 上記完了後、残余 GAP の有無を再評価し、必要なら D101-34-C（テスト強化）へ
