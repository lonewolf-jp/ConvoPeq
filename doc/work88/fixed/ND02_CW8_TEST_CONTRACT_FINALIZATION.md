# ND-02 — CW-8 PublishedWorldObservation Test Contract Finalization（read-only）

```text
Date: 2026-09-01
Production source: 0 / Test source: 0 / CMake: 0 / Build: 0 / CTest: 0 / stress: 0
baseline: ConvoPeq.md Generated 2026-09-01 16:28:11（--check FRESH・NEWER_SRC_COUNT=0 実測）
前提: ND-01 = GO / Case A（evidence/ND01_CW8_PRE_IMPL_CONTRACT_AUDIT.md）
```

## 総合判定

> ## **ND-02 = GO — ND-03（最小実装）に進行可**
>
> GO 条件 9 項目すべて成立・NO-GO 条件 7 項目すべて不該当（§6）。T-CW8 は 7 案から
> **4 実装テスト + 2 境界委譲 + 1 契約明示**に縮小（§1）。T-CW8-7 は private member +
> private ctor + friend factory の型設計で「独立 pair 構築を構造的に不能」にする（§3）。

---

## 1. T-CW8-1〜7 の採否（ND-02-1・ND-02-5）

実測に基づく再監査の結果、**既存テストとの重複を排除して 4 実装 + 2 委譲 + 1 契約明示に縮小**する。

| ID | 採否 | 根拠（実測） |
|---|---|---|
| **T-CW8-1** pair identity | **採用（runtime）** | `obs.identity == &obs.world->publication` の**アドレス同一性**を検証（ND-01 §2 の実測: identity は world 内部 member のみ）。既存テストは未カバー（ObservePath/Projection はソース文字列検査型・ISRSemanticValidationTests の X4-B テストは read API の null 検証のみ — `testRuntimeWorldAuthorityAdapter` 実測） |
| **T-CW8-2** N → N+1 | **採用（runtime・T-CW8-1 と同一フローに統合）** | `authority.publish()` はテストから駆動可能（前例: ISRSoakTests.cpp:419 `RuntimeWorldAuthority authority(*coordinator)` + `RuntimeState::createForBuilder`）。2 publish 後 `obsN.identity != obsN1.identity`・sequenceId 単調。**lifetime 衝突なし**: テストでは Bridge tail（retirePublishedRuntimeWorldNonRt）を呼ばないため oldWorld はテスト自身が所有し、dangling なし。本テストは EBR grace を検証しない（ND-02-5 の境界） |
| **T-CW8-3** rapid consecutive | **縮小統合** | 大量 publish は目的外。`observe→publish→observe` の deterministic 最小ケースを T-CW8-2 に含める（指示どおり） |
| **T-CW8-4** transition atomicity | **軽量 black-box のみ・ordering 自体は既存に委譲** | bake（`commit()` 内 `pubWorld->publication = PublicationSemantic{...}` — ISRRuntimePublicationCoordinator.cpp:112-118 実測）→ swap の ordering は既存契約（Test 7 コメント固定 + `tools/publication_authority_verifier.py` 静的検査）。production instrumentation は要求しない。テストは「観測された identity.sequenceId == publish 時 metadata.sequenceId（bake 値がそのまま観測される）」の black-box 表明のみ |
| **T-CW8-5** observation × lifetime | **契約明示のみ・新規テストなし（委譲）** | observation は非所有 borrow であり lifetime semantic を追加しない。EBR/retire は既存 suite（D8_2_B_2_Tests の DeferredDeletionQueue 系・retire 系テスト）がカバー。**「observation 保持で reclaim 遅延」という契約は追加禁止**（指示どおり・EBR reader protection と ownership の混同防止） |
| **T-CW8-6** non-owning type | **採用（型レベル static_assert）** | `is_trivially_copyable_v` / `!is_aggregate_v` / `!is_default_constructible_v` 等。ただし trivially copyable ≠ lifetime-safe であり、これは ownership contract の表明である（指示どおり） |
| **T-CW8-7** construction restriction | **採用・再設計（最重要 — §3）** | 型システムで「world N + identity N+1 の独立構築」を不能にする |

## 2. 各テストの重複確認（ND-02-5）

| 既存テスト | 方式 | CW-8 新規テストとの重複 |
|---|---|---|
| ObservePathSingleSourceTests | ソース文字列 contains 検査（main + throw 型） | なし（read path 単一 source の静的契約） |
| RuntimeWorldAuthorityProjectionTests | ソース文字列 contains / 再帰 scan（RuntimeReadHandle opaque 契約等） | なし。ただし CW-8 実装時に「observation 経由でも forbidden pattern が無いこと」を scan 対象に含まれることを確認（scan は src 全体のため自動的に適用される — 追加作業不要） |
| ISRSemanticValidationTests | **動作系**（Authority 構築ハーネス実在・static_assert 前例 h:839-850） | X4-B Test（null 観測・WriteAccess move-only）とは検証対象が異なる。**CW-8 テストは本ファイルに追記**（§9） |

## 3. T-CW8-7 再設計 — construction restriction の具体的な型設計（ND-02-2）

### ①〜④ 調査結果

| 生成経路 | 調査結果 | 対策 |
|---|---|---|
| ① aggregate init `PublishedWorldObservation{w, i}` | data member を public にすると aggregate となり可能（C++20 は default member initializer 付きでも aggregate） | **data member を private にする**（非 aggregate 化） |
| ② public constructor | private ctor + friend factory で遮断 | **ctor は private**・`friend class RuntimeWorldAuthority` |
| ③ copy / move / designated init | 非 aggregate 化で designated init 不可。**implicit copy/move は許容**（既存 observation の複製であり独立 pair 構築には使えない・trivial 性も維持） | copy/move は implicit のまま（user-declared にすると trivially_copyable を失う） |
| ④ `identity` の public member | public data member は ① を可能にするため NG | **private member + accessor `world()` / `identity()`** |

### 確定する型設計（ユーザー第一候補を採用 — convention 前例あり）

```cpp
// 設計前例: 同一ヘッダの ReadToken（private ctor + friend class RuntimeWorldAuthority・
// RuntimeWorldAuthority.h:164-171）— 現行 coding convention と整合する。
class PublishedWorldObservation
{
public:
    PublishedWorldObservation(const PublishedWorldObservation&) noexcept = default;
    PublishedWorldObservation& operator=(const PublishedWorldObservation&) noexcept = default;

    [[nodiscard]] const RuntimeState* world() const noexcept { return world_; }
    // identity の lifetime == world の lifetime（同一オブジェクト内部 pointer）。
    [[nodiscard]] const PublicationSemantic* identity() const noexcept { return identity_; }

private:
    friend class RuntimeWorldAuthority;
    PublishedWorldObservation(const RuntimeState* world,
                              const PublicationSemantic* identity) noexcept
        : world_(world), identity_(identity) {}

    const RuntimeState* world_ = nullptr;
    const PublicationSemantic* identity_ = nullptr;
};
```

**実測に基づく型性質（ND-03 実装時に static_assert で固定）:**

- `is_trivially_copyable_v` = true（implicit copy/move/dtor が trivial・private ctor は trivial copyability に影響しない）
- `is_aggregate_v` = **false**（private member + user-provided ctor）→ brace 指定初期化不能
- `is_default_constructible_v` = **false**（private ctor）
- `is_constructible_v<PublishedWorldObservation, const RuntimeState*, const PublicationSemantic*>` = **false**（テスト TU は friend ではないため）
- pointer form を採用する理由: **identity の lifetime を world の lifetime と構造的に結合**する（値 copy にすると identity が world 解放後も生存し「observation lifetime ≤ world protection」の契約が曖昧になる）。null world 観測時は `{nullptr, nullptr}`。

## 4. API lifetime contract の明文化（ND-02-3）

`observePublishedObservation(const ReadToken&)` に付与する契約（ヘッダコメントとして固定）:

```text
Ownership:  none — 非所有 borrow pair。observation は world / publication の寿命に影響しない
Lifetime:   返却 pointer は既存 reader/EBR 契約下でのみ有効
            （identity pointer の寿命 == world の寿命 — 同一オブジェクト内部）
Mutation:   observation は publication を変更しない（const のみ）
Publication: observation は publish に参加しない（Commit/swamp/registry に触れない）
Retire:      observation は retire/reclaim を起動・遅延・無効化しない
Thread:      既存 read API（observePublishedWorld / consumeWorldHandle）と同一の
             read-path 制約に従う（RT から呼び出し可能な既存 path と同等）
ReadToken:   意味を拡張しない — opaque token のまま・epoch ownership を追加しない
             （ND-01 §5 実測: 保護本体は retire 側 EBR・minReaderEpoch drain）
```

## 5. 既存 API との関係（ND-02-4 — delegation topology 確定）

```text
RuntimeStore::observe()   [acquire load — 単一 physical source・INV-X4-B]
        ↓ 1 回だけ呼ぶ
PublishedWorldObservation factory（RuntimeWorldAuthority 内・identity は &w->publication 導出）
```

- **新 API → 既存 physical source**（指示の推奨どおり）。`observePublishedWorld()` / `consumeWorldHandle()` の実装を複製・別実装化しない（同一 `runtimeStore_.observe()` 呼び出し）。
- 既存 3 API の semantics は**一切変更しない**（後方互換）。
- factory は **ReadToken 付き 1 overload のみ**（surface 最小化 — 既存 caller は `acquireReadToken` パターンを 6 箇所で使用中・実測）。tokenless overload は要求が出るまで追加しない。

## 6. ND-03 GO / NO-GO 判定

### GO 条件（9 項目 — すべて成立）

```text
[x] world source = RuntimeStore::current（単一・INV-X4-B 実測）
[x] identity source = world->publication（bake 済み・別 storage なし）
[x] new atomic = 0（読取 path の atomic は現行どおり 1 個のみ）
[x] new ownership authority = 0（非所有 borrow pair・authority 追加なし）
[x] new lifetime authority = 0（EBR/retire/reclaim に触れない）
[x] observation is non-owning（trivially copyable・deleter/refcount なし）
[x] independent pair construction is prohibited（private member + private ctor + friend factory）
[x] existing read API semantics unchanged（3 API 無変更・同一 observe() 委譲）
[x] 既存テストとの責務重複整理済み（§2 — 4 実装 + 2 委譲 + 1 契約明示）
```

### NO-GO 条件（7 項目 — すべて不該当）

```text
[ ] identity を別 storage に置く必要がある        → 不該当（bake 済み member で充足）
[ ] second atomic が必要                          → 不該当（単一 load で充足）
[ ] RuntimeStore ownership変更が必要              → 不該当（Case A）
[ ] ReadToken/EBR semantics変更が必要             → 不該当（opaque 維持）
[ ] existing API semanticsを変更しないと成立しない → 不該当（追記のみ）
[ ] public APIからpairを独立構築できる            → 不該当（§3 の型設計で構造遮断）
[ ] production instrumentationが必要              → 不該当（T-CW8-4 は black-box + 既存委譲）
```

**→ ND-03 GO。**

## 7. production diff の最小範囲（ND-03 前提）

| ファイル | 内容 |
|---|---|
| `src/audioengine/RuntimeWorldAuthority.h` のみ | `PublishedWorldObservation`（§3 の型・lifetime contract コメント block 付き）+ `observePublishedObservation(const ReadToken&)` factory 1 件（内部は `runtimeStore_.observe()` 1 回 + identity 導出） |

行数は ND-01 の「+30 行前後」を前提とせず、上記要素（struct + accessor 2 + private ctor + friend + factory + lifetime contract コメント）の実必要分で決める（見込み: +40〜55 行。contract コメントを含む）。**delegate 先は既存 observe() のため既存 API・RuntimeStore・Coordinator への変更は 0。**

## 8. test diff の最小範囲

| ファイル | 内容 |
|---|---|
| `src/tests/ISRSemanticValidationTests.cpp` への追記のみ | T-CW8-1+2+3 統合（runtime: Authority 構築 → publish ×2 → pair address identity・単調性）+ T-CW8-4 black-box 表明 + T-CW8-6/7 の static_assert 群。既存 `testRuntimeWorldAuthorityAdapter` の隣に配置（同ハーネス再利用・Coordinator はヒープ確保の前例どおり） |

新規テストファイルは作らない（CMake 変更回避）。

## 9. CMake 変更要否

**不要。** 実測: `ISRSemanticValidationTests` は既存 `add_executable`（CMakeLists.txt:181-208）に登録済みで、追記先が同一ファイルのため CMake diff 0 を維持できる。新規ファイルを作らない限り CMake に触れない。

## 10. ND-03 GO / NO-GO

> **GO** — §6 のとおり NO-GO 条件は 1 つも該当しない。ND-02 の契約（§3 型設計 / §4 lifetime contract / §5 delegation topology / §1 テスト縮小版）をそのまま実装仕様として ND-03 へ進める。

---

## 遷移

```text
ND-01 GO / Case A
   ↓
ND-02 Test Contract Finalization = GO   ← 本報告（コード変更 0）
   ↓
ND-03 最小実装（RuntimeWorldAuthority.h のみ・CMake 変更なし）
   ↓
ND-04 targeted CW-8 tests + 既存 3 suite + 全体 CTest
   ↓
ND-06 BuildError retry contract audit → CR-α（保留継続）
```
