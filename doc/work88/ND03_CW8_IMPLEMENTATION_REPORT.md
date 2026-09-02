# ND-03 — CW-8 PublishedWorldObservation Minimal Implementation（Work Report）

```text
ND-03 — CW-8 PublishedWorldObservation Minimal Implementation

Production source: 1 file changed（src/audioengine/RuntimeWorldAuthority.h・+60 / −0・純追加）
Test source: 1 file changed（src/tests/ISRSemanticValidationTests.cpp・+162 / −0・純追加）
CMake: 0
Build: NOT RUN
CTest: NOT RUN
stress: 0

baseline:
ConvoPeq.md Generated 2026-09-01 16:28:11
（※ 実装後の snapshot 鮮度: --check が STALE / NEWER_SRC_COUNT=2 を正しく報告 — 対象は本変更 2 ファイル。
   再生成は ND-04 gate で実施する。D160 ツールの動作確認を兼ねた。）
```

## 総合判定

> ## **ND-03 完了 — 契約どおり最小 diff（2 ファイル・純追加のみ）/ forbidden changes 0 / ND-04 GO**

---

## 1. production diff 全体（RuntimeWorldAuthority.h・+60 行）

| 区間 | 内容 |
|---|---|
| namespace 冒頭（PendingPublishRegistry の後） | `PublishedWorldObservation` class（CW-8 lifetime contract コメント全文付き）+37 行 |
| read API 群（`observePublishedWorld()` の直後） | `observePublishedObservation(const ReadToken&)` factory +24 行 |

既存行への変更（削除/書換）: **0 行**。既存 3 read API（`observePublishedWorld` / `consumeWorldHandle` ×2 / `acquireReadToken`）・`publish()`・`clearPublishedRuntimeSnapshotsNonRt`・`ReadToken` は 1 文字も触れていない（numstat 0 削除で構造的に保証）。

## 2. test diff 全体（ISRSemanticValidationTests.cpp・+162 行）

- `testPublishIntentQueueFullBackpressure()` の直後に `testCW8_PublishedWorldObservation()` を追加（既存 `testRuntimeWorldAuthorityAdapter` と同一ハーネスパターン）。
- `main()` の B3 backpressure テストの直後に 1 行で登録（throw 型・既存規約どおり）。

## 3. PublishedWorldObservation の最終型

```cpp
class PublishedWorldObservation
{
public:
    PublishedWorldObservation(const PublishedWorldObservation&) noexcept = default;
    PublishedWorldObservation& operator=(const PublishedWorldObservation&) noexcept = default;
    [[nodiscard]] const RuntimeState* world() const noexcept;
    [[nodiscard]] const PublicationSemantic* identity() const noexcept;
private:
    friend class RuntimeWorldAuthority;
    PublishedWorldObservation(const RuntimeState*, const PublicationSemantic*) noexcept;
    const RuntimeState* world_ = nullptr;
    const PublicationSemantic* identity_ = nullptr;
};
```

## 4. constructor / access control

- ctor: **private**（friend = `RuntimeWorldAuthority` のみ）。copy ctor/assign は `= default`（既存 observation の複製は可・独立 pair 構築には使えない）。
- data member: **private**（public data member にすると aggregate init が可能になるため）。accessor `world()` / `identity()` のみ公開。
- header に lifetime contract（Ownership none / Lifetime = 既存 reader・EBR 契約 / Mutation none / Publication 不参加 / Retire 不参加 / Thread = 既存 read path と同等 / ReadToken 拡張なし）をコメント固定。

## 5. factory の read topology

```text
runtimeStore_.observe()   [1 回 acquire load — 既存 API と同一 physical source]
        ↓ 単一方向
world（null なら {nullptr, nullptr} を返して終了 — nullptr deref は発生しない）
        ↓
&world->publication       （identity は同一オブジェクト内部 pointer）
```

- 二段 read / 別 atomic / tokenless overload: **なし**。
- ★ 実装上の決定 1 件（ND-02 からの確認済み逸脱はなし・実装上の必要措置）: `RuntimeState` は本ヘッダでは前方宣言のみ（AudioEngine.h 循環回避・ヘッダ先頭コメントの既存制約）のため、`&world->publication` を含む factory 本体は **member function template**（`template <typename StateT = RuntimeState>` + 完全型要求 static_assert）とし、呼び出し点（AudioEngine.h 可視 TU・既存 caller はすべて該当）で遅延 instantiation する。ヘッダ本体では compile されない。既存 caller への影響 0（本 factory は新規 API のため既存 instantiation に干渉しない）。

## 6. nullptr semantics

- 未 publish（Store 初期値）: factory は `world == nullptr` を検査して `{nullptr, nullptr}` を返す。`&world->publication` は評価されない。
- shutdown clear（null swap）: 同一経路で null が観測される。テストで request 未呼出 no-op（nullptr 返却・current 不変）→ request 後（null swap → 破棄）→ `{nullptr, nullptr}` 観測までを動作確認済み（コード審査レベル）。

## 7. T-CW8 実装内容

| Test | 実装 |
|---|---|
| **T-CW8-1/2/3 統合**（runtime） | 未 publish null 観測 → `createForBuilder` + `sealRecursively()`（PR-5 本番順序の再現）→ `publish()` ×2（metadata {1,1,1} → {2,2,2}・monotonicity 満たす）→ 各 observation で `obs.identity() == &obs.world()->publication`（address identity）・`obsN.world() == oldN1`（swap 戻り値と観測の一致）・pair の非同一性（world も identity address も別）・旧 observation の凍結・deterministic observe→publish→observe 反復での pair integrity |
| **T-CW8-4**（black-box） | bake 値 == observed `world->publication` == observed identity（sequenceId/epoch/mappedRuntimeGeneration を metadata と照合）。production hook は追加していない |
| **T-CW8-6** | `is_trivially_copyable_v` / `is_nothrow_copy_constructible_v` / runtime copy 複製が同一 pair を指すこと。**trivially copyable を lifetime 証明とするコメントはしていない**（ownership 表明として明記） |
| **T-CW8-7** | `!is_aggregate_v`（brace/designed init 遮断）/ `!is_default_constructible_v` / `!is_constructible_v<Obs, const RuntimeState*, const PublicationSemantic*>`（テスト TU は非 friend → 独立 pair 構築不能）を static_assert で固定。test code への friend 追加は行っていない（diff で 0 件確認） |
| shutdown clear 追補 | null swap 後の `{nullptr, nullptr}` 観測（CW-8 null semantics） |

破棄責務: テストは Bridge tail を呼ばないため、`publish()` が返した oldWorld（N）を production deleter と同一契約（`unseal()` + `~RuntimeState()` + `convo::aligned_free`）で明示破棄。shutdown clear で残存最新 world も回収し、最終観測が null であることを確認（リークなし）。

## 8. forbidden changes が発生していないこと（diff audit 実測）

| 項目 | 実測 |
|---|---|
| 変更ファイル | `RuntimeWorldAuthority.h` + `ISRSemanticValidationTests.cpp` の **2 件のみ**（numstat 実測） |
| RuntimeStore 変更 | **0**（`src/core/RuntimeStore.h` diff 空） |
| Coordinator / PublishExecutor / Retire / EBR / Shutdown 変更 | **0**（対象 8 ファイル diff 空） |
| CMake 変更 | **0** |
| 新 atomic | **0**（production diff 内 `atomic` 追加 0 件） |
| `currentWorld_` 混入 | **0**（diff 内 0 件・残存 11 件はすべて既存の歴史コメント） |
| 既存 read API の意味変更 | **0**（純追加・既存メソッド非改変） |
| `PublishedWorldObservation` の定義箇所 | RuntimeWorldAuthority.h + テスト TU の 2 ファイルのみ（19 hits 実測） |
| `runtimeStore_.observe()` 呼び出し | 7 件（既存 6 + 新 factory 1・全て同一単一 source） |

## 9. ND-03 contract compliance

ND-02 §3〜§5 の仕様（型設計 / lifetime contract / delegation topology）+ 本ターン指示の修正点（nullptr case 明示 / `is_constructible` の test TU 評価としての扱い / trivially copyable コメント規則 / CMake 不変）にすべて適合。逸脱は **member function template 形式 1 件のみ** — これは AudioEngine.h 循環回避の既存制約による構造上の必要措置であり、呼び出し構文・契約は ND-02 どおり（報告書 §5 に記録）。

## 10. ND-04 GO / NO-GO

> **GO** — diff audit・grep 再確認・compile-level inspection すべて問題なし。
> 次工程: **ND-04**（snapshot 再生成 → Debug/Release targeted CW-8 tests + 既存 3 suite + 全体 CTest）。
> CR-α（BuildError retry）は CW-8 closure（ND-04 PASS）まで保留継続。
