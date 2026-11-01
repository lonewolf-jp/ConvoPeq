# B-2 実装時確認 3 点の事前調査報告（2026-11-01 第 8 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 詳細: `doc/residual-20261101/B2-IMPL-READINESS-20261101.md`

## 1. `deferDestroyRetry` の型変更（`void*` → `DSPCore*`）

**DSPCore 破棄は「唯一の漏斗」であり、コードベースに明記されています。**

> `destroyDSPCoreNodeRetryable()` is the ONLY place that destroys a DSPCore /
> `DSPLifetimeManager` hardcodes `&AudioEngine::destroyDSPCoreNode` as the deleter
> （`DeferredPublicationTestAccess.h:489-499`、`AudioEngine.h:4708` にも同旨）

| 利用箇所 | 実測 |
|---|---|
| deleter 登録 | `DSPLifetimeManager.cpp:59/122/156` |
| 直接呼び出し | `RebuildDispatch.cpp:999/1075/1157`、`DSPLifetimeManager.cpp:182`、`DeferredPublicationTestAccess.h:102` |
| `deferDestroyRetry` の呼び出し元 | **1 箇所のみ**（`Threading.cpp:47`） |

**型変更で触るのは 5 箇所**です（署名・実装・唯一の呼び出し・vector 型・走査）。
**破棄経路（`~DSPCore` / `aligned_free`）は変更しません。** 再登録経路も同じです（重複防止は既存の同一性検査）。

**入口の `void*` は残ります**が、これは `void(void*)` の関数ポインタ契約によるもので、
**漏斗不変条件**（DSPCore 以外は渡らない）により正当化されます。teardown の detach は**型付き vector** を
走査するためキャストを伴いません。

## 2. `sink_` の atomic 化とアクセス順序

| 操作 | スレッド | 順序 |
|---|---|---|
| bind | NonRT（`prepare`） | **world の publish より前** |
| RT 読み | RT | publish の **acquire** 後（HB） |
| detach | NonRT（teardown） | **RT join 後**（並行なし） |

**HB により素のポインタでも正しい**が、将来の誤用に備え **`std::atomic<DiagSink*>`（load acquire / store release）を推奨**。
**「x64 だからコストなし」は確定仕様にしない** — 実装時に対象ビルド環境で
`std::atomic<DiagSink*>::is_always_lock_free` を評価して記録します（（Owner 指摘））。
atomic はポインタの一貫性のみを保証し、sink の内容は sink 自身の atomic が保護します。
`DSPCore` には既に **14 個の atomic メンバ**があり、同方針が既に確立しています。

## 3. `sizeof` と ODR

| 項目 | 実測 |
|---|---|
| 追加メンバ | `uint64_t`(8) ＋ `std::atomic<DiagSink*>`(8) ＝ **名目 +16 B**（実測は実装時。パディングで 0〜16 の幅） |
| `sizeof(DSPCore)` への依存 | **1 箇所のみ**（`V11EngineRetireProtectionTests.cpp:3400`）。`note=informational_only` の**表示のみ**で固定期待値ではない |
| 割り当て | `aligned_make_unique`（サイズ依存のレイアウト計算なし） |
| ODR | 定義は `AudioEngine.h` の 1 箇所。マクロは**ターゲット単位**（`CMakeLists.txt:1634`/`:1646`、ハーネス `:2291`）で、ファイル単位の上書きは無い |
| **診断 ON/OFF のサイズ** | **一致しない**（`#if` 内に非静的メンバ 3 個: `AudioEngine.h:1086-1088`）。要求は「構成を変えない」ではなく **「1 バイナリ内で単一値」＋「構成正ごとに記録」** |

**`offsetof` は採用しません（ただし「使用禁止」ではなく「適格性が未確定」）**:
`DSPCore` は非静的メンバに `ConvolverProcessor`（`:894`）と `EQProcessor`（`:895`）を持ち、
両者は `juce::ChangeBroadcaster` 継承（`src/ConvolverProcessor.h:62`、`src/eqprocessor/EQProcessor.h:147`）、
その基底は `virtual ~ChangeBroadcaster();`（`JUCE/.../juce_ChangeBroadcaster.h:54`）を持つため、
**両クラス自身が非標準レイアウト**です。C++20 [class.prop]/3.1 は「非標準レイアウトクラス型の
非静的データメンバを持たないこと」を要求するので、**規格上は `DSPCore` も非標準レイアウトと導けます**。

ただし `std::is_standard_layout_v` による**実測は未了**（ビルド未承認）であり、
Owner 指示に従い本断定は**未確定として扱います**。

> 前版の「多態型メンバを持つから非標準レイアウト」という書き方は理由として不正確でした。
> 正しくは「メンバの**型が**非標準レイアウトだから」([class.prop]/3.1) です。

**実装時の検証**: 構成正（ON / OFF）ごとに `sizeof` を記録し、メンバ位置は
**実インスタンスのアドレス差**で測ります（標準レイアウト性に依存しない）。
`static_assert` は**補助**（サイズ一致の証明にはしない）。
`offsetof` は、実測で標準レイアウトと確認できた場合でも採用せず、測定方法を 1 通りに保ちます。

**測定仕様（§5.2.4 に固定）**: 対象は M1 `sizeof(DSPCore)`、M2 `ownerId_` のオフセット、
M3 `sink_` のオフセット、M4 `is_standard_layout_v`、M5 `is_always_lock_free`、M6 構成マクロ値、
M7 コンパイラ・ビルド構成です。方法は実在インスタンス（測定専用に 1 個確保し publish しない）に対する
`reinterpret_cast<const char*>(&obj.member) - reinterpret_cast<const char*>(&obj)`、NonRT 実行、
クラスメンバを追加しないクラス外プローブです。**記録のみで合否は判定しません**。

## 4. 評価

**この 3 点は「実装時の確認事項」として残りますが、設計上の障害ではありません。**

## 4.1 Owner 指摘 2 点の反映と、実測による訂正（2026-11-01 22:20 / 22:45 更新）

追加調査は行わず、指摘 2 点の反映に必要な範囲だけを読み取りで確認しました。その結果、
**以前の記述 2 件が事実と異なっていた**ため訂正します。

| # | 以前の記述 | 実測 | 扱い |
|---|---|---|---|
| 1 | 「多態型メンバを持つ型は標準レイアウトになれない」 | 規格の要件は「**非標準レイアウト型の**非静的データメンバを持たない」（[class.prop]/3.1）。多態性そのものが伝播するわけではない | **理由の書き方を訂正**。結論（`DSPCore` は非標準レイアウトと導ける）は 2 段の導出で維持するが、**実測未了のため未確定**として扱う |
| 2 | 「`#if` でメンバを出し分けない」 | `AudioEngine.h:635-1093` の `#if` 内に**非静的メンバが 3 個**（`:1086-1088`）→ **ON/OFF で `sizeof` は一致しない** | 要求を「同一バイナリ内で単一のマクロ値」＋「構成正ごとに記録」へ変更 |

`is_always_lock_free` は**実装時に評価して記録**する項目として §5.1 に明記しました
（「x64 でコストなし」を確定仕様から外しました）。

## 5. 遵守事項

製品コード・テスト・CI 定義変更 **なし**（追跡差分 0）／build・test・debugger・replay **未実行**／
U5 FAIL・既定 OFF、U3/U4 HOLD を維持／B-2 実装 **HOLD**。

### 5.1 残課題の分解（実装承認で解除されるもの／されないもの）

**「残るのは実装承認のみ」とは言えない。** 実装承認で解除されるのは B-2 と A2→A3-a だけで、
以下は別扱いのまま残る。

| # | 項目 | 現在 | 解除条件 | 実装承認で解除されるか |
|---|---|---|---|---|
| 1 | B-2 実装 | **HOLD** | Owner の明示的な実装承認 | する（承認時） |
| 2 | A2→A3-a 実装 | **HOLD** | Owner の明示的な実装承認 | する（承認時） |
| 3 | build / test / debugger / replay | 未承認 | 別途の実行許可 | **しない** |
| 4 | CI 定義変更 | 未承認 | 別途の許可 | **しない** |
| 5 | Git 操作 | 未承認 | 別途の許可 | **しない** |
| 6 | U5 | **FAIL・既定 OFF** | 技術的解決（未着手） | **しない**（判定は変わらない） |
| 7 | U3 / U4 | **HOLD** | 別途の判断 | **しない** |
| 8 | M1〜M7（実装時の測定・記録） | 未測定 | B-2 実装後の測定 | 間接（実装後に実施） |
