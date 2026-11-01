# B-2 実装時確認 3 点の事前調査（2026-11-01）

- 状態: **調査・設計資料のみ。製品コード・テスト・CI 定義は未変更。build / test 未実行**
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner が示した「実装時確認 3 点」を、実装前に読み取り調査で詰める

## 1. `deferDestroyRetry` の引数型変更の影響範囲

### 1.1 DSPCore 破棄の「唯一の漏斗」がコードベースに明記されている

`src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h:489-499`:

> `destroyDSPCoreNodeRetryable()` is the ONLY place that destroys a DSPCore
> / `DSPLifetimeManager` hardcodes `&AudioEngine::destroyDSPCoreNode` as the deleter

`AudioEngine.h:4708` にも「DSPCore 解放の唯一の漏斗」と明記されている。

### 1.2 全利用箇所（実測）

| 種別 | 箇所 |
|---|---|
| deleter として登録 | `DSPLifetimeManager.cpp:59/122/156`（`&AudioEngine::destroyDSPCoreNode`） |
| 直接呼び出し | `AudioEngine.RebuildDispatch.cpp:999/1075/1157`、`DSPLifetimeManager.cpp:182`、`DeferredPublicationTestAccess.h:102` |
| テスト（Retryable を直接） | `V11EngineRetireProtectionTests.cpp:839/1228/1319`、`DeferredPublicationTestAccess.h:499` |

### 1.3 `deferDestroyRetry` の呼び出し元は 1 箇所だけ

`AudioEngine.Threading.cpp:47`（`destroyDSPCoreNodeRetryable` の pin 残留分岐）のみ。

### 1.4 型変更（`void*` → `DSPCore*`）で触る範囲

| # | 対象 | 変更 |
|---|---|---|
| 1 | `deferDestroyRetry` の署名（`AudioEngine.h:5233`） | `void*` → `DSPCore*` |
| 2 | `deferDestroyRetry` の実装（`Threading.cpp:70-86`） | 引数型のみ（比較・push_back は同じ） |
| 3 | 唯一の呼び出し（`Threading.cpp:47`） | 型付きポインタを渡す |
| 4 | `deferredDestroy_`（`AudioEngine.h:5254`） | `std::vector<void*>` → `std::vector<DSPCore*>` |
| 5 | `retryDeferredDestroys` の走査（`Threading.cpp:118`） | `destroyDSPCoreNodeRetryable(p)` はそのまま（`DSPCore*` → `void*` の暗黙変換） |

**閉じる範囲は 5 箇所**であり、破棄経路（`core->~DSPCore(); aligned_free(core)`）は変更しない。

### 1.5 再登録経路（自己再登録）

```
retryDeferredDestroys → destroyDSPCoreNodeRetryable(core)
   ├─ pin なし → 破棄
   └─ pin あり → deferDestroyRetry(core)（再登録）
```

型付きにしても経路は同じ。**重複登録は既存の同一性検査（`:75-79`）で防止**される。

### 1.6 残る `void*` キャストについて（Owner の禁止事項との整合）

`destroyDSPCoreNodeRetryable(void* p)` は **`void(void*)` の関数ポインタとして登録される**ため、
入口の `void*` は残らざるを得ない。ただし:

- **漏斗不変条件**（唯一の破棄経路）により、この関数に DSPCore 以外は渡らない（コード内で文書化済み・§1.1）。
- したがって入口キャストは「未検証の `void*` を任意型へキャスト」ではなく、**契約が文書化された唯一の入口**である。
- teardown の detach は**型付き vector** を走査するため、**キャストを伴わない**。

## 2. `sink_` の atomic 化とアクセス順序

### 2.1 同期契約

| 操作 | スレッド | 順序 |
|---|---|---|
| bind（`sink_.store`） | **NonRT**（rebuild または message の `prepare`） | **world の publish（release）より前** |
| RT 読み（`sink_.load`） | **RT**（audio callback） | publish の **acquire** 後（HB 成立） |
| detach（`sink_.store(nullptr)`） | **NonRT**（teardown の走査 / コア dtor） | **RT join 後**（並行なし） |

### 2.2 素のポインタで足りるか / atomic にするか

- **HB により素のポインタでも正しい**（bind は publish 前、detach は join 後）。
- ただし将来の誤用（publish 後の書き換え等）に備え、**`std::atomic<DiagSink*>`（load: acquire / store: release）を推奨**。
  x64 では追加コストがない。
- **atomic はポインタの一貫性のみ**を保証する。sink の内容（リング）は sink 自身の atomic で保護される（既存実装）。

### 2.3 既存の前例

`DSPCore` には既に **14 個の `std::atomic` メンバ**がある（`AudioEngine.h:633-1300` 内）。
RT が読む値も NonRT が書く値も atomic で扱う方針が既に確立している。

## 3. `sizeof` と ODR

### 3.1 追加メンバと見積り

| メンバ | 型 | 想定サイズ（x64） |
|---|---|---|
| `ownerId_` | `std::uint64_t` | 8 B |
| `sink_` | `std::atomic<DiagSink*>` | 8 B（ポインタサイズ） |
| 合計 | — | **名目 +16 B** |

**実測は実装時**。既存メンバのアラインメント／パディングにより、実際の増加は **0〜16 B** の幅があり得る。
「+16 B」は**見積り**であり確定仕様ではない。

### 3.2 `sizeof(DSPCore)` に依存するコードは 1 箇所のみ（実測）

`V11EngineRetireProtectionTests.cpp:3400`:

```cpp
" sizeof_dspcore=%llu ... note=informational_only\n",
static_cast<unsigned long long>(sizeof(AudioEngine::DSPCore)),
```

**情報表示のみで、固定期待値との比較はしていない**（`note=informational_only`）。
したがって 16 B の増加でテストが壊れることはない。

割り当ても `aligned_make_unique`（`RuntimeBuilder.cpp:469`、`PrepareToPlay.cpp:268`）であり、
サイズ依存のレイアウト計算は存在しない。

### 3.3 ODR 整合性（2026-11-01 22:20 実測により訂正）

**訂正**: 「追加メンバを診断 ON/OFF で出し分けない」は現行コードの実態と一致しない。
`DSPCore` は既に `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`（`AudioEngine.h:635`〜`:1093`）の
**内側に非静的データメンバを持つ**。

| メンバ | 行 | 種別 |
|---|---|---|
| `std::atomic<std::uint64_t> diagGeneration` | `AudioEngine.h:1086` | 非静的 |
| `DiagFootprint diagFootprint` | `:1087` | 非静的 |
| `bool diagFootprintCaptured` | `:1088` | 非静的 |

したがって **`sizeof(DSPCore)` は診断 ON と OFF で一致しない**（OFF では上記 3 メンバが除外される）。
「両構成で一致すること」を要求にすると、現行コードに対して成立しない要求になる。

**正しい要求**:

| 方針 | 内容 |
|---|---|
| 単一定義 | `DSPCore` の定義は `AudioEngine.h` の 1 箇所のみ |
| 1 バイナリ内の一貫性 | マクロは**ターゲット単位**で定義される（`CMakeLists.txt:1634` の `target_compile_definitions(ConvoPeq PRIVATE ...)` 内 `:1646`、ハーネス側 `:2291`）。ファイル単位の上書きは無い → **同一バイナリ内で ON/OFF が混在しない** |
| 既定値 | 未定義時は `DiagnosticsConfig.h:68-70` で `0`。CMake は `option(... OFF)`（`CMakeLists.txt:129`） |
| 記録 | **構成正（ON / OFF）ごとに** `sizeof` と配置の実測値を記録する。**構成間の一致は要求しない** |
| `static_assert` | **補助**。サイズの一致ではなく「同一構成内での前提」を検査する用途に限る |

**検証コード（実装フェーズ・`offsetof` を使わない）**:

```cpp
// DSPCore の標準レイアウト性は未確定（§5.2）。offsetof は使わずアドレス差で測る。
// 実インスタンスのアドレス差は標準レイアウト要件に依存しない。
const auto base = reinterpret_cast<const char*>(&core);
const auto offOwner = static_cast<std::size_t>(
    reinterpret_cast<const char*>(&core.ownerId_) - base);
const auto offSink = static_cast<std::size_t>(
    reinterpret_cast<const char*>(&core.sink_) - base);
std::printf("sizeof(DSPCore)=%zu off_owner=%zu off_sink=%zu diag=%d\n",
            sizeof(AudioEngine::DSPCore), offOwner, offSink,
            CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS);
```

## 4. まとめ

| # | 確認項目 | 結果 |
|---|---|---|
| 1 | `deferDestroyRetry` の型変更 | **5 箇所に閉じる**。破棄経路は不変。入口 `void*` は「唯一の漏斗」契約で正当化される |
| 2 | `sink_` の atomic と順序 | **HB で足りるが atomic を推奨**。load acquire / store release。**「x64 でコストなし」は前提にせず `is_always_lock_free` を実測**（§5.1） |
| 3 | `sizeof` / ODR | **固定期待値への依存は 1 箇所のみ（情報表示）**。実測は実装時。ODR は単一定義＋ターゲット単位のマクロ定義で担保（**`#if` は既に使われており、ON/OFF でサイズは一致しない** → §3.3 訂正） |

**この 3 点は「実装時の確認事項」として残るが、設計上の障害ではない。**

## 5. Owner 指摘による追記（2026-11-01 22:05）

実装時に必ず確認する事項として、以下を追加する。

### 5.1 `sink_` の atomic について（「x64 でコストなし」を確定仕様にしない）

| 確認項目 | 方法 |
|---|---|
| ロックフリーであること | **`std::atomic<DiagSink*>::is_always_lock_free`** を対象ビルド環境で確認する（`static_assert` 可）。「x64 だからコストなし」を前提にしない |
| acquire が対応する release を観測すること | bind は **world publish の release より前**に store する。RT の load は publish の **acquire 後**に実行される。この対を実装コメントと `assert`／計装で確認する |
| すべての RT 読み取り経路が publish 後であること | `process()` 系の全入口（14 push サイト）が publish 済みの world からのみ呼ばれることを確認する |
| atomic は sink 本体の寿命管理の代替ではないこと | 寿命は「エンジン所有＋teardown の detach」で担保する。atomic は**ポインタ値の一貫性のみ** |

### 5.2 `offsetof` の適格性は未確定（測定方法はアドレス差に統一）

**状態: `DSPCore` が標準レイアウト型かは未確定。** 規格要件（§5.2.1）に照らすと「非標準レイアウト」と導けるが、
`std::is_standard_layout_v` による実測はビルドが未承認のため未了である。
**Owner 判断により、実測が完了するまで本資料は断定しない。**

#### 5.2.1 規格の要件（引用）

C++20（N4868）[class.prop]/3 の標準レイアウトクラスの要件:

1. (3.1) 非標準レイアウトクラス型（またはその配列）または参照の**非静的データメンバを持たない**
2. (3.2) 仮想関数および仮想基底クラスを持たない
3. (3.3) すべての非静的データメンバのアクセス制御が同一
4. (3.4) 非標準レイアウトの基底クラスを持たない
5. (3.5) 同一型の基底クラス部分オブジェクトを高々 1 つ持つ
6. (3.6) 非静的データメンバとビットフィールドがすべて同一クラスで最初に宣言される
7. (3.7) 基底クラスに M(S) の型が無い

現行ドラフト [`class.prop]/2.1` も同旨。**したがって「多態型のメンバを持つから」ではなく、
「非標準レイアウト型の非静的メンバを持つから」(3.1) で導くのが正しい。**

> 前版の記述「多態型（仮想関数を持つ型）のメンバを持つ `DSPCore` は標準レイアウト型になれない」は
> **理由の書き方が不正確**だった（多態性そのものが伝播するかのように読める）。結論の導出は次の二段:

#### 5.2.2 `DSPCore` の各要件の確認（読み取り）

| 要件 | `DSPCore` の実測 | 判定 |
|---|---|---|
| (3.2) 仮想関数・仮想基底 | `struct DSPCore`（`AudioEngine.h:633`）に基底なし、自クラスの `virtual` 宣言 **0 件** | 満たす |
| (3.3) アクセス制御の同一性 | `private:`（`:1095`）以降は**静的メンバのみ**（`:1096-1099`）→ 非静的メンバはすべて public | 満たす |
| (3.4)(3.5)(3.6)(3.7) | 基底クラスなし | 満たす |
| **(3.1) 非静的メンバの型** | `ConvolverProcessor convolver;`（`:894`）、`EQProcessor eq;`（`:895`） | **要確認** |

`ConvolverProcessor` / `EQProcessor` は `juce::ChangeBroadcaster` を継承し
（`src/ConvolverProcessor.h:62`、`src/eqprocessor/EQProcessor.h:147`）、その基底は
`virtual ~ChangeBroadcaster();`（`JUCE/modules/juce_events/broadcasters/juce_ChangeBroadcaster.h:54`）を持つ。
仮想関数を継承するため、**両クラス自身が (3.2) を満たさず非標準レイアウト**となる。
その結果 `DSPCore` は (3.1) を満たさず、規格要件に照らすと非標準レイアウトと導ける。

ただしこれは**読み取りによる導出**であり、実測（`std::is_standard_layout_v`）は未了である。
**Owner はこの導出を未確定と判断しており、本資料もそれに従って断定しない。**

#### 5.2.3 測定方法（判定結果に依存させない）

| 目的 | 方法 |
|---|---|
| メンバのオフセット実測 | **実インスタンスのアドレス差**（`reinterpret_cast<const char*>(&obj.member) - reinterpret_cast<const char*>(&obj)`）。標準レイアウト性に依存しない |
| 型特性の記録 | 実装時に `std::is_standard_layout_v<AudioEngine::DSPCore>` を**評価して記録**する。`static_assert` で固定しない |
| `offsetof` | **採用しない**（適格性が未確定のため）。実測で標準レイアウトと確認できた場合でも、測定方法はアドレス差に統一して 1 通りに保つ |

**設計上の影響: なし。** どちらの判定でも測定方法はアドレス差であり、実装内容は変わらない。

#### 5.2.4 測定仕様（対象・方法・時期・記録項目）

Owner 指示「測定対象と方法を明記する」に対応し、実装時に実施する測定を次のとおり固定する。

**測定対象（メンバ）**

| # | 対象 | 種別 | 備考 |
|---|---|---|---|
| M1 | `sizeof(AudioEngine::DSPCore)` | 型 | 構成正ごとに記録 |
| M2 | `ownerId_` のオフセット | 追加予定メンバ | アドレス差で算出 |
| M3 | `sink_` のオフセット | 追加予定メンバ | アドレス差で算出 |
| M4 | `std::is_standard_layout_v<AudioEngine::DSPCore>` | 型特性 | 真偽をそのまま記録（判定しない） |
| M5 | `std::atomic<DiagSink*>::is_always_lock_free` | 型特性 | 真偽をそのまま記録 |
| M6 | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の値 | 構成正 | `0` / `1` |
| M7 | コンパイラ識別子・版、ビルド構成（Debug/Release）、対象 triple | 環境 | 記録の再現性のため |

**測定方法**

```cpp
// 実インスタンスが必要（nullptr に対する差分は不可）。
// 測定専用インスタンスを aligned_make_unique で 1 個確保し、publish しない。
auto probe = convo::aligned_make_unique<AudioEngine::DSPCore>();
const char* base = reinterpret_cast<const char*>(probe.get());
const std::size_t offOwner = static_cast<std::size_t>(
    reinterpret_cast<const char*>(&probe->ownerId_) - base);
const std::size_t offSink  = static_cast<std::size_t>(
    reinterpret_cast<const char*>(&probe->sink_) - base);
```

| 項目 | 規定 |
|---|---|
| 計測式 | `reinterpret_cast<const char*>(&obj.member) - reinterpret_cast<const char*>(&obj)` |
| インスタンス | 実在する `DSPCore`（測定専用に 1 個確保し、world へ publish しない） |
| 実行スレッド | NonRT のみ。**RT パスに計測コードを置かない** |
| プローブの置き場所 | クラス外の関数（テスト／診断用 TU）。**クラスメンバを追加しない**ためレイアウトに影響しない |
| アクセス | `ownerId_` / `sink_` は `private:`（`:1095`）より前＝public に置く前提。public でない場合はテスト用アクセサ経由とし、アクセサ自体はレイアウトに影響しない |
| 実施回数 | 構成正（診断 ON / OFF）ごとに 1 回以上 |
| `offsetof` | 使用しない（M2/M3 は上記の式で算出） |

**合否の扱い**: 本測定は**記録のみ**で合否を判定しない（標準レイアウト性の真偽、サイズの大小を問わない）。
ただし M1〜M7 の記録が欠けた場合は「実装時確認が未完了」として扱う。

### 5.3 ODR 確認の維持事項（Owner 指示の再掲・実測で訂正）

- `DSPCore` の定義は単一定義（`AudioEngine.h:633`）。
- **診断 ON/OFF でメンバ構成は変わる**（`#if` 内に非静的メンバが 3 個: `:1086-1088`）。
  したがって要求は「構成を変えない」ではなく、**「1 バイナリ内で単一のマクロ値」**（ターゲット単位定義で担保）と
  **「構成正ごとにサイズ・配置を記録する」**である。
- `static_assert` は補助的な検査として扱う（サイズ一致の証明にはしない）。
- **サイズ・配置の実測結果とコンパイル構成を記録する。**

## 6. 状態

状態の正本は `STATUS-20261101.md`。B-2 実装は **HOLD**、A2→A3-a 実装は **HOLD**、
U5 は **FAIL・既定 OFF**、U3/U4 は **HOLD**。build / test / debugger / replay、CI 定義変更、
Git 操作はいずれも未承認。実装承認で解除されるのは B-2 と A2→A3-a の実装 HOLD のみで、
U5・U3/U4・各種承認は別扱いのまま残る。
