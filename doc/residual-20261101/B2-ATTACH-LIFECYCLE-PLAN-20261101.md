# B-2 attach ライフサイクル確定資料 + 診断ON ASan 検証計画（2026-11-01）

- 状態: **調査・計画のみ。製品コード・テスト・CI 定義は変更していない。build / test も未実行**
- 基準: `main` = `origin/main` = `6bb7c8d5`（ソースは `fd92c6e7` と同一）
- 目的: Owner 指示の B-2-1（attach 位置・冪等性）、B-2-2（DSPCore 生成順序）、
  B-2-3（診断 ON ASan のローカル検証計画）に回答する

## 0. 結論

| 項目 | 決定 |
|---|---|
| `DSPCore` の attach 位置 | **`DSPCore::prepare(..., AudioEngine* owner)` の内部**（owner が確定し、publish より前） |
| エンジン直下プロセッサの attach 位置 | **`AudioEngine::initialize()`**（現行のグローバル設定箇所と同じ位置。RT 開始前） |
| 冪等性 | `attach` は `store(sink, release)` であり**冪等**。再 prepare でも同じ sink を再設定するだけ |
| 非所有インスタンス | **`sink_` は nullptr のまま**（下記 §4 で全経路を列挙） |
| 別エンジンの sink 誤設定 | **構造的に不可能**（§5: エンジン自身が自分の sink を bind する API にする） |
| detach | DSPCore の破棄時に自分の sink を clear、エンジン破棄時に直下プロセッサを clear。さらに**join が一次保証** |

## 1. 生成・破棄経路の全数（実測）

### 1.1 `ConvolverProcessor` のインスタンス（製品コードは 2 箇所のみ）

| # | インスタンス | 宣言 | 生成タイミング | 破棄タイミング |
|---|---|---|---|---|
| C1 | `AudioEngine::DSPCore::convolver` | `AudioEngine.h:894`（`struct DSPCore` は `:633` から） | **DSPCore の生成ごと**（rebuild のたび） | DSPCore の破棄ごと（retire / shutdown） |
| C2 | `AudioEngine::uiConvolverProcessor` | `AudioEngine.h:2317` | エンジン生成時（メンバ） | エンジン破棄時 |

- `getConvolverProcessor()` は **C2 を返す**（`AudioEngine.h:1290`）。
- テストのローカル `ConvolverProcessor`（23 個）は**エンジンに所有されない**。

### 1.2 `DSPCore` の生成経路（全数）

| # | 経路 | 生成 | prepare | sink の所有者 |
|---|---|---|---|---|
| D1 | **rebuild** | `RuntimeBuilder.cpp:469` `aligned_make_unique<DSPCore>()` | **`:474` `prepare(..., &engine)`** | エンジン（引数で渡る） |
| D2 | **placeholder**（prepareToPlay） | `PrepareToPlay.cpp:268` | `:272` `prepare(..., owner)` | エンジン |
| D3 | **テスト**（V11 等） | `V11EngineRetireProtectionTests.cpp:457` ほか | **prepare を呼ばない**（`registerDSPHandleForRuntime` で直接登録） | **なし → nullptr のまま** |

### 1.3 `AudioEngine::initialize()` の呼び出し（全数）

| 呼び出し元 | 回数 |
|---|---|
| `AudioEngineHarness.cpp:29`（`start()`） | エンジンごとに 1 回 |
| `MainWindow.cpp:279` | 起動時に 1 回 |

`initialize()` に再入ガードは無いが、**呼び出し側は 1 回だけ**呼ぶ。
（`prepareToPlay` は reconfigure で複数回呼ばれ得るため、attach を `prepareToPlay` に置くと
「毎回 attach」になる。**冪等なので害は無いが、置く理由も無い**）

## 2. B-2-1: attach 位置の決定と理由

### 2.1 決定

| 対象 | attach 位置 | 理由 |
|---|---|---|
| DSPCore の `convolver` / `eq` | **`DSPCore::prepare(owner, ...)` 内**（`owner != nullptr` のとき） | ① owner が**引数で確定**している ② prepare は **publish より前**に完了する（`RuntimeBuilder.cpp:474` は publish 前） ③ 生成（`:469`）から prepare までに RT は触れない |
| `uiConvolverProcessor` | **`AudioEngine::initialize()`**（現行 `setConvDiagBuffer` の位置） | エンジンのメンバは生成済みで、RT 開始前。`prepareToPlay` より前で十分 |
| 非所有インスタンス | **attach しない**（sink は nullptr のまま） | 所有エンジンが存在しない |

### 2.2 `Init` と `prepareToPlay` の比較（実経路からの判断）

| 観点 | `initialize()` | `prepareToPlay()` |
|---|---|---|
| 実行回数 | エンジンごとに 1 回 | reconfigure で**複数回**あり得る |
| RT 開始前か | **はい**（ハーネスは initialize → prepareToPlay → スレッド開始） | はい |
| DSPCore を網羅できるか | **できない**（コアは rebuild で後から生成される） | できない |
| 向き | エンジン直下（C2）に適する | 毎回呼ばれるため attach の置き場としては不適 |

**結論**: エンジン直下は `initialize()`、DSPCore は `prepare()`。**「Init か prepareToPlay か」の二択ではなく、
インスタンスの生成点で attach する**のが実経路に合う。

## 3. 冪等性

```cpp
void attachDiagSink(DiagSink* s) noexcept { sink_.store(s, std::memory_order_release); }
void detachDiagSink() noexcept           { sink_.store(nullptr, std::memory_order_release); }
```

- **同じ sink を何度 attach しても結果は同じ**（store は冪等）。
- `prepare` が複数回呼ばれても、`owner` が同じなら同じ sink を再設定するだけ。
- **異なる sink を attach できてしまうと危険**なので、API を §5 の形にして呼び出し元を限定する。

## 4. 非所有インスタンスの `sink_` が nullptr であることの全経路確認

| 経路 | `sink_` の初期値 | attach されるか | 根拠 |
|---|---|---|---|
| C1（DSPCore の convolver） | `nullptr` | `prepare(owner != nullptr)` のときだけ | §2.1 |
| C2（uiConvolverProcessor） | `nullptr` | `initialize()` で attach | §2.1 |
| D3（テストの DSPCore） | `nullptr` | **されない**（prepare を呼ばない。`V11EngineRetireProtectionTests.cpp:457`〜 は生成→`registerDSPHandleForRuntime` のみ） | 実測 |
| テストのローカル `ConvolverProcessor`（23 個） | `nullptr` | **されない**（エンジンが存在しない） | 実測 |
| `ConvolverProcessor` のコピー/ムーブ | 既定で `nullptr`（コピー禁止または明示） | — | 実装時に `= delete` を検討 |

**「非所有は nullptr」は、attach を「エンジンからのみ」呼べる API にすることで構造的に保証する（§5）。**

## 5. 別エンジンの DiagSink を誤って設定できない構造

**採用案**: `ConvolverProcessor` に `DiagSink*` を渡す公開 API を**作らない**。
代わりに **エンジンが自分の sink を自分のコアへ bind する**。

```cpp
// AudioEngine（private）
void bindDiagnostics(DSPCore& core) noexcept {   // ← エンジン自身が呼ぶ
    core.convolverRt().attachDiagSink(&diagSink_);
    core.eqRt().attachDiagSink(&diagSink_);
}
```

- `DSPCore::prepare(AudioEngine* owner, ...)` の内部で `owner->bindDiagnostics(*this)` を呼ぶ。
- `attachDiagSink` は `DSPCore` の friend か private にし、**外部から任意の sink を渡せない**。
- これにより「別エンジンの DiagSink を誤設定する」ことが**型・アクセス制御で不可能**になる。

**代替案（不採用）**: `attach(DiagSink&)` を公開し、呼び出し規約で正しさを担保する。
→ 呼び出し側の規律に依存するため、誤用を防げない。

## 6. detach と参照寿命の証明

### 6.1 一次保証（これが本体）

**P1（混線の排除）**: 各生産者は自分のエンジンの sink のみを持つ（§5）。グローバルが無いため、
別エンジンのバッファを参照する経路が存在しない。

**P2（非所有の排除）**: エンジンを持たないインスタンスは `sink_ == nullptr`（§4）。
`if (auto* s = sink_.load(acquire))` で**書き込み自体が発生しない**。

**P3（破棄前の静止）**: 生産者の実行は音声コールバック（またはメッセージスレッド）に限られ、
エンジン破棄前に **join / lock で静止**する（`B2-LIFETIME-PROOF-20261101.md` §1.4）。
したがって「バッファ破棄後に push する生産者」は存在しない。

**P4（防御）**: `DSPCore::~DSPCore()` で自分の sink を clear、`~AudioEngine` で直下プロセッサを clear。
仮に P3 が破れても、clear 後は null チェックで no-op になる。

### 6.2 「detach 後に取得済みポインタを使う」競合

```
producer: p = sink_.load(acquire)
engine:   detach
producer: p->push(...)
```

**P3 により、detach の時点で生産者は実行していない**（join/lock 済み）。
したがって load と use の間に他スレッドが入らない。
atomic 化だけでは P3 の代わりにならない（値の一貫性しか保証しない）。

### 6.3 残余（明示）

- `abandonEngine()`（テスト専用）はエンジンを**破棄せずに leak** する。この場合、コアの sink は
  有効なまま残るが、**RT が存在しない**ため push は起きない（メモリは leak する）。
- DSPCore がエンジンより長生きする経路（retire の遅延）は、`releaseResources` の drain で
  エンジン破棄前に回収される。**leak 以外では残らない**。

## 7. B-2-3: 診断 ON の ASan ローカル検証計画（CI 変更は未承認）

### 7.1 目的

現行の ASan 構成（`build-asan-local`）は `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` のため、
**今回のクラッシュ経路を一度も検証していない**。診断 ON の ASan 構成で、
(a) 現行コードの欠陥を ASan が検出できること、(b) B-2 適用後に検出されないことを確認する。

### 7.2 構成

| 項目 | 値 |
|---|---|
| ビルドディレクトリ（新規） | `build-asan-diag`（ローカルのみ。CI には追加しない） |
| ジェネレータ | 既存 ASan 構成に合わせる（`build-asan-local` の cache を踏襲） |
| 主要オプション | `-DENABLE_ASAN=ON` かつ `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON` |
| 構成 | RelWithDebInfo（既存 ASan 構成と同じ） |
| 対象ターゲット | `AudioEngineHarness`（51 テストを含む） |

### 7.3 実行と期待結果

| 段 | 実行 | 期待（B-2 適用前） | 期待（B-2 適用後） |
|---|---|---|---|
| S1 | `AudioEngineHarness` を 1 回 | **ASan が use-after-free / heap-buffer-overflow を報告する可能性**（再現率 8.3% のため、報告されないこともある） | 報告 0 |
| S2 | 同 10 回反復 | 再現すれば UAF のスタックが得られる（**原因の独立確認**） | 10/10 で報告 0 |
| S3 | `ConvolverStateRoundTripTests` 相当（スタンドアロン conv を含むテスト群） | グローバル経由の書き込みが起きる | **sink が nullptr のため書き込みなし** |
| S4 | 2 エンジン同時（該当テストがあれば） | 混線 | 各自のバッファのみ |

- S1/S2 で **ASan が UAF を報告すれば、今回の原因（解放済みバッファへの書き込み）が独立に裏付けられる**。
  これが本計画の最大の価値である（ダンプ 1 件に依存しない）。
- 検出されない場合も「発生率が低いため」であり、否定材料にはならない（8.3% の 10 回で約 58%）。

### 7.4 実行しないこと（今回）

- **build / test は Owner の禁止事項**のため、本計画は文書のみ。
- CI（GitHub Actions）への追加は**別途承認**が必要。本計画では対象外。

## 8. 実装時の変更対象（参考）

| ファイル | 変更 |
|---|---|
| `AudioEngine.h` | `DiagSink` 型、`diagSink_` メンバ、`bindDiagnostics(DSPCore&)`、`DSPCore` に `attachDiagSink/detachDiagSink` |
| `AudioEngine.Init.cpp:116-119` | グローバル設定 → `uiConvolverProcessor` への attach |
| `AudioEngine.Processing.DSPCoreLifecycle.cpp:57-69` | コンストラクタは attach しない（sink は nullptr のまま） |
| `RuntimeBuilder.cpp:474` / `PrepareToPlay.cpp:272` | `prepare(owner)` 内で `owner->bindDiagnostics(*this)` |
| `ConvolverProcessor.Runtime.cpp:85-104, 764-786, 849-870` | グローバル削除、`sink_` 参照へ |
| `DSPCoreFloat.cpp:56-90` / `DSPCoreDouble.cpp:16-45, 647` / `DSPCoreIO.cpp:435` | `logEqTime` 等に `DiagSink*` を渡す（グローバル削除） |
| `AudioEngine.CtorDtor.cpp`（dtor） | 直下プロセッサの detach（防御） |

## 9. 未確定（実装時に確定してよいもの）

1. `DSPCore` に `eqRt()` があるか（`convolverRt()` と同様のアクセサ）。無ければ追加。
2. `logEqTime` の引数追加が RT コストに与える影響（引数 1 個の追加のみ。測定は不要と判断）。
3. `DiagSink` を `AudioEngine` の private 入れ子にするか、別ヘッダの構造体にするか
   （アクセス制御の強さとインクルード都合のトレードオフ）。
