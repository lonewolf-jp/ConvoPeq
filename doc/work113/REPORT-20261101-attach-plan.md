# B-2 attach ライフサイクル確定報告（2026-11-01 第 4 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テスト・CI 定義は未変更。build / test / debugger / replay も未実行**
- 詳細: `doc/residual-20261101/B2-ATTACH-LIFECYCLE-PLAN-20261101.md`

## 1. attach 位置の決定（B-2-1）

**「`Init` か `prepareToPlay` か」の二択を捨て、インスタンスの生成点で attach する**のが実経路に合います。

| 対象 | attach 位置 | 根拠（実測） |
|---|---|---|
| DSPCore の `convolver` / `eq` | **`DSPCore::prepare(..., AudioEngine* owner)` の内部**（`owner != nullptr`） | owner が引数で確定（`RuntimeBuilder.cpp:480` は `&engine`、`PrepareToPlay.cpp:278` は `this`）。prepare は **publish より前**に完了する。生成（`:469`）から prepare まで RT は触れない |
| `uiConvolverProcessor` | **`AudioEngine::initialize()`**（現行のグローバル設定位置） | メンバは生成済みで RT 開始前。`prepareToPlay` は reconfigure で複数回呼ばれ得るため置き場として不適 |

`initialize()` はエンジンごとに **1 回**（`AudioEngineHarness.cpp:29`、`MainWindow.cpp:279`）。

## 2. 冪等性（B-2-1）

`attach = sink_.store(s, std::memory_order_release)` は**冪等**です。再 prepare でも同じ sink を再設定するだけ。
`prepare` が複数回呼ばれても状態は変わりません。

## 3. DSPCore の生成順序と attach（B-2-2）

| 経路 | 生成 | prepare | sink |
|---|---|---|---|
| rebuild | `RuntimeBuilder.cpp:469` | **`:474`（owner = `&engine`）** | エンジン |
| placeholder | `PrepareToPlay.cpp:268` | **`:272`（owner = `this`）** | エンジン |
| テスト（V11 等） | `V11EngineRetireProtectionTests.cpp:457` ほか | **呼ばない**（直接 register） | **nullptr のまま** |

**DSPCore はエンジン初期化より後に生成される**ため、「initialize で既存コアを一括 attach」では網羅できません。
**生成点（prepare）で attach する**のが唯一の網羅的な位置です。

## 4. 非所有インスタンスの `sink_ == nullptr`（B-2-2）

| 経路 | 初期値 | attach | 保証 |
|---|---|---|---|
| DSPCore（エンジン所有） | nullptr | prepare で attach | — |
| DSPCore（テスト・owner なし） | nullptr | **されない**（prepare を呼ばない） | 実測 |
| テストのローカル `ConvolverProcessor` 23 個 | nullptr | **されない**（エンジンが存在しない） | 実測 |
| `uiConvolverProcessor` | nullptr | initialize で attach | — |

**構造的保証**: `ConvolverProcessor` に `DiagSink*` を渡す公開 API を作らず、
**エンジンが自分の sink を自分のコアへ bind** する形（`owner->bindDiagnostics(*this)`）にします。
これにより**別エンジンの sink を誤設定することが型・アクセス制御で不可能**になります。

## 5. detach と参照寿命の証明（4 段）

| # | 内容 |
|---|---|
| **P1** | 各生産者は自分のエンジンの sink のみを持つ（グローバル廃止）→ **混線経路が存在しない** |
| **P2** | 非所有インスタンスは `sink_ == nullptr` → **書き込み自体が起きない** |
| **P3** | **一次保証**: エンジン破棄前に join / lock で RT を静止（`B2-LIFETIME-PROOF` §1.4）→ 破棄後に push する生産者はいない |
| **P4** | 防御: `DSPCore::~DSPCore` と `~AudioEngine` で sink を clear（P3 が破れても null チェックで no-op） |

**detach 後に取得済みポインタを使う競合**は、P3 により「detach の時点で生産者が実行していない」ため閉じます。
atomic 化だけでは P3 の代替になりません。

**残余（明示）**: `abandonEngine()`（テスト専用）はエンジンを破棄せず leak します。コアの sink は
有効なまま残りますが、**RT が存在しないため push は起きません**（メモリは leak）。

## 6. 診断 ON の ASan ローカル検証計画（B-2-3・CI 変更は未承認）

| 項目 | 内容 |
|---|---|
| 構成 | 新規 `build-asan-diag`（`-DENABLE_ASAN=ON` かつ `-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`、RelWithDebInfo） |
| 対象 | `AudioEngineHarness`（51 テスト）、スタンドアロン conv を含むテスト群 |
| S1/S2 | 1 回 / 10 回反復。**B-2 適用前に ASan が UAF を報告すれば、ダンプ 1 件に依存しない独立確認になる** |
| S3 | スタンドアロン conv を含むテスト → B-2 後は sink が nullptr で書き込みなし |
| S4 | 2 エンジン同時 → 各自のバッファのみ |
| 実行 | **build / test は禁止のため未実行**（計画のみ）。CI 追加は別途承認 |

## 7. 実装時の変更対象（参考・未実施）

`AudioEngine.h`（`DiagSink`・`diagSink_`・`bindDiagnostics`）、`AudioEngine.Init.cpp`（グローバル設定 → attach）、
`DSPCoreLifecycle.cpp`（コンストラクタは attach しない）、`RuntimeBuilder.cpp` / `PrepareToPlay.cpp`（prepare 内で bind）、
`ConvolverProcessor.Runtime.cpp` / `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp` / `DSPCoreIO.cpp`（グローバル削除・`sink_` 参照）、
`AudioEngine.CtorDtor.cpp`（防御的 detach）。

## 8. 遵守事項

| 制約 | 状態 |
|---|---|
| 製品コード・テスト変更 | **なし**（追跡差分 **0**） |
| build / test / debugger / replay | **未実行** |
| CI 定義変更 | **なし** |
| U5 / U3・U4 | **FAIL・既定 OFF / HOLD を維持** |
| 未追跡 | 既存 24 件は不変。本フェーズの追加文書を含めた総数は `git status` で確認（整理・削除・追跡なし） |
| 作業拠点 | `C:\VSC_Project\ConvoPeq` の main-develop のみ |

## 9. 次段

**この 2 点（attach 位置・生成順序）と ASan 計画が確定したため、B-2 の実装承認にお進みいただける状態です。**
実装にはコード変更・ビルド・テスト実行の明示承認が必要です。現時点では HOLD を維持しています。
