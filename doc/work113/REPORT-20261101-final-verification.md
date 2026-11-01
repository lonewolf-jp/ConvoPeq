# B-2 最終確認報告（2026-11-01 第 7 部）

- 基準: `main` = `origin/main` = **`6bb7c8d5`**（ソースは `fd92c6e7` と同一）
- **製品コード・テスト・CI 定義は未変更。build / test / debugger / replay 未実行**
- 詳細: `doc/residual-20261101/B2-FINAL-VERIFICATION-20261101.md`

## 1. `deferredDestroy_` の型消去（指示 1）

**結論: この vector は DSPCore 専用であり、型付きに変更できる。**

| 証拠 | 内容 |
|---|---|
| 唯一の書き手 | `deferDestroyRetry`（`AudioEngine.Threading.cpp:70-86`） |
| その唯一の呼び出し元 | `destroyDSPCoreNodeRetryable` の pin 残留分岐（`:42-51`） |
| 同関数の破棄処理 | `static_cast<DSPCore*>(p); core->~DSPCore(); aligned_free(core)`（`:56-58`） |
| 重複登録 | 防止済み（`:75-79` で同一ポインタをスキップ） |
| teardown 中の変異 | `retryDeferredDestroys` が **mutex 下で swap**（`:102-109`） |

**混同注意**: `enqueueDeferredDeleteNonRt`（`CacheMap` / `RuntimePublishWorld` など複数型）は**別ストア**。
`deferredDestroy_` とは無関係です。

**型安全化**: `std::vector<void*>` → **`std::vector<DSPCore*>`**。teardown では型付きで走査し、
**未検証の `void*` をキャストしない**。

## 2. leak コアの detach 安全性（指示 2）

| 項目 | 実測 |
|---|---|
| `activeReaderCount()` の意味 | **RCU 読み取り区間（depth > 0）に滞在中のスレッド数**（`EpochDomain.h:305-315`）。登録数ではない |
| reader は sink を使うか | **使わない**（`process()` を呼ばない） |
| 診断生産者 | **RT のみ**（14 箇所） |
| RT の停止 | `StopAudio` で join / `audioCallbackLock` により停止済み |

**→ reader 残留と診断生産者残留は独立。leak コアの detach は reader の安全性を損なわない。**

**detach 時に「sink を取得済みの生産者」がいない根拠**: ①生産者は RT のみ ②RT は join 済み
③reader は `process()` を呼ばない ④したがって load 済みポインタを持つスレッドが存在しない。

**順序**: `StopAcceptingWork` → `StopAudio` → `StopWorkers`（Coordinator/Builder join）→ `DrainRetire`
→ `retryDeferredDestroys` → **【新規】leak コアの detach** → member teardown。

## 3. ownerId の一意性（指示 3）

| 項目 | 設計 |
|---|---|
| 生成 | プロセス全体の `std::atomic<uint64_t>` から `fetch_add` |
| 0 | **予約（未束縛）**。採番は 1 から |
| 再利用 | **なし**（単調増加。破棄済みエンジンの ID は再発行しない） |
| 重複 | 構造的に不可 |
| オーバーフロー | 実質到達不能。到達時は **assert + 異常終了（fail-closed）**。0 に戻さない |

## 4. prepare の実行スレッドとアクセス制御（指示 4）

| # | 呼び出し元 | スレッド | owner |
|---|---|---|---|
| 1 | `RuntimeBuilder.cpp:474` | **rebuild スレッド** | `&engine` |
| 2 | `PrepareToPlay.cpp:272` | **message スレッド** | `this` |

テストは `prepare` を呼びません。bind はこの 2 箇所のみ、detach はコア dtor と teardown の走査のみで、
**いずれも rebuild join 後**です。

**アクセス**: `DSPCore`（入れ子）→ 外側 `AudioEngine` の private は**可**（[class.access.nest]/1）。
ロジックを入れ子側に置くため **`friend` 不要**。外側から入れ子の private へは触れません。

**ownerId 設定後・sink 未設定の状態**: `prepare` は **publish より前**に完了するため RT から観測されません。
仮に観測されても `sink_ == nullptr` で push なし（T9）。

## 5. 診断バッファの参照経路（指示 5）

| 生産者 | sink の取得元 |
|---|---|
| `AudioBlock.cpp` / `BlockDouble.cpp`（8 箇所） | エンジンの `diagSink_`（**元からグローバル非経由**） |
| `DSPCoreFloat/Double/IO`（4 箇所） | **DSPCore が渡す sink**（引数化。グローバル廃止） |
| `ConvolverProcessor.Runtime.cpp`（2 箇所） | **`process(block, sink)` の引数**（グローバル廃止） |

**同一 owner の同一 sink** を指すことの確認: エンジン直下は `&engine.diagSink_`、DSPCore の bind も
`owner->diagnosticsSink()` = **同じオブジェクト**。別エンジンの sink は構造的に渡りません。

## 6. 変更規模の見積り

| 種別 | 箇所数 |
|---|---|
| グローバル定義の削除 | 2 群 |
| push サイトの書き換え | **14** |
| シグネチャ変更（sink 受け渡し） | 5（`process` 1、`logEqTime` 2、`processOutput`/`processOutputDouble` 2） |

## 7. 残る不確定（実装時）

1. `deferDestroyRetry` の引数型を `DSPCore*` に変える範囲（void deleter 経由との整合）。
2. `sink_` を `std::atomic` にするか素のポインタにするか（publish の HB で足りるが atomic(relaxed) を推奨）。
3. `sizeof` の実測（`static_assert` は補助。ODR は単一定義で担保）。

## 8. 遵守事項

製品コード・テスト・CI 定義変更 **なし**（追跡差分 0）／build・test・debugger・replay **未実行**／
U5 FAIL・既定 OFF、U3/U4 HOLD を維持／既存の未追跡 18 件は不変。
