# shutdown v2-a: 計測準備と実装単位の再確認（実装なし）

- 作業日時: 2026-11-13 17:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **設計・計測準備のみ**。ソース変更・ビルド・テスト・人工 stuck 試験は行っていない。
- 前提: Owner 判断（2026-11-13）により v2-a の実装 GO と M-2 は未承認。

---

## 1. 既存ログで取得できる計測値（今回の収集結果）

既存のログには teardown のマーカーが揃っている。時刻は持たないため、**外部から 5 ms 間隔でログを
ポーリングして各マーカーの出現時刻を記録**した（コード変更なし）。

### 1.1 使えるマーカー

| key | ログ行 |
| --- | --- |
| AUTOEXIT | `[CLI] Auto-exit scheduled` |
| SHUT_BEGIN | `[D123] SHUTDOWN_BEGIN` |
| STEP4 | `[DIAG] ~MainWindow: step 4 setProcessor(nullptr)` |
| RR_ENTER | `[DIAG] releaseResources: enter` |
| W1_BEFORE / W1_AFTER | `[DIAG] releaseResources: before/after stopRebuildThread` |
| UI_BEFORE / UI_AFTER | `[DIAG] releaseResources: before/after ui processor release` |
| STEP6 / STEP7 | `[DIAG] ~MainWindow: step 6 saveSettings` / `step 7 removeAudioCallback` |
| STEP8 / STEP9 / STEP10 | `closeAudioDevice` / `audioEngineProcessor.reset` / `UI reset` |
| RESET_DONE | `[D123] SHUTDOWN: mainWindow.reset() completed` |
| LOGDETACH | `[D123] LOGGER_DETACH / SHUTDOWN_END` |

### 1.2 実測（`--cli-run --cli-exit-ms 2000`、3 回）

| 項目 | run1 | run2 | run3 |
| --- | ---: | ---: | ---: |
| 総経過 | 6,353 ms | 6,308 ms | 6,306 ms |
| 起動（AUTOEXIT） | 2,011 ms | 1,933 ms | 1,964 ms |
| SHUTDOWN_BEGIN | 3,872 ms | 3,857 ms | 3,858 ms |
| **C 全体**（SHUT_BEGIN→RESET_DONE） | **2,436 ms** | **2,417 ms** | **2,418 ms** |
| W1 停止（before→after） | 0 ms | 0 ms | 32 ms |
| W5/W6（ui processor release） | 44 ms | 12 ms | 45 ms |
| saveSettings（step6→step7） | 58 ms | 36 ms | 46 ms |
| step8→step10 | 47 ms | 64 ms | 46 ms |
| step4→RR_enter | 26 ms | 0 ms | 0 ms |
| RR_enter→W1_before | 0 ms | 65 ms | 15 ms |

読み方:

- **W1 の停止は 0〜32 ms**、**W5/W6 は 12〜45 ms**、`saveSettings` は 36〜58 ms。
  U-1 で問題にしている停止待ちは、通常条件では 2 桁 ms である。
- **C 全体は約 2.42 秒**で、上記の内訳（合計 0.2 秒未満）では説明できない。
  残りは `~AudioEngine` の graceful drain（`kGracefulDrainMaxMs = 5000` のポーリング）と
  `closeAudioDevice()` で、**どちらもマーカーが無い**。
- したがって「C が 2.4 秒」の主因は `B_pre` ではなく drain である。U-1 の bound を議論するとき、
  C 全体と混同しないこと。

### 1.3 既存ログでは取れないもの

| # | 対象 | 理由 |
| --- | --- | --- |
| 1 | W2（`coordinatorLoop_`）の停止時間 | before/after のマーカーが無い（`stopThread(2000)` の内側） |
| 2 | `stopLearning()` の join | マーカーが無い |
| 3 | drain の内部（待機回数・最終カウント） | マーカーが無い（timeout 時のみ `[AUDIT] Graceful drain timeout` が出る） |
| 4 | 走行中の W5/W6 の停止 | 走行状態そのものを作れない |
| 5 | 全ログ行の時刻 | ログ形式に時刻が無い |

## 2. M-1〜M-4 の取得可能性

| # | 対象 | 現状 | 必要なもの |
| --- | --- | --- | --- |
| M-1 | W2 の停止時間 | **取得不可** | `shutdownCoordinatorLoop` の前後に 1 行ずつログ追加（時刻は外部ポーリングで足りる） |
| M-2 | W5/W6 の**走行中**停止 | **取得不可** | 走行中を作る計測専用スイッチ（未承認。§6 で再申請） |
| M-3 | `stopLearning()` の join | **取得不可** | 呼出し前後に 1 行ずつログ追加 |
| M-4 | W1 の停止 | **取得済み**（§1.2） | 追加不要 |

M-1・M-3 は「ログ行の追加のみ」であり、判定・分岐・順序を変えない。それでもソース変更であるため、
実施には承認が要る（本指示では未承認）。

## 3. `bound` の決定材料

### 3.1 現時点のデータ

| 対象 | 観測（通常条件・3 件） | 備考 |
| --- | --- | --- |
| W1 | 0 / 0 / 32 ms | M-4 で取得済み |
| W5/W6 | 44 / 12 / 45 ms | 走行していない状態 |
| W2 | 未計測 | M-1 待ち |
| `stopLearning()` | 未計測 | M-3 待ち |

### 3.2 規則の適用

`20261113-1630_v2a改訂案` §1.4 の `clamp(1.5 × p99, 500 ms, 5000 ms)` を当てると、
観測最大 45 ms では `1.5 × 45 = 68 ms` となり**下限 500 ms** に張り付く。

### 3.3 提案

- 暫定候補を **500 ms** とする（5,000 ms ではなく）。根拠は「通常条件の停止は 2 桁 ms であり、
  500 ms でも 10 倍以上の余裕がある」こと。
- ただし W2・`stopLearning()`・走行中 W5/W6 が未計測のため、**確定はしない**。
- 5,000 ms を維持する理由は「既存 `kGracefulDrainMaxMs` と同じ桁」だけであり、根拠としては弱い。
- 500 ms に下げると、停止に 500 ms 以上かかる異常時に保持経路へ落ちる頻度が上がる。
  これは安全側（プロセスを残す）であり、危険側ではない。

## 4. P1〜P8 の依存関係・受入れ条件・失敗時の経路（再確認）

| # | 単位 | 依存 | 受入れ条件 | 失敗時 |
| --- | --- | --- | --- | --- |
| P1 | 状態機械 | なし | S0→S1 が MT のみ。既存 40 箇所の判定時期が変わらない | 遷移しない＝S0 のまま（従来動作） |
| P2 | W1 状態フラグ | なし | ループ末尾で true、起動点で false | フラグ未設定＝preflight が W1 を「未停止」と判定し保持へ |
| P3 | preflight | P1, P2 | 要求が join を呼ばない（I-A3）。deadline を共有 | 期限内に停止しなければ S3（保持） |
| P4 | producer 拒否 | P1 | S1 以降に開始しない（I-A1） | 拒否漏れ＝停止中の新規ロード（設計違反。テストで固定） |
| P5 | 非破壊分岐 | P1 | `workersStuck_` 中は何も破棄しない（I-B1） | 破棄してしまうと保持が成立しない（H-1〜H-4） |
| P6 | callback 抑止 | P1 | 抑止しても解放は行う（I-C2） | 抑止漏れ＝停止中の callback 実行 |
| P7 | 同期区間 | P1 | 同期区間内でメッセージループを回さない（I-E1） | 導入すると S1 が成立しない |
| P8 | 保持 | P3, P5 | `~MainWindow` / `~AudioEngine` を実行しない（I-F2） | 実行すると join で固まる／`std::terminate` |

失敗時の終了経路（再確認）:

- S3（保持）に入った場合、プロセスは「音声デバイスを閉じ、UI と Worker を残したまま」終了まで走る。
- 終了コードは 0（`MainApplication::shutdown()` は保持経路でもそのまま戻る）。
- 保持中も `closeAudioDevice()` が無界である点は既存挙動（U-13）。ここだけは保証できない。

## 5. 実装単位ごとの変更対象・検証・ロールバック

| # | 変更対象（想定） | 検証 | ロールバック |
| --- | --- | --- | --- |
| P1 | `AudioEngine.h`（atomic 1 個と述語）、`ISRShutdown.cpp`（閉鎖点） | 述語の真理値表テスト、既存 61 件 | 追加した atomic と述語を削除（参照箇所は P1 内のみ） |
| P2 | `AudioEngine.h`、`Init.cpp`、`PrepareToPlay.cpp`、`RebuildDispatch.cpp` | W1 停止の実測（M-4）と単体テスト | フラグの設定/クリアを削除（preflight は P3 側で無効化） |
| P3 | `AudioEngine.*`、`ConvolverProcessor.*` | 停止時間の実測（M-1・M-2）、I-A3 のテスト | 2 メソッドを削除し、呼出し側を従来経路へ戻す |
| P4 | `LoadPipeline.cpp`、`Lifecycle.cpp` | 各入口の拒否テスト | 追加した判定を削除 |
| P5 | `ReleaseResources.cpp`、`ConvolverProcessor.Lifecycle.cpp` | 保持経路の実走 | 分岐を削除して従来の teardown へ |
| P6 | `LoaderThread.cpp`、`ProgressiveUpgradeThread.cpp`、`LoadPipeline.cpp` | 抑止テスト | 追加した epoch 判定を削除 |
| P7 | `ConvolverProcessor.*` | 同期区間のテスト | トークンを削除 |
| P8 | `MainApplication.cpp`、`MainWindow.*` | 保持経路の実走＋ログ | 保持を削除し、従来どおり破棄する |

いずれの単位も「追加した分岐とメンバを削除すれば従来動作に戻る」形にする（既存の制御フローを書き換えない）。

## 6. M-2（計測専用スイッチ）の再申請

| # | Owner 指定項目 | 内容 |
| --- | --- | --- |
| 1 | 追加箇所と有効化条件 | `LoadPipeline.cpp` の `loadImpulseResponse` 入口と `AudioEngine.Parameters.cpp` の progressive upgrade 有効化点に、環境変数 `CONVOPEQ_MEASURE_BUSY_WORKERS=1` を読む分岐を 1 箇所ずつ追加。**コンパイル定義 `CONVOPEQ_ENABLE_MEASURE_HOOKS` がある場合のみ**コンパイルされ、環境変数が未設定なら従来動作 |
| 2 | 計測対象と取得する値 | W5（`activeLoader`）と W6（`upgradeThread`）が走行中の停止時間。取得は §1.1 の既存マーカー（`UI_BEFORE` / `UI_AFTER`）＋外部ポーリング |
| 3 | 通常動作への影響 | 製品ビルドではコードが存在しない。計測ビルドでも未設定時は分岐 1 回（真偽判定のみ） |
| 4 | 撤去手順 | 環境変数を削除し、`#if` ブロックを削除する（常用しない） |
| 5 | 製品動作に残らないことの確認 | リリース exe に `CONVOPEQ_MEASURE_BUSY_WORKERS` の文字列が含まれないことを `findstr` で確認 |
| 6 | 実施環境と回数 | ローカル Release、W5 単独 10 回・W6 単独 10 回（計 20 回）。1 回あたり `--cli-run --cli-exit-ms 2000` |

## 7. 未確定のまま残る事項

| # | 事項 | 状態 |
| --- | --- | --- |
| R-1 | W2 の停止時間 | M-1（ログ 2 行）の承認待ち |
| R-2 | `stopLearning()` の join | M-3（ログ 2 行）の承認待ち |
| R-3 | 走行中 W5/W6 | M-2 の承認待ち |
| R-4 | `bound` の確定値 | 上記 3 つの計測後 |
| R-5 | 人工 stuck 試験 | 計画は `20261113-1630_U-3整合` §8。実施承認待ち |

## 8. 停止条件の確認

- ソース変更・ビルド・テスト・人工 stuck 試験は行っていない（本課題のため）。
- 計測は既存バイナリと既存ログのポーリングのみで行った。
- M-2 は再申請の提示にとどめ、実施していない。
- Git の変更操作を行っていない。
