# soak 手動実行の記録（2026-11-06）

- 目的: RES2-12 の判断材料として、oneAPI 導入済み開発機で soak を 1 回実行し、時間・合否・ログを記録する。
- 実施: 2026-11-06 03:19:58 開始、03:32:23 終了（ローカル時刻）。CI は使っていない。
- 結論: `all` シナリオは ALL PASS。実行時間は 744.8 秒（12.4 分）。

## 1. 実行環境

| 項目 | 値 |
|---|---|
| OS | Windows 11 Home build 26300 |
| CPU | Intel Core i7-11700K |
| メモリ | 31.8 GB |
| oneAPI | あり（`C:\Program Files (x86)\Intel\oneAPI`、MKL と compiler） |
| ビルド | `build`（Visual Studio 18 2026、Release） |
| 実行ファイル | `build\Release\AudioEngineHarness.exe` |
| ビルド日時 | 2026-11-06 03:09:29 |
| SHA-256 | `2B23692C3A7218084DDD1B5F6F58E97F43EA059CEAEB441BFA03A105E842C912` |
| ビルド時間 | 990.5 秒（Release、`AudioEngineHarness` ターゲット） |
| 基準 commit | `ce386431`（作業ツリーに本作業の変更 2 ファイル） |

ハーネスは基準 commit のソースから作り直した。既存の実行ファイルは 2026-11-04 のもので、以降の変更を含んでいなかった。

## 2. 実行条件

```
build\Release\AudioEngineHarness.exe --soak --scenario=all
```

- 実行中の負荷: なし。WSL は停止し、ビルドも完了した状態で実行した。
- 追加の引数・環境変数: なし。

## 3. 結果

| シナリオ | 出力 |
|---|---|
| s1 | `published=100000 accepted=100000 rejected=0 finalSeq=100003 backlog=0` |
| s2b | `requested=80000 accepted=80000 rejected=0 peakBacklog=0` |
| s3 | `rapid=300 drained_to_seq=180303 backlog=0` |
| s4 | `published=10001 epoch_start=360606 epoch_final=380608 pending=0` |
| s5 | `samples=158 peak=284.7MB final=284.7MB (converged)` |
| 合計 | `issued=180300 accepted=180300 rejected=0 peakBacklog=0` |
| 判定 | `ALL PASS` |

| 項目 | 値 |
|---|---|
| 終了コード | 0 |
| 実行時間 | 744.8 秒 |
| ログ | `soak-all-20261106-031958.log`（7 行） |
| 前回（2026-11-05） | 799.9 秒、ALL PASS |

前回との差は 55.1 秒（6.9%）。条件は同じだが、前回の測定時刻と負荷状態は記録していないため、差の原因は特定していない。

## 4. 失敗時の復旧手順

1. ログの末尾で `ALL PASS` の有無と `rejected`・`peakBacklog` を確認する。
2. 失敗したシナリオだけを `--scenario=<name>` で再実行し、再現するかを見る。
3. ハーネスのビルドが古い場合は `cmake --build build --config Release --target AudioEngineHarness` で作り直す。
4. 再現する場合は、変更した commit と実行ファイルのハッシュを記録して Owner に報告する。

## 5. 運用上の評価

| 項目 | 実測・所見 |
|---|---|
| 1 回の時間 | 12.4 分（13.3 分の見積もりを下回った） |
| 失敗時の復旧 | シナリオ単位の再実行が可能。全体の作り直しは不要 |
| 常時稼働 | 未構築。GitHub のセルフホストランナー登録は行っていない |
| 費用 | この開発機を使う限り追加費用は無い。常時稼働させる場合の電気代と保守は別 |
| 運用責任 | ランナーの更新、ディスク、セキュリティの担当者は未定 |

## 6. 未実施・未確認

- GitHub のセルフホストランナーとしての登録と、CI からの実行。登録にはリポジトリ側の設定変更が必要で、本作業では行っていない。
- 週次実行と `schedule` の追加（未承認）。
- 連続実行による安定性（1 回のみ）。
- 他環境への一般化。この結果は本機の 1 回の記録である。
