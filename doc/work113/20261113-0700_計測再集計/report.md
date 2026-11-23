# 停止レイテンシ再集計（コホート分離と shutdown 時待機）

- 作業日時: 2026-11-13 07:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`
- 種別: **計測専用変更**（Owner 承認 2026-11-13）。shutdown 本体は実装していない。
- 本稿は `doc/work113/20261113-0600_停止計測2/report.md` の**集計を置き換える**（生データは同梱のまま）。
- 前稿の集計不整合（n=18 と実データ 27 / 23 の差）を解消し、W5/W6 を**停止理由別**に分離し、
  **shutdown 時点で走行中の Worker を停止した待機時間**を新たに測った。

---

## 1. 前稿の集計不整合（原因と訂正）

前稿の n=18 は、`m2_w2.jsonl`(12) と `m3_w6.jsonl`(6) の 2 ファイルだけを合算した値だった。
`m2_w5.jsonl`(5) と `m2_learn.jsonl`(4) を無言で除外していた。**除外条件を書いていなかったのは誤りである。**

本稿では、コホートを明示して全て集計する。

| ファイル | 試行 | 内容 |
| --- | ---: | --- |
| `m2_w2.jsonl` | 12 | 通常起動 → 終了（追加の負荷なし） |
| `m2_learn.jsonl` | 4 | 学習 ON で起動 → 終了 |
| `m2_w5.jsonl` | 5 | 非同期ロードを 8 回連続発行（置換キャンセル） |
| `m3_w6.jsonl` | 6 | upgrade 有効 + 6 回ロード（置換キャンセル） |
| `m4_w5shutdown.jsonl` | 6 | 終了 200 ms 前に LoaderThread を起動（**shutdown 時停止**） |
| `m4_w6shutdown.jsonl` | 6 | 同上（キャッシュ温のため upgrade は起動せず。W2 用のみ有効） |
| `m5_w6shutdown.jsonl` | 6 | 終了 400 ms 前に upgrade を起動（**shutdown 時停止**・未使用 IR で 6 通り） |

---

## 2. コホート定義

| # | コホート | 含むファイル | 除外 |
| --- | --- | --- | --- |
| C1 | W2 `coord_stop`（全実行） | 上記 7 ファイルすべて | なし |
| C2 | `stopLearning` terminal（学習 OFF） | C1 のうち `m2_learn.jsonl` を除く | 学習 ON の 4 試行 |
| C3 | `stopLearning` terminal（学習 ON） | `m2_learn.jsonl` | − |
| C4 | W5 **置換キャンセル**（新ロードが旧 LoaderThread を止める） | `m2_w5.jsonl` | shutdown 時停止 |
| C5 | W5 **shutdown 時停止**（`forceCleanup()` の `stopThread(500)`） | `m4_w5shutdown.jsonl` | 置換キャンセル |
| C6 | W6 **置換キャンセル**（新 upgrade が旧 upgrade を止める） | `m3_w6.jsonl` | shutdown 時停止・探索 |
| C7 | W6 **探索**（前ビルド・参考値） | `m1_E4.jsonl` | 本集計から分離 |
| C8 | W6 **shutdown 時停止**（`stopUpgradeThread()` の `stopThread(2000)`） | `m5_w6shutdown.jsonl` | 置換キャンセル |

C4/C5、C6/C8 を混ぜない理由は、**停止要求の発行元が違う**ことである。
置換キャンセルは「次のロードが来たので前を止める」、shutdown 時停止は「プロセス終了のために止める」。
待つ側のタイムアウト（500 ms / 2000 ms）も、待つ場所（`forceCleanup()` / `releaseResources`）も異なる。

---

## 3. 結果

単位はマイクロ秒。**max は観測値であり保証値ではない。**

| # | コホート | n | p50 | p95 | max | min | 協調終了 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C1 | W2 `coord_stop` | 45 | 1,744 | 4,296 | 4,568 | 21 | 45/45（timeout 0） |
| C2 | `stopLearning`（学習 OFF） | 41 | 3 | 5 | 22 | 0 | − |
| C3 | `stopLearning`（学習 ON） | 4 | 2,828 | 9,125 | 10,223 | 2,435 | − |
| C4 | W5 置換キャンセル `request`→`run_exit` | 35 | 23,059 | 29,137 | 31,320 | 20,169 | − |
| C5 | W5 **shutdown 時** `stopThread(500)` 待機 | 6 | 6,106 | 12,171 | 12,362 | 4,147 | **6/6**（強制終了 0） |
| C6 | W6 置換キャンセル `request`→`run_exit` | 25 | 39,114 | 83,206 | 86,372 | 2,111 | − |
| C7 | W6 探索（参考・別集計） | 5 | 10,742 | 39,138 | 44,999 | 3,091 | − |
| C8 | W6 **shutdown 時** `stopThread(2000)` 待機 | 6 | 39,469 | 49,109 | 51,063 | 36,574 | **6/6**（強制終了 0） |

### 3.1 C5 / C8 が「走行中」を測れていることの確認

各試行の最初の `request` レコードの `already_exited` を見る。

| コホート | 各試行の 1 回目の要求 | 2 回目（`~LoaderThread` / デストラクタ由来） |
| --- | --- | --- |
| C5 | `already_exited = 0`（走行中）× 6 | `1`（既に終了）× 6 |
| C8 | `already_exited = 0`（走行中）× 6 | `1` × 6 |

C5/C8 は**要求時点で Worker が走っていた**標本である。前稿の C5 相当（`forceCleanup` の 5 件）は
すべて `already_exited=1` で、走行中の停止を測れていなかった。本稿でそこを埋めた。

### 3.2 W6 の分布（C6）が二峰性になる理由

- 小さい側（2.1〜15.7 ms）: `convertToHighRes`（13.2〜30.7 ms）の途中、または FFT ステップ境界で止まる。
- 大きい側（17.7〜86.4 ms）: `cache_load`（93.4〜96.7 ms）または `cache_save_evict`（99.1〜105.0 ms）の
  実行中に要求が出る。**この 2 つは内部にキャンセル点が無い。**
- C8 が 36.6〜51.1 ms に集まったのは、shutdown の時点で `cache_save_evict` が実行中だったためである
  （例: 要求 5,400 ms、`cache_save_evict` は 5,349〜5,442 ms、`upgrade_stopThread` は 39.9 ms）。

### 3.3 W5 の分布（C4 と C5）の違い

- C4（置換）は 20.2〜31.3 ms。`req_to_first_check_us` が 19.1〜29.7 ms を占め、
  **次のチェックポイントまでの待ち**が支配的である。
- C5（shutdown）は 4.1〜12.4 ms。停止要求が出た時点で比較的早いチェックポイントに近かったためである。
  同じ LoaderThread でも、実行中のステップによって待ち時間が変わる。

### 3.4 探索データの扱い

`m1_E4.jsonl` は、停止待機をスパンで測る前のビルドで取得した 5 標本である（要求の発行元は C6 と同じ置換）。
本集計には含めず、C7 として参考値のみ示す。**同一母集団に混ぜる根拠は無い**（ビルドが異なる）。

---

## 4. `B_pre` への含意（値は確定しない）

- ~~W5 は shutdown 時の証拠として不足~~ → C5 で補った（n=6、max 12.4 ms）。
- W6 は停止理由別に分離した（C6 と C8）。shutdown 時の max は 51.1 ms。
- **n=6 は分布を語るには少ない。** p95 は 6 標本では実質 max と同義である。
- W1（`rebuildThread`）は依然として停止が有界でない（`LoopExited` までしか確認できない）。
- `closeAudioDevice()` は無界。
- したがって **`B_pre = 500 ms` の採用は引き続き見送る**。500 ms を採っても総 shutdown 時間は有界にならない。

**次に `B_pre` を決めるなら、C5/C8 を各 30 試行程度まで増やし、負荷（音声コールバック＋EQ/バイパス切替）を
かけた条件を加える必要がある。** 現時点の 6 標本はその前段の証拠である。

---

## 5. 再現手順（shutdown 時停止）

```
rem W5: 終了 200 ms 前に LoaderThread を起動する
ConvoPeq.exe --cli-run --cli-measure-ir <3秒IR> --cli-measure-late-load-at-ms 5000 ^
             --cli-exit-ms 5200 --cli-log-file <log>

rem W6: 終了 400 ms 前に upgrade を起動する（未使用の IR×phase で 4096 キャッシュを避ける）
ConvoPeq.exe --cli-run --cli-measure-ir <実IR> --cli-phase Mixed --cli-measure-enable-upgrade ^
             --cli-measure-late-load-at-ms 5000 --cli-exit-ms 5400 --cli-log-file <log>
```

注意: 4096 FFT のキャッシュが温まっていると `appliedFft = 4096` になり upgrade が起動しない
（`startProgressiveUpgrade` の `currentFFTSize >= targetFFT` で return）。試行ごとに未使用の IR か phase を選ぶ。

---

## 6. 生データ

`probe/` に同梱（`MANIFEST.md` に SHA-256 と行数）。

| ファイル | 内容 |
| --- | --- |
| `m2_w2.jsonl` / `m2_learn.jsonl` | C1 / C2 / C3 の基礎 |
| `m2_w5.jsonl` | C4（置換キャンセル 35 標本） |
| `m3_w6.jsonl` | C6（置換キャンセル 25 標本） |
| `m1_E4.jsonl` | C7（探索・参考） |
| `m4_w5shutdown.jsonl` | C5（shutdown 時停止 6 標本） |
| `m4_w6shutdown.jsonl` | upgrade 不成立（キャッシュ温）。C1/C2 にのみ寄与 |
| `m5_w6shutdown.jsonl` | C8（shutdown 時停止 6 標本） |

---

## 7. 計測専用変更の追加分（本稿のための差分）

| 箇所 | 追加 |
| --- | --- |
| `ConvolverProcessor.StateAndUI.cpp` `forceCleanup()` | `loader_stopThread` スパン（`stopThread(500)` の待機）＋ `_result` マーク |
| `ConvolverProcessor.LoadPipeline.cpp` `stopUpgradeThread()` | `upgrade_stopThread` スパン＋ `_result` マーク |
| `MainWindow.cpp` | `--cli-measure-late-load-at-ms`（終了直前に Worker を起動する opt-in） |

いずれも既定 OFF。通常実行の挙動は変えていない。

---

## 8. 停止条件の確認

- shutdown 本体・状態機械・停止 API・所有権・破棄順序は変更していない。
- 変更は計測専用（スパン 2 箇所、マーク 2 箇所、CLI 1 フラグ）。
- git の commit / push を行っていない。HEAD は `173e6995…`。
