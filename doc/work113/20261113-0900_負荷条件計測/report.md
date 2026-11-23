# 負荷条件別の shutdown 停止計測（C5 / C8）

- 作業日時: 2026-11-13 09:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`
- 種別: **計測専用変更**（Owner 承認 2026-11-13「C5/C8 の追加計測を承認」）。
  shutdown 本体・状態機械・停止 API・所有権・破棄順序は変更していない。
- 承認された条件: C5/C8 各 30 試行以上、通常負荷と負荷条件（音声コールバック稼働・EQ/バイパス切替）。
  条件を混ぜず、コホート別に集計する。

---

## 1. 結論（先に）

1. **通常負荷では、shutdown 時点で走行中の Worker を停止した待機時間が取れた。**
   C5 は 29/30 試行で走行中（1.6〜46.5 ms）、C8 は 27/27 試行で走行中（30.6〜72.2 ms）。
2. **切替負荷では、走行中の Worker を shutdown 時点に残せなかった。**
   EQ 切替もバイパス切替も、**shutdown の到着自体を 1.2〜3.1 秒遅らせる**ため、
   その間に LoaderThread（約 240 ms）と UpgradeThread（約 500〜750 ms）が終了してしまう。
   切替条件の 90 試行すべてで `stopThread` の待機は 0〜2 µs（待ち対象なし）だった。
3. したがって **切替負荷条件の停止待ち分布は取得できていない**。取得できない理由は測定手順ではなく、
   「切替が shutdown 到着を遅らせる」という実測された構造である（§4）。
4. この結果から `B_pre` は決められない。むしろ、負荷時は **停止待ちより「shutdown 到着の遅れ」が支配的**である。

---

## 2. 条件と試行数

| コホート | 内容 | 試行 | 有効標本 |
| --- | --- | ---: | ---: |
| C5-N | W5 通常負荷（終了 200 ms 前に LoaderThread 起動） | 30 | 30 |
| C5-S1 | W5 EQ 切替（4,800 ms から 2 s、EQ gain + bypass 交互） | 30 | 30 |
| C5-S2 | W5 bypass 切替（4,980 ms から 2 s、bypass のみ） | 30 | 30 |
| C5-S3 | W5 bypass 切替（260 ms・6 秒 IR） | 30 | **0（無効）** |
| C8-N | W6 通常負荷（終了 400 ms 前に upgrade 起動） | 30 | 27 |
| C8-S1 | W6 EQ 切替 | 30 | 24 |
| C8-S2 | W6 bypass 切替（2 s） | 30 | 6 |
| C8-S3 | W6 bypass 切替（260 ms） | 30 | 0 |

「有効標本」は、その実行で対象 Worker が実際に起動し `stopThread` のスパンが記録された数である。
C8 で減っているのは、4096 FFT のキャッシュが温まっている回で upgrade が起動しないため
（`startProgressiveUpgrade` の `currentFFTSize >= targetFFT` で return）。

---

## 3. 結果（通常負荷）

`stopThread` スパンと、その直前に記録された停止要求を 1 対 1 で対応づけた。

| コホート | n | 要求時に走行中 | p50 | p95 | max | min | 協調終了 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| C5-N | 30 | **29/30** | 12.9 ms | 34.5 ms | 46.5 ms | 0 ms | 30/30 |
| C8-N | 27 | **27/27** | 37.2 ms | 62.8 ms | 72.2 ms | 30.6 ms | 27/27 |

走行中だけの部分母集団:

| コホート | n | p50 | p95 | max | min |
| --- | ---: | ---: | ---: | ---: | ---: |
| C5-N | 29 | 13.1 ms | 34.6 ms | 46.5 ms | 1.6 ms |
| C8-N | 27 | 37.2 ms | 62.8 ms | 72.2 ms | 30.6 ms |

C5 の 1 件（走行中でない）は、要求の直前に Worker が終了していた（待機 0 ms）。
C8 の max 72.2 ms は、停止要求が `cache_save_evict`（約 95 ms、内部キャンセル点なし）の
実行中に出た例である。

---

## 4. 結果（切替負荷）— 走行中を残せない理由

| コホート | n | 要求時に走行中 | stopThread 待機 | ロード開始→shutdown の遅延 |
| --- | ---: | ---: | ---: | --- |
| C5-S1 | 30 | 0/30 | 0〜2 µs | p50 1,478 ms（1,273〜1,732） |
| C5-S2 | 30 | 0/30 | 0〜1 µs | p50 1,373 ms（1,243〜1,886） |
| C8-S1 | 24 | 0/24 | 0〜1 µs | p50 2,727 ms（1,504〜3,111） |
| C8-S2 | 6 | 0/6 | 0〜1 µs | p50 2,923 ms（0〜3,120） |

比較: 通常負荷の遅延は C5-N 208 ms、C8-N 401 ms。

つまり切替負荷では、**shutdown の到着が 1.2〜3.1 秒遅れる**。LoaderThread の実所要は約 240 ms、
UpgradeThread は約 500〜750 ms なので、到着した時点で Worker は既に終了している。

遅延の原因は切替そのものである。

- EQ gain の切替は `submitRebuildIntent` を誘発し、構造再構築（メッセージスレッド上の同期処理）を走らせる。
- バイパス切替だけでも、5 ms 間隔の連続切替が change 通知と crossfade 処理を積み、終了タイマの処理を遅らせる。
- 切替時間を 2,000 ms から 260 ms に短縮しても遅延は約 1.36 s のままで、**継続時間ではなく切替の発生自体**が効いている。

---

## 5. 無効標本（除外理由）

C5-S3 は 6 秒 IR を使ったが、**標本 0 件**だった。ロードが即座に拒否されている。

- 6 秒 × 384 kHz = 2,304,000 samples。
- IR の受理上限は `2^21 + 1 = 2,097,153` samples（`IRLoadAdmission` の凍結値。`CacheManager.h:14` のコメントと同値）。
- したがって 6 秒 IR は上限超過で拒否され、LoaderThread は変換に入らない（probe に `resample` 等のスパンが出ない）。

この条件は「より長い IR で走行中を確保する」目的だったが、上限のため成立しない。
C8-S3 も同じ試行設計で upgrade スパンが 0 件だったため、集計から除外する。

---

## 6. 記録項目（Owner 指定）

| 項目 | 記録状況 |
| --- | --- |
| Worker 種別と識別情報 | `kind`（loader / upgrade）× `w` |
| 停止要求時点の走行状態 | `request` の `already_exited`（stopThread 直前の要求と 1 対 1 対応） |
| 要求→最初のキャンセル確認 | `first_check` の `req_to_first_check_us`（C5-N p50 約 12 ms） |
| `stopThread()` の待機時間 | `span *_stopThread` |
| 協調終了の成否 | `mark *_stopThread_result`（全標本 1） |
| 強制終了・timeout | 0 件（`stopThread` タイムアウトなし。500 ms / 2000 ms の範囲内） |
| IR / FFT / phase / cache 状態 | `loadIR_begin`（targetFFT, phase）、`upgrade_start`（appliedFft 512/4096 = 冷/温） |
| 音声負荷条件 | コホートで分離（通常 / EQ 切替 / bypass 切替） |
| EQ・バイパス状態 | 切替コホートは 5 ms 間隔の交互切替、通常は固定 |
| 試行数・p50・p95・max | §2・§3 |

---

## 7. `B_pre` への含意

- 通常負荷の shutdown 時停止待ちは、C5 max 46.5 ms、C8 max 72.2 ms（n=29/27）。
- 切替負荷では停止待ちの標本が取れない（0〜2 µs）。代わりに **shutdown 到着の遅延 1.2〜3.1 秒**が観測された。
- `B_pre` は「Worker の停止を待つ時間」の上限であり、到着遅延は別の項である。
  到着遅延は **有界でない**（切替の量・内容に依存）。この項は `B_pre` では吸収できない。
- したがって `B_pre` の候補 500 ms は、**通常負荷の実測（max 72.2 ms）に対しては余裕がある**が、
  切替負荷の停止待ちを測れていない以上、確定はできない。**引き続き保留**とする。

---

## 8. 再現手順

```
rem 通常負荷（C5）
ConvoPeq.exe --cli-run --cli-measure-ir <3秒IR> --cli-measure-late-load-at-ms 5000 ^
             --cli-exit-ms 5200 --cli-log-file <log>

rem 切替負荷（bypass）
ConvoPeq.exe --cli-run --cli-measure-ir <3秒IR> --cli-measure-late-load-at-ms 5000 ^
             --cli-exit-ms 5200 --cli-measure-bypass-from-ms 4980 --cli-measure-bypass-span-ms 2000 ^
             --cli-log-file <log>

rem EQ 切替は --cli-measure-bypass-from-ms で EQ gain も切り替える旧実装（本計測の C5-S1 / C8-S1）
```

計測用フラグ（すべて既定 OFF・`--cli-run` 時のみ）:
`--cli-measure-ir` / `--cli-measure-async-load-count` / `--cli-measure-async-load-interval-ms` /
`--cli-measure-enable-upgrade` / `--cli-measure-late-load-at-ms` /
`--cli-measure-bypass-from-ms` / `--cli-measure-bypass-span-ms`。

---

## 9. 生データ

`probe/` に同梱（`MANIFEST.md` に SHA-256 と行数）。`m6_*` は EQ 切替を含む第 1 群、
`m7_*` は bypass のみの第 2 群、`m8_*` は無効標本（§5）。

---

## 10. 停止条件の確認

- 変更は計測専用（スパン 2・マーク 2・CLI フラグ 2 追加）。既定 OFF。
- shutdown 本体・状態機械・停止 API・所有権・破棄順序は変更していない。
- git の commit / push を行っていない。HEAD は `173e6995…`。
