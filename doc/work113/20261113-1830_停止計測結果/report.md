# shutdown v2-a: 停止時間の計測結果（M-1〜M-4）

- 作業日時: 2026-11-13 18:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **計測のみ**。承認された M-1／M-3 のログ追加と M-2 の計測専用スイッチは**実装していない**（§2 に理由）。
- 判定: **計測完了**（W1・W2・`stopLearning`・走行中 W5/W6 を取得）。v2-a の実装 GO は未付与のまま。

---

## 1. 結論

1. 承認された 3 点のうち「計測」は完了した。W1・W2・`stopLearning()`・走行中 W5・走行中 W6 の停止時間を実測した。
2. **コードの追加は不要だった。** 既存の停止プローブ（`src/convolver/ConvolverStopProbe.h`、`CONVOPEQ_STOP_PROBE=1`）と
   既存の CLI 計測スイッチ（`--cli-measure-ir` / `--cli-measure-late-load-at-ms` / `--cli-measure-enable-upgrade`）で
   すべて取得できた。M-1／M-3 のログ 2 行と M-2 のスイッチは実装していない（§2）。
3. **通常終了でも毎回 2 秒の drain タイムアウトが発生している。** C 全体（約 2.4 秒）の大半はこれで、
   Worker 停止（最大 94 ms）ではない。`waitForDrain(2000, 2)` がタイムアウトし、
   ログに `[DIAG] releaseResources: drain timeout reached` が出る（3 回中 3 回）。
4. 実測最大の停止時間は **W6 の 94.3 ms**。`bound` の暫定候補 500 ms は、この最大値に対して 5 倍以上の余裕がある。

## 2. なぜコードを追加しなかったか

| 承認された作業 | 実装 | 理由 |
| --- | --- | --- |
| M-1（W2 の停止時間のログ 2 行） | **不要** | `shutdownCoordinatorLoop()` には既に `convo::probe::Span("coord_stop")` がある（`AudioEngine.Threading.cpp:380`） |
| M-3（`stopLearning()` のログ 2 行） | **不要** | 既に `Span("stopLearning_terminal")` がある（`ReleaseResources.cpp:150`） |
| M-2（走行中 W5/W6 を作るスイッチ） | **不要** | `--cli-measure-ir` ＋ `--cli-measure-late-load-at-ms` で走行中を作れる（`MainWindow.cpp:478-573`、既定 OFF） |
| M-4（W1） | 不要 | 既存の `[DIAG] releaseResources: before/after stopRebuildThread` で取得 |

プローブは `CONVOPEQ_STOP_PROBE=1` のときだけ動き、未設定なら `enabled()` の静的 bool 参照だけで完全に no-op である
（`ConvolverStopProbe.h:37-45`）。製品挙動は変わらない。

## 3. 計測方法

- プローブ出力: `CONVOPEQ_STOP_PROBE=1`、`CONVOPEQ_STOP_PROBE_FILE=<temp>`（JSON Lines、`ts` は steady_clock の µs）。
- 走行中 W5: `--cli-measure-ir sampledata/test_music.wav --cli-measure-late-load-at-ms 1995 --cli-exit-ms 2000`。
- 走行中 W6: `--cli-measure-ir build/ir_5s.wav --cli-measure-enable-upgrade --cli-measure-late-load-at-ms 1500 --cli-exit-ms 2000`。
  `build/ir_5s.wav` は 5 秒・48 kHz・float32 の減衰ノイズで、計測用に一時生成した（計測後に削除済み）。
- W1 と C 全体: 既存ログのマーカーを外部から 5 ms 間隔でポーリングして時刻を付けた。

## 4. 実測結果

### 4.1 Worker 停止時間

| 対象 | 計測値 | 件数 | 取得元 |
| --- | --- | ---: | --- |
| W1 `rebuildThread` | 0〜59 ms | 6 | ログ `before/after stopRebuildThread` |
| W2 `coordinatorLoop_` | 39〜707 µs | 12 | `coord_stop` span |
| `stopLearning()`（terminal） | 5〜27 µs | 12 | `stopLearning_terminal` span |
| W5 `activeLoader`（走行中） | 5.2〜27.9 ms | 5 | `loader_stopThread` span / `req_to_exit_us` |
| W6 `upgradeThread`（走行中） | 2.5〜94.3 ms | 4 | `upgrade_stopThread` span / `req_to_exit_us` |

W6 の最大 94.3 ms は、停止要求時に `cache_save_evict`（235.7 ms）の途中だったためで、
`req_to_first_check_us = 92.3 ms`（＝停止フラグを見るまでに 92 ms かかった）と整合する。
**停止の遅れはポーリング間隔ではなく、中断できない区間の長さで決まる。**

### 4.2 teardown 全体（1 回分の内訳）

| マーカー | 時刻 |
| --- | ---: |
| `SHUTDOWN_BEGIN` | 3,877 ms |
| `step 4 setProcessor(nullptr)` | 3,891 ms |
| `releaseResources: enter` | 3,923 ms |
| `before/after stopRebuildThread`（W1） | 3,923 ms（0 ms） |
| `before/after ui processor release`（W5/W6） | 3,955 → 3,985 ms（30 ms） |
| `step 6 saveSettings` | 6,016 ms |
| `step 8 closeAudioDevice` | 6,062 ms |
| `step 10 UI reset` | 6,131 ms |
| `SHUTDOWN: mainWindow.reset() completed` | 6,301 ms |

- C 全体（SHUTDOWN_BEGIN → reset completed）は **2,424 ms**。
- そのうち **UI release 後から step 6 までに 2,031 ms** の空白がある。これが `waitForDrain(2000, 2)` の
  タイムアウト（`ReleaseResources.cpp:653`）であり、ログの `drain timeout reached` と一致する。
- Worker 停止の合計は 0.1 秒未満である。**C の 8 割以上は drain の待ち時間**で、停止処理そのものではない。

### 4.3 通常／走行中の比較

| 条件 | 総経過 | C 全体 | 備考 |
| --- | ---: | ---: | --- |
| 通常（3 回） | 6,259〜6,567 ms | 2,377〜2,423 ms | 差はノイズの範囲 |
| W5 走行中（3 回） | 6,265〜6,382 ms | 2,331〜2,423 ms | 停止は 6〜28 ms |
| W6 走行中（4 回） | 5,938〜6,319 ms | 2,364〜2,392 ms | 停止は 2.5〜94.3 ms |

走行中の Worker がいても C 全体はほぼ変わらない（drain の 2 秒が支配的なため）。

## 5. `bound = 500 ms` の評価

- 実測最大の停止は W6 の 94.3 ms。`clamp(1.5 × p99, 500 ms, 5000 ms)` を当てると、p99 が 94 ms 前後でも
  `1.5 × 94 = 141 ms` となり**下限 500 ms** に張り付く。
- 500 ms は最大値の約 5.3 倍であり、暫定候補として妥当である。
- ただし次の条件は未計測であり、**確定値としては扱わない**:
  - W6 の中断できない区間（`cache_save_evict`）がさらに長くなる条件（大きい IR・遅いディスク）。
  - W3（`WorkerThread`）・W4（`RetryScheduler`）・W7（`DeferredFreeThread`）の停止時間。
  - 停止しなかった Worker が存在する状態（人工 stuck）での挙動。
- 500 ms を超えた Worker を「停止した」と扱わない方針（Owner 判断）は、本書の計測でも前提として維持している。

## 6. 未計測・未実施

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | W3・W4・W7 の停止時間 | U-3 の対象。v2-a の preflight 対象外であり、今回の範囲外 |
| 2 | W6 の最悪条件（大きい IR・遅いディスク） | 5 秒 IR のみ。より長い IR は DSP の上限（5.461 s @384 kHz）を超えて変換に失敗する |
| 3 | 人工 stuck 試験 | 未承認（計画は `20261113-1630_U-3整合` §8） |
| 4 | `closeAudioDevice()` の無界性 | 既存挙動（U-13）。計測しても上限は保証できない |
| 5 | drain タイムアウトの原因 | 本課題の範囲外（§7 に事実のみ記載） |

## 7. 追加で判明した事実（今回の範囲外・報告のみ）

- 通常終了で **毎回 2 秒**の drain タイムアウトが発生している（3 回中 3 回）。
  `waitForDrain(2000, 2)` が quiescence に到達せず、タイムアウト後に `tryReclaim` へ落ちる。
- これは v2-a の `B_pre` とは独立した固定コストであり、終了時間を短くするには drain 側の調査が必要になる。
  原因の特定・修正は本課題の対象外なので、事実の記録にとどめる。

## 8. 残留リスク

- 計測はローカル環境 1 台・各条件 3〜4 回である。分布の裾（外れ値）は捉えきれていない。
- プローブは既定 OFF だが、有効化すると `std::mutex` と `fprintf` が停止経路に入る。
  計測値は「プローブ有効時の値」であり、無効時より僅かに長い可能性がある（W2 が 0.7 ms まで出た例がある）。
- 走行中 W6 の 94 ms は 1 例であり、再現性は確認していない。

## 9. テスト結果

- 今回の計測で**ソースは変更していない**（`git status` の追跡ファイル 42 件は前回までの変更のまま）。
- 直近の全件テストは、プリセット保存 UI の実装後に行った `ctest -C Release` で **61/61 成功（328.47 秒）**。
  その後にソース変更は無いため、この結果が現在のツリーに対応する。

## 10. 判定

**計測完了。** M-1〜M-4 の値を取得し、`bound` の暫定候補 500 ms の妥当性を確認した。
v2-a の実装 GO は未付与のまま据え置く。追加で判明した 2 秒の drain タイムアウトは、
v2-a とは別の課題として記録した。
