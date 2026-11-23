# 停止レイテンシ実測（W2 / W5 / W6 / stopLearning）

- 作業日時: 2026-11-13 06:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（HEAD = origin/main、作業ツリー）
- 種別: **計測専用変更**（Owner 承認 2026-11-13「U-1・U-2 の証拠取得に必要な計測専用変更に限定」）。
  shutdown 本体・状態機械・停止 API・所有権・破棄順序・`shutdownAudioOnlyKeepingState()` の実装は行っていない。
- 前提: 計測は既定 OFF（`CONVOPEQ_STOP_PROBE`）。通常実行では何も動かない。
- 対象: W2 `coordinatorLoop` / W5 `LoaderThread` / W6 `ProgressiveUpgradeThread` / `stopLearning()`。

---

## 1. 承認範囲との対応

| 承認対象 | 本報告での実施 |
| --- | --- |
| W2 の停止要求・待機・終了確認のスパン | `coord_stop` スパン + `coord_stop_result`（join 成否） |
| W5 の走行中 Worker 停止時間 | 走行中 LoaderThread を停止させ、`request` → `run_exit` を 35 標本 |
| W6 の走行中 Worker 停止時間 | 走行中 UpgradeThread を停止させ、`request` → `run_exit` を 30 標本 |
| `stopLearning()` の join スパン | terminal pass で計測（学習 OFF 18 標本 / 学習 ON 4 標本） |
| 再現用 opt-in 経路 | `--cli-measure-*` の 4 フラグ（既定 OFF、`--cli-run` 時のみ） |

`reconfigure` pass は CLI からデバイス切替を起こせないため未計測（§7）。

---

## 2. 計測実装（追加したもの）

### 2.1 probe の拡張（`src/convolver/ConvolverStopProbe.h` は既存。呼出しを追加）

| 追加位置 | 記録 | 内容 |
| --- | --- | --- |
| `audioengine/AudioEngine.Threading.cpp` `shutdownCoordinatorLoop()` | `span coord_stop` | `stopLoop()`（`signalThreadShouldExit` + `stopThread(2000)`）の所要 |
| 同 | `mark coord_stop_result a=joined` | 1 = 協調終了、0 = タイムアウト（強制終了） |
| `audioengine/AudioEngine.Processing.ReleaseResources.cpp` terminal pass | `span stopLearning_terminal` | learner join の所要 |
| 同 reconfigure pass | `span stopLearning_reconfigure` | 同上（今回未到達） |
| `convolver/ConvolverProcessor.StateAndUI.cpp` `forceCleanup()` | `mark loader_stopThread a=stopped` | 1 = 協調終了、0 = `stopThread(500)` タイムアウト |
| `convolver/ConvolverProcessor.LoadPipeline.cpp` `stopUpgradeThread()` | `mark upgrade_stopThread a=stopped` | 1 = 協調終了、0 = `stopThread(2000)` タイムアウト |

W5/W6 の `request` / `first_check` / `run_exit` は前回の計測で追加済みのものをそのまま使う。

### 2.2 opt-in 経路（`MainWindow::runCommandLineAutomation`）

| フラグ | 動作 |
| --- | --- |
| `--cli-measure-ir <path>` | 計測用の IR。**通常の `--cli-ir` は使わない**（既存の遅延ロードと混ざらないため） |
| `--cli-measure-async-load-count N` | 300 ms 後から N 回、`interval` 間隔でロードを発行する |
| `--cli-measure-async-load-interval-ms M` | 発行間隔（既定 50 ms 以上） |
| `--cli-measure-enable-upgrade` | 発行の直前に `setEnableProgressiveUpgrade(true)` を立て、`loadIR()` を呼ぶ |

`--cli-measure-enable-upgrade` が無い場合、ロードは `loadImpulseResponse(file, false)`（非同期 LoaderThread）で発行する。
有る場合は `loadIR(file)`（同期ロード + `startProgressiveUpgrade`）を発行する。
いずれも `juce::Timer::callAfterDelay` + `SafePointer` で行い、通常実行では 1 度も通らない。

### 2.3 計測環境

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 x64 |
| ビルド | Release / MSVC 14.52.37025 / Ninja Multi-Config |
| 処理サンプルレート | 384,000 Hz |
| 目標 FFT | 4096（low-res 512） |
| ブロック | 1024 |
| probe | `CONVOPEQ_STOP_PROBE=1`、出力は JSON Lines |

---

## 3. 結果

数値はすべて**マイクロ秒**。**max は観測値であり保証値ではない。**

| 対象 | 試行数 n | p50 | p95 | max | min | 強制終了 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| W2 `coord_stop`（`stopThread(2000)`） | 18 | 1,912 | 4,239 | 4,443 | 28 | 0 件（join 成功 18/18） |
| `stopLearning()` terminal（学習 OFF） | 18 | 4 | 8 | 22 | 3 | − |
| `stopLearning()` terminal（学習 ON） | 4 | 2,828 | 9,125 | 10,223 | 2,435 | − |
| W5 `LoaderThread` `request`→`run_exit` | 35 | 23,059 | 29,137 | 31,320 | 20,169 | 0 件（協調終了 5/5） |
| W6 `ProgressiveUpgradeThread` `request`→`run_exit` | 30 | 33,576 | 81,350 | 86,372 | 2,111 | 0 件（協調終了 30/30） |

### 3.1 W2（`coordinatorLoop`）

18 標本すべて `stopThread(2000)` の範囲内（max 4.4 ms）で協調終了した。タイムアウト・強制終了は 0 件。
`stopLoop()` は `signalThreadShouldExit()` の直後に `stopThread(2000)` を呼ぶため、この値は
「シグナル → スレッド終了」の実測に相当する。

### 3.2 W5（`LoaderThread`）

35 標本。`request`（`signalThreadShouldExit`）から `run_exit` までが 20.2〜31.3 ms に収まった。
内訳は `req_to_first_check_us`（p50 21,533 / max 29,714）が支配的で、**次のチェックポイントまでの待ち**である。
チェックポイントはステップ境界にあり、1 ステップの所要がそのまま下限になる。

観測された 1 ステップの所要（同一条件のスパン）:

| 処理 | 実測 |
| --- | --- |
| `resample` | 29.7〜32.2 ms（`cancelled=0`：中断点なしで完走） |
| `minimum_phase`（キャンセル時） | 79.9〜84.9 ms の途中で `cancelled=1` を返して中断 |
| `mixed_phase`（完走時） | 1,780 ms（3 秒 IR の最後の 1 本） |

### 3.3 W6（`ProgressiveUpgradeThread`）

30 標本で 2.1〜86.4 ms。**分布が二峰性**である。

- 小さい側（2.1〜15.7 ms）: `convertToHighRes`（13.2〜30.7 ms）の途中、または FFT ステップの境界で止まった例。
- 大きい側（17.7〜86.4 ms）: `cache_load`（93.4〜96.7 ms）または `cache_save_evict`（99.1〜105.0 ms）の
  実行中に要求が出た例。**この 2 つは内部にキャンセル点が無い**ため、完了するまで止まれない。

したがって W6 の停止レイテンシの上限は「実行中の最も長い中断不能処理の残り時間」で決まる。
観測最大 86.4 ms は `cache_save_evict` の残り時間に対応する。

### 3.4 `stopLearning()`

- 学習 OFF（18 標本）: 3〜22 µs。learner スレッドが動いていないため join は即時。
- 学習 ON（4 標本）: 2.4〜10.2 ms。`--cli-start-learning` で学習を開始した状態。
- **reconfigure pass は未計測**（§7）。

---

## 4. 記録項目（Owner 指定）

| 項目 | 記録状況 |
| --- | --- |
| Worker 種別と識別情報 | `kind`（loader / upgrade）× `w`（オブジェクトアドレス） |
| 停止要求時刻 | `request` レコードの `ts` |
| wait 開始・終了時刻 | スパンの `ts - us` と `ts`（W2 / stopLearning） |
| 実際の終了確認時刻 | `run_exit` の `ts`、および `coord_stop_result` / `*_stopThread` |
| timeout の有無 | `coord_stop_result` の `a`（0 = timeout）、`*_stopThread` の `a` |
| 強制終了の有無 | 上記の 0 件（全標本で協調終了） |
| 計測環境と再現条件 | §2.3 と §6 |
| 試行回数 | §3 の n |

---

## 5. `B_pre` への含意

- W2・W5・W6 はいずれも**数百 ms 以内**に協調終了する。観測上限は W6 の 86.4 ms。
- ただし **max は観測値**であり、保証値ではない。`B_pre` は「この時間まで待ち、超えたら保持へ移る」判断値である。
- 暫定 5000 ms は観測上限の約 58 倍で、安全性の観点では過大（保持へ落ちる機会を不必要に遅らせる）。
  実測に基づく候補は **500 ms**（W6 max 86.4 ms の約 5.8 倍）である。
  ただし W1（`rebuildThread`）は停止が有界でなく（`LoopExited` までしか確認できない）、
  W5/W6 の分布も負荷依存で伸び得るため、**500 ms を採用する判断は Owner の承認事項**とする。
- 参考: 現行の `stopThread` 予算は W5 が 500 ms、W6・W2 が 2000 ms。実測はすべてこの範囲に収まった。

---

## 6. 再現手順

```
rem 1) ビルド（probe は既定 OFF）
cmd /c C:\Users\user\AppData\Local\Temp\opencode\build_probe3.bat

rem 2) W2 / stopLearning（通常の終了経路を 1 回通すだけ）
set CONVOPEQ_STOP_PROBE=1
set CONVOPEQ_STOP_PROBE_FILE=<out>.jsonl
ConvoPeq.exe --cli-run --cli-exit-ms 3000 --cli-log-file <log>

rem 3) W5（走行中 LoaderThread を作ってキャンセル）
ConvoPeq.exe --cli-run --cli-measure-ir <3秒IR> --cli-measure-async-load-count 8 ^
             --cli-measure-async-load-interval-ms 120 --cli-exit-ms 6000 --cli-log-file <log>

rem 4) W6（走行中 UpgradeThread を作ってキャンセル）
rem    ※ 4096 のキャッシュが温まると upgrade が起動しないため、(IR × phase) を毎回変える
ConvoPeq.exe --cli-run --cli-measure-ir <実IR> --cli-phase Mixed ^
             --cli-measure-async-load-count 6 --cli-measure-async-load-interval-ms 350 ^
             --cli-measure-enable-upgrade --cli-exit-ms 9000 --cli-log-file <log>
```

---

## 7. 未計測（明示）

| 項目 | 理由 | 再現に必要なこと |
| --- | --- | --- |
| `stopLearning()` の reconfigure pass | CLI からデバイス切替（SR/BS 変更・デバイス再列挙）を起こせない | デバイス切替を伴う CLI 手順、または計測専用の強制 reconfigure |
| W1（`rebuildThread`）の join 時間 | `rebuildPendingAfterLoad` を CLI から発火できない | 専用スイッチ（未承認）または ring buffer overflow の再現 |
| `closeAudioDevice()` の所要 | 終了経路の最後で計測点を置いていない | 追加の計測点（本承認の範囲外） |
| 負荷をかけた状態での W5/W6 | 音声コールバック＋高負荷の同時実行を CLI で作っていない | 負荷生成の手順が必要 |

---

## 8. 生データ

`probe/` に同梱（`MANIFEST.md` に SHA-256 と行数）。

| ファイル | 内容 |
| --- | --- |
| `m2_w2.jsonl` | W2 + stopLearning terminal（12 試行） |
| `m2_learn.jsonl` | 学習 ON の stopLearning（4 試行） |
| `m2_w5.jsonl` | W5（5 試行 / 35 標本） |
| `m1_E4.jsonl` | W6 探索（1 試行 / 5 標本） |
| `m3_w6.jsonl` | W6 本計測（6 試行 / 25 標本 + W2/stopLearning 各 6） |

---

## 9. 停止条件の確認

- 変更したのは計測専用コード（probe 呼出し 6 箇所と CLI 4 フラグ）のみ。既定 OFF。
- shutdown 本体・状態機械・停止 API・所有権・破棄順序は変更していない。
- git の commit / push を行っていない。HEAD は `173e6995…`。
