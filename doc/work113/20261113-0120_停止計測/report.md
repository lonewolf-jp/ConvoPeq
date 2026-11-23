# 停止レイテンシ計測（計測専用変更・既定 OFF）

- 作業日時: 2026-11-13 00:20 JST（再計測 00:50）
- 対象: `C:\VSC_Project\ConvoPeq`（HEAD = origin/main = `173e6995`、Release ビルド）
- 種別: **計測専用変更のみ**（Owner 承認 2026-11-13）。shutdown protocol の実装、`requestTerminalRelease()` の意味変更、
  admission 制御変更、`releaseResources()` 変更、Worker の停止・所有権構造変更、IRState guard 実装は行っていない。
- 本報告は「計測結果」と「そこから言えること・言えないこと」だけを扱う。設計改訂版は別報告とする。

---

## 1. 結論（先に）

1. **走行中 Worker への停止要求は 1 件も観測されなかった。** CLI 自動化で到達できる経路に、走っている
   `LoaderThread` / `ProgressiveUpgradeThread` が存在しないためである（§4）。したがって
   「停止要求 → `run()` 終了」の分布は本構成では取得できず、**preflight の `bound` の根拠にはできない**。
2. 代わりに、**中断できない区間の実測値**が取れた。実 IR の同期ロードは 1 回 **102〜144 ms**、
   同期リビルドは **334〜638 ms**（192,000 samples / FFT 2048）で、いずれも途中で止まらない（全スパン `cancelled=0`）。
3. **自動終了予定を 150 ms にしても、プロセスは 751 ms まで終了しなかった。** Message Thread 上で走る
   ロード／リビルドは shutdown の開始そのものを遅らせる。Message Thread で待機する preflight は、
   この区間では実行すらされない。
4. `queueFinalizeOnMessageThread` の finalize は `callAsync` の fire-and-forget で、ワーカーは Message Thread を
   待たない（`ConvolverProcessor.LoaderThread.cpp:288-317`）。待機によるデッドロックは起きないが、
   **遅延実行された finalize が shutdown 後に走り得る**。`weakOwner` が生きていれば実行されるので、
   generation 判定だけでは抑止にならない（Owner 指摘 C の裏付け）。
5. `ProgressiveUpgradeThread` は `enableProgressiveUpgrade` の既定が false（`ConvolverProcessor.h:91`）で、
   CLI 構成では起動しない。有効化は `AudioEngine.Parameters.cpp:822` 経由のみ。同スレッドの停止時間は未計測。

---

## 2. 計測実装

### 2.1 追加したヘッダ

`src/convolver/ConvolverStopProbe.h`（新規・ヘッダオンリー）

- `convo::probe::enabled()` … `CONVOPEQ_STOP_PROBE` が未設定または `0` なら false。**既定 OFF**。
- 出力先 … `CONVOPEQ_STOP_PROBE_FILE`（既定はカレントディレクトリの `convopeq-stop-probe.jsonl`）。
- 記録形式 … JSON Lines。`ts` は `steady_clock` のマイクロ秒（絶対値）。毎行 flush。
- 記録種別 … `run_start` / `first_check` / `request` / `run_exit` / `span` / `mark`。
- 有効時のみ `std::mutex` と `std::fprintf` を使う。無効時は静的 bool の参照だけ。
- **音声スレッドには計測呼び出しを一切置いていない**（挿入先はローダー、アップグレード、Message Thread のみ）。

### 2.2 挿入点（削除手順用の棚卸し）

| ファイル | 行 | 内容 |
| --- | --- | --- |
| `src/convolver/ConvolverStopProbe.h` | 全体 | 新規（ファイル削除で元に戻る） |
| `src/convolver/ConvolverProcessor.LoaderThread.cpp` | 14 | include |
| 同 | 45 | `~LoaderThread` で `noteStopRequest` |
| 同 | 54-60 | `run()` の `ProbeExitMark` / `noteRunStart` |
| 同 | 134 | `performLoad` 冒頭の `noteMark("perform_load", thread!=nullptr)` |
| 同 | 140 | `threadShouldExit()` 初回観測 |
| 同 | 238 | `Span("nuc_build")`（`initializeConvolverSynchronously` 全体） |
| 同 | 666 | `Span("resample")`（ラムダ内・タイト） |
| 同 | 897 | `Span("minimum_phase")`（ラムダ内・タイト） |
| 同 | 918 | `Span("mixed_phase")`（ラムダ内・タイト） |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | 10 | include |
| 同 | 23 | `noteMark("load_request", isRebuild, optimizeForRealTime)` |
| 同 | 55 | `noteStopRequest(activeLoader)` |
| 同 | 270 | `noteMark("loadIR_begin", targetFFT, phase)` |
| 同 | 278 / 300 | `Span` cache_direct / cache_low |
| 同 | 324 / 327 | `Span` convert_low_outer（外側）/ convert_low（変換のみ） |
| 同 | 349 | `Span` cache_save_low |
| 同 | 361 | `Span` apply_ir |
| 同 | 370 | `noteMark("upgrade_start", appliedFft, targetFFT)` |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | 10 | include |
| 同 | 1053 | `noteStopRequest(loader)`（`forceCleanup` の `stopThread(500)` 直前） |
| `src/ProgressiveUpgradeThread.cpp` | 15 | include |
| 同 | 57 / 67 | `cancel()` の要求、`isGenerationValid()` の初回観測 |
| 同 | 87-89 | `run()` の `ProbeExitMark` / `noteRunStart` |
| 同 | 126 / 140 / 167 | `Span` cache_load / convertToHighRes / cache_save_evict |

「★ gate measurement」または `convo::probe::` で機械的に検索できる。既存の未コミット変更には触れていない。

### 2.3 有効化と実行

```
set CONVOPEQ_STOP_PROBE=1
set CONVOPEQ_STOP_PROBE_FILE=<出力先>
ConvoPeq.exe --cli-run --cli-ir <IR> --cli-phase <AsIs|Minimum|Mixed> [--cli-rebuild] ^
             [--cli-ir-reload-count N --cli-ir-reload-interval-ms M] ^
             --cli-exit-ms <ms> --cli-log-file <log>
```

ビルドは `build_probe3.bat`（`vcvars64.bat` → PATH に MSVC 14.52 の bin を前置 → `MKLROOT`/`IPPROOT` 設定 →
`cmake --build build --config Release --target ConvoPeq`）。既存 `build/` を再利用した。

---

## 3. 計測環境

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 x64 |
| コンパイラ | MSVC 14.52.37025（`CMAKE_CXX_COMPILER` 実測値） |
| ジェネレータ | Ninja Multi-Config |
| 構成 | Release（`ConvoPeq_artefacts/Release/ConvoPeq.exe` 48,010,240 B / 2026-11-13 00:15:53 時点） |
| 処理サンプルレート | 384,000 Hz（CLI ログの実測値） |
| 目標 FFT | 4096（low-res は 512） |
| ブロックサイズ | 1024 |

---

## 4. 測定結果から確定した経路構造

計測前の前提「IR ロードは非同期 Worker」は、この構成では成立していなかった。実測で確定した事実は次の 4 点。

1. **`ConvolverProcessor::loadIR()` は Message Thread で同期実行**（`LoadPipeline.cpp:237-239` に `JUCE_ASSERT_MESSAGE_THREAD`）。
   変換（`irConverter->convertFile`）、キャッシュ入出力、`applyComputedIR` まで同じスレッドで走る。
   `loadIR_begin` と `upgrade_start` の間に `run_start` は 1 件も無い。
2. **`loadImpulseResponse(file)` の本番呼出しは 1 箇所だけ**（`Lifecycle.cpp:220`。`rebuildPendingAfterLoad` を
   timer で拾う経路）。UI からの IR ファイル指定は `loadIR()` を通る。
3. **`rebuildAllIRsSynchronous` はスタック上の `LoaderThread` を `runSynchronously()` する**
   （`Rebuild.cpp:88-95` → `performLoad(nullptr)`）。したがって `run()` を通らず `run_start`/`run_exit` は出ない。
   `--cli-rebuild` がこの経路。計測では `perform_load a=0`（thread == nullptr）として観測された。
4. **`ProgressiveUpgradeThread` は起動しない**。`startProgressiveUpgrade()` は `isProgressiveUpgradeEnabled()`
   が false なら即 return（`LoadPipeline.cpp:120-121`）し、既定値は false（`ConvolverProcessor.h:91`）。
   `upgrade_start` の mark は出るが `run_start` は 0 件。

したがって、シナリオ A〜F・N・S・T の `request` レコードは **一度も走っていない `~LoaderThread`**（スタック
オブジェクト）のものであり、停止レイテンシの標本ではない。`already_exited=0` は「走っていない」ことの反映にすぎない。

---

## 5. 実測値

数値はすべて ms。`span` の範囲は「その操作の直前から直後まで」に締めてある（§5.1 の `convert_low_outer` だけは
外側ブロック全体）。

### 5.1 同期ロード（`loadIR`、Message Thread）

| シナリオ | IR | 位相 | cache_direct | cache_low | convert_low | cache_save_low | apply_ir | `loadIR` 合計 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | `25t_ir_wgw.wav`（キャッシュ温） | Mixed | 0.16 | 49.4 | — | — | （注） | **115.3** |
| Q | `ir_shoe_wgw.wav`（キャッシュ冷） | Mixed | 0.13 | 0.11 | **19.8** | 60.8 | **59.0** | **144.0** |
| R | `25t_ir_treeverb.wav`（キャッシュ冷） | Minimum | 0.12 | 0.09 | **15.7** | 30.7 | **54.4** | **102.6** |

（注）キャッシュ温の経路（`LoadPipeline.cpp:282` / `:309` の `applyComputedIR`）には `apply_ir` のスパンを
置いていないため内訳は取れない。合計から逆算すると約 63 ms。

読み方:

- 変換そのものは 16〜20 ms しかない。**重いのはキャッシュ書き込み（31〜61 ms）と `applyComputedIR`（54〜59 ms）**。
- キャッシュ温でも `cache_low`（分割データの読み戻し）に 49 ms かかる。
- 1 回のロードで **102.6〜144.0 ms** は Message Thread が占有される。

### 5.2 同期リビルド（`rebuildAllIRsSynchronous`、Message Thread）

スコープを締めた直接測定（実 IR を読み込んだ直後のリビルド）。

| シナリオ | IR | minimum_phase | mixed_phase | nuc_build | 総時間（`perform_load` → `nuc_build` 終了） |
| --- | --- | ---: | ---: | ---: | ---: |
| S | `25t_ir_wgw.wav` | 103.4 | 206.3 | 14.6 | **334.2** |
| T | `25t_ir_wgw.wav`（終了予定 150 ms） | 107.1 | 201.4 | 14.6 | **333.4** |

総時間のみの追加標本（スパンのスコープが広い旧版で計測。総時間はスコープに依存しない）。

| シナリオ | IR | 位相 | 総時間 |
| --- | --- | --- | ---: |
| A / B / C / D / F | 既定 IR 192,000 samples | Mixed | 550.6 / 578.8 / 589.0 / 554.1 / 638.1 |
| E | 既定 IR 192,000 samples | Minimum | 225.7 |

読み方:

- 同じ 192,000 samples でも 334 ms と 551〜638 ms の幅がある。差は IR のデータ内容に依存する
  （位相変換の内部反復がデータ依存と見られる）。**データ依存の幅は未解明**なので、上限としては大きい方を採る。
- 全スパンで `cancelled=0`。リビルド中に停止要求は 1 件も発行されていない（Message Thread が塞がっていて
  要求を出す主体が動けない）。

### 5.3 終了予定 150 ms でもリビルドが優先される（シナリオ T）

```
0.0 ms    loadIR_begin（targetFFT=4096, phase=Mixed）
0.5 ms    cache_direct 0.16
46.9 ms   cache_low 44.8
104.7 ms  upgrade_start（512 → 4096）
417.3 ms  perform_load（thread == nullptr = 同期実行）
422.3 ms  minimum_phase 開始
529.4 ms  minimum_phase 終了（107.1）
530.0 ms  mixed_phase 開始
731.4 ms  mixed_phase 終了（201.4）
736.1 ms  nuc_build 開始
750.7 ms  nuc_build 終了（14.6）
750.9 ms  request（未起動 LoaderThread のデストラクタ）
```

終了予定は 150 ms だが、プロセスは 751 ms まで終了しなかった（終了コード 0）。終了予定 5000 ms の S も
751 ms で同じ結果であり、**終了予定の値はロード／リビルドの進行に影響していない**。
shutdown の起点は Message Thread 上のタイマなので、同期区間の間は起点そのものが動かない。

### 5.4 停止要求イベントの集計

| 種別 | 件数 |
| --- | --- |
| 走行中 Worker への `request` | **0** |
| 未起動 `LoaderThread` への `request`（A〜F、N、S、T） | 9（各 1 件、`run_exit` なし） |
| `run_start` / `run_exit` | **0 / 0** |
| `first_check` | 0 |

停止レイテンシ（`req_to_exit_us`）の標本は 0 件である。**分布は算出できない。**

---

## 6. 解釈

### 6.1 `bound` の根拠にできる数値／できない数値

- 根拠にできる: 中断不能区間の実測。実 IR 1 本の同期ロードが 102〜144 ms、192,000 samples の同期リビルドが
  334〜638 ms。いずれも途中で止まらないことを全スパンの `cancelled=0` で確認した。
- 根拠にできない: Worker の停止レイテンシ。走行中の Worker が 1 件も無いため、`bound` を実測で決められない。
  「5 秒で足りる」とも「足りない」とも、この計測からは言えない。

### 6.2 スケーリング

`minimum_phase` / `mixed_phase` は IR 長にほぼ比例する。192,000 samples で 103〜222 ms（最小位相）と
201〜368 ms（混合位相）。3 秒の IR（384 kHz で約 1.15 M samples）を同期リビルドすると、単純比例で
**1 秒台後半〜数秒**になる。1 秒未満の待ちで済むという設計は成立しない。
`kGracefulDrainMaxMs = 5000` を据えるなら、長い IR では**タイムアウト側に落ちる前提**で保持経路（案 C）を
設計する必要がある。

### 6.3 `callAsync` の性質（Owner 指摘 C の裏付け）

`queueFinalizeOnMessageThread` は `callAsync` で finalize を投げて終わり（`LoaderThread.cpp:288-317`）、
ワーカーは Message Thread の完了を待たない。したがって「待つ」設計でデッドロックはしない。
一方、投げられたラムダは `weakOwner.get()` が生きている限り実行される。shutdown 中に Worker が stuck していても、
その finalize が後から走って `finalizeNUCEngineOnMessageThread` を呼び得る。抑止は generation 判定だけでは足りない。

### 6.4 preflight の待機を Message Thread で行うことの問題

シナリオ T が示すとおり、同期ロード／リビルドの間は Message Thread が塞がる。この区間では shutdown の起点
（タイマ、UI 操作、CLI の自動終了）が動かない。したがって「Message Thread で preflight を回して Worker を待つ」
設計は、Worker が動いている場合には有効だが、**同期区間には原理的に間に合わない**。同期区間の対策は別の層
（ロード開始点での admission 閉鎖と、同期処理そのものへの中断点追加）でしか解けない。

---

## 7. 暫定案（`bound` の置き方）

1. `bound` は「Worker 待ち」と「同期区間」で分けて考える。同期区間には待ちを置かない（置いても動かない）。
2. Worker 待ちの `bound` は、走行中 Worker を再現できる構成で測り直すまで確定しない。暫定は既存の
   `kGracefulDrainMaxMs = 5000` を維持し、**タイムアウトしたら保持して先へ進む**（案 C）ことを前提にする。
3. したがって `workersStuck_` の sticky 維持（Owner 指摘 D）と `AudioEngine::releaseResources()` の
   非破壊モード（同 B）は、`bound` の値に関係なく必要である。

---

## 8. 未計測（明示）

| 項目 | 理由 | 再現に必要なこと |
| --- | --- | --- |
| 走行中 `LoaderThread` の request→exit | `loadImpulseResponse(file)` が `rebuildPendingAfterLoad` 経由でしか呼ばれず、CLI から発火できない | バッファサイズ変更／オーバーフローを CLI で起こす、または計測専用 CLI スイッチ（要承認） |
| `ProgressiveUpgradeThread` の停止 | `enableProgressiveUpgrade` が既定 false、CLI で有効化する手段が無い | 設定の一時切替（`AudioEngine.Parameters.cpp:822` 経路）または計測専用 CLI スイッチ（要承認） |
| `stopThread(500)` のタイムアウト発火 | 上記 2 件が走らないため未到達 | 同上 |
| 実 IR の target FFT 4096 への高解像度化 | アップグレードスレッドが動かないため未到達 | 同上 |
| 3 秒以上の長い IR での同期リビルド | 手持ちの実 IR が短い（いずれも 0.5 s 相当に整えられる） | 長い実 IR の用意、またはリビルド対象 IR の差し替え |

補足: 自作の float32 stereo WAV は `loadIR` で `no FFT applied` になり変換に到達しなかった。実 IR（PCM）では成功した。
IR 入力の受理形式の差と思われるが、本報告の対象外（製品仕様の確認は別途）。

---

## 9. 再現手順

```
rem 1) ビルド（probe は既定 OFF なので production と同じバイナリ挙動）
cmd /c C:\Users\user\AppData\Local\Temp\opencode\build_probe3.bat

rem 2) 計測（例: 同期リビルド・終了予定 150 ms）
set CONVOPEQ_STOP_PROBE=1
set CONVOPEQ_STOP_PROBE_FILE=C:\Users\user\AppData\Local\Temp\opencode\p5_T.jsonl
ConvoPeq.exe --cli-run --cli-ir <実IR> --cli-phase Mixed --cli-rebuild ^
             --cli-exit-ms 150 --cli-log-file <log>
```

---

## 10. 付録 — 生データ一覧

| ファイル | 内容 |
| --- | --- |
| `probe/p2_A..F.jsonl` | 同期リビルド（既定 IR 192,000 samples。旧スコープ。総時間のみ使用） |
| `probe/p3_I..M.jsonl` | 実 IR のロード／再ロード（旧スコープ。§5 では未使用、記録として同梱） |
| `probe/p4_N.jsonl` | 終了予定 200 ms + 同期リビルド（旧スコープ。§5.3 と同傾向の確認用） |
| `probe/p5_P.jsonl` | 実 IR 温キャッシュ、Mixed（タイトスコープ） |
| `probe/p5_Q.jsonl` | 実 IR 冷キャッシュ、Mixed（タイトスコープ） |
| `probe/p5_R.jsonl` | 実 IR 冷キャッシュ、Minimum（タイトスコープ） |
| `probe/p5_S.jsonl` | 同期リビルド、終了予定 5000 ms（タイトスコープ） |
| `probe/p5_T.jsonl` | 同期リビルド、終了予定 150 ms（タイトスコープ・§5.3） |
