# B-07 復元要求と公開値の同期（Issue #278）

- 作業日時: 2026-11-24 08:00
- 対象: `main` `080f3bbb`（統合クローン `C:\VSC_Project\integ\ConvoPeq`）
- 関連: Issue #278、`doc/work113/20261123-1951_統合実施/report.md` §1.2

## 結論

復元要求（`NoiseShaperLearner::setState()`）と公開値の間に、実在する競合を 2 件見つけて修正した。3 件目は「安全」と判断し、その根拠をコード上の不変条件として示す。

1. 保存スナップショットの世代混入。学習状態 XML が「ある世代の CMA mean」と「別の世代の係数・スコア」を同時に含み得た。旧実装の実測で 174 回中 173 回が混入した。
2. 復元が worker 専用フィールドを書いていた。`accumulatedPlaybackSeconds` と `currentPhase` はヘッダで「Worker thread専用（非atomic）」と宣言されているのに、`setState()` がメッセージスレッドから直接書いていた（`AudioEngine.Timer.cpp` の復旧経路）。データ競合である。
3. 復元後の世代番号の巻き戻り。worker のローカル `generation` が復元値を引き継がないため、復元直後の世代公開で `progress.iteration` が小さい値に戻っていた。

公開側（`AudioEngine` の係数バンク `bank.state`）は C-03 と `stateMutex` で十分だった。新しい競合は無い。

## 1. Issue の 3 項目の調査結果

### 1.1 学習中の復元は世代境界で適用されるか

sampling、評価、`optimizer.update()`、結果確定の 4 段すべてで epoch（`optimizerStateEpoch`）を照合している。復元が挟まれば、その世代は破棄され、復元値が残る。

- `sampling` と epoch の読み取りを同じ臨界区間に入れ、復元を丸ごと観測する（`NoiseShaperLearner.cpp` の主ループ）。
- `optimizer.update()` は epoch が一致するときだけ実行する。
- `publishGenerationResult()` は epoch 不一致なら `false` を返し、係数も `bestScore` も書かない。
- 確定後の `latestScore` の公開も、同じ臨界区間で epoch を再確認する。
- 最後のエンジン書き込み（`storeLearnedCoeffs` と `setAdaptiveNoiseShaperState`）は `callAsync` の中で `optimizerMutex` を取り、epoch を再確認してから行う。

復元が「後から来たものが勝つ」ことを、P1（後述）で固定した。ただし世代番号だけは例外で、worker のローカル変数が復元値を引き継いでいなかった。3 章で修正する。

### 1.2 保存スナップショットに別世代が混入しないか

混入していた。`saveLearnedState()` は `bestScore`、`phase`、`elapsedPlaybackSeconds`、`bestCoefficients[]` を 1 個ずつ atomic から読み、その後に `getState()` で `mean` を読んでいた。`getState()` は `optimizerMutex` を取るが、その前の読み取りは取っていない。復元が読みの間に挟まると、XML は「新しい mean + 古い係数」になる。これが Issue の言う別世代の混入である。

ミューテーション試験（修正を戻して同じ試験を実行）で、174 回中 173 回が混入した。偶然ではなく、ほぼ毎回起きる。

### 1.3 `getAdaptiveNoiseShaperState()` のロックは復元と同時でも壊れないか

壊れない。根拠は次の 3 点で、いずれもコード上で確認した。

- `bank.state` に触るのは `getAdaptiveNoiseShaperState()` と `setAdaptiveNoiseShaperState()` の 2 関数だけ（`grep` で `src/audioengine` 全体を確認）。両方が同一の `bank.stateMutex` を取る。
- `convo::NoiseShaperLearnerState` は `double` 配列と整数だけの POD で、ポインタも所有権も持たない。500 B 超のコピー中に参照先が破棄される経路が無い。
- 復元の適用は `bank.state = inState` の全体代入 1 回で、部分更新が無い。

C-03 が守るのは学習器の optimizer であり、バンク側はこの mutex が守る。守備範囲が分かれていて重複しない。既存の `NoiseShaperStateRaceTests` を再実行し、競合 0 件（読み 9,218,768 回、書き 11,394,322 回、torn 0）を確認した。

## 2. 修正内容

| ファイル | 修正 |
| --- | --- |
| `src/NoiseShaperLearner.cpp` | `saveLearnedState()` を単一スナップショットに変更。`getState()` を 1 回だけ呼び、XML の全属性をその 1 世代から書く |
| `src/NoiseShaperLearner.cpp` | `setState()` から worker 専用フィールドへの代入を削除。復元値は `progress` へ publish するだけにした |
| `src/NoiseShaperLearner.cpp` | 主ループの epoch 変化検出時に、worker が復元値を取り込む（`accumulatedPlaybackSeconds`、`currentPhase`、`generation`） |
| `src/NoiseShaperLearner.cpp` | `generation` の初期値を `progress.iteration` から取る（再開時に 0 へ戻さない） |
| `src/NoiseShaperLearner.cpp` | モード切替（worker スレッド上の `setState()`）だけは、呼び出し直後に自分で復元値を取り込む |
| `src/NoiseShaperLearner.cpp` | `progress.iteration` の公開位置に、許容する観測ゆらぎをコメントで明示 |
| `src/NoiseShaperLearner.h` | テスト用アクセサ `NoiseShaperRestoreSyncTestAccess` を friend に追加（製品コードの経路は不変） |
| `src/tests/AudioEngineHarness/NoiseShaperRestoreSyncTests.cpp` | 新規。P1〜P3 の 3 試験（opt-in、約 2 秒） |
| `CMakeLists.txt` | 上記テストを `AudioEngineHarness` に追加 |

設計の考え方は「復元要求は公開値だけを書き、worker が世代境界で自分のローカル状態へ取り込む」。B-07 の設計書が想定した大きな権限移譲ではなく、学習器の内側で閉じる小さい境界で足りた。

## 3. 検証

| 項目 | 結果 | 根拠 |
| --- | --- | --- |
| P1 復元前の epoch で確定しようとした世代 | 拒否。係数は復元値 0.750、`bestScore` は 22.0 のまま | `NS-RESTORE: P1 stalePublished=0 ... PASS` |
| P2 復元と保存の競合（修正後） | 177 回保存して混入 0 | `saves=177 mixedFiles=0 restores=24957860` |
| P2 同じ試験を旧実装で実行 | 174 回中 173 回が混入 | `saves=174 mixedFiles=173` → `FAIL (P1=1 P2=0 P3=1)` |
| P3 復元と公開スナップショットの競合 | 177 回読んで混入 0 | `snapshots=177 mixedSnapshots=0` |
| Q3 バンクの get/set 競合 | torn 0 | `NS-RACE: reads=9218768 writes=11394322 torn=0` |
| Release `ctest` | 69/69 合格（190.27 秒） | `integ_ctest.cmd` |
| 実アプリの起動と終了 | 終了コード 0、4.7 秒、`[FAULT]` と drain タイムアウトの出力なし | `ConvoPeq.exe --cli-run --cli-exit-ms 2500` |

P2 は修正を戻すと必ず落ちる。試験の検出力はこのミューテーションで確認した。P1 は epoch ゲートを外すと落ちる（`publishGenerationResult` の epoch 照合を外すと係数 0.5 と `bestScore` 33.0 が書かれる）。

## 4. 残課題

- `progress.iteration` は確定済み世代の通し番号として `optimizerMutex` の外で公開している。`getState()` は「係数は世代 N、iteration は N+1」を観測し得る。状態そのもの（mean、係数、`bestScore`）は同一臨界区間で揃うため、混在ではなく「確定済み世代が 1 つ先に見える」だけとして許容した。判断はコードのコメントに残した。
- `learned_state.xml` は iteration を保存しない。ファイルだけから復帰した場合、世代番号は 0 から再開する（係数、mean、スコアは復元される）。形式の変更（version 3）は今回の範囲外。
- 復元の適用が世代境界に揃うのは「音声が流れていて世代が回る」間だけである。`WaitingForAudio` の間は worker が世代を作らないため、復元値は公開値にだけ現れ、worker のローカル状態は次の世代まで古いまま残る。公開値は復元済みなので表示・保存は正しく、音声処理に使う係数も `progress` 経由でないため影響しない。

## 5. 再現手順

```bat
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
cmake --build build --config Release --target AudioEngineHarness
set CONVOPEQ_NS_RESTORE_SYNC_TEST=1
build\Release\AudioEngineHarness.exe
set CONVOPEQ_NS_STATE_RACE_TEST=1
build\Release\AudioEngineHarness.exe
```

`AudioEngineHarness` の通常スイートには含めない（opt-in）。どちらも約 2 秒で終わる。
