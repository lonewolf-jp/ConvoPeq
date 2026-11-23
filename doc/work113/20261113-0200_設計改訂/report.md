# shutdown 設計 改訂版（v2・READ-ONLY）

- 作業日時: 2026-11-13 02:00 JST
- 基準 commit: `173e6995`（HEAD = origin/main）
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。
  本稿の作成中に実行したのは、計測用バイナリ（既定 OFF の probe 入り）による実測と、ソース読解だけである。
- 前稿: `doc/work113/20261113-0100_停止順序設計/report.md`（2 段階プロトコル。Owner 判定「未完成」）
- 入力: `doc/work113/20261113-0120_停止計測/report.md`（実測。以下「計測報告」）

---

## 0. 判定

```
所有権の方向（MainWindow を process-lifetime で保持）  = 維持
前稿の 2 段階プロトコル                                = 骨格は維持。A〜D の閉じ方が不足
今回の提示                                             = A〜F を閉じる v2。E（同期区間）を独立項目として追加
実装 GO                                                = 未承認（本稿は設計提示のみ）
```

前稿との差分は 3 点。

1. **A**: `requestTerminalRelease()` は admission FSM を変更しない（`AudioEngine.h:1197` に明記）。
   前稿はこの時点で producer が止まる前提だったが、止まらない。**ゲート前に admission を閉じる新しい入口**が要る。
2. **B**: 非破壊分岐を `ConvolverProcessor::releaseResources()` だけに置くと足りない。
   `AudioEngine::releaseResources()` は terminal pass の途中（`:236-238`）で `stopRebuildThread()`
   （`join`・タイムアウト無し）へ進む。入口で止めなければ、stuck でも join で固まる。
3. **E**: 計測で、shutdown の起点そのものが Message Thread の同期処理に閉じ込められることが判明した
   （終了予定 150 ms に対し実終了 751 ms）。Worker 待ちの設計だけでは閉じない。

---

## 1. 計測で確定した前提（E の根拠）

計測報告 §4・§5 の再掲。数値は Release / 384 kHz / 目標 FFT 4096 / block 1024。

| 事実 | 数値 | 根拠 |
| --- | --- | --- |
| ファイル指定の IR ロードは Message Thread 同期 | 1 回 102.6〜144.0 ms | `loadIR` の `cache_direct`+`cache_low`+`convert_low`+`cache_save_low`+`apply_ir` |
| 同期リビルドは Message Thread 同期 | 334〜638 ms（192,000 samples） | `perform_load(thread==nullptr)` → `minimum_phase` + `mixed_phase` + `nuc_build` |
| 上記区間は中断できない | 全スパン `cancelled=0` | 同上 |
| shutdown の起点は同期区間の間、動かない | 終了予定 150 ms → 実終了 751 ms | シナリオ T |
| 走行中 Worker への停止要求 | 0 件（CLI 経路に走行中 Worker が無い） | `run_start`/`run_exit` 0 件 |

重要な帰結が 2 つある。

- **同期区間は「Worker を止める」問題ではない。** Message Thread が処理中の間、shutdown を開始する主体が動けない。
  したがって preflight の待機対象にできない。設計は「開始しない」ことで守る（§7）。
- **`bound` の実測根拠はまだ無い。** 5 秒は既存 `kGracefulDrainMaxMs = 5000`
  （`AudioEngine.Processing.ReleaseResources.cpp:281`）の踏襲であり、Worker 停止時間の実測値ではない。
  §11 の U-1 に「未検証の暫定値」として登録する。

---

## 2. 状態モデル

既存の 2 系統の状態（`EngineLifecycleState` と `ShutdownPhase`）に、**terminal intent** を明示的に追加する。
新しい enum は作らず、既存の `ShutdownPhase`（`AudioEngine.h:2946-2955`）と、新規 atomic 1 個で表現する。

### 2.1 追加する状態

| 名前 | 型 | 意味 | 実装 |
| --- | --- | --- | --- |
| `terminalStopRequested_` | `std::atomic<bool>`（新規） | terminal shutdown の意図。**consume しない**（`terminalReleaseRequested_` と別物） | 新規 |
| `workersStuck_` | `std::atomic<bool>`（新規） | preflight が期限内に全停止できなかった。**sticky**。実体は `AudioEngine` に 1 つだけ置き、`isWorkersStuck()` で読む | 新規 |

`terminalReleaseRequested_`（`AudioEngine.h:3030`）は既存のまま残す。あれは「releaseResources のどちらの pass を
実行するか」を選ぶ consume-once 信号であり、A の用途には使えない（`AudioEngine.h:1195-1198`）。

### 2.2 段階と許可

| 段階 | 遷移条件 | 新規 IR ロード | ISR publication | Worker 起動 | teardown |
| --- | --- | --- | --- | --- | --- |
| S0 Running | 起動直後 | 可 | 可（admission Open） | 可 | − |
| S1 IntentSet | `beginTerminalStopAcceptingWork()` | **拒否** | **拒否**（admission Closing） | **拒否** | − |
| S2 WorkersStopped | preflight 成功 | 拒否 | 拒否 | 拒否 | 実行（通常経路） |
| S3 Retained | preflight 失敗 | 拒否 | 拒否 | 拒否 | **実行しない** |

S1 は S2 と S3 の共通前置。S0 → S1 は Message Thread のみが実行する。

### 2.3 述語

```
AudioEngine::isTerminalStopRequested()  = consumeAtomic(terminalStopRequested_)        // 新規
AudioEngine::isStopAcceptingWork()      = isTerminalStopRequested()
                                        || consumeAtomic(shutdownPhase) >= StopAcceptingWork   // 新規
AudioEngine::isShutdownInProgress()     = 既存のまま（lifecycleState と ShutdownRuntime::phase_ の OR）
```

**`isShutdownInProgress()` を A に流用しない。** 実装は `ShutdownRuntime::phase_` を見る
（`ISRShutdown.cpp:154-158`）が、`closeAdmission()` は `phase_` を進めない（`ISRShutdown.cpp:445-470`）。
admission を閉じても `isShutdownInProgress()` は false のままであり、既存の判定（約 40 箇所）の時期を動かさずに済む。
代わりに、A の対象 producer だけが `isStopAcceptingWork()` を見る。

---

## 3. A — preflight 前の admission 閉鎖と producer 停止順序

### 3.1 現状（裏づけ）

- `requestTerminalRelease()` は `terminalReleaseRequested_` を立てるだけ。
  コメントに「admission FSM を変更しない」「ShutdownRuntime への reopen API は存在しない（INV-LIFE-9）」と明記
  （`AudioEngine.h:1197-1198`）。
- admission を閉じる唯一の呼出しは `shutdownRuntime_.closeAdmission()`（`ReleaseResources.cpp:109`）。
  これは `releaseResources()` の中、つまり **`mainWindow.reset()` の後**である。
- IR ロードは admission を通らない。`loadIR()` の入口にあるのはファイル存在チェックだけ
  （`LoadPipeline.cpp:237-247`）。`tryAdmit` も `isShutdownInProgress()` も見ていない。

### 3.2 設計

新設 API は 2 段に分ける。**意図の記録**と**実際の閉鎖**を混ぜないためである。

| API | 呼べるスレッド | 内容 |
| --- | --- | --- |
| `AudioEngine::requestTerminalShutdownIntent()` | どこからでも | `terminalStopRequested_` の publish のみ（atomic 1 回。teardown も close もしない） |
| `AudioEngine::beginTerminalStopAcceptingWork()` | Message Thread 専用・冪等・noexcept | 下記 4 行 |

```
if (exchangeAtomic(terminalStopRequested_, true, release)) return;   // 冪等
setShutdownPhase(ShutdownPhase::StopAcceptingWork, "beginTerminalStopAcceptingWork");
shutdownRuntime_.closeAdmission();            // 冪等（ISRShutdown.cpp:453-454）
runtimePublicationBridge_.requestShutdown();  // ReleaseResources.cpp:97 と同じ
fetchAddAtomic(shutdownEpoch_, 1, acq_rel);   // §5.2 の callAsync 抑止に使う単調カウンタ
```

`releaseResources()` 側は既存の 3 行（`:95-97` と `:109`）をそのまま残す。二重実行しても
`closeAdmission()` は `state != Open` で即 break（`ISRShutdown.cpp:453-454`）、
`setShutdownPhase` は同値なら log を出さずに戻る（`AudioEngine.h:2976-2978`）。

`lifecycleRuntime_.enterRelease()`（`:93`）は **releaseResources に残す**。lifecycle token の寿命と結びついており、
preflight 段階で取ると token の解放漏れになる。

### 3.3 producer 側の拒否点（StopAccepting）

| producer | 追加する判定 | 位置 |
| --- | --- | --- |
| `ConvolverProcessor::loadIR()` | `if (isStopAcceptingWork()) { handleLoadError("shutdown in progress"); return; }` | `LoadPipeline.cpp:239` の直後 |
| `ConvolverProcessor::loadImpulseResponse()` | 同（戻り値 false） | `LoadPipeline.cpp:24` の直後 |
| `ConvolverProcessor::startProgressiveUpgrade()` | 同（即 return） | `LoadPipeline.cpp:120` の前 |
| `ConvolverProcessor::prepareToPlay()` | stuck/StopAccepting 中は再開しない | `Lifecycle.cpp` の `isPrepared` publish 前 |
| engine `submitRebuildIntent` / Recovery / Publication | 既存 `isShutdownInProgress()` に `isStopAcceptingWork()` を OR | `RebuildDispatch.cpp:241` ほか |

`ConvolverProcessor` から engine を見る経路は既にある（`getRcuProvider()` の owner）。
判定は engine へ委譲し、`ConvolverProcessor` 側に第 2 のフラグを持たない（二重管理を避ける）。
owner を解決できない単体利用（テストハーネス等）では「Running 扱い」にフォールバックし、
その場合は従来どおりの挙動にする（既存テストの意味を変えない）。

### 3.4 順序（A の答え）

```
[MT] requestTerminalRelease()               ← releaseResources の pass 選択（既存）
[MT] beginTerminalStopAcceptingWork()      ← producer はここで止まる
[MT] engine->requestWorkerStop()             ← W1/W2/W5/W6 へ要求のみ（殺さない）
[MT] waitForWorkersToExit(bound)            ← preflight
[MT] 成功 → mainWindow.reset() / 失敗 → markStuck + audio only + 保持
```

「止めてから閉じる」ではなく「**閉じてから止める**」。閉じる前に停止要求を出すと、停止中の producer が
新しい作業を積めるため、待ちが収束しない。

---

## 4. B — `AudioEngine::releaseResources()` の非破壊分岐

### 4.1 現状（裏づけ）

`AudioEngine::releaseResources()`（`ReleaseResources.cpp:34-`）は、terminal pass で次の順に進む。

| 行 | 処理 | 破壊性 |
| --- | --- | --- |
| 50-58 | `terminalReleaseRequested_` を exchange で消費。false なら reconfigure pass へ | − |
| 61-90 | `lifecycleState` を `Releasing` へ CAS（二重解放の抑止） | 状態のみ |
| 93 | `lifecycleRuntime_.enterRelease()` | − |
| 95-109 | `StopAcceptingWork` / `AudioStopped` / `requestShutdown` / **`closeAdmission()`** | 一方向 |
| 127-153 | rebuild 理由の破棄、`cancelPendingUpdate`、crossfade reset、learner 停止、`StopAudio` | 一部破壊 |
| 169-233 | runtime slot の capture と clear、crossfade reset | **破壊** |
| 236-238 | `StopWorkers` → `shutdownCoordinatorLoop()` → **`stopRebuildThread()`（join・タイムアウト無し）** | **停止（固まり得る）** |
| 250-256 | `joinProducers()` の retry | 待つ |
| 262-274 | epoch 前進、`RetireClosed`、`closeReaderRegistration()` | 一方向 |
| 280-300 | graceful drain（最大 5000 ms ポーリング） | 待つ |
| 558 付近 | `uiConvolverProcessor.releaseResources()` | **破壊** |

`stopRebuildThread()` は `rebuildThread.join()`（`RebuildDispatch.cpp:826`）で、**タイムアウトが無い**。
`shutdownCoordinatorLoop()` は `stopThread(2000)`（`ISRCoordinatorLoop.cpp:28`）で、**タイムアウト時に強制終了する**。
stuck のまま通常の releaseResources に入ると、前者で永久に固まるか、後者でスレッドを殺すかのどちらかになる。

### 4.2 設計

terminal pass の**入口**（`ReleaseResources.cpp:59` の直後、`lifecycleState` の CAS より前）に分岐を置く。

```
if (isWorkersStuck())
{
    // 非破壊モード: 何も破棄しない。audio は呼出し側（JUCE）が既に止めている。
    // ShutdownPhase は進めない（StopAudio を飛ばして StopWorkers に飛ぶと診断が嘘になる）。
    diagLog("[DIAG] releaseResources: suppressed (workersStuck)");
    return;
}
```

- これ以降の teardown（slot clear / join / drain / IRState free / buffer free）を一切実行しない。
- `closeAdmission()` は 3.2 で既に済んでいるため、ここで飛ばしても admission は Closing のまま（望ましい）。
- `lifecycleState` を `Releasing` にしないので、`isShutdownInProgress()` は
  `shutdownRuntime_` 側（`phase_` が Running のまま）でも false になり得る。
  **したがって stuck 経路の「shutdown 中」判定は `isStopAcceptingWork()`（S1 で true）で行う。**

**reconfigure pass も同じフラグで止める。** terminal pass は consume-once でフラグを食うため、
その後にデバイス由来で `releaseResources()` が再入すると reconfigure pass に入る。
reconfigure pass 自体は learner 停止とメータ reset だけで DSP/IRState を壊さない
（`ReleaseResources.cpp:811-833`）が、`noiseShaperLearner->stopLearning()` が内部スレッドの join を含む
（`NoiseShaperLearner.cpp:210-223`）。stuck 経路ではこれも待ちになるので、同じ分岐で早期 return する。

```
if (isWorkersStuck() && !terminalRelease)   // reconfigure pass 側
{
    diagLog("[DIAG] releaseResources(reconfigure): suppressed (workersStuck)");
    return;
}
```

### 4.3 ConvolverProcessor 側

判定の実体は `AudioEngine` に 1 つだけ置く（`workersStuck_`）。`ConvolverProcessor` は owner
（`getRcuProvider()` の解決先）経由で `isWorkersStuck()` を読む。第 2 のフラグを持たない。

`ConvolverProcessor::releaseResources()`（`Lifecycle.cpp:468-520`）も入口で同じ判定を置く。
現在は `stopUpgradeThread()` → `forceCleanup()`（`stopThread(500)`）→ buffer/FFT free → `currentIRState` の即時 free
→ `deferredFreeThread->shutdownAndDrain()`（`join`）まで一気に進む。stuck 中に走らせてはいけない。

---

## 5. C — 予約済み `callAsync` callback の実行抑止

### 5.1 現状（裏づけ）

ワーカーと engine が `callAsync` で投げる callback は 35 箇所。うち shutdown 後に走ると問題になるのは次。

| 箇所 | 内容 | 危険 |
| --- | --- | --- |
| `LoaderThread.cpp:87` | 失敗時の `isLoading`/`isRebuilding` 解除 | 軽微（atomic のみ） |
| `LoaderThread.cpp:120` | `handleLoadError` | UI 状態 |
| `LoaderThread.cpp:288` | `finalizeNUCEngineOnMessageThread`（NUC 構築・IR 適用） | **重い** |
| `LoaderThread.cpp:331` | 失敗通知 | UI 状態 |
| `ProgressiveUpgradeThread.cpp:218` | `applyComputedIR` | **重い** |
| `LoadPipeline.cpp:843` / `:952` | completion / 通知 | 中 |
| `Rebuild.cpp:36` | dispatch 通知 | 中 |
| `StateAndUI.cpp:817` | completion | 中 |

いずれも `juce::WeakReference` / `weakOwner` で生存確認している。**しかし保持経路では owner は生きている。**
`weakOwner.get()` が非 nullptr を返すため、stuck 中に積まれた callback は普通に実行される。
`queueFinalizeOnMessageThread` は fire-and-forget（`LoaderThread.cpp:288-317`）でワーカーは待たないので、
shutdown が進んだ後に finalize だけが走る窓が現実に存在する。

### 5.2 設計

1. **enqueue 時の epoch を刻む。** `AudioEngine` に `std::atomic<uint64_t> shutdownEpoch_`（新規）を置き、
   `beginTerminalStopAcceptingWork()` が 1 回だけ進める（§3.2）。`ConvolverProcessor` は同じ値を
   owner 経由で読み、callback は enqueue 時の値を捕捉して、実行時に不一致なら**何もせず return**する。
   フラグを Convolver 側に複製しないので、§3.3 の方針（二重管理しない）と矛盾しない。
2. **抑止は弱参照に頼らない。** `weakOwner` は保持経路で有効なままなので、判定は epoch と
   `isStopAcceptingWork()` の AND で行う。
3. **解放漏れを残さない。** `LoaderThread.cpp:279-287` の生ポインタ（`irLRaw` 等）は、callback が早期 return しても
   確実に解放されるよう、判定は RAII ラッパー（`ScopedAlignedPtr` / `unique_ptr`）を生成した**後**に置き、
   早期 return ではラッパーに解放させる。判定をラムダの先頭に置くと生ポインタが漏れる。
4. `LoaderThread.cpp:87` の `isLoading` 解除だけは**抑止しない**。フラグを立てたままにすると、
   stuck 後の UI が永久にローディング表示になる。atomic の後始末は許容する（破壊しない）。

対象は §5.1 の表のうち「重い」「中」に限る。UI コンポーネント（`ConvolverControlPanel` / `MainWindow` /
`DeviceSettings`）の `callAsync` は `SafePointer` で既に守られており、本設計の対象外とする。

---

## 6. D — `workersStuck_` は sticky

前稿は「Worker が終了したら `workersStuck_` をクリアする」としていた。**これを撤回する。**

理由:

- 保持経路に入った時点で、`MainWindow` / `AudioEngine` / router / IRState は「破棄しない」と決めている。
  一部の Worker だけが後から終了しても、他の Worker が残っていれば同じ前提は崩れたままである。
- クリアすると、後続の `releaseResources()` が通常 teardown に入り、**残っている Worker の依存状態を壊す**。
  「全部止まった」ことを個別フラグの集合で再証明する仕組みは現状存在しない。
- プロセスは終了する。sticky にして困るのは「アプリを続行したい」場合だけであり、その要件は無い
  （保持経路 = プロセス終了待ち）。

したがって:

```
markWorkersStuck():  publishAtomic(workersStuck_, true, release);   // 以降 false にしない API は用意しない
```

解除 API を作らないことを不変条件 I-D1 とする。

---

## 7. E — Message Thread 同期区間の lifecycle 契約（新規）

### 7.1 前提の整理

- `loadIR()` は `JUCE_ASSERT_MESSAGE_THREAD`（`LoadPipeline.cpp:239`）。
- `rebuildAllIRsSynchronous` は `runSynchronously()` を呼び、これもスタック上の `LoaderThread` で
  `performLoad(nullptr)` を実行する（`Rebuild.cpp:88-95` → `LoaderThread.cpp:344-347`）。
- `MainApplication::shutdown()` も Message Thread（`MainApplication.cpp:204-229`）。

したがって**通常のメッセージループでは両者は重ならず、順序は「ロード完了 → shutdown 開始」に固定される**。
計測シナリオ T（終了予定 150 ms → 実終了 751 ms）はこの性質の実測例である。

重なりが生じ得るのは次の 2 経路だけ。

| 経路 | 内容 | 扱い |
| --- | --- | --- |
| 入れ子メッセージループ | 同期区間の内側でメッセージを回すと、その中で shutdown が走り得る | **禁止**を契約にする。同期区間の内側で `runDispatchLoop` 系を呼ばない |
| 非 MT からの shutdown 要求 | OS 信号、外部スレッド、テストハーネス等（CLI の自動終了は MT のタイマなので含まない） | `requestTerminalShutdownIntent()` で意図の記録だけ行い、teardown は MT に戻ってから |

### 7.2 契約（E-1〜E-5）

| # | 契約 |
| --- | --- |
| E-1 | 同期区間（`loadIR` / `rebuildAllIRsSynchronous` / `applyComputedIR`）の内側でメッセージループを回さない。 |
| E-2 | 非 MT の shutdown 要求は `requestTerminalShutdownIntent()`（atomic publish のみ）に限定し、`beginTerminalStopAcceptingWork()` は MT だけが呼ぶ。 |
| E-3 | 同期区間の入口で `isStopAcceptingWork()` を確認し、true なら開始しない（§3.3）。 |
| E-4 | 同期区間の実行中は、その所要時間がそのまま shutdown 遅延になる。遅延の上限は保証しない（実測 102.6〜144.0 ms / 334〜638 ms）。 |
| E-5 | 同期区間の入口に再入ガード（`std::atomic<bool> syncWorkActive_`）を置き、二重起動と dtor 進入を `jassert` で検出する。 |

E-4 を「保証しない」と書くのは、IR 長と位相モードで伸びるため（計測報告 §6.2）。
保証として書けるのは「**その区間では shutdown を開始しない**」ことまでである。

### 7.3 「止められない区間」を設計にどう織り込むか

1. shutdown 遅延の見積りは「Worker 停止時間 + 同期区間の残り時間」の和になる。両者を別々に記録する
   （stuck テレメトリに `syncWorkActive` と `workersStuck` を分けて出す）。
2. 同期区間中に届いた終了要求は、同期区間の終端で初めて評価される。これは仕様であり、
   タイムアウトで打ち切る対象ではない（打ち切る手段が無い）。
3. したがって `bound` は Worker 待ちにだけ適用する。同期区間に `bound` を適用してはならない
   （適用しても観測されないので、実装すると「守っているつもり」になる）。
   v2-a では、その Worker 待ちの対象は W1・W2・W5・W6 である（§8.2）。

### 7.4 E-1 の裏づけ（全数確認）

`src/` 配下（tests を含む）で `runDispatchLoop` / `ModalComponentManager` / `AlertWindow::show` を全走査した。

| パターン | production（`src/` の tests 以外） | tests |
| --- | ---: | ---: |
| `runDispatchLoop` | **0** | 4 ファイル（`T1`〜`T4Measurement.cpp`、`ConvolverStateRoundTripTests.cpp` はコメントのみ） |
| `ModalComponentManager` | **0** | 0 |
| `AlertWindow::show` | **0** | 0 |

同期ロード経路（`IRConverter.cpp` / `CacheManager.cpp` / `LoaderThread.cpp` / `LoadPipeline.cpp` /
MixedPhase / ResampleAndFallback / PreparedIRState / IRTrim / InputInspection）の内側にも
メッセージループは無い。待機は `waitForThreadToExit(0)`（非ブロッキング、`LoadPipeline.cpp:697, 718`）だけである。

したがって E-1 は「現状の性質」ではなく「**今後も導入しない契約**」として扱う。production に
入れ子ループを入れる変更が将来入った場合、本設計の S1 は成立しなくなる（その時点で再設計）。

---

## 8. F — Worker の停止・保持対象 inventory

`src/` 配下（tests を除く）で `startThread` / `std::thread` / `ThreadPool` / `juce::Thread` を全走査した結果。

### 8.1 engine / convolver が所有する Worker（teardown 対象）

| # | Worker | 型 | 現在の停止 API | タイムアウト時の挙動 | 要求のみ（殺さない）で止められるか |
| --- | --- | --- | --- | --- | --- |
| W1 | `rebuildThread` | `std::thread`（`Init.cpp:42` / `PrepareToPlay.cpp:109`） | `stopRebuildThread()`（`ReleaseResources.cpp:238`） | `join()` なので**固まる**（`RebuildDispatch.cpp:826`） | **可能**。`rebuildThreadShouldExit` の publish + `rebuildCV.notify_all()` + `rebuildThreadIsRunning` の期限付き polling（`RebuildDispatch.cpp:820-826` と同じ材料） |
| W2 | `coordinatorLoop_` | `juce::Thread`（`Init.cpp:133`） | `shutdownCoordinatorLoop()`（`ReleaseResources.cpp:237`） | `stopThread(2000)` で**強制終了**（`ISRCoordinatorLoop.cpp:28`） | **可能**。`signalThreadShouldExit()`（`ISRCoordinatorLoop.cpp:26`）+ `waitForThreadToExit(bound)`。`run()` は `isShutdownInProgress()` でも抜ける（`:35`） |
| W3 | `m_workerThread` | `convo::WorkerThread`（`Init.cpp:186`） | `shutdownWorkerThread()`（`CtorDtor.cpp:275`） | `join()` なので**固まる**（`WorkerThread.cpp:53`） | **不可**（要求 API が無い。`WorkerThread` に stop 要求が未実装） |
| W4 | `retryScheduler_` | `std::thread`（`RetryScheduler.cpp:7`） | `RetryScheduler::shutdown()`（`CtorDtor.cpp:173-174`） | `join()`（`RetryScheduler.cpp:60`） | **不可**（`shutdown()` は join まで行う一体 API） |
| W5 | `activeLoader` / `loaderTrashBin` | `LoaderThread`（`LoadPipeline.cpp:99`） | `forceCleanup()` の `stopThread(500)`（`StateAndUI.cpp:1054`）、`~LoaderThread`（`LoaderThread.cpp:46`） | `stopThread` が**強制終了** | **可能**。`signalThreadShouldExit()` + `waitForThreadToExit(bound)`（`signalThreadShouldExit` は既存 API） |
| W6 | `upgradeThread` | `ProgressiveUpgradeThread`（`LoadPipeline.cpp:142`） | `stopUpgradeThread()`（`LoadPipeline.cpp:103-111`） | `cancel()` → `stopThread(2000)` で**強制終了** | **可能**。`cancel()`（`ProgressiveUpgradeThread.cpp:57`）+ `waitForThreadToExit(bound)` |
| W7 | `deferredFreeThread` | `DeferredFreeThread`（`Lifecycle.cpp:440`） | `shutdownAndDrain()`（`Lifecycle.cpp:513`） | `join()`（`DeferredFreeThread.h:102`） | **不可**（`shutdownAndDrain()` は join 一体） |

**W1・W3・W4・W7 は「止まらないと join で固まる」タイプ**であり、stuck 経路では絶対に呼んではならない。
**W2・W5・W6 は「タイムアウトで殺す」タイプ**であり、stuck 経路では殺してはならない（JUCE の `stopThread` は
タイムアウトで `TerminateThread`）。

`~AudioEngine` も同じ join 群をもう一度持つ（`CtorDtor.cpp:173-177` の順で RetryScheduler → CoordinatorLoop →
RebuildThread）。`releaseResources()` を非破壊で飛ばしても、デストラクタに入れば同じ join に到達する。
**したがって保持経路では `~AudioEngine` に到達しないこと（I-F2）が、B と同じ重さで重要である。**

### 8.2 停止 API の不足（F の結論）

棚卸しの結果、**既存の API だけで「要求のみ」が可能なのは W1・W2・W5・W6** である。
W3・W4・W7 は停止 API が join 一体型で、要求と待機を分けられない。

v2 では次の 2 段階で扱う。

- **v2-a（本稿の範囲）**: preflight の対象を W1・W2・W5・W6 とする。いずれも
  「シグナル（atomic / `signalThreadShouldExit`）→ 期限付き待機」で実装できる。
  新しいのは `AudioEngine` 側の 2 メソッドだけで、**各 Worker のクラスに停止 API を足す必要はない**。
  W3・W4・W7 は preflight に含めない。teardown 自体を §4 の非破壊分岐で止めることで守る。
preflight の API は次の 2 つで足りる（新規は 2 つだけ。どちらも Message Thread・noexcept・冪等）。

```
AudioEngine::requestWorkerStop()      // W1: publish(rebuildThreadShouldExit) + rebuildCV.notify_all()
                                      // W2: coordinatorLoop_->signalThreadShouldExit()（null なら skip。未起動時）
                                      // その後 convolver->requestWorkerStop()（W5/W6）
AudioEngine::waitForWorkersToExit(boundMs)
                                      // W1: rebuildThreadIsRunning（atomic）を期限付き polling
                                      // W2: coordinatorLoop_->waitForThreadToExit(残り時間)（null なら skip）
                                      // W5/W6: 各 waitForThreadToExit(残り時間)
                                      // 単一 deadline を全員で共有する（per-worker の累積を避ける）
```

- **v2-b（要承認・別単位）**: W3・W4・W7 に `requestStopOnly()`（フラグ publish のみ）と
  `waitStopped(bound)` を追加し、preflight の対象に加える。`std::thread` なので
  「フラグ + 条件変数 + `wait_for`」で実装し、`join` は成功時のみ行う。

保持経路では W3・W4・W7（および停止要求に応じなかった W1・W2・W5・W6）が動作を続ける。
音声デバイスは閉じているので RT 経路は止まっており、プロセス終了までの継続を許容する。
これが「保持」の意味である。

### 8.3 所有していないが shutdown で止まる Worker（参考・対象外）

| Worker | 実体 | 理由 |
| --- | --- | --- |
| `g_irPreviewThreadPool` | `juce::ThreadPool(1)`（`ConvolverControlPanel.cpp:16`） | UI 所有。`SafePointer` 付き `callAsync` で完結 |
| `g_saveThreadPool` | `juce::ThreadPool(1)`（`NoiseShaperLearner.cpp:23`） | 保存専用。UI 所有 |
| `NoiseShaperLearner` の内部スレッド | `workerThread` / `slot.thread`（`NoiseShaperLearner.cpp:210-223, 530, 581`） | engine から `stopLearning()` 経由。stuck 経路では呼ばない（§4.2） |
| `IRDSP` の `std::async` | `IRDSP.cpp:52` | 一時タスク。所有なし |

これらは「保持」も「停止」もしない。プロセス終了で消える。

---

## 9. 順序図

### 9.1 通常経路

```
[MT] MainApplication::shutdown()
       engine->requestTerminalRelease()              （既存。releaseResources の pass 選択）
       engine->beginTerminalStopAcceptingWork()      S0 → S1（producer 停止・admission Closing）
       engine->requestWorkerStop() + convolver->requestWorkerStop()   W1/W2/W5/W6 へ要求のみ
       waitForWorkersToExit(bound)  ─── 全部停止 ──▶ S2
       mainWindow.reset()
         ~MainWindow: setProcessor(nullptr) → removeAudioCallback → closeAudioDevice
           AudioEngineProcessor::releaseResources → AudioEngine::releaseResources（terminal pass 全実行）
             closeAdmission は冪等 / stopRebuildThread join / drain
           ConvolverProcessor::releaseResources（stuck=false: 全破棄）
         ~AudioEngine（coordinator / rebuild / worker / retry の停止）
         メンバ破棄（coordinator → router → epochDomain → … → convolver）
```

### 9.2 タイムアウト（保持）経路

```
[MT] MainApplication::shutdown()
       engine->requestTerminalRelease()
       engine->beginTerminalStopAcceptingWork()      S1
       engine->requestWorkerStop() + convolver->requestWorkerStop()
       waitForWorkersToExit(bound)  ─── 残存 ──────▶ S3
       engine->markWorkersStuck()                    sticky
       mainWindow->shutdownAudioOnlyKeepingState()
         setProcessor(nullptr) → releaseResources（stuck: 入口で return。何も破棄しない）
         removeAudioCallback → closeAudioDevice → stopTimer
       StuckProcessHolder::keep(mainWindow.release())
       （プロセス終了まで MainWindow / AudioEngine / router / IRState / Worker は生存）
```

### 9.3 同期区間と終了要求

```
[MT] loadIR()（102.6〜144.0 ms） または rebuildAllIRsSynchronous()（334〜638 ms）
       ├─ 非 MT から terminal 要求 → terminalStopRequested_ の publish のみ（E-2）
       └─ 区間終端で MT が shutdown を評価 → 9.1 または 9.2 へ
（区間の内側ではメッセージループを回さない — E-1）
```

---

## 10. 不変条件

| # | 不変条件 |
| --- | --- |
| I-A1 | S1 以降、新規の IR ロード・upgrade・ISR publication・rebuild intent を開始しない。 |
| I-A2 | admission を閉じるのは `beginTerminalStopAcceptingWork()` と `releaseResources()` の 2 箇所だけで、どちらも冪等。 |
| I-A3 | `requestWorkerStop()`（engine / convolver）は要求のみ。`stopThread` / `join` / `TerminateThread` を呼ばない。 |
| I-B1 | `workersStuck_` が true の間、`AudioEngine::releaseResources()` / `ConvolverProcessor::releaseResources()` は何も破棄しない。 |
| I-B2 | stuck 経路では W1・W3・W4・W7 の `join` を呼ばない。W2・W5・W6 の `stopThread` を呼ばない。 |
| I-C1 | shutdown epoch を刻んだ `callAsync` callback は、epoch 不一致または StopAccepting で本体を実行しない。 |
| I-C2 | 抑止しても、生ポインタ／バッファの解放は行う（リークを作らない）。 |
| I-D1 | `workersStuck_` を false に戻す API を作らない。 |
| I-E1 | 同期区間の内側でメッセージループを回さない。 |
| I-E2 | 同期区間に `bound` を適用しない（適用しても観測されない）。 |
| I-F1 | preflight の対象は「要求のみ可能な Worker」に限る。要求 API の無い Worker は preflight に含めない。 |
| I-F2 | 保持経路では、`~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` を実行しない。 |
| I-G1 | `~ConvolverProcessor` から retire provider を呼ばない（`Lifecycle.cpp:121-124` の既存方針を維持）。 |

---

## 11. 未解決（実装前に決める必要があるもの）

| # | 項目 | 内容 | 状態 |
| --- | --- | --- | --- |
| U-1 | `bound` | 5 秒は既存 `kGracefulDrainMaxMs` の踏襲で、**実測根拠なし**。W5/W6 の走行中停止を再現して測る必要がある | 未検証の暫定値 |
| U-2 | `stopLearning()` の join | stuck 経路の reconfigure pass でも learner の join が走る可能性（`NoiseShaperLearner.cpp:210-223`）。§4.2 の早期 return で回避する設計だが、`stopLearning()` の所要は未計測 | 未計測 |
| U-3 | W3・W4・W7 の `requestStopOnly()` | 新設 API。`std::thread` 側に条件変数と `wait_for` を入れる設計変更（W1・W2・W5・W6 は既存 API で v2-a に含める） | 要承認 |
| U-4 | 入れ子メッセージループの有無 | E-1 の前提確認 | **確認済み**（§7.4） |
| U-5 | `shutdownAudioOnlyKeepingState()` の副作用 | `DeviceSettings::saveSettings` を呼ばない、ChangeListener 解除順序、`stopTimer` の位置 | 要設計 |
| U-6 | stuck テレメトリ | 保持件数、`TimedOut` 理由、生存 Worker ID、`syncWorkActive` | 要設計 |
| U-7 | IRState guard 9 箇所 / (b) パック + ペア getter / snapshot 2 経路 | 前稿からの継続項目（本稿の対象外） | 未 |

---

## 12. 計測との対応（何を測れば U-1 が閉じるか）

| 対象 | 再現方法（案） | 必要なもの |
| --- | --- | --- |
| W5 の走行中停止 | `rebuildPendingAfterLoad` を CLI から発火させる、または計測専用スイッチで `loadImpulseResponse(file)` を強制 | 計測専用 CLI スイッチ（要承認） |
| W6 の走行中停止 | `enableProgressiveUpgrade` を一時的に true にする（`AudioEngine.Parameters.cpp:822` 経路） | 設定の一時切替と復元手順 |
| W1・W2 の停止時間 | `[DIAG] releaseResources: before/after stopRebuildThread` の前後に時刻を記録して差を取る | 時刻の付与（現在のログ行に時刻は無い） |

W1・W2 は `[DIAG]` ログの前後に時刻を足せば停止時間が読める（現状のログ行に時刻は無い）。
W5・W6 は走行中そのものが未再現であり、`bound` の根拠にはまだ使えない。

---

## 13. 証跡

- 計測: `doc/work113/20261113-0120_停止計測/report.md` と同 `probe/`（17 ファイル）。
- 本稿の引用はすべて HEAD `173e6995` のソース行番号。主な参照:
  `AudioEngine.h:1199-1207, 1626-1637, 2946-2988, 3026-3030` /
  `ReleaseResources.cpp:34-109, 140-153, 236-300, 811-833` /
  `ISRShutdown.cpp:154-158, 445-470` / `ISRShutdown.h:38-54, 173-223, 304-323` /
  `LoadPipeline.cpp:18-101, 103-143, 237-383` / `LoaderThread.cpp:42-60, 271-347` /
  `ProgressiveUpgradeThread.cpp:52-67, 218` / `StateAndUI.cpp:1036-1058` /
  `Lifecycle.cpp:468-520` / `Rebuild.cpp:36, 88-95` / `RebuildDispatch.cpp:826` /
  `ISRCoordinatorLoop.cpp:24-35` / `RebuildDispatch.cpp:815-832` / `RetryScheduler.cpp:7, 48-60` /
  `WorkerThread.cpp:53` / `DeferredFreeThread.h:97-102` / `MainApplication.cpp:204-233`。

---

## 14. 停止条件の確認

- production source / test / CMake を**本稿のために**変更していない。
  （計測用 probe は前作業の計測専用変更で、既定 OFF・未コミット。本稿はそれを変更していない。）
- build / test / debugger / replay を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e6995`。
- AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9（`C3AE5099…`）も不変。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。
