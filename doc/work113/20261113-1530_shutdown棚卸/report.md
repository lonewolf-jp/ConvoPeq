# shutdown 関連の未着手事項の棚卸し

- 作業日時: 2026-11-13 15:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 判定: **未着手（実装承認が無いため、今回は実装しない）**。判断材料を §6 に示す。

---

## 1. 参照した資料

| 資料 | 内容 |
| --- | --- |
| `doc/work113/20261113-0200_設計改訂/report.md` | shutdown 設計 v2 の本体（W1–W7 棚卸し、v2-a / v2-b の分割、I-B/I-F 不変条件） |
| `doc/work113/20261113-0300_設計v2修正/report.md` | v2-a の実装単位 P1–P8、未解決 U-1〜U-8、**「実装 GO・Gate 1 承認は付与しない」** |
| `doc/work113/20261113-0400_設計v2a再修正/report.md` | v2-a の再修正 |
| `doc/work113/20261113-0500_保持経路と予算/report.md` | 保持経路と待ち時間の予算 |
| `doc/work113/20261110-2045_shutdown追跡/report.md` | 現行 teardown の追跡 |

## 2. 現行の停止順序（ソースで確認）

```
MainApplication::shutdown()                         MainApplication.cpp:204
  └ engine->requestTerminalRelease()                :217-222   （terminal 意図の consume-once 設定）
  └ mainWindow.reset()                              :229       （~MainWindow 経由で teardown 全体）
       └ オーディオコールバック停止 → UI 破棄 → ~AudioEngine
  └ evidence export → LOGGER_DETACH                 :237-247

AudioEngine::releaseResources()                     ReleaseResources.cpp:90-
  └ enterRelease / StopAcceptingWork / AudioStopped :94-97
  └ runtimePublicationBridge_.requestShutdown()     :98
  └ closeAdmission()                                :110       （admission を閉じる・冪等）
  └ MMCSS 委譲 → finalizeMmcssShutdown()            :115-124
  └ rebuild 理由の破棄・cancelPendingUpdate         :128-132
  └ （以降で W2 stopThread / W1 join を実施）        :198 付近・:238-239

~AudioEngine()                                      CtorDtor.cpp:161-
  └ cancelPendingUpdate / requestShutdown           :162-163
  └ stopTimer()                                     :167
  └ retryScheduler_->shutdown()          （W4）     :173-174   join
  └ shutdownCoordinatorLoop()            （W2）     :176       stopThread(2000)
  └ stopRebuildThread()                  （W1）     :177       join
  └ shutdownWorkerThread()               （W3）     :275       join
  └ ForceEpochAdvance → DrainRetire                 :277-309   最大 5 秒ポーリング
  └ published runtime の clear / drain              :311-320
```

`~ConvolverProcessor` は `shutdownAndDrain()`（W7、`ConvolverProcessor.Lifecycle.cpp:514`）で join する。

実測（今回、`--cli-run --cli-exit-ms 2000`）では終了区間が約 4.2〜4.5 秒、
そのうち `--cli-exit-ms` の 2 秒を除いた teardown は約 2.2〜2.5 秒である。
`~AudioEngine` の DrainRetire は最大 5 秒待つ設計だが、通常は数十 ms で 0 に到達する。

## 3. Worker の棚卸し（現行ソースで再確認）

| # | Worker | 停止 API | タイムアウト時 |
| --- | --- | --- | --- |
| W1 | `rebuildThread` | `stopRebuildThread()`（`ReleaseResources.cpp:238`） | `join()`（無期限） |
| W2 | `coordinatorLoop_` | `shutdownCoordinatorLoop()`（`:237`） | `stopThread(2000)` で強制終了 |
| W3 | `m_workerThread` | `WorkerThread::stop()`（`WorkerThread.cpp:47-53`） | `join()` |
| W4 | `retryScheduler_` | `RetryScheduler::shutdown()`（`RetryScheduler.cpp:48-62`） | `join()` |
| W5 | `activeLoader` / `loaderTrashBin` | `forceCleanup()` の `stopThread(500)` | 強制終了 |
| W6 | `upgradeThread` | `stopUpgradeThread()`（`LoadPipeline.cpp:103-111`） | `cancel()` → `stopThread` |
| W7 | `deferredFreeThread` | `shutdownAndDrain()`（`DeferredFreeThread.h:97-105`） | `join()` |

W3・W4・W7 は「要求と待機が分離できない（join 一体型）」ままである。
W4 は内部に `shouldExit_` と条件変数を持つが、公開 API は `shutdown()` のみで join まで行う
（`RetryScheduler.cpp:48-62`）。

## 4. 設計 v2-a の実装状況

設計 v2-a は P1〜P8 の 8 単位（`20261113-0300` §7）。現行ソースを全走査した結果、
**いずれも実装されていない**。

| 単位 | 識別子 | 出現数 |
| --- | --- | --- |
| P1 | `terminalStopState_` / `shutdownEpoch_` | 0 / 0 |
| P2 | `rebuildThreadLoopExited` | 0 |
| P3 | `requestWorkerStop` / `waitForWorkersToExit` | 0 / 0 |
| P7 | `SyncWorkToken` | 0 |
| P8 | `StuckProcessHolder` / `shutdownAudioOnlyKeepingState` | 0 / 0 |

既存で存在するのは `isShutdownInProgress()`（`AudioEngine.h:1626`、`shutdownRuntime_.phase_` 依存）、
`closeAdmission()`（`ISRShutdown.cpp:445-470`）、`kGracefulDrainMaxMs`（`CtorDtor.cpp:288`）である。

## 5. 未着手事項の一覧

| # | 項目 | 状態 | 根拠 |
| --- | --- | --- | --- |
| S-1 | v2-a（preflight を W1/W2/W5/W6 に限定した 2 段停止） | 未着手 | 設計はあるが `0300` §10 が「実装 GO を付与しない」 |
| S-2 | v2-a の P8（保持経路 `StuckProcessHolder` / audio-only teardown） | 未着手 | 同上。`shutdownAudioOnlyKeepingState()` は存在しない |
| S-3 | U-5（audio-only teardown の副作用設計） | 要設計 | `0300` §8 U-5 |
| S-4 | U-6（stuck テレメトリの項目） | 要設計 | `0300` §8 U-6 |
| S-5 | U-1（`bound` 5 秒の根拠） | 未検証の暫定値 | `0300` §8 U-1 |
| S-6 | U-2（`stopLearning()` の所要） | 未計測 | `0300` §8 U-2 |
| S-7 | U-3（W3/W4/W7 の停止要求 API） | 未承認 | `0300` §8 U-3。別報告 `20261113-1530_U-3棚卸` |

## 6. 実装前の判断材料（Owner への確認事項）

1. **v2-a を今実装するか。** 設計は 8 単位・複数ファイル（`AudioEngine.h`、`Init.cpp`、
   `PrepareToPlay.cpp`、`RebuildDispatch.cpp`、`LoadPipeline.cpp`、`ReleaseResources.cpp`、
   `ConvolverProcessor.*`、`MainApplication.cpp`、`MainWindow.*`）に及び、停止経路そのものを変える。
   `0300` は「実装 GO を付与しない」で終わっており、本指示の「既存の安全性を損なう可能性がある事項は
   独断で補完せず実装前に報告」に該当するため、今回は実装していない。
2. **検証体制。** v2-a の受入れは「`retireLatencyD1` の `max(L_i) <= 50 ms`」等の統計判定を含み、
   `20261110-1418` が指摘した環境速度変動（同一関数で 9.9〜13.0 秒）の中で判定する必要がある。
   判定基準の確定が先である。
3. **順序。** 設計は P1 → P2 → (P3, P4, P5, P6, P7) → P8。P3 と P5 は独立に検証できる。
   どの単位までを 1 回の作業とするか。

## 7. 今回行わなかったこと

- ソース変更、ビルド、テスト、デバッガ実行は行っていない（この課題のため）。
- 停止順序・所有権・ライフタイムに触れる変更は一切していない。
- `shutdown` の仕様を独自に補完していない。

## 8. 残留リスク

- 現行の停止経路は W1/W3/W4/W7 が join 一体型のため、Worker が停止しない場合に
  プロセスが固まる構造はそのまま残っている（設計 v2-a が対処する範囲）。
- 本棚卸しは識別子の有無と設計書の記述に基づく。実装時に、設計書の行番号が現在のソースと
  ずれている可能性がある（設計書は 2026-11-13 時点のもの）。

## 9. 判定

**未着手。** 仕様（設計 v2-a）は存在するが実装承認が無く、停止経路の安全性に直結するため、
判断材料の提示にとどめた。実装する場合は §6 の 1〜3 への回答が必要である。
