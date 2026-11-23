# U-3: 停止要求 API（W3・W4・W7）の棚卸し

- 作業日時: 2026-11-13 15:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 判定: **未着手（未承認のため実装しない）**。判断材料を §5 に示す。

---

## 1. U-3 とは何か

`doc/work113/20261113-0200_設計改訂/report.md` §8.2 で定義された v2-b の単位である。

> W3・W4・W7 に `requestStopOnly()`（フラグ publish のみ）と `waitStopped(bound)` を追加し、
> preflight の対象に加える。`std::thread` なので「フラグ + 条件変数 + `wait_for`」で実装し、
> `join` は成功時のみ行う。

`doc/work113/20261113-0300_設計v2修正/report.md` §8 では
「U-3 | W3・W4・W7 の停止要求 API | 本稿の対象外（Owner 未承認） | 未承認」と記録され、
§10 で「U-3 も未承認のままとする」と明示されている。

## 2. 現行の実装（ソースで確認）

| # | Worker | 停止 API | 実装 | join の位置 |
| --- | --- | --- | --- | --- |
| W3 | `m_workerThread` | `WorkerThread::stop()` | `WorkerThread.cpp:47-53` | `running=false` の直後に `thread.join()` |
| W4 | `retryScheduler_` | `RetryScheduler::shutdown()` | `RetryScheduler.cpp:48-62` | `shouldExit_=true` → `cv_.notify_all()` → `join()` |
| W7 | `deferredFreeThread` | `DeferredFreeThread::shutdownAndDrain()` | `DeferredFreeThread.h:97-105` | `stop()` → `join()` → `drainAllRetired()` |

いずれも「要求（フラグ/条件変数）」と「待機（join）」が 1 つの関数に束ねられている。
W4 は内部に `shouldExit_` と条件変数を持ち、要求部分は既に分離可能な形になっているが、
公開 API は join まで行う。

呼出元は `~AudioEngine`（W3: `CtorDtor.cpp:275`、W4: `:173-174`）と
`~ConvolverProcessor`（W7: `ConvolverProcessor.Lifecycle.cpp:514`）である。

## 3. 設計が求めている形

| 項目 | 現行 | v2-b（U-3） |
| --- | --- | --- |
| 停止要求 | `stop()` / `shutdown()` / `shutdownAndDrain()` のみ | `requestStopOnly()` を追加（フラグ publish と notify のみ） |
| 停止待機 | 要求と同時に無期限 join | `waitStopped(bound)` を追加（条件変数 + `wait_for`） |
| join | 常に実行 | **停止を確認できたときのみ**実行 |
| 停止しない場合 | プロセスが固まる | 呼び出し側へ戻り、保持経路へ委ねる |

## 4. 未確定事項（実装前に決める必要がある）

| # | 事項 | 内容 |
| --- | --- | --- |
| 1 | `bound` の値 | 設計は 5 秒（`kGracefulDrainMaxMs` 踏襲）を暫定としているが、W3/W4/W7 の実測が無い（`0300` §8 U-1） |
| 2 | 停止しなかった Worker の扱い | 保持経路（P8）で生かしたまま終了する前提。`StuckProcessHolder` が未実装のため、現状は受け皿が無い |
| 3 | `join` を成功時のみにする影響 | 現行は必ず join するため、`~WorkerThread` / `~RetryScheduler` / `~DeferredFreeThread` のデストラクタで join する経路が残る。`joinable()` のまま破棄すると `std::terminate` になるため、破棄順序の再設計が要る |
| 4 | 呼出順序 | U-3 は v2-a（preflight）に W3/W4/W7 を加える設計であり、v2-a 自体が未実装（別報告 `20261113-1530_shutdown棚卸` §4） |

## 5. 実装前の判断材料

1. **前提が未実装である。** U-3 は v2-a の preflight（`requestWorkerStop` / `waitForWorkersToExit`）へ
   W3/W4/W7 を追加する設計であり、v2-a 自体が存在しない。U-3 だけを先に実装すると、
   呼び出し側（preflight）が無い API を作ることになる。
2. **安全性への影響が大きい。** `join` を「成功時のみ」に変えると、停止しなかった Worker の
   オブジェクトが `joinable()` のまま破棄される経路が生まれ得る。`std::thread` の破棄は
   `std::terminate` を呼ぶため、保持経路（P8）とデストラクタの扱いを同時に設計する必要がある。
3. **検証手段。** 停止しない Worker を人工的に作る試験（意図的なハング）は、実環境を不安定化させる
   試験に該当するため、実施前に承認が要る。

## 6. 今回行わなかったこと

- ソース変更、ビルド、テストは行っていない。
- `WorkerThread` / `RetryScheduler` / `DeferredFreeThread` の API 追加はしていない。
- shutdown 全体の仕様変更はしていない（U-3 を理由に他へ波及させていない）。

## 7. 残留リスク

- W3/W4/W7 は join 一体型のままである。停止しなかった場合にプロセスが固まる構造は変わらない。
- U-3 を実装する際は、§4 の 1〜4 を確定させる必要がある。

## 8. 判定

**未着手（未承認）。** 設計は存在するが、前提（v2-a）が未実装で、`join` の扱いを変えると
ライフタイムと破棄順序に影響する。独断で補完せず、判断材料の提示にとどめた。
