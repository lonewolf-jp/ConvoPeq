# U-3 整合整理（v2-a との依存関係）

- 作業日時: 2026-11-13 16:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **設計整理のみ**。実装・ビルド・テストは行っていない。
- 前提: Owner 判断（2026-11-13）により U-3 は未承認。v2-a の設計確定後に別途承認する。

---

## 1. U-3 の位置づけ

U-3 は `20261113-0200` §8.2 の **v2-b** である。

> W3・W4・W7 に `requestStopOnly()`（フラグ publish のみ）と `waitStopped(bound)` を追加し、
> preflight の対象に加える。`std::thread` なので「フラグ + 条件変数 + `wait_for`」で実装し、
> `join` は成功時のみ行う。

v2-a（P1〜P8）は W1・W2・W5・W6 を preflight の対象にする。U-3 はその対象を W3・W4・W7 へ広げる単位である。
したがって **v2-a の P3（preflight）と P8（保持）が無い状態で U-3 だけを実装すると、呼び出し側の無い API を作ることになる**。

## 2. 現行の停止実装（再確認）

| # | Worker | 停止 API | 要求と待機 | 停止しない場合 |
| --- | --- | --- | --- | --- |
| W3 | `m_workerThread` | `WorkerThread::stop()`（`WorkerThread.cpp:47-53`） | `running=false` → 即 `join()`（一体） | 無期限に固まる |
| W4 | `retryScheduler_` | `RetryScheduler::shutdown()`（`RetryScheduler.cpp:48-62`） | `shouldExit_=true` → `cv_.notify_all()` → `join()`（一体） | 無期限に固まる |
| W7 | `deferredFreeThread` | `shutdownAndDrain()`（`DeferredFreeThread.h:97-105`） | `stop()` → `join()` → `drainAllRetired()` | 無期限に固まる |

W4 は内部に条件変数と `shouldExit_` を既に持つため、要求部分の分離は小さい。
W3・W7 は `join` が同じ関数内にあり、分離には API 追加が要る。

## 3. 責務分離（v2-a と整合させる形）

| API | 責務 | 禁止事項 |
| --- | --- | --- |
| `requestStopOnly()` | 停止フラグの publish と条件変数の notify のみ。**待たない** | `join` / `wait` / `TerminateThread` を呼ばない（I-A3 と同じ） |
| `waitStopped(boundMs)` | 単一 deadline まで待ち、停止したかを bool で返す。**停止させる責任は持たない** | タイムアウト時に強制終了しない |

v2-a の `requestWorkerStop()` / `waitForWorkersToExit(boundMs)` と同じ形にし、
**deadline は全 Worker で共有する**（per-worker の累積待ちをしない）。
U-3 を入れる場合、preflight の対象は W1・W2・W3・W4・W5・W6・W7 の 7 つになる。

## 4. `bound` の扱い

- v2-a 改訂案（`20261113-1630_v2a改訂案` §1）と同じ規則で決める:
  `clamp(ceil(1.5 × p99), 500 ms, 5000 ms)`、p99 が取れない場合は `max`。
- W3・W4・W7 の停止時間は**未計測**である。U-3 の `bound` は、W3・W4・W7 の計測が揃うまで確定しない。
- 計測手段は v2-a の M-1（既存ログへの時刻付与）を W3・W4・W7 に広げるだけで足りる見込み。
  走行中を作る必要があるのは W7（解放待ちがある状態）だけで、そこは M-2 と同じ承認が要る。

## 5. 停止しなかった Worker の保持

- v2-a の P8（保持）にそのまま乗せる。保持対象は `MainWindow` と `AudioEngine` を含むため、
  W3・W4・W7 のオブジェクトも一緒に残る。
- 保持の成立条件 H-1〜H-4（`20261113-1630_v2a改訂案` §5.3）を U-3 でも前提にする。

## 6. `join` を成功時のみ行う場合の危険と排除

### 6.1 危険

`std::thread` は **joinable のまま破棄されると `std::terminate` を呼ぶ**。
「停止を確認できたときだけ join する」に変えると、停止しなかった Worker の `std::thread` が
joinable のまま残る。そのオブジェクトが破棄される経路が 1 つでもあれば即プロセス異常終了になる。

現行で破棄が起こり得る経路:

| # | 経路 | 現状 |
| --- | --- | --- |
| D-1 | `~AudioEngine`（W3・W4 のメンバ） | 保持経路では実行しない（H-2） |
| D-2 | `~ConvolverProcessor`（W7 のメンバ） | 保持経路では実行しない（I-F2） |
| D-3 | メンバの明示 `reset()` / 再代入 | 該当なし（要再確認） |

### 6.2 排除する不変条件（U-3 用に追加）

| # | 不変条件 |
| --- | --- |
| I-U3-1 | joinable な `std::thread` を破棄する経路を作らない。破棄は `waitStopped()` が true を返した後だけ。 |
| I-U3-2 | `requestStopOnly()` は待たない。`waitStopped()` は停止させない。 |
| I-U3-3 | `waitStopped()` が false を返した場合、当該 Worker は保持経路へ移り、デストラクタを実行しない。 |
| I-U3-4 | 保持経路では `~AudioEngine` / `~ConvolverProcessor` を実行しない（I-F2 と同じ）。 |

I-U3-1 は「実装で気をつける」ではなく、**破棄を行う箇所を列挙して 0 件であることをテストで固定する**対象にする。

## 7. 停止要求・停止完了・保持経路の関係

```
S1 IntentSet
  ├ requestStopOnly(W3,W4,W7)      … 要求のみ（待たない）
  ├ waitStopped(共有 deadline)     … 停止の確認
  │    ├ 全員 true  → S2 WorkersStopped → 通常 teardown（join する）
  │    └ 1 つでも false → S3 Retained → 何も破棄しない（join しない）
  └ S3 では W3・W4・W7 が動作を続ける（保持）
```

## 8. 人工的に「停止しない Worker」を作る試験の計画（実施前の提示）

実環境を不安定化させる試験に該当するため、**実施前に本計画の承認を得る**。

### 8.1 方式（案）

- 計測・試験専用の環境変数 `CONVOPEQ_TEST_STUCK_WORKER`（例: `W3:5000`）を追加し、
  指定された Worker が停止フラグを指定時間無視する。
- 追加するのは **テスト用の分岐 1 箇所**（各 Worker のループ条件）に限定し、既定（未設定）では従来と同一。
- 対象は Debug／Release のテストビルドのみ。製品ビルドでは環境変数を読まないよう `#if` で囲む案と、
  製品にも入れて既定無効にする案の 2 つがある。**どちらにするかは承認時に決める**。

### 8.2 手順

1. `CONVOPEQ_TEST_STUCK_WORKER=W3:5000` で起動し、`--cli-run --cli-exit-ms 2000` で終了させる。
2. 期待: `B_pre` 経過後に S3 へ落ち、`[SHUTDOWN] retained ...` が 1 行出て、プロセスが終了する。
3. 確認: `std::terminate` が起きないこと、`~MainWindow` が走らないこと（ログ）、
   保持件数が 1 であること、終了コードが 0 であること。
4. 後始末: 環境変数を外し、通常起動で `retained` が 0 のまま終了することを確認する。

### 8.3 影響と危険

- プロセスが 5 秒以上残るため、自動テストのタイムアウトに注意する。
- 保持した `MainWindow` は Debug でリーク報告される（意図的）。
- 失敗時はプロセスが終了しない可能性があるため、試験はタイムアウト付きで実行する。

## 9. 未確定事項

| # | 事項 | 状態 |
| --- | --- | --- |
| 1 | W3・W4・W7 の停止時間 | 未計測（v2-a の M-1 を拡張） |
| 2 | U-3 の `bound` の値 | §4 の規則で計測後に確定 |
| 3 | 人工 stuck 試験の実装方式（環境変数の扱い） | 承認待ち |
| 4 | v2-a の P3・P8 の完成 | v2-a 側の判断待ち |

## 10. 停止条件の確認

- ソース変更・ビルド・テスト・デバッガ実行は行っていない。
- `WorkerThread` / `RetryScheduler` / `DeferredFreeThread` の API は追加していない。
- shutdown 全体の仕様を U-3 の都合で変更していない。
- 本書は整合整理であり、U-3 の実装承認を意味しない。
