# 通常終了の 2 秒 drain タイムアウト調査（read-only）

- 作業日時: 2026-11-13 19:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）
- 種別: **読み取りとログ分析のみ**。ソース変更・ビルド・修正は行っていない。
- 対象: 通常終了時に毎回発生する `waitForDrain(2000, 2)` のタイムアウト（shutdown v2-a とは独立課題）。

---

## 1. 結論

1. タイムアウトは **3 回中 3 回**で再現した。終了時間 2.4 秒のうち約 2.03 秒がこの待ちである。
2. `waitForDrain` は `isFullyDrained()` が真になるまで回り続ける。真にならない直接の候補は
   **`m_retireRouter->pendingRetireCount() == 2`** である（ドレイン窓の直後・フォールバック後の実測値）。
3. タイムアウト後もプロセスは正常終了する（exit 0・クラッシュなし）。ただし
   **公開済み World（worldId 1）が retire されないまま**終わっており、`retiredCount = 0` である。
4. 2 秒待ちは「bounded graceful drain」として意図された設計だが、**毎回フォールバック側に落ちている**。
   設計意図と実挙動が一致していない。
5. タイムアウト値を短くする判断はしない。未完了の drain 対象を残したまま終了する危険を評価できていない。

## 2. 再現と証跡

`--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` を、作業ディレクトリを一時フォルダにして 3 回実行した
（リポジトリの `evidence/` を汚さないため）。出力は一時フォルダの `evidence/` に生成される。

| 実行 | exit | phase | `sh3_pendingRetire` | activeWorld | published | retired | reclaimAttempts | reclaimSuccess |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | ShutdownComplete | **2** | 1 | 1 | 0 | 123,455 | 6 |
| 2 | 0 | ShutdownComplete | **2** | 1 | 1 | 0 | 100,297 | 6 |
| 3 | 0 | ShutdownComplete | **2** | 1 | 1 | 0 | 105,423 | 6 |

- `sh3_pendingRetire` は `ReleaseResources.cpp:765-775` で
  `m_retireRouter->pendingRetireCount()` をそのまま `shutdownRuntime_` へ渡した値である
  （ドレイン待ちとフォールバックの**後**に取得される）。
- `reclaimAttemptCount` が 10 万件を超えるのは、2 ms 間隔のポーリング（2 秒で約 1,000 回）で
  `drainDeferredRetireQueues(true)` を呼び続けた結果である。成功は 6 件のみ。
- ログには毎回 `[DIAG] releaseResources: drain timeout reached, performing safe tryReclaim (drainAll skipped)`
  が出る。

## 3. `isFullyDrained()` の条件と、除外できたもの

`AudioEngine::isFullyDrained()`（`AudioEngine.Threading.cpp:243-303`）は 9 個の積を取る。

| # | 条件 | 判定材料 | 評価 |
| --- | --- | --- | --- |
| 1 | `!hasDeferredCommit` | 内部フラグ | 未確認 |
| 2 | `pendingReclaimHandles_` が空 | mutex 保護の集合 | 未確認 |
| 3 | **`pendingRetireCount() == 0`** | ルータの実測 | **偽（= 2）** |
| 4 | `lifetime().pendingIntentCount() == 0` | 実測 | 未確認 |
| 5 | overflow ring の resident == 0 | 実測 | 未確認（`overflowCount = 0` の証跡あり） |
| 6 | DSP quarantine resident == 0 | 実測 | 未確認 |
| 7 | retire quarantine resident == 0 | 実測 | 未確認 |
| 8 | terminal reclaim resident == 0 | 実測 | 未確認 |
| 9 | `runtimePublicationBridge_.isFullyDrained()` | キュー・各種 residency・recovery 義務 | 未確認 |

除外できたもの:

- **Reader は停止している。** `reader_registry_final.json` は `activeAtStage: []`、`enterFailures: 0`。
- **World の残存は #9 の直接原因ではない。** `ShutdownScheduler::isFullyDrained()`
  （`ISRRuntimePublicationCoordinator.cpp:521-572`）はキューとカウンタだけを見ており、公開中 World の有無を含まない。
- **プロセスは正常終了している。** 3 回とも exit 0。

## 4. なぜ 2 件が残るのか（推定・未確定）

2 件が残る理由は、証跡だけでは 1 つに絞れない。候補は次の 2 つである。

1. **EngineISR ドメインの epoch が進まない。** `m_retireRouter` の provider は `m_epochDomain`
   （`RetireDomainId::EngineISR`、`AudioEngine.h:5456`、`AudioEngine.CtorDtor.cpp:62`）であり、
   残存 2 件はこのドメインのキューにある。`drainDeferredRetireQueues(true)` が epoch を前進させても
   `isOlder(entry.epoch, minReaderEpoch)` が成立しない（＝より新しい epoch で積まれた）なら、
   2 件は 2 秒間ずっと残る。
2. **公開中 World を retire する経路が通っていない。** `world_lifecycle_audit.json` は
   `publishedCount 1 / retiredCount 0 / retireEpoch 0 / retireTimestampUs 0` であり、
   `releaseResources.cpp:579-584` の clear→retire が効いていない可能性がある。
   ただし #9 は World の有無を見ないため、これだけでは `isFullyDrained()` は偽にならない。

どちらかを確定するには、**タイムアウト時に 9 条件の値を 1 行だけログへ出す**計測が必要である。
これはソース変更なので、実施には承認が要る（本書では行っていない）。

## 5. タイムアウト後の終了は安全か

証跡から言えること:

| 観点 | 事実 |
| --- | --- |
| プロセス終了 | 3 回とも exit 0、クラッシュ・ハングなし |
| 2 段目の drain | タイムアウト後は `tryReclaim` → `drainDeferredRetireQueues(true)`、その後 `~AudioEngine` が最大 5 秒の drain をもう一度実行する。`step 10 UI reset` から `mainWindow.reset() completed` までが 170 ms なので、2 段目は短時間で終わっている |
| Reader | 終了時点で active 0 |
| 退役記録 | `retire_latency_post_teardown.json` の postTeardown 段で `postShutdownRequests 42` / `postShutdownCompletions 21` / `recordMisses 42`。**shutdown 開始後にも退役要求が 42 件発生**し、記録漏れが 42 件ある |
| World | 公開済み World 1 件が retire されないまま（プロセス終了で解放される） |

判断: **機能的には安全に終了しているが、きれいではない。** 退役要求が shutdown 後にも発生しており、
その一部は記録されていない。World 1 件はプロセス終了まで保持される。

## 6. 仕様か、待ち合わせの不成立か

- 設計上は「bounded graceful drain（上限 2 秒、超過時は安全側のフォールバック）」であり、
  タイムアウト自体は異常ではない。
- しかし **毎回タイムアウトしている**ということは、graceful 条件がこの構成では成立していない。
  すなわち「通常は成功し、稀に超過する」という想定ではなく、**フォールバックが常態**になっている。
- したがって、これは「意図された待ち」ではなく **待ち合わせの不成立**として扱うのが妥当である。
- ただし、タイムアウトを短くすると未完了の退役対象を残したまま次段へ送ることになる。
  短縮は解決ではなく、危険の先送りになり得る。

## 7. 追加で必要な計測（提案）

| # | 内容 | 必要なもの |
| --- | --- | --- |
| 1 | タイムアウト時に 9 条件の値を 1 行ログへ出す（`pendingRetireCount` / `pendingReclaimHandles_.size()` / `pendingIntentCount` / ring / DSP quarantine / retire quarantine / terminal reclaim / bridge 判定 / `hasDeferredCommit`） | ソース変更（ログ 1 行）の承認 |
| 2 | 残った 2 件のドメイン（EngineISR / EQ）を出す | 上と同じログに `retireLatencyClassified.domains[]` 相当を含める |
| 3 | 公開 World が retire されない理由（clear→retire の戻り値） | 上と同じログに clear/retire の成否を含める |

## 8. 推奨（今回は実施しない）

1. まず計測（§7 の 1〜3）で 9 条件のどれが偽かを確定する。
2. 確定後、原因に応じて次のどちらかを選ぶ。
   - EQ ドメインの epoch 前進漏れなら、shutdown 経路で明示的に前進させる（v2-a の P5 とも関係する）。
   - World retire の経路漏れなら、clear→retire の成否を確認して経路を直す。
3. タイムアウト値の変更は、上記の確定後に行う。短縮を先行させない。

## 9. 未確認事項と残留リスク

| # | 内容 | 状態 |
| --- | --- | --- |
| 1 | 9 条件のうち偽になっているものの特定 | 未確認（`pendingRetireCount = 2` のみ実測） |
| 2 | 残存 2 件のドメイン | **EngineISR と確定**（`m_epochDomain` は `RetireDomainId::EngineISR`、`AudioEngine.h:5456`。`pendingRetireCount()` はこのドメインへ委譲される）。中身（型・サイズ）は未確認 |
| 3 | World が retire されない理由 | 未確認（`retiredCount 0` の事実のみ） |
| 4 | shutdown 後の退役要求 42 件の発生源 | 未確認 |
| 5 | 2 秒を短縮した場合の安全性 | 評価していない（推奨しない） |

## 10. 判定

**調査完了（原因は 1 点に絞れず）。** 再現性・影響範囲・タイムアウト後の安全性は確認できたが、
9 条件のどれが偽かは証跡だけでは確定できない。確定には 1 行のログ追加（要承認）が必要である。
修正は行っていない。
