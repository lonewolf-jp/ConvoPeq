# A/C 計測の結果 — c9 の内訳と epoch 安全条件

- 作業日時: 2026-11-13 20:21 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋未コミットの診断ログ
- 種別: Owner 承認済み A2（c9 の 14 条件）と C（epoch 比較）の実装・計測。**原因修正は行っていない。**
- 計測: `--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` を 3 回（作業ディレクトリは一時フォルダ）

---

## 1. 結論

3 回とも同じ結果だった。**c3 と c9 は別々の原因**で、どちらも単独で `isFullyDrained()` を偽にする。

| 条件 | 実測 | 原因 |
| --- | --- | --- |
| c3（`pendingRetireCount()` = 1） | 先頭 entry の `retireEpoch` = 12、`minReaderEpoch` = 12、`currentEpoch` = 12 | `isOlder(12, 12)` が偽。**2 秒間 epoch が 1 度も前進していない** |
| c9（bridge 未 drain） | 14 条件のうち **`retireBacklogCount_` = 143 だけが非 0**、残り 13 は 0 | `retireBacklogCount_` に**本番の減算経路が無い**（`onRetireConsumed()` の呼出し元が 0） |

つまり、c3 を解消しても c9 が残るため、2 秒タイムアウトは消えない。

## 2. 実装（A2 + C）

### 2.1 変更一覧

| # | ファイル | 箇所 | 内容 | 行数 |
| --- | --- | --- | --- | ---: |
| 1 | `src/audioengine/ISRRuntimePublicationCoordinator.h` | `:222-243` | `DrainBreakdown`（14 フィールド）と `collectDrainBreakdown()` 宣言 | 22 |
| 2 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `:479-501` | `collectDrainBreakdown()` の実装 | 23 |
| 3 | `src/DeferredDeletionQueue.h` | `:277-303` | `peekHead()`（読み取り専用・POD コピー） | 27 |
| 4 | `src/core/EpochDomain.h` | `:558-601` | `RetireHeadSnapshot` と `inspectRetireHead()` | 44 |
| 5 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `:711-719`, `:756-780` | 診断の取得とログ項目の追加 | 33 |
| 6 | `src/tests/ISRSemanticValidationTests.cpp` | `:2597-2653`, `:2913-2915` | 対応を固定するテスト AC1 | 59 |

判定関数 `isFullyDrained()` は 1 行も変えていない（A2）。`git diff` で
`ShutdownScheduler::isFullyDrained()`（`ISRRuntimePublicationCoordinator.cpp:521-572`）に差分が無いことを確認済み。

### 2.2 A2: 診断側の 14 条件

診断側の `collectDrainBreakdown()` は `isFullyDrained()` と同じ式を短絡なしで読む。

| # | フィールド | 式 | 元の条件 |
| --- | --- | --- | --- |
| 1 | `swapPending` | `consumeAtomic(swapPending_, acquire)` | 早期 return |
| 2 | `intentQueue` | `intentQueue_.sizeApprox()` | `== 0` |
| 3 | `observeDeferredRing` | `observeDeferredRing_.size()` | `== 0` |
| 4 | `quarantineFallback` | `quarantineFallbackQueue_.sizeApprox()` | `== 0` |
| 5 | `recoveryIntentQueue` | `recoveryIntentQueue_.size()` | `== 0` |
| 6 | `retireBacklog` | `consumeAtomic(retireBacklogCount_, acquire)` | `== 0` |
| 7 | `publicationBacklog` | `consumeAtomic(publicationBacklogCount_, acquire)` | `== 0` |
| 8 | `publicationIntentResidency` | `consumeAtomic(publicationIntentResidencyCount_, acquire)` | `== 0` |
| 9 | `pendingIntent` | `consumeAtomic(pendingIntentCount_, acquire)` | `== 0` |
| 10 | `reclaimInFlight` | `consumeAtomic(reclaimInFlightCount_, acquire)` | `== 0` |
| 11 | `quarantineIntentResidency` | `consumeAtomic(quarantineIntentResidencyCount_, acquire)` | `== 0` |
| 12 | `quarantineRingResidency` | `consumeAtomic(quarantineRingResidencyCount_, acquire)` | `== 0` |
| 13 | `recoveryAdmissionPending` | `consumeAtomic(recoveryAdmissionPending_, acquire)` | `false` |
| 14 | `liveRecoveryObligation` | `liveLogicalRecoveryObligationCount()` | `== 0` |

複写の乖離は **テスト `testAC1_drainBreakdownMatchesPredicate`** で固定した。

- 初期状態で `isFullyDrained() == true` かつ内訳が全 0。
- 既存 getter を持つ 9 項目（1, 6〜12, 14）は値が一致すること。
- Publish Intent を 1 件積むと両者とも「未 drain」になり、滞留が内訳に現れること。

`ctest` の `ISRSemanticValidationRejects` 内で実行され、PASS した。

### 2.3 C: 先頭 entry の epoch

先頭参照の `peekHead()` は Vyukov キューの公開規則に合わせ、**コピー前後で sequence と dequeuePos を確認**する。

1. `pos = dequeuePos`（acquire）
2. `seqBefore = sequences[pos & mask]`（acquire）。`seqBefore != pos + 1` なら「空／未 publish」で false
3. `snapshot = ringBuffer[pos & mask]`（POD コピー。ポインタは辿らない）
4. `seqAfter` と `posAfter` を読み直し、**どちらかが変わっていれば false**（コピー中に上書き・破棄された）

これにより「payload がコピー中に上書き・破棄されないこと」を、producer join 済みという前提だけに頼らず
実装上も確認している。ABA は 4096 回の周回が必要で、shutdown 時（producer join 済み・consumer 1 本）には
成立しない。

スナップショットを返す `inspectRetireHead()` は `minReaderEpoch`（`getMinReaderEpoch()`）・
`currentEpoch`・先頭 entry の `epoch`・`type`・`domainId`・初回走査からの経過を 1 回でまとめ、`isOlder(entry.epoch, minReaderEpoch)`
（`reclaim()` と同じ式）を `epochOlderThanMinReader` として返す。

### 2.4 ログ（既存行に追加・行数は増えない）

```
[DRAIN_DIAG] timedOut=1 domain=EngineISR failed=c3,c9 ... c9_swap=0 c9_iq=0 c9_oring=0 c9_qfb=0
  c9_rq=0 c9_rbl=143 c9_pbl=0 c9_pir=0 c9_pi=0 c9_rif=0 c9_qir=0 c9_qrr=0 c9_rap=0 c9_lo=0
  head=present headRetireEpoch=12 headType=0 headDomain=2 headScanned=1 headScanAgeMs=2001
  minReaderEpoch=12 currentEpoch=12 retireEpochLtMinReader=0
```

- `headType=0` は Generic、`headDomain=2` は EngineISR（`RetireDomainId` の値）。
- 先頭が無い／sequence 確認に失敗した場合は `head=unavailable` とし、`headRetireEpoch`・`headType`・
  `headDomain`・`headScanned`・`headScanAgeMs`・`retireEpochLtMinReader` の 6 項目は `na` にする
  （推定値を出さない）。
- `minReaderEpoch` と `currentEpoch` は常に実値。
- **1 shutdown あたり 2 行**（`worldClear` 1 行 + `timedOut` 1 行）。3 回とも 2 行だった。

## 3. 計測結果（3 回）

| 実行 | exit | 総経過 | 行数 | c3 | c9 | `c9_rbl` | 他の c9 | `headRetireEpoch` | `minReaderEpoch` | `currentEpoch` | `retireEpochLtMinReader` | `headScanAgeMs` | reclaim 試行/成功 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 6.57 s | 2 | 1 | 0 | **143** | 全 0 | 12 | 12 | 12 | **0** | 2001 | 76,964 / 6 |
| 2 | 0 | 6.29 s | 2 | 1 | 0 | **143** | 全 0 | 12 | 12 | 12 | **0** | 2000 | 89,158 / 6 |
| 3 | 0 | 6.31 s | 2 | 1 | 0 | **143** | 全 0 | 12 | 12 | 12 | **0** | 2002 | 112,588 / 6 |

なお `c9_rbl=143` は 3 回とも同じ値だった（EQ の退役回数が決定的であることを示す）。

## 4. c3 の原因（epoch が前進しない）

- 先頭 entry は `epoch=12`、`minReaderEpoch=12`、`currentEpoch=12`。`isOlder(12, 12)` は偽なので
  `reclaim()` は先頭で `break` する（`DeferredDeletionQueue.h:174`）。
- `currentEpoch` は 2 秒間 12 のままだった。`headScanAgeMs=2001` は「最初の走査から 2,001 ms 経過」で、
  先頭 entry が一度も進まなかったことを示す。
- コード上の根拠: `waitForDrain` が 2 ms ごとに呼ぶ `drainDeferredRetireQueues(true)` は
  `tryReclaim()` と `reclaim()` を呼ぶだけで、**epoch を前進させない**
  （`AudioEngine.Retire.cpp:58-83`）。
- epoch を進めるのは `~AudioEngine` の `publishEpoch()`（`AudioEngine.CtorDtor.cpp:278`）と、
  その後の 5 秒 drain ループ（同 `:299`）。いずれも 2 秒の `waitForDrain` の**後**に走る。
- したがって「shutdown 中に積まれた entry は、その epoch のままでは 2 秒窓で回収できない」。
  Reader は 0 なので grace 待ちではない。

## 5. c9 の原因（`retireBacklogCount_` に減算経路が無い）

14 条件のうち非 0 は `retireBacklogCount_` = 143 だけだった。増減の経路を追った。

| 方向 | 実装 | 呼出し元 |
| --- | --- | --- |
| +1 | `onRetireAccepted()`（`ISRRuntimePublicationCoordinator.cpp:198`） | `RuntimeIntentCoordinator::enqueueRetire()` の成功時（同 `:179`） |
| -1 | `onRetireConsumed()`（同 `:206`） | **本番の呼出し元が 0**（`src` 全体を検索して定義と宣言のみ） |

増加側の `enqueueRetire()` は EQ から呼ばれている。`EQProcessor` の `m_retireCoordinator` は
`AudioEngine` の `runtimePublicationBridge_` そのもの（`AudioEngine.CtorDtor.cpp:73`、
`DSPCoreLifecycle.cpp:113`）で、`EQProcessor.Core.cpp:60` から `enqueueRetire` を呼ぶ。
つまり **EQ の退役 1 件ごとに `retireBacklogCount_` が +1 され、戻らない**。

- 143 は 2 秒の実行で EQ が退役した回数と一致する（3 回とも同値）。
- したがって `ShutdownScheduler::isFullyDrained()` の `retireBacklogCount_ == 0` は、
  EQ 退役が 1 件でも起きた時点で**恒久的に偽**になる。
- コード内のコメント `ISRRuntimePublicationCoordinator.cpp:177`「本メソッドは production で未使用」は
  **現状と一致しない**（`EQProcessor.Core.cpp:60` が本番で呼ぶ）。コメントの更新が必要。

## 6. 証跡との整合

| 値 | ログ | 証跡 |
| --- | --- | --- |
| 残存 retire | `c3_retireDepth=1`（タイムアウト時点） | `shutdown_trace.json` の `sh3_pendingRetire=2`（フォールバック後） |
| 完了 phase | — | `phase=10` / `phaseName=ShutdownComplete` |
| World | `published=4 retired=3 activeWorld=1` | `world_retirement_telemetry.json` の 4 / 3 / 1 |
| Reader | `readers=0` | `reader_registry_final.json` の `activeAtStage: []` |

`sh3_pendingRetire` が 1 ではなく 2 なのは、2 秒待ちの**後**に走る `m_coordinator.finalizeShutdown()`
（`SnapshotCoordinator::retireCurrentAndTarget`）が `publishEpoch()` で epoch を 1 進めてから
`GlobalSnapshot` を 1 件 enqueue するためである（追加調査で確定。
`doc/work113/20261113-2031_c3c9設計検討/` 参照）。

## 7. ビルド・テスト

| 項目 | 結果 |
| --- | --- |
| ビルド | `cmake --build build --config Release` → 256 ターゲット、エラー 0（警告 2 件は `MklFftEvaluator.h` の既存 C4390） |
| 追加ビルド | ログ項目名の修正後に `--target ConvoPeq` → 成功 |
| ctest | **61/61 成功**（348.06 s）。新規 AC1 は `ISRSemanticValidationRejects` 内で PASS |
| バイナリ確認 | exe に `c9_swap` / `headRetireEpoch` / `retireEpochLtMinReader` / `minReaderEpoch` / `currentEpoch` / `headScanned` が存在 |

## 8. ConvoPeq.md の再生成

| 項目 | 旧 | 新 |
| --- | --- | --- |
| ヘッダ stamp | 2026-11-12 00:48:16 | **2026-11-13 20:16:37** |
| サイズ | 7,011,745 B | 7,223,852 B |
| スナップショット | `doc/ConvoPeqMD/20261112004816/` | `doc/ConvoPeqMD/20261113201637/` |

- `python output_sourcecode_markdown.py` で生成（`TARGET_ITEMS = src / build.bat / CMakeLists.txt`）。
- `--verify-snapshot doc/ConvoPeqMD/20261113201637/ConvoPeq.md` → `name == header: OK` /
  **`content == current: OK`**（working tree と一致）。
- A/C の API が含まれることを確認: `collectDrainBreakdown` 9 箇所、`DrainBreakdown` 14、
  `peekHead` 4、`inspectRetireHead` 2、`RetireHeadSnapshot` 3、
  `testAC1_drainBreakdownMatchesPredicate` 2、`c9_swap` 1、`retireEpochLtMinReader` 1。
- 2 MB 以下の分割を 4 ファイル生成し、**結合すると元と一致**することを確認（改行コードも保持）。

| 分割ファイル | サイズ |
| --- | ---: |
| `ConvoPeq_part01.md` | 1,993,209 B |
| `ConvoPeq_part02.md` | 1,988,438 B |
| `ConvoPeq_part03.md` | 1,985,715 B |
| `ConvoPeq_part04.md` | 1,256,742 B |

## 9. 未実施・未確認

| # | 内容 | 理由 |
| --- | --- | --- |
| 1 | 原因修正 | 承認範囲外（Owner 指示: 計測結果を見てから修正設計） |
| 2 | EQ 退役 143 件の内訳（どの操作で発生したか） | 今回の計測対象外。必要なら別途 |
| 3 | `retireBacklogCount_` を減らすべき正しい契機の設計 | 修正設計の領域 |
| 4 | Debug 構成での確認 | 今回は Release のみ |
| 5 | 実機 GUI での通常終了 | CLI 実行のみ（既存の計測と同じ条件） |

## 10. 残留リスク

- 診断ログは読み取りのみだが、`peekHead` が `deletionQueue` の slot を 1 回コピーする。ロック・CAS・
  書き込みは無く、producer join 済みの時点で 1 回だけ実行される。
- EQ 退役が 1 件でも起きると `retireBacklogCount_` が 0 に戻らないため、**この構成では
  `AudioEngine::isFullyDrained()` が真にならない**。2 秒の待ちは常に無駄になり、実際の回収は
  `~AudioEngine` の 5 秒 drain が担っている。これは「タイムアウトが常態」という以前の評価の
  構造的な裏づけである（EQ 退役が 0 件ならこの限りではない）。
- c3 は epoch 前進が 2 秒窓に無いことによる。修正する場合、epoch 前進の追加は retire 順序と
  Reader grace の契約に触れるため、設計判断が要る。

## 11. ロールバック

| 対象 | 撤去方法 |
| --- | --- |
| ログ項目（`ReleaseResources.cpp:711-719`, `:756-780`） | 追加分を削除（行は前回の 2 行に戻る） |
| `EpochDomain::RetireHeadSnapshot` / `inspectRetireHead()` | 削除 |
| `DeferredDeletionQueue::peekHead()` | 削除 |
| `DrainBreakdown` / `collectDrainBreakdown()` | 削除 |
| テスト AC1 | 関数と `main()` の呼出しを削除 |

`isFullyDrained()` は変更していないため、戻す作業に判定ロジックの復元は含まれない。

## 12. 判定

- 承認範囲（A2 + C）の実装と計測は**完了**。3 回とも同じ値で、ログは 2 行、`ctest` は 61/61。
- **c3 の原因**: 先頭 entry の epoch（12）が `minReaderEpoch`（12）と等しく、2 秒窓で epoch が
  前進しないため `isOlder` が成立しない。
- **c9 の原因**: `retireBacklogCount_` = 143。EQ 退役で +1 されるが、本番の減算経路が無い。
- **c3 と c9 は独立**。今回の構成（EQ 退役 143 件）では、c3 を解消しても c9 が残るため
  タイムアウトは消えない。逆も同じ。
- 原因修正の設計は本承認の範囲外。次段階の判断材料として、上記 2 点を提示する。
