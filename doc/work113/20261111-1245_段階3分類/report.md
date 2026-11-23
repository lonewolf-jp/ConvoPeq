# 段階③ `firstScanTs` と 3 分類診断の実装

Owner 承認（2026-11-11「段階③ firstScanTs・3分類 着手承認」）に基づく実装報告である。**domain tag の追加は承認範囲外**のため実装していない。commit／merge／release も未実施（HEAD `c94d3767`）。

## §0 要旨

- `DeferredDeletionQueue` の entry に `firstScanTsNs` を追加し、**最初の reclaim 走査時に 1 回だけ**記録する。
- 観測を `driverWait` / `ebrWait` / `execWait` / `unclassified` に分解し、合成が `latency` と厳密に一致することを検証した。
- 既存の `retireLatencyD1`（判定に使う）は**一切変更していない**。分類は `retireLatencyClassified` という別セクションに出力する。
- 実測で「これまで一括りに `ebr` と呼ばれていた時間」が 2 種類に分かれることが確認できた（§4）。**eq シナリオの最大値は走査待ち（driverWait）**であり、以前の「epoch 前進待ち」という説明は誤りだった（§5 で訂正）。

## §1 実装

### 1.1 `src/DeferredDeletionQueue.h`

| 変更 | 内容 |
|---|---|
| entry | `uint64_t firstScanTsNs{0}` を追加（trivially copyable 維持） |
| enqueue | `entry.firstScanTsNs = 0;` でスロット再利用時の残留を防ぐ |
| `reclaim()` | 先頭 entry を評価する直前に `if (entry.firstScanTsNs == 0) entry.firstScanTsNs = steadyNowNs();` |
| `recordComplete` | Queue 経路と DrainAll 経路で `firstScanTsNs` を渡す |

- 書き手は consumer（`reclaim()`）だけであり、`seq` の acquire 後＝producer の書き込み完了後に読むため、データ競合はない。
- 時計は `steady_clock`（`RetireLatencyRecorder::nowNs()` と同一基準）。異なる時計を混ぜると分類が壊れるため、コメントで明示した。

### 1.2 `src/audioengine/ISRRetireLatency.h`

- `recordComplete(object, path, firstScanTsNs = 0)` に拡張（既定値により既存呼出しは無変更でコンパイル可）。
- 純関数 `classifyTimes(ts, firstScanTs, safeTs, now)` を追加（テスト可能にするため public static）。

```text
latency = now - ts
safeTs == 0            -> excluded（既存 unobservedSafe に計上・分類しない）
firstScanTs == 0       -> unclassified = latency（走査されていない）
firstScanTs < ts       -> inconsistent（consistencyErrors++）
safeTs < firstScanTs   -> inconsistent（consistencyErrors++）
sum > latency          -> inconsistent（時計の異常。0 埋めしない）
それ以外               -> driverWait = firstScanTs - ts
                          ebrWait    = safeTs - firstScanTs
                          execWait   = now - safeTs
                          unclassified = latency - sum（実残差。通常 0）
```

- 集計はパス別に `driverWait` / `ebrWait` / `execWait`（count / sumNs / maxNs / over50ms）と `unclassified`（count / sumNs）、`excluded`（count）。
- 総計は `requests` / `completions` / `classified` / `unclassified` / `excluded` / `consistencyErrors` / `pending`。
- **未分類は実際に未分類のときだけ数える**（分類できた要求を unclassified に加算しない）。

### 1.3 `src/audioengine/ISRHB.cpp`

`retireLatencyD1` の直後に `retireLatencyClassified` を追加した。

```json
"retireLatencyClassified": {
  "note": "diagnostic only; the verdict uses retireLatencyD1.runtime and is unchanged",
  "thresholdNs": 50000000,
  "required": 59,
  "paths": [ { "name": "queue",
               "driverWait": {"count":…,"sumNs":…,"maxNs":…,"over50ms":…},
               "ebrWait": {...}, "execWait": {...},
               "unclassified": {"count":…,"sumNs":…},
               "excluded": {"count":…} } ],
  "totals": { "requests":…, "completions":…, "classified":…, "unclassified":…, "excluded":…,
              "consistencyErrors":…, "pending":… },
  "countsConsistent": true
}
```

`note` に「判定は従来どおり」であることを明記し、判定関数・閾値・`N_req` には触れていない。

## §2 検証

`STG11EQRetireTests` に `checkLatencyClassificationContract()` を追加した（7 項目：5b を含む）。

| # | 検証 | 期待 | 結果 |
|---|---|---|---|
| 1 | 合成 | ts=1000 / firstScan=5000 / safe=8000 / now=9000 → 4000+3000+1000+0 = 8000 | PASS |
| 2 | `firstScanTs == 0` | `unclassified` = latency 全額（0 埋めしない） | PASS |
| 3 | `safeTs == 0` | 分類対象外（`excluded`） | PASS |
| 4 | `firstScanTs < ts` | `inconsistent` ＋ `unclassified` = 全額 | PASS |
| 5 | `safeTs < firstScanTs` | `inconsistent` ＋ `unclassified` = 全額 | PASS |
| 5b | `safeTs < ts`（安全時刻が要求より前） | `inconsistent`（`excluded` にしない） | PASS |
| 6 | 実キュー由来の観測 | `firstScanTs` が記録され、合成誤差 0、`classified + unclassified + excluded == completions` | PASS（`observations_with_firstScan=235 / classified=235 / compositionErrors=0`） |

- テスト実行: `STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b/Gate1/Stage1/Stage3)`
- ctest: **60/60 合格（363.75 秒：修正 2 件を含む最終ツリー）**。
- 実装中に 2 件の不具合を検出して修正した。
  1. 分類成立時にも `unclassified` カウンタを加算していたため `classified + excluded != completions` になった（残差があるときだけ数えるよう修正）。
  2. `recordComplete` が分類器に渡す `safeTs` を `(safeTs >= ts) ? safeTs : 0` に丸めていたため、`safeTs < ts` の順序不整合が **excluded に隠れていた**。
     生の `safeTs` を渡すよう修正し、テスト 5b で固定した（`observations` 側の `safeTsNs` は従来どおりのクランプ値で、D1 の出力は不変）。

## §3 既存判定が変わっていないこと

| 項目 | 確認 |
|---|---|
| 50 ms 閾値 | `kThresholdNs` 定義に差分なし |
| `N_req = 59` | `retireLatencyD1.required` の出力行に差分なし（実測でも 59） |
| `evaluateRetireDelay` | 差分なし（分類は判定関数の外で計算・出力） |
| `retireLatencyD1` の出力 | キー集合が段階①前後で同一（`thresholdNs` / `required` / `snapshotRtViolations` / `timing` / `runtime` / `final` / `observations` / `pending`） |
| verdict | 3 シナリオとも `fail (latency_exceeds_threshold)` のまま。分類は診断であり判定を改善して見せない |

## §4 実測（3 シナリオ・12 秒）

最終ビルド（修正 2 件を含む）での実測。

| シナリオ | classified | unclassified | excluded | consistencyErrors | countsConsistent |
|---|---|---|---|---|---|
| burst40 | 946 | 2 | 903 | 0 | true |
| burstfast | 290 | 2 | 294 | 0 | true |
| eq | 101 | 2 | 84 | 0 | true |

パス別（`queue`）の内訳:

| シナリオ | driverWait max | ebrWait max | execWait max | driverWait > 50 ms | ebrWait > 50 ms |
|---|---|---|---|---|---|
| burst40 | 730.1 ms | **2,178.2 ms** | 15.3 µs | 742 | 40 |
| burstfast | 740.8 ms | **2,164.3 ms** | 0.2 µs | 115 | 7 |
| eq | **11,914.8 ms** | 2,162.4 ms | 0.2 µs | 68 | 4 |

- `execWait` は全シナリオで**マイクロ秒**であり、deleter 実行そのものは遅くない。
- `excluded` は主に `drainAll`（強制解放）で safeTs を観測しない要求である（設計どおり分類から除外）。
- `unclassified` の 2 件は `sync`（700 ns 前後）と `drain`（11〜12 ms）で、いずれも `reclaim()` の走査を通らない経路＝`firstScanTs` が存在しない。
  設計 §5 のとおり**全額を unclassified として明示**しており、0 埋めはしていない。
- `consistencyErrors` は 3 シナリオとも 0。時刻の逆転や欠損は観測されなかった。

## §5 実測から判明したこと（以前の説明の訂正）

1. **burst 系の最大値（2.16〜2.18 s）は `ebrWait`**、すなわち「最初の走査は済んでいるが、epoch 安全条件が成立するまで待った」時間である。
   段階①の報告で「末尾 entry が次の pending まで待つ」と説明したのは、burst 系については正しい。
2. **eq の最大値（11.9 s）は `driverWait`**、すなわち「一度も reclaim 走査されないまま待った」時間である。
   段階①の報告でこれを「次の pending 待ち」と説明したのは**誤り**で、実際は走査そのものが起きていない。
   - EQ ドメインは段階①ドライバが 100 ms ごとに `tryReclaim()` するため、EQ の D キュー entry の `driverWait` は 100 ms を大きく超えない。
   - したがって `driverWait` が 730 ms / 11.9 s に達する entry は **EQ ドメイン以外（エンジン側 ISR ドメイン）のキュー**に属する可能性が高い。エンジン側は `tryReclaimResources()` の呼出しタイミングに依存し、周期的な走査ドライバを持たない。
   - ただし D1 の観測に **domain ID が無い**ため、帰属は断定できない。これが domain tag を必要とする理由である（§7）。

## §6 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/DeferredDeletionQueue.h` | entry に `firstScanTsNs`・enqueue でリセット・初回走査で記録・2 経路で recorder へ伝達・`steadyNowNs()` |
| `src/audioengine/ISRRetireLatency.h` | `Observation` に分類値、`classifyTimes()`、パス別・総計の分類集計、`recordComplete` の引数追加、`reset()`／`snapshot()` の拡張 |
| `src/audioengine/ISRHB.cpp` | `retireLatencyClassified` セクションの出力（既存セクションは不変） |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | `checkLatencyClassificationContract()`（7 項目：5b を含む） |

差分は `patches/stage3_only/` に 4 本置いた。

| ファイル | 作り方 | 備考 |
|---|---|---|
| `01_deferred_deletion_queue.patch` | HEAD 基準 diff から段階③関係のハンクだけを抽出（6/7 ハンク） | 同一ハンクに以前の未コミット作業が混じ得る |
| `02_isr_retire_latency.patch` | 段階③着手前のファイルを復元して diff（10 ハンク・厳密） | このファイルはまだ `git add` されていない（untracked）ため HEAD diff が使えない |
| `03_isrhb_json.patch` | HEAD 基準 diff のハンク抽出（1/5） | 上と同じ留意 |
| `04_tests.patch` | HEAD 基準 diff のハンク抽出（2/3） | 上と同じ留意 |

なお、HEAD `c94d3767` 基準の全体差分は前回までと同じ方法で取得できる（`git diff`）。段階①までの差分は
`doc/work113/20261111-1222_段階1実装/patches/stage1_only/` にある。

## §7 未着手・保留（Owner 判断待ち）

- **domain tag**: 未実装（承認範囲外）。Owner 指摘の 3 点（付与段階 / 伝播経路 / 集計と既存出力の対応）を設計差分として提示する必要がある。本報告 §5 の帰属不明は、この設計で解消する。
- **publish 理由タグのスレッド前提**（段階①への診断上の留保）: `m_publishReasonTag` は atomic だが「タグ書き込み」と「flush による消費」が別操作のため、複数の flush が同時に走ると理由が入れ替わり得る。対応方針は 2 案。
  1. `prepareToPlay` / `releaseResources` が Message Thread 専用であることを `jassert` で固定し、同時実行を契約違反として扱う（コード変更は 1 行）。
  2. 理由タグを「入口→flush」の引数として渡す（`flushPendingEpochAdvance(PublishReason)`）ように API を分ける（呼出し側 2 か所の変更）。
  - 段階①の競合テストは**あえて別スレッドから**ライフサイクル関数を呼んでいるため、案 1 を採るならテスト側の呼び出し方法も合わせて見直す必要がある。
- Reader レジストリ統合・ASan・commit／merge／release は未承認・未実施。

## §8 成果物

- 本報告: `doc/work113/20261111-1245_段階3分類/report.md`
- 差分: `patches/stage3_only/`、`patches/`
- 証跡: `evidence/`（分類テスト PASS ログ、3 シナリオの JSON、分類の抜粋、ctest ログ）
- 結合ソース: `ConvoPeq.md` と 2 MB 未満の分割
- Dropbox: `/ConvoPeq-stage3-20261111/`、`/ConvoPeq-source-20261111-stage3/`
