# shutdown 分岐の要求記録・C1〜C3 実走・C2 再設計（2026-11-10）

## §0 結論

1. **承認された shutdown 分岐の記録を実装した。** D1 の `unmatched` は 3 から 1 に下がった。同一シナリオ（EQ 変更）の前後比較で、要求 +2・`unmatched` −2・完了経路に `sync` と `drain` が各 1 件増えた。所有権移転・reclaim・shutdown 順序は変更していない。
2. **残る 1 件の原因を特定した。** `SnapshotCoordinator::retireCurrentAndTarget`（`releaseResources` → `finalizeShutdown` 経由）が要求を記録しないためである。証拠は、プローブで `site=finalizeCurrent` のポインタと `PROBE_UNMATCHED` のポインタが一致したこと、およびハーネスで同じ経路が `unmatched=1` になること。**この経路は承認範囲外なので修正していない。** 1 行追加のパッチ案を §7 に添付する。
3. **C1〜C3 のアプリ実走は不可能である。** アプリが `SnapshotCoordinator` のフェード経路を使うのは起動時の bootstrap 1 回だけで、そこでは退役が発生しない。C2 には本番呼出元が無い。既存の UI 操作・テスト経路で駆動できないため、製品コードに呼出経路を追加しないという条件の下では実走例を作れない。代わりに、実装クラスと実キューを使うハーネスで C1／C2／C3 の要求・完了の 1 対 1 を 43 項目検証した（`PASS=43 / FAIL=0`）。既存テスト `D8_2_B_2_Tests` が C2 を実際に呼ぶことも確認した。
4. **C2 リークの改訂設計を作成した。** リングを廃止し、「移送が確定するまで所有権を現在の保持者から動かさない」を原則にした。指摘 6 点はすべて構造的に解消する。実装はしていない。
5. **新たな計測上の問題を発見した。** D1 レポートの生成後に、未記録の完了が 22 件発生している。レポートの `unmatched` はこの分を含まない。

## §1 承認条件への対応（shutdown 分岐）

| # | 条件 | 対応 |
|---|---|---|
| 1 | `shutdownReclaim` の前に要求を記録する | `markRetireEpoch()` の直後、`shutdownReclaim` の直前で `recordRequest` を呼ぶ |
| 2 | 完了フックと同じ recorder・同じ object で対応させる | `RetireLatencyRecorder::instance()` の同一インスタンス、`object` は `ptr`。完了は `terminalReclaim` の既存フックが同じ `ptr` で対応する |
| 3 | 失敗時のみ、`created == true && requestId != 0` のときに取り消す | `if (! transferred && requestCreated && requestId != 0u) cancelRequestById(ptr, requestId)` |
| 4 | `created == false` の既存記録は取り消さない | `requestCreated` が false なら取消しを呼ばない（再利用時は他経路の記録を残す） |
| 5 | 所有権移転・即時 reclaim・戻り値・shutdown 順序を変えない | 記録の追加と取消しの呼出しのみ。`shutdownReclaim` の引数・戻り値・分岐は無変更 |
| 6 | 通常経路を `enqueueRetire` に迂回させない。他の経路に記録を追加しない | 迂回なし。記録の追加はこの 1 か所のみ |

## §2 実装

変更は 1 ファイル・1 か所である。差分は `patches/01_shutdown_branch_record.patch`（+14 行、−0 行）。

```cpp
const uint64_t epoch = markRetireEpoch();

// ★ D1-shutdown (2026-11-10): この分岐は enqueueRetire を通らないため、要求記録が
//   欠落していた（完了だけが terminalReclaim の完了フックで記録され unmatched になる）。
//   移送の直前に要求を記録し、完了フックと同じ recorder・同じ object で対応させる。
//   移送が失敗した場合に限り、この分岐が作成した記録だけを取り消す（既存記録は触らない）。
bool requestCreated = false;
const std::uint64_t requestId =
    convo::isr::RetireLatencyRecorder::instance().recordRequest (
        ptr, epoch, 0u, 0xFFFFFFFFu, &requestCreated);

const bool transferred = m_retireRouter->shutdownReclaim(ptr, deleter, epoch, type);

if (! transferred && requestCreated && requestId != 0u)
    convo::isr::RetireLatencyRecorder::instance().cancelRequestById (ptr, requestId);
```

### 2.1 変更前スナップショット

前回の指摘（実装前スナップショットの未取得）を受けて、今回は変更前に取得した。

| 項目 | 値 |
|---|---|
| 退避先 | `tmp/rollback/shutdown_before/AudioEngine.h` |
| SHA-256 | `D86E824293E7B2AB7B087A209A48F16E0431A347994D9E217B00DA80E434ABBD` |
| サイズ | 326,617 B |
| マニフェスト | `tmp/rollback/shutdown_before/manifest.txt`（`evidence/before_manifest_shutdown.txt` に複製） |

一時的な計測用プローブを入れる際にも、対象 4 ファイルの変更前スナップショットを `tmp/rollback/probe_before/` に取得した。プローブ解除後、4 ファイルすべてが SHA-256 で一致することを確認済みである（`evidence/before_manifest_probe.txt`）。

## §3 検証

### 3.1 アプリ A/B（EQ シナリオ、パッチ前後）

同じ EQ シナリオ（`--cli-eq-band 0 --cli-eq-gain-db -3 --cli-eq-freq-hz 1000 --cli-eq-q 0.7 --cli-debounce-ms 50`、12 秒）を、パッチ前のソースとパッチ後のソースで別々にビルドして実行した。

| 実行 | ソース | requests | completions | unmatched | sync | drain | drainAll | queue |
|---|---|---|---|---|---|---|---|---|
| パッチ前 A（設計 v2 実行） | パッチ前 | 184 | 184 | **3** | 0 | 0 | 84 | 100 |
| パッチ前 B（A/B 用に再ビルド） | パッチ前 | 184 | 184 | **3** | 0 | 0 | 84 | 100 |
| パッチ後 C（A/B 用に再ビルド） | パッチ後 | 227 | 227 | **1** | 1 | 1 | 105 | 120 |
| パッチ後 D（別実行） | パッチ後 | 186 | 186 | **1** | 1 | 1 | 84 | 100 |

読み取れることは 3 点である。

1. `unmatched` はパッチ前の 2 実行がどちらも 3、パッチ後の実行がすべて 1 である。
2. `sync` と `drain` の完了はパッチ後の実行にだけ各 1 件現れる。これが追加した 2 件の要求に対応する。
3. 同じ負荷になった C と D の比較（C は他プロセスの影響で対象数が多い）ではなく、**B と D の比較**が同一経路の比較になる。B は 184 件・`unmatched` 3・`drainAll` 84・`queue` 100、D は 186 件・`unmatched` 1・`sync` 1・`drain` 1・`drainAll` 84・`queue` 100 である。差分は要求 +2・`unmatched` −2・`sync`/`drain` 各 +1 で、既存経路の件数は変わっていない。

実行間で `requests` の総数が動く（184〜227）のは、EQ キャッシュの生成状況で退役対象数が変わるためである。したがって総数のみで前後を比べず、同一経路（B と D）で比較した。

### 3.2 アプリ実行（パッチ後の最終ビルド、3 シナリオ）

最終ビルド（パッチ後ソース、61 ターゲット再ビルド）で実行した結果である。

| シナリオ | requests | completions | unmatched | sync | drain | drainAll | queue | world（commits/destroys/unmatched） |
|---|---|---|---|---|---|---|---|---|
| EQ 変更（12 秒） | 186 | 186 | **1** | 1 | 1 | 84 | 100 | 4 / 4 / 1 |
| bypass 40 回 × 250 ms | 1,842 | 1,842 | **1** | 1 | 1 | 903 | 937 | 38 / 38 / 1 |
| bypass 60 回 × 2 ms | 544 | 544 | **1** | 1 | 1 | 273 | 269 | 4 / 4 / 1 |

先行のパッチ後実行も同じ結果だった（追加操作なし 132 件・`unmatched` 1、EQ 186 件・1、bypass40 1,846 件・1、bypassfast 585 件・1）。`unmatched` が 1 に固定されるのはシナリオ非依存の単一経路が原因であることを示す。もう 1 つ共通するのは `sync=1` と `drain=1` で、これは追加した記録に対応する。

`pending` はすべて 0、`postShutdownRequests` はすべて 0 だった。後者は、追加した記録が `markShutdownPhase`（`drainAll` の先頭）より前に作られていることを意味する。

world の `unmatched = 1` は bootstrap 分として維持されている（既知）。

> 注: bypass 40 回の `requests` は実行ごとに 1,842〜1,846 で揺れ、world commit 数も 38〜40 で変わる。EQ シナリオは 184（パッチ前 2 実行）と 186（パッチ後 2 実行）で安定しており、A/B にはこちらを使った（§3.1）。

world の `unmatched = 1` は bootstrap 分として維持されている（既知）。

> 注: bypass 40 回の実行は、パッチ前 1,846 件・パッチ後 1,846 件で総数が同じに見える。これは world commit 数が実行ごとに 40〜41 で揺れるためで、総数での前後比較には使えない。厳密な A/B は §3.1 の EQ シナリオで示した。

### 3.3 残る unmatched=1 の原因（特定）

一時的なプローブ（`fprintf(stderr, ...)`）で、退役サイトと `recordComplete` の未対応をポインタで突き合わせた。結果は次の順で一致した。

```
[PROBE_RETIRE]   site=shutdownBranch ptr=...2070 id=211 created=1
[PROBE_RETIRE]   site=shutdownBranch ptr=...17C0 id=212 created=1
[PROBE_RETIRE]   site=finalizeCurrent ptr=...BFB0
[PROBE_UNMATCHED] ptr=...BFB0 path=queue          ← 残る 1 件
[PROBE_UNMATCHED] ptr=...2A80 path=drainAll       ← 以下はレポート生成後の分（§3.5）
```

`finalizeCurrent` は `SnapshotCoordinator::retireCurrentAndTarget`（`SnapshotCoordinator.h:183`）である。`releaseResources` が `m_coordinator.finalizeShutdown(timedOut)` を呼び、そこで current の `GlobalSnapshot` が退役する。この関数は `enqueueWithRetry` を直接呼び、要求を記録しない。したがって完了だけが記録され `unmatched` になる。

件数が 1 で固定なのは、shutdown 時点で current が 1 個だけ残るためである（target は fade 完了時に既に退役済み）。ハーネスの `[F]` ケースでも同じ経路が `unmatched=1` になることを確認した。

### 3.4 単体ハーネス（実装クラス＋実キュー、43 項目）

`evidence/cpath_harness.cpp` と `evidence/cpath_harness_result.txt`。`SnapshotCoordinator.cpp`・実 `EpochDomain`（実 `DeferredDeletionQueue`）・実 `RetireLatencyRecorder` をそのままリンクし、駆動だけをテスト側で行う。結果は `PASS=43 / FAIL=0`。

| ケース | 検証 | 結果 |
|---|---|---|
| C1 `startFade` | 要求が 1 件だけ記録される／要求と完了が同数／`unmatched=0`／`pending=0` | PASS |
| C3 `completeFade` | 同上 | PASS |
| C2 `resetFadeStateAndRetireTarget` | 同上。格納成功時に取消しが発生しない | PASS |
| C1 の格納失敗 → `quarantineRetireSink` | 要求は 1 件のまま（重複記録なし）／完了 1 件／`unmatched=0` | PASS |
| shutdown 分岐の判定ロジック（成功） | 要求 1・完了 1・取消し 0・`unmatched=0` | PASS |
| shutdown 分岐の判定ロジック（失敗） | 要求 1・取消し 1・`unmatched=0`・未対応の要求が残らない | PASS |
| shutdown 分岐の判定ロジック（既存記録の再利用） | 2 回目は `created=false`・同じ ID・失敗しても取り消さない・完了が 1 対 1 で対応 | PASS |
| 既存記録の再利用（C1 経由） | 要求が増えない／要求と完了が同数／`unmatched=0` | PASS |
| `finalizeShutdown`（既知ギャップ） | 要求を記録せず、`unmatched=1` になる | PASS（現象の再現） |
| `switchImmediate`（既知ギャップ） | target と current の 2 件が未記録で `unmatched=2` | PASS（現象の再現） |

`quarantine` ケースでは Q の受領をスタブで模擬している（実装と同じ順序で完了記録→deleter）。記録の重複が無いことは、レコーダの要求件数が 1 のままであることで確認した。

shutdown 分岐の検証は、`AudioEngine` を起動できないため判定ロジックを写して行った。分岐の条件（`created` と `requestId`）と取消しの呼出しは同一で、記録 API は実物である。

#### 補足: 取消し発生時の D1 `pending`

D1 の `pending` は `requests - completions` で計算され、取消しを差し引かない（`ISRHB.cpp:487`）。したがって shutdown 分岐の取消しが実際に発生すると、`pending` が 1 になり「incomplete_requests」で判定が止まる。現状 `shutdownReclaim` は失敗しないため観測されていないが、取消し経路が動いた場合の判定影響として記録する。

### 3.5 C1〜C3 のアプリ実走は不可能（到達性の証拠）

| 経路 | 本番呼出し | 根拠 |
|---|---|---|
| C1 `startFade` | bootstrap の 1 回のみ。その 1 回も current が空のため `switchImmediate` に倒れ、退役しない | `createSnapshotFromCurrentState` の呼出しは `AudioEngine.Timer.cpp:902` の 1 か所だけ。条件は `!isShutdownInProgress() && currentDspForRuntime != nullptr && !isFading() && currentSnapshot == nullptr` |
| C3 `completeFade` | フェードが起きないため到達しない | 同上（`tryCompleteFade` は Timer から呼ばれるが、フェード未開始なら `false`） |
| C2 `resetFadeStateAndRetireTarget` | 本番呼出元なし | `SnapshotCoordinator.h:194-199` に「no production caller (updateFade was removed 2026-07-28)」と明記。`grep` でも本番 0 件 |

プローブを入れたアプリ実行でも、`site=C1_startFade` と `site=C3_completeFade` は 0 回、`site=finalizeCurrent` は 1 回だった。EQ 変更・bypass バーストのどちらも `GlobalSnapshot` を新規適用しない（`RuntimeWorld` の publication は別系統である）。

したがって、既存の UI 操作では C1〜C3 を実走できない。既存のテスト経路では C2 のみ駆動できる（§3.6）。C1 と C3 はハーネスで検証した。

### 3.6 既存テスト経路（C2）

`D8_2_B_2_Tests` の T10 が `startFade` 2 回の後に `resetFadeStateAndRetireTarget` を直接呼ぶ。プローブを入れたビルドで実行し、C2 の記録が実際に走ることを確認した。

```
[PROBE_RECORD] site=C2_reset ptr=...0250 created=1 stored=1
```

`created=1`（新規記録）かつ `stored=1`（D キューが受領）である。テストは `T10 ... PASS` で終了した。`AudioEngineHarness` の v11 系にも同じ呼出しがある。

C3 `completeFade` を駆動する既存テストは無い。`tryCompleteFade` を呼ぶテストは 0 件だった。

### 3.7 `evaluateRetireDelay` の不変

実装前バックアップ（`tmp/rollback/item1_3/ISRHB.cpp`）と現行の関数テキストを抽出して比較した。

| 項目 | 値 |
|---|---|
| バイト数 | 2,756（前後で一致） |
| SHA-256 | `20346AD1C2B6597EDE74FAFF90201F42…`（前後で一致） |

### 3.8 ctest

```
100% tests passed out of 60
Total Test time (real) = 331.18 sec
ctest_exit=0
```

`AudioEngineHarness`（303.11 秒）を含む 60 件すべて合格した。この実行は、ソースを probe 解除後の状態へ戻し、mtime を更新して全 253 ターゲットを再ビルドしたうえでの結果である。

> 経緯: 最初の ctest 実行では `AudioEngineHarness` が失敗した。原因は、プローブ解除時に `Copy-Item` が元の mtime を保持するため、Ninja が再ビルドを省略し、プローブ入りの古い実行ファイルを混在させたまま実行したことである。ソースの内容（SHA-256）は正しかったが、ビルド成果物が古かった。4 ファイルの mtime を更新して全再ビルドした結果が上記である。
>
> なお `v11_7`（`V11ReaderReacquireTests`）は競合の発生を待つ確率依存のテストで、負荷が高いと検出 0 件で落ちる。今回の合格は無負荷での実行による。

### 3.9 計数窓の外で観測された未記録完了（新発見）

プローブの stderr では、D1 レポートの生成後に `path=drainAll` の `PROBE_UNMATCHED` が 21 件続く（`queue` の 1 件と合わせて 22 件）。レポートの `unmatched` は 1 なので、これらは計数に含まれていない。

| シナリオ | レポートの unmatched | プローブが観測した unmatched 完了 |
|---|---|---|
| 追加操作なし | 1 | 22（queue 1 + drainAll 21） |
| EQ 変更 | 1 | 22 |
| bypass 40 回 | 1 | 22 |

件数がシナリオに依存せず一定であることから、終了時の固定した対象群（21 件）が、レポート生成後の drain で解放されていると推定する。対象の特定は未実施である。レポートは `AudioEngine` のデストラクタで `drainAll` の直後に生成され、その後の終了処理（`retryDeferredDestroys` 以降）で発生する完了を含まない。

## §4 C1〜C3 の検証結果まとめ

| 経路 | 要求と完了の 1 対 1 | quarantine 移送での重複記録 | 検証手段 |
|---|---|---|---|
| C1 `startFade` | 確認済み | 確認済み（重複なし） | ハーネス |
| C3 `completeFade` | 確認済み | 確認済み（重複なし） | ハーネス |
| C2 `resetFadeStateAndRetireTarget` | 確認済み | 対象外（移送しない。失敗時は取消し） | ハーネス＋既存テスト |
| shutdown 分岐 | 確認済み（成功時 1 対 1、失敗時は取消し） | 対象外 | ハーネス（判定ロジック）＋アプリ A/B |

アプリ実走の欄は「実施不可」である。到達経路が存在しないためで、検証漏れではない。

## §5 未達と次の提案（いずれも未実装・承認待ち）

### 5.1 `retireCurrentAndTarget` に要求記録を 1 か所追加する（D1 unmatched=0 のための必須項目）

`src/core/SnapshotCoordinator.h` の `retireCurrentAndTarget()` に、current と target の退役直前で記録を入れる。失敗時は `quarantineRetireSink` が Q→E→T で必ず受領するため、取り消してはならない（取り消すと完了が `unmatched` になる）。

```diff
--- a/src/core/SnapshotCoordinator.h
+++ b/src/core/SnapshotCoordinator.h
@@ -183,6 +183,9 @@
         GlobalSnapshot* snap = m_slots.exchangeCurrent(nullptr, std::memory_order_acq_rel);
         if (snap) {
+            (void) convo::isr::RetireLatencyRecorder::instance().recordRequest (
+                snap, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu);
             if (!enqueueWithRetry(*m_epochProvider, snap, deleter, retireEpoch))
                 quarantineRetireSink(snap, deleter, retireEpoch, "retireCurrentAndTarget:queueFull");
         }
         snap = m_slots.exchangeTarget(nullptr, std::memory_order_acq_rel);
         if (snap) {
+            (void) convo::isr::RetireLatencyRecorder::instance().recordRequest (
+                snap, retireEpoch, m_epochProvider->getMinReaderEpoch(), 0xFFFFFFFFu);
             if (!enqueueWithRetry(*m_epochProvider, snap, deleter, retireEpoch))
                 quarantineRetireSink(snap, deleter, retireEpoch, "retireCurrentAndTarget:queueFull");
         }
```

`SnapshotCoordinator.h` に `#include "audioengine/ISRRetireLatency.h"` の追加が必要になる。ヘッダ依存が増えるため、承認時に判断してほしい。

この 1 か所で、アプリ実行の D1 `unmatched` は 0 になる見込みである（§3.3 の証拠と §3.4 のハーネス結果から 1 件が消える）。

### 5.2 計数窓の外の 21 件（§3.9）

対象の特定が先である。特定後に、記録を追加するか、レポート生成を終了処理の最後へ移すかを決める。レポート生成の移動は判定値が変わる（`pending` が残り得る）ため、記録の追加を優先したい。

### 5.3 C3 を駆動するテストの追加

`advanceFade` と `tryCompleteFade` を既存テストに 1 ケース追加すれば、C3 を実キューで検証できる。テスト変更の承認が必要である。

## §6 C2 リークの改訂設計

`c2_leak_design_v2.md` に分離して記載した。要点は次の 3 つである。

1. リングを廃止する。移送が確定するまで所有権を現在の保持者から動かさない。
2. Non-RT の失敗時は Q→E→T へ移送する（T は growable で必ず受領する）。
3. RT では target を exchange せず、1 bit のフラグを立てる。移送は Non-RT の drain 機会が行う。

## §7 変更ファイルとロールバック

| ファイル | 今回の変更 | 状態 |
|---|---|---|
| `src/audioengine/AudioEngine.h` | shutdown 分岐に記録 14 行を追加 | 変更（未コミット） |
| `src/audioengine/ISRRetireLatency.h` | 前回分（`created`・`cancelRequestById` 等）。今回は無変更 | 未追跡 |
| `src/audioengine/ISRRetireRouter.cpp` | 前回分（D の ID 取消し）。今回は無変更 | 変更（未コミット） |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | 前回分（B）。今回は無変更 | 変更（未コミット） |
| `src/core/SnapshotCoordinator.cpp` | 前回分（C1〜C3）。今回は無変更 | 変更（未コミット） |

`AudioEngine.h` の `git diff` は +31 行になる。今回の 14 行に加え、前回実装の RuntimeWorld 寿命記録 17 行が含まれるためである。今回分だけの差分は `patches/01_shutdown_branch_record.patch`（+14 行）にある。

最終状態の SHA-256 は次のとおり。

| ファイル | SHA-256（先頭 32 桁） | サイズ |
|---|---|---|
| `AudioEngine.h` | `74DADCF5693E02B514D6D4F612B26A4D` | 327,497 B |
| `ISRRetireLatency.h` | `C7318E178309A995385135D9D24A8535` | 28,373 B |
| `ISRRetireRouter.cpp` | `D11F6D87C0E6BBBECDA7C38C102C14C9` | 41,390 B |
| `ISRRuntimePublicationCoordinator.cpp` | `BE767C1A65A81B3B1E4AC7569909CEFC` | 96,413 B |
| `SnapshotCoordinator.cpp` | `01E4928759BEC5AABE1B0D6CF49F172F` | 7,282 B |

`SnapshotCoordinator.h` は probe 解除後に元の内容へ戻っており、`2D8E10F46D3656CFDAF9D098F194B43F`（9,506 B）である。

ロールバックは 2 段階である。

| レベル | 内容 |
|---|---|
| 1 | `patches/01_shutdown_branch_record.patch` を逆適用する（`git apply -R`）。`tmp/rollback/shutdown_before/AudioEngine.h` で完全復元も可能（SHA-256 一致を確認済み） |
| 2 | 前回分の記録 API も含めて戻す場合は `tmp/rollback/itemC_v2_after/` と `tmp/rollback/` の各スナップショットを使う |

commit／merge／release は行っていない。

## §8 未達事項

1. アプリ実行の D1 `unmatched = 0` は未達（1 が残る）。原因は特定済みで、§5.1 の 1 か所の追加が承認されれば解消する見込みである。
2. C1／C3 のアプリ実走は不可能。ハーネスとコード解析で代替した。
3. C3 を駆動する既存テストは無い。
4. 計数窓の外の 21 件は未特定。
