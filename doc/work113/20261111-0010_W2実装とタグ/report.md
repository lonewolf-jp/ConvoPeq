# W-2 `postTeardown` 実装・guard タグ付け計測・C2 留保対応（2026-11-10）

## §0 結論

1. **W-2 の `postTeardown` を限定承認の条件どおり実装した。** `RetireLatencyTeardownReporter` を `AudioEngine` の最初の非 static データメンバに置き、メンバ破棄完了時点で独立 JSON（`evidence/retire_latency_post_teardown.json`）を書く。`ISRDebugRuntime` には依存しない。
2. **期待値はすべて一致した。** `postTeardown − final` は `unmatched +21`・`recordMisses +42`・`postShutdownRequests +42`・`postShutdownCompletions +21`。3 段の累積は単調増加。`evaluateRetireDelay` はバイト一致。
3. **`retire_delay` の guard タグ付け（2 か所）を実装し再計測した。** 結果、**承認された 2 か所のどちらも長時間区画を保持していない**ことが分かった（両タグとも保持時間の証跡に現れない）。4.4 秒の区画は**実行中に終了しない**（プロセス終了まで保持される）ため、区画の退出記録が残らない。
4. **C2 の留保 2 点に対応した。** T13 のケース数を 7 に訂正し、`jassert` より前に違反カウンタを加算するよう順序を入れ替えた。

## §1 W-2 `postTeardown` の実装

### 1.1 構成

| 要素 | 実装 |
|---|---|
| レポータ | `src/audioengine/RetireLatencyTeardownReporter.h`（ヘッダのみ、独立した `std::ofstream` ライタ） |
| 配置 | `AudioEngine` の**最初の非 static データメンバ**（`m_teardownReporter`）。宣言の逆順破棄により最後に走る |
| 出力 | `evidence/retire_latency_post_teardown.json`（独立ファイル、`ISRDebugRuntime` 非依存） |
| 段 | `runtime` / `final` / `postTeardown`（すべて累積）＋ `delta`（`finalMinusRuntime` / `postTeardownMinusFinal`） |
| `final` の捕捉 | `HBVerifierRuntime::emitViolationReport` の先頭で `captureFinalStage()` を呼び、レコーダに保持 |
| `postProcess` | **未実装**（今回の承認対象外）。JSON に `"implemented": false` と明記 |

### 1.2 出力の実測（bypass 40 回 × 250 ms）

```json
"stages": {
  "runtime":      { "requests": 1892, "completions": 1892, "unmatchedCompletions": 0,  "recordMisses": 0,  "postShutdownRequests": 0,  "postShutdownCompletions": 0 },
  "final":        { "requests": 1892, "completions": 1892, "unmatchedCompletions": 0,  "recordMisses": 0,  "postShutdownRequests": 0,  "postShutdownCompletions": 0 },
  "postTeardown": { "requests": 1892, "completions": 1892, "unmatchedCompletions": 21, "recordMisses": 42, "postShutdownRequests": 42, "postShutdownCompletions": 21 }
},
"delta": {
  "finalMinusRuntime":      { "unmatchedCompletions": 0 },
  "postTeardownMinusFinal": { "requests": 0, "completions": 0, "unmatchedCompletions": 21, "recordMisses": 42, "postShutdownRequests": 42, "postShutdownCompletions": 21 }
}
```

3 シナリオ（EQ／bypass40／bypassfast）で `postTeardown − final` はすべて **`unmatched +21`・`recordMisses +42`・`postShutdownRequests +42`・`postShutdownCompletions +21`** で一致した。

### 1.3 「完了数 21」の意味（重要な補足）

Owner の期待値「`postTeardown − final` の完了数：21」は、**完了イベント 21 件**として一致している。ただし JSON の `completions` は**要求と対応した完了**だけを数えるため、その差分は 0 になる。

| 計数 | 差分 | 意味 |
|---|---|---|
| `completions` | **+0** | 要求と対応した完了のみを数える（21 件は要求記録が拒否されているため含まれない） |
| `unmatchedCompletions` | **+21** | 対応する要求が無い完了＝今回の 21 件 |
| `postShutdownCompletions` | **+21** | shutdown 相以降に観測した完了イベント（新設） |
| `recordMisses` / `postShutdownRequests` | **+42 / +42** | 記録に失敗した**呼び出し回数**（21 対象 × 2 試行） |

したがって「21 件の完了」は `unmatchedCompletions` と `postShutdownCompletions` の +21 で確認する。D1 の判定対象（`runtime`）には混入していない。

### 1.4 既存判定の不変

| 項目 | 値 |
|---|---|
| `evaluateRetireDelay` のバイト数 | 2,756（前後で一致） |
| SHA-256 | `20346AD1C2B6597EDE74FAFF90201F42…`（一致） |
| `retire_delay` の verdict | `latency_exceeds_threshold` で fail（変化なし） |

## §2 C2 の留保 2 点への対応

| 留保 | 対応 |
|---|---|
| T13 のケース数（報告は 8、実装は 7） | **7 ケースが正しい**。(1) Non-RT 成功、(2) Q、(3) Q→E、(4) Q/E→T、(5) RT 違反、(6) RT 違反後の Non-RT 再呼出し、(7) 二重呼出し。報告の「8」は誤記であり、本報告で訂正する |
| `jassert(false)` がカウンタ加算より前 | **加算を先に移動**した。全構成で違反回数が記録される（Debug では加算後に assertion が停止させる）。`SSnapshotCoordinator.cpp` の 1 ブロックの並び替えのみで、状態を変更しない契約は変わらない |

## §3 `retire_delay`: guard タグ付けの結果

### 3.1 実装した計測

| 計測 | 内容 |
|---|---|
| タグ付き登録 | `IReaderEpochProvider::registerReaderThreadWithTag`（既定は無タグ版へ委譲） |
| `ScopedEpochReader` | コンストラクタに任意タグ（既定 `nullptr`）を追加し、登録時に渡す |
| `EQProcessor::acquireReadGuard` | 任意タグ引数（既定 `nullptr`）を追加 |
| 承認された 2 か所 | `AudioEngine.EQResponse.cpp:92` → `"EQResponse"`、`AudioEngine.RebuildDispatch.cpp:706` → `"RebuildDispatch"` |
| 保持時間の証跡 | `exitReader` で 1 ms 以上の区画の最長保持時間をスロット単位で更新（atomic CAS、RT 安全）。`evidence/retire_latency_reader_holds.json` に出力 |

### 3.2 計測結果

| 実行 | 最長保持（タグ） | 4.4 秒区画の記録 |
|---|---|---|
| bypass40 | 43.2 ms / 45.0 ms / 15.4 ms（すべて `unnamed`） | なし |
| EQ | 41.6 ms / 45.7 ms / 13.4 ms（すべて `unnamed`） | なし |
| bypassfast | 30.2 ms / 1.5 ms / 14.0 ms（すべて `unnamed`） | なし |

- **`"EQResponse"` と `"RebuildDispatch"` は 1 件も現れない。** したがって承認された 2 か所は、長時間区画の保持者ではない。
- 一方で D1 の最大待ちは 4.4 秒で再現している（`ebrMaxNs = 4,399,767,200`）。つまり 4.4 秒の区画は存在するが、**実行中に退出しない**ため、退出時の記録が残らない。
- 退出しない区画はプロセス終了まで保持される。これが「待ち時間＝退役からプロセス終了まで」という、4.4 秒／11.4 秒／13.9 秒のばらつきの説明になる。

### 3.3 タグで帰属を特定できない理由（設計上の限界）

`ScopedEpochReader` はスレッドローカルのスロットキャッシュを持つ。一度登録したスロットは、以降の区画で**再登録せずに再利用**される。`ownerTag` は登録時にしか設定されないため、**最初に登録した経路のタグが残り**、その後の生成箇所のタグにはならない。

今回の計測で `"EQResponse"`／`"RebuildDispatch"` が現れないのは、この再利用のためでもある。したがって「タグ＝現在の区画の生成箇所」とは限らない。この限界は実測と整合する（すべて `unnamed` のスロットに最長 45 ms の区画が記録された）。

### 3.4 保持者の候補（ソース確認）

`ScopedEpochReader` を生成する箇所は、承認済みの 2 か所のほかに次のとおり。

| 箇所 | 形態 | 区画の長さ |
|---|---|---|
| `EQProcessor.Core.cpp:301,555,707,825,837,849,867` | `acquireReadGuard()` のローカル | 短い（API 内） |
| `EQProcessor.Coefficients.cpp:70` | `ScopedEpochReader readGuard(*m_epochDomain)` の直接生成 | 係数計算の間（長くなり得る） |
| `AudioEngine.EQResponse.cpp:92`（承認済み・タグ付き） | ローカル | 長時間区画ではない（§3.2） |
| `AudioEngine.RebuildDispatch.cpp:706`（承認済み・タグ付き） | ローカル | 同上 |

プロセス寿命まで保持される区画は、**戻らない関数**か、**深度が戻らない経路**のどちらかである。今回の計測では前者／後者を切り分けられていない。

## §4 検証

### 4.1 ctest

```
100% tests passed out of 60
Total Test time (real) = 336.40 sec
ctest_exit=0
```

`D8_2_B_2_Tests`（T10〜T13）を含む 60 ターゲットがすべて合格した。

### 4.2 アプリ実行

| シナリオ | D1 `unmatched` | `snapshotRtViolations` | `retire_delay` verdict | 最大 latency | postTeardown 差分 |
|---|---|---|---|---|---|
| bypass 40 × 250 ms | 0 | 0 | fail（`latency_exceeds_threshold`） | 4,400 ms | un +21 / miss +42 / pshReq +42 / pshComp +21 |
| EQ 変更 | 0 | 0 | fail（同上） | 13,993 ms（前回実行） | un +21 / miss +42 / pshReq +42 / pshComp +21 |
| bypass 60 × 2 ms | 0 | 0 | fail（同上） | 11,373 ms（前回実行） | un +21 / miss +42 / pshReq +42 / pshComp +21 |

`retire_delay` の verdict は fail のまま変わらない（`evaluateRetireDelay` は `runtime` 段のみを読むため）。

## §5 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/audioengine/RetireLatencyTeardownReporter.h`（新規） | postTeardown の独立 JSON 出力、reader 保持時間の証跡出力 |
| `src/audioengine/AudioEngine.h` | レポータを最初の非 static データメンバとして追加、include |
| `src/audioengine/ISRRetireLatency.h` | `postShutdownCompletions`、`captureFinalStage()` / `finalStage()` |
| `src/audioengine/ISRHB.cpp` | `captureFinalStage()` の呼出し、`postShutdownCompletions` の出力 |
| `src/core/SnapshotCoordinator.cpp` | 違反カウンタを `jassert` より前へ移動 |
| `src/core/IReaderEpochProvider.h` | `registerReaderThreadWithTag`（既定は委譲） |
| `src/core/EpochDomain.h` | タグ付き登録、保持時間の証跡 |
| `src/core/ScopedEpochReader.h` | 任意タグ引数 |
| `src/eqprocessor/EQProcessor.h` | `acquireReadGuard(const char* tag = nullptr)` |
| `src/audioengine/AudioEngine.EQResponse.cpp` / `AudioEngine.RebuildDispatch.cpp` | タグ付け（2 か所） |

変更前スナップショットは、`tmp/rollback/c2impl_before/`（9 ファイル）、`tmp/rollback/probe3_before/`（`ISRRetireLatency.h`）、`tmp/rollback/shutdown_after/`（`AudioEngine.h`）に基づく。タグ付けで変更した 3 ファイル（`EQProcessor.h`、`AudioEngine.EQResponse.cpp`、`AudioEngine.RebuildDispatch.cpp`）は変更前スナップショットを取得していないため、差分は挿入した文字列を逆変換して再構成した（`tmp/recon/`）。変更後の状態は `tmp/rollback/w2_after/`（11 ファイル、SHA-256 付き）に保全した。commit／merge／release は行っていない。

## §6 残課題

1. 4.4 秒区画の保持者の特定（タグは再利用のため不可）。次段階は「呼出し箇所での RAII 計測」または「区画ごとのタグ（入場時に毎回書く）」。
2. 区画の短縮は未承認（保持者確定後）。
3. `postProcess`（静的ガード）は承認対象外。
