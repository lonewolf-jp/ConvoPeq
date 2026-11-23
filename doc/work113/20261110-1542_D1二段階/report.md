# A（D1）の 2 段階レポート・経路別判定・測定駆動（2026-11-10）

## §0 承認との対応

| Owner 判断 | 今回の実装 |
|---|---|
| ① レポート出力順（最終 drain 後に出力し、直前の状態も残す） | 実装した。`runtime`（最終 drain 直前）と `final`（最終 drain 後）の 2 段階を 1 つのレポートに併記し、最終 drain 後に書き直す |
| ② 経路別の合格基準（`queue` を含める、`drainAll` は分離） | 実装した。通常 = `sync`／`drain`／`queue`／`quarantine`、分離 = `drainAll`／`queueOverflow` |
| ③ 観測数を確保する駆動 | 実装した。`--cli-bypass-burst-alternate 1`（毎回値を反転）で 3,570 件の有効観測を確認 |
| ④ B 修正案 A の最終レビュー | 未着手（保留の指示どおり） |
| 50 ms の閾値 | 変更していない |
| `N_req = 59` | 変更していない（判定は通常経路の完了数で行う） |

変更したファイル（今回追加分）は次のとおり。

| ファイル | 内容 |
|---|---|
| `src/audioengine/ISRRetireLatency.h` | 2 段階記録（`StageStats` と `captureRuntimeStage`） |
| `src/audioengine/ISRHB.cpp` | `runtime`／`final` の出力、経路分類による判定 |
| `src/audioengine/ISRDebugRuntime.cpp` / `.h` | 最終 drain 後のレポート再出力 |
| `src/audioengine/ISREvidenceExporter.cpp` / `.h` | metadata 再付与の公開ラッパー |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | 最終 drain 後の再出力呼出 |
| `src/MainWindow.cpp` | 交互バースト（測定駆動） |

## §1 実装

### 1.1 2 段階レポート

- `ISRRetireRouter::drainAll()` の先頭で `markShutdownPhase()` を呼び、その時点のカウンタと経路別統計を `StageStats` として保存する（**runtime 段階**）。
- レポートの `retireLatencyD1` は次の構造になった。

```json
"retireLatencyD1": {
  "thresholdNs": 50000000,
  "required": 59,
  "runtime": { "captured": true, "requests": …, "completions": …, "pending": …,
               "unmatchedCompletions": …, "overflow": …, "idExhausted": …,
               "postShutdownRequests": …, "paths": [ … ] },
  "final":   { … 同上 … },
  "observations": [ … 最大 256 件 … ],
  "pending": [ … 最大 8 件 … ]
}
```

- 最終 drain の後、`DebugRuntime::emitFinalViolationReport()` がレポートを書き直す。書き直しの前に既存ファイルから `runId` と `generatedAtNs` を退避し、書き直し後に同じ値で metadata を再付与する（provenance を失わない）。
- `runtime` の `pending` は最終 drain 後も消さない（Owner 条件）。`final` は drainAll による解放を含む。
- 判定は **runtime 段階**で行い、`drainAll` の解放を通常経路の完了数に混ぜない。

### 1.2 経路別の判定

| 経路 | 扱い | 50 ms 判定 |
|---|---|---|
| `sync` | 通常 | 適用 |
| `drain` | 通常 | 適用 |
| `queue` | 通常（EBR 待ちを含む） | 適用 |
| `quarantine` | 通常 | 適用 |
| `drainAll` | 終了時の強制排水。通常と分離 | 適用しない（別枠で報告） |
| `queueOverflow` | 異常経路。通常に混ぜない | 適用しない（別枠で報告） |

判定順は、`overflow`／`idExhausted` → `incomplete_requests`（runtime の pending > 0）→ `insufficient_observations`（通常経路の完了数 < 59）→ `latency_exceeds_threshold`（通常経路の最大 > 50 ms）→ pass。

### 1.3 測定駆動

`--cli-bypass-burst-alternate 1` を追加した。バーストのたびに bypass 値を反転するため、同一値の連投で publish が起きない場合でも状態変化を作れる。80 回・250 ms 間隔の実行で、3,570 件の要求と 3,570 件の完了が得られた（§2.1）。

```cmd
ConvoPeq.exe --cli-run --cli-exit-ms 45000 --verify-runtime-evidence-live ^
  --cli-bypass-burst-count 80 --cli-bypass-burst-interval-ms 250 --cli-bypass-burst-alternate 1
```

### 1.4 測定対象の明記

今回の D1 は **DSP ハンドル／ライフタイムの退役要求から解放完了まで**である。`RuntimeWorld` の寿命は測っていない（`runtimePublicationBridge_.retire(...)` は入力検証のみで、world は解放待ち行列に積まれない）。レポートや説明で「RuntimeWorld の退役遅延」と表現しない。

## §2 検証

### 2.1 実測（交互バースト 80 回・250 ms、待機 45 秒）

| 項目 | runtime 段階 | final 段階 |
|---|---|---|
| requests | 3,570 | 3,570 |
| completions | 3,570 | 3,570 |
| pending | **0** | **0** |
| unmatchedCompletions | 3 | 3 |
| overflow / idExhausted | 0 / 0 | 0 / 0 |
| postShutdownRequests | 0 | 0 |
| `queue` | 1,827 件・**50 ms 超 1,642 件**・最大 **40.32 秒** | 同左 |
| `drainAll` | 1,743 件・50 ms 超 0 件・最大 **68.6 µs** | 同左 |

`retire_delay` の判定は **fail**（`latency_exceeds_threshold`、観測 1,827 件、必要 59 件、最大 40.32 秒）。`insufficient` ではなく、閾値超過として確定した。

**駆動なしの実行**（`--cli-run --cli-exit-ms 4000`、バーストなし）でも次の値が得られた。

| 項目 | 値 |
|---|---|
| requests / completions / pending | 130 / 130 / 0 |
| `queue` | 67 件・50 ms 超 64 件・最大 22.16 秒 |
| `drainAll` | 63 件・最大 38.6 µs |
| 判定 | fail（`latency_exceeds_threshold`、観測 67 件） |

この実行でも通常経路の観測が 59 件を超え、判定は `insufficient` ではなく閾値判定になった。

- 有効観測は通常経路で 1,827 件あり、`N_req = 59` を大きく上回る。
- `pending` は 0 で、未完了による不足は解消した。
- `unmatchedCompletions = 3` は、RT 経路（`retireRT`）が要求フックを通らないためである（測定対象外の経路として記録されている）。

### 2.2 レポートの構造確認

`hb_graph_trace.json` は `hb_trace_v1` のまま（イベント 4 件、D1 の混入なし）。`hb_violation_report.json` は `hb_violation_report_v1` のまま `retireLatencyD1` を持ち、`runtime`・`final`・`observations`・`pending` を備える。

最終 drain 後の書き直しで metadata が失われないことも確認した（`provenance=runtime`、`runId=runtime-local`、`generatedAtNs=7336662717400`）。

### 2.3 v5 による実証跡の検査

生成された実証跡に v5（2 検査入り）を実行した。スキーマ検査は通過し、`status = violation` により `HB violation report indicates violation` で不合格になった。判定どおりの不合格であり、新しいレポート構造を v5 が読み取れている（スキーマ検査は通過）。

### 2.4 ctest

Release 構成で全ターゲットをビルドし、`build/` で `ctest -C Release --output-on-failure` を実行した（最終コード）。

```
100% tests passed out of 60
Total Test time (real) = 377.35 sec
ctest_exit=0
```

途中の実行（metadata 修正の前）では 59/60 で、`HeadlessAudioPathVerification` が `ConvoPeq did not exit within timeout (30s)`（30.55 秒）で落ちた。これは既知の終了時間と 30 秒タイムアウトの競合であり、機能判定の失敗ではない。最終コードの実行では同じテストを含めて 60 件すべて合格した。

## §3 判明したこと

1. **`queue` 経路の D1 は EBR 待ちで支配される。** 1,827 件中 1,642 件が 50 ms 超、最大 40.32 秒だった。一方、終了時の強制排水（`drainAll`）は最大 68.6 µs である。50 ms を `queue` に適用する Owner 判断のとおり、D1 は fail になる。
2. **交互バーストで観測数が安定した。** 同一値の連投では 23 件しか出ないことがあったが、反転方式では 3,570 件（pending 0）が得られた。バーストなしでも 130 件（pending 0）が得られている。
3. **2 段階記録は機能している。** runtime と final が同じ値になったのは、最終 drain で未完了が残らなかったためである（`pending = 0`）。未完了がある実行では runtime の `pending` が残り、final で解消したことが並べて読める。
4. **`unmatchedCompletions = 3` は RT 経路の取りこぼし**であり、要求フックが Non-RT の `enqueueRetire` にあるためである。RT 経路を測るには別のフックが必要（未実装）。

## §4 未確認事項と残課題

- `queue` の 40 秒級の待ちが「読者 epoch の停滞」によるものか、drain の契機が少ないためかは未特定（EBR 待ちであることは確実だが、内訳は未計測）。
- `RuntimeWorld` の寿命測定は未実装（別の測定対象）。
- RT 経路（`retireRT`）の要求記録は未実装。
- 実機負荷時の分布、TSan／LSan、聴感確認。
- ctest（実行中）。

## §5 承認をお願いする事項

1. **`queue` 経路の 40 秒級の待ちをどう扱うか。** 現状は Owner 判断どおり 50 ms を適用して fail になる。EBR 待ちを短くする設計変更（drain 契機の追加）を行うか、この値を「設計上の待ち」として別枠報告にするかの判断が必要。
2. **`RuntimeWorld` 寿命の測定**を別途設計するか。
3. **RT 経路（`retireRT`）の要求記録**を追加するか（`unmatched` を減らす）。
4. B 修正案 A の最終レビュー（保留のまま）。
