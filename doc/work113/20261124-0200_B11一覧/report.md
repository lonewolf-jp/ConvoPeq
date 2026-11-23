# B-11: 実測必須 artifact の一覧と、現状の gap

作業日時: 2026-11-24 02:00
対象: main `1712983d`（`src/audioengine/ISREvidenceExporter.cpp` ほか）

## 1. 目的

Owner 判断（2026-11-24）で、B-11 の証跡検証強化は「実測必須 artifact の一覧を先に確定する」ことになった。本稿は、いま何が実測され、何が placeholder で、どれを必須とすべきかをコードから確定させる。

## 2. 出力の構造

`EvidenceExporter::exportEvidence(snapshot, monotonicViolationCount, forceFullEvidence)` は 11 種の artifact を持つ。書出しの条件は 3 通り。

| 起動 | 書かれるもの |
| --- | --- |
| Release（通常） | `evidence_manifest.json` のみ（`minimalEvidence: true`。artifact 一覧は空） |
| Debug | 8 種（`debugArtifacts`） |
| `forceFullEvidence = true` | 11 種すべて |

ただし `forceFullEvidence` に `true` を渡す呼出し元は無い。ヘッダのコメントは「CI の run 経路だけが true を渡す」と書いているが、実際の呼出しは `AudioEngine.Commit.cpp:733` の `exportEvidence()` と `MainApplication.cpp:140` の `exportEvidence(nullptr, 0)` だけで、どちらも既定値のままである。

各 artifact は「先に実測ファイルがあればそれを残し、無ければ placeholder を書く」。manifest の `artifactDetail[].measured` は、**その時点でファイルが存在したか**（`fileExistsAndNonEmpty`）であって、内容が実測由来かは見ていない。

## 3. artifact の一覧

| artifact | schema | 実測値を書く主体 | 実測される条件 | Release 既定 | 区分 |
| --- | --- | --- | --- | --- | --- |
| `evidence_manifest.json` | `evidence_manifest_v1` | exporter | 常に | 書く（最小） | 台帳 |
| `closure_graph.json` | `closure_graph_v1` | `ClosureGraphWalker::emitClosureArtifact`（publish 時） | 公開（publish）が 1 回でも走る | 実測 | 必須 |
| `mutation_fault_trace.json` | `mutation_fault_trace_v1` | exporter（seal 違反数・monotonic 数） | 常に | 実測 | 必須 |
| `hb_graph_trace.json` | `hb_trace_v1` | `DebugRuntime::emitHBTrace` | Release 以外、または `isForceEvidenceEmission()` | placeholder | 必須（Debug/CI） |
| `hb_violation_report.json` | `hb_violation_report_v1` | `DebugRuntime::emitCIArtifacts` | 同上 | placeholder | 必須（Debug/CI） |
| `retire_timeline.json` | `retire_timeline_v2` | `AudioEngine::Commit` の `emitRetireTimeline` | Commit 経路が走る | 実測 | 必須 |
| `shutdown_trace.json` | `shutdown_trace_v1` | `ShutdownRuntime::emitShutdownTrace` | 終了処理が trace を書く | exporter は placeholder を先に書く（後に実測で上書き） | 必須 |
| `retire_latency_report.json` | `retire_latency_report_v1` | CI シナリオランナー | CI の計測を実行したとき | placeholder | CI 必須 |
| `payload_tier_report.json` | `payload_tier_report_v1` | CI シナリオランナー | 同上 | placeholder | CI 必須 |
| `publication_failure_log.json` | telemetry 由来 | exporter（`TelemetrySnapshot`） | `snapshot` を渡す呼出しがあるとき | 空 | 条件付き |
| `publication_progress_log.json` | telemetry 由来 | 同上 | 同上 | 空 | 条件付き |
| `deferred_health.json` | telemetry 由来 | 同上 | 同上 | 空 | 条件付き |
| `live_runtime_evidence.json` | `live_runtime_evidence_v1` | `writeLiveRuntimeEvidence`（プロセス観測） | `--verify-runtime-evidence-live` かつ `--cli-run` + `--cli-exit-ms` | 書かない | 必須（live モード時） |

派生して `runtime_budget_report.json`、`recovery_trace.json`、`runtime_snapshot.json` も書かれる（Release 以外）。

## 4. 実測必須の定義（提案）

「その run が証明だと名乗るなら、placeholder であってはならない」集合を次のように定める。

1. **常時必須**: `mutation_fault_trace.json`、`closure_graph.json`、`retire_timeline.json`、`shutdown_trace.json`
2. **Debug / CI 必須**: `hb_graph_trace.json`、`hb_violation_report.json`（Release では記録自体が無効なので、Release の証跡に含めるなら「未測定」と明示して証明から外す）
3. **CI 計測必須**: `retire_latency_report.json`、`payload_tier_report.json`（CI のシナリオ実行を伴う run のみ）
4. **live モード必須**: `live_runtime_evidence.json`（要求した run のみ）
5. **任意（理由必須）**: `publication_failure_log.json`、`publication_progress_log.json`、`deferred_health.json`（telemetry を渡していない run では「未測定」でよいが、理由を残す）

## 5. 現状の gap

1. **`measured` が実測由来を意味しない。** 実行前に `isr-seed-evidence.ps1` が置いたファイルでも `measured: true` になる。seed 側は `"provenance": "seed"` を持つので、判別材料は既にある。
2. **`forceFullEvidence` が実質デッド。** Release の `--verify-runtime-evidence` は manifest だけを書き、11 種は seed に依存する。ヘッダの記述と実装が食い違っている。
3. **HB 系は Release で構造的に未測定。** `DebugRuntime::emitCIArtifacts` は Release かつ `isForceEvidenceEmission()` が偽なら即 return する。Release の証跡で HB を「実測」と扱ってはいけない。
4. **telemetry 系は常に空。** `exportEvidence` に `TelemetrySnapshot` を渡す呼出し元が無い。
5. **`shutdown_trace.json` の順序。** exporter が先に placeholder を書き、その後の終了処理が実測で上書きする。上書きに失敗すると placeholder が残るが、manifest 上は `measured: true` になる（1 と同じ問題）。

## 6. 次に必要な作業

| # | 作業 | 根拠 |
| --- | --- | --- |
| 1 | manifest の `measured` を「実測由来か」に変える（`provenance: seed` と `status: not_measured` を実測扱いしない） | §5-1 |
| 2 | 検証器（`isr-verify-*`）に必須集合を入れ、placeholder と seed を落とす | §4 |
| 3 | CI の runtime evidence run で `forceFullEvidence = true` を渡す（または `isForceEvidenceEmission` を ON にする） | §5-2 |
| 4 | `TelemetrySnapshot` を渡す経路を作る（Commit のテレメトリ dump から） | §5-4 |
| 5 | Release の証跡では HB を「未測定」と明示し、証明の対象から外す | §5-3 |

1 と 2 が本質で、3 から 5 は「実測を増やす」作業である。順序は 1 → 2 → 3 が自然で、4 と 5 は独立して進められる。

## 7. 未確定

- CI の `isr-verify-evidence-provenance.ps1` が現在どの artifact を必須として扱っているかは、本稿では一覧化していない。実装前に確認が要る。
- `shutdown_trace.json` を「終了後に読む」検証にするか、「終了前に確定させる」設計にするかは、検証器の実行タイミング次第である。
