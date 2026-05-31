# Discovery Cross-check Report

## Scope

- Authority / Descriptor / Semantic / Publication / Construction
- Sources: grep, Serena, CodeGraph

## Cross-check Summary

### Authority

- grep: `RuntimeReadView`, `RuntimePublishView`, `getRuntimeGraph(...)` callsites を複数検出
- Serena: 同シンボル定義と使用箇所を `AudioEngine.h`, `AudioEngine.Commit.cpp`, `AudioEngine.*` で検出
- CodeGraph: `AudioEngine.h` 実体（`RuntimePublishView::graph`, `RuntimeReadView::graph`）を読み取り確認
- Verdict: 三系統一致（RuntimeGraph pointer露出は実在）

### Descriptor

- grep: `RuntimeState`, `kFieldDescriptors`(size=9), `validateDescriptorSet` を検出
- Serena: `kFieldDescriptors`, `validateDescriptorSet`, `PublicationSemantic::validateDescriptorSet` を検出
- CodeGraph: `AudioEngine.h` 上の RuntimeState セクションを確認
- Verdict: 三系統一致（descriptor管理契約は実在、実体との網羅性監査が必要）

### Semantic

- grep: `semanticHash`/publication precheck系を検出
- Serena: semanticHash assignment 群（generation/topology/execution/routing/publication/overlap/retire）を検出
- CodeGraph: RuntimeState/Commit読み取りで semantic/publication 経路を確認
- Verdict: 三系統一致（semantic source はあるが RuntimeGraph直参照残存）

### Publication/Retire

- grep: `runPublicationPrecheckNonRt`, `PublicationIntent`, retire intent/backlog 更新を検出
- Serena: precheck/retire/reclaim/quarantine の遷移関連を検出
- CodeGraph: `AudioEngine.Commit.cpp:260-420` で retire pending -> reclaim/quarantine を確認
- Verdict: 三系統一致（publish->retire state path は実在）

### Construction

- grep: `RuntimeState(BuilderToken)`, `createForBuilder`, `makeRuntimeReadView` などを検出
- Serena: builder/view construction のシンボル群を検出
- CodeGraph: `RuntimePublishView`/`RuntimeReadView` constructors の実体を確認
- Verdict: 三系統一致（construction pathは把握可能）

## Open Differences

- CodeGraph自然文クエリはノイズ混入があり、該当箇所抽出に不向きなケースあり。
- 対応: CodeGraphは file-content read で確証取得し、自然文クエリ結果は参考扱い。

## Status

- 差分比較票: 作成済み
- 未解決差分0件: 未達（CodeGraph自然文クエリの精度差をリスク登録済み）
- 追加探索2サイクル0件: 未達
- Stop Rule達成: 未達
