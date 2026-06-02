# notfinished3.md 妥当性検証レポート

- 対象: `doc/work14/notfinished3.md`
- 検証日: 2026-06-02
- 検証方針: 「主張の事実性（コード上で確認できるか）」と「結論の妥当性（未達/未証明と言えるか）」を分離して判定
- 証拠ソース: `src/audioengine/*`, `src/core/*`, `src/tests/*`

---

## 総評（先に結論）

`notfinished3.md` は、**重要な懸念をいくつか正しく捉えている一方、既に実装・検証済みの点を「未証明」としている箇所や、事実誤認（例: descriptor件数）も混在**しています。

- 妥当（事実+結論とも強い）: 7件
- 一部妥当（事実はあるが結論が強すぎる/前提依存）: 10件
- 不正確（現コード証拠と不一致）: 5件

今回提示されたレビューコメントを再照合した結果、**方向性は概ね妥当**でした。特に `#10/#11/#14/#22` の再評価は本レポートの結論と整合します。

---

## 主要証拠（抜粋）

1. **Bridge から AudioEngine フック呼び出しは実在**
   - `src/audioengine/AudioEngine.h:2769,2774,2785,2793,2808`
   - `engine_->runPublicationPrecheckNonRt`, `engine_->onRuntimePublishedNonRt`, `engine_->onRuntimeRetiredNonRt`, `engine_->enqueueDeferredDeleteNonRt`

2. **Publication 失敗時 reject は実装済み**
   - `src/core/RuntimePublicationCoordinator.h:95-114`
   - `validatePublicationNonRt(*worldOwner)` が false の場合に `retireRuntimePublishWorldNonRt` して publish しない

3. **schema completeness / monotonicity 等の precheck は実装済み**
   - `src/audioengine/AudioEngine.Commit.cpp:23-52,155-325`
   - `validateSemanticCompleteness`, `validateRuntimeGraphAuthorityContract`, sequence/generation monotonic check

4. **Observe path 単一路化を検査するテストが存在**
   - `src/tests/ObservePathSingleSourceTests.cpp`
   - `AudioEngine.Processing.AudioBlock.cpp` / `BlockDouble.cpp` に `authority.preparedCrossfade` 必須、`runtimeGraph->` 禁止

5. **Crossfade executor-local 契約テストが存在**
   - `src/tests/CrossfadeExecutorLocalContractTests.cpp`
   - `Commit/Timer` に `preparedCrossfade` を使わせない契約

6. **Field descriptor件数は 17 ではない**
   - `src/audioengine/AudioEngine.h:197` → `RuntimeFieldDescriptor, 21`
   - `src/audioengine/RuntimeGraph.h:54` → `RuntimeFieldDescriptor, 25`

7. **epoch/generation 対応は実装あり**
   - `src/audioengine/RuntimeBuilder.cpp:212-214` (`sequenceId`, `epoch`, `mappedRuntimeGeneration`)
   - `src/audioengine/AudioEngine.Commit.cpp:38-47,191,209`（整合・単調性チェック）
   - `src/audioengine/ISRRuntimePublicationCoordinator.cpp:87-101`（sequence/epoch/mappedGeneration の単調チェック）

8. **Semantic transaction state machine は実装あり**
   - 定義: `src/audioengine/ISRRuntimeSemanticSchema.h:529-549`
   - 適用: `src/audioengine/AudioEngine.Commit.cpp:157,183,321,341`

---

## 項目別判定（1〜22）

| # | 項目 | 判定 | コメント |
| --- | --- | --- | --- |
| 1 | AudioEngine が Publication Authority の一部を保持 | 妥当（事実）/結論は要件依存 | Bridge から Engine hook 呼び出しは実在。完全分離を必須とするかは設計方針次第。 |
| 2 | RuntimeWorld Self-contained 化未証明 | 一部妥当 | `AudioEngine* engine_` は事実。ただしそれだけで RuntimeWorld 非 self-contained とは断定不可。 |
| 3 | Observe Source 単一化未証明 | 不正確（未達扱い不可） | Audio callback は world 起点参照で、`ObservePathSingleSourceTests` により `authority.preparedCrossfade` 必須・`runtimeGraph->` 禁止が契約化されている。 |
| 4 | Crossfade Authority Collapse 未証明 | 一部妥当（監査継続対象） | 懸念自体は妥当だが、`CrossfadeExecutorLocalContractTests` により Commit/Timer での `preparedCrossfade` 参照禁止が拘束済み。未達断定は不可だが drift 監視は必要。 |
| 5 | Legacy Runtime Semantic Removal 未達 | 妥当（事実） | `publishState(current,next,...)` は現存。 |
| 6 | Runtime Meaning Source Collapse 未完 | 一部妥当 | 複数概念は共存。ただし authority inventory と契約テストで制約済み。 |
| 7 | Publication API Zero-Call 未達 | 妥当（事実） | API形状として `publishState` が主経路。 |
| 8 | RuntimeGeneration 単一性未証明 | 一部妥当 | 識別子は複数共存。ただし `runtimeVersion` は diagnostic mirror 指定、分岐利用は未検出。 |
| 9 | RuntimeSemanticSchema 完全一致未証明 | 一部妥当 | schema に `EngineRuntime/RuntimeGraph` 本体は無いが、`RuntimeState` では descriptor/inventory で分類済み。 |
| 10 | Partial Publication Reject 未証明 | 不正確（未達扱い不可） | precheck reject 実装あり、`PartialPublicationRejectTests` も存在。 |
| 11 | PublicationEpoch ↔ RuntimeGeneration 未証明 | 不正確（未達扱い不可） | builder/commit に加え `ISRRuntimePublicationCoordinator` でも epoch・mappedGeneration の整合/単調性チェックあり。 |
| 12 | Snapshot Non-Authority 証明不足 | 一部妥当 | `sealedSnapshot` 関与は事実。ただし deterministic build input とする設計意図と競合する可能性。 |
| 13 | Topology Authority Leakage 未証明 | 一部妥当 | `graph` は残るが、`validateRuntimeGraphAuthorityContract` で整合を fail-closed 検証。 |
| 14 | RuntimeFieldDescriptor 完全被覆未達（17件） | 不正確（事実誤認） | 実際は `RuntimeState=21`, `RuntimeGraph=25`。 |
| 15 | RuntimeSemanticSchema と Plan Schema 乖離 | 一部妥当（計画依存） | semantic 拡張は事実。分類運用自体は inventory で存在。 |
| 16 | SchedulingSemantic と ExecutionSemantic 二重表現 | 妥当（事実） | フィールド重複が存在。 |
| 17 | ActivationEpoch の重複保持 | 妥当（事実） | `GenerationSemantic` と `TimingSemantic` の双方に保持。 |
| 18 | Semantic Hash Coverage 不完全 | 妥当（事実） | hash 対象は schema 全項目 1:1 ではない。 |
| 19 | Semantic Equivalence が schema全体比較でない | 妥当（事実） | `classifySemanticEquivalence` は hash 8項目比較。 |
| 20 | ContractRegistry 実装未確認 | 一部妥当 | 明示名 `ContractRegistry` は未確認、`kRequiredVerifierTable` は存在。 |
| 21 | Fail-Closed Publication が Engine Hook 依存 | 妥当（事実）/結論は要件依存 | generic coordinator は bridge 経由、実体は `runPublicationPrecheckNonRt`。Fail-Closed を満たす限り許容設計とみなせる余地がある。 |
| 22 | Runtime Semantic Lifecycle Verifier 未確認 | 不正確（未達扱い不可） | テストだけでなく runtime の `semanticTransactionState_` 遷移検証ロジックも実装済み。 |

---

## レビュー提案カテゴリ（A/B/C）の検証結果

レビューで提示された再分類は、次のように**概ね採用可能**です。

### Category A（未達の可能性が高い）

- `#5 Legacy Runtime Semantic Removal`
- `#7 Publication API Zero-Call`
- `#16 Execution/Scheduling 重複`
- `#17 activationEpoch 重複`

→ いずれも「現状の設計課題として残る」評価で妥当。

### Category B（設計改善候補 / 将来リスク監査対象）

- `#1 #2 #8 #9 #12 #13 #15 #18 #19 #20 #21`

→ 直ちに不具合断定は難しいが、将来の drift / 保守リスク低減の観点で妥当。

### Category C（現時点で未達扱いすべきでない）

- `#3 #4 #10 #11 #14 #22`

→ `#10/#11/#14/#22` はコード証拠上も未達扱い不可。`#3` は契約テスト水準で達成済み評価が妥当。`#4` は達成済み寄りだが、Authority/Projection/ExecutorLocal の再混線防止のため監査継続対象。

---

## 重点監査5点（レビュー反映後）

実装の現実性と将来リスクの両面から、次の5点を優先監査対象とする判断は妥当です。

1. `publishState(current,next,...)` が残っている理由と移行戦略
2. `ExecutionSemantic` と `SchedulingSemantic` の重複解消方針
3. `activationEpoch` 二重保持の主従整理
4. `RuntimeSemanticHash` / `classifySemanticEquivalence` の対象範囲定義
5. Fail-Closed 検証ロジックの Engine Hook 依存度低減

---

## 残タスク優先度（実装計画に落とす場合）

レビュー提案と再検証結果を統合すると、実務上の優先順位は次の通り。

1. **最優先（未解決の設計課題）**: `#5`, `#7`, `#16`, `#17`
2. **次点監査（技術的負債の先回り）**: `#18`, `#19`
3. **継続監査（退行監視）**: `#4`, `#21` を含む Category B/C 境界項目

---

## 追加で見えたリスク（実害寄り）

1. `ExecutionSemantic` と `SchedulingSemantic` の重複（#16）は、将来 drift の温床になり得る。
2. `activationEpoch` 二重保持（#17）は、意味の主従が崩れると監査が難化。
3. hash/equivalence 対象の設計意図（#18,#19）は、仕様として明文化しないと誤解を生みやすい。

---

## 推奨アクション（最小）

1. `doc/work14/notfinished3.md` を次の3区分で再編する:
   - **事実確定（コード証拠あり）**
   - **未証明（証拠不足）**
   - **要件依存（目標定義が必要）**
2. #14, #10, #11, #22 は現状の文面だと誤解を招くため修正。
3. #16/#17/#18/#19 は「改善候補」として独立チケット化（実害優先）。

---

## 参考ファイル（本検証で参照）

- `src/core/RuntimePublicationCoordinator.h`
- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Commit.cpp`
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`
- `src/audioengine/AudioEngine.Processing.BlockDouble.cpp`
- `src/audioengine/RuntimeBuilder.cpp`
- `src/audioengine/ISRRuntimeSemanticSchema.h`
- `src/audioengine/ISRRuntimePublicationCoordinator.cpp`
- `src/audioengine/RuntimeGraph.h`
- `src/tests/PartialPublicationRejectTests.cpp`
- `src/tests/ObservePathSingleSourceTests.cpp`
- `src/tests/CrossfadeExecutorLocalContractTests.cpp`
- `src/tests/OverlapAuthoritySingularTests.cpp`
- `src/tests/RuntimeWorldAuthorityProjectionTests.cpp`
