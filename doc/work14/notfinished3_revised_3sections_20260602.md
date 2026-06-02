# notfinished3 改訂版（3区分自動整形）

対象原文: `doc/work14/notfinished3.md`
整形日: 2026-06-02

本改訂版は、原文の22項目を **事実確定 / 未証明 / 要件依存** の3区分へ再配置したものです。
（注）「要件依存」には、現行コードと不一致で**文面修正が必要**な項目も含めています。

---

## 事実確定

以下は、現行コード上で主張の核となる事実が確認できる項目です。

1. **#1 AudioEngine が Publication Authority の一部を保持**
   - Bridge が `engine_->runPublicationPrecheckNonRt` / `onRuntimePublishedNonRt` / `onRuntimeRetiredNonRt` / `enqueueDeferredDeleteNonRt` を呼ぶ構造は実在。

2. **#5 Legacy Runtime Semantic Removal 未達（`publishState(current,next,...)` 残存）**
   - `publishState` ベースの公開APIが残存。

3. **#7 Publication API Zero-Call 未達**
   - `publishState` 呼び出しが現行主経路。

4. **#16 SchedulingSemantic と ExecutionSemantic の二重表現**
   - `transitionActive / crossfadeStartDelayBlocks / crossfadeDryHoldSamples` が重複。

5. **#17 ActivationEpoch の重複保持**
   - `GenerationSemantic.activationEpoch` と `TimingSemantic.activationEpoch` が併存。

6. **#18 Semantic Hash Coverage 不完全**
   - `RuntimeSemanticHash` は schema 全項目 1:1 カバレッジではない。

7. **#19 Semantic Equivalence が Schema 全体を見ていない**
   - `classifySemanticEquivalence` は hash 8系統比較。

8. **#21 Fail-Closed Publication が Engine Hook 依存**
   - generic coordinator 層の validate は bridge 経由で engine 実装へ委譲される。

---

## 未証明

以下は、問題提起として妥当だが、現時点の証拠だけでは断定に至らない項目です。

1. **#2 RuntimeWorld Self-contained 化が未証明**
   - `AudioEngine* engine_` 保持は事実だが、これのみで RuntimeWorld 非 self-contained を断定するには不足。

2. **#4 Crossfade Authority Collapse が未証明**
   - 懸念は妥当。
   - 一方で `preparedCrossfade` 使用範囲に対する契約テストが存在し、完全無統制ではない。

3. **#6 Runtime Meaning Source Collapse が未完**
   - 複数概念の共存は事実。
   - ただし authority inventory / contract test による制約実装も確認済み。

4. **#12 Snapshot Non-Authority の完全証明不足**
   - `sealedSnapshot` の build 関与は事実。
   - ただし現設計意図（deterministic build input）との整合整理が必要。

5. **#13 Topology Authority Leakage 未証明**
   - `RuntimeGraph graph` は残る。
   - ただし `validateRuntimeGraphAuthorityContract` 等で整合を fail-closed 検証しているため、即断は不可。

6. **#20 ContractRegistry 実装未確認**
   - `ContractRegistry` という明示名は未確認。
   - ただし `kRequiredVerifierTable` は存在し、運用台帳相当の実体はある。

---

## 要件依存

以下は「最終到達条件の定義次第」で評価が変わる項目、または原文の文面に修正が必要な項目です。

1. **#3 Observe Source 単一化未証明**（文面修正推奨）
   - 現行は world 起点参照経路と契約テストが実装済み。
   - 「未証明」断定は強すぎるため、要件定義に合わせて再記述が必要。

2. **#8 RuntimeGeneration 単一性未証明**（要件依存）
   - `generation/runtimeVersion/transitionId/generationSemantic` は併存。
   - ただし `runtimeVersion` は diagnostic mirror 指定で、分岐利用は未検出。

3. **#9 RuntimeSemanticSchema 完全一致未証明**（要件依存）
   - schema 本体に `EngineRuntime/RuntimeGraph` の直接表現はない。
   - 一方で `RuntimeState` 側 inventory で分類済み。
   - 「一致」の定義（schema直載せ必須か否か）を要件化すべき。

4. **#10 Partial Publication Reject 未証明**（文面修正必須）
   - 現行は reject 実装・rejectテストとも存在。
   - 本項は「未証明」ではなく「既実装/既検証」に修正が必要。

5. **#11 PublicationEpoch ↔ RuntimeGeneration Contract 未証明**（文面修正推奨）
   - builder/commit で整合・単調性チェック実装あり。
   - さらなる厳格契約（例: 明示写像仕様）を要求するなら要件として定義すべき。

6. **#14 RuntimeFieldDescriptor 完全被覆未達（17件）**（文面修正必須）
   - 件数認識が不一致（実際は `RuntimeState=21`, `RuntimeGraph=25`）。

7. **#15 RuntimeSemanticSchema と Plan Schema の乖離**（要件依存）
   - semantic 拡張は事実。
   - 乖離を「不具合」とみなすには計画側の拘束条件を明文化する必要あり。

8. **#22 Runtime Semantic Lifecycle Verifier が未確認**（文面修正推奨）
   - 単体テストに加え、runtime 側にも `semanticTransactionState_` 遷移検証がある。
   - 追加保証が必要なら、どの状態機械を本番強制対象にするか要件定義が必要。

---

## 付記（この改訂版の使い方）

- すぐ実装に落とすなら、優先度は次の順を推奨:
  1. 事実確定のうち実害が高い重複源（#16, #17, #18, #19）
  2. 未証明群のうち契約テストを追加しやすい項目（#4, #13）
  3. 要件依存群の定義固め（#8, #9, #15）

- 原文 `notfinished3.md` を更新する場合は、最低でも #10/#14 の文面は事実整合のため先に修正すること。
