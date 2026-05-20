# REV3.2運用優先注記 監査レポート（2026-05-20）

## 監査方法（機械抽出）

- 対象: `doc/work/*.md`
- 注記有無判定: `REV3.2` 文字列を含むか
- 齟齬リスク判定語: `self-proving|autonomous emit|runtime-resident|RuntimeAuthorityCoordinator|RuntimeWorldRetireManager|PublicationWorld|ExecutionWorld|stale handle|federation|HBRuntimeCore|HBTraceRuntime|HBVerifierRuntime`
- 欠落候補条件: **`REV3.2` 未記載 かつ リスク語を1つ以上含む**

---

## 欠落候補（優先度順）

| 優先度 | ファイル | リスク語数 | 主な論点 | 提案 |
| --- | --- | ---: | --- | --- |
| 高 | `ISR_Minimal_Phase0_Recommended.md` | 9 | 10層/2-world/authority/stale-handle | REV3.2注記を冒頭「目的」直下に1ブロック追加 |
| 高 | `ISR_Completeness_Risk_Backlog.md` | 5 | authority/world/stale-handle運用 | REV3.2運用優先の評価基準（CI/Debug/Release）を追記 |
| 高 | `ISR_World_Bridge_Runtime.md` | 4 | 3-world参照と2-world運用優先 | REV3.2で「helper扱い/authority rootではない」を明記 |
| 中 | `ISR_Retire_Authority_Graph.md` | 2 | capability-first運用の優先順位 | REV3.2注記を追加（coordinatorは互換shimとして解釈） |
| 中 | `ISR_Runtime_State_Matrix.md` | 2 | authority列の解釈一貫性 | 表冒頭注記にREV3.2優先（delegate解釈）を追記 |
| 中 | `ISR_DSPHandle_Runtime.md` | 1 | stale-handle build別運用 | REV3.2注記を追加しCI/Debug/Releaseを明示 |
| 中 | `ISR_DSPHandle_Allocator_Policy.md` | 1 | stale-handle/reuseの運用文脈 | REV3.2注記を追加（policyはRelease fail-safe優先） |
| 低 | `ISR_RT_Execution_Frame.md` | 1 | RTルールとevidence/CI責務 | REV3.2注記で `runtime exposes evidence / CI validates evidence` を追記 |
| 低 | `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md` | 1 | epoch戦略と運用優先 | REV3.2注記で「RetireRuntime内部責務統合優先」を追記 |

---

## 既にREV3.2が確認できるファイル（参考）

- `plan5.md`
- `ISR_10Layer_Implementation_Specification.md`
- `ISR_Execution_Authority_Convergence.md`
- `ISR_Formal_Guarantee_Package.md`
- `ISR_Runtime_Object_Model_Integration.md`
- `ISR_Runtime_Proof_and_Recovery_Integration.md`
- `ISR_DSPCore_Decomposition_Analysis.md`

---

## 推奨の共通追記テンプレート（最小差分）

```md
### REV3.2運用優先注記

- 本書の理論/参照設計記述は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- 解釈衝突時は few-authority（7 subsystem）/ 2-world（Publication/Execution）/ capability-first（coordinatorは互換shim）を優先する。
```

必要に応じてファイル固有の1行を追加:

- stale handle あり: `CI=Abort / Debug=Assert / Release=Quarantine+Silence` を明記
- world系: `WorldBridgeRuntime は helper（authority root ではない）` を明記
- retire系: `RuntimeWorldRetireManager は capability-first 制約下の実装委譲` を明記
