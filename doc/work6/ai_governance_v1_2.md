# Practical Stable ISR Bridge Runtime

## AI 実装・詳細設計統治規約 v1.2

## 0. 文書目的

本規約は、ConvoPeq を "Practical Stable ISR Bridge Runtime" へ安全に移行するために、AI に詳細設計・実装・レビュー・修正を実施させる際の統治規約を定義する。

本規約の最上位目的は以下である。

- 長時間運用で semantic drift を発生させない
- runtime authority を単一化する
- publication / observe / retire の整合性を保証する
- Bridge migration 中の silent divergence を防止する
- AI による誤実装・過剰実装・局所最適化を防止する
- fail-closed governance を維持する

本規約は以下を前提とする。

- `doc/work6/base_plan.md` v2.3
- RuntimeSemanticSchema
- Runtime publication governance
- Shadow Compare governance
- Retire governance
- Verifier governance

---

## 1. 最上位原則

### 1.1 Semantic Stability First

AI は以下を成功条件として扱ってはならない。

- compile success
- unit test success
- audio output success
- crash absence

最上位成功条件は以下である。

- semantic equivalence
- publication monotonicity
- observe coherence
- runtime authority singularity
- retire ordering stability

AI は常に「この変更が semantic drift を発生させないか」を最優先で検証しなければならない。

---

### 1.2 Single Authoritative Runtime Principle

AI は RuntimeWorld を唯一の semantic authority として扱わなければならない。

以下は禁止。

- hidden authority
- duplicate semantic ownership
- side-channel branching
- snapshot authority
- executor-local authority
- transition-local authority

RuntimeWorld 外に semantic branching source を追加してはならない。

---

### 1.3 Snapshot Non-Authority Principle

snapshot は projection artifact であり semantic authority ではない。

AI は以下を実装してはならない。

- snapshot writeback
- snapshot branching
- snapshot semantic override
- snapshot-derived authority

許可される用途は以下のみ。

- diagnostics
- telemetry
- UI projection
- observation cache

---

### 1.4 Publication = Synchronization Semantic Layer

publication は単なる状態更新ではない。

publication は以下を持つ synchronization semantic layer として扱わなければならない。

- visibility ordering
- HB edge
- semantic boundary
- epoch transition
- reclaim safety

AI は publication bypass を実装してはならない。

---

### 1.5 Fail-Closed Principle

AI は以下を許容してはならない。

- ambiguity
- partial semantic visibility
- unverifiable authority
- uncertain ownership
- unresolved ordering

不明点がある場合は以下を優先する。

- fail
- reject
- throttle
- fallback

---

## 2. AI に許可される変更

### 2.1 許可される変更

- RuntimeWorld authority singularization
- publication path singularization
- observe path collapse
- retire governance 強化
- shadow compare 強化
- verifier 強化
- diagnostics channel 分離
- semantic contract enforcement
- RuntimeSemanticSchema 整合化
- ownership safety 改善
- memory ordering 明示化
- monotonicity enforcement
- topology authority collapse

---

### 2.2 AI に禁止される変更

以下は AI が独断で実施してはならない。

- 新 authority source 追加
- hidden mutable state 追加
- snapshot authority 化
- executor-local semantic branching
- publication bypass
- atomic ordering downgrade
- retire shortcut
- verifier disable
- fail-open fallback
- silent recovery
- silent queue drop
- semantic state cache duplication
- ownership semantics 変更
- lifetime contract 暗黙変更
- RuntimeSemanticSchema 外 semantic 追加
- topology authority 外部化
- generation semantic 多重化

---

## 3. Runtime Semantic Schema 規約

### 3.1 RuntimeSemanticSchema 外 authority 禁止

semantic authority を持てるのは RuntimeSemanticSchema 内のみ。

以下は禁止。

- implicit semantic fields
- undocumented semantic ownership
- semantic side effects
- external semantic coupling

---

### 3.2 Schema 拡張ルール

AI が RuntimeSemanticSchema を拡張する場合、以下を必須とする。

- semantic role 定義
- authority owner 定義
- publication visibility 定義
- retire semantic 定義
- verifier 定義
- telemetry 定義
- rollback semantic 定義

未定義のまま field を追加してはならない。

---

### 3.3 ExecutionSemantic 規約

ExecutionSemantic に含む authority は最低限以下。

- processing graph order
- graph execution order
- activation semantic
- overlap execution semantic
- latency compensation semantic
- scheduling semantic

ExecutionSemantic は RuntimeWorld 以外へ authority を持ってはならない。

---

### 3.4 OverlapSemantic 規約

overlap authority は RuntimeWorld のみ。

以下は禁止。

- executor overlap authority
- crossfade branching authority
- overlap scheduling override
- overlap state authority leakage

executor-local overlap state は diagnostics / projection のみ許可。

---

## 4. Projection Formal Definition（新規）

### 4.1 Projection の定義

projection は authoritative runtime semantic から導出される非権威データであり、以下を満たすこと。

- derived-only
- non-authoritative
- write-back prohibited
- branch-source prohibited

### 4.2 Projection 許可用途

- UI
- diagnostics
- telemetry
- observation cache

### 4.3 Projection 禁止事項

- projection を用いた semantic branching
- projection から RuntimeWorld への write-back
- projection 側での authority override

---

## 5. Publication Governance 規約

### 5.1 Single Publication Authority

publication authority は単一。

AI は以下を追加してはならない。

- duplicate publication path
- partial publication path
- side-channel publish
- shadow publish

---

### 5.2 Visibility Monotonicity

以下を常に保証すること。

`ObservedPublicationSequence >= LastObservedPublicationSequence`

publish 順と observe 順が逆転してはならない。

---

### 5.3 PublicationEpoch Contract

PublicationEpoch は RuntimeGeneration に monotonic mapping しなければならない。

禁止。

- independent publication timeline
- non-monotonic epoch mapping
- generation regression

---

### 5.4 Partial Publication 禁止

partial publication を禁止する。

partial publication とは、RuntimeSemanticSchema の subset のみが visibility される状態を指す。

publication 時は schema semantic 全体が coherent visibility を持たなければならない。

---

### 5.5 Publication Bypass 禁止

以下は禁止。

- direct runtime mutation visibility
- publication-free observe update
- local observe injection
- external semantic patch

observe path は publication を経由しなければならない。

---

## 6. Memory Ordering Governance（新規）

### 6.1 Ordering Downgrade 禁止

以下を禁止する。

- release/acquire を relaxed へ downgrade
- publication ordering contract と不整合な ordering 変更
- retire/reclaim ordering を未定義のまま変更

### 6.2 Publication Ordering Contract

publication semantic を扱う操作は少なくとも以下を満たすこと。

- publish path: release 以上
- observe/consume path: acquire 以上
- HB edge を崩す変更は禁止

### 6.3 Retire/Reclaim Ordering Contract

retire queue / reclaim queue の可視性境界をまたぐ操作は ordering 根拠を実装コメントまたは設計注記で明示すること。

### 6.4 ABA 対応 Ordering Rule

ABA リスクのある経路では、世代番号・sequence・tag などの補助機構を持たない状態で ordering 緩和を行ってはならない。

---

## 7. Observe Governance 規約

### 7.1 Observe Path Singularization

AI は observe path を一本化しなければならない。

禁止。

- dual observe path
- mixed semantic observe
- stale snapshot observe
- split visibility source

---

### 7.2 Observe Projection Rule

projection は branch authority を持たない。

projection 用途は 4章の定義に従う。

---

## 8. Topology Governance 規約

### 8.1 RuntimeTopology Authority

topology authority は RuntimeWorld のみ。

禁止。

- slot-based topology branching
- processing-order authority leakage
- transition topology override
- executor topology override

---

### 8.2 Topology Mutation Rule

Topology mutation は以下を経由しなければならない。

- publication boundary
- generation transition

runtime execution 中の topology authority mutation を禁止する。

---

## 9. SIMD / Cache Coherency Governance（新規）

### 9.1 Alignment Invariant

SIMD で扱うバッファ・状態の alignment invariant を崩す変更を禁止する。

### 9.2 False Sharing 防止

以下を遵守する。

- 高頻度更新 atomic/state の隣接配置を避ける
- cache-line ownership を明示する
- false sharing を増やすレイアウト変更は禁止

### 9.3 RT Cache Thrash 防止

RT path で cache miss を有意に増加させる設計変更は、検証（ベンチ/計測）なしで導入してはならない。

---

## 10. Retire Governance 規約

### 10.1 Silent Drop 禁止

以下は禁止。

- silent retire discard
- implicit reclaim
- silent queue truncation
- hidden retire fallback

---

### 10.2 Retire Pressure Policy

retire pressure 発生時は以下のみ許可。

| 状態 | 許可動作 |
| --- | --- |
| mild | rebuild coalescing |
| medium | publication throttle |
| severe | rebuild reject |
| critical | emergency drain |

---

### 10.3 Retire Ordering Stability

retire ordering drift を禁止する。

AI は以下を実装してはならない。

- reclaim reorder
- retire bypass
- out-of-order retire

---

### 10.4 Ownership Evaluation Rule

ownership model を変更する場合、以下を必須検証とする。

- atomic traffic
- cache contention
- epoch progression
- ABA risk
- reclaim latency
- RT worst-case latency
- retire backlog growth

未検証変更は禁止。

---

## 11. Crossfade / Transition 規約

### 11.1 Crossfade Diagnostic-Only Rule

crossfade state は semantic authority を持ってはならない。

許可。

- diagnostics
- telemetry
- visualization

禁止。

- branching authority
- publication authority
- retire authority
- topology authority

---

### 11.2 Transition Semantic Leakage 禁止

TransitionState は semantic authority を持ってはならない。

transition local state が以下を保持してはならない。

- runtime semantic
- topology semantic
- generation semantic

---

## 12. Shadow Compare 規約

### 12.1 Shadow Compare 必須

Bridge migration 中は新 runtime と legacy runtime の semantic compare を必須化する。

---

### 12.2 Compare 項目

最低比較対象。

- generation
- topology
- overlap semantic
- publication sequence
- retire ordering
- runtime semantic hash
- execution ordering
- timing drift

---

### 12.3 Mismatch Taxonomy

mismatch は severity 分類する。

| mismatch | severity |
| --- | --- |
| topology mismatch | Critical |
| publication regression | Critical |
| semantic hash mismatch | Critical |
| overlap mismatch | High |
| retire ordering mismatch | High |
| timing drift | Medium |

---

### 12.4 Hash Collision Rule

- hash mismatch は semantic mismatch とみなす。
- hash match は semantic equality guarantee ではない。
- hash のみで semantic equality を証明してはならない。

---

### 12.5 Hash Coverage Contract（新規）

RuntimeSemanticHash は以下の必須カバレッジを持つこと。

- generation semantic
- topology semantic
- execution semantic
- publication semantic
- overlap semantic
- retire semantic

必須項目の欠落、または未承認の除外は禁止。

---

### 12.6 Compare Cadence Governance（新規）

- minimum compare cadence を設定する（例: 1 sec 以下間隔）。
- churn/burst 検出時は compare cadence を自動増加する。
- drift シグナル検出時は escalation compare を実施する。

---

## 13. Rollback Governance 規約

### 13.1 Rollback Hysteresis

rollback oscillation を禁止する。

必須。

- cooldown period
- escalation rule
- repeated failure threshold

暫定デフォルト値（環境で調整可）。

- cooldown: 30 sec 以上
- repeated threshold: 5 failures / 10 min
- escalation: Soft -> Medium -> Hard -> Emergency

---

### 13.2 Rollback Levels

| レベル | 動作 |
| --- | --- |
| Soft | publication freeze |
| Medium | shadow-only mode |
| Hard | legacy fallback |
| Emergency | runtime rebuild suspend |

AI は fail-open recovery を実装してはならない。

### 13.3 Rollback Entry / Exit Rule

- Entry は trigger 条件と severity で決定する。
- Exit は cooldown 経過 + mismatch 収束 + verifier pass を必須とする。
- Exit 条件未達での復帰は禁止する。

---

## 14. Telemetry Governance 規約

### 14.1 Telemetry Contract

telemetry は以下を持たなければならない。

- monotonic clock
- stable sampling
- coherent retention
- overwrite policy（silent overwrite 禁止）

---

### 14.2 Percentile Rule

threshold は percentile ベースを優先する。

推奨。

- P95
- P99

安全系ガードとして hard threshold を併用してよい。

---

### 14.3 Telemetry Non-Authority Rule

telemetry は semantic authority を持たない。

telemetry-driven semantic branching を禁止する。

---

### 14.4 Telemetry Overhead Governance（新規）

- telemetry は RT path で動的メモリ確保を行ってはならない。
- telemetry は RT path で lock/blocking を発生させてはならない。
- sampling overhead budget を定義し、超過時は degrade/sampling reduction を行う。

---

## 15. Verifier Governance 規約

### 15.1 Fail-Closed Verifier

verifier failure は build/release failure として扱う。

原則として warning-only verifier を恒久運用してはならない。

---

### 15.2 Incubation Verifier Rule

新規 verifier は以下を満たす場合のみ incubation（警告）を許可する。

- owner 明記
- expiry 明記（最大 30 日）
- fail-closed 移行日を明記
- compensating controls を明記

期限超過は fail-closed とする。

---

### 15.3 Verifier Suppression Governance（新規）

- temporary disable / allow-failure / soft-pass を無承認で実施してはならない。
- suppression には owner, approver, rationale, expiry, compensating verifier を必須とする。
- suppression 期限超過は build fail とする。

---

### 15.4 Verifier Dependency Rule

verifier dependency graph を維持する。

上位 verifier は prerequisite verifier pass を前提とする。

dependency 循環は禁止。

---

### 15.5 Verifier Tier Governance

verifier は tier 分離する。

| Tier | 用途 |
| --- | --- |
| PR | static / cheap |
| nightly | dynamic |
| release | full soak |

---

### 15.6 LegacyTemporary Rule

LegacyTemporary は以下を必須とする。

- expiration
- owner
- approver
- risk classification
- removal phase
- verifier linkage
- compensating controls
- removal verifier

恒久化禁止。

### 15.7 Governance Consistency Rule（新規）

governance 変更は局所最適で追加してはならない。以下を必須とする。

- governance dependency DAG の循環禁止
- taxonomy 変更時 mandatory review
- verifier/compare/rollback 契約整合チェック
- ContractRegistry / VerifierRegistry 同時更新

不整合は fail-closed とする。

### 15.8 Semantic Density Budget Rule（新規）

semantic interaction density の爆発を禁止する。

- compare axis count bounded
- semantic dependency depth bounded
- verifier fanout bounded
- publication side-effects bounded

密度予算超過変更は decomposition plan なしで受け入れない。

---

## 16. CI / Soak Governance 規約

### 16.1 Soak First Principle

長時間 semantic stability を最優先する。

短時間 pass を成功条件としてはならない。

---

### 16.2 Mandatory Soak Cases

最低限。

- rebuild storm
- automation storm
- sample-rate churn
- overlap churn
- bypass churn
- UI attach/detach churn

---

### 16.3 Long-Run Drift Detection

AI は以下を長時間運用前提で検証しなければならない。

- semantic drift
- publication drift
- retire drift
- overlap drift
- topology drift

---

### 16.4 Soak Abort / Retry Rule

- Critical mismatch 検出時は soak を中断し rollback 判定へ遷移。
- Retry は root-cause 修正後に 1 回のみ許可。
- 2 回連続失敗で release block とする。

### 16.5 Adaptive Operational Envelope Rule（新規）

固定 threshold 前提を禁止する。AI は以下を考慮し envelope を更新すること。

- allocator state
- cache topology
- device load
- telemetry density
- rebuild frequency

false alarm storm / blind spot が継続する設定を放置してはならない。

### 16.6 Global Equilibrium / Feedback Rule（新規）

個別 subsystem 安定を全体安定と誤認してはならない。

- cross-subsystem oscillation を監視
- feedback amplification loop を検知
- loop gain 抑制（sampling/compare/throttle 制御）を適用

### 16.7 Probabilistic Tail Rule（新規）

tail stability を deterministic 前提で設計してはならない。

- OS scheduling
- NUMA/cache eviction
- IRQ/device jitter

を含む確率的 tail 指標で受入判定を行う。

---

## 17. AI 変更スコープ統治（新規）

### 17.1 No Unrelated Semantic Modification Rule

AI は対象課題に無関係な semantic 変更を行ってはならない。

### 17.2 Scope Expansion Approval Rule

影響範囲が初期スコープを超える場合、変更理由・必要性・代替案を明示した上で承認なしに拡張実装してはならない。

### 17.3 One-Objective Change Rule

1変更は1目的を原則とし、複数目的は分割する。

---

## 18. アーキテクチャ改変禁止規約（新規）

以下は明示承認なしに実施してはならない。

- architecture rewrite
- framework replacement
- ownership paradigm rewrite
- concurrency model rewrite
- broad abstraction migration（例: ECS全面化、actor全面化）

---

## 19. 実装レビュー規約

### 19.1 AI 実装レビュー必須項目

AI は各変更について以下を必ずレビューする。

- authority singularity
- semantic duplication
- publication monotonicity
- visibility coherence
- topology authority
- overlap authority
- retire ordering
- rollback safety
- verifier coverage
- telemetry coherence
- shadow compare compatibility

---

### 19.2 レビュー時の禁止事項

以下は禁止。

- 「動くので問題ない」
- 「compile pass したのでOK」
- 「クラッシュしないので安全」
- 「unit test pass したので完了」

semantic governance を伴わない安全判定は禁止。

### 19.3 Risk-weighted Compare Rule（新規）

compare を diff engine 化してはならない。

- 全項目常時比較を禁止
- risk tier に応じた比較頻度を採用
- compare cost budget 超過設定を禁止

### 19.4 Perceptual Equivalence Rule（新規）

DSP 実運用では perceptual equivalence class を許容する。

許容候補:

- coefficient epsilon drift
- overlap phase 微差
- SIMD reorder 差
- denormal suppression 差

可聴差・因果差に到達する場合は mismatch とする。

### 19.5 Semantic Priority Governance Rule（新規）

publication を同一優先度で扱ってはならない。

- high severity semantic（bypass/topology/timing）優先
- low severity payload（telemetry/analyzer）は干渉最小化

### 19.6 Evidence Hierarchy Rule（新規）

`verifier pass = operationally safe` の推論を禁止する。

operational trust は verifier に加えて soak / telemetry / disturbance replay を含む。

---

## 20. AI 出力規約

AI は各変更に対し、以下を明示しなければならない。

- authority owner
- visibility owner
- publication semantic impact
- retire semantic impact
- rollback semantic impact
- verifier impact
- telemetry impact
- shadow compare impact
- phase impact（Phase 0-9 のどこに作用するか）
- exit criteria impact（DoD への影響）

### 20.1 必須出力テンプレート

```text
[Change Summary]
- Scope:
- Files:

[Semantic Authority]
- Authority owner:
- Visibility owner:
- New/changed semantic field:

[Governance Impact]
- Publication impact:
- Observe impact:
- Retire impact:
- Rollback impact:

[Verification]
- Required verifiers:
- Dependency prerequisites:
- Tier coverage (PR/Nightly/Release):

[Telemetry / Shadow Compare]
- Telemetry metrics affected:
- Shadow compare dimensions affected:

[Risk & Controls]
- Drift risk:
- Failure mode:
- Mitigation:
```

---

## 21. 実装安全制約（RT）

AI は本リポジトリの RT 制約を遵守しなければならない。

特に audio thread では以下を禁止する。

- blocking / lock / wait
- 動的メモリ確保・解放
- I/O
- fail-open リカバリ

具体制約はプロジェクト既存規約（`copilot-instructions.md` 等）を優先適用する。

### 21.1 RT Locality Preservation Rule（新規）

`semantic purity` を理由に RT locality を破壊してはならない。

禁止:

- scratch 再利用の全面禁止
- SIMD/cache reuse の破壊
- allocator churn を増幅する purity-only rewrite

許可:

- authority 非侵害の executor-local mutable reuse
- bounded operational economics を満たす最適化

---

## 22. 運用台帳整合規約

### 22.1 ContractRegistry / VerifierRegistry

- ContractRegistry は契約の唯一台帳。
- VerifierRegistry は verifier 依存と tier 配線の唯一台帳。
- 変更時は両台帳を同時更新する。
- 台帳不整合は CI fail とする。

### 22.2 Hash Authority Prohibition（新規）

`same hash => same runtime` の推論を禁止する。

- hash は diagnostic fingerprint
- authority identity 判定に hash を使わない
- rollback trigger を hash 単独で決定しない

### 22.3 Projection Semantic Austerity Rule（新規）

projection を secondary semantic cache に進化させてはならない。

禁止:

- cachedResolvedRouting 等の authority substitute 利用
- projection を semantic branching source とする設計

### 22.4 Migration Decomposition Deadline Rule（新規）

dual publication / compatibility shim / migration telemetry の恒久残置を禁止する。

- removal milestone
- expiry
- completion evidence

を必須化し、期限超過は escalation + merge gate 制御対象とする。

### 22.5 Complexity Budget Governance（新規）

AI が runtime complexity asymmetry を蓄積させないよう次を予算管理する。

- semantic layer count growth
- verifier rule growth
- telemetry channel growth
- rollback branch growth

予算超過変更は decomposition plan なしで受け入れない。

### 22.6 Reclaim Convergence SLA Rule（新規）

`そのうち reclaim される` という楽観を禁止する。

- reclaim debt bounded
- retire lag bounded
- retire queue residency bounded
- reclaim burst bounded

継続違反は degrade -> fail-closed へ遷移する。

### 22.7 Semantic Migration Compatibility Rule（新規）

topology migration だけでなく semantic migration 互換性を必須とする。

- retire semantic
- publication ordering
- compare tolerance
- timing source

変更時は compatibility evidence を必須化する。

### 22.8 Retire Lifecycle State Rule（新規）

retire を単純 queue と見なしてはならない。

`visible -> compare -> telemetry -> replay(optional) -> reclaim-eligible -> reclaimed`

の多段 lifecycle で管理する。

### 22.9 Projection Freshness Rule（新規）

projection の鮮度契約を必須化する。

- freshness metadata
- max staleness budget
- stale projection の authority誤用禁止

### 22.10 Complexity Decay Governance（新規）

complexity の単調増加を禁止する。

- verifier/telemetry/shim の定期削減
- expired temporary の強制回収
- net complexity delta の監査

### 22.11 Runtime-native Layout Priority Rule（新規）

RuntimeWorld を serialization 都合で変形してはならない。

- hot-path locality 優先
- hot/cold 分離維持
- indirection 増加は bounded

### 22.12 Crossfade Authority Boundary Rule（新規）

crossfade が timing authority を支配してはならない。

- effective activation timing authority は RuntimeWorld.timing のみ
- crossfade は executor-local smoothing に限定

---

## 23. 定量ゲート（初期値）

以下は初期値であり、運用測定で調整可能。

- Critical mismatch: 1 件で release block
- High mismatch: 3 件 / 10 min 超で release block
- publication monotonicity violation: 1 件で fail-closed
- retire starvation violation: 1 件で fail-closed
- rollback re-trigger: 5 回 / 10 min 超で 1 段階 escalation

追加ゲート（上位契約反映）:

- reclaim convergence SLA 継続違反: release block
- compare cost budget 継続超過: release block
- migration decomposition deadline 超過: policy escalation
- complexity budget 超過継続: release block
- semantic density budget 継続超過: release block
- global oscillation bound 超過継続: release block
- disturbance 後の収束未達: release block

---

## 24. 最終原則

Practical Stable ISR Bridge Runtime における最重要目的は「理想 runtime を構築すること」ではない。

最重要目的は「semantic drift を発生させず、長時間運用で破綻しない runtime へ安全移行すること」である。

AI は以下を最優先で警戒しなければならない。

- hidden semantic divergence
- long-run drift
- partial visibility
- governance bypass
- silent failure

加えて次を常時警戒する。

- governance drift
- operational divergence
- complexity explosion
- equilibrium collapse

---

## 付録A: v1.1 から v1.2 への差分要約

- Projection の formal definition（derived-only / non-authoritative / no write-back / no branching）を追加
- Memory Ordering Governance（release/acquire downgrade 禁止、retire/reclaim ordering、ABA対応）を追加
- SIMD / Cache Coherency Governance（alignment invariant、false sharing防止）を追加
- Hash Coverage Contract（必須カバレッジ）を追加
- Shadow Compare Cadence Governance（minimum cadence、burst/escalation compare）を追加
- Verifier Suppression Governance（temporary disable/allow-failure の承認統治）を追加
- LegacyTemporary を強化（risk classification / removal verifier 追加）
- AI変更スコープ統治（無関係変更禁止、スコープ拡張承認、1変更1目的）を追加
- アーキテクチャ改変禁止規約（承認なし全面改変禁止）を追加
- Telemetry Overhead Governance（RT非侵入・overhead budget）を追加
