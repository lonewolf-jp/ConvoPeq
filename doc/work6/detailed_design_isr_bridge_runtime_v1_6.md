# Practical Stable ISR Bridge Runtime 詳細設計書 v1.6

## 0. 目的と適用範囲

本書は以下を設計入力として、AI実装可能な拘束条件付き詳細設計を定義する。

- 基本設計: `doc/work6/base_plan.md` (v2.3)
- 統治規約: `doc/work6/ai_governance_v1_2.md`
- 既存詳細設計系統: v1.5までの内容を統合済み

本書の目的は、数か月〜数年スケールの churn / rebuild / automation / migration 下で、
**semantic coherence**、**bounded operational equilibrium**、**self-stabilizing behavior** を同時維持すること。

---

## 1. 最上位設計原則

1. Semantic Stability First
2. Single Authoritative Runtime (RuntimeWorld)
3. Projection Non-Authority
4. Publication as Synchronization Semantic Layer
5. Fail-Closed Governance
6. Performance-Preserving Rewrite
7. Scalability-Preserving Publication
8. Bounded Operational Economics
9. Recoverable Long-Run Behavior
10. Anti-Erosion Governance

補足: compile/test pass は受入条件ではない。契約・verifier・soak・運用境界達成が受入条件。

---

## 2. Semantic Category Taxonomy v1.6

| Category | 内容 | 代表対象 |
| --- | --- | --- |
| AcousticSemantic | DSP出力意味 | gain/filter/IR効果 |
| TopologySemantic | 構造意味 | graph/node/edge |
| PublicationSemantic | 可視化同期意味 | sequence/epoch/HB |
| LifetimeSemantic | 寿命意味 | retire/reclaim/order |
| SchedulingSemantic | 時間割付意味 | activation/defer/timing |
| OperationalSemantic | コスト境界意味 | CPU/memory/latency budget |

### 2.1 Semantic Dependency DAG Contract

semantic層の循環依存を禁止する。

| Semantic | allowed dependencies |
| --- | --- |
| TopologySemantic | none |
| RoutingSemantic | TopologySemantic |
| TimingSemantic | TopologySemantic, RoutingSemantic |
| PublicationSemantic | TimingSemantic |
| RetireSemantic | PublicationSemantic |
| AutomationSemantic | TimingSemantic, RoutingSemantic |
| CoefficientSemantic | TopologySemantic, AutomationSemantic |

禁止:

- semantic cyclic governance
- cross-layer back-edge without approval

### 2.2 semantic causality contract（新規）

`publication ordering` と `audible/causal semantic ordering` を明確に分離する。

- publish later != semantically newer
- delayed activation / overlap window / scheduled transition を因果順評価に含める

### 2.3 governance consistency contract（新規）

governance 自体の drift を防止するため、以下を強制する。

| contract | rule |
| --- | --- |
| governance dependency DAG | 循環禁止 |
| semantic ownership | semanticごとに authority owner 固定 |
| taxonomy change review | category追加/変更時 mandatory review |
| contract compatibility | rollback/compare/verifier の整合義務 |
| governance gate | rule追加時に矛盾検査を CI で必須化 |

矛盾検査で fail した governance 変更は fail-closed とする。

### 2.4 semantic density budget（新規）

semantic correctness を維持していても interaction density 爆発を許可しない。

| 指標 | 制限 |
| --- | --- |
| compare axis count | bounded |
| semantic dependency depth | bounded |
| verifier fanout | bounded |
| publication side-effects | bounded |

密度予算超過時は decomposition または削減計画なしに merge してはならない。

---

## 3. Runtime Semantic Schema v1.6

```cpp
struct RuntimeSemanticSchema {
  GenerationSemantic generation;
  TopologySemantic topology;
  RoutingSemantic routing;
  ExecutionSemantic execution;
  PublicationSemantic publication;
  OverlapSemantic overlap;
  RetireSemantic retire;
  TimingSemantic timing;
  LatencySemantic latency;
  SchedulingSemantic scheduling;
  ResourceSemantic resource;
  AffinitySemantic affinity;
  AutomationSemantic automation;
  CoefficientSemantic coefficient;
};
```

### 3.1 field分類強制

- Authoritative
- Derived
- Diagnostic
- ExecutorLocal
- LegacyTemporary

未分類 field は CI fail。

### 3.2 authority と representation の分離

```text
authoritative semantic may be physically segmented
```

`single semantic authority` は `single storage object` を意味しない。

### 3.3 schema evolution governance

- `RuntimeSemanticSchemaVersion` を定義
- schema 変更時は version bump 必須
- verifier compatibility update 必須
- shadow baseline refresh + soak replay 必須

### 3.4 migration compatibility rule

- backward comparison contract
- replay compatibility policy
- rollback compatibility policy

### 3.4.1 semantic migration compatibility contract（新規）

topology migration だけでなく semantic migration の互換性を必須化する。

対象例:

- retire semantic 変更
- publication ordering 変更
- compare tolerance 変更
- timing source 変更

Topology identical でも semantic drift が起こる変更は互換性検証なしに許可しない。

### 3.5 semantic vs resource separation

RuntimeWorld authority に含めるのは semantic のみ。

resource layer として分離する。

- thread affinity
- SIMD kernel cache
- allocator state
- FFT plan
- device handle

### 3.6 mutable state taxonomy（新規）

`No mutable authority` と `No mutable state` を分離する。

許可される mutable state:

- scratch
- reusable cache
- bounded telemetry buffers

禁止される mutable authority:

- topology/routing/publication/retire semantic flags

---

## 4. Authority Regression Governance（新規）

### 4.1 authority escape prohibition

RuntimeWorld 外の semantic authority 追加を禁止。

### 4.2 diagnostic influence prohibition

diagnostic state を semantic branch source に使用してはならない。

### 4.3 projection authority escalation prohibition

projection -> authority 昇格を禁止。

### 4.4 regression detection

以下の新規状態が追加された場合は必ず authority regression audit を実施する。

- TelemetryRuntimeView
- AutomationRuntimeState
- OptimizationRuntimeCache
- BackgroundAnalyzerState

---

## 5. Ownership / Allocator / Immutable Segmentation

### 5.1 Ownership Strategy Evaluation

必須測定:

- atomic traffic
- cache contention
- epoch progression
- ABA risk
- reclaim latency
- RT worst-case latency
- retire backlog growth

### 5.2 ownership economics governance

`smart pointer = safe` の誤認を禁止。

- bounded reclamation semantics 必須
- destruction predictability 証跡必須

### 5.3 allocator boundary

- allocator ownership/lifetime/affinity を定義
- cross-thread contention 抑制

### 5.4 immutable boundary contract

`immutable = rebuild everything` を禁止。

| Layer | rebuild policy |
| --- | --- |
| topology | immutable snapshot |
| coefficients | versioned delta apply |
| FFT cache | reusable pooled |
| telemetry metadata | externalized |
| scratch/workspace | executor-local reusable |

### 5.5 historical retention governance（新規）

immutable state の履歴保持は retire debt を増幅しない範囲に限定。

- retention TTL
- retention budget
- retention decay policy

履歴用途（replay/debug/migration分析）は retention contract を満たす場合のみ許可。

---

## 6. Publication Architecture v1.6

### 6.1 canonical path

最終 canonical publication path は `core::RuntimePublicationCoordinator`。

`audioengine::ISRRuntimePublicationCoordinator` は migration bridge adapter とし、最終的に thin forwarding 経路へ収束させる。

### 6.2 authority と execution の分離

```text
Authority singularity != execution serialization
```

### 6.3 publication domains

- publication ordering domain
- publication parallelism domain
- retire independence domain

### 6.4 publication completion semantic

| phase | 意味 |
| --- | --- |
| visible | observe可能 |
| causally stable | retire/order整合 |
| diagnostically stable | compare/telemetry整合 |
| reclaim-stable | reclaim可能 |

release gate は `causally stable` 以上を必須とする。

### 6.5 semantic priority governance（新規）

all publication equal を禁止し、semantic severity に応じて優先度を制御する。

| publication class | severity |
| --- | --- |
| bypass toggle / topology rebuild | high |
| execution mode transition | high |
| telemetry metadata / analyzer update | low |

高severity publication の遅延を低severity payload が阻害してはならない。

---

## 7. Publication Lineage / Sequence Lattice

- Prepared / Published / Observed / Retired
- multi-publication overlap
- publication DAG
- semantic sequence lattice
- coefficient publication atomicity

cross-domain monotonicity 崩壊は Critical。

---

## 8. Deterministic / Temporal Hash & Invariant Validation

### 8.1 deterministic serialization

- canonical field ordering
- stable topology ordering
- pointer-address exclusion
- allocator-state exclusion
- deterministic normalization

### 8.2 epistemic limitation

```text
hash mismatch => definite mismatch
hash match => probabilistic equivalence only
```

### 8.2.1 hash authority prohibition（新規）

`same hash => same runtime` の推論を禁止する。

- hash は diagnostic fingerprint のみ
- authority identity / lifecycle 判定に hash を使用しない
- rollback trigger は hash 単独で発火させない

### 8.3 temporal semantic class

time-varying semantic は windowed compare で評価。

### 8.4 semantic invariant validator（新規）

hash は補助。主役は invariant validator。

必須 invariant:

- sequence lattice consistency
- coefficient-topology atomicity
- retire visibility closure
- timing causality consistency

---

## 9. RuntimeWorldReadShim Purity & Locality

### 9.1 consistent observation epoch

同一epochでの観測を保証。

### 9.2 ReadShim minimality

- `getRuntimeWorldSnapshot()` 以外の拡張Facade API禁止
- legacy/crossfade混載禁止

### 9.2.1 observe path minimality governance（新規）

single observe path を bottleneck 化しないため、hot path での複合処理を制限する。

禁止:

- observe中の compare hook 常時実行
- observe中の telemetry aggregation 常時実行
- observe中の projection deep lookup
- observe中の hidden synchronization

許可:

- O(1) snapshot pointer acquire
- bounded metadata read
- defer-only instrumentation enqueue

### 9.3 ReadShim purity

- no semantic cache
- no mutable authority
- no write-back
- no hidden synchronization

### 9.4 projection usage prohibition

projection は publication/retire/semantic branching に影響してはならない。

### 9.5 projection invalidation economics

- invalidate coalescing
- invalidate budget per interval
- invalidation backoff

### 9.6 projection locality governance（新規）

global invalidation storm を禁止。

- locality-aware invalidation (domain/section)
- hot-path projection 保護

### 9.7 projection semantic austerity rule（新規）

projection を secondary semantic cache へ拡張してはならない。

禁止例:

- cachedResolvedRouting を authority substitute として利用
- cachedEffectiveTopology を semantic branching source に利用
- cachedTransitionInfo を publication causal 判定に利用

projection は ephemeral read aid に限定し、semantic decision source として扱わない。

### 9.8 projection freshness contract（新規）

`projection == runtime truth` の誤認を防ぐため鮮度契約を定義する。

- max staleness budget
- freshness metadata 必須
- stale projection の compare/branch 利用禁止

freshness 逸脱時は UI/diagnostic を degrade 表示し、semantic authority と混同させない。

---

## 10. Topology / Transition / ExecutorLocal

- topology authority = RuntimeWorld.topology
- transition decomposition
- ExecutorLocal admissibility
- executor-local lifetime

### 10.1 crossfade authority boundary（新規）

crossfade subsystem が timing authority を侵食してはならない。

- effective activation timing の authority は RuntimeWorld.timing のみ
- crossfade は timing decision を上書きしない
- crossfade は smoothing/executor-local 補助に限定

---

## 11. Retire / Reclaim Economics v1.6

### 11.1 pressure policy

| 状態 | 許可動作 |
| --- | --- |
| mild | rebuild coalescing |
| medium | publication throttle |
| severe | rebuild reject |
| critical | emergency drain |

### 11.2 partial reclamation

- subtree reclaim
- reclaim batch budget
- reclaim burst 抑制

### 11.3 reclaim debt governance

- debt budget
- sustained lag threshold
- convergence metric

### 11.3.1 reclaim convergence SLA（新規）

`eventual cleanup optimism` を禁止し、bounded convergence を SLO/SLA として定義する。

| metric | requirement |
| --- | --- |
| reclaim debt | bounded |
| retire lag | bounded |
| retire queue residency | bounded |
| reclaim burst | bounded |

SLA連続違反時は `degrade -> fail-closed` へ段階遷移する。

### 11.4 retire semantic closure

- stale observe 終了
- compare contamination 終了
- telemetry divergence 閉鎖
- generation aliasing 防止

### 11.5 retire completion semantic（新規）

retire 完了は以下すべてを満たす必要がある。

- observe不可
- compare不可
- telemetry参照不可
- projection invalidated
- reclaim完了

queueからの除去のみで完了扱いにしてはならない。

### 11.6 retire lifecycle state machine（新規）

retire を single-dimensional queue として扱わない。

Lifecycle:

`visible -> compare-eligible -> telemetry-retained -> replay-retained(optional) -> reclaim-eligible -> reclaimed`

各遷移で可視性・保持責務・削除責務を明示し、状態スキップを禁止する。

---

## 12. Shadow Compare v1.6

### 12.1 Semantic Compare Contract

implementation detail comparison を禁止。

### 12.2 behavioral semantic class

bounded audible equivalence を許容。

### 12.2.1 perceptual equivalence class（新規）

DSP実運用における practical stable 判定は perceptual class を許容する。

許容例:

- coefficient epsilon drift
- overlap phase 微差
- SIMD reorder 差
- denormal suppression 差

ただし可聴差・因果差に到達する場合は mismatch として扱う。

### 12.2.2 audible divergence budget / transition smoothness（新規）

perceptual stability を定量化する。

- audible divergence budget
- transition smoothness metric（crossfade discontinuity 指標）
- listener-perceived stability gate

### 12.5 risk-weighted compare governance（新規）

compare を diff engine 化しない。

| risk tier | compare policy |
| --- | --- |
| Critical semantic | exhaustive / high frequency |
| High semantic | targeted / medium frequency |
| Medium semantic | sampled |
| Low semantic | opportunistic |

`全部比較` は禁止。compare cost budget を超過する設定変更は reject する。

### 12.3 production shadow sampling policy

| mode | compare rate |
| --- | --- |
| debug | full |
| nightly | dense |
| production | sampled |

releaseで完全OFF禁止。

### 12.4 migration safety infrastructure contract（新規）

shadow compare は debug-only feature ではなく migration safety infrastructure と定義。

---

## 13. Rollback / Recovery v1.6

### 13.1 rollback limitation

rollback は damage containment。

### 13.2 causal rollback ordering

publication -> retire -> shadow -> telemetry の順序拘束。

### 13.3 recovery convergence contract

recover後に drift/debt が収束すること。

### 13.4 recovery equilibrium semantic（新規）

- reclaim debt 減衰
- queue burst 収束
- mismatch trend 収束

rollback 依存ではなく self-healing convergence を優先。

### 13.5 dynamic convergence control（新規）

single-event rollback 前提を禁止し、persistent degraded oscillation を制御する。

対象:

- publication latency oscillation
- retire debt oscillation
- compare mismatch oscillation
- telemetry overload oscillation

制御方針: isolation / throttling / adaptive backoff / staged recovery。

---

## 14. Verifier Governance v1.6

### 14.1 epistemic limitation

verifier は known failure detector。

### 14.2 normalization policy

state explosion 抑制。

### 14.3 runtime evidence hierarchy

| evidence | 信頼度 |
| --- | --- |
| static verifier | 中 |
| short soak | 中 |
| long churn soak | 高 |
| production telemetry | 最高 |

### 14.4 governance source hierarchy（新規）

設計意思決定の優先順:

1. Contract (source of truth)
2. Policy (operation rules)
3. Verifier (validation)

Verifier が policy engine として振る舞うことを禁止。

### 14.5 evidence hierarchy governance（新規）

`verifier pass = operational trust` への昇格を禁止する。

- verifier pass は必要条件
- operational trust は soak + telemetry + disturbance replay を含む十分条件で判定

---

## 15. Temporary / Migration Decay Governance

### 15.1 temporary decay

| age | action |
| --- | --- |
| 7日 | warning |
| 30日 | CI fail |
| 90日 | merge prohibited |

### 15.2 migration decay contract（新規）

Bridge runtime の永久化を禁止。

対象:

- dual publication
- compatibility shim
- migration telemetry
- migration-only shadow overlays

各項目に removal milestone / expiry / verifier linkage を必須化。

### 15.3 migration decomposition deadlines（新規）

migration safety 機構の恒久残置を防ぐため、以下を義務化する。

- dual-runtime artifacts ごとの decomposition owner
- removal deadline（calendar-bound）
- completion evidence（soak + telemetry）
- deadline超過時の escalation

期限超過で未分解の場合、新規 migration feature merge を一時停止する。

---

## 16. Progress Guarantee / RT Starvation / Tail Governance

### 16.1 progress taxonomy

| class | RT許可 |
| --- | --- |
| blocking | 禁止 |
| lock-based | 禁止 |
| lock-free | 条件付き |
| wait-free bounded | 推奨 |

### 16.2 tail-latency governance

- callback stall P99.9
- reclaim burst tail
- compare overhead tail

平均値のみ評価を禁止。

### 16.3 probabilistic stability governance（新規）

tail を deterministic 前提で扱わない。

外乱要因:

- OS scheduling
- cache eviction
- NUMA
- IRQ interference
- device jitter

確率的 tail 指標（quantile band / exceedance probability）で stability 判定する。

---

## 17. Telemetry / Shadow Runtime 防止

### 17.1 observer effect

A/B 比較を定期実施。

### 17.2 telemetry dependency minimization（新規）

telemetry の shadow runtime 化を禁止。

禁止:

- own transitions
- own semantic timeline
- own authority-like projections

許可:

- passive observation
- bounded aggregation

### 17.3 feedback-loop stability analysis（新規）

runtime を feedback system として扱い、増幅ループを監視する。

例:

`telemetry increase -> compare pressure -> publication latency -> queue backlog -> telemetry increase`

増幅検出時は loop gain を下げる制御（sampling reduction / compare throttling / queue shedding）を適用する。

---

## 18. Stability Contract

| 指標 | 定義 |
| --- | --- |
| semantic stability | drift budget 内 |
| RT stability | callback percentile budget 内 |
| reclaim stability | bounded debt 内 |
| publication stability | monotonicity 維持 |
| migration stability | rollback/recovery convergence 達成 |

---

## 19. Operational Cost Contract

| metric | budget type |
| --- | --- |
| publication latency | bounded |
| reclaim burst | bounded |
| cache miss increase | bounded |
| allocation amplification | bounded |
| compare overhead | bounded |

### 19.1 failure economics governance（新規）

`No Failure Runtime` を目的化しない。目的は以下の bounded failure 特性である。

- failure bounded
- failure detectable
- failure recoverable
- failure convergent

過剰な `verifier/compare/abstraction` 増設で cost collapse を起こす変更は禁止。

---

## 20. Operational Stability Envelope v1.6

### 20.1 定量 envelope

| metric | target (example) |
| --- | --- |
| publication/sec | <= X |
| rebuild/sec | <= Y |
| overlap concurrency | <= Z |
| reclaim lag | <= N ms |
| automation density | <= M / sec |

### 20.1.1 adaptive operational envelope（新規）

運用境界を固定閾値で凍結しない。環境変化に応じて適応更新する。

入力要因:

- allocator state
- cache topology
- device load
- telemetry density
- rebuild frequency

出力規則:

- threshold auto-tuning（bounded range）
- false-alarm suppression window
- blind-spot detection backtest

### 20.2 equilibrium semantics

- sustained degradation の非継続
- debt/lag の収束傾向
- runaway cost の非発生

### 20.3 global equilibrium governance（新規）

局所安定の合成を global stability と見なさない。

同時外乱:

- rebuild storm
- automation storm
- telemetry burst
- UI attach/detach
- device migration
- preset morphing

cross-subsystem oscillation を観測し、global equilibrium 指標で受入判定する。

---

## 21. Entropy Accumulation Governance

監視対象:

- allocator fragmentation
- retire lag drift
- telemetry skew
- compare debt
- queue imbalance

entropy budget 超過時は再平衡を発火。

### 21.1 memory equilibrium governance（新規）

steady-state memory drift を監視対象に明示追加する。

- allocator fragmentation trend
- PMR arena drift
- projection cache growth
- retire backlog oscillation

長期平均が equilibrium band を逸脱した場合、rebalancing plan を必須化する。

---

## 22. Pragmatic Ceiling / Performance-Preserving Rewrite

```text
abstraction increase without operational justification prohibited
```

性能証跡とRT worst-case latency比較を必須化。

### 22.1 RT locality preservation contract（新規）

semantic purity は `bounded operational economics` 下で成立させる。

禁止:

- scratch 再利用の全面禁止
- SIMD/local cache reuse の破壊
- allocator churn を増幅する purity-only rewrite

許可:

- semantic authority を保った executor-local mutable reuse
- locality 最適化を伴う bounded mutation

### 22.2 complexity budget governance（新規）

AI改修での runtime complexity asymmetry 累積を抑制する。

| dimension | budget |
| --- | --- |
| semantic layer count growth | bounded |
| verifier rule growth | bounded |
| telemetry channel growth | bounded |
| rollback branch growth | bounded |

予算超過変更は decomposition plan なしでは受け入れない。

### 22.3 complexity decay governance（新規）

complexity の単調増加を禁止する。

- verifier/telemetry/semantic layer/shim の定期削減サイクル
- expired temporary の強制回収
- net complexity delta の監査

### 22.4 runtime-native layout priority（新規）

RuntimeWorld を serialization 都合で歪めない。

優先順位:

1. runtime hot-path locality
2. hot/cold separation
3. bounded indirection
4. serialization convenience

serialization hostage 化を禁止する。

---

## 23. CI / Soak 実行設計 v1.6

### 23.1 PR

- static verifier core
- ordering/observe/topology/coeff core checks

### 23.2 Nightly

- shadow compare dynamic
- retire debt / telemetry pressure
- observer-effect A/B
- entropy trend checks

### 23.3 Release

- full verifier suite
- full soak matrix
- fail-closed gate

### 23.4 soak success criteria

- mismatch budget 内
- publication latency budget 内
- reclaim debt budget 内
- retire starvation budget 内
- drift-free duration 達成
- envelope 指標内で安定
- recovery convergence 達成

---

## 24. 定量ゲート

- Critical mismatch: 1件で release block
- High mismatch: 3件/10min超で release block
- publication monotonicity violation: 1件で fail-closed
- retire starvation violation: 1件で fail-closed
- sustained reclaim lag threshold 超過で degrade/fail
- rollback retrigger: 5件/10min超で 1段 escalation
- entropy budget 超過継続で fail/degrade

---

## 25. AI実装統治ルール

- 1変更 = 1目的
- 無関係 subsystem 変更禁止
- スコープ拡張時は理由/代替/承認必須
- architecture rewrite / framework replacement / concurrency model rewrite は明示承認必須

---

## 26. トレーサビリティ（v1.6追加）

| 追加要求 | セクション |
| --- | --- |
| Authority regression governance | 4 |
| Semantic monolith防止 | 3.2, 5 |
| Historical retention governance | 5.5 |
| Semantic causality contract | 2.2 |
| Semantic invariant validator | 8.4 |
| Phase-specific stability | 23 |
| Shutdown causality関連（runbook前提） | 13, 23 |
| Retire completion semantic | 11.5 |
| Recovery equilibrium | 13.4, 20.2 |
| Telemetry shadow runtime防止 | 17.2 |
| Governance source hierarchy | 14.4 |
| Projection locality | 9.6 |
| Audible equivalence運用 | 12.2 |
| Migration decay contract | 15.2 |
| Self-stabilizing semantics | 20.2, 21 |
| Governance self-consistency | 2.3 |
| Adaptive operational envelope | 20.1.1 |
| RuntimeWorld archive化防止 | 5.5, 21.1 |
| Risk-weighted compare | 12.5 |
| RT locality preservation | 22.1 |
| Observe bottleneck防止 | 9.2.1 |
| Reclaim convergence SLA | 11.3.1 |
| Projection austerity | 9.7 |
| Migration decomposition deadlines | 15.3 |
| Hash authority prohibition | 8.2.1 |
| Failure economics governance | 19.1 |
| Perceptual equivalence class | 12.2.1 |
| AI runtime drift / complexity budget | 22.2 |
| Semantic density budget | 2.4 |
| Global equilibrium governance | 20.3 |
| Feedback-loop stability | 17.3 |
| Probabilistic tail stability | 16.3 |
| Semantic migration compatibility | 3.4.1 |
| Audible divergence / smoothness | 12.2.2 |
| Semantic priority governance | 6.5 |
| Retire lifecycle state machine | 11.6 |
| Dynamic convergence control | 13.5 |
| Projection freshness contract | 9.8 |
| Complexity decay governance | 22.3 |
| Crossfade authority boundary | 10.1 |
| Runtime-native layout priority | 22.4 |
| Evidence hierarchy governance | 14.5 |
| Disturbance convergence semantics | 13.5, 20.3 |

---

## 27. 完了定義（v1.6）

本詳細設計の完了は以下を満たした時とする。

- `base_plan.md v2.3` 契約が実装拘束へ分解済み
- `ai_governance_v1_2.md` 必須規約を満たす
- v1.5までの不足 + 追加レビュー指摘を反映済み
- anti-erosion, recovery, equilibrium が設計拘束化済み
- 数年スケール保守で authority regression / operational collapse を抑止可能

---

## 付録A: AI実装開始チェックリスト

- [ ] authority regression audit を実施した
- [ ] semantic dependency DAG を破っていない
- [ ] publication completion phase を定義した
- [ ] retire completion semantic を満たす
- [ ] production shadow sampling を維持する
- [ ] verifier を policy sourceとして扱っていない
- [ ] temporary/migration items に decay条件がある
- [ ] tail-latency(P99.9) と entropy budget を監視する
- [ ] recovery convergence と equilibrium 指標が設定済み
