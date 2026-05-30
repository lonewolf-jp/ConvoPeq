# ConvoPeq — Practical Stable ISR Bridge Runtime 完全移行計画 v2.3

## サブタイトル

drift せず壊さず到達するための、実装可能かつ運用可能な ISR migration governance specification。

---

## 0. 目的と適用範囲

本計画の目的は、ConvoPeq を次へ移行することである。

> Single Authoritative Runtime Semantic Source

ただし本計画は「理想 Runtime の定義」だけを目的としない。
本質は次である。

> 既存 mutable / hybrid runtime から、long-run semantic drift を起こさず安全に到達すること。

`doc/work6/problem.md` で整理された課題（observe duplication / authority duplication / publication fragmentation / snapshot semantic duplication / retire ambiguity / transition leakage / long-run instability）を段階的に閉じる。

---

## 1. 設計原則

### 1.1 最上位原則

1. Semantic First
   - `compile success` / `short test pass` を成功条件にしない。
   - `semantic equivalence` を成功条件にする。
2. Fail-Closed Governance
   - verifier fail は build fail。
3. Safety before Purity
   - 理想純度より移行中破綻回避を優先。
4. Single Source Contracts
   - authority / observe / publication / generation / retire を契約で単一化。
5. Operational Realism
   - PR / Nightly / Release のゲートを分離。

### 1.2 非目標

- 一括全面 rewrite。
- verifier 未整備の先行実装。
- RT 制約を無視した ownership 方式の即断。

---

## 2. 現状の危険構造

1. Observe Path Duplication
2. Authority Duplication
3. Publication Fragmentation
4. Snapshot Semantic Duplication
5. Transition/Crossfade Leakage
6. Retire Pressure/Starvation Risk
7. Runtime Topology Authority Split

---

## 3. Runtime Semantic Schema（必須）

authority 境界を曖昧化しないため、runtime semantic の正式 schema を定義する。

```cpp
struct RuntimeSemanticSchema
{
    GenerationSemantic generation;
    TopologySemantic topology;
    RoutingSemantic routing;
    ExecutionSemantic execution;
    PublicationSemantic publication;
    OverlapSemantic overlap;
    RetireSemantic retire;
};
```

### 3.1 基本契約

- `RuntimeWorld` 内の schema 定義項目のみ authority を持つ。
- schema 外の状態は projection か diagnostics に分類する。
- projection / diagnostics は branch authority を持たない。

### 3.2 ExecutionSemantic authoritative fields（新規明確化）

ExecutionSemantic は最低限次の authority field を持つ。

- processing graph order
- activation epoch
- latency compensation semantic
- deferred activation semantic
- processing mode gates（runtime execution branch を決めるもの）

禁止:

- executor-local transient state を execution authority として扱うこと。

### 3.3 OverlapSemantic authority contract（新規明確化）

- overlap authority は `RuntimeWorld.overlap` のみ。
- executor overlap state は projection（実行補助）であり authority 不可。
- overlap decision branch は `RuntimeWorld.overlap` のみ参照。

### 3.4 完了条件

- 全 runtime field が `Authoritative / Derived / Diagnostic / ExecutorLocal / LegacyTemporary` に分類済み。
- schema と authority inventory が一致。

---

## 4. 最終到達条件（Definition of Complete Migration）

| 項目 | 完了条件 |
| --- | --- |
| Observe source | `RuntimeWorld` のみ |
| Authority source | `RuntimeSemanticSchema` で定義された単一 authority |
| Publication | single-source + monotonic + sequence-verified + visibility-monotonic |
| Generation | 単一 authoritative generation |
| PublicationEpoch mapping | RuntimeGeneration に単調対応 |
| Topology authority | `RuntimeWorld.topology` のみ |
| Snapshot semantics | projection artifact のみ（authority不可） |
| Transition/Crossfade | executor-local（diagnostic-only 可） |
| Retire | pressure + starvation contract 準拠 |
| Legacy mutable semantic | 0 |
| Long-run drift | Soak 契約内で不検出 |

---

## 5. v2.3 契約セット

### 5.1 Publication Authority Contract

- single-source publication path
- monotonic publication sequence
- sequence verification
- out-of-order reject
- legacy publication API zero-call（最終）

#### 5.1.1 識別子

```cpp
PublicationSequenceId
PublicationEpoch
```

#### 5.1.2 visibility monotonicity

```text
ObservedPublicationSequence >= LastObservedPublicationSequence
```

#### 5.1.3 PublicationEpoch ↔ RuntimeGeneration contract（新規）

```text
PublicationEpoch must map monotonically to RuntimeGeneration.
```

- `PublicationEpoch(a) < PublicationEpoch(b)` なら `RuntimeGeneration(a) <= RuntimeGeneration(b)` を満たす。
- 別 timeline 化（drift source）を禁止する。

### 5.2 Runtime Topology Authority Contract

- topology authority は `RuntimeWorld` 内のみ。
- projection topology は branch authority を持たない。

禁止:

- processing-order branching by projection topology
- slot-based topology branching
- transition-state topology override

### 5.3 Snapshot Semantic Contract

```text
RuntimeWorld = authoritative semantic container
snapshot = projection artifact
```

- snapshot は branch / retire / publish authority を持たない。
- snapshot 由来の write-back authority を禁止。

### 5.4 Partial Publication Formal Definition（新規）

```text
Partial publication = RuntimeSemanticSchema の proper subset のみが可視化され、
同一 PublicationSequence 内で semantic completeness を満たさない状態。
```

契約:

- Partial publication は 0。
- schema completeness を満たさない publish は reject。

### 5.5 RuntimeSemanticHash Contract

```cpp
struct RuntimeSemanticHash
{
    uint64_t topologyHash;
    uint64_t executionHash;
    uint64_t routingHash;
    uint64_t payloadHash;
    uint64_t publicationSemanticHash;
    uint64_t overlapSemanticHash;
    uint64_t retireSemanticHash;
};
```

#### 5.5.1 collision policy（新規）

- hash mismatch => mismatch（fail/rollback判定対象）。
- hash match は semantic equality の十分条件ではない。
- 重要判定では hash + direct contract checks を併用。

### 5.6 Shadow Semantic Compare Contract

比較対象:

- generation
- topology
- overlap decision
- retire ordering
- execution ordering
- publication timing
- visibility delay
- semantic hash

#### 5.6.1 mismatch taxonomy

| mismatch 種別 | severity | 標準動作 |
| --- | --- | --- |
| topology mismatch | Critical | fail + rollback candidate |
| semantic hash mismatch | Critical | fail + rollback candidate |
| overlap mismatch | High | block release + investigate |
| retire ordering mismatch | High | block release + pressure mode |
| publication timing drift | Medium | throttle/coalesce + monitor |
| visibility delay drift | Medium | monitor + threshold action |

### 5.7 Retire Pressure Contract

| 状態 | 必須動作 |
| --- | --- |
| QueuePressure (mild) | rebuild coalescing |
| QueuePressure (medium) | publication throttle |
| QueueFull (severe) | rebuild reject / admission strict |
| Critical | emergency drain + protective mode |

starvation prevention:

- `maxRetireDeferralEpochs` 契約化
- QueueFull 継続時の無限滞留禁止

### 5.8 Rollback Execution Contract

| レベル | 動作 |
| --- | --- |
| Soft rollback | publication freeze + shadow-only compare |
| Medium rollback | shadow-only mode + new path write stop |
| Hard rollback | legacy fallback path へ切替 |
| Emergency | runtime rebuild suspend + emergency drain |

#### 5.8.1 rollback hysteresis（新規）

- cooldown: 30 sec 以上（環境で調整可）
- repeated trigger threshold: N 回超で escalation
- escalation: Soft → Medium → Hard → Emergency
- flapping（短周期往復）を禁止

### 5.9 Ownership Evaluation Contract

ownership 方式比較で必須測定:

- cache contention
- atomic traffic
- epoch progression impact
- RT reclaim worst-case latency
- ABA risk analysis

### 5.10 Percentile Telemetry Contract

| 指標 | 契約例 |
| --- | --- |
| publication latency | P95 <= threshold |
| retire defer epochs | P99 <= threshold |
| visibility delay blocks | percentile threshold |
| overlap timing drift | percentile threshold |

### 5.11 Telemetry Storage Contract（新規）

- clock source contract（monotonic clock を使用）
- retention window を固定（例: PR短期, Nightly長期）
- sampling monotonicity（時系列逆行禁止）
- overwrite policy 明文化（silent overwrite 禁止）

### 5.12 Governance Consistency Contract（新規）

governance 自体の drift を防ぐため、以下を上位契約として固定する。

- governance dependency DAG の循環禁止
- semantic ownership の固定化（categoryごとの authority owner）
- taxonomy変更時の mandatory review
- verifier / compare / rollback 契約の整合義務
- governance registry 不整合時 fail-closed

### 5.13 Long-run Equilibrium Contracts（新規）

長期運用での silent degradation を防ぐため、以下を上位契約に昇格する。

- adaptive operational envelope（固定閾値前提を禁止）
- historical retention budget（RuntimeWorld archive化防止）
- risk-weighted compare governance（全部比較禁止）
- reclaim convergence SLA（eventual cleanup 楽観を禁止）
- projection semantic austerity（secondary authority化禁止）
- migration decomposition deadlines（dual runtime 恒久化防止）
- hash authority prohibition（hashは診断指紋）
- failure economics governance（failure bounded/detectable/recoverable/convergent）
- complexity budget governance（AI起因の複雑性非対称成長を抑制）
- perceptual equivalence class（DSP実運用許容差を明示）

追加 survivability 契約:

- semantic density budget（interaction density 爆発抑制）
- global equilibrium governance（cross-subsystem oscillation 検知）
- feedback-loop stability analysis（増幅ループ抑制）
- probabilistic tail governance（確率的外乱を前提化）
- semantic migration compatibility（topology同一でも drift 検出）
- semantic priority governance（publication severity 優先制御）
- retire lifecycle state machine（multi-stage retire 管理）
- projection freshness contract（stale truth 誤認防止）
- complexity decay governance（単調成長禁止）
- runtime-native layout priority（serialization hostage 防止）
- evidence hierarchy governance（verifier pass の過信防止）
- disturbance convergence semantics（外乱後の再収束保証）

---

## 6. フェーズ構成（v2.3）

## Phase 0 — Migration Safety Layer（必須）

### Phase 0 目的

本移行で「壊さない」ための安全契約を先に固定する。

### Phase 0 実施項目

1. Coordinator collapse strategy
2. Ownership evaluation plan
3. Shadow compare infrastructure
4. Rollback execution + hysteresis contract
5. Telemetry/percentile/storage contract
6. Mismatch taxonomy の確定
7. Governance consistency baseline の確定
8. Equilibrium/complexity/reclaim SLA 初期値の確定

### Phase 0 Exit Criteria

- 後続フェーズ契約が文書化済み
- shadow compare レポート生成可能
- rollback 手順が実行可能手順として整備済み

---

## Phase 1 — Authority Freeze

### Phase 1 目的

authority 増殖停止。

### Phase 1 実施項目

- authoritative generation 単一化
- runtimeVersion を diagnostic-only 化
- transitionId を trace専用化
- authority registry と schema の一致化

### Phase 1 DoD

- authoritative generation 1系統
- runtimeVersion branching 0
- schema未分類 field 0

---

## Phase 2 — Coordinator Collapse & Publication Authority Collapse

### Phase 2 目的

publication semantic fragmentation を解消。

### Phase 2 実施項目

- publication path 統一
- PublicationSequenceId / PublicationEpoch 導入
- monotonicity + visibility monotonicity 検証
- PublicationEpoch↔Generation mapping 検証
- partial publication reject 実装
- legacy publication API 呼び出し削減

### Phase 2 DoD

- publication authority source 1
- publication monotonicity / visibility monotonicity pass
- mapping contract pass
- partial publication 0
- legacy publication API は manifest例外を除き 0

---

## Phase 3 — Observe Path Collapse

### Phase 3 目的

AudioThread observe を RuntimeWorld に一本化。

### Phase 3 実施項目

- direct slot/state observe 禁止
- shim 経由で RuntimeWorld read のみ許可

### Phase 3 DoD

- AudioThread legacy observe 0
- observe verifier pass

---

## Phase 4 — Snapshot Semantic Unification

### Phase 4 目的

snapshot semantic duplication を解消。

### Phase 4 実施項目

- snapshot を projection artifact に限定
- snapshot authority を全面禁止

### Phase 4 DoD

- snapshot authority usage 0
- snapshot contract verifier pass

---

## Phase 5 — RuntimeWorld Self-contained化

### Phase 5 目的

world semantic の自己完結化。

### Phase 5 実施項目

- ownership model を評価結果で確定
- world 外 mutable state 依存排除
- RuntimeWorld 単位 publish 固定

### Phase 5 DoD

- external semantic dependency 0
- partial publication 0

---

## Phase 6 — Transition/Crossfade Collapse

### Phase 6 目的

transition/crossfade の authority 漏れを除去。

### Phase 6 実施項目

- transition authority 剥離
- overlap authority を RuntimeWorld 側へ固定
- crossfade executor-local 化
- diagnostics channel を diagnostic-only で維持

### Phase 6 DoD

- transition branching authority 0
- overlap authority leakage 0
- crossfade semantic leakage 0

---

## Phase 7 — Retire Governance Hardened

### Phase 7 目的

retire drift / starvation / residency explosion 防止。

### Phase 7 実施項目

- pressure policy 完全適用
- starvation prevention 実装
- retire telemetry 契約化

### Phase 7 DoD

- silent drop 0
- starvation violation 0
- retire pressure verifier pass

---

## Phase 8 — Legacy Runtime Semantic Removal

### Phase 8 目的

legacy mutable semantic の最終除去。

### Phase 8 実施項目

- legacy authority/observe/publication path 段階削除
- LegacyTemporary の期限回収

### Phase 8 DoD

- legacy semantic 0
- LegacyTemporary entries 収束

---

## Phase 9 — Soak Validation & Operational Hardening

### Phase 9 目的

long-run 条件で drift 非発生を確認。

### Phase 9 テスト層

| 層 | 内容 |
| --- | --- |
| PR | 静的 verifier + 短時間 churn |
| Nightly | 長時間 soak |
| Release | full matrix + fail-closed |

### Phase 9 受け入れ条件

- critical mismatch 0
- percentile 契約内
- rollback trigger 非発火（既知一時例外を除く）
- adaptive envelope が false alarm storm / blind spot を発生させない
- reclaim convergence SLA 違反が継続しない
- complexity budget の継続超過がない
- feedback amplification が抑制されている
- global equilibrium 指標が oscillation bound 内に収まる
- disturbance 後に bounded-cost equilibrium へ再収束する

---

## 7. Verifier 体系（v2.3）

必須 verifier:

1. authority duplication verifier
2. runtime semantic schema verifier
3. execution semantic authority verifier
4. overlap semantic authority verifier
5. publication single-source verifier
6. publication monotonicity verifier
7. publication visibility-monotonicity verifier
8. publication epoch-generation mapping verifier
9. publication sequence verifier
10. partial publication verifier
11. observe path verifier
12. audio-thread forbidden observe verifier
13. runtime topology authority verifier
14. snapshot semantic contract verifier
15. transition collapse verifier
16. crossfade leakage verifier
17. diagnostic-only boundary verifier
18. world ownership verifier
19. retire pressure verifier
20. retire starvation verifier
21. semantic hash equivalence verifier
22. shadow compare contract verifier
23. telemetry storage contract verifier
24. rollback hysteresis verifier
25. legacy manifest expiry verifier

### 7.1 Verifier Dependency Graph

| verifier | depends on |
| --- | --- |
| publication monotonicity | publication single-source, authority duplication |
| publication visibility-monotonicity | publication monotonicity, observe path |
| mapping verifier | publication sequence, generation semantic |
| partial publication verifier | runtime semantic schema verifier |
| shadow compare contract | semantic hash equivalence, topology authority, retire pressure |
| release gate | critical mismatch 0, percentile contracts pass |

### 7.2 Verifier Tier Cost Governance（新規）

| tier | verifier 方針 |
| --- | --- |
| PR strict | cheap/static 中心 + 必須安全契約 |
| Nightly dynamic | shadow compare / telemetry / pressure 系を追加 |
| Release full | 全 verifier + full soak + fail-closed |

### 7.3 Governance/Equilibrium Verifier追加（新規）

上位契約の逆流反映として、次を必須 verifier 群へ追加する。

- governance consistency verifier
- adaptive envelope verifier
- historical retention budget verifier
- risk-weighted compare budget verifier
- reclaim convergence SLA verifier
- projection austerity verifier
- migration decomposition deadline verifier
- complexity budget verifier
- semantic density budget verifier
- feedback-loop stability verifier
- probabilistic tail stability verifier
- semantic migration compatibility verifier
- semantic priority governance verifier
- retire lifecycle state verifier
- projection freshness verifier
- complexity decay verifier
- runtime-native layout verifier
- evidence hierarchy verifier

---

## 8. Shadow Compare 運用契約（v2.3）

### 8.1 必須比較項目

- generation
- topology
- overlap
- retire ordering
- execution ordering
- publication timing
- visibility delay
- semantic hash

### 8.2 卒業条件

- mismatch rate <= threshold
- critical mismatch 0
- P95/P99 契約内
- rollback trigger 未発火

---

## 9. Rollback / Degrade ポリシー

### 9.1 Trigger

- publication sequence violation
- visibility regression
- semantic hash drift 継続
- retire starvation violation

### 9.2 実行レベル

- Soft / Medium / Hard / Emergency（5.8 契約）

### 9.3 Anti-oscillation

- hysteresis 契約に従い rollback flapping を禁止

---

## 10. 運用ガバナンス

### 10.1 Fail-Closed

- verifier fail => build fail
- policy schema/expiry mismatch => build fail

### 10.2 LegacyTemporary manifest

`.github/isr-legacy-temporary.json` を継続。
必須: owner / issue / rationale / expiry / replacement_authority / removal_phase / deadline。

### 10.3 例外管理

- 一時例外は manifest に owner/期限付きで明示
- 期限超過は fail-closed

### 10.4 Governance self-consistency（新規）

- ContractRegistry（契約の唯一台帳）
- VerifierRegistry（verifier依存とtier配線の唯一台帳）

### 10.5 Bounded Equilibrium Economics（新規）

最終運用目標を以下に固定する。

`runtime は長時間後に bounded-cost equilibrium へ自然収束する`

禁止:

- sustained cost escalation
- permanent dual-runtime dependency
- unbounded reclaim/telemetry/compare debt

---

## 11. 実装優先順位（v2.3）

### 最優先

1. Phase 0
2. Phase 2
3. Phase 3
4. Phase 4

### 次点

1. Phase 5
2. Phase 6
3. Phase 7

### 最終

1. Phase 8
2. Phase 9

---

## 12. v2.3 完了定義

- schema/contract/verifier/dependency が整合
- PR/Nightly/Release ゲートが稼働
- rollback/hysteresis 実行手順が運用可能
- telemetry storage 契約が運用で成立
- long-run drift 契約が継続的に満たされる
- governance consistency 契約が継続的に満たされる
- equilibrium/complexity/reclaim SLA 契約が継続的に満たされる

---

## 13. 最終宣言

以下を満たした時のみ、

> Practical Stable ISR Bridge Runtime 完全移行達成

と宣言する。

- RuntimeWorld が唯一の authoritative semantic container
- publication/observe/retire が単一契約へ収束
- snapshot/diagnostics は authority 非保持
- long-run soak で drift 契約内
- fail-closed governance 継続可能
