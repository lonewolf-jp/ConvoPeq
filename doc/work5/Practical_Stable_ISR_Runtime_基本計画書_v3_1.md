# ConvoPeq — Practical Stable ISR Runtime 基本計画書 v3.1

## — Single Authoritative Observable Runtime Migration —

---

## 0. 文書目的

本計画書は、ConvoPeq の Runtime を **長時間実運用で破綻しにくい形**へ段階移行するための基本計画である。
本計画の主目的は、理論上の美しさよりも以下を優先することにある。

- 実運用安定性（長時間連続運転耐性）
- 移行中の複雑化抑止
- authority 増殖防止
- verify-first / fail-closed 運用
- 緊急時の監査可能な例外運用

---

## 1. 最重要方針（最終定義）

本計画の本質は、機能追加ではなく以下である。

> **runtime authority topology を収束させる**

最終目標は次の1行で定義する。

> **Single Authoritative Observable Runtime**

すなわち、Runtime は最終的に

- observe 単一
- authority 単一
- publication 単一
- generation 単一
- retire 単一

を満たす。

---

## 2. 最終完成像

Audio Thread は次のみを実行する。

- `const RuntimeWorld* world = runtimeCoordinator.consume();`
- `executor.process(world);`

Audio Thread で禁止するもの:

- runtime mutation
- authority resolution
- transition coordination
- retire coordination
- graph mutation
- visibility routing

---

## 3. 設計原則

### 3.1 本質原則

ISR の本質は「mutable state 削減」ではなく、次である。

> **runtime meaning source の単一化**

### 3.2 最重要禁止事項

移行中に最も危険な行為を明示禁止する。

1. 新 observe path の追加
2. 新 transition authority source の追加
3. field-level partial publication
4. crossfade semantic source の多重化
5. legacy authority の無期限共存
6. non-authoritative state による runtime branching

---

## 4. 現状課題（要約）

問題の本質は以下の重複である。

### 4.1 runtime activity の重複

- `activeRuntimeDSPSlot`
- `fadingRuntimeDSPSlot`
- `TransitionState.active`
- `RuntimeGraph.activeNode`

### 4.2 transition meaning の重複

- preparedCrossfade
- fade atomics
- pending flags
- executor fade state

### 4.3 generation identity の重複

- generation
- runtimeVersion
- transitionId

---

## 5. Governance Hardened 仕様（v3.1 追加）

### 5.1 Authority Classification System

Runtime state は必ず次で分類する。

- `Authoritative`
- `Derived`
- `Diagnostic`
- `ExecutorLocal`
- `LegacyTemporary`

未分類 state は禁止。

### 5.2 RuntimeGeneration 規約

唯一 authoritative な generation は `RuntimeGeneration` とする。

- 原則: runtime branching は `==` / `!=` のみ
- 例外: `RuntimeCoordinator` 内部限定で `isNewer(a,b)` を1実装のみ許可
- 禁止: ordering logic の全域拡散

`runtimeVersion` は diagnostic only、`transitionId` は trace only とする。

### 5.3 Branch Rule

以下による execution/publish/retire branching を禁止:

- runtimeVersion
- transitionId
- debug flag

分岐は authoritative state のみ許可。

### 5.4 Publication Rule

唯一許可される publication 単位:

- `publish(RuntimeWorld*)`

以下の分割 publish を禁止:

- graph / fade / snapshot / transition 単位 publish

### 5.5 Observe Path Rule

最終 observe path は `RuntimeWorld` のみ。
新 visibility slot / pending flag / transition observable / visibility atomic 追加を禁止。

### 5.6 Crossfade Rule

crossfade は observable semantic state ではなく、

> **executor implementation detail**

として扱う。
移行順:

1. observe path 統一
2. legacy semantic source 廃止
3. executor-local execution 化

### 5.7 Retire Governance

`RetireEnqueueResult` を導入:

- `Success`
- `QueuePressure`
- `QueueFull`
- `Shutdown`

`QueuePressure` は failure ではなく backpressure signal と定義。
Coordinator は pressure 時に coalescing/throttling/restart を実施。

### 5.8 Break-glass Rule（運用例外）

監査可能な緊急例外のみ許可。永続例外は禁止。

`BreakGlassOverride` 必須要素:

- id
- owner
- reason
- expiration
- rollback_plan

必須規約:

1. expiration 必須
2. CI warning mandatory
3. release build で persistent override 禁止
4. override 使用時 soak mandatory

### 5.9 LegacyTemporary Manifest

`LegacyTemporary` はコメント管理のみを禁止し、次で機械管理する。

- `.github/isr-legacy-temporary.json`

必須項目:

- symbol
- owner
- replacement_authority
- removal_phase
- deadline
- scope

未登録 legacy は build fail。

### 5.10 Coordinator 肥大化防止

RuntimeCoordinator は **meaning coordinator** に限定。

禁止責務:

- DSP ownership / lifecycle execution
- UI orchestration
- cache management
- async IO

### 5.11 Governance Budget Rules

- Authority Migration Budget: phase あたり authority source 増加禁止
- Observe Growth Budget: phase あたり observe path 増加禁止（0 or negative）
- Legacy Lifetime Cap: LegacyTemporary は2 phase超存続禁止
- Semantic Duplication Budget: 同一 semantic state の3箇所以上共存禁止

---

## 6. Phase 計画

### Phase 1: Authority Freeze

DoD:

- authoritative generation singularization
- publication authority singularization
- authority classification complete
- verify-first mandatory
- non-authoritative branch prohibition

### Phase 2: Observe Path Unification

DoD:

- Audio Thread observe = RuntimeWorld only

### Phase 3: Legacy Authority Removal

DoD:

- dual authority coexistence 終了
- legacy observe path 全廃

### Phase 4: Crossfade Executor-local Migration

DoD:

- transition semantic source singularization
- crossfade semantic source の world 外化

### Phase 5: Publication Atomicity Completion

DoD:

- `publish(RuntimeWorld*)` のみ

### Phase 6: Retire Pressure Governance

DoD:

- silent drop 完全廃止
- pressure feedback 完成
- backlog slope stable

---

## 7. Verification-First / CI

### 7.1 運用原則

- verify-first mandatory
- fail 時は warning ではなく build fail（fail-closed）

### 7.2 必須 CI

1. authoritative branch verifier
2. authority duplication verifier
3. observe path expansion verifier
4. publication authority verifier
5. legacy expansion verifier
6. mixed generation observe detector
7. partial publication detector
8. executor-local leakage detector
9. crossfade overlap detector
10. retire backlog slope detector
11. world retention leak detector
12. non-authoritative observe detector

---

## 8. Soak Test 仕様

短時間の成功は合格条件にしない。長時間耐性を必須化する。

### 8.1 必須 Soak

- IR reload storm（10Hz〜50Hz, 4h以上）
- bypass storm
- automation storm
- sample rate churn（44.1/48/96/192）
- UI attach/detach storm
- suspend/resume storm

### 8.2 必須 metrics

- XRUN count
- stale observe count
- authority duplication count
- retire backlog slope
- world leak count
- publication latency drift
- overlap rejection count

### 8.3 Failure Taxonomy

- Class-A: audio corruption
- Class-B: generation drift
- Class-C: stale observe
- Class-D: retire backlog divergence
- Class-E: world retention leak
- Class-F: authority duplication regression

---

## 9. 最終結論

本計画は ISR 機能拡張計画ではない。
目的は以下の収束である。

> **runtime meaning source collapse**

ConvoPeq が目指すべき最終形は `single-world runtime` であり、

- observe 単一
- authority 単一
- publication 単一
- generation 単一
- retire 単一

を満たす Runtime である。
これを **Practical Stable ISR Runtime** の定義とする。

---

## 付録A: 採用判定

- v3.1 を正式採用対象とする
- ただし fail-closed CI と期限付き例外運用を同時導入すること
- 例外運用は break-glass 経路へ統一し、監査不能な緩和を禁止する
