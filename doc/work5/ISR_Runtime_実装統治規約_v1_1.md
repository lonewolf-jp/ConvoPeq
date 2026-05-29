# ConvoPeq ISR Runtime 実装統治規約 — AI Implementation Governance Rules v1.1

本規約は `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md` を基本設計として、AI に詳細設計・実装・修正・レビュー・CI追加を実施させる際の必須統治規約である。

本規約の目的は、ISR 理論純化・コード量削減・機能追加高速化ではない。目的は以下である。

> 長時間実運用で破綻しにくい Runtime への収束

---

## 1. Safety-First Clause（最優先）

AI は以下の優先順位で判断すること。

1. **実運用安全性（最優先）**
2. authority singularization
3. observe path singularization
4. legacy authority reduction
5. beautification / abstraction

### 1.1 実運用安全性での禁止（非悪化必須）

以下を悪化させる変更を禁止する。

- XRUN 増加
- audio corruption
- RT stall
- deadlock
- retention leak 悪化
- suspend/resume 破綻
- host interoperability 悪化

---

## 2. 最重要原則

AI は常に以下を最優先概念として維持する。

> runtime meaning source singularization

### 2.1 禁止（目的の取り違え）

以下を目的化することを禁止する。

- architecture beautification
- abstraction purity
- generalized frameworkization
- reusable runtime platformization
- actor-system 化
- event-bus 化
- reactive framework 化

---

## 3. AI の最重要禁止事項

### 3.1 authority source 増殖

禁止例:

- 新 runtime slot
- 新 visibility pointer
- 新 pending flag
- 新 transition ownership
- 新 semantic cache

### 3.2 observe path 増殖

禁止例:

- 新 atomic observe
- 新 visibility routing
- 新 side-channel state
- 新 snapshot export

### 3.3 部分 publish

以下のような分割 publish を禁止する。

- `publishGraph(...)`
- `publishFade(...)`
- `publishTransition(...)`

### 3.4 RuntimeCoordinator 肥大化

禁止責務:

- DSP ownership
- cache ownership
- async IO
- UI orchestration
- lifecycle orchestration

### 3.5 executor-local state export

禁止:

- fade progression publish
- meter atomic export
- interpolation phase export

### 3.6 non-authoritative branching

以下による分岐を禁止:

- `if (runtimeVersion ...)`
- `if (transitionId ...)`
- `if (debugFlag ...)`

### 3.7 legacy coexistence 放置

禁止:

- temporary の無期限存続

---

## 4. 実装前必須手順

コード変更前に以下を列挙すること。

1. authority source
2. observe path
3. publication path
4. generation source
5. retire ownership

### 4.1 必須提出物（機械可読）

- 変更前: `Current Authority Inventory`
- 変更後: `Post-Migration Authority Inventory`

### 4.2 Inventory 必須項目

- state
- authority_class
- owner
- readers
- writers
- thread_domain
- publication_path
- observe_path
- retirement_owner

---

## 5. 変更提案時の必須分析

変更前に `authority impact analysis` を提示すること。

### 5.1 authority 増減

- authority source 増加有無
- observe path 増加有無
- semantic source 増加有無

### 5.2 migration safety

- dual authority 発生期間
- coexistence 期間
- rollback safety

### 5.3 runtime safety

- stale observe risk
- partial publish risk
- overlap risk
- retire pressure risk

---

## 6. 実装時必須規約

### Rule-1 RuntimeWorld immutable rule

publish 後 mutation 禁止。

### Rule-2 RuntimeGeneration rule

唯一 authoritative generation は `RuntimeGeneration` のみ。

### Rule-3 ordering logic confinement

generation ordering logic は `RuntimeCoordinator` 内限定。

### Rule-4 single publication rule

唯一許可される publication は `publish(RuntimeWorld*)`。

### Rule-5 observe singularization

Audio Thread observe は `RuntimeWorld only` へ収束。

### Rule-6 executor-local isolation

executor-local state は以下を禁止:

- publish
- atomic export
- cross-thread observe

### Rule-7 overlap semantics prohibition

overlap 時に許可される挙動は次のみ:

- reject
- coalesce
- restart

semantic merge は禁止。

### Rule-8 fail-closed mandatory

verify fail は warning 禁止、build fail 必須。

---

## 7. Allowed Recovery Paths（許可修正経路）

### 7.1 overlap 発生時

許可:

- reject
- coalesce
- restart

禁止:

- semantic merge
- hidden retry state

### 7.2 retire pressure 時

許可:

- throttle
- defer
- non-RT drain
- rebuild coalescing

禁止:

- silent drop
- hidden queue expansion

### 7.3 generation drift 時

許可:

- authoritative source への収束

禁止:

- temporary synchronization layer
- compatibility shadow state

---

## 8. AuthorityClass 強制規約

runtime-related state はすべて `AuthorityClass` 注釈必須。

- 未分類 state 禁止

例:

`ISR_FIELD(AuthorityClass::Authoritative, RuntimeGeneration, generation);`

---

## 9. LegacyTemporary 規約

`LegacyTemporary` は存在を許可するが増殖は禁止。

### 9.1 必須属性

全 legacy に以下を必須化:

- owner
- replacement_authority
- removal_phase
- deadline
- scope

### 9.2 必須 manifest

- `.github/isr-legacy-temporary.json` 更新必須
- manifest 未登録 legacy は禁止（CI fail）

---

## 10. Phase Governance / Override Clause

### 10.1 原則

phase 越境実装は禁止。

### 10.2 例外（限定許可）

`BreakGlassOverride` 経由のみ phase override を許可。

### 10.3 必須条件

- expiration mandatory
- rollback_plan mandatory
- CI warning mandatory
- release branch approval mandatory
- soak mandatory
- deadline expiration → CI fail

---

## 11. Break-glass 規約

監査不能な temporary bypass を禁止し、`BreakGlassOverride` のみ許可。

### 11.1 必須項目

- expiration
- owner
- reason
- rollback_plan

### 11.2 禁止

- 永続 override
- release persistent override
- undocumented suppression

---

## 12. Verification Matrix Rule

runtime 変更時は `verification impact` 分析を必須化し、変更種別ごとの mandatory verification matrix を定義・更新する。

### 12.1 必須CI更新対象（基底セット / 実スクリプト名）

runtime 変更時は最低限、以下の verifier を変更影響に応じて必須実行対象に含めること。

- `.github/scripts/isr-run-tiered-verification.ps1`（統合ゲート）
- `.github/scripts/isr-verify-v1-immutability.ps1`（immutable / publication 基本検証）
- `.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1`（runtime graph immutable 検証）
- `.github/scripts/isr-verify-v4.ps1`（publication/構造系検証）
- `.github/scripts/isr-verify-v5-retire-authority-lane.ps1`（retire authority lane 検証）
- `.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1`（RT/NonRT retire bridge 検証）
- `.github/scripts/isr-verify-phase4-generation-drift.ps1`（generation drift 検証）
- `.github/scripts/isr-verify-crossfade-observable-state.ps1`（crossfade observable state 検証）
- `.github/scripts/isr-verify-observe-shim-usage.ps1`（observe path 拡張/迂回検証）
- `.github/scripts/isr-verify-v73-residency-telemetry.ps1`（residency/retention telemetry 検証）
- `.github/scripts/isr-verify-v73-shutdown-reclaim.ps1`（shutdown reclaim / leak 系検証）
- `.github/scripts/isr-verify-rtmutable-boundary.ps1`（RT mutable boundary 検証）

### 12.2 変更種別 × 必須スクリプト（mandatory verification matrix）

| 変更種別 | 必須スクリプト（最低） | 目的 |
| --- | --- | --- |
| RuntimeWorld 構造/公開パス変更 | `isr-run-tiered-verification.ps1` / `isr-verify-v1-immutability.ps1` / `isr-verify-v3-runtime-graph-immutability.ps1` / `isr-verify-v4.ps1` / `isr-verify-observe-shim-usage.ps1` / `isr-verify-phase4-generation-drift.ps1` | partial publication・mixed generation・observe path 拡張の検知 |
| retire 系変更（lane/bridge/reclaim） | `isr-run-tiered-verification.ps1` / `isr-verify-v5-retire-authority-lane.ps1` / `isr-verify-v7-rt-nonrt-retire-bridge.ps1` / `isr-verify-v73-shutdown-reclaim.ps1` / `isr-verify-v73-residency-telemetry.ps1` | retire backlog slope・retention leak・RT/NonRT bridge 破綻の検知 |
| crossfade / overlap 変更 | `isr-run-tiered-verification.ps1` / `isr-verify-crossfade-observable-state.ps1` / `isr-verify-observe-shim-usage.ps1` / `isr-verify-rtmutable-boundary.ps1` | overlap semantics 逸脱・executor-local leakage・observe 逆流の検知 |
| generation / ordering 変更 | `isr-run-tiered-verification.ps1` / `isr-verify-phase4-generation-drift.ps1` / `isr-verify-v1-immutability.ps1` | generation drift・non-authoritative branch 逆流の検知 |
| admission / rebuild funnel 変更 | `isr-run-tiered-verification.ps1` / `isr-verify-v73-admission-funnel.ps1` / `isr-verify-latency-alignment.ps1` / `isr-verify-v73-residency-telemetry.ps1` | rebuild admission 逸脱・latency 悪化・residency 異常の検知 |
| workflow / policy / gate 変更 | `isr-run-tiered-verification.ps1` / `isr-verify-gate-wiring.ps1` / `isr-verify-validator-tiering.ps1` / `isr-verify-workflow-dispatch-input-policy.ps1` / `isr-verify-policy-top-level-governance.ps1` | CI配線不整合・policy bypass・tiering 破綻の検知 |

### 12.3 Tier 1対1対応の運用実行表（`isr-run-tiered-verification.ps1` 準拠）

`isr-run-tiered-verification.ps1` の Tier は `basic/strict/exhaustive` ではなく、**`smoke/standard/exhaustive`** である。
本規約の運用実行表は、同スクリプト内の配列定義に **1対1対応** させる。

| Tier | 実行スクリプト集合（1対1対応） | 運用位置づけ |
| --- | --- | --- |
| smoke | `isr-verify-v1-immutability.ps1` / `isr-verify-v2-seal.ps1` / `isr-verify-v3-runtime-graph-immutability.ps1` / `isr-verify-v4-dsp-handle-policy.ps1` / `isr-verify-v5-retire-authority-lane.ps1` / `isr-verify-v6-domain-f-ordering.ps1` / `isr-verify-v7-rt-nonrt-retire-bridge.ps1` / `isr-verify-v8-shared-split-readiness.ps1` / `isr-verify-phase4-generation-drift.ps1` / `isr-verify-v6.ps1` / `isr-verify-workflow-dispatch-input-policy.ps1` / `isr-verify-gate-wiring.ps1` | 最短ゲート（最低限の不変条件・配線・橋渡し検証） |
| standard | **smoke 全件** + `isr-verify-v3.ps1` / `isr-verify-v4.ps1` / `isr-verify-v5.ps1` / `isr-verify-v7.ps1` / `isr-verify-v8.ps1` / `isr-verify-v9.ps1` / `isr-verify-v10.ps1` / `isr-verify-v10-ownership-cycle.ps1` / `isr-verify-evidence-provenance.ps1` / `isr-verify-runtime-reduction-gate.ps1` / `isr-verify-proof-scope.ps1` / `isr-verify-r11-r25-closed-coverage.ps1` / `isr-verify-drained-resurrection-guard.ps1` / `isr-verify-trigger-policy.ps1` / `isr-verify-trigger-symbol-usage.ps1` / `isr-verify-observe-shim-usage.ps1` / `isr-verify-trigger-ast.ps1` / `isr-trigger-audit.ps1` / `isr-prune-cleanup-deferred.ps1` / `isr-rebuild-admission-8_1-metrics.ps1` / `isr-verify-enforcement-adoption.ps1` / `isr-verify-enforcement-source-purity.ps1` / `isr-verify-trigger-cleanup-readiness.ps1` / `isr-verify-cleanup-deferred.ps1` / `isr-verify-flag-dependency-graph.ps1` / `isr-verify-rollback-matrix.ps1` / `isr-verify-metric-governance.ps1` / `isr-verify-8_1-close-policy.ps1` / `isr-verify-8_1-workflow-input-contract.ps1` / `isr-verify-8_1-workflow-input-coherence.ps1` / `isr-verify-policy-top-level-governance.ps1` / `isr-verify-rtmutable-boundary.ps1` / `isr-verify-facade-bypass.ps1` / `isr-verify-latency-alignment.ps1` / `isr-verify-crossfade-observable-state.ps1` / `isr-verify-canary-baseline-normalization.ps1` / `isr-verify-ownership-migration.ps1` / `isr-verify-validator-tiering.ps1` / `isr-verify-trigger-cleanup-completion.ps1` / `isr-verify-backlog-specfixed-residual.ps1` / `isr-verify-bridge-plan-completeness.ps1` / `isr-verify-clang-tidy-readiness.ps1` / `isr-verify-clang-tidy-audit.ps1` / `isr-verify-v73-admission-funnel.ps1` / `isr-verify-v73-shutdown-reclaim.ps1` / `isr-verify-v73-residency-telemetry.ps1` / `check-src-atomic-dotcall.ps1` / `check-list-compliance.ps1` / `isr-verify-p3-governance.ps1` | 通常運用ゲート（policy/trigger/evidence/latency/v7.3 を包含） |
| exhaustive | **standard 全件** + `isr-verify-v5.ps1` / `isr-verify-v6.ps1` / `isr-verify-v7.ps1` / `isr-verify-v8.ps1` / `isr-verify-v9.ps1` | 最終運用ゲート（追加再実行で収束確認） |

補足:

- `exhaustive` では `v5/v6/v7/v8/v9` が **再実行** される（スクリプト定義どおり）。
- `standard`/`exhaustive` では `isr-verify-v73-admission-funnel.ps1` / `isr-verify-v73-shutdown-reclaim.ps1` / `isr-verify-v73-residency-telemetry.ps1` の3本を必須とする配線検証が内蔵されている。
- Tier 実行の起点は常に `isr-run-tiered-verification.ps1 -Tier <smoke|standard|exhaustive>` とし、個別実行のみで代替してはならない。

### 12.4 禁止

runtime 変更のみ実施し、CI未更新で終了すること。

---

## 13. Project Hard Constraints Clause（ConvoPeq 固有）

### 13.1 Vendor Source Rule

以下編集禁止:

- `JUCE/`
- `r8brain-free-src/`

例外は `BreakGlassOverride + explicit approval` のみ。

### 13.2 Audio Thread Hard Prohibition

Audio Thread で以下を禁止:

- allocation / deallocation
- lock
- blocking wait
- file IO
- MessageManager access
- sleep
- `condition_variable` wait
- SEH
- exception propagation
- uncontrolled libm usage

### 13.3 SIMD / oneMKL Rule

oneMKL / SIMD 用 memory は non-RT 64-byte aligned allocation mandatory。

### 13.4 Exception Rule

ISR Runtime path は exception-free mandatory。

---

## 14. State Addition Exception Rule

新 state 追加は原則禁止。

### 14.1 例外許可（全条件必須）

- BreakGlassOverride
- deadline
- removal_phase
- CI guard
- soak validation

### 14.2 必須分類

temporary state は `LegacyTemporary` 分類必須。

---

## 15. Soak Test 規約

runtime 変更後は最低限次の影響分析を必須化:

- rebuild storm
- overlap storm
- automation storm
- suspend/resume storm

### 15.1 必須分析項目（非悪化）

- XRUN risk
- stale observe risk
- retire backlog growth
- world retention leak
- authority duplication regression

---

## 16. Runtime Safety Regression Rule

`authority削減成功` のみで成功判定してはならない。以下の非悪化を保証すること。

- XRUN
- suspend/resume
- rebuild latency
- publication latency
- retire backlog slope
- leak count
- stale observe count

---

## 17. Documentation Scope Rule

runtime semantics を変更した場合、最低限次を更新必須とする。

- `doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md`
- topology 差分文書
- authority inventory
- `.github/isr-legacy-temporary.json`
- verification matrix

コードのみ変更は禁止。

---

## 18. レビュー時必須観点

レビュー時は `runtime meaning duplication` を最優先確認。

- authority duplication
- observe duplication
- generation drift
- stale observe
- partial publish
- overlap semantics
- retire pressure divergence
- legacy lifetime drift

---

## 19. 最終原則

AI は常に次を最優先に自問すること。

> この変更は実運用安全性を維持したまま authority を減らすか？

最も危険なのは temporary coexistence の恒久化である。これを絶対に回避すること。
