# ConvoPeq ISR Bridge Runtime AI詳細設計・実装統制規約書 v7.3

## 0. 目的

本規約は、ConvoPeq ISR Bridge Runtime の詳細設計・実装を AI に実施させる際に、次を強制するための統制文書である。

- 実運用で破綻しにくい runtime survivability の維持
- long-session DAW stability の優先
- shutdown determinism の維持
- bounded residency / bounded reclaim の維持
- authority ambiguity の排除
- AI による過剰抽象化・過剰純化・全面 redesign の防止

本規約は ISR purity の理論完成を目的としない。優先順位は以下とする。

1. Operational Safety
2. Runtime Survivability
3. Shutdown Completion
4. Bounded Residency
5. Deterministic Reclaim
6. Architectural Purity

## Rule-0: 最上位原則

### Rule-0A

AI は ISR purity を理由に既存 hardening を破壊してはならない。

禁止対象:

- 全面 lock-free rewrite
- epoch system 再実装
- runtime graph 全面再設計
- shared_ptr 全面導入
- DSP immutable 化の全面強制
- publication coordinator rewrite
- crossfade architecture 全面変更

### Rule-0B

Tier0 の目的は「死なない runtime」を完成させることである。

Tier0 の最優先:

- runaway suppression
- bounded reclaim
- deterministic shutdown
- rebuild storm suppression
- residency convergence
- unload safety

### Rule-0C

AI は authority を増やしてはならない。

追加禁止:

- 第2 admission authority
- 第2 drain authority
- 第2 residency authority
- subsystem-local reclaim policy
- subsystem-local saturation semantics

## Rule-1: Rebuild Admission Governance

### Rule-1A

全 rebuild 要求は必ず single admission funnel を通過しなければならない。

唯一の外部入口:

```cpp
submitRebuildIntent(...)
```

### Rule-1B

以下からの direct rebuild 実行は禁止する。

禁止対象:

- Timer callback
- UI callback
- helper utility
- test helper
- legacy compatibility path
- worker helper
- background task

禁止:

```cpp
requestRebuild(...)
```

の直呼び。

### Rule-1C

`requestRebuild(...)` は public rebuild API ではない。

定義:

- internal execution primitive
- funnel 内部専用
- external caller 禁止

### Rule-1D

CI/Lint は direct rebuild 呼び出しを機械検出しなければならない。

許可箇所:

- definition site（関数定義箇所）
- allowlisted funnel implementation

それ以外は CI fail。

### Rule-1E

Tier0 では rebuild execution semantic を増やしてはならない。

許可:

```cpp
Replaceable
MustExecute
```

禁止:

- urgent
- critical
- alwaysRun
- highPriority
- realtimeCritical
- lowPriorityReplaceable

### Rule-1F

互換 shim は一時的に許可される。

許可:

- forceMustExecute
- bridge bool
- alias enum
- temporary adapter

条件:

- semantic 拡張禁止
- 最終収束先は RebuildExecutionClass
- Tier1 完了までに整理

型対応の固定:

```text
Current implementation type: RebuildTelemetryPolicy
Canonical governance semantic: RebuildExecutionClass
```

### Rule-1G

saturation suppression は rebuild queue 追加前に実施しなければならない。

禁止:

- enqueue 後 suppression
- deferred suppression
- telemetry-only suppression

### Rule-1H

saturation suppression は Replaceable のみ抑止可能。MustExecute は reject 不可。

### Rule-1I

latest-generation-wins collapse は維持必須。AI は obsolete collapse を削除してはならない。

### Rule-1J

saturation enter 条件は hysteresis 必須。

最低条件:

```cpp
retireQueueDepth_ >= retireHighWatermark_
```

### Rule-1K

saturation exit 条件は hysteresis 必須。

最低条件:

```cpp
retireQueueDepth_ <= retireLowWatermark_
```

追加制約:

- `retireHighWatermark_ > retireLowWatermark_` を常時維持
- enter/exit flapping 禁止

### Rule-1L

suppression reason は telemetry 可観測でなければならない。

最低記録項目:

- saturation reject count
- Replaceable suppress count
- MustExecute bypass count
- duplicate collapse count

### Rule-1M（追加）

suppression authority は admission funnel に限定する。

```cpp
Only admission funnel may suppress rebuild intents.
No subsystem-local suppression allowed.
```

禁止対象:

- worker-local suppression
- timer-local suppression
- reclaim-thread suppression
- telemetry-triggered local suppression

### Rule-1N（追加）

suppression 定義域を分離しなければならない。

定義:

- `RebuildIntentSuppression`（admission funnel 専属）
- `QueueAdmissionSuppression`（queue/backlog 保護）
- `SnapshotDropSuppression`（snapshot/command buffer 保護）

拘束:

- Rule-1M は `RebuildIntentSuppression` に限定適用する。
- `QueueAdmissionSuppression` / `SnapshotDropSuppression` を Rule-1M 違反として誤検知してはならない。

### Rule-1O（追加）

`requestRebuild(...)` 直呼び封鎖は mechanical enforcement を必須とする。

必須:

- allowlist
- grep/lint
- CI fail
- PR block

移行期間は段階適用を許可する。

- Phase-A: warn
- Phase-B: fail

allowlist には `owner/issue/rationale/expiry` を必須化し、expiry 超過は fail-closed とする。

## Rule-2: Residency Governance

### Rule-2A

residency は bounded でなければならない。

禁止:

- 永続増加 residency
- reclaim 不可能 residency
- unbounded queue growth

### Rule-2B

禁止対象は unbounded / permanent append-only residency である。

禁止:

- drain condition のない append-only queue
- reclaim authority 不明 residency
- 永続蓄積 vector

許可:

- bounded staging buffer
- bounded append queue
- short-lived quarantine

### Rule-2C

全 residency は owner を持たなければならない。

| Residency | Drain Authority | Owner |
| --- | --- | --- |
| retire queue | coordinator | coordinator |
| fallback queue | coordinator | coordinator |
| deferred retire | coordinator | coordinator |
| epoch retire staging | coordinator | coordinator |

### Rule-2D

residency authority は enum / contract としてコード化しなければならない。

禁止:

- implicit ownership
- comment-only ownership
- convention-only ownership

### Rule-2E

AI は subsystem-local reclaim policy を追加してはならない。reclaim authority は global coordinator に集約する。

### Rule-2F（追加）

residency の生成元（producer）を固定する。

| Residency | Producer | Owner | Reclaim Trigger |
| --- | --- | --- | --- |
| retire queue | publication coordinator | coordinator | reclaim scheduler |
| fallback queue | publication bridge | coordinator | drain scheduler |
| deferred retire | epoch retire path | coordinator | epoch advance |

### Rule-2G（追加）

全 residency は boundedness の定量契約を持たなければならない。

必須:

- hard upper bound
- warn threshold
- force-drain trigger

未定量 residency は Tier0 受入不可とする。

## Rule-3: Drain / Shutdown Governance

### Rule-3A

`waitForDrain(timeout)` は non-RT only。

意味論:

- bounded wait
- best-effort convergence wait
- timeout 到達で false
- full quiescence 保証ではない

### Rule-3B

`waitForDrain()` は shutdown phase machine を迂回してはならない。

禁止:

- immediate unload guarantee
- direct destruction guarantee
- forced synchronous reclaim

### Rule-3C

drain completion authority は単一 source-of-truth に固定する。

唯一の authority:

```cpp
coordinator.isFullyDrained()
```

### Rule-3D

queue 個別状態を drain 完了条件として直接使用してはならない。

禁止:

- queue empty 単独判定
- backlog counter 単独判定
- telemetry counter 判定

### Rule-3E

`coordinator.isFullyDrained()` は最低限以下を内部集約対象とする。

- publication backlog
- pending publication intents
- retire residency
- fallback residency
- reclaim in-flight state
- publication coordinator staging

### Rule-3F

debug/release の挙動は分離する。

debug:

- assert 許可

release:

- silent fail
- false return
- RT blocking 禁止

### Rule-3G（追加）

shutdown phase bypass を禁止する。

禁止:

- direct reclaim during unload
- direct queue clear
- direct coordinator reset
- force-retire without shutdown phase gate

`waitForDrain()` 成功後でも phase machine を経由しない解放を禁止する。

## Rule-4: Reclaim Governance

### Rule-4A

reclaim は bounded cadence でなければならない。

必須:

- max reclaim iterations per cycle
- min reclaim interval
- reclaim pressure upper bound

### Rule-4B

emergency reclaim は blocking reclaim を意味しない。

禁止:

- synchronous drain completion
- RT wait
- forced full reclaim

許可:

- reclaim cadence boost
- obsolete retire prioritization
- fallback aggressive drain

### Rule-4C

subsystem 個別 reclaim optimizer を禁止する。reclaim policy は global authority が管理する。

### Rule-4D

reclaim pressure authority は単一 coordinator に固定する。

禁止:

- local reclaim escalator
- local reclaim budget
- independent reclaim thread semantics

### Rule-4E（追加）

reclaim starvation 防止を必須化する。

必須:

- reclaim starvation timeout
- reclaim retry escalation ceiling
- obsolete residency prioritization

## Rule-5: Snapshot / Seal Governance

### Rule-5A

RuntimeBuildSnapshot は finalize 後 immutable 扱いとする。

### Rule-5B

seal 前 commit を禁止する。

禁止:

```cpp
unsealed snapshot publish
```

### Rule-5C

capture → finalize → seal → publish の順序を破壊してはならない。

### Rule-5D

AI は mutable bridge state を拡張してはならない。crossfade mutable state は Tier2 まで縮退対象とする。

## Rule-6: Telemetry Governance

### Rule-6A

telemetry は operational visibility 目的である。telemetry は authority ではない。

### Rule-6B

全 telemetry counter は ownership を持たなければならない。

最低定義:

- owner subsystem
- increment condition
- decrement/reset condition
- shutdown finalization behavior

### Rule-6C

telemetry を lifecycle 判定 authority に使用してはならない。

禁止:

- counter-only shutdown判定
- counter-only reclaim completion
- counter-only drain completion

### Rule-6D

counter finalization は counter ごとに固定する。

許可:

- freeze final value
- export snapshot
- explicit reset

### Rule-6E（追加）

telemetry write authority を固定する。

```cpp
Telemetry counters may only be mutated by their declared owner subsystem.
```

禁止:

- cross-subsystem increment
- helper-side counter mutation
- debug utility mutation

## Rule-7: Crossfade / Mutable Runtime Governance

### Rule-7A

Tier0/Tier1 では crossfade purity を優先しない。

優先順位:

1. saturation safety
2. residency convergence
3. deterministic shutdown
4. reclaim stability
5. crossfade purity

### Rule-7B

AI は RT mutable state を増やしてはならない。

### Rule-7C

crossfade bridge mutable は縮小のみ許可。拡張禁止。

## Rule-8: Tier Enforcement

### Rule-8A

Tier0 完了前に Tier2/3 redesign を開始してはならない。

禁止:

- graph redesign
- DSP ownership redesign
- publication architecture rewrite
- immutable DSP migration

### Rule-8B

例外は以下のみ許可。

- Tier0 blocker fix
- crash fix
- unload failure fix
- RT safety regression fix

条件:

- minimum diff
- bounded scope
- no architectural expansion
- no semantic broadening

### Rule-8C

AI は「きれいだから」という理由で設計拡張してはならない。

必要条件:

- operational risk reduction
- measurable survivability gain
- bounded implementation cost

## Rule-OPS: Operational Hardening Rules

### OPS-1

single admission authority 必須。

### OPS-2

single drain authority 必須。

### OPS-3

single reclaim authority 必須。

### OPS-4

single residency authority 必須。

### OPS-5

authority bypass は CI fail。

### OPS-6

suppression reason table を固定する。

| Reason | Suppressible |
| --- | --- |
| saturation | Replaceable only |
| duplicate collapse | Replaceable only |
| obsolete generation | Replaceable only |
| shutdown phase | all except MustExecute shutdown-safe path |
| invalid state | policy-defined |

Tier0 で許可される reason は `Saturation / Duplicate / Shutdown / Obsolete / InvalidState` のみとし、
新規 reason は Tier1 review mandatory とする。

### OPS-7

AI は authority ambiguity を追加してはならない。

## Rule-CI: 機械検証

### CI-1

`requestRebuild(` の direct usage を grep/lint 検出する。

```text
Definition sites excluded.
Allowlisted funnel implementation excluded.
All other direct invocations fail CI.
```

### CI-2

allowlist 外 usage は build fail。

### CI-3

新 execution semantic 検出時は build fail。

### CI-4

新 residency queue 検出時は review mandatory。

### CI-5

new mutable RT state は review mandatory。

## 最終原則

ConvoPeq ISR Bridge Runtime の Tier0/Tier1 では、

- purity
- abstraction beauty
- architectural elegance

よりも、

- bounded behavior
- deterministic shutdown
- runaway suppression
- survivability
- operational safety

を優先する。

AI は「きれいにする」ことよりも、「壊れにくくする」ことを優先しなければならない。
