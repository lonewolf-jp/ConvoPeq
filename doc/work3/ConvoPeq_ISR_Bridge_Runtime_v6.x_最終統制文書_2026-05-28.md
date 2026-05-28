# ConvoPeq ISR Bridge Runtime v6.x 最終統制文書

- Project: ConvoPeq
- Date: 2026-05-28
- Status: Final Control Document
- Scope: AI に詳細設計・実装をさせる際の最終統制文書
- Objective: **理論的完全性ではなく、DAW 実運用で破綻しにくい ISR Bridge Runtime を段階的 hardening すること**

---

## 0. 文書の役割

本書は、ConvoPeq における ISR Bridge Runtime 改修の**最終統制文書**である。

本書は次の3層を統合する。

1. **基本設計**: operational survivability を優先する設計方針
2. **実装規約**: AI が守るべき禁止事項と必須条件
3. **最終追補**: AI の誤読・暴走・過剰実装を抑止する補足固定

本書の目的は、ISR purity を理論的に極めることではない。
目的は、**長時間 DAW 運用で破綻しにくい runtime を完成させること**である。

---

## 1. 文書の優先順位

本書内で矛盾がある場合、優先順位は次の通りとする。

1. Final Clarification Addendum
2. Final Operational Hardening Addendum
3. 基本設計本文

より具体的には、AI 実装時には「より厳しいほう」ではなく、**より実運用に整合するほう**を採用する。

---

## 2. 現状認識

現時点の ConvoPeq は、既に次の状態に到達している。

- RuntimeBuildSnapshot 化
- Unified Admission Gate
- capture → finalize → seal
- publication validation
- shutdown phase control
- obsolete rebuild collapse
- bounded RT critical sections
- partial execution-local mutability

したがって、今必要なのは ISR purity の理論追求ではなく、**operational survivability の完成**である。

---

## 3. 設計の最上位原則

### 3.1 優先軸

優先順位は次の順序とする。

1. crash 防止
2. runaway 防止
3. unload / shutdown 完了性
4. bounded memory
5. long-session stability
6. RT safety
7. ISR purity

ISR purity を理由に既存安定動作を壊してはならない。

### 3.2 設計姿勢

- 既存 hardening は再実装しない
- 足りない operational guard のみを追加する
- 入口で抑止し、後段で救済しない
- 測定だけで終わらせず flow control に接続する
- 理論的に美しいからという理由で redesign しない

### 3.3 禁止する方向

- 全面 rewrite
- 全面 lock-free 化
- DSP 全 immutable 化
- runtime graph の全面再設計
- epoch system の再実装
- shared_ptr の pervasive 化
- publication coordinator の全面 rewrite

---

## 4. 既存骨格の扱い

ConvoPeq の ISR Bridge Runtime は、次の骨格を維持する。

- `AudioEngine` を中心とした admission / publication / shutdown / rebuild dispatch
- `RuntimePublicationCoordinator` による publication world 管理
- `RuntimeBuildSnapshot` による worker 入力の固定
- `ShutdownRuntime` による phase control
- `RetireRuntime` / `RetireRuntimeEx` による retire / reclaim 分離

既存 hardening を壊さず、そこに operational guard を追加する。

---

## 5. スレッド責務の固定

- Audio Thread: consume-only
- Message Thread: admission / publication / telemetry / lifecycle control
- Rebuild Worker: sealed snapshot ベース再構築
- Retire / Reclaim 処理: non-RT 側で best effort に実行

実行ローカル mutable は許容するが、runtime authority を持たせてはならない。

許容される例:

- local fade progression
- thread-local scratch
- smoothing accumulators
- readonly cache materialization

---

## 6. Unified Admission Gate

`acceptsRuntimePublication()` を publication 経路の全入口に適用する。

適用対象の代表例:

- `requestRebuild`
- `submitRebuildIntent`
- `prepareCommit`
- `executeCommit`
- `appendPublicationIntent`

shutdown 受理フェーズ以降は publication intent を受け付けない。

```cpp
if (!acceptsRuntimePublication())
{
    return;
}
```

---

## 7. Snapshot Seal Contract

snapshot lifecycle は次の順序に固定する。

```text
capture -> finalize -> seal -> handoff
```

### 7.1 finalize の意味

finalize は deterministic normalization である。

許可:

- canonical ordering
- parameter normalization
- immutable fingerprint generation
- bounded validation

禁止:

- runtime state injection
- wall clock / timing / thread-order 依存
- allocation order 依存
- pointer identity 依存
- environment-derived mutation
- mutable patch injection
- async augmentation

### 7.2 seal の意味

seal は immutable 契約である。

seal 後の mutation を禁止する。
debug build では assert を推奨する。

### 7.3 worker 側契約

worker は sealed snapshot のみを受け取る。

禁止:

- active runtime pointer の参照
- fading runtime pointer の参照
- old runtime traversal
- runtime pointer 由来の補正

---

## 8. Tier0: Saturation / Residency / Shutdown / Reclaim

### 8.1 Saturation Admission Suppression

retire saturation 発生時に rebuild inflow を止める。

**入口で止める。queue に積んでから止めない。**

許可:

- Replaceable rebuild の reject
- MustExecute rebuild の継続
- collapse / coalescing の維持

#### 入口方針

```cpp
if (retireSaturationActive_ && intent.policy == Replaceable)
    return Reject;
```

#### AI 実装時の禁止

- queue.push(task); 後に reject する実装
- saturation を「観測だけ」して抑止しない実装
- saturation 判定を RT path に持ち込む実装

### 8.2 Rebuild Execution Class 収束

rebuild execution semantics は最終的に単一概念へ収束させる。

正式 canonical source は次のみとする。

- `RebuildExecutionClass::Replaceable`
- `RebuildExecutionClass::MustExecute`

移行途中の compatibility shim は一時許可する。

許可される例:

- `forceMustExecute`
- bridge bool
- legacy helper
- alias semantic

ただし、以下は禁止する。

- 新しい execution semantic の追加
- 第3 execution class の導入
- bool と enum の意味乖離

AI が compatibility shim を追加する場合は、temporary / bounded scope / migration-only を明示し、恒久 API 化してはならない。

### 8.3 Latest-Generation Collapse

obsolete rebuild は以下の各段階で collapse 可能とする。

- enqueue 前
- worker 実行前
- publication 前

latest-generation-wins を維持し、safe-to-collapse でないものは collapse しない。

### 8.4 Residency Accounting の一本化

`retire queue` / `fallback queue` / `retiredWorlds_` / `quarantine` を分散管理したままにしない。

禁止対象は次である。

- unbounded residency
- permanent append-only residency
- drain condition を持たない residency

許可されるのは次である。

- bounded append buffer
- short-lived staging queue
- bounded temporary residency
- deterministic reclaim を伴う append structure

AI は residency container を追加する場合、次を必ず定義しなければならない。

- boundedness
- reclaim owner
- drain condition
- shutdown completion path

巨大な新 Manager を追加してはならない。
許可されるのは bounded counters / accounting hooks / drain visibility のみである。

### 8.5 `waitForDrain(timeout)`

`waitForDrain(timeout)` は non-RT 限定で導入する。

#### 動作意味論

- convergence attempt
- bounded operational wait
- best-effort synchronization

成功保証 API ではない。

```text
drained converged -> true
timeout reached    -> false
infinite wait      -> prohibited
```

#### 実装条件

- non-RT only
- bounded wait
- timeout 必須
- debug build では assert 可
- release build では silent fail + false return を優先

#### AI に対する禁止

- retry-until-success loop
- unbounded blocking
- RT-assisted waiting
- RT thread / audio callback からの呼び出し

### 8.6 Drain Completion Single Source Of Truth

shutdown completion / drain completion 判定は `coordinator.isFullyDrained()` に単一集約する。

個別 queue ごとの独自 drained 判定や subsystem ごとの独立 completion 判定を追加してはならない。

`coordinator.isFullyDrained()` は少なくとも次を内部集約対象とする。

- publication backlog
- pending publication intents
- retire residency
- fallback residency
- reclaim in-flight operations
- publication coordinator staging residency
- deferred retire residency
- publication swap pending state

AI は、これらを bypass する独自 drained 判定を追加してはならない。
telemetry counter を drained source-of-truth に使用してはならない。

### 8.7 Emergency Reclaim

emergency reclaim は best-effort 非同期でなければならない。

禁止:

- RT reclaim
- blocking reclaim
- synchronous full drain

許可:

- reclaim cadence boost
- aggressive fallback drain
- obsolete retire prioritization

飽和時の回収方針は、古い obsolete retire 優先を原則とする。

### 8.8 Retire Backpressure の基本値

- `HWM = 3072`
- `LWM = 1024`
- `highWatermark > lowWatermark` を常に保証する

saturation 状態では安全側方向のみを許可する。

---

## 9. Tier1: Quarantine / Prioritization / Tuning

### 9.1 Quarantine Telemetry 実体化

quarantine を単なる名前にせず、残留量を観測可能にする。

### 9.2 Reclaim Prioritization

飽和時は、古い / obsolete なものを優先回収する。

### 9.3 Saturation Policy Tuning

HWM / LWM の関係は維持しつつ、saturation 時の安全側 tuning のみを許可する。

---

## 10. Tier2: Crossfade / Seal

### 10.1 Crossfade Mutable Reduction

`latencyDelayOld_RT` / `latencyDelayNew_RT` / `dspCrossfade*` などのグローバル mutable を段階的に減らす。

ただし、これは Tier0 の安全性確保後に進める。

### 10.2 Execution-Local 化

crossfade purity は Tier2 扱いとする。

AI は saturation / residency / shutdown より優先して crossfade purity を触ってはならない。

許可されるのは次のみ。

- mutable reduction
- authority narrowing
- execution-local snapshotization

全面 rewrite は禁止する。

### 10.3 Deeper Seal Propagation

seal / finalize の責務を必要最小限で拡張する。
DSP 全 immutable 化は行わない。

---

## 11. Telemetry / Debug / Finalization

### 11.1 Telemetry Ownership

各 telemetry counter は必ず次を定義する。

- owner subsystem
- increment condition
- decrement / reset condition
- shutdown finalization behavior

複数 subsystem が同一 counter を更新してはならない。
必要な場合は aggregator / snapshot export / derived metric として分離する。

### 11.2 Telemetry の役割

telemetry は operational visibility であり、source-of-truth ではない。

以下は禁止:

- lifecycle 判定
- shutdown completion 判定
- reclaim safety 判定

を telemetry のみで行うこと。

### 11.3 Finalization Policy

shutdown finalization 時、各 telemetry counter は次のいずれかを明示的に採る。

- freeze final value
- export snapshot
- explicit reset

暗黙 reset は禁止する。
counter semantic を shutdown 中に変更してはならない。

---

## 12. AI 実装統制の禁止事項

AI は以下を禁止する。

- “理論的に正しい” を理由にした subsystem rewrite
- telemetry を lifecycle authority に昇格させること
- append-only residency の増設
- shutdown completion source の分散
- reclaim ownership の再分散
- queue topology の再発明
- Tier0 未完了状態での purity expansion
- manager explosion
- coordinator layering
- dispatcher nesting

AI は以下を優先する。

- incremental hardening
- bounded behavior
- deterministic teardown
- operational stability
- survivability under automation storm

---

## 13. Tier Execution Policy

原則として、Tier0 完了前に Tier2 / Tier3 の拡張へ進んではならない。

ただし、以下のみ例外として許可する。

- Tier0 blocker fix
- crash fix
- unload failure fix
- RT safety regression fix

例外修正は次を必須とする。

- minimum diff
- bounded scope
- no architectural expansion
- no semantic broadening

---

## 14. 受入判定

次を満たしたときのみ、v6.x 最終統制文書に適合した改修とみなす。

### 14.1 Saturation

- rebuild storm で runaway しない
- Replaceable reject が機能
- MustExecute starvation がない

### 14.2 Residency

- resident count が bounded
- retire / fallback が drain 可能
- unbounded append-only residency が存在しない

### 14.3 Shutdown

- `waitForDrain(timeout)` が bounded best-effort で動作する
- unload race が発生しない
- resurrection が発生しない
- drain completion は `coordinator.isFullyDrained()` に集約される

### 14.4 RT Safety

- RT allocation 増加なし
- RT lock 導入なし
- RT assisted waiting 導入なし

### 14.5 Operational

- long-session DAW 運用でメモリ増加が収束する
- IR spam / automation spam で崩壊しない
- reclaim convergence が維持される

---

## 15. Final Engineering Principle

ConvoPeq ISR Bridge Runtime は、次の runtime として hardening を進める。

- ISR purity を完成させる runtime ではない
- DAW 長時間運用で破綻しない operational runtime である

AI は purity optimization より、次を優先する。

- survivability
- boundedness
- convergence
- operational determinism
- unload safety

したがって本書の最終原則は次である。

```text
survivability
    >
boundedness
    >
determinism
    >
RT safety
    >
architectural purity
```

---

## 16. 結論

ConvoPeq に必要なのは ISR purity の理論的完成ではない。
必要なのは、DAW 実運用で以下を満たす runtime である。

- runaway しない
- saturation で収束する
- unload failure を起こしにくい
- drain completion を判定できる
- 長時間セッションで破綻しにくい

そのため、本統制文書の中心は次の 4 点に集約される。

1. saturation suppression
2. bounded residency
3. deterministic drain
4. reclaim pressure control
