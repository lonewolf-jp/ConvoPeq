# ISR Execution Authority Convergence 仕様（REV3.1）

## 位置づけ

本書は、REV2 で整備された runtime object 群を **実運用で破綻しにくい few-authority architecture** へ収束させる補完正本である。

目的は以下の差分を閉塞すること：

```text
「runtime object は存在する」
!=
「実装者が将来変更しても invariant を維持しやすい」
```

---

## 背景

REV2 で以下は成立した：

- Layer 0: LifecycleIsolationRuntime
- DSPHandleRuntime
- RTExecutionFrame / RTCapabilityFirewall
- WorldBridgeUtility / EpochArbitrationHelper
- HBRuntimeCore + HBTraceRuntime + HBVerifierRuntime
- ShutdownConvergenceRuntime
- bounded runtime budget

ただし未完の本質は **authority convergence** と **execution window determinism** にある。

---

## 中核方針

```text
many-runtime architecture
->
few-authority architecture
```

過剰な runtime object 追加より、authority/ownership 境界の単純化を優先する。

---

## 実コード subsystem（7）

以下を実コード上の主要 subsystem とする。

1. LifecycleHostAdapter
2. RuntimePublication
3. DSPHandleRuntime
4. RTExecutionRuntime
5. RetireRuntime
6. ShutdownRuntime
7. DebugRuntime（Debug/CI 限定）

### Runtime Core / Optional Debug Layer

- Runtime Core（5）: RuntimePublication / DSPHandleRuntime / RTExecutionRuntime / RetireRuntime / ShutdownRuntime
- Host Adapter: LifecycleHostAdapter
- Optional Debug Layer（1）: DebugRuntime
- 原則: **debug system ≠ runtime core**

---

## 統合・削減ルール

- `ClosureBuilder` + `SealedRuntime` -> `RuntimePublication`
- `CrossfadeAuthorityRuntime` -> `DSPHandleRuntime`
- `DSPQuarantineRuntime` -> `RetireRuntime`
- `HBTraceRuntime` + `HBVerifierRuntime` -> `DebugRuntime`
- `Validation/Budget/Proof` -> `DebugRuntime`

新規 runtime object 追加は `RuntimeReductionGate` 承認時のみ許可。

---

## world model（2-world固定）

```cpp
struct RuntimeBoundary
{
    RuntimeSnapshot publication;
    ExecutionFrame execution;
};
```

- PublicationWorld: immutable runtime
- ExecutionWorld: audio callback

ConvoPeq は single plugin process 前提のため、full federation semantics は採用しない。

運用固定（REV3.2）:

- `WorldBridgeUtility` は utility helper として扱い、authority root にはしない。
- epoch arbitration は RetireRuntime 内部責務への統合を優先する。

## RuntimePublication の責務境界

RuntimePublication は mega-runtime 化を避けるため、責務を以下へ限定する。

### 許可

- immutable snapshot publish
- publish version management
- frozen snapshot acquire
- publish-time closure / seal / tier validation の結果反映
- closure は publish ownership closure（create/connect/seal/validateAcyclic 相当）に限定

### 禁止

- proof/evidence validator の常駐実装
- introspection console の保持
- HB graph 常時生成
- runtime budget governance の常駐判断
- recovery orchestration の中心化

Debug/CI 向けの trace / validation / evidence export は `DebugRuntime` 側へ委譲する。
Runtime Core には publish/retire/shutdown の最小 barrier と fail-safe quarantine のみを残す。

---

## Authority model（capability-first）

```cpp
struct PublishAuthority {};
struct RetireAuthority {};
struct ShutdownAuthority {};
```

invariants:

- AC-1: authority capability is explicit at type level
- AC-2: authority transfer is explicit at call-site
- AC-3: invalid capability path is compile-time or fail-fast reject

運用固定（REV3.2）:

- capability 分離（`PublishAuthority` / `RetireAuthority` / `ShutdownAuthority`）を第一優先とする。
- runtime coordinator の導入・拡張を禁止し、authority は capability tag で固定する。
- RT callback scope での authority acquire/release（または同等挙動）を禁止する。

```cpp
// 注記: 既存互換レイヤが必要な場合でも、coordinator は
// build-time/helper scope（非RT・非core）でのみ扱い、
// runtime authority lifecycle を新規導入しない。
```

---

## callback-local consistency（RTExecutionRuntime）

```cpp
struct RTExecutionFrame
{
    RuntimeSnapshot snapshot;
    DSPHandle activeDSP;
    DSPHandle fadingDSP;
    uint64_t callbackEpoch;
};
```

invariants:

- RT-1: callback中 snapshot immutable
- RT-2: callback中 authority mutation 禁止

---

## DSPHandleRuntime（ownership単純化）

```cpp
class DSPHandleRuntime
{
public:
    DSPView acquireForCallback();
    void publish(std::shared_ptr<DSPGraph>);
    void finalizeCrossfade();
};
```

invariants:

- DSP-1: DSP lifetime source-of-truth singular
- DSP-2: crossfade complete before retire
- DSP-3: RetireRuntime以外 delete/reclaim 禁止

## LifecycleRuntime（host chaos 正規化）

```cpp
class HostChaosNormalizer
{
public:
    NormalizedLifecycleEvent normalize(RawJUCECallbackEvent);
};
```

invariants:

- HC-1: duplicate prepare collapse mandatory
- HC-2: release-before-prepare reject mandatory
- HC-3: callback during Releasing reject mandatory

## RetireRuntime（centralized reclaim）

invariants:

- RET-1: RetireRuntime 以外 reclaim 禁止
- RET-2: pending retire は callback 外でのみ解決

---

## ShutdownRuntime（bounded deterministic teardown）

phase:

```text
Running
-> StopAccepting
-> DrainCallbacks
-> FinalizeCrossfade
-> RetireDSP
-> ReleaseRuntime
-> Complete
```

```cpp
invariants:

- SH-1: callback count == 0
- SH-2: active crossfade == 0
- SH-3: pending retire == 0
- SH-4: observer == 0

---

## DebugRuntime（Debug/CI限定）

```cpp
invariants:

- DBG-1: Release build で proof/trace/verify 最小化
- DBG-2: Debug build で trace 部分有効
- DBG-3: CI build で verify/simulation 完全有効

運用固定（最新版レビュー反映）:

- Release runtime core は publish safety / retire safety / RT firewall / shutdown timeout を最小責務とする。
- ownership audit / HB instrumentation / verifier / artifact emitter は DebugRuntime（Optional Debug Layer）へ分離する。

stale DSP handle build方針:

- Release: quarantine + silence（必要時 fail-safe bypass）
- Debug: assert（必要時 trap）
- CI: abort

```text
RuntimeReductionGate（mandatory）
1. measurable RT gain
2. measurable invariant closure gain
3. measurable debug value
4. authority ambiguity evidence
```

---

## 統合 invariant set（REV3.1）

- ISR-1: publish後runtime immutable
- ISR-2: callback中runtime snapshot stable
- ISR-3: RetireRuntime以外reclaim禁止
- ISR-4: crossfade完了前retire禁止
- ISR-5: RT thread authority mutation禁止
- ISR-6: shutdown bounded completion mandatory
- GI-7: verification must never perturb RT deadline（RT path は bounded fixed-size telemetry のみ）

上記 ISR-1..ISR-6 を Release 完了ゲートとする。

## shutdown 最小完了モデル

shutdown は proof completeness より bounded deterministic completion を優先する。

```text
Running
-> StopAccepting
-> DrainCallbacks
-> FinalizeCrossfade
-> RetireDSP
-> ReleaseRuntime
-> Complete
```

必要 invariant:

- SH-1: callback count == 0
- SH-2: active crossfade == 0
- SH-3: pending retire == 0
- SH-4: observer == 0

---

## 推奨導入順（few-authority）

```text
LifecycleRuntime
    -> RuntimePublication
    -> DSPHandleRuntime
    -> RTExecutionRuntime
    -> RetireRuntime
    -> ShutdownRuntime
    -> DebugRuntime
```

---

## 関連文書

- `plan5.md`（ハブ）
- `ISR_JUCE_Lifecycle_Isolation.md`
- `ISR_DSPHandle_Runtime.md`
- `ISR_RT_Execution_Frame.md`
- `ISR_World_Bridge_Runtime.md`
- `ISR_Runtime_Proof_and_Recovery_Integration.md`
- `ISR_Runtime_Reduction_Strategy.md`
