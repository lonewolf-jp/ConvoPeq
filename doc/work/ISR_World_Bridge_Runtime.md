# ISR World Bridge Runtime 仕様

## 位置づけ

本書は **ISR 参照設計における world/epoch ambiguity の整理メモ**であり、ConvoPeq 実装では helper-level reference として扱う。

ISR の参照設計には 3-world 記述が存在するが、
ConvoPeq 実装運用は PublicationWorld / ExecutionWorld の2-world固定を優先する。
cross-world 操作における authority / epoch / reclaim の曖昧性は、runtime object 追加ではなく helper/utility で整理する。
本仕様はその world/epoch ambiguity を参照設計として整理する。実装authorityは few-authority / 2-world 固定を優先する。

位置づけ: `plan5.md` → REV2 未閉塞系統 D → 本書（詳細）

運用注記（REV3.1 優先）:

- 本書の 3-world モデル（Snapshot/Runtime/DSP）は **参照モデル**。
- ConvoPeq の現行実装運用では `plan5.md` REV3.1 に従い、
  `PublicationWorld / ExecutionWorld` の 2-world 簡略プロファイルを優先する。
- 衝突時の優先順位は `plan5.md` の「文書優先順位（解釈衝突時）」に従う。

### REV3.2運用優先注記

- 本書の federated / 3-world 記述は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- `WorldBridgeUtility` は utility helper であり、authority root ではない。
- epoch arbitration は RetireRuntime 内部責務への統合を優先する。
- Release 常駐の full federation runtime は採用しない。

---

## 問題の本質

### World/Epoch ambiguity（reference）

```text
現状のリスク:
  SnapshotWorld で retire された DSP が RuntimeWorld でまだ参照されている可能性
  RuntimeWorld の epoch と DSPWorld の epoch が独立して進行し mismatch する可能性
  "single world settled ≠ global settled" を runtime が detect しない
  cross-world の authority routing が実装者の責任になっている

要求:
  runtime core での cross-world 常駐を必須化しない
  world は Publication/Execution の2-world運用を優先する
  World/Epoch helper は Debug/CI 参照用途を主とする
```

---

## 定義

### RuntimeWorldId

```cpp
enum class RuntimeWorldId
{
    SnapshotWorld,  // immutable snapshot objects の世界
    RuntimeWorld,   // live runtime objects の世界
    DSPWorld        // DSP instance lifetime の世界
};
```

### WorldEpoch

```cpp
struct WorldEpoch
{
    RuntimeWorldId worldId;
    uint64_t       epoch;
};
```

---

## WorldBridgeUtility（helper reference）

cross-world publish / retire / validate の参照設計を整理する。
実装運用では Debug helper 以上へ昇格させない。

```cpp
namespace WorldBridgeUtility
{
  // NonRT: PublicationWorld への publish
    RuntimePublishToken publishToRuntimeWorld(
        RuntimeWorldId source,
        PublishPayload payload);

    // NonRT: SnapshotWorld への publish
    SnapshotToken publishToSnapshotWorld(
        RuntimeWorldId source,
        SnapshotPayload payload);

    // NonRT: world から retire（全 world で参照が消えた後にのみ有効）
    void retireFromWorld(
        RuntimeWorldId world,
        RetireHandle handle);

    // NonRT: 2-world 整合性検証
    TwoWorldValidationResult validateTwoWorld();

    // NonRT: epoch 参照（RetireRuntime helper）
    WorldEpoch globalEpoch(RuntimeWorldId world) const;
  }
```

---

## EpochArbitrationHelper（helper reference）

複数 world にまたがる epoch 整合の参照設計を示す。ConvoPeq 実装では
RetireRuntime 内部責務統合を優先し、helper-level utility を超えて昇格させない。

```cpp
namespace EpochArbitrationHelper
{
    // world の epoch を advance（publish / retire 時に呼ばれる）
    void advance(RuntimeWorldId world);

    // cross-world epoch mismatch を検出
    // mismatched → FED-2 violation → Abort
    bool isConsistent(RuntimeWorldId worldA, RuntimeWorldId worldB) const;

    // 全 world が "settled"（活動中操作ゼロ）か確認
    bool allSettled() const;

    // 現在の全 world epoch をダンプ
    void emitEpochTrace(const std::filesystem::path& outputPath) const;
  }
```

---

## AuthorityRoutingMatrix

どの runtime がどの世界に対してどの操作を持つかを定義する。

```cpp
enum class WorldOperation
{
    Publish,
    Retire,
    Reclaim,
    Observe
};

struct AuthorityEntry
{
    RuntimeWorldId    world;
    WorldOperation    operation;
    const char*       authorizedRuntime;  // runtime 名（文字列識別）
    bool              rtCallable;         // RT thread から呼べるか
};

// 実装ガイド: constexpr 配列として静的定義する
// 動的変更禁止（GI-1: publish graph immutable after seal に準ずる）
```

### 典型的 authority mapping

| operation | world | authorized runtime | RT callable |
| --- | --- | --- | --- |
| Publish | RuntimeWorld | WorldBridgeUtility | No |
| Retire | RuntimeWorld | WorldBridgeUtility | No |
| Reclaim | DSPWorld | DSPHandleRuntime | No |
| Observe | SnapshotWorld | IntrospectionRuntime | Yes（read-only） |
| Retire | SnapshotWorld | WorldBridgeUtility | No |

---

## Invariants

| 識別子 | 内容 | 違反時アクション |
| --- | --- | --- |
| FED-1 | cross-world reclaim authority ambiguity 禁止 | Abort |
| FED-2 | cross-world epoch mismatch 禁止 | Abort |
| FED-3 | single-world settled ≠ global settled（過剰な shutdown 許可禁止） | Abort |
| GI-5 | plugin 運用では world/epoch helper の常駐必須化を行わず、2-world + RetireRuntime統合を優先する | Abort |

---

## artifacts（Debug/CI reference）

### federation_epoch_trace.json

```json
{
  "schema": "federation_epoch_trace_v1",
  "worlds": [
    {
      "worldId": "RuntimeWorld",
      "currentEpoch": 42,
      "settled": true
    },
    {
      "worldId": "SnapshotWorld",
      "currentEpoch": 42,
      "settled": true
    },
    {
      "worldId": "DSPWorld",
      "currentEpoch": 41,
      "settled": false
    }
  ],
  "allSettled": false,
  "invariant_violations": []
}
```

---

## shutdown convergence との連携

`ShutdownConvergenceRuntime`（Layer 6）は shutdown 完了を宣言する前に
RetireRuntime 側の settle 条件が満たされることを確認しなければならない。

`EpochArbitrationHelper::allSettled()` は Debug/CI 参照経路で利用可能とする。

```text
Shutdown sequence:
  1. ShutdownConvergenceRuntime: phase → ShutdownInitiated
  2. DSPHandleRuntime: 全 DSP を retire
  3. WorldBridgeUtility: 全 world に retire を通知
  4. EpochArbitrationHelper: allSettled() を polling
  5. allSettled() == true → ShutdownConvergenceRuntime: phase → Converged
  6. ShutdownConvergenceRuntime: shutdown evidence artifact を export
```

---

## Closed criteria

- [ ] WorldBridgeUtility が helper reference として 2-world 運用へ読み替え可能である
- [ ] EpochArbitrationHelper が RetireRuntime 内部責務へ統合可能な形で記述されている
- [ ] FED-1 / FED-2 / FED-3 の違反が runtime abort で閉塞されている
- [ ] AuthorityRoutingMatrix が静的に定義され、動的変更が禁止されている
- [ ] shutdown は allSettled() == true の確認後にのみ完了宣言される
- [ ] federation_epoch_trace.json は Debug/CI 参照証跡としてのみ扱われる

---

## 関連文書

- `plan5.md`: REV2 未閉塞4系統 D 系統参照、GI-5
- `ISR_10Layer_Implementation_Specification.md`: 修正版実装順序（ステップ 12）
- `ISR_Shutdown_State_Machine.md`: shutdown convergence FSM
- `ISR_Retire_Authority_Graph.md`: retire authority 詳細
- `ISR_DSPHandle_Runtime.md`: DSPWorld reclaim
- `ISR_Runtime_Proof_and_Recovery_Integration.md`: recovery + shutdown convergence
