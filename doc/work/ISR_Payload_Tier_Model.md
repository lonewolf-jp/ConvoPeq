# ConvoPeq ISR Payload Tier Model

## 目的

本書は **R12: Payload Tier System** の authoritative specification である。

publish payload に含まれる全オブジェクトを tier に分類し、
tier に応じた publish ルール・CI 強制ゲートを定義することで
payload boundary ambiguity を形式的に排除する。

---

## 背景・動機

現状は payload boundary が概念レベルに留まっており、
「どのオブジェクトが publish closure に入れるか」が開発者の判断に依存している。

tier を固定することで：

- 不適切オブジェクトの publish を CI で自動 reject できる
- `ISR_Runtime_Closure_Descriptor.md` の ClosureNode の mutability/lifetime 分類と対応させる

---

## PayloadTier 定義

```cpp
enum class PayloadTier
{
    InlineImmutable,   // publish 時 inline 保持、以降 immutable
    ImmutableShared,   // 共有参照、epoch-managed、read-only
    ExternalPinned,    // 外部 allocate、lifetime は呼び出し側が保証
    RTLocalOnly,       // Audio Thread ローカル、publish 禁止
    Forbidden          // publish closure への混入を CI が reject
};
```

---

## PayloadCapability 定義

各 payload node は `CapabilitySet` を保持する。

```cpp
enum class PayloadCapability
{
    ImmutableRead,    // post-publish 読み取りのみ許可
    EpochManaged,     // EpochDomain による retire/reclaim 管理
    ExternalPinned,   // 外部 lifetime 保証による参照
    RTLocal,          // Audio Thread 専用、publish closure 禁止
    Reclaimable,      // retire authority による reclaim 対象
    AsyncVisible      // HB domain を必要とする非同期可視
};

using CapabilitySet = uint32_t; // bitmask of PayloadCapability
```

---

## Tier ↔ CapabilitySet 対応

| Tier              | 必須 Capability                         | 禁止 Capability |
| ----------------- | --------------------------------------- | ---------------- |
| InlineImmutable   | ImmutableRead                           | RTLocal          |
| ImmutableShared   | ImmutableRead, EpochManaged, Reclaimable| RTLocal          |
| ExternalPinned    | ImmutableRead, ExternalPinned           | RTLocal          |
| RTLocalOnly       | RTLocal                                 | ImmutableRead, EpochManaged, AsyncVisible |
| Forbidden         | （該当なし）                            | （全 Capability）|

---

## Object Family → Tier 固定割り当て表

| Object Family          | Tier              | 備考                              |
| ---------------------- | ----------------- | --------------------------------- |
| RuntimeGraphNode       | InlineImmutable   | publish 根幹ノード                |
| DSPHandle              | ImmutableShared   | epoch-managed、retire authority 必須 |
| CoeffBuffer / IR blob  | ExternalPinned    | mkl_malloc で確保、外部 lifetime  |
| FFTPlan                | ImmutableShared   | publish 後 read-only              |
| smoothing state        | RTLocalOnly       | Audio Thread ローカル             |
| JUCE device ptr        | Forbidden         | JUCE mutable ptr は publish 禁止  |
| telemetry counter      | RTLocalOnly       | 観測専用、publish closure 外      |
| async callback state   | Forbidden         | 非同期可変状態は publish 禁止     |

---

## publish ルール

### P1: RTLocal → publish closure 禁止

```text
RTLocal capability を持つ node が
publish closure に含まれてはならない
```

### P2: AsyncVisible → HB domain 必須

```text
AsyncVisible capability を持つ node は
ClosureHBDomain が明示されていなければならない
```

### P3: Reclaimable → retire authority 必須

```text
Reclaimable capability を持つ node は
retire authority (reclaimAuthority) が有効でなければならない
```

---

## validatePayloadCapabilities()

publish 前に `validateClosureGraph()` と連携して呼び出すこと。

```cpp
enum class PayloadCapabilityValidationResult
{
    Valid,
    RTLocalInPublishClosure,      // P1 violation
    AsyncVisibleMissingHBDomain,  // P2 violation
    ReclaimableWithoutAuthority,  // P3 violation
    ForbiddenTierInClosure        // Forbidden tier が closure 内に存在
};

PayloadCapabilityValidationResult validatePayloadCapabilities(const ClosureGraph& graph);
```

---

## CI 強制ゲート

### Forbidden tier reject

```text
Forbidden tier オブジェクトが publish closure に含まれた場合:
→ CI fail（merge reject）
```

対象：

- JUCE mutable device ptr
- async callback state
- hidden singleton
- 未登録 mutable cache

### RTLocalOnly tier reject

```text
RTLocalOnly オブジェクトが publish closure に含まれた場合:
→ CI fail（merge reject）
```

---

## 関連正本

- `ISR_Runtime_Closure_Descriptor.md` — ClosureNode 構造と validateClosureGraph()
- `ISR_Verification_Pipeline.md` V4 — Payload Tier Validation ステージ
- `ISR_Formal_Guarantee_Package.md` P2 — 統合保証パッケージ参照

## Backlog 参照

- `ISR_Completeness_Risk_Backlog.md` R12 — Closed 最小検証項目

## ステータス

- Spec-Fixed: 2026-05-20
- Closed: 未完（実装・CI検証未実施）
