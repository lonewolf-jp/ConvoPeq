# ConvoPeq ISR Formal Guarantee Package

## 目的

ISR を「設計規律」から **formal enforceable system** へ移行するための統合仕様。
本書は package 正本であり、各項目の詳細は既存正本へ反映して閉じる。

---

## P1. Closure Descriptor System（R11）

### P1仕様

- RuntimePublishWorld は transitive ownership を `PayloadClosureDescriptor` で明示する
- descriptor node は object kind / ownership / mutability / lifetime / HB domain を保持する

### P1必須ルール

- Rule-C1: RuntimePublishWorld 外 mutable dependency 禁止
- Rule-C2: descriptor 未登録 object の publish ptr-chain 参照禁止
- Rule-C3: descriptor に retire authority / HB domain / allocator family を必須保持

### P1最小検証

- validateRuntimeClosure(world) で mutable leak / unmanaged ownership / external ref を reject

---

## P2. Payload Tier System（R12）

### P2仕様

- payload object を tier 分類する
  - InlineImmutable
  - ImmutableShared
  - ExternalPinned
  - RTLocalOnly
  - Forbidden

### P2必須ルール

- Rule-P1: Forbidden tier は publish payload 内禁止
- Rule-P2: RTLocalOnly は RuntimePublishWorld 内禁止

---

## P3. Immutable Facade + Mutable Core（R13）

### P3仕様

- publish graph は read-only projection のみ公開
- mutable cache / lazy-init / mutex は RuntimePublishWorld から隔離

### P3必須ルール

- Rule-I1: publish graph 内 mutable atomic 禁止
- Rule-I2: publish graph 内 mutex 禁止
- Rule-I3: publish graph 内 lazy-init 禁止

---

## P4. Deferred Retire Intent Queue（R14）

### P4仕様

- RT thread は completion detect のみ
- retire authority 実行は NonRT bridge へ委譲

### P4必須ルール

- Rule-R1: RT reclaim 禁止
- Rule-R2: RT delete 禁止
- Rule-R3: RT retire authority 禁止

---

## P5. Shutdown HB FSM（R15）

### P5仕様

- shutdown phase を formal state machine で管理
- phase transition は atomic で管理し、HB chain を固定

### P5必須順序

- AudioStop -> ObserverDrain -> RetireStop -> EpochSettlement -> ReclaimComplete -> AllocatorShutdown

### P5禁止事項

- StopRetireIngress 以降 new retire enqueue 禁止
- EpochSettlement 以降 new observe 禁止

---

## P6. HB Failure Spec + Reorder Simulation（R16）

### P6仕様

- bug2 を最小HB欠落モデルとして定義
- failure ordering と required HB を対照表で管理

### P6検証

- CI で forced reorder / epoch lag / retire delay / observe race を注入する

---

## P7. Epoch Abstraction Layer（R17）

### P7仕様

- epoch 実装へ直接依存せず、coordinator interface を経由する
- shared/split/hybrid を runtime policy として切替可能にする

### P7目的

- shared epoch を architecture invariant にしない

---

## P8. CI Verification Pipeline（R18）

### P8必須ステージ

1. Atomic Dot-Call Scan
2. Post-publish Mutation Detector
3. Ownership Closure Validator
4. HB Reorder Simulator
5. Shutdown Sequencing Verifier
6. Retire Latency Monitor

### P8運用

- 失敗は warning ではなく merge blocker とする

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_Completeness_Risk_Backlog.md`
