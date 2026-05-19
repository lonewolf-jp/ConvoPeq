# ConvoPeq DSPCore Decomposition Analysis（確定版 v2）

## 目的

`DSPCore` を単一 immutable object とみなす前提を廃止し、
**DSPConfig / DSPExecutionInstance / Telemetry** の3層分解を確定仕様として定義する。

---

## 分解結果（確定）

## Layer 1: DSPConfig（immutable publish layer）

含有:

- coefficients
- FIR/IR metadata
- routing topology
- processing order

性質:

- Message/prepare 側で構築
- publish 後 read-only
- RuntimeWorldRetireManager 管理下で retire/reclaim

## Layer 2: DSPExecutionInstance（mutable RT layer）

含有:

- overlap/circular buffers
- SIMD accumulators
- delay lines
- fade progress runtime state

性質:

- Audio Thread 専用更新
- 直接 publish しない
- 終端イベントで retire enqueue（authority: RuntimeWorldRetireManager）

終端規則（確定）:

- crossfade 完了
- runtime world 切替完了
- engine shutdown path

destroy 規則（確定）:

- shared EpochDomain の grace 条件成立後に destroy callback 実行

## Layer 3: RTStatistics / Telemetry

含有:

- processed block counters
- underrun/overrun diagnostics
- monitor 用統計値

性質:

- RT write / NonRT read
- lifetime判定に使用しない
- Domain C として独立

---

## DSPHandle 具体方式（確定）

`refOrHandle` は **generation-index handle + slot indirection** を採用。

```text
DSPHandle {
  slotIndex
  generation
  retireAuthorityId
}
```

規則:

- raw pointer retire 禁止
- generation mismatch は stale handle として reject
- handle 無効化は retire enqueue 時に行う

---

## Epoch 戦略（確定）

- GlobalSnapshot と DSP lifetime は **shared EpochDomain**
- 理由: reclaim 経路の統一と検証単純化

---

## 非採用案（却下）

- intrusive_refcount（RT経路での参照更新コストを回避）
- epoch-retained raw ptr（pointer leak リスクが高い）
- stable arena handle（再配置制約が強すぎる）

---

## 実装前 Gate

- [x] DSPConfig / DSPExecutionInstance / Telemetry の境界を確定
- [x] DSPHandle 方式を確定
- [x] destroy authority を RuntimeWorldRetireManager に単一化
- [x] epoch 方針を shared に確定
- [ ] 実装後に bug2 系 UAF 検証を実施

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_Runtime_State_Matrix.md`
- `doc/work/ISR_Retire_Authority_Graph.md`
- `doc/work/ISR_HB_Graph_Specification.md`
