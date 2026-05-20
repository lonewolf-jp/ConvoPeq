# ConvoPeq ISR Runtime State Matrix（確定版 v2）

## 目的

本書は、実装前に runtime state を object family 単位で固定し、
ownership leakage / reclaim hole / HB 欠落を防止するための正本である。

運用注記（REV3.1 優先）:

- 本表の `Retire/Reclaim Authority` 列に現れる `RuntimeWorldRetireManager` は
  capability-first 制約下での実装委譲名を指す（coordinator は互換 shim）。
- authority source-of-truth は capability-first を優先する。

### REV3.2運用優先注記

- 本表の state/object 分類は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
  `runtime exposes evidence / CI validates evidence` を固定方針とする。
- authority 列の解釈衝突時は capability-first を優先し、
  coordinator は互換 shim、manager 名は実装委譲として扱う。

用語正規化（齟齬回避）:

- 本表では `RuntimePublication` を正規記法として扱う。

---

## Runtime State Matrix（確定）

| Object/State | Mutable | Writer Thread | Reader Thread | Publish | Lifetime | Retire/Reclaim Authority | Epoch | HB Domain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GlobalSnapshot | No | Message publish path | Audio observe path | Yes | snapshot | SnapshotRetireManager | Yes (shared) | A |
| ObservedRuntime.guard | scope-bound | observer | observer | No | scope | N/A | Yes | A |
| RuntimePublication pointer（公開ポインタ） | Atomic ptr | Message publish owner | Audio observer | Yes | runtime | RuntimeWorldRetireManager | Yes (shared) | A |
| RuntimePublication payload closure metadata（公開メタデータ） | No (publish-fixed) | Message build path | Audio observer | Yes | runtime | RuntimeWorldRetireManager | Yes (shared) | A/F bridge |
| EngineRuntime (publish payload) | Yes (sealed-at-publish) | Message build path | Audio read-only | Yes | runtime | RuntimeWorldRetireManager | Yes (shared) | A |
| RuntimeGraph (publish payload) | Yes (sealed-at-publish) | Message build path | Audio read-only | Yes | runtime | RuntimeWorldRetireManager | Yes (shared) | A/B bridge |
| DSPHandle | logical mutable | Message transition owner | Audio + reclaim | Yes (token publish) | runtime | RuntimeWorldRetireManager | Yes (shared) | B |
| activeDSP slot | Yes | Message transition owner | Message/Audio bridge | No (via handle) | runtime | RuntimeWorldRetireManager | Yes (shared) | B |
| currentDSP visibility | Atomic | Message transition owner | Audio | No (derived) | runtime | RuntimeWorldRetireManager | Yes (shared) | B |
| fadingOutDSP state | Atomic | Message transition owner | Audio | No (derived) | runtime | RuntimeWorldRetireManager | Yes (shared) | B |
| queuedOldDSP state | Atomic | Message transition owner | Reclaim path | No (derived) | runtime | RuntimeWorldRetireManager | Yes (shared) | B |
| DSPExecutionInstance | Yes | Audio | Audio | No | runtime | RuntimeWorldRetireManager (end-of-life enqueue) | Yes (shared) | B / RT local |
| IR resource binding | Yes (handoff-controlled) | Background loader + Message | Runtime consumer | Yes (handoff token) | runtime | RuntimeWorldRetireManager | Yes (shared) | E |
| convolution scratch buffers | Yes | Audio | Audio | No | callback/block | N/A | No | RT local |
| fade transition objects | Yes | Audio + Message transition | Audio | No/partial | runtime | RuntimeWorldRetireManager | Yes (shared) | B/F |
| FFT plan/cache | Yes (init後準固定) | NonRT prepare | Audio read/use | Yes (prepared handoff) | runtime | RuntimeWorldRetireManager | Yes (shared) | E/F |
| JUCE prepareToPlay lifecycle state | Yes (phase state) | Message/host callback | Audio callback | No | runtime | RuntimeWorldRetireManager | No | F |
| JUCE releaseResources lifecycle state | Yes (phase state) | Message/host callback | Audio callback / NonRT | No | runtime | RuntimeWorldRetireManager | No | F |
| JUCE callback ownership state | Yes | Message/host callback | Audio/UI | No | runtime | RuntimeWorldRetireManager | No | F/D |
| Callback bridge snapshot token | Atomic | Audio | NonRT/UI | Yes (snapshot handoff) | runtime | N/A | No | C/D/F bridge |
| RTStatistics.processedBlockCount | Atomic | Audio | NonRT/UI | No | runtime | N/A | No | C |
| UI panel snapshot | Yes | NonRT state owner | UI | Yes | runtime | N/A | No | D |

---

## Domain キー

- A: Runtime Publish/Observe
- B: DSP Lifetime Retire/Reclaim
- C: Telemetry Observation
- D: UI Interaction
- E: Async Background Loading / IR Streaming
- F: Parameter Smoothing / Audio Callback Sync

---

## 確定事項

1. EpochDomain は shared strategy（GlobalSnapshot と Runtime world で共有）
2. RuntimePublication は runtime-only publish world
3. DSP pointer retire は禁止、DSPHandle 経由のみ
4. RTLocalState は execution-local のみ（ownership-bearing state 禁止）
5. JUCE callback lifecycle（prepare/release）は Domain F で順序管理する
6. RuntimePublication の payload closure は metadata 行で追跡する

---

## Gate（実装着手判定）

- [x] 未確定項目の authority が全て割当済み
- [x] Epoch 方針（shared）が確定済み
- [x] HB Domain が object 単位で割当済み
- [ ] 実装後の検証（bug2 系再現試験）は未実施
