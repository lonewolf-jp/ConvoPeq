# ConvoPeq ISR HB Graph Specification

## 目的

本書は、ConvoPeq ISR 移行で必要な **happens-before（HB）関係** を object/domain 単位で仕様化する。

狙い:

- memory_order を「点」ではなく「経路」で定義する
- publish / observe / retire / reclaim を混同しない
- telemetry を独立 domain として扱い、atomic 汚染を防ぐ

---

## 基本方針

1. HB は domain ごとに定義する
2. domain 間の同期依存は明示される場合のみ許可
3. `acquire/release` は対応ペアを持つ箇所にのみ適用
4. epoch/grace は reclaim 可否判定の仕様として独立定義

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## Domain 定義

### Domain A: Runtime Publish/Observe

対象:

- `GlobalSnapshot`
- Runtime publish payload（RuntimePublication 側の payload）

標準HB:

```text
construct
  -> publish store(release)
     -> observe load(acquire)
        -> read-only use
```

要件:

- publish owner は Message Thread 側
- observe は Audio Thread で read-only
- observe scope は lifetime guard で保護

### Domain B: DSP Lifetime Retire/Reclaim

対象:

- DSPHandle / DSP token family
- `activeDSP/currentDSP/fadingOutDSP/queuedOldDSP` 相当の寿命管理

標準HB:

```text
lifetime token published
  -> retire enqueue
     -> epoch/grace satisfied
        -> reclaim destroy
```

要件:

- retire authority は単一
- reclaim は grace 条件成立後のみ
- raw pointer 直接 retire は禁止

### Domain C: Telemetry Observation

対象:

- RT write / NonRT read の観測値

標準HB:

```text
RT atomic write
  -> NonRT atomic read (snapshot)
```

要件:

- execution-local state と分離
- publish domain と同一仕様を流用しない

### Domain D: UI Interaction

対象:

- UI state snapshot
- monitor/panel refresh

標準HB:

```text
NonRT state snapshot publish
  -> UI thread read/apply
```

### Domain E: Async Background Loading / IR Streaming

対象:

- async IR load result
- background prepared resource handoff

標準HB:

```text
background prepare complete
  -> publish handoff(release)
     -> nonRT/RT consumer acquire
```

### Domain F: Parameter Smoothing / Audio Callback Sync

対象:

- parameter smoothing state handoff
- callback-visible control values

標準HB:

```text
control publish
  -> audio callback observe
```

JUCE callback 拡張（確定）:

- F1: control publish -> `getNextAudioBlock()` observe
- F2: `prepareToPlay()` complete -> first audio callback start
- F3: last audio callback end -> `releaseResources()` start

制約:

- `prepareToPlay()` 未完了状態で audio callback が publish payload を参照してはならない
- `releaseResources()` 開始後は新規 publish payload observe を禁止する
- host callback 由来の再入可能経路は Domain F 内で単一順序規約に従う

---

## Object 別 domain mapping（確定版 v2）

| Object | Primary Domain | Writer | Reader | HB Key |
| --- | --- | --- | --- | --- |
| GlobalSnapshot pointer | A | Message publish owner | Audio observer | release->acquire |
| ObservedRuntime guard scope | A | observer | observer | scope-bound validity |
| RuntimePublication pointer | A | Message publish owner | Audio observer | release->acquire（確定） |
| DSPHandle lifecycle | B | retire authority | reclaim authority | enqueue->grace->destroy |
| `currentDSP` visibility | B/A bridge | Message transition owner | Audio | publish visibility + lifetime token |
| RTStatistics counters | C | Audio | NonRT/UI | atomic write->read |
| RTLocalState counters | RT local (outside A/B/C) | Audio | Audio | thread-private |
| UI panel state | D | NonRT snapshot owner | UI | snapshot publish->UI read |
| IR async handoff token | E | background loader | runtime consumer | prepare->publish->acquire |
| smoothing control state | F | control publisher | audio callback | publish->callback observe |

---

## Bridge Rules（domain間境界）

### A -> B（publish から retire へ）

- publish payload に含まれる lifetime token は、retire authority へ正しく受け渡されること
- pointer だけ渡して authority を渡さない設計は禁止

### B -> C（lifetime と telemetry）

- telemetry は lifetime判定に使わない
- telemetry 遅延/欠落で reclaim 判定が変化しないこと

### F -> B（RT detect から NonRT retire enqueue）

- Audio Thread は completion detect のみを行い、retire enqueue を実行してはならない
- retire enqueue は NonRT bridge request を経由して authority へ到達すること
- callback 直enqueueは違反として検出・失敗扱いにする

### A <-> C（publish と telemetry）

- telemetry 値を publish correctness の唯一根拠にしない
- publish 成否は A domain の HB で判定する

### A -> F（publish と audio callback）

- RuntimePublication の payload は callback 可視化前に closure 完了していること
- callback 側で追加の ownership 解決を要求する payload 構造は禁止

### F -> D（callback と UI）

- callback 側状態を UI が読む場合、直接参照は禁止
- Domain C または D の snapshot 経路を経由すること

---

## RuntimePublication payload closure 規則（確定）

payload closure は次を満たす:

1. observer が参照する ownership-bearing object が payload 内 handle 経路で閉じる
2. payload 外 raw pointer 依存を持たない
3. lifetime 判定は payload 内 token + authority 規約のみで完結する

違反例:

- callback 側で外部 mutable singleton 参照が必要
- payload が retire authority 情報を欠く
- payload publish 後に ownership graph を追記する

---

## Memory Order 方針（簡易）

- publish store: release
- observe load: acquire
- telemetry write/read: object特性に応じた atomic order（最低限整合性を文書化）
- reclaim 判定: epoch/grace 仕様に従う（単純な acquire/release だけで代替しない）

---

## 検証項目（Gate）

- [ ] 全 object が domain に割当済み
- [ ] release/acquire ペアが経路として定義済み
- [ ] retire->reclaim の grace 条件が明文化済み
- [ ] domain bridge rule の違反ケースを禁止事項へ反映済み
- [ ] bug2 系シナリオで HB 欠落がない
- [ ] JUCE callback ordering（prepareToPlay / getNextAudioBlock / releaseResources）の順序検証が完了
- [ ] RuntimePublication の payload closure 違反ケースが禁止事項へ反映済み
- [ ] RT detect -> NonRT enqueue bridge（R9）の検証が完了

現時点判定:

- [x] Domain A/B/C の必須HBは定義済み
- [x] Domain D/E/F の追加要否は「必要」で確定
- [ ] 実装後の bug2 系シナリオ検証は未実施

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_Runtime_State_Matrix.md`
- `doc/work/ISR_Retire_Authority_Graph.md`
- `doc/work/ISR_DSPCore_Decomposition_Analysis.md`
