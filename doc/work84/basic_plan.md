はい。最新版の `ConvoPeq.md` を基準に解析し、さらに併せて情報源の **Practical Stable ISR Bridge Runtime** の設計とも照合した結果、ConvoPeq は既に ISR の骨格（RuntimeWorld・RuntimeStore・Coordinator・Admission・RCU/Epoch・Crossfade 分離）を備えていますが、**まだ「実装単位」で責務が残留している箇所がある**と判断します。

私であれば、今後は「バグ修正」ではなく**Runtime全体をISR Runtime OS化する**方向へ統合します。

優先順位は次のようになります。

| 優先度 | 改修テーマ                    | 効果    |
| --- | ------------------------ | ----- |
| S   | Runtime Authority統合      | ★★★★★ |
| A   | Runtime Transaction化     | ★★★★★ |
| A   | RuntimeWorld完全Immutable化 | ★★★★★ |
| A   | Retire Pipeline完全一本化     | ★★★★★ |
| B   | Runtime Intent統一         | ★★★★☆ |
| B   | Crossfade Runtime純化      | ★★★★☆ |
| C   | HealthMonitor自己修復化       | ★★★☆☆ |

---

## ① RuntimeCoordinatorを「唯一のRuntime OS」にする

現在でもCoordinatorがありますが、まだ

* Builder
* RuntimeStore
* Crossfade
* Lifetime
* Retire

が個別に状態を持っています。

これを

```
RuntimeCoordinator
```

だけが状態遷移を管理する構造へ統合します。

例えば

```
Build

↓

Validate

↓

Publish

↓

Crossfade

↓

Retire

↓

Delete
```

という状態機械(State Machine)をCoordinatorだけが持ちます。

すると

```
RuntimeStore

DSPTransition

SnapshotCoordinator

Lifetime
```

は全て

```
Coordinator.execute()
```

を呼ぶだけになります。

これが一番大きな改善になります。

---

## ② Runtime Transactionを導入する

現在は

```
Builder
↓

Publish
↓

Crossfade
```

が分離しています。

これを

```
RuntimeTransaction
```

にまとめます。

例

```
struct RuntimeTransaction
{
    RuntimeIntent

    RuntimeWorld

    ValidationResult

    CrossfadePlan

    RetirePlan

    PublishToken
};
```

になります。

Coordinatorは

```
Transaction

↓

Commit
```

だけ行います。

これで途中状態が存在しなくなります。

---

## ③ RuntimeWorldを100% Immutableにする

Practical ISRでも

```
Publish後変更禁止
```

が最重要になっています。

私なら

RuntimeWorld内部は

```
const
```

しか置きません。

例えば

```
RuntimeWorld
```

は

```
DSPGraph

Projection

Routing

Metadata

CrossfadePlan
```

まで全部含めます。

逆に

```
lazy cache

mutable

optional initialize
```

は全廃します。

---

## ④ CrossfadePlanをRuntimeWorldへ格納する

現在は

```
CrossfadeRuntime
```

と

```
CrossfadeAuthority
```

が分離しています。

しかし実際には

Crossfade Runtimeは

```
Decision
```

を持つべきではありません。

私は

```
RuntimeWorld
```

生成時に

```
CrossfadePlan
```

まで完成させます。

RTは

```
plan.execute()
```

だけになります。

---

## ⑤ Retire AuthorityをEpochだけへ集約する

これはかなり重要です。

現在でも

```
RuntimeStore

Coordinator

Lifetime

Epoch
```

に寿命責務が散っています。

最終形は

```
RuntimeStore

↓

RetireIntent

↓

EpochDomain

↓

Delete
```

だけです。

つまり

```
delete
```

できる場所を

```
EpochWorker
```

だけにします。

---

## ⑥ RuntimeIntentを導入する

UIから

```
Parameter

Preset

Automation

IR Load
```

が直接Builderへ行っています。

これを

```
RuntimeIntent
```

に統一します。

例

```
ParameterIntent

PresetIntent

IRIntent

AutomationIntent
```

です。

Coordinatorは

```
Intent

↓

RuntimeWorld
```

を生成します。

これで

将来

Network

MIDI

OSC

Remote

まで追加してもBuilderは変わりません。

---

## ⑦ ValidationをRuntime Semantic Validationへ拡張する

現在は

```
Resource

Topology
```

中心ですが、

さらに

```
Semantic Validation
```

を追加します。

例えば

```
Projection

Routing

Oversampling

Latency

DSPGraph

Channel Layout

Crossfade
```

まで検証します。

これで

```
Publish後に判明するバグ
```

をなくせます。

---

## ⑧ HealthMonitorをRecoverまで拡張する

Practical ISRでも

```
Detect

↓

Diagnose

↓

Recover

↓

Verify
```

が理想形になっています。

ConvoPeqでも

例えば

```
Retire Stall
```

なら

```
Drain

Advance Epoch

Retry

Verify
```

まで自動化できます。

---

## ⑨ Runtime Capabilityを導入する

これは長期運用向けですが非常に効果があります。

RuntimeWorldに

```
Capability
```

を持たせます。

例えば

```
SupportsLinearPhase

SupportsTruePeak

SupportsMixedPhase

SupportsOversampling

SupportsBackgroundIRLoad
```

です。

BuilderはCapabilityを見るだけになります。

将来DSPが増えても

```
if(version)
```

が消えます。

---

## ⑩ Runtime Generation管理

最後に追加したいのは

```
Generation
```

です。

例えば

```
Generation 125
```

をWorldに持たせます。

全て

```
Publish

Crossfade

Retire

Delete

Monitor
```

がGeneration単位になります。

すると

ログ解析

障害解析

リーク解析

まで非常に容易になります。

---

## 私が考える最終形

現状のConvoPeqは、既に **RuntimeWorld・Coordinator・RuntimeStore・EpochDomain** というISRの中核要素を持っています。

一方で、実運用でさらに破綻しにくい構造にするには、「機能の追加」ではなく「責務の削減」を進めるべきです。

最終形では、各コンポーネントの責務は次のように極小化されます。

* **RuntimeBuilder**：`RuntimeWorld` を構築するだけ。
* **RuntimePublicationValidator**：構築済み `RuntimeWorld` の検証だけ。
* **RuntimeCoordinator**：Publish・Crossfade・Retire を決定する唯一の Authority。
* **RuntimeStore**：アクティブ `RuntimeWorld` の atomic 公開だけ。
* **Audio Thread**：`RuntimeWorld` を読み取り DSP を実行するだけ。
* **EpochDomain**：寿命管理と Reclaim だけ。
* **DeletionWorker**：実際の `delete` を行うだけ。

この構成では、Practical Stable ISR Bridge Runtime が掲げる「RTは待たない・解放しない・判断しない」「Coordinator が唯一の Authority」「Retire は必ず Epoch を経由する」という不変条件と一致し、責務境界がさらに明確になります。

私が現時点で提案する統合案は、この**「Runtime OS化（Coordinator中心化）」**を最終到達点とするものです。これにより、将来的に IR 管理、バックグラウンドロード、マルチインスタンス、DSP モジュール追加などを行っても、RT・寿命管理・状態遷移の設計を崩さず拡張できる構造になります。



最新版 ConvoPeq の方向性（Authority の集約、RuntimeWorld 中心、SnapshotCoordinator、RuntimeStore、Epoch/Retire）と Practical Stable ISR Bridge Runtime の設計思想を前提にすると、私なら ISR 統合は「全面リライト」ではなく、**約10フェーズで既存資産を維持しながら移行**します。

---

# Phase 0 現状固定（必須）

目的

* Runtime の責務を可視化
* Authority の分散を完全に洗い出す

実施

Runtime 全体について

* Publish 呼び出し
* Retire 呼び出し
* Crossfade 判定
* Atomic Load/Store
* RuntimeWorld 更新

を一覧化します。

成果物

```
Authority Matrix

Publish Authority

Retire Authority

Crossfade Authority

Snapshot Authority

Delete Authority
```

これは以後の改修の基準になります。

---

# Phase 1 RuntimeIntent層新設

現在

```
UI

Automation

Preset

IR Load
```

が直接 RuntimeBuilder を呼んでいます。

これを

```
RuntimeIntent
```

へ統一します。

例

```
ParameterIntent

PresetIntent

IRIntent

AutomationIntent

HostIntent
```

Builder は

```
Intent

↓

RuntimeWorld
```

だけになります。

期待効果

* UI依存除去
* 将来MIDI・OSC追加容易

---

# Phase 2 RuntimeBuilder純化

Builderから

* Publish
* Retire
* Crossfade
* RuntimeStore

への依存を除去します。

Builder責務

```
Build RuntimeWorld
```

だけ。

Builderは

```
RuntimeWorld
```

しか返さない。

---

# Phase 3 RuntimePublicationValidator強化

現状Validationを

```
Topology

Resource

Semantic

Projection

Latency

Routing

DSPGraph
```

まで拡張します。

Validation失敗時は

```
Publish禁止
```

を徹底します。

---

# Phase 4 RuntimeTransaction導入

新規

```
RuntimeTransaction
```

追加。

例

```
RuntimeIntent

↓

RuntimeWorld

↓

Validation

↓

CrossfadePlan

↓

RetirePlan
```

ここまでを

```
Transaction
```

として保持。

Coordinatorは

```
Commit()
```

だけになります。

途中状態をなくします。

---

# Phase 5 Coordinator一本化

最重要フェーズです。

現在

```
SnapshotCoordinator

RuntimeStore

Lifetime

Crossfade

DSPTransition
```

などにAuthorityがあります。

これを

```
RuntimeCoordinator
```

だけへ集約します。

Coordinatorだけが

```
Publish

Crossfade開始

Retire

Rollback
```

できます。

その他は

```
Coordinator.execute()
```

だけ。

---

# Phase 6 RuntimeWorld完全Immutable

RuntimeWorldを

```
const
```

だけで構成します。

禁止

```
mutable

lazy init

cache

optional initialize
```

です。

Build

↓

Freeze

↓

Publish

以降

変更禁止

---

# Phase 7 Crossfade完全分離

Crossfade Runtimeは

```
Decision
```

を持ちません。

Builder時に

```
CrossfadePlan
```

を生成します。

RTでは

```
execute(plan)
```

のみ。

---

# Phase 8 Retire Runtime統合

現在

Retire経路が複数あります。

これを

```
RetireIntent

↓

EpochDomain

↓

DeferredDelete
```

一本にします。

Deleteできる場所

```
DeletionWorker
```

のみ。

---

# Phase 9 RuntimeStore簡素化

RuntimeStore責務

```
Publish

Atomic Swap

Current Runtime取得
```

のみ。

禁止

```
Policy

Crossfade

Retire

Validation
```

---

# Phase 10 Runtime Health

HealthMonitorを

```
Detect

↓

Diagnose

↓

Recover

↓

Verify
```

へ拡張します。

例

```
Retire Stall

↓

Drain

↓

Advance Epoch

↓

Verify
```

---

# Phase 11 Shutdown Pipeline統一

Shutdownを

```
Stop Accept

↓

Stop Publish

↓

Stop Reader

↓

Drain Intent

↓

Drain Retire

↓

Advance Epoch

↓

Delete

↓

Verify Empty
```

へ統一します。

これにより終了時リークを完全排除できます。

---

# Phase 12 Runtime Capability

RuntimeWorldへ

```
Capability
```

追加。

例

```
SupportsMixedPhase

SupportsLinearPhase

SupportsTruePeak

SupportsBackgroundIR

SupportsOversampling
```

DSP側はCapabilityを見るだけになります。

---

# Phase 13 Runtime Generation

RuntimeWorldへ

```
GenerationID
```

追加。

例

```
Generation=125
```

Publish

Retire

Crossfade

Delete

Monitor

全てGeneration単位になります。

ログ解析が容易になります。

---

# 最終構造

最終的には責務を次のように固定します。

| コンポーネント                     | 唯一の責務                                     |
| --------------------------- | ----------------------------------------- |
| RuntimeIntent               | UI・Automation 等を RuntimeIntent に変換        |
| RuntimeBuilder              | RuntimeWorld 構築のみ                         |
| RuntimePublicationValidator | RuntimeWorld 検証のみ                         |
| RuntimeCoordinator          | Publish・Crossfade・Retire・Rollback の唯一の決定権 |
| RuntimeStore                | RuntimeWorld の保持・Atomic Publish のみ        |
| Audio Thread                | RuntimeWorld を読み取り DSP 実行のみ               |
| CrossfadeRuntime            | CrossfadePlan の実行のみ                       |
| EpochDomain                 | Retire・Reclaim のみ                         |
| DeletionWorker              | delete のみ                                 |
| HealthMonitor               | 観測・診断・回復・検証のみ                             |

## 実装優先度

私なら実装順は次のようにします。

1. **Phase 5（Coordinator一本化）** — Authority の重複を解消する最重要項目。
2. **Phase 8（Retire統合）** — 寿命管理を一本化し、RT安全性を強化。
3. **Phase 6（RuntimeWorld Immutable化）** — Publish 後の不変性を保証。
4. **Phase 4（RuntimeTransaction導入）** — Build→Validate→Publish を一つのトランザクションとして扱う。
5. **Phase 7（Crossfade分離）** — RT を「実行のみ」に近づける。
6. **Phase 1～3（Intent・Builder・Validator整理）** — 上位層の責務を整理。
7. **Phase 9～13（Store・Health・Shutdown・Capability・Generation）** — 保守性・監視性・拡張性を高める。

この順序であれば、既存の ConvoPeq の設計資産を活かしながら、Practical Stable ISR Bridge Runtime が目指す「RT は判断しない・所有しない・解放しない」という構造へ、段階的かつリスクを抑えて移行できます。
