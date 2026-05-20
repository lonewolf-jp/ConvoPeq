# ConvoPeq ISR Retire Authority Graph

## 目的

本書は、ConvoPeq の ISR 移行で最重要となる **retire/reclaim authority の単一化** を定義する。

狙い:

- ownership-bearing object の retire 実行者を一意にする
- destroy 経路を一本化する
- epoch/grace 判定の責務を明確化する
- UAF（特に `activeDSP` 系）の再発防止

---

## 用語

- **Retire authority**: 「この object を retire enqueue してよい唯一の主体」
- **Reclaim authority**: 「grace 条件を満たした後に destroy を実行する主体」
- **Ownership closure**: lifetime 責務が複数経路へ漏れない状態

---

## 設計原則

1. 1 object family = 1 retire authority
2. retire と reclaim を分離（enqueue と destroy の段階分離）
3. reclaim は epoch/grace 条件を満たした場合のみ実行
4. Audio Thread は retire/reclaim authority を持たない
5. authority identity と implementation detail を分離する

運用注記（REV3.1 優先）:

- authority 制約は capability-first（Publish/Retire/Shutdown）を優先する。
- 本書で記述する `RuntimeWorldRetireManager` は、上記 root 配下の
   **retire/reclaim 実装委譲コンポーネント名**として扱う。

### REV3.2運用優先注記

- 本書の authority/queue/worker 分解は設計参照表現として扱う。
- 実装運用は `plan5.md` REV3.2 を優先し、
   `runtime exposes evidence / CI validates evidence` を固定方針とする。
- 解釈衝突時は capability-first を優先し、
   runtime coordinator lifecycle は導入せず、`RuntimeWorldRetireManager` は実装委譲として解釈する。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。

---

## Authority Graph（論理図）

```text
[Message Thread / Publish Path]
   |
   | retire enqueue (authority)
   v
[Retire Authority Component]
   |
   | grace/epoch check delegation
   v
[Epoch / Grace Evaluator]
   |
   | reclaim eligible
   v
[Reclaim Authority Component]
   |
   | destroy callback
   v
[Object Destruction]
```

---

## Object Family 別 authority 定義（確定版 v2）

| Object Family | Retire Authority | Reclaim Authority | Epoch/Grace | Notes |
| --- | --- | --- | --- | --- |
| GlobalSnapshot | SnapshotCoordinator（via SnapshotRetireManager） | SnapshotRetireManager | 必須 | 既存系統を canonical とする |
| RuntimePublication payload | RuntimeWorldRetireManager | RuntimeWorldRetireManager | 必須（共有Epoch） | runtime-only publish world として管理 |
| DSP lifetime tokens | RuntimeWorldRetireManager | RuntimeWorldRetireManager | 必須（共有Epoch） | `activeDSP/currentDSP/fadingOutDSP/queuedOldDSP` を handle 化して統合 |
| RT execution-local state | N/A | N/A | 不要 | Audio Thread private、retire不要 |
| Telemetry state | channel owner（必要時のみ） | N/A（通常 destroy不要） | 原則不要 | lifecycle が runtime と同一なら runtime 終了で解放 |

---

## DSP 系（課題ドメイン）

現状課題:

- `activeDSP`
- `currentDSP`
- `fadingOutDSP`
- `queuedOldDSP`

が分散し、retire authority が曖昧。

確定方針:

- DSP pointer を直接 retire 判断しない
- `DSPHandle`（slotIndex / generation / retireAuthorityId）で管理
- retire enqueue は **RuntimeWorldRetireManager** のみ実行
- reclaim は共有 EpochDomain の grace 条件に従って RuntimeWorldRetireManager が実行

推奨概念:

```text
DSPHandle {
   slotIndex
   generation
   retireAuthorityId
}
```

---

## Authority と実装分離（manager肥大化対策）

目的:

- RuntimeWorldRetireManager への責務過積載を防ぎつつ、authority 単一性を維持する

規則（確定）:

1. **authority source-of-truth** は capability-first を優先（runtime coordinator lifecycle 非導入）
2. RuntimeWorldRetireManager は retire/reclaim の実装委譲 identity として運用する
3. enqueue/destroy の具体実装は family ごとに sub-component へ委譲可
4. 委譲先は authority を再定義してはならない
5. 外部コンポーネントが独自 retire queue を持つことを禁止

許可される形:

- RetireAuthority（type-level capability source-of-truth）
- RuntimeWorldRetireManager（retire/reclaim 実装委譲 identity）
- DSPFamilyRetireWorker（実装委譲）
- RuntimePayloadReclaimWorker（実装委譲）

禁止される形:

- DSP module 独自の retire authority 宣言
- callback 側での shortcut retire queue

---

## 禁止事項

- Audio Thread から retire enqueue を実行する
- 同一 object family に複数の retire 経路を作る
- grace 判定前に destroy を実行する
- token なし raw pointer retire を許可する

---

## 完了条件（Gate）

- [x] 全 object family で retire authority が一意
- [x] reclaim authority が明示されている
- [x] destroy callback 経路が文書化済み
- [x] DSP family が token 化方式で定義済み
- [ ] bug2 系 UAF シナリオに対する防止証跡（実装後検証）

---

## 参照

- `doc/work/plan5.md`
- `doc/work/ISR_Runtime_State_Matrix.md`
- `doc/work/ISR_DSPCore_Decomposition_Analysis.md`
- `doc/work/ISR_Runtime_Ownership_Graph_完全可視化.md`
