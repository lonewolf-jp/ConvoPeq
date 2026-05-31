# Practical Stable ISR Bridge Runtime 統合マスタープラン

作成日: 2026-05-31
対象: ConvoPeq / `src/audioengine`, `src/convolver`, `src/eqprocessor`, `src/core`
目的: 監査・再レビュー結果を統合し、実装途中の破綻リスクを最小化する順序で移行計画を固定する。

---

## 0. 固定前提

以下 7 項目は、現行コードベース上の設計前提として固定する。

1. `RuntimeState = RuntimeWorld` として扱う（新規 `RuntimeWorld` 型は作らない）。
2. 最優先は `Publication Authority Collapse`。
3. `Phase0.5 Coordinator Topology Decision` は **Phase1 開始条件** とする。
4. `BuildSnapshot` の逆依存は `Convolver` 統合より前（Phase8-A）に崩す。
5. `Convolver` 統合（Phase8-B）後に `RuntimeState` 自己完結化（Phase2）を行う。
6. `DSP Selection Migration` は設計承認ゲート（Phase5-Gate）を必須化する。
7. `RetireManager` は既存移行ではなく **新規設計案件** として扱う。

---

## 1. 調査結果（2026-05-31 再採取）

### 1.1 判定対象スコープ

本計画の機械判定対象（Production Runtime Tree）を以下に固定する。

- `src/audioengine/**`
- `src/convolver/**`
- `src/eqprocessor/**`
- `src/core/**`

`src/tests/**`, `doc/**` は完了条件の grep 判定対象から除外する。

### 1.2 主要証跡サマリ

| 観点 | 現状 | 主要証跡 |
| --- | --- | --- |
| `publishState()` callsite | 9（audioengine 本番 `.cpp`） | `AudioEngine.Commit.cpp`, `AudioEngine.Processing.PrepareToPlay.cpp`, `AudioEngine.Processing.ReleaseResources.cpp`, `AudioEngine.Timer.cpp` |
| commit 系エントリ | 3 存在 | `AudioEngine::prepareCommit`, `AudioEngine::executeCommit`, `AudioEngine::commitNewDSP` in `AudioEngine.Commit.cpp` |
| `transition.active` 実行分岐 | 5 | `AudioEngine.Processing.AudioBlock.cpp`, `BlockDouble.cpp`, `Snapshot.cpp`, `AudioEngine.Timer.cpp` |
| RuntimeBuilder の BuildSnapshot 依存 | 2（宣言+実装） | `RuntimeBuilder.h: build(..., ConvolverProcessor::BuildSnapshot)`, `RuntimeBuilder.cpp` 実装 |
| EQ 側 retire fallback | 複数実使用 | `EQProcessor.h` と `EQProcessor.Core.cpp` の `deferredDeleteFallbackQueue` / `drainDeferredDeleteFallbackQueue` |

---

## 2. 既存計画からの差分

### 2.1 追加事項（採用）

1. `Phase0.5` を必須ゲート化（成果物: `runtime-coordinator-topology-decision.md`）。
2. `Phase1` 完了条件を 2 段化（Exit-A / Exit-B）。
3. `Phase5-Gate` を追加（DSP Selection State Machine 設計承認）。
4. `Phase8-A` の出口条件を明文化（RuntimeBuilder 依存崩し）。
5. `Phase0` に Architecture Inventory を統合。

### 2.2 削除事項（採用）

1. 「新規 `RuntimeWorld` 型を作る」記述。
2. 「TransitionState を先に削除する」表現（責務移設後に診断化へ置換）。
3. 「Retire 統合」単独表現（設計→実装→移行へ置換）。

### 2.3 修正事項（採用）

1. 順序を `Phase8-A -> Phase8-B -> Phase2` に固定。
2. grep 条件を Production Runtime Tree 限定に修正。
3. `publishState() callsite = 1` を「本番ツリー限定」に修正。
4. `canRollback` を `RetireRuntimeEx` 側責務として切り分け。

---

## 3. 採用版マスタープラン（固定版）

```text
Phase0
Authority Inventory + Architecture Inventory

↓

Phase0.5
Coordinator Topology Decision（Phase1 開始条件）

↓

Phase1
Publication Authority Collapse
  Exit-A: publishState() callsite = 1
  Exit-B: prepareCommit/executeCommit/commitNewDSP 到達不能

↓

Phase8-A
BuildSnapshot Dependency Collapse
  Exit-1: RuntimeBuilder が ConvolverProcessor 型を参照しない
  Exit-2: RuntimeBuilder.* から BuildSnapshot 参照消滅

↓

Phase8-B
Convolver Runtime Integration
  Exit: PendingParams / PreparedIRState / SafeStateSwapper = 0

↓

Phase2
RuntimeState Self-contained

↓

Phase3
AudioEngine Non-Authority

↓

Phase4
Snapshot Classification

↓

Phase5-Gate
DSP Selection State Machine 設計承認

↓

Phase5
DSP Selection Migration

↓

Phase6
Generation Authority Collapse

↓

Phase7
Retire Governance Singularity

↓

Phase9
Legacy Semantic Purge

↓

Phase10
Contract Enforcement
```

---

## 4. フェーズ詳細設計

### 4.1 Phase0: Authority + Architecture Inventory

- 目的

- 権威経路と構造経路を同時に棚卸しし、改修対象を固定する。

- 入力

- `src/audioengine/**`, `src/convolver/**`, `src/eqprocessor/**`, `src/core/**`

- 主要タスク

- Authority 抽出: `commit|publish|retire|build|generation|transition|snapshot`
- Architecture 抽出: `SafeStateSwapper|BuildSnapshot|PendingParams|PreparedIRState|TransitionState|GlobalSnapshot`
- 呼び出しグラフ生成（commit/publish/retire 主経路）

- 成果物

- `authority_inventory.md`
- `architecture_inventory.md`

- 出口条件

- 主要シンボルの owner / caller / lifecycle が列挙済み。

### 4.2 Phase0.5: Coordinator Topology Decision（必須ゲート）

- 目的

- `convo::RuntimePublicationCoordinator`（実行）と `convo::isr::RuntimePublicationCoordinator`（状態追跡/判定補助）の将来像を固定する。

- 検討軸

- Publish Authority
- Drain / Shutdown Authority
- Reject Reason / Precheck
- SwapPending / FullyDrained
- Rollback 連携境界（`RetireRuntimeEx` との責務分離）

- 成果物

- `runtime-coordinator-topology-decision.md`

- Phase1 開始条件

- 上記成果物で「保持/統合/削除」の方針が確定していること。

### 4.3 Phase1: Publication Authority Collapse

- 目的

- publish 経路を単一路化し、commit 旧経路を実行上無効化する。

- 主要タスク

- `publishState()` 呼び出し点を 1 箇所へ集約。
- `prepareCommit` / `executeCommit` / `commitNewDSP` の実行到達を遮断。
- 既存 shutdown / drain 経路との整合を維持。

- Exit-A

- `publishState()` callsite = 1（Production Runtime Tree）

- Exit-B

- `prepareCommit` / `executeCommit` / `commitNewDSP` が呼び出しグラフ上で到達不能。

- リスク

- 到達不能化前に削除すると回帰調査が困難。先に非到達化、後に削除。

### 4.4 Phase8-A: BuildSnapshot Dependency Collapse

- 目的

- RuntimeBuilder から Convolver 型依存を切離し、逆依存を除去する。

- 主要タスク

- `RuntimeBuilder::build(..., ConvolverProcessor::BuildSnapshot)` の置換 API を導入。
- Builder 入力を `BuildInput` + Runtime 側 DTO へ再定義。

- Exit-1

- RuntimeBuilder が `ConvolverProcessor` 型を参照しない。

- Exit-2

- `RuntimeBuilder.h`, `RuntimeBuilder.cpp` から `BuildSnapshot` 参照消滅。

- リスク

- 互換 API 期間を短く保たないと二重経路が固定化する。

### 4.5 Phase8-B: Convolver Runtime Integration

- 目的

- Convolver 内の Runtime 外 SoT を除去し、Runtime 統合を完了する。

- 主要タスク

- `PendingParams` 排除。
- `PreparedIRState` 排除。
- `SafeStateSwapper` 排除。
- RuntimeState 読み出し中心へ切替。

- Exit

- 上記 3 シンボルが Production Runtime Tree で 0。

- リスク

- IR ロード・キャッシュ・再適用シーケンスの順序崩れに注意。

### 4.6 Phase2: RuntimeState Self-contained

- 目的

- `RuntimeState` 単体で実行可能な閉包を成立させる。

- 出口条件

- `processBlock()` に必要な実行情報が RuntimeState 由来で完結（外部 mutable SoT なし）。

- 注記

- Phase8-A/B 未完了では本フェーズは成立しない。

### 4.7 Phase3: AudioEngine Non-Authority

- 目的

- AudioEngine を Observe/Process/Measure に限定する。

- 出口条件

- Authority 操作（commit/publish/retire/build/activate）の主語が AudioEngine から外れている。

### 4.8 Phase4: Snapshot Classification

- 目的

- Snapshot を Authority / Projection に完全分類し、Authority を撤去する。

- 出口条件

- `snapshot.(generation|version|active)` が実行分岐を持たない。

### 4.9 Phase5-Gate: DSP Selection State Machine 承認

- 目的

- `transition.active` 依存除去前に、代替状態機械を設計承認する。

- 必須成果物

- 状態図（Stable / Entering / Retiring など）
- AudioThread 分岐置換表
- 互換期間のフェイルセーフ

- Phase5 開始条件

- 設計レビュー承認済み。

### 4.10 Phase5: DSP Selection Migration

- 目的

- DSP 選択責務を TransitionState から新 state machine へ移す。

- 出口条件

- AudioThread で `transition.active` を参照せず同等動作。

### 4.11 Phase6: Generation Authority Collapse

- 目的

- generation の権威源を 1 本化し、他は diagnostic mirror 化する。

- 出口条件

- Authority generation source = 1。
- それ以外は診断用途であることが明文化されている。

### 4.12 Phase7: Retire Governance Singularity

- 目的

- Retire 経路を単一統治へ移す（新規 RetireManager 設計を含む）。

- 主要タスク

- `RetireManager` 設計
- AudioEngine 側退避経路統合
- EQProcessor 側 fallback queue 統合

- 出口条件

- Retire authority = 1。

### 4.13 Phase9: Legacy Semantic Purge

- 目的

- 互換ブリッジや暫定セマンティクスを除去。

- 出口条件

- `Legacy|Mutable|Temporary` の残存を方針どおり解消。

### 4.14 Phase10: Contract Enforcement

- 目的

- 再発防止を CI 契約として固定。

- 最低ルール

- `AudioEngine::commit*` 禁止
- `AudioEngine::publish*` 禁止
- `AudioEngine::retire*` 禁止
- `publishState()` 呼び出し元 1
- RuntimeState 外 mutable state 監査
- Snapshot 実行分岐禁止

---

## 5. 完了条件（機械検証版）

### 5.1 既存 C1〜C10（要旨）

- C1: `publishState` 呼び出し元 = 1
- C2: `commitRuntimePublication` 削除/非使用
- C3: `retireRuntimePublication` 削除/非使用
- C4: AudioEngine に commit/publish/retire/build/activate が存在しない
- C5: RuntimeState 単独で DSP 実行可能
- C6: TransitionState が実行分岐に使われない
- C7: Generation authority = 1
- C8: Retire authority = 1
- C9: Snapshot authority = 0
- C10: Runtime authority = RuntimeState のみ

### 5.2 追加 C11〜C15（スコープ固定版）

- C11: `SafeStateSwapper` in Production Runtime Tree = 0
- C12: `PendingParams` in Production Runtime Tree = 0
- C13: `PreparedIRState` in Production Runtime Tree = 0
- C14: `publishState()` callsite in Production Runtime Tree = 1
- C15: `EQProcessor::deferredDeleteFallbackQueue` in Production Runtime Tree = 0

---

## 6. 直近実行チェックリスト

1. Phase0/0.5 成果物 2 点を先に作成
   - `authority_inventory.md`
   - `runtime-coordinator-topology-decision.md`
2. Phase1 Exit-A/B の測定スクリプトを先に作成
3. Phase8-A の API 置換計画を先に作成
4. Phase5-Gate 設計レビューを先に実施
5. Phase7 の RetireManager 契約を先に定義

---

## 7. 監査要約

本計画は、方向性評価の段階から、開始条件・フェーズゲート・退出条件・機械検証条件が揃った実装可能計画へ収束した。
特に、`Phase0.5` 必須化、`Phase1` 2段退出、`Phase5-Gate`、`Phase8-A` 明確出口条件が、AI 実装時の破綻確率を下げる主要因である。
