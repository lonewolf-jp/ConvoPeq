# Runtime Coordinator Topology Decision（Phase0.5成果物）

作成日: 2026-05-31
対象: Publication 実行系 / ISR ガバナンス系の二層Coordinator

---

## 1. 目的

Phase1（Publication Authority Collapse）着手前に、Coordinator 二層構造の将来像を固定し、実装中の責務衝突を防止する。

---

## 2. 現状トポロジー（As-Is）

```text
Publication Executor Layer
  convo::RuntimePublicationCoordinator<World, Handle, Bridge>
      └ publishState() / RuntimeStore::publishAndSwap()

Governance & Telemetry Layer
  convo::isr::RuntimePublicationCoordinator (runtimePublicationBridge_)
      └ precheckPublish(), commit(), retire(), isFullyDrained(), state/backlog tracking
```

---

## 3. 実コード証跡（抜粋）

### 3.1 実行レイヤ

- `src/core/RuntimePublicationCoordinator.h`
  - `publishState(...)` で world を生成・検証・`publishAndSwap` 実行

### 3.2 ガバナンス/テレメトリレイヤ

- `src/audioengine/ISRRuntimePublicationCoordinator.h`
  - `precheckPublish`, `commit`, `retire`, `isFullyDrained`, `isSwapPending`, `getState`, `lastRejectReason`
- `src/audioengine/AudioEngine.h`
  - `precheckRuntimePublication(...) -> runtimePublicationBridge_.precheckPublish(...)`
  - `commitRuntimePublication(...) -> runtimePublicationBridge_.commit(...)`
  - `retireRuntimePublication(...) -> runtimePublicationBridge_.retire(...)`
- `src/audioengine/AudioEngine.Threading.cpp`
  - `isFullyDrained()` が `runtimePublicationBridge_.isFullyDrained()` を使用

---

## 4. 意思決定オプション

### Option A: 二層維持（責務固定）

- 内容:
  - 実行権威: `convo::RuntimePublicationCoordinator`
  - ガバナンス/圧力/shutdown判定: `convo::isr::RuntimePublicationCoordinator`
- 長所:
  - 既存の ISR 計測・防護を活かせる
  - Phase1 の変更範囲が限定される
- 短所:
  - 命名が同名で混乱しやすい
  - API境界が曖昧だと再拡散しやすい

### Option B: 完全統合（core側に集約）

- 内容:
  - ISR runtimePublicationBridge_ の責務を core coordinator へ吸収
- 長所:
  - Coordinator 1 本化の説明が明快
- 短所:
  - 変更影響が大きく、Phase1と同時実施すると破綻リスクが高い

### Option C: ISR側縮退（precheck/stateのみ残す）

- 内容:
  - ISR側の commit/retire APIを段階縮退し、state/pressure判定のみに限定
- 長所:
  - 将来統合へ滑らかに移行可能
- 短所:
  - 一時的に二層＋互換層が残る

---

## 5. 採用決定（To-Be）

採用方針: Option A（短中期） + Option C（中長期移行）

理由:

1. 現状、`isFullyDrained` や pressure/backlog 連携が ISR 層に存在し、即時統合（Option B）はPhase1と干渉する。
2. Phase1 の第一目的は publish 実行権威の単一路化であり、Coordinator統合の大改修を同時に行うべきではない。
3. まず責務境界を固定し、後続フェーズ（Phase6/7以降）で ISR側 API を縮退するのが安全。

---

## 6. 責務境界（固定）

### 6.1 実行権威（Executor）

主体: `convo::RuntimePublicationCoordinator`

責務:

- `publishState()` の実行
- `RuntimeStore::publishAndSwap()` の唯一実行経路
- old world retire hook 呼び出し

禁止:

- 直接的な pressure policy / rollback policy 判断（ISR層の責務）

### 6.2 ガバナンス権威（Governance/Telemetry）

主体: `convo::isr::RuntimePublicationCoordinator`（`runtimePublicationBridge_`）

責務:

- `precheckPublish()` による入力妥当性チェック
- publication/retire backlog, fallback, reclaim-in-flight の計測状態管理
- `isFullyDrained()` / shutdown lifecycle 状態判定
- reject reason / coordinator state の観測インタフェース

禁止:

- `RuntimeStore` へ直接 publish/swap すること
- 実行権威（world所有権遷移）を持つこと

### 6.3 Rollback責務境界

主体: `RetireRuntimeEx`

- rollback 可否判断（`canRollback`）は Retire ドメイン責務
- publication coordinator は rollback判定の主語を持たない

---

## 7. Phase1開始条件（必須ゲート条件）

以下が満たされるまで Phase1 を開始しない。

1. 本ドキュメントの責務境界に合意済み
2. 実行権威は `publishState()` に限定する方針を確定
3. `runtimePublicationBridge_` が RuntimeStore 実行権威を持たないことを確認
4. rollback 主語は `RetireRuntimeEx` と明記済み

---

## 8. Phase1終了判定への接続

- Exit-A: `publishState()` callsite（Production Runtime Tree）= 1
- Exit-B: `prepareCommit` / `executeCommit` / `commitNewDSP` 到達不能

補助確認:

- `runtimePublicationBridge_` は precheck/state/drain/shutdown のみで利用されること
- world swap 実行は core coordinator 以外から呼ばれないこと

---

## 9. 後続フェーズへの申し送り

- Phase6/7 で Coordinator 役割再評価を実施
- 必要なら ISR coordinator の commit/retire API を縮退し、観測/判定専用へ移行
- 最終形（Phase10）では「実行権威 1、ガバナンス責務は明示分離」の契約を CI で検証する

---

## 10. 要約

Phase0.5 の決定として、**実行権威は core coordinator、ガバナンス/圧力/drainは ISR coordinator** に固定する。
この境界固定を Phase1 の開始条件とすることで、Authority Collapse 中の再拡散リスクを抑制する。
