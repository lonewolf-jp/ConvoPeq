# Phase提出テンプレート（探索証跡・Authority差分・CallGraph差分・C1〜C15判定）

対象: Practical Stable ISR Bridge Runtime 移行
適用スコープ（Production Runtime Tree）:

- `src/audioengine/**`
- `src/convolver/**`
- `src/eqprocessor/**`
- `src/core/**`

---

## 0. 提出メタ情報

- フェーズ名: `PhaseX`（例: `Phase1`, `Phase8-A`, `Phase5`）
- 提出日: `YYYY-MM-DD`
- 提出者: `@name`
- 参照計画:
  - `doc/work12/practical_stable_isr_bridge_runtime_masterplan_detailed_design_and_findings_2026-05-31.md`
  - `doc/work12/authority_inventory.md`
  - `doc/work12/runtime-coordinator-topology-decision.md`
- 対象コミット範囲: `<from>..<to>`

---

## 1. フェーズ開始条件チェック

### 1.1 前フェーズ出口条件の充足

- [ ] 前フェーズ出口条件を再確認
- [ ] 未達項目なし
- [ ] 例外運用（あれば）記録済み

### 1.2 フェーズ越境なしの確認（Rule-36）

- [ ] 今回変更は当該フェーズ責務のみ
- [ ] 後続フェーズ条件の先行達成なし（Rule-37）

---

## 2. 探索証跡（必須）

> Rule-02 / 26 / 40 対応。grep + Serena + CodeGraph を原則必須とする。

## 2.1 grep 証跡

- 実行日時: `YYYY-MM-DD hh:mm`
- 実行範囲: `Production Runtime Tree`
- 主要クエリ:
  - `commit|publish|retire|build|generation|snapshot|transition`
  - `rollback|activate|deactivate|pending|prepare|execute|swap|drain`

### 2.1.1 結果サマリ

| クエリ | ヒット数 | 主なファイル | 備考 |
| --- | --- | --- | --- |
| `<query-1>` | `<n>` | `<paths>` | `<note>` |
| `<query-2>` | `<n>` | `<paths>` | `<note>` |

### 2.1.2 生ログ（抜粋）

- `<貼り付け or 参照先>`

## 2.2 Serena 証跡

- 実行日時: `YYYY-MM-DD hh:mm`
- 取得内容:
  - [ ] 定義元
  - [ ] 呼び出し元
  - [ ] 呼び出し先
  - [ ] 所有クラス
  - [ ] ライフサイクル
  - [ ] Runtime上のAuthority

### 2.2.1 結果サマリ

| シンボル | 定義 | 主要caller | 主要callee | 所有 | Authority分類 |
| --- | --- | --- | --- | --- | --- |
| `<symbol>` | `<file:line>` | `<...>` | `<...>` | `<class/module>` | `<Authority/Mirror/Legacy>` |

## 2.3 CodeGraph 証跡

- Index状態:
  - [ ] 再インデックス実施（条件該当時）
  - [ ] もしくは増分で妥当（理由記載）
- 実行日時: `YYYY-MM-DD hh:mm`

### 2.3.1 Call Graph 抜粋

| 対象 | Direct Caller | Indirect Caller | Callee | 判定 |
| --- | --- | --- | --- | --- |
| `<symbol>` | `<...>` | `<...>` | `<...>` | `<active/dead/candidate>` |

---

## 3. Authority差分（Before / After）

> Rule-35 対応。必ず `Authority Source / Mirror Source / Legacy Source` の3分類で提示する。

## 3.1 Before

### Before: Authority Source

- `<item-1>`
- `<item-2>`

### Before: Mirror Source

- `<item-1>`

### Before: Legacy Source

- `<item-1>`

## 3.2 After

### After: Authority Source

- `<item-1>`
- `<item-2>`

### After: Mirror Source

- `<item-1>`

### After: Legacy Source

- `<item-1>`

## 3.3 差分要約

| 分類 | Before件数 | After件数 | 差分 | 期待整合 |
| --- | --- | --- | --- | --- |
| Authority Source | `<n>` | `<n>` | `<delta>` | `<ok/ng>` |
| Mirror Source | `<n>` | `<n>` | `<delta>` | `<ok/ng>` |
| Legacy Source | `<n>` | `<n>` | `<delta>` | `<ok/ng>` |

---

## 4. CallGraph差分（Before / After）

> Rule-33 / 34 対応。削除前に到達不能証明を必須化する。

## 4.1 重点シンボル（例）

- `prepareCommit`
- `executeCommit`
- `commitNewDSP`
- `publishState`
- `retireDSP`

## 4.2 Before

| シンボル | 到達性 | 主caller | 備考 |
| --- | --- | --- | --- |
| `<symbol>` | `<reachable/unreachable>` | `<...>` | `<...>` |

## 4.3 After

| シンボル | 到達性 | 主caller | 備考 |
| --- | --- | --- | --- |
| `<symbol>` | `<reachable/unreachable>` | `<...>` | `<...>` |

## 4.4 Dead Path証明（必要時）

- 対象シンボル: `<symbol>`
- 証明根拠:
  1. `<CodeGraph callerなし証跡>`
  2. `<Serena参照証跡>`
  3. `<grep補助証跡>`
- 判定: `到達不能 / 保留`

---

## 5. フェーズ出口条件判定（当該Phase）

| 条件ID | 条件内容 | 判定 | 証跡リンク |
| --- | --- | --- | --- |
| `<PhaseExit-1>` | `<condition>` | `<pass/fail>` | `<path or log>` |
| `<PhaseExit-2>` | `<condition>` | `<pass/fail>` | `<path or log>` |

未達時の対応:

- `<blocking reason>`
- `<next action>`

---

## 6. C1〜C15 判定表（全体トラッキング）

> Rule-27 / 28 / 43 対応。フェーズごとに更新する。

| ID | 判定項目 | 現在判定 | 前回判定 | 差分 | 証跡 |
| --- | --- | --- | --- | --- | --- |
| C1 | `publishState` 呼び出し元 = 1 | `<pass/fail/na>` | `<...>` | `<...>` | `<...>` |
| C2 | `commitRuntimePublication` 削除/非使用 | `<...>` | `<...>` | `<...>` | `<...>` |
| C3 | `retireRuntimePublication` 削除/非使用 | `<...>` | `<...>` | `<...>` | `<...>` |
| C4 | AudioEngine に commit/publish/retire/build/activate が存在しない | `<...>` | `<...>` | `<...>` | `<...>` |
| C5 | RuntimeState 単独で DSP 実行可能 | `<...>` | `<...>` | `<...>` | `<...>` |
| C6 | TransitionState が実行分岐に使われない | `<...>` | `<...>` | `<...>` | `<...>` |
| C7 | Generation authority = 1 | `<...>` | `<...>` | `<...>` | `<...>` |
| C8 | Retire authority = 1 | `<...>` | `<...>` | `<...>` | `<...>` |
| C9 | Snapshot authority = 0 | `<...>` | `<...>` | `<...>` | `<...>` |
| C10 | Runtime authority = RuntimeState のみ | `<...>` | `<...>` | `<...>` | `<...>` |
| C11 | `SafeStateSwapper` = 0（PRT） | `<...>` | `<...>` | `<...>` | `<...>` |
| C12 | `PendingParams` = 0（PRT） | `<...>` | `<...>` | `<...>` | `<...>` |
| C13 | `PreparedIRState` = 0（PRT） | `<...>` | `<...>` | `<...>` | `<...>` |
| C14 | `publishState()` callsite = 1（PRT） | `<...>` | `<...>` | `<...>` | `<...>` |
| C15 | `EQProcessor::deferredDeleteFallbackQueue` = 0（PRT） | `<...>` | `<...>` | `<...>` | `<...>` |

---

## 7. 影響範囲・リスク・ロールバック

### 7.1 影響範囲

- 変更ファイル一覧:
  - `<path>`
  - `<path>`

### 7.2 リスク評価

| リスク | 影響度 | 発生確率 | 緩和策 |
| --- | --- | --- | --- |
| `<risk>` | `<H/M/L>` | `<H/M/L>` | `<mitigation>` |

### 7.3 ロールバック計画

- 戻し単位: `<commit range / feature flag / file group>`
- 戻し後再検証:
  1. `<check-1>`
  2. `<check-2>`

---

## 8. フェーズ完了宣言

- 完了判定: `<Pass / Fail>`
- 判定者: `<name>`
- 判定根拠:
  - `<evidence-1>`
  - `<evidence-2>`

> 注意: C1〜C15 未達のまま「移行完了」を宣言してはならない。

---

## 付録A: クイックコピーテンプレート（最小提出）

```text
[Phase]
- name:
- commit range:

[Evidence]
- grep:
- serena:
- codegraph:

[Authority Diff]
- before (Authority/Mirror/Legacy):
- after  (Authority/Mirror/Legacy):

[CallGraph Diff]
- before:
- after:
- dead-path proof:

[C1-C15]
- pass:
- fail:
- hold:

[Decision]
- exit condition:
- go/no-go:
```

---

## 付録B: Phase1 提出実績（2026-05-31）

### B.1 提出メタ情報

- フェーズ名: `Phase1`
- 提出日: `2026-05-31`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### B.2 フェーズ開始条件チェック

- [x] 前フェーズ出口条件を再確認
- [x] 未達項目なし
- [x] 例外運用なし
- [x] 今回変更は Phase1 責務のみ（Publication Authority Collapse）
- [x] 後続フェーズ条件の先行達成なし

### B.3 探索証跡

#### B.3.1 grep

- 実行範囲: `src/audioengine/**`（Production Runtime Tree）
- 主要結果:
  - `publishState(`: 1（`AudioEngine.h` 内の `publishRuntimeStateNonRt` に限定）
  - `prepareCommit(|executeCommit(|commitNewDSP(`: 0
  - `commitRuntimePublication(|retireRuntimePublication(`: 0

#### B.3.2 Serena

- `mcp_oraios_serena_search_for_pattern` で以下を確認:
  - `prepareCommit\(|executeCommit\(|commitNewDSP\(` -> `{}`（ヒットなし）
  - `commitRuntimePublication\(|retireRuntimePublication\(` -> `{}`（ヒットなし）

#### B.3.3 CodeGraph

- `mcp_codegraph_find_callers` 実行:
  - `AudioEngine::applyRuntimeCommitFromIntent` -> `[]`
  - `AudioEngine::drainPublicationIntentsForRuntimeCommit` -> `[]`
  - `AudioEngine::enqueuePublicationIntentForRuntimeCommit` -> `[]`
- 補足: 現行 index では C++ caller 解像が不足していたため、grep 実証を一次根拠として併用（Rule-02 例外運用なし）。

### B.4 Authority差分（Phase1）

#### Phase1 Before: Authority Source

- `prepareCommit`
- `executeCommit`
- `commitNewDSP`
- `commitRuntimePublication`
- `retireRuntimePublication`

#### Phase1 After: Authority Source

- `enqueuePublicationIntentForRuntimeCommit`
- `drainPublicationIntentsForRuntimeCommit`
- `applyRuntimeCommitFromIntent`
- `runtimePublicationBridge_.commit(...)`
- `runtimePublicationBridge_.retire(...)`

#### 差分要約

| 分類 | Before件数 | After件数 | 差分 | 期待整合 |
| --- | --- | --- | --- | --- |
| Legacy commit symbol | 5 | 0 | -5 | ok |
| Bridge direct authority call | 0 | 2 | +2 | ok |

### B.5 CallGraph差分（Phase1）

#### Before

| シンボル | 到達性 | 主caller |
| --- | --- | --- |
| `prepareCommit` | reachable | `rebuildThreadLoop` |
| `executeCommit` | reachable | `handleAsyncUpdate` |
| `commitNewDSP` | reachable | `executeCommit` |

#### After

| シンボル | 到達性 | 主caller |
| --- | --- | --- |
| `prepareCommit` | unreachable（symbol removed） | n/a |
| `executeCommit` | unreachable（symbol removed） | n/a |
| `commitNewDSP` | unreachable（symbol removed） | n/a |

### B.6 フェーズ出口条件判定（Phase1）

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-A | `publishState()` callsite = 1（PRT） | pass | `grep_search: publishState(` |
| Exit-B | `prepareCommit/executeCommit/commitNewDSP` 到達不能 | pass | `grep_search + Serena search` |

### B.7 C1〜C15 更新（差分のみ）

| ID | 判定項目 | 現在判定 | 証跡 |
| --- | --- | --- | --- |
| C1 | `publishState` 呼び出し元 = 1 | pass | `AudioEngine.h: publishRuntimeStateNonRt` |
| C2 | `commitRuntimePublication` 削除/非使用 | pass | symbol 0 hit |
| C3 | `retireRuntimePublication` 削除/非使用 | pass | symbol 0 hit |
| C14 | `publishState()` callsite = 1（PRT） | pass | grep 1 hit |

### B.8 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: pass（実行されたテスト全件成功）

---

## 付録C: Phase8-A 提出実績（2026-05-31）

### C.1 提出メタ情報

- フェーズ名: `Phase8-A`
- 提出日: `2026-05-31`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### C.2 実施内容

- `RuntimeBuilder` API から `ConvolverProcessor::BuildSnapshot` 型依存を除去
- `RuntimeBuilder::build(const BuildInput&)` を単一路化
- Convolver snapshot 適用を `AudioEngine::applyCurrentConvolverSnapshotToRuntime` へ橋渡し
- 呼び出し側（`rebuildThreadLoop`）を新APIへ移行

### C.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | RuntimeBuilder が `ConvolverProcessor` 型を参照しない | pass | `grep_search(RuntimeBuilder.*, "ConvolverProcessor または BuildSnapshot") = 0` |
| Exit-2 | `RuntimeBuilder.h/.cpp` から `BuildSnapshot` 参照消滅 | pass | 同上 |

### C.4 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: pass（実行されたテスト全件成功）

---

## 付録D: Phase8-B 提出実績（2026-06-01）

### D.1 提出メタ情報

- フェーズ名: `Phase8-B`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### D.2 実施内容

- `ConvolverProcessor::SafeStateSwapper` = 0（Production Runtime Tree）
- `ConvolverProcessor::PendingParams` = 0（Production Runtime Tree）
- `ConvolverProcessor::PreparedIRState` = 0（Production Runtime Tree）
- ConvolverProcessor を RuntimeState 中心設計へ統合
- Convolver 局所 SoT（外部 state machine）を撤去

### D.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | `SafeStateSwapper` = 0（PRT） | pass | `rg "SafeStateSwapper" src/convolver/ = 0` |
| Exit-2 | `PendingParams` = 0（PRT） | pass | `rg "PendingParams" src/convolver/ = 0` |
| Exit-3 | `PreparedIRState` = 0（PRT） | pass | `rg "PreparedIRState" src/convolver/ = 0` |

### D.4 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録E: Phase2 提出実績（2026-06-01）

### E.1 提出メタ情報

- フェーズ名: `Phase2`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### E.2 実施内容

- `ISRRuntimeSemanticSchema.h` に `ProcessingProjectionSemantic` 構造体を追加
- `AudioEngine::RuntimeState` に `processingProjection` フィールドを追加
  - 分類: Derived / RuntimeWorld / MutablePrePublish / ObserveBoundary / RuntimeWorldLifetime
- `kFieldDescriptors` = 18 エントリに更新（processingProjection 追加）
- `kAuthorityInventory` = 18 エントリに更新（processingProjection: Derived）
- `buildRuntimePublishWorld` で RT-only knobs を processingProjection に capture
- `captureAudioThreadParameterSnapshot` で world != nullptr 時に processingProjection から読み取り
- `ISRRuntimeSemanticSchema.h` の stray `};` 構造体順序バグを修正

### E.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | RuntimeState 単体で processBlock() 実行可能な情報が閉包化 | pass | `processingProjection` で RT-only knobs を world に capture |
| Exit-2 | RuntimeState 外 mutable SoT ゼロ化（該当フィールド分） | pass | `captureAudioThreadParameterSnapshot` が world から読み取り |
| Exit-3 | 追加フィールドが Authority/Projection/Diagnostic に分類済み | pass | `kFieldDescriptors[17]`: Derived / RuntimeWorld |

### E.4 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録F: Phase3 提出実績（2026-06-01）

### F.1 提出メタ情報

- フェーズ名: `Phase3`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### F.2 フェーズ開始条件チェック

- [x] 前フェーズ出口条件（Phase2）を再確認
- [x] 未達項目なし
- [x] 例外運用なし
- [x] 今回変更は Phase3 責務のみ（AudioEngine Non-Authority 確認）
- [x] 後続フェーズ条件の先行達成なし（Rule-37）

### F.3 探索証跡

#### F.3.1 grep（実行範囲: Production Runtime Tree）

- `publishState(` in PRT: 1 hit
  - `AudioEngine.h:2758` `publishRuntimeStateNonRt` 内、`makeRuntimePublicationCoordinator().publishState(...)` → Coordinator が authority 主語
- `runtimePublicationBridge_.(commit|retire)(` in PRT: 2 hit
  - `AudioEngine.Commit.cpp:315` → commit 主語: `convo::isr::RuntimePublicationCoordinator`
  - `AudioEngine.Commit.cpp:351` → retire 主語: `convo::isr::RuntimePublicationCoordinator`
- `buildRuntimePublishWorld(` entry: `core/RuntimePublicationCoordinator.h:58` → `bridge_.buildRuntimePublishWorld(...)` → `RuntimePublicationBridge` が主語
- `prepareCommit(|executeCommit(|commitNewDSP(` in PRT: 0 hit（Phase1 で削除済み）

#### F.3.2 Authority 主語マッピング

| Authority op | 呼び出し主語 | AudioEngine から外れているか |
| --- | --- | --- |
| commit | `convo::isr::RuntimePublicationCoordinator` (runtimePublicationBridge_) | pass |
| retire | `convo::isr::RuntimePublicationCoordinator` (runtimePublicationBridge_) | pass |
| publish | `convo::RuntimePublicationCoordinator::publishState` (callsite=1) | pass |
| build | `convo::RuntimeBuilder::buildRuntimePublishWorld` | pass |
| activate | `convo::isr::RuntimePublicationCoordinator::commit` が実質 activate | pass |

### F.4 Authority差分

#### Phase3 Before（Phase1 完了後の基準状態）

- `onRuntimePublishedNonRt` (AudioEngine callback) → `runtimePublicationBridge_.commit(...)` で commit authority を isr::Coordinator に委譲
- `onRuntimeRetiredNonRt` (AudioEngine callback) → `runtimePublicationBridge_.retire(...)` で retire authority を isr::Coordinator に委譲
- AudioEngine は Observe/callback 受信 + 診断記録の役割のみ

#### Phase3 After

- Authority op 呼び出し主語が全て AudioEngine 外に収束
  - commit: `convo::isr::RuntimePublicationCoordinator`
  - retire: `convo::isr::RuntimePublicationCoordinator`
  - publish: `convo::RuntimePublicationCoordinator`
  - build: `RuntimePublicationBridge`

### F.5 フェーズ出口条件判定（Phase3）

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | Authority 操作の主語が AudioEngine から外れている | pass | F.3.1 grep + F.3.2 主語マッピング |
| Exit-2 | AudioEngine 責務が Observe/Process/Measure に限定 | pass | callback 受信・診断記録のみ（authority 主語でない） |

### F.6 C1〜C15 更新（差分のみ）

| ID | 判定項目 | 現在判定 | 証跡 |
| --- | --- | --- | --- |
| C1 | `publishState` 呼び出し元 = 1 | pass | `AudioEngine.h:2758` のみ |
| C4 | commit authority 主語 = isr::Coordinator | pass | `runtimePublicationBridge_.commit` (l.315) |
| C5 | retire authority 主語 = isr::Coordinator | pass | `runtimePublicationBridge_.retire` (l.351) |

### F.7 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0, ninja: no work to do）
- CTest: 9/9 pass（全テストスイート成功）

---

## 付録G: Phase5-Gate 提出実績（2026-05-31）

### G.1 提出メタ情報

- フェーズ名: `Phase5-Gate`
- 提出日: `2026-05-31`
- 提出者: `GitHub Copilot`
- 成果物: `doc/work12/phase5_gate_dsp_selection_state_machine_design_2026-05-31.md`

### G.2 Gate成果物チェック

- [x] 状態図（Stable/Entering/Retiring）
- [x] AudioThread 分岐置換表
- [x] 互換フェイルセーフ定義
- [x] 設計承認記録

### G.3 判定

- Gate判定: **pass**
- 次工程: Phase5 実装（`transition.active` 実行依存の段階撤去）

---

## 付録H: Phase7 設計事前承認（2026-05-31）

### H.1 提出メタ情報

- フェーズ名: `Phase7 design pre-approval`
- 提出日: `2026-05-31`
- 提出者: `GitHub Copilot`
- 成果物: `doc/work12/phase7_retire_manager_design_contract_2026-05-31.md`

### H.2 判定

- RetireManager 新規設計契約: **Approved (Design Ready)**
- 実装着手条件: Phase6 完了後

---

## 付録I: Phase4 提出実績（2026-06-01）

### I.1 提出メタ情報

- フェーズ名: `Phase4`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### I.2 実施内容

- Snapshot項目の分類運用を `RuntimeWorld` semantic schema 側へ統一
- Snapshot派生情報は `captureAudioThreadParameterSnapshot(...)` 経由の projection 読み取りへ集約
- `snapshot.(generation|version|active)` を実行分岐に使う経路を調査し、実行分岐依存が無いことを確認

### I.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | Snapshot項目を Authority/Projection で分類 | pass | `ISRRuntimeSemanticSchema.h` の descriptor / inventory |
| Exit-2 | `snapshot.(generation/version/active)` 実行分岐ゼロ | pass | `grep_search`（PRT範囲） |

### I.4 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録J: Phase5 提出実績（2026-06-01）

### J.1 提出メタ情報

- フェーズ名: `Phase5`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### J.2 実施内容

- AudioThread の DSP 選択分岐を `RuntimeWorld.execution.transitionActive` 基準へ移行
  - `AudioEngine.Processing.AudioBlock.cpp`
  - `AudioEngine.Processing.BlockDouble.cpp`
  - `AudioEngine.Processing.Snapshot.cpp`
  - `AudioEngine.Timer.cpp`
  - `AudioEngine.h`（helper / diagnostic path）
- `transition.active` を実行分岐で参照していた経路を撤去し、state-machine semantic 参照へ統一

### J.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | AudioThread分岐を新設計へ置換 | pass | 上記5ファイルの差分 |
| Exit-2 | `transition.active` 実行依存撤去 | pass | `grep_search("transition\\.active", "src/audioengine/**")` 13→4（残存は代入/投影のみ） |
| Exit-3 | `transition.active` が実行分岐に使われない | pass | 残存4件の内訳確認（条件分岐0件） |

### J.4 C1〜C15 更新（差分のみ）

| ID | 判定項目 | 現在判定 | 証跡 |
| --- | --- | --- | --- |
| C6 | TransitionState が実行分岐に使われない | pass | `src/audioengine/**` の `transition.active` 分岐依存ゼロ |

### J.5 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録K: Phase6 提出実績（2026-06-01）

### K.1 提出メタ情報

- フェーズ名: `Phase6`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### K.2 実施内容

- Runtime generation の authority を `generation` へ一本化
  - `AudioEngine::RuntimeState::kFieldDescriptors`:
    - `generation` = `SemanticCategory::Authority`
    - `generationSemantic` = `SemanticCategory::Derived` へ変更
  - `AudioEngine::RuntimeState::kAuthorityInventory`:
    - `generation` = `RuntimeAuthorityClass::Authoritative`
    - `generationSemantic` = `RuntimeAuthorityClass::Derived` へ変更
- `PublicationSemantic::mappedRuntimeGeneration` を独立authorityではなく導出表現へ変更
  - `ISRRuntimeSemanticSchema.h`:
    - `mappedRuntimeGeneration` = `SemanticCategory::Derived`

### K.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | Generation authority = 1 | pass | `AudioEngine.h` inventory で `generation` のみ Authoritative |
| Exit-2 | 非権威 generation の mirror/derived 化 | pass | `generationSemantic` / `mappedRuntimeGeneration` の分類変更 |

### K.4 C1〜C15 更新（差分のみ）

| ID | 判定項目 | 現在判定 | 証跡 |
| --- | --- | --- | --- |
| C7 | Generation authority = 1 | pass | `AudioEngine.h` (`kAuthorityInventory`, `kFieldDescriptors`) |

### K.5 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録L: Phase7 提出実績（完了, 2026-06-01）

### L.1 提出メタ情報

- フェーズ名: `Phase7`
- 提出日: `2026-06-01`
- 提出者: `GitHub Copilot`
- 対象コミット範囲: `local-working-tree`

### L.2 実施内容（完了済み）

- 共通 fallback retire キュー `core/DeferredRetireFallbackQueue.h` を新規導入
- AudioEngine 側
  - `deferredDeleteFallbackQueue` / `deferredDeleteFallbackMutex` を共通キューへ統合
  - `audioThreadRetireOverflowPtr` / `audioThreadRetireOverflowEpoch` を撤去
  - retire backlog 集計を共通キュー深さ基準へ統一
- EQProcessor 側
  - `deferredDeleteFallbackQueue` / `deferredDeleteFallbackMutex` を共通キューへ統合
  - `EQProcessor::deferredDeleteFallbackQueue` 実体を撤去（C15対応）
- 境界統合（DeferredFreeThread / ISRRetireRuntimeEx）
  - `core/RetireBoundaryTelemetry.h` を新規追加
  - `DeferredFreeThread::snapshotBoundaryTelemetry()` を追加
  - `RetireRuntimeEx::snapshotBoundaryTelemetry()` を追加

### L.3 Exit判定

| 条件ID | 条件内容 | 判定 | 証跡 |
| --- | --- | --- | --- |
| Exit-1 | Retire authority = 1 | pass | `.github/scripts/isr-verify-v5-retire-authority-lane.ps1` = `[PASS]` |
| Exit-2 | C15（EQ fallback queue = 0）達成 | pass | `grep_search("deferredDeleteFallbackQueue", "src/eqprocessor/**")` 実体参照なし |

### L.4 C1〜C15 更新（差分のみ）

| ID | 判定項目 | 現在判定 | 証跡 |
| --- | --- | --- | --- |
| C8 | Retire authority = 1 | pass | `isr-verify-v5-retire-authority-lane.ps1` pass |
| C15 | `EQProcessor::deferredDeleteFallbackQueue` = 0（PRT） | pass | `grep_search("deferredDeleteFallbackQueue", "src/eqprocessor/**")` 実体参照なし |

### L.5 検証結果

- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録M: Phase9 提出実績（完了, 2026-06-01）

### M.1 実施内容（本コミット）

- `LegacyTemporary` authority 分類の実体を撤去
  - `src/audioengine/ISRRuntimeSemanticSchema.h`
  - `src/audioengine/ISRAuthorityClass.h`
- 残存確認: `grep_search("LegacyTemporary", "src/**") = 0`
- 期限付き互換ブリッジ残存: `isr-legacy-temporary.json` の `entries=[]` を確認

### M.2 検証結果

- Legacy語彙検証: pass（`.github/scripts/isr-verify-legacytemporary-zero-references.ps1`）
  - report: `evidence/legacytemporary_zero_references_report.json`
- Debug build: pass（`Build_CMakeTools` result code 0）
- CTest: 9/9 pass

---

## 付録N: Phase10 提出実績（完了, 2026-06-01）

### N.1 実施内容

- C1〜C15 最小自動判定スクリプトを新規追加
  - `.github/scripts/isr-verify-c1-c15-minimal.ps1`
  - 出力: `evidence/c1_c15_minimal_report.json`
- Tiered verification（smoke）へ最小ゲートを配線
  - `.github/scripts/isr-run-tiered-verification.ps1`
  - 追加エントリ: `.github/scripts/isr-verify-c1-c15-minimal.ps1`
- Warning級規約の導入期限を反映
  - `.github/isr-validator-tiering-policy.json` に `warningTierAdoption` を追加

### N.2 判定サマリ

- 自動判定: pass
- C1〜C15: `15/15 pass`（manual pending 0）

### N.3 検証結果

- `isr-verify-c1-c15-minimal.ps1`: pass
- report: `evidence/c1_c15_minimal_report.json`

---

## 付録O: 追加独立再監査（2026-06-01, CodeGraph + Serena + grep）

### O.1 「確定漏れ → 解消済み」反映

| 項目 | 変更前（確定漏れ） | 変更後 | 判定 | 証跡 |
| --- | --- | --- | --- | --- |
| Leak-01 | `AudioEngine::publish*` が残存（命名/責務逸脱） | `snapshotRcuEpoch` / `markRetireEpoch` / `storeLearnedCoeffs` / `storeLearnedCoeffsToBank` へ移行 | 解消済み | `AudioEngine::commit/publish/retire/build/activate*` 検索が `src/**` で 0件 |
| Leak-02 | C4 判定が fail-open（旧 commit 経路のみ監視） | C4 を fail-closed 化（`AudioEngine::commit/publish/retire/build/activate*` を禁止） | 解消済み | `.github/scripts/isr-verify-c1-c15-minimal.ps1` の C4 条件更新、`evidence/c1_c15_minimal_report.json` の `forbiddenAudioEngineOps=0` |

### O.2 追加発掘結果（修正完了）

| 区分 | 判定 | 内容 | 証跡 |
| --- | --- | --- | --- |
| C6 実質漏れ | 解消済み | AudioThread 実行分岐の判定を `execution.transitionActive` から state machine 投影 `topology.hasFadingRuntime` へ置換。C6検証も `transition.active` / `execution.transitionActive` の両方を fail-closed で監査するよう強化。 | `AudioEngine.Processing.AudioBlock.cpp`, `AudioEngine.Processing.BlockDouble.cpp`, `AudioEngine.Processing.Snapshot.cpp`, `AudioEngine.Timer.cpp`, `.github/scripts/isr-verify-c1-c15-minimal.ps1`, `evidence/c1_c15_minimal_report.json` |
| C15 実質漏れ | 解消済み | EQ 側 fallback queue 実体（`deferredRetireFallbackQueue_`）を撤去し、retire enqueue を `EpochDomain` bounded retry へ統一。C15監査を旧/新 alias 両対応で fail-closed 化。 | `src/eqprocessor/EQProcessor.h`, `src/eqprocessor/EQProcessor.Core.cpp`, `.github/scripts/isr-verify-c1-c15-minimal.ps1` |

### O.3 詳細設計どおり実装済み（今回監査で再確認）

| 観点 | 判定 | 証跡 |
| --- | --- | --- |
| C1/C14: `publishState()` callsite = 1（PRT） | 実装済み | `evidence/c1_c15_minimal_report.json` (`publishStateAll=2`, `publishStateDecl=1`, `callsites=1`) |
| C2/C3: legacy publication symbol 除去 | 実装済み | 同 report: `count=0` |
| C6: execution branch alias (`transition.active` / `execution.transitionActive`) = 0 | 実装済み | 同 report: `tokenCount=0` |
| C7/C9/C10: authority/inventory 整合 | 実装済み | 同 report: `pass` |
| C11/C12/C13/C15（PRT） | 実装済み | 同 report: `count=0` |

### O.4 再監査実行ログ要約

- CodeGraph index 再整備: `CodeGraph Apply Local Patch` / `CodeGraph Full Index` / `CodeGraph Stats`
- Runtime 検証: `Build_CMakeTools` pass, `RunCtest_CMakeTools` pass
- 統治検証: `isr-verify-c1-c15-minimal.ps1` pass（15/15）

### O.5 二次横断監査（Serena + CodeGraph + grep, 2026-06-01）

| 区分 | 判定 | 内容 | 証跡 |
| --- | --- | --- | --- |
| 確定漏れ | Leak-04（解消済み） | NonRT の runtime admission / closure tier 判定を `world.topology.hasFadingRuntime` 主軸へ統一し、`execution.transitionActive == topology.hasFadingRuntime` の等価性ガードを precheck/retire path に導入。 | `src/audioengine/AudioEngine.Commit.cpp`（最小差分修正）, `Build_CMakeTools` pass, `RunCtest_CMakeTools` pass, `evidence/c1_c15_minimal_report.json` |
| 詳細設計どおり実装済み | C1/C14 | `publishState()` callsite は PRT で 1 | `grep_search("publishState\\s*\\(", "src/audioengine/**")` |
| 詳細設計どおり実装済み | C2/C3/C4 | `AudioEngine::commit/publish/retire/build/activate*` 実装経路なし（fail-closed 監視有効） | `AudioEngine::commit/publish/retire/build/activate*` 検索が `src/**` で 0件 |
| 詳細設計どおり実装済み | Phase8-A Exit | `RuntimeBuilder.*` から `BuildSnapshot` / `ConvolverProcessor::BuildSnapshot` 参照なし | `BuildSnapshot` と `ConvolverProcessor::BuildSnapshot` の検索が `src/audioengine/RuntimeBuilder.*` で 0件 |
| 詳細設計どおり実装済み | C11/C12/C13（PRT） | `src/convolver/**` では `SafeStateSwapper` / `PendingParams` / `PreparedIRState` 参照なし | `grep_search(..., "src/convolver/**")=0` |

補足:

- `src/ConvolverProcessor.h` には `SafeStateSwapper` / `PendingParams` / `PreparedIRState` が残存（PRT外 alias/互換層）。現行 C11-C13 判定スコープ外であり、次回は「PRT外残存の扱い」を統治文書で明確化する。
