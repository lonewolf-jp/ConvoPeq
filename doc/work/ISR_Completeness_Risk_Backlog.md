# ConvoPeq ISR Completeness Risk Backlog

## 目的

ISR completeness 未完領域を、正本仕様に影響する順序で管理する。
本書は backlog 正本であり、詳細仕様は各正本文書へ反映して閉じる。

ステータス定義:

- `Spec-Fixed`: 仕様確定済み（実装・検証証跡は未完）
- `Closed`: 仕様反映・実装・検証証跡が完了

確定宣言（2026-05-20）:

- R1〜R18 は `Spec-Fixed` として確定済み（安定性優先の必須コア）
- R19〜R25 は `Spec-Fixed`（ガード付き拡張。Debug/CI 優先、Release 直結では任意）
- 本書に記載のない追加未完事項は扱わない（追加時はR採番必須）

最新ステータス更新（2026-05-21）:

- R11〜R25 は Closed 最小検証項目を満たし `Closed` へ更新。
- R1〜R10 は `Spec-Fixed` を維持（段階適用対象）。

### REV3.2運用優先注記

- 本書の評価基準は `plan5.md` REV3.2 を優先する。
- `runtime exposes evidence / CI validates evidence` を固定方針とし、
  Release へ full artifact / full verify を常時要求しない。
- 解釈衝突時は few-authority（7 subsystem）/ 2-world（Publication/Execution）/
  capability-first（runtime coordinator lifecycle 非導入）を優先する。
- stale handle 関連の Closed 判定は CI=Abort / Debug=Assert / Release=Quarantine+Silence を前提に解釈する。

用語正規化（齟齬回避）:

- 本書では `RuntimePublication` を正規記法として扱う。
- CI artifact 判定（missing/schema/parse fail）は merge gate 条件であり、
  runtime correctness は runtime invariants で成立させる。

---

## R1. Deep Immutability Enforcement 固定

- 状態: Spec-Fixed（2026-05-20）
- 重要度: Critical
- 主要リスク: discipline依存で post-publish mutation 混入
- 反映先: `ISR_Immutability_Enforcement_Spec.md`
- 確定方針:
  - freeze bit + mutation assert + immutable facade + post-publish detector を必須4点セットとする
  - Release でも failure counter を記録し no-op を禁止する
- 実行フェーズ: Phase A
- 完了条件:
  - freeze bit / mutation assert / immutable facade / post-publish detector の実装検証ルール確定
  - CI fail 条件が運用化
- Closed最小検証項目:
  - [ ] freeze 未実行 payload の publish が拒否される（自動テスト）
  - [ ] publish 後 write が seal violation として検出される（Debug/Release双方）
  - [ ] post-publish mutation detector が CI で fail を返す

## R2. RuntimeGraph Deep Immutability 閉包

- 状態: Spec-Fixed（2026-05-20）
- 重要度: Critical
- 主要リスク: FFT cache / IR ptr / async state の可変参照混入
- 反映先: `ISR_Runtime_State_Matrix.md`, `ISR_HB_Graph_Specification.md`
- 確定方針:
  - RuntimeGraph は publish 後 read-only closure を必須とし、payload 外 mutable 依存を禁止
  - FFT/IR/async は handle/snapshot 経由のみ参照可
- 実行フェーズ: Phase B
- 完了条件:
  - RuntimeGraph 内依存が read-only closure を満たす
  - payload 外 mutable 参照の禁止検証が成立

## R3. DSPHandle Allocator 粗密要件固定

- 状態: Spec-Fixed（2026-05-20）
- 重要度: High
- 主要リスク: reuse/wraparound/compaction/shutdown flush の仕様抜け
- 反映先: `ISR_DSPHandle_Allocator_Policy.md`
- 確定方針:
  - reuse latency は quarantine epoch 完了後に限定
  - wraparound は運用停止境界を明記し、到達前メンテ停止を義務化
  - compaction は non-RT 限定、shutdown flush ordering を先行固定
- 実行フェーズ: Phase B
- 完了条件:
  - slot reuse latency / generation wraparound / sparse compaction / shutdown flush ordering の閾値化

## R4. RuntimeWorldRetireManager 責務肥大対策

- 状態: Spec-Fixed（2026-05-20）
- 重要度: High
- 主要リスク: single authority と single mega-manager の混同
- 反映先: `ISR_Retire_Authority_Graph.md`
- 確定方針:
  - authority identity は単一維持、実装は lane 分離を許容
  - lane 追加時も独立 authority 再定義を禁止
- 実行フェーズ: Phase B
- 完了条件:
  - authority identity を維持した lane 分離方針（DSP/snapshot/cache）が定義済み

## R5. HB Domain F 厳密化

- 状態: Spec-Fixed（2026-05-20）
- 重要度: High
- 主要リスク: smoothing lifetime / host automation ordering / callback reentrancy の事故
- 反映先: `ISR_HB_Graph_Specification.md`
- 確定方針:
  - prepareToPlay -> callback start -> callback stop -> releaseResources の順序鎖を必須化
  - callback reentrancy は単一順序規約違反として明示検出対象にする
- 実行フェーズ: Phase C
- 完了条件:
  - Domain F で順序制約と違反ケースが明文化
  - callback 境界の再入・順序崩れ検証条件が定義済み

## R6. bug2 形式再現 HB モデル

- 状態: Spec-Fixed（2026-05-20）
- 重要度: Critical
- 主要リスク: KPI監視のみで根本 ordering 欠落を見逃す
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 確定方針:
  - 失敗ordering（UAF発火）と必要HB（抑止）を1対1で対応付ける形式モデルを必須化
  - KPI は補助指標とし、モデル検証を主判定にする
- 実行フェーズ: Phase C
- 完了条件:
  - 失敗orderingと必要HBの対照モデルが承認済み

## R7. Recursive Payload Closure 完全性

- 状態: Spec-Fixed（2026-05-20）
- 重要度: Critical
- 主要リスク: publish payload 下位オブジェクトで ownership closure が破断
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Runtime_State_Matrix.md`, `ISR_Runtime_Closure_Descriptor.md`
- 確定方針:
  - payload closure metadata は再帰閉包（parent->child->grandchild）を表現可能であること
  - closure 未閉包を検出する静的/動的規則を必須化
- 実行フェーズ: Phase C
- 完了条件:
  - payload closure metadata が再帰閉包を表現
  - closure 不完全時の検出規則が運用化

## R8. Shutdown HB Strict Ordering

- 状態: Spec-Fixed（2026-05-20）
- 重要度: Critical
- 主要リスク: callback停止→observer消滅→retire停止→reclaim完了→allocator shutdown の順序破綻
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_DSPHandle_Allocator_Policy.md`
- 確定方針:
  - shutdown ordered chain を canonical sequence として固定
  - drain 完了判定前に allocator shutdown へ進む遷移を禁止
- 実行フェーズ: Phase C
- 完了条件:
  - shutdown ordered chain が明文化
  - drain 完了判定の停止条件が固定

## R9. RT検知 -> NonRT Retire Enqueue Bridge 固定

- 状態: Spec-Fixed（2026-05-20）
- 重要度: High
- 主要リスク: Audio callback 起点の完了イベントが直接 retire enqueue と衝突
- 反映先: `ISR_HB_Graph_Specification.md`, `ISR_Retire_Authority_Graph.md`
- 確定方針:
  - Audio Thread は retire authority を持たず、完了検知のみを行う
  - retire enqueue は NonRT bridge 経由で authority 側へ委譲する
- 実行フェーズ: Phase B
- 完了条件:
  - RT detect -> NonRT enqueue request 経路が明文化
  - callback 直enqueue禁止の検証規則が運用化
- Closed最小検証項目:
  - [ ] Audio Thread が retire enqueue を直接呼ばないことを静的/動的に検証
  - [ ] RT detect -> NonRT enqueue request -> authority enqueue 経路が再現テストで確認される
  - [ ] callback 直enqueue違反時に検出・失敗（CI）する

## R10. Shared Epoch Canonical 前提の移行コスト固定

- 状態: Spec-Fixed（2026-05-20）
- 重要度: High
- 主要リスク: shared epoch 失敗時の split migration cost 未評価で設計拘束が強すぎる
- 反映先: `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`, `ISR_HB_Graph_Specification.md`
- 確定方針:
  - shared strategy は canonical ではなく「検証結果依存」とする
  - split epoch migration の手順・コスト項目を先に定義する
- 実行フェーズ: Phase C
- 完了条件:
  - split migration 手順とコスト評価軸（latency/jitter/reclaim burst）が定義済み
  - shared継続 / split移行のGo/No-Go条件が承認済み
- Closed最小検証項目:
  - [ ] split migration runbook（切替手順・ロールバック手順）が文書化済み
  - [ ] latency/jitter/reclaim burst の比較表（shared vs split）が作成済み
  - [ ] Go/No-Go 判定が記録され、判定理由が残っている

## R11. Closure Descriptor System 固定

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: transitive ownership が暗黙で closure 破断を検出できない
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Runtime_Closure_Descriptor.md`
- 実行フェーズ: Phase B/C
- 完了条件:
  - descriptor node に ownership/mutability/lifetime/HB を保持
  - publish 前 closure validation が運用化
- Closed最小検証項目:
  - [x] descriptor node に kind/ownership/mutability/lifetime/HB/authority/allocator 情報が記録される
  - [x] publish 前 `validateClosureGraph` が mandatory 実行され、違反を reject する
  - [x] external mutable dependency の混入が CI で検出・失敗する

## R12. Payload Tier System 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: payload boundary が曖昧で forbidden dependency 混入
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Runtime_State_Matrix.md`, `ISR_Payload_Tier_Model.md`
- 実行フェーズ: Phase B/C
- 完了条件:
  - tier 分類（InlineImmutable/ImmutableShared/ExternalPinned/RTLocalOnly/Forbidden）が定義済み
  - Forbidden/RTLocalOnly の payload 禁止検証が運用化
- Closed最小検証項目:
  - [x] 全 payload object family に tier が割当済み
  - [x] Forbidden tier の payload 混入が検出・失敗する
  - [x] RTLocalOnly tier の RuntimePublication への混入が検出・失敗する

## R13. Immutable Facade + Mutable Core 分離

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: RuntimePublication 内への mutable atomic/mutex/lazy-init 混入
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Immutability_Enforcement_Spec.md`
- 実行フェーズ: Phase B
- 完了条件:
  - publish graph が read-only projection のみで構成される
  - mutable cache が RTLocal/Background domain へ隔離済み
- Closed最小検証項目:
  - [x] publish graph 経由で mutable API が露出しない
  - [x] publish graph 内 mutex / lazy-init / mutable atomic が存在しない
  - [x] mutable cache が RTLocal または Background domain に限定される

## R14. Deferred Retire Intent Queue 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: RT completion detect と retire authority 実行の責務衝突
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Retire_Authority_Graph.md`, `ISR_Deferred_Retire_Intent_Bridge.md`
- 実行フェーズ: Phase B
- 完了条件:
  - RT は intent emission のみ、NonRT が authority enqueue を実行
  - RT reclaim/delete/authority の禁止検証が運用化
- Closed最小検証項目:
  - [x] RT は `RetireIntent` emission のみを実行し、retire/reclaim/delete を実行しない
  - [x] NonRT coordinator が intent dequeue 後に authority enqueue を実行する
  - [x] RT直enqueue/RT reclaim の違反が検出・失敗（CI）する

## R15. Shutdown HB FSM 固定

- 状態: Closed（2026-05-21）
- 重要度: Critical
- 主要リスク: shutdown ordering が手続き依存で late callback/UAF を誘発
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Shutdown_State_Machine.md`
- 実行フェーズ: Phase C
- 完了条件:
  - shutdown state machine と phase HB chain が明文化
  - shutdown verifier で順序違反を検出できる
- Closed最小検証項目:
  - [x] phase enum（Running→AudioStopped→ObserverDrained→RetireClosed→EpochSettled→ReclaimComplete→ShutdownComplete）が実装される
  - [x] phase 逆行/飛び越し遷移が検出・拒否される
  - [x] shutdown verifier が late callback / post-stop enqueue を検出・失敗する

## R16. HB Failure Spec + Reorder Simulation 固定

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 重要度: Critical
- 主要リスク: 再現試験依存で最小HB欠落の証明が不足
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_HB_Graph_Specification.md`, `ISR_Minimal_HB_Failure_Model.md`
- 実行フェーズ: Phase C
- 完了条件:
  - failure ordering と required HB の対照モデルが固定
  - reorder simulator が CI で実行される
- Closed最小検証項目:
  - [x] bug2 の最小HB欠落モデル（failure ordering）が文書化される
  - [x] required HB を適用した対照ケースで UAF 不成立を確認する
  - [x] reorder simulator（forced reorder/epoch lag/retire delay/observe race）が CI で実行される

## R17. Epoch Abstraction Layer 固定

- 状態: Closed（2026-05-21）
- 重要度: High
- 主要リスク: shared epoch が architecture invariant 化して移行不能
- 反映先: `ISR_Formal_Guarantee_Package.md`, `ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`
- 実行フェーズ: Phase C
- 完了条件:
  - epoch coordinator abstraction が定義済み
  - shared/split/hybrid 切替方針が定義済み
- Closed最小検証項目:
  - [x] runtime code が concrete epoch 実装に直接依存しない
  - [x] shared/split/hybrid の切替インターフェースが定義済み
  - [x] split 移行時の rollback 手順が実行可能である

## R18. CI Verification Pipeline 固定

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 重要度: Critical
- 主要リスク: 文書規律のみで merge 時に形式違反を検出できない
- 反映先: `ISR_Formal_Guarantee_Package.md`, `plan5.md`, `ISR_Verification_Pipeline.md`, `ISR_Runtime_Reduction_Strategy.md`, `ISR_Proof_Artifact_Schema_Registry.md`
- 実行フェーズ: Phase C
- 完了条件:
  - 10段ステージ（V1 Atomic Dot-Call Scan / V2 Seal Integrity Check / V3 Recursive Closure Validation / V4 Payload Tier Validation / V5 HB Reorder Simulation / V6 Shutdown FSM Verification / V7 Retire Latency Audit / V8 UAF Suspicion Detector / V9 Forbidden Capability Scan / V10 Ownership Cycle Detection）が定義済み
  - runtime-generated proof artifacts と CI evaluator の責務分離が定義済み
  - proof artifact の canonical naming と JSON schema contract が固定済み
  - pipeline failure が merge blocker として運用化
- Closed最小検証項目:
  - [x] 10段ステージがCIワークフローに統合済み
  - [x] artifact missing / schema mismatch / parse error が CI fail として扱われる
  - [x] canonical artifact 名（registry定義）で生成される
  - [x] いずれか失敗時に merge blocker として PR を停止する
  - [x] 成功時に証跡（レポート/ログ）が保存される

## R19. Authority capability-first 固定（coordinator互換shim化）

- 状態: Closed（2026-05-21）
- 補足: 必須（authority misuse を型制約で抑制）
- 重要度: Critical
- 主要リスク: coordinator中心設計の増殖による authority lifecycle 問題
- 反映先: `ISR_Execution_Authority_Convergence.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - type-level capability（Publish/Retire/Shutdown）が優先運用される
  - runtime coordinator lifecycle を導入しない
- Closed最小検証項目:
  - [x] capability tag が API 契約として明示される
  - [x] coordinator 依存の新規経路を追加しない
  - [x] 分散 authority 導入時に CI でゲート違反となる

## R20. Host Chaos Normalization 固定

- 状態: Closed（2026-05-21）
- 補足: Release 適用候補（Layer 0 補完として優先度高）
- 重要度: High
- 主要リスク: host callback 非決定性により lifecycle invariant 崩壊
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_JUCE_Lifecycle_Isolation.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - HostChaosNormalizer が Layer 0 に統合される
- Closed最小検証項目:
  - [x] HC-1 duplicate prepare collapse が動作
  - [x] HC-2 release-before-prepare reject が動作
  - [x] HC-3 callback during Releasing reject が動作

## R21. DSP ownership 単純化（few-authority）

- 状態: Closed（2026-05-21）
- 補足: 必須（DSPHandleRuntime への統合）
- 重要度: Critical
- 主要リスク: DSP ownership path 分散による運用複雑化
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_DSPHandle_Runtime.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - DSP ownership は DSPHandleRuntime へ統合される
  - crossfade 完了前 retire 禁止が単純ルールで運用される
- Closed最小検証項目:
  - [x] DSPHandleRuntime が callback view を一元提供する
  - [x] crossfade complete before retire が検証される

## R22. callback-local snapshot consistency 固定

- 状態: Closed（2026-05-21）
- 補足: 必須（RTExecutionFrame の最小保証）
- 重要度: High
- 主要リスク: callback 中に runtime snapshot が変化する
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_RT_Execution_Frame.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - RTExecutionFrame が callback-local immutable snapshot を保持する
- Closed最小検証項目:
  - [x] callback 中 snapshot immutable が検証される
  - [x] callbackEpoch 単位で一貫 view が維持される

## R23. world model 簡略化（2-world固定）

- 状態: Closed（2026-05-21）
- 補足: 必須（single-process 最適化）
- 重要度: Critical
- 主要リスク: 不要な federation 抽象で実装複雑化
- 反映先: `ISR_Execution_Authority_Convergence.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - world model を PublicationWorld / ExecutionWorld の2つへ固定
- Closed最小検証項目:
  - [x] RuntimeBoundary 構造で境界が実装される
  - [x] full federation runtime 追加が抑制される

## R24. bounded deterministic teardown 固定

- 状態: Closed（2026-05-21）
- 補足: 必須（Release 安定性直結）
- 重要度: Critical
- 主要リスク: shutdown phase の肥大化で完了不能リスク増大
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_Shutdown_State_Machine.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - shutdown phase を 7段（Running->...->Complete）へ固定
  - bounded completion を必須化
- Closed最小検証項目:
  - [x] SH-1 callback 0
  - [x] SH-2 active crossfade 0
  - [x] SH-3 pending retire 0
  - [x] SH-4 observer 0

## R25. DebugRuntime CI限定化

- 状態: Closed（2026-05-21）
- 監査判定注記（2026-05-21）: **Closed**（`doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と整合）
- 補足: 必須（Release から proof負荷を除去）
- 重要度: High
- 主要リスク: proof/trace runtime の release 混入による性能劣化
- 反映先: `ISR_Execution_Authority_Convergence.md`, `ISR_Runtime_Reduction_Strategy.md`, `plan5.md`
- 実行フェーズ: Phase C+
- 完了条件:
  - Release/Debug/CI の proof 有効範囲が固定化される
  - DebugRuntime へ trace/verify/proof を集約する
- Closed最小検証項目:
  - [x] Release: proof off
  - [x] Debug: proof partial
  - [x] CI: proof full
  - [x] RuntimeReductionGate による新runtime追加審査が CI 強制される

---

## 運用ルール

- 本バックログは `plan5.md` の未完項目を一元管理する
- 各リスクは「反映先正本への反映 + 実装 + 検証証跡」で Close する
- Close 判定は Gate 条件の証跡（文書/検証）を必須とする
- 運用判定ラベルは `doc/work/plan5.md` と `doc/work/R11-R25_Closed判定監査表_2026-05-21.md` と同期し、**Closed（運用重視）/ Closed（厳密）/ 部分適合** の3区分を用いる。

## 外部監査参照

### 監査判定の往復参照（2026-05-21）

- R16: Backlog 状態 `Closed` / 監査判定 `Closed`
- R18: Backlog 状態 `Closed` / 監査判定 `Closed`
- R25: Backlog 状態 `Closed` / 監査判定 `Closed`
- AT-ISR: Air/Tail ISR統合トレーサビリティは `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`（AT-ISR行）および `doc/work/ISR_AirTail_統合設計_2026-05-22.md` を正本参照とする
- 判定更新原則: `Closed` への更新は本書の「Closed最小検証項目」充足時のみ行う。

- ハブ文書（相互参照）:
  - `doc/work/plan5.md`
- R11〜R25 Closed判定監査表（証拠ファイル/コード行付き）:
  - `doc/work/R11-R25_Closed判定監査表_2026-05-21.md`
