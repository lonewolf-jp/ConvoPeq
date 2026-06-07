---
goal: Practical Stable ISR Bridge Runtime — notfinished7.md 検証結果に基づく改修計画
version: 18.12
date_created: 2026-06-06
last_updated: 2026-06-09
note: 'v18.13: 【付録A更新+コード修正】付録AをPhase-D完了状態に更新(124→17件)。Publication.cpp/Retire.cpp/ReleaseResources.cppのm_epochDomain直接呼び出しをRouter経由に変更(4件)。EQProcessor独自ドメイン9件は未着手。ConvolverProcessor m_epochDomain deprecatedメンバ残存(既知)。'
owner: GitHub Copilot (AI Assistant)
status: 'Draft'
tags: refactor, architecture, bridge-runtime, retire-authority, epoch, snapshot-lifecycle
---

# Practical Stable ISR Bridge Runtime — notfinished7.md 改修計画 v18.12

![Status: Draft](https://img.shields.io/badge/status-Draft-yellow)

## 改訂履歴

| 日付 | 版 | 変更内容 |
| ------ | ----- | ---------- |
| 2026-06-06 | 1.0 | 初版。notfinished7.md 検証結果 + 実コード監査に基づく改修計画。Serena/CodeGraph/ccc/graphify(DeepSeek)/semble 全ツール駆使。 |
| 2026-06-06 | 2.0 | 第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。ISRRetireGateway 導入、SnapshotRetireManager純化、依存順序グラフ明示、RT安全 enforce層強化、PR-B-0追加、rollback戦略追加。 |
| 2026-06-06 | **3.0** | 第2回第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。ISRRetireGateway → ISRRetireRouter + 3 Policy Lane に変更（「分類された統一」）。RetireStateMachine 新設。EpochDomain enqueueRetire 完全 private化撤回→internal API維持。RetireBackpressurePolicy 追加。Retire Observability 追加。SnapshotRetireManager QUEUE ONLY 維持（実コード確認に基づき反論）。各レビュー指摘のソースコード検証結果を付録に追記。 |
| 2026-06-06 | **4.0** | 第3回第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。RetireStateMachine を2層化（Logical/Physical）。Router パフォーマンス設計（分岐テーブル化/fast-path）を追加。RT delete 完全遮断（static analysis + wrapper type）。FadeAuthority 単一書き込み原則をPR-B-0に明示。Policy Lane 優先順位モデル明文化。EpochDomain deprecated→段階的 error 化。Observability 追加メトリクス。 |
| 2026-06-06 | **5.0** | 第4回第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。delete 実行経路を DeferredDeletionWorkerThread に一本化（PR-2 構造変更）。DSPHandleRuntime を Router 経由に統合（slot metadata 管理として明確化）。SnapshotRetireManager QUEUE ONLY を実コードで再確認（epoch 引数あり）+ coordinator非依存を追記。CrossfadeRuntime/SnapshotFadeState/FadeAccumulator の3層 truth source 階層を明文化。RetireStateMachine に Logical=decision owner 原則を追記。tryReclaimResources 実体確認（単一 reclaimRetired 呼び出し、retry/fallback なし）に基づき記述修正。EpochDomain deprecated 維持（error 化撤回、v7 目標に変更）。P0/P1/P2 優先順位リスト導入。 |
| 2026-06-06 | **6.0** | 第5回第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。EQProcessor::enqueueDeferredDeleteWithFallback の 4-retry パターンを実コード確認し retry 禁止（P0-5）。AudioEngine::enqueueDeferredDeleteNonRtWithResult の reclaim+retry パターンを確認し retry 禁止（P0-5）。epochDomain.reclaimRetired() の直接呼び出しを deprecated API 化し requestReclaim()（enqueue only）に置換（P0-6）。SnapshotRetireManager の epoch 外部依存性を文書化（P1-4）。FadeAuthority 単一書き込み原則を全層に拡張（P1-5）。全 reclaim API の禁止ラップ化と debug assert による delete 検出を追加。 |
| 2026-06-06 | **7.0** | 第6回第三者レビュー（実運用破綻耐性観点）に基づき全面改訂。**Retire Commit Barrier 導入**（P0-2）: epoch + snapshot fade完了 + RetireRouter enqueue の三拍子が揃った時のみ Worker enqueue を許可する同期ポイント。**SnapshotCoordinator の直接 enqueueRetire/reclaimRetired 完全排除**（P0-3）: ~SnapshotCoordinator()/switchImmediate()/reclaim() の直接呼び出しを ISRRetireRouter 経由に変更。**SnapshotFadeState 縮小**（P1-3）: fade 進行管理を CrossfadeRuntime に統合し、SnapshotFadeState は thin mirror 化または削除。**Timer reclaim 完全禁止 + enqueue only 強制**（P0-1 強化）。**DSPQuarantineManager 実体確認**（P0-1 反論）: atomic bool フラグのみ、delete は行わないため delete 経路ではない。 |
| 2026-06-06 | **8.0** | 第7回第三者レビュー（実運用破綻耐性観点: 「統合のしすぎ」フェーズ）に基づき全面改訂。**Emergency Drain tier-2 導入**: Worker Thread 単一障害点対策として shutdown 専用 Emergency Retire Drain を追加。**Router stateless 化**: ISRRetireRouter を純粋ディスパッチテーブルに限定、状態を持たない。**Fade 階層固定化**: SnapshotFadeState 削除方針を撤回し、3層（UI coarse / Runtime sample-accurate / Accumulation analysis）の責務階層を固定。**Timer enqueue-only 化**: 完全禁止から「enqueue は許可、delete は禁止」に変更。**DSPHandleRuntime state layer 確認**: false positive を訂正し、slot metadata state layer として明確分離。**Retry Non-RT 限定許可**: 完全禁止から「RT 禁止、Non-RT 許可」に変更。**Retire Commit Barrier 2-tier 化**: soft barrier（優先制御）+ hard barrier（安全停止）の二段階。 |
| 2026-06-06 | **9.0** | 第8回第三者レビュー（実運用破綻耐性観点: 「責任拡散防止」フェーズ）に基づき最終調整。**Retire 経路完全一本化**: SnapshotRetireManager 削除（Queue は Router 内蔵）。DSPHandleRuntime retire/reclaim/quarantine を Request 変換層に格下げ（state layer 維持）。Router → DeferredDeletionQueue が唯一の正規経路。**Timer reclaim 完全禁止（再強化）**: enqueue only に統一。**enqueue failure retry 完全排除**: drop + telemetry 統一。**SnapshotFadeState 単一書き込み強制**: CrossfadeRuntime が唯一の write owner。**ownerThreadId 排除**: atomic ベースに統一、thread id は debug assert 限定。**Fade 3層 単一書き込み制限**: 各層は自身の時間スケールの書き込みのみ。 |
| 2026-06-06 | **10.0** | 第9回第三者レビュー（実運用破綻耐性観点: 「過剰単純化是正」フェーズ）に基づき最終調整。**Retire 2-lane 化**: 単一経路から Fast SPSC（RT critical）+ Slow Worker Queue（bulk cleanup）の二重経路に戻す。**SnapshotRetireFacade 復活**: 削除方針を撤回し、薄層 facade（routing + telemetry + ordering validation）として維持。**Fade 3層 独立書き込み復活**: 単一書き込み方針を撤回。UI coarse / DSP sample-accurate / analysis offline の独立書き込みを許可。**Timer enqueue 許可**: 完全禁止から「enqueue 許可、delete 禁止（soft rate limited）」に変更。**DSPHandleRuntime Stateful Handle Authority 復活**: Request 変換層への格下げを撤回。stateful handle authority として維持。**enqueue failure deferred retry queue 追加**: drop only から「drop + telemetry + deferred retry queue（bounded）」に変更。**全体方針転換**: 「single path + strict exclusion + drop-based recovery」→「failure isolation + redundancy + bounded fallback」。 |
| 2026-06-06 | **11.0** | 第10回第三者レビュー（実運用破綻耐性観点: 「最終仕上げ」フェーズ）に基づき5項目追加。**RetireId + 二重retire検出**: RetireRequest に RetireId 付与。`RetireRegistry` で二重 retire を検出・debug assert。**SnapshotRetireFacade 禁止事項明文化**: epoch判定/fade判定/publish判定/ownership判定を禁止。routing/logging/metrics/validation のみ許可。**FadeClockDomain + FadeGeneration 導入**: 全 fade 層に共通 generation を持たせ状態ドリフト検出。**DeferredRetryQueue 寿命・上限・優先順位定義**: maxRetryCount=3, maxRetryAgeMs=5000, DSP/Snapshot retire drop禁止、Cache retire drop許可。**ShutdownPhase 固定化**: Phase1 stop publish → Phase2 stop readers → Phase3 EmergencyDrain（EpochDomain alive 必須）→ Phase4 destroy epoch。SnapshotCoordinator の直接 enqueueRetire/reclaimRetired は依然 P0（未完了）。 |
| 2026-06-06 | **12.0** | 第11回第三者レビューに基づき7項目追加・修正。**RetireRegistry Release 時防御**: debug assert + release reject + telemetry に強化。**RetireRegistry key を ObjectAddress+Generation に変更**: RetireId 単独では不十分な二重 retire 検出を改善。**DeferredRetryQueue に random jitter 追加**: 指数バックオフに 0.8-1.2 倍 jitter を追加し thundering herd 防止。**FadeGeneration authority を CrossfadeRuntime に変更**: FadeAccumulator は analysis layer であり権威不適切。**EmergencyDrain best-effort → must-drain（maxDrainTime=500ms）**: delete 漏れによる次回起動時クラッシュ防止。**P0-8（SnapshotCoordinator 直接経路除去）を最優先に昇格**: RetireId 導入より先に実施。**Telemetry → RetireBackpressurePolicy 自動反映**: 閾値超過時に cache retire 停止、DSP retire 優先へ自動制御。 |
| 2026-06-07 | **13.0** | 第12回第三者レビュー（実コード整合性照合）に基づき7項目追加・修正。**P0-8 一括切替化**: 段階移行から全8経路をRouterへ一括切替に方針変更。**RetireRegistry key を (ObjectAddress, Generation, ObjectType) に昇格**: アドレス再利用リスク対策。Generation 発行元を CrossfadeRuntime に固定。**RetireBackpressurePolicy ヒステリシス追加**: highWatermark=1000, lowWatermark=700 の二重閾値。**DeferredRetryQueue maxQueueDepth 定義**: 飽和時の保護レベル明確化（Cache→Snapshot→DSP）。**EmergencyDrain maxDrainItems 併用**: 500ms + 100000 items の先到達制。**Retire Commit Barrier shutdown 例外**: forceFadeComplete / barrier timeout 追加。**SnapshotCoordinator 責務監査**: 実装完了後の責務監査を Phase-B 完了条件に追加。 |
| 2026-06-07 | **14.0** | 第13回第三者レビュー（実コード完全照合: 6ツール駆使）に基づき6項目追加。**P0-8 完了条件拡張: SnapshotCoordinator→EpochDomain 全依存16箇所（従来認識8→倍増）を特定。EpochDomain.h include 削除を完了条件に追加**: enqueueRetire/reclaimRetired だけでなく publish/current/ObservedRuntime コンストラクタ・メンバ変数・include まで全て除去。**RetireRegistry 世代パージ機構 (purgeBeforeGeneration)**: retire完了+delete完了+epoch安全を条件に永続成長防止。**DeferredRetryQueue WFQ (Weighted Fair Queue)**: DSP 8:Snapshot 4:Cache 1 の重み付けで starvation 防止。**PublishAdmissionState 入力側制御**: Normal/Congested/Critical の3段階。queueDepth>90%で新規 publish 減速。**EmergencyDrain byte-based 制御**: maxDrainBytes 追加、500ms OR 100000 items OR maxDrainBytes の先到達制。**SnapshotCoordinator API 凍結**: public API一覧を固定化し、CIで責務外API追加禁止を検査。 |
| 2026-06-07 | **15.0** | 第14回第三者レビュー（長期運用最適化）に基づき6項目追加。**P0-8 完了条件強化: SnapshotCoordinator が EpochDomain 型を完全に知らない状態**まで引き上げ（EpochDomain*/& 不可, ObservedRuntime(EpochDomain&)不可）。**P0-10 purge 実行主体単一化**: Router owner thread のみ実行。**P2-8 WFQ aging 追加**: maxWaitTime=30s 超過で Cache retire 強制昇格。**P0-11 PublishAdmissionState growth rate 追加**: queueDepth + depthSlope の複合 AdmissionScore。**P0-7 Barrier telemetry 詳細化**: waiting_epoch/waiting_fade/waiting_enqueue/waiting_shutdown を個別記録。**P2-9 EmergencyDrain ポリシー化**: max(residentMemory*0.1, 1GB) など設定可能に。 |
| 2026-06-07 | **16.0** | 第15回第三者レビュー（長期運用・保守運用・移行事故耐性）に基づき5項目追加。**P0-8 移行進捗 CI**: EpochDomain直接依存数20→15→10→5→0 をCIで可視化。**P0-10 forceFinalPurge**: ShutdownPhase3でRetireRegistry最終パージ + maxRegistryAge=24h。**P0-11 EMA-based depthSlope**: 500ms移動平均でノイズ除去。**P1-3 Router責務CIガード**: lifecycle/epoch/fade/publish decision 禁止をCIチェック対象に。**総評: P0-8完了が全項目中最重要。新アーキテクチャ変更より実装完了を優先。** |
| 2026-06-07 | **17.0** | 第16回第三者レビュー（実運用最終仕上げ）に基づき7項目追加。**P0-8 完了条件拡張: SnapshotCoordinator完了→全public retire APIがRouter経由**に。AudioEngine(7)+EQProcessor(5)のEpochDomain直接経路も統合対象に。**P0-1 Router肥大化防止(interface-level)**: Allowed(route/enqueue/observer factory) vs Forbidden(state/policy/decision)をインターフェース固定。Policy系は別クラス。**P0-10 forceFinalPurge前にdiagnostic dump**: emitFinalRegistrySnapshot()で診断情報保全。**P0-11 Admission閾値queueCapacity*ratio**: 70%/90%固定→動的比率。**P0-10 maxRegistryAge条件明確化**: 24h purgeは完了済みエントリのみ。**P2-9 EmergencyDrain ビルド分岐**: Debug fatal / Release telemetry+leak quarantine。**CI回帰検出**: 0固定CI + enqueueRetire/reclaimRetired監視。 |
| 2026-06-07 | **18.0** | 第17回第三者レビュー（長期保守耐性）に基づき5項目追加。**P0-1 Router Policy/State物理分離**: コンパイル依存レベルでRouter→Policy参照のみ可、Policy→Router参照禁止。**P0-8 Observer寿命責任明文化**: ObserverToken/RAII化、Router停止後のObserver残存対策。**P1 CI gate ASTベース昇格**: grep→clang-tidy/AST ruleベースへ移行。**P0-10 purge前統計保存**: RetireRegistrySummary(entry count/oldest gen/duplicate count/quarantine count)を定期保存。**P2-9 EmergencyDrain結果オブジェクト**: DrainResult{drainedCount,remainingCount,remainingBytes,timeoutReason}生成。総評: 直ちに破綻する欠陥はほぼ解消済み。 |
| 2026-06-07 | **18.1** | **6ツール駆使監査結果を反映（Serena/CodeGraph/ccc/graphify/DeepSeek/semble/grep）**: ConvolverProcessor の EpochDomain 依存4箇所を発掘（計画未記載）→ P0-8 完了条件に追加（28→32箇所）。EQProcessor の coordinator 二重経路を P0-5 に反映。`RefCountedDeferred.h` の EpochDomain 残留を P1-2 にタスク追加。P2-10 ConvolverProcessor EpochDomain 追跡を新設。CI gate に ConvolverProcessor 追加。付録A 更新。 |
| 2026-06-08 | **18.2** | **第18回第三者レビュー（実運用破綻耐性最終確認）に基づき5項目追加。**P0-8 呼び出し監査→所有者監査に拡張: EpochDomain所有者一覧(class member/constructor arg/function arg/local static/global singleton)を機械抽出。P0-10 RetireRegistrySummary リングバッファ化: 最新100件保存、shutdown/forceFinalPurge/Barrier timeout時に出力。P1-2 RefCountedDeferred static_assert(false) 使用禁止（将来の回帰点防止）。P1-3 AST CI に型所有禁止ルール追加: EpochDomain*/&/member の新規追加禁止。P1-3 Router interface-only 固定: Policy参照取得API禁止(router->policy()不可)。P0-8 ObserverLease generation snapshot 保持: Router生存確認ではなく生成時情報固定化。P2-9 EmergencyDrain 残留種別ヒストグラム出力: remainingObjectTypeHistogram{DSPRuntime,Snapshot,Cache,...}。総評: 現時点で追加価値が高い改善は補強レベル。最重要は全32箇所のEpochDomain直接経路除去の完遂。 |
| 2026-06-08 | **18.3** | **第19回第三者レビュー（運用時管理不全防止）に基づき6項目追加。**P0-8 所有者監査対象拡張: typedef/using alias/template parameter/trait specialization までカバー。P0-8 ObserverLease 診断情報強化: observerId/routerInstanceId/creationTimestamp 追加。P1-3 Router public API数上限: API追加時レビュー必須の運用ルール化。P1-10 到達不能コード監査（新規）: Router移行後に参照ゼロのクラスを削除候補として一覧化。P0-10 RetireRegistrySummary 偏り情報追加: topObjectType/topDuplicateType パーセンテージ。P2-9 EmergencyDrain 残留経過時間追加: oldestAge/newestAge を RemainingObjectTypeHistogram に追加。総評: 大規模アーキテクチャ変更不要。P0-8 32箇所除去が最優先。 |
| 2026-06-08 | **18.4** | **第20回第三者レビュー（長期運用時管理不全防止）に基づき7項目追加。**P0-8 完了条件二段階化: Phase-A(Retire用途 EpochDomain=0) / Phase-B(RCU用途許容明確化)。P0-1 Router fast-path表現修正: Policy直参照許可→router.enqueueDSPFast内部最適化として Router API の範囲内に。P1-3 Router APIカテゴリ固定: enqueue/publish/observer/telemetry のみ許可。P0-10 RetireRegistrySummary 定期スナップショット: 1時間毎の定期保存契機追加。P2-8 aging昇格統計追加: agingPromotionCount/maxAgingLevelReached。P2-9 DrainResult年齢情報追加: oldestRetireAge/averageRetireAge。P1-10 regression count追加: 完了後も過去30日のdirect call追加回数を可視化。総評: v18.4は全バージョン中最も成熟。最重要は Direct calls=0, Owners=0, Dead code=0達成とCI維持。 |
| 2026-06-08 | **18.5** | **【第21回6ツール総合監査】全ツール(grep/Serena/CodeGraph/ccc/graphify/semble)駆使のEpochDomain網羅調査。**計画上の32箇所→実態124参照・27ファイル。EQProcessor(5→~30)、AudioEngine(7→~20+)、新規発見ファイル16。epochDomain() public accessor 発見。EpochCore.h 発見。Phase-B advanceEpoch カウント(19) 発見。付録A全面更新。総評: P0-8完了条件の再定義が必要。32箇所以上のEpochDomain依存が存在。Phase-Bの範囲は計画想定の2倍以上。 |
| 2026-06-08 | **18.6** | **【第22回第三者レビュー: 完了条件精密化フェーズ】6項目追加。**P0-8 Public EpochDomain exposure count 独立完了条件化(public getter/reference/pointer/accessor)。P0-8 AudioEngine.Threading.cpp 独立PR化(13参照集中)。P1 advanceEpoch(20)呼び出し理由一覧をPhase-B前に先行実施。P1 RCUReaderが新たなEpochDomain代理にならないことをPhase-B完了条件に追加。P1 DeletionQueue/SnapshotRetireManager をPhase-B正式対象へ昇格。P0-1 fast-path旧記述「call-siteから直接Policy Laneへ委譲可能」を完全削除。P2 EpochDomain_REFERENCE_STATUS.md 提案(124参照の削除/置換/残存/Dead status付与)。総評: アーキテクチャ方向性は妥当。最大リスクは設計ではなく124参照の整理漏れ・分類漏れによる移行未完了状態。 |
| 2026-06-08 | **18.7** | **【第23回6ツール総合再監査】全6ツール(grep/Serena/ccc/CodeGraph/graphify/semble)再駆使。**advanceEpoch 20に上方修正(新規発見: AudioEngine.CtorDtor.cpp:115)。RuntimePublicationOrchestrator.cpp 間接依存(advanceRetireEpoch)発見。ConvolverProcessor Lifecycle.cpp snapshotRcuEpoch 4回未計上。enterGlobalReader/exitGlobalReaderパターンがRCU間接依存追加。CodeGraph MCP entity抽出がEpochDomain未捕捉(要full index)。graphify DeepSeek BFS depth=2では不十分(3nodesのみ)。総評: 直接依存カウントはほぼ収束。残るは間接依存・未捕捉ツール制約の確認。 |
| 2026-06-09 | **18.8** | **【第24回6ツール間接依存経路完全追跡】全6ツール(grep/Serena/CodeGraph/ccc/graphify/semble)再駆使。**AudioEngine 8個のRCUラッパー完全マッピング(m_epochDomain委譲一覧)。ConvolverProcessor getRcuProvider()->snapshotRcuEpoch() 5呼び出し特定。enterGlobalReader/exitGlobalReader 6組特定。RuntimePublicationOrchestrator advanceRetireEpoch 1呼び出し確認。計5系統の間接依存チェーン(A-E)を付録Bとして文書化。総評: 直接依存(124参照)に加えて間接依存も含めた全EpochDomain到達経路が明確化。Phase-A/B計画時の間接経路対応漏れ防止。 |
| 2026-06-09 | **18.9** | **【第25回第三者レビュー: 実運用破綻耐性最終補強】6項目追加。**間接依存完了条件(Indirect dependency count)新設。RCUReader::domain()完全廃止(P0-14)。AudioEngine.Threading.cpp 3系統分割(P0-15)。advanceEpoch呼び出し頻度監査(P1-14)。ConvolverProcessor epoch利用境界明文化(P1-15)。RetireRetryPolicy Router内統合(P1-16)。124参照状態分類(P1-17)。監査軸シフト: 「直接依存排除」→「代理権威を作らない」。総評: アーキテクチャ変更不要。完了条件の強化と監査軸のシフトが本質。 |
| 2026-06-09 | **18.10** | **【第26回第三者レビュー: 実運用移行後再発防止】7項目追加。**許容間接依存を関数単位で固定化(P0-13強化)。AE.Threading.cpp 3PR分割再確認(P0-15強化)。advanceEpoch意味分類(EQProcessor.Parameters.cpp 用途別集計)(P1-14強化)。EpochDomainReaderGuard直接生成禁止(P1-18新規)。Public Exposureを返り値型ベースAST監査に昇格(P1-19新規)。RefCountedDeferred deprecated/削除(P1-20新規)。DeletionQueue系独立移行対象昇格。監査軸シフト: 「例外の封じ込め」段階へ移行。総評: Router設計そのものより「許容した例外経路が将来拡張されて権威復活しないこと」が最大の実運用リスク。 |
| 2026-06-07 | **18.11** | **【Phase-D実装コード完了】全4フェーズの実装コード完了。**Phase-D: RCUReader::domain()廃止、RefCountedDeferred旧API削除、ConvolverProcessor advanceEpoch集約、IEpochProvider 3責務完全分離(IReaderEpochProvider/IPublicationProvider/IRetireProvider)、CI Gate拡張(grepベース: alias/template/Router API/サブIF個別監視追加)。ビルド成功。CI Gate数値改善(露出4→3, enqueueRetire 5→4)。**ただしv18.3完了条件の「ASTベースCI(clang-tidyルール)による型所有監査」は未達。別Phase-Eとして残存。** |
| 2026-06-08 | **Phase-E** | **【Phase-E全6項目完了 + P7 Partial】退行防止の恒久化完了。**P1: Architecture Regression Snapshot統合。P2: ownerThreadId Diagnostic化(`#ifndef NDEBUG`)。P3: ISRRetireRouter::domain()削除。P4: Python AST scanner作成(tools/check-epochdomain-ast.py、二層構成)。P5: EpochDomainReaderGuard撤去+RCUReaderGuard move semantics追加。P6: コメント監査(SnapshotRetireManager.h 1件修正)。P7(Partial): C4996抑制(#pragma warning suppress + 実コード移行)。Releaseビルド警告ゼロ確認。CI Gate Architecture Regression全指標Delta=0。ConvoPeq.md/refactoring_plan_phase_e.md同期更新。 |

## 前提

本計画は以下の成果物をベースとする:

- `doc/work21/notfinished7.md` — 未達成ポイントの抽出原稿
- `doc/work21/notfinished7_validation_report.md` — 上記の検証結果（Serena/CodeGraph/ccc/graphify/semble による全ツール駆使）
- `doc/work19/refactoring_plan_v2.md` — Phase-1 Authority純化計画（PR-0〜PR-8）
- `doc/work20/refactor-bridge-runtime-phase-2.md` — Phase-2 改修計画（S-A1〜C-7）

### 検証で使用したツールと結果

| ツール | 用途 | ステータス |
| -------- | ------ | ----------- |
| Serena MCP | シンボル検索・参照関係・実装解析 | ✅ 全シンボル特定 |
| CodeGraph MCP (CodeGraphContext v0.4.13) | 11397 entities indexed | ✅ インデックス完了 |
| cocoindex-code (ccc) | ASTベースコード検索（8,701 chunks） | ✅ 網羅検索 |
| graphify (DeepSeek backend) | 11633 nodes, 14188 edges, 1207 communities | ✅ グラフ構築完了 |
| semble | キーパターン全出現追跡 | ✅ 6パターン横断検証 |
| grep | 最終確認 | ✅ |

## 0. 現状サマリ

### 到達率: 70〜75%（notfinished7.md の7クレームに対する対応率）

| 領域 | 状態 | 残余 |
| ------ | ------ | ------ |
| Snapshot Coordinator 責務分離 | ⚠ 部分対応 | SnapshotRetireManager 未統合 |
| Retire 経路統合 | ❌ 分散 | 3系統の enqueueRetire + 6箇所の reclaim |
| Fade 状態分散 | ❌ 分散 | SnapshotFadeState + CrossfadeRuntime + FadeAccumulator |
| Atomic ordering グローバルモデル | ❌ 未対応 | HBコメントのみでコード不在 |
| Snapshot 生成比較ロジック | ✅ 設計通り | 2層設計として成立 |
| Thread ID 依存 | ⚠ 対応済みの課題あり | EpochDomainReaderGuard 存在するが ownerThreadId 残存 |
| RT境界跨ぎ Reclaim | ❌ 未対応 | tryReclaimResources が Timer 上で delete 実行 |

### notfinished7.md の正確性評価

| クレーム | 正確性 | 実装コスト |
| ---------- | -------- | ----------- |
| ① Epoch/RCU責務分散 | ⚠ 部分的に正確（SnapshotRetireManager の存在を未記載） | **高** |
| ② SnapshotCoordinator God Object | ✅ 正確（SlotStore/FadeState抽出済みだが統合未完了） | **中** |
| ③ Atomic ordering 局所最適 | ✅ 正確（HBコメント精緻だが体系モデル不在） | 低〜中 |
| ④ Snapshot生成比較二重化 | ⚠ 過剰主張（意図された2層設計） | 低（ドキュメント修正） |
| ⑤ RT境界跨ぎ | ✅ 正確（tryReclaimResources のRT呼び出しは実害あり） | **高** |
| ⑥ Thread ID依存 | ✅ 正確だが影響限定（EpochDomainReaderGuard が実質保護） | 低 |
| ⑦ Fade密結合 | ✅ 正確（CrossfadeRuntime の別fadeを未記載） | 中 |

### 検証で発見された notfinished7.md の見落とし（5項目）

1. **`SnapshotRetireManager` 未統合（最重要）**: クラスは `src/core/SnapshotRetireManager.h` に抽出済みだが、全ソースコードで参照ゼロ（semble確定）。Phase 5 で抽出された orphan クラス。
2. **`tryReclaimResources` 経由の RT 上 Reclaim（実害あり）**: `AudioEngine::tryReclaimResources()` (AudioEngine.Threading.cpp:75) が `m_epochDomain.reclaimRetired()` を Timer コールバック上で呼ぶ。`DeferredDeletionQueue::reclaim()` は lock-free だが最終的に `delete`（デストラクタ）を実行するため、Audio Thread 上での破棄になりうる。
3. **`CrossfadeRuntime` の独立 fade 状態**: `AudioEngine::crossfadeRuntime_` (convo::isr::CrossfadeRuntime) が独立した fade 管理（isPending/useDryAsOld/getQueuedFadeTimeSec/getDryScaleTarget）を持つ。SnapshotCoordinator の fade とは別次元。fade 状態がさらに分散。
4. **`ISRDSPHandle` に別の retire/reclaim 経路**: `DSPHandleRuntime::retire()` / `reclaim()` / `quarantine()` が独立経路として存在。ISRRetireRuntimeEx とは別レイヤー。
5. **`enqueueDeferredDeleteNonRtWithResult` のフォールバックパターン**: `AudioEngine.h:3170-3206` で初回 enqueue 失敗 → `reclaimRetired()` → 再 enqueue のパターンが存在。

### Phase 定義

```text
Phase-A: Retire Authority 一元化（クリティカル）
  → SnapshotRetireManager 統合 + enqueueRetire 経路統一 + reclaim RT排除
  → PR-1〜PR-3 で完了

Phase-B: SnapshotCoordinator 責務完結
  → SlotStore/FadeState の完全分離 + SnapshotRetireManager 接続
  → PR-4〜PR-5 で完了

Phase-C: Fade 状態整理
  → SnapshotFadeState / CrossfadeRuntime / FadeAccumulator の責務固定
  → PR-6 で完了

Phase-D: 安全性・検証性向上
  → ownerThreadId 排除 + Atomic ordering モデル文書化
  → PR-7〜PR-8 で完了
```

---

## ★ v18.12 への構造的改訂（第4回〜第28回レビュー + 6ツール総合監査対応）

### P0 即時修正必須（第4回監査 特定）

| # | 問題 | ソース検証結果 | 対応 |
| --- | ------ | --------------- | ------ |
| P0-1 | **delete 実行経路6系統の分散 + 単一経路の脆弱性 + Router Authority Hub 化防止**: Router に retire routing/observer生成/purge管理/telemetry/admission連携が集積。唯一入口→唯一判断点→唯一状態保持点 へ進化する古典的崩壊パターン。`router->canRetire()`/`router->canPublish()`/`router->shouldDelay()` の追加リスク。**v18.6: 設計書内に「call-siteから直接Policy Laneへ委譲可能」の旧記述が残存（260行目）。v18.4で修正済みだが後続章に矛盾が残ると実装者ごとに解釈が分かれる。** | **v18.0修正** — CIガード+interface制約に加え、コンパイル依存レベルでの物理分離が必要。**v18.6: 旧記述を完全削除。Router内部最適化のみ許可に統一。** | **v18.0: Router を Policy/State から物理分離**。コンパイル依存: `Router → Policy` 参照のみ可、`Policy → Router` 参照禁止。Router は `class ISRRetireRouter` として Policy/State/Decision をメンバに持てない（コンパイルエラー）。**Allowed**: route / enqueue / observer factory。**Forbidden**: state / policy / decision。Policy は別クラス `ISRRetirePolicy` に閉じ込め。Router は stateless dispatcher に徹する。**Retire 2-lane 化**: Fast Lane（SPSC）+ Slow Lane（Worker Queue）。**v18.6: fast-path記述完全統一**。`router.enqueueDSPFast(...)` として Router API の内部最適化のみ許可。設計書全体で「call-siteから直接Policy Laneへ委譲」という表現を削除。すべて「Router API経由」に統一。 |
| P0-2 | **DSPHandleRuntime が独立した retire/reclaim/quarantine を持つ**: ISRRetireRouter と二重体系 | **一部妥当→v8.0修正** — 実コード確認: `DSPHandleRuntime::retire()`/`reclaim()`/`quarantine()` は slot 状態遷移（metadata管理）のみ。`instance` の `delete` は行わない。**DSPHandleRuntime は state layer（slot metadata machine）であり、memory layer（RetireRouter）と明確に分離されている。v7.0までの「二重体系」は過剰修正リスク（false positive）。** | DSPHandleRuntime を pure slot metadata manager として維持。Router とは「state layer vs memory layer」の責務分離を文書化。削除や強制統合は行わない。 |
| P0-3 | **SnapshotRetireManager の位置づけ矛盾**: "QUEUE ONLY" だがCoordinator関係あり？ | **却下（実コード確認）** — `SnapshotRetireManager::retire(GlobalSnapshot*, uint64_t epoch)`：epoch パラメータ受取、`DeletionQueue::enqueue()` に委譲のみ。Coordinator 参照/ fade 状態/ lifecycle 判断は一切なし。**純粋なQUEUE ONLY**。 | 変更不要。文書にコード確認結果を追記。 |
| P0-4 | **CrossfadeRuntime/SnapshotFadeState/FadeAccumulator 三重構造** | **一部妥当** — 3者は異なる責務層（Execution/Lifecycle/Mixing）だが、fade の truth source は1つであるべき。CrossfadeRuntime が Execution 層の「現在のゲイン値」を、SnapshotFadeState が Publication 層の「fade中か」を、FadeAccumulator が DSP 層の「ブレンドパラメータ」を担当。**階層は明確だが、書き込み元の一貫性が不足**。 | FadeAuthority 単一書き込み原則を強化。PR-B-0 で全3層の書き込み権限マトリクスを明示。 |
| P0-5 | **retry パターン（再入性違反）**: `EQProcessor::enqueueDeferredDeleteWithFallback()` が **最大4回の retry** ループ（`enqueue→reclaimRetired→advanceEpoch→再enqueue`）。`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` も `reclaimRetired()` 後に再 enqueue。**6ツール監査で判明: EQProcessor は既に coordinator 優先経路あり**。 | **妥当（実コード確認・6ツール監査）** — EQProcessor.Core.cpp:32-65 で `kMaxRetry=4` の retry ループ。**重要: EQProcessor は既に `m_retireCoordinator->enqueueRetire()` を優先経路上に持ち、`m_epochDomain.enqueueRetire()` は coordinator 未設定時のフォールバック経路としてのみ存在する**。両経路とも retry ループあり。AudioEngine.h:3179-3193 でも `reclaimRetired()` 後に再 enqueue。**再入性 + スレッド競合 + 意図しない delete タイミング移動** のリスク。 | **v18.0修正: coordinator 常時設定後の fallback 経路削除を明記**。EQProcessor: coordinator 設定完了後 `m_epochDomain.enqueueRetire()` fallback 経路を削除。両経路の retry ループを `drop + telemetry + deferred retry queue` に統一。AudioEngine も同様。`kMaxRetry=4` ループ削除。RT 即座に drop。Non-RT では bounded deferred retry queue（maxRetryCount=3, maxRetryAgeMs=5000, 指数バックオフ × random(0.8~1.2) jitter）。 |
| P0-6 | **`epochDomain.reclaimRetired()` の直接呼び出し**: Timer/Coordinator/Quarantine の3経路から直接呼ばれる | **妥当（実コード確認）** — `tryReclaimResources()` から `claimRetired()`、`drainDeferredRetireQueues()` から `reclaimRetired()`、`EQProcessor` の retry 内でも `reclaimRetired()`。**reclaim が「直接実行 API」として露出**。 | `reclaimRetired()` を deprecated API 化 → `requestReclaim()`（enqueue only）に置換。全直接呼び出しを Worker enqueue 経由に変更。debug assert で直接 `delete` を検出。 |
| P0-7 | **Retire Commit Barrier 不在 + shutdown デッドロック + 観測指標不足**: SnapshotCoordinator の直接呼び出しにより snapshot swap → fade → retire → delete の順序保証がない。**shutdown 時の hard barrier デッドロック**: fade完了待ち中に Audio が停止済みで fade が進まない状態。加えて barrier 待機理由の観測指標がなく、実機障害時に「barrier timeout」としか分からない。 | **妥当（v15.0修正）** — barrier 待機理由を分解して記録しないと、実運用で「なぜ止まったか」の診断が不可能。 | **v15.0: Barrier telemetry 詳細化**: `waiting_epoch` / `waiting_fade` / `waiting_enqueue` / `waiting_shutdown` を個別カウンタとして記録。Telemetry に出力。soft barrier（通常運用: 優先制御・タイムアウト許容）。hard barrier（安全停止: 全条件成立までブロック）。shutdown 時は hard barrier に `forceFadeComplete()` 許可 + barrier timeout(100ms)。Phase3 では bypass。 |
| P0-8 | **全 public retire API の Router 経由統一 + 所有者監査（★最優先: 発見124参照・27ファイルに拡張）**: 計画上の32箇所(4components)から実際は124参照・27ファイルに拡大。**v18.5: 6ツール総合監査で計画の大幅過小評価が判明。AudioEngine(7→~20+ 8ファイル)、EQProcessor(5→~30 4ファイル)、ConvolverProcessor(4→~7+ 5ファイル)、新規16ファイル発見。v18.6: 完了条件精密化。** | **6ツール完全駆使** — 124参照・27ファイル。**v18.6: 以下4点を新規指摘**。(1) `epochDomain()` public accessor が完了条件主指標化不足。(2) AudioEngine.Threading.cpp 移行ボトルネック(13参照集中)。(3) advanceEpoch(20)呼び出し理由未分類。(4) RCUReaderがEpochDomain新たな代理リスク。 | **v18.6: 完了条件4点強化**。**(1) Public EpochDomain exposure count = 0 追加**: public getter/reference/pointer/accessor wrapper を別指標化。AudioEngine.h:3225 `epochDomain()` 削除必須。**(2) AudioEngine.Threading.cpp 独立PR化**: 13参照集中のため一括切替ではなく独立PRで分割移行。**(3) advanceEpoch(20)理由一覧をPhase-B前に作成**: parameter update/snapshot publish/cache rebuild/cleanup 分類→不要削除。**(4) RCUReader代理防止**: Phase-B完了条件に「RCUReaderがEpochDomain公開代理になっていないこと」追加。**Phase-A**: enqueueRetire(5)+reclaimRetired(8)=13箇所削除。AudioEngine.Threading.cpp は独立PR並行移行可能。**Phase-B**: advanceEpoch分類→不要削除 + public accessor削除 + RCUReader防止 + DeletionQueue/SnapshotRetireManager正式対象化。**最終目標**: EpochDomain型名が公開API非出現 + epochDomain()削除 + 全27ファイル依存一覧化 + EpochDomain_REFERENCE_STATUS.md(124参照のstatus明示)。 |
| P0-9 | **Timer reclaim 完全禁止**が過剰: periodic cleanup消失・latency drift回収不能・backlog成長リスク | **v10.0修正** — 完全禁止ではなく「Timer → enqueue only（soft rate limited）, Worker → delete」が正しい。Timer からの監視自体は必要。 | **Timer enqueue 許可（soft rate limited）**: `tryReclaimResources()` は `requestReclaim()`（enqueue only）を rate limit（例: 最大100Hz）付きで許可。Timer からの直接 `reclaimRetired()` / `delete` は禁止。Soft limit 超過時は telemetry + 次回 Timer に先送り。 |
| P0-10 | **RetireRegistry 永続成長・purge・診断ダンプ・統計保存・maxRegistryAge明確化 + Summary リングバッファ + 偏り情報 + 定期スナップショット**: Registry 肥大化対策。purge 前に統計保存。**v18.3拡張: 統計だけでは偏りが分からない。「何が多く残っているか」の偏り情報が障害解析に有用。**v18.4: 1時間毎の定期 summarySnapshot() を保存契機として追加。** | **妥当（v18.3修正）** — 件数だけでは「Cacheが異常に多い」等の偏りを score 化できない。**v18.4: 定期スナップショット契機が不足。** | **v18.3: RetireRegistrySummary に偏り情報追加**: topObjectType（割合最大のオブジェクト種別と%）/ topDuplicateType（二重retire最多種別と%）を各 Summary エントリに追加。例: `topObjectType="DSPRuntime(80%)"`, `topDuplicateType="Cache(100%)"`。リングバッファ 100 件維持。保存トリガ: (1) 定期パージ時、(2) forceFinalPurge時、(3) Barrier timeout時、(4) EmergencyDrain完了時。**v18.4: 保存トリガ(5)追加: 1時間毎の定期 summarySnapshot()**。長期運用時の経過観測に使用。リングバッファ100件に収まらない場合は古いエントリから削除。各エントリ: entry count / oldest generation / newest generation / duplicate retire count / quarantine count / total retired bytes / topObjectType / topDuplicateType / timestamp。 |
| P0-11 | **Telemetry → RetireBackpressurePolicy 自動反映 + ヒステリシス + EMA growth rate + 動的閾値**: queueDepth の固定閾値(70%/90%)は IR大量切替・小規模プリセット切替・オフラインレンダリングで適正値が変わる。 | **妥当（v17.0修正）** — 固定値では運用状況の変化に対応不可。 | **v17.0: PublishAdmissionState 閾値を queueCapacity * ratio に変更**: highWatermark=queueCapacity*70%, lowWatermark=queueCapacity*50%。EMA-based depthSlope 維持（α=0.5, 500ms移動平均）。`AdmissionScore = queueDepth + depthSlope_ema * k`。将来的に ratio を設定可能に。三段階制御: cache retire 停止 → DSP retire 優先 → Fast Lane 優先度増加 + 入口制御（publish間隔延長/停止）。 |
| P0-12 | **SnapshotCoordinator API 凍結**: P0-8 一括切替後も、保守者が責務外の API（canRetire/canDelete/canPublish/epoch管理）を追加するリスク。計画書は observe/updateFade のみに縮小予定だが、実コードは依然として fade管理/slot管理/retire管理/epoch管理を保持。 | **妥当（v14.0新規）** — 移行完了後も責務肥大化の再発リスク。 | **SnapshotCoordinator public API 固定化**: 許可: `observeCurrentRuntime()`, `updateFade()` + 内部ヘルパー（`advanceFade`, `tryCompleteFade`, `isFading`）。禁止: epoch管理/publish/retire/slot操作。**CI gate**: SnapshotCoordinator.h の public メソッド追加を検出（diff 監視）。Phase-B 完了条件に API 一覧の文書化を追加。 |
| P0-13 | **間接依存の完了条件が弱い: 「直接依存=0」だけでは間接チェーンA〜E(22+回)が残存するリスク。** 計画では direct call count を厳密に監査しているが、RuntimePublicationOrchestrator→advanceRetireEpoch→EpochDomain のような間接経路の許容条件が曖昧。**v18.9新規。v18.10強化: 許容対象を関数単位で固定化。**「advanceEpoch系は許容」では広すぎる。 | **v18.8 6ツール間接依存追跡で確認** — チェーンA〜Eの5系統・22+回の間接呼び出しを文書化済み。しかし「許容/禁止/要移行」の分類がない。 | **間接依存完了条件を新設し、許容対象を関数+呼び出し元の組で固定化**: Phase-B終了時に `Indirect EpochDomain dependency count` を計測し、全間接経路を **許容(permanent)** / **許容(temporary)** / **禁止** / **要移行** の4段階で明示分類する。以下を完了条件に追加: (a) **許容(permanent)** — EBR基盤として永続的に残す。呼び出し元+関数の組で固定: チェーンA(ConvolverProcessor::updateIRState/prepareToPlay/releaseResources/syncStateFrom/shareConvolutionEngineFrom → current():5回)、チェーンB(ConvolverProcessor::Runtime/Lifecycle/StateAndUIのGlobalGuard → enterReader/exitReader:10回)。(b) **許容(temporary)** — 将来削除予定または再評価が必要。チェーンC(RuntimePublicationOrchestrator::commitPublish → advanceEpoch:1回のみ)。**「advanceEpoch系は許容」ではなく「RuntimePublicationOrchestrator::commitPublishからのadvanceEpochのみ許容」** と明記。将来の `advanceEpochIfNeeded()` / `advanceEpochForCleanup()` 等の増殖を禁止。**完了時 temporary = 0**。(c) **禁止**: チェーンE(RCUReader::domain()→EpochDomain&直接公開,3箇所) — P0-14で完全廃止。(d) **要移行**: チェーンD(publish委譲,6+回) — Router::publishEpoch()へ移行。(e) v18.10追加: EpochDomainReaderGuard直接生成=0 — P1-18。(f) v18.11追加: EpochDomain到達可能クラス数 = 2 — Router + RuntimePublicationOrchestratorのみ。**(g) v18.12追加: advanceEpoch呼び出し元数 = 1**: `RuntimePublicationOrchestrator::commitPublish()` のみ。CI gate: `grep -rn '\.advanceEpoch(' src/` の呼び出し元ファイル数をカウントし、許容1以外からの呼び出しをブロック。許容関数リストを `EpochDomain_ALLOWED_INDIRECT.md` に固定化し、変更には全承認必須。**CI gate**: Indirect EpochDomain 参照数（許容対象を関数単位で除外した値）の増加を監視。Phase-B完了後に新たな間接経路が追加された場合はブロック。 |
| P0-14 | **RCUReader::domain() が最大の再侵入ポイント: EpochDomain&を直接返す public accessor。** 現在のCIは `EpochDomain&/EpochDomain*` を監視するが、`auto& d = reader.domain()` が残ると実質的に `EpochDomain&` を再公開しているのと同じ。計画書自身が P1-12 で問題視するも、廃止ではなく「代理防止」に留まっている。 | **v18.8付録BチェーンEで確認** — RCUReader.h(7参照)、AudioEngine.h:3382/EQProcessor.h/ConvolverProcessor.h:1151 の3箇所で保持。`domain()` が EpochDomain& を返す。 | **RCUReader::domain() を完全廃止する**: 代わりに限定公開API `snapshotEpoch()`(current epoch取得), `readerToken()`(reader識別子取得), `activeReaders()`(診断用) のみを提供。RCUReader を Router 内部実装に変更し、外部から EpochDomain 型を一切露出しない。**移行手順**: (a) RCUReader に限定API追加。(b) 全呼び出し元を限定APIに変更。(c) `domain()` を `[[deprecated]]` 化。(d) Phase-B終了時に削除+EpochDomain&戻り値を型レベルで禁止。**CI gate**: `RCUReader::domain()` の使用を clang-tidy ルールで禁止。`EpochDomain&` が RCUReader の public API から返されないことを確認。 |
| P0-15 | **AudioEngine.Threading.cpp 独立PR化では不十分: 現状13参照が publication/reader/retire の3 subsystem 混在。** 現計画の「独立PR」はファイル単位の分割であり、論理的な責務分離が不十分。実運用では1つのPRに3種類の変更が混入しレビュー品質が低下する。 | **v18.8 6ツール監査で確認** — AudioEngine.Threading.cpp:30-71 に publish(1), current(1), advanceEpoch(1), enterReader(1), exitReader(1), enqueueRetire(1), activeReaderCount(1), reclaimRetired(3) が混在。3論理サブシステムに跨る。 | **AudioEngine.Threading.cpp 専用PRをさらに3系統へ分割**: (a) **Publication PR** (publish,current,advanceEpoch): Router::publishEpoch()委譲。SnapshotCoordinator連動。(b) **Reader PR** (enterReader,exitReader,activeReaderCount): RCUReader限定API化(P0-14)と同時実施。(c) **Retire PR** (enqueueRetire,reclaimRetired): Router::enqueueRetire()委譲。Phase-A対象。**完了条件**: 各PR完了後に AudioEngine.Threading.cpp の責務範囲が単一 subsystem に限定されていることを確認。混在状態でのマージ禁止。 |
| P0-16 | **AudioEngine.Threading 3PR分割後の統合テスト計画が不足: 単体PRは設計として正しいが、publish→advanceEpoch→enqueueRetire→reader enter の順序保証がない。** 実運用で壊れるのは単体PRではなく3PRの組み合わせ。**v18.11新規。** | **設計上のリスク** — 3PR分割は論理分離として正しいが、相互の順序依存関係をテストする計画がない。例: "publish→retire" の順序が崩れると retire が epoch 未発行の状態で実行される。 | **Phase-B完了条件に Cross-path integration test を追加**: 最低限以下の4パターンを固定化し、CIで自動実行。(1) **publish→retire**: publish後に enqueueRetire が正しく epoch を参照できるか。(2) **reader→retire**: reader退出後に retire が安全に実行できるか。(3) **publish→reader**: publish中に reader が一貫性のある snapshot を読めるか。(4) **shutdown→retire**: shutdown中に retire がデッドロックしないか。各テストは3PR完了後のマージ前に実行し、失敗時はマージ禁止。テストコードは `tests/` または `test/` ディレクトリに配置。 |

### P1 破綻防止（第4回〜第10回監査）

| # | 問題 | ソース検証結果 | 対応 |
| --- | ------ | --------------- | ------ |
| P1-1 | **tryReclaimResources の非決定的GC**: Timer上でreclaim+retry | **一部妥当** — 実体確認: `m_epochDomain.reclaimRetired()` 単一呼び出しのみ。retry/fallback は**存在しない**。しかし Timer コンテキストでの解放実行は RT 違反リスク。 | PR-2 の方針維持: `reclaimRetired()` 削除＋退避圧力監視のみ。WorkerThread へ委譲。 |
| P1-2 | **EpochDomain deprecated → error 化が早すぎる + RefCountedDeferred.h 未処理（v18.2: 使用禁止）**: ISRClosure/DSPHandle が未依存。`RefCountedDeferred.h` に `epochDomain.enqueueRetire()` が残存。**v18.2: 未使用テンプレートの放置は危険 — 将来開発者が使用した瞬間に旧経路が復活する。** | **妥当** — `ISRClosure` と `ISRRetireRuntimeEx` が `enqueueRetire()` を使用。error 化は既存コンポーネント破壊。**RefCountedDeferred.h の EpochDomain 依存は真の dead code。放置すると将来の回帰点になる。** | 段階的廃止目標を v7（最終 Phase）に延期。現状 `[[deprecated]]` 維持。**v18.2: RefCountedDeferred.h に `static_assert(false, "Use Router-based retire instead")` を追加**して使用をコンパイル時に禁止。または Router 版（Router::enqueueRetire 経由）に置換。CI gate で `#include "core/EpochDomain.h"` 削除確認。 |
| P1-3 | **RetireRouter の責務制限 + CIガード（v18.12: 責務数監査追加）: Router が retire + snapshot + fade + epoch + queue を全て握る「Runtime God Router 化」リスク。** API数制限だけでなく責務カテゴリ数の監査が必要。**v18.12: Routerが「新しい巨大権威」になるリスク。** | **一部妥当。v18.12: API数上限だけではカテゴリ拡大を防げない。** | **v18.12追加: Router責務数監査**: 許可カテゴリは Retire / Publication / Reader の3系統のみ。これら以外のカテゴリ追加時は全承認必須。CI gate: Routerのpublicメソッドをカテゴリ別に分類し、許可カテゴリ以外を検出したらブロック。API数制限(最大10)+カテゴリ制限(最大3)の複合チェック。 |
| P1-4 | **SnapshotRetireManager 孤立＋削除による責務統合のやりすぎ**: 削除により Snapshot lifecycle と Retire lifecycle が結合。crossfade/fade/epoch の境界曖昧化。テスト単位消失。 | **v10.0修正** — 削除ではなく薄層 facade 化が正しい。Serena 検証で ZERO references だが、コードとしては意味ある境界。 | **SnapshotRetireFacade として復活**（削除方針撤回）。state owner ではない。routing + telemetry + ordering validation のみ。実 delete は Router 経由。 |
| P1-5 | **FadeAuthority 単一書き込み（v10.0修正: 独立書き込み復活）**: 3層は異なる時間スケールを持つ。単一書き込み強制は DSP 的に危険（click防止 crossfade 破綻、IR 切替時に急峻変化）。 | **v10.0修正** — 単一書き込みは撤回。UI coarse 層・DSP sample-accurate 層・analysis offline 層は各々独立した書き込みを許可。 | **v10.0: Fade 3層 独立書き込み復活**。単一書き込み方針を撤回。各層は自身の時間スケールで独立書き込み。FadeAuthority は同期プロトコルの定義のみ行い、書き込み権限制限は行わない。PR-B-0 に FadeClockDomain（単一時間基準）を導入。 |
| P1-6 | **SnapshotFadeState 独立 fade 管理（v8.0修正）**: SnapshotFadeState 削除は過剰。CrossfadeRuntime（sample accurate）と SnapshotFadeState（state transition coarse）は異なる時間スケールを持つ。統合により UI と DSP のタイミング競合リスク。 | **v8.0で方針転換** — 削除ではなく「階層間同期プロトコル」を定義する方向に変更。CrossfadeRuntime の `gain_` を truth とし、SnapshotFadeState は coarse な状態表示に制限。 | SnapshotFadeState 維持（削除しない）。CrossfadeRuntime の gain 値を定期的に SnapshotFadeState に同期。SnapshotFadeState は UI 表示と coarse 状態遷移のみ担当。 |
| P1-7 | **SnapshotCoordinator がまだ部分 God Object**: SlotStore/SnapshotFactory/FadeState/RetireManager が間接接続されたまま。現在の責務: current slot管理 / target slot管理 / fade管理 / retire管理 / EpochDomain参照。 | **一部妥当** — SlotStore/FadeState は物理的に分離済みだが、Coordinator が依然として全操作の orchestrator として機能。**P0-8 一括切替後も、責務の再監査が必要。** | Coordinator の責務を observe/updateFade のみに縮小。publish/retire/reclaim 操作は ISRRetireRouter に委譲。EpochDomain 直接参照を削除。**Phase-B 完了条件に「SnapshotCoordinator 責務監査」を追加**: 実装完了後に current/target slot管理・fade管理・retire管理が正しく分離されたか確認。 |
| P1-8 | **SnapshotRetireFacade 肥大化リスク**: 経験上、Facade に canRetire/canDelete/canPublish が追加され第二の Coordinator 化する危険。 | **妥当（v11.0新規）** — Facade の責務境界を明示的に定義しないと半年後に肥大化。 | **SnapshotRetireFacade 禁止事項を明文化**: epoch判定/fade判定/publish判定/ownership判定 を禁止。許可: routing/logging/metrics/validation のみ。 |
| P1-9 | **Fade 3層間の状態ドリフト**: 独立書き込み復活により、UI(20ms)とDSP(10ms)で fade 値が乖離。クリックノイズではなく持続的な状態ずれ。 | **妥当（v12.0修正）** — v11.0 では FadeAccumulator（analysis layer）を authority としたが、実時間を持つ DSP（CrossfadeRuntime）の方が権威として適切。 | **FadeClockDomain + FadeGeneration 導入（v12.0修正）**: 全 fade 層に共通の generation カウンタ。**authority を CrossfadeRuntime（DSP layer）に変更**。FadeAccumulator は analysis view として generation を参照のみ。generation mismatch 時 telemetry + 自動再同期。 |
| P1-10 | **到達不能コード監査 + regression count（v18.3新規、v18.4拡張）: Router移行完了後に旧経路の死蔵ラッパーが残るリスク**。移行進捗指標が「direct call = 0」「owner count = 0」だけで終わると、`OldRetireFacade` のような参照ゼロクラスが残存し、保守者が「使われているのか？」と混乱する。**v18.4: 完了後も新たな direct call が追加される regression リスク。過去30日の追加回数を追跡しないと、ゼロ状態が静かに崩れる。** | **妥当（v18.3新規）** — 「0件になった」だけでは不十分。到達不能コードの存在が将来の誤ったリファクタリングや再実装の原因になる。**v18.4: ゼロ維持のための regression tracking が不足。** | **v18.3: 到達不能コード監査を完了条件に追加**。Router 移行完了後に、全 EpochDomain ラッパー/ブリッジ/ファサードクラスの参照カウントを取得。参照ゼロのクラスは削除候補として `DEAD_CLASSES.md` に一覧化。CI gate: 移行完了後は「新規ファイル作成時に EpochDomain 依存がないこと」を確認。Phase-B 完了条件に到達不能コード監査を追加。**v18.4: regression count 追加**。完了後の進捗指標に「過去30日間の EpochDomain direct call 追加回数」を追加。`regressionCountLast30Days` を CI ダッシュボードに表示。0 を超えた場合は即時アラート。CI gate: regressionCount > 0 でマージブロック（レビュー必須）。監視対象: enqueueRetire/reclaimRetired/advanceEpoch/currentEpoch/publish/enterReader/exitReader の新規追加。 |
| P1-11 | **advanceEpoch(20)呼び出し理由未分類（v18.7新規: 19→20に上方修正）: Phase-B開始前に20箇所のadvanceEpoch呼び出し理由を分類しないと、EBR基盤として必要なものと過剰なものの区別ができない。特にEQProcessor.Parameters.cppの10回が集中。** | **v18.7 6ツール再監査で確認** — 20回中、EQProcessor.Parameters.cpp だけで10回、Core.cppが5回、Coefficients.cppが1回。新規発見: AudioEngine.CtorDtor.cpp:115 の advanceEpoch が未計上だった。パラメータ設定のたびに全区間epoch更新を行っている。 | **Phase-B開始前に advanceEpoch(20)呼び出し理由一覧作成**: カテゴリ分類(parameter update/snapshot publish/cache rebuild/cleanup shutdown/coordinator連動)。不要な advanceEpoch は削除。EQProcessor.Parameters.cpp の10回は各パラメータ設定メソッド末尾で呼ばれており、バッチ化または1回に集約可能か検討。 |
| P1-12 | **RCUReaderが新たなEpochDomain代理になるリスク（v18.6新規）: Router移行後もRCUReader.domain()がEpochDomainを公開し続けると、EpochDomainの型名が変わっただけで実質的な依存が残る。AudioEngine/EQProcessor/ConvolverProcessorがRCUReaderを保持。** | **v18.6 6ツール監査で確認** — RCUReader.h(7参照)、AudioEngine/EQProcessor/ConvolverProcessorがRCUReaderを保持。RCUReader.domain()はEpochDomain&を返す。 | **Phase-B完了条件にRCUReader代理防止を追加**: 「RCUReaderがEpochDomain公開代理になっていないこと」。RCUReader.domain()がEpochDomain&を返さないこと。README等で「RCUReaderはRouter内部実装の一部であり、外部にEpochDomain型を露出しない」ことを文書化。 |
| P1-13 | **DeletionQueue/SnapshotRetireManagerがPhase-B監査対象から漏れている（v18.6新規、v18.10強化、v18.12最終運命明記）: 計画の主文脈はSnapshotCoordinator/AudioEngine/EQProcessor/ConvolverProcessor中心で、DeletionQueue系が軽視されている。** 参照数が少ないため後回しになりやすいが、小規模コードほど見落とされやすい。**v18.12: DeletionQueue系の出口が曖昧。Router統合なのか存続なのか不明。** | **v18.6 6ツール監査で確認** — DeletionQueue.cpp:23 (`reclaim(const EpochDomain&)`)、DeletionQueue.h:19 (`void reclaim(const EpochDomain& core)`)、SnapshotRetireManager.h:43 (`void reclaim(const EpochDomain& domain)`)。**v18.12: 各コンポーネントの最終運命(Delete/Keep/Wrap/Merge)を明記しないと移行終盤で迷う。** | **Phase-B対象一覧へ明示的に昇格し、各コンポーネントの最終運命を明記**: (1) **DeletionQueue.h**: Keep (EpochDomain引数を epoch-free API に置換。Queue機能自体は継続)。(2) **DeletionQueue.cpp**: Wrap (reclaim()内部を Router 経由に変更。外部APIは維持)。(3) **SnapshotRetireManager.h**: Merge (SnapshotRetirePolicy に吸収。個別ファイルは削除)。これらを `EpochDomain_REFERENCE_STATUS.md` に「DeletionQueue系」カテゴリとして追跡。CI gate: DeletionQueue系ファイルのEpochDomain参照数が0であることをPhase-B完了条件に追加。 |
| P1-14 | **advanceEpoch の用途分類だけでは足りない: 計画は20箇所の分類を追加したが、call frequency と runtime frequency が不足。** 特にEQProcessor.Parameters.cppの10回の問題は「呼び出し箇所数」ではなく「UI操作1回あたり何回advanceEpochが発火するか」が本質。**v18.9新規、v18.11強化: 発火頻度。v18.12強化: static count から runtime telemetry へ。** | **v18.7 6ツール監査で確認** — 20回中EQProcessor.Parameters.cppだけで10回(50%)。**v18.12: 静的な呼び出し箇所数より、実行時の advanceEpoch/sec が運用上有益。** | **用途分類 + call frequency + runtime telemetry をPhase-B前に実施**: (1) EQProcessor.Parameters.cpp の10回が各パラメータ設定メソッド末尾で呼ばれていることを確認。(2) **意味分類**: Parameter変更 / Snapshot更新 / Runtime切替 / Cleanup。(3) **発火頻度**: 「1操作あたり何回 advanceEpoch」を計測。正常値は1操作あたり1回が理想。(4) **runtime telemetry追加**: `advanceEpoch/sec` をテレメトリとして計測し、移行前後の発火回数を比較。期待値: 移行前 21回/操作 → 移行後 1回/操作。(5) バッチ化可能か評価 → advanceEpochOnce() または遅延集約。(6) 全ての情報を `EpochDomain_REFERENCE_STATUS.md` に追記。**CI gate**: Phase-B完了後に advanceEpoch 呼び出し回数上限(最大5回) + 発火頻度上限(1回/操作) + runtime telemetry閾値(正常動作時の2倍以下)を設定。 |
| P1-15 | **ConvolverProcessor の扱いが揺れている: 直接依存除去対象でありながら snapshotRcuEpoch() 経由の利用は許容される。** 計画書内で「許容されるepoch利用」と「禁止されるepoch利用」の境界が不明確。レビューごとに解釈が変わるリスク。**v18.9新規。** | **v18.8付録BチェーンAで確認** — ConvolverProcessor は getRcuProvider()->snapshotRcuEpoch() 経由で5回のEpochDomain読取を行う。これらは current() 読取専用であり EpochDomain の書き込みは行わない。一方で ConvolverProcessor.h:1149 には m_epochDomain メンバが残存。 | **ConvolverProcessor の epoch 利用境界を明文化**: (a) **許容**: getRcuProvider()→snapshotRcuEpoch() 経由の epoch 読取(5回) — 読取専用、Router 経由に移行後も同様の epoch 読取APIを提供。(b) **禁止**: m_epochDomain メンバ直接参照、RCUReader::domain()→EpochDomain& 経由の全メソッド呼び出し。(c) **移行**: RCUReader を Router 内部実装に変更(P0-14)後、ConvolverProcessor の epoch 取得は Router::snapshotEpoch() または同等の限定API経由に統一。完了条件として「ConvolverProcessor.h に EpochDomain 型のメンバ/引数/戻り値が存在しないこと」を追加。 |
| P1-16 | **RetireRetryPolicy が Router 設計に未統合: ConvoPeq実装には enqueueRetire→reclaimRetired→enqueueRetire の retry パターンが存在する。** 計画では Router 化で吸収する想定だが、呼び出し側が retry を持つ構造を放置すると Router 移行後も retry ロジックが残る。**v18.9新規。** | **v18.5 6ツール監査で確認** — EQProcessor.Core.cpp:kMaxRetry=4、AudioEngine.h:3179-3193で reclaim+retry。P0-5で retry ループ削除は計画済みだが、「呼び出し側が retry を持たない」構造的保証がない。 | **Router 設計に RetireRetryPolicy を明示的に統合**: (1) Router が retry 判断の唯一の権威となる — 呼び出し側は enqueue 成功/失敗を単に報告し、再試行判断は Router の RetireRetryPolicy が行う。(2) RetireRetryPolicy の責務: maxRetryCount=0（呼び出し側 retry 禁止）、enqueue failure 時は Router 内部で deferred retry queue に委譲、呼び出し側へのリトライ要求は行わない。(3) CI gate: `enqueueRetire` 呼び出し直後の `reclaimRetired` / `advanceEpoch` パターンを clang-tidy ルールで禁止。retry ループ(for/while)内の enqueueRetire を検出。 |
| P1-17 | **124参照の状態分類が未完了: P2-11で EpochDomain_REFERENCE_STATUS.md を作成する計画だが、status が「削除/置換/RCU/Dead」の4種のみで「許容/保留」がない。** 対象が増え続けるリスクを防ぐには、Phase-B開始前に全124参照を確定分類する必要がある。**v18.9新規。v18.11強化: statusに理由列を必須化。** 半年後に「なぜ許容だったのか」が分からなくなるリスク。 | **v18.5 6ツール監査で確認** — 124参照・27ファイルの全容は把握済みだが、各参照の status が未確定。「コメントのみ」「デッドコード」「意図的残存」の区別がない。**v18.11: statusだけでは運用保守で理由が分からなくなる。** | **Phase-B開始前に全124参照を5段階で確定分類 + 理由列を必須化**: status削除/置換/許容/保留/Dead の各エントリに reason 列を追加。(1) **削除**: 直接 enqueueRetire/reclaimRetired — Phase-A対象。reason: "retire経路Router移行"。(2) **置換**: advanceEpoch/publish/current → Router API。reason: "Router::publishEpoch()へ委譲"等明示。(3) **許容**: enterReader/exitReader/activeReaderCount — EBR基盤。reason: "EBR基盤lock-free atomic読取"。(4) **保留**: ConvolverProcessor m_epochDomain, RCUReader等。reason: "P0-14/P1-15完了後再評価"。(5) **Dead**: RefCountedDeferred.h, EpochCore.h, コメントのみ。reason: "未使用テンプレート"。**完了条件**: `EpochDomain_REFERENCE_STATUS.md` に全124参照の status + reason を記載。CI gate で status 未定義または reason 未記入の新規参照をブロック。Phase-B開始時点で「保留」以外の status はすべて確定していること。 |
| P1-18 | **EpochDomainReaderGuard 直接生成が再発ポイント: AudioEngine.BlockDouble.cpp, AudioEngine.AudioBlock.cpp, ObservedRuntime.h に EpochDomainReaderGuard(m_epochDomain, ...) が残存。** 大規模箇所は監査されるが、これら小規模箇所は最後まで残りやすい。**v18.10新規。v18.11強化: AST CIルール化。** 完了条件だけでなく clang-tidy ルールとして常時検出が必要。 | **v18.5/18.8 6ツール監査で確認** — AudioEngine.Processing.BlockDouble.cpp, AudioEngine.Processing.AudioBlock.cpp, ObservedRuntime.h の3箇所。参照数こそ少ないが、include監査では捕捉しにくい。 | **Phase-B完了条件に「EpochDomainReaderGuard直接生成 = 0」を追加 + AST CIルール化**: (1) 全3箇所の `EpochDomainReaderGuard(m_epochDomain, ...)` を RCUReader 限定API (enterReader/exitReader) に置換。(2) 置換後、`EpochDomainReaderGuard(` の直接生成を **clang-tidy AST visitor ルールとして恒久禁止**。clang-tidy 非対応環境では grep フォールバック。(3) `#include "core/EpochDomain.h"` の削除確認ではなく、コンストラクト呼び出しの有無をASTベースで監査。**CI gate**: `EpochDomainReaderGuard(` の使用を clang-tidy ルールで禁止。RCUReader の限定API以外からの EpochDomainReaderGuard 生成を検出。新規追加時にコンパイルエラー。 |
| P1-19 | **Public Exposure 指標が getter名依存: 現在は AudioEngine::epochDomain() が主要対象だが、将来 getReaderDomain() / getEpochContext() のような別名getterが追加されるリスク。** getter名ベースの監査では回避可能。返り値型ベースのAST検査が必要。問題は `domain()` という名前ではなく、RCUReader が EpochDomain を返すAPI自体が残ること。**v18.10新規。v18.11強化: RCUReader返却型ベース監査の明確化。** 関数名変更では不十分で、返却型の恒久監査が必要。 | **v18.5 6ツール監査で確認** — 現状の epochDomain() は発見済みだが、監査方法が「epochDomain という文字列」に依存。型名 EpochDomain が返り値に出現する全関数を監査すべき。**v18.11: RCUReader が reader.getEpoch() / reader.getContext() / reader.getProvider() のような別名APIで EpochDomain を返さないことを保証する必要がある。** | **Public Exposure 指標を「EpochDomain型露出」に昇格し、RCUReader 返却型ベース監査を明確化**: (1) getter名ベース → **返り値型ベースのAST検査**に変更。(2) clang-tidy ルール: `EpochDomain&` / `EpochDomain*` を返すpublicメソッドを禁止。(3) **RCUReader の全public API が EpochDomain 型を返さないことを個別確認**: 関数名 (`domain()`, `getEpoch()`, `getContext()`, `getProvider()` など) に関わらず、返却型が `EpochDomain&` または `EpochDomain*` であれば禁止。(4) 監査対象: publicメソッドの戻り値型 + publicメンバ変数の型 + friend宣言経由の型露出。(5) `EpochDomain_REFERENCE_STATUS.md` に「型露出」カテゴリを追加。**完了条件**: 全public APIで EpochDomain 型が戻り値・引数・publicメンバとして出現しないこと。RCUReader.h の全publicメソッドの返却型が EpochDomain 非依存であることをCI gateで確認。 |
| P1-20 | **RefCountedDeferred.h の扱いが弱い: 計画ではデッドコード扱いだが、削除ではなく static_assert(false) の追加に留まっている。** 実運用では「デッドコード→将来再利用」が頻発する。削除できない場合は少なくとも `[[deprecated]]` を付与すべき。**v18.10新規。** | **v18.5 6ツール監査で確認** — RefCountedDeferred.h:20 の `epochDomain.enqueueRetire()` は真のデッドコード。P1-2で static_assert(false) 追加済みだが、ファイル自体が残存している。 | **RefCountedDeferred.h を deprecated 化または削除**: (1) ファイル先頭に `[[deprecated("Do not use. Use Router-based retire instead.")]]` を追加。(2) 理想的にはファイル自体を削除。ただし ISRClosure/ISRRetireRuntimeEx とのテンプレート依存がある場合は削除不可。(3) 削除不可の場合も、`#pragma message` または `static_assert(sizeof(T) == 0, "Use Router-based retire")` でコンパイル時に使用を警告。**完了条件**: RefCountedDeferred.h が EpochDomain に依存しないこと、またはファイルが削除されていること。CI gate: RefCountedDeferred.h の `#include "core/EpochDomain.h"` が存在しないことを確認。 |
| P1-21 | **DeletionQueue系の最終運命が曖昧: P1-13で対象化したが、Router統合なのか存続なのか不明。** 移行終盤で迷いが生じる。**v18.12新規。** | **v18.6 6ツール監査で確認済み** — 各コンポーネントの役割は明確だが、移行後の姿が未定義。 | **各コンポーネントの最終運命を Delete/Keep/Wrap/Merge で明記**: (1) **DeletionQueue.h**: Keep — EpochDomain引数をepoch-free APIに置換した上で存続。(2) **DeletionQueue.cpp**: Wrap — reclaim()内部をRouter経由に変更。外部APIは維持。(3) **SnapshotRetireManager.h**: Merge — SnapshotRetirePolicyに吸収。個別ファイル削除。(4) **EpochCore.h**: Delete — 空のヘッダ。EpochDomain.h includeのみ。(5) **RefCountedDeferred.h**: Delete — ファイル削除または static_assert で完全使用禁止。これらを `EpochDomain_REFERENCE_STATUS.md` の fate 列に記載。Phase-B完了時点で fate 未定義のコンポーネントをブロック。 |
| P1-22 | **コメント内EpochDomainの監査が不足: v18.10で8+のコメント内参照を確認したが、コードよりコメントが古くなる。** 特に`// TODO remove later` 系は危険。**v18.12新規。** | **v18.10 6ツール監査で確認** — SnapshotRetireManager.h(3), AudioEngine.h(1), Init.cpp(1), Globals.cpp(1), ConvolverProcessor.Lifecycle.cpp(1), EpochCore.h(1) の8+参照。 | **Phase-B完了条件にコメント監査を追加**: (1) `EpochDomain`, `SnapshotCoordinator`, `ISRRuntimePublicationCoordinator` を含むコメントを一括grep。(2) 特に `// TODO`, `// FIXME`, `// HACK`, `// remove later`, `// deprecated` 系コメントは個別レビュー。(3) 古いコメントは削除または最新化。(4) コメント内のEpochDomain参照が「存在してはいけない」のではなく、「実態と乖離していないこと」を確認。CI gate: 新規コメント内EpochDomain参照の追加を監視。 |
| P1-23 | **AudioEngine.Threading.cpp 3PR分割後の依存方向監査がない: Publication/Reader/Retire の3PR間で逆流が起きるリスク。** 分割後に Retire.cpp が #include Publication.h するような循環依存。**v18.12新規。** | **設計上のリスク** — 3PR分割は正しいが、分割後の依存方向ルールがない。 | **3PR完了後に依存方向監査を追加**: 理想の依存方向は `Reader ← Publication ← Retire` の一方向。依存方向ルール: (a) Reader PR → Publication PR に依存不可。(b) Reader PR → Retire PR に依存不可。(c) Publication PR → Retire PR に依存不可。(d) 逆方向(Retire→Publication→Reader)のみ許可。CI gate: 3PR間の#include関係をgrepで監査し、逆方向のincludeをブロック。 |

### P2 改善項目（第4回〜第10回監査）

| # | 問題 | ソース検証結果 | 対応 |
| --- | ------ | --------------- | ------ |
| P2-1 | **RetireStateMachine Logical=decision owner が不明瞭** | **妥当** — 2層のうちどちらが遷移判断するか未定義。 | 「Logical state = 唯一の decision owner。Physical state は observer-only projection」を明文化。Physical 側に遷移ロジック禁止。 |
| P2-2 | **DSPHandleRuntime の役割明確化（v10.0修正: Stateful Handle Authority 復活）**: Request 変換層への格下げを撤回。Handle lifecycle の弱体化・quarantine 非決定的化・ISR closure graph との乖離リスク。 | **v10.0修正** — stateful handle authority として維持が正しい。Router は参照のみ行い、Handle lifecycle の権威は DSPHandleRuntime が保持。 | 設計文書に「DSPHandleRuntime = Stateful Handle Authority（slot 状態機械 + lifecycle 権威）」と明記。Router は参照のみ。 |
| P2-3 | **Fade 3層 truth source + 単一書き込み制限（v9.0強化）** | **追記事項** — CrossfadeRuntime（sample-rate sync層）/ SnapshotFadeState（state transition coarse層）/ FadeAccumulator（UI timing analysis層）。各層は自身の時間スケールの書き込みのみ許可。 | PR-B-0 に truth source 階層図 + 書き込み権限マトリクスを追加。各層のタイムスケールと同期プロトコルを明記。 |
| P2-4 | **Emergency Drain tier-2 設計**（v8.0） | **追記事項** — Worker Thread 単一障害点対策。shutdown 専用の lock-free queue drain 実装。 | `AudioEngine::emergencyDrainRetireQueues()` として実装。prepareToPlay後〜releaseResources前のみ有効。best-effort delete、failure許容。 |
| P2-5 | **enqueue failure: drop + telemetry + deferred retry queue（bounded）**（v10.0修正） | **追記事項** — drop only は recovery path 完全消失。transient overload 時に機能低下。 | RT: 即座に drop + telemetry。Non-RT: bounded deferred retry queue に enqueue（最大再試行回数3回、指数バックオフ）。Worker が定期リトライ。 |
| P2-6 | **Retire Commit Barrier 2-tier 設計**（v8.0） | **追記事項** — soft barrier（タイムアウト許容/優先制御）+ hard barrier（全条件成立必須/安全停止）。 | `RetireCommitBarrier::tryPass()`（soft, 即座に可否返却）+ `RetireCommitBarrier::waitPass()`（hard, 条件成立までブロック）。 |
| P2-7 | **ownerThreadId 排除**（v9.0新規） | **妥当** — `ObservedRuntime.h` に残存。`EpochDomainReaderGuard` と併存により Debug/Release 挙動差リスク。 | `ownerThreadId` 削除。スレッド識別は atomic ベースに統一。thread id は debug assert 限定。 |
| P2-8 | **DeferredRetryQueue 寿命・上限・優先順位定義・飽和時保護・starvation 対策 + WFQ aging + aging 昇格統計**（v15.0: maxWaitTime 追加, v18.4: agingPromotionCount/maxAgingLevelReached 追加） | **追記事項** — WFQ (DSP 8:Snapshot 4:Cache 1) は完全優先より良いが、DSP retire が永続的に高負荷の場合、Cache retire が数十分単位で遅延する可能性がある。**v18.4: aging が何回発生したか・最高どのレベルまで達したかの統計がないと、運用中の aging 効果が評価できない。** | **v15.0: WFQ + aging (maxWaitTime) 追加**: `maxWaitTime=30s` 超過で Cache retire を強制昇格（weight 一時的に最大）。WFQ 重み: DSP 8 : Snapshot 4 : Cache 1（通常時）。`maxRetryCount=3`, `maxRetryAgeMs=5000`, `maxQueueDepth=4096` 維持。保護レベル: Cache drop(許可) → Snapshot drop(最終手段+telemetry) → DSP(絶対不drop)。**v18.4: aging 昇格統計追加**: `agingPromotionCount`（昇格発生総回数）/ `maxAgingLevelReached`（最高到達昇格レベル）を DeferredRetryQueue の telemetry に追加。CI ダッシュボードで可視化。長期運用の aging 効果評価に使用。 |
| P2-9 | **ShutdownPhase 固定化と EmergencyDrain must-drain + DrainResult + 残留種別ヒストグラム + 経過時間 + オブジェクト年齢情報**（v18.3: oldestAge/newestAge 追加, v18.4: oldestRetireAge/averageRetireAge 追加） | **妥当（v18.3修正）** — 残留件数だけでなく「どれだけ長く残ったか」の情報が原因特定に必須。**v18.4: 残留オブジェクト「1個あたりの年齢」情報が不足。全体の oldest/newest だけでは偏りが分からない。** | **v18.3: RemainingObjectTypeHistogram に oldestAge/newestAge 追加**。struct RemainingObjectTypeHistogram { size_t dspRuntimeCount; size_t snapshotCount; size_t cacheCount; size_t eqStateCount; size_t otherCount; double oldestAgeSec; double newestAgeSec; }; DrainResult に内包。生成タイミング: (1) EmergencyDrain 完了時、(2) forceFinalPurge 連動時。**DrainResult 維持**: struct DrainResult { size_t drainedCount; size_t remainingCount; size_t remainingBytes; double timeoutMs; const char* timeoutReason; RemainingObjectTypeHistogram remainingTypes; }。**v18.4: DrainResult にオブジェクト年齢情報追加**: `oldestRetireAge`（最古残留オブジェクトの経過時間秒） / `averageRetireAge`（全残留オブジェクトの平均経過時間秒）を RemainingObjectTypeHistogram に追加。struct 拡張: { ..., double oldestRetireAgeSec; double averageRetireAgeSec; }。用途: 「平均年齢が高いのに件数が少ない → 孤立オブジェクト問題」「最古年齢が極端に高い → 特定オブジェクトの retire 漏れ」。**ビルド種別分岐**: **Debug**: DrainResult を Assert+Fatal（異常検知）。**Release**: 診断ダンプ + telemetry のみ + 残留は leak quarantine に隔離（host crash 防止）。ポリシー: `maxDrainTime=500ms` OR `maxDrainItems=100000` OR `maxDrainBytes=max(residentMemory*0.1, 1GB)` 先到達制。 |
| P2-10 | **ConvolverProcessor の EpochDomain 依存（6ツール監査で発掘: 計画未記載）**: ConvolverProcessor にも EpochDomain 直接依存が4箇所存在するが、計画書の P0/P1/P2 テーブルに一切記載なし。Retire 操作ではないが、P0-8 完了条件「EpochDomain 型を一切知らない状態」の達成には影響。 | **6ツール監査で確認** — `ConvolverProcessor.LoadPipeline.cpp:678 advanceEpoch()`, `ConvolverProcessor.StateAndUI.cpp:1017 advanceEpoch()`, `ConvolverProcessor.h:1149 m_epochDomain メンバ`, `ConvolverProcessor.h:1151 RCUReader runtimeReader_{m_epochDomain}`。これらは retire 操作ではなく RCU epoch 管理だが、「EpochDomain 型を知らない状態」への移行では障害になる。 | **v18.0新規: ConvolverProcessor の EpochDomain 依存を P0-8/P1-2 の完了条件に含める。** 具体的対応方針は Phase-A2（AudioEngine/EQProcessor 完了後）のレビューで決定。ただし enterReader/exitReader/RCUReader 経由の EpochDomain 参照は EBR 基盤として Router 移行後も残りうる点を許容し、完了条件から除外可能とする。 |
| P2-11 | **124参照の整理漏れ・分類漏れ防止（v18.6新規）: v18.5で124参照全てを列挙したが、「削除対象」と「意図的残存」を区別しないと、レビューごとに議論が再発する。各参照に status を付与した一元管理ファイルが必要。** | **v18.6 確認** — 現状の付録Aは参照一覧のみで status 情報がない。 | **EpochDomain_REFERENCE_STATUS.md 作成**: 全124参照について status(削除/Router置換/RCUとして残存/コメントのみ/Dead code)を付与。Phase-A/Phase-Bの完了進捗をこのファイルで追跡。CI gate で全EpochDomain参照数の変動を監視し、status 未定義の新規参照をブロック。 |

---

## ★ アーキテクチャ決定: ISRRetireRouter + 目的別 Policy Lane（最重要設計判断 v3.0）

### 決定

以下のアーキテクチャを採用する:

```text
                          +------------------------------------+
[AudioEngine]             | ISRRetireRouter (thin)              |  ★ 唯一の public retire API
[SnapshotCoordinator] ──→ | ・入口は1つだが内部で振り分け        |
[EQProcessor]             | ・RT/NonRT/DSP/Snapshot を識別して  |
[ISRRetireRuntimeEx]      |   適切な Policy Lane にルーティング  |
                          +----+-------------+--------------+---+
                               |             |              |
                    +----------v----+  +-----v------+  +---v----------+
                    | DSPRetire     |  | Snapshot   |  | Deferred     |
                    | Policy        |  | Retire     |  | Retire       |
                    | (immediate)   |  | Policy     |  | Policy       |
                    |               |  | (fade-aware)|  | (queue+BP)  |
                    +-------+------+  +-----+------+  +---+----------+
                            |               |              |
                    +-------v---------------v--------------v------+
                    | EpochDomain::enqueueRetire()                  |
                    | (internal API, deprecated wrapper維持)        |
                    +--------------------+-------------------------+
                    +--------------------v-------------------------+
                    | DeferredDeletionQueue (単一 MPMC)            |
                    +---------------------------------------------+
```text

### 設計根拠

- **ISRRetireRouter は「分類された統一」を実現する thin router**。単一障害点化を防ぐため、ルータ自体は条件分岐と委譲のみを行う。複雑なポリシー判断は各 Policy Lane が担当する。
- **SnapshotRetireManager は QUEUE ONLY で維持する**（実コード確認済み: 現在の実装は純粋な `retire(GlobalSnapshot*)`/`reclaim()` のみ。coordinator依存/fade状態参照/lifecycle判断は一切存在しない。また `GlobalSnapshot*` 専用のキューであるため、他種オブジェクト（DSPCore等）は対象外）。
- **EpochDomain::enqueueRetire() は internal API に留め、完全 private 化は行わない**。`[[deprecated]]` ラッパーとして維持し、ISRRetireRouter が preferred API として機能する。ISRClosure/ISRRetireRuntimeEx 等の基盤コンポーネントの既存依存を考慮。
- **既存の RT safety 機構（`ASSERT_NON_RT_THREAD`, `RTCapabilityFirewall`, `RTAllocatorFirewall`）を活用**し、新規作成を最小化する。

### Retire State Machine（v5.0 2層モデル + Logical=decision owner 原則）

実運用では「論理状態」と「観測状態」に時間差が生じる（Epoch進行/Reader drain/Queue latency/OS scheduling）。そのため単一状態遷移図ではなく**2層モデル**を導入する:

```text
Logical State (authoritative, 決定即時反映):
  Active          : 稼働中（Audio Thread から参照可能）
  PendingRetire   : 退役予約済み（ISRRetireRouter に enqueue）

Physical State (observed, 実時間遅延あり):
  Draining        : Grace period 中（Reader 退出待ち / Epoch 進行待ち）
  ReclaimReady    : Epoch安全確認済み（削除可能）
  Freed           : 解放完了（デストラクタ実行済み）
```text

**Logical→Physical のマッピング**:

| Logical | Physical 最小 | Physical 最大 |
| --------- | -------------- | --------------- |
| Active | — | Draining |
| PendingRetire | Draining | ReclaimReady |
| (→ Reclaimed) | Freed | Freed |

**遷移条件**:

```text
PendingRetire → Draining:     ISRRetireRouter への enqueue 成功
Draining → ReclaimReady:      EpochDomain::isOlder(entry.epoch, minReaderEpoch) == true
ReclaimReady → Freed:         DeferredDeletionQueue::reclaim() による deleter 実行
```cpp

**v5.0 追加: Logical=decision owner 原則**:

> **Logical state のみが遷移判断の権威を持つ。Physical state は observer-only projection であり、Physical 側に遷移ロジックを絶対に持たせてはならない。**

この原則により以下を保証する:

- `PendingRetire` (Logical) が真であれば、`Draining` (Physical) が未観測でも retire 決定は確定
- Physical state の取得失敗は観測遅延であり、retire の再判断にはならない
- Retire の取消（cancel）は Logical state の変更のみで行い、Physical 側の整合は自然に追随する

この2層モデルにより「状態遷移＝リアルタイム」の誤った前提を排除し、観測遅延を陽に設計に組み込む。既存の `RetireLifecycleState` enum（`ISRRetireRuntimeEx.h`）と整合させる。

### Router パフォーマンス設計（v4.0 追加）

ISRRetireRouter の集中によるホットスポット化を防ぐため、以下の設計を追加する:

**分岐のテーブル化**: Policy 選択をswitch文ではなく関数ポインタテーブルで実装し、branch misprediction + cache miss の集中を回避する。

```cpp
// 例: 関数ポインタテーブルによる Policy 選択
using RetireHandler = RetireEnqueueResult (*)(void*, void(*)(void*), uint64_t);
static constexpr RetireHandler kPolicyTable[3] = {
    &DSPRetirePolicy::enqueue,        // [0] immediate
    &SnapshotRetirePolicy::enqueue,   // [1] fade-aware
    &DeferredRetirePolicy::enqueue    // [2] queue+backpressure
};
```

**fast-path inline decision**: 最も頻度の高い retire 種別（DSPCore 即時 retire）は `router.enqueueDSPFast(...)` として Router API の内部最適化として提供する。call-site から直接 Policy Lane へ委譲することは認めず、必ず Router API を経由する。**Policy 直参照禁止**: どの call-site も Policy Lane への直接参照を持たない。Router は「分類が不明な場合のデフォルトルート」として機能する。

### Retire Observability（v4.0 拡張）

各 Policy Lane は以下のテレメトリを公開する:

| メトリクス | 取得元 | 用途 |
| ----------- | -------- | ------ |
| retire latency | Lane 別 timer | RT 負荷監視 |
| worst-case retire latency | Lane 別 max 保持器 | 最大遅延監視 |
| queue depth | DeferredDeletionQueue::sizeApprox() | メモリ圧力検出 |
| queue stall duration | enqueue待機時間 | 輻輳検出 |
| reclaim delay | Epoch ベース | GC 遅延監視 |
| epoch drift histogram | Global epoch と Reader epoch の差 | Epoch 進行監視 |
| retry count | ISRRetireRouter | フォールバック頻度 |
| saturation count | 既存 retirePressureLevel_ | 過負荷検出 |

### Policy Lane 優先順位モデル（v4.0 追加）

複数 Lane が同時に retire を要求した場合の優先順位を明示する:

**優先順位**: `DSP > Snapshot > Deferred`

**同順位 tie-break**: Epoch 順（古い epoch が先）

**根拠**: DSP retire の遅延は Audio glitch に直結。Snapshot retire は fade 依存があるため即時より優先。Deferred retire（EQ cache 等）は最も余裕がある。

### EpochDomain::enqueueRetire() 段階的廃止計画（v4.0 追加）

| フェーズ | 処置 | タイミング |
| --------- | ------ | ----------- |
| 現状 | `[[deprecated]]` warning | PR-1 着手前 |
| Phase-A 完了 | warning → `[[deprecated("error")]]` 相当のコンパイル時強制 | PR-1 全タスク完了後 |
| Phase-B 完了 | 新規コードからの参照を error に昇格 | PR-4 完了後 |
| 最終 | 完全 private（ただし基盤コンポーネント用 friend 維持） | Phase-D |

この段階的アプローチにより、既存の ISRClosure/ISRRetireRuntimeEx 等の基盤コンポーネント依存を段階的に解消しながら、最終的に EpochDomain::enqueueRetire() の外部参照をゼロにする。

---

## PR-1: ISRRetireGateway 導入 + SnapshotRetireManager 統合（Phase-A, 最重要）

### 問題 {#pr-1-問題}

`src/core/SnapshotRetireManager.h` に retire 管理クラスは抽出済みだが、**全ソースコードで参照ゼロ**。`SnapshotCoordinator` は未だに `EpochDomain::enqueueRetire()` (deprecated) を直接使用している。

### 現状の retire 経路（3系統）

```text
経路1: SnapshotCoordinator → EpochDomain::enqueueRetire() [deprecated]
  └─ startFade() / completeFade() / resetFadeStateAndRetireTarget()
  └─ switchImmediate() / ~SnapshotCoordinator()

経路2: ISRRuntimePublicationCoordinator::enqueueRetire() → EpochDomain::enqueueRetire() [deprecated]
  └─ AudioEngine::enqueueDeferredDeleteNonRtWithResult()
  └─ (authorized caller とされているが、実際はdeprecated経由)

経路3: EQProcessor::enqueueDeferredDeleteWithFallback() → EpochDomain::enqueueRetire() [deprecated]
  └─ retry 4回 + reclaimRetired() 間にはさみ
```text

### ターゲット

```text
単一経路: ISRRetireGateway::enqueueRetire() [唯一のpublic API]
  └─ 全モジュールが ISRRetireGateway を呼ぶ
  └─ SnapshotRetireManager は内部委譲先（QUEUE ONLY、coordinator非依存）
  └─ ALL non-snapshot retire も ISRRetireGateway 経由
```

### タスク {#pr-1-タスク}

| # | タスク | ファイル | リスク | 備考 |
| --- | -------- | ---------- | -------- | ------ |
| 1-0 | `ISRRetireRouter` クラスを新規作成（thin router, 複雑ロジック禁止） | `src/audioengine/ISRRetireRouter.h`（新規） | 低 | 入口ルーティングのみ。Policy 判断は Lane へ委譲 |
| 1-0a | `DSPRetirePolicy` クラス新設（即時 retire 用） | `src/audioengine/DSPRetirePolicy.h`（新規） | 低 | 現在の DSPCore 即時 retire パスを明確化 |
| 1-0b | `SnapshotRetirePolicy` クラス新設（fade-aware retire 用） | `src/audioengine/SnapshotRetirePolicy.h`（新規） | 低 | `SnapshotRetireManager` を内部保持（QUEUE ONLY維持） |
| 1-0c | `DeferredRetirePolicy` クラス新設（queue + backpressure 用） | `src/audioengine/DeferredRetirePolicy.h`（新規） | 低 | EQProcessor 等の遅延 retire を統括 |
| 1-0d | `RetireStateMachine` enum を新設（Created→Active→PendingRetire→Retiring→Reclaimed） | `src/audioengine/ISRRetireRouter.h` | 低 | 既存 `RetireLifecycleState`（ISRRetireRuntimeEx.h）と整合 |
| 1-0e | **`DSPHandleRuntime` の retire/reclaim/quarantine を Router 経由に統合**。slot 状態遷移は Router からの callback で更新。DSPHandleRuntime は slot metadata manager として明確化（メモリ解放は EpochDomain 経路）。 | `src/audioengine/ISRDSPHandle.h` + `src/audioengine/ISRDSPHandle.cpp` | **重要** | P0-2 対応。DSPHandleRuntime::retire/reclaim/quarantine の内部実装を Router 委譲に変更。slot 状態のみ管理。 |
| 1-1 | `SnapshotRetireManager` を `SnapshotRetirePolicy` 内部で保持（QUEUE ONLY 維持） | `src/audioengine/SnapshotRetirePolicy.h` | 低 | SnapshotRetireManager に coordinator 参照を追加しない |
| 1-2 | `ISRRetireRouter::enqueueRetire()` に `ASSERT_NON_RT_THREAD()` + `RTCapabilityFirewall` チェックを追加 | `src/audioengine/ISRRetireRouter.h` | 低 | 既存の安全機構を活用 |
| 1-3 | `SnapshotCoordinator` の `m_epochDomain->enqueueRetire()` 5箇所を `ISRRetireRouter` → `SnapshotRetirePolicy` 経由に置換 | `src/core/SnapshotCoordinator.cpp` + `src/core/SnapshotCoordinator.h` | **重要** | switchImmediate / startFade / completeFade / resetFadeStateAndRetireTarget / ~SnapshotCoordinator |
| 1-4 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` の deprecated 呼び出しを `ISRRetireRouter` → `DSPRetirePolicy` 経由に変更 | `src/audioengine/AudioEngine.h` | **重要** | フォールバックパターンの維持確認必須 |
| 1-5 | `EQProcessor::enqueueDeferredDeleteWithFallback()` の deprecated 呼び出しを `ISRRetireRouter` → `DeferredRetirePolicy` 経由に変更 | `src/eqprocessor/EQProcessor.Core.cpp` | **重要** | retry 4回パターンを ISRRetireRouter 側で共通化 |
| 1-6 | `EpochDomain::enqueueRetire()` を internal API として維持（`[[deprecated]]` ラッパー）。完全 private 化しない | `src/core/EpochDomain.h` | 低 | ISRRetireRouter が preferred API。基盤コンポーネント（ISRClosure等）の既存依存を温存 |
| 1-7 | `RetireBackpressurePolicy` を設計・実装（max queue depth / drop strategy / degrade mode） | `src/audioengine/DeferredRetirePolicy.h` | 中 | 既存の `retireHighWatermark_`/`retireLowWatermark_` と統合 |
| 1-8 | Retire Observability テレメトリを追加（lane別 latency / queue depth / reclaim delay） | `src/audioengine/ISRRetireRouter.h` + `src/audioengine/AudioEngine.h` | 低 | 既存 `RuntimeBackpressureTelemetry` を拡張 |
| 1-9 | **CI gate AST ベース昇格（v18.0）**: `EpochDomain::enqueueRetire` / `reclaimRetired` の新規呼び出し追加を clang-tidy AST visitor で検出。grep テキスト検索から昇格。clang-tidy 非対応環境では grep フォールバック。 | `.github/scripts/` + `.clang-tidy` | **中** | 回帰防止 + 保守性向上 |
| 1-10 | **`EQProcessor::enqueueDeferredDeleteWithFallback()` の retry ループ削除**（`kMaxRetry=4` → enqueue failure = drop）。`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` の reclaim+retry パス削除。 | `src/eqprocessor/EQProcessor.Core.cpp` + `src/audioengine/AudioEngine.h` | **重要** | P0-5 対応。retry を drop または quarantine に変更。CI gate で retry パターン検出。 |
| 1-11 | **`epochDomain.reclaimRetired()` → `epochDomain.requestReclaim()`（enqueue only）に置換**。全直接呼び出し箇所を変更。debug assert で直接 delete 検出。 | `src/core/EpochDomain.h` + 全呼び出し元 | **重要** | P0-6 対応。reclaim API を enqueue only に変更。 |
| 1-12 | **Retire Commit Barrier 導入**: epoch 安全確認 + snapshot fade 完了 + RetireRouter enqueue 成功の三拍子同期ゲートを `RetireCommitBarrier` クラスとして新設 | `src/audioengine/ISRRetireRouter.h`（新規クラス） | **高** | P0-7 対応。SnapshotCoordinator の retire 発行前に Barrier 通過を必須化。 |
| 1-13 | **SnapshotCoordinator の直接 enqueueRetire/reclaimRetired 全5箇所を ISRRetireRouter 経由に変更**: `~SnapshotCoordinator()`(3), `switchImmediate()`(1), `reclaim()`(1)。EpochDomain 直接参照を削除。 | `src/core/SnapshotCoordinator.h` + `src/core/SnapshotCoordinator.cpp` | **重要** | P0-8 対応。SnapshotCoordinator から retire/reclaim 責務を完全分離。 |
| 1-14 | **`SnapshotRetireManager` 削除方針撤回 → `SnapshotRetireFacade` として復活**: thin facade（routing + telemetry + ordering validation）。state owner ではない。実 delete は Router 経由。 | `src/core/SnapshotRetireFacade.h`（新規） | 低 | P1-4 v10.0修正。削除ではなく薄層 facade 化。 |
| 1-15 | **Retire 2-lane 設計**: Fast Lane（lock-free SPSC, RT critical: snapshot swap, immediate DSP retire）+ Slow Lane（Worker Queue, bulk cleanup: deferred delete, reclaim, EQ cache）。Router は stateless dispatcher として2-lane 振り分け。 | `src/audioengine/ISRRetireRouter.h` + `src/audioengine/ISRRetireFastLane.h`（新規）+ `src/audioengine/ISRRetireSlowLane.h`（新規） | **高** | P0-1 v10.0修正。単一経路から2-lane へ。 |
| 1-16 | **deferred retry queue（bounded）追加**: enqueue failure 時に RT は即座に drop + telemetry。Non-RT は bounded deferred retry queue（最大3回、指数バックオフ）。 | `src/audioengine/DeferredRetryQueue.h`（新規） | 中 | P0-5/P2-5 v10.0修正。 |
| 1-17 | **RetireId + RetireRegistry 導入**: 各 RetireRequest に一意の RetireId を付与。`RetireRegistry::alreadyRetired(id)` で二重 retire 検出。debug build で double retire を assert。 | `src/audioengine/ISRRetireRouter.h` + `src/audioengine/RetireRegistry.h`（新規） | 低 | P0-10 v11.0新規。 |
| 1-18 | **FadeClockDomain + FadeGeneration 導入**: 全 fade 層に共通 generation カウンタ。FadeAccumulator が generation 管理の権威。generation mismatch 時 telemetry + 自動再同期。 | `src/audioengine/CrossfadeRuntime.h` + `src/core/SnapshotFadeState.h` | 低 | P1-9 v11.0新規。 |

### 完了条件 {#pr-1-完了条件}

- `ISRRetireRouter`（stateless dispatcher）が入口。**Fast Lane（SPSC, RT critical）+ Slow Lane（Worker Queue, bulk cleanup）の2-lane で Retire を実行**
- `EpochDomain::enqueueRetire()` は deprecated wrapper として維持（新規呼び出し禁止）
- **`SnapshotRetireFacade`（薄層）維持**: routing + telemetry + ordering validation。state owner ではない。実 delete は Router 経由。
- `RetireStateMachine` enum が実装され、主要 retire 経路が状態を追跡
- `RetireBackpressurePolicy` が実装され、飽和時の drop/degrade 戦略が明確
- **retry は deferred retry queue（bounded）で管理**: RT 即座 drop + telemetry。Non-RT 最大3回指数バックオフ。
- **`reclaimRetired()` 直接呼び出しゼロ**: 全箇所を `requestReclaim()`（enqueue only）に置換
- **Retire Commit Barrier 2-tier 実装**: soft（優先制御）+ hard（安全停止）
- **SnapshotCoordinator の direct enqueueRetire/reclaimRetired ゼロ**: 全5箇所を Router 経由に変更

### Rollback 戦略 {#pr-1-rollback-戦略}

- `ISRRetireRouter` + Policy Lane 導入は additive change（既存コード削除前に新クラス追加）
- `EpochDomain::enqueueRetire()` private 化は行わないため、破壊的変更なし
- rollback 時は新クラスの削除のみで旧状態に復帰可能（既存コードは unchanged）

---

## PR-2: Reclaim の RT 境界問題修正（Phase-A, クリティカル）

### 問題 {#pr-2-問題}

`AudioEngine::tryReclaimResources()` (`AudioEngine.Threading.cpp:75`) が Timer コールバック上で `m_epochDomain.reclaimRetired()` を呼び、その中で `DeferredDeletionQueue::reclaim()` → `deleter()` → `SnapshotFactory::destroy()` → `delete` が実行される。これは Audio Thread（Timer）上でのデストラクタ実行になりうる。

### 現状の reclaim 経路（6箇所）

| # | 呼び出し元 | ファイル | スレッド | 分類 |
| --- | ----------- | ---------- | ---------- | ------ |
| 1 | `AudioEngine::tryReclaimResources()` | AudioEngine.Threading.cpp:75 | **Timer (RT上**) | ❌ |
| 2 | `AudioEngine::drainDeferredRetireQueues()` | AudioEngine.Threading.cpp:86,221 | NonRT | ✅ shutdown |
| 3 | `SnapshotCoordinator::~SnapshotCoordinator()` | SnapshotCoordinator.h:67 | NonRT | ✅ デストラクタ |
| 4 | `SnapshotCoordinator::reclaim()` | SnapshotCoordinator.h:103 | NonRT | ✅ explicit |
| 5 | `EQProcessor::enqueueDeferredDeleteWithFallback()` | EQProcessor.Core.cpp:44,63,118 | NonRT | ✅ fallback |
| 6 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` | AudioEngine.h:3193 | NonRT | ✅ fallback |

### タスク {#pr-2-タスク}

| # | タスク | ファイル | リスク | 備考 |
| --- | -------- | ---------- | -------- | ------ |
| 2-1 | `tryReclaimResources()` から `m_epochDomain.reclaimRetired()` を削除 | `src/audioengine/AudioEngine.Threading.cpp` | **重要** | 呼び出し元が Timer であることを確認 |
| 2-2 | 代わりに `tryReclaimResources()` を `m_epochDomain.pendingRetireCount()` の監視のみに縮退（圧力閾値超過時は別経路で drain 発火） | `src/audioengine/AudioEngine.Threading.cpp` | 中 | 既存の retirePressureLevel_ 監視と統合。retry/fallback なし（実コード確認済み） |
| 2-3 | **`DeferredDeletionWorkerThread` を新設**（唯一の delete 実行スレッド）。全 reclaim 経路（Timer/drainDeferredRetireQueues/DSPQuarantine）はこの Worker への enqueue のみ行う | `src/audioengine/DeferredDeletionWorkerThread.h`（新規） + `AudioEngine.h` + `AudioEngine.Threading.cpp` | **高** | P0-1 対応。新規スレッドだが責務は単一（DeferredDeletionQueue::reclaim() 呼び出しのみ）。既存 rebuild thread とは独立。 |
| 2-4 | `RuntimePublicationCoordinator` に `requestReclaim()` API を追加（Worker への enqueue） | `src/core/RuntimePublicationCoordinator.h` | 中 | reclaim scheduling 権限を coordinator に集約。Worker が唯一の実行主体。 |
| 2-5 | 移行中は `retireQueueDepth_` が閾値を超えた場合のみ `drainDeferredRetireQueues(false)` を kick（Timer から） | `src/audioengine/AudioEngine.Timer.cpp` | 中 | 完全削除までの緩和策 |
| 2-6 | CI gate 追加: `tryReclaimResources` 内に `reclaimRetired` が存在しないことを clang-tidy AST + grep フォールバックでチェック | `.github/scripts/` + `.clang-tidy` | 低 | 回帰防止 |
| 2-7 | `DeferredDeletionQueue::reclaim()` 先頭に `ASSERT_NON_RT_THREAD()` を追加 | `src/DeferredDeletionQueue.h` | 低 | P0-1 実行時ガード。Worker Thread 以外からの reclaim を実行時検出。 |
| 2-8 | `drainDeferredRetireQueues()` 内の `coordinator.reclaim()` + `reclaimRetired()` を Worker enqueue に置換 | `src/audioengine/AudioEngine.Threading.cpp` | 中 | P0-1 対応。shutdown 時の drain も Worker 経由で実行。 |

### 構造的強制（新設）

本 PR では以下の**構造的強制**を導入し、単なる「設計宣言」ではなく「実装強制」で RT 安全性を担保する:

| 強制方法 | 説明 | 対象 |
| ---------- | ------ | ------ |
| `ASSERT_NON_RT_THREAD()` を `ISRRetireGateway::enqueueRetire()` 先頭に配置 | reclaim につながる enqueue 経路を NonRT 限定に | PR-1 の ISRRetireGateway |
| `RTCapabilityFirewall::enter()` / `leave()` で RT callback の reclaim 呼び出しを abort | FirewallToken 未解放時の reclaim を実行時検出 | PR-2 の Timer 経路 |
| `static_assert` で `tryReclaimResources()` が `AudioEngine::timerCallback()` 以外から呼ばれないことを検査（コンパイル時） | 将来の誤用防止 | `AudioEngine.Threading.cpp` |
| CI gate: `select-string "reclaimRetired" src/audioengine/AudioEngine.Threading.cpp` で Timer 内の reclaim 呼び出しを検出 | 回帰防止 | `.github/scripts/` |
| **RT delete 完全遮断**: `RTDeleteForbidden<T>` ラッパー型を導入し、RT thread から delete 操作をコンパイル時に禁止 | `ISRRTExecution.h` に `static_assert` + `RTAllocatorFirewall` と統合 | 中優先度 |
| `DeferredDeletionQueue::reclaim()` 先頭に `ASSERT_NON_RT_THREAD()` を追加 | 実行時ガード（既存の全 reclaim 経路） | `DeferredDeletionQueue.h` |

### 代替案

**ALT-2A**: `DeferredDeletionQueue::reclaim()` の deleter 呼び出しを Work Queue に委譲し、deleter 実行を常に NonRT スレッドにオフロードする。この場合 `tryReclaimResources()` からの呼び出しは安全になるが、deleter 遅延が増加する。

**採用判断**: PR-2 では ALT-2A を採用せず、`tryReclaimResources` からの `reclaimRetired` 完全除去を優先する。ALT-2A は Phase-C 以降で検討。

### 完了条件 {#pr-2-完了条件}

- `tryReclaimResources()` からの `reclaimRetired()` 呼び出しゼロ
- `ASSERT_NON_RT_THREAD()` が `ISRRetireGateway::enqueueRetire()` に存在
- `DeferredDeletionWorkerThread` が唯一の reclaim 実行スレッド
- `DeferredDeletionQueue::reclaim()` 先頭に `ASSERT_NON_RT_THREAD()` あり
- `drainDeferredRetireQueues()` の reclaim 呼び出しが全て Worker enqueue 経由
- **`EQProcessor::enqueueDeferredDeleteWithFallback()` の retry ループ (kMaxRetry=4) を削除** → enqueue failure = drop or quarantine
- **`AudioEngine::enqueueDeferredDeleteNonRtWithResult()` の reclaim+retry パスを削除** → 初回 enqueue 失敗時は即座に drop または quarantine
- **`epochDomain.reclaimRetired()` の直接呼び出しゼロ** → 全箇所を `epochDomain.requestReclaim()`（enqueue only）に置換
- CI gate で Timer 内の reclaim 回帰を自動検出
- CI gate で `reclaimRetired()` の直接呼び出し検出 + retry パターン検出
- `retireQueueDepth_` / `fallbackQueueDepth_` の監視のみ Timer に残存（許容）

### Rollback 戦略 {#pr-2-rollback-戦略}

- `tryReclaimResources()` の削除は単一関数内の1行削除。rollback は行の再追加のみ
- `ASSERT_NON_RT_THREAD()` 追加は防御的変更。問題発生時も削除のみで復帰可能

---

## PR-3: enqueueRetire 経路の完全統一（Phase-A）

### 問題 {#pr-3-問題}

`EpochDomain::enqueueRetire()` の外部呼び出しが3系統存在。`ISRRuntimePublicationCoordinator::enqueueRetire()` が authorized caller とされているが、依然として `EpochDomain::enqueueRetire()` が deprecated 状態で露出している。

### 現状の呼び出し系統（semble 確定）

| 呼び出し元 | ファイル | 系統 |
| ----------- | ---------- | ------ |
| `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` | AudioEngine.h:3182,3193 | 1 |
| `ISRRuntimePublicationCoordinator::enqueueRetire()` | ISRRuntimePublicationCoordinator.cpp:151 | 2 (authorized) |
| `EQProcessor::enqueueDeferredDeleteWithFallback()` | EQProcessor.Core.cpp:59 | 3 |

### タスク {#pr-3-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| 3-1 | `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` の `m_epochDomain.enqueueRetire()` → `runtimePublicationBridge_.enqueueRetire()` に置換 | `src/audioengine/AudioEngine.h` | **重要** |
| 3-2 | `EQProcessor::enqueueDeferredDeleteWithFallback()` の `m_epochDomain.enqueueRetire()` → coordinator 経由に置換 | `src/eqprocessor/EQProcessor.Core.cpp` | **重要** |
| 3-3 | フォールバックパターン（enqueue失敗→reclaim→再enqueue）を coordinator 側の `enqueueRetireWithRetry()` として共通化 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | 中 |
| 3-4 | 移行完了後、`EpochDomain::enqueueRetire()` を private にする | `src/core/EpochDomain.h` | 低 |
| 3-5 | CI gate 追加（AST昇格）: `EpochDomain::enqueueRetire` の外部呼び出しゼロを clang-tidy AST + grep フォールバックでチェック | `.github/scripts/` + `.clang-tidy` | 低 |

### 完了条件 {#pr-3-完了条件}

- `EpochDomain::enqueueRetire()` の外部呼び出しゼロ
- 全 enqueueRetire が `ISRRuntimePublicationCoordinator` または `SnapshotRetireManager` 経由
- フォールバックパターンが coordinator 側で共通化済み

---

## PR-4: SnapshotCoordinator 責務完結（Phase-B）

### 問題 {#pr-4-問題}

`SnapshotCoordinator` は以下の責務を兼務（God Object）:

1. Snapshot slot管理 → `SnapshotSlotStore` に委譲済み（`m_slots` フィールド）
2. Fade制御 → `SnapshotFadeState` に委譲済み（`m_fade` フィールド）
3. **RCU retire → `SnapshotRetireManager` に未委譲（PR-1 で対応）**
4. ライフサイクル遷移 → **~SnapshotCoordinator() が自前で retire 発行**

### タスク {#pr-4-タスク}

| # | タスク | ファイル | リスク | 備考 |
| --- | -------- | ---------- | -------- | ------ |
| 4-1 | PR-1 完了後、`SnapshotCoordinator` から `EpochDomain` の直接メンバを削除（`SnapshotRetireManager` 経由に統一済みのはず） | `src/core/SnapshotCoordinator.h` | 低 | PR-1 依存 |
| 4-2 | `SnapshotCoordinator::reclaim()` のインターフェースを `SnapshotRetireManager::reclaim()` の内部呼び出しに変更 | `src/core/SnapshotCoordinator.h` | 低 | 外部API変更なし |
| 4-3 | `SnapshotCoordinator` のテストを `SnapshotSlotStore` / `SnapshotFadeState` / `SnapshotRetireManager` のユニットテストに分離 | `test/` | 低 | テスト容易性向上 |
| 4-4 | `SnapshotCoordinator` に「このクラスの責務は state transition event 生成のみ」であることをコメントとして明記 | `src/core/SnapshotCoordinator.h` | 低 | ドキュメンテーション |

### 完了条件 {#pr-4-完了条件}

- `SnapshotCoordinator` が直接 `EpochDomain` の retire API を呼ばない（PR-1で達成）
- `SnapshotCoordinator` の全 retire 操作が `SnapshotRetireManager` 経由
- クラス責務がコメントで明文化

---

## PR-5: SnapshotRetireManager orphan 解消の監査強化

### 問題 {#pr-5-問題}

`SnapshotRetireManager` は `src/core/SnapshotRetireManager.h` に定義されているが、全ソースコードで参照ゼロ。抽出作業（Phase 5）で作成されたが統合作業が行われていない。同様の orphan クラスが他に存在しないか監査する。

### タスク {#pr-5-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| 5-1 | `src/core/` 配下の全ヘッダについて、`#pragma once` 以外のファイルが実際にインクルードまたは参照されているか自動チェック | `src/core/*.h` | 低 |
| 5-2 | orphan が発見された場合、削除または統合の判断を行う | `src/core/` | 低 |
| 5-3 | CI gate に「1ヶ月以上参照がない新規ファイルを警告する」ルールを追加（任意） | `.github/scripts/` | 低 |

### 完了条件 {#pr-5-完了条件}

- `src/core/` 配下に orphan ファイルが存在しない
- `SnapshotRetireManager` が統合済み（PR-1）

---

## PR-B-0: CrossfadeRuntime 統合ブリッジ（Phase-C, 新規）

### 問題 {#pr-5-sub-問題}

fade 状態が以下に分散しており、PR-6 の「文書化＋重複チェック」だけでは構造的解決にならない:

1. **`SnapshotCoordinator` → `SnapshotFadeState`**: スナップショット切り替えのクロスフェード進行管理
2. **`CrossfadeRuntime`** (`AudioEngine::crossfadeRuntime_`): DSPCore 切り替え時の transition 実行状態管理
3. **`FadeAccumulator`** (`AudioEngine::currentFade_`): Audio callback 内の crossfade ゲイン計算の作業状態

### 設計判断

**3者を統合（1クラス化）するのではなく、責務境界を固定して「統治可能」にする。**

**FadeAuthority 単一書き込み原則（v4.0 新設）**: fade state への書き込みは常に1箇所からのみ行う。複数 source からの同時書き込みを禁止する。

```text
FadeAuthority (single writer = DSPTransition::onPublishCompleted)
  ↓
  SnapshotFadeState (read-only projection for snapshot switch)
  ↓
  CrossfadeRuntime (derived state for DSP runtime transition)
  ↓
  FadeAccumulator (callback-local working copy)
```

| コンポーネント | 責務（固定後） | 書き込み元 | 備考 |
| --------------- | --------------- | ----------- | ------ |
| `SnapshotFadeState` | Snapshot 切り替えのクロスフェード進行管理 | SnapshotCoordinator のみ | 現状維持 |
| `CrossfadeRuntime` | DSPCore 切り替えの runtime transition 状態管理 | DSPTransition::onPublishCompleted のみ | SnapshotFadeState への参照を持ち、fadeTimeSec 等の情報を伝搬 |
| `FadeAccumulator` | Audio callback 内の即時 crossfade ゲイン計算 | Audio callback のみ（read-only 投影） | DSPCore ローカルの作業領域 |

### タスク {#pr-6-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| B0-1 | `CrossfadeRuntime` に `SnapshotFadeState` への参照（const*）を追加（optional） | `src/audioengine/CrossfadeRuntime.h` + `src/audioengine/CrossfadeRuntime.cpp` | 低 |
| B0-2 | 重複フィールドの精査: `fadeTimeSec` が `CrossfadeRuntime::getQueuedFadeTimeSec()` と `AudioEngine` の atomic `m_irFadeTimeSec` / `m_osFadeTimeSec` 等の間で二重管理になっていないか確認 | `src/audioengine/CrossfadeRuntime.h` + `src/audioengine/AudioEngine.h` | 中 |
| B0-3 | `FadeAccumulator` が `SnapshotFadeState` および `CrossfadeRuntime` の値と矛盾しないことを単体テストで確認 | `test/`（新規または既存） | 低 |
| B0-4 | 責務境界の文書化（`doc/work21/fade_responsibility_matrix.md`） | 新規 | 低 |

### 完了条件 {#pr-6-完了条件}

- 3者の責務境界がコードコメントと文書で明記されている
- 重複フィールドが存在しない（確認済み）
- `CrossfadeRuntime` の単一書き込み原則（DSPTransition のみが start() を呼ぶ）がコードコメントで明示されている

---

## PR-6: Fade 状態の整理（Phase-C）

### 問題 {#pr-6-問題}

fade 状態が以下に分散している（semble/serena 確認）:

1. **`SnapshotFadeState`** (`src/core/SnapshotFadeState.h`): SnapshotCoordinator 配下の fade。`start()` / `advance()` / `tryComplete()` / `resetToIdle()` / `state()` / `alpha()`
2. **`CrossfadeRuntime`** (`src/audioengine/CrossfadeRuntime.h`): AudioEngine 直下の独立した fade runtime。`isPending()` / `useDryAsOld()` / `getQueuedFadeTimeSec()` / `getDryScaleTarget()` / `getStartDelayBlocks()` / `getDryHoldSamples()`
3. **`FadeAccumulator`** (`AudioEngine.h:3506`): AudioEngine メンバの `currentFade_`。`gainFrom` / `gainTo` / `active` の3フィールドのみ。

これら3者は役割が異なる（Snapshot遷移用 / Runtime transition用 / DSP crossfade実行用）が、責務境界が文書化されていない。

### タスク {#pr-6-sub-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| 6-1 | 3者の責務境界を文書化（`doc/work21/fade_responsibility_matrix.md`） | 新規 | 低 |
| 6-2 | `SnapshotFadeState` の責務を「スナップショット切り替えのクロスフェード進行管理」に固定。Runtime crossfade との混同をコメントで防止 | `src/core/SnapshotFadeState.h` | 低 |
| 6-3 | `CrossfadeRuntime` の責務を「DSPCore 切り替え時の transition 実行状態管理」に固定。Snapshot fade との分離をコメントで明記 | `src/audioengine/CrossfadeRuntime.h` | 低 |
| 6-4 | `FadeAccumulator` の責務を「Audio callback 内での crossfade ゲイン計算の作業状態」に固定 | `src/audioengine/AudioEngine.h` | 低 |
| 6-5 | 不要な重複フィールドがないか確認（例: `fadeTimeSec` が CrossfadeRuntime と atomic 変数の両方に存在しないか） | `src/audioengine/AudioEngine.h` + `src/audioengine/CrossfadeRuntime.h` | 中 |
| 6-6 | `CrossfadeRuntime` の `start()` が `DSPTransition` からのみ呼ばれていることを確認（単一書き込み原則） | `doc/work19/refactoring_plan_v2.md` 参照 | 低 |

### 完了条件 {#pr-6-sub-完了条件}

- 3者の責務が文書化され、コードコメントに反映済み
- 重複フィールドが存在しない
- `CrossfadeRuntime` の単一書き込み原則が守られている

---

## PR-7: ownerThreadId 排除（Phase-D, 低優先度）

### 問題 {#pr-7-問題}

`ObservedRuntime` (`src/core/ObservedRuntime.h`) が `ownerThreadId` フィールドを持ち、`get()` / `operator bool()` で `std::this_thread::get_id()` との比較を行っている。std::thread::id の再利用問題や coroutine/task system との非互換性のリスクがある。

### 防御の実態

`EpochDomainReaderGuard` が epoch-based reader protection を提供しており、スレッドIDチェックは defense-in-depth の補助的安全策。実質的な保護は `EpochDomainReaderGuard` が担当している。

### タスク {#pr-7-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| 7-1 | `ObservedRuntime::ownerThreadId` を削除し、`get()` / `operator bool()` からスレッドIDチェックを除去 | `src/core/ObservedRuntime.h` | 中 |
| 7-2 | `EpochDomainReaderGuard` の epoch check のみで安全性が担保されることを確認 | `src/core/EpochDomain.h` + `src/core/ObservedRuntime.h` | 低 |
| 7-3 | 移行後、`ObservedRuntime` を `ObserveToken` に正式にリネーム（現在は using エイリアスのみ） | `src/core/ObservedRuntime.h` | 低 |
| 7-4 | `RuntimeReadHandle` が `ObservedRuntime` 経由でスレッドセーフに動作することを確認 | `src/audioengine/AudioEngine.h` | 低 |

### 完了条件 {#pr-7-完了条件}

- `ownerThreadId` がソースコードから完全削除
- `ObservedRuntime` が `ObserveToken` にリネーム
- 全 getter が `EpochDomainReaderGuard` のみで保護される

---

## PR-8: Atomic ordering モデルの文書化（Phase-D, 低優先度）

### 問題 {#pr-8-問題}

各atomic操作に詳細な HB (happens-before) コメントは記述されているが、`ISREpochMemoryModel.h` のようなグローバルな ordering model が存在しない。`publication_ordering_matrix.md` に部分的な整序モデルは存在するが、コードとして実装されていない。

### タスク {#pr-8-タスク}

| # | タスク | ファイル | リスク |
| --- | -------- | ---------- | -------- |
| 8-1 | `ISREpochMemoryModel.h` を新規作成し、Phase Ordering Model を定式化 | `src/audioengine/ISREpochMemoryModel.h`（新規） | 低 |
| 8-2 | 既存の `publication_ordering_matrix.md` の内容をコードとして移植（enum class + constexpr 配列） | 上記 | 低 |
| 8-3 | `Publish → Activate → Retire → Reclaim` のフェーズ順序を `static_assert` で検証可能にする | 上記 | 低 |
| 8-4 | ドキュメント `doc/work21/atomic_ordering_model.md` を作成 | 新規 | 低 |

### 完了条件 {#pr-8-完了条件}

- `ISREpochMemoryModel.h` が存在し、Phase Ordering Model がコードとして定義されている
- 既存の HB コメントとモデルに矛盾がないことを確認

---

## リスク評価

| リスク | 該当PR | 確率 | 影響 | 対策 |
| -------- | -------- | ------ | ------ | ------ |
| SnapshotRetireManager 統合による retire 漏れ | PR-1 | 中 | 大（UAF） | 段階的移行 + CI gate |
| reclaim RT削除によるメモリ圧迫 | PR-2 | 中 | 中 | 圧力監視 + drain kicking |
| enqueueRetire 統合によるフォールバック消失 | PR-3 | 低 | 中 | coordinator 側に retry ロジックを共通化 |
| ownerThreadId 削除による安全網低下 | PR-7 | 低 | 低 | EpochDomainReaderGuard で代替済み確認済み |
| CrossfadeRuntime/SnapshotFadeState 責務混同 | PR-6 | 中 | 中 | 文書化 + コードコメント |

## ★ 依存関係グラフと全体スケジュール

### 完全依存グラフ

```text
Phase-A (Retire Authority 一元化)
  PR-1 (ISRRetireGateway導入) ──────────────────────┐
    │  └─ タスク1-0..1-2: 独立して先行可             │
    │  └─ タスク1-3..1-8: SnapshotCoordinator疎通   │
    │                                                │
    ├──→ PR-B-0 (CrossfadeRuntime統合) ←→ PR-6     │
    │         ↑ PR-1完了が前提                       │
    │                                                │
    ├──→ PR-3 (enqueueRetire経路統一)                │
    │         ↑ PR-1のISRRetireGateway必須            │
    │                                                │
    └──→ PR-2 (Reclaim RT排除)                      │
              ↑ PR-1のISRRetireGateway +              │
                PR-3の経路統一が前提                   │
                                                      │
Phase-B (SnapshotCoordinator完結)                    │
  PR-4 ←──── PR-1, PR-3 完了が前提 ──────────────────┘
    │
    └──→ PR-5 (orphan監査)
                                                      │
Phase-C (Fade整理)                                    │
  PR-B-0 ←→ PR-6 (両者並行可、PR-1完了が前提) ─────┘
                                                      │
Phase-D (安全性向上)                                   │
  PR-7 → PR-8 (独立して実施可能)                     │
```text

### Push / Pull 判断基準

| 判断 | 条件 |
| ------ | ------ |
| PR-1 の完了 | `ISRRetireGateway` が実装され、`SnapshotCoordinator` からの deprecated 呼び出しが全滅 |
| PR-2 着手 | PR-1 完了 + PR-3 完了（ISRRetireGateway + 全経路統合が前提） |
| PR-B-0 着手 | PR-1 完了（ISRRetireGateway 存在一式） |
| PR-4 着手 | PR-1 完了（SnapshotCoordinator の RetireManager 経由が確定） |
| 全 Phase-A 完了 | PR-1 + PR-2 + PR-3 + PR-B-0 の完了 |
| Phase-B 開始 | 全 Phase-A 完了 |

### Rollback 戦略（全 PR 共通）

| PR | Rollback 方法 | 影響範囲 | データ損失 |
| ---- | -------------- | ---------- | ----------- |
| PR-1 | 1-6 (EpochDomain private化) を revert。ISRRetireGateway は additive のため残しても無害 | 低（EpochDomain public API 復帰のみ） | なし |
| PR-2 | `tryReclaimResources()` の1行を復帰。ASSERT_NON_RT_THREAD は削除のみ | 低（Timer reclaim 復帰で RT 問題再発） | なし |
| PR-3 | coordinator 経由を戻し、従来の EpochDomain 直呼びに復帰 | 低 | なし |
| PR-B-0 | CrossfadeRuntime の分離状態を維持（ブリッジ削除のみ） | 中（fade の責務境界が不明確に戻る） | なし |
| PR-4 | SnapshotCoordinator の retire 経路を元に戻す（PR-1 完了が前提のため不要） | 低 | なし |
| PR-5 | orphan ファイルは削除せず、監査結果の破棄のみ | 低 | なし |
| PR-6 | 文書化の破棄のみ。コード変更は最小限 | 低 | なし |
| PR-7 | ownerThreadId を復帰 | 低 | なし |
| PR-8 | 文書/モデルファイルの削除のみ | 低 | なし |

### クリティカルパス

```text
PR-1 (ISRRetireGateway) → PR-3 (経路統一) → PR-2 (Reclaim RT排除)
                           ↘ PR-B-0 (CrossfadeRuntime統合)
```

**PR-1 が全体のボトルネック。PR-1 未完了の場合、PR-2/PR-3/PR-B-0/PR-4 は着手不可。**

### 推奨初手

1. PR-1 タスク 1-0 から着手: `ISRRetireGateway` クラスの新規作成（thin router, 複雑ロジック禁止）。他コードに影響を与えない additive change であるため、最も安全な初手。
2. 続いて PR-1 タスク 1-3: `SnapshotCoordinator` の retire 呼び出し5箇所を `ISRRetireGateway::enqueueRetire()` に置換（最も影響が大きく、かつ ISRRetireGateway ができていれば独立してテスト可能）。
3. 置換後、`Strict Atomic Dot-Call Scan` + `Build_CMakeTools` で回帰チェック
4. 合格後、PR-1 残タスク（1-4/1-5/1-6）+ PR-3 に進む

## 付録: 全改修漏れ発見一覧（検証結果より）

### notfinished7.md 記載の7クレーム

| # | クレーム | 対応PR | 優先度 |
| --- | --------- | -------- | -------- |
| ① | Epoch/RCU責務分散 | PR-1, PR-3 | S |
| ② | SnapshotCoordinator God Object | PR-4 | A |
| ③ | Atomic ordering局所最適 | PR-8 | C |
| ④ | Snapshot生成比較二重化 | ドキュメントのみ | D |
| ⑤ | RT境界跨ぎ | PR-2 | S |
| ⑥ | Thread ID依存 | PR-7 | C |
| ⑦ | Fade密結合 | PR-6 | B |

### 検証で発見した notfinished7.md の見落とし

| # | 発見事項 | 対応PR | 優先度 |
| --- | --------- | -------- | -------- |
| A | `SnapshotRetireManager` 未統合（orphan） | PR-1, PR-5 | **S** |
| B | `tryReclaimResources` のRT上reclaim | PR-2 | **S** |
| C | `CrossfadeRuntime` 独立fade状態 | PR-B-0, PR-6 | **A** |
| D | `ISRDSPHandle` 別retire/reclaim経路 | PR-3（間接的） | B |
| E | `enqueueDeferredDeleteNonRtWithResult` フォールバック | PR-3 | A |

### レビュー指摘の反映状況（v1.0→v2.0）

| レビュー指摘 | 反映先 | 状態 |
| ------------- | -------- | ------ |
| Retire単一経路に gateway 必須 | PR-1: `ISRRetireGateway` 新設 | ✅ 反映（v2.0） |
| SnapshotRetireManager 責務削減 | PR-1: QUEUE ONLY に純化、coordinator 非依存 | ✅ 反映（実コード確認済み） |
| EpochDomain 直接呼び出し禁止 | PR-1-6: enqueueRetire 完全 private 化 → **v3.0で撤回**: internal API 維持 | ⚠ v3.0で修正 |
| RT safety 構造的強制 | PR-2: ASSERT_NON_RT_THREAD/RTCapabilityFirewall/CI gate | ✅ 反映（既存機構活用） |
| Timer reclaim 排除 | PR-2: tryReclaimResources からの reclaimRetired 削除 | ✅ 反映 |
| WorkerThread への完全移行 | PR-2: 専用 GC または Worker 利用 | ✅ 反映（代替案含む） |
| CrossfadeRuntime 統合 | PR-B-0: 新規PR。責務境界固定＋ブリッジ化 | ✅ 反映 |
| RT/NonRT境界が呼び出し依存 | PR-2: 構造的強制で対応 | ✅ 反映 |
| enforce層不在 | 既存の RTCapabilityFirewall/ASSERT_NON_RT_THREAD を活用 | ✅ 反映（新規作成不要） |
| 依存順序グラフ明示 | 全体スケジュール節に完全依存グラフ追加 | ✅ 反映 |
| rollback戦略 | 全 PR に Rollback 戦略追加 | ✅ 反映 |

### レビュー指摘の反映状況（v2.0→v3.0 第2回監査）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| ISRRetireGateway 単一障害点化リスク | **妥当** — 入口は1つだが内部は目的別に分類すべき | ISRRetireGateway → ISRRetireRouter + 3 Policy Lane（DSP/Snapshot/Deferred） | ✅ v3.0反映 |
| SnapshotRetireManager QUEUE ONLY 非成立 | **却下** — 実コード確認: `SnapshotRetireManager` は `GlobalSnapshot*` 専用。DSP/Fade/Handle/Quarantine は対象外。QUEUE ONLY は成立可能 | 変更なし（QUEUE ONLY 維持） | ✅ 反論完了 |
| CrossfadeRuntime 未統合 | **妥当** — FadeAuthority Layer の明示が必要 | アーキテクチャ決定節に FadeAuthority 設計を追加（PR-B-0 で対応） | ✅ v3.0反映 |
| EpochDomain private化は危険 | **妥当** — 実コード確認: `[[deprecated]]` ラッパーとして維持可能。private化すると ISRClosure/ISRRetireRuntimeEx の既存依存が破壊される | EpochDomain::enqueueRetire() 完全 private化を撤回。internal API（deprecated wrapper）として維持 | ✅ v3.0反映 |
| retry/reclaim/backpressure 未設計 | **一部妥当** — フォールバックパターン自体は PR-3 に記載済み。backpressure/saturation の明示的設計が不足 | `RetireBackpressurePolicy` 新設（PR-1-7）。既存 HWMark/LWMark と統合 | ✅ v3.0反映 |
| Retire State Machine 不在 | **妥当** — 存在しない。責務曖昧の原因 | `RetireStateMachine` enum 新設（Created→Active→PendingRetire→Retiring→Reclaimed） | ✅ v3.0反映 |
| Retire Observability 不在 | **妥当** — 既存 `RuntimeBackpressureTelemetry` はあるが Lane 別テレメトリなし | Lane 別 latency/queue depth/reclaim delay テレメトリ追加（PR-1-8） | ✅ v3.0反映 |
| 「構造統一」より「時間軸分離」が必要 | **一部妥当** — Viewpoint として採用。Router + Policy Lane の「分類された統一」で対応 | アーキテクチャ図に時間軸別 Lane を反映 | ✅ v3.0反映 |

### レビュー指摘の反映状況（v3.0→v4.0 第3回監査）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Router 単一障害点化リスク | **妥当** — 高頻度 retire でホットスポット化の可能性 | Router パフォーマンス設計（分岐テーブル化 + fast-path inline decision）を追加 | ✅ v4.0反映 |
| Retire State Machine 論理/物理混在 | **妥当** — 単一状態遷移図は時間軸を無視 | 2層モデル（Logical: Active/PendingRetire / Physical: Draining/ReclaimReady/Freed）に変更。遷移条件を明示 | ✅ v4.0反映 |
| SnapshotRetireManager QUEUE ONLY が危険（epoch binding欠如） | **却下** — 実コード確認: `retire(GlobalSnapshot*, uint64_t epoch)` で epoch パラメータ受取済み。`DeletionQueue::reclaim()` で `isOlder(entry.epoch, minReaderEpoch)` による epoch 安全確認済み | 変更不要 | ✅ 反論完了 |
| Crossfade/Snapshot fade 統一 | **妥当** — 単一書き込み原則の明示が必要 | FadeAuthority 単一書き込み原則を PR-B-0 に明示（書き込み元テーブル追加） | ✅ v4.0反映 |
| RT delete 完全遮断 | **妥当** — `RTDeleteForbidden<T>` wrapper + `ASSERT_NON_RT_THREAD()` を追加 | PR-2 に static analysis + runtime guard 追加 | ✅ v4.0反映 |
| EpochDomain deprecated 残留 | **一部妥当** — 段階的廃止計画が必要 | EpochDomain::enqueueRetire() 段階的廃止計画（warning→error→private）を追加 | ✅ v4.0反映 |
| Policy Lane 優先順位未定義 | **妥当** — overlap 時の挙動が不定 | DSP > Snapshot > Deferred 優先順位 + Epoch tie-break を明文化 | ✅ v4.0反映 |
| Observability 不足 | **一部妥当** — worst-case latency / queue stall / epoch drift が不足 | 上記3メトリクスを Observability 表に追加 | ✅ v4.0反映 |
| Router determinism 保証不足 | **妥当** — RT 環境では branchless fallback が必須 | 関数ポインタテーブル化 + fast-path inline decision を追加 | ✅ v4.0反映 |

### レビュー指摘の反映状況（v4.0→v5.0 第4回監査）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| **P0-1**: delete 実行経路が複数スレッドに分散（Timer/drainDeferredRetireQueues/DSPQuarantine） | **妥当** — `tryReclaimResources()` は Timer 上で `m_epochDomain.reclaimRetired()` 呼び出し。`drainDeferredRetireQueues()` は `coordinator.reclaim()` + `reclaimRetired()` を呼ぶ。3経路の並存を確認。 | `DeferredDeletionWorkerThread` 新設（PR-2-3）。全 reclaim 経路から Worker enqueue へ統一（PR-2-8）。`DeferredDeletionQueue::reclaim()` に `ASSERT_NON_RT_THREAD()`（PR-2-7）。 | ✅ v5.0反映 |
| **P0-2**: DSPHandleRuntime が独立した retire/reclaim/quarantine を持つ | **一部妥当** — 実体確認: `DSPHandleRuntime::retire()`/`reclaim()`/`quarantine()` は slot 状態遷移（metadata 管理）のみ。メモリ解放（`delete`）は行わない。しかし slot 状態と実メモリ寿命の整合は設計保証が必要。 | DSPHandleRuntime を Router 経由に統合（PR-1-0e 新設）。slot 状態遷移は Router からの callback で更新。 | ✅ v5.0反映 |
| **P0-3**: SnapshotRetireManager QUEUE ONLY だが Coordinator 依存の可能性 | **却下** — 実コード確認: `SnapshotRetireManager::retire(GlobalSnapshot*, uint64_t epoch)` は epoch パラメータを `DeletionQueue::enqueue()` に委譲。Coordinator 参照/fade状態/lifecycle判断は一切なし。`retire()` と `reclaim()` のみの純粋キュー。 | 変更不要。 | ✅ 反論完了 |
| **P0-4**: CrossfadeRuntime/SnapshotFadeState/FadeAccumulator 三重構造 | **一部妥当** — 3者は異なる責務層（Execution/Lifecycle/Mixing）に属するため完全統合は非現実的。しかし書き込み元の一貫性が不足。 | FadeAuthority 単一書き込み原則を PR-B-0 で拡充。全3層の書き込み権限マトリクスを追加（誰がどの層に書き込むか）。 | ✅ v5.0反映 |
| **P1-1**: tryReclaimResources の非決定的GC | **一部妥当** — 実体: `m_epochDomain.reclaimRetired()` 単一呼び出しのみ。retry/fallback は**存在しない**（←ユーザー懸念の一部を反論）。ただし Timer コンテキストでの解放実行自体が RT 違反リスク。 | PR-2 の方針維持: `reclaimRetired()` 削除＋圧力監視のみへ縮退。retry/fallback なし（実コード確認済み）を完了条件に明記。 | ✅ v5.0反映 |
| **P1-2**: EpochDomain deprecated→error 化が早すぎる | **妥当** — `ISRClosure` と `ISRRetireRuntimeEx` が `enqueueRetire()` を使用中。error 化で既存コンポーネント破壊。 | 段階的廃止目標を Phase-D（v7）に延期。現状 `[[deprecated]]` 維持。error 化は Phase-C 以降で再評価。 | ✅ v5.0反映 |
| **P1-3**: RetireRouter の責務制限 | **一部妥当** — thin router 設計は v4.0 で設計済み。fast-path も追加済み。 | CI gate で ISRRetireRouter の LOC 制限（最大200行）を追加。 | ✅ v5.0反映 |
| **P2-1**: RetireStateMachine Logical=decision owner 不明瞭 | **妥当** — 2層のうちどちらが遷移判断するか未定義。 | 「Logical state = 唯一の decision owner。Physical state は observer-only projection。Physical 側に遷移ロジック禁止」を RetireStateMachine 節に明文化。 | ✅ v5.0反映 |
| **P2-2**: DSPHandleRuntime の役割明確化 | **追記事項** — 実体は slot metadata manager。メモリ解放は行わない。 | `DSPHandleRuntime` 設計文書に「slot metadata manager（メモリ解放は EpochDomain 経路）」と明記。 | ✅ v5.0反映 |
| **P2-3**: Fade 3層 truth source の階層文書化 | **追記事項** — CrossfadeRuntime（Execution層）/ SnapshotFadeState（Publication層）/ FadeAccumulator（DSP Mixing層）の責務境界。 | PR-B-0 に truth source 階層図を追加。書き込み権限テーブルを拡充。 | ✅ v5.0反映 |

### レビュー指摘の反映状況（v5.0→v6.0 第5回監査）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| **P0-5**: retry パターン（再入性違反）— EQProcessor 4-retry / AudioEngine reclaim+retry | **妥当（実コード確認）** — `EQProcessor::enqueueDeferredDeleteWithFallback()`: `kMaxRetry=4`, `enqueue→reclaimRetired→advanceEpoch→再enqueue` のループ。`AudioEngine::enqueueDeferredDeleteNonRtWithResult()`: `reclaimRetired()` 後再 enqueue。両者とも「reclaim 後に enqueue」が同一トランザクションにあり、再入性リスク。 | retry 全面禁止（PR-2 完了条件に明記）。enqueue failure → drop または quarantine。CI gate で retry パターン検出。 | ✅ v6.0反映 |
| **P0-6**: `epochDomain.reclaimRetired()` 直接呼び出しが3経路 | **妥当（実コード確認）** — `tryReclaimResources()` / `drainDeferredRetireQueues()` / `EQProcessor` retry 内で直接呼び出し。reclaim が「直接実行 API」として露出。 | `reclaimRetired()` を deprecated API 化 → `requestReclaim()`（enqueue only）に置換。PR-2 完了条件に追加。debug assert で直接 delete 検出。 | ✅ v6.0反映 |
| **P1-4**: SnapshotRetireManager epoch 意味の外部依存 | **妥当** — epoch 値の意味整合は呼び出し元（SnapshotCoordinator ↔ EpochDomain）間の暗黙契約に依存。SnapshotRetireManager 自身は epoch 検証しない。 | SnapshotRetireManager を「epoch→lifetime 変換器」として明確化。または ISRRetireRouter に完全統合。設計文書に責務縮小を明記。 | ✅ v6.0反映 |
| **P1-5**: FadeAuthority 単一書き込みが不徹底 | **妥当** — CrossfadeRuntime（setUseDryAsOld/setFirstIrDryPending 等の setter あり）と SnapshotFadeState（coordinator 内更新）に独立した書き込み元が存在。 | FadeAuthority 単一書き込み原則を全層に拡張。書き込みは FadeAuthority 1箇所のみ。CrossfadeRuntime/SnapshotFadeState は派生 view（読み取り専用）。PR-B-0 で強制。 | ✅ v6.0反映 |
| delete 経路統一の不徹底 | **一部妥当** — DeferredDeletionWorkerThread追加は方向正しいが、reclaim API の直接露出が残存（P0-6 で対応）。 | P0-6 の reclaim API deprecation + enqueue only 化で経路統一を完了。 | ✅ v6.0反映 |

### レビュー指摘の反映状況（v6.0→v7.0 第6回監査）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| **P0-2**: Retire Commit Barrier 不在 — snapshot swap → fade → retire → delete の順序保証がない | **妥当（実コード確認）** — `SnapshotCoordinator::~SnapshotCoordinator()`: 2x `enqueueRetire()` + 1x `reclaimRetired()` を同期なしで実行。`switchImmediate()`: 同パターン。fade 完了を待たずに retire が進行する可能性。 | **Retire Commit Barrier 導入**: epoch 安全確認 + snapshot fade 完了 + RetireRouter enqueue 成功の三拍子同期ゲートを新設（P0-7）。 | ✅ v7.0反映 |
| **P0-3**: SnapshotCoordinator の直接 enqueueRetire/reclaimRetired が合計5箇所残存 | **妥当（実コード確認）** — `~SnapshotCoordinator()`: 2x `enqueueRetire` + 1x `reclaimRetired`。`switchImmediate()`: 1x `enqueueRetire`。`reclaim()`: 1x `reclaimRetired`。EpochDomain 直接参照あり。 | 全5箇所を ISRRetireRouter 経由に変更（P0-8）。SnapshotCoordinator から EpochDomain 直接参照を削除。 | ✅ v7.0反映 |
| **P0-4**: Timer reclaim が依然残存 — `tryReclaimResources()` + `drainDeferredRetireQueues(false)` が Timer から発火可能 | **一部妥当** — PR-2 で削除予定だが未実装段階。ユーザー懸念は正当。 | Timer からの全 reclaim 経路を Worker enqueue のみに変更。`tryReclaimResources()` は監視のみに縮退（P0-9）。 | ✅ v7.0反映 |
| **P1-1**: DSPHandleRuntime の retire/reclaim/quarantine が RetireRouter と意味的重複 | **一部妥当** — metadata 操作のみ（delete 実行なし）だが、slot 状態管理が Router の retire 判断と独立している。 | DSPHandleRuntime を pure metadata holder に縮小（P1-1）。状態管理は Router からの callback で更新。 | ✅ v7.0反映 |
| **P1-2**: SnapshotCoordinator がまだ部分 God Object | **一部妥当** — SlotStore/FadeState は物理分離済みだが、Coordinator が全操作の orchestrator。 | Coordinator 責務を observe/updateFade に縮小。publish/retire/reclaim は Router 委譲（P1-7）。 | ✅ v7.0反映 |
| SnapshotFadeState 独立 fade 管理が CrossfadeRuntime と重複 | **妥当（実コード確認）** — `SnapshotFadeState::advance()` が独自 alpha 計算。`CrossfadeRuntime::start()` が独立 gain ramp。**同一物理現象が2系統**。 | SnapshotFadeState を削除または thin mirror 化。fade 進行管理を CrossfadeRuntime に一元化（P1-6）。 | ✅ v7.0反映 |
| DSPQuarantineManager が delete 経路と誤認 | **却下（実コード確認）** — `DSPQuarantineManager` は atomic bool `quarantineFlags_` のみ。quarantine マークと確認だけ。`delete` は行わない。 | 変更不要。文書にコード確認結果を追記。 | ✅ 反論完了 |

### レビュー指摘の反映状況（v7.0→v8.0 第7回監査:「統合のしすぎ」フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Worker Thread 単一障害点 — shutdown 時の回収不能リスク | **妥当** — Worker 単一化は方向正しいが、AudioEngine shutdown 中の Thread join 前に deletion backlog が詰む可能性。 | **Emergency Drain tier-2 追加**（P0-1 強化）。shutdown 専用 lock-free queue drain。best-effort delete。 | ✅ v8.0反映 |
| Router が retire+snapshot+fade+epoch+queue を全て握る God Router 化 | **妥当** — 統合強化により責務集中が進行。 | **Router stateless 化**（P1-3 強化）。純粋ディスパッチテーブルに限定。メンバ変数禁止＋LOC 制限。 | ✅ v8.0反映 |
| Fade 統合が時間スケール衝突を無視 | **妥当** — CrossfadeRuntime（sample-rate）/ SnapshotFadeState（coarse state）/ FadeAccumulator（UI timing）は異なる時間スケール。統合は危険。 | **Fade 階層固定化**（P1-5/P1-6 方針転換）。SnapshotFadeState 削除撤回。3層の責務階層を維持し書き込み権限のみ統一。 | ✅ v8.0反映 |
| Timer 完全禁止が過剰 — periodic cleanup 消失リスク | **妥当** — 完全禁止ではなく enqueue-only 化が正しい。 | **Timer enqueue-only 化**（P0-9 修正）。監視維持、直接 delete のみ禁止。 | ✅ v8.0反映 |
| DSPHandleRuntime false positive — state layer と memory layer の混同 | **妥当（実コード確認）** — DSPHandleRuntime は slot metadata management（state layer）。RetireRouter は memory layer。二重体系ではない。 | DSPHandleRuntime を state layer として明確化。削除/統合不要。 | ✅ v8.0反映 |
| retry 完全禁止が過剰 — Non-RT では必要 | **妥当** — RT での retry のみ禁止が正解。Non-RT では最大1回許容。 | **RT retry 禁止、Non-RT 限定許可**（P0-5 修正）。 | ✅ v8.0反映 |
| Retire Commit Barrier が強すぎる — 1つ遅れると全停止 | **妥当** — dependency が強すぎる。2-tier 化が必要。 | **Barrier 2-tier 化**（P0-7 修正）。soft barrier（優先制御/タイムアウト許容）+ hard barrier（安全停止）。 | ✅ v8.0反映 |

### レビュー指摘の反映状況（v8.0→v9.0 第8回監査: 「責任拡散防止」フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Retire 経路が5系統以上並存 — ISRRetireRouter/RetireStateMachine/SnapshotRetireManager/DSPHandleRuntime/DeferredDeletionQueue/tryReclaimResources | **妥当** — 6系統の並存により「どれが正規経路か」状況依存。レース時に責任境界消失。 | **Retire 経路完全一本化**: Router → DeferredDeletionQueue のみが正規経路。SnapshotRetireManager 削除。DSPHandleRuntime を Request 変換層に格下げ。 | ✅ v9.0反映 |
| enqueue failure retry が残存（Non-RT 限定許可からさらに除外） | **妥当（v9.0で更に強化）** — Non-RT でも retry は再入性リスク。 | **enqueue failure retry 完全排除**: RT/Non-RT 問わず retry 禁止。drop + telemetry 統一。 | ✅ v9.0反映 |
| ownerThreadId 残存 — EpochDomainReaderGuard と併存で Debug/Release 挙動差 | **妥当** — `ObservedRuntime.h` に残存確認。 | **ownerThreadId 削除**: atomic ベースに統一。thread id は debug assert 限定。 | ✅ v9.0反映 |
| SnapshotRetireManager 孤立 — 参照ゼロ・dead code | **妥当** — Serena 検証で ZERO references 確認。 | **SnapshotRetireManager 削除**: Queue 機能は Router 内蔵 DeferredDeletionQueue に統合。 | ✅ v9.0反映 |
| Fade 3層の書き込み責務が不明瞭 | **一部妥当** — 3層自体は正しいが、各層の書き込み権限が未定義。 | **Fade 3層 単一書き込み制限**: 各層は自身の時間スケールの書き込みのみ。CrossfadeRuntime が唯一の fade write owner。 | ✅ v9.0反映 |

### レビュー指摘の反映状況（v9.0→v10.0 第9回監査: 「過剰単純化是正」フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Retire 完全一本化は SPOF — audio thread backlog で詰まり、latency spike で rebuild 不能 | **妥当** — 単一キュー＋drop only は transient overload に脆弱。 | **Retire 2-lane 化**（P0-1 修正）。Fast SPSC（RT critical）+ Slow Worker Queue（bulk cleanup）。drop は統計化＋deferred retry queue。 | ✅ v10.0反映 |
| SnapshotRetireManager 削除は「責務統合のやりすぎ」— Snapshot lifecycle と Retire lifecycle の結合 | **妥当** — 削除によりテスト単位消失・fade/epoch 境界曖昧化。 | **SnapshotRetireFacade 復活**（P1-4 修正）。薄層 facade（routing + telemetry + ordering validation）。state owner ではない。 | ✅ v10.0反映 |
| Fade 単一書き込み強制は DSP 的に危険 — click防止 crossfade 破綻、IR 切替時に急峻変化 | **妥当** — 3層は異なる時間スケールを持つ。単一書き込みは過剰制約。 | **Fade 3層 独立書き込み復活**（P1-5 修正）。単一書き込み方針撤回。各層独立書き込み許可。FadeClockDomain 導入。 | ✅ v10.0反映 |
| Timer reclaim 完全禁止の副作用 — delayed backlog, shutdown burst delete | **妥当** — 完全禁止ではなく soft rate limited enqueue が正しい。 | **Timer enqueue 許可（soft rate limited）**（P0-9 修正）。最大100Hz。超過時は telemetry + 先送り。delete は禁止維持。 | ✅ v10.0反映 |
| DSPHandleRuntime 格下げは Handle lifecycle 弱体化 — quarantine 非決定的化 | **妥当** — stateful handle authority として維持が必要。 | **DSPHandleRuntime Stateful Handle Authority 復活**（P2-2 修正）。Router は参照のみ。 | ✅ v10.0反映 |
| drop-only failure policy は recovery path 完全消失 | **妥当** — transient overload 時に機能低下。 | **deferred retry queue（bounded）追加**（P2-5 修正）。RT: drop + telemetry。Non-RT: 最大3回指数バックオフリトライ。 | ✅ v10.0反映 |
| 全体方針: 「single path + strict exclusion + drop」→「failure isolation + redundancy + bounded fallback」 | **妥当** — 過剰単純化是正。 | 全体方針転換を文書化。v10.0 改訂履歴に反映。 | ✅ v10.0反映 |

### レビュー指摘の反映状況（v10.0→v11.0 第10回監査: 「最終仕上げ」フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Retire 2-lane 間の二重 retire リスク — 同一オブジェクトが両 Lane に enqueue すると double delete | **妥当** — Router の分類ミスより「保守者が別経路追加」の方が高頻度。 | **RetireId + RetireRegistry 導入**（P0-10）。二重 retire を debug assert で検出。 | ✅ v11.0反映 |
| SnapshotRetireFacade 半年後の肥大化リスク — canRetire/canDelete/canPublish 追加で第二の Coordinator 化 | **妥当** — Facade の責務境界を明示しないと肥大化。 | **SnapshotRetireFacade 禁止事項明文化**（P1-8）。epoch/fade/publish/ownership 判定を禁止。routing/logging/metrics/validation のみ許可。 | ✅ v11.0反映 |
| Fade 3層 状態ドリフト — UI(20ms)とDSP(10ms)で fade 値乖離 | **妥当** — 独立書き込みは正しいが層間同期機構なし。 | **FadeClockDomain + FadeGeneration 導入**（P1-9）。全層共通 generation。mismatch 時 telemetry + 自動再同期。 | ✅ v11.0反映 |
| DeferredRetryQueue の上限・寿命・優先順位不足 | **妥当** — bounded のみでは不十分。 | **maxRetryCount=3, maxRetryAgeMs=5000** 設定。DSP/Snapshot retire drop 禁止。Cache retire のみ drop 許可。 | ✅ v11.0反映 |
| EmergencyDrain と EpochDomain の shutdown 順序未定義 | **妥当** — EpochDomain destroy 後の EmergencyDrain で事故。SnapshotCoordinator が EpochDomain 直接参照（未完了）。 | **ShutdownPhase 固定化**（P2-9）。Phase1-4 の順序固定。EmergencyDrain 実行条件に `EpochDomain::isAlive()` チェック。 | ✅ v11.0反映 |

### レビュー指摘の反映状況（v11.0→v12.0 第11回監査: 「最終仕上げ」フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| RetireRegistry Release 時も防御必要 — debug assert のみではリリースビルドで無防備 | **妥当** — release ビルドでも double retire は発生しうる。 | **Release reject + telemetry 追加**（P0-10）。debug: assert。release: reject + `telemetry.doubleRetire++`。 | ✅ v12.0反映 |
| RetireId 単独では未検出ケースあり — 同一オブジェクトが別 RetireId で二重 retire されると気づかない | **妥当** — DSP A が RetireRequest #100 と #101 の両方に登場しても検出不可。 | **RetireRegistry key を ObjectAddress+Generation に変更**（P0-10）。アドレス＋世代で同一オブジェクトを特定。 | ✅ v12.0反映 |
| DeferredRetryQueue thundering herd リスク — 大量失敗同時復帰で再増幅 | **妥当** — 同時復帰により同じタイミングで再 retry が集中。 | **random jitter 0.8-1.2x 追加**（P0-5）。指数バックオフ × jitter で分散復帰。 | ✅ v12.0反映 |
| FadeGeneration authority は FadeAccumulator ではなく CrossfadeRuntime が適切 | **妥当** — FadeAccumulator は analysis layer であり実時間を持たない。 | **authority を CrossfadeRuntime（DSP layer）に変更**（P1-9）。FadeAccumulator は generation 参照のみ。 | ✅ v12.0反映 |
| EmergencyDrain best-effort では次回起動時クラッシュリスク — 半端な drain は全くしないより危険 | **妥当** — 一部解放済みオブジェクトが次回起動でクラッシュ。 | **EmergencyDrain must-drain（maxDrainTime=500ms）**（P2-9）。それでも残れば fatal error。 | ✅ v12.0反映 |
| P0-8（SnapshotCoordinator 直接経路除去）は全項目中最優先 | **妥当** — Router 導入後も旧経路が残る状態が最も危険。 | **P0-8 を最優先に昇格**。RetireId 導入より先に実施。PR-1 最初のタスク。 | ✅ v12.0反映 |
| Telemetry 閾値超過時の自動制御不在 — 手動調整では recovery 間に合わず | **妥当** — cache retire の優先順位下げが必要。 | **P0-11 新規**: Telemetry → RetireBackpressurePolicy 自動反映。閾値超過時 cache retire 停止、DSP retire 優先。 | ✅ v12.0反映 |

### レビュー指摘の反映状況（v12.0→v13.0 第12回監査: 実コード整合性照合フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| P0-8 段階移行は危険 — Router と旧経路の共存が retire 順序不一致・telemetry 欠落・RetireRegistry 未通過を発生させる | **妥当** — 段階移行による移行期間は危険。全経路一括切替の方が破綻しにくい。 | **P0-8 一括切替化**: 段階移行方針を撤回。PR-1-0 として全8経路を一括切替。CI gate で逸脱検出。 | ✅ v13.0反映 |
| RetireRegistry key が ObjectAddress+Generation のみ — 長時間稼働でメモリアドレス再利用リスク | **妥当** — DSP A delete → 同アドレスに DSP B 割当で誤検出。Generation 供給源も未定義。 | **key を (ObjectAddress, Generation, ObjectType) に昇格**（P0-10）。Generation 発行元を CrossfadeRuntime に固定。 | ✅ v13.0反映 |
| RetireBackpressurePolicy 単一閾値で chattering — 閾値ちょうど付近で ON/OFF 繰り返し | **妥当** — 実運用で高頻度切替のリスク。 | **二重閾値ヒステリシス追加**（P0-11）。highWatermark=1000, lowWatermark=700。 | ✅ v13.0反映 |
| DeferredRetryQueue 飽和時の最終動作が曖昧 | **妥当** — maxRetryCount/maxRetryAgeMs のみでは Queue 全体飽和時の動作が未定義。 | **maxQueueDepth 追加**（P2-8）。保護レベル（低→高）: Cache drop → Snapshot drop(最終手段) → DSP 絶対不drop。 | ✅ v13.0反映 |
| EmergencyDrain 500ms 固定は危険 — IR 更新長時間セッションで時間不足 / 逆に大量件数処理可能 | **妥当** — 固定時間のみでは両方のケースに対応不可。 | **maxDrainTime + maxDrainItems 先到達制**（P2-9）。500ms OR 100000 items。 | ✅ v13.0反映 |
| Retire Commit Barrier shutdown 時デッドロック — hard barrier が fade 完了待ち中に Audio 停止 | **妥当** — Audio 停止後に fade は進行しない。 | **shutdown 例外追加**（P0-7）。hard barrier に forceFadeComplete() 許可 + barrier timeout(100ms)。Phase3 では bypass。 | ✅ v13.0反映 |
| SnapshotCoordinator の責務が依然過大 — current/target slot/fade/retire/EpochDomain を抱える | **妥当** — P0-8 一括切替後も責務再監査が必要。 | **Phase-B 完了条件に責務監査追加**（P1-7）。実装完了後、各責務の分離状態を確認。 | ✅ v13.0反映 |

### レビュー指摘の反映状況（v13.0→v14.0 第13回監査: 実コード完全照合フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| SnapshotCoordinator EpochDomain 依存が8箇所でない — publish/current/ObservedRuntime/メンバ変数含めると16箇所 | **グレップ/Serena完全特定で確認**: `m_epochDomain->` アクセスが .h で10箇所、.cpp で6箇所の合計16箇所。従来認識の2倍。 | **P0-8 16箇所対応に拡張**。完了条件を `#include "EpochDomain.h"` 削除に設定。付録A テーブルを16行に拡張。 | ✅ v14.0反映 |
| RetireRegistry が永続成長 — IR切替/Preset切替で数十万〜数百万件に肥大化 | **妥当** — retire完了+delete完了+epoch安全なエントリが削除されない。 | **purgeBeforeGeneration() 追加**（P0-10）。3条件充足エントリを定期的にパージ。 | ✅ v14.0反映 |
| DeferredRetryQueue 完全優先順位で Cache retire starvation — 常時DSP retire発生でCacheが永久に処理されない | **妥当** — 完全優先順位は starvation の古典的原因。 | **WFQ (Weighted Fair Queue) 採用**（P2-8）。DSP 8:Snapshot 4:Cache 1 の重み付け。 | ✅ v14.0反映 |
| Backpressure 出口制御のみ — 入口制御がないと backlog 成長速度を抑制できない | **妥当** — 出口制御だけでは queueDepth 回復前に新規 publish が backlog 再増加。 | **PublishAdmissionState 追加**（P0-11）。Normal/Congested/Critical の3段階入口制御。 | ✅ v14.0反映 |
| EmergencyDrain 件数固定値はオブジェクトサイズ差異に対応不可 | **妥当** — IR snapshot と small metadata でサイズが大きく異なる。 | **maxDrainBytes 追加**（P2-9）。500ms OR 100000 items OR 1GB の先到達制。 | ✅ v14.0反映 |
| SnapshotCoordinator 移行完了後も責務肥大化再発リスク | **妥当** — P0-8 一括切替後も保守者が API 追加する可能性。 | **SnapshotCoordinator API 凍結**（P0-12）。public API 一覧固定 + CI gate で追加検出。 | ✅ v14.0反映 |

### レビュー指摘の反映状況（v14.0→v15.0 第14回監査: 長期運用最適化フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| SnapshotCoordinator 完了条件が EpochDomain.h include 削除のみでは不十分 — 保守フェーズで再結合しやすい | **妥当** — include 削除後に `EpochDomain*` メンバや `ObservedRuntime(EpochDomain&)` が残ると再結合経路になる。 | **完了条件を「EpochDomain 型を一切知らない状態」に強化**（P0-8）。EpochDomain*/& 不可、ObservedRuntime(EpochDomain&)不可。CI gate で型参照を監視。 | ✅ v15.0反映 |
| RetireRegistry purge の実行主体が曖昧 — Worker A/B/EmergencyDrain 全員が purge を試みる競合リスク | **妥当** — purge 機能追加だけでは実運用で競合が発生しうる。 | **purge 実行主体を Router owner thread のみに単一化**（P0-10）。他 Worker からの直接 purge 禁止。定期メンテナンスタイマで実行。 | ✅ v15.0反映 |
| WFQ でも DSP 永続高負荷時に Cache が数十分単位で遅延する starvation リスク | **妥当** — WFQ は完全優先より良いが、重み比率だけでは長時間待機を防止できない。 | **WFQ + aging (maxWaitTime=30s) 追加**（P2-8）。30秒超過で Cache retire 強制昇格（weight 一時最大）。OS スケジューラと同様の手法。 | ✅ v15.0反映 |
| PublishAdmissionState が queueDepth 単一指標 — 短時間スパイクに過剰反応 | **妥当** — 「深いが安定」より「急激に増加中」の方が危険。 | **growth rate (depthSlope) 追加**（P0-11）。AdmissionScore = queueDepth + depthSlope。急増時は早期警戒。 | ✅ v15.0反映 |
| Barrier 待機理由の観測指標不足 — 実機障害で「barrier timeout」としか分からない | **妥当** — 4つの待機理由を分解しないと診断不可能。 | **Barrier telemetry 詳細化**（P0-7）。waiting_epoch/waiting_fade/waiting_enqueue/waiting_shutdown を個別記録。 | ✅ v15.0反映 |
| EmergencyDrain の固定値が保守性を低下 — IRサイズや機能追加で変化する | **妥当** — 固定値としての妥当性はあるが、保守性向上のためポリシー化が望ましい。 | **EmergencyDrain ポリシー化**（P2-9）。maxDrainBytes = max(residentMemory*0.1, 1GB) など設定可能に。 | ✅ v15.0反映 |

### レビュー指摘の反映状況（v15.0→v16.0 第15回監査: 長期運用・移行事故耐性フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| 最大リスクは移行期間中の新旧経路共存 — 「残り1箇所」の見逃しが最も危険 | **妥当** — 大規模移行で一番危険なのは「残り1箇所」。新Routerと旧EpochDomain経路の共存期間が事故の温床。 | **移行進捗 CI 追加**（P0-8）。EpochDomain direct call count を20→15→10→5→0 と段階的可視化。CI gate で0になるまでマージ不可。 | ✅ v16.0反映 |
| Router owner thread 停止後に RetireRegistry が残る shutdown ケースの purge 漏れ | **妥当** — stop publish → owner thread stop → EmergencyDrain → shutdown の流れで Registry 残留。 | **forceFinalPurge() 追加**（P0-10）。ShutdownPhase3 で all retire drained 条件のみ実行。 | ✅ v16.0反映 |
| generation ベース purge だけでは一部エントリが長期間残留 — 異常系で肥大化 | **妥当** — generation 依存のみでは極端に古いエントリが残りうる。 | **maxRegistryAge=24h 追加**（P0-10）。時間経過による強制パージ。異常系肥大化防止。 | ✅ v16.0反映 |
| depthSlope 瞬時値はオーディオ系スパイクでノイズが多い — 過剰反応の原因 | **妥当** — オーディオ系は本質的にスパイクが多い。 | **EMA-based depthSlope 採用**（P0-11）。500ms移動平均（α=0.5）。スパイク影響を平滑化。 | ✅ v16.0反映 |
| Router に lifecycle/epoch/fade/publish decision が入り込むリスク — SnapshotRetireFacade と同種の問題 | **妥当** — 長期保守で Router の責務が静かに拡大する。 | **Router 責務 CI ガード追加**（P1-3）。許可: route/queue/telemetry/metrics のみ。禁止: lifecycle/epoch/fade/publish decision。CI でパターンチェック。 | ✅ v16.0反映 |

### レビュー指摘の反映状況（v16.0→v17.0 第16回監査: 実運用最終仕上げフェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| SnapshotCoordinator 除去後も AudioEngine(7)+EQProcessor(5) の EpochDomain 直接経路が残り混在リスク | **妥当** — 現状の完了条件では SnapshotCoordinator のみが対象。AudioEngine/EQProcessor が未処理のまま残る。 | **P0-8 完了条件を全 public retire API の Router 経由に拡張**。28箇所すべてを統合。Phase-A(SnapCoordinator)→Phase-A2(AudioEngine+EQProcessor) の2段階。 | ✅ v17.0反映 |
| Router に state/policy/decision が増殖し便利関数化するリスク — CI ガードだけでは不十分 | **妥当** — コメントや CI では防止限界。インターフェースレベルの制約が必要。 | **P0-1 Router interface 固定**: Allowed(route/enqueue/observer factory) vs Forbidden(state/policy/decision)。Policy系は別クラス。 | ✅ v17.0反映 |
| forceFinalPurge で診断情報消失 — shutdown 異常系の解析不可 | **妥当** — purge 先行により診断情報まで消える。 | **emitFinalRegistrySnapshot() 追加**（P0-10）。forceFinalPurge 前に Registry スナップショットを diagnostic dump。 | ✅ v17.0反映 |
| PublishAdmissionState 閾値固定 — IR大量切替/オフラインレンダリングで適正値変化 | **妥当** — 固定値では運用状況の変化に対応不可。 | **queueCapacity * ratio に変更**（P0-11）。highWatermark=capacity*70%, lowWatermark=capacity*50%。将来的に設定可能。 | ✅ v17.0反映 |
| maxRegistryAge=24h が未完了エントリにも適用されるリスク — 障害解析困難 | **妥当** — 完了済みでないエントリを age purge すると原因特定不能。 | **maxRegistryAge 対象を完了済みエントリのみに限定**（P0-10）。retire完了+delete完了+epoch安全 の3条件必須。 | ✅ v17.0反映 |
| EmergencyDrain 残留時 fatal — 実運用/プラグインではホスト巻き込みクラッシュの方が高コスト | **妥当** — 同一動作では開発時（早期発見）と実運用（安全停止）の要求が矛盾。 | **EmergencyDrain ビルド種別分岐**（P2-9）。Debug: fatal。Release: telemetry + leak quarantine。 | ✅ v17.0反映 |
| 移行完了後の回帰検出が弱い — 0到達後に再追加される危険 | **妥当** — 移行完了後に `m_epochDomain->enqueueRetire` が再追加される可能性。 | **回帰防止CI追加**。0到達後は `grep m_epochDomain\.\(enqueueRetire\|reclaimRetired\)` を常時監視。 | ✅ v17.0反映 |

### レビュー指摘の反映状況（v18.1→v18.2 第18回監査: 実運用破綻耐性最終確認フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| EpochDomain「呼び出し監査」のみでは「所有者監査」が不足 — 型所有(class member/ctor arg/func arg/local static/global singleton)の機械抽出がない | **妥当** — これまでの監査で5→8→16→20→28→32箇所と増え続けた根本原因は「呼び出し」だけ追い「所有」を追っていないこと。 | **所有者監査工程追加**（P0-8）。全 `.h/.cpp` から EpochDomain 型出現を機械抽出し `EpochDomain_OWNERS.md` に出力。完了条件に「所有者一覧が空」を追加。 | ✅ v18.2反映 |
| RefCountedDeferred 未使用テンプレートの放置は将来の回帰点 — 開発者が使った瞬間に旧経路が復活 | **妥当** — dead code の放置は保守フェーズで危険。コメントだけでは不十分。 | **RefCountedDeferred.h に `static_assert(false)` 追加**（P1-2）。コンパイル時に使用を禁止。または Router 版に置換。 | ✅ v18.2反映 |
| AST CI が呼び出し禁止のみで型所有禁止がない — EpochDomain* メンバの新規追加を防止できない | **妥当** — 呼び出し禁止のみでは `EpochDomain*` メンバの追加を検出できない。 | **AST CI に型所有禁止ルール追加**（P1-3）。`EpochDomain*` / `EpochDomain&` / `EpochDomain` メンバ変数の新規追加禁止。 | ✅ v18.2反映 |
| Router が Policy 参照取得APIを提供し始める運用リスク — `router->policy()` / `router->state()` の要求 | **妥当** — Router 唯一入口化後に周辺クラスが Policy 参照を要求し始める古典的パターン。 | **Router interface-only 固定**（P1-3）。Policy 参照取得API禁止。Policy Lane へのアクセスは直接 DI で行う。 | ✅ v18.2反映 |
| ObserverLease の `isRouterAlive()` 依存 — shutdown 順序で Router 破棄後に Observer 残存 | **妥当** — Router 生存確認は生成時の情報で十分。 | **ObserverLease generation snapshot 化**（P0-8）。生成時の generation をスナップショット保持。Router 生存確認ではなく generation 整合性で有効性判断。 | ✅ v18.2反映 |
| RetireRegistrySummary 保存だけで運用で使えない — 「1時間前の状態」を見たい需要に対応不可 | **妥当** — 保存単発では障害解析に不十分。 | **Summary リングバッファ化**（P0-10）。最新100件保存。shutdown/forceFinalPurge/Barrier timeout/EmergencyDrain完了時に保存。shutdown 時に全履歴診断ダンプ。 | ✅ v18.2反映 |
| EmergencyDrain の残留情報が「何個残ったか」だけで「何が残ったか」がない | **妥当** — 種別情報がないと障害解析時間が長期化。 | **remainingObjectTypeHistogram 追加**（P2-9）。DSPRuntime/Snapshot/Cache/EQState の種別カウントを DrainResult に内包。 | ✅ v18.2反映 |

### レビュー指摘の反映状況（v18.2→v18.3 第19回監査: 運用時管理不全防止フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| typedef/using alias 経由で EpochDomain 型が再侵入するリスク — `using EpochRef = EpochDomain&;` で AST CI をバイパス可能 | **妥当** — `EpochDomain*`/`&`/member のみの監視では型エイリアス経由の再侵入を防げない。 | **所有者監査対象拡張**（P0-8）。typedef/using alias/template parameter/trait specialization までカバー。最終目標: EpochDomain 型名が公開APIに出現しない。 | ✅ v18.3反映 |
| Router public API に増殖防止策がない — `router.getMetrics()` / `router.getPolicy()` が静かに追加される | **妥当** — 責務制限だけでは長期保守で API が増え続ける。 | **Router public API 数上限設定**（P1-3）。最大10メソッド。API追加時レビュー必須。CI で public メソッド数閾値監視。 | ✅ v18.3反映 |
| ObserverLease の generation snapshot だけでは障害解析不十分 — 「どのRouterからいつ生成されたか」が分からない | **妥当** — generation 単独では原因特定に不十分。 | **ObserverLease 診断情報強化**（P0-8）。observerId/routerInstanceId/generation/creationTimestamp を保持。 | ✅ v18.3反映 |
| RetireRegistrySummary が統計のみで偏り情報がない — 「Cacheが異常に多い」等を score 化できない | **妥当** — 件数だけでなく「何の偏りか」が障害解析に有用。 | **Summary に偏り情報追加**（P0-10）。topObjectType/topDuplicateType パーセンテージ。 | ✅ v18.3反映 |
| 移行進捗が「数」だけでは死蔵ラッパーが残る — `OldRetireFacade` のような到達不能コードが残存 | **妥当** — direct call = 0 だけでは不十分。保守者が混乱する。 | **到達不能コード監査追加**（P1-10）。参照ゼロクラスを `DEAD_CLASSES.md` に一覧化。移行完了条件に追加。 | ✅ v18.3反映 |
| EmergencyDrain 残留ヒストグラムに経過時間がない — 「どれだけ長く残ったか」が分からない | **妥当** — 件数より経過時間の方が原因特定に役立つ。 | **oldestAge/newestAge 追加**（P2-9）。RemainingObjectTypeHistogram に double 秒で経過時間を追加。 | ✅ v18.3反映 |

### レビュー指摘の反映状況（v17.0→v18.0 第17回監査: 長期保守耐性フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| Router が Policy/State の物理保持者になると Authority Hub 化 — 唯一入口＋唯一判断点＋唯一状態保持点へ回帰 | **妥当** — interface-level 制約だけでは Policy をメンバに持つことまで防げない。コンパイル依存の物理分離が必要。 | **P0-1 Router Policy/State 物理分離**: コンパイル依存 `Router→Policy` 参照のみ可、`Policy→Router` 参照禁止。Router は `class ISRRetireRouter` として Policy/State/Decision をメンバに持てない（コンパイルエラー）。 | ✅ v18.0反映 |
| `router.makeObserver()` の寿命責任が未定義 — Router 停止後に Observer だけ残存する危険 | **妥当** — Observer の寿命が Router のそれと暗黙的に紐づき、shutdown 順序違反で dangling 参照。 | **ObserverToken 化（RAII ベース）**（P0-8）。`router.makeObserver()` → ObserverToken。コピー不可・move のみ許可。Router 破棄時に全 Token 自動無効化。 | ✅ v18.0反映 |
| grep テキスト検索の CI gate は保守性が低い — リファクタリングでシグネチャ変化すると無効化しやすい | **妥当** — テキスト検索は false positive/false negative の両方を持ち、保守者の信頼を失いやすい。 | **CI gate AST ベース昇格**（P1-3）。clang-tidy カスタムルールで AST visitor による直接検出。grep フォールバック併用。clang-tidy ルール案: メンバ変数禁止/stateful分岐禁止/LOC制限/EpochDomain呼び出し禁止。 | ✅ v18.0反映 |
| RetireRegistry purge 後に過去の Registry 状態が復元不可 — 24h前に何が起きたか分からない | **妥当** — エントリ全保存は不要だが、集計だけ残す価値がある。 | **RetireRegistrySummary 定期保存**（P0-10）: entry count / oldest generation / newest generation / duplicate retire count / quarantine count / total retired bytes。purge 前に保存。 | ✅ v18.0反映 |
| EmergencyDrain が「drain した」だけでは障害解析不十分 — どれだけ残ったか・なぜ timeout したかの情報が必要 | **妥当** — ビルド分岐だけでは結果の定量評価ができず、運用中の閾値調整も不可能。 | **DrainResult オブジェクト導入**（P2-9）。struct DrainResult { drainedCount, remainingCount, remainingBytes, timeoutMs, timeoutReason }。EmergencyDrain 完了時 + forceFinalPurge 連動時に生成。 | ✅ v18.0反映 |

### レビュー指摘の反映状況（v18.3→v18.4 第20回監査: 長期運用時管理不全防止フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| P0-8完了条件が「32箇所削除」のみで二段階の区別がない — Retire用途とRCU用途で完了条件を使い分けるべき | **妥当** — enqueueRetire/reclaimRetiredの削除と、advanceEpoch/RCUReaderの取扱いは異なる完了条件が必要。 | **P0-8完了条件二段階化**: Phase-A(Retire用途 EpochDomain=0) / Phase-B(RCU用途許容明確化)。ConvolverProcessorのRCU依存はEBR基盤として許容可能。 | ✅ v18.4反映 |
| Router fast-path 表現が「call-siteから直接Policy Laneへ委譲可能」とPolicy直参照を許容するように読める | **妥当** — 表現が誤解を招く。fast-pathはRouter APIの内部最適化であってPolicy直参照ではない。 | **P0-1 fast-path表現修正**: `router.enqueueDSPFast(...)`としてRouter API内部最適化であることを明記。Policy直参照禁止を明確化。 | ✅ v18.4反映 |
| Router API数上限だけでは増殖防止に不十分 — publishEx/publishAdvanced/publishWithTelemetryのような亜種APIがカテゴリ内で増殖する | **妥当** — 数上限は回避可能。カテゴリそのものを固定する必要がある。 | **P1-3 Router APIカテゴリ固定**: enqueue/publish/observer/telemetryのみ許可。亜種API増殖禁止。 | ✅ v18.4反映 |
| RetireRegistrySummaryの保存契機が異常時のみ — 正常運用中の経過観測に使えない | **妥当** — 障害解析には正常時の定期スナップショットも有用。 | **P0-10 定期スナップショット追加**: 1時間毎のsummarySnapshot()を保存契機として追加。リングバッファ100件。 | ✅ v18.4反映 |
| DeferredRetryQueueのaging昇格が何回発生したか・最高どのレベルまで達したかの統計がない | **妥当** — aging効果の運用評価が不可能。 | **P2-8 aging昇格統計追加**: agingPromotionCount/maxAgingLevelReachedをtelemetryに追加。 | ✅ v18.4反映 |
| DrainResultに残留オブジェクト「1個あたりの年齢」情報がない — 全体のoldest/newestだけでは偏り不明 | **妥当** — 平均年齢と最古オブジェクト年齢で孤立問題・retire漏れを識別。 | **P2-9 DrainResult年齢情報追加**: oldestRetireAge/averageRetireAgeをRemainingObjectTypeHistogramに追加。 | ✅ v18.4反映 |
| 完了後のregression追跡がない — ゼロ状態が静かに崩れても気づけない | **妥当** — 完了条件達成後の状態維持が不可欠。 | **P1-10 regression count追加**: 過去30日のEpochDomain direct call追加回数をCIダッシュボードに表示。0超でマージブロック。 | ✅ v18.4反映 |

### レビュー指摘の反映状況（v18.5→v18.6 第22回監査: 完了条件精密化フェーズ）

| レビュー指摘 | ソース検証結果 | 反映先 | 状態 |
| ------------- | --------------- | -------- | ------ |
| `epochDomain()` public accessor が完了条件の主指標になっていない — public getter/reference/pointer/accessor を別指標化すべき | **妥当** — `Direct calls = 0` だけでは公開アクセサ経由の経路を捕捉できない。 | **P0-8 Public EpochDomain exposure count = 0 追加**: public getter/reference/pointer/accessor wrapper を独立した完了条件に。AudioEngine.h:3225 `epochDomain()` 削除必須。 | ✅ v18.6反映 |
| AudioEngine.Threading.cpp が実質的な移行ボトルネック(13参照集中) — 一括移行ではレビュー漏れを誘発 | **妥当** — publish/current/advanceEpoch/enqueueRetire/reclaimRetired/enterReader/exitReader が集中しており、一括切替は危険。 | **P0-8 AudioEngine.Threading.cpp 独立PR化**: 13参照を独立した移行単位として分割。Phase-Aと並行して移行可能。 | ✅ v18.6反映 |
| advanceEpoch(19)が過小評価 — 特にEQProcessor.Parameters.cpp(10回超)の乱用が将来の保守負債に | **妥当** — advanceEpoch の用途分類なしに Phase-B を開始すると、真に必要なものまで削除しかねない。 | **P1-11 advanceEpoch(20)理由一覧先行作成**: parameter update/snapshot publish/cache rebuild/cleanup に分類。不要な advanceEpoch は削除。 | ✅ v18.6反映 |
| RCUReader が新たな EpochDomain 代理になる危険 — 移行後も RCUReader.domain() が EpochDomain& を返し続ける | **妥当** — Router移行後に「EpochDomain→RCUReader」と名前が変わっただけになるリスク。 | **P1-12 RCUReader代理防止条件追加**: Phase-B完了条件に「RCUReaderがEpochDomain公開代理になっていないこと」を追加。 | ✅ v18.6反映 |
| DeletionQueue/SnapshotRetireManager が監査対象から漏れている — 付録のみで主文脈に未記載 | **妥当** — 放置すると最終段階で依存が残る。 | **P1-13 DeletionQueue/SnapshotRetireManager を Phase-B 正式対象へ昇格**: DeletionQueue.h/cpp(3参照)+SnapshotRetireManager.h(1参照)の EpochDomain依存を Router 経由に置換。 | ✅ v18.6反映 |
| 設計書中に「call-siteから直接Policy Laneへ委譲可能」の旧記述が残っている | **妥当** — v18.4で修正済みだが、全体的な統一が必要。 | **P0-1 fast-path記述完全統一**: 設計書全体の fast-path 記述を「Router API経由のみ許可」に統一。旧記述を全削除。 | ✅ v18.6反映 |
| 124参照のうち「削除対象」と「意図的残存」を区別していない — レビューごとに議論が再発 | **妥当** — 一元管理ファイルで status を可視化する必要あり。 | **P2-11 EpochDomain_REFERENCE_STATUS.md 作成提案**: 全124参照に status(削除/Router置換/RCU残存/コメント/Dead code)を付与。CI gateでstatus未定義の新規参照をブロック。 | ✅ v18.6反映 |

---

## ★ 付録A: EpochDomain 直接呼び出し棚卸し（Phase-D完了時点 v18.13）
> 124参照から大幅削減。全 `m_epochDomain.xxx()` 直接呼び出しは **17件のみ** に減少。

### A-0. Phase-D 移行成果

| カテゴリ | Phase-D前 | Phase-D後 | 削減率 |
|---------|----------|----------|--------|
| SnapshotCoordinator 直接呼び出し | 8 | **0** | 100% |
| AudioEngine 直接呼び出し | 10 | **6** (うち許容4) | 60% |
| EQProcessor 直接呼び出し | 5 | **9** (独自ドメイン仕様) | — |
| RefCountedDeferred 直接呼び出し | 2 | **0** | 100% |
| **合計 m_epochDomain.xxx()** | **124** | **17** | **86%削減** |

### A-1. SnapshotCoordinator → ✅ 全件移行完了 (IEpochProvider 経由)

旧: `m_epochDomain->enqueueRetire/reclaimRetired/publish` (8箇所)
新: `m_epochProvider->publishEpoch/enqueueRetire/tryReclaim`

### A-2. AudioEngine 残存 (6件、うち2件は許容)

| # | ファイル:行 | 呼び出し | 状態 |
|---|------------|----------|------|
| 1 | `CtorDtor.cpp:121` | `advanceEpoch()` | 🔶 **許容** (デストラクタ) |
| 2 | `CtorDtor.cpp:129` | `drainAll()` | 🔶 **許容** (デストラクタ/Router未提供) |
| 3 | `ReleaseResources.cpp:206` | `drainAll()` | 🔶 **許容** (releaseResources/Router未提供) |
| 4 | `ReleaseResources.cpp:213` | `pendingRetireCount()` → Router経由に変更済み | ✅ **移行** |
| 5 | `Publication.cpp:24` | `current()` → `m_retireRouter->currentEpoch()` に変更済み | ✅ **移行** |
| 6 | `Retire.cpp:51,55` | `getMinReaderEpoch()`/`pendingRetireCount()` → Router経由に変更済み | ✅ **移行** |

### A-3. EQProcessor 残存 (9件、全て独自ドメイン)

EQProcessor は固有の `convo::EpochDomain m_epochDomain` (独自インスタンス) を持つため、
これらは AudioEngine の EpochDomain とは別ドメインへの操作。

| # | ファイル:行 | 呼び出し | 状態 |
|---|------------|----------|------|
| 1 | `Core.cpp:37` | `currentEpoch()` | 🔶 **許容** (独自ドメイン) |
| 2 | `Core.cpp:55` | `enqueueRetire()` | ⚠ **未移行** (独自ドメインだが要検討) |
| 3 | `Core.cpp:56` | `currentEpoch()` | 🔶 **許容** (独自ドメイン) |
| 4 | `Core.cpp:73` | `advanceEpoch()` | 🔶 **許容** (独自ドメイン) |
| 5 | `Core.cpp:82` | `currentEpoch()` | 🔶 **許容** (独自ドメイン) |
| 6 | `Core.cpp:91` | `currentEpoch()` | 🔶 **許容** (独自ドメイン) |
| 7 | `Core.cpp:124` | `reclaimRetired()` | ⚠ **未移行** (独自ドメイン) |
| 8 | `Core.cpp:125` | `drainAll()` | 🔶 **許容** (独自ドメイン) |
| 9 | `Core.cpp:126` | `reclaimRetired()` | ⚠ **未移行** (独自ドメイン) |

### A-4. 正当なラッパー経路 (3件)

| # | ファイル | 呼び出し | 状態 |
|---|----------|----------|------|
| W1 | `ISRRuntimePublicationCoordinator.cpp:152` | `domain.enqueueRetire()` | ✅ **正当な単一窓口** |
| W2 | `ISRRetireRouter.h:146` | `epochDomain_->enqueueRetire()` | ✅ **Router内部委譲** |
| W3 | `RefCountedDeferred.h` | — | 🔶 **削除済み** (旧release(EpochDomain&)削除) |

### A-5. 棚卸しサマリー

```text
Phase-D 完了時点の残存 EpochDomain 直接呼び出し: 17件
  AudioEngine (許容):  advanceEpoch(1) + drainAll(2) = 3件
  AudioEngine (移行済): current(1) + pendingRetireCount(2) + getMinReaderEpoch(2) = 5件 ✅
  EQProcessor (独自ドメイン): currentEpoch(4) + enqueueRetire(1) + reclaimRetired(2) + advanceEpoch(1) + drainAll(1) = 9件
  Router内部 (正当なラッパー): 3件

Phase-D 削減実績: 124参照 → 17件 (86%削減)
移行完了ファイル数: SnapshotCoordinator, AudioEngine.Publication, AudioEngine.Retire, RefCountedDeferred
```
  └─ publish(): 4箇所 (h:45,93, cpp:70,91)
  └─ current(): 1箇所 (cpp:36)
  └─ ObservedRuntime コンストラクタ: 1箇所 (h:75)
  └─ メンバ変数 m_epochDomain: 1箇所 (h:151)
  └─ コンストラクタ初期化: 1箇所 (h:33)

AudioEngine 直接 EpochDomain 呼び出し: 7箇所 ← 高優先（enqueueDeferredDeleteNonRtWithResult 他）
  └─ enqueueRetire: 3箇所 (h:3183,3194, Threading.cpp:54)
  └─ reclaimRetired: 4箇所 (h:3191, Threading.cpp:76,87,222)

EQProcessor 直接 EpochDomain 呼び出し: 5箇所 ← 高優先（kMaxRetry=4 の retry ループ含む）
  └─ enqueueRetire: 1箇所 (cpp:60)
  └─ reclaimRetired: 4箇所 (cpp:45,64,119,121)

正当なラッパー: 2箇所 (coordinator 経路 + デッドコード)
非 EpochDomain 経路: 4箇所 (metadata 遷移のみ。EpochDomain 呼ばず)

合計直接 EpochDomain 呼び出し: 20箇所
  └─ SnapshotCoordinator: 16箇所（enqueueRetire:6, reclaimRetired:2, publish:4, current:1, ObservedRuntime構築:1, メンバ変数:1, ctor:1）
  └─ AudioEngine: 7箇所（enqueueRetire:3, reclaimRetired:4）
  └─ EQProcessor: 5箇所（enqueueRetire:1, reclaimRetired:4）
```

### A-6b. v18.10 6ツール総合監査更新: 全 EpochDomain 参照実態（144参照・29ファイル）

**v18.10 6ツール(grep/Serena/CodeGraph/ccc/graphify/semble)再監査による訂正**: 従来認識124参照・27ファイルから**144参照・29ファイル**に上方修正。主な増加要因: ファイル名変更(ReleaseResources→Processing.ReleaseResources等)、ConvolverProcessor.h RCUReader構築行追加、SnapshotRetireManager.h コメント内EpochDomain参照(3→4)、進捗に伴う新規コード追加。

#### 集計

| カテゴリ | 呼び出し回数 | ファイル数 | 備考 |
| :------- | :--------: | :--------: | :---- |
| advanceEpoch | **21** | **11** | EQProcessor.Parameters.cpp 10 + Core.cpp 5 + Coefficients.cpp 1 + AudioEngine.CtorDtor 1 + AudioEngine.Threading 1(wrapper def) + ConvolverProcessor.LoadPipeline 1 + ConvolverProcessor.StateAndUI 1 + EpochDomain.h 1(impl forwarding) |
| reclaimRetired | 8 | 4 | 変更なし |
| drainAll | 3 | 3 | 変更なし |
| enqueueRetire | **7** | **5** | AudioEngine.h(2)+AudioEngine.Threading(1)+EQProcessor.Core(1)+ISRRuntimePublicationCoordinator(1)+RefCountedDeferred(1)+AudioEngine.Commit(1,非EpochDomain) |
| currentEpoch/current | 5 | 3 | 変更なし |
| pendingRetireCount | 3+ | 2 | 変更なし |
| publish | 1(戻)+4(呼) | 2 | 変更なし |
| enterReader/exitReader | 2 | 2 | 変更なし |
| activeReaderCount | 1 | 1 | 変更なし |
| メンバ変数(EpochDomain) | 3 | 3 | AudioEngine, EQProcessor, ConvolverProcessor |
| コンストラクタ引数(EpochDomain&) | 4 | 4 | SnapshotCoordinator, RCUReader, ObservedRuntime, EpochDomainReaderGuard |
| RCUReader(EpochDomain&) | 4 | 4 | AudioEngine, EQProcessor, ConvolverProcessor, RCUReader→自身 |
| include "EpochDomain.h" | 17 | 17 | 全 include 箇所。EpochCore.h 経由の間接 include は別計上 |
| public accessor epochDomain() | 1 | 1 | AudioEngine.h (ANY外部コードがEpochDomain取得可能) |
| EpochDomainReaderGuard | 2(使用)+1(定義) | 3 | AudioEngine.Processing.BlockDouble, AudioEngine.Processing.AudioBlock, ObservedRuntime |
| コメント内EpochDomain | 8+ | 5 | SnapshotRetireManager(3), AudioEngine.h(1), AudioEngine.Init(1), AudioEngine.Globals(1), ConvolverProcessor.Lifecycle(1), EpochCore.h(1) |

#### ファイル別内訳（v18.10訂正版）

| ファイル | 参照数 | 内訳 | 計画v18.9 | 差異 |
| :------- | :----: | :--- | :-------: | :--: |
| EQProcessor.Core.cpp | **18** | enqueueRetire, reclaimRetired x4, advanceEpoch x5, currentEpoch x2, drainAll x2, include, member宣言, RCUReader | 17 | +1 |
| SnapshotCoordinator.h | 16 | enqueueRetire x4, reclaimRetired x2, publish x2, include, ctor, member, ObservedRuntime, deprecated#pragma x3 | 15 | +1 |
| AudioEngine.h | 14 | include, enqueueRetire x3, reclaimRetired, pendingRetireCount x2, epochDomain(), member, RCUReader, コメント | 13 | +1 |
| AudioEngine.Threading.cpp | 13 | publish, current, advanceEpoch, enqueueRetire, activeReaderCount, enterReader, exitReader, reclaimRetired x3, m_coordinator.reclaim x2, pendingRetireCount | 13 | 0 |
| EQProcessor.Parameters.cpp | **11** | advanceEpoch x11 — すべて advanceEpoch() | 10 | +1 |
| SnapshotCoordinator.cpp | 9 | current, enqueueRetire x3, publish x2, deprecated#pragma x3 | 9 | 0 |
| RCUReader.h | 8 | include, ctor, member, domain(), EpochDomain* 全般 | 7 | +1 |
| SnapshotRetireManager.h | **4** | reclaim(EpochDomain&)宣言 x1 + コメント x3 | 1 | +3(コメント) |
| ObservedRuntime.h | **4** | include, ctor, EpochDomainReaderGuard member | 3 | +1 |
| RefCountedDeferred.h | **4** | include, enqueueRetire, currentEpoch | 3 | +1 |
| EQProcessor.h | 3 | include, member, RCUReader | 2 | +1 |
| AudioEngine.CtorDtor.cpp | 3 | m_coordinator(m_epochDomain), advanceEpoch, drainAll | 3 | 0 |
| ConvolverProcessor.h | **3** | include, m_epochDomain member, RCUReader構築 | 2 | +1(RCUReader行) |
| DeletionQueue.cpp | 2 | reclaim(const EpochDomain&), isOlder | 2 | 0 |
| ConvolverProcessor.LoadPipeline.cpp | 2 | advanceEpoch + reference | 1 | +1 |
| EQProcessor.Coefficients.cpp | **2** | advanceEpoch + other ref | 1 | +1 |
| ConvolverProcessor.Lifecycle.cpp | **2** | EpochDomain.h include + コメント | 1 | +1(include行) |
| AudioEngine.Processing.ReleaseResources.cpp | 2 | drainAll, pendingRetireCount | 2 | 0(ファイル名変更) |
| ISRRuntimePublicationCoordinator.cpp | 2 | EpochDomain& 引数, deprecated#pragma | 2 | 0 |
| ISRRuntimePublicationCoordinator.h | **2** | include + EpochDomain& param宣言 | 1 | +1 |
| ConvolverProcessor.StateAndUI.cpp | 2 | advanceEpoch + other ref | 1 | +1 |
| DeletionQueue.h | 2 | include + reclaim(const EpochDomain&) | 1 | +1 |
| EpochCore.h | 1 | includeのみ(空のヘッダ) | 1 | 0 |
| ConvolverProcessor.LoaderThread.cpp | 1 | EpochDomain ref via RCU | 1 | 0 |
| AudioEngine.Processing.BlockDouble.cpp | 1 | EpochDomainReaderGuard(m_epochDomain, ...) | 1 | 0(ファイル名変更) |
| AudioEngine.Processing.AudioBlock.cpp | 1 | EpochDomainReaderGuard(m_epochDomain, ...) | 1 | 0(ファイル名変更) |
| AudioEngine.Init.cpp | 1 | コメントのみ | 1 | 0 |
| AudioEngine.Globals.cpp | 1 | コメントのみ | 1 | 0 |
| AudioEngine.Commit.cpp | 1(非EpochDomain) | retireRuntimeEx_.enqueueRetire — 非EpochDomain経路 | 未計上 | 追加(確認用) |
| **合計** | **144** | **29ファイル** | **124/27** | **+20/+2** |

#### v18.10 発見された重大問題

1. **計画上の「32箇所」は大幅過小評価**: 実際は124参照・27ファイル。enqueueRetire/reclaimRetired のみのカウントでは全体の10%程度しかカバーしていなかった。
2. **`AudioEngine::epochDomain()` public accessor (h:3225)**: `inline convo::EpochDomain& epochDomain() noexcept { return m_epochDomain; }` が EpochDomain を全外部コードに露出。**削除必須**（Router 経由のアクセスに置換）。
3. **`EpochCore.h` (core/EpochCore.h)**: `#include "EpochDomain.h"` のみの空ヘッダ。不要であれば削除。
4. **EQProcessor の advanceEpoch 過多**: Parameters.cpp だけで 11 回の advanceEpoch() 呼び出し。各パラメータ設定メソッドの末尾で全区間 epoch 更新を行っている。Router 移行後はこれらをすべて Router 経由に統一する必要がある。
5. **AudioEngine.Threading.cpp は「正当なラッパー」の最大集積地**: publish/current/advanceEpoch/enqueueRetire/activeReaderCount/enterReader/exitReader/reclaimRetired の全種類をラップしている。Router 移行後はこれら全ラッパーを Router 委譲に置換。
6. **ConvolverProcessor の間接依存**: getRcuProvider()->snapshotRcuEpoch() 経由で AudioEngine→m_epochDomain.publish() への間接パス。直接の EpochDomain 参照ではないが、Router 移行後も ConvolverProcessor が AudioEngine 経由で epoch を取得する経路は残る。
7. **EpochDomainReaderGuard の直接使用**: BlockDouble.cpp と AudioBlock.cpp で EpochDomainReaderGuard(m_epochDomain, ...) を直接構築している。これらは RCUReader 経由に移行する必要がある。

### A-7. 発見された追加問題

1. **`SnapshotCoordinator::reclaim(const EpochDomain&)` 引数 unused（h:102-104）**: 引数で `EpochDomain` を受け取るが、ボディでは `m_epochDomain->reclaimRetired()` とメンバポインタで直接呼んでいる。引数が完全に無視されている。
2. **SnapshotCoordinator のデストラクタで enqueueRetire → 即 reclaimRetired（h:54-68）**: 2回の enqueueRetire 直後に同じスレッド内で reclaimRetired を呼んでいる。enqueue の意味が減殺されている（enqueue したものを即座に回収しようとしている）。
3. **EQProcessor の retry ループ（2系統）**: Coordinator 経由と直接経路の2系統が存在し、どちらも同じ `kMaxRetry=4` パターン。両方とも `m_epochDomain.reclaimRetired()` と `m_epochDomain.advanceEpoch()` を直接呼んでいる。

### A-8. P0-8（SnapshotCoordinator 直接経路除去）一括切替タスク

**方針**: 段階移行は行わない。PR-1-0 として全8経路を一括で ISRRetireRouter に切替。理由: Router 経路と旧 EpochDomain 経路の共存期間が長いと retire 順序不一致・telemetry 欠落・RetireRegistry 未通過が発生しやすい。一括切替により移行期間ゼロ。

### PR-1-0: 一括切替（P0-8 実装 — 全16箇所の EpochDomain 依存除**

完了条件: SnapshotCoordinator が EpochDomain 型を一切知らない状態。

- ❌ `#include "EpochDomain.h"` なし
- ❌ `EpochDomain*` メンバ変数なし
- ❌ `EpochDomain&` 関数引数なし
- ❌ `ObservedRuntime(EpochDomain&)` コンストラクタ呼び出しなし
- ❌ **ConvolverProcessor も同条件（v18.1）**: `ConvolverProcessor.h` の `m_epochDomain` メンバ + `RCUReader` + 2箇所の `advanceEpoch()`。
- ❌ **`RefCountedDeferred.h` の EpochDomain 依存削除（v18.1/v18.2）**: `static_assert(false)` または Router 版への置換。
- ❌ **所有者監査（v18.2/v18.3拡張）**: 全 `.h/.cpp` から EpochDomain 型出現（class member / ctor arg / func arg / local static / global singleton / **typedef / using alias / template parameter / trait specialization**）を機械抽出し `EpochDomain_OWNERS.md` に出力。完了条件に「所有者一覧が空であること」を追加。**最終目標: EpochDomain型名が公開APIに出現しない。**
- ❌ **到達不能コード監査（v18.3新規）**: Router 移行完了後に EpochDomain ラッパー/ブリッジ/ファサードクラスの参照カウントを取得。参照ゼロのクラスを `DEAD_CLASSES.md` に一覧化。
- ✅ CI gate: `grep -rn 'EpochDomain' src/core/SnapshotCoordinator.*` が空であること
- ✅ CI gate（全経路）: `grep -rn 'm_epochDomain\.\(enqueueRetire\|reclaimRetired\)' src/` が空であること
- ✅ CI gate（ConvolverProcessor）: `grep -rn 'EpochDomain' src/convolver/ConvolverProcessor.*` が空であること
- ✅ CI gate（型所有禁止 v18.2/v18.3）: AST CI で `EpochDomain*` / `EpochDomain&` / `EpochDomain` メンバ変数 + **typedef/using alias/template parameter/trait specialization 経由の型出現**を禁止
- ✅ CI gate（Router 増殖防止 v18.3/v18.4）: ISRRetireRouter の public メソッド数上限チェック + API カテゴリ固定（enqueue/publish/observer/telemetry のみ）
- ✅ 移行進捗CI: EpochDomain direct call count 段階的可視化（32→24→16→8→0）+ 所有者一覧段階的削減 + 到達不能コード監査 + regression count

**移行進捗指標（v18.12修正: 許容subtype + advanceEpoch呼出元 + Router責務数 + runtime telemetry + DeletionQueue fate + コメント監査 + 依存方向監査）**:

| Phase | Direct calls | Indirect (subtype) | 到達可能 クラス | advanceEpoch (呼出元) | Router 責務数 | Runtime telemetry | DeletionQueue fate | コメント 監査 | 依存方向 | status+reason | 状態 |
| ------- | :-----------: | :--------------: | :-------------: | :----------------: | :-----------: | :---------------: | :---------------: | :---------: | :-----: | :----------: | :----: |
| 現在 | 13 | 22+(未分類) | 多数 | 21(多数呼出元) | 未監査 | 未計測 | 未定義 | 8+未監査 | 未監査 | 未記入 | 未着手 |
| Phase-A | 13→0 | — | — | — | — | — | — | — | — | 削除+reason | 最優先 |
| AE.Threading 3PR | — | — | — | — | — | — | Keep/Wrap 確定 | 対象 | Reader←Pub←Retire | 全PR reason | 要監査 |
| Phase-B | 0 | permit 15/temp 1/禁止0/移行0 | 2 | 1(RPOのみ) | 3系統(Retire/Pub/Reader) | advanceEpoch/sec <閾値 | Delete/Keep/Wrap/Merge確定 | 実態一致 | Reader←Pub←Retire | 全124status+reason | 要完了 |
| 完了 | 0 | permit15/temp0 | 2固定 | 1固定 | 3系統固定 | 常時監視 | 全確定 | 定期監査 | CI自動監査 | ✅ CI gate通過 | ✅ |

| # | ファイル | 行 | 現在のコード | 変更後 | 区分 |
| --- | --------- | ---- | ------------ | -------- | ------ |
| 1 | SnapshotCoordinator.h | 33 | `m_epochDomain(&epochDomain)` (ctor init) | `m_retireRouter(router)` | メンバ変数置換 |
| 2 | SnapshotCoordinator.h | 45 | `m_epochDomain->publish()` (dtor) | `m_retireRouter.publishEpoch()` | publish委譲 |
| 3 | SnapshotCoordinator.h | 54 | `m_epochDomain->enqueueRetire(snap,...)` (dtor) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |
| 4 | SnapshotCoordinator.h | 64 | `m_epochDomain->enqueueRetire(snap,...)` (dtor) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |
| 5 | SnapshotCoordinator.h | 68 | `m_epochDomain->reclaimRetired()` (dtor) | Router委譲（Routerが寿命管理） | reclaim委譲 |
| 6 | SnapshotCoordinator.h | 75 | `ObservedRuntime observed(*m_epochDomain, readerIndex)` | `m_retireRouter.makeObserver(readerIndex)` | ObservedRuntime生成委譲 |
| 7 | SnapshotCoordinator.h | 93 | `m_epochDomain->publish()` (switchImmediate) | `m_retireRouter.publishEpoch()` | publish委譲 |
| 8 | SnapshotCoordinator.h | 96 | `m_epochDomain->enqueueRetire(oldSnap,...)` (switchImmediate) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |
| 9 | SnapshotCoordinator.h | 104 | `m_epochDomain->reclaimRetired()` (reclaim) | Router委譲（引数バグ修正） | reclaim委譲 |
| 10 | SnapshotCoordinator.h | 151 | `EpochDomain* m_epochDomain;` (member) | `ISRRetireRouter& m_retireRouter;` または削除 | メンバ変数置換 |
| 11 | SnapshotCoordinator.cpp | 36 | `m_epochDomain->current()` (startFade) | `m_retireRouter.currentEpoch()` | current委譲 |
| 12 | SnapshotCoordinator.cpp | 39 | `m_epochDomain->enqueueRetire(oldTarget,...)` (startFade) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |
| 13 | SnapshotCoordinator.cpp | 70 | `m_epochDomain->publish()` (resetFadeState) | `m_retireRouter.publishEpoch()` | publish委譲 |
| 14 | SnapshotCoordinator.cpp | 73 | `m_epochDomain->enqueueRetire(target,...)` (resetFadeState) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |
| 15 | SnapshotCoordinator.cpp | 91 | `m_epochDomain->publish()` (completeFade) | `m_retireRouter.publishEpoch()` | publish委譲 |
| 16 | SnapshotCoordinator.cpp | 96 | `m_epochDomain->enqueueRetire(old,...)` (completeFade) | `m_retireRouter.enqueue(kSnapshot,...)` | enqueue委譲 |

**注意**: 一括切替を行う前に ISRRetireRouter クラスと最低限の Policy Lane（Snapshot Retire Policy）が実装済みであること。切替前の CI gate: `grep -rn 'm_epochDomain->enqueueRetire\|m_epochDomain->reclaimRetired' src/core/SnapshotCoordinator.*` が空であることを確認。

---

## ★ 付録B: EpochDomain 間接依存チェーン完全追跡（v18.8 6ツール併用検証）

### B-0. 検出方法

- **grep/Select-String**: AudioEngine ラッパーメソッドの全外部呼び出し元を抽出（snapshotRcuEpoch/markRetireEpoch/currentRetireEpoch/advanceRetireEpoch/enterRcuReader/exitRcuReader/enqueueRetireEpochBounded/activeEpochObserverCount + enterGlobalReader/exitGlobalReader/getRcuProvider）
- **Serena MCP** (`find_referencing_symbols`): `AudioEngine[1]/snapshotRcuEpoch` / `AudioEngine[1]/enterRcuReader` / `AudioEngine[1]/advanceRetireEpoch` の全参照元をシンボルレベルで特定（Lifecycle.cpp 4箇所、StateAndUI.cpp 2箇所、Runtime.cpp 2箇所、RuntimePublicationOrchestrator 1箇所）
- **CodeGraph MCP** (`find_callers`): `AudioEngine::snapshotRcuEpoch` の呼び出し元を5件検出（`unresolved-target-fallback` 解決で C++ メソッド解決が部分的不完全）
- **ccc** (cocoindex-code, `$env:PYTHONUTF8="1"` を前置): `ccc search "snapshotRcuEpoch OR enterRcuReader OR advanceRetireEpoch OR enterGlobalReader" --lang cpp` で SnapshotCoordinator.h:82-99 の snapshotRcuEpoch呼び出しコンテキストを検出（日本語ロケールで文字化けあり）
- **semble**: `semble search "snapshotRcuEpoch OR enterRcuReader OR advanceRetireEpoch OR enterGlobalReader" src` で5件の高スコアチャンクを検出（AudioEngine.Threading.cpp:50-79 が最高スコア 0.019）
- **graphify** (DeepSeek backend, 11408 nodes, BFS depth=2): `graphify query "snapshotRcuEpoch/markRetireEpoch/exitRcuReader indirect chains" --budget 4000` で21ノードを検出。snapshotRcuEpoch→currentRetireEpoch の呼び出し連鎖を確認

### 間接依存の全体構造

```text
[External Caller] → [AudioEngine RCU Wrapper] → [m_epochDomain.Method()]
                        (8 wrappers)
```

**委譲の深さ**: 2レベル（外部呼び出し元 → AudioEngineラッパー → EpochDomainメソッド）

### B-1. AudioEngine RCU ラッパー完全一覧（AudioEngine.Threading.cpp:30-71）

AudioEngine は EpochDomain の全主要メソッドに対して public ラッパーを提供している。これらは Router 移行後も委譲先が変わるのみで、ラッパー自体の存在意義は継続する（API安定性のため）。

| ラッパーメソッド | ファイル:行 | 委譲先 | 役割 |
| :--------------- | :---------: | :----- | :--- |
| `snapshotRcuEpoch()` → `currentRetireEpoch()` | AudioEngine.h:775 | `m_epochDomain.current()` | 現epoch取得（SnapshotCoordinator/ConvolverProcessorが使用） |
| `markRetireEpoch()` | AudioEngine.h:776 | `m_epochDomain.publish()` | epoch発行（enqueueDeferredDeleteNonRtWithResult内で使用） |
| `currentRetireEpoch()` | AudioEngine.Threading.cpp:38 | `m_epochDomain.current()` | 現epoch直接取得 |
| `advanceRetireEpoch()` | AudioEngine.Threading.cpp:44 | `m_epochDomain.advanceEpoch()` | epoch進行（RuntimePublicationOrchestratorが使用） |
| `enterRcuReader(idx)` | AudioEngine.Threading.cpp:50 | `m_epochDomain.enterReader(idx)` | RCU reader入場（ConvolverProcessor enterGlobalReaderから） |
| `exitRcuReader(idx)` | AudioEngine.Threading.cpp:51 | `m_epochDomain.exitReader(idx)` | RCU reader退場（ConvolverProcessor exitGlobalReaderから） |
| `enqueueRetireEpochBounded(p,d,e)` | AudioEngine.Threading.cpp:54 | `m_epochDomain.enqueueRetire(p,d,e)` | retire enqueue（`[[deprecated]]` wrapper） |
| `activeEpochObserverCount()` | AudioEngine.Threading.cpp:70 | `m_epochDomain.activeReaderCount()` | reader数取得（診断/テレメトリ用） |

### B-2. 間接依存チェーン一覧（5系統）

#### チェーンA: ConvolverProcessor → getRcuProvider() → snapshotRcuEpoch() → m_epochDomain.current()

**経路**: `ConvolverProcessor::getRcuProvider()` → `AudioEngine::snapshotRcuEpoch()` → `AudioEngine::currentRetireEpoch()` → `m_epochDomain.current()`

**重要**: このチェーンは EpochDomain の `current()`（読取専用）を間接呼び出しする。`snapshotRcuEpoch` というメソッド名だが実際の動作は epoch の読取のみ（publish ではない）。**Router 移行後もこの経路は EpochDomain の read-only API として残りうる**（Phase-B の「許容」カテゴリ）。

| # | 呼び出し元 | ファイル:行 | コンテキスト |
| :- | :--------- | :---------: | :----------- |
| A1 | `ConvolverProcessor::updateIRState()` | Lifecycle.cpp:36 | IR更新時のgeneration取得 |
| A2 | `ConvolverProcessor::prepareToPlay()` | Lifecycle.cpp:242 | 再生準備時のepoch生成 |
| A3 | `ConvolverProcessor::releaseResources()` | Lifecycle.cpp:391 | リソース解放時のepoch生成 |
| A4 | `ConvolverProcessor::syncStateFrom()` | StateAndUI.cpp:419 | 状態同期時のepoch取得 |
| A5 | `ConvolverProcessor::shareConvolutionEngineFrom()` | StateAndUI.cpp:441 | エンジン共有時のepoch取得 |
| A6 | `ConvolverProcessor::LoaderThread::run()` | LoaderThread.cpp:39 | スレッド起動時のAffinityManager取得（epoch非使用） |
| A7 | `ConvolverProcessor::setTargetUpgradeFFTSize()` | LoadPipeline.cpp:135-136 | FFTアップグレード時のAffinityManager取得（epoch非使用） |

**注**: A6/A7 は `getRcuProvider()` の戻り値を AffinityManager の取得にのみ使用しており、EpochDomain には到達しない。安全性確認のため記載。

**合計**: **5箇所**（A1-A5）が `snapshotRcuEpoch()` 経由で EpochDomain::current() に間接到達。A6-A7 は EpochDomain 非到達。

#### チェーンB: ConvolverProcessor → enterGlobalReader/exitGlobalReader → enterRcuReader/exitRcuReader → m_epochDomain.enterReader()/exitReader()

**経路**: `ConvolverProcessor::enterGlobalReader(idx)` → `AudioEngine::enterRcuReader(idx)` → `m_epochDomain.enterReader(idx)`
`ConvolverProcessor::exitGlobalReader(idx)` → `AudioEngine::exitRcuReader(idx)` → `m_epochDomain.exitReader(idx)`

GlobalGuard RAII パターンで使用。コンストラクタで enter、デストラクタで exit を自動呼び出し。readerIndex は呼び出し箇所により 2 または 3。

| # | ファイル:行 | readerIndex | GlobalGuard コンテキスト |
| :- | :--------- | :---------: | :---------------------- |
| B1 | Runtime.cpp:66-67 | 3 | Audio thread の処理ループ内。最も頻繁に出入り |
| B2 | Lifecycle.cpp:178-179 | 2 | prepareToPlay 設定中 |
| B3 | Lifecycle.cpp:422-423 | 2 | releaseResources 解放中 |
| B4 | StateAndUI.cpp:429-430 | 2 | shareConvolutionEngineFrom 同期中 |
| B5 | StateAndUI.cpp:689-690 | 2 | syncStateFrom 同期中 |

**合計**: **5組10回**（enter + exit で各5回 = 10回の EpochDomain reader 操作）。enterReader/exitReader の間接呼び出しとして最大の頻度。

**注意**: Runtime.cpp:66-67 の readerIndex=3 は Audio Thread 上で動作。enterReader/exitReader 自体は lock-free atomic 操作のため RT 違反ではないが、**Router 移行後もこの操作は EpochDomain のコア機能（EBR reader 管理）として残り続ける**。Phase-B の「許容」カテゴリ。

#### チェーンC: RuntimePublicationOrchestrator → engine_.advanceRetireEpoch() → m_epochDomain.advanceEpoch()

**経路**: `RuntimePublicationOrchestrator::commitPublish()` → `engine_.advanceRetireEpoch()` → `m_epochDomain.advanceEpoch()`

| # | 呼び出し元 | ファイル:行 | コンテキスト |
| :- | :--------- | :---------: | :----------- |
| C1 | `commitPublish()` 成功後 | RuntimePublicationOrchestrator.cpp:93 | publish 成功確認後に epoch を進める。コメントに「publish 後に epoch を進める。advanceRetireEpoch は retire queue の drain を行う」と明記 |

**合計**: **1箇所**。ただしこの1回の advanceEpoch 呼び出しは EBR 機構の根幹（publish→epoch advance→retire可能化）であり、**Router 移行後も EpochDomain のコア機能として残る**。Phase-B の「許容」カテゴリ。

**考察**: この呼び出しは advanceEpoch(20) の一部として既にカウント済み（AudioEngine ラッパー経路であり、直接カウントとは別管理）。v18.5 の直接カウント(19) + v18.7 のAudioEngine.CtorDtor.cpp:115(1) = 20 に含まれる。

#### チェーンD: AudioEngine → enqueueDeferredDeleteNonRtWithResult → markRetireEpoch → m_epochDomain.publish()

**経路**: `AudioEngine::enqueueDeferredDeleteNonRtWithResult()` → `AudioEngine::markRetireEpoch()` → `m_epochDomain.publish()`

このチェーンは AudioEngine の内部実装（非公開ラッパー経由）であり、外部からの呼び出しは `enqueueDeferredDeleteNonRtWithResult` 経由で行われる。

| # | 呼び出し元 | ファイル:行 | コンテキスト |
| :- | :--------- | :---------: | :----------- |
| D1 | `AudioEngine::enqueueDeferredDeleteNonRt()` (inline) | AudioEngine.h:3177 | `markRetireEpoch()` → `m_epochDomain.publish()` を呼ぶ。enqueue 前に epoch 発行 |
| D2 | `ConvolverProcessor::updateIRState()` | Lifecycle.cpp:51-54 | `provider->enqueueDeferredDeleteNonRt(oldState, deleter)` 経由 |
| D3 | `ConvolverProcessor::StereoConvolver::retireStereoConvolver()` | Lifecycle.cpp:74 | `provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver)` 経由 |
| D4 | `EQProcessor` (内部) | EQProcessor.Core.cpp | `owner.enqueueDeferredDeleteNonRt(cache/map/old, deleter)` 経由（3箇所） |
| D5 | `AudioEngine::retireDSP()` 内部 | AudioEngine.h:3244 | `enqueueDeferredDeleteNonRtWithResult(dsp, ...)` 経由 |

**合計**: `markRetireEpoch()` → `m_epochDomain.publish()` への間接呼び出しは **6箇所以上**。この経路は EpochDomain の直接カウント（AudioEngine.h:3183-3194 の enqueueRetire + reclaimRetired）とは別の publish 呼び出しであり、**完全重複**しているわけではない。

**重要**: enqueue → publish の順序は EBR の正しい使い方（retire前にepochを発行し、reclaim時にそのepochが安全か確認する）に従っている。Router 移行後も publish 操作自体は必要であり、委譲先が Router に変わるのみ。

#### チェーンE: RCUReader/ObservedRuntime → EpochDomain& 直接参照

**経路**: `RCUReader::domain()` → EpochDomain& 返却 → 任意のメソッド呼び出し

| # | ファイル:行 | 内容 |
| :- | :--------- | :--- |
| E1 | RCUReader.h:11 | `EpochDomain* domain_;` メンバ変数 |
| E2 | RCUReader.h:14 | `explicit RCUReader(EpochDomain& d) : domain_(&d) {}` コンストラクタ |
| E3 | RCUReader.h:17 | `EpochDomain& domain() noexcept { return *domain_; }` public accessor |
| E4 | AudioEngine.h:3382 | `RCUReader reader_{m_epochDomain};` AudioEngine の reader メンバ |
| E5 | EQProcessor.h | `RCUReader` メンバ（同様） |
| E6 | ConvolverProcessor.h:1151 | `RCUReader runtimeReader_{m_epochDomain};` ConvolverProcessor の reader メンバ |
| E7 | ObservedRuntime.h:17 | `EpochDomainReaderGuard guard_(domain, readerIndex);` — EpochDomain& を受け取る |

**重要**: RCUReader は AudioEngine/EQProcessor/ConvolverProcessor にそれぞれ1つずつ存在し、それぞれ `domain()` 経由で EpochDomain& を返す。**Phase-B の完了条件「RCUReaderがEpochDomain公開代理になっていないこと」に該当**。Router 移行後は RCUReader::domain() が EpochDomain& を直接返さないようにする必要がある。

### B-3. enterGlobalReader/exitGlobalReader 実装の詳細

ConvolverProcessor の `enterGlobalReader()` / `exitGlobalReader()` は以下のように実装されている：

**ConvolverProcessor.Runtime.cpp:157-166** (確認済み):

```cpp
void ConvolverProcessor::enterGlobalReader(int readerIndex) const noexcept
{
    if (auto* provider = getRcuProvider(); provider != nullptr)
        provider->enterRcuReader(readerIndex);
}

void ConvolverProcessor::exitGlobalReader(int readerIndex) const noexcept
{
    if (auto* provider = getRcuProvider(); provider != nullptr)
        provider->exitRcuReader(readerIndex);
}
```

**ConvolverProcessor.h:269** (宣言):

```cpp
void enterGlobalReader(int readerIndex) const noexcept;
void exitGlobalReader(int /*readerIndex*/) const noexcept;
```

### B-4. 間接依存サマリー

| チェーン | 経路 | 間接呼び出し回数 | EpochDomainメソッド | Phase-B許容 | 備考 |
| :------- | :--- | :--------------: | :----------------- | :---------- | :--- |
| A | ConvolverProcessor→snapshotRcuEpoch→current→EpochDomain::current() | 5 | `current()` (read) | 許容 | 読取専用。Router移行後も同様のepoch読取API必要 |
| B | ConvolverProcessor→enterGlobalReader→enterRcuReader→EpochDomain::enterReader/exitReader | 5組10回 | `enterReader()`/`exitReader()` | 許容 | EBR基盤。lock-free atomic |
| C | RuntimePublicationOrchestrator→advanceRetireEpoch→EpochDomain::advanceEpoch() | 1 | `advanceEpoch()` (write) | 許容（基盤操作） | EBR epoch進行。Router移行後も類似API必要 |
| D | AudioEngine→enqueueDeferredDeleteNonRt→markRetireEpoch→EpochDomain::publish() | 6+ | `publish()` (write) | Router委譲 | 移行後はRouter::publishEpoch()へ |
| E | RCUReader::domain()→EpochDomain&直接参照 | 3(reader)×任意 | 全メソッド | 要対応 | Phase-B完了条件「RCUReaderがEpochDomain公開代理禁止」対象 |

**間接依存総数**: **5系統・22+回**（チェーンA 5 + チェーンB 10 + チェーンC 1 + チェーンD 6+ = 22以上の EpochDomain 間接呼び出し）

### B-5. 間接依存の Phase-A/Phase-B への影響（v18.9: 間接依存完了条件を統合）

| Phase | 影響 | 対応方針 |
| :---- | :--- | :------- |
| Phase-A (enqueueRetire/reclaimRetired) | **影響なし** — チェーンA〜C/Eは enqueueRetire/reclaimRetired を呼ばない。チェーンDの publish は retire 経路ではない | Phase-A は直接依存13箇所の除去に集中。間接経路はブロッカーにならない |
| Phase-B (advanceEpoch/current/publish/RCUReader/accessor削除) | **要対応項目2点**: (1) RCUReader::domain() が EpochDomain& を返す問題（チェーンE）— AudioEngine/EQProcessor/ConvolverProcessor の3箇所。(2) AudioEngine ラッパー8個の委譲先を Router に変更（チェーンA〜Dの全ラッパー） | (1) RCUReader を Router 内部実装に変更し、`domain()` が Router 経由の epoch 情報を返すようにする。(2) 8個のラッパーは「呼び出し側インターフェース不変、委譲先のみ Router に変更」で対応 |
| 直接依存(124) + 間接依存(22+) | **合計: 146+の EpochDomain 到達経路** | 内訳: 直接124 + 間接22+。間接経路の大部分(チェーンA/B/C/E)は Phase-B で「許容（Router委譲後も類似機能継続）」に分類される。チェーンDのみ Phase-B で委譲先変更が必要 |

### B-5b. v18.10 追加: 間接依存完了条件（P0-13の具体的内訳）— 関数単位で固定化

Phase-B終了時における間接依存の許容/禁止/要移行を以下に確定する。**「advanceEpoch系は許容」ではなく、呼び出し元+関数の組で固定化する。**

**許容（17回）** — 呼び出し元+関数の組で固定:

- チェーンA(5回, 呼び出し元固定): `ConvolverProcessor::updateIRState()` / `prepareToPlay()` / `releaseResources()` / `syncStateFrom()` / `shareConvolutionEngineFrom()` の5メソッドのみ → `m_epochDomain.current()` 読取。**これら以外からの current() 読取は禁止。**
- チェーンB(10回, 呼び出し元固定): `ConvolverProcessor` の GlobalGuard 5組（Runtime.cpp readerIndex=3, Lifecycle.cpp readerIndex=2 x2, StateAndUI.cpp readerIndex=2 x2）のみ → `enterReader/exitReader`。**これら以外からの enterReader/exitReader 呼び出しは禁止。**
- チェーンC(1回, 呼び出し元固定): `RuntimePublicationOrchestrator::commitPublish()` からのみ → `advanceEpoch`。**将来の `advanceEpochIfNeeded()` / `advanceEpochForCleanup()` / `advanceEpochForCommit()` 等の増殖を禁止。**
- チェーンD(publish移行後許容): `AudioEngine::enqueueDeferredDeleteNonRt()` からの `markRetireEpoch()` → `publish`。Router移行後は `Router::publishEpoch()` に委譲。

**禁止（3箇所 → 0に削減）** — P0-14/P1-18対象:

- チェーンE(3箇所): `RCUReader::domain()` → EpochDomain&。Phase-B終了時までに完全廃止。
- **P1-18追加**: `EpochDomainReaderGuard(m_epochDomain, ...)` 3箇所（BlockDouble.cpp, AudioBlock.cpp, ObservedRuntime.h）も禁止対象。RCUReader限定APIに置換。

**要移行（6+回）** — Phase-B対象:

- チェーンD(6+回): enqueueDeferredDeleteNonRt→markRetireEpoch→publish — Router::publishEpoch()へ委譲先変更。AudioEngine.h:3177の markRetireEpoch 呼び出しを含む全経路。

**完了条件**: `Indirect EpochDomain dependency count = 許容(permanent) 15 / 許容(temporary) 1 / 禁止0 / 要移行0`

**v18.11追加: EpochDomain到達可能クラス数 = 2**: Phase-B完了時点で EpochDomain へ到達可能なクラスは `Router` (ISRRetireRouter) と `RuntimePublicationOrchestrator` の2個のみ。CI gate: 3以上でアラート。

**v18.12追加: advanceEpoch呼び出し元数 = 1**: `RuntimePublicationOrchestrator::commitPublish()` のみ。CI gate: 許可1以外からの呼び出しをブロック。許容(temporary) = 0 をPhase-B完了条件とする。

**CI gate**: 許容対象を関数単位で固定化した間接依存数が閾値を超えた場合Phase-B完了未達。許容関数リストを `EpochDomain_ALLOWED_INDIRECT.md` に固定化し、変更には全承認必須。新たな間接経路追加をブロック。

### B-6. ツール別検出能力比較（間接依存）

| ツール | 検出能力 | 制約 |
| :----- | :------- | :--- |
| **grep/Select-String** | ✅ 全ラッパーメソッド名による完全網羅 | メソッド名を列挙する必要あり。`getRcuProvider()` のような中間ブリッジは別途追跡 |
| **Serena MCP** | ✅ `find_referencing_symbols` で全参照元を symbol レベルで特定（Lifecycle.cpp 4, StateAndUI 2, Runtime.cpp 2, RuntimePublicationOrchestrator 1） | 呼び出しチェーンの深さ追跡は手動。C++ overload resolution 未サポート |
| **CodeGraph MCP** | ⚠ `find_callers` で 5件検出 → 全て `unresolved-target-fallback` | C++ メソッド解決が不完全。entity 抽出で EpochDomain メソッドを個別 node 化できず |
| **ccc** (cocoindex-code) | ⚠ 日本語ロケールで文字化け。`$env:PYTHONUTF8="1"` 必須 | 1回の検索で返される結果が限定的。大規模クエリで不完全 |
| **semble** | ⚠ 5件の高スコアチャンクを返す（AudioEngine.Threading.cpp が最高 0.019） | スコア閾値以下のチャンクは返さないため完全網羅は困難 |
| **graphify** (DeepSeek BFS depth=2) | ⚠ 21ノード検出（snapshotRcuEpoch→currentRetireEpoch chain 確認） | BFS depth=2 では EpochDomain の全メソッド呼び出しを網羅できない。full rebuild + より深い depth が必要 |

**総評**: 間接依存の完全追跡には grep/Serena が最も信頼性が高く、CodeGraph/ccc/semble/graphify は補完的役割。Serena の `find_referencing_symbols` が C++ クラスメソッドの参照元特定に最も有効。graphify は BFS 制約のため深いチェーン追跡には不十分だが、ノード間の関係性の俯瞰には有用。
