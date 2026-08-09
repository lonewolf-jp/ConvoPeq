# REPAIR_PLAN2 検証結果ダッシュボード（REPAIR_PLAN2-dash）

**作成日:** 2026-08-09
**対象:** `doc/work88/REPAIR_PLAN2.md`（2026-08-08 更新版, 1309行）
**検証方法:** 実コード照合（六次レビュー）・semble/cocoindex/graphify/serena/AiDex による検索・構造検証
**目的:** 新規バグ・既存バグ・修正漏れ・未実装・改善提案・残余リスクを整理し、Phase 実装前のアクションを明確化する。

---

## 1. 修正済みバグ（六次レビューで発見・修正・検証済み）

### 1.1 🔴 RetireQuarantineStore::drain() / drainAllUnsafe() の mutex 保持中 deleter 呼び出し
- **種別:** 新規バグ（実装契約違反）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/RetireQuarantineStore.h:98-130`
- **問題:** 三次レビュー指示（「lock → safe entries 抽出 → unlock → deleter」）に違反し、`std::lock_guard` 内で `e.deleter(e.ptr)` を実行。deleter が再entrant（別の quarantine/retire を呼ぶ）とデッドロック。
- **修正:** lock 内で safe エントリを抽出（`pendingPtrs`/`pendingDeleters`）→ unlock 後に deleter 実行。`drain()` と `drainAllUnsafe()` の両方を修正。

### 1.2 🔴 retireDSPHandleForRuntime の shutdownReclaim 残存（reclaim→enqueue 逆転）
- **種別:** 既存バグ（計画書 Phase 3 の修正漏れ）
- **状態:** ✅ runtime 経路は修正済み
- **根拠:** `src/audioengine/AudioEngine.h:4193-4222`
- **問題:** `dspHandleRuntime_.shutdownReclaim(handle)`（epoch ゲートなしの即時 reclaim）が runtime 経路に残存。reclaim が enqueue より先に実行。
- **修正:** `requestReclaimHandle()` 共通ヘルパーを新設し、`retireDSPHandleForRuntime` と `retireByHandle` の両方に適用。epoch 安全確認（`currentEpoch() < minReaderEpoch()`）→ requestReclaim（retire→waitReaders→reclaim）or 保留リスト。
- **残存:** `shutdownReclaim` は shutdown 専用の正当用途のみ（AudioEngine.h:2027 CacheMap::dtor / ReleaseResources.cpp:415,420）。

### 1.3 🔴 retireByHandle（Observe 経路）が requestReclaim を呼ばず slot リーク
- **種別:** 既存バグ（Observe 経路の slot リーク）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/DSPLifetimeManager.cpp:84-90`
- **問題:** `ObserveIntentHandler`/`drainObserveDeferred` から呼ばれる `retireByHandle` は `retire(handle)` のみで `requestReclaim` を呼ばず、Observe 経路の handle が Retired のまま Reclaimed にならず **256 slot が枯渇**。
- **修正:** `engine_.requestReclaimHandle(handle)` を追加。

### 1.4 🔴 Recovery 発行経路（quarantine → Recovery）が未配線 = デッドコード
- **種別:** 既存バグ（FUTURE-3 の配線漏れ）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:73-100`
- **問題:** 計画書（:654-656）は「quarantine 検出時に Recovery を発行」と設計。しかし `QuarantineIntentHandler` は `executeQuarantine` のみで Recovery を発行せず、`submitRecoveryRequest` は誰からも呼ばれない = **Recovery は決して実行されない**。
- **修正:** (a) AudioEngine に `currentBuildSnapshot_`（現在の publish 構成 snapshot）を追加、`enqueuePublicationIntentForRuntimeCommit` で更新。(b) `getCurrentBuildSnapshotForRecovery()` を追加。(c) `QuarantineIntentHandler` が quarantine 成功後に `submitRecoveryIntent` を呼ぶ。

### 1.5 🔴 submitRecoveryRequest が push 失敗を無視（INV-5 違反 + shutdown ハング）
- **種別:** 既存バグ（INV-5 違反）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/ISRRuntimePublicationCoordinator.cpp:643-669`
- **問題:** `recoveryIntentQueue_`（SPSC, 256）が full のとき push 失敗を無視 → Recovery が静かに drop + `pendingIntentCount_ + 1` が実行され、`isFullyDrained` が false のまま = shutdown ハング。
- **修正:** push 戻り値をチェック — 成功時のみ `pendingIntentCount_ + 1`、失敗時は `recoveryIntentDropCount_` に記録。`drainDeferredRetireQueues` の overflow→throttle 判定と `RuntimeBackpressureTelemetry` に統合。

### 1.6 🔴 requestReclaim の TOCTOU による slot リーク
- **種別:** 新規バグ（修正 1.2 で導入した潜在問題）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/ISRRuntimePublicationCoordinator.cpp:573-610` / `AudioEngine.h:4237` / `AudioEngine.Retire.cpp:83`
- **問題:** `requestReclaimHandle` と `drainDeferredRetireQueues` の epoch 事前チェック後、`requestReclaim` 内部の再確認で epoch が進んだ場合（TOCTOU）、`reclaimInFlightCount_` +1 して return し、**handle が保留リストに再登録されず喪失**。
- **修正:** `requestReclaim` を `bool` 返しに変更（true=reclaim 完了, false=遅延）。呼び出し元は false を受け取った場合、handle を `pendingReclaimHandles_` に再登録。

### 1.7 🔴 QuarantineIntentHandler が quarantine 失敗時にも Recovery を発行
- **種別:** 新規バグ（修正 1.4 で導入した配線漏れ）
- **状態:** ✅ 修正済み
- **根拠:** `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:82-99`
- **問題:** `executeQuarantine` の戻り値を無視し、`!request.handle.isNull()` のみで Recovery を発行。quarantine 失敗時（既に隔離済み・無効 handle）にも Recovery を発行し、無意味な世界再構築を引き起こす。
- **修正:** `executeQuarantine` の戻り値 `QuarantineResult.stateChanged` を確認し、**`stateChanged == true` かつ handle 非 null の場合のみ** Recovery を発行。

---

## 2. 未修正バグ・修正漏れ（P2 レベル）

### 2.1 🟡 per-type admission policy は統一機構として未実装
- **状態:** ⚠️ 分散実装で機能的には満たすが、統一機構なし
- **根拠:** `submitObserve`/`submitQuarantine`/`submitRecoveryRequest`/`enqueuePublicationIntent` に個別実装。統一 `AdmissionPolicy` エンジンなし。
- **推奨:** Phase 5 で統一機構を検討（将来のポリシー変更が各関数の修正を要する）。

### 2.2 🟡 MpscBoundedRing の producer hole が cross-type FIFO を弱める
- **状態:** ⚠️ 許容範囲（遅延 1ms 未満）だが厳密な FIFO 保証なし
- **根拠:** `processIntent`（ProcessIntent.cpp:36）の `while(intentQueue_.pop())` が producer hole で途中終了。
- **推奨:** Phase 6 で cross-type FIFO 順序保証テスト。

### 2.3 🟡 setPendingIntentCount(0) により isFullyDrained の pending count チェックが実質無効
- **状態:** ⚠️ 既存設計の問題
- **根拠:** `processIntent`（ProcessIntent.cpp:43）末尾の `setPendingIntentCount(0)`。残留 intent が pending count に反映されない。
- **推奨:** Phase 6 で `isFullyDrained` を pending count ではなくキュー空チェックに依存する方式へ改善。

### 2.4 🟡 Recovery の coalesce（マージ）が未実装
- **状態:** ⚠️ 計画書:860 との乖離
- **根拠:** `submitRecoveryRequest` は同一 quarantinedHandle の重複 Recovery をマージしない。
- **推奨:** Phase 5 で coalesce を実装するか、accept するか判断。

### 2.5 🟡 テストカバレッジ欠落（INV-3/INV-5 のテスト未実装）
- **状態:** ⚠️ 計画書:1181 の `invariant_*.cpp` 形式のテストが存在しない
- **根拠:** `requestReclaim` bool 化 / `pendingReclaimHandles_` / `retireByHandle` / Recovery 発行 / `recoveryIntentDropCount` がテスト・ハーネスで一切参照されていない。
- **推奨:** Phase 6/7 で `invariant_INV-3`/`invariant_INV-5` と (a)-(d) の単体テストを追加。

### 2.6 🟡 最近の修正のコンパイル検証が未実施
- **状態:** ⚠️ 環境制約（vcvarsall.bat 未初期化）
- **根拠:** `build-output-icx.txt` は recent fixes 前のログ。ソース（6/8 06:32）はログ（8/8 23:30）後に変更。
- **推奨:** 開発者コマンドプロンプトで `build.bat Release` を実行し、コンパイル確認（Phase 1 完了条件）。

### 2.7 🟡 shutdown 時の pendingReclaimHandles_ エッジケース
- **状態:** ⚠️ 異常系のみ（stuck Reader 時）
- **根拠:** `~AudioEngine`（CtorDtor.cpp:185-224）で pendingReclaim が残ると `markShutdownComplete` → `Faulted` の可能性。
- **推奨:** Phase 6 で「pendingReclaimHandles_ が shutdown 完了前に空になること」をテスト固定（INV-6）。

---

## 3. 既存バグ・既知事項（今回の改修とは無関係・改善提案）

### 3.1 🟡 2つの ShutdownPhase enum が共存
- **状態:** ⚠️ 既存問題
- **根拠:** `AudioEngine::ShutdownPhase`（AudioEngine.h:2510, `int`）と `convo::isr::ShutdownPhase`（ISRShutdown.h:25, `uint8_t`）の2種類が共存。CtorDtor.cpp は前者、ReleaseResources.cpp は両方を手動で対応付け。
- **推奨:** 対応表を invariant テストで固定（enum 追加時の不整合防止）。

### 3.2 🟡 PublishReceiptWaiter は high-water mark（厳密な per-seqId FIFO ではない）
- **状態:** ⚠️ 設計上許容
- **根拠:** AudioEngine.h:3607 `if (seqId > lastCompleted_) lastCompleted_ = seqId`。後の seqId が先に完了すると先の waitFor が即 true。
- **推奨:** SPSC（single consumer）前提が維持される限り問題なし。MPSC 化時に不変条件としてテスト固定。

### 3.3 🟡 ConvolverProcessor の LinearRamp は BUG-028 の対象外（分離実装）
- **状態:** ✅ 設計判断（対象外）
- **根拠:** `ConvolverProcessor.h:910,935,945` の `latencySmoother`/`crossfadeGain`/`mixSmoother` は CrossflareRuntime の `gain_`/`dryScaleGain_` と別個。
- **推奨:** 将来の独立 RT-safety 検証（ConvolverProcessor 固有）。

### 3.4 🟡 BlockDouble の finalizeCrossfadeMixPath(..., false) で dryScaleGain_ 未リセット
- **状態:** ⚠️ 事前存在の差異（work88 対象外）
- **根拠:** BlockDouble.cpp:434=false、AudioBlock.cpp:458=true。double パスで dryScaleGain_ がリセットされない。
- **推奨:** Phase 6（soak）で BUG-028 の RT-only ownership 契約との整合を確認。

### 3.5 🟡 bootstrap publishWorld 失敗の ignoreUnused
- **状態:** ⚠️ 軽微なロバストネス事項
- **根拠:** Init.cpp:55 `juce::ignoreUnused(result)`。失敗時は次の操作が null world を検出。
- **推奨:** 必要なら `jassert` 追加（Phase 9）。

---

## 4. 残余リスク（文書化・対応推奨）

| # | 残余リスク | 検証結果 | 対応推奨 |
|---|-----------|---------|----------|
| R1 | `recoveryIntentQueue_` は SPSC（:434）。将来 Timer 等から直接呼ぶ場合は MPSC 化が必要 | ✅ 妥当（Producer は現在 CoordinatorLoop のみ） | Phase 5 将来拡張メモ |
| R2 | `observeIntentQueue_`/`observeFallbackQueue_` は Observe 統合後デッドメンバ | ✅ 妥当・**削除済み**（コメントのみ残存） | 対応済み |
| R3 | Recovery 経路はプロダクション producer 未接続 | 🔴 **不正確（接続済み）** — QuarantineIntentHandler → submitRecoveryIntent が有効 | 対応済み |
| R4 | retire 順序逆転は残るが quarantine fallback で UAF/リーク排除。requestReclaim 一本化は保留 | ✅ 妥当（runtime 経路の shutdownReclaim は排除済み） | runtime 経路対応済み。shutdownReclaim は shutdown 専用 |
| R5 | bootstrap ignoreUnused による null world リスク | ✅ 妥当（軽微） | 要対応なし |
| R6 | BlockDouble finalizeCrossfadeMixPath(false) で dryScaleGain_ 未リセット | ✅ 妥当（work88 対象外） | Phase 6 確認 |

---

## 5. 改善提案（Phase 5/6/7 の候補）

### 5.1 Phase 5（実装フェーズ）
- **P5-1:** per-type admission policy の統一機構（AdmissionPolicy エンジン）導入検討
- **P5-2:** Recovery coalesce（同一 quarantinedHandle のマージ）実装検討
- **P5-3:** MpscBoundedRing の producer hole テスト + cross-type FIFO 順序保証テスト

### 5.2 Phase 6（検証フェーズ）
- **P6-1:** `invariant_INV-3`/`invariant_INV-5` テスト追加（pendingReclaim/Recovery/TOCTOU の回帰）
- **P6-2:** `isFullyDrained` を pending count ではなくキュー空チェックに依存する方式へ改善
- **P6-3:** pendingReclaimHandles_ が shutdown 完了前に空になることのテスト固定（INV-6）
- **P6-4:** BlockDouble の finalizeCrossfadeMixPath(false) と BUG-028 契約の整合確認
- **P6-5:** dual ShutdownPhase enum の対応表を invariant テストで固定
- **P6-6:** PublishReceiptWaiter の FIFO 不変条件テスト（MPSC 化後）

### 5.3 Phase 7（統合フェーズ）
- **P7-1:** ビルド検証（`build.bat Release`）と全テスト実行
- **P7-2:** Recovery 発行経路の Integration Test（quarantine → Recovery → build → publish 一連）

---

## 6. 検証済みの実装（修正・追記の根拠）

| 項目 | 状態 | 根拠 |
|------|------|------|
| BUG-014（MmcssPolicy enum atomic） | ✅ 実装済み | AudioEngine.h:2347-2383 + static_assert |
| BUG-028（CrossflareRuntime RT 専有化） | ✅ 実装済み | start() の setCurrentAndTargetValue(0.0) 削除 / complete() の atomic publish / crossfadeGeneration_ |
| BUG-015/027（RetireQuarantineStore） | ✅ 実装済み | array+index 配置（allocation-free）/ Router API / drain 配線 |
| FUTURE-3（Recovery buildSource） | ✅ 実装済み | RecoveryIntent::buildSource + submitRecoveryRequest(handle, buildSource) |
| FUTURE-10（MpscBoundedRing） | ✅ 実装済み | intentQueue_ MPSC 化 / quarantineFallbackQueue_ / Observe 統合 |
| RecoveryIntentHandler | ✅ 実装済み | enqueue-only + Builder Work Queue 転送（将来拡張用 dead code） |
| FUTURE-5（MemoryPool） | ✅ 実装済み | freeSlots_ フリーリスト |
| FUTURE-6（HandleTable） | ✅ 実装済み | DSPHandleTable（固定 512 open addressing） |
| SHUTDOWN-7（順序） | ✅ 実装済み | ReleaseResources.cpp:75→188→189→190→191→430 |

---

## 7. 優先度別サマリ

### 🔴 P0（即時対応必須 — すべて修正済み）
1.1 RetireQuarantineStore drain mutex / 1.2 retireDSPHandleForRuntime shutdownReclaim / 1.3 retireByHandle slot リーク / 1.4 Recovery 発行経路デッドコード / 1.5 submitRecoveryRequest push drop / 1.6 requestReclaim TOCTOU / 1.7 quarantine 失敗時 Recovery

### 🟡 P2（Phase 5/6/7 で対応）
2.1 per-type admission 統一 / 2.2 producer hole FIFO / 2.3 pending count 無効 / 2.4 Recovery coalesce / 2.5 テストカバレッジ / 2.6 ビルド未検証 / 2.7 shutdown pendingReclaim

### 🟢 P3（改善提案・記録）
3.1 dual ShutdownPhase / 3.2 PublishReceiptWaiter high-water / 3.3 Convolver LinearRamp 分離 / 3.4 BlockDouble false / 3.5 bootstrap ignoreUnused / R1-R6 残余リスク

---

## 8. 実装前のアクション（推奨順序）

1. **Phase 1 完了条件:** `build.bat Release` を vcvarsall 初期化済み環境で実行（2.6）
2. **Phase 6 完了条件:** invariant_INV-3/INV-5 テスト追加（2.5, P6-1）、isFullyDrained 改善（2.3, P6-2）、shutdown pendingReclaim テスト（2.7, P6-3）
3. **Phase 5 改善候補:** per-type admission 統一（2.1）、Recovery coalesce（2.4）、producer hole テスト（2.2）
4. **将来拡張メモ:** recoveryIntentQueue_ MPSC 化（R1）、複数 DSP world の Recovery spec 管理（項目18）
