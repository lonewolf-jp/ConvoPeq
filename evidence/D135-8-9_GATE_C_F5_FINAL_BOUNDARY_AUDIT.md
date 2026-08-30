# D135-8/9 Gate C-F5 — 実装前最終パッチ境界・状態遷移整合性監査

Date: 2026-08-30
Scope: **production/test source 0変更・実装0・テスト0・CTest0・ビルド0**。最新 working tree（ConvoPeq.md 2026-08-30 版と同一）基準。F4 の指摘どおり「Rejected* = 全部終端」の一般化を撤回し、完全表に置換した。

## Verdict: **GO — F6 実装へ進める**（条件付き、条件は §D に内包）

F4 の修正案（Option A + fade-complete wake + watchdog + obligation-keyed TTL）は、
最新ソースの全状態遷移に当てはめても既存契約を壊さないことを確認した。ただし F4 文書の
以下の記述は本監査で**訂正**する（訂正済み契約で F6 に引き継ぐ）。

---

## F5-1. `recoveryObligationId` 全ライフサイクル（追跡結果）

| 段階 | 箇所 | 挙動 |
| --- | --- | --- |
| 生成 | ISRRuntimePublicationCoordinator.cpp:938/:1098（`pendingRecoveryAdmission_.recoveryObligationId = oblId`） | recovery admission が table の obligationId を採番 |
| PublishRequest 付与 | :1136（intent.obligationId）→ RebuildDispatch:1040/:1120（`enqueuePublicationIntentForRuntimeCommit(..., recovery->obligationId)`）→ Commit.cpp:813（`req.recoveryObligationId`） | 通常 publish は既定 0（AudioEngine.h:2591 既定値、:4653 明示） |
| Deferred 化 | Orch:377-378（DeferredFadingActive → enqueueDeferred(req)） | req は obligationId を保持したまま slot に入る |
| re-drive | consume→submit→Deferred→再 enqueue | 同一 (gen, obligationId) が維持される |
| publish 成功 | trySubmitImpl:320（Route A）/ onPublishCommitted:354（Route B）→ `resolveRecoveryObligation(id, Published)` | **id==0 は no-op（early-return、:352-353 明記）** |
| Retry | RejectedPressure → resolveIfRecovery(Retry) + `rearmRecoveryRetry`（:1028-1035、pendingRecoveryAdmission_ が同一 oblId かつ Building のときのみ settle。他 obligation を上書きしない） | 非終端 |
| transient failure | RejectedNotFinalized → `markTransientFailure`（:998-1024、delivery→None・counter+1・枯渇時 ResolvedFailed）/ RejectedPublishFailure → trySubmitImpl 内で markTransientFailure（R18、:306-311） | 非終端（bridge 側で再駆動適格化） |
| stale / shutdown | resolveIfRecovery(StaleSuperseded / ShutdownDiscarded)（:385/:420） | 終端 |
| 同一 generation 再利用 | recoveryGeneration = 現在値（RebuildDispatch:1035-1036、++ せず） | **gen 単独では identity にならない**（F4-1 確認） |

**`(generation, recoveryObligationId)` を deferred identity に導入した場合の通常 publish**:
`obligationId==0` は全 bridge 関数（markTransientFailure:1000、rearmRecoveryRetry:1030、
redriveDeferredRecovery:1060、resolveRecoveryObligation）で early-return が保証済み。
identity 比較でも (G,0) は (G,O≠0) と明確に区別される。✅

## F5-2. F4 identity 修正と recovery semantics の衝突 → **衝突なし（別系の証明）**

| 機構 | 状態 | 所属 |
| --- | --- | --- |
| `deferredRetryGeneration_/Count_` + `kMaxDeferredRetries` | deferred **publish slot** の再駆動会計 | Orchestrator（RebuildThread single owner） |
| `recoveryAdmissions_` table + `consecutiveFailureCount` + `kMaxObligationConsecutiveFailures` | **Recovery Bridge** の obligation retry（真の Type A 相当） | RuntimeIntentCoordinator（bridge） |
| `lastRecoveryPublishSeq_` | recovery 発行 world の seq 相関スタンプ（write-only latch、D135-5 監査済み） | Orchestrator |
| `recoveryRetryReady` | recovery wake の provenance atomic（predicate 非参加） | AudioEngine（rebuildMutex 保護域） |
| `deferredRecoveryRearmed_` | **削除済み確認**（grep 0件。D135-4 計画どうり dead code 除去済み） | — |

→ **deferred retry identity と Recovery Bridge obligation identity は別系**。F4 の identity 拡張
`(gen, obligationId)` は bridge の table identity（oblId 主キー）と衝突しない（単に slot 側のキーを
精緻化するだけ）。✅

## F5-3. obligation timestamp の lifetime（single-owner 安全性）

新設 `deferredObligationCreatedAtUs` + key=(gen, obligationId) は
`deferredRetryGeneration_/Count_` と同一の RebuildThread single-owner 域（enqueueDeferred /
processDeferredAdmission / finishView / clearDeferredForShutdown のみ触る）。遷移表:

| 事象 | 動作 |
| --- | --- |
| 初回 Deferred (G,O) | key=(G,O), createdAt=now |
| 同一 (G,O) re-drive | **createdAt 維持**（TTL dwell 継続） |
| 別 (G',O') enqueue | key 不一致 → createdAt 更新（新 obligation） |
| SupersededDiscard | 旧 slot DSP retire（Orch:460-468）→ 新 enqueue が key 置換。旧 obligation の createdAt は自然に失効（key 一致でのみ読まれる） |
| StaleDiscard / Expired | finishView（reason≠None）→ **key を無効化**（`deferredObligationCreatedAtUs=0` 相当）。terminal 後の再読みなし |
| ShutdownDiscard | clearDeferredForShutdown が retry メンバと併せて reset（Orch:548-549 と同位置） |
| Accepted（Success） | consume→finishView（reason=None）→ **key 無効化**（obligation 終端） |

slot と obligation metadata の lifetime: slot は consume/discard 毎に消滅・再作成されるが、
**obligation の同一性は key で判定**するため一致する（slot の生死 ≠ obligation の生死、が正しい分離）。✅

## F5-4. TTL source-of-truth（二重 timestamp 回避）

現状 slot 内に timestamp が2つある: `metadata.enqueueTimestampUs`（evaluateDeferred が読む、PA.cpp:75）と
`slot.enqueueTimestampUs`（maxDeferredAgeMs 計算用、Orch:449）。決定:
- **TTL の source-of-truth = obligation-keyed `deferredObligationCreatedAtUs`**。
- enqueue 時に `metadata.enqueueTimestampUs := (key 一致 ? 既存 createdAt : now)` — つまり metadata は
  **keyed member から導出**し、evaluateDeferred は無変更で読める（PA 変更なし）。
- slot 側の `enqueueTimestampUs` は「今回の enqueue 時刻」の意味のまま（overwrite age 専用）。
  両者の意味をコメントで固定し、「片方だけ更新」を構造的に防ぐ（更新は enqueueDeferred 内 1 箇所で同時実施）。✅

## F5-5. fade-complete wake の exact insertion point（固定）

実コード順（Timer.cpp:928-1001）: `tryCompleteFade()` → `notifyFadeComplete`/`consumeCompletedFade` →
`endCrossfade`/`unregisterCrossfade` → `fadingRuntimeDSPSlot` CAS クリア → `submitObserve`（retire intent）→
`crossfadeRuntime_.complete()` → delay/dryhold リセット → `refreshCrossfadePreparedSnapshotFromAtomics()` →
**idle publish（commitRuntimePublication :994-997）** → `sendChangeMessage()`。

**発行点: ブロック末尾（idle publish 完了後、sendChangeMessage の直後）** — world 状態が
「fading なし + idle commit 済み」で整合してから再評価を依頼する。条件:
```cpp
if (!isShutdownInProgress() && runtimeOrchestrator_ != nullptr
    && runtimeOrchestrator_->hasDeferredRequest()) {
    { std::lock_guard<std::mutex> lock(rebuildMutex); publishRetryReady = true; }
    rebuildCV.notify_one();
}
```
- fade 完了前に wake しない: ブロック自体が fadeCompleted 時のみ実行 ✅
- deferred 不在で no-op: `hasDeferredRequest()` ガード ✅
- shutdown 中 wake なし: `isShutdownInProgress()` ガード ✅
- provenance 非汚染: `recoveryRetryReady` に触れない ✅（recovery handoff Timer.cpp:1746-1751 とは異なり recovery フラグを立てない）
- lock ordering: timerCallback 本体は mutex 非保持（grep 確認済み）、rebuildMutex はリーフレール ✅

## F5-6. watchdog の scheduler ownership（3候補比較 → 最小変更を確定）

| 候補 | 内容 | 評価 |
| --- | --- | --- |
| **(a) CoordinatorLoop 既存 phase** | `runCoordinatorPhase`（Threading.cpp:280-289）の deferred 再通知を tick カウンタで間引く（毎 tick → 1/N tick） | **採用**。同一スレッド・同一 handoff・既存コード 1 箇所のゲート追加のみ。predicate/CV 契約不変 |
| (b) MessageThread timer | timerCallback に別カウンタ | producer 側が二重化（coordinator + message thread）。余計な同期検討増 |
| (c) rebuild thread wait_for | `rebuildCV.wait_for` 化 | CV 使用形態の変更（predicate loop + timeout）で lost-wake 解析を再やる必要。リスク最大 |

→ **(a) 確定**。周期 N は named constant として実装時決定（F4-5 維持、数値未固定）。✅

## F5-7. watchdog と TTL の関係（TTL 跨ぎ再評価なし）

```text
retention（slot 保持）→ watchdog / fade-complete wake → processDeferredAdmission
→ evaluateDeferred → ageUs = now - obligationCreatedAt > TTL → Discard(StaleDiscard)
→ view.discard → finishView（slot reset, hasDeferred_=false, key 無効化）
```
終端後: coordinator poll は `hasDeferredRequest()==false` で wake せず、再 enqueue 経路は
`submitPublishRequest → DeferredFadingActive` のみだが、その req は既に破棄済みで存在しない。
**TTL expiration 後の再 enqueue 経路なし** ✅。watchdog は TTL 以内の dwell 監視のみ（≤ dwell/TTL 間隔分）。

## F5-8. `publishRetryReady` producer-consumer-provenance 最終表

| producer | 箇所 | 設定フラグ | provenance |
| --- | --- | --- | --- |
| P1 coordinator（→ watchdog 化） | Threading.cpp:286 | publishRetryReady | なし（ordinary） |
| P2 recovery | Timer.cpp:1748-1749 | **recoveryRetryReady + publishRetryReady** | recoveryRetryReady（exchange 消費 RebuildDispatch:888） |
| P3 fade-complete（**新設・F6**） | Timer.cpp fadeCompleted 末尾 | publishRetryReady のみ | なし（ordinary 扱い・wasRecoveryWake=false） |
| consumer | RebuildDispatch:889-890 | read-and-clear（rebuildMutex 内） | doDeferredPublish へ |
| 消去 | CtorDtor:182 / PrepareToPlay:82 / ReleaseResources:175 / RebuildDispatch:866 | false | lifecycle |

再検索結果: 現行 producer は依然 **P1/P2 の2箇所のみ**（増減なし）。✅

## F5-9. retry accounting 最終契約（Type A の所在を確定）

**訂正**: F4 の「Type A retry は実在しない」は**不正確**。真の Type A 相当は
**Recovery Bridge 側に実在**する（`markTransientFailure` の `consecutiveFailureCount` +
`kMaxObligationConsecutiveFailures` 枯渇→ResolvedFailed、delivery→None で redrive 再適格）。
deferred publish slot の `kMaxDeferredRetries` は最初から **Type B churn bound** として設計された
（enum コメント「busy-loop を有限回に停止させる唯一の ownership-release path」）。

最終契約:
```text
ordinary fading retention（DeferredFadingActive 再 enqueue）→ slot count 不変
recovery wake（recoveryRetryReady）→ slot budget reset（P3 維持）
Ready→Accepted→publish 失敗（trySubmitImpl 内）→ bridge markTransientFailure（bridge 側 retry）
RejectedPressure/NotFinalized（recovery）→ bridge Retry/rearm・transient（非終端）
Rejected*（non-recovery）→ 終端（現行契約維持）
```
→ **slot の `kMaxDeferredRetries` は dormant 化（削除せず）**。churn bound は watchdog 間隔化 +
obligation-keyed TTL（30s dwell）が担う。将来 slot レベルの Type A を導入する場合のみ再起用。✅

## F5-10. Rejected 系 terminal 完全表（F4 からの訂正）

| Decision | non-recovery (id=0) | recovery (id≠0) | DSP 所有権 |
| --- | --- | --- | --- |
| Accepted | Success | Success + resolve(Published) | live world へ昇格 |
| DeferredFadingActive | retention（slot） | retention + bridge delivery  residency 維持 | slot 内保持 |
| RejectedStaleGeneration | **終端**（req 消滅） | 終端 + resolve(StaleSuperseded) | ⚠ slot 由来 re-drive 時は consume 済み req の handle 回収は §follow-up |
| RejectedNotFinalized | **終端** | **非終端**: markTransientFailure（delivery→None・counter+1・枯渇で ResolvedFailed） | 同上 |
| RejectedPressure | **終端** | **非終端**: resolve(Retry) + guarded rearmRecoveryRetry | 同上 |
| RejectedShutdown | 終端 | 終端 + resolve(ShutdownDiscarded) | shutdown teardown 回収 |
| RejectedPublishFailure | 終端 | 非終端（trySubmitImpl で markTransientFailure 済み、R18） | **回収済み**（destroyRolledBackDSP、Orch:304-305 明記）✅ |
| RetryExhaustedDiscard | dormant 化（到達不能） | 同左 | （Type A 専用だった経路） |

**F4-7 の訂正**: 「Rejected* は DSP を retire せず消滅」は **RejectedPublishFailure では誤り**
（destroyRolledBackDSP で回収済み）。non-recovery の Pressure/NotFinalized/StaleGeneration 経路での
handle 滞留可能性は残る → **別パッチ #7 の follow-up として維持**（本パッチ範囲外）。

---

## A. 最終 Deferred state machine

```text
PublishRequest (gen G, obligationId O)
   │ submitPublishRequest → trySubmitImpl → evaluate
   ├ Accepted ────────────────→ Success（resolve(Published) if O≠0）
   ├ DeferredFadingActive ───→ enqueueDeferred
   │     key=(G,O) 一致 → createdAt 維持 / 相違 → 新 createdAt
   │     [RETAIN] hasDeferred_=true
   │        wake: fade-complete(P3) / watchdog(P1 間引) / recovery(P2)
   │        → processDeferredAdmission（RebuildThread 専用）
   │        → evaluateDeferred: Shutdown→TTL(createdAt 起点)→Gen→Seq
   │        ├ Discard → finishView（slot reset・key 無効化・telemetry）終端
   │        └ Ready → consume → submitPublishRequest（上記分岐へ）
   ├ RejectedStaleGeneration → 終端（recovery: resolve(StaleSuperseded)）
   ├ RejectedNotFinalized → non-rec: 終端 / rec: markTransientFailure（非終端）
   ├ RejectedPressure → non-rec: 終端 / rec: Retry+rearm（非終端）
   ├ RejectedShutdown → 終端（resolve(ShutdownDiscarded)）
   └ RejectedPublishFailure → 終端（DSP は destroyRolledBackDSP 回収済み）
                              / rec: markTransientFailure（非終端）
   ※ RetryExhaustedDiscard: dormant（retention は count しない）
   ※ 新要求の DeferredFadingActive enqueue → 旧 slot SupersededDiscard（旧 DSP 明示 retire）
```

## B. `(generation, recoveryObligationId)` identity / timestamp lifecycle 表

→ §F5-1（lifecycle）+ §F5-3（timestamp 遷移表）に統合済み。要点: identity は tuple、
createdAt は同一 tuple でのみ維持、terminal（discard/accepted/shutdown）で key 無効化。

## C. `publishRetryReady` producer-consumer-provenance 表

→ §F5-8。producer 3（P1 watchdog 化 / P2 recovery / P3 fade-complete 新設）、consumer 1（read-and-clear）、
provenance は recoveryRetryReady 独立 atomic。

## D. 実装パッチ境界（F6 へ引き継ぐ確定事項）

**必須（F6 で実装）**
1. retention accounting 分離: enqueueDeferred に retention 伝達、identity=(gen, obligationId)、count 不変
2. obligation timestamp 保存: `deferredObligationCreatedAtUs` keyed メンバ + metadata 導出（§F5-4）
3. fade-complete wake: Timer.cpp fadeCompleted 末尾、§F5-5 の条件・位置・ガード通り
4. watchdog 化: Threading.cpp:280-289 を tick カウンタで間引く（候補 a。周期は named constant）
5. 契約コメント/文書統一: Orch.h:283-289、State.h:16-20、D135-1 文書（`>` 意味論 + retention 除外 + dwell TTL + kMax dormant）

**別パッチ（混ぜない）**
6. Expired enum 活性化
7. Rejected*（non-recovery）の handle 回収 / retry 接続の再設計（§F5-10 ⚠ 列）
8. telemetry schema（oldestDeferredAgeMs dwell 化・メンバ改名）

**不変確認事項（実装後の Gate B/C で再検証）**: CV predicate 不変、単一 slot（INV-DEFERRED-2）不変、
retire pipeline 不変、recovery provenance 経路不変、テストソース無変更。

## GO / NO-GO 判定

**GO**。F4 の2点（Rejected* 一般化、Type A 不在）は本監査で訂正済みで、訂正後の契約でも
Option A + event wake + watchdog + obligation-keyed TTL は全既存状態遷移と整合する。
F6 実装パッチ設計へ進んでよい。
