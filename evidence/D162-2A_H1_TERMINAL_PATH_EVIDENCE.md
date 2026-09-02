# D162-2-A — H1 Terminal Path Evidence（read-only・実証引用）

Date: 2026-09-02
Source basis: 現行 working tree（D162-1R-B 計装後・ConvoPeq.md Generated: 2026-09-02 21:53:52）
+ evidence/D162-1P_soak.log（91,105 行）

## 1. D162-1P soak log からの実証（49 DSP の経路）

### 1.1 計数

| 事象 | count | 意味 |
| --- | ---: | --- |
| [D133] queue snapshot / [D133] enqueue | 60 / 60 | 全 gen が enqueuePublicationIntentForRuntimeCommit に到達（= 全件 handle registered・Commit.cpp:804） |
| trySubmit: executor_.publish SUCCEEDED | 11 | published gens = [5,6,12,18,24,30,36,42,48,54,60] |
| trySubmit: executor_.publish FAILED | 0 | publish 実行 failure なし |
| commitRuntimePublication FAILED | 0 | 同上 |
| [D135] re-defer (new) | 49 | 49 非出版 gen が DeferredFadingActive → enqueueDeferred に入った |
| [D135] re-defer (retain) | 1678 | deferred re-drive（retention）の反復 |
| CoordExit | 11 | trySubmitImpl Phase2 に到達したのは published 11 のみ |
| [D117_DESTROY] | 11 | placeholder + replaced published 10 |
| [D117_RETIRE(lifetime)] | 2 lines (1 event) | DSPLifetimeManager::retire は placeholder 1 件のみ |
| [D117_RETIRE_BY_HANDLE] HIT | 10 | replaced published 10（Observe 経路の retireByHandle） |
| StaleDiscard / ShutdownDiscard ログ | 0 | S2 discard は不発（TTL 30s > 7s/gen で overwrite が先行） |
| RejectedPressure/Shutdown telemetry | 0 | S4 は不発（soak 環境） |
| [HEALTH] Deferred publish starved | 0 | dormant RetryExhaustedDiscard 不発（F6-6 どおり） |

### 1.2 published old-DSP の uuid 突合（CoordExit）

```text
gen=5  currentUuid=4  nextUuid=3   ← gen5 が placeholder(uuid3) を置換
gen=6  currentUuid=5  nextUuid=4   ← gen6 が gen5 を置換
gen=12 currentUuid=11 nextUuid=5   ← gen12 の old = uuid5 = gen6 の DSP（gen7..11 は一度も active でない）
```

→ 非 publish DSP は oldHandle として到達不可能（publish 経路の retire は対象外）。
D162-1R-B short soak でも同一構造（enqueue==retained gens [5..15]、destroy 2、DC live 1→10）。

### 1.3 算術閉包

```text
総 DSPCore = placeholder 1 + 60 gen = 61
destroy    = 11 (placeholder 1 + replaced published 10)
live 終端  = 50 (= 61 - 11)  … MEM_SNAP DC live=50 と一致
orphan     = 49 (= 50 - 1 active) … re-defer (new) ×49 と一致
内訳: S1 overwrite orphan 48（最終 slot 残留 1 を除く全非出版 gen）+ S3 shutdown clear orphan 1
```

S1 overwrite の retire は `engine_.retireDSPHandleForRuntime` 直呼び（Orchestrator.cpp:466）で
DSPLifetimeManager を通らないため D117 ログ site が存在しない → 「黙示」に一致
（D117_RETIRE=1 event しかないにもかかわらず overwrite は 48 回発生し得る）。

## 2. 現行コードの terminal path 表（行番号・現行 tree）

### 2.1 完全 retire（T1: EBR）

| site | file:line | 内容 |
| --- | --- | --- |
| retire 唯一の完全実装 | audioengine/DSPLifetimeManager.cpp:40-77 | :45 台帳解除 + :58 enqueueWithRetry(dsp, destroyDSPCoreNode, epoch) |
| publish 完了 tail | audioengine/DSPTransition.h:49-160 | :79/:82/:129/:153 lifetime.retire(oldDSP) |
| Observe retire | audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp:96-105 → DSPLifetimeManager.cpp:79-137 | retireByHandle → :121 enqueueWithRetry |
| EBR 所有権取得点 | audioengine/ISRRetireRouter.cpp:303（不変式 :25「never returns with ptr unowned」） | destroy enqueue（Q/E/T/overflow 転送込み） |
| fading 完了 retire | audioengine/AudioEngine.Timer.cpp（fade-complete wake → DSPLifetimeManager::retire） | epoch 伝搬 HW-1 receipt 付き |

### 2.2 direct destroy（T2: 未登録 DSP）

| site | file:line | 内容 |
| --- | --- | --- |
| DSPGuard 契約 | audioengine/AudioEngine.RebuildDispatch.cpp:938-965 | retireDSPHandleForRuntime false → destroyDSPCoreNode（二重解放 CAVEAT 付き） |
| recovery warmup fail | audioengine/AudioEngine.RebuildDispatch.cpp:1036 / :1118 | destroyDSPCoreNode 直呼び |
| publish 実行失敗 | audioengine/RuntimePublicationOrchestrator.cpp:290-291 | destroyRolledBackDSP → destroyDSPCoreNode |
| build 失敗 | audioengine/RuntimeBuilder.cpp:425-443（unique_ptr 未 release） | RAII 破壊 |

### 2.3 H1 脱落 site（disposition 欠落・49 の脱落点）

| sub-path | site | 現行挙動 | 欠落 |
| --- | --- | --- | --- |
| S1 deferred overwrite | audioengine/RuntimePublicationOrchestrator.cpp:460-468 | `engine_.retireDSPHandleForRuntime(oldDSP)`（台帳解除のみ） | EBR enqueue（破壊権移譲）なし → **orphan 48** |
| S1' dormant | Orchestrator.cpp:497-510 | 同型（RetryExhaustedDiscard 分岐・現行不発） | 同上（latent） |
| S2 deferred discard | Orchestrator.cpp:734-738（view->discard = :686-692） | lastDiscardReason 記録 + finishView のみ | retire/destroy なし（登録済みのまま） |
| S3 shutdown clear | Orchestrator.cpp:546-570（呼出元: AudioEngine.Processing.ReleaseResources.cpp:359 / RebuildDispatch.cpp:915→:596-607） | slot reset + metadata 無効化のみ | retire/destroy なし → **orphan 1（soak）** |
| S4 admission Rejected* | Orchestrator.cpp:380-432 switch | telemetry / obligation 処理のみ | newDSP disposition なし（latent） |
| latent: quarantine | audioengine/AudioEngine.Threading.cpp:63-84（:84） | retireDSPHandleForRuntime 直呼び | destroy enqueue なし（quarantine resident 意味論の確認要） |

### 2.4 reclaim 系が DSPCore を破壊しない証明

- RuntimeIntentCoordinator::reclaimNormal（audioengine/ISRRuntimePublicationCoordinator.cpp:684-713）
  :706 「物理削除は retire path の enqueueWithRetry が担当」 — reclaim は slot 状態遷移のみ。
- AudioEngine::drainDeferredRetireQueues（audioengine/AudioEngine.Retire.cpp:45-140）—
  pendingReclaimHandles_ の再試行は requestReclaim（reclaim のみ）。
- よって S1 の台帳解除後、DSPCore は handle map / World / EBR のどこからも参照されない orphan。

## 3. Audio Thread trySubmit の現状

- 定義: audioengine/RuntimePublicationOrchestrator.cpp:34-38（trySubmitImpl への委譲）。
- production 呼出元: **0**（全 src/ 検索 — 「audio-thread trySubmit」は RuntimePublishExecutor.h:18 の
  コメント上の歴史参照のみ）。RT boundary（B-I3）に修正が触れる経路は現行なし。

## 4. thread 契約（S3 修正の制約）

- processDeferredAdmission / peekDeferred / finishView: RebuildThread jassert（Orchestrator.cpp:702/:621/:637）。
- clearDeferredForShutdown: assert-free・非 RebuildThread からも呼ばれる
  （ReleaseResources.cpp:359 EmergencyDrain / C1 fallback — D135-8 Gate A 審査済み）。
- enqueueWithRetry: NonRT-safe（mutex + allocation 可・B-I3 RT boundary 注記 ISRRetireRouter.cpp:308）。
  → S3 修正は「clear 時に authority retire を呼ぶ」形でも thread 契約と整合可能。
