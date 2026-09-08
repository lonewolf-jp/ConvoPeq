# D162-2-A Work Report — Retention Repair Design / Pre-Implementation Audit（read-only）

- Work item: D162-2-A（H1 修正対象を現行コードの所有権・publish・crossfade・retire 遷移から一意確定）
- Date: 2026-09-02
- Mode: **production source 変更 0**（src/ / tests/ / CMakeLists.txt 無変更。本 report と evidence のみ追加）
- 行番号の基準: D162-1R-B 計装適用後の現行 working tree（ConvoPeq.md `Generated: 2026-09-02 21:53:52`）
- 前提実測: D162-1P（60 gen・49 retained・destroy 11）/ D162-1R-B（retained DSP 142.48 MB・Δ0.5% 閉包）
- 関連: evidence/D162-2A_H1_TERMINAL_PATH_EVIDENCE.md（本 report の実証引用）

---

## 0. Executive Summary

1. **H1 の 49 DSP は「retire candidate にされなかった」のではない。retire candidate になったが、
   呼ばれた retire が「台帳解除のみ」の下位プリミティブであり、破壊権移譲（EBR enqueue）を含まないため
   orphan 化した。** 分類は指定枠の **H1-E**（詳細は §3）。
2. 脱落点は **deferred publish 系 4 sub-path** に集約される（§2.3）。D162-1P soak の実ログから、
   49 件の内訳は **48 件 = deferred slot overwrite 時の handle-only retire** ＋
   **1 件 = shutdown 時 clearDeferredForShutdown の黙示 slot 破棄** と推定される（§3.2）。
3. 破壊の唯一の authority は **`ISRRetireRouter::enqueueWithRetry`（EBR 所有権取得点）** と
   **未登録 DSP の直接破壊（DSPGuard 契約）** の 2 系統のみ。49 DSP はいずれにも到達していない。
4. 修正案の比較結果、**推奨は案 C′（既存単一 retire authority `DSPLifetimeManager::retire` への
   統一）**。4 脱落点を authority 経由に寄せる最小差分で、未 publish DSP の意味論を変えない（§4）。
5. 判定: **root cause と terminal owner が一意に確定 → 次工程は D162-2-B Repair Implementation**（§6）。

---

## 1. Phase A-1: DSPCore Lifecycle Matrix

DSP クラス × 遷移の現行事実。✓=到達する / ✗=到達しない / †=H1 脱落点。

| # | DSP class | construct | handle registered | published | active | fading | retire invoked | handle lookup | EBR enqueue | destroy | terminal |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | published（正常系・初回） | ✓ | ✓ Commit:804 | ✓ | ✓ | − | n/a（old なし） | − | − | n/a | active 継続 |
| 2 | replaced published（旧 active） | ✓ | ✓ | ✓(過去) | ✓(過去) | 場合による | ✓ DSPTransition:79/129/153 | ✓ | ✓ DSM:58 | ✓ | **EBR** |
| 3 | fading DSP（crossfade 中） | ✓ | ✓ | ✓(過去) | − | ✓ | ✓ fade-complete Timer / CAS 失敗 DSPTransition:129 | ✓ | ✓ DSM:58 | ✓ | **EBR** |
| 4 | same-DSP republish（D132(5)） | ✓ | ✓ | ✓ | ✓ | − | ✗（意図的免除） | − | − | ✗ | active 継続（正しい） |
| 5 | rebuild-obsolete（未登録） | ✓ | ✗ | ✗ | ✗ | ✗ | guard dtor | lookup=null | ✗ | ✓ direct | **direct destroy**（DSPGuard 契約） |
| 6 | failed-build（例外/不正入力） | ✓/✗ | ✗ | ✗ | ✗ | ✗ | − | − | − | ✓ unique_ptr/guard | direct destroy |
| 7 | recovery warmup-fail | ✓ | ✗ | ✗ | ✗ | ✗ | − | − | − | ✓ direct（RD:1036/1118） | direct destroy |
| 8 | publish-execution-fail | ✓ | ✓ | ✗ | ✗ | ✗ | ✗（rollback 済み） | − | − | ✓ Orch:291 | direct destroy |
| 9 | world-build-fail（A/B） | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ Orch:181/249 | ✓ | ✓ DSM:58 | ✓ | EBR |
| 10 | **deferred → overwrite** | ✓ | ✓ | ✗ | ✗ | ✗ | **△ Orch:466（handle 解除のみ）** | ✓ | **✗** | **✗** | **† なし（orphan）** |
| 11 | **deferred → TTL/generation/sequence discard** | ✓ | ✓ | ✗ | ✗ | ✗ | **✗（view->discard のみ Orch:736）** | − | **✗** | **✗** | **† なし（orphan・登録済みのまま）** |
| 12 | **deferred → shutdown clear** | ✓ | ✓ | ✗ | ✗ | ✗ | **✗（slot reset のみ Orch:555）** | − | **✗** | **✗** | **† なし（orphan）** |
| 13 | **admission rejected（stale/pressure/shutdown/publishFailure）** | ✓ | ✓ | ✗ | ✗ | ✗ | **✗（switch に処分なし Orch:374-433）** | − | **✗** | **✗** | **† なし（orphan・latent）** |
| 14 | quarantined | ✓ | ✓ | 場合による | ✗ | ✗ | △ Threading:84（handle 解除のみ） | ✓ | ✗ | ✗ | **latent（要 B で意味論確認）** |

凡例: DSM=DSPLifetimeManager、Orch=RuntimePublicationOrchestrator、RD=AudioEngine.RebuildDispatch、
Commit=AudioEngine.Commit、Threading=AudioEngine.Threading。

**注記（#14）**: `quarantineSlot`（Threading.cpp:63-84）も `retireDSPHandleForRuntime` 直呼びで
destroy enqueue を行わない。quarantine resident として意図的保持の可能性があるため、本 report では
H1 主因と断定せず **latent 同型リスク**として D162-2-B の意味論確認対象に挙げる（D162-1P soak では
quarantine=0 で不発）。

---

## 2. Phase A-3: Exact Call-Chain（現行コード基準・全件）

### 2.1 construction → ownership acquisition

```text
RebuildThreadLoop (AudioEngine.RebuildDispatch.cpp)
 ├─ RuntimeBuilder::build (src/audioengine/RuntimeBuilder.cpp:407)
 │   ├─ aligned_make_unique<DSPCore>            RuntimeBuilder.cpp:425
 │   ├─ DSPCore::DSPCore (DSPCoreLifecycle.cpp:57) — runtimeUuid 採番
 │   ├─ runtime->prepare(...)                   RuntimeBuilder.cpp:430
 │   │    └─ convolverState->prepare → ConvolverProcessor::prepareToPlay
 │   │        （固定バッファ確保 = D162-1R-B [CONV_FOOTPRINT] 100MB・Lifecycle.cpp:301-365）
 │   └─ 失敗時: result.runtime 未 release → unique_ptr で破壊（直destroy 同等）
 ├─ dspGuard.ptr = buildResult.runtime         RebuildDispatch.cpp:1208
 │   （DSPGuard dtor = 未登録 DSP の直接破壊契約: RebuildDispatch.cpp:938-965
 │     retireDSPHandleForRuntime false → destroyDSPCoreNode。CAVEAT: 二重解放前例あり）
 └─ obsolete 判定 → continue（guard dtor で direct destroy）
```

### 2.2 publication

```text
RebuildDispatch.cpp:1369  dspGuard.ptr = nullptr（所有権を intent へ移譲）
RebuildDispatch.cpp:1387  enqueuePublicationIntentForRuntimeCommit(dspToCommit, task.generation, ...)
  （recovery 経路: :1060 / :1140 recoveryGeneration 付き）
AudioEngine.Commit.cpp:782  enqueuePublicationIntentForRuntimeCommit
 ├─ :804  registerDSPHandleForRuntime(newDSP)   AudioEngine.h:4325
 │        （handle map 登録 — ここで全 enqueue 済み DSP が registered になる）
 ├─ :860  runtimeOrchestrator_->submitPublishRequest(req)
 │    RuntimePublicationOrchestrator.cpp:357 submitPublishRequest
 │      └─ :40 trySubmitImpl
 │         ├─ :46 PublicationAdmission::evaluate (PublicationAdmission.cpp:6)
 │         │    shutdown:12 / staleGen:18 / notFinalized:22 / pressure:29-47 /
 │         │    fading → DeferredFadingActive:49-57 / Accepted:59
 │         ├─ [Rejected 時] :47-50 caller 処理（retire しない）
 │         ├─ [Accepted] world build :176（失敗→ :181 lifetime_.retire → EBR ✓）
 │         │   crossfade 再build :246（失敗→ :249 lifetime_.retire → EBR ✓）
 │         │   executor_.publish :280（失敗→ :291 destroyRolledBackDSP ✓）
 │         └─ [DeferredFadingActive] → submitPublishRequest switch :377-379 → enqueueDeferred
 ├─ submitPublishRequest switch（:374-433）:
 │    Accepted:375 / DeferredFadingActive:377→enqueueDeferred / RejectedStaleGeneration:380 /
 │    RejectedNotFinalized:389 / RejectedPressure:403 / RejectedShutdown:415 /
 │    RejectedPublishFailure:427 — **いずれの Rejected* ケースも newDSP の処分を行わない**
 └─ enqueueDeferred（Orchestrator.cpp:437-543）
      ├─ :460-468 overwrite 時: engine_.retireDSPHandleForRuntime(oldDSP)
      │    AudioEngine.h:4362 — handle map erase + :4389 requestReclaimHandle
      │    ★ destroy enqueue なし（EBR 所有権取得に到達しない）
      ├─ :497-510 dormant RetryExhaustedDiscard（同型・現行不発）
      └─ :513 deferredSlot_ = req（単一スロット・hasDeferred_=true）
```

### 2.3 retirement candidate creation → retirement → destruction（正常系と脱落系）

```text
[正常系 — published old DSP]
ISR CoordinatorLoop: PublishIntentHandler::handle
  (ISRRuntimePublicationCoordinator_ProcessIntent.cpp:148-150)
  └─ PublishExecutor::executePublish (RuntimePublishExecutor.h:20)
      ├─ authority.commit()（RuntimeWorldAuthority）:60-77
      ├─ bridge.retirePublishedRuntimeWorldNonRt(oldWorld) :76
      ├─ :104 transition.onPublishCompleted (DSPTransition.h:49)
      │    ├─ :61/:97 lifetime.activate(newDSP)（= registerDSPHandleForRuntime）
      │    ├─ :71/:104 dspHandleRuntime_.activate(newHandle)
      │    ├─ :79/:82/:129/:153 lifetime.retire(oldDSP)
      │    │    DSPLifetimeManager::retire (DSPLifetimeManager.cpp:40)
      │    │      ├─ :45 engine_.retireDSPHandleForRuntime(dsp)（台帳解除）
      │    │      └─ :58 router_->enqueueWithRetry(dsp, destroyDSPCoreNode, epoch)
      │    │          ★ ここで初めて EBR が DSPCore の破壊所有権を取得
      │    │            (ISRRetireRouter.cpp:303 / 不変式「never returns with ptr unowned」:25)
      │    └─ :119 beginCrossfade（fading 成立時）
      └─ :110 advanceRetireEpoch
[fading 完了] AudioEngine.Timer → DSPLifetimeManager::retire → 同上（EBR）
[Observe 経路] ObserveIntentHandler (ProcessIntent.cpp:96-105)
  → lifetimeMgr.retireByHandle (DSPLifetimeManager.cpp:79) → :121 enqueueWithRetry（EBR）✓

[脱落系 — non-published DSP の 4 sub-path]
(S1) deferred overwrite: enqueueDeferred :460-468 → retireDSPHandleForRuntime のみ
     → **EBR enqueue なし → destroyDSPCoreNode 不発 → orphan**（D162-1P: 48 件と推定）
(S2) deferred discard: processDeferredAdmission :734-738 → view->discard(reason)
     (Orchestrator.cpp:686-692: lastDiscardReason 記録 + finishView のみ)
     → retire/destroy なし → orphan（登録済みのまま残る）
(S3) shutdown clear: clearDeferredForShutdown :546-570（slot reset のみ）
     → retire/destroy なし → orphan（呼出元: ReleaseResources.cpp:359、
       RebuildDispatch.cpp:915 経由の drainDeferredClearIfRequested :596-607）
(S4) admission Rejected*: submitPublishRequest switch :380-432 — 処分なし → orphan（latent）

[reclaim 系の補足]
requestReclaimHandle (AudioEngine.h:4378-4404) → requestReclaim (Coordinator.cpp:668)
  → reclaimNormal (Coordinator.cpp:684-713): 「:706 物理削除は retire path の enqueueWithRetry が担当」
  = **reclaim は slot 状態遷移のみで DSPCore を破壊しない**（drainDeferredRetireQueues
    AudioEngine.Retire.cpp:45 も同様 — pendingReclaimHandles_ の再試行は reclaim のみ）。
  よって S1 で handle が解除されても DSPCore はどこからも参照されない orphan として残留する。
```

### 2.4 thread 契約（修正制約として重要）

- `enqueuePublicationIntentForRuntimeCommit` の呼出元は全て RebuildThread
  （RD:1060/1140/1387）。`submitPublishRequest` / `enqueueDeferred` / `processDeferredAdmission`
  （:702 jassert RebuildThread）も RebuildThread。
- **Audio Thread の `trySubmit` は現行 production 呼出元 0**（定義 Orchestrator.cpp:34 のみ・
  コメント上の歴史経路）。よって脱落点の修正に RT 制約は発生しない（B-I3 RT boundary に触れない）。
- `clearDeferredForShutdown` は Message/RebuildThread 以外からも呼ばれ得る
  （ReleaseResources.cpp:359 EmergencyDrain / C1 fallback — D135-8 Gate A 審査済みの assert-free 契約）。

---

## 3. H1 Root-Cause Verdict

### 3.1 判定: **H1-E（ownership-handoff truncation＝「半分だけの retire」）**

指定枠との整合:

| 枠 | 判定 | 根拠 |
| --- | --- | --- |
| H1-A「retire が呼ばれていない」 | **部分一致だが不正確** | `DSPLifetimeManager::retire` は呼ばれていない。しかし S1 では `retireDSPHandleForRuntime`（handle 解除）は呼ばれている。「retire candidate にされなかった」わけではない |
| H1-B「retire は呼ばれるが handle lookup が失敗」 | **否定** | 49 DSP は Commit.cpp:804 で registered。S1 の lookup は成功する（erase される） |
| H1-C「EBR enqueue まで行くが reclaim されない」 | **否定** | EBR enqueue 自体が発生しない。D162-1P で H2(Router backlog) REFUTED 済み・pend=0/ovf=0/rec 単調増加。C を安易に選ばない指示どおり |
| H1-D「destroy は行われるが別の DSP が残る」 | **否定** | destroy 11 = placeholder + replaced published 10 で算術閉包（61−11=50 live）。D162-1P §1.1 と一致 |
| **H1-E「上記以外」** | **採用** | **retire candidate 化 → 台帳解除（S1）または黙示 slot 破棄（S3）まで進むが、破壊権移譲（EBR enqueue）を伴わないため DSPCore が single-owner model の全 terminal path から脱落する** |

本質的記述: 現行設計は「DSPCore 破壊」を (a) EBR 経由（`DSPLifetimeManager::retire` =
台帳解除 **＋** EBR enqueue の 2 段を 1 callee に束ねた唯一の完全 retire）または (b) 未登録 DSP の
直接破壊（DSPGuard 契約）に集約している。しかし deferred 系が **(a) の前半（台帳解除）だけを
直接呼ぶ** という第 3 の path を事実上作ってしまっており、そこで所有権移譲が途切れる。
`retireDSPHandleForRuntime`（AudioEngine.h:4362）の header comment（:4323「代わりに
retireDSPHandleForRuntime() を直接使用すること」）と :4349-4354 の注释（「DSPCore* の物理削除は
呼び出し元が enqueueWithRetry で行う」）が、この直接呼びを「完全 retire」と誤認して使う誘発を
生んでいる（API 名は retire だが機能は台帳解除のみ）。

### 3.2 49 件の内訳（D162-1P soak 実ログによる）

| sub-path | 件数（推定） | 証拠 |
| --- | ---: | --- |
| S1 deferred overwrite（handle-only retire） | **48** | re-defer (new) ×49 = 各 gen が deferred に入った事実。published 11 のうち gen5/6 は placeholder/gens5 が old で処理され、gen12..60 が前 published を置換。非 published 49 のうち最終 slot 残留 1 を除く 48 が次 gen の enqueueDeferred で overwrite された。overwrite retire は D117 ログ site を持たない（DSPLifetimeManager 経由でないため黙示）→ D117_DESTROY=11 と整合 |
| S3 shutdown clear | **1** | 最終 gen（64）の request が slot に残留 → shutdown 時 clearDeferredForShutdown（slot reset のみ・ログ site なし） |
| S2 / S4 | 0 | StaleDiscard ログ 0・TTL 30s > gen 周期 ~7s のため overwrite が先に発生。RejectedPressure/Shudown は soak 中不発 |

整合検算: 総 DSPCore 61 = placeholder 1 + 60 gen。destroy 11。live 50 = published active 1 +
replaced 待ち 0 + **orphan 49**。D162-1P MEM_SNAP DC live=50、D162-1R-B [DSP_FOOTPRINT] retained
142.48MB × 49 ≒ private 増加と一致。

### 3.3 「retire candidate になったのか」への回答（Phase A-2 の要求）

**なった。** ただしその retire は完全 retire ではない。S1 の overwrite branch は D132
(INV-DEFERRED-2) として「旧 deferred DSP を retire する」意図で書かれているが、呼んでいるのが
`engine_.retireDSPHandleForRuntime`（台帳解除のみ）であるため、意図（retire）と効果（参照切断のみ）
が不一致。S2/S3/S4 はそもそも retire 呼出自体が無い（candidate 化の宣言すら無い）。

---

## 4. Phase A-4: 修正案比較（実装なし・比較のみ）

前提: 49 DSP は **handle registered** である点（Commit.cpp:804）と、未 publish = どの published
World からも参照されない点が共通条件。

| 案 | 概要 | 長所 | 短所・リスク | 評価 |
| --- | --- | --- | --- | --- |
| **A: publish 時に全 DSP を handle 登録し、retire を既存 EBR 経路へ統一** | 登録は既に全件実施済み（Commit:804）。実質「全 DSP の破壊を EBR へ」 | 単一 authority | 未 publish DSP の意味論変更の可能性が案内に指摘されているが、実は既に登録済みであり A の新規性が薄い。すべてを EBR 化すると直接破壊契約（DSPGuard）とdomain が曖昧化 | △ 課題設定が現状とズレている |
| **B: non-published/obsolete DSP 専用の直接 destruction を明示保証** | deferred discard / shutdown clear / rejected で「未 publish = RT 参照なし」を根拠に直接 destroy | EBR を経ない即時回収・overflow ring 等への負荷ゼロ | (1)「未 publish = RT 参照なし」の証明責任が毎回発生する（published world の dspProjection / fading slot からの到達可能性を全部列挙する必要）。(2) S1 の overwrite は D132 として既に「EBR 経由も可」の前例（world-build-fail Orch:181/249 は lifetime_.retire = EBR）と整合しない。(3) EBR と direct の 2 系統判断が call-site 毎に分散し INV-D162-3 違反（二重破壊）の温床になる — DSPGuard CAVEAT（二重解放前例）が示す通り | △ 境界判断が分散し危険 |
| **C: 単一 retire authority への再構成** | 「registered DSP を手放す全 call-site」を `DSPLifetimeManager::retire`（台帳解除＋EBR enqueue）に統一。未登録 DSP のみ DSPGuard 直接破壊（既存契約）を維持 | (1) 2 値規則が単純: registered → EBR / unregistered → direct。(2) S1 は「意図は既に retire」なので authority 経由に寄せるだけ＝意味論不変。(3) epoch-safe が構造的に保証（INV-D162-5 自動充足）。(4) 全脱落点 4 か所＋quarantine 1 か所が同一 pattern で直る | (1) EBR queue への流入増（49 DSP/60 gen ≈ ~0.7/min — 実測 pend 最大 1 の router に余裕）。(2) `DSPLifetimeManager` は一時 object（`DSPLifetimeManager lm(*this)` 使用 pattern）だが state を持たない so 構築コスト僅少。(3) AudioEngine.h:4323 の誘発的 comment の是正がセットで必要 | **◎ 推奨（C′）** |
| **C′変形: Orchestrator に薄い disposition helper（`retireDeferredDSP(req)`）** | C と同内容を Orchestrator 内 1 関数に集約（内部で DSPLifetimeManager::retire、null-safe） | call-site 4 か所が 1 行ずつ・監査容易 | 新規 helper の placement 論争が僅か | ◎（C の実装形態として最良） |
| **D: deferred slot 自体に ownership を持たせる（slot 破壊時に必ず disposition）** | finishView/discard/clear に disposition hook | slot lifecycle に閉じた設計 | finishView は RebuildThread jassert 契約・clear は非 RebuildThread からも呼ばれる → hook の thread 契約が複雑化。consume（Ready→再 submit）では disposition 不要で hook の分岐が増える | ○ C′より複雑 |
| **E: Rejected switch の各 case に disposition 追加** | S4 のみ個別修正 | 最小差分 | S1/S2/S3 を残し H1 を解消しない。部分修正にすぎない | ✗ 不十分 |

**推奨**: **案 C′**。理由: (i) D162-1P/R-B の全実測が「49 = registered & unpublished」で一致しており、
EBR 経由の epoch-safe destroy が既存不変式（INV-EPOCH-1/2）とそのまま整合する。(ii) direct destroy を
新規 call-site に広げない（INV-D162-3 の二重破壊 risk を増やさない）。(iii) quarantine（#14）も同型で
直る候補として残る（ただし quarantine resident 意味論の確認を B で先行実施）。
**A を正解と仮定しない**という指示に従い、A は評価のみ（△）で採用していない。

---

## 5. Repair Contract（invariant 定義・既存用語準拠）

現行プロジェクトの用語（EBR / enqueueWithRetry / RetireRouter / reclaim / DSPGuard / handle map /
OwnerChannel / epoch-safe）に合わせて定義する。番号は指示の INV-D162-1..5 に対応。

```text
INV-D162-1（単一 terminal path — Every DSPCore has exactly one terminal ownership path）
  すべての DSPCore は、その破壊を正確に 1 回実行する terminal path を 1 つだけ持つ:
    (T1) EBR 経路: DSPLifetimeManager::retire / retireByHandle → enqueueWithRetry
         （ISRRetireRouter.cpp:25 の不変式「enqueueWithRetry() never returns with ptr unowned」）
    (T2) direct 経路: 未登録 DSPCore（handle map に不在）に対する destroyDSPCoreNode 直接呼出
         （RebuildDispatch.cpp:938-965 DSPGuard 契約 / :1036/:1118 / Orchestrator.cpp:291）
  registered DSPCore が T1/T2 以外の経路で参照を失うことを禁止する
  （S1-S4 = 本契約違反の現行 site）。

INV-D162-2（eventual destruction — no longer reachable ⇒ eventually destroyed）
  handle map から参照されず、いずれの RuntimePublishWorld（active/fading/pending registry 含む）
  にも topology に現れない DSPCore は、最終的に T1 または T2 のいずれかへ到達しなければならない
  （waitForDrain の完全性条件に包含される）。
  deferred slot に保持されている間は「reachable（pending publish candidate）」とみなす。

INV-D162-3（所有権二重化の禁止 — no EBR-owned AND directly-destroyed）
  enqueueWithRetry に渡された DSPCore に対する destroyDSPCoreNode 直接呼出を禁止する
  （DSPGuard CAVEAT の 0xC0000005 前例 = 二重解放）。逆に handle map に存在する DSPCore を
  retireDSPHandleForRuntime だけで処理し終えること（T1 の前半のみの実行）も禁止する —
  「台帳解除」と「破壊権移譲」は DSPLifetimeManager::retire の 1 callee として同時に実行する。

INV-D162-4（RT ownership semantics 不変）
  本修正は NonRT publish/retire path（RebuildThread / CoordinatorLoop / shutdown drain）のみに
  適用する。Audio Thread の RT path（Add/Get/observe 読み取り・active/fading slot の参照契約・
  B-I3 RT boundary）は一切変更しない。Audio Thread trySubmit は production 呼出元 0 であることを
  前提に、将来復活させる場合も D162-2-A の thread 契約表（§2.4）を更新してからに限る。

INV-D162-5（published 破壊の epoch-safe）
  published（または一度でも published world に現れた）DSPCore の破壊は T1（EBR・epoch gate 付き）
  のみで行う（INV-EPOCH-1/2 準拠）。未 publish DSP を T1 で破壊することは epoch-safe であるため
  妨げられない（over-conservative だが安全側）。
```

追加の現行整合事項（修正時に触れてはならない既存契約）:

- D132 (INV-DEFERRED-2) overwrite retire の意図は維持する（経路を authority に差し替えるのみ）。
- D135-8/F6 の deferred accounting（retention ≠ retry）、clearDeferredForShutdown の assert-free 契約、
  drainDeferredClearIfRequested の RebuildThread-only 契約は不変。
- recovery obligation accounting（D105-R5-9 / R18 / R20）は本修正と独立・不変。

---

## 6. D162-2-A 完了後の分岐判定

| 指定の分岐基準 | 本監査の結果 |
| --- | --- |
| H1 root cause と terminal owner が一意 | **該当** — root cause = deferred 系 4 sub-path の disposition 欠落（単一の所有権移譲欠落 class）、terminal owner = T1（EBR）/ T2（direct）の 2 値で一意に定義可能 |
| 複数の retire authority が競合 | 否 — authority は既に単一（DSPLifetimeManager::retire）。問題は bypass site の存在 |
| 未 publish DSP の扱いが未定義 | 部分 — 「unpublished でも registered」の中間状態の扱いが未文書化だった点は本 report の INV-D162-1/3 で定義済み |
| handle lookup が主因 | 否（H1-B 否定） |
| retire 未呼出が主因 | 部分（S2/S3/S4）だが S1 は「呼ぶが不完全」— 全体としては disposition 欠落 class として統一 |
| EBR 側の問題を再発見 | 否 — Router 不変式 (:25) と reclaim の責務分離 (:706) は正常動作。D162-1P H2 REFUTED を再確認 |

**→ 次工程: D162-2-B Repair Implementation（案 C′ を入力として実装設計へ）**

---

## 7. D162-2-B への引き継ぎ事項（実装時の必須確認リスト）

1. S1（Orchestrator.cpp:460-468）/ S2（:734-738）/ S3（:546-570）/ S4（:380-432）の 4 site +
   dormant :497-510 の disposition 追加（案 C′）。
2. AudioEngine.h:4323 の comment 是正（「直接使用すること」→「台帳解除専用・破壊は
   DSPLifetimeManager::retire 経由」）と AudioEngine.h:4362 側への契約注記。
3. quarantineSlot（Threading.cpp:84）の quarantine resident 意味論確認 — 意図的保持なら
   INV-D162-2 の例外として明文化、orphan なら同修正。
4. EBR 流入増の実測（pend/ovf/rec の許容確認 — 実測 pend 最大 1 に対し ~0.7/min 増は微小）。
5. 検証: D162-1P と同一条件 soak で `[D117_DESTROY]` が 60 + 1（全 gen + placeholder）に一致し、
   MEM_SNAP DC live が定常 1〜2 に収束すること。D162-1R-B の [DSP_FOOTPRINT]/[DSP_DESTROY_FOOTPRINT]
   による footprint 回収の 1:1 確認。
6. shutdown 時 S3 修正により clearDeferredForShutdown の thread 契約（非 RebuildThread 呼出可）
   を壊さないこと（enqueueWithRetry は NonRT-safe・mutex あり — B-I3 前提の再確認）。

---

## 8. 禁止事項の遵守確認

production source 修正なし / LiveAllocRegistry 追加なし / instrumentation 追加なし / allocation サイズ・
block size・Convolver バッファ・NUC・AoS/SoA・EBR capacity・queue size・crossfade 設計・DSPTransition・
shutdown UAF の修正 — **すべて未実施**。本 report は既存証拠（D162-1P soak log / D162-1R-B soak log）と
現行ソースの読解のみで構成されている。
