# D162-2-C — Shutdown Teardown / Residual DSP Ownership Audit（read-only）

- Work item: D162-2-C（read-only root-cause audit・production source 変更 **0**）
- Date: 2026-09-03
- 基準ソース: D162-2-B 適用後の working tree（ConvoPeq.md `Generated: 2026-09-03 03:19:12`）
- 判定: **C-F（複数 ownership authority / hidden reference が存在し、追加監査が必要）** — C-A〜C-E の
  単独証明に至らなかった根拠と、確定した部分事実を以下に記録する（§12）。
- 成果物: 本ドキュメント（指示の 1 本化形式）

---

## 0. Executive Summary

1. **0xC0000005 は「破壊した瞬間」の AV ではなく、heap corruption の後発表面（delayed surface）である。**
   7 件のクラッシュダンプ全てで faulting RIP は `ntdll.dll + 0x161914`（ヒープ free 経路・同一命令）、
   かつ RWDI 60-gen クラッシュ run の faulting thread stack には
   `PublicationAdmission::evaluateDeferred` / `RuntimePublicationOrchestrator::finishView` /
   `DeferredPublishSlot::_Assign` / `AudioEngine::DSPCore::DSPCore`（ctor）が並んでおり、
   **RebuildThread の deferred-admission 経路のアロケーションが破損ヒープに当たって落ちている**。
   すなわち「shutdown で DSP を破壊したから落ちた」のではなく「破壊が heap を破損し、後続の
   alloc/free が不運にもそれに接触した」形である。
2. したがって S3/V-D の「破壊を安全な順序へ移動する」（C-A）だけでは不十分。破損の作成者
   （corrupting free）を先に特定する必要がある。本監査で corrupting free の特定には至らなかった
   （理由: RWDI/FPO 最適化ビルドのダンプから完全 unwind が不可能・詳細 §9）。
3. **確定した部分事実（本監査の実質的成果）**:
   - shutdown teardown の完全順序表（§1・実コード行番号付き）
   - member destruction order と依存方向の静的列挙（§2/§5）
   - S3: slot reset 後も deferred DSP は `runtimeDSPHandleMap_` に**残存**する（map erase は
     `retireDSPHandleForRuntime` か rollback のみ — §3）。よって将来 S3 を再有効化する際、
     `retireDSPHandleForRuntime` ベースなら二重処分は構造的に不可。
   - V-D: `dspHandleRuntime_.retire + tryShutdownQuiescentReclaim` 後も handle map 残存（§4）。
   - residual gen 9/35/41/53 は **S1/S2 の未網羅窓口**であり観測分類の不完全ではない（§11）:
     re-defer(new) を 1 回発行した後、retain/discard/overwrite-retire のいずれのログも残らず
     slot から消失している。
   - address reuse は clean-exit run でも 1 件発生しており、**reuse 単独では crash しない**
     （crash 条件は reuse＋α。§10）。
4. D162-1R-B 時点で ASAN が指摘した既存 teardown UAF（`EQCacheManager::CacheMap` dtor →
   `tryShutdownQuiescentReclaim` → `resolve` → delete、AudioEngine.h:2172 現行）は本監査でも
   構造的に確認された（§5）。これは S3/V-D 破壊とは独立の pre-existing 課題であり、
   **C-B/C-C の候補領域と重なる**。

---

## 1. Current shutdown call graph（実コード順序・行番号）

### 1.1 `AudioEngine::releaseResources()`（AudioEngine.Processing.ReleaseResources.cpp）

| # | 行 | 処理 | thread | DSPCore ownership への作用 |
|---|---:|---|---|---|
| 1 | :127 | `setShutdownPhase(StopAudio)` | Message | — |
| 2 | :135 | `DSPLifetimeManager lifetimeForShutdown(*this)` 生成 | Message | — |
| 3 | :138-197 | rebuildMutex 下: `activeToRelease = getActiveRuntimeDSP()`(:150)・slot null 化(:152)・fading CAS 取得(:156-164)・pendingTask.currentDSP 取得(:172)・**idle publish #4**(:187) | Message | slot 切離しのみ・解放なし |
| 4 | :201-202 | `shutdownCoordinatorLoop()` join → `stopRebuildThread()` join | Message | producer 停止（RebuildThread 死滅） |
| 5 | :233 | `m_epochDomain.closeReaderRegistration()` | Message | reader 新規登録を永久遮断 |
| 6 | :244-305 | graceful drain（最大 5s: publishEpoch + tryReclaim + OverflowRing 再注入 + `drainDeferredRetireQueues(false)`） | Message | EBR 内 destroy 実行（epoch 安全分のみ） |
| 7 | :330-336 | `lifetimeForShutdown.retire(activeToRelease/fading/pendingNew/pendingCurrent)` | Message | **注意: activeToRelease は activeRuntimeDSPSlot 由来**（§4 の dangling 問題） |
| 8 | :341-342 | `drainDeferredRetireQueues(true)` → `transitionTo(ReclaimComplete)` | Message | EBR 全 drain（shutdown 許可） |
| 9 | :350-438 | EmergencyDrain（要求時）: **`clearDeferredForShutdown()`**(:359)・tryReclaim(:363) | Message | S3 対象（現行: slot reset のみ） |
| 10 | :444-478 | **VerifyDrained**: `activeHandle/fadingHandle` 取得(:444-445) → `dspHandleRuntime_.retire`(:470/:477) → `tryShutdownQuiescentReclaim`(:473/:480) → D162-2-B で追加した resolve(:459-461)＋destroy（**無効化中**） | Message | slot 状態遷移のみ（map erase なし） |
| 11 | :487-494 | `uiConvolverProcessor.releaseResources()` / `uiEqEditor.releaseResources()` | Message | UI processor の engine 解放（forceCleanup 含む） |
| 12 | :503-508 | `requestShutdownClearNonRt()` → `clearPublishedRuntimeSnapshotsNonRt()` → `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` | Message | published World 解放（DSPCore* は非所有） |
| 13 | :523-524 | `drainAllQuarantineStore()`（quiescence 時） | Message | Q+E+T 強制 drain |
| 14 | :530-547 | D162-2-B が配置した最終 active/fading destroy block（**無効化中**） | Message | — |
| 15 | :562 | `drainPendingRetireIntentsForShutdown()` | Message | slot-state System1 残余 drain |
| 16 | :573-576 | drain timeout → `drainDeferredRetireQueues(true)` + `m_epochDomain.tryReclaim()` | Message | — |
| 17 | :586-598 | `m_coordinator.finalizeShutdown` → OwnerChannel residual `drainAllNonRt`（World 破壊 deleter） | Message | World のみ（DSPCore ではない） |
| 18 | :641 | `markShutdownComplete()`（Coordinator Faulted 化のログあり） | Message | — |
| 19 | :660-665 | `transitionTo(ShutdownComplete)` → `emitShutdownTrace` → `emitEvidenceTickNonRt(true)` | Message | — |
| 20 | :668 | `lifecycleRuntime_.leaveRelease(token)` | Message | — |

### 1.2 `~AudioEngine`（AudioEngine.CtorDtor.cpp）

| # | 行 | 処理 | 備考 |
|---|---:|---|---|
| D1 | :110-125 | phase 遷移（防御的再実行） | releaseResources 済みなら冪等 |
| D2 | :132-166 | `activeToRelease = getActiveRuntimeDSP()`(:149)・fading CAS(:155-165)・pendingTask.currentDSP retire(:174-179) | **activeToRelease は activeRuntimeDSPSlot 由来** |
| D3 | :186-190 | `lifetimeMgr.retire(activeToRelease/fadingToRelease)` | **dangling slot 値の map lookup が行われる**（§4.2） |
| D4 | :203 | `shutdownWorkerThread()`（learner worker join） | NoiseShaperLearner jthread join |
| D5 | :205-236 | FORCE_EPOCH_ADVANCE → graceful drain（5s poll: publishEpoch + tryReclaim） | EBR 残破壊実行 |
| D6 | :240-246 | `worldAuthority_.requestShutdownClearNonRt()` → `clearPublishedRuntimeSnapshotsNonRt()` → `retirePublishedRuntimeWorldNonRt(clearedWorld, true)` | World 破壊（deferred） |
| D7 | :247-253 | `drainDeferredRetireQueues(true)` → `drainPendingRetireIntentsForShutdown()` | — |
| D8 | :255-265 | quiescence 時 `m_retireRouter->drainAll()` / 非時 `m_epochDomain.drainAll()` + `drainAllQuarantineStore()` | **全 EBR エントリ破壊実行点** |
| D9 | :268 | `runtimePublicationBridge_.markShutdownComplete()`（Faulted 診断ログ） | — |
| D10 | :280-284 | `latencyBufOldL/R/NewL/R` aligned_free（engine 保有 4 buffer ≈23.5MB） | DSPCore のものではない |
| D11 | :287 | `setShutdownPhase(Destroy)` → body 終了 | **以降 member teardown** |

### 1.3 member destruction（AudioEngine.h 宣言順の逆順 → 破壊順）

宣言行（昇順）→ 破壊順（降順）:

```text
5061 runtimeDSPHandleMap_   (DSPHandleTable)
5056 dspHandleRuntime_      (DSPHandleRuntime)
5029 m_healthMonitor
5022 shutdownRuntime_       ← tryShutdownQuiescentReclaim の Permit 源（先に死ぬ）
5017 dspQuarantineManager_
4988 worldAuthority_        (RuntimeWorldAuthority — lifetime()/ownerChannel()/registry() 含む)
4986 runtimePublicationBridge_ (RuntimeIntentCoordinator)
4982 lifecycleRuntime_
4932 pendingReclaimHandles_ (vector<ReclaimIdentity>)
4849 m_workerThread         (convo::WorkerThread — learner command pump)
4842 m_coordinator
4835 m_retireRouter         (ISRRetireRouter — EBR 本体)
4830 m_epochDomain
3693 runtimeOrchestrator_
2790 noiseShaperLearner     (jthread worker + AudioSegmentBuffer)
2444 eqCacheManager         (EQCacheManager — CacheMap dtor が shutdownPhase >= Destroy で
                             tryShutdownQuiescentReclaim → resolve → delete EQCoeffCache)
2381 crossfadeRuntime_
2223 analyzerFifo
2221 uiEqEditor
2220 uiConvolverProcessor   (ConvolverProcessor — ~dtor で forceCleanup + engine 解放 +
                             deferredFreeThread drain。rcuProvider/Router は既に破壊済み)
```

**重要な順序逆転（C-C/C-B 候補の構造的事実）**:

- `shutdownRuntime_`(:5022) は `worldAuthority_`(:4988) / `m_retireRouter`(:4835) より**先に**破壊される。
  → member teardown 中に `tryShutdownQuiescentReclaim`（Permit を ShutdownRuntime から製造）を
  呼べる対象は `shutdownRuntime_` 破壊前に死ぬ `worldAuthority_`/`m_retireRouter` に限定される。
  実際には member teardown でそれらを呼ぶコードはない（CacheMap dtor は `shutdownPhase` 读取のみ —
  ただし `shutdownPhase` は `shutdownRuntime_` のメンバ**ではなく** `AudioEngine::shutdownPhase`
  （AudioEngine.h:2707 の exchangeAtomic 対象・別 atomic）であるため CacheMap dtor 自体は成立する）。
- `eqCacheManager`(:2444) は `m_retireRouter`(:4835) / `worldAuthority_`(:4988) / `dspHandleRuntime_`
  (:5056) より**後に**破壊される。CacheMap dtor が `owner->dspHandleRuntime_` と
  `owner->tryShutdownQuiescentReclaim`（内部で `m_retireRouter` / `m_epochDomain` /
  `runtimePublicationBridge_` を参照）に触れるため、**eqCacheManager の破壊時点でこれらが生存して
  いることは宣言順から保証される**（5056/4988/4835 は 2444 より後 = 破壊は先）。OK だが、
  `shutdownRuntime_`(5022) は 2444 より後 = 破壊は先 → **CacheMap dtor 内の
  tryShutdownQuiescentReclaim は shutdownRuntime_ 生存前提**であり、member teardown の順序では
  `shutdownRuntime_` はまだ生存（5022 は 2444 より後ろ→破壊は先…逆。宣言 5022 > 2444 →
  破壊順は 5022 が先）。**訂正**: 破壊は宣言降順なので 5022 shutdownRuntime_ は 2444 eqCacheManager
  より**先に**破壊される → CacheMap dtor 時点で `tryShutdownQuiescentReclaim` が
  `shutdownRuntime_.tryMakeQuiescenceProof` に触れると **UAF**。
  ただし CacheMap dtor は `shutdownPhase >= Destroy` の場合のみ reclaim 系を呼ぶ。
  `AudioEngine::shutdownPhase`（:2707 の atomic、ShutdownRuntime とは別物）は D11 で Destroy に
  設定済み → **CacheMap dtor は Destroy branch を通り tryShutdownQuiescentReclaim を呼ぶ**。
  この時 `shutdownRuntime_`（member）は既に破壊済みの可能性がある → **構造的 UAF 経路が実在する**
  （ASAN が D162-1R-B で検出した signature と一致 — §5）。

### 1.4 DSPCore / MKL 破壊の実行点まとめ

| 実行点 | 経路 | 対象 |
| --- | --- | --- |
| graceful drain / drainAll（D5/D8、releaseResources #6/#8/#13） | EBR enqueue 済み destroyDSPCoreNode | published 置換 DSP・S1/S2 retire DSP |
| RebuildThread 中の EBR drain（運転中） | 同上 | S1 overwrite/discard retire DSP |
| `destroyRolledBackDSP`（trySubmitImpl 失敗時） | direct | publish 失敗 DSP |
| DSPGuard dtor / recovery warmup fail | direct | 未登録 DSP |

MKL 由来 free: `aligned_free` → `mkl_free`（JUCE_DSP_USE_INTEL_MKL 時）、NUC 内部は
`DIAG_MKL_FREE(allocSizes)`、IPP は `ippsFree`（mkl ではない）。

---

## 2. Thread / ownership matrix

| 領域 | owner（唯一の処分権） | 触る thread | 備考 |
| --- | --- | --- | --- |
| runtimeDSPHandleMap_ | mutex 保護・erase は `retireDSPHandleForRuntime`(h:4389) と rollback(h:4577) のみ | Message/Rebuild/CoordinatorLoop | V-D の `dspHandleRuntime_.retire(handle)` は **map を erase しない**（registry 状態のみ） |
| DSPHandleRuntime registry | CAS（create/retire/reclaim/quarantine/rollback） | 同上 | resolve は Reclaimed/Quarantined を拒否 |
| deferredSlot_ | RebuildThread 単一所有（jassert）＋ clearDeferredForShutdown は assert-free | Rebuild / Message | slot reset は map に触れない |
| published World topology | worldAuthority_（immutable 公開） | RT read / NonRT write | DSPCore* を保持するが非所有 |
| EBR queue (ISRRetireRouter) | enqueueWithRetry で ownership 取得（h:25 不変式） | producer 複数 / drain は Message | destroy は drain 時 |
| EQCoeffCache | EQCacheManager（DSPHandle で間接管理） | Message/RT | CacheMap dtor が 2 経路（§5） |
| pendingReclaimHandles_ | mutex + RebuildThread drain | 複数 producer / RebuildThread drain | reclaim のみ・破壊なし |

---

## 3. S3: deferred DSP exact lifecycle（破壊を実装しない現行挙動）

```text
DSPCore constructed            (RuntimeBuilder.cpp:425 — aligned_make_unique)
    ↓
registered                     (Commit.cpp:804 — registerDSPHandleForRuntime → map 登録, state=Constructing)
    ↓
deferred slot owner            (submitPublishRequest → DeferredFadingActive → enqueueDeferred:
                                Orchestrator.cpp:377-379 → :513 deferredSlot_ = req)
    ↓ [overwrite]              次の gen の enqueueDeferred で S1 retire（D162-2-B で実装・動作実績あり）
    ↓ [discard]                processDeferredAdmission → evaluateDeferred → view->discard → S2 retire
    ↓ [shutdown clear]         clearDeferredForShutdown（Orchestrator.cpp:546-570・EmergencyDrain :359）
    ↓
slot owner 消滅                 deferredSlot_.reset()（:555相当）— request(=handle) は消失
    ↓
handle map は？                 ★残存する★ — slot reset は map に触れない。
                                map erase は retireDSPHandleForRuntime(h:4389) と rollback(h:4577) のみで、
                                現行 clearDeferredForShutdown はどちらも呼ばない（D162-2-B で無効化済み）。
    ↓
World topology は？             未 publish のため、いずれの RuntimePublishWorld にも出現しない（零参照）。
    ↓
EBR は？                        enqueue されていない（破壊権は谁にも移譲されていない）。
    ↓
最終 owner は？                 ★なし★ — handle map が state=Constructing のまま最後の参照を保持するが、
                                ~AudioEngine のどの経路もこの map エントリを解決・破壊しない。
                                = INV-D162-2 の技術的例外（process exit 時 OS 回収）。
```

**二重処分判定（将来 S3 を再有効化する場合）**: map 残存が保証されるため、
`retireDSPHandleForRuntime` ベース（map erase を含む）なら「既に処分済みなら find 失敗で no-op」となり
二重処分は構造的に不可能。一方、**map erase を伴わない直接 destroy を先に実施した場合**、
map に残ったエントリが後続の何か（例: dtor の dangling retire・§4.2）によって再解決される危険が生じる
ため、S3 再有効化時は「map erase を含む経路で処分する」か「erase と destroy を同一 closure で行う」
ことが必須。

---

## 4. V-D: active/fading DSP exact lifecycle

### 4.1 正常系の参照関係

```text
Active World (RuntimePublishWorld topology) — DSPCore* を保持（非所有）
    ↓
RuntimeWorld clear                (releaseResources:503-508 requestShutdownClear → snapshots clear
                                   → retirePublishedRuntimeWorldNonRt(clearedWorld, true))
    ↓                              ※ D162-2-B の分離試験で「world clear 前破壊」も「clear 後破壊」も
    ↓                                同一 AV だったため、clear 順序は AV の要因ではない（§10）
active DSP resolve                (VerifyDrained :444 getActiveRuntimeDSPHandle → :459 resolveDSPHandle
                                   → state=Active なので instance 取得可)
    ↓
handle registry                   (:470 dspHandleRuntime_.retire(handle) — state Active→Retired。
                                   ★ runtimeDSPHandleMap_ は erase されない)
    ↓
tryShutdownQuiescentReclaim       (:473 — ShutdownRuntime Proof→Permit→reclaimShutdownQuiescent
                                   → handleRuntime.reclaim → state Reclaimed)
    ↓
DSPTransition / lifetime          関与なし（publish 完了済み）
    ↓
EBR                               現行 V-D は enqueue しない（D162-2-B で無効化）。
                                   将来再有効化時: retireDSPHandleForRuntime（map erase 含む）なら
                                   二重処分不可（S3 と同一の安全性議論）。
```

### 4.2 「world clear → destroy 間に DSP を参照する者はいるか」

静的に列挙した結果、**以下の参照者が解消されないまま残る**:

1. `runtimeDSPHandleMap_` エントリ（state=Reclaimed・instance ポインタ保持）— resolve は
   Reclaimed を拒否するため解決は不可能（参照としては不活性）。
2. `activeRuntimeDSPSlot` / `fadingRuntimeDSPSlot` — VerifyDrained の block では null 化されない
   （null 化は releaseResources 先頭 :152 / :156-164 と dtor :151-165 のみ）。**ただし
   activeRuntimeDSPSlot は bootstrap 以降 update されておらず placeholder の dangling 値を保持し続ける**
   （唯一の setter は PrepareToPlay.cpp:264。publish 時の activate は `dspHandleRuntime_.activate` のみで
   slot を更新しない — AudioEngine.h:2256 setter 呼出は 4 件のみ確認）。
3. `pendingReclaimHandles_` — reclaim 遅延時に `ReclaimIdentity{handle}` が残り得る（handle のみ・
   DSPCore* ではない）。
4. EQCacheManager — DSPHandle を保持（EQCoeffCache 用・DSPCore handle とは別 handle 空間）。
   DSPCore 自体は保持しない。
5. published World — clear 済み（RetireQuarantineStore 経由で遅延破壊。RuntimeState dtor は
   DSPCore* を deref しない設計）。

結論: **静的に列挙できる DSPCore 実参照は world clear 後に解消される。** したがって
「world clear したから destroy できる」は**参照の側面では成立する**。しかし実測（§10）では
destroy を入れた全バリアントで AV が発生しており、静的列挙で捕捉できない隠れ参照（hidden reference）
が存在することを示している。候補は §5/§6 の通り。

### 4.3 dangling activeRuntimeDSPSlot の構造的危険（本監査で確定した新規所見）

- `activeRuntimeDSPSlot` は bootstrap の placeholder ポインタで set され（PrepareToPlay.cpp:264）、
  placeholder が gen5 publish 時に EBR 破壊された後も **null 化されないまま dtor まで保持される**。
- dtor D3（CtorDtor.cpp:190）が `retire(activeToRelease)` を呼ぶ。`retireDSPHandleForRuntime` は
  **map.find をポインタ値で行う**ため、placeholder のアドレスが別の生存 DSPCore に再利用されていた場合、
  **生存 DSP の map エントリを erase し、その DSP を enqueue で破壊してしまう**（二重破壊の起点）。
- 実測: 最終 clean run でも 1 件の D→C address reuse を確認（§10）。placeholder address の再利用は
  今回のログでは観測されなかったため発火に至らなかったが、**run-to-run の AV 有無の揺れを説明する
  有力な構造的要因**である（S1 導入により破壊→再構築の address reuse 頻度が激増した）。

---

## 5. EQCacheManager dependency

```text
参照方向（静的列挙）:
  EQCacheManager → AudioEngine& owner（全寿命）
  EQCacheManager.cacheMapPtr → CacheMap*（atomic・fallback maps 含む）
  CacheMap.map → DSPHandle（EQCoeffCache 用。DSPCore の handle ではない）
  CacheMap dtor → owner->dspHandleRuntime_（resolve/retire）
  CacheMap dtor → owner->tryShutdownQuiescentReclaim（shutdownRuntime_ / m_epochDomain /
                  m_retireRouter / runtimePublicationBridge_ に触れる）
  EQCoeffCache → DSPCore を参照しない（plain data: EQCoeffsSVF[20] + metadata）
  DSPCore → EQCacheManager を参照しない
```

**確定した構造的 UAF 経路（§1.3 訂正ブロック）**: member 破壊は宣言降順のため
`shutdownRuntime_`(5022) は `eqCacheManager`(2444) より先に破壊される。一方 `AudioEngine::shutdownPhase`
（別 atomic・AudioEngine.h:2707）は dtor body の D11 で `Destroy` に設定済みのため、
`~CacheMap` は Destroy branch に入り `tryShutdownQuiescentReclaim` を呼ぶ。
この関数（h:4443-4478）は `shutdownRuntime_.tryMakeQuiescenceProof` / `tryMakeReclaimPermit` に触れる —
**`shutdownRuntime_` は既に破壊済み** → member teardown 中に CacheMap が空でなければ UAF。
ただし通常 shutdown では releaseResources #10 の CacheMap 置換フローで CacheMap は空になり得るため、
発火条件は「member teardown 時点で map 非空」。D162-1R-B ASAN が記録した同一 signature UAF
（EQCacheManager::CacheMap dtor）と整合する。**これは S3/V-D と独立の pre-existing 課題**であり、
shutdown destroy を入れた場合の heap corruption と混同してはならない（ただし corrupting free の
候補として排除もされていない）。

---

## 6. RuntimeWorld / rcuSwapper dependency

```text
RuntimePublishWorld/RuntimeState → DSPCore* (topology.current 等) — 非所有・参照のみ
  破壊: retirePublishedRuntimeWorldNonRt → enqueueDeferredDeleteNonRt → shutdown 時
        m_retireRouter->shutdownReclaim（Terminal）→ drainAllUnsafe で解放。
        RuntimeState dtor は DSPCore* を deref しない（静的確認済み）。

ConvolverProcessor.rcuSwapper / deferredFreeThread:
  DSPCore の ConvolverProcessor メンバ。~ConvolverProcessor で deferredFreeThread dtor →
  shutdownAndDrain（自キューの drain・thread join）。
  DSPCore 破壊時（destroyDSPCoreNode）にこの drain が走る — engine router には依存しない
  （StereoConvolver::retireStereoConvolver(sc, nullptr) 強制ローカル経路: Lifecycle.cpp:118-122
  「Destructor phase must not depend on rcuProvider」コメント付き）。
  → DSPCore 単独破壊の内部完結性は保たれている。

uiConvolverProcessor.rcuSwapper:
  AudioEngine member（:2220）→ 破壊は member teardown の最終盤。
  releaseResources #11 で forceCleanup()（:488）により active engine 解放済みのため、
  dtor 時の queue は空のはず（ただし assert なし）。
```

---

## 7. DSPHandle registry dependency

- `dspHandleRuntime_`(:5056) は `runtimeDSPHandleMap_`(:5061) より**先に**破壊される（宣言降順）。
- `runtimeDSPHandleMap_`（DSPHandleTable）は POD 的で dtor 依存なし。
- `EQCacheManager`(2444) は両者より先に破壊されるため CacheMap dtor の `dspHandleRuntime_` 参照は
  生存保証あり（ただし §5 の shutdownRuntime_ 逆転は別問題）。
- `pendingReclaimHandles_`(4932) は `m_retireRouter`(4835) より**後に**破壊される → Router dtor 後も
  vector 自体は破壊可能（ POD 構造体）。drain 経路が Router に触れない限り安全。

---

## 8. RetireRouter / EBR dependency

- `m_retireRouter`(:4835) は `worldAuthority_`(:4988) / `runtimePublicationBridge_`(:4986) より
  **先に**破壊される。
- dtor body 内 D8 の `drainAll()`（CtorDtor.cpp:261）は body 中（全 member 生存）で実行される —
  これは安全。
- **member teardown 中に Router の queue を触る経路**: `uiConvolverProcessor` の
  `retireStereoConvolver(sc, getRcuProvider())` は provider が非 null の場合
  `enqueueDeferredDeleteNonRt` → `m_retireRouter->enqueueWithRetry/shutdownReclaim` に触れる。
  しかし ~ConvolverProcessor は forceCleanup 後に `retireStereoConvolver(oldConv, nullptr)`
  （Lifecycle.cpp:122・ローカル破壊強制）を使用しており、Router に触れない設計（コメント明記）。
  **この防御は D162-2B で追加した DSPCore 側 destroy には存在しない**: `destroyDSPCoreNode` は
  EBR drain（dtor body D8 内）で実行されるため engine 全員生存 — こちらは安全側。
- `pendingReclaimHandles_` に Reclaimed 済み handle が残った状態で Router が死んでも、
  drainDeferredRetireQueues は dtor 内で完結（body 中）。

---

## 9. Crash dump stack analysis（C-2）

### 9.1 ダンプ群の特徴（7 件全て共通）

| 項目 | 値 |
| --- | --- |
| ExceptionCode | EXCEPTION_ACCESS_VIOLATION（0xC0000005） |
| Faulting RIP | **ntdll.dll + 0x161914（7 件全て同一命令）** |
| ExceptionInformation | [0 (READ), 可変アドレス] — 0xFFFFFFFFFFFFFFFF / 0x142... / 0x222... 等 |
| Faulting thread | メインスレッド（exit AV 系）または RebuildThread 系 |

同一命令への集中 = 特定のヒープ free/consolidation 経路が破損ブロックに触れていることを示す
（ランダムな deref ではない）。

### 9.2 RWDI 60-gen クラッシュ run（dump ConvoPeq.exe.24172 / .20688・exe 03:00 build と一致）

faulting thread stack（module range走査・llvm-symbolizer nearest symbol）に含まれる本プロジェクト
シンボル:

```text
convo::isr::PublicationAdmission::evaluateDeferred
convo::isr::RuntimePublicationOrchestrator::finishView
std::_Optional_construct_base<convo::isr::DeferredPublishSlot>::_Assign
convo::isr::RuntimePublicationStateOwner::onPublished
std::exchange<convo::FrozenRuntimeWorld*, nullptr_t>
AudioEngine::DSPCore::DSPCore (ctor)
（周辺に MKL DFT 内部シンボル多数 — mkl_dft_avx512_* / mklgDFTFwdBatch_64fc）
```

→ **AV 発生時点で RebuildThread が deferred-admission/publish 経路のアロケーション（DeferredPublishSlot
の optional 代入・DSPCore ctor の aligned_make_unique）を実行中**。すなわち破損ヒープへの接触が
「shutdown 後」ではなく「運転中の alloc/free」で起きている。dump .448（Debug・S3-EBR 時代）では
`mkl_serv_check_fast_memory_size`（mkl_free の所有ブロック照合）が faulting 関数 —
**破損は free 時に作られた**ことを補強。

### 9.3 unwind の限界

RWDI(/O2 /Ob1) は FPO のため minidump からの RSP 走査では正確な frame chain 再構築が不能
（nearest-symbol noise が JUCE GUI 系シンボルに多数混入）。Debug dump（448 等）は inline 情報付きで
部分復号に成功したが、当該 dump は S3-EBR 実験時代のビルドであり現行コードと 1:1 でない。
**corrupting free の呼出 site を特定するには D162-2-D で PageHeap/Application Verifier または
Debug ビルド縛りの短縮再現が必要**（production source 変更 0 で実施可能 — テスト環境ツールは
source 変更に含まれない）。

### 9.4 「MKL が悪い」ではない根拠

- faulting RIP は ntdll（Windows heap manager）であり MKL 内部ではない。
- `mkl_serv_check_fast_memory_size` は mkl_free が「ポインタが MKL 管理ブロックか」を照合する
  ルーチンであり、AV は照合対象メモリの読み出しで発生 → **free すべきでない/既に free 済みの
  ポインタが mkl_free に渡された**ことの帰結であり、MKL 自体の欠陥を示唆しない。
- D162-2B の新規破壊経路が free 対象を増やした時点で AV が顕在化した時系列（分離マトリクス、
  evidence/D162-2B_EVIDENCE.md §4）と整合。

---

## 10. S3/V-D isolation evidence（D162-2-B で実施済みの分離マトリクス再掲 + 本監査の補強）

| 試験 | S3 | V-D | destroys | exit | dump |
| --- | --- | --- | ---: | --- | --- |
| 初版（S1/S2/S4 + S3-EBR + VD-EBR） | EBR | EBR | 61 | 0xC0000005 | 23044/15388/22460 系 |
| VD 無効化 | EBR | off | 61 | 0xC0000005 | 24172/20688 |
| + S3 無効化 | off | off | 60 | 0x00000000 | — |
| + S3 direct destroy | direct | off | 7（active 残存） | 0x00000000 | — |
| + S3 direct + VD direct（world clear 後配置） | direct | direct | 7/7 | **0xC0000005** | 26632/21604 系 |
| 最終（S1/S2/S4 のみ） | off | off | 54 | 0x00000000 ×2 | — |

補強事実（本監査のログ再解析）:

- **AV は free ではなくその後の alloc で表面化**（§9.2）。よって「S3/V-D をどの順序で呼ぶか」の
  調整だけでは回避できず（world clear 後 direct でも再現済み）、corrupting free の同定が必須。
- **address reuse**: clean-exit run でも 1 件の D→C reuse（0x8DF080: destroy@56867 → construct@116036）
  を確認。reuse 単独では crash しない。crashing run での reuse 計測は log が上書き済みのため不能。
- **dangling activeRuntimeDSPSlot**（§4.3）: 構造的に実在。発火条件は placeholder address 再利用。
  確率論的で、run 毎の AV 有無の揺れと整合する。

---

## 11. Residual gen 9/35/41/53 tracing（C-6）

60-gen 最終 soak（evidence/D162-2B_soak.log・59 enqueued）における 4 件のライフサイクル:

| gen | construct | retained(re-defer new) | 以降の RETIRE/DISCARD/OVERWRITE/DESTROY |
| --- | --- | --- | --- |
| 9 | :19316 | :19481（1 回のみ） | **なし** — ptr 0xDCC080 に対する S1/S2/destroy ログ皆無 |
| 35 | :111597 | :111781（1 回のみ） | **なし** |
| 41 | :130321 | :130486（1 回のみ） | **なし** |
| 53 | :162627 | :162793（1 回のみ） | **なし** |

S1/S2 retired gens（43 件）: 5,6,7,8,11..61 のうち published 10 件と上記 4 件を除く全て。
published: 4,10,15,20,25,30,36,42,48,54,60。

**パターン**: gen 9 = published 4 の直後第 1 deferred gen。35 = 30 の直後。41 = 36 の直後。
53 = 48 の直後。すなわち **publish 直後の最初の deferred gen が漏れる**。60-gen で gen 5..8 が
漏れていないのは publish 4 が初回（placeholder 置換）だからと推定される。

**推定メカニズム（要 D162-2-D で確定）**: re-defer(new) 後、当該 gen の slot エントリが
retain も discard も overwrite も経由せず消失している。候補:
(a) `processDeferredAdmission` の consume（Ready）→ `submitPublishRequest` が
`RejectedStaleGeneration` 等で終わった際、S4 retire は呼ばれるはず（D162-2-B で実装）だが origin=
rejected-* が 0 件 → S4 経路を通っていない;
(b) enqueueDeferred の overwrite 分岐が「hasDeferred_==false で slot 残置」の状態窓を通った;
(c) D135 accounting の sameObligation 判定と slot 実体の不整合。
**いずれも推論であり、追加の 1 行 instrumentation（D162-2-D 許可範囲）なしには確定できない。**

---

## 12. Root-cause classification（C-A..C-F）

| 候補 | 評価 |
| --- | --- |
| **C-A** 破壊順序だけの問題 | **否定** — world clear 前後・EBR/direct の全順序バリアントで AV 再現（§10）。AV は破壊時点ではなく後続 alloc で表面化 |
| **C-B** EQCache/registry/RCU teardown ordering が主因 | **部分該当の可能性** — §5 の CacheMap dtor × shutdownRuntime_ 破壊順序逆転は構造的 UAF 経路として実在（ASAN 記録と一致）。ただし corrupting free との因果は未証明 |
| **C-C** AudioEngine member destruction order が root cause | **部分該当の可能性** — §4.3 dangling activeRuntimeDSPSlot（dtor retire が address reuse 時に生存 DSP を破壊）は実在する危険。placeholder reuse は今回未観測のため発火未証明 |
| **C-D** S3 と V-D は別々の root cause | **判定不能** — 両者を同時に無効化しないと clean exit にならない分離結果（S3 単独無効では VD-EBR で AV、VD 単独無効では S3-EBR で AV）からは「共通の下位要因（heap corruption の作成者が S3/V-D の destroy に共通する何か）」が示唆される。両者に共通するのは「DSPCore を shutdown 文脈で破壊すること」自体 |
| **C-E** residual 4 件が別の未発見 disposition path | **該当（S1/S2 カバレッジ問題として確定）** — §11。publish 直後第 1 deferred gen が無処分で消失 |
| **C-F** 複数 authority / hidden reference が存在し追加監査が必要 | **採用（最終判定）** — (i) corrupting free 未同定（§9.3 の unwind 限界）、(ii) §5 の teardown 順序逆転 UAF、(iii) §4.3 の dangling slot、(iv) §11 の S1/S2 網羅窓口、が並存。単一の C-A..C-E に還元できない |

**C-F 採用の理由（3 軸からの証明状況）**:

- ownership 軸: S3/V-D の静的参照列挙は完了（§3/§4）。「destroy できるはず」は静的には成立するが
  実測と矛盾 → hidden reference あり。
- dependency 軸: EQCacheManager × shutdownRuntime_ の順序逆転 UAF を新規に確定（§5）。
- teardown order 軸: 完全順序表を作成（§1）し、member teardown 中に reclaim 系 authority を
  触る経路の構造的危険を明示。

---

## 13. Required repair contract（D162-2-D 用）

既存 INV-D162-1..5 に加え、本監査で確定した事実に基づく追加契約:

```text
INV-D162-6（teardown 中の reclaim authority 呼出禁止）:
  AudioEngine member teardown が開始された後（dtor body 終了後）は、
  tryShutdownQuiescentReclaim / ShutdownRuntime の Permit 製造 / RetireRouter の enqueue を
  一切呼んではならない。CacheMap dtor を含む全 member dtor は、
  「shutdownPhase >= Destroy なら reclaim 系を呼ばず、permit 不要の静的解放または no-op」に
  限定しなければならない。（shutdownRuntime_ が eqCacheManager より先に破壊される宣言順序は
  変更しないことを前提とする。）

INV-D162-7（dangling slot 値の map lookup 禁止）:
  activeRuntimeDSPSlot / fadingRuntimeDSPSlot から取得した DSPCore* を
  retireDSPHandleForRuntime（map.find by pointer value）に渡してはならない。
  これら slot は publish 時に更新されないため placeholder の dangling 値を保持し得る
  （AudioEngine.h:2256 setter の呼出は bootstrap のみ）。
  dtor の retire は handle registry の state 遷移（dspHandleRuntime_.retire(handle)）か、
  resolve 成功（generation 検証付き）を前提に限定する。

INV-D162-8（shutdown DSP destroy は EBR 単経路）:
  shutdown 文脈での DSPCore 破壊は必ず enqueueWithRetry（EBR）経由とし、
  drain（D5/D8）は dtor body 内（全 member 生存）で完了させる。
  member teardown に EBR 未処理エントリを持ち越してはならない
  （D8 drainAll 後に pendingRetireCount()==0 を assert）。

INV-D162-9（deferred slot 出口の全数カバレッジ）:
  deferredSlot_ のエントリが消える全経路（consume→submit / discard / overwrite /
  clearDeferredForShutdown / drainDeferredClearIfRequested）は、
  対応する disposition（S1〜S4）を必ず実行し、
  「slot 消滅後に handle map に Constructing エントリが残存する」状態を invariant 違反として
  診断ログ（DIAG）で検出可能にすること（D162-2-D の 1 行 instrumentation 許可範囲）。
```

---

## 14. D162-2-D gate decision

**判定: D162-2-D の実装可否は条件付き（conditional GO / 分割必須）。**

指示の基準「C-A〜C-E のいずれかを証明できない場合は D162-2-D に進まず停止」に対し、
本監査は C-F を採用した。したがって **D162-2-D をそのまま開始してはならない**。
ただし C-F は「監査の失敗」ではなく、監査が 4 件の独立した構造的欠陥（§12 の (i)〜(iv)）を
分離した結果である。以下の分割を提案する:

1. **D162-2-D（heap corruption 同定・read-only+診断のみ）**:
   PageHeap/Application Verifier または Debug ビルド短縮再現で corrupting free の呼出 site を特定する
   （production source 変更 0・テスト環境ツールは許容と解釈。ダンプ 7 件は保存済み）。
   これが完了するまで S3/V-D の destroy は**再有効化禁止**を維持。
2. **D162-2-E（INV-D162-6/7/8/9 の実装）**:
   (a) CacheMap dtor の Destroy branch 修正、(b) dtor dangling retire の修正、
   (c) S1/S2 カバレッジ窓口（publish 直後第 1 deferred gen）の閉鎖。
   いずれも corrupting free 同定と独立して安全に実施可能な範囲に限る。
3. **S3/V-D destroy 再有効化は 1+2 の完了後、60-gen soak で exit 0x0 かつ
   retained = final active 1 件のみ** を確認してから行う。

---

## 15. 本監査の証跡

- クラッシュダンプ 7 件: `C:\Users\user\AppData\Local\CrashDumps\ConvoPeq.exe{,(1),.448,.15388,.20688,.21604,.22460,.23044,.24172,.26632}.dmp`
- stack walk スクリプト出力: evidence/D162-2C_dump_stack_candidates.txt（24172 RWDI）/
  evidence/D162-2C_dump22572_stack.txt / evidence/D162-2C_dump448_srcframes.txt（Debug inline 解析）
- 分離マトリクス: evidence/D162-2B_EVIDENCE.md §4
- 60-gen 最終 soak: evidence/D162-2B_soak.log（C-6 遷移追跡の一次資料）
- 6-gen 反復: evidence/D162-2B_soak6.log（address reuse・exit 0 確認）

production source 変更: **0**（本監査では src/ に一切触れていない。D162-2-B の無効化注記は
D162-2-B の成果物のまま）。
