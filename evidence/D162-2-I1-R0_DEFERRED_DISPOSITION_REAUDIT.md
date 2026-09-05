# D162-2-I1-R0 — Deferred-slot Terminal Disposition / Shutdown Ordering Re-audit（read-only）

```text
Date:        2026-09-04
Type:        read-only audit / production source 変更 0 / test 0 / build 0 / soak 0 / B-2〜B-6 実施禁止
Baseline:    ConvoPeq.md Generated 2026-09-04 20:28:50（G4 tree・B-1 実行後も変更 0）
Position:    D162-2-I1 Profile B-1 STOP（residual=1）の修復設計確定監査。
             「どこに無条件 terminal disposition を置くべきか」を現行 source のみから決定する。
参照:        evidence/D162-2-I1/PROFILE_B1_STOP_RECORD.md / evidence/D162-2C §3 / evidence/D162-2G0 §2
判定:        **GO — 修復案 = Candidate A′（releaseResources VerifyDrained: world clear 後に
             clearDeferredForShutdown を無条件呼出）**。Candidate B / C は却下（§4）。
```

---

## 0. 総合判定

> **R0 = GO。** B-1 residual=1 の原因は source 構造として再確認できた
> （deferred slot の terminal disposition が 3 経路すべて条件付き trigger に依存し、
> shutdown pipeline に無条件処分が存在しない）。責任箇所は一意に決定:
> **releaseResources / VerifyDrained phase の world clear 後**に
> `runtimeOrchestrator_->clearDeferredForShutdown()` を無条件呼出する（Candidate A′）。
> Thread ownership（post-join MessageThread）・authority（既存 S3 経路の再利用）・
> EBR 順序（dtor body D5/D8 drain）・V-D との二重処分防止（map erase guard + 順序）が
> すべて現行構造で保証される。Candidate C（dtor）は member 破壊順により UAF で明確に却下。

---

## 1. R0-1: shutdown ordering 完全追跡（実ファイル行番号）

### 1.1 releaseResources 内の時系列（AudioEngine.Processing.ReleaseResources.cpp）

```text
:199-202  stopRebuildThread()                       ← rebuildThreadShouldExit=true (:39634)
                                                      + rebuildCV.notify_all + rebuildThread.join()
                                                      ★ RebuildThread が完全停止（join 済み）
:224      after stopRebuildThread
:342      transitionTo(ReclaimComplete)
:349      transitionTo(EmergencyDrain)
:350-380  EmergencyDrain body（★ isEmergencyDrainRequested()==true の場合のみ実行）
            :356? clearDeferredForShutdown()        ← S3 site（条件付き）
:381?     quarantine cleanup（destroyForShutdown 等）
:441      transitionTo(VerifyDrained)
:44x      V-D: getActive/FadingRuntimeDSPHandle → resolve → retire → tryShutdownQuiescentReclaim
:48x      requestShutdownClearNonRt + clearPublishedRuntimeSnapshotsNonRt（world clear）
:49x      drainAllQuarantineStore（activeReaderCount==0 時）
:50x      V-D 破壊 block（G2: lifetimeMgrForFinalDSP.retire — 既存 authority）
:5xx      waitForDrain(2000, 2)
:6xx      transitionTo(ShutdownComplete)
```

### 1.2 ~AudioEngine 側（CtorDtor.cpp）

```text
:31245-31262  stopRebuildThread()（再確認・join 済み）
:31313        pendingTask.currentDSP retire（pendingTask のみ・deferred slot は触れない）
:31339-31349  E-2 retireByHandle（active/fading handle）
:31363-31427  publishEpoch → graceful drain(D5) → world clear → drainDeferredRetireQueues(true)
              → drainPendingRetireIntentsForShutdown → drainAll(D8)
:31432-31440  E-3 assert（pendingRetireCount==0）
:31459        drainForShutdown（E-1）
→ member teardown（宣言の逆順）
```

### 1.3 指示 9 項目の回答

| # | 項目 | 結論 |
| --- | --- | --- |
| 1 | `rebuildThreadShouldExit` が true になる時点 | releaseResources :202 側 `stopRebuildThread()`（:39634 で store）および ~AudioEngine :31262。**両者とも EmergencyDrain(:349)/VerifyDrained(:441) より前** |
| 2 | RebuildThread 完全停止時点 | `stopRebuildThread()` 内 `rebuildThread.join()` — **EmergencyDrain / VerifyDrained 到達時点で join 済み** |
| 3 | `deferredSlot_` の最終 writer | **RebuildThread**（enqueueDeferred :548 / processDeferredAdmission finishView）。B-1 では shutdown 開始（Auto-exit flush :3763）の 20 行前 :3743 に最終 CREATE。join 後の writer は不在 |
| 4 | 安全に処理できる時点 | **join 後ならいつでも** — MessageThread からの clear は既存契約（EmergencyDrain :356 呼出 + C1 fallback。D135-9 Gate F で assert-free 契約 VALID 済み） |
| 5 | `clearDeferredForShutdown()` をその地点から呼べるか | 安全（noexcept・no-jassert・`retireRegisteredDSP` は mutex + EBR の NonRT producer） |
| 6 | `DSPLifetimeManager::retire()` → EBR enqueue の整合 | retire → map erase → registry Retired → requestReclaim → `enqueueWithRetry`。**digest は既存 dtor body D5/D8 drain**（G4 60-gen で最終 destroy が dtor body 内で実行済みの実績） |
| 7 | `drainDeferredRetireQueues(true)` 前後 | dtor :31405 で実行。VerifyDrained で enqueue した EBR entry は **D8 drainAll で必ず消化**（E-3 assert が保証） |
| 8 | `m_epochDomain.tryReclaim()` の位置 | EmergencyDrain block 内（:38209 相当）— VerifyDrained の V-D/deferred 処分より**前**。retire はここより後に置いても D5/D8 が消化 |
| 9 | member destruction order との関係 | **破壊順（宣言の逆順）**: `dspHandleRuntime_`(h:5034) → `shutdownRuntime_`(h:5000) → `m_retireRouter`(h:4813) → **`runtimeOrchestrator_`(h:3671)** → `noiseShaperLearner`(h:2768) → `eqCacheManager`(h:2422)。→ Orchestrator dtor 時点で `m_retireRouter` / `dspHandleRuntime_` は**破壊済み**（§4 Candidate C 却下根拠） |

**R0-1 結論**: RebuildThread join 後（EmergencyDrain 遷移以降〜VerifyDrained）であれば、
MessageThread から deferred slot の terminal disposition が単一 writer 契約上安全に実行できる。
**安全な位置は存在するが、現行 source にはその位置での無条件呼出が存在しない** — これが B-1 leak の構造的原因。

## 2. R0-3: gen7「published + deferred slot 残留」の意味論監査

### 2.1 B-1 の観測事実（log 確定分）

```text
:164   BUILD_PHASE gen=5 → :183 [PUBLISH] seq=5 gen=5 worldId=5      （gen5 publish 正常）
:830   BUILD_PHASE gen=6
:847   D127_TAIL seq=7 needsCrossfade=1
:848   [PUBLISH] seq=7 gen=7 worldId=7
:849   trySubmit: executor_.publish SUCCEEDED gen=6                   ← executor publish は gen=6
:854   [XFADE] start expected=0.010s                                  ← gen6→gen7 crossfade 開始
:856   [WORLD] Active=1 Fading=1                                      ← fading 残存
:1240  DSP_FOOTPRINT dsp=DE2D0080 construct（gen7 DSP 構築）
:1424  BUILD_PHASE generation=7
:1445  event=CREATE gen=7 dsp=DE2D0080                                ← retention re-defer churn 開始
       …69 CREATE / 68 CONSUME（[D135] re-defer (retain) gen=7 oblId=0）…
:3743  最終 CREATE（slot 残留）
:3763  Auto-exit flush → shutdown
:3814  [ISR][Shutdown] Drain incomplete: … deferred=1 …（shutdown 自身が観測）
:3811  V-D fading-final → C356B080（gen5・midrun 破壊済みの dangling）→ retired=0 no-op
:3808  V-D active-final → CC843080（gen6）→ retired=1 → destroy remaining=0
       gen7 DE2D0080: retire 0 / EBR 0 / destroy 0 → **residual=1**
```

### 2.2 判定: 「published + deferred 残留」の意味論

**DSP-level evidence（最も信頼できる）**:
- V-D の active-final が **gen6（CC843080）** であった = VerifyDrained 時点の
  active world には **gen6 が active であり、gen7 DSP は active world に Admission
  されていない**。
- gen7 DSP は `phase=retained`（deferred slot resident）タグ付きで 69 回の CREATE を持ち、
  retire/EBR/destroy が **0 件**。
- churn の 1 サイクル = `CONSUME → submitPublishRequest → hasFadingRuntimeInWorld()==true
  → DeferredFadingActive → enqueueDeferred（CREATE・retention）`。
  分類条件は `PublicationAdmission.cpp:55-58`（`hasFading` が true の間 DeferredFadingActive）。

**→ 判定: gen7 DSP は「publish 済み」ではなく「publish 待ち（DeferredFadingActive で
retention 状態）」の個体として deferred slot に残留した。** すなわち
「published DSP + deferred 残留」という ownership 不整合は**成立していない**（予備判定）。

**注意点（タグ意味論の留保）**: `[PUBLISH] seq=7 gen=7 worldId=7`（:848）が gen7 の
世界 publish を示すと読むと上記と矛盾する。ただし (i) この行は gen7 の BUILD_PHASE（:1424）
より**前に**出現しており gen7 DSP（DE2D0080）の publish ではあり得ないこと、(ii) 直後の
`trySubmit: executor_.publish SUCCEEDED gen=6` と整合すること、から、このタグの seq/gen は
publication-log bookkeeping の採番（worldId/seq と rebuild gen のずれ）であり、
**DSP 個体の publish 状態の証拠としては採用しない**。実装修復時に 1 点だけ確認する
（§5 留保）。

**churn の駆動源**: `[XFADE] start expected=0.010s`（:854）の crossfade が
**completed まで到達しなかった**（`[XFADE] completed` 0 件・`[WORLD] Fading=1` 消えず）。
`hasFadingRuntimeInWorld()` が true のまま → Admission が常に DeferredFadingActive を返し、
Coordinator re-drive（D135-8 F3 の re-defer churn・TTL 30s で bounded・run は 15s で TTL 未満）
が継続した。**なぜ 10ms crossfade が complete しなかったか（Timer.cpp:928 tryCompleteFade の
成立条件）は別 root-cause 課題**として DEFER 登録する（本監査の leak 修復とは独立・
churn は B-1 の leak の「量を増やした要因」であり「zero にする要因」ではない）。

**→ R0-3 結論**: B-1 の leak は「publish 済み DSP の二重残留」ではなく、
**「未 publish（deferred retention）DSP が shutdown で無処分になった」**こと。
S3/E-4d/V-D の既存契約はこの個体を正しく扱える（authority retire で処分可能）。
意味論の不整合ではなく、**disposition の呼出欠落**が原因。

## 3. R0-2: 3 候補比較

### Candidate A — VerifyDrained / releaseResources 側（**採用・A′**）

**位置**: `releaseResources` の VerifyDrained phase、**world clear 後・V-D retire block の直後**
（既存の無効化 V-D destroy block と同じ closure。md:38367-38390 領域の直後）。

```cpp
// ★ D162-2-I1 修復: shutdown 時 deferred slot の無条件 terminal disposition。
//    EmergencyDrain / Timer midrun / C1 はすべて条件付き trigger のため、
//    short-run profile で slot 残留 DSP が leak する（B-1 実測）。
//    world clear 後であるため world topology 参照は解消済み。
//    RebuildThread join 済み（:202）のため単一 writer 契約成立。
if (runtimeOrchestrator_)
    runtimeOrchestrator_->clearDeferredForShutdown();
```

- `clearDeferredForShutdown()` は内部で `hasDeferred_` を確認するため slot 空なら no-op。
  slot 保持時は G1 の S3 block（`retireRegisteredDSP(req, "shutdown-clear")`）が実行される。
- **INV-D162-8**: EBR enqueue → dtor body D5/D8 drain で消化（G4 実績どおり）。E-3 assert が保証。✓
- **V-D との二重処分**: V-D は active/fading final（B-1 では gen6/gen5）を扱い、deferred DSP
  （gen7）は別個体。万一同一個体でも map erase guard で no-op（INV-D162-3）。✓
- **EmergencyDrain との二重呼び**: EmergencyDrain body で既に clear 済みの場合、2 回目は
  `hasDeferred_==false` で no-op。冪等。✓
- **B-era ordering 制約（world clear 後に破壊）**: world clear（:38347-38353）の後に配置するため
  world topology 参照は解消済み。deferred DSP が world に参照されていてもいなくても安全。✓

### Candidate B — EmergencyDrain を常時 disposition に変更（**却下**）

- `isEmergencyDrainRequested()` 分岐の外しは、(i) PolicyEngine の「EmergencyDrain は optional な
  最終手段」という意味論契約の変更、(ii) body 内の `m_epochDomain.tryReclaim()` 強制実行・
  `crossfadeRuntime_.reset()`（crossfade recovery 強制）という**別の副作用**を normal shutdown
  に巻き込む、という問題がある。
- 「deferred slot の処分」だけが目的なら Candidate A′ で足りる。**単なる if 外しとしては
  採用しない**（指示どおり）。

### Candidate C — dtor 側（**明確に却下: UAF**）

- member 破壊順（宣言の逆順）: `dspHandleRuntime_`(h:5034) → `shutdownRuntime_`(h:5000) →
  `m_retireRouter`(h:4813) → **`runtimeOrchestrator_`(h:3671)** → …
  → Orchestrator dtor が走る時点で `m_retireRouter` / `dspHandleRuntime_` /
  `runtimeDSPHandleMap_` は**破壊済み**。
- `retireRegisteredDSP` は `resolveDSPHandle`（dspHandleRuntime_ 依存）と `map.erase`
  を必要とし、`enqueueWithRetry` は `m_retireRouter` を必要とする → **すべて UAF**。
- さらに EBR enqueue しても digest 実行主体（tryReclaim / drainAll）は ~AudioEngine body で
  完了済みのため、**destroy callback が永遠に走らない**。
- これは E-1（CacheMap dtor を engine 非依存 no-op にし、事前 drain へ責務移譲した）と
  同型の設計判断であり、**同じ結論（member teardown 中の authority 触碰禁止・INV-D162-6 系）が
  deferred slot にも適用される**。却下。

### Candidate D — 別の既存 authority 経路

- dtor body の `pendingTask.currentDSP` retire（CtorDtor :31313）は pendingTask のみ対象で
  deferred slot は別物。
- `drainDeferredRetireQueues(true)` は EBR queue の drain であり slot 処分ではない。
- **既存経路に deferred slot の最終処分は存在しない** — これ自体が B-1 の原因。追加は Candidate A′。

### Candidate E — 解釈誤りの可能性

- 「S3 が呼ばれなかった」だけの観測ではなく、(i) E-4 会計不成立（CREATE 69 ≠ exits 68）、
  (ii) `[ISR][Shutdown]` 自身の `deferred=1` 観測、(iii) gen7 DSP の destroy 0 件、
  (iv) 呼出経路 3 系統の全 trigger 不成立、が source 構造と突合済み。**解釈誤りなし。**
- 留保 1 件のみ: `[PUBLISH] seq=7 gen=7` タグの意味論（§2.2）— 修復実装時に
  「slot 残留 DSP が active world に含まれるか」を 1 回確認すればよい（A′ は world clear 後
  配置のためどちらでも安全）。

## 4. R0-4: 最小修復単位（決定）

```text
決定: Candidate A′
対象関数:  AudioEngine::releaseResources()（VerifyDrained phase）
呼出位置:  world clear（requestShutdownClearNonRt + clearPublishedRuntimeSnapshotsNonRt +
           retirePublishedRuntimeWorldNonRt）および V-D retire block の直後
           （既存の V-D destroy block closure の後・waitForDrain の前）
呼出内容:  runtimeOrchestrator_->clearDeferredForShutdown()（1 行・無条件）
呼出順序:  requestShutdown → audio stop → stopRebuildThread(join) → … → world clear
           → V-D retire（active/fading） → ★ deferred slot disposition（新規）
           → waitForDrain(2000,2) → … → ~AudioEngine D5/D8 drain → E-3 assert
Thread:    MessageThread・RebuildThread join 済み（単一 writer 契約・assert-free clear 契約）
Authority: retireRegisteredDSP（G1 S3 と同一経路）→ DSPLifetimeManager::retire → EBR
EBR:       enqueue → dtor body D5/D8 drain で digest（INV-D162-8 準拠・E-3 assert が保証）
二重防止:  (i) hasDeferred_ check で冪等（EmergencyDrain で先に clear されたら no-op）
           (ii) map erase guard（既に retire 済みなら no-op）
published/deferred 意味論: gen7 は publish 待ち retention 個体（§2.2 判定）。
           仮に world 参照が残っていても world clear 後配置のため安全。
INV 影響:  INV-D162-1（authority 統一 ✓）/ 2（eventual destruction の穴を塞ぐ ✓）/
           3（冪等 ✓）/ 4（NonRT ✓）/ 5（world clear 後・EBR epoch ✓）/
           6（member teardown 触らず ✓）/ 7（legacy slot 不使用 ✓）/
           8（EBR 単経路・dtor body drain ✓）/ 9（CLEAR 出口 disposition 完備 ✓）
```

**副次効果**: この修復により、**S3 standalone `retired=1`（G1-G4 の未観測事項）が
B profile で実測可能になる**（slot 保持 DSP が RebuildThread join 後に残留する状態で
S3 block が到達するため）。R0 実装後の B 再試験で併せて確認する。

## 5. GO 条件の判定（9 項目）

| GO 条件 | 判定 |
| --- | --- |
| B-1 residual=1 の原因が source 構造で再確認できる | ✓（3 経路すべて条件付き + 無条件 site 不在） |
| shutdown 時 deferred terminal disposition の責任箇所が一意に決まる | ✓（releaseResources VerifyDrained world clear 後） |
| Thread ownership が明確 | ✓（post-join MessageThread・D135-9 契約） |
| `retireRegisteredDSP()` を使用できる | ✓（既存 S3 block をそのまま実行） |
| EBR 順序が保証できる | ✓（dtor D5/D8 drain・G4 実績） |
| V-D との二重処分を構造的に防げる | ✓（map erase guard + 別個体 + 順序） |
| published/deferred 状態の意味論が説明できる | ✓（§2.2: gen7 は retention 個体・タグ意味論の留保 1 点を実装時確認項目として明記） |
| dtor 依存 UAF を持ち込まない | ✓（Candidate C 却下・A′ は releaseResources 内） |
| I0/I1 の既存測定契約を変更する必要がない | ✓（契約どおり B profile が residual 0 になることを目指す） |

# **判定: R0 = GO（Candidate A′）。次工程 = 修復実装（1 行 + 注記）→ Build/CTest → B-1 再試験 → B-2〜B-6。**

## 6. DEFER / 残置（本監査で確定した周辺課題）

1. **crossfade 非完了の root cause**（B-1: `[XFADE] start expected=0.010s` が completed に
   到達せず fading が残存 → churn 駆動）: `tryCompleteFade`（Timer.cpp:928）の成立条件と
   crossfade runtime の状態遷移の監査。leak 修復とは独立（churn は TTL bounded）だが、
   B-1 の churn 69 回を生んだ起点。
2. **`[PUBLISH] seq=7 gen=7` タグ意味論**（worldId/seq/gen の採番ずれ）: 修復実装時に
   1 回確認（slot 残留 DSP の world 参照有無の確定）。
3. V-D fading-final の dangling resolve（B-1 実測・V-D-b が防止）: registry reclaim pending
   時の resolve 挙動 — INV-D162-7 の観点で residual register に記録済み。
