# D172-1 — MEM_SNAP Lifetime / Ownership Audit（evidence）

- 日付: 2026-09-07
- Type: **read-only audit** — production source 0 / test source 0 / CMake 0 / build 0 / CTest 0
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-07 21:38:42`
- 前提: D171-1 PASS（MEM_SNAP hazard = AUDIT REQUIRED → 本 audit）
- 対象: `activeRuntimeDSPSlot`（AudioEngine.h:2228）と MEM_SNAP sampler（Timer.cpp:1079-1088）の lifetime/ownership
- 原則: Practical Stable ISR Bridge Runtime — Observer は観測専用・所有権を持たない・Retire/Delete は Observer の責務ではない

---

## P1 — `activeRuntimeDSPSlot` 全 access graph

### P1-1. slot 定義と契約

```cpp
// AudioEngine.h:2228
std::atomic<DSPCore*> activeRuntimeDSPSlot{nullptr};
```

- 契約コメント（AudioEngine.h:2265-2267）: 「placeholder 専用のレガシースロット。通常動作（runtime world 公開後）では null」
- 契約コメント（PrepareToPlay.cpp:285）: 「activeRuntimeDSPSlot は非所有 topology mirror（h:2245）であり所有権を運ばない」
- 契約コメント（ReleaseResources.cpp:151-161, D169-1R RC-D169-1-2）: 「placeholder 専用レガシースロット…capture は topology observation 用にのみ使用する（**ownership authority に昇格させない**）」「rebuild publish は pointer slot を更新しない」

### P1-2. writer 全箇所（4 箇所・網羅実測）

| # | 箇所 | 操作 | スレッド / 文脈 |
|---|---|---|---|
| W1 | `AudioEngine.CtorDtor.cpp:153` | `setActiveRuntimeDSP(nullptr)` | dtor 内 clear（rebuildMutex 下） |
| W2 | `AudioEngine.Processing.PrepareToPlay.cpp:287` | `setActiveRuntimeDSP(placeholderRaw)` | placeholder 生成後・world publish 前 |
| W3 | `AudioEngine.Processing.PrepareToPlay.cpp:318` | `setActiveRuntimeDSP(nullptr)` | tryAdmit 失敗 → `destroyRolledBackDSP(placeholderRaw)`（:317）直後（同一 MessageThread シーケンス内） |
| W4 | `AudioEngine.Processing.ReleaseResources.cpp:180` | `setActiveRuntimeDSP(nullptr)` | releaseResources・rebuildMutex 下（:178 capture → :180 clear） |

- `releaseActiveRuntimeDSP()`（h:2275, exchange null）: **呼び出し元 0 件**（dormant accessor）。
- **retire / reclaim / EBR destroy 経路に writer は 1 件も存在しない**（P2 で証明）。

### P1-3. reader 全箇所（取得 vs dereference の分離）

| # | reader | 種別 | dereference |
|---|---|---|---|
| R1 | `AudioEngine.Timer.cpp:1079`（MEM_SNAP block 内） | 取得 | **あり** — `activeDSP->collectTrackedMemoryStatistics()`（:1082） |
| R2 | `AudioEngine.Processing.Latency.cpp:97` | 取得（world 未公開時 fallback のみ） | **あり**（後続メンバアクセス）。ただし world null ⇔ slot null の共 清排が同一 critical section（W4→world idle publish）で成立するため、dangling 窓は構造的に閉じる（P2-6） |
| R3 | `AudioEngine.h:3850`（`logRuntimeTransitionEvent` 内） | 取得 | **あり** — `current->runtimeUuid` 読み取り（h:984 member）。ただし本 inline の production 呼び出し元は 0 件（dormant diagnostic） |
| R4 | `AudioEngine.h:3884`（`validateDistinctRuntimeSlots`） | 取得 | **通常 path なし**（pointer 比較のみ）。異常検出時の getUuid ログでのみ dereference |
| R5 | `AudioEngine.Processing.ReleaseResources.cpp:170/178/223` | 取得 | **なし** — capture 後 `juce::ignoreUnused`（D169-1R で pointer-value retire 廃止済み・観測専用契約どおり） |
| R6 | `AudioEngine.CtorDtor.cpp:141/151/168` | 取得 | 同 R5（dtor path 観測専用） |
| R7 | `DSPLifetimeManager.cpp:145`（`getActive()` → `void*`） | 取得 | **なし** — 呼び出し元 0 件（dormant） |
| R8 | `AudioEngine.h:2261`（`hasActiveRuntimeDSP`） | 取得 | **なし**（null 比較のみ） |
| R9 | tests（`PublishPipelineIntegrationTests.cpp:207/238/731/780/868/942` 等） | 取得 | テスト内 pointer 比較主体（MEM_SNAP 並行書込みとの race は D169-2-5 で CriticalSection 直列化済み） |

### P1-4. access graph

```text
[W2] PrepareToPlay:287 setActiveRuntimeDSP(placeholderRaw)   (MessageThread)
        │
        ▼
activeRuntimeDSPSlot  (std::atomic<DSPCore*>, 非所有 mirror)
        │
        ├─[R1] Timer.cpp:1079 getActiveRuntimeDSP()  (MessageThread, timerCallback 毎 tick)
        │        └─ :1082 activeDSP->collectTrackedMemoryStatistics()   ← 実 dereference ①
        ├─[R2] Latency.cpp:97 fallback (world null 時のみ)              ← dereference ②（窓閉じ）
        ├─[R3] AudioEngine.h:3850 logRuntimeTransitionEvent (dormant)   ← dereference ③（休止）
        ├─[R4] validateDistinctRuntimeSlots :170/178/223/141/151/168    ← 比較のみ
        └─[W1/W3/W4] clear (nullptr)
```

**writer→slot→reader→member access のうち、destroy 側から slot へのフィードバック経路は存在しない。**

---

## P2 — Lifetime chain の完全証明

### P2-1. 対象 DSP（placeholder P）の一生

```text
① construction        PrepareToPlay.cpp（DSPCore 生成、aligned alloc）
② slot publication    :287  setActiveRuntimeDSP(P)              ← 非所有 mirror 設定
③ World publication   :301  commitRuntimePublication(
                            needsRegistration(P))               ← handle map 登録（H 付与）・world gen1 current=P
④ replacement         rebuild publish → Orchestrator.cpp:83
                            oldHandle = dspHandleRuntime_.getActiveRuntimeDSPHandle()   (= P の H)
⑤ retire handle       DSPLifetimeManager::retire(P)（DSPLifetimeManager.cpp:36-69）
                          → engine_.retireDSPHandleForRuntime(P)（台帳解除 + Retired 遷移 — AudioEngine.h:4332-4348 契約）
⑥ EBR enqueue         router_->enqueueWithRetry(P, &destroyDSPCoreNode, epoch, Generic)
                          epoch = publicationEpoch（または currentEpoch）= E_r
                          （AudioEngine.h:4311 / Publication.cpp:16 markRetireEpoch = publishEpoch）
⑦ pending reclaim     epoch 不安全時は pendingReclaimHandles_ に保留（Retire.cpp:88-140）
⑧ EBR grace           drain 実行: entry.epoch < minReaderEpoch のときのみ deleter 実行
                          （ISRRetireRouter.h:77-79 / EpochDomain.h:201-210 getMinReaderEpoch）
⑨ destroyDSPCoreNode  Threading.cpp:38-40  core->~DSPCore(); convo::aligned_free(core)
```

### P2-2. destroy 実行主体（スレッド）

`drainDeferredRetireQueues(false)` の呼び出し箇所:

| 箇所 | スレッド |
|---|---|
| `AudioEngine.Threading.cpp:383`（`runCoordinatorPhase` 末端・E-1.9-B Phase2） | **CoordinatorLoop worker thread**（enqueueWithRetry の `signalDrainWakeup()` でイベント駆動 + 1ms fallback — ISRRetireRouter.cpp:375-378/470） |
| `AudioEngine.Timer.cpp:1827/1843`（`executeRecoveryAction` Recover/Restore — CtorDtor.cpp:69 登録の policy callback） | Recovery 実行文脈（非 RT） |
| `AudioEngine.CtorDtor.cpp:270` | shutdown（MessageThread・allowDuringShutdown=true） |

→ **物理破壊は CoordinatorLoop worker thread を含む非 Message thread で実行され得る。**

### P2-3. destroy 実行時点で `activeRuntimeDSPSlot` に何が保証されているか

**何も保証されていない。**

- writer 4 箇所（P1-2）のいずれも retire/reclaim/drain/destroy 経路に含まれない。
- したがって **P が物理破壊された後も slot は P を保持し続ける**（次の W1/W3/W4 まで）。
- slot 値と DSP lifetime を結ぶ invariant は source 上に存在しない。RC-D169-1-2 コメントはこれを意図的契約（「ownership authority に昇格させない」）として明文化している。

### P2-4. EBR は slot reader を保護するか — **保護しない**

EBR の意味論（EpochDomain.h で実測）:

- `enterReader(tid)`（EpochDomain.h:114-131）: reader は **enter 時点の currentEpoch** を pin する（`slot.epoch = currentEpoch()`）。
- `exitReader(tid)`（:141-160）: 最終 exit で `slot.epoch = kInactiveEpoch`（pin 解除）。
- `getMinReaderEpoch()`（:201-210）: **active（depth>0）な reader の pin 中 epoch の最小値**。active reader が 0 なら `currentEpoch()`。
- destroy 条件: `isOlder(entry.epoch, minReaderEpoch) == true` — つまり **retire entry より前の epoch を pin している active reader が 1 つもいなければ即破壊**。

MessageThread reader の状況:

- `timerCallback` 冒頭（Timer.cpp:427-428）で `makeRuntimeReadHandle(messageCtx)` → `ObservedRuntime`（`RCUReaderGuard` 保持、ObservedRuntime.h:26-31）→ **RAII で callback 全体（MEM_SNAP block を含む）にわたり MessageThread は EBR reader として active**。
- しかし **pin される epoch は「その tick 開始時の current epoch」**である。placeholder P の retire entry は epoch E_r で enqueue 済みであり、retire より後の tick で MessageThread が pin する epoch は E_r **以上**。epoch ≥ E_r の reader は E_r entry の破壊を**遅延しない**（EBR は「enter 時点より後に publish された対象」を守る機構で、既に通過した epoch の entry には無力）。
- inter-tick window（timerCallback 終了〜次回開始、通常数十 ms）では MessageThread は inactive（epoch = kInactiveEpoch）→ **P の破壊はこの窓でCoordinatorLoop drain により合法的に完了する**。
- 仮に retire が tick N の callback 中に起きていても、MEM_SNAP が読む slot 値は **prepare 時（generation G1）に書かれた stale 値**であり、現在の read section の epoch 規律の下で取得された pointer ではない。EBR は「read section 内で取得した pointer」を保護するが、「section 外で取得され保持された stale pointer」は再検証しない。

**結論: EBR は slot reader を保護しない。slot の raw pointer は epoch 規律の下に一度も置かれていない。**

### P2-5. `collectTrackedMemoryStatistics()` 呼び出し時点の lifetime proof — **不存在**

- 呼び出し時点で保持されているのは「slot の acquire load 値」のみ。handle 検証なし・RCU world 解決なし・pin された epoch による保護なし（P2-4）。
- 「MessageThread だから安全」は**証明として不成立**:
  1. destroy は CoordinatorLoop worker で走る（P2-2）→ cross-thread 窓が実在する。
  2. 仮に destroy が同一 MessageThread シーケンスで完結する場合でも、**次 tick 以降の dereference は解放後オブジェクトへの access そのもの**（順序性の問題ではなく、stale 値保持の問題）。

### P2-6. slot と DSP lifetime を結ぶ同一 invariant の有無

| 質問 | 答え |
|---|---|
| slot 非null ⇒ pointee 生存、が invariant か | **否**（retire 経路が slot を更新しないため破綻） |
| EBR が slot reader を保護するか | **否**（P2-4） |
| Timer/MEM_SNAP は epoch reader か | world 読み取りとしては Yes（timerCallback 冒頭 guard）。ただし slot dereference の保護には**ならない** |
| destroy 前に slot clear が必須か | dereference reader（R1/R2/R3）が存在する限り、retire 経路での clear（または reader 側の authority 解決への切替）なしには成立しない |
| slot clear absent でも別 authority で安全が成立するか | **成立しない** — slot 値を dereference する reader に対する lifetime authority は現行 source に存在しない |

---

## P3 — MEM_SNAP 実 dereference の判定

### P3-1. 呼び出しチェーン（無条件到達）

```cpp
// Timer.cpp:1022  #if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
// Timer.cpp:1078-1088
auto* activeDSP = getActiveRuntimeDSP();          // slot acquire load
if (activeDSP != nullptr)
{
    auto stats = activeDSP->collectTrackedMemoryStatistics();   // ← 実 dereference
    ...
}
```

- 単なる null 比較ではなく、非 null なら **毎 tick 無条件に** member function call が発生する。
- guard は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` のみ（CMakeLists.txt:129 `option(... OFF)` — **production 既定では compile-out**、diagnostic build で有効）。

### P3-2. `collectTrackedMemoryStatistics()` 内部（DSPCoreLifecycle.cpp:337-377）

| アクセス | 内容 | 性質 |
|---|---|---|
| `this` dereference | あり（非 virtual member function 呼び出し） |解放後オブジェクトへの呼び出し成立 |
| member read | `oversamplingFactor` / `maxSamplesPerBlock` / `maxInternalBlockSize` / `alignedCapacity` / `dryBypassCapacityDouble` | POD member（非 atomic）— freed memory からは garbage 値 |
| nested object access | `histories().fixedLatencyBufferSize`（h:951 `HistoryRuntimeState& histories()` — member 参照返し） | nested member read — 同上 |
| static/live counter のみか | **否** — 全アクセスが `this` 経由（engine 側 live counter は MEM_SNAP の別項目） | — |
| 解放済み object でも成立し得るアクセスか | **成立する** — noexcept・lock/alloc/vtable 呼び出しなし。AV は page decommit / 再利用時のみで、通常は silent garbage 値を出力 | — |

### P3-3. 判定

> **「dangling pointer がある」だけでなく「dangling pointer が member dereference される」まで source 上で確定。**
>
> 条件: ① diagnostic build、② prepare で placeholder 生成・world publish 成功、③ 以後に 1 回以上の rebuild publish 成功（P が retire→EBR destroy される）、④ releaseResources / 再 prepare がまだ来ていない。③ 以降の全 MEM_SNAP tick が dereference 状態に入る（構造的・確定的。非決定なのは AV が顕在化するか silent garbage で済むかのみ）。

---

## P4 — Dynamic reproduction 設計（実施は D172-2・本 audit では設計のみ）

### P4-1. sequence（diagnostic build）

```text
1. CLI start → prepareToPlay（placeholder P 生成・slot=P・world gen1 publish）
2. settle
3. 構造 rebuild を 1 回発火（SR/BS 変更またはパラメータ変更）
   → gen2 publish → oldHandle(P) retire → EBR enqueue(epoch=E_r)
4. drain 完了待ち（telemetry: retireQueueDepth_==0 / runtimeReclaimCount 増加確認）
5. release せず session 維持 → timerCallback を複数 tick 走行（MEM_SNAP 連続出力）
6. shutdown
```

### P4-2. 観測方式（test source 変更 0 で可能な方式を含む）

**方式 α（log 相関・source 変更 0）** — 既存 diagnostic ログのみで成立:

- 破壊側は既存で address を出力する: `[D117_DESTROY] dsp=%p`（Threading.cpp:22）・`[DSP_FOOTPRINT_RELEASED] dsp=%p remaining=0`（Threading.cpp:44）。
- `[MEM_SNAP] PUBLISH ... TRK: total=...` が、対応する `dsp=P` の破壊ログ**以後**の tick で非 zero TRK 値を出し続けること = 「解放済みオブジェクトの member read が継続して成功している」ことの観測（P3-2 のとおり TRK 値は全て `this` 経由のため）。
- 判定規則: `t_destroy(P) < t_MEM_SNAP_tick ∧ TRK≠0` のペアが 1 つでも存在すれば lifetime violation の観測成立（crash 不要・決定的）。

**方式 β（pointer identity 束縛・要 source 変更 1 行 — D172-2 で判断）**: MEM_SNAP の log format に slot pointer（`%p`）を 1 項目追加し、destroy ログの `dsp=%p` と pointer 一致で突合。timestamp 相関より厳密。

**方式 γ（動的検証器）**: 既存 diagnostic exe を Dr.Memory（use-after-free 検出）で実行、または x64dbg で `destroyDSPCoreNode` 後の当該 block へ write breakpoint。D172-2 の実施手段候補（本 audit では未実行 — build 0 制約）。

### P4-3. 記録すべき相互項目

`slot value`（β で追加可能）・`DSP object address`（D117_DESTROY 既存）・`generation`（MEM_SNAP PUBLISH gen= 既存）・`retirement`（D117_RETIRE 既存）・`reclaim`（RetireRouter telemetry 既存）・`Timer callback`（tick 時刻）・`MEM_SNAP invocation`（PUBLISH 行時刻）— **方式 α は全項目が既存ログで充足**（slot value のみ方式 β 必須）。

---

## P5 — 修復候補の比較（実装はしない・D172-2 の選定材料）

| 案 | 内容 | Observer/authority 契約との整合 | 主な懸念 | 評価 |
|---|---|---|---|---|
| **A** | MEM_SNAP（R1）の DSP 解決を RuntimeWorld/current world 経由に変更。`runtimeReadHandle`（timerCallback 冒頭 :428 で取得・callback 全体で生存）の下で `resolveActiveRuntimeDSPFromRuntimeWorldOnly` を使用 — RT path（Latency.cpp:85-89 comment）と同一解決 | **整合**。Observer は world authority から read し、read section 中は EBR が current DSP の retire→destroy を遅延する（retire は read section 開始より後の epoch で enqueue されるため） | (1) TRK の意味が「legacy slot の placeholder」→「world current DSP」に変化（観測対象としてはより正確・telemetry 契約の明記が必要）。(2) world 未公開期は TRK=0（実害なし・旧挙動との差分は diagnostic ログ比較で吸収） | **推奨** — 最小差分・契約整合。既存の `runtimeReadHandle` を再利用するため新規 atomic/queue/authority は不要 |
| **B** | destroy/retire 側で slot を CAS clear | **不整合**。「非所有 mirror・ownership authority に昇格させない」（RC-D169-1-2）の契約を lifetime coordination participant に変えてしまう。destroy path が raw pointer 値で slot 突合を行うことになり、**D169-1 で排除した pointer-value identity 突合パターンの再導入**（address reuse で誤突合 → 二重破壊クラスの再燃リスク） | 契約破壊 + 既に実証済みの危険パターン | **却下推奨**（ユーザー警告どおり安易に採用しない） |
| **C** | raw DSP dereference を廃止し、`diagFootprint`（DSPCore 保存値・D117_DESTROY で出力済み）/ LiveAllocRegistry / engine 級 counter から TRK を集約 | 整合可能だが authority 設計が新規（「生存 DSP の列挙」は handle map mutex か新 registry が必要） | (1) 観測精度: per-category breakdown（OS/EQ/AL/LT）は現在 `this` の config 値から計算 — registry 化すると同一計算を live DSP 全体に適用する機構が要る。(2) 変分量が最大 | **予備**（A の意味論変化が許容できない場合の fallback） |

---

## P6 — STOP / GO 判定

### GO 条件の検証（いずれも不成立）

1. 「lifetime が別の既存 invariant により完全に証明できる」— **否**。slot と DSP lifetime を結ぶ invariant は存在しない（P2-3/P2-6）。
2. 「MEM_SNAP reader が lifetime protection を正しく取得している」— **否**。MessageThread は world 読み取りの EBR reader ではあるが、slot の stale pointer はその保護の射程外（P2-4）。
3. 「dangling 状態にならないことを source transition から証明できる」— **否**。rebuild publish → retire → destroy の transition が slot を不変に保つことは writer 完全列挙で実証済み（P1-2）。

### 判定

> ## **STOP — CONFIRMED LIFETIME HAZARD**
>
> - slot は dangling になり得る（rebuild 後・release までの全期間、**確定的**）
> - reader（MEM_SNAP R1）がその pointer を dereference する（P3 実証）
> - dereference 時点の lifetime protection が存在しない（EBR 非射程・handle 検証なし）
>
> **スコープ注記**: 発現は diagnostic build（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF` 既定 → production は compile-out）。実害形態は「silent garbage telemetry + timing 依存 flaky AV」。production crash vector ではないが、Observer 契約（観測専用・dereference は lifetime 保証の下でのみ）に対する違反状態が diagnostic 運用中に常態化している。

**本 audit では修正しない。** 次工程 D172-2 = 「修復候補選定（P5 の A/B/C 比較を基礎）→ authority/invariant impact audit → implementation contract」。

### D172-2 への引き継ぎ事項

1. 推奨案 A（world 解決切替）の詳細契約: TRK telemetry 意味論の変更明記・world 未公開期の TRK=0 扱い・`runtimeReadHandle` 生存範囲の確認（Timer.cpp:428 — callback 全体スコープ実測済み）。
2. 案 B は却下方向（RC-D169-1-2 契約違反 + D169-1 の pointer-value 突合パターン再導入リスク）。
3. R2（Latency fallback）/ R3（logRuntimeTransitionEvent・dormant）の取り扱いも同じ契約審査の対象に含めること。
4. 動的実証は P4 方式 α（source 変更 0 の log 相関）で先行実施可能。方式 β/γ は D172-2 の実装窓で判断。

---

## Final verdict

| 項目 | 判定 |
|---|---|
| P1 access graph | writer 4 / dereference reader 3（R1 現役・R2 窓閉じ・R3 dormant）/ 比較専用 reader 残り |
| P2 lifetime chain | destroy 時点の slot 保証 **なし**。EBR は slot reader を保護 **しない**（stale pointer は epoch 規律の射程外） |
| P3 actual dereference | **成立**（`this` member 一式の読み取り・毎 tick 無条件） |
| P4 reproduction | source 変更 0 の log 相関（方式 α）で観測可能な設計を提示 |
| P5 repair | A 推奨 / B 却下方向 / C 予備 |
| P6 | **STOP — CONFIRMED LIFETIME HAZARD（diagnostic build scope）→ D172-2 repair contract** |

## 実測コマンド系譜（主要分）

```bash
rg -n "setActiveRuntimeDSP|releaseActiveRuntimeDSP|getActiveRuntimeDSP|activeRuntimeDSPSlot" src/
rg -n "retireDSPHandleForRuntime" src/audioengine/          # h:4332-4348 契約コメント
sed -n '4330,4360p' src/audioengine/AudioEngine.h           # retire 契約（物理破壊は DSPLifetimeManager::retire）
sed -n '36,69p'  src/audioengine/DSPLifetimeManager.cpp     # retire → enqueueWithRetry(P, &destroyDSPCoreNode, epoch)
sed -n '38,44p'  src/audioengine/AudioEngine.Threading.cpp  # ~DSPCore + aligned_free
sed -n '360,383p' src/audioengine/AudioEngine.Threading.cpp # runCoordinatorPhase 末端 drain（CoordinatorLoop worker）
sed -n '425,432p' src/audioengine/AudioEngine.Timer.cpp     # timerCallback 冒頭 runtimeReadHandle（callback 全体スコープ）
sed -n '114,160p' src/core/EpochDomain.h                    # enterReader / exitReader（pin 範囲 = active 中のみ）
sed -n '201,210p' src/core/EpochDomain.h                    # getMinReaderEpoch（active reader のみ・基底 currentEpoch）
sed -n '24,75p'  src/core/RCUReader.h                       # RCUReader enter/exit
sed -n '1,45p'   src/core/ObservedRuntime.h                 # ObservedRuntime が RCUReaderGuard を保持
sed -n '330,377p' src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp  # collectTrackedMemoryStatistics 全体
serena search_for_pattern("getActiveRuntimeDSP\(\);")        # reader 6 file 裏付け
rg -n "logRuntimeTransitionEvent\(" src/audioengine/*.cpp    # production 呼び出し元 0（dormant 確認）
# 交差検証: cppcheck（MEM_SNAP 領域指摘 0 — 構造的 UAF は静的解析射程外であることを裏付け）・semble（位置裏付け）
```

## 限界

- 動的実証（方式 α〜γ の実行）は本 audit の scope 外（build 0 / CTest 0 契約）。P4 の設計に基づき D172-2 で実施。
- 「通常動作（runtime world 公開後）では null」という h:2265 コメントは、writer 完全列挙の結果（prepare 成功後に null 化する経路なし）と**矛盾**している。コメント自体が stale か、あるいは publish 後 clear が意図されていたかは D172-2 の契約審査で確定させる。
