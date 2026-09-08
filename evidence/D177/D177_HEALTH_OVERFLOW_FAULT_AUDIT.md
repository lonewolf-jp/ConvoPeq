# D177 — NonRT Health / Backpressure / Overflow / Fault Recovery Audit（evidence）

- 日付: 2026-09-08
- Type: **read-only structural + behavioral contract audit** — production source 0 / test source 0 / CMake 0 / build・CTest 0
- Authority stamp: `ConvoPeq.md` — `Generated: 2026-09-08 20:32:26`（NEWER_SRC_COUNT=0 FRESH）
- 目的: D176 が lifetime/authority を閉じたあと、「詰まったとき・失敗したときに、所有権を失わず、無限化せず、停止しないか」を一次ソースから再証明する

---

## D177-0 — Source Authority【PASS】

- `ConvoPeq.md` Generated **2026-09-08 20:32:26** / `--check` STATUS FRESH（NEWER_SRC_COUNT=0）
- 基準 commit 54ba7b40（D172-3）以降、source 変更なし（D175-2 は commit 前、D176/D177 は read-only）

## D177-1 — Overflow / Backpressure Conservation【PASS】

### ① owned object（ptr）チェーン — `enqueueWithRetry`（ISRRetireRouter.cpp:303-380 全文実測）

```text
Stage 1  D（DeferredDeletionQueue）: enqueueRetire 成功 → D owns ptr ✅
Stage 2  bounded retry（kMaxRetry=2）: tryReclaim + drainEmergencyAndTerminal → 再 enqueue
Stage 3  Q（RetireQuarantineStore）: stored → Q owns ptr ✅
Stage 4  E（EmergencyQuarantineStore）: estored → E owns ptr ✅
Stage 5  T（TerminalReclaimAuthority）: growable → 常に受領（tstored 常に true）→ Terminal owns ptr ✅
```

- 原文不変式: 「**ptr を手放す前に、必ず次の authority に ownership が移る**」「**assert(false) → return という経路は残さない（Release で L>0 が発生する）**」「enqueueWithRetry() never returns with ptr unowned」（:25）
- 移送は単一方向・各段の full 時のみ次段（二重登録構造なし）。Q full 時に `drainAllUnsafe` を**呼ばない理由**まで明記（「ptr はまだ Q に無いため空にしても空きは増えない」— 誤った退避解放の排除）
- epoch unsafe 時は保持（drain が epoch safe になった時に解放）— 無限化なし（entry は保持され、reader 通過後に確定解放）
- Q/E/T のどの段に入ったかも `reason` 文字列で追跡可能（"enqueueWithRetry:QueuePressure" / ":EmergencyQuarantine" / ":TerminalReclaim"）

### ② RT 側単発経路（enqueueRetire・ISRRetireRouter.cpp:331-368）

- QueueFull → **500ms cooldown 付き**強制 tryReclaim 1 回 → 再試行 → 失敗時 `m_overflowCount_++`（Rate 監視）+ QueuePressure 返却 — **silent drop なし**
- **生産 caller 実測**: RT 側 `retireRT` / `RefCountedDeferred::releaseRT` の本番呼び出し元 = **0 件**（interface 予約・dormant）。稼働中の release は NonRT のみ（`release()` → `Router::retire` → P-4 chain）。→ dormant API として Finding F-1 記録

### ③ resident counter と実 container の一致

- quarantine 時: store mutex 下で `residentAtomic_` increment → **単一 signal point** `signalDrainWakeup()`（drainCvMtx_ acquire 後に notify — B-R3 fix・lost-wake 窓排除、:373-379 実測）
- drain 時: 抽出（Entry clear）→ lock 外 deleter 実行 → `fetchSubAtomic(residentAtomic_, pending.size())`（TerminalReclaimAuthority::drain:77-79）— counter と container 同期
- isFullyDrained Layer 1 は実測値直接参照（pendingReclaimHandles_ / pendingRetireCount — INV-X3-5・dash2 §1.4 B0-7 で上書き系 setter 廃止済み）

### ④ RetireIntent（純値・slot 調整 metadata）の 4 段 fallback（ISRRetire.cpp:30-75）

```text
MPSC slot（256）full → bounded spin 64 → tombstone
  ↓
fallbackQueue_（mutex・容量あり）→ full
  ↓
OverflowRing（SPSC lock-free・容量 16384・RT-safe tryPush）→ full
  ↓
dropped = true → fallbackOverflowCount_++ + overflowCount_++ + droppedIntentCount_++
```

- **drop は silent ではない**: droppedIntentCount_ + quarantineFallbackDropCount + recoveryIntentDropCount を **backpressure 判定に入力**（AudioEngine.Retire.cpp C-1.3 実測: drop delta>0 一度でも → **Critical 昇格** → RuntimeHealthMonitor injectBackpressureSignal → PolicyEngine 停止判断 — BUG-015/027 配線漏れ解消）
- intent は所有権を運ばない（ptr は P-4 chain 側）ため drop しても lifetime violation 不成立。slot 側 durable 台帳 + escalation（quarantineSlot RetireDeferralTimeout — D176-8）で再駆動可能

### ⑤ quarantine capacity 到達後の terminal path は bounded

Q→E→T の固定 5 段（retry 2 含む）で、再帰・無限 loop なし。T は growable だが epoch unsafe 時保持分は quiescence 後確定解放。

### ⑥ shutdown 中 overflow と通常時 overflow の authority 混線

- `isShutdownInProgress()` 分岐は `shutdownReclaim(ptr, deleter, epoch, type)`（TerminalReclaimAuthority 移送・「caller → authority への ownership transfer が成立してから return」）— ptr を捨てない（P-4・enqueueDeferredDeleteNonRtWithResult:4304-4313 実測）
- shutdown 専用 drain: `drainPendingRetireIntentsForShutdown`（ReleaseResources.cpp:790-855）— OverflowRing pop 全数 → emitRetireIntent 再注入 → MPSC/fallback dequeue → `reclaim(dspSlot)`、**bounded 3 iter**（「no RT thread is pushing」前提の再注入検知ループ）
- オーバーフロー元（通常時 `emitRetireIntentRT` 呼出元は NonRT commit path — `onRuntimeRetiredNonRt`）と shutdown drain は同一 LifetimeState authority の読み出し側 — authority 混線なし

## D177-2 — HealthMonitor 責務境界【PASS — observation-only】

### 保持 authority 参照の使用実態（RuntimeHealthMonitor.cpp 全 call-site 走査）

| 参照 | 使用 | 分類 |
|---|---|---|
| `m_retireRouter` | pendingRetireCount / activeReaderCount / detectStuckReaders / terminalStoreCount / terminalReclaimResidentCount / emergencyQuarantineResidentCount / quarantineOverflowCount | **全て const 読み取り** |
| `m_orchestrator` | getPendingIntentCount / hasDeferredRequest / getPublicationBacklogCount / isPublicationStalled / getMaxDeferredAgeMs | 読み取り |
| `m_orchestrator` | **`updateProgressObservation()`（:334、唯一の書き込み）** | 停滞監視 own state（lastObservedSequence/progressTimestamp — relaxed atomic・world/crossfade/ownership 無影響）Orchestrator.h:251 実測 |
| `m_crossfadeRuntime` | isPending / getFadeAgeUs | const 読み取り（**policy 変更 0**） |

### 判定軸

- `publish`: 0 件 — 直接 publish なし。Restore step2 の `publishIdleWorldOnly`（CtorDtor.cpp:72-82 callback）は **AudioEngine 側が決定**（`m_restoreGeneration_` vs `m_lastHardResetGeneration_` CAS 重複排除・MessageThread read handle で dsp 生存解決）→ 既存単一 gateway（commitRuntimePublication）経由。monitor は「発火要求」のみ
- `retire`: 0 件
- Crossfade policy 変更: 0 件（isPending/getFadeAgeUs の読み取りのみ）
- Recovery obligation 直接生成: 0 件（executeRecoveryAction は AudioEngine の method — setActionCallback で AudioEngine 側に委譲、CtorDtor:68-69）
- Fault 判定の漏れ: monitor は `emitValidationEvent`（記録）のみ — Faulted 遷移は Coordinator state の所管
- telemetry の ownership/state 副作用: 0 件（全 atomic read・event emit）
- executeRecoveryAction の作用（Timer.cpp:1812-1874 実測）: suppression flag / tryReclaimResources / drainDeferredRetireQueues(false) / requestDeferredClear / epoch rollback（canRollback gate）/ learner rollback / emergency drain request / admission strict flag — **すべて既存 NonRT authority の正規 API 呼び**。RT 待機・publish・直接 delete なし

## D177-3 — Faulted / Shutdown abnormal path【PASS — ownership 残留なし】

### Faulted 発生源（ISRRuntimePublicationCoordinator.cpp 実測 4 箇所）

- :84 / :104 / :127（publish 事务・sequence/epoch 整合違反系）、:204（counter 不整合 — 「他カウンタと異なり count==0 で Faulted」）
- Faulted 効果: publish() の `coordinator_.getState()==Faulted → swap しない`（commit-before-swap — 二十二次レビュー必須修正 1）+ Proof 生成不能

### Faulted / abnormal 時の各 store 残量 disposition（dtor 実測 — CtorDtor.cpp:236-300）

| store | shutdown 処置 | 残留可能性 |
|---|---|---|
| graceful drain | closeReaderRegistration → max 5s（publishEpoch+tryReclaim tick）→ timeout `[AUDIT] forcing drain` | 観測付きで強制進行 |
| D（DeferredDeletionQueue） | drainDeferredRetireQueues(true) → epoch-gated → quiescence 時 drainAll / stuck-reader 時 `m_epochDomain.drainAll()`（AudioThread 停止済みで slot 到達不能＝安全） | なし（強制 drain） |
| Q + E + T | quiescence 時 `m_retireRouter->drainAll()` / stuck-reader 時 **drainAllQuarantineStore()（epoch 非依存・「Audio Thread 停止後のみ」契約）** | なし |
| RetireIntent（MPSC/fallback/OverflowRing） | drainPendingRetireIntentsForShutdown（re-emit→dequeue→reclaim、bounded 3 iter）→ 残余 intent は slot 台帳側で保持（intent は値のみ） | ptr 残留なし |
| pendingReclaimHandles_ | drainDeferredRetireQueues の epoch 再試行ループ（Retire.cpp:88-140）| epoch 安全まで保持＝安全側 |
| RetryScheduler | `queue_.clear()`（shutdown）— RetryScheduleRequest は純値（kind/reason/class/policy の 4 field・所有権なし）discard 安全 | なし |
| Recovery obligation | ShutdownDiscarded → terminal 台帳 closure（recoveryShutdownDiscardCount_）| なし（obligation は ptr 非所有） |
| publish() 失敗（Faulted 由来） | owner unique_ptr が消費され delete（未公開 world＝PublishedDomain 外＝NonRT 即時 delete 安全）— leak なし・double-delete なし | なし |
| 残留観測 | D162-2-E（E-3/INV-D162-8）: 全 store drain 後 pendingRetireCount 確認 → 残留時 member teardown 持ち越しをログ（観測のみ） | 観測あり |

### 「安全に完了」と「正常系扱い」の区別（ご指摘対応）

- Faulted は**異常系**: dtor の `[FAULT] coordinator in Faulted state after markShutdownComplete`（D172-3 run2 で実観測）+ `[AUDIT] Graceful drain timeout` + `[DRAIN] stuck-reader fallback` はすべて異常シグナルとして記録される経路が独立している
- **安全側のトレードオフ**: stuck reader（activeReaderCount>0）時、強制 delete による UAF を回避するため「leak rather than UAF」を選択。process-exit 回収で最終所有権は失われない（D162-2-I1-D-R0 §5 で既判定の設計）。これは「Faulted が正常系として扱われている」ことではなく、**UAF 回避を優先した明示の異常処理**
- Faulted でも drain/reclaim 進行路（epoch-gated / 契約充足の強制 drain）が閉じており、**ownership の宙吊り（ptr 未所持者・二重所持）は生成されない**

## D177-4 — Retry / Recovery / Overflow cross-domain【PASS — 混線なし】

| ドメイン | counter | queue / capacity | wake 経路 |
|---|---|---|---|
| Site 3 warmup retry | `warmupRetryCount`（function-local・generation rebind） | RetryScheduler `queue_`（kCapacity=8）| signalDrainWakeup 無関係（CV own） |
| Recovery spin | `recoveryConsecutiveFailures`（K=4・local） | `recoveryIntentQueue_`（256）+ durable slot 1 | rebuildCV |
| Recovery obligation | `recoveryRetryDeferredCount_` 等 6 counter（D171-1 2-C）・D105-R18 Dual-LP | Coordinator slots_（32/256）| recoveryRetryReady / publishRetryReady（D135-7/8 provenance 分離） |
| Retire backpressure | `overflowCount_`（ISRRetire.h:156・LifetimeState＝**intent 系**） / `m_overflowCount_`（ISRRetireRouter.h:397・**D queue 系**）— 同名別クラス・共有なし | D 4096 / Q / E / T / OverflowRing 16384 | signalDrainWakeup（Q/E/T 単一点） |
| Shutdown discard | `recoveryShutdownDiscardCount_` / `recoveryObligationShutdownDiscardCount_` | — | — |

- **retry が backpressure 解消のために ownership authority を迂回する経路: 0 件**（実測）— Site 3 schedule reject = attempt 消費 drop（ND-07 §7・ptr なし）・fallback = submitRebuildIntent（intent authority 経由）・Recovery retry = DurablePending（durable slot 台帳経由）・Retire retry = Router 内部 tryReclaim（正規 authority）・`releaseDirect()`（EBR 迂回即時 delete）は **ShutdownPhase::Destroy 以上限定契約**（RefCountedDeferred.h:44-45 — publish 後呼び出し禁止明記）で publish path から到達経路なし
- capacity/counter の相互引用なし（上表の定義箇所が別クラス・別名前空間）

## 判定表

| Gate | 判定 | 根拠（一次実測） |
|---|---|---|
| D177-0 Source Authority | **PASS** | 20:32:26 FRESH・NEWER_SRC_COUNT=0 |
| D177-1 Overflow | **PASS** | P-4 5 段 chain（never unowned）・RT 側 QueuePressure 監視付・drop 昇格・dormant RT caller 0 |
| D177-2（表中 Backpressure） | **PASS** | bounded（全段有限）・silent loss なし（drop → Critical 昇格）|
| D177-2 HealthMonitor | **PASS** | observation-only・唯一の書込みは own progress state・action は AudioEngine 委譲・既存 gateway 経由 |
| D177-3 Faulted shutdown | **PASS** | 全 store disposition 実測・Faulted は異常系区別（[FAULT]/[AUDIT]/[DRAIN]）・leak-rather-than-UAF は明示設計 |
| D177-4 Retry cross-domain | **PASS** | counter/queue/capacity 分離・ownership 迂回 0 件・releaseDirect は Destroy-phase 限定 |

## Findings

### blocking: **0 件**

### non-blocking（audit finding / future design consideration として記録・今回は修正対象外）

| # | 記録 | 深刻度 | 種別 |
|---|---|---|---|
| F-1 | `RefCountedDeferred::releaseRT` / `IRetireRouter::retireRT` は本番 caller 0 件の dormant interface。将来 RT から使用する際、`QueueFull`（false）の呼び出し側処置が未定義（refcount 0 + 未 enqueue = 回収不能 object の潜在経路）。**使用開始時に契約確定が必要** | LOW（dormant） | future design consideration |
| F-2 | stuck-reader fallback は「leak rather than UAF」+ process-exit 回収（15-P-5 明示設計）。`[DRAIN] stuck-reader fallback` / `[FAULT]` ログは異常シグナルとして監視継続 | INFO（設計トレードオフ） | monitoring item |
| F-3 | D176 N-1（Case 3 μs 窓）/ N-2（Faulted ログのみ）/ N-3（reclaim 命名多層）— 引き継ぎ、判定変化なし | INFO | known boundary |

## Final Decision

> ## **D177 PASS — blocking finding 0 件・すべて PASS / INFO**
>
> 「詰まったとき・失敗したとき」の三原則が現行実装で成立:
> 1. **所有権を失わない** — ptr は 5 段 growable-final chain（never unowned）+ intent は純値（durable 台帳再駆動 + drop→Critical 昇格）
> 2. **無限化しない** — 全段 bounded（retry 2・spin 64・ring 16384・scheduler 8・drain 3 iter・5s graceful timeout）
> 3. **停止しない** — Faulted でも drain/reclaim 進行路が閉じている（commit-before-swap で publish だけ安全に止まり、lifetime は独立進行）。Monitor は観測主体のまま（直接 publish/retire/crossfade 変更 0・Recovery obligation 生成 0）
>
> → **次は D178（通常開発サイクル復帰）**。D172→D177 の read-only 一巡で lifetime / authority / overflow / backpressure / health / fault の全域が閉じた。これ以上の総合監査は収益逓減。新規 track は指示・trigger 発生時のみ開始。

## 実測コマンド系譜（主要分）

```bash
sed -n '303,380p' src/audioengine/ISRRetireRouter.cpp     # enqueueWithRetry 5段（P-4）
sed -n '331,368p' src/audioengine/ISRRetireRouter.cpp     # enqueueRetire RT側・500ms cooldown・overflow counter
sed -n '30,75p'   src/audioengine/ISRRetire.cpp           # intent 4段 fallback（tombstone→fallback→ring→dropped）
sed -n '65,110p'  src/audioengine/ISRRetireOverflowRing.h # SPSC tryPush/pop/drainAll
sed -n '198,235p' src/audioengine/AudioEngine.Retire.cpp  # drop→Critical 昇格（C-1.3）
grep -n "m_retireRouter->\|m_orchestrator->\|m_crossfadeRuntime->" src/audioengine/RuntimeHealthMonitor.cpp  # 20 call-site 全読取（書込み1=progress state）
sed -n '245,257p' src/audioengine/RuntimePublicationOrchestrator.h  # updateProgressObservation
sed -n '71,82p'   src/audioengine/AudioEngine.CtorDtor.cpp          # RestoreStep2 callback（単一gateway publish・CAS dedupe）
sed -n '1836,1905p' src/audioengine/AudioEngine.Timer.cpp           # executeRecoveryAction 全 action
sed -n '236,300p' src/audioengine/AudioEngine.CtorDtor.cpp          # Faulted/shutdown drain・residual 観測
sed -n '790,855p' src/audioengine/AudioEngine.Processing.ReleaseResources.cpp  # intent shutdown drain（bounded 3）
grep -rn "releaseRT\|retireRT(" src/                                   # 本番 caller 0（dormant 確認）
grep -n "overflowCount_" src/audioengine/ISRRetire.h src/audioengine/ISRRetireRouter.h  # 同名別クラス確認
```
