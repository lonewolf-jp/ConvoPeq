# P1-5-IR-P2 — Step 5-AM / P3-5-R31: Terminalization Observability Redesign Gate

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-5-R31）
- **種別**: **read-only design / source audit**。production 変更 0・test 変更 0・counter 0・getter 0・
  build 0・run 0（R31 では実装しない）。
- **判定**: **R31-B（主）**。
  terminalization は **既存 shutdown contract 内で観測可能**であることが source＋既存 test 実績で確定した
  （`h.stop()` → `getPhase()==ShutdownComplete` ＋ `collectResult().completed==true`）。
  R30 の STOP-E は **観測子の選択と実行順序**（Running 中の `waitForDrain`）が原因であり、
  recovery 経路の欠陥ではない。修正は **test-only vehicle の実行順序のみ**。
  ただし **per-episode T1 識別と E2（二重 terminalization 不存在）の証明は既存 API では不能**
  → その部分は **R31-C**（新観測子が必要・本 Step では実装しない）。

---

## 1. State Freeze

```text
HEAD 1e9e63e34bed7adb9342ebc81259ded9689fc48a
kP15FullMatrix = true／CONVOPEQ_CORRECT_POLYPHASE_GAIN = OFF
R16/R19/R22/R27 counters／R23 vehicle／R30 D1 vehicle（--p1-recovery-origin）保持
ConvoPeq.md = R30 で再生成済み（2026-09-23 23:04:35）を R31 の authoritative source とする
production diff = R27 のみ／test diff = R23 vehicle ＋ R30 D1 vehicle／CMake clean
R31 の差分は本ドキュメントのみ（read-only gate）
```

- `ConvoPeq(3).md` は本環境に存在しない（探索 root 全域）。最新 `ConvoPeq.md` を authority とした。

## 2. Latest ConvoPeq Source Reconciliation

R31 §1 の対象を実コードで再確認（R30 の結論を仮定せず再検証）：

| 対象 | 位置 | 実コード確認 |
| --- | --- | --- |
| `AudioEngine::waitForDrain` | AudioEngine.h:1601／Threading.cpp:215-247 | `jassert(phase ∈ {AudioStopped … ShutdownComplete})`（:221-229）・`jlimit(1,10000,timeoutMs)`・loop `while(!isFullyDrained())` |
| `AudioEngine::isFullyDrained` | Threading.cpp:153-213 | 8 predicate ＋ `runtimePublicationBridge_.isFullyDrained()`（:212） |
| `AudioEngine::collectDrainAudit` | AudioEngine.h:1604／Threading.cpp:109-151 | 露出 field は §3 の表のとおり |
| waitForDrain 内部 caller | ReleaseResources.cpp:253（100ms）／:615（2000ms） | **releaseResources のみ** |
| `releaseResources()` | ReleaseResources.cpp:34 | terminal intent 必須（:50-58） |

```text
R30 の記述（waitForDrain は shutdown 専用）は latest source で再確認され、修正なし。
```

## 3. waitForDrain Contract Audit

### 3.1 phase precondition（contract 明記）

```text
Threading.cpp:217-229
  ASSERT_NON_RT_THREAD();
  jassert(phase == AudioStopped || ObserverDrained || RetireClosed || EpochSettled ||
          ReclaimComplete || EmergencyDrain || TimedOut || Failed || ShutdownComplete);
  // ★ コメント: 「waitForDrain は AudioStopped 以降でのみ呼ばれる。」
```

- Running 中に呼ぶと **contract 違反**（Debug は jassert、Release は黙過）。
  R30 の `waitForDrain` は Running 中に呼ばれており、これが STOP-E の直接原因。

### 3.2 isFullyDrained() の全 predicate（Threading.cpp:204-212）

```text
!hasDeferredCommit                                    (orchestrator->hasDeferredRequest())
&& pendingReclaimEmpty                                (pendingReclaimHandles_.empty())
&& retireDepth == 0                                   (router->pendingRetireCount())
&& lifetimeRetireIntentPending == 0                   (lifetime().pendingIntentCount())
&& ringResident == 0                                  (lifetime().getOverflowRing()->residentCount())
&& dspQuarantineResident == 0                         (dspQuarantineManager_.residentCount())
&& retireQuarantineResident == 0                      (router->quarantineResidentCount())
&& terminalReclaimResident == 0                       (router->terminalReclaimResidentCount())
&& runtimePublicationBridge_.isFullyDrained()
```

`runtimePublicationBridge_.isFullyDrained()`（Coordinator.cpp:506-561）はさらに：

```text
swapPending_ == false
&& intentQueue_.sizeApprox()==0 && observeDeferredRing_.size()==0
&& quarantineFallbackQueue_.sizeApprox()==0 && recoveryIntentQueue_.size()==0
&& retireBacklogCount_==0 && publicationBacklogCount_==0
&& publicationIntentResidencyCount_==0 && pendingIntentCount_==0
&& reclaimInFlightCount_==0
&& quarantineIntentResidencyCount_==0 && quarantineRingResidencyCount_==0
&& !recoveryAdmissionPending_
&& liveLogicalRecoveryObligationCount() == 0          ← ★ T1 が T3 に内在（D105-R13）
```

### 3.3 collectDrainAudit() が露出する predicate

| isFullyDrained predicate | collectDrainAudit field |
| --- | --- |
| hasDeferredCommit | `deferredPublish` ✅ |
| pendingReclaimEmpty | **露出なし** ❌ |
| retireDepth | `routerPendingRetire`（fallback 込み）△ |
| lifetimeRetireIntentPending | `pendingRetire` ✅ |
| ringResident | `overflowRingResident` ✅ |
| dspQuarantineResident | `quarantineResident` ✅ |
| retireQuarantineResident | `quarantineResident` に混在 △ |
| terminalReclaimResident | **露出なし** ❌ |
| coordinator 内部（queues／residency／reclaimInFlight／recoveryAdmissionPending／**liveCount**） | **露出なし** ❌ |

- → R30 の baseline「全 audit 成分 0 なのに isFullyDrained=0」は、**非露出側**
  （`pendingReclaimHandles_`／`terminalReclaimResident`／coordinator 内部）が原因。
  本 Step では新 getter 禁止のため分解しない（STOP-5 対象）。

### 3.4 releaseResources 内の呼び出し順序

```text
:50-58   terminal intent を exchange（未設定なら reconfigure pass へ退避 → drain なし）
:96      shutdownRuntime_.transitionTo(AudioStopped)
:109     shutdownRuntime_.closeAdmission()
:237-238 shutdownCoordinatorLoop() / stopRebuildThread()（producer join）
:250-256 joinProducers() retry loop（outstanding()>0 の間 waitForDrain(100,1)）
:253     waitForDrain(100,1)                         ← 1 回目の drain
:259     transitionTo(ObserverDrained)
:264-265 transitionTo(RetireClosed) / transitionTo(EpochSettled)
:376     transitionTo(ReclaimComplete)
:383     transitionTo(EmergencyDrain)
:475     transitionTo(VerifyDrained)
:615     waitForDrain(2000,2)                        ← 2 回目（＝本命）
:616     timedOut = !drainedWithinBudget
:624-633 if (timedOut) markTimedOut(reason)          → phase_ = TimedOut（terminal）
:654-662 if (!drained || !isFullyDrained()) → drainDeferredRetireQueues + tryReclaim
:664     m_coordinator.finalizeShutdown(timedOut)
:744     shutdownRuntime_.transitionTo(ShutdownComplete)
```

## 4. Shutdown Call Chain（実 source 順序・仮説図の置換）

```text
h.stop()（AudioEngineHarness.cpp:38-53）
  1. stopAudioOnly()                     … audio thread join（running_=false）
  2. engine_->requestTerminalRelease()   … terminalReleaseRequested_=true（:1162-1164）
  3. engine_->releaseResources()         … terminal pass（:50-52 で intent を consume）
       ↓
  AudioStopped（:96）→ closeAdmission（:109）
       ↓  producers join（:237-238）→ ObserverDrained（:259）
       ↓  RetireClosed（:264）→ EpochSettled（:265）→ ReclaimComplete（:376）
       ↓  EmergencyDrain（:383）→ VerifyDrained（:475）
       ↓  waitForDrain(2000,2)（:615）── 成功 → drainedWithinBudget=true
       │                              └─ 失敗 → markTimedOut → phase=TimedOut（:632）
       ↓  finalizeShutdown(timedOut)（:664）
       ↓  transitionTo(ShutdownComplete)（:744）
  ← h.stop() から戻る（engine_ は生存。phase は ShutdownComplete か TimedOut/Failed）
```

**イベントの分離（同一視しない）**

| イベント | 実体 | 観測 |
| --- | --- | --- |
| AudioStopped | `ShutdownRuntime::phase_` 遷移（:96） | `getPhase()` |
| Drain 開始 | `waitForDrain` 内 loop 開始（:615） | `getLastNonTerminalPhase()` |
| Drain 完了 | `drainedWithinBudget==true`（:615/616） | （直接は非露出・T3 経由） |
| Reclaim 完了 | `transitionTo(ReclaimComplete)`（:376） | `getPhase()`/`getLastNonTerminalPhase()` |
| ShutdownComplete | `transitionTo(ShutdownComplete)`（:744） | `getPhase()==ShutdownComplete` |
| Recovery obligation terminalization | `recoveryAdmissions_.resolve` CAS（Coordinator.cpp:508-531） | **直接非露出**（T3 に内在） |

**termination の分岐**

```text
phase が TimedOut になった場合、transitionTo(ShutdownComplete) は
「terminal をスキップする遷移」としては許可されない（TimedOut は terminal かつ t < c ではない）。
→ TimedOut 後は phase は TimedOut のまま留まる ⇒ collectResult().completed == false。
（ISRShutdown.cpp:124-151 の allowed 判定 + isTerminalPhase）
```

## 5. T1 / T2 / T3 Terminalization Separation

```text
T1  Recovery obligation terminalization
      resolveRecoveryObligation（CAS Live→terminal）／liveCount→0
      ⇒ Full AudioEngine からの direct getter なし（AudioEngine passthrough 0）
      ⇒ coordinator isFullyDrained に内在（liveCount==0）

T2  Runtime / publication drain
      pending publication／bridge／retire／crossfade／deferred／quarantine
      ⇒ collectDrainAudit（部分）／coordinator isFullyDrained（残り）

T3  Engine shutdown completion
      releaseResources terminal pass → VerifyDrained → waitForDrain → finalize → ShutdownComplete
      ⇒ getPhase()／collectResult().completed／getBlockingReason()／transitionViolations
```

**関係（同一視しない）**

```text
T3 ⊇ T2 ⊇ T1（述語の包含関係）:
  isFullyDrained は T2 の全 predicate と T1（liveCount==0）を含む。
  したがって T3 が成立（ShutdownComplete かつ completed==true）すれば、
  **VerifyDrained 時点で T1 も成立していた**ことが entails される。
  ただしこれは「集約的な含意」であり、T1 を独立・per-episode に観測するものではない。
逆は成立しない: T1 成立 ≠ T3 成立（他 predicate が残り得る）。
```

## 6. Existing Observability Matrix

| 観測子 | runtime 中 | AudioStopped 後 | ShutdownComplete 後 | Recovery-specific |
| --- | --- | --- | --- | --- |
| seq | 有効（publish ごと +1） | 有効（新しい publish なし） | 有効（値固定） | ✗（起源ラベルなし） |
| coord / cmt / take / bld / req / que / dup / blo / drp | 有効 | 有効 | 有効 | ✗（Main bucket 集約） |
| collectDrainAudit() | 有効だが T1/T2 非露出分あり（§3.3） | 有効（より空に近づく） | 有効（all-zero 期待） | ✗ |
| isFullyDrained() | **false 固定**（§3.2 非露出 predicate が残る） | 単独では false のまま | true 期待（T3 成立時） | △（T1 内在・分離不能） |
| waitForDrain() | **contract 違反**（jassert） | 有効（ただし timeout 次第） | n/a（既に完了） | △ |
| lifecycle diagnostics | 有効 | 有効 | 有効 | ✗ |
| **getPhase()** | `Running` | `AudioStopped`…`VerifyDrained` | **`ShutdownComplete` または `TimedOut`/`Failed`** | △（T3 経由） |
| **collectResult().completed** | false | false | **true（ShutdownComplete のとき）** | △（T3 経由・T1 含意） |
| **getBlockingReason()** | None | None | TimedOut/Failed 時の理由 | ✗ |
| admissionState()/outstanding() | Open | Closing→Closed | Closed | ✗ |

- 「値が変わる」ことと「Recovery obligation terminalization を証明できる」ことは別である。
  Recovery-specific に真であるものは **皆無**。T3（＋T1 含意）のみ。

## 7. Recovery Episode Terminalization Model

```text
episode（R30 で確立）:
  submitRecoveryIntent → obligation(oblId≠0) → transport → Builder → publish
    → seq +1／coord 0／cmt 0（3/3 再現・control dSeq=0）
  publish 成功時:
    Route A（trySubmitImpl:324）と Route B（onPublishCommitted:358）が
    同一 obligation を resolve attempt → CAS で単一 winner → liveCount −1（一度だけ）

terminalization の表現（既存 API での到達可能な観測）:
  [E0] Recovery-origin publish 発生        → 実測可能（seq/coord/cmt・R30 3/3）
  [E1] 当該 obligation の terminal 化       → **集約のみ**（後段 T3 の completed==true が含意）
  [E2] 同一 obligation の二重 terminal 化なし → **観測不能**（winner/liveCount が非露出）
```

## 8. Exactly-once Observability Analysis

### E0（発生）— PASS（R30）

```text
control dSeq=0／episode dSeq=+1・dCoord=0・dCmt=0・dTake=0・dBld=0（3/3）
→ Recovery-origin publish の発生は window 粒度で観測済み。
```

### E1（terminalization）— **既存 API で観測可能（T3 経由・集約）**

```text
R31 で確定した契約的観測:
  h.stop()（= requestTerminalRelease + terminal releaseResources pass）
    → e.isrShutdownRuntime().getPhase() == ShutdownPhase::ShutdownComplete
    → e.isrShutdownRuntime().collectResult(health,0).completed == true
    → transitionViolations == 0
  ⇒ VerifyDrained(:475) の isFullyDrained(:615) が true であった
     ⇒ liveLogicalRecoveryObligationCount()==0 が成立していた（T1 の含意）
     ⇒ episode 由来の obligation が terminal 化済みであることと整合
根拠（既存実績）:
  PublishPipelineIntegrationTests.cpp:632-663（D167）が h.stop() 後に
  admissionState==Closed／getPhase==ShutdownComplete／collectResult().completed==true／
  transitionViolations==0 を assert し、**default harness 実行で [PASS]**（R27/R28 実測）。
```

- 制約：E1 は **episode 単位ではない**。どの episode の obligation がいつ terminal 化したかは識別できない
  （R30 §8 の per-task attribution 制限と同じ）。
- 制約：E1 は「VerifyDrained 時点で Live が 0」の**含意**であり、terminal 化の回数・winner を言わない。

### E2（二重 terminalization なし）— **既存 API では証明不能**

```text
観測不能な理由:
  - resolveRecoveryObligation の `won` は両 call site で破棄（すでに確認済み）
  - liveCount_／winner を表す counter・getter は Full AudioEngine から非露出
  - したがって「同一 obligation が 1 回だけ terminal 化した」ことを観測で示せない

source 上の保証（観測ではない）:
  RecoveryAdmissionTable::resolve（Coordinator.cpp:508-531）は
  「w.state==Live のときのみ full-word CAS し、winner のみ liveCount_ を −1」する。
  非 Live／id 不一致／lost CAS は false（no double −1・underflow なし）。
  ⇒ E2 は **構造的に保証**されるが、これは **source contract の性質**であって
     本 vehicle の観測事実ではない（R28/R29 の「静的 PASS を動的 PASS に置換しない」を維持）。
```

## 9. Test-only Vehicle Design（実装は R32 以降）

R30 の D1 vehicle を**そのまま再利用**し、実行順序のみを契約に合わせる（新 vehicle 不要）：

```text
（R30 既存）
  start → authoritative Runtime settle → control 5s
  → Recovery episode × N（各 episode:  pre → submitRecoveryIntent → seq 前進待ち → post）
（R31 の順序修正・test-only）
  → 全 episode の publish 完遂を確認（in-flight recovery なしを保証）
  → h.stop()                       … terminal intent 発行 + terminal pass
  → e.isrShutdownRuntime().getPhase()            == ShutdownComplete ?
  → e.isrShutdownRuntime().collectResult(h,0)    .completed == true ?
  → transitionViolations == 0 ?
  → （参考）e.collectDrainAudit() / e.isFullyDrained()
```

- **Running 中および stopAudioOnly 直後の `waitForDrain` 呼び出しは削除**（contract 違反・R30 STOP-E の直接原因）。
- 再利用する観測（R31 §5）：`control dSeq=0`／`episode dSeq=+1`・`dCoord=0`・`dCmt=0`・
  `dTake=0`・`dBld=0`。
- 追加の新 counter／新 getter／新 logger は不要（すべて public 既存 API）。

### 設計上の注意（D167 実測の finding を継承）

```text
PublishPipelineIntegrationTests.cpp:625-630 が記録する
「in-flight rebuild × terminal race」に留意し、terminal 移行前に
recovery publish の完遂（seq 前進＋settle）を待つことで窓を閉じる。
R30 vehicle は episode ごとに seq 前進を待っているため、この条件は既に満たす。
```

## 10. Production Change Necessity

| 目的 | 既存 API で可能か | 必要な変更 |
| --- | --- | --- |
| T3（shutdown completion）観測 | **可能** | なし（`getPhase()` / `collectResult()`） |
| T1 の **集約**含意（T3 経由） | **可能** | なし |
| T1 の **per-episode** 識別 | **不能** | 新 observability（counter/getter または linkage）が必要 |
| E2（二重 terminalization 不存在）の観測 | **不能** | 新 observability（winner/liveCount 露出）が必要 |
| T2 の全 predicate | 部分 | `pendingReclaimHandles_`／`terminalReclaimResident`／coordinator 内部の露出が必要 |

- **R31 では production 変更・新 counter/getter を実装しない**（指示 §7・§8 STOP-5）。
- T3/T1 集約は **test-only vehicle の順序修正のみ**で達成可能（production 0・CMake 0）。

## 11. STOP Classification

```text
STOP-1（waitForDrain/isFullyDrained が shutdown 専用である再確認）        : **該当**
        Threading.cpp:217-229 の jassert/コメント・内部 caller が releaseResources のみ。
STOP-2（existing API だけでは Recovery obligation terminalization を識別不能）: **部分的に該当**
        集約（T3 経由）は可能。per-episode 識別は不能。
STOP-3（shutdown completion と recovery obligation completion の対応関係が証明不能）: **非該当**
        isFullyDrained が liveCount==0 を含むため、T3 成立 → T1 成立の含意が source 上証明可能。
STOP-4（winner identity なしでは exactly-once を証明できない）            : **該当**
        E2 は観測では示せない（構造的保証のみ）。
STOP-5（observability 追加に production counter/getter が必要）          : **該当**
        STOP-2（per-episode）・STOP-4 の解消には新 observability が必要。
STOP-6（Recovery semantics 変更が必要）                                   : 非該当
```

- STOP-1/2（部分）/4/5 が該当 → **本 Step はここで停止**（実装しない）。

## 12. R31 Gate

```text
R31-A（既存 shutdown contract ＋ existing observability だけで十分観測可能）:
  **部分的に成立**（T3/T1 集約は可能）。ただし per-episode T1・E2 は不能。
R31-B（shutdown contract 内なら観測可能だが、現 vehicle の実行順序が不適切）:
  **ADOPTED（主判定）**
  R30 STOP-E の真因は「Running 中の waitForDrain」＋「terminal 結果を読まない」という
  実行順序の誤り。h.stop() 後の getPhase()/collectResult() で契約的に観測可能
  （D167 が既存 test で同型を assert し PASS 済み）。
  ⇒ 修正は **test-only vehicle の順序のみ**（production 0・CMake 0）。
R31-C（terminalization は完了している可能性があるが既存 API では観測不能）:
  **per-episode T1 と E2 に限り該当**。新 observability が必要（実装しない）。
R31-D（source 上 terminalization 自体の contract が不明確）:
  非該当（resolve の single −1 CAS・isFullyDrained の liveCount 条件は source 上明確）。
```

- したがって R31 の結論は **R31-B を主、R31-C を従（per-episode/winner 限定）**とする。
- R30-B/STOP-E は「観測子の選択ミス」であり、**Recovery 経路・D105 契約の欠陥ではない**ことを確定。
- source は R27 production＋R30 test vehicle を保持。revert なし。
- R4 境界・保留事項・P3-5 §7解釈制約・H-B 対象外を維持する。

## 13. Next Step

```text
R31-B → R32 = test-only vehicle 順序修正（terminal observation の契約化）
  対象: --p1-recovery-origin
  変更: episodes 後の waitForDrain(20s)/drain_pre/drain_post/drain_after_audio_stop を撤去し、
        h.stop() 後に getPhase()/collectResult()/transitionViolations を観測する形へ置換
  制約: production 0・CMake 0・新 counter/getter 0（既存 public API のみ）
  期待: episodes seq+1/coord0/cmt0 の再現 ＋ ShutdownComplete/completed==true
        （= T3 成立、したがって T1 の集約含意）

R31-C（従）は別 gate:
  per-episode T1 識別 / E2 winner observability が必要になった時点で、
  production observability 追加の是非を独立に design audit する（本 Step では実装しない）。
```

- R31 は本ドキュメントで完結（read-only）。R32 の vehicle 修正・R31 Full-pipeline Measurement には進まない。
