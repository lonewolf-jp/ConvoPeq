# D169-2-1 — Duplicate-Prepare Collapse / Abort Race Audit（read-only）

```text
D169-2-1 — Duplicate-Prepare Collapse / Abort Race Audit
Date:        2026-09-07
Type:        read-only source audit（production/test/CMake/build.bat/tool 変更 0・実装禁止）
Prior:       D167 §8-2 finding 2（0xC0000409 実測）・D169-1 / D170 は physical lifetime track（本 audit の根拠に不使用）
Audit target: prepare transaction state machine（physical DSP lifetime ではない）
Verdict:     **Case A — Defect confirmed（availability / fail-stop クラス）**
             memory-safety defect（leak / 二重 destroy / ownership 二重化）は不成立
```

---

## 0. 判定サマリ

> 同一 SR/BS の **sequential duplicate prepare** が `LifecycleIsolationRuntime::enterPrepare`
> の collapse 経路（`ISRLifecycle.cpp:27-36` — phase を Prepared のまま token 返却）と
> `prepareToPlay` 本体（早期 return 無しで全 body を実行）と `leavePrepare` の
> `phase==Preparing` 前提（`ISRLifecycle.cpp:52-55` → `std::abort()`）の **3 者の契約不整合**
> であり、production JUCE 経路（同一 SR/BS での device restart）で到達可能。
> 帰結は memory corruption ではなく **`std::abort()`（0xC0000409）による process 異常終了**。
> 並行 duplicate prepare（P1/P2 同時進行）は単一 Message Thread 入口 + enterPrepare の
> overlap abort で構造的に阻止される（LIF-1 設計どおり）。

## 1. Caller map（prepareToPlay 到達経路の全列挙）

`AudioEngine::prepareToPlay(int, double)` の production caller は **1 系統のみ**:

| # | Caller | 位置 | Thread | Synchronization | 再入防止 | 前回 prepare 完了保証 | Transaction identity | Rollback ownership |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `AudioEngineProcessor::prepareToPlay` | `AudioEngineProcessor.cpp:35-38` | Message Thread（JUCE AudioDeviceManager device lifecycle） | JUCE が device open/close を Message Thread で直列化。`AudioProcessorPlayer::audioDeviceAboutToStart` → `setProcessor` swap（`juce_AudioProcessorPlayer.cpp:364-372` → `:180`）経由 | engine 側: `lifecycleState` CAS（Preparing/Releasing/Destroyed で block、`PrepareToPlay.cpp:51-67`）+ `lifecycleRuntime_.enterPrepare`（overlap abort 強制） | JUCE 直列化 + lifecycleState CAS（block 方式） | **無し**（nominal `LifecycleToken`・`leavePrepare` は token を照合しない） | `rollbackPrepareFailure`（engine 内部バッファ）+ CallerDestroy（placeholder） |
| 2 | `AudioEngineHarness::start`（test） | `AudioEngineHarness.cpp:32` | test thread | テスト単一直列 | 同上 | test 直列 | 同上 | 同上 |
| 3 | `PublishPipelineIntegrationTests` ×3（test） | `:235/:577/:608` | test thread | テスト単一直列 | 同上 | test 直列 | 同上 | 同上 |

備考:
- `AudioEngine.h:692/729` は `ConvolverProcessor`/`EQProcessor` の prepareToPlay（AudioEngine ではない sub-component）で、いずれも `AudioEngine::prepareToPlay` 本体内部からの呼出。
- JUCE `AudioDeviceManager::setAudioDeviceSetup` は **同一 setup の場合 early return**（`juce_AudioDeviceManager.cpp:815-818`）— 同一 setup の apply は re-prepare を起こさない。
- device restart（type 切替・reopen・recovery）は必ず `audioDeviceStopped` → `audioDeviceAboutToStart` の順で呼ばれ、AboutToStart 側は player の `isPrepared` が `audioDeviceStopped` で false 化済みのため `:367` の release を skip し `setProcessor` swap で **prepareToPlay を 1 回**呼ぶ（release+prepare が 1:1 対応）。

### releaseResources 対（reconfigure/terminal — D167 境界）

| # | Caller | Thread | pass 分類 |
| --- | --- | --- | --- |
| 1 | `AudioEngineProcessor::releaseResources`（`isEnginePrepared()` guard 付き） | Message Thread | terminal intent が無ければ **reconfigure pass**（lifecycleState=Prepared 維持・phase 不変） |
| 2 | `AudioEngineHarness::stop`（`requestTerminalRelease` 後） | test thread | terminal pass（lifecycleState→Unprepared・phase Released） |

## 2. Prepare transaction lifetime graph

```text
prepareToPlay entry (Message Thread)
 ├─ ASSERT_NON_RT_THREAD                       (PrepareToPlay.cpp:18)
 ├─ lifecycleRuntime_.enterPrepare(sr, bs)     (:20  ★ lifecycleState CAS より先に phase を遷移させる)
 │    ├─ phase==Prepared && same(sr,bs) → COLLAPSE: token{epoch, Prepared} を返す・phase 不変   (ISRLifecycle.cpp:27-36)
 │    └─ else → validateTransition → phase: (Uninitialized|Prepared|Released)→Preparing        (:38-45)
 ├─ lifecycleState CAS → Preparing             (:61-67  block: Releasing/Destroyed/Preparing → early return)
 │    ★ block 時は enterPrepare の phase 遷移が undone にならない（non-collapse の場合 phase 残留）
 ├─ rebuild thread restart (if !joinable) + pendingTask/publishRetryReady reset   (:77-88)
 ├─ rebuildRequestGeneration ← 0               (:96  ★ in-flight rebuild を全て stale 化)
 ├─ resetProgressObservation                   (:101)
 ├─ SR/BS publish + rateChanged/blockSizeChanged 検出             (:121-125)
 ├─ crossfadeRuntime reset + gain re-init                        (:133-135)
 ├─ idle publish #2（world に current/fading があれば再 publish）  (:138-161)
 ├─ analyzerFifo/level/crossfade buffer re-init                  (:164-172)
 ├─ latency buffer realloc（失敗時 rollbackPrepareFailure → Unprepared + return）(:186-204, :30-44)
 ├─ lifecycleState → Prepared                  (:230)
 ├─ placeholder path（gate: !hasPublishedCurrent && !hasActiveRuntimeDSP）(:236-304)
 │    ├─ construct → prepare → setActiveRuntimeDSP(placeholderRaw)  (:239-270)
 │    ├─ idle publish #3 (needsRegistration)                        (:283-285)
 │    ├─ Transferred: slot 保持（owner = world/slot）
 │    └─ !committed && CallerDestroy: destroyRolledBackDSP + slot=null（owner = prepare 本体）(:296-302)
 ├─ uiConvolverProcessor.prepareToPlay + invalidatePendingLoads   (:307-309)
 ├─ submitRebuildIntent(Structural) if rateChanged||blockSizeChanged||!hasCurrentRuntime (:311-323)
 └─ lifecycleRuntime_.leavePrepare(token)      (:327  ★ phase==Preparing 前提 — 違反で std::abort)
```

**transaction owner**: prepare transaction に専用の owner object は存在しない。
実質 owner は **AudioEngine 自身**（lifecycleState / lifecycleRuntime_.phase_ の 2 つの FSM）で、
placeholder の ownership は (a) 成功時 = activeRuntimeDSPSlot + published world、
(b) rollback 時 = prepare 本体（CallerDestroy 契約・D162-2-I2）、に分割して委譲される。

## 3. Duplicate prepare の成立条件

### 3.1 形式 I — sequential same-SR/BS re-prepare（★ defect 本体）

```text
[t0] device 動作中（SR=A, BS=B）→ engine lifecycleState=Prepared・phase=Prepared・
     lastPrepared=(A,B)
[t1] 同一 SR/BS での device restart が発生
     → audioDeviceStopped → AudioProcessorPlayer::audioDeviceStopped
       → engine.releaseResources() = reconfigure pass（D167: lifecycleState/phase 不変）
[t2] device reopen → audioDeviceAboutToStart(SR=A, BS=B)
     → AudioProcessorPlayer: setProcessor(nullptr)+setProcessor(old)
       → AudioEngineProcessor::prepareToPlay(A, B)
[t3] engine.prepareToPlay:
       enterPrepare: phase==Prepared && same(A,B) → COLLAPSE token（phase 不変）(ISRLifecycle.cpp:27-36)
       lifecycleState CAS Prepared→Preparing 成功 → **全 body を実行**（早期 return は存在しない）
       :327 leavePrepare: phase==Prepared ≠ Preparing → std::abort()               (ISRLifecycle.cpp:52-55)
[t4] 0xC0000409（fail-fast）— process 異常終了
```

D167 §8-2 で unit test 環境の 0xC0000409 として実測済み。abort 前の body 副作用
（:96 generation reset・:138 idle publish #2・latency realloc）は実行されるが
abort により process 全体が終了するため破綻は観測されない。

**production 到達経路（4 系統確認）:**

1. `DeviceSettings.cpp:1070-1098` — recovery 経路: `closeAudioDevice()` →
   `setCurrentAudioDeviceType` → `setAudioDeviceSetup(savedSetup)`。reopen 後の
   SR/BS が直前 prepare と同一（正常動作中からの復旧の典型）なら collapse abort。
2. 同一 SR/BS を保った device type 切替（`MainWindow.cpp:457` / `DeviceSettings.cpp:1094`）—
   切替先 device が同一 SR/BS で開いた場合。
3. ASIO control panel からの同設定 reopen（driver 側で device が張り直される）。
4. device hot-plug 再接続（同一保存 setup での自動復帰）。

対照的に CLI runtime validation（D165-D170）は switch 先 SR/BS が既定と異なるため
collapse を踏まず（D168_DS: `enter spb=2560 sr=44100.00` は 1 回のみ）。

### 3.2 形式 II — concurrent P1/P2（構造的に阻止）

- prepareToPlay の production 入口は Message Thread のみ（§1）。JUCE は device lifecycle
  を Message Thread で直列化するため **P2 は P1 の完了前に開始できない**。
- 仮に cross-thread で P2 が進入しても `enterPrepare` は `phase==Preparing` を観測し
  `validateTransition(Preparing→Preparing)` invalid → abort（`ISRLifecycle.cpp:174-176`）。
  overlap は **待機ではなく fail-stop で阻止**（ISRLifecycle.h:51 LIF-1 の設計宣言どおり）。

### 3.3 形式 III — blocked-return 時の phase 残留（latent・production 不可達）

`prepareToPlay` が lifecycleState=Releasing/Preparing/Destroyed で block return する場合
（`:53-59`）、non-collapse の `enterPrepare` は既に phase を遷移済みのため
**leavePrepare が呼ばれず phase が Preparing に残留**する。production では
releaseResources/prepareToPlay が同一 Message Thread のため観測不可能だが、
残留が成立すると後続の `enterRelease`（:95-98 abort）・`enterAudioCallback`
（:70-72 abort）が全て fail-stop になる。修復契約（D169-2-2）で block 経路の
phase 復帰を扱うかを判定する。

### 3.4 形式 IV — P1 fail → rollback → P2 継続

rollback は 2 種のみ: (a) `rollbackPrepareFailure` — latency buffer のみ・DSP に触らない・
lifecycleState→Unprepared（phase は非 collapse 遷移の場合 Preparing 残留 → 後続 abort、
production では buffer alloc 失敗は Message Thread 直列内で完結し P2 は P1 完了後）；
(b) CallerDestroy — 未登録 placeholder の direct destroy（`destroyRolledBackDSP`・
registry rollback 済みのため authority 不要）。生存 registered DSP を rollback が
処分する経路は存在しない。

## 4. Generation 分析（R4/R5）

| 項目 | 事実 | 位置 |
| --- | --- | --- |
| rebuildRequestGeneration | int・prepare 開始で **0 に reset**・rebuild request 毎に ++ | PrepareToPlay.cpp:96 / RebuildDispatch.cpp:672 |
| publication gate | `req.generation != currentGen` → RejectedStaleGeneration（新 DSP は S4 authority retire で disposition） | PublicationAdmission.cpp:15-18 / Orchestrator.cpp:390-397 |
| rebuild stale check | `isRebuildObsolete(task.generation)`（等号は obsolete と見なさない） | RebuildDispatch.cpp:1174-1175 |
| prepare transaction identity | **存在しない**。LifecycleToken は epochId を運ぶが `leavePrepare(token)` は token を一切照合しない（引数未使用） | ISRLifecycle.cpp:48-58 |
| epochCounter_ | Prepared/Released 遷移で ++ するが token 照合に使われない | ISRLifecycle.cpp:206-208 |

結論: generation は **rebuild publish の logical ordering/protection** としては機能するが、
**prepare transaction validity は保護していない**。保護は (1) Message Thread 直列化、
(2) lifecycleState CAS、(3) LifecyclePhase FSM の abort 強制、の 3 層で行われている。
collapse 経路の `rebuildRequestGeneration ← 0` は in-flight rebuild を stale 化するが、
RejectedStaleGeneration → S4 disposition で安全に処理される（leak ではない）。
`generation == current` だけの commit 許可（publication admission）と、
prepare transaction identity の欠如は別問題 — 後者は本 audit の R3 判定（No）の根拠。

## 5. Placeholder / handle 対応（R7/R8）

- placeholder は `commitRuntimePublication(RegistrationContext::needsRegistration(placeholderRaw))`
  （`PrepareToPlay.cpp:283-285`）で registration + activate され、その handle は
  `activeRuntimeDSPHandle_` に公開（D170-1 確定事項の再利用）。
- **re-prepare（non-collapse）は新 placeholder を作らない**: gate
  `!hasPublishedCurrent && !hasActiveRuntimeDSP()`（`:236`）— D167 reconfigure pass が
  world/active DSP を生存させるため、Prepare B は Prepare A の DSP を継続使用する
  （world と registry は 1:1 維持）。
- Prepare A rollback（CallerDestroy）後の Prepare B: slot=null に復帰済みのため
  Prepare B は新 placeholder を作る — 古い placeholder は destroy 済みで 1:1 維持。
- 同時に 2 つの placeholder が生存する経路は無い（直列化 + gate + slot 衛生）。
  collapse 経路は DSP を 1 個も生成しないため registry 乖離も発生しない。

## 6. R1〜R12 判定

| ID | 判定 | 根拠 |
| --- | --- | --- |
| R1 duplicate prepare は source 上成立するか | **Yes**（sequential same-SR/BS 形。concurrent 形は abort で阻止） | §3.1-3.2 |
| R2 同時に複数 prepare transaction が存在可能か | **No**（Message Thread 単一入口 + enterPrepare overlap abort。blocked-return phase 残留は latent・production 不可達） | §3.2-3.3 |
| R3 transaction identity は存在するか | **No**（LifecycleToken は照合されない。prepare 専用 generation 無し） | §4 |
| R4 generation は transaction validity を完全に保護するか | **No**（rebuild publish の ordering gate のみ。prepare level は thread 直列化 + FSM abort が担う） | §4 |
| R5 stale prepare の commit が可能か | **No**（直列化により stale prepare は構造的に不在。in-flight rebuild は gen gate → S4 dispose で安全） | §4 |
| R6 stale rollback が生存 DSP を処分可能か | **No**（rollback 対象は未登録 placeholder のみ — CallerDestroy 契約。registered DSP は authority retire のみ） | §3.4 |
| R7 placeholder ownership が二重化するか | **No**（作成 gate + slot 衛生 + 直列化。re-prepare は既存 DSP を継続使用） | §5 |
| R8 handle registry と prepare state が乖離するか | **No**（collapse は registry 不触。乖離が発生しうる相は全て abort で fail-stop） | §5 |
| R9 shutdown と prepare completion の race があるか | **Conditional**（production は Message Thread 直列で race 無し。terminal release 進行中の prepare 進入は enterPrepare/enterRelease の abort（fail-stop）で process 死 — memory-safety race は無い） | §3.2-3.3 |
| R10 leak / double destroy / stale publish の具体的成立 chain | **memory-safety: No** / **availability: Yes** — collapse → 全 body 実行 → leavePrepare abort → 0xC0000409（D167 §8-2 実測）。abort 前副作用（gen reset・idle publish #2）は abort により不成立 | §3.1 |
| R11 existing tests が interleaving をカバーしているか | **No** — same-SR/BS collapse はテストが意図的に回避（`PublishPipelineIntegrationTests.cpp:594-597` コメント「SR を交互に変えて collapse 経路を回避」）。blocked-return 形も未カバー。カバー済みは Prepared→Preparing（異パラメータ・D167 test）と Released→Preparing（T-I2-1）のみ | §7 |
| R12 repair に必要な最小 state-machine change | **Direction only** — (a) collapse token を真の no-op 化（prepareToPlay が collapse を検出して early return）または (b) collapse 廃止（常に完全 re-prepare）。block 経路の phase 復帰を契約範囲に含めるかは D169-2-2 で選定 | §8 |

## 7. Existing test coverage

| テスト | 経路 | collapse/abort 系カバー |
| --- | --- | --- |
| `testD167ReconfigureKeepsAdmissionOperational`（:531-664） | reconfigure → prepare（**SR 交互で collapse を回避**・:594-597 に明記） | No（意図的回避） |
| `testCallerDestroyTerminalDisposition`（T-I2-1・:196-250） | terminal release → prepare（Released→Preparing・同一 48000/512 でも phase=Released のため collapse 不発） | No |
| `testD167AdmissionClosedTelemetry` | closeAdmission → rebuild | No |
| その他 40 tests | — | duplicate-prepare collapse / blocked-return / concurrent 形の coverage **0 件** |

## 8. Repair direction（D169-2-2 契約候補・実装はしない）

1. **候補 a — collapse を真の no-op 化**: `enterPrepare` が collapse token を返した場合、
   `prepareToPlay` は早期 return する（JUCE の re-prepare 要求を冪等 no-op として扱う）。
   副作用が最小（SR/BS 未変更・world 生存のため再初期化不要という前提が契約条件）。
2. **候補 b — collapse 廃止**: `enterPrepare` は常に `Prepared→Preparing` 遷移を行い、
   完全 re-prepare を実行させる（現行 body は冪等再初期化として設計済み）。
   leavePrepare の前提と自然に整合。副作用は現行 reconfigure→prepare と同一。
3. **共通副題**: blocked-return（:53-59）時の phase 残留を契約に含めるか
   （production 不可達のため scope 外とする選択もあり — D169-2-2 で判定）。
4. **禁止事項（D169-1R と同様の原則）**: mutex 追加・atomic 追加・generation 追加・
   transaction ID 追加・prepare lock/cancellation 追加・shutdown FSM 変更は
   最小修復の範囲外（fail-stop 構造は保持）。

## 9. 変更範囲

production 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary 0（本監査文書のみ新規）。

## 10. 判定

**Case A — Defect confirmed。** ただし defect クラスは memory-safety ではなく
**availability（fail-stop abort・0xC0000409）**。production 4 経路で到達可能
（§3.1）。D169-1/D170 で修復した physical lifetime 路とは無関係であり、
D170 の成果を本判定の根拠には使用していない（prepare transaction state machine
単独の source 証明）。

次: **D169-2-2 Repair Contract Approval**（§8 の候補選定・RC 文書化）→
D169-2-3 Preflight → D169-2-4 Minimal Implementation → D169-2-5〜7 検証。
