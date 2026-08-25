# D101-33-A — Path B Publication Admission Audit（実施報告書）

- **実施日**: 2026-08-25
- **作業種別**: read-only audit（ソースコード変更 **0**）
- **判定**: **GAP（C）— PASS 不成立。ここで停止**（D101-33-B 実装案への先走りは行わない）
- **基準**: ConvoPeq.md **2026-08-25 13:03 再生成版**
- **前提**: D101-31-D / D101-32-A〜F すべて PASS

---

## 0. 総括（先に結論）

旧監査の「Path B は shutdown gate を**経由せず** enqueuePublicationIntent に到達する」状態は
**解消済み**。現行の `enqueuePublicationIntent()` には `CoordinatorState::ShuttingDown` gate
（Coordinator.h:361-363）が存在し、X5 residency の reservation→push→rollback も単一 choke point
で正しく実装されている。

しかし、以下の理由で **PASS としない**:

1. **Gate は check-then-act（TOCTOU）であり、shutdown close と線形化されていない**
   （`state_` の acquire load と `intentQueue_.push` の間に window が存在）。
2. **Path B（commitRuntimePublication 直接 producer 群）は `AdmissionPackedState`
   （tryAdmit/release）に一切接続していない**。Q0（outstanding()==0）は Path B の
   in-flight enqueue を観測できない。
3. **ProcessIntent に Publish Intent の shutdown discard 処理が存在しない**
   （ProcessIntent.cpp 内に shutdown 分岐ゼロ）→ 閉鎖と競合して入った Publish Intent は
   shutdown 開始後でも commit（world swap）され得る。

よって本監査の判定は **No-Resurrection = GAP**、総合 = **GAP（C）**。
D101-33-B（admission transaction 設計）への具体的な入力を §6 に記録し、停止する。

---

## 1. A-1 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 13:03)
git diff --check                       → whitespace エラーなし
```

---

## 2. A-2 Path B 完全 call graph（現行ソース実測）

```
【直接 Producer 群】
AudioEngine.Processing.PrepareToPlay.cpp:155/:277
AudioEngine.Processing.ReleaseResources.cpp:175
AudioEngine.Timer.cpp:994
AudioEngine.Transition.cpp:25
    ↓ commitRuntimePublication (AudioEngine.h:4602, 同期ラッパ)
【deferred resubmit】
CoordinatorLoop → PublicationExecutor.cpp:57-66（waitForReceipt=false）
    ↓ enqueueRuntimePublicationFireAndForget（同一コア）
enqueueRuntimePublicationFireAndForget (AudioEngine.h:4524)
    ├─ registerDSPHandleForRuntime（rollbackHandle 確保, ScopeExit guard）
    ├─ makePublishDecisionSnapshot (:4559)
    ├─ worldAuthority_.registry().registerPublish (:4562)
    ├─ worldAuthority_.ownerChannel().enqueue (:4566)     … world 所有権移譲
    │     失敗 → registry.unregister + Failed/CallerDestroy
    └─ runtimePublicationBridge_.enqueuePublicationIntent (:4584)  ★最終 authority API
          【Coordinator.h:353-378】
          ① state_ == ShuttingDown gate（acquire load）→ false（queue 触らず）★linearization point と自己規定
          ② publicationIntentResidencyCount_ fetchAdd（+1, reservation-before-push）
          ③ intentQueue_.push(prepared) → 成功 true
             失敗 → residency fetchSub（rollback）→ false
               ↓ false 返却時（呼出し元 AudioEngine.h:4586-4590）
          ownerChannel().take(key) + registry.unregister → Failed/CallerDestroy
【consumer】processIntent（ProcessIntent.cpp:44-66）
    intentQueue_.pop → type 分岐で publicationIntentResidencyCount_--（Publish のみ）
    → PublishExecutor{}.executePublish (:149)
```

旧監査記載の「AudioEngine.h:4364/:4424 → Coordinator.h:324-339、state_ check なし」は
現在の行番号・実装とも変化しており、**gate は存在する**。

---

## 3. A-3 Admission authority の現状

### 3.1 Shutdown gate

| レイヤ | 存在 | 内容 |
|---|---|---|
| `CoordinatorState::ShuttingDown` | ✅ あり | enqueuePublicationIntent 冒頭（h:361-363）。requestShutdown()（ReleaseResources.cpp:75 / CtorDtor.cpp:113）で確定 |
| `AdmissionPackedState` | ❌ **Path B では未使用**（§4） |
| `isShutdownInProgress()` 事前遮断 | △ caller 側に散在（Timer.cpp:406/734/770/817/1192、Transition.cpp:15 等）。commitRuntimePublication 自体は持たない |

### 3.2 Linearization point の評価

現行 gate は:

```cpp
if (consumeAtomic(state_, acquire) == ShuttingDown) return false;   // T1: check
fetchAdd(residency);                                                 // T2: reserve
intentQueue_.push(prepared);                                         // T3: act
```

**「shutdown を確認した」と「競合しない obligation を取得した」が区別されていない。**
T1〜T3 の間に `requestShutdown()`（= state_ への release store）が完了した場合、
閉鎖後に residency+push が成立する（Case C race — §5）。
単一 CAS word での linearization は存在しない。コメント自身も「defense-in-depth 二次防衛」
と位置づけており、一次防衛（isShutdownInProgress 事前遮断）との間に隙間がある。

---

## 4. A-5 AdmissionPackedState との関係（重点監査）

### 4.1 tryAdmit の全 production site（再確認）

| site | 経路 | acquisition → release |
|---|---|---|
| RuntimePublicationOrchestrator.cpp:69（trySubmitImpl） | **Path A**（Orchestrator 経由 publish） | evaluate()→Accepted 後 tryAdmit(1)、RAII ReservationGuard。executor_.publish 成功後「obligation が durable state（ISR intent queue）に存続」した時点で release(1)（:330-333） |
| AudioEngine.RebuildDispatch.cpp:319 | Recovery（rebuild 経路） | 同型 |
| AudioEngine.h:4443（submitRecoveryIntent） | Recovery（wake 経路） | submitRecoveryRequest の gate 通過後 tryAdmit、関数末尾で release |

### 4.2 Path B の接続状態

| 項目 | 現状 |
|---|---|
| ownership | ShutdownRuntime（AdmissionPackedState）。ただし **Path B 直接 producer は誰も取得しない** |
| acquisition | Path A: Orchestrator が代理取得。**Path B 直接（PrepareToPlay/ReleaseResources/Timer/Transition → commitRuntimePublication）: 取得なし** |
| close | ReleaseResources.cpp:198 closeAdmission() — **shutdownCoordinatorLoop()(:191) および stopRebuildThread()(:192) の後**。「Producers are joined (Q2)」が前提 |
| race | closeAdmission vs tryAdmit は単語 CAS で安全（G-H）。**しかし Path B はそもそも tryAdmit しないため Q0（outstanding()==0）が Path B の in-flight を観測できない** |
| release | enqueue failure/success 後の解放は Path A/Recovery では実装済み。Path B は取得自体がないため N/A |
| generation | version wrap（D101-31-D 修正済み）は Proof identity 用。Path B の stale 判定には未使用 |
| linearization | tryAdmit/release = packedState_ 単語 CAS。Path B の実効 linearization は CoordinatorState load（非CAS） |

> **結論: Path B の publication enqueue は AdmissionPackedState mechanism に結び付いていない。**
> 「tryAdmit() が存在する」だけの監査で終わらせないという指示どおり、この非接続を
> 明示的に GAP として記録する。D101-31-B 契約（Publication/Recovery/Build の3経路）のうち、
> 「Publication 経路」の予約取得は Orchestrator 経由の publish のみが担い、
> commitRuntimePublication 直接呼び出し（5 production site）が漏れる。

---

## 5. A-4 X5 residency transaction integrity

| 項目 | 確認結果 | 判定 |
|---|---|---|
| +1 の唯一性 | ✅ 単一 choke point（enqueuePublicationIntent 内 h:372 のみ。全3 enqueue 経路集約） | PASS |
| rollback 完全性 | ✅ push 失敗時 h:375 fetchSub。それ以外の脱出経路なし | PASS |
| push 成功後 ownership | ✅ residency は queue residency として保持。Owner は ownerChannel_ 側 | PASS |
| pop 成功時 −1 | ✅ ProcessIntent.cpp:56（Publish case のみ。pendingIntentCount_ は触らない — INV-X5-1/X6-4 の type 分岐減算が一元管理） | PASS |
| 二重 decrement | ✅ 減算は processIntent while ループに一元化（HANDLER-1: handler で行わない） | PASS |
| failed push 後の漏れ | ✅ なし（caller 側 Owner reclaim + registry unregister も完備 AudioEngine.h:4586-4590） | PASS |
| shutdown close 後の reservation | ⚠️ gate 通過後（TOCTOU window 内）の reservation は shutdown 中も生存し、pop されるか CoordinatorLoop join 後は残留し得る（§5 Case C） | PARTIAL |
| reservation と queue residency の時間関係 | ✅ INV-X5-1 定義どおり（並行中 >=、quiescence 後 ==） | PASS |

---

## 6. A-6 No-Resurrection 4 Case 判定

| Case | シナリオ | 現行挙動 | 判定 |
|---|---|---|---|
| A | shutdown 前の通常 flow | gate 通過 → reservation → push → 正常受理・実行 | ✅ PASS |
| B | shutdown 確定後の request | state_==ShuttingDown → false。owner reclaim 完了 | ✅ PASS |
| C | **T1: gate の state_ load（≠ShuttingDown）/ T2: requestShutdown() 完了 / T3: residency+push 成功** | **閉鎖後に Publish Intent が queue に入る**。CoordinatorLoop が稼働中なら pop され **shutdown 開始後にもかかわらず executePublish（world commit）が実行され得る**（ProcessIntent に shutdown discard 分岐なし）。:191 の loop join 以降に残留した場合は residency>0 が残存（drain 条件恒偽 false → shutdown budget 消費） | ❌ **GAP** |
| D | queue full | residency rollback + owner take + registry unregister + CallerDestroy | ✅ PASS |

Case C の詳細 timeline:

```text
T0  Producer: commitRuntimePublication 開始（reservation なし — Path B は tryAdmit しない）
T1  enqueuePublicationIntent: state_ load → Running（gate 通過）
T2  Shutdown thread: bridge.requestShutdown() → state_=ShuttingDown（ReleaseResources.cpp:75）
     ＊ closeAdmission()(:198) はまだ — CoordinatorLoop join (:191) のさらに後
T3  Producer: residency fetchAdd(+1) → intentQueue_.push 成功
T4a （loop 稼働中なら）processIntent が pop → residency-1 → executePublish
     → shutdown 開始後の world commit が成立し得る
T4b （join 後に残留した場合）residency>0 のまま Consumer 不在
     → Layer2 isFullyDrained 恒偽 → shutdown budget 消費 / OwnerChannel 残留
```

厳密読みでは T3 は「close 後の enqueue」であり No-Resurrection 違反
（admission closed ⇒ 新規 publish obligation 不受諾）に該当。発生確率は window 分のみだが、
**構造的に排除されていない**点が本監査の核心 GAP。

---

## 7. A-7 Path A との比較（authority 二重化の判定）

| 観点 | Path A（Orchestrator 経由） | Path B（commitRuntimePublication 直接） |
|---|---|---|
| 事前遮断 | PublicationAdmission::evaluate + RejectedShutdown（:301-303） | caller 散在の isShutdownInProgress（不定） |
| ShutdownRuntime reservation | **あり**（tryAdmit→release, B-8） | **なし** |
| transport gate | enqueuePublicationIntent の state_ gate（共通） | 同左 |
| 最終 linearization | tryAdmit（CAS）＋ state_ load の二段 | **state_ load のみ（非CAS）** |

**判定: admission authority は二重化している。**
同じ Publication domain で (1) ShutdownRuntime packedState CAS（Path A/Recovery のみ）と
(2) CoordinatorState load（全経路の最終 gate）が併存し、相互の線形化関係が定義されていない。
旧設計の「4 経路を common shutdown admission に接続」方向に対し、現行は未達成
（これ自体は D101-33-B の設計対象 — 本監査では事実確認のみ）。

---

## 8. A-9 4 admission path 境界比較

| 経路 | admission 取得 | reservation 作成 | queue ownership | release | shutdown close 同期 |
|---|---|---|---|---|---|
| 1. Publish Intent enqueue（Path B core） | **なし**（直接producer）/ Orchestrator が代理（Path A時） | residency +1（choke point） | ownerChannel_ + intentQueue_ | pop 成功時 residency−1／push失敗 rollback | state_ gate（TOCTOU）のみ |
| 2. Recovery enqueue | submitRecoveryRequest 内 gate（Path C 型）→ **tryAdmit あり**（RebuildDispatch:319 / AudioEngine.h:4443） | pendingIntentCount_ +1（rollback 付き）＋ durable fallback | recoveryIntentQueue_ / recoveryAdmissionPending_ | pop 成功 fetchSub | gate + tryAdmit（二段） |
| 3. Build admission | RebuildDispatch.cpp:319 の tryAdmit（Recovery build 含む） | 同上 | Builder Work Queue | release 済み設計 | tryAdmit（CAS） |
| 4. Publish / runtime publication（Path A） | Orchestrator tryAdmit（B-8） | residency +1（executePublish 内部の enqueue 経由） | 同 1 | publish 完了後 release（:330-333） | tryAdmit（CAS）＋ state_ gate |

→ 共通 shutdown admission への接続度: Recovery/Build = 接続済み、Publish = **Path A のみ接続、
Path B（直接 producer + deferred resubmit）未接続**。

---

## 9. A-10/A-11 Gate 判定表

| Gate | 現行実装 | Authority | Linearization | Race-safe | 判定 |
|---|---|---|---|---|---|
| Path B shutdown admission | state_ gate あり（旧監査の「check なし」は解消） | CoordinatorState（AdmissionPackedState 非接続） | **load のみ・非CAS** | ❌ TOCTOU 残存 | **B PARTIAL** |
| publication reservation | 単一 choke point（X5 §6.5） | publicationIntentResidencyCount_ | fetchAdd acq_rel | ✅ | **A PASS** |
| queue push | MpscBoundedRing・single gen site | intentQueue_ | push 完成順 | ✅ | **A PASS** |
| push rollback | h:375 fetchSub + caller Owner take/unregister | — | — | ✅ | **A PASS** |
| consumer decrement | ProcessIntent 一元管理（type 分岐） | — | — | ✅ 二重 decrement なし | **A PASS** |
| shutdown close | closeAdmission は loop/rebuild join 後（:191→:198） | ShutdownRuntime CAS | ✅（ShutdownRuntime 側は安全） | ⚠️ Path B が Q0 非観測 | **B PARTIAL** |
| Path A/B convergence | Path A=tryAdmit+gate / Path B=gate のみ | 二重化 | 未定義 | ❌ | **C GAP** |
| No-Resurrection | Case A/B/D OK・**Case C race window** | — | — | ❌ | **C GAP** |

### 総合判定: **GAP（C）**

- 旧監査の「state_ check なし」は解消（B PARTIAL 相当へ改善）
- しかし (a) check-then-act のため shutdown close と競合した enqueue が閉鎖後に成立し得る
  （Case C）、(b) ProcessIntent に shutdown discard がなく競合 intent が shutdown 後に
  commit され得る、(c) Path B が AdmissionPackedState に未接続で Q0 の観測外 —
  の3点により **PASS 不成立**。

---

## 10. Path B 完全 transaction（時系列・現行 + 競合ケース含む）

```text
【正常系】
T0  producer thread: commitRuntimePublication(world, regCtx, oldHandle)
T1  handle 登録 / decision snapshot / registry.registerPublish(seqId)
T2  ownerChannel_.enqueue(key, world)            … world 所有権移譲
T3  enqueuePublicationIntent:
      T3.1 state_ load (=Running)                … gate 通過
      T3.2 residency fetchAdd(+1)                 … X5 reservation
      T3.3 intentQueue_.push → true
T4  fire-and-forget return {Success, Transferred}（rollbackHandle null 化）
T5  CoordinatorLoop processIntent: pop → residency−1 → executePublish → commit
T6  onPublishCommitted → notifyPublishReceipt

【競合系 — Case C（本監査の GAP）】
T1' producer: state_ load (=Running)              … gate 通過済み
T2' shutdown thread: requestShutdown() 完了       … state_=ShuttingDown（閉鎖）
T3' producer: residency +1 → push 成功            … ★閉鎖後の obligation 受諾が成立
T4' (i)  loop 稼働中 → pop→commit：shutdown 開始後の world swap が実行され得る
    (ii) loop join 後残留 → residency>0 残留、drain 恒偽、OwnerChannel 滞留
```

---

## 11. A-12 PASS 条件チェック

* [x] 最新 ConvoPeq.md 基準化（13:03 版）
* [x] Path B call graph 完全追跡（§2）
* [x] final admission authority 特定（enqueuePublicationIntent = 最終 gate / X5 residency = transport authority）
* [x] AdmissionPackedState 接続状態確認（**未接続を特定**）
* [x] shutdown close の linearization point 特定（state_ load・非CAS / closeAdmission は CAS）
* [x] reservation +1 / rollback / −1 全件確認（§5 全項 PASS）
* [x] queue-full rollback 確認（Case D PASS）
* [ ] post-shutdown enqueue 不可能性 — **不可能性を示せず（Case C）**
* [x] shutdown/enqueue race 確認（Case C として記録）
* [x] Path A/B authority convergence 確認（**二重化を確認 — 未収束**）
* [x] publicationBacklogCount_ 非依存（Path B 実装は同 counter を参照しない ✅）
* [x] X5 publicationIntentResidencyCount_ 整合（transaction integrity 全項 PASS）
* [x] No-Resurrection invariant 判定（**GAP**）
* [x] コード変更 0

# VERDICT: D101-33-A = **GAP（C）— PASS せず在此停止**

---

## 12. D101-33-B への引き継ぎ（設計入力の記録のみ・実装案は含まない）

Race の構造的原因と、設計時に解くべき制約の事実列挙:

1. 競合点は `state_ load → residency fetchAdd → push` の非原子3段。
   close 側は `requestShutdown()` の state_ release store（ReleaseResources.cpp:75 /
   CtorDtor.cpp:113）。
2. Path B に ShutdownRuntime reservation を接続する場合、release タイミングの契約
   （Path A は「obligation durable 後 release」— Orchestrator:330-333 の前例）が必要。
3. ProcessIntent に shutdown discard 分岐がない事実（競合 intent の行き先未定義）。
4. closeAdmission は CoordinatorLoop join 後に行われるため、tryAdmit を Path B に入れても
   producer thread join 前提が崩れない範囲での接続点選択が課題。
5. publicationBacklogCount_ は使用禁止（D101-32-F 確定）/ X5 residency が live authority。

以上を input として D101-33-B（Path B transaction design）で解決策を設計すること。
