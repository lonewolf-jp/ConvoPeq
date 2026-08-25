# D101-33-B — Path B Publication Admission Transaction Design（設計確定報告書）

- **実施日**: 2026-08-25
- **作業種別**: 設計確定（コード変更 **0** / setter・test seam 追加なし）
- **判定**: **PASS**（B-9 全条件充足 — §13）
- **基準**: ConvoPeq.md **2026-08-25 13:17 再生成版**（ローカル実ソースから再生成）
- **入力**: D101-33-A GAP 判定（TOCTOU / AdmissionPackedState 非接続 / ProcessIntent shutdown 意味論未定義）

---

## 1. 基準化

```
python output_sourcecode_markdown.py   → ConvoPeq.md 再生成 (2026-08-25 13:17)
git diff --check                       → whitespace エラーなし
```

指定15シンボルの fresh trace 実施（D101-33-A の実測と同一結果 — ソース未変更のため再現）。
主要な確定値:

| シンボル | production 実在位置 |
|---|---|
| `ShutdownRuntime::tryAdmit/release` | ISRShutdown.cpp:498/:521。呼び出し3経路 = Orchestrator.cpp:69(Path A) / RebuildDispatch.cpp:319(Recovery/Build) / AudioEngine.h:4443(Recovery wake) |
| `closeAdmission` / `joinProducers` | ReleaseResources.cpp:198（**shutdownCoordinatorLoop :191 / stopRebuildThread :192 の後**） |
| `requestShutdown` | ReleaseResources.cpp:75 / CtorDtor.cpp:113（**shutdown 入口で最も早い**） |
| `enqueuePublicationIntent` | Coordinator.h:353-378（state_ gate + X5 residency 単一 choke point） |
| `enqueueRuntimePublicationFireAndForget` / `commitRuntimePublication` | AudioEngine.h:4524 / :4602（Path B 唯一の producer コア） |
| `processIntent` / `IntentType::Publish` / `executePublish` | ProcessIntent.cpp:44-66 / :149（**shutdown 分岐なし** — D101-33-A 確定のまま） |

---

## 2. 現行 shutdown lifecycle（fresh trace 確定）

```text
T_s0  releaseResources 開始（EngineLifecycleState→Releasing, AudioStopped phase）
T_s1  runtimePublicationBridge_.requestShutdown()        … CoordinatorState::ShuttingDown 確定
      ＊ この時点で enqueuePublicationIntent の state_ gate が閉じる（Path B のみの防衛）
T_s2  （各種 drain / clear）
T_s3  shutdownCoordinatorLoop()                           … Consumer join
T_s4  stopRebuildThread()                                 … Rebuild producer join
T_s5  shutdownRuntime_.closeAdmission()                   … packedState_ Open→Closing ★D101-31-D 修正済み wrap
T_s6  while(!joinProducers()) waitForDrain(...)           … outstanding()==0 待ち
T_s7  以降: VerifyDrained → Proof → reclaim
```

### 「closeAdmission より前に Path B producer が保証すべきこと」

現行構造の前提は「Producers are joined (Q2)」= T_s3/T_s4 で producer thread が静化していること。
しかし以下が**保証されていない**:

- (a) Message/Timer 系直接 producer（PrepareToPlay / Timer / Transition）は T_s1〜T_s5 の間
  スケジュールされ得る（thread join 対象外）。T_s1〜T_s5 の間に enqueue すると TOCTOU または
  **loop join 後の push**（residency 残留・Consumer 不在）が発生し得る — D101-33-A Case C。
- (b) T_s1 で閉じるのは CoordinatorState（Path B gate 用）のみで、`packedState_` は T_s5 まで
  Open。つまり T_s1〜T_s5 は「admission は開いているのに transport gate は閉じている」
  不整合区間であり、2つの authority が異なる時刻に閉じる。

**設計要件 D-REQ**: closeAdmission（admission authority の閉鎖）は shutdown request 時点までに
前倒しし、producer はそれ以降「拒否される」か「reservation によって観測される」かの
いずれかに一意に帰着しなければならない。

---

## 3. Candidate A/B/C 比較

### Candidate A — Path B を AdmissionPackedState に接続（token 方式）

```text
tryAdmit(1)                                    ← packedState_ への CAS（close と同一単語）
    ↓
register / registry / ownerChannel enqueue     … 失敗時は即 release(1)
    ↓
enqueuePublicationIntent（X5 residency + push）
    ├─ 成功 → obligation durable → release(1)
    └─ 失敗 → rollback（owner take 等）→ release(1)
```

| 評価項目 | 結果 |
|---|---|
| closeAdmission との線形化 | ✅ **同一 atomic word の CAS**。modification order が全順序を与える |
| producer join との関係 | ✅ reservation が outstanding() に出るため joinProducers retry loop（T_s6・既存実装）がそのまま機能 |
| push failure 時 rollback | ✅ token release + Owner take + registry unregister |
| obligation 範囲 | tryAdmit 成功 → enqueuePublicationIntent true（durable）/ 全失敗、のいずれかまで |
| release timing | **durable 点（enqueue 成功直後）**— Path A 前例（Orchestrator:330-333「intent is enqueued, obligation now survives in durable state」）と完全同型。コード上で適用可能を確認済み |
| authority convergence | ✅ 4 path すべて packedState_ に収束 |
| outstanding()==0 の意味 | **全 publication/recovery/build transaction の非存在**（Path B 観測外問題の解消） |

**課題**: 現行 enqueuePublicationIntent 内の state_==ShuttingDown gate が token と衝突する
（T_s1〜T_s5 区間で正当 token が gate により拒否され得る → reservation 取得後 reject の
曖昧な partial state）。→ gate の再編が必要（§5）。

### Candidate B — enqueuePublicationIntent 自体を admission authority にする

案: state load + fetchAdd + push を線形化させる。検討した方式:

- (i) load→reserve→push のリトライループ: requestShutdown 側の store と producer 側の
  reserve が**異なるアドレス**のため、検証用 load を挟んでも ABA 的 window が消えない。
  完全線形化には close 側が residency word を触る必要があり、transport counter に
  admission 意味を混入させる（INV-X5-1/X6-4 の分離原則に反する）。
- (ii) state_ を CAS で書く方式: requestShutdown を CAS 化しても、producer の
  check-reserve-push 3命令のどれとも原子にならない。結局 TOCTOU。

**判定: 不合格。** 単一 word CAS（Candidate A）以外に、close vs admit+enqueue の
線形化を安価に保証する構造がない。requestShutdown 側との同期関係について、
Candidate B では「どちらが先に linearize するか」を定義できる点が存在しない
（load と store の real-time 順に依存するだけ）。

### Candidate C — Admission token と transport residency の明示分離

```text
AdmissionPackedState（shutdown obligation authority）
        ↓ publication admission token
publicationIntentResidencyCount_（transport residency authority — X5, 変更なし）
        ↓
intentQueue（transport）
```

token lifetime: acquire = transaction 開始 / release = obligation durable 点（A と同一）。
実質的に Candidate A と同じ機構だが、**責務の二層分離を明文化**する点が価値:

- token（packedState_）: shutdown に対する admission 可否だけを答える。transport 状態は見ない
- residency（X5）: transport の drain 判定だけを答える。shutdown 可否は見ない
  （D101-32-F 確定の維持）

**判定: 採用。A の機構 + C の層分離規律 = 採用案「A′」。**

---

## 4. 採用案 — A′「Admission-First Token + Early Close Convergence」

### 4.1 構成要素（すべて既存 primitive の再配置・追加呼び出しのみ）

| # | 要素 | 内容 |
|---|---|---|
| 1 | **Token-first acquisition** | `enqueueRuntimePublicationFireAndForget` 冒頭（registerDSPHandleForRuntime より前）で `tryAdmit(1)`。失敗 → 即 `RejectedShutdown` 相当 return（Side effect ゼロ） |
| 2 | **RAII TokenGuard** | Orchestrator の ReservationGuard（:70-76）と同型。全失敗経路で release(1)。成功時は durable 点で手動 release + guard 無効化 |
| 3 | **release 点 = obligation durable 点** | `enqueuePublicationIntent == true` 直後。Path A 前例（Orchestrator B-8）と同一契約 |
| 4 | **Early Close Convergence** | `closeAdmission()` を T_s1（requestShutdown 直後）へ移動。T_s5 の closeAdmission は削除し、T_s6 joinProducers retry loop は**現位置に維持**（outstanding>0 を待つ既存実装がそのまま使える） |
| 5 | **state_ gate の廃止** | enqueuePublicationIntent の `state_==ShuttingDown → false` を削除（admission authority は packedState_ token に一本化）。CoordinatorState::ShuttingDown は drain-mode signal として processIntent/HealthMonitor 等の既存用途を維持 |
| 6 | **X5 residency 変更なし** | reservation→push→rollback→pop-sub の現行トランザクションを完全維持 |
| 7 | **ProcessIntent 無条件 commit** | queue 内 Publish Intent は「閉鎖前に正当 admittされたもの」のみとなるため discard 不要（§8） |

### 4.2 変更後の Path B transaction

```text
T0  tryAdmit(1)                          ★唯一の linearization point（closeAdmission と同一CAS word）
T1  registerDSPHandleForRuntime          … 失敗 → release+return Failed
T2  registry.registerPublish(seqId)      … 失敗 → release+rollbackHandle+return
T3  ownerChannel.enqueue                 … 失敗 → unregister + release + CallerDestroy
T4  enqueuePublicationIntent:
      residency fetchAdd(+1)             … X5（無変更）
      intentQueue_.push                  … 成功 → durable
      失敗 → fetchSub rollback → false
T5  durable 判定:
      true  → release(1)（obligation durable）、{Success, Transferred}
      false → ownerChannel.take + unregister + release(1)、{Failed, CallerDestroy}
```

---

## 5. Admission transaction state machine

```
                    tryAdmit(1) [CAS on packedState_]
   ┌────────────────────────────────────────────────────────┐
   │ fail (Closing/Closed/Faulted) → REJECTED（副作用ゼロ）   │
   ▼                                                        ▼
HELD ──register fail──► RELEASED(Rollback)            [shutdown側]
  ││                                                    closeAdmission()
  │├─world/seq fail─► RELEASED(Rollback)                … CAS Open→Closing
  │├─ownerCh full───► RELEASED(Rollback+CallerDestroy)    （いつでも安全・冪等）
  │├─push fail──────► RELEASED(Rollback+take+unreg)
  │└─push ok────────► DURABLE → release(1) → RELEASED(Durable)
  │                        │
  └─ (HELD の間) outstanding()>0 → joinProducers は待つ（既存 retry loop）
```

状態遷移の完全性: HELD から抜ける経路は RELEASED(Rollback) / RELEASED(Durable) の2種のみ。
全経路で token・residency・registry・Owner の整合が保たれる（§7 matrix）。

---

## 6. Linearization proof（Case C の構造的排除）

**主張**: 採用案では「shutdown close と競合して queue に入る admission」は存在しない。

**証明**: `tryAdmit(1)` と `closeAdmission()` は同一 atomic 変数 `packedState_` への
単語 CAS である（ISRShutdown.cpp:498 / :429）。同一アドレスへの全 CAS は全順序な
modification order を持つ。よって任意の producer transaction P と close イベント C について
次の**いずれか一方**が排他的に成立する:

- **(i) P.tryAdmit < C**（modification order 上先行）:
  P は reservation を保持する。C 後も P は enqueue を完了できるが、
  `outstanding()>0` のため `joinProducers()` は完了しない（ISRShutdown.cpp:471-472 の
  count!=0 ガード）。P が durable 点で release すると初めて joinProducers が進む。
  → **P は Q0 によって観測される**（D101-33-A の「観測外」問題の解消）。
  また P の push は CoordinatorLoop join 前（T_s3 ≪ T_s5'）に完了する producer thread
  前提（Q2）+ outstanding 待ちの二重化により Consumer 不在残留が構造的に発生しない。
- **(ii) C < P.tryAdmit**:
  packedState_ の state 領域は Closing/Closed なので tryAdmit は false を返す
  （ISRShutdown.cpp:504 `state != Open → return false`）。P は副作用ゼロで reject。

check-then-act window は存在しない。「check（Open 判定）」と「reserve（fetchAdd 相当）」が
同一 CAS 内で完結していることが、D101-33-A の GAP（3命令 TOCTOU）との決定的な差である。

**Early Close Convergence の役割**: close を T_s1 へ前倒しすることで、「admission は開いているが
transport gate は閉じている」不整合区間（旧 T_s1〜T_s5）を消滅させる。CoordinatorState::ShuttingDown
は drain-mode signal に専念し、admission 可否の回答は packedState_ 単語に一元化される
（authority singularization）。

---

## 7. Ownership / rollback matrix（B-5）

失敗点ごとの完全状態（採用案適用後）。Token = AdmissionPackedState reservation、
Res = X5 residency、Reg = PendingPublishRegistry、Owner = ownerChannel_:

| 失敗点 | ownership 保持者 | destroy/reclaim 者 | token | Res | Reg entry |
|---|---|---|---|---|---|
| T0 tryAdmit 失敗 | caller（world 未渡し） | caller | ❌（取得不可） | ❌ | ❌ |
| T1 register fail | caller | caller（jassert + Failed） | ✅→release | ❌ | ❌ |
| T2 world null / seqId==0 | caller | caller（ScopeExit rollbackHandle） | ✅→release | ❌ | ❌ |
| T3 ownerChannel full | caller | caller（CallerDestroy） | ✅→release | ❌ | ❌ |
| T4 push fail（queue full） | caller（ownerChannel.take で回収） | caller（CallerDestroy） | ✅→release | ✅→rollback済 | ✅→unregister 済 |
| 成功 | **CoordinatorLoop（executePublish）** | 不要（commit） | ✅→release（durable） | ✅→pop で −1 | executePublish が解決 |

partial success の禁止: token あり・Res あり・Reg 残留 の組合せは T4 のみであり、
そこでは 3者すべてが同一ブロック（AudioEngine.h:4586-4590 相当 + release）で解消される。
「token だけ残る」「Res だけ残る」経路は存在しないよう token-first 順序で保証。

---

## 8. ProcessIntent shutdown semantics（B-4）

4 option の比較:

| Option | 内容 | 評価 |
|---|---|---|
| 1. 通常どおり commit | pop された Publish は無条件 executePublish | ✅ **採用**。採用案では queue 内 intent は全て「閉鎖前に token 取得・正当に受諾された obligation」であるため、commit は No-Resurrection 違反ではない（admission boundary は push 時に通過済み）。discard 用の ownership destroy 経路・telemetry も不要 |
| 2. shutdown discard | ShuttingDown 中の pop を破棄 | ❌ 正当 admitt 済み obligation の喪失。destroy 経路・drop telemetry 新設が必要となり契約が複雑化。benefit なし（Option 1 で安全） |
| 3. 条件付き commit | 条件は？ | ❌ 条件式が新たな authority を生む（曖昧さの温床） |
| 4. token 状態による判定 | pop時にadmission状態照会 | ❌ Coordinator は ShutdownRuntime を参照しない（循環防止）。token は push 時に検証済みで二重検査は意味を持たない |

### 「commit してよい Publish」と「違反となる Publish」の定義

```
commit してよい: enqueuePublicationIntent が true を返した Publish Intent
                （= packedState_ Open 下で token 取得・X5 residency 保持）
違反:           packedState_ Closing/Closed 下で新たに受理される Publish
                （採用案では構造的に不可能 — §6 の証明）
```

境界は **push 時点（token CAS）** で確定し、pop 時点では再判定しない。
`publicationIntentResidencyCount_` は transport residency authority のままであり
shutdown admission authority として使用しない（D101-32-F 確定の維持）。

---

## 9. Path A/B/C/D convergence（B-6）

| Path | admission acquisition | authority | linearization point | transport reservation | queue ownership | release point | shutdown close との関係 | stale/generation identity | shutdown discard |
|---|---|---|---|---|---|---|---|---|---|
| A: Orchestrator publication | trySubmitImpl 内 tryAdmit(1)（:69・変更なし） | packedState_ | 同 CAS | X5 residency（executePublish 内部 enqueue 経由） | ownerChannel_ + intentQueue_ | durable 後 release（:330-333・変更なし） | close 後は RejectedShutdown | PublicationSequenceId / evaluate の stale generation 判定 | なし（§8 採用） |
| B: commitRuntimePublication 直接 | **facade 冒頭 tryAdmit(1)（新設）** | packedState_ | 同 CAS | 同上 | 同上 | durable 点 release（新設） | 同上 | seqId/epoch/mappedGen | なし |
| C: Recovery | submitRecoveryRequest gate → tryAdmit（RebuildDispatch:319 / AudioEngine.h:4443・変更なし） | packedState_ | 各 CAS | pendingIntentCount_ +1（rollback 付き）＋ durable fallback | recoveryIntentQueue_ / recoveryAdmissionPending_ | durable 後 release（既存） | close 後は submitRecoveryRequest gate + tryAdmit が拒否 | epoch（RuntimeStore::current 由来） | なし |
| D: Build admission | RebuildDispatch.cpp:319 tryAdmit（変更なし） | packedState_ | 同 CAS | pendingIntentCount_（Recovery 経由） | Builder Work Queue | 既存 | 同上 | build generation | なし |

**収束結果: Publication domain の shutdown admission authority は `packedState_`
（AdmissionPackedState）一つに収束。** 4 path の linearization point はそれぞれ自 path の
tryAdmit CAS であり、close 側 closeAdmission CAS と同一単語で競合する。
CoordinatorState::ShuttingDown は admission authority ではなく drain-mode signal に降格・専念。
publicationBacklogCount_ は使用しない。pendingIntentCount_ を Publish admission に流用しない
（lane 分離 INV-ISR-02/X5 の維持）。

---

## 10. No-Resurrection 4-case 再評価（採用案適用後）

| Case | admitted? | reservation? | queue push? | consumer? | commit? | ownership? | 判定 |
|---|---|---|---|---|---|---|---|
| A: shutdown 前通常 | ✅ token 取得 | ✅ | ✅ | ✅ pop→commit | ✅ | Transferred | ✅ PASS |
| B: shutdown 確定後 | ❌ tryAdmit 失敗（Closing/Closed） | ❌ | ❌（queue 触らず） | — | ❌ | caller keep / CallerDestroy | ✅ PASS |
| C: admission vs close 競合 | **CAS 全順序で必ず一方**: (i) 先行 admit → reservation 観測下で完遂・joinProducers が待つ (ii) close 先行 → reject | (i)✅(ii)❌ | (i)✅(ii)❌ | (i) loop join 前提で pop | (i)✅ | 両 case とも整合 | ✅ **構造的排除**（§6 証明） |
| D: queue full | ✅ | ✅→rollback | ❌ | — | ❌ | take + CallerDestroy | ✅ PASS |

Case C の排除は「同一 word CAS の全順序性」によるものであり、タイミング・スケジューラに依存しない。

---

## 11. Invariants（採用案が維持/強化する不変条件）

| Invariant | 内容 | 状態 |
|---|---|---|
| INV-LIFE-9 | Closed→Open 不存在（no resurrection） | ✅ closeAdmission 早期化でも維持（Closing/Closed から戻る経路なし） |
| Q0 強化 | outstanding()==0 ⇔ 全4 path の in-flight transaction 非存在 | ✅ **強化**（旧: Path B 非観測） |
| INV-X5-1 | residency = Publish queue residency + producer reservation | ✅ 変更なし |
| INV-X6-4 / INV-X1-* | lane 分離・durable 二重計上禁止 | ✅ 変更なし |
| INV-ISR-02 | pendingIntentCount_ に Publish を計上しない | ✅ 変更なし |
| D101-31-D 契約 | packedState_ layout・version wrap・G-H race | ✅ 未触碰（tryAdmit/closeAdmission の利用増のみ） |
| X2 §6.2 | receipt timeout ≠ publish failure | ✅ 変更なし |
| AC-ISR-1 | token API は NonRT からのみ | ✅（facade は NonRT 専用のまま） |

---

## 12. Implementation boundary for D101-33-C（実装範囲の界定）

D101-33-C（実装タスク）で行うべきこと / 行ってはならないこと:

**行う**:
1. `enqueueRuntimePublicationFireAndForget` 冒頭への tryAdmit(1) + TokenGuard 追加、
   durable 点 release、全失敗経路 release
2. `closeAdmission()` の ReleaseResources.cpp:75 付近（requestShutdown 直後）への移動、
   旧 :198 の closeAdmission 呼び出し削除（joinProducers retry loop は維持）
3. enqueuePublicationIntent の state_ gate 削除（コメントで token authority への収束を記録）
4. テスト追加: Case B（close 後 reject・副作用ゼロ）/ Case C（並行 close vs admit の
   反復ストレス — D101-31-D の G-H race test パターンを流用可能）/ Case D（既存）
5. Debug/Release build + 全 CTest

**行わない**:
- ProcessIntent への discard 実装（§8 により不要）
- X5 residency / pendingIntentCount_ / publicationBacklogCount_ への触碰
- Orchestrator / Recovery / Build の既存 token 経路の改変
- state_ の意味変更（drain-mode signal として維持）

**既知の意味変化（実装時にテスト更新が必要）**:
- T_s1〜T_s5 区間での publish は「RejectedShutdown」になる（旧: state_ gate 通過後も
  T_s5 まで受理され得た）。意図的な tightning であり、該当するテスト・DIAG ログの期待値更新を
  D101-33-C で実施すること。

---

## 13. PASS 条件チェックリスト

* [x] 最新 ConvoPeq.md をローカルソースから再生成（13:17 版）
* [x] 現行 shutdown lifecycle を fresh trace（§2 — T_s0〜T_s7）
* [x] Candidate A/B/C を比較（§3 — B は線形化不能で不合格と判定）
* [x] admission authority を一つに決定（packedState_ / AdmissionPackedState）
* [x] linearization point をコード構造として説明可能（§6 — 同一 word CAS の全順序性）
* [x] producer join / closeAdmission の順序と整合（close 前倒し + joinProducers retry 維持）
* [x] Path B を AdmissionPackedState の観測対象にするか否かを確定（**する** — token first）
* [x] X5 residency との責務分離を維持（§3 Candidate C 層分離・§8）
* [x] processIntent() の shutdown semantics を確定（Option 1 無条件 commit・§8）
* [x] ownership rollback 全経路を確定（§7 matrix — partial success 曖昧さなし）
* [x] Path A/B/C/D authority convergence を確認（§9）
* [x] No-Resurrection Case C の構造的排除を説明（§6 証明）
* [x] publicationBacklogCount_ に依存しない
* [x] pendingIntentCount_ を Publish admission に流用しない
* [x] コード変更 0

# VERDICT: D101-33-B = **PASS**

次タスク: **D101-33-C**（本設計の実装 + race test + Debug/Release + 全 CTest）。
実装はユーザーの別指示で開始すること。
