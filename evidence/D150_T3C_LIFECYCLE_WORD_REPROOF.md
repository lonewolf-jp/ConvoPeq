# D150 — T3c Lifecycle Single-Atomic Ownership Domain Design Re-proof (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only disproof-first design re-proof. **Production source: 0 / Test source: 0 / Ring: 0 / W5 migration: 0 / P5/P6: 0 / Phase 2 implementation: 0.**
**基準:** `ConvoPeq.md Generated: 2026-08-31 15:32:58`（mtime 実測: src/**.{cpp,h} で ConvoPeq.md より新しいもの 0 件 → 基準有効）。D147/D148/D149 の結論は候補仮説として扱い、事実認定は現行ソースへ戻して再導出。
**ツール:** AiDex（シンボル/呼び出し網羅）、serena（resolve=h:422-435 0-based 位置検証）、semble（意味交差）、ccc grep（field アクセス棚卸し — rg と完全一致）、graphify（lifecycle/transition 概念確認）、WSL rg/sed（rtk）、ctx_execute（mtime 監査）、WebFetch/ctx_fetch_and_index（16B atomic 文献）。

---

## 0. Final Verdict（先出し）

```text
T3c = GO  — ただし唯一の load-bearing 条件付き:
   delivery を lifecycle word W に fold すること（W = {id, state, pending, adjudicated, delivery}）。
   delivery を W 外に plain のまま残す案は §8 で NO-GO（terminal-write 窓が affinity 非依存に残存）。

D149 の STOP #5（resolve reset が old adjudication に上書き）→ W 内 CAS で構造排除（§4/§5/CE-3）。
D149 の STOP #6（apply destination identity が affinity 依存）→ id-tagged CAS で affinity 非依存に（§7/CE-4）。
naive T3 / T3b の NO-GO は不変。Phase 2 実装は D150 GO により「T3c 実装設計ゲート」まで解禁、
ただし本監査は read-only — 実装は次ゲート（inventory → patch spec → read-only impl audit）。
```

---

## 1. Baseline / Source Validity

- ConvoPeq.md mtime `2026-08-31T06:33:05Z`（=15:33 JST）、内部 `Generated: 2026-08-31 15:32:58`。src 配下でより新しいソース 0 件。
- 変更 0 を git 側でも担保（本監査は Read/rg/ccc/graphify のみ、Edit/Write は evidence/doc のみ）。

---

## 2. Current Lifecycle Model（現行ソース再導出・実測 line numbers）

### 2.1 W 候補 field と atomic/plain 実態（ISRRuntimePublicationCoordinator.h）
| field | 宣言 | 型 |
|---|---|---|
| `id` | h:348 | `std::atomic<LogicalRecoveryObligationId>`（=u64, h:264） |
| `state` | h:350 | `std::atomic<ObligationState>`（8 値 h:306-316） |
| `delivery` | h:356 | **plain** `ObligationDeliveryState`（3 値 h:325-329） |
| `consecutiveFailureCount` | h:361 | `std::atomic<std::uint8_t>`（=adjudicated 相当） |
| `liveCount_` | h:450 | `std::atomic<u64>`（table 全体・単一 −1 authority は state CAS） |

→ 指示前提（id/state atomic・delivery plain・counter 別 atomic）を**現行コードで実証**。`pending`（未 adjudicate signal）は**現行に存在しない**（T3c 新設）。

### 2.2 全 writer / reader（ccc grep + rg 実測・両者一致）
- **id**: writer=tryInsert h:399（CL のみ・唯一 call site cpp:947）。reader=resolve h:425 / markTransientFailure cpp:1064 / redrive cpp:1114,1128 / submit cpp:938,955 / shutdown cpp:1350。
- **state**: writer=tryInsert h:401（CL）/ resolve CAS h:428（**任意スレッド**）/ coalesce CAS cpp:932（CL）。reader=findByKey h:384 / tryInsert h:397 / markTransientFailure cpp:1066 / redrive cpp:1110,1136。
- **delivery**: writer=tryInsert h:402（CL）/ submit cpp:979,996,1001（CL）/ redrive cpp:1159,1167（CL）/ **markTransientFailure cpp:1071（RebuildThread = W5）**。reader=cpp:901,944（CL）/ cpp:1112,1138（CL）。resolve は delivery を**読まない・書かない**（h:423-436 実測）。
- **consecutiveFailureCount**: writer=tryInsert h:407（CL）/ resolve h:429（任意）/ markTransientFailure cpp:1074 fetch_add（RebuildThread）。reader=telemetry h:444,543。
- **liveCount_**: +1 tryInsert h:408（CL）/ −1 resolve h:430（任意・state CAS 成功時のみ）。

### 2.3 スレッドトポロジー（T3c 当事者・実測）
| 行為 | T3c スレッド | 根拠 |
|---|---|---|
| postSignal（6 サイト） | RebuildThread | RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401（rebuildThreadLoop / RebuildThread 同期 submitPublishRequest — Commit:822←enqueuePublicationIntentForRuntimeCommit RebuildDispatch:1051/1131/1337） |
| adjudicate（drain+apply） | **CL** | runCoordinatorPhase=Threading.cpp:258、CL=ISRCoordinatorLoop.cpp:8/39 |
| tryInsert | **CL** のみ | 唯一 call site cpp:947 ← submitRecoveryRequest ← submitRecoveryIntent ← QuarantineIntentHandler ProcessIntent.cpp:140（CL） |
| resolve（terminal 化） | CL（Route B :354）+ **RebuildThread（Route A :320 / Route C :371）** | Orchestrator 実測 |
| shutdown close | join 後単一スレッド | shutdownCoordinatorLoop(Threading:249)→stopRebuildThread→discardRecoveryRequestsOnShutdown(RebuildDispatch:810) |

**実 audio callback（AudioEngineProcessor.cpp / DSPCoreIO / DSPCoreFloat/Double / BlockDouble）は obligation を 0 件**（rg 実測 EXIT 一致・hit なし）。h:468「Callable from ISR」は契約上の許可であり、実経路の terminalizer は CL（Route B）+ RebuildThread（Route A/C）。→ **W の触达者 = {CL, RebuildThread}、実 ISR 非接触**（§12 の lock-pool 可否に直結）。

### 2.4 現行の構造欠陥（T3c が解く対象）
`markTransientFailure`（cpp:1059-1085）は契約文言（h:475 "Atomically performs"）に反し実装は **check-then-act**（cpp:1066 Live 判定 → cpp:1071 delivery plain 書込 → cpp:1074 counter fetch_add → cpp:1080 resolve）。判定と 3 書込が**別 atomic domain**。加えて delivery 書込 cpp:1071 のみ RebuildThread で、CL delivery 書込（cpp:979 等）と**同一 slot 上で data race（UB・W5）**。

---

## 3. Proposed T3c Lifecycle Word

```text
W = { obligationId:64, state:3, pending:3, adjudicated:3, delivery:2 }   // 意味幅 75bit → 16B に収まる
    （pending/adjudicated は実装簡便のため各 8bit に丸めても 64+8+8+8+2=90bit < 128bit）
```
- **単一 atomic 語**: 全 lifecycle 所有権判定（identity / liveness / 未 adjudicate signal / adjudicate 済み budget / transport 住処）が**同一 CAS ドメイン**の遷移として表現される。
- linearization point = 各 `cmpxchg16b`（W CAS）成功。
- `liveCount_` は W 外だが、その −1 は「W の Live→terminal CAS 勝者のみ」に gate される（単一 CAS authority — §5.5）。
- **禁止条件（明文化）**: `CAS 成功 → state 再判定 → 別 atomic counter/delivery 更新` の旧構造（§2.4）を一切残さない。counter/delivery の更新は必ず W CAS **内**で完結させる。

---

## 4. postSignal Proof（§3.1）

```text
producer(RebuildThread, O):
  loop { w = W.load(acquire);
         if (w.id != O)        return no-op;   // stale / unknown / reuse 後（id タグ）
         if (w.state != Live)  return no-op;   // terminal 化済み（liveness を同一 CAS 条件に内包）
         if (w.pending >= K)   return no-op;   // 飽和（inert）
         if (CAS(w, {O, Live, w.pending+1, w.adjudicated, w.delivery}, acq_rel)) return posted; }
```
確認事項への回答:
- **obligationId==O を CAS 条件に含められるか** → YES（id は W の先頭 64bit、expected に含む）。
- **state==Live を同一 CAS 条件に含められるか** → YES（T3b との決定的差: T3b producer は id のみ、T3c は id∧state∧pending を単一 expected に束縛）。
- **pending<K を同一 CAS 条件に含められるか** → YES。
- **stale signal が reuse 後の N に入らない** → 入れない。reuse 後 W.id==N、expected.id==O で CAS 失敗 → no-op。**affinity 非依存**（§7 CE-1）。
- **overflow/saturation 意味論** → pending>=K で no-op。K 件の観測は記録済みなので次 adjudicate で adjudicated>=K → Failed。現行「4 回目 terminal・5 回目 Live チェック no-op」と**terminal 帰結同一**（§16 CE-11 で等価性確認）。
- **postSignal 失敗時の durable fallback が invariant を破らないか** → postSignal の no-op は「観測の破棄」であり obligation の所有権（id/state/delivery/liveCount）を**一切変更しない**。durable/transport 表現は submit/redrive 側（CL）が保持し、postSignal は無関係。invariant 不変。
- **重要（指示）**: 「tagged だから安全」ではなく、**Live state と destination identity が同一 CAS 条件に内包**されていることを確認済み。

---

## 5. adjudicate Proof（§3.2・STOP #5 解消の中心）

```text
consumer(CL, slot i):
  loop { w = W.load(acquire);
         if (w.pending == 0) break;                       // 何もない
         const uint8 n = w.pending;
         const uint8 a2 = w.adjudicated + n;
         if (a2 >= K) {
             if (CAS(w, {w.id, Failed, 0, 0, None}, acq_rel)) { liveCount_--; exhausted++; }  // 勝者のみ
             break;
         } else {
             if (CAS(w, {w.id, Live, 0, a2, None}, acq_rel)) break;   // pending=0 ∧ adjudicated+=n ∧ delivery=None 同一遷移
         } }
```
検証項目への回答:
1. **drain と counter application が別操作か** → **別操作ではない**。`pending=0` ∧ `adjudicated+=n` ∧ `delivery=None` が**単一 CAS**（§3 禁止条件を充足）。
2. **state 判定と counter 更新の間に resolve が割り込めるか** → **割り込めない**。判定と更新が同一 CAS。resolve が先に勝てば本 CAS の expected.state==Live が失敗 → 再試行で Failed 観測 → drop。
3. **resolve 成功時、古い adjudication CAS が成功し得ないこと** → resolve が W を {O,T,0,0,d} に変えると、adjudicate の expected（{O,Live,c,a,d}）と不一致 → 失敗。**成立しない**。
4. **pending=0 化と adjudicated+=n が同一遷移** → YES（上記 CAS 1 発）。
5. **exhaustion の liveCount-- が terminal CAS 勝者だけ** → `liveCount_--` は exhaustion CAS **成功ブロック内**のみ。敗者（resolve が先に terminal 化）は CAS 失敗 → 減算しない。**単一 authority**。
6. **delivery=None を W 外に置くことが安全か** → **安全でない**（§8）。よって T3c は delivery を W に fold する（上記 CAS の `None` は W 内遷移）。

**禁止構造の消滅確認**: 旧 `CAS → state check → 別 atomic fetch_add` は存在しない。counter も delivery も state も expected/new 語の一部。

---

## 6. resolve Proof（§4・STOP #5 完全潰し）

```text
resolve(id, T):  // 任意スレッド（CL Route B / RebuildThread Route A,C / shutdown）
  for i: if (W[i].load(acquire).id != id) continue;
         loop { w = W[i].load(acquire);
                if (w.id != id || w.state != Live) return false;   // 既 terminal / 再スキャンで id 変化
                if (CAS(w, {id, T, 0, 0, w.delivery}, acq_rel)) { liveCount_--; return true; } }
```
- **必要条件**: resolve の terminal 遷移と adjudicate の counter 遷移が**同じ W を競合** → YES（同一 16B 語）。
- **`adjudicate CAS succeeds` XOR `resolve CAS succeeds`（同一語版に対して）**: 両者とも expected.state==Live を要求。cmpxchg16b は語単位で直列化 → 同一バージョンで同時に成功不能。adjudicate が Live→Live に勝つと resolve は新語で再 CAS（Live→T）して勝つ — これは「両方 commit」ではなく**順序付き合成**（adjudication 記録後に terminal 化、counter は破棄）。liveCount-- は resolve 1 回のみ。
- **反証要求 `resolve CAS → counter.store(0) → old adjudicate fetch_add`**: **不可能**。counter は W 内。resolve CAS 後 W={O,T,0,0,d}。old adjudicate の expected.state==Live が失敗 → fetch_add 相当の遷移は発火しない。**T3c はこの経路を構造的に持たない**。

---

## 7. tryInsert / Reuse Proof（§5・affinity 非依存）

```text
tryInsert(key):  // CL
  for i: if (W[i].load(acquire).state != Live) {
             // 観測した terminal 語から CAS（plain store ではなく CAS で affinity 崩しに耐える）
             if (CAS(W[i], {observedId, observedT, 0, 0, observedD}, {N, Live, 0, 0, None}, release)) {
                 identity=key; ...; liveCount_++; return i; } }
```
- **old O adjudication が N に作用しないことを affinity 非依存で証明**: old adjudicate CAS expected.id==O、reuse 後 W.id==N → **id 不一致で失敗**。`O→N` 混入は**構造的に不可能**（§10 ABA）。
- **これが T3b と T3c の重要差**: T3b は consumer apply が `state==Live` のみ再判定（reuse で N が Live だと**通過してしまう ABA**）。T3c は expected に **id を含む**ため、N-Live でも O-expected と不一致 → 失敗。
- tryInsert を plain store でなく **terminal 語からの CAS** にすると、仮に tryInsert が複数スレッド化しても（affinity 崩し）二重 insert しない（勝者のみ liveCount++）。単一 CL writer なら plain でも無害だが、**affinity 非依存の証明**には CAS が十分条件を与える。

---

## 8. Delivery Ownership Proof（§8・別軸再判定）

**writer = CoordinatorLoop only?** T3c で markTransientFailure の RebuildThread delivery 書込（cpp:1071）を排除し adjudication（CL）へ移す。残る delivery writer = tryInsert/submit/redrive/adjudicate = **全て CL**。reader = cpp:901/944/1112/1138 = **全て CL**。よって **data race は消滅**（h:318-324 の主張が真化）。

**terminal obligation への stale delivery write の分類**（指示の 3 分類）:
- delivery を **W 外 plain** に残す案: CL adjudicate が CAS 成功（Live 時点）→ **RebuildThread resolve が terminal 化** → CL が delivery=None を plain 書込、という順序が**成立する**（terminal 化は別スレッドなので affinity で防げない）。これは *data race*（否・単一 CL writer）ではなく **semantic invariant violation**（「delivery は Live obligation の住処を表す」所有権不変条件の破棄）。*observable state corruption* は否（全 reader Live gate・tryInsert が上書き）。→ 指示の「inert だから GO」禁止により、**この案は §8 で NO-GO**。
- delivery を **W に fold** する案（採用）: delivery=None は adjudicate CAS の**内側**。resolve が先に terminal 化すると adjudicate CAS が失敗し **delivery 書込自体が発火しない**。terminal への delivery write が**構造的に生成不能**。→ **GO**。

**結論**: T3c の GO は delivery-in-W に依存。これが本再証明の load-bearing 条件。

---

## 9. Race Matrix（§6・A-J）

| Case | 順序 | 合格条件 | 判定（W CAS 解析） |
|---|---|---|---|
| A | postSignal → adjudicate | O にだけ適用 | ✓ expected.id==O∧Live、adjudicated+=n |
| B | adjudicate → resolve | どちらか一方だけ commit | ✓ 同一語版で expected.state==Live 競合、直列化 |
| C | resolve → adjudicate | old adjudicate 失敗 | ✓ resolve 後 state=T、adjudicate CAS 失敗 |
| D | resolve → tryInsert(N) → old adjudicate | N 汚染なし | ✓ expected.id==O vs W.id==N → 失敗（**affinity 非依存**） |
| E | old signal → tryInsert(N) | N に signal 混入なし | ✓ postSignal expected.id==O vs N → 失敗 |
| F | tryInsert(N) → old resolve(O) | N 破壊されない | ✓ resolve id 再スキャンで id!=O → skip（§6） |
| G | exhaustion adjudicate ∥ resolve | liveCount −1 一度だけ | ✓ 勝者のみ fetchSub（§5.5/§6） |
| H | duplicate signal O × N | O/N identity 混線なし | ✓ 各 CAS が id タグで宛先固定 |
| I | pending drain ∥ reuse | stale pending が N へ移らない | ✓ resolve が pending=0 で terminal 化、reuse は {N,Live,0,0}、old drain CAS id 不一致 |
| J | shutdown resolve ∥ adjudicate | terminal/shutdown invariant 維持 | ✓ shutdown は join 後単一スレッド（§2.3）で並行なし。仮に並行でも W CAS 直列化 |

**D/E/F は affinity を意図的に崩して検証**（tryInsert を別スレッドと仮定）: いずれも **id-tagged CAS が失敗させる**ため混入なし。→ D149 の affinity 依存を解消。

---

## 10. ABA / Identity Proof

- `nextId_` は単調増加 u64（h:398 `++nextId_`、h:451 初期値 1、単一 writer CL）。0 は「obligation なし」予約（resolveRecoveryObligation cpp:1018 で early-return）。
- 再利用 slot は**厳密に大きい id** を受ける → 旧 id==O の CAS は永久に不一致 → **ABA 不成立**。
- id 幅: u64 のまま（RecoveryIntent.obligationId / PublishPayload.recoveryObligationId が u64 — 型変更の波及を避ける）。2^64 mint は実時間到達不能。
- 代替案: id を u32 に縮小すれば W を 8B に収め lock-free  natives 化可能だが、u64 契約の広範な変更を要す → **不採用**（16B で足りる）。

---

## 11. Memory Ordering Proof

- W CAS: `acq_rel`（全 4 遷移）。load: `acquire`。
- liveCount_: `fetchSub(release)` / `liveCount()` `load(acquire)`（h:430/439）。W CAS の acq_rel が terminal 可視化と減算を束ねる（減算は CAS 成功後に program-order で sequenced）。
- delivery/identity/buildSource: W fold 後は delivery は W 内。identity/buildSource は**CL 単一書込・CL 単一读取**（§2.2）で release store（tryInsert h:401 の state=Live release）に sequenced-before → CL 内 program order で十分。
- 最小十分: 跨スレッド共有は W と liveCount_ のみ。強い順序は要求しない。

---

## 12. 16-byte Atomic Feasibility（§7・最後に検証）

**まず意味幅（semantic correctness）**: id 64 + state 3（8 値）+ pending 3（0..4）+ adjudicated 3（0..4）+ delivery 2（3 値）= **75 bit < 128 bit**。丸め（各 8bit）でも 90bit。→ 16B で表現可能。

**実装可能性**:
- MSVC x64 の `std::atomic<16B struct>`: cppreference 実測の通り `is_always_lock_free` は実装定義で「never or sometimes lock-free なら false」。**MSVC STL は 16B を lock pool（critical section）で実装**（`is_always_lock_free==false`）。
- `_InterlockedCompareExchange16b`（→ `cmpxchg16b`）: x64 で利用可（AMD64 baseline・Windows x64 は CX16 必須）、**16 バイト整列必須**、**hardware lock-free**。
- **lock-pool 可否の判定（短絡しない）**: 実 call graph で W 触达者 = {CL, RebuildThread}、**実 audio callback 非接触**（§2.3 実測 0 hit）。よって lock-pool でも RT 制約は違反しない。ただし lock-pool は待機可能性・将来の ISR 接触追加で危険 → **`_InterlockedCompareExchange16b` 直実装を推奨**（lock-free・アロケーション不要）。
- 実装注記: slot 配列を `alignas(16)`（LogicalRecoveryObligation は既に DSPHandle alignas16 を内包 h:972 pragma、C4324 抑制済）。`std::atomic` 経由でなく intrinsic + 明示 16B 整列構造体でラップ。

---

## 13. Shutdown / Failure Path

- shutdown close（cpp:1349-1353）は全 Live slot を `resolveRecoveryObligation(ShutdownDiscarded)` → W CAS(Live→ShutdownDiscarded)。呼び出しは **shutdownCoordinatorLoop(join)→stopRebuildThread(join)→discard(RebuildDispatch:810)** の join 後単一スレッド（§2.3）→ adjudicate/postSignal と並行せず、W CAS は決定的。
- 「shutdown が lifecycle CAS domain 外」か → **否**。shutdown も同一 W CAS を通る（§10 NO-GO 条件「shutdown outside CAS domain」を回避）。
- Route A/C（RebuildThread）の terminal 化と CL adjudicate の競合は §5/§6 で直列化証明済み。

---

## 14. Telemetry Classification（§9・意味論と分離）

observability は GO 判定に混ぜない。分類:
- **exhausted 過計上**: 現行 cpp:1079 は resolve 成否前に増加。T3c は exhaustion CAS **成功時のみ** `exhausted++`（§5）→ 過計上解消。
- **stale signal drop**: postSignal no-op を reason 別カウンタ（id-mismatch / terminal / saturated）。
- **reuse overwrite drop**: resolve が pending=0 で terminal 化するため「無音の消失」は発生しない（T3b の tryInsert {N,0} による暗黙破棄は、T3c では terminal 化時点で pending が 0 にされるため観測分類が明確）。
- **terminal-after-adjudication**: adjudicate(Live→Live) 後 resolve(Live→T) — 正常合成、drop ではない。
- **CAS failure**: retry であって事象ではない（カウンタ不要）。
- **shutdown discard**: 既存 recoveryObligationShutdownDiscardCount_（cpp:1040）。
→ 意味論的 GO は §15 で独立に判定。observability は実装時チェックリストへ回付。

---

## 15. T3c GO/NO-GO Criteria（§10）

| 基準 | 判定 |
|---|---|
| ✓ signal identity protected | ✓ §4（id in CAS） |
| ✓ destination identity protected | ✓ §5/§7（adjudicate/tryInsert expected.id） |
| ✓ Live/terminal state protected | ✓ §5.2/§6（state in expected） |
| ✓ counter transition protected | ✓ §5.1/5.4（adjudicated+=n in CAS） |
| ✓ resolve reset protected | ✓ §6（reset=W 内、old adjudicate 失敗） |
| ✓ reuse protected without affinity | ✓ §7/§9-D,E（id-tag、affinity 非依存） |
| ✓ old O cannot modify N | ✓ §10（単調 id・ABA 無） |
| ✓ resolve and adjudicate cannot both commit | ✓ §6（同一語版で直列化） |
| ✓ liveCount decrement single CAS authority | ✓ §5.5/§6（勝者のみ fetchSub） |
| ✓ delivery writer ownership proven | ✓ §8（**delivery-in-W 条件で**全遷移 W 内・terminal write 不能） |
| ✓ shutdown transition proven | ✓ §13（同一 W CAS・join 後決定的） |

**NO-GO 条件のいずれも残存せず**（delivery を W 外に置く案のみ §8 で NO-GO → 本 T3c は delivery-in-W を採用して回避）。

---

## 16. Counterexample Search（§16・能動的破壊試み）

- CE-1 postSignal(RebuildThread) ∥ tryInsert(CL) reuse: expected.id==O vs N → 失敗。✓
- CE-2 adjudicate(CL) ∥ resolve(RebuildThread): 同一 W、Live 要求 → 直列化。✓
- CE-3 **D149 #5 再現**: adjudicate CAS(Live→Live) 成功 → resolve CAS(Live→Failed) 成功 → 旧 fetch_add… **T3c に fetch_add は無い**（counter 加算は adjudicate CAS 内で完結、resolve 後は postSignal も state!=Live で no-op）。terminal O の counter は 0 のまま。**#5 消滅**。✓
- CE-4 **D149 #6 再現（affinity 崩し）**: tryInsert を別スレッドと仮定しても adjudicate CAS expected.id==O が N で失敗。**#6 消滅（affinity 非依存）**。✓
- CE-5 exhaustion ∥ Published terminal: 勝者のみ liveCount--。✓
- CE-6 delivery-in-W で submit(CL) が delivery 変更 ∥ resolve(RebuildThread): resolve CAS が delivery 変化で失敗→再試行で新 delivery 保持し terminal。lost update 無。✓
- CE-7 単調 id で O→N 混入: 不可。✓
- CE-8 identity plain の publication 順序: tryInsert が identity を W.state=Live store **前**に書込（release）→ findByKey は Live 観測後に identity 読取。✓
- CE-9 buildSource/handle/epoch/recoveryGeneration: W 外 plain だが **CL 単一書込・CL 単一读取**（redrive cpp:1143-1157）。所有権判定でなく payload → W 外可。✓
- CE-10 跨スレッド state 消費者は resolve/postSignal のみ（両者 CAS 安全）。✓
- CE-11 飽和と terminal 帰結の等価性: pending cap K、adjudicated 累計 >=K で Failed。現行「4 目 terminal・5 目 no-op」と帰結同一。✓
- **残存反証なし**。

---

## 17. Final Verdict

```text
T3c = GO（delivery を lifecycle word W に fold する設計に限定）。
D149 の STOP #5/#6 は id-tagged 単一 W CAS により構造的に消滅（affinity 非依存）。
delivery-in-W が成立条件。delivery を W 外 plain に残す案は §8 で NO-GO。
```

---

## 18. Implementation Freeze Decision

- Phase 2 実装は「T3c 実装設計ゲート」まで**凍結継続**（本監査は read-only・実装 0）。
- 解禁は次ゲートで: 実装前 inventory → patch specification → read-only implementation audit → build/CTest。
- 実装時に h:318-324（delivery 単一書込者）が真化、cpp:1071 の RebuildThread delivery 書込が消滅（W5 解消）。

---

## 19. Required Follow-up（実装設計ゲートで確定すべき事項）

1. **W の exact bit layout + alignas(16)**（pending/adjudicated を 3bit か 8bit か）。
2. **`_InterlockedCompareExchange16b` ラッパ**（lock-free）か `std::atomic<16B>`（lock-pool・実 ISR 非接触ゆえ可）かの選択と根拠固定。
3. **tryInsert を terminal 語からの CAS 化**（affinity 非依存の十分条件）。
4. **adjudicate の W 走査コスト**（32 slot × CAS ループ）と 1ms tick 予算。
5. **R18/R20 テスト適配**（markTransientFailure 直後同期アサート → 「postSignal + 即 adjudicate」互換セマンティクス）。
6. **telemetry reason 分類**（§14）の実装。
7. Route A/C（RebuildThread）と CL の W CAS 競合を扱う**回帰テスト**（決定論的 hook は production 不要 — 2 スレッドハーネス）。
8. ObligationState の ResolvedRetry/ResolvedSuperseded が W の 3bit 表現で保持されることの確認（現行 production で ResolvedRetry は cpp:1023 で early-return、ResolvedSuperseded dormant）。
