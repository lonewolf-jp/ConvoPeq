# D149 — T3b Adjudication/Resolve/Reuse Race Re-proof (read-only, disproof-first)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only race re-proof. **Production source: 0 / Test source: 0 / Ring: 0 / W5 migration: 0 / P5/P6: 0 / Phase 2 implementation: 0.**
**基準:** `ConvoPeq.md Generated: 2026-08-31 15:32:58`（mtime 実測: ConvoPeq.md より新しい src/**.{cpp,h} 0 件 → 基準有効）。D147/D148 の証明は前提とせず現行コードから再導出。
**ツール:** AiDex（シンボル/呼び出し網羅）、serena（シンボル位置検証 resolve=h:422-435 0-based）、semble（意味交差検証）、WSL rg/sed（rtk）、ctx_execute（mtime 監査）。

---

## 0. 最終判定（先出し）

```text
T3b（tagged (oblId,count) signal word のみ）= NO-GO
  → STOP 条件 #5「resolve の counter reset が old adjudication に上書きされる」が
     提案配置のまま到達可能（§4 Case B′、反証構成 §5）。
  → STOP 条件 #6「apply destination identity」は thread affinity でのみ証明可能で、
     tagged CAS は consumer 側を保護していない（D148 の主張は producer 側にのみ有効）。
  → #1/#2/#3（旧 O → N 混入）は提案配置では到達不能だが、その理由は CAS ではなく
     「adjudication と tryInsert が同一 CoordinatorLoop」という未記載の不変条件（§4 Case C）。

結論分岐 = B。Phase 2 実装凍結継続。T3c（signal identity + adjudication identity +
counter/state transition を単一 atomic ownership domain へ統合）の設計再証明（D150）へ。
```

naive T3 の NO-GO（D148 §2）は訂正なし（§7 でトポロジー訂正後も成立することを確認）。

---

## 1. 現行ソース再導出（実測 line numbers）

### 1.1 `LogicalRecoveryObligation`（ISRRuntimePublicationCoordinator.h:347-362）

| field | 宣言 | 実態 |
|---|---|---|
| `id` | h:348 `std::atomic<LogicalRecoveryObligationId>` | atomic |
| `identity` | h:349 plain | plain（tryInsert のみ書込・CL） |
| `state` | h:350 `std::atomic<ObligationState>` | atomic |
| `delivery` | h:356 plain | **plain**（h:318-324「CoordinatorLoop-only writer」の主張は**現行では偽** — 下記 §1.4） |
| `consecutiveFailureCount` | h:361 `std::atomic<std::uint8_t>` | **別 atomic**（state と非束縛） |

指示の前提（id/state atomic・delivery plain・counter 別 atomic）は**現行コードで実証**。

### 1.2 `RecoveryAdmissionTable::tryInsert`（h:392-413）
- 本番呼び出し箇所は **cpp:947 のみ**（`submitRecoveryRequest` 内・CL）。AiDex 実測: 本番 code hit 1 件。
- 順序: `id.store(N,relaxed)` h:399 → `identity` h:400 → `state.store(Live,release)` h:401 → `delivery=None` h:402 → `recoveryGeneration` h:404 → `counter.store(0,release)` h:407 → `liveCount +1` h:408。
- 再利用ゲート: `state.load() != Live`（h:397）。**terminal 化なしに reuse は構造的に不可能**。

### 1.3 `RecoveryAdmissionTable::resolve`（h:423-436）
- id 再スキャン（h:425, acquire）→ `state CAS(Live→terminal, acq_rel)` h:428 → 成功時のみ `counter.store(0,release)` h:429 + `liveCount −1` h:430。
- **`delivery` と `id` は触らない**（terminal 後も id==O のまま）。counter reset は CAS と**別語**（h:420-422 コメント自体が「store *after* the CAS」と明言 = 非原子）。
- 単一 −1 authority（state CAS）は現行で成立。

### 1.4 `markTransientFailure`（cpp:1059-1085・本番呼び出し 6 サイト全 RebuildThread）
- cpp:1063-1065 id スキャン → cpp:1066 `state==Live` 判定 → **cpp:1071 `delivery=None` plain 書込（RebuildThread）** → cpp:1073-1075 `counter.fetch_add(1)` → 枯渇時 cpp:1079 telemetry++ → cpp:1080 `resolve(id, Failed)`。
- delivery 書込者一覧（実測）: CL 側 = h:402（tryInsert）/ cpp:979（Transport）/ cpp:996（None deferred）/ cpp:1001（Durable）/ cpp:1159/1167（redrive）に対し、**cpp:1071 のみ RebuildThread** → h:318-324「CoordinatorLoop-only・non-atomic by design」は現行で**偽**（既知 W5）。同一 slot への CL plain 書込（例: coalesce 経路 cpp:979）と理論上 **data race（UB）**。T3b の delivery CL 一元化はこの data race を閉じる（改善点 — §6.1）。

### 1.5 `resolveRecoveryObligation`（cpp:1016-1041）
- h:468「Callable from ISR (onPublishCommitted) and RebuildThread」= **契約上の許可**。
- 本番呼び出し実測: Orchestrator.cpp:320（Route A・RebuildThread）、:354（Route B・**CoordinatorLoop**）、:371（Route C・RebuildThread）、cpp:1352（shutdown close・join 後単一スレッド）。
- **トポロジー訂正（D148 に対する本監査の発見）**: `onPublishCommitted` の実呼び出しは ISR（audio callback）ではなく **CoordinatorLoop**。経路: `processIntent`（Threading.cpp:263・CL）→ `PublishIntentHandler::handle`（ProcessIntent.cpp:148-149）→ `PublishExecutor::executePublish`（RuntimePublishExecutor.h:114 で onPublishCommitted）。AudioEngine.h:3727「complete() は CoordinatorLoop の単一スレッド（executePublish → …）」が実態。コメント群の「ISR」は CoordinatorLoop の歴史名（ISRCoordinatorLoop.cpp）。**実 audio callback は obligation を触らない**。→ naive T3 反証の terminalizer は「ISR」ではなく「CL（Route B）+ RebuildThread（Route A/C）」に読み替え。判定不変（§7）。

### 1.6 スレッドトポロジー確定表（T3b 当事者）

| 行為 | T3b でのスレッド | 実測根拠 |
|---|---|---|
| postSignal（6 サイト） | RebuildThread | RebuildDispatch:1006/1033/1091/1115 + Orchestrator:311/401（rebuildThreadLoop / RebuildThread 同期 submitPublishRequest — Orchestrator.h:137, Threading.cpp:291） |
| adjudicate（drain CAS→check→apply） | **CL**（提案: processIntent 後・redrive 前） | runCoordinatorPhase=Threading.cpp:258、CL=ISRCoordinatorLoop.cpp:8/39 |
| tryInsert | **CL** のみ | 唯一箇所 cpp:947 ← submitRecoveryRequest ← submitRecoveryIntent ← QuarantineIntentHandler（ProcessIntent.cpp:140・CL） |
| resolve（terminal 化 + counter reset） | CL（Route B :354）+ **RebuildThread（Route A :320 / Route C :371 / 枯渇 :1080）** | §1.5 |
| delivery 書込 | T3b 後 CL のみ（cpp:1071 が adjudicate へ移動） | §1.4 |

**核心**: terminalizer（resolve）は apply（CL）と**別スレッド（RebuildThread Route A/C）から並行可能**。reuse（tryInsert）は apply と**同一スレッド（CL）**。

---

## 2. T3b 提案の再掲と保護範囲の切り分け

```text
producer(RebuildThread): CAS while (id==O ∧ count<K)   ← id タグで「加算瞬間」を保護
consumer(CL):  CAS (O,c)→(O,0) 保持 w → state==Live 判定 → apply{delivery=None; counter+=n; ≥K→resolve(Failed)}
tryInsert(CL): store {N,0}
```

tagged CAS が保護するのは **signal の取り出し（extraction）瞬間の id** のみ。
`apply()` の 3 書込（delivery plain / counter fetch_add / resolve）は **extraction と同一語ではない**。
指示の中核主張は**実証された**:

> tagged signal の identity は保護されているが、adjudication の destination slot identity と
> `consecutiveFailureCount` の更新は atomic に束縛されていない。

D148 §2 の「適用先は w の id — 現在の slot id と原子対」は不正確: 適用先は **slot index i** であり、w.id==slot.id は（a）reuse が apply に割り込まない（affinity）ことでのみ成立し、（b）state の liveness は一切保証しない（resolve は id を変えないため）。

---

## 3. D148 race matrix の欠落の特定

D148 §3 Case B/C が解析したのは「exchange 後に **producer が signal を加算** → 次 drain で state!=Live drop」= **signal 側**の stale。
未解析の交差順 = **「exchange 成功 → state 判定通過 → resolve（他スレッド）→ apply 実行」** = **apply 側**の stale。本監査の反証はここを突く（Case B′/§5）。

---

## 4. Race matrix（D149・A-G）

記号: `drain`=CL CAS (O,c)→(O,0) 成功・w={O,c} 保持 / `chk`=CL state==Live 判定 / `apply`=delivery=None + counter+=n (+≥K resolve) / `R`=resolve(O)（Route A/C=RebuildThread、Route B=CL）/ `T`=tryInsert(N)（CL）

| Case | 順序 | 結果 | 判定 |
|---|---|---|---|
| A | drain → chk(Live) → apply | n が O（Live）に正しく適用。delivery=None・counter+=n・K 判定正常 | **正常 ✓** |
| B | drain → chk(Live) → **R(RebuildThread, Published/Stale)** → apply | apply の destination は slot i（id==O のまま・terminal）。`delivery=None` が **terminal obligation への書込**に、`counter.fetch_add` が resolve の `store(0)` を**上書き**（terminal O に counter=n>0）。liveCount は state CAS 単一 authority で −1 正確 | **STOP #5 到達 → ✗** |
| C | drain → chk(Live) → R → **T(N)** → apply | apply が N の slot に delivery=None + counter+=n（N の実効 K 短縮 + N の delivery 強制 None → redrive 重複表現の危険）。**提案配置では到達不能**: T と apply が同一 CL で program-order 上 apply→T（apply は drain と同一 adjudicate パス内、T は processIntent パス内）。ただし保護の本体は **affinity であって tagged CAS ではない** | **到達不能（affinity 依存）・契約明記必須** |
| D | （drain なし）R → T → 旧 word {O,c} は store {N,0} で消滅 | 旧 O の未 adjudicate signal が**無音で消失**（drop telemetry 経路を通らない — D148 §10 の drop は「drain 済み + state!=Live」のみ分類）。O terminal につき意味論的損失なし、ただし観測分類の欠落 | **意味論 ✓ / telemetry 分類ギャップ（軽微）** |
| E | R → T（signal なし） | 通常 reuse。tryInsert が id/state/delivery/counter/word を一括初期化 | **正常 ✓** |
| F | drain → chk(Live) → （R なし）→ T → apply | T のゲートは `state!=Live`（h:397）。R なしに Live slot は再利用されず、同一 CL で apply 前に T は走らない → **単独では到達不能**（R を伴えば Case B/C に帰着） | **到達不能 ✓** |
| G | drain → chk → apply: counter+=n ≥K → resolve(Failed)[CL] ∥ R(RebuildThread, Published) | state CAS の勝者のみ terminal 化・−1（二重 −1 なし ✓）。敗者の CAS は idempotent no-op ✓。ただし (i) CL は resolve 成否によらず `recoveryRetryExhaustedCount_` を先に増加 → **telemetry 過計上**（現行 cpp:1079 と同型・非新規）、(ii) counter は勝者の reset と敗者/先行 add の順序で terminal O に 0 または n が残る（**#5 と同一ドメイン問題**） | **liveCount ✓ / counter invariant ✗** |

**C/F/G の GO 可否**: F=GO（構造的到達不能）、G=liveCount は GO だが counter invariant は NO-GO、**C=提案配置に限り到達不能（affinity 依存・未文書化）**。

---

## 5. 最重要反証構成（STOP #5・実在経路のみ）

```text
t0  slot i: id=O, state=Live, word=(O,1)   ← RebuildThread postSignal(O) 済み（意図 #1 失敗観測）
t1  RebuildThread: 意図 #2 の recovery build 成功 → publish commit（CL Route B）→
    trySubmitImpl 復帰 → Orchestrator.cpp:320 resolveRecoveryObligation(O, Published) [Route A]
t2  CL adjudicate: CAS word (O,1)→(O,0) 成功、w={O,1} 保持
t3  CL: state.load()==Live 通過（t1 の CAS 可視化前）
t4  RebuildThread resolve: CAS Live→ResolvedSuccess 成功 → counter.store(0) h:429 → liveCount−1
t5  CL apply: delivery=None（h:356 plain・terminal O へ）; counter.fetch_add(1) → **counter=1 on terminal O**
    （resolve の reset が old adjudication に上書きされた — STOP #5）
t6  （任意）次 tick CL tryInsert(N) が slot i 再利用 → counter=0 で消えるが、
    t5..t6 窓の不変条件破棄は消えない（指示: 中間状態込みで検証）
```
- t1 の意図二重在席は実在: coalesce re-push が同一 oblId の transport 表現を複数許す（cpp:962-969 明記「ALLOWED to re-push」）。失敗観測（postSignal）と後続意図の成功 terminal 化（Route A/C）が同一 obligation で重なることは設計上正常。
- 各段は現行コードの実在命令。窓は微小だが到達不能ではない。
- 「最終 state が Live なら問題ない」では無効化しない: **terminal 化済み obligation の counter が resolve の契約値 0 から逸脱**し、delivery が terminal 後に書き換えられる（所有権 invariant 破棄）。

### 順序入れ替え版（指示の第 2 構成）
```text
t3 R resolve(O) → t4 T tryInsert(N) → t5 CL apply(old O, n)
```
→ **提案配置では不成立**（t4 と t5 が同一 CL・apply は drain と同一パスで T より先）。成立条件は「adjudication の CL 外配置」または「drain と apply の phase 分離」— いずれも Phase 2 実装で破り得るため、**契約不変条件として明記＋grep 強制**が必須（§8 I-2/I-3）。破られた場合 t5 は N の delivery/counter を汚染（Case C 実効化）。

---

## 6. 指示 §追加検証への回答

### 6.1 delivery の race
- `state check → resolve → delivery=None` が terminal obligation への write になり得るか → **なる**（Case B′ t5。resolve は delivery を触らない h:423-436、apply の delivery 書込は state と非束縛）。
- ただし T3b は delivery 書込者を CL へ一元化するため **data race（現行 W5 の UB・§1.4）は閉じる**。残るのは「terminal への論理 write」（inert: 全 reader は Live ゲート付き — redrive cpp:1110、findByKey h:384、wasDeferredBefore cpp:901）。inert でも invariant 違反（§0）。

### 6.2 consecutiveFailureCount の race（最重要）
- resolve の `store(0)`（h:429・任意スレッド）と apply の `fetch_add`（CL）は**別 atomic domain**。`store(0) → fetch_add` 順序が Case B′ で成立 → **reset が上書きされる**。
- slot reuse 込みの「O terminal → N inserted → old O adjudication → N counter modified」は、N 汚染として**提案配置では到達不能**（affinity）。ただし証明は §5 第 2 構成の成立条件禁止に依存し、tagged word 自体は寄与しない。
- 現行コードとの比較（新規悪化の有無）: 現行 markTransientFailure（RebuildThread）は tryInsert（CL）と**別スレッド**のため、同一 oblId の重複 transport 表現（cpp:962-969）下で **Case C が現行で実際に到達可能**（cpp:1066 通過後 → CL Route B resolve + CL tryInsert → cpp:1071/1074 が N に着地）。T3b の CL 移設はこの現行混入を affinity で閉じる。**つまり T3b は現行より改善だが、指示の STOP 基準（#5・#6）は満たさない。**

### 6.3 resolve の id 再スキャン論法の consumer 側適用
- resolve 自身の destination 保護（ABA 回避）には id 再スキャン＋単調 id（h:415-419）で十分 — これは**維持**。
- 同じ論法を apply へ適用すると破綻する: apply の destination は drain 時の slot index。CAS 時点で id==O でも、**write 時点で id==O かつ Live** は (i) reuse 非干渉（affinity）と (ii) terminal 化非干渉の 2 条件を要するが、tagged word は (i) にしか寄与せず、(ii) には一切寄与しない（resolve は id を変えないため id 再検証でも terminal を排除できない — **terminal 化は id 不変・state のみ変化**）。
- 明示的切り分け: 「signal の CAS 時点で O」≠「counter/delivery の write 時点で O（かつ Live）」。後者を保証するには destination identity と liveness と counter を**同一 CAS ドメイン**で束縛するしかない → T3c。

### 6.4 markTransientFailure の「単一 lifecycle protection」分断検証（指示の中心論点）
- 現行 markTransientFailure（cpp:1059-1085）の「delivery None 化 + counter 加算 + 枯渇 resolve」は**契約上 "Atomically performs"（h:475）と謳われるが実装は check-then-act**（cpp:1066 判定 → 1071/1074 書込）で、もともと atomic domain ではない。すなわち T3b は「元々一つだった atomic protection を分断」したのではなく、**元々分断されていた protection の所在を RebuildThread から CL へ移し、signal 移送だけを tagged 化した**。
- その結果、分断は (a) signal transport（tagged・安全）と (b) adjudication apply（**非 tagged・terminal-write 窓 + affinity 依存**）に残る。**T3b は lifecycle protection を再結合していない** — これが NO-GO の本質。

---

## 7. D148 に対する訂正・不変点

| 項目 | D148 の記述 | D149 実測 | 影響 |
|---|---|---|---|
| Route B terminalizer | 「ISR: onPublishCommitted」 | **CoordinatorLoop**（executePublish は processIntent 経由・AudioEngine.h:3727） | naive T3 反証は CL+RebuildThread の並行で**依然成立**（判定不変）。lock-pool 許容論拠（「ISR 非接触」）は**より強くなる**（実 ISR 完全非接触） |
| consumer 保護 | 「適用先は w の id — slot id と原子対」 | 適用先は slot index。id 一致は affinity 由来、liveness は未保護 | **T3b GO 判定を NO-GO へ覆す**（§4-5） |
| matrix | Case A-D（signal 側 stale のみ） | apply 側 stale（drain→check→resolve→apply）未解析 | 欠落反証を本件 §5 で構成 |
| 14 GO 条件 | 全 ✓ | #5/#6 相当が不成立 | 条件付き GO → **NO-GO** |

---

## 8. T3c 設計要求（D150 へ — 実装しない）

単一 atomic ownership domain への統合要件:

```text
(signal identity) + (adjudication identity) + (counter/state transition) を 1 CAS に束縛
```

推奨形（設計案・D150 で再証明）:
- **Lifecycle word**: `W = { obligationId:64, state:8, pending:8, adjudicated:8 }`（16B・alignas 16・cmpxchg16b or MSVC atomic struct。実 ISR 非接触が §1.5 で確定済み → lock-pool 許容、lock-free 要請時 cmpxchg16b）。
- postSignal(O): `CAS W (id==O ∧ Live ∧ pending<K) → pending+1`。
- adjudicate: `CAS W (id==O ∧ Live ∧ pending>0) → (pending=0, adjudicated+=n)` — **drain と counter 適用が単一遷移**。`adjudicated+n ≥ K` なら同一 CAS で `(terminal Failed, pending=0, adjudicated=0)` + liveCount−1（勝者のみ −1）。
- resolve(id, T): `CAS W (id==O ∧ Live) → (T, 0, 0)` — **reset が terminal 化と同一語**で、old adjudication に上書きされ得ない（#5 構造排除）。
- tryInsert: `store W {N, Live, 0, 0}` — id/state/counter/pending が単一語で原子（D148 §1-E 順序要求も内包）。
- delivery: plain のまま**CL 単一書込者**（§6.1 の data race 解消は T3b 同等に維持）。
- 契約不変条件（明記 + grep 強制）: **I-1** resolve の −1 authority は W の Live→terminal CAS 勝者のみ / **I-2** adjudication は CL のみ・drain と apply は同一 CAS 内（分離禁止）/ **I-3** tryInsert は CL のみ（本番 call site 1 件の維持）/ **I-4** delivery 書込者 = CL のみ。
- D150 で残課題として検証: Route A/C（RebuildThread）の W CAS と CL の W CAS の memory order 最小十分性、telemetry（exhausted 過計上・reuse-overwrite drop 分類）、R18/R20 同期アサート適配、16B 境界の alignas/padding（C4324 系）、`ObligationState` 6 値 + pending/adjudicated のビット幅（K=4 → 8bit 十分）。

---

## 9. STOP 条件最終判定

| # | 条件 | 判定 |
|---|---|---|
| 1 | 旧 O の signal が N に混入 | 提案配置では到達不能（affinity・CAS 非依存）→ 契約明記なしに GO 不可 |
| 2 | 旧 O の adjudication が N の counter を変更 | 同上 |
| 3 | 旧 O の adjudication が N の delivery を変更 | 同上 |
| 4 | terminal 後の旧 adjudication が現役 slot state を破壊 | 「現役」への破壊は不成立。terminal slot への delivery/counter 書込は**成立**（inert だが invariant 破棄） |
| 5 | **resolve の counter reset が old adjudication に上書きされる** | **到達可能（§5）→ NO-GO 発動** |
| 6 | slot reuse 後の apply destination identity を証明できない | CAS では証明不能。affinity + 同一 phase 依存。**未文書化のため証明不成立** |

**Verdict: T3b = NO-GO（分岐 B）。Phase 2 実装凍結継続。D150（T3c 設計再証明）へ回付。**
