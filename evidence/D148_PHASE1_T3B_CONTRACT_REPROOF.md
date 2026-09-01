# D148 — Phase 1: T3 Saturating Failure-Signal Contract Re-proof (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only disproof-first design audit. **Production/Test source changes: 0. 実装 0。Ring 0。W5 移動 0。**
**基準:** `ConvoPeq.md Generated: 2026-08-31 15:32:58`（D146 後版）。D147 の結論は前提とせず現行コードから再検証。

## 最終判定（先出し）

```text
naive T3（D147 原文: id-scan → Live check → 独立 counter fetch_add）= NO-GO
  → STOP 条件 #1 が現行コードで到達可能（下記 §2 で反証構成）
修約 T3b（tagged word: (obligationId, count) を単一 CAS ドメイン化）= GO
  → 14 GO 条件すべて証明可能（§11 表）
Phase 2 は T3b に対してのみ許可（naive T3 の実装は禁止）
```

---

## 1. Current source baseline（実測）
- `LogicalRecoveryObligation`: `atomic<u64> id`（h:348）、`atomic<ObligationState> state`（h:350）、`delivery` plain（h:356）、`atomic<u8> consecutiveFailureCount`（h:361）。
- `tryInsert`（h:392-413）: `id.store(relaxed)` → identity → `state.store(Live, release)` → delivery/generation → `counter.store(0, release)` → liveCount +1。
- `resolve`（h:423-436）: id 再スキャン + `state CAS(Live→terminal, acq_rel)`、成功時のみ counter reset + liveCount −1。**単一の −1 authority**（INV）。
- **`resolveRecoveryObligation` は ISR から呼べる**（h:468「Callable from ISR (onPublishCommitted)」実測）→ O の terminal 化は CL/RebuildThread 以外の非同期事象。
- 現行 `markTransientFailure`（cpp:1066-1092）: id スキャン → `state.load(acquire)==Live` → delivery plain 書込 → counter fetch_add → 枯渇 resolve。**この scan→act 二段構造自体に §2 と同一の TOCTOU が既存潜在**（T3b はこれを設計で排除する）。
- R18 テスト（test:1486 他）: `markTransientFailure(*id);` 直後に `liveLogicalRecoveryObligationCount()` を**同期アサート** → Phase 2 の API 契約に直結（§12）。

## 2. Slot lifetime / stale-id proof — **naive T3 の反証（STOP 条件 #1 成立）**

反証構成（naive T3: `scan(id==O ∧ Live) → signals[i].fetch_add(1)`）:
```text
t0  Builder(RebuildThread): postSignal(O) — slot i で id==O, state==Live を観測・通過
t1  ISR: onPublishCommitted(O) → resolve CAS(Live→ResolvedSuccess)   [h:468 経路・非同期]
t2  CL: processIntent の submit → tryInsert が slot i を再利用（id:=N, signals reset, Live）
t3  Builder: signals[i].fetch_add(1) → **N に phantom failure 帰属**
t4  CL adjudication: n=1 を N に適用 → N の実効 K が 3 に短縮（意味論違反）
```
- 「id が単調だから ABA なし」は **resolve の id 再スキャンにのみ有効**で、producer の scan→add 間隔を保護しない（指示の指摘どおり）。
- 各段は全て現行コードに存在する経路（ISR resolve / CL tryInsert / RebuildThread producer）で、窓は微小だが**到達不能とは言えない**。
- **よって naive T3 は §Phase 1-A の要求「id==O 観測から counter 変更まで slot reuse/terminal が誤帰属させない」を証明できない → NO-GO。**

### 修約 T3b（tagged word）
```cpp
// slot 内: (obligationId, count) を単一 CAS ドメインに統合
struct FailureSignal { std::uint64_t obligationId; std::uint8_t count; };
std::atomic<FailureSignal> failureSignals;   // 16B。触达者=RebuildThread(producer)/CL(consumer/tryInsert)
                                             //   — ISR 非接触（resolve は state 側のみのため lock-pool 許容）
producer(O):  loop { w = load(acquire);
                    if (w.obligationId != O) return no-op;   // stale/unknown/reuse 一括排除
                    if (w.count >= K) return no-op;           // 飽和（§4 で inert 証明）
                    if (CAS(w, {O, w.count+1}, acq_rel)) return posted; }
consumer():   loop per slot { w = load(acquire);
                    if (w.count == 0) continue;
                    if (!CAS(w, {w.obligationId, 0}, acq_rel)) continue;
                    apply(w.obligationId, w.count); }          // 適用先は w の id — 現在の slot id と原子対
tryInsert:    failureSignals.store({N, 0}) — id 書換と count クリアが単一語で原子
```
- **誤帰属不能の証明**: 加算は「CAS 瞬間に id==O」を要求。reuse は (id,count) を一体で交換するため、t3 の CAS は id==N を見て失敗（no-op）。t2 前に成功した加算は (O,c) に載り、consumer はその id=O を保持して読む → 適用先は O（terminal なら §4 の drop 経路）。**phantom は生成不能。**
- 実装注記: MSVC x64 の 16B atomic は lock-pool 使用の可能性（`is_lock_free()==false`）。触达者が ISR でないこと（resolve は state のみ）と、NonRT 2 スレッドのみであることで許容。lock-free 必須なら `InterlockedCompareExchange16b` 明示実装（Phase 2 選択）。

## 3. Signal-vs-resolve race matrix（T3b・Case A-D）

| Case | 順序 | 結果 | 判定 |
|---|---|---|---|
| A | signal → exchange → adjudicate | n が正しく O に適用（delivery=None、counter+=n） | ✓ |
| B | exchange(0) → signal(+1) → resolve | signal は (O,1) として載る。次 drain で n=1 だが state!=Live → **drop（telemetry）**。resolve の −1 は 1 回のみ（state CAS authority 不変） | ✓ |
| C | signal → exchange → terminal resolve → stale signal | 2 つ目の signal CAS は id==O かつ count<K なら成功しうる（terminal でも id は O のまま）→ 次 drain で state!=Live により drop。**counter 二重加算なし・別 obligation 混入なし** | ✓ |
| D | signal → exchange → slot reuse(N) → 旧 O の stale signal | producer CAS は `w.obligationId==O` 要求 → reuse 後は id==N → **失敗 no-op**。N への混入構造的不可能 | ✓ |

lost-signal 分析: exchange と CAS の競合は CAS ループで直列化（consumer が勝てば producer は新値で再試行、producer が勝てば consumer が再試行）。**signal は失われないか、意図的 drop（terminal/reuse）に分類される。**

## 4. Saturation semantic proof（Phase 1-C）

「K 超過=inert」の条件付き成立を全 terminal 経路で検証:
- **枯渇 terminal（T3b 自身の経路）**: counter+=n が ≥K を判定。現行の 1 回ずつ加算（5 回目 counter=5→terminal）と**到達する terminal 帰結が同一**（≥K 判定のため）。飽和は「terminal を遅延させない」方向にのみ作用。
- **Published（ISR/Route A）**: terminal 化後の residual signal は consumer の state 判定で drop。現行コードも 5 目以降の直接呼びは Live チェックで no-op — **同一**。
- **StaleSuperseded / ShutdownDiscarded**: 同上（resolve 経路は state CAS を共有）。
- **Retry（resolveRecoveryObligation の Retry）**: Live 維持・delivery 不変・counter 不変（cpp:1030-1031）。signal は Live に加算されるので混同なし。ただし **Retry は signals を消費しない** — 現行の rearm/settle 経路と独立（§6）。
- **slot reuse**: §2/§3-D で構造排除。
- 唯一の意味論差: 現行は「観測時刻」に counter 加算、T3b は「adjudication 時刻」にバッチ加算。**terminal 判定（≥K）と delivery=None の最終状態は同一**、変化するのは None 化のタイミング ≤1 tick（§6）。
→ **条件「consumer 側 state 判定必須」付きで飽和は意味論保存。無条件の『5 回目は不要』は不採用。**

## 5. Reset / reuse ordering proof（Phase 1-E）

現行 tryInsert の実順序（h:398-408）から再導出:
```text
1. id.store(N, relaxed)        ← 現行
2. identity/delivery/generation 書込
3. counter.store(0, release)   ← 現行
4. state.store(Live, release)  ← 現行の publication fence
+ T3b: failureSignals.store({N, 0}) を **1 と同一タイミング**（id 書換と原子対）で行う
```
- producer は id を**タグ込み CAS**でのみ読むため、(N,0) 交換完了前は旧 O の加算が成功し、完了後は失敗する — 境界が単一語で原子。
- consumer は exchange した id を保持して適用するため、reuse 後の drain が旧 count を N に流すことはない。
- **禁止順序**: state=Live を先に立ててから word をクリアする（窓が生じる）。契約に明記。

## 6. delivery ownership migration proof（Phase 1-D）

```text
現行: RebuildThread markTransientFailure = { delivery=None(W5), counter++, 枯渇 resolve }
T3b : RebuildThread postSignal = { tagged CAS のみ }
      CL drainAdjudication = { exchange → delivery=None(1回) → counter+=n → ≥K resolve }
```
順序追跡:
- **failure → redrive**: delivery=None が ≤1 tick 遅延。窓内 redrive は Transport 表示をスキップ（二重住処なし）→ 次 tick の None 化後 redrive 候補化。**遅延のみ・喪失なし**。
- **failure → exhaustion**: 判定は adjudication 移動。K 到達帰結同一（§4）。
- **Transport→None repair（R18_T9）/ Durable→None repair**: 適用主体が CL になるだけ。durable 窓（slot=O ∧ None）は §4 の adjudication 後に生じ、P3 repair（AlreadyRepresented）で回収 — 経路不変。
- **delivery reader（cpp:900/943/1116/1142）は全て CL** — 書き手も CL 化により **delivery 単一スレッド化が完成**（h:318-324 の主張が真になる）。
- wake/liveness: adjudication は runCoordinatorPhase の processIntent 直後・redrive 前に配置 → 同一 tick で None→redrive→P2 latch→wake。**現行（次 tick redrive）と同一か 1 手早い**。

## 7. memory-order proof（Phase 1-F）

- signal が運ぶ情報は **(oblId, count) のみ**。producer は payload/delivery/state を一切触らない（STOP 条件 #3 充足）。
- 従って word 内部の release/acquire で十分: producer CAS `acq_rel`、consumer load/CAS `acq_rel`（対になる）。adjudication が読む `delivery` は **同一 CL スレッドの plain**（program order で足りる、跨がない）。`consecutiveFailureCount` は atomic 独立。
- 「強ければ安全」ではなく**最小十分**: 跨スレッドで共有されるのは word と state/counter/id の既存 atomic のみ。

## 8. counter implementation proof（Phase 1-G）

- 型: `uint8 count`（0..K=4 のみ必要 — 飽和により 255 wraparound 到達不能: `count>=K` で打ち切り、fetch_add 単体使用禁止）。
- saturation primitive: §2 の CAS ループ。**linearization point = 各 CAS 成功**。
- concurrent producer×n: CAS 直列化で 1 加算ずつ。concurrent consumer: 同上。terminal 遷移（state CAS）とは別語 — 整合は consumer の state 再判定で担保（§3-B/C）。

## 9. revised exactly-once terminology（Phase 1-H）

提案契約名: **「bounded failure-observation accounting」**
```text
1 failure observation → 1 successful tagged increment,  while (id==O ∧ count<K)
                      → inert（no-op）,                 id≠O（stale/reuse/unknown）または count==K（飽和）
```
既存 R20-3「exactly once failure counting」との整合: 同テストは「1 呼び=1 加算（+2 にならない）」を要求 — T3b では「1 観測=1 成功加算（飽和後は加算自体が発生せず、terminal 帰結は同一）」で**観測単位の一対一を維持**。K 超過破棄は「terminal 確定後の観測破棄」と論理同値（§4）。契約文書では「exactly-once」を**観測→加算**の対応に限定し、**加算→adjudication** はバッチ（n 件一括）であることを明記。

## 10. responsibility split（Phase 1-I）

```text
pendingFailureSignals.tagged count = 未 adjudicate 観測（id タグ付き）
consecutiveFailureCount            = adjudicate 済み連続失敗
```
- `signals=3, count=1` は正常中間状態（adjudication 未実行）。
- 移送途中の失敗: consumer の CAS 成功後 apply 前に ISR resolve → apply 時 state!=Live → drop（telemetry `recoveryFailureSignalDroppedCount_`）。obligation は terminal 済みで情報損失なし。CAS 失敗（producer 競合）は再試行 — **lost も double もなし**。
- tryInsert は両者を同時にリセット（§5）。

## 11. state-machine closure（Phase 1-J）

```text
            postSignal(O) [id==O∧c<K で CAS 成功]
   ┌───────────────┐
   │ Live (id=O)   │◄──────────────┐
   │ c∈[0,K)       │               │ tryInsert: store({N,0}) + state Live(release)
   └──────┬────────┘               │ （旧 O の postSignal は id≠O で no-op）
          │ CL drain: CAS (O,c)→(O,0)
          ▼
   adjudicate(O, n):
     state==Live? ──no──► drop(n) + telemetry   ← Published/StaleSuperseded/ShutdownDiscarded 後・reuse 後
     │yes
     ▼
   delivery=None(CL) ; counter+=n
     ├ counter<K → Live 継続（redrive 候補化→P3 repair/fallback→wake）
     └ counter≥K → resolve(Failed) [state CAS] + exhausted telemetry
   並行事象:
     ISR resolve(Published) anytime → state CAS。drain は state 再判定で整合（§3-B/C）
     rearm(RejectedPressure): settle CAS のみ。signals 生成なし（§6 独立）
     shutdown: CL join → Builder join → discard。post-join の postSignal は id スキャン no-op
```
全遷移が閉じている（未定義矢印なし）。

## 12. GO / CONDITIONAL / NO-GO + Phase 2 contract

**14 GO 条件判定（T3b 基準）**: 1 ✓(§2 tagged) / 2 ✓(§3-D) / 3 ✓(§3) / 4 ✓(§4 条件付き→条件実装込み) / 5 ✓(§5) / 6 ✓(§6) / 7 ✓(§7) / 8 ✓(§8) / 9 ✓(§9) / 10 ✓(§10) / 11 ✓(§3-B/C/§11) / 12 ✓(§6) / 13 ✓(§7: 新 plain access 追加なし、word は atomic) / 14 ✓(§11)。

```text
Verdict: CONDITIONAL → GO（修約 T3b に限る）
  naive T3 は STOP 条件 #1 で実装禁止。
  Phase 2 実装対象（T3b）:
    + LogicalRecoveryObligation::failureSignals（tagged (oblId,count)、16B atomic or InterlockedCompareExchange16b）
    + postRecoveryFailureSignal(oblId)  — 6 サイトの markTransientFailure 直接呼びを置換
    + adjudicateRecoveryFailureSignals() — CL: runCoordinatorPhase の processIntent 後・redrive 前
    + W5 廃止（delivery=None は adjudication 内 CL 書込へ）
    + telemetry: signalSaturated / signalDropped /（既存 exhausted/deferred と突合可能に）
    + tryInsert: word store({N,0}) を id 書換と同時（§5 順序）
    + テスト適配（重要）: R18/R20 群は markTransientFailure 直後の同期アサートを持つ（test:1486 他実測）。
      markTransientFailure を「signal+即 adjudicate（CL 文脈互換）」。
      split した postSignal 単体テストは明示的 drain 呼び出しで構成。既存 40 の意味は変えない。
    + h:318-324 delivery 単一書込者コメントの真化（実装後に更新）
  Phase 2 で追加検証: Debug/Release CTest、AV stress 200、構造監査（delivery writer=CL のみ grep 証明）
```

**STOP — 実装 0。Phase 2（T3b 実装）の指示を待つ。P5/P6 非着手。**
