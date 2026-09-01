# D148 — Phase 1: T3 Contract Re-proof（Work Report）

**Status: CONDITIONAL → GO（修約 T3b に限る）。naive T3 は NO-GO。変更 0・実装 0。**
**詳細:** `evidence/D148_PHASE1_T3B_CONTRACT_REPROOF.md` / **基準:** ConvoPeq.md `15:32:58`（D146 後版）

## 中核の発見: naive T3 は STOP 条件 #1 が現行コードで到達可能
反証構成（実測根拠付き）:
```text
t0 Builder: postSignal(O) — slot i で id==O ∧ Live を観測・通過
t1 ISR: onPublishCommitted → resolve(O)（h:468「Callable from ISR」実測 — 非同期 terminalizer の存在）
t2 CL: tryInsert が slot i を再利用（id:=N）
t3 Builder: signals[i].fetch_add → N に phantom failure（実効 K が 3 に短縮）
```
「id 単調性」は resolve の再スキャンにのみ有効で、producer の scan→add 間隔を保護しない — 指示の懸念どおり。**naive T3（id-scan + 独立 counter）は実装禁止。** なお同一の TOCTOU は現行 markTransientFailure にも潜在している（T3b はこれを設計で排除）。

## 修約 T3b: tagged word（(obligationId, count) を単一 CAS ドメインに統合）
- producer: `CAS while (id==O ∧ count<K)` — id 不一致（stale/unknown/**reuse 後**）は構造 no-op
- consumer(CL): `CAS (id,c)→(id,0)` 後、**交換した id を保持して適用**、state!=Live なら drop（telemetry）
- tryInsert: `store({N,0})` — id 書換と count クリアが単一語で原子
→ 誤帰属は CAS 瞬間の id 一致を要求されるため**生成不能**。16B atomic は触达者が RebuildThread/CL のみ（ISR 非接触）なので lock-pool でも許容、lock-free 必須なら InterlockedCompareExchange16b（Phase 2 選択）。

## 各 Phase の証明結果
- **1-B race matrix（Case A-D）**: 全 Case で lost なし・混入なし・二重加算なし（§3 表）。
- **1-C K 飽和**: 「5 回目は不要」は無条件では不採用。**consumer 側 state 判定必須**の条件付きで意味論保存（Published/StaleSuperseded/ShutdownDiscarded/Retry/slot reuse の全経路を確認、§4）。terminal 帰結（≥K 判定）は現行と同一、変化するのは delivery=None のタイミング ≤1 tick。
- **1-D delivery 移行**: W5 廃止→adjudication で CL 化。failure→redrive は ≤1 tick 遅延（喪失なし）、adjudication を processIntent 後・redrive 前に配置すれば同一 tick で None→redrive→wake。delivery 単一スレッド化が完成し h:318-324 が真になる。
- **1-E reset 順序**: 現行 tryInsert の publication 順序から再導出。word store は id 書換と同時、**state=Live 先行→word クリアは禁止**。
- **1-F memory order**: signal は (oblId,count) のみ運び producer は payload/delivery/state を触らない → acq_rel の対で最小十分（「強いほど安全」不採用）。
- **1-G saturation**: fetch_add 単体使用禁止（255 wrap）、CAS ループ、linearization point=CAS 成功。
- **1-H 用語**: 「exactly-once」→**bounded failure-observation accounting**（1 観測↔1 成功加算 while id==O∧c<K、以降 inert）。R20-3 との整合は観測単位の 1:1 として維持。
- **1-I 責務分離**: signals=未 adjudicate / consecutiveFailureCount=adjudicate 済み。移送途中の ISR resolve は drop+telemetry で無損失。
- **1-J 状態機械**: 全遷移閉包（§11 図）。

## 14 GO 条件: 全 ✓（T3b 基準）
**Phase 2 実装契約（T3b に限定）**: failureSignals tagged word / postRecoveryFailureSignal(oblId)（6 サイト置換）/ CL adjudicate drain / W5 廃止 / telemetry（saturated・dropped）/ tryInsert 順序 / **R18・R20 テスト適配**（markTransientFailure 直後同期アサートが実測あるため「signal+即 adjudicate」互換セマンティクスを明示）/ h:318-324 更新は実装後。

**STOP — 実装 0。Phase 2（T3b 実装）の指示を待つ。P5/P6 非着手。**
