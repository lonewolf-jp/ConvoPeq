# D146 — Phase 2 Atomic-State Implementation + Immediate Structural Audit (Record)

**Date:** 2026-08-31 (+09:00)
**Type:** implementation（I-HS2 の Phase 2 部分のみ）+ 即時構造監査。**RecoveryFailure ring は未実装（D147 に分離）。**
**Build evidence:** `evidence/d146_ctest.log` — Debug full build→CTest `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release 同（REL_CTEST_EXIT=0）
**AV stress:** Release 200 反復 → **0 失敗**（D142 基準維持）
**ConvoPeq.md:** 実装後再生成 `Generated: 2026-08-31 15:32:58`（`D146` マーカー 25 件 = 実ツリー同期確認）
**Diff scope:** coordinator.cpp +135 / coordinator.h +37 / tests +144（vs D145 時点）。RebuildDispatch/Threading は P1/P2 値のまま無変更。TEMP 残留 0。
**Verdict: GO（D146 全基準成立）→ D147 進出可**

---

## D146-0 実装前スナップショット（要点）
- state 接触 11 箇所（991/998/1100/1158/1159/1214/1227/1234/1243/1260/1261）、predicate 6 箇所（549/1007/1168/1233/1247/1266）を全数固定。
- 既存 X1 テスト（DurableAdmission/Coalesce/LeaseRetry）の期待値は「take 1 回・settle 遷移・hasPending」のみで、no-op 化と互換（実装前に確認）。

## D146-1..6 実装内容

### state atomic 化（.h）
`State state = NoAdmission` → **`std::atomic<State> state{State::NoAdmission}`**。
- `static_assert(trivially_copyable)` は削除（atomic メンバにより非 trivial化。構造体は memcpy されず、reset は明示 field クリア＋state release の順序規約へ）。
- 契約コメントを CAS プロトコル基準に更新（旧「SPSC/plain・競合なし」記述を置換。h:960-961 の coalesce 系陳腐コメントも同時訂正）。

### take = CAS lease（D146-2）
```cpp
State expected = DurablePending;
if (!state.compare_exchange_strong(expected, Building, acquire, relaxed))
    return std::nullopt;          // CAS 失敗 = lease 取得失敗、payload 非読取
// acquire が attach の release と対 → payload 可視・安定
```
二段判定（load→store）は廃止。**CAS 成功そのものが lease 取得**。

### settle CAS（D146-3）
- retry=true: `CAS(Building→DurablePending, release)`（非 Building は防御的 no-op）。
- retry=false: **`payload reset → CAS(Building→NoAdmission, release) → predicate false`**（reset 順序固定、逆順禁止をコメント明文化）。predicate と state の対応: predicate は isFullyDrained 用の粗フラグ、**権限は state**。

### 単一 attach primitive（D146-4/5）— `tryAttachDurableRecovery`
```
state.load(acquire)
 ├ NoAdmission → payload 全書込 → CAS(NoAdmission→DurablePending, release) → Attached
 │   （CAS failure は live 中不能: NoAdmission からの唯一退出権限=CL 自身、discard は join 後。
 │     万一時も残骸 payload は NoAdmission 下で非読取・次 attach が全面上書 → rollback 不要）
 └ 占有中 → oblId==O ? AlreadyRepresented（no-op・payload 非書込）: OccupiedByOther
```
- **submit durable 経路**: 旧「guard 通過→無条件上書」を廃止し helper へ収束。OccupiedByOther のみ defer（W3 意味・recoveryRetryDeferredCount_ 不変）、Attached/AlreadyRepresented は delivery=Durable + true。
- **redrive**: 付着・P3 repair・transport fallback が同一 primitive に収束（Attached∪AlreadyRepresented→delivery=Durable+latch、OccupiedByOther→transport fallback=C13 不変）。
- **rearm（D146-6 / Option A）**: `state.load(acquire)==Building ∧ oblId==O` → settle(true) の CAS へ一元化。Builder 側維持（Building は take CAS 成功=RebuildThread のみ生成、Orchestrator:413 も RebuildThread → 同一スレッド lease 内操作）。

## D146-8 テスト（S-1..S-7、public API のみ）
| T | 検証 | 結果 |
|---|---|---|
| S-1 | take CAS lease 排他（二重 take 失敗）+ payload 正しい | ✅ |
| S-2 | settle(true) CAS で payload 保持・再 take 可 | ✅ |
| S-3 | settle(false) で payload reset・state NoAdmission・hasPending false | ✅ |
| S-4 | 同一 oblId + DurablePending → submit no-op・true・**payload 非上書**（sampleRate drift で検証） | ✅ |
| **S-5** | **同一 oblId + Building → no-op・lease 維持・payload 非上書（D144 Case C 直接回帰）** | ✅ |
| S-6 | 異 oblId + Building → defer（clobber なし・O1 の lease/表現無傷） | ✅ |
| S-7 | discard（join 後単一スレッド）で payload/state/predicate が一貫リセット | ✅ |
既存 40 テスト（X1 coalesce/lease、C11-C16、R18/R20/R21、G43、T-P2、T-P3 含む）は全構成で変らず合格 — **修約 3 の挙動差分（same-oblId 非上書）は既存契約と衝突しないことを実測確認**。

## D146-9 構造再監査（実測 grep）
- **A. plain state access = 0**（全 11 箇所が load/store/CAS に移行、grep 実測）。
- **B. payload 書込は 2 箇所に集約**: `tryAttachDurableRecovery`（NoAdmission acquire 観測後・CAS publish 前）と `resetDurableAdmissionPayload`（settle(false)=lease 内 / discard=join 後）。take は読取のみ。cross-thread conflicting plain access = 0（書込権限の排他は D145 Phase 1-C の権限論証 + CAS linearization）。
- **C. Building 生成 = take CAS 成功のみ**（grep: Building への store は 1190 の CAS 1 箇所。1243/1252 は expected 側=消費）。
- **D. Building 中 payload overwrite = 0**（旧 guard 経路消滅、S-5 が回帰固定）。
- **E. P3 repair の stale combination 消滅**: repair は helper 内 `state.load(acquire)!=NoAdmission → oblId 読`。oblId 書込は (a) CL の pre-publish（NoAdmission 観測時）または (b) lease 内 reset のみで、reset は NoAdmission release に sequenced-before。よって **acquire した state が非 NoAdmission なら oblId 読はその state と対になる値**（D140 §3 の stale strand は生成不能）。same-holder repair / different-holder fallback / repeated redrive / failure→redrive / wake latch は T-P3-1..6 + S-4/S-6 で不変を実証。

## GO 基準判定
```text
P4-2 durable mutation exclusion        ✓（B/C/D 実測 + 権限論証）
P4-3 payload snapshot consistency      ✓（CAS release/acquire + sequenced-before 規約、E）
P4-4 non-HB conflicting plain access   ✓（A/B: state 全 atomic 化、payload 書込 2 権限に集約）
INV-X1-1 exactly-one durable state     ✓（S-1/S-2/S-3 + 既存 lease テスト合格）
D136-C Building overwrite              ✓ eliminated（S-5 直接回帰）
P3 repair stale combination            ✓ eliminated（E + T-P3 群合格）
same-oblId semantic-difference test    ✓ fixed（S-4/S-5 の sampleRate drift 非反映）
shutdown join/discard                  ✓（S-7 + 順序不変・Phase 7 非touch）
existing retry semantics               ✓（既存 40 テスト全構成合格・K=4 不変）
```
**NO-GO 項目なし → D147（RecoveryFailure Event Transport）へ進出可。**

## D147 への引継ぎ事項
1. **容量は未証明として再導出のこと**（D145-0-4 の教訓: 128 は反証済み、512+128 も「≲520」レベルの概算。Phase 0 で最大 event production を実コードから厳密化）。
2. W5（markTransientFailure の delivery=None）は**まだ RebuildThread に存在**（Phase 4 の delivery 集約は D147 のイベント実装と同時に行うまで、本 Gate の範囲外）。現行の delivery plain race は D147 完了まで形式的 UB のまま（AV 無関係・D141 済）。
3. Orchestrator:311/401 の siteId 対応、rearm の settle CAS 化は済（Phase 5 完了）。
4. 残旧コメント: h:318-324（delivery 単一書込者主張）は **D147 実装後に真となる** — それまで更新保留（実態との一致を待つ）。

**STOP — D146 完了。D147 の指示を待つ。P5/P6 非着手。**
