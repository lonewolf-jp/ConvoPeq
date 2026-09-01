# D146 — Phase 2 Atomic-State Implementation + Immediate Structural Audit（Work Report）

**Status: GO（全基準成立）→ D147 進出可。**
**詳細:** `evidence/D146_PHASE2_ATOMIC_STATE_IMPLEMENTATION.md` / **ビルド:** `evidence/d146_ctest.log`

## 実装（I-HS2 の Phase 2 部分のみ・RecoveryFailure ring は未実装）
- **state atomic 化**: `std::atomic<State> state{NoAdmission}`。trivially-copyable 静的検証は削除（reset は明示クリア＋release 順序規約へ）。旧 SPSC/plain 契約コメントを CAS プロトコル基準に更新。
- **take = CAS lease**: `CAS(DurablePending→Building, acquire)` 成功そのものが lease 取得。失敗時は payload を読まない（二段判定廃止）。
- **settle CAS 化**: retry=true=`CAS(Building→DurablePending, release)` / retry=false=**payload reset → CAS(Building→NoAdmission, release) → predicate false**（reset 順序固定、逆順禁止を明文化）。
- **単一 attach primitive `tryAttachDurableRecovery`**: NoAdmission を acquire 観測時のみ payload 書込→CAS publish。占有中は同一 oblId=AlreadyRepresented（no-op・非上書）/ 異 oblId=OccupiedByOther（defer・clobber 禁止）。submit durable 経路と redrive（付着・P3 repair・transport fallback）をこの 1 経路に収束。CAS failure は live 中不能（NoAdmission 唯一退出権限=CL）で rollback 不要。
- **rearm（Option A 維持）**: `load(acquire)==Building ∧ oblId 一致` → settle の CAS へ一元化。同一スレッド lease 内操作。

## 検証
```text
Debug:   full build → CTest 100% tests passed out of 40（DBG_CTEST_EXIT=0）
Release: full build → CTest 100% tests passed out of 40（REL_CTEST_EXIT=0）
Release AV stress: 200 反復 → 0 失敗
ConvoPeq.md 15:32:58 再生成（D146 マーカー 25・同期確認）、TEMP 残留 0
diff: coordinator.cpp +135 / .h +37 / tests +144、RebuildDispatch/Threading 無変更
```
**S-1..S-7 新規**（S-5=同一 oblId+Building の Case C 直接回帰、S-4=sampleRate drift での非上書検証）。既存 40（X1 coalesce/lease、C11-C16、R18/R20/R21、G43、T-P2、T-P3）全構成合格 — 修約 3 の挙動差分は既存契約と衝突しないことを実証。

## 構造再監査（D146-9 実測）
plain state access **0**／payload 書込は helper（NoAdmission-acquire 限定）と reset（lease 内・join 後）の 2 権限に集約／Building 生成は take CAS 1 箇所のみ／Building 中 overwrite 経路消滅／P3 repair の stale combination は「reset は NoAdmission release に sequenced-before」規約で生成不能。GO 基準 9 項目すべて ✓、NO-GO なし。

## D147 引継ぎ
1. 容量（512+128）は**未証明**として Phase 0 で最大 event production を厳密再導出（128 反証の教訓）。
2. **W5（markTransientFailure の delivery=None）はまだ RebuildThread** — delivery 集約（Phase 4）は D147 のイベント実装と同時。現行 delivery plain race は D147 完了まで形式的 UB のまま（AV とは無関係・D141 済）。
3. h:318-324 の delivery 単一書込者コメントは D147 後に真となるため、その間は更新保留。

**STOP — D147（RecoveryFailure Event Transport）の指示を待つ。P5/P6 非着手。**
