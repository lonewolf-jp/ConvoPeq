# D145 Phase 0-1 — I-HS2 Pre-Implementation Re-Derivation & Contract-3 Proof (read-only)

**Date:** 2026-08-31 (+09:00)
**Type:** read-only（Phase 0 + Phase 1）。**Production source changes: 0. Test source changes: 0。**
**基準:** `ConvoPeq.md Generated: 2026-08-31 13:27:23`（D142 後版・D143/D144 と同一、ツリー無変更確認）。
**結論先出し:**
- **Phase 0-4: D144 の 64+64=128 非 drop 証明は反証された**（同一 obligation の複数 transport intent burst）。→ 指示 4 の分岐「そうでなければ ring 容量を再設計」により **512+128+昇格** へ再設計（導出付き）。これは D145-R1 ブロックではない。
- **Phase 1 A-D: すべて実コードから証明成立**（same-oblId no-op は意味論安全、CAS failure は live 中に発生不能）。
- **判定: GO → Phase 2（ただし容量は再設計値を採用）。**

---

## Phase 0-1/2 — durable slot field 別 access・oblId lifecycle 再確認

D143-1 の field 個別表（state/oblId/pending/recoveryGeneration/buildSource/handle/epoch/intentId/reservationOwned × CL/RebuildThread/shutdown）は D142 以降ソース無変更のため現行そのまま有効（行番号同一を git status で確認）。追補として **oblId lifecycle** を実コードで確定:

- 採番: `const LogicalRecoveryObligationId id = ++nextId_`（h:398、tryInsert 内のみ、単一書込者=CL h:451）。
- 不変条件: 「reused slot is always assigned a strictly-greater id (nextId_ is monotonic)」（h:417、C9 ABA テストで実証済）。
- durable slot の `recoveryObligationId` 書込は attach/overwrite（cpp:1006/1167）のみ、reset は構造体リセット（settle false cpp:1266 / discard cpp:1244）のみ。take/settle(true)/rearm は oblId を触らない。
- **帰結: oblId は生涯 1 回の mint 専有（ globally unique・非再利用）**。durable slot に現れる oblId==O は「O の attach/overwrite 時に書かれた値」に他ならない。

## Phase 0-3 — 容量定数の実コード再導出

| 定数 | 実コード | 値 |
|---|---|---|
| L_logical_max | `kMaxLogicalRecoveryObligations`（h:363、INV-CAP-7） | **32** |
| Q（transport） | `kRecoveryIntentQueueCapacity`（h:927、LockFreeRingBuffer） | **256** |
| durable | 単一スロット | **1** |
| L_residency_max | I4_DESIGN_CONTRACT.md:943/984 = Q+1 | **257** |
| K | `kMaxObligationConsecutiveFailures`（h:369） | **4** |
| E_max / O_max | **Phase-I の実コードには存在しない**（episode 語彙は Phase-II deferred、D105-R23/D138-G2 確認。G-0/G-1 の E_max=256×O_max≤32 は episode 導入時の将来値であり現行契約ではない） | n/a |

## Phase 0-4 — 64+64=128 の再検証 → **反証成立・容量再設計**

### 反証（実コード根拠）
D144 の bound 前提「obligation あたり未処理イベント ≤ K=4」は **transport 経路で成立しない**:
- `fillRecoveryQueue`（test:1076-1081）が示すとおり、**同一 {h,target} の submit を 256 回 push でき（C16 の再 push 許容設計、cpp:961-968 NOTE）、すべて同一 oblId の intent として queue に載る**。
- P1 導入後の Builder は pop 毎の build/warmup 失敗で **1 失敗観測=1 イベント** を post する（RebuildDispatch:1006/1033）。CL の次 tick までの間に最大 **Q=256 件** が 1 obligation 分として積み得る。
- さらに publish 失敗イベント（Orchestrator:311）が消費数に比例、durable spin が +4/run。
→ 128 は上界ではない。**反証確定。**

### 再設計（D145 Phase 3 で採用する値）
```text
primary  MpscBoundedRing<RecoveryFailureEvent, 512>
fallback MpscBoundedRing<RecoveryFailureEvent, 128>
両満杯  → HealthEvent/Critical 昇格 + drop counter（silent loss 禁止 INV-5、
          obligation は Live 維持 = safety 不変、liveness は現行 stranding 相当に劣化）
```
導出: 1 tick 窓内の worst-case post 数 ≤ transport burst(256) + publish 失敗(≤消費数 256+4) + durable spin(4×run) ≲ 520 ≤ 512+128=640。昇格経路により**容量証明に依存せず非 silent 化**（防御の二重化）。イベント 24B 想定で計 ~15KB。
**指示の STOP 条件「recovery failure event が drop し得る」への回答**: drop は到達不能な上界+昇格 telemetry で構成し、drop しても obligation は Live+delivery 不変のまま安全側に留まる（失われるのは retry 機会のみ、次 submit/redrive で回復可能）。

## Phase 0-5 / Phase 1 — same-oblId overwrite の実コード意味論と修約 3 適合証明

### 現行意味論（確定）
submit durable 経路（cpp:991-1008）: `state!=NoAdmission ∧ oblId==O` のとき **state 値を問わず** payload 全書換+state:=DurablePending。これが Case C の発生源（D144）。

### A. Building 中 overwrite の消滅
修約 3（CL の payload 書込は `state.load(acquire)==NoAdmission` 観測時のみ）により、`Building` 観測時は書込経路自体に到達しない。**消滅 ✓**

### B. DurablePending 中同一 oblId の非上書
修約 3 で no-op + true 返却。既存表現（slot の payload）が有効なまま。**✓**

### C. NoAdmission のみ新 payload + CAS failure 安全性
```text
state.load(acquire)==NoAdmission → payload 書込 → CAS(NoAdmission→DurablePending, release)
```
- **CAS failure は live 中に発生不能**: NoAdmission から他へ遷移させる書込者は CL の attach のみ（Builder は NoAdmission を生成するだけで消費しない; discard は post-join で CL 停止済）。同一スレッド（CL）の load→store 間に他者遷移が存在しないため、CAS の expected は常に一致。
- 万一の failure（防御コード上）: payload 残骸は NoAdmission 下では誰にも読まれず、次 attach が全面上書 → **無害。retry/rollback 不要**（ただし実装では CAS 失敗時に publish しない分岐を残す）。
- Builder 側の対向遷移（take=CAS(DurablePending→Building)）は publish 後の話で、release/acquire により payload 書込が take の読みに可視。**✓**

### D. 「same O ⇒ same target」の再証明（前提でなく実コードで）
1. **mint 専有**: oblId は tryInsert で `++nextId_` が単一採番（h:398）され、**その瞬間の CoalesceIdentity cid={handle,target} とともに slot へ書かれる**（h:400 identity=key）。単調性（h:417）により oblId は他 identity で再利用されない。
2. **durable への出現**: durable slot の oblId==O は、O を Live として表す submit/redrive が attach 時に書いた値。その submit の cid は findByKey 一致（coalesce）または tryInsert の cid で、**いずれも O の mint 時 identity と一致**（identity は slot 不変・h:346「identity is immutable once admitted」）。
3. **よって同一 O に到達する 2 つの submit は同一 handle + 同一 SemanticRecoveryTarget（5 semantic 値 + buildInputHash 一致）**。buildInput は target 構成要素 ⇒ payload の buildInput は一致。build は convolver 現在値を build 時に再読（cpp:809-811）⇒ recovery 意味論不変。
4. **差分として失われるフィールドの消費確認**:
   - `epoch`: recovery 消費経路で読まれない（RebuildDispatch の recovery セクションに `recovery->epoch` 読取 0 件を grep 実測）。publish の generation は enqueue 時現在値（RebuildDispatch:1035-1036）で durable payload の epoch 非依存。
   - `intentId`: 診断専用（h:240）。
   - `recoveryGeneration`: slot 不変値（G-4.2）で上書いても同一値 ⇒ 実質差分なし。
   - snapshot-level metadata（sampleRate 等）: target 非構成要素・build 時再読。
→ **no-op 化は意味論不変。修約 3 を単純 no-op true として採用可。✓**
（緊張関係の解消: D105-R23 の「episode 不在」は Phase-I の事実であり、本証明は episode に依存せず **oblId の mint 専有性**のみを用いる。CoalesceIdentity は現行実体（h:297-304）で矛盾なし。）

## Phase 0 追補 — 実装時に更新すべき旧コメント/契約記述（洗い出し・今回は未修正）

| 場所 | 旧記述 | 実態（D143/D144） |
|---|---|---|
| h:937-946 | 「SPSC…競合なし」「Producer=CL Consumer=Builder」 | state/oblId/payload に shutdown thread + rearm(CL 外) 接触。I-HS2 で CAS 化後、記述を CAS プロトコル基準に書換 |
| h:968 | 「plain 構造体 — atomic 不要」 | state は atomic 化対象。payload は権限規約で plain 維持 |
| h:318-324 | 「delivery = CoordinatorLoop-only field (written solely on the producer thread)」 | W5 移動後に真となる — 実装後に検証コメントへ更新 |
| h:960-961 | 「rebuildRequestGeneration（coalesce 判定用）」「latest（coalesce で更新）」 | G-4.2 で陳腐化（D139-T-R 観察）。overwrite 禁止で「coalesce で更新」は消滅 |
| cpp:880-881 | 「SPSC-safe for … pendingRecoveryAdmission_」 | CAS プロトコル記述へ更新 |
| cpp:1059-1063 | markTransientFailure caller 列挙コメント | イベント経路の説明へ更新（Phase 3 時） |
| h:485-489 | rearm「documented single-slot limitation」 | I-HS2 下での lease 内操作として更新（Phase 5 時） |

## 判定

```text
Phase 0:
  0-1/2 ✓（oblId lifecycle 確定）
  0-3   ✓（L=32/Q=256/K=4/257 再導出、E_max/O_max は Phase-I 実体なしと明記）
  0-4   ✗→再設計（128 反証、512+128+昇格を採用値として固定）
  0-5   ✓（現行意味論確定）
Phase 1:
  A ✓ / B ✓ / C ✓（CAS failure live 中不能・残骸無害）/ D ✓（mint 専有性で証明、前提不採用）

STOP 条件チェック: 該当なし（drop は昇格で非 silent、1:1 posting 不変、shutdown 順序不変、
  P3 repair は Phase 6 で再証明予定、RT/ISR 非接触）

→ GO: Phase 2（state atomic 化）へ進んでよい。採用容量 = primary 512 / fallback 128 / 昇格。
```

**本ターンの指示範囲（Phase 0 → Phase 1）完了。実装 0。STOP — Phase 2 以降の指示を待つ。**
