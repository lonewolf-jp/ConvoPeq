# D155 — Recovery Transport / Coalesce 現状再監査（read-only）

**Date:** 2026-09-01 (+09:00)
**Type:** 完全 read-only 監査。**Production source: 0 変更 / Test source: 0 変更 / CMake: 0 / build: 0 / CTest: 0。**
**Baseline:** `ConvoPeq.md` `Generated: 2026-08-31 23:39:12`（T3c Close Audit と同一・D154-F2 適用後最新）。実コード追跡は現行 `src/audioengine/ISRRuntimePublicationCoordinator.{h,cpp}` + 呼び出し経路で実施。
**前提:** T3c = CLOSED（evidence/T3C_CLOSE_AUDIT.md）。lifecycle ownership domain は変更禁止領域。

---

## 0. 総合判定（先出し）

> ## **A. NO CHANGE / DEFER**

現在の SPSC topology は構造的に成立しており、R1（MPSC 化）の必要性は発生していない。Coalesce は Phase-I として既に実装・監査済み（G-4.1→G-4.3 チェーン）であり、REPAIR_PLAN2-dash2 の「将来対応」記述は 8 月前半のスナップショット時点のもので stale。残る将来項目は Phase-II Supersession（D18 で設計凍結済み）のみで、現行意味論（equality-only containment）の下では不要。

---

## 1. D155-C（先に確定）: T3c との境界

```text
T3c CLOSED（evidence/T3C_CLOSE_AUDIT.md）
   │
   ├── lifecycle ownership domain = RecoveryLifecycleWord 16B
   │     （state / pending / adjudicated / delivery を full-word CAS で commit、load は advisory）
   │     → 変更禁止領域（本監査でも 0 触れ）
   │
   └── recovery transport / admission = 本監査の対象
         recoveryIntentQueue_（SPSC ring）/ pendingIntentCount_（reservation accounting）/
         pendingRecoveryAdmission_（durable slot）/ redrive / coalesce admission protocol
```

- R1 / coalesce のいかなる将来変更も、**lifecycle CAS 層の再設計を含んではならない**。T3c の full-word CAS（tryInsert Live 公開 / coalesce identity CAS / resolve terminal / casDelivery）は transport の producer 構成と独立しており、R1 は transport 層のみを対象とする。
- 本監査は lifecycle domain への変更を **0 行**行った（read-only）。

---

## 2. D155-A: R1（recoveryIntentQueue_ MPSC 化）の再判定

### 2.1 recoveryIntentQueue_ の現状

| 項目 | 実測 |
|---|---|
| 型 / capacity | `LockFreeRingBuffer<RecoveryIntent, kRecoveryIntentQueueCapacity=256>`（h:1061-1062）— SPSC ring |
| push サイト | **2 件のみ**（cpp:991 submitRecoveryRequest / cpp:1257 redriveDeferredRecoveryObligations） |
| pop サイト | **1 件**（cpp:1433 popRecoveryRequest 本体） |
| producer | **CoordinatorLoop のみ**。submitRecoveryRequest の唯一 production 呼び出し元 = `AudioEngine::submitRecoveryIntent`（h:4486）→ 呼び出し元は `QuarantineIntentHandler::handle`（ProcessIntent.cpp:140/169）= CL の processIntent phase（Threading.cpp:263）。redrive は runCoordinatorPhase（Threading:278・CL）+ submitRecoveryRequest 内部（CL）からのみ |
| consumer | **RebuildThread のみ**。popRecoveryRequest の唯一 production 呼び出し元 = `rebuildThreadLoop`（RebuildDispatch.cpp:980）。 shutdown 破棄（`discardRecoveryRequestsOnShutdown` cpp:1452）は stopRebuildThread join 後の単一スレッド（RebuildDispatch:810） |
| RT / MessageThread / Timer | **接触 0 件** — `AudioEngine.Timer.cpp` の recovery API 呼び出し = **0**（RT affinity audit と整合） |

### 2.2 pendingIntentCount_（reservation accounting）

| サイト | 意味論 |
|---|---|
| fetchAdd（cpp:990 recovery / 1256 redrive / 633 quarantine） | **reservation-before-push**（push 前に +1。INV-5: 静かな消失禁止） |
| fetchSub（cpp:997 recovery / 1262 redrive） | push 失敗時の **rollback** |
| fetchSub（cpp:1440 popRecoveryRequest） | pop 成功時の reservation 消費（pop 成功 ⇒ 対応 reservation 存在が保証 — underflow 不変条件） |
| fetchSub（quarantine fallback 経路 cpp:652） | lane 移動時の相殺（recovery と quarantine で counter 共有） |
| `setPendingIntentCount`（h:176） | **TEST-ONLY** — production 絶対値上書きは P2-1 §1.1.5 で廃止済み（Commit/Threading/ProcessIntent に廃止コメント、残存呼び出しは tests のみ） |
| shutdown/drain | `isFullyDrained` は `recoveryIntentQueue_.size()==0 && pendingIntentCount_==0` を source of truth の一部（cpp:525/532）+ discardRecoveryRequestsOnShutdown が残留 pop（fetchSub 経由で counter 整合） |

**residency 意味論**: 1 push = 1 reservation、pop/discard で消費、push 失敗で rollback — カウンタは「transport residency + producer reservation」であり INV-ISR-02（二重計上禁止）と整合。

### 2.3 pendingRecoveryAdmission_（durable single slot）

- writer: `tryAttachDurableRecovery`（CL・唯一の publish 前書込経路 h:1079）/ `takePendingRecoveryAdmission` + `settlePendingRecoveryAdmission`（RebuildThread） / `resetDurableAdmissionPayload`（遷移権限保持者）。
- **state は atomic CAS プロトコル（D146）で既に保護** — plain read/write 禁止・payload は書込権限規約（NoAdmission 観測時のみ pre-CAS 書込）。
- durable slot は SPSC queue とは独立した単一 slot フォールバックであり、**R1（queue の MPSC 化）と同時変更する必要はない**。writer 集合 {CL, RebuildThread} は R1 でも不変。

### 2.4 SPSC invariant の成立判定

**成立（構造的に証明）**: producer = CL 単一（全 push 経路の呼び出しグラフが CL に収束）、consumer = RebuildThread 単一（+ join 後 shutdown 単一スレッド）。ソース明記の SPSC 契約（h:603 / cpp:881 / cpp:1187「Called only from the CoordinatorLoop (producer) thread — SPSC-safe」）と実呼び出しグラフが一致。

### 2.5 R1 のトリガ条件と変更量（将来参照用）

- **トリガ条件**: 「将来 Timer 等から直接 submit/pop を呼ぶ」— **現時点で未発生**（Timer.cpp の recovery API 呼び出し 0 件実測）。
- MPSC 化した場合の変更対象（既存設計どおり queue 交換だけでは不十分）:
  1. `LockFreeRingBuffer` SPSC → MPSC 交換 + producer-hole 契約（REPAIR_PLAN2-dash §1.3 / F10）
  2. `pendingIntentCount_` reservation invariant の MPSC 下再検証（fetchAdd/fetchSub 自体は atomic だが、reservation–push–consume の直列化前提の注释・不変条件を再証明）
  3. shutdown proof 再設計（isFullyDrained の queue emptiness + producer 停止順序固定 + discard 経路）
  4. SPSC-safe 契約コメント（h:603 / cpp:881 / 1187）と関連テスト（単一 producer 前提の C10/redrive 系）の再監査
  5. telemetry（recoveryIntentDropCount 等）の意味論再確認
- いずれも lifecycle domain には触れない。

---

## 3. D155-B: Recovery coalesce の再判定

### 3.1 現行実装の確認（実コード）

| 項目 | 実測 |
|---|---|
| `CoalesceIdentity` | `{quarantinedHandle, SemanticRecoveryTarget}` — 等価には **handle と target の両方**が必要（MUST-3 / R8: (H1,T)≠(H2,T) → 2 obligations）（h:324-337） |
| `SemanticRecoveryTarget` | 5 semantic values の `operator==`（irIdentityHash / convolutionConfigHash / convolverFingerprint / dspParameterHash / buildInputHash）+ domainCoverage は必要条件 metadata（等価比較から除外）（h:295-322） |
| obligation identity | obligationId = 単調採番（tryInsert `++nextId_`）・identity は W の obligationId フィールド（T3c） |
| `recoveryGeneration` | 専用単調 counter（`nextRecoveryGeneration_`、G-4.2）— intentId / BuildGeneration と分離・durable slot にも thread 済み |
| `buildSource` | payload（W 外 plain）— tryInsert の pre-CAS 書込・durable slot は tryAttachDurableRecovery の pre-CAS 書込（権限規約） |
| `PublicationEpoch` | caller 供給 payload（Phase G R7 — coordinator は currentWorld を参照しない） |
| transport residency | `W.delivery`（T3c lifecycle word 内・None/Transport/Durable）— coalesce attach は `casDelivery` の full-word CAS のみ |
| coalesce linearization | G-4.3: findByKey → **full-word identity CAS（Live→Live）**・win=COALESCE（ΔL=0, oblId 再利用）/ lose(terminal)=tryInsert NEW。CAS 後 slot 変異 0（A5 解消済み） |
| terminal → resubmit | **新規 obligation 再承認（NEW）** — T7 semantic。ST-1 で競合下実証済み（F-2） |
| durable slot overwrite | tryAttachDurableRecovery は CAS NoAdmission 観測時のみ payload 書込 — 無条件上書きなし。redrive は delivery==None の Live のみ（二重配送禁止 R10-3/C16） |
| `Superseded` | **未実装を確認** — `ResolvedSuperseded` は宣言のみ（h:341 dormant）・cpp 内遷移 = **0 件**（grep 実測）。Phase-II 項目として意図的に凍結 |
| 同一 handle・異なる target | **区別される** — CoalesceIdentity が handle+target 両方を要求（C4 テストで実証）。単純な「same handle → 最新で上書き」は実装不可能な構造 |

### 3.2 REPAIR_PLAN2-dash2 記述の鮮度判定

- dash2 の「Recovery coalesce は将来対応（別設計）」は **stale**。当時の設計（`lastRecoveryHandle_` による連続同一 handle 判定・A→B→A 不可の P3）は現行ソースに **残存 0 件**（`lastRecoveryHandle_` grep = 0）であり、G-4.1（6-field target 導入）→ G-4.2（identity wiring）→ G-4.3（CAS-linearized coalesce）→ G-4.3-R/T-R（audit + regression 12/12 + CTest 40/40）で Phase-I coalesce は**既に実装・監査・ストレス検証済み**（ST-1 の T5 coalesce storm + exhausted 厳密計上）。
- 残る将来項目は **Phase-II Supersession**（D18 で設計凍結: canSupersede = same handle + same RecoveryEpisodeId + newer generation + isSemanticSuperset）のみ。現行意味論（equality-only containment）は健全であり、ST-1 で実証済みのため、Phase-II を今着手する根拠はない。

---

## 4. 判定ルールへの適合

| 選択肢 | 適合性 |
|---|---|
| **A. NO CHANGE / DEFER** | **該当** — SPSC invariant 成立・R1 トリガ未発生・coalesce Phase-I 完了・durable slot は R1 と独立 |
| B. R1 IMPLEMENTATION CANDIDATE | 不該当 — 複数 producer は存在しない（実測）・SPSC invariant は崩れていない |
| C. COALESCE DESIGN AUDIT REQUIRED | 不該当 — coalesce は既に設計・実装・監査済み（G-4.x チェーン）。Phase-II Supersession は D18 設計凍結中で、着手根拠なし |
| D. BLOCKED | 不該当 — 安全性問題は不存在 |

**判定: A. NO CHANGE / DEFER。** R1 の実装前設計・coalesce 設計監査への着手条件は下記のとおり記録し、着手は保留。

### 着手条件（将来参照用）

| 将来変更 | 起きたら |
|---|---|
| Timer 等の第 2 producer が recovery transport を直接呼ぶ | R1 実装前設計（§2.5 の変更対象 5 項目を含む）を開始 |
| Supersession（A→B 上書き）が製品要件になる | Phase-II 設計監査（D18 規範: RecoveryEpisodeId 必須化 + canSupersede + durable table 拡張）を開始 |

## 5. 監査手順の記録

grep 実測: recoveryIntentQueue_（decl h:1062 + push ×2 / pop ×1）/ pendingIntentCount_（14 サイト全列挙）/ pendingRecoveryAdmission_（writer 関数特定: tryAttach/take/settle/reset）/ submitRecoveryRequest 呼び出しグラフ（AudioEngine.h:4486 ← ProcessIntent:140/169 ← CL）/ popRecoveryRequest（RebuildDispatch:980 ← rebuildThreadLoop）/ Timer.cpp recovery 呼び出し 0 件 / `lastRecoveryHandle_` 0 件 / `ResolvedSuperseded` 遷移 0 件 / `setPendingIntentCount` = TEST-ONLY（h:176）。

**D155 = 判定確定: A. NO CHANGE / DEFER。実装・ソース変更は 0。**
