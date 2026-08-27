# D103 — Semantic Supersession Implementation-Readiness Audit (Audit 1-6)

- **実施日**: 2026-08-26
- **作業種別**: read-only / audit-only — production source 0 changes / test source 0 changes / CMake 0 changes / contract 0 changes
- **対象ソース**: `C:\VSC_Project\ConvoPeq` (ConvoPeq.md 2026-08-26 20:06:47 + src/ tree)
- **契約基準**: `doc/work88/I4_DESIGN_CONTRACT.md` (D12-D17, D18) ＋ `doc/work88/I1_DESIGN_REVIEW.md` ＋ `doc/work88/I2_DESIGN_FIX.md` ＋ `doc/work88/I3_DESIGN_CONTRACT.md`
- **ツール**: AiDex query / serena find_symbol / grep / read_file (context-mode 沿用)
- **判定**: **Phase I 実装 NO-GO（D9/D12 未解決）** — 6 audits すべてを実行した結果、`canSupersede()` / `isSemanticSuperset()` / `RecoveryEpisodeId` / `SemanticRecoveryTarget` は**いずれも現行ソースに未実装**であり、D12の4条件のうち**3/4がMISSING**、1/4（same handle）がPARTIAL（`DSPHandle::operator==` にて部分実装）。

---

## 0. Audit Scope

6つの監査を実行（読み取り専用・コード変更なし）:

| Audit | 内容 | I4 対応 |
|-------|------|---------|
| Audit 1 | 現行 supersession 実装の完全追跡 | D12.4 |
| Audit 2 | D12 の4条件→現行型のマッピング | D12/D13/D16 |
| Audit 3 | target containment の実装可能性 | D12.2/D18.2 |
| Audit 4 | D16 RecoveryGeneration 検証 | D16 |
| Audit 5 | D15 ownership conservation 接続 | D15/D18.3 |
| Audit 6 | D9 の最終判定 | D9/D12/D18 |

---

## Audit 1 — 現行 supersession 実装の完全追跡

### 1.1 検索対象キーワード

`canSupersede`, `isSemanticSuperset`, `isSemanticTargetSuperset`, `isDomainSuperset`, `supersede`, `superseded`, `SupersessionDecision`, `SemanticRecoveryTarget`, `RecoveryEpisodeId`, `LogicalRecoveryIdentity`, `RecoveryProvenance`, `admittedLogicalObligationCount`

### 1.2 検索結果サマリー (AiDex query / grep)

| Keyword | Production Match | Location | Status |
|---------|-----------------|----------|--------|
| `canSupersede` | **0** | — | MISSING (design-only: I2_D4.md:146, I3_D4'.md:144, I4_D12.4) |
| `isSemanticSuperset` | **0** | — | MISSING (design-only: I4_D12.2) |
| `isSemanticTargetSuperset` | **0** | — | MISSING (design-only: I4_D12.2) |
| `isDomainSuperset` | **0** | — | MISSING (design-only: I4_D12.2) |
| `supersede` | **1** | `RuntimePublicationState.h:14` (`SupersededDiscard`) | PARTIAL (DiscardReason enum only) |
| `superseded` | **1** | `RuntimePublicationState.h:14` (`SupersededDiscard`) | PARTIAL (DiscardReason enum only) |
| `SupersessionDecision` | **0** | — | MISSING (design-only: I2_D4, I3_D4'/D5', I4_D12) |
| `SemanticRecoveryTarget` | **0** | — | MISSING (design-only: I4_D12.2) |
| `RecoveryEpisodeId` | **0** | — | MISSING (design-only: I4_D13.1) |
| `LogicalRecoveryIdentity` | **0** | — | MISSING (design-only: I2_D1, I3_D5') |
| `RecoveryProvenance` | **0** | — | MISSING (design-only: I3_D5') |
| `admittedLogicalObligationCount` | **0** | — | MISSING (design-only: I4_D15.2) |
| `successCount` | **4** | `invariant_INV3_INV5.cpp:827,831,833,840` | TEST-ONLY |
| `supersededCount` | **0** | — | MISSING |
| `shutdownDiscardCount` | **2** | `ISRRuntimePublicationCoordinator.cpp` (as `recoveryShutdownDiscardCount_`) | PARTIAL |

### 1.3 Causal chain: caller → supersession decision → obligation disposition → ownership/count updates

**The causal chain is NOT connected.** The current code has:

1. **Caller**: `QuarantineIntentHandler::handle()` (`ISRRuntimePublicationCoordinator_ProcessIntent.cpp:111`) → calls `ctx.engine.submitRecoveryIntent()` (`AudioEngine.h:4405`)
2. **Admission**: `submitRecoveryIntent()` → `submitRecoveryRequest()` (`ISRRuntimePublicationCoordinator.cpp:812`) — performs **transport enqueue → fallback to single-slot durable admission** (no supersession decision)
3. **Disposition**: Builder Loop (`AudioEngine.RebuildDispatch.cpp:922`) pops via `popRecoveryRequest()` or `takePendingRecoveryAdmission()`
4. **No supersession decision point**: The `submitRecoveryRequest()` code path (`ISRRuntimePublicationCoordinator.cpp:812-920`) performs:
   - shutdown gate check
   - `RecoveryIntent` construction with `quarantinedHandle`, `epoch`, `intentId`, `buildSource`
   - `pendingIntentCount_.fetchAdd(1)` (reservation)
   - `recoveryIntentQueue_.push()` (transport)
   - **ON PUSH FAILURE**: rollback + `pendingRecoveryAdmission_` overwrite (single-slot, **no coalesce check, no supersession check, no identity comparison**)

### 1.4 RecoveryIdentity 現状

現行の `RecoveryIntent` struct (`ISRRuntimePublicationCoordinator.h:216-228`):

```cpp
struct RecoveryIntent {
    DSPHandle handle;
    PublicationEpoch epoch;
    uint64_t intentId;
    convo::RuntimeBuildSnapshot buildSource;
};
```

- `handle` = `DSPHandle{slot, generation}` — handle 同一性は `operator==` で判定可能（`slot + generation`）。
- `epoch` = `PublicationEpoch` — **これは EBREpoch / PublicationSequence ドメイン**、I4-D13で指摘される `RecoveryEpisodeId` とは異なる。
- `intentId` = `nextRecoveryIntentId_++` — 単調増加シーケンス番号（**RecoveryGenerationではない**）。
- `buildSource` = `RuntimeBuildSnapshot` — sealed + `isRuntimeBuildSnapshotSealedAndCompatible()` で比較可能。

**結論**: `RecoveryIntent` は `LogicalRecoveryIdentity`（handle + RecoveryGeneration + semanticIdentity + epoch）の概念を持たない。D12.4の"same handle"は `DSPHandle::operator==` で部分実装されているが、"same recovery episode" / "newer RecoveryGeneration" / "semantic target containment" は**すべべて未実装**。

### 1.5 現行の admission ロジックの根本的欠陥

`PendingRecoveryAdmission` (`ISRRuntimePublicationCoordinator.h:664-682`):

```cpp
struct PendingRecoveryAdmission {
    enum class State : uint8_t { NoAdmission, DurablePending, Building };
    State state = State::NoAdmission;
    bool pending = false;
    uint64_t recoveryGeneration = 0;
    convo::RuntimeBuildSnapshot buildSource{};
    bool reservationOwned = false;
    DSPHandle handle{};
    PublicationEpoch epoch{0};
    uint64_t intentId{0};
};
```

**問題**: `recoveryGeneration` フィールドは `intent.intentId`（単調カウンタ）をそのまま代入している（`ISRRuntimePublicationCoordinator.cpp:867`。D16で定義される `RecoveryGeneration` arithmetic contract を満たさない — modular比較未対応）。

**coalesce**: コメントに「coalesce: 単一スロット — 既存 durable があれば最新で上書き」とあるが、**実際の上書きロジックは `submitRecoveryRequest()` 内のフィールド代入のみ** — handle/epoch/intentId の一致確認なし、semantic target の一致確認なし、**Blind overwrite**（D18.1 Step 0 の CoalesceIdentity 検索未実装）。

---

## Audit 2 — D12 の4条件→現行型のマッピング

### 2.1 D12.4 canSupersede() 4条件

```
canSupersede(newer, older) =
      same handle                （DifferentHandle）
    + same recovery episode      （DifferentEpisode）        ← RecoveryEpisodeId（D13）
    + newer RecoveryGeneration   （isAfter・RecoveryGeneration ドメイン・D16）
    + semantic target containment（isSemanticSuperset・D12.2 → NotSemanticSuperset）
```

### 2.2 マッピング結果

| D12 Condition | I4 Design Type | 現行実装 | 判定 | 理由 |
|---------------|----------------|----------|------|------|
| **1. same handle** | `LogicalRecoveryIdentity.handle == newer.handle` | `DSPHandle::operator==` (slot + generation) | **PARTIAL** | `DSPHandle` は存在・比較可能だが、`RecoveryIntent.handle` は `quarantinedHandle` をそのまま保持するのみ — supersession決定の入力としては未接続（canSupersede自体未実装） |
| **2. same recovery episode** | `RecoveryEpisodeId` (D13.1 新設) | **なし** | **MISSING** | `RecoveryEpisodeId` は D13.1 で新設された概念。現行ソースでは `Epoch`/`generation` が乱用されている（D13.2: "epochをsupersessionのgateに使用しない"違反）。quarantine episode の識別子は存在しない |
| **3. newer RecoveryGeneration** | `isAfter(newer.gen, older.gen)` (D16) | `PendingRecoveryAdmission.recoveryGeneration = intent.intentId` | **MISSING** | `recoveryGeneration` フィールドは `nextRecoveryIntentId_`（push/popシーケンス）をそのまま代入。`RecoveryGeneration` ドメインの dedicated counter が存在しない。`SequenceArithmetic.h` の `isAfter` は `PublicationSequence` / `PublicationEpoch` ドメインでのみ使用されており、`RecoveryGeneration` ドメインの**独立証明**未実施（D16: "Phase H の PublicationSequence 検証を流用しない"） |
| **4. semantic target containment** | `isSemanticSuperset()` (D12.2) | `isRuntimeBuildSnapshotSealedAndCompatible()` | **MISSING** | `isRuntimeBuildSnapshotSealedAndCompatible()` (`RuntimeBuildTypes.h:301`) はフィールド-byフィールド等価性チェックだが、`isSemanticTargetSuperset()` とは異なる概念（equalityではなくsuperset）。`buildInputHash`（D12.2のSemanticRecoveryTarget構造体のフィールド）は**存在しない** (`buildInputHash` grep: 0 hits)。`RuntimeBuildFingerprint` は `irIdentityHash`/`convolutionConfigHash`/`dspParameterHash` を持つが、`SemanticRecoveryTarget`構造体は存在しない |

### 2.3 "PublicationSequence / EBREpoch / ActivationEpoch を代用品として採用してはいけません" の検証

D13.2: "supersession の同一 lineage 判定に `epoch` を使わない。`newer.epoch == older.epoch` 条件を**廃止**し、**`RecoveryEpisodeId`**（明示的な lineage identifier）に置き換える。"

現行コードの `RecoveryIntent.epoch` は `currentPublicationEpoch()` から取得される — これは **PublicationEpoch**（= `lastCommittedRuntimeGeneration_` または publication epoch）であり、D13.1 で定義される `RecoveryEpisodeId` とは異なるドメイン。**PublicationEpoch は EBR/lifetime ordering用、RecoveryEpisodeId は config lineage 用**（D13.1）。

**判定**: **CONFLICT** — `epoch` が `RecoveryEpisodeId` の代用として誤用されている可能性あり（D13.2違反）。

---

## Audit 3 — target containment の実装可能性

### 3.1 D12.2 SemanticRecoveryTarget 構造体

```cpp
struct SemanticRecoveryTarget {
    ObligationDomains domainCoverage;
    std::uint64_t irIdentityHash;
    std::uint64_t convolutionConfigHash;
    std::uint64_t convolverFingerprint;
    std::uint64_t dspParameterHash;
    std::uint64_t buildInputHash;
};
```

### 3.2 フィールド実装可能性調査

| Field | 現行ソースでの取得可能性 | 判定 | 根拠 |
|-------|--------------------------|------|------|
| `irIdentityHash` | 可能 | `RuntimeBuildTypes.h:41` `RuntimeBuildFingerprint.irIdentityHash` | `captureRuntimeBuildSnapshot()` (`AudioEngine.RebuildDispatch.cpp:97`) で設定済み |
| `convolutionConfigHash` | 可能 | `RuntimeBuildTypes.h:42` `RuntimeBuildFingerprint.convolutionConfigHash` | `AudioEngine.RebuildDispatch.cpp:98` で `convolverSnapshot.fingerprint` を設定 |
| `convolverFingerprint` | 可能 | `RuntimeBuildTypes.h:52` / `RuntimeBuildSnapshot.convolverFingerprint` | `RuntimeBuildTypes.h:333` で `snapshot.convolverFingerprint` 比較対象 |
| `dspParameterHash` | 可能 | `RuntimeBuildTypes.h:43` `RuntimeBuildFingerprint.dspParameterHash` | `finalizeRuntimeBuildSnapshot()` (`AudioEngine.RebuildDispatch.cpp:139`) で FNV-1aハッシュ算出済み |
| `buildInputHash` | 不可能 | — | **0 hits** — `BuildInput` 構造体は個別フィールドの等価性比較のみ（`isRuntimeBuildSnapshotSealedAndCompatible()`）。ハッシュ値としての `buildInputHash` は**存在しない** |

### 3.3 isSemanticTargetSuperset の実装可能性

D12.2 で定義される `isSemanticTargetSuperset` は **equality (Phase I 採用)** — つまり5つのハッシュ値の全値等価性:

```cpp
bool isSemanticTargetSuperset(const SemanticRecoveryTarget& n, const SemanticRecoveryTarget& o) noexcept {
    return n.irIdentityHash == o.irIdentityHash
        && n.convolutionConfigHash == o.convolutionConfigHash
        && n.convolverFingerprint == o.convolverFingerprint
        && n.dspParameterHash == o.dspParameterHash
        && n.buildInputHash == o.buildInputHash;
}
```

**実装可能性**: **PARTIAL** (4/5フィールド取得可能)

- `RuntimeBuildFingerprint{irIdentityHash, convolutionConfigHash, dspParameterHash}` + `RuntimeBuildSnapshot.convolverFingerprint` はすべて `buildSource`（`RuntimeBuildSnapshot`）から取得可能。
- `buildInputHash` は**存在しない**。`BuildInput` のハッシュ化は未実装（個別フィールド比較のみ）。
- D18.2: "Phase I では semantic containment == exact SemanticRecoveryTarget 全値等価（isSemanticTargetSuperset）" とあり、equality なので `buildInputHash` は `buildInput` の等価性判定で代替可能だが、**D12.2 のインターフェース契約（`buildInputHash`フィールド）を満たさない**。

---

## Audit 4 — D16 RecoveryGeneration

### 4.1 D16 arithmetic contract

```
RecoveryGeneration arithmetic:
  - equality:      uint64 等価
  - strict-after:  isAfter(a,b) = isBefore(b,a)（SequenceArithmetic・modular < 2^63）
  - wraparound:    modulo 2^64（kSeqHalfModulus）
  - zero-init:     RecoveryGeneration{0} = 未割当/なし を予約
  - exhaustion:    alloc が 0 を生成しない
```

### 4.2 現行実装

**RecoveryGeneration を扱う場所**:

1. `AudioEngine.RebuildDispatch.cpp:994-997` (`rebuildThreadLoop`):
```cpp
const int recoveryGeneration =
    convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
auto recoverySnapshot = recovery->buildSource;
recoverySnapshot.generation = recoveryGeneration;
recoverySnapshot.sealed = true;
enqueuePublicationIntentForRuntimeCommit(dspToCommit, recoveryGeneration, recoverySnapshot);
```

2. `AudioEngine.RebuildDispatch.cpp:1066-1072` (retry path, 同じロジック):
```cpp
const int recoveryGeneration =
    convo::consumeAtomic(rebuildRequestGeneration, std::memory_order_acquire);
recoverySnapshot.generation = recoveryGeneration;
enqueuePublicationIntentForRuntimeCommit(...);
```

3. `ISRRuntimePublicationCoordinator.cpp:867`:
```cpp
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;
```

4. `ISRRuntimePublicationCoordinator.h:675` (comment):
```cpp
uint64_t recoveryGeneration = 0;  // 入ってきた時点の rebuildRequestGeneration（coalesce 判定用）
```

### 4.3 判定

| D16 Aspect | 現行実装 | 判定 | 理由 |
|------------|----------|------|------|
| **Dedicated counter** | `rebuildRequestGeneration` (AudioEngine.h:2552) | **PARTIAL** | `rebuildRequestGeneration` は `int` 型（D16は `uint64_t`要求）。また `rebuildRequestGeneration` は**rebuild request**の世代であり、`RecoveryGeneration` ドメインの独立カウンタではない |
| **SequenceArithmetic適用** | `SequenceArithmetic.h` | **PARTIAL** | `isAfter` 等は `PublicationSequence`/`PublicationEpoch` ドメインでのみ使用（`ISRRuntimePublicationCoordinator.cpp:95-103` monotonicity check）。`RecoveryGeneration` ドメインの**独立証明**未実施（D16: "Phase H の PublicationSequence 検証を流用しない"） |
| **equality** | `==` operator | **CLOSED** | `RuntimeBuildSnapshot.generation`（`int`）の等価性は比較可能 |
| **strict-after** | `isRebuildObsolete` (`generation != rebuildRequestGeneration`) | **MISSING** | `isRebuildObsolete` は **equality** チェック（`!=`）のみ — **strict-after（isAfter）未実装**。`rebuildRequestGeneration` の monotonic decrease はないが、D16の `isAfter`（modular comparison）は未使用 |
| **wraparound** | N/A | **MISSING** | `int` 型カウンタ、wraparound コントラクト未定義 |
| **zero-init** | `recoveryGeneration = 0` (default) | **PARTIAL** | `PendingRecoveryAdmission.recoveryGeneration` は default 0（`uint64_t`）。しかし `rebuildRequestGeneration` は `int` で 0 から開始。D16の "zero-init = 未割当" の予約セマンティクスは未文書化 |
| **exhaustion** | N/A | **MISSING** | 0回避ロジック未実装 |
| **retry invariant** (`Building → fail → DurablePending → Building`) | `settlePendingRecoveryAdmission(true)` (`ISRRuntimePublicationCoordinator.cpp:962-966`) | **PARTIAL** | retry は構造的に保証されるが、`recoveryGeneration` は `intent.intentId` で上書かれる（D16で定義される `RecoveryGeneration` とは異なる型・意味） |

**結論**: D16 arithmetic contract は**未実装**。現行の `recoveryGeneration` は `intentId`（push/popシーケンス）の別名にすぎない。`RecoveryGeneration` ドメインの dedicated counter + SequenceArithmetic適用 + zero-init/exhaustion コントラクトは**すべてMISSING**。

---

## Audit 5 — D15 ownership conservation

### 5.1 D15.2 ownership conservation equation

```
liveOwnershipCount      = transportCount + durableCount + buildingCount + stalledCount
terminalDispositionCount = successCount + supersededCount + shutdownDiscardCount

admittedLogicalObligationCount
    = liveOwnershipCount + terminalDispositionCount
```

### 5.2 現行実装の disposition

| D15 disposition | 現行実装 | live/terminal | カウンタ存在 |
|----------------|----------|---------------|-------------|
| `transportCount` | `pendingIntentCount_` (`ISRRuntimePublicationCoordinator.cpp:884`, `popRecoveryRequest:947`) | live | **(fetchAdd on push, fetchSub on pop)** |
| `durableCount` | `hasPendingRecoveryAdmission()` (single-slot boolean) | live | **PARTIAL — 0/1 boolean, not a count** |
| `buildingCount` | `pendingRecoveryAdmission_.state == Building` | live | **PARTIAL — single-slot state enum** |
| `stalledCount` | `RuntimeHealthMonitor.cpp:89` (`entry.stalledCount`) | NOT connected | **MISSING — Recovery stall counter ではない (HealthMonitor の stalled recovery エピソード数)** |
| `successCount` | `invariant_INV3_INV5.cpp:827` (TEST-ONLY) | terminal | **MISSING (production)** |
| `supersededCount` | **0 hits** | terminal | **MISSING** |
| `shutdownDiscardCount` | `recoveryShutdownDiscardCount_` (`ISRRuntimePublicationCoordinator.cpp:585, 935, 948`) | terminal | **CLOSED** |
| `admittedLogicalObligationCount` | **0 hits** | — | **MISSING** |

### 5.3 現行の counter architecture

`RuntimeIntentCoordinator` は以下のatomic counterを持つ:

```cpp
std::atomic<uint64_t> pendingIntentCount_;      // transport reservation
std::atomic<uint64_t> recoveryIntentDropCount_; // queue-full diagnostic
std::atomic<uint64_t> recoveryShutdownDiscardCount_; // ShutdownDiscard
std::atomic<bool> recoveryAdmissionPending_;    // single-slot durable
PendingRecoveryAdmission pendingRecoveryAdmission_; // single-slot struct
```

- `pendingIntentCount_` は "transport residency + producer reservation" を追跡（`ISRRuntimePublicationCoordinator.h:250` コメント）。
- `recoveryShutdownDiscardCount_` は shutdown discard のみ記録。
- **coalesce** は single-slot overwrite（`submitRecoveryRequest` で `pendingRecoveryAdmission_` フィールドを上書き） — **coalesce count も不存在**。
- **superseded** は**完全に実装されていない** — `RecoveryOutcome` は `HealthMonitor` のみで使用され、`SupersededDiscard` は `DiscardReason` enumの値だが**決定ロジックがない**。

### 5.4 判定

現行の counter architecture は **pre-D15**（D11.5 refinement 前）の構造:
- `admittedLogicalObligationCount` は**存在しない**。
- `supersededCount` は**存在しない**。
- `successCount` は**test-only**（`invariant_INV3_INV5.cpp`）。
- D15.2 conservation equation を検証するcounterが **3/7**が存在（`pendingIntentCount_`, `recoveryShutdownDiscardCount_`, `recoveryAdmissionPending_` as 0/1 boolean）。
- D18.3 (user指摘4, NO-GO修正)の `liveOwnershipCount`/`terminalDispositionCount` 分離は未実装 — 現行は `pendingIntentCount_` が transport reservation + publication intent residency を混在させている。

---

## Audit 6 — D9 final judgment

### 6.1 D101-27/28/30, D102-C3/C4 との関連

D102-C3-C4-Remaining-Gates-Inventory-Audit.md:85,111,155,163 によれば:

| D102-C3/C4 Gate | 判定 |
|-----------------|------|
| D9: `canSupersede()` semantic target containment | **OPEN** — I4 2026-08-15 の最重要 NO-GO |
| D102 numerical gates (R_required, R_cap, M_scope) | CLOSED (D102-C2) |

**重要**: D102 numerical GO は **Phase I 実装 GO とは無関係**（D102は T1/M-boundの測定統計量、D9はsupersession implementation gate）。

### 6.2 I4.D12-D17/D18 判定ステータス

| Design Item | I4 ステータス | 現行ソース | Audit結果 |
|-------------|---------------|------------|-----------|
| D12 (Semantic Supremum Contract) | CLOSED (Design-5) | **未実装** | **MISSING** |
| D12.4 canSupersede() | CLOSED (Design-5) | **未実装** | **MISSING** |
| D13 (RecoveryEpisodeId) | CLOSED (Design-5) | **未実装** | **MISSING** |
| D14 (reservation-first / backpressure) | CLOSED (Design-5) | **PARTIAL** | `pendingIntentCount_` は reservation-firstだが backpressure/STALL未実装 |
| D15 (ownership conservation) | CLOSED (Design-5) | **未実装** | **MISSING** |
| D16 (RecoveryGeneration arithmetic) | CLOSED (Design-5) | **PARTIAL** | `rebuildRequestGeneration` is int, not uint64; isAfter/strict-after未実装 |
| D17 (baseline ownership) | CLOSED (Design-5) | **PARTIAL** | `buildSource` は値コピーだが RecoveryEpisodeId/baselineIdentity未実装 |
| D18.1 (CoalesceIdentity) | CLOSED (Design-5) | **PARTIAL** | single-slot overwriteのみ、CoalesceIdentity検索未実装 |
| D18.2 (Phase I semantic supersession) | CLOSED (Design-5) | **未実装** | **MISSING** |
| D18.3 (ownership conservation修正式) | CLOSED (Design-5) | **未実装** | **MISSING** |

### 6.3 D9 final judgment table

| Prerequisite (D12-D17) | I4 Status | Source Status | Implementation Readiness |
|------------------------|-----------|---------------|------------------------|
| D12: `canSupersede()` (4 conditions) | CLOSED | **MISSING** (0/4 implemented) | **NO-GO** |
| D12: `isSemanticSuperset()` / `isSemanticTargetSuperset()` | CLOSED | **MISSING** (equality only via `isRuntimeBuildSnapshotSealedAndCompatible`) | **NO-GO** |
| D12: `SemanticRecoveryTarget` struct | CLOSED | **MISSING** (no `buildInputHash`) | **NO-GO** |
| D13: `RecoveryEpisodeId` (lineage) | CLOSED | **MISSING** (epoch used instead) | **NO-GO** |
| D14: reservation-first + backpressure + STALL | CLOSED | **PARTIAL** (reservation exists, backpressure/STALL NO-GO) | **NO-GO** |
| D15: ownership conservation equation | CLOSED | **MISSING** (admittedLogicalObligationCount/supersededCount absent) | **NO-GO** |
| D16: RecoveryGeneration arithmetic | CLOSED | **PARTIAL** (int counter, no isAfter, no wraparound contract) | **NO-GO** |
| D17: baseline ownership / episode lifecycle | CLOSED | **PARTIAL** (buildSource copy only, no episode lifecycle) | **NO-GO** |
| D18.1: CoalesceIdentity search | CLOSED | **PARTIAL** (blind single-slot overwrite, no identity check) | **NO-GO** |
| D18.2: Phase I semantic target containment | CLOSED | **MISSING** (no `canSupersede`, no `isSemanticTargetSuperset`) | **NO-GO** |

### 6.4 Phase I Implementation Readiness Verdict

| 項目 | 判定 |
|------|------|
| **Phase I 実装 readiness** | **NO-GO** |
| **肯定的判定条件 (D9 sufficient condition)** | `canSupersede()` / `isSemanticSuperset()` / `SemanticRecoveryTarget` / `RecoveryEpisodeId` の**すべて**が実装されるまで |
| **現状 (実装済み)** | 0/10 prerequisites CLOSED |
| **Phase I NO-GO 根拠** | D12 D13 D15 D16 D17 D18.1 D18.2 — すべての supersession 関連設計が **design-only（実装前固定）** であり、現行コードは pre-D9/pre-D12 の single-slot overwrite モデル |
| **D102 numerical GO との関係** | **無関係** — D102 は T1/M-bound measurement statistics。D9 は supersession implementation gate |
| **次工程** | (1) D18-D26 の design → Code 実装 (R1-R17) → (2) T17-T21 テスト → (3) Phase I GO |

---

## 付録 A — ソースコード参照 (grep/AiDex 結果)

### A.1 主要ソースファイル

| File | Role | Relevant Lines |
|------|------|----------------|
| `src/audioengine/ISRRuntimePublicationCoordinator.h` | RecoveryIntent, PendingRecoveryAdmission, submitRecoveryRequest | 216-228 (RecoveryIntent), 664-682 (PendingRecoveryAdmission), 247 (submitRecoveryRequest decl) |
| `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | submitRecoveryRequest impl, takePendingRecoveryAdmission, settlePendingRecoveryAdmission | 812-920 (submitRecoveryRequest), 925-958 (takePendingRecoveryAdmission), 960-967 (settle) |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | rebuildThreadLoop, recoveryGeneration, captureRuntimeBuildSnapshot | 85-110 (capture/finalize/seal), 990-1000 (recoveryGeneration), 1060-1075 (retry path) |
| `src/audioengine/RuntimeBuildTypes.h` | RuntimeBuildSnapshot, RuntimeBuildFingerprint, isRuntimeBuildSnapshotSealedAndCompatible | 35-62 (structs), 301-337 (comparison) |
| `src/audioengine/ISRDSPHandle.h` | DSPHandle (slot + generation), operator== | 29-50 |
| `src/audioengine/ISRDSPQuarantine.h` | QuarantineEntry (slot, generation, reason, epoch) | 23-37 |
| `src/audioengine/ISRRuntimeSemanticSchema.h` | PublicationSemantic, PublicationEpoch, PublicationSequenceId | 197-236 |
| `src/audioengine/SequenceArithmetic.h` | isAfter/isBefore (for PublicationSequence/Epoch only) | 1-50 |
| `src/audioengine/AudioEngine.h` | rebuildRequestGeneration, currentBuildSnapshot_, submitRecoveryIntent | 2552-2553, 4405-4440 |
| `src/audioengine/ISRRuntimePublicationCoordinator_ProcessIntent.cpp` | QuarantineIntentHandler, RecoveryIntentHandler | 109-170 |
| `src/audioengine/RuntimeWorldAuthority.h` | publish (commit metadata bake), PublishMetadata | 195-255 |

### A.2 Design documents (canSupersede / isSemanticSuperset 定義箇所)

| File | Section | Content |
|------|---------|---------|
| `doc/work88/I4_DESIGN_CONTRACT.md` | D12.2 | `SemanticRecoveryTarget` struct, `isDomainSuperset`, `isSemanticTargetSuperset`, `isSemanticSuperset` |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D12.4 | `canSupersede(newer, older)` 4 conditions |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D13.1 | `RecoveryEpisodeId` 7ドメイン分離 |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D13.2 | epoch ≠ RecoveryEpisodeId |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D14-D17 | reservation-first, ownership conservation, RecoveryGeneration arithmetic, baseline ownership |
| `doc/work88/I4_DESIGN_CONTRACT.md` | D18 | CoalesceIdentity, Phase I semantic supersession (equality), ownership conservation fix |
| `doc/work88/I1_DESIGN_REVIEW.md` | §1.2.2, R3 | `SupersessionDecision` enum, `canSupersede()` |
| `doc/work88/I2_DESIGN_FIX.md` | D4 | `canSupersede()` / `SupersessionDecision` / `isSemanticSuperset` |
| `doc/work88/I3_DESIGN_CONTRACT.md` | D4', D5' | `canSupersede()` 改訂 (epoch → RecoveryEpisodeId), bounded durable table |

---

## 付録 B — 判定理由の詳細

### B.1 canSupersede が未実装である理由

1. **`SupersessionDecision` enum**: grep結果0 hits in `src/` (design-only: `I2_DESIGN_FIX.md:135`)
2. **`canSupersede()` function**: grep結果0 hits in `src/` (design-only: `I2_DESIGN_FIX.md:146`, `I3_DESIGN_CONTRACT.md:144`)
3. **現行 `submitRecoveryRequest()`**: `ISRRuntimePublicationCoordinator.cpp:884-920` — queue full時に `pendingRecoveryAdmission_` フィールドを **Blind overwrite** するだけ。ハンドル比較・epoch比較・generation比較・semantic target比較 **すべて実施しない**。
4. **REPAIR_PLAN2-dash2.md:315**: "pendingRecoveryAdmission_（h:590）は single durable slot" — 明記されているように single-slot構造であり、複数logical obligationを保持するbounded durable table（I3_D5'/D5'で提案）への拡張は **未実装**。

### B.2 RecoveryEpisodeId が未実装である理由

1. **I4-D13.1**: `RecoveryEpisodeId` は **新設概念**（I4で D8.4 → D13.1 で定義）。現行ソースには**0 hits**。
2. **D13.2**: "epochをsupersessionのgateに使用しない" — 現行 `RecoveryIntent.epoch` は `currentPublicationEpoch()` から取得される `PublicationEpoch`（EBR/lifetime ordering）。これを `RecoveryEpisodeId`（config lineage）の代用に使用していると D13.2違反。
3. **QuarantineEpisodeの概念**: `QuarantineEntry` (`ISRDSPQuarantine.h:23`) は `slot`/`generation`/`reason`/`quarantineEpoch`/`quarantineTimestampUs` を持つが、**recovery episode identity** は持たない。`quarantineEpoch` は timestamp（us単位）であり lineage identifierではない。

### B.3 ownership conservation が未実装である理由

1. **`admittedLogicalObligationCount`**: 0 hits in `src/`
2. **`supersededCount`**: 0 hits in `src/`
3. **`successCount`**: test-only (`invariant_INV3_INV5.cpp:827`)
4. **D15.2 conservation equation**（`admittedLogicalObligationCount == liveOwnershipCount + terminalDispositionCount`）を検証するcounterが **3/7**しか存在しない（`pendingIntentCount_`, `recoveryShutdownDiscardCount_`, `recoveryAdmissionPending_` as 0/1 boolean）。
5. **D18.3 (user NO-GO修正)**: `liveOwnershipCount`/`terminalDispositionCount` の分離は未実装 — 現行は `pendingIntentCount_` が transport reservation + publication intent residency を混在させている。

---

## 結論

**Phase I 実装 NO-GO** — I4_DESIGN_CONTRACT.md D9/D12-D17/D18 の supersession contract は**design-closed**だが、現行ソースコード（`src/`）では**すべて未実装**（0/10 prerequisites CLOSED）。

Audit 1-6 の詳細なソースコード照合結果により、以下が確認された:

1. **canSupersede() / SupersessionDecision**: 0実装 — design-only（I1/I2/I3/I4）
2. **D12 4条件**: 3/4 MISSING（same recovery episode, newer RecoveryGeneration, semantic target containment）、1/4 PARTIAL（same handle = DSPHandle::operator==）
3. **RecoveryEpisodeId**: 0実装 — epoch が代用として誤用されている可能性あり（D13.2違反）
4. **RecoveryGeneration**: `rebuildRequestGeneration`（int）が `intentId` と混同されている — D16 arithmetic contract 未満
5. **SemanticRecoveryTarget**: `buildInputHash` フィールド欠落 — 4/5フィールドのみ取得可能
6. **ownership conservation**: D15.2/D18.3のconservation equation検証カウンタ 3/7のみ存在

D102-C3/C4の数値GO（R_required=4121, R_cap=5120, M_scope=4120）は**T1 telemetry / M-bound measurement**に関するものであり、D9（supersession implementation gate）とは**無関係**。D9が未解決（NO-GO）の間、Phase I実装は開始できない。
