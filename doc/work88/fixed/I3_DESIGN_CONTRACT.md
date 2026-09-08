# I3 Design Contract — Phase I-Design-3（semantic contract review）

- 日付: 2026-08-15
- 判定: **Phase I-Design-2 は HOLD（D3/D4 の意味論が未固定のため）。Design-3 で D8〜D10 を追加し、D3/D4/D5/D6 を改訂。実装は NO-GO 継続。**
  （→ **Design-4 確定** は下記 D11。→ **Design-5 再固定** は `doc/work88/I4_DESIGN_CONTRACT.md`。）
- ⚠️ **Design-5 反映（2026-08-15）**: `doc/work88/I4_DESIGN_CONTRACT.md`（正本）により、**D9（domain-based
  supersession）を NO-GO として再設計**（SemanticRecoveryTarget = domain coverage + target containment に分離）。
  **D11.3（capacity 24）/ D11.5（ownership conservation）/ D11.1（RecoveryGeneration arithmetic）/
  D11.2（baseline ownership）も改訂**（I4 の D13〜D17）。**実装時は I4 を正本とすること**（本ドキュメントの
  D9 の「ObligationDomains 集合包含 = supersession」は**廃止** — 必要条件であり十分条件でないため）。
- 前提: `doc/work88/I1_DESIGN_REVIEW.md`（10項目突合）・`doc/work88/I2_DESIGN_FIX.md`（D1〜D7 設計固定・D3/D4 に意味論欠陥）。
- 本ドキュメントの目的: **「実装可能な精度」と「意味論の正しさ」は別**（ユーザー指摘）。D3/D4 の `changedDomains` 単体導出と generation/epoch の semantic domain 未分離を解消し、実装者が誤った supersession semantics をコードに固定しないための semantic contract を定める。
- コード裏付け: `RuntimeBuildTypes.h`（RuntimeBuildFingerprint:38 / RuntimeBuildSnapshot:48）/ `RebuildDispatch.cpp:982-985`（build 時 generation 代入）/ `ISRRuntimePublicationCoordinator.cpp:855-925`（submitRecoveryRequest）

---

## 0. Design-2 のレビュー指摘（ユーザー）と対応方針

| # | 指摘 | 対応 |
|---|------|------|
| 1 | `changedDomains` を snapshot 単体から導出 → superset 判定は基準状態がなく不安定 | **D9** で義務ドメイン集合を固定 baseline に対する diff として再定義（admission 時に確定・不変）+ truth table 明文化 |
| 2 | generation の semantic domain 未分離（RecoveryGeneration / BuildGeneration / PublicationSequence / Epoch） | **D8** でドメイン分離 + RecoveryGeneration の lifetime rule 固定 |
| 3 | `epoch` と `PublicationSequence` の混同。canSupersede の same epoch domain の意味を明記 | **D8.4** で各 serial の順序付け対象を明記。SequenceArithmetic 適用は各ドメインで別途証明 |
| 4 | D6: supersession ≠ Builder lease cancellation の境界 | **D6'** で invariant 明記 |
| 5 | STALL の transport-level semantics（incoming の所有者・retry trigger・shutdown 時） | **D10** で ownership contract 固定 |

---

## D8 — RecoveryGeneration semantic contract（R1 の generation を確定）

### D8.1 semantic domain 分離（同型でも別ドメイン）
| 名前 | 順序付け対象 | 生成タイミング | logical identity の構成要素か |
|------|-------------|----------------|------------------------------|
| `PublicationSequence`（PublicationSequenceId） | published RuntimeWorld の順序 | publish 毎（fetch_add） | ❌（world 順序） |
| `Epoch`（PublicationEpoch） | **lifetime / EBR の reclamation ordering + activation epoch の anchor** | publish/transition 毎 | ⚠️ 同一世代判定（D4' 条件2）に使用 |
| `RecoveryGeneration`（新設） | **recovery/build request lineage の順序** | **admission 時（submitRecoveryRequest 受理点）** | ✅（core） |
| `BuildGeneration` | 実際に build された RuntimeWorld の generation | build 完了時（rebuildRequestGeneration から） | ❌（world 属性） |

> **ユーザー指摘の核心**: 数値型が同じ（uint64_t）でも semantic domain は別。`SequenceArithmetic<uint64_t>` を使えることと、`isAfter(recoveryGenerationA, recoveryGenerationB)` が正しいことは**別個に証明が必要**。Phase H の semantic-domain 検証（PublicationSequence ドメイン）を Phase I にそのまま流用してはならない。

### D8.2 RecoveryGeneration の定義（確定）
- **意味**: recovery logical obligation の lineage を順序付ける serial number。同一 handle の recovery obligation 間で「どちらが新しい obligation か」を決める。
- **ソース（推奨・決定待ち A）**: Coordinator に新設する単調カウンタ `nextRecoveryGeneration_`（`fetch_add(1, relaxed)`）を **admission 時**（submitRecoveryRequest が obligation を受理した点 = transport push 成功 または durable admit 成功 または stall 受理）に割り当てる。`intentId`（診断 sequence）とは**別カウンタ**。
- **lifetime rule（ユーザー要件・明記）**:
  > RecoveryGeneration は **Recovery obligation が admission された時点で確定**し、**その後 build によって変更されない**。coalesce（同一 logical identity の buildSource 更新）でも不変。
- **非使用**: intentId（診断のみ）・rebuildRequestGeneration（build 時の BuildGeneration）・PublicationSequence。
- **monotonicity 証明（SequenceArithmetic 適用条件）**: `nextRecoveryGeneration_` は単一 Producer（CoordinatorLoop）の単調カウンタ → 同一ドメイン内で strict increase。`isAfter(a,b)` の適用は RecoveryGeneration ドメインで独立に正当（Phase H の PublicationSequence 検証を流用しない）。

### D8.3 BuildGeneration（RecoveryGeneration と分離）
- build 完了時、`RebuildDispatch.cpp:982-985` が `consumeAtomic(rebuildRequestGeneration)` を読んで
  `recoverySnapshot.generation = recoveryGeneration` と代入 — これは **BuildGeneration**（実際に build された world の generation）。
- **BuildGeneration は logical identity に含めない**。`RuntimeWorld::generation` に反映される world 属性。
- ⚠️ **現行の混同**: 現在は `buildSource.generation` が admission 時と build 時の両方に使われている（submit 時は未確定、build 時は BuildGeneration で上書き）。**RecoveryGeneration を別フィールドとして導入し、buildSource.generation（BuildGeneration）と分離する**。RecoveryIntent / PendingRecoveryAdmission に `recoveryGeneration` フィールドを新設（buildSource.generation とは別）。

### D8.4 Epoch の意味（混同解消）
- identity の `epoch` = **activation epoch**（handle H を含む最後の published world の publication.epoch — quarantine 時に確定）。lifetime/EBR の reclamation ordering の anchor。
- canSupersede の `same epoch domain`（D4' 条件2）= **同一 activation epoch の config 世代**（同一エラ/同一 lineage）。別 epoch = 別 config 世代 → coalesce/supersede 不可（R9）。
- `epoch` に SequenceArithmetic を適用する場合も、それが **lifetime/EBR ordering の serial** であることを明示し、そのドメインで正当性を証明する（PublicationSequence の検証を流用しない）。

---

## D9 — Semantic supersession contract（R3・changedDomains 導出の修正）

### D9.1 指摘の確認（ユーザー）
> 「どの domain が変更されたか」は本質的に**2つの semantic state の差分**。snapshot 単体から
> `deriveChangedDomains(snapshot)` を作る場合、基準状態がない。
> `isSemanticSuperset(newer, older) = newer.changedDomains ⊇ older.changedDomains` は変更集合の定義が不安定。

例:
```
A = {IR=A, EQ=1}   B = {IR=B, EQ=1}   C = {IR=B, EQ=2}
A → B : IR changed / B → C : EQ changed / A → C : IR + EQ changed
```
→ 「何に対して diff したか」が未定義だと変更集合が一意にならない。

### D9.2 修正: ObligationDomains を「固定 baseline に対する diff」として定義
- `SemanticBuildIdentity`（Recovery が build する semantic fingerprint・`RuntimeBuildTypes.h:38/333-336` を基盤）:
  ```cpp
  struct SemanticBuildIdentity {
      std::uint64_t irIdentityHash;         // rebuildFingerprint.irIdentityHash        （IR ドメイン）
      std::uint64_t convolutionConfigHash;  // rebuildFingerprint.convolutionConfigHash （Conv ドメイン）
      std::uint64_t dspParameterHash;       // rebuildFingerprint.dspParameterHash      （EQ ドメイン）
      std::uint64_t convolverFingerprint;   // snapshot.convolverFingerprint            （Conv ドメイン）
      std::uint64_t buildInputHash;         // BuildInput の canonical hash             （Config ドメイン）
  };
  ```
- **`ObligationDomains`（changedDomains を改名・基準を固定）**:
  ```cpp
  enum ObligationDomain : uint32_t {
      ObligationNone   = 0,
      ObligationIR     = 1u << 0,
      ObligationConv   = 1u << 1,
      ObligationEQ     = 1u << 2,
      ObligationConfig = 1u << 3,
  };
  ```
  導出（**admission 時に一度だけ計算・以後不変**）:
  ```cpp
  // baseline = handle H の quarantine エピソード開始時点の authoritative published world の
  //            SemanticBuildIdentity（RuntimeStore::current から取得・単一 source — Phase G 整合）
  ObligationDomains deriveObligationDomains(const SemanticBuildIdentity& recovery,
                                            const SemanticBuildIdentity& baseline) noexcept
  {
      ObligationDomains d = ObligationNone;
      if (recovery.irIdentityHash         != baseline.irIdentityHash)         d |= ObligationIR;
      if (recovery.convolutionConfigHash  != baseline.convolutionConfigHash
          || recovery.convolverFingerprint != baseline.convolverFingerprint)  d |= ObligationConv;
      if (recovery.dspParameterHash       != baseline.dspParameterHash)       d |= ObligationEQ;
      if (recovery.buildInputHash         != baseline.buildInputHash)         d |= ObligationConfig;
      return d;
  }
  ```
  - **基準の固定が本質**: `deriveChangedDomains(snapshot)` ではなく、**エピソード開始 baseline に対する diff**。同一 handle の全 recovery obligation が**同じ baseline** を共有するため、変更集合の定義が一意・安定になる。
  - **不変条件**: ObligationDomains は admission 時に確定し、義務の lifetime 中は**不変**（build により再計算しない）。
  - ⚠️ **実装前検証項目**: quarantine エピソード開始 baseline の捕捉（quarantine record に baseline SemanticBuildIdentity を保持する小さな追加が必要）。

### D9.3 isSemanticSuperset の truth table（明文化・決定）
- **意味**: `isSemanticSuperset(newer, older)` = newer の Recovery obligation が older の obligation を**完全に包含**（older が担当する全 semantic domain を newer が担当）するか。**ObligationDomains の set inclusion** で判定（D9.2 の固定 baseline により定義が安定）。
  ```cpp
  // newer.obligationDomains ⊇ older.obligationDomains
  bool isSemanticSuperset(const SemanticBuildIdentity& newer, const SemanticBuildIdentity& older) noexcept {
      return (older.obligationDomains & ~newer.obligationDomains) == 0;
  }
  ```
- **明示 truth table**（実際の build semantics に基づく）:
  ```
  older \ newer      { }       {IR}     {EQ}     {IR,EQ}
  { }                ✅        ✅       ✅       ✅     （空義務 = 全て包含）
  {IR}               ✗         ✅       ✗        ✅     （IR+EQ は IR を包含）
  {EQ}               ✗         ✗        ✅       ✅     （IR+EQ は EQ を包含）
  {IR,EQ}            ✗         ✗        ✗        ✅     （同一のみ包含）
  ```
  - **EQ change does NOT supersede IR change**（{EQ} ⊉ {IR}）✓
  - **IR change does NOT supersede EQ change**（{IR} ⊉ {EQ}）✓
  - **IR+EQ may supersede IR**（{IR,EQ} ⊇ {IR}）✓ / **may supersede EQ**（{IR,EQ} ⊇ {EQ}）✓
  - **identical semantic identity supersedes identical older obligation**（{IR,EQ} ⊇ {IR,EQ}）✓（coalesce と等価）
- **Generation が新しいだけでは supersede しない**（ユーザー要件）: ObligationDomains の包含が無い限り `NotSemanticSuperset`。EQ 変更を IR 変更で supersede しない。

---

## D4' — canSupersede() 改訂（D8/D9 反映）

```cpp
SupersessionDecision canSupersede(const LogicalRecoveryIdentity& newer,
                                  const LogicalRecoveryIdentity& older) noexcept
{
    if (newer.handle != older.handle)                 return DifferentHandle;         // R8
    if (newer.epoch   != older.epoch)                 return NotSameGenerationDomain; // R9（同 config 世代）
    if (!isAfter(newer.recoveryGeneration, older.recoveryGeneration))
        return NotSameGenerationDomain;                                             // 新 lineage でない
    if (!isSemanticSuperset(newer.semanticIdentity, older.semanticIdentity))
        return NotSemanticSuperset;                                                 // R10（D9 truth table）
    return CanSupersede;
}
```
- 条件1: same handle（R8）
- 条件2: same epoch domain = same activation epoch（同 config 世代・R9）
- 条件3: `isAfter(newer.RecoveryGeneration, older.RecoveryGeneration)` — RecoveryGeneration ドメインの
  monotonic serial（D8・独自証明）
- 条件4: `isSemanticSuperset`（D9 truth table・R10）
- `isAfter` の適用は RecoveryGeneration ドメインで正当（Phase H の PublicationSequence 検証を流用しない）。

---

## D5' — bounded durable table + STALL ownership（D10 反映）

### D5'.1 構造（I2 D5 維持）
```cpp
static constexpr size_t kMaxDurableRecoveryAdmissions = 16;  // 暫定採用（レビュー可能）
struct DurableRecoverySlot {
    LogicalRecoveryIdentity identity;            // {handle, RecoveryGeneration, semanticIdentity, epoch}
    ObligationDomains obligationDomains;         // D9.2（admission 時に確定・不変）
    RecoveryProvenance provenance;               // 診断のみ
    convo::RuntimeBuildSnapshot buildSource;     // latest（coalesce で更新・RecoveryGeneration 不変）
    PendingRecoveryAdmission::State state;       // NoAdmission / DurablePending / Building
    bool reservationOwned;                       // INV-X1-5
    uint64_t intentId;                           // 診断のみ
};
std::array<DurableRecoverySlot, kMaxDurableRecoveryAdmissions> durableTable_{};
```

### D5'.2 挿入アルゴリズム（D5 維持 + stall 所有権明示）
```
admitDurable(incoming):
 1. 同一 LogicalRecoveryIdentity → COALESCE（buildSource 更新・RecoveryGeneration/reservation 不変）
 2. canSupersede(incoming, slot) == CanSupersede → SUPERSEED（置換）
 3. 空 slot → INSERT
 4. full → STALL（D10）
```

### D10 — STALL ownership contract（固定・決定待ち B）
- **incoming obligation の所有者**: STALL 中は **CoordinatorLoop が所有**する bounded stall set
  `recoveryStall_`（capacity = kMaxDurableRecoveryAdmissions と同数・ObligationDomains 確定済みで保持）。
  所有権は「transport queue / durable table / stall set」の**ちょうど1つ**に常に属する（INV-X1-6 拡張:
  double-owner 禁止）。queue は transport-only のまま（STALL 分を queue に戻さない）。
- **再試行 trigger**: CoordinatorLoop が各サイクル（Builder settle で slot が空いた契機・または既存の
  wake 経路）で `admitDurable` を stall 順（FIFO）に再試行。slot が空いた時点で INSERT される。
- **shutdown 中に retry が来た場合**: RecoveryAdmissionClosed（`state == ShuttingDown`）確定後は、
  既存の `discardPendingRecoveryAdmission` / `recoveryShutdownDiscardCount_` 経路で **ShutdownDiscard** として
  明示破棄（観測可能・silent loss でない・INV-5 整合）。stall set も同様に drain して discard 記録。
- **不変条件（ユーザー要件・明記）**:
  > table full で coalesce/supersede/insert 不能でも、**incoming obligation は絶対に drop しない**。
  > **非 superseded obligation は絶対に evict しない**（P0-7 構造的排除）。
- **telemetry**: `recoveryDurableOverflowCount_`（stall 発生回数）・`recoveryStallResidentCount_`（滞留数）。
  Debug `assert(false)`（構造的不変条件違反の観測可能化）。

---

## D6' — Building 中 supersession（RC-11）+ lease cancellation 境界

### D6'.1 遷移（I2 D6 維持）
```
Slot A: Building（lease 済み・immutable）   +   Slot B: DurablePending（superseding・canSupersede(B,A)）
A success        → A obligation cleared・B remains pending → 次 take で B が Building
A transient fail → A returns DurablePending・B remains pending（B=newer 優先で次 take）
```

### D6'.2 境界 invariant（ユーザー指摘・明記）
> **Supersession は durable obligation の admission policy であり、既に取得された Builder lease の
> cancellation authority ではない。** `canSupersede(B, A)` が成立しても **A の Building lease を取り消さない**
> （in-flight build は A の buildSource で完了させる）。B は table の別 slot に DurablePending として保持され、
> A の settle 後に昇格する。

---

## 実装 GO 条件（Design-3 反映）

### 決定待ち（レビューで確定 → 実装 GO）
- **A. RecoveryGeneration のソース**: 専用カウンタ `nextRecoveryGeneration_`（推奨・admission 時 fetch_add）を採用するか。
- **B. STALL ownership**: CoordinatorLoop 所有の bounded stall set（推奨・capacity = kMax・FIFO 再試行）を採用するか。
- **C. quarantine エピソード baseline の捕捉方法**: quarantine record に baseline SemanticBuildIdentity を保持する
  小さな追加（推奨）を採用するか。`ObligationDomains` はこの固定 baseline に対する diff で確定。

### 実装時の必須変更（D8/D9/D10 反映）
- `ISRRuntimePublicationCoordinator.h`: `nextRecoveryGeneration_`・`RecoveryGeneration` フィールド追加 /
  `SemanticBuildIdentity` / `ObligationDomains`（baseline diff 導出）/ `recoveryStall_`（bounded stall set）/
  single slot → table 化。
- `ISRRuntimePublicationCoordinator.cpp`: `cpp:905-915` の latest-wins 撤去 → admitDurable（coalesce→supersede→insert→stall）。
  RecoveryGeneration 割当（admission 時）。ObligationDomains 導出（baseline diff・admission 時 1 回）。
- `AudioEngine.RebuildDispatch.cpp`: BuildGeneration と RecoveryGeneration の分離（buildSource.generation を
  BuildGeneration として維持・identity は不変）。
- quarantine 記録: baseline SemanticBuildIdentity 捕捉（決定待ち C）。

### 正式判定
**Phase I 実装 NO-GO（維持）。Design-3 で D8〜D10 を固定した。決定待ち A/B/C を確認後、実装 GO。**
（→ **Design-4 確定** は下記 D11 参照。Design-3: CONDITIONAL GO を確定し、implementation-ready 化。）

---

## D11 — Design-4 確定（2026-08-15・ユーザー判定: A=GO / B=refinement 必須 / C=GO）

ユーザー最終判定（Design-3: CONDITIONAL GO / Implementation: NO-GO）に基づき、実装前残件3つを固定。
**Design-4 を最終 implementation-ready contract とする（実装はユーザー GO 後に開始）。**

### D11.1 A — RecoveryGeneration（GO・D8 確定）
D8.2 を補強し、producer/allocation/lifetime/identity の契約を固定:
```
RecoveryGeneration
    producer:   Recovery admission authority（Coordinator・submitRecoveryRequest 受理点）
    allocation: admission creation（専用カウンタ nextRecoveryGeneration_ の fetch_add(1)）
    lifetime:   logical recovery obligation と同一（obligation 存続中は値が不変）
    identity:   LogicalRecoveryIdentity に含む
    intentId:   identity に使用しない（別カウンタ・診断のみ）
    PublicationSequence / BuildGeneration / EBR Epoch: 別 domain（D8.1）
```
- ★ **retry-invariant（最重要・ユーザー要件）**: **build retry で RecoveryGeneration は変化しない**。
  ```
  R17: DurablePending → Building → transient fail → DurablePending → Building
  ⇒ R17.generation == constant
  ```
  retry が新しい logical recovery と誤認されると supersession semantics が壊れるため、
  RecoveryGeneration は **admission 時に確定し、以後（coalesce / retry / supersede 判定含む）不変**。

### D11.2 C — quarantine episode baseline（GO・D9 確定）
- ★ **episode baseline は episode 開始時に capture し、episode に属する全 Recovery admission が
  同一 baseline を参照する（immutable）**。
  ```
  RecoveryAdmission
      └── baselineIdentity   （episode 開始時の authoritative published world identity・不変）
  ```
- admission 側に **immutable 値コピー**で保持する（episode authority への mutable 参照はしない —
  lifetime 安全性のため）。
- ⚠️ **禁止**: mutable な current RuntimeWorld（RuntimeStore::current）を後から参照して
  diff を再計算する設計。ObligationDomains は admission 時に baseline と 1 回だけ diff して確定・不変。
- 反例（ユーザー）: A accepted → published→S1 → B の baseline を S1 とすると A:{IR}, B:{EQ} になり
  supersession 判定が変わる。**baseline は episode 全体で S0 のまま固定**。

### D11.3 B — STALL ownership + 総 logical obligation 容量（refinement・D10 確定）
★ **「stall set capacity = kMax」だけでは P0-7 を解決しない**（16 durable + 16 stalled = 32 で
  S17 を保持不能 — ユーザー指摘）。**logical obligation の総容量を単一 bounded resource として定義**する。
```
Admission authority
    ├── durable table             kMaxDurableRecoveryAdmissions  = 16（暫定）
    └── pending admission queue   kMaxPendingRecoveryAdmissions  =  8（暫定）
単一 bound: kMaxLogicalRecoveryObligations = 24（= 16 + 8・暫定）
不変: durableOccupancy + stalledOccupancy ≤ kMaxLogicalRecoveryObligations
```
- budget 残がある限り、full durable table でも **pending admission queue に保持**（drop しない）。
- budget 枯渇（残 0）で coalesce/supersede 不能 → **明示的 terminal-failure policy**（INV-X1-7）:
  `recoveryObligationOverflowCount_` + Debug `assert(false)`（観測可能・silent でない）。既存 obligation は不 touch。
- 容量根拠: DSPHandleTable（512）内の実同時 quarantined 数に比して十分大きい（構造的非到達）。

### D11.4 INV-X1-7 — Logical Obligation Preservation（Phase I 最重要 invariant）
```
For every admitted Recovery obligation:

    exactly one of
        Transport / DurablePending / Building / Stalled
    owns the obligation.

No transition may:
    duplicate the logical obligation,
    silently discard it,
    overwrite a non-supersedable obligation,
    or cancel a Building lease through supersession.

A logical obligation may disappear only by:
    Success,
    explicit Superseded decision,
    ShutdownDiscard,
    or an explicitly observable terminal failure policy.
```
- D6'「supersession ≠ Builder lease cancellation」は本 invariant の **Building-lease 節**として昇格。
- `kMaxLogicalRecoveryObligations`（単一 bounded resource）と本 invariant が **P0-7 の構造的排除**を保証。

### D11.5 ownership conservation（テスト用）
```
logicalAdmissionCount == 1 per LogicalRecoveryIdentity（INV-X1-5）
transportCount + durableCount + buildingCount + stalledCount + terminalCount
    == 発行総数（monotone・各遷移後に保存）
```
→ P0-7（latest-wins による obligation 喪失）の検出能力を大幅に向上。D7 テストに追加。

### D11.6 D7 テスト追加（Design-4）
| # | テスト | 検証 |
|---|--------|------|
| T13 | retry 不変性 | Building → fail → DurablePending → Building で RecoveryGeneration 不変 |
| T14 | baseline immutability | published が S1 に進んでも B の baseline は S0（{IR,EQ} 判定が不変） |
| T15 | ownership conservation | 各遷移後に transport/durable/building/stalled/terminal の和 == 発行総数 |
| T16 | budget 枯渇 | 全 24 占有で coalesce/supersede 不能 → terminal-failure（観測可能）・既存不 evict |

### D11.7 最終決定待ち（値のみ・レビュー可能）
- `kMaxDurableRecoveryAdmissions` / `kMaxPendingRecoveryAdmissions` / `kMaxLogicalRecoveryObligations`
  （候補: 16 / 8 / 24・暫定採用）。構造（A/B/C + INV-X1-7）は確定（GO）。

### Design-4 最終判定
**Design-4 = final implementation-ready contract（Design-3 の CONDITIONAL GO を確定）。**
残件3つ（RecoveryGeneration 専用 counter / baseline immutable ownership / 総容量 + ownership conservation）を固定。
実装順序: `LogicalRecoveryIdentity → Supersession → bounded admission → RC-11 → invariant tests`。
**実装 GO はユーザー最終確認後。**
