# I2 Design Fix — Phase I-Design-2（Recovery coalesce 設計固定）

- 日付: 2026-08-15
- 判定: **Phase I 実装 NO-GO / 設計固定（Design-2）フェーズ**。本ドキュメントで実装者が曖昧さなく実装できる設計を固定する。
- 前提: `doc/work88/I1_DESIGN_REVIEW.md`（10項目突合）完了。プラン §1.2.1〜§1.2.3（R1〜R17）をコードに基づき具体化。
- ⚠️ **Design-3 反映（2026-08-15）**: `doc/work88/I3_DESIGN_CONTRACT.md` により、本ドキュメントの
  **D1（generation）/ D3（changedDomains 導出）/ D4（canSupersede）/ D5（STALL）/ D6（RC-11）を改訂**し、
  **D8（RecoveryGeneration semantic contract）/ D9（Semantic supersession contract）/ D10（STALL ownership
  contract）を追加**した。**実装時は I3 の D8〜D10・D3'/D4'/D5'/D6' を正本とすること**（本ドキュメントの
  D3/D4 の `changedDomains` 単体導出は**廃止** — 基準状態がなく superset 判定が不安定なため）。
- 設計順序（ユーザー指定）: `LogicalRecoveryIdentity → RecoveryProvenance → semantic build identity → canSupersede() → bounded durable admission → Building 中 pending supersession → invariant/adversarial tests → 実装 GO`
- コード裏付け: `RuntimeBuildTypes.h`（RuntimeBuildSnapshot:48 / BuildInput:20 / RuntimeBuildFingerprint:38 / snapshot 等価性:333-336）/ `ISRRuntimePublicationCoordinator.cpp:855-925`（submitRecoveryRequest durable 経路）/ `AudioEngine.RebuildDispatch.cpp:960-1070`（recovery build 経路）

---

## 0. 現行コードの問題点（実装開始時に必ず撤去）

**`ISRRuntimePublicationCoordinator.cpp:905-915` の無条件 latest-wins 上書き**:
```cpp
// durable admission へ保持（coalesce: 単一スロット — 既存 durable があれば最新で上書き）
pendingRecoveryAdmission_.state = ...DurablePending;
pendingRecoveryAdmission_.recoveryGeneration = intent.intentId;  // 🔴 intentId を generation に代用
pendingRecoveryAdmission_.buildSource = buildSource;             // 🔴 無条件上書き
...
```
- 🔴 **P0-7 顕在**: 既存 obligation（DurablePending / Building 含む）を supersession 証明なしに上書き → 正当な Recovery 喪失。
- 🔴 **generation の誤用**: `recoveryGeneration` に `intentId`（診断 sequence）を代入。プラン §1.2.3「intentId は identity ではない」違反。
- **Phase I 最初の実装でこの経路を残さない**（ユーザー指示: 絶対に残さない）。

---

## D1. LogicalRecoveryIdentity（R1）

```cpp
struct LogicalRecoveryIdentity {
    DSPHandle handle;                    // recovery 対象（quarantined DSPHandle）
    std::uint64_t generation;            // ★ authoritative source: RuntimeBuildSnapshot::generation
    SemanticBuildIdentity semanticIdentity;  // D3
    PublicationEpoch epoch;              // activation/publication epoch（quarantine 時点の authoritative epoch）
};
```

### D1.1 generation の authoritative source（決定）
- **`RuntimeBuildSnapshot::generation` を使用する**。`RuntimeBuildSnapshot.generation` は build 時に
  `rebuildRequestGeneration`（authoritative rebuild generation counter・`consumeAtomic(rebuildRequestGeneration)`
  で読み、`recoverySnapshot.generation = recoveryGeneration` と代入 — `RebuildDispatch.cpp:982-985`）から
  値コピーされる。したがって **buildSource.generation は epoch-consistent な generation を既に運んでいる**。
- **現行 `cpp:910` の `recoveryGeneration = intent.intentId` を廃止**し、`generation = buildSource.generation` に変更。
- **intentId は identity/generation に一切使用しない**（diagnostic sequence のみ・`nextRecoveryIntentId_`）。
- ⚠️ **実装前検証項目**: submitRecoveryRequest に渡る `buildSource.generation` が quarantine 時点で
  実値（rebuildRequestGeneration 由来）であることを確認する。未設定（0）の場合は、Phase G R7 と同様の
  **explicit parameter パターン**で `submitRecoveryRequest(handle, buildSource, epoch, generation)` に
  rebuildRequestGeneration を明示渡しする（決定待ち 4 参照）。

### D1.2 epoch の関係（明文化）
- `epoch` = quarantine 時点の authoritative publication epoch（`AudioEngine::currentPublicationEpoch()` 由来・
  RuntimeStore::current 単一 source — Phase G 整合）。
- **generation domain 判定（R9）**: 同一 handle でも epoch domain が異なる（別 activation 世代の quarantined
  config）場合は coalesce/supersede 不可。`epoch` が generation domain の anchor になる。
- `generation`（rebuildRequestGeneration 由来・グローバル単調）と `epoch`（publication）は別カウンタだが、
  両方とも LogicalRecoveryIdentity に含め、**両者の組で「どの config 世代の復旧か」を特定**する。

### D1.3 等価性
- 全フィールドの構造的等価性（handle + generation + semanticIdentity + epoch）。完全一致 = 同一 logical recovery
  → **coalesce**（buildSource 最新化・reservation 不変・INV-X1-5）。

---

## D2. RecoveryProvenance（R2）— identity と分離

```cpp
enum class RecoveryProvenance : std::uint8_t {
    Transport,   // transport-layer failure（IR load timeout in flight）
    Durable,     // durable-layer failure（build snapshot failed）
    Retry,       // transient retry after failure（Building → DurablePending）
    Quarantine   // quarantined due to age or error
};
```
- **identity ≠ provenance**: `LogicalRecoveryIdentity` に provenance を**含めない**（§1.2.1 レビュー指摘）。
- 同一 LogicalRecoveryIdentity + 異なる provenance = **同一 logical recovery**（coalesce 対象）。
- provenance は診断・テレメトリのみ（`RecoveryIntent` に診断フィールドとして保持・identity 比較に不使用）。
- 二重計上防止（INV-X1-6）: transport（`recoveryIntentQueue_` residency）と durable（table residency）は
  排他 — 同一 obligation が両方に存在しない（現行の reservation-before-push + rollback を維持）。

---

## D3. SemanticBuildIdentity（R1 の semantic build identity + canonicalization）

### D3.1 構成要素（RuntimeBuildSnapshot から導出・既存フィールド再利用）
```cpp
struct SemanticBuildIdentity {
    std::uint64_t irIdentityHash;         // rebuildFingerprint.irIdentityHash      （IR 変更検知）
    std::uint64_t convolutionConfigHash;  // rebuildFingerprint.convolutionConfigHash（conv config 変更）
    std::uint64_t dspParameterHash;       // rebuildFingerprint.dspParameterHash    （EQ/DSP 変更検知）
    std::uint64_t convolverFingerprint;   // snapshot.convolverFingerprint
    std::uint64_t buildInputHash;         // BuildInput の canonical hash（D3.2）
    std::uint32_t changedDomains;         // 変更セマンティックドメイン bitmask（D3.3・superset 判定用）
};
```
- **基盤**: `RuntimeBuildTypes.h:333-336` の snapshot 等価性が既に
  `convolverFingerprint + irIdentityHash + convolutionConfigHash + dspParameterHash` を比較 — これを
  semantic identity のベースに再利用する。
- **EQ 変更 vs IR 変更の区別**: EQ 変更 → `dspParameterHash` 変化。IR 変更 → `irIdentityHash` 変化。
  この分解により「G10=EQ change / G11=IR change → 別 semantic target」を判定可能（D4）。

### D3.2 buildInputHash の canonicalization（決定）
- `BuildInput`（sampleRate / blockSize / eqBypassed / convBypassed / oversamplingFactor /
  processingOrder / noiseShaperType / ditherBitDepth / autoGainStagingEnabled 等）を**決定的に** hash。
- canonical 化規則:
  - `double sampleRate` は `memcpy` で `uint64` に bit-canonical 化（浮動小数比較に依存しない）。
  - フィールド順序を固定し FNV-1a 64 で結合（order-independent でなく **order-fixed** — 同一入力は常に同一 hash）。
  - hash 衝突は実用上無視（2^64 space・build 入力の識別用途）。

### D3.3 changedDomains（superset 判定の基盤・R3 の鍵）
- `isSemanticSuperset` を計算可能にするため、**「どのセマンティックドメインが変更されたか」の bitmask**
  を identity に保持する。
  ```cpp
  enum SemanticDomain : std::uint32_t {
      DomainNone   = 0,
      DomainIR     = 1u << 0,  // irIdentityHash 変更
      DomainConv   = 1u << 1,  // convolutionConfigHash / convolverFingerprint 変更
      DomainEQ     = 1u << 2,  // dspParameterHash 変更（EQ/パラメータ）
      DomainConfig = 1u << 3,  // buildInputHash 変更（SR/block/OS/処理順）
  };
  ```
- **導出方法（実装前検証項目）**: quarantine/admission 時に、入ってくる RuntimeBuildSnapshot を
  **直前 published world の build identity と diff** して changedDomains を導出する関数
  `deriveChangedDomains(snapshot, currentPublishedIdentity)` を新設。
  - ⚠️ **決定待ち 2**: snapshot に changedDomains を直接保持する新フィールドを追加するか、
    導出関数（admission 時 diff）にするか。**推奨: 導出関数**（snapshot を変更せず・既存 fingerprint から
    diff で計算 — ソース変更を最小化）。

---

## D4. canSupersede() / SupersessionDecision（R3・R7〜R10）

```cpp
enum class SupersessionDecision : std::uint8_t {
    CanSupersede,            // newer が older の Recovery obligation を完全に代替できる
    DifferentHandle,         // R8: handle 不一致
    DifferentSemanticTarget, // R7: semantic build identity の対象が異なる
    NotSameGenerationDomain, // R9: generation/epoch domain 不一致
    NotSemanticSuperset,     // R10: newer が older の semantic superset でない
};

SupersessionDecision canSupersede(const LogicalRecoveryIdentity& newer,
                                  const LogicalRecoveryIdentity& older) noexcept;
```

### D4.1 成立条件（全て満たすときのみ CanSupersede）
1. `newer.handle == older.handle`（else `DifferentHandle`）。
2. 同一 generation domain: `newer.epoch == older.epoch`（同一 activation epoch の config 世代。
   else `NotSameGenerationDomain`）。
3. `isAfter(newer.generation, older.generation)`（`SequenceArithmetic.h` の modular 比較 —
   strictly greater。else `NotSameGenerationDomain`）。
4. `isSemanticSuperset(newer.semanticIdentity, older.semanticIdentity)`（else `NotSemanticSuperset`）。

### D4.2 isSemanticSuperset（D3.3 の changedDomains で定義・決定）
```cpp
// newer の変更セマンティックドメイン集合が older のそれを包含する（set inclusion）。
bool isSemanticSuperset(const SemanticBuildIdentity& newer, const SemanticBuildIdentity& older) {
    // 変更ドメイン集合の包含: newer.changedDomains ⊇ older.changedDomains
    return (older.changedDomains & ~newer.changedDomains) == 0;
}
```
- **G10 = {EQ}（dspParameterHash 変更）・G11 = {IR}（irIdentityHash 変更）**:
  `{IR} ⊉ {EQ}` → NOT supersede ✓（**EQ 変更を IR 変更で supersede しない** — ユーザー指示の明確化）。
- **G10 = {IR, EQ}・G11 = {IR, EQ, Config}**: `{IR,EQ,Config} ⊇ {IR,EQ}` → supersede ✓。
- **同一 identity**（changedDomains 完全一致）: `a == b` → superset（coalesce と等価）。
- ⚠️ **「同じ新しい recovery」だからという理由で supersede しない**: generation が新しいだけで
  changedDomains が包含でない場合は `NotSemanticSuperset`（ユーザー指示の明確化）。

### D4.3 純粋関数
- `canSupersede` / `isSemanticSuperset` は **副作用なしの純粋関数**（NonRT・CoordinatorLoop のみで呼ぶ）。

---

## D5. Bounded durable admission table（I-3・P0-7 解消）

### D5.1 構造（single slot → bounded array）
```cpp
static constexpr size_t kMaxDurableRecoveryAdmissions = 16;  // ★ 決定待ち 1（候補 8/16/32）

struct DurableRecoverySlot {
    LogicalRecoveryIdentity identity;            // coalesce/supersede 判定の key
    RecoveryProvenance provenance;               // 診断のみ（identity に含めない）
    convo::RuntimeBuildSnapshot buildSource;     // latest（coalesce で更新・reservation 不変）
    PendingRecoveryAdmission::State state;       // NoAdmission / DurablePending / Building
    bool reservationOwned;                       // INV-X1-5（1 identity = 1 reservation）
    uint64_t intentId;                           // 診断のみ
};
std::array<DurableRecoverySlot, kMaxDurableRecoveryAdmissions> durableTable_{};
```

### D5.2 挿入アルゴリズム（transport-full fallback 時・CoordinatorLoop 単一スレッド）
```
admitDurable(incoming):
 1. 既存 slot に同一 LogicalRecoveryIdentity がある → COALESCE:
      buildSource を最新化・reservation 不変（INV-X1-5）・provenance 更新（診断）
      → return Accepted
 2. canSupersede(incoming, slot.identity) == CanSupersede を満たす slot がある
      → SUPERSEED: 該当 slot を incoming で置換（旧 obligation は完全に包含済み = 安全）
      → return Accepted
 3. 空 slot（state == NoAdmission）がある → INSERT: return Accepted
 4. それ以外（table full・coalesce/supersede 不能）→ OVERFLOW POLICY（D5.4）
```

### D5.3 不変条件（INV-R15・ユーザー要件）
- **非 superseded obligation は絶対に evict しない**。
- eviction が許されるのは: (a) 別の logical obligation による **supersession 証明**（手順2）のみ。
  (b) Builder の settle（成功 / Discarded）。
- coalesce（手順1）は obligation を保持したまま buildSource を更新（喪失なし）。
- `logicalAdmissionCount == 1` per LogicalRecoveryIdentity（R5）。

### D5.4 OVERFLOW POLICY（容量超過時の喪失防止・ユーザー要件の核心）
- **容量を超えたとき**: coalesce（同一 identity 吸収）+ supersede（包含置換）+ insert（空 slot）が全て不能
  = kMax 個の**異なる非 supersedable obligation** が同時に outstanding。
- **ポリシー（推奨・決定待ち 3）: 喪失を構造的に防ぐ — STALL（drop 禁止・evict 禁止）**:
  1. 既存 obligation は一切触らない（evict しない）。
  2. incoming obligation を**捨てない**: CoordinatorLoop（単一 producer）は quarantine intent の
     recovery handoff を完了せず、**次サイクルで再試行**する（slot が Builder settle で空いた時点で
     admit）。quarantine intent 自体は drop しない。
  3. `recoveryDurableOverflowCount_` テレメトリ + Debug `assert(false)`（構造的不変条件の違反を観測可能化）。
- **容量の決定（決定待ち 1）**: kMax は「Builder stall 中に同時に outstanding になり得る異なる
  logical recovery obligation 数」を上回るよう設定。候補 16（推奨）: quarantine 対象 handle 数は
  DSPHandleTable（512）に上限されるが、実際の同時 quarantined は少数・Builder は高速消費。
  16 なら 2^4 で power-of-2（index 計算が簡潔）。**ユーザー確認で確定**。

---

## D6. Building 中の pending supersession（RC-11 / H.11.27.3・I-4）

### D6.1 問題
- 既存 slot が **Building**（Builder に lease 済み・identity = I_old・in-flight build 進行中）のときに、
  `canSupersede(I_new, I_old) == CanSupersede` を満たす新規 recovery I_new が来た場合。

### D6.2 設計
- **Building lease は immutable**: in-flight build は I_old の buildSource で完了する（途中変更しない）。
- **I_new は Building slot を上書きしない**: bounded table（D5）の別 slot に DurablePending として保持
  （Building slot と共存可能 — **これが single slot では不能だった点・table 化の動機**）。
- **settle 時の昇格**:
  - `settlePendingRecoveryAdmission(false)`（I_old 成功）: I_new（superseding）を DurablePending → 次の
    take で Building。既に publish された I_old はそのまま（次世代が上書き）。
  - `settlePendingRecoveryAdmission(true)`（I_old transient fail）: **I_new（newer）を優先**して
    DurablePending → 再 take で Building（newer buildSource で retry）。
- **不変条件**: superseding obligation は Building lease を clobber せず・drop されず・table に保持される。
- 状態機械: `NoAdmission → DurablePending → Building → [TransientFail → DurablePending] → [Success → NoAdmission]`
  （§1.2.2 手順3 の定義どおり）に、Building 中の superseding obligation を table 内 DurablePending として
  保持する拡張。

---

## D7. invariant / adversarial tests（実装 GO 時に追加するテスト仕様）

新規 `RecoveryCoalesceTests`（ISRSemanticValidationTests パターン）または既存に追加:

| # | テスト | 検証する不変条件 |
|---|--------|------------------|
| T1 | 同一 LogicalRecoveryIdentity を2回 admit → coalesce | logicalAdmissionCount == 1（INV-X1-5）・buildSource = latest |
| T2 | A(G10), B(G10), C(G11)（canSupersede(C,A)&&(C,B)）→ C のみ保持 | obsolete policy（§1.2.2）・非 superseded 削除禁止（R15） |
| T3 | EQ change（G10）vs IR change（G11）→ supersede しない | R7/R10・**EQ を IR で supersede しない** |
| T4 | 異なる handle → coalesce しない | R8（DifferentHandle） |
| T5 | 異なる epoch/generation domain → coalesce しない | R9（NotSameGenerationDomain） |
| T6 | non-superset（changedDomains 非包含）→ coalesce しない | R10（NotSemanticSuperset） |
| T7 | Building 中に superseding 到来 → pending 保持・settle 後に昇格 | RC-11 / I-4 |
| T8 | table full + 非 supersedable incoming → 既存 evict なし・incoming stall（overflow telemetry） | D5.4 / 喪失防止 |
| T9 | 各 coalesce 後に logicalAdmissionCount == 1 | R5（exactly-one-logical-obligation） |
| T10 | transport/durable 二重計上なし（同時存在しない） | INV-X1-6 |
| T11 | queue-full → durable fallback で obligation 保持 | INV-X1-2 / R13 |
| T12 | adversarial: transport-full + durable-full + coalesce + supersede の連続 | 総合 invariant（喪失 0・二重計上 0） |

---

## 実装 GO 条件と決定待ち事項

### 決定（2026-08-15 自律採用・レビュー可能）
> ユーザー不在時の「work autonomously and make good decisions」指示に基づき、下記の推奨デフォルトを
> **設計として固定**した。ユーザーレビューで調整可能（値・方針の変更は実装前に反映する）。

1. **`kMaxDurableRecoveryAdmissions = 16`**（power-of-2・Builder stall 中の同時異種 obligation を十分
   カバー。DSPHandleTable 512 の上限内で現実的な同時 quarantined を吸収）。
2. **changedDomains は導出関数 `deriveChangedDomains(snapshot, currentPublishedIdentity)`**（snapshot
   非変更・既存 fingerprint から diff で計算・ソース変更を最小化）。
3. **OVERFLOW POLICY = STALL**（evict 禁止・drop 禁止・CoordinatorLoop で次サイクル再試行 +
   `recoveryDurableOverflowCount_` telemetry + Debug assert）。
4. **generation は `submitRecoveryRequest(handle, buildSource, epoch, generation)` に明示渡し**
   （Phase G R7 と同パターン）。実装時に `buildSource.generation` が実値であることを確認し、
   実値ならそれを使用（明示渡しは防御的整合のための冗長化として維持）。

### 実装 GO 条件
- 上記4決定の設計固定で **実装準備完了（implementation-ready）**。
- **実装 GO はユーザー確認後**（Phase I は設計レビュー完了 + 設計固定完了・実装 NO-GO を維持。
  「設計固定」と「実装」を混同しない — 本ドキュメントは実装コードを変更しない）。

### 実装順序（決定待ち確定後・ユーザー指定パイプライン）
```
D1 LogicalRecoveryIdentity（generation = buildSource.generation・intentId 排除）
  ↓
D2 RecoveryProvenance enum
  ↓
D3 SemanticBuildIdentity + deriveChangedDomains（canonicalization）
  ↓
D4 canSupersede() / SupersessionDecision / isSemanticSuperset
  ↓
D5 bounded durable admission table（coalesce → supersede → insert → stall・cpp:905-915 撤去）
  ↓
D6 Building 中 pending supersession（RC-11）
  ↓
D7 invariant / adversarial tests（T1〜T12）
  ↓
I-5 build + ctest 全 PASS（rollback point）
```

### 現行コードの変更対象（実装時）
- `ISRRuntimePublicationCoordinator.h`: `PendingRecoveryAdmission` single slot → `DurableRecoverySlot` 配列
  （`kMaxDurableRecoveryAdmissions`）/ `LogicalRecoveryIdentity` / `RecoveryProvenance` /
  `SemanticBuildIdentity` / `SupersessionDecision` 型新設。`submitRecoveryRequest` / `takePendingRecoveryAdmission` /
  `settlePendingRecoveryAdmission` / `discardPendingRecoveryAdmission` の table 対応。
- `ISRRuntimePublicationCoordinator.cpp`: `cpp:905-915` の無条件上書き → admitDurable アルゴリズム。
- `AudioEngine.RebuildDispatch.cpp`: recovery build 経路（durable table からの消費・settle 時 supersession 昇格）。
- `src/tests/`: `RecoveryCoalesceTests` 新設（T1〜T12）。

### 正式判定
**Phase I 実装 NO-GO（維持）。決定待ち 1〜4 をユーザーが確認した時点で実装 GO に移行する。**
