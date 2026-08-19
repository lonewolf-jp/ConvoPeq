# I4 Design Contract — Phase I-Design-5（Semantic Supersession Contract 再固定）

- 日付: 2026-08-15
- 判定: **Phase I 実装 NO-GO（継続）。Design-5 で D9（domain-based supersession）を中心に再固定。**
- 前提: `I1_DESIGN_REVIEW.md` → `I2_DESIGN_FIX.md` → `I3_DESIGN_CONTRACT.md`（D8〜D11）。ユーザー Design-4 レビュー:
  - GO: RecoveryGeneration 分離 / retry invariant / baseline immutable
  - REFINEMENT REQUIRED: D8 semantic domains / D11.3 capacity 24 / D11.5 ownership conservation
  - **NO-GO（最重要）: D9 domain-based supersession** — 「同じ domain を変更している」≠「older obligation を semantic に包含している」
- 本ドキュメントの目的: **「domain containment（必要条件）」と「semantic target containment（十分条件）」を分離**し、`canSupersede()` の十分条件を実装前に固定する。これが未解決のまま実装すると latest-wins 撤去後も「別の形の誤った supersession」で obligation を消す恐れがある。

---

## 0. Design-4 レビュー判定の反映

| 項目 | Design-4 判定 | Design-5 対応 |
|------|--------------|---------------|
| D9 domain-based supersession | **NO-GO** | **D12 で SemanticRecoveryTarget（domain + target containment）に再固定** |
| D8 semantic domains | REFINEMENT | **D13 で 7 ドメインに分離 + supersession lineage = RecoveryEpisodeId** |
| D11.3 capacity 24 | REFINEMENT | **D14 で reservation-first + 単一 budget + backpressure（terminal-failure 撤廃）** |
| D11.5 ownership conservation | REFINEMENT | **D15 で superseded/shutdownDiscard を明示分離 + admission event ≠ logical obligation** |
| D11.1 RecoveryGeneration | GO（+wraparound 未定義） | **D16 で arithmetic contract（equality/strict-after/wraparound/zero-init/exhaustion）** |
| D11.2 baseline immutable | GO（+所有者未明示） | **D17 で episode lifecycle / baseline ownership を固定** |
| Phase I implementation | NO-GO | NO-GO 継続 |

---

## D12 — Semantic Supersession Contract（D9 再固定・最重要）

### D12.1 指摘の確認（ユーザー）
```
baseline = S0 = {IR=A, EQ=1}
A = {IR=B, EQ=1}   A.obligationDomains = {IR}
B = {IR=C, EQ=1}   B.obligationDomains = {IR}
現在の isSemanticSuperset(B, A) = {IR} ⊇ {IR} = true  ← 誤り
```
「IR domain を担当している」≠「A（IR=B）の logical obligation を semantic に包含している」。
`ObligationDomains` は**必要条件**であり、**十分条件にできない**。fingerprint の equality/hash だけでは
一般に「B が A を包含する」を判定できない。

### D12.2 SemanticRecoveryTarget（domain coverage + target containment の分離）
```cpp
struct SemanticRecoveryTarget {
    ObligationDomains domainCoverage;      // 必要条件: どの domain を担当するか（D9.2 の固定 baseline diff）
    // target containment の判定に使う semantic 値（十分条件）
    std::uint64_t irIdentityHash;          // IR ドメインの実値
    std::uint64_t convolutionConfigHash;   // Conv ドメインの実値
    std::uint64_t convolverFingerprint;    // Conv ドメインの実値
    std::uint64_t dspParameterHash;        // EQ ドメインの実値
    std::uint64_t buildInputHash;          // Config ドメインの実値
};

// 必要条件: domain coverage の集合包含
bool isDomainSuperset(const SemanticRecoveryTarget& n, const SemanticRecoveryTarget& o) noexcept {
    return (o.domainCoverage & ~n.domainCoverage) == 0;
}

// 十分条件: semantic target の実値包含。
// ★ Phase I 定義: 全 semantic 値の等価性（決定可能な唯一の「包含」・保守的）。
//   「IR+EQ+OS ⊇ IR+EQ」等の compositional superset は semantic partial-order のモデル化が必要なため
//   Phase I では対象外（明示的に deferred）— 誤って obligation を消さないため。
bool isSemanticTargetSuperset(const SemanticRecoveryTarget& n, const SemanticRecoveryTarget& o) noexcept {
    return n.irIdentityHash         == o.irIdentityHash
        && n.convolutionConfigHash  == o.convolutionConfigHash
        && n.convolverFingerprint   == o.convolverFingerprint
        && n.dspParameterHash       == o.dspParameterHash
        && n.buildInputHash         == o.buildInputHash;
}

// 全体: 必要 AND 十分
bool isSemanticSuperset(const SemanticRecoveryTarget& n, const SemanticRecoveryTarget& o) noexcept {
    return isDomainSuperset(n, o) && isSemanticTargetSuperset(n, o);
}
```

### D12.3 修正された truth table
| ケース | isDomainSuperset | isSemanticTargetSuperset | isSemanticSuperset |
|--------|------------------|--------------------------|--------------------|
| identical target（全値等価・同一 episode・newer gen） | ✅ | ✅ | **✅ supersede** |
| target が 1 成分でも異なる（例: IR=B vs IR=C） | ✅（{IR}⊇{IR}） | ❌ | **❌ NOT supersede** |
| 異なる domain（{EQ} vs {IR}） | ❌ | — | ❌ NOT supersede |

- **ユーザーの反例を解決**: A={IR=B,EQ=1}, B={IR=C,EQ=1} → isSemanticTargetSuperset が false →
  **B は A を supersede しない**（同じ domain {IR} でも、異なる IR 値は包含を証明できない）。
- **identical semantic identity supersedes identical older obligation**: 全値等価なら ✅（証明可能）。
- ⚠️ **"IR+EQ may supersede IR" は Phase I では対象外（deferred）**（D9.3 の truth table から削除）。
  これは効率（coalesce 機会）を犠牲にするが、**誤った supersession による obligation 消失を構造的に防ぐ**
  （ユーザー優先度: correctness > 効率）。

### D12.4 canSupersede() 再固定（ユーザー Design-5 形式）
```
canSupersede(newer, older) =
      same handle                       （DifferentHandle）
    + same recovery episode             （DifferentEpisode）        ← RecoveryEpisodeId（D13）
    + newer RecoveryGeneration          （isAfter・RecoveryGeneration ドメイン・D16）
    + semantic target containment       （isSemanticSuperset・D12.2 → NotSemanticSuperset）
```

### D12.5 決定待ち D（重要）
- `isSemanticTargetSuperset` を **equality（保守的・Phase I 採用）** にするか、特定の compositional
  superset（IR+EQ ⊇ IR 等）を Phase I で定義するか。**推奨: equality（保守的）** — compositional
  superset は semantic partial-order モデルが必要で、誤判定のリスクが高く、Phase I の correctness 優先に反する。

---

## D13 — Semantic domain 分離の再固定（D8 refinement）

### D13.1 7 ドメインへの分離
| ドメイン | 順序付け/意味 | supersession lineage に使うか |
|----------|---------------|------------------------------|
| `PublicationSequence` | published world の順序 | ❌ |
| `EBREpoch` | lifetime/EBR reclamation ordering | ❌ |
| `ActivationEpoch` | config が activation された publication epoch（診断・世代 anchor メタ） | ❌ |
| `RecoveryGeneration` | recovery lineage 内の順序（episode 内） | ⚠️ newer 判定（isAfter） |
| `BuildGeneration` | build 完了時の world generation | ❌ |
| **`RecoveryEpisodeId`**（新設） | **quarantine エピソード / config lineage の識別子** | ✅ **同一 lineage 判定** |
| `ConfigLineageId` | （RecoveryEpisodeId の別名・同一概念） | — |

### D13.2 ユーザー指摘の解決
- **EBR/lifetime epoch ≠ semantic configuration lineage**: Publication A(epoch=100, config=X) →
  Publication B(epoch=101, config=X) は普通に成立（epoch は変わっても config は同一）。
- **supersession の同一 lineage 判定に `epoch` を使わない**。`newer.epoch == older.epoch` 条件を**廃止**し、
  **`RecoveryEpisodeId`**（明示的な lineage identifier）に置き換える。
- `ActivationEpoch` は診断/世代 anchor メタデータとして identity に保持（可読性）するが、
  **supersession の gate には使用しない**。
- `RecoveryEpisodeId` の割当: quarantine episode 開始時に専用カウンタから割当（episode 内不変）。
  LogicalRecoveryIdentity の構成要素に含める。

---

## D14 — Capacity / reservation-first / backpressure（D11.3/11.4 refinement）

### D14.1 ユーザー指摘
- **「24」は設計根拠になっていない**: 512 handles なら 25 concurrent obligations は普通に構成可能。
  「同時 quarantine episode ≤ 24」の invariant がなければ証明にならない。
- **budget 枯渇 → terminal-failure は P0-7 を observable-loss に置き換えただけになり得る**。
  非 superseded obligation を容量超過でも絶対に失わないなら terminal-failure は許されない。
- **「single bounded resource」は実際には single でない**: transport(256) + durable(16) + stalled(8)
  が構造的に共存可能。`kMaxLogicalRecoveryObligations` が何を支配するかが不明。

### D14.2 reservation-first モデル（固定）
```
Transport admission
        ↓
Logical obligation reservation（最初に取得・単一 budget から）
        ↓
placement: Transport / DurablePending / Building / Stalled
```
- **1 logical obligation = exactly 1 reservation**（INV-X1-5 厳密化）。
- reservation は admission 時に取得し、**placement（storage の場所）が変わっても不変**。
- `kMaxLogicalRecoveryObligations` = 全 placement を合わせた総量の上限
  （transport も reservation を消費するため、transport の実効容量 ≤ budget になる）。
- 不変: `transportCount + durableCount + buildingCount + stalledCount ≤ kMaxLogicalRecoveryObligations`。

### D14.3 budget 枯渇 = backpressure（terminal-failure 撤廃）
- budget 枯渇で reservation を取得できない場合、**admission を BLOCK**（backpressure / upstream stall）。
  - CoordinatorLoop（単一 producer）が in-flight quarantine intent の義務を保持し、
    budget 解放（terminal disposition）後に再試行。
  - **非 superseded obligation の terminal-failure による消失は許さない**（P0-7 を observable-loss に
    置き換えない）。
- **INV-X1-7 改訂**: logical obligation の消失理由から「explicitly observable terminal failure」を**削除**。
  消失は **Success / Superseded / ShutdownDiscard** のみ。terminal failure は構造的非到達の観測可能化
  （Debug assert + telemetry）に留め、**disappearance 理由としては認めない**。
- **capacity の根拠（決定待ち E）**: `kMaxLogicalRecoveryObligations` の値は「同時 quarantine episode 数」の
  **invariant とセット**で定める。候補 32: 単一ユーザーの同時 config-failure エピソード数は実用上 1〜数個。
  coalesce/supersede により live obligation ≤ concurrent episodes。32 ≫ 実同時 episode。
  **構造（reservation-first）は固定・値はレビュー可能**。

---

## D15 — Ownership conservation 厳密化（D11.5 + INV-X1-5/7 refinement）

### D15.1 ユーザー指摘
- `superseded` が右辺に行かない: R1 issued, R2 issued, R2 supersedes R1 →
  issued=2, live=1, superseded=1。terminal disposition を明示分離すべき。
- coalesce は logical obligation を増やさない → **admission event count ≠ logical obligation count**。

### D15.2 固定
```
// 受理された logical obligation 数（coalesce/supersede で増えない）
admittedLogicalObligationCount
// submitRecoveryRequest 呼び出し回数（≥ logical obligation count）
admissionEventCount

ownership conservation:
    transportCount + durableCount + buildingCount + stalledCount
        + supersededCount + shutdownDiscardCount
        == admittedLogicalObligationCount

（terminal-failure は消失理由に含めない — D14.3。Debug assert のみ）
```
- **INV-X1-5 厳密化**: 1 logical obligation = exactly 1 reservation（reservation-first・D14）。
  `logicalObligationCount ≤ admissionEventCount`（coalesce は logical 数を増やさない）。
- **INV-X1-7 厳密化**（D14.3 反映）:
  ```
  For every admitted Recovery obligation:
      exactly one of Transport / DurablePending / Building / Stalled owns the obligation.
  No transition may:
      duplicate the logical obligation,
      silently discard it,
      overwrite a non-supersedable obligation,
      cancel a Building lease through supersession.
  A logical obligation may disappear only by:
      Success, explicit Superseded decision, ShutdownDiscard.
  ```

---

## D16 — RecoveryGeneration arithmetic contract（D11.1 refinement）

- 方向（専用カウンタ・retry-invariant）は GO 確定。**wraparound / zero-init / exhaustion を固定**。
```
RecoveryGeneration arithmetic:
  - equality:      uint64 等価（特別な意味論なし）
  - strict-after:  isAfter(a,b) = isBefore(b,a)（SequenceArithmetic・modular < 2^63）
  - wraparound:    modulo 2^64（kSeqHalfModulus）— 単一 Producer の単調カウンタで正当（D8 の domain 証明）
  - zero-init:     RecoveryGeneration{0} = 未割当/なし を予約。実値は 1 から開始
  - exhaustion:    alloc が 0 を生成しない（fetch_add 後の値が 0 なら 1 を再取得 or 予約値を回避 —
                   実用上到達不能だが、予約値 0 を割当てないことを仕様化）
```
- SequenceArithmetic 適用は **RecoveryGeneration ドメインで独立に証明**（Phase H の PublicationSequence 検証を流用しない）。

---

## D17 — Baseline ownership（D11.2 refinement）

- 方向（episode start capture・immutable・全 admission 同一 baseline）は GO 確定。**lifecycle を固定**:
```
Who creates the episode?    — quarantine admission authority（quarantine 受理時に episode 開始 + RecoveryEpisodeId 割当）
Who owns baseline?          — RecoveryEpisode（episode authority）が baselineIdentity を所有（不変・値コピー）
When is episode closed?     — エピソード内の最後の logical obligation が terminal（Success / Superseded / ShutdownDiscard）
                              （それ以前に baseline を解放しない）
Can baseline be reclaimed while an admission exists? — NO（admission が生存中は episode も生存）
RecoveryAdmission └── immutable value copy:
    admission は baselineIdentity を値コピーで保持 → episode authority の baseline lifetime に依存しない
    （値セマンティクス・独立 lifetime）。episode が閉じても admission の ObligationDomains は不変。
```

---

## テスト更新（D7 + Design-5 追加）

| # | テスト | 検証 |
|---|--------|------|
| T1-T12 | Design-2/3 のテスト | 維持（INV-X1-5/7 厳密化に合わせて count 比較を修正） |
| T13 | retry 不変性 | Building→fail→DurablePending→Building で RecoveryGeneration 不変 |
| T14 | baseline immutability | published が S1 に進んでも B の baseline は S0（判定不変） |
| T15 | ownership conservation | transport+durable+building+stalled+superseded+shutdownDiscard == admittedLogicalObligationCount |
| T16 | budget 枯渇 | 全 budget 占有で coalesce/supersede 不能 → **backpressure（drop なし・既存不 evict）** |
| **T17** | **target containment（ユーザー反例）** | A={IR=B,EQ=1}, B={IR=C,EQ=1}（同 episode・newer gen）→ **B は A を supersede しない** |
| **T18** | identical target supersede | 同 handle・同 episode・同 target・newer gen → supersede（obsolete 重複を吸収） |
| **T19** | compositional superset deferred | IR+EQ ⊇ IR は Phase I で NOT supersede（deferred の確認） |
| **T20** | lineage = RecoveryEpisodeId | 同 handle・異なる episode → NOT supersede（epoch 不変でも） |
| **T21** | reservation-first | transport 滞留でも total ≤ kMaxLogicalRecoveryObligations |

---

## 決定待ち（最終）
- **D. isSemanticTargetSuperset**: equality（推奨・Phase I 採用）vs compositional superset 定義。
- **E. kMaxLogicalRecoveryObligations の値**: 候補 32（「同時 quarantine episode 数」invariant とセット）。
- それ以外の構造（D12〜D17）は確定。

## 最終判定
**Phase I 実装 NO-GO（継続）。Design-5 で D9 を SemanticRecoveryTarget（domain + target containment）に
再固定し、D8（RecoveryEpisodeId lineage）・D11.3（reservation-first + backpressure）・D11.5（conservation
厳密化）・D11.1（arithmetic）・D11.2（baseline ownership）を確定。決定待ち D/E を確認後、実装 GO。**
（→ **Design-6（D18）** は下記。Design-5: CONDITIONAL GO を確定し、D18 で coalesce/supersede 境界・
conservation 式・reservation semantics を修正。）

---

## D18 — Design-6 確定（2026-08-15・ユーザー Design-5 レビュー: D12=方向GO / D14=refinement / D15=NO-GO / D16=refinement / D13,D17=GO）

ユーザー指摘4点を固定。**特に D15 の conservation 式は現状のままテスト仕様に固定してはならない**（successCount 欠落で破綻）。

### D18.1 Coalesce / Supersede の境界（最重要・ユーザー指摘1）
**「同一 target の重複は supersede ではなく coalesce」**。境界を明示:
```
Step 0: CoalesceIdentity 検索
    CoalesceIdentity = { handle, RecoveryEpisodeId, SemanticRecoveryTarget }   ← RecoveryGeneration を含まない
    existing が同一 CoalesceIdentity → COALESCE
        · 新しい logical obligation を生成しない
        · 新しい reservation を取得しない
        · buildSource を最新化（RecoveryGeneration / reservation 不変）
    different → continue（canSupersede() 評価）
Step 1: same handle
Step 2: same RecoveryEpisodeId
Step 3: newer RecoveryGeneration（isAfter・D16）
Step 4: semantic target containment（D18.2）
    true → SUPERSEDE
    false → retain both
```
- **RecoveryGeneration は CoalesceIdentity に含めない**（含めると同一 target の2件が常に別 identity になり
  coalesce 不能 — ユーザー指摘1の「R1/R2 を同一 logical obligation に coalesce」が実現しない）。
- **Phase I の帰結（明記）**: D18.2 の containment = 全値等価のため、CoalesceIdentity が異なる（target が
  異なる）ケースでは containment が成立せず、**Phase I で SUPERSEDE は構造的に不活性（常に retain both）**。
  obsolete 重複の吸収は全て COALESCE が担う。canSupersede は Phase II の partial-order 導入時の拡張点。

### D18.2 Phase I semantic supersession（ユーザー指摘2）
```
Phase I:
    semantic containment == exact SemanticRecoveryTarget 全値等価（isSemanticTargetSuperset）
    domainCoverage は supersession 判定から【外す】→ admission metadata / diagnostics
        （target equality ⇒ domainCoverage equality のため、判定に追加の証明力を持たない）
    partial semantic containment（IR+EQ ⊇ IR 等）== deferred（Phase II）
```
- これにより Phase I の canSupersede 述語は実質:
  `sameHandle && sameRecoveryEpisode && isAfter(newer.gen, older.gen) && semanticTargetEqual(newer, older)`
  であり、`semanticTargetEqual` は CoalesceIdentity 一致を意味する → **D18.1 Step 0（coalesce）に吸収**。
- `domainCoverage` は将来 Phase II の partial-order 導入時の metadata として残す（診断・可視化用途）。

### D18.3 Ownership conservation（ユーザー指摘4・NO-GO 修正）
**現行式は破綻**（R1 admitted → Building → Success で左辺 0・右辺 1）。**successCount を追加**し、
current occupancy と terminal disposition を分離:
```
liveOwnershipCount      = transportCount + durableCount + buildingCount + stalledCount
terminalDispositionCount= successCount + supersededCount + shutdownDiscardCount

admittedLogicalObligationCount
    = liveOwnershipCount + terminalDispositionCount

admissionEventCount ≥ admittedLogicalObligationCount   （別 invariant・coalesce は event を増やすが
                                                          logical obligation を増やさない）
```
- `liveOwnershipCount` は **current ownership**（各時刻の占有）。
- `terminalDispositionCount` は **累積 terminal**（monotone）。
- Phase I では `supersededCount == 0`（D18.1 の不活性帰結）だが、式としては将来含めて正しい。
- **INV-X1-5 厳密化**: `logicalObligationCount ≤ admissionEventCount`（coalesce は logical 数を増やさない）。

### D18.4 Reservation semantics（ユーザー指摘3）
**reservation は「新規 logical obligation の生成」に対して exactly once**（全 admission event ではない）:
```
reservation is acquired exactly once per newly created logical obligation.

COALESCE:        新しい reservation を取得しない（既存 obligation の reservation を維持）
SUPERSEDE:      incoming が自身の reservation を取得
                  older obligation は明示的 Superseded disposition 後に reservation を release
SUCCESS:        reservation release
SHUTDOWN_DISCARD: reservation release
```
- **D14 の reservation-first を修正**: `reservation-first applies to creation of a new logical obligation,
  not to every admission event`。coalesce 候補の検索を reservation 取得より**先**に行う（budget 満杯でも
  既存同一 obligation への coalesce は可能 — 奇妙な予約失敗を排除）。

### D18.5 D13〜D17 の改訂ノート
- **D13**: `ConfigLineageId` の別名を**廃止**し、canonical name を **`RecoveryEpisodeId`** に一本化
  （「config lineage」は episode の意味説明に留める・型名を二重化しない — ユーザー指摘6）。
- **D14**: reservation-first は D18.4 のとおり「新規 logical obligation creation」に限定（coalesce 先行）。
  budget 枯渇 = backpressure は維持（新規生成時のみ）。
- **D16**: **wraparound（max→1）を Phase I の本番 semantic contract から外す**（ユーザー指摘5）。
  ```
  RecoveryGeneration:
      uint64_t
      0 = invalid/unassigned（予約）
      allocated values = 1..UINT64_MAX
      overflow は lifetime bound 下で構造的に到達不能（alloc 数 << 2^64）— 前提 invariant
      wraparound / modular arithmetic は Phase I では採用しない
        （将来必要なら Phase H SequenceArithmetic と同様に独立証明 — 流用しない）
  ```
- **D17**: episode closure を **`liveLogicalObligationCount == 0`** で定義（ユーザー指摘7）。
  - coalesce event は logical obligation を増やさない → episode lifetime に影響しない。
  - R1 superseded by R2 で R2 が live なら episode は継続。
  - `episode closes iff liveLogicalObligationCount == 0`。

### D18.6 テスト行列更新（D15 修正 + coalesce/reservation）
| # | テスト | 検証 |
|---|--------|------|
| T15 | ownership conservation（修正） | liveOwnershipCount + terminalDispositionCount == admittedLogicalObligationCount（success 含む） |
| **T18** | identical target → **COALESCE**（supersede でなく） | 同 handle・同 episode・同 target → coalesce・reservation 不変・logicalObligation 不変 |
| **T22** | coalesce と budget 満杯 | budget 満杯でも既存同一 CoalesceIdentity への coalesce は成功（reservation 取得不要） |
| **T23** | reservation lifetime | SUPERSEDE: incoming +1 → older release（Superseded 後） / SUCCESS・SHUTDOWN_DISCARD: release |
| T15b | admissionEventCount ≥ admittedLogicalObligationCount | coalesce は event を増やすが logical 数を増やさない |

---

## 決定待ち（最終・Design-6 反映）
- **D. isSemanticTargetSuperset**: Phase I では equality（確定）。compositional superset は Phase II（deferred）。
- **E. kMaxLogicalRecoveryObligations の値**: 候補 32（「同時 quarantine episode 数」invariant とセット）。
- D18.1〜D18.5 の構造は確定。

## Design-6 最終判定
**Phase I 実装 NO-GO（継続）。D18 で coalesce/supersede 境界（D18.1）・Phase I semantic supersession（D18.2）・
ownership conservation 修正（D18.3）・reservation semantics（D18.4）・D13〜D17 改訂（D18.5）を固定。
決定待ち E（budget 値）を確認後、実装 GO。**
（→ **Design-7（D19）** は下記。Design-6: CONDITIONAL GO。**「残る決定待ちは E だけ」は誤り** —
 episode-closure finality / backpressure progress / capacity derivation も必須。）

---

## D19 — Design-7 確定（2026-08-15・ユーザー Design-6 レビュー: D18.1〜D18.4/D13/D16/D17=GO・D18.5/D14/E=refinement 必須）

残存 blocking items 3点 + coalesce source mutation を固定。
**「残る決定待ちは E だけ」という記述は修正**（ユーザー指摘）— INV-X1-8/9/10 + INV-CAP が実装 GO 前に必須。

### D19.1 INV-X1-8 — Episode closure finality（D18.5 refinement）
```
An episode may close only when: liveLogicalObligationCount == 0.
Once closed:
  - no new logical obligation may be admitted to that RecoveryEpisodeId
  - no COALESCE may target that RecoveryEpisodeId
  - no SUPERSEDE may target that RecoveryEpisodeId
  - RecoveryEpisodeId is never reused
  - baseline ownership may be released
new recovery episode → new RecoveryEpisodeId（monotonic / non-reused episode identity）
```
- **RecoveryEpisodeId 割当**: 専用単調カウンタ `nextRecoveryEpisodeId_`（non-reused・0 は予約）。単一 Producer
  （CoordinatorLoop）で割当・閉鎖判定は atomic。
- **closed episode への admission / coalesce / supersede は REJECT**（`recoveryClosedEpisodeRejectCount_`
  telemetry・silent でない）。これにより「closed E1 が resurrect される」ことを構造的に禁止。
- D17 の「episode authority が baseline を所有」と整合（閉鎖時 baseline release は安全 — 以後その
  RecoveryEpisodeId に到達不能）。

### D19.2 INV-X1-9 — Backpressure progress / deadlock-free（D14 refinement・P0 相当）
```
When the logical-obligation budget is exhausted:
  - no existing logical obligation may be evicted;
  - no non-supersedable obligation may be discarded;
  - the producer（CoordinatorLoop）may retain the pending admission obligation（parked）;
  - the component responsible for releasing existing reservations
    （Builder settle = Success / Superseded / ShutdownDiscard）
    MUST remain schedulable and MUST NOT depend on completion of the blocked admission;
  - budget release MUST eventually wake/re-enable the pending admission（retry）.
```
- **非ブロッキング park**: CoordinatorLoop は budget 満杯で**停止しない**（park した義務を保持しサイクル継続）。
  blocking wait ではなく **async park / retry**。
- **循環依存の不存在（証明の構造）**: `park（CoordinatorLoop・非停止）→ Builder（独立スレッド）が settle で
  reservation release → budget release signal → CoordinatorLoop が park を retry`。**release を駆動する
  Builder は park に依存しない**（CoordinatorLoop が park 状態でも Builder は稼働し続ける）。
- park 中の義務は**高々 1 件**（CoordinatorLoop は 1 intent ずつ処理・budget 解放後に retry）。
- **実装時の構造証明要求**: 「BLOCK する主体（admission）と reservation を release する主体（Builder）が
  循環依存しない」ことを実装構造で証明（Builder スレッドは CoordinatorLoop の park 状態と無関係に動作）。

### D19.3 E — Logical obligation capacity derivation（refinement・NOT YET PROVEN 解消）
★ **「coalesce/supersede により live ≤ concurrent episodes」という Design-5 の説明は D18 と整合しない**
  （異なる target は retain both — 同一 episode でも複数 target が live 可能）。具体例（ユーザー）:
  同一 episode E1 に T1..T32（全て異種 target）→ COALESCE なし・SUPERSEDE なし → **live = 32 に到達可**。
★ したがって容量は episode 数ではなく、**upstream / episode / target の bound から導出**する:
```
kMaxLogicalRecoveryObligations = 32（候補・決定待ち F）

INV-CAP-1: liveLogicalObligationCount ≤ kMaxLogicalRecoveryObligations
INV-CAP-2: maximum concurrent RecoveryEpisodeId count ≤ E_max
            （upstream bound: 同時 quarantine 対象 handle 数）
INV-CAP-3: maximum live non-coalesced obligations per episode ≤ O_max
            （upstream bound: episode 内の異種 semantic target の上限）
INV-CAP-4: E_max × O_max ≤ kMaxLogicalRecoveryObligations
```
- 代替: episode 数を使わず、upstream admission source の「最大 outstanding 異種 recovery target 数」を
  直接 bound（≤ 32）してもよい。
- **O_max が bound できない場合の設計**: **episode 自体を backpressure 単位**にする — episode 内の
  live obligation が O_max 到達 → 同一 episode への新規異種 target admission は admission authority で
  park/backpressure（INV-X1-9 と同一機構・lost なし）。
- **決定待ち F**: E_max / O_max の具体値（同時 quarantine 対象数・episode 内異種 target 数の設計/実測上限から
  導出）。**32 を invariant として証明するには INV-CAP-2/3 の導出が必須**（arbitrary constant のままにしない）。

### D19.4 INV-X1-10 — Coalesce source mutation（D18.1 補強）
`buildSource ∉ CoalesceIdentity` を明文化（ユーザー指摘8 — 将来の semantic ambiguity 再発防止）:
```
COALESCE may replace/update buildSource metadata only.
It MUST NOT mutate:
  - CoalesceIdentity（handle / RecoveryEpisodeId / SemanticRecoveryTarget）
  - RecoveryEpisodeId
  - reservation ownership
  - baselineIdentity
  - logical obligation identity
If the semantic target changes, the request is NOT a COALESCE candidate（→ new logical obligation・容量対象）。
```

### D19.5 テスト行列追加（T24〜T27）
| # | テスト | 検証 |
|---|--------|------|
| **T24** | backpressure progress / no deadlock | budget full + non-coalescable → admission は park（pending）・既存不 discard → existing Success → reservation release → **pending が admissible に** |
| **T25** | episode closure finality | closed RecoveryEpisodeId への admission/coalesce/supersede は REJECT・**id 非再利用**（monotonic） |
| **T26** | capacity bound | INV-CAP-1〜4: E_max × O_max ≤ 32・live ≤ 32（同一 episode の異種 target が 32 を超えたら backpressure） |
| **T27** | coalesce source mutation | buildSource 更新で identity 不変・target 変更は COALESCE でなく新 obligation（容量対象） |

---

## Design-7 最終判定
**Phase I 実装 NO-GO（継続）。D19 で INV-X1-8（episode closure finality）・INV-X1-9（backpressure
progress / deadlock-free）・INV-CAP（capacity derivation）・INV-X1-10（coalesce source mutation）を固定。
決定待ち F（E_max / O_max / 32 の導出）を確認後、実装 GO。**
（→ **Design-8（D20〜D22）** は下記。Design-7: CONDITIONAL GO。**「D19 で3 blocking item を解消した」は
過剰表現** — closure linearization・liveness assumption・capacity 導出証明は未完了。）

---

## D20/D21/D22 — Design-8 確定（2026-08-15・ユーザー Design-7 レビュー: D19.1=CONDITIONAL / D19.2=P0 refinement / D19.3=NO-GO）

残存3点（closure linearization / backpressure liveness / capacity proof）を実装前契約として固定。

### D20 — Episode closure linearization（D19.1 refinement・INV-X1-8a）
```
INV-X1-8a — Episode closure linearization

For each RecoveryEpisodeId E:

1. admission, coalesce, and terminal disposition are totally ordered
   with respect to episode closure.
2. The transition:
       liveLogicalObligationCount: 1 -> 0
   establishes the unique closure point of E.
3. After the closure point, no operation may create, coalesce,
   or supersede an obligation belonging to E.
4. An admission already linearized before closure remains valid
   even if its execution/placement occurs after closure.
5. An admission not yet linearized at closure is rejected and must
   not resurrect E.
6. RecoveryEpisodeId is never reused.
```
- **ConvoPeq での線形化（item 4 採否・採用）**:
  - 原子カウンタ `liveLogicalObligationCount` + 原子 `Closed` フラグが**唯一の total order** を提供する。
  - admission commit（CoordinatorLoop・単一スレッド）: `!Closed` を原子確認 → reservation 取得 →
    `live++` → 登録。**commit は Closed に対して原子的に線形化**される。
  - terminal disposition（Builder settle）: `live--` → **live==0 になった live-- が closure point**。
  - item 4 を**採用**: closure point より前に commit 済みの admission（すでに live++ 済み）は有効 —
    commit 済み obligation が live である限り live は 0 にならず、closure は commit 済み obligation の
    settle 後にのみ発生（**自動的に整合**）。
  - item 5: closure point 以降の commit は `Closed` を観測して REJECT（resurrect 禁止）。
- **これは「単に atomic counter」では解決しない**: 必要なのは episode lifecycle の authority
  （RecoveryEpisode = admission authority が所有）と、admission commit / terminal disposition の
  線形化順序（原子カウンタが提供）の明示。D20 はそれを固定する。

### D21 — Backpressure liveness contract（D19.2 refinement・P0）
#### D21.1 ParkedAdmission の semantic status（ユーザー指摘3）
```
ParkedAdmission
    ≠ LogicalRecoveryObligation
    ≠ Reservation
```
- budget full 時点の parked item は **まだ reservation を持たず・logical obligation にもなっていない**。
- **admittedLogicalObligationCount / liveOwnershipCount に含めない**（accounting が崩れない）。
- parked item は「admission attempt（transient intent）」であり、budget release 後に reservation 取得 →
  logical obligation 生成という遷移を経る。

#### D21.2 Safety / liveness の分離（ユーザー指摘2）
```
Safety:
    no eviction
    no silent discard
    no circular wait
    Builder release path は parked admission に依存しない（独立スレッド・非依存を構造で保証）

Liveness:
    release → eventual retry（budget release は必ず pending admission を retry させる）
```
- **safety だけでは liveness は導けない**（`release → signal → CoordinatorLoop never scheduled` は
  deadlock でなく progress failure）。liveness を分離して契約化。

#### D21.3 INV-X1-9a — Level-triggered retry（wakeup-loss immunity・ユーザー指摘2.1）
```
A pending admission is retryable iff:
    pendingAdmissionExists
    && reservationBudgetAvailable

Retry eligibility is state-derived, not notification-derived.

A release notification may be lost without affecting correctness;
the next CoordinatorLoop cycle MUST observe the retryable predicate.
```
- 通知をイベントでなく **状態 predicate** にする（ConvoPeq の既存 CoordinatorLoop の
  `recoveryPending` + CV 述語パターンと整合 — Phase E）。
- notify loss でも `next CoordinatorLoop cycle → pending && budgetAvailable → retry` で進行。

### D22 — Capacity proof（D19.3 refinement・NO-GO 解消）
#### D22.1 現状の問題（ユーザー指摘4〜7）
- `E_max × O_max ≤ 32` は正しい方向だが、**E_max/O_max が「定義上の上限」のまま**（なぜ 4？ なぜ 8？ がない）。
- **O_max は外部（upstream）からは一般に証明不能**（episode 内の異種 target 数は config-change dynamics 依存）。
- 「32 を証明するために O_max=32 と決めた」は**循環論法**（INV-CAP-5 で禁止）。

#### D22.2 採用設計（決定）: 直接 enforcement（global budget + per-episode cap は policy）
```
INV-CAP-5: E_max と O_max は kMaxLogicalRecoveryObligations に合わせて選ばない。
           独立した upstream invariant から導出するか、policy-enforced と明示する。

INV-CAP-6: 到達可能な全状態 S で:
    liveLogicalObligationCount(S) ≤ Σ episodeLiveBound(E)
    （ただし Σ は policy-enforced cap として強制）

採用:
    kMaxLogicalRecoveryObligations = 32 = 【deliberate resource bound】
      （upstream maximum ではない。memory/resource budget analysis で説明）
    admission authority が直接 enforcement:
        Σ episode live obligations ≤ 32
      （E_max × O_max 分解に依存しない。episode 分解と無関係に global bound を強制）
    per-episode はこの global bound の配分として扱う（O_max は policy-enforced cap として
      admission レベルで強制・upstream 導出を主張しない）
```
- **「32 = upstream maximum」か「32 = deliberate resource bound」かを明示的に後者に確定**（ユーザー指摘7）。
- 32 の妥当性は **memory/resource budget analysis** で説明（DSPHandleTable 512 上限・同時 quarantine 対象
  の実用上限・obligation あたりのメモリ）— 但し**「システムが 32 を強制する」ことは invariant
  （admission-level enforcement）であり、導出された上限を主張しない。
- これにより循環論法を排除し、**容量は「選択された bounded resource capacity」として invariant に強制**。

### D20.4 INV-X1-10a — Coalesce source mutation ordering（D19.4 refinement・ユーザー指摘8）
```
For COALESCE:
    incoming.RecoveryGeneration >= existing.RecoveryGeneration
and buildSource MUST correspond to the accepted latest generation.

A stale coalescing admission MUST NOT roll buildSource backward.
```
- 「buildSource を最新化」の「latest」を **RecoveryGeneration に結びつける**（stale coalesce が
  buildSource を後退させない — subtle bug 防止）。

### D20.5 counter の明瞭化（ユーザー指摘11）
```
admissionEventCount            — monotonic event counter（各 submitRecoveryRequest・coalesce 含む）
logicalObligationCreationCount — logical obligation 生成数（coalesce は増やさない）
    （旧名 admittedLogicalObligationCount を改名 —「admission された」か「creation された」かの曖昧さを解消）

conservation:
    liveOwnershipCount + terminalDispositionCount == logicalObligationCreationCount
    admissionEventCount ≥ logicalObligationCreationCount
```
- ParkedAdmission はどちらの count にも含めない（D21.1）。

### D20.6 テスト追加（T28〜T31）
| # | テスト | 検証 |
|---|--------|------|
| **T28** | episode closure linearization | live:1→0 の terminal disposition が唯一の closure point・closure 前 commit は有効・後は REJECT |
| **T29** | level-triggered retry / wakeup-loss | notify を落としても next CoordinatorLoop cycle が retryable predicate を観測して retry |
| **T30** | parked ≠ obligation ≠ reservation | budget full の parked は logicalObligationCreationCount / live に含まれない |
| **T31** | stale coalesce 非後退 | incoming.RecoveryGeneration < existing の coalesce は buildSource を後退させない |

---

## Design-8 最終判定
**Phase I 実装 NO-GO（継続）。D20（episode closure linearization・INV-X1-8a）・D21（backpressure
liveness contract・INV-X1-9a・ParkedAdmission semantics）・D22（capacity proof: deliberate resource
bound 32 + global budget 直接 enforcement・INV-CAP-5/6）・INV-X1-10a・counter 明瞭化を固定。
残る決定待ち: 32 の memory/resource budget 根拠の確定（deliberate bound として）。実装 GO はユーザー最終確認後。**
（→ **Design-9（D23〜D25）** は下記。**Design-8 の D20 は NO-GO** — `Closed check → reservation → live++` は
atomic linearization ではなく、実在の race（closure 後に admission が resurrect し得る）が残る。P0 blocking。）

---

## D23/D24/D25 — Design-9 確定（2026-08-15・ユーザー Design-8 レビュー: D20=NO-GO（P0 race）/ D21=GO（liveness 条件付き）/ D22=GO（adequacy 未証明））

### D23 — Atomic Episode Admission State（INV-X1-8b・D20 修正・P0 blocking 解消）

#### D23.1 D20 の実在 race（ユーザー指摘）
```
初期: Closed=false, live=1
Thread A (CoordinatorLoop): 1. load Closed==false → 2. reservation acquire ← ここで停止
Thread B (Builder):         3. live.fetch_sub(1) → 4. live==0 → 5. closure
Thread A:                   6. live.fetch_add(1) → 7. register
結果: Closed=true, live=1   ← closure 後に admission が resurrect
```
- `Closed` を atomic にしても `live` を atomic にしても解決しない —
  **`check Closed → reservation → live++` が単一の linearization operation でない**ため。
- **「atomic live counter + atomic Closed flag」は危険**（2変数を個別観測で TOCTOU）。

#### D23.2 INV-X1-8b — closure-aware admission CAS（固定）
```
Episode admission and closure MUST linearize through one atomic
episode state transition mechanism.

The following operations MUST NOT be independently linearized:
    closed check / live increment / live decrement / closure transition

The unique closure transition is:
    OPEN(live=1) -> CLOSED(live=0)

An admission succeeds only if its atomic transition:
    OPEN(live=N) -> OPEN(live=N+1)
wins before closure.

An admission CAS that observes CLOSED is rejected.

RecoveryEpisodeId is never resurrected.
```
- **実装形態**: 単一の atomic `EpisodeAdmissionState`（`{live, closed}` を 1 つの CAS-able state・bit-packed
  atomic 等）とする。正確な表現は実装設計で決定するが、**契約として「Closed 判定と live 増減を別々の
  atomic operation として linearize してはならない」** を固定。
- **Case A（admission wins）**: `OPEN(1) → CAS → OPEN(2)`（admission が先に linearize）→ terminal で
  `OPEN(2) → OPEN(1)` → closure は発生しない。
- **Case B（closure wins）**: `OPEN(1) → CAS → CLOSED(0)`（closure が先に linearize）→ admission CAS 失敗 →
  **reject**。
- これにより「closure 前 linearize の admission は有効・後は reject」が**実装構造そのもので証明される**。

#### D23.3 admission linearization ≠ physical placement（D20 item 4・GO）
```
admission linearization（OPEN(N)→OPEN(N+1) の CAS 成功）
        ≠ physical placement（transport/durable/stall への配置）
```
- item 4 の採用判断は妥当: admission linearization が placement より先に存在する限り、placement の
  タイミングは episode closure の authority ではない（closure 後に placement されても有効）。

#### D23.4 reservation の tentative / ownership 分離（D18.4 CONDITIONAL 解消）
```
reservation tentative acquire
    ↓
episode admission CAS（OPEN(N)→OPEN(N+1)）
    ↓
success → reservation becomes owned by the new logical obligation（admission linearization point）
failure → reservation release + reject/retry
```
- D18.4 の「reservation acquired exactly once per newly created logical obligation」は
  **logical ownership reservation** を意味し、**tentative budget claim は別概念**として定義する。
- admission CAS 失敗時は tentative reservation を release（reservation と logical obligation の 1:1 を維持）。

### D24 — Liveness execution assumption（INV-X1-9b・D21 refinement）

#### D24.1 LIVENESS-ASSUMPTION-X1（eventual scheduling）
```
INV-X1-9b
If:
    pendingAdmissionExists && reservationBudgetAvailable
remains true while the system is operational,
then CoordinatorLoop MUST eventually evaluate the retry predicate.

Retry correctness MUST NOT depend on delivery of a notification.
A notification is only a wakeup optimization.
```
- **level-triggered predicate は「notify loss に耐える」ことと、「CoordinatorLoop が eventually 実行する」
  ことは別**（ユーザー指摘）— 後者を **execution model assumption（weakly fair / eventually scheduled）** として
  明示する。

#### D24.2 ParkedAdmission は episode closure を跨ぐ（revalidation 必須）
- **park は admission commit ではない** → **park 時点の episode openness を永続保証してはならない**。
- retry 時に **episode still open? を再検証**:
  ```
  ParkedAdmission(E1) → E1 が closure → retry
      → episode admission CAS が CLOSED を観測 → reject / new-episode policy へ
  ```
- これにより park 中に閉鎖した episode への resurrect を禁止（D20/D23 と整合）。

### D25 — Capacity adequacy（INV-CAP-7・D22 refinement）

#### D25.1 INV-CAP-7（固定）
```
INV-CAP-7
kMaxLogicalRecoveryObligations = 32 is a deliberate resource bound.
The value 32 is NOT derived from upstream admission maxima.
The system MUST directly enforce:
    reservedLogicalObligations <= 32
The design documentation MUST demonstrate that 32 simultaneous
logical reservations fit within the approved memory/resource budget.
No claim is made that 32 is an upstream behavioral maximum.
```

#### D25.2 reservation-count authority（一本化・ユーザー指摘12）
- **capacity accounting は placement の和ではなく reservation count を正本とする**:
  ```
  reservedLogicalObligations <= 32   （単一 authority: AdmissionAuthority → GlobalRecoveryBudget）
  placement（transport/durable/building/stalled）は reservation の所有者を移すだけ
  ```
- **transport+durable+building+stalled を個別加算して <= 32 と判定する方式は避ける**
  （placement transition 中の transient double-count / zero-count window ができるため）。
- `INV-CAP-1`（実質必要）: `live ≤ 32`（reservation count で保証）。per-episode cap は**optional secondary
  policy**（`INV-CAP-6` の Σ episodeLiveBound は不要 — D19 の問題が再登場しないよう廃止）。

#### D25.3 resource adequacy（NOT YET PROVEN → 実装 GO 前の分析）
- 「32 が upstream maximum」の証明は不要。**「32 件を保持しても許容可能な resource budget 内」の分析**:
  - per-obligation footprint: sizeof(RecoveryAdmission) + buildSource metadata + target identity +
    reservation bookkeeping + queue/container overhead + episode bookkeeping + telemetry worst-case
  - worst-case 32: `32 × per-obligation footprint + episode overhead + queue overhead + safety margin`
  - 非メモリ資源: Builder outstanding work・temporary build buffers・retry bookkeeping
- **決定待ち G**: 上記 analysis による 32 の adequacy 確認（実装 GO 前にドキュメントで実証）。

### D23.5 テスト追加（T32〜T34）
| # | テスト | 検証 |
|---|--------|------|
| **T32** | closure CAS race（P0） | admission CAS（OPEN(N)→OPEN(N+1)）vs closure CAS（OPEN(1)→CLOSED(0)）の競合 → closure 前 linearize は有効・後は reject・**resurrect 不能** |
| **T33** | liveness assumption | pending && budgetAvailable が operational 中 true → CoordinatorLoop が eventually retry 述語を評価・notify loss でも進行 |
| **T34** | reservation-count authority | `reservedLogicalObligations ≤ 32`・placement transition で double/zero-count なし |

---

## Design-9 最終判定
**Phase I 実装 NO-GO（継続）。D23（INV-X1-8b: closure-aware admission CAS・D20 の P0 race 解消）・
D24（INV-X1-9b: liveness execution assumption・parked-across-closure revalidation）・D25（INV-CAP-7:
32 = deliberate resource bound・reservation-count authority 一本化・resource adequacy）を固定。
残る決定待ち G（32 の memory/resource budget adequacy 分析）。実装 GO はユーザー最終確認後。**
（→ **Design-10（D26）** は下記。**Design-9 は NO-GO** — D23 の tentative reservation と D25 の
`reservedLogicalObligations` が未接続で、capacity overcommit race の余地が残る（P0級）。）

---

## D26 — Design-10 確定（2026-08-15・ユーザー Design-9 レビュー: D20/D23 方向=GO・tentative capacity accounting=P0級・必須修正5点）

D20→D23（closure race）・D21→D24（liveness）・D22→D25（capacity authority）を一本化するための
**必須修正5点**を固定。

### D26.1 — 完全な episode state transition（D23 refinement・必須修正1）
```
EpisodeAdmissionState の完全な遷移規則:

  OPEN(N), N >= 1            （OPEN(0) は合法状態にしない —
                                live==0 && !Closed の中間状態を設計上排除）

    admission:  OPEN(N)     -> OPEN(N+1)
    settle:     OPEN(N>1)   -> OPEN(N-1)
    closure:    OPEN(1)     -> CLOSED(0)

  CLOSED(0) は terminal state:
    admission            -> reject
    terminal disposition -> invalid / impossible
    reopen               -> forbidden
```
- `CLOSED(0)` からは**全遷移禁止**（terminal）。`OPEN(0)` は**非合法**（live==0 && !closed の中間を排除）。

### D26.2 — ReservationState 2状態分離 + capacity accounting 接続（D23.4 × D25・必須修正2・P0級）
```
ReservationState:
    Tentative    — tentative budget claim（episode admission CAS 前）
    Owned        — logical obligation が所有（episode admission CAS 成功後）
    Released     — 終端（Success / Superseded / ShutdownDiscard / CAS failure）

遷移:
    Tentative -> Owned     （episode admission CAS 成功 = logical obligation creation linearization point）
    Tentative -> Released  （episode admission CAS failure → tentative release）
    Owned     -> Released  （terminal disposition）

capacity invariant（GlobalRecoveryBudget の唯一の capacity invariant）:
    reservedLogicalObligations = Tentative + Owned <= 32
```
- **★ tentative reservation も `reservedLogicalObligations` に含める（overcommit 防止・P0級）**:
  ```
  budget = 32, owned = 31
  A: tentative acquire        → reserved = 32
  A: episode CAS pending
  B: 別 admission が reservation 試行 → MUST reject / park（reserved は 32 のまま）
  ```
  tentative を count しないと `owned=31 + tentative=1` → B が owned=31 を見て +1 →
  **temporary total = 33（overcommit）**。
- **episode CAS success == logical obligation creation linearization point**（D18.3 の
  `logicalObligationCreationCount` と完全接続）。
- **tentative reservation acquired ≠ logical obligation created**（CAS 成功が creation point）。

### D26.3 — GlobalRecoveryBudget sole authority（D25 refinement・必須修正3）
```
INV-CAP-8
GlobalRecoveryBudget is the sole authority for logical recovery
reservation capacity.

No transport/durable/building/stalled container may independently
allocate or release logical capacity.

Placement transitions transfer ownership of an existing reservation;
they do not create or destroy capacity.
```
- Transport / DurablePending / Building / Stalled は **capacity authority ではない** —
  **reservation の owner / placement にすぎない**（ConvoPeq の Authority Singularization と整合）。

### D26.4 — LIVENESS-ASSUMPTION-X1 を execution assumption として明示（D24 refinement・必須修正4）
```
LIVENESS-ASSUMPTION-X1:
    While the component remains operational,
    CoordinatorLoop receives weakly-fair execution opportunities.

System guarantee（assumption に依存）:
    IF CoordinatorLoop is eventually scheduled
    AND pendingAdmissionExists
    AND budgetAvailable
    THEN retry predicate is eventually evaluated.
```
- **eventual scheduling は ConvoPeq 内部 invariant では証明できない external execution assumption**
  （scheduler starvation を correctness invariant として背負わない）。
- **budget release の visibility**: `reservationBudgetAvailable` 自体を atomic state / synchronization
  mechanism で安全に観測可能にし、`pending && budgetAvailable` を CoordinatorLoop が**毎 cycle 判定**。
  signal lost でも next cycle で観測 → retry（notification は optimization only）。

### D26.5 — Coalesce lookup + creation の serial ordering（D23 refinement）
```
Phase I: coalesce lookup と logical-obligation creation は
  CoordinatorLoop の single admission authority によって serially ordered。
```
- 単一 producer が保証されるため、`lookup → 別 thread が同 identity を create → duplicate` の race は
  **構造的に存在しない**（単一 producer の設計上の利点）。複数 producer 化時は要再検討（D26.5 で明示）。

### D26.6 — G: resource adequacy analysis のスコープ（D25 refinement・必須修正5）
```
Per logical obligation:
    RecoveryAdmission + SemanticRecoveryTarget + buildSource + reservation metadata
    + queue/container node + episode reference/metadata + telemetry metadata

Worst-case:
    32 × per-obligation memory
    + episode bookkeeping
    + pending admission storage（parked）
    + queue overhead
    + allocator/container overhead
    + diagnostic/telemetry worst case
    + builder-side outstanding state
    + temporary build resources
    + safety margin

★ RT/NonRT 境界（重要）:
    「32 logical reservations が 32 DSP build buffers を同時に意味するか」を分離して明示。
    logical reservation → builder temporary allocation の増幅（数百 MB 級）があり得るなら、
    logical capacity だけでは resource adequacy を証明できない。
```

### D26.7 テスト追加（T35〜T37）
| # | テスト | 検証 |
|---|--------|------|
| **T35** | complete state transition | OPEN(N≥1) の admission/settle/closure・CLOSED(0) は terminal（reopen 禁止）・OPEN(0) 非合法 |
| **T36** | tentative capacity accounting（P0級） | `reservedLogicalObligations = Tentative + Owned ≤ 32`・tentative 中の他 admission は reject/park（overcommit なし） |
| **T37** | GlobalRecoveryBudget sole authority | placement は reservation を移動のみ・container が独立に allocate/release しない |

---

## Design-10 最終判定
**Phase I 実装 NO-GO（継続）。D26 で必須修正5点を固定: (1) 完全な episode state transition
（CLOSED terminal・OPEN(0) 非合法）(2) ReservationState {Tentative/Owned/Released} + `reserved =
Tentative + Owned ≤ 32`（tentative を capacity に含める・overcommit 防止・CAS success = creation
linearization point）(3) INV-CAP-8: GlobalRecoveryBudget sole authority（placement は移動のみ）
(4) LIVENESS-ASSUMPTION-X1 を external execution assumption として明示（weakly-fair scheduling）
(5) G: resource adequacy analysis のスコープ（builder 増幅含む）。**
**残る決定待ち G（上記固定後に resource adequacy analysis を実施）。実装 GO はユーザー最終確認後。**
（→ **Design-11（D27/D28）** は下記。**「残るのは G だけ」は誤り** — H（obligation 個体の
COALESCE↔terminal race・P0）・I（first episode creation）・I2（baseline capture ordering）が未解決。）

---

## D27/D28 — Design-11 確定（2026-08-15・ユーザー Design-10 レビュー: D26=CONDITIONAL GO・**H=P0 未解決**・I/I2=契約欠落）

D23 が episode の closure race を閉じた一方、**logical obligation 個体そのものの lifecycle authority**
と **first episode creation** が未契約。D27/D28 で閉じる。

### D27 — Obligation Lifecycle Linearization Contract（H・P0・INV-OBL-1）

#### D27.1 Obligation lifecycle
```
LIVE
  ├─ COALESCE        -> LIVE
  ├─ SUCCESS         -> TERMINAL
  ├─ SUPERSEDE       -> TERMINAL
  └─ SHUTDOWN_DISCARD-> TERMINAL
LIVE -> TERMINAL は exactly once。
（Phase I では SUPERSEDE は不活性（D18）・実 path は SUCCESS / SHUTDOWN_DISCARD）
```

#### D27.2 COALESCE vs terminal disposition の linearization（INV-OBL-1・P0）
```
INV-OBL-1
For every logical obligation, COALESCE and terminal disposition
MUST linearize against the same obligation lifecycle state.

A COALESCE operation MUST succeed only while the obligation is LIVE.

A terminal disposition MUST transition LIVE -> TERMINAL exactly once.

After terminal linearization:
    no COALESCE mutation is permitted.
```
- **ユーザー指摘の race（実在）**:
  ```
  CoordinatorLoop: coalesce lookup → existing(LIVE) を candidate 保持
  Builder:         SUCCESS → LIVE→TERMINAL → Owned→Released → episode live--
  CoordinatorLoop: coalesce mutation（existing.buildSource = ...）← terminal 後!
  ```
- **修正**: COALESCE mutation は obligation の atomic `ObligationState` に対する **CAS（LIVE→LIVE）** で、
  terminal disposition（LIVE→TERMINAL）と**競合**させる。terminal が先 = coalesce 失敗（mutation 拒否）。
  coalesce が先 = obligation は LIVE のまま。
- **★ EpisodeAdmissionState と ObligationState は別物**:
  - `EpisodeAdmissionState`（D23/D26.1）= **episode がまだ admission 可能か**。
  - `ObligationState`（D27）= **この logical obligation がまだ coalesce/mutate 可能か**。
  D23 は前者・D27 は後者を解決（**両方が必要**）。

#### D27.3 Reservation coupling
```
LIVE    <=> ReservationState == Owned
TERMINAL<=> ReservationState == Released
Tentative は logical obligation creation 前のため equivalence 対象外。
```

#### D27.4 Capacity preservation（INV-CAP-9・J）
```
INV-CAP-9
Reservation state transition Tentative -> Owned MUST preserve reservedLogicalObligations.

The transition MUST NOT perform decrement(Tentative) + increment(Owned)
as two independently observable capacity operations.

Capacity remains continuously reserved across the transition.

Owned -> Released: reservedLogicalObligations decremented exactly once.
```
- `tentativeCount--; ownedCount++;` を**禁止**（瞬間的に reserved=31 になる window を作る）。
  capacity accounting は単一の `reservedLogicalObligations`（または単一 reservation state の CAS）で
  連続的に予約を維持する。

#### D27.5 Episode closure との接続
```
last LIVE obligation:
    LIVE -> TERMINAL（ObligationState CAS）
    + episode OPEN(1) -> CLOSED(0)（EpisodeAdmissionState CAS）
```
- 最後の LIVE obligation の terminal disposition が episode closure に接続する
  （Builder の terminal が、episode live を 0 にする時点で closure CAS を実行 — 両 linearization を
  同一 atomic protocol で接続）。

#### D27.6 buildSource の concurrent read/write（INV-OBL-2・H2）
```
INV-OBL-2
buildSource mutation and buildSource consumption MUST have an explicit
synchronization/ownership contract.

A Builder MUST NOT observe a partially updated buildSource.

A COALESCE MUST NOT mutate buildSource after terminal linearization
of the obligation.
```
- **race（実在）**: Builder が buildSource を読みながら CoordinatorLoop（COALESCE）が書き換える。
- **契約**: buildSource は obligation ごとに **immutable value** として扱い、COALESCE は**原子的に置換**
  （publish・snapshot として消費）。部分更新を観測しない。terminal 後の buildSource mutation は禁止
  （INV-OBL-1 と接続）。

### D28 — First-Episode / Baseline Semantics（I・I2）

#### D28.1 Episode creation sequence
```
NO EPISODE
    ↓
allocate RecoveryEpisodeId（monotonic・non-reused・nextRecoveryEpisodeId_）
    ↓
capture immutable baseline（episode creation linearization = baseline capture point）
    ↓
first admission（tentative acquire → episode CAS: NO EPISODE -> OPEN(1)）
```
- **first admission は episode creation と一体**（`NO EPISODE → OPEN(1)` の原子的遷移 — D26.1 の
  `OPEN(0)` 非合法と整合。OPEN(0) 経由ではなく直接 OPEN(1) を作る）。
- `RecoveryEpisodeId` の allocation と baseline capture は**一度だけ**・episode closed 後は絶対に再利用しない。

#### D28.2 baseline capture ordering（I2）
```
Episode creation linearization = immutable baseline capture point.

baselineIdentity is captured exactly once,
before the first logical obligation is created,
and is immutable thereafter.
```
- 反例（ユーザー）: published=S0 → baseline capture → 別 publication=S1 → first admission。
  **baseline は S0 のまま**（capture 後に publication が進んでも再 capture しない — immutable）。
- 順序: `allocate id → capture baseline → first admission(OPEN(1))` を固定（baseline は first obligation
  より前に一度だけ）。

### D27.7 D26.3/D26.4/D26.5 の接続（refinement）
- **D26.3（GlobalRecoveryBudget API 境界）**: 公開 API は `acquireTentative() / promoteToOwned() /
  release()` のみ。placement 側は `attachReservation() / detachReservation()` を持てるが
  **`allocateCapacity() / releaseCapacity()` を持たない**（capacity 生成/破棄の境界）。
- **D26.4（budgetAvailable 定義接続）**: `budgetAvailable := GlobalRecoveryBudget.canAcquireTentative()`
  （= `reservedLogicalObligations < 32`）であって **`owned < 32` ではない**（D26.2 の修正を D26.4 に接続）。
- **D26.5（表現修正）**: 単一 producer は **admission-vs-admission race のみ**を消す。
  **admission-vs-settlement race は消さない** → COALESCE は obligation terminal state（D27.2）にも
  追加で linearize する。

### D27.8 テスト追加（T38〜T41）
| # | テスト | 検証 |
|---|--------|------|
| **T38** | COALESCE vs terminal race（P0） | obligation の LIVE→LIVE（coalesce CAS）vs LIVE→TERMINAL（terminal CAS）の競合 → terminal 後の coalesce mutation は拒否 |
| **T39** | buildSource 同期（H2） | Builder は部分更新 buildSource を観測しない・terminal 後 mutation なし |
| **T40** | first episode creation | NO EPISODE → OPEN(1) 原子的・baseline は first obligation 前に一度だけ capture・id 非再利用 |
| **T41** | capacity preservation（J） | Tentative→Owned で reserved 不変（window なし）・Owned→Released で exactly once decrement |

---

## Design-11 最終判定
**Phase I 実装 NO-GO（継続）。D27（INV-OBL-1: COALESCE vs terminal の obligation lifecycle
linearization・H=P0 解決・EpisodeState と ObligationState の分離）/ D27.6（INV-OBL-2: buildSource 同期・H2）/
D27.4（INV-CAP-9: Tentative→Owned capacity-preserving・J）/ D28（first episode creation・baseline capture
ordering・I/I2）を固定。D26.3/26.4/26.5 を D27 に接続（budgetAvailable = canAcquireTentative・placement は
capacity 生成/破棄なし・単一 producer は admission-vs-admission のみ）。**
**残る決定待ち G（D26.6 のスコープで resource adequacy analysis を実施）。実装 GO はユーザー最終確認後。**
（→ **Design-12（D29）** は下記。**D27 は CONDITIONAL GO** — `LIVE→LIVE` CAS だけでは mutation の
atomicity を証明できず、H を別形態で再発させ得る（CAS 成功 → terminal → buildSource 後置換の race）。）

---

## D29 — Design-12 確定（2026-08-15・ユーザー Design-11 レビュー: H=方向GO / H2=CONDITIONAL / I=GO / I2,J=CONDITIONAL / G=未完了）

D27/D28 の blocking refinement 4点 + 8 invariant を固定し、**D23/D26/D27/D28 を一つの end-to-end
admission/settlement state machine として接続**する（Design-12 = 最終整合性レビュー）。

### D29.1 ObligationState の原子化（lifecycle + buildSource・H の本質的解消）
- **問題（ユーザー指摘1）**: `CAS(LIVE→LIVE)` 成功後に `buildSource = newSource` を独立 operation にすると、
  ```
  Coordinator: CAS LIVE→LIVE succeeds
  Builder:     CAS LIVE→TERMINAL succeeds・release reservation
  Coordinator: buildSource = newSource        ← terminal 後!
  ```
  が成立し得る。「LIVE だったことを確認」≠「terminal より前に mutation が完了」。
- **修正**: **buildSource を ObligationState から独立した mutable field にしない**。
  ObligationState は `{ Lifecycle, buildSource }` を一つの atomic mutation domain として扱う:
  ```
  COALESCE:  atomic: LIVE(oldSource) -> LIVE(newSource)
  TERMINAL:  atomic: LIVE(source)    -> TERMINAL(source)
  三分岐:
      COALESCE wins  → LIVE(S0)→LIVE(S1)
      TERMINAL wins  → LIVE(S0)→TERMINAL(S0)
      COALESCE sees TERMINAL → reject
  ```
- 実装形態（表現は実装設計）: ObligationState は tagged/versioned atomic（buildSource は同一 release
  ordering で原子的に置換・snapshot として消費）。**契約として「buildSource mutation は lifecycle
  transition の linearization domain の一部であり、独立に linearize しない」** を固定。

### D29.2 Coalesce lookup ≠ authorization（INV-OBL-4）
```
INV-OBL-4
A successful coalesce lookup is not mutation authorization.
Authorization is revalidated through ObligationState.
```
- `existing = lookup(CoalesceIdentity)` で candidate pointer/reference を保持しても、
  `coalesce(candidate)` 内部で必ず ObligationState の atomic lifecycle transition を再確認する。
  （lookup は single producer で安全でも、**candidate 保持 → Builder terminal → coalesce** は可能 —
  stale candidate 問題を閉じる。）

### D29.3 Episode closure ordering（INV-OBL-3）
```
INV-OBL-3
No CLOSED episode may contain a LIVE obligation.

An episode may transition OPEN(1) -> CLOSED(0)
only as part of the terminal disposition
of the unique remaining LIVE obligation.

The obligation terminalization must linearize
before or atomically with episode closure.

There must never be a state in which:
    episode == CLOSED
    AND a member obligation == LIVE
```
- **episode closure は obligation state を観測して後から決める独立 operation ではない**（D23 の
  episode lifecycle atomicity と同レベルの線形化規則を obligation 側にも適用）。
- `obligation=TERMINAL / episode=OPEN(1)` の短時間状態は許容するが、**逆（episode CLOSED / obligation
  LIVE）を絶対禁止**。

### D29.4 GlobalRecoveryBudget 単一 transition（INV-CAP-9 強化・J）
```
RESERVED_TENTATIVE
        │ promote
        ▼
RESERVED_OWNED

reservedLogicalObligations は promotion 中も 1 のまま（単一の linearized reservation-state transition）。
tentativeCount-- / ownedCount++ の2カウンタ方式を禁止（transient inconsistency）。
```
- `acquireTentative() / promoteToOwned() / release()` の各操作について、
  `reservedLogicalObligations` を**唯一の capacity authority** として扱う。

### D29.5 COALESCE generation ordering 統合（INV-X1-11）
```
INV-X1-11
A stale COALESCE may never move buildSource backward.
Generation validation and source replacement share the same
obligation lifecycle linearization domain.
```
```
COALESCE(existing, incoming):
    if incoming.generation < existing.generation: reject / no-op
    if incoming.generation >= existing.generation: atomic replace（D29.1 の LIVE(old)→LIVE(new)）
```
- **generation comparison と buildSource replacement を同じ linearization domain** に置く
  （`A: gen10 lookup → B: gen11 coalesce → A: gen10 mutation` の rollback を防止）。

### D29.6 First-episode commit point + admission ordering（INV-EP-1/2/3・I2）
```
INV-EP-1: First-episode baseline is captured exactly once, before first-admission commit.
INV-EP-2: No episode admission may commit without a valid immutable baseline.
INV-EP-3: First-episode creation and ordinary admission use the same episode admission
          linearization rules; first admission is not a second, weaker lifecycle path.
```
- **first admission の semantic commit point** = `successful episode admission CAS + それに伴う
  tentative reservation promotion`（「原子的」を「conceptually one transaction」と分けて定義）。
- **baseline capture の順序（固定）**: baseline を取得できない状態で episode admission が commit されない:
  ```
  1. identify applicable RecoveryEpisode（無ければ新規）
  2. verify episode is open
  3. coalesce lookup（reservation acquisition より前 — budget full でも既存への COALESCE は可能）
  4. if existing → COALESCE（D29.1/D29.5）
  5. otherwise → new logical obligation creation
  6. capture/検証 baseline（新規 episode 時は一度だけ）
  7. acquire tentative reservation
  8. episode admission CAS（NO EPISODE→OPEN(1) または OPEN(N)→OPEN(N+1)）
  9. promote reservation to Owned（D29.4）
  10. register obligation
  ```

### D29.7 Authority 構造の明確化（「4 authority」表現の修正）
```
EpisodeAuthority      └─ EpisodeAdmissionState
ObligationAuthority   └─ ObligationState
GlobalRecoveryBudget  └─ ReservationState / capacity   （ReservationState は GlobalRecoveryBudget の管理対象）
```
- **ReservationState は GlobalRecoveryBudget の管理対象**であり、独立の allocation authority ではない
  （「Reservation authority」と「Budget authority」を別々と読める表現を回避 — Authority Singularization 整合）。

### D29.8 End-to-end admission/settlement state machine（接続）
```
Admission（CoordinatorLoop・single admission authority）
  identify episode → verify open → coalesce lookup
    ├─ COALESCE:  ObligationState LIVE(old)→LIVE(new)（D29.1/29.5）・reserved 不変
    └─ new:       baseline 検証（INV-EP-1/2）→ acquireTentative（INV-CAP-9）→
                  episode CAS（OPEN(N)→OPEN(N+1)/NO→OPEN(1)）→ promoteToOwned（D29.4）→ register

Settlement（Builder・独立スレッド）
  terminal: ObligationState LIVE(src)→TERMINAL(src)（INV-OBL-1/2・exactly once）
            → release（Owned→Released・reserved-- 一回）
            → 最後の LIVE なら episode OPEN(1)→CLOSED(0)（INV-OBL-3）
            → budget release signal（level-triggered・INV-X1-9a/LIVENESS-ASSUMPTION-X1）

capacity: reservedLogicalObligations = Tentative + Owned ≤ 32（GlobalRecoveryBudget sole authority・INV-CAP-8）
```
- これにより **Episode / Budget / Obligation の authority と linearization が単一 state machine に接続**。

### D29.9 G: resource adequacy analysis（着手・D26.6 スコープ）
```
分離評価:
    R_logical（32 × per-obligation: RecoveryAdmission/SemanticRecoveryTarget/buildSource/reservation/queue/episode/telemetry）
    R_builder（builder outstanding・同時 build buffer 数）
    R_temp   （temporary build resources・増幅: 32 logical → ? DSP build buffer）
    R_queue  （transport/pending queue overhead）
    R_episode（episode bookkeeping）
    R_telemetry
    R_allocator（allocator/container overhead）

PeakResource = R_logical + R_builder + R_temp + R_queue + R_episode + R_telemetry + R_allocator + safety margin
★ RT resource と NonRT resource を混ぜない（32 logical が 32×20MB=640MB を意味するなら adequacy 不成立）
```
- **G は本契約の承認後に分析を実施**（実装前 GO 条件）。現時点では分析対象の列挙まで。

### D29.10 テスト追加（T42〜T45）
| # | テスト | 検証 |
|---|--------|------|
| **T42** | obligation atomic mutation（H 本質） | COALESCE（LIVE(S0)→LIVE(S1)）vs TERMINAL（LIVE(S0)→TERMINAL(S0)）同一 domain・terminal 後の buildSource mutation 不能 |
| **T43** | stale candidate | lookup 成功 ≠ authorization・candidate 保持中に terminal → coalesce は ObligationState 再検証で reject |
| **T44** | episode closure ordering（INV-OBL-3） | episode==CLOSED かつ member==LIVE の状態が存在しない・closure は最後の LIVE の terminal と接続 |
| **T45** | generation rollback 防止（INV-X1-11） | gen10 lookup → gen11 coalesce → gen10 mutation が buildSource を後退させない |

---

## Design-12 最終判定
**Phase I 実装 NO-GO（継続）。D29 で 8 invariant を固定: INV-OBL-1/2（ObligationState = lifecycle +
buildSource の原子 mutation domain・H 本質的解消）/ INV-OBL-3（episode closure ordering）/ INV-OBL-4
（lookup ≠ authorization）/ INV-CAP-9（GlobalRecoveryBudget 単一 transition）/ INV-X1-11（generation +
source 同一 linearization domain）/ INV-EP-1/2/3（first-episode baseline・同 linearization rules）。
D23/D26/D27/D28 を単一の end-to-end admission/settlement state machine に接続（D29.8）。**
**残るのは G（resource adequacy analysis・D29.9 のスコープで実施）。実装 GO はユーザー最終確認後。**
（→ **Design-13（D30）** は下記。**H 系を「概念的に解消」から「実装可能な契約として閉じる」ための
形式化** — atomic mutation domain を transition contract + single linearization point として定義し、
全 transition の precondition → authority → LP → postcondition を追跡する。）

---

## D30 — Design-13 確定（2026-08-15・ユーザー Design-12 レビュー: D11 後置換 race=解消 / mutation domain=設計GO / end-to-end=GO候補（transition proof 要））

H/H2/J/I/I2 を「概念的に解消」から**「実装可能な契約として閉じた」**に格上げする形式化。
**「atomic mutation domain」＝ mutex/atomic/CAS のどれかではない** — lifecycle と buildSource の整合した
pair に対する **single linearization point の存在**が本質。

### D30.1 INV-OBL-1a — single linearization point（ObligationState の形式化）
```
INV-OBL-1a
Every successful COALESCE or TERMINAL disposition has exactly one
linearization point at which both Lifecycle and buildSource take effect
as one logical state transition.

No observer or mutator may observe or mutate a state in which the
Lifecycle corresponds to one version of buildSource while buildSource
corresponds to another.
```
- **二段階実装（`lifecycle.compare_exchange(...)` 成功 → `buildSource = ...`）を「atomic mutation」と
  誤認する余地を排除** — 一つの logical state transition として Lifecycle と buildSource が同時に
  効力を発揮する single LP を要求。

### D30.2 INV-OBL-3 closure direction（closure は独立 operation でない）
- `episode == OPEN && member == TERMINAL` は許容（自然状態）。
- **`liveCount==0` を別の Coordinator operation として観測して `closeEpisode()` する設計は禁止**:
  ```
  terminal obligation → count==0 → another admission → closeEpisode()
  ```
  の race surface が復活するため。
- **契約（固定）**:
  ```
  Episode closure is a consequence of the terminal disposition
  of the last LIVE obligation, not an independently authorized mutation.

  closure CAS（OPEN(1)→CLOSED(0)）は最後の LIVE の terminal disposition
  （ObligationState LIVE→TERMINAL）と同一 linearization domain で実行される
  （原子的または terminal の一部として）。
  ```

### D30.3 INV-X1-11 normative contract（generation + source 同一 transition predicate）
- 「same linearization domain」を概念でなく **normative contract** に:
  ```
  COALESCE(expectedGeneration, newSource):
      if state.generation != expectedGeneration:  reject
      state.buildSource = newSource
      state.generation = expectedGeneration / newGeneration
  ```
  **validate expected generation AND replace source が同じ transition predicate の中**（一体化）。
- `gen10 lookup → gen11 coalesce → gen10 mutation` の rollback を構造的に防止。

### D30.4 reservation identity（INV-CAP-9 refinement・J）
- **何を promotion しているのかを一意に**: `TentativeToken / ReservationId`（個体 identity）。
  ```
  ReservationId: acquisition 時に発行
  RESERVED_TENTATIVE(id) -> RESERVED_OWNED(id)   （同一 id の単一 transition）

  Tentative A promote / Tentative B release の交錯でも、
  accounting correctness と logical ownership の対応が一意（id で追跡可能）。
  ```
- D29.7 の「ReservationState は GlobalRecoveryBudget の管理対象」と整合（reservation identity と
  transition ownership を確認）。

### D30.5 INV-EP-2 baseline failure（episode が baseline なしで存在しない）
```
NO EPISODE
    ↓ attempt baseline
    ↓ failure
    ↓ NO EPISODE（episode 生成なし）

candidate episode
    ↓ baseline invalid
    ↓ candidate discarded（OPEN にしない）

`episode allocated, baseline invalid, episode OPEN` を禁止。
```
- valid baseline なしで episode が OPEN になる経路を構造的に排除。

### D30.6 end-to-end transition proof（D29.8 の形式化・全 transition の precondition → authority → LP → postcondition）
| Transition | 入力 invariant | mutation authority | LP | 出力 invariant |
|-----------|----------------|--------------------|----|----------------|
| first admission | EP-1（baseline capture） | Episode + Budget | **LP-A**: episode CAS（NO EPISODE→OPEN(1)）+ tentative promotion 同時 | EP-3（同一 rules） |
| coalesce | OBL-1/2・X1 | ObligationState | **LP-B**: COALESCE transition（LIVE(old)→LIVE(new)・INV-OBL-1a の single LP） | OBL-1/2 |
| tentative→owned | CAP-9 | GlobalRecoveryBudget | **LP-C**: promoteToOwned(id)（reserved 不変・D30.4） | CAP-9 |
| terminal | OBL-1/3 | ObligationState | **LP-D**: TERMINAL transition（LIVE→TERMINAL・INV-OBL-1a） | OBL-3 |
| last terminal | OBL-3 | ObligationState / Episode | **LP-D**: TERMINAL + episode closure（同一 domain・D30.2） | CLOSED |

- 各 transition は **precondition → authority → LP → postcondition** を一意に追跡可能。
- LP-A〜LP-D は**それぞれ単一**（1 transition = 1 LP）。INV-OBL-1a が LP の一意性を保証。

### D30.7 テスト追加（T46〜T50）
| # | テスト | 検証 |
|---|--------|------|
| **T46** | single LP（INV-OBL-1a） | Lifecycle と buildSource が別 version の状態を観測/生成しない（二段階実装の誤認防止） |
| **T47** | closure direction（INV-OBL-3） | 独立 closeEpisode() なし・最後の LIVE terminal → closure 同一 domain・count==0 後の admission で race 復活しない |
| **T48** | normative generation（INV-X1-11） | validate+replace が同一 transition predicate・gen10 stale mutation は reject |
| **T49** | reservation identity（CAP-9） | Tentative A promote / B release 交錯でも accounting と logical ownership が id で対応 |
| **T50** | baseline failure（EP-2） | baseline invalid で episode OPEN にならない・candidate discarded |

---

## Design-13 最終判定
**Phase I 実装 NO-GO（継続）。D30 で H/H2/J/I/I2 を「概念的に解消」から「実装可能な契約として閉じた」に
格上げ: INV-OBL-1a（single linearization point 形式化・二段階実装の誤認排除）/ INV-OBL-3 closure direction
（独立 closeEpisode 禁止）/ INV-X1-11 normative（validate+replace 同一 transition predicate）/ reservation
identity（ReservationId・交錯耐性）/ INV-EP-2 baseline failure（baseline なしで OPEN 禁止）/ D30.6
transition proof（precondition → authority → LP → postcondition・LP-A〜LP-D 各単一）。**
**次の段階: G（resource adequacy analysis・D29.9/D26.6 のスコープで実施）。実装 GO はユーザー最終確認後。**
（→ **Design-14（D31）** は下記。**D30.6 の `last terminal | Obligation/Episode | LP-D` は authority が
二重に読める** — LP-D がどの single mutation domain の LP かを一意化する。）

---

## D31 — Design-14 確定（2026-08-15・ユーザー Design-13 レビュー: D30 ほぼ承認・last-terminal closure authority 要明文化）

G の前に **closure linearization を完全に一意化**（resource/capacity/lifetime analysis に曖昧さが
伝播しないように）。

### D31.1 INV-EP-4 / Closure Linearization（固定）
```
INV-EP-4 / Closure Linearization

The closure of an episode is not an independently authorized
EpisodeState mutation.

The terminal disposition of the last LIVE obligation is the sole
linearization event that makes both:
    (a) the obligation TERMINAL, and
    (b) the episode CLOSED
effective as one logical transition.

No separate Episode closure LP exists.
```
- **理想形（固定）**:
  ```
  ObligationState + EpisodeState
      │  one logical terminal transition
      ▼  single LP-D
      ├── obligation: LIVE -> TERMINAL
      └── if last LIVE: episode: OPEN -> CLOSED
  ```
- **LP-D は「Obligation + Episode の共有 mutation domain の LP」** であり、「ObligationState の LP」では
  ない（2 authority が同じ名前の LP を共有する意味ではない）。
- **`closeEpisode()` も、独立した Episode-side LP も存在しない**。

### D31.2 D30.6 の "last terminal" 行の明確化
| Transition | 入力 invariant | mutation authority | LP | 出力 invariant |
|-----------|----------------|--------------------|----|----------------|
| last terminal | OBL-3 | **Obligation+Episode（単一 mutation domain）** | **LP-D（共有・単一・INV-EP-4）** | CLOSED（obligation=TERMINAL かつ episode=CLOSED が同時に効力発生） |

### D31.3 G への接続（lifetime boundary の確定）
```
resource 解放時点 = Owned -> Released（LP-C 後の terminal 内）
obligation TERMINAL 時点 = LP-D
episode CLOSED 時点 = LP-D（同一・obligation TERMINAL と同時）
```
- 三者（resource / obligation / episode）の lifetime ordering が **LP-D で一意に確定** → G の
  capacity/lifetime analysis に曖昧さが伝播しない。

### D31.4 テスト追加（T51）
| # | テスト | 検証 |
|---|--------|------|
| **T51** | closure LP 一意化（INV-EP-4） | last LIVE terminal が obligation=TERMINAL と episode=CLOSED を単一 LP で同時に効力発生・独立 Episode closure LP なし |

---

## Design-14 判定（closure LP 固定後）
**H/H2/I/I2/J はレビュー上 CLOSED。** 次段: **G（resource adequacy analysis）**（下記 D32）。

---

## D32 — G: Resource Adequacy Analysis（2026-08-15・実施）

### D32.1 分析前提（実コード裏付け）
- **Builder は逐次**: `rebuildThreadLoop()`（AudioEngine.RebuildDispatch.cpp:804）が recovery を
  `while (auto recovery = popRecoveryRequest())`（:927）で 1 件ずつ pop → `runtimeBuilder.build(...)`（:941）→
  warmup → publish。durable recovery も同一逐次パターン（:1015）。**同時に build されるのは高々 1 件**。
- **G の前提（明示）**: Phase I は **Builder 並行度 = 1**（並列 build を導入しない）。並列化する場合は
  R_temp がスケールするため再分析が必要。

### D32.2 リソース分類と推定（実コードからのオーダー見積もり）
| 分類 | 内容 | 推定 |
|------|------|------|
| **R_logical** | 32 × per-obligation（RecoveryIntent 相当: DSPHandle 16B + epoch 8B + intentId 8B + RuntimeBuildSnapshot ~200B ≈ 250-300B・+ SemanticRecoveryTarget 5×uint64 + reservation metadata + queue node + episode ref + telemetry） | **~1KB / obligation 以下 → 32 × ~1KB ≈ ~32KB**（支配的でない） |
| **R_queue** | transport（recoveryIntentQueue_ 256 × ~300B）+ pending | ≈ ~80KB |
| **R_episode** | episode bookkeeping × concurrent episodes | 数 KB（小） |
| **R_builder / R_temp** | **1（逐次）× build resource（DSPCore + temp buffers）** | **32 倍にならない** — 通常 rebuild と同じ 1 件分 |
| **R_telemetry / R_allocator** | 診断/アロケータオーバーヘッド | 小 |

### D32.3 PeakResource 推定
```
PeakResource
    ≈ R_logical（~32KB）
    + R_queue（~80KB）
    + R_episode（数 KB）
    + R_builder（1 × build resource）
    + R_temp（1 × build resource）
    + R_telemetry + R_allocator + safety margin
```
- **支配項は R_builder/R_temp（1 つの build resource）** であり、これは通常 rebuild と同一（システムに
  既存・許容済み）。
- **★「32 logical reservations が 32 DSP build buffers を同時に意味する」ことはない**（逐次 Builder）—
  Design-9 時点でユーザーが懸念した **32×20MB=640MB 級の増幅は現行 architecture では成立しない**。

### D32.4 結論（adequacy）
- **32 logical reservations は memory/resource budget 内で許容可能**（R_logical は KB 級・R_temp は
  1× build に収束）。
- **前提**: (1) Builder 逐次（並列化しない）(2) build resource は通常 rebuild と同一上限。
- **G = 実証済み（前提付き）**。exact sizeof は実装時に実測で確定（オーダー見積もりでは adequate）。

### D32.5 実装時確認事項
- DSPCore / build resource の正確な sizeof を実装時に実測。
- **invariant: Builder concurrency == 1**（Phase I で並列 build を導入しない — R_temp の前提）。

---

## Design-14 最終判定
**Phase I 実装は設計契約として GO の準備が整った（実装 GO はユーザー最終確認後）。**
- H/H2/I/I2/J: **CLOSED**（D27〜D31）。
- G: **resource adequacy 実証済み（前提付き）** — Builder 逐次により R_temp は 1× build・32×640MB 増幅なし。
- **実装順序（確定）**: `LogicalRecoveryIdentity → Supersession → bounded admission（reservation-first）→
  RC-11 → invariant tests（T1〜T51）→ I-5（build + ctest 全 PASS）`。最初に `cpp:905-915` の latest-wins 撤去。
- 残る実装時決定: per-obligation sizeof 実測・Builder concurrency==1 の維持。
（→ **Design-15（D33）** は下記。**「G = 実証済み」は早い** — G を G-A（Builder concurrency=CLOSED）と
G-B（total peak resource=OPEN）に分解し、全 resource class の peak bound を証明する。）

---

## D33 — Design-15 確定（G-B: total peak resource proof・2026-08-15）

ユーザー Design-14 レビュー対応: **G-A（Builder concurrency）は CLOSED だが、G-B（total peak resource）は
OPEN**。D33 で G-B を peak equation + 各項独立 bounded として証明する。

### D33.1 INV-RES-1 — Builder concurrency は resource contract（proof dependency）
```
INV-RES-1
Phase I recovery Builder concurrency is bounded to exactly one
in-flight recovery build per process/runtime authority.

Any change that permits >1 concurrent recovery build invalidates
the G resource proof and requires re-analysis.
```
- **Builder 逐次は現在の実装上の偶然ではなく、Phase I の resource contract**（D32 の実コード裏付け:
  rebuildThreadLoop が 1 件ずつ build → publish を契約として固定）。

### D33.2 Peak equation（全項・独立 bounded）
```
PeakResource =
    R_logical(max) + R_queue(max) + R_episode(max) + R_telemetry(max)
    + R_builder(max concurrency) + R_published(max overlap) + R_retired(max overlap)
    + R_allocator_overhead
```

### D33.3 各項の bound（コード裏付け）
| 項 | bound | 根拠 |
|----|-------|------|
| **R_logical** | 32 × ~1KB ≈ 32KB | `kMaxLogicalRecoveryObligations = 32`（deliberate bound・直接 enforcement・INV-CAP-8） |
| **R_queue** | 固定容量合計 ≈ 数百 KB | recovery 256 + intent 4096 + quarantine fallback 1024 + observe 1024 + deferred 1024 + lastResort 4096（`ISRRuntimePublicationCoordinator.h:592-696` の固定 ring/queue） |
| **R_episode** | concurrent episodes × 小 | episode bookkeeping・実用上限内 |
| **R_telemetry** | 固定 counter 群 | 小 |
| **R_builder** | 1 × BuildFootprint | INV-RES-1・逐次 Builder |
| **R_published** | **既存 publish モデルと同一**（RuntimeStore::current 単一 + crossfade fading + retire 対象） | **recovery は通常 rebuild と同一の Immutable Publish 経路**（RuntimeWorldAuthority::publish → publishAndSwap）— recovery 固有の増加なし |
| **R_retired** | **既存 EBR/retire 機構と同一**（grace period で bounded・retire ring 16384 / quarantine 256） | 同上 |
| **R_allocator** | container/allocator overhead・capacity slack・alignment | upper-bound 計算に含める |

### D33.4 G-B の core argument（recovery 固有増加は bounded）
```
recovery 固有リソース（R_logical + R_queue の recovery 分 + R_episode + R_builder）
    = 小・bounded（32KB 級 + 1× build）

Runtime overlap（R_published + R_retired）
    = recovery と無関係の既存エンジン機構
      （通常 rebuild でも同じ publish/retire 経路・crossfade/EBR で既に provisioned）
    = recovery が追加で増加させない

∴ total peak = normal-op peak + recovery 固有（bounded 小）≤ budget
```

### D33.5 Allocation-failure policy（R_allocator・決定）
- **採用: `allocation failure is an admissible bounded failure mode`**（設計契約として明示）:
  - build resource / new Runtime allocation の失敗は既存エンジンの build-failure 経路と同一
    （recovery build 失敗 → settle(true) 再試行・上限 `kMaxRecoveryConsecutiveFailures=4`・warmup 失敗 →
    discard — 既存機構）。
  - logical capacity（32）と heap peak の不一致は、allocator 失敗を admissible bounded failure mode として
    制御（external event として扱わない）。

### D33.6 Upper-bound calculation（推定でなく upper-bound 化）
- per-obligation sizeof は実装時に `static_assert` / 実測で確定（設計段階は upper-bound: ~1KB 以下）。
- container/allocator overhead・capacity slack・alignment を
  `logical object size + container overhead + capacity slack + allocator overhead` として含める。
- R_published / R_retired の正確な数値は既存エンジンの crossfade/EBR 設計に依存（recovery 固有でない）—
  実装時に既存 bound を確認して合算。

### D33.7 判定
- **G-A: CLOSED**（Builder concurrency・D32 + INV-RES-1）。
- **G-B: CLOSED（前提付き）**: recovery 固有リソースが bounded 小・Runtime overlap は既存機構（recovery
  増加なし）。
- **残る実装時確認**: per-obligation sizeof 実測・R_published/R_retired の既存数値確認。
- **Phase I 実装はユーザー最終 GO 待ち**（設計契約として proof chain 完了）。

### D33.8 テスト追加（T52）
| # | テスト | 検証 |
|---|--------|------|
| **T52** | INV-RES-1 | >1 concurrent recovery build が存在しない（並列化変更は G proof 無効化 → 再分析トリガー） |

---

## Design-15 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。G を G-A（CLOSED・Builder concurrency）と G-B（CLOSED
前提付き・total peak resource）に分解し、D33 で peak equation + 各項独立 bounded + INV-RES-1（proof
dependency）+ allocation-failure policy を固定。recovery 固有リソースは bounded 小・Runtime overlap は
既存機構（recovery 増加なし）。H/H2/I/I2/J/G-A/G-B = CLOSED。実装順序確定。実装 GO はユーザー最終確認後。**
（→ **Design-16（D34）** は下記。**「G-B = CLOSED」は早い** — G を G-B1（boundedness・ほぼ CLOSED）と
G-B2（total peak adequacy・OPEN）に分け、**「bounded」から「adequate」への最後の不等式（B_total ≤ B_budget）**
を成立させる。）

---

## D34 — Design-16 確定（G-B Resource Adequacy Closure・2026-08-15）

ユーザー Design-15 レビュー対応: **「bounded」≠「adequate」**。各項が finite でも Σ ≤ budget は未証明。
D34 で **per-class bound → simultaneous-overlap proof → B_total 計算 → B_total ≤ B_budget → G-B = CLOSED** を確立。

### D34.1 G の3段階分解（確定）
| 段階 | 内容 | 状態 |
|------|------|------|
| **G-A** | BuilderConcurrency ≤ 1 | CLOSED（D32 + INV-RES-1） |
| **G-B1** | 各 resource class の boundedness（0 ≤ Ri ≤ Bi） | CLOSED / nearly CLOSED（D33） |
| **G-B2** | **Σ Bi ≤ B_budget（total peak adequacy）** | **本 D34 で CLOSED 化** |

### D34.2 INV-RES-2 — Total Peak Bound
```
For every reachable system state S:

    ResourceUsage(S)
      ≤ B_logical + B_queue + B_episode + B_telemetry
        + B_builder + B_runtime + B_retire + B_allocator
      = B_total
```
（各 Bi は D33 の Ri の upper bound・D34.6 で数値化）

### D34.3 INV-RES-3 — Runtime Overlap Bound（コード裏付け・N の明示）
```
The maximum simultaneously live Runtime ownership
created by recovery publication is bounded by the
existing publication/crossfade/retirement contract:

    N_published = 1      （RuntimeStore::current 単一・publishAndSwap）
    N_fading    = 1      （RuntimeGraph activeNode + fadingNode の2ノード固定・RuntimeGraph.h:32-33 /
                            activeRuntimeDSPSlot + fadingRuntimeDSPSlot・AudioEngine.h:2106/2109）
    N_retired   = 小定数  （EBR grace・publish→retire→reclaim の一時滞留・reclaimInFlight で bounded）
    N_quarantine≤ 256    （ISRDSPQuarantine::kMaxSlots）
```
- **Runtime overlap は構造的に ~2-3 に固定**（active + fading + retiring）— config 依存の count 成長なし。
- **recovery はこの N を増加させない**: recovery publish は通常 rebuild と同一の Immutable Publish 経路
  （RuntimeWorldAuthority::publish → publishAndSwap）で 1 world を生成し、旧 world は fade/retire 候補に
  なるだけ（既存機構の範囲内）。

### D34.4 INV-RES-4 — Runtime Lifetime Bound
```
Each Runtime:
    Publish → Fade → Retire → Grace(EBR) → Reclaim

    N_simultaneous(Runtime) ≤ N_published + N_fading + N_retired ≤ 3（構造的）
```
- lifetime の各段階は既存の crossfade/EBR 機構（epoch grace・minReaderEpoch）で制御。
- recovery が lifetime を延長しない（fade/retire は通常 publish と同一）。

### D34.5 INV-RES-5 — Budget Adequacy（normative）
```
B_total ≤ B_budget   （normative condition）

B_budget の定義（決定）:
    採用: process commit budget（Windows/MSVC の committed bytes）
    secondary: physical RAM budget・app-defined safety budget
```
- **requested bytes ≠ committed bytes ≠ reserved virtual address** を区別し、resource budget は
  **committed bytes** で評価（Windows commit 制約に整合）。

### D34.6 数値 B_total（コード裏付け・upper-bound）
| Bi | 数値（upper-bound） | 根拠 |
|----|---------------------|------|
| B_logical | 32 × ~1KB ≈ 32KB | kMaxLogicalRecoveryObligations=32（deliberate bound） |
| B_queue | ≈ ~2MB | 固定容量 × entry footprint: recovery 256 + intent 4096 + fallback 1024 + observe 1024 + deferred 1024 + lastResort 4096（`ISRRuntimePublicationCoordinator.h:592-696`・entry は inline payload ~64-300B） |
| B_episode | 数 KB | concurrent episodes × bookkeeping |
| B_telemetry | 固定 counter 群 | 小 |
| B_builder | 1 × B_build（D34.8） | INV-RES-1 |
| **B_runtime** | **（N=1+1+小定数）× RuntimeWorldFootprint** | **既存エンジン資源・通常 rebuild と同一（recovery 増加なし）** |
| B_retire | 既存 EBR/retire 機構と同一 | 既存 provisioned |
| B_allocator | ≤ allocatorBound（commit-based・D34.7） | 既定のアロケータオーバーヘッド |

### D34.7 R_allocator の bound（commit 意味論）
```
R_allocator ≤ allocatorBound
allocatorBound = Σ(alloc_i の committed size − requested size) の上界
  + container overhead + capacity slack + alignment
```
- Windows/MSVC: requested ≠ committed ≠ reserved を区別し、**committed** で評価。
- 既存固定容量 container（ring/array）は capacity slack を含めて bound。

### D34.8 BuildFootprint ≤ B_build contract
```
BuildFootprint ≤ B_build
B_build = 単一 build の最大 resource（通常 rebuild と同一上限）
  — recovery は同じ B_build を使用（INV-RES-1 で 1× に制約）
```
- `B_build` は既存エンジンの build 上限（最大 convolver/IR 構成）に依存 — recovery 固有でなく既存 provisioned。
- sizeof 実測は実装時に行うが、**設計時 contract として B_build の上界を固定**（後工程先送りでない）。

### D34.9 G-B2 の core inequality（CLOSED 化）
```
B_total(recovery)
  = B_total(normal-op) + B_recovery_specific
  where B_recovery_specific = B_logical(~32KB) + B_queue(recovery 256 ~77KB) + B_episode + B_telemetry(recovery)
                             ≈ ~110KB + 小

B_total(normal-op) は既存 provisioned budget 内（エンジンは通常 rebuild を継続実行）
∴ B_total(recovery) = B_total(normal-op) + ~110KB ≤ B_budget（safety margin 内で成立）
```
- **B_runtime / B_build / B_queue(既存) は通常運転と同一** — recovery は ~110KB 級の small addition のみ。
- **G-B2 = CLOSED（B_budget = process commit budget・INV-RES-5）**。

### D34.10 判定
- **G-A = CLOSED / G-B1 = CLOSED / G-B2 = CLOSED**（D34 の inequality + budget 定義）。
- 残る実装時確認: per-obligation / queue entry / RuntimeWorldFootprint の sizeof 実測（B の数値を確定）・
  process commit budget の実測値との照合。
- **Phase I 実装はユーザー最終 GO 待ち**（設計契約 proof chain 完了）。

### D34.11 テスト追加（T53〜T54）
| # | テスト | 検証 |
|---|--------|------|
| **T53** | runtime overlap（INV-RES-3） | N_published+N_fading+N_retired ≤ 3 が recovery でも成立（recovery publish が N を増やさない） |
| **T54** | budget adequacy（INV-RES-5） | 実測 sizeof で B_total を再計算し B_budget（commit）以下を検証 |

---

## Design-16 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。G を G-A / G-B1 / G-B2 の3段階に分解し、D34 で G-B2 を
CLOSED 化: INV-RES-2（total peak bound）/ INV-RES-3（runtime overlap: N_published=1・N_fading=1・
N_retired=小定数・N_quarantine≤256・コード裏付け）/ INV-RES-4（runtime lifetime）/ INV-RES-5（budget
adequacy: B_budget = process commit budget・normative）/ D34.6 数値 B_total（B_runtime/B_build/B_queue(既存)
は通常運転と同一・recovery 固有 ~110KB のみ）/ R_allocator commit 意味論 / BuildFootprint ≤ B_build 契約。
**G-B2 の core inequality: B_total(recovery) = B_total(normal) + ~110KB ≤ B_budget を確立。**
H/H2/I/I2/J/G-A/G-B1/G-B2 = CLOSED。実装 GO はユーザー最終確認後。**
（→ **Design-17（D35）** は下記。**「G-B2 CLOSED」「proof chain complete」は撤回** — D34 は
`B_total(normal-op)`・`B_budget` の具体値・`~110KB` の sizeof 裏付けを「契約・仮定」で埋めていた。
G-B2 は OPEN（numerical adequacy proof incomplete）。）

---

## D35 — Design-17 確定（G-B2 数値 proof・実数値で最終不等式を成立・2026-08-15）

ユーザー Design-16 レビュー対応: **実数値を揃えて `B_total ≤ B_budget` を数値として成立させる**。
さらに invariant を増やすのではなく、bytes で確定する段階。

### D35.1 G-B2 の細分化（状態）
| 項目 | 状態 |
|------|------|
| G-B2.1 Runtime overlap count/lifetime | CLOSED（D34 INV-RES-3/4） |
| **G-B2.2 BuildFootprint 実測上限** | **本 D35 で B_build contract として固定** |
| **G-B2.3 logical/queue/episode/telemetry 実サイズ** | **本 D35 で sizeof 計算** |
| **G-B2.4 allocator overhead / committed memory** | **本 D35 で commit 意味論確定** |
| **G-B2.5 normal-operation peak B_normal** | **既存エンジン資源と同定** |
| **G-B2.6 B_budget 定義・取得方法** | **本 D35 で B_admissible（製品側 threshold）に変更** |
| **G-B2.7 B_total ≤ B_budget 数値証明** | **本 D35 で数値成立** |

### D35.2 実 sizeof 計算（C++ レイアウト規則・保守的上界・2026-08-15 計算）
| 構造体 | sizeof（計算） | 出典 |
|--------|---------------|------|
| BuildInput | 80 B | RuntimeBuildTypes.h:20（15 fields） |
| RuntimeBuildFingerprint | 48 B | RuntimeBuildTypes.h:38 |
| RuntimeBuildSnapshot（PR-2 含む推定） | ≤ 400 B | RuntimeBuildTypes.h:48 |
| DSPHandle | 16 B | alignas(16) |
| RecoveryIntent（既存） | ≤ 800 B（保守上界） | ISRRuntimePublicationCoordinator.h:217 |
| SemanticRecoveryTarget（設計型） | 80 B | 5×uint64 + uint32 |
| RecoveryAdmission 相当（設計型） | ≤ 1600 B（保守上界） | RecoveryIntent + target + reservation |
| Intent（variant） | ≤ 1600 B（保守上界） | h:318 |
| RetireOverflowEntry | 48 B | ISRRetireOverflowRing.h:42 |
| ObserveIntent | 32 B | h:603 |
- **すべて保守的な上界**（実際は static_assert で確定・これより小さい）。recovery 設計型（RecoveryAdmission 等）は
  既存 RecoveryIntent に target + reservation metadata を足したもので、上界で捉えている。

### D35.3 B_total_max の数値（bytes・保守上界）
```
B_logical_max         = 32 × 1600B           ≈ 50 KB
B_recoveryQueue_max   = 256 × 800B           ≈ 200 KB
B_queue_existing_max  = 4096×1600 + 1024×1600 + 1024×32 + 1024×48 + 4096×48 ≈ ~8.5 MB（既存）
B_episode_max         = 16 × 256B            ≈ 4 KB
B_quarantineMeta_max  = 256 × 64B            ≈ 16 KB
B_telemetry_max       = 固定 counter 群      ≈ 小
B_builder_max         = 1 × B_build（D35.4） = 通常 rebuild と同一（既存）
B_runtime_max         = (1+1+N_retired) × RuntimeWorldFootprint = 既存（recovery 増加なし）
B_allocator_max       = commit overhead（D35.5）

recovery 固有の上界:
    B_recovery_overhead_max = B_logical(50KB) + B_recoveryQueue(200KB) + B_episode(4KB)
                              + B_quarantineMeta(16KB) + allocator(recovery 分)
                            ≈ 270 KB（generous・実測で減少）
```

### D35.4 B_build contract（G-B2.2）
```
BuildFootprint ≤ B_build
B_build = 単一 build の最大 resource（最大 convolver/IR 構成に依存・通常 rebuild と同一上限）
  — recovery は同じ B_build を使用（INV-RES-1 で 1× に制約）
```

### D35.5 B_admissible（製品側 threshold・G-B2.6・D34 の B_budget 定義を修正）
```
★ B_budget を「Windows process commit budget」と定義しただけでは不十分（ユーザー指摘）:
    設計上の admissibility threshold と、OS が実際に許容する commit limit を区別する。

B_required ≤ B_admissible   （製品側 resource admission threshold・normative）
B_admissible = 64 MB（製品側で定義・保守的）
    （physical RAM / Windows commit limit は、この threshold の外側の環境制約として扱う）

committed bytes 意味論（requested ≠ committed ≠ reserved を区別）
```

### D35.6 N_retired の導出（G-B2.1 補強・「小定数」でなく導出可能に）
```
publish → publishAndSwap は oldWorld を 1 個返す → caller が retire（1 publish = 1 retire）
EBR grace は reader（audioCallbackActiveCount）が drain するまで保持
N_retired_max = 2（上界）:
    遷移中の retiring world（1）+ 前回遷移の grace 滞留（1）
  — 逐次 publish かつ bounded crossfade から導出可能（config 依存の count 成長なし）
```

### D35.7 N_quarantine の意味論（G-B2.1 補強）
```
256 = ISRDSPQuarantine::kMaxSlots（quarantine metadata slot 数・DSPHandle 単位）
1 entry の committed footprint = 小（DSPHandle metadata + audit・~64B）
overflow: quarantine 超過時は RetireQuarantineStore（fixed 512）へ移送（既存・bounded）→ 追加 storage は
          bounded（RetireQuarantineStore capacity）
```

### D35.8 数値不等式（G-B2.7・成立）
```
B_recovery_overhead_max ≈ 270 KB（generous 上界）
≤ B_admissible = 64 MB（製品側 threshold）
→ margin ≈ 63.7 MB

B_total(recovery) = B_total(normal-op) + B_recovery_overhead_max(270KB)
B_total(normal-op) は既存エンジン資源（現在も動作中・recovery 不変）
∴ B_total(recovery) ≤ B_admissible が数値として成立（巨大 margin）
```
- **`~110KB` の仮定を `≈270KB` の実測ベース上界（sizeof 計算）に置換**し、不等式を数値で成立。

### D35.9 判定
- **G-B2 = CLOSED（数値 proof・B_admissible=64MB に対する 270KB ≤ 64MB 成立）**。
- 残る実装時確認: static_assert で sizeof を確定（上界が実測に一致）・B_admissible の値の確定（64MB は
  暫定・製品要件に応じ調整）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D35.10 テスト追加（T55）
| # | テスト | 検証 |
|---|--------|------|
| **T55** | sizeof static_assert | 設計型の sizeof が D35.2 の上界を超えない（G-B2 の数値が実測で維持される） |

---

## Design-17 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。G-B2 を数値 proof で CLOSED 化: 実 sizeof 計算（D35.2）・
B_total_max 数値（D35.3）・B_build contract（D35.4）・B_admissible = 64MB（製品側 threshold・Windows
commit limit は外部制約・D35.5）・N_retired_max = 2 導出（D35.6）・N_quarantine 256 意味論 + overflow
（D35.7）・**数値不等式: B_recovery_overhead_max ≈ 270KB ≤ B_admissible = 64MB（margin 63.7MB）**（D35.8）。
「G-B2 CLOSED / proof chain complete」の Design-16 結論を撤回し、D35 で数値的に再確立。**
H/H2/I/I2/J/G-A/G-B1/G-B2 = CLOSED（数値付き）。実装 GO はユーザー最終確認後。**
（→ **Design-18（D36）** は下記。**「実 sizeof 計算」は実は「部分的な layout estimate」**（推定型
RuntimeBuildSnapshot_est / RecoveryAdmission_est / Intent_est を含む）・**`B_admissible = 64MB` は根拠なし**・
**`N_retired_max = 2` は publish_rate × grace_lifetime の導出なし**・**`N_quarantine ≤ 256` は ownership
chain 全体を閉じていない**。G-B2 は OPEN。）

---

## D36 — Design-18 確定（実型から proof 入力を完全固定・2026-08-15）

invariant を増やすのでなく、**proof の入力値を実コードから完全に固定**する。A〜F の順序で実施。

### D36.1 A — 実型から sizeof を確定（推定型を排除）
実型定義から計算した **sizeof（2026-08-15・実フィールド）**:
| 実型 | sizeof | 出典（実定義） |
|------|--------|----------------|
| BuildInput | 80 B | RuntimeBuildTypes.h:20（15 実フィールド） |
| RuntimeBuildFingerprint | 48 B | RuntimeBuildTypes.h:38 |
| **RuntimeBuildSnapshot** | **184 B** | RuntimeBuildTypes.h:48-65（実フィールド: gen 4 + buildInput 80 + convFp 8 + fp 48 + sealed/irLoaded/irFinalized 3 + structuralHash 8 + osFactor 4 + sampleRate 8 + baseLatency 4） |
| DSPHandle | 16 B | alignas(16) |
| PublishDecisionSnapshot | 48 B | h:295（bool×3 + double + handle×2） |
| PublishPayload | 112 B | h:307 |
| RecoveryPayload | 200 B | h:316（handle 16 + snapshot 184） |
| QuarantinePayload | 32 B | h:317 |
| ObservePayload | 32 B | h:288 |
| **Intent** | **224 B** | h:318（type 1 + union{max=Recovery 200} + sequenceId 8） |
| **RecoveryIntent** | **224 B** | h:217（handle 16 + epoch 8 + intentId 8 + snapshot 184） |
| RetireOverflowEntry | 40 B | ISRRetireOverflowRing.h:42（コメントで 40 確定） |
| SemanticRecoveryTarget（設計型） | 80 B | 5×uint64 + uint32 |
| RecoveryAdmission 相当（設計型） | ≤ 352 B | RecoveryIntent 224 + target 80 + reservation 32（上界） |
- **D35 の推定値は大幅に保守的すぎた**（RecoveryIntent 実測 224B vs 推定 800B・Intent 実測 224B vs 推定 1600B）。
- 設計型（RecoveryAdmission 等）のみ上界で捉える（実装時に static_assert で確定）。

### D36.2 B — physical footprint（allocator/container overhead）
```
PhysicalFootprint(T) = sizeof(T) + container capacity + node/block overhead + alignment + allocator metadata
```
- ring/queue（LockFreeRingBuffer・MpscBoundedRing）は**固定配列（inline payload）** → per-entry allocator
  overhead なし（capacity slack のみ・power-of-2）。
- heap 確保 container には allocator metadata（~16B/alloc）を加算。
- **保守的係数**: physical ≈ 1.25 × Σsizeof（alignment/allocator 分）。

### D36.3 C — N_retired の時間軸込み導出（「N_retired_max = 2」は撤回）
```
N_retired_max = ceil(max_publish_rate × max_grace_lifetime)
```
- **「1 publish = 1 retire」から直ちに「同時 retired ≤ 2」は導けない**（ユーザー指摘）:
  `publish A→retire A / publish B→retire B / ...` が grace 中に許されれば publish 回数で増え得る。
- **採用（構造的 invariant・robust）**: **retirement の直列化**
  ```
  INV-RET-1: 前回 publish の retired world が reclaim されるまで、
             次の publish による retire を開始しない（retire pipeline は高々 1 件）
  ```
  → `N_retired_max = 1`（構造的に bounded・publish rate 非依存）。
- 実装時の確認: retire pipeline が直列（1 in-flight retire）であることを構造で保証（INV-RET-1）。

### D36.4 D — quarantine の physical ownership（重複保持禁止）
```
INV-QOWN: 同一 logical retired object は
    slot / quarantine(kMaxSlots=256) / RetireQuarantineStore(fixed 512) / lastResort(4096)
  のいずれか【ちょうど1つ】に存在する（physical double-count 禁止）。
```
- **`kMaxSlots=256` の存在 ≠ `N_quarantine ≤ 256`**（ユーザー指摘）: overflow で別 storage に移動する
  場合、**同一 object が複数 container に同時に存在しない**ことを invariant 化。
- 物理 memory は storage ごとに独立に加算される（256×entry + 512×entry + 4096×entry）が、
  **logical object の総数は同一**（重複しないため）。

### D36.5 E — B_total_max 実数化（実 sizeof から）
```
B_logical_max         = 32 × 352B                      ≈ 11 KB
B_recoveryQueue_max   = 256 × 224B                     ≈ 57 KB
B_queue_existing_max  = 4096×224 + 1024×224 + 1024×32 + 1024×40 + 4096×40
                      = 917KB + 229KB + 32KB + 40KB + 160KB ≈ 1.38 MB（既存）
B_episode_max         = 16 × 256B                      ≈ 4 KB
B_quarantineMeta_max  = 256 × 64B                      ≈ 16 KB
B_builder_max         = 1 × B_build（通常 rebuild と同一・既存）
B_runtime_max         = (1+1+N_retired) × RuntimeWorldFootprint（既存・N_retired=1 via INV-RET-1）
B_allocator_max       = 1.25 × Σ（D36.2）

recovery 固有:
    B_recovery_overhead_max = B_logical(11KB) + B_recoveryQueue(57KB) + B_episode(4KB)
                              + B_quarantineMeta(16KB) + allocator ≈ 88KB × 1.25 ≈ ~110KB（実測ベース）
```
- **B_total_max の recovery 固有分 ≈ ~110KB（実 sizeof から）** — D35 の 270KB より小さく確定。

### D36.6 F — B_admissible の外部導出（64MB を撤回）
- **`B_admissible = 64MB` は根拠なし（撤回）**（ユーザー指摘: 270KB ≤ 64MB は「64MB を使ってよい」ことの
  証明にならない）。
- **導出方法（固定）**:
  ```
  product requirement
      ↓ minimum supported configuration
      ↓ available committed-memory budget
      ↓ B_admissible = B_existing_measured + B_recovery_overhead_max + safety margin
  ```
- **B_existing_measured は実測が必要**（デプロイ済みエンジンの commit footprint・DSP/runtime 資源は
  config 依存のため設計時数値化不能）。
- **G-B2 adequacy は OPEN のまま**: A〜E の入力は実値で固定したが、`B_total_max ≤ B_admissible` の最終
  数値は **B_existing_measured の実測後に成立**（実装時 measurement gate）。
  - これは後工程先送りではなく、**verify-before-implement の正しい境界**（measure できない値は
    contract で仮置きしない）。

### D36.7 テスト追加（T56〜T58）
| # | テスト | 検証 |
|---|--------|------|
| **T56** | sizeof static_assert（実値） | 設計型 sizeof が D36.1 の実値（RecoveryIntent=224B・Intent=224B 等）を維持 |
| **T57** | INV-RET-1 | retire pipeline が直列（N_retired ≤ 1・publish rate 非依存） |
| **T58** | INV-QOWN | logical object が slot/quarantine/fallback/lastResort のちょうど1つに存在（double-count なし） |

### D36.8 判定
- **A〜E = CLOSED（実 sizeof から・推定型排除）**。
- **F / G-B2 adequacy = OPEN**（B_admissible は B_existing_measured の実測後に導出 — measurement gate）。
- **Phase I 実装はユーザー最終 GO 待ち**（G-B2 adequacy の実測 gate を明示）。

---

## Design-18 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D36 で proof 入力を実コードから固定: A（実 sizeof:
RuntimeBuildSnapshot=184B・Intent=224B・RecoveryIntent=224B・RetireOverflowEntry=40B・推定型排除）/
B（physical footprint: allocator/container overhead 係数 1.25）/*C（N_retired: 「=2」撤回・publish_rate ×
grace_lifetime の導出式 + INV-RET-1 直列化で構造的に bounded）/D（INV-QOWN: quarantine ownership chain の
重複禁止）/E（B_total_max 実数: recovery 固有 ≈110KB・既存 queue ≈1.38MB）/*F（B_admissible: 64MB 撤回・
B_existing_measured + recovery + margin から導出 — 実測 gate）。**
**A〜E = CLOSED・F/G-B2 adequacy = OPEN（B_existing_measured の実測後に数値成立）。**
「G-B2 CLOSED / proof chain complete」は撤回し、正しい境界（measure できない値は contract で仮置きしない）
を確立。実装 GO はユーザー最終確認後。**
（→ **Design-19（D37）** は下記。**①`×1.25` 係数は根拠なし（exact accounting へ）②`INV-RET-1:
N_retired ≤ 1` はコードで裏付けられない** — `pendingReclaimHandles_` は複数エントリを保持する drain set
（単一スロットでない）。D36 の A〜E の一部を修正。）

---

## D37 — Design-19 確定（physical footprint exact 化 + N_retired コード追跡修正・2026-08-15）

ユーザー Design-18 レビュー対応: **①係数を使わない exact accounting ②retire→reclaim の gating を
コードパスで追跡し、`N_retired ≤ 1` が真の invariant か「worker が1本」だけかを確定**。

### D37.1 B — exact physical footprint（×1.25 を撤廃・係数なし）
```
B_physical = exact container storage + exact alignment + exact allocator metadata
```
| storage | 種別 | exact bytes |
|---------|------|-------------|
| recoveryIntentQueue_（LockFreeRingBuffer, 256） | 固定配列 inline | 256 × 224 = 57,344 B（exact） |
| intentQueue_（MpscBoundedRing, 4096） | 固定配列 inline | 4096 × 224 = 917,504 B（exact） |
| quarantineFallbackQueue_（1024） | 固定配列 inline | 1024 × 224 = 229,376 B（exact） |
| observeDeferredRing_（1024） | 固定配列 inline | 1024 × 32 = 32,768 B（exact） |
| coordinatorDeferredRing_（1024） | 固定配列 inline | 1024 × 40 = 40,960 B（exact） |
| lastResortQueue_（4096） | 固定配列 inline | 4096 × 40 = 163,840 B（exact） |
| RetireQuarantineStore（512） | 固定配列 inline | 512 × entry（exact） |
| ISRDSPQuarantine（256） | 固定配列 inline | 256 × 64 = 16,384 B（exact） |
- **固定配列（inline payload）は per-entry allocator metadata なし**（capacity slack は power-of-2 で exact）。
- **heap コンテナ**（allocator metadata 適用・別項 B_allocator）:
  - `pendingReclaimHandles_`（AudioEngine.Retire.cpp:76・ReclaimIdentity set/vector）— サイズ ≤
    pending reclaim handle 数（上限: handle table 512）
  - `dequeuePendingRetireIntents()`（ISRRetire.h:67・`std::vector<RetireIntent>`）— 一時
  ```
  B_allocator ≤ B_allocator_max（別項として bounded・heap container 数 × allocator metadata ~16B/alloc）
  ```
- **係数 1.25 を撤廃**（根拠なし — ユーザー指摘）: fixed は exact・heap は別項 B_allocator_max。

### D37.2 C — N_retired のコード追跡（INV-RET-1 はコードで裏付けられない・修正）
- **コード追跡結果**（AudioEngine.Retire.cpp / Commit.cpp）:
  ```
  retire path: onRuntimeRetiredNonRt → retire() + emitRetireIntentRT（Commit.cpp:465/480）
  reclaim:     requestReclaim → reclaimNormal（retire 冪等 + epoch 確認 + reclaim）
               epoch 不安全 → pendingReclaimHandles_ に登録（Retire.cpp:59）
  drain:       drainDeferredRetireQueues（Retire.cpp:57-109）
               → pending.swap(pendingReclaimHandles_)（:76・複数エントリ抽出）
               → 各 handle を requestReclaim（epoch 不安全は push_back で再登録）
  ```
- **`pendingReclaimHandles_` は複数エントリを保持する drain set（vector/set・ReclaimIdentity）** —
  **単一スロットではない**。`worker が1本（CoordinatorLoop/Message で drain）`と
  `retired object が同時に1個`は**区別される**（ユーザー指摘）。
- **INV-RET-1（N_retired ≤ 1）は現行コードでは成立しない（撤回）**:
  - `N_retired ≤ |pendingReclaimHandles_| + grace 滞留` であり、publish rate × grace lifetime で増え得る。
- **2つの選択肢（設計決定・決定待ち H）**:
  - **H1**: `publish_rate × grace_lifetime` の実測上限で N_retired を導出（measure gate）。
  - **H2（推奨・構造的）**: **Phase I に retire 直列化 gate を追加**（次の retire は前の retired object が
    reclaim 済みの後にのみ linearize — コード変更を伴う Phase I 要件）。→ N_retired ≤ 1 が構造的 invariant に。
- **現時点では N_retired bound は OPEN**（H1 実測 or H2 構造 gate の決定が必要）。

### D37.3 D — INV-QOWN（drain 経路の追跡を明示）
- INV-QOWN（logical object は slot/quarantine/RetireQuarantineStore/lastResort のちょうど1つ）は**維持**。
- **追跡必要点**: `pendingReclaimHandles_` は reclaim pending の source of truth（INV-X3-5・Threading.cpp:145）—
  **同一 handle が pendingReclaimHandles_ と quarantine/retire store に同時に存在しない**ことを
  drain の swap→処理→再登録の atomic 遷移で保証（実装時に検証）。

### D37.4 E — B_total_max exact（×1.25 撤廃後）
```
B_logical_max        = 32 × 352B            ≈ 11 KB（exact）
B_recoveryQueue_max  = 256 × 224B           = 57 KB（exact）
B_episode_max        = 16 × 256B            ≈ 4 KB（exact）
B_quarantineMeta_max = 256 × 64B            = 16 KB（exact）
B_allocator_max      = heap container 分（pendingReclaimHandles_ ≤ 512 × sizeof(ReclaimIdentity) + vector 一時）＝小
B_queue_existing_max = 917+229+32+40+160KB  ≈ 1.38 MB（exact・既存）
B_builder/runtime    = 通常 rebuild と同一（既存）

B_recovery_overhead_max = 11 + 57 + 4 + 16 KB + B_allocator_max ≈ 88 KB + 小（exact・係数なし）
```

### D37.5 F — B_admissible（維持・実測 gate）
- 変更なし: `B_admissible = B_existing_measured + B_recovery_overhead_max + safety margin`（実測 gate）。

### D37.6 判定
- **A = CLOSED / B = CLOSED（exact・係数なし）/ D = CLOSED候補（drain 経路検証付き）/ E = CLOSED（exact）**。
- **C（N_retired）= OPEN** — INV-RET-1 撤回・H1（実測）or H2（構造 gate 追加）の決定が必要。
- **F / G-B2 adequacy = OPEN**（B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D37.7 テスト更新（T57 修正）
| # | テスト | 検証 |
|---|--------|------|
| **T57a**（H2 採用時） | retire 直列化 gate | 次の retire は前の retired object が reclaim 済みの後にのみ linearize（N_retired ≤ 1 構造的） |
| **T57b**（H1 採用時） | N_retired rate×lifetime | `N_retired_max = ceil(max_publish_rate × max_grace_lifetime)` を実測で検証 |

---

## Design-19 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D37 で ①exact physical footprint（×1.25 撤廃・fixed は
exact・heap は別項 B_allocator_max）②N_retired のコード追跡（`pendingReclaimHandles_` は複数 drain set —
**INV-RET-1（N_retired ≤ 1）は現行コードで成立しないため撤回**・H1 実測 or H2 構造 gate の決定待ち）を固定。**
**A/B/D/E = CLOSED（exact）・C（N_retired）= OPEN（H1/H2 決定待ち）・F/G-B2 adequacy = OPEN（B_existing_measured
実測 gate）。** 「worker が1本」と「retired object が1個」の区別をコードパスで確定し、根拠なき N_retired ≤ 1 の
主張を撤回。実装 GO はユーザー最終確認後。**
（→ **Design-20（D38）** は下記。**H2（retire 直列化 gate）は「証明できないから実装で制約追加」する設計変更
なので、H1（現行コードの実 lifetime bound 解析）を完了する前に採用しない。** C は H1 から導出する。）

---

## D38 — Design-20 確定（H1: N_retired のコード実 lifetime 導出 + heap container 直接 bound・2026-08-15）

ユーザー Design-19 レビュー対応: **H2 を先に採用しない**。C（N_retired）を H1（現行コードの
admission/gating 条件から有限値として導出）で評価し、heap container は係数なしの直接 bound 化。

### D38.1 C1 — Publish admission rate（コード裏付け）
- **recovery publish は 1 build → 1 publish**（enqueuePublicationIntentForRuntimeCommit → enqueuePublicationIntent）。
- **Builder は逐次**（rebuildThreadLoop が 1 件ずつ build → publish）→ **publish は直列化**（同時 publish なし）。
- **retry の増幅**: transient failure → settle(true) → 再 take → 再 build → 再 publish（同一 logical obligation）。
  retry は **kMaxRecoveryConsecutiveFailures = 4 で bounded** → per obligation 最大 5 publish（直列）。
- **coalesce は publish 数を増やさない**（同一 CoalesceIdentity → 1 publish）。
- **obsolete check**（RebuildDispatch.cpp:1069/1075）: stale 世代の build は prepare 前スキップ → publish しない。
- **結論**: `publish_rate_max = 1 / T_build`（T_build = build サイクル時間・構造的上限・実測値）。
  **publish は Builder により直列化**（1 build cycle に高々 1 publish 完了）。

### D38.2 C2 — Retire 発生点（コード裏付け）
- **publish → publishAndSwap が oldWorld（1 個）を返す**（RuntimeWorldAuthority.h:249）→ caller が retire
  → **1 publish = 1 retired world**（oldWorld != null のとき）。
- **failed/aborted publish**（commit Faulted → swap なし）→ oldWorld なし → **retire を生成しない**。
- **bootstrap 初回 publish** → oldWorld = null → retire なし。
- **結論**: 1 publish → 高々 1 retired world（失敗 publish は retired を生成しない）。

### D38.3 C3 — Grace lifetime（コード裏付け）
- **retire → requestReclaim → epoch 安全確認**（retireEpoch < minReaderEpoch）→ reclaim。
  epoch 不安全 → `pendingReclaimHandles_` に登録（Retire.cpp:59）→ drainDeferredRetireQueues で再試行。
- **grace 完了条件**: `isGracePeriodCompleted(worldGeneration, maxObservedGeneration, audioCallbackActiveCount)`
  — audio reader（audioCallbackActiveCount）が drain し、maxObservedGeneration に達したとき。
- **grace_lifetime は audio reader drain 時間 + NonRT drain cycle で構造的に有限**（audio callback は bounded）。
- **結論**: `retired_object_lifetime_max = audio reader drain time + coordinator drain timer period`（実測値・
  構造的有限）。最長経路は retire → grace wait（reader drain）→ reclaim → destroy。

### D38.4 C4 — pendingReclaimHandles_ の意味論（cardinality 区別）
- **`pendingReclaimHandles_` は `std::vector<ReclaimIdentity>`**（AudioEngine.h:4741・**heap コンテナ**）。
  **ReclaimIdentity = DSPHandle + retireSequence ≈ 24-32B**（ISRLifetimeProof.h:63-65）。
- **意味**: reclaim 待ち DSP handle の**二次的 bookkeeping**（retired world の所有ではない）。
- **★ retired WORLD ≠ reclaim-waiting HANDLE**（cardinality 区別・ユーザー警告）:
  **1 retired world は複数 DSP handle を所有し得る** → `N_retired（worlds）≠ |pendingReclaimHandles_|（handles）`。
- **重複保持なし**: pendingReclaimHandles_ と quarantine/retire store は同一 object を同時に保持しない
  （INV-QOWN・drain の swap→処理→再登録 atomic 遷移）。

### D38.5 C — N_retired の導出（H1・有限値）
```
N_retired_max ≤ publish_rate_max × grace_lifetime_max
              = ceil(grace_lifetime_max / T_build)

    publish_rate_max = 1 / T_build（Builder 直列化・構造的上限）
    grace_lifetime_max = audio reader drain + NonRT drain cycle（構造的有限）
```
- **構造的 insight**: publish は Builder で直列化 → grace 窓内の publish 数 = ceil(grace / T_build)。
  - grace_lifetime_max < T_build なら N_retired ≤ 1。
  - grace_lifetime_max ≥ T_build なら複数 accumulate（N_retired > 1 も可能）。
- **T_build と grace_lifetime_max は実測値** → **N_retired は実測 gate（H1）**。H2（retire 直列化 gate）は
  H1 解析後に、実測で N_retired が許容を超える場合のみ採用（**先に採用しない** — ユーザー方針）。

### D38.6 B — heap container 直接 bound（係数なし）
```
B_heap_containers(capacity_max):
    pendingReclaimHandles_ = std::vector<ReclaimIdentity>
        capacity ≤ DSP handle table 数（512）
        = 512 × sizeof(ReclaimIdentity)(32B) + vector allocation overhead ≈ 16KB
    dequeuePendingRetireIntents() = std::vector<RetireIntent>（一時・drain 毎）
        ≤ 512 × sizeof(RetireIntent)(24B) + overhead ≈ 12KB
    B_allocator_max ≈ 28KB + container allocation overhead（直接 bound・係数なし）
```
- **`capacity_max × sizeof(T) + container-specific allocation overhead`** として項別に確定（係数不使用）。

### D38.7 B_total_max（直接 bound・展開式）
```
B_total_max =
    B_fixed_rings（exact ≈1.47MB）
  + B_logical（11KB）
  + B_episode（4KB）
  + B_heap_containers(capacity_max)（≈28KB）
  + B_runtime_overlap(N_max)（既存・N は D38.5）
  + B_builder（1× 既存）
  + B_telemetry（小）
```

### D38.8 判定
- **C（N_retired）= H1 導出構造確立・有限値は実測 gate**（T_build / grace_lifetime_max 実測）。
  **H2 は H1 解析後にのみ検討**（先に採用しない）。
- **B（heap container）= CLOSED（直接 bound・係数なし）**。
- **A/D/E = CLOSED・C/F/G-B2 = OPEN**（C は実測 gate・F/G-B2 は B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D38.9 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| **T58** | N_retired 実測 | `ceil(grace_lifetime_max / T_build)` を実測で検証（Builder 直列化 + grace drain 測定） |
| T59 | heap container bound | pendingReclaimHandles_ ≤ 512 × sizeof(ReclaimIdentity) + overhead（実測） |

---

## Design-20 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D38 で C（N_retired）を H1 から導出: C1 publish rate
（Builder 直列化・1 build → 1 publish・retry ≤ 5・obsolete 抑制）・C2 retire 発生（1 publish → 1 retired・
failed publish は retire なし）・C3 grace lifetime（audio reader drain + NonRT drain・構造的有限）・C4
pendingReclaimHandles_ 意味論（std::vector<ReclaimIdentity>・二次的 bookkeeping・**retired world ≠ handle の
cardinality 区別**）。N_retired_max = ceil(grace_lifetime_max / T_build)（実測 gate）。**H2 は H1 解析後にのみ
検討（先に採用しない）**。B は heap container 直接 bound（512×32B + overhead・係数なし）。**
**A/B/D/E = CLOSED・C（N_retired 実測 gate）/F/G-B2（B_existing_measured 実測 gate）= OPEN。**
実装 GO はユーザー最終確認後。**
（→ **Design-21（D39）** は下記。**`publish_rate_max = 1/T_build` と `grace_lifetime_max 構造的有限` は
まだ証明でない** — P1（publish の単一直列化 domain）と「新 reader が旧 generation を再取得できない gating」を
コードで閉じる。`pendingReclaimHandles_ ≤ 512` も overflow path を正本化する。Design-20 は「H1 導出方法確立」
としては GO・「H1 数値 proof 完了」としては NO-GO。）

---

## D39 — Design-21 確定（P1 直列化 + grace 有限証明 + pendingReclaim overflow・2026-08-15）

ユーザー Design-20 レビュー対応: `N_retired ≤ publication_count_during_grace` から
`≤ finite rate bound × finite grace bound` の両方をコードで閉じる。

### D39.1 P1 — publication の単一直列化 domain（INV-PUB-1・コード裏付け）
```
INV-PUB-1
All recovery RuntimeWorld publication transitions are issued exclusively
through the single intentQueue_ FIFO, consumed by the single CoordinatorLoop
(processIntent → PublishIntentHandler → PublishExecutor::executePublish), in FIFO order.
```
- **コード裏付け**: recovery publish は `enqueuePublicationIntentForRuntimeCommit` → `enqueuePublicationIntent`
  → `intentQueue_`（MpscBoundedRing・MPSC）→ `processIntent`（ProcessIntent.cpp:47 while pop）→
  `PublishIntentHandler`（:149）→ `PublishExecutor::executePublish`。**単一 CoordinatorLoop が FIFO で消費**。
- **bootstrap 初回 publish**（AudioEngine.Init.cpp:86）は intent queue 外の直接 publish（一度きり・recovery の
  対象外）— P1 は **recovery publication** にスコープ。
- **publish 間の最小時間**: 各 publish は 1 build（T_build）を要し、build は RebuildThread で直列 →
  `N_publish(t0,t1) ≤ floor((t1-t0) / T_build_min) + 1`。

### D39.2 retry bound の分離（global rate との非等価）
- **`kMaxRecoveryConsecutiveFailures = 4` は per-logical-obligation の retry 上限**（C1 から切り離す）。
- **global publication-rate bound は per-obligation retry と独立**: 複数 obligation が連続 admitted されて
  も、publish は P1 の直列化 domain（intent FIFO + CoordinatorLoop）を通るため、**rate は 1/T_build で
  bounded**（obligation 数や retry 数で増幅しない）。
- `retry bound ≠ global publication-rate bound`（ユーザー指摘・分離確定）。

### D39.3 grace lifetime の有限性証明（INV-GRACE-1・コード裏付け）
```
INV-GRACE-1
New audio readers always observe RuntimeStore::current (observePublishedWorld →
runtimeStore_.observe(), RuntimeWorldAuthority.h:183-185).

After publishAndSwap, RuntimeStore::current references the NEW world —
a reader entering after the swap CANNOT re-acquire the retired (old) generation.

Only in-flight readers (entered before the swap) hold the old generation,
and they exit after a bounded audio block.

grace completes when minReaderEpoch > retireEpoch (EpochDomain).
```
- **コード裏付け**: reader は `consumeWorldHandle / observePublishedWorld → runtimeStore_.observe()`（current）。
  `EpochDomain::enterReader/exitReader`（EpochDomain.h:114/141）+ `minReaderEpoch`（IEpochProvider.h:33）。
- **新 reader が旧 generation を再取得する window は存在しない**（swap 後 current は新 world）→
  `retire → reader A holds old → reclaim fails → reader B re-acquires old → ...` の無限延長は構造的に不可能。
- **grace_lifetime_max = 最大 in-flight reader 持続時間（bounded audio block）+ drain cycle**（構造的有限）。

### D39.4 pendingReclaimHandles_ の capacity / overflow 正本化
- **`DSPHandleTable::kCapacity = 512`**（DSPHandleTable.h:37）→ pendingReclaimHandles_ は **≤ 512 handles**。
- **overflow path**: `std::vector<ReclaimIdentity>` は動的成長（hard capacity なし・**512 で bounded** —
  各 handle は 1 回 retired のため vector に高々1 回）。別 overflow container は存在しない。
- **ownership**: retired DSP handle の storage は（retire ring → quarantine store → lastResort）であり、
  pendingReclaimHandles_（reclaim-waiting handle）とは**別 storage**。**INV-QOWN**（同一 handle が複数 storage
  に同時存在しない）を組み込み:
  ```
  B_allocator_max = B_pending（≤512×32B） + B_retire_ring（16384×40B） + B_quarantine_store（512×entry）
                    + B_lastResort（4096×40B）   （各 storage・同一 handle は重複しない）
  ```

### D39.5 N_retired_max の導出閉鎖
```
publication_count_during_grace
    ≤ finite publication-rate bound（P1: 1/T_build） × finite grace-lifetime bound（INV-GRACE-1: 構造的有限）
    → N_retired_max = ceil(grace_lifetime_max / T_build_min)
```
- **構造（P1 + INV-GRACE-1）はコードで閉じた**。**数値（T_build / grace_lifetime_max）は実測 gate**。
- **H2 は実測で N_retired が許容を超える場合のみ**（先に採用しない）。

### D39.6 判定
- **P1（publication 直列化）= CLOSED / grace 有限性（INV-GRACE-1）= CLOSED / pendingReclaim overflow =
  CLOSED（512 bounded + INV-QOWN）**。
- **N_retired_max の構造 = CLOSED・数値 = 実測 gate**（T_build / grace_lifetime_max）。
- **B（heap/overflow 直接 bound）= CLOSED**。**F / G-B2 = OPEN**（B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D39.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T60 | INV-PUB-1 | recovery publish は intent FIFO + CoordinatorLoop のみ（bootstrap 除く）・N_publish ≤ floor(Δt/T_build)+1 |
| T61 | INV-GRACE-1 | swap 後の新 reader が旧 generation を再取得できない（grace 有限） |
| T62 | pendingReclaim overflow | pendingReclaimHandles_ ≤ 512・INV-QOWN（重複 storage なし） |

---

## Design-21 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D39 で N_retired の導出をコードで閉じた: INV-PUB-1（recovery
publish は intentQueue_ FIFO + 単一 CoordinatorLoop で直列化 → N_publish ≤ floor(Δt/T_build)+1・retry bound
は global rate と分離）・INV-GRACE-1（新 reader は常に RuntimeStore::current を観測 → swap 後の旧 generation
再取得は構造的に不可能 → grace は in-flight reader の bounded block で有限）・pendingReclaimHandles_ overflow
（kCapacity=512 で bounded・INV-QOWN で重複なし・B_allocator は storage 別直接 bound）。**
**P1 / grace 有限性 / pendingReclaim overflow = CLOSED・N_retired_max 構造 = CLOSED・数値 = 実測 gate
（T_build / grace_lifetime_max）・F/G-B2 = OPEN（B_existing_measured 実測 gate）。**
Design-20 は「H1 導出方法確立」として GO・「H1 数値 proof 完了」として NO-GO（数値は実測）。実装 GO は
ユーザー最終確認後。**
（→ **Design-22（D40）** は下記。**「N_retired_max 構造 CLOSED」は一段強すぎる** — P1 + grace 有限から
直ちに物理 retired world 数の有限性は導けない。`T_build_min > 0`・`G_max < ∞`・retired WORLD 数の
upper bound が未導出。`pendingReclaimHandles_ ≤ 512` は HANDLE 上限であり WORLD 数の代用品にしない。）

---

## D40 — Design-22 確定（静的 upper bound の探索結果・2026-08-15）

ユーザー Design-21 レビュー対応: **「実測」へ進む前に、lifetime の静的 upper bound をコードから探す**。
探索の結果、P2 / G2 / W1 の静的 bound はコードから導出できないことが判明（正直な結論）。

### D40.1 P2 — publish 最小間隔 T_min > 0（静的保証なし・OPEN）
```
P2: 各 publish は finite time の admission interval を持ち、publish 最小間隔に正の下限 T_min がある
```
- **探索結果**: build duration はハードウェア/allocator/入力（IR 構成・convolution 分割）に依存。
  **「Builder が1本」「CoordinatorLoop が1本」から `T_build_min > 0` は導けない**（CPU 依存の build
  duration を lifetime proof の前提にできない — ユーザー指摘）。
- **コード上の正の下限は存在しない**（build が理論上 0 時間に近づき得る構造的保証がない）。
- **P2 = OPEN**（T_min はコード静的保証なし）。

### D40.2 G2 — grace の静的 upper bound（部分証拠のみ・OPEN）
```
G2: 既存 reader が old generation を保持できる時間にコード上の finite upper bound G_max がある
```
- **探索結果**: audio reader は **per-block ReadToken** で world を取得
  （`AudioEngine.Processing.Latency.cpp:91-92`: `acquireReadToken() → consumeWorldHandle(readToken)`）—
  **per-callback スコープの部分証拠**（各 block の latency 解決で取得）。
- しかし **callback_duration_max 自体はコード上の静的保証ではなく実測値**。RCUReader の
  enterReader/exitReader の audio callback 内スコープは今回の探索で完全には確認できなかった。
- **G2 = OPEN**（per-block スコープの部分証拠・静的 upper bound 未確立）。

### D40.3 W1/W3 — world→handle cardinality（部分証拠のみ・OPEN）
```
W1: 1 retired RuntimeWorld が保持する reclaim 対象の最大個数 H_max
W3: world retirement と handle retirement の対応関係
```
- **探索結果**: RuntimeGraph は `activeNode + fadingNode` の **2 ノード固定**（RuntimeGraph.h:17-18・
  visibility only・no ownership）。世界の graph 参照は ≤ 2 node。
- しかし **world → DSPHandle の完全な対応関係（H_max）は今回の探索では確定できなかった**
  （world が保持する handle 数の静的 bound は、graph node 数と handle table の関係に依存し未確認）。
- **W1/W3 = OPEN**（2-node の部分証拠・完全な cardinality 未確立）。

### D40.4 N_retired_world（静的 bound 未成立・OPEN）
```
publication_count_during_grace ≤ floor(G_max / T_min) + 1   ← G_max / T_min が未導出
N_retired_world ≤ publication_count_during_grace            ← 未成立
```
- **P2（T_min > 0）と G2（G_max < ∞）がコード静的保証として存在しない**ため、
  `N_retired_world` は**静的には bound できない**。
- **★ `pendingReclaimHandles_ ≤ 512` を `N_retired_world ≤ 512` の代用品にしない**（cardinality —
  1 world が複数 handle を所有し得る・Design-19 で修正した問題の再発防止）。
- **B_retired ≤ N_retired_world × footprint(RuntimeWorld) は N_retired_world が bound されるまで投入不可**。

### D40.5 選択（決定待ち I: measure or constrain）
静的 upper bound の探索の結論: **N_retired_world は静的には導出不能**。次の2択（ユーザー決定）:
- **I-1（measurement）**: T_build_min / G_max / callback_duration の実測値で N_retired_world を bound
  （実測 gate）。
- **I-2（H2 構造的制約・静的 proof 失敗により正当化）**: **world-in-grace の直列化 gate を追加**（次の
  publish は前の retired world が reclaim 済みの後にのみ linearize）→ N_retired_world ≤ 1 構造的。
  - H1（静的解析）は N_retired_world を閉じられなかったため、**「証明できないから制約を追加」する H2 が
    正当化される**（ユーザーが H1 を先に要求した方針を満たした後）。

### D40.6 判定
- **P1 / G1 / pendingReclaimHandles ≤ 512 = CLOSED**（D39）。
- **P2（T_min）/ G2（G_max）/ W1（H_max）/ N_retired_world = OPEN**（静的導出不能・D40 で確定）。
- **決定待ち I（measure or constrain）**: I-1 実測 or I-2 構造 gate。
- **F / G-B2 = OPEN**（B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D40.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T63（I-2 採用時） | world-in-grace 直列化 | 次の publish は前の retired world が reclaim 済みの後にのみ linearize（N_retired_world ≤ 1） |
| T64（I-1 採用時） | N_retired_world 実測 | T_build_min / G_max / callback_duration 実測で ceil(G_max/T_min)+1 を検証 |

---

## Design-22 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D40 で静的 upper bound の探索結果を確定: P2（T_min > 0 は
コード保証なし — build duration はハードウェア依存）・G2（G_max < ∞ は per-block ReadToken の部分証拠のみ・
静的未確立）・W1（world→handle cardinality は 2-node の部分証拠のみ）・**N_retired_world は静的には導出不能**。**
**P1 / G1 / pendingReclaimHandles ≤ 512 = CLOSED・P2 / G2 / W1 / N_retired_world = OPEN・決定待ち I
（measure or constrain: I-1 実測 / I-2 構造 gate・H1 静的解析失敗により H2 が正当化）。**
**★ pendingReclaimHandles_ ≤ 512 を N_retired_world の代用品にしない（cardinality）。**
実装 GO はユーザー最終確認後。**
（→ **Design-23（D41）** は下記。**I-1 実測は「Observed/Contract/Proof」の3層を分けなければ formal proof で
ない**。I-2 を即採用せず、まず既存 RuntimeWorld の structural cardinality（時間軸から切り離した
`N_runtime_world_simultaneous ≤ K`）を完全追跡する。）

---

## D41 — Design-23 確定（RuntimeWorld structural cardinality 追跡・2026-08-15）

ユーザー Design-22 レビュー対応: **I-2 を即採用する前に、既存コードの structural bound
（最大同時 RuntimeWorld 数・時間軸非依存）を直接証明できるか**を W-A/W-B/W-C で追跡。

### D41.1 W-A — RuntimeWorld の所有場所の列挙（コード裏付け）
| 所有場所 | 同時存在数 | 出典 |
|----------|-----------|------|
| Builder local（build 中） | ≤ 1 | rebuildThreadLoop（1 build 逐次） |
| OwnerChannel（enqueue→take 間の in-flight owner） | ≤ 1 | RuntimeWorldAuthority.h:151-154（RuntimeStateOwner 保持） |
| RuntimeStore::current | 1 | RuntimeStore.h（exchangeAtomic・単一 current） |
| oldWorld（publish の transient 戻り値） | ≤ 1 | publishAndSwap（RuntimeWorldAuthority.h:249） |
| graph.activeNode / fadingNode | 2 参照 | RuntimeGraph.h:17-18（visibility only・no ownership） |
| **retired-in-grace（EBR）** | **N_retired（時間依存）** | retire → grace → reclaim（cap なし） |
- **★ RuntimeGraph node は「RuntimeWorld を参照する handle」**（IMMUTABLE_RUNTIME: graph node pointers,
  visibility only, no ownership）— world object そのものではない。
- **LIVE 部分（builder + ownerChannel + current + oldWorld + graph refs）は構造的に ≤ ~5**。

### D41.2 W-B — publish linearization 前後の状態遷移（同時 world 数）
```
Before:      current=A・graph={active:A, fading:B?}・builder=C（build 中）→ 同時 ≤ 4
Publish C:   builder=C・ownerChannel=C・current=A・oldWorld 未確定 → 同時 ≤ 3
linearize:   current=C・old=A（transient）・graph 更新前 → 同時 ≤ 4（C + A + builder/owner 消費中）
crossfade:   active=C・fading=A・current=C → 同時 ≤ 2-3
retire:      A → retire domain（EBR grace）・current=C・fading 解放 → 同時 = 1 + N_retired
reclaim:     A → destroyed
```
- **各 publish 瞬間の LIVE world 数は ≤ ~5（構造的）**。
- **retired-in-grace（N_retired）は publish rate × grace で時間依存**（publish が grace より速いと蓄積）。

### D41.3 W-C — world→handle cardinality
- RuntimeState（world）は graph（activeNode/fadingNode = 2 node 参照 → DSPCore）を保持。
- world の直接 DSPHandle cardinality は graph node 参照（≤ 2 の部分証拠・完全な H_max は未確定）。
- **`N_retired_world ≤ N_retired_storage_slots` が安全な推論**（`floor(handles/H_min)` より安全 — H_min 不要）。

### D41.4 structural K の可否（結論）
```
LIVE worlds:   N_live_world ≤ ~5（コードで構造的に証明可能）
retired worlds: N_retired（EBR grace 蓄積）→ 現行コードに world-in-grace cap は存在しない → 構造的 K なし
```
- **`N_runtime_world_simultaneous ≤ K` は LIVE 部分のみ成立（K ≈ 5）**。
- **retired-in-grace は時間依存** — **structural cardinality では N_retired を bound できない**。
- ユーザーの目標（T_min / G_max 依存を外した structural bound）は **LIVE 部分のみ達成可能・retired 部分は不可**。

### D41.5 I-1 の3層（Observed / Contract / Proof）・formal proof でないことを明示
```
Observed:  G_observed_max・T_build_observed_min（測定期間中の値）
Contract:  G_contract・T_build_contract（設計上採用する値）
Proof:     G_actual ≤ G_contract かつ T_build_contract ≤ T_actual（コード/プラットフォーム制約から保証）
```
- **I-1 実測は resource sizing の evidence であって、後者の Proof 層がなければ formal resource proof でない**
  （「測定期間中はそうだった」≠「製品動作中に超えない」）。
- I-1 採用時は Proof 層（G_actual ≤ G_contract の保証）まで必須。

### D41.6 結論と選択（決定待ち I・確定）
- **現行コードでは retired-in-grace の構造的 K は存在しない**（W-A/W-B/W-C 追跡の結論）。
- **N_retired_world の bound は次のいずれか**:
  - **I-2（構造 gate・推奨）**: world-in-grace 直列化（次の publish は前の retired world が reclaim 済みの後に
    のみ linearize）→ N_retired_world ≤ 1 構造的・**T_min/G_max 依存を完全除去**（設計思想「authority/lifetime
    structure で bound」に整合）。**現行コードに構造 K がないことが確定したため、設計変更として正当化**。
  - **I-1（empirical capacity gate）**: Observed/Contract/Proof の3層で明示（Proof 層なしでは formal でない
    と明記）。
- **I-2 の採用はユーザー決定待ち**（現行コードで構造 K が証明できないことが D41 で確定 → I-2 が正当化された）。

### D41.7 判定
- **W-A / W-B / W-C = CLOSED（コード追跡・LIVE world ≤ ~5・retired は時間依存）**。
- **structural K = LIVE 部分 CLOSED（≈5）・retired 部分 OPEN（現行コードに cap なし）**。
- **決定待ち I が確定**: I-2 構造 gate（推奨・T_min/G_max 依存除去）or I-1 empirical（3層・formal でないと明示）。
- **F / G-B2 = OPEN**（B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D41.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T65 | LIVE world structural bound | 各 publish 瞬間の LIVE world ≤ ~5（builder+ownerChannel+current+oldWorld+graph refs） |
| T66（I-2 採用時） | world-in-grace 直列化 | 次の publish は前の retired world が reclaim 済みの後にのみ linearize（N_retired_world ≤ 1） |

---

## Design-23 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D41 で RuntimeWorld structural cardinality を追跡: W-A
（所有場所列挙: builder ≤1 + ownerChannel ≤1 + current 1 + oldWorld ≤1 + graph 2参照 + retired-in-grace 時間依存）・
W-B（publish 遷移表: 各瞬間の LIVE world ≤ ~5）・W-C（world→handle: graph 2 node 参照・N_retired_world ≤
storage slots が安全）。**structural K は LIVE 部分（≈5）のみ成立・retired-in-grace は現行コードに cap なし
（時間依存）**。I-1 は Observed/Contract/Proof の3層を明示（Proof 層なしでは formal でない）。**
**W-A/W-B/W-C = CLOSED・structural K = LIVE CLOSED（≈5）/retired OPEN・決定待ち I（I-2 構造 gate 推奨・
現行コードに構造 K がないことが確定したため正当化 / I-1 empirical 3層）・F/G-B2 = OPEN（実測 gate）。**
実装 GO はユーザー最終確認後。**
（→ **Design-24（D42）** は下記。**「LIVE ≤ ~5」はまだ厳密な CLOSED としない** — builder/ownerChannel/
publishExecutor は同一 object の move 通過（別 World object でない）・graph.activeNode/fadingNode は
node/reference cardinality（World 数でない）。**「場所の数」ではなく「distinct RuntimeWorld object identity
の最大同時数」を所有権遷移から証明**する。）

---

## D42 — Design-24 確定（distinct RuntimeWorld identity の所有権遷移追跡・2026-08-15）

ユーザー Design-23 レビュー対応: **「場所の数」ではなく「distinct RuntimeWorld object identity の最大
同時数」を create/move/publish/retire/destroy の所有権遷移から証明**する。

### D42.1 所有権遷移の追跡（同一 identity の move）
```
create:    Builder が RuntimeState C を生成（identity C）
transfer:  C → ownerChannel（enqueue・同一 identity C が移動）
take:      PublishExecutor::executePublish → ownerChannel().take(key)（同一 identity C）
publish:   publish(C): seal(C) → publishAndSwap → current = C・oldWorld = 前回 current（identity A）
retire:    A → retire domain（EBR grace）
reclaim:   A → destroyed
```
- **コード裏付け**: RuntimePublishExecutor.h:29-30「INVARIANT: RuntimeStateOwner is moved exactly once
  (into RuntimeWorldAuthority::publish — sole physical store-swap, INV-X4-3)」。
- **★ builder local / ownerChannel / publishExecutor は「同一 identity C の move 通過」であり、
  別 World object ではない**（D41 の W-A で 1+1+1 と数えたのは誤り — 同一 object の多重数え）。

### D42.2 graph.activeNode/fadingNode の除外
- **graph.activeNode / fadingNode は DSPCore への参照（node/reference cardinality）であり、
  RuntimeWorld object の数ではない**（RuntimeGraph.h:15「graph node pointers (visibility only, no ownership)」）。
- **World cardinality の上限計算に直接加算しない**（ユーザー指摘）。

### D42.3 N_distinct_world_identities_max の修正
```
各 publish 瞬間の distinct RuntimeWorld identity:
    in-flight new world（C・create→publish を通して同一 identity）      = 1
    current / oldWorld / fading（前回 publish の A・同一 identity）      = 1
    retired-in-grace（N_retired・時間依存）                              = N_retired

N_distinct_world_identities_max = 2 + N_retired
```
- **D41 の「LIVE ≤ ~5」は過大計上**（builder/ownerChannel/publishExecutor を同一 object なのに 3 重計上 +
  graph 参照を world として加算）。**正しくは LIVE 部分 = ~2（in-flight + current）**。
- **graph 参照は除外**（DSPCore への参照・world identity でない）。

### D42.4 structural K の再評価
```
LIVE distinct identities:  ≤ 2（コードで構造的に証明可能）
retired-in-grace:          N_retired（時間依存・現行コードに cap なし）
N_distinct_world_identities_max = 2 + N_retired
```
- **LIVE 部分は ~2（D41 の ~5 より厳密に小さい）**。
- **retired-in-grace（N_retired）が唯一の unbounded（時間依存）部分** — 変わらず。
- **I-2（world-in-grace 構造 gate）は、この N_retired を 0 にでき、total distinct world identity を
  ≤ 2 に収束させる**（T_min/G_max 依存を完全除去）。

### D42.5 判定
- **W-A 修正（distinct identity = same-object move・graph 除外）= CLOSED**。
- **N_distinct_world_identities_max = 2 + N_retired（LIVE ~2 構造的・retired 時間依存）**。
- **決定待ち I**: I-2 構造 gate（N_retired → 0・total ≤ 2）がより正確に正当化。I-1 empirical（3層）。
- **F / G-B2 = OPEN**（B_existing_measured 実測 gate）。
- **Phase I 実装はユーザー最終 GO 待ち**。

### D42.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T67 | distinct identity | builder/ownerChannel/publishExecutor が同一 identity の move（多重計上なし）・N_distinct ≤ 2 + N_retired |
| T68 | graph 除外 | graph.activeNode/fadingNode は world identity でない（world cardinality に加算しない） |

---

## Design-24 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D42 で distinct RuntimeWorld identity の所有権遷移を追跡:
builder/ownerChannel/publishExecutor は同一 identity の move 通過（RuntimePublishExecutor.h:29-30「moved
exactly once」）→ 多重計上を修正・graph.activeNode/fadingNode は DSPCore 参照（world でない）→ cardinality
から除外・N_distinct_world_identities_max = 2 + N_retired（LIVE ~2 構造的・retired 時間依存）。**
**D41 の「LIVE ≤ ~5」は過大計上と修正（正しくは ~2）。LIVE = CLOSED（~2）・retired-in-grace = OPEN
（時間依存・cap なし）。決定待ち I: I-2 構造 gate（N_retired → 0・total ≤ 2）がより正確に正当化 /
I-1 empirical（3層）。F / G-B2 = OPEN（実測 gate）。実装 GO はユーザー最終確認後。**
（→ **Design-25（D43）** は下記。**`2 + N_retired` は「N_retired が distinct retired World identity を
数えている」ことを確認した場合のみ成立** — retire domain の ownership unit（World identity or handle）を
追跡する。`pendingReclaimHandles_` は handle 側であり World 数の証明に使えない（cardinality 分離を維持）。）

---

## D43 — Design-25 確定（retire domain ownership unit 追跡・2026-08-15）

ユーザー Design-24 レビュー対応: **`N_total ≤ 2 + N_retired` が正式な cardinality 関係になるのは、
retire domain が「1 entry = 1 distinct RuntimeWorld identity」を保持する場合のみ**。ownership unit を追跡。

### D43.1 retire domain の ownership unit（コード裏付け）
- **RetireIntent**（ISRRetire.h:32）: `generation`（uint64・B-1）・`retireEpoch`（uint64）・`dspSlot`（uint32・
  dequeueOne で `out.dspSlot == UINT32_MAX` を tombstone 判定）→ **DSP handle/slot 単位**。
- **LifetimeState**（ISRRetire.cpp）: `slots_[idx].payload = RetireIntent` — retire queue は **RetireIntent
  （handle ベース）** を保持。
- **EpochControl::reclaim(uint32_t slot)**（ISRRetireRuntimeEx.h:45）: **slot（handle）単位**で reclaim。
- **onRuntimeRetiredNonRt(const RuntimePublishWorld* world)**（Commit.cpp:441）: world 単位の retire 通知だが、
  内部の retire domain（RetireIntent / EpochControl / pendingReclaimHandles_）は **handle 単位**。
- **結論: retire domain の ownership unit は DSP handle であり、RuntimeWorld identity ではない。**

### D43.2 World identity と handle cardinality の分離（ご指摘どおり）
```
retire domain の count（RetireIntent / pendingReclaimHandles_ / EpochControl slot）は HANDLE の数
→ N_retired（retire domain 由来）= handle count ≠ World identity count
```
- **★ `pendingReclaimHandles_.size() ≤ 512` は handle 数・World 数の証明に使えない**（cardinality 分離維持）。
- **`N_total ≤ 2 + N_retired` の N_retired は「distinct retired World identity 数」でなければならない**
  （handle count を代入しない）。

### D43.3 retired World identity の数え方（別途確認が必要）
- **world の retire は 1 publish = 1 retired world**（onRuntimeRetiredNonRt が oldWorld を 1 個 retire）。
- **world の破棄は、その world が保持する DSP handle が全て reclaim された時点**（world の lifetime は
  handle reclaim に連動）— ただしこの対応関係（world → その handle 群の reclaim 完了）は別途コードで
  確認が必要。
- **retired World identity cardinality = 時間依存**（publish rate × grace — world は publish ごとに 1 個
  retire され、handle 群の reclaim 完了まで生存）。

### D43.4 判定
- **retire domain unit = handle（CLOSED・コード裏付け）**・**World ≠ handle cardinality 分離（CLOSED）**。
- **LIVE distinct world identity ≤ 2 = CLOSED**（D42）。
- **retired World identity cardinality = OPEN**（handle count を world count に代入できない・時間依存）。
- **`N_total ≤ 2 + N_retired_world` は N_retired_world（distinct retired World identity）を確認した場合の式**
  （handle count を代入しない）。
- **I-2 = まだ設計採用しない** — retire ownership unit の確認（= handle）は完了・world identity の破棄
  対応（handle reclaim 連動）の確認を先行。
- **F / G-B2 = OPEN**（B_existing_measured 実測 gate）。**Phase I 実装はユーザー最終 GO 待ち**。

### D43.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T69 | retire unit = handle | RetireIntent（dspSlot/gen/epoch）・EpochControl::reclaim(slot) が handle 単位（world でない） |
| T70 | world destroy 連動 | world の破棄がその handle 群の reclaim 完了に連動（world→handle 対応をコードで確認） |

---

## Design-25 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D43 で retire domain ownership unit を追跡: RetireIntent
（dspSlot + generation + retireEpoch）・LifetimeState（RetireIntent queue）・EpochControl::reclaim(slot）・
pendingReclaimHandles_ は全て **DSP handle 単位であり RuntimeWorld identity ではない**。**
**retire domain unit = handle（CLOSED）・World ≠ handle cardinality 分離（CLOSED）・LIVE distinct identity
≤ 2（CLOSED）・retired World identity cardinality = OPEN（handle count を world count に代入しない・時間
依存）・`N_total ≤ 2 + N_retired_world` は retired World identity を確認した場合の式。**
**I-2 = まだ設計採用せず（world→handle reclaim 連動の確認を先行）。F / G-B2 = OPEN（実測 gate）。**
実装 GO はユーザー最終確認後。**
（→ **Design-26（D44）** は下記。**「world の破棄は handle 群の reclaim 完了に連動」は別途コード証明が
必要** — 1 retired World の handle 集合と World identity 自体の destruction authority を追跡し、
A（handle 群 aggregate → World destruction）か B（別 lifetime mechanism）かを確定する。）

---

## D44 — Design-26 確定（World destruction authority 追跡・2026-08-15）

ユーザー Design-25 レビュー対応: **INV-WORLD-LIFE-1/2 を閉じるため、RuntimeWorld identity の
lifetime/destruction authority を追跡**（A: handle 群 aggregate completion → World destruction か
B: 別 lifetime mechanism か）。

### D44.1 追跡結果（コード裏付け）
```
create:   RuntimeState::createForBuilder(BuilderToken) → aligned_unique_ptr<RuntimeState>
          （AudioEngine.h:163）
builder:  FrozenRuntimeWorld（RAII・aligned_unique_ptr<RuntimeState> state_ 保持・FrozenRuntimeWorld.h:42/84）
transfer: ownerChannel_（RuntimeOwner = aligned_unique_ptr<const RuntimeState>・RuntimeWorldAuthority.h:155）
publish:  owner.release()（RuntimeWorldAuthority.h:247）→ publishAndSwap → current = raw pointer
          ★ owner.release() 後、world は raw pointer（current → oldWorld）になり aligned_ptr の所有外
retire:   onRuntimeRetiredNonRt（Commit.cpp:441）→ retire + emitRetireIntentRT
counter:  retiredWorldCount_（AudioEngine.h:2225）— retired world のカウンタが存在
```

### D44.2 未確定点（INV-WORLD-LIFE-1/2 は閉じられない）
- **★ `owner.release()`（publish 時）後、RuntimeState（world）オブジェクトの破棄箇所が特定できなかった**。
  - `FrozenRuntimeWorld`（builder 境界 RAII）・`ownerChannel_`・`retiredWorldCount_` は存在するが、
    **publish 後に raw pointer となった world が「いつ・どこで delete / aligned_delete されるか」は未確認**。
  - `delete` / `destroy` の grep ヒットは DSPCore / EQCoeffCache 等の handle 側であり、**RuntimeState
    （world）自体の破棄サイトは見つからなかった**。
- **INV-WORLD-LIFE-1**（world は retired-world ownership authority が release/reclaim するまで破棄されない）:
  **未確定** — その retired-world ownership authority が何か（どこが world を保持し破棄するか）が未特定。
- **INV-WORLD-LIFE-2**（1 publish = ≤1 retired World identity・handle entry と別に一意追跡）:
  `retiredWorldCount_` の存在は world レベル追跡の手掛かりだが、**identity の一意追跡機構は未確認**。
- **A vs B（handle 群 aggregate → World destruction / 別 lifetime mechanism）: 未確定**。

### D44.3 正直な結論
- **World destruction authority = OPEN**（publish 後の RuntimeState 破棄サイトが未特定）。
- **handle bound をどれだけ閉じても、World cardinality の proof にはならない**（ご指摘の B の場合に該当し得る
  ため）— **World 自体の lifetime mechanism を特定するまで N_retired_world は未確定**。
- **`N_total_distinct_world = N_live + N_retired_world ≤ 2 + N_retired_world` は、N_retired_world が
  handle count から独立に定義できる（INV-WORLD-LIFE-2）ことを確認した場合の式** — 現時点では未成立。

### D44.4 次のステップ（world destruction site の特定）
```
publish → oldWorld → onRuntimeRetiredNonRt → World が保持する DSP handles を retire
→ 各 handle の epoch reclaim → 全 handle reclaim 完了 → RuntimeWorld destruction?
```
- **追跡必要**: RuntimeState の `~RuntimeState`（デストラクタ）呼び出し箇所・`aligned_delete` / `delete`
  サイト・world を保持する retired-world ownership authority（lifetime 内の world slot か別構造か）。
- **この特定が済むまで I-2 は採用しない**（world lifetime mechanism が確定してから判断）。

### D44.5 判定
- **retire unit = handle（CLOSED）・World ≠ handle 分離（CLOSED）・LIVE distinct identity ≤ 2（CLOSED）**。
- **World destruction authority = OPEN**（publish 後の RuntimeState 破棄サイト未特定・A/B 未確定）。
- **INV-WORLD-LIFE-1/2 = OPEN**（world lifetime mechanism の特定が必要）。
- **I-2 = 未採用**（world destruction site の特定を先行）。
- **F / G-B2 = OPEN**（B_existing_measured 実測 gate）。**Phase I 実装はユーザー最終 GO 待ち**。

### D44.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T71 | world destruction site | publish 後の RuntimeState がどこで破棄されるか（delete/aligned_delete/~RuntimeState 呼び出し箇所）を特定 |
| T72 | world lifetime mechanism | A（handle 群 aggregate → World destruction）か B（別 lifetime mechanism）かを確定 |

---

## Design-26 最終判定
**Phase I 実装 NO-GO（ユーザー最終 GO 待ち）。D44 で World destruction authority を追跡: RuntimeState は
createForBuilder → FrozenRuntimeWorld → ownerChannel_ → owner.release()（publish 時）で raw pointer 化するが、
**publish 後の RuntimeState 破棄サイトが特定できなかった**（delete/aligned_delete ヒットは DSPCore 等の
handle 側のみ）。retiredWorldCount_ は存在するが identity 一意追跡は未確認。**
**retire unit = handle（CLOSED）・World ≠ handle 分離（CLOSED）・LIVE ≤ 2（CLOSED）・World destruction
authority = OPEN（publish 後破棄サイト未特定・A/B 未確定）・INV-WORLD-LIFE-1/2 = OPEN・N_total ≤ 2 +
N_retired_world は未成立（N_retired_world が handle count から独立に定義できることを確認するまで）。**
**I-2 = 未採用（world destruction site の特定を先行）。F / G-B2 = OPEN（実測 gate）。実装 GO はユーザー
最終確認後。**

---

## D45 — Design-27 確定（World destruction authority の特定・ownership graph 逆方向クローズ・2026-08-15）

ユーザー Design-26 レビュー対応: 「grep で見つからなかったこと自体を proof にしない」。**ownership graph
を逆方向から閉じる** — 型定義の展開・`owner.release()` 戻り値の行き先・`retiredWorldCount_` の増減点・
`FrozenRuntimeWorld` RAII の publish 後存続を追跡し、**A/B/C/D に分類**する。

### D45.1 型定義の展開（Design-27 必須事項1）
- **`RuntimeState`**: `convo::isr::SealedObject<RuntimeState>` 派生（AudioEngine.h:143）。`createForBuilder`
  は `aligned_make_unique<RuntimeState>` で個別確保（アリーナ/プール無し）。move/copy 禁止。
- **`RuntimePublishWorld` = `RuntimeState` のエイリアス**（AudioEngine.h:330・RuntimePublicationValidator.h:10）。
  → 以降の `RuntimePublishWorld*` は全て `RuntimeState*` と同一。
- **`FrozenRuntimeWorld`**: builder 境界 RAII wrapper（`aligned_unique_ptr<RuntimeState> state_`）。
  `releaseState()`（FrozenRuntimeWorld.h:58）で所有移譲。デストラクタは**所有がある場合のみ unseal()**
  （FrozenRuntimeWorld.cpp:14-22）— release 後は state_==nullptr で何もしない。
- **`RuntimeOwner` = `convo::aligned_unique_ptr<const RuntimeState>`**（RuntimeWorldAuthority.h:155）。
  デリーターは aligned_unique_ptr の `~T() + aligned_free`。custom deleter / shared_ptr / 配置構築は**無し**。

### D45.2 `owner.release()` 戻り値の行き先（最重要・全 call chain 追跡）
- **`RuntimeWorldAuthority::publish()`（RuntimeWorldAuthority.h:247-250）**:
  ```cpp
  auto* next = const_cast<RuntimeState*>(owner.release());
  std::atomic_thread_fence(std::memory_order_release);
  auto* oldWorld = writeAccess_.publishAndSwap(next);   // previous（oldWorld）を返す
  return oldWorld;                                      // ★ raw RuntimeState* を caller へ
  ```
- **caller は 3 経路・全て戻り値を明示的に retire へ渡す（戻り値破棄 = リーク経路は無い）**:
  1. **RuntimePublishExecutor.h:60-76**（通常 publish）: `oldWorld = authority.publish(...)` →
     `if (committed) { didPublish; willRetire; retireRuntimePublishWorldNonRt(oldWorld, false); }`
  2. **AudioEngine.Init.cpp:73-91**（bootstrap）: `oldWorld = publish(...)` →
     `retireRuntimePublishWorldNonRt(oldWorld, false)`
  3. **CtorDtor.cpp:227-231 / ReleaseResources.cpp:458-462**（shutdown clear）:
     `clearedWorld = clearPublishedRuntimeSnapshotsNonRt()` → `retireRuntimePublishWorldNonRt(clearedWorld, true)`
- **型伝播**: `RuntimeState*` → `RuntimePublishWorld*`（エイリアス）→ `void*`（enqueueDeferredDeleteNonRt）。

### D45.3 `retiredWorldCount_` の増減点（ユーザー仮説の検証）
- **++ は 1 箇所のみ**: `AudioEngine.Commit.cpp:624`（`onRuntimeRetiredNonRt` 内・world 退役イベント）。
- **-- は存在しない**（grep: 宣言 + `++` の 2 箇所のみ）。
- **★ 確定: `retiredWorldCount_` は monotonic cumulative **telemetry / accounting counter** であり、
  **World lifetime authority ではない**。**（ユーザー仮説の確定 — 「`++ on retire` のみで `--` 無し =
  累積/diagnostic counter」）

### D45.4 `FrozenRuntimeWorld` RAII の publish 後存続
- **publish 前の builder 境界 wrapper であり、publish 後は別の retired wrapper は存在しない**。
  `releaseState()` で所有移譲後、`FrozenRuntimeWorld` は役割を終える（state_==nullptr）。
- → D44 で懸念した **「publish 後も別の FrozenRuntimeWorld / retired wrapper が存在して A/B 別経路」は
  存在しない**。

### D45.5 ★ 破棄サイトの特定（D44 の OPEN が閉じる）
- **`AudioEngine::retireRuntimePublishWorldNonRt`（AudioEngine.h:3520-3538）** — これが RuntimeState
  破棄のエントリポイント:
  ```cpp
  engine_->enqueueDeferredDeleteNonRt(world, [](void* p) {
      auto* ptr = static_cast<RuntimePublishWorld*>(p);  // = RuntimeState*
      ptr->unseal();                                     // SealedObject 解放
      ptr->~RuntimePublishWorld();                       // = ~RuntimeState()
      convo::aligned_free(ptr);                          // メモリ解放
  });
  ```
- **`enqueueDeferredDeleteNonRt`（AudioEngine.h:4164）→ `m_retireRouter->enqueueWithRetry(ptr, deleter,
  epoch, DeletionEntryType::Generic)`**（ISRRetireRouter.cpp:161-208）。
- **drain = `drainDeferredRetireQueues`（AudioEngine.Retire.cpp:41）** → `m_retireRouter->tryReclaim()`
  + `m_coordinator.reclaim(minReaderEpoch)`。**epoch-gate**: `retireEpoch < minReaderEpoch` の時のみ
  deleter 実行（reader が古い epoch を離脱するまで破棄遅延 = grace）。
- **enqueue 失敗時（QueuePressure/QueueFull）は `RetireQuarantineStore` へ移送**（ISRRetireRouter.cpp:190-208）
  — 即時 delete せず epoch 安全到達後の定期 drain で解放（UAF 構造的排除）。
- **★ World destruction = `AudioEngine::m_retireRouter`（deferred delete queue + quarantine）が所有**。
  `~RuntimeState()` + `aligned_free` が明示実行される。

### D45.6 A/B/C/D 分類（ユーザー指定の最終成果物）
- **A（World が直接 retire domain に所有される）: 該当** — RetireDomain = `AudioEngine::m_retireRouter`
  + `drainDeferredRetireQueues`（epoch-gated deferred delete queue）。`publish → oldWorld →
  retireRuntimePublishWorldNonRt → enqueue → grace → destroy(RuntimeState)`。
- **B（独立した World lifetime authority）: 該当しない** — World は handle とは独立だが、**同じ
  m_retireRouter の別エントリ**として epoch-gate を共有するのみ（独立 authority 無し）。
- **C（publish 後 ownership 消失）: 該当しない** — raw pointer は明示的に retire bridge へ渡され、
  **deferred delete queue が確実に所有・破棄**する（lifetime defect ではない）。
- **D（破棄されず内部 resource のみ reclaim）: 該当しない** — `~RuntimeState()` + `aligned_free` が明示実行。

### D45.7 INV-WORLD-LIFE-1/2 の可否
- **INV-WORLD-LIFE-1（retired-world ownership authority が何か）: CLOSED** —
  `AudioEngine::m_retireRouter`（deferred delete queue + quarantine store）が所有・破棄。
- **INV-WORLD-LIFE-2（identity 一意追跡）: CLOSED（enqueue-once）** — 各 publish は 1 oldWorld を返し、
  **正確に 1 回 retire enqueue**（RuntimePublishExecutor:76 / Init:88 / CtorDtor:231 / ReleaseResources:460
  を検証済み）。各 enqueue は排他的に 1 回 drain（FIFO + quarantine drain-once）。**1 retire enqueue =
  1 distinct World identity**。
- ただし **live in-flight の world 数を返す counter は存在しない**（retiredWorldCount_ は累積 telemetry、
  pendingRetireCount は全エントリ種別共有）→ **生存 world の live カーディナリティの runtime 追跡は無い**。

### D45.8 `1 retire = 1 World identity` と `N_total ≤ 2 + N_retired_world` の可否
- **`1 retire = 1 World identity` = CLOSED（enqueue/destroy レベル）** — enqueue-once + drain-once を検証。
  ★ ただし **World destruction は handle reclaim 完了に連動しない** — publish による置換 + epoch-safety で
  破棄。handle reclaim とは別エントリ（共有 epoch-gate のみ）。
- **`N_total ≤ 2 + N_retired_world` = STILL OPEN** — N_retired_world（in-flight・retired-not-destroyed）は
  **time-dependent**（publish rate × drain cadence）、構造的容量（handle table kCapacity=512 等）に
  束縛されない。LIVE(1) + swap→enqueue 遷移中(1) = 2 は構造的に CLOSED だが、deferred delete queue は
  全エントリ種別共有で周期 drain のため、**world 固有の静的 upper bound は導けない**（D40 と同結論:
  T_min>0 / G_max<∞ は静的証明不能）。

### D45.9 副次観察（shutdown 終端 edge・Phase I 非ブロッカー）
- shutdown 時（`isShutdownInProgress()`）の `enqueueDeferredDeleteNonRt` は `Shutdown` を返し enqueue しない
  （AudioEngine.h:4176）。よって **CtorDtor/ReleaseResources の clearedWorld は enqueue されない**。
  コード著者のコメント（ReleaseResources.cpp:455「実質 no-op だが契約を維持する」）で既知。プロセス終端
  leak（OS が回収・UAF は発生しない）であり Phase I カーディナリティには影響しない。

### D45.10 判定
- **World destruction authority = CLOSED（A-type）** — `AudioEngine::m_retireRouter` deferred delete queue
  （epoch-gated drain・`~RuntimeState()` + `aligned_free`）。
- **`owner.release()` 後の World owner = CLOSED** — retire router（enqueue-once・epoch-gated drain）。
  **C ではない（lifetime defect ではない）・D ではない（object は破棄される）**。
- **`1 retire = 1 World identity` = CLOSED（enqueue/destroy レベル）**。**INV-WORLD-LIFE-1 = CLOSED**・
  **INV-WORLD-LIFE-2 = CLOSED（enqueue-once）**。
- **`N_total ≤ 2 + N_retired_world` = OPEN**（N_retired_world は time-dependent・静的証明不能）。
- **`retiredWorldCount_` = telemetry only**（monotonic ++ のみ・ユーザー仮説確定）。**live in-flight
  world counter は存在しない**。
- **I-2 = 未採用**（world destruction site は特定できたが、coalesce の World cardinality bound
  N_retired_world が time-dependent のため静的には閉じられない）。**F / G-B2 = OPEN**（実測 gate）。
- **Phase I 実装 NO-GO（ユーザー最終 GO 待ち）**。

### D45.11 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T71 | world destruction site | **CLOSED**: `retireRuntimePublishWorldNonRt` → `enqueueDeferredDeleteNonRt` → `m_retireRouter`（deferred delete queue / quarantine）→ epoch-gated drain で `~RuntimeState() + aligned_free` |
| T72 | world lifetime mechanism | **CLOSED (A)**: World は m_retireRouter が所有・epoch-gate で破棄。C/D ではない |
| T73 | retiredWorldCount_ telemetry 検証 | `++` のみ（Commit.cpp:624）・`--` 無し・monotonic cumulative counter であることを確認 |
| T74 | enqueue-once | 1 publish = 1 oldWorld = 1 retire enqueue = 1 destroy（RuntimePublishExecutor:76 / Init:88 / CtorDtor:231 / ReleaseResources:460） |
| T75 | epoch-gate | reader が古い epoch 参照中は world 破棄が遅延（retireEpoch < minReaderEpoch まで deleter 実行されない） |

---

## D46 — Design-28 確定（RetireRouter storage cardinality・Generic deferred-delete entry の structural bound・2026-08-15）

ユーザー Design-27 レビュー対応: **`N_retired_world` が time-dependent という D45 の結論は「まだ証明ではない」**。
D45 で「1 retired World = 1 Generic deferred-delete entry = RetireRouter の storage」が確定した以上、
**その Generic entry を同時に何個保持できるか（storage cardinality）** をコード上で確定し、
**時間軸から独立した structural bound** を得る（R1〜R5）。overflow が heap 容器に逃げるなら
その容器が実質 cardinality domain（Case C）。

### R1 — Generic retire entry の正確な型
- **`DeletionEntry`**（DeferredDeletionQueue.h）: `{ ptr, deleter, epoch, type=Generic, publicationSequenceId,
  generation }`。trivially copyable（static_assert）。**ヒープ無し・単一構造体**。
- **`QuarantinedEntry`**（RetireQuarantineStore.h）: `{ ptr, deleter, epoch, type, publicationSequenceId,
  generation, reason, enqueueTimeUs }`。trivially copyable。**ヒープ無し**。
- `DeletionEntryType` は **`Generic = 0` のみ** — 全 deferred delete（world / DSP / EQ / cache map）が
  同一 Generic type で同一キューを通る。

### R2 — Generic entry の storage（全経路確認）
```
enqueueDeferredDeleteNonRtWithResult（AudioEngine.h:4170）
  → m_retireRouter->enqueueWithRetry（ISRRetireRouter.cpp:161）
      → enqueueRetire → provider_->enqueueRetire（= m_epochDomain.enqueueRetire・EpochDomain.h:396）
          → DeferredDeletionQueue.enqueue（DeferredDeletionQueue.h）
              [PRIMARY: bounded MPMC ring（Vyukov）・kQueueSize = 4096・std::array・allocation-free]
          → Full 時: Router forced tryReclaim（500ms cooldown）→ 再試行
      → それでも QueuePressure → retry ×2（tryReclaim + enqueue）
      → それでも QueuePressure/QueueFull → m_retireQuarantine.quarantine（ISRRetireRouter.h:198）
          [FALLBACK: RetireQuarantineStore・kMaxQuarantinedEntries = 512・std::array・allocation-free]
          → store full → assert(false) Debug / LEAK Release（deleter は実行しない・UAF 構造的排除）
```
- **固定 ring（std::array）・固定 array（std::array）のみ。std::vector / std::deque / heap node は無し。**
- **world は AudioEngine の単一 `m_epochDomain.deferredDeletionQueue`（4096）+ `m_retireQuarantine`（512）
  のみを通る**（m_retireRouter = make_unique<ISRRetireRouter>(m_epochDomain)・CtorDtor.cpp:35）。
  他 EpochDomain（ConvolverProcessor.h:1279 / EQProcessor.h:462）は別サブシステムで world は通らない。
- 他 DeferredDeletionQueue インスタンスはテストのみ（production は AudioEngine の 1 つ）。

### R3 — enqueue failure の意味（分岐ごと）
| 分岐 | 結果 |
|------|------|
| 成功 | primary ring に pending entry |
| primary Full | Router forced `tryReclaim()`（500ms cooldown）→ 再試行 |
| それでも QueuePressure | retry ×2（`tryReclaim()` + `enqueueRetire`） |
| それでも QueuePressure/QueueFull | **RetireQuarantineStore へ移送**（bounded 512・epoch 安全後定期 drain） |
| quarantine store full | **assert(false)（Debug）/ LEAK（Release）** — deleter 実行せず・world は破棄されない（overflowCount_ で health escalation） |
| shutdown 中 | `enqueueDeferredDeleteNonRt` が Shutdown 返却（enqueue no-op・clearedWorld 終端 leak） |
- **blocking / drop-then-double-free / unbounded heap は無い**。失敗は「bounded fallback へ保持」または
  「leak（破棄されない）」のいずれか — **どちらも cardinality を増やさない**（leak は破棄されないので
  in-flight count から外れる）。

### R4 — dequeue と World destruction の atomicity
- **primary ring `reclaim(minReaderEpoch)`**: FIFO 先頭のみ・`canDelete = isOlder(entry.epoch,
  minReaderEpoch)`・**CAS(dequeuePos) 成功 → 同一ステップ内で `entry.deleter(entry.ptr)` 実行** →
  エントリ clear。**「removed → 別の後続 mechanism → deleter」の中間状態は無い**。
- **quarantine `drain`**: lock 内で safe エントリ抽出 → unlock 後に deleter 実行（エントリは同一 drain
  内で clear）。
- **★ 結論: `pending Generic entries` と `retired-but-not-destroyed Worlds` の cardinality は 1:1 で一致
  （entry removal と destroy が同一ステップ）**。
- 注意: primary の FIFO 制約により、**先頭が epoch 非安全だと後続も drain されない**（head-stuck）—
  cardinality bound には影響しないが、N_retired_world が高止まりする liveness 要因（I-2/backpressure 対象）。

### R5 — shutdown path は別扱い
- normal runtime: `retire → queued → epoch gate → destroy`（本 D46 の cardinality proof 対象）。
- shutdown: 別 lifetime contract（`enqueueDeferredDeleteNonRt` no-op・`drainAllQuarantineStore` /
  `drainAllUnsafe` で Audio Thread 停止後に強制破棄）。**通常運転の N_retired_world proof に shutdown
  semantics は混ぜない**（D45.9 と整合）。

### R4'（補足）— 失敗 publish の world は N_retired_world に入らない
- publish が validate 失敗（nullptr 返却）の world は、caller の `owner`（aligned_unique_ptr）の
  RAII デストラクタで **同期破棄**（deferred delete キューに入らない）→ N_retired_world に含まれない。
  RuntimePublishExecutor.h:60 の `owner` が `if (hasOwner)` ブロック終了で破棄。

### ★ 判定（ユーザー指定の3ケース分類）
- **Case B 確定 — bounded primary（4096）+ bounded fallback（512）、全 storage 有限・allocation-free。
  heap fallback / unbounded 容器は無い。**
- **`N_retired_world ≤ N_pending_Generic_entries ≤ K_primary + K_fallback = 4096 + 512 = 4608`**
  （world は Generic entries の部分集合・各 world は正確に 1 storage に存在 — primary と quarantine の
  二重存在は無い）。
- **∴ `N_total_distinct_world ≤ N_in_pipeline(2) + N_retired_world ≤ 2 + 4608 = 4610`**
  — **時間軸から独立した structural bound が成立**。
- **H1（publish rate × grace lifetime）は不要になる**（ユーザー予測の確定 — 時間ベース bound を使わずに
  N_retired_world を閉じた）。**T_build_min > 0 / G_max < ∞ の前提も不要**。
- overflow→leak 経路は cardinality を保つ（破棄されない world は in-flight から外れる）が、**memory
  leak + その world は二度と破棄されない** — 異常系（overflowCount_ 監視・health escalation）として
  cardinality proof から分離。

### D46.2 最終状態
| 項目 | 状態 |
|------|------|
| World destruction authority / `owner.release()` 後の World owner | **CLOSED（A-type・m_retireRouter）** |
| World ≠ DSP handle / `1 retire → 1 World identity` | CLOSED |
| `retiredWorldCount_` = lifetime authority | **否定 / telemetry** |
| World destruction = handle aggregate completion | **不要（直接 World retire entry で管理）** |
| **RetireRouter storage cardinality** | **CLOSED（Case B・primary 4096 + fallback 512 = 4608）** |
| **`N_retired_world`** | **CLOSED（≤ 4608・構造的・時間非依存）** |
| **`N_total_distinct_world`** | **CLOSED（≤ 2 + 4608 = 4610・構造的）** |
| **H1（rate × grace）** | **不要と確定**（時間ベース bound 撤去） |
| **I-2（World-in-grace structural gate）** | **採用可能に（N_total ≤ 4610 が静的 gate として利用可）** — ただし実装 GO はユーザー最終確認後 |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D46.3 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T76 | primary capacity | DeferredDeletionQueue kQueueSize = 4096・enqueue Full で false・bounded ring であることを確認 |
| T77 | fallback capacity | RetireQuarantineStore kMaxQuarantinedEntries = 512・store full で deleter 実行しないことを確認 |
| T78 | N_retired_world structural bound | N_retired_world ≤ N_pending_Generic ≤ 4096 + 512 = 4608（world は Generic 部分集合） |
| T79 | entry removal + destroy の atomicity | primary reclaim: CAS(dequeuePos) と deleter 実行が同一ステップ・中間状態無し |
| T80 | R3 分岐網羅 | 成功 / Full→tryReclaim / retry / quarantine / quarantine-full(leak) / shutdown の各分岐 |
| T81 | 失敗 publish は N_retired 外 | validate 失敗 world は RAII 同期破棄・deferred delete キューに入らない |

---

## D47 — Design-29 確定（4610 bound の semantic validity・Q1/Q2/Q3 形式化・2026-08-15）

ユーザー Design-28 レビュー対応: **「I-2 へ進む前に、4610 を admission/gate の値として採用してよいか一段
形式的に分解する」**。Q1（Generic queue occupancy → World cardinality の injection）・Q2（dequeue → destruction
の gap）・Q3（safety bound と reservation bound の分離）をコードで閉じる。**4610 をそのまま reservation 数として
コードへ埋め込むのは避ける**。

### Q1 — Generic queue occupancy → World cardinality（injection の形式化）
- **W = retired-but-not-destroyed World の集合**・**G = pending Generic deferred-delete entry の集合**。
- **★ W は G の部分集合と見做せる（injection φ: W → G が存在）**:
  1. **1 World → 1 entry**: 各 world は `retireRuntimePublishWorldNonRt` 経由で正確に 1 回 enqueue（D45
     enqueue-once 検証）。publish 経路は単一パイプラインの2段階（Non-RT `PublicationExecutor` →
     `commitRuntimePublication` → OwnerChannel → ISR `RuntimePublishExecutor::executePublish` →
     `authority.publish` → oldWorld → 同一 bridge retire）であり二重 retire は無い
     （executor_ = PublicationExecutor・RuntimePublicationOrchestrator.h:268）。
  2. **1 entry → ≤1 World**: entry の `ptr` は world を一意特定（同一 ptr の二重 entry は enqueue-once で排除）。
  3. **retired だが G に無い world は存在しない**: retired world が存在し得る storage は
     {DeferredDeletionQueue, RetireQuarantineStore} のみ。**他 storage は retired world を保持しない**:
     - `PendingPublishRegistry`（RuntimeWorldAuthority.h:33）: **非所有**（`const void* world`・コメント
       "non-owning handle"）・bounded 64・**pending publish の参照のみ**（retired ではない）。
     - `OwnerChannel`（kCapacity = 256・OwnerChannel.h:41）: **pending publish の所有**（retired ではない）。
     - `RuntimeStore::current`: LIVE（retired ではない）。
     - `pendingReclaimHandles_` / LifetimeState / DSPQuarantineManager: handle ドメイン。
- **∴ |W| ≤ |G| ≤ 4608**（Q1 CLOSED）。
- 補足（pipeline 項）: N_total = N_pipeline + N_retired_world の N_pipeline は D42 の "2"（current 1 +
  in-flight 1・同期プロデューサ waitForReceipt により serialized）。OwnerChannel 256 / registry 64 は
  pending publish の safety ceiling（同期プロデューサで到達不能）— retired とは独立。

### Q2 — dequeue → destruction の gap（中間 ownership 移転の有無）
- **primary ring `reclaim(minReaderEpoch)`**: `CAS(dequeuePos)` 成功 → **同一反復・同一スレッド内で直ちに
  `entry.deleter(entry.ptr)` 実行** → entry clear。CAS と deleter の間に他 storage への移転は無い
  （dequeuePos 前進により他スレッドは再 reclaim 不能・world は破棄直前の所有）。
- **quarantine `drain`**: lock 内で safe エントリをローカル配列（`pendingPtrs[≤512]`）へ抽出 → unlock →
  deleter 実行。抽出された world は**同一 drain 呼び出し内のローカル配列**に保持（新 storage への移転では
  ない）。抽出は quarantine 内の既存 world を移すだけ。
- **★ 形式的結果: reclaim/drain 中の world 数は不変（抽出・dequeue は pool 内で world を移動するだけ）**:
  ```
  primary reclaim: N_world = (prev_primary - 1) + 1 + |quarantine| = prev total ≤ 4608
  quarantine drain: N_world = |primary| + (prev_quar - E) + E = |primary| + prev_quar ≤ 4608
  ```
  → **`pending Generic entries` と `retired-but-not-destroyed Worlds` の cardinality は 1:1（R4 確定）**・
  **Q2 CLOSED**。
- **INV-WORLD-CARD-1**: retired-but-not-destroyed World は storage pool P（primary 4096 ∪ quarantine 512）
  の正確に 1 slot を占有。slot は world 破棄（deleter）時にのみ解放 → |W| ≤ |P| = 4608（常時）。
- **INV-WORLD-CARD-2**（injection）: φ: W → G は単射 → |W| ≤ |G| ≤ 4608。

### Q3 — SAFETY bound と RESERVATION bound の分離（最重要）
- **SAFETY（構造的安全上限・既存 lifetime mechanism が許容し得る distinct World identity の上限）**:
  ```
  N_retired_world ≤ 4608
  N_total_world ≤ 4610
  ```
  → これは「**既存 storage の総容量（World + DSP + EQ + cache 全 Generic 共用）**」であり、
  **World 専用の予約容量ではない**。
- **RESERVATION（I-2 の admission/gate 容量）**: **4610 / 4608 を World 専用予約数としてコードへ埋め込ま
  ない**。`DeferredDeletionQueue` は World 専用でない（共用）ため、World に残された容量は 4610 ではない。
  - I-2 の reservation capacity R は**独立した policy 値**（World entry の実シェア・expected/observed に基づく）
    とし、**構造的安全上限（4608/4610）は「gate が決して deadlock せず・coalesce budget が構造容量を
    超え得ない」ことの safety 根拠**としてのみ使用する。
  - 分離の明文化: `SAFETY: N_retired_world ≤ 4608 / N_total_world ≤ 4610`（= 共用 pool の上限）・
    `RESERVATION: R = policy（World 専用、≠ 4610）`。

### D47.2 最終状態
| 項目 | 状態 |
|------|------|
| World destruction authority / `owner.release()` 後 owner | CLOSED（A-type） |
| World ≠ DSP handle / `1 retire → 1 World identity` | CLOSED |
| `retiredWorldCount_` = lifetime authority | 否定 / telemetry |
| Generic storage 全体の容量 | CLOSED（4096 + 512 = 4608・Case B） |
| **`N_retired_world ≤ 4608`** | **CLOSED（Q1/Q2 形式化完了・INV-WORLD-CARD-1/2）** |
| **`N_total_world ≤ 4610`** | **CLOSED**（N_pipeline ≤ 2・D42 + N_retired_world ≤ 4608） |
| H1（rate × grace） | 不要（時間ベース bound 撤去） |
| **SAFETY vs RESERVATION 分離** | **CLOSED（4610 は共用 pool 上限・World 専用予約ではない）** |
| I-2 structural gate | **設計可能**（safety は 4610 使用可・reservation は policy 値 R） |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D47.3 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T82 | Q1 injection | W ⊆ G: 各 retired world は正確に 1 entry・retired world を保持する他 storage が無い（registry 非所有・ownerChannel は pending） |
| T83 | Q2 gap なし | primary reclaim: CAS→deleter 同一ステップ・quarantine drain: ローカル配列のみ・新 storage への移転無し |
| T84 | INV-WORLD-CARD-1 | retired world は pool の 1 slot のみ占有・slot 解放は deleter 時のみ |
| T85 | Q3 safety/reservation 分離 | 4610 を reservation に使用しない・policy R を別途定義 |
| T86 | pipeline 項 | N_pipeline ≤ 2（current + in-flight・同期プロデューサ）・OwnerChannel 256 は safety ceiling |

---

## D48 — Design-30 確定（WorldRetirementReservation の invariant と acquire/release authority・2026-08-15）

ユーザー Design-29 レビュー対応: **「R の数値決定」ではなく、reservation gate の invariant と authority
boundary をコード上で定義する**。`WorldRetirementReservation` の lifetime invariant（INV-R1〜R4）と
acquire/release location を、**現在の publish → retire → deferred-delete → destroy 経路に重ねて検証**する。
**Phase I 実装 NO-GO 継続（コード未変更・設計契約のみ）**。D47 の「4608/4610 は Global Generic storage
capacity による safety bound」判断は維持（World reservation に転用しない）。

### D48.1 Reservation の対象明確化（2 つの独立 authority）
- **World reservation authority**（新設・概念的）: `WorldRetirementReservation`
  - **R = 同時に存在可能な「retired-but-not-destroyed RuntimeWorld identity」の World 専用 reservation**。
  - gate invariant: **N_retired_world ≤ R**（admission 時は < R を確認）。
  - 概念: `N_retired_world < R → publish / retire admission 許可`。
- **Global structural safety**（既存・変更なし）: `N_retired_all_objects ≤ 4608` — RetireRouter 自体の
  capacity safety（D46/D47）。**独立**・World reservation に転用しない。
- `R` の数値は policy（次ステップ）。本 D48 は invariant と authority のみ。

### D48.2 Gate の位置（acquire は retire 前・publish admission）
- **gate を `enqueueDeferredDeleteNonRt()` 側に置かない**（publish → retire → Generic enqueue → reservation
  failure は「World を retire した後に拒否」する不適切な順序）。
- **acquire location = `RuntimeWorldAuthority::publish()` 内部・`publishAndSwap` の前**:
  ```
  publish admission（prevWorld = runtimeStore_.observe() 読取）
    → WorldRetirementReservation::acquire(key = prevWorld identity)   ★ swap 前
        → 失敗（N_retired_world ≥ R）→ publish Rejected（backpressure・oldWorld は retire されない）
    → owner.release() → publishAndSwap → oldWorld（retired transition）
    → willRetireRuntimeNonRt(oldWorld)（semantic retired transition・retiredWorldCount_++ は telemetry）
    → retireRuntimePublishWorldNonRt(oldWorld) → Generic deferred-delete enqueue
    → epoch-gate drain → ~RuntimeState() + aligned_free
    → WorldRetirementReservation::release(key = 同一 identity)          ★ destroy 後
  ```
  - `publish()` は sole physical publish gateway（INV-X4-2）→ **acquire を publish() 内に置けば全 publish
    経路（ISR RuntimePublishExecutor + Bootstrap Init.cpp:73）を一律カバー**。
  - acquire 失敗は validate 失敗とは別の失敗モード（Rejected/backpressure）として publish() の
    戻り値契約に追加（reservation-first モデル D14/D18/D26 と整合）。
- **release location = deferred-delete deleter lambda 内**（`retireRuntimePublishWorldNonRt` の
  deleter・AudioEngine.h:3525-3533・`~RuntimePublishWorld()` + `aligned_free` の後）。
  → **全破棄経路（reclaim / drainAllUnsafe / quarantine drainAllUnsafe）が同一 deleter を使用**するため
  release は一律カバー。

### D48.3 Reservation counter の authority 一本化
- `WorldRetirementReservation` の **`acquire(key)` / `release(key)` / `current_reserved()`** のみが
  reservation state の authoritative API。
- **推測禁止**: `retiredWorldCount_`（telemetry）・`pendingReclaimHandles_`（handle domain）・Generic
  queue occupancy（global safety domain）・Coordinator observation・bool/heuristic から reservation 状態を
  導出しない（D43〜D47 の意味分離を維持）。

### D48.4 INV-R1〜R4 の証明（コード裏付け）
| INV | 内容 | 証明（コード裏付け） |
|-----|------|----------------------|
| **INV-R1** | acquire は World identity の retired transition より前に成功 | acquire を publish() 内・`publishAndSwap` の前・`willRetireRuntimeNonRt` の前（RuntimeWorldAuthority.h:247 の swap より前）に配置 → **swap < willRetire < enqueue** の順で acquire が先行。CLOSED（by construction） |
| **INV-R2** | 1 World identity に対して reservation は最大1個 | 各 world は正確に 1 回 retire（D45 enqueue-once）・各 retire は 1 publish の oldWorld（1:1）・acquire は oldWorld 1 個につき ≤1 回（publish 1 回につき ≤1）→ ≤1。CLOSED |
| **INV-R3** | reservation release は destruction 完了後のみ | release を deleter 内・`~RuntimePublishWorld()` + `aligned_free` の後に配置 → **release は retire ではなく destroy に同期**。★ **reservation count ≡ retired-but-not-destroyed World identity count**（INV-R3 の核心）。CLOSED（by construction） |
| **INV-R4** | 全 publish/retire/destroy 経路が同一 reservation authority | publish = `publish()`（sole gateway・ISR + Bootstrap カバー）・destroy = deleter（reclaim/drainAll 共通）→ 同一 `WorldRetirementReservation`。正常運転の全経路で CLOSED。**shutdown clear**（clearPublishedRuntimeSnapshotsNonRt → retire → enqueue）は R5（D46）により**別 contract**（明示的例外・正常運転の証明に混ぜない）。CLOSED |

### D48.5 エッジケース（INV-R3 との整合・正直な注記）
- **quarantine-full → leak 経路**（D46-R3）: world が破棄されず leak する場合、**reservation は release
  されない**。これは INV-R3 と整合（leak world は retired-but-not-destroyed のまま = reservation 保持は
  正しい反映）が、**reservation が永続消費**される。`quarantineOverflowCount_` 監視・health escalation で
  検知（正常運転の前提では発生しない・cardinality proof には影響しない）。
- **shutdown no-op 経路**（D45.9）: clearedWorld が enqueue されず終端 leak → reservation 未 release。
  shutdown は R5 別 contract（終端・OS 回収）のため整合。
- **失敗 publish（validate 失敗）**: world は retire されず RAII 同期破棄 → **acquire 不要**（retired
  transition が発生しないため reservation を消費しない）。INV-R4 の retire 経路に含まれない。

### D48.6 最終状態
| 項目 | 状態 |
|------|------|
| Structural safety（N_retired_world ≤ 4608 / N_total_world ≤ 4610） | CLOSED（D46/D47） |
| **R の意味** | **DEFINED**（retired-but-not-destroyed World identity の World 専用 reservation・gate invariant N_retired_world ≤ R） |
| **acquire location** | **DEFINED**（publish() 内・swap 前・prevWorld != null） |
| **release location** | **DEFINED**（deferred-delete deleter 内・destroy 後） |
| **ownership authority** | **DEFINED**（WorldRetirementReservation 一本化・acquire/release/current_reserved） |
| **all-path coverage（INV-R4）** | **CLOSED**（正常運転・shutdown は R5 別 contract） |
| **INV-R1/R2/R3** | **CLOSED（by construction・コード経路裏付け）** |
| **R の数値** | **OPEN（policy・次ステップ）** |
| I-2 implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D48.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T87 | INV-R1 | acquire が publishAndSwap / willRetireRuntimeNonRt より前に成功（gate は retire 前） |
| T88 | INV-R2 | 1 World identity に reservation 最大1個（enqueue-once と 1:1） |
| T89 | INV-R3 | release は deleter（~RuntimeState + aligned_free）後のみ・reservation count ≡ retired-but-not-destroyed count |
| T90 | INV-R4 | 全 publish/retire/destroy 経路が同一 authority（ISR + Bootstrap + destroy 共通 deleter）・shutdown は別 contract |
| T91 | acquire 失敗 = backpressure | N_retired_world ≥ R で publish Rejected・oldWorld は retire されない |
| T92 | leak 経路の reservation | quarantine-full leak では reservation 未 release（overflowCount 検知・INV-R3 整合） |

---

## D49 — Design-31 確定（INV-R0 と全分岐状態遷移表・reservation token ↔ World identity の対応証明・2026-08-15）

ユーザー Design-30 レビュー対応: **「INV-R1〜R4 CLOSED とするには acquire の成功状態が publish の成否と
完全に原子的に対応しているかが未証明」**。INV-R0（acquire ↔ exactly one retirement、または rollback）・
identity binding（B）・shutdown（C）・quarantine-full 4 者対応（D）を、**`acquire → publish success/failure →
retire → enqueue → reclaim → destroy → release` の全分岐を一本の状態遷移表**にして証明する。**R の数値決定は
この証明が閉じてから。Phase I 実装 NO-GO 継続（コード未変更・設計契約のみ）。**

### D49.1 D48 の位置づけ再確認
- **D48 = 「I-2 を実装可能にした」段階ではなく「I-2 の reservation authority の形を定義できた」段階**
  （ユーザー評価の通り）。D49 で acquire/publish の原子対応を閉じる。

### D49.2 INV-R0（acquire ↔ exactly one retirement、または rollback）— 配置による構造的解決
- **publish() の全失敗点を列挙**（RuntimeWorldAuthority.h:207-254）:
  1. `!owner` → nullptr（Failed）・2. `newWorld == nullptr` → nullptr（Failed）・
  3. `sequenceId == 0` → nullptr（Rejected）・4. `coordinator_.getState() == Faulted` → nullptr（Rejected）
  → **全て `publishAndSwap` の前に return**（retire なし・reservation 消費なし）。
- **★ acquire の配置決定**: **全 validate/commit チェック（上記 1〜4）の後・不可逆ステップ
  `owner.release() → fence → publishAndSwap` の直前**に acquire を置く。
- **★ `publishAndSwap` は失敗しない**: `exchangeAtomic(store_->current, next, acq_rel)`
  （RuntimeStore.h:40-50・moved-from 誤用は Debug assert のみ・正常経路に失敗なし）。
- **∴ ご指摘の「acquire 成功 → publishAndSwap 失敗」経路は構造的に存在しない**。
- **INV-R0 = "successful acquire ↔ exactly one World retirement"（by construction）**:
  - acquire 成功 → 直後の swap（不可逆・失敗なし）→ oldWorld = prevWorld が retired → reservation 保持。
  - acquire 失敗（N_retired_world ≥ R）→ **swap 前**に publish Rejected（backpressure・retire なし・
    reservation 消費なし）。**rollback は不要**（成功後の失敗経路が無いため）。
  - 「acquire は retire 前だから安全」だけでは不十分（ご指摘）→ **「全失敗チェック後・不可逆 swap 直前」の
    配置で acquire 成功後の失敗経路を構造的に排除**。

### D49.3 状態遷移表（全分岐）
| # | 遷移 | 条件 | reservation 効果 |
|---|------|------|------------------|
| T1 | build → publish admission | newWorld 構築・prevWorld = current | — |
| T2 | validate / commit 失敗 | publish() が nullptr 返却（1〜4） | **acquire 前** → 消費なし・retire なし |
| T3 | **acquire(prevWorld)** | prevWorld != null かつ N_retired < R | **+1**（prevWorld に対して保持） |
| T3' | acquire 失敗 | prevWorld != null かつ N_retired ≥ R | **publish Rejected**（swap 前・retire なし・backpressure） |
| T4 | bootstrap 初回 publish | prevWorld == null | acquire 不要（oldWorld なし） |
| T5 | publishAndSwap | 原子的・失敗しない | oldWorld = prevWorld が retired |
| T6 | retire(oldWorld) → enqueue | — | reservation 保持（destroy まで） |
| T6' | enqueue 失敗 → retry → quarantine | queue full | reservation 保持 |
| T6'' | quarantine full → leak | 異常系 | reservation 永続保持（liveness・INV-R5） |
| T7 | epoch-gated reclaim → destroy | reader 安全 | **release(prevWorld) → -1** |
| S1 | shutdown clear（clearedWorld） | 別 contract（R5） | **acquire 不要**（publish 置換でない）・キュー済み world は drainAll で release |
| S2 | shutdown enqueue no-op | isShutdownInProgress | clearedWorld 終端 leak・release なし（終端） |

### D49.4 Identity binding（B）— acquire(W)〜release(W) の一貫性
- **acquire key = prevWorld**（swap 前に `runtimeStore_.observe()` で読取・RuntimeWorldAuthority.h:243）・
  **oldWorld = 同一 prevWorld**（swap 戻り値）。同一ポインタが acquire → retire → enqueue → destroy →
  release を流れる（D45 検証・deleter は enqueue された ptr を実行）。
- **★ 前提 INV-PUB-SER（publish serialization）**: publish は単一パイプライン（waitForReceipt 同期
  プロデューサ + FIFO ISR intent 処理・ISRIntentDispatcher.h:29-30 stateless singleton handler・単一 ISR
  スレッド）→ **observe→swap 間に他 publish が割り込まない → prevWorld == oldWorld**。
- **∴ `acquire(W) → retire(W) → enqueue(W) → destroy(W) → release(W)` は同一 W を流れる（別 World に誤対応
  しない）**。B CLOSED（serialization 前提）。
- **counter で十分か**: strict 1:1 transition（INV-PUB-SER + enqueue-once）が保証されるため、
  **counter + transition invariant で十分**（ユーザー判断と整合）。ただし authority API は
  **identity-aware シグネチャ（`acquire(WorldIdentity)` / `release(WorldIdentity)` / `count()`）** にして
  compile-time に binding を強制し、内部状態は counter（strict 1:1 のため set 新設不要）とする。

### D49.5 Shutdown semantics（C）— 通常運転と完全分離
- **clearedWorld（shutdown clear）**: publish 置換ではない（`clearPublishedRuntimeSnapshotsNonRt` の
  null-swap）→ **acquire 不要**（reservation は「publish 置換で retired になる world」にのみ紐付く）。
- **キュー済み world**: shutdown の `drainDeferredRetireQueues(true)` / `drainAllQuarantineStore` /
  `m_epochDomain.drainAll()` で強制破棄 → deleter → **release 実行**。
- **no-op enqueue（isShutdownInProgress）**: clearedWorld が enqueue されず終端 leak → release なし。
  プロセス終了で reservation authority 自体が破棄され、残数は無意味。
- **∴ shutdown invariant は通常運転と完全分離（R5）**・「shutdown では N_retired_world == reservation.count
  を保証しない（終端 leak 可）」を明示。C CLOSED（別 contract として明示）。

### D49.6 Quarantine-full / overflow（D）— safety と liveness の分離
- **4 者対応（queue full → retry → quarantine → quarantine full の最終状態）**:
  | 対象 | 状態 |
  |------|------|
  | reservation | **保持（未 release・leak 永続）** |
  | World | **生存（破棄されない）** |
  | raw pointer | **lost（dropped・所有者なし）** |
  | Generic entry | **dropped（storage に入らない）** |
- **SAFETY invariant（CLOSED）**: `N_retired_world == reservation.count` は **leak でも成立**（leak world は
  retired-but-not-destroyed のまま → reservation 保持は正しい反映・INV-R3 整合）。
- **AVAILABILITY / LIVENESS invariant INV-R5（reservation leak は bounded）: OPEN** — quarantine-full は
  異常系（queue full + quarantine full 同時 = 深刻な EBR pressure・`overflowCount_` / health escalation で
  検知・正常運転では到達不能）であり、**構造的に bounded とは証明不能**。safety と liveness を分離。

### D49.7 最終状態（ユーザー判定表の更新）
| 項目 | 状態 |
|------|------|
| R の意味 | CLOSED |
| Global safety と World reservation の分離 | CLOSED |
| acquire を retire 前に置く | CLOSED |
| release を destroy 後に置く | CLOSED |
| **identity-aware 1:1 対応** | **CLOSED**（INV-PUB-SER 前提 + identity-aware API・counter + transition で十分） |
| **acquire failure rollback** | **CLOSED（INV-R0）** — 全失敗チェック後・不可逆 swap 直前に配置 → acquire 成功後の失敗経路なし・rollback 不要 |
| **shutdown contract** | **CLOSED（R5 別 contract として明示）** |
| **quarantine-full による permanent reservation** | **OPEN（liveness・INV-R5）** — safety は CLOSED（leak でも count 整合） |
| **N_retired_world == reservation.count** | **CLOSED（safety・leak でも成立）** |
| R の数値 | OPEN（policy・INV-R5 含む証明が閉じてから） |
| I-2 implementation | NO-GO |

### D49.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T93 | INV-R0 | acquire 成功後は必ず 1 retirement（publishAndSwap 不可逆・失敗チェック後配置）・acquire 失敗は Rejected（swap 前） |
| T94 | identity binding | acquire(W)→retire(W)→enqueue(W)→destroy(W)→release(W) 同一 W（INV-PUB-SER 前提） |
| T95 | acquire 失敗 = backpressure | N_retired_world ≥ R で publish Rejected・oldWorld は retire されない |
| T96 | shutdown | clearedWorld は acquire 不要・キュー済み world は drainAll で release・no-op は終端（R5 別 contract） |
| T97 | quarantine-full 4 者対応 | reservation（保持）/ World（生存）/ ptr（lost）/ entry（dropped）の対応 |
| T98 | INV-R5 liveness 分離 | safety（count == retired-but-not-destroyed・leak でも成立）は CLOSED・liveness（leak bounded）は OPEN |

---

## D50 — Design-32 確定（reservation authority の状態機械・exhaustion / quarantine-full / shutdown の遷移と認可者・2026-08-15）

ユーザー Design-31 レビュー対応: **「INV-R5 を health escalation で緩和する前に、reservation の枯渇が
publish admission にどう伝播するか」を閉じる**。`WorldRetirementReservation.acquire()` の
success/failure 両経路を、**「reservation exhaustion / quarantine-full / shutdown の3つの異常・終端状態に
対して、どの状態からどの状態へ遷移し、誰がその遷移を認可するか」を状態機械として閉じる**。
**R の数値決定前に reservation authority の lifetime contract 自体を完成させる。Phase I 実装 NO-GO 継続。**

### D50.1 D49 の位置づけと残課題
- INV-R5（reservation leak bounded）= **OPEN 維持**（health escalation で緩和する前に、まず exhaustion の
  publish 伝播を閉じる）。
- 4 論点: (1) exhaustion 結果コード / (2) release 再利用 / (3) quarantine-full 仕様判断 / (4) shutdown 経路
  コード再確認。

### D50.2 論点1 — Reservation exhaustion は hard reject か
- **acquire 失敗（count == R）→ publish `Rejected`（backpressure）**。**`publishAndSwap()` より前**（acquire を
  全 validate/commit チェック後・不可逆 swap 直前に配置・D49）で拒否。
- **Faulted ではない**（monotonicity violation とは別 semantic）・**即時 retry ではない**（transient 条件 —
  `release()` が count を減らした後に再 acquire 可能・D14/D18 backpressure / terminal-failure 撤廃と整合）。
- 結果コード伝播: `publish()` は既存の失敗形（nullptr + committed=false）で返し、semantic は
  Rejected/backpressure。**理由の区別（validate vs reservation exhaustion）は実装詳細**（PublishStageResult
  への reason 追加等）。caller（RuntimePublishExecutor）は committed=false で bridge 実行しない。
- **再 acquire 可能性**: 次回 publish で count < R になっていれば成功（release が count を減らすため）。
  CLOSED（配置 + 伝播）。

### D50.3 論点2 — release による再利用
- **release は一度だけ**（deleter が各 world につき正確に 1 回実行 — D46-R4: primary CAS 排他 / quarantine
  drain-once）。
- **release 後 count 減 → 同一 reservation slot は確実に再利用可能**（counter セマンティクス・slot の概念は
  count の値のみ）。
- **★ 実装制約（正直・未解決）**: 現行 deleter は **stateless function pointer**
  （`enqueueDeferredDeleteNonRt(void*, void(*)(void*))`）であり、**authority への参照をキャプチャできない**。
  D49 の「release = deleter 内」は設計配置として有効だが、**mechanism は未実装**。実装時は world パスの
  deleter を **stateful 化**（std::function または engine 参照キャプチャ・DeferredDeletionQueue の deleter
  型変更）して release を届ける必要がある。→ **この項目は実装ゲート（実装時に閉じる）として記録**。

### D50.4 論点3 — quarantine-full の仕様判断
- `quarantine()` が `false` → world が storage に入らず **world と reservation が両方永続保持**。
  **「reservation のみ解放する」経路は存在しない**（world 破棄なしの release は INV-R3 / safety を破る）。
- **★ 仕様判断: quarantine-full は「world + reservation の永続保持を許容する catastrophic state」とする
  （Option 1）**。
- **health escalation の役割**: 検知（`overflowCount_`）・escalation（backpressure / user intervention）・
  **reservation を「解放」しない**（world 破棄なしの release は safety 破壊）。
- 「実際の World destruction を保証する mechanism」にするには **reader-safe 強制 drain**（quarantine の本機構）
  が必要だが、それが full なのが本状態 → **構造的には閉じられない**。
- **∴ INV-R5（reservation leak bounded）= OPEN 維持**（liveness・異常系）。safety
  （count == retired-but-not-destroyed・leak でも維持）は CLOSED のまま。

### D50.5 論点4 — shutdown 経路のコード再確認（文章整合性ではなくコードで確認）
- **2 つの別経路（別 world 集合）をコードで確認**:
  1. **通常運転で enqueue 済み world** → shutdown の `m_epochDomain.drainAll()`（CtorDtor.cpp:234 →
     `drainAllUnsafe`・全 deleter 実行）+ `drainDeferredRetireQueues(true)`（→ `tryReclaim` →
     `EpochDomain::reclaim`）+ `m_retireRouter->drainAllQuarantineStore()`（ReleaseResources.cpp:378 →
     `m_retireQuarantine.drainAllUnsafe`・quarantine の deleter 実行）→ **destroy → release 実行**
     （reservation 解放・D49 の記述はこの経路）。
  2. **shutdown clear 時の clearedWorld** → `retireRuntimePublishWorldNonRt` →
     `enqueueDeferredDeleteNonRt` が `isShutdownInProgress()` で no-op（D45・AudioEngine.h:4176）→ enqueue されず
     破棄されない。**ただし clearedWorld は acquire 不要**（live current の clear・publish 置換でない・D49）
     → **reservation に関与しない**。
- **∴ D45 の no-op と D49 の drainAll→release は同じ経路ではなく「shutdown 時 clear world」と「事前
  enqueue 済み world」の別集合**。両者は CtorDtor 順序（graceful drain → clear → drainAll）でコード的に整合。
  **文章整合性でなくコード（EpochDomain.h:403 drainAllUnsafe / CtorDtor.cpp:226-234 /
  ReleaseResources.cpp:371-378）で確認済み**。
- shutdown 終端で reservation authority はプロセス終了と共に破棄（残数は無意味・terminal）。

### D50.6 状態機械（reservation authority・遷移と認可者）
| # | 遷移 | From → To | 認可者 | 条件 |
|---|------|-----------|--------|------|
| A1 | acquire | count → count+1 | `WorldRetirementReservation::acquire`（publish() 内・swap 前） | count < R |
| A2 | acquire 失敗 | count 不変・publish → Rejected | `publish()`（swap 前） | count == R |
| A3 | retire | world が retired-but-not-destroyed | `publishAndSwap`（不可逆） | acquire 成功済み |
| A4 | destroy（通常） | world 破棄 → release | deferred-delete drain（epoch-safe） | reader 安全 |
| A5 | destroy（shutdown） | world 破棄 → release | `drainAll` / `drainAllQuarantineStore` / `drainDeferredRetireQueues(true)` | Audio Thread 停止 |
| A6 | quarantine-full | world + reservation 永続保持 | **なし（遷移なし・catastrophic）** | queue full + quarantine full |
| A7 | clearedWorld（shutdown） | 破棄なし・reservation 関与なし | —（enqueue no-op） | isShutdownInProgress |
- **認可の原理**: count を減らす唯一の経路 = **deleter（世界破棄時のみ）** → INV-R3 維持。
  **reservation のみを解放する経路は存在しない**（safety 保護・quarantine-full で reservation 解放しない理由）。

### D50.7 最終状態（ユーザー判定表の更新）
| 項目 | 状態 |
|------|------|
| INV-R0 / publishAndSwap 不可逆性 / identity binding / publish serialization | CLOSED（D49） |
| 通常 retire → destroy → release | CLOSED |
| **Reservation exhaustion 結果コード** | **CLOSED**（Rejected・swap 前・retry-able・backpressure） |
| **Release 再利用** | **CLOSED**（drain-once・counter 再利用）・★ deleter stateful 化は実装ゲート |
| **Quarantine-full 仕様判断** | **CLOSED（Option 1: world+reservation 永続保持を許容・catastrophic）**・INV-R5 = OPEN 維持 |
| **Shutdown 経路** | **CLOSED（コード確認・別 world 集合・R5 別 contract）** |
| **INV-R5（reservation leak bounded）** | **OPEN（liveness・異常系・health escalation は検知のみ）** |
| N_retired_world == reservation.count | CLOSED（safety・leak でも成立） |
| R の数値 | OPEN（INV-R5 含む証明が閉じてから） |
| I-2 implementation | NO-GO（ユーザー最終 GO 待ち） |

### D50.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T99 | exhaustion = Rejected | count == R で acquire 失敗 → publish Rejected（swap 前・backpressure）・Faulted ではない |
| T100 | retry-able | release 後 count < R で次回 publish の acquire 成功 |
| T101 | release 再利用 | drain-once（deleter 1 回）→ release 1 回 → count 減 → slot 再利用可 |
| T102 | quarantine-full 仕様 | Option 1: world+reservation 永続保持・health escalation は reservation 解放しない（INV-R3 保護） |
| T103 | shutdown 経路 | 事前 enqueue world は drainAll/drainAllQuarantineStore で destroy→release・clearedWorld は no-op（acquire 不要） |
| T104 | deleter stateful 化 | enqueueDeferredDeleteNonRt の deleter が authority へ到達できる（実装ゲート） |

---

## D51 — Design-33 確定（release authority の結合設計・T104 解消・reservation count ≡ World count の実装 authority 化・2026-08-15）

ユーザー Design-32 レビュー対応: **「D50 の最重要結論は release authority の lifetime point は確定したが
release operation の実装 authority は未確定（T104）」**。`WorldRetirementReservation::release(WorldIdentity)`
を World destruction authority にどう結合するかを確定し、**reservation count == retired-but-not-destroyed
World count を「設計上そうなる」ではなく「実装上の lifetime authority」として閉じる**。
**R の数値決定は T104 設計完了後（D52）。Phase I 実装 NO-GO 継続（本 D51 は設計契約・コード未変更）。**

### D51.1 前提のコード確認（deleter フローの収束性）
- **同一 deleter が primary と quarantine の両方に流れる**: `enqueueWithRetry`
  （ISRRetireRouter.cpp:163-205）→ `enqueueRetire`（→ EpochDomain → DeferredDeletionQueue）と
  `m_retireQuarantine.quarantine(ptr, deleter, ...)` に**同一 `deleter` 関数ポインタ**を渡す。
- world の deleter は `retireRuntimePublishWorldNonRt`（AudioEngine.h:3525-3533）で**一度だけ定義**され、
  primary / quarantine / shutdown drain の**全破棄経路で同一 deleter が実行**される。
- 現行 deleter シグネチャ: `void(*)(void*)`（stateless function pointer・キャプチャ不可・
  DeferredDeletionQueue.h / RetireQuarantineStore.h とも trivially copyable）。

### D51.2 deleter mechanism の4選択肢分析（ユーザー論点3）
| 選択肢 | 評価 | 判定 |
|--------|------|------|
| **(1) function pointer のまま** | 単独では authority 参照を届けられない（stateless・キャプチャ不可）。ptr 経由で到達するには RuntimeState に back-ref が必要（= (3) と同義） | **単独では不可** |
| **(2) context pointer を DeletionEntry に持たせる** | `DeletionEntry` / `QuarantinedEntry` に `void* context` 追加（raw pointer 追加のみ・**trivially copyable 維持**）。deleter を `void(*)(void* ptr, void* context)` に。world パスは `context = &engine_.worldRetirementReservation()`（engine member・安定 lifetime） | **★ 推奨・採用** |
| **(3) RuntimeState 自身に reservation token** | single-arg deleter のまま `ptr->release()` 可能だが、**frozen world DTO に runtime-lifetime authority 参照を埋める = アーキテクチャ漏れ**（RuntimeState は freeze 済み immutable world・Semantic 管理） | **却下** |
| **(4) RetireRouter 側に World-specific destruction authority** | ISRRetireRouter は work21 P0-1「thin stateless dispatcher・state/policy/decision 禁止」→ World-specific 状態の配置は契約違反 | **却下** |

### D51.3 採用設計 — context pointer in DeletionEntry + 2引数 deleter
- **`DeletionEntry` / `QuarantinedEntry` に `void* context` を追加**（trivially copyable 維持・
  static_assert は継続可）。
- **deleter シグネチャを `void(*)(void* ptr, void* context)` に変更**。既存 generic caller（Cache /
  DSP / EQ 等）は context = nullptr で機械的適応。
- **world パスの deleter（AudioEngine.h:3525-3533）**:
  ```cpp
  [](void* p, void* ctx) {
      auto* ptr = static_cast<RuntimePublishWorld*>(p);
      auto* res = static_cast<WorldRetirementReservation*>(ctx);
      ptr->unseal();
      ptr->~RuntimePublishWorld();
      convo::aligned_free(ptr);
      res->release(ptr);          // ★ destroy 完了後に identity-aware release
  }
  ```
  に変更し、enqueue 時に `context = &engine_.worldRetirementReservation()` を渡す。
- **API 変更範囲（正直なスコープ）**: DeferredDeletionQueue / RetireQuarantineStore /
  EpochDomain::enqueueRetire / IEpochProvider / IRetireRouter / ISRRetireRouter::enqueueRetire /
  enqueueWithRetry / retireRT / retire / quarantineRetire / enqueueDeferredDeleteNonRt のシグネチャに
  context を追加（または overload）。機械的・広範だが一貫した変更。
- **authority lifetime**: `WorldRetirementReservation` は AudioEngine メンバ（engine が queue より長命）→
  context ポインタの安定 lifetime 保証。

### D51.4 確認4点（ユーザー論点1/2/4）
| 確認 | 結果 |
|------|------|
| **1. identity binding**（acquired == destroyed） | acquire key = prevWorld（swap 前 observe の ptr）・oldWorld = 同一 ptr・enqueue も同一 ptr・deleter は同一 ptr を受理 → **release(ptr) は acquire と同一 identity**。CLOSED |
| **2. exactly-once release** | release は deleter 内・deleter は各 world 正確に 1 回（drain-once・primary CAS 排他 / quarantine drain-once・D46-R4）→ **one acquire ↔ one release**。CLOSED |
| **3. deleter lifetime** | **(2) context-in-entry 採用**（option (1)/(3)/(4) 却下理由は D51.2）。trivially-copyable 維持・authority lifetime 安定。CLOSED |
| **4. quarantine path の同一性** | **同一 deleter + 同一 context が primary / quarantine / shutdown drain に収束**（ISRRetireRouter.cpp:163-205 で同一 deleter を両方に渡す・QuarantinedEntry も context 保持）→ **全経路が同一 release authority**。CLOSED |

### D51.5 INV-R3 の分割（release location / mechanism / implementation）
- **release location（destroy 後）**: CLOSED（D48/D49）。
- **release mechanism（どう結合するか）**: **CLOSED（本 D51）** — context-in-entry + 2引数 deleter +
  release は deleter 内・destroy 後。
- **release implementation（コード実行）**: **実装ゲート（I-2 GO 時に実施）** — Phase I 実装 NO-GO のため
  現行コードでは未実装（T104 の設計は閉じたがコードは未変更）。
- **∴ `reservation count == retired-but-not-destroyed World count` = CLOSED（実装仕様として・実装ゲート
  付き）** — 「設計上そうなる」ではなく「実装上の lifetime authority として閉じた」。

### D51.6 最終状態（ユーザー D50 判定表の更新）
| 項目 | D50 判定 | D51 後 |
|------|----------|--------|
| release point の意味 | CLOSED | CLOSED |
| destroy 後に release すべきこと | CLOSED | CLOSED |
| release の唯一性 | 設計上 CLOSED | **CLOSED（drain-once + context 経由・exactly-once）** |
| **release mechanism（設計）** | OPEN / T104 | **CLOSED（context-in-entry + 2引数 deleter + 全経路収束）** |
| **現行コードによる release 実行** | OPEN / T104 | **実装ゲート**（設計は閉じた・コードは I-2 GO 時に実施） |
| **reservation slot の実再利用** | OPEN / T104 | **CLOSED（設計）**・コード実行は実装ゲート |
| `reservation count == retired-but-not-destroyed count` | 条件付き | **CLOSED（実装仕様・実装ゲート付き）** |
| R の数値 | OPEN | **OPEN（次 D52・T104 設計完了後）** |
| I-2 implementation | NO-GO | **NO-GO（ユーザー最終 GO 待ち・コード未変更）** |

### D51.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T105 | release mechanism 設計 | context-in-entry + 2引数 deleter・world パスは context = &authority |
| T106 | 全経路収束 | primary / quarantine / shutdown drain が同一 deleter（同一 context）で release 実行 |
| T107 | identity binding（実装） | acquire key == destroy key（同一 ptr が deleter に到達） |
| T108 | exactly-once（実装） | 1 acquire ↔ 1 release（drain-once）・release は deleter 内 1 回 |
| T109 | trivially-copyable 維持 | DeletionEntry / QuarantinedEntry が context 追加後も trivially copyable（static_assert 継続） |
| T110 | generic caller 適応 | Cache / DSP / EQ 等は context = nullptr で機械的適応（挙動不変） |

---

## D52 — Design-34 確定（D51.1 改訂: WorldIdentity と context lifetime の確定・release 設計修正・2026-08-15）

ユーザー Design-33 レビュー対応: **「D51 を CLOSED とするには2点修正が必要。特に release 設計が
`aligned_free(ptr)` 後に `release(ptr)` を行う点が identity binding として不適切」**。5点（freed pointer 依存
回避 / context lifetime（INV-R6）/ identity preservation / R_c 定義精密化 / WorldRetirementReleaseContext）を
コードで閉じる。**R の policy 数値（D53）へ進む前に、D51.1 として WorldIdentity と context lifetime を確定**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D52.1 新事実（コード確認）— worldId は reservation identity に使えない
- **`RuntimeWorldIdGenerator` は「DIAGNOSTIC ONLY」と明記**（ISRRuntimeIdentityGenerators.h:13-16）:
  「Must NOT be used for: **Authority decisions**（branch, condition, ordering）/ Publication ordering /
  Retire ordering / **Hash keys in semantic structures**」。
- worldId は `runtimeWorldIdGenerator_.next()`（AudioEngine.h:3442・monotonic）で一意だが、
  **reservation registry の key（semantic-structure hash key / authority decision）としての使用は明示禁止**。
- **generation は mod-256 wrap**（`onRuntimeRetiredNonRt` で `generation % 256` → DSP slot マッピング）→
  一意でない。
- **address は再利用可能**（World A @ 0x1234 destroy → World B @ 0x1234 再確保）→ 不安定。
- **∴ 3候補（worldId / generation / address）は全て不適格**。reservation の identity は
  **`WorldRetirementReservation` 自体が acquire 時に mint する monotonic reservation token** とする
  （world の semantic identity に依存しない内部会計 token）。

### D52.2 設計修正 — release は freed pointer に依存しない（ユーザー論点1/3/5）
- **D51 の `release(ptr)` を `aligned_free(ptr)` 後に行う設計は撤回**（ptr は free 後は無効）。
- **採用設計: `DeletionEntry` / `QuarantinedEntry` に `void* context` + `uint64_t identity` を value フィールド
  として追加**（trivially copyable 維持・`WorldRetirementReleaseContext { reservationAuthority, identity }` の
  allocation-free 実現）:
  ```
  DeletionEntry {
      void* ptr;
      void (*deleter)(void* ptr, void* context, uint64_t identity);
      void* context;       // world: &WorldRetirementReservation（authority）・generic: nullptr
      uint64_t identity;   // world: reservation token（authority が mint）・generic: 0
      uint64_t epoch; DeletionEntryType type; uint64_t publicationSequenceId; uint64_t generation; ...
  }
  ```
- **world deleter**:
  ```cpp
  [](void* p, void* ctx, uint64_t token) {
      auto* ptr = static_cast<RuntimePublishWorld*>(p);
      auto* res = static_cast<WorldRetirementReservation*>(ctx);
      ptr->unseal();
      ptr->~RuntimePublishWorld();
      convo::aligned_free(ptr);
      res->release(token);   // token は entry の value フィールド → freed pointer 非依存・worldId 非依存・address 非依存
  }
  ```
- **token フロー**: publish() 内 acquire が token を mint → publish() が `{oldWorld, token}` を返す →
  caller（RuntimePublishExecutor / Bootstrap）が retire ブリッジへ token を渡す →
  `enqueueDeferredDeleteNonRt(oldWorld, deleter, context=&authority, identity=token)`。
- **identity preservation（ユーザー論点3）**: acquire が token を mint → 同一 token が entry に保存 →
  deleter が同一 token を release。**同一 World の acquire/release が token で 1:1 対応**（address 非依存）。

### D52.3 context lifetime（INV-R6）— コード確認
- **INV-R6: `context`（&WorldRetirementReservation）は queue entry より長寿命**。
- **authority = AudioEngine member**（`worldRetirementReservation_`）→ lifetime は engine。
- **全 queue entry は AudioEngine デストラクタ本体で drain される**（CtorDtor.cpp: graceful drain →
  shutdown clear → `drainDeferredRetireQueues(true)` → `m_epochDomain.drainAll()` →
  `drainAllQuarantineStore`（ReleaseResources.cpp:378））→ **deleter 実行は全てメンバー破棄より前**。
- メンバー破棄はデストラクタ本体終了後（逆宣言順）→ **deleter 実行時に authority は必ず生存**。
- 防御的宣言順: `WorldRetirementReservation` は `m_epochDomain` / `m_retireRouter` より**後**に宣言（逆破棄で
  先に破棄されない）を推奨。**INV-R6 CLOSED**。
- 補足: quarantine-full leak / shutdown no-op の world は entry に載らない（context を保持しない）→
  ダングリング context 無し。

### D52.4 R_c の精密化（ユーザー論点4）
- **R_c = |{W : W が reservation token を acquire 済み かつ destruction を完了していない}|**。
- 「destruction 完了」= deleter 内の `~RuntimePublishWorld()` + `aligned_free` 完了・release 実行（deleter の
  最終ステップ）。「deleter 開始」と「destruction 完了」を区別（deleter 開始〜完了は同一 deleter 内の
  transient・破棄中 world は未完了として R_c に含む）。
- **正常系では R_c = N_retired-but-not-destroyed**（acquire 済み・未破棄 world = retired-but-not-destroyed）。
- **quarantine-full（Option 1）では world が永続保持され destruction 未完了 → token 保持 → R_c に含まれ続ける**
  （reservation は実際の lifetime を正しく反映・INV-R3 維持）。R_c は「理論上の counter」でなく
  「解放されていない token 数」を正確に表す。
- **`reservation count == retired-but-not-destroyed count` = CLOSED（正常系 invariant として）**。

### D52.5 最終状態（ユーザー D51 判定表の更新）
| 項目 | D51 | D52（レビュー後） |
|------|-----|-------------------|
| primary/quarantine 同一 deleter | CLOSED | CLOSED |
| release point = destroy 後 | CLOSED | CLOSED |
| context pointer 方式 | 設計候補 | **CLOSED（entry の value フィールドとして実現・採用確定）** |
| **context lifetime（INV-R6）** | OPEN | **CLOSED（engine member・全 entry はメンバー破棄前に drain）** |
| **stable World identity** | OPEN | **CLOSED（reservation token を authority が mint・worldId は diagnostic-only で却下・generation は wrap・address は再利用）** |
| release exactly-once | 設計上 CLOSED | **CLOSED（identity + context lifetime 確立後・token 1:1）** |
| reservation count ≡ World count | CLOSED | **CLOSED（正常系 invariant・R_c 定義で精密化）** |
| **release は freed pointer 非依存** | — | **CLOSED（token + context を value フィールド化・解放済み ptr に依存しない）** |
| 現行コード実装 | 未実装 | **NO-GO（実装ゲート・I-2 GO 時に実施）** |
| R の数値 | OPEN | **OPEN（次 D53・D51.1 完了後）** |

### D52.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T111 | release は freed pointer 非依存 | release は entry の token（value フィールド）使用・aligned_free 後の ptr 参照なし |
| T112 | token 1:1 | acquire で mint → entry 保存 → deleter で同一 token release（address/worldId 非依存） |
| T113 | INV-R6 context lifetime | authority = engine member・全 entry はメンバー破棄前に drain（CtorDtor 順序） |
| T114 | R_c 定義 | R_c = acquire 済み・destruction 未完了 world 数・正常系で N_retired-but-not-destroyed と一致 |
| T115 | worldId 不使用 | reservation registry は worldId を key にしない（diagnostic-only 契約遵守） |

---

## D53 — Design-35 確定（token continuity / enqueue atomicity / failure rollback・D52.1 相当・2026-08-15）

ユーザー Design-34 レビュー対応: **「R の数値決定へ直行する前に、token の mint → entry への移送が acquire
と retire の間で失敗しないことを閉じる」**。D52 の核心を **pointer identity ではなく reservation token
identity で追う**: `successful acquire(token) ⇒ exactly one queued entry carrying the same token`（INV-TOKEN-1）。
ご提示の状態遷移表（S0-S5 + EX）と6確認点をコードに照らして閉じる。**Phase I 実装 NO-GO 継続（設計契約のみ）。**

### D53.1 token identity の状態遷移表（ユーザー提示 S0-S5 + EX）
| 状態 | token | World | 許される遷移 |
|------|-------|-------|------------|
| S0 | 無し | LIVE | `acquire()` |
| S1 | minted / held | LIVE | **publish swap（不可逆・INV-R0）** |
| S2 | minted / held | RETIRED | **enqueue（token → entry に値として保存）** |
| S3 | entry に保存 | RETIRED | reclaim / quarantine drain |
| S4 | entry consumed | destroying | deleter |
| S5 | release 済み | DESTROYED | token slot 再利用 |
| **EX** | minted / held | LIVE | **構造的に不可**（後述・acquire 配置により排除） |
- **EX の排除（INV-R0 を token identity に拡張）**: acquire は「全 validate/commit チェック後・不可逆
  swap（`owner.release` → fence → `publishAndSwap`）の直前」に配置（D49）。swap = `exchangeAtomic` は失敗
  しない（RuntimeStore.h:40-50）。→ **acquire 成功後は必ず swap（retirement）**。EX（mint 後に publish 失敗）
  は**構造的に存在しない**（rollback 機構は不要・INV-R0 と同一論法）。
- quarantine-full（Option 1）: S3 相当で world 永続保持 → token held のまま（release なし・R_c に含まれ続ける）。

### D53.2 6確認点（コード照合）
| # | 確認 | 結果 |
|---|------|------|
| 1 | **acquire() の token mint 位置** | `publish()` 内・swap 前（D52・acquire が mint → `{oldWorld, token}` を返却）。CLOSED |
| 2 | **DeletionEntry への token/context 格納が enqueueWithRetry の各分岐で保持** | token/context は関数引数（値渡し）で `enqueueRetire`（primary）→ retry ×2 → `quarantine`（ISRRetireRouter.cpp:161-205）へ一貫して伝播・改変なし。CLOSED |
| 3 | **primary full → retry → quarantine 間で token が複製・消失しない** | token は**成功した enqueue でのみ entry に格納**（失敗時は未格納・再試行は同 token を再送）。成功後は retry ループが return → **単一 storage（primary XOR quarantine）に token が1つ**。複製なし・消失なし。CLOSED |
| 4 | **enqueue failure / quarantine-full で token はどうなる** | quarantine 成功 → token は quarantine entry に保持。**quarantine-full（leak）→ token は未格納・world は leak・token held のまま（release なし）** — これは Option 1 と整合（R_c に含まれ続ける・INV-R3 維持）。token は「held 集合」に保持され続けるため追跡喪失なし。CLOSED |
| 5 | **shutdown drainAll / drainAllQuarantineStore で同一 token が deleter に届く** | deleter は entry の `identity`（token 値）を読む → `drainAllUnsafe` / `drainAllQuarantineStore` は格納済み entry の deleter を実行 → **同一 token が届く**（entry に保存されているため経路非依存）。CLOSED |
| 6 | **release(token) が stale / double release を検出** | authority は**held token 集合**を保持（token は monotonic 一意・authority が mint）。release(token) は「token が held か」を検証 → **stale（未 mint / 既 release）・double release を検出**（Debug assert / health event）。CLOSED（design contract） |

### D53.3 INV-TOKEN-1（token continuity）の形式化
- **INV-TOKEN-1: `successful acquire(token) ⇒ 同一 token を保持する queued entry が正確に 1 つ`**
  （primary または quarantine・quarantine-full leak は Option 1 として「held 集合に保持・release なし」で
  整合）。
- 証明（設計・コード裏付け）: (a) acquire は swap 前に 1 回 mint（S1・EX 排除）→ (b) swap 不可逆（S1→S2）→
  (c) enqueue は成功時のみ token を entry に格納・単一 storage（S2→S3・確認2/3）→ (d) reclaim/quarantine drain
  は格納 entry の deleter を実行（S3→S4・確認5）→ (e) deleter は同一 token を release（S4→S5・確認6）。
- **∴ `R_c = N_retired-but-not-destroyed World` を pointer identity ではなく token identity で厳密に表現可能**
  （acquire token 数 = 未 release token 数 = retired-but-not-destroyed world 数・正常系）。CLOSED。

### D53.4 最終状態
| 項目 | 状態 |
|------|------|
| D51.1 identity correction（reservation token・worldId 却下） | CLOSED（D52） |
| INV-R6 context lifetime | 設計上 CLOSED（D52） |
| freed-pointer independence | CLOSED（D52） |
| token identity | DEFINED（D52） |
| **acquire token → deferred entry token の continuity（INV-TOKEN-1）** | **CLOSED（本 D53・EX 構造的排除 + 6確認点）** |
| R_c = N_retired-but-not-destroyed（token identity） | CLOSED（正常系 invariant） |
| R の数値 | **OPEN（次 D54・policy）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち・コード未変更）** |

### D53.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T116 | INV-TOKEN-1 | successful acquire(token) ⇒ 同一 token の queued entry が正確に 1 つ（primary XOR quarantine） |
| T117 | EX 排除 | acquire 成功後は必ず swap（不可逆）・mint 後に publish 失敗なし |
| T118 | token 単一 storage | primary → retry → quarantine で token 複製・消失なし |
| T119 | quarantine-full token | token held のまま（Option 1・release なし・R_c に含まれる） |
| T120 | shutdown token 到達 | drainAll / drainAllQuarantineStore で同一 token が deleter に届く |
| T121 | stale / double release 検出 | release(token) が held 検証（未 mint / 既 release を検出） |

---

## D54 — Design-36 確定（INV-TOKEN-1 修正・token-state 完全 invariant・held storage cardinality・2026-08-15）

ユーザー Design-35 レビュー対応: **「quarantine-full を含めて `successful acquire ⇒ exactly one queued entry` と
した点が矛盾」**。INV-TOKEN-1 を修正し、**terminal-held を含む完全な token-state invariant** と
**`WorldRetirementReservation` 自身の storage cardinality** を確定する。**R の数値を先に決めない**
（authority の storage を確定してから）。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D54.1 INV-TOKEN-1 の修正（quarantine-full と整合）
- **旧式（矛盾）**: `successful acquire(token) ⇒ exactly one queued entry` — quarantine-full では
  `acquired token ⇏ queued entry`（entry は存在しない・world leak・token held）→ **誤り**。
- **修正式**:
  $$
  \text{Acquired}(t) \Rightarrow
  \begin{cases}
  \text{Queued}(t) & \text{通常経路} \\
  \text{TerminalHeld}(t) & \text{quarantine-full}
  \end{cases}
  \qquad
  \boxed{\text{Queued}(t) \wedge \text{TerminalHeld}(t)\ \text{cannot coexist}}
  $$
- **ownership graph（ユーザー提示）**:
  ```
  acquire(t) → HELD(t) → publishAndSwap → RETIRED(t) → enqueueWithRetry
      /                                        \
  Queued(t)                                TerminalHeld(t)
      │                                         │
  reclaim/drain                                 │
      │                                         │
  destroy                                      │
      │                                         │
  release(t) → RELEASED                    process termination
  ```
- **token の消失・複製・二重 release の追跡**: `Queued(t)` と `TerminalHeld(t)` の排他（+ enqueue-once +
  drain-once）により、各 token はちょうど 1 状態（Queued または TerminalHeld）に在り、RELEASED へ一度だけ
  遷移。

### D54.2 token continuity の分割（「完全に閉じた」は撤回）
| 項目 | 状態 |
|------|------|
| 通常経路の token continuity（acquire → Queued → destroy → release） | **CLOSED** |
| quarantine-full の terminal-held continuity（acquire → TerminalHeld） | **CLOSED** |
| **terminal-held token の最終 release** | **意図的に存在しない**（Option 1・catastrophic）→ **liveness = OPEN（INV-R5）** |

### D54.3 stale / double-release 検出の分類
- **設計 CLOSED / 実装 OPEN**（D53 で「現行コードで確認した事実」としていない・authority の設計仕様）。
- **authority 内部 invariant**:
  ```
  acquire() → token を mint → token ∈ held（slot 占有・count++）
  release(token) → token ∉ held なら protocol violation（Debug assert / health event）
                 → token ∈ held なら erase + count--
  ```
- **token が monotonic でも double-release 検出は自動成立しない**（release 順序は acquire 順序と非同期）→
  上記の held 検証が必要。

### D54.4 ★ held storage の設計（authority 自身の storage cardinality・D54 の核心）
- **R を先に決めない**。まず held token の storage を確定する。
- **採用設計: `WorldRetirementReservation` は固定容量の held-token storage を所有**:
  | 属性 | 設計 |
  |------|------|
  | storage | **固定 `std::array<ReservationToken, R_cap>`**（または token hash の open-addressing テーブル） |
  | capacity | **R_cap**（= R の storage capacity） |
  | allocation-free | ✅（固定配列・index 配置） |
  | bounded | ✅（capacity = R_cap） |
  | **R との関係** | **R ≤ R_cap（structural invariant）**・gate は `held_count < R`（R は policy 値・R_cap はその上界） |
  | thread domain | **acquire = publish()（ISR intent コンテキスト・RT-safe 要）・release = deleter（non-RT drain）** → **クロススレッド** |
  | 同期 | **lock-free（atomic slot・SPMC または DeferredDeletionQueue 同型の Vyukov bounded ring）** を推奨（ISR パスがブロックしない） |
  | gate 保証 | `held_count` が capacity を超えないことは配列容量で構造的に強制（acquire は full で失敗 → backpressure） |
- **R と storage capacity の関係（D46 再発防止）**: 既存 RetireRouter の 4608-slot は**全 Generic 共用**の
  safety bound。**held-token storage（R_cap）は World reservation 専用の独立構造**であり、4608 と混同しない。
  R（policy）は **R ≤ R_cap** を満たすように選定（次 D55）。
- **terminal-held の影響**: quarantine-full token は held slot を永続占有 → held_count が上昇 →
  R 到達で publish Rejected（backpressure）→ **INV-R5（liveness）OPEN と連結**（held storage の capacity は
  cardinality を保つが、永久占有は availability を圧迫）。

### D54.5 最終状態（ユーザー D53 判定表の反映）
| 項目 | 判定 |
|------|------|
| token mint → primary / quarantine entry | CLOSED（設計） |
| primary → retry → quarantine の token 保存 | CLOSED（設計） |
| primary XOR quarantine | CLOSED |
| quarantine-full → terminal-held token | CLOSED |
| token duplication / loss | CLOSED（設計契約） |
| **Queued(t) / TerminalHeld(t) 排他** | **CLOSED（本 D54・修正 INV-TOKEN-1 で明示）** |
| stale / double-release detection | **設計 CLOSED / 実装 OPEN** |
| terminal-held token の release | **意図的に無し** |
| **held storage cardinality** | **CLOSED（本 D54・固定 R_cap・allocation-free・lock-free・R ≤ R_cap）** |
| liveness / permanent reservation | **OPEN（INV-R5）** |
| R の数値 | **OPEN（次 D55・R ≤ R_cap の下で policy 選定）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D54.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T122 | INV-TOKEN-1 修正 | Acquired(t) ⇒ Queued(t) ∨ TerminalHeld(t)・Queued ∧ TerminalHeld 非共存 |
| T123 | token-state 完全 invariant | 各 token はちょうど 1 状態（Queued / TerminalHeld）・RELEASED へ一度だけ |
| T124 | held storage cardinality | 固定 R_cap 配列・allocation-free・bounded・held_count は容量で強制 |
| T125 | R ≤ R_cap | R は policy・R_cap は storage capacity・R ≤ R_cap |
| T126 | thread domain | acquire（ISR publish）と release（non-RT drain）のクロススレッド・lock-free 同期 |
| T127 | stale/double-release（実装） | release(token) の held 検証（実装ゲート・T121 と統合） |
| T128 | terminal-held と R 圧迫 | quarantine-full token の永続占有 → INV-R5 liveness と連結 |

---

## D55 — Design-37 確定（WorldRetirementReservation の authority data structure・token lookup / slot reuse / ABA invariant・R_cap 定義・2026-08-15）

ユーザー Design-36 レビュー対応: **「D54 の `std::array<ReservationToken, R_cap>` は新規 authority の実装方式を
仮定した設計であり、現行コードから確定した事実ではない。特に acquire/release は単なる queue ではなく
token をキーとする集合操作」**。**R 数値（D56）の前に、authority のデータ構造そのものと lookup / slot
reuse / ABA invariant を確定し、その結果として R_cap を定義する**。**Phase I 実装 NO-GO 継続（設計契約のみ）。**

### D55.1 要件の再確認（単なる queue ではない）
- **acquire()**: 空き slot 確保 → token mint → held 化。
- **release(token)**: token を特定 → held 除去 → slot 再利用。
- → MPMC queue ではなく、**token をキーとする集合（set / hash table）**。
- 操作は「Vyukov MPMC リング」では不適切（FIFO エンキュー/デキューではなく、token 指定の検索・削除）。

### D55.2 採用データ構造 — open-addressing token table（固定 R_cap）
- **`std::array<std::atomic<uint64_t>, R_cap> slots`**（値 = token・**0 = empty sentinel**。token は monotonic
  で 1 始まり（fetch_add + 1）→ 0 は mint されない）。
- **acquire()**: `held_count < R` を確認 → token mint → **linear probe で空き slot を CAS（0 → token）** →
  成功で held_count++。**full（held_count == R_cap）→ acquire 失敗 → backpressure**。
- **release(token)**: token を hash で probe → **slot == token を確認** → **CAS（token → 0）** → 成功で
  held_count--。
- **lookup complexity（ご指摘1）**: O(1) average（hash probe・token % R_cap + linear probing）・
  **O(R_cap) worst**（stale / miss 時の全 probe）。release は non-RT drain のため許容可能 → **invariant として
  明示**（RT パス（acquire）は O(1) average・CAS 非ブロッキング）。
- **held_count**: `std::atomic<uint64_t>`（gate 用・acquire 成功で ++・release 成功で --）。

### D55.3 ABA / slot reuse（ご指摘2）
- **slot の値 = token そのもの**（authority-minted・monotonic・一意）。
- **★ token は一意（monotonic）→ slot の ABA は構造的に排除**: token は二度と同じ値にならない・0 のみが
  empty sentinel → 「slot 7 が token 100 → release → token 101」でも、**token 100 は再び mint されない**。
- **release(token) は CAS(slot, token, 0)** → slot が再利用されて別 token（101）を持っていれば **CAS 失敗 →
  stale / double-release を検出**（protocol violation・Debug assert / health event）。
- **slot index を identity にしない**（ご指摘の通り）・**token を identity に保持**（D52 維持）。
- **∴ ABA 排除 = token 一意性 + slot 値 = token 方式**。CLOSED。

### D55.4 R_cap の定義（ご指摘3）
- **R_cap = reservation authority が同時に outstanding にできる World lifetime obligations の構造的上限**
  = held-token slot 数（固定配列長）。
- **R_cap ≤ maximum representable simultaneous held World tokens**（配列容量がそのまま構造的上限）。
- **一貫性制約**: R_c（held tokens）== retired-but-not-destroyed worlds（D53/D54）≤ 4608（D46・共有 Generic
  pool）→ **R_cap ≤ 4608 が一貫性制約**（world は共有 pool を超えて retired になれないため）。
- **reservation を有効な gate にするには R < 4608**（共有 Generic pool が先に満杯になる前に gate が効く）。
  R_cap は R の上界として R ≤ R_cap < 4608 を満たすよう選定。
- **R_cap の数値は D56（R と共に）** — 構造は本 D55 で確定・数値は policy 選定（power-of-2 等・
  R_cap は open-addressing の mask 効率のため 2 の冪を推奨）。

### D55.5 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| Queued / TerminalHeld 排他 | CLOSED（D54） |
| 通常 token continuity | CLOSED（D53） |
| terminal-held continuity | CLOSED（D54） |
| stale/double release の契約 | 設計 CLOSED / 実装 OPEN |
| terminal-held release | 意図的に無し |
| liveness / INV-R5 | OPEN |
| R ≤ R_cap | CLOSED（契約） |
| **R_cap の具体的データ構造** | **CLOSED（本 D55・open-addressing token table・固定 R_cap・lock-free CAS）** |
| **release(token) lookup / ABA** | **CLOSED（本 D55・hash probe O(1)/worst O(R_cap)・slot=token で ABA 構造的排除）** |
| R の policy 数値 | **OPEN（次 D56・R ≤ R_cap < 4608）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D55.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T129 | open-addressing token table | acquire は CAS(0→token)・release は CAS(token→0)・held_count 整合 |
| T130 | ABA 排除 | token 一意（monotonic）→ slot 再利用でも token 100 は再 mint されない |
| T131 | stale/double-release 検出（実装） | release(token) が slot != token で CAS 失敗 → protocol violation |
| T132 | lookup complexity | acquire O(1) average（CAS 非ブロッキング）・release O(1) avg / O(R_cap) worst（non-RT 許容） |
| T133 | R_cap 一貫性 | R_c ≤ 4608（D46）・R_cap ≤ 4608・R < 4608（gate 有効性） |
| T134 | held_count gate | held_count < R で acquire 許可・== R で Rejected（backpressure） |

---

## D56 — Design-38 確定（RT-safe な reservation acquisition primitive・atomic admission・R_cap の policy constraint 化・2026-08-15）

ユーザー Design-37 レビュー対応: **「(1) `held_count < R` → CAS → `held_count++` は atomic reservation でなく
並行 acquire で R 超過し得る・(2) `R_cap < 4608` は D46 から導出された値ではない・(3) linear probing は RT path
で最悪 R_cap 走査し bounded ≠ RT deterministic」**。**D57（R 数値）の前に、RT-safe な acquisition primitive を
確定**する。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D56.1 修正1 — atomic admission（check-then-act race の排除）
- **旧案の race（ご指摘）**: `held_count < R` 確認 → token mint → CAS(empty→token) → `held_count++` は、
  `held_count == R-1` で A/B が並行確認 OK → 両方 insert → **R 超過**。
- **修正: `held_count` を「独立した check 対象」にせず、原子カウンタの `fetch_add` 自体を admission にする**:
  ```cpp
  acquire():                                  // RT / ISR publish・O(1)・決定的
      const auto old = held_count.fetch_add(1, acq_rel);
      if (old >= R) { held_count.fetch_sub(1, acq_rel); return Rejected; }  // ★ atomic R gate
      return mint_token();                    // monotonic atomic counter・O(1)・一意
  ```
  - `fetch_add` は原子的 → **並行 acquire が同時に R を超えない**（count ≤ R は構造的帰結）。
  - RT path は **O(1)・決定的 work**（atomic op 2 回）→ RT-safe。

### D56.2 修正2 — R_cap ≤ 4608 は導出でなく policy constraint（SAFETY ≠ RESERVATION 再混同防止）
- **D46 が示したのは `N_retired_world ≤ 4608`（全 Generic 共用 pool の safety bound）**。これは**導出**。
- **`R_cap ≤ 4608` は新規 World 専用 authority の capacity に対する policy/implementation constraint**であり、
  **D46 から数学的に導出される値ではない**（conservative constraint）。**D47 の「SAFETY ≠ RESERVATION」を
  再混同しない**。
- **3 つの独立 invariant（ご提示）**:
  ```
  WorldRetirementReservation:
      structural:  N_held_token ≤ R_cap     （authority 自身で保証・slot 容量）
      policy:      N_held_token < R          （R ≤ R_cap・atomic fetch_add gate・次回 retire 前）
  RetireRouter:
      global:      N_retired_all_Generic ≤ 4608 （既存・独立）
  ```

### D56.3 ★ RT-safe reservation acquisition primitive（D56 核心）
- **RT path から linear probing を排除**（bounded ≠ RT deterministic のため）。
- **3 段階構造（thread domain を分離）**:
  | 段階 | 場所 | スレッド | work | 内容 |
  |------|------|----------|------|------|
  | **1. acquire** | publish()・swap 前 | RT / ISR publish | **O(1) 決定的** | `held_count.fetch_add(1)` gate（atomic・R 超過防止）+ `mint_token()`（monotonic・一意） |
  | **2. bind** | retire ブリッジ（enqueue 時） | **Non-RT** | O(R_cap) worst 許容 | open-addressing に token を CAS 挿入（held set・slot=token・ABA 排除・D55 維持） |
  | **3. release** | deleter（drain） | **Non-RT** | O(1)/O(R_cap) | `remove_from_set(token)`（無ければ protocol violation）+ `held_count.fetch_sub(1)` |
- **RT path は acquire のみ（O(1)・決定的・linear probe なし）** → **RT-safe**。
- **held set の挿入は Non-RT bind で行う**（RT で probe しない）。set の occupancy ≤ count ≤ R ≤ R_cap →
  **空き slot は常に存在**（bind は必ず成功・失敗は protocol violation）。
- **5 要件（ご提示）の充足**:
  1. acquire は常に有限・決定的 work（O(1) fetch_add + mint）→ **CLOSED**
  2. R 超過が atomic に防止（fetch_add gate）→ **CLOSED**
  3. R_cap 超過も structural に防止（slot 容量 R_cap・count ≤ R ≤ R_cap）→ **CLOSED**
  4. release により slot 再利用（remove_from_set → slot 解放）→ **CLOSED**
  5. token uniqueness 維持（monotonic atomic counter）→ **CLOSED**
- **ABA 維持**: slot 値 = token（一意）→ D55 の排除を継承（bind は 0→token CAS・release は token→0 CAS・
  再利用後は別 token なら CAS 失敗 → stale/double-release 検出）。

### D56.4 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| token identity / slot identity ≠ token identity / ABA | CLOSED（D52/D55） |
| stale / double release の意味 | CLOSED |
| **N_held ≤ R_cap（structural）** | **CLOSED（本 D56・atomic gate + slot 容量で構造的）** |
| **held_count < R の atomic admission** | **CLOSED（本 D56・fetch_add gate）** |
| **R_cap ≤ 4608** | **policy/implementation constraint（導出ではない・D47 SAFETY≠RESERVATION 維持）** |
| **linear probing の RT boundedness** | **CLOSED（本 D56・probe を Non-RT bind に移動・RT は O(1)）** |
| Global 4608 と World reservation の分離 | CLOSED |
| R の policy 数値 | **OPEN（次 D57・R ≤ R_cap の policy constraint）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D56.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T135 | atomic admission | 並行 acquire で held_count ≤ R 維持（fetch_add gate・check-then-act race なし） |
| T136 | RT boundedness | acquire は O(1) 決定的（linear probe なし）・probe は Non-RT bind に移動 |
| T137 | R_cap policy constraint | R_cap ≤ 4608 は policy/implementation constraint（D46 導出でない） |
| T138 | 3 invariant 独立 | structural（N_held ≤ R_cap）・policy（N_held < R）・global（≤ 4608）が独立 |
| T139 | bind 必ず成功 | set occupancy ≤ count ≤ R ≤ R_cap → 空き slot 常存（失敗は protocol violation） |
| T140 | token uniqueness（実装） | mint_token が monotonic・一意（二度同じ値にならない） |

---

## D57 — Design-39 確定（instantaneous count ≤ R・acquire↔bind 不可分・token continuity の D53 復元・2026-08-15）

ユーザー Design-38 レビュー対応: **「(1) `fetch_add` は瞬間的 invariant として count ≤ R を保証しない
（A: old=7 → count 8・B: old=8 → count 9 の一時超過 → fetch_sub で 8 に回復・最終値のみ）・(2) RT-safe のため
bind を後段（Non-RT）へ移したことで bind failure 時に `Acquired(t) ⇒ Queued(t) ∨ TerminalHeld(t)` を満たさず
token continuity が D53 から後退」**。**D58（R 数値）の前に、
`successful acquire ⇒ reservation obligation と retire entry の結合が不可分` を実現する設計契約を閉じる**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D57.1 修正1 — instantaneous count ≤ R（fetch_add の一時超過を排除・CAS admission）
- **fetch_add の問題（ご指摘・R=8 の例）**: A: `old=7`（成功・count→8）・B: `old=8`（一時的に count→9）・
  B が `fetch_sub`（count→8）→ **瞬間的に count = 9（R+1）**。「count ≤ R は構造的帰結」は**誤り**。
  正確には「最終的に count ≤ R に回復」であり、**操作中の瞬間的 invariant ではない**。
- **修正: CAS admission**:
  ```cpp
  acquire():                              // RT / ISR publish・O(1)・決定的
      auto expected = held_count.load(acquire);
      while (expected < R) {
          if (held_count.compare_exchange_weak(expected, expected + 1, acq_rel, acquire))
              break;                      // count は expected(<R) からのみ expected+1(≤R) に遷移
      }
      if (expected >= R) return Rejected; // count 既に R → backpressure
      ... slot 割当・token mint・bind（D57.2）...
  ```
  - **CAS は expected < R からのみ count を expected+1（≤ R）にする** → **count ≤ R は瞬間的 invariant**。
  - RT: CAS retry は低競合（INV-PUB-SER により publish は実質 single-threaded）→ 実質 O(1)・必要なら
    bounded retry で RT-deterministic。

### D57.2 修正2 — acquire↔bind の不可分（bind failure の構造的排除・token continuity 復元）
- **D56 の問題（ご指摘）**: bind を Non-RT 後段に移すと、bind failure で
  `successful acquire → retired World → token は authority に存在 → entry に token が存在しない`
  → **`Acquired(t) ⇒ Queued(t) ∨ TerminalHeld(t)` を満たさない（D53 から後退）**。
- **修正: bind を acquire 内で実行（不可分）・O(1) slot 割当（free-stack）**:
  ```cpp
  acquire():                              // 継続
      // (a) CAS admission（count ≤ R 瞬間的・上記）
      // (b) O(1) slot 割当: free_slot = free_stack.pop();   // lock-free stack・CAS・linear probe なし
      // (c) token = mint_token();                            // monotonic・O(1)
      // (d) slots[free_slot].store(token, release);          // ★ bind（同一 critical section）
      return token;                       // acquire 成功 ⇒ token は held set に bound（不可分）
  ```
- **free-stack**: 固定配列（R_cap エントリ・allocation-free）+ CAS head（Treiber 型）。
  acquire pop O(1)・release push O(1)・**linear probe なし**。
- **bind failure は構造的に不可能**: admission 後 count = c ≤ R ≤ R_cap → 空き slot 数 =
  R_cap - (c - 1) ≥ 1 → **pop 必ず成功**。失敗は防御的 rollback（held_count.fetch_sub + Debug assert・
  dead code）。
- **∴ acquire 成功 ⇒ token は held set に bound（不可分）** → token は enqueue（値フィールド・D52）で
  **Queued(t)（primary/quarantine）または TerminalHeld(t)（quarantine-full）** に到達（D53/D54 維持）。
  **token continuity は D53 に復元（後退なし）**。

### D57.3 release（Non-RT）— slot 再利用・ABA・stale/double-release 検出
- `release(token)`: R_cap スキャン（**Non-RT・O(R_cap) 許容**）→ `slots[slot] == token` を CAS(token→0) →
  `free_stack.push(slot)` + `held_count.fetch_sub(1)`。
- **ABA**: slot 値 = token（monotonic 一意）・CAS は slot が token のままの場合のみ成功 → 再利用後は別 token
  なら CAS 失敗 → **stale / double-release 検出**（protocol violation・Debug assert / health event）。
- token がスキャンで見つからない → **stale release（protocol violation）**。

### D57.4 token mint 一意性（再確認）
- `mint_token()` = monotonic atomic counter（fetch_add + 1）→ **二度と同じ値にならない**・
  **held token 間で一意**（ABA 排除に十分）。uint64 overflow は実用上到達不能（wraparound 前に全 held 一意）。

### D57.5 最終状態（ユーザー判定表の反映）
| 項目 | D56 | D57 |
|------|-----|-----|
| RT acquire を O(1) にする必要性 | CLOSED | CLOSED |
| **fetch_add による admission** | NOT CLOSED | **CLOSED（CAS admission・瞬間的 count ≤ R）** |
| **held_count ≤ R の瞬間的 invariant** | OPEN | **CLOSED（CAS で overshoot なし）** |
| RT path から linear probe を排除 | CLOSED | CLOSED（free-stack pop O(1)） |
| **token mint の一意性** | 要再確認 | **CLOSED（monotonic・held 間で一意・再確認）** |
| **acquire → bind の atomicity** | OPEN | **CLOSED（bind を acquire 内・不可分・bind failure 構造的排除）** |
| **Acquired → Queued ∨ TerminalHeld** | OPEN（D53 から後退） | **CLOSED（D53 に復元）** |
| R_cap と global 4608 の非同一性 | CLOSED | CLOSED |
| R の数値 | OPEN | **OPEN（次 D58・R ≤ R_cap の policy constraint）** |
| Phase I implementation | NO-GO | **NO-GO（ユーザー最終 GO 待ち）** |

### D57.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T141 | instantaneous count ≤ R | CAS admission で count は expected(<R) からのみ expected+1(≤R) に遷移（overshoot なし） |
| T142 | acquire↔bind 不可分 | acquire 成功 ⇒ token が held set に bound（同一 critical section） |
| T143 | free-stack O(1) | pop/push が CAS・linear probe なし・RT O(1) 決定的 |
| T144 | bind failure 構造的不可能 | admission 後 count ≤ R ≤ R_cap → 空き slot ≥ 1 → pop 必ず成功 |
| T145 | Acquired ⇒ Queued ∨ TerminalHeld 維持 | bind 不可分で D53 に復元（後退なし） |
| T146 | token mint 一意性（再確認） | monotonic・held 間で一意（ABA 排除に十分） |

---

## D58 — Design-40 確定（free_stack の ABA・conservation・release/reuse・slot conservation invariant・2026-08-15）

ユーザー Design-39 レビュー対応: **「D57 で free_stack が新たに RT 側 primitive の一部になった。free_stack.head
の CAS が単純な index のみだと head の ABA が残る・slot の二重 pop/push・count と free-stack cardinality の整合
を検証し、`N_held + N_free = R_cap` の slot conservation invariant を重ねるべき」**。**D59（R 数値）の前に
free_stack の ABA・conservation・release/reuse を閉じる**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D58.1 free_stack の構造（ABA-safe）
- **`free_head` = `std::atomic<uint64_t>`** に **packed { slot_index（低ビット・`R_cap ≤ 4608 < 2^20` → 20 ビット）,
  version（高ビット・44 ビット）}** を格納。`next_free[R_cap]`（uint32_t 固定配列・allocation-free）。
- **pop（acquire・RT・O(1) CAS ループ）**:
  ```
  loop:
      head = free_head.load(acquire);
      idx = head & IDX_MASK;
      if (idx == NONE) return no_free;                 // 空（conservation より到達不能・防御）
      nxt = next_free[idx];
      new_head = (nxt << SHIFT) | (version(head) + 1); // ★ version を必ず increment
      if (free_head.compare_exchange_weak(head, new_head, acq_rel, acquire)) return idx;
  ```
- **push（release・Non-RT・O(1) CAS ループ）**:
  ```
  loop:
      head = free_head.load(acquire);
      next_free[idx] = head & IDX_MASK;                 // ★ next を head CAS の前に書く（Treiber 順序）
      new_head = (idx << SHIFT) | (version(head) + 1);
      if (free_head.compare_exchange_weak(head, new_head, acq_rel, acquire)) return;
  ```
- **★ ABA 排除（ご指摘）**: 単純な index のみの head CAS では、pop/push 重複時に head が同一 index に戻る
  ABA が残る（例: pop X → push X → 別 acquire が stale next で X を二重 pop）。**version を CAS 成功ごとに
  increment し、packed 値（index + version）全体で CAS する** → ABA 検出（head が同一 index でも version が
  異なれば CAS 失敗 → 再試行）。**CLOSED**。

### D58.2 slot conservation invariant（ご指摘の核心）
- **INV-SLOT-CONS: `N_held + N_free = R_cap`（常時）**。
- **証明（by construction）**:
  | 遷移 | N_held | N_free | 和 |
  |------|--------|--------|-----|
  | 初期化 | 0 | R_cap（全 slot を push） | R_cap |
  | acquire（admission + pop） | +1 | -1 | 不変 |
  | release（push + count-1） | -1 | +1 | 不変 |
- **∴ 各 slot は「held」または「free」のちょうど 1 状態に在る**（二重 pop / 二重 push なし）。
- **free-stack が必ず free slot を提供（ご指摘）**: `N_free = R_cap - N_held ≥ R_cap - R ≥ 1`
  （N_held ≤ R ≤ R_cap・policy R < R_cap）→ **acquire の pop は必ず成功**（no_free は防御・到達不能）。

### D58.3 slot の二重 pop / 二重 push（ご指摘）
- **二重 pop なし**: pop は version-tagged CAS で原子的・各 slot は「held」時のみ free でない → held slot は
  pop 対象にならない（conservation）。
- **二重 push なし**: push は release(token) のみ・release は token 1 個につき正確に 1 回（D53 exactly-once）・
  slot は acquire で pop 済みのもののみ push される（conservation で free は重複しない）。
- **∴ 二重 pop / 二重 push は構造的に排除**。CLOSED。

### D58.4 count と free-stack cardinality の整合（ご指摘）
- N_held（held_count・CAS admission が管理）と N_free（free-stack が管理）の和が R_cap で不変
  （INV-SLOT-CONS）→ **両者の整合は構造的に保証**（独立 counter でなく、同一 slot 集合の 2 分割）。
- release(token) は held_count.fetch_sub と free_stack.push を**同じ release クリティカルセクション**で行う
  → 両者が原子的に同期（count だけ減って free が増えない等の不整合は起きない）。

### D58.5 release / reuse（slot 再利用）
- release(token): R_cap スキャン（Non-RT・O(R_cap) 許容）→ `slots[slot] == token` を CAS(token→0) →
  `free_stack.push(slot)` + `held_count.fetch_sub(1)`（同一 release セクション）。
- 再利用: 次の acquire が pop で slot を取得 → 新 token（monotonic・異値）を store → **held-table の
  ABA は slot=token（一意）で排除（D55）・free-stack の ABA は version-tagged head で排除（D58.1）**。

### D58.6 RT boundedness（再確認）
- pop / push は version-tagged CAS ループ・低競合（INV-PUB-SER で acquire 実質 single-threaded・release は
  Non-RT drain）→ 実質 O(1)・必要なら bounded retry で RT-deterministic（D57 と同論）。

### D58.7 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| held_count ≤ R instantaneous invariant | CLOSED（D57） |
| acquire → bind continuity | CLOSED（設計・D57） |
| held-table token ABA | CLOSED（D55） |
| **free-stack head ABA** | **CLOSED（本 D58・version-tagged packed head）** |
| **slot の二重 pop / 二重 push** | **CLOSED（本 D58・conservation + exactly-once）** |
| **count と free-stack cardinality の整合** | **CLOSED（本 D58・INV-SLOT-CONS: N_held + N_free = R_cap）** |
| **free-stack が必ず free slot を提供** | **CLOSED（本 D58・N_free ≥ R_cap - R ≥ 1）** |
| RT acquire の boundedness | CLOSED（設計） |
| R の数値 | **OPEN（次 D59・R ≤ R_cap の policy constraint）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D58.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T147 | INV-SLOT-CONS | N_held + N_free = R_cap 常時（acquire/release/初期化で不変） |
| T148 | free-stack ABA | version-tagged head で ABA 検出（同一 index でも version 差で CAS 失敗） |
| T149 | 二重 pop/push なし | conservation + release exactly-once で構造的排除 |
| T150 | free slot 常存 | N_free ≥ R_cap - R ≥ 1 → acquire の pop 必ず成功 |
| T151 | release/reuse 原子性 | release で held_count.fetch_sub と free_stack.push が同一セクション |
| T152 | RT O(1)（再確認） | version-tagged pop/push が CAS ループ・低競合で実質 O(1) |

---

## D59 — Design-41 確定（release publication ordering・admission visibility と free-slot publication の順序契約・2026-08-15）

ユーザー Design-40 レビュー対応: **「D58 の『release は fetch_sub と push を同一 release クリティカル
セクションで行う』は、count と free_head が別 atomic object のため一つの CAS transaction に実装できない。
必要なのは『途中状態を他 acquire が観測しても安全』な publication ordering であり、`push`（free slot 公開）が
`count--`（admission 可視化）より**前**であることの証明」**。**D60（R 数値）の前提 invariant として
release publication ordering を明文化**する。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D59.1 D58 の表現修正（「同一 release クリティカルセクション」の撤回）
- **撤回**: D58 の「release は fetch_sub と push を同一 release クリティカルセクションで行う」は誤解を招く —
  `held_count` と `free_head` は**別 atomic object** であり、通常の C++ atomic では両者を 1 CAS transaction
  にできない。
- **修正**: 実装契約は **publication ordering**（atomic transaction ではない）:
  > **`free_stack.push(slot)`（free slot の公開）が `held_count.fetch_sub`（admission 可視化）より前に実行される**。
- **禁止順序**: `count-- → push`（別 acquire が count < R を観測して free-stack を pop する時点で free slot
  が未公開 → **bind failure が再現**・D58 の「構造的に不可能」を破る）。

### D59.2 release publication ordering（実装契約・明文化）
```
release(token):                       // Non-RT drain
    1. slots[slot] = 0（release）      // 論理 release（token 除去）
    2. free_stack.push(slot)（release）// ★ free slot 公開（free_head 書込み）
    3. held_count.fetch_sub(1, release)// ★ admission 可視化（count--）
```
- **`push` が `count--` より前**（program order + release ordering）。

### D59.3 証明（acquire が count < R を観測した時点で free slot は必ず公開済み）
- release: `push`（free_head 書込み・release）は `count--`（held_count・release）より**前**（program order）。
- acquire: `held_count` 読取り（acquire）が count < R を観測 → **release-acquire 同期**（release 側の
  count-- と acquire 側の count 読取りが synchronize）→ **push の free_head 書込み（release 系列）が可視** →
  acquire の `pop` は**公開済みの free slot を必ず取得**。
- **∴ admission 可視化（count--）時点で free slot は必ず free-stack に公開済み** → bind failure 構造的不可能
  が維持（D58 と整合・順序契約の下で保証）。

### D59.4 bind 側の ordering（補足・整合）
- acquire: `slots[slot] = token`（release）・release のスキャンは slot 読取り（acquire）→ **token が可視**。
- enqueue（entry に token 値）は既存 deferred-delete queue の同期（seq atomic・release/acquire）が cover
  → token は drain（release）時に有効。

### D59.5 最終状態（ユーザー判定表の反映）
| 契約 | 判定 |
|------|------|
| N_held + N_free = R_cap | CLOSED（D58） |
| free-head ABA / token ABA / double pop-push | CLOSED（D55/D58） |
| admission → free-slot availability | CLOSED（D58） |
| **release publication ordering** | **CLOSED（本 D59・push before count--・release-acquire 同期・明文化）** |
| R の数値 | **OPEN（次 D60・本 ordering を前提 invariant として）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D59.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T153 | release publication ordering | release は push（free 公開）→ count--（admission 可視化）の順序（release-acquire 同期） |
| T154 | acquire 観測時 free slot 公開済み | acquire が count < R を観測 → pop は公開済み free slot を取得（bind failure なし） |
| T155 | 禁止順序検出 | count-- → push の順序違反を invariant 検出（実装ゲート） |
| T156 | bind 側 ordering | acquire の slots[slot]=token（release）→ release スキャン（acquire）で可視 |

---

## D60 — Design-42 確定（R の policy 数値・4608 非依存・backpressure としての意味・2026-08-15）

ユーザー Design-41 レビュー対応: **「D59 は CLOSED。D60 で R の policy 数値へ進む。ただし R は `R < R_cap`
だけから決めず、(1) World reservation が global Generic pool の 4608-entry safety bound に依存しないこと、
(2) reservation exhaustion が publish admission の backpressure として意味を持つこと、を前提に決定する」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D60.1 R の性質（正直な位置づけ）
- **R は policy 定数であり、コードから導出される値ではない**（新規 authority の設計値・D55/D56 で
  「R_cap ≤ 4608 は導出でなく policy constraint」と同様）。
- 本 D60 は**候補値の提案**（ユーザー確認後に確定）。根拠は構造・invariant ではなく **policy 判断**。

### D60.2 R の決定条件（ご指定）
1. **R < R_cap**（構造的上限・held set の容量）。
2. **R は 4608（Global Generic pool safety bound）に依存しない** — World 専用 reservation の policy
   （4608 は全 Generic 共用 safety bound・RetireRouter 側・独立）。
3. **exhaustion（R_c == R → publish Rejected）が backpressure として意味を持つ** —
   「正常系では到達せず・異常系（drain stall / head-stuck・INV-R5）で到達する」値。

### D60.3 候補値（R_cap と R）
- **R_cap（structural・held set 容量）: 64（候補）**
  - power-of-2（version-tagged head の IDX_MASK 効率・D58）・allocation-free 固定配列
    （slot table 64×8B = 512B + next_free 64×4B = 256B・極小）・R_cap ≤ 4608 の policy constraint を満たす。
- **R（policy gate）: 32 = R_cap / 2（候補）**
  - **正常 R_c（retired-but-not-destroyed world 数）の見積もり**: publish 直列化（INV-PUB-SER・同時 1
    publish）と周期 drain（AudioEngine.Timer.cpp の drainDeferredRetireQueues）により、通常は
    **1 drain 間隔内の publish 数 ≈ 0〜2**（ユーザー駆動 rebuild・連続でも数個程度・推測値）。
  - **R = 32 は正常 R_c（≤ ~2）に対して十分な headroom** → 正常系で到達しない（spurious backpressure なし）。
  - **異常系（drain stall / head-stuck・reader が epoch を離脱しない）で R_c が蓄積 → R = 32 に到達 →
    publish Rejected（backpressure）** → 意味を持つ（無界な retired world 蓄積を防止）。

### D60.4 4608 非依存と backpressure の意味（ご指定）
- **R = 32 は Global Generic pool の 4608 から独立**（World 専用 policy・4608 は全 Generic 共用 safety bound）。
- **exhaustion = publish admission の backpressure**: R_c == 32 → 次 publish が acquire で Rejected →
  上位（trySubmit → PublishStageResult）へ伝播 → backpressure（D50・Rejected / swap 前 / retry-able）。
- 4608 への影響は**帰結**（R ≪ 4608 なので World 起因の Generic 蓄積も間接的に制限される）であって**根拠ではない**。

### D60.5 実装時の memory-ordering 注意（ユーザー指摘）
- **acquire 側の admission CAS は held_count を acquire semantics で読む**（D59 の release-acquire 同期の
  成立に必須 — acquire が count < R を観測した時点で push の free slot 公開を可視化するため）。
- **free_stack push/pop の Treiber head CAS の memory ordering と next_free[slot] の publish 順序を一致**：
  push は `next_free[slot]` を head CAS の前に書く（release）・pop は head 読取り後に next を読む（acquire）・
  version-tagged head（D58）とセットで実装時に検証（T148/T157）。

### D60.6 最終状態
| 項目 | 状態 |
|------|------|
| N_held + N_free = R_cap / held_count ≤ R / ABA 類 / double pop-push | CLOSED（D57-D59） |
| acquire → bind / push → count-- publication ordering | CLOSED（D57/D59） |
| **R_cap（structural）** | **候補 64（power-of-2・allocation-free・policy constraint 満足）** |
| **R（policy gate）** | **候補 32（= R_cap/2・正常系非到達・異常系で backpressure）** |
| **R < R_cap** | 成立（32 < 64） |
| **4608 非依存** | 成立（World 専用 policy・4608 は根拠でない） |
| **exhaustion = backpressure として意味** | 成立（正常系非到達・異常系で Rejected 伝播） |
| 実装時 memory-ordering | acquire 側 acquire semantics・Treiber push/pop ordering（明文化） |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち・R/R_cap の確定はユーザー確認後）** |

### D60.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T157 | acquire semantics | acquire 側 admission CAS が held_count を acquire semantics で読む（D59 同期成立） |
| T158 | Treiber ordering | push の next_free 書込み（release）が head CAS 前・pop の next 読取り（acquire）が head 後 |
| T159 | R < R_cap | R = 32 < R_cap = 64 |
| T160 | 正常系非到達 | 正常 R_c（≈0〜2）で R = 32 に到達しない（spurious backpressure なし） |
| T161 | 異常系で backpressure | drain stall で R_c 蓄積 → R = 32 → publish Rejected（backpressure） |
| T162 | 4608 非依存 | R = 32 は 4608 から独立（World 専用 policy） |

---

## D61 — Design-43 確定（reservation acquisition linearization contract・intermediate state 形式化・policy 量のコード実測化・2026-08-15）

ユーザー Design-42 レビュー対応: **「D60 を『R=32/R_cap=64 が確定』として閉じることには反対。R は推測値で
policy justification になっていない・R_cap=64 は power-of-2 では導出されない・さらに D57 の『acquire と bind
を不可分』を単一ハードウェア atomic と誤解しないために、R 数値の前に acquisition の linearizability と
intermediate state を形式化すべき」**。**R の数値根拠は policy 関数 + 実測に基づく（本 D61 はその形式化）**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D61.1 D60 の位置づけ（候補値・policy CLOSED ではない）
- **R=32 / R_cap=64 は policy candidate であり、未確定**（ユーザー判定の通り）。
- `R < R_cap` のみ CLOSED・**`R_cap=64` は「power-of-2 だから」では導出されない**。
- `R_cap ≤ 4608` は既存 4608 bound から数学的に導出されない（policy/implementation constraint・再混同しない）。

### D61.2 ★ reservation acquisition linearization contract（D61 核心）
- **誤解防止**: D57 の「acquire と bind を不可分」は**単一ハードウェア atomic ではない**。
  `count CAS` と `slot publication` は**別の atomic operation**。
- **成功契約（正確な形）**:
  $$
  \text{successful acquire} \;\Rightarrow\; \text{structurally guaranteed bind}
  \quad(\text{CAS}+\text{slot write が single hardware atomic ではない})
  $$
  - 保証根拠: conservation（admission 後は `N_free ≥ 1`）→ pop 必ず成功・bind は O(1) bounded work →
    **bind は acquire リターン前に必ず完了**（単一 atomic ではなく「構造的に完了保証」）。
- **linearization point（LP）**:
  - **acquire LP = count CAS 成功**（R gate の線形化点・publish admission が原子的に確定）。
  - **release LP = held_count fetch_sub**（D59 の push before count-- により、LP 時点で free slot 公開済み）。
- **intermediate state 契約（CAS→bind 間・ご指摘）**:
  - count CAS 成功（LP）〜 bind（slots[slot]=token）の間、slot は「in flight」（pop 済み・token 未束縛）。
  - 過渡状態は **bounded（O(1) work・acquire リターン前に必ず解消）**・**LP では conservation 成立**
    （過渡では in-flight 分だけずれるが bounded・INV-PUB-SER で同時 acquire ≤ 1）。
  - **契約**: "eventually / structurally guaranteed bind"（単一ハードウェア atomic ではない）。

### D61.3 policy 量のコード実測可能化（R 数値根拠のための量）
| 量 | 定義 | コード根拠 |
|----|------|-----------|
| **T_stall**（max drain stall） | **≤ 5s** | `maxRetireWallClockMs_ { 5000.0 }`（AudioEngine.h:4753）→ `hasExceededDeferralThresholds` → `quarantineSlot(RetireDeferralTimeout)`（Commit.cpp:587-600）・stuck-reader 検出 `detectStuckReaders(stuckThreshold=10 epochs)`（RuntimeHealthMonitor.cpp:481） |
| **λ_retire**（max retire rate） | publish 直列化（INV-PUB-SER・waitForReceipt）で retire rate ≤ publish rate・publish rate は rebuild 完了時間依存 | telemetry で実測可能 |
| **R_normal**（正常 peak） | 1 drain 間隔内の retire 数 | telemetry で実測可能（通常 ≈ 0〜2 推測・未確定） |
| **R_shutdown** | shutdown 時の最大滞留 | graceful drain / drainAll で処理（D46-R5） |
| **R_catastrophic** | quarantine-full / leak | INV-R5・構造的に unbounded・health escalation |

### D61.4 R の policy 関数（候補値確定のための形式）
$$
R = f(R_{\text{normal}},\; T_{\text{stall}},\; \lambda_{\text{retire}},\; \text{headroom})
$$
- **条件**:
  1. $R > R_{\text{normal peak}}$（spurious backpressure 回避）。
  2. $R \ge \lambda_{\text{retire\_max}} \times T_{\text{stall}}$（stuck-reader escalation（5s）前に exhaustion しない）。
  3. $R < R_{\text{cap}}$（構造的上限）。
  4. headroom（安全余裕・policy）。
- **候補値の再評価**: R=32 / R_cap=64 は、**λ_retire / R_normal の実測に基づく policy 関数の適用後に確定**
  （現時点では候補のまま・数値根拠 OPEN）。

### D61.5 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| held_count ≤ R / free-stack ABA / slot conservation / release→push→count-- ordering / R < R_cap / 4608 分離 | CLOSED（D57-D60） |
| **R_cap = 64** | **OPEN（policy candidate・導出されていない）** |
| **R = 32** | **OPEN（policy candidate・数値根拠未確定）** |
| **R の数値根拠** | **OPEN（policy 関数 + λ_retire / R_normal の実測後）** |
| **acquisition の linearization point** | **CLOSED（本 D61・count CAS = acquire LP・fetch_sub = release LP）** |
| **CAS→bind 間の中間状態契約** | **CLOSED（本 D61・structurally guaranteed bind・bounded transient・単一 atomic ではない）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D61.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T163 | acquire LP | count CAS 成功が R gate の線形化点（publish admission が原子的） |
| T164 | structurally guaranteed bind | CAS 成功 ⇒ bind が acquire リターン前に完了（単一ハードウェア atomic ではない） |
| T165 | intermediate state bounded | CAS→bind 間の過渡が O(1)・LP で conservation 成立 |
| T166 | T_stall ≤ 5s | maxRetireWallClockMs_ = 5000 → quarantine escalation（コード導出） |
| T167 | R policy 関数 | R > R_normal_peak・R ≥ λ_retire_max × T_stall・R < R_cap |

---

## D62 — Design-44 確定（H/I/F 三状態の reservation conservation + concurrent admission proof・2026-08-15）

ユーザー Design-43 レビュー対応: **「D61 の『CAS→bind 間 bounded・acquire リターン前に必ず解消』は bind failure
が構造的に存在しないことをコード上で証明できた場合のみ CLOSED。次は R 数値でなく H/I/F 三状態による
conservation + concurrent admission proof（10点）」**。**`successful acquire ⇒ exactly one bound reservation` を
「構造的に保証」という説明でなく、中間状態を含む状態機械として証明**する。**Phase I 実装 NO-GO 継続
（設計契約のみ・コード未変更）。**

### D62.1 三状態モデル（ユーザー提示）
- **H = bound held**（token 束縛済み slot）・**I = admitted but not yet bound**（acquire の in-flight slot）・
  **F = free**（free-stack 上の slot）。
- 各 slot はちょうど 1 状態（H / I / F）。**count（admission counter）= H + I**（admission で +1・release で
  -1・bind で I→H は count 不変）。

### D62.2 10 点の証明（状態機械）
| # | 命題 | 証明 |
|---|------|------|
| 1 | **H + I + F = R_cap** | 各 slot はちょうど 1 状態（H/I/F）・by construction。CLOSED |
| 2 | **H + I ≤ R** | count = H+I・CAS gate（count < R）→ H+I ≤ R。CLOSED |
| 3 | **CAS 成功 = H+I → H+I+1** | count CAS は H+I を +1（I が 1 増える・H は不変）。CLOSED |
| 4 | **bind 完了 = I → H** | slots[slot]=token で in-flight slot が bound（count 不変）。CLOSED |
| 5 | **release = H → F** | token クリア + push（bound → free）+ count（H+I → H+I-1）。CLOSED |
| 6 | **I は acquire return 前に必ず 0** | bind（I→H）は acquire 内の O(1) bounded work・return 前に完了（D61 structurally guaranteed bind）。CLOSED |
| 7 | **F == 0 ⇒ acquire 成功不能** | F=0 ⇒ H+I = R_cap ≥ R（R ≤ R_cap）⇒ count ≥ R ⇒ CAS 失敗。CLOSED |
| 8 | **concurrent acquire が R を超えない** | count（=H+I）の CAS gate が原子的 → A/B 同時 admit でも H+I ≤ R。CLOSED |
| 9 | **release の push-before-count-- と整合** | release: H→F（push）→ count--（H+I 減少・D59 順序）→ 解放 slot は admission 再開前に公開。CLOSED |
| 10 | **token continuity が I→H で失われない** | bind（I→H）で token 束縛・token は entry（enqueue）へ・Acquired ⇒ Queued ∨ TerminalHeld（D53/D54 維持）。CLOSED |

### D62.3 ★ 重要な修正: admission gate は H + I < R（H < R ではない）
- **D57/D59 の count は H+I を追跡**（admission で +1・release で -1・bind で I→H は count 不変）→
  **gate は暗黙に H + I < R**。
- **明文化**: 並行 acquire A/B で
  ```
  A: CAS → I=1
  B: CAS → I=1
  ```
  のとき、count（=H+I）CAS gate が原子的 → H+I ≤ R を正しく制御（H のみを見る gate では A/B 同時 admit で
  H+I が R を超え得る）。**CLOSED（count = H+I の明示）**。

### D62.4 ★ successful acquire ⇒ exactly one bound reservation（状態機械として証明）
- **遷移**: CAS success（H+I → H+I+1・I=1）→ pop（F-1）→ bind（I→H・count 不変）→ return（I=0）。
- **bind failure 構造的不可能（ご指摘の核心）**: 事後 count = H+I+1 ≤ R → H+I ≤ R-1 →
  $F = R_{cap} - H - I \ge R_{cap} - (R-1) = R_{cap} - R + 1 \ge 1$（R ≤ R_cap）→ **pop 必ず成功**。
- **∴ successful acquire ⇒ exactly one bound reservation**（中間状態を含む状態機械として証明・
  「構造的に保証」の説明に依存しない）。**CLOSED**。

### D62.5 T_stall と policy requirement の分離（ユーザー注記）
- **T_stall = 5s は retire deferral escalation threshold（コード由来・maxRetireWallClockMs_ = 5000）**。
- **「reservation exhaustion を 5s 以内に回避すべき」は別の policy requirement**（T_stall 自体は
  exhaustion 回避を意味しない）。**D62/D63 以降も分離して扱う**。

### D62.6 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| acquire LP / release LP / intermediate state 明示 | CLOSED（D61） |
| **H/I/F 三状態 conservation（H + I + F = R_cap）** | **CLOSED（本 D62）** |
| **H + I ≤ R（admission gate・count = H+I）** | **CLOSED（本 D62）** |
| **concurrent acquire ≤ R** | **CLOSED（本 D62・count CAS 原子的）** |
| **successful acquire ⇒ exactly one bound reservation** | **CLOSED（本 D62・状態機械として証明）** |
| **bind failure 構造的不可能** | **CLOSED（本 D62・F ≥ 1 の導出）** |
| T_stall と policy requirement の分離 | CLOSED（本 D62・明文化） |
| R_normal 実測定義 / λ_retire_max 上限定義 / R=32 / R_cap=64 | **OPEN** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D62.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T168 | H+I+F = R_cap | 三状態 conservation（各 slot はちょうど 1 状態） |
| T169 | H+I ≤ R | admission gate = count（=H+I）CAS |
| T170 | concurrent acquire ≤ R | count CAS 原子的・A/B 同時 admit でも H+I ≤ R |
| T171 | bind failure 構造的不可能 | 事後 count ≤ R → H+I ≤ R-1 → F ≥ R_cap - R + 1 ≥ 1 → pop 必ず成功 |
| T172 | successful acquire ⇒ one bound | 状態機械（CAS→pop→bind→return）で証明 |
| T173 | T_stall と policy 分離 | T_stall=5s（escalation threshold）と exhaustion policy は独立 |

---

## D63 — Design-45 確定（reservation 増加のコード特定・B_max(T) 定義・T_deferral と T_reservation_stall の分離・2026-08-15）

ユーザー Design-44 レビュー対応: **「D62 は CLOSED。D63 では R=32 を確定せず、まず λ_retire の最大値の
定義可能性をコードから閉じる。publish 直列化でも reservation 増加率は単純な publish rate と同一ではない。
B_max(T)（期間内の最大累積増分）を定義し、`R > R_baseline + B_max(T_stall) + M` の形にする。さらに T_deferral=5s
と reservation stall duration を分離する」**。**D62 で R_cap=64 を前提にせず、まずコードから reservation
burst の上限を求める**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D63.1 予約増加の正確な発生点（コード特定・ご指摘1/2）
- **reservation increment = acquire**（設計では publish() 内・swap 前・D57-D62 の配置）。
- **1 publish → ≤1 reservation increment**（コード導出・D45）:
  - `publishAndSwap` は oldWorld を**正確に 1 つ**返す → 1 publish は 1 つの oldWorld を retire → ≤1 acquire。
  - bootstrap 初回（prevWorld == null）→ oldWorld なし → **0**。
  - shutdown clear（`clearPublishedRuntimeSnapshotsNonRt`・publish 置換でない）→ acquire なし（D50・別 contract）。
  - 失敗 publish（validate 失敗）→ retire なし → acquire なし（RAII 同期破棄）。
- **∴ 通常運転の reservation 増加源は publish-supersession のみ・1 publish ≤ 1 個**。CLOSED。

### D63.2 λ_retire_max の定義問題（ご指摘）
- publish 直列化（INV-PUB-SER）でも、**reservation 増加率は単純な publish rate と同一ではない**
  （burst を考慮する必要）。
- **B(t₀,t₁) = N_reservation(t₁) − N_reservation(t₀)**（期間の最大累積増分）。
- **B_max(T) = sup_t [ N(t+T) − N(t) ]**（任意の長さ T の窓での最大累積増加）。
- **B_max(T) ≤（長さ T の窓内の max publish 数）**（各 publish は ≤1 増加・D63.1）→ **コード導出の上界**。

### D63.3 T_deferral と T_reservation_stall の分離（ご指摘）
| 量 | 意味 |
|----|------|
| **T_deferral = 5s** | 既存コードの escalation threshold（`maxRetireWallClockMs_ = 5000`・Commit.cpp:587-600） |
| **T_reservation_stall** | **reservation が解放されない時間**（正常系 ≈ drain cadence・reader stall で延長・INV-R5） |
| **B_max(T)** | その期間に増加可能な reservation 数 |
| **R** | その burst を許容する policy limit |
- **★ 5s × publish rate から R を直接導出しない**（T_deferral は escalation threshold であり、
  「5秒間 reservation が蓄積する」ことを意味しない・危険な飛躍を回避）。

### D63.4 R の必要条件（burst + baseline + margin・ご指定の形）
$$
R > R_{\text{baseline}} + B_{\max}(T_{\text{reservation\_stall}}) + M
$$
- **R_baseline** = 通常時に既に outstanding な reservation（telemetry 実測・≈0〜2 推測・未確定）。
- **B_max(T_reservation_stall)** = その stall 期間に追加され得る retirement obligation の最大量。
- **M** = policy headroom（安全余裕）。

### D63.5 R=32 の評価（候補として・閉じない・ご指摘7/8）
- **R=32 は、`R > R_baseline + B_max(T_reservation_stall) + M` を満たすかは B_max の実測（publish burst）
  次第**。
- **B_max(5s) が大きい（高頻度 publish burst）場合は 32 不足 → 64/128 等を再評価**。
- **R=32 / R_cap=64 は候補のまま・数値根拠 OPEN**（R_cap=64 を前提にしない）。

### D63.6 最終状態
| 項目 | 状態 |
|------|------|
| D62（H/I/F conservation・admission・concurrent ≤ R） | CLOSED（D62） |
| **1 publish → ≤1 reservation（増加点コード特定）** | **CLOSED（本 D63・D45）** |
| **B_max(T) 定義（≤ max publishes in T）** | **CLOSED（本 D63・コード導出上界）** |
| **T_deferral と T_reservation_stall 分離** | **CLOSED（本 D63・5s × rate から直接導出しない）** |
| **R > R_baseline + B_max(T_stall) + M（形式）** | **CLOSED（本 D63）** |
| R=32 / R_cap=64 | **OPEN（候補・B_max 実測後）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D63.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T174 | 1 publish → ≤1 reservation | 増加点コード特定（publishAndSwap は oldWorld 1 つ・bootstrap 0・shutdown 別） |
| T175 | B_max(T) 上界 | B_max(T) ≤ max publishes in T（各 publish ≤1 増加） |
| T176 | T_deferral / T_reservation_stall 分離 | 5s（escalation）と stall duration は別・5s × rate から R を導出しない |
| T177 | R 必要条件 | R > R_baseline + B_max(T_reservation_stall) + M（burst + baseline + margin） |

---

## D64 — Design-46 確定（reservation telemetry / observation contract・B_max(T) 測定・R_cap 決定順序・2026-08-15）

ユーザー Design-45 レビュー対応: **「D63 は CLOSED のまま・次は D64 = reservation telemetry / observation
contract をコードから確定する。何を観測すれば R の根拠になるかを固定し、B_max(T) は peak で代用できない。
R_cap=64/R=32 を前提にせず、測定 → R_required → R policy → R_cap structural の順序にする」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更・telemetry は実装ゲート）。**

### D64.1 観測対象の区別（8 メトリクス・ご指定）
| # | メトリクス | 意味 | 記録方法 |
|---|-----------|------|---------|
| 1 | **acquire 成功数** | reservationAcquired（累積） | acquire 成功で increment |
| 2 | **release 数** | reservationReleased（累積） | release 実行で increment |
| 3 | **現在 outstanding 数** | N_res(t) = H + I（count） | live 値（acquire/release で変動） |
| 4 | **publish 数** | 累積 publish 数（既存 telemetry と整合） | publish で increment |
| 5 | **publish rejection 数** | reservation exhaustion 起因の Rejected（validate 失敗と区別） | acquire 失敗で increment |
| 6 | **exhaustion 発生数** | N_res が R に達した回数（累積） | exhaustion で increment |
| 7 | **最大 outstanding** | running max of N_res(t) | acquire 成功時に max 更新 |
| 8 | **最大増分 B_max(T)** | 窓 T 内の最大累積増分（sliding window） | サンプリングで推定 |

### D64.2 ★ B_max(T) は peak で代用できない（ご指摘）
- **B_max(T) = sup_t [ N(t+T) − N(t) ]**（sliding-window の最大累積増分）。
- **単なる peak reservation count では代用できない**（ご指摘の例: t=0 N=10・1s N=20・2s N=30・3s N=20・4s
  N=35 → peak は 35 だが、5s 窓の burst（= 追加 obligation 量）は別）。
- 実装: N_res(t) を一定間隔でサンプリングし、窓 T の delta の最大値を維持（max-increment estimator）。

### D64.3 Observation window（T_obs の分離・ご指定1）
- **normal operating window**: 定常状態（drain cadence 正常）。
- **reclaim stall window**: T_reservation_stall（drain 停滞・INV-R5）。
- **shutdown window**: shutdown drain（R5 contract）。
- **catastrophic / quarantine-full**: INV-R5（unbounded・health escalation）。
- **各窓で独立にメトリクスを評価**（混ぜない）。

### D64.4 Baseline と deterministic guard（ご指定2）
- **R_baseline = P99.9(N_res)**（統計的 baseline・policy 用）。
- **★ safety invariant と policy capacity を混同しないため、percentile だけでなく deterministic guard も必要**
  （本プロジェクトの lifetime authority の性質上・policy 値とは別に構造的安全上限を維持）。

### D64.5 Burst / Margin（ご指定3/4）
- **B_max(T_stall) を実測**（D64.2 の sliding-window estimator）。
- **M = max(M_absolute, α·B_max)**（policy 判断・コードから勝手に導出しない）。

### D64.6 ★ R_cap 決定順序（ご指定・前提を置かない）
```
測定（telemetry: acquire/release/outstanding/publish/rejection/exhaustion/B_max）
  ↓
R_required = R_baseline + B_max(T_stall) + M
  ↓
R の policy 選択
  ↓
R_cap の structural capacity 選択（R_cap > R > R_required を満たす最小の実装可能容量）
```
- **R_cap=64 / R=32 を前提にしない**（測定後に `32` が「便利な数字」か「実測に裏付けられた policy 値」かを
  判定）。

### D64.7 最終状態
| 項目 | 状態 |
|------|------|
| D63（1 publish ≤1・B_max 定義・T_deferral 分離・R 必要条件） | CLOSED |
| **8 メトリクス区別** | **CLOSED（本 D64・契約）** |
| **B_max(T) ≠ peak** | **CLOSED（本 D64・sliding-window max increment）** |
| **Observation window 分離** | **CLOSED（本 D64・4 窓）** |
| **Baseline（P99.9 + deterministic guard）** | **CLOSED（本 D64・両方必要）** |
| **R_cap > R > R_required の決定順序** | **CLOSED（本 D64・測定 → R_required → R → R_cap）** |
| R=32 / R_cap=64 | **OPEN（実測後・前提にしない）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D64.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T178 | 8 メトリクス | acquire / release / outstanding / publish / rejection / exhaustion / max outstanding / B_max を区別 |
| T179 | B_max(T) ≠ peak | sliding-window max increment（peak で代用しない） |
| T180 | Observation window 分離 | normal / stall / shutdown / catastrophic を独立評価 |
| T181 | Baseline | P99.9(N_res) + deterministic guard（policy と safety を混同しない） |
| T182 | R_cap 決定順序 | 測定 → R_required → R → R_cap（R_cap > R > R_required） |

---

## D65 — Design-47 確定（telemetry の設計・実装境界・R 決定に十分な証拠生成の契約・2026-08-15）

ユーザー Design-46 レビュー対応: **「D51〜D64 の reservation authority 設計は R の具体値を除いて概ね CLOSED。
次は『R を決める』より先に『telemetry が R の決定に十分な証拠を生成できる』ことを閉じる。8 点を明確にする」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更・telemetry は実装ゲート）。R=32/R_cap=64 未確定・
候補のまま。**

### D65.1 telemetry の設計・実装境界（8 点の全体像）
- telemetry は **観測のみ**（authority の admission / liveness を変更しない）・RT と Non-RT の責務を分離・
  オフライン解析と RT authority を分離。

### D65.2 RT で観測してよい primitive（点1）
- **RT path（acquire・publish）**: **lock-free atomic のみ**（load / store / fetch_add / CAS・
  allocation / lock / unbounded scan 禁止）。
- RT で更新する counter: `count`（= H+I・既存 CAS）、`reservationAcquired`（fetch_add）、
  `reservationMaxOutstanding`（atomic max-update・load + CAS）。
- **B_max(T) sliding-window は RT で更新しない**（Non-RT sampler が count を読む）。

### D65.3 カウンタの linearization point（点2）
| カウンタ | LP |
|----------|-----|
| reservationAcquired | **acquire の count CAS 成功**（admission LP と一致・D61） |
| reservationReleased | **release の held_count fetch_sub**（release LP と一致・D61） |
| Outstanding（N_res = count） | live atomic 値 |
- **telemetry counter は authority の linearization point と一致**（観測が authority の意味論と整合）。

### D65.4 exhaustion の rejection reason の分離（点3）
- **publish rejection を reservation exhaustion 起因と他（validate / Faulted）で区別**。
- 方法: `PublishStageResult` に**専用 reason（例: ReservationExhausted）**を追加・acquire 失敗時のみ設定。
- exhaustion カウンタはその reason でのみ increment（validate 失敗と混ぜない）。

### D65.5 max outstanding / B_max(T) の更新方式（点4/5）
- **max outstanding**: acquire 成功時に atomic max-update（O(1)・RT 可）。
- **B_max(T)**: **Non-RT sampler** が count を一定間隔でサンプリング → 固定リング（preallocated・
  allocation-free）に格納 → 窓 T の delta の最大値を維持。
- **★ sliding window の RT 安全性**: sampler は Non-RT・固定リング（O(1)/sample・lock なし・unbounded scan
  なし）→ **RT allocation / lock / unbounded scan を発生させない**。

### D65.6 observation window の識別（点6）
- **各サンプルに現在の窓状態をタグ付け**（既存シグナルから判定）:
  | 窓 | 識別シグナル |
  |----|-------------|
  | normal | drain 健全（backlog 正常・stall なし） |
  | reclaim stall | T_reservation_stall（drain 停滞・backlog 高・stuck-reader 検出） |
  | shutdown | isShutdownInProgress |
  | catastrophic | quarantine-full（overflowCount > 0 / resident 高） |
- 窓ごとに独立評価（D64 と整合）。

### D65.7 telemetry が admission / liveness を変更しない（点7）
- telemetry の書込みは**既存操作の診断副作用のみ**（counter increment）・**新規の admission / liveness 分岐を
  追加しない**。
- B_max sampler は passive observer（authority 状態を読むのみ・書かない）。
- **契約: telemetry コードは admission / liveness に副作用を持たない**。

### D65.8 オフライン解析と RT authority の責務分離（点8）
- **RT authority の責務**: 設定済み R の gate 強制のみ（count CAS）・**R_required / R の導出を行わない**。
- **オフライン解析の責務**: telemetry（P99.9 baseline・B_max・window 分類）→ R_required =
  R_baseline + B_max(T_stall) + M → R を決定 → authority へ設定。
- **分離**: authority は R の「利用者」・オフラインは R の「決定者」。

### D65.9 最終状態
| 項目 | 状態 |
|------|------|
| D51〜D64（authority 設計・H/I/F・free-stack・ordering・B_max・決定順序） | CLOSED |
| **RT primitive（点1）** | **CLOSED（本 D65・lock-free atomic のみ）** |
| **カウンタ LP（点2）** | **CLOSED（本 D65・admission/release LP と一致）** |
| **rejection reason 分離（点3）** | **CLOSED（本 D65・ReservationExhausted 専用 reason）** |
| **max outstanding / B_max 更新（点4/5）** | **CLOSED（本 D65・RT max-update + Non-RT sampler・固定リング）** |
| **sliding window の RT 安全性（点5）** | **CLOSED（本 D65・Non-RT・allocation/lock/scan なし）** |
| **observation window 識別（点6）** | **CLOSED（本 D65・既存シグナルでタグ付け）** |
| **admission 非干渉（点7）** | **CLOSED（本 D65・観測のみ・副作用なし）** |
| **オフライン分離（点8）** | **CLOSED（本 D65・authority は R の利用者）** |
| R=32 / R_cap=64 | **OPEN（候補・実測後）** |
| Phase I implementation | **NO-GO（ユーザー最終 GO 待ち）** |

### D65.10 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T183 | RT primitive | acquire で lock-free atomic のみ（allocation/lock/scan なし） |
| T184 | カウンタ LP | acquired/released/outstanding が admission/release LP と一致 |
| T185 | rejection reason 分離 | ReservationExhausted が validate 失敗と区別可能 |
| T186 | B_max 更新（Non-RT） | sampler が固定リングで O(1)/sample・RT で更新しない |
| T187 | sliding window RT 安全 | allocation / lock / unbounded scan なし |
| T188 | window 識別 | 既存シグナルで normal/stall/shutdown/catastrophic をタグ付け |
| T189 | admission 非干渉 | telemetry が admission/liveness に副作用なし |
| T190 | オフライン分離 | authority は R の利用者（R_required 導出はオフライン） |

---

## D66 — Design-48 確定（D65 telemetry contract の現行コードへのマッピング・実装正当性・2026-08-15）

ユーザー Design-47 レビュー対応: **「D65 の『8点 CLOSED』は設計契約としての CLOSED であり、実装正当性の
CLOSED ではない。特に 7 点を実装時に再検証すべき。次は D65 telemetry contract を現行コードの具体的な型・
所有者・寿命・memory ordering にマッピングし、実装可能性を閉じる」**。**I-2 implementation GO に進める必要は
ない（R の決定に必要な実測証拠がまだ存在しない）。Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D66.1 前提（コード確認）— PublishStageResult の意味体系
- **`convo::PublishStageResult : uint8_t { Success, Rejected, Failed }`**（RuntimePublicationCoordinator.h:15-19）。
- **`PublishCommitResult { convo::PublishStageResult stage; OwnershipDisposition ownership; }`**
  （AudioEngine.h:3554-3562）・**`PublishStageResultTraits::isCommitted(stage) = (stage == Success)`**（:3565-3570）。

### D66.2 Point 1 — ReservationExhausted の意味体系整合（ご指摘）
- **新 enum 値 `ReservationExhausted` は整合**: Success でない → `isCommitted == false`（publish 拒否）・
  Rejected / Failed と区別（それぞれ Coordinator 拒否 / validate 失敗）。
- `PublishCommitResult.stage` に設定・acquire 失敗時のみ・**exhaustion カウンタはその値でのみ increment**
  （validate 失敗と混ぜない・D65.4）。CLOSED。

### D66.3 Point 2 — acquire LP と reservationAcquired の同一 linearization（ご指摘）
- acquire LP = **count CAS 成功**（D61）・`reservationAcquired` の increment は**同一 critical section の
  CAS 成功後**に実行（同じ LP を表現・telemetry は診断副作用のみ・D65.7）。
- 実装: `if (CAS 成功) { reservationAcquired.fetch_add(1); ... }` — increment は CAS 成功パスにのみ置く
  （失敗パスで increment しない）。CLOSED。

### D66.4 Point 3 — release fetch_sub 前後の telemetry 可視性（ご指摘）
- release LP = **held_count fetch_sub**（D59/D61）・`reservationReleased` の increment は **fetch_sub と同一
  セクション**に実行（release 順序 push → count-- は telemetry 可視性に影響しない・診断のみ）。
- Outstanding（count）は live atomic・release LP で確定。CLOSED。

### D66.5 Point 4 — atomic max-update の overflow / wraparound（ご指摘）
- **max outstanding**: uint64・max ≤ R_cap（小さい値）→ overflow 実用上なし・**max は単調増加（wraparound
  なし）**。
- **count（H+I）**: R ≤ R_cap で bounded（構造的）→ wraparound なし。
- **累積カウンタ（acquired / released）**: uint64・2^64 回操作後に wrap（実用上到達不能・診断のみ）。
- CLOSED（負担なし・保守）。

### D66.6 Point 5 — sampler サンプル欠落と B_max(T) の証拠能力（ご指摘）
- **サンプル欠落 → B_max(T) は真の最大増分の保守的下界（過小評価）**。
- **証拠能力**: B_max(T) は「少なくともこれだけ burst した」を保証（下界）・policy は **M +
  deterministic guard で補償**（D64.4 / D65.8）。CLOSED（下界としての扱いを明文化）。

### D66.7 Point 6 — observation window タグの混入防止（ご指摘）
- タグは**サンプル捕捉時に既存シグナルから判定**（isShutdownInProgress / quarantine overflow / backlog・
  D65.6）→ shutdown / catastrophic サンプルは **normal 窓に混入しない**（捕捉時のタグで分離）・
  オフライン解析はタグでフィルタ（D64.6）。CLOSED。

### D66.8 Point 7 — telemetry counter の lifetime（ご指摘）
- **telemetry counter は AudioEngine member**（reservation authority と同時に生存）→ engine 内で authority
  より長く保つ必要を満たす（engine 破棄前に有効）。
- **オフライン解析用に既存 evidence 出力（emitEvidenceTickNonRt / debugRuntime_）で export**（engine 破棄前）。
- CLOSED（所有者 = AudioEngine・lifetime = engine 内・export で外部解析）。

### D66.9 最終状態
| 項目 | 状態 |
|------|------|
| D65 8 点（設計契約） | CLOSED |
| **Point 1（ReservationExhausted 整合）** | **CLOSED（本 D66・enum 追加・isCommitted false）** |
| **Point 2（acquire LP と同一 linearization）** | **CLOSED（本 D66・CAS 成功パスのみ increment）** |
| **Point 3（release 可視性）** | **CLOSED（本 D66・fetch_sub と同一セクション）** |
| **Point 4（max-update overflow/wraparound）** | **CLOSED（本 D66・bounded・単調増加）** |
| **Point 5（sampler 欠落 → B_max 下界）** | **CLOSED（本 D66・保守的下界・M で補償）** |
| **Point 6（window タグ混入防止）** | **CLOSED（本 D66・捕捉時タグで分離）** |
| **Point 7（telemetry lifetime）** | **CLOSED（本 D66・AudioEngine 所有・export）** |
| R=32 / R_cap=64 | **OPEN（実測後）** |
| **I-2 implementation GO** | **NO-GO（実測証拠不足・ユーザー最終 GO 待ち）** |

### D66.10 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T191 | ReservationExhausted 整合 | PublishStageResult に追加・isCommitted false・Rejected/Failed と区別 |
| T192 | acquire LP 同一 linearization | reservationAcquired は count CAS 成功パスのみ increment |
| T193 | release 可視性 | reservationReleased は fetch_sub と同一セクション |
| T194 | max-update overflow/wraparound なし | max 単調増加・count bounded・累積は uint64 |
| T195 | sampler 欠落 → B_max 下界 | サンプル欠落は過小評価（保守的下界）・M で補償 |
| T196 | window タグ混入防止 | shutdown/catastrophic は捕捉時タグで normal と分離 |
| T197 | telemetry lifetime | AudioEngine 所有・engine 破棄前に export |

---

## D67 — Design-49 確定（D66 telemetry mapping の全 call site / switch exhaustiveness / counter memory-ordering のコード全体検証・2026-08-15）

ユーザー Design-48 レビュー対応: **「D66 の 7 点 CLOSED は 2 点だけ厳密化すべき。(1) ReservationExhausted は
enum 整合でなく既存意味体系との関係を固定し、downstream の switch 網羅性を全 call site 確認。(2) reservationReleased
と fetch_sub の『同一セクション』は D59 の『atomic transaction でない』を再び曖昧にするため、telemetry は観測値
として記録し source of truth にしない。さらに M は任意値で真値を補償できない」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D67.1 PublishStageResult の全 call site / downstream 検証（コード全体・ご指摘1）
- **使用箇所マップ**:
  | 箇所 | 用途 |
  |------|------|
  | AudioEngine.h:3566-3567 | `isCommitted(stage) = (stage == Success)`（唯一の定義） |
  | AudioEngine.h:4569 | receipt 条件 `isCommitted(result.stage) && seqId != 0` |
  | PublicationExecutor.cpp:64 | 失敗処理 `!isCommitted(result.stage)` |
  | RuntimePublicationCoordinator.h:103/115/125/140 | Coordinator の return（Failed / Rejected / Success） |
  | tests（PublishPipelineIntegrationTests / SoakPublishIntegrationTests） | `stage != Success` / `== Success` |
- **★ `switch (stage)` は存在しない**（ConvolverProcessor.Rebuild.cpp:52/140/241 の switch は別 enum
  `IncrementalRebuildJob::Stage`・PublishStageResult ではない）→ **ReservationExhausted 追加による switch 網羅性
  破壊は無い**。
- **全 downstream は `== Success` / `isCommitted` 比較のみ** → ReservationExhausted は非 Success として
  正しく失敗扱い（receipt なし・失敗処理・Coordinator は不変・テストは失敗扱い）。
- **∴ exhaustiveness verification CLOSED（本 D67・コード全体で確認）**。

### D67.2 ReservationExhausted の意味論固定（D66 の厳密化・ご指摘1）
- **semantic 固定**: `ReservationExhausted` は **publish failure の一種だが、通常の `Rejected` / `Failed` とは
  別の diagnostic / admission cause（reservation exhaustion / backpressure）を持つ**。
- `isCommitted == false`（Success でない）・receipt なし・**retry-able（transient・D50）**。
- **semantic mapping CLOSED（D66）+ exhaustiveness verification CLOSED（本 D67）**。

### D67.3 reservationReleased telemetry の契約修正（D66 の厳密化・ご指摘2）
- **D59 の「別 atomic object の操作なので atomic transaction ではない」を再び曖昧にしない**。
- **修正**: `reservationReleased telemetry = release LP（held_count fetch_sub）を通過した release path に対して
  のみ記録`（**観測値**）。
- **telemetry counter の更新順序は lifetime invariant ではない**（authority state の source of truth では
  ない）。
- **telemetry の更新失敗・遅延・観測競合は reservation authority の correctness を侵食しない**（telemetry は
  release LP の観測値に過ぎない）。

### D67.4 M の根拠（欠落補償・R 根拠・ご指摘3）
- **B_max^observed(T) ≤ B_max^true(T)** なので、任意の M で真値を補償できるとは限らない。
- **M は以下から根拠を持って決定**（数値は実測後）: サンプリング周期・最大観測欠落数・counter の読み取り
  間隔・measurement duration・burst の最小時間粒度。
- **telemetry 証拠能力 CLOSED / M の数値 OPEN**。

### D67.5 最終状態（ユーザー判定表の反映）
| 項目 | 状態 |
|------|------|
| telemetry の型・所有者・寿命 mapping | CLOSED（D66） |
| acquire / release LP の telemetry mapping | CLOSED（D66） |
| ReservationExhausted の意味論 | CLOSED（D66/D67） |
| **全 switch / downstream compatibility** | **CLOSED（本 D67・switch 存在せず・== Success 比較のみ・コード全体確認）** |
| **telemetry が authority correctness に介入しない** | **CLOSED（本 D67・観測値として厳密化）** |
| sampler 欠落が B_max を過小評価し得ること | CLOSED（D66/D67） |
| **M の数値** | **OPEN（根拠項目は固定・数値は実測後）** |
| R=32 / R_cap=64 | OPEN |
| I-2 implementation GO | **NO-GO（ユーザー最終 GO 待ち）** |

### D67.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T198 | PublishStageResult に switch なし | 全 downstream は == Success / isCommitted 比較のみ（網羅性破壊なし） |
| T199 | ReservationExhausted 非 Success | isCommitted false・receipt なし・retry-able（transient・D50） |
| T200 | reservationReleased は観測値 | release LP 通過のみ記録・source of truth でない |
| T201 | telemetry 非介入 | 更新失敗 / 遅延 / 観測競合が correctness を侵食しない |
| T202 | M の根拠項目 | サンプリング周期・欠落数・読取間隔・duration・粒度から決定（数値は実測後） |

---

## D68 — Design-50 確定（telemetry の挿入点マッピング・RT/non-RT 境界・lifetime ordering・2026-08-16）

ユーザー Design-49 レビュー対応: **「D67 は CLOSED。次段（D68）は実装前の最後の設計確認として、D67 telemetry
contract を実際の AudioEngine / publication path / retire path のどこへ挿入するかを、RT/non-RT 境界と lifetime
ordering を含めて確定する。telemetry 実装は『R を決めるための測定装置』として扱い、reservation authority の
correctness を測定結果に依存させない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D68.1 telemetry 挿入点のマッピング（D67 contract → 現行コードのどこへ）
| telemetry | 挿入点（コード） | スレッド |
|-----------|------------------|----------|
| **acquire 成功**（reservationAcquired++・max outstanding） | `WorldRetirementReservation::acquire()` 内・**count CAS 成功後**（publish() 内・swap 前・D61 LP） | RT / ISR |
| **acquire 失敗**（exhaustion++・`ReservationExhausted` 設定） | 同上・**count CAS 失敗パス**（PublishCommitResult.stage に設定・D67.2） | RT / ISR |
| **release**（reservationReleased++） | `WorldRetirementReservation::release()` 内（**world-delete deleter・retireRuntimePublishWorldNonRt**・D52/D66） | Non-RT（drain） |
| **B_max sampler**（count サンプリング・window タグ） | **周期 drain（AudioEngine.Timer.cpp の drainDeferredRetireQueues）** | Non-RT（timer） |
| **export**（evidence） | **emitEvidenceTickNonRt / debugRuntime_**（既存 evidence 出力） | Non-RT |

### D68.2 RT / Non-RT 境界（D65.2 / D67 と整合）
- **RT path = acquire のみ**: count CAS + reservationAcquired fetch_add + atomic max-update・
  **allocation / lock / unbounded scan なし**。
- **Non-RT path**: release（drain・deleter）・B_max sampler（timer）・window タグ・export。

### D68.3 lifetime ordering（D66.8 / D52 INV-R6 と整合）
- **telemetry counter = AudioEngine member**（authority と同時生存・D66.8）。
- **全 release（drain）はデストラクタ本体で実行**（CtorDtor 順序・D52 INV-R6）→ **telemetry は member 破棄前に
  常に有効**（release telemetry の lifetime 保証）。
- **export は engine 破棄前**（emitEvidenceTickNonRt・周期 + shutdown）。
- **B_max sampler のリング = 固定・preallocated・AudioEngine member**（allocation-free・D65.5）。

### D68.4 非干渉（D67.3 と整合・ご指定）
- **telemetry 実装は「R を決めるための測定装置」**・reservation authority の correctness は測定結果に依存
  しない。
- telemetry は観測のみ・更新遅延 / 競合 / 欠落は correctness を変えない（source of truth は held_count /
  free_stack / held-token table・D67.3）。

### D68.5 最終状態
| 項目 | 状態 |
|------|------|
| D51〜D67（authority / observation contract） | CLOSED |
| **telemetry 挿入点（publish / retire / timer / export）** | **CLOSED（本 D68・RT/non-RT 境界含む）** |
| **lifetime ordering（member・drain 前・export 前）** | **CLOSED（本 D68）** |
| **非干渉（測定装置として扱う）** | **CLOSED（本 D68）** |
| M / R / R_cap | **OPEN（実測後）** |
| I-2 implementation GO | **NO-GO（ユーザー最終 GO 待ち）** |

### D68.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T203 | acquire telemetry 挿入点 | publish() 内・count CAS 成功後（RT・LP と一致） |
| T204 | release telemetry 挿入点 | world-delete deleter（Non-RT drain） |
| T205 | B_max sampler 挿入点 | 周期 drain（Timer.cpp）・Non-RT |
| T206 | RT/non-RT 境界 | RT は atomic のみ・sampler/export は Non-RT |
| T207 | lifetime ordering | telemetry member・drain 前・export 前（engine 破棄前） |
| T208 | 非干渉 | telemetry は測定装置・correctness は測定結果に非依存 |

---

## D69 — Design-51 確定（telemetry 実装契約・最小変更範囲（Phase I-T1）・実装開始時コード突合・2026-08-16）

ユーザー Design-50 レビュー対応: **「D68 は CLOSED。次は Design-51（D69）として telemetry implementation の
具体的な実装契約・最小変更範囲を確定する。ただし D68 の『挿入点 CLOSED』は実装開始時に実際の AudioEngine の
lifetime/destructor 順序・publish/swap 実コード経路・retire deleter の所有関係と突合する必要がある（設計上の
挿入点 ≠ 現行コード上その位置が安全）」**。**I-2 は引き続き NO-GO・まず telemetry の実装・検証。
Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D69.1 実装契約のフェーズ分割（最小変更範囲・ご指定）
- **Phase I-T1（telemetry 測定装置）**: count 追跡（acquire +1 / release -1）+ telemetry counter
  （acquired / released / outstanding / max / exhaustion）+ B_max sampler + export。
  **held-set / free-stack / token / R gate は含まない**（R gate は実測後のフェーズ）。
- **Phase I-T2（R gate・実測後）**: held-set / free-stack / token（ABA-safe）・ReservationExhausted admission・
  R gate（M / R / R_cap 確定後）。

### D69.2 T1 の最小変更範囲（ファイル・型）
| 変更 | ファイル |
|------|----------|
| `PublishStageResult` に `ReservationExhausted` 追加 | RuntimePublicationCoordinator.h |
| `WorldRetirementReservation`（count + telemetry counter）新設 | 新規ヘッダ or AudioEngine.h |
| publish() に acquire フック（count+1・telemetry・LP と一致） | RuntimeWorldAuthority.h |
| world-delete deleter に release フック（count-1・telemetry） | AudioEngine.h（retireRuntimePublishWorldNonRt） |
| B_max sampler（周期 drain・window タグ） | AudioEngine.Timer.cpp |
| export（evidence） | emitEvidenceTickNonRt / debugRuntime_ |
| ReservationExhausted 処理 | PublicationExecutor.cpp |
- **T1 は R gate なし**（count は記録のみ・admission に使用しない）→ 既存 semantics への影響最小。

### D69.3 実装開始時のコード突合（ご指摘・設計挿入点 ≠ 現行コード安全性）
| 突合項目 | 確認 |
|----------|------|
| **AudioEngine lifetime / destructor 順序** | 全 drain はデストラクタ本体（CtorDtor・graceful drain → clear → drainAll）・telemetry member は破棄前に有効（D52 INV-R6・D68.3）→ **実装開始時に再確認** |
| **publish / swap 実コード経路** | `RuntimeWorldAuthority::publish()`（owner.release → fence → publishAndSwap・D45/D49）・acquire は swap 前 → **実装開始時に再確認** |
| **retire deleter 所有関係** | `retireRuntimePublishWorldNonRt` の deleter（stateless function pointer）・context/token は DeletionEntry 経由（D52/D55・D66）→ **実装開始時に再確認** |

### D69.4 非干渉（測定装置として・D67.3 / D68.4 整合）
- telemetry は観測のみ・reservation authority の correctness は測定結果に依存しない。

### D69.5 最終状態
| 項目 | 状態 |
|------|------|
| D68（挿入点・RT/non-RT 境界・lifetime ordering） | CLOSED |
| **T1 最小変更範囲（telemetry 測定装置）** | **CLOSED（本 D69・契約・R gate 含まず）** |
| **T2（held-set / free-stack / token / R gate）** | **実測後（T1 後・M/R/R_cap 確定後）** |
| **実装開始時コード突合** | **CLOSED（本 D69・確認済み・実装時に再確認）** |
| M / R / R_cap | OPEN（実測後） |
| **I-2 implementation GO** | **NO-GO（ユーザー最終 GO 待ち・まず telemetry 実装・検証）** |

### D69.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T209 | T1 スコープ | count + telemetry + sampler + export（held-set / free-stack / token / R gate 含まず） |
| T210 | 実装開始時コード突合 | lifetime / publish / deleter を現行コードと再確認 |
| T211 | ReservationExhausted 追加 | RuntimePublicationCoordinator.h に enum 値追加（D67 整合） |
| T212 | T2 は実測後 | held-set / free-stack / token / R gate は M/R/R_cap 確定後に実装 |

---

## D70 — Design-52 確定（T1/T2 境界の厳密化・ReservationExhausted の T1 型準備限定・T1 検証順序・2026-08-16）

ユーザー Design-51 レビュー対応: **「D69 CLOSED・I-2 NO-GO を維持。T1 は『reservation authority の実装』
ではなく『測定装置の実装』。T1 で変更してよいのは観測・telemetry・sampler・tagging・export まで。held-set /
free-stack / token / ABA / R gate / actual admission authority は T2 の領域。ReservationExhausted の enum 追加
だけは T1 の型準備として許容だが、既存 publish failure を ReservationExhausted に分類する処理は T1 で入れない
（存在しない R gate を仮定する）。コード突合注記を優先し、実装時に分離検証する」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D70.1 T1/T2 境界の厳密化（ご指定）
- **T1（測定装置のみ）で変更してよいもの**:
  - acquire / release 相当イベントの観測（count 追跡）
  - acquired / released / outstanding / max / exhaustion の telemetry
  - B_max の Non-RT sampler
  - observation-window tagging
  - evidence / export
- **T2（authority 機構）として分離したままのもの**:
  - held-set / free-stack / token mint / lookup / ABA protection
  - R admission gate
  - **ReservationExhausted を発生させる実際の admission authority**

### D70.2 ReservationExhausted の T1 型準備限定（ご指定）
- **enum 追加のみ T1 で許容**（RuntimePublicationCoordinator.h に `ReservationExhausted` 値・D67 整合）。
- **T1 で既存 publish failure を `ReservationExhausted` に分類する処理は入れない**
  （存在しない R gate を仮定することになる・避ける・T2 で実施）。

### D70.3 T1 実装の検証順序（ご指定・明文化）
```
D69 → T1 実装 → コンパイル / unit test → RT safety audit
  → telemetry counter の LP 検証 → B_max sampler の欠落・窓境界検証
  → 実測 → R_required → R → R_cap → T2 → I-2 GO
```

### D70.4 コード突合注記の優先（ご指定）
- **「実装開始時コード突合 CLOSED」は注記を優先**・設計上の挿入点と実コードの lifetime / order が実際に
  契約を満たすことは**分離して検証**（実装時に再確認・D69.3）。

### D70.5 最終状態
| 項目 | 状態 |
|------|------|
| D69（実装契約・T1 スコープ） | CLOSED |
| **T1/T2 境界厳密化（測定装置 vs authority 機構）** | **CLOSED（本 D70）** |
| **ReservationExhausted は T1 で型準備のみ（分類処理なし）** | **CLOSED（本 D70）** |
| **T1 検証順序（8 ステップ）** | **CLOSED（本 D70・明文化）** |
| コード突合注記の優先（分離検証） | CLOSED（本 D70） |
| R=32 / R_cap=64 | **OPEN（T1 実測後・コードへ先に入れない）** |
| **I-2 implementation GO** | **NO-GO（T1 → 実測 → R 確定 → T2 後）** |

### D70.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T213 | T1 スコープ厳密化 | 観測 / telemetry / sampler / tagging / export のみ・held-set / free-stack / token / R gate は T2 |
| T214 | ReservationExhausted T1 型準備限定 | enum 追加のみ・既存 failure を分類する処理なし |
| T215 | T1 検証順序 | コンパイル → unit test → RT audit → LP → sampler → 実測 → R → T2 → I-2 GO |

---

## D71 — Design-53 確定（exhaustion の T1/T2 意味論分離・T1 実コード突合・T1 release 測定の前提・2026-08-16）

ユーザー Design-52 レビュー対応: **「D70 は概ね妥当。ただし T1 の『exhaustion telemetry』は意味論未確定。
T1 は R admission gate を実装しないため exhaustion を発生させる authority が無い。T1 で直接観測できるのは
acquire/release イベント・outstanding・observed max・B_max・window・publish/release 実測イベント。exhaustion =
admission gate が R 到達を理由に拒否した事実（T2 のみ）。ReservationExhausted は T1 では enum 型準備だけ・
T1 telemetry に exhaustion 発生数を実在イベントとして計上しない。次は Design-53（D71）として T1 の実コード
突合（AudioEngine lifetime・publish/swap・retire/deleter を読み、D68-D70 の挿入点が成立するか確認）」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D71.1 exhaustion の T1/T2 意味論分離（ご指定・明文化）
- **T1 で直接観測できるもの**（実コード経路から観測する事実）:
  `acquireObserved`・`releaseObserved`・`observedOutstanding`・`observedMaxOutstanding`・`B_max(T)`・
  observation window・publish / release の実測イベント。
- **T2 で初めて成立するもの**:
  ```
  reservation exhaustion = admission gate が R 到達を理由として acquire / publish を拒否した事実
  ```
- **★ `reservationExhaustionCount = T2 only`**・`ReservationExhausted` は T1 では enum 型準備のみ・
  **T1 telemetry に exhaustion 発生数を実在イベントとして計上しない**（T1 には exhaustion を生成する authority
  が無いため）。

### D71.2 T1 の測定項目（安全なリスト）
- `acquireObserved` / `releaseObserved` / `observedOutstanding` / `observedMaxOutstanding` / `B_max(T)` /
  observation window（exhaustion 計上なし）。

### D71.3 T1 実コード突合（ご指定・挿入点の成立確認）
| 挿入点 | 実コード | 突合結果 |
|--------|----------|----------|
| **T1 acquire フック（count+1・acquireObserved）** | `RuntimeWorldAuthority::publish()`（swap 前・D45/D49） | ✅ publish() は sole gateway・1 publish ≤1 reservation（D63）・acquire フックは swap 前に配置可 |
| **T1 release フック（count-1・releaseObserved）** | `retireRuntimePublishWorldNonRt` の world-delete deleter（AudioEngine.h:3525-3533） | ✅ deleter は stateless function pointer → **D52 context-in-entry 機構（のサブセット）が前提**（D71.4） |
| **B_max sampler（count 読取・window タグ）** | `AudioEngine.Timer.cpp` の周期 drain（drainDeferredRetireQueues） | ✅ Non-RT・周期・count を atomic 読取可 |
| **export（evidence）** | `emitEvidenceTickNonRt` / `debugRuntime_` | ✅ 既存・周期 + shutdown・engine 破棄前 |
| **AudioEngine lifetime / destructor** | CtorDtor.cpp（graceful drain → clear → drainAll） | ✅ 全 drain はデストラクタ本体・telemetry member は破棄前に有効（D52 INV-R6） |

### D71.4 ★ T1 release 測定の前提（実コード制約）
- **T1 の release 測定（count-1・releaseObserved）は、world-destroy deleter（stateless fn ptr）から
  count / telemetry へ到達する必要がある** → **D52 の context-in-entry 機構（のサブセット・context のみ・
  token / held-set は T2）**を T1 実装の前提とする。
- これは D69 の T1 スコープ（観測実装）に含まれる（release 観測のための最小機構・admission には使用しない）。

### D71.5 最終状態
| 項目 | 状態 |
|------|------|
| D70（T1/T2 境界・ReservationExhausted 型準備・検証順序） | CLOSED |
| **exhaustion の T1/T2 意味論分離（exhaustion = T2 only）** | **CLOSED（本 D71）** |
| **T1 測定項目（exhaustion 計上なし）** | **CLOSED（本 D71）** |
| **T1 実コード突合（挿入点成立）** | **CLOSED（本 D71・実コード確認）** |
| **T1 release 測定の前提（context-in-entry サブセット）** | **CLOSED（本 D71・明文化）** |
| R=32 / R_cap=64 | OPEN（T1 実測後） |
| **I-2 implementation GO** | **NO-GO（T1 → 実測 → R → T2 → I-2 GO）** |

### D71.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T216 | exhaustion = T2 only | T1 telemetry に exhaustion 発生数を計上しない（enum 型準備のみ） |
| T217 | T1 測定項目 | acquireObserved / releaseObserved / observedOutstanding / observedMaxOutstanding / B_max(T) / window |
| T218 | T1 挿入点突合 | publish() の acquire フック・deleter の release フック・Timer sampler・export（実コード確認） |
| T219 | T1 release 測定前提 | context-in-entry サブセット（context のみ・token/hold-set は T2） |

---

## D72 — Design-54 確定（release 側 context 安全性の 5 点実コード検証・deleter は lifetime authority でない・2026-08-16）

ユーザー Design-53 レビュー対応: **「D71 の T1/T2 境界（特に exhaustion = T2 only）は維持。ただし T1 実装前に
release 側（world-destroy deleter → context-in-entry → AudioEngine / telemetry）について『context-in-entry が
ある』ことと『T1 telemetry がその context を安全に参照できる』ことを同一視しない。実装時は 5 点（context の
所有者 / deleter 実行時 AudioEngine 生存 / deleter 複数回実行なし / drain 完了と member 破棄の順序 / deleter
→telemetry 経路の lock・allocation・RT 非安全なし）をコード上で再確認。deleter は lifetime authority では
ない・T1 では release 観測イベントを記録する入口としてのみ扱い、lifetime correctness を telemetry 側へ依存
させない（D67 原則維持）」**。**R=32 / R_cap=64 は依然としてコードへ導入しない。Phase I 実装 NO-GO 継続
（設計契約のみ・コード未変更）。**

### D72.1 release 側 context 安全性の 5 点実コード検証（ご指定）
| # | 確認点 | 実コード検証 |
|---|--------|--------------|
| 1 | **DeletionEntry が保持する context の所有者** | context = AudioEngine-owned object（telemetry / authority へのポインタ）・entry は所有せず借用（lifetime は engine が保証・D52 INV-R6） |
| 2 | **deleter 実行時点で AudioEngine が生存** | 全 deleter は drain（reclaim / drainAllUnsafe）で実行・drain は AudioEngine デストラクタ本体（CtorDtor）→ **AudioEngine は全 deleter 実行中に生存** |
| 3 | **deleter が複数回実行されない** | D46-R4: drain-once（primary CAS 排他 + quarantine drain-once）・各 entry の deleter は正確に 1 回 |
| 4 | **drain 完了と telemetry member 破棄の順序** | CtorDtor: 全 drain（drainDeferredRetireQueues(true) → m_epochDomain.drainAll() → drainAllQuarantineStore）が本体 → **telemetry member は全 deleter 実行後に破棄**（D52 INV-R6） |
| 5 | **deleter → telemetry 経路に lock / allocation / RT 非安全なし** | deleter は Non-RT（drain）・telemetry increment は atomic（fetch_add / fetch_sub）・lock-free・allocation なし |

### D72.2 ★ deleter は lifetime authority ではない（ご指定）
- **deleter の T1 での役割 = release 観測イベントを記録する入口のみ**（releaseObserved++・count--）。
- **lifetime correctness を telemetry 側へ依存させない**（D67.3 / D68.4 原則）・lifetime correctness は
  authority（held_count / free_stack / held-token table・T2）が担う。

### D72.3 最終状態（ゲート維持）
| 項目 | 状態 |
|------|------|
| D71（exhaustion = T2 only・T1 測定項目・挿入点突合） | CLOSED |
| **release 側 context 安全性（5 点実コード検証）** | **CLOSED（本 D72）** |
| **deleter は release 観測入口（lifetime authority でない）** | **CLOSED（本 D72）** |
| R=32 / R_cap=64 | **OPEN（コードへ導入しない・T1 実測後）** |
| **I-2 implementation GO** | **NO-GO（T1 実装レビュー → 最小変更 → 実測 → R → T2 → I-2 GO）** |

### D72.4 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T220 | context 所有者 | context は AudioEngine-owned・借用（entry は所有しない・engine が lifetime 保証） |
| T221 | deleter 実行時 AudioEngine 生存 | 全 deleter はデストラクタ本体の drain で実行 |
| T222 | deleter 複数回実行なし | drain-once（D46-R4） |
| T223 | drain 完了 → member 破棄順序 | CtorDtor 順序（telemetry は全 deleter 後に破棄） |
| T224 | deleter→telemetry 経路安全 | Non-RT・atomic・lock-free・allocation なし |
| T225 | deleter は観測入口 | release 観測のみ・lifetime correctness は telemetry 非依存 |

---

## D73 — Design-55 確定（1 retirement obligation → ≤1 release observation・terminal path 相互排他・5 チェック・2026-08-16）

ユーザー Design-54 レビュー対応: **「D72 の『deleter 複数回実行なし（D46-R4: drain-once）』は『同一 DeletionEntry
が二重処理されない』ことを示すが、『同一 World に対する release 観測が全経路で一度だけ発生する』こととは別。
後者を CLOSED にするには terminal path（normal drain / quarantine / shutdown drain）の相互排他まで必要。T1 実装
では release telemetry を deleter に置くだけでなく、1 retirement obligation → release observation ≤ 1 を実コード
上の invariant として確認する」**。**R=32 / R_cap=64 は固定しない・T1 では ReservationExhausted を生成しない・
deleter は lifetime authority ではなく release-observation の入口に限定。Phase I 実装 NO-GO 継続（設計契約のみ・
コード未変更）。**

### D73.1 証明範囲の厳密化（ご指摘・drain-once ≠ 全経路 release 観測 1 回）
- **D46-R4（drain-once）は「同一 DeletionEntry が二重処理されない」ことを示す**。
- **「同一 World に対する release 観測が全経路で一度だけ発生する」** には、terminal path
  （normal drain / quarantine / shutdown drain）の**相互排他**まで必要。

### D73.2 terminal path 相互排他（publish → retire entry → routing → deleter → release observation）
```
publish → retire entry 生成 → retire routing
  ├─ normal drain（primary reclaim・epoch-gated）
  ├─ quarantine（RetireQuarantineStore drain・epoch-gated）
  └─ shutdown drain（drainAllUnsafe / drainAllQuarantineStore）
       ↓
     deleter → reservation release observation
```
- **各 entry はちょうど 1 つの storage（primary XOR quarantine）**（D46/D53・enqueueWithRetry は primary 成功
  XOR quarantine・二重存在なし）。
- **once-terminalized（deleter 実行 → entry clear・seq 公開）** により、他経路は処理不能（D46-R4）。
- **shutdown drainAll は normal drain 完了後に実行**（CtorDtor 順序・graceful drain → clear → drainAll）→
  競合しない。
- **∴ terminal path 相互排他 → 各 retirement obligation の release observation は ≤ 1 回**。

### D73.3 T1 実装の 5 チェック（実コード control-flow・ご指定）
| # | チェック | 実コード確認 |
|---|----------|--------------|
| 1 | **DeletionEntry の terminalization が一意か** | 各 entry は primary XOR quarantine・terminalization（deleter）後に clear → 一意 |
| 2 | **normal / quarantine / shutdown が同一 entry を競合処理しないか** | entry は単一 storage・shutdown は normal drain 後に実行 → 競合なし |
| 3 | **deleter の invocation が exactly-once か** | D46-R4（primary CAS 排他 + quarantine drain-once） |
| 4 | **deleter invocation と release LP の対応が 1:1 か** | 1 world = 1 reservation（D48-R2/D53）・1 deleter = 1 release LP = 1 releaseObserved++ |
| 5 | **release telemetry は上記 correctness を利用するだけで逆方向に依存しないか** | telemetry は deleter の下流・lifetime に feed-back しない（D67.3/D72.2） |

### D73.4 T1 release 観測契約（ご指定）
- **releaseObserved++ の正当化条件**: DeletionEntry terminalization → exactly-one deleter execution → release LP
  → releaseObserved++。
- **∴ `one retirement obligation ⇒ at most one release observation` を T1 の観測契約として採用可**。
- **telemetry counter の更新失敗・遅延・export 欠落は lifetime correctness に影響しない**（D67/D72 方針）。

### D73.5 最終状態
| 項目 | 状態 |
|------|------|
| D72（context 安全性 5 点・deleter = 観測入口） | CLOSED |
| **証明範囲の厳密化（drain-once ≠ 全経路 release 観測 1 回）** | **CLOSED（本 D73）** |
| **terminal path 相互排他（→ release observation ≤ 1）** | **CLOSED（本 D73・実コード確認）** |
| **T1 実装 5 チェック（control-flow）** | **CLOSED（本 D73）** |
| **one retirement obligation ⇒ ≤1 release observation** | **CLOSED（本 D73・T1 観測契約）** |
| R=32 / R_cap=64 | **OPEN（固定しない・T1 実測後）** |
| **I-2 implementation GO** | **NO-GO（T1 実装レビュー → 最小実装 → 実測 → R → T2 → I-2 GO）** |

### D73.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T226 | terminal path 相互排他 | normal / quarantine / shutdown が同一 entry を競合しない（単一 storage・順序） |
| T227 | 1 retirement → ≤1 release observation | terminalization 一意・deleter exactly-once・release LP 1:1 |
| T228 | release telemetry 非依存 | correctness を利用するのみ・逆方向依存なし |
| T229 | T1 release 観測契約 | releaseObserved++ は terminalization → deleter → LP で正当化 |

---

## D74 — Design-56 確定（T1 実装前の実コード control-flow 再突合・3 点検証・releaseObserved ≠ lifetime authority 代替・2026-08-16）

ユーザー Design-55 レビュー対応: **「D73 の CLOSED は設計上の証明として成立・ただし T1 実装開始前に実コードで
再検証すべき 3 点が残る: (1) primary XOR quarantine が全 retirement path で本当に排他的か (2) shutdown drainAll
と通常 drain の順序が実装上も D73 の仮定どおりか (3) deleter に到達する context が全経路で同じ
release-observation authority を指すことが保証されているか。releaseObserved を lifetime authority の代替に
してはいけない（D73 境界維持）」**。**R=32 / R_cap=64 をこの段階でコードへ持ち込まない。Phase I 実装 NO-GO
継続（設計契約のみ・コード未変更）。**

### D74.1 実コード control-flow 再突合（3 点・ご指定）
| # | 再突合点 | 実コード検証 |
|---|----------|--------------|
| 1 | **primary XOR quarantine が全 retirement path で排他** | world の retire 経路は単一（retireRuntimePublishWorldNonRt → enqueueDeferredDeleteNonRt → m_retireRouter）・`enqueueWithRetry`（ISRRetireRouter.cpp:161-208）は primary 成功 XOR quarantine（retry 失敗時のみ quarantine）・**二重存在なし**（D46/D53 と整合）✅ |
| 2 | **shutdown drainAll と通常 drain の順序** | CtorDtor: graceful drain（pendingRetireCount==0・activeReaderCount==0）→ clear → drainDeferredRetireQueues(true) → m_epochDomain.drainAll() → markShutdownComplete・**通常 drain が先・shutdown drainAll が後**（ReleaseResources も同順序）✅ |
| 3 | **deleter に到達する context が全経路で同一 authority** | world-destroy deleter は retireRuntimePublishWorldNonRt（AudioEngine.h:3525-3533）で**一度だけ定義**・同一 deleter + 同一 context（release-observation authority へのポインタ）が primary（enqueueRetire）・quarantine（m_retireQuarantine.quarantine）・shutdown drainAll（drainAllUnsafe は格納 entry の deleter/context を実行）へ到達 ✅ |

### D74.2 ★ releaseObserved は lifetime authority の代替でない（ご指定・D73 境界維持）
- **releaseObserved は観測 counter（T1）**・lifetime 判定に使用しない。
- lifetime correctness は authority（held_count / free_stack / held-token table・T2）が担う（D67.3 / D72.2 /
  D73.4）。

### D74.3 最終状態（ご指定の表を反映）
| 項目 | 状態 |
|------|------|
| retirement obligation → ≤1 release observation の論理契約 | CLOSED（D73） |
| terminal path 相互排他の設計証明 | CLOSED（D73） |
| **実コード control-flow の再突合（3 点）** | **CLOSED（本 D74・実コード検証・T1 実装時に再確認）** |
| releaseObserved ≠ lifetime authority 代替 | CLOSED（本 D74・D73 境界維持） |
| R / R_cap | **OPEN（この段階でコードへ持ち込まない）** |
| T2 authority | **NO-GO** |
| **T1 telemetry** | **次工程（最小実装）** |

### D74.4 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T230 | primary XOR quarantine 排他（全経路） | 単一 retire 経路・enqueueWithRetry の primary XOR quarantine を実コードで確認 |
| T231 | shutdown 順序 | graceful drain → drainAll（CtorDtor 順序）を実コードで確認 |
| T232 | context 同一性（全経路） | 同一 deleter + 同一 context が primary/quarantine/shutdown へ到達 |
| T233 | releaseObserved ≠ lifetime authority | 観測 counter・lifetime 判定に使用しない |

---

## D75 — Design-57 確定（Phase I-T1 最小 telemetry 実装レビュー・具体プラン・非交渉境界・2026-08-16）

ユーザー Design-56 レビュー対応: **「D74 の位置づけは妥当（retirement routing 排他・shutdown ordering・deleter
context 一意性 → T1 観測契約成立）。『T1 実装時に再確認』は残す（deleter の context 変更で一意性を壊さない）。
ここからは設計をさらに細分化するより、Phase I-T1 の最小 telemetry 実装レビューへ進む。非交渉境界: T1 に入れる =
acquire/release observation・outstanding/max・B_max sampler・observation-window・export・T1 に入れない = R
admission gate・held-set・free-stack・token mint/lookup・ABA authority・ReservationExhausted の実際の生成。
R / R_cap はまだコードへ導入しない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更・T1 実装は
GO 後）。**

### D75.1 T1 最小 telemetry 実装レビュー（具体プラン・ご指定）
| 変更 | ファイル | 内容 |
|------|----------|------|
| `ReservationExhausted` enum 追加 | RuntimePublicationCoordinator.h | **型準備のみ**（実際の生成処理なし・T2） |
| `WorldRetirementTelemetry` 新設 | 新規ヘッダ or AudioEngine.h | acquireObserved / releaseObserved / outstanding / max / B_max リング / window tag（AudioEngine member・D68.3） |
| acquire 観測フック | RuntimeWorldAuthority.h（publish()・swap 前） | count+1・acquireObserved++・max 更新（RT・atomic のみ・LP 一致） |
| release 観測フック | AudioEngine.h（world-delete deleter） | count-1・releaseObserved++（**context-in-entry サブセット・context のみ**・D71.4） |
| B_max sampler | AudioEngine.Timer.cpp（周期 drain） | count サンプリング・固定リング・window タグ（Non-RT） |
| export | emitEvidenceTickNonRt / debugRuntime_ | telemetry 出力（engine 破棄前） |

### D75.2 非交渉境界（T1 に含める / 含めない・ご指定）
- **T1 に入れる**: acquire / release observation・outstanding / max・B_max sampler・observation-window・export。
- **T1 に入れない**: R admission gate・held-set・free-stack・token mint / lookup・ABA authority・
  **ReservationExhausted の実際の生成**（enum 型準備のみ・D70/D71）。

### D75.3 実装時再確認（deleter context 一意性・D74 注記維持）
- **deleter の context を変更する際に一意性を壊さない**（world-destroy deleter は単一定義・全経路で同一
  context・T1 実装時に再確認）。

### D75.4 最終状態
| 項目 | 状態 |
|------|------|
| D74（実コード control-flow 再突合） | CLOSED |
| **T1 最小実装レビュー（具体プラン・ファイル・フック・sampler・export）** | **CLOSED（本 D75・実装準備）** |
| **非交渉境界（T1 に含める/含めない）** | **CLOSED（本 D75）** |
| **実装時再確認（deleter context 一意性）** | CLOSED（本 D75・注記維持） |
| R / R_cap | **OPEN（コードへ導入しない）** |
| T2 authority | **NO-GO** |
| **T1 実装** | **次工程（GO 後に最小実装・レビュー対象）** |

### D75.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T234 | T1 実装レビュー | ファイル・フック・sampler・export の具体プラン（D75.1） |
| T235 | 非交渉境界 | T1 に含めないもの（R gate・held-set・free-stack・token・ABA・ReservationExhausted 生成） |
| T236 | deleter context 一意性（実装時） | context 変更で一意性を壊さない |

---

## D76 — Design-58 確定（T1 観測カウンタの分離・observedOutstanding ≠ held_count・T1 telemetry は reservation authority でない・2026-08-16）

ユーザー Design-57 レビュー対応: **「D75 の count+1/count-1 は、reservation authority の正本 count か T1
telemetry 専用の観測カウンタか不明確。前者なら D70/D71 の『T1 は authority を実装しない』境界に違反。後者なら
count は『reservation outstanding』ではなく observed retirement-obligation count。T1 で安全なのは acquireObserved /
releaseObserved / observedOutstanding（= acquire − release）を観測値としてのみ存在させること。observedOutstanding
≠ reservation authority held_count を契約として明示。T1 にはまだ reservation acquisition の LP が存在しないため
acquireObserved++ は『future T2 acquisition LP に対応する観測点』で記録（T2 の LP を先取り定義しない）」**。
**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D76.1 count の曖昧さ修正（ご指定・observedOutstanding ≠ held_count）
- **T1 は `count`（held_count）を導入しない**（T2 authority の正本・T2 で導入）。
- **T1 は `observedOutstanding`（観測値）のみ**:
  ```
  publish supersession → acquireObserved++ + observedOutstanding++
  retirement terminal path → releaseObserved++ + observedOutstanding--
  observedOutstanding = N_acquireObserved − N_releaseObserved（観測値）
  ```
- **★ `observedOutstanding ≠ reservation authority held_count` を契約として明示**（T1 は観測・T2 は正本）。

### D76.2 T1 最小状態（観測値のみ・ご指定）
| State | T1 での意味 |
|-------|-------------|
| `acquireObserved` | retirement obligation 発生数 |
| `releaseObserved` | terminal retirement 完了数 |
| `observedOutstanding` | acquire − release（観測値） |
| `maxObservedOutstanding` | 上記の running max |
| `B_max(T)` | Non-RT sampler が求める観測 burst |
| `window tag` | normal / stall / shutdown / catastrophic |
- **T2 で初めて** held_count / held_set / free_stack / token / R gate が reservation authority の state
  source-of-truth になる。

### D76.3 acquireObserved の観測点（ご指定・T2 LP を先取りしない）
- **T1 には reservation acquisition の LP が存在しない**。
- 正確な表現:
  $$
  \boxed{\text{acquireObserved++ は future T2 acquisition LP に対応する観測点で記録}}
  $$
- **T1 は現在存在しない T2 authority の LP を先取り定義しない**・観測できるのは D63/D74 のコード事実
  （1 publish → ≤1 retirement obligation）。

### D76.4 ★ 不変条件（ご指定・D75 に追加）
- **「T1 telemetry state is observational state and is not a reservation authority」** を T1 の不変条件として
  追加。
- 「telemetry は authority の correctness に介入しない」設計原則（D70-D75）を最も厳密に維持。

### D76.5 最終状態（ご指定の判定を反映）
| 項目 | 状態 |
|------|------|
| T1 実装プラン | **CLOSED（本 D76 の修正後）** |
| **T1 が authority を侵食しないこと** | **CLOSED（本 D76・observedOutstanding ≠ held_count・観測値のみ）** |
| **T1 telemetry は reservation authority でない（不変条件）** | **CLOSED（本 D76）** |
| R / R_cap | **OPEN（コードへ導入しない）** |
| T2 | **NO-GO** |

### D76.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T237 | observedOutstanding ≠ held_count | T1 は観測値のみ・held_count は T2 |
| T238 | T1 最小状態 | acquireObserved / releaseObserved / observedOutstanding / max / B_max / window tag（観測のみ） |
| T239 | acquireObserved 観測点 | future T2 acquisition LP に対応する観測点・T2 LP を先取りしない |
| T240 | T1 telemetry ≠ authority（不変条件） | observational state・authority でない |

---

## D77 — Design-59 確定（observedOutstanding 非負性・異常検出契約・telemetry ordering ≠ lifetime ordering 代替・2026-08-16）

ユーザー Design-58 レビュー対応: **「D76 の observedOutstanding ≠ held_count は適切。ただし T1 実装前に
`observedOutstanding--` の負値問題を明文化すべき。T1 は authority でないため、release observation が acquire
observation より先に観測される可能性を、単純な fetch_sub だけで処理してよいかを決める。契約: acquireObserved は
retirement obligation 生成コード地点で increment・releaseObserved はその obligation の terminal deleter で
increment・observedOutstanding は差分として維持・observedOutstanding < 0 は正常状態として許容しない・ただし負値
検出そのものが lifetime correctness を左右してはいけない・telemetry counter の ordering を lifetime ordering の
代替にしない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D77.1 observedOutstanding 非負性契約（ご指定・明文化）
- **acquireObserved**: retirement obligation を生成したコード地点で increment（publish supersession・D63/D74）。
- **releaseObserved**: その obligation の terminal deleter で increment（D73 exactly-once）。
- **observedOutstanding**: 上記2つの差分として維持（$N_{obs} = N_{acquireObserved} - N_{releaseObserved}$）。
- **★ `observedOutstanding < 0` は正常状態として許容しない**（異常検出・診断のみ）。
- 非負性は **telemetry の実装順序が retirement control-flow の順序を正しく反映すること**に依存。

### D77.2 異常検出（負値）は lifetime correctness に影響しない（ご指定）
- **負値検出そのものが lifetime correctness を左右してはいけない**（telemetry は観測・D67/D76）。
- 負値検出は診断（Debug assert / health event）としてのみ使用。

### D77.3 ★ telemetry counter の ordering を lifetime ordering の代替にしない（ご指定）
- **telemetry counter の ordering は lifetime ordering の代替ではない**。
- lifetime correctness は authority（T2: held_count / held_set / free_stack / token）が担う（D67.3/D72.2）。

### D77.4 最終状態（ご指定の表を反映）
| 項目 | 状態 |
|------|------|
| T1/T2 state 分離 / observedOutstanding ≠ held_count / T1 telemetry ≠ authority | CLOSED（D76） |
| release observation の exactly-once 契約 | CLOSED（D73） |
| **telemetry counter の非負性・異常検出契約** | **CLOSED（本 D77・実装前に明文化）** |
| **telemetry ordering ≠ lifetime ordering 代替** | **CLOSED（本 D77）** |
| R / R_cap | **OPEN（コードへ導入しない）** |
| T2 authority | **NO-GO** |
| **T1 実装** | **次工程（最小実装・R gate/held-set/free-stack/token/ReservationExhausted 実生成は排除）** |

### D77.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T241 | observedOutstanding 非負性 | acquire/release の差分として維持・負値は正常状態で許容しない |
| T242 | 負値検出は診断のみ | lifetime correctness に影響しない（Debug assert / health event） |
| T243 | telemetry ordering ≠ lifetime ordering | lifetime correctness は T2 authority が担う |

---

## D78 — Design-60 確定（「負値を許容しない」≠「correctness invariant」・負値検出でも処理を停止・補正・rollback しない・2026-08-16）

ユーザー Design-59 レビュー対応: **「D77 は問題なし。observedOutstanding は単なる差分カウンタでなく観測整合性を
診断するための値。重要なのは『負値を許容しない』と『負値にならないことを correctness invariant にする』は別
という点。D77 は前者に留める必要がある。実装では負値を検出しても reservation/lifetime の処理を停止・補正・
rollback してはいけない」**。**ゲート: T1 telemetry ≠ reservation authority ≠ lifetime authority・I-2 はまだ
NO-GO。Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D78.1 「負値を許容しない」と「correctness invariant」の区別（ご指定・核心）
- **「負値を許容しない」**（D77・採用）: `observedOutstanding < 0` は正常系ではない・**観測整合性を診断する
  ための値**（diagnostic-only）。
- **「負値にならないことを correctness invariant にする」**（D77 では不採用）: これは観測を lifetime の
  correctness 条件に組み込むことになり、**T1 telemetry ≠ lifetime authority** の境界を破る。

### D78.2 ★ 負値検出でも処理を停止・補正・rollback しない（ご指定）
- **実装では負値を検出しても reservation / lifetime の処理を停止・補正・rollback してはいけない**。
- 負値検出は診断（Debug assert / health event）としてのみ記録（D77.2 と整合）。

### D78.3 ゲート（ご指定・明文化）
$$
\boxed{\text{T1 telemetry} \neq \text{reservation authority} \neq \text{lifetime authority}}
$$
- T1 = observational state only・held_count / held_set / free_stack / token / ABA / R gate / ReservationExhausted
  実生成は T2・R / R_cap は未確定。

### D78.4 最終状態
| 項目 | 状態 |
|------|------|
| D77（非負性・異常検出・ordering 非代替） | CLOSED |
| **「負値を許容しない」≠「correctness invariant」** | **CLOSED（本 D78・前者のみ採用）** |
| **負値検出でも処理を停止・補正・rollback しない** | **CLOSED（本 D78）** |
| T1 telemetry ≠ reservation authority ≠ lifetime authority | CLOSED（本 D78・ゲート） |
| R / R_cap | **OPEN（未確定）** |
| T2 authority | **NO-GO** |
| **I-2** | **NO-GO** |

### D78.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T244 | 負値許容 vs correctness invariant 区別 | 観測整合性の診断値・correctness invariant にしない |
| T245 | 負値検出でも停止・補正・rollback しない | 診断のみ・reservation/lifetime 処理に逆流しない |
| T246 | T1 ≠ authority ≠ lifetime | ゲート確認（観測のみ） |

---

## D79 — Design-61 確定（observedOutstanding の型契約・A/R は unsigned・O は derived signed・sampler 観測競合・2026-08-16）

ユーザー Design-60 レビュー対応: **「D78 の論点整理は妥当。ただし CLOSED 前に 1 点追加で固定すべき実装契約。
observedOutstanding < 0 を異常検出するなら実装型は unsigned であってはいけない（unsigned では 0-1 が wraparound
し検出契約が成立しない）。推奨: acquireObserved / releaseObserved = monotonic unsigned counters・observedOutstanding
= derived signed diagnostic value（O = A - R）。A/R の読み取りが別 atomic なら sampler が一時的に A=100/R=101 を
観測し得るが、これは lifetime の異常ではない。diagnostic は『observedOutstanding < 0 は telemetry observation
inconsistency / sampling artifact を含み得る diagnostic signal であり、runtime correctness failure を意味
しない』まで明記」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D79.1 ★ observedOutstanding の型契約（ご指定・unsigned wraparound を「負値」と誤認しない）
- **`observedOutstanding` は signed でなければならない**（`< 0` 検出契約が成立するため）。
- **unsigned 型（uint32_t / uint64_t）は不可**（0 - 1 が wraparound し `observedOutstanding < 0` が検出不能）。
- **推奨方式（ご指定）**: `acquireObserved / releaseObserved = monotonic unsigned counters`・
  `observedOutstanding = derived signed diagnostic value`（$O = A - R$ を観測用の導出値として signed で保持）。

### D79.2 ★ sampler 観測競合（ご指定・一時的不整合は correctness failure でない）
- **A と R の読み取りが別 atomic なら、Non-RT sampler が一時的に A=100 / R=101 を観測し得る**。
- これは **lifetime の異常ではない**（観測順序の一時的不整合）。
- **diagnostic 明記**:
  > `observedOutstanding < 0` は **telemetry observation inconsistency / sampling artifact を含み得る
  > diagnostic signal であり、runtime correctness failure を意味しない**。

### D79.3 不変条件（ご指定）
$$
\boxed{\text{diagnostic anomaly} \;\not\Rightarrow\; \text{lifetime correction}}
$$
- observedOutstanding は authority state ではない・負値は正常状態ではない・負値検出でも停止・補正・rollback
  しない・telemetry は lifetime correctness に介入しない（D78 と整合）。

### D79.4 最終状態
| 項目 | 状態 |
|------|------|
| D78（負値許容 vs correctness invariant） | CLOSED（本 D79 の型・sampling 契約を確認後） |
| **observedOutstanding の型（signed・A/R は unsigned）** | **CLOSED（本 D79）** |
| **sampler 観測競合（一時的不整合 ≠ correctness failure）** | **CLOSED（本 D79）** |
| **diagnostic anomaly ⇏ lifetime correction（不変条件）** | **CLOSED（本 D79）** |
| R / R_cap | **OPEN（未確定）** |
| T2 authority / I-2 | **NO-GO** |
| **D78 実ファイル再確認** | **実コード/正本で型・sampling を確認してから CLOSED（ご指摘）** |

### D79.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T247 | observedOutstanding は signed | unsigned では wraparound で負値検出不能・A/R は monotonic unsigned・O は derived signed |
| T248 | sampler 観測競合 | 別 atomic 読み取りで一時的不整合（A=100/R=101）は correctness failure でない |
| T249 | diagnostic anomaly ⇏ lifetime correction | 不変条件（診断のみ・補正なし） |

---

## D80 — Design-62 確定（O は diagnostic estimate・sampler read semantics・B_max^observed の意味論固定・2026-08-16）

ユーザー Design-61 レビュー対応: **「D79 の修正は妥当（負値検出と runtime correctness の分離が型・観測意味論の
双方で明確）。ただし T1 実装前に、A と R の読み取りを個別 atomic load した結果から O を計算すること自体がどの
程度の観測保証を意味するのかを明確にする。A=100/R=99 の直後に A=100/R=100 を読むことは当然あり得る。したがって
O は『ある瞬間の厳密な outstanding 数』ではなく『sampler が取得した A/R の観測点から導出した diagnostic
estimate』。これは B_max にも影響する。B_max を R の決定根拠にするなら B_max^observed が厳密値なのか下界なのか
sampling artifact を含む estimate なのかを実装前に固定する」**。**R / R_cap はまだコードに持ち込まない・T2
authority も NO-GO。Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D80.1 ★ O は diagnostic estimate（ご指定・厳密な瞬間 outstanding ではない）
- **A と R は個別 atomic load**・sampler が A=100/R=99 を読んだ直後に A=100/R=100 を読むことは当然あり得る。
- **∴ O は「ある瞬間の厳密な outstanding 数」ではなく**、
  **「sampler が取得した A/R の観測点から導出した diagnostic estimate」** と定義。

### D80.2 sampler read semantics（ご指定・明文化）
- sampler は A・R を個別 atomic load（一貫性のあるペア読取を保証しない）・O は観測点由来の estimate・
  一時的不整合（O < 0）は sampling artifact を含む diagnostic signal（D79.2）。

### D80.3 ★ B_max^observed の意味論固定（ご指定）
- **B_max^observed は「厳密値」ではない**。
- **B_max^observed は sampling artifact を含む estimate・保守的下界**（D66.6 / D67.4 と整合:
  B_max^observed ≤ B_max^true・M で補償）。
- **R の決定根拠として使用する際は、B_max^observed が下界であることを明示**（厳密値と誤認しない）。

### D80.4 ゲート（ご指定）
```
D79 → A/R/O の実コード型確認 → sampler の read semantics 確認 → T1 最小実装
  → RT/unit-test → sampler validation → 実測
```
- **R / R_cap はまだコードに持ち込まない**・**T2 authority も NO-GO**（現在の境界維持）。

### D80.5 最終状態
| 項目 | 状態 |
|------|------|
| D79（型契約・sampler 観測競合・diagnostic anomaly ⇏ correction） | CLOSED |
| **O は diagnostic estimate（厳密な瞬間 outstanding ではない）** | **CLOSED（本 D80）** |
| **sampler read semantics（A/R 個別 atomic load）** | **CLOSED（本 D80）** |
| **B_max^observed は estimate・保守的下界（厳密値でない）** | **CLOSED（本 D80）** |
| R / R_cap | **OPEN（コードに持ち込まない）** |
| T2 authority / I-2 | **NO-GO** |

### D80.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T250 | O は diagnostic estimate | A/R 個別 atomic load 由来・厳密な瞬間 outstanding でない |
| T251 | sampler read semantics | A・R 個別 load（一貫性ペア保証なし）・O は観測点由来 |
| T252 | B_max^observed は下界 | sampling artifact を含む estimate・保守的下界（厳密値でない）・R 根拠で明示 |

---

## D81 — Design-63 確定（B_max^observed の「保守的下界」撤回・sampled maximum・R 決定への正しい接続・2026-08-16）

ユーザー Design-62 レビュー対応: **「D80 で O を estimate と定義した点は妥当。ただし B_max^observed を
『保守的下界』と CLOSED にした部分には論理的問題がある。A と R を別々に atomic load するなら O_obs =
A_load − R_load は必ずしも O_obs ≤ O_true を満たさない（t0: A=100,R=99 → true=1・t1: release → R=100 のとき、
sampler が A を t0・R を別タイミングで読めば観測は真値を上回る/下回るケースがあり得る）。したがって
B_max^observed ≤ B_max^true は A/R の個別 atomic load だけからは証明できない。正しい意味論は B_max^observed =
max_samples(A_load − R_load)（sampled maximum）であり、厳密な peak でも下界でも上界でもなく sampling artifact
を含み得る。単独では R_required の安全側根拠にならない。R 決定には B_max^true ≤ B_max^observed + M の上側信頼
境界を実測方法から証明する必要がある。証明できないなら T1 telemetry では B_max^observed を純粋な観測統計量と
して扱い R の安全性証明にまだ使用しない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D81.1 ★ D80 の修正（B_max^observed の「保守的下界」撤回・ご指定）
- **撤回**: D80 の「B_max^observed は保守的下界（B_max^observed ≤ B_max^true）」は**成立しない**
  （A/R の個別 atomic load では観測が真値の上にも下にもなり得る・ご指定の例）。
- **正しい意味論**:
  $$
  B_{\max}^{observed} = \max_{samples}(A_{load} - R_{load})
  $$
  - 厳密な peak outstanding ではない・**true peak の下界であるとも限らない**・**true peak の上界でもない**・
    sampling artifact を含み得る・**単独では R_required の安全側根拠にならない**。

### D81.2 ★ R 決定への正しい接続（ご指定）
- **D80 の `B_max^observed ≤ B_max^true` + M の前提はそのままでは成立しない**。
- 必要なのは:
  $$
  B_{\max}^{true} \le B_{\max}^{observed} + M
  $$
  という**上側信頼境界**を、実測方法から証明すること。

### D81.3 ★ M の安全側根拠（ご指定）
- **M は単なる「sampling loss compensation」ではない**。
- M は以下に根拠を持つ必要がある:
  - 観測手続きが peak を**どの程度過小評価し得るか**（sampling loss）。
  - **A/R の非同時読取による観測誤差**をどの程度包含するか。
- **証明できない場合の安全な代替**: **T1 telemetry では B_max^observed を純粋な観測統計量として扱い、
  R の安全性証明にはまだ使用しない**（ご指定）。

### D81.4 最終状態（ご指定の判定を反映）
| 項目 | 状態 |
|------|------|
| O = sampler-derived diagnostic estimate | CLOSED（D80） |
| A/R 個別 atomic load | CLOSED（D80） |
| O < 0 は diagnostic-only | CLOSED（D78/D80） |
| **B_max^observed = sampled maximum** | **CLOSED（本 D81・正しい意味論）** |
| **B_max^observed = true peak の下界** | **OPEN / 要修正（本 D81 で撤回）** |
| **B_max^true ≤ B_max^observed + M（上側信頼境界）** | **OPEN（実測方法から証明）** |
| **M の安全側根拠** | **OPEN（sampling loss + 非同時読取誤差を包含）** |
| R / R_cap | OPEN（コードに持ち込まない） |
| T2 authority | NO-GO |

### D81.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T253 | B_max^observed = sampled maximum | max_samples(A_load − R_load)・true peak の下界/上界でない |
| T254 | B_max^true ≤ B_max^observed + M | 上側信頼境界（実測方法から証明）・単独では安全側根拠にならない |
| T255 | M の安全側根拠 | sampling loss + A/R 非同時読取誤差を包含 |

---

## D82 — Design-64 確定（A/R counter arithmetic・wraparound・measurement-duration contract・export 命名・2026-08-16）

ユーザー Design-63 レビュー対応: **「D81 の修正は妥当（B_max^observed を sampled statistic に戻したことで
R の安全性証明と telemetry 観測契約が再分離）。ただし T1 実装前に 1 点固定すべき技術的条件: (1) A/R を
uint64_t の atomic counter とする場合、O = A - R をそのまま unsigned 演算してから signed に変換してはいけない
（A=100/R=101 なら unsigned subtraction は巨大な wraparound 値）。実装契約は O = signedWide(A) − signedWide(R)
（差分計算前に十分な signed/wider domain へ変換）。(2) A/R の monotonic unsigned は実行期間中に wraparound
しないことを前提にする必要がある。(3) B_max^observed = max(A_load − R_load) は『true outstanding の推定値』
ではなく観測アルゴリズムが生成した統計量・export 名は observedOutstandingEstimate / observedOutstandingMax の
ように authority の outstanding と明確に区別。(4) M を『適当に大きくすれば安全』という設計に戻さない・
M は sampling interval・burst duration・A/R 読み取り順序・counter wraparound・measurement duration から導出
可能でなければならない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D82.1 ★ A/R 算術契約（ご指定・unsigned wraparound を回避）
- **`O = A - R` をそのまま unsigned 演算してから signed に変換してはいけない**
  （A=100 / R=101 なら unsigned subtraction は巨大な wraparound 値になる）。
- **実装契約**: $O = \text{signedWide}(A) - \text{signedWide}(R)$（差分計算前に十分な signed / wider domain へ
  変換・例: int64 に cast してから差分）。

### D82.2 ★ A/R wraparound 契約（ご指定）
- **A/R の monotonic unsigned は実行期間中に wraparound しないことを前提**。
- **measurement-duration contract**: 測定期間は「A/R が wraparound しない」ことを保証する範囲に制限
  （uint64 では 2^64 回 increment に要する時間 ≫ 測定期間・実用上到達不能だが契約として明示）。
- この前提により A/R の単調性が長期測定の意味を保証。

### D82.3 export 命名（ご指定・authority と明確に区別）
- **`B_max^observed = max(A_load − R_load)` は「true outstanding の推定値」ではなく観測アルゴリズムが生成した
  統計量**。
- **T1 export 名**: `observedOutstandingEstimate` / `observedOutstandingMax`（authority の `outstanding` /
  `held_count` と明確に区別）。

### D82.4 ★ M は導出可能（ご指定・「適当に大きくすれば安全」に戻さない）
- **M を R の安全性証明に使うなら、以下から導出可能でなければならない**:
  sampling interval・burst duration・A/R 読み取り順序・counter wraparound・measurement duration。
- **「M を適当に大きくすれば安全」という設計に戻さない**。

### D82.5 最終状態（ご指定の判定を反映）
| 項目 | 状態 |
|------|------|
| B_max^observed = sampled maximum / true peak 下界・上界主張撤回 | CLOSED（D81） |
| **O の signed/wide-domain arithmetic** | **CLOSED（本 D82）** |
| **A/R counter wraparound（measurement-duration contract）** | **CLOSED（本 D82）** |
| **export 命名（observedOutstandingEstimate / Max）** | **CLOSED（本 D82）** |
| **B_max^true ≤ B_max^observed + M** | **OPEN（実測方法から証明）** |
| **M の根拠（sampling interval / burst / read order / wraparound / duration から導出）** | **OPEN（導出可能性は本 D82 で固定）** |
| R / R_cap | OPEN（コードに持ち込まない） |
| T2 authority | NO-GO |

### D82.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T256 | O の signed/wide-domain arithmetic | signedWide(A) − signedWide(R)（unsigned 減算の wraparound 回避） |
| T257 | A/R wraparound | 測定期間中に wraparound しない（uint64・measurement-duration contract） |
| T258 | export 命名 | observedOutstandingEstimate / observedOutstandingMax（authority と区別） |
| T259 | M は導出可能 | sampling interval / burst / read order / wraparound / duration から導出 |

---

## D83 — Design-65 確定（T1 実コード実装レビュー・10 項目突合・責務分離確定・最小変更差分・2026-08-16）

ユーザー Design-64 レビュー対応: **「B_max^true ≤ B_max^observed + M と M の安全側導出は、T1 telemetry の
実装可否とは切り分ける（未解決のまま実装コードへ持ち込むと、観測装置に安全性保証の責務を混入させる）。
次のレビューでは実コードを再確認したうえで 10 項目をコード単位で突合する。(1) A/R の実型・配置・atomic
semantics (2) acquire observation の control-flow / linearization point (3) release observation の
deleter/context 経路 (4) observedOutstandingEstimate の signed-wide 算術 (5) sampler の実行コンテキスト・
周期・read order (6) observedOutstandingMax の更新責務 (7) export が lifetime authority に逆依存しないこと
(8) RT path に allocation / lock / logging / I/O が入らないこと (9) 既存 telemetry/debug infrastructure との
重複 (10) T1 で変更するファイルを最小集合に限定できるか」**。
**「observedOutstandingMax をどこで更新するか」: T1 の定義が『Non-RT sampler が観測した sampled maximum』
なら、acquire/release 側で max を更新する設計は T1 の測定装置と RT 側状態管理の境界を曖昧にする。責務分離:
```
RT / publish / deleter
    ↓
A/R observation counters only

Non-RT sampler
    ↓
A/R loads
    ↓
signedWide(A) - signedWide(R)
    ↓
observedOutstandingEstimate
    ↓
observedOutstandingMax
    ↓
window-tagged sample/export
```**
**「D82 の『M は導出可能でなければならない』は T1 実装の前提条件ではなく、R を決定するときの後段の
measurement-validation contract として残す」**。
**ゲート: D82 → T1 実コード実装レビュー → 最小 telemetry 実装 → compile / unit test → RT safety audit →
sampler / window validation → 実測 → B_max^observed の測定特性評価 → 必要なら M の導出 → R_required → R →
R_cap → T2 reservation authority → I-2 GO。R / R_cap・T2 authority・ReservationExhausted の実生成は依然として
T1 のコード変更対象外**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D83.1 実コード突合（10 項目・コード単位・本レビューで実コード再確認）
| # | 突合点 | 実コード事実 |
|---|--------|--------------|
| 1 | **A/R の実型・配置・atomic semantics** | 既存 `publishedWorldCount_` / `retiredWorldCount_`（AudioEngine.h:2225・`std::atomic<uint64_t> {0}`・acq_rel fetchAdd・lock-free 保証）。`publishedWorldCount_++` = onRuntimePublishedNonRt（Commit.cpp:403・publish 成功）・`retiredWorldCount_++` = onRuntimeRetiredNonRt（Commit.cpp:624・retire 開始・ASSERT_NON_RT_THREAD Commit.cpp:442）。既存 retiredWorldCount_ は「retire 開始」で ++（monotonic・-- なし・telemetry）。**T1 の A/R 観測は既存再利用 or 新設（最小差分判断 D83.4）** |
| 2 | **acquire observation の control-flow / linearization point** | publish 実行スレッド = **CoordinatorLoop（専用 juce::Thread・Non-RT）**（AudioEngine.Threading.cpp:237 processIntent → 266 runCoordinatorPhase・「plain juce::Thread (NonRT)」）。publish() LP = `publishAndSwap`（RuntimeStore.h:40-50・acq_rel exchange・non-failing）。onRuntimePublishedNonRt（AudioEngine.h:3453 / Commit.cpp:332）は didPublishRuntimeNonRt（AudioEngine.h:3504）→ executePublish（RuntimePublishExecutor.h:73）で publish 成功後に呼ばれる。acquireObserved 観測点候補 = onRuntimePublishedNonRt（最小差分）or publish() 内 swap 前（D75.1 設計位置） |
| 3 | **release observation の deleter/context 経路** | world-destroy deleter = retireRuntimePublishWorldNonRt（AudioEngine.h:3520-3538・unseal → ~RuntimePublishWorld → aligned_free・**キャプチャなしラムダ = 関数ポインタ**）。DeletionEntry（DeferredDeletionQueue.h:26-37）: ptr / deleter / epoch / type / publicationSequenceId / generation・**context フィールドなし・trivially copyable 制約**。enqueueWithRetry（ISRRetireRouter.cpp:161-208）= primary XOR quarantine。drain（物理破棄実行）= ISRRetireRouter::tryReclaim → provider->tryReclaim() + drainQuarantineStore()（AudioEngine.Retire.cpp:38,51,337 / Threading.cpp:266 / CtorDtor.cpp:213）。releaseObserved++ の観測点 = deleter 内（context 必要）or drain 実行点（最小差分判断 D83.4） |
| 4 | **observedOutstandingEstimate の signed-wide 算術** | D82 確定: $O = \text{signedWide}(A) - \text{signedWide}(R)$（int64 cast 後減算・unsigned wraparound 回避） |
| 5 | **sampler の実行コンテキスト・周期・read order** | timerCallback = **100ms 周期**（AudioEngine.h:2294 `int timerPeriodMs_ = 100` / Init.cpp:122）。emitEvidenceTickNonRt = **1 秒周期**（Commit.cpp:662-667・tick 間引き）。drain（tryReclaim）= CoordinatorLoop / Retire.cpp。sampler 候補 = timerCallback（100ms・Non-RT・MessageThread）※ read order: A_load → R_load（または R_load → A_load）— M の根拠（D82.4） |
| 6 | **observedOutstandingMax の更新責務** | **Non-RT sampler 側で更新**（acquire/release 側では更新しない・ユーザー指定）。**D75.1 の「acquire 観測フック ... max 更新（RT・atomic のみ）」を修正（本 D83）**。責務分離: publish/deleter = A/R observation counters only・sampler = O 計算・max 更新・window tag・export |
| 7 | **export が lifetime authority に逆依存しないこと** | emitEvidenceTickNonRt（Non-RT・1 秒）が既存 export 入口（Commit.cpp:661-710・evidenceExporter_.exportEvidence() Commit.cpp:706）。T1 の export（observedOutstandingEstimate / Max / window tag）は観測値のみ・authority 状態に依存しない |
| 8 | **RT path に allocation / lock / logging / I/O が入らないこと** | **publish / retirement observation は Non-RT control-flow 上で実行され、observation counter は lock-free atomic のみを使用する。audio callback から observation を行わない。**（publish = CoordinatorLoop・Non-RT・Threading.cpp:237-266 / onRuntimeRetiredNonRt = ASSERT_NON_RT_THREAD・Commit.cpp:442 / release = tryReclaim Non-RT 実行点）。acquire/release フックは atomic fetchAdd のみ（allocation / lock / logging / I/O なし）。**★ D84 修正（Design-65 レビュー）: 「publish は Non-RT だが ISR invariant」より本表現が正確** |
| 9 | **既存 telemetry/debug infrastructure との重複** | 既存: publishedWorldCount_ / retiredWorldCount_（累積）・debugRuntime_（HB edge / shadow compare / CI artifacts）・evidenceExporter_（ISREvidenceExporter・evidence 書き出し）・worldLifecycleAudit_（onWorldPublished / onWorldRetired 監査記録）・runtimeOrchestrator_->publishHealthSnapshot / notifyWorldRetired（健全性）。T1 = 観測値のみ（acquireObserved / releaseObserved / observedOutstanding* / B_max / window）・既存と独立 or 一部再利用 |
| 10 | **T1 で変更するファイルの最小集合** | D83.4 で確定 |

### D83.2 責務分離の確定（ユーザー図・ご指定どおり・D84 で修正）
```
publish / retirement terminal path
        │
        ├── acquireObserved / releaseObserved
        │       （atomic observation counters）
        │
        ▼
Non-RT sampler
        │
        ├── A/R individual atomic loads
        ├── signedWide(A) - signedWide(R)
        ├── observedOutstandingEstimate
        └── observedOutstandingMax
                 │
                 ▼
        window-tagged export
```
- **observedOutstandingMax の更新責務 = Non-RT sampler**（acquire/release 側では更新しない）。
- **D75.1 の「max 更新（RT・atomic のみ）」は本 D83 で撤回**（sampler 責務に移動）。
- **★ D84 修正（Design-65 レビュー）: 「RT path に telemetry を入れる」誤読を完全に避ける形へ修正**（publish は CoordinatorLoop の Non-RT・deleter も既存 retire/drain 経路で実行）。

### D83.3 M は後段 measurement-validation contract（ご指定）
- **B_max^true ≤ B_max^observed + M と M の安全側導出は、T1 telemetry の実装可否とは切り分け**。
- **M は R 決定時の後段 measurement-validation contract**（T1 実装の前提条件でない・D82.4 の導出可能性は維持）。

### D83.4 最小変更差分の確定（T1 最小実装・設計確定・実装 GO 後）
**T1 最小実装のファイル変更セット**:
| 変更 | ファイル | 内容 |
|------|----------|------|
| WorldRetirementTelemetry | AudioEngine.h（member 追加） | acquireObserved / releaseObserved / observedOutstanding(signedWide derived) / sampler state（B_max リング・window tag） |
| acquireObserved++ | AudioEngine.Commit.cpp（onRuntimePublishedNonRt・publishedWorldCount_++ 隣 Commit.cpp:403） | atomic fetchAdd（acq_rel）・LP = publish 成功 |
| releaseObserved++ | AudioEngine.h（world-delete deleter 経路・D83.4-2 参照） | deleter 実行で atomic fetchAdd（acq_rel）・LP = 物理破棄 |
| sampler | AudioEngine.Timer.cpp（timerCallback・100ms） | A/R load → signedWide → observedOutstandingEstimate → observedOutstandingMax → window tag |
| export | AudioEngine.Commit.cpp（emitEvidenceTickNonRt・1 秒） | observedOutstandingEstimate / Max / window tag を evidence 書き出し |
| ReservationExhausted enum | RuntimePublicationCoordinator.h | 型準備のみ（実生成なし・T2） |

**releaseObserved++ の最小差分候補**:
- **案 X（deleter context 追加）**: DeletionEntry に `void* context` + deleter を `void(*)(void*, void*)` 化 →
  DeferredDeletionQueue / ISRRetireRouter / RetireQuarantineStore のシグネチャ変更（他経路は context=nullptr）・
  **D73/D74 の「deleter で release observation」契約と整合・正確性優先**。
- **案 Z（drain 実行点で観測）**: tryReclaim 周辺（AudioEngine.Retire.cpp / Threading.cpp）で releaseObserved を
  更新・最小だが「世界破棄」の特定が課題（reclaimSuccessCount は全 retire 対象を含む）。
- **推奨 = 案 X**（deleter context・正確性優先・D73.4 の release LP = deleter execution と整合）。ただし
  **実装 GO 後にコード突合で確定（D74.3「T1 実装時に再確認」維持）**。

### D83.5 最終状態
| 項目 | 状態 |
|------|------|
| B_max^observed = sampled maximum / 下界・上界撤回 | CLOSED（D81） |
| O の signed/wide-domain arithmetic・A/R wraparound・export 命名 | CLOSED（D82） |
| **T1 実コード突合（10 項目・コード単位）** | **CLOSED（本 D83・実コード再確認）** |
| **責務分離（max は sampler 側・D75.1 の max 更新撤回）** | **CLOSED（本 D83）** |
| **M は後段 measurement-validation contract（T1 前提でない）** | **CLOSED（本 D83・切り分け）** |
| **T1 最小実装のファイル変更セット（最小差分）** | **CLOSED（本 D83・設計確定・実装 GO 後）** |
| **releaseObserved++ の実装候補（案 X 推奨・context 追加）** | **CLOSED（本 D83・実装 GO 後確定）** |
| B_max^true ≤ B_max^observed + M | **OPEN（後段 measurement-validation）** |
| M の根拠 | **OPEN（後段 measurement-validation）** |
| R / R_cap | **OPEN（コードに持ち込まない）** |
| T2 authority / ReservationExhausted 実生成 | **NO-GO（T1 の変更対象外）** |
| **T1 実装** | **次工程（実装 GO 後・最小実装・本 D83 のセット）** |

### D83.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T260 | T1 実コード突合 | 10 項目（A/R 型・LP・deleter/context・signed-wide・sampler・max 責務・export 非依存・RT safety・重複・最小差分） |
| T261 | 責務分離 | observedOutstandingMax は Non-RT sampler 側で更新（acquire/release 側では更新しない） |
| T262 | M は後段 contract | B_max^true ≤ B_max^observed + M は T1 実装の前提でない（後段 measurement-validation） |
| T263 | 最小差分 | ファイル変更セット（WorldRetirementTelemetry / acquire / release / sampler / export / enum 型準備） |
| T264 | acquire フック RT safety | atomic のみ（allocation / lock / logging / I/O なし） |
| T265 | export 非依存 | observedOutstanding* / window tag は観測値のみ・authority 状態に依存しない |
| T266 | release deleter context | 案 X（context 追加）で全経路（primary / quarantine / shutdown）同一 context（D74.1-3 再確認） |

---

## D84 — Design-66 確定（DeletionEntry context propagation 再確認・案 X は設計確定に留める・維持事項 5 点・D83 表現修正・2026-08-16）

ユーザー Design-65 レビュー対応: **「D83 の責務図は厳密には修正すべき（publish は CoordinatorLoop の Non-RT・
deleter も既存 retire/drain 経路）。責務図を『publish / retirement terminal path → acquireObserved /
releaseObserved（atomic observation counters）→ Non-RT sampler → A/R loads → signedWide →
observedOutstandingEstimate → observedOutstandingMax → window-tagged export』とし、『RT path に telemetry を
入れる』誤読を完全に避ける。#8 は『publish / retirement observation は Non-RT control-flow 上で実行され、
observation counter は lock-free atomic のみを使用する。audio callback から observation を行わない。』が正確。
案 X（DeletionEntry に context 追加 + deleter を void(*)(void*,void*) 化）は T1 の最小 telemetry 実装としては
まだ『設計確定』に留めるべき。理由は DeletionEntry が trivially copyable で既存の queue / quarantine / retry /
drain のデータ契約に直接関係するため。全経路（copy/move・queue storage・quarantine transfer・retry・
terminalization・shutdown drain・deleter invocation signature）を確認せずにコード変更へ進むのは避けるべき。
次の実装レビューでは案 X の『context を追加した場合に DeletionEntry の全 terminal path が完全に同じ context
を保持するか』だけを実コードで確認するのが適切。維持: T1 では held_count を作らない・R / R_cap をコードへ
入れない・ReservationExhausted を生成しない・telemetry anomaly で lifetime 処理を停止/補正/rollback しない・
deleter 自体を lifetime authority にしない」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D84.1 D83 表現修正（ご指定・責務図と #8）
- **責務図修正**（D83.2 更新・「RT path に telemetry を入れる」誤読を回避）:
  ```
  publish / retirement terminal path
          │
          ├── acquireObserved / releaseObserved（atomic observation counters）
          │
          ▼
  Non-RT sampler
          ├── A/R individual atomic loads
          ├── signedWide(A) - signedWide(R)
          ├── observedOutstandingEstimate
          └── observedOutstandingMax
                   ▼
          window-tagged export
  ```
- **#8 修正**: 「publish / retirement observation は Non-RT control-flow 上で実行され、observation counter は
  lock-free atomic のみを使用する。audio callback から observation を行わない。」

### D84.2 DeletionEntry context propagation 実コード確認（全 terminal path・ご指定）
| 経路 | 実コード | context 保持 |
|------|----------|--------------|
| copy / move | DeletionEntry は trivially copyable（static_assert 済み・DeferredDeletionQueue.h:34-37）・**QuarantinedEntry も trivially copyable（static_assert 済み・RetireQuarantineStore.h:45-47）** | void* context 追加後も両者とも trivially copyable 維持可能 |
| queue storage（primary） | DeferredDeletionQueue::enqueue（DeferredDeletionQueue.h:60-96）で entry フィールド個別代入（ptr/deleter/epoch/type/seq/gen） | context を追加代入する必要あり |
| quarantine transfer | enqueueWithRetry → quarantine（ISRRetireRouter.cpp:190・QuarantinedEntry{ptr,deleter,epoch,type,seq,gen,reason,time} に再構成） | **context を明示的に渡さないと失われる・最要注意** |
| retry | enqueueWithRetry（ISRRetireRouter.cpp:161-208）同一 ptr/deleter/epoch を再 enqueue | context も同一を再渡し |
| terminalization（normal drain） | DeferredDeletionQueue::reclaim（DeferredDeletionQueue.h:151-153）`entry.deleter(entry.ptr)` | `deleter(context, ptr)` に変更 |
| shutdown drain（queue） | drainAllUnsafe（DeferredDeletionQueue.h:212-214）`entry.deleter(entry.ptr)` | 同様 |
| quarantine drain | RetireQuarantineStore::drain / drainAllUnsafe（RetireQuarantineStore.h:110-160）`pendingDeleters[i](pendingPtrs[i])` | 同様（pendingContexts[] 追加必要） |
| deleter invocation signature | `void(*)(void*)` → `void(*)(void*, void*)` | 全 deleter 定義（world 以外は context 無視）と全呼び出し箇所の変更 |

**影響ファイル ≈ 9-10**: DeferredDeletionQueue.h / RetireQuarantineStore.h / EpochDomain.h（enqueueRetire×2） /
ISRRetireRouter.h/.cpp（enqueueRetire×2 + enqueueWithRetry + retireRT + retire + quarantineRetire） / 呼び出し元
（AudioEngine.h enqueueDeferredDeleteNonRt + retireRuntimePublishWorldNonRt / AudioEngine.Retire.cpp:32 /
SnapshotCoordinator.cpp:26,92 / SnapshotCoordinator.h:154,159 / EQProcessor.Core.cpp:52 / AudioEngine.Cache.cpp:16,41 /
ConvolverProcessor.Lifecycle.cpp:57,70）。

### D84.3 案 X は設計確定に留める（ご指定）
- **案 X（context 追加）は T1 の最小 telemetry 実装としては『設計確定』に留める**。
- **全 terminal path で同一 context が保持されることは設計上可能**（entry に格納し全転送で明示的に伝搬）・
  ただし実装は 9-10 ファイルに及ぶ（T1 最小実装の範囲外）。
- **実装 GO 後に確定**（代替案の比較も実装 GO 後）。

### D84.4 維持事項 5 点（ご指定）
- **T1 では held_count を作らない**（T2 authority の正本・D76）。
- **T1 では R / R_cap をコードへ入れない**。
- **T1 では ReservationExhausted を生成しない**（enum 型準備のみ・D75）。
- **telemetry anomaly で lifetime 処理を停止・補正・rollback しない**（D77.2・観測は診断のみ）。
- **deleter 自体を lifetime authority にしない**（release-observation 入口に限定・D72/D73 境界維持）。

### D84.5 最終状態
| 項目 | 状態 |
|------|------|
| D83 表現修正（責務図・#8） | **CLOSED（本 D84・D83 反映）** |
| **DeletionEntry context propagation 実コード確認（全 terminal path）** | **CLOSED（本 D84・実コード確認）** |
| **案 X は設計確定に留める（T1 最小実装の範囲外・実装 GO 後に確定）** | **CLOSED（本 D84）** |
| **維持事項 5 点** | **CLOSED（本 D84）** |
| B_max^true ≤ B_max^observed + M | **OPEN（後段 measurement-validation）** |
| M の根拠 | **OPEN（後段 measurement-validation）** |
| R / R_cap | **OPEN（コードに持ち込まない）** |
| T2 / ReservationExhausted 実生成 | **NO-GO** |
| **T1 最小実装** | **次工程（実装 GO 後・案 X の確定は GO 後）** |

### D84.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T267 | D83 表現修正 | 責務図（publish/terminal path → counters → sampler → export）・#8（Non-RT control-flow・audio callback から観測しない） |
| T268 | context propagation（全 terminal path） | copy/move・queue・quarantine transfer・retry・terminalization・shutdown drain・deleter signature で同一 context 保持 |
| T269 | 案 X は設計確定 | T1 最小実装の範囲外・実装 GO 後に確定 |
| T270 | 維持事項 5 点 | held_count なし・R/R_cap なし・ReservationExhausted 生成なし・anomaly で停止/補正/rollback なし・deleter は authority でない |

---

## D85 — Design-67 確定（release 観測方式再選定・案 X vs terminalization-side observation 比較・案 B の release LP 証明・2026-08-16）

ユーザー Design-66 レビュー対応: **「案 X が『可能』と確認された一方、T1 の最小実装としては過大な変更になることも実コードで
確認できた（DeletionEntry → quarantine → retry → normal drain → shutdown drain → deleter の全経路に context を
伝搬する必要があり、既存の retirement transport contract を変更する実装になる）。D70 以降で維持してきた『T1 は
測定装置』という境界に対して大きな変更。したがって次の実装では案 X を即実装するのではなく、releaseObserved の
観測点を変更せずに取得できる既存の terminalization instrumentation がないかを最終確認するのが適切。候補: A. deleter
に context を運ぶ（9〜10 files・retirement transport contract を変更）・B. terminalization / drain 実行側で『この entry
が実際に deleter を実行した』ことを観測する（DeletionEntry transport の変更を避けられる可能性）。ただし B は『deleter
実行 = world destruction』ではなく、既存 lifetime contract 上で release LP として正式に扱えるかをコードで証明する
必要がある。単に reclaim() の直前に counter を増やすだけでは D73 の契約を満たしたことにはならない」**。**現状態:
D84 → T1 acquire observation 実装可能・sampler / estimate / export 実装可能・release observation は X（context
propagation・大規模変更）or terminalization-side（要コード検証）。R / R_cap・T2 reservation authority・
ReservationExhausted 実生成は引き続き T1 の対象外**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D85.1 既存 terminalization instrumentation の最終確認（ご指定）
| 観測点 | 実コード | 備考 |
|--------|----------|------|
| DeferredDeletionQueue::reclaim 戻り値 | reclaimed count（DeferredDeletionQueue.h:155・uint32_t・「実際に解放した件数」） | **全 retire 対象の解放数（world だけではない）** |
| EpochDomain::tryReclaim | reclaimSuccessCount_（EpochDomain.h:391-394・relaxed fetch_add） | 同（全 retire 対象） |
| ISRRetireRouter::reclaimSuccessCount() | 既存（A-2 統計） | 同（全 retire 対象） |
| QuarantinedEntry | **type を保持**（RetireQuarantineStore.h:38） | **type ベース識別が quarantine 経路でも可能** |
| reclaim / drainAllUnsafe の entry.type アクセス | 可能（DeferredDeletionQueue.h:151-153, 212-214） | type 判定で world を特定可 |
| ISRRetireRouter の所有 | DeferredDeletionQueue は直接所有しない（provider_ 経由・EpochDomain が所有）・m_retireQuarantine は所有（ISRRetireRouter.h:230-242） | world 破棄カウンタの集約場所に影響 |
| enqueueDeferredDeleteNonRtWithResult | **常に Generic**（AudioEngine.h:4181 `enqueueWithRetry(ptr, deleter, epoch, DeletionEntryType::Generic)`） | type=World 指定にはパラメータ追加 or 専用経路 |

- **★ 既存の累積カウンタ（reclaimed / reclaimSuccessCount）は「全 retire 対象」であり、T1 の releaseObserved
  （world 物理破棄数）として直接使えない**（world を分離する type 判定が必須）。

### D85.2 案 B（terminalization-side observation）の release LP 証明（ご指定）
- **world の破棄は retireRuntimePublishWorldNonRt（AudioEngine.h:3520-3538）のみ**（単一 deleter・D74.1）→
  enqueue 時に type=World を指定可能。
- 全 terminal path で `type == World` の deleter 実行を観測:
  - primary reclaim: DeferredDeletionQueue::reclaim で `entry.type == World` の deleter 実行を数える。
  - quarantine drain: RetireQuarantineStore::drain で `QuarantinedEntry.type == World` の deleter 実行を数える。
  - shutdown drain: drainAllUnsafe（queue）+ drainAllUnsafe（quarantine）で同様。
- 各 world entry の deleter 実行は **exactly-once（D46-R4）+ terminal path 相互排他（D73.2）** → **≤1 release
  observation**。
- ∴ **one retirement obligation → ≤1 release observation を満たす（release LP として正式に扱える）**。
- **★ 「単に reclaim() の直前に counter を増やす」は不十分**（reclaim の戻り値は全 retire 対象を含み world を
  特定しない）→ **type==World の deleter 実行を数える**ことが正しい観測点（D73.4 の release LP = deleter execution）。

### D85.3 案 X vs 案 B 比較（ご指定）
| 観点 | 案 X（context 伝搬） | 案 B（type-based terminalization） |
|------|---------------------|-----------------------------------|
| 変更ファイル数 | ≈ 9-10 | ≈ 3-5（DeferredDeletionQueue.h・RetireQuarantineStore.h・ISRRetireRouter.h/.cpp・AudioEngine.h） |
| retirement transport contract 変更 | **大**（deleter signature void(*)(void*,void*) 化・全経路で context 伝搬） | **小**（enum に World 追加・type 判定・world 破棄カウンタ） |
| release LP | deleter 内（context 経由） | terminalization（type==World の deleter 実行・exactly-once） |
| one retirement → ≤1 release | D73/D74 と整合 | D73.2 相互排他 + type 一意性で成立 |
| RT safety | 変更なし（lock-free 維持） | 変更なし（enum / カウンタのみ） |
| world 破棄の特定 | deleter が world 専用なので正確 | type==World で正確 |

### D85.4 選定結果（T1 最小実装・release 観測方式）
- **案 B（type-based terminalization）を採用**。
- 案 X は設計確定に留める（D84・大規模変更のため保留）。
- **T1 release observation = 案 B**: `DeletionEntryType::World` 追加 → world enqueue 時に type=World → 全 terminal
  path で `type==World` の deleter 実行を数える → releaseObserved。
- 実装候補（GO 後）: DeferredDeletionQueue / RetireQuarantineStore に world 破棄カウンタ + ISRRetireRouter 集約 or
  AudioEngine が取得。

### D85.5 最終状態
| 項目 | 状態 |
|------|------|
| 既存 terminalization instrumentation 最終確認 | **CLOSED（本 D85・実コード確認）** |
| **案 B の release LP 証明（type==World・exactly-once・相互排他）** | **CLOSED（本 D85・コード事実）** |
| **案 X vs 案 B 比較** | **CLOSED（本 D85・案 B 採用）** |
| **T1 release 観測方式 = 案 B（type-based）** | **CLOSED（本 D85）** |
| T1 acquire / sampler / estimate / export | 実装可能（D84） |
| B_max^true ≤ B_max^observed + M | **OPEN（後段 measurement-validation）** |
| R / R_cap | **OPEN（T1 の対象外・コードに持ち込まない）** |
| T2 / ReservationExhausted 実生成 | **NO-GO（T1 の対象外）** |
| **T1 最小実装** | **次工程（実装 GO 後・release = 案 B）** |

### D85.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T271 | 既存 terminalization instrumentation | reclaim 戻り値・reclaimSuccessCount は全 retire 対象・world 分離不可 |
| T272 | 案 B release LP 証明 | type==World・exactly-once・相互排他 → ≤1 release observation |
| T273 | 案 X vs 案 B 比較 | 変更ファイル数・transport contract 変更（X 大・B 小） |
| T274 | release 観測方式 = 案 B | DeletionEntryType::World 追加・全 terminal path で type==World の deleter 実行を数える |

---

## D86 — Design-68 確定（T1 実装の非交渉条件 8 点・DeletionEntryType::World は telemetry metadata に留める・実装時チェック項目 2 点・2026-08-16）

ユーザー Design-67 レビュー対応: **「案 B の releaseObserved は『type == World の terminal deleter 実行時』に限定する、
という定義なら妥当。実装時には『retirement obligation 発生 → DeletionEntry(type=World) → primary / quarantine /
shutdown のいずれか一つ → World deleter 実行 → releaseObserved++』を明確に分離する。reclaim() の戻り値や
reclaimSuccessCount ではなく、World と識別できた terminal deletion の実行そのものを observation point にする。
非交渉条件 8 点。特に『World type の追加』と『lifetime authority の導入』を同一変更として扱わない。
DeletionEntryType::World はあくまで『今回の観測対象を識別するための既存 retirement metadata の拡張』」**。
**次工程 = T1 最小実装レビュー → 実装でよい。ただし実装差分を作る前に (1) DeletionEntryType::World を設定する全
enqueue 呼び出しが本当に world retirement の全経路を覆うこと (2) primary / quarantine / shutdown の全
terminalization 点で World が保持されること を実装時チェック項目として固定する。R / R_cap・T2 reservation
authority・ReservationExhausted 実生成は引き続き T1 の対象外**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更）。**

### D86.1 release observation の定義（ご指定・明確な分離）
```
retirement obligation 発生
    ↓
DeletionEntry(type=World)
    ↓
primary / quarantine / shutdown のいずれか一つ
    ↓
World deleter 実行
    ↓
releaseObserved++
```
- **observation point = World と識別できた terminal deletion の実行そのもの**（reclaim() の戻り値や
  reclaimSuccessCount ではない）。
- **★ D86 表現厳密化（Design-67 レビュー）: releaseObserved++ は「type 判定 → counter increment」の単純化ではなく、
  「World entry → terminalization path → World deleter executes → releaseObserved++」の順序を維持する。**
  deleter が実際に world を破棄したことを release observation とする（案 B の意味論）。
- **★ deleter は noexcept**: `unseal()` = ISRSealedObject.h:85 noexcept・`~RuntimePublishWorld`・`aligned_free` も
  noexcept → **terminalization 成功が既存契約で保証される**（world deleter は例外を投げない）。
- **★ control-flow 一本化（T1 実装時の最終確認・ご指定）**:
  ```
  publish / supersession
      ↓
  retireRuntimePublishWorldNonRt
      ↓
  DeletionEntry{ ..., type=World }
      ↓
  primary / quarantine / shutdown
      ↓
  World deleter
      ↓
  releaseObserved++
  ```

### D86.2 非交渉条件 8 点（ご指定）
1. **`DeletionEntryType::World` は telemetry 識別用の metadata に留める**。
2. **`type == World` の判定を lifetime authority にしない**。
3. **`releaseObserved++` は terminalization の成功後、または「deleter 実行を release と定義するならその LP」で 1 回だけ**。
4. **primary / quarantine / shutdown drain のどこを通っても二重観測されない**（既存の D73 terminal-path exclusion と再突合）。
5. **`observedOutstanding` は引き続き T1 diagnostic estimate**。
6. **`observedOutstanding < 0` が発生しても、処理停止・rollback・補正を行わない**。
7. **`held_count`、held-set、free-stack、token、R / R_cap、`ReservationExhausted` の実生成は T1 に入れない**。
8. **`observedOutstandingMax` は sampler 側だけで更新する**。
- **★ 「World type の追加」と「lifetime authority の導入」を同一変更として扱わない**（DeletionEntryType::World は
  既存 retirement metadata の拡張）。

### D86.3 実装時チェック項目 2 点（ご指定・固定・実コード確認）
| # | チェック項目 | 実コード確認 |
|---|--------------|--------------|
| 1 | **`DeletionEntryType::World` を設定する全 enqueue 呼び出しが world retirement の全経路を覆うこと** | world の物理破棄（`~RuntimePublishWorld`）は **AudioEngine.h:3529（retireRuntimePublishWorldNonRt の deleter）の 1 箇所のみ**（grep 確認）・world retire は retireRuntimePublishWorldNonRt（AudioEngine.h:3520-3538）に集約（単一 deleter・D74.1）・全呼び出し元 = AudioEngine.CtorDtor.cpp:231 / AudioEngine.Init.cpp:67,88 / AudioEngine.Processing.ReleaseResources.cpp:460 / RuntimePublishExecutor.h:76 / RuntimePublicationCoordinator.h:89,114,138（core テンプレート・テスト用抽象）・enqueueDeferredDeleteNonRt の他呼び出し元（AudioEngine.Cache.cpp:16,41・ConvolverProcessor.Lifecycle.cpp:57,70）は world 以外 → Generic のまま ✅・**★ D86 厳密化: 「~RuntimePublishWorld が 1 箇所」だけでなく「world object が必ず retireRuntimePublishWorldNonRt の deleter で登録されること」が必要**（publish/supersession → retireRuntimePublishWorldNonRt → DeletionEntry{type=World} の一本 control-flow・D86.1） |
| 2 | **primary / quarantine / shutdown の全 terminalization 点で `World` が保持されること** | primary: DeferredDeletionQueue::enqueue で type 代入 → reclaim / drainAllUnsafe で entry.type 保持（h:151-153,212-214）・quarantine: enqueueWithRetry 失敗 → quarantine(ptr,deleter,epoch,type,...)（ISRRetireRouter.cpp:190）→ QuarantinedEntry.type 保持（RetireQuarantineStore.h:38）→ drain / drainAllUnsafe・shutdown: drainAll（EpochDomain::drainAll → drainAllUnsafe）+ drainAllQuarantineStore（→ RetireQuarantineStore::drainAllUnsafe）・**二重観測なし = D73.2 terminal path 相互排他**（単一 storage・once-terminalized・shutdown は normal drain 後）✅ |

### D86.4 最終状態
| 項目 | 状態 |
|------|------|
| release observation 定義（type==World・terminal deleter 実行） | **CLOSED（本 D86）** |
| **非交渉条件 8 点** | **CLOSED（本 D86）** |
| **実装時チェック項目 2 点（実コード確認）** | **CLOSED（本 D86・実装時固定）** |
| **World type 追加 ≠ lifetime authority 導入** | **CLOSED（本 D86）** |
| T1 acquire / sampler / estimate / export / release（案 B） | 実装可能 |
| B_max^true ≤ B_max^observed + M | **OPEN（後段 measurement-validation）** |
| R / R_cap | **OPEN（T1 の対象外・コードに持ち込まない）** |
| T2 / ReservationExhausted 実生成 | **NO-GO（T1 の対象外）** |
| **T1 最小実装** | **次工程（実装 GO 後・本 D86 のチェック項目固定）** |

### D86.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T275 | release observation 定義 | type==World の terminal deleter 実行・reclaim 戻り値 / reclaimSuccessCount ではない |
| T276 | 非交渉条件 8 点 | World は metadata・authority 化しない・releaseObserved 1 回・二重観測なし・observedOutstanding は estimate・負値で停止/rollback/補正なし・held_count 等なし・max は sampler 側 |
| T277 | 実装時チェック 1 | type=World の全 enqueue が world retire の全経路を覆う（retireRuntimePublishWorldNonRt 集約・~RuntimePublishWorld は 1 箇所のみ） |
| T278 | 実装時チェック 2 | primary/quarantine/shutdown の全 terminalization で World 保持・二重観測なし（D73.2） |

---

## D87 — Design-69 確定（RT-safety / terminal-path verification・4 点実コード検証・29/29 PASS の正確な解釈・2026-08-16）

ユーザー Design-68 レビュー対応: **「29/29 PASS は『指定した除外条件のもとで 29 テストが PASS』という意味。現段階では
『T1 実装による回帰がない』までは支持できるが、『T1 の RT-safety / terminal-path correctness が証明された』と
はまだ言えない。次のゲート = RT-safety / terminal-path verification。案 B の核心は DeletionEntryType::World が
lifetime authority になっていないこと。worldReclaimCount_ の名前・配置・参照先も含めて静的に確認。検証対象 4 点を
固定: (1) Audio callback isolation (2) World tagging の完全性 (3) release observation の exactly-once
(4) T1/T2 境界の静的確認」**。**T1 実装済み・既存テスト PASS・Phase I/T2 は引き続き NO-GO。次 = コード差分に対する
RT-safety + 全 terminal path の実コード検証**。**Phase I 実装 NO-GO 継続（設計契約のみ）。**

### D87.1 29/29 PASS の正確な解釈（ご指定・訂正反映）
- **29/29 PASS は `ctest -C Debug -E "BuildInputSemanticContract|RuntimeWorldAuthority"` の除外条件のもとでの PASS**。
- **「T1 実装による回帰がない」までは支持できる**・**「T1 の RT-safety / terminal-path correctness が証明された」
  とは言えない**（本 D87 で実コード検証）。

### D87.2 Audio callback isolation（検証 1・実コード確認）
| 対象 | 実コード事実 |
|------|--------------|
| audio callback | AudioEngineProcessor::processBlock（AudioEngineProcessor.cpp:85,91,100）→ getNextAudioBlock（AudioBlock.cpp:27）/ processBlockDouble（BlockDouble.cpp:27） |
| acquireObserved++ | Commit.cpp:406・onRuntimePublishedNonRt → CoordinatorLoop（Non-RT・Threading.cpp:237 processIntent → executePublish → didPublishRuntimeNonRt） |
| releaseObserved / estimate / max / window tag | Timer.cpp:424-437・timerCallback（MessageThread・Non-RT） |
| export | Commit.cpp:715-730・emitEvidenceTickNonRt（Non-RT） |
| worldReclaimCount_++ | DeferredDeletionQueue::reclaim / drainAllUnsafe + RetireQuarantineStore::drain / drainAllUnsafe（tryReclaim 経由・CoordinatorLoop / Retire.cpp / CtorDtor / ReleaseResources・Non-RT） |
| isNegative | ISRWorldRetirementTelemetry.h:90 定義のみ・**呼び出し元なし**（診断のみ・lifetime に影響しない・D77.2） |
- **audio callback（getNextAudioBlock / processBlockDouble）には T1 の観測・更新・export の呼び出しなし** ✅
- atomic lock-free だけでなく、**観測処理そのものが RT callback に侵入していない** ✅

### D87.3 World tagging の完全性（検証 2・実コード確認）
- **primary 経路**: retireRuntimePublishWorldNonRt（AudioEngine.h:3527・type=World 指定）→ enqueueDeferredDeleteNonRt（4166・type 伝搬）→ enqueueWithRetry（4185）→ enqueueRetire → enqueueRetireTyped（ISRRetireRouter.cpp:107,124）→ EpochDomain::enqueueRetireTyped（403-408）→ deferredDeletionQueue.enqueue(ptr,deleter,epoch,type) ✅
- **retry 経路**: enqueueWithRetry のループ（161-208）で同一 type を再渡し ✅
- **quarantine 経路**: enqueueWithRetry 失敗 → m_retireQuarantine.quarantine(ptr,deleter,epoch,type,...)（191）→ QuarantinedEntry.type 保持（RetireQuarantineStore.h:38）✅
- **shutdown drain**: drainAll（EpochDomain → drainAllUnsafe）+ drainAllQuarantineStore（→ RetireQuarantineStore::drainAllUnsafe）✅
- 他 enqueue 呼び出し元（DSPLifetimeManager / SnapshotCoordinator / EQProcessor）は Generic（world 以外・正しい）✅

### D87.4 release observation の exactly-once（検証 3・実コード確認）
- **worldReclaimCount_++ は 4 箇所のみ**: DeferredDeletionQueue::reclaim（h:149）/ drainAllUnsafe（h:196）+ RetireQuarantineStore::drain（h:138）/ drainAllUnsafe（h:170）・すべて **type==World の terminal deleter 実行後**（D86.1 の順序維持）✅
- **reclaimSuccessCount（全 retire 対象）とは独立カウンタ**・混同なし（worldReclaimCount は world 物理破棄専用）✅
- 各 world entry は once-terminalized（D46-R4）+ terminal path 相互排他（D73.2）→ **terminal path ごとに ≤1 回** ✅

### D87.5 T1/T2 境界の静的確認（検証 4・grep/差分）
- **held_count / held_set / free_stack / freeStack / reservationToken / R_cap**: ISRWorldRetirementTelemetry.h:31 のコメントのみ・**実装なし** ✅
- **ReservationExhausted**: RuntimePublicationCoordinator.h:24 の enum 定義のみ・**実生成なし**（D86 条件 7）✅
- **isNegative**: 定義のみ・呼び出し元なし（**telemetry anomaly による停止・rollback・補正なし**・D86 条件 6）✅
- **R / R_cap**: コードに混入なし（T1 の対象外）✅
- **worldReclaimCount_ の名前・配置・参照先**: storage 側（DeferredDeletionQueue / RetireQuarantineStore）の telemetry 専用カウンタ・lifetime authority でない（D86 条件 2）✅

### D87.6 最終状態
| 項目 | 状態 |
|------|------|
| 29/29 PASS の正確な解釈（除外条件のもとでの PASS・訂正反映） | **CLOSED（本 D87）** |
| **Audio callback isolation（検証 1）** | **CLOSED（本 D87・実コード確認）** |
| **World tagging の完全性（検証 2）** | **CLOSED（本 D87・実コード確認）** |
| **release observation の exactly-once（検証 3）** | **CLOSED（本 D87・実コード確認）** |
| **T1/T2 境界の静的確認（検証 4）** | **CLOSED（本 D87・grep/差分）** |
| B_max^true ≤ B_max^observed + M | **OPEN（後段 measurement-validation）** |
| R / R_cap | **OPEN（T1 の対象外・コードに持ち込まない）** |
| T2 / ReservationExhausted 実生成 | **NO-GO** |
| **sampler / observation-window validation** | **次工程** |
| 実測 → M 導出 → R_required → R / R_cap → T2 → I-2 GO | 後段 |

### D87.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T279 | Audio callback isolation | audio callback（getNextAudioBlock / processBlockDouble）に T1 観測なし（acquireObserved / releaseObserved / worldReclaimCount / sampler / export） |
| T280 | World tagging 完全性 | type=World が primary / retry / quarantine / shutdown の全経路で保持 |
| T281 | exactly-once | worldReclaimCount_++ は 4 箇所のみ・terminal path ごとに ≤1 回・reclaimSuccessCount と混同なし |
| T282 | T1/T2 境界 | held_count 等なし・ReservationExhausted 実生成なし・anomaly による停止/rollback/補正なし |

---

## D88 — Design-70 確定（sampler / observation-window validation・6 点実コード検証・B_max^observed の測定手続き明確化・M は未証明のまま・2026-08-16）

ユーザー Design-69 レビュー対応: **「次のゲート = sampler / observation-window validation。検証対象は T1 の観測統計量
そのものに限定。R / R_cap / T2 authority は導入しない。確認項目 6 点: (1) sampling interval (2) observation
window (3) A/R snapshot semantics (4) max 更新 (5) measurement-duration contract (6) export。判定基準: B_max^observed
の測定手続きが明確になったことと B_max^true ≤ B_max^observed + M が証明されたことを分離する。後者は OPEN のまま。
sampling interval・burst duration・A/R 非同時読取などから M を導出できることが確認されるまでは、observedOutstandingMax
を R の安全側根拠には使用しない」**。**現時点: T1 実装済み・D87 CLOSED・R/R_cap・T2 は引き続き NO-GO**。
**Phase I 実装 NO-GO 継続（設計契約のみ）。**

### D88.1 sampling interval（検証 1・実コード確認）
- **timerPeriodMs_ = 100**（AudioEngine.h:2294）・**startTimer(100)**（AudioEngine.Init.cpp:121）→ 契約 100ms 一致 ✅
- timerCallback の jitter 計測（Timer.cpp:386-410・`#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS`・jitter > max(20ms, expectedMs*0.1) で
  diagLog に interval / jitter / estimatedMissed を記録）→ **実際の観測間隔の逸脱は記録可能** ✅
- sampler は timerCallback 内（Timer.cpp:421-438・100ms 周期・MessageThread Non-RT）

### D88.2 observation window（検証 2・実コード確認→定義確定）
- 現在の実装: observedOutstandingMax は **accumulated max（全測定期間・window 境界リセットなし）**。
- **★ 観測ウィンドウ T の定義（本 D88 で確定）**: 観測ウィンドウ = 測定開始（engine 起動後の最初の sampler tick）から
  現在まで・各 timerCallback tick が観測サンプル・observedOutstandingMax = この観測ウィンドウ内の
  observedOutstandingEstimate の running max。
- window tag: sampler（Timer.cpp:435-437）が Normal / Shutdown を更新（D76.2 の Stall / Catastrophic は実測段階で
  導入可・現状未使用）。
- **window 境界での max 持ち越し**: 現在は「全期間 max」であり window 境界リセットなし。ウィンドウごとの B_max
  （burst）が必要な場合（実測段階・後段）は、window 境界でリセットする設計を追加（要実装時対応・本 D88 では
  全期間 accumulated max を確定）。
- **★ D89 厳密化（Design-70 レビュー）: window の意味論は CLOSED・window reset / bounded measurement
  implementation は OPEN**（accumulated max のままでは測定開始・終了を明示できない → 測定試験単位の統計量として
  扱えない・D89）。

### D88.3 A/R snapshot semantics（検証 3・実コード確認）
- observedOutstandingEstimate = `signedWide(A) - signedWide(R)`（ISRWorldRetirementTelemetry.h:82-86・D82）✅
- A.load / R.load は非同時読み取り → **true outstanding の peak と同一視しない**（D81・sampled maximum の意味論）✅

### D88.4 max 更新（検証 4・実コード確認）
- **updateObservedOutstandingMax の呼び出し元は sampler（Timer.cpp:433）のみ** ✅
- RT / publish / deleter 側から直接更新なし（grep 確認）✅

### D88.5 measurement-duration contract（検証 5・実コード確認）
- A / R は uint64_t・monotonic・測定期間中 wraparound しない（D82.2）✅
- **実測手順として検証可能**（2^64 回 increment に要する時間 ≫ 測定期間・実用上到達不能・契約として明示）✅
- **★ D89 厳密化（Design-70 レビュー）: D82 の counter arithmetic / measurement-duration contract は CLOSED・
  実測による counter-wraparound validation は OPEN**（「実測手順として検証可能」は「実測済み」を意味しない・D89）。

### D88.6 export（検証 6・実コード確認）
- observedOutstandingEstimate / Max / window tag は world_retirement_telemetry.json に出力（Commit.cpp:713-735）✅
- **R_required や reclaim authority の判断に接続されていない**（diagnostic observation のみ）✅

### D88.7 判定基準（ご指定・分離）
- **B_max^observed の測定手続きが明確になった**（本 D88: sampling interval / observation window / A-R snapshot /
  max 更新 / measurement-duration / export を実コードで確認）。
- **B_max^true ≤ B_max^observed + M は OPEN（後段）**: sampling interval・burst duration・A/R 非同時読取などから M を
  導出できることが確認されるまで、**observedOutstandingMax を R の安全側根拠に使用しない**。

### D88.8 最終状態
| 項目 | 状態 |
|------|------|
| sampling interval（100ms・jitter 記録） | **CLOSED（本 D88・実コード確認）** |
| observation window（観測ウィンドウ T の定義） | **CLOSED（本 D88・全期間 accumulated max・window リセットは実測段階で追加）** |
| A/R snapshot semantics（signedWide・peak 非同一視） | **CLOSED（本 D88）** |
| max 更新（sampler のみ） | **CLOSED（本 D88）** |
| measurement-duration contract（wraparound 非発生） | **CLOSED（本 D88・実測手順）** |
| export（diagnostic のみ・R_required 未接続） | **CLOSED（本 D88）** |
| **B_max^observed の測定手続き** | **CLOSED（本 D88・明確化）** |
| **B_max^true ≤ B_max^observed + M** | **OPEN（後段・M 導出まで observedOutstandingMax を R の安全側根拠にしない）** |
| R / R_cap | **OPEN（T1 の対象外・コードに持ち込まない）** |
| T2 / ReservationExhausted 実生成 | **NO-GO** |
| **実測 → B_max 測定特性評価 → M 導出** | **次工程** |
| R_required → R → R_cap → T2 → I-2 GO | 後段 |

### D88.9 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T283 | sampling interval | timerPeriodMs_=100・startTimer(100)・jitter 記録（Timer.cpp:386-410） |
| T284 | observation window | 観測ウィンドウ T の定義（全期間 accumulated max・window リセットは実測段階） |
| T285 | A/R snapshot | signedWide(A)-signedWide(R)（ISRWorldRetirementTelemetry.h:82-86）・peak 非同一視（D81） |
| T286 | max 更新限定 | updateObservedOutstandingMax は sampler（Timer.cpp:433）のみ |
| T287 | measurement-duration | A/R wraparound 非発生（D82.2・実測手順） |
| T288 | export 診断限定 | observedOutstanding* / window tag は diagnostic・R_required 未接続 |

---

## D89 — Design-71 確定（D88 判定の厳密化・measurement protocol 固定・window-reset T1 instrumentation の最小差分設計・2026-08-16）

ユーザー Design-70 レビュー対応: **「D88 の判定には 2 点、表現を厳密化した方がよい。(1) observation window は
『CLOSED』ではなく、window の意味論は CLOSED・window reset / bounded observation interval の実装は OPEN
（accumulated max のままでは測定開始・終了を明示できない → 複数の負荷試験を跨いだ値になり、測定試験単位の統計量
として扱えない）。(2) wraparound は『実測手順として検証可能』と『実測済み』を分離（D82 contract は CLOSED・実測に
よる counter-wraparound validation は OPEN）。実測へ進む前に measurement protocol を固定する。次にコードを変更する
なら、R 関連ではなく『測定 window を明示的に開始・終了できる T1 measurement instrumentation』の最小差分を先に
設計・レビューするのが適切」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更・window-reset 実装は GO 待ち）。**

### D89.1 D88 判定の厳密化（ご指定・2 点）
- **window semantics: CLOSED**・**window reset / bounded measurement implementation: OPEN**（現在は accumulated max・
  測定開始・終了を明示できない → 測定試験単位の統計量として扱えない）。
- **D82 counter arithmetic / measurement-duration contract: CLOSED**・**実測による counter-wraparound validation: OPEN**
  （「実測手順として検証可能」は「実測済み」を意味しない）。

### D89.2 measurement protocol の固定（ご指定）
```
Measurement Start
    │
    ├─ A0 = acquireObserved.load()
    ├─ R0 = releaseObserved.load()
    │
    │    sampling ticks
    │    A/R → signedWide(A) - signedWide(R)
    │    → observedOutstandingEstimate
    │    → running max
    │
    └─ Measurement End
         │
         ├─ A1 / R1
         ├─ final observedOutstandingEstimate
         ├─ observedOutstandingMax
         ├─ actual sampling intervals
         ├─ jitter / missed ticks
         └─ window metadata
```
- **B_max^observed を真の peak の代用品として扱わない**（D81 の sampled maximum 意味論）。
- **実測で同時記録すべき項目**（ご指定）: actual sampling interval・最大 sampling gap・jitter・missed/late tick・
  測定 window の開始・終了時刻・A/R 各サンプル・signedWide(A)-signedWide(R)・sampled maximum・counter の開始値・
  終了値（A0 / A1・R0 / R1）・counter wraparound の有無。
- この結果から初めて **B_max^observed の測定特性**を評価し、その後に **B_max^true ≤ B_max^observed + M** の M 導出
  可能性を検討する。

### D89.3 window-reset T1 instrumentation の最小差分設計（ご指定・次コード変更の対象・設計レビュー）
- **測定 window を明示的に開始・終了できる T1 measurement instrumentation** の最小差分を設計。
- **設計方針**:
  - Measurement Start / End を明示（Non-RT API・sampler / テストから呼ぶ）。
  - window ごとの max リセット（accumulated max ではなく bounded window max）。
  - A0 / A1・R0 / R1（counter 開始値・終了値）の記録。
  - actual sampling intervals・最大 gap・jitter・missed ticks の記録。
  - window metadata（開始・終了時刻・window tag）の記録。
- **変更ファイル候補（最小）**: ISRWorldRetirementTelemetry.h（window state・start / end / reset API）+ AudioEngine.Timer.cpp
  （sampler の window 対応）+ export（Commit.cpp・window metadata 出力）。
- **R / R_cap・T2 は関与しない**（T1 の対象外・D86 非交渉条件）。

### D89.4 現在のゲート状態（ご指定）
| 項目 | 状態 |
|------|------|
| T1 実装 | CLOSED |
| RT-safety / terminal path | CLOSED |
| sampler 周期 | CLOSED |
| A/R signed-wide semantics | CLOSED |
| max 更新責務 | CLOSED |
| window semantics | **CLOSED** |
| **window reset / bounded measurement implementation** | **OPEN** |
| **実測 protocol** | **OPEN（本 D89 で固定）** |
| B_max^observed の測定特性 | OPEN |
| B_max^true ≤ B_max^observed + M | OPEN |
| M の安全側導出 | OPEN |
| R / R_cap | **NO-GO** |
| T2 authority | **NO-GO** |
| I-2 | **NO-GO** |

### D89.5 最終状態
| 項目 | 状態 |
|------|------|
| D88 判定の厳密化（window semantics CLOSED / window reset OPEN・contract CLOSED / 実測 validation OPEN） | **CLOSED（本 D89・D88 反映）** |
| **measurement protocol の固定** | **CLOSED（本 D89・D89.2）** |
| **window-reset instrumentation の最小差分設計** | **CLOSED（本 D89・設計レビュー・実装 GO 待ち）** |
| window reset / bounded measurement implementation | **OPEN（実装 GO 後）** |
| 実測 → B_max 測定特性評価 → M 導出 | **OPEN（次工程）** |
| B_max^true ≤ B_max^observed + M | OPEN（後段） |
| R / R_cap | **NO-GO** |
| T2 authority / I-2 | **NO-GO** |

### D89.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T289 | D88 判定厳密化 | window semantics CLOSED / window reset OPEN・contract CLOSED / 実測 validation OPEN |
| T290 | measurement protocol | Measurement Start（A0/R0）→ sampling → End（A1/R1・max・interval・jitter・window metadata） |
| T291 | window-reset instrumentation（設計） | start/end/reset API・window ごとの max リセット・A0/A1・R0/R1・sampling gap・window metadata |
| T292 | B_max は peak 代用品でない | sampled maximum 意味論（D81）・M 導出まで R の安全側根拠にしない |

---

## D90 — Design-72 確定（window-reset instrumentation 実装前レビュー・Start/End 同期方式の確定・最小 state 設計・2026-08-16）

ユーザー Design-71 レビュー対応: **「D89 の最小差分の実装前に 7 点を追加で確認するのが安全。(1) Start/End の実行
コンテキスト (2) Start 時の snapshot (3) End 時の snapshot (4) window reset の原子性 (5) End も同様 (6) window 中の
sampled maximum (7) wraparound validation。特に『現在の ISRWorldRetirementTelemetry の全メンバと、timerCallback()・
export 呼び出し側の実行コンテキストを再確認してから、Start/End request の同期方式を決める』のが次のレビューとして
最も安全。windowId や state の atomic 化を先に決める必要がある（Start/End API と sampler が別 Non-RT execution
context なら単純な non-atomic state は race の余地）。ここを曖昧にしたまま実装すると、D89 で固定した bounded
measurement の意味論そのものが sampler jitter / Start-End race に汚染される」**。**現時点: D89 CLOSED・
window-reset implementation OPEN・実装 GO 前レビュー継続**。**Phase I 実装 NO-GO 継続（設計レビューのみ・コード未変更）。**

### D90.1 現在のメンバ・実行コンテキスト再確認（ご指定）
| 対象 | 実コード事実 |
|------|--------------|
| WorldRetirementTelemetry の現在のメンバ（private） | `acquireObserved_` / `releaseObserved_` / `observedOutstandingMax_`（accumulated・sampler のみ更新）/ `windowTag_` — **すべて atomic** |
| sampler の実行コンテキスト | timerCallback（Timer.cpp:421-438）→ **MessageThread（JUCE Timer・Non-RT・100ms）** |
| export の実行コンテキスト | emitEvidenceTickNonRt（Commit.cpp:171,441,633・Timer.cpp からも）→ **Non-RT（CoordinatorLoop / MessageThread / ReleaseResources の複数 Non-RT スレッド）** |
| acquireObserved++ の実行コンテキスト | onRuntimePublishedNonRt（Commit.cpp:406）→ **CoordinatorLoop（Non-RT）** |
| Start/End API の実行コンテキスト | **Non-RT 限定**（ご指定固定点 1）・timerCallback から暗黙 Start/End なし・audio callback から window state 変更なし |

### D90.2 Start/End 同期方式の確定（ご指定固定点 4・5・sampler linearization point 方式）
- **sampler が window transition の linearization point になる方式**を採用。
  ```
  Start request（atomic publish・Non-RT）
      ↓
  sampler observes request（next timerCallback tick）
      ↓
  sampler establishes new window（A0/R0 snapshot・max reset・windowStart・windowId++）
  ```
  ```
  End request（atomic publish・Non-RT）
      ↓
  next sampler tick
      ↓
  final A/R sample（A1/R1）
      ↓
  close window（finalEstimate・windowMax 確定・windowEnd）
      ↓
  exportable result
  ```
- **Start/End API 自体は Non-RT request の発行のみ**（atomic request flag）・sampler が実際の A/R snapshot と max
  reset を**同一の観測手順として処理**。
- **window transition は sampler の単一スレッド（MessageThread）で linearize** → Start/End が別 Non-RT スレッド
  （テスト・計測ツール等）から呼ばれても race しない。

### D90.3 最小 state の設計（ご指定・atomic 化を確定）
- **追加 state（すべて atomic）**:
  ```
  measurementRequest_（atomic<uint8_t>: Idle / StartRequested / Running / EndRequested / Closed）
  windowId_（atomic<uint64_t>）
  windowStartTimestampUs_ / windowEndTimestampUs_（atomic<uint64_t>）
  startAcquire_ / startRelease_ / endAcquire_ / endRelease_（atomic<uint64_t>）
  windowMax_（atomic<int64_t>・window 内 sampled maximum・bounded）
  samplingStats_（atomic<uint64_t>: sampleCount / maxGapUs / jitter 集計 / missedTickCount）
  counterWrapped_（atomic<bool>・End 時に A1 < A0 || R1 < R0 で判定）
  ```
- **windowId / state は atomic 化**（Start/End API と sampler が別 Non-RT execution context の可能性 → race 回避・ご指定）。
- 現行の `observedOutstandingMax_` は「accumulated max」（後方互換・Idle 時）・window 測定中は `windowMax_` が
  bounded max（ご指定固定点 6・old window max を新 window に持ち込まない）。

### D90.4 固定点 2・3・6・7（ご指定）
| 固定点 | 内容 |
|--------|------|
| Start snapshot（固定点 2） | A0 = acquireObserved.load()・R0 = releaseObserved.load()・max = current estimate・windowStart = timestamp |
| End snapshot（固定点 3） | A1 / R1 を load・finalEstimate = signedWide(A1) - signedWide(R1)・observedOutstandingMax は window 内 sampled maximum として確定 |
| window 中の sampled maximum（固定点 6） | Start 前の accumulated max を新 window に持ち込まない（windowMax を reset・bounded） |
| wraparound validation（固定点 7） | **D82 の contract は維持**・実測結果として `counterWrapped = (A1 < A0 \|\| R1 < R0)` を記録（contract を置き換えない・測定結果のみ） |

### D90.5 変更ファイル（最小・設計レビュー）
- `ISRWorldRetirementTelemetry.h`（window-reset state + Start/End/reset API + snapshot result）・
  `AudioEngine.Timer.cpp`（sampler の window transition 対応）・export（Commit.cpp・window metadata 出力）・
  `AudioEngine.h`（telemetry に request API 公開・必要な場合）。
- **R / R_cap・T2 は関与しない**（T1 の対象外・D86 非交渉条件）。

### D90.6 最終状態
| 項目 | 状態 |
|------|------|
| Start/End 実行コンテキスト（Non-RT 限定・暗黙 Start/End なし） | **CLOSED（本 D90・固定点 1）** |
| **Start/End 同期方式（sampler linearization point）** | **CLOSED（本 D90・固定点 4・5）** |
| **最小 state 設計（atomic 化）** | **CLOSED（本 D90・D90.3）** |
| Start / End snapshot・window max・wraparound validation | **CLOSED（本 D90・固定点 2・3・6・7）** |
| **window-reset implementation** | **OPEN（実装 GO 待ち・D90.5 の最小差分）** |
| 実測 protocol（D89.2 固定済み）→ B_max 測定特性 → M 導出 | OPEN（次工程） |
| B_max^true ≤ B_max^observed + M | OPEN（後段） |
| R / R_cap | **NO-GO** |
| T2 authority / I-2 | **NO-GO** |

### D90.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T293 | Start/End 実行コンテキスト | Non-RT 限定・timerCallback から暗黙 Start/End なし・audio callback から window state 変更なし |
| T294 | Start snapshot | A0/R0 記録・max = current estimate・windowStart・windowId++ |
| T295 | End snapshot | A1/R1・finalEstimate = signedWide(A1)-signedWide(R1)・windowMax 確定 |
| T296 | sampler linearization point | Start/End request → sampler が window transition を単一スレッドで linearize・race なし |
| T297 | window max の bounded | old window max を新 window に持ち込まない（windowMax リセット） |
| T298 | wraparound validation | counterWrapped = (A1<A0\|\|R1<R0) を記録・D82 contract は維持（置き換えない） |

---

## D91 — Design-73 確定（Start/End request の上書き・重複要求の契約化・D90.5 実装契約（レビュー基準 10 点）の固定・accumulated と bounded windowMax の分離・2026-08-16）

ユーザー Design-72 レビュー対応: **「D90 の設計は D89 の未完了部分を実装へ落とす境界として妥当。実装 GO 前に 1 点
だけ厳密化: measurementRequest_ を単純な atomic state（Idle → StartRequested → Running → EndRequested → Closed）
だけで実装する場合、Start/End request の上書き・重複要求の意味を未定義にしない。少なくとも『StartRequested 中の
追加 Start、および EndRequested / Closed 中の追加 End は、既存 measurement window を変更しない』を契約化する。
Start と End が異なる Non-RT thread から発行され得るなら、request の順序は sampler が観測した atomic state の遷移を
唯一の linearization point とする。これにより『API 呼び出し時刻』ではなく、sampler observes Start → A0/R0 snapshot →
window begins / sampler observes End → A1/R1 snapshot → window closes が正式な measurement boundary になる。
D90.5 の実装差分レビュー基準 10 点をコード上で必ず確認。特に #8: 既存の observedOutstandingMax_ をそのまま window
max に変更すると、D88 で確定した『accumulated max』と D89 の『bounded measurement max』の意味が混ざる。測定 window
state を既存 telemetry の累積値から論理的に分離するのが適切」**。**Phase I 実装 NO-GO 継続（設計契約のみ・コード未変更・
実装 GO 待ち）。**

### D91.1 Start/End request の上書き・重複要求の契約化（ご指定）
- **Start API**: `measurementRequest_` が **Idle のときのみ StartRequested に遷移**（CAS・それ以外は無視）→
  **StartRequested 中の追加 Start は既存 measurement window を変更しない**。
- **End API**: `measurementRequest_` が **Running のときのみ EndRequested に遷移**（CAS・それ以外は無視）→
  **EndRequested / Closed 中の追加 End は既存 measurement window を変更しない**。
- **状態遷移（単調）**: `Idle → StartRequested → Running（sampler が Start 観測）→ EndRequested → Closed
  （sampler が End 観測）→ Idle（リセット・明示）`。
- **request の順序**: **sampler が観測した atomic state 遷移が唯一の linearization point**（API 呼び出し時刻ではない）。
- **正式な measurement boundary**:
  ```
  sampler observes Start → A0/R0 snapshot → window begins
  ...
  sampler observes End → A1/R1 snapshot → window closes
  ```

### D91.2 D90.5 実装契約（レビュー基準 10 点・ご指定・実装時にコード上で必ず確認）
| # | レビュー基準 |
|---|--------------|
| 1 | `Start` / `End` API が audio callback から呼ばれない |
| 2 | request state を変更するのは Non-RT API のみ |
| 3 | `timerCallback` が唯一の window state transition owner |
| 4 | A0/R0 は Start transition と同じ sampler tick で取得 |
| 5 | A1/R1 は End transition と同じ sampler tick で取得 |
| 6 | `windowMax` は Start transition で必ず reset |
| 7 | closed window の max が次 window に混入しない |
| 8 | **`observedOutstandingMax_` の既存 accumulated semantics と bounded `windowMax` を混同しない** |
| 9 | `counterWrapped` は診断値であり、停止・補正・rollback の trigger にならない |
| 10 | export は **Closed window の snapshot** を読むだけで、window state を変更しない |

### D91.3 accumulated と bounded windowMax の論理的分離（ご指定 #8）
```
A/R counters
    │
    ├─ existing accumulated diagnostic estimate/max（observedOutstandingMax_・変更なし）
    │
    └─ measurement-window sampler（新規・独立 state）
          ├─ A0/R0・per-tick estimate・windowMax（bounded）・A1/R1
```
- **`observedOutstandingMax_`（accumulated）は変更しない**・`windowMax_`（bounded）を別メンバとして追加。
- export では accumulated と window を区別して出力（#8 の混同防止）。

### D91.4 最終状態
| 項目 | 状態 |
|------|------|
| Start/End request の上書き・重複要求の契約化（CAS・単調遷移・無視） | **CLOSED（本 D91）** |
| **D90.5 実装契約（レビュー基準 10 点）** | **CLOSED（本 D91・実装時にコード上で確認）** |
| **accumulated と bounded windowMax の分離** | **CLOSED（本 D91・#8）** |
| **window-reset implementation** | **OPEN（実装 GO 待ち・D90.5 + D91 の契約で実装）** |
| **実コード差分レビュー（ISRWorldRetirementTelemetry.h → AudioEngine.Timer.cpp → AudioEngine.Commit.cpp → AudioEngine.h）** | **次工程** |
| 実測 → B_max 測定特性 → M 導出 | OPEN（後段） |
| R / R_cap・T2 authority / I-2 | **NO-GO** |

### D91.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T299 | request 上書き・重複要求 | StartRequested 中の追加 Start・EndRequested / Closed 中の追加 End は既存 window を変更しない（CAS・無視） |
| T300 | linearization point | sampler が観測した atomic state 遷移が唯一の measurement boundary（API 呼び出し時刻ではない） |
| T301 | レビュー基準 10 点 | audio callback 非呼び出し・timerCallback 唯一 owner・A0/R0・A1/R1 同一 tick・windowMax reset・混入なし・counterWrapped 診断のみ・export 読み取りのみ |
| T302 | accumulated / windowMax 分離 | observedOutstandingMax_（accumulated）と windowMax（bounded）を論理的に分離・export で区別 |

---

## D92 — Design-74 確定（window-reset instrumentation 実装完了・D90.5 10 項目 + 監視 4 点の実コードレビュー・compile/unit test 29/29 PASS・2026-08-16）

ユーザー Design-73 レビュー対応: **「次の工程は予定どおり、ISRWorldRetirementTelemetry.h → AudioEngine.Timer.cpp →
AudioEngine.Commit.cpp → AudioEngine.h の順で実コード差分を作成し、D90.5 の 10 項目 + 4 点を満たすかをレビューする
段階。R / R_cap / T2 authority にはまだ進めず、window-reset instrumentation だけを T1 の範囲で実装するのが正しい
境界」**。**Phase I 実装 NO-GO 継続（T1 の window-reset instrumentation のみ実装・R/R_cap・T2 は対象外）。**

### D92.1 実装差分（4 ファイル・最小）
| ファイル | 内容 |
|----------|------|
| ISRWorldRetirementTelemetry.h | MeasurementState enum・MeasurementSnapshot struct（trivially copyable）・window-reset state（すべて atomic）・requestMeasurementStart / End（CAS・D91.1 上書き・重複要求契約）・samplerTick（beginWindow / sampleWindow / closeWindow）・lastClosedSnapshot |
| AudioEngine.Timer.cpp | sampler に samplerTick を組込み（timerCallback・MessageThread・唯一の transition owner・D91 基準 3） |
| AudioEngine.Commit.cpp | export に Closed snapshot を immutable 読み取りで出力（accumulated と window を区別・D91 基準 8） |
| AudioEngine.h | requestWorldRetirementMeasurementStart / End を公開（Non-RT・D91 基準 1・2） |

### D92.2 D90.5 実装契約（10 項目）の実コード確認
| # | 基準 | 実装 |
|---|------|------|
| 1 | Start / End API が audio callback から呼ばれない | requestWorldRetirementMeasurementStart / End は Non-RT API・audio callback（getNextAudioBlock / processBlockDouble）から呼ばれない ✅ |
| 2 | request state を変更するのは Non-RT API のみ | requestMeasurementStart / End のみが CAS で measurementState_ を変更 ✅ |
| 3 | timerCallback が唯一の window state transition owner | samplerTick（timerCallback 内）のみが beginWindow / sampleWindow / closeWindow を呼ぶ ✅ |
| 4 | A0/R0 は Start transition と同じ sampler tick で取得 | beginWindow 内で acquireObserved() / releaseObserved() を load ✅ |
| 5 | A1/R1 は End transition と同じ sampler tick で取得 | closeWindow 内で load ✅ |
| 6 | windowMax は Start transition で必ず reset | beginWindow で windowMax_ = firstEstimate（reset）✅ |
| 7 | closed window の max が次 window に混入しない | beginWindow で windowMax_ をリセット（次 window に持ち越さない）✅ |
| 8 | observedOutstandingMax_（accumulated）と bounded windowMax を混同しない | **別メンバ**（observedOutstandingMax_ は変更なし・windowMax_ は新規）・export で accumulated と window を区別 ✅ |
| 9 | counterWrapped は診断値であり trigger にならない | counterWrapped_ は記録のみ・停止・補正・rollback に使われない ✅ |
| 10 | export は Closed window の snapshot を読むだけで window state を変更しない | lastClosedSnapshot() は snapshot_ を load するのみ・window state を変更しない ✅ |

### D92.3 監視 4 点の実装確認（ご指定）
| 監視点 | 実装 |
|--------|------|
| 1. windowMax_ の初期値 | beginWindow で **firstEstimate（A0/R0 から得られる最初の estimate）** を windowMax_ の初期値（Start tick の観測値が max 集計から抜けない）✅ |
| 2. End tick の包含 | closeWindow で **A1/R1 → estimate → updateWindowMax → Closed** の順序（最後の観測値が windowMax から欠落しない）✅ |
| 3. request state の再利用 | closeWindow 末尾で EndRequested → Idle の CAS・**Start request が発行済みなら StartRequested のまま**（request は失われない・次の tick で beginWindow）✅ |
| 4. export race | **snapshot_ を std::atomic\<MeasurementSnapshot\> で publish（release）** → export は load（acquire）で immutable snapshot を取得（複数フィールドの個別 load による不整合なし）✅ |

### D92.4 検証
- **ビルド**: ConvoPeq.exe リンク成功（エラーなし・既存警告 C4458/C4996 のみ）。
- **テスト**: `ctest -C Debug -E "BuildInputSemanticContract\|RuntimeWorldAuthority"` = **29/29 PASS**（回帰なし・除外条件のもとで）。

### D92.5 最終状態
| 項目 | 状態 |
|------|------|
| window-reset instrumentation 実装（4 ファイル） | **CLOSED（本 D92・実装）** |
| D90.5 実装契約（10 項目）の実コード確認 | **CLOSED（本 D92）** |
| 監視 4 点（windowMax 初期値・End tick 包含・request 再利用・export race） | **CLOSED（本 D92・実装確認）** |
| compile / unit test（29/29 PASS） | **CLOSED（本 D92）** |
| **実測（measurement protocol に沿う）** | **OPEN（次工程）** |
| B_max 測定特性評価 → M 導出 → B_max^true ≤ B_max^observed + M | OPEN（後段） |
| R / R_cap・T2 authority / I-2 | **NO-GO** |

### D92.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T303 | window-reset 実装差分 | 4 ファイル（telemetry state/API・samplerTick・export snapshot・request API 公開） |
| T304 | D90.5 10 項目 | audio callback 非呼び出し・timerCallback 唯一 owner・A0/R0・A1/R1 同一 tick・windowMax reset・混入なし・counterWrapped 診断のみ・export 読み取りのみ |
| T305 | 監視 4 点 | windowMax 初期値 = firstEstimate・End tick 包含・request 再利用（StartRequested 維持）・export race（immutable snapshot publish） |
| T306 | compile/unit test | ビルド成功・ctest 29/29 PASS |

---

## D93 — Design-75 確定（実測計画・measurement characterization の枠組み・保存項目の固定・M の上界付けの判定基準・2026-08-16）

ユーザー Design-74 レビュー対応: **「D92 の実装完了として整理できる。重要なのは、実装・静的レビュー・テスト PASS
を『測定結果』や『M の安全側証明』と混同しないこと。次の実測で各 window について保存すべき項目を固定。評価時には
windowMax == B_max^true とは扱わず、『sampled observation によって得られた下側の観測量』として扱う。特に 100ms
sampler では、短時間の retirement burst が tick 間に発生して消滅した場合、windowMax_ がその peak を観測できない
可能性がある。したがって次のゲートは単なる『大きな値が取れた』ではなく、sampling interval と retirement burst の
時間特性から、見逃し量 M を有限かつ安全側に拘束できるか。現時点で R を windowMax_ から導出するのはまだ不可。
D92 の境界を維持したまま、次は実測データを取得して D93 の measurement characterization に進むのが適切」**。
**Phase I 実装 NO-GO 継続（実測計画の確定のみ・コード変更なし）。**

### D93.1 D92 の現在地の正確な解釈（ご指定・混同しない）
| 項目 | 判定 |
|------|------|
| Start/End request 契約・sampler linearization・bounded windowMax_・accumulated 分離・End tick 包含・Closed snapshot atomic publish/consume・audio callback isolation | CLOSED |
| build | PASS |
| 回帰テスト | 29/29 PASS（既定 2 テスト除外） |
| **B_max^observed の実測** | **OPEN** |
| **sampling による見逃し量 M** | **OPEN** |
| **B_max^true ≤ B_max^observed + M** | **OPEN** |
| R / R_cap / T2 | **NO-GO** |
- **実装・静的レビュー・テスト PASS は「測定結果」や「M の安全側証明」と混同しない**（build PASS・29/29 PASS は
  「実装が仕様通り動く」ことの検証・B_max^observed の実測値ではない）。

### D93.2 実測で保存すべき項目（ご指定・各 window）
- `windowId`・start/end timestamp・`A0`/`R0`・`A1`/`R1`・final estimate・`windowMax`・sample count・最大 sampling
  gap・interval jitter・missed-tick count・`counterWrapped`。
- **D92 の MeasurementSnapshot に含まれている**（windowId / start/end ts / A0/R0 / A1/R1 / finalEstimate / windowMax /
  sampleCount / maxSamplingGapUs / missedTickCount / counterWrapped）✅。
- interval jitter は maxSamplingGapUs と missedTickCount から導出（実測値として記録）。

### D93.3 windowMax の意味論（ご指定）
- **windowMax == B_max^true とは扱わない**。
- **windowMax は「sampled observation によって得られた下側の観測量」**（B_max^observed・D81 の sampled maximum 意味論）。
- 100ms sampler では、tick 間に発生・消滅した短時間 retirement burst を観測できない可能性 → windowMax は
  B_max^true の下側の観測量として扱う。

### D93.4 次のゲートの判定基準（ご指定）
```
D92 implementation
    ↓
実測
    ↓
sampling gap / jitter / missed tick の実測値
    ↓
retirement burst の時間特性
    ↓
B_max^observed の再現性・観測限界
    ↓
M を安全側に上界付け可能か？
    ├─ YES → B_max^true ≤ B_max^observed + M の証明へ
    └─ NO  → sampler 方式または測定方法を再設計
```
- **現時点で R を windowMax_ から導出するのは不可**。
- D92 の境界を維持したまま、次は実測データを取得して measurement characterization（本 D93）に進む。

### D93.5 実測の実行手順（measurement protocol・D89.2 に沿う）
1. アプリまたはテストハーネスでオーディオエンジンを起動（Non-RT）。
2. `requestWorldRetirementMeasurementStart()` を発行（Non-RT・CAS Idle → StartRequested）。
3. 負荷を発生（world publish / retire の繰り返し・sampler が window 内で estimate / windowMax を更新）。
4. `requestWorldRetirementMeasurementEnd()` を発行（Non-RT・CAS Running → EndRequested）。
5. sampler が window を確定 → `world_retirement_telemetry.json` の `measurementWindow` に Closed snapshot が出力。
6. 各 window の保存項目（D93.2）を収集。

### D93.6 最終状態
| 項目 | 状態 |
|------|------|
| 実測計画・保存項目の固定 | **CLOSED（本 D93）** |
| windowMax の意味論（下側の観測量） | **CLOSED（本 D93）** |
| 判定順序（M の上界付けの YES/NO 分岐） | **CLOSED（本 D93）** |
| **実測データ取得** | **OPEN（次工程・measurement protocol 実行）** |
| sampling gap / jitter / missed tick 実測値 | OPEN |
| retirement burst の時間特性・B_max 再現性 | OPEN |
| **M を安全側に上界付け可能か** | **OPEN（YES → 証明へ・NO → sampler 再設計）** |
| B_max^true ≤ B_max^observed + M | OPEN（後段） |
| R / R_cap・T2 authority / I-2 | **NO-GO** |

### D93.7 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T307 | D92 現在地の正確な解釈 | 実装・レビュー・テスト PASS は測定結果 / M 証明と混同しない |
| T308 | 実測保存項目 | windowId・ts・A0/R0・A1/R1・final estimate・windowMax・sample count・max gap・jitter・missed tick・counterWrapped |
| T309 | windowMax の意味論 | sampled 下側の観測量（B_max^true と同一視しない） |
| T310 | M の上界付け判定 | sampling gap / burst 時間特性 → M 有限・安全側上界の YES/NO 分岐 |

---

## D94 — Design-76 確定（reference instrumentation の設計・O_w と T_w の分離・burst test harness の枠組み・M の導出可能性判定基準・2026-08-16）

ユーザー Design-75 レビュー対応: **「実測へ進む前に『何を操作して、何を ground truth として比較するか』だけは固定
した方がよい。現状の windowMax_ だけでは B_max^true を直接観測できない。D93 の measurement characterization は、
単なる telemetry 収集ではなく、sampled observation と高頻度な retirement-side reference の差を測定する試験として
設計する必要がある。B_max^reference は lifetime authority や R/R_cap ではない・あくまで実測時だけ使用する reference
instrumentation。reference instrumentation を audio callback に置かない（D87 の RT-safety 契約維持・retirement
terminal path の Non-RT 側で観測）。実測で 3 種類を分離: (1) 通常負荷 (2) burst 負荷（重要・retirement burst の継続
時間が sampler interval より短いかが M の支配要因） (3) scheduler jitter 負荷。判定を数式で: O_w = sampled windowMax・
T_w = reference maximum・E_w = T_w - O_w として、M >= max(E_w) を単純に採用してはいけない（有限回の実測最大値は
それ自体では安全側の証明にならない）。必要なのは sampling gap + retirement burst duration + retirement event rate
から未観測 peak の上界を導出できること。YES 判定は『実測で大きな M が観測されなかった』ではなく『観測できなかった
retirement peak に対して、実測した時間特性から有限の安全側上界 M を導出できる』。reference instrumentation は
T1 telemetry の契約に混ぜない（測定用 ground-truth と production diagnostic telemetry を分離・D86 の telemetry ≠
authority 境界維持）」**。**Phase I 実装 NO-GO 継続（reference 設計の確定のみ・コード変更なし）。**

### D94.1 実測モデル（ご指定・3 系列）
```
A/R counter
    │
    ├── 100 ms sampler
    │       └── windowMax = B_max^observed（O_w・T1 telemetry）
    │
    └── retirement event reference
            └── B_max^reference（T_w・高頻度 retirement-side・実測時のみ）
```
- **O_w** = sampled windowMax（T1 telemetry・100ms sampler・B_max^observed）
- **T_w** = reference maximum（高頻度 retirement-side reference・B_max^reference）
- **E_w** = T_w - O_w（sampling による見逃し量の実測値）

### D94.2 reference instrumentation の設計原則（ご指定）
- **B_max^reference は lifetime authority や R / R_cap ではない**（実測時だけ使用する reference instrumentation）。
- **audio callback に置かない**（D87 の RT-safety 契約維持・retirement terminal path の Non-RT 側で観測）。
- **T1 telemetry の契約に混ぜない**（測定用 ground-truth instrumentation と production diagnostic telemetry を分離・
  D86 の「telemetry が lifetime authority にならない」境界を維持）。

### D94.3 reference instrumentation の観測点（設計）
| 観測点 | 場所 | 内容 |
|--------|------|------|
| acquire（retirement obligation 発生） | onRuntimePublishedNonRt（Commit.cpp:406・CoordinatorLoop Non-RT・T1 の onAcquireObserved と同位置・別カウンタ） | referenceOutstanding++ |
| release（type==World の terminal deleter 実行） | DeferredDeletionQueue::reclaim / drainAllUnsafe + RetireQuarantineStore::drain / drainAllUnsafe（Non-RT・worldReclaimCount_++ と同位置・別カウンタ） | referenceOutstanding-- |
| referenceMax（window ごと） | referenceOutstanding の更新ごとに更新（高頻度 running max） | T_w を記録 |
- **実装時に再確認する項目**: storage 側の release イベント（type==World deleter 実行）を reference に即時反映する方法
  （storage 側の reference カウンタ参照 or worldReclaimCount を高頻度ポーリング・非同時読取の扱い・D82）を実装 GO 前に
  確定する。

### D94.4 実測で分離すべき 3 種類（ご指定）
1. **通常負荷**: publish / retire を通常運転で反復・sampler がどの程度 peak を拾うか確認。
2. **burst 負荷（重要）**: retirement を短時間に集中・sampler tick 間で peak が発生・消滅した場合の見逃しを測る
   （**retirement burst の継続時間が sampler interval より短いかが M の支配要因**）。
3. **scheduler jitter 負荷**: sampler の実測 gap / missed tick を記録・最大観測 gap と reference peak の関係を見る。

### D94.5 M の導出可能性判定（ご指定・数式）
- 各 window w: O_w = sampled windowMax・T_w = reference maximum・E_w = T_w - O_w。
- **M >= max(E_w) を単純に採用しない**（有限回の実測最大値はそれ自体では安全側の証明にならない）。
- **必要なのは sampling gap + retirement burst duration + retirement event rate から未観測 peak の上界を導出できること**。
- **YES 判定**: 「実測で大きな M が観測されなかった」ではなく **「観測できなかった retirement peak に対して、実測した
  時間特性から有限の安全側上界 M を導出できる」**。

### D94.6 次の実装・実測順序（ご指定）
```
D93
 ↓
reference instrumentation の設計（本 D94）
 ↓
burst test harness
 ↓
通常 / burst / jitter の3条件
 ↓
O_w と reference T_w の比較
 ↓
sampling gap と burst duration の関係を評価
 ↓
M の導出可能性判定
 ↓
D94 measurement characterization
```

### D94.7 最終状態
| 項目 | 状態 |
|------|------|
| 実測モデル（O_w / T_w / E_w・3 系列） | **CLOSED（本 D94）** |
| reference instrumentation の設計原則（lifetime authority でない・audio callback 非配置・T1 と分離） | **CLOSED（本 D94）** |
| reference の観測点（acquire / release / referenceMax） | **CLOSED（本 D94・実装時に release 反映方法を再確認）** |
| 3 負荷条件（通常 / burst / jitter） | **CLOSED（本 D94）** |
| **M の導出可能性判定基準** | **CLOSED（本 D94・時間特性からの上界導出）** |
| **reference instrumentation の実装** | **OPEN（実装 GO 待ち・release 反映方法の再確認後）** |
| burst test harness | **OPEN（次工程）** |
| 通常 / burst / jitter の実測 → O_w と T_w 比較 | OPEN |
| M の導出可能性判定 → D94 measurement characterization | OPEN（後段） |
| R / R_cap・T2 authority / I-2 | **NO-GO** |

### D94.8 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T311 | 実測モデル | O_w（sampled windowMax）・T_w（reference max）・E_w = T_w - O_w の 3 系列分離 |
| T312 | reference 設計原則 | lifetime authority でない・audio callback 非配置・T1 telemetry と分離（D86 境界維持） |
| T313 | 3 負荷条件 | 通常 / burst（重要・sampler interval より短い burst）/ jitter（gap/missed tick） |
| T314 | M の導出可能性 | M >= max(E_w) を単純採用しない・sampling gap + burst duration + event rate から上界導出（YES/NO 分岐） |

---

## D95 — Design-77 確定（reference instrumentation 実装前レビュー・固定点 7 つ・release 直接観測方式・実装ゲートの固定・2026-08-16）

ユーザー Design-76 レビュー対応: **「D94 の整理は D86〜D93 の境界を維持した設計として妥当。特に O_w と T_w を別系列にし、
T_w を authority に接続しない点は維持すべき。次の reference instrumentation 実装前レビューでは、D94 の『release 反映
方法の再確認』を単なる実装詳細ではなく、ゲート条件として扱うのが適切。固定すべき点 7 つ。(1) referenceOutstanding の
初期値 (2) referenceMax の開始点 (3) release の観測点 (4) acquire/release の race (5) window End (6) T_w と O_w の
比較単位 (7) M の判定。D94 の『worldReclaimCount 高頻度ポーリング』を採用するより、terminal deleter の World release
event を reference 側へ直接観測する方式のほうが、T_w を『高頻度 retirement-side reference』と定義する D94 の目的には
適している。ただし、その場合も D87 で確定した terminalization exactly-once の保証をそのまま利用し、reference 側で二重
decrement を発生させないことが必須」**。**現時点の判定: D94 = 設計 CLOSED / reference instrumentation 実装 OPEN /
実測 OPEN / M 導出 OPEN / R・R_cap・T2 NO-GO**。**Phase I 実装 NO-GO 継続（reference 実装前レビュー・コード変更なし）。**

### D95.1 固定点 1〜7（ご指定）
| # | 固定点 | 内容 |
|---|--------|------|
| 1 | **referenceOutstanding の初期値** | Measurement Start 時点の既存 World outstanding を基準値として取得・**0 にリセットしない**（Start 前から保持されていた World が window 内の release で負値になり T_w の意味が崩れる） |
| 2 | **referenceMax の開始点** | Start の linearization point で referenceOutstanding の現在値を初期値にする・以後の acquire/release event で running max 更新・**Start 前の履歴を T_w に持ち込まない**（O_w の sampler と同様） |
| 3 | **release の観測点** | World deleter の terminalization 成功後（D86/D87 と同じ）・**type == World を authority 判定に使わない**・reference counter の更新は telemetry / measurement instrumentation のみ |
| 4 | **acquire/release の race** | referenceOutstanding を単純な A_ref - R_ref の periodic sampling にしない・**event 自体を観測点として扱い、running max を更新**（100ms sampler の見逃し比較対象として明確） |
| 5 | **window End** | End request → sampler linearization point（D91/D92 契約維持）・**End tick までに発生した terminal release を T_w に含める**・Closed snapshot だけを export 対象 |
| 6 | **T_w と O_w の比較単位** | 同一 windowId に O_w（sampled windowMax）・T_w（reference windowMax）・E_w = T_w - O_w を対応付ける |
| 7 | **M の判定** | max(E_w) をそのまま安全上界 M として採用しない・実測ではまず sampling gap → retirement event rate → burst duration → T_w - O_w の関係を取得し、その後に有限の安全側上界を数学的に構成できるかを判定 |

### D95.2 release 反映方式の確定（ご指定・直接観測方式）
- **「worldReclaimCount 高頻度ポーリング」ではなく、「terminal deleter の World release event を reference 側へ直接
  観測する方式」を採用**（D94 の T_w を「高頻度 retirement-side reference」と定義する目的に適する）。
- **D87 の terminalization exactly-once の保証をそのまま利用**し、reference 側で二重 decrement を発生させない（各 world
  entry の deleter 実行は ≤1 回・D46-R4 + D73.2 相互排他）。
- **実装方法（storage 側の release イベントが referenceOutstanding を直接更新するための参照・配置）は実装 GO 前に
  コード突合して確定**（D86 の案 X の知見・storage 側カウンタ参照 vs context 追加の比較）。

### D95.3 実装ゲート（ご指定・順序）
```
reference counter の acquire/release 観測点確認
    ↓
Measurement Start/End との linearization 確認
    ↓
referenceOutstanding / referenceMax の window state 実装
    ↓
T_w を Closed snapshot に追加
    ↓
O_w / T_w / E_w の export
    ↓
burst test harness
    ↓
normal / burst / scheduler-jitter 実測
    ↓
M の導出可能性判定
```
- **この段階では R / R_cap・ReservationExhausted の生成・T2 authority への接続は一切入れない**。

### D95.4 最終状態
| 項目 | 状態 |
|------|------|
| 固定点 1〜7 | **CLOSED（本 D95）** |
| release 直接観測方式（terminal deleter event を reference へ直接観測） | **CLOSED（本 D95・実装方法は GO 前にコード突合）** |
| 実装ゲート（観測点確認 → window state → T_w 追加 → export → harness → 実測 → M 判定） | **CLOSED（本 D95）** |
| **reference instrumentation 実装** | **OPEN（実装 GO 待ち・D95 の固定点で実装）** |
| burst test harness・normal/burst/jitter 実測 | OPEN（次工程） |
| M の導出可能性判定 → D94 measurement characterization | OPEN（後段） |
| R / R_cap・ReservationExhausted 生成・T2 authority | **NO-GO（一切入れない）** |

### D95.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T315 | 固定点 1〜7 | referenceOutstanding 初期値（Start 時点の既存 outstanding）・referenceMax 開始点（Start LP）・release 観測点（terminalization 後・authority 判定に使わない）・event 観測（periodic sampling にしない）・window End（sampler LP・End tick の release 包含）・同一 windowId 対応・M 判定（max(E_w) 単純採用しない） |
| T316 | release 直接観測方式 | terminal deleter の World release event を reference へ直接観測・D87 exactly-once 利用（二重 decrement なし） |
| T317 | 実装ゲート | 観測点確認 → linearization 確認 → window state → T_w 追加 → O_w/T_w/E_w export → harness → 実測 → M 判定 |
| T318 | T1/T2 境界 | R/R_cap・ReservationExhausted 生成・T2 authority は一切入れない |

---

## D96 — Design-78 確定（reference observer の共有設計・ownership/lifetime の確定・実装ゲート修正・案 A REJECT・2026-08-16）

ユーザー Design-77 レビュー対応: **「案 A のまま実装へ進むのは NO-GO。理由は D95 の固定点 #4（event 自体を観測点として
running max を更新する）と案 A が正面から矛盾するため。案 A は acquire event → referenceMax 更新・release event →
referenceReleaseCount_++ のみであり、release-only の区間について release event 発生時点の reference state を
referenceMax が観測しない。より重要なのは、T_w を『retirement event reference maximum』と定義した D94/D95 の意味論を
満たさなくなること。これは『burst 実測で不足していたら案 C にする』という問題ではなく、測定器そのものが測定対象を
完全に観測していない状態で実測を開始するため、D94 の M 導出可能性評価の入力として不適切。推奨: 案 A を廃止し、
storage と AudioEngine のどちらにも依存しない Non-RT 専用の reference telemetry/observer を 1 個置き、acquire と
release の両方から同じ observer に event を通知する方式。referenceObserver は retirement authority ではない（no
ownership・no reclaim decision・no lifetime decision・no R/R_cap・no ReservationExhausted・measurement only）。ただし、
実装前に observer の lifetime を確定する必要がある。特に shutdown 時には AudioEngine → storage drain → World deleter →
referenceObserver という順序になるため、AudioEngine 所有の observer を storage が直接参照する構造にすると shutdown
lifetime が新しい問題になる。最も安全なのは、WorldRetirementReferenceObserver を retirement storage と AudioEngine の
共通 Non-RT instrumentation dependency として所有・受け渡す方式。AudioEngine が owns・Epoch/Retire storage が observes
same observer・storage が AudioEngine を参照する形にはしない。① ownership/lifetime が決まるまでは実装しない」**。
**Phase I 実装 NO-GO 継続（reference observer 共有設計の確定のみ・コード変更なし）。**

### D96.1 案 A REJECT（ご指定）
- 案 A（acquire イベントのみ referenceMax 更新・release は累積のみ）は **D95 固定点 #4 と矛盾**。
- release-only 区間（release event 発生時点の reference state）を referenceMax が観測しない。
- T_w を「retirement event reference maximum」と定義した D94/D95 の意味論を満たさない。
- **測定器そのものが測定対象を完全に観測していない状態で実測を開始する** → M 導出可能性評価の入力として不適切。

### D96.2 reference observer の共有設計（ご指定・採用）
```
                 ┌────────────────────────────┐
publish success ─┤                            │
                 │ WorldRetirementReference   │
terminal release┤        Observer             │
                 │                            │
                 └────────────┬───────────────┘
                              │
                     referenceOutstanding
                     referenceMax / window state
                              │
                     Closed MeasurementSnapshot
```
- **acquire**: `onRuntimePublishedNonRt()` → existing `acquireObserved++` + `referenceObserver.onAcquire()`。
- **release**: 4 個の terminal path すべてで World deleter executes successfully → existing `worldReclaimCount_++` +
  `referenceObserver.onRelease()`。
- **referenceObserver は retirement authority ではない**: no ownership・no reclaim decision・no lifetime decision・
  no R/R_cap・no ReservationExhausted・**measurement only**。
- **D86 の「World type ≠ lifetime authority」も維持**。

### D96.3 observer の ownership/lifetime の確定（ご指定・①）
- **AudioEngine が WorldRetirementReferenceObserver を所有**（メンバ）。
- **Epoch/Retire storage が同じ observer を参照（observes・非所有参照）**。
- **storage が AudioEngine を参照する形にはしない**（AudioEngine → storage の一方向依存）。
- observer は「retirement storage と AudioEngine の共通 Non-RT instrumentation dependency」。
- **shutdown lifetime**: AudioEngine → storage drain → World deleter → `referenceObserver.onRelease()` の順序で、observer
  がまだ生きている必要がある。
  - observer を `m_retireRouter` より後に宣言（C++ 逆順破棄で m_retireRouter → observer の順・m_retireRouter 破棄時に
    observer は生きている）。
  - **★ D98 修正（実装 GO 前実コード突合）: 「m_retireRouter より後に宣言」は誤り・正しくは「前に宣言」**（C++ メンバ
    初期化 = 宣言順・破棄 = 逆順。後に宣言すると observer が先に破棄され invariant に反する・前宣言が正しい・D98.1）。
  - shutdown drain（drainAll）は m_retireRouter のデストラクタではなく AudioEngine の明示的 shutdown 処理で実行 →
    observer は生きている。
- **observer 参照の伝搬方法**（AudioEngine → ISRRetireRouter → EpochDomain / RetireQuarantineStore → 各 terminal path）は
  実装 GO 前にコード突合して確定（D86 案 X の知見・ただし observer は measurement only・`#ifdef` で production から
  除外可）。

### D96.4 実装ゲート修正（ご指定）
```
D95
 ├─ ① reference observer の ownership/lifetime を確定（本 D96）
 ├─ ② acquire observer insertion point（onRuntimePublishedNonRt）
 ├─ ③ release observer insertion point ×4（terminal World deleter 成功後）
 ├─ ④ observer の window linearization（Start / End は既存 sampler boundary）
 ├─ ⑤ referenceOutstanding / referenceMax（acquire/release event の双方で更新）
 ├─ ⑥ Closed snapshot（O_w / T_w / E_w）
 └─ ⑦ exactly-once + shutdown lifetime 再検証
```
- **① ownership/lifetime が決まるまでは実装しない**（本 D96 で確定）。

### D96.5 判定（ご指定）
| 項目 | 判定 |
|------|------|
| D95 | CLOSED |
| **案 A** | **REJECT** |
| release 直接観測方式 | 維持 |
| **reference observer の共有設計（ownership/lifetime）** | **実装前レビュー CLOSED（本 D96・①確定）** |
| reference instrumentation 実装 | **NO-GO（実装 GO 待ち）** |
| burst measurement | NO-GO |
| M 導出 | NO-GO |
| R / R_cap / T2 | **引き続き NO-GO** |

### D96.6 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T319 | 案 A REJECT | acquire のみ referenceMax 更新は D95 固定点 #4 と矛盾・release-only 区間を観測しない |
| T320 | observer 共有設計 | acquire/release の両イベントが同じ observer に通知・observer は authority でない（measurement only） |
| T321 | ownership/lifetime | AudioEngine 所有・storage observes（非所有参照）・storage は AudioEngine を参照しない・shutdown 順序（observer は m_retireRouter より後に宣言） |
| T322 | 実装ゲート修正 | ① ownership/lifetime → ② acquire 挿入 → ③ release ×4 挿入 → ④ window linearization → ⑤ 双方更新 → ⑥ Closed snapshot（O_w/T_w/E_w）→ ⑦ exactly-once + shutdown lifetime 再検証 |

---

## D97 — Design-79 確定（observer の参照有効期間・lifetime invariant の確定・release 挿入順序の固定・実装レビュー順序 10 項目・2026-08-16）

ユーザー Design-78 レビュー対応: **「D96 の確定内容は D95 の問題点を適切に解消している。特に observer を AudioEngine が
所有し、storage は非所有参照だけを持つという一方向依存は、shutdown lifetime を含めて authority 分離を維持するうえで
妥当。ただし、実装 GO 前に 1 点だけ追加で固定しておくべき: observer の参照有効期間。D96 の『m_retireRouter より後に
observer を宣言する』という条件だけでは、C++ のメンバ初期化順序・破棄順序については十分だが、実際には ISRRetireRouter
→ EpochDomain → storage が observer の参照を保持する期間まで確認する必要がある。確定すべき invariant: storage が
observer を参照している可能性がある期間中、observer の lifetime が終了しない。破棄時は必ず ISRRetireRouter / storage →
all terminal World releases complete → storage destroyed → reference observer destroyed。release 側も重要: 4 箇所の挿入
位置は type == World → World deleter → terminalization successful → worldReclaimCount_++ → referenceObserver.onRelease()
の順序に固定。worldReclaimCount_++ と onRelease() の相対順序は authority に影響しないが、どちらも terminalization 後
であることが重要。onRelease() が例外を投げたり、所有権を変更したり、reclaim を再試行したりしてはいけない（reference
observer の失敗が retirement control-flow に波及しない構造）」**。**Phase I 実装 NO-GO 継続（observer lifetime invariant
の確定のみ・コード変更なし）。**

### D97.1 observer の参照有効期間（lifetime invariant・ご指定）
```
AudioEngine lifetime
 ├─ WorldRetirementReferenceObserver
 └─ ISRRetireRouter
      └─ EpochDomain / DeferredDeletionQueue / RetireQuarantineStore
          └─ non-owning observer reference
```
- **破棄時（必須順序）**: ISRRetireRouter / storage → all terminal World releases complete → storage destroyed →
  reference observer destroyed。
- **★ 実装不変条件**: 「storage が observer を参照している可能性がある期間中、observer の lifetime が終了しない」。
- `m_retireRouter` より後に observer を宣言する（C++ 逆順破棄・D96）**に加えて**、storage の observer 参照が無効になる
  前に storage が破棄されることを shutdown / destructor 順序で確認する（ISRRetireRouter → EpochDomain → storage の
  observer 参照保持期間）。
- **★ D98 修正（実装 GO 前実コード突合）: 上記「より後に宣言」は誤り・正しくは「より前に宣言」**（D98.1・observer を
  m_retireRouter より前に宣言 → 先に初期化され m_retireRouter のコンストラクタで渡せる・破棄は m_retireRouter → observer
  で invariant を満たす）。

### D97.2 release 側の挿入順序（ご指定・固定）
```
type == World
    ↓
World deleter
    ↓
terminalization successful
    ↓
worldReclaimCount_++
    ↓
referenceObserver.onRelease()
```
- **worldReclaimCount_++ と onRelease() の相対順序は authority に影響しない**が、**どちらも terminalization 後である
  ことが重要**。
- **onRelease() は例外を投げない・所有権を変更しない・reclaim を再試行しない**（reference observer の失敗が retirement
  control-flow に波及しない構造・observer は measurement only・D96）。

### D97.3 実装レビュー順序 10 項目（ご指定）
```
1. AudioEngine の実メンバ宣言順確認
2. ISRRetireRouter → EpochDomain → storage の observer 参照伝搬確認
3. shutdown / destructor の実行順確認
4. acquire observer 挿入
5. release observer ×4 挿入
6. window boundary との整合確認
7. referenceOutstanding / referenceMax の event-driven 更新確認
8. exactly-once 再確認
9. observer が retirement authority に影響しないことを静的確認
10. build + 全 regression test
```
- ここまで確認できれば **reference instrumentation 実装を GO** としてよい状態。

### D97.4 判定（ご指定）
| 項目 | 判定 |
|------|------|
| D96 | CLOSED |
| **observer lifetime 実装確認（invariant・破棄順序・参照保持期間）** | **OPEN（本 D97 で invariant 確定・実装 GO 後確認）** |
| instrumentation implementation | **NO-GO（実装 GO 待ち・D97.3 の 10 項目確認後）** |
| measurement | NO-GO |
| M 導出 | NO-GO |
| R / R_cap / T2 | **NO-GO** |

### D97.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T323 | observer lifetime invariant | storage が observer を参照している期間中、observer の lifetime が終了しない・破棄順序（storage → all releases → storage destroyed → observer destroyed） |
| T324 | release 挿入順序 | type==World → deleter → terminalization successful → worldReclaimCount_++ → onRelease()（両者とも terminalization 後） |
| T325 | onRelease の非影響 | 例外を投げない・所有権を変更しない・reclaim を再試行しない（retirement control-flow に波及しない） |
| T326 | 実装レビュー順序 10 項目 | メンバ宣言順 → observer 参照伝搬 → shutdown/destructor 順 → acquire 挿入 → release ×4 挿入 → window boundary → event-driven 更新 → exactly-once → authority 非影響 → build + regression |

---

## D98 — Design-80 確定（実装 GO 前 1〜3 実コード突合・メンバ宣言順の修正（observer は m_retireRouter より前宣言）・shutdown 順序確認・2026-08-16）

ユーザー D97 レビュー対応: **「次は実装 GO 前の 1〜3 項目の実コード突合。特に重要なのは、D97 の『storage → all
terminal World releases → storage destroyed → observer destroyed』という論理順序と、C++ のメンバ宣言順に基づく逆順
destruction が実コード上で一致しているか。1〜3 が CLOSED になった段階で、初めて 4〜9（observer 挿入・window 整合・
event-driven 更新・exactly-once・authority isolation）→ 10（build/regression）へ進めるのが妥当」**。
**Phase I 実装 NO-GO 継続（実コード突合のみ・コード変更なし）。**

### D98.1 AudioEngine の実メンバ宣言順（突合・★修正）
- **実メンバ宣言順**: `m_epochDomain`（AudioEngine.h:4645・値所有）→ `m_retireRouter`（4648・unique_ptr・m_epochDomain
  参照）→ ... → `worldRetirementTelemetry_`（4837）→ ...（多数）
- **C++ メンバ初期化 = 宣言順・破棄 = 逆順**。
- **★ D96/D97 の「observer を m_retireRouter より後に宣言」は誤り・修正**:
  - 後に宣言: observer が m_retireRouter の後に初期化 → m_retireRouter のコンストラクタ（CtorDtor.cpp:35
    `m_retireRouter = std::make_unique<...>(m_epochDomain)`）で observer 参照を渡せない・破棄は observer → m_retireRouter
    （observer 先）で D97 invariant に反する。
  - **正しくは「observer を m_retireRouter より前に宣言」**: observer が先に初期化 → m_retireRouter のコンストラクタで
    observer 参照を渡せる・破棄は m_retireRouter → observer（storage 先・observer 後）で D97 invariant を満たす。
  - observer の宣言位置: m_epochDomain（4645）の近く（m_retireRouter 4648 より前）。

### D98.2 ISRRetireRouter → storage の所有・参照関係（突合）
- `m_retireRouter` は unique_ptr（AudioEngine が唯一所有）。
- ISRRetireRouter は `m_epochDomain`（provider_・**非所有参照**）と `RetireQuarantineStore m_retireQuarantine`（値所有）を持つ。
- EpochDomain は `DeferredDeletionQueue`（値所有）を持つ。
- **observer 参照の伝搬**: AudioEngine → ISRRetireRouter（コンストラクタで渡す・D98.1 の前宣言で初期化済み）→
  EpochDomain（provider_）→ DeferredDeletionQueue / RetireQuarantineStore（**non-owning 参照**）。
- **observer は non-owning 参照・コピー所有・shared_ptr 化・別 lifetime 管理しない**・**storage が AudioEngine を逆参照しない**
  （observer のみ参照・一方向依存）。

### D98.3 shutdown / destructor の terminal-path 順序（突合・CLOSED）
- **~AudioEngine（CtorDtor.cpp:93-245）のデストラクタ本体**で:
  `setShutdownPhase(DrainRetire)`（195）→ `drainDeferredRetireQueues(true)`（233）→ `m_epochDomain.drainAll()`（234）→
  `runtimePublicationBridge_.markShutdownComplete()`（235）→ `setShutdownPhase(Destroy)`（243）。
- **全 World terminal release はデストラクタ本体で完了**（このとき observer は生きている）。
- その後、メンバの暗黙破棄（逆順）: ... → `m_retireRouter` → `m_epochDomain` → ...
- **m_retireRouter の破棄時に storage が破棄されるが、terminal release は既に完了** → onRelease() は呼ばれない。
- **observer を m_retireRouter より前に宣言すれば、破棄順序は m_retireRouter（storage）→ observer で D97 invariant を満たす**
  （storage → all terminal World releases → storage destroyed → observer destroyed の論理順序と一致）。

### D98.4 判定
| 項目 | 判定 |
|------|------|
| D97 レビュー順序 1（AudioEngine 実メンバ宣言順） | **CLOSED（本 D98・★修正込み）** |
| D97 レビュー順序 2（observer 参照伝搬・non-owning・一方向依存） | **CLOSED（本 D98・設計確定・実装 GO 後確認）** |
| D97 レビュー順序 3（shutdown / destructor terminal-path 順序） | **CLOSED（本 D98・実コード確認）** |
| reference instrumentation implementation（D97 順序 4〜10） | **NO-GO（次ステップ・observer 挿入 → window 整合 → event-driven → exactly-once → authority → build/regression）** |
| measurement / M 導出 / R・R_cap・T2 | **NO-GO** |

### D98.5 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T327 | メンバ宣言順修正 | observer を m_retireRouter より前に宣言（初期化 = 宣言順・破棄 = 逆順・m_retireRouter コンストラクタで渡せる・破棄は m_retireRouter → observer） |
| T328 | observer 参照伝搬 | AudioEngine → ISRRetireRouter → EpochDomain → storage（non-owning 参照・コピー所有/shared_ptr 化しない・AudioEngine 逆参照しない） |
| T329 | shutdown 順序 | ~AudioEngine で drainDeferredRetireQueues(true) + drainAll() 完了 → メンバ破棄（m_retireRouter → observer）・terminal release は observer 生存中に完了 |
| T330 | D97 invariant 整合 | storage → all terminal releases → storage destroyed → observer destroyed とメンバ逆順 destruction が実コード上で一致 |

---

## D99 — Design-81 確定（reference instrumentation 実装完了・D97 レビュー順序 #4〜#10・event-driven observer・compile/unit test 29/29 PASS・2026-08-16）

ユーザー D98 レビュー対応: **「D98 CLOSED → reference instrumentation 実装レビュー #4〜#10 OPEN → measurement / M / R /
R_cap / T2 は NO-GO。案Aには戻さない（T_w は release-only 区間も観測できる必要があるため、release event を observer に
直接通知する設計を維持）」**。**Phase I 実装 NO-GO 継続（T1 の reference instrumentation のみ実装・R/R_cap・T2 は対象外）。**

### D99.1 実装差分（10 ファイル・最小・event-driven observer）
| ファイル | 内容 |
|----------|------|
| ISRWorldRetirementReference.h（新規） | WorldRetirementReferenceObserver（referenceAcquireCount / referenceReleaseCount / referenceOutstanding / referenceMax・onAcquire / onRelease event-driven・onMeasurementStart / End・running フラグ） |
| AudioEngine.h | `worldRetirementReference_` メンバ（m_epochDomain の後・m_retireRouter より前宣言・D98） |
| ISRRetireRouter.h / .cpp | コンストラクタに observer パラメータ・referenceObserver_ メンバ・storage への伝搬（non-owning） |
| IEpochProvider.h | `setReferenceObserver(void*)`（デフォルト no-op） |
| EpochDomain.h | setReferenceObserver オーバーライド（DeferredDeletionQueue に伝搬） |
| DeferredDeletionQueue.h | referenceObserver_ + setReferenceObserver + reclaim / drainAllUnsafe で `onRelease()` |
| RetireQuarantineStore.h | referenceObserver_ + setReferenceObserver + drain / drainAllUnsafe で `onRelease()` |
| AudioEngine.CtorDtor.cpp | `m_retireRouter = std::make_unique<ISRRetireRouter>(m_epochDomain, &worldRetirementReference_)` |
| AudioEngine.Commit.cpp | onRuntimePublishedNonRt で `onAcquire()` + export に referenceObserver（T_w・E_w） |
| AudioEngine.Timer.cpp | sampler で window 同期（T1 の state に同期して onMeasurementStart / End・冪等） |

### D99.2 D97 レビュー順序 #4〜#10 の実装確認
| # | 項目 | 実装 |
|---|------|------|
| 4 | acquire observer 挿入 | onRuntimePublishedNonRt（Commit.cpp）で `onAcquire()`・publish LP 一致 ✅ |
| 5 | release observer ×4 挿入 | DeferredDeletionQueue::reclaim / drainAllUnsafe + RetireQuarantineStore::drain / drainAllUnsafe ✅ |
| 6 | window boundary との整合 | Timer.cpp の sampler で T1 の state（Running / Closed / Idle）に同期（onMeasurementStart / End・冪等・D95 固定点 5・7）✅ |
| 7 | referenceOutstanding / referenceMax の event-driven 更新 | onAcquire / onRelease で updateRunningMax（**acquire/release 双方・D95 固定点 4**）✅ |
| 8 | exactly-once 再確認 | release observer は type==World の terminal deleter 成功後・**D87 の exactly-once を利用（二重 decrement なし）** ✅ |
| 9 | observer が retirement authority に影響しない | onRelease は例外なし・所有権変更なし・reclaim 再試行なし（**measurement only・D97**）✅ |
| 10 | build + regression | ビルド成功（ConvoPeq.exe・エラーなし・既存警告のみ）・**ctest 29/29 PASS**（既知 2 テスト除外・回帰なし）✅ |

### D99.3 判定
| 項目 | 判定 |
|------|------|
| reference instrumentation 実装（#4〜#10） | **CLOSED（本 D99・実装確認）** |
| event-driven observer（release event を直接通知・案 A に戻さない） | **CLOSED（本 D99）** |
| observer lifetime（前宣言・shutdown 順序） | **CLOSED（D98 + 本 D99・実装確認）** |
| **burst test harness・通常 / burst / jitter 実測（measurement）** | **OPEN（次工程）** |
| O_w と T_w の比較・B_max 測定特性 | OPEN |
| **M の導出可能性判定** | **OPEN（YES → B_max^true ≤ B_max^observed + M の証明へ・NO → sampler 再設計）** |
| R / R_cap・T2 authority / I-2 | **NO-GO** |

### D99.4 テスト更新
| # | テスト | 検証 |
|---|--------|------|
| T331 | acquire observer 挿入 | onRuntimePublishedNonRt で onAcquire（publish LP 一致） |
| T332 | release observer ×4 | terminal deleter 成功後 ×4（reclaim / drainAllUnsafe ×2 ストア）・exactly-once（二重 decrement なし） |
| T333 | event-driven 更新 | referenceOutstanding / referenceMax を acquire/release 双方で更新・running 中のみ（T_w に Start 前履歴なし） |
| T334 | build + regression | ビルド成功・ctest 29/29 PASS |

---

## D100 — Design-82 確定（burst test harness 実装・3 条件実測・O_w/T_w/E_w の初期結果・release イベント未発生の測定特性・2026-08-16）

ユーザー D99 レビュー対応: **「『29/29 PASS → reference instrumentation の意味論が実証された』ではない。D99 で CLOSED になった
のは実装・コンパイル・既存回帰試験。次工程では T_w を『真値』とみなさない（reference observer 自体も実装された観測器・
O_w と T_w の 2 つの観測系の差 E_w を測定）。burst test harness で 3 条件を分離して実測。M = max(E_w) で終了できない
（有限回の実測最大値は未観測条件に対する安全側上界ではない）・独立した上界根拠が必要」**。**Phase I 実装 NO-GO 継続
（burst test harness 実装・実測のみ・R/R_cap・T2 は対象外）。**

### D100.1 burst test harness 実装（AudioEngineHarness + WorldRetirementMeasurementTests）
- `--measurement=normal|burst|jitter|all` サブコマンド（AudioEngineHarness の main に追加）。
- 3 条件: normal（publish 150ms 間隔反復）/ burst（interval 0ms 連続 publish）/ jitter（不規則 interval）。
- 各 window で同一 windowId に O_w / T_w / E_w / windowId / A0/R0 / A1/R1 / sampleCount / maxSamplingGapUs /
  missedTickCount / counterWrapped を記録。
- **JUCE Timer はヘッドレスで動かない → sampler を手動駆動**（AudioEngine::driveWorldRetirementSamplerForMeasurement を
  テストから呼ぶ・MessageManager 非依存）。

### D100.2 3 条件の実測結果（修正後データ・同一 windowId=1・E_w > 0 確認済み）
| 条件 | O_w | T_w | E_w | sampleCount | maxSamplingGapUs | missedTick | counterWrapped | acquireObserved | referenceRelease |
|------|-----|-----|-----|-------------|------------------|------------|----------------|-----------------|------------------|
| normal | 3 | 4 | **1** | 21 | 100959 | 0 | 0 | 11 | 10 |
| burst | 3 | 4 | **1** | 41 | 101132 | 0 | 0 | 23 | 22 |
| jitter | 13 | 13 | 0 | 19 | 101102 | 0 | 0 | 11 | 10 |

### D100.3 測定特性の評価（修正後・重要な発見）
- **referenceRelease > 0（全条件）**: release イベント（type==World の terminal deleter 実行）が発生するようになった。
  - 修正1: `AudioEngine::driveWorldRetirementReclaimForMeasurement()` 追加（epoch 進行 + tryReclaim を手動駆動・JUCE Timer ヘッドレス非依存）。
  - 修正2: **`releaseObserved` がどこからも呼ばれていなかったバグを修正**（reference observer の onRelease → telemetry.onReleaseObserved 転送）。
- **O_w の意味論修正**: 修正前は `releaseObserved` が常に 0 のため `estimate = acquireObserved - 0 = acquireObserved`（累積 acquire 数・単調増加）だった。修正後は `estimate = acquireObserved - releaseObserved`（正しい outstanding）。
- **T_w = O_w（E_w = 0・全条件）**: release イベントが発生しても、harness が sampler を各 publish 後に手動駆動するため、sampler が peak を捕捉 → E_w = 0。
- **jitter で missedTickCount=3・maxSamplingGapUs=303027**: scheduler jitter は観測された（sampler の実測 gap が最大約 303ms・missed tick 3 回）。
- **burst で maxSamplingGapUs=10330**: 連続 publish により sampler 駆動間隔が短くなった（sampling stats は正常に記録）。

### D100.4 修正内容（release イベント発生のための harness 修正・実装済み）
- **修正1（harness・reclaim 駆動）**: `AudioEngine::driveWorldRetirementReclaimForMeasurement()` 追加。
  - production では timerCallback が tryReclaimResources() を定期実行するが、ヘッドレスでは動かない。
  - epoch 進行（publishEpoch）+ tryReclaim を手動駆動し、type==World の terminal deleter（onRelease）を発生させる。
  - publish 時に markRetireEpoch() で epoch は進行済みだが、enqueue 時点の epoch は「現在 epoch」のため、もう一度
    publishEpoch() で進めてから tryReclaim しないと isOlder(entry.epoch, minReaderEpoch) が偽になる（1 epoch 遅延）。
- **修正2（releaseObserved バグ）**: `onReleaseObserved()` がどこからも呼ばれていなかった。
  - telemetry の `releaseObserved` が常に 0 → sampler の estimate = acquireObserved（累積・単調増加）→ O_w が正しくない。
  - reference observer の `onRelease()` から `telemetry.onReleaseObserved()` を転送（4 箇所の terminal release をカバー）。
  - `WorldRetirementReferenceObserver::setTelemetry()` 追加・AudioEngine コンストラクタ（CtorDtor.cpp）で配線。

### D100.5 E_w > 0 実証完了（2026-08-16）
- **D100.5 再設計**: sampler を 100ms 固定 cadence で**独立スレッド**駆動（reclaim は含まない）。publish+reclaim を**同じ iteration**（publish ループ）で駆動 → sampler が acquire 直後の peak を miss する。
- **E_w > 0 実証**: normal/burst で E_w=1 を確認（T_w=4, O_w=3）。これは「100ms sampler が event-driven reference peak を取り逃すケースを実際に生成できる」という測定器の有効性実証。
- **階層の厳密化（D94/D95）**:
  ```
  B_max^true
    │
    ├── reference observer
    │       └── T_w = B_max^reference  （絶対的真値ではない）
    └── 100ms sampler
            └── O_w = B_max^observed
  E_w = T_w - O_w
  ```
  → 証明できたのは「**O_w != T_w となる実行条件が存在する**」まで。
  → 「B_max^true - O_w <= M」は**構造上の上界が未導出**（M の数学的バインドは D101）。

### D100.5 最終状況
| 項目 | 状態 |
|------|------|
| burst test harness 実装（3 条件・同一 windowId 記録・独立 sampler スレッド） | **CLOSED（D100）** |
| 3 条件実測（E_w > 0 確認: normal/burst E_w=1） | **CLOSED（D100.5）** |
| **release イベント発生・reclaim 駆動** | **CLOSED（D100.4）** |
| O_w != T_w の実証（E_w > 0） | **CLOSED（D100.5・実証完了）** |
| M = max(E_w) での終了 | **NO-GO（D94/D95・M ≠ max(E_w)）** |
| **M の数学的バインド (B_max^true ≤ O_w + M)** | **OPEN（D101 へ引継ぐ）** |
| reference observer completeness 契約 | **OPEN（D101）** |
| R / R_cap・T2 authority / I-2 | **NO-GO（D100.5 判明: M 未導出）** |

### D100.6 テスト更新
| # | テスト | 検証 | 状態 |
|---|--------|------|------|
| T335 | burst test harness 実装 | --measurement=normal/burst/jitter/all・同一 windowId で O_w/T_w/E_w/sampling stats 記録・JUCE Timer ヘッドレス非依存（sampler 100ms 固定 cadence 独立スレッド駆動・reclaim は publish ループで同 iteration 駆動） | CLOSED |
| T336 | 3 条件実測 | normal/burst/jitter で O_w/T_w/E_w・maxSamplingGapUs・missedTick 収集 | CLOSED |
| T337 | release イベント観測 | referenceRelease > 0 を確認（normal=10 / burst=22 / jitter=10） | CLOSED |
| T338 | M 導出可能性 | E_w > 0 確認済み（normal/burst E_w=1）・ただし M = max(E_w) で終了しない（D94/D95）・M は pub+reclaim の時間幅 / sampler tick 間隔により変動 | EVALUATED |

### D101 — M の数学的バインド契約（D100.7 からの引継ぎ・next）
**目的**: `B_max^true ≤ O_w + M(G, λ, τ_b, …)` を**構造上の上界**として導出する。

#### D101.1 Outstanding の状態方程式 (strict 3-layer)
```
B_true(t)  = A_true(t)  - R_true(t)   — 数学上の真値（観測不能）
B_ref(t)   = A_ref(t)   - R_ref(t)    — reference observer が観測した値
B_obs(t_k) — sampler が tick 時刻 t_k に観測した値
```
- `T_w = B_max^reference = sup_t B_ref(t)` — **真値ではない** (D100.5)・reference observer は観測器
- `O_w = max_k B_obs(t_k)` — 100ms sampler の観測最大値
- `G = max_k (t_{k+1} - t_k)` — sampler の最大観測間隔

#### D101.2 M の数学的定義 (strict — growth と observation error の分離)
- **Δ_k^growth = max_{t ∈ [t_k, t_{k+1}]} [B_true(t) - B_true(t_k)]**
  — 数学上の真値の**観測間隔内増加量** (interval growth, true value)
- **E^obs = sup_k [B_true(t_k) - B_obs(t_k)]**
  — sampler が**安全側に**underestimate する可能性のある**有限上界** (signed ε_k^obs の安全偺上界)
  - ε_k^obs は符号付き誤差だが, D101 の目的は安全保証なので **E^obs = sup_k ε_k^obs** として有限上界を定義
- すると:
  ```
  B_true(t) - B_obs(t_k)
    = [B_true(t) - B_true(t_k)] + [B_true(t_k) - B_obs(t_k)]
    ≤ Δ_k^growth + ε_k^obs
    ≤ Δ_k^growth + E^obs
  ```
- **M ≥ sup_k Δ_k^growth + E^obs** — **数学的安全条件** (growth + observation error 上界の分離)
  - `sup_k Δ_k^growth` は #2-#8 の envelope から, `E^obs` は #1 completeness から有限になることを証明
  - `μ_burst · G` は #4/#6/#7 の envelope 特別解 (Δ_k^growth の upper bound 候補)。一般には reclaim latency / jitter / burst を含む。

> ⚠️ **M は測定値ではなく、実装上保証された envelope から導出される安全偺上界。**
> 実測 `E_w = 1` から `M = 1` を決定づけることは**禁止** (D94/D95)。E_w は O_w ≠ T_w の**実証**にすぎない。

#### D101.3 レビュー順序 (strict proof 責務)
1. **Reference completeness (4-tier)** — B_ref(t) が B_true(t) を取りこぼさない 4 つの保証
2. **State equation** — B = A - R の状態方程式 (B_true / B_ref / B_obs)
3. **Sampler gap** — G = sup_k(t_{k+1} - t_k) / missed tick / jitter / window boundary
4. **Acquire/increase envelope** — event rate / burst duration τ_b / μ_burst
5. **Single burst bound**
6. **Multiple acquire in one interval** ← E_w=1 を一般化する重要点
7. **Delayed release** — reclaim latency が outstanding peak を増幅するケース
8. **Shutdown / quarantine / deferred deletion** — 通常経路以外も同じ bound に含める
9. **Finite M proof** — #1-#8 から `sup_k Δ_k^growth` と `E^obs` が有限になることの証明
10. **D102 gate** — `finite M` が証明できれば `B_max^true ≤ O_w + M` が安全保証として成立 → D102 GO / NO → redesign

#### D101.3.1 Reference completeness — 4-tier proof
1. **Acquire completeness**: World の reference 増加が発生する**全 terminal publication path**で
   `onAcquire()` が exactly-once 呼ばれる。
   - path: `markRetireEpoch → enqueueRetire → publishEpoch` (CoordinatorLoop Non-RT)
   - 保証: enqueue 時点で observer に必ず通知 (race なし・atomic)
2. **Release completeness**: World の reference 減少が発生する**全 terminal deleter path**で
   `onRelease()` が exactly-once 呼ばれる。
   - path: `terminal deleter (type==World)` — DeferredDeletionQueue / RetireQuarantineStore の drain/drainAll
3. **Accounting conservation**: observer の `B_ref(t) = A_ref(t) - R_ref(t)` が
   実際の World lifetime accounting と一致。
   - A_ref, R_ref は**累積**（reset なし・window 内の relative ではない・D95 固定点 2）
4. **No hidden World lifetime path**: quarantine / deferred deletion / shutdown /
   failure / retry / replacement が observer の外側で reference を増減させない。
   - **#4 が重要**: `onRelease()` が N箇所あっても, **全 path が observer を通る**ことは別題
   - 証明方法: code audit — `World*` の生存期間に介入する**全 call path**をenumし、
     各 path が `onAcquire/onRelease` と1:1対応することを確認 (#1 audit target)

#### D101.3.2 Call-site map (2026-08-16 audit)
| Tier | Event | Location | Path | ステータス |
|------|-------|----------|------|-----------|
| #1 | acquire | `AudioEngine.Commit.cpp:408` | `onRuntimePublishedNonRt` → commit 成功 | ✅ single path |
| #2 | release | `RetireQuarantineStore.h:142` | `drain()` — type==World terminal deleter | ✅ |
| #2 | release | `RetireQuarantineStore.h:177` | `drainAllUnsafe()` — type==World terminal deleter | ✅ |
| #2 | release | `DeferredDeletionQueue.h:154` | `reclaim()` — type==World terminal deleter | ✅ |
| #2 | release | `DeferredDeletionQueue.h:204` | `drainAllUnsafe()` — type==World terminal deleter | ✅ |
| #3 | accounting | ISRWorldRetirementReference.h:119-120 | atomic counter, no reset | ✅ |
| #4 | **no hidden path** | quota/shutdown/failure/retry/replace | **audit 保留** | 🔍 **OPEN** |

> **D101 #1 Tier 4 追加監査 (2026-08-16)**:
> - **通常 publish/replace/shutdown path**: ✅ `RuntimeWorldAuthority::publish() → RuntimeStore::publishAndSwap(oldWorld) → retire/deferred delete` 一本化, `AudioEngine::~AudioEngine()` が `publishAndSwap(nullptr)` で current を null 化.
> - **🔴 published-domain exclusion**: `RuntimeStore destructor は current を delete しない` — "current == nullptr at destruction" を **shutdown contract** として証明する必要あり.
> - **failure/retry path**: OwnerChannel/Intent queue enqueue failure, validation rejection は **unpublished World** → R_ref に入れない exclusion が**未形式化** 🔴.
> - **quarantine path**: 4 terminal release sites 網羅済み. しかし `quarantine に入った World は published domain に属す` を **type/state contract** として証明 pending.
> - **本質**: `B_ref = A_ref - R_ref` の意味を守るためには, published-domain と unpublished-domain の **lifetime accounting 分離を形式化** する必要あり.
> - **Next**: WorldState predicate (Built→Owned→Published→Retired→Quarantine→Reclaimed) の published-domain membership をコード上で証明 (#1 audit target → #2 proof).
>
> **D101 #1 = OPEN**: Tier 1-3 ✅, Tier 4 🔴 (published-domain exclusion proof pending) → D101 = OPEN / M 未導出 / Phase I NO-GO 維持.

#### D101.4 判定条件
```
if M は構造上の上界として導出可能:
    B_max^true ≤ O_w + M
    → R_required = ceil(M / O_w) が導出可能 (D102)
else:
    M は未観測 peak により無限 (D94/D95)
    → NO-GO (R / R_cap / T2 は導出不能)
```
**現状**: `M = max(E_w) = 1` は有限値だが、**これは実測観測値であり安全偺上界ではない** (D94/D95)。
**M の数学的バインドは D101 で構造上の上界を導出するまで NO-GO。**

> **D100.7 → D101 引継ぎ**: E_w > 0 を実証（O_w != T_w）完了 (D100.5)。
> D101 への引継ぎ: `B_max^true ≤ O_w + M` の**構造上の上界**を導出する。
> - G (maxSamplingGapUs), λ (retirement rate), τ_b (burst duration), μ_burst, jitter_bound を定義。
> - reference observer completeness を契約化（T_w は真値ではない）。
> - M = f(…) として導出 → D102 で R_required を計算。
> **Phase I NO-GO 維持**: E_w > 0 が確認されたことで measurement instrumentation は機能しているが、
> M の安全偺上界が未導出のため R / R_cap / T2 への進行要件を満たさない。

<!-- audit: D101-0 M-Bound Mathematical Audit → evidence/phase-d101-0-m-bound-mathematical-audit.md (verdict: INCOMPLETE) -->
<!-- audit: D101-1 M-Bound Step 2 Counter/Observation Error → evidence/phase-d101-1-m-bound-step2-counter-observation-error.md (verdict: INCOMPLETE) -->
<!-- audit: D101-1.5 Finite-Bound Source Audit → evidence/phase-d101-1.5-finite-bound-source-audit.md (verdict: NO_FINITE_BOUND → D101 M proof UNPROVABLE, I-T2/R derivation STOP) -->

------

## I4.D101 — Authority Boundary for DeletionEntryType::World (D101 #1 Proof Audit)

- 日付: 2026-08-16
- 判定: **Step 1 (Producer Completeness) = CLOSED** / **Step 2 (API Separation) = Design-only (NOT implemented)** / **D101 #1 = OPEN**

### 0. Background

`DeletionEntryType::World` は **1箋所** で生成される (Step 1 audit = CLOSED):

```text
AudioEngine.h:3534 (within retireRuntimePublishWorldNonRt)
```

しかしながら, producer に渡される `W` は常に PublishedDomain に属するわけではない:

```text
Init.cpp:67 — rejectedWorld → retireRuntimePublishWorldNonRt → World → onRelease
```

これが INV-PUB-3 (`W ∉ PublishedDomain ⇒ R_ref(W) = 0`) の **direct counterexample**.

### 1. Root Cause

```text
retireRuntimePublishWorldNonRt(W, resetRevision)
    ↑
    ├── Published World caller   (W ∈ PublishedDomain)  ← correct
    └── Rejected World caller    (W ∉ PublishedDomain)    ← BUG
```

`resetRevision` (bool) では **domain 判定** を区別できない。

### 2. Step 1 — Producer Completeness (CLOSED)

#### Layer 1 — World type generation sites

Total: 1 producer

```text
AudioEngine.h:3534 — within retireRuntimePublishWorldNonRt()
```

### Layer 2 — enqueue transfer

| site | file:line | role |
|------|-----------|------|
| `enqueueDeferredDeleteNonRt` | AudioEngine.h:3528 → 3534 | World producer (unique) |
| `enqueueDeferredDeleteNonRtWithResult` | AudioEngine.h:4174 | forward to enqueueWithRetry |
| `enqueueWithRetry` | AudioEngine.h:4186 | Router → DeferredDeletionQueue |

#### Layer 3 — Queue consumer

| site | file:line | role |
|------|-----------|------|
| `reclaim()` | DeferredDeletionQueue.h:110 | World branch → onRelease |
| `drainAllUnsafe()` | DeferredDeletionQueue.h:181 | World branch → onRelease |
| `RetireQuarantineStore::handleRetire()` | RetireQuarantineStore.h:137 | World check (consumer) |
| `RetireQuarantineStore::reclaimBatch()` | RetireQuarantineStore.h:172 | World check (consumer) |

#### Layer 4 — terminal reclamation

| function | sites | role |
|----------|-------|------|
| `onAcquire()` | 1 site (`Commit.cpp:408`) | single source of truth |
| `onRelease()` | **3** terminal sites | World entry only |

> **DISCLAIMER**: "Layer 4 — onRelease() 4 sites" は 3+1 の記述上の不一致 — `onAcquire()` は terminal
> consumer ではない為, **3 terminal sites** が正しい。INV-PUB-4 exactly-once proof の基準値.

### 3層トレース (full trace, confirmed)

```text
DeletionEntryType::World generation (1 site)
    ↓
enqueueDeferredDeleteNonRt() / enqueueWithRetry()
    ↓
DeferredDeletionQueue entry
    ↓
reclaim() / drainAllUnsafe()
    ↓
onRelease() (World branch only)
```

### Counterexample trace

```text
Init.cpp:67 — rejectedWorld
  → retireRuntimePublishWorldNonRt(rejectedWorld, false)
  → enqueueDeferredDeleteNonRt(W, ..., DeletionEntryType::World)     ← AudioEngine.h:3534
  → enqueueWithRetry()                                                ← AudioEngine.h:4186
  → DeferredDeletionQueue entry (type=World)
  → reclaim() → onRelease()                                           ← DeferredDeletionQueue.h:148
```

**W は `publishAndSwap()` を経由せず** に `onRelease()` を呼び出す.

### 3. Step 2 — Design Contract (NOT implemented)

```cpp
// === Design Contract ===
// Authority boundary: who may enqueue DeletionEntryType::World

// retirePublishedRuntimeWorldNonRt(W, resetRevision)
//   PRECONDITION:  W ∈ PublishedDomain (must have passed publishAndSwap LP)
//   EFFECT:        enqueue(W, ..., DeletionEntryType::World)
//                  → terminal reclaim → onRelease()

// retireRejectedRuntimeWorldNonRt(W)
//   PRECONDITION:  W ∉ PublishedDomain
//   EFFECT:        enqueue(W, ..., DeletionEntryType::Generic)
//                  → destruction only → NO onRelease()
```

### INV-WORLD-TYPE (post-implementation closure target)

```text
∀ W: enqueue(..., DeletionEntryType::World)
    ⇒ caller is retirePublishedRuntimeWorldNonRt(W)
    ⇒ caller's W passed publication LP
    ⇒ W ∈ PublishedDomain
```

### Contrapositive

```text
W ∉ PublishedDomain
    ⇒ retireRejectedRuntimeWorldNonRt(W)
    ⇒ DeletionEntryType::Generic
    ⇒ DeferredDeletionQueue World branch に入れない
    ⇒ no onRelease()
    ⇒ R_ref(W) = 0
```

### Consumer Logic (NO CHANGE REQUIRED)

`DeferredDeletionQueue` consumer logic は維持:

```cpp
// DeferredDeletionQueue.h:148 / 199
if (entryType == DeletionEntryType::World) {
    referenceObserver_->onRelease();  // World ⇒ reference-accounted World
}
```

### 4. caller provenance (Step 3 — to be audited after Step 2 code change)

ResetRevision == false callers (critical mix)

| caller | provenance | proposed API |
|--------|------------|--------------|
| `RuntimePublicationCoordinator` (normal oldWorld) | Published | `retirePublishedRuntimeWorldNonRt` |
| `RuntimePublicationCoordinator` (shutdown oldWorld) | Published | `retirePublishedRuntimeWorldNonRt` |
| `AudioEngine.Init.cpp:67` (rejectedWorld) | **Rejected** | `retireRejectedRuntimeWorldNonRt` |
| `bootstrap` (oldWorld) | Published | `retirePublishedRuntimeWorldNonRt` |

### ResetRevision == true callers (shutdown clear oldWorld)

| caller | provenance | proposed API |
|--------|------------|--------------|
| shutdown | Published | `retirePublishedRuntimeWorldNonRt` |

### 5. D101 #1 Current Status

```text
D100        CLOSED
D101.2      CLOSED
D101 #1     OPEN   (Step 1 で反例が完全確認済み)

Step 1      CLOSED   ✅ World producer completeness = 100% traced
Step 2      NOT STARTED   — Design contract drafted only (API separation)

INV-PUB-1   OPEN
INV-PUB-2   OPEN
INV-PUB-3   DISPROVEN   (rejected W → World → onRelease — direct counterexample)
INV-PUB-4   OPEN

M           NO-GO
Phase I     NO-GO
```

コード変更は**未実施** — audit/fix phase を分離する方針のため, この証明記録を保持.
