# D135-8/9 Gate G-4.1 — R1+R2 型定義実装（Work Report）

**Gate:** G-4.1 (R1+R2 type introduction only). **STOP after this — no G-4.2.**
**Scope:** type/contract-expression introduction only. NO admission control / blind-overwrite / publish wiring / canSupersede / durable table / reservation-change / reclaim/shutdown wiring.
**Base:** HEAD `5f6f48c` (F6) + I4 D12/D18 (G-2 normative). Single file: `src/audioengine/ISRRuntimePublicationCoordinator.h` (57 insertions / 8 deletions).

## What was added (all type-only, inert, behavior-freeze)
1. `using RecoveryEpisodeId = std::uint64_t;` — lineage id (D13; NOT epoch / NOT intentId).
2. `using RecoveryGeneration = std::uint64_t;` — in-episode ordinal (newer-is-after ONLY; != intentId).
3. `enum class ObligationDomains : uint8_t { None, IR, Conv, EQ, Config, OS };` — domain-coverage bitmask (necessary condition, never sufficient).
4. `SemanticRecoveryTarget` extended to I4 D12.2 **6-field**: `{ irIdentityHash, convolutionConfigHash, dspParameterHash, domainCoverage, convolverFingerprint, buildInputHash }` + `operator==` comparing the **5 semantic values** (domainCoverage excluded per D12.2 — coverage is a separate necessary-condition check).
5. `LogicalRecoveryIdentity` = `{ handle, episodeId, target }` + `operator==` (handle + episodeId + target). **RecoveryGeneration NOT in identity** (D18; only a canSupersede isAfter operand).
6. `ObligationState::ResolvedSuperseded` (dormant in Phase I — containment equality-only ⇒ no production transition; no firing path added).
7. `RecoveryIntent` += `RecoveryEpisodeId episodeId{0}` (value wired in G-4.2). `PendingRecoveryAdmission` += `RecoveryEpisodeId episodeId{0}` + `recoveryGeneration` field type changed `uint64_t` → `RecoveryGeneration` (alias, no-op).

## Deliberate deviation (documented)
The user's G-4.1 spec listed `domainCoverage` FIRST in `SemanticRecoveryTarget`. I placed the **3 original hashes first** (`irIdentityHash, convolutionConfigHash, dspParameterHash, domainCoverage, convolverFingerprint, buildInputHash`) for **aggregate-init backward-compatibility**: the existing `submitRecoveryRequest` constructs `SemanticRecoveryTarget{irIdentityHash, convolutionConfigHash, dspParameterHash}` positionally (cpp:836-841), so keeping the 3 hashes index-stable preserves behavior **(G-4.1 behavioral freeze, .cpp untouched)**. All 6 fields are present; only field order differs. If the canonical D12.2 order (domainCoverage-first) is desired, it must be paired with a .cpp aggregate-init update — deferred to G-4.2 (which touches .cpp).

## V1-V7 verification (read-only)

| check | result |
|---|---|
| V1 type presence (EpisodeId/Generation/Domains/SemanticRecoveryTarget/LogicalRecoveryIdentity/ResolvedSuperseded) | ✅ all 1 (ResolvedSuperseded in ObligationState enum h:325) |
| V2 identity structure (handle + episodeId + target; NO generation in operator==) | ✅ LogicalRecoveryIdentity h:292, operator== = handle&episodeId&target |
| V3 semantic target (5 semantic fields + domainCoverage) | ✅ h:274-290 |
| V4 RecoveryProvenance absent | ✅ 0 |
| V5 no new intentId-as-RecoveryGeneration code | ✅ only pre-existing .cpp lines (932/1092, unchanged); no new conflation |
| V6 behavioral freeze | ✅ type-only; SemanticRecoveryTarget order backward-compat; .cpp untouched; aggregate init `{a,b,c}`→ir/conv/dspParam unchanged |
| V7 diff boundary | ✅ only `ISRRuntimePublicationCoordinator.h` (57+/8-); no .cpp/test changed |

## Note
- A serena regex-replace leak produced one literal `\n` on the inserted comment line; fixed with an exact Edit (verified 0 remaining).
- `submitRecoveryRequest` / `markTransientFailure` / `resolveRecoveryObligation` / `settlePendingRecoveryAdmission` / `redriveDeferredRecovery` — **unchanged & semantics preserved** (V6).

## STOP
G-4.1 complete & PASS. **No G-4.2 started** (per instruction: PASS confirmed before proceeding). Await G-4.2 (identity wiring: RecoveryEpisodeId threading, `recoveryGeneration`→buildSource.generation, SemanticRecoveryTarget field wiring in cpp) explicit go.
