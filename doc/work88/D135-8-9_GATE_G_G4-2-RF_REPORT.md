# D135-8/9 Gate G-4.2-RF — RecoveryEpisodeId Production-Wiring Removal（Work Report）

**Status:** PASS
**Type:** implementation / contract restoration（D105-R23 Phase-I 復元）. **STOP after this — G-4.3 not started.**
**Files changed:** `src/audioengine/ISRRuntimePublicationCoordinator.h` + `.cpp`（113+/17- vs HEAD `5f6f48c`; これら2ファイルのみ）

## Removed（EpisodeId production wiring 全除去）
- `LogicalRecoveryObligation::episodeId`（h フィールド）
- `RecoveryIntent::episodeId`（h フィールド）
- `PendingRecoveryAdmission::episodeId`（h フィールド）
- `RecoveryAdmissionTable::nextEpisodeId_`（カウンタ宣言）
- `slots_[i].episodeId = ++nextEpisodeId_;`（tryInsert 割当）
- episode threading：`intent.episodeId = ...`（submitRecoveryRequest cpp:959 / redrive cpp:1143 / take cpp:1195）、`pendingRecoveryAdmission_.episodeId = ...`（submit cpp:987 / redrive cpp:1150）
- `LogicalRecoveryIdentity`（episode-including、production 未使用のため削除）
- `using RecoveryEpisodeId = std::uint64_t;`（type alias；現状 production 参照 0 → 不要のため削除。コメントのみ「Phase-II deferred」注記を残置）

## Preserved（維持・絶対に変更しない）
- `RecoveryGeneration`（using alias h:223）+ `RecoveryGeneration recoveryGeneration{0}`（LogicalRecoveryObligation h:355 / RecoveryIntent h:251 / PendingRecoveryAdmission h:960）
- `nextRecoveryGeneration_{1}`（h:452）+ `slots_[i].recoveryGeneration = ++nextRecoveryGeneration_;`（h:404）
- `intent.recoveryGeneration = slot/s/pending.recoveryGeneration;` threading（cpp:959/985/1141/1146/1191）
- `intentId` 生成・意味（`nextRecoveryIntentId_` 未変更）
- `BuildGeneration`（`buildSource.generation`＝RuntimeBuildSnapshot::generation 未変更）
- `SemanticRecoveryTarget` 6-field（ir/conv/dspParam/domainCoverage/convolverFingerprint/buildInputHash）
- `buildInputHash`（FNV-1a 実値）・`domainCoverage`（metadata のみ）
- `CoalesceIdentity = { quarantinedHandle, SemanticRecoveryTarget }`（h:297-301、episode 成分なし）

## Verification（V1-V8）
| V | 確認 | 結果 |
|---|---|---|
| V1 | RecoveryEpisodeId production references | **0**（コメント1件「Phase-II deferred」のみ） |
| V2 | nextEpisodeId_ references | **0** |
| V3 | episode threading | **0** |
| V4 | intentId conflation（`recoveryGeneration = intent.intentId`） | **0** |
| V5 | BuildGeneration conflation（`buildSource.generation = recoveryGeneration`） | **0** |
| V6 | CoalesceIdentity = {handle, target} | ✅ h:297-301（episode 成分なし） |
| V7 | behavioral decision logic unchanged | ✅（episode 追加分のみ削除。tryInsert capacity/resolve/delivery/K=4 retry/terminal 決定ロジック不変） |
| V8 | diff boundary | ✅ `ISRRuntimePublicationCoordinator.h` + `.cpp` のみ |

## Restored Phase-I model (D105-R23)
```
Phase I
────────────────────
Logical Recovery Obligation
        ├── quarantinedHandle
        ├── SemanticRecoveryTarget
        └── RecoveryGeneration（lineage ordinal）
CoalesceIdentity = { handle, SemanticRecoveryTarget }
RecoveryEpisodeId = DESIGN ONLY / Phase-II deferred
```
`liveCount_` = 単一 live-obligation counter。episode 単位の production counter は Phase-II 延期（復元済み）。

## Notes
- literal `\n` leak = 0（両ファイル）。
- `SemanticRecoveryTarget` フィールド順・buildInputHash/domainCoverage は G-4.2 のまま維持（R23-consistent）。
- 再監査（G-4.2-RF Audit）は今回の実装後に別途指示ください。**G-4.3 は開始していません。**

## STOP
G-4.2-RF complete & PASS（V1-V8）. Await G-4.2-RF Audit（read-only）→ PASS → G-4.3 instruction. No G-4.3 / coalesce / durable-table / admission rewrite performed.
