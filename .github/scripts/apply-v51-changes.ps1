param(
    [string]$FilePath = "doc/work37/recovery_system_plan.md"
)

$ErrorActionPreference = "Stop"
$content = [System.IO.File]::ReadAllText((Resolve-Path $FilePath).Path, [System.Text.Encoding]::UTF8)

# === Change 1: Header v4.9 → v5.1 ===
$content = $content -replace '^# Practical Stable ISR Bridge Runtime — 回復システム改修計画書 v4\.9', '# Practical Stable ISR Bridge Runtime — 回復システム改修計画書 v5.1'

# === Change 2: Fix segmentBuffer safety note (9.4/9.33) ===
$oldNote = '注意:.*?segmentBuffer.getNumAvailableSamples\(\).*?スレッド安全性は別途確認が必要。.*?追加の同期機構が必要。'
$newNote = '注意: AudioSegmentBuffer.h:82 の getNumAvailableSamples() は convo::consumeAtomic(totalSamples, memory_order_acquire) を使用しており、完全にロックフリーかつスレッドセーフである。追加の同期機構は不要。'
$content = $content -replace $oldNote, $newNote

# === Change 3: Add 9.37 RecoveryAction単一実行ルール ===
$insertPoint = '#### 9.27 PolicySource 拡張（Phase 9 最終版 v4.9）'
$newSection37 = @"

#### 9.37 【P0】RecoveryAction 単一実行ルール

**背景**: 複数Actionの同時発火は危険。同一tick最大1個のActionのみ実行。

**ルール**:
- 同一tick最大1Actionのみ実行
- 高いRecoveryActionLevel優先
- 同Level内はenum定義順(None＜EmergencyDrain)
- Cooldown制御で再実行間隔管理

"@
$content = $content.Replace($insertPoint, $newSection37 + "`r`n" + $insertPoint)

# === Change 4: Add 9.38 ResetLearner ===
$newSection38 = @"

#### 9.38 【P0】ResetLearner RecoveryAction

**背景**: PauseLearnerだけではFIFO(bufferedSamples=3840000)が解消されない。
**DataRace注意**: AudioThreadはpushAdaptiveCaptureBlocksでadaptiveCaptureActiveRt未確認のままFIFOに書き込む。

**処理**: adaptiveCaptureActiveRt=false → GracePeriod(100us) → clearFifo → setState → startLearning

"@
$content = $content.Replace($insertPoint, $newSection38 + "`r`n" + $insertPoint)

# === Change 5: Add 9.39 PendingDeployment→ForceSnapshotPublish ===
$newSection39 = @"

#### 9.39 【P0】PendingDeployment→ForceSnapshotPublish 回復経路

**Policy**: pendingIRAge>30s + WorldConsistent + pendingRetire<threshold + noCrossfade → ForceSnapshotPublish

"@
$content = $content.Replace($insertPoint, $newSection39 + "`r`n" + $insertPoint)

# === Change 6: Update 9.27 PolicySource (30→32) ===
$content = $content.Replace(
    'SuppressionLoopOscillation,   // 9.35',
    'SuppressionLoopOscillation,   // 9.35' + "`r`n    // ★ v5.0:" + "`r`n    ResetLearner,                   // 9.38" + "`r`n    PendingDeploymentRecovery,      // 9.39"
)
$content = $content.Replace('_Count  // 30 source', '_Count  // 32 source')

# === Change 7: Update 9.28 RecoveryAction (20→21) ===
$content = $content.Replace(
    'EnterSafeMode,                 // 9.34',
    'EnterSafeMode,                 // 9.34' + "`r`n    // ★ v5.0:" + "`r`n    ResetLearner,                   // 9.38"
)
$content = $content.Replace('_Count  // 20 action', '_Count  // 21 action')

# === Change 8: Enhance 9.16 Rollback with Fingerprint check + SyncBack ===
$targetFP = 'Fingerprint 全一致 → RollbackToLastHealthyWorld。不一致 → EnterSafeMode。'
$newFP = 'Fingerprint 全一致 → RollbackToLastHealthyWorld。不一致 → EnterSafeMode。\n\n★ Rollback後はAtomic変数同期(SyncBack)必須: Rollback先WorldのSemanticSchemaから値を抽出しmanualOversamplingFactor等を上書き。sendChangeMessage()でUI反映。'
$content = $content.Replace($targetFP, $newFP)

# === Change 9: Enhance 9.32 RetireBlockerEvidence with worldId/revision/intentId ===
$oldEv = 'int readerIndex{-1};'
$newEv = 'int readerIndex{-1};' + "`r`n    uint64_t worldId{0};     // ★ v5.0" + "`r`n    uint64_t publishRevision{0};" + "`r`n    uint64_t intentId{0};"
$content = $content.Replace($oldEv, $newEv)

# === Change 10: Remove stale duplicate 9.20 sections ===
# Remove the second 9.20/9.21 that's superseded by 9.27/9.28
$dupePattern = '#### 9\.20 PolicySource 拡張（Phase 9 統合最終版）[\s\S]*?#### 9\.21 RecoveryAction 拡張（Phase 9 統合最終版）[\s\S]*?_Count  // 12 action'
$content = $content -replace $dupePattern, ''

# Write back
$content = $content.TrimEnd() + "`r`n"
[System.IO.File]::WriteAllText((Resolve-Path $FilePath).Path, $content, [System.Text.Encoding]::UTF8)
Write-Host "Done. File size: $((Get-Item $FilePath).Length) bytes"
