# Practical Stable ISR Bridge Runtime — 実装チェックリスト

**基準日**: 2026-06-13
**ベース文書**: `doc/work33/remediation_plan.md`

---

## Phase 1: A-3 Reader→Shutdown 結合 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 1.1 | `RuntimeDrainAudit` に Reader フィールド追加 | `RuntimeDrainAudit.h` | ✅ |
| 1.2 | `collectDrainAudit()` で detectStuckReaders 収集 | `AudioEngine.Threading.cpp` | ✅ |
| 1.3 | `getPrimaryBlockingReason()` に ReaderActive 追加 | `RuntimeDrainAudit.h` | ✅ |
| 1.4 | `onHealthEvent()` に EVENT_READER_STUCK ハンドラ追加 | `AudioEngine.Timer.cpp` | ✅ |
| 1.5 | `VerifyDrained` で markTimedOut(ReaderActive) | `AudioEngine.Processing.ReleaseResources.cpp` | ✅ |

## Phase 2: A-2 DrainAudit Reader統合 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 2.1 | `RuntimeDrainAudit` フィールド追加（Phase 1.1 と共通） | `RuntimeDrainAudit.h` | ✅ |
| 2.2 | `isAllZero()` に Reader 条件を追加しない確認 | `RuntimeDrainAudit.h` | ✅ |

## Phase 3: A-5 Double-Retire Telemetry ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 3.1 | `WorldLifecycleAudit` に doubleRetireCount 追加 | `WorldLifecycleAudit.h` | ✅ |
| 3.2 | `onWorldRetired()` 二重retire検出時に telemetry+evidence | `WorldLifecycleAudit.h` | ✅ |
| 3.3 | WorldLifecycleAudit.cpp に evidence 出力追加 | `WorldLifecycleAudit.cpp` | ✅ |

## Phase 4: 8.6 ReaderStuck Evidence定期出力 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 4.1 | `RuntimeHealthMonitor` に定期Evidenceタイマー追加 | `RuntimeHealthMonitor.h` | ✅ |
| 4.2 | `diagnoseRetireStall()` に定期Evidence出力追加 | `RuntimeHealthMonitor.cpp` | ✅ |

## Phase 5: B-2 HealthState統合 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 5.1 | `RuntimeDrainAudit` に healthState フィールド追加 | `RuntimeDrainAudit.h` | ✅ |
| 5.2 | `collectDrainAudit()` で healthState 収集 | `AudioEngine.Threading.cpp` | ✅ |

## Phase 6: B-1 World Consistency ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 6.1 | `RuntimeDrainAudit` に verifyWorldConsistency 追加 | `RuntimeDrainAudit.h` | ✅ |
| 6.2 | `VerifyDrained` で Evidence 出力追加 | `AudioEngine.Processing.ReleaseResources.cpp` | ✅ |

## Phase 7: C-4 HealthState Reset ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 7.1 | `RuntimeHealthMonitor` に reset() 追加 | `RuntimeHealthMonitor.h` | ✅ |
| 7.2 | `reset()` 実装（HealthState のみ） | `RuntimeHealthMonitor.cpp` | ✅ |
| 7.3 | `prepareToPlay()` 開始時に reset() 呼出 | `AudioEngine.Processing.PrepareToPlay.cpp` | ✅ |
| 3.3 | WorldLifecycleAudit.cpp に evidence 出力追加 | `WorldLifecycleAudit.cpp` | ✅ |

## Phase 4: 8.6 ReaderStuck Evidence定期出力 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 4.1 | `RuntimeHealthMonitor` に定期Evidenceタイマー追加 | `RuntimeHealthMonitor.h` | ✅ |
| 4.2 | `diagnoseRetireStall()` に定期Evidence出力追加 | `RuntimeHealthMonitor.cpp` | ✅ |

## Phase 5: B-2 HealthState統合 ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 5.1 | `RuntimeDrainAudit` に healthState フィールド追加 | `RuntimeDrainAudit.h` | ✅ |
| 5.2 | `collectDrainAudit()` で healthState 収集 | `AudioEngine.Threading.cpp` | ✅ |

## Phase 6: B-1 World Consistency ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 6.1 | `RuntimeDrainAudit` に verifyWorldConsistency 追加 | `RuntimeDrainAudit.h` | ✅ |
| 6.2 | `VerifyDrained` で Evidence 出力追加 | `AudioEngine.Processing.ReleaseResources.cpp` | ✅ |

## Phase 7: C-4 HealthState Reset ✅ 完了

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 7.1 | `RuntimeHealthMonitor` に reset() 追加 | `RuntimeHealthMonitor.h` | ✅ |
| 7.2 | `reset()` 実装（HealthState のみ） | `RuntimeHealthMonitor.cpp` | ✅ |
| 7.3 | `prepareToPlay()` 開始時に reset() 呼出 | `AudioEngine.Processing.PrepareToPlay.cpp` | ✅ |

## Phase 8: A-4 Crossfade経路統一（設計継続 — 未着手）

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 8.1 | notifyTransitionComplete 4責務の定義文書化 | - | ⬜ 設計継続 |
| 8.2 | publishIdleWorldOnly 責務境界の明確化 | - | ⬜ 設計継続 |
| 8.3 | 3経路完全差分表 + 責務境界文書 | - | ⬜ 設計継続 |

## Phase 9: B-3 Warmup Validation（未着手）

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 9.1 | validateRuntimeIntegrity 宣言 | `RuntimeBuilder.h` | ⬜ |
| 9.2 | validateRuntimeIntegrity 実装 | `RuntimeBuilder.cpp` | ⬜ |
| 9.3 | rebuildThreadLoop に呼出追加 | `AudioEngine.RebuildDispatch.cpp` | ⬜ |

## Phase 10: C-2 EmergencyDrain（未着手 — optional）

| # | タスク | ファイル | 状態 |
|---|--------|---------|------|
| 10.1 | ShutdownPhase に EmergencyDrain 追加 | `ISRShutdown.h` | ⬜ |
| 10.2 | advancePhase に EmergencyDrain 遷移追加 | `ISRShutdown.cpp` | ⬜ |
| 10.3 | releaseResources に EmergencyDrain 追加 | `AudioEngine.Processing.ReleaseResources.cpp` | ⬜ |

---

## 凡例

- ✅ 完了
- 🔄 進行中
- ⬜ 未着手
- ➡️ 設計継続（実装開始条件未達）
