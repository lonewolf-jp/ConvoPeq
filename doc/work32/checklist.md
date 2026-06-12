# 実装チェックリスト

**基礎文書**: `doc/work32/implementation_plan.md`
**作成日**: 2026-06-12 | **最終更新**: 2026-06-12
**凡例**: ⬜ 未着手 / 🔷 作業中 / ✅ 完了 / ❌ 保留

---

## Tier 1（まず着手）— ✅ 完了

### S-2: Health→Control Pipeline (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 3.3-1 | AudioEngine に HealthStateRef 公開メソッド追加 | `AudioEngine.h` | ✅ |
| 3.3-2 | Rebuild Admission 改修（shouldRejectRebuildAdmissionForPressure） | `AudioEngine.Threading.cpp` | ✅ |
| 3.3-3 | RuntimeBuilder 改修（setHealthStateRef + build() チェック） | `RuntimeBuilder.h`, `RuntimeBuilder.cpp` | ✅ |
| 3.3-4 | CrossfadeAuthority evaluate() に HealthState チェック追加 | `CrossfadeAuthority.cpp` | ✅ |
| 3.3-5 | DSPTransition onPublishCompleted() に HealthState チェック追加 | `DSPTransition.h` | ✅ |
| 3.3-6 | RuntimeBuilder 配線（全構築サイト） | `RebuildDispatch.cpp`, `Init.cpp`, `PrepareToPlay.cpp`, `ReleaseResources.cpp` | ✅ |

**変更ファイル**: 10ファイル（AudioEngine.h, AudioEngine.Threading.cpp, RuntimeBuilder.h/.cpp, CrossfadeAuthority.cpp, DSPTransition.h, RebuildDispatch.cpp, Init.cpp, PrepareToPlay.cpp, ReleaseResources.cpp）

### S-1: Epoch の意味の純化 (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 2.3-1 | DSPLifetimeManager::retire() publishEpoch→currentEpoch | `DSPLifetimeManager.h` | ✅ |
| 2.3-2 | 影響確認（他 publishEpoch 使用箇所は変更なし） | — | ✅ |

**変更ファイル**: 1ファイル（DSPLifetimeManager.h）— 1行修正

---

## Tier 2 — ✅ 完了

### A-1: Health→Recovery（自己防衛）(✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 4.3-1 | Reader Exhaustion → Admission 強制停止 + 診断ダンプ | `AudioEngine.Timer.cpp` | ✅ |
| 4.3-2 | Publication Stall → deferred drain | `AudioEngine.Timer.cpp` | ✅ |
| 4.3-3 | Retire Stall → Builder Throttle + 強制 Reclaim | `AudioEngine.Timer.cpp` | ✅ |
| 4.3-4 | Evidence 強制出力連携 | `AudioEngine.Timer.cpp` | ✅ |

**変更ファイル**: 1ファイル（AudioEngine.Timer.cpp）

---

## Tier 3 — ✅ 完了

### B-1: Reader Ownership Telemetry (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 5.3-1 | IEpochProvider に ReaderSlotDetail 構造体 + 仮想メソッド追加 | `IEpochProvider.h` | ✅ |
| 5.3-2 | EpochDomain に getReaderSlotDetail override 実装 | `EpochDomain.h` | ✅ |
| 5.3-3 | ISRRetireRouter に委譲メソッド追加 | `ISRRetireRouter.h`, `ISRRetireRouter.cpp` | ✅ |
| 5.3-4 | checkReaderSlotUsage() で最長滞留 Reader の詳細情報設定 | `RuntimeHealthMonitor.cpp` | ✅ |

**変更ファイル**: 5ファイル（IEpochProvider.h, EpochDomain.h, ISRRetireRouter.h/.cpp, RuntimeHealthMonitor.cpp）

---

## Tier 4 — ✅ 完了

### C-1: RuntimeDrainAudit ↔ WorldLifecycleAudit (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 6.3-1 | RuntimeDrainAudit に activeWorldCount/publishedCount/retiredCount 追加 | `RuntimeDrainAudit.h` | ✅ |
| 6.3-2 | collectDrainAudit() で WorldLifecycleAudit の値を取得 | `AudioEngine.Threading.cpp` | ✅ |

**変更ファイル**: 2ファイル（RuntimeDrainAudit.h, AudioEngine.Threading.cpp）

### C-2: BuildError 分類拡充 (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 7.3-1 | BuildError enum 拡張（MKLFailure/ConvolverFailure/PrepareFailure） | `RuntimeBuilder.h` | ✅ |
| 7.3-2 | toString() 拡張 | `RuntimeBuilder.cpp` | ✅ |

**変更ファイル**: 2ファイル（RuntimeBuilder.h, RuntimeBuilder.cpp）

### C-3: Reader Slot 可観測性強化 (✅ 完了)

| # | Step | ファイル | 状態 |
|---|---|---|---|
| 8.3-1 | ReaderSlot に ownerThreadId / ownerTag 追加 | `EpochDomain.h` | ✅ |
| 8.3-2 | registerReaderThread() にタグ名パラメータ追加 + デフォルト実装 | `EpochDomain.h` | ✅ |

**変更ファイル**: 1ファイル（EpochDomain.h）+ include追加（<thread>, <cstring>）

---

## 集計

| カテゴリ | 変更ファイル数 | ステップ数 | 進捗 |
|---|---|---|---|
| **全 Tier** | **19ファイル** | **26ステップ** | **100% ✅** |
| Tier 1 (S-2 + S-1) | 11ファイル | 8ステップ | 100% |
| Tier 2 (A-1) | 1ファイル | 4ステップ | 100% |
| Tier 3 (B-1) | 5ファイル | 4ステップ | 100% |
| Tier 4 (C-1/C-2/C-3) | 5ファイル | 6ステップ | 100% |
