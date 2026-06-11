# ISR Bridge Runtime 改修 実装チェックリスト

> 作成: 2026-06-11 | ベース: `doc/work29/ISR Bridge Runtime 改修計画.md`

---

## Sランク（実施推奨）

### Phase 2: Retire Epoch 統一

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T2-1 | DSPLifetimeManager 全面適用 | ⬜ | CtorDtor/RebuildDispatch/Timer/DSPLifetimeManager.h |
| T2-2 | EpochDomain 経由 Reclaim 一元化 | ⬜ | DSPLifetimeManager.h / EpochDomain.h |
| T2-3 | retireDSP 禁止 CI 追加 | ⬜ | `.github/scripts/verify-isr-maturity.ps1` |

### Phase 4.5: Reader Residency Diagnostics

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T4-5-1 | ReaderSlot 拡張 (steady_clock timestamp) | ⬜ | `src/core/EpochDomain.h` |
| T4-5-2 | HealthEvent へ Reader 詳細追加 | ⬜ | `RuntimeHealthMonitor.h/.cpp` |
| T4-5-3 | Shutdown 時 Reader 残留診断強化 | ⬜ | `ReleaseResources.cpp` |

### Phase 8: 検証基盤

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T8-1 | 7条件自動検証スクリプト | ⬜ | `.github/scripts/verify-isr-maturity.ps1` |
| T8-2 | 既存 verifier 拡張 | ⬜ | `isr-verify-common.ps1` 等 |

---

## Aランク

### Phase 4: Diagnose 拡張

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T4-1 | diagnoseRetireStall 実装 | ⬜ | `RuntimeHealthMonitor.cpp` |

### Phase 3: VerifyDrained

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T3-1 | VerifyDrained フェーズ追加 | ⬜ | `ISRShutdown.h/.cpp` |
| T3-2 | VerifyDrained 具体実装 | ⬜ | `ReleaseResources.cpp` |

---

## Bランク

### Phase 1: Overflow 改善

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T1-1 | FallbackQueue 統合 (動的上限) | ⬜ | `ISRRetire.h/.cpp` |
| T1-2 | QueueFull Escalation 経路確立 | ⬜ | `ISRRetireRouter.cpp` |

---

## Cランク

### Phase 5: Coordinator Authority 文書化

| # | タスク | 状態 | ファイル |
|---|-------|------|---------|
| T5-1 | Authority 文書化 + CI 検証強化 | ⬜ | `RuntimePublicationCoordinator.h` |
| T5-2 | retireDSP 全削除 | ⬜ | `AudioEngine.h/.Retire.cpp` |

---

## 凡例

- ⬜ = 未着手
- 🔄 = 作業中
- ✅ = 完了
- ❌ = 保留/非推奨
