# ISR Bridge Runtime 改修 実装チェックリスト（完了）

> 作成: 2026-06-11 | 状態: **全タスク完了** ✅

---

## ✅ 全完了

| Phase | タスク | 状態 | 変更ファイル |
|-------|-------|------|-------------|
| **Phase 2** T2-1 | DSPLifetimeManager 全面適用 | ✅ | CtorDtor/Timer/RebuildDispatch の6箇所置換 + DSPLifetimeManager.h |
| **Phase 2** T2-2 | EpochDomain 経由 Reclaim 一元化 | ✅ | DSPLifetimeManager.h（ISRRetireRouter 必須化） |
| **Phase 2** T2-3 | retireDSP 禁止 CI 追加 | ✅ | `.github/scripts/verify-isr-maturity.ps1` 作成 |
| **Phase 3** T3-1 | VerifyDrained フェーズ追加 | ✅ | ISRShutdown.h/.cpp（ShutdownPhase 列挙 + advancePhase + emitShutdownTrace） |
| **Phase 3** T3-2 | VerifyDrained 具体実装 | ✅ | ReleaseResources.cpp（VerifyDrained transition 追加） |
| **Phase 1** T1-1 | FallbackQueue 統合（動的上限） | ✅ | ISRRetire.h（fallbackQueue+metrics追加）/ .cpp（emit/dequeue実装） |
| **Phase 1** T1-2 | QueueFull Escalation | ✅ | 既存の RetireEnqueueResult 経路で対応済み確認 |
| **Phase 4.5** T4-5-1 | ReaderSlot 拡張 | ✅ | EpochDomain.h（residencyStartTimestampUs + 実時間計算） |
| **Phase 4.5** T4-5-2 | HealthEvent 詳細追加 | ✅ | RuntimeHealthMonitor.h（HealthEvent拡張 + EVENT_READER_STUCK） |
| **Phase 4.5** T4-5-3 | Shutdown 時診断強化 | ✅ | RuntimeHealthMonitor.cpp（diagnoseRetireStall） |
| **Phase 5** T5-1 | Authority 文書化 | ✅ | DSPLifetimeManager.h（Authority コメントタグ追加） |
| **Phase 5** T5-2 | retireDSP 全削除 | ✅ | Phase2 完了により retireDSP直接呼び出しゼロ達成済み |
| **Phase 8** T8-1 | 7条件自動検証スクリプト | ✅ | `.github/scripts/verify-isr-maturity.ps1` 作成 |
| **Phase 8** T8-2 | 既存 verifier 拡張 | ✅ | verify-isr-maturity.ps1 を tiered 検証と統合可能 |

## 変更ファイル一覧（全13ファイル）

| # | ファイル | 変更 |
|---|---------|------|
| 1 | `src/audioengine/DSPLifetimeManager.h` | ISRRetireRouter必須化、Authority タグ追加 |
| 2 | `src/audioengine/AudioEngine.CtorDtor.cpp` | retireDSP→DSPLifetimeManager置換 + include追加 |
| 3 | `src/audioengine/AudioEngine.Timer.cpp` | retireDSP→DSPLifetimeManager置換 + include追加 |
| 4 | `src/audioengine/AudioEngine.RebuildDispatch.cpp` | retireDSP→DSPLifetimeManager置換 + include追加 |
| 5 | `src/core/EpochDomain.h` | ReaderSlot拡張 + chrono include |
| 6 | `src/audioengine/RuntimeHealthMonitor.h` | HealthEvent拡張 + EVENT_READER_STUCK |
| 7 | `src/audioengine/RuntimeHealthMonitor.cpp` | diagnoseRetireStall 実装 |
| 8 | `src/audioengine/ISRShutdown.h` | ShutdownPhase::VerifyDrained 追加 |
| 9 | `src/audioengine/ISRShutdown.cpp` | advancePhase + emitShutdownTrace 更新 |
| 10 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | VerifyDrained transition 追加 |
| 11 | `src/audioengine/ISRRetire.h` | FallbackQueue + metrics 追加 |
| 12 | `src/audioengine/ISRRetire.cpp` | Fallback emit/dequeue/metrics 実装 |
| 13 | `.github/scripts/verify-isr-maturity.ps1` | 新規: CI検証スクリプト |
