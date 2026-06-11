# ISR Bridge Runtime 改修 実装チェックリスト

> 作成: 2026-06-11 | 最終更新: 2026-06-11 実装完了
> ベース: `doc/work29/ISR Bridge Runtime 改修計画.md`

---

## Sランク ✅ 全完了

### Phase 2: Retire Epoch 統一

| # | タスク | 状態 | 変更内容 |
|---|-------|------|---------|
| T2-1 | DSPLifetimeManager 全面適用 | ✅ | CtorDtor(2), Timer(2), RebuildDispatch(2) の計6箇所の retireDSP() 直呼びを DSPLifetimeManager::retire() に置換 |
| T2-2 | EpochDomain 経由 Reclaim 一元化 | ✅ | DSPLifetimeManager に ISRRetireRouter 必須化、Router→EpochDomain へ直接委譲、retireDSP() ラッパ脱却 |
| T2-3 | retireDSP 禁止 CI 追加 | ✅ | `.github/scripts/verify-isr-maturity.ps1` 作成（8条件自動検証） |

### Phase 4.5: Reader Residency Diagnostics

| # | タスク | 状態 | 変更内容 |
|---|-------|------|---------|
| T4-5-1 | ReaderSlot 拡張 | ✅ | `EpochDomain.h`: ReaderSlot に `residencyStartTimestampUs` 追加、StuckReaderInfo に `residencyTimeUs` 追加、detectStuckReaders で実時間計算（steady_clock） |
| T4-5-2 | HealthEvent 詳細追加 | ✅ | `RuntimeHealthMonitor.h`: HealthEvent に readerIndex/readerEpoch/readerDepth/residencyTimeUs 追加、EVENT_READER_STUCK(3001) 追加、diagnoseRetireStall 宣言追加 |
| T4-5-3 | Shutdown 時診断強化 | ✅ | `RuntimeHealthMonitor.cpp`: tick() に diagnoseRetireStall() 呼び出し追加。epochGap + pendingRetire から Reader 残留を診断 |

### Phase 8: 検証基盤

| # | タスク | 状態 | 変更内容 |
|---|-------|------|---------|
| T8-1 | 7条件自動検証スクリプト | ✅ | `.github/scripts/verify-isr-maturity.ps1`: 8条件（RT delete/lock/retireDSP/VerifyDrained/droppedIntentCount/Diagnose/Coordinator/EpochDomain）を自動検証 |

---

## 変更ファイル一覧

| ファイル | 変更内容 |
|---------|---------|
| `src/audioengine/DSPLifetimeManager.h` | ISRRetireRouter 必須化、Router→EpochDomain 直接委譲 |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | retireDSP()→DSPLifetimeManager::retire() 置換（3箇所）+ include追加 |
| `src/audioengine/AudioEngine.Timer.cpp` | retireDSP()→DSPLifetimeManager::retire() 置換（2箇所）+ include追加 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | retireDSP()→DSPLifetimeManager::retire() 置換（DSPGuard含む2箇所）+ include追加 |
| `src/core/EpochDomain.h` | ReaderSlot に residencyStartTimestampUs 追加、detectStuckReaders に実時間計算追加、<chrono> include追加 |
| `src/audioengine/RuntimeHealthMonitor.h` | HealthEvent に Reader 診断フィールド追加、EVENT_READER_STUCK追加、diagnoseRetireStall宣言 |
| `src/audioengine/RuntimeHealthMonitor.cpp` | tick() に diagnoseRetireStall() 追加、実装追加 |
| `.github/scripts/verify-isr-maturity.ps1` | 新規: 8条件自動検証スクリプト |
