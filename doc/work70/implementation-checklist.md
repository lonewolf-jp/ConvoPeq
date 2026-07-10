# work70 実装チェックリスト

**設計書**: doc/work70/modification-plan-v3.md (v5.38)
**日付**: 2026-07-11

凡例: ✅ 完了 / 🔄 進行中 / ⬜ 未着手 / ❌ 中止

---

## P1-a: publish 経路への handle 登録追加 ✅ 完了

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| A-1 | `PublishCommitResult` + `PublishStageResultTraits` | `AudioEngine.h` | ✅ | stage のみ保持, isCommitted() 委譲 |
| A-2 | `RegistrationContext` + 静的ファクトリ | `AudioEngine.h` | ✅ | needsRegistration/alreadyRegistered/none |
| A-3 | `ScopeExit` RAII テンプレート | `AudioEngine.h` | ✅ | 参照キャプチャコメント付き |
| A-4 | `DSPHandleRuntime::rollbackRegistration()` | `ISRDSPHandle.h/.cpp` | ✅ | CAS Constructing→Reclaimed, state only |
| A-5 | `eraseByHandle()` private helper | `AudioEngine.h` | ✅ | O(n) full scan |
| A-6 | `rollbackDSPHandleRegistration(DSPHandle)` | `AudioEngine.h` | ✅ | CAS→eraseByHandle, bool戻り値 |
| A-7 | `commitRuntimePublication()` | `AudioEngine.h` | ✅ | ScopeExit + invalidate パターン |
| A-8 | Init.cpp publishWorld 置換 | `AudioEngine.Init.cpp` | ✅ | RegistrationContext::none() |
| A-9 | PrepareToPlay 2箇所置換 | `PrepareToPlay.cpp` | ✅ | needsRegistration + 重複削除 |
| A-10 | ReleaseResources 置換 | `ReleaseResources.cpp` | ✅ | RegistrationContext::none() |
| A-11 | Timer 置換 | `Timer.cpp` | ✅ | needsRegistration |
| A-12 | Transition 置換 | `Transition.cpp` | ✅ | needsRegistration + 重複削除 |
| A-13 | PublicationExecutor シグネチャ変更 | `PublicationExecutor.h/.cpp` | ✅ | `DSPHandle existingHandle` + commitRuntimePublication |
| A-14 | Orchestrator 呼び出し修正 | `RuntimePublicationOrchestrator.cpp` | ✅ | `req.newDSP` 渡し |

## P1-b: advanceFade 配線 ✅ 完了

| # | 項目 | ファイル | 状態 |
|:-:|:-----|:---------|:----|
| B-1 | `m_coordinator.advanceFade(numSamples)` | `AudioEngine.Processing.AudioBlock.cpp` | ✅ |

## P1-c: MEM_SNAP 監視強化 ✅ 完了

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| C-1 | `DSPCore::liveCount` MEM_SNAP 出力 | `AudioEngine.Timer.cpp` | ✅ | DC: live= フィールド |
| C-2 | `StereoConvolver::liveCount` MEM_SNAP 出力 | `AudioEngine.Timer.cpp` | ✅ | SC: live= フィールド |
| C-3 | `retiringGeneration` atomic field | `DSPLifetimeManager.h` | ✅ | DSPLifetimeManager 唯一 Authority |
| C-4 | `retire()` 内で generation 保存 | `DSPLifetimeManager.h` | ✅ | lastCommittedRuntimeGeneration_ 参照 |
| C-5 | `retiringGeneration` MEM_SNAP 出力 | `AudioEngine.Timer.cpp` | ✅ | Ret: gen= フィールド |

## その他修正

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| D-1 | DIAG_MKL_MALLOC ガード構造修正 | `DiagnosticsConfig.h` | ✅ | 外側 #if ガード内→外に移動 |

## 変更ファイル一覧

| ファイル | 変更種別 |
|:---------|:--------|
| `src/audioengine/ISRDSPHandle.h` | 宣言追加 `rollbackRegistration()` |
| `src/audioengine/ISRDSPHandle.cpp` | 実装追加 `rollbackRegistration()` |
| `src/audioengine/AudioEngine.h` | 構造体+PublishCommitResult+Traits+RegistrationContext+ScopeExit+eraseByHandle+rollbackDSPHandle+commitRuntimePublication |
| `src/audioengine/AudioEngine.Init.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | 2箇所置換 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/AudioEngine.Timer.cpp` | publishWorld→commitRuntimePublication + MEM_SNAP拡張 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | advanceFade 追加 |
| `src/audioengine/AudioEngine.Transition.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/PublicationExecutor.h` | publish() シグネチャ拡張 |
| `src/audioengine/PublicationExecutor.cpp` | coordinator.publishWorld→commitRuntimePublication |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | executor_.publish() 引数追加 |
| `src/audioengine/DSPLifetimeManager.h` | retiringGeneration atomic 追加 |
| `src/DiagnosticsConfig.h` | DIAG_MKL_MALLOC ガード構造修正 |

## 検証

| 項目 | 状態 | 備考 |
|:-----|:----|:------|
| Debug ビルド | ✅ | エラー 0、警告 `[[nodiscard]]` のみ（意図的） |
| ctest (Debug) | ⬜ | P1-a/b/c 完了後実施 |
| DIAG ログ確認 | ⬜ | 実機テスト時に確認 |
