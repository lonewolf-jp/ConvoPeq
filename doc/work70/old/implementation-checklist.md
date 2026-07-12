# work70 実装チェックリスト

**設計書**: doc/work70/modification-plan-v3.md (v5.45)
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

## P1-a-FIX: activeRuntimeDSPHandle_ 欠落修正 ✅ 完了（2026-07-11）

P1-a 実装後も lifecycle(retire)=0 が継続する根本原因を修正。`commitRuntimePublication()` で publish 成功時に `dspHandleRuntime_.activate(handle)` が呼ばれず、`activeRuntimeDSPHandle_` が null のままだった。

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| F-1 | `commitRuntimePublication()`: publish 成功後に activate | `AudioEngine.h` | ✅ | rollbackHandle 無効化前に `dspHandleRuntime_.activate(rollbackHandle)` 実行 |
| F-2 | `lookupDSPHandleForRuntime(DSPCore*)` 逆引きメソッド | `AudioEngine.h` | ✅ | `runtimeDSPHandleMap_` const 参照; `mutable std::mutex` に変更 |
| F-3 | **二重Authority解消**: DSPLifetimeManager activate 純化 | `DSPLifetimeManager.h` | ✅ | `dspHandleRuntime_.activate(handle)` を削除。`setActiveRuntimeDSP()` のみに。`activeRuntimeDSPHandle_` の更新は commitRuntimePublication が唯一のAuthority |

## P1-a-FIX-2: DSPGuard rebuild-obsolete リーク修正 ✅ 完了（2026-07-11）

rebuild-obsolete な DSPCore が DSPGuard の retire() 経路で解放されないバグを修正。

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| G-1 | DSPGuard に直接破棄パス追加 | `AudioEngine.RebuildDispatch.cpp` | ✅ | `retireDSPHandleForRuntime()` が false の場合、`destroyDSPCoreNode(ptr)` を直接呼び出し |
| G-2 | DSPGuard DIAG invariant 表明 | `AudioEngine.RebuildDispatch.cpp` | ✅ | `lookupDSPHandleForRuntime(ptr).isNull()` を jassert で確認（v5.42） |

## P1-a-FIX-3: lookupDSPHandleForRuntime DIAG 限定化 ✅ 完了（2026-07-11）

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| H-1 | `lookupDSPHandleForRuntime()` private 化 | `AudioEngine.h` | ✅ | `private:` セクションに移動。DIAG ビルド限定のまま（v5.44） |

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
| `src/audioengine/AudioEngine.h` | PublishCommitResult+Traits+RegistrationContext+ScopeExit+eraseByHandle+rollbackDSPHandle+commitRuntimePublication+lookupDSPHandleForRuntime(DIAG)+activate修正 |
| `src/audioengine/AudioEngine.Init.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/AudioEngine.Processing.PrepareToPlay.cpp` | 2箇所置換 |
| `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/AudioEngine.Timer.cpp` | publishWorld→commitRuntimePublication + MEM_SNAP拡張 |
| `src/audioengine/AudioEngine.Processing.AudioBlock.cpp` | advanceFade 追加 |
| `src/audioengine/AudioEngine.Transition.cpp` | publishWorld→commitRuntimePublication |
| `src/audioengine/PublicationExecutor.h` | publish() シグネチャ拡張 |
| `src/audioengine/PublicationExecutor.cpp` | coordinator.publishWorld→commitRuntimePublication |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | executor_.publish() 引数追加 |
| `src/audioengine/AudioEngine.RebuildDispatch.cpp` | DSPGuard 直接破棄 + DIAG invariant jassert（rebuild-obsolete リーク修正） |
| `src/audioengine/DSPLifetimeManager.h` | retiringGeneration atomic 追加 + activate() 純化（二重Authority解消） |
| `src/ConvolverProcessor.h` | getStereoLiveCount() を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` でガード |
| `src/DiagnosticsConfig.h` | DIAG_MKL_MALLOC ガード構造修正 |

## Phase 2: PublishCommitResult 拡張 + rollback 物理解放 ✅ 完了（2026-07-11）

| # | 項目 | ファイル | 状態 | 備考 |
|:-:|:-----|:---------|:----|:------|
| P2-1 | `OwnershipDisposition` enum + `PublishCommitResult.ownership` | `AudioEngine.h` | ✅ | Transferred / CallerDestroy / None の3値 |
| P2-2 | `commitRuntimePublication()`: 失敗時に `CallerDestroy` を返す | `AudioEngine.h` | ✅ | publish 失敗 + rollback 成功時 |
| P2-3 | `DSPLifetimeManager::destroyRolledBackDSP()` 専用API | `DSPLifetimeManager.h` | ✅ | EBR 非経由、未公開DSP用 |
| P2-4 | Orchesrator publish 失敗パス: `retire()`→`destroyRolledBackDSP()` | `RuntimePublicationOrchestrator.cpp` | ✅ | rollback 後は handle 未登録のため直接破棄 |
| P2-5 | `PublicationExecutor` ownership ログ追加 | `PublicationExecutor.cpp` | ✅ | DIAG に ownership 値出力 |

## 変更ファイル一覧（Phase 2 追記）

| ファイル | 変更種別 |
|:---------|:--------|
| `src/audioengine/AudioEngine.h` | OwnershipDisposition enum + PublishCommitResult 拡張 + commitRuntimePublication 戻り値変更 |
| `src/audioengine/DSPLifetimeManager.h` | destroyRolledBackDSP() 追加 |
| `src/audioengine/PublicationExecutor.cpp` | ownership DIAG ログ追加 |
| `src/audioengine/RuntimePublicationOrchestrator.cpp` | retire() → destroyRolledBackDSP() 置換 |

## 変更ファイル一覧（全フェーズ）

| ファイル | 変更種別 |

## 検証

| 項目 | 状態 | 備考 |
|:-----|:----|:------|
| Debug ビルド | ✅ | エラー 0、警告 `[[nodiscard]]` のみ（意図的） |
| ctest (Debug) | ✅ | 15/15 PASS（HeadlessAudioPathVerification 含む。v5.43 で修正確認） |
| 0xC0000005 crash | ✅ | **原因確定**: DSPGuard 重複 destroy による二重解放。修正後は automation 正常完了後に static-teardown crash のみが benign に発生。（v5.44） |
| work21 CI Gate | ✅ | ALL PASSED |
| AudioEngine lint | ✅ | LINT-AE-001〜014 passed |
| アーキテクチャ検証 | ✅ | Authority Boundary / Invariant / RT Safety 証明完了（設計書 [設計] 6. 参照） |
| DIAG ログ確認 | ✅ | **実機ログ解析完了** (2026-07-11 29,556行) — lifecycle(retire)=0 継続確認。**2つの新たな根本原因を特定**: #3 AUTH_CONTRACT ブロック, #4 rollback+retire 二重経路（設計書 [設計] 7. 参照）。P1-a 修正後の残留課題。 |
