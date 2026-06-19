# 実装チェックリスト: Publication Validator Silent Rejection 修正

**ドキュメントID**: WR-48-101
**作成日**: 2026-06-19
**基盤文書**: `doc/work48/publication_validator_silent_rejection_report.md`
**HEAD修正優先度**: P0 > P1 > P2

---

## 凡例

- ✅ **完了**
- 🔄 **作業中**
- ⏸ **保留中**
- ❌ **未着手**

---

## P0: Silent Success 修正（HEAD と Working Tree に共通、最優先）

| # | 項目 | ファイル | 状態 | 備考 |
|---|---|---|---|---|
| P0-1 | `publishWorld()` → `PublishStageResult` 返却に変更 | `RuntimePublicationCoordinator.h` | ✅ | void → `[[nodiscard]] PublishStageResult`。Validation拒否時は `Rejected`、異常時は `Failed`、成功時は `Success` |
| P0-2 | `PublicationExecutor::publish()` 成否伝播 | `PublicationExecutor.cpp` | ✅ | `coordinator.publishWorld()` の戻り値を `PublishStageResult` で受け取り `PublishResult` にマッピング |
| P0-3 | Validator 失敗理由の HealthMonitor 伝播 | `AudioEngine.h (Bridge)` | ✅ | Bridge の `validatePublicationNonRt` で `emitValidationEvent` を発火。`runPublicationPrecheckNonRt` の重複 emit は削除 |

**P0-1 + P0-2 はセット。片方だけでは無意味。→ 両方同時に修正完了済み。**

---

## P1: Validator 過剰拒否の修正（Working Tree commit 前に必要）

| # | 項目 | ファイル | 状態 | 備考 |
|---|---|---|---|---|
| P1-1 | `validateResources`: ditherBitDepth=32 を許容 | `RuntimePublicationValidator.cpp:68` | ✅ | `kAdaptiveBitDepthValues = {16, 24, 32}` との整合性を確保 |
| P1-2 | `validateResources`: noiseShaperType=3 (Fixed15Tap) を許容 | `RuntimePublicationValidator.cpp:72` | ✅ | `NoiseShaperType::Fixed15Tap = 3` が正規設定として許容される |
| P1-3 | `placeholderDSP→prepare()` の dither/NS 継承問題の確認 | `PrepareToPlay.cpp:230` | ✅ | 調査完了。placeholderDSP は原子値(ditherBitDepth/noiseShaperType)を直接読む。P1-1/P1-2 の修正で reject されなくなったため問題なし |

---

## P2: アーキテクチャ改善・二次的問題

| # | 項目 | ファイル | 状態 | 備考 |
|---|---|---|---|---|
| P2-1 | `validateTopology`: runtimeUuid==0 チェックを選択肢A（Authoritative不変条件）に変更 | `RuntimePublicationValidator.cpp:44-46` | ✅ | `generation > 0 && runtimeUuid == 0` → runtimeUuid=0 時の不変条件検証（transitionActive/hasFadingRuntime/fadingRuntimeUuid の整合性） |
| P2-2 | 二重Validation解消: `runPublicationPrecheckNonRt` の static Validator 呼び出し削除 | `AudioEngine.Commit.cpp` | ✅ | Bridge の `validator_` と重複していた static Validator 呼び出しを削除。Precheck はエンジン固有チェックに専念 |
| P2-3 | テスト修正: generation=0 仮定の修正、新テスト追加 | `PublicationValidatorIsolationTests.cpp` | ✅ | `ValidateTopology_Bootstrap_Accept` の generation=0→1。dither=32/NS=3 許容テスト追加。runtimeUuid=0 Accept/Reject テスト追加。拒否テスト条件更新 |
| P2-4 | `setNoiseShaperType` typeName mapping 漏れ | `AudioEngine.Parameters.cpp:430-434` | ✅ | `Fixed15Tap` の typeName 判定を追加。診断ログが正しい typeName を表示するよう修正 |

---

## 進捗サマリー

| 区分 | 総数 | ✅ 完了 | 🔄 作業中 | ❌ 未着手 |
|---|---|---|---|---|
| P0 | 3 | 3 | 0 | 0 |
| P1 | 3 | 3 | 0 | 0 |
| P2 | 4 | 4 | 0 | 0 |
| **合計** | **10** | **10** | **0** | **0** |

---

## 変更影響マップ

```
RuntimePublicationCoordinator.h          ← P0-1: publishWorld()戻り値変更
    ↓
PublicationExecutor.cpp                  ← P0-2: 成否伝播
    ↓
RuntimePublicationOrchestrator.cpp       ← 変更不要（既に結果チェック実装済み）
    ↓
AudioEngine (Bridge層)
    ↓
RuntimePublicationValidator.cpp          ← P1-1, P1-2, P2-1
    ↓
AudioEngine.Commit.cpp                   ← P0-3, P2-2
    ↓
PublicationValidatorIsolationTests.cpp   ← P2-3
    ↓
AudioEngine.Parameters.cpp               ← P2-4
```
