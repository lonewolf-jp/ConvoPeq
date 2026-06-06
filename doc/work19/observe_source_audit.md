# Observe Source Audit

作成日: 2026-06-06
ベース: Serena find_referencing_symbols + grep 調査

---

## 1. getActiveRuntimeDSP() 全使用箇所

| # | ファイル | 行 | 用途分類 | 説明 |
|---|---|---|---|---|
| 1 | `RuntimePublicationOrchestrator.cpp` | 30 | **Semantic** | クロスフェード判定の入力として oldDSP を取得 |
| 2 | `DSPLifetimeManager.h` | 35 | **Execution** | getActive() の委譲 |
| 3 | `AudioEngine.h` (hasActiveRuntimeDSP) | 1486 | **Execution** | nullptr 確認 |
| 4 | `AudioEngine.h` (releaseActiveRuntimeDSP) | 1491 | **Execution** | 解放操作の前処理 |
| 5 | `AudioEngine.CtorDtor.cpp` | 59,69,79 | **Execution** | デストラクタでのクリーンアップ |
| 6 | `AudioEngine.Processing.Latency.cpp` | 83 | **Execution** | レイテンシ計算の入力 |
| 7 | `AudioEngine.Processing.PrepareToPlay.cpp` | 243 | **Execution** | Builder への引数 (Execution Object) |
| 8 | `AudioEngine.Processing.ReleaseResources.cpp` | 93,99,137 | **Execution** | リソース解放 |
| 9 | `AudioEngine::logRuntimeTransitionEvent()` | 2761 | **Execution** | ログ出力用 |

**Semantic 用途: 1箇所** → `RuntimePublicationOrchestrator::trySubmit()`
**Execution 用途: 8箇所以上**

## 2. setActiveRuntimeDSP() 全使用箇所

(調査予定)

## 3. getActiveRuntimeDSPHandle() 全使用箇所

(調査予定)

## 4. exchangeFadingRuntimeDSP() 全使用箇所

(調査予定)

## 5. 結論

- `getActiveRuntimeDSP()` の Semantic 用途は **`RuntimePublicationOrchestrator::trySubmit()` の1箇所のみ**
- PR-1 で CrossfadeAuthority を RuntimeWorld 化した後、この Semantic 用途は `runtimeStore.observe()` に置換される
- Execution 用途 (8箇所以上) は変更不要
