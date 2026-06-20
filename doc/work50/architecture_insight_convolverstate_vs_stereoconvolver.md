# アーキテクチャ分析: ConvolverState 経路 vs StereoConvolver 経路

**作成日**: 2026-06-20
**元文書**: `bug_review_verification_report.md` (No.3, No.7 調査にて発見)

---

## 発見の経緯

10件のバグレビュー検証中、No.3 (targetIRLengthSec) と No.7 (numPartitions) の調査過程で、ConvoPeq の畳み込みエンジンに2つの独立した経路が存在することが判明した。

---

## 2系統の概要

### 系統A: StereoConvolver 経路（アクティブ）

| 要素 | 所在地 |
|------|--------|
| **エンジン** | `StereoConvolver` (`src/ConvolverProcessor.h` L636-750) |
| **内部実装** | `MKLNonUniformConvolver` (`src/MKLNonUniformConvolver.cpp/.h`) |
| **データ** | 時間領域IR (`irData[0]`, `irData[1]`, `irDataLength`) |
| **FFT処理** | `MKLNonUniformConvolver::SetImpulse()` 内部で自前実行 |
| **更新関数** | `switchEngineOnMessageThread()` → `exchangeActiveEngine()` |
| **読出関数** | `loadActiveEngine()` |
| **使用元** | `ConvolverProcessor::process()` (Audio Thread) |
| **リソース** | StereoConvolver 経由で `DeferredDeletionQueue` に退役 |
| **状態** | ✅ **運用中** |

### 系統B: ConvolverState 経路（設計遺産）

| 要素 | 所在地 |
|------|--------|
| **エンジン** | `ConvolverState` (`src/ConvolverState.h` L43-221) |
| **内部実装** | `ConvolverRuntime` (`src/ConvolverRuntime.h`) |
| **データ** | 周波数領域パーティション (`partitionData`) |
| **FFT処理** | `ConvolverState` 内の `fftHandle` (DFTI descriptor) — 未使用 |
| **更新関数** | `updateConvolverState()` → `rcuSwapper.swap()` |
| **読出関数** | `getConvolverState()` / `rcuSwapper.getState()` |
| **使用元** | `AudioEngine.Snapshot.cpp` (UIスナップショット、`stateId` のみ参照) |
| **リソース** | `SafeStateSwapper` の retired キュー経由で退役 |
| **状態** | ❌ **Audio Thread から未使用（デッドコード）** |

---

## エビデンス

### 系統Bが Audio Thread で読まれていない証拠

```
grep/Serena/AiDex/CodeGraph/semble 全ツール確認:
- ConvolverState::partitionData の Audio Thread 読み取り: 0件
- ConvolverState::numPartitions のアルゴリズム使用: 0件（コピー専用）
- ConvolverRuntime の process() 内参照: 0件
  - runtime.reallocate(): LoadPipeline.cpp:500（書込みのみ）
  - runtime.clear(): Lifecycle.cpp:420（クリアのみ）
- rcuSwapper.getState() の Audio Thread 呼び出し: 0件
  - LoadPipeline.cpp:219（isCacheEntrySafeToDelete 内、Message Thread）
- getConvolverState() の呼出元:
  - AudioEngine.Snapshot.cpp:28（UIスナップショット、stateIdのみ参照）
```

### 系統Aが Audio Thread で使われている証拠

```
ConvolverProcessor::process() の冒頭 (Runtime.cpp:214):
  auto* conv = loadActiveEngine(std::memory_order_acquire);
  → StereoConvolver*

loadActiveEngine() の実装 (ConvolverProcessor.h:913):
  return fromEngineBits(convo::consumeAtomic(m_activeEngineBits, ...));
```

---

## 今後の対応方針

### 短期（P0/P1）

1. **No.1 修正**: `IRCacheKey::operator<` の SWO 違反を修正（10分作業）
2. **No.6 確認**: CMA-ES 初期値修正がビルド可能であることを確認（修正済み）

### 中期（設計整理項目）

1. **ConvolverState 系統の整理方針決定**:
   - オプションA: 維持（現状のUIスナップショット用途で十分）
   - オプションB: 縮退（`partitionData` を削除し `stateId` のみ保持）
   - オプションC: 削除（`ConvolverState` + `ConvolverRuntime` + `SafeStateSwapper` を全削除）
2. **CacheManager キャッシュフォーマット**:
   - 現状 `partitionData` を保存しているが、削除時は後方互換性対応が必要
   - `PreparedIRState` の `partitionData` / `partitionSizeBytes` フィールドの存廃判断

### 注意点

- `ConvolverState` の `fftHandle` は MKL DFTI descriptor → 未使用だが解放漏れ防止のため `cleanup()` の動作確認必須
- `ConvolverRuntime` の `overlapBuffer` / `inputBuffer` / `outputBuffer` は `mkl_malloc` 確保 → 解放漏れ防止のためデストラクタパス確認必須
- `SafeStateSwapper` の RCU retire キューは `ConvolverState` の解放に使用 → 削除時は退役パスの再設計が必要
