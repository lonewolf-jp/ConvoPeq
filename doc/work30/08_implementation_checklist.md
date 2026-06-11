# 実装チェックリスト — Practical Stable ISR Bridge Runtime 改修 v7.3

**作成日**: 2026-06-11
**根拠文書**: `doc/work30/07_remediation_plan_final.md`
**進捗管理**: 各タスク完了後に build green + grep 確認

---

## Phase 0: 即時修正（P0）✅ BUILD GREEN

### P0-A: IEpochProvider 拡張 ✅

- [x] `src/core/IRetireProvider.h` — `pendingRetireCount()` / `drainAll()` 純粋仮想関数追加
- [x] `src/core/EpochDomain.h` — `override` 明示 + `enqueueRetire()` 2-param public 維持
- [x] `src/core/ISRRetireRouter.h` — `epochContext_` / `pendingRetireFn_` / `drainAllFn_` 関数ポインタ削除
- [x] `src/audioengine/ISRRetireRouter.cpp` — dynamic_cast 削除、委譲に変更
- [x] `SnapshotCoordinator` のコンパイル確認（IEpochProvider 経由の呼び出し）

### P0-B: EQProcessor Fallback 削除 ✅

- [x] `src/eqprocessor/EQProcessor.Core.cpp` — Fallback 経路削除 + `if (m_retireCoordinator == nullptr) return false;`

## Phase 1: 監視・制御閉ループ（P1）

### P1-A: RCUReader rootEnterSucceeded_ ✅ BUILD GREEN

- [x] `src/core/RCUReader.h` — `bool rootEnterSucceeded_` 追加
- [x] `enter()` 成功時 `rootEnterSucceeded_ = true`
- [x] `enter()` 失敗時 `rootEnterSucceeded_ = false`
- [x] `exit()` ネスト対応リセット（previousDepth==0/1 時のみ false）

### P1-B: HealthMonitor → Admission 閉ループ ⏳ PENDING

### P1-C: Crossfade Event SPSC ⏳ PENDING

## Phase 2: 防御的改善（P2）

### P2-A: Deprecated API 整理 ✅ BUILD GREEN

- [x] `EpochDomain.h` — `advanceEpoch()` private 化
- [x] `EpochDomain.h` — `reclaimRetired()` private 化
- [x] `EpochDomain.h` — `enqueueRetire()` 4-param private 化

### P2-B / P2-C ⏳ PENDING

## Phase 3: ⏳ PENDING

## Practical Items: ⏳ PENDING

## 検証結果 ✅ BUILD GREEN

- [x] **Debug build green** (127/128 → link OK)
- [x] `grep "dynamic_cast<EpochDomain\*>" src/` = 0 ✅
- [x] `grep "Fallback.*direct EpochDomain" src/` = 0 ✅
