# pendingOverride 移行 — 削除候補優先順 & 実装プラン

**作成日**: 2026-05-17
**ステータス**: 参照切り分け済み → 実装順決定

---

## 概要

前回の参照ベース切り分けで、「RT 必須」と「pendingOverride 寄せ」が明確に分かれた。
本ドキュメントは、**pendingOverride へ寄せられるもの** 14 個を、**削除実装順** まで具体化する。

### 大原則

- グループごとに **1 つの PR** を想定
- グループ内の削除は同時実行（パラレル）
- 削除順序は **参照複雑度の低い順** → **依存度が低い順**

---

## グループ 1: Tail & Mixed Phase 制御（最優先）

### メンバ

```
- tailProcessingMode
- tailRolloffStartHz
- tailRolloffStrength
- partitionTailStrength
```

### 理由

- これらは **IR 再構築時の条件値** であり、**実行中の音声処理** では直接参照されない
- `pendingOverride` 化しても実行中のサウンドは壊しにくい
- 参照箇所が比較的少ない（5-10 箇所程度）

### 削除スコープ

| 対象ファイル | 削除対象 | 残すもの | 難度 |
|-----------|--------|--------|------|
| `ConvolverProcessor.h` | メンバ変数 4 個 + getter 4 個 | なし | ⭐ |
| `ConvolverProcessor.cpp` | 初期化、getter 実装 | なし | ⭐ |
| `ConvolverProcessor.State.cpp` | snapshot/restore 処理 | なし | ⭐ |
| `ConvolverAudioEngine.cpp` | setTail**/state sync 関数 | rebuild 条件への read | ⭐⭐ |
| `ConvolverAudioEngine.h` | 対応関数宣言 | なし | ⭐ |

### 実装順

**Step 1**: ヘッダから宣言削除 → getter 削除

```cpp
// 削除対象
float getTailRolloffStartHz() const;
float getTailRolloffStrength() const;
void setTailProcessingMode(int m);
void setTailRolloffStartHz(float hz);
void setTailRolloffStrength(float str);
void setPartitionTailStrength(float str);
```

**Step 2**: ConvolverProcessor.cpp から get/set 実装を削除

**Step 3**: ConvolverProcessor.State.cpp から snapshot/restore を削除

**Step 4**: ConvolverAudioEngine.cpp から sync 関数と setter を削除

- ただし `rebuild()` / `recalculateTailPlan()` 内の **読み取り** は残す → pendingOverride から read に変更

**Step 5**: ビルド & Strict Atomic Scan 実行

### リスク評価

| リスク | 評価 | 対策 |
|-------|------|------|
| 参照漏れ | 低 | `rg "tailProcessingMode\|tailRolloff\|partitionTail"` で残存確認 |
| rebuild 処理への影響 | 低 | pendingOverride 化しても読み取り元が同じ |
| state 保存の欠落 | 低 | snapshot/restore 削除時に全箇所確認 |

### テスト

```bash
# 1. ビルド確認
cmake --build build --config Release

# 2. Lint 確認
powershell -NoProfile -ExecutionPolicy Bypass -File ".github/scripts/check-src-atomic-dotcall.ps1"

# 3. 機能テスト（手動）
# - IR ロード後、tail 設定を変更（pendingOverride 経由）
# - rebuild が正しく完了することを確認
# - state 保存・復元が正しく行われることを確認
```

---

## グループ 2: IR Length & Mixed Transition（中優先）

### メンバ

```
- mixedTransitionStartHz
- mixedTransitionEndHz
- mixedPreRingTau
- targetIRLengthSec
- autoDetectedIRLengthSec
- irLengthManualOverride
```

### 理由

- これらも **IR 構築・ロード時の条件値** が主体
- mixed transition と IR length は state 保存時に必要 → state 機構との綿密な確認が必要
- グループ 1 より参照箇所が多い（20-30 箇所）が、依存は局所的

### 削除スコープ

| 対象ファイル | 削除対象 | 残すもの | 難度 |
|-----------|--------|--------|------|
| `ConvolverProcessor.h` | メンバ変数 6 個 + getter 6 個 | なし | ⭐⭐ |
| `ConvolverProcessor.cpp` | 初期化、getter 実装 | なし | ⭐ |
| `ConvolverProcessor.State.cpp` | snapshot/restore 処理（特に IR length） | なし | ⭐⭐⭐ |
| `ConvolverAudioEngine.cpp` | mixed transition 読み取り、IR length sync | rebuild 判定での read | ⭐⭐⭐ |
| `ConvolverAudioEngine.h` | setter/getter 宣言 | なし | ⭐ |
| `IRLoader.cpp` | targetIRLengthSec, autoDetectedIRLengthSec 読み取り | なし | ⭐⭐ |

### 実装順

**Step 1**: 参照マップ作成

```bash
rg "targetIRLengthSec|autoDetectedIRLengthSec|irLengthManualOverride|mixedTransitionStartHz|mixedTransitionEndHz|mixedPreRingTau" src/ --type cpp --context 3 > /tmp/ir_length_refs.txt
```

**Step 2**: ConvolverProcessor.h から宣言削除（6 メンバ + 6 getter）

**Step 3**: ConvolverProcessor.cpp から初期化・getter 実装削除

**Step 4**: ConvolverProcessor.State.cpp から snapshot/restore を **慎重に** 削除

- ここで IR length は state に含まれるため、削除箇所の確認は 2 重チェック推奨
- pendingOverride のどの項目に対応するか明記すること

**Step 5**: ConvolverAudioEngine.cpp から setter 削除 & rebuild 条件内の read を pendingOverride に変更

- `rebuildNeedsIRLength()` や `selectBestFFTSize()` などの判定ロジックはそのまま
- read 元を `atomicIRLengthSec.load()` → `pendingOverride.targetIRLengthSec` に変更

**Step 6**: IRLoader.cpp での read を確認 & pendingOverride 読み取りに変更

**Step 7**: ビルド & Lint

### リスク評価

| リスク | 評価 | 対策 |
|-------|------|------|
| state 復元時のロジック欠落 | **高** | snapshot/restore 削除時に全テスト実施 |
| IR length 判定での read 漏れ | 中 | rebuild 条件に関わる全関数をコードレビュー |
| mixed transition 参照の散在 | 中 | 参照マップで漏れなく確認 |

### テスト

```bash
# 1. ビルド確認
cmake --build build --config Release

# 2. Lint
powershell -File ".github/scripts/check-src-atomic-dotcall.ps1"

# 3. 機能テスト（重点）
# - IR ロード → state 保存 → 再起動 → 復元
#   → targetIRLengthSec / autoDetectedIRLengthSec / irLengthManualOverride が正しく復元されることを確認
# - mixed transition パラメータ変更 → rebuild 判定が正しく行われることを確認
# - 複数 SR / Buffer size での IR rebuild 成功を確認
```

---

## グループ 3: UI & 非実行時設定（低優先）

### メンバ

```
- rebuildDebounceMs
- targetUpgradeFFTSize
- enableProgressiveUpgrade
- maxCacheEntries
```

### 理由

- これらは完全に **UI 設定 / パフォーマンス チューニング** の類
- 実行中の音声処理には直結しない
- 参照箇所が少ない（3-8 箇所）

### 削除スコープ

| 対象ファイル | 削除対象 | 残すもの | 難度 |
|-----------|--------|--------|------|
| `ConvolverProcessor.h` | メンバ変数 4 個 + getter 4 個 | なし | ⭐ |
| `ConvolverProcessor.cpp` | 初期化、getter 実装 | なし | ⭐ |
| `ConvolverProcessor.State.cpp` | snapshot/restore（UI/debug 用） | なし | ⭐ |
| `ConvolverAudioEngine.cpp` | FFT upgrade 判定での read | なし | ⭐⭐ |
| UI コンポーネント（不要に応じて） | パラメータ setter/getter UI バインディング | なし | ⭐ |

### 実装順

**Step 1**: ConvolverProcessor.h から宣言削除

**Step 2**: ConvolverProcessor.cpp から初期化・getter 削除

**Step 3**: ConvolverProcessor.State.cpp から snapshot/restore 削除

**Step 4**: ConvolverAudioEngine.cpp での read を pendingOverride に変更

- `shouldUpgradeFFTSize()` や debounce 判定で read している場合、pendingOverride から read に置換

**Step 5**: UI コンポーネント内のダイアログ/パラメータ表示を確認

- もし UI 側に対応する setter/getter バインディングがあれば、pendingOverride への連携に変更

**Step 6**: ビルド & Lint

### リスク評価

| リスク | 評価 | 対策 |
|-------|------|------|
| UI の欠落 | 低 | UI レイアウトで対応パラメータが消えていないか確認 |
| FFT upgrade 判定への影響 | 低 | rebuild 判定ロジックは変わらない（read 元が変わるのみ） |
| state 保存の欠落 | 低 | debug/UI 用なので影響は限定的 |

### テスト

```bash
# 1. ビルド確認
cmake --build build --config Release

# 2. Lint
powershell -File ".github/scripts/check-src-atomic-dotcall.ps1"

# 3. 機能テスト
# - UI で FFT size upgrade パラメータ変更
# - Rebuild debounce が正しく機能することを確認
# - キャッシュサイズ設定が反映されることを確認（メモリ使用量で確認可能）
```

---

## 全体実装スケジュール

### Phase 1: 最優先グループ（グループ 1）

**期間**: 1-2 日
**PR**: `feat: migrate tail/phase control to pendingOverride`

```bash
# リスト削除
rg "tailProcessingMode|tailRolloffStartHz|tailRolloffStrength|partitionTailStrength" src/ --type cpp | wc -l
# → 5-10 個所程度のはず

# 削除実施
# 1. .h from declaration
# 2. .cpp から getter/setter/sync
# 3. rebuild 条件での read は pendingOverride に変更
# 4. build & lint
```

### Phase 2: 中優先グループ（グループ 2）

**期間**: 2-3 日
**PR**: `feat: migrate ir-length and mixed-transition to pendingOverride`

```bash
# 参照マップ確認（事前）
rg "targetIRLengthSec|autoDetectedIRLengthSec|irLengthManualOverride|mixedTransitionStartHz|mixedTransitionEndHz|mixedPreRingTau" src/ --type cpp | wc -l
# → 20-30 個所のはず

# state 復元ロジック確認（重点）
# ConvolverProcessor.State.cpp の snapshot/restore で該当メンバを確認
# → 削除前に完全に理解した上で実装

# 削除実施
# 1. .h from declaration
# 2. .cpp から 実装削除
# 3. State.cpp から state 機構削除（最も慎重）
# 4. ConvolverAudioEngine.cpp の rebuild 判定読み取り変更
# 5. IRLoader.cpp の read 変更
# 6. build & lint & state 復元テスト
```

### Phase 3: 低優先グループ（グループ 3）

**期間**: 1 日
**PR**: `feat: migrate ui-settings to pendingOverride`

```bash
# 参照確認
rg "rebuildDebounceMs|targetUpgradeFFTSize|enableProgressiveUpgrade|maxCacheEntries" src/ --type cpp | wc -l
# → 3-8 個所のはず

# 削除実施
# 1. .h from declaration
# 2. .cpp から実装削除
# 3. rebuild/upgrade 判定の read 変更
# 4. build & lint
```

---

## 各グループの難度 & リスク サマリー

| グループ | 削除数 | 参照数 | 難度 | リスク | 推奨 PR 範囲 |
|----------|-------|-------|------|--------|----------|
| グループ 1（Tail） | 4 | 5-10 | ⭐ | 低 | 単一 PR 推奨 |
| グループ 2（IR Length） | 6 | 20-30 | ⭐⭐⭐ | 中～高 | 単一 PR（state 確認必須） |
| グループ 3（UI Settings） | 4 | 3-8 | ⭐ | 低 | 単一 PR 推奨 |

---

## 今すぐ実装すべき順序

### 1️⃣ **グループ 1（Tail）** → 最速で削除

- 難度低、リスク低、参照少ない
- PR 作成 & merge に 2-3 時間で完了可能
- グループ 2 の実装前に確認を取る意味でも有効

### 2️⃣ **グループ 3（UI Settings）** → 次点で削除

- グループ 1 とほぼ同じ難度・リスク
- グループ 2 より参照が少ないので、グループ 1 確認後に並行実施可能

### 3️⃣ **グループ 2（IR Length）** → 最後に慎重に

- state 機構との綿密な連携が必要
- グループ 1 & 3 で削除パターンが確立してから実施
- Phase 2 時点では十分な時間を確保すること

---

## 次ステップ

1. グループ 1 の参照マップを再確認（`rg` で全箇所列挙）
2. グループ 1 の削除パターンを確定 → PR 作成
3. グループ 1 merge 後、グループ 3 へ
4. 双方 merge 後、グループ 2 へ進む（state 機構 review は詳細に）
