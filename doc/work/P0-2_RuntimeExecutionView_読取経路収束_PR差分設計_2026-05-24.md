# P0-2 RuntimeExecutionView 読取経路収束 専用PR 差分設計

作成日: 2026-05-24  
対象ブランチ: `feature/p0-1-observetoken-formalization` から派生  
上位計画: `doc/work/bridge_runtime_migration_plan.md`  
厳守規約: `doc/work/ISR_Bridge_Runtime_AI_暴走防止規約.md`

---

## 1. このPRの責務（1PR=1責務）

本PRは **RuntimeExecutionView{snapshot, local} への読取経路収束** のみを扱う。

- 許可: 読取経路の明示化・重複読取の削減・参照経路の整流
- 禁止: publish/retire/crossfade/latency/ownership の挙動変更
- 禁止: RuntimeGraph の責務追加

---

## 2. 非スコープ（変更禁止）

- クロスフェード開始/完了条件
- retire queue/epoch reclaim
- validator/CI 仕様変更
- cleanup（削除系）
- mutable state の新設

---

## 3. 現状課題（P0-2観点）

- `AudioEngine` 内で snapshot / runtime graph の読取が複数パターンで散在
- observe path が明示的に統一されておらず、将来の逸脱点が増えやすい

---

## 4. 差分方針（最小・非破壊）

1. 既存API互換を維持
2. 読取経路を `RuntimeExecutionView` 相当の補助関数に寄せる
3. 挙動変更は入れない（計算順序・条件分岐の意味は不変）

---

## 5. 変更候補ファイル（最小）

- `src/audioengine/AudioEngine.h`（読取ヘルパーの整理）
- `src/audioengine/AudioEngine.Processing.AudioBlock.cpp`（読取入口の統一）
- 必要時のみ `AudioEngine.Processing.BlockDouble.cpp` を追従

> 1PR=1責務のため、上記以外へ波及しないこと。

---

## 6. 機械判定可能な完了条件

1. 新規 observe path 追加 0
2. `RuntimeExecutionView` 経由読取の適用率が対象範囲で増加
3. crossfade/retire 関連ロジック差分 0
4. Debug build 成功

---

## 7. 検証計画

- 検索確認: observe 呼び出し点の増減
- 差分確認: crossfade / retire への変更混入なし
- ビルド確認: Debug Build
- 診断確認: 変更ファイル diagnostics 0

---

## 8. ロールバック

- 変更ファイル限定で revert 可能
- 挙動変更を持たない構造整流のみのため巻き戻し容易

---

## 9. レビュー観点（固定）

1. callback中 snapshot固定（IR-A）を壊していないか
2. observe path増殖を抑制できているか
3. dual authority 暴走に繋がる変更がないか
4. purity志向の過剰抽象化になっていないか

---

## 10. 実装順（このPR内）

1. 読取入口の現状マップ作成（対象関数限定）
2. 最小ヘルパー導入（必要最小限）
3. 呼び出し置換（意味不変）
4. 検証（検索/差分/build/diagnostics）
