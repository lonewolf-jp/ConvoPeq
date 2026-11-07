# #12 TruePeakDetector の非有限入力ガード

- 実施: 2026-11-07
- 基準: `main` = `ae91385d`（最新 `ConvoPeq.md` は `7faf9fa9` のソースを反映）
- 承認: Owner 指示（#12 の Hardening）
- 前提: #12 はバグではなく堅牢化。対象を `TruePeakDetector` の入力境界に限定する

## 1. 実装前の確認

- 該当箇所: `TruePeakDetector::processBlock`（`TruePeakDetector.cpp`）。`ConvoPeq.md` の L147509 と一致。
- 現行動作: 入力の NaN/Inf をそのまま `interpolateStage` の履歴へコピーする。履歴が非有限になると FIR 出力が非有限になり、履歴から流れ出るまで検出値が落ちる。
- 本番経路では手前のスクラブで非有限値を防いでいるため、通常運用では発生しない。
- RT 制約: `processBlock` は Audio Thread から呼ばれる。確保・ロック・例外の追加は不可。
- 既存テスト: `TruePeakDetector` を直接検証するテストは無かった。

### 受入条件（実装前に対象と条件を確定）

1. 非有限値を含むブロックを処理しても、その後の有限入力に対する出力が、同じ実効入力（非有限値を 0 に置換）を受けた参照検出器と一致する。
2. `processBlock` の戻り値が非有限にならない。
3. 有限入力のみの経路ではコピーを行わず、出力が変わらない。
4. `processBlock` 内でメモリ確保・ロックを行わない。

## 2. 変更内容

| ファイル | 変更 |
|---|---|
| `src/TruePeakDetector.h` | 非有限入力の遮断用バッファ `sanitizedInput[2]` と容量を追加（`prepare` で確保） |
| `src/TruePeakDetector.cpp` | `processBlock` で入力の非有限値を検出し、該当ブロックだけ 0 に置換したコピーを使用。ピーク計算後に非有限値の防御クランプを追加 |
| `src/tests/AudioEngineHarness/ConvolverStateRoundTripTests.cpp` | 非有限入力の回帰テスト `checkTruePeakNonFiniteInput` を追加 |

- 正常入力ではスキャンのみで、コピーは行わない。
- バッファは `prepare` で確保し、`processBlock` では確保しない。
- 履歴のリセットは行わない（正しい履歴を保持したまま、汚染だけを防ぐ）。

## 3. 検証

| 項目 | 結果 |
|---|---|
| 非有限入力の回帰テスト | PASS（[#12] true-peak non-finite guard: PASS） |
| ハーネス全体 | exit 0、`all publish pipeline tests PASS` |
| `ctest -C Release` | 54 / 54 PASS（334.7 秒、フック有効ビルド） |
| CI（PR #167） | 15 件中 14 pass / 1 skipping（意図的失敗プローブは dispatch のみ） |

## 4. 影響評価

- **正常系**: 有限入力では追加のスキャン（2 チャンネル × ブロック長の `isfinite`）のみ。出力は従来と同一。
- **エラー時**: 非有限値は 0 として扱う。クリックを零に置換するが、検出器の状態を壊さない。
- **RT**: 追加の確保・ロックなし。バッファは `prepare` 時に確保済み。
- **仕様変更**: 検出値の意味（ピークホールド）は変えない。

## 5. 残存リスク・未検証

- 非有限値の「正しい」扱い（0 に置換）は運用上の選択であり、値の保持や直前値での補間ではない。
- 実機での非有限入力の発生頻度は未計測（本番経路では手前で防止済み）。

## 6. 状態

| 項目 | 状態 |
|---|---|
| 実装 | 完了 |
| テスト | 追加・PASS |
| PR | 作成済み（#167、マージ未実施） |

## 7. Owner 判断が必要な事項

1. 本 PR のマージ可否。
2. 非有限値を 0 に置換する現行方針を固定するか。
3. 本番経路の事前スクラブとの役割分担を文書化するか。
