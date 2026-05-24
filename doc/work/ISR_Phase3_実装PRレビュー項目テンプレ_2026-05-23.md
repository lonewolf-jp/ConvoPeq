# ISR Phase 3 実装PR レビュー項目テンプレ（コピペ運用用）

作成日: 2026-05-23
適用対象: Phase 3（RT規約適合: libm排除）

- Task 3-1: Convolver runtime から `std::floor` を排除

参照:

- `doc/work/ISR_バグ精査結果_2026-05-23.md`
- `doc/work/ISR_バグ精査結果_実装修正タスク分解_2026-05-23.md`
- `doc/work/ISR_Phase2_実装PRレビュー項目テンプレ_2026-05-23.md`

---

## 0) フェーズ前提ゲート（軽量G1〜G5）

- [ ] **G1 スコープ固定**: Phase 3（Task 3-1）以外を混在させない
- [ ] **G2 定量要件固定**: wrap/負値/frac近傍の境界評価条件を事前固定
- [ ] **G3 RT/ISR制約**: Audio Thread制約と ISR不変条件を破壊しない
- [ ] **G4 レビュー承認**: 実装者/レビューアでゲート充足を明示確認
- [ ] **G5 証跡添付**: Build/Scan/境界ケース検証結果をPR添付

未充足なら着手/マージしない。

## 1) PR本文テンプレ（作成者用）

以下をPR本文にそのまま貼り付けて使用。

```md
## 目的
- [ ] Phase 3（RT規約適合）を実装し、Audio Thread 経路の libm 呼び出しを排除する
- [ ] Convolver の遅延読み出し正規化を no-libm 実装へ置換し、既存音質挙動を維持する

## フェーズゲート情報（必須）
- [ ] G1: Phase 3スコープ固定（Task 3-1のみ）
- [ ] G2: 評価条件固定（wrap/負値/frac近傍）
- [ ] G3: RT/ISR制約遵守
- [ ] G4: レビュー承認方針を確認
- [ ] G5: 証跡添付方針を確認

## 変更範囲（ファイル）
- [ ] src/convolver/ConvolverProcessor.Runtime.cpp
- [ ] （必要に応じて）関連テスト/検証ドキュメント

## 実装内容
### Task 3-1
- [ ] `std::floor` を用いた正規化ロジックを no-libm 実装へ置換
- [ ] wrap / 負値 / frac近傍 の境界ケースで従来同等結果を保持
- [ ] RT経路に新規ブロッキング・動的確保・libm依存を持ち込まない

## ISR不変条件（Invariant）
- [ ] Audio Thread 経路に libm 呼び出しが存在しない
- [ ] 置換後も遅延読み出しの連続性（クリック抑制）を破壊しない
- [ ] 置換により snapshot整合性/reader運用を変更しない

## 検証結果
### Build
- [ ] Debug 成功
- [ ] Release 成功

### Static checks
- [ ] Strict Atomic Dot-Call Scan Pass

### Runtime checks
- [ ] Audio Thread 経路で `std::floor` が呼ばれないことを確認
- [ ] wrap/負値/frac近傍ケースで出力不連続が増加しない
- [ ] snapshot遷移・bypass遷移の既存挙動を悪化させない

## 影響範囲とリスク
- [ ] Convolver runtime の遅延補間/読み出し正規化
- [ ] RT処理負荷と音声連続性
- [ ] 境界値（delay/position）計算

## 非対象（このPRでは実施しない）
- [ ] Phase 4（mix=0最適化: 仕様FIX後）
```

---

## 2) レビューア用チェックリスト（そのままコメント可能）

```md
### ISR Phase 3 Review Checklist

#### A. G1〜G2（スコープ/要件）
- [ ] Audio Thread 実行経路から `std::floor` が除去されている
- [ ] 代替実装が no-libm である
- [ ] 代替実装に新規ブロッキング/動的確保がない

#### B. G3（RT/ISR制約）
- [ ] wrap ケースで読み出し位置が破綻しない
- [ ] 負値入力ケースで正規化が安定する
- [ ] frac近傍（0/1近傍）で不連続が悪化しない

#### C. G4（レビュー承認観点）
- [ ] snapshot遷移挙動に副作用がない
- [ ] bypass遷移挙動に副作用がない
- [ ] 音量・位相の不連続が増えていない

#### D. G5（検証証跡）
- [ ] G5: Build（Debug/Release）ログが添付されている
- [ ] G5: Strict scan 結果が添付されている
- [ ] G5: 境界ケース検証（wrap/負値/frac近傍）の結果が添付されている
```

---

## 3) 差し戻しコメントテンプレ（レビュー指摘用）

```md
差し戻し理由（ISR Phase 3基準未達）:

- [ ] G1不足: Phase 3スコープ外変更の混在
- [ ] G2不足: 境界評価条件または結果不足
- [ ] G3不足: RT/ISR制約違反の懸念
- [ ] G4不足: レビュー承認観点の確認不足
- [ ] G5不足: Build/Scan/境界ケース検証の証跡不足
- [ ] RT経路に libm 呼び出し（`std::floor` など）が残存
- [ ] no-libm 置換後に境界ケース（wrap/負値/frac近傍）で破綻
- [ ] 音声連続性（クリック/段差）が悪化
- [ ] 検証証跡（Build/Scan/境界ケース結果）が不足

対応依頼:
1. 該当箇所の修正
2. Debug/Release + Strict scan 再実行
3. 境界ケース検証結果を再添付
```

---

## 4) マージ判定テンプレ（承認時）

```md
## Merge Decision (ISR Phase 3)

判定: ✅ Approve

根拠:
- [ ] G1〜G5 がすべて満たされている
- [ ] RT経路から libm 呼び出しが除去されていることを確認
- [ ] 境界ケース（wrap/負値/frac近傍）の安定性を確認
- [ ] 音声連続性の回帰がないことを確認
- [ ] Build/Scan/境界ケース検証の証跡を確認

備考:
- Phase 4（性能最適化）は仕様FIX後に別PRで実施
```

---

## 5) 運用メモ

- 前提として Phase 2 マージ済みを推奨。
- 1PR 1目的（Phase 3のみ）を厳守。
- mix=0最適化（Phase 4）は仕様確定前に混在させない。
