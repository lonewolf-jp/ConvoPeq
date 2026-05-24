# ISR Phase 4 実装PR レビュー項目テンプレ（コピペ運用用）

作成日: 2026-05-23
適用対象: Phase 4（仕様FIX前提: mix=0最適化）

- Task 4-1: `needsConvolution` 活用方針を仕様化し実装

参照:

- `doc/work/ISR_バグ精査結果_2026-05-23.md`
- `doc/work/ISR_バグ精査結果_実装修正タスク分解_2026-05-23.md`
- `doc/work/ISR_Phase3_実装PRレビュー項目テンプレ_2026-05-23.md`

---

## 0) 仕様FIX前提ゲート（PR着手前に必須）

以下が未充足の場合、Phase 4 PRは作成しない（未承認なら未着手）。

- [ ] **G1 方針固定（単一選択）**: `完全skip` を確定し、仕様IDを明記
- [ ] **G2 定量要件固定**: CPU改善・P95・クリック・RMS差・追加遅延の合格閾値を事前確定
- [ ] **G3 RT安全制約**: lock/blocking/dynamic alloc/libm再導入なし
- [ ] **G4 3者承認**: Tech Lead / QA / Product(Spec) Owner の承認（日時・コメント・仕様ID付き）
- [ ] **G5 証跡添付**: Build/Scan/CPU比較/連続性評価/承認情報をPR添付

---

## 1) PR本文テンプレ（作成者用）

以下をPR本文にそのまま貼り付けて使用。

```md
## 目的
- [ ] Phase 4（mix=0最適化）を仕様FIXに基づいて実装する
- [ ] Dry-only時の不要な畳み込み実行を削減し、RT余裕を改善する

## 仕様FIX情報（必須）
- [ ] G1 方針: （完全skip）
- [ ] G1 仕様ID:
- [ ] G2 定量要件（CPU/P95/クリック/RMS差/追加遅延）:
- [ ] G4 承認者（Tech Lead / QA / Product）:
- [ ] G4 承認日時:
- [ ] G4 承認コメント:

## 変更範囲（ファイル）
- [ ] src/convolver/ConvolverProcessor.Runtime.cpp
- [ ] （必要に応じて）関連設計/検証ドキュメント

## 実装内容
### Task 4-1
- [ ] 仕様FIX方針に従い `conv->process(...)` 呼び出し条件を反映
- [ ] `needsConvolution` 判定と実行経路の整合を担保
- [ ] Dry-only時の不要処理削減と再mix時の連続性を両立

## ISR不変条件（Invariant）
- [ ] 最適化後も Audio Thread の安全制約（非ブロッキング/非動的確保）を維持
- [ ] Snapshot整合性・reader運用（Phase 0〜3）を破壊しない
- [ ] 仕様FIXで定義した連続性要件を満たす

## 検証結果
### Build
- [ ] Debug 成功
- [ ] Release 成功

### Static checks
- [ ] Strict Atomic Dot-Call Scan Pass

### Runtime checks
- [ ] G2/G5: Dry-only時に不要畳み込みが抑制される
- [ ] G2/G5: 再mix時のクリック/段差が仕様許容範囲内
- [ ] G2/G5: snapshot遷移・bypass遷移の既存挙動を悪化させない

### Performance checks
- [ ] G2/G5: 最適化前後のCPU負荷比較結果を添付
- [ ] G2/G5: 測定条件（SR/Buffer/OS/入力素材）を明記

## 影響範囲とリスク
- [ ] Convolver runtime の処理分岐
- [ ] mix遷移の連続性
- [ ] RT余裕と音質トレードオフ

## 非対象（このPRでは実施しない）
- [ ] 仕様FIXの再定義（本PRは実装のみ）
```

---

## 2) レビューア用チェックリスト（そのままコメント可能）

```md
### ISR Phase 4 Review Checklist

#### A. G1〜G2 の充足
- [ ] G1: 方針（完全skip）と仕様IDが明示されている
- [ ] G2: 合格閾値（CPU/P95/クリック/RMS差/追加遅延）が明示されている
- [ ] G2: 連続性要件が定量的に評価されている

#### B. G3〜G4 の充足
- [ ] G3: RT安全制約（非ブロッキング/非動的確保/libm再導入なし）を維持
- [ ] G4: 3者承認（Tech Lead/QA/Product）が揃い、日時・コメント・仕様IDが記録されている

#### C. 実装整合
- [ ] `needsConvolution` 判定と `conv->process(...)` 実行条件が整合
- [ ] Dry-only時の不要処理削減が確認できる
- [ ] 再mix時の挙動が仕様どおり

#### C-2. 再mix連続性のレビュー観点
- [ ] `mix=0` 区間は dry path のみで通過し、wet 側の不要処理が走っていない
- [ ] `mix=0 -> mix>0` への復帰で、`activeMixSmoother` と equal-power gain が連続している
- [ ] クリック監視は再mix直前/直後の短区間を比較し、単発スパイクの有無を確認している
- [ ] 20ms 窓 RMS差は再mix点を跨いで評価し、許容範囲内である
- [ ] bypass 遷移と snapshot 遷移に副作用がないことを併記している

#### D. ISR回帰防止
- [ ] Phase 0〜3で確立した不変条件を壊していない
- [ ] Audio Thread 制約（非ブロッキング/非動的確保/RT安全）を維持
- [ ] snapshot遷移/bypass遷移への副作用がない

#### E. G5 証跡
- [ ] G5: Build（Debug/Release）ログが添付されている
- [ ] G5: Strict scan 結果が添付されている
- [ ] G5: 性能比較結果（条件付き）が添付されている
- [ ] G5: 音質/連続性検証結果が添付されている
- [ ] G5: 承認情報（承認者・日時・コメント・仕様ID）が添付されている

#### E-2. G5 記入先
- [ ] `doc/work/ISR_Phase4_G5_証跡テンプレ_2026-05-23.md` に各証跡を記入している
- [ ] テンプレ上の空欄と PR 添付物の内容が一致している
```

---

## 3) 差し戻しコメントテンプレ（レビュー指摘用）

```md
差し戻し理由（ISR Phase 4基準未達）:

- [ ] G1不足: 方針固定/仕様IDが不足
- [ ] G2不足: 定量要件（CPU/P95/クリック/RMS差/追加遅延）が不足または未達
- [ ] G3不足: RT安全制約違反の懸念がある
- [ ] G4不足: 3者承認情報（承認者/日時/コメント/仕様ID）が不足
- [ ] G5不足: 検証証跡（Build/Scan/性能/音質/承認情報）が不足
- [ ] ISR不変条件（Phase 0〜3）への回帰がある

対応依頼:
1. G1〜G5 の不足項目を補完し、実装を仕様へ整合
2. Debug/Release + Strict scan 再実行
3. 性能比較と連続性評価結果を再添付
```

---

## 4) マージ判定テンプレ（承認時）

```md
## Merge Decision (ISR Phase 4)

判定: ✅ Approve

根拠:
- [ ] G1〜G5 がすべて満たされている
- [ ] Dry-only最適化と再mix連続性が両立している
- [ ] ISR不変条件（Phase 0〜3）への回帰がない
- [ ] Build/Scan/性能/音質検証の証跡を確認した

備考:
- 仕様FIX変更が必要な場合は実装PRと分離して再審議する
```

---

## 5) 運用メモ

- 前提として Phase 3 マージ済みを推奨。
- 1PR 1目的（Phase 4のみ）を厳守。
- 仕様FIXの再交渉は別PR/別ドキュメントで扱い、実装PRへ混在させない。
