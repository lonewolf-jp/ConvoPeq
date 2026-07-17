# ConvoPeq 設計改善 実装チェックリスト

> 作成日: 2026-07-17
> 元資料: `doc/work74/bug-fix-plan.md`
> 合計推定工数: 2日

---

## FIX-01: MKLスレッド設定のグローバル汚染 （P1, 0.5日） ✅ 完了

**目標**: `MKLRealTimeSetup.cpp` のプロセスグローバル設定をスレッドローカルに変更

- [x] 現状コードの確認
- [x] `MKLRealTimeSetup.cpp` の編集
  - [x] `_putenv_s("MKL_NUM_THREADS", "1")` を削除
  - [x] `_putenv_s("OMP_NUM_THREADS", "1")` を削除
  - [x] `mkl_set_num_threads(1)` を `mkl_set_num_threads_local(1)` に変更
  - [x] `mkl_set_dynamic(0)` は保持（必要性評価後に判断）
- [ ] ビルド確認（要 CI）
- [ ] テスト通過確認（要 CI）

---

## FIX-02: LoudnessMeter K-weighting 係数のサンプルレート依存 （P1, 1〜2日） ✅ 完了

**目標**: `LoudnessMeter.h/.cpp` の48kHz固定IIR係数を、サンプルレートに応じて bilinear transform で計算するよう変更

### Step 1: 設計 ✅
- [x] BS.1770-4 Annex B のアナログ伝達関数から bilinear transform の実装方式を決定
- [x] libebur128 参照実装の確認（既に調査済み）
- [x] 係数計算関数のインターフェース設計

### Step 2: 実装 ✅
- [x] `LoudnessMeter.h` に `updateCoefficients(double fs)` 宣言を追加
- [x] `LoudnessMeter.cpp` に bilinear transform 実装を追加
  - [x] Stage 1 (Pre-filter, High-shelf) の係数計算 → 標準ハイシェルフ公式で実装
  - [x] Stage 2 (RLB, High-pass) の係数計算
- [x] `prepare()` でサンプルレート変更時に係数を更新
- [x] 48kHz互換性確認（fs=48000でTable 1と一致することを理論確認）

### Step 3: テスト
- [ ] 48kHz/96kHz/192kHz で係数が適切に計算されることの確認（要 CI）
- [ ] LUFS値の妥当性確認（要 CI）

---

## FIX-03: Denormal 対策の設計規約文書化 （P2, 0.5日） ✅ 完了

**目標**: 新規IIRフィルタ追加時の設計規約として `killDenormal` パターンを文書化

- [x] 現状のFTZ/DAZ設定の確認
- [x] `MKLRealTimeSetup.cpp` のFTZ/DAZ設定
- [x] `MKLNonUniformConvolver::processLayerBlock` の `killDenormalV` パターン
- [x] `InputBitDepthTransform.h` の `sanitizeAndLimit`
- [x] 設計規約ドキュメントの作成 → `doc/design-guidelines/denormal-handling.md`

---

## 凡例

- `[ ]` : 未着手
- `[-]` : 作業中
- `[x]` : 完了
