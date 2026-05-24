# ISR Phase 0 実装PR レビュー項目テンプレ（コピペ運用用）

作成日: 2026-05-23
適用対象: Phase 0（Snapshot整合性の一本化）

- Task 0-1: DSPCore内 snapshot 再取得廃止
- Task 0-2: Snapshot処理経路の寿命検証レイヤ統一

参照:

- `doc/work/ISR_バグ精査結果_2026-05-23.md`
- `doc/work/ISR_バグ精査結果_実装修正タスク分解_2026-05-23.md`

---

## 0) フェーズ前提ゲート（軽量G1〜G5）

- [ ] **G1 スコープ固定**: Phase 0（Task 0-1/0-2）以外を混在させない
- [ ] **G2 定量要件固定**: 1ブロック1snapshot・Reader Index運用・hash一致の評価条件を事前固定
- [ ] **G3 RT/ISR制約**: Audio Thread制約と ISR不変条件を破壊しない
- [ ] **G4 レビュー承認**: 実装者/レビューアでゲート充足を明示確認
- [ ] **G5 証跡添付**: Build/Scan/hash一致テストの結果をPR添付

未充足なら着手/マージしない。

## 1) PR本文テンプレ（作成者用）

以下をPR本文にそのまま貼り付けて使用。

```md
## 目的
- [ ] Phase 0（Snapshot整合性）を実装し、1ブロック1snapshotを保証する
- [ ] Audio Thread の reader index 運用を `kAudioEpochReaderIndex` に統一する

## フェーズゲート情報（必須）
- [ ] G1: Phase 0スコープ固定（Task 0-1/0-2のみ）
- [ ] G2: 評価条件固定（hash一致・Reader Index・1ブロック1snapshot）
- [ ] G3: RT/ISR制約遵守
- [ ] G4: レビュー承認方針を確認
- [ ] G5: 証跡添付方針を確認

## 変更範囲（ファイル）
- [ ] src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp
- [ ] src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
- [ ] src/audioengine/AudioEngine.Processing.Snapshot.cpp
- [ ] src/audioengine/AudioEngine.Processing.AudioBlock.cpp
- [ ] src/audioengine/AudioEngine.h

## 実装内容
### Task 0-1
- [ ] DSPCore::process()/processDouble() での `observeCurrentRuntime()` 再取得を廃止
- [ ] EQ参照情報を ProcessingState 経由で注入
- [ ] Audio Thread 経路で `kControlEpochReaderIndex` 非使用を保証

### Task 0-2
- [ ] snapshot経路で resolverベース参照へ統一
- [ ] old/new同一DSP実体フォールバックを禁止
- [ ] fading未解決時の安全フォールバック（dry-old or fade無効）を実装

## ISR不変条件（Invariant）
- [ ] 1ブロック内で snapshotFrom/snapshotTo を固定し再取得しない
- [ ] snapshot fade 時は `snapshotTo != nullptr` を必須化
- [ ] `snapshotFrom == nullptr` 時は明示フォールバック
- [ ] old/new同一DSPを禁止（assert + fallback）

## 検証結果
### Build
- [ ] Debug 成功
- [ ] Release 成功

### Static checks
- [ ] Strict Atomic Dot-Call Scan Pass

### Runtime checks
- [ ] 世代混在検出テスト: hash一致（ProcessingState時点 vs EQ参照時点）
- [ ] 高頻度パラメータ更新下で同一ブロック内 hash不一致なし

## 影響範囲とリスク
- [ ] Audio callback経路（float/double）
- [ ] snapshot fade経路
- [ ] EQ係数参照経路

## 非対象（このPRでは実施しない）
- [ ] Phase 1以降（double対称化詳細、libm排除、性能最適化）
```

---

## 2) レビューア用チェックリスト（そのままコメント可能）

```md
### ISR Phase 0 Review Checklist

#### A. G1〜G2（スコープ/要件）
- [ ] 1ブロック内 snapshot取得が1回に統一されている
- [ ] DSPCore内で runtime snapshot 再取得が残っていない
- [ ] old/new同一DSP実体が成立しない

#### B. G3（RT/ISR制約）
- [ ] Audio Thread から `kControlEpochReaderIndex` を使用していない
- [ ] Audio Thread では `kAudioEpochReaderIndex` のみ使用

#### C. G4（レビュー承認観点）
- [ ] snapshot経路で resolver参照に統一されている
- [ ] fading未解決時のフォールバックが明示されている
- [ ] `snapshotFrom == nullptr` 時の挙動が安全

#### D. G5（検証証跡）
- [ ] G5: Build（Debug/Release）ログが添付されている
- [ ] G5: Strict scan 結果が添付されている
- [ ] G5: hash一致テスト結果（世代混在なし）が添付されている
```

---

## 3) 差し戻しコメントテンプレ（レビュー指摘用）

```md
差し戻し理由（ISR Phase 0基準未達）:

- [ ] G1不足: Phase 0スコープ外変更の混在
- [ ] G2不足: hash一致/Reader Index要件の定義または結果不足
- [ ] G3不足: RT/ISR制約違反の懸念
- [ ] G4不足: レビュー承認観点の確認不足
- [ ] G5不足: Build/Scan/hash一致の証跡不足
- [ ] 同一ブロック内で snapshot 再取得が残存
- [ ] Audio Thread 経路で `kControlEpochReaderIndex` 使用が残存
- [ ] old/new同一DSP実体を許容するフォールバックが残存
- [ ] snapshotFrom/snapshotTo の null条件が不十分
- [ ] hash一致の検証証跡が不足

対応依頼:
1. 該当箇所の修正
2. Debug/Release + Strict scan 再実行
3. 世代混在検出テスト結果の再添付
```

---

## 4) マージ判定テンプレ（承認時）

```md
## Merge Decision (ISR Phase 0)

判定: ✅ Approve

根拠:
- [ ] G1〜G5 がすべて満たされている
- [ ] Snapshot整合性（1ブロック1snapshot）を確認
- [ ] Reader index運用（Audio=0, Control非使用）を確認
- [ ] old/new同一DSP禁止と安全フォールバックを確認
- [ ] Build/Scan/世代混在検証の証跡を確認

備考:
- Phase 1 以降は別PRで実施（スコープ分離）
```

---

## 5) 運用メモ

- 1PR 1目的（Phase 0のみ）を厳守。
- 変更がPhase 1以上へ跨る場合はPR分割。
- 重大バグ回避より先に ISR不変条件の破壊を許容しない。
