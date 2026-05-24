# ISR Phase 2 実装PR レビュー項目テンプレ（コピペ運用用）

作成日: 2026-05-23
適用対象: Phase 2（異常系防御: クラッシュ/状態機械停止）

- Task 2-1: 0ch/null防御を I/O 出力経路へ統一適用
- Task 2-2: EQバイパス遷移進行を dryCopy確保可否から分離
- Task 2-3: `numSamples <= 0` 防御分岐の clear 呼び出し安全化

参照:

- `doc/work/ISR_バグ精査結果_2026-05-23.md`
- `doc/work/ISR_バグ精査結果_実装修正タスク分解_2026-05-23.md`
- `doc/work/ISR_Phase1_実装PRレビュー項目テンプレ_2026-05-23.md`

---

## 0) フェーズ前提ゲート（軽量G1〜G5）

- [ ] **G1 スコープ固定**: Phase 2（Task 2-1〜2-3）以外を混在させない
- [ ] **G2 定量要件固定**: 0ch/null/非正samples/dryCopy未確保ケースの評価条件を事前固定
- [ ] **G3 RT/ISR制約**: Audio Thread制約と ISR不変条件を破壊しない
- [ ] **G4 レビュー承認**: 実装者/レビューアでゲート充足を明示確認
- [ ] **G5 証跡添付**: Build/Scan/異常系実行結果をPR添付

未充足なら着手/マージしない。

## 1) PR本文テンプレ（作成者用）

以下をPR本文にそのまま貼り付けて使用。

```md
## 目的
- [ ] Phase 2（異常系防御）を実装し、0ch/null/非正サンプル数でのクラッシュ経路を排除する
- [ ] EQバイパス遷移が dryCopy 未確保時に停止しないことを保証する

## フェーズゲート情報（必須）
- [ ] G1: Phase 2スコープ固定（Task 2-1〜2-3のみ）
- [ ] G2: 評価条件固定（0ch/null/非正samples/dryCopy未確保）
- [ ] G3: RT/ISR制約遵守
- [ ] G4: レビュー承認方針を確認
- [ ] G5: 証跡添付方針を確認

## 変更範囲（ファイル）
- [ ] src/audioengine/AudioEngine.Processing.DSPCoreIO.cpp
- [ ] src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
- [ ] src/eqprocessor/EQProcessor.Processing.cpp

## 実装内容
### Task 2-1
- [ ] `numChannels <= 0` の早期returnを明示
- [ ] `dataL == nullptr` 前提で到達するループ/コピー経路を排除
- [ ] `dc.output*` 呼び出しを null-safe 条件に変更

### Task 2-2
- [ ] EQバイパス遷移の ramp進行を dryCopy有無と独立化
- [ ] `rtBypassedShadow` 更新が dryCopy未確保でも前進することを保証
- [ ] dryCopy不足時フォールバック挙動（wet-only または即時遷移）を明示

### Task 2-3
- [ ] `numSamples < 1` は clearせず即return
- [ ] clear 呼び出しは `numSamples > 0` が保証される経路に限定

## ISR不変条件（Invariant）
- [ ] 異常入力（0ch/null/非正samples）でも Audio Thread で未定義動作に入らない
- [ ] 異常系で状態機械（バイパス遷移）が停止しない
- [ ] 防御分岐の追加により通常系の処理順序を破壊しない

## 検証結果
### Build
- [ ] Debug 成功
- [ ] Release 成功

### Static checks
- [ ] Strict Atomic Dot-Call Scan Pass

### Runtime checks
- [ ] 0ch入力時に null逆参照が発生しない
- [ ] `numSamples <= 0` 入力時に不正clear経路が発生しない
- [ ] dryCopy未確保ケースでバイパス遷移が停止しない

## 影響範囲とリスク
- [ ] I/O 出力処理（float/double）
- [ ] EQバイパス遷移状態機械
- [ ] 異常系の fail-safe 分岐

## 非対象（このPRでは実施しない）
- [ ] Phase 3以降（Convolver libm排除、性能最適化）
```

---

## 2) レビューア用チェックリスト（そのままコメント可能）

```md
### ISR Phase 2 Review Checklist

#### A. G1〜G2（スコープ/要件）
- [ ] `numChannels <= 0` で早期returnする
- [ ] `dataL == nullptr` 前提アクセスが残っていない
- [ ] `dc.output*` 呼び出しが null-safe になっている

#### B. G3（RT/ISR制約）
- [ ] dryCopy未確保時も ramp進行が止まらない
- [ ] `rtBypassedShadow` 更新が dryCopy有無に依存していない
- [ ] フォールバック挙動（wet-only/即時遷移）がコード上で明示されている

#### C. G4（レビュー承認観点）
- [ ] `numSamples < 1` 時に clear呼び出しを行わずreturnする
- [ ] clear 呼び出し箇所は `numSamples > 0` 前提が担保されている

#### D. G5（検証証跡）
- [ ] G5: Build（Debug/Release）ログが添付されている
- [ ] G5: Strict scan 結果が添付されている
- [ ] G5: 0ch / 非正samples / dryCopy未確保の実行結果が添付されている
```

---

## 3) 差し戻しコメントテンプレ（レビュー指摘用）

```md
差し戻し理由（ISR Phase 2基準未達）:

- [ ] G1不足: Phase 2スコープ外変更の混在
- [ ] G2不足: 異常系評価条件または結果不足
- [ ] G3不足: RT/ISR制約違反の懸念
- [ ] G4不足: レビュー承認観点の確認不足
- [ ] G5不足: Build/Scan/異常系検証の証跡不足
- [ ] 0ch/null 経路で防御漏れが残存
- [ ] 非正 `numSamples` で不正clear経路が残存
- [ ] EQバイパス遷移が dryCopy未確保時に停止する
- [ ] 異常系検証証跡（Build/Scan/実行結果）が不足

対応依頼:
1. 該当箇所の修正
2. Debug/Release + Strict scan 再実行
3. 0ch / 非正samples / dryCopy未確保の確認結果を再添付
```

---

## 4) マージ判定テンプレ（承認時）

```md
## Merge Decision (ISR Phase 2)

判定: ✅ Approve

根拠:
- [ ] G1〜G5 がすべて満たされている
- [ ] 0ch/null 防御の網羅を確認
- [ ] 非正samples の不正clear排除を確認
- [ ] EQバイパス遷移の前進保証を確認
- [ ] Build/Scan/異常系実行結果の証跡を確認

備考:
- Phase 3 以降は別PRで実施（スコープ分離）
```

---

## 5) 運用メモ

- 前提として Phase 1 マージ済みを推奨。
- 1PR 1目的（Phase 2のみ）を厳守。
- libm排除（Phase 3）や性能最適化（Phase 4）は本PRへ混在させない。
