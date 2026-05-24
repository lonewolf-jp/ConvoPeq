# ISR Phase 1 実装PR レビュー項目テンプレ（コピペ運用用）

作成日: 2026-05-23
適用対象: Phase 1（double経路の安全対称化）

- Task 1-1: `processBlockDouble` への runtime安全ガード導入
- Task 1-2: double経路への snapshot fade 実処理導入
- Task 1-3: snapshot専用経路の sample-rate 不整合ガード追加
- Task 1-4: OS>1 doubleバイパスの Dry/Wet 合成統一

参照:

- `doc/work/ISR_バグ精査結果_2026-05-23.md`
- `doc/work/ISR_バグ精査結果_実装修正タスク分解_2026-05-23.md`
- `doc/work/ISR_Phase0_実装PRレビュー項目テンプレ_2026-05-23.md`

---

## 0) フェーズ前提ゲート（軽量G1〜G5）

- [ ] **G1 スコープ固定**: Phase 1（Task 1-1〜1-4）以外を混在させない
- [ ] **G2 定量要件固定**: shutdown fail-safe・rate mismatch防御・OS対称性評価条件を事前固定
- [ ] **G3 RT/ISR制約**: Audio Thread制約と ISR不変条件を破壊しない
- [ ] **G4 レビュー承認**: 実装者/レビューアでゲート充足を明示確認
- [ ] **G5 証跡添付**: Build/Scan/遷移比較結果をPR添付

未充足なら着手/マージしない。

## 1) PR本文テンプレ（作成者用）

以下をPR本文にそのまま貼り付けて使用。

```md
## 目的
- [ ] Phase 1（double経路の安全対称化）を実装し、float経路との設計非対称を解消する
- [ ] snapshot遷移品質・バイパス遷移品質をdouble経路で等価化する

## フェーズゲート情報（必須）
- [ ] G1: Phase 1スコープ固定（Task 1-1〜1-4のみ）
- [ ] G2: 評価条件固定（shutdown/rate mismatch/OS対称）
- [ ] G3: RT/ISR制約遵守
- [ ] G4: レビュー承認方針を確認
- [ ] G5: 証跡添付方針を確認

## 変更範囲（ファイル）
- [ ] src/audioengine/AudioEngine.Processing.BlockDouble.cpp
- [ ] src/audioengine/AudioEngine.Processing.Snapshot.cpp
- [ ] src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp
- [ ] （必要に応じて）src/audioengine/AudioEngine.h

## 実装内容
### Task 1-1
- [ ] `isShutdownInProgress()` 早期return + clear を追加
- [ ] lifecycle/firewall scope を float経路相当で導入
- [ ] callback active count / RT context mark を対称化

### Task 1-2
- [ ] `updateAudioThreadSnapshotFade()` 結果を実際の処理に反映
- [ ] double経路に old/new snapshot 合成経路を導入
- [ ] fade stateだけ進む未反映状態を解消

### Task 1-3
- [ ] `processWithSnapshot` に sample-rate 一致ガードを追加
- [ ] 不整合時の clear + return を実装

### Task 1-4
- [ ] OS>1分岐（processDown後）にも Dry/Wet 合成を適用
- [ ] OS=1分岐と同等のバイパス遷移特性へ統一

## ISR不変条件（Invariant）
- [ ] snapshot fade 時は `snapshotTo != nullptr` を必須化
- [ ] `snapshotFrom == nullptr` のときは明示フォールバック
- [ ] double経路でも snapshot遷移が音声へ必ず反映される
- [ ] callback入口の安全ガード（shutdown/lifecycle/firewall）は float/double で機能対称

## 検証結果
### Build
- [ ] Debug 成功
- [ ] Release 成功

### Static checks
- [ ] Strict Atomic Dot-Call Scan Pass

### Runtime checks
- [ ] shutdown遷移中 double callback の安全無音化を確認
- [ ] snapshot専用経路の rate mismatch で clear + return を確認
- [ ] OS on/off でバイパスフェード挙動の一貫性を確認

## 影響範囲とリスク
- [ ] Audio callback double経路
- [ ] snapshot fade経路
- [ ] oversampling + bypass 合成経路

## 非対象（このPRでは実施しない）
- [ ] Phase 2以降（0ch/null防御の横展開、EQ状態機械停止対策、libm排除、性能最適化）
```

---

## 2) レビューア用チェックリスト（そのままコメント可能）

```md
### ISR Phase 1 Review Checklist

#### A. G1〜G2（スコープ/要件）
- [ ] double経路に shutdownガードがある
- [ ] double経路に lifecycle enter/leave がある
- [ ] double経路に firewall mark/enter/leave がある
- [ ] float経路と同等の fail-safe（clear + return）になっている

#### B. G3（RT/ISR制約）
- [ ] double経路で snapshotFrom/snapshotTo/snapshotAlpha が実際に使用される
- [ ] fade stateのみ進行して音声未反映になる経路がない
- [ ] snapshot専用経路に sample-rate 不整合ガードがある

#### C. G4（レビュー承認観点）
- [ ] OS>1分岐で Dry/Wet 合成が適用される
- [ ] OS=1 と OS>1 で遷移音量特性が一致する

#### D. G5（検証証跡）
- [ ] G5: Build（Debug/Release）ログが添付されている
- [ ] G5: Strict scan 結果が添付されている
- [ ] G5: snapshot遷移と bypass遷移の比較結果が添付されている
```

---

## 3) 差し戻しコメントテンプレ（レビュー指摘用）

```md
差し戻し理由（ISR Phase 1基準未達）:

- [ ] G1不足: Phase 1スコープ外変更の混在
- [ ] G2不足: shutdown/rate mismatch/OS対称の要件または結果不足
- [ ] G3不足: RT/ISR制約違反の懸念
- [ ] G4不足: レビュー承認観点の確認不足
- [ ] G5不足: Build/Scan/遷移比較の証跡不足
- [ ] double callback 入口ガードが float 相当まで到達していない
- [ ] double経路で snapshot fade 状態が進むのみで音声反映が不十分
- [ ] snapshot専用経路の sample-rate ガードが不足
- [ ] OS>1 bypass で Dry/Wet 合成が未適用
- [ ] 検証証跡（Build/Scan/遷移比較）が不足

対応依頼:
1. 該当箇所の修正
2. Debug/Release + Strict scan 再実行
3. shutdown/snapshot/bypass の確認結果を再添付
```

---

## 4) マージ判定テンプレ（承認時）

```md
## Merge Decision (ISR Phase 1)

判定: ✅ Approve

根拠:
- [ ] G1〜G5 がすべて満たされている
- [ ] double callback 入口ガードの対称化を確認
- [ ] double snapshot fade の実反映を確認
- [ ] snapshot専用経路の rate mismatch 防御を確認
- [ ] OS>1 bypass の Dry/Wet 合成統一を確認
- [ ] Build/Scan/遷移比較の証跡を確認

備考:
- Phase 2 以降は別PRで実施（スコープ分離）
```

---

## 5) 運用メモ

- 前提として Phase 0 マージ済みを推奨。
- 1PR 1目的（Phase 1のみ）を厳守。
- 異常系防御の横展開（Phase 2）は本PRへ混在させない。
