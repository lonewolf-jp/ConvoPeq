# ConvoPeq: 誤検知除外版 Issue ドラフト（統合版）

作成日: 2026-05-24
統合元:

- `review16_filtered_issue_drafts_2026-05-24.md`
- `review7_filtered_issue_drafts_2026-05-24.md`

---

## 統合サマリ

- 起票推奨: **9件**
  - 16項目レビュー由来: 5件
  - 7項目レビュー由来: 4件
- 起票見送り（誤検知/根拠不足）:
  - 16項目レビュー: 11件
  - 7項目レビュー: 3件

---

## 起票推奨バックログ（統合）

### U1 (P0): Audio Thread 経路の `PsychoacousticDither::killDenormal` から libm依存を除去

- 由来: 7項目レビュー #1
- 背景: `src/PsychoacousticDither.h` の `killDenormal` が `std::fabs` を使用。
- 提案: bit-cast ベースの非libm実装へ置換し、既存数値ポリシーに統一。
- 受け入れ条件:
  - [ ] `PsychoacousticDither.h` に `std::fabs` が残らない
  - [ ] Debug/Release ビルド通過
  - [ ] 音質回帰なし（null test / smoke test）
- ラベル案: `P0`, `rt-safety`, `dsp`, `compliance`

### U2 (P1): `DeviceSettings::loadSettings` の BulkRestore を RAII 化

- 由来: 16項目レビュー Issue 1
- 背景: begin/end の手動対管理が将来変更時の取りこぼしリスク。
- 提案: スコープガード型を導入して `endBulkParameterRestore(true)` を自動化。
- 受け入れ条件:
  - [ ] 手動 begin/end 管理を解消
  - [ ] 既存復元フローの挙動維持
  - [ ] Debug/Release ビルド通過
- ラベル案: `P1`, `refactor`, `safety`, `settings`

### U3 (P1): `StereoConvolver` の破棄契約を型で強制

- 由来: 16項目レビュー Issue 2
- 背景: `destroyStereoConvolver` 前提設計を API で表現しきれていない。
- 提案: 所有ハンドルを明確化し、誤った通常 delete 経路を封じる。
- 受け入れ条件:
  - [ ] 破棄経路が1系統に統一
  - [ ] rebuild/cleanup/shutdown の挙動維持
  - [ ] 二重解放・リーク関連 assertion の悪化なし
- ラベル案: `P1`, `refactor`, `memory`, `convolver`

### U4 (P1): `diagLog` の Release 出力ポリシー統一

- 由来: 7項目レビュー #5
- 背景: `AudioEngine.Commit.cpp` の `diagLog` が `Logger::writeToLog` を常時実行。
- 提案: `JUCE_DEBUG` / `CONVO_CI_BUILD` などで出力をゲートし、Releaseの過剰I/Oを抑制。
- 受け入れ条件:
  - [ ] Release通常運用で高頻度診断ログが抑制
  - [ ] Debug/CI で必要診断は維持
  - [ ] ビルドと基本動作に回帰なし
- ラベル案: `P1`, `logging`, `performance`, `maintenance`

### U5 (P2): `CustomInputOversampler` の `corruptionDetected` 判定運用を段階化

- 由来: 16項目レビュー Issue 3
- 背景: 現行は安全側設計だが、運用上の過敏反応を可観測化したい。
- 提案: 発火条件の計測を追加し、回復可能ケースと回復不能ケースを分離。
- 受け入れ条件:
  - [ ] 発火条件の診断可能化
  - [ ] 回復戦略の定義
  - [ ] 音質/安定性の退行なし
- ラベル案: `P2`, `investigation`, `dsp`, `safety`

### U6 (P2): `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_*` 常時ON前提の分岐整理

- 由来: 16項目レビュー Issue 4
- 背景: 実質常時有効な条件分岐が読解性を低下。
- 提案: 小分けPRで不要分岐を段階整理。
- 受け入れ条件:
  - [ ] ビルド/動作維持
  - [ ] 分割ファイル構成との整合
  - [ ] レビュー可能な小さな差分
- ラベル案: `P2`, `refactor`, `build-system`, `readability`

### U7 (P2): `CmaEsOptimizerDynamic` の allocator 方針を規約整合で明文化

- 由来: 7項目レビュー #3
- 背景: Worker側 `std::vector` 多用は即時バグではないが、規約解釈と齟齬が残る。
- 提案: 「許容/非許容」の合意を先に取り、必要なら段階置換。
- 受け入れ条件:
  - [ ] 規約解釈の合意
  - [ ] 置換範囲・非対象範囲の文書化
  - [ ] 変更時の性能退行なし
- ラベル案: `P2`, `tech-debt`, `allocator`, `compliance`

### U8 (P2): `SafeStateSwapper` の `tryReclaim` 利用契約（Single Consumer前提）を明文化

- 由来: 7項目レビュー #4
- 背景: 現設計は成立しているが、将来の呼び出し拡張で誤用余地がある。
- 提案: 呼び出しスレッド前提をコメント/APIで固定し、debug検知を強化。
- 受け入れ条件:
  - [ ] 利用契約がヘッダと実装で一貫
  - [ ] 誤用を検知可能
  - [ ] RT安全性維持
- ラベル案: `P2`, `thread-safety`, `documentation`, `rcu`

### U9 (P3): `AudioEngine.h` の段階的分割計画を策定

- 由来: 16項目レビュー Issue 5
- 背景: 巨大ヘッダ（約141KB）による保守性・ビルド効率負債。
- 提案: まず分割設計Issueを作成し、実装Issueを分離。
- 受け入れ条件:
  - [ ] 分割対象と順序の計画書
  - [ ] ABI/初期化順序/RT制約のチェック観点定義
  - [ ] 実装フェーズ子Issue作成
- ラベル案: `P3`, `architecture`, `refactor`, `tech-debt`

---

## 起票見送り（誤検知/根拠不足）

### 16項目レビュー由来（11件）

- #1 `ScopedAlignedPtr` 非トリビアル配列 UB（`static_assert` 防御あり）
- #2 `DSPCore::~DSPCore` 二重解放断定（根拠不足）
- #3 `activeDSP/fadingOutDSP` retire漏れ（解放経路あり）
- #4 `accumBuf` 未初期化パディング（全域ゼロ初期化）
- #5 `LockFreeRingBuffer::pop` フェンス不足断定（SPSC前提で成立）
- #6 `applyAllpassToIR` fftSizeメモリ破壊（補正処理あり）
- #7 AVX unaligned load/store 指摘（`loadu/storeu` 使用済み）
- #9 `ic1eq/ic2eq` clamp 指摘（実際は output clamp）
- #11 `adaptiveCoeffSet` ポインタ寿命切れ（設計上アドレス安定）
- #12 Nyquist符号判定バグ断定（断定根拠不足）
- #16 `reusableGainInterleaved` 毎回確保（再利用実装あり）

### 7項目レビュー由来（3件）

- #2 `ringWrite` overflow callback 漏れ（現実装に callback 呼び出しあり）
- #6 `EQProcessor` 除算ゼロガード不足（`numSamples<=0` 早期 return あり）
- #7 `OutputFilter` prefetch 境界不備（境界ガード実装済み）

---

## GitHub 起票用テンプレ（統合版）

```text
## 背景

## 問題

## 提案

## 受け入れ条件
- [ ]
- [ ]

## 影響範囲

## 備考
```
