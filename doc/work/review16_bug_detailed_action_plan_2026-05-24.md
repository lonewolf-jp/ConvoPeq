# review16 統合Issue（U1〜U9）詳細改修計画書

作成日: 2026-05-24
対象: `doc/work/review16_filtered_issue_drafts_2026-05-24.md` の起票推奨バックログ（U1〜U9）

---

## 1. 目的

- 統合Issueの9件を、**実装可能な順序**・**依存関係**・**検証条件**まで分解する。
- Audio Thread安全性と既存挙動維持を最優先し、段階的に技術負債を削減する。
- 変更粒度を小さく保ち、レビュー容易性とロールバック容易性を確保する。

---

## 2. 前提・制約

- 編集禁止:
  - `JUCE/`
  - `r8brain-free-src/`
- Audio Thread では以下を禁止（既存規約準拠）:
  - ブロッキング、ロック待ち、I/O、動的確保/解放、例外、libm重依存処理
- oneMKL 利用箇所は既存方針（64-byte alignment、非RT確保）を順守する。
- 変更後は少なくとも Debug/Release ビルドを通す。

---

## 3. 優先順位と依存関係

### 優先度

- **P0**: U1
- **P1**: U2, U3, U4
- **P2**: U5, U6, U7, U8
- **P3**: U9

### 依存関係（実施順）

1. U1（RT即時是正）
2. U4（ログI/O抑制）
3. U2（設定復元の安全化）
4. U3（所有権契約強化）
5. U8（利用契約明文化）
6. U5（検知運用の段階化）
7. U6（条件分岐整理）
8. U7（allocator方針の文書合意）
9. U9（分割設計計画）

---

## 4. Issue別 詳細改修計画

<!-- markdownlint-disable MD024 -->

## U1 (P0): `PsychoacousticDither::killDenormal` の libm依存除去

### 対象

- `src/PsychoacousticDither.h`

### 実装方針

- `std::fabs` 依存を廃止し、bit-level判定（符号ビット除去 + 閾値比較）へ置換する。
- denormal flush条件を定数化し、処理分岐を最小化する。
- 既存音質に影響しないよう、しきい値は現行ロジック互換を優先する。

### 作業ステップ

1. 現行 `killDenormal` の呼び出し経路を再確認（Audio Thread到達を明示）。
2. 非libm版 `killDenormal` 実装へ置換。
3. `std::fabs` 残存検索（同ヘッダ内、関連ヘッダ内）。
4. 単体差分レビュー（演算意味が同等であることを確認）。

### 完了条件

- `PsychoacousticDither.h` から `std::fabs` が除去されている。
- Debug/Release ビルド通過。
- 既存の簡易音質確認（null test / smoke）で重大退行なし。

### リスク/ロールバック

- リスク: 閾値解釈差で微小ノイズ挙動が変化する可能性。
- ロールバック: 条件付きコンパイルで旧実装を一時退避し比較可能にする。

---

## U4 (P1): `diagLog` の Release 出力ポリシー統一

### 対象

- `src/audioengine/AudioEngine.Commit.cpp`
- 必要に応じてログヘルパ定義箇所

### 実装方針

- `JUCE_DEBUG` と CI 用マクロを使って出力経路を明示的に分岐する。
- Releaseでは高頻度ログを抑制し、必要時のみ出力可能な仕組みを残す。

### 作業ステップ

1. `diagLog` の呼び出し頻度が高い箇所を棚卸し。
2. ビルド構成別ガード（Debug/CIのみ詳細出力）を適用。
3. Release時の副作用（ログ欠落による診断不能）がないか確認。

### 完了条件

- Release通常運用で不要なログI/Oが抑制される。
- Debug/CIで必要ログは維持される。

### リスク/ロールバック

- リスク: 障害解析時にログ不足。
- 緩和: CI専用フラグで詳細ログを再有効化できる導線を残す。

---

## U2 (P1): `DeviceSettings::loadSettings` の BulkRestore RAII化

### 対象

- `src/DeviceSettings.cpp`（または同等実装箇所）

### 実装方針

- begin/end の手動管理をスコープガードへ移行。
- 早期returnや例外相当経路でも `endBulkParameterRestore(true)` が漏れない構造にする。

### 作業ステップ

1. begin/end の現行対管理を抽出。
2. ローカルRAIIガード型（小規模）を導入。
3. `loadSettings` 全経路で end が保証されることを確認。
4. 既存復元順序・副作用の非変更を確認。

### 完了条件

- 手動 begin/end 管理が消えている。
- 復元挙動（成功/失敗時）が既存と整合。

### リスク/ロールバック

- リスク: 復元フロー終端タイミングの差異。
- 緩和: ログ/アサートで begin-end 対応回数を一時検証。

---

## U3 (P1): `StereoConvolver` 破棄契約の型強制

### 対象

- `src/audioengine/AudioEngine.h`
- `src/audioengine/AudioEngine.Commit.cpp`
- 関連 destroy/retire 実装

### 実装方針

- 通常 delete 不能な所有ハンドル（custom deleter）を導入し、破棄経路を一本化する。
- 既存 `destroyStereoConvolver` を唯一の解放実体として維持する。

### 作業ステップ

1. `StereoConvolver` 所有点・解放点を一覧化。
2. 型ラッパー（例: unique_ptr + custom deleter）設計。
3. 代入/退役/シャットダウン経路を順に差し替え。
4. 二重解放・解放漏れ観点をレビュー。

### 完了条件

- 破棄経路が実質1系統に統一される。
- shutdown/rebuild/cleanup 挙動が維持される。

### リスク/ロールバック

- リスク: 所有権移動時のヌル化漏れ。
- 緩和: debugアサートで「同時所有なし」を検証。

---

## U8 (P2): `SafeStateSwapper::tryReclaim` の利用契約明文化

### 対象

- `src/SafeStateSwapper.h`

### 実装方針

- Single Consumer 前提をコメントと debug assert で固定化。
- 誤用時に開発時早期検知できるようにする。

### 作業ステップ

1. `tryReclaim` 呼び出し元を再確認。
2. APIコメントに前提条件を追記。
3. debug時の契約チェックを追加。

### 完了条件

- 利用契約がヘッダ上で明確。
- 誤用時に検知可能。

---

## U5 (P2): `CustomInputOversampler` の `corruptionDetected` 運用段階化

### 対象

- `src/CustomInputOversampler.*`（該当実装）

### 実装方針

- まずは観測強化（発火原因分類、頻度計測）を先行。
- その後、回復可能ケースと回復不能ケースで処理を分離。

### 作業ステップ

1. 現在の発火条件を列挙。
2. 診断カウンタ/軽量ログを追加（非RT配慮）。
3. 回復戦略を2段階化（soft recovery / hard fallback）。
4. 音質と安定性を回帰確認。

### 完了条件

- 発火条件が追跡可能。
- 回復戦略が定義され、過敏反応が抑制される。

---

## U6 (P2): `CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_*` 分岐整理

### 対象

- `src/audioengine/AudioEngine.*`（split関連）

### 実装方針

- 常時ON前提の死に分岐を一括削除せず、PRを分割して段階整理。
- 挙動差が出る変更と、単純削除変更を分離する。

### 作業ステップ

1. 対象マクロと条件分岐を全列挙。
2. 「削除だけで意味不変」な箇所を先に処理。
3. 依存が強い箇所は後続PRへ分割。

### 完了条件

- 読解性が改善し、ビルド・挙動維持。
- 差分がレビュー可能なサイズで分割されている。

---

## U7 (P2): `CmaEsOptimizerDynamic` allocator方針の規約整合

### 対象

- `src/CmaEsOptimizerDynamic.h`
- `src/CmaEsOptimizerDynamic.cpp`
- 関連規約ドキュメント

### 実装方針

- まず「どこまで strict にするか」を文書合意。
- 合意後、必要な箇所のみ段階置換する。

### 作業ステップ

1. MKL利用箇所とメモリ確保箇所を仕分け。
2. 現状許容案と置換案の比較表を作成。
3. 合意後に限定置換（必要時のみ）。

### 完了条件

- 規約解釈が文書化される。
- 実装変更時に性能退行がない。

---

## U9 (P3): `AudioEngine.h` 分割計画策定

### 対象

- `src/audioengine/AudioEngine.h`
- 分割先候補ヘッダ

### 実装方針

- 先に設計計画のみ作成し、実装は子Issueへ分離。
- 分割単位は責務ベース（State/Runtime/Commit/Processing/Diagnostics 等）で検討。

### 作業ステップ

1. セクション単位の責務マップ作成。
2. include依存と初期化順序依存を洗い出し。
3. 分割順序（低リスク→高リスク）を定義。
4. 実装子Issueを作成。

### 完了条件

- 分割設計書と移行順序が定義済み。
- 子Issueに作業が分解済み。

---

## 5. 横断テスト計画

<!-- markdownlint-enable MD024 -->

## ビルド検証

- Debug ビルド
- Release ビルド

## 静的確認

- 問題パネルで変更ファイルエラーゼロ
- `killDenormal` の libm依存除去確認
- 破棄契約・観測契約のコメント/アサート整合確認

## 実行確認（スモーク）

- アプリ起動〜停止
- 設定ロード/保存
- 畳み込み再構築（可能な範囲）
- 音声処理経路（ノイズ、無音、通常入力）

---

## 6. マイルストーン案

- **M1（即時）**: U1, U4
- **M2（安定化）**: U2, U3, U8
- **M3（運用改善）**: U5, U6, U7
- **M4（構造改善）**: U9（計画策定のみ）

---

## 7. 完了判定（Definition of Done）

- U1〜U9の各Issueで、受け入れ条件がトレース可能。
- 優先度順の実施計画と依存関係が明示されている。
- 変更後のビルド/静的確認/スモーク確認の手順が定義済み。
- RT安全性と既存挙動維持の観点が全Issueで担保されている。

---

## 8. 実装用タスク分解（PR単位）

以下は、指定順 **U1 → U4 → U2 → U3 → U8 → U5 → U6 → U7 → U9** で実施するPR計画。

### PR-01: U1 `killDenormal` libm依存除去（P0）

- 目的: Audio Thread経路から `std::fabs` 依存を除去。
- 変更対象（想定）:
  - `src/PsychoacousticDither.h`
- 実装タスク:
  - [ ] `killDenormal` の非libm実装へ置換
  - [ ] 閾値判定の意味等価レビュー
  - [ ] `std::fabs` 残存確認
- 検証タスク:
  - [ ] Debug/Release build
  - [ ] 音声スモーク（無音/通常入力）
- PR完了条件:
  - [ ] `PsychoacousticDither.h` に libm依存が残らない
  - [ ] 回帰なし

### PR-02: U4 `diagLog` Release出力ポリシー統一（P1）

- 目的: Release時の過剰ログI/O抑制。
- 変更対象（想定）:
  - `src/audioengine/AudioEngine.Commit.cpp`
- 実装タスク:
  - [ ] `diagLog` を構成別（Debug/CI/Release）で分岐
  - [ ] 必要診断をDebug/CIで維持
- 検証タスク:
  - [ ] Release時ログ出力量の確認
  - [ ] Debug/CI相当で診断ログ維持確認
- PR完了条件:
  - [ ] Release通常運用で不要ログ抑制
  - [ ] デバッグ性維持

### PR-03: U2 `DeviceSettings::loadSettings` RAII化（P1）

- 目的: begin/end手動対管理の事故防止。
- 変更対象（想定）:
  - `src/DeviceSettings.cpp`
- 実装タスク:
  - [ ] `endBulkParameterRestore(true)` を自動化するガード導入
  - [ ] 早期return経路の網羅確認
- 検証タスク:
  - [ ] 設定ロード/保存スモーク
  - [ ] begin/end対応の崩れがないことをレビュー
- PR完了条件:
  - [ ] 手動対管理の解消
  - [ ] 既存復元挙動維持

### PR-04: U3 `StereoConvolver` 破棄契約の型強制（P1）

- 目的: 破棄経路を型で一本化し誤解放を予防。
- 変更対象（想定）:
  - `src/audioengine/AudioEngine.h`
  - `src/audioengine/AudioEngine.Commit.cpp`
- 実装タスク:
  - [ ] 所有ハンドル（custom deleter）導入
  - [ ] `destroyStereoConvolver` 経由へ統一
  - [ ] shutdown/rebuild/retire 経路の追従
- 検証タスク:
  - [ ] 二重解放/リーク観点レビュー
  - [ ] Debug/Release build
- PR完了条件:
  - [ ] 解放経路が一意
  - [ ] 既存挙動維持

### PR-05: U8 `SafeStateSwapper::tryReclaim` 契約明文化（P2）

- 目的: Single Consumer前提の誤用防止。
- 変更対象（想定）:
  - `src/SafeStateSwapper.h`
- 実装タスク:
  - [ ] APIコメントで前提条件明示
  - [ ] debug検知（assert）強化
- 検証タスク:
  - [ ] 呼び出し元との契約整合レビュー
- PR完了条件:
  - [ ] 前提がコード上で明確
  - [ ] 誤用検知可能

### PR-06: U5 `corruptionDetected` 運用段階化（P2）

- 目的: 過敏反応を抑え、診断可能性を上げる。
- 変更対象（想定）:
  - `src/CustomInputOversampler.*`
- 実装タスク:
  - [ ] 発火原因の分類情報を取得
  - [ ] soft/hard の回復戦略を分離
- 検証タスク:
  - [ ] 発火時挙動の期待通り確認
  - [ ] 音質・安定性スモーク
- PR完了条件:
  - [ ] 発火トレース可能
  - [ ] 回復戦略が定義済み

### PR-07: U6 splitマクロ分岐整理（P2）

- 目的: 常時ON前提分岐の読解性改善。
- 変更対象（想定）:
  - `src/audioengine/AudioEngine.*`
- 実装タスク:
  - [ ] マクロ分岐の棚卸し
  - [ ] 意味不変の枝を先行削除
  - [ ] 依存が強い箇所は別PR候補として切り出し
- 検証タスク:
  - [ ] Debug/Release build
  - [ ] 差分レビュー（意味不変性）
- PR完了条件:
  - [ ] 可読性向上
  - [ ] 挙動維持

### PR-08: U7 allocator方針の規約整合（P2）

- 目的: 実装と規約解釈の齟齬解消。
- 変更対象（想定）:
  - `src/CmaEsOptimizerDynamic.h`
  - `src/CmaEsOptimizerDynamic.cpp`
  - 関連ドキュメント
- 実装タスク:
  - [ ] 許容案/置換案の比較表作成
  - [ ] 合意内容の文書化
  - [ ] 必要最小限の置換（必要時のみ）
- 検証タスク:
  - [ ] 性能退行有無の確認
- PR完了条件:
  - [ ] 規約整合が明文化
  - [ ] 変更時に性能退行なし

### PR-09: U9 `AudioEngine.h` 分割計画（設計PR, P3）

- 目的: 巨大ヘッダ分割を安全に進める設計確立。
- 変更対象（想定）:
  - `src/audioengine/AudioEngine.h`
  - `doc/work/*`（計画書）
- 実装タスク:
  - [ ] 責務マップ作成
  - [ ] include依存/初期化順序依存の整理
  - [ ] 分割順序と子Issue定義
- 検証タスク:
  - [ ] 計画レビュー（RT制約/ABI/初期化順序観点）
- PR完了条件:
  - [ ] 分割実装へ移行可能な設計計画が完成

---

## 9. PR運用ルール（共通）

- 1PR = 1目的を厳守し、横展開は次PRへ分離する。
- 各PRの説明テンプレ:
  1. 背景
  2. 変更点（箇条書き）
  3. 非変更点（やらないこと）
  4. 検証結果（Debug/Release、追加確認）
  5. リスクとロールバック
- マージゲート:
  - [ ] 変更ファイルにエラーなし
  - [ ] ビルド成功（Debug/Release）
  - [ ] RT制約違反なし（レビュー）
  - [ ] 受け入れ条件チェック完了

---

## 10. 実装ブランチ作業手順（コミット粒度）

本節は、`PR-01` から順番に実装する際の「ブランチ運用」と「コミット分割」の標準手順を定義する。

### 10.1 共通手順（全PR）

1. `main` 最新化後に対象PR専用ブランチを作成。
2. 変更は **実装コミット** と **検証/文書コミット** を分離。
3. 1コミット1論点を厳守（混在禁止）。
4. Debug/Release 検証を行い、結果をPR本文へ反映。
5. マージ後、次PRブランチは必ず `main` から新規作成（チェーンブランチ禁止）。

### 10.2 ブランチ命名規約

- `feature/u1-killdenormal-no-libm`
- `feature/u4-diaglog-release-gating`
- `feature/u2-bulkrestore-raii`
- `feature/u3-stereoconvolver-owned-handle`
- `feature/u8-safestateswapper-contract`
- `feature/u5-corruption-detection-staged`
- `feature/u6-audioengine-split-branch-cleanup`
- `feature/u7-cmaes-allocator-policy`
- `feature/u9-audioengine-header-split-plan`

### 10.3 PR別コミット分割

### PR-01 / U1

- C1: `refactor(dsp): remove libm dependency from killDenormal`
  - `killDenormal` を非libm実装へ置換。
- C2: `test(rt): verify no fabs usage and keep denormal behavior`
  - 残存確認、回帰観点メモ追記（必要ならコメント）。

### PR-02 / U4

- C1: `refactor(logging): gate diagLog by build configuration`
  - Debug/CI/Release 分岐導入。
- C2: `chore(logging): document release logging policy`
  - ログ運用意図をコメントで固定。

### PR-03 / U2

- C1: `refactor(settings): add RAII guard for bulk restore`
  - begin/end の自動対管理化。
- C2: `test(settings): verify loadSettings early-return safety`
  - 早期経路で end 漏れがないことを確認。

### PR-04 / U3

- C1: `refactor(audioengine): introduce owned handle for StereoConvolver`
  - custom deleter ハンドル導入。
- C2: `refactor(audioengine): route destruction via destroyStereoConvolver`
  - 解放経路統一。
- C3: `test(audioengine): validate rebuild/shutdown lifecycle`
  - ライフサイクル観点の検証反映。

### PR-05 / U8

- C1: `docs(thread-safety): define tryReclaim single-consumer contract`
  - 利用契約コメント追加。
- C2: `debug(thread-safety): add misuse assertion for tryReclaim`
  - 誤用検知アサート追加。

### PR-06 / U5

- C1: `feat(oversampler): add corruption detection telemetry`
  - 発火原因の可視化。
- C2: `refactor(oversampler): split recovery into soft/hard paths`
  - 回復戦略段階化。
- C3: `test(oversampler): verify stability and audio smoke`
  - 回帰確認結果の反映。

### PR-07 / U6

- C1: `refactor(audioengine): remove no-op split branches (safe subset)`
  - 意味不変の枝削除のみ。
- C2: `chore(audioengine): annotate deferred branch cleanup items`
  - 後続分離項目を明文化。

### PR-08 / U7

- C1: `docs(allocator): add cmaes allocator policy comparison`
  - 許容案/置換案の文書化。
- C2: `refactor(cmaes): apply agreed minimal allocator changes`
  - 合意範囲のみ変更。
- C3: `perf(cmaes): record no-regression verification`
  - 性能退行なしを記録。

### PR-09 / U9

- C1: `docs(audioengine): map header responsibilities for split plan`
  - 責務マップ作成。
- C2: `docs(audioengine): define include/init-order risk matrix`
  - 依存/初期化順序リスク整理。
- C3: `docs(audioengine): create child issues for split execution`
  - 実装フェーズ分割。

### 10.4 進行ゲート（次PR着手条件）

- 現PRが以下を満たしたら次PRへ進む:
  - [ ] レビュー指摘の必須対応完了
  - [ ] Debug/Release の最終結果をPR本文に記録
  - [ ] 受け入れ条件をチェック済み
  - [ ] `main` へマージ済み
