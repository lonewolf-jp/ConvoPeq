# H-2-R1（R-04）測定記録（2026-10-30 実行 / 記録 2026-11-02）

Owner 指示 3（`ci/h2r1` は現状維持。測定値・実行 ID・運用コストを追跡可能な文書に記録する）に
基づく記録。**CI workflow・ブランチの変更、main への統合、削除は行っていない。**

## 1. 実行の識別情報

| 項目 | 値 |
|---|---|
| 実行 ID | **`40996108355`** |
| URL | `https://github.com/lonewolf-jp/ConvoPeq/actions/runs/40996108355` |
| ワークフロー | `H2R1 Release measurement (R-04)`（`.github/workflows/h2r1-release-measurement.yml`） |
| 実行ブランチ | `ci/h2r1-release-measurement-20261030` |
| 実行契機 | `push`（当該ブランチへの push） |
| 作成 | 2026-10-30T03:12:56Z |
| ジョブ開始 | 2026-10-30T03:12:58Z |
| ジョブ完了 | 2026-10-30T03:59:56Z |
| ジョブ所要 | **46 分 58 秒** |
| 結論 | **success** |

## 2. 測定値（5 回すべて同一）

コマンド: `ctest --test-dir build-ci -C Release -R '^AudioEngineHarness$' -V`

| 回 | 終了コード | 所要 | `[H2R1]` 行 |
|---|---|---|---|
| 1 | 0 | 303.4 s | `trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80` |
| 2 | 0 | 302.9 s | 同上 |
| 3 | 0 | 302.6 s | 同上 |
| 4 | 0 | 302.5 s | 同上 |
| 5 | 0 | 302.3 s | 同上（行末 `:: PASS`） |

## 3. 合否基準との対応

ワークフロー冒頭に定義された基準に対する充足状況。

| 基準 | 内容 | 実測 | 判定 |
|---|---|---|---|
| C1 | `mixed_trials == 0` | 5 回とも 0 | 充足 |
| C2 | `distinct_worlds == 2 * trials`（= 80） | 5 回とも 80 | 充足 |
| C3 | 5 回すべて終了コード 0 | 5/5 で rc=0 | 充足 |

## 4. 留保（過大評価を避けるための記録）

- 測定コード自身が「`[H2R1]` 行は判定を返さない。0 件は**観測限界**であって不在の証明ではない」と
  明記している（`src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp:1262`）。
  本記録は「基準 C1〜C3 を満たした」ことを示すもので、欠陥の不在を証明するものではない。
- 測定は `captureBuildParameterSnapshot()` が 13 フィールドを個別の acquire 読み込みで取得する
  ため、(new headroom, old makeup) という混成タプルが **published world になる頻度と持続時間**を
  数えるもの（同ファイル L1252〜L1264）。
- ローカル実行では `D18-D structural ordering` と `D12-3 UI bit depth list stays in set` が
  環境要因で FAIL するため、**同等の測定は CI でしか成立しない**
  （`CONVOPEQ_SOURCE_ROOT` は `CMakeLists.txt:2351` の `set_tests_properties` が ctest 経由でのみ供給する）。

## 5. 運用コスト

| 項目 | 値 |
|---|---|
| 常時のコスト | **0**（`pull_request` トリガ無し。`push` は自ブランチ 1 本のみ。main では発火しない） |
| 1 回の実行 | windows-latest で約 **47 分**（ビルド約 20 分 ＋ 5 回 × 約 5 分） |
| 上限 | `timeout-minutes: 300` |
| 実行方法 | 手動 dispatch、または当該ブランチへの push |

## 6. 現状維持の内容（本記録時点）

- ブランチ `ci/h2r1-release-measurement-20261030` は **保持**（tip `d7036618`）。
- ワークフローは **main に統合していない**。恒久化も削除もしていない。
- 内容は 3 か所で保全されている: ブランチ tip、アーカイブタグ
  `archive/ci-h2r1-release-measurement-20261030-d7036618`、本記録の実行 ID。
- **実行ログは GitHub の保持期間で失効する**。本記録が計測値の一次証拠となる。

## 7. 残件

| # | 残件 | 内容 |
|---|---|---|
| 1 | **統合後の再測定** | 2026-10-30 の測定後に B-2 / A2→A3-a（`1f258686`）と j237 ガード（`0dfa39e3`）が main に入った。測定対象のコード（`captureBuildParameterSnapshot()` と測定関数）は両マージで未変更だが、**統合後の再測定は未実施**。必要なら 1 回の dispatch で確認できる |
| 2 | 現行 main を測る手段 | 当該ブランチは main より **71 コミット遅れ**。現行 main を測るには、ワークフローを main 側に置く（`workflow_dispatch` のみに整理）必要がある。恒久化は CI 定義変更のため**未承認** |
| 3 | 実行ログの失効 | 保持期間経過後は本記録とタグのみが証拠になる |

## 8. 変更していないもの

- `.github/workflows/h2r1-release-measurement.yml`：未変更（main に存在しない）
- `.github/workflows`・policy・検査スクリプト：本作業で変更なし
- ブランチ・タグ：`ci/h2r1` は保持（アーカイブタグは既存）
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD：維持
