# `ci/h2r1` の扱いに関する判断案（2026-11-02）

Owner 指示 5（`ci/h2r1` は恒久化・削除のいずれも実施せず、必要性・運用コスト・main への影響を
整理した判断案を提出）に対する資料。**本作業ではブランチもワークフローも変更していない。**

## 1. 対象の実測

| 項目 | 値 |
|---|---|
| ブランチ | `ci/h2r1-release-measurement-20261030` |
| tip | `d7036618` |
| 追加ファイル | `.github/workflows/h2r1-release-measurement.yml`（11,575 B / 235 行） |
| policy 差分 | `.github/isr-ai-governance-policy.json` を 1 行（**allowlist の旧内容**） |
| main との位置関係 | 分岐 `ff483c2c`、3 ahead / 71 behind |
| アーカイブタグ | `archive/ci-h2r1-release-measurement-20261030-d7036618`（削除しても内容は復元可能） |

### 1.1 ワークフローの構成

- `name: H2R1 Release measurement (R-04)`
- トリガ: `workflow_dispatch` と、**`push` の対象をこのブランチ 1 本に限定**
  （`branches: [ci/h2r1-release-measurement-20261030]`）。
  **main への push でも PR でも起動しない。**
- ジョブ: `measure` 1 本。`runs-on: windows-latest`、`timeout-minutes: 300`、
  `permissions: contents: read`、`concurrency` は同一 ref で直列化。
- 内容: Release ビルド後、`ctest -C Release -R '^AudioEngineHarness$' -V` を **5 回**実行し、
  各回の `[H2R1]` 行と終了コードを検証する。

### 1.2 なぜ CI で測るのか（ワークフロー自身の記載）

ローカルで直接叩くと、`D18-D structural ordering` が `CONVOPEQ_SOURCE_ROOT` 未設定で FAIL し
（`CMakeLists.txt:2351` の `set_tests_properties` が ctest 経由でのみ供給する）、
`D12-3 UI bit depth list stays in set` もテストの CWD から `src/DeviceSettings.cpp` を
解決できず FAIL する。ctest 経由なら CWD = ビルドディレクトリで両方が満たされる。
つまり**この 2 テストが通る環境を再現するため**に CI を使っている。

### 1.3 測定は完了している（実測）

実行 `40996108355`（2026-10-30、event=push、conclusion=**success**）。

| 回 | 終了コード | 所要 | 計測値 |
|---|---|---|---|
| 1 | 0 | 303.4 s | `trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80` |
| 2 | 0 | 302.9 s | 同上 |
| 3 | 0 | 302.6 s | 同上 |
| 4 | 0 | 302.5 s | 同上 |
| 5 | 0 | 302.3 s | 同上（`:: PASS`） |

- ジョブ所要: 03:12:58 → 03:59:56 = **46 分 58 秒**（run 全体も 03:12:56 → 03:59:56）。
- ワークフロー冒頭に書かれた基準（C1 `mixed_trials==0`、C2 `distinct_worlds==2*trials`、
  C3 5 回すべて終了コード 0）は、**この実行で充足している**
  （`mixed_trials=0`、`distinct_worlds=80 = 2×40`、5/5 で rc=0）。

## 2. 必要性

| 観点 | 評価 |
|---|---|
| 当初の目的 | **達成済み**（2026-10-30 の実行で C1〜C3 を充足） |
| 測定対象のコード | `captureBuildParameterSnapshot()`（`AudioEngine.RebuildDispatch.cpp:43`）と測定関数 `measureH2R1MixedGainTupleWindow()`（`PublishPipelineIntegrationTests.cpp:1265`）は、B-2 / A2→A3-a のマージ `1f258686` でも j237 ガードの `0dfa39e3` でも**変更されていない**（同ファイルへの差分は opt-in テストの入口追加 14 行のみ） |
| 統合後の再測定 | **未実施**。B-2 は `AudioEngine.h` / `ISRDSPHandle.cpp` を変更しており、測定対象の経路への間接影響を否定できない。必要なら 1 回の dispatch で確認できる |
| 将来の再測定 | 起こり得る（統合後の再確認、回帰調査）。その場合このワークフローが最短手段 |
| 代替手段 | ローカルでは D18-D と D12-3 が環境要因で FAIL するため、**同等の測定は CI でしか成立しない** |
| 注意 | テスト自身が「`[H2R1]` 行は判定を返さない。0 件は観測限界であって不在の証明ではない」と明記している（`PublishPipelineIntegrationTests.cpp:1262`）。したがって本ワークフローの PASS は「基準 C1〜C3 の充足」を意味し、欠陥の不在証明ではない |

## 3. 運用コスト

| 項目 | 値 |
|---|---|
| 常時のコスト | **0**（main / PR では起動しない。ブランチへの push か手動 dispatch のみ） |
| 1 回の実行コスト | windows-latest で約 **47 分**（ビルド約 20 分 ＋ 5 回 × 約 5 分） |
| 失敗時のコスト | 同上（`timeout-minutes: 300` が上限） |
| 恒久化した場合の追加コスト | 0（トリガを `workflow_dispatch` のみにすれば、明示的に回した時だけ消費） |
| 放置した場合のコスト | 0（実行されない限り消費しない） |

## 4. main へ取り込んだ場合の影響

| 観点 | 影響 |
|---|---|
| PR チェック | **なし**（トリガに `pull_request` が無い） |
| main への push | **なし**（`push` の対象が ci ブランチ 1 本に限定されているため、main では発火しない） |
| 必須チェック | 追加されない（ワークフローを足すだけでは required status checks は変わらない） |
| 既存ワークフローとの競合 | なし（`concurrency.group` は自前の ref 単位） |
| **注意点 1** | ブランチ側の policy 1 行は**旧 allowlist**（`std::vector<void*>`）。main は既に `std::vector<DSPCore*>` へ同期済み（`c9e74603`）。**この 1 行は取り込んではならない** |
| **注意点 2** | そのまま入れると `push` トリガの対象ブランチが main に存在せず、**死んだトリガ**が残る（動作上の害は無いが、意図が読みにくい） |
| 恒久化の手間 | トリガを `workflow_dispatch` のみに整理し、policy 差分を落として 1 ファイルだけを追加する形にするのが自然 |

## 5. 選択肢

| 案 | 内容 | 利点 | 欠点 | 必要な承認 |
|---|---|---|---|---|
| **A. 恒久化** | `workflow_dispatch` のみに整理した 1 ファイルを main へ追加 | 再測定が誰でも 1 クリックで可能。手順がコードとして残る | main に測定専用ファイルが 1 つ増える。CI 定義変更にあたる | **CI 定義変更の個別承認** |
| **B. 現状維持（ブランチ保持）** | 何もしない | リスク 0・追加変更 0。すぐ回せる | ブランチが 1 本残る。実行ログは 90 日で失効する | 不要（現状） |
| **C. 削除** | ブランチを削除（tip はタグで保全） | 棚卸しが進む | 再測定のたびにタグから復元する手間。ログ失効後は計測値の一次証拠が消える | ブランチ削除の個別承認 |

## 6. 推奨

**「計測値の記録を追跡文書に残す」ことを先に実施し、恒久化（案 A）は別途承認を待つ。**
そのうえで、当面は **案 B（現状維持）** を採る。

理由:

1. **記録が最優先**。実行ログは GitHub の保持期間（既定 90 日、2027-01 頃に失効）で消える。
   一方、計測値（`trials=40` / `mixed_trials=0` / `distinct_worlds=80` / 5 回すべて rc=0）は
   **現在どの追跡文書にも記載が無い**（`distinct_worlds=80` の一致 0 件）。
   ここが欠けると、後から「基準を満たした事実」を証明できなくなる。
2. 案 A は技術的リスクが低い（PR・main に発火しない）が、**CI 定義変更**なので
   Owner の個別承認が前提。判断材料は本資料で揃った。
3. 案 C は記録を残した後なら妥当。ただし急ぐ理由が無く、`ci/r49` を保持する方針とも
   整合させやすい。

### 6.0 統合後の再測定を行いたい場合の注意

- 現ブランチは **main より 71 コミット遅れている**。このブランチ上で dispatch しても
  **現行 main の測定にはならない**。
- 現行 main を測るには、案 A でワークフローを main へ入れ、main 上で dispatch するのが
  最も素直（`workflow_dispatch` のみに整理すれば main では push 発火しない）。
- この用途があるなら、案 B（現状維持）ではなく **案 A を選ぶ理由**になる。

### 6.1 記録すべき内容（案）

| 項目 | 値 |
|---|---|
| 実行 ID | `40996108355` |
| 日時 | 2026-10-30 03:12:56 → 03:59:56（ジョブ 03:12:58 → 03:59:56） |
| 構成 | Release / windows-latest / ctest `^AudioEngineHarness$` を 5 回 |
| 各回 | rc=0、302.3〜303.4 s |
| 計測値 | `trials=40 mixed_trials=0 max_mixed_ms=0.0 distinct_worlds=80`（5 回とも同一） |
| 基準充足 | C1・C2・C3 すべて充足 |
| 未実施 | 統合（B-2 / A2→A3-a / j237 ガード）後の再測定 |
| 留保 | `[H2R1]` は判定を返さない。0 件は観測限界であって不在の証明ではない |

## 7. 承認をお願いする項目

| # | 項目 | 内容 |
|---|---|---|
| 1 | 計測値の追跡文書化 | 上記 6.1 を `doc/` に記録する（CI 定義は変えない） |
| 2 | 案 A の可否 | 恒久化する場合、`workflow_dispatch` のみに整理し policy 差分を落として 1 ファイル追加 |
| 3 | 案 C の可否 | 削除する場合、記録の追跡化後に削除（tip はタグ保全済み） |

## 8. 変更していないもの

- `ci/h2r1-release-measurement-20261030` ブランチ: 未変更（tip `d7036618`）
- `.github/workflows` / policy / 検査スクリプト: 未変更（`6bb7c8d5..HEAD` で workflow 0 件、スクリプト 0 件）
- ソース・テスト: 未変更
- U5 = FAIL / 既定 OFF、U3/U4 = HOLD: 維持
