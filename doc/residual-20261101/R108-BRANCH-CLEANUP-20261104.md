# R-108 ブランチ整理と最終棚卸し（2026-11-04）

Owner 指示（R-108 の安全条件 5 項目）に基づくブランチ削除の記録。削除前にアーカイブタグを作成し、SHA を固定している。

## 1. 判定条件（全ブランチ共通）

| # | 条件 | 確認方法 |
|---|---|---|
| 1 | main に必要な変更が統合済み | `git merge-base --is-ancestor` または `git diff main <branch>` の追加行 0 |
| 2 | 未統合の独自変更がない | `git diff main <branch> -- src CMakeLists.txt build.bat .github` の **追加行が 0** |
| 3 | 未完了 PR・他作業からの参照がない | `gh pr list --state open` が空、参照は記録文書のみ |
| 4 | 証跡・成果物が失われない | 削除前に `archive/*` タグを付与し、コミットを復元可能にした。実行ログは Dropbox に保存済み |
| 5 | 削除前 SHA と削除後状態を記録 | 下表と本記録 |

## 2. 削除したブランチ（ci/h2r1・ci/r49）

| ブランチ | 削除前 SHA | アーカイブタグ | 判定 |
|---|---|---|---|
| `ci/h2r1-release-measurement-20261030` | `d70366182b93` | `archive/ci-h2r1-20261030-d7036618` | 変更は main のワークフローに統合済み（main は oneAPI 4 点検証などを追加した上位版） |
| `ci/r49-selected-tests-probe-20261029` | `96b46aa0eac4` | `archive/ci-r49-20261029-96b46aa0` | 診断 3 ステップは main の `windows-msvc-build.yml` に移植済み |

削除はローカル・リモートの両方で実施した。削除後、`refs/heads/ci/*` は **0 本**。

## 3. 併せて整理した ci ブランチ（同一条件を満たす）

| ブランチ | SHA | アーカイブタグ | 根拠 |
|---|---|---|---|
| `ci/h2r1-permanent-20261102` | `4e435b18` | `archive/ci-h2r1-permanent-20261102-4e435b18` | 変更は `.github/workflows/h2r1-release-measurement.yml` のみ。main は同ファイルの上位版 |
| `ci/r49-diagnostics-20261102` | `db25bef5` | `archive/ci-r49-diagnostics-20261102-db25bef5` | 変更は `windows-msvc-build.yml` のみ。診断 3 ステップは main に移植済み |
| `ci/sanitizer-oneapi-r48-20261029` | `45af6df3` | `archive/ci-sanitizer-oneapi-r48-20261029-45af6df3` | main の祖先（完全マージ） |
| `ci/windows-msvc-build-20261029` | `32932387` | `archive/ci-windows-msvc-build-20261029-32932387` | main の祖先（完全マージ） |

## 4. 併せて整理した作業ブランチ（squash マージ済み）

`git diff main <branch> -- src CMakeLists.txt build.bat .github` の**追加行がすべて 0**（main が上位版）であることを確認して削除した。

| ブランチ | SHA | アーカイブタグ |
|---|---|---|
| `docs/final-20261103b` | `6b745937` | `archive/docs-final-20261103b-6b745937` |
| `docs/ledger-u3-final-20261103` | `f293cb5a` | `archive/docs-ledger-u3-final-20261103-f293cb5a` |
| `docs/r101-r102-recheck-20261104` | `6f6a5bf2` | `archive/docs-r101-r102-recheck-20261104-6f6a5bf2` |
| `docs/r106-metric-proposal-20261103` | `8328a26c` | `archive/docs-r106-metric-proposal-20261103-8328a26c` |
| `feat/r106-metric-v2-measurement` | `a0bad07a` | `archive/feat-r106-metric-v2-measurement-a0bad07a` |
| `feat/r106-metric-v2-result` | `a756dea3` | `archive/feat-r106-metric-v2-result-a756dea3` |
| `feat/r113-r111-oneapi-u3-20261103` | `2405e545` | `archive/feat-r113-r111-oneapi-u3-20261103-2405e545` |

## 5. 削除後の状態

| 項目 | 結果 |
|---|---|
| ローカルブランチ | `main`、`measure/u5-20261102`（U5 の `[U5ATTR]` 保持のため残す）、作業中の `feat/r101-oneapi-cache` |
| リモートの `ci/*` | **0 本** |
| main | 変更なし（`7a34fa40`） |
| 作業ツリー | clean |

## 6. リモート全体の棚卸し（49 本）

`refs/heads/*`（main と作業中の `feat/r101-oneapi-cache` を除く）を 1 本ずつ検査した。

| 区分 | 本数 | 判定 | 処置 |
|---|---|---|---|
| main の祖先（完全マージ） | **42** | コミットは main の履歴に含まれる（未統合変更 0） | 削除（タグ不要） |
| 非祖先・独自コミットあり | **7** | コード差分の追加行はすべて**旧版の内容**（main が後続の refactor で置換済み）。新規の製品コードなし | アーカイブタグを作成して削除 |

非祖先 7 本と根拠:

| ブランチ | SHA | アーカイブタグ | 根拠 |
|---|---|---|---|
| `docs/r101-status-20261103` | `1b9227de` | `archive/docs-r101-status-20261103-1b9227de` | 差分は U4/U3 前の旧コード。独自の新規コードなし |
| `docs/r101-final2-20261103` | `d24d554e` | `archive/docs-r101-final2-20261103-d24d554e` | 同上 |
| `docs/r101-complete-20261103` | `3f1aa21b` | `archive/docs-r101-complete-20261103-3f1aa21b` | 同上 |
| `docs/final-status-20261103` | `d20897ab` | `archive/docs-final-status-20261103-d20897ab` | 同上 |
| `docs/final-20261102` | `fc0b9731` | `archive/docs-final-20261102-fc0b9731` | 同上 |
| `feat/r107-u4-lock-measure` | `e0ad7759` | `archive/feat-r107-u4-lock-measure-e0ad7759` | U4 実装の旧版。main は #108 で同内容を統合済み |
| `fix/stg8-midrun-clear-test` | `a2caa201` | `archive/fix-stg8-midrun-clear-test-a2caa201` | `ISRDSPHandle.cpp` は `LockScope` 化前の旧版（main が新しい） |

検査方法: `git diff --numstat main <branch> -- src CMakeLists.txt build.bat .github`。追加行があっても、その内容が U4 の `LockScope` 化前・U3 の `RtAllocProbe` 追加前といった**旧版**であることを個別に確認した（例: `fix/stg8-midrun-clear-test` の `ISRDSPHandle.cpp` は main が `convo::rtmeasure::LockScope`、ブランチが `std::lock_guard`）。

## 7. 削除後のリモート状態

| 項目 | 結果 |
|---|---|
| リモートブランチ | `main` と `feat/r101-oneapi-cache`（作業中）のみ |
| アーカイブタグ（リモート） | **37** 本（今回追加 13 本を含む） |
| main | `8561091d`（#121 マージ後。ブランチ削除時点は `7a34fa40`） |
| 証跡 | 削除前 SHA を本記録とタグに保存。実行ログは Dropbox に保管済み |
