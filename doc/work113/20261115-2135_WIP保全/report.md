# 未マージ WIP の保全と復元可能性の確認（2026-11-15 21:35）

- 対象: `C:\VSC_Project\ConvoPeq` の未コミット WIP（HEAD `4d476d5c65261fd975de2606f4d8b5c843cd86e1`、ブランチ `main`）
- 目的: Owner 方針（「別ブランチへ退避を先行し、その後 `296ecfeb` へ rebase・統合」）の第 1 段階。**保全と復元可能性の確認まで**を行い、ブランチ作成・rebase・統合は実施していない
- 制約: 元の作業ツリーは一切変更していない（`git stash` / `checkout` / `rebase` / 未追跡の削除はいずれも未実施）

## 1. 結論

| 項目 | 結果 |
| --- | --- |
| 保全物 | `C:\VSC_Project\ConvoPeq-WIP-4d476d5c-20261115\`（398 ファイル、約 94 MB） |
| 復元検証 | **394 ファイルを比較して SHA-256 がすべて一致（MISMATCH 0 / MISSING 0）** |
| 元の作業ツリー | 未変更（HEAD `4d476d5c`、staged 0 件、status 192 エントリ） |
| 副産物 | 未追跡の実数は **352 ファイル**（`git status` の 149 エントリはディレクトリ集約のため過小） |
| 判明した注意点 | 追跡ファイル 4 件が **CRLF** で保存されており、`git apply` では復元できない（後述） |

## 2. 保全物の構成

`C:\VSC_Project\ConvoPeq-WIP-4d476d5c-20261115\`

| ファイル／ディレクトリ | 内容 | サイズ |
| --- | --- | --- |
| `tracked.patch` | `git diff --binary HEAD`（追跡 42 件の差分。テキストのみ） | 756,330 B |
| `files/` | 追跡 42 件の**バイトコピー**（改行を保持。復元の主経路） | 8.47 MB |
| `untracked/` | 未追跡 352 件のバイトコピー（パス構造を保持） | 84.95 MB |
| `status.txt` | `git status --porcelain` の生出力 | 13,058 B |
| `hashes.tsv` | WIP 全 394 ファイルの SHA-256・サイズ・パス | 49,762 B |
| `MANIFEST.md` | 上記の要約と復元手順 | 1,272 B |

`untracked/` の大半（84.95 MB）は過去の `doc/work113/*/ConvoPeq.md` とその分割版（各 6.7〜6.8 MB）で、ソースコードの未追跡分は 11 ファイル（`src/InputInspection.cpp/.h`、`src/learner/` の 3 件、`src/convolver/ConvolverStopProbe.h`、`src/convolver/IRTrailingSilence.h`、テスト 6 本）と `.github/scripts/isr-verify-evidence-consistency.ps1`、`resf_raw.json` のみ。

## 3. 復元検証（実行した内容）

1. `git worktree add --detach <verify-dir> 4d476d5c…` で基点のクリーンなツリーを作成
2. `git apply --binary tracked.patch` で差分を適用
3. `files/` を上書きコピー（42 件）
4. `untracked/` をコピー（352 件）
5. `hashes.tsv` の全 394 行について SHA-256 を比較

結果: **COMPARED=394 MISMATCH=0 MISSING=0**。検証用 worktree は削除済み（`git worktree list` に残っていない）。

## 4. 判明した注意点 — CRLF の 4 ファイル

`git apply` だけでは 4 ファイルがバイト一致しなかった。

| ファイル | 元（作業ツリー） | patch 適用後 | 差 |
| --- | ---: | ---: | ---: |
| `ConvoPeq.md` | 7,067,912 B | 6,911,809 B | 156,103 B |
| `src/MainWindow.cpp` | 81,952 B | 80,041 B | 1,911 B |
| `src/convolver/ConvolverProcessor.LoadPipeline.cpp` | 48,228 B | 47,214 B | 1,014 B |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | 53,857 B | 52,744 B | 1,113 B |

差はすべて CRLF → LF の変換量と一致する。`.gitattributes` は `*.cpp` / `*.h` に `eol=lf` を指定しており、`git apply` はそれに従って LF で書き出す。一方、**現在の作業ツリーのこの 4 ファイルは CRLF で保存されている**（`core.autocrlf=false`）。したがって:

- `tracked.patch` は差分のレビュー用であり、単独ではバイト完全な復元にならない
- 復元には `files/` のバイトコピーが必須（本スナップショットはそれを含む）
- この CRLF は WIP 側の状態であり、`296ecfeb` へ統合する際に LF へ正規化するかどうかを決める必要がある（統合時に無関係な全行差分が出る原因になる）

## 5. 復元手順（元の作業ツリーを変更しない）

```bash
# 1. 基点のツリーを別ディレクトリに用意
git worktree add --detach <restore-dir> 4d476d5c65261fd975de2606f4d8b5c843cd86e1

# 2. 追跡ファイルの差分（レビュー用。改行は LF に正規化される）
git -C <restore-dir> apply --binary <snapshot>/tracked.patch

# 3. バイトコピーで上書き（これが復元の本体）
cp -r <snapshot>/files/.     <restore-dir>/
cp -r <snapshot>/untracked/. <restore-dir>/

# 4. 照合
#    hashes.tsv の各行（sha256 / size / path）と実ファイルを比較し、全 394 件一致を確認
```

`files/` と `untracked/` をコピーするだけでも復元できる（`tracked.patch` は差分確認用）。

## 6. 次の段階（Owner 承認待ち）

1. WIP 専用ブランチの作成（元の作業ツリーを切り替えず、`git worktree` で別ディレクトリに作る方法を推奨）
2. 復元したツリーで WIP の内容と目的を整理（何を意図した変更かの帰属確定）
3. `296ecfeb` を基点に rebase。重複 3 ファイル（`CMakeLists.txt`、`src/MKLNonUniformConvolver.cpp`、`src/tests/TailLayerIntegrityTests.cpp`）は PR #216 の変更を維持したまま個別に統合
4. ビルド・テスト・独立レビューを経て PR 化

## 7. 未実施（今回の範囲外）

- WIP 専用ブランチの作成、rebase、統合、PR 化
- 元の作業ツリー上でのあらゆる変更操作（stash / checkout / 未追跡の削除を含む）
- WIP の内容の帰属確定（どのタスクの変更かの特定）
- Dropbox への `untracked/doc/`（84.95 MB の過去スナップショット）の複製。ローカル保全のみとした（過去の ConvoPeq.md と分割版で、既に別フォルダへ配信済みのものが大半のため）

## 9. Dropbox 配置

`/ConvoPeq-WIP-4d476d5c-20261115/`

| パス | 内容 |
| --- | --- |
| `report_wip_preservation.md` | 本報告 |
| `report_open_items_corrected.md` | 未修正バグ・残課題の一覧（未追跡数を 352 に訂正した版） |
| `MANIFEST.md` / `hashes.tsv` / `status.txt` | スナップショットの要約・全 394 ファイルの SHA-256・status 生出力 |
| `tracked.patch` | 追跡 42 件の差分（756,330 B） |
| `files/` | 追跡 42 件のバイトコピー（8.47 MB、`ConvoPeq.md` を含む） |
| `untracked_nondoc/` | 非 doc の未追跡 15 ファイル（`src/InputInspection.*`、`src/learner/`、`ConvolverStopProbe.h`、`IRTrailingSilence.h`、ISR 検証スクリプト、テスト 6 本、`resf_raw.json`） |

`untracked/doc/`（337 ファイル）はローカルのスナップショットにのみ存在する。

## 8. 補足 — 既存報告の数値訂正

`doc/work113/20261115-2115_残課題一覧/report.md` §6.2 の「未追跡 149」は `git status` のエントリ数（ディレクトリ集約）で、実ファイル数は **352**。同報告を訂正済み。
