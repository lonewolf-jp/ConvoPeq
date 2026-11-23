# PR #211 の main マージ報告

- 実施日時: 2026-11-14 23:10〜23:20 JST（CI 結果は §4 に追記）
- 対象 PR: **#211** `fix(resf): RES-F を仕様 §3.3 の測定方法に合わせ、Release 専用の性能テストへ分離`
- ブランチ: `fix/resf-spec-compliance` → `main`
- マージ方式: merge commit（リポジトリの既存運用に合わせる）
- マージコミット: **`783f3d24dbda110827770c66a2108a370698117f`**
- 親: `4d476d5c65261fd975de2606f4d8b5c843cd86e1`（直前の main）＋ `e32e7d058b6c38325e5c5c795430b7cb25ae16d1`（PR head）

---

## 1. マージ前確認（指示 2.1）

| # | 確認項目 | 実測 | 判定 |
| --- | --- | --- | --- |
| 1 | PR 番号 | `#211` | 一致 |
| 2 | base / head | `main` / `fix/resf-spec-compliance` | 一致 |
| 3 | head commit | `e32e7d058b6c38325e5c5c795430b7cb25ae16d1`（2 コミット: `4b796f6f` + `e32e7d05`） | 一致 |
| 4 | 必須 CI | 15 チェック success（pending / failure 0。dispatch 専用の probe のみ skipping） | 一致 |
| 5 | マージ可能性 | `mergeable: MERGEABLE`、`mergeStateStatus: CLEAN` | 一致 |
| 6 | 変更ファイル | `CMakeLists.txt`、`src/tests/TailLayerIntegrityTests.cpp`、`src/tests/TailLayerPerfTests.cpp` の 3 件 | 一致 |
| 7 | 差分 | `+62/0`、`+17/5`、`+300/0` = **379 insertions / 5 deletions** | 一致 |

すべて成立したためマージを実行した。

---

## 2. マージ結果（指示 2.2）

| 項目 | 値 |
| --- | --- |
| PR 状態 | **MERGED**（`mergedAt: 2026-11-14T14:16:56Z`） |
| マージコミット | `783f3d24dbda110827770c66a2108a370698117f` |
| 件名 | `Merge pull request #211 from lonewolf-jp/fix/resf-spec-compliance` |
| `origin/main` | `783f3d24`（リモート main と一致） |
| マージで変わったファイル | `4d476d5c..783f3d24` で **3 ファイルのみ** |
| 差分（マージ後） | `CMakeLists.txt +62/0`、`TailLayerIntegrityTests.cpp +17/5`、`TailLayerPerfTests.cpp +300/0` = 379 / 5 |

### 2.1 main の内容確認

`origin/main` の各ファイルを直接読んで確認した。

| 確認 | 実測 |
| --- | --- |
| `src/tests/TailLayerPerfTests.cpp` の行数 | 300 行（新規ファイルが存在） |
| `CMakeLists.txt` の `TailLayerPerfTests` 参照 | 16 箇所（ターゲット・登録・リンク設定） |
| `TailLayerIntegrityTests.cpp` の `RES-F-cpu`（SKIP 出力） | 1 箇所 |
| 同 `RES-M`（メモリ判定） | 1 箇所 |
| 同 旧 `report ("RES-F", ...)` の CPU 判定 | **0 箇所**（分離済み） |
| 性能テストの仕様準拠マーカー | ウォームアップ除外・`kMinAudioSecPerSample`・`kDisturbSpread` を確認 |

---

## 3. 作業ツリーの状態（指示 2.3・3.6・3.7）

| 項目 | 値 |
| --- | --- |
| 本体 worktree（`C:\VSC_Project\ConvoPeq`）の HEAD | `4d476d5c`（**ローカルは未更新**。理由は下記） |
| 他タスクの未コミット変更 | **38 件を維持**（変更・削除なし） |
| untracked | 141 件（変更なし） |
| staged | **0 件** |
| 破壊的操作 | `reset --hard`・`clean`・他ファイルの checkout は未使用 |

**ローカル main を進めなかった理由**: 他タスクが `CMakeLists.txt` と `src/tests/TailLayerIntegrityTests.cpp` を未コミットで変更しているため、`git merge --ff-only origin/main` は
「ローカルの変更が上書きされる」として拒否された（実行結果: exit 1、HEAD は `4d476d5c` のまま、作業ツリーは無傷）。
他タスクの変更を破棄しないことを優先し、ローカル main は据え置いた。正規の main（`origin/main`）は `783f3d24` で更新済みである。
他タスクの作業をコミットまたは退避した後、`git merge --ff-only origin/main` で追随できる。

---

## 4. main の CI 結果（指示 3.4）

マージコミット `783f3d24` に対して 6 ワークフローが起動し、**すべて success** で完了した。

| ワークフロー | 結果 | 所要 |
| --- | --- | ---: |
| Windows MSVC Build | success | 23.9 分 |
| Push on main（CodeQL: actions / c-cpp / csharp / javascript-typescript） | success | 53.0 分 |
| Sanitizer CI（Debug TSan / RelWithDebInfo ASan） | success | 84.5 分 |
| ISR Authority Compliance v6.4 | success | 0.9 分 |
| audioengine-lint | success | 0.5 分 |
| list-compliance | success | 0.6 分 |

Windows MSVC Build が success したことで、追加した `TailLayerPerfTests` ターゲットが CI のクリーンな環境でも Release 構成でビルドできることを確認した。

---

## 5. 未解決事項

- ローカル main（作業ツリー）は `4d476d5c` のまま。他タスクの未コミット変更を保護するための意図的な据え置きで、`origin/main` は `783f3d24` に更新済み
- 作業用 worktree `C:\VSC_Project\resf-work`（ブランチ `fix/resf-spec-compliance`）は残置。ブランチはマージ済み
- RES-F の CI 性能ゲートは Owner 判断 1 のとおり保留のまま（本マージで必須化していない）
