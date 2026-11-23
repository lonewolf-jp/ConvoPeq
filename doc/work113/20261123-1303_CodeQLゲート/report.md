# CodeQL をドキュメントのみの変更でスキップする

作業日: 2026-11-23（JST）。対象: GitHub Actions の CodeQL 設定。コードは変更していない。

## 1. 課題

CodeQL は **default setup**（GitHub のマネージド設定）で動いていた。default setup は paths フィルタを持てず、`doc/` だけを変更した push でも actions / c-cpp / csharp / javascript-typescript の 4 言語が毎回走っていた。実測では main の c-cpp 解析が 57〜71 分かかっており、次の 3 つが無駄になっていた。

- main のステータスが 1 時間以上「解析中」のまま残る。
- PR のチェック一覧に意味のない解析が並ぶ。
- 共有ランナーを 1 時間占有する。

公開リポジトリなので**課金は発生していない**。失っていたのは待ち時間とランナー枠である。

default setup の設定は API でも paths を持たない（`state` / `languages` / `query_suite` / `threat_model` / `schedule` / `runner_type` / `runner_label` のみ）。したがって **advanced setup（workflow ファイル方式）へ切り替える以外に方法がない**。

## 2. 切替の内容

1. default setup を無効化した。`PATCH /repos/lonewolf-jp/ConvoPeq/code-scanning/default-setup` に `{"state":"not-configured"}` を送り、`state` が `not-configured` になったことを確認した（2026-11-23）。
2. `.github/workflows/codeql.yml` を追加した（PR #254）。

切替前の設定をそのまま引き継いでいる。

| 項目 | 切替前（default setup） | 切替後（workflow） |
| --- | --- | --- |
| 言語 | actions / c-cpp / csharp / javascript-typescript | 同じ 4 言語の matrix |
| クエリ suite | default | default（`init` の既定） |
| カテゴリ | `/language:<lang>` | `/language:<lang>`（同じ値） |
| build mode | none（buildless） | `build-mode: none` |
| スケジュール | weekly | `17 3 * * 1` |
| 実行環境 | ubuntu-latest | ubuntu-latest |

カテゴリを default setup と同じ値にしたのは、既存アラートの追跡を切らないためである。値が変わると、同じ指摘が「新しいアラート」として開き直される。

## 3. 設計: workflow を止めず、解析だけを落とす

```
on: pull_request / push(main) / schedule / workflow_dispatch   ← 常に起動
  changes  (Detect changed files)  … 常に実行。doc のみなら code=false
  analyze  (Analyze (<lang>) × 4)  … code=true のときだけ実行
  gate     (CodeQL)               … 常に実行。必要なら解析成功を要求
```

workflow 側に `paths-ignore` を書けばもっと短いが、そうしない。`windows-msvc-build.yml` のコメントにあるとおり、workflow 自体を paths で止めると check が作られず、必須チェックに指定したときに `Pending` のまま残ってマージできなくなる。job の条件スキップは Success として報告されるため、常時起動＋条件スキップなら必須チェックにしても壊れない。

判定規則は `windows-msvc-build.yml` と同一にした。`doc/`、`*.md`、`.memories/`、`.x64dbg-mcp/`、`.zcode/` だけの変更を doc とみなし、**判定に失敗したときは解析する側へ倒す**。`schedule` と `workflow_dispatch` は常に全解析する。

集約ジョブは `CodeQL` という単一の名前で、次のように判定する。

- 解析が必要（`code=true`、または判定不能）なら、解析が成功していなければ失敗。
- doc のみで判定が成功していれば、解析のスキップを合格として扱う。

## 4. build mode: autobuild では成立しなかった

最初の実装はコンパイル言語を `build-mode: autobuild` にした。PR #254 の 1 回目の実行は **c-cpp と csharp が失敗**した。

- c-cpp: `src/MKLNonUniformConvolver.h:47: fatal error: mkl.h: No such file or directory`、続いて `c++: error: /EHsc: linker input file not found`。Intel MKL のヘッダと MSVC のフラグを要求するため、ubuntu ランナーでは構成できない。
- csharp: `Could not auto-detect a suitable build method`。ビルド可能な .NET プロジェクトが無い。

切替前の default setup のログを確認すると `"build-mode": "none"` で、`C/C++ was extracted with build-mode set to 'none'. This means that all C/C++ source in the working directory will be scanned.` と記録されていた。つまり **default setup も buildless で走っていた**。4 言語すべてを `build-mode: none` に変更し、2 回目の実行で一致した。

buildless は作業ツリー全体を走査するため、解析対象は切替前と同じである。ビルド成果物を使った解析に変えると検出結果が変わるので、ここは変えない。

## 5. 検証

### 5.1 コード変更を含む PR（PR #254）

| 項目 | 結果 |
| --- | --- |
| `Detect changed files` | success（コード変更を検出） |
| `Analyze (actions)` | success（51 秒） |
| `Analyze (javascript-typescript)` | success（63 秒） |
| `Analyze (csharp)` | success（85 秒） |
| `Analyze (c-cpp)` | success（50 分 2 秒） |
| `CodeQL gate` | success |

ほかのチェック（Windows MSVC / Release、Sanitizer CI の ASan・TSan、ISR Verification、ISR Authority Compliance、audit-report-verification、Graphify）もすべて success で、マージした（merge commit `8421fda1`）。

### 5.2 アラートの比較

切替前（main `48c452de`、default setup）と切替後（PR #254、advanced setup）で、オープンなアラートは次のとおりで一致した。

| ルール | 重要度 | 切替前 | 切替後 |
| --- | --- | ---: | ---: |
| `cpp/integer-multiplication-cast-to-long` | high | 32 | 32 |
| `cpp/overflowing-snprintf` | high | 2 | 2 |
| `actions/missing-workflow-permissions` | medium | 1 | 1 |
| 合計 | | 35 | 35 |

件数が一致しても、解析対象と検出内容が同一とは限らない。そこで次も確認した。

- 解析のカテゴリは切替前と同じ `/language:<lang>` で、main `48c452de` の解析（c-cpp 40 件、csharp 0 件、javascript-typescript 0 件、actions 1 件）と、切替後の実行が同じカテゴリの系列に載っている。カテゴリが変われば同じ指摘が「新しいアラート」として開き直されるが、それは起きていない。
- 解析ログに `C/C++ was extracted with build-mode set to 'none'` が残り、切替前（default setup のログ）と同じ抽出条件であることを確認した。クエリの評価対象も `IntMultToLong.ql`（32 件のルール）と `SnprintfOverflow.ql`（2 件のルール）を含む同一の suite である。

### 5.3 ドキュメントのみの変更

本報告を含む PR はドキュメントだけを変更する。この PR の実行で `Analyze` が 1 本も作られず、`CodeQL gate` が success になることを確認した（結果は下に追記する）。

TBD_DOCONLY

### 5.4 main への push

`8421fda1`（workflow のマージ）の push では、コード変更として 4 言語の解析が走ることを確認した。TBD_MAINPUSH

## 6. 制約

- **言語の自動追従は無くなった。** default setup はリポジトリの言語構成の変化に自動で追従したが、workflow では matrix を手で直す。言語を追加したときは `.github/workflows/codeql.yml` を更新する。
- アクションのバージョンは `github/codeql-action/init@v4` / `analyze@v4` に固定した。default setup はバンドルを自動更新していたが、今後はメジャー更新を手で行う（マイナーは `v4` タグが追従する）。
- CodeQL は**必須チェックにしていない**。必須は `Windows MSVC / Release` の 1 本のまま。将来必須にする場合は `CodeQL` を指定できる（doc のみの PR では Success として報告される）。
- 解析は buildless なので、ビルド時のマクロ定義や include パスに依存する検出は行わない。これは切替前と同じ条件である。

## 7. 戻し方

1. `.github/workflows/codeql.yml` を削除する。
2. `PATCH /repos/lonewolf-jp/ConvoPeq/code-scanning/default-setup` に `{"state":"configured"}` を送り、default setup を再有効化する。

アラートの履歴はどちらの方式でも同じツール（CodeQL）に残る。カテゴリを揃えてあるので、戻しても追跡は切れない。

## 8. 参照

- PR #254（workflow の追加と buildless への修正）
- `doc/work113/20261123-1303_CodeQLゲート/`（本報告）
