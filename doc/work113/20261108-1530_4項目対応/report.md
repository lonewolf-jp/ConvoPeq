# build.bat のアプリ限定ビルド、CI の文書スキップ、ツール更新、フォルダ整理

- 実施: 2026-11-08 15:00〜
- 対象: 開始時点の `main` = `969a988e`。第 1〜6 節は PR #180 でマージ済み（`main` = `f2995a44`）。第 7 節と CI の修正は PR #181、第 8 節は PR #182 で扱う。
- 目的: 次の 4 件を実施する。(1) `build.bat` にアプリ限定ビルドのオプションを追加、(2) 文書だけの変更で GitHub Actions をスキップ、(3) context-mode と graphify の更新と学習、(4) プロジェクト内フォルダの扱いの整理。

## 1. `build.bat` のアプリ限定ビルド（`app`）

### 変更

- 引数 `app` を追加した。`ConvoPeq` ターゲットだけをビルドし、テストターゲットと Phase 8 の追加ビルドを省略する。
- 使い方の行と説明を更新した（`build.bat [Debug|Release] [clean] [nopause] [pgo-gen|pgo-use] [icx|icpx] [app] [-DVAR]`）。
- `TARGET_MODE` を追加し、起動時に表示する。
- `app` のときは `cmake --build ... --target ConvoPeq` を使う。

### 実測

| コンパイラ | コマンド | ターゲット | 結果 | 実行ファイル |
|---|---|---|---|---|
| icx | `build.bat Release icx app clean nopause` | 139（通常は 678） | exit 0 | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`（48,434,176 B、15:28:34） |
| MSVC | `build.bat Release app clean nopause` | 139 | exit 0 | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`（47,606,784 B、15:33:57） |

- どちらも `Building Phase 8 test targets` は実行されていない。
- 所要は icx が約 8.9 分、MSVC が約 3.2 分（どちらもクリーンから、ログの作成から最終書き込みまで）。同じ環境のフルビルド（678 ターゲット）は icx で約 39 分だった。icx はメモリ対策で `-j 2` 固定のため、削減率は目標数ほど大きくはない。

### 注意

- `build.bat` はビルド同一性ゲートを持つ。コミットやソース変更の後は stamp と一致せず、`clean` を付けないと拒否される（設計どおり）。アプリ限定でも `clean` が必要な場面がある。
- MSVC モードは `vcvars64.bat` を済ませた環境で実行する必要がある（`build.bat` は MSVC では vcvars を呼ばない）。icx モードは `setvars.bat` を内部で呼ぶため不要。

## 2. 文書だけの変更で CI をスキップ

### 仕組み

- 必須チェック `Windows MSVC / Release` は workflow 自体を `paths` で止めると Pending のままになりマージがブロックされる。一方、**job を条件でスキップすると "Success" として報告される**（GitHub の仕様）。この違いを利用した。
- `windows-msvc-build.yml` に `changes` ジョブを追加し、変更ファイルを判定して `code` を出力する。`release` ジョブは `needs: changes` と次の条件を付けた。
  - `if: always() && (needs.changes.result != 'success' || needs.changes.outputs.code == 'true')`
  - 判定に失敗した場合はビルドする側に倒す（安全側）。
- 変更の取得方法は event で分ける。
  - `pull_request`: `origin/<base>..HEAD`。checkout されるのはマージコミットなので、base の先端との 2 点リビジョンが PR の変更そのものになる。3 点リビジョンはマージベースを必要とし、浅いクローンでは答えられないことがあるため使わない。
  - `push`: `before..sha`。マージコミットの push なら `before` は `HEAD^1` で、`fetch-depth: 2` に含まれる。含まれない場合は `git fetch` を試し、取れなければビルドする。
  - それ以外の event と、差分の取得に失敗した場合はビルドする。
- 文書とみなすパターン: `doc/`、`*.md`、`.memories/`、`.x64dbg-mcp/`、`.zcode/`。
- パスの取得は `git -c core.quotepath=false diff --name-only -z` と `tr '\0' '\n'` を使う。既定の `git diff --name-only` は非 ASCII のパスを引用符で囲むため、日本語のディレクトリ名を持つこのリポジトリでは文書を文書と判定できない（第 7 節）。
- 必須ではない `sanitizer-ci.yml` と `isr-authority-compliance.yml` には `push` と `pull_request` の両方、`isr-verification.yml` には `pull_request` に `paths-ignore` を追加した（workflow ごとスキップ）。main へ文書だけをマージした場合も重い処理が動かない。

### 実測

- 文書だけの PR を作成して確認した。1 回目は判定を誤ってビルドが走ったため、判定の取得方法を修正した（第 7 節）。修正後の確認結果は第 8 節に記す。

## 3. ツールの更新

### context-mode 2.1.4 → 2.1.6

- 更新: `npm install -g context-mode@latest`。
- 確認: `context-mode --version` が `2.1.6`。MCP ハンドシェイクは 9 ツールで成功。
- 変更点（リリースノート）:
  - 2.1.5: doctor の Codex プラグイン一覧の読み取り、Stop フックの空応答、Read サイズ制限で画像と PDF を除外、各ホストの応答スキーマに合わせたフック検証。
  - 2.1.6: **OpenCode の `ctx_*` ツールが OpenCode の権限ルールを通る**、コンパクト後の再呼び出しが現在のプロンプトだけを対象にする、スキルとルールの全文がモデルに届く、`CTX_REPLACE_TRIGGER_BYTES` 設定、Node の版チェック、Codex の `CODEX_HOME`、ZCode の 16 KB 超画像、Issue 取り込みの型ラベル。

### graphify 0.9.103 → 0.9.104

- 更新: `uv tool install "graphifyy[mcp]" --force`。実行中の `graphify-mcp.exe` が `Scripts` を掴むため、先に停止した。
- スキル: `graphify install --platform opencode` と `--platform copilot` で同期した。
- 確認: `graphify --version` が 0.9.104。MCP ハンドシェイクは 10 ツールで成功。
- 変更点（リリースノート）:
  - **セキュリティ**: `/graphify` の手順がスキャン対象パスを Python のコードへ埋め込まなくなり、信頼済みの `graphify-out/.graphify_root` から読む。ディレクトリ名に引用符や `$(...)` が含まれる場合のコード実行を塞いだ（#4762）。
  - Solidity: コントラクト型のレシーバ、`using A for T`、`_` 接頭辞だけが異なる同名シンボルの解決。
  - C#: コンストラクタ（インスタンス・静的・プライマリ）と `: base(...)` / `: this(...)` の解決、プロパティ・インデクサ・演算子本体の呼び出しの帰属。
  - query: 同一ラベルの唯一の候補がシードから落ちないようにした。
  - watch: inotify のマスクから読み取り専用イベントを外し、キュー溢れの自己増殖を防ぐ。`tsconfig.json` などの解決入力の変更で全体を再構築。`graph.json` が無いときの部分グラフ書き込みを修正。
  - cache: `\\?\` の長いパス表記を同一視し、UNC 共有の空 stat 書き込み、ジャンクションの別名の誤共有を修正（Windows）。
  - YAML: CloudFormation の `!Ref` などを、評価せずに解析する。
  - ingest: 同じ URL の同時取り込みで衝突しない。
  - AST キャッシュのスキーマを 40 に更新。

### 反映に必要な操作

- 稼働中のアプリは起動時の MCP プロセスを使い続ける。context-mode 2.1.6 と graphify 0.9.104 をセッションに反映するには OpenChamber（OpenCode サーバー）の再起動が必要。
- OpenCode の `plugins` に書かれた `context-mode@2.1.6` はサーバーが読み込んでいる（`opencode.log` に `loading plugin id=context-mode@2.1.6`、失敗の記録なし）。更新が必要だったのは npm global の MCP サーバー側で、プラグインではない。
- 更新のため graphify の MCP プロセスを停止した。再起動までは graphify の MCP ツールは利用できない。

## 4. プロジェクトフォルダの整理

| フォルダ | GitHub レポジトリ | ローカル | 実施 |
|---|---|---|---|
| `.memories/repo` | 削除 | 残置 | `git rm -r --cached .memories`、`.gitignore` に `.memories/` を追加 |
| `.x64dbg-mcp` | 削除 | 残置 | `git rm -r --cached .x64dbg-mcp`（`.gitignore` は既存） |
| `.zcode` | 削除 | 削除 | `git rm -r .zcode` とローカルの削除 |

- `.memories/repo` のローカルファイルは 1 件、`.x64dbg-mcp` は 2 件で、いずれも残置を確認した。
- `.zcode` はローカルにも存在しない。

## 5. 変更ファイル

| 種別 | ファイル |
|---|---|
| ビルド | `build.bat`、`BUILD_GUIDE_WINDOWS.md`（`app` オプションの追記） |
| CI | `.github/workflows/windows-msvc-build.yml`、`sanitizer-ci.yml`、`isr-authority-compliance.yml`、`isr-verification.yml` |
| 整理 | `.gitignore`、`.memories/repo/...`（追跡解除）、`.x64dbg-mcp/*`（追跡解除）、`.zcode/config.json`（削除） |
| 報告 | 本ファイル |

## 6. 未実施・注意

- ツール更新のセッションへの反映は再起動後。
- soak、TX 実機、TSan の実行、#10・#11 の文書修正は行っていない。

## 7. 文書のみ PR でのスキップ確認（1 回目は判定を誤った）

### 1 回目の結果

文書だけを変更した PR #181 で確認した。`Detect changed files` は pass した。しかし `Windows MSVC / Release` はスキップされなかった。ビルドが始まった。

原因は、変更ファイルの取得方法にある。`git diff --name-only` は既定（`core.quotepath=true`）で、非 ASCII のパスを引用符と 8 進エスケープで囲む。

```
"doc/work113/20261108-1530_4\351\240\205\347\233\256\345\257\276\345\277\234/report.md"
```

この文字列は先頭が `"` で末尾も `"` だ。`^doc/` にも `\.md$` にも一致しない。誤判定である。日本語のディレクトリ名を使うこのリポジトリでは、文書だけの PR がすべてビルド対象になってしまう。

### 修正

`git -c core.quotepath=false diff --name-only -z` に変え、NUL 区切りを `tr '\0' '\n'` で 1 行 1 パスに戻した。`-z` はパスを引用しないため、非 ASCII のパスも `doc/.../report.md` のまま届く。

## 8. 文書のみ PR でのスキップ確認（修正後）

修正を main に取り込んだ後、文書だけを変更した PR #182 で確認した。

| 確認 | 結果 |
|---|---|
| Detect changed files | pass（11 秒） |
| Windows MSVC / Release（必須） | skipping（スキップ。マージを妨げない） |
| Windows MSVC Build 全体 | success（実行時間は 14 秒） |
| Sanitizer CI | 起動しない |
| ISR Authority Compliance | 起動しない |
| ISR Verification | 起動しない |
| CodeQL | 起動しない |
| udioengine-lint / list-compliance | 起動しない（paths が src/** などのため） |

- gh pr checks はスキップした必須チェックを skipping と表示する。必須チェックとしては成功として扱われ、マージはブロックされない。
- 起動したワークフローは Windows MSVC Build の 1 本だけで、そのうち実行されたのは変更判定のジョブだけである。ビルドもテストも動かない。
- 文書だけの PR の CI はこれで 14 秒で終わる。以前は同じ PR で約 40 分のビルドが走っていた。

### 8.1 main へのマージ後（push）

文書だけの PR #182 を main にマージした直後の push でも、同じ判定が働いた。

| 確認 | 結果 |
|---|---|
| Detect changed files | success（11 秒、efore..sha の判定） |
| Windows MSVC / Release | skipped |
| Sanitizer CI | 起動しない |
| ISR Authority Compliance | 起動しない |
| ISR Verification | 起動しない |
| CodeQL | 起動する（変更判定を持たないワークフロー） |

- 参考として、直前のマージ c4f50ef はワークフロー自体を変更していたため、push でも約 40 分のビルドが走った。差分が文書だけなら 11 秒で終わる。
