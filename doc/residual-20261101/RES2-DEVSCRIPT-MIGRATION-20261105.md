# RES2-20 開発スクリプトの移行設計（`.venv` 参照 5 本）

- 作成: 2026-11-05
- 前提: `.venv` は再作成も削除もしない（Owner 判断）。headroom は MCP と CLI のみを使い、proxy は使わない（同日の方針）。
- 位置づけ: 設計のみ。スクリプトの変更・削除は行っていない。

## 1. 対象と現状

| スクリプト | 役割 | 参照している実行ファイル | 現状 |
|---|---|---|---|
| `tools/test_headroom_mcp_tools.py` | MCP の `compress`／`retrieve`／`stats` を順に呼ぶ結合テスト | `.venv\Scripts\headroom.exe` | 起動不可（`.venv` の基準 Python 不在） |
| `tools/find_headroom_process.py` | headroom のプロセス・開いているファイル・8787 の待受を一覧する | `.venv\Scripts\headroom.exe`、`psutil` | 起動不可（パス不正に加え、システム Python 3.15 に `psutil` が無い） |
| `tools/verify_headroom_proxy.py` | proxy の健全性と headroom ライブラリの圧縮を突き合わせる | `.venv\Scripts\headroom.exe`、`.venv\Scripts\python.exe` | proxy 廃止方針により用途が消滅 |
| `tools/headroom-proxy-bench.ps1` | proxy の圧縮率とレイテンシを測る | `.venv\Scripts\headroom.exe`、8787 | proxy 廃止方針により用途が消滅 |
| `tools/start-headroom-proxy.ps1` | proxy の起動・停止・状態表示 | `.venv\Scripts\headroom.exe`、8787 | proxy 廃止方針により用途が消滅 |

### 実測した依存関係

- `C:\Python315\python.exe -c "import psutil"` は失敗する（`psutil` 未導入）。`find_headroom_process.py` は仮にパスを直しても動かない。
- `C:\Users\user\AppData\Roaming\uv\tools\headroom-ai\Scripts\python.exe` は Python 3.14.7 で、`import headroom` が通る。ライブラリとしての検証はここで代替できる。
- 5 本はいずれも CI から参照されていない（`.github` の workflow に記述なし）。開発端末の補助スクリプトである。

## 2. 分類

| 分類 | 対象 | 方針 |
|---|---|---|
| MCP 系（価値が残る） | `test_headroom_mcp_tools.py` | uv ツールの `headroom.exe` を指すよう移行する |
| 診断系（条件付き） | `find_headroom_process.py` | パスを uv ツールへ変え、`psutil` 依存を外す（`Get-CimInstance`／`tasklist` で代替）か、uv ツールの Python で動かす |
| proxy 系（用途消滅） | `verify_headroom_proxy.py`、`headroom-proxy-bench.ps1`、`start-headroom-proxy.ps1` | 移行しない。削除もせず、非推奨として残す（削除は別承認） |

## 3. 移行方式（案）

### 3.1 実行ファイルの解決

優先順を次の 1 行に統一する。環境ごとの差し替えは `HEADROOM_EXE` で行う。

1. 環境変数 `HEADROOM_EXE`
2. `%APPDATA%\uv\tools\headroom-ai\Scripts\headroom.exe`
3. 従来の `.venv\Scripts\headroom.exe`（最終フォールバック）

`test_headroom_mcp_tools.py` は subprocess の引数 1 つを差し替えるだけで済む。呼び出しの順序・待ち時間・出力形式は変えない。

### 3.2 Python の解決

- `verify_headroom_proxy.py` は proxy 廃止で不要になるため、Python の差し替えは行わない。
- ライブラリ検証を残す場合は、`%APPDATA%\uv\tools\headroom-ai\Scripts\python.exe` を使う。システム Python 3.15 には `headroom` も `psutil` も入っていない。

### 3.3 互換性の確認項目

| 項目 | 確認方法 |
|---|---|
| MCP の 3 ツールが呼べる | `test_headroom_mcp_tools.py` を実行し、`compress`／`retrieve`／`stats` の応答が返る |
| 出力形式が変わらない | 標準出力の先頭行と JSON のキーを、変更前後で比較する |
| proxy 非依存 | 8787 が停止した状態で `test_headroom_mcp_tools.py` が成功する |
| CI 非依存 | `.github` に参照が無いことを再確認する |
| 端末差 | `HEADROOM_EXE` 未設定でも uv ツールがあれば動く |

## 4. 実装結果（2026-11-05）

Owner の承認（移行実装）を受けて実施した。`.venv` の再作成・削除は行っていない。

| 対象 | 処置 | 確認 |
|---|---|---|
| `test_headroom_mcp_tools.py` | `HEADROOM_EXE` → uv ツール → `.venv` の順で解決するよう変更 | 実行して exit 0。`compress` と `stats` が応答 |
| `find_headroom_process.py` | `psutil` を廃止し、CIM（PowerShell）でプロセスを取得。自分自身と問い合わせプロセスを除外 | 実行して MCP 4 件を検出、8787 は CLOSED、exit 0 |
| `verify_headroom_proxy.py` | 非推奨ヘッダを追加（削除はしない） | — |
| `headroom-proxy-bench.ps1` | 非推奨ヘッダを追加 | — |
| `start-headroom-proxy.ps1` | 非推奨ヘッダを追加 | — |

- プロセス検出の精度: 実行ファイル名またはコマンドラインに `headroom.exe` を含むものだけを対象にした。これにより、`headroom` の語を含むだけの PowerShell や実行中のスクリプト自身を誤検出しない。
- 失った情報: 開いているファイルハンドルの一覧（CIM では取得できない）。プロセス名・PID・コマンドライン・8787 の疎通は維持している。

## 5. 実装手順（当初案）

1. `test_headroom_mcp_tools.py` の実行ファイル解決を 3.1 の順に変える。
2. `find_headroom_process.py` を `psutil` 非依存へ書き換えるか、uv ツールの Python で実行する形にする。
3. proxy 系 3 本の先頭に「非推奨（proxy 廃止方針）」のコメントを足す。削除はしない。
4. 8787 停止状態で `test_headroom_mcp_tools.py` を実行し、3 ツールの応答を確認する。
5. 変更は 1 コミットにまとめ、CI（doc 以外を含むため）の pending=0・fail=0 を確認する。

## 5. リスクと判断が必要な点

- `find_headroom_process.py` の書き換えは、`psutil` の代替（CIM）で取得できる情報が減る。プロセス名と PID だけに絞るなら同等。
- proxy 系 3 本を残すと、誤って起動すると方針違反になる。先頭の非推奨コメントだけでは不十分なら、`tools/archive/` への移動を検討する（移動も別承認）。
- スクリプトの変更は製品コードではないが、リポジトリの差分になる。実装の可否はこの設計を確認したうえで決める。
