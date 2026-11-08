# RES2-20 移行実装と RES2-12 初回 soak の結果

- 実施: 2026-11-05 23:30–2026-11-06 00:10（Windows 11 x64）
- 承認範囲: RES2-20 の限定移行（2 本）と 3 本の廃止、RES2-12 の初回手動 `all`、RES2-11 の計測、RES2-13 の再定義案
- 製品コード・既定値・RT 経路は変更していない。`.venv` の再作成・削除も行っていない。

## 1. RES2-20: 開発スクリプトの移行

| 対象 | 処置 | 確認 |
|---|---|---|
| `test_headroom_mcp_tools.py` | `HEADROOM_EXE` → uv ツール → `.venv` の順で解決 | 実行して exit 0。`compress`・`stats` が応答（uv ツールを解決） |
| `find_headroom_process.py` | `psutil` を廃止し CIM で取得。自分自身と問い合わせプロセスを除外 | MCP 4 件を検出、8787 は CLOSED、誤検出なし、exit 0 |
| `verify_headroom_proxy.py` | 非推奨ヘッダ（削除しない） | — |
| `headroom-proxy-bench.ps1` | 非推奨ヘッダ | — |
| `start-headroom-proxy.ps1` | 非推奨ヘッダ | — |

- 検出精度: 実行ファイル名またはコマンドラインに `headroom.exe` を含むものだけを対象にした。`headroom` の語を含むだけの PowerShell やスクリプト自身は除外する。
- 失った情報: 開いているファイルハンドルの一覧（CIM では取得不能）。プロセス名・PID・コマンドライン・8787 の疎通は維持。
- 構文: `py_compile` 成功（2 本）。

## 2. RES2-12: 初回手動 soak（`all`）

### CI での実行

- `workflow_dispatch` で `scenario=all` を実行（run 42044137638）。
- 結果は success・0.6 分。ただしログは **skip** である。

> Soak CI skipped
> - Reason: Intel oneAPI not available on this runner.
> - MKLNonUniformConvolver requires mkl.h/ipp.h (Intel oneAPI).

- GitHub の `windows-latest` には oneAPI が無いため、CI では soak がビルドできない。**週次化しても実測・回帰検出にならない。**

### ローカルでの実行

oneAPI のある開発機で `AudioEngineHarness.exe --soak --scenario=all` を実行した。

| シナリオ | 結果 |
|---|---|
| s1 | published 100,000 / accepted 100,000 / rejected 0 / finalSeq 100,003 / backlog 0 |
| s2b | requested 80,000 / accepted 80,000 / rejected 0 / peakBacklog 0 |
| s3 | rapid 300 / drained_to_seq 180,303 / backlog 0 |
| s4 | published 10,001 / epoch 360,606 → 380,608 / pending 0 |
| s5 | samples 158 / peak 287.4 MB / final 287.4 MB（収束） |
| 合計 | issued 180,300 / accepted 180,300 / rejected 0 / peakBacklog 0 / **ALL PASS** |

- 実行時間: **799.9 秒（13.3 分）**。60 分以内の条件を満たす。
- メモリ: 287.4 MB で収束し、増加を続けない。

## 3. RES2-11: TSan の実測

| 項目 | 結果 |
|---|---|
| 直近 30 run | success 26、未完了 4 |
| 完了 10 run の TSan ジョブ時間 | 0.4〜1.1 分 |
| ログ | `[TSan] Skipped (no build configured — MKL/IPP not available on Linux)` |
| 実行された TSan テスト | 0（標本すべて skip） |

- `success 26` は TSan の実行を意味しない。skip 率は標本で 100%。C2（skip 0）は未充足である。
- 設計資料に対象テスト案（現行 1＋候補 5）と、失敗時の扱い・安定性基準を追記した。

## 4. RES2-13: 再定義案

- 旧 TX-02／TX-10 の要求を現行アーキテクチャ（`RuntimeWorld`）へ対応付けた。
- 順序・回数・最終値・retire は自動テストで確認できる。実機でしか見えないのは XRUN と可聴クリックである。
- publish 回数・最終値・rebuild 回数の数値基準案を提示した。判断は未確定のまま（再定義の承認待ち）。

## 5. TOOL-03: `opencode.json`

- 旧 context-mode MCP 定義（重複）を削除。差分は 12 行削除のみで、整形は変えていない。
- 平文の秘密情報なし（API キーは `{env:...}` 参照）。JSON 妥当、サーバー 11 件。

## 6. 検証と PR

| 項目 | 結果 |
|---|---|
| `py_compile` | 2 本とも成功 |
| MCP 結合テスト | exit 0 |
| 診断 | MCP 4 件、8787 CLOSED、exit 0 |
| `opencode.json` | JSON 妥当、context-mode なし |
| PR | #142（`e50e7328`）。ASan を含む 12 チェックが pass、fail 0 |
| 統合 | 完了。squash merge で `7f4997f6`。統合後に MCP 結合テスト（exit 0）と診断（8787 CLOSED）を再実行 |

## 7. 未実施・要判断

- TSan の必須化（C1〜C5 未充足。skip を解消するには Linux で configure が通る対象の固定が必要）。
- soak の週次化（CI ランナーに oneAPI が無いため、現状の週次実行は skip になる）。
- TX-02／TX-10 の再定義の承認、実機検証、免除の確定。
- v1 context-mode データ・旧キャッシュ・rtk 履歴の削除（TOOL-02 により保留）。
