# Tool Inventory — Cline Desktop 環境 (2026-09-20 確認)

## ✅ 確認済み・使用可能

### WSL ツール (rtk 経由推奨: `wsl bash -lc "rtk <cmd>"`)
| ツール | パス | バージョン |
|---|---|---|
| grep | /usr/bin/grep | GNU 3.12 |
| rg (ripgrep) | /usr/bin/rg | 15.1.0 |
| ast-grep (sg) | /usr/local/bin/ast-grep | 0.44.0 |
| fdfind (fd) | /usr/bin/fdfind | 10.3.0 |
| ag | /usr/bin/ag | 2.2.0 |
| fzf | /usr/bin/fzf | 0.67.0 |
| sed / awk | /usr/bin/ | OK |
| rtk | ~/.local/bin/rtk | 0.49.0 (CLI出力圧縮) |

### Windows CLI
| ツール | パス | 状態 |
|---|---|---|
| serena | `C:\Users\user\.local\bin\serena.exe` | v1.7.0 / MCP 設定済み |
| ccc (cocoindex-code) | `C:\Users\user\.local\bin\ccc.exe` | 動作確認済。`ccc search "<query>" --limit N`、`ccc status`。ConvoPeq インデックス済 (305k chunks, 2581 files) |
| graphify | `...\Python314\Scripts\graphify.exe` | graphifyy 0.9.64 / graphify-out/graph.json 既存 (61029 nodes)。`graphify query "<node>"` |
| semble | `C:\Users\user\.local\bin\semble.exe` | 動作確認済。`semble search "<query>" <path> -k 5 --format text` |
| cppcheck | `C:\Program Files\Cppcheck\cppcheck.exe` | 2.21.0 |
| clang-tidy | `C:\Program Files\LLVM\bin\clang-tidy.exe` | LLVM 23.1.1 |
| Dr. Memory | `C:\Program Files (x86)\Dr. Memory\bin\drmemory.exe` | 2.6.20434 |
| tgrep | `C:\Windows\system32\tgrep.exe` | OK (trigram-indexed grep) |
| GNU Octave | `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe` | 11.3.0 |
| Maxima | `C:\maxima-5.50.0\bin\maxima.bat` | 5.50.0 |

### Python ライブラリ (Python 3.14)
NumPy 2.5.2 / SciPy 1.18.0 / Matplotlib / pandas / IPython / crawl4ai / ddgs / trafilatura 2.2.0 — すべて import OK。

### MCP (Cline Desktop: `C:\Users\user\.cline\data\settings\cline_mcp_settings.json` に設定済)
serena, headroom, context-mode, context7, firecrawl, brave-search, github(remote: api.githubcopilot.com/mcp/), microsoft-learn(remote: learn.microsoft.com/api/mcp), aidex
- API キー環境変数: CONTEXT7_API_KEY / FIRECRAWL_API_KEY / BRAVE_API_KEY / GH_TOKEN すべて設定済
- **Cline 再起動で有効化される**
- AiDex インデックス: `.aidex/index.db` (486 files / 20491 items) 作成済

### トークン削減3層 (常時使用)
1. **headroom** v0.37.0 — proxy port 8787 稼働中 (ANTHROPIC_BASE_URL 設定済)、`headroom mcp serve` を MCP 登録済
2. **context-mode** v1.0.169 — doctor PASS、MCP 登録済
3. **rtk (WSL)** 0.49.0 — CLI 出力圧縮に使用

役割分担: ファイル分析→context-mode、トラフィック自動圧縮→headroom proxy、CLI出所圧縮→rtk。ダブル圧縮回避。

## ⚠️ 注意
- **Obscura**: GitHub 版は Rust 製ヘッドレスブラウザ。Rust ツールチェインはあるがビルドは重量級のため未インストール。pip の `obscura` 0.1.1 (無関係な暗号化パッケージ) を誤インストールしたが害はない。必要時 `cargo install` またはリリースバイナリで対応可能。
