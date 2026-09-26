# Tool Inventory — OpenChamber / OpenCode V2.0 (ConvoPeq)

**Verified:** 2026-09-26 · **Host:** Windows + WSL · **Repo:** `C:\VSC_Project\ConvoPeq` (CMake + Ninja Multi-Config + MSVC 14.51, JUCE C++ audio app)

Every item below was **executed**, not just path-checked. Where a tool does not work,
the exact failure is recorded and a working alternative is given.

---

## 1. Executive summary

| Layer | Status | Note |
|---|---|---|
| WSL CLI toolchain (grep/ast-grep/rg/fdfind/ag/fzf/sed/awk) | **9/9 working** | no changes needed |
| Windows CLI (ccc, graphify, semble, cppcheck, clang-tidy, tgrep) | **7/7 working** | no changes needed |
| Python libs (numpy/scipy/matplotlib/pandas/IPython/crawl4ai/trafilatura/ddgs) | **7/7 working** | — |
| Octave / Maxima | **both working** | Maxima needs `-batch <fwd-slash-path>` or stdin; `--eval` does not exist |
| Dr. Memory | **BROKEN — host blocked** | DynamoRIO cannot inject; use ASan instead |
| MCP servers | **10 → 15** | 5 were installed but unregistered; now registered |
| 3-layer token pipeline | **2 of 3 layers real** | **headroom proxy does NOT compress OpenChamber traffic** |

### CORRECTIONS (measured 2026-09-26 after a restart — read before trusting anything below)

1. **`cwd` / `enabled` / `timeout` are NOT valid keys** in this build's `mcp.servers.<name>`.
   Adding any one of them makes the whole entry get dropped, logged as
   `kind=invalid action="skipped malformed recognized value"`. Valid keys are `type`, `command`,
   `disabled`, `environment`, `env`, `url`, `headers`. Enable with `"disabled": false`.
   The official opencode.ai docs list `cwd`/`enabled`/`timeout`, but this implementation
   silently discards the entry instead of using them.
2. **Console windows are now suppressed.** OpenChamber is a GUI (Electron) app and spawns MCP
   servers without `CREATE_NO_WINDOW`, so every console-subsystem binary and every `.cmd` shim
   flashed a console window, repeating on each reconnect. All 13 local servers are now launched
   through `C:\Users\user\.local\bin\mcp\nowin.py` under `pythonw.exe` with
   `creationflags=CREATE_NO_WINDOW` (0x08000000), piping fd 0/1/2 through unchanged. Verified
   13/13 handshake through the launcher. Remote servers (context7, mslearn) spawn no process and
   were left alone. Rollback: each config has a `.bak-before-nowin` copy.
3. **The plugin contract is `export default { id, setup }` (or `{ id, effect }`).** The official
   named-export form (`export const X: Plugin = async (ctx) => ({...})`) is rejected 100% of the
   time with `PluginModule.LoadError: ... SchemaError(Missing key at ["default"])`. So the repo's
   pre-existing `rtk-wsl` plugin was already correct — earlier text in this document called it V1
   dead code, and that was wrong. Vendor `context-mode` v1.0.169 hits the same error, so its
   plugin cannot load here: **context-mode's automatic Read/Shell routing does not work in this
   environment.** Use the `ctx_*` tools manually through MCP.
4. **`opencode mcp list` is useless here** — it always reports "No MCP servers configured"
   regardless of what is actually loaded. The only reliable source of truth is
   `C:\Users\user\.local\share\opencode\log\opencode.log`, which logs
   `message="mcp connected" server=<name> tools=<n>` for each server that really starts.
5. **No rtk auto-rewrite plugin is installed, deliberately.** An auto-rewriter has to spawn a
   child process per shell command, and that is precisely what flashes a console window on every
   command. Use an explicit `rtk <cmd>` prefix instead; WSL `~/.bashrc` also aliases
   `ls`/`grep`/`cat`/`find`/`diff`/`status` to rtk.

### Two findings that change standing instructions

1. **The headroom proxy contributes zero compression here.** OpenChamber/OpenCode does
   not route model traffic through `ANTHROPIC_BASE_URL`. Measured: `headroom savings`
   Today `0/0 tokens`, `/stats` → `api_requests: 0`. The proxy *is* running and healthy
   on `127.0.0.1:8787`, but nothing flows through it. Do not report it as active.
2. **context-mode was installed but completely inert** — not registered as an MCP server
   *and* not registered as a plugin. The MCP server is now registered and works; the
   plugin still cannot load (see CORRECTION 3), so routing is manual, not automatic.
---

## 2. MCP servers (15)

Registered in `C:\VSC_Project\ConvoPeq\opencode.json` (12) and
`C:\Users\user\.config\opencode\opencode.jsonc` (3). All verified by real JSON-RPC
`initialize` + `tools/list` handshake via `scripts`-style probe.

### 2.1 Newly registered (were installed but NOT wired up)

| Server | Launch | Tools | cwd |
|---|---|---|---|
| **context-mode** | `C:/Users/user/AppData/Roaming/npm/context-mode.cmd` | **11** | repo root |
| **obscura** | `C:/Users/user/.local/bin/obscura/obscura.exe mcp` | **37** | repo root |
| **ddgs** | `uvx --from "ddgs[mcp]" ddgs mcp` | **6** | — |
| **graphify** | `.../Python314/Scripts/graphify-mcp.exe --transport stdio` | **10** | repo root |
| **cocoindex** | `C:/Users/user/.local/bin/ccc.exe mcp` | **1** | repo root |

Notes:
- **ddgs must go through `uvx`.** `pip install ddgs[mcp]` into the shared site-packages
  pulls `mcp>=2.0`, which breaks `headroom` and `github-mcp-server` (both need `mcp<2`).
  `uvx` gives it an isolated venv. Keep `mcp<2` pinned in the shared site.
- **obscura**: use `~/.local/bin/obscura/obscura.exe` (the headless browser).
  `~/.local/bin/obscura.exe` is an unrelated argon2 tool. Docker variant not used.
- **CORRECTION 1: `cwd` is NOT supported and breaks the entry.** cwd dependence is
  handled by putting absolute paths directly in `command` instead (see CORRECTIONS).

### 2.2 context-mode — the 11 tools

`ctx_execute` · `ctx_execute_file` · `ctx_index` · `ctx_search` ·
`ctx_fetch_and_index` · `ctx_batch_execute` · `ctx_stats` · `ctx_doctor` ·
`ctx_upgrade` · `ctx_purge` · `ctx_insight`

> The upstream README table lists only 10 — it omits `ctx_insight`. Verified 11 by handshake.

- `ctx_execute_file` is **confined to the project root**; absolute paths, `../`, and
  escaping symlinks are rejected (`File access blocked`).
- Progressive throttle is built in: `ctx_search` calls 1–3 normal, 4–8 reduced,
  **9+ blocked** and redirected to `ctx_batch_execute`.
- Storage: `C:\Users\user\AppData\Roaming\opencode\context-mode\{sessions,content}`
  (override with `CONTEXT_MODE_DIR`, must be absolute).
- CLI: `context-mode index|search|doctor|upgrade|hook|statusline`; bare = stdio server.

### 2.3 ddgs — the 6 tools

`search_text` · `search_images` · `search_news` · `search_videos` · `search_books` · `extract_content`

No API key needed. Backend engines for `text()`: bing, brave, duckduckgo, google,
grokipedia, mojeek, startpage, yandex, yahoo, wikipedia.

### 2.4 obscura — 37 browser tools

`browser_navigate` `browser_snapshot` `browser_click` `browser_fill` `browser_type`
`browser_press_key` `browser_select_option` `browser_evaluate` `browser_wait_for`
`browser_network_requests` `browser_console_messages` `browser_close` `browser_markdown`
`browser_links` `browser_interactive_elements` `browser_back` `browser_forward`
`browser_reload` `browser_get_cookies` `browser_set_cookie` `browser_clear_cookies`
`browser_wait_for_text` `browser_detect_forms` `browser_fill_form` `browser_scroll`
`browser_get_attribute` `browser_count` `browser_extract` `browser_tab_new`
`browser_tab_list` `browser_tab_switch` `browser_tab_close` `browser_search`
`browser_storage_state` `browser_set_storage_state` `browser_screenshot` `browser_pdf`

Global flags: `--stealth` (fingerprint + TLS impersonation), `--obey-robots` (fetch/scrape),
`--allow-private-network` (needed for localhost), `--v8-flags`.

### 2.5 graphify — 10 tools

`query_graph` `get_node` `get_neighbors` `get_community` `god_nodes` `graph_stats`
`shortest_path` `list_prs` `get_pr_impact` `triage_prs`

There is **no `graphify serve`** — the MCP is the separate `graphify-mcp.exe`.
Default graph: `graphify-out/graph.json`.

### 2.6 Pre-existing servers (unchanged, verified)

`serena` (32) · `aidex` (31) · `firecrawl` (27) · `github` (31) · `context7` (2) ·
`brave` (8) · `mslearn` (3) · `headroom` (3) · `semble` (2) · `mcp-windbg` (10)

API keys all present as Windows user env vars: `CONTEXT7_API_KEY`, `FIRECRAWL_API_KEY`,
`BRAVE_API_KEY`, `GH_TOKEN`. `github` uses a registry-reading wrapper
(`.local\bin\mcp\github-mcp.cmd`) so the token never lands in a config file.

### 2.7 serena — usage

Version 1.7.0. `--context ide` **disables** `create_text_file`, `read_file`,
`execute_shell_command`, `find_file`, `list_dir`; `single_project` disables
`activate_project`. Use OpenCode's built-in file/shell tools instead.

**Call `initial_instructions` first, every session** — it returns the authoritative
name-path syntax, overload-index rules, and the `replace_in_files` dry-run protocol.
OpenCode drops the server-pushed copy, so it must be requested explicitly.

C++/JUCE workflow:
1. `get_symbols_overview` on the header (`depth=1..2`) — never read whole files
2. `find_symbol` with `name_path_pattern` (e.g. `"MyClass/processBlock"`), `include_body=True`
3. `find_referencing_symbols` to size the blast radius
4. Edit with the narrowest tool: `replace_symbol_body` (whole body — **must have
   retrieved with `include_body=True` first**) > `insert_after_symbol` /
   `insert_before_symbol` > `replace_content` (partial, regex-capable) >
   `replace_in_files` (multi-file, **always `dry_run=True` first**)
5. `get_diagnostics_for_file`; `safe_delete_symbol` instead of manual deletion;
   `rename_symbol` for renames — never search-and-replace

Gotchas: line numbers are **0-based** in serena but 1-based in OpenCode's `read`
(top off-by-one source). Overload index is 0-based: `MyClass/my_method[1]`.
`delete_lines`/`replace_lines`/`insert_at_line` require a prior `read_file`, which is
disabled in this config — **use `replace_content` instead**. `replace_symbol_body`
replaces only the body, so `JUCE_DECLARE_NON_COPYABLE…` / `JUCE_LEAK_DETECTOR` tails
must be re-added when replacing a whole class. Lambdas and `AudioProcessor` callbacks
are usually not indexed as symbols — fall back to `search_for_pattern`.

Memories: `.serena/memories/` (45+ files, versioned; `/cache` and `project.local.yml` ignored).
Language servers in `.serena/project.yml`: `[cpp, python_basedpyright, bash]`.
Note the key is `python_basedpyright`, **not** `python` (renamed in 1.7.0).

### 2.7 AiDex

Version 2.3.0, 31 tools. **Index refreshed** via `aidex_global_refresh`:
484 files / 20,787 items / 8,038 methods / 714 types.

Caveat: embeddings are sparse — only **1,209 of ~8,752 symbols** are embedded
(`modelId: jina-code`, dim 768). So `aidex_query` (semantic) is weak; `aidex_search`
and `aidex_signature` (structural) are the reliable paths. Check freshness with
`aidex_status` before trusting it — it never warns about staleness.

---

## 3. Token-reduction pipeline — what actually works here

### 3.1 Layer status

| Layer | Status | Reality |
|---|---|---|
| **context-mode MCP** | **PRIMARY — working** | 11 tools + plugin now registered |
| **rtk (WSL)** | **WORKING — 60–95% reduction** | both binaries present; plugin fixed |
| **headroom proxy** | **RUNNING BUT USELESS HERE** | `api_requests: 0`; client ignores `ANTHROPIC_BASE_URL` |
| **headroom MCP/CLI** | **working** | 3 tools: compress / retrieve / stats |

### 3.2 headroom — measured truth

- v0.37.0. **The CLI has NO `compress` or `retrieve` subcommand** (31 subcommands,
  confirmed from `--help`). Those exist **only as MCP tools**. The prior AGENTS.md note
  on this point is correct.
- CCR markers look like `<<ccr:abc123,base64,2.0KB>>` or `<<ccr:HASH N_rows_offloaded>>`.
  Originals live in `C:\Users\user\.headroom\ccr_store.db`; restore with
  `headroom_retrieve` using the hash. Reactive use only.
- `headroom doctor` → 0 failures, 6 warnings. `claude`/`codex` = "not routed",
  `savings` = "no tokens yet". `perf` is empty because the proxy was never started
  with `--log-file`.
- Verify traffic actually flows: `Invoke-WebRequest http://127.0.0.1:8787/stats` and
  watch `summary.api_requests`. **Today it is 0.**

### 3.3 rtk — fixes applied

- **Corrupt tracking DB fixed.** `~/.local/share/rtk/history.db` was malformed
  (`database disk image is malformed: Error code 11`), breaking `rtk gain`. Deleted;
  rtk recreates it. Compression was never affected.
- **A native `rtk.exe` 0.49.0 exists at `C:\WINDOWS\system32\rtk.exe`** in addition to
  the WSL binary. Both work; the WSL one is what matters for WSL commands.
- **CORRECTION 3: the V1-shaped plugin was NOT dead code.** `.opencode/plugins/rtk-wsl/index.ts`
  used `export default { id, setup }`, which is exactly the contract this build requires.
  The official named-export form is what gets rejected. My earlier claim that it was
  V1 dead code was wrong.
  (`headroom-proxy.ts`, `rtk-wsl.ts`) were already backed up there.
- **No rtk plugin is installed (deliberately).** An auto-rewriter must spawn a child
  process per shell command, and *that* is what flashes a console window every time.
  Use an explicit `rtk <cmd>` prefix instead.
  `rtk rewrite "wsl bash -lc 'ls -la'"` returns **empty** (verified). Guards:
  empty rewrite → passthrough; already `rtk ` → skip; any error → passthrough.

`rtk rewrite` behaviour (measured):

| Input | Output |
|---|---|
| `ls -la` | `rtk ls -la` |
| `rg -n TODO src` | `rtk rg -n TODO src` |
| `git status` | `rtk git status` |
| `cat CMakeLists.txt` | `rtk read CMakeLists.txt` |
| `grep -rn foo src` | `rtk grep -rn foo src` |
| `ast-grep run -p x` | *(empty — unsupported, correct)* |
| `cmake --build build` | *(empty — unsupported, correct)* |

Measured reduction: `rg` 95.4% bytes / 93.9% lines · `ls -la` 80.7% · `rtk smart` 99.97%.
Honest negatives: `rtk` does **not** compress `git status` in a repo with 170 real
modifications (~1%), and `rtk read` is a no-op on clean prose (0%). Elided output is
recoverable with `rtk recall`.

WSL permanence: `~/.bashrc` lines ~121–137 (aliases `ls`/`grep`/`cat`/`find`/`diff`/
`status` → rtk). `~/.wslrc` does not exist and is not needed.

> **Security note:** `~/.bashrc` exports a plaintext `PERPLEXITY_API_KEY`. Treat as
> compromised and rotate.

### 3.4 Decision table (no double compression)

The three layers act at *different points*, so they are not substitutes:
context-mode removes bytes **before** they enter context; rtk removes bytes **at the
WSL shell boundary**; headroom proxy would remove bytes **on the wire** (inert here).

| # | Situation | Use | Do NOT use |
|---|---|---|---|
| 1 | Big source/log/CSV to scan, count, aggregate | `ctx_execute_file` | `Read`, `headroom_compress` |
| 2 | 2+ independent commands | `ctx_batch_execute` (4–8) | serial shell calls |
| 3 | Recall an earlier session | `ctx_search`, `ctx_insight` | re-`Read`ing the file |
| 4 | Web page / doc site / API JSON | `ctx_fetch_and_index` → `ctx_search` | raw `webfetch` output |
| 5 | Long single WSL CLI output | `rtk <cmd>` (or `.bashrc` alias) | bare `rg`/`git`/`ls` in WSL |
| 6 | WSL `ast-grep`/`ag`/`fdfind` | bare binary | `rtk ast-grep` (no-op) |
| 7 | Exact lines needed for an edit | `Read` / `Edit` | `ctx_execute_file` for a 5-line change |
| 8 | Huge content already in window, no index/tee | `headroom_compress` | `ctx_execute` (can't un-read) |
| 9 | A `<<ccr:HASH,…>>` marker appears | `headroom_retrieve` | guessing, re-running |
| 10 | Fresh authoritative bytes | `Read` then edit | every other layer |

Hard rules: never `Read` a >300-line file you only need to *scan*; never issue >2
independent shell commands serially; never let raw HTML touch context; don't nest
(`ctx_execute("shell","rtk ls")` is pointless).

**Caveat:** the routing policy above is currently **advisory**. context-mode's plugin
is registered but only takes effect after an OpenCode restart. Until then, route manually.

---

## 4. Code search

### Index freshness (all refreshed 2026-09-26)

| Index | Size | State |
|---|---|---|
| `.cocoindex_code/` | 1.87 GB, 97,676 chunks / 1,889 files | **FRESH, live daemon** |
| `graphify-out/graph.json` | 87.6 MB, 55,419 nodes / 69,790 edges / 2,843 communities | **FRESH** |
| `.tgtep/` | 59.8 MB, 2,010 files / 227,259 trigrams | **FRESH** |
| `.aidex/` | 36.1 MB, 484 files / 20,787 items | **FRESH** (embeddings sparse) |

git HEAD: `1e9e63e3` (2026-09-22). **These indexes never warn about staleness — check
mtime before trusting any of them.**

### 4.1 cocoindex-code (`ccc` 0.2.41)

Two independent modes:

- **`ccc grep` — structural, by example, NO index, NO daemon, immune to staleness.**
  Metavariables use `\`: `\NAME` = one node, `\(ARGS*\)` = a run of siblings,
  `\_`/`\*` = anonymous.
- **`ccc search` — embedding similarity. Requires the daemon + index.**

```powershell
ccc grep 'class \NAME:' --lang cpp src
ccc grep 'void \NAME(\(ARGS*\))' --lang cpp src
ccc search --lang cpp --limit 5 "convolution reverb tail decay"
ccc search --lang cpp --path 'src/audioengine/*' "plugin bypass state"
ccc status ; ccc daemon restart
```

> Always pass `--lang cpp`. The index holds 53,967 markdown + 33,185 html chunks vs
> only 7,895 cpp, so unqualified searches drown in docs.

### 4.2 tgrep (1.0.4) — **not semantic**

Trigram-indexed regex grep, ripgrep-compatible, client/server. No embeddings, no LLM.

> **Correction to prior AGENTS.md:** tgrep's default index path is **`.tgrep/`**, not
> `.tgtep/`. `.tgtep/` is a repo-local convention. **Every** `status`/`index`/`serve`/
> search call must repeat `--index-path .tgtep` or it silently finds nothing.

```powershell
tgrep --index-path .tgtep -t cpp -n -- 'processBlock'
tgrep --index-path .tgtep -t cpp -l -- 'juce::AudioProcessor'
tgrep --index-path .tgtep --vimgrep -- 'prepareToPlay' .
tgrep index . --index-path .tgtep --force      # refresh
tgrep serve . --index-path .tgtep              # self-refreshing daemon
```

### 4.3 graphify (0.9.64 / MCP 1.30.0)

```powershell
graphify query "how does audio flow from device callback to the convolver"
graphify path "AudioEngine" "MKLNonUniformConvolver"
graphify affected "AudioEngineProcessor" --depth 2
graphify god-nodes --top 15
graphify update .          # code-only refresh, NO LLM/API key — the cheap path
graphify update . --force  # after deletions/refactors
```

Index creation is the **`/graphify` skill** or `graphify extract`; there is no
`graphify serve`. `graphify update .` needs no API key.

### 4.4 Division of labour

| Need | Tool |
|---|---|
| Find a symbol by name | **serena `find_symbol`** (LSP-precise, never stale) → tgrep |
| Find all callers of X | **serena `find_referencing_symbols`** (only true edge accuracy) |
| Find code that *semantically* does X | **`ccc search`** |
| Find code by *syntax shape* | **`ccc grep`** (instant, no index) |
| Understand call paths / architecture | **graphify** `path` / `affected` / `query` |
| Bulk repo-wide regex sweep | **tgrep** (with `--index-path .tgtep`) |

---

## 5. Static / dynamic analysis

### 5.1 cppcheck 2.22.0

> **`--output-format=json` does not exist in 2.22.** Valid: `text`, `sarif`, `xml`
> (deprecated), `xmlv2`, `xmlv3`. Text output is via `--template=`.
> Also: cppcheck 2.22 has **no `unknownMacro` id** — the real suppressor for the JUCE
> `#error` noise is `preprocessorErrorDirective`.

**The no-compile-database path cannot parse JUCE headers.** Measured cascade: no flags →
`#error "Unknown platform!"`; `-D_WIN32` → `#error unknown compiler`; `+_MSC_VER` →
`#error "JUCE requires Visual Studio 2017 or later"`; `+_MSC_FULL_VER` →
`#error "JUCE requires C++17 or later"`; `+_MSC_LANG` → still fails, because cppcheck
refuses to let `-D` redefine `__cplusplus`/`_MSVC_LANG`.

**Always use `--project=`.** The repo DB is a valid **2,112-entry JSON array**
(5.79 MB, at repo root and in `build/`) — a real array, **not** JSONL.

```powershell
& "C:\Program Files\Cppcheck\cppcheck.exe" `
  --project="C:\VSC_Project\ConvoPeq\build\compile_commands.json" `
  --enable=warning,performance,portability --inconclusive --inline-suppr `
  --suppress=missingIncludeSystem --suppress=missingInclude --suppress=unusedFunction `
  --suppress=unmatchedSuppression --std=c++20 --platform=win64 -j 8 `
  --template=gcc --relative-paths="C:\VSC_Project\ConvoPeq" `
  --file-filter="*ConvolutionEngine*.cpp"
```

### 5.2 clang-tidy 23.1.2

25 checks enabled by default; **602** in the full registry
(`--list-checks "-checks=*"` — must be a *separate* argument; the `-checks=*` joined
form errors as an invalid boolean).

`-p` takes a **directory**, not a file. A JSONL `compile_commands.json` is rejected
with `json-compilation-database: Expected array.`, after which it degrades to
"Running without flags" and emits ~96 bogus warnings. The repo DB is fine, but
`compile_commands_clang.json` (3 entries) and `tmp\clangdb\compile_commands.json` are
much faster for iteration.

> **New gotcha:** a **multi-config** DB analyzes each file **once per config**.
> `-p C:\VSC_Project\ConvoPeq` (Ninja Multi-Config) ran one test file **4×** and took
> **27 s**. Use the single-config 3-entry DB for fast loops.

```powershell
New-Item -ItemType Directory -Force "C:\Users\user\AppData\Local\Temp\opencode\clangdb" | Out-Null
Copy-Item C:\VSC_Project\ConvoPeq\compile_commands_clang.json "C:\Users\user\AppData\Local\Temp\opencode\clangdb\compile_commands.json"

& "C:\Program Files\LLVM\bin\clang-tidy.exe" `
  -p "C:\Users\user\AppData\Local\Temp\opencode\clangdb" `
  --checks="-*,bugprone-*,clang-analyzer-*,performance-*,concurrency-*,cert-*" `
  --header-filter=".*/src/.*" --fix-errors --format-style=file `
  "C:\VSC_Project\ConvoPeq\src\audioengine\ConvolutionEngine.cpp"
```

A conservative `.clang-tidy` already exists at the repo root. `git-clang-format` and
`run-clang-tidy.py` are **not shipped** in LLVM 23's `bin` (`clang-format.exe` is).

### 5.3 Dr. Memory 2.6.20434 — **NOT USABLE ON THIS HOST**

Fails on a trivial non-project target, so it is not a JUCE/project issue —
DynamoRIO cannot inject, most likely blocked by endpoint security software:

```
Dr. Memory internal crash at PC 0x034cba94.
unable to locate results file ... (code=2)
Dr. Memory failed to start the target application, perhaps due to
interference from invasive security software.
EXITCODE: -1
```

Do not spend more time on it. **Use AddressSanitizer instead — verified working:**

- `clang_rt.asan_dynamic-x86_64.dll` exists at
  `C:\Program Files\LLVM\lib\clang\23\lib\windows\`
- The project **already has it wired**: `CMakeLists.txt` has
  `option(ENABLE_ASAN "Enable AddressSanitizer (Debug only)" OFF)`, applies
  `/fsanitize=address` (MSVC) / `-fsanitize=address` (Clang), forces `/MDd`, strips
  `/RTC1`, forbids combining with PGO. `build-asan\` exists with `ENABLE_ASAN:BOOL=ON`.
- A deliberate `heap-buffer-overflow` compiled with `clang++ -fsanitize=address` **was
  detected** once the DLL dir was on `PATH`.

```powershell
$env:PATH = "C:\Program Files\LLVM\lib\clang\23\lib\windows;$env:PATH"
.\build.bat Debug nopause -DENABLE_ASAN=ON
$env:ASAN_OPTIONS = "detect_leaks=1:abort_on_error=0:detect_stack_use_after_return=1"
& ".\build-asan\ConvoPeq_artefacts\Debug\ConvoPeq.exe"
```

Second fallback, no rebuild: `C:\Windows\System32\appverif.exe` exists —
`appverif.exe -enable Heaps -for <exe> -logs .\appverif -v`.

A 47 MB ASan-instrumented GUI app needs a lot of virtual memory; prefer running the
CTest targets under `build-asan` over the full app.

---

## 6. WSL toolchain — all working

`grep 3.12` · `ast-grep 0.44.0` (`sg`) · `ripgrep 15.1.0` · `fdfind 10.3.0` ·
`ag 2.2.0` · `fzf 0.67.0` · `sed 4.9` · `awk 5.3.2` · `rtk 0.49.0`

> **pwsh → WSL quoting is the #1 practical trap.** `wsl bash -lc "... 2>&1 ..."`
> gets `2>&1` rewritten by pwsh into a file named `C:\dev\null`, and the AGENTS.md
> `<<'EOS'` heredoc recipe fails because pwsh's parser rejects `<<`.
> **Write the script to a file and run it:** `wsl bash /mnt/c/.../script.sh`.
> Inside WSL use POSIX paths (`/mnt/c/...`, `/c/...`), never `C:\...`.

---

## 7. Math / numerics

- **Octave 11.3.0** — `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin\octave-cli.exe`.
  Works: `octave-cli --eval "..."`.
- **Maxima 5.50.0** — `C:\maxima-5.50.0\bin\maxima.bat` (shim: `~/.local/bin/maxima.cmd`).
  > **`--eval` is NOT a Maxima option** — it is consumed by the SBCL layer and Maxima
  > reports `Warning: argument eval not recognized`. It does not work.
  > **Two working forms:**
  > ```powershell
  > # (a) batch file — MUST use forward slashes (backslashes are eaten)
  > & "C:\maxima-5.50.0\bin\maxima.bat" --quiet -batch "C:/Users/user/AppData/Local/Temp/opencode/m.mac"
  > # (b) stdin — terminator required
  > "print(3/4); diff(sin(x)*exp(x),x);" | & "C:\maxima-5.50.0\bin\maxima.bat" --quiet
  > ```
- **numpy 2.5.2 · scipy 1.18.0 · matplotlib 3.11.1 · pandas 3.0.5 · IPython 9.17.1 ·
  crawl4ai 0.9.3 · trafilatura 2.2.0 · ddgs 9.16.0** — all import and run.

---

## 8. Config changes made

| File | Change |
|---|---|
| `ConvoPeq\opencode.json` | registered `context-mode`, `obscura`, `ddgs`, `graphify`, `cocoindex`; all 13 local servers then wrapped in the windowless launcher |
| `~\.config\opencode\opencode.jsonc` | `context-mode upgrade` added `"plugin": ["context-mode"]` (backup: `opencode.jsonc.bak`) |
| `~\.config\opencode\plugins\rtk.ts` | **new** — V2.0 rtk plugin with WSL-wrapper support |
| `ConvoPeq\.opencode\plugin-v1-backup\rtk-wsl-dir-v1\` | V1 dead-code plugin moved out of the V2 auto-load dir |
| `~\.local\share\rtk\history.db` (WSL) | corrupt DB deleted → `rtk gain` works again |
| `.aidex/`, `.tgtep/`, `graphify-out/`, `.cocoindex_code/` | indexes refreshed |

**Action required: restart OpenCode** for the 5 new MCP servers and the context-mode
plugin to take effect. Until then context-mode routing is advisory.

## 9. Open items

1. **Restart OpenCode** — required for the new servers + plugin.
2. **Rotate `PERPLEXITY_API_KEY`** — plaintext in WSL `~/.bashrc`.
3. **Decide on the headroom proxy.** It is running and healthy but structurally cannot
   compress OpenChamber traffic. Either accept it as CLI/MCP-only, or investigate whether
   OpenCode V2.0 supports a proxy-configured provider endpoint.
4. **Consider trimming the server set.** 15 MCP servers inject tool schemas every turn;
   obscura alone contributes 37. `tools: { "obscura_*": false }` in `opencode.json`
   disables a server's tools without unregistering it.
5. **Duplicate MCP processes** were observed (2× serena, 4× mcp-windbg, 2× semble,
   2× ccc, 3× headroom) — likely from concurrent sessions. Close stale sessions.
6. **Dr. Memory** — no further action; use ASan.
