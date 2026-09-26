# Tool Environment Inventory — OpenCode Desktop V2 (2026-09-25)

## Status: ✅ All Tools Available and Verified

This document records every tool available in the OpenCode Desktop V2 environment,
confirmed working as of 2026-09-25. It replaces `doc/tool-inventory-2026-09-20.md`
and `doc/tool-inventory-2026-09-21-freebuff.md`.

---

## 1. MCP Servers (11 servers, 134 tools total)

All MCP servers are connected and their tools are callable via `tools["<namespace>"]["<tool-name>"]`
inside `execute()`.

| Server | Tools | Namespace | Configuration | Notes |
|--------|-------|-----------|---------------|-------|
| **serena** | 32 | `tools.serena.*` | project `opencode.json` → `serena.exe start-mcp-server --context ide --mode editing --add-mode interactive` | Run `tools.serena.initial_instructions()` before coding tasks. Symbols, diagnostics, editing, memory. |
| **aidex** | 31 | `tools.aidex.*` | project `opencode.json` → node + `aidex-mcp/build/index.js` | `.aidex/index.db` (479 files, 20,385 items, embeddings enabled). Start with `aidex_session`. |
| **brave** | 8 | `tools.brave.*` | project `opencode.json` → npx @brave/brave-search-mcp-server | `brave_image_search`, `brave_summarizer`, `brave_llm_context`, `brave_local_search`, `brave_news_search`, `brave_place_search`, `brave_video_search`, `brave_web_search`. API key: `BRAVE_API_KEY`. |
| **context-mode** | 11 | `tools["context-mode"].*` | global `opencode.jsonc` → node + `context-mode/cli.bundle.mjs` | `ctx_execute`, `ctx_batch_execute`, `ctx_search`, `ctx_index`, `ctx_fetch_and_index`, `ctx_doctor`, `ctx_stats`, `ctx_purge`, `ctx_insight`, `ctx_upgrade`. Version: v1.0.169. See §5. |
| **context7** | 2 | `tools.context7[*].*` | project `opencode.json` → remote `https://mcp.context7.com/mcp` | `resolve-library-id`, `query-docs`. API key: `CONTEXT7_API_KEY`. |
| **firecrawl** | 27 | `tools.firecrawl.*` | project `opencode.json` → npx firecrawl-mcp | `firecrawl_scrape`, `firecrawl_search`, `firecrawl_crawl`, `firecrawl_map`, `firecrawl_agent`, `firecrawl_parse`, + monitor/research tools. API key: `FIRECRAWL_API_KEY`. |
| **github** | 31 | `tools.github.*` | global `opencode.jsonc` → `github-mcp.cmd` wrapper | `get_me`, `list_issues`, `search_issues`, `create_branch`, `create_pull_request`, `list_pull_requests`, `merge_pull_request`, `add_issue_comment`, + more. Token: `GH_TOKEN` (mapped to `GITHUB_PERSONAL_ACCESS_TOKEN`). |
| **headroom** | 3 | `tools.headroom.*` | project `opencode.json` → `.venv/Scripts/headroom.exe mcp serve` | `headroom_compress`, `headroom_retrieve`, `headroom_stats`. Proxy running on port 8787. See §5. |
| **mslearn** | 3 | `tools.mslearn.*` | project `opencode.json` → remote `https://learn.microsoft.com/api/mcp` | `microsoft_docs_search`, `microsoft_docs_fetch`, `microsoft_code_sample_search`. |
| **semble** | 2 | `tools.semble.*` | global `opencode.jsonc` → `uvx --from semble[mcp]==0.6.0 semble` | `search`, `find_related`. Version: 0.6.0. |
| **opencode** | 5 | `tools.opencode.*` | built-in | `session_move`, `session_rename`, `models`, `list_mcp_resources`, `read_mcp_resource`. |

### MCP Server Instructions

- **brave**: Use tools from this server through `execute` under `tools["brave"]`.
  Use this server to search the Web for various types of data via the Brave Search API.

- **context7**: Use tools from this server through `execute` under `tools["context7"]`.
  Use this server to fetch current documentation whenever the user asks about a library, framework,
  SDK, API, CLI tool, or cloud service — even well-known ones like React, Next.js, Prisma, Express,
  Tailwind, Django, or Spring Boot. Do not use for refactoring, writing scripts from scratch,
  debugging business logic, code review, or general programming concepts.

- **firecrawl**: Use tools from this server through `execute` under `tools["firecrawl"]`.
  Firecrawl provides web search, page retrieval, site URL discovery, multi-page collection,
  structured page data, monitoring, and asynchronous research.

- **github**: Use tools from this server through `execute` under `tools["github"]`.
  Tool selection: Use `list_*` tools for broad retrieval, `search_*` tools for targeted queries.
  Always call `get_me` first to understand permissions. Check for PR templates before creating PRs.

- **mslearn**: Use tools from this server through `execute` under `tools["mslearn"]`.
  Workflow: `microsoft_docs_search` → `microsoft_code_sample_search` → `microsoft_docs_fetch`.

- **semble**: Use tools from this server through `execute` under `tools["semble"]`.
  Call `search` once with a focused query for code search. Use `find_related` to discover similar code.
  Pass `C:/VSC_Project/ConvoPeq` as `repo` for local projects.

- **serena**: Use tools from this server through `execute` under `tools["serena"]`.
  CRITICAL: Call `initial_instructions` before coding tasks. Use `find_symbol` / `get_symbols_overview`
  for code navigation, `search_for_pattern` for pattern search. Line numbers are 0-based.

---

## 2. CLI Tools

### Search & Code Analysis Tools

| Tool | Path | Version | Access |
|------|------|---------|--------|
| **ast-grep** | `/usr/local/bin/ast-grep` (WSL) | 0.44.0 | `wsl bash -lc 'ast-grep ...'` or `wsl ast-grep ...` |
| **rg (ripgrep)** | `/usr/bin/rg` (WSL) | 15.1.1 | `wsl bash -lc 'rg ...'` |
| **fdfind** | `/usr/bin/fdfind` (WSL) | — | `wsl bash -lc 'fdfind ...'` |
| **fd** | `/home/user/.local/bin/fd` (WSL) + `fd.exe` (Windows) | — | `wsl bash -lc 'fd ...'` or `fd.exe` |
| **ag (silver-searcher)** | `/usr/bin/ag` (WSL) | 2.2.0 | `wsl bash -lc 'ag ...'` |
| **fzf** | `/usr/bin/fzf` (WSL) + `fzf.exe` (Windows) | 0.67.0 | `wsl bash -lc 'fzf ...'` or `fzf.exe` |
| **tgrep** | `C:\Windows\system32\tgrep.exe` | 1.0.4 | `tgrep status`, `tgrep "query" .tgtep` |
| **ccc (cocoindex-code)** | `C:\Users\user\.local\bin\ccc.exe` | 0.2.41 | `ccc search`, `ccc grep`, `ccc status`, `ccc doctor`, `ccc mcp` |
| **graphify** | `Scripts\graphify.exe` (Python) + `uv` | 0.9.64 | `graphify query`, `graphify update`, `graphify explain`, `graphify god-nodes` |
| **semble** | `C:\Users\user\.local\bin\semble.exe` | 0.6.0 | `semble search "q" [path]`, `semble find-related <file> <line>`, `semble savings` |

### LSP / Language Tools

| Tool | Path | Version | Access |
|------|------|---------|--------|
| **serena** | `C:\Users\user\.local\bin\serena.exe` | 1.7.0 | MCP via `serena.exe start-mcp-server --context ide --mode editing --add-mode interactive` |
| **BasedPyright** | via serena LSP | — | Python type checking (through serena) |
| **clangd** | (via serena cpp LSP) | — | C++ language server (through serena) |
| **bash-language-server** | (via serena bash LSP) | — | Bash language server (through serena) |
| **cppcheck** | `C:\Program Files\Cppcheck\cppcheck.exe` | 2.22.0 | `cppcheck --enable=warning,style,portability --std=c++17 src/` |
| **clang-tidy** | `C:\Program Files\LLVM\bin\clang-tidy.exe` | LLVM 23.1.2 | `clang-tidy -p build src/file.cpp -- -std=c++17` |
| **Dr.Memory** | `C:\Program Files (x86)\Dr. Memory\bin\drmemory.exe` | — | `drmemory.exe -- ./app.exe args` |

### Build & Debug Tools

| Tool | Path | Version | Access |
|------|------|---------|--------|
| **rtk** | `/home/user/.local/bin/rtk` (WSL) | 0.49.0 | `wsl bash -lc 'rtk git status'` (auto-rewrites commands for compression) |
| **cdb** | (Windows Debugger) | — | User-mode debugging (`cdb.exe`) |
| **cdb (mcp-windbg)** | `C:\VSC_Project\ConvoPeq\tmp\cdb.exe` | — | Via `mcp-windbg` MCP server (uvx `mcp-windbg==1.3.0`) |

### Web & Content Tools

| Tool | Path | Version | Access |
|------|------|---------|--------|
| **obscura** | `C:\Users\user\.local\bin\obscura\obscura.exe` | — | `obscura fetch <url>`, `obscura scrape`, `obscura serve --port 9222`, `obscura mcp` |
| **crawl4ai** | Python module | — | `python -c "import crawl4ai; ..."` |
| **ddgs** | Scripts\ddgs.exe + Python | — | `ddgs text "q"`, `ddgs news`, `ddgs extract <url>` |
| **trafilatura** | Python module | — | `trafilatura -u URL --markdown` |

### Package Managers & Runtimes

| Tool | Path | Version | Notes |
|------|------|---------|-------|
| **Node.js** | `C:\Program Files\nodejs\node.exe` | 26.8.1 | NOT nvm4w path — system PATH has correct node |
| **uv** | `C:\Users\user\.local\bin\uv.exe` | — | Python package installer/resolver |
| **uvx** | `C:\Users\user\.local\bin\uvx.exe` | — | Run Python tools in isolated environments |
| **Python** | system `python` | 3.14.7 | Used by serena bridge (if needed) and numerical computing |
| **Python3** | `C:\Python314` | 3.14.x | System Python for numerical computing |

### Numerical Computing

| Tool | Path | Version |
|------|------|---------|
| **NumPy** | Python package | 2.5.2 |
| **SciPy** | Python package | 1.18.0 |
| **Matplotlib** | Python package | 3.11.1 |
| **pandas** | Python package | 3.0.5 |
| **IPython** | Python package | 9.17.1 |
| **GNU Octave** | `C:\Program Files\GNU Octave\Octave-11.3.0\mingw64\bin\octave.exe` | 11.3.0 |
| **Maxima** | `C:\maxima-5.50.0\bin\maxima.bat` or `~/.local/bin/maxima` | 5.50.0 |

---

## 3. Index Databases & Knowledge Graphs

| Index | Path | Status | Updated |
|-------|------|--------|---------|
| **AiDex** | `.aidex/index.db` (35MB) | 479 files, 20,385 items, embeddings enabled (jina-code 768d, 1,509 vectors) | 2026-09-25 20:26 |
| **Cocoindex** | `.cocoindex_code/target_sqlite.db` (1GB) | 305,797 chunks / 2,595 files | 2026-09-24 |
| **tgrep** | `.tgtep/index.bin` | 2,859 files / 260,018 trigrams | 2026-09-21 21:53 |
| **Graphify** | `graphify-out/graph.json` (88MB) | 74,421 nodes | 2026-09-21 12:58 |

### Index commands
- **AiDex**: `aidex_status({ path })` → `aidex_session({ path })` → `aidex_query({ path, term })` or `aidex_search({ query, scope: "linked" })`
- **Cocoindex**: `ccc search "<q>"`, `ccc grep 'pattern'`, `ccc status`, `ccc doctor`
- **tgrep**: `tgrep "query" --index-path .tgtep`, `tgrep status --index-path .tgtep`
- **Graphify**: `graphify query "Q"`, `graphify explain X`, `graphify path A B`, `graphify god-nodes`, `graphify update .`

---

## 4. Running Services

| Service | Port | Status |
|---------|------|--------|
| **Headroom proxy** | 127.0.0.1:8787 | ✅ LISTENING — `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` |
| **Serena MCP (stdio)** | via subprocess | ✅ Running (via opencode.json `serena.exe start-mcp-server`) |

---

## 5. 3-Layer Token Reduction Pipeline

The pipeline operates as follows:

```
┌──────────────────────────────────────────────────────────┐
│  Layer 1: Prevent tokens from entering context            │
│  (context-mode MCP — Think-in-Code)                       │
│  • ctx_execute: Run code in sandbox, only print summary   │
│  • ctx_batch_execute: Parallel I/O with auto-indexing     │
│  • ctx_execute_file: Analyze files without reading them   │
│  • ctx_search: Query indexed content + session memory     │
│  • ctx_index: Store docs for on-demand retrieval          │
└────────────────────────┬──────────────────────────────────┘
                          │ (raw bytes stay in sandbox/storage)
┌────────────────────────▼──────────────────────────────────┐
│  Layer 2: Auto-compress traffic via proxy                 │
│  (headroom proxy — port 8787)                             │
│  • All API requests routed through ANTHROPIC_BASE_URL     │
│  • Automatic compression (40% target ratio)              │
│  • Tools: headroom_compress / headroom_retrieve (MCP)     │
└────────────────────────┬──────────────────────────────────┘
                          │ (compressed tokens)
┌────────────────────────▼──────────────────────────────────┐
│  Layer 3: Compress CLI output                             │
│  (rtk — WSL version 0.49.0)                               │
│  • Auto-rewrites wsl commands with rtk prefix             │
│  • 60-90% compression on git/shell outputs                │
│  • rtk-wsl.ts plugin intercepts `wsl bash -lc` commands    │
└──────────────────────────────────────────────────────────┘
```

### Role assignment (optimal split)

| Task | Primary tool | Fallback |
|------|-------------|----------|
| File analysis / aggregation / extraction | `ctx_execute` / `ctx_execute_file` | serena `get_symbols_overview` + `find_symbol` |
| Parallel command execution | `ctx_batch_execute` (concurrency 4-8 for I/O) | WSL + rtk |
| Past content search | `ctx_search` (knowledge base + session memory) | aidex `aidex_search` / semble `search` |
| Web page fetch | `ctx_fetch_and_index` → `ctx_search` | firecrawl `firecrawl_scrape` |
| Large context storage | `headroom_compress` (MCP) | `ctx_index` |
| CLI commands | `wsl bash -lc 'rtk <cmd>'` | WSL shell |
| File editing | Read + Edit (OpenCode built-in) | serena `replace_content` / `replace_symbol_body` |
| Code search (priority chain) | 1. aidex `aidex_query` → 2. serena `find_symbol` → 3. semble `search` → 4. ccc `search` → 5. WSL `rg` | — |

---

## 6. Search Priority Chain

When searching for code/symbols, use this priority order:

1. **AiDex** (`aidex_query` / `aidex_search`) — PREFERRED when `.aidex/` exists
   - Fast identifier/signature search with embeddings
   - Call `aidex_session({ path })` at session start
2. **Serena** (`find_symbol` / `search_for_pattern`) — Symbol-level search
   - LSP-backed, precise, with editing support
3. **semble** (`search`) — Natural language code search
   - Passes query through LLM for semantic matching
   - 99% token savings vs reading files
4. **ccc** (`search` / `grep`) — Semantic + structural search
   - AST-based search with language support
5. **tgrep** — Trigram-indexed grep on large trees
   - Use `--index-path .tgtep`
6. **graphify** — Knowledge graph traversal
   - `query`, `path`, `explain`, `god-nodes`
7. **WSL CLI** (`rg`, `ast-grep`, `ag`) — Fallback text/AST search
   - Always prefix with rtk: `wsl bash -lc 'rtk rg "pattern"'`

---

## 7. API Keys (Windows User Environment Variables)

| Key | Variable | Status |
|-----|----------|--------|
| Context7 | `CONTEXT7_API_KEY` | ✅ Set (ctx7sk-...) |
| Firecrawl | `FIRECRAWL_API_KEY` | ✅ Set (fc-...) |
| Brave Search | `BRAVE_API_KEY` | ✅ Set (BSAONVU...) |
| GitHub | `GH_TOKEN` | ✅ Set (ghp_...) |

### API Key Access Note
- **Context7 / Firecrawl / Brave** MCP servers use `{env:VAR_NAME}` syntax in opencode.json — OpenCode resolves these automatically.
- **GitHub** MCP server uses a wrapper script (`C:\Users\user\.local\bin\mcp\github-mcp.cmd`) that reads `GH_TOKEN` from `HKCU\Environment` registry and maps it to `GITHUB_PERSONAL_ACCESS_TOKEN` for `github-mcp-server.exe`.
- Key verification: `C:\Users\user\.local\bin\mcp\check-keys.cmd`

---

## 8. Configuration Files

### Project-level (C:\VSC_Project\ConvoPeq)

| File | Purpose |
|------|---------|
| `opencode.json` | OpenCode config: plugins, MCP servers, LSP settings |
| `.mcp.json` | Claude Code/Copilot MCP config (serena bridge, context-mode, aidex) |
| `.serena/project.yml` | Serena project config: language servers [cpp, python_basedpyright, bash] |
| `.serena/project.local.yml` | Local overrides for serena project config |
| `.aidex/` | AiDex index database and summary |
| `.cocoindex_code/` | Cocoindex code index |
| `.tgtep/` | tgrep index |
| `graphify-out/` | Graphify knowledge graph output |

### Global (C:\Users\user\.config\opencode)

| File | Purpose |
|------|---------|
| `opencode.jsonc` | Global OpenCode config: MCP servers (semble, github, context-mode) |
| `service.json` | OpenCode service password |
| `context-mode/` | context-mode sessions and content storage |

### Global (C:\Users\user\.local)

| Path | Purpose |
|------|---------|
| `.local/bin/` | CLI tools: serena, ccc, semble, headroom, obscura, uv, uvx, fd, fzf |
| `.local/bin/mcp/` | MCP wrapper scripts: github-mcp.cmd, context7-mcp.cmd, etc. |
| `.local/bin/obscura/` | Obscura headless browser binaries |
| `.serena/` | Serena global config (serena_config.yml) |

---

## 9. WSL Integration

- WSL distribution: Ubuntu 26.04 (running)
- Project path in WSL: `/mnt/c/VSC_Project/ConvoPeq`
- rtk path: `/home/user/.local/bin/rtk` (v0.49.0)
- WSL tools: rg, ast-grep, ag, fzf, sed, awk, fdfind (all in PATH)
- **rtk-wsl plugin** (`.opencode/plugin/rtk-wsl.ts`): Auto-rewrites `wsl bash -lc` commands through rtk for 60-90% CLI output compression

### WSL command patterns
```bash
wsl bash -lc 'rtk git status'           # Compressed git status
wsl bash -lc 'rtk rg "pattern" src/'    # Compressed search
wsl bash -lc 'rtk --help'               # Show rtk version and commands
```

---

## 10. Usage Examples by Category

### Code Search
```javascript
// AiDex semantic search
tools.aidex.aidex_search({ query: "retry with backoff", scope: "linked" })

// AiDex identifier search
tools.aidex.aidex_query({ path: "C:/VSC_Project/ConvoPeq", term: "AudioEngine" })

// Serena symbol search
tools.serena.find_symbol({ name_path_pattern: "AudioEngine", relative_path: "src" })

// semble natural language search
tools.semble.search({ query: "error handling pattern", repo: "C:/VSC_Project/ConvoPeq" })

// ccc code search
cmd /c ccc.exe search "class AudioEngine"
```

### Web Content
```javascript
// Brave web search (LLM context)
tools.brave.brave_llm_context({ query: "JUCE audio framework best practices" })

// Firecrawl scrape
tools.firecrawl.firecrawl_scrape({ url: "https://example.com", formats: ["markdown"] })

// Microsoft Learn docs
tools.mslearn.microsoft_docs_fetch({ url: "https://learn.microsoft.com/..." })

// Context7 library docs
tools.context7["resolve-library-id"]({ libraryName: "JUCE" })
tools.context7["query-docs"]({ libraryId: "/wearerokero/juce", query: "audio buffer processing" })
```

### GitHub
```javascript
// Get authenticated user
tools.github.get_me()

// Search issues
tools.github.search_issues({ query: "repo:ConvoPeq/ConvoPeq is:open", per_page: 5 })

// Create PR
tools.github.create_pull_request({
  owner: "ConvoPeq", repo: "ConvoPeq",
  head: "feature-branch", base: "main",
  title: "Add feature X", body: "..."
})
```

### Token Pipeline
```javascript
// Compress large content
tools.headroom.headroom_compress({ content: "large content here..." })

// Retrieve compressed content
tools.headroom.headroom_retrieve({ hash: "079eaf846b34d1be4ee27dd3" })

// Run code in sandbox (keeps output out of context)
tools["context-mode"].ctx_execute({
  language: "javascript", code: "const fs = require('fs'); console.log(fs.readdirSync('src').length + ' files')"
})

// Batch parallel commands
tools["context-mode"].ctx_batch_execute({
  commands: [
    { label: "disk", command: "df -h" },
    { label: "memory", command: "free -m" }
  ],
  queries: ["disk usage", "memory usage"],
  concurrency: 2
})

// Compressed WSL command
wsl bash -lc 'rtk git log --oneline -10'
```

### Static Analysis
```bash
# C++ static analysis
cppcheck --enable=warning,style,portability --std=c++17 --project=compile_commands.json src/
clang-tidy -p build src/file.cpp -- -std=c++17

# Dr. Memory
drmemory.exe -- ./AudioEngineHarness.exe
```

### Serena Memory Management
```javascript
// List tool-related memories
tools.serena.list_memories({ topic: "tools" })

// Read specific memory
tools.serena.read_memory({ memory_name: "tools/token-reduction-pipeline-2026-09-20" })
```

---

## 11. Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:8787` | Routes all Anthropic API traffic through headroom proxy for auto-compression |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8787/v1` | OpenAI compatibility endpoint for headroom proxy |
| `CONTEXT7_API_KEY` | `ctx7sk-...` | Context7 MCP documentation API |
| `FIRECRAWL_API_KEY` | `fc-...` | Firecrawl web scraping API |
| `BRAVE_API_KEY` | `BSAONVUAdVA4cFmb8NJz_...` | Brave Search API |
| `GH_TOKEN` | `ghp_...` | GitHub authentication (mapped to GITHUB_PERSONAL_ACCESS_TOKEN for MCP) |

---

*Generated: 2026-09-25*
*OpenCode Desktop V2 | Environment: Windows 11 + WSL2 (Ubuntu 26.04)*
