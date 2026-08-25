# Tool Availability Verification Report
**Date:** 2026-08-24
**Environment:** VS Code (Windows)
**Workspace:** `c:\VSC_Project\ConvoPeq`

---

## ✅ WSL Search Tools (All Available)

| Tool | Version | Status | Notes |
|------|---------|--------|-------|
| `grep` | GNU grep 3.12 | ✅ Available | WSL |
| `rg` (ripgrep) | 15.1.0 | ✅ Available | WSL |
| `ast-grep` | 0.44.0 | ✅ Available | WSL |
| `fdfind` (fd) | 10.3.0 | ✅ Available | WSL |
| `fzf` | 0.67.0 (debian) | ✅ Available | WSL |
| `sed` | GNU sed 4.9 | ✅ Available | WSL |
| `awk` | GNU Awk 5.3.2 | ✅ Available | WSL |
| `ag` (silver-searcher) | 2.2.0 | ✅ Available | WSL |

### Usage
```bash
# WSL direct
wsl sh -c "rg 'pattern' src/"
wsl sh -c "ast-grep --version"
wsl sh -c "fdfind '*.cpp'"

# With RTK prefix (recommended)
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk rg "pattern" src/'
```

---

## ✅ RTK (Token-Optimized CLI)

| Property | Value |
|----------|-------|
| Location | `~/.local/bin/rtk` (WSL) |
| Version | v0.43.0 |
| Token savings | 25.0M tokens (98.8%) across 1,774 commands |
| Status | ✅ Available |

### Usage (WSL mandatory)
```bash
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk git status'
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk rg "pattern" src/'
wsl bash -c 'cd /mnt/c/VSC_Project/ConvoPeq && ~/.local/bin/rtk grep -rn "pattern" src/'
```

---

## ✅ Windows CLI Tools

| Tool | Path | Version | Status | Functional |
|------|------|---------|--------|------------|
| cocoindex (ccc.exe) | `C:\Users\user\.local\bin\ccc.exe` | 0.2.41 | ✅ Installed | ✅ `ccc --help` / `ccc search` working |
| semble | `C:\Users\user\.local\bin\semble.exe` | 0.5.5 | ✅ Installed | ✅ `semble search` working (tested: found 5 results for "class MainWindow") |
| graphify | `C:\Users\user\AppData\Roaming\Python\Python314\Scripts\graphify.exe` | 0.9.48 | ✅ Installed | ✅ `graphify query` working (44,244 nodes in graph) |
| serena | `C:\Users\user\.local\bin\serena.exe` | 1.7.0 | ✅ Installed | ✅ MCP server running (see MCP section) |

### Usage
```cmd
# cocoindex (cocoindex-code)
ccc search "MainWindow" --project .
ccc search "function" --project . --top-k 10

# semble
semble search "class MainWindow" . --max-snippet-lines 5

# graphify
graphify query "MainWindow" --budget 500
graphify path "src/MainWindow.h" "src/MainApplication.cpp"

# serena (MCP only, see below)
```

---

## ✅ MCP Servers (VS Code Configuration)

### Configuration Files
- **`.vscode/mcp.json`** — Workspace MCP config
- **`C:\Users\user\AppData\Roaming\Code\User\settings.json`** — User MCP config (`mcpServers`)

### MCP Server Status

| MCP Server | Version | Status | Available Tools | Notes |
|------------|---------|--------|-----------------|-------|
| **serena** | 1.7.0 | ✅ Running | `get_symbols_overview`, `initial_instructions` (partial) | Configured with `--context ide --project C:\VSC_Project\ConvoPeq --mode editing --add-mode interactive` |
| **context-mode** | — | ✅ Running | All context-mode tools (`ctx_execute`, `ctx_batch_execute`, etc.) | Workspace `.vscode/mcp.json` config |
| **headroom** | v1 | ✅ Running | `headroom_compress`, `headroom_retrieve`, `headroom_stats` | Via `tools/headroom-mcp.cmd` |
| **aidex** | 2.3.0 | ✅ Installed, ⚠️ Tools disabled | N/A | Registered in `opencode.json` and VS Code settings. Tools available but currently disabled at environment level. |

### Serena MCP Server
- **Config file:** `C:\VSC_Project\ConvoPeq\.serena\project.yml` ✅ exists
- **Global config:** `C:\Users\user\.serena\serena_config.yml` ✅ exists
- **Language servers:** cpp, python, bash
- **Verified working:** `mcp_serena_get_symbols_overview` successfully returned symbols from `src/MainApplication.h`, `src/MainWindow.cpp`, `src/ConvolverProcessor.h`
- **Tools currently disabled:** `find_symbol`, `find_referencing_symbols`, `get_diagnostics_for_file`, `search_for_pattern`, `onboarding` (disabled by user/environment in VS Code Copilot Chat)

### AiDex MCP Server
- **Package:** `aidex-mcp` v2.3.0 (npm global)
- **Entry point:** `C:\Users\user\AppData\Roaming\npm\node_modules\aidex-mcp\build\index.js`
- **Index DB:** `C:\VSC_Project\ConvoPeq\.aidex\index.db` (25.5MB) ✅ exists with WAL/SHM files
- **Graphify:** `C:\VSC_Project\ConvoPeq\graphify-out\` ✅ exists (graph.json: 45.7MB, 44,244 nodes)
- **Status:** Registered in both `.vscode/mcp.json` and VS Code user settings, but MCP tools show "currently disabled by the user" in VS Code Copilot Chat. The server is installed and the index exists, but tool availability may require a VS Code restart or MCP server reconnection.

### Context-Mode MCP Server
- **Type:** STDIO MCP server
- **Source:** `.vscode/mcp.json` (workspace-level)
- **Available tools:** ✅ All context-mode tools verified working (`ctx_execute`, `ctx_batch_execute`, `ctx_execute_file`)

### Headroom MCP Server
- **Config:** `tools/headroom-mcp.cmd`
- **Available tools:** Should provide `headroom_compress`, `headroom_retrieve`, `headroom_stats`

---

## ✅ Graphify

- **CLI:** `graphify.exe` v0.9.48 (`C:\Users\user\AppData\Roaming\Python\Python314\Scripts\graphify.exe`)
- **Knowledge graph:** `graphify-out/graph.json` (45.7MB, 44,244 nodes) ✅
- **Graph report:** `graphify-out/GRAPH_REPORT.md` (990KB) ✅
- **Verified:** `graphify query "MainWindow"` successfully returned 131 nodes with BFS depth=2 traversal

### Usage
```bash
graphify query "MainWindow"          # BFS search
graphify query "function" --depth 3  # Deeper traversal
graphify path "src/A.h" "src/B.cpp"  # Shortest path between nodes
graphify explain "architecture"      # Concept explanation
```

---

## Summary

| Category | Tools | Status |
|----------|-------|--------|
| WSL search tools | grep, rg, ast-grep, fdfind, fzf, sed, awk, ag | ✅ All available |
| RTK | Token-optimized CLI wrapper | ✅ Available (WSL) |
| Windows CLI | ccc (cocoindex), semble, graphify, serena | ✅ All installed & functional |
| MCP: serena | Code symbol navigation | ✅ Server running, partial tool availability |
| MCP: context-mode | File analysis & parallel execution | ✅ Fully working |
| MCP: headroom | Context compression | ✅ Configured |
| MCP: aidex | Persistent code index | ⚠️ Installed & indexed, tools disabled in VS Code |

**Key notes:**
1. **AiDex MCP tools are disabled** in VS Code Copilot Chat despite being configured. The server is installed (v2.3.0), the index exists (25.5MB), and the graphify knowledge graph is present (45.7MB). This likely needs a VS Code restart or MCP server reconnection to activate.
2. **Serena MCP tools are partially available** — `get_symbols_overview` works but `find_symbol` and others are disabled. The server is running (v1.7.0).
3. All **WSL search tools and RTK** are fully functional and ready for use.
