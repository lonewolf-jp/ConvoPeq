# Tool Versions — 2026-09-26 Update

## Updated Tools

### 1. semble (v0.6.1)
- **Package**: `semble[mcp]==0.6.1` (via uvx)
- **Repo**: https://github.com/MinishLab/semble
- **Installation**: `uvx --from "semble[mcp]==0.6.1" semble`
- **Verify**: `uvx --from "semble[mcp]==0.6.1" semble --version` → `0.6.1`
- **Note**: Not installed as a persistent CLI; run via uvx. Python import: `import semble`

### 2. graphify (v0.9.69)
- **Package**: `graphifyy` (pip/npm)
- **Repo**: https://github.com/safishamsi/graphify
- **Updated from**: v0.9.64 → v0.9.69
- **Skill integrations updated**:
  - `C:\Users\user\.copilot\skills\graphify\SKILL.md` (Copilot)
  - `C:\Users\user\.claude\skills\graphify\SKILL.md` (Claude)
  - CLAUDE.md registered in both skill locations
- **Verify**: `graphify --version` → `0.9.69`
- **New features in v0.9.69**: (check release notes)
- **Usage**: `graphify update .` for AST-only index refresh, `graphify query "<question>"` for graph queries
- **Knowledge graph**: 56,581 nodes, 70,876 edges, 2,919 communities (AST-only rebuild complete)

### 3. AiDex MCP (v2.4.0)
- **Package**: `aidex-mcp@2.4.0` (npm global)
- **Repo**: https://github.com/CSCSoftware/AiDex
- **Updated from**: v2.3.0 → v2.4.0
- **Verify**: `npm ls -g aidex-mcp` → `aidex-mcp@2.4.0`
- **Usage**: MCP server only (no standalone CLI). Indexing via `mcp_aidex_aidex_init` tool.
- **Project re-indexed**: ConvoPeq — 480 files, 82,332 items, 8,037 methods, 714 types
- **Global index**: 3 projects, 1,440 files, 24,111 methods, 2,142 types
- **Note**: Global index has duplicate ConvoPeq entries (path variant `c:/` vs `C:/`). This is cosmetic — all entries show identical counts.

## Code Search Tool Priority (this project)
1. AiDex (`aidex_query`) — semantic/identifier search
2. graphify (`graphify query/path/explain`) — knowledge graph
3. semble — natural language code search
4. tgrep — trigram regex (AST-aware)
5. ctx_execute (sandbox) — code analysis/aggregation
6. rtk grep — fallback

## Verification Status (2026-09-27)
- semple v0.6.1: ✅ Working (uvx)
- graphify v0.9.69: ✅ Working (skills updated)
- AiDex MCP v2.4.0: ✅ Working (query returns results)
