# ★ context-mode — ABSOLUTE MANDATORY routing rules

**このプロジェクトでは、データ処理・ファイル読み取り・Web取得のすべてに context-mode MCP ツールを最優先で使用すること。**
**通常の Read/Grep/WebFetch は context-mode が使えない場合の最終手段とする。**

context-mode MCP tools available. Rules protect context window from flooding. One unrouted command dumps 56 KB into context.

## Think in Code — MANDATORY

Analyze/count/filter/compare/search/parse/transform data: **write code** via `ctx_execute(language, code)`, `console.log()` only the answer. Do NOT read raw data into context. PROGRAM the analysis, not COMPUTE it. Pure JavaScript — Node.js built-ins only (`fs`, `path`, `child_process`). `try/catch`, handle `null`/`undefined`. One script replaces ten tool calls.

## BLOCKED — do NOT attempt

### curl / wget — BLOCKED
Terminal `curl`/`wget` intercepted and blocked. Do NOT retry.
Use: `ctx_fetch_and_index(url, source)` or `ctx_execute(language: "javascript", code: "const r = await fetch(...)")`

### Inline HTTP — BLOCKED
`fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, `http.request(` — intercepted. Do NOT retry.
Use: `ctx_execute(language, code)` — only stdout enters context

### WebFetch / fetch — BLOCKED
Use: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)`

## REDIRECTED — use sandbox

### Terminal / run_in_terminal (>20 lines output)
Terminal ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`.
Otherwise: `ctx_batch_execute(commands, queries)` or `ctx_execute(language: "javascript", code: "...")`. Use `language: "shell"` only when code matches the host shell.

### read_file (for analysis)
Reading to **edit** → read_file correct. Reading to **analyze/explore/summarize** → `ctx_execute_file(path, language, code)`.

### grep / search (large results)
Use `ctx_execute(language: "javascript", code: "...")` in sandbox for portable filtering/counting.

## Tool selection

0. **MEMORY**: `ctx_search(sort: "timeline")` — after resume, check prior context before asking user.
1. **GATHER**: `ctx_batch_execute(commands, queries)` — runs all commands, auto-indexes, returns search. ONE call replaces 30+. Each command: `{label: "header", command: "..."}`.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — all questions as array, ONE call (default relevance mode).
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — sandbox, only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — store in FTS5 for later search.

### Parallel I/O batches
Pass `concurrency: 4-8` to `ctx_batch_execute` and `ctx_fetch_and_index` for network/API batches. Keep `concurrency: 1` for CPU-bound work (test, build, lint). GitHub gh: cap at 4.

## Output

Write artifacts to FILES — never inline. Return: file path + 1-line description.
Descriptive source labels for `ctx_search(source: "label")`.

## Session Continuity

Skills, roles, and decisions persist for the entire session. Do not abandon them as the conversation grows.

## Memory

Session history is persistent and searchable. On resume, search BEFORE asking the user:

| Need | Command |
|------|---------|
| "what did we decide about X" | `ctx_search(queries: ["decision about X"], source: "decision", sort: "timeline")` |
| "what errors did we hit" | `ctx_search(queries: ["error"], source: "error", sort: "timeline")` |
| "what was the user's last request" | `ctx_search(queries: ["last user prompt"], source: "user-prompt", sort: "timeline")` |
| "what's the current plan" | `ctx_search(queries: ["plan"], source: "plan", sort: "timeline")` |

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call `ctx_stats` MCP tool, display full output verbatim |
| `ctx doctor` | Call `ctx_doctor` MCP tool, run returned shell command, display as checklist |
| `ctx upgrade` | Call `ctx_upgrade` MCP tool, run returned shell command, display as checklist |
| `ctx purge` | Call `ctx_purge` MCP tool with confirm: true. Warns before wiping knowledge base. |

After /clear or /compact: knowledge base and session stats preserved. Use `ctx purge` to start fresh.

<!-- rtk-instructions v2 -->
# RTK — Token-Optimized CLI

**rtk** is a CLI proxy that filters and compresses command outputs, saving 60-90% tokens.

## Rule

Always prefix shell commands with `rtk`:

```bash
# Instead of:              Use:
git status                 rtk git status
git log -10                rtk git log -10
cargo test                 rtk cargo test
docker ps                  rtk docker ps
kubectl get pods           rtk kubectl pods
npm test                   rtk npm test
grep pattern               rtk grep pattern
cat file.txt               rtk read file.txt
```

## Supported Commands (100+)

### Git
`rtk git status`, `rtk git diff`, `rtk git log`, `rtk git add/commit/push`

### File Operations
`rtk ls`, `rtk tree`, `rtk read`, `rtk find`, `rtk wc`

### Search
`rtk grep`, `rtk rg` (ripgrep)

### Development
`rtk npm`, `rtk npx`, `rtk cargo`, `rtk go`, `rtk dotnet`, `rtk gradlew`, `rtk mvn`

### Testing
`rtk test`, `rtk jest`, `rtk vitest`, `rtk pytest`, `rtk rspec`, `rtk rake`

### Linting / Type Checking
`rtk lint`, `rtk tsc`, `rtk ruff`, `rtk mypy`, `rtk rubocop`, `rtk prettier`, `rtk format`, `rtk golangci-lint`

### Infrastructure
`rtk docker`, `rtk kubectl`, `rtk aws`, `rtk gh`, `rtk glab`

### Database
`rtk psql`, `rtk prisma`

### Utility
`rtk curl`, `rtk wget`, `rtk json`, `rtk log`, `rtk deps`, `rtk env`, `rtk diff`, `rtk summary`, `rtk err`, `rtk smart`, `rtk pip`, `rtk pnpm`, `rtk next`, `rtk playwright`

## Meta commands

```bash
rtk gain                    # Token savings dashboard
rtk gain --history          # Per-command savings history
rtk gain --daily            # Daily breakdown
rtk gain --graph            # ASCII graph (last 30 days)
rtk gain --all --format json  # JSON export
rtk discover                # Find missed rtk opportunities
rtk discover --all --since 7  # All projects, last 7 days
rtk session                 # Show RTK adoption across sessions
rtk proxy <cmd>             # Run raw (no filtering) but track usage
rtk init --show             # Verify installation
rtk config                  # Show configuration
rtk verify                  # Verify hook integrity
rtk cc-economics            # Claude Code economics analysis
```

## Ultra-Compact Mode

Add `--ultra-compact` or `-u` for maximum token savings:
```bash
rtk -u git status
rtk -u npm test
```

## Windows Notes

On native Windows (cmd.exe/PowerShell), RTK filters work fully but the auto-rewrite hook requires WSL. Always manually prefix commands with `rtk`. The hook file at `.github/hooks/rtk-rewrite.json` provides Copilot integration.

## Current Savings

RTK has saved **25.0M tokens (98.8%)** across 1,774 commands in this environment.
<!-- /rtk-instructions -->
