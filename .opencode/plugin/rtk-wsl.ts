export default async ({ $ }) => ({
  "tool.execute.before": async (input, output) => {
    const tool = String(input?.tool ?? "").toLowerCase();
    if (tool !== "bash" && tool !== "shell") return;
    const args = output?.args;
    if (!args || typeof args !== "object") return;
    const command = (args as Record<string, unknown>).command;
    if (typeof command !== "string" || !command) return;

    const match = command.match(/^wsl\s+(?:bash\s+-[lc]\s*["']?)?(.+?)["']?\s*(?:2>&1)?\s*$/i);
    if (!match) return;

    try {
      const inner = match[1].replace(/["']$/, "").trim();
      const result = await $`wsl bash -lc "rtk rewrite ${inner} 2>/dev/null"`.quiet().nothrow();
      const rewritten = String(result.stdout).trim();
      if (rewritten && rewritten !== inner) {
        const q = command.includes('"') ? "'" : '"';
        (args as Record<string, unknown>).command = `wsl bash -lc ${q}${rewritten}${q}`;
      }
    } catch {}
  },
});
