import { spawn } from "node:child_process";
import http from "node:http";

const PORT = 8787;
const BASE = `http://127.0.0.1:${PORT}`;
const PID_FILE = process.env.TEMP + "\\headroom-proxy.pid";
const HEALTH_URL = `${BASE}/health`;

let proxy = null;
let healthTimer = null;

function healthCheck() {
  return new Promise((resolve) => {
    const req = http.get(HEALTH_URL, { timeout: 3000 }, (res) => {
      let d = "";
      res.on("data", (c) => { d += c; });
      res.on("end", () => { resolve(res.statusCode === 200); });
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

function waitForHealth(timeoutMs = 10000, intervalMs = 400) {
  return new Promise((resolve) => {
    const start = Date.now();
    const poll = async () => {
      if (await healthCheck()) return resolve(true);
      if (Date.now() - start >= timeoutMs) return resolve(false);
      setTimeout(poll, intervalMs);
    };
    poll();
  });
}

function killExisting() {
  try {
    const pid = require("fs").readFileSync(PID_FILE, "ascii").trim();
    if (pid) { try { process.kill(parseInt(pid)); } catch {} }
  } catch {}
  try { require("fs").unlinkSync(PID_FILE); } catch {}
}

function startProxy() {
  if (proxy) return;
  const exe = ".venv\\Scripts\\headroom.exe";

  proxy = spawn(exe, [
    "proxy", "--port", String(PORT), "--host", "127.0.0.1",
    "--mode", "token", "--target-ratio", "0.40", "--memory",
    "--intercept-tool-results", "--rpm", "200", "--tpm", "500000",
    "--keepalive-expiry", "30", "--protect-tool-results", "Bash,WebFetch,Read",
  ], { stdio: ["ignore", "pipe", "pipe"], env: { ...process.env, HEADROOM_ROLLOUT_CHANNEL: "canary" } });

  const log = (m) => console.error(`[headroom:${PORT}] ${m}`);
  proxy.stdout?.on("data", (d) => { const s = d.toString().trim(); if (s) log(s); });
  proxy.stderr?.on("data", (d) => { const s = d.toString().trim(); if (s) log(s); });
  proxy.on("error", (e) => { log(`Error: ${e.message}`); proxy = null; healthTimer || scheduleHealthCheck(); });
  proxy.on("exit", (c) => { log(`Exited (${c})`); proxy = null; healthTimer || scheduleHealthCheck(); });

  try { require("fs").writeFileSync(PID_FILE, String(proxy.pid), "ascii"); } catch {}
}

async function ensureProxy() {
  if (proxy && await healthCheck()) return true;
  proxy = null;
  killExisting();
  startProxy();
  return await waitForHealth();
}

function scheduleHealthCheck() {
  healthTimer = setInterval(async () => {
    if (proxy && await healthCheck()) return;
    const ok = await ensureProxy();
    if (!ok) { console.error(`[headroom:${PORT}] restart failed, retrying...`); }
  }, 30000);
}

export default async () => {
  killExisting();
  startProxy();
  const ready = await waitForHealth();

  if (ready) {
    process.env.ANTHROPIC_BASE_URL = BASE;
    process.env.OPENAI_BASE_URL = `${BASE}/v1`;
    console.error(`[headroom:${PORT}] healthy, ANTHROPIC_BASE_URL → ${BASE}`);
  } else {
    console.error(`[headroom:${PORT}] startup timeout — direct API will be used`);
    proxy?.kill();
    proxy = null;
  }

  scheduleHealthCheck();
  const cleanup = () => {
    if (healthTimer) clearInterval(healthTimer);
    if (proxy) { proxy.kill(); proxy = null; }
    try { require("fs").unlinkSync(PID_FILE); } catch {}
  };
  process.on("exit", cleanup);
  process.on("SIGINT", cleanup);
  process.on("SIGTERM", cleanup);

  return {};
};
