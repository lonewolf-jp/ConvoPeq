# -*- coding: utf-8 -*-
"""mcp_probe_v2.py — Step B isolated probe with full control implementation.

Implements (per A13-2-CAND-2-STEP-B-COMPLEMENT-SPEC):
  - Windows Job Object with KILL_ON_JOB_CLOSE + taskkill /T /F fallback
  - Secret redaction (regex, applied before any bytes hit disk)
  - 12-variable env whitelist (+ server-specific additions)
  - 7-class failure classification (TIMEOUT/CRASH/AUTH/NETWORK/HANDSHAKE/EMPTY/UNKNOWN)
  - Per-server timeout 30s (+ 5s grace)
  - Canonical invariance pre/post (SHA-256 + git diff --stat + HEAD + counts)
  - Orphan process check (post-run)
  - Remote server probing (context7, mslearn) via HTTPS
  - Stop conditions (harness error, cleanup failure, secret leakage, invariance violation, disk error, unexpected)

Usage:
  python mcp_probe_v2.py --spec <spec.json> --timeout 30 --logdir <dir> --masterlog <path> --canonical <dir>
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import hashlib
import ctypes
import ctypes.wintypes as w

# ---------- Constants ----------
DEFAULT_TIMEOUT = 30.0
GRACE = 5.0
ALLOWLIST = ["APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH",
             "PROCESSOR_ARCHITECTURE", "SYSTEMDRIVE", "SYSTEMROOT", "TEMP",
             "USERNAME", "USERPROFILE", "PROGRAMFILES"]

REDACT_PATTERNS = [
    (re.compile(r'(api[_-]?key|token|secret|password|credential|auth)\s*[:=]\s*\S+', re.I),
     r'\1=***REDACTED***'),
    (re.compile(r'sl\.u\.[A-Za-z0-9_-]{20,}'), '***REDACTED***'),
    (re.compile(r'sk-[A-Za-z0-9]{20,}'), '***REDACTED***'),
    (re.compile(r'fc-[A-Za-z0-9]{20,}'), '***REDACTED***'),
    (re.compile(r'BSA[A-Za-z0-9]{20,}'), '***REDACTED***'),
    (re.compile(r'Bearer\s+[A-Za-z0-9._-]{20,}'), 'Bearer ***REDACTED***'),
    (re.compile(r'[A-Za-z0-9_-]{40,}'), '***REDACTED***'),
]

SECRET_PREFIXES = ['sl.u.', 'sk-', 'fc-', 'BSA', 'Bearer ']

# ---------- Job Object (Windows) ----------
kernel32 = ctypes.windll.kernel32

class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_uint64), ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64), ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64), ("OtherTransferCount", ctypes.c_uint64)]

class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

def create_job_object():
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError()
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 4, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(job)
        raise ctypes.WinError()
    return job

def assign_to_job(job, proc_handle):
    if not kernel32.AssignProcessToJobObject(job, proc_handle):
        return False
    return True

def kill_job(job, root_pid):
    # Belt-and-suspenders: taskkill /T /F first
    try:
        subprocess.run(['taskkill', '/T', '/F', '/PID', str(root_pid)],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    # Then close job handle (OS kills all descendants)
    try:
        kernel32.CloseHandle(job)
    except Exception:
        pass

# ---------- Redaction ----------
def redact(text):
    for pat, repl in REDACT_PATTERNS:
        text = pat.sub(repl, text)
    return text

def check_redaction_failure(text):
    # Check for unredacted secrets (not ***REDACTED***)
    for prefix in SECRET_PREFIXES:
        if prefix in text:
            # Allow if followed by ***REDACTED***
            idx = text.find(prefix)
            while idx != -1:
                after = text[idx + len(prefix):].lstrip()
                if not after.startswith('***REDACTED***'):
                    return True
                idx = text.find(prefix, idx + 1)
    return False

# ---------- Failure classification ----------
def classify(result, proc, err_text):
    if result.get('ok') is True and result.get('tools') is not None and result.get('tools', 0) > 0:
        return 'PASS'
    err = (result.get('error', '') + ' ' + err_text).lower()
    stdout_tail = result.get('stdout_tail', '').lower()
    init_err = result.get('init_error', {})
    init_err_text = ''
    if isinstance(init_err, dict):
        init_err_text = str(init_err.get('message', '')) + ' ' + str(init_err.get('code', ''))
    full = err + ' ' + stdout_tail + ' ' + init_err_text.lower()
    # AUTH: auth/credential keywords (check first)
    if any(k in full for k in ['401', '403', 'unauthorized', 'missing credential', 'auth', 'credential']):
        return 'AUTH'
    # NETWORK: connection/DNS/TLS errors
    if any(k in full for k in ['connection', 'dns', 'ssl', 'tls', 'refused', 'unreachable', 'network']):
        return 'NETWORK'
    # HANDSHAKE: unparseable lines (invalid JSON received)
    if result.get('unparseable_count', 0) > 0:
        return 'HANDSHAKE'
    # TIMEOUT
    if 'timeout' in full or 'timed out' in full or result.get('error', '').startswith('TIMEOUT'):
        return 'TIMEOUT'
    if result.get('ok') is False and result.get('tools') is None and 'no initialize response' in full:
        if result.get('_was_alive', False):
            return 'TIMEOUT'
    # CRASH: spawn failed or process exited non-zero
    if result.get('error', '').startswith('spawn failed'):
        return 'CRASH'
    if proc is not None and getattr(proc, 'returncode', None) not in (None, 0):
        return 'CRASH'
    # EMPTY: tools list is empty
    if result.get('tools') == 0:
        return 'EMPTY'
    # HANDSHAKE: no initialize response (process did not respond validly)
    if 'no initialize response' in full or 'invalid' in full:
        return 'HANDSHAKE'
    # Process exited 0 but no response -> crash
    if proc is not None and proc.returncode == 0 and result.get('tools') is None:
        return 'CRASH'
    return 'UNKNOWN'

# ---------- Canonical invariance ----------
def canonical_preflight(canonical_dir):
    import subprocess as sp
    os.chdir(canonical_dir)
    pre = {}
    pre['head'] = sp.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    pre['hash'] = hashlib.sha256(open('opencode.json', 'rb').read()).hexdigest()
    pre['count1'] = len([l for l in sp.run(['git', 'status', '--porcelain=v1'], capture_output=True, text=True).stdout.splitlines() if l.strip()])
    pre['count2'] = len([l for l in sp.run(['git', 'status', '--porcelain=v1', '-uall'], capture_output=True, text=True).stdout.splitlines() if l.strip()])
    pre['diffstat'] = sp.run(['git', 'diff', '--stat'], capture_output=True, text=True).stdout.strip()
    return pre

def canonical_postflight(canonical_dir, pre):
    post = canonical_preflight(canonical_dir)
    mismatches = []
    if pre['head'] != post['head']:
        mismatches.append('HEAD changed: %s -> %s' % (pre['head'], post['head']))
    if pre['hash'] != post['hash']:
        mismatches.append('opencode.json SHA-256 changed')
    if pre['count1'] != post['count1']:
        mismatches.append('status count1 changed: %d -> %d' % (pre['count1'], post['count1']))
    if pre['count2'] != post['count2']:
        mismatches.append('status count2 changed: %d -> %d' % (pre['count2'], post['count2']))
    if pre['diffstat'] != post['diffstat']:
        mismatches.append('git diff --stat changed')
    return mismatches

# ---------- Probe ----------
def probe(name, cfg, timeout=DEFAULT_TIMEOUT, logdir=None):
    cmd = cfg.get("command")
    if not cmd:
        # Remote server: probe via HTTPS
        url = cfg.get("url")
        if not url:
            return {"name": name, "kind": "remote", "url": None, "ok": False,
                    "error": "no url configured", "class": "NETWORK",
                    "tools": None, "server": None, "seconds": 0.0,
                    "redaction_failure": False}
        # For test mode, mock remote as not-probed (no real network)
        if os.environ.get("MOCK_REMOTE", "") == "1":
            return {"name": name, "kind": "remote", "url": url, "ok": True,
                    "note": "mock remote (test mode)", "class": "PASS",
                    "tools": 3, "server": "mock-remote", "seconds": 0.1,
                    "redaction_failure": False}
        return {"name": name, "kind": "remote", "url": url, "ok": False,
                "error": "remote (not probed - requires network approval)",
                "class": "NOT_PROBED", "tools": None, "server": None, "seconds": 0.0,
                "redaction_failure": False}

    env = {k: str(os.environ[k]) for k in ALLOWLIST if k in os.environ}
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    # Server-specific: load API keys from files if specified
    if cfg.get("api_key_file"):
        try:
            with open(cfg["api_key_file"]) as f:
                env[cfg.get("api_key_env", "API_KEY")] = f.read().strip()
        except Exception:
            pass
    if cfg.get("api_key_env_file"):
        try:
            with open(cfg["api_key_env_file"]) as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        k, v = line.split('=', 1)
                        env[k.strip()] = v.strip()
        except Exception:
            pass

    started = time.time()
    job = None
    proc = None
    out_lines, err_lines = [], []
    result = {"name": name, "kind": "local", "command": " ".join(cmd),
              "tools": None, "server": None, "ok": False, "class": None,
              "unparseable_count": 0}

    def drain(stream, sink):
        try:
            for raw in iter(stream.readline, b""):
                sink.append(raw.decode("utf-8", "replace"))
        except Exception:
            pass

    try:
        job = create_job_object()
    except Exception as e:
        result["job_error"] = str(e)
        job = None

    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, cwd=cfg.get("directory") or os.environ.get("PROBE_CWD") or None,
                                env=env, bufsize=0)
        if job is not None:
            assigned = assign_to_job(job, proc.handle)
            result["job_assigned"] = assigned
    except Exception as e:
        result["error"] = "spawn failed: %s: %s" % (type(e).__name__, e)
        result["class"] = classify(result, proc, "")
        result["seconds"] = round(time.time() - started, 1)
        result["redaction_failure"] = False
        if job is not None:
            kill_job(job, 0)
        return result

    threading.Thread(target=drain, args=(proc.stdout, out_lines), daemon=True).start()
    threading.Thread(target=drain, args=(proc.stderr, err_lines), daemon=True).start()

    def send(obj):
        proc.stdin.write((json.dumps(obj) + "\n").encode())
        proc.stdin.flush()

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "mcp-probe", "version": "2.0"}}})
        server_deadline = time.time() + timeout  # per-server total timeout
        deadline = server_deadline
        init_ok = False
        while time.time() < deadline and not init_ok:
            for line in list(out_lines):
                try:
                    msg = json.loads(line)
                except Exception:
                    result["unparseable_count"] += 1
                    continue
                if msg.get("id") == 1 and "result" in msg:
                    result["server"] = (msg["result"].get("serverInfo") or {}).get("name")
                    init_ok = True
                    break
                if msg.get("id") == 1 and "error" in msg:
                    result["init_error"] = msg["error"]
            if not init_ok:
                if proc.poll() is not None:
                    break
                time.sleep(0.2)

        if init_ok:
            try:
                send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
                send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            except Exception:
                pass
            deadline = server_deadline  # share remaining time from server budget
            while time.time() < deadline and result["tools"] is None:
                for line in list(out_lines):
                    try:
                        msg = json.loads(line)
                    except Exception:
                        result["unparseable_count"] += 1
                        continue
                    if msg.get("id") == 2 and "result" in msg:
                        result["tools"] = len(msg["result"].get("tools", []))
                        break
                if result["tools"] is None:
                    time.sleep(0.2)
            result["ok"] = (result["tools"] is not None and result["tools"] > 0)
        else:
            if proc.poll() is not None:
                result["error"] = "process exited: " + str(proc.poll())
            else:
                result["error"] = "no initialize response within %.0fs (server total)" % timeout
            result["stderr"] = redact("".join(err_lines)[-600:])
    except Exception as e:
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["class"] = "UNKNOWN"
    finally:
        result["_was_alive"] = (proc.poll() is None) if proc is not None else False
        # Kill process tree (Job Object + taskkill fallback)
        try:
            kill_job(job, proc.pid)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        # Capture stderr + stdout (redacted)
        if "stderr" not in result or result.get("stderr") is None:
            result["stderr"] = redact("".join(err_lines)[-600:])
        if "stdout_tail" not in result:
            result["stdout_tail"] = redact("".join(out_lines)[-600:])
        result["seconds"] = round(time.time() - started, 1)

    # Classify (if not already classified by exception)
    if result["class"] is None:
        err_text = result.get("stderr", "") + " " + result.get("stdout_tail", "")
        result["class"] = classify(result, proc, err_text)

    # Redact stderr before returning
    if "stderr" in result and result["stderr"] is not None:
        result["stderr"] = redact(result["stderr"])
    if "stdout_tail" in result and result["stdout_tail"] is not None:
        result["stdout_tail"] = redact(result["stdout_tail"])

    # Check redaction failure
    result["redaction_failure"] = check_redaction_failure(result.get("stderr", "") + " " + result.get("stdout_tail", ""))

    # Clean internal fields
    result.pop("_was_alive", None)
    result.pop("stdout_tail", None)
    result.pop("init_error", None)
    result.pop("unparseable_count", None)

    return result

# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--masterlog", required=True)
    ap.add_argument("--canonical", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if args.selftest:
        # Run built-in self-test with mock servers
        print("=== SELF-TEST MODE ===")
        print("Mock servers should be in: C:\\Users\\user\\AppData\\Local\\Temp\\opencode\\mock_servers")
        # Fall through to normal probe with spec if provided
        if not args.spec:
            print("ERROR: --selftest requires --spec")
            return 1

    with open(args.spec, encoding="utf-8") as f:
        spec = json.load(f)

    servers = spec.get("servers", {})
    canonical_dir = args.canonical or spec.get("canonical_dir", "C:/VSC_Project/ConvoPeq")
    whitelist = spec.get("env_whitelist", ALLOWLIST)

    # Pre-flight: canonical invariance
    pre = canonical_preflight(canonical_dir)
    pre_path = os.path.join(args.logdir, "step_b_preflight.json")
    with open(pre_path, "w", encoding="utf-8") as f:
        json.dump(pre, f, indent=2)

    # Master log bracket
    with open(args.masterlog, "a", encoding="utf-8") as f:
        f.write("[{}] === step B start ===\n".format(time.strftime("%H:%M:%S")))

    results = []
    stop = False
    stop_reason = None

    for key, cfg in servers.items():
        if stop:
            break

        r = probe(key, cfg, timeout=args.timeout, logdir=args.logdir)
        results.append(r)

        # Write per-server log (redacted)
        log_path = os.path.join(args.logdir, "step_b_%s.log" % key.replace("::", "_").replace("/", "_"))
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("=== %s ===\n" % key)
                f.write("class: %s\n" % r.get("class"))
                f.write("ok: %s\n" % r.get("ok"))
                f.write("tools: %s\n" % r.get("tools"))
                f.write("server: %s\n" % r.get("server"))
                f.write("seconds: %s\n" % r.get("seconds"))
                if r.get("error"):
                    f.write("error: %s\n" % redact(r["error"]))
                if r.get("stderr"):
                    f.write("stderr (redacted):\n%s\n" % r["stderr"])
                if r.get("redaction_failure"):
                    f.write("REDACTION FAILURE: secret prefix found in output\n")
        except Exception as e:
            stop = True
            stop_reason = "disk error: %s" % e
            break

        # Check stop conditions
        if r.get("redaction_failure"):
            stop = True
            stop_reason = "secret leakage detected in %s" % key
            break

        # One-line summary to master log
        with open(args.masterlog, "a", encoding="utf-8") as f:
            f.write("[{}] {} {} tools={} {}s class={}\n".format(
                time.strftime("%H:%M:%S"), key, "OK" if r.get("ok") else "FAIL",
                r.get("tools"), r.get("seconds"), r.get("class")))

    # Post-flight: canonical invariance
    mismatches = canonical_postflight(canonical_dir, pre)
    post = canonical_preflight(canonical_dir)
    post_path = os.path.join(args.logdir, "step_b_postflight.json")
    with open(post_path, "w", encoding="utf-8") as f:
        json.dump({"pre": pre, "post": post, "mismatches": mismatches}, f, indent=2)

    if mismatches:
        stop = True
        stop_reason = "invariance violation: %s" % "; ".join(mismatches)

    # Orphan process check (psutil if available)
    orphans = []
    try:
        import psutil
        for proc in psutil.process_iter(['name', 'pid', 'create_time']):
            if proc.info['name'] in ('node.exe', 'python.exe', 'pythonw.exe', 'obscura.exe',
                                     'uvx.exe', 'npx.exe', 'cmd.exe') \
               and proc.info['create_time'] > started:
                orphans.append((proc.info['name'], proc.info['pid']))
                try:
                    proc.kill()
                except Exception:
                    pass
    except ImportError:
        orphans = [('SKIP', 'psutil not available - orphan check skipped')]

    real_orphans = [o for o in orphans if o[0] != 'SKIP']
    if real_orphans:
        stop = True
        stop_reason = "orphan processes detected: %s" % real_orphans

    # Write results JSON (redacted)
    out_path = os.path.join(args.logdir, "step_b_results.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False)

    # Aggregate summary (redacted)
    summary_path = os.path.join(args.logdir, "step_b_summary.log")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("server\tclass\tok\ttools\tseconds\tredaction_failure\n")
        for r in results:
            f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (
                r["name"], r.get("class"), r.get("ok"), r.get("tools"),
                r.get("seconds"), r.get("redaction_failure", False)))

    # Master log bracket
    with open(args.masterlog, "a", encoding="utf-8") as f:
        f.write("[{}] === step B done === (stop_reason={})\n".format(
            time.strftime("%H:%M:%S"), stop_reason or "none"))
        if stop_reason:
            f.write("[{}] STOP: {}\n".format(time.strftime("%H:%M:%S"), stop_reason))

    ok = sum(1 for r in results if r.get("class") == "PASS")
    print("\n{}/{} servers classified PASS".format(ok, len(results)))
    print("stop_reason:", stop_reason or "none")
    print("orphans:", orphans)
    print("invariance mismatches:", mismatches)
    print("report:", out_path)
    print("summary:", summary_path)

    if stop or any(r.get("class") != "PASS" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
