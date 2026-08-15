#!/usr/bin/env python3
"""Drive an MCP stdio server with a single tools/call via JSON-RPC.

Usage:
    python tools/mcp_query.py "<server-cmd>..." '<tool-name>:<json-args>' ['<tool2>:<json-args>' ...]

Examples:
    python tools/mcp_query.py "serena start-mcp-server" \
        'activate_project:{"project": "ConvoPeq"}' \
        'search_for_pattern:{"substring_pattern": "ownerThreadId"}'
    python tools/mcp_query.py "C:/Users/user/AppData/Roaming/npm/aidex.cmd" \
        'aidex_query:{"path": "C:/VSC_Project/ConvoPeq", "term": "fetchAddAtomic", "mode": "contains", "limit": 20}'
"""
import json
import subprocess
import sys
import threading
import time

def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    server_cmd = sys.argv[1]
    # Remaining args: "tool:jsonargs" pairs, optionally prefixed by "<pre>:"
    calls = []
    for a in sys.argv[2:]:
        if ":" in a:
            t, _, j = a.partition(":")
            try:
                calls.append((t, json.loads(j)))
            except json.JSONDecodeError as e:
                print(f"invalid JSON args for {t}: {e}")
                return 2
        else:
            print(f"expected tool:jsonargs, got: {a}")
            return 2

    # server_cmd is a full command line (may contain spaces); split on Windows/Git-Bash
    cmd = server_cmd.split()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def reader(pipe, sink):
        for line in pipe:
            sink.append(line)

    err_lines: list[str] = []
    t_err = threading.Thread(target=reader, args=(proc.stderr, err_lines), daemon=True)
    t_err.start()

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "codebuff-mcp-query", "version": "0.1"},
    }})

    # wait for initialize response
    deadline = time.time() + 60
    init_done = False
    out_line = proc.stdout.readline()
    while out_line and time.time() < deadline:
        try:
            msg = json.loads(out_line)
            if msg.get("id") == 0:
                init_done = True
                break
        except json.JSONDecodeError:
            pass
        out_line = proc.stdout.readline()

    if not init_done:
        print("initialize timed out / failed")
        print("stderr:", "".join(err_lines[-20:]))
        proc.kill()
        return 1

    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    time.sleep(0.5)

    for idx, (tool, args) in enumerate(calls, start=1):
        send({"jsonrpc": "2.0", "id": idx, "method": "tools/call",
              "params": {"name": tool, "arguments": args}})

        deadline = time.time() + 240
        got = False
        while time.time() < deadline:
            out_line = proc.stdout.readline()
            if not out_line:
                time.sleep(0.2)
                if proc.poll() is not None:
                    break
                continue
            try:
                msg = json.loads(out_line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == idx:
                got = True
                result = msg.get("result", {})
                content = result.get("content", [])
                for c in content:
                    if c.get("type") == "text":
                        print(c.get("text", ""))
                if result.get("isError"):
                    print(f"--- TOOL ERROR ({tool}) ---", file=sys.stderr)
                break

        if not got:
            print(f"tools/call timed out ({tool})")
            print("stderr:", "".join(err_lines[-30:]))
            proc.kill()
            return 1

    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return 0

if __name__ == "__main__":
    sys.exit(main())
