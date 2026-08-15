#!/usr/bin/env python3
"""List MCP tools (with JSON schema) from a stdio server, optionally filtering by name.

Usage:
    python tools/mcp_list_tools.py "<server-cmd>..." [name-substring]
"""
import json
import subprocess
import sys
import time

def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    server_cmd = sys.argv[1].split()
    filt = sys.argv[2] if len(sys.argv) > 2 else ""

    proc = subprocess.Popen(server_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
        "protocolVersion": "2024-11-05", "capabilities": {},
        "clientInfo": {"name": "mcp-list", "version": "0.1"}}})

    deadline = time.time() + 60
    init_done = False
    while time.time() < deadline:
        out = proc.stdout.readline()
        if not out:
            time.sleep(0.2)
            continue
        try:
            msg = json.loads(out)
            if msg.get("id") == 0:
                init_done = True
                break
        except json.JSONDecodeError:
            continue
    if not init_done:
        print("initialize timed out")
        proc.kill()
        return 1

    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
    send({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})

    deadline = time.time() + 60
    while time.time() < deadline:
        out = proc.stdout.readline()
        if not out:
            time.sleep(0.2)
            continue
        try:
            msg = json.loads(out)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == 1:
            for t in msg.get("result", {}).get("tools", []):
                name = t.get("name", "")
                if filt and filt not in name:
                    continue
                print(f"== {name} ==")
                print(json.dumps(t.get("inputSchema", {}), indent=1, ensure_ascii=False)[:3000])
            proc.kill()
            return 0
    print("tools/list timed out")
    proc.kill()
    return 1

if __name__ == "__main__":
    sys.exit(main())
