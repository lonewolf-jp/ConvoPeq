"""Find headroom processes and check the proxy port.

Migrated 2026-11-05 (RES2-20): the psutil dependency is gone. Process data comes
from CIM through Windows PowerShell, so the script runs on the stock Python.
Detection covers process name, command line and the 8787 port. Open-file handles
are not enumerated: CIM does not expose them, and the proxy is retired anyway.
"""
import json
import os
import socket
import subprocess

CIM_QUERY = (
    "Get-CimInstance Win32_Process | "
    "Where-Object { $_.Name -eq 'headroom.exe' -or $_.CommandLine -match 'headroom\\.exe' } | "
    "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
)


def list_processes():
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", CIM_QUERY],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 - report and continue
        print(f"CIM query failed: {exc}")
        return []
    out = (completed.stdout or "").strip()
    if not out:
        return []
    try:
        rows = json.loads(out)
    except json.JSONDecodeError as exc:
        print(f"CIM output was not JSON: {exc}")
        return []
    return rows if isinstance(rows, list) else [rows]


def classify(cmd):
    low = (cmd or "").lower()
    if "mcp" in low:
        return "MCP"
    if "proxy" in low:
        return "PROXY"
    return "OTHER"


def check_port(port=8787):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
    finally:
        sock.close()
    return result == 0


def main():
    rows = list_processes()
    own_pid = os.getpid()
    matches = []
    for row in rows:
        pid = row.get("ProcessId")
        name = (row.get("Name") or "").lower()
        cmd = row.get("CommandLine") or ""
        # skip this script and the PowerShell helper that ran the CIM query
        if pid == own_pid:
            continue
        if name in ("powershell.exe", "pwsh.exe") and "Get-CimInstance" in cmd:
            continue
        matches.append(row)

    if not matches:
        print("No headroom processes found via CIM")
    for row in matches:
        cmd = row.get("CommandLine") or ""
        print(f"{classify(cmd)} MATCH: PID={row.get('ProcessId')} NAME={row.get('Name')} CMD={cmd[:200]}")
    print(f"Port 8787 connect test: {'OPEN' if check_port() else 'CLOSED'}")


if __name__ == "__main__":
    main()
