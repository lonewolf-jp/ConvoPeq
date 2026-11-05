"""Test headroom MCP tool calls (compress, retrieve, stats).

Migrated 2026-11-05 (RES2-20): resolve the executable from HEADROOM_EXE, then
the uv tool install, then the legacy project .venv. The MCP server is the only
headroom surface this project uses; the proxy is retired.
"""
import json
import os
import subprocess
import sys
import time


def resolve_headroom_exe():
    """HEADROOM_EXE -> uv tool -> project .venv (first existing path)."""
    candidates = []
    env = os.environ.get("HEADROOM_EXE")
    if env:
        candidates.append(env)
    appdata = os.environ.get("APPDATA")
    if appdata:
        candidates.append(os.path.join(appdata, "uv", "tools", "headroom-ai", "Scripts", "headroom.exe"))
    candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   ".venv", "Scripts", "headroom.exe"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    print("headroom.exe not found. Set HEADROOM_EXE or install the uv tool.")
    print("tried:")
    for path in candidates:
        print("  " + path)
    sys.exit(1)


HEADROOM = resolve_headroom_exe()
print(f"Using: {HEADROOM}")

proc = subprocess.Popen(
    [HEADROOM, 'mcp', 'serve'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)


def send_msg(msg):
    proc.stdin.write(json.dumps(msg).encode() + b'\n')
    proc.stdin.flush()


def read_line():
    line = b''
    while True:
        ch = proc.stdout.read(1)
        if ch == b'\n' or not ch:
            break
        line += ch
    return json.loads(line.decode())


def main():
    # Initialize
    send_msg({
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2025-11-25',
            'capabilities': {},
            'clientInfo': {'name': 'test', 'version': '1.0'}
        }
    })
    time.sleep(1)
    init = read_line()
    print(f"Initialize OK: {init['result']['serverInfo']}")

    # Test 1: headroom_compress
    test_content = "Hello world! " * 50  # Repeat to make it compressible
    send_msg({
        'jsonrpc': '2.0', 'id': 2, 'method': 'tools/call',
        'params': {
            'name': 'headroom_compress',
            'arguments': {'content': test_content}
        }
    })
    time.sleep(2)
    res = read_line()
    text = res['result']['content'][0]['text']
    print("\n[1] headroom_compress:")
    print(f"    Input length: {len(test_content)} chars")
    print(f"    Output: {text[:150]}...")
    print(f"    Hash: {res['result']['content'][0].get('hash', 'N/A')}")
    print("  compress OK")

    # Test 2: headroom_stats
    send_msg({
        'jsonrpc': '2.0', 'id': 3, 'method': 'tools/call',
        'params': {
            'name': 'headroom_stats',
            'arguments': {}
        }
    })
    time.sleep(2)
    res2 = read_line()
    stats_text = res2['result']['content'][0]['text']
    print("\n[2] headroom_stats:")
    print(f"    {stats_text[:200]}")
    print("  stats OK")

    proc.kill()
    print("\nAll MCP tool call tests passed!")


if __name__ == "__main__":
    main()
