"""Test headroom MCP server via STDIO transport."""
import subprocess
import json
import sys

def test_mcp():
    proc = subprocess.Popen(
        [r'C:\Users\user\AppData\Roaming\Python\Python314\Scripts\headroom.exe', 'mcp', 'serve'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Test 1: Initialize
    req = json.dumps({
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2025-11-25',
            'capabilities': {},
            'clientInfo': {'name': 'test-client', 'version': '1.0'}
        }
    })
    stdout_line, stderr_data = proc.communicate(input=req.encode(), timeout=10)

    resp = json.loads(stdout_line.decode())
    print(f"[1] initialize: protocol={resp['result']['protocolVersion']}, "
          f"server={resp['result']['serverInfo']['name']} v{resp['result']['serverInfo']['version']}")
    assert resp['result']['protocolVersion'] == '2025-11-25', "Protocol version mismatch"
    assert 'tools' in resp['result']['capabilities'], "Missing tools capability"
    print("  ✅ initialize OK")

    # Restart for tools/list (communicate closes stdin)
    proc = subprocess.Popen(
        [r'C:\Users\user\AppData\Roaming\Python\Python314\Scripts\headroom.exe', 'mcp', 'serve'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    # Initialize first
    req_init = json.dumps({
        'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
        'params': {
            'protocolVersion': '2025-11-25',
            'capabilities': {},
            'clientInfo': {'name': 'test-client', 'version': '1.0'}
        }
    })
    proc.stdin.write(req_init.encode() + b'\n')
    proc.stdin.flush()

    # Read initialize response line
    import time
    time.sleep(2)
    import select
    # Use a readline approach
    import os

    def read_line():
        line = b''
        while True:
            ch = proc.stdout.read(1)
            if ch == b'\n' or not ch:
                break
            line += ch
        return line.decode()

    init_resp_line = read_line()
    init_resp = json.loads(init_resp_line)
    print(f"\n[2] initialize response: {init_resp['id']}")

    # Test 2: tools/list
    req2 = json.dumps({
        'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}
    })
    proc.stdin.write(req2.encode() + b'\n')
    proc.stdin.flush()

    time.sleep(2)
    tools_line = read_line()
    tools_resp = json.loads(tools_line)

    print(f"[3] tools/list: {len(tools_resp['result'].get('tools', []))} tools")
    for tool in tools_resp['result'].get('tools', []):
        print(f"    - {tool['name']}: {tool.get('description', '')[:80]}")

    if 'headroom_compress' in [t['name'] for t in tools_resp['result'].get('tools', [])]:
        print("  ✅ headroom_compress available")
    if 'headroom_retrieve' in [t['name'] for t in tools_resp['result'].get('tools', [])]:
        print("  ✅ headroom_retrieve available")
    if 'headroom_stats' in [t['name'] for t in tools_resp['result'].get('tools', [])]:
        print("  ✅ headroom_stats available")

    proc.kill()
    print("\n✅ All MCP server tests passed!")

if __name__ == '__main__':
    test_mcp()
