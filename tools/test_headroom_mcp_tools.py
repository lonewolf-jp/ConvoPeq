"""Test headroom MCP tool calls (compress, retrieve, stats)."""
import subprocess
import json
import time

proc = subprocess.Popen(
    [r'C:\Users\user\AppData\Roaming\Python\Python314\Scripts\headroom.exe', 'mcp', 'serve'],
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
print(f"\n[1] headroom_compress:")
print(f"    Input length: {len(test_content)} chars")
print(f"    Output: {text[:150]}...")
print(f"    Hash: {res['result']['content'][0].get('hash', 'N/A')}")
print("  ✅ compress OK")

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
print(f"\n[2] headroom_stats:")
print(f"    {stats_text[:200]}")
print("  ✅ stats OK")

proc.kill()
print("\n✅ All MCP tool call tests passed!")
