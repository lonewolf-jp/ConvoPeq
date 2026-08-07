import json, urllib.request, urllib.error, subprocess, os

PROXY = 'http://127.0.0.1:8787'
HR = r'C:\VSC_Project\ConvoPeq\.venv\Scripts\headroom.exe'
VP = r'C:\VSC_Project\ConvoPeq\.venv\Scripts\python.exe'

def sz(b):
    try: return len(b.encode())
    except Exception: return len(b)

print("=== A) Large JSON (SmartCrusher) via /v1/compress ===")
data = {
    "files": [
        {"path": f"src/module_{i}.py", "lang": "python",
         "lines": [{"num": j, "code": f"def func_{i}_{j}(x): return x + {i*100+j}"} for j in range(60)]}
        for i in range(40)
    ]
}
big_json = json.dumps(data)
payload = {"model": "claude-sonnet-4-6",
           "messages": [{"role": "user", "content": big_json}],
           "max_tokens": 100}
req = urllib.request.Request(PROXY + '/v1/compress',
    data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
try:
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    print('  in bytes      :', sz(big_json))
    print('  tokens_before :', resp.get('tokens_before'))
    print('  tokens_after  :', resp.get('tokens_after'))
    print('  tokens_saved  :', resp.get('tokens_saved'))
    print('  ratio         :', resp.get('compression_ratio'))
    print('  transforms    :', resp.get('transforms_applied'))
except urllib.error.HTTPError as he:
    print('HTTP', he.code, he.read().decode()[:300])
except Exception as e:
    print('ERROR:', e)

print("\n=== B) headroom library (direct) on same JSON ===")
code = (
    "import headroom, json\n"
    "data = {'files':[{"
    "'path':f'src/module_{i}.py','lang':'python',"
    "'lines':[{'num':j,'code':f'def f{i}_{j}(x): return x + {i*100+j}'} for j in range(60)]}"
    " for i in range(40)]}\n"
    "msgs=[{'role':'user','content':json.dumps(data)}]\n"
    "r=headroom.compress(msgs, model='claude-sonnet-4-6')\n"
    "print('in_chars :',len(msgs[0]['content']))\n"
    "print('out_chars:',len(r.messages[0]['content']))\n"
    "print('tokens_before:',getattr(r,'tokens_before',None))\n"
    "print('tokens_after :',getattr(r,'tokens_after',None))\n"
    "print('transforms   :',getattr(r,'transforms_applied',None))\n"
)
out = subprocess.run([VP, '-c', code], capture_output=True, text=True, cwd=r'C:\VSC_Project\ConvoPeq')
print(out.stdout.strip())
if out.stderr.strip():
    print('stderr:', out.stderr.strip()[-400:])

print("\n=== C) MCP server (headroom mcp serve) -- opencode.json path ===")
print("headroom version:", subprocess.run([HR,'--version'],capture_output=True,text=True).stdout.strip())
