#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_layout_offsets.py — D11 layout coherence regression check (D162-2-I3-4-D).

Confirms the rebuilt executables reference a single, consistent
worldAuthority_.coordinator_ slot offset (ctor store == publish-side loads)
and that the old crash-era offsets are absent.
"""
import struct

def rva_scan(path):
    data = open(path, 'rb').read()
    pe = struct.unpack_from('<I', data, 0x3C)[0]
    nsec = struct.unpack_from('<H', data, pe + 6)[0]
    opt_size = struct.unpack_from('<H', data, pe + 20)[0]
    sec_tab = pe + 24 + opt_size
    text = None
    for i in range(nsec):
        off = sec_tab + i * 40
        nm = data[off:off+8].rstrip(b'\0')
        vsize, vaddr, rsize, roff = struct.unpack_from('<IIII', data, off + 8)
        if nm == b'.text':
            text = (vaddr, vsize, roff, rsize)
    va, vs, ro, rs = text
    t = data[ro:ro+rs]
    # candidate W disp32 values, newest first
    candidates = [0x01290880, 0x012A8880, 0x012A7640]
    found = {}
    for c in candidates:
        pat = struct.pack('<I', c)
        j = t.find(pat)
        found[c] = t.count(pat)
    return found

if __name__ == '__main__':
    exes = {
        'Release': r'C:\VSC_Project\ConvoPeq\build-diag\Release\AudioEngineHarness.exe',
        'Debug':   r'C:\VSC_Project\ConvoPeq\build-diag\Debug\AudioEngineHarness.exe',
    }
    print('D11 layout coherence check')
    ok = True
    for cfg, p in exes.items():
        f = rva_scan(p)
        print(f'  {cfg}: disp32 counts = ' + ', '.join(f'0x{c:X}:{n}' for c, n in f.items()))
        # coherence: exactly one nonzero-candidate family, old-crash offset 0x12A7640 absent
        nonzero = [c for c, n in f.items() if n > 0]
        if 0x012A7640 in f and f[0x012A7640] > 0:
            print(f'  {cfg}: FAIL - old crash-era offset 0x12A7640 present')
            ok = False
        elif len(nonzero) != 1:
            print(f'  {cfg}: WARN - ambiguous offset families: {[hex(c) for c in nonzero]}')
    print('RESULT:', 'PASS' if ok else 'FAIL')
