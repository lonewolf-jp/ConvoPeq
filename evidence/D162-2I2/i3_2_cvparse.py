# -*- coding: utf-8 -*-
"""I3-2: COFF obj .debug$S (CodeView) parser — faulting instruction -> source line."""
import struct, sys

p = 'C:/VSC_Project/ConvoPeq/build-diag/CMakeFiles/AudioEngineHarness.dir/Release/src/audioengine/ISRRuntimePublicationCoordinator.cpp.obj'
data = open(p, 'rb').read()

pat = bytes.fromhex('c6426502c64264014d89')  # faulting instruction + following
pat_off = data.find(pat)
# BIGOBJ detection (Sig1=0, Sig2=0xFFFF, Version=2)
bigobj = struct.unpack_from('<H', data, 2)[0] == 0xFFFF
if bigobj:
    num_sects = struct.unpack_from('<I', data, 44)[0]
    secs_off = 56
    symtab_off = struct.unpack_from('<I', data, 48)[0]
    num_syms = struct.unpack_from('<I', data, 52)[0]
else:
    num_sects = struct.unpack_from('<H', data, 2)[0]
    secs_off = 20
    symtab_off = struct.unpack_from('<I', data, 8)[0]
    num_syms = struct.unpack_from('<I', data, 12)[0]
print('BIGOBJ:', bigobj, 'num_sections:', num_sects, 'pattern obj offset:', hex(pat_off))

secs = []
for s in range(num_sects):
    o = secs_off + s * 40
    name = data[o:o + 8].rstrip(b'\x00').decode('latin-1')
    vsize, va, rawsize, rawptr = struct.unpack_from('<IIII', data, o + 8)
    chars = struct.unpack_from('<I', data, o + 36)[0]
    if rawptr and rawsize:
        secs.append((s + 1, name, va, rawptr, rawsize, chars))

crash_sect = None
for (idx, name, va, rawptr, rawsize, chars) in secs:
    if rawptr <= pat_off < rawptr + rawsize:
        crash_sect = idx
        crash_off_in_sect = pat_off - rawptr
        print('crash pattern: section idx %d (%s) offset-in-section 0x%x chars=0x%x COMDAT=%s' %
              (idx, name, crash_off_in_sect, chars, bool(chars & 0x1050)))

# ---- parse .debug$S subsections ----
procs = []      # (section_idx, code_off, length, name)
lines = []      # (section_idx, code_off, line)
files_by_off = {}


def parse_debugS(idx, payload):
    off = 4  # skip signature
    end = len(payload)
    while off + 8 <= end:
        kind, length = struct.unpack_from('<II', payload, off)
        body = payload[off + 8: off + 8 + length]
        if kind == 0xF1:  # DEBUG_S_SYMBOLS
            o = 0
            while o + 4 <= len(body):
                reclen = struct.unpack_from('<H', body, o)[0]
                if reclen < 2:
                    break
                rectyp = struct.unpack_from('<H', body, o + 2)[0]
                if rectyp in (0x110F, 0x1110):  # S_LPROC32 / S_GPROC32
                    # pParent, pEnd, pNext, len, DbgStart, DbgEnd, typeref, offset, segment
                    plen, dbgstart, dbgend, typeref, codeoff, segment = struct.unpack_from(
                        '<IIIIII', body, o + 4 + 12)
                    namez = body.index(b'\x00', o + 4 + 12 + 24)
                    name = body[o + 4 + 12 + 24:namez].decode('latin-1')
                    procs.append((idx, codeoff, plen, name))
                o += 2 + reclen
        elif kind == 0xF2:  # DEBUG_S_LINES
            flags, = struct.unpack_from('<I', body, 0)
            has_cols = flags & 1
            sec = struct.unpack_from('<H', body, 4)[0]
            cod, blk, cnt = struct.unpack_from('<III', body, 8)
            o2 = 20
            while o2 + 12 <= len(body):
                foff, = struct.unpack_from('<I', body, o2)
                fcnt, fsize = struct.unpack_from('<II', body, o2 + 4)
                o2 += 12
                for i in range(fcnt):
                    loff, = struct.unpack_from('<I', body, o2)
                    lno, = struct.unpack_from('<I', body, o2 + 4)
                    lines.append((sec, cod + loff, lno))
                    o2 += 8 + (4 if has_cols else 0)
        off += 8 + length + ((4 - length % 4) % 4)


for (idx, name, va, rawptr, rawsize, chars) in secs:
    if name == '.debug$S' and rawptr:
        payload = data[rawptr:rawptr + rawsize]
        try:
            parse_debugS(idx, payload)
        except Exception:
            pass  # subsection tail padding — non-fatal

print('procs:', len(procs), 'line entries:', len(lines))

# ---- match crash ----
crash = None
for (secidx, codeoff, plen, name) in procs:
    if secidx == crash_sect and codeoff <= crash_off_in_sect < codeoff + plen:
        crash = (codeoff, plen, name)
        break
print()
if crash:
    print('=== enclosing function ===')
    print('proc:', crash[2])
    print('code offset 0x%x..0x%x, crash at 0x%x (+0x%x into func)' %
          (crash[0], crash[0] + crash[1], crash_off_in_sect, crash_off_in_sect - crash[0]))
best_line = None
for (secidx, codeoff, lno) in lines:
    if secidx == crash_sect and codeoff <= crash_off_in_sect:
        if best_line is None or codeoff > best_line[1]:
            best_line = (codeoff, lno)
if best_line:
    print('=== crash source line ===')
    print('line', best_line[1], '(instruction at line-block offset 0x%x)' % best_line[0])
# show nearby lines
near = sorted([(c, l) for (s, c, l) in lines if s == crash_sect])
print()
print('=== line table (this COMDAT) ===')
for (c, l) in near:
    mark = ' <== CRASH' if c <= crash_off_in_sect < c + 8 else ''
    print('  off 0x%04x  line %d%s' % (c, l, mark))
