#!/usr/bin/env python3
"""Crash dump analyzer - reads MINIDUMP directly."""
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"C:\VSC_Project\ConvoPeq\build\crash.dmp"

with open(path, 'rb') as f:
    sig = f.read(4)
    ver = struct.unpack('I', f.read(4))[0]
    nstreams = struct.unpack('I', f.read(4))[0]
    dir_rva = struct.unpack('I', f.read(4))[0]
    f.read(4)  # check_sum
    ts = struct.unpack('I', f.read(4))[0]
    f.read(16)  # flags

    print(f"Signature: {sig}")
    print(f"Version: {ver}")
    print(f"Num streams: {nstreams}")
    print()

    f.seek(dir_rva)
    streams = {}
    for i in range(nstreams):
        stype = struct.unpack('I', f.read(4))[0]
        datasize = struct.unpack('I', f.read(4))[0]
        rva = struct.unpack('I', f.read(4))[0]
        streams[stype] = (rva, datasize)

    for st, (rva, sz) in sorted(streams.items()):
        names = {0:"Unused",3:"ThreadList",4:"ModuleList",5:"MemoryList",6:"MemoryInfoList",
                 7:"Exception",8:"SystemInfo",9:"ThreadExList",10:"Memory64List",
                 11:"CommentA",12:"CommentW",13:"HandleData",14:"FunctionTable",
                 15:"UnloadedModuleList",16:"MiscInfo",17:"MemoryInfoList64",
                 18:"Segment",19:"ProcessVmCounters",20:"VariableInfo",21:"LastReserved"}
        sname = names.get(st, f"Stream{st}")
        print(f"  {sname}({st}): RVA=0x{rva:x}, size={sz}")

    print()

    EXCEPTION_STREAM = 7
    if EXCEPTION_STREAM in streams:
        rva, sz = streams[EXCEPTION_STREAM]
        f.seek(rva)
        tid = struct.unpack('I', f.read(4))[0]
        f.read(4)  # alignment
        exc_code = struct.unpack('I', f.read(4))[0]
        exc_flags = struct.unpack('I', f.read(4))[0]
        exc_record = struct.unpack('Q', f.read(8))[0]
        exc_addr = struct.unpack('Q', f.read(8))[0]
        nparams = struct.unpack('I', f.read(4))[0]
        f.read(4)  # padding
        params = [struct.unpack('Q', f.read(8))[0] for _ in range(min(nparams, 15))]

        print(f"=== EXCEPTION (thread {tid}) ===")
        print(f"  Code:    0x{exc_code:08X}")
        if exc_code == 0xC0000005:
            print(f"  -> ACCESS_VIOLATION")
            if nparams >= 2:
                print(f"  Access:  {'WRITE' if params[0] == 1 else 'READ'} at 0x{params[1]:016X}")
        elif exc_code == 0x80000003:
            print(f"  -> BREAKPOINT (jassert/assert)")
        elif exc_code == 0x406D1388:
            print(f"  -> JUCE debug message")
        elif exc_code == 0xE06D7363:
            print(f"  -> C++ EXCEPTION")
        else:
            print(f"  -> Unknown exception")
        print(f"  Address: 0x{exc_addr:016X}")
        print(f"  Flags:   0x{exc_flags:08X}")
        print(f"  Params:  {nparams}")
        for i, p in enumerate(params):
            print(f"    [{i}] 0x{p:016X}")

        if exc_code == 0xC0000005 and nparams >= 2:
            pass  # already printed

    THREAD_LIST = 3
    if THREAD_LIST in streams:
        rva, sz = streams[THREAD_LIST]
        f.seek(rva)
        nthr = struct.unpack('I', f.read(4))[0]
        print(f"\n=== THREADS: {nthr} ===")
        for t in range(nthr):
            tid = struct.unpack('I', f.read(4))[0]
            suspend = struct.unpack('I', f.read(4))[0]
            prio = struct.unpack('I', f.read(4))[0]
            ppc = struct.unpack('Q', f.read(8))[0]
            pc = struct.unpack('Q', f.read(8))[0]
            marker = " ACTIVE" if tid == tid else ""
            print(f"  Thread {t}: tid={tid} pc=0x{pc:016X}")

    # Module list (stream 4) to resolve addresses
    MODULE_LIST = 4
    if MODULE_LIST in streams:
        rva, sz = streams[MODULE_LIST]
        f.seek(rva)
        nmod = struct.unpack('I', f.read(4))[0]
        print(f"\n=== MODULES: {nmod} ===")
        for m in range(nmod):
            # MINIDUMP_MODULE_LIST entry = MINIDUMP_MODULE
            # ULONG32 ModuleNameRva (relative to the module's name)
            # ULONG32 Reserved
            # ULONG64 BaseOfImage
            # ULONG32 SizeOfImage
            # ULONG32 CheckSum
            # ULONG32 TimeDateStamp
            # ULONG32 NameRva (RVA to MINIDUMP_STRING for the module name)
            # then VersionInfo (VS_FIXEDFILEINFO) - 56 bytes
            # then CVRecord (location descriptor)
            # then MiscRecord (location descriptor)
            # then Reserved (8 bytes)
            # then Reserved2 (8 bytes)

            # Actually simpler: MINIDUMP_MODULE is 108 bytes on 64-bit
            # Let's read it
            namerva_raw = struct.unpack('I', f.read(4))[0]
            f.read(4)  # Reserved
            base = struct.unpack('Q', f.read(8))[0]
            size = struct.unpack('I', f.read(4))[0]
            f.read(4)  # CheckSum
            ts = struct.unpack('I', f.read(4))[0]
            namerva = struct.unpack('I', f.read(4))[0]
            # Skip VS_FIXEDFILEINFO (56 bytes) + CVRecord (8) + MiscRecord (8) + Reserved(16)
            f.read(88)

            # Read name
            save = f.tell()
            f.seek(namerva)
            nchars = struct.unpack('I', f.read(4))[0]
            name_bytes = f.read(nchars * 2)
            try:
                n = name_bytes.decode('utf-16-le', errors='replace').rstrip('\x00')
            except:
                n = f"<binary:{len(name_bytes)}>"
            # Trim to last path component
            short_name = n.split('\\')[-1] if '\\' in n else n
            print(f"  0x{base:016X} {short_name}")
            f.seek(save)

