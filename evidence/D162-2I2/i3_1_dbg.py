# -*- coding: utf-8 -*-
"""D162-2-I3-1 diagnostic mini-debugger (read-only audit tooling).

Launches a target exe as a debuggee, logs debug events, and on an
unhandled (second-chance) exception captures the full thread context +
stack memory so the faulting instruction / call chain can be resolved.

Pure Win32 via ctypes. Writes a JSON evidence file per run.
"""
import ctypes, ctypes.wintypes as wt, json, os, struct, sys, time

k32 = ctypes.windll.kernel32
DEBUG_PROCESS = 0x00000001
DEBUG_ONLY_THIS_PROCESS = 0x00000002
INFINITE = 0xFFFFFFFF
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_DEBUG_EVENT = 1
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_PROCESS_DEBUG_EVENT = 5
LOAD_DLL_DEBUG_EVENT = 6

CONTEXT_FULL64 = 0x00100F0F

class EXCEPTION_RECORD64(ctypes.Structure):
    _fields_ = [("ExceptionCode", ctypes.c_uint32),
                ("ExceptionFlags", ctypes.c_uint32),
                ("ExceptionRecord", ctypes.c_uint64),
                ("ExceptionAddress", ctypes.c_uint64),
                ("NumberParameters", ctypes.c_uint32),
                ("__unusedAlignment", ctypes.c_uint32),
                ("ExceptionInformation", ctypes.c_uint64 * 15)]

class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("ExceptionRecord", EXCEPTION_RECORD64), ("dwFirstChance", ctypes.c_uint32)]

class DEBUG_EVENT(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [("Exception", EXCEPTION_DEBUG_INFO), ("raw", ctypes.c_byte * 160)]
    _fields_ = [("dwDebugEventCode", ctypes.c_uint32), ("dwProcessId", ctypes.c_uint32),
                ("dwThreadId", ctypes.c_uint32), ("u", U)]

class M128A(ctypes.Structure):
    _fields_ = [("Low", ctypes.c_uint64), ("High", ctypes.c_int64)]

class CONTEXT(ctypes.Structure):
    _fields_ = [("P1Home", ctypes.c_uint64), ("P2Home", ctypes.c_uint64),
                ("P3Home", ctypes.c_uint64), ("P4Home", ctypes.c_uint64),
                ("P5Home", ctypes.c_uint64), ("P6Home", ctypes.c_uint64),
                ("ContextFlags", ctypes.c_uint32), ("MxCsr", ctypes.c_uint32),
                ("SegCs", ctypes.c_uint16), ("SegDs", ctypes.c_uint16),
                ("SegEs", ctypes.c_uint16), ("SegFs", ctypes.c_uint16),
                ("SegGs", ctypes.c_uint16), ("SegSs", ctypes.c_uint16),
                ("EFlags", ctypes.c_uint32),
                ("Dr0", ctypes.c_uint64), ("Dr1", ctypes.c_uint64), ("Dr2", ctypes.c_uint64),
                ("Dr3", ctypes.c_uint64), ("Dr6", ctypes.c_uint64), ("Dr7", ctypes.c_uint64),
                ("Rax", ctypes.c_uint64), ("Rcx", ctypes.c_uint64), ("Rdx", ctypes.c_uint64),
                ("Rbx", ctypes.c_uint64), ("Rsp", ctypes.c_uint64), ("Rbp", ctypes.c_uint64),
                ("Rsi", ctypes.c_uint64), ("Rdi", ctypes.c_uint64),
                ("R8", ctypes.c_uint64), ("R9", ctypes.c_uint64), ("R10", ctypes.c_uint64),
                ("R11", ctypes.c_uint64), ("R12", ctypes.c_uint64), ("R13", ctypes.c_uint64),
                ("R14", ctypes.c_uint64), ("R15", ctypes.c_uint64), ("Rip", ctypes.c_uint64),
                ("FltSave", ctypes.c_byte * 512),
                ("VectorRegister", M128A * 26),
                ("VectorControl", ctypes.c_uint64), ("DebugControl", ctypes.c_uint64),
                ("LastBranchToRip", ctypes.c_uint64), ("LastBranchFromRip", ctypes.c_uint64),
                ("LastExceptionToRip", ctypes.c_uint64), ("LastExceptionFromRip", ctypes.c_uint64)]

class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR),
                ("lpTitle", wt.LPWSTR), ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32),
                ("dwXSize", ctypes.c_uint32), ("dwYSize", ctypes.c_uint32),
                ("dwXCountChars", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32),
                ("dwFillAttribute", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", ctypes.c_void_p),
                ("hStdOutput", ctypes.c_void_p), ("hStdError", ctypes.c_void_p)]

class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]


def read_mem(hproc, addr, size):
    """部分読み成功を許容（ERROR_PARTIAL_COPY=299 でも読めた分を返す）。ページ単位で縮小再試行。"""
    buf = ctypes.create_string_buffer(size)
    got = ctypes.c_size_t(0)
    ok = k32.ReadProcessMemory(hproc, ctypes.c_void_p(addr), buf, size, ctypes.byref(got))
    if ok or got.value > 0:
        return buf.raw[:got.value]
    # 512 バイト単位に縮めて再試行
    out = b""
    a = addr
    remain = size
    while remain > 0:
        chunk = ctypes.create_string_buffer(512)
        got2 = ctypes.c_size_t(0)
        if k32.ReadProcessMemory(hproc, ctypes.c_void_p(a), chunk, 512, ctypes.byref(got2)) and got2.value:
            out += chunk.raw[:got2.value]
            a += got2.value
            remain -= got2.value
        else:
            a += 512
            remain -= 512
    return out


psapi = ctypes.windll.psapi


def get_main_module_base(hproc):
    """EnumProcessModuleHandles の先頭 = 実行イメージ。psapi 経由で確実に取得する。"""
    arr = (ctypes.c_void_p * 1024)()
    needed = ctypes.c_uint32()
    if psapi.EnumProcessModuleHandles(hproc, ctypes.byref(arr), ctypes.sizeof(arr),
                                      ctypes.byref(needed)):
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        if count > 0:
            return arr[0]
    return None


def main():
    exe = sys.argv[1]
    outfile = sys.argv[2]
    exe_args = sys.argv[3] if len(sys.argv) > 3 else None
    si = STARTUPINFO(); si.cb = ctypes.sizeof(si)
    pi = PROCESS_INFORMATION()
    cmdline = exe + ((" " + exe_args) if exe_args else "")
    ok = k32.CreateProcessW(exe, cmdline, None, None, False,
                            DEBUG_ONLY_THIS_PROCESS, None, None,
                            ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        print("CreateProcess failed", ctypes.GetLastError()); sys.exit(1)
    hproc, hthread = pi.hProcess, pi.hThread
    print("debuggee pid=%d" % pi.dwProcessId)

    ev = DEBUG_EVENT()
    main_base = None
    log = []
    captured = False
    t0 = time.time()
    image_bases = {}

    while True:
        if not k32.WaitForDebugEvent(ctypes.byref(ev), 1000):
            if ctypes.GetLastError() != 121:  # not a timeout
                print("WaitForDebugEvent err", ctypes.GetLastError())
            if time.time() - t0 > 600:
                print("timeout"); break
            continue
        code, tid = ev.dwDebugEventCode, ev.dwThreadId
        status = DBG_CONTINUE

        if code == CREATE_PROCESS_DEBUG_EVENT:
            raw = bytes(ev.u.raw)
            log.append({"ev": "create_process_raw", "head": raw[:32].hex()})
            print("create_process raw head:", raw[:32].hex())
            # 実測: lpBaseOfImage は raw[24:32]（DEBUG_EVENT union は event+24）
            base = struct.unpack_from("<Q", raw, 24)[0]
            main_base = base
            image_bases["main"] = base
            log.append({"ev": "create_process", "pid": ev.dwProcessId, "image_base": hex(base)})
            print("create_process base=%#x" % base)
        elif code == LOAD_DLL_DEBUG_EVENT:
            base = struct.unpack_from("<Q", bytes(ev.u.raw), 8)[0]
            log.append({"ev": "load_dll", "base": hex(base)})
        elif code == EXCEPTION_DEBUG_EVENT:
            er = ev.u.Exception.ExceptionRecord
            exc_code = er.ExceptionCode & 0xFFFFFFFF
            first = ev.u.Exception.dwFirstChance
            addr = er.ExceptionAddress
            entry = {"ev": "exception", "tid": tid,
                     "code": "0x%08X" % exc_code,
                     "first_chance": first, "address": hex(addr)}
            log.append(entry)
            print("exception code=%s first=%d addr=%#x" %
                  (entry["code"], first, addr))
            # capture on second-chance AV/heap-corruption, or on c0000374 fast-fail
            # (fast-fail は second chance を持たないため first chance で捕獲)
            want = (not first and exc_code in (0xC0000005, 0xC0000374, 0xC0000409)) \
                or (exc_code == 0xC0000374)
            if want and not captured:
                captured = True
                hthread = k32.OpenThread(0x001FFFFF, False, tid)
                ctx = CONTEXT(); ctx.ContextFlags = CONTEXT_FULL64
                if k32.GetThreadContext(hthread, ctypes.byref(ctx)):
                    rsp = ctx.Rsp
                    stack = read_mem(hproc, rsp, 0x4000)
                    rets = []
                    if main_base and stack:
                        lo, hi = main_base, main_base + 0x4000000
                        for off in range(0, len(stack) - 8, 8):
                            v = struct.unpack_from("<Q", stack, off)[0]
                            if lo <= v < hi:
                                rets.append({"rsp_off": off, "addr": hex(v)})
                    code_bytes = read_mem(hproc, ctx.Rip, 32)
                    log.append({
                        "event": "CAPTURED_UNHANDLED",
                        "tid": tid,
                        "exception_code": entry["code"],
                        "exception_address": hex(addr),
                        "faulting_instruction_bytes": code_bytes.hex(),
                        "rip": hex(ctx.Rip), "rsp": hex(ctx.Rsp), "rbp": hex(ctx.Rbp),
                        "rax": hex(ctx.Rax), "rcx": hex(ctx.Rcx), "rdx": hex(ctx.Rdx),
                        "rbx": hex(ctx.Rbx), "rsi": hex(ctx.Rsi), "rdi": hex(ctx.Rdi),
                        "r8": hex(ctx.R8), "r9": hex(ctx.R9), "r10": hex(ctx.R10),
                        "r11": hex(ctx.R11), "r12": hex(ctx.R12), "r13": hex(ctx.R13),
                        "r14": hex(ctx.R14), "r15": hex(ctx.R15),
                        "eflags": hex(ctx.EFlags),
                        "stack_return_candidates": rets[:80],
                        "stack_dump_hex": stack[:0x800].hex(),
                    })
                    print("CAPTURED rip=%s code=%s" % (hex(ctx.Rip), entry["code"]))
                k32.CloseHandle(hthread)
                k32.TerminateProcess(hproc, 1)
                k32.DebugActiveProcessStop(pi.dwProcessId)
                break
            # ★ 修正: first chance はアプリの SEH に渡す（DBG_CONTINUE で吞むと
            #   同一アドレスの AV ループになる）。初期ブレークポイント(0x80000003 first)
            #   のみ継続、それ以外は NOT_HANDLED。
            if exc_code == 0x80000003 and first:
                status = DBG_CONTINUE
            else:
                status = DBG_EXCEPTION_NOT_HANDLED
        elif code == EXIT_PROCESS_DEBUG_EVENT:
            print("exit_process")
            log.append({"ev": "exit_process"})
            break

        k32.ContinueDebugEvent(ev.dwProcessId, ev.dwThreadId, status)

    with open(outfile, "w", encoding="utf-8") as f:
        json.dump({"image_base": hex(main_base) if main_base else None,
                   "events": log[-200:]}, f, indent=1)
    print("done, log:", outfile)


if __name__ == "__main__":
    main()
