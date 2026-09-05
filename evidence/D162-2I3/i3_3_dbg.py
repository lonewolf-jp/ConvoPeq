# -*- coding: utf-8 -*-
"""D162-2-I3-3 diagnostic mini-debugger v4 (read-only audit tooling).

Writer-attribution via hardware write watchpoints armed BEFORE construction:

  W = &worldAuthority_.coordinator_ = AudioEngine_instance + 0x12A7640
  (byte-verified: worldAuthority_ subobject offset within the AudioEngine
   instance; crash-path caller 0x1f64f35 loads rcx=[r13+0x12A7640]).

  1. One-shot int3 at the AudioEngine constructor entry (RVA 0x1F5D080,
     prologue byte-verified against the Release exe from the obj COFF symbol
     ??0AudioEngine@@QEAA@XZ).
  2. At the hit: rcx = engine instance. W = rcx + 0x12A7640.
     Restore the byte, rewind RIP, install DR0/DR1 4-byte WRITE watchpoints
     covering [W, W+8) on every existing thread and every future thread.
  3. Expected hit sequence:
       #1: the reference initialization store (new value = bridge address)
           — proves W is correct (legit ctor write);
       #2+: any later write — the zeroing writer (new value expected 0).
     Each hit logs RIP/RVA, instruction bytes, registers, stack scan, W value.
  4. Continue to the second-chance AV (expected RVA 0x1F7F89D) and capture.

Pure Win32 via ctypes. Writes one JSON evidence file per run.
"""
import ctypes, ctypes.wintypes as wt, json, os, struct, sys, time

k32 = ctypes.windll.kernel32
DEBUG_ONLY_THIS_PROCESS = 0x00000002
DBG_CONTINUE = 0x00010002
DBG_EXCEPTION_NOT_HANDLED = 0x80010001
EXCEPTION_DEBUG_EVENT = 1
CREATE_THREAD_DEBUG_EVENT = 2
CREATE_PROCESS_DEBUG_EVENT = 3
EXIT_THREAD_DEBUG_EVENT = 4
EXIT_PROCESS_DEBUG_EVENT = 5

EXCEPTION_BREAKPOINT = 0x80000003
EXCEPTION_SINGLE_STEP = 0x80000004
STATUS_ACCESS_VIOLATION = 0xC0000005

CONTEXT_FULL64 = 0x00100F0F
CONTEXT_DEBUG_REGISTERS = 0x00100010

CTOR_ENTRY_RVA = 0x1F5D080
ENGINE_W_OFF = 0x12A7640
CRASH_RVA_EXPECT = 0x1F7F89D

DR7_TWO_WRITES = (1 << 0) | (1 << 2) | (1 << 16) | (1 << 20) | (3 << 18) | (3 << 22)


class EXCEPTION_RECORD64(ctypes.Structure):
    _fields_ = [("ExceptionCode", ctypes.c_uint32), ("ExceptionFlags", ctypes.c_uint32),
                ("ExceptionRecord", ctypes.c_uint64), ("ExceptionAddress", ctypes.c_uint64),
                ("NumberParameters", ctypes.c_uint32), ("__unusedAlignment", ctypes.c_uint32),
                ("ExceptionInformation", ctypes.c_uint64 * 15)]


class EXCEPTION_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("ExceptionRecord", EXCEPTION_RECORD64), ("dwFirstChance", ctypes.c_uint32)]


class CREATE_THREAD_DEBUG_INFO(ctypes.Structure):
    _fields_ = [("hThread", ctypes.c_void_p), ("lpThreadLocalBase", ctypes.c_void_p),
                ("lpStartAddress", ctypes.c_void_p)]


class DEBUG_EVENT(ctypes.Structure):
    class U(ctypes.Union):
        _fields_ = [("Exception", EXCEPTION_DEBUG_INFO),
                    ("CreateThread", CREATE_THREAD_DEBUG_INFO),
                    ("raw", ctypes.c_byte * 160)]
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
                ("SegEs", ctypes.c_uint16), ("SegSs", ctypes.c_uint16),
                ("EFlags", ctypes.c_uint32),
                ("Dr0", ctypes.c_uint64), ("Dr1", ctypes.c_uint64), ("Dr2", ctypes.c_uint64),
                ("Dr3", ctypes.c_uint64), ("Dr6", ctypes.c_uint64), ("Dr7", ctypes.c_uint64),
                ("Rax", ctypes.c_uint64), ("Rcx", ctypes.c_uint64), ("Rdx", ctypes.c_uint64),
                ("Rbx", ctypes.c_uint64), ("Rsp", ctypes.c_uint64), ("Rbp", ctypes.c_uint64),
                ("Rsi", ctypes.c_uint64), ("Rdi", ctypes.c_uint64),
                ("R8", ctypes.c_uint64), ("R9", ctypes.c_uint64), ("R10", ctypes.c_uint64),
                ("R11", ctypes.c_uint64), ("R12", ctypes.c_uint64), ("R13", ctypes.c_uint64),
                ("R14", ctypes.c_uint64), ("R15", ctypes.c_uint64), ("Rip", ctypes.c_uint64),
                ("FltSave", ctypes.c_byte * 512), ("VectorRegister", M128A * 26),
                ("VectorControl", ctypes.c_uint64), ("DebugControl", ctypes.c_uint64),
                ("LastBranchToRip", ctypes.c_uint64), ("LastBranchFromRip", ctypes.c_uint64),
                ("LastExceptionToRip", ctypes.c_uint64), ("LastExceptionFromRip", ctypes.c_uint64)]


class STARTUPINFO(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("lpReserved", wt.LPWSTR), ("lpDesktop", wt.LPWSTR),
                ("lpTitle", wt.LPWSTR), ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32),
                ("dwXSize", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32),
                ("dwFillAttribute", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
                ("lpReserved2", ctypes.c_void_p), ("hStdInput", ctypes.c_void_p),
                ("hStdOutput", ctypes.c_void_p), ("hStdError", ctypes.c_void_p)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]


class MEMORY_BASIC_INFORMATION64(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_uint64), ("AllocationBase", ctypes.c_uint64),
                ("AllocationProtect", ctypes.c_uint32), ("__alignment1", ctypes.c_uint32),
                ("RegionSize", ctypes.c_uint64), ("State", ctypes.c_uint32),
                ("Protect", ctypes.c_uint32), ("Type", ctypes.c_uint32),
                ("__alignment2", ctypes.c_uint32)]


def set_debug_privilege():
    adv = ctypes.windll.advapi32

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

    class LUID_AND_ATTR(ctypes.Structure):
        _fields_ = [("Luid", LUID), ("Attributes", ctypes.c_uint32)]

    class TOKEN_PRIVILEGES(ctypes.Structure):
        _fields_ = [("PrivilegeCount", ctypes.c_uint32), ("Privileges", LUID_AND_ATTR * 1)]

    tok = wt.HANDLE()
    adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0028, ctypes.byref(tok))
    luid = LUID()
    adv.LookupPrivilegeValueW(None, "SeDebugPrivilege", ctypes.byref(luid))
    tp = TOKEN_PRIVILEGES()
    tp.PrivilegeCount = 1
    tp.Privileges[0].Luid = luid
    tp.Privileges[0].Attributes = 0x00000002
    adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None)
    k32.CloseHandle(tok)


def read_mem(hproc, addr, size):
    buf = ctypes.create_string_buffer(size)
    got = ctypes.c_size_t(0)
    ok = k32.ReadProcessMemory(hproc, ctypes.c_void_p(addr), buf, size, ctypes.byref(got))
    if ok or got.value > 0:
        return buf.raw[:got.value]
    return b""


def write_mem(hproc, addr, data):
    buf = ctypes.create_string_buffer(data, len(data))
    written = ctypes.c_size_t(0)
    old_prot = ctypes.c_uint32(0)
    k32.VirtualProtectEx(hproc, ctypes.c_void_p(addr), len(data), 0x40, ctypes.byref(old_prot))
    ok = k32.WriteProcessMemory(hproc, ctypes.c_void_p(addr), buf, len(data), ctypes.byref(written))
    k32.VirtualProtectEx(hproc, ctypes.c_void_p(addr), len(data), old_prot.value, ctypes.byref(old_prot))
    return ok and written.value == len(data)


def mem_region_info(hproc, addr):
    mbi = MEMORY_BASIC_INFORMATION64()
    ret = k32.VirtualQueryEx(hproc, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
    if not ret:
        return {"base": None}
    return {"base": hex(mbi.BaseAddress), "allocBase": hex(mbi.AllocationBase),
            "size": hex(mbi.RegionSize), "state": hex(mbi.State),
            "protect": hex(mbi.Protect), "type": hex(mbi.Type)}


def get_ctx(hthread):
    ctx = CONTEXT()
    ctx.ContextFlags = CONTEXT_FULL64
    if k32.GetThreadContext(hthread, ctypes.byref(ctx)):
        return ctx
    return None


def set_ctx(hthread, ctx):
    return k32.SetThreadContext(hthread, ctypes.byref(ctx))


def apply_watchpoints(hthread, dr_addrs):
    ctx = CONTEXT()
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS
    ctx.Dr0 = dr_addrs[0]
    ctx.Dr1 = dr_addrs[1] if len(dr_addrs) > 1 else 0
    ctx.Dr7 = DR7_TWO_WRITES
    return set_ctx(hthread, ctx)


def scan_stack_for_module_rets(hproc, rsp, base, size=0x6000):
    stack = read_mem(hproc, rsp, size)
    rets = []
    if stack and base:
        lo, hi = base, base + 0x4000000
        for off in range(0, len(stack) - 8, 8):
            v = struct.unpack_from("<Q", stack, off)[0]
            if lo <= v < hi:
                rets.append({"rsp_off": off, "rva": hex(v - base)})
    return rets, stack


def regs_dict(ctx, base):
    return {"rip": hex(ctx.Rip), "rva": hex(ctx.Rip - base),
            "rax": hex(ctx.Rax), "rcx": hex(ctx.Rcx), "rdx": hex(ctx.Rdx),
            "rbx": hex(ctx.Rbx), "rsp": hex(ctx.Rsp), "rbp": hex(ctx.Rbp),
            "rsi": hex(ctx.Rsi), "rdi": hex(ctx.Rdi),
            "r8": hex(ctx.R8), "r9": hex(ctx.R9), "r10": hex(ctx.R10), "r11": hex(ctx.R11),
            "r12": hex(ctx.R12), "r13": hex(ctx.R13), "r14": hex(ctx.R14), "r15": hex(ctx.R15)}


def main():
    if len(sys.argv) < 3:
        print("usage: i3_3_dbg.py <exe> <outfile.json> [label]")
        sys.exit(1)
    exe_arg = os.path.abspath(sys.argv[1])
    out_arg = os.path.abspath(sys.argv[2])
    evidence_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.dirname(out_arg) != evidence_dir:
        print("outfile must be in evidence dir")
        sys.exit(1)
    run_label = sys.argv[3] if len(sys.argv) > 3 else "run"

    set_debug_privilege()
    si = STARTUPINFO(); si.cb = ctypes.sizeof(si)
    pi = PROCESS_INFORMATION()
    ok = k32.CreateProcessW(exe_arg, exe_arg, None, None, False,
                            DEBUG_ONLY_THIS_PROCESS, None, None, ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        print("CreateProcess failed", ctypes.GetLastError()); sys.exit(1)
    hproc = pi.hProcess
    print("[%s] debuggee pid=%d" % (run_label, pi.dwProcessId))

    ev = DEBUG_EVENT()
    base = None
    log = {"run": run_label, "exe": exe_arg, "events": [], "captures": [],
           "W_layout": {"engine_offset": hex(ENGINE_W_OFF), "ctor_entry_rva": hex(CTOR_ENTRY_RVA),
                        "expected_crash_rva": hex(CRASH_RVA_EXPECT)}}
    t0 = time.time()
    st = {"phase": "startup", "W": None, "engine": None, "watch": False,
          "dr_addrs": [], "hits": 0, "max_hits": 16, "prev_w": None}
    threads = {pi.dwThreadId: pi.hThread}

    def finish(status=0):
        log["summary"] = {"phase": st["phase"], "engine": hex(st["engine"]) if st["engine"] else None,
                          "W": hex(st["W"]) if st["W"] else None,
                          "dr_hits": st["hits"], "prev_w": st["prev_w"]}
        with open(out_arg, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=1)
        print("[%s] done status=%d summary=%s" % (run_label, status, log["summary"]))

    while True:
        if not k32.WaitForDebugEvent(ctypes.byref(ev), 1000):
            if ctypes.GetLastError() != 121:
                print("WaitForDebugEvent err", ctypes.GetLastError())
            if time.time() - t0 > 240:
                log["events"].append({"ev": "timeout"})
                break
            continue
        code, tid = ev.dwDebugEventCode, ev.dwThreadId
        status = DBG_CONTINUE

        if code == CREATE_PROCESS_DEBUG_EVENT:
            raw = bytes(ev.u.raw)
            base = struct.unpack_from("<Q", raw, 24)[0]
            log["image_base"] = hex(base)
            log["events"].append({"ev": "create_process", "base": hex(base)})
            original = read_mem(hproc, base + CTOR_ENTRY_RVA, 1)
            log["ctor_orig_byte"] = original.hex()
            write_mem(hproc, base + CTOR_ENTRY_RVA, b"\xCC")
            st["phase"] = "wait_ctor"
            h_file, h_p2, h_t2 = struct.unpack_from("<QQQ", raw, 8)
            for h in (h_file, h_t2):
                if h:
                    k32.CloseHandle(ctypes.c_void_p(h))

        elif code == CREATE_THREAD_DEBUG_EVENT:
            th = ev.u.CreateThread.hThread
            threads[tid] = th
            log["events"].append({"ev": "create_thread", "tid": tid,
                                  "start": hex(ev.u.CreateThread.lpStartAddress)})
            if st["watch"]:
                apply_watchpoints(th, st["dr_addrs"])

        elif code == EXIT_THREAD_DEBUG_EVENT:
            log["events"].append({"ev": "exit_thread", "tid": tid})

        elif code == EXCEPTION_DEBUG_EVENT:
            er = ev.u.Exception.ExceptionRecord
            exc_code = er.ExceptionCode & 0xFFFFFFFF
            first = ev.u.Exception.dwFirstChance
            addr = er.ExceptionAddress
            log["events"].append({"ev": "exception", "tid": tid, "code": "0x%08X" % exc_code,
                                  "first_chance": first, "address": hex(addr)})

            if (exc_code == EXCEPTION_BREAKPOINT and first and st["phase"] == "wait_ctor"
                    and addr == base + CTOR_ENTRY_RVA):
                hthread = threads.get(tid)
                ctx = get_ctx(hthread)
                engine = ctx.Rcx
                st["engine"] = engine
                st["W"] = engine + ENGINE_W_OFF
                write_mem(hproc, base + CTOR_ENTRY_RVA, bytes([int(log["ctor_orig_byte"], 16)]))
                ctx.Rip = base + CTOR_ENTRY_RVA
                set_ctx(hthread, ctx)
                pre = read_mem(hproc, st["W"], 8)
                pre_val = struct.unpack("<Q", pre)[0] if len(pre) == 8 else None
                log["phase_a"] = {
                    "ctor_thread": tid, "engine_addr": hex(engine), "W_addr": hex(st["W"]),
                    "W_preinit_value": hex(pre_val) if pre_val is not None else None,
                    "note": "anchor at ctor entry; DR write watchpoints armed BEFORE init",
                }
                print("[%s] Phase A: engine=%#x W=%#x preinit=%s" % (
                    run_label, engine, st["W"], hex(pre_val) if pre_val is not None else "?"))
                dr_addrs = [st["W"] & ~0x3, (st["W"] + 4) & ~0x3]
                st["dr_addrs"] = dr_addrs
                armed = 0
                for t_id, h_t in threads.items():
                    if apply_watchpoints(h_t, dr_addrs):
                        armed += 1
                log["events"].append({"ev": "watch_armed", "threads": armed,
                                      "dr0": hex(dr_addrs[0]), "dr1": hex(dr_addrs[1])})
                st["watch"] = True
                st["phase"] = "watching"
                print("[%s] Phase B: DR0/DR1 write watchpoints on %d threads" % (run_label, armed))
                k32.ContinueDebugEvent(ev.dwProcessId, tid, DBG_CONTINUE)
                continue

            elif exc_code == EXCEPTION_SINGLE_STEP and st["phase"] == "watching" and st["W"]:
                hthread = threads.get(tid)
                ctx = get_ctx(hthread)
                if ctx and (ctx.Dr6 & 0x3):
                    st["hits"] += 1
                    rip = ctx.Rip
                    code_bytes = read_mem(hproc, rip, 16)
                    raw8 = read_mem(hproc, st["W"], 8)
                    w_val = struct.unpack("<Q", raw8)[0] if len(raw8) == 8 else None
                    rets, stack = scan_stack_for_module_rets(hproc, ctx.Rsp, base)
                    cap = {"event": "WRITE_HIT", "n": st["hits"], "tid": tid,
                           "dr6": hex(ctx.Dr6), "regs": regs_dict(ctx, base),
                           "instruction_bytes": code_bytes.hex(),
                           "W_value_after": hex(w_val) if w_val is not None else None,
                           "prev_W_value": st["prev_w"],
                           "stack_return_candidates": rets[:80],
                           "W_region": mem_region_info(hproc, st["W"])}
                    st["prev_w"] = hex(w_val) if w_val is not None else None
                    log["captures"].append(cap)
                    print("[%s] WRITE_HIT #%d tid=%d rip=%#x rva=%s W=%s" % (
                        run_label, st["hits"], tid, rip, hex(rip - base),
                        cap["W_value_after"]))
                    ctx.Dr6 = 0
                    set_ctx(hthread, ctx)
                    apply_watchpoints(hthread, st["dr_addrs"])
                    if st["hits"] >= st["max_hits"]:
                        print("[%s] max hits reached; terminating" % run_label)
                        k32.TerminateProcess(hproc, 2)
                        k32.DebugActiveProcessStop(pi.dwProcessId)
                        finish(2)
                        return
                k32.ContinueDebugEvent(ev.dwProcessId, tid, DBG_CONTINUE)
                continue

            elif exc_code == STATUS_ACCESS_VIOLATION and not first:
                hthread = threads.get(tid)
                ctx = get_ctx(hthread)
                code_bytes = read_mem(hproc, ctx.Rip, 16)
                rets, stack = scan_stack_for_module_rets(hproc, ctx.Rsp, base)
                raw8 = read_mem(hproc, st["W"], 8) if st["W"] else b""
                cap = {"event": "UNHANDLED_AV", "tid": tid,
                       "exception_address": hex(addr),
                       "exception_information": [er.ExceptionInformation[0],
                                                 er.ExceptionInformation[1]],
                       "instruction_bytes": code_bytes.hex(),
                       "regs": regs_dict(ctx, base),
                       "stack_return_candidates": rets[:80],
                       "W_at_crash": hex(struct.unpack("<Q", raw8)[0]) if len(raw8) == 8 else None,
                       "stack_dump_hex": stack[:0x800].hex()}
                log["captures"].append(cap)
                print("[%s] UNHANDLED_AV rip=%#x rva=%s W=%s" % (
                    run_label, ctx.Rip, hex(ctx.Rip - base), cap["W_at_crash"]))
                k32.TerminateProcess(hproc, 1)
                k32.DebugActiveProcessStop(pi.dwProcessId)
                finish(1)
                return
            elif exc_code == EXCEPTION_BREAKPOINT and first:
                status = DBG_CONTINUE
            else:
                status = DBG_EXCEPTION_NOT_HANDLED

        elif code == EXIT_PROCESS_DEBUG_EVENT:
            log["events"].append({"ev": "exit_process"})
            finish(0)
            return

        k32.ContinueDebugEvent(ev.dwProcessId, tid, status)

    k32.TerminateProcess(hproc, 3)
    k32.DebugActiveProcessStop(pi.dwProcessId)
    finish(3)


if __name__ == "__main__":
    main()
