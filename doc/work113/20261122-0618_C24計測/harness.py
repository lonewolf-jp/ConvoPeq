"""C-24 再測定ハーネス。

条件 E0-E3 を 4x4 の均衡ラテン方格で 20 周し、計測専用ビルドが出す
QueryPerformanceCounter のマークから S1-S4 を求める。
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import time

EXE = r"C:\VSC_Project\m231\build-msvc\ConvoPeq_artefacts\Release\ConvoPeq.exe"
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m24run"
FIX = os.path.join(WORK, "fixtures")
OUT = os.path.join(WORK, "c24_measure.csv")
LOG = os.path.join(WORK, "cli.log")
TRACE = os.path.join(WORK, "trace.txt")

kernel32 = ctypes.windll.kernel32
qpc_freq = ctypes.c_longlong()
kernel32.QueryPerformanceFrequency(ctypes.byref(qpc_freq))


def qpc():
    c = ctypes.c_longlong()
    kernel32.QueryPerformanceCounter(ctypes.byref(c))
    return c.value


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def prepare_condition(cond):
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        p = os.path.join(APPDATA, name)
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(os.path.join(FIX, "device_settings.xml"),
                    os.path.join(APPDATA, "device_settings.xml"))
    if cond in ("E1", "E3"):
        shutil.copyfile(os.path.join(FIX, "learned_state.xml"),
                        os.path.join(APPDATA, "learned_state.xml"))
    if cond in ("E2", "E3"):
        shutil.copyfile(os.path.join(FIX, "noise_shaper_learn.xml"),
                        os.path.join(APPDATA, "noise_shaper_learn.xml"))


def run_once(cond):
    prepare_condition(cond)
    files = {}
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        p = os.path.join(APPDATA, name)
        files[name] = (os.path.getsize(p), sha(p)) if os.path.exists(p) else (0, "")

    trace_value = os.path.join(WORK, "trace_%s.txt" % cond)
    if os.path.exists(trace_value):
        os.remove(trace_value)
    env = dict(os.environ)
    env["CONVOPEQ_M24_TRACE"] = trace_value
    if os.path.exists(LOG):
        os.remove(LOG)

    t0 = qpc()
    proc = subprocess.Popen([EXE, "--cli-run", "--cli-exit-ms", "2000",
                             "--cli-log-file", LOG],
                            env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    proc.wait()
    t1 = qpc()

    marks = {}
    if os.path.exists(trace_value):
        for line in open(trace_value, encoding="utf-8", errors="replace"):
            parts = line.split()
            if len(parts) == 2:
                marks[parts[0]] = int(parts[1])
    need = ["init_start", "init_end", "s2_start", "s2_end", "s4_start", "s4_end",
            "shutdown_start", "shutdown_end"]
    if any(k not in marks for k in need):
        raise RuntimeError("missing marks for %s: %s" % (cond, sorted(marks)))
    f = float(qpc_freq.value) / 1000.0  # ticks per ms
    d = lambda a, b: (marks[b] - marks[a]) / f
    return {
        "S1": d("init_start", "init_end"),
        "S2": d("s2_start", "s2_end"),
        "S3": d("shutdown_start", "shutdown_end"),
        "S4": d("s4_start", "s4_end"),
        "res_start": (marks["init_start"] - t0) / f,
        "res_end": (t1 - marks["shutdown_end"]) / f,
        "total": (t1 - t0) / f,
        "nsl_size": files["noise_shaper_learn.xml"][0],
        "nsl_hash": files["noise_shaper_learn.xml"][1],
        "ls_size": files["learned_state.xml"][0],
        "ls_hash": files["learned_state.xml"][1],
    }


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    conds = ["E0", "E1", "E2", "E3"]
    # 行内は均衡ラテン方格。行の並びは、境界の先行関係も釣り合う順に固定した。
    # これにより、各条件が他の各条件に先行する回数は 20 行を通して 6 回で等しくなる。
    square = [[0, 1, 2, 3], [1, 3, 0, 2], [2, 0, 3, 1], [3, 2, 1, 0]]
    row_order = [0, 3, 2, 2, 1, 2, 0, 2, 1, 2, 3, 0, 1, 3, 3, 1, 1, 0, 0, 3]
    rows = [square[i] for i in row_order[:rounds]]
    assert len(rows) == rounds, "the fixed sequence has 20 rows"
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        out.write("round,pos,cond,S1_ms,S2_ms,S3_ms,S4_ms,res_start_ms,res_end_ms,total_ms,"
                  "nsl_size,nsl_hash,ls_size,ls_hash\n")
        for r, order in enumerate(rows):
            for pos, idx in enumerate(order):
                cond = conds[idx]
                m = run_once(cond)
                out.write("%d,%d,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%d,%s,%d,%s\n" % (
                    r + 1, pos + 1, cond, m["S1"], m["S2"], m["S3"], m["S4"],
                    m["res_start"], m["res_end"], m["total"],
                    m["nsl_size"], m["nsl_hash"], m["ls_size"], m["ls_hash"]))
                out.flush()
                print("round %2d %s S1=%.0f S2=%.0f S3=%.0f S4=%.0f" % (
                    r + 1, cond, m["S1"], m["S2"], m["S3"], m["S4"]), flush=True)


if __name__ == "__main__":
    main()
