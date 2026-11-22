"""Interleaved A/B: same build except the #236 rebuild-collapse fix. E2 only, 20 runs each."""
import ctypes
import io
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_guard

WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
PRE = os.path.join(WORK, "exe_pre.exe")
POST = os.path.join(WORK, "exe_post.exe")
OUT = os.path.join(WORK, "ab236.csv")

kernel32 = ctypes.windll.kernel32
freq = ctypes.c_longlong()
kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
F = freq.value / 1000.0


def prepare():
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        p = os.path.join(APPDATA, name)
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(os.path.join(FIX, "device_settings.xml"), os.path.join(APPDATA, "device_settings.xml"))
    shutil.copyfile(os.path.join(FIX, "noise_shaper_learn.xml"), os.path.join(APPDATA, "noise_shaper_learn.xml"))


def run(exe, tag, i):
    prepare()
    trace = os.path.join(WORK, "ab_%s_%02d.txt" % (tag, i))
    if os.path.exists(trace):
        os.remove(trace)
    env = dict(os.environ)
    env["CONVOPEQ_M236_TRACE"] = trace
    proc = subprocess.Popen([exe, "--cli-run", "--cli-exit-ms", "1000",
                             "--cli-log-file", os.path.join(WORK, "ab_cli.log")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    m = {}
    for line in io.open(trace, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 2:
            m[p[0]] = int(p[1])
    d = lambda a, b: (m[b] - m[a]) / F
    return {"tag": tag, "run": i, "S2": d("s2_begin", "s2_end"),
            "init": d("init_begin", "init_end"), "restore": d("restore_begin", "restore_end"),
            "final": d("restore_end", "s2_end"), "S1": d("app_init_start", "app_init_end")}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    guard = profile_guard.Guard()
    guard.save()
    try:
        _run_all(n)
    finally:
        guard.restore()


def _run_all(n):
    rows = []
    for i in range(n):
        order = [("pre", PRE), ("post", POST)] if i % 2 == 0 else [("post", POST), ("pre", PRE)]
        for tag, exe in order:
            rows.append(run(exe, tag, i))
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("run,exe,S2_ms,init_ms,restore_ms,final_ms,S1_ms\n")
        for r in rows:
            f.write("%d,%s,%.1f,%.1f,%.1f,%.1f,%.1f\n" % (
                r["run"], r["tag"], r["S2"], r["init"], r["restore"], r["final"], r["S1"]))
    for tag in ("pre", "post"):
        v = sorted(r["S2"] for r in rows if r["tag"] == tag)
        n2 = len(v)
        med = (v[n2 // 2 - 1] + v[n2 // 2]) / 2
        print("%-4s S2 median=%.1f min=%.1f max=%.1f" % (tag, med, v[0], v[-1]))


if __name__ == "__main__":
    main()
