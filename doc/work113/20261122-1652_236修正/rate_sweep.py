"""Vary the saved ASIO rate and measure init, to separate format cost from open cost."""
import ctypes
import csv
import os
import re
import shutil
import subprocess
import sys

WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
kernel32 = ctypes.windll.kernel32
_qpc_freq = ctypes.c_longlong()
kernel32.QueryPerformanceFrequency(ctypes.byref(_qpc_freq))
F = _qpc_freq.value / 1000.0  # ticks per millisecond
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
EXE = r"C:\VSC_Project\fix236\build-msvc\ConvoPeq_artefacts\Release\ConvoPeq.exe"


def run(rate, tag, runs):
    base = open(os.path.join(FIX, "device_settings.xml"), encoding="utf-8").read()
    text = re.sub(r'audioDeviceRate="[^"]*"', 'audioDeviceRate="%s"' % rate, base)
    rows = []
    for i in range(runs):
        for name in ("noise_shaper_learn.xml", "learned_state.xml"):
            p = os.path.join(APPDATA, name)
            if os.path.exists(p):
                os.remove(p)
        open(os.path.join(APPDATA, "device_settings.xml"), "w", encoding="utf-8", newline="\n").write(text)
        trace = os.path.join(WORK, "rate_%s_%d.txt" % (tag, i))
        log = os.path.join(WORK, "rate_%s.log" % tag)
        if os.path.exists(trace):
            os.remove(trace)
        env = dict(os.environ)
        env["CONVOPEQ_M236_TRACE"] = trace
        proc = subprocess.Popen([EXE, "--cli-run", "--cli-exit-ms", "1000", "--cli-log-file", log],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()
        marks, cpus = {}, {}
        for line in open(trace, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) >= 2:
                marks[p[0]] = int(p[1])
                cpus[p[0]] = int(p[2]) if len(p) > 2 else 0
        rows.append((rate, (marks["init_end"] - marks["init_begin"]) / f,
                     (cpus["init_end"] - cpus["init_begin"]) / 1e4,
                     (marks["s2_end"] - marks["s2_begin"]) / f))
    return rows


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    all_rows = []
    for rate in ("48000.0", "96000.0", "192000.0"):
        all_rows += run(rate, rate.split(".")[0], runs)
    out = os.path.join(WORK, "rate_sweep.csv")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("rate,init_ms,init_cpu_ms,S2_ms\n")
        for r in all_rows:
            f.write("%s,%.1f,%.1f,%.1f\n" % r)
    for r in all_rows:
        print("rate %-9s init=%7.1f cpu=%7.1f S2=%7.1f" % r)
