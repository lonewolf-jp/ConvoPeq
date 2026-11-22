"""S2 breakdown harness for issue #236.

Runs the measurement build with QPC marks, alternating E0 (no learning XML) and
E2 (real noise_shaper_learn.xml, no learned_state.xml) so drift cancels within
each pair. Writes one CSV row per run with every mark interval.

Usage: python harness236.py <exe> <runs-per-condition> <out.csv>
"""
import ctypes
import hashlib
import os
import shutil
import subprocess
import sys

APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"

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


def prepare(cond):
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        p = os.path.join(APPDATA, name)
        if os.path.exists(p):
            os.remove(p)
    shutil.copyfile(os.path.join(FIX, "device_settings.xml"),
                    os.path.join(APPDATA, "device_settings.xml"))
    if cond == "E2":
        shutil.copyfile(os.path.join(FIX, "noise_shaper_learn.xml"),
                        os.path.join(APPDATA, "noise_shaper_learn.xml"))


SEGMENTS = [
    ("noise", "noise_begin", "noise_end"),
    ("bulk", "bulk_begin", "close_begin"),
    ("close", "close_begin", "close_end"),
    ("file", "file_begin", "file_end"),
    ("parse", "parse_begin", "parse_end"),
    ("init", "init_begin", "init_end"),
    ("relaxed", "relaxed_begin", "relaxed_end"),
    ("fallback", "fallback_begin", "fallback_end"),
    ("default_init", "default_init_begin", "default_init_end"),
    ("channel", "channel_begin", "channel_end"),
    ("restore", "restore_begin", "restore_end"),
    ("adaptive", "adaptive_begin", "adaptive_end"),
    ("convolver", "convolver_begin", "convolver_end"),
]


def run_once(exe, cond, idx, log):
    prepare(cond)
    sizes = {}
    for name in ("noise_shaper_learn.xml", "learned_state.xml", "device_settings.xml"):
        p = os.path.join(APPDATA, name)
        sizes[name] = (os.path.getsize(p), sha(p)) if os.path.exists(p) else (0, "")

    trace = os.path.join(WORK, "trace_%s_%02d.txt" % (cond, idx))
    if os.path.exists(trace):
        os.remove(trace)
    env = dict(os.environ)
    env["CONVOPEQ_M236_TRACE"] = trace

    t0 = qpc()
    proc = subprocess.Popen([exe, "--cli-run", "--cli-exit-ms", "2000",
                             "--cli-log-file", log],
                            env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    proc.wait()
    t1 = qpc()

    marks = {}
    cpus = {}
    order = []
    for line in open(trace, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 2:
            marks[p[0]] = int(p[1])
            cpus[p[0]] = int(p[2]) if len(p) > 2 else 0
            order.append(p[0])
    f = float(qpc_freq.value) / 1000.0
    row = {"cond": cond, "run": idx, "total": (t1 - t0) / f,
           "marks": len(marks), "order": "|".join(order)}
    row["S2"] = (marks["s2_end"] - marks["s2_begin"]) / f
    row["S2_cpu"] = (cpus["s2_end"] - cpus["s2_begin"]) / 10000.0
    row["prep"] = (marks["prep_end"] - marks["prep_begin"]) / f if "prep_end" in marks else -1.0
    row["prep_cpu"] = (cpus["prep_end"] - cpus["prep_begin"]) / 10000.0 if "prep_end" in cpus else -1.0
    row["aeprep"] = (marks["aeprep_end"] - marks["aeprep_begin"]) / f if "aeprep_end" in marks else -1.0
    row["aeprep_cpu"] = (cpus["aeprep_end"] - cpus["aeprep_begin"]) / 10000.0 if "aeprep_end" in cpus else -1.0
    row["rep"] = (marks["rep_end"] - marks["rep_begin"]) / f if "rep_end" in marks else -1.0
    row["rb"] = (marks["rb_end"] - marks["rb_begin"]) / f if "rb_end" in marks else -1.0
    row["rb_cpu"] = (cpus["rb_end"] - cpus["rb_begin"]) / 10000.0 if "rb_end" in cpus else -1.0
    for name, a, b in SEGMENTS:
        row[name] = (marks[b] - marks[a]) / f if (a in marks and b in marks) else -1.0
        row[name + "_cpu"] = (cpus[b] - cpus[a]) / 10000.0 if (a in cpus and b in cpus) else -1.0
    covered = sum(v for k, v in row.items() if k in [s[0] for s in SEGMENTS] and v > 0)
    row["unattributed"] = row["S2"] - covered
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        row[name] = sizes[name][0]
        row[name + "_sha"] = sizes[name][1]
    return row


def main():
    exe = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(WORK, "breakdown.csv")
    log = os.path.join(WORK, "cli.log")
    if os.path.exists(log):
        os.remove(log)

    cols = ["idx", "cond", "S2"] + [s[0] for s in SEGMENTS] + \
           ["unattributed", "total", "S2_cpu", "prep", "prep_cpu", "aeprep", "aeprep_cpu",
            "rep", "rb", "rb_cpu", "marks",
            "noise_shaper_learn.xml", "noise_shaper_learn.xml_sha",
            "learned_state.xml", "learned_state.xml_sha"]
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        out.write(",".join(cols) + "\n")
        for i in range(runs):
            for cond in (("E0", "E2") if i % 2 == 0 else ("E2", "E0")):
                row = run_once(exe, cond, i, log)
                vals = [str(i * 2 + (0 if cond == "E0" else 1)), cond]
                vals += ["%.1f" % row["S2"]]
                vals += ["%.1f" % row[s[0]] for s in SEGMENTS]
                vals += ["%.1f" % row["unattributed"], "%.1f" % row["total"],
                         "%.1f" % row["S2_cpu"], "%.1f" % row["prep"], "%.1f" % row["prep_cpu"],
                         "%.1f" % row["aeprep"], "%.1f" % row["aeprep_cpu"],
                         "%.1f" % row["rep"], "%.1f" % row["rb"], "%.1f" % row["rb_cpu"],
                         str(row["marks"]),
                         str(row["noise_shaper_learn.xml"]), row["noise_shaper_learn.xml_sha"],
                         str(row["learned_state.xml"]), row["learned_state.xml_sha"]]
                out.write(",".join(vals) + "\n")
                out.flush()
                print("%-3s S2=%7.1f noise=%6.1f close=%5.1f file=%5.1f parse=%5.1f "
                      "init=%7.1f def_init=%7.1f channel=%5.1f restore=%6.1f "
                      "unattr=%6.1f" % (row["cond"], row["S2"], row["noise"], row["close"],
                                        row["file"], row["parse"], row["init"],
                                        row["default_init"], row["channel"],
                                        row["restore"], row["unattributed"]), flush=True)


if __name__ == "__main__":
    main()
