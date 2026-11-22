"""#236 post-fix campaign: E0-E3, 20 rounds, balanced Latin square (C-24 conditions)."""
import ctypes
import hashlib
import io
import os
import shutil
import subprocess
import sys

EXE = r"C:\VSC_Project\fix236\build-msvc\ConvoPeq_artefacts\Release\ConvoPeq.exe"
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
OUT = os.path.join(WORK, "campaign236.csv")
LOG = os.path.join(WORK, "campaign_cli.log")

kernel32 = ctypes.windll.kernel32
freq = ctypes.c_longlong()
kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
F = freq.value / 1000.0


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def prepare(cond):
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


def run_once(cond, round_idx, pos):
    prepare(cond)
    sizes = {}
    for name in ("noise_shaper_learn.xml", "learned_state.xml", "device_settings.xml"):
        p = os.path.join(APPDATA, name)
        sizes[name] = (os.path.getsize(p), sha(p)) if os.path.exists(p) else (0, "")
    trace = os.path.join(WORK, "camp_%s_%02d_%d.txt" % (cond, round_idx, pos))
    if os.path.exists(trace):
        os.remove(trace)
    env = dict(os.environ)
    env["CONVOPEQ_M236_TRACE"] = trace
    proc = subprocess.Popen([EXE, "--cli-run", "--cli-exit-ms", "1000", "--cli-log-file", LOG],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    m, c = {}, {}
    for line in io.open(trace, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) >= 2:
            m[p[0]] = int(p[1])
            c[p[0]] = int(p[2]) if len(p) > 2 else 0

    def d(a, b):
        return (m[b] - m[a]) / F if (a in m and b in m) else -1.0

    def dc(a, b):
        return (c[b] - c[a]) / 10000.0 if (a in c and b in c) else -1.0

    return {
        "round": round_idx + 1, "pos": pos + 1, "cond": cond,
        "S1": d("app_init_start", "app_init_end"),
        "S2": d("s2_begin", "s2_end"),
        "S3": d("shutdown_start", "shutdown_end"),
        "S4": d("s4_begin", "s4_end"),
        "S2_noise": d("noise_begin", "noise_end"),
        "S2_close": d("close_begin", "close_end"),
        "S2_file": d("file_begin", "file_end"),
        "S2_parse": d("parse_begin", "parse_end"),
        "S2_init": d("init_begin", "init_end"),
        "S2_prep": d("prep_begin", "prep_end"),
        "S2_channel": d("channel_begin", "channel_end"),
        "S2_restore": d("restore_begin", "restore_end"),
        "S2_final_rebuild": d("restore_end", "s2_end"),
        "S2_cpu": dc("s2_begin", "s2_end"),
        "init_cpu": dc("init_begin", "init_end"),
        "nsl": sizes["noise_shaper_learn.xml"][0], "nsl_sha": sizes["noise_shaper_learn.xml"][1],
        "ls": sizes["learned_state.xml"][0], "ls_sha": sizes["learned_state.xml"][1],
        "ds_sha": sizes["device_settings.xml"][1],
    }


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    conds = ["E0", "E1", "E2", "E3"]
    square = [[0, 1, 2, 3], [1, 3, 0, 2], [2, 0, 3, 1], [3, 2, 1, 0]]
    row_order = [0, 3, 2, 2, 1, 2, 0, 2, 1, 2, 3, 0, 1, 3, 3, 1, 1, 0, 0, 3]
    rows = [square[i] for i in row_order[:rounds]]
    cols = ["round", "pos", "cond", "S1", "S2", "S3", "S4", "S2_noise", "S2_close",
            "S2_file", "S2_parse", "S2_init", "S2_prep", "S2_channel", "S2_restore",
            "S2_final_rebuild", "S2_cpu", "init_cpu", "nsl", "nsl_sha", "ls", "ls_sha", "ds_sha"]
    with open(OUT, "w", encoding="utf-8", newline="\n") as out:
        out.write(",".join(cols) + "\n")
        for r, order in enumerate(rows):
            for pos, idx in enumerate(order):
                row = run_once(conds[idx], r, pos)
                out.write(",".join(
                    [str(row["round"]), str(row["pos"]), row["cond"]] +
                    ["%.1f" % row[k] for k in cols[3:17]] +
                    [str(row["nsl"]), row["nsl_sha"], str(row["ls"]), row["ls_sha"], row["ds_sha"]]) + "\n")
                out.flush()
                print("r%02d p%d %s S1=%6.0f S2=%6.1f init=%6.1f restore=%5.1f final=%5.1f S3=%6.1f S4=%5.1f" % (
                    r + 1, pos + 1, row["cond"], row["S1"], row["S2"], row["S2_init"],
                    row["S2_restore"], row["S2_final_rebuild"], row["S3"], row["S4"]), flush=True)


if __name__ == "__main__":
    main()
