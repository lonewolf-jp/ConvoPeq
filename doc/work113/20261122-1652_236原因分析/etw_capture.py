"""Capture one WPR trace (CPU + FileIO) around a single app run.

Usage: python etw_capture.py <exe> <cond> <out.etl>
"""
import os
import shutil
import subprocess
import sys
import time

APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"


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


def main():
    exe, cond, out = sys.argv[1], sys.argv[2], sys.argv[3]
    prepare(cond)
    log = os.path.join(WORK, "etw_cli.log")
    marks = os.path.join(WORK, "etw_marks.txt")
    for p in (log, marks):
        if os.path.exists(p):
            os.remove(p)

    subprocess.run(["wpr", "-cancel"], capture_output=True)
    r = subprocess.run(["wpr", "-start", "CPU", "-start", "FileIO"], capture_output=True, text=True)
    print("wpr start rc", r.returncode, r.stdout.strip()[:120], r.stderr.strip()[:120])
    if r.returncode != 0:
        raise SystemExit("wpr start failed")

    env = dict(os.environ)
    env["CONVOPEQ_M236_TRACE"] = marks
    started = False
    try:
        proc = subprocess.Popen([exe, "--cli-run", "--cli-exit-ms", "2000", "--cli-log-file", log],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        started = True
        pid = proc.pid
        proc.wait()
    finally:
        # never leave a recording running, even if the launch failed
        r = subprocess.run(["wpr", "-stop", out] if started else ["wpr", "-cancel"],
                           capture_output=True, text=True)
    print("wpr stop rc", r.returncode, r.stdout.strip()[:160])
    print("pid", pid, "etl", os.path.getsize(out) if os.path.exists(out) else -1)
    print("marks", open(marks).read().strip().replace("\n", " ") if os.path.exists(marks) else "none")


if __name__ == "__main__":
    main()
