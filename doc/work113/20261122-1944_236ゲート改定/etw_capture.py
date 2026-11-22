"""Capture one WPR trace (CPU + FileIO) around a single app run.

Usage: python etw_capture.py <exe> <cond> <out.etl>
"""
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_guard

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
    log = os.path.join(WORK, "etw_cli.log")
    marks = os.path.join(WORK, "etw_marks.txt")

    # prepare() rewrites the profile, so the snapshot has to exist before it runs
    # and has to be restored even when wpr cannot start.
    guard = profile_guard.Guard()
    guard.save()
    failure = None
    try:
        prepare(cond)
        for p in (log, marks):
            if os.path.exists(p):
                os.remove(p)

        subprocess.run(["wpr", "-cancel"], capture_output=True)
        r = subprocess.run(["wpr", "-start", "CPU", "-start", "FileIO"],
                           capture_output=True, text=True)
        print("wpr start rc", r.returncode, r.stdout.strip()[:120], r.stderr.strip()[:120])
        if r.returncode != 0:
            failure = "wpr start failed"
        else:
            env = dict(os.environ)
            env["CONVOPEQ_M236_TRACE"] = marks
            started = False
            pid = -1
            try:
                proc = subprocess.Popen([exe, "--cli-run", "--cli-exit-ms", "2000",
                                         "--cli-log-file", log],
                                        env=env, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL)
                started = True
                pid = proc.pid
                try:
                    proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    # an incomplete run has no usable trace: report it as a failure
                    failure = "application timed out after 120 s"
            finally:
                # never leave a recording running, even if the launch failed
                r = subprocess.run(["wpr", "-stop", out] if started else ["wpr", "-cancel"],
                                   capture_output=True, text=True)
            if r.returncode != 0 and failure is None:
                failure = "wpr stop failed: %s" % (r.stdout + r.stderr).strip()[:200]
            print("wpr stop rc", r.returncode, r.stdout.strip()[:160])
            print("pid", pid, "etl",
                  os.path.getsize(out) if os.path.exists(out) else -1)
            print("marks",
                  open(marks).read().strip().replace("\n", " ")
                  if os.path.exists(marks) else "none")
    finally:
        guard.restore()

    if failure is not None:
        raise SystemExit(failure)


if __name__ == "__main__":
    main()
