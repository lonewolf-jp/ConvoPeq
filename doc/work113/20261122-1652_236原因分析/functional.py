"""Functional checks for device restore paths (issue #236)."""
import io
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_guard

WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
EXE = os.path.join(WORK, "exe_post.exe")


def run(tag, setup):
    for name in ("noise_shaper_learn.xml", "learned_state.xml"):
        p = os.path.join(APPDATA, name)
        if os.path.exists(p):
            os.remove(p)
    log = os.path.join(WORK, "func_%s.log" % tag)
    if os.path.exists(log):
        os.remove(log)
    setup()
    proc = subprocess.Popen([EXE, "--cli-run", "--cli-exit-ms", "700", "--cli-log-file", log],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    text = io.open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
    created = os.path.exists(os.path.join(APPDATA, "device_settings.xml"))
    size = os.path.getsize(os.path.join(APPDATA, "device_settings.xml")) if created else 0
    flags = []
    for kw in ("restore failed", "relaxed retry", "using default device", "Could not restore"):
        if kw in text:
            flags.append(kw)
    print("%-10s rc=%d settings_written=%s size=%d flags=%s shutdown_end=%s" % (
        tag, proc.returncode, created, size, flags or "-", "SHUTDOWN_END" in text))
    return proc.returncode, flags


def no_file():
    p = os.path.join(APPDATA, "device_settings.xml")
    if os.path.exists(p):
        os.remove(p)


def corrupt():
    open(os.path.join(APPDATA, "device_settings.xml"), "w", encoding="utf-8", newline="\n").write("<DEVICESETUP deviceType=\"ASIO\"\n")


def bad_device():
    t = open(os.path.join(FIX, "device_settings.xml"), encoding="utf-8").read()
    t = re.sub(r'audioOutputDeviceName="[^"]*"', 'audioOutputDeviceName="Nonexistent ASIO Device"', t)
    t = re.sub(r'audioInputDeviceName="[^"]*"', 'audioInputDeviceName="Nonexistent ASIO Device"', t)
    open(os.path.join(APPDATA, "device_settings.xml"), "w", encoding="utf-8", newline="\n").write(t)


def normal():
    shutil.copyfile(os.path.join(FIX, "device_settings.xml"), os.path.join(APPDATA, "device_settings.xml"))


if __name__ == "__main__":
    guard = profile_guard.Guard()
    guard.save()
    try:
        run("normal", normal)
        run("no-file", no_file)
        run("corrupt", corrupt)
        run("bad-device", bad_device)
    finally:
        guard.restore()
