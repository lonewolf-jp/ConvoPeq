"""Compare init cost across JUCE device types (ASIO vs WASAPI vs DirectSound)."""
import io
import os
import re
import shutil
import subprocess
import sys

WORK = r"C:\Users\user\AppData\Local\Temp\opencode\m236"
APPDATA = os.path.join(os.environ["APPDATA"], "ConvoPeq")
FIX = r"C:\Users\user\AppData\Local\Temp\opencode\m24run\fixtures"
EXE = r"C:\VSC_Project\fix236\build-msvc\ConvoPeq_artefacts\Release\ConvoPeq.exe"

CASES = [
    ("asio", "ASIO", "Voicemeeter Virtual ASIO", "Voicemeeter Virtual ASIO"),
    ("wasapi", "Windows Audio",
     "Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
     "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"),
    ("directsound", "DirectSound",
     "Voicemeeter Input (VB-Audio Voicemeeter VAIO)",
     "Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)"),
]


def build_xml(dev_type, out_name, in_name):
    t = open(os.path.join(FIX, "device_settings.xml"), encoding="utf-8").read()
    t = re.sub(r'deviceType="[^"]*"', 'deviceType="%s"' % dev_type, t)
    t = re.sub(r'audioOutputDeviceName="[^"]*"', 'audioOutputDeviceName="%s"' % out_name, t)
    t = re.sub(r'audioInputDeviceName="[^"]*"', 'audioInputDeviceName="%s"' % in_name, t)
    t = re.sub(r'audioDeviceRate="[^"]*"', 'audioDeviceRate="48000.0"', t)
    t = re.sub(r'audioDeviceBufferSize="[^"]*"', 'audioDeviceBufferSize="512"', t)
    return t


def run(tag, text, runs):
    rows = []
    for i in range(runs):
        for name in ("noise_shaper_learn.xml", "learned_state.xml"):
            p = os.path.join(APPDATA, name)
            if os.path.exists(p):
                os.remove(p)
        open(os.path.join(APPDATA, "device_settings.xml"), "w", encoding="utf-8", newline="\n").write(text)
        trace = os.path.join(WORK, "type_%s_%d.txt" % (tag, i))
        log = os.path.join(WORK, "type_%s.log" % tag)
        if os.path.exists(trace):
            os.remove(trace)
        env = dict(os.environ)
        env["CONVOPEQ_M236_TRACE"] = trace
        proc = subprocess.Popen([EXE, "--cli-run", "--cli-exit-ms", "1000", "--cli-log-file", log],
                                env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.wait()
        marks, cpus = {}, {}
        for line in io.open(trace, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) >= 2:
                marks[p[0]] = int(p[1])
                cpus[p[0]] = int(p[2]) if len(p) > 2 else 0
        logtext = io.open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""
        fallback = "restore failed" in logtext or "relaxed retry" in logtext
        rows.append((tag, (marks["init_end"] - marks["init_begin"]) / 1e4,
                     (cpus["init_end"] - cpus["init_begin"]) / 1e4,
                     (marks["s2_end"] - marks["s2_begin"]) / 1e4, fallback))
    return rows


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out_rows = []
    for tag, dev, out_name, in_name in CASES:
        out_rows += run(tag, build_xml(dev, out_name, in_name), runs)
    for r in out_rows:
        print("%-12s init=%7.1f cpu=%7.1f S2=%7.1f fallback=%s" % r)
    with open(os.path.join(WORK, "type_sweep.csv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("type,init_ms,init_cpu_ms,S2_ms,fallback\n")
        for r in out_rows:
            f.write("%s,%.1f,%.1f,%.1f,%s\n" % r)
