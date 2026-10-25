#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_header_odr.py - ODR gate for the engine headers.

Why: AudioEngine.h used to add a DATA MEMBER (`testFadingRuntimePresent_`) inside
`#if defined(CONVOPEQ_UNIT_TESTS)`. Engine TUs compiled with the test macro and plugin TUs
compiled without it therefore saw DIFFERENT class layouts in one program -> ODR/layout
mismatch (silent memory corruption, not a link error).

What: preprocess one probe TU twice - once WITH and once WITHOUT CONVOPEQ_UNIT_TESTS - and
require the resulting token streams to be identical (line markers normalised). The command
line is taken verbatim from the project's layout-probe batch file (so the include/define
set is exactly the one the TU sees) and only the probe path / mode / macro are changed.

Usage:
    python tools/check_header_odr.py [--keep]
Exit code 0 = identical, 1 = divergence found (details printed), 2 = environment error.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "tmp", "odr_probe.cpp")
BAT = os.path.join(tempfile.gettempdir(), "opencode", "crash", "layout", "run_layout.bat")

PROBE_SRC = r"""// ODR probe: every header that defines a long-lived engine type.
#include "audioengine/AudioEngine.h"
#include "audioengine/ISRShutdown.h"
#include "audioengine/ISRDSPHandle.h"
#include "audioengine/ISRRetireRouter.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"
int convopeq_odr_probe_translation_unit = 0;
"""


def cl_line():
    if not os.path.isfile(BAT):
        return None
    for line in open(BAT, "r", encoding="utf-8", errors="replace"):
        s = line.strip()
        if s.startswith("cl "):
            return s
    return None


def make_command(with_macro, out_path):
    base = cl_line()
    if base is None:
        return None, None
    # switch to preprocess-only
    base = base.replace(" /c     ", " ").replace(" /c ", " ").replace(" /Zs ", " ")
    base = base.replace(" /d1reportSingleClassLayoutAudioEngine ", " ")
    # replace the probe path + redirect with ours
    replacement = '"%s" > "%s" 2>&1' % (PROBE, out_path)
    base = re.sub(r'"[^"]*layout_probe\.cpp"\s*>\s*"[^"]*"',
                  lambda m: replacement, base)
    base = re.sub(r'"[^"]*layout_probe\.cpp".*$', lambda m: replacement, base)
    if with_macro:
        if "/DCONVOPEQ_UNIT_TESTS=1" not in base:
            base = base.replace("cl /nologo", "cl /nologo /DCONVOPEQ_UNIT_TESTS=1", 1)
    else:
        base = base.replace(" /DCONVOPEQ_UNIT_TESTS=1", "")
    base = base.replace("cl /nologo", "cl /nologo /E", 1)
    return base, out_path


def preprocess(with_macro, out_path, bat_path):
    cmd, _ = make_command(with_macro, out_path)
    if cmd is None:
        return None
    with open(bat_path, "w", encoding="ascii", newline="\r\n") as f:
        f.write("@echo off\n")
        f.write('call "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\VC'
                '\\Auxiliary\\Build\\vcvars64.bat" >nul 2>&1\n')
        f.write(cmd + "\n")
    return subprocess.run(["cmd", "/c", bat_path], cwd=ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace")


def normalise(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#line") or line.startswith("# "):
                continue
            s = " ".join(line.split())
            if not s:
                continue
            out.append(s)
    return out


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        i = min(len(a), len(b))
        return i, a[i] if i < len(a) else "<eof>", b[i] if i < len(b) else "<eof>"
    return None


def main():
    keep = "--keep" in sys.argv
    os.makedirs(os.path.dirname(PROBE), exist_ok=True)
    with open(PROBE, "w", encoding="utf-8", newline="\n") as f:
        f.write(PROBE_SRC)

    if cl_line() is None:
        print("ENVIRONMENT: no 'cl ' command line found in %s" % BAT)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="odr_")
    with_macro = os.path.join(tmpdir, "with_macro.i")
    without = os.path.join(tmpdir, "without_macro.i")

    p1 = preprocess(True, with_macro, os.path.join(tmpdir, "run_with.bat"))
    p2 = preprocess(False, without, os.path.join(tmpdir, "run_without.bat"))

    for label, path, proc in (("WITH", with_macro, p1), ("WITHOUT", without, p2)):
        if proc is None or not os.path.isfile(path) or os.path.getsize(path) == 0:
            print("ENVIRONMENT: preprocessing %s the macro failed" % label)
            if proc is not None and proc.stdout:
                print(proc.stdout[-2000:])
            return 2

    a = normalise(with_macro)
    b = normalise(without)
    d = first_diff(a, b)
    print("check_header_odr: with=%d lines, without=%d lines" % (len(a), len(b)))
    if d is None:
        print("RESULT: PASS - identical token streams with and without CONVOPEQ_UNIT_TESTS")
        return 0

    i, x, y = d
    print("RESULT: FAIL - header definitions diverge at normalised line %d" % i)
    print("  WITH   : %s" % x[:300])
    print("  WITHOUT: %s" % y[:300])
    lo = max(0, i - 3)
    print("  context (WITH):")
    for l in a[lo:i + 3]:
        print("    " + l[:200])
    print("  context (WITHOUT):")
    for l in b[lo:i + 3]:
        print("    " + l[:200])
    if keep:
        print("preprocessed files kept in %s" % tmpdir)
    return 1


if __name__ == "__main__":
    sys.exit(main())
