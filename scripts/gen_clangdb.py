#!/usr/bin/env python3
"""Generate static-analysis compilation databases for clang-tidy and cppcheck.

The repository-root ``compile_commands.json`` is produced by the Ninja Multi-Config
generator and is **not** a valid JSON array: it has no enclosing ``[`` ... ``]`` and
its object separators are trailing commas. Neither clang-tidy nor cppcheck can parse
it directly, so this script normalises it into two purpose-built databases:

  tmp/clangdb/compile_commands.json          (arguments form, MSVC -> clang flags)
  tmp/cppcheckdb/compile_commands.json       (command form, MSVC flags kept)

Why two databases
-----------------
clang-tidy drives clang, which rejects MSVC-style switches (``/D``, ``/I``, ``/O2``,
``-Fo...``). It also requires *exactly one* compile job per source file, so the
Multi-Config duplicates (192 files x 3 configs = 1584 entries) must be collapsed.
cppcheck understands the MSVC dialect but chokes on the clang-translated form.

Usage
-----
    python scripts/gen_clangdb.py                 # regenerate both databases
    python scripts/gen_clangdb.py --config Debug  # prefer a specific build config

Then:
    clang-tidy -p tmp/clangdb src/audioengine/AudioEngine.Threading.cpp
    cppcheck --project=tmp/cppcheckdb/compile_commands.json --enable=warning,performance,portability \
             --suppress=missingIncludeSystem '-DJUCE_DECLARE_WEAK_REFERENCEABLE(x)=' -q \
             --file-filter='*AudioEngine.Threading.cpp'
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DB = os.path.join(REPO_ROOT, "compile_commands.json")
CLANG_DB_DIR = os.path.join(REPO_ROOT, "tmp", "clangdb")
CPPCHECK_DB_DIR = os.path.join(REPO_ROOT, "tmp", "cppcheckdb")
CLANG_DRIVER = r"C:\Program Files\LLVM\bin\clang++.exe"

BUILD_CONFIGS = ("RelWithDebInfo", "Debug", "Release")
SOURCE_SUFFIXES = (".cpp", ".c", ".cc", ".cxx", ".c++", ".m", ".mm")

# MSVC switch -> clang equivalent. Empty list means "drop it".
FLAG_MAP: dict[str, list[str]] = {
    "/nologo": [], "/TP": ["-xc++"], "/EHsc": ["-fexceptions"],
    "/Zi": [], "/Z7": [], "/RTC1": [], "/utf-8": ["-finput-charset=UTF-8"],
    "/MP1": [], "/bigobj": [], "/W4": ["-Wall", "-Wextra"], "/W3": ["-Wall"],
    "/WX": ["-Werror"],
    "/wd4100": ["-Wno-unused-variable"], "/wd4189": ["-Wno-unused-local-typedefs"],
    "/wd4244": ["-Wno-conversion"], "/wd4459": ["-Wno-declaration-after-statement"],
    "/wd4996": ["-Wno-deprecated-declarations"],
    "/O2": ["-O2"], "/O1": ["-O1"], "/O0": [], "/Od": [], "/Ob2": [], "/Ob1": [], "/Ob0": [],
    "/Oi": ["-finline-functions"], "/Oy": [], "/Gw": [], "/GF": [],
    "/MD": [], "/MT": [], "/MDd": [], "/MTd": [],
    "/arch:AVX2": ["-mavx2"], "/arch:AVX": ["-mavx"], "/arch:SSE2": ["-msse2"],
    "/fp:precise": [], "/fp:fast": ["-ffast-math"],
    "/Zc:__cplusplus": [], "/Zc:preprocessor": [], "/permissive-": [],
    "/diagnostics:caret": [], "/FS": [], "/FD": [], "/JMC": [],
}

# Prefixes of MSVC switches that only affect the build driver, never the frontend.
DROP_PREFIXES = (
    "/Fo", "/Fd", "/Fe", "/F6", "/Fm", "/FR", "/link", "/MANIFEST",
    "/MANIFESTUAC", "/INCREMENTAL", "/NOLOGO", "/sourceDependencies",
    "/analyze", "/guard:", "/errorReport", "/pdb", "/showIncludes",
)


def load_entries(path: str) -> list[dict]:
    """Parse the Ninja Multi-Config compile_commands.json.

    Handles three defects in the generated file: a missing enclosing array,
    trailing commas between objects, and a stray closing bracket at EOF.
    """
    with open(path, encoding="utf-8-sig") as fh:
        raw = fh.read()
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)

    decoder = json.JSONDecoder()
    entries: list[dict] = []
    pos, end = 0, len(raw)
    while pos < end:
        while pos < end and raw[pos] in " \t\r\n,]":
            pos += 1
        if pos >= end:
            break
        try:
            obj, pos = decoder.raw_decode(raw, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def config_of(command: str) -> str:
    match = re.search(r'CMAKE_INTDIR=..([A-Za-z]+)', command or "")
    return match.group(1) if match else "none"


def unquote(arg: str) -> str:
    if len(arg) > 1 and arg[0] == '"' and arg[-1] == '"':
        return arg[1:-1]
    return arg


def to_clang_arg(arg: str) -> list[str]:
    """Translate one MSVC-style switch into zero or more clang arguments."""
    if "=" in arg and arg.startswith("/"):
        key, value = arg.split("=", 1)
        return {"/I": ["-I" + value], "/D": ["-D" + value], "/U": []}.get(key, [])

    upper = arg.upper()
    if upper in FLAG_MAP:
        return FLAG_MAP[upper]
    if any(upper.startswith(prefix.upper()) for prefix in DROP_PREFIXES):
        return []
    if upper.startswith("/D"):
        return ["-D" + arg[2:]] if len(arg) > 2 else []
    if upper.startswith("/I"):
        return ["-I" + arg[2:]]
    if upper.startswith("/STD:"):
        return ["-std=" + arg[5:]]
    if upper.startswith("/W"):
        return []
    # Anything not MSVC-shaped (e.g. clang's own -isystem / -internal-isystem) passes through.
    return [] if arg.startswith("/") else [arg]


def build_clang_entry(entry: dict) -> dict:
    args: list[str] = []
    for raw in shlex.split(entry["command"], posix=False):
        arg = unquote(raw)
        if arg.lower().endswith("cl.exe"):
            continue
        if arg in ("/c", "-c"):
            continue
        if arg.lower().endswith(SOURCE_SUFFIXES):
            continue  # clang-tidy appends the file it was asked to check
        args.extend(to_clang_arg(arg))

    # clang's driver splits "-x c++" into two tokens, which the JSON database
    # replays verbatim; the single-token spelling survives the round trip.
    if "-x" in args:
        idx = args.index("-x")
        if idx + 1 < len(args) and args[idx + 1] == "c++":
            args[idx : idx + 2] = ["-xc++"]

    args.append(entry["file"])
    return {"directory": entry["directory"], "file": entry["file"], "arguments": [CLANG_DRIVER] + args}


def build_cppcheck_entry(entry: dict) -> dict:
    # cppcheck understands MSVC switches but not the output-file flags.
    command = re.sub(r"\s/(Fo|Fd|Fi|FS)[^\s]*", "", entry["command"])
    return {"directory": entry["directory"], "file": entry["file"], "command": command}


def pick_one_per_file(entries: list[dict], preferred: tuple[str, ...], builder) -> list[dict]:
    chosen: dict[str, dict] = {}
    for config in preferred:
        for entry in entries:
            if config_of(entry.get("command", "")) == config and entry["file"] not in chosen:
                chosen[entry["file"]] = builder(entry)
    return list(chosen.values())


def write_db(directory: str, stem: str, payload: list[dict]) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, stem)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="preferred build config (RelWithDebInfo default)")
    args = parser.parse_args()

    if not os.path.isfile(SRC_DB):
        print(f"error: {SRC_DB} not found - configure the build first", file=sys.stderr)
        return 1

    entries = load_entries(SRC_DB)
    if not entries:
        print("error: no compile entries parsed", file=sys.stderr)
        return 1

    preferred = (args.config,) + tuple(c for c in BUILD_CONFIGS if c != args.config) if args.config else BUILD_CONFIGS
    configs = sorted({config_of(e.get("command", "")) for e in entries})

    clang_all = pick_one_per_file(entries, preferred, build_clang_entry)
    cppcheck_all = pick_one_per_file(entries, preferred, build_cppcheck_entry)

    print(f"parsed {len(entries)} entries, {len({e['file'] for e in entries})} source files, configs={configs}")
    for config in BUILD_CONFIGS:
        subset = pick_one_per_file(entries, (config,), build_clang_entry)
        if subset:
            path = write_db(CLANG_DB_DIR, f"compile_commands_{config}.json", subset)
            print(f"  clang-tidy  {config:<16} {len(subset):>4} files -> {os.path.relpath(path, REPO_ROOT)}")

    path = write_db(CLANG_DB_DIR, "compile_commands.json", clang_all)
    print(f"  clang-tidy  {'(default)':<16} {len(clang_all):>4} files -> {os.path.relpath(path, REPO_ROOT)}")
    path = write_db(CPPCHECK_DB_DIR, "compile_commands.json", cppcheck_all)
    print(f"  cppcheck    {'(default)':<16} {len(cppcheck_all):>4} files -> {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
