#!/usr/bin/env python3
"""
dead_code_callers_verifier.py

Verifies that designated "dead-code" functions are NOT called anywhere in the
codebase. If any call site is introduced, the script fails with a non-zero exit
code so CI can catch regressions.

Background (see doc/work89/INTEGRATED-BUG-LIST.md §11):
- DSPCore::reset() / EQProcessor::reset() / EQProcessor::syncStateFrom() /
  EQProcessor::syncGlobalStateFrom() / EQProcessor::syncBandNodeFrom() /
  ConvolverProcessor::syncStateFrom() are all dead code (zero call sites).
- This script guards against accidental re-activation (which would resurrect
  the data-race risks described in §9/§10: rt-shadow writes from Non-RT threads).

Usage:
    python tools/dead_code_callers_verifier.py [--src <path>] [--exclude <glob> ...]

Exit codes:
    0 = PASS (no call sites)
    1 = FAIL (call site(s) detected)

Known limitations:
- Comments (line / block / trailing) and string literals are stripped before
  matching, so commented-out calls are NOT flagged.
- The reset-family guard uses a whitelist of receiver names observed in the
  codebase (eqRt, uiEqEditor, eq, ...). A call through an unlisted receiver
  name would be a false negative; the sync* family additionally has a
  receiver-agnostic catch-all to reduce that risk.
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Watched functions.
#
# Each entry describes:
#   name      : display name (for messages)
#   patterns  : list of regexes that match a *call site* (not a definition or
#               declaration).
# ---------------------------------------------------------------------------

# A definition/declaration of the watched functions themselves must never be
# flagged as a call site. Note: call sites never begin with these keywords
# (a call like `eqRt().reset();` starts with the receiver expression).
DEF_OR_DECL_RE = re.compile(
    r"^\s*(?:virtual\s+)?(?:inline\s+)?(?:void|bool|auto|std::optional|int|"
    r"uint32_t|uint64_t|double|float)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_:]*::)?"
    r"(?:reset|syncStateFrom|syncGlobalStateFrom|syncBandNodeFrom)\s*\([^;]*\)"
    r"\s*(?:const\s*)?(?:noexcept)?\s*(?:;|\{)"
)

# ---------------------------------------------------------------------------
# Watched functions and their call-site patterns.
# ---------------------------------------------------------------------------

WATCHED = [
    {
        "name": "DSPCore::reset",
        "patterns": [
            # `->reset()` on the known DSPCore receiver names. We intentionally
            # do NOT match `.reset()` here: DSPCore instances are held via
            # aligned_unique_ptr (RuntimeBuilder.cpp), so `.reset()` would be
            # the unique_ptr member (memory release), not DSPCore::reset().
            r"\b(?:runtime|placeholderDSP|newDSP|currentDSP|activeDSP|fadingDSP|dspCore|core)\s*->\s*reset\s*\(\s*\)",
        ],
    },
    {
        "name": "EQProcessor::reset",
        "patterns": [
            # eqRt() returns EQProcessor& (AudioEngine.h:921). uiEqEditor is a
            # value member of type EQEditProcessor (AudioEngine.h:1221).
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor)\s*\.\s*reset\s*\(\s*\)",
            # eqRt() is a member function returning EQProcessor&; a direct
            # eqRt().reset() is a real EQProcessor::reset() call.
            r"\beqRt\s*\(\s*\)\s*\.\s*reset\s*\(\s*\)",
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor)\s*->\s*reset\s*\(\s*\)",
        ],
    },
    {
        "name": "EQProcessor::syncStateFrom",
        "patterns": [
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*\.\s*syncStateFrom\s*\(",
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*->\s*syncStateFrom\s*\(",
        ],
    },
    {
        "name": "EQProcessor::syncGlobalStateFrom",
        "patterns": [
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*\.\s*syncGlobalStateFrom\s*\(",
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*->\s*syncGlobalStateFrom\s*\(",
        ],
    },
    {
        "name": "EQProcessor::syncBandNodeFrom",
        "patterns": [
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*\.\s*syncBandNodeFrom\s*\(",
            r"\b(?:eqRt|uiEqEditor|eq|eqProcessor|eqProc|editor|processor)\s*->\s*syncBandNodeFrom\s*\(",
        ],
    },
    {
        "name": "ConvolverProcessor::syncStateFrom",
        "patterns": [
            r"\b(?:convolverRt|convolver|uiConvolverProcessor|conv|convProcessor)\s*\.\s*syncStateFrom\s*\(",
            r"\b(?:convolverRt|convolver|uiConvolverProcessor|conv|convProcessor)\s*->\s*syncStateFrom\s*\(",
        ],
    },
]

# ---------------------------------------------------------------------------
# Receiver-agnostic detection for the sync* family: these function names are
# unique enough that ANY member call (`->` or `.`) is suspicious, regardless
# of the receiver variable name. Group 1 captures the actual function name.
# ---------------------------------------------------------------------------

GENERIC_SYNC_CALL_RE = re.compile(
    r"(?:->|\.)\s*"
    r"(syncStateFrom|syncGlobalStateFrom|syncBandNodeFrom)\s*\("
)


def strip_comments_and_strings(line, in_block):
    """Remove comments (line / block / trailing) and blank string-literal
    contents so that call-site patterns inside comments or strings are never
    matched. Returns (stripped_line, in_block_after).

    Examples:
      "http://eqRt().reset()"  ->  ""          (content blanked)
      foo(); // eq.reset()      ->  foo();      (line comment dropped)
    """
    out = []
    i, n = 0, len(line)
    while i < n:
        if in_block:
            idx = line.find("*/", i)
            if idx == -1:
                return "", True  # rest of the line is inside a block comment
            i = idx + 2
            in_block = False
            continue
        c = line[i]
        if c in ('"', "'"):
            # blank the string/char literal content (keep the quotes so
            # surrounding tokens do not join), still honoring escapes so the
            # closing quote is found correctly.
            quote = c
            out.append(c)
            i += 1
            while i < n:
                if line[i] == "\\" and i + 1 < n:
                    i += 2
                    continue
                if line[i] == quote:
                    out.append(quote)
                    i += 1
                    break
                i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break  # line comment: drop the rest
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out), in_block


def iter_source_files(src_dir, exclude_globs):
    """Yield .h/.cpp files under src_dir, skipping excluded paths.

    Paths are normalized to forward slashes so fnmatch works identically on
    Windows (where os.path.relpath returns backslash-separated paths).
    """
    import fnmatch

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ("JUCE", "r8brain-free-src", "build")]
        for fname in files:
            if not fname.endswith((".h", ".hpp", ".cpp")):
                continue
            path = os.path.join(root, fname)
            rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
            if any(fnmatch.fnmatch(rel, g.replace("\\", "/")) for g in exclude_globs):
                continue
            yield path, rel


def check_file(filepath, relpath):
    """Return list of (line_no, original_line, function_name) violations."""
    found = []
    in_block = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            original = raw.rstrip("\n")
            line, in_block = strip_comments_and_strings(original, in_block)
            if not line.strip():
                continue
            if DEF_OR_DECL_RE.match(line):
                continue
            # --- receiver-agnostic sync-family member calls ---
            m = GENERIC_SYNC_CALL_RE.search(line)
            if m:
                found.append((lineno, original.strip(), "*::" + m.group(1)))
                continue
            # --- receiver-specific patterns ---
            for w in WATCHED:
                for pat in w["patterns"]:
                    if re.search(pat, line):
                        found.append((lineno, original.strip(), w["name"]))
                        break
    return found


def main():
    parser = argparse.ArgumentParser(description="Dead-code call-site verifier")
    parser.add_argument(
        "--src",
        default=os.path.join(REPO_ROOT, "src"),
        help="Source directory to scan (default: <repo>/src)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob patterns (relative to repo root) to exclude, e.g. 'src/tests/*'",
    )
    args = parser.parse_args()

    violations = []
    for filepath, rel in iter_source_files(args.src, args.exclude):
        for lineno, original, fname in check_file(filepath, rel):
            violations.append((rel, lineno, original, fname))

    if violations:
        print(f"[FAIL] Found {len(violations)} call site(s) of dead-code functions:")
        for rel, lineno, original, fname in violations:
            print(f"  - {rel}:{lineno}  [{fname}]")
            print(f"      {original}")
        print(
            "\nThese functions are intentionally dead code (see doc/work89/"
            "INTEGRATED-BUG-LIST.md §11). "
            "If you need one of them, re-design it first: Non-RT threads must "
            "NOT write rt-shadow variables (data race). Use serial-based sync "
            "(§9/§10)."
        )
        return 1

    print("[PASS] No call sites of dead-code functions detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
