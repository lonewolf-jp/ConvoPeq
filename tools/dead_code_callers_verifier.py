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
- The sync* family uses a receiver-agnostic catch-all; the reset family now also
  covers the wrapper chain via resetForRuntime()/ref().reset() patterns (R-2).

work89 R-2 addition (DORMANT_EDGE registry, REMEDIATION_PLAN_R123_20260910 §2 R-2):
- The wrapper chain DSPCore::reset() -> eqState->resetForRuntime() ->
  EQRuntimeState::resetForRuntime() -> ref().reset() == EQProcessor::reset() is
  *wired but dormant*: DSPCore::reset() itself has zero direct callers. The old
  receiver whitelist could not see `ref().reset()` inside the wrapper (documented
  false negative). R-2 closes that gap with an explicit registry:
    ACTIVE        : detection -> FAIL (default)
    DORMANT_EDGE  : detection -> WARN **only if** the actual callee set inside the
                    registered {file, function} matches expected_callees exactly
                    (callee addition/removal, function change, file change -> FAIL)
  The allowlist is structural ({file, function, expected_callee, reason}), never
  line-number based: "this edge is intended by design", not "this line is OK".
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

# ---------------------------------------------------------------------------
# work89 R-2: DORMANT_EDGE registry — explicit dormant wiring (structural).
# Detection patterns for the wrapper chain. Any occurrence OUTSIDE a registered
# DORMANT_EDGE region is an ACTIVE violation (FAIL).
# ---------------------------------------------------------------------------

RESET_FOR_RUNTIME_CALL_RE = re.compile(
    r"\b(?:eqState|convolverState)\s*->\s*resetForRuntime\s*\("
)
REF_RESET_RE = re.compile(r"\bref\s*\(\s*\)\s*\.\s*reset\s*\(\s*\)")

DORMANT_EDGES = [
    {
        "name": "DSPCore::reset dormant wiring",
        "kind": "function_region",
        "file": "src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp",
        "function_header_re": r"^void\s+AudioEngine::DSPCore::reset\s*\(\s*\)",
        "expected_callees": [
            (r"\bconvolverState\s*->\s*resetForRuntime\s*\(\s*\)",
             "convolverState->resetForRuntime()"),
            (r"\beqState\s*->\s*resetForRuntime\s*\(\s*\)",
             "eqState->resetForRuntime()"),
        ],
        "reason": ("intentional dormant wiring: DSPCore::reset() itself has zero direct "
                   "callers (doc/work89/REMEDIATION_PLAN_R123_20260910 §1.4); activation "
                   "requires Audio-Thread-stopped context and a fresh audit"),
    },
    {
        "name": "RuntimeState wrapper resetForRuntime definitions",
        "kind": "wrapper_structs",
        "file": "src/audioengine/AudioEngine.h",
        "wrappers": ["EQRuntimeState", "ConvolverRuntimeState"],
        "expected_callees": [
            (r"\bref\s*\(\s*\)\s*\.\s*reset\s*\(\s*\)", "ref().reset()"),
        ],
        "reason": ("wrapper definitions forwarding resetForRuntime() to "
                   "EQProcessor::reset()/ConvolverProcessor::reset() — dormant unless "
                   "the DSPCore::reset edge is activated"),
    },
]


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


def check_file(filepath, relpath, covered=None):
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
            hit = False
            for w in WATCHED:
                for pat in w["patterns"]:
                    if re.search(pat, line):
                        found.append((lineno, original.strip(), w["name"]))
                        hit = True
                        break
                if hit:
                    continue
            # --- work89 R-2: wrapper-chain patterns outside DORMANT_EDGE regions ---
            if covered is not None and (relpath, lineno) not in covered:
                if RESET_FOR_RUNTIME_CALL_RE.search(line) or REF_RESET_RE.search(line):
                    found.append((lineno, original.strip(), "*dormant-chain-active*"))
    return found


# ---------------------------------------------------------------------------
# work89 R-2: DORMANT_EDGE structural verification.
# ---------------------------------------------------------------------------


def _read_stripped(filepath):
    """Return list of (lineno, original, stripped) with comments/strings stripped."""
    out = []
    in_block = False
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            original = raw.rstrip("\n")
            stripped, in_block = strip_comments_and_strings(original, in_block)
            out.append((lineno, original, stripped))
    return out


def _depth_annotated(stripped_lines):
    """Annotate each (lineno, original, stripped) with the brace depth *before* it."""
    out = []
    depth = 0
    for lineno, original, stripped in stripped_lines:
        out.append((lineno, original, stripped, depth))
        for ch in stripped:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
    return out


def _function_region(annotated, header_re):
    """Return (lineno, stripped) lines of the top-level function whose signature
    matches header_re (region ends at the first '}' in column 0). None if absent."""
    region = []
    started = False
    for lineno, _orig, stripped, _depth in annotated:
        if not started:
            if re.match(header_re, stripped):
                started = True
                region.append((lineno, stripped))
            continue
        region.append((lineno, stripped))
        if stripped.startswith("}"):
            break
    return region if started else None


def _struct_regions(annotated, struct_names):
    """Map struct_name -> list of (lineno, stripped) lines inside `struct <Name>`
    (declaration line through the matching closing brace)."""
    names_re = re.compile(
        r"\bstruct\s+(" + "|".join(re.escape(n) for n in struct_names) + r")\b"
    )
    regions = {}
    pending_name = None
    decl_depth = None
    body = []
    seen_open = False
    for lineno, _orig, stripped, depth_before in annotated:
        depth_after = depth_before
        for ch in stripped:
            if ch == "{":
                depth_after += 1
            elif ch == "}":
                depth_after -= 1
        if pending_name is None:
            m = names_re.search(stripped)
            if m:
                pending_name = m.group(1)
                decl_depth = depth_before
                body = [(lineno, stripped)]
                seen_open = "{" in stripped
                if seen_open and depth_after == decl_depth:
                    pending_name = None  # degenerate one-line struct, nothing to watch
            continue
        body.append((lineno, stripped))
        if "{" in stripped:
            seen_open = True
        if seen_open and depth_after == decl_depth:
            regions[pending_name] = body
            pending_name = None
    return regions


def scan_dormant_edges(src_dir, exclude_globs):
    """work89 R-2: verify the DORMANT_EDGE registry.

    Returns (warns, fails, covered):
      warns   : list of (edge_name, file, reason, callee_summary) — structure matched
      fails   : list of (file, lineno, message, edge_name) — structural mismatch
      covered : set of (relpath, lineno) inside registered dormant regions (excluded
                from the ACTIVE global scan)
    """
    del exclude_globs  # registry targets are core sources; excludes apply to the active scan
    warns, fails = [], []
    covered = set()

    for edge in DORMANT_EDGES:
        target = os.path.join(REPO_ROOT, edge["file"].replace("/", os.sep))
        if not os.path.isfile(target):
            fails.append((edge["file"], 0,
                          f"DORMANT_EDGE target file missing: {edge['file']}",
                          edge["name"]))
            continue
        annotated = _depth_annotated(_read_stripped(target))
        rel = edge["file"]

        if edge["kind"] == "function_region":
            region = _function_region(annotated, edge["function_header_re"])
            if region is None:
                fails.append((rel, 0,
                              "registered function not found (moved/renamed?)",
                              edge["name"]))
                continue
            counts = {label: 0 for _re, label in edge["expected_callees"]}
            extras = []
            for lineno, stripped in region:
                matched = False
                for callee_re, label in edge["expected_callees"]:
                    if re.search(callee_re, stripped):
                        counts[label] += 1
                        matched = True
                        covered.add((rel, lineno))
                if not matched and (RESET_FOR_RUNTIME_CALL_RE.search(stripped)
                                    or REF_RESET_RE.search(stripped)):
                    extras.append(f"line {lineno}: {stripped.strip()}")
            missing = [label for label, n in counts.items() if n == 0]
            if extras or missing or any(n > 1 for n in counts.values()):
                detail = ", ".join(f"{label} x{n}" for label, n in counts.items())
                msg = f"callee set mismatch ({detail})"
                if extras:
                    msg += "; unexpected callee(s): " + "; ".join(extras)
                fails.append((rel, 0, msg, edge["name"]))
            else:
                warns.append((edge["name"], edge["file"], edge["reason"],
                              ", ".join(f"{label} x{n}" for label, n in counts.items())))

        elif edge["kind"] == "wrapper_structs":
            regions = _struct_regions(annotated, edge["wrappers"])
            for name in edge["wrappers"]:
                region = regions.get(name)
                if region is None:
                    fails.append((rel, 0,
                                  f"registered wrapper struct not found: {name}",
                                  edge["name"]))
                    continue
                count = 0
                for lineno, stripped in region:
                    for callee_re, _label in edge["expected_callees"]:
                        if re.search(callee_re, stripped):
                            count += 1
                            covered.add((rel, lineno))
                if count != 1:
                    fails.append((rel, 0,
                                  f"{name}: ref().reset() count = {count} "
                                  "(expected exactly 1)",
                                  edge["name"]))
                else:
                    warns.append((f"{edge['name']} [{name}]", edge["file"],
                                  edge["reason"], "ref().reset() x1"))

    return warns, fails, covered


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

    # --- work89 R-2: DORMANT_EDGE structural verification (first pass) ---
    d_warns, d_fails, covered = scan_dormant_edges(args.src, args.exclude)

    # --- existing dead-code call-site scan + ACTIVE wrapper-chain scan ---
    violations = []
    for filepath, rel in iter_source_files(args.src, args.exclude):
        for lineno, original, fname in check_file(filepath, rel, covered):
            violations.append((rel, lineno, original, fname))

    for w in d_warns:
        print(f"[DORMANT_EDGE][WARN] {w[0]}")
        print(f"    file   : {w[1]}")
        print(f"    callees: {w[3]}")
        print(f"    reason : {w[2]}")

    if d_fails or violations:
        if d_fails:
            print(f"[FAIL] DORMANT_EDGE structural mismatch: {len(d_fails)}")
            for rel, lineno, msg, name in d_fails:
                print(f"  - {rel}:{lineno}  [{name}]")
                print(f"      {msg}")
        if violations:
            print(f"[FAIL] Found {len(violations)} call site(s) of dead-code functions:")
            for rel, lineno, original, fname in violations:
                print(f"  - {rel}:{lineno}  [{fname}]")
                print(f"      {original}")
        print(
            "\nDORMANT_EDGE entries are structural allowlists "
            "({file, function, expected_callee, reason}). Any structural change "
            "(callee added/removed, function/file changed) is a FAIL — re-audit "
            "the dormant chain before wiring it (doc/work89/"
            "REMEDIATION_PLAN_R123_20260910 §2 R-2). "
            "If you need a dead-code function, re-design it first: Non-RT threads "
            "must NOT write rt-shadow variables (data race). Use serial-based sync."
        )
        return 1

    print(f"[PASS] No call sites of dead-code functions detected "
          f"({len(d_warns)} dormant edge(s) verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
