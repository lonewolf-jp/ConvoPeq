#!/usr/bin/env python3
"""
diagnostic_field_verifier.py

Phase-2: Verifies that diagnostic fields are NOT used in decision-making
contexts (if/while/switch conditions, ternary operators).

Diagnostic fields (AuthorityClass::Diagnostic) must be trace/correlation only
and must NOT drive runtime branching.

Usage:
    python diagnostic_field_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known Diagnostic fields across the codebase
DIAGNOSTIC_FIELDS = [
    'worldId',
    'runtimeVersion',
    'transitionId',
    'captureSessionId',
    'affinity',
    'projectionFreshness',
    'semanticHash',
    'eqAgcAttackCoeffTable',
    'eqAgcReleaseCoeffTable',
    'eqAgcSmoothCoeffTable',
    'eqAgcCoeffTableCapacity',
]


def scan_for_diagnostic_decisions(filepath):
    """Scan for diagnostic fields used in decision contexts."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        # Skip declarations
        if re.match(r'.*(std::atomic|uint64_t|int|bool|double)\s+\w+', stripped):
            continue

        for field in DIAGNOSTIC_FIELDS:
            if field not in stripped:
                continue
            # In a struct/class definition line
            if re.match(r'\s+\S+\s+' + field + r'\s*(?:=|;)', stripped):
                continue
            # In a descriptor/inventory array entry
            if '"' + field + '"' in stripped:
                continue

            # Check conditional usage
            cond = re.search(r'(if|while|switch)\s*\([^)]*' + field, stripped)
            tern = re.search(r'\b' + field + r'\s*\?', stripped)
            cmp = re.search(r'\b' + field + r'\s*[=!]=', stripped)

            if cond:
                issues.append((i, f"Diagnostic field in condition: {stripped[:80]}"))
            if tern:
                issues.append((i, f"Diagnostic field in ternary: {stripped[:80]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Diagnostic Field Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"), help="Source directory")
    args = parser.parse_args()

    all_issues = {}
    for root, dirs, files in os.walk(args.src):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)
                issues = scan_for_diagnostic_decisions(filepath)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} diagnostic field decision usage(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
    else:
        print("[PASS] No diagnostic field decision usage detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
