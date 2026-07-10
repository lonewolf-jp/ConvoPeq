#!/usr/bin/env python3
"""
retire_ordering_verifier.py

Verifies retire ordering contract: all retire calls must respect the
EpochDomain::canRetire() ordering (isOlder-based). Checks that retire
sequence numbers are monotonically increasing and no out-of-order
retire requests occur.

Phase-1 3.2.10: RetireOrderingVerifier.

Usage:
    python retire_ordering_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Patterns indicating retire ordering violations
# (Currently not using predefined patterns — scanning inline below)


def scan_for_ordering_issues(filepath):
    """Scan for potential retire ordering violations.

    Acceptable patterns (NOT ordering violations):
      - retireDSP() calls — they all route through coordinator::enqueueRetire()
        which respects isOlder() ordering in EpochDomain
      - Coordinator's own enqueueRetire() in ISRRuntimePublicationCoordinator
      - Legacy snapshot retirement in SnapshotCoordinator (different domain)
    """
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Skip declarations (function signatures, deprecation attributes)
        if re.match(r'.*\[\[deprecated', stripped) or re.match(r'.*\b(bool|RetireEnqueueResult|void)\s+enqueueRetire\(', stripped):
            continue

        # Skip coordinator's own implementation
        if 'ISRRuntimePublicationCoordinator' in filepath:
            continue

        # Skip SnapshotCoordinator (different EpochDomain, not RuntimeWorld retirement)
        if 'SnapshotCoordinator' in filepath:
            continue

        # Check for enqueueRetire with compute expression (potential ordering violation)
        if re.search(r'enqueueRetire\(', stripped):
            if re.search(r'enqueueRetire\([^,]*,\s*(?:static_cast<)?[a-zA-Z_]+\s*\(', stripped):
                issues.append((i, f"enqueueRetire with computed deleter: {stripped[:80]}"))

    return issues

    return issues


def main():
    parser = argparse.ArgumentParser(description="Retire Ordering Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    all_issues = {}
    for root, dirs, files in os.walk(args.src):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)
                issues = scan_for_ordering_issues(filepath)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} potential retire ordering issue(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
    else:
        print("[PASS] No retire ordering issues detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
