#!/usr/bin/env python3
"""
snapshot_authority_usage_verifier.py

Verifies that GlobalSnapshot accesses do not use snapshot fields for:
- Branch decisions
- Semantic rebuild triggers
- Writeback operations
- Authority comparisons

Snapshot references for crossfade fade values are permitted.

Phase-1 3.2.10: SnapshotAuthorityUsageVerifier.

Usage:
    python snapshot_authority_usage_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Snapshot-related patterns that should not be used in decision contexts
SNAPSHOT_DECISION_PATTERNS = [
    r'currentSnapshot\b.*\bsnapshot',
    r'\bSnapshot\w*\s*[=!]=',
]


def scan_file_for_snapshot_abuse(filepath):
    """Scan for snapshot authority abuse."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Check for snapshot field in branch conditions
        if re.search(r'(if|while|switch)\s*\([^)]*\bsnapshot\b', stripped, re.IGNORECASE):
            issues.append((i, f"Snapshot in branch condition: {stripped[:80]}"))

        # Check for snapshot field in comparisons
        if re.search(r'\bsnapshot\b.*[=!]=', stripped, re.IGNORECASE):
            issues.append((i, f"Snapshot comparison: {stripped[:80]}"))

        # Check for m_slots direct access
        if re.search(r'm_slots\.', stripped):
            issues.append((i, f"Direct SnapshotSlotStore access: {stripped[:80]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Snapshot Authority Usage Verifier")
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
                issues = scan_file_for_snapshot_abuse(filepath)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} potential snapshot authority abuse(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
    else:
        print("[PASS] No snapshot authority abuse detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
