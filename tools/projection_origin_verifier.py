#!/usr/bin/env python3
"""
projection_origin_verifier.py

Phase-2: Verifies that Projection fields (in RuntimeGraph, GlobalSnapshot)
do NOT depend on or cause semantic updates. Projections must be read-only
mirrors of authoritative semantic state.

Usage:
    python projection_origin_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Projection fields in RuntimeGraph (must not be written to authority structures)
PROJECTION_FIELDS = [
    'activeNode',
    'fadingNode',
    'eqAgcAttackCoeffTable',
    'eqAgcReleaseCoeffTable',
    'eqAgcSmoothCoeffTable',
    'eqAgcCoeffTableCapacity',
]


def scan_projection_to_semantic_flow(filepath):
    """Scan for projection-to-semantic data flow (forbidden pattern)."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        for field in PROJECTION_FIELDS:
            if field not in stripped:
                continue

            # Check if projection field is being assigned to a semantic field
            # Pattern: semantic_field = ... graph.field ... or ... snapshot.field ...
            if 'graph.' + field in stripped and '=' in stripped:
                lhs = stripped.split('=')[0].strip()
                # If LHS references a non-projection structure, flag it
                if any(s in lhs for s in ['worldOwner->', 'world->', 'runtime.', 'engine.']):
                    issues.append((i, f"Projection->Semantic flow: {stripped[:80]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Projection Origin Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"), help="Source directory")
    args = parser.parse_args()

    all_issues = {}
    for root, dirs, files in os.walk(args.src):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)
                issues = scan_projection_to_semantic_flow(filepath)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[FAIL] Found {total} projection-to-semantic flow(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
        return 1

    print("[PASS] No projection-to-semantic flow detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
