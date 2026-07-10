#!/usr/bin/env python3
"""
identity_authority_verifier.py

Verifies that diagnostic identity fields (runtimeVersion, transitionId, worldId)
are NOT used in conditional expressions (if/while/switch/ternary).

Phase-1 3.1: IdentityAuthorityVerifier.

Usage:
    python identity_authority_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDENTITY_FIELDS = ['runtimeVersion', 'transitionId', 'worldId']


def scan_field_usage(filepath):
    """Scan a single file for prohibited usage of identity fields."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        for field in IDENTITY_FIELDS:
            if field not in stripped:
                continue

            # Skip struct/class member declarations
            if re.search(r'\bstd::atomic.*' + field, stripped) or re.search(r'\b(uint64_t|int)\s+' + field, stripped):
                continue

            # Check for conditional usage
            if re.search(r'(if|while)\s*\([^)]*' + field, stripped):
                issues.append((i, field, f"Conditional (if/while) on identity field: {stripped[:80]}"))
            if re.search(r'switch\s*\([^)]*' + field, stripped):
                issues.append((i, field, f"Switch on identity field: {stripped[:80]}"))
            if re.search(field + r'\s*\?', stripped):
                issues.append((i, field, f"Ternary operator with identity field: {stripped[:80]}"))

    return issues


def scan_source_tree(src_dir):
    """Scan all source files for prohibited identity field usage."""
    all_issues = {}

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]

        for f in files:
            if f.endswith(('.cpp', '.h')) and not f.endswith('_Test.cpp') and not f.endswith('Tests.cpp'):
                filepath = os.path.join(root, f)
                issues = scan_field_usage(filepath)
                if issues:
                    relpath = os.path.relpath(filepath, REPO_ROOT)
                    all_issues[relpath] = issues

    return all_issues


def main():
    parser = argparse.ArgumentParser(description="Identity Authority Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    all_issues = scan_source_tree(args.src)

    if all_issues:
        print(f"[FAIL] Found {sum(len(v) for v in all_issues.values())} prohibited identity field usage(s) "
              f"in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, field, msg in issues:
                print(f"    L{line} ({field}): {msg}")
        return 1

    print("[PASS] No prohibited identity field usage detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
