#!/usr/bin/env python3
"""
capture_session_id_verifier.py

Detects prohibited uses of captureSessionId:
- Comparison operators (<, >, <=, >=, ==, !=)
- Ordering (sort, min, max)
- Hash key (unordered_map, unordered_set key)
- Conditional expressions (if, while, switch, ?:)

Phase-1 3.1: CaptureSessionIdVerifier.

Usage:
    python capture_session_id_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_prohibited_usage(filepath):
    """Scan a single file for prohibited captureSessionId usage."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments and empty lines
        if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Skip definition/declaration lines
        if 'captureSessionId' in stripped and ('=' in stripped or 'std::atomic' in stripped):
            continue

        if 'captureSessionId' in stripped:
            # Check for comparison operators
            if re.search(r'captureSessionId\s*[=!]=', stripped):
                issues.append((i, f"Comparison operator (==/!=) on captureSessionId: {stripped[:80]}"))
            if re.search(r'captureSessionId\s*[<>]', stripped):
                issues.append((i, f"Relational operator (< / >) on captureSessionId: {stripped[:80]}"))
            if re.search(r'captureSessionId\s*[<>]=', stripped):
                issues.append((i, f"Relational operator (<= / >=) on captureSessionId: {stripped[:80]}"))

            # Check for conditional usage
            if re.search(r'(if|while)\s*\([^)]*captureSessionId', stripped):
                issues.append((i, f"Conditional (if/while) on captureSessionId: {stripped[:80]}"))

            # Check for switch on captureSessionId
            if re.search(r'switch\s*\([^)]*captureSessionId', stripped):
                issues.append((i, f"Switch on captureSessionId: {stripped[:80]}"))

            # Check for ternary operator
            if re.search(r'captureSessionId\s*\?', stripped) or re.search(r'\?\s*[^:]*captureSessionId', stripped):
                issues.append((i, f"Ternary operator with captureSessionId: {stripped[:80]}"))

            # Check for ordering usage
            if re.search(r'(sort|min|max)\s*\([^)]*captureSessionId', stripped):
                issues.append((i, f"Ordering function with captureSessionId: {stripped[:80]}"))

            # Check for hash key usage
            if re.search(r'(unordered_map|unordered_set|std::hash)\s*[<([][^>)]*captureSessionId', stripped):
                issues.append((i, f"Hash key usage of captureSessionId: {stripped[:80]}"))

    return issues


def scan_source_tree(src_dir):
    """Scan all source files for prohibited captureSessionId usage."""
    all_issues = {}

    for root, dirs, files in os.walk(src_dir):
        # Skip JUCE and r8brain directories
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]

        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                issues = scan_prohibited_usage(filepath)
                if issues:
                    relpath = os.path.relpath(filepath, REPO_ROOT)
                    all_issues[relpath] = issues

    return all_issues


def main():
    parser = argparse.ArgumentParser(description="CaptureSessionId Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    all_issues = scan_source_tree(args.src)

    if all_issues:
        print(f"[FAIL] Found prohibited captureSessionId usage in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
        return 1

    print("[PASS] No prohibited captureSessionId usage detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
