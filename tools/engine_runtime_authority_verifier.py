#!/usr/bin/env python3
"""
engine_runtime_authority_verifier.py

Verifies that EngineRuntime fields are NOT used in decision/branch conditions.
EngineRuntime is deprecated (see 3.2.7) and fields should be accessed via
RuntimeSemanticSchema instead.

Phase-1 3.2.10: EngineRuntimeAuthorityVerifier.

Usage:
    python engine_runtime_authority_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def scan_file_for_engine_runtime_decisions(filepath):
    """Scan for EngineRuntime field access used in decision/branch contexts."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments and declarations
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if re.match(r'.*struct EngineRuntime|EngineRuntime\s+\w+;', stripped):
            continue

        # Look for engine.XXX or world->engine.XXX used in conditions
        if re.search(r'(if|while|switch)\s*\([^)]*\bengine\.', stripped):
            issues.append((i, f"EngineRuntime field in condition: {stripped[:80]}"))

        # Look for .engine. in ternary
        if re.search(r'\.engine\.\w+\s*\?', stripped):
            issues.append((i, f"EngineRuntime field in ternary: {stripped[:80]}"))

    return issues


def scan_source_tree(src_dir):
    """Scan source tree for EngineRuntime decision usage."""
    all_issues = {}
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)
                issues = scan_file_for_engine_runtime_decisions(filepath)
                if issues:
                    all_issues[relpath] = issues
    return all_issues


def main():
    parser = argparse.ArgumentParser(description="EngineRuntime Authority Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    all_issues = scan_source_tree(args.src)

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} EngineRuntime decision usage(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
    else:
        print("[PASS] No EngineRuntime decision usage detected")

    return 0


if __name__ == "__main__":
    sys.exit(main())
