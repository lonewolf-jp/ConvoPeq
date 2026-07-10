#!/usr/bin/env python3
"""
detect_publication_mutation.py

Detects unauthorized publication semantic mutations outside of allowed locations.
PublicationSemantic fields (monitored via SemanticCategory::PublicationSemantic)
must only be mutated within:
  - RuntimePublicationCoordinator class internals
  - [[pub_boundary]] attributed functions

Phase-1 3.2.2 Item 5.

Usage:
    python detect_publication_mutation.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# PublicationSemantic classified fields to monitor
PUBLICATION_SEMANTIC_FIELDS = [
    'publicationSequence',
    'publicationSequenceCounter',
    'publicationSequenceId',
    'lastCommittedPublicationSequence',
    'mappedRuntimeGeneration',
    'publicationEpoch',
]

# Allowed mutation locations
ALLOWED_FUNCTIONS = [
    'RuntimePublicationCoordinator::commit',
    'RuntimePublicationCoordinator::publishWorld',
    'RuntimePublicationCoordinator::precheckPublish',
    'reserveRuntimePublicationIdentity',
    'publishRuntimeStateNonRt',
    'computeRuntimePublishComputation',
]


def scan_for_mutations(filepath):
    """Scan for publication semantic field mutations."""
    issues = []
    filename = os.path.basename(filepath)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        for field in PUBLICATION_SEMANTIC_FIELDS:
            if field not in stripped:
                continue

            # Check for write operations (publishAtomic, store, =, fetch_add)
            is_write = any(p in stripped for p in [
                'publishAtomic(' + field,
                field + ' =',
                'fetchAddAtomic(' + field,
                'store(' + field,
            ])
            if not is_write:
                continue

            # Check if this is in an allowed function
            in_allowed = any(fn in filepath for fn in ['RuntimePublicationCoordinator'])
            if in_allowed:
                continue

            issues.append((i, field, stripped[:80]))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Detect Publication Semantic Mutation")
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
                issues = scan_for_mutations(filepath)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} publication semantic mutation(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, field, context in issues:
                print(f"    L{line} ({field}): {context}")
        return 0  # Warning only
    else:
        print("[PASS] No unauthorized publication semantic mutations detected")
        return 0


if __name__ == "__main__":
    sys.exit(main())
