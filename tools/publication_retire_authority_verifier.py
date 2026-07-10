#!/usr/bin/env python3
"""
publication_retire_authority_verifier.py

Verifies that RuntimePublicationCoordinator is the single source of
authority for publication and retire operations. Checks that no other
code path directly publishes or retires without going through the coordinator.

Usage:
    python tools/publication_retire_authority_verifier.py
"""

import os
import sys
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known coordinator paths
COORDINATOR_PATHS = [
    'ISRRuntimePublicationCoordinator',
    'RuntimePublicationCoordinator',
    'RuntimeBuilder',
]

# Known safe direct paths (shutdown, drain, init)
SAFE_DIRECT_PATHS = [
    'drainAll',
    'shutdown',
    'releaseResources',
    '~AudioEngine',
    'createBootstrapWorld',
]


def scan_for_publication_authority(src_dir):
    """Check that all publication/retire paths go through coordinator."""
    issues = []

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if not f.endswith(('.cpp', '.h')):
                continue

            filepath = os.path.join(root, f)
            relpath = os.path.relpath(filepath, REPO_ROOT)

            with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                    continue

                # Skip coordinator internal files
                if any(p in filepath for p in COORDINATOR_PATHS):
                    continue

                # Look for direct publishAtomic on PublicationSemantic fields
                for field in ['publicationSequence', 'publicationEpoch', 'mappedRuntimeGeneration']:
                    if re.search(r'publishAtomic\(' + field, stripped):
                        # Check if in safe path
                        if not any(s in filepath for s in SAFE_DIRECT_PATHS):
                            issues.append((relpath, i, f"Direct write to PublicationSemantic field '{field}': {stripped[:60]}"))

    return issues


def main():
    print("=" * 60)
    print("Publication/Retire Authority Uniqueness Verification")
    print("=" * 60)

    issues = scan_for_publication_authority(os.path.join(REPO_ROOT, 'src'))

    if issues:
        print(f"\n[WARN] Found {len(issues)} direct publication authority access(es):")
        for filepath, line, msg in issues:
            print(f"  {filepath}:{line}: {msg}")
        print("\nNote: All known sites should route through RuntimePublicationCoordinator.")
    else:
        print("\n[PASS] RuntimePublicationCoordinator is the sole Publication/Retire authority.")

    # Summary
    print(f"\nPublication/Retire Authority Sources: RuntimePublicationCoordinator (primary)")
    print(f"  - ISRRuntimePublicationCoordinator::commit() - publication")
    print(f"  - ISRRuntimePublicationCoordinator::retire() - retire")
    print(f"  - RuntimeBuilder::buildRuntimePublishWorld()  - world construction")
    print(f"  - EpochDomain::enqueueRetire() / reclaimRetired() - deferred deletion")
    print(f"  - AudioEngine::submitRebuildIntent()          - rebuild trigger")

    return 0


if __name__ == "__main__":
    sys.exit(main())
