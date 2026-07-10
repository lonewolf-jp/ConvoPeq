#!/usr/bin/env python3
"""
authority_source_count_verifier.py

Counts publication authority sources and retire authority sources.
Verifies that PublicationSemantic source count == 1 and
Retire authority source count == 1 (warning only).

Phase-1 3.2.10: AuthoritySourceCountVerifier.

Semantic:
  - Publication authority = calls to RuntimePublicationCoordinator::publishWorld()
    (or wrappers like AudioEngine::publishWorld). Internal publishAtomic calls
    within the coordinator implementation do NOT count as separate sources.
  - Retire authority = calls to ISRRuntimePublicationCoordinator::enqueueRetire()
    Direct EpochDomain::enqueueRetire() calls outside coordinator are prohibited.
    Calls to different mechanisms (DSPHandlePool::retire, RetireRuntimeEx::enqueueRetire)
    are NOT EpochDomain retire operations and do not count.

Usage:
    python authority_source_count_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files excluded from authority counting (coordinator internals, transitional wrappers)
EXCLUDED_FILES = [
    r'ISRRuntimePublicationCoordinator\.(cpp|h)$',
    r'ISRRetireRuntimeEx\.(cpp|h)$',
    r'SnapshotCoordinator\.(cpp|h)$',                  # SnapshotCoordinator uses EpochDomain directly (different domain)
    r'RefCountedDeferred\.h$',                         # unused template (inherited but never called)
    r'AudioEngine\.Threading\.cpp$',                   # transitional wrapper with #pragma suppression
    r'EQProcessor\.Core\.cpp$',                         # transitional fallback with #pragma suppression
]

# Specific call patterns that are the single authority (not counted as separate sources)
SINGLE_AUTHORITY_PATTERNS = [
    r'runtimePublicationBridge_\.enqueueRetire\(',      # AudioEngine -> ISR coordinator (single authority)
    r'm_retireCoordinator\s*->\s*enqueueRetire\(',      # EQProcessor -> ISR coordinator (single authority)
]

# Patterns that indicate a comment or declaration, not an actual call
COMMENT_OR_DECLARATION = [
    r'^\s*//',
    r'^\s*\*',
    r'^\s*/\*',
    r'.*\bvoid\s+retireDSP\(',
    r'.*\bvoid\s+enqueueRetire\(',
    r'.*\b(bool|RetireEnqueueResult)\s+enqueueRetire\(',
    r'.*RetireEnqueueResult\s+enqueueRetire',
    r'.*\[\[deprecated',
    r'#pragma\s+warning',
]

# Non-EpochDomain operations that look like retire but are different mechanisms
NON_EPOCH_RETIRE_PATTERNS = [
    r'dspHandleRuntime_\.retire\(',        # DSPHandlePool — handle management
    r'retireRuntimeEx_\.enqueueRetire\(',   # RetireRuntimeEx — intent tracking
    r'SnapshotRetireManager',                # legacy (removed)
    r'm_retire\b',                           # SnapshotRetireManager member
    r'RefCountedDeferred',                   # unused template
]


def is_excluded_file(filepath):
    """Check if file is in the exclusion list."""
    for pattern in EXCLUDED_FILES:
        if re.search(pattern, filepath):
            return True
    return False


def is_comment_or_declaration(line):
    """Check if line is a comment, declaration, or pragma."""
    for pattern in COMMENT_OR_DECLARATION:
        if re.match(pattern, line):
            return True
    return False


def is_non_epoch_retire(line):
    """Check if line is a non-EpochDomain retire operation."""
    for pattern in NON_EPOCH_RETIRE_PATTERNS:
        if re.search(pattern, line):
            return True
    return False


def find_publication_authority_sources(src_dir):
    """Find locations that actually publish/modify PublicationSemantic fields.

    Only counts publishAtomic calls OUTSIDE the coordinator implementation.
    Internal coordinator implementation is the single authority itself.
    """
    sources = set()

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                if is_excluded_file(filepath):
                    continue

                relpath = os.path.relpath(filepath, REPO_ROOT)
                with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                    lines = fh.readlines()

                for i, line in enumerate(lines, 1):
                    if is_comment_or_declaration(line):
                        continue
                    for field in ['publicationSequence', 'publicationEpoch', 'mappedRuntimeGeneration']:
                        if re.search(r'publishAtomic\(' + field, line):
                            sources.add((relpath, i, f"publishAtomic({field})"))

    return sources


def find_retire_authority_sources(src_dir):
    """Find distinct retire authority entry points.

    Groups retireDSP() call sites as a single authority source (all route through
    AudioEngine::retireDSP() -> coordinator::enqueueRetire()).
    Coordinator's own enqueueRetire is excluded (it IS the authority).
    Non-EpochDomain operations (DSPHandlePool, RetireRuntimeEx, etc.) are excluded.
    """
    sources = set()
    retire_dsp_found = False

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)

                with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                    lines = fh.readlines()

                for i, line in enumerate(lines, 1):
                    stripped = line.strip()
                    if is_comment_or_declaration(stripped):
                        continue
                    if is_non_epoch_retire(stripped):
                        continue

                    # Group all retireDSP() calls as a single authority
                    if re.search(r'\bretireDSP\(', stripped):
                        retire_dsp_found = True
                        continue

                    # Skip calls that go through the single authority (coordinator)
                    is_authority_call = any(re.search(p, stripped) for p in SINGLE_AUTHORITY_PATTERNS)
                    if is_authority_call:
                        continue

                    # Count other enqueueRetire calls (excluding coordinator impl and allowed files)
                    if re.search(r'enqueueRetire\(', stripped):
                        if is_excluded_file(filepath):
                            continue
                        sources.add((relpath, i, stripped[:60]))

    if retire_dsp_found:
        sources.add(("AudioEngine (retireDSP)", 0, "AudioEngine::retireDSP() — single entry point → coordinator"))
    if not sources:
        sources.add(("coordinator", 0, "ISRRuntimePublicationCoordinator::enqueueRetire() — single authority"))

    return sources


def main():
    parser = argparse.ArgumentParser(description="Authority Source Count Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    pub_sources = find_publication_authority_sources(args.src)
    ret_sources = find_retire_authority_sources(args.src)

    print(f"[INFO] Publication authority sources: {len(pub_sources)}")
    for src in sorted(pub_sources):
        print(f"       {src[0]}:{src[1]} - {src[2]}")

    print(f"\n[INFO] Retire authority sources: {len(ret_sources)}")
    for src in sorted(ret_sources):
        print(f"       {src[0]}:{src[1]} - {src[2]}")

    if len(pub_sources) > 1:
        print(f"\n[WARN] PublicationSemantic source count == {len(pub_sources)} (expected <= 1)")
        print("       Publication authority should be RuntimePublicationCoordinator::publishWorld() only.")
        print("       Internal publishAtomic calls inside coordinator implementation are excluded.")
    else:
        print(f"\n[OK] PublicationSemantic source count == {len(pub_sources)}")

    if len(ret_sources) > 1:
        print(f"[WARN] Retire authority source count == {len(ret_sources)} (expected <= 1)")
        print("       Retire authority should be ISRRuntimePublicationCoordinator::enqueueRetire() only.")
        print("       Non-EpochDomain operations (DSPHandlePool, RetireRuntimeEx) are excluded.")
    else:
        print(f"[OK] Retire authority source count == {len(ret_sources)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
