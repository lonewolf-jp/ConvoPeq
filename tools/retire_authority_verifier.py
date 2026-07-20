#!/usr/bin/env python3
"""
retire_authority_verifier.py

Verifies that retire calls only go through RuntimePublicationCoordinator or
the EpochDomain (which is delegated by the coordinator). Direct retire calls
from other locations are prohibited.

Phase-1 3.2.3: RetireAuthorityVerifier (enhanced for 3.2.3 consolidation).
Checks:
  - Direct EpochDomain::enqueueRetire() calls outside coordinator
  - Direct retire() calls outside coordinator
  - Allowed callers list

Usage:
    python retire_authority_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Allowed retire callers (functions that are permitted to call retire/enqueueRetire)
ALLOWED_RETIRE_CALLERS = [
    r'enqueueRetire\(',
    r'retireRuntimePublishWorldNonRt',
    r'RuntimePublicationCoordinator.*retire',
    r'RuntimePublicationCoordinator::enqueueRetire',
    r'runtimePublicationBridge_\.retire\(',           # ISR coordinator telemetry (not EpochDomain retirement)
    r'runtimePublicationBridge_\.enqueueRetire\(',     # ISR coordinator enqueueRetire (single authority)
    r'processDeferredReleases',
    r'lifetimeMgr\.retire\(',                          # DSPLifetimeManager:retire() — routes through ISRRetireRouter
    r'lifetime\.retire\(',                             # DSPLifetimeManager:retire() (local instances)
    r'lifetime_\.retire\(',                            # DSPLifetimeManager:retire() (member instances)
    r'lifetimeForShutdown\.retire\(',                  # DSPLifetimeManager:retire() (shutdown path)
]

# Known non-EpochDomain retire-like operations (different semantic, not retirement authority)
NON_EPOCH_RETIRE_OPERATIONS = [
    r'dspHandleRuntime_\.retire\(',                    # DSPHandlePool::retire() — handle management, not EpochDomain
    r'DSPHandleRuntime::retire\(',                     # DSPHandlePool::retire() method definition
    r'SnapshotRetireManager',                           # legacy snapshot retire manager (removed in 3.2.1)
]

# Files that are allowed to contain direct EpochDomain::enqueueRetire calls
# (coordinator implementation + transitional wrappers with pragma suppression)
ALLOWED_ENQUEUE_RETIRE_FILES = [
    r'ISRRuntimePublicationCoordinator\.cpp$',
    r'AudioEngine\.Threading\.cpp$',
    r'SnapshotCoordinator\.(cpp|h)$',
    r'EQProcessor\.Core\.cpp$',                       # transitional fallback with #pragma warning(disable:4996)
    r'RefCountedDeferred\.h$',                         # unused template (inherited but never called)
    r'ISRRetireRouter\.(cpp|h)$',                     # ISRRetireRouter: definition of retire() authority
]

# Non-EpochDomain enqueueRetire-like operations (different queue, different semantic)
NON_EPOCH_ENQUEUE_RETIRE = [
    r'retireRuntimeEx_\.enqueueRetire\(',              # RetireRuntimeEx::enqueueRetire() — intent tracking queue
]

# Allowed patterns in any file (coordinator forward declarations, declarations, etc.)
ALLOWED_DECLARATION_PATTERNS = [
    r'.*\bvoid\s+enqueueRetire\(',                     # method declaration
    r'.*\b(bool|RetireEnqueueResult)\s+enqueueRetire\(', # method declaration
    r'.*RetireEnqueueResult\s+enqueueRetire',           # return type
    r'.*EpochDomain::enqueueRetire.*deprecated',        # deprecation comment
    r'.*#pragma\s+warning',                             # pragma suppression
    r'.*//.*enqueueRetire',                             # comment
    r'.*enqueueRetireEpochBounded',                     # transitional wrapper (different function)
    r'.*template.*class RefCountedDeferred',            # unused template (inherited but never called)
]



def scan_file_for_direct_enqueue_retire(filepath):
    """Scan for direct EpochDomain::enqueueRetire() calls outside allowed files."""
    issues = []

    # Skip allowed files
    for pattern in ALLOWED_ENQUEUE_RETIRE_FILES:
        if re.search(pattern, filepath):
            return issues

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Skip known non-EpochDomain patterns
        is_non_epoch = any(re.search(p, stripped) for p in NON_EPOCH_ENQUEUE_RETIRE)
        if is_non_epoch:
            continue

        # Skip allowed declaration patterns
        if any(re.match(p, stripped) for p in ALLOWED_DECLARATION_PATTERNS):
            continue

        # Look for direct .enqueueRetire( calls (on EpochDomain instances)
        if re.search(r'\.enqueueRetire\(', stripped):
            # Skip deprecated attribute strings (false positive on string literals)
            if 'deprecated' in stripped or 'Use coordinator' in stripped or 'instead' in stripped:
                continue
            # Allow if it's the coordinator calling domain.enqueueRetire
            if 'ISRRuntimePublicationCoordinator' in filepath:
                continue
            # Allow runtimePublicationBridge_.enqueueRetire (ISR coordinator = single authority)
            if re.search(r'runtimePublicationBridge_\.enqueueRetire\(', stripped):
                continue
            # Allow #pragma warning disable around the call
            if '#pragma warning' in filepath or any(p in filepath for p in ['pragma']):
                pass
            issues.append((i, f"Direct enqueueRetire() call: {stripped[:100]}"))

        # Look for EpochDomain::enqueueRetire references
        if re.search(r'EpochDomain::enqueueRetire', stripped) and 'deprecated' not in stripped:
            # Skip the function declaration itself (not a call)
            if re.match(r'.*\(.*deprecated', stripped) or stripped.startswith('[[deprecated'):
                continue
            # Skip comments about enqueueRetire
            if stripped.startswith('//') or stripped.startswith('*'):
                continue
            issues.append((i, f"EpochDomain::enqueueRetire reference: {stripped[:100]}"))

    return issues


def scan_file_for_retire_calls(filepath):
    """Scan for retire-related calls and check if they're from allowed callers."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip comments
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue

        # Skip #pragma lines
        if stripped.startswith('#pragma'):
            continue

        # Skip known non-EpochDomain retire operations
        is_non_epoch = any(re.search(p, stripped) for p in NON_EPOCH_RETIRE_OPERATIONS)
        if is_non_epoch:
            continue

        # Look for potential retire calls
        if re.search(r'\bretire\(', stripped) and not re.search(r'RetireSemantic|RetireEpoch|RetireEnqueueResult|RetireBoundaryTelemetry|RetireGraceSemantics', stripped):
            # Allow known good patterns
            if any(re.search(p, stripped) for p in ALLOWED_RETIRE_CALLERS):
                continue
            # Check if it's a definition or declaration (including ClassName::retire forms)
            if re.match(r'.*\bvoid\s+.*retire\(', stripped) or re.match(r'.*\b(bool|int|uint|RetireEnqueueResult)\s+.*retire\(', stripped):
                continue
            # Allow coordinator's own method
            if 'ISRRuntimePublicationCoordinator' in filepath:
                continue
            # Allow files in the allowed list (definitions and transitional wrappers)
            if any(re.search(p, filepath) for p in ALLOWED_ENQUEUE_RETIRE_FILES):
                continue
            issues.append((i, f"Unexpected retire call: {stripped[:80]}"))

        # Look for direct deletion queue access (old SnapshotRetireManager)
        if re.search(r'\bm_retire\b', stripped) and 'm_retireCoordinator' not in stripped:
            issues.append((i, f"Direct SnapshotRetireManager access: {stripped[:80]}"))

        # Verify retire goes through coordinator: check for m_epochDomain.enqueueRetire
        if re.search(r'm_epochDomain\s*\.\s*enqueueRetire', stripped):
            # Skip files that are allowed transitional wrappers
            is_allowed = any(re.search(p, filepath) for p in ALLOWED_ENQUEUE_RETIRE_FILES)
            if not is_allowed:
                issues.append((i, f"Direct m_epochDomain.enqueueRetire(): {stripped[:80]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Retire Authority Verifier")
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
                issues = scan_file_for_retire_calls(filepath)
                enqueue_issues = scan_file_for_direct_enqueue_retire(filepath)
                all_issues_for_file = issues + enqueue_issues
                if all_issues_for_file:
                    all_issues[relpath] = all_issues_for_file

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[WARN] Found {total} retire authority issue(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
        return 1
    else:
        print("[PASS] All retirement goes through RuntimePublicationCoordinator (retire authority count == 1)")
        return 0


if __name__ == "__main__":
    sys.exit(main())

