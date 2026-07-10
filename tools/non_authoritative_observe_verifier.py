#!/usr/bin/env python3
"""
non_authoritative_observe_verifier.py

Verifies that Audio Thread observation is restricted to RuntimeWorld only.
Detects consumeAtomic calls on non-RuntimeWorld atomics that should be
migrated to world-based reads.

Phase-1 3.2.10: NonAuthoritativeObserveVerifier.

Usage:
    python non_authoritative_observe_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known world-proxied atomics that should no longer be directly consumed
KNOWN_NON_AUTHORITATIVE_OBSERVES = [
    r'consumeAtomic\(currentSampleRate',
    r'consumeAtomic\(manualOversamplingFactor',
    r'consumeAtomic\(inputHeadroomGain',
    r'consumeAtomic\(outputMakeupGain',
    r'consumeAtomic\(convolverInputTrimGain',
    r'consumeAtomic\(saturationAmount',
    r'consumeAtomic\(ditherBitDepth',
    r'consumeAtomic\(noiseShaperType',
    r'consumeAtomic\(softClipEnabled',
    r'consumeAtomic\(eqBypassed',
    r'consumeAtomic\(convBypassed',
    r'consumeAtomic\(adaptiveCoeffBankIndex',
    r'consumeAtomic\(adaptiveCoeffGeneration',
]


def scan_file_for_non_authoritative_observes(filepath, processing_patterns):
    """Scan for non-authoritative observe patterns."""
    issues = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    for pattern in KNOWN_NON_AUTHORITATIVE_OBSERVES:
        for match in re.finditer(pattern, content):
            # Get line number
            line_num = content[:match.start()].count('\n') + 1
            # Check if this is in a processing file (audio thread)
            is_processing = any(p in filepath for p in processing_patterns)
            if is_processing:
                issues.append((line_num, f"Direct atomic observe in audio thread: {match.group()[:60]}"))

    return issues


def main():
    parser = argparse.ArgumentParser(description="Non-Authoritative Observe Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"),
                        help="Source directory to scan")
    args = parser.parse_args()

    # Strict RT processing files (actual getNextAudioBlock path).
    # Excludes:
    #   - PrepareToPlay.cpp (runs with audio thread stopped)
    #   - Latency.cpp (utility reads for UI/analyzer, not real-time processing)
    #   - ConvolverProcessor.Runtime.cpp (mixed RT/non-RT functions;
    #     consumeAtomic reads there are for one-time smoother reset
    #     and parameter bounds checking, not per-block processing)
    processing_patterns = [
        'AudioEngine.Processing.AudioBlock',
        'AudioEngine.Processing.BlockDouble',
        'AudioEngine.Processing.DSPCoreDouble',
        'AudioEngine.Processing.DSPCoreFloat',
        'AudioEngine.Processing.DSPCoreIO',
    ]

    all_issues = {}
    for root, dirs, files in os.walk(args.src):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.cpp', '.h')):
                filepath = os.path.join(root, f)
                relpath = os.path.relpath(filepath, REPO_ROOT)
                issues = scan_file_for_non_authoritative_observes(filepath, processing_patterns)
                if issues:
                    all_issues[relpath] = issues

    if all_issues:
        total = sum(len(v) for v in all_issues.values())
        print(f"[FAIL] Found {total} non-authoritative observe(s) in {len(all_issues)} file(s):")
        for filepath, issues in sorted(all_issues.items()):
            print(f"\n  File: {filepath}")
            for line, msg in issues:
                print(f"    L{line}: {msg}")
        return 1

    print("[PASS] No non-authoritative observes detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
