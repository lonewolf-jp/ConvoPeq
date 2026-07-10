#!/usr/bin/env python3
"""
authority_duplication_verifier.py

Phase-2: Detects when the same semantic information (e.g., eqBypassed)
exists in multiple authority structures. This causes ambiguity about
the canonical source of truth.

Usage:
    python authority_duplication_verifier.py [--src <path>]
"""

import os
import sys
import re
import argparse
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Known field duplications that are intentional (Projection mirrors)
# Format: field_name -> (canonical_structure, projection_structure)
KNOWN_DUPLICATIONS = {
    'eqBypassed': ('RuntimeState.routing', 'RuntimeGraph (historical)'),
    'convBypassed': ('RuntimeState.routing', 'RuntimeGraph (historical)'),
    'softClipEnabled': ('RuntimeState.automation', 'RuntimeGraph (historical)'),
    'saturationAmount': ('RuntimeState.automation', 'RuntimeGraph (historical)'),
    'inputHeadroomGain': ('RuntimeState.automation', 'RuntimeGraph (historical)'),
    'outputMakeupGain': ('RuntimeState.automation', 'RuntimeGraph (historical)'),
    'convolverInputTrimGain': ('RuntimeState.automation', 'RuntimeGraph (historical)'),
    # dspProjection appears in both kRuntimeAuthorityInventory and
    # kRuntimeReadAuthorityInventory (subset) — intentional per ISR design.
    'dspProjection': ('RuntimeState.authority_inventory', 'RuntimeState.read_inventory'),
}

# Structural/semantic group names that appear in both descriptor and inventory (false positives)
STRUCTURAL_NAMES = {
    'automation', 'coefficient', 'engine', 'execution', 'generationSemantic',
    'latency', 'metadata', 'overlap', 'publication', 'resource',
    'retire', 'routing', 'timing', 'topology', 'graph',
    'affinity', 'projectionFreshness', 'semanticHash',
}


def find_inventory_entries(src_dir):
    """Find all field names across all inventory arrays in source files."""
    field_locations = defaultdict(list)

    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in ('JUCE', 'r8brain-free-src')]
        for f in files:
            if f.endswith(('.h', '.cpp')):
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8', errors='replace') as fh:
                    content = fh.read()
                pattern = re.compile(r'\{"(\w+)",\s*convo::isr::RuntimeAuthorityClass')
                for match in pattern.finditer(content):
                    field_locations[match.group(1)].append(os.path.relpath(filepath, REPO_ROOT))

    return field_locations


def main():
    parser = argparse.ArgumentParser(description="Authority Duplication Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"), help="Source directory")
    args = parser.parse_args()

    field_locations = find_inventory_entries(args.src)

    issues = []
    for field, locations in sorted(field_locations.items()):
        # Skip fields that appear in only one place
        if len(locations) <= 1:
            continue

        # Skip structural/group names (kFieldDescriptors struct names)
        if field in STRUCTURAL_NAMES:
            continue

        # Check if it's a known duplication
        if field in KNOWN_DUPLICATIONS:
            continue

        issues.append((field, locations))

    if issues:
        print(f"[FAIL] Found {len(issues)} field(s) duplicated across multiple structures:")
        for field, locs in issues:
            print(f"\n  {field} appears in:")
            for loc in locs:
                print(f"    - {loc}")
        return 1

    print("[PASS] No unexpected authority duplications detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
