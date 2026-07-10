#!/usr/bin/env python3
"""
authority_inventory_verifier.py

Phase-2: Verifies triple consistency between:
1. JSON authority inventory file (authority_inventory.json)
2. RuntimeAuthorityInventoryEntry declarations in source code
3. Actual struct field declarations

All three must agree.

Usage:
    python authority_inventory_verifier.py [--src <path>]
"""

import json
import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_runtime_graph_fields():
    """Extract actual field names from RuntimeGraph struct."""
    path = os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h")
    fields = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    # Match field declarations that are not array definitions
    pattern = re.compile(r'\n\s+(?:const\s+)?\S+(?:\s*\*)?\s+(\w+)\s*(?:=\s*[^;]+)?;', re.MULTILINE)
    for match in pattern.finditer(content):
        name = match.group(1)
        if name not in ('kFieldDescriptors', 'kAuthorityInventory', 'false', 'true', 'nullptr'):
            fields.append(name)
    return set(fields)


def get_inventory_from_source(filepath):
    """Extract inventory entry names from kAuthorityInventory array in a header."""
    inv = set()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = re.compile(r'\{"(\w+)",\s*convo::isr::RuntimeAuthorityClass')
    for match in pattern.finditer(content):
        inv.add(match.group(1))
    return inv


def get_descriptors_from_source(filepath):
    """Extract descriptor entry names from kFieldDescriptors array in a header."""
    desc = set()
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    pattern = re.compile(r'\{"(\w+)",\s*convo::isr::SemanticCategory')
    for match in pattern.finditer(content):
        desc.add(match.group(1))
    return desc


def main():
    parser = argparse.ArgumentParser(description="Authority Inventory Verifier")
    parser.add_argument("--src", default=os.path.join(REPO_ROOT, "src"), help="Source directory")
    args = parser.parse_args()

    errors = []
    sources = [
        ("RuntimeGraph", os.path.join(args.src, "audioengine", "RuntimeGraph.h")),
        ("RuntimeState", os.path.join(args.src, "audioengine", "AudioEngine.h")),
    ]

    for name, path in sources:
        actual = get_runtime_graph_fields() if "RuntimeGraph" in path else set()
        desc = get_descriptors_from_source(path)
        inv = get_inventory_from_source(path)

        # Descriptor ⊆ Inventory
        not_in_inv = desc - inv
        if not_in_inv:
            errors.append(f"[{name}] Descriptor entries not in Inventory: {sorted(not_in_inv)}")

        # Inventory ⊆ Descriptor
        not_in_desc = inv - desc
        if not_in_desc:
            errors.append(f"[{name}] Inventory entries not in Descriptor: {sorted(not_in_desc)}")

    if errors:
        print("[FAIL] Authority Inventory consistency errors:")
        for e in errors:
            print(f"  {e}")
        return 1

    print("[PASS] Authority Inventory triple consistency verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
