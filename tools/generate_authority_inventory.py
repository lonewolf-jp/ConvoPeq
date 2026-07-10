#!/usr/bin/env python3
"""
generate_authority_inventory.py

Generates authority_inventory.json from RuntimeFieldDescriptor declarations
in source code. Reads kFieldDescriptors arrays and extracts field metadata.

Phase-0: Generator/validator for authority inventory.

Usage:
    python generate_authority_inventory.py [--output <path>]
"""

import json
import os
import sys
import re
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Header files containing kFieldDescriptors arrays
SOURCE_FILES = [
    "src/audioengine/RuntimeGraph.h",
    "src/audioengine/AudioEngine.h",
    "src/audioengine/ISRRuntimeSemanticSchema.h",
]


def extract_descriptors(filepath):
    """Extract field descriptor entries from a kFieldDescriptors array."""
    descriptors = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Match kFieldDescriptors array entries
    pattern = re.compile(
        r'\{\s*"(\w+)"\s*,\s*'  # fieldName
        r'convo::isr::SemanticCategory::(\w+)\s*,\s*'  # semanticCategory
        r'convo::isr::OwnershipClass::(\w+)\s*,\s*'  # ownership
        r'convo::isr::MutabilityClass::(\w+)\s*,\s*'  # mutability
        r'convo::isr::VisibilityClass::(\w+)\s*,\s*'  # visibility
        r'convo::isr::LifetimeClass::(\w+)\s*'  # lifetime
        r'\}',
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        descriptors.append({
            "fieldName": match.group(1),
            "semanticCategory": match.group(2),
            "ownership": match.group(3),
            "mutability": match.group(4),
            "visibility": match.group(5),
            "lifetime": match.group(6),
        })

    return descriptors


def extract_inventory(filepath):
    """Extract authority inventory entries from a kAuthorityInventory array."""
    inventory = []

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    pattern = re.compile(
        r'\{\s*"(\w+)"\s*,\s*'
        r'convo::isr::RuntimeAuthorityClass::(\w+)\s*'
        r'\}',
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        inventory.append({
            "fieldName": match.group(1),
            "authorityClass": match.group(2),
        })

    return inventory


def verify_descriptor_inventory_consistency(descriptors, inventory):
    """Verify that descriptors and inventory are consistent."""
    errors = []
    desc_names = {d["fieldName"] for d in descriptors}
    inv_names = {i["fieldName"] for i in inventory}

    not_in_inv = desc_names - inv_names
    if not_in_inv:
        errors.append(f"Descriptors not in Inventory: {sorted(not_in_inv)}")

    not_in_desc = inv_names - desc_names
    if not_in_desc:
        errors.append(f"Inventory not in Descriptors: {sorted(not_in_desc)}")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate Authority Inventory")
    parser.add_argument("--output", default="config/authority_inventory.json",
                        help="Output path for authority_inventory.json")
    parser.add_argument("--verify", action="store_true",
                        help="Verify consistency without writing output")
    args = parser.parse_args()

    all_descriptors = []
    all_inventory = []
    source_map = {}

    for relpath in SOURCE_FILES:
        filepath = os.path.join(REPO_ROOT, relpath)
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            continue

        descs = extract_descriptors(filepath)
        inv = extract_inventory(filepath)

        if descs:
            all_descriptors.extend(descs)
            source_map.setdefault("descriptors", {})[relpath] = len(descs)
        if inv:
            all_inventory.extend(inv)
            source_map.setdefault("inventory", {})[relpath] = len(inv)

    # Verify consistency
    errors = verify_descriptor_inventory_consistency(all_descriptors, all_inventory)

    if args.verify:
        if errors:
            print("[FAIL] Authority inventory consistency errors:")
            for e in errors:
                print(f"  {e}")
            return 1
        print(f"[PASS] Descriptor/Inventory consistent: {len(all_descriptors)} descriptors, {len(all_inventory)} inventory entries")
        return 0

    # Build output
    output = {
        "schema_version": 1,
        "generated_from": SOURCE_FILES,
        "source_counts": source_map,
        "descriptors": all_descriptors,
        "inventory": all_inventory,
        "consistency_errors": errors,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"[OK] Generated: {args.output} ({len(all_descriptors)} fields)")

    if errors:
        print(f"[WARN] {len(errors)} consistency error(s) detected")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
