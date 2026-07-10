#!/usr/bin/env python3
"""
coverage_verifier.py

Verifies coverage relationships:
  Actual fields ⊆ Descriptor ⊆ Inventory
  AND Inventory == Matrix

Phase-0 2.12, Phase-1 3.1: CoverageVerifier.

Usage:
    python coverage_verifier.py [--matrix <path>]
"""

import json
import os
import sys
import argparse
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_fields_in_runtime_graph():
    """Find all actual field names in RuntimeGraph struct."""
    path = os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h")
    if not os.path.exists(path):
        return [], [f"File not found: {path}"]

    fields = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    field_pattern = re.compile(
        r'\n\s+(?:(?:std::|const\s+)?\w+(?:<[^>]*>)?(?:\s*\*)?)\s+(\w+)\s*(?:=\s*[^;]+)?;',
        re.MULTILINE
    )
    for match in field_pattern.finditer(content):
        name = match.group(1)
        if name not in ('kFieldDescriptors', 'kAuthorityInventory'):
            # Skip known non-field matches (values like false, true, nullptr, 0)
            if name in ('false', 'true', 'nullptr'):
                continue
            fields.append(name)

    return fields, None


def extract_descriptor_names_from_header(filepath):
    """Extract descriptor field names from kFieldDescriptors array."""
    if not os.path.exists(filepath):
        return set()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    names = set()
    pattern = re.compile(r'\{"(\w+)",\s*convo::isr::SemanticCategory')
    for match in pattern.finditer(content):
        names.add(match.group(1))

    return names


def extract_inventory_names_from_header(filepath):
    """Extract inventory field names from kAuthorityInventory array."""
    if not os.path.exists(filepath):
        return set()

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    names = set()
    pattern = re.compile(r'\{"(\w+)",\s*convo::isr::RuntimeAuthorityClass')
    for match in pattern.finditer(content):
        names.add(match.group(1))

    return names


def load_matrix(matrix_path):
    """Load migration matrix from JSON."""
    if not os.path.exists(matrix_path):
        return None, [f"Matrix file not found: {matrix_path}"]
    with open(matrix_path, 'r') as f:
        return json.load(f), None


def verify():
    """Run coverage verification."""
    errors = []

    # RuntimeGraph fields
    rg_fields, err = find_fields_in_runtime_graph()
    if err:
        errors.extend(err)
        rg_fields = []

    # Descriptors from RuntimeGraph.h
    rg_desc = extract_descriptor_names_from_header(
        os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h"))

    # Inventory from RuntimeGraph.h
    rg_inv = extract_inventory_names_from_header(
        os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h"))

    rg_field_set = set(rg_fields)

    # 1. Actual fields ⊆ Descriptor
    not_in_desc = rg_field_set - rg_desc
    if not_in_desc:
        errors.append(f"RuntimeGraph actual fields not in Descriptor: {sorted(not_in_desc)}")

    # 2. Descriptor ⊆ Inventory
    not_in_inv = rg_desc - rg_inv
    if not_in_inv:
        errors.append(f"RuntimeGraph Descriptor fields not in Inventory: {sorted(not_in_inv)}")

    # 3. Verify RuntimeState coverage too
    rs_path = os.path.join(REPO_ROOT, "src", "audioengine", "AudioEngine.h")
    rs_desc = extract_descriptor_names_from_header(rs_path)
    rs_inv = extract_inventory_names_from_header(rs_path)

    not_in_inv_rs = rs_desc - rs_inv
    if not_in_inv_rs:
        errors.append(f"RuntimeState Descriptor fields not in Inventory: {sorted(not_in_inv_rs)}")

    if errors:
        print("[FAIL] Coverage verification failed:")
        for e in errors:
            print(f"  {e}")
        return False

    print(f"[PASS] Coverage verification passed")
    print(f"  RuntimeGraph: {len(rg_fields)} actual fields, {len(rg_desc)} descriptors, {len(rg_inv)} inventory")
    print(f"  RuntimeState: {len(rs_desc)} descriptors, {len(rs_inv)} inventory")
    return True


def main():
    parser = argparse.ArgumentParser(description="Coverage Verifier")
    parser.add_argument("--matrix", help="Path to migration matrix JSON")

    args = parser.parse_args()

    ok = verify()

    if args.matrix:
        matrix, err = load_matrix(args.matrix)
        if err:
            for e in err:
                print(f"  {e}")
            return 1

        inv_path = os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h")
        inv = extract_inventory_names_from_header(inv_path)

        matrix_fields = set(matrix.get("fields", []))
        extra_in_matrix = matrix_fields - inv
        if extra_in_matrix:
            print(f"[FAIL] Fields in Matrix but not in Inventory: {sorted(extra_in_matrix)}")
            ok = False

        not_in_matrix = inv - matrix_fields
        if not_in_matrix:
            print(f"[FAIL] Fields in Inventory but not in Matrix: {sorted(not_in_matrix)}")
            ok = False

        if not extra_in_matrix and not not_in_matrix:
            print(f"[PASS] Matrix-Inventory complete match: {len(inv)} fields")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
