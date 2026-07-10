#!/usr/bin/env python3
"""
runtime_graph_authority_verifier.py

Verifies that RuntimeGraph has no Authoritative fields (Phase-1 strict mode).
Supports phased introduction: warning -> baseline -> strict.

Phase-0 2.2: RuntimeGraphAuthorityVerifier phased introduction.
Phase-1 3.1: Strict mode - RuntimeGraph Authoritative fields prohibited.

Usage:
    python runtime_graph_authority_verifier.py --mode [warning|baseline|strict]
"""

import json
import os
import sys
import argparse
import re

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INVENTORY_PATH = os.path.join(REPO_ROOT, "src", "audioengine", "ISRRuntimeSemanticSchema.h")
RUNTIME_GRAPH_HEADER = os.path.join(REPO_ROOT, "src", "audioengine", "RuntimeGraph.h")
BASELINE_PATH = os.path.join(REPO_ROOT, "config", "runtime_graph_baseline.json")
AUTHORITY_INVENTORY_PATH = os.path.join(REPO_ROOT, "config", "authority_inventory.json")


def find_authoritative_fields_in_runtime_graph():
    """Find all Authoritative fields declared in RuntimeGraph via kAuthorityInventory."""
    if not os.path.exists(RUNTIME_GRAPH_HEADER):
        return [], f"File not found: {RUNTIME_GRAPH_HEADER}"

    authoritative_fields = []
    with open(RUNTIME_GRAPH_HEADER, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Extract from kAuthorityInventory: entries with RuntimeAuthorityClass::Authoritative
    inventory_pattern = re.compile(
        r'\{"(\w+)",\s*convo::isr::RuntimeAuthorityClass::Authoritative\}',
        re.MULTILINE
    )
    for match in inventory_pattern.finditer(content):
        authoritative_fields.append(match.group(1))

    return authoritative_fields, None


def find_fields_in_runtime_graph():
    """Find all declared fields in RuntimeGraph struct."""
    if not os.path.exists(RUNTIME_GRAPH_HEADER):
        return [], f"File not found: {RUNTIME_GRAPH_HEADER}"

    fields = []
    with open(RUNTIME_GRAPH_HEADER, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    field_pattern = re.compile(
        r'\n\s+(?:const\s+)?\S+(?:\s*\*)?\s+(\w+)\s*(?:=\s*[^;]+)?;',
        re.MULTILINE
    )
    for match in field_pattern.finditer(content):
        field_name = match.group(1)
        if field_name not in ('kFieldDescriptors', 'kAuthorityInventory'):
            fields.append(field_name)

    return fields, None


def load_baseline():
    """Load the baseline inventory."""
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH, 'r') as f:
        return json.load(f)


def load_authority_inventory():
    """Load the authority inventory JSON."""
    if not os.path.exists(AUTHORITY_INVENTORY_PATH):
        return None
    with open(AUTHORITY_INVENTORY_PATH, 'r') as f:
        return json.load(f)


def check_authoritative_fields(fields, mode):
    """Check authoritative fields based on mode."""
    if mode == 'warning':
        if fields:
            print(f"[WARNING] RuntimeGraph contains {len(fields)} Authoritative fields:")
            for f in fields:
                print(f"  WARN: {f}")
            return False  # Don't fail in warning mode
        return True

    elif mode == 'baseline':
        baseline = load_baseline()
        expected = set(baseline.get('authoritative_fields', [])) if baseline else set()
        actual = set(fields)

        newly_added = actual - expected
        if newly_added:
            print(f"[FAIL] New Authoritative fields added to RuntimeGraph (baseline violation):")
            for f in sorted(newly_added):
                print(f"  FAIL: {f}")
            return False

        resolved = expected - actual
        if resolved:
            print(f"[INFO] Authoritative fields resolved from RuntimeGraph:")
            for f in sorted(resolved):
                print(f"  RESOLVED: {f}")

        return True

    elif mode == 'strict':
        if fields:
            print(f"[FAIL] RuntimeGraph still contains {len(fields)} Authoritative fields (strict mode):")
            for f in fields:
                print(f"  FAIL: {f}")
            return False
        return True

    return True


def save_baseline():
    """Save current authoritative fields as baseline."""
    fields, err = find_authoritative_fields_in_runtime_graph()
    if err:
        print(f"[ERROR] {err}")
        return False

    baseline = {
        "version": 1,
        "authoritative_fields": fields,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }

    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"[BASELINE] Saved {len(fields)} authoritative fields to {BASELINE_PATH}")
    return True


def check_descriptor_inventory_consistency():
    """Check: actual RuntimeGraph fields ⊆ Descriptor ⊆ Inventory."""
    actual_fields, err = find_fields_in_runtime_graph()
    if err:
        return False, [f"Field scan error: {err}"]

    # Extract descriptors from RuntimeGraph.h kFieldDescriptors
    with open(RUNTIME_GRAPH_HEADER, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    desc_fields = set()
    desc_pattern = re.compile(r'\{"(\w+)",\s*convo::isr::SemanticCategory')
    for match in desc_pattern.finditer(content):
        desc_fields.add(match.group(1))

    inv_fields = set()
    inv_pattern = re.compile(r'\{"(\w+)",\s*convo::isr::RuntimeAuthorityClass')
    for match in inv_pattern.finditer(content):
        inv_fields.add(match.group(1))

    errors = []
    actual_set = set(actual_fields)

    # Actual ⊆ Descriptor
    not_in_desc = actual_set - desc_fields
    if not_in_desc:
        errors.append(f"Fields in struct but not in Descriptor: {sorted(not_in_desc)}")

    # Descriptor ⊆ Inventory
    not_in_inv = desc_fields - inv_fields
    if not_in_inv:
        errors.append(f"Fields in Descriptor but not in Inventory: {sorted(not_in_inv)}")

    return len(errors) == 0, errors


def main():
    parser = argparse.ArgumentParser(description="RuntimeGraph Authority Verifier")
    parser.add_argument("--mode", choices=['warning', 'baseline', 'strict', 'check-consistency'],
                        default='warning', help="Verification mode")
    parser.add_argument("--save-baseline", action='store_true',
                        help="Save current state as baseline")

    args = parser.parse_args()

    if args.save_baseline:
        success = save_baseline()
        return 0 if success else 1

    if args.mode == 'check-consistency':
        ok, errors = check_descriptor_inventory_consistency()
        if ok:
            print("[PASS] Descriptor/Inventory consistency check passed")
            return 0
        else:
            print("[FAIL] Descriptor/Inventory consistency check failed:")
            for e in errors:
                print(f"  {e}")
            return 1

    fields, err = find_authoritative_fields_in_runtime_graph()
    if err:
        print(f"[ERROR] {err}")
        return 1

    print(f"[INFO] RuntimeGraph has {len(fields)} Authoritative fields")
    for f in fields:
        print(f"  FIELD: {f}")

    ok = check_authoritative_fields(fields, args.mode)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
