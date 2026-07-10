#!/usr/bin/env python3
"""
generate_publication_manifest.py

Reads pub_boundary_registry.json and generates publication_manifest.json.
Used by CI to verify publication boundary integrity.

Usage:
    python generate_publication_manifest.py [--registry <path>] [--output <path>]

Phase-0 2.3: Meta information and boundary function information separation.
"""

import json
import os
import sys
import argparse


def resolve_path(path: str, repo_root: str) -> str:
    """Resolve relative paths against repo_root."""
    if os.path.isabs(path):
        return path
    return os.path.join(repo_root, path)


def load_registry(registry_path: str) -> dict:
    """Load the pub boundary registry JSON file."""
    with open(registry_path, 'r') as f:
        return json.load(f)


def generate_manifest(registry: dict) -> dict:
    """Generate publication manifest from registry data."""
    manifest = {
        "manifest_version": 1,
        "generated_from": "pub_boundary_registry.json",
        "description": "Publication boundary manifest. Lists all authorized publication boundary functions.",
        "boundary_functions": []
    }

    for entry in registry.get("boundary_functions", []):
        manifest["boundary_functions"].append({
            "name": entry["name"],
            "file": entry["file"],
            "category": entry.get("category", "unknown")
        })

    return manifest


def verify_function_exists(manifest: dict, repo_root: str) -> list:
    """Verify that each function in the manifest exists in the source code.
    Returns a list of verification errors (empty list = all passed)."""
    errors = []

    for entry in manifest["boundary_functions"]:
        file_path = os.path.join(repo_root, entry["file"])
        func_name = entry["name"]

        # Extract the short function name from fully qualified name
        short_name = func_name.split("::")[-1].split("<")[0]

        if not os.path.exists(file_path):
            errors.append(f"File not found: {entry['file']} (function: {func_name})")
            continue

        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        # Check for function signature presence
        if short_name not in content:
            errors.append(f"Function '{short_name}' not found in {entry['file']} (full: {func_name})")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Generate publication manifest from boundary registry")
    parser.add_argument("--registry", default="config/pub_boundary_registry.json",
                        help="Path to pub_boundary_registry.json")
    parser.add_argument("--output", default="config/publication_manifest.json",
                        help="Output path for publication_manifest.json")
    parser.add_argument("--repo-root", default=".",
                        help="Repository root path (for verification)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify registered functions exist in source code")
    args = parser.parse_args()

    # [Practical Stable] Resolve relative paths against repo_root
    registry_path = resolve_path(args.registry, args.repo_root)
    output_path = resolve_path(args.output, args.repo_root)

    registry = load_registry(registry_path)
    manifest = generate_manifest(registry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')

    print(f"[generate_publication_manifest] Generated: {output_path}")

    if args.verify:
        errors = verify_function_exists(manifest, args.repo_root)
        if errors:
            print("[generate_publication_manifest] VERIFICATION FAILED:")
            for err in errors:
                print(f"  FAIL: {err}")
            sys.exit(1)
        else:
            print("[generate_publication_manifest] Verification passed: all functions found.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
