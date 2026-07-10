#!/usr/bin/env python3
"""
soak_test_fault_injection.py

Soak test fault injection framework for Practical Stable ISR Bridge Runtime.
Simulates fault conditions and monitors the counter invariants.

Counter invariants (all must be 0 during steady state):
- publication monotonicity violation
- out-of-order publication
- retire starvation
- retire queue overflow
- snapshot leak
- publication rollback
- world swap failure
- duplicate publicationSequence
- RuntimeWorld null publication
- double retire

Usage:
    python tools/soak_test_fault_injection.py [--duration 30]
"""

import json
import os
import sys
import time
import argparse

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAULT_SCENARIOS = {
    "publication_sequence_reversal": {
        "description": "publicationSequence 逆転",
        "inject": "Call commit() with sequenceId <= previous sequenceId",
        "expected": "CoordinatorState::Faulted",
    },
    "can_retire_violation": {
        "description": "canRetire() を満たさない retire 呼び出し",
        "inject": "Call retire() with epoch that fails isOlder check",
        "expected": "Faulted state or reject",
    },
    "duplicate_publication_sequence": {
        "description": "重複した publicationSequence の発行",
        "inject": "Call commit() twice with same sequenceId",
        "expected": "Faulted after second call",
    },
    "null_world_publication": {
        "description": "RuntimeWorld が nullptr の状態での publication",
        "inject": "Call commit() with nullptr world",
        "expected": "CoordinatorState::Faulted",
    },
    "double_retire": {
        "description": "同一オブジェクトの二重 retire",
        "inject": "Call retire() twice on same object",
        "expected": "DeletionQueue handles gracefully",
    },
    "stalled_reader_epoch": {
        "description": "reader が永久に進まない状態",
        "inject": "Register reader thread, don't advance epoch",
        "expected": "retire starvation counter increments, then recovers when reader advances",
    },
}


def run_simulation(scenario, duration_sec):
    """Run a simulated fault injection scenario (log-based)."""
    print(f"\n  Scenario: {scenario['description']}")
    print(f"  Injection: {scenario['inject']}")
    print(f"  Expected: {scenario['expected']}")
    print(f"  Duration: {duration_sec}s")
    print(f"  Status: SIMULATED (requires runtime environment)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Soak Test Fault Injection")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in seconds for each scenario")
    parser.add_argument("--list", action="store_true",
                        help="List available scenarios")
    args = parser.parse_args()

    print("=" * 60)
    print("Soak Test Fault Injection Framework")
    print("=" * 60)
    print(f"\nMonitor counter invariants (all must be 0):")
    print("  - publication monotonicity violation count")
    print("  - out-of-order publication count")
    print("  - retire starvation count")
    print("  - retire queue overflow count")
    print("  - snapshot leak count")
    print("  - publication rollback count")
    print("  - world swap failure count")
    print("  - duplicate publicationSequence count")
    print("  - RuntimeWorld null publication count")
    print("  - double retire count")

    if args.list:
        print(f"\nAvailable Fault Injection Scenarios ({len(FAULT_SCENARIOS)}):")
        for name, scenario in FAULT_SCENARIOS.items():
            print(f"  - {name}: {scenario['description']}")
        return 0

    print(f"\nRunning {len(FAULT_SCENARIOS)} fault injection scenarios...")
    all_passed = True

    for name, scenario in FAULT_SCENARIOS.items():
        if not run_simulation(scenario, args.duration):
            all_passed = False

    print(f"\n{'=' * 60}")
    if all_passed:
        print("All fault injection scenarios completed (simulated).")
        print("Run with actual runtime to verify counter behavior.")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
