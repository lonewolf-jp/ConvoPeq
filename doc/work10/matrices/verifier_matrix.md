# Verifier Matrix

- Entry
  - Verifier Script:
  - Purpose:
  - Enforced Contract/DoD:
  - Trigger Tier:
  - Expected Result:
  - Evidence: execution log path

- Entry
  - Verifier Script: `.github/scripts/isr-run-tiered-verification.ps1`
  - Purpose: Tiered verification orchestrator
  - Enforced Contract/DoD: fail-closed governance baseline
  - Trigger Tier: pre-merge/full
  - Expected Result: all configured scripts pass
  - Evidence: script list (`isr-run-tiered-verification.ps1` lines 186+)

- Entry
  - Verifier Script: `.github/scripts/isr-verify-shadow-compare-contract.ps1`
  - Purpose: shadow compare contract integrity
  - Enforced Contract/DoD: shadow compare correctness
  - Trigger Tier: governance/semantic
  - Expected Result: PASS
  - Evidence: previously recorded validation report + script presence

- Entry
  - Verifier Script: `.github/scripts/isr-verify-shadow-compare-cadence.ps1`
  - Purpose: shadow compare cadence/evidence enforcement
  - Enforced Contract/DoD: operational cadence
  - Trigger Tier: governance/operational
  - Expected Result: PASS
  - Evidence: previously recorded validation report + script presence

- Entry
  - Verifier Script: `.github/scripts/isr-verify-verifier-selftest.ps1`
  - Purpose: verifier self-test framework
  - Enforced Contract/DoD: verifier integrity governance (DoD55)
  - Trigger Tier: CI/nightly
  - Expected Result: PASS when self-test detects intentional violation sample behavior correctly
  - Evidence: script added + execution PASS (`evidence/verifier_selftest_report.json`)
