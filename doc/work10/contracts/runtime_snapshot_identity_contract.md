# Runtime Snapshot Identity Contract

## Purpose

Guarantee snapshot identity is unique and never reused.

## Rules

- Snapshot identity must remain stable during its lifetime.
- Reuse after retire is forbidden.

## Verification

- `isr-verify-runtime-snapshot-identity.ps1`
- `isr-verify-runtime-snapshot-never-reuse.ps1`
