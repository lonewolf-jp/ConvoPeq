# Descriptor UUID Stability Contract

## Purpose

Descriptor UUIDs are immutable identifiers for runtime authority classification.

## Rules

- UUIDs must remain stable across compatible revisions.
- Any incompatible change requires a migration note and verifier update.

## Verification

- `isr-verify-descriptor-uuid-stability.ps1`
