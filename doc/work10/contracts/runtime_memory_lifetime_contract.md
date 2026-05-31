# Runtime Memory Lifetime Contract

## Purpose

Define the lifetime of runtime-owned memory and metadata.

## Rules

- Create -> Publish -> Observe -> Retire -> Destroy must be explicit.
- No premature reuse of retired memory is allowed.

## Verification

- `isr-verify-runtime-memory-lifetime.ps1`
