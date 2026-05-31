# RuntimeWorld Builder Internal Mutation Governance

## Scope

- Internal setter/inject paths used during RuntimeWorld assembly
- Mutable staging objects before freeze/seal

## Governance Rules

1. Internal mutation is allowed only within builder staging scope before `freeze` and `sealRecursively`.
2. After freeze/seal, no builder internal mutation API may alter semantic authority fields.
3. Mutation order must follow publication state machine (`Draft -> Publishing -> Published`).
4. Any new internal mutation hook requires RFC with rollback strategy.

## Forbidden Patterns

- `const_cast` mutation of frozen world
- post-freeze semantic field rewrites
- direct runtimeStore mutation from builder internal helpers

## Verification

- `isr-verify-runtimeworld-builder-internal-mutation-governance.ps1`
