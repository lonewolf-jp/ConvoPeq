# RuntimeWorld Layout Governance

## Scope

- RuntimeWorld top-level field layout
- Field ordering/type/optionality compatibility

## Governance Rules

1. Top-level RuntimeWorld field additions require RFC.
2. Field order and type changes must comply with ABI and serialization contracts.
3. Semantic authority fields must not migrate to projection-only sections without contract update.
4. Layout changes must include migration plan and verifier evidence updates.

## Mandatory References

- `runtimeworld_abi_contract.md`
- `runtimeworld_serialization_contract.md`

## Verification

- `isr-verify-runtimeworld-layout-governance.ps1`
