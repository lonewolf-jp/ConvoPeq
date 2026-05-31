# RuntimeWorld ABI Contract

## Scope

RuntimeWorld binary-facing layout constraints used by dump/audit/serialization tools.

## Fixed Elements

- Field order (no reorder without RFC)
- Field type (no type substitution without migration)
- Alignment assumptions
- Optionality semantics

## Change Policy

Any ABI-affecting change requires:

1. RFC with before/after layout
2. Migration plan
3. Backward compatibility assessment
4. Verifier update evidence

## Verification

- `isr-verify-runtimeworld-abi-contract.ps1`
- Runtimeworld serialization compatibility checks
