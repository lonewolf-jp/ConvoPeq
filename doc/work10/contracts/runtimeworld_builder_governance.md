# RuntimeWorld Builder Governance

## Scope

- RuntimeWorld build path
- BuilderToken ownership and issuance
- Publication prebuild authority boundaries

## Governance Rules

1. RuntimeWorld construction must be initiated only via BuilderToken-governed path.
2. Builder API additions or authority injection changes require RFC approval.
3. NonRT is the only authoritative writer for BuilderToken-derived semantic fields.
4. Build path must preserve publication single-path contract (`publish(RuntimeWorld*)`).

## Prohibited Changes Without RFC

- New builder entrypoints bypassing BuilderToken
- Direct mutation entrypoints from AudioThread
- Alternate publication path insertion

## Verification

- `isr-verify-runtimeworld-builder-governance.ps1`
