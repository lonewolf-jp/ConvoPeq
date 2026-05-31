# Semantic Closure Allowed External Inputs

## Allowed Inputs (strict allowlist)

- RuntimeWorld semantic fields produced by publication build path
- Publication metadata explicitly defined in RuntimeSemanticSchema
- Approved diagnostic counters that do not affect authority decisions

## Usage Rule

Only allowlisted inputs may participate in semantic authority decisions.

## Verification

- `isr-verify-semantic-closure-allowlist.ps1`
