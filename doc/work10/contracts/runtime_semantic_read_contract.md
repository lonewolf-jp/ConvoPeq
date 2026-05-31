# Runtime Semantic Read Contract

## Scope

- Semantic reads for publication precheck / observe / execution

## Contract

1. Semantic reads must use RuntimeWorld fields and semantic APIs only.
2. Direct reads from EngineRuntime mutable internals are forbidden in semantic decision paths.
3. Direct RuntimeGraph authority reads are forbidden for semantic authority decisions.
4. Thread-local or ad-hoc mutable globals are forbidden inputs.

## Allowed Sources

- `RuntimePublishWorld` semantic fields
- Publication metadata under RuntimeWorld
- Approved diagnostic counters (non-authoritative)

## Verification

- `isr-verify-runtime-semantic-read-contract.ps1`
