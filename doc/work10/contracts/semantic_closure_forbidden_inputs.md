# Semantic Closure Forbidden Inputs

## Forbidden for Semantic Authority Decision

- EngineRuntime direct state
- RuntimeGraph direct authority fields
- AudioEngine mutable globals/atomics outside allowlist
- DSPCore internal mutable state
- Debug flags / diagnostic mutable state
- Thread-local runtime decisions

## Enforcement

Forbidden inputs must not be read in semantic authority decision path.

## Verification

- `isr-verify-semantic-closure-forbidden-inputs.ps1`
