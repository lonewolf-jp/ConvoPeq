# Runtime Recovery Semantic

## Classes

- Recoverable: can continue with bounded degradation
- Retryable: transient failure, retry policy applies
- Fatal: requires transition to safe stop / rollback path

## Domain Coverage

- Publication failure
- Retire failure
- Crossfade failure
- Shadow compare failure

## Policy

- Recovery decision must be explicit and logged
- Retry has bounded attempts and cooldown
- Fatal transitions must preserve ownership integrity

## Verification

- `isr-verify-runtime-recovery-semantic.ps1`
