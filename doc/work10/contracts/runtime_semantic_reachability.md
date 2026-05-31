# Runtime Semantic Reachability

## Target Reachable States

- CrossfadeComplete
- RetireSettled
- PublicationStable

## Requirements

- each target state must be reachable via at least one valid path
- dead-end transitions must be documented and rejected

## Verification

- `isr-verify-semantic-reachability.ps1`
