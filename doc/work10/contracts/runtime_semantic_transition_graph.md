# Runtime Semantic Transition Graph

## Nodes

- Publication states
- Overlap states
- Retire states
- Recovery states

## Edge Rules

- Edges must be explicitly guarded
- Conditional edges require documented predicates

## Invalid Patterns

- cyclic authority regressions
- implicit transitions without guard

## Verification

- `isr-verify-runtime-semantic-transition-graph.ps1`
