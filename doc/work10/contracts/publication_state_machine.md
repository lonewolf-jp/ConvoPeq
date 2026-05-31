# Publication State Machine Contract

## States

- Draft
- Publishing
- Published
- Retiring
- Retired
- Destroyed

## Allowed Transitions

- Draft -> Publishing
- Publishing -> Published
- Publishing -> Draft (rollback only)
- Published -> Retiring
- Retiring -> Retired
- Retired -> Destroyed

## Forbidden Transitions

- Published -> Draft
- Retired -> Published
- Destroyed -> any

## Guards

- Publishing -> Published requires semantic precheck pass
- Published -> Retiring requires retire intent emission and ownership release
- Retiring -> Retired requires eligibility gate pass

## Evidence

- Commit path logs
- Retire intent logs
- Verifier outputs (`isr-verify-publication-state-machine.ps1`)
