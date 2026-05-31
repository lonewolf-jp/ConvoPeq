# RuntimeWorld Serialization Contract

## Fixed Contract

- Field order
- Field type
- Field optionality
- Version migration rules

## Migration

- schemaVersion must increment on incompatible change
- migration notes required for each incompatible revision

## Compatibility Goal

- dump/audit readers must parse previous supported revisions

## Verification

- `isr-verify-runtimeworld-serialization-contract.ps1`
