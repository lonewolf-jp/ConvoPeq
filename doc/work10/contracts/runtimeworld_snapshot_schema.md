# RuntimeWorld Snapshot Schema

## Required Fields

- schemaVersion
- publicationSequence
- generation
- topology
- execution
- routing
- publication
- overlap
- retire
- semanticHash

## Optional Fields

- diagnostic and telemetry sections explicitly tagged as optional

## Constraints

- deterministic key naming
- stable numeric representation
- monotonic publication sequence visibility

## Verification

- schema validation test in runtime semantic tests
