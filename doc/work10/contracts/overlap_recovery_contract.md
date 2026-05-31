# Overlap Recovery Contract

## Purpose

Define how overlap and crossfade failures recover without authority drift.

## Rules

- Recovery path must be explicit.
- Overlap state must return to semantic ownership boundaries.

## Verification

- `isr-verify-runtime-recovery-semantic.ps1`
