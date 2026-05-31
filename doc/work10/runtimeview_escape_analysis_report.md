# RuntimeView Escape Analysis Report

## Purpose

Report whether runtime views escape their intended lifetime boundary.

## Rules

- Capture lambdas and async tasks must not retain retired views.
