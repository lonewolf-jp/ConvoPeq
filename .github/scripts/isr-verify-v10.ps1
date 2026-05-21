$ErrorActionPreference = 'Stop'
& "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "runtime_budget_report.json" -Schema "runtime_budget_report_v1" -RequiredKeys @("artifactTotalBytes")
