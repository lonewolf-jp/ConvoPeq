$ErrorActionPreference = 'Stop'

$manifest = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName "evidence_manifest.json" -Schema "evidence_manifest_v1" -RequiredKeys @("generationMode", "artifacts")

Assert-ValueInSet -Value $manifest.generationMode -Allowed @("seed", "runtime") -FieldName "generationMode"
Assert-IsArray -Value $manifest.artifacts -FieldName "artifacts"

$schemaByArtifact = @{
    "closure_graph.json"         = "closure_graph_v1"
    "mutation_fault_trace.json"  = "mutation_fault_trace_v1"
    "hb_graph_trace.json"        = "hb_trace_v1"
    "hb_violation_report.json"   = "hb_violation_report_v1"
    "retire_timeline.json"       = "retire_timeline_v2"
    "shutdown_trace.json"        = "shutdown_trace_v1"
    "retire_latency_report.json" = "retire_latency_report_v1"
    "payload_tier_report.json"   = "payload_tier_report_v1"
}

if ($manifest.generationMode -eq 'runtime') {
    $strictRuntimeMode = [string]$env:ISR_REQUIRE_RUNTIME_EVIDENCE -eq '1'

    Assert-HasProperty -Object $manifest -Name "runtimeRunId"
    $runtimeRunId = [string]$manifest.runtimeRunId
    if ([string]::IsNullOrWhiteSpace($runtimeRunId)) {
        throw "Manifest runtimeRunId must not be empty in runtime mode"
    }

    Assert-HasProperty -Object $manifest -Name "generatedAtNs"
    Assert-NonNegativeInteger -Value $manifest.generatedAtNs -FieldName "manifest.generatedAtNs"
    if ([long]$manifest.generatedAtNs -le 0) {
        throw "Manifest generatedAtNs must be > 0 in runtime mode"
    }

    if ($strictRuntimeMode) {
        if ($manifest.PSObject.Properties.Name -contains "generator") {
            $generator = [string]$manifest.generator
            if ($generator -eq 'isr-generate-runtime-evidence.ps1') {
                throw "Strict runtime mode prohibits synthetic generator-only evidence (generator=$generator)"
            }
        }
    }

    $artifactList = @($manifest.artifacts)
    if ($artifactList.Count -eq 0) {
        $evidenceRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\evidence"))
        if (Test-Path $evidenceRoot) {
            $artifactList = @(
                Get-ChildItem -Path $evidenceRoot -File -Filter "*.json" |
                Where-Object { $_.Name -ne "evidence_manifest.json" } |
                ForEach-Object { $_.Name }
            )
        }

        if ($artifactList.Count -eq 0) {
            throw "Runtime mode requires non-empty artifact list"
        }
    }

    foreach ($artifact in $artifactList) {
        $artifactName = [string]$artifact
        if (-not $schemaByArtifact.ContainsKey($artifactName)) {
            continue
        }

        $data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName $artifactName -Schema $schemaByArtifact[$artifactName] -RequiredKeys @("provenance", "runId", "generatedAtNs")

        if ([string]$data.provenance -ne 'runtime') {
            throw "Artifact provenance mismatch: $artifactName expected=runtime actual=$($data.provenance)"
        }

        if ([string]$data.runId -ne $runtimeRunId) {
            throw "Artifact runId mismatch: $artifactName artifactRunId=$($data.runId) runtimeRunId=$runtimeRunId"
        }

        Assert-NonNegativeInteger -Value $data.generatedAtNs -FieldName "$artifactName.generatedAtNs"
        if ([long]$data.generatedAtNs -le 0) {
            throw "Artifact generatedAtNs must be > 0: $artifactName"
        }
    }

    Write-Host "[PASS] evidence provenance gate (runtime mode)"
    return
}

if ($manifest.generationMode -ne 'seed') {
    throw "Unsupported generationMode: $($manifest.generationMode)"
}

Assert-HasProperty -Object $manifest -Name "runId"
$seedRunId = [string]$manifest.runId
if ([string]::IsNullOrWhiteSpace($seedRunId)) {
    throw "Manifest runId must not be empty in seed mode"
}

$requiredArtifacts = @(
    "closure_graph.json",
    "mutation_fault_trace.json",
    "hb_graph_trace.json",
    "hb_violation_report.json",
    "retire_timeline.json",
    "shutdown_trace.json",
    "retire_latency_report.json",
    "payload_tier_report.json"
)

$manifestSet = New-Object 'System.Collections.Generic.HashSet[string]'
foreach ($name in @($manifest.artifacts)) {
    [void]$manifestSet.Add([string]$name)
}

foreach ($artifact in $requiredArtifacts) {
    if (-not $manifestSet.Contains($artifact)) {
        throw "Manifest missing required artifact entry: $artifact"
    }

    $data = . "$PSScriptRoot\isr-verify-common.ps1" -ArtifactName $artifact -Schema $schemaByArtifact[$artifact] -RequiredKeys @("provenance", "runId")

    if ([string]$data.provenance -ne 'seed') {
        throw "Artifact provenance mismatch: $artifact expected=seed actual=$($data.provenance)"
    }

    if ([string]$data.runId -ne $seedRunId) {
        throw "Artifact runId mismatch: $artifact artifactRunId=$($data.runId) manifestRunId=$seedRunId"
    }
}

Write-Host "[PASS] evidence provenance gate (seed mode)"
