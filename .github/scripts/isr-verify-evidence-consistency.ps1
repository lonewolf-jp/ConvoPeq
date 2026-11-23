param(
    # 検査対象の evidence ルート。既定はリポジトリ直下の evidence/。
    [string]$EvidenceRoot = ""
)

$ErrorActionPreference = 'Stop'

# ★ B-11: manifest と実体の整合性を検査する。
#   - manifest.artifacts の全件が実在する
#   - 実体にある *.json が manifest に列挙されている
#   - manifest.artifactDetail が全件を覆い、measured が provenance/status と一致する
#   - synthesized を measured として扱っていない
#   - runId が manifest と一致する
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($EvidenceRoot)) {
    $EvidenceRoot = Join-Path $repoRoot "evidence"
}

$manifestPath = Join-Path $EvidenceRoot "evidence_manifest.json"
if (-not (Test-Path $manifestPath)) {
    throw "evidence_manifest.json not found: $manifestPath"
}

$manifest = Get-Content -Path $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($null -eq $manifest) {
    throw "Failed to parse evidence manifest: $manifestPath"
}

$listed = @()
if ($manifest.PSObject.Properties.Name -contains "artifacts") {
    $listed = @($manifest.artifacts)
}
if ($listed.Count -eq 0) {
    throw "manifest.artifacts is empty: $manifestPath"
}

$missing = @($listed | Where-Object { -not (Test-Path (Join-Path $EvidenceRoot $_)) })
if ($missing.Count -gt 0) {
    throw ("manifest lists artifacts that do not exist: " + ($missing -join ", "))
}

$onDisk = @(Get-ChildItem -Path $EvidenceRoot -File -Filter "*.json" |
            Where-Object { $_.Name -ne "evidence_manifest.json" } |
            ForEach-Object { $_.Name })
$unlisted = @($onDisk | Where-Object { $listed -notcontains $_ })
if ($unlisted.Count -gt 0) {
    throw ("artifacts exist on disk but are not listed in manifest: " + ($unlisted -join ", "))
}

$detail = @()
if ($manifest.PSObject.Properties.Name -contains "artifactDetail") {
    $detail = @($manifest.artifactDetail)
}
if ($detail.Count -ne $listed.Count) {
    throw "artifactDetail count ($($detail.Count)) != artifacts count ($($listed.Count))"
}

$measuredCount = 0
foreach ($entry in $detail) {
    $artifactPath = Join-Path $EvidenceRoot $entry.name
    $json = Get-Content -Path $artifactPath -Raw -Encoding UTF8 | ConvertFrom-Json

    $provenance = if ($json.PSObject.Properties.Name -contains "provenance") { [string]$json.provenance } else { "unknown" }
    $status = if ($json.PSObject.Properties.Name -contains "status") { [string]$json.status } else { "unknown" }
    $measured = ($provenance -eq "runtime" -and $status -ne "not_measured" -and $status -ne "not_run")

    if ([bool]$entry.measured -ne $measured) {
        throw "artifactDetail.measured mismatch for $($entry.name) (detail=$($entry.measured) actual=$measured)"
    }
    if ($provenance -eq "synthesized" -and [bool]$entry.measured) {
        throw "synthesized artifact is marked measured: $($entry.name)"
    }
    if ($json.PSObject.Properties.Name -contains "runId" -and
        $manifest.PSObject.Properties.Name -contains "runtimeRunId" -and
        [string]$json.runId -ne [string]$manifest.runtimeRunId) {
        throw "runId mismatch for $($entry.name): artifact=$($json.runId) manifest=$($manifest.runtimeRunId)"
    }

    if ($measured) { ++$measuredCount }
}

Write-Host "[PASS] evidence consistency ($($listed.Count) artifacts, measured=$measuredCount)"
