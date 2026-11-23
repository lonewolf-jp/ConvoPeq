param(
    [Parameter(Mandatory = $true)][string]$ArtifactName,
    [Parameter(Mandatory = $true)][string]$Schema,
    [string[]]$RequiredKeys = @()
)

$ErrorActionPreference = 'Stop'
$requireRuntimeEvidence = ($env:ISR_REQUIRE_RUNTIME_EVIDENCE -eq '1')

function Resolve-EvidenceManifestPath {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\..\evidence\evidence_manifest.json"),
        (Join-Path $PSScriptRoot "..\..\build\evidence\evidence_manifest.json")
    )

    foreach ($path in $candidates) {
        $full = [System.IO.Path]::GetFullPath($path)
        if (Test-Path $full) { return $full }
    }

    return $null
}

function Resolve-EvidencePath {
    $candidates = @(
        (Join-Path $PSScriptRoot "..\..\evidence\$ArtifactName"),
        (Join-Path $PSScriptRoot "..\..\build\evidence\$ArtifactName")
    )

    foreach ($path in $candidates) {
        $full = [System.IO.Path]::GetFullPath($path)
        if (Test-Path $full) { return $full }
    }

    throw "Artifact not found: $ArtifactName"
}

function Assert-HasProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if (-not ($Object.PSObject.Properties.Name -contains $Name)) {
        throw "Missing required property '$Name'"
    }
}

function Assert-ValueInSet {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string[]]$Allowed,
        [Parameter(Mandatory = $true)][string]$FieldName
    )

    if ($Allowed -notcontains [string]$Value) {
        throw "Invalid value for '$FieldName'. actual='$Value' allowed='$($Allowed -join ',')'"
    }
}

function Assert-NonNegativeInteger {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$FieldName
    )

    if ($Value -isnot [int] -and $Value -isnot [long]) {
        throw "Field '$FieldName' must be integer. actualType='$($Value.GetType().FullName)'"
    }

    if ([long]$Value -lt 0) {
        throw "Field '$FieldName' must be non-negative. actual='$Value'"
    }
}

function Assert-IsArray {
    param(
        [Parameter(Mandatory = $true)]$Value,
        [Parameter(Mandatory = $true)][string]$FieldName
    )

    if ($Value -is [string] -or $Value -isnot [System.Collections.IEnumerable]) {
        throw "Field '$FieldName' must be array."
    }
}

$artifactPath = Resolve-EvidencePath
$raw = Get-Content -Path $artifactPath -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($raw)) {
    throw "Artifact is empty: $artifactPath"
}

$data = $raw | ConvertFrom-Json
if ($null -eq $data) {
    throw "Artifact JSON parse failed: $artifactPath"
}

if ($requireRuntimeEvidence) {
    if ($data.PSObject.Properties.Name -contains 'provenance') {
        if ([string]$data.provenance -eq 'seed') {
            throw "Strict evidence mode: seeded artifact is not allowed: $ArtifactName"
        }
    }

    $manifestPath = Resolve-EvidenceManifestPath
    if ($null -ne $manifestPath) {
        $manifestRaw = Get-Content -Path $manifestPath -Raw -Encoding UTF8
        if (-not [string]::IsNullOrWhiteSpace($manifestRaw)) {
            $manifest = $manifestRaw | ConvertFrom-Json
            if ($null -ne $manifest) {
                if (($manifest.PSObject.Properties.Name -contains 'generationMode' -and [string]$manifest.generationMode -eq 'seed') -or
                    ($manifest.PSObject.Properties.Name -contains 'generator' -and [string]$manifest.generator -eq 'isr-seed-evidence.ps1')) {
                    throw "Strict evidence mode: seeded evidence manifest detected: $manifestPath"
                }
            }
        }
    }
}

Assert-HasProperty -Object $data -Name 'schema'

if ($data.schema -ne $Schema) {
    throw "Schema mismatch for $ArtifactName. expected=$Schema actual=$($data.schema)"
}

foreach ($key in $RequiredKeys) {
    Assert-HasProperty -Object $data -Name $key
}

# ★ B-11: placeholder（未測定）を証跡として受理しない。
#   アプリ側のエクスポータは、実測していない artifact に status=not_measured /
#   not_run を付けて書き出す。これを合格にすると「生成できた = 安全」と誤読されるため、
#   検証器は明示的に失敗させる。CI のシナリオ生成物はこの status を使わない。
if ($data.PSObject.Properties.Name -contains 'status') {
    $statusValue = [string]$data.status
    if ($statusValue -eq 'not_measured' -or $statusValue -eq 'not_run') {
        throw "Artifact '$ArtifactName' is a placeholder (status=$statusValue): evidence was not measured in this run."
    }
}

# ★ B-11: 合成証跡を実測証跡として受理しない。
#   run スクリプトが -AllowSynthesis で補った artifact は provenance=synthesized を持つ。
#   これを合格にすると「合成できた = 安全」と誤読されるため、明示的に失敗させる。
if ($data.PSObject.Properties.Name -contains 'provenance') {
    $provenanceValue = [string]$data.provenance
    if ($provenanceValue -eq 'synthesized') {
        throw "Artifact '$ArtifactName' is synthesized (provenance=synthesized): a measured artifact is required."
    }
}

Write-Host "[PASS] $ArtifactName schema=$Schema"
return $data
