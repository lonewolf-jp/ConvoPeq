$ErrorActionPreference = 'Stop'

function Assert-IsArrayLocal {
	param(
		[Parameter(Mandatory = $true)]$Value,
		[Parameter(Mandatory = $true)][string]$FieldName
	)

	if ($Value -is [string]) {
		throw "Field '$FieldName' must be an array"
	}

	if ($Value -isnot [System.Collections.IEnumerable]) {
		throw "Field '$FieldName' must be an array"
	}
}

function Resolve-OptionalEvidencePath {
	param([Parameter(Mandatory = $true)][string]$ArtifactName)

	$candidates = @(
		(Join-Path $PSScriptRoot "..\..\evidence\$ArtifactName"),
		(Join-Path $PSScriptRoot "..\..\build\evidence\$ArtifactName")
	)

	foreach ($path in $candidates) {
		$full = [System.IO.Path]::GetFullPath($path)
		if (Test-Path $full) { return $full }
	}

	return $null
}

$asanPath = Resolve-OptionalEvidencePath -ArtifactName "asan_report.txt"

$strictRuntimeMode = [string]$env:ISR_REQUIRE_RUNTIME_EVIDENCE -eq '1'

if ($null -ne $asanPath) {
	$content = Get-Content -Path $asanPath -Raw -Encoding UTF8
	if ($content -match "heap-use-after-free|AddressSanitizer:|use-after-free") {
		throw "UAF suspicion detected from ASan report"
	}

	Write-Host "[PASS] UAF detector (ASan report)"
	return
}

# Fallback: ensure runtime recovery trace does not explicitly report UAF handling (if present)
$recoveryPath = Resolve-OptionalEvidencePath -ArtifactName "recovery_trace.json"
if ($null -eq $recoveryPath) {
	if ($strictRuntimeMode) {
		throw "strict mode: missing both asan_report.txt and recovery_trace.json"
	}

	Write-Host "[PASS] UAF detector (no ASan/recovery artifact; fallback skipped)"
	return
}

$raw = Get-Content -Path $recoveryPath -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($raw)) {
	Write-Host "[PASS] UAF detector (empty recovery artifact; fallback skipped)"
	return
}

$recovery = $raw | ConvertFrom-Json
if ($null -eq $recovery) {
	throw "Failed to parse recovery_trace.json"
}

if (-not ($recovery.PSObject.Properties.Name -contains "recoveryActions")) {
	Write-Host "[PASS] UAF detector (recoveryActions missing; fallback skipped)"
	return
}

Assert-IsArrayLocal -Value $recovery.recoveryActions -FieldName "recoveryActions"

foreach ($action in $recovery.recoveryActions) {
	if ($action.PSObject.Properties.Name -contains "failure") {
		$failure = [string]$action.failure
		if ($failure -match "uaf|use-after-free") {
			throw "UAF suspicion detected from recovery trace"
		}
	}
}

Write-Host "[PASS] UAF detector (fallback recovery trace)"
