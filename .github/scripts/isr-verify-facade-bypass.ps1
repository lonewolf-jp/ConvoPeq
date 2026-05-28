$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot 'src\audioengine'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'facade_bypass_report.json'
$triggerAuditReportPath = Join-Path $evidenceDir 'trigger_audit_report.json'

if (-not (Test-Path $audioRoot)) {
    throw "Missing audioengine source root: $audioRoot"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

if (-not (Test-Path -LiteralPath $triggerAuditReportPath)) {
    throw "Missing trigger audit report: $triggerAuditReportPath"
}

$files = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)

    if ([System.IO.Path]::GetPathRoot($baseFull) -ne [System.IO.Path]::GetPathRoot($targetFull)) {
        return $targetFull.Replace('\\', '/')
    }

    $baseWithSep = if ($baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar) -or $baseFull.EndsWith([System.IO.Path]::AltDirectorySeparatorChar)) {
        $baseFull
    }
    else {
        $baseFull + [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = New-Object System.Uri($baseWithSep)
    $targetUri = New-Object System.Uri($targetFull)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)

    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', '/')
}

function Find-Matches {
    param(
        [string]$Pattern
    )

    $foundEntries = New-Object System.Collections.Generic.List[object]
    foreach ($file in $files) {
        $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
        foreach ($match in [regex]::Matches($text, $Pattern)) {
            $foundEntries.Add([ordered]@{
                path = (Get-RelativePathCompat -BasePath $repoRoot -TargetPath $file.FullName)
                value = $match.Value
            }) | Out-Null
        }
    }
    return $foundEntries.ToArray()
}

$helperCreateMatches = Find-Matches -Pattern 'RuntimePublicationCoordinatorFactory::create\('
$directCreateMatches = Find-Matches -Pattern 'RuntimePublicationCoordinator::create\('
$directMemberMatches = Find-Matches -Pattern 'runtimePublicationCoordinator_\.'

$triggerAuditReport = Get-Content -LiteralPath $triggerAuditReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$triggerAuditSchema = "$($triggerAuditReport.schema)"
if ($triggerAuditSchema -ne 'trigger_audit_report_v1') {
    throw "Unexpected trigger audit report schema: $triggerAuditSchema"
}

$violations = New-Object System.Collections.Generic.List[string]

if ($helperCreateMatches.Count -lt 1) {
    $violations.Add('Expected at least one helper create call inside AudioEngine.h')
}

foreach ($entry in $helperCreateMatches) {
    if (($entry.path -replace '\\', '/') -ne 'src/audioengine/AudioEngine.h') {
        $violations.Add("Helper create call must live in src/audioengine/AudioEngine.h, found $($entry.path)")
    }
}

if ($directCreateMatches.Count -gt 0) {
    foreach ($entry in $directCreateMatches) {
        $violations.Add("Direct RuntimePublicationCoordinator::create call found in $($entry.path)")
    }
}

if ($directMemberMatches.Count -gt 0) {
    foreach ($entry in $directMemberMatches) {
        $violations.Add("Direct runtimePublicationCoordinator_ member access found in $($entry.path)")
    }
}

if ($null -eq $triggerAuditReport.metrics -or $null -eq $triggerAuditReport.metrics.retireFacadeRuntimeExecutionCount) {
    $violations.Add('Trigger audit report missing metrics.retireFacadeRuntimeExecutionCount')
}
elseif ([int]$triggerAuditReport.metrics.retireFacadeRuntimeExecutionCount -eq 0 -and $directCreateMatches.Count -gt 0) {
    $violations.Add('Trigger audit reports retireFacadeRuntimeExecutionCount=0 but facade bypass gate detected direct create matches')
}

if ($null -eq $triggerAuditReport.metrics -or $null -eq $triggerAuditReport.metrics.retireFacadeRawDependencyCount) {
    $violations.Add('Trigger audit report missing metrics.retireFacadeRawDependencyCount')
}
elseif ([int]$triggerAuditReport.metrics.retireFacadeRawDependencyCount -eq 0 -and ($directCreateMatches.Count + $directMemberMatches.Count) -gt 0) {
    $violations.Add('Trigger audit reports retireFacadeRawDependencyCount=0 but facade bypass gate detected direct dependencies')
}

$report = [ordered]@{
    schema = 'facade_bypass_report_v1'
    generatedAt = (Get-Date -Format 'o')
    triggerAuditReport = $triggerAuditReportPath
    helperCreateMatches = $helperCreateMatches
    directCreateMatches = $directCreateMatches
    directMemberMatches = $directMemberMatches
    violations = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] facade bypass report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Facade bypass violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] facade bypass gate verified'
