$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioEngineHeader = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'rtmutable_boundary_report.json'

if (-not (Test-Path $audioEngineHeader)) {
    throw "Missing AudioEngine header: $audioEngineHeader"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$text = Get-Content -LiteralPath $audioEngineHeader -Raw -Encoding UTF8

function Get-StructBody {
    param(
        [string]$Source,
        [string]$StructName
    )

    $pattern = [regex]::Escape("struct $StructName") + '\s*\{(?<body>.*?)\n\s*\};'
    $match = [regex]::Match($Source, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $match.Success) {
        throw "Unable to locate struct body: $StructName"
    }
    return $match.Groups['body'].Value
}

$rtLocalBody = Get-StructBody -Source $text -StructName 'RTLocalState'
$rtAuxBody = Get-StructBody -Source $text -StructName 'RTAuxMutable'

$violations = New-Object System.Collections.Generic.List[string]

$requiredRtLocalFields = @(
    'audioCallbackEpochCounter',
    'audioSampleCursorCounter',
    'audioCallbackActiveCount',
    'audioThreadRetireEnqueueDropped'
)

foreach ($field in $requiredRtLocalFields) {
    if ($rtLocalBody -notmatch [regex]::Escape($field)) {
        $violations.Add("RTLocalState missing required field: $field")
    }
}

$forbiddenRtLocalPatterns = @(
    'std::shared_ptr',
    'std::mutex',
    'std::unordered_map',
    '\bDSPCore\b',
    '\bRuntimeGraph\b',
    '\bunique_ptr\b',
    '\bshared_ptr\b'
)

foreach ($pattern in $forbiddenRtLocalPatterns) {
    if ($rtLocalBody -match $pattern) {
        $violations.Add("RTLocalState contains forbidden token: $pattern")
    }
}

$forbiddenRTAuxPatterns = @(
    'std::shared_ptr',
    'std::weak_ptr',
    'std::mutex',
    'std::unordered_map',
    '\bDSPCore\b',
    '\bRuntimeGraph\b',
    '\bNonOwningPtr\b',
    '\bunique_ptr\b',
    '\bshared_ptr\b'
)

$forbiddenRTAuxFieldDeclarationPatterns = @(
    '(?m)^\s*[^/\n;]*\*\s*[A-Za-z_]\w*\s*(?:\{[^;]*\})?\s*;',
    '(?m)^\s*std::\s*(?:unique_ptr|shared_ptr|weak_ptr)\s*<[^>]+>\s*[A-Za-z_]\w*\s*(?:\{[^;]*\})?\s*;'
)

foreach ($pattern in $forbiddenRTAuxPatterns) {
    if ($rtAuxBody -match $pattern) {
        $violations.Add("RTAuxMutable contains forbidden token: $pattern")
    }
}

foreach ($pattern in $forbiddenRTAuxFieldDeclarationPatterns) {
    if ($rtAuxBody -match $pattern) {
        $violations.Add("RTAuxMutable contains forbidden field declaration pattern: $pattern")
    }
}

$report = [ordered]@{
    schema                                 = 'rtmutable_boundary_report_v1'
    generatedAt                            = (Get-Date -Format 'o')
    headerPath                             = $audioEngineHeader
    requiredRtLocalFields                  = $requiredRtLocalFields
    forbiddenRtLocalPatterns               = $forbiddenRtLocalPatterns
    forbiddenRTAuxPatterns                 = $forbiddenRTAuxPatterns
    forbiddenRTAuxFieldDeclarationPatterns = $forbiddenRTAuxFieldDeclarationPatterns
    violations                             = $violations
}

Set-Content -LiteralPath $reportPath -Value ($report | ConvertTo-Json -Depth 6) -Encoding UTF8
Write-Host "[INFO] RT mutable boundary report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "RT mutable boundary violations detected. count=$($violations.Count)"
}

Write-Host '[PASS] RT mutable boundary gate verified'
