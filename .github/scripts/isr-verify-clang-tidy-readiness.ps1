param(
    [string]$RegistryPath = '.github/isr-clang-tidy-rule-registry.json',
    [string]$ClangTidyPath = '.clang-tidy',
    [string]$CMakePath = 'CMakeLists.txt'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'clang_tidy_readiness_report.json'
$resolvedRegistryPath = if ([System.IO.Path]::IsPathRooted($RegistryPath)) { $RegistryPath } else { Join-Path $repoRoot $RegistryPath }
$resolvedClangTidyPath = if ([System.IO.Path]::IsPathRooted($ClangTidyPath)) { $ClangTidyPath } else { Join-Path $repoRoot $ClangTidyPath }
$resolvedCMakePath = if ([System.IO.Path]::IsPathRooted($CMakePath)) { $CMakePath } else { Join-Path $repoRoot $CMakePath }

foreach ($path in @($resolvedRegistryPath, $resolvedClangTidyPath, $resolvedCMakePath)) {
    if (-not (Test-Path $path)) {
        throw "Missing required file: $path"
    }
}

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$registry = Get-Content -LiteralPath $resolvedRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
$clangTidyText = Get-Content -LiteralPath $resolvedClangTidyPath -Raw -Encoding UTF8
$cmakeText = Get-Content -LiteralPath $resolvedCMakePath -Raw -Encoding UTF8

if ($registry.schema -ne 'clang_tidy_rule_registry_v1') {
    throw "Unexpected clang-tidy registry schema: $($registry.schema)"
}

$violations = New-Object System.Collections.Generic.List[string]

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ([string]::IsNullOrWhiteSpace("$($registry.$field)")) {
        $violations.Add("clang-tidy registry missing required field: $field")
    }
}

$today = (Get-Date).Date
$expiry = [datetime]::ParseExact("$($registry.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ($today -gt $expiry.Date) {
    $violations.Add("clang-tidy registry entry expired: expiry=$($registry.expiry) owner=$($registry.owner) issue=$($registry.issue)")
}

if (-not $registry.requiredChecks -or $registry.requiredChecks.Count -lt 3) {
    $violations.Add('clang-tidy registry requires at least 3 required checks')
}

$enabledChecks = New-Object System.Collections.Generic.List[string]
foreach ($check in $registry.requiredChecks) {
    $name = "$check"
    if ([string]::IsNullOrWhiteSpace($name)) {
        $violations.Add('clang-tidy registry contains empty required check')
        continue
    }

    if ($clangTidyText -notmatch [regex]::Escape($name)) {
        $violations.Add(".clang-tidy missing required check token: $name")
    }
    else {
        $enabledChecks.Add($name) | Out-Null
    }
}

if ($cmakeText -notmatch 'option\(CONVOPEQ_ENABLE_CLANG_TIDY') {
    $violations.Add('CMakeLists missing CONVOPEQ_ENABLE_CLANG_TIDY option')
}
if ($cmakeText -notmatch 'find_program\(CLANG_TIDY_EXECUTABLE\s+clang-tidy\)') {
    $violations.Add('CMakeLists missing clang-tidy executable discovery')
}
if ($cmakeText -notmatch 'set\(CMAKE_EXPORT_COMPILE_COMMANDS\s+ON\)') {
    $violations.Add('CMakeLists missing compile_commands export required for clang-tidy')
}

$report = [ordered]@{
    schema = 'clang_tidy_readiness_report_v1'
    generatedAt = (Get-Date -Format 'o')
    registryPath = $resolvedRegistryPath
    clangTidyPath = $resolvedClangTidyPath
    cmakePath = $resolvedCMakePath
    requiredCheckCount = if ($registry.requiredChecks) { $registry.requiredChecks.Count } else { 0 }
    detectedCheckCount = $enabledChecks.Count
    detectedChecks = $enabledChecks
    violations = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] clang-tidy readiness report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "clang-tidy readiness gate failed. count=$($violations.Count)"
}

Write-Host '[PASS] clang-tidy readiness gate verified'
