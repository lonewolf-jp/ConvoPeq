param(
    [string]$RegistryPath = '.github/isr-clang-tidy-rule-registry.json',
    [string]$PolicyPath = '.github/isr-clang-tidy-audit-policy.json',
    [ValidateSet('smoke', 'standard', 'exhaustive')]
    [string]$Tier = 'standard',
    [switch]$RequireClangTidy
)

$ErrorActionPreference = 'Stop'

# clang-tidy may emit progress/status lines on stderr even on successful runs.
# In newer PowerShell versions, native stderr can be promoted to Error records
# under ErrorActionPreference=Stop, which would abort this gate prematurely.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'clang_tidy_audit_report.json'
$resolvedRegistryPath = if ([System.IO.Path]::IsPathRooted($RegistryPath)) { $RegistryPath } else { Join-Path $repoRoot $RegistryPath }
$resolvedPolicyPath = if ([System.IO.Path]::IsPathRooted($PolicyPath)) { $PolicyPath } else { Join-Path $repoRoot $PolicyPath }
$compileCommandsPath = Join-Path $repoRoot 'build/compile_commands.json'

if (-not (Test-Path $resolvedRegistryPath)) {
    throw "Missing clang-tidy registry: $resolvedRegistryPath"
}
if (-not (Test-Path $resolvedPolicyPath)) {
    throw "Missing clang-tidy audit policy: $resolvedPolicyPath"
}

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$registry = Get-Content -LiteralPath $resolvedRegistryPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($registry.schema -ne 'clang_tidy_rule_registry_v1') {
    throw "Unexpected clang-tidy registry schema: $($registry.schema)"
}

$policy = Get-Content -LiteralPath $resolvedPolicyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($policy.schema -ne 'clang_tidy_audit_policy_v1') {
    throw "Unexpected clang-tidy audit policy schema: $($policy.schema)"
}
foreach ($field in @('mode', 'owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "clang-tidy audit policy missing required field: $field"
    }
}

$policyMode = "$($policy.mode)"
if (@('monitor', 'enforce') -notcontains $policyMode) {
    throw "Unsupported clang-tidy audit policy mode: $policyMode"
}

$policyExpiry = [datetime]::ParseExact("$($policy.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "clang-tidy audit policy expired: $($policy.expiry)"
}

$enforceByTier = $false
if ($policy.enforceTiers) {
    foreach ($candidateTier in @($policy.enforceTiers)) {
        if ("$candidateTier" -eq $Tier) {
            $enforceByTier = $true
            break
        }
    }
}

$effectiveRequireClangTidy = [bool]($RequireClangTidy -or $policyMode -eq 'enforce' -or $enforceByTier)

$requiredChecks = @()
if ($registry.requiredChecks) {
    $requiredChecks = @($registry.requiredChecks | ForEach-Object { "$_" } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
if ($requiredChecks.Count -eq 0) {
    throw 'clang-tidy registry requires non-empty requiredChecks for audit gate'
}

function Find-ClangTidyCandidates {
    $results = New-Object System.Collections.Generic.List[string]
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)

    $pathProbe = Get-Command clang-tidy -ErrorAction SilentlyContinue
    if ($pathProbe -and -not [string]::IsNullOrWhiteSpace("$($pathProbe.Source)")) {
        $src = "$($pathProbe.Source)"
        if ($seen.Add($src)) { $results.Add($src) | Out-Null }
    }

    try {
        $whereOut = & where.exe clang-tidy 2>$null
        foreach ($line in @($whereOut)) {
            $candidate = "$line".Trim()
            if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
                if ($seen.Add($candidate)) { $results.Add($candidate) | Out-Null }
            }
        }
    }
    catch {
    }

    $probeDirs = @(
        'C:/Program Files/LLVM/bin',
        'C:/Program Files (x86)/LLVM/bin',
        'C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/Llvm/x64/bin',
        'C:/Program Files/Microsoft Visual Studio/2022/Professional/VC/Tools/Llvm/x64/bin',
        'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/Llvm/x64/bin',
        'C:/Program Files/Microsoft Visual Studio/18/Enterprise/VC/Tools/Llvm/x64/bin',
        'C:/Program Files/Microsoft Visual Studio/18/Professional/VC/Tools/Llvm/x64/bin',
        'C:/Program Files/Microsoft Visual Studio/18/Community/VC/Tools/Llvm/x64/bin'
    )

    foreach ($dir in $probeDirs) {
        $exePath = Join-Path $dir 'clang-tidy.exe'
        if (Test-Path $exePath) {
            $normalized = [System.IO.Path]::GetFullPath($exePath)
            if ($seen.Add($normalized)) { $results.Add($normalized) | Out-Null }
        }
    }

    return @($results)
}

$clangTidyCmd = Get-Command clang-tidy -ErrorAction SilentlyContinue
$clangTidyCandidates = Find-ClangTidyCandidates
$clangTidyExecutablePath = $null
if ($clangTidyCmd -and -not [string]::IsNullOrWhiteSpace("$($clangTidyCmd.Source)")) {
    $clangTidyExecutablePath = "$($clangTidyCmd.Source)"
}
elseif ($clangTidyCandidates.Count -gt 0) {
    $clangTidyExecutablePath = "$($clangTidyCandidates[0])"
}
$status = 'ok'
$violations = New-Object System.Collections.Generic.List[string]
$auditOutput = ''
$targetFile = $null
$usedChecks = ($requiredChecks -join ',')

if ([string]::IsNullOrWhiteSpace($clangTidyExecutablePath)) {
    if ($effectiveRequireClangTidy) {
        $violations.Add('clang-tidy executable not found while RequireClangTidy is enabled')
        $status = 'failed'
    }
    else {
        $status = 'skipped'
        $violations.Add('clang-tidy executable not found (monitor mode skip)')
    }
}
elseif (-not (Test-Path $compileCommandsPath)) {
    if ($effectiveRequireClangTidy) {
        $violations.Add("compile_commands.json missing: $compileCommandsPath")
        $status = 'failed'
    }
    else {
        $status = 'skipped'
        $violations.Add("compile_commands.json missing (monitor mode skip): $compileCommandsPath")
    }
}
else {
    $compileCommands = Get-Content -LiteralPath $compileCommandsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $candidate = $compileCommands |
    Where-Object {
        $filePath = "$($_.file)"
        -not [string]::IsNullOrWhiteSpace($filePath) -and
        $filePath.Replace('\\', '/').Contains('/src/audioengine/') -and
        $filePath -match '\.(cpp|cc|cxx)$'
    } |
    Select-Object -First 1

    if (-not $candidate) {
        $candidate = $compileCommands |
        Where-Object {
            $filePath = "$($_.file)"
            -not [string]::IsNullOrWhiteSpace($filePath) -and
            $filePath -match '\.(cpp|cc|cxx)$'
        } |
        Select-Object -First 1
    }

    if (-not $candidate) {
        if ($effectiveRequireClangTidy) {
            $violations.Add('No C++ translation unit found in compile_commands.json')
            $status = 'failed'
        }
        else {
            $status = 'skipped'
            $violations.Add('No C++ translation unit found in compile_commands.json (monitor mode skip)')
        }
    }
    else {
        $targetFile = "$($candidate.file)"
        if (-not [System.IO.Path]::IsPathRooted($targetFile)) {
            $targetFile = Join-Path "$($candidate.directory)" $targetFile
        }

        if (-not (Test-Path $targetFile)) {
            if ($effectiveRequireClangTidy) {
                $violations.Add("clang-tidy target source file not found: $targetFile")
                $status = 'failed'
            }
            else {
                $status = 'skipped'
                $violations.Add("clang-tidy target source file not found (monitor mode skip): $targetFile")
            }
        }
        else {
            $runArgs = @(
                $targetFile,
                "-p=$([System.IO.Path]::GetDirectoryName($compileCommandsPath))",
                '--quiet',
                "--checks=$usedChecks"
            )

            $prevErrorActionPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $outputLines = & $clangTidyExecutablePath @runArgs 2>&1
            }
            finally {
                $ErrorActionPreference = $prevErrorActionPreference
            }
            $exitCode = $LASTEXITCODE
            $auditOutput = ($outputLines | Out-String)

            if ($exitCode -ne 0) {
                $status = 'failed'
                $violations.Add("clang-tidy invocation failed (exit=$exitCode) target=$targetFile")
            }
        }
    }
}

$report = [ordered]@{
    schema                    = 'clang_tidy_audit_report_v1'
    generatedAt               = (Get-Date -Format 'o')
    tier                      = $Tier
    policyPath                = $resolvedPolicyPath
    policyMode                = $policyMode
    enforceByTier             = $enforceByTier
    requireClangTidy          = [bool]$RequireClangTidy
    effectiveRequireClangTidy = $effectiveRequireClangTidy
    registryPath              = $resolvedRegistryPath
    compileCommandsPath       = $compileCommandsPath
    status                    = $status
    checkCount                = $requiredChecks.Count
    checks                    = $requiredChecks
    clangTidyPath             = if ([string]::IsNullOrWhiteSpace($clangTidyExecutablePath)) { '' } else { $clangTidyExecutablePath }
    clangTidyCandidates       = $clangTidyCandidates
    targetFile                = $targetFile
    violations                = $violations
    outputPreview             = if ([string]::IsNullOrWhiteSpace($auditOutput)) { '' } else { $auditOutput.Substring(0, [Math]::Min(4000, $auditOutput.Length)) }
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] clang-tidy audit report written: $reportPath"

if ($status -eq 'failed') {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "clang-tidy audit gate failed. count=$($violations.Count)"
}

if ($status -eq 'skipped') {
    foreach ($violation in $violations) {
        Write-Host "[WARN] $violation"
    }
    Write-Host '[PASS] clang-tidy audit gate skipped in monitor mode'
    return
}

Write-Host '[PASS] clang-tidy audit gate verified'
