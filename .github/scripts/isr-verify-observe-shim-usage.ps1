param(
    [string]$AllowlistPath = '.github/isr-observe-shim-allowlist.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"
$evidenceDir = Join-Path $repoRoot "evidence"
$reportPath = Join-Path $evidenceDir "observe_shim_usage_report.json"
$resolvedAllowlistPath = if ([System.IO.Path]::IsPathRooted($AllowlistPath)) { $AllowlistPath } else { Join-Path $repoRoot $AllowlistPath }

if (-not (Test-Path $audioRoot)) {
    throw "Missing audioengine source root: $audioRoot"
}
if (-not (Test-Path $resolvedAllowlistPath)) {
    throw "Missing observe shim allowlist: $resolvedAllowlistPath"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$allowlist = Get-Content -LiteralPath $resolvedAllowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($allowlist.schema -ne 'observe_shim_allowlist_v1') {
    throw "Unexpected observe shim allowlist schema: $($allowlist.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $allowlist.$field -or [string]::IsNullOrWhiteSpace("$($allowlist.$field)")) {
        throw "observe shim allowlist missing required field: $field"
    }
}

$allowlistExpiry = [datetime]::ParseExact("$($allowlist.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $allowlistExpiry.Date) {
    throw "observe shim allowlist expired: expiry=$($allowlist.expiry) owner=$($allowlist.owner) issue=$($allowlist.issue)"
}

$ruleCount = if ($null -eq $allowlist.rules) { 0 } else { [int]$allowlist.rules.Count }

$today = (Get-Date).Date
$activeRules = New-Object System.Collections.Generic.List[object]
$policyViolations = New-Object System.Collections.Generic.List[string]

foreach ($rule in $allowlist.rules) {
    foreach ($field in @('symbol', 'pathRegex', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $rule.$field -or [string]::IsNullOrWhiteSpace("$($rule.$field)")) {
            throw "Allowlist rule missing required field: $field"
        }
    }

    $expiry = [datetime]::ParseExact("$($rule.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ($today -gt $expiry.Date) {
        $policyViolations.Add("Expired observe shim allowlist rule: symbol=$($rule.symbol) pathRegex=$($rule.pathRegex) expiry=$($rule.expiry) owner=$($rule.owner) issue=$($rule.issue)")
    }
    else {
        $activeRules.Add($rule) | Out-Null
    }
}

$files = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc
$pattern = '\bruntimeStore\.observe\s*\('
$allMatches = New-Object System.Collections.Generic.List[object]
$blockedMatches = New-Object System.Collections.Generic.List[object]

foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($repoRoot.Length + 1).Replace('\', '/')
    $lineNo = 0

    foreach ($line in Get-Content -LiteralPath $file.FullName -Encoding UTF8) {
        $lineNo++
        if ($line -match $pattern) {
            $isAllowed = $false
            foreach ($allowedRule in $activeRules) {
                if ("$($allowedRule.symbol)" -eq 'runtimeStore.observe' -and $relativePath -match "$($allowedRule.pathRegex)") {
                    $isAllowed = $true
                    break
                }
            }

            $entry = [ordered]@{
                symbol = 'runtimeStore.observe'
                path = $relativePath
                line = $lineNo
                allowed = $isAllowed
            }
            $allMatches.Add($entry) | Out-Null

            if (-not $isAllowed) {
                $blockedMatches.Add($entry) | Out-Null
            }
        }
    }
}

$report = [ordered]@{
    schema = 'observe_shim_usage_report_v1'
    generatedAt = (Get-Date -Format 'o')
    allowlistPath = $resolvedAllowlistPath
    allowlistOwner = "$($allowlist.owner)"
    allowlistIssue = "$($allowlist.issue)"
    allowlistExpiry = "$($allowlist.expiry)"
    ruleCount = $ruleCount
    totalMatches = $allMatches.Count
    blockedMatches = $blockedMatches.Count
    policyViolations = $policyViolations
    blocked = $blockedMatches
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] observe shim usage report written: $reportPath"
Write-Host "[INFO] runtimeStore.observe matches=$($allMatches.Count) blockedMatches=$($blockedMatches.Count) ruleCount=$ruleCount"

if ($policyViolations.Count -gt 0) {
    foreach ($violation in $policyViolations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Observe shim allowlist policy violations detected. count=$($policyViolations.Count)"
}

if ($blockedMatches.Count -gt 0) {
    $first = $blockedMatches[0]
    throw "Observe shim usage gate failed: path=$($first.path) line=$($first.line)"
}

Write-Host '[PASS] observe shim usage gate verified'
