param(
    [string]$AllowlistPath = '.github/isr-trigger-symbol-allowlist.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"
$evidenceDir = Join-Path $repoRoot "evidence"
$reportPath = Join-Path $evidenceDir "trigger_symbol_usage_report.json"
$resolvedAllowlistPath = if ([System.IO.Path]::IsPathRooted($AllowlistPath)) { $AllowlistPath } else { Join-Path $repoRoot $AllowlistPath }

if (-not (Test-Path $audioRoot)) {
    throw "Missing audioengine source root: $audioRoot"
}
if (-not (Test-Path $resolvedAllowlistPath)) {
    throw "Missing trigger symbol allowlist: $resolvedAllowlistPath"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$allowlist = Get-Content -LiteralPath $resolvedAllowlistPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($allowlist.schema -ne 'trigger_symbol_allowlist_v1') {
    throw "Unexpected trigger symbol allowlist schema: $($allowlist.schema)"
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $allowlist.$field -or [string]::IsNullOrWhiteSpace("$($allowlist.$field)")) {
        throw "Trigger symbol allowlist missing required field: $field"
    }
}

$allowlistExpiry = [datetime]::ParseExact("$($allowlist.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
if ((Get-Date).Date -gt $allowlistExpiry.Date) {
    throw "Trigger symbol allowlist expired: expiry=$($allowlist.expiry) owner=$($allowlist.owner) issue=$($allowlist.issue)"
}

if (-not $allowlist.rules -or $allowlist.rules.Count -eq 0) {
    throw 'Trigger symbol allowlist rules must be non-empty'
}

$today = (Get-Date).Date
$activeRules = New-Object System.Collections.Generic.List[object]
$policyViolations = New-Object System.Collections.Generic.List[string]
$seenRuleKeys = New-Object 'System.Collections.Generic.HashSet[string]'

foreach ($rule in $allowlist.rules) {
    foreach ($field in @('symbol', 'pathRegex', 'owner', 'issue', 'rationale', 'expiry')) {
        if ($null -eq $rule.$field -or [string]::IsNullOrWhiteSpace("$($rule.$field)")) {
            throw "Allowlist rule missing required field: $field"
        }
    }

    $ruleKey = "$($rule.symbol)::$($rule.pathRegex)"
    if (-not $seenRuleKeys.Add($ruleKey)) {
        $policyViolations.Add("Duplicate allowlist rule detected: key=$ruleKey")
    }

    try {
        [void][regex]::new("$($rule.pathRegex)")
    }
    catch {
        $policyViolations.Add("Invalid allowlist pathRegex: symbol=$($rule.symbol) pathRegex=$($rule.pathRegex) reason=$($_.Exception.Message)")
    }

    $expiry = [datetime]::ParseExact("$($rule.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ($today -gt $expiry.Date) {
        $policyViolations.Add("Expired allowlist rule: symbol=$($rule.symbol) pathRegex=$($rule.pathRegex) expiry=$($rule.expiry) owner=$($rule.owner) issue=$($rule.issue)")
    }
    else {
        $activeRules.Add($rule) | Out-Null
    }
}

$files = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc
$allMatches = New-Object System.Collections.Generic.List[object]
$blockedMatches = New-Object System.Collections.Generic.List[object]
$symbolStats = @{}

foreach ($rule in $allowlist.rules | Select-Object -Property symbol -Unique) {
    $symbol = "$($rule.symbol)"
    $pattern = '\b' + [regex]::Escape($symbol) + '\b'
    $symbolStats[$symbol] = [ordered]@{
        symbol = $symbol
        totalMatches = 0
        blockedMatches = 0
    }

    foreach ($file in $files) {
        $relativePath = $file.FullName.Substring($repoRoot.Length + 1).Replace('\', '/')
        $lineNo = 0

        foreach ($line in Get-Content -LiteralPath $file.FullName -Encoding UTF8) {
            $lineNo++
            if ($line -match $pattern) {
                $isAllowed = $false
                foreach ($allowedRule in $activeRules) {
                    if ("$($allowedRule.symbol)" -eq $symbol -and $relativePath -match "$($allowedRule.pathRegex)") {
                        $isAllowed = $true
                        break
                    }
                }

                $entry = [ordered]@{
                    symbol = $symbol
                    path = $relativePath
                    line = $lineNo
                    allowed = $isAllowed
                }
                $allMatches.Add($entry) | Out-Null
                $symbolStats[$symbol].totalMatches = [int]$symbolStats[$symbol].totalMatches + 1

                if (-not $isAllowed) {
                    $blockedMatches.Add($entry) | Out-Null
                    $symbolStats[$symbol].blockedMatches = [int]$symbolStats[$symbol].blockedMatches + 1
                }
            }
        }
    }
}

$symbolStatsList = New-Object System.Collections.Generic.List[object]
foreach ($symbolKey in ($symbolStats.Keys | Sort-Object)) {
    $symbolStatsList.Add($symbolStats[$symbolKey]) | Out-Null
}

$report = [ordered]@{
    schema = 'trigger_symbol_usage_report_v1'
    generatedAt = (Get-Date -Format 'o')
    allowlistPath = $resolvedAllowlistPath
    totalMatches = $allMatches.Count
    blockedMatches = $blockedMatches.Count
    symbolStats = $symbolStatsList
    policyViolations = $policyViolations
    blocked = $blockedMatches
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] trigger symbol usage report written: $reportPath"
Write-Host "[INFO] totalMatches=$($allMatches.Count) blockedMatches=$($blockedMatches.Count)"

if ($policyViolations.Count -gt 0) {
    foreach ($violation in $policyViolations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Trigger symbol allowlist policy violations detected. count=$($policyViolations.Count)"
}

if ($blockedMatches.Count -gt 0) {
    $first = $blockedMatches[0]
    throw "Trigger symbol usage gate failed: symbol=$($first.symbol) path=$($first.path) line=$($first.line)"
}

Write-Host '[PASS] trigger symbol usage gate verified'
