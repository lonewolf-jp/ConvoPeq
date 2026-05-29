param(
    [ValidateSet('warn', 'fail')]
    [string]$Mode = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyPath = Join-Path $repoRoot '.github\isr-ai-governance-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_shutdown_reclaim_report.json'

$policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ("$($policy.schema)" -ne 'isr_ai_governance_policy_v1') {
    throw "Unexpected policy schema: $($policy.schema)"
}

function ConvertTo-IsoDateStrict {
    param(
        [Parameter(Mandatory = $true)][string]$Raw,
        [Parameter(Mandatory = $true)][string]$Context
    )

    try {
        return [datetime]::ParseExact($Raw, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    }
    catch {
        throw "Invalid date format at ${Context}: '$Raw' (expected yyyy-MM-dd)"
    }
}

function Assert-AllowlistContract {
    param(
        [Parameter(Mandatory = $true)]$Entries,
        [Parameter(Mandatory = $true)][string]$Context,
        [Parameter(Mandatory = $true)][string[]]$RequiredFields
    )

    $today = (Get-Date).Date
    $index = 0
    foreach ($entry in @($Entries)) {
        $index++
        foreach ($field in $RequiredFields) {
            $value = "$($entry.$field)"
            if ([string]::IsNullOrWhiteSpace($value)) {
                throw "$Context[$index] missing required field: $field"
            }
        }

        $expiry = ConvertTo-IsoDateStrict -Raw "$($entry.expiry)" -Context "$Context[$index].expiry"
        if ($today -gt $expiry.Date) {
            throw "$Context[$index] expired: expiry=$($entry.expiry)"
        }
    }
}

function Assert-RegexCompiles {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Context
    )

    if ([string]::IsNullOrWhiteSpace($Pattern)) {
        return
    }

    try {
        [void][System.Text.RegularExpressions.Regex]::new($Pattern)
    }
    catch {
        throw "Invalid regex at ${Context}: '$Pattern'"
    }
}

function Assert-RegexFields {
    param(
        [Parameter(Mandatory = $true)]$Entries,
        [Parameter(Mandatory = $true)][string]$Context,
        [Parameter(Mandatory = $true)][string[]]$Fields
    )

    $index = 0
    foreach ($entry in @($Entries)) {
        $index++
        foreach ($field in $Fields) {
            $value = "$($entry.$field)"
            Assert-RegexCompiles -Pattern $value -Context "$Context[$index].$field"
        }
    }
}

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ($null -eq $policy.$field -or [string]::IsNullOrWhiteSpace("$($policy.$field)")) {
        throw "Policy missing required field: $field"
    }
}

$policyExpiry = ConvertTo-IsoDateStrict -Raw "$($policy.expiry)" -Context 'policy.expiry'
if ((Get-Date).Date -gt $policyExpiry.Date) {
    throw "Policy expired: expiry=$($policy.expiry)"
}

Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.allowlist) -Context 'shutdownReclaimChecks.allowlist' -RequiredFields @('pathRegex', 'functionNameContains', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.requiredBoundedReclaimApplications) -Context 'shutdownReclaimChecks.requiredBoundedReclaimApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications) -Context 'shutdownReclaimChecks.requiredDrainAuthorityApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications) -Context 'shutdownReclaimChecks.requiredBoundedDrainWaitApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist) -Context 'shutdownReclaimChecks.waitForDrainCallsiteAllowlist' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist) -Context 'shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.requiredEmergencyReclaimApplications) -Context 'shutdownReclaimChecks.requiredEmergencyReclaimApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationApplications) -Context 'shutdownReclaimChecks.requiredReclaimPrioritizationApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.allowlist) -Context 'shutdownReclaimChecks.allowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.requiredBoundedReclaimApplications) -Context 'shutdownReclaimChecks.requiredBoundedReclaimApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications) -Context 'shutdownReclaimChecks.requiredDrainAuthorityApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications) -Context 'shutdownReclaimChecks.requiredBoundedDrainWaitApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist) -Context 'shutdownReclaimChecks.waitForDrainCallsiteAllowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist) -Context 'shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.requiredEmergencyReclaimApplications) -Context 'shutdownReclaimChecks.requiredEmergencyReclaimApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationApplications) -Context 'shutdownReclaimChecks.requiredReclaimPrioritizationApplications' -Fields @('pathRegex', 'lineRegex')

foreach ($pattern in @($policy.shutdownReclaimChecks.forbiddenPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.forbiddenPatterns[]'
}
foreach ($pattern in @($policy.shutdownReclaimChecks.requiredBoundedReclaimPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.requiredBoundedReclaimPatterns[]'
}
foreach ($pattern in @($policy.shutdownReclaimChecks.requiredDrainAuthorityPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.requiredDrainAuthorityPatterns[]'
}
foreach ($pattern in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.requiredBoundedDrainWaitPatterns[]'
}
foreach ($pattern in @($policy.shutdownReclaimChecks.requiredEmergencyReclaimPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.requiredEmergencyReclaimPatterns[]'
}
foreach ($pattern in @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationPatterns)) {
    Assert-RegexCompiles -Pattern "$pattern" -Context 'shutdownReclaimChecks.requiredReclaimPrioritizationPatterns[]'
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

function Get-RelativePathCompat {
    param([string]$BasePath, [string]$TargetPath)
    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)
    if ($targetFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $targetFull.Substring($baseFull.Length).TrimStart([char[]]@([char]'\', [char]'/')).Replace('\', '/')
    }
    return $targetFull.Replace('\', '/')
}

$effectiveMode = $Mode
if ([string]::IsNullOrWhiteSpace($effectiveMode)) {
    $effectiveMode = "$($policy.shutdownReclaimChecks.phase)"
    if ([string]::IsNullOrWhiteSpace($effectiveMode)) { $effectiveMode = 'warn' }
}

if (@('warn', 'fail') -notcontains $effectiveMode) {
    throw "Invalid shutdownReclaimChecks.phase: $effectiveMode (expected warn|fail)"
}

$sourceRoot = Join-Path $repoRoot 'src'
$cppFiles = Get-ChildItem -Path $sourceRoot -Recurse -File -Filter '*.cpp'
$releaseResourcesPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.ReleaseResources.cpp'
$ctorDtorPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.CtorDtor.cpp'
$sourceRelativePaths = @()
foreach ($f in @($cppFiles)) {
    $sourceRelativePaths += Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
}

function Test-RegexMatchesAnyPath {
    param(
        [Parameter(Mandatory = $true)][string]$PathRegex,
        [Parameter(Mandatory = $true)][string[]]$RelativePaths
    )

    foreach ($rp in $RelativePaths) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($rp, $PathRegex)) {
            return $true
        }
    }

    return $false
}

$violations = @()
$functionScopeViolations = @()

foreach ($pattern in @($policy.shutdownReclaimChecks.forbiddenPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$pattern")) { continue }
    foreach ($f in $cppFiles) {
        $hits = Select-String -Path $f.FullName -Pattern "$pattern" -Encoding UTF8
        $fileText = Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8
        $functionHeaders = [System.Text.RegularExpressions.Regex]::Matches($fileText,
            '([\w:<>~]+\s+)?(AudioEngine|EQProcessor)::(?<fname>\w+|~\w+)\s*\(',
            [System.Text.RegularExpressions.RegexOptions]::Singleline)
        foreach ($h in $hits) {
            $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $h.Path
            $lineText = "$($h.Line)"
            $charIndex = 0
            $linePrefix = Select-String -InputObject $fileText -Pattern "^(?:.*`n){$([Math]::Max(0, $h.LineNumber - 1))}" -AllMatches
            if ($linePrefix -and $linePrefix.Matches.Count -gt 0) {
                $charIndex = $linePrefix.Matches[0].Length
            }

            $enclosingFunction = ''
            foreach ($fh in $functionHeaders) {
                if ($fh.Index -le $charIndex) {
                    $enclosingFunction = $fh.Groups['fname'].Value
                }
                else {
                    break
                }
            }

            $allowed = $false
            $functionAllowed = $false
            foreach ($allow in @($policy.shutdownReclaimChecks.allowlist)) {
                if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($allow.pathRegex)") -and
                    [System.Text.RegularExpressions.Regex]::IsMatch($lineText, "$($allow.lineRegex)")) {
                    $allowed = $true
                    $needle = "$($allow.functionNameContains)"
                    if (-not [string]::IsNullOrWhiteSpace($needle) -and $enclosingFunction -like "*$needle*") {
                        $functionAllowed = $true
                    }
                    break
                }
            }

            if (-not $allowed) {
                $violations += [ordered]@{
                    file    = $relativePath
                    line    = $h.LineNumber
                    snippet = $lineText.Trim()
                    checkId = 'CI-SHUTDOWN-001'
                    message = "forbidden shutdown/reclaim pattern detected: $pattern"
                }
            }
            elseif (-not $functionAllowed) {
                $functionScopeViolations += [ordered]@{
                    file    = $relativePath
                    line    = $h.LineNumber
                    snippet = $lineText.Trim()
                    checkId = 'CI-SHUTDOWN-002'
                    message = "allowlist matched but functionNameContains mismatch (function='$enclosingFunction')"
                }
            }
        }
    }
}

$requiredPatternViolations = @()
$requiredPatterns = @($policy.shutdownReclaimChecks.requiredBoundedReclaimPatterns)
foreach ($required in $requiredPatterns) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $cppFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-001'
            message = "required bounded reclaim pattern not found: $required"
        }
    }
}

$requiredApplicationViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredBoundedReclaimApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-001'
            message = 'invalid requiredBoundedReclaimApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $cppFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-001'
            message = "required bounded reclaim application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$requiredDrainPatternViolations = @()
foreach ($required in @($policy.shutdownReclaimChecks.requiredDrainAuthorityPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $cppFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredDrainPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-003'
            message = "required drain authority pattern not found: $required"
        }
    }
}

$requiredDrainApplicationViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredDrainApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-003'
            message = 'invalid requiredDrainAuthorityApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $cppFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredDrainApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-003'
            message = "required drain authority application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$requiredDrainWaitPatternViolations = @()
foreach ($required in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $cppFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredDrainWaitPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-004'
            message = "required bounded drain wait pattern not found: $required"
        }
    }
}

$requiredDrainWaitApplicationViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredDrainWaitApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-004'
            message = 'invalid requiredBoundedDrainWaitApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $cppFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredDrainWaitApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-004'
            message = "required bounded drain wait application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$requiredEmergencyReclaimPatternViolations = @()
foreach ($required in @($policy.shutdownReclaimChecks.requiredEmergencyReclaimPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $cppFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredEmergencyReclaimPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-002'
            message = "required emergency reclaim pattern not found: $required"
        }
    }
}

$requiredEmergencyReclaimApplicationViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredEmergencyReclaimApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredEmergencyReclaimApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-002'
            message = 'invalid requiredEmergencyReclaimApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $cppFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredEmergencyReclaimApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-002'
            message = "required emergency reclaim application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$requiredReclaimPrioritizationPatternViolations = @()
foreach ($required in @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $cppFiles) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredReclaimPrioritizationPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-003'
            message = "required reclaim prioritization pattern not found: $required"
        }
    }
}

$requiredReclaimPrioritizationApplicationViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredReclaimPrioritizationApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-003'
            message = 'invalid requiredReclaimPrioritizationApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $cppFiles) {
        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
        if (-not [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, $pathRegex)) {
            continue
        }

        if (Select-String -Path $f.FullName -Pattern $lineRegex -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredReclaimPrioritizationApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-RECLAIM-003'
            message = "required reclaim prioritization application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$waitForDrainCallsiteViolations = @()
foreach ($f in $cppFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    $hits = Select-String -Path $f.FullName -Pattern 'waitForDrain\s*\(' -Encoding UTF8

    foreach ($h in $hits) {
        $lineText = "$($h.Line)"
        $allowMatched = $false

        foreach ($allow in @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist)) {
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($allow.pathRegex)") -and
                [System.Text.RegularExpressions.Regex]::IsMatch($lineText, "$($allow.lineRegex)")) {
                $allowMatched = $true
                break
            }
        }

        if (-not $allowMatched) {
            $waitForDrainCallsiteViolations += [ordered]@{
                file    = $relativePath
                line    = $h.LineNumber
                snippet = $lineText.Trim()
                checkId = 'CI-SHUTDOWN-005'
                message = 'waitForDrain callsite is not allowlisted'
            }
        }
    }
}

$isFullyDrainedCallsiteViolations = @()
foreach ($f in $cppFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $f.FullName
    $hits = Select-String -Path $f.FullName -Pattern 'isFullyDrained\s*\(' -Encoding UTF8

    foreach ($h in $hits) {
        $lineText = "$($h.Line)"
        $allowMatched = $false

        foreach ($allow in @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist)) {
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($allow.pathRegex)") -and
                [System.Text.RegularExpressions.Regex]::IsMatch($lineText, "$($allow.lineRegex)")) {
                $allowMatched = $true
                break
            }
        }

        if (-not $allowMatched) {
            $isFullyDrainedCallsiteViolations += [ordered]@{
                file    = $relativePath
                line    = $h.LineNumber
                snippet = $lineText.Trim()
                checkId = 'CI-SHUTDOWN-006'
                message = 'isFullyDrained callsite is not allowlisted'
            }
        }
    }
}

$drainAuthorityPolicyCoverageViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications)) {
    $requiredPathRegex = "$($requiredApp.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($allow in @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist)) {
        $allowPathRegex = "$($allow.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($allowPathRegex)) {
            continue
        }
        if ($allowPathRegex -eq $requiredPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $drainAuthorityPolicyCoverageViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $requiredPathRegex
            checkId = 'CI-SHUTDOWN-007'
            message = 'requiredDrainAuthorityApplications.pathRegex is not covered by isFullyDrainedCallsiteAllowlist.pathRegex'
        }
    }
}

$boundedDrainWaitPolicyCoverageViolations = @()
foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications)) {
    $requiredPathRegex = "$($requiredApp.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($allow in @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist)) {
        $allowPathRegex = "$($allow.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($allowPathRegex)) {
            continue
        }
        if ($allowPathRegex -eq $requiredPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $boundedDrainWaitPolicyCoverageViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $requiredPathRegex
            checkId = 'CI-SHUTDOWN-008'
            message = 'requiredBoundedDrainWaitApplications.pathRegex is not covered by waitForDrainCallsiteAllowlist.pathRegex'
        }
    }
}

$drainAuthorityAllowlistOverreachViolations = @()
foreach ($allow in @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist)) {
    $allowPathRegex = "$($allow.pathRegex)"
    $allowLineRegex = "$($allow.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($allowPathRegex)) {
        continue
    }

    # Definition-site allowlist entries are authority declarations and do not require
    # mirrored requiredDrainAuthorityApplications entries.
    if (-not [string]::IsNullOrWhiteSpace($allowLineRegex) -and
        [System.Text.RegularExpressions.Regex]::IsMatch($allowLineRegex, '^\s*bool\s+.*::isFullyDrained\s*\\\(')) {
        continue
    }

    $covered = $false
    foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications)) {
        $requiredPathRegex = "$($requiredApp.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
            continue
        }
        if ($requiredPathRegex -eq $allowPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $drainAuthorityAllowlistOverreachViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $allowPathRegex
            checkId = 'CI-SHUTDOWN-009'
            message = 'isFullyDrainedCallsiteAllowlist.pathRegex has no matching requiredDrainAuthorityApplications.pathRegex'
        }
    }
}

$boundedDrainWaitAllowlistOverreachViolations = @()
foreach ($allow in @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist)) {
    $allowPathRegex = "$($allow.pathRegex)"
    $allowLineRegex = "$($allow.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($allowPathRegex)) {
        continue
    }

    # Definition-site allowlist entries are bounded-wait API declarations and do not require
    # mirrored requiredBoundedDrainWaitApplications entries.
    if (-not [string]::IsNullOrWhiteSpace($allowLineRegex) -and
        [System.Text.RegularExpressions.Regex]::IsMatch($allowLineRegex, '^\s*bool\s+AudioEngine::waitForDrain\s*\\\(')) {
        continue
    }

    $covered = $false
    foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications)) {
        $requiredPathRegex = "$($requiredApp.pathRegex)"
        if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
            continue
        }
        if ($requiredPathRegex -eq $allowPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        foreach ($requiredApp in @($policy.shutdownReclaimChecks.requiredBoundedReclaimApplications)) {
            $requiredPathRegex = "$($requiredApp.pathRegex)"
            if ([string]::IsNullOrWhiteSpace($requiredPathRegex)) {
                continue
            }
            if ($requiredPathRegex -eq $allowPathRegex) {
                $covered = $true
                break
            }
        }
    }

    if (-not $covered) {
        $boundedDrainWaitAllowlistOverreachViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $allowPathRegex
            checkId = 'CI-SHUTDOWN-010'
            message = 'waitForDrainCallsiteAllowlist.pathRegex has no matching requiredBoundedDrainWaitApplications.pathRegex'
        }
    }
}

$shutdownRequiredDuplicatePathViolations = @()

$requiredDrainPathCounts = @{}
foreach ($entry in @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $key = "$pathRegex || $lineRegex"
    if (-not $requiredDrainPathCounts.ContainsKey($key)) {
        $requiredDrainPathCounts[$key] = 0
    }
    $requiredDrainPathCounts[$key]++
}
foreach ($key in $requiredDrainPathCounts.Keys) {
    if ($requiredDrainPathCounts[$key] -gt 1) {
        $shutdownRequiredDuplicatePathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-SHUTDOWN-011'
            message = 'duplicate pathRegex+lineRegex found in requiredDrainAuthorityApplications'
        }
    }
}

$requiredDrainWaitPathCounts = @{}
foreach ($entry in @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $key = "$pathRegex || $lineRegex"
    if (-not $requiredDrainWaitPathCounts.ContainsKey($key)) {
        $requiredDrainWaitPathCounts[$key] = 0
    }
    $requiredDrainWaitPathCounts[$key]++
}
foreach ($key in $requiredDrainWaitPathCounts.Keys) {
    if ($requiredDrainWaitPathCounts[$key] -gt 1) {
        $shutdownRequiredDuplicatePathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-SHUTDOWN-011'
            message = 'duplicate pathRegex+lineRegex found in requiredBoundedDrainWaitApplications'
        }
    }
}

$shutdownAllowlistDuplicatePathViolations = @()

$isFullyDrainedAllowPathCounts = @{}
foreach ($entry in @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $key = "$pathRegex || $lineRegex"
    if (-not $isFullyDrainedAllowPathCounts.ContainsKey($key)) {
        $isFullyDrainedAllowPathCounts[$key] = 0
    }
    $isFullyDrainedAllowPathCounts[$key]++
}
foreach ($key in $isFullyDrainedAllowPathCounts.Keys) {
    if ($isFullyDrainedAllowPathCounts[$key] -gt 1) {
        $shutdownAllowlistDuplicatePathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-SHUTDOWN-012'
            message = 'duplicate pathRegex+lineRegex found in isFullyDrainedCallsiteAllowlist'
        }
    }
}

$waitForDrainAllowPathCounts = @{}
foreach ($entry in @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }
    $key = "$pathRegex || $lineRegex"
    if (-not $waitForDrainAllowPathCounts.ContainsKey($key)) {
        $waitForDrainAllowPathCounts[$key] = 0
    }
    $waitForDrainAllowPathCounts[$key]++
}
foreach ($key in $waitForDrainAllowPathCounts.Keys) {
    if ($waitForDrainAllowPathCounts[$key] -gt 1) {
        $shutdownAllowlistDuplicatePathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-SHUTDOWN-012'
            message = 'duplicate pathRegex+lineRegex found in waitForDrainCallsiteAllowlist'
        }
    }
}

$shutdownPolicyOrphanPathViolations = @()

$shutdownPolicyPathEntries = @()
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.allowlist)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.requiredBoundedReclaimApplications)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.requiredDrainAuthorityApplications)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.requiredBoundedDrainWaitApplications)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.waitForDrainCallsiteAllowlist)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.isFullyDrainedCallsiteAllowlist)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.requiredEmergencyReclaimApplications)
$shutdownPolicyPathEntries += @($policy.shutdownReclaimChecks.requiredReclaimPrioritizationApplications)

foreach ($entry in @($shutdownPolicyPathEntries)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) {
        continue
    }

    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $shutdownPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-SHUTDOWN-013'
            message = 'shutdown/reclaim policy pathRegex matches no source files'
        }
    }
}

$coordinatorShutdownTransitionViolations = @()
if ((Test-Path -LiteralPath $releaseResourcesPath) -and (Test-Path -LiteralPath $ctorDtorPath)) {
    $releaseText = Get-Content -LiteralPath $releaseResourcesPath -Raw -Encoding UTF8
    $ctorDtorText = Get-Content -LiteralPath $ctorDtorPath -Raw -Encoding UTF8

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($releaseText, 'runtimePublicationBridge_\.requestShutdown\(\);')) {
        $coordinatorShutdownTransitionViolations += [ordered]@{
            file    = 'src/audioengine/AudioEngine.Processing.ReleaseResources.cpp'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-014'
            message = 'releaseResources must request coordinator shutdown before bounded drain.'
        }
    }

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($releaseText, 'runtimePublicationBridge_\.markShutdownComplete\(\);')) {
        $coordinatorShutdownTransitionViolations += [ordered]@{
            file    = 'src/audioengine/AudioEngine.Processing.ReleaseResources.cpp'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-014'
            message = 'releaseResources must finalize coordinator shutdown after bounded drain evaluation.'
        }
    }

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($ctorDtorText, 'runtimePublicationBridge_\.requestShutdown\(\);')) {
        $coordinatorShutdownTransitionViolations += [ordered]@{
            file    = 'src/audioengine/AudioEngine.CtorDtor.cpp'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-015'
            message = 'AudioEngine destructor must request coordinator shutdown in direct teardown path.'
        }
    }

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($ctorDtorText, 'runtimePublicationBridge_\.markShutdownComplete\(\);')) {
        $coordinatorShutdownTransitionViolations += [ordered]@{
            file    = 'src/audioengine/AudioEngine.CtorDtor.cpp'
            line    = 0
            snippet = ''
            checkId = 'CI-SHUTDOWN-015'
            message = 'AudioEngine destructor must finalize coordinator shutdown after forced drain.'
        }
    }
}
else {
    $coordinatorShutdownTransitionViolations += [ordered]@{
        file    = 'src/audioengine/**'
        line    = 0
        snippet = ''
        checkId = 'CI-SHUTDOWN-014'
        message = 'required shutdown path source files were not found for coordinator transition checks.'
    }
}

$allViolations = @($violations + $functionScopeViolations + $requiredPatternViolations + $requiredApplicationViolations + $requiredDrainPatternViolations + $requiredDrainApplicationViolations + $requiredDrainWaitPatternViolations + $requiredDrainWaitApplicationViolations + $requiredEmergencyReclaimPatternViolations + $requiredEmergencyReclaimApplicationViolations + $requiredReclaimPrioritizationPatternViolations + $requiredReclaimPrioritizationApplicationViolations + $waitForDrainCallsiteViolations + $isFullyDrainedCallsiteViolations + $drainAuthorityPolicyCoverageViolations + $boundedDrainWaitPolicyCoverageViolations + $drainAuthorityAllowlistOverreachViolations + $boundedDrainWaitAllowlistOverreachViolations + $shutdownRequiredDuplicatePathViolations + $shutdownAllowlistDuplicatePathViolations + $shutdownPolicyOrphanPathViolations + $coordinatorShutdownTransitionViolations)
$report = [ordered]@{
    schema         = 'isr_v73_shutdown_reclaim_report_v1'
    generatedAt    = (Get-Date -Format 'o')
    mode           = $effectiveMode
    checks         = @('CI-SHUTDOWN-001', 'CI-SHUTDOWN-002', 'CI-SHUTDOWN-003', 'CI-SHUTDOWN-004', 'CI-SHUTDOWN-005', 'CI-SHUTDOWN-006', 'CI-SHUTDOWN-007', 'CI-SHUTDOWN-008', 'CI-SHUTDOWN-009', 'CI-SHUTDOWN-010', 'CI-SHUTDOWN-011', 'CI-SHUTDOWN-012', 'CI-SHUTDOWN-013', 'CI-SHUTDOWN-014', 'CI-SHUTDOWN-015', 'CI-RECLAIM-001', 'CI-RECLAIM-002', 'CI-RECLAIM-003')
    violationCount = $allViolations.Count
    violations     = $allViolations
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($allViolations.Count -gt 0) {
    foreach ($v in $allViolations) {
        Write-Host "[WARN] [$($v.checkId)] $($v.file):$($v.line) $($v.message)"
    }

    if ($effectiveMode -eq 'fail') {
        throw "ISR v7.3 shutdown/reclaim checks failed. violations=$($allViolations.Count)"
    }

    Write-Host "[WARN] ISR v7.3 shutdown/reclaim checks completed in warn mode. violations=$($allViolations.Count)"
    exit 0
}

Write-Host '[PASS] ISR v7.3 shutdown/reclaim checks passed'
exit 0
