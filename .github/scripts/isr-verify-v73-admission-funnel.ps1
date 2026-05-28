param(
    [ValidateSet('warn', 'fail')]
    [string]$Mode = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyPath = Join-Path $repoRoot '.github\isr-ai-governance-policy.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_admission_report.json'

if (-not (Test-Path -LiteralPath $policyPath)) {
    throw "Missing policy file: $policyPath"
}

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

Assert-AllowlistContract -Entries @($policy.requestRebuildDirectCall.allowlist) -Context 'requestRebuildDirectCall.allowlist' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications) -Context 'requestRebuildDirectCall.requiredAdmissionGateApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.requestRebuildDirectCall.requiredCollapseSafetyApplications) -Context 'requestRebuildDirectCall.requiredCollapseSafetyApplications' -RequiredFields @('pathRegex', 'lineRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.suppressionAuthorityFunctions) -Context 'suppressionAuthorityFunctions' -RequiredFields @('pathRegex', 'functionNameContains', 'domain', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.suppressionDomainTags) -Context 'suppressionDomainTags' -RequiredFields @('pathRegex', 'lineRegex', 'domain', 'owner', 'issue', 'rationale', 'expiry')
Assert-AllowlistContract -Entries @($policy.forbiddenExecutionSemanticExclusions) -Context 'forbiddenExecutionSemanticExclusions' -RequiredFields @('pathRegex', 'owner', 'issue', 'rationale', 'expiry')
Assert-RegexFields -Entries @($policy.requestRebuildDirectCall.allowlist) -Context 'requestRebuildDirectCall.allowlist' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications) -Context 'requestRebuildDirectCall.requiredAdmissionGateApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.requestRebuildDirectCall.requiredCollapseSafetyApplications) -Context 'requestRebuildDirectCall.requiredCollapseSafetyApplications' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.suppressionAuthorityFunctions) -Context 'suppressionAuthorityFunctions' -Fields @('pathRegex')
Assert-RegexFields -Entries @($policy.suppressionDomainTags) -Context 'suppressionDomainTags' -Fields @('pathRegex', 'lineRegex')
Assert-RegexFields -Entries @($policy.forbiddenExecutionSemanticExclusions) -Context 'forbiddenExecutionSemanticExclusions' -Fields @('pathRegex', 'lineRegex')

foreach ($requiredPattern in @($policy.requestRebuildDirectCall.requiredAdmissionGatePatterns)) {
    Assert-RegexCompiles -Pattern "$requiredPattern" -Context 'requestRebuildDirectCall.requiredAdmissionGatePatterns[]'
}
foreach ($requiredPattern in @($policy.requestRebuildDirectCall.requiredCollapseSafetyPatterns)) {
    Assert-RegexCompiles -Pattern "$requiredPattern" -Context 'requestRebuildDirectCall.requiredCollapseSafetyPatterns[]'
}

$validSuppressionDomains = @('RebuildIntentSuppression', 'QueueAdmissionSuppression', 'SnapshotDropSuppression')
foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $domain = "$($entry.domain)"
    if ($validSuppressionDomains -notcontains $domain) {
        throw "suppressionAuthorityFunctions has invalid domain: $domain"
    }
}
foreach ($tag in @($policy.suppressionDomainTags)) {
    $pathRegex = "$($tag.pathRegex)"
    $lineRegex = "$($tag.lineRegex)"
    $domain = "$($tag.domain)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex) -or [string]::IsNullOrWhiteSpace($domain)) {
        throw 'suppressionDomainTags entry requires pathRegex/lineRegex/domain'
    }
    if ($validSuppressionDomains -notcontains $domain) {
        throw "suppressionDomainTags has invalid domain: $domain"
    }
}

$tagDomainSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
foreach ($tag in @($policy.suppressionDomainTags)) {
    $null = $tagDomainSet.Add("$($tag.domain)")
}

if (-not $tagDomainSet.Contains('RebuildIntentSuppression')) {
    throw 'suppressionDomainTags must contain RebuildIntentSuppression domain'
}

foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $entryDomain = "$($entry.domain)"
    if (-not $tagDomainSet.Contains($entryDomain)) {
        throw "suppressionAuthorityFunctions domain is not declared in suppressionDomainTags: $entryDomain"
    }
}

if ($null -eq $policy.suppressionReasonAllowlist -or @($policy.suppressionReasonAllowlist).Count -eq 0) {
    throw 'Policy missing suppressionReasonAllowlist entries'
}

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)

    if ($targetFull.StartsWith($baseFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $targetFull.Substring($baseFull.Length).TrimStart([char[]]@([char]'\', [char]'/')).Replace('\', '/')
    }

    return $targetFull.Replace('\', '/')
}

function Test-PolicyAllowlistEntry {
    param(
        [Parameter(Mandatory = $true)]$Entry,
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$Line
    )

    $pathRegex = "$($Entry.pathRegex)"
    $lineRegex = "$($Entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        return $false
    }

    if (-not [System.Text.RegularExpressions.Regex]::IsMatch($RelativePath, $pathRegex)) {
        return $false
    }

    return [System.Text.RegularExpressions.Regex]::IsMatch($Line, $lineRegex)
}

function Resolve-SuppressionDomain {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$LineText,
        [Parameter(Mandatory = $true)]$Tags
    )

    foreach ($tag in @($Tags)) {
        if ([System.Text.RegularExpressions.Regex]::IsMatch($RelativePath, "$($tag.pathRegex)") -and
            [System.Text.RegularExpressions.Regex]::IsMatch($LineText, "$($tag.lineRegex)")) {
            return "$($tag.domain)"
        }
    }

    return ''
}

$effectiveMode = $Mode
if ([string]::IsNullOrWhiteSpace($effectiveMode)) {
    $effectiveMode = "$($policy.requestRebuildDirectCall.phase)"
    if ([string]::IsNullOrWhiteSpace($effectiveMode)) {
        $effectiveMode = 'warn'
    }
}

if (@('warn', 'fail') -notcontains $effectiveMode) {
    throw "Invalid requestRebuildDirectCall.phase: $effectiveMode (expected warn|fail)"
}

$searchPatterns = @('*.cpp', '*.cc', '*.cxx', '*.h', '*.hpp')
$sourceRoot = Join-Path $repoRoot 'src'
$sources = foreach ($p in $searchPatterns) { Get-ChildItem -Path $sourceRoot -Recurse -File -Filter $p }

$sourceRelativePaths = @()
foreach ($srcFile in @($sources)) {
    $sourceRelativePaths += Get-RelativePathCompat -BasePath $repoRoot -TargetPath $srcFile.FullName
}

$policyScopeRelativePaths = @()
$policyScopeRoots = @(
    (Join-Path $repoRoot 'src'),
    (Join-Path $repoRoot '.github')
)
$policyScopeExts = @('.cpp', '.cc', '.cxx', '.h', '.hpp', '.ps1', '.json', '.yml', '.yaml')
foreach ($scopeRoot in $policyScopeRoots) {
    if (-not (Test-Path -LiteralPath $scopeRoot)) {
        continue
    }

    $scopeFiles = Get-ChildItem -Path $scopeRoot -Recurse -File | Where-Object {
        $policyScopeExts -contains $_.Extension.ToLowerInvariant()
    }

    foreach ($scopeFile in @($scopeFiles)) {
        $policyScopeRelativePaths += Get-RelativePathCompat -BasePath $repoRoot -TargetPath $scopeFile.FullName
    }
}

$requestRebuildFindings = @()
foreach ($file in $sources) {
    foreach ($m in (Select-String -Path $file.FullName -Pattern '\brequestRebuild\s*\(' -Encoding UTF8)) {
        $line = "$($m.Line)"
        $lineForCheck = $line
        if ($lineForCheck.Contains('//')) {
            $lineForCheck = $lineForCheck.Substring(0, $lineForCheck.IndexOf('//'))
        }
        if ([string]::IsNullOrWhiteSpace($lineForCheck)) {
            continue
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($line, '\bvoid\s+AudioEngine::requestRebuild\s*\(') -or
            [System.Text.RegularExpressions.Regex]::IsMatch($line, '\brequestRebuild\s*\([^\)]*\)\s*(noexcept)?\s*\{')) {
            continue
        }

        if ([System.Text.RegularExpressions.Regex]::IsMatch($lineForCheck, '^\s*(?:[\w:<>~]+\s+)+requestRebuild\s*\([^\)]*\)\s*(?:const\s*)?(?:noexcept\s*)?;\s*$')) {
            continue
        }

        $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $m.Path
        $isAllowed = $false
        foreach ($entry in @($policy.requestRebuildDirectCall.allowlist)) {
            if (Test-PolicyAllowlistEntry -Entry $entry -RelativePath $relativePath -Line $lineForCheck) {
                $isAllowed = $true
                break
            }
        }

        if (-not $isAllowed) {
            $requestRebuildFindings += [ordered]@{
                file    = $relativePath
                line    = $m.LineNumber
                snippet = $line.Trim()
                checkId = 'CI-ADMISSION-001'
                message = 'allowlist外 requestRebuild direct call'
            }
        }
    }
}

$forbiddenSemanticViolations = @()
$forbiddenSemanticsRegex = '\b(?:' + (($policy.forbiddenExecutionSemantics | ForEach-Object { [System.Text.RegularExpressions.Regex]::Escape("$_") }) -join '|') + ')\b'
if (-not [string]::IsNullOrWhiteSpace($forbiddenSemanticsRegex)) {
    $targetRoots = @(
        (Join-Path $repoRoot 'src'),
        (Join-Path $repoRoot '.github')
    )
    $allowedExtensions = @('.cpp', '.cc', '.cxx', '.h', '.hpp', '.ps1', '.json', '.yml', '.yaml')

    foreach ($root in $targetRoots) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $files = Get-ChildItem -Path $root -Recurse -File | Where-Object {
            $allowedExtensions -contains $_.Extension.ToLowerInvariant()
        }
        foreach ($f in $files) {
            $hits = Select-String -Path $f.FullName -Pattern $forbiddenSemanticsRegex -AllMatches -CaseSensitive -Encoding UTF8
            foreach ($h in $hits) {
                $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $h.Path

                $excluded = $false
                foreach ($exclude in @($policy.forbiddenExecutionSemanticExclusions)) {
                    $pathOk = [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($exclude.pathRegex)")
                    $lineRegex = "$($exclude.lineRegex)"
                    $lineOk = [string]::IsNullOrWhiteSpace($lineRegex) -or [System.Text.RegularExpressions.Regex]::IsMatch("$($h.Line)", $lineRegex)
                    if ($pathOk -and $lineOk) {
                        $excluded = $true
                        break
                    }
                }

                if ($excluded) {
                    continue
                }

                $forbiddenSemanticViolations += [ordered]@{
                    file    = $relativePath
                    line    = $h.LineNumber
                    snippet = "$($h.Line)".Trim()
                    checkId = 'CI-ADMISSION-002'
                    message = 'forbidden execution semantic token detected'
                }
            }
        }
    }
}

$suppressionViolations = @()
$suppressionScanFiles = Get-ChildItem -Path $sourceRoot -Recurse -File -Filter '*.cpp'
$suppressedHits = foreach ($scanFile in $suppressionScanFiles) {
    Select-String -Path $scanFile.FullName -Pattern 'RebuildTelemetryDecision::Suppressed' -Encoding UTF8
}
foreach ($hit in $suppressedHits) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $hit.Path
    $lineText = "$($hit.Line)"

    $domain = Resolve-SuppressionDomain -RelativePath $relativePath -LineText $lineText -Tags $policy.suppressionDomainTags

    if ([string]::IsNullOrWhiteSpace($domain)) {
        $suppressionViolations += [ordered]@{
            file    = $relativePath
            line    = $hit.LineNumber
            snippet = $lineText.Trim()
            checkId = 'CI-ADMISSION-003'
            message = 'suppression domain tag missing'
        }
        continue
    }

    if ($domain -eq 'RebuildIntentSuppression') {
        $allowed = $false
        foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
            if ([System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($entry.pathRegex)") -and
                "$($entry.domain)" -eq 'RebuildIntentSuppression') {
                $allowed = $true
                break
            }
        }

        if (-not $allowed) {
            $suppressionViolations += [ordered]@{
                file    = $relativePath
                line    = $hit.LineNumber
                snippet = $lineText.Trim()
                checkId = 'CI-ADMISSION-003'
                message = 'RebuildIntentSuppression outside authority allowlist'
            }
        }
    }
}

$suppressionReasonViolations = @()
$suppressionReasonAllow = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
foreach ($entry in @($policy.suppressionReasonAllowlist)) {
    $name = "$entry"
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        [void]$suppressionReasonAllow.Add($name)
    }
}

foreach ($scanFile in $suppressionScanFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $scanFile.FullName
    $text = Get-Content -LiteralPath $scanFile.FullName -Raw -Encoding UTF8
    $pattern = 'emitRebuildTelemetry\(\s*RebuildTelemetryEvent::\w+\s*,\s*[^,]+\s*,\s*RebuildTelemetryReason::(?<reason>\w+)\s*,\s*RebuildTelemetryDecision::Suppressed'
    $reasonMatches = [System.Text.RegularExpressions.Regex]::Matches($text, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    foreach ($m in $reasonMatches) {
        $reason = $m.Groups['reason'].Value
        if (-not $suppressionReasonAllow.Contains($reason)) {
            $lineNumber = 1
            if ($m.Index -gt 0) {
                $lineNumber = 1 + ([System.Text.RegularExpressions.Regex]::Matches($text.Substring(0, $m.Index), "`n")).Count
            }

            $suppressionReasonViolations += [ordered]@{
                file    = $relativePath
                line    = $lineNumber
                snippet = "RebuildTelemetryReason::$reason"
                checkId = 'CI-ADMISSION-004'
                message = 'suppressed reason is not in suppressionReasonAllowlist'
            }
        }
    }
}

$suppressionAuthorityFunctionViolations = @()
foreach ($scanFile in $suppressionScanFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $scanFile.FullName
    $text = Get-Content -LiteralPath $scanFile.FullName -Raw -Encoding UTF8

    $suppressedMatches = [System.Text.RegularExpressions.Regex]::Matches($text,
        'emitRebuildTelemetry\(\s*RebuildTelemetryEvent::\w+\s*,\s*[^,]+\s*,\s*RebuildTelemetryReason::(?<reason>\w+)\s*,\s*RebuildTelemetryDecision::Suppressed',
        [System.Text.RegularExpressions.RegexOptions]::Singleline)

    foreach ($sm in $suppressedMatches) {
        $idx = $sm.Index
        $lineText = "$($sm.Value)"
        $functionHeaderMatches = [System.Text.RegularExpressions.Regex]::Matches($text.Substring(0, $idx),
            '([\w:<>~]+\s+)?AudioEngine::(?<fname>\w+)\s*\(',
            [System.Text.RegularExpressions.RegexOptions]::Singleline)

        $enclosingFunction = ''
        if ($functionHeaderMatches.Count -gt 0) {
            $enclosingFunction = $functionHeaderMatches[$functionHeaderMatches.Count - 1].Groups['fname'].Value
        }

        $domain = Resolve-SuppressionDomain -RelativePath $relativePath -LineText $lineText -Tags $policy.suppressionDomainTags

        $authorityMatched = $false
        foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
            $pathOk = [System.Text.RegularExpressions.Regex]::IsMatch($relativePath, "$($entry.pathRegex)")
            $nameNeedle = "$($entry.functionNameContains)"
            $nameOk = (-not [string]::IsNullOrWhiteSpace($nameNeedle)) -and ($enclosingFunction -like "*$nameNeedle*")
            $domainOk = "$($entry.domain)" -eq $domain
            if ($pathOk -and $nameOk -and $domainOk) {
                $authorityMatched = $true
                break
            }
        }

        if (-not $authorityMatched) {
            $lineNumber = 1
            if ($idx -gt 0) {
                $lineNumber = 1 + ([System.Text.RegularExpressions.Regex]::Matches($text.Substring(0, $idx), "`n")).Count
            }

            $suppressionAuthorityFunctionViolations += [ordered]@{
                file    = $relativePath
                line    = $lineNumber
                snippet = "Suppressed telemetry in function '$enclosingFunction'"
                checkId = 'CI-ADMISSION-005'
                message = "suppression authority mismatch (function/domain). domain='$domain'"
            }
        }
    }
}

$requiredAdmissionGatePatternViolations = @()
foreach ($required in @($policy.requestRebuildDirectCall.requiredAdmissionGatePatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $sources) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredAdmissionGatePatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-006'
            message = "required admission gate pattern not found: $required"
        }
    }
}

$requiredAdmissionGateApplicationViolations = @()
foreach ($requiredApp in @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredAdmissionGateApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-006'
            message = 'invalid requiredAdmissionGateApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $sources) {
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
        $requiredAdmissionGateApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-006'
            message = "required admission gate application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$requiredCollapseSafetyPatternViolations = @()
foreach ($required in @($policy.requestRebuildDirectCall.requiredCollapseSafetyPatterns)) {
    if ([string]::IsNullOrWhiteSpace("$required")) { continue }

    $matched = $false
    foreach ($f in $sources) {
        if (Select-String -Path $f.FullName -Pattern "$required" -Quiet -Encoding UTF8) {
            $matched = $true
            break
        }
    }

    if (-not $matched) {
        $requiredCollapseSafetyPatternViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-007'
            message = "required collapse safety pattern not found: $required"
        }
    }
}

$requiredCollapseSafetyApplicationViolations = @()
foreach ($requiredApp in @($policy.requestRebuildDirectCall.requiredCollapseSafetyApplications)) {
    $pathRegex = "$($requiredApp.pathRegex)"
    $lineRegex = "$($requiredApp.lineRegex)"

    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        $requiredCollapseSafetyApplicationViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-007'
            message = 'invalid requiredCollapseSafetyApplications entry (pathRegex/lineRegex required)'
        }
        continue
    }

    $matched = $false
    foreach ($f in $sources) {
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
        $requiredCollapseSafetyApplicationViolations += [ordered]@{
            file    = 'src/**'
            line    = 0
            snippet = ''
            checkId = 'CI-ADMISSION-007'
            message = "required collapse safety application not found: pathRegex=$pathRegex lineRegex=$lineRegex"
        }
    }
}

$suppressionDomainCoverageViolations = @()
foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $entryPathRegex = "$($entry.pathRegex)"
    $entryDomain = "$($entry.domain)"
    if ([string]::IsNullOrWhiteSpace($entryPathRegex) -or [string]::IsNullOrWhiteSpace($entryDomain)) {
        continue
    }

    $covered = $false
    foreach ($tag in @($policy.suppressionDomainTags)) {
        $tagPathRegex = "$($tag.pathRegex)"
        $tagDomain = "$($tag.domain)"
        if ([string]::IsNullOrWhiteSpace($tagPathRegex) -or [string]::IsNullOrWhiteSpace($tagDomain)) {
            continue
        }

        if ($tagDomain -eq $entryDomain -and $tagPathRegex -eq $entryPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $suppressionDomainCoverageViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = "$entryDomain :: $entryPathRegex"
            checkId = 'CI-ADMISSION-008'
            message = 'suppressionAuthorityFunctions entry is not covered by suppressionDomainTags (domain/pathRegex)'
        }
    }
}

$suppressionDomainOverreachViolations = @()
$authorityDomainSet = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::Ordinal)
foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $domain = "$($entry.domain)"
    if (-not [string]::IsNullOrWhiteSpace($domain)) {
        [void]$authorityDomainSet.Add($domain)
    }
}

foreach ($tag in @($policy.suppressionDomainTags)) {
    $tagPathRegex = "$($tag.pathRegex)"
    $tagDomain = "$($tag.domain)"
    if ([string]::IsNullOrWhiteSpace($tagPathRegex) -or [string]::IsNullOrWhiteSpace($tagDomain)) {
        continue
    }

    if (-not $authorityDomainSet.Contains($tagDomain)) {
        continue
    }

    $covered = $false
    foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
        $entryPathRegex = "$($entry.pathRegex)"
        $entryDomain = "$($entry.domain)"
        if ([string]::IsNullOrWhiteSpace($entryPathRegex) -or [string]::IsNullOrWhiteSpace($entryDomain)) {
            continue
        }

        if ($entryDomain -eq $tagDomain -and $entryPathRegex -eq $tagPathRegex) {
            $covered = $true
            break
        }
    }

    if (-not $covered) {
        $suppressionDomainOverreachViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = "$tagDomain :: $tagPathRegex"
            checkId = 'CI-ADMISSION-009'
            message = 'suppressionDomainTags entry is not covered by suppressionAuthorityFunctions (domain/pathRegex)'
        }
    }
}

$directCallAllowlistCoverageViolations = @()
foreach ($allow in @($policy.requestRebuildDirectCall.allowlist)) {
    $allowPathRegex = "$($allow.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($allowPathRegex)) {
        continue
    }

    $covered = $false
    foreach ($requiredApp in @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications)) {
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
        $directCallAllowlistCoverageViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $allowPathRegex
            checkId = 'CI-ADMISSION-010'
            message = 'requestRebuildDirectCall.allowlist.pathRegex is not covered by requiredAdmissionGateApplications.pathRegex'
        }
    }
}

$admissionPolicyDuplicateViolations = @()

$suppressionAuthorityKeyCounts = @{}
foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $pathRegex = "$($entry.pathRegex)"
    $functionNameContains = "$($entry.functionNameContains)"
    $domain = "$($entry.domain)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($functionNameContains) -or [string]::IsNullOrWhiteSpace($domain)) {
        continue
    }

    $key = "$pathRegex || $functionNameContains || $domain"
    if (-not $suppressionAuthorityKeyCounts.ContainsKey($key)) {
        $suppressionAuthorityKeyCounts[$key] = 0
    }
    $suppressionAuthorityKeyCounts[$key]++
}
foreach ($key in $suppressionAuthorityKeyCounts.Keys) {
    if ($suppressionAuthorityKeyCounts[$key] -gt 1) {
        $admissionPolicyDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-011'
            message = 'duplicate pathRegex+functionNameContains+domain found in suppressionAuthorityFunctions'
        }
    }
}

$suppressionDomainTagKeyCounts = @{}
foreach ($entry in @($policy.suppressionDomainTags)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    $domain = "$($entry.domain)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex) -or [string]::IsNullOrWhiteSpace($domain)) {
        continue
    }

    $key = "$pathRegex || $lineRegex || $domain"
    if (-not $suppressionDomainTagKeyCounts.ContainsKey($key)) {
        $suppressionDomainTagKeyCounts[$key] = 0
    }
    $suppressionDomainTagKeyCounts[$key]++
}
foreach ($key in $suppressionDomainTagKeyCounts.Keys) {
    if ($suppressionDomainTagKeyCounts[$key] -gt 1) {
        $admissionPolicyDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-011'
            message = 'duplicate pathRegex+lineRegex+domain found in suppressionDomainTags'
        }
    }
}

$admissionApplicationDuplicateViolations = @()

$directAllowlistKeyCounts = @{}
foreach ($entry in @($policy.requestRebuildDirectCall.allowlist)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $key = "$pathRegex || $lineRegex"
    if (-not $directAllowlistKeyCounts.ContainsKey($key)) {
        $directAllowlistKeyCounts[$key] = 0
    }
    $directAllowlistKeyCounts[$key]++
}
foreach ($key in $directAllowlistKeyCounts.Keys) {
    if ($directAllowlistKeyCounts[$key] -gt 1) {
        $admissionApplicationDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-012'
            message = 'duplicate pathRegex+lineRegex found in requestRebuildDirectCall.allowlist'
        }
    }
}

$requiredAdmissionGateKeyCounts = @{}
foreach ($entry in @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $key = "$pathRegex || $lineRegex"
    if (-not $requiredAdmissionGateKeyCounts.ContainsKey($key)) {
        $requiredAdmissionGateKeyCounts[$key] = 0
    }
    $requiredAdmissionGateKeyCounts[$key]++
}
foreach ($key in $requiredAdmissionGateKeyCounts.Keys) {
    if ($requiredAdmissionGateKeyCounts[$key] -gt 1) {
        $admissionApplicationDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-012'
            message = 'duplicate pathRegex+lineRegex found in requiredAdmissionGateApplications'
        }
    }
}

$requiredCollapseSafetyKeyCounts = @{}
foreach ($entry in @($policy.requestRebuildDirectCall.requiredCollapseSafetyApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    $lineRegex = "$($entry.lineRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex) -or [string]::IsNullOrWhiteSpace($lineRegex)) {
        continue
    }

    $key = "$pathRegex || $lineRegex"
    if (-not $requiredCollapseSafetyKeyCounts.ContainsKey($key)) {
        $requiredCollapseSafetyKeyCounts[$key] = 0
    }
    $requiredCollapseSafetyKeyCounts[$key]++
}
foreach ($key in $requiredCollapseSafetyKeyCounts.Keys) {
    if ($requiredCollapseSafetyKeyCounts[$key] -gt 1) {
        $admissionApplicationDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-012'
            message = 'duplicate pathRegex+lineRegex found in requiredCollapseSafetyApplications'
        }
    }
}

$admissionPolicyArrayDuplicateViolations = @()

$suppressionReasonCounts = @{}
foreach ($reason in @($policy.suppressionReasonAllowlist)) {
    $reasonName = "$reason"
    if ([string]::IsNullOrWhiteSpace($reasonName)) {
        continue
    }

    if (-not $suppressionReasonCounts.ContainsKey($reasonName)) {
        $suppressionReasonCounts[$reasonName] = 0
    }
    $suppressionReasonCounts[$reasonName]++
}
foreach ($key in $suppressionReasonCounts.Keys) {
    if ($suppressionReasonCounts[$key] -gt 1) {
        $admissionPolicyArrayDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-013'
            message = 'duplicate token found in suppressionReasonAllowlist'
        }
    }
}

$forbiddenSemanticCounts = @{}
foreach ($token in @($policy.forbiddenExecutionSemantics)) {
    $tokenName = "$token"
    if ([string]::IsNullOrWhiteSpace($tokenName)) {
        continue
    }

    if (-not $forbiddenSemanticCounts.ContainsKey($tokenName)) {
        $forbiddenSemanticCounts[$tokenName] = 0
    }
    $forbiddenSemanticCounts[$tokenName]++
}
foreach ($key in $forbiddenSemanticCounts.Keys) {
    if ($forbiddenSemanticCounts[$key] -gt 1) {
        $admissionPolicyArrayDuplicateViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $key
            checkId = 'CI-ADMISSION-013'
            message = 'duplicate token found in forbiddenExecutionSemantics'
        }
    }
}

$admissionPolicyOrphanPathViolations = @()

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

foreach ($entry in @($policy.requestRebuildDirectCall.allowlist)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'requestRebuildDirectCall.allowlist.pathRegex matches no source files'
        }
    }
}

foreach ($entry in @($policy.requestRebuildDirectCall.requiredAdmissionGateApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'requiredAdmissionGateApplications.pathRegex matches no source files'
        }
    }
}

foreach ($entry in @($policy.requestRebuildDirectCall.requiredCollapseSafetyApplications)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'requiredCollapseSafetyApplications.pathRegex matches no source files'
        }
    }
}

foreach ($entry in @($policy.suppressionAuthorityFunctions)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'suppressionAuthorityFunctions.pathRegex matches no source files'
        }
    }
}

foreach ($entry in @($policy.suppressionDomainTags)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $sourceRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'suppressionDomainTags.pathRegex matches no source files'
        }
    }
}

foreach ($entry in @($policy.forbiddenExecutionSemanticExclusions)) {
    $pathRegex = "$($entry.pathRegex)"
    if ([string]::IsNullOrWhiteSpace($pathRegex)) { continue }
    if (-not (Test-RegexMatchesAnyPath -PathRegex $pathRegex -RelativePaths $policyScopeRelativePaths)) {
        $admissionPolicyOrphanPathViolations += [ordered]@{
            file    = 'policy'
            line    = 0
            snippet = $pathRegex
            checkId = 'CI-ADMISSION-014'
            message = 'forbiddenExecutionSemanticExclusions.pathRegex matches no policy scope files'
        }
    }
}

$violations = @($requestRebuildFindings + $forbiddenSemanticViolations + $suppressionViolations + $suppressionReasonViolations + $suppressionAuthorityFunctionViolations + $suppressionDomainCoverageViolations + $suppressionDomainOverreachViolations + $directCallAllowlistCoverageViolations + $admissionPolicyDuplicateViolations + $admissionApplicationDuplicateViolations + $admissionPolicyArrayDuplicateViolations + $admissionPolicyOrphanPathViolations + $requiredAdmissionGatePatternViolations + $requiredAdmissionGateApplicationViolations + $requiredCollapseSafetyPatternViolations + $requiredCollapseSafetyApplicationViolations)
$report = [ordered]@{
    schema         = 'isr_v73_admission_report_v1'
    generatedAt    = (Get-Date -Format 'o')
    mode           = $effectiveMode
    checks         = @('CI-ADMISSION-001', 'CI-ADMISSION-002', 'CI-ADMISSION-003', 'CI-ADMISSION-004', 'CI-ADMISSION-005', 'CI-ADMISSION-006', 'CI-ADMISSION-007', 'CI-ADMISSION-008', 'CI-ADMISSION-009', 'CI-ADMISSION-010', 'CI-ADMISSION-011', 'CI-ADMISSION-012', 'CI-ADMISSION-013', 'CI-ADMISSION-014')
    violationCount = $violations.Count
    violations     = $violations
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[WARN] [$($v.checkId)] $($v.file):$($v.line) $($v.message)"
    }

    if ($effectiveMode -eq 'fail') {
        throw "ISR v7.3 admission checks failed. violations=$($violations.Count)"
    }

    Write-Host "[WARN] ISR v7.3 admission checks completed in warn mode. violations=$($violations.Count)"
    exit 0
}

Write-Host '[PASS] ISR v7.3 admission checks passed'
exit 0
