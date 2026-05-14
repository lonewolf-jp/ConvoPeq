Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$targetDir = Join-Path $repoRoot "src\audioengine"

if (-not (Test-Path $targetDir)) {
    Write-Error "Target directory not found: $targetDir"
    exit 2
}

$rules = @(
    @{
        Id          = "LINT-AE-001"
        Description = "No zero-arg getRuntimeGraphState()/getEngineRuntimeState() in src/audioengine/*.cpp"
        Pattern     = "getRuntimeGraphState\(\)|getEngineRuntimeState\(\)"
    },
    @{
        Id          = "LINT-AE-002"
        Description = "No single-arg resolveCurrent/resolveFading DSP runtime publish calls in src/audioengine/*.cpp"
        Pattern     = "resolveCurrentDSPFromRuntimePublish\(\s*[^,\)]*\)|resolveFadingDSPFromRuntimePublish\(\s*[^,\)]*\)"
    },
    @{
        Id          = "LINT-AE-003"
        Description = "No direct crossfade atomic loads in src/audioengine/*.cpp"
        Pattern     = "dspCrossfadePending\.load\(|dspCrossfadeUseDryAsOld\.load\(|dspCrossfadeStartDelayBlocks\.load\("
    }
)

$warningRule = @{
    Id          = "LINT-AE-004"
    Description = "Warn when getRuntimePublishWorld() is called 3+ times in the same src/audioengine/*.cpp function"
    Pattern     = "getRuntimePublishWorld\(\)"
}

function Test-IsLikelyFunctionSignature {
    param(
        [string]$Line
    )

    if ([string]::IsNullOrWhiteSpace($Line)) { return $false }

    $trimmed = $Line.Trim()
    if ($trimmed.StartsWith("//") -or $trimmed.StartsWith("/*") -or $trimmed.StartsWith("*") -or $trimmed.StartsWith("#")) { return $false }
    if ($trimmed -match '^(if|for|while|switch|catch|else|do|namespace|class|struct|enum)\b') { return $false }
    if ($trimmed.EndsWith(";")) { return $false }

    return ($trimmed -match '\)\s*(const\b|noexcept\b|override\b|final\b|\{|$)')
}

function Get-FunctionScopedWorldWarnings {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Content,
        [Parameter(Mandatory = $true)]
        [string]$RelativePath,
        [Parameter(Mandatory = $true)]
        [string]$Pattern,
        [Parameter(Mandatory = $true)]
        [string]$RuleId,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    $results = @()
    if ([string]::IsNullOrEmpty($Content)) {
        return $results
    }

    $Lines = @($Content -split "`r?`n")
    if ($Lines.Count -eq 0) {
        return $results
    }

    $braceDepth = 0
    $pendingSignatureLine = $null
    $currentFunction = $null

    for ($index = 0; $index -lt $Lines.Length; $index++) {
        $lineNumber = $index + 1
        $line = $Lines[$index]
        $trimmed = $line.Trim()

        if ($braceDepth -eq 0 -and $currentFunction -eq $null -and (Test-IsLikelyFunctionSignature -Line $line)) {
            $pendingSignatureLine = $lineNumber
        }

        if ($pendingSignatureLine -ne $null -and $currentFunction -eq $null -and $braceDepth -eq 0 -and $line.Contains("{")) {
            $currentFunction = [PSCustomObject]@{
                StartLine = $pendingSignatureLine
                StartText = ($Lines[$pendingSignatureLine - 1]).Trim()
                Count     = 0
            }
            $pendingSignatureLine = $null
        }

        if ($currentFunction -ne $null) {
            $matches = [System.Text.RegularExpressions.Regex]::Matches(
                $line,
                $Pattern,
                [System.Text.RegularExpressions.RegexOptions]::None
            )
            $currentFunction.Count += $matches.Count
        }

        foreach ($ch in $line.ToCharArray()) {
            switch ($ch) {
                '{' { $braceDepth++ }
                '}' {
                    $braceDepth--
                    if ($braceDepth -lt 0) { $braceDepth = 0 }
                    if ($braceDepth -eq 0 -and $currentFunction -ne $null) {
                        if ($currentFunction.Count -ge 3) {
                            $results += [PSCustomObject]@{
                                Rule        = $RuleId
                                Description = $Description
                                File        = $RelativePath
                                Line        = $currentFunction.StartLine
                                Snippet     = $currentFunction.StartText
                                Count       = $currentFunction.Count
                            }
                        }
                        $currentFunction = $null
                    }
                }
            }
        }
    }

    return $results
}

$cppFiles = Get-ChildItem -Path $targetDir -Recurse -File -Filter "*.cpp"
$violations = @()
$warnings = @()

foreach ($file in $cppFiles) {
    $content = Get-Content -Path $file.FullName -Raw -Encoding UTF8
    if ([string]::IsNullOrEmpty($content)) {
        continue
    }

    $lines = @($content -split "`r?`n")
    $relativePath = [System.IO.Path]::GetRelativePath($repoRoot, $file.FullName).Replace('\\', '/')

    foreach ($rule in $rules) {
        $matches = [System.Text.RegularExpressions.Regex]::Matches(
            $content,
            $rule.Pattern,
            [System.Text.RegularExpressions.RegexOptions]::None
        )

        foreach ($match in $matches) {
            $line = 1 + [System.Text.RegularExpressions.Regex]::Matches($content.Substring(0, $match.Index), "`n").Count
            $lineText = ""
            if ($line -ge 1 -and $line -le $lines.Length) {
                $lineText = $lines[$line - 1].Trim()
            }

            $violations += [PSCustomObject]@{
                Rule        = $rule.Id
                Description = $rule.Description
                File        = $relativePath
                Line        = $line
                Snippet     = $lineText
            }
        }
    }

    $warnings += Get-FunctionScopedWorldWarnings `
        -Content $content `
        -RelativePath $relativePath `
        -Pattern $warningRule.Pattern `
        -RuleId $warningRule.Id `
        -Description $warningRule.Description
}

if ($violations.Count -gt 0) {
    Write-Host "AudioEngine lint violations detected: $($violations.Count)"
    Write-Host ""

    $grouped = $violations | Group-Object Rule
    foreach ($group in $grouped) {
        Write-Host "[$($group.Name)]"
        foreach ($item in $group.Group) {
            Write-Host "  $($item.File):$($item.Line)"
            if ($item.Snippet) {
                Write-Host "    $($item.Snippet)"
            }
        }
        Write-Host ""
    }

    exit 1
}

if ($warnings.Count -gt 0) {
    Write-Host "AudioEngine lint warnings detected: $($warnings.Count)"
    Write-Host ""

    foreach ($warning in $warnings) {
        Write-Host "[$($warning.Rule)] $($warning.File):$($warning.Line)"
        Write-Host "  occurrences: $($warning.Count)"
        if ($warning.Snippet) {
            Write-Host "  $($warning.Snippet)"
        }
        Write-Host ""
    }
}

Write-Host "AudioEngine lint passed (LINT-AE-001/002/003)."
exit 0
