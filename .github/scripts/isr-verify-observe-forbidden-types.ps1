$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$audioBlockPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.AudioBlock.cpp'
$blockDoublePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Processing.BlockDouble.cpp'
$timerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Timer.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'observe_forbidden_types_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

$forbiddenTypePatterns = @(
    'RuntimeGraph\s*\*',
    'RuntimeBuildSnapshot\s*\*',
    'PublicationIntent\s*\*',
    'TransitionState\s*\*'
)

foreach ($path in @($headerPath, $audioBlockPath, $blockDoublePath, $timerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        $violations.Add("Missing observe-forbidden target: $path") | Out-Null
    }
}

function Get-StructBlockText {
    param(
        [Parameter(Mandatory = $true)]
        $Lines,
        [Parameter(Mandatory = $true)]
        [string]$StructName
    )

    if ($null -eq $Lines) {
        return $null
    }

    $lineArray = @()
    if ($Lines -is [string]) {
        $lineArray = @($Lines -split "`r?`n")
    }
    else {
        $lineArray = @($Lines)
    }

    $start = -1
    for ($i = 0; $i -lt $lineArray.Count; $i++) {
        if ("$($lineArray[$i])" -match "^\s*struct\s+$([regex]::Escape($StructName))\b") {
            $start = $i
            break
        }
    }

    if ($start -lt 0) {
        return $null
    }

    $braceDepth = 0
    $seenOpenBrace = $false
    $buffer = New-Object 'System.Collections.Generic.List[string]'

    for ($i = $start; $i -lt $lineArray.Count; $i++) {
        $line = "$($lineArray[$i])"
        $buffer.Add($line) | Out-Null

        $opens = ([regex]::Matches($line, '\{')).Count
        $closes = ([regex]::Matches($line, '\}')).Count

        if ($opens -gt 0) {
            $seenOpenBrace = $true
        }

        $braceDepth += $opens
        $braceDepth -= $closes

        if ($seenOpenBrace -and $braceDepth -eq 0) {
            break
        }
    }

    return ($buffer -join "`n")
}

if (Test-Path -LiteralPath $headerPath) {
    $headerLines = Get-Content -LiteralPath $headerPath -Encoding UTF8

    foreach ($blockName in @('AudioCallbackAuthorityView', 'RuntimeReadView', 'RuntimePublishView')) {
        $blockText = Get-StructBlockText -Lines $headerLines -StructName $blockName
        if ([string]::IsNullOrWhiteSpace($blockText)) {
            $violations.Add("$blockName definition missing") | Out-Null
            continue
        }

        foreach ($pattern in $forbiddenTypePatterns) {
            if ([regex]::IsMatch($blockText, $pattern)) {
                $violations.Add("Observe forbidden-type violation: $blockName contains pattern '$pattern'") | Out-Null
            }
        }
    }
}

foreach ($path in @($audioBlockPath, $blockDoublePath, $timerPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        continue
    }

    $text = Get-Content -LiteralPath $path -Raw -Encoding UTF8
    foreach ($pattern in $forbiddenTypePatterns) {
        if ([regex]::IsMatch($text, $pattern)) {
            $violations.Add("Observe forbidden-type violation: pattern '$pattern' detected in $path") | Out-Null
        }
    }
}

$report = [ordered]@{
    schema                = 'observe_forbidden_types_report_v1'
    generatedAt           = (Get-Date -Format 'o')
    headerPath            = $headerPath
    targets               = @($audioBlockPath, $blockDoublePath, $timerPath)
    forbiddenTypePatterns = $forbiddenTypePatterns
    violations            = @($violations)
    ready                 = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw 'observe forbidden-type verification failed'
}

Write-Host '[PASS] observe forbidden-type verification passed'
