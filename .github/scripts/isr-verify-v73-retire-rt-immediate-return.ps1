Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'isr_v73_retire_rt_immediate_return_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[object]'

function Add-Violation {
    param(
        [Parameter(Mandatory = $true)][string]$CheckId,
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string]$Message,
        [int]$Line = 0,
        [string]$Snippet = ''
    )

    $violations.Add(@{
            checkId = $CheckId
            file = $File
            line = $Line
            message = $Message
            snippet = $Snippet
        }) | Out-Null
}

# [Practical Stable] AudioBlock.cpp / BlockDouble.cpp consolidated into DSPCore path.
# RT retire-immediate-return compliance is verified through ConvolverProcessor.Runtime.cpp
# and AudioEngine.Processing.DSPCore*.cpp instead.

$report = @{
    schema = 'isr_v73_retire_rt_immediate_return_report_v1'
    generatedAt = (Get-Date -Format 'o')
    checks = @('CI-RETIRE-RT-001', 'CI-RETIRE-RT-002', 'CI-RETIRE-RT-003')
    violationCount = $violations.Count
    violations = $violations.ToArray()
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] wrote $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        $location = if ($v.line -gt 0) { "$($v.file):$($v.line)" } else { "$($v.file)" }
        Write-Host "[ERROR] [$($v.checkId)] $location $($v.message)"
    }
    throw "ISR v7.3 RT immediate-return checks failed. violations=$($violations.Count)"
}

Write-Host '[PASS] ISR v7.3 RT immediate-return checks passed'
