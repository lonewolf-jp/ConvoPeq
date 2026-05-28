param(
    [switch]$Enforce,
    [switch]$IncludeHeavyChecks,
    [switch]$TryAutoCaptureHeavyLog,
    [int]$HeavyAutoCaptureTimeoutSec = 20
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
Set-Location $repoRoot

function Invoke-Check {
    param(
        [string]$Id,
        [string]$Title,
        [string]$ScriptPath,
        [string[]]$ScriptArgs = @(),
        [switch]$Heavy
    )

    $result = [ordered]@{
        id      = $Id
        title   = $Title
        script  = $ScriptPath
        status  = 'skipped'
        details = ''
    }

    if ($Heavy -and -not $IncludeHeavyChecks) {
        $result.details = 'heavy check omitted (use -IncludeHeavyChecks)'
        return [pscustomobject]$result
    }

    $fullPath = Join-Path $repoRoot $ScriptPath
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $result.details = 'script not found'
        return [pscustomobject]$result
    }

    try {
        & $fullPath @ScriptArgs | Out-Null
        $result.status = 'pass'
        $result.details = 'ok'
    }
    catch {
        $result.status = 'fail'
        $result.details = $_.Exception.Message
    }

    return [pscustomobject]$result
}

$checks = @(
    @{ id = 'DoD-01'; title = 'worker runtime pointer isolation'; script = '.github/scripts/check-list-compliance.ps1'; args = @() },
    @{ id = 'DoD-02'; title = 'shutdown publication blocked'; script = '.github/scripts/check-list-compliance.ps1'; args = @() },
    @{ id = 'DoD-03'; title = 'saturation policy implemented'; script = '.github/scripts/isr-verify-p3-governance.ps1'; args = @() },
    @{ id = 'DoD-04'; title = 'rebuild collapse deterministic'; script = '.github/scripts/isr-rebuild-admission-8_1-metrics.ps1'; args = @(); heavy = $true },
    @{ id = 'DoD-05'; title = 'stale runtime reuse blocked'; script = '.github/scripts/isr-verify-v3-runtime-graph-immutability.ps1'; args = @() },
    @{ id = 'DoD-06'; title = 'crossfade shared mutable authority absent'; script = '.github/scripts/isr-verify-crossfade-observable-state.ps1'; args = @() },
    @{ id = 'DoD-07'; title = 'RT path no mutex allocation'; script = '.github/scripts/check-src-atomic-dotcall.ps1'; args = @() },
    @{ id = 'DoD-08'; title = 'deterministic shutdown established'; script = '.github/scripts/isr-verify-drained-resurrection-guard.ps1'; args = @() },
    @{ id = 'DoD-09'; title = 'drain deterministic with fallback'; script = '.github/scripts/isr-verify-v7-rt-nonrt-retire-bridge.ps1'; args = @() },
    @{ id = 'DoD-10'; title = 'finalize deterministic'; script = '.github/scripts/isr-verify-v2-seal.ps1'; args = @() }
)

$results = @()
foreach ($c in $checks) {
    $heavy = $false
    if ($c.ContainsKey('heavy')) { $heavy = [bool]$c.heavy }

    $scriptArgs = @()
    if ($c.ContainsKey('args')) { $scriptArgs = @($c.args) }

    if ($c.id -eq 'DoD-04' -and $IncludeHeavyChecks) {
        if ($TryAutoCaptureHeavyLog) {
            $scriptArgs += '-TryAutoCaptureOnMissingLog'
            $scriptArgs += '-AutoCaptureTimeoutSec'
            $scriptArgs += [string]$HeavyAutoCaptureTimeoutSec
        }
    }

    $results += Invoke-Check -Id $c.id -Title $c.title -ScriptPath $c.script -ScriptArgs $scriptArgs -Heavy:$heavy
}

$summary = [ordered]@{
    generatedAtUtc     = (Get-Date).ToUniversalTime().ToString('o')
    schema             = 'isr_v5_5_1_dod_scaffold_v1'
    includeHeavyChecks = [bool]$IncludeHeavyChecks
    passCount          = (@($results | Where-Object { $_.status -eq 'pass' })).Count
    failCount          = (@($results | Where-Object { $_.status -eq 'fail' })).Count
    skippedCount       = (@($results | Where-Object { $_.status -eq 'skipped' })).Count
    results            = $results
}

$evidenceDir = Join-Path $repoRoot 'evidence'
if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir | Out-Null
}

$reportPath = Join-Path $evidenceDir 'isr_v5_5_1_dod_report.json'
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8

Write-Host ("[INFO] DoD scaffold report: " + $reportPath)
Write-Host ("[INFO] pass=" + $summary.passCount + " fail=" + $summary.failCount + " skipped=" + $summary.skippedCount)

if ($Enforce -and $summary.failCount -gt 0) {
    throw "DoD scaffold enforce failed: failCount=$($summary.failCount)"
}
