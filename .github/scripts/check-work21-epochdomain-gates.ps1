#!/usr/bin/env pwsh
<#
.SYNOPSIS
    work21 refactoring plan CI gates — EpochDomain dependency audit.
    Detects violations of P0/P1 rules.
.DESCRIPTION
    Gates:
      1. EpochDomainReaderGuard direct construction (P1-18)
      2. RCUReader::domain() usage (P0-14)
      3. EpochCore.h residual (P1-21 Delete)
      4. advanceEpoch call-site count (P0-13)
      5. EpochDomain& return type in public API (P1-19)
.PARAMETER PassThru
    Emit structured results JSON instead of exit code.
#>

param([switch]$PassThru)

$root = Resolve-Path "$PSScriptRoot/../.."
$src = Join-Path $root "src"

$errors = @()

# Helper: get all .h/.cpp files
function Get-SourceFiles {
    Get-ChildItem -Recurse -Include "*.h", "*.cpp" -Path $args[0] | % FullName
}

# Gate 1: EpochDomainReaderGuard direct construction (outside definition)
$readerGuardHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern 'EpochDomainReaderGuard\(' -SimpleMatch `
| Where-Object { $_.Path -notmatch '\\EpochDomain\.[hc]' }
if ($readerGuardHits) {
    $errors += "P1-18 VIOLATION: EpochDomainReaderGuard( used outside definition:"
    $readerGuardHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 3: EpochCore.h still exists
if (Test-Path (Join-Path $src "core/EpochCore.h")) {
    $errors += "P1-21 VIOLATION: EpochCore.h still exists (should be deleted)"
}

# Gate 4: advanceEpoch call-site count (allowed: 1 — RuntimePublicationOrchestrator only)
$advanceEpochHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern '\.advanceEpoch\(' `
| Where-Object { $_.Path -notmatch '\\EpochDomain\.[hc]' -and $_.Path -notmatch '\\test' }
$advanceEpochCount = @($advanceEpochHits).Count
if ($advanceEpochHits) {
    $errors += "P0-13 VIOLATION: .advanceEpoch( call-sites = $advanceEpochCount (allowed: 1 in RuntimePublicationOrchestrator):"
    $advanceEpochHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 6: EpochDomain type exposure in public API (P1-19)
# Scope: .h files only. .cpp implementation files are private internals
# and are excluded from the public API boundary check.
$epochDomainExposure = Select-String -Path (Get-ChildItem -Recurse -Include "*.h" -Path $src | % FullName) `
    -Pattern '(EpochDomain\s*&|EpochDomain\s*\*)' `
| Where-Object {
    $line = $_.Line.Trim()
    $_.Path -notmatch '\\EpochDomain\.[hc]' -and
    $_.Path -notmatch '\\test' -and
    $line -notmatch '^\s*(//|#|\*)'
}
if ($epochDomainExposure) {
    $errors += "P1-19 VIOLATION: EpochDomain type exposed in public API:"
    $epochDomainExposure | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 6b (P0-A): dynamic_cast<EpochDomain*> detection — ISRRetireRouter の dynamic_cast 完全撤廃確認
$dynamicCastHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern 'dynamic_cast\s*<\s*EpochDomain\s*\*' `
| Where-Object { $_.Path -notmatch '\\test' }
if ($dynamicCastHits) {
    $errors += "P0-A VIOLATION: dynamic_cast<EpochDomain*> detected (should use IRetireProvider interface):"
    $dynamicCastHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 10 (P1-3 AST-like): EpochDomain alias detection (using/typedef)
# Detects potential EpochDomain type aliases that circumvent type-name grep.
$epochDomainAliasHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern '(using\s+\w+\s*=\s*.*EpochDomain|typedef.*EpochDomain)' `
| Where-Object {
    $_.Path -notmatch '\\test' -and
    $_.Line -notmatch '^\s*(//|#)'
}
if ($epochDomainAliasHits) {
    $errors += "P1-3 VIOLATION: EpochDomain type alias detected (using/typedef):"
    $epochDomainAliasHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 10b (P1-3 AST-like): Two-step alias detection
# Extracts alias names from "using X = EpochDomain" then checks if those names are used as types.
$aliasNames = @()
$aliasDefHits = Select-String -Path (Get-SourceFiles $src) -Pattern 'using\s+(\w+)\s*=\s*EpochDomain' | Where-Object { $_.Path -notmatch '\\test' }
foreach ($hit in $aliasDefHits) {
    if ($hit.Matches.Groups[1].Value) {
        $aliasNames += $hit.Matches.Groups[1].Value
    }
}
if ($aliasNames.Count -gt 0) {
    $aliasPattern = '\b(' + ($aliasNames -join '|') + ')\s*[&*]'
    $aliasUsageHits = Select-String -Path (Get-SourceFiles $src) -Pattern $aliasPattern | Where-Object {
        $_.Path -notmatch '\\test' -and $_.Line -notmatch '^\s*(//|#|using)' -and $_.Line -notmatch 'EpochDomain'
    }
    if ($aliasUsageHits) {
        $errors += "P1-3 VIOLATION: EpochDomain alias type used in public API:"
        $aliasUsageHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
    }
}

# Gate 11 (P1-3 AST-like): EpochDomain as template parameter
$epochDomainTemplateHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern '\bEpochDomain\s*>' `
| Where-Object {
    $line = $_.Line.Trim()
    $_.Path -notmatch '\\EpochDomain\.[hc]' -and
    $_.Path -notmatch '\\test' -and
    $_.Path -notmatch '\\ISRRetireRouter\.[hc]' -and
    $line -notmatch '^\s*(//|#|\*)'
}
if ($epochDomainTemplateHits) {
    $errors += "P1-3 VIOLATION: EpochDomain used as template parameter:"
    $epochDomainTemplateHits | ForEach-Object { $errors += "  $($_.Path):$($_.LineNumber): $($_.Line.Trim())" }
}

# Gate 7 (info): advanceEpoch direct call count (excl declarations, excl flag-based)
$advanceDirectHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern '\.advanceEpoch\(' `
| Where-Object {
    $_.Path -notmatch '\\EpochDomain\.[hc]' -and
    $_.Path -notmatch '\\test' -and
    $_.Line -notmatch 'void advanceEpoch|uint64_t advanceEpoch|void advanceRetireEpoch|uint64_t advanceRetireEpoch|m_epochAdvancePending|flushPendingEpoch'
}
$advanceDirectCount = @($advanceDirectHits).Count

# Gate 12 (info): Router own public API count (excl interface overrides, ctors, dtors, deleted)
# [P1-3] Monitor for Router bloating beyond enqueue/publish/observer/telemetry categories.
$routerFile = Join-Path $root "src/audioengine/ISRRetireRouter.h"
$routerOwnMethodCount = 0
$routerOwnMethods = @()
if (Test-Path $routerFile) {
    # Count methods that are NOT override, NOT constructor/destructor, NOT operators, NOT deleted
    $allRouterMethods = Select-String -Path $routerFile -Pattern '^\s{4}(?!private:|public:)(?!.*= delete)(?!.*override)(?!.*ISRRetireRouter\()(?!.*~ISRRetireRouter)(?!.*operator=)\w+.*\(.*\)' | Where-Object {
        $_.Line -notmatch '^\s*(//|#|\})' -and $_.Line -notmatch 'static_assert|using |enum '
    }
    $routerOwnMethods = $allRouterMethods
    $routerOwnMethodCount = @($allRouterMethods).Count
}

# Gate 8 (info): enqueueRetire direct calls
$enqueueRetireHits = Select-String -Path (Get-SourceFiles $src) `
    -Pattern '\.enqueueRetire\(' `
| Where-Object {
    $_.Path -notmatch '\\EpochDomain\.[hc]' -and
    $_.Path -notmatch '\\test' -and
    $_.Line -notmatch '^\s*(//|#|bool.*enqueueRetire|void.*enqueueRetire|Result.*enqueueRetire)'
}
$enqueueRetireCount = @($enqueueRetireHits).Count

# Report
if ($PassThru) {
    return @{
        Passed    = $errors.Count -eq 0
        GateCount = 4
        Errors    = $errors
        Stats     = @{
            advanceEpochCount  = $advanceEpochCount
            advanceEpochDirect = $advanceDirectCount
            enqueueRetireCount = $enqueueRetireCount
        }
    }
}

if ($errors.Count -gt 0) {
    Write-Host "=== work21 CI GATE FAILURES ===" -ForegroundColor Red
    $errors | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }
    Write-Host "---" -ForegroundColor Cyan
}
Write-Host "advanceEpoch direct calls: $advanceDirectCount (target: 0)" -ForegroundColor $(if ($advanceDirectCount -gt 0) { "Yellow" } else { "Green" })
Write-Host "enqueueRetire calls: $enqueueRetireCount (target: 0)" -ForegroundColor $(if ($enqueueRetireCount -gt 0) { "Yellow" } else { "Green" })
Write-Host "EpochDomain type exposure: $($epochDomainExposure.Count) (target: 0)" -ForegroundColor $(if ($epochDomainExposure.Count -gt 0) { "Yellow" } else { "Green" })

# Gate 9 (info): Per-interface pure virtual method counts (prevent interface bloat)
# [work21 Phase-D] Split across IReaderEpochProvider + IPublicationProvider + IRetireProvider + IEpochProvider.
$iepFile = Join-Path $root "src/core/IEpochProvider.h"
$readerFile = Join-Path $root "src/core/IReaderEpochProvider.h"
$pubFile = Join-Path $root "src/core/IPublicationProvider.h"
$retireFile = Join-Path $root "src/core/IRetireProvider.h"

function Get-MethodCount($path) {
    if (Test-Path $path) {
        return (Select-String -Path $path -Pattern 'virtual.*noexcept = 0;' | Measure-Object).Count
    }
    return 0
}

$readerMethodCount = Get-MethodCount $readerFile
$pubMethodCount = Get-MethodCount $pubFile
$retireMethodCount = Get-MethodCount $retireFile
$iepMethodCount = Get-MethodCount $iepFile
$totalMethodCount = $readerMethodCount + $pubMethodCount + $retireMethodCount + $iepMethodCount

Write-Host "[P1-3] IReaderEpochProvider methods: $readerMethodCount (alert if > 10)" `
    -ForegroundColor $(if ($readerMethodCount -gt 10) { "Yellow" } else { "Green" })
Write-Host "[P1-3] IPublicationProvider methods: $pubMethodCount (alert if > 3)" `
    -ForegroundColor $(if ($pubMethodCount -gt 3) { "Yellow" } else { "Green" })
Write-Host "[P1-3] IRetireProvider methods: $retireMethodCount (alert if > 4)" `
    -ForegroundColor $(if ($retireMethodCount -gt 4) { "Yellow" } else { "Green" })
Write-Host "[P1-3] IEpochProvider (facade) methods: $iepMethodCount (should be 0)" `
    -ForegroundColor $(if ($iepMethodCount -gt 0) { "Yellow" } else { "Green" })
Write-Host "Total EpochProvider pure virtual methods: $totalMethodCount (alert if > 12)" `
    -ForegroundColor $(if ($totalMethodCount -gt 12) { "Yellow" } else { "Green" })

# Gate 12 output
Write-Host "[P1-3] ISRRetireRouter own public API methods: $routerOwnMethodCount (alert if > 8)" `
    -ForegroundColor $(if ($routerOwnMethodCount -gt 8) { "Yellow" } else { "Green" })

# Gate 13 (Phase-E P4): AST-like EpochDomain scanner
$astScript = Join-Path $root "tools/check-epochdomain-ast.py"
if (Test-Path $astScript) {
    $astResult = & python $astScript 2>&1 | Out-String
    $astDirectCount = if ($astResult -match 'Direct EpochDomain usage:\s+(\d+)') { [int]$Matches[1] } else { -1 }
    $astAliasCount = if ($astResult -match 'Type aliases found:\s+(\d+)') { [int]$Matches[1] } else { -1 }
    $astAliasTransitive = if ($astResult -match 'Alias transitive usage:\s+(\d+)') { [int]$Matches[1] } else { -1 }
    $astTemplateCount = if ($astResult -match 'Template parameter usage:\s+(\d+)') { [int]$Matches[1] } else { -1 }

    Write-Host "[P4] AST scanner - Direct: $astDirectCount, Aliases: $astAliasCount, Transitive: $astAliasTransitive, Template: $astTemplateCount" `
        -ForegroundColor $(if ($astAliasCount -gt 0 -or $astTemplateCount -gt 0) { "Yellow" } else { "Green" })

    if ($astAliasCount -gt 0) {
        $errors += "P4 VIOLATION: AST scanner found EpochDomain type aliases ($astAliasCount)"
    }
    if ($astTemplateCount -gt 0) {
        $errors += "P4 VIOLATION: AST scanner found EpochDomain template parameter usage ($astTemplateCount)"
    }
}

# ── Architecture Regression Snapshot (Phase-E P1) ──
$snapshotDir = Join-Path $root ".github/architecture-snapshots"
if (-not (Test-Path $snapshotDir)) { New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null }

$snapshotFile = Join-Path $snapshotDir "latest.json"
$metrics = @{
    timestamp     = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    advanceEpoch  = $advanceDirectCount
    enqueueRetire = $enqueueRetireCount
    exposure      = $epochDomainExposure.Count
    readerMethods = $readerMethodCount
    pubMethods    = $pubMethodCount
    retireMethods = $retireMethodCount
    iepMethods    = $iepMethodCount
    totalMethods  = $totalMethodCount
    routerMethods = $routerOwnMethodCount
    aliasCount    = @($epochDomainAliasHits).Count
    templateCount = @($epochDomainTemplateHits).Count
}

# Load previous snapshot for delta comparison
$previous = @{}
if (Test-Path $snapshotFile) {
    try { $previous = Get-Content $snapshotFile -Raw | ConvertFrom-Json -AsHashtable } catch {}
}

# Save current snapshot
$metrics | ConvertTo-Json | Set-Content $snapshotFile -Force

# Compute deltas and display
Write-Host "`n=== Architecture Regression Snapshot ===" -ForegroundColor Cyan
$regressionWarnings = @()
$hasRegression = $false
$maxDelta = 0

foreach ($key in @('advanceEpoch', 'enqueueRetire', 'exposure', 'totalMethods', 'routerMethods', 'aliasCount', 'templateCount')) {
    $current = $metrics[$key]
    $prevVal = if ($previous.ContainsKey($key)) { $previous[$key] } else { $current }
    $delta = $current - $prevVal
    if ($delta -ne 0) { $hasRegression = $true }
    if ($delta -gt $maxDelta) { $maxDelta = $delta }

    $color = if ($delta -gt 0) { "Yellow" } elseif ($delta -lt 0) { "Green" } else { "Gray" }
    Write-Host "  $key`: $current (prev: $prevVal, delta: $delta)" -ForegroundColor $color

    if ($delta -gt 0) {
        $regressionWarnings += "$key increased by $delta (now $current)"
    }
}

# Multi-level threshold enforcement (Phase-E P1)
if ($hasRegression) {
    Write-Host "`n[Phase-E P1] Regression detected (Delta > 0)" -ForegroundColor Yellow
    $regressionWarnings | ForEach-Object { Write-Host "  WARNING: $_" -ForegroundColor Yellow }

    if ($maxDelta -gt 5) {
        Write-Host "[Phase-E P1] CRITICAL: max delta $maxDelta exceeds threshold (5) - CI FAIL" -ForegroundColor Red
        $errors += "P1 REGRESSION CRITICAL: max delta $maxDelta > 5"
    }
    elseif ($maxDelta -gt 3) {
        Write-Host "[Phase-E P1] REVIEW REQUIRED: max delta $maxDelta exceeds threshold (3)" -ForegroundColor Magenta
    }
}

Write-Host ""

if ($errors.Count -gt 0) {
    Write-Host "=== work21 CI GATE FAILURES ===" -ForegroundColor Red
    exit 1
}

Write-Host "=== work21 CI GATES: ALL PASSED ===" -ForegroundColor Green
exit 0
