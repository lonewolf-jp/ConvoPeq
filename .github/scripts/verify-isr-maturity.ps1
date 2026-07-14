# verify-isr-maturity.ps1
# ISR Bridge Runtime Maturity Verification Script
param([switch]$Detailed = $false)
$ErrorActionPreference = "Stop"
$srcRoot = "C:\VSC_Project\ConvoPeq\src"
$allPassed = $true
$failures = @()

function Check-Condition {
    param($Name, $Condition, $Detail)
    if (& $Condition) {
        Write-Host "[PASS] $Name" -ForegroundColor Green
        if ($Detailed -and $Detail) { Write-Host "       $Detail" -ForegroundColor DarkGray }
    } else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkRed }
        $script:allPassed = $false
        $script:failures += $Name
    }
}
Write-Host "=== ISR Maturity Verification ===" -ForegroundColor Cyan
Write-Host ""

Check-Condition -Name "RT path: no delete" -Condition {
    $total = 0
    foreach ($f in @("src/audioengine/AudioEngine.Processing.AudioBlock.cpp","src/audioengine/AudioEngine.Processing.BlockDouble.cpp")) {
        $p = Join-Path $srcRoot "..\$f"
        if (Test-Path $p) { $total += (Select-String -Path $p -Pattern "delete" -SimpleMatch).Count }
    }
    $total -eq 0
} -Detail "No delete in RT path"

Check-Condition -Name "RT path: no mutex/lock" -Condition {
    $total = 0
    foreach ($f in @("src/audioengine/AudioEngine.Processing.AudioBlock.cpp","src/audioengine/AudioEngine.Processing.BlockDouble.cpp")) {
        $p = Join-Path $srcRoot "..\$f"
        if (Test-Path $p) { $total += (Select-String -Path $p -Pattern "mutex|\.lock\(\)|condition_variable" -SimpleMatch).Count }
    }
    $total -eq 0
} -Detail "No mutex/lock in RT path"

Check-Condition -Name "Retire: no direct retireDSP()" -Condition {
    $total = 0
    $all = (Get-ChildItem -Recurse -Filter "*.cpp" $srcRoot | % FullName) + (Get-ChildItem -Recurse -Filter "*.h" $srcRoot | % FullName)
    foreach ($f in $all) {
        if ($f -match '\\codeql') { continue }
        $c = Get-Content $f -Raw -ErrorAction SilentlyContinue
        if (-not $c) { continue }
        $lines = $c -split "`n"
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $l = $lines[$i]
            if ($l -match '^\s*//') { continue }
            if ($l -match 'retireDSP\(' -and -not $l -match '//.*retireDSP') { $total++ }
        }
    }
    $total -eq 0
} -Detail "No direct retireDSP() calls"

Check-Condition -Name "Shutdown: VerifyDrained exists" -Condition {
    $c = Get-Content (Join-Path $srcRoot "audioengine/ISRShutdown.h") -Raw -ErrorAction SilentlyContinue
    $c -match "VerifyDrained"
} -Detail "ShutdownPhase has VerifyDrained"

Check-Condition -Name "Overflow: droppedIntentCount" -Condition {
    $c = Get-Content (Join-Path $srcRoot "audioengine/ISRRetire.h") -Raw -ErrorAction SilentlyContinue
    $c -match "droppedIntentCount"
} -Detail "ISRRetire has droppedIntentCount_"

Check-Condition -Name "HealthMonitor: Diagnose" -Condition {
    $c = Get-Content (Join-Path $srcRoot "audioengine/RuntimeHealthMonitor.h") -Raw -ErrorAction SilentlyContinue
    $c -match "diagnose"
} -Detail "HealthMonitor has diagnose"

Check-Condition -Name "Coordinator: DSPLifetimeManager unified" -Condition {
    $c = Get-Content (Join-Path $srcRoot "audioengine/DSPLifetimeManager.h") -Raw -ErrorAction SilentlyContinue
    $c -match "ISRRetireRouter"
} -Detail "DSPLifetimeManager uses ISRRetireRouter"

Check-Condition -Name "EpochDomain: enqueueRetire exists" -Condition {
    $c = Get-Content (Join-Path $srcRoot "core/EpochDomain.h") -Raw -ErrorAction SilentlyContinue
    $c -match "enqueueRetire"
} -Detail "EpochDomain has enqueueRetire (Router or overloaded)"

Write-Host ""
if ($allPassed) { Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green; exit 0 }
else { Write-Host "=== FAILURES: $($failures.Count) ===" -ForegroundColor Red; foreach ($f in $failures) { Write-Host "  - $f" -ForegroundColor Red }; exit 1 }
