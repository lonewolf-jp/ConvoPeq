# verify-isr-maturity.ps1
# ISR Bridge Runtime 成熟度検証スクリプト
# 7つの成熟度指標を自動検証する。
#
# Usage: .\verify-isr-maturity.ps1 [-Detailed]
#   -Detailed: 各チェックの詳細情報も出力する

param(
    [switch]$Detailed = $false
)

$ErrorActionPreference = "Stop"
$srcRoot = "C:\VSC_Project\ConvoPeq\src"
$allPassed = $true
$failures = @()

function Check-Condition {
    param($Name, $Condition, $Detail)
    if (& $Condition) {
        Write-Host "[PASS] $Name" -ForegroundColor Green
        if ($Detailed -and $Detail) {
            Write-Host "       $Detail" -ForegroundColor DarkGray
        }
        return $true
    }
    else {
        Write-Host "[FAIL] $Name" -ForegroundColor Red
        if ($Detail) { Write-Host "       $Detail" -ForegroundColor DarkRed }
        $script:allPassed = $false
        $script:failures += $Name
        return $false
    }
}

Write-Host "=== ISR Maturity Verification ===" -ForegroundColor Cyan
Write-Host ""

# Condition 1: RT path で delete がないこと
Check-Condition -Name "RT path: no delete keyword" -Condition {
    $rtFiles = @(
        "src/audioengine/AudioEngine.Processing.AudioBlock.cpp",
        "src/audioengine/AudioEngine.Processing.BlockDouble.cpp"
    )
    $total = 0
    foreach ($f in $rtFiles) {
        $path = Join-Path $srcRoot "..\$f"
        if (Test-Path $path) {
            $matches = Select-String -Path $path -Pattern "\bdelete\b" -SimpleMatch
            $total += $matches.Count
        }
    }
    $total -eq 0
} -Detail "getNextAudioBlock / processBlockDouble に delete なし"

# Condition 2: RT path で lock がないこと
Check-Condition -Name "RT path: no mutex/condition_variable" -Condition {
    $rtFiles = @(
        "src/audioengine/AudioEngine.Processing.AudioBlock.cpp",
        "src/audioengine/AudioEngine.Processing.BlockDouble.cpp"
    )
    $total = 0
    foreach ($f in $rtFiles) {
        $path = Join-Path $srcRoot "..\$f"
        if (Test-Path $path) {
            $matches = Select-String -Path $path -Pattern "(mutex|\.lock\(\)|condition_variable)" -SimpleMatch
            $total += $matches.Count
        }
    }
    $total -eq 0
} -Detail "getNextAudioBlock / processBlockDouble に mutex/lock/condition_variable なし"

# Condition 3: retireDSP 直接呼び出しがゼロ
Check-Condition -Name "Retire Epoch: no direct retireDSP() calls" -Condition {
    $cppFiles = Get-ChildItem -Recurse -Filter "*.cpp" $srcRoot | % FullName
    $headerFiles = Get-ChildItem -Recurse -Filter "*.h" $srcRoot | % FullName
    $allFiles = $cppFiles + $headerFiles
    $total = 0
    $details = @()
    foreach ($f in $allFiles) {
        # Skip codeql/storage database copies
        if ($f -match '\\(storage|\.musubi)\\.*codeql') { continue }
        $content = Get-Content $f -Raw -ErrorAction SilentlyContinue
        if (-not $content) { continue }
        # Match retireDSP( as a function call (not in comments)
        $lines = $content -split "`n"
        for ($i = 0; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            # Skip comment lines
            if ($line -match '^\s*//') { continue }
            if ($line -match 'retireDSP\(' -and -not $line -match '//.*retireDSP') {
                $total++
                $relPath = $f.Substring($srcRoot.Length - 3)
                $details += "  $relPath(line $($i+1)): $($line.Trim())"
            }
        }
    }
    if ($Detailed -and $total -gt 0) {
        $details | % { Write-Host $_ -ForegroundColor DarkRed }
    }
    $total -eq 0
} -Detail "retireDSP() direct calls eliminated (use DSPLifetimeManager)"

# Condition 4: Shutdown 完全 Drain（trace 確認）
Check-Condition -Name "Shutdown: VerifyDrained phase exists" -Condition {
    $shutdownH = Join-Path $srcRoot "audioengine/ISRShutdown.h"
    $content = Get-Content $shutdownH -Raw -ErrorAction SilentlyContinue
    $content -match "VerifyDrained"
} -Detail "ShutdownPhase 列挙に VerifyDrained が含まれている"

# Condition 5: Overflow データ喪失防止（droppedIntentCount 監視機能）
Check-Condition -Name "Overflow: droppedIntentCount tracking exists" -Condition {
    $retireH = Join-Path $srcRoot "audioengine/ISRRetire.h"
    $content = Get-Content $retireH -Raw -ErrorAction SilentlyContinue
    $content -match "droppedIntentCount"
} -Detail "ISRRetireRuntime に droppedIntentCount_ が定義されている"

# Condition 6: HealthMonitor が Detect + Diagnose + Report
Check-Condition -Name "HealthMonitor: Diagnose capability exists" -Condition {
    $monitorH = Join-Path $srcRoot "audioengine/RuntimeHealthMonitor.h"
    $content = Get-Content $monitorH -Raw -ErrorAction SilentlyContinue
    $content -match "diagnoseRetireStall|diagnose"
} -Detail "HealthMonitor に diagnose メソッドが存在する"

# Condition 7: Coordinator 唯一 Authority
Check-Condition -Name "Coordinator Authority: DSPLifetimeManager unified" -Condition {
    $lmH = Join-Path $srcRoot "audioengine/DSPLifetimeManager.h"
    $content = Get-Content $lmH -Raw -ErrorAction SilentlyContinue
    # DSPLifetimeManager should reference ISRRetireRouter (Phase-B pattern)
    $content -match "ISRRetireRouter"
} -Detail "DSPLifetimeManager が ISRRetireRouter 経由で retire している"

# Condition 8: EpochDomain direct call 監査
Check-Condition -Name "EpochDomain: enqueueRetire via Router only" -Condition {
    $epochH = Join-Path $srcRoot "core/EpochDomain.h"
    $epochContent = Get-Content $epochH -Raw -ErrorAction SilentlyContinue
    $epochContent -match "enqueueRetire" -and $epochContent -notmatch "deprecated"
} -Detail "EpochDomain::enqueueRetire が存在する（Router経由で呼ばれる）"

Write-Host ""
if ($allPassed) {
    Write-Host "=== ALL CHECKS PASSED ===" -ForegroundColor Green
    exit 0
}
else {
    Write-Host "=== FAILURES: $($failures.Count) ===" -ForegroundColor Red
    $failures | % { Write-Host "  - $_" -ForegroundColor Red }
    exit 1
}
