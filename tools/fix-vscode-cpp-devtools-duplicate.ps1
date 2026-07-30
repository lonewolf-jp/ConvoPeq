<#
.SYNOPSIS
    Fix duplicate C/C++ DevTools entries in VS Code "Configure Tools" panel.
.DESCRIPTION
    VS Code 1.127.0+ regression (issue #326153): synthetic per-extension tool sets
    missing hiddenInToolsPicker flag. This script detects and fixes the issue.
.PARAMETER DryRun
    Check only, do not apply fix.
.PARAMETER Force
    Skip confirmation prompt.
#>

param(
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# Find VS Code installation directory
$possiblePaths = @(
    "$env:LOCALAPPDATA\Programs\Microsoft VS Code",
    "$env:ProgramFiles\Microsoft VS Code",
    "$env:ProgramFiles(x86)\Microsoft VS Code"
)

$vsCodeRoot = $null
foreach ($p in $possiblePaths) {
    if (Test-Path $p) {
        $versionDirs = Get-ChildItem -Path $p -Directory | Where-Object { $_.Name -match '^[a-f0-9]{9,11}$' }
        foreach ($dir in $versionDirs) {
            $testPath = Join-Path $dir.FullName "resources\app\out\vs\workbench\workbench.desktop.main.js"
            if (Test-Path $testPath) {
                $vsCodeRoot = $dir.FullName
                break
            }
        }
        if ($vsCodeRoot) { break }
    }
}

if (-not $vsCodeRoot) {
    Write-Warning "VS Code installation not found."
    exit 1
}

$targetFile = Join-Path $vsCodeRoot "resources\app\out\vs\workbench\workbench.desktop.main.js"
Write-Host "[INFO] VS Code path: $vsCodeRoot" -ForegroundColor Cyan

if (-not (Test-Path $targetFile)) {
    Write-Warning "Target file not found: $targetFile"
    exit 1
}

Write-Host "[INFO] Target file: $targetFile" -ForegroundColor Cyan

# Check current state: hiddenInToolsPicker:!0 should appear 2+ times
$hiddenCount = (Select-String -Path $targetFile -Pattern "hiddenInToolsPicker:!0" -SimpleMatch).Length
Write-Host "[INFO] hiddenInToolsPicker:!0 count: $hiddenCount" -ForegroundColor Yellow

# Check if synthetic tool set is missing hiddenInToolsPicker (dynamic variable name)
$needsFix = $false
$pattern = 'e\.createToolSet\([^,]+,[^,]+,[^,]+\.identifier\.value,\{icon:[a-zA-Z_$][a-zA-Z0-9_$]*\.extensions,description:n\.description\.displayName\?\?n\.description\.name\}\)'
$matchResult = Select-String -Path $targetFile -Pattern $pattern

if ($matchResult) {
    $needsFix = $true
    Write-Host "[WARN] Fix needed: synthetic tool set is missing hiddenInToolsPicker" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Already fixed" -ForegroundColor Green
}

if ($DryRun) {
    if ($needsFix) {
        Write-Host ""
        Write-Host "[DRY-RUN] Would add hiddenInToolsPicker:!0 to the synthetic tool set's createToolSet call."
        exit 2
    }
    exit 0
}

if (-not $needsFix) {
    exit 0
}

if (-not $Force) {
    Write-Host ""
    $confirm = Read-Host "Apply fix? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "Cancelled."
        exit 0
    }
}

# Apply the fix
Write-Host "[INFO] Applying fix..." -ForegroundColor Cyan

$content = Get-Content $targetFile -Raw

# Dynamic detection: find the createToolSet call that's missing hiddenInToolsPicker
# by looking for the pattern of synthetic extension tool set creation
$capVar = '([a-zA-Z_$][a-zA-Z0-9_$]*)'  # captures a JS variable name

# The synthetic tool set pattern: e.createToolSet(a, l, n.description.identifier.value, {icon: <var>.extensions, description: ... })
# We need to find this and add hiddenInToolsPicker:!0 before the closing })
$syntheticPattern = '(e\.createToolSet\([^,]+,[^,]+,[^,]+\.identifier\.value,\{icon:[a-zA-Z_$][a-zA-Z0-9_$]*\.extensions,description:n\.description\.displayName\?\?n\.description\.name)(\}\))'
$replacement = '$1,hiddenInToolsPicker:!0$2'

if ($content -match $syntheticPattern) {
    # Check if hiddenInToolsPicker is already present
    if ($content -match [regex]::Escape('hiddenInToolsPicker')) {
        Write-Host "[OK] hiddenInToolsPicker already present in file." -ForegroundColor Green
        Write-Host "[INFO] However, if duplicate still appears, the VS Code update may have changed the pattern."
        Write-Host "[INFO] Attempting re-application to ensure coverage..."
    }

    $newContent = $content -replace $syntheticPattern, $replacement

    if ($newContent -ne $content) {
        Set-Content -Path $targetFile -Value $newContent -NoNewline -Encoding UTF8
        Write-Host "[OK] Fix applied successfully!" -ForegroundColor Green
        Write-Host "[INFO] Added hiddenInToolsPicker:!0 to the synthetic extension tool set." -ForegroundColor Cyan
        Write-Host "[INFO] Please restart VS Code for the change to take effect." -ForegroundColor Yellow

        # Register fix in repair plan note
        $notePath = "$env:USERPROFILE\.vscode\cpp-devtools-fix-note.txt"
        "VS Code fix applied on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $notePath -Encoding UTF8
    } else {
        Write-Warning "Pattern matched but replacement produced no change."
        exit 1
    }
} else {
    Write-Warning "Synthetic tool set pattern not found. File structure may have changed significantly."
    Write-Host "[INFO] Attempting fallback: searching for any createToolSet without hiddenInToolsPicker..." -ForegroundColor Yellow

    # Fallback: find any createToolSet( among non-internal sets that lacks hiddenInToolsPicker
    $lines = Get-Content $targetFile
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]
        if ($line -match '\.createToolSet\(' -and $line -notmatch 'hiddenInToolsPicker' -and $line -notmatch 'Internal') {
            Write-Host "[WARN] Found unpatched createToolSet at line $i" -ForegroundColor Yellow
            Write-Host "[WARN] $($line.Substring(0, [Math]::Min(120, $line.Length)))" -ForegroundColor Gray
            $found = $true
        }
    }
    if (-not $found) {
        Write-Host "[INFO] No unpatched synthetic tool sets found. The fix may not be needed for this version." -ForegroundColor Green
    }
    exit 1
}
