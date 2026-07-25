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

# Check if synthetic tool set is missing hiddenInToolsPicker
$needsFix = $false
$pattern = 'e\.createToolSet\([^,]+,[^,]+,[^,]+\.identifier\.value,\{icon:R\.extensions,description:n\.description\.displayName\?\?n\.description\.name\},p=new O'
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

$oldPattern = 'e.createToolSet(a,l,n.description.identifier.value,{icon:R.extensions,description:n.description.displayName??n.description.name}),p=new O'
$newPattern = 'e.createToolSet(a,l,n.description.identifier.value,{icon:R.extensions,description:n.description.displayName??n.description.name,hiddenInToolsPicker:!0}),p=new O'

if ($content -match [regex]::Escape($oldPattern)) {
    $content = $content -replace [regex]::Escape($oldPattern), $newPattern
    Set-Content -Path $targetFile -Value $content -NoNewline -Encoding UTF8
    Write-Host "[OK] Fix applied successfully!" -ForegroundColor Green
    Write-Host "[INFO] Please restart VS Code for the change to take effect." -ForegroundColor Yellow
} else {
    Write-Warning "Pattern not found. File structure may have changed."
    exit 1
}
