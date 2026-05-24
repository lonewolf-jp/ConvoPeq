param(
    [string]$RepoRoot = "",
    [switch]$Full,
    [switch]$NoCommunity
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[CodeGraphRun] $Message"
}

if ([string]::IsNullOrWhiteSpace($RepoRoot)) {
    $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
}
else {
    $RepoRoot = (Resolve-Path $RepoRoot).Path
}

$patchScript = Join-Path $RepoRoot "tools\codegraph\patch-codegraph-tools.ps1"
$pruneScript = Join-Path $RepoRoot "tools\codegraph\prune-codegraph-noise.ps1"
$codegraphExe = Join-Path $RepoRoot ".venv-codegraph\Scripts\codegraph-mcp.exe"

if (-not (Test-Path -LiteralPath $patchScript)) {
    throw "Patch script not found: $patchScript"
}
if (-not (Test-Path -LiteralPath $pruneScript)) {
    throw "Prune script not found: $pruneScript"
}
if (-not (Test-Path -LiteralPath $codegraphExe)) {
    throw "CodeGraph executable not found: $codegraphExe"
}

Write-Info "Step 1/3: Apply local patch"
& powershell -NoProfile -ExecutionPolicy Bypass -File $patchScript
if ($LASTEXITCODE -ne 0) {
    throw "Patch step failed with exit code $LASTEXITCODE"
}

Write-Info "Step 2/3: Run index"
$indexArgs = @("index", $RepoRoot)
if ($Full) {
    $indexArgs += "--full"
}
if ($NoCommunity) {
    $indexArgs += "--no-community"
}

& $codegraphExe @indexArgs
if ($LASTEXITCODE -ne 0) {
    throw "Index step failed with exit code $LASTEXITCODE"
}

Write-Info "Step 3/3: Prune venv noise"
& powershell -NoProfile -ExecutionPolicy Bypass -File $pruneScript -RepoRoot $RepoRoot
if ($LASTEXITCODE -ne 0) {
    throw "Prune step failed with exit code $LASTEXITCODE"
}

Write-Info "Completed successfully (patch -> index -> prune)."
