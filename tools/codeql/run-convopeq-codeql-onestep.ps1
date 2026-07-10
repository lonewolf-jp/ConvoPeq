param(
    [string]$DatabaseName = "convopeq-cpp-standard",
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",
    [string]$QuerySuite = "codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls",
    [ValidateRange(1, 64)]
    [int]$BuildParallel = 1,
    [switch]$Clean,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..\")).Path.TrimEnd([char]92, [char]47)

$createDbScript = Join-Path $scriptDir "create-convopeq-codeql-db.ps1"
if (-not (Test-Path $createDbScript)) {
    throw "create script was not found: $createDbScript"
}

$codeqlDefault = Join-Path $env:USERPROFILE "tools\codeql\codeql.exe"
$codeqlPath = if ($env:CODEQL_PATH -and (Test-Path $env:CODEQL_PATH)) { $env:CODEQL_PATH } elseif (Test-Path $codeqlDefault) { $codeqlDefault } else { $null }
if (-not $codeqlPath) {
    throw "CodeQL CLI was not found. Set CODEQL_PATH or place codeql.exe at '$codeqlDefault'."
}

if ($QuerySuite -notmatch ":") {
    Write-Warning "QuerySuite '$QuerySuite' looks like a local path or legacy shorthand."
    Write-Warning "Recommended pack spec: codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls"
}

$dbPath = Join-Path (Join-Path $repoRoot "storage\codeql\databases") $DatabaseName
$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$buildTag = "run-$runStamp"
$runDir = Join-Path (Join-Path $repoRoot "storage\codeql\query-runs") (Join-Path $DatabaseName $runStamp)
$sarifPath = Join-Path $runDir "results.sarif"

New-Item -ItemType Directory -Path $runDir -Force | Out-Null

Write-Host "[ConvoPeq CodeQL] One-step pipeline start"
Write-Host "[ConvoPeq CodeQL] database  : $DatabaseName"
Write-Host "[ConvoPeq CodeQL] config    : $Config"
Write-Host "[ConvoPeq CodeQL] parallel  : $BuildParallel"
Write-Host "[ConvoPeq CodeQL] buildTag  : $buildTag"
Write-Host "[ConvoPeq CodeQL] suite     : $QuerySuite"
Write-Host "[ConvoPeq CodeQL] runDir    : $runDir"

# Step 1: DB create
$dbParams = @{
    DatabaseName  = $DatabaseName
    Config        = $Config
    BuildParallel = $BuildParallel
    BuildTag      = $buildTag
}
if ($Clean) {
    $dbParams.Clean = $true
}
if ($DryRun) {
    $dbParams.DryRun = $true
}

& $createDbScript @dbParams
if ($LASTEXITCODE -ne 0) {
    throw "Database creation step failed (exit=$LASTEXITCODE)"
}

# Step 2: Analyze
$analyzeArgs = @(
    "database", "analyze", $dbPath,
    $QuerySuite,
    "--download",
    "--format=sarif-latest",
    "--output", $sarifPath,
    "--threads=0"
)

if ($DryRun) {
    Write-Host "[DryRun] Analyze command preview:"
    Write-Host "`"$codeqlPath`" $($analyzeArgs -join ' ')"
    exit 0
}

& $codeqlPath @analyzeArgs
if ($LASTEXITCODE -ne 0) {
    throw "CodeQL analyze step failed (exit=$LASTEXITCODE)"
}

Write-Host "[ConvoPeq CodeQL] Analyze completed"
Write-Host "[ConvoPeq CodeQL] SARIF: $sarifPath"
