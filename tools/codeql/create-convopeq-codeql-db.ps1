param(
    [string]$DatabaseName = "convopeq-cpp-standard",
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Debug",
    [ValidateRange(1, 64)]
    [int]$BuildParallel = 1,
    [string]$BuildTag = "default",
    [switch]$Clean,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\..\")).Path.TrimEnd([char]92, [char]47)

$codeqlDefault = Join-Path $env:USERPROFILE "tools\codeql\codeql.exe"
$codeqlPath = if ($env:CODEQL_PATH -and (Test-Path $env:CODEQL_PATH)) { $env:CODEQL_PATH } elseif (Test-Path $codeqlDefault) { $codeqlDefault } else { $null }
if (-not $codeqlPath) {
    throw "CodeQL CLI was not found. Set CODEQL_PATH or place codeql.exe at '$codeqlDefault'."
}

$vcvars = "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat"
if (-not (Test-Path $vcvars)) {
    throw "vcvarsall.bat was not found: $vcvars"
}

$oneApiSetvars = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (-not (Test-Path $oneApiSetvars)) {
    throw "oneAPI setvars.bat was not found: $oneApiSetvars"
}

$storageRoot = Join-Path $repoRoot "storage\codeql"
$dbBaseDir = Join-Path $storageRoot "databases"
$tmpDir = Join-Path $storageRoot "tmp"
$buildTagSanitized = ($BuildTag -replace '[^A-Za-z0-9_-]', '-')
if ([string]::IsNullOrWhiteSpace($buildTagSanitized)) {
    $buildTagSanitized = "default"
}
$buildDir = Join-Path $repoRoot "build"
$dbPath = Join-Path $dbBaseDir $DatabaseName

if (-not (Test-Path $buildDir)) {
    throw "Expected existing build directory was not found: $buildDir"
}

New-Item -ItemType Directory -Path $dbBaseDir -Force | Out-Null
New-Item -ItemType Directory -Path $tmpDir -Force | Out-Null

if ($Clean) {
    if (Test-Path $dbPath) {
        Remove-Item -Recurse -Force $dbPath
    }
}

$buildCmdFile = Join-Path $tmpDir "codeql-build-$($Config.ToLowerInvariant()).cmd"
$buildCmdLines = @(
    "@echo off",
    "setlocal",
    "call `"$vcvars`" x64",
    "if errorlevel 1 exit /b %errorlevel%",
    "call `"$oneApiSetvars`" intel64",
    "if errorlevel 1 exit /b %errorlevel%",
    "cmake --build `"$buildDir`" --config $Config --parallel $BuildParallel",
    "exit /b %errorlevel%"
)

Set-Content -Path $buildCmdFile -Value ($buildCmdLines -join "`r`n") -Encoding Ascii

$codeqlArgs = @(
    "database", "create", $dbPath,
    "--language=cpp",
    "--source-root", $repoRoot,
    "--command", $buildCmdFile,
    "--overwrite"
)

Write-Host "[ConvoPeq CodeQL] repoRoot: $repoRoot"
Write-Host "[ConvoPeq CodeQL] config  : $Config"
Write-Host "[ConvoPeq CodeQL] parallel: $BuildParallel"
Write-Host "[ConvoPeq CodeQL] buildTag: $buildTagSanitized"
Write-Host "[ConvoPeq CodeQL] dbPath  : $dbPath"
Write-Host "[ConvoPeq CodeQL] buildCmd: $buildCmdFile"
Write-Host "[ConvoPeq CodeQL] codeql  : $codeqlPath"

if ($DryRun) {
    Write-Host "[DryRun] Command preview:"
    Write-Host "`"$codeqlPath`" $($codeqlArgs -join ' ')"
    exit 0
}

& $codeqlPath @codeqlArgs
if ($LASTEXITCODE -ne 0) {
    throw "CodeQL database create failed (exit=$LASTEXITCODE)"
}

Write-Host "[ConvoPeq CodeQL] Database created: $dbPath"
