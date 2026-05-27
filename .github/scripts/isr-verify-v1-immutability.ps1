$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioEngineHeader = Join-Path $repoRoot "src\audioengine\AudioEngine.h"
$audioEngineCommit = Join-Path $repoRoot "src\audioengine\AudioEngine.Commit.cpp"

if (-not (Test-Path $audioEngineHeader)) {
    throw "Missing file: $audioEngineHeader"
}

if (-not (Test-Path $audioEngineCommit)) {
    throw "Missing file: $audioEngineCommit"
}

$headerText = Get-Content -LiteralPath $audioEngineHeader -Raw -Encoding UTF8
$commitText = Get-Content -LiteralPath $audioEngineCommit -Raw -Encoding UTF8

if ($headerText -notmatch 'worldOwner->freeze\s*\(\s*\)\s*;') {
    throw "Freeze call not found in publish builder: $audioEngineHeader"
}

if ($commitText -notmatch 'world\.isSealedRecursively\s*\(\s*\)') {
    throw "Recursive seal precheck not found in publish precheck: $audioEngineCommit"
}

Write-Host "[PASS] R1 immutability freeze/seal precheck verified"
