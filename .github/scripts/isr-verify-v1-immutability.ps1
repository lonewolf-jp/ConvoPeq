$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runtimeBuilderSource = Join-Path $repoRoot "src\audioengine\RuntimeBuilder.cpp"
$audioEngineCommit = Join-Path $repoRoot "src\audioengine\AudioEngine.Commit.cpp"

if (-not (Test-Path $runtimeBuilderSource)) {
    throw "Missing file: $runtimeBuilderSource"
}

if (-not (Test-Path $audioEngineCommit)) {
    throw "Missing file: $audioEngineCommit"
}

$builderText = Get-Content -LiteralPath $runtimeBuilderSource -Raw -Encoding UTF8
$commitText = Get-Content -LiteralPath $audioEngineCommit -Raw -Encoding UTF8

if ($builderText -notmatch 'worldOwner->freeze\s*\(\s*\)\s*;') {
    throw "Freeze call not found in publish builder: $runtimeBuilderSource"
}

if ($commitText -notmatch 'world\.isSealedRecursively\s*\(\s*\)') {
    throw "Recursive seal precheck not found in publish precheck: $audioEngineCommit"
}

if ($commitText -notmatch 'world\.isFrozen\s*\(\s*\)') {
    throw "Frozen precheck not found in publish precheck: $audioEngineCommit"
}

Write-Host "[PASS] R1 immutability freeze/seal precheck verified"
