$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runtimeGraphHeader = Join-Path $repoRoot "src\audioengine\RuntimeGraph.h"
$audioEngineHeader = Join-Path $repoRoot "src\audioengine\AudioEngine.h"

if (-not (Test-Path $runtimeGraphHeader)) {
    throw "Missing file: $runtimeGraphHeader"
}

if (-not (Test-Path $audioEngineHeader)) {
    throw "Missing file: $audioEngineHeader"
}

$runtimeGraphText = Get-Content -LiteralPath $runtimeGraphHeader -Raw -Encoding UTF8
$audioEngineText = Get-Content -LiteralPath $audioEngineHeader -Raw -Encoding UTF8

$forbiddenRuntimeGraphTokens = @(
    'std::mutex',
    'std::shared_mutex',
    'std::unique_ptr',
    'std::shared_ptr',
    'std::vector',
    'std::function',
    'mutable ',
    'freeze\s*\(',
    'seal\s*\('
)

foreach ($token in $forbiddenRuntimeGraphTokens) {
    if ($runtimeGraphText -match $token) {
        throw "RuntimeGraph immutability violation: forbidden token '$token' found in $runtimeGraphHeader"
    }
}

if ($audioEngineText -notmatch 'inline\s+const convo::RuntimeGraph\*\s+getRuntimeGraph\s*\(') {
    throw "RuntimeGraph accessor must remain const-only: $audioEngineHeader"
}

if ($audioEngineText -match 'runtimeExecutionView\.graph(?!\s*\.)') {
    throw "Direct runtimeExecutionView.graph access detected in $audioEngineHeader"
}

Write-Host "[PASS] R2 runtime graph immutability verified"
