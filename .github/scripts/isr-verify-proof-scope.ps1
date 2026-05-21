$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceExporterPath = Join-Path $repoRoot "src\audioengine\ISREvidenceExporter.cpp"
$debugRuntimePath = Join-Path $repoRoot "src\audioengine\ISRDebugRuntime.cpp"

if (-not (Test-Path $evidenceExporterPath)) {
    throw "ISREvidenceExporter.cpp not found: $evidenceExporterPath"
}
if (-not (Test-Path $debugRuntimePath)) {
    throw "ISRDebugRuntime.cpp not found: $debugRuntimePath"
}

$evidenceExporter = Get-Content -Path $evidenceExporterPath -Raw -Encoding UTF8
$debugRuntime = Get-Content -Path $debugRuntimePath -Raw -Encoding UTF8

# R25-1: Release proof off / Debug partial / CI full の静的契約確認
if ($evidenceExporter -notmatch 'if \(mode == "Release"\)\s*\{\s*return "off";\s*\}') {
    throw "Proof scope contract violation: Release must map to proofLevel=off"
}
if ($evidenceExporter -notmatch 'if \(mode == "Debug"\)\s*\{\s*return "partial";\s*\}') {
    throw "Proof scope contract violation: Debug must map to proofLevel=partial"
}
if ($evidenceExporter -notmatch 'return "full";') {
    throw "Proof scope contract violation: CI/default must map to proofLevel=full"
}

# R25-2: Release で minimal evidence のみを許容
if ($evidenceExporter -notmatch 'if \(isRelease\)\s*\{') {
    throw "Release evidence scope violation: isRelease block not found"
}
if ($evidenceExporter -notmatch 'minimalEvidence') {
    throw "Release evidence scope violation: minimalEvidence marker not found"
}
if ($evidenceExporter -notmatch 'if \(isRelease\)[\s\S]*?return;') {
    throw "Release evidence scope violation: early return for release path not found"
}

# R25-3: DebugRuntime 側で Release抑制・CIでHB emit を確認
if ($debugRuntime -notmatch 'if \(mode == "Release"\)\s*\{\s*return;\s*\}') {
    throw "DebugRuntime contract violation: Release short-circuit return not found"
}
if ($debugRuntime -notmatch 'if \(mode == "CI"\)\s*\{\s*emitHBTrace\(\);\s*\}') {
    throw "DebugRuntime contract violation: CI must emit HB trace"
}

Write-Host "[PASS] Proof scope verification (R25)"
