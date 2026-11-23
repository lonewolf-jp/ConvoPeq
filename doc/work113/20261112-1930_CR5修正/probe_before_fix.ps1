#requires -Version 7.0
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$verifier = 'C:\Users\user\AppData\Local\Temp\opencode\oldverifier2\.github\scripts\isr-verify-proof-scope.ps1'
$exporter = 'C:\VSC_Project\ConvoPeq-ci-fix\src\audioengine\ISREvidenceExporter.cpp'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("crprobe-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null

function New-Fixture([string]$Name, [string]$Src) {
    $root = Join-Path $tempRoot $Name
    $ae = Join-Path $root 'src\audioengine'
    New-Item -ItemType Directory -Path $ae -Force | Out-Null
    Copy-Item $exporter (Join-Path $ae 'ISREvidenceExporter.cpp')
    Set-Content (Join-Path $ae 'ISRDebugRuntime.cpp') -Value $Src -Encoding utf8NoBOM
    return $root
}

$tail = @'

void DebugRuntime::emitFinalViolationReport()
{
    const auto path = std::filesystem::current_path() / "evidence" / "hb_violation_report.json";
    hbVerifier_.setTraceRuntime(&hbTraceRuntime_);
    hbVerifier_.emitViolationReport(path);
}
} // namespace convo::isr
'@

$head = @'
namespace convo::isr
{
bool isForceEvidenceEmission()
{
    return false;
}
'@

$goodTrace = @'
void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    hbTraceRuntime_.emitTrace(path);
}
'@

# A) disabled #if 0 definition (valid) comes first, active definition has no guard and no trace call
$cases = @()
$cases += [PSCustomObject]@{ Name='A-if0-disabled-definition'; Expected=1; Source = $head + @'

#if 0
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}
#endif

void DebugRuntime::emitCIArtifacts()
{
}

'@ + $goodTrace + $tail }

# B) lowercase literal
$cases += [PSCustomObject]@{ Name='B-lowercase-release-literal'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

'@ + $goodTrace + $tail }

# C) guard nested in a block that never runs
$cases += [PSCustomObject]@{ Name='C-nested-guard'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (false) {
        if (mode == "Release" && ! isForceEvidenceEmission()) {
            return;
        }
    }
    emitHBTrace();
}

'@ + $goodTrace + $tail }

# D) trace call reachable only in Release
$cases += [PSCustomObject]@{ Name='D-trace-in-release-block'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    if (mode == "Release") {
        emitHBTrace();
    }
}

'@ + $goodTrace + $tail }

# E) emitHBTrace has the guard but no emitTrace call
$cases += [PSCustomObject]@{ Name='E-emitTrace-missing'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    emitHBTrace();
}

void DebugRuntime::emitHBTrace()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
}

'@ + $tail }

foreach ($c in $cases) {
    $root = New-Fixture $c.Name $c.Source
    $out = & pwsh -NoProfile -ExecutionPolicy Bypass -File $verifier -Root $root 2>&1
    $code = $LASTEXITCODE
    $verdict = if ($code -eq $c.Expected) { 'DETECTED' } else { 'MISSED (finding valid)' }
    Write-Host ("{0,-30} exit={1} expected={2} => {3}" -f $c.Name, $code, $c.Expected, $verdict)
}
Remove-Item -Recurse -Force $tempRoot
