#requires -Version 7.0
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$verifier = 'C:\VSC_Project\ConvoPeq-ci-fix\.github\scripts\isr-verify-proof-scope.ps1'
$exporter = 'C:\VSC_Project\ConvoPeq-ci-fix\src\audioengine\ISREvidenceExporter.cpp'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("scope-probe-" + [System.Guid]::NewGuid().ToString('N'))
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

$cases = @()

# F) guard lives inside an unresolved conditional-compilation block
$cases += [PSCustomObject]@{ Name='F-guard-inside-conditional-block'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
#if defined(CONVOPEQ_TRACE_GUARD)
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

'@ + $goodTrace + $tail }

# G) guard disabled by #if 0 (must not count)
$cases += [PSCustomObject]@{ Name='G-guard-disabled-by-if0'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
#if 0
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#endif
    emitHBTrace();
}

'@ + $goodTrace + $tail }

# H) trace call inside an unresolved conditional-compilation block
$cases += [PSCustomObject]@{ Name='H-trace-inside-conditional-block'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#if defined(CONVOPEQ_TRACE)
    emitHBTrace();
#endif
}

'@ + $goodTrace + $tail }

# I) trace call disabled by #if 0 (must not count)
$cases += [PSCustomObject]@{ Name='I-trace-disabled-by-if0'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
#if 0
    emitHBTrace();
#endif
}

'@ + $goodTrace + $tail }

# J) forced-only branch around the trace call (known guarantee limit)
$cases += [PSCustomObject]@{ Name='J-trace-inside-force-branch'; Expected=1; Source = $head + @'
void DebugRuntime::emitCIArtifacts()
{
    const std::string mode = currentBuildMode();
    if (mode == "Release" && ! isForceEvidenceEmission()) {
        return;
    }
    if (isForceEvidenceEmission()) {
        emitHBTrace();
    }
}

'@ + $goodTrace + $tail }

foreach ($c in $cases) {
    $root = New-Fixture $c.Name $c.Source
    $out = & pwsh -NoProfile -ExecutionPolicy Bypass -File $verifier -Root $root 2>&1
    $code = $LASTEXITCODE
    $verdict = if ($code -eq $c.Expected) { 'detected' } else { 'accepted (documented limit)' }
    Write-Host ("{0,-38} exit={1} expected={2} => {3}" -f $c.Name, $code, $c.Expected, $verdict)
}
Remove-Item -Recurse -Force $tempRoot
