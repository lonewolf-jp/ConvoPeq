#requires -Version 7.0
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$verifier = 'C:\VSC_Project\ConvoPeq-ci-fix\.github\scripts\isr-verify-proof-scope.ps1'
$exporterSrc = 'C:\VSC_Project\ConvoPeq-ci-fix\src\audioengine\ISREvidenceExporter.cpp'
$runtimeSrc = 'C:\VSC_Project\ConvoPeq-ci-fix\src\audioengine\ISRDebugRuntime.cpp'
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("exporter-scope-" + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot | Out-Null

$exporter = Get-Content -Path $exporterSrc -Raw -Encoding UTF8
$runtime = Get-Content -Path $runtimeSrc -Raw -Encoding UTF8

# proofLevel の Release 判定を、解決できない条件コンパイルで包む
$wrapped = $exporter.Replace(
    '    if (mode == "Release") {
        return "off";
    }',
    '#if defined(CONVOPEQ_PROOF_LEVEL)
    if (mode == "Release") {
        return "off";
    }
#endif'
)
if ($wrapped -eq $exporter) { throw "exporter fixture not modified (anchor missing)" }

$root = Join-Path $tempRoot 'proof-level-conditional'
$ae = Join-Path $root 'src\audioengine'
New-Item -ItemType Directory -Path $ae -Force | Out-Null
Set-Content -Path (Join-Path $ae 'ISREvidenceExporter.cpp') -Value $wrapped -Encoding utf8NoBOM
Set-Content -Path (Join-Path $ae 'ISRDebugRuntime.cpp') -Value $runtime -Encoding utf8NoBOM

$out = & pwsh -NoProfile -ExecutionPolicy Bypass -File $verifier -Root $root 2>&1
$code = $LASTEXITCODE
Write-Host ("exporter proofLevel mapping inside #if defined(...) => exit={0}" -f $code)
if ($code -eq 0) { Write-Host "accepted (documented limit: exporter checks are not region-aware)" }
else { Write-Host "detected" }
Remove-Item -Recurse -Force $tempRoot
