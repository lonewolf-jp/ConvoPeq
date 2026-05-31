$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_closure_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$allowScript = Join-Path $PSScriptRoot 'isr-verify-semantic-closure-allowlist.ps1'
$forbiddenScript = Join-Path $PSScriptRoot 'isr-verify-semantic-closure-forbidden-inputs.ps1'

foreach ($p in @($allowScript, $forbiddenScript)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "Missing dependent verifier: $p" }
}

& $allowScript
& $forbiddenScript

$report = [ordered]@{ schema='semantic_closure_report_v1'; generatedAt=(Get-Date -Format 'o'); checks=@('allowlist','forbidden-inputs'); ready=$true }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
Write-Host '[PASS] semantic closure verification passed'
