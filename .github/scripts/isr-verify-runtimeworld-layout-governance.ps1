$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_layout_governance.md'
$abiContractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_abi_contract.md'
$serializationContractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_serialization_contract.md'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtimeworld_layout_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($required in @($contractPath, $abiContractPath, $serializationContractPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        $violations.Add("Missing required contract: $required") | Out-Null
    }
}

if (Test-Path -LiteralPath $contractPath) {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('Top-level RuntimeWorld field additions require RFC', 'Field order', 'type', 'migration plan')) {
        if ($text -notmatch [regex]::Escape($token)) {
            $violations.Add("Layout governance contract missing token: $token") | Out-Null
        }
    }
}

$report = [ordered]@{ schema='runtimeworld_layout_governance_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; abiContractPath=$abiContractPath; serializationContractPath=$serializationContractPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeworld layout governance verification failed' }
Write-Host '[PASS] runtimeworld layout governance verification passed'
