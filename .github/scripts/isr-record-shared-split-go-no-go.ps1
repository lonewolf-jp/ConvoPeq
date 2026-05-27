param(
    [string]$Decision = 'Pending',
    [string]$Reason = 'Awaiting shared/split comparison evidence',
    [string]$Ticket = '',
    [string]$OutputPath = (Join-Path $PSScriptRoot "..\..\doc\work\ISR_Shared_EpochDomain_GoNoGo_2026-05-27.md")
)

$ErrorActionPreference = 'Stop'

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# ISR Shared EpochDomain Go/No-Go Record') | Out-Null
$lines.Add('') | Out-Null
$lines.Add(('作成日: {0}' -f (Get-Date -Format 'yyyy-MM-dd'))) | Out-Null
$lines.Add('対象: R10 shared / split migration 判定記録') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('---') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## 1. 判定') | Out-Null
$lines.Add('') | Out-Null
$lines.Add(('- Decision: {0}' -f $Decision)) | Out-Null
$lines.Add(('- Reason: {0}' -f $Reason)) | Out-Null
$lines.Add(('- Ticket: {0}' -f ($(if ([string]::IsNullOrWhiteSpace($Ticket)) { '未設定' } else { $Ticket })))) | Out-Null
$lines.Add('') | Out-Null
$lines.Add('## 2. 参考') | Out-Null
$lines.Add('') | Out-Null
$lines.Add('- `doc/work/ISR_Shared_EpochDomain_Scalability_Validation_Plan.md`') | Out-Null
$lines.Add('- `doc/work/ISR_Shared_EpochDomain_SplitMigration_Runbook_2026-05-27.md`') | Out-Null
$lines.Add('- `doc/work/ISR_Shared_EpochDomain_Shared_vs_Split_Comparison_2026-05-27.md`') | Out-Null

Set-Content -LiteralPath $OutputPath -Value ($lines -join "`n") -Encoding UTF8
Write-Host "[PASS] go/no-go record written: $OutputPath"
