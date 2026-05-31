$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\publication_state_machine.md'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'publication_state_machine_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'
function Add-Violation([string]$m) { $violations.Add($m) | Out-Null }

if (-not (Test-Path -LiteralPath $contractPath)) {
    Add-Violation "Missing contract: $contractPath"
}
else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($state in @('Draft', 'Publishing', 'Published', 'Retiring', 'Retired', 'Destroyed')) {
        if ($text -notmatch [regex]::Escape($state)) {
            Add-Violation "State not found in contract: $state"
        }
    }
    foreach ($edge in @('Draft -> Publishing', 'Publishing -> Published', 'Published -> Retiring', 'Retiring -> Retired', 'Retired -> Destroyed')) {
        if ($text -notmatch [regex]::Escape($edge)) {
            Add-Violation "Required transition missing: $edge"
        }
    }
}

$report = [ordered]@{
    schema = 'publication_state_machine_report_v1'; generatedAt = (Get-Date -Format 'o'); contractPath = $contractPath; violations = @($violations); ready = ($violations.Count -eq 0)
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw "publication state machine verification failed" }
Write-Host '[PASS] publication state machine verification passed'
