$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtime_semantic_reachability.md'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_reachability_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'
function Add-Violation([string]$m){ $violations.Add($m) | Out-Null }

if (-not (Test-Path -LiteralPath $contractPath)) {
    Add-Violation "Missing contract: $contractPath"
}
else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($state in @('CrossfadeComplete','RetireSettled','PublicationStable')) {
        if ($text -notmatch [regex]::Escape($state)) {
            Add-Violation "Reachability target missing: $state"
        }
    }
    foreach ($token in @('reachable','dead-end')) {
        if ($text -notmatch $token) {
            Add-Violation "Reachability contract missing token: $token"
        }
    }
}

$report = [ordered]@{ schema='semantic_reachability_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw "semantic reachability verification failed" }
Write-Host '[PASS] semantic reachability verification passed'
