$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtime_semantic_transition_graph.md'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtime_semantic_transition_graph_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'
function Add-Violation([string]$m) { $violations.Add($m) | Out-Null }

if (-not (Test-Path -LiteralPath $contractPath)) {
    Add-Violation "Missing contract: $contractPath"
}
else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('Nodes', 'Edge Rules', 'Conditional edges', 'Invalid Patterns')) {
        if ($text -notmatch [regex]::Escape($token)) {
            Add-Violation "Transition graph contract missing token: $token"
        }
    }
}

$report = [ordered]@{ schema = 'runtime_semantic_transition_graph_report_v1'; generatedAt = (Get-Date -Format 'o'); contractPath = $contractPath; violations = @($violations); ready = ($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw "runtime semantic transition graph verification failed" }
Write-Host '[PASS] runtime semantic transition graph verification passed'
