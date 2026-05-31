$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$baselinePath = Join-Path $repoRoot 'storage\isr_inventory\authority_drift_hash_baseline.txt'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_drift_hash_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$lines = @($post.entries | ForEach-Object { "$($_.state)|$($_.authority_class)|$($_.owner)|$($_.publication_path)|$($_.observe_path)" } | Sort-Object)
$joined = [string]::Join("`n", $lines)
$hash = [System.BitConverter]::ToString((New-Object System.Security.Cryptography.SHA256Managed).ComputeHash([System.Text.Encoding]::UTF8.GetBytes($joined))).Replace('-','').ToLowerInvariant()

$violations = New-Object 'System.Collections.Generic.List[string]'
$warnings = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $baselinePath)) {
    Set-Content -LiteralPath $baselinePath -Value $hash -Encoding UTF8
    $warnings.Add("Baseline created: $baselinePath") | Out-Null
}

$baseline = (Get-Content -LiteralPath $baselinePath -Raw -Encoding UTF8).Trim()
if ($hash -ne $baseline) {
    $violations.Add("Authority drift hash mismatch: baseline=$baseline actual=$hash") | Out-Null
}

$report = [ordered]@{ schema='authority_drift_hash_report_v1'; generatedAt=(Get-Date -Format 'o'); baselinePath=$baselinePath; baseline=$baseline; actual=$hash; warnings=@($warnings); violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
foreach ($w in $warnings) { Write-Host "[WARN] $w" }
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'authority drift hash verification failed' }
Write-Host '[PASS] authority drift hash verification passed'
