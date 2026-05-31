$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$scripts=Get-ChildItem -LiteralPath (Join-Path $repoRoot '.github\scripts') -File -Filter 'isr-verify-*.ps1' | Sort-Object Name
$manifestPath=Join-Path $repoRoot 'evidence\verifier_manifest_hash.txt'
$reportPath=Join-Path $repoRoot 'evidence\verifier_manifest_hash_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$payload=($scripts|ForEach-Object{$_.Name}) -join "`n"
$hash=[System.BitConverter]::ToString((New-Object System.Security.Cryptography.SHA256Managed).ComputeHash([System.Text.Encoding]::UTF8.GetBytes($payload))).Replace('-','').ToLowerInvariant()
if(-not(Test-Path -LiteralPath $manifestPath)){Set-Content -LiteralPath $manifestPath -Value $hash -Encoding UTF8}
$baseline=(Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8).Trim()
$violations=New-Object 'System.Collections.Generic.List[string]'
if($hash -ne $baseline){$violations.Add("Verifier manifest hash mismatch: baseline=$baseline actual=$hash")|Out-Null}
$report=[ordered]@{schema='verifier_manifest_hash_report_v1';generatedAt=(Get-Date -Format 'o');baseline=$baseline;actual=$hash;verifierCount=$scripts.Count;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'verifier manifest hash verification failed'}
Write-Host '[PASS] verifier manifest hash verification passed'
