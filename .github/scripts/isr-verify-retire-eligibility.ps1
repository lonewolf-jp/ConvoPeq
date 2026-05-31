$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath=Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$reportPath=Join-Path $repoRoot 'evidence\retire_eligibility_report.json'
if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(Test-Path -LiteralPath $commitPath){$txt=Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8; foreach($token in @('canTransitionRetirePendingToFree','graceCompleted','authoritativeOwnershipReleased')){ if($txt -notmatch [regex]::Escape($token)){ $violations.Add("Retire eligibility token missing: $token")|Out-Null } }} else { $violations.Add("Missing source: $commitPath")|Out-Null }
$report=[ordered]@{schema='retire_eligibility_report_v1';generatedAt=(Get-Date -Format 'o');sourcePath=$commitPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'retire eligibility verification failed'}
Write-Host '[PASS] retire eligibility verification passed'
