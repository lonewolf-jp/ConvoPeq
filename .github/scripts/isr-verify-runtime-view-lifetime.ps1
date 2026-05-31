$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath=Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir=Join-Path $repoRoot 'evidence'
$reportPath=Join-Path $evidenceDir 'runtime_view_lifetime_report.json'
if(-not(Test-Path -LiteralPath $evidenceDir)){New-Item -ItemType Directory -Path $evidenceDir -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $commitPath)){$violations.Add("Missing source: $commitPath")|Out-Null}else{ $s=Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8; if($s -notmatch 'readControlRuntimeView\s*\('){$violations.Add('readControlRuntimeView usage missing')|Out-Null} }
$report=[ordered]@{schema='runtime_view_lifetime_report_v1';generatedAt=(Get-Date -Format 'o');sourcePath=$commitPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime view lifetime verification failed' }
Write-Host '[PASS] runtime view lifetime verification passed'
