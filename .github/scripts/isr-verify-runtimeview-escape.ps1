$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$audioRoot=Join-Path $repoRoot 'src\audioengine'
$evidenceDir=Join-Path $repoRoot 'evidence'
$reportPath=Join-Path $evidenceDir 'runtimeview_escape_report.json'
if(-not(Test-Path -LiteralPath $evidenceDir)){New-Item -ItemType Directory -Path $evidenceDir -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
$hits=New-Object 'System.Collections.Generic.List[string]'
$files=Get-ChildItem -LiteralPath $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc
foreach($f in $files){ $txt=Get-Content -LiteralPath $f.FullName -Raw -Encoding UTF8; if([regex]::IsMatch($txt,'RuntimeReadView\s*\*')){ $hits.Add($f.FullName)|Out-Null } }
if($hits.Count -gt 0){ foreach($h in $hits){ $violations.Add("Potential RuntimeView escape pattern (pointer) detected: $h")|Out-Null } }
$report=[ordered]@{schema='runtimeview_escape_report_v1';generatedAt=(Get-Date -Format 'o');sourceRoot=$audioRoot;hitCount=$hits.Count;hits=$hits.ToArray();violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeview escape verification failed' }
Write-Host '[PASS] runtimeview escape verification passed'
