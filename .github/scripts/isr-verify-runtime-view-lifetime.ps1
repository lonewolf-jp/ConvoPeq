$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath=Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir=Join-Path $repoRoot 'evidence'
$reportPath=Join-Path $evidenceDir 'runtime_view_lifetime_report.json'
if(-not(Test-Path -LiteralPath $evidenceDir)){New-Item -ItemType Directory -Path $evidenceDir -Force|Out-Null}
$violations=New-Object 'System.Collections.Generic.List[string]'
# ★ work94 sync (D170 handle 一本路・9e0db5f7 2026-06-02 系): readControlRuntimeView は削除され、
#   control 側 lifetime-safe read は worldAuthority_.consumeWorldHandle()（Commit.cpp:587）に収束。
#   検証意図（commit 経路の control read が handle-scoped であること）は同一。
if(-not(Test-Path -LiteralPath $commitPath)){$violations.Add("Missing source: $commitPath")|Out-Null}else{ $s=Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8; if($s -notmatch 'worldAuthority_\.consumeWorldHandle\s*\('){$violations.Add('handle-scoped control runtime read (worldAuthority_.consumeWorldHandle) missing')|Out-Null} }
$report=[ordered]@{schema='runtime_view_lifetime_report_v1';generatedAt=(Get-Date -Format 'o');sourcePath=$commitPath;violations=@($violations);ready=($violations.Count -eq 0)}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"; if($violations.Count -gt 0){ foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime view lifetime verification failed' }
Write-Host '[PASS] runtime view lifetime verification passed'
