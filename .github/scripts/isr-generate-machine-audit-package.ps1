$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir=Join-Path $repoRoot 'evidence'
$outPath=Join-Path $evidenceDir 'machine_generated_audit_package.json'
if(-not(Test-Path -LiteralPath $evidenceDir)){New-Item -ItemType Directory -Path $evidenceDir -Force|Out-Null}
$files=Get-ChildItem -LiteralPath $evidenceDir -File | Select-Object -ExpandProperty Name
$payload=[ordered]@{schema='machine_generated_audit_package_v1';generatedAt=(Get-Date -Format 'o');fileCount=@($files).Count;files=@($files)}
$payload|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Host "[INFO] generated audit package index: $outPath"
Write-Host '[PASS] machine audit package generation passed'
