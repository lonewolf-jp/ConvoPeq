$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_identity_baseline_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$expected = @(
    'RuntimeGraph::sampleRate',
    'RuntimeGraph::activeNode',
    'EngineRuntime::current',
    'TransitionState::current',
    'RuntimeGraph::runtimeUuid',
    'BuilderToken::generation',
    'BuilderToken::worldId',
    'BuilderToken::RuntimePublishWorld',
    'BuilderToken::schemaVersion'
)

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$actual = @($post.entries | ForEach-Object { "$($_.state)" })
$missing = @($expected | Where-Object { $actual -notcontains $_ })
$unexpected = @($actual | Where-Object { $expected -notcontains $_ })
$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($m in $missing) { $violations.Add("Missing baseline state: $m") | Out-Null }
foreach ($u in $unexpected) { $violations.Add("Unexpected baseline state: $u") | Out-Null }

$report = [ordered]@{ schema='authority_identity_baseline_report_v1'; generatedAt=(Get-Date -Format 'o'); expected=@($expected); actual=@($actual); missing=@($missing); unexpected=@($unexpected); violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'authority identity baseline verification failed' }
Write-Host '[PASS] authority identity baseline verification passed'
