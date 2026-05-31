$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_builder_internal_mutation_governance.md'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtimeworld_builder_internal_mutation_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $contractPath)) {
    $violations.Add("Missing contract: $contractPath") | Out-Null
} else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('freeze', 'sealRecursively', 'RFC', 'Forbidden Patterns')) {
        if ($text -notmatch [regex]::Escape($token)) {
            $violations.Add("Builder internal mutation contract missing token: $token") | Out-Null
        }
    }
}

if (-not (Test-Path -LiteralPath $commitPath)) {
    $violations.Add("Missing source: $commitPath") | Out-Null
} else {
    $src = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
    if (-not [regex]::IsMatch($src, 'world\.isFrozen\s*\(\s*\)')) { $violations.Add('Missing world.isFrozen() precheck in commit path') | Out-Null }
    if (-not [regex]::IsMatch($src, 'world\.isSealedRecursively\s*\(\s*\)')) { $violations.Add('Missing world.isSealedRecursively() precheck in commit path') | Out-Null }
}

$report = [ordered]@{ schema='runtimeworld_builder_internal_mutation_governance_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; sourcePath=$commitPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeworld builder internal mutation governance verification failed' }
Write-Host '[PASS] runtimeworld builder internal mutation governance verification passed'
