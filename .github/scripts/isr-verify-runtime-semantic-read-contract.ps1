$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtime_semantic_read_contract.md'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtime_semantic_read_contract_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $contractPath)) {
    $violations.Add("Missing contract: $contractPath") | Out-Null
} else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    if ($text -notmatch 'RuntimePublishWorld') { $violations.Add('Runtime semantic read contract missing token: RuntimePublishWorld') | Out-Null }
    if ($text -notmatch 'forbidden') { $violations.Add('Runtime semantic read contract missing token: forbidden') | Out-Null }
    if (($text -notmatch 'RuntimeGraph') -or ($text -notmatch 'forbidden')) {
        $violations.Add('Runtime semantic read contract missing RuntimeGraph-forbidden rule') | Out-Null
    }
    if ($text -notmatch 'Allowed Sources') { $violations.Add('Runtime semantic read contract missing token: Allowed Sources') | Out-Null }
}

if (Test-Path -LiteralPath $commitPath) {
    $src = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
    if (-not [regex]::IsMatch($src, 'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)')) {
        $violations.Add('Semantic read entrypoint signature missing') | Out-Null
    }
} else {
    $violations.Add("Missing source: $commitPath") | Out-Null
}

$report = [ordered]@{ schema='runtime_semantic_read_contract_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; sourcePath=$commitPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtime semantic read contract verification failed' }
Write-Host '[PASS] runtime semantic read contract verification passed'
