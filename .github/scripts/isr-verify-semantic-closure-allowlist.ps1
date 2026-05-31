$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\semantic_closure_allowed_external_inputs.md'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_closure_allowlist_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $contractPath)) {
    $violations.Add("Missing allowlist contract: $contractPath") | Out-Null
} else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('RuntimeWorld semantic fields','Publication metadata','diagnostic counters','allowlist')) {
        if ($text -notmatch [regex]::Escape($token)) {
            $violations.Add("Allowlist contract missing token: $token") | Out-Null
        }
    }
}

if (-not (Test-Path -LiteralPath $commitPath)) {
    $violations.Add("Missing commit file: $commitPath") | Out-Null
} else {
    $commit = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
    if ($commit -notmatch 'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)') {
        $violations.Add('Semantic precheck signature missing const RuntimePublishWorld& world') | Out-Null
    }
}

$report = [ordered]@{ schema='semantic_closure_allowlist_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; commitPath=$commitPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'semantic closure allowlist verification failed' }
Write-Host '[PASS] semantic closure allowlist verification passed'
