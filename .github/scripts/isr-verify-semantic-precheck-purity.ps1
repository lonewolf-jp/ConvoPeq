$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_precheck_purity_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $commitPath)) { throw "Missing commit source: $commitPath" }

$source = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
$violations = New-Object 'System.Collections.Generic.List[string]'

$m = [regex]::Match($source, 'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)\s*noexcept\s*\{(?<body>.*?)\n\}', [System.Text.RegularExpressions.RegexOptions]::Singleline)
if (-not $m.Success) {
    $violations.Add('Unable to locate runPublicationPrecheckNonRt body') | Out-Null
} else {
    $body = $m.Groups['body'].Value

    foreach ($forbidden in @('runtimeStore\s*\.', 'thread_local', 'MessageManager', 'std::mutex', 'std::lock_guard')) {
        if ([regex]::IsMatch($body, $forbidden)) {
            $violations.Add("Forbidden dependency in semantic precheck body: $forbidden") | Out-Null
        }
    }

    if (-not [regex]::IsMatch($body, 'world\.isFrozen\s*\(\s*\)')) {
        $violations.Add('Precheck purity requirement missing: world.isFrozen()') | Out-Null
    }
    if (-not [regex]::IsMatch($body, 'world\.isSealedRecursively\s*\(\s*\)')) {
        $violations.Add('Precheck purity requirement missing: world.isSealedRecursively()') | Out-Null
    }
}

$report = [ordered]@{ schema='semantic_precheck_purity_report_v1'; generatedAt=(Get-Date -Format 'o'); source=$commitPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'semantic precheck purity verification failed' }
Write-Host '[PASS] semantic precheck purity verification passed'
