$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\semantic_closure_forbidden_inputs.md'
$commitPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'semantic_closure_forbidden_inputs_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $contractPath)) {
    $violations.Add("Missing forbidden-inputs contract: $contractPath") | Out-Null
} else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('EngineRuntime direct state','RuntimeGraph direct authority fields','AudioEngine mutable globals/atomics','DSPCore internal mutable state','Thread-local runtime decisions')) {
        if ($text -notmatch [regex]::Escape($token)) {
            $violations.Add("Forbidden-inputs contract missing token: $token") | Out-Null
        }
    }
}

if (-not (Test-Path -LiteralPath $commitPath)) {
    $violations.Add("Missing commit file: $commitPath") | Out-Null
} else {
    $commit = Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8
    $match = [regex]::Match($commit, 'runPublicationPrecheckNonRt\(const RuntimePublishWorld& world\)\s*noexcept\s*\{(?<body>.*?)\n\}', [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $match.Success) {
        $violations.Add('Failed to locate precheck body for semantic closure scan') | Out-Null
    } else {
        $body = $match.Groups['body'].Value
        foreach ($pattern in @('runtimeStore\s*\.', 'EngineRuntime\s*::', 'thread_local')) {
            if ([regex]::IsMatch($body, $pattern)) {
                $violations.Add("Forbidden semantic precheck input pattern detected: $pattern") | Out-Null
            }
        }
    }
}

$report = [ordered]@{ schema='semantic_closure_forbidden_inputs_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; commitPath=$commitPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'semantic closure forbidden-inputs verification failed' }
Write-Host '[PASS] semantic closure forbidden-inputs verification passed'
