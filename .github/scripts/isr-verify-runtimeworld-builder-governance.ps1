$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$contractPath = Join-Path $repoRoot 'doc\work10\contracts\runtimeworld_builder_governance.md'
$headerPath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtimeworld_builder_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $contractPath)) {
    $violations.Add("Missing contract: $contractPath") | Out-Null
} else {
    $text = Get-Content -LiteralPath $contractPath -Raw -Encoding UTF8
    foreach ($token in @('BuilderToken', 'RFC', 'NonRT', 'publish(RuntimeWorld*)')) {
        if ($text -notmatch [regex]::Escape($token)) {
            $violations.Add("Builder governance contract missing token: $token") | Out-Null
        }
    }
}

if (-not (Test-Path -LiteralPath $headerPath)) {
    $violations.Add("Missing source header: $headerPath") | Out-Null
} else {
    $h = Get-Content -LiteralPath $headerPath -Raw -Encoding UTF8
    if ($h -notmatch 'buildRuntimePublishWorld\s*\(') {
        $violations.Add('buildRuntimePublishWorld(...) declaration not found in AudioEngine.h') | Out-Null
    }
}

$report = [ordered]@{ schema='runtimeworld_builder_governance_report_v1'; generatedAt=(Get-Date -Format 'o'); contractPath=$contractPath; headerPath=$headerPath; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){Write-Host "[ERROR] $v"}; throw 'runtimeworld builder governance verification failed' }
Write-Host '[PASS] runtimeworld builder governance verification passed'
