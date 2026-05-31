$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$postPath = Join-Path $repoRoot 'storage\isr_inventory\post_authority_inventory.json'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'authority_classification_rule_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $postPath)) { throw "Missing inventory: $postPath" }

$post = Get-Content -LiteralPath $postPath -Raw -Encoding UTF8 | ConvertFrom-Json
$entries = @($post.entries)
$violations = New-Object 'System.Collections.Generic.List[string]'

foreach ($e in $entries) {
    $state = "$($e.state)"
    $class = "$($e.authority_class)"

    if ($state -match 'RuntimePublishWorld' -and $class -ne 'Diagnostic') {
        $violations.Add("RuntimePublishWorld must be Diagnostic: state=$state class=$class") | Out-Null
    }

    if (($state -match 'schemaVersion|generation|worldId|runtimeUuid|sampleRate') -and $class -ne 'Authoritative') {
        $violations.Add("State must be Authoritative: state=$state class=$class") | Out-Null
    }

    if (($state -match 'activeNode|current$') -and $class -ne 'Derived') {
        $violations.Add("State must be Derived: state=$state class=$class") | Out-Null
    }
}

$report = [ordered]@{ schema='authority_classification_rule_report_v1'; generatedAt=(Get-Date -Format 'o'); total=$entries.Count; violations=@($violations); ready=($violations.Count -eq 0) }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach($v in $violations){ Write-Host "[ERROR] $v" }; throw 'authority classification rule verification failed' }
Write-Host '[PASS] authority classification rule verification passed'
