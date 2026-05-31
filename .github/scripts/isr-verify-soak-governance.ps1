$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$policyPath = Join-Path $repoRoot '.github\isr-pr-sla-policy.json'
$tierRunnerPath = Join-Path $repoRoot '.github\scripts\isr-run-tiered-verification.ps1'
$workflowPath = Join-Path $repoRoot '.github\workflows\isr-verification.yml'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'soak_governance_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'
$checks = [ordered]@{
    policyExists                    = $false
    policySchemaOk                  = $false
    requiredSoakNotesPresent        = $false
    tierRunnerForwardsSoakMinutes   = $false
    workflowDispatchHasSoakInput    = $false
    workflowForwardsSoakMinutes     = $false
    workflowScheduleHasSoakDefaults = $false
}

if (-not (Test-Path -LiteralPath $policyPath)) {
    $violations.Add("Missing PR SLA policy: $policyPath") | Out-Null
}
else {
    $checks.policyExists = $true
    $policy = Get-Content -LiteralPath $policyPath -Raw -Encoding UTF8 | ConvertFrom-Json

    if ("$($policy.schema)" -ne 'isr_pr_sla_policy_v1') {
        $violations.Add("Unexpected PR SLA policy schema: $($policy.schema)") | Out-Null
    }
    else {
        $checks.policySchemaOk = $true
    }

    $requiredNotes = @()
    if ($null -ne $policy.classes) {
        foreach ($entry in @($policy.classes)) {
            $notes = @($entry.requiredNotes)
            foreach ($note in $notes) {
                $requiredNotes += [string]$note
            }
        }
    }

    $hasMediumSoak = $requiredNotes -contains 'soak medium 24h'
    $hasLongSoak = $requiredNotes -contains 'soak long 72h'
    $hasExtremeSoak = $requiredNotes -contains 'soak extreme 1week'
    if ($hasMediumSoak -and $hasLongSoak -and $hasExtremeSoak) {
        $checks.requiredSoakNotesPresent = $true
    }
    else {
        $violations.Add('PR SLA policy must include required soak notes: soak medium 24h / soak long 72h / soak extreme 1week') | Out-Null
    }
}

if (-not (Test-Path -LiteralPath $tierRunnerPath)) {
    $violations.Add("Missing tier runner: $tierRunnerPath") | Out-Null
}
else {
    $tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
    if ($tierRunnerText.Contains('[int]$SoakMinutes') -and $tierRunnerText.Contains('SoakMinutes = $SoakMinutes')) {
        $checks.tierRunnerForwardsSoakMinutes = $true
    }
    else {
        $violations.Add('Tier runner must forward SoakMinutes to isr-verify-pr-sla.ps1') | Out-Null
    }
}

if (-not (Test-Path -LiteralPath $workflowPath)) {
    $violations.Add("Missing workflow: $workflowPath") | Out-Null
}
else {
    $workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8

    if ($workflowText.Contains('soakMinutes:')) {
        $checks.workflowDispatchHasSoakInput = $true
    }
    else {
        $violations.Add('Workflow dispatch input missing: soakMinutes') | Out-Null
    }

    if ($workflowText.Contains("'-SoakMinutes'")) {
        $checks.workflowForwardsSoakMinutes = $true
    }
    else {
        $violations.Add('Workflow must forward soakMinutes into tier runner arguments') | Out-Null
    }

    $scheduleSoakDefaultsReady =
    $workflowText.Contains('$workflowDispatchSoakMinutes = 10080') -and
    $workflowText.Contains('$workflowDispatchSoakMinutes = 1440') -and
    $workflowText.Contains('github.event_name }}" -eq ''workflow_dispatch'' -or "${{ github.event_name }}" -eq ''schedule''')

    if ($scheduleSoakDefaultsReady) {
        $checks.workflowScheduleHasSoakDefaults = $true
    }
    else {
        $violations.Add('Workflow schedule path must set soak defaults (nightly=1440, weekly=10080) and forward SoakMinutes') | Out-Null
    }
}

$report = [ordered]@{
    schema         = 'soak_governance_report_v1'
    generatedAt    = (Get-Date -Format 'o')
    policyPath     = $policyPath
    tierRunnerPath = $tierRunnerPath
    workflowPath   = $workflowPath
    checks         = $checks
    violations     = @($violations)
    ready          = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[ERROR] $v" }
    throw 'soak governance verification failed'
}

Write-Host '[PASS] soak governance verification passed'
