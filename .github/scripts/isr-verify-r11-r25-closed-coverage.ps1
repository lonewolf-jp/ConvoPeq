$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$workflowPath = Join-Path $repoRoot ".github\workflows\isr-verification.yml"
$tierRunnerPath = Join-Path $repoRoot ".github\scripts\isr-run-tiered-verification.ps1"
$strategyDocPath = Join-Path $repoRoot "doc\work\ISR_R11-R25_GateCoverage_Strategy_2026-05-27.md"
$backlogPath = Join-Path $repoRoot "doc\work\ISR_Completeness_Risk_Backlog.md"

foreach ($path in @($workflowPath, $tierRunnerPath, $strategyDocPath, $backlogPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

$workflowText = Get-Content -LiteralPath $workflowPath -Raw -Encoding UTF8
$tierRunnerText = Get-Content -LiteralPath $tierRunnerPath -Raw -Encoding UTF8
$strategyText = Get-Content -LiteralPath $strategyDocPath -Raw -Encoding UTF8
$backlogText = Get-Content -LiteralPath $backlogPath -Raw -Encoding UTF8

$requiredClosedSections = 11..25 | ForEach-Object { "## R$_." }
foreach ($section in $requiredClosedSections) {
    if ($backlogText -notmatch [regex]::Escape($section)) {
        throw "Backlog missing R-section: $section"
    }
}

$requiredWorkflowInvocations = @(
    '.github/scripts/isr-verify-v3.ps1',
    '.github/scripts/isr-verify-v4.ps1',
    '.github/scripts/isr-verify-v5.ps1',
    '.github/scripts/isr-verify-v6.ps1',
    '.github/scripts/isr-verify-v7.ps1',
    '.github/scripts/isr-verify-v8.ps1',
    '.github/scripts/isr-verify-v10.ps1',
    '.github/scripts/isr-verify-v10-ownership-cycle.ps1',
    '.github/scripts/isr-verify-p3-governance.ps1',
    '.github/scripts/isr-verify-runtime-reduction-gate.ps1',
    '.github/scripts/isr-verify-proof-scope.ps1',
    '.github/scripts/check-list-compliance.ps1',
    '.github/scripts/isr-verify-r11-r25-closed-coverage.ps1'
)

foreach ($scriptRef in $requiredWorkflowInvocations) {
    $scriptPath = Join-Path $repoRoot $scriptRef
    if (-not (Test-Path $scriptPath)) {
        throw "Missing script file required for R11-R25 coverage: $scriptRef"
    }

    $workflowCallPattern = [regex]::Escape("& ./$scriptRef")
    $tierRunnerPattern = [regex]::Escape("'$scriptRef'")
    $wiredInWorkflow = [regex]::IsMatch($workflowText, $workflowCallPattern)
    $wiredInTierRunner = [regex]::IsMatch($tierRunnerText, $tierRunnerPattern)

    if (-not $wiredInWorkflow -and -not $wiredInTierRunner) {
        throw "Workflow/tier runner missing invocation required for R11-R25 coverage: $scriptRef"
    }
}

$requiredStrategyTokens = @(
    'R11 Closure Descriptor',
    'R15 Shutdown FSM',
    'R18 CI Verification Pipeline',
    'R25 DebugRuntime CI限定',
    'isr-verify-r11-r25-closed-coverage.ps1'
)

foreach ($token in $requiredStrategyTokens) {
    if ($strategyText -notmatch [regex]::Escape($token)) {
        throw "R11-R25 strategy document missing token: $token"
    }
}

Write-Host '[PASS] R11-R25 closed coverage gate verified'
