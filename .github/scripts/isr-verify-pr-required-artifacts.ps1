$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'pr_required_artifacts_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Test-HasProperty {
    param(
        [Parameter(Mandatory = $true)]$Object,
        [Parameter(Mandatory = $true)][string]$Name
    )

    if ($null -eq $Object) {
        return $false
    }

    return ($null -ne $Object.PSObject.Properties[$Name])
}

$artifacts = @(
    @{ key = 'authorityInventory'; path = (Join-Path $repoRoot 'evidence\authority_inventory_report.json'); schema = 'authority_inventory_report_v1' },
    @{ key = 'inventoryDiff'; path = (Join-Path $repoRoot 'storage\isr_inventory\inventory_diff_report.json'); schema = 'authority_inventory_diff_report_v1' },
    @{ key = 'safetyRegression'; path = (Join-Path $repoRoot 'evidence\safety_regression_report.json'); schema = 'safety_regression_report_v1' },
    @{ key = 'prSla'; path = (Join-Path $repoRoot 'evidence\pr_sla_report.json'); schema = 'pr_sla_report_v1' },
    @{ key = 'validatorTiering'; path = (Join-Path $repoRoot 'evidence\validator_tiering_report.json'); schema = 'validator_tiering_report_v3' },
    @{ key = 'documentationScope'; path = (Join-Path $repoRoot 'evidence\documentation_scope_rule_report.json'); schema = 'documentation_scope_rule_report_v1' },
    @{ key = 'designDocsCoverage'; path = (Join-Path $repoRoot 'evidence\design_docs_coverage_report.json'); schema = 'design_docs_coverage_report_v1' },
    @{ key = 'runtimeCoordinatorStateMachine'; path = (Join-Path $repoRoot 'evidence\runtime_coordinator_state_machine_report.json'); schema = 'runtime_coordinator_state_machine_report_v1' },
    @{ key = 'taxonomyPhaseMapping'; path = (Join-Path $repoRoot 'evidence\taxonomy_phase_mapping_report.json'); schema = 'taxonomy_phase_mapping_report_v1' }
)

$violations = New-Object 'System.Collections.Generic.List[string]'
$details = New-Object 'System.Collections.Generic.List[object]'

foreach ($artifact in $artifacts) {
    $json = Read-JsonFile -Path $artifact.path
    if ($null -eq $json) {
        $violations.Add("Missing required PR artifact: key=$($artifact.key) path=$($artifact.path)")
        $details.Add([ordered]@{
                key = $artifact.key
                path = $artifact.path
                exists = $false
                schema = $null
            }) | Out-Null
        continue
    }

    $details.Add([ordered]@{
            key = $artifact.key
            path = $artifact.path
            exists = $true
            schema = "$($json.schema)"
        }) | Out-Null

    if ("$($json.schema)" -ne "$($artifact.schema)") {
        $violations.Add("Artifact schema mismatch: key=$($artifact.key) expected=$($artifact.schema) actual=$($json.schema)")
    }
}

$topologyDiffDocs = @(Get-ChildItem -LiteralPath (Join-Path $repoRoot 'doc\work5') -File -Filter 'Practical_Stable_ISR_Runtime_topology_diff_*.md' | Sort-Object -Property Name)
if ($topologyDiffDocs.Count -eq 0) {
    $violations.Add('Missing required topology diff document under doc/work5 (Practical_Stable_ISR_Runtime_topology_diff_*.md)')
}
else {
    $latestTopologyDoc = $topologyDiffDocs[-1]
    $latestTopologyText = Get-Content -LiteralPath $latestTopologyDoc.FullName -Raw -Encoding UTF8
    foreach ($requiredToken in @('Topology Diff', 'authority source', 'observe path', 'publication path', 'retire ownership')) {
        if (-not $latestTopologyText.Contains($requiredToken)) {
            $violations.Add("Topology diff document missing required token: path=$($latestTopologyDoc.FullName) token=$requiredToken")
        }
    }

    $details.Add([ordered]@{
            key = 'topologyDiffDocument'
            path = $latestTopologyDoc.FullName
            exists = $true
            schema = $null
        }) | Out-Null
}

$authorityInventory = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\authority_inventory_report.json')
if ($null -ne $authorityInventory) {
    if ($null -eq $authorityInventory.summary) {
        $violations.Add('authority_inventory_report missing summary (authority impact analysis)')
    }
    else {
        foreach ($requiredSummaryField in @('diffAddedCount', 'diffObservePathChangedCount', 'diffRetirementOwnerChangedCount')) {
            if ($authorityInventory.summary.PSObject.Properties.Name -notcontains $requiredSummaryField) {
                $violations.Add("authority_inventory_report summary missing required field: $requiredSummaryField")
            }
        }
    }
}

$safetyRegression = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\safety_regression_report.json')
if ($null -ne $safetyRegression) {
    if ($safetyRegression.PSObject.Properties.Name -notcontains 'safetyPass') {
        $violations.Add('safety_regression_report missing safetyPass field')
    }

    if ($safetyRegression.PSObject.Properties.Name -notcontains 'checks') {
        $violations.Add('safety_regression_report missing checks field')
    }
}

$prSla = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\pr_sla_report.json')
if ($null -ne $prSla) {
    foreach ($requiredField in @('declaredClass', 'declaredClassSource', 'tier', 'ready', 'releaseRequiresExhaustive', 'releaseWindow', 'needsRevalidation', 'labelSuggestions', 'openedAt', 'openedAtSource', 'eventHeadSha', 'currentHeadSha', 'staleEvaluation', 'deadlineAt')) {
        if ($prSla.PSObject.Properties.Name -notcontains $requiredField) {
            $violations.Add("pr_sla_report missing required field: $requiredField")
        }
    }

    if ((Test-HasProperty -Object $prSla -Name 'needsRevalidation') -and
        (Test-HasProperty -Object $prSla -Name 'labelSuggestions')) {
        $labelSuggestions = @($prSla.labelSuggestions | ForEach-Object { "$_" })
        if ([bool]$prSla.needsRevalidation -and ($labelSuggestions -notcontains 'needs-revalidation')) {
            $violations.Add('pr_sla_report inconsistency: needsRevalidation=true requires labelSuggestions to include needs-revalidation')
        }
    }
}

$validatorTiering = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\validator_tiering_report.json')
if ($null -ne $validatorTiering) {
    if ($validatorTiering.PSObject.Properties.Name -notcontains 'slaFreshness') {
        $violations.Add('validator_tiering_report missing slaFreshness field (verification impact)')
    }
}

$documentationScope = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\documentation_scope_rule_report.json')
if ($null -ne $documentationScope) {
    if (-not (Test-HasProperty -Object $documentationScope -Name 'ready')) {
        $violations.Add('documentation_scope_rule_report missing ready field')
    }
    elseif (-not [bool]$documentationScope.ready) {
        $violations.Add('documentation_scope_rule_report must be ready=true')
    }

    if (-not (Test-HasProperty -Object $documentationScope -Name 'documents')) {
        $violations.Add('documentation_scope_rule_report missing documents field')
    }
    else {
        $documentRoles = @($documentationScope.documents | ForEach-Object { "$($_.role)" })
        foreach ($requiredRole in @('plan-v3_1', 'governance-v1_1', 'design-v1_2', 'tasks-v1_0', 'topology-diff')) {
            if ($documentRoles -notcontains $requiredRole) {
                $violations.Add("documentation_scope_rule_report missing required document role: $requiredRole")
            }
        }
    }
}

$designDocsCoverage = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\design_docs_coverage_report.json')
if ($null -ne $designDocsCoverage) {
    if (-not (Test-HasProperty -Object $designDocsCoverage -Name 'ready')) {
        $violations.Add('design_docs_coverage_report missing ready field')
    }
    elseif (-not [bool]$designDocsCoverage.ready) {
        $violations.Add('design_docs_coverage_report must be ready=true')
    }
}

$runtimeCoordinatorStateMachine = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\runtime_coordinator_state_machine_report.json')
if ($null -ne $runtimeCoordinatorStateMachine) {
    if (-not (Test-HasProperty -Object $runtimeCoordinatorStateMachine -Name 'ready')) {
        $violations.Add('runtime_coordinator_state_machine_report missing ready field')
    }
    elseif (-not [bool]$runtimeCoordinatorStateMachine.ready) {
        $violations.Add('runtime_coordinator_state_machine_report must be ready=true')
    }

    if (-not (Test-HasProperty -Object $runtimeCoordinatorStateMachine -Name 'requiredStates')) {
        $violations.Add('runtime_coordinator_state_machine_report missing requiredStates field')
    }
}

$taxonomyPhaseMapping = Read-JsonFile -Path (Join-Path $repoRoot 'evidence\taxonomy_phase_mapping_report.json')
if ($null -ne $taxonomyPhaseMapping) {
    if (-not (Test-HasProperty -Object $taxonomyPhaseMapping -Name 'ready')) {
        $violations.Add('taxonomy_phase_mapping_report missing ready field')
    }
    elseif (-not [bool]$taxonomyPhaseMapping.ready) {
        $violations.Add('taxonomy_phase_mapping_report must be ready=true')
    }

    if (-not (Test-HasProperty -Object $taxonomyPhaseMapping -Name 'checks')) {
        $violations.Add('taxonomy_phase_mapping_report missing checks field')
    }
    else {
        foreach ($requiredCheck in @('mappingRows', 'actionRules')) {
            if ($taxonomyPhaseMapping.checks.PSObject.Properties.Name -notcontains $requiredCheck) {
                $violations.Add("taxonomy_phase_mapping_report checks missing required field: $requiredCheck")
            }
        }
    }
}

$authorityImpactAnalysisCheck = ($null -ne $authorityInventory -and $null -ne $authorityInventory.summary)
$verificationImpactCheck = ($null -ne $validatorTiering -and (Test-HasProperty -Object $validatorTiering -Name 'slaFreshness'))
$safetyPassTableCheck = ($null -ne $safetyRegression -and (Test-HasProperty -Object $safetyRegression -Name 'checks'))
$documentationScopeCheck = ($null -ne $documentationScope -and (Test-HasProperty -Object $documentationScope -Name 'ready') -and [bool]$documentationScope.ready)
$designDocsCoverageCheck = ($null -ne $designDocsCoverage -and (Test-HasProperty -Object $designDocsCoverage -Name 'ready') -and [bool]$designDocsCoverage.ready)
$runtimeCoordinatorStateMachineCheck = ($null -ne $runtimeCoordinatorStateMachine -and (Test-HasProperty -Object $runtimeCoordinatorStateMachine -Name 'ready') -and [bool]$runtimeCoordinatorStateMachine.ready)
$taxonomyPhaseMappingCheck = ($null -ne $taxonomyPhaseMapping -and (Test-HasProperty -Object $taxonomyPhaseMapping -Name 'ready') -and [bool]$taxonomyPhaseMapping.ready)

$artifactDetails = $details.ToArray()
$violationDetails = $violations.ToArray()

$report = @{
    schema = 'pr_required_artifacts_report_v1'
    generatedAt = (Get-Date -Format 'o')
    artifacts = $artifactDetails
    checks = @{
        authorityImpactAnalysis = $authorityImpactAnalysisCheck
        verificationImpact = $verificationImpactCheck
        safetyPassTable = $safetyPassTableCheck
        documentationScopeReady = $documentationScopeCheck
        designDocsCoverageReady = $designDocsCoverageCheck
        runtimeCoordinatorStateMachineReady = $runtimeCoordinatorStateMachineCheck
        taxonomyPhaseMappingReady = $taxonomyPhaseMappingCheck
    }
    violations = $violationDetails
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] PR required artifacts report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }

    throw "PR required artifact verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] PR required artifact verification passed'
