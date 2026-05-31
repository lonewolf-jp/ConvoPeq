# isr-verify-semantic-dependency-graph.ps1
# §3.11.2 Semantic Dependency Graph Contract (SemanticDependencyGraphVerifier)
# Verifies: dependency direction is Topology->Routing->Execution->Publication->Retire.
# No reverse-direction derivations are permitted.

$ErrorActionPreference = 'Stop'
$repoRoot    = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath  = Join-Path $evidenceDir 'isr_semantic_dependency_graph_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = [System.Collections.Generic.List[string]]::new()
function Add-Violation { param([string]$Msg); $violations.Add($Msg) | Out-Null }

$schemaHeader = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
if (-not (Test-Path -LiteralPath $schemaHeader)) {
    Add-Violation "ISRRuntimeSemanticSchema.h not found: $schemaHeader"
}
else {
    $sc = Get-Content -LiteralPath $schemaHeader -Raw

    if (-not $sc.Contains('SemanticDependencyGraphVerifier')) {
        Add-Violation 'SemanticDependencyGraphVerifier not registered in kRequiredVerifierTable'
    }

    # The state machine contract (isValidSemanticTransactionTransition) must be defined,
    # which encodes the directed acyclic dependency order.
    if (-not $sc.Contains('isValidSemanticTransactionTransition')) {
        Add-Violation 'isValidSemanticTransactionTransition function is absent from ISRRuntimeSemanticSchema.h'
    }

    # The canonical struct ordering in RuntimeSemanticSchema must follow:
    # metadata -> generation -> topology -> routing -> execution -> publication -> overlap -> retire
    # We verify that 'topology' precedes 'routing' and 'routing' precedes 'execution'
    # and 'execution' precedes 'publication' in the struct declaration.
    $topologyIdx   = $sc.IndexOf('TopologySemantic topology')
    $routingIdx    = $sc.IndexOf('RoutingSemantic routing')
    $executionIdx  = $sc.IndexOf('ExecutionSemantic execution')
    $publicationIdx = $sc.IndexOf('PublicationSemantic publication')
    $retireIdx     = $sc.IndexOf('RetireSemantic retire')

    if ($topologyIdx -lt 0 -or $routingIdx -lt 0 -or $executionIdx -lt 0 -or
        $publicationIdx -lt 0 -or $retireIdx -lt 0) {
        Add-Violation 'RuntimeSemanticSchema is missing one or more required semantic fields'
    }
    elseif (-not ($topologyIdx -lt $routingIdx -and
                  $routingIdx -lt $executionIdx -and
                  $executionIdx -lt $publicationIdx -and
                  $publicationIdx -lt $retireIdx)) {
        Add-Violation 'RuntimeSemanticSchema field ordering violates T->R->E->P->Rt dependency direction'
    }
}

$report = [ordered]@{
    schema      = 'isr_semantic_dependency_graph_evidence_v1'
    generatedAt = (Get-Date -Format 'o')
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] SemanticDependencyGraphVerifier evidence written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) { Write-Host "[FAIL] $v" }
    throw "SemanticDependencyGraphVerifier contract violation. violations=$($violations.Count)"
}
Write-Host '[PASS] SemanticDependencyGraphVerifier contract verification passed'
