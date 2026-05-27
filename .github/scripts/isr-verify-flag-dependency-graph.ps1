param(
    [string]$GraphPath = '.github/isr-flag-dependency-graph.json',
    [string]$MatrixPath = '.github/isr-rollback-compatibility-matrix.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'flag_dependency_graph_report.json'
$resolvedGraphPath = if ([System.IO.Path]::IsPathRooted($GraphPath)) { $GraphPath } else { Join-Path $repoRoot $GraphPath }
$resolvedMatrixPath = if ([System.IO.Path]::IsPathRooted($MatrixPath)) { $MatrixPath } else { Join-Path $repoRoot $MatrixPath }

foreach ($path in @($resolvedGraphPath, $resolvedMatrixPath)) {
    if (-not (Test-Path $path)) {
        throw "Missing file: $path"
    }
}

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$graph = Get-Content -LiteralPath $resolvedGraphPath -Raw -Encoding UTF8 | ConvertFrom-Json
$matrix = Get-Content -LiteralPath $resolvedMatrixPath -Raw -Encoding UTF8 | ConvertFrom-Json

if ($graph.schema -ne 'flag_dependency_graph_v1') {
    throw "Unexpected flag dependency graph schema: $($graph.schema)"
}
if ($matrix.schema -ne 'rollback_compatibility_matrix_v1') {
    throw "Unexpected rollback matrix schema: $($matrix.schema)"
}

$violations = New-Object System.Collections.Generic.List[string]

foreach ($field in @('owner', 'issue', 'rationale', 'expiry')) {
    if ([string]::IsNullOrWhiteSpace("$($graph.$field)")) {
        $violations.Add("Flag dependency graph missing required field: $field")
    }
}

if ([string]::IsNullOrWhiteSpace("$($graph.expiry)") -eq $false) {
    $graphExpiry = [datetime]::ParseExact("$($graph.expiry)", 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
    if ((Get-Date).Date -gt $graphExpiry.Date) {
        $violations.Add("Flag dependency graph expired: expiry=$($graph.expiry) owner=$($graph.owner) issue=$($graph.issue)")
    }
}

if (-not $graph.nodes -or $graph.nodes.Count -lt 2) {
    $violations.Add('Flag dependency graph requires at least two nodes')
}
if (-not $graph.edges -or $graph.edges.Count -lt 1) {
    $violations.Add('Flag dependency graph requires at least one edge')
}

$nodeSet = New-Object System.Collections.Generic.HashSet[string]
$duplicateNodes = New-Object System.Collections.Generic.List[string]
foreach ($node in $graph.nodes) {
    $name = "$node"
    if ([string]::IsNullOrWhiteSpace($name)) {
        $violations.Add('Flag dependency graph contains empty node name')
        continue
    }

    if (-not $nodeSet.Add($name)) {
        $duplicateNodes.Add($name) | Out-Null
    }
}

foreach ($dup in $duplicateNodes) {
    $violations.Add("Flag dependency graph duplicate node: $dup")
}

$adjacency = @{}
foreach ($node in $nodeSet) {
    $adjacency[$node] = New-Object System.Collections.Generic.List[string]
}

foreach ($edge in $graph.edges) {
    $from = "$($edge.from)"
    $to = "$($edge.to)"
    if ([string]::IsNullOrWhiteSpace($from) -or [string]::IsNullOrWhiteSpace($to)) {
        $violations.Add('Flag dependency graph contains edge with empty endpoint')
        continue
    }

    if (-not $nodeSet.Contains($from)) {
        $violations.Add("Flag dependency edge references unknown from-node: $from")
        continue
    }
    if (-not $nodeSet.Contains($to)) {
        $violations.Add("Flag dependency edge references unknown to-node: $to")
        continue
    }

    $adjacency[$from].Add($to)
}

$globalFlag = "$($matrix.globalFlag)"
if (-not $nodeSet.Contains($globalFlag)) {
    $violations.Add("Global rollback flag missing from dependency graph: $globalFlag")
}

foreach ($entry in $matrix.subsystemFlags) {
    $flag = "$($entry.flag)"
    if (-not $nodeSet.Contains($flag)) {
        $violations.Add("Subsystem rollback flag missing from dependency graph: $flag")
    }

    if ($adjacency.ContainsKey($globalFlag)) {
        $reachableDirect = $false
        foreach ($next in $adjacency[$globalFlag]) {
            if ($next -eq $flag) {
                $reachableDirect = $true
                break
            }
        }
        if (-not $reachableDirect) {
            $violations.Add("Subsystem rollback flag is not directly gated by global rollback flag: $flag")
        }
    }
}

$state = @{}
foreach ($node in $nodeSet) {
    $state[$node] = 0
}

$hasCycle = $false
$cycleNodes = New-Object System.Collections.Generic.List[string]

function Invoke-FlagNodeVisit {
    param(
        [string]$Node,
        [hashtable]$StateMap,
        [hashtable]$GraphAdj,
        [System.Collections.Generic.List[string]]$CycleList,
        [ref]$HasCycleRef
    )

    $StateMap[$Node] = 1
    foreach ($next in $GraphAdj[$Node]) {
        if ($StateMap[$next] -eq 0) {
            Invoke-FlagNodeVisit -Node $next -StateMap $StateMap -GraphAdj $GraphAdj -CycleList $CycleList -HasCycleRef $HasCycleRef
        }
        elseif ($StateMap[$next] -eq 1) {
            $HasCycleRef.Value = $true
            $CycleList.Add("$Node->$next") | Out-Null
        }
    }
    $StateMap[$Node] = 2
}

foreach ($node in $nodeSet) {
    if ($state[$node] -eq 0) {
        Invoke-FlagNodeVisit -Node $node -StateMap $state -GraphAdj $adjacency -CycleList $cycleNodes -HasCycleRef ([ref]$hasCycle)
    }
}

if ($hasCycle) {
    $violations.Add("Flag dependency graph must be acyclic: cycles=$($cycleNodes -join ';')")
}

$report = [ordered]@{
    schema = 'flag_dependency_graph_report_v1'
    generatedAt = (Get-Date -Format 'o')
    graphPath = $resolvedGraphPath
    matrixPath = $resolvedMatrixPath
    nodeCount = $nodeSet.Count
    edgeCount = $graph.edges.Count
    globalFlag = $globalFlag
    hasCycle = $hasCycle
    cycleCandidates = $cycleNodes
    violations = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] flag dependency graph report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Flag dependency graph verification failed. count=$($violations.Count)"
}

Write-Host '[PASS] flag dependency graph gate verified'
