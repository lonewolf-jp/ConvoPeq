$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'documentation_scope_rule_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

function Get-RelativePathCompat {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)

    if ([System.IO.Path]::GetPathRoot($baseFull) -ne [System.IO.Path]::GetPathRoot($targetFull)) {
        return $targetFull.Replace('\\', '/')
    }

    $baseWithSep = if ($baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar) -or $baseFull.EndsWith([System.IO.Path]::AltDirectorySeparatorChar)) {
        $baseFull
    }
    else {
        $baseFull + [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = New-Object System.Uri($baseWithSep)
    $targetUri = New-Object System.Uri($targetFull)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)

    return [System.Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', '/')
}

$work5Dir = Join-Path $repoRoot 'doc\work5'
if (-not (Test-Path -LiteralPath $work5Dir)) {
    throw "Documentation Scope Rule missing required directory: $work5Dir"
}

$requiredDocs = @(
    [ordered]@{
        role           = 'plan-v3_1'
        glob           = 'Practical_Stable_ISR_Runtime_*_v3_1.md'
        requiredTokens = @(
            'Single Authoritative Observable Runtime',
            'Break-glass Rule',
            'Governance Budget Rules'
        )
    },
    [ordered]@{
        role           = 'governance-v1_1'
        glob           = 'ISR_Runtime_*_v1_1.md'
        requiredTokens = @(
            'BreakGlassOverride',
            'fail-closed',
            'Phase Governance / Override Clause',
            'Verification Matrix Rule'
        )
    },
    [ordered]@{
        role           = 'design-v1_2'
        glob           = 'Practical_Stable_ISR_Runtime_*_v1_2.md'
        requiredTokens = @(
            'RuntimeCoordinator',
            'PR SLA',
            'SafetyPass'
        )
    },
    [ordered]@{
        role           = 'tasks-v1_0'
        glob           = 'Practical_Stable_ISR_Runtime_*_v1_0.md'
        requiredTokens = @(
            'Phase 1: Authority Freeze',
            'Phase 6: Retire Pressure Governance',
            'Task ID: `P{phase}-Txx` / `P{phase}-Vxx` / `X-Txx`'
        )
    },
    [ordered]@{
        role           = 'topology-diff'
        glob           = 'Practical_Stable_ISR_Runtime_topology_diff_*.md'
        requiredTokens = @(
            'Topology Diff',
            'authority source',
            'observe path',
            'publication path',
            'retire ownership'
        )
    }
)

$requiredArtifacts = @(
    [ordered]@{ role = 'inventory-current'; path = 'storage/isr_inventory/current_authority_inventory.json' },
    [ordered]@{ role = 'inventory-post'; path = 'storage/isr_inventory/post_authority_inventory.json' },
    [ordered]@{ role = 'inventory-diff'; path = 'storage/isr_inventory/inventory_diff_report.json' },
    [ordered]@{ role = 'inventory-report'; path = 'evidence/authority_inventory_report.json' }
)

$violations = New-Object 'System.Collections.Generic.List[string]'
$details = New-Object 'System.Collections.Generic.List[object]'
$artifactDetails = New-Object 'System.Collections.Generic.List[object]'

foreach ($doc in $requiredDocs) {
    $candidates = @(Get-ChildItem -LiteralPath $work5Dir -File -Filter $doc.glob | Sort-Object -Property Name)
    $resolved = $null
    $candidateMatchesWithTokens = New-Object 'System.Collections.Generic.List[object]'
    if ($candidates.Count -gt 0) {
        foreach ($candidate in $candidates) {
            $candidateText = Get-Content -LiteralPath $candidate.FullName -Raw -Encoding UTF8
            $allTokensFound = $true
            foreach ($token in $doc.requiredTokens) {
                if (-not $candidateText.Contains($token)) {
                    $allTokensFound = $false
                    break
                }
            }

            if ($allTokensFound) {
                $candidateMatchesWithTokens.Add($candidate) | Out-Null
            }
        }

        if ($candidateMatchesWithTokens.Count -eq 1) {
            $resolved = $candidateMatchesWithTokens[0]
        }
        elseif ($candidateMatchesWithTokens.Count -gt 1) {
            $violations.Add("Documentation Scope Rule role resolves ambiguously (expected single canonical doc): role=$($doc.role) glob=$($doc.glob) matches=$($candidateMatchesWithTokens.Count)")
        }
    }

    $exists = ($null -ne $resolved)
    $missingTokens = New-Object 'System.Collections.Generic.List[string]'

    if (-not $exists) {
        $violations.Add("Documentation Scope Rule missing required document role: role=$($doc.role) glob=$($doc.glob)")
        $details.Add([ordered]@{
                role          = $doc.role
                glob          = $doc.glob
                path          = $null
                exists        = $false
                missingTokens = @($doc.requiredTokens)
            }) | Out-Null
        continue
    }

    $absolutePath = $resolved.FullName
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $absolutePath
    $text = Get-Content -LiteralPath $absolutePath -Raw -Encoding UTF8
    foreach ($token in $doc.requiredTokens) {
        if (-not $text.Contains($token)) {
            $missingTokens.Add($token) | Out-Null
            $violations.Add("Documentation Scope Rule token missing: role=$($doc.role) path=$relativePath token=$token")
        }
    }

    $details.Add([ordered]@{
            role          = $doc.role
            glob          = $doc.glob
            path          = $relativePath
            exists        = $true
            missingTokens = @($missingTokens)
        }) | Out-Null
}

foreach ($artifact in $requiredArtifacts) {
    $relativeArtifactPath = $artifact.path.Replace('/', [System.IO.Path]::DirectorySeparatorChar)
    $absoluteArtifactPath = Join-Path $repoRoot $relativeArtifactPath
    $exists = Test-Path -LiteralPath $absoluteArtifactPath

    if (-not $exists) {
        $violations.Add("Documentation Scope Rule missing required artifact: role=$($artifact.role) path=$($artifact.path)")
    }

    $artifactDetails.Add([ordered]@{
            role   = $artifact.role
            path   = $artifact.path
            exists = $exists
        }) | Out-Null
}

$report = [ordered]@{
    schema      = 'documentation_scope_rule_report_v1'
    generatedAt = (Get-Date -Format 'o')
    documents   = $details
    artifacts   = $artifactDetails
    violations  = @($violations)
    ready       = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] documentation scope rule report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Documentation Scope Rule verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] Documentation Scope Rule verification passed'
