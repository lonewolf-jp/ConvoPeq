$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'design_docs_coverage_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$documents = @(
    [ordered]@{
        role = 'plan-v3_1'
        path = 'doc/work5/Practical_Stable_ISR_Runtime_基本計画書_v3_1.md'
        requiredTokens = @(
            'Single Authoritative Observable Runtime',
            'publish(RuntimeWorld*)',
            'Authority Classification System'
        )
    },
    [ordered]@{
        role = 'governance-v1_1'
        path = 'doc/work5/ISR_Runtime_実装統治規約_v1_1.md'
        requiredTokens = @(
            'Safety-First Clause',
            'Rule-8 fail-closed mandatory',
            'Documentation Scope Rule'
        )
    },
    [ordered]@{
        role = 'design-v1_2'
        path = 'doc/work5/Practical_Stable_ISR_Runtime_詳細設計_v1_2.md'
        requiredTokens = @(
            'RuntimeCoordinator 状態機械',
            'Tier × PR SLA',
            'Runtime Safety Regression 判定'
        )
    },
    [ordered]@{
        role = 'tasks-v1_0'
        path = 'doc/work5/Practical_Stable_ISR_Runtime_フェーズ別実装タスク分解_v1_0.md'
        requiredTokens = @(
            'Phase 1: Authority Freeze',
            'Phase 6: Retire Pressure Governance',
            'X-T10'
        )
    }
)

$violations = New-Object 'System.Collections.Generic.List[string]'
$details = New-Object 'System.Collections.Generic.List[object]'

foreach ($document in $documents) {
    $fullPath = Join-Path $repoRoot $document.path
    if (-not (Test-Path -LiteralPath $fullPath)) {
        $violations.Add("Missing required design document: role=$($document.role) path=$fullPath")
        $details.Add([ordered]@{
                role = $document.role
                path = $fullPath
                exists = $false
                matchedTokens = @()
                missingTokens = @($document.requiredTokens)
            }) | Out-Null
        continue
    }

    $text = Get-Content -LiteralPath $fullPath -Raw -Encoding UTF8
    $matchedTokens = New-Object 'System.Collections.Generic.List[string]'
    $missingTokens = New-Object 'System.Collections.Generic.List[string]'

    foreach ($token in @($document.requiredTokens)) {
        if ($text.Contains($token)) {
            $matchedTokens.Add($token) | Out-Null
        }
        else {
            $missingTokens.Add($token) | Out-Null
            $violations.Add("Design document token missing: role=$($document.role) token=$token path=$fullPath")
        }
    }

    $details.Add([ordered]@{
            role = $document.role
            path = $fullPath
            exists = $true
            matchedTokens = @($matchedTokens)
            missingTokens = @($missingTokens)
        }) | Out-Null
}

$report = [pscustomobject]@{}
$report | Add-Member -NotePropertyName 'schema' -NotePropertyValue 'design_docs_coverage_report_v1'
$report | Add-Member -NotePropertyName 'generatedAt' -NotePropertyValue (Get-Date -Format 'o')
$report | Add-Member -NotePropertyName 'documents' -NotePropertyValue @($details.ToArray())
$report | Add-Member -NotePropertyName 'violations' -NotePropertyValue @($violations.ToArray())
$report | Add-Member -NotePropertyName 'ready' -NotePropertyValue ($violations.Count -eq 0)

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] design docs coverage report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }

    throw "Design docs coverage verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] design docs coverage verification passed'
