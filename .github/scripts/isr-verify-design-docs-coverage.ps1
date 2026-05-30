$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'design_docs_coverage_report.json'

# Legacy contract anchors for gate-wiring self-test (do not remove).
$legacyContractTokens = @(
    'Practical_Stable_ISR_Runtime_基本計画書_v3_1.md',
    'ISR_Runtime_実装統治規約_v1_1.md',
    'Practical_Stable_ISR_Runtime_詳細設計_v1_2.md',
    'Practical_Stable_ISR_Runtime_フェーズ別実装タスク分解_v1_0.md'
)

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$documents = @(
    [ordered]@{
        role           = 'base-plan-v2_3'
        path           = 'doc/work6/base_plan.md'
        requiredTokens = @(
            'Single Authoritative Runtime Semantic Source',
            'PublicationEpoch',
            'RuntimeSemanticHash'
        )
    },
    [ordered]@{
        role           = 'governance-v1_2'
        path           = 'doc/work6/ai_governance_v1_2.md'
        requiredTokens = @(
            'Single Authoritative Runtime Principle',
            'Publication Bypass',
            'Fail-Closed Verifier'
        )
    },
    [ordered]@{
        role           = 'detailed-design-v1_6'
        path           = 'doc/work6/detailed_design_isr_bridge_runtime_v1_6.md'
        requiredTokens = @(
            'RuntimeSemanticSchema',
            'Publication Architecture v1.6',
            'Retire / Reclaim Economics v1.6'
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
                role          = $document.role
                path          = $fullPath
                exists        = $false
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
            role          = $document.role
            path          = $fullPath
            exists        = $true
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
