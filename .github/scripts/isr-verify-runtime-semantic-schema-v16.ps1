$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$schemaPath = Join-Path $repoRoot 'src\audioengine\ISRRuntimeSemanticSchema.h'
$runtimeStatePath = Join-Path $repoRoot 'src\audioengine\AudioEngine.h'
$authorityClassPath = Join-Path $repoRoot 'src\audioengine\ISRAuthorityClass.h'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'runtime_semantic_schema_v16_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) {
    New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null
}

$violations = New-Object 'System.Collections.Generic.List[string]'

if (-not (Test-Path -LiteralPath $schemaPath)) {
    throw "Missing schema header: $schemaPath"
}

if (-not (Test-Path -LiteralPath $runtimeStatePath)) {
    throw "Missing runtime state header: $runtimeStatePath"
}

if (-not (Test-Path -LiteralPath $authorityClassPath)) {
    throw "Missing authority class header: $authorityClassPath"
}

$schemaText = Get-Content -LiteralPath $schemaPath -Raw -Encoding UTF8
$runtimeText = Get-Content -LiteralPath $runtimeStatePath -Raw -Encoding UTF8
$authorityText = Get-Content -LiteralPath $authorityClassPath -Raw -Encoding UTF8
$runtimeLines = Get-Content -LiteralPath $runtimeStatePath -Encoding UTF8

$requiredSchemaTokens = @(
    'kRuntimeSemanticSchemaVersion',
    'struct GenerationSemantic',
    'struct TopologySemantic',
    'struct RoutingSemantic',
    'struct ExecutionSemantic',
    'struct PublicationSemantic',
    'struct OverlapSemantic',
    'struct RetireSemantic',
    'struct TimingSemantic',
    'struct LatencySemantic',
    'struct SchedulingSemantic',
    'struct ResourceSemantic',
    'struct AffinitySemantic',
    'struct AutomationSemantic',
    'struct CoefficientSemantic',
    'struct RuntimeSemanticSchema',
    'struct ProjectionFreshness',
    'struct RuntimeSemanticHash',
    'generationSemanticHash'
)

$requiredAuthorityEnumTokens = @(
    'enum class AuthorityClass',
    'Authoritative',
    'Derived',
    'Diagnostic',
    'ExecutorLocal',
    'LegacyTemporary'
)

$requiredRuntimeStateTokens = @(
    'generationSemantic',
    'topology',
    'routing',
    'execution',
    'publication',
    'overlap',
    'retire',
    'timing',
    'latency',
    'scheduling',
    'resource',
    'affinity',
    'automation',
    'coefficient',
    'projectionFreshness',
    'semanticHash'
)

$requiredMappingTokens = @(
    'worldOwner->generationSemantic',
    'worldOwner->topology',
    'worldOwner->routing',
    'worldOwner->execution',
    'worldOwner->publication',
    'worldOwner->overlap',
    'worldOwner->retire',
    'worldOwner->timing',
    'worldOwner->latency',
    'worldOwner->scheduling',
    'worldOwner->resource',
    'worldOwner->affinity',
    'worldOwner->automation',
    'worldOwner->coefficient',
    'worldOwner->projectionFreshness',
    'worldOwner->semanticHash.generationSemanticHash',
    'worldOwner->semanticHash'
)

$requiredClassifiedRuntimeFields = @(
    'generationSemantic',
    'topology',
    'routing',
    'execution',
    'publication',
    'overlap',
    'retire',
    'timing',
    'latency',
    'scheduling',
    'resource',
    'affinity',
    'automation',
    'coefficient',
    'projectionFreshness',
    'semanticHash'
)

$allowedClassComments = @(
    'Authoritative',
    'Derived',
    'Diagnostic',
    'ExecutorLocal',
    'LegacyTemporary'
)

function Get-NearestAuthorityClassComment {
    param(
        [Parameter(Mandatory = $true)]
        [object[]]$Lines,
        [Parameter(Mandatory = $true)]
        [int]$StartIndex
    )

    for ($i = $StartIndex - 1; $i -ge 0; $i--) {
        $line = $Lines[$i].Trim()
        if ($line -eq '') {
            continue
        }

        if ($line -match '^//\s*AuthorityClass::([A-Za-z0-9_]+)') {
            return $Matches[1]
        }

        if ($line -like '// *') {
            continue
        }

        break
    }

    return $null
}

foreach ($token in $requiredSchemaTokens) {
    if (-not $schemaText.Contains($token)) {
        $violations.Add("Schema token missing: $token") | Out-Null
    }
}

foreach ($token in $requiredAuthorityEnumTokens) {
    if (-not $authorityText.Contains($token)) {
        $violations.Add("AuthorityClass token missing: $token") | Out-Null
    }
}

foreach ($token in $requiredRuntimeStateTokens) {
    if (-not $runtimeText.Contains($token)) {
        $violations.Add("RuntimeState token missing: $token") | Out-Null
    }
}

foreach ($token in $requiredMappingTokens) {
    if (-not $runtimeText.Contains($token)) {
        $violations.Add("Runtime publish mapping token missing: $token") | Out-Null
    }
}

foreach ($field in $requiredClassifiedRuntimeFields) {
    $fieldIndex = -1
    for ($lineIndex = 0; $lineIndex -lt $runtimeLines.Count; $lineIndex++) {
        if ($runtimeLines[$lineIndex] -match "\b$field\b\s*\{\}\s*;|\b$field\b\s*=|\b$field\b\s*;") {
            $fieldIndex = $lineIndex
            break
        }
    }

    if ($fieldIndex -lt 0) {
        $violations.Add("RuntimeState classified field not found: $field") | Out-Null
        continue
    }

    $className = Get-NearestAuthorityClassComment -Lines $runtimeLines -StartIndex $fieldIndex
    if ($null -eq $className) {
        $violations.Add("RuntimeState field missing AuthorityClass annotation: $field") | Out-Null
        continue
    }

    if ($allowedClassComments -notcontains $className) {
        $violations.Add("RuntimeState field has invalid AuthorityClass annotation: $field => $className") | Out-Null
    }
}

$report = [ordered]@{
    schema                          = 'runtime_semantic_schema_v16_report_v1'
    generatedAt                     = (Get-Date -Format 'o')
    schemaPath                      = $schemaPath
    runtimeStatePath                = $runtimeStatePath
    requiredSchemaTokens            = $requiredSchemaTokens
    requiredAuthorityEnumTokens     = $requiredAuthorityEnumTokens
    requiredRuntimeStateTokens      = $requiredRuntimeStateTokens
    requiredMappingTokens           = $requiredMappingTokens
    requiredClassifiedRuntimeFields = $requiredClassifiedRuntimeFields
    violations                      = @($violations)
    ready                           = ($violations.Count -eq 0)
}

$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] runtime semantic schema report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($v in $violations) {
        Write-Host "[ERROR] $v"
    }
    throw "Runtime semantic schema v1.6 verification failed. violations=$($violations.Count)"
}

Write-Host '[PASS] runtime semantic schema v1.6 verification passed'
