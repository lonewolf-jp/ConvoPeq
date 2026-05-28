param(
    [string]$PolicyGlob = '.github/isr-*.json'
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'policy_top_level_governance_report.json'

if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$searchRoot = if ([System.IO.Path]::IsPathRooted($PolicyGlob)) {
    Split-Path -Path $PolicyGlob -Parent
}
else {
    Join-Path $repoRoot (Split-Path -Path $PolicyGlob -Parent)
}

$pattern = Split-Path -Path $PolicyGlob -Leaf
if (-not (Test-Path $searchRoot)) {
    throw "Policy search root not found: $searchRoot"
}

$policyFiles = @(Get-ChildItem -Path $searchRoot -Filter $pattern -File | Sort-Object -Property FullName)
if ($policyFiles.Count -eq 0) {
    throw "No policy files matched: $PolicyGlob"
}

$violations = New-Object System.Collections.Generic.List[string]
$details = New-Object System.Collections.Generic.List[object]
$requiredFields = @('owner', 'issue', 'rationale', 'expiry')
$today = (Get-Date).Date
$schemaToPaths = @{}
$issueToPaths = @{}

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

foreach ($file in $policyFiles) {
    $relativePath = Get-RelativePathCompat -BasePath $repoRoot -TargetPath $file.FullName
    $json = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $schemaRaw = "$($json.schema)"
    $issueRaw = "$($json.issue)"

    if ([string]::IsNullOrWhiteSpace($schemaRaw)) {
        $violations.Add("$relativePath missing required top-level field: schema")
    }
    else {
        if (-not $schemaToPaths.ContainsKey($schemaRaw)) {
            $schemaToPaths[$schemaRaw] = New-Object System.Collections.Generic.List[string]
        }
        $schemaToPaths[$schemaRaw].Add($relativePath)
    }

    if (-not [string]::IsNullOrWhiteSpace($issueRaw)) {
        if (-not $issueToPaths.ContainsKey($issueRaw)) {
            $issueToPaths[$issueRaw] = New-Object System.Collections.Generic.List[string]
        }
        $issueToPaths[$issueRaw].Add($relativePath)
    }

    $missingFields = @()
    foreach ($field in $requiredFields) {
        if ($null -eq $json.$field -or [string]::IsNullOrWhiteSpace("$($json.$field)")) {
            $missingFields += $field
            $violations.Add("$relativePath missing required top-level field: $field")
        }
    }

    $expiryRaw = "$($json.expiry)"
    $expiryValid = $false
    $expired = $false
    if ([string]::IsNullOrWhiteSpace($expiryRaw)) {
        $violations.Add("$relativePath missing expiry value")
    }
    else {
        try {
            $expiry = [datetime]::ParseExact($expiryRaw, 'yyyy-MM-dd', [System.Globalization.CultureInfo]::InvariantCulture)
            $expiryValid = $true
            $expired = $today -gt $expiry.Date
            if ($expired) {
                $violations.Add("$relativePath expired: expiry=$expiryRaw owner=$($json.owner) issue=$($json.issue)")
            }
        }
        catch {
            $violations.Add("$relativePath has invalid expiry format: '$expiryRaw' (expected yyyy-MM-dd)")
        }
    }

    $details.Add([ordered]@{
            path          = $relativePath
            schema        = $schemaRaw
            owner         = "$($json.owner)"
            issue         = "$($json.issue)"
            rationale     = "$($json.rationale)"
            expiry        = $expiryRaw
            expiryValid   = $expiryValid
            expired       = $expired
            missingFields = $missingFields
        }) | Out-Null
}

foreach ($schemaKey in $schemaToPaths.Keys) {
    $paths = @($schemaToPaths[$schemaKey])
    if ($paths.Count -gt 1) {
        $violations.Add("Duplicate policy schema detected: schema=$schemaKey files=$($paths -join ',')")
    }
}

foreach ($issueKey in $issueToPaths.Keys) {
    $paths = @($issueToPaths[$issueKey])
    if ($paths.Count -gt 1) {
        $violations.Add("Duplicate policy issue detected: issue=$issueKey files=$($paths -join ',')")
    }
}

$report = [ordered]@{
    schema         = 'policy_top_level_governance_report_v2'
    generatedAt    = (Get-Date -Format 'o')
    policyGlob     = $PolicyGlob
    policyCount    = $policyFiles.Count
    requiredFields = $requiredFields
    policies       = $details
    violations     = $violations
}

$reportJson = $report | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $reportJson -Encoding UTF8
Write-Host "[INFO] policy top-level governance report written: $reportPath"

if ($violations.Count -gt 0) {
    foreach ($violation in $violations) {
        Write-Host "[ERROR] $violation"
    }
    throw "Policy top-level governance violations detected. count=$($violations.Count)"
}

Write-Host "[INFO] scanned policy files=$($policyFiles.Count)"
Write-Host '[PASS] policy top-level governance gate verified'
