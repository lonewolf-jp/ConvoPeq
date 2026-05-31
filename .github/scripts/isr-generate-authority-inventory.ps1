param(
    [string]$RepoRoot = (Join-Path $PSScriptRoot '..\..'),
    [string]$OutputDir = 'storage\isr_inventory',
    [switch]$RefreshCurrent
)

$ErrorActionPreference = 'Stop'

$repoRootResolved = [System.IO.Path]::GetFullPath($RepoRoot)
$outputDirResolved = Join-Path $repoRootResolved $OutputDir
if (-not (Test-Path -LiteralPath $outputDirResolved)) {
    New-Item -ItemType Directory -Path $outputDirResolved -Force | Out-Null
}

$targetFiles = @(
    'src\audioengine\AudioEngine.h',
    'src\audioengine\RuntimeGraph.h',
    'src\audioengine\RuntimeTransition.h'
)

$authorityPattern = 'AuthorityClass::(?<class>Authoritative|Derived|Diagnostic|ExecutorLocal)'
$declarationPattern = '^(?<indent>\s*)(?<type>[\w:\<\>\*\&\s]+?)\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]+\])?\s*(?:=[^;]*)?;\s*$'
$structPattern = '^\s*struct\s+(?<name>[A-Za-z_][A-Za-z0-9_]*)\b'

$records = New-Object System.Collections.Generic.List[object]

foreach ($relativePath in $targetFiles) {
    $absolutePath = Join-Path $repoRootResolved $relativePath
    if (-not (Test-Path -LiteralPath $absolutePath)) {
        continue
    }

    $lines = Get-Content -LiteralPath $absolutePath -Encoding UTF8
    $activeClass = $null
    $activeStruct = ''

    for ($i = 0; $i -lt $lines.Count; $i++) {
        $line = $lines[$i]

        $structMatch = [regex]::Match($line, $structPattern)
        if ($structMatch.Success) {
            $activeStruct = $structMatch.Groups['name'].Value
        }

        $authorityMatch = [regex]::Match($line, $authorityPattern)
        if ($authorityMatch.Success) {
            $activeClass = $authorityMatch.Groups['class'].Value
            continue
        }

        if ([string]::IsNullOrWhiteSpace($activeClass)) {
            continue
        }

        $trimmed = $line.Trim()
        if ($trimmed.StartsWith('//') -or $trimmed.StartsWith('/*') -or $trimmed.StartsWith('*')) {
            continue
        }

        $declarationMatch = [regex]::Match($line, $declarationPattern)
        if ($declarationMatch.Success) {
            $fieldName = $declarationMatch.Groups['name'].Value
            $stateName = if ([string]::IsNullOrWhiteSpace($activeStruct)) {
                $fieldName
            }
            else {
                "$activeStruct::$fieldName"
            }

            $records.Add([ordered]@{
                    state            = $stateName
                    authority_class  = $activeClass
                    owner            = 'AudioEngineRuntime'
                    readers          = @('AudioThread', 'NonRT')
                    writers          = @('NonRT')
                    thread_domain    = 'PublishedRuntime'
                    publication_path = 'publish(RuntimeWorld*)'
                    observe_path     = 'RuntimeWorld'
                    retirement_owner = 'RetireManager'
                    source_file      = $relativePath.Replace('\', '/')
                    source_line      = $i + 1
                })

            $activeClass = $null
        }
    }
}

$recordsSorted = $records | Sort-Object -Property state

$timestampUtc = (Get-Date).ToUniversalTime().ToString('o')
$postInventory = [ordered]@{
    schema         = 'authority_inventory_v1'
    generatedAtUtc = $timestampUtc
    source         = 'isr-generate-authority-inventory.ps1'
    entries        = @($recordsSorted)
}

$currentPath = Join-Path $outputDirResolved 'current_authority_inventory.json'
$postPath = Join-Path $outputDirResolved 'post_authority_inventory.json'
$diffPath = Join-Path $outputDirResolved 'inventory_diff_report.json'

$postInventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $postPath -Encoding UTF8

if ($RefreshCurrent -or -not (Test-Path -LiteralPath $currentPath)) {
    $postInventory | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $currentPath -Encoding UTF8
}

$currentInventory = Get-Content -LiteralPath $currentPath -Raw -Encoding UTF8 | ConvertFrom-Json
$currentEntries = @($currentInventory.entries)
$postEntries = @($postInventory.entries)

$currentByState = @{}
foreach ($entry in $currentEntries) {
    $currentByState["$($entry.state)"] = $entry
}

$postByState = @{}
foreach ($entry in $postEntries) {
    $postByState["$($entry.state)"] = $entry
}

$added = New-Object System.Collections.Generic.List[object]
$removed = New-Object System.Collections.Generic.List[object]
$classChanged = New-Object System.Collections.Generic.List[object]
$observePathChanged = New-Object System.Collections.Generic.List[object]
$publicationPathChanged = New-Object System.Collections.Generic.List[object]
$retirementOwnerChanged = New-Object System.Collections.Generic.List[object]
$ownerChanged = New-Object System.Collections.Generic.List[object]

foreach ($state in $postByState.Keys) {
    if (-not $currentByState.ContainsKey($state)) {
        $added.Add([ordered]@{ state = $state; authority_class = $postByState[$state].authority_class })
        continue
    }

    $before = "$($currentByState[$state].authority_class)"
    $after = "$($postByState[$state].authority_class)"
    if ($before -ne $after) {
        $classChanged.Add([ordered]@{ state = $state; before = $before; after = $after })
    }

    $observeBefore = "$($currentByState[$state].observe_path)"
    $observeAfter = "$($postByState[$state].observe_path)"
    if ($observeBefore -ne $observeAfter) {
        $observePathChanged.Add([ordered]@{ state = $state; before = $observeBefore; after = $observeAfter })
    }

    $publicationBefore = "$($currentByState[$state].publication_path)"
    $publicationAfter = "$($postByState[$state].publication_path)"
    if ($publicationBefore -ne $publicationAfter) {
        $publicationPathChanged.Add([ordered]@{ state = $state; before = $publicationBefore; after = $publicationAfter })
    }

    $retirementOwnerBefore = "$($currentByState[$state].retirement_owner)"
    $retirementOwnerAfter = "$($postByState[$state].retirement_owner)"
    if ($retirementOwnerBefore -ne $retirementOwnerAfter) {
        $retirementOwnerChanged.Add([ordered]@{ state = $state; before = $retirementOwnerBefore; after = $retirementOwnerAfter })
    }

    $ownerBefore = "$($currentByState[$state].owner)"
    $ownerAfter = "$($postByState[$state].owner)"
    if ($ownerBefore -ne $ownerAfter) {
        $ownerChanged.Add([ordered]@{ state = $state; before = $ownerBefore; after = $ownerAfter })
    }
}

foreach ($state in $currentByState.Keys) {
    if (-not $postByState.ContainsKey($state)) {
        $removed.Add([ordered]@{ state = $state; authority_class = $currentByState[$state].authority_class })
    }
}

$diffSummary = New-Object PSObject
$diffSummary | Add-Member -NotePropertyName currentCount -NotePropertyValue $currentEntries.Count
$diffSummary | Add-Member -NotePropertyName postCount -NotePropertyValue $postEntries.Count
$diffSummary | Add-Member -NotePropertyName addedCount -NotePropertyValue $added.Count
$diffSummary | Add-Member -NotePropertyName removedCount -NotePropertyValue $removed.Count
$diffSummary | Add-Member -NotePropertyName classChangedCount -NotePropertyValue $classChanged.Count
$diffSummary | Add-Member -NotePropertyName observePathChangedCount -NotePropertyValue $observePathChanged.Count
$diffSummary | Add-Member -NotePropertyName publicationPathChangedCount -NotePropertyValue $publicationPathChanged.Count
$diffSummary | Add-Member -NotePropertyName retirementOwnerChangedCount -NotePropertyValue $retirementOwnerChanged.Count
$diffSummary | Add-Member -NotePropertyName ownerChangedCount -NotePropertyValue $ownerChanged.Count

$diffReport = New-Object PSObject
$diffReport | Add-Member -NotePropertyName schema -NotePropertyValue 'authority_inventory_diff_report_v1'
$diffReport | Add-Member -NotePropertyName generatedAtUtc -NotePropertyValue $timestampUtc
$diffReport | Add-Member -NotePropertyName source -NotePropertyValue 'isr-generate-authority-inventory.ps1'
$diffReport | Add-Member -NotePropertyName currentPath -NotePropertyValue ((Resolve-Path -LiteralPath $currentPath).Path)
$diffReport | Add-Member -NotePropertyName postPath -NotePropertyValue ((Resolve-Path -LiteralPath $postPath).Path)
$diffReport | Add-Member -NotePropertyName summary -NotePropertyValue $diffSummary
$diffReport | Add-Member -NotePropertyName addedEntries -NotePropertyValue (@($added | ForEach-Object { "$($_.state):$($_.authority_class)" }))
$diffReport | Add-Member -NotePropertyName removedEntries -NotePropertyValue (@($removed | ForEach-Object { "$($_.state):$($_.authority_class)" }))
$diffReport | Add-Member -NotePropertyName classChangedEntries -NotePropertyValue (@($classChanged | ForEach-Object { "$($_.state):$($_.before)->$($_.after)" }))
$diffReport | Add-Member -NotePropertyName observePathChangedEntries -NotePropertyValue (@($observePathChanged | ForEach-Object { "$($_.state):$($_.before)->$($_.after)" }))
$diffReport | Add-Member -NotePropertyName publicationPathChangedEntries -NotePropertyValue (@($publicationPathChanged | ForEach-Object { "$($_.state):$($_.before)->$($_.after)" }))
$diffReport | Add-Member -NotePropertyName retirementOwnerChangedEntries -NotePropertyValue (@($retirementOwnerChanged | ForEach-Object { "$($_.state):$($_.before)->$($_.after)" }))
$diffReport | Add-Member -NotePropertyName ownerChangedEntries -NotePropertyValue (@($ownerChanged | ForEach-Object { "$($_.state):$($_.before)->$($_.after)" }))

$diffReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $diffPath -Encoding UTF8

Write-Host "[PASS] authority inventory generated"
Write-Host "  current: $currentPath"
Write-Host "  post:    $postPath"
Write-Host "  diff:    $diffPath"
