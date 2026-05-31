$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$commitPath=Join-Path $repoRoot 'src\audioengine\AudioEngine.Commit.cpp'
$reportPath=Join-Path $repoRoot 'evidence\retire_escalation_safety_report.json'

if(-not(Test-Path -LiteralPath (Split-Path $reportPath -Parent))){
    New-Item -ItemType Directory -Path (Split-Path $reportPath -Parent) -Force|Out-Null
}

$violations=New-Object 'System.Collections.Generic.List[string]'
if(-not(Test-Path -LiteralPath $commitPath)){
    $violations.Add("Missing source: $commitPath")|Out-Null
}else{
    $s=Get-Content -LiteralPath $commitPath -Raw -Encoding UTF8

    $requiredPatterns = @(
        'hasExceededDeferralThresholds\(',
        'fetchAddAtomic\(retireEscalationCount_',
        'retireRuntimeEx_\.quarantine\(',
        'retireRuntimeEx_\.canReclaimAfterEscalation\(',
        'retireRuntimeEx_\.reclaim\('
    )

    foreach ($pattern in $requiredPatterns) {
        if ($s -notmatch $pattern) {
            $violations.Add("Missing retire escalation safety contract pattern: $pattern")|Out-Null
        }
    }

    $exceededIndex = $s.IndexOf('if (exceededDeferralThresholds)', [System.StringComparison]::Ordinal)
    $countIndex = $s.IndexOf('fetchAddAtomic(retireEscalationCount_', [System.StringComparison]::Ordinal)
    $quarantineIndex = $s.IndexOf('retireRuntimeEx_.quarantine(', [System.StringComparison]::Ordinal)
    $canReclaimAfterEscalationIndex = $s.IndexOf('retireRuntimeEx_.canReclaimAfterEscalation(', [System.StringComparison]::Ordinal)

    if ($exceededIndex -lt 0) {
        $violations.Add('Retire escalation branch not found: if (exceededDeferralThresholds)')|Out-Null
    }
    else {
        if ($countIndex -lt 0 -or $countIndex -le $exceededIndex) {
            $violations.Add('Retire escalation contract violation: retireEscalationCount_ must increment inside exceededDeferralThresholds branch')|Out-Null
        }

        if ($quarantineIndex -lt 0 -or $quarantineIndex -le $exceededIndex) {
            $violations.Add('Retire escalation contract violation: quarantine must execute inside exceededDeferralThresholds branch')|Out-Null
        }

        if ($canReclaimAfterEscalationIndex -lt 0 -or $canReclaimAfterEscalationIndex -le $exceededIndex) {
            $violations.Add('Retire escalation contract violation: canReclaimAfterEscalation guard must execute inside exceededDeferralThresholds branch')|Out-Null
        }
    }
}

$report=[ordered]@{
    schema='retire_escalation_safety_report_v1'
    generatedAt=(Get-Date -Format 'o')
    sourcePath=$commitPath
    violations=@($violations)
    ready=($violations.Count -eq 0)
}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if($violations.Count -gt 0){
    foreach($v in $violations){Write-Host "[ERROR] $v"}
    throw 'retire escalation safety verification failed'
}

Write-Host '[PASS] retire escalation safety verification passed'
