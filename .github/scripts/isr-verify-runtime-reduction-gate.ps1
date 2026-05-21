$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"

if (-not (Test-Path $audioRoot)) {
    throw "audioengine directory not found: $audioRoot"
}

$allowedRuntimeClasses = @(
    "DebugRuntime",
    "DSPHandleRuntime",
    "CrossfadeAuthorityRuntime",
    "HBTraceRuntime",
    "HBVerifierRuntime",
    "LifecycleIsolationRuntime",
    "LifecycleBarrierRuntime",
    "RetireRuntime",
    "ShutdownRuntime"
)

$runtimeClasses = New-Object System.Collections.Generic.HashSet[string]

Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h, *.hpp, *.cpp, *.cxx, *.cc | ForEach-Object {
    $content = Get-Content -Path $_.FullName -Raw -Encoding UTF8
    $hitInfo = Select-String -InputObject $content -Pattern "\bclass\s+([A-Za-z_][A-Za-z0-9_]*Runtime)\b" -AllMatches
    foreach ($h in $hitInfo) {
        foreach ($m in $h.Matches) {
            [void]$runtimeClasses.Add($m.Groups[1].Value)
        }
    }
}

$unexpected = @()
foreach ($name in $runtimeClasses) {
    if ($allowedRuntimeClasses -notcontains $name) {
        $unexpected += $name
    }
}

if ($unexpected.Count -gt 0) {
    $joined = ($unexpected | Sort-Object) -join ", "
    throw "RuntimeReductionGate violation: unexpected runtime class(es): $joined"
}

Write-Host "[PASS] RuntimeReductionGate"
Write-Host "Allowed runtime classes: $($allowedRuntimeClasses -join ', ')"
