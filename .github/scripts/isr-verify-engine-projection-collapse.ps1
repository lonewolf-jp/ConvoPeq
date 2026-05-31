$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$audioRoot = Join-Path $repoRoot 'src\audioengine'
$evidenceDir = Join-Path $repoRoot 'evidence'
$reportPath = Join-Path $evidenceDir 'engine_projection_collapse_report.json'

if (-not (Test-Path -LiteralPath $evidenceDir)) { New-Item -ItemType Directory -Path $evidenceDir -Force | Out-Null }
if (-not (Test-Path -LiteralPath $audioRoot)) { throw "Missing audio root: $audioRoot" }

$violations = New-Object 'System.Collections.Generic.List[string]'
$hits = New-Object 'System.Collections.Generic.List[object]'

$forbiddenByFile = @(
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp'; pattern = 'runtimeGraph\s*->' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.BlockDouble.cpp'; pattern = 'runtimeGraph\s*->' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.Snapshot.cpp'; pattern = 'runtimeGraph\s*->' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Timer.cpp'; pattern = 'runtimeGraph\s*->' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp'; pattern = '\bgetRuntimeGraph\s*\(' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.BlockDouble.cpp'; pattern = '\bgetRuntimeGraph\s*\(' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Timer.cpp'; pattern = '\bgetRuntimeGraph\s*\(' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.AudioBlock.cpp'; pattern = '\b(?:convo::)?consumeAtomic\s*\(\s*currentSampleRate\b' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.BlockDouble.cpp'; pattern = '\b(?:convo::)?consumeAtomic\s*\(\s*currentSampleRate\b' },
    [pscustomobject]@{ path = 'src/audioengine/AudioEngine.Processing.Snapshot.cpp'; pattern = '\b(?:convo::)?consumeAtomic\s*\(\s*currentSampleRate\b' }
)

foreach ($rule in $forbiddenByFile) {
    $absolutePath = Join-Path $repoRoot $rule.path
    if (-not (Test-Path -LiteralPath $absolutePath)) {
        $violations.Add("Missing scoped file for collapse verification: $($rule.path)") | Out-Null
        continue
    }

    $text = Get-Content -LiteralPath $absolutePath -Raw -Encoding UTF8
    $m = [regex]::Matches($text, $rule.pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if ($m.Count -gt 0) {
        $hits.Add([pscustomobject]@{ path = $rule.path; pattern = $rule.pattern; count = $m.Count }) | Out-Null
        $violations.Add("Engine projection collapse violation: forbidden pattern matched path=$($rule.path) pattern=$($rule.pattern) count=$($m.Count)") | Out-Null
    }
}

$totalHits = [int](($hits | Measure-Object -Property count -Sum).Sum)

$hitsArray = $hits.ToArray()
$violationsArray = $violations.ToArray()

$report = @{
    schema      = 'engine_projection_collapse_report_v1'
    generatedAt = (Get-Date -Format 'o')
    sourceRoot  = $audioRoot
    patterns    = @($forbiddenByFile | ForEach-Object { $_.pattern })
    totalHits   = $totalHits
    hits        = $hitsArray
    violations  = $violationsArray
    ready       = ($violations.Count -eq 0)
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding UTF8
Write-Host "[INFO] report: $reportPath"
if ($violations.Count -gt 0) { foreach ($v in $violations) { Write-Host "[ERROR] $v" }; throw 'engine projection collapse verification failed' }
Write-Host '[PASS] engine projection collapse verification passed'
