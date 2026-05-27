param(
    [switch]$RequireAst
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$audioRoot = Join-Path $repoRoot "src\audioengine"
$evidenceDir = Join-Path $repoRoot "evidence"
$reportPath = Join-Path $evidenceDir "trigger_ast_report.json"

if (-not (Test-Path $audioRoot)) {
    throw "Missing audioengine source root: $audioRoot"
}
if (-not (Test-Path $evidenceDir)) {
    New-Item -Path $evidenceDir -ItemType Directory | Out-Null
}

$candidateNames = @('ast-grep', 'sg')
$sgCommand = $null
$probeFailureNotes = New-Object System.Collections.Generic.List[string]

foreach ($candidate in $candidateNames) {
    $cmd = Get-Command -Name $candidate -ErrorAction SilentlyContinue
    if (-not $cmd) {
        continue
    }

    $probeOutput = & $cmd.Source --version 2>&1
    $probeExitCode = $LASTEXITCODE
    $probeText = ($probeOutput | Out-String)
    $probeLooksBroken = ($probeText -match 'Cannot find module' -or $probeText -match 'Error:\s+' -or $probeText -match 'MODULE_NOT_FOUND')

    if ($probeExitCode -eq 0 -and -not $probeLooksBroken) {
        $sgCommand = $cmd
        break
    }

    $probeFailureNotes.Add("$($cmd.Name): exit=$probeExitCode") | Out-Null
}

$result = [ordered]@{
    schema = 'trigger_ast_report_v1'
    generatedAt = (Get-Date -Format 'o')
    required = [bool]$RequireAst
    available = $false
    tool = $null
    commandOk = $false
    fadingOutDspWriteMatches = 0
    fadingOutDspWriteFallbackMatches = 0
    fadingOutDspWriteEffectiveMatches = 0
    fadingOutDspWriteEffectiveSource = 'none'
    note = ''
}

$textFallbackCount = 0
$files = Get-ChildItem -Path $audioRoot -Recurse -File -Include *.h,*.hpp,*.cpp,*.cxx,*.cc
foreach ($file in $files) {
    $text = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8
    $textFallbackCount += ([regex]::Matches($text, '\bfadingOutDSP\s*=')).Count
}
$result.fadingOutDspWriteFallbackMatches = $textFallbackCount

if (-not $sgCommand) {
    $noteSuffix = ''
    if ($probeFailureNotes.Count -gt 0) {
        $noteSuffix = " failedCandidates=$($probeFailureNotes -join '; ')"
    }
    $result.note = "ast-grep tool is not available or unusable. Running in monitor-only mode.$noteSuffix"
    $resultJson = $result | ConvertTo-Json -Depth 8
    Set-Content -LiteralPath $reportPath -Value $resultJson -Encoding UTF8
    Write-Host "[WARN] $($result.note)"

    if ($RequireAst) {
        throw 'AST trigger check required but ast-grep tool is unavailable.'
    }

    Write-Host '[PASS] trigger AST gate skipped (monitor mode)'
    return
}

$result.available = $true
$result.tool = $sgCommand.Name

try {
    $output = & $sgCommand.Source run --pattern 'fadingOutDSP = $RHS' --lang cpp $audioRoot --json=stream 2>&1
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0 -and $exitCode -ne 1) {
        throw "ast-grep returned unexpected exit code: $exitCode"
    }

    $matchCount = 0
    foreach ($line in $output) {
        $lineText = "$line".Trim()
        if ([string]::IsNullOrWhiteSpace($lineText)) {
            continue
        }

        if ($lineText.StartsWith('{') -and $lineText.Contains('"file"')) {
            $matchCount++
        }
    }

    $result.commandOk = $true
    $result.fadingOutDspWriteMatches = $matchCount
    if ($RequireAst) {
        $result.fadingOutDspWriteEffectiveMatches = $matchCount
        $result.fadingOutDspWriteEffectiveSource = 'astOnly'
    }
    else {
        $result.fadingOutDspWriteEffectiveMatches = [Math]::Max($matchCount, $textFallbackCount)
        $result.fadingOutDspWriteEffectiveSource = 'astOrFallbackMax'
    }
    $result.note = 'AST scan completed'
}
catch {
    $result.commandOk = $false
    $result.note = "AST scan failed: $($_.Exception.Message)"
    $result.fadingOutDspWriteEffectiveMatches = $textFallbackCount

    if ($RequireAst) {
        $resultJson = $result | ConvertTo-Json -Depth 8
        Set-Content -LiteralPath $reportPath -Value $resultJson -Encoding UTF8
        throw $result.note
    }

    Write-Host "[WARN] $($result.note)"
}

if ($RequireAst -and $result.commandOk -and $result.fadingOutDspWriteFallbackMatches -gt $result.fadingOutDspWriteMatches) {
    $result.commandOk = $false
    $result.note = "AST required mode rejected fallback-dominant result: astMatches=$($result.fadingOutDspWriteMatches) fallbackMatches=$($result.fadingOutDspWriteFallbackMatches)"
}

$resultJson = $result | ConvertTo-Json -Depth 8
Set-Content -LiteralPath $reportPath -Value $resultJson -Encoding UTF8

if ($RequireAst -and -not $result.commandOk) {
    throw 'AST trigger check required but scan did not complete successfully.'
}

Write-Host "[INFO] trigger AST report written: $reportPath"
Write-Host "[INFO] fadingOutDSP AST write matches=$($result.fadingOutDspWriteMatches) fallbackMatches=$($result.fadingOutDspWriteFallbackMatches) effectiveMatches=$($result.fadingOutDspWriteEffectiveMatches) tool=$($result.tool)"

if ($result.commandOk -and $result.fadingOutDspWriteFallbackMatches -gt $result.fadingOutDspWriteMatches) {
    Write-Host '[WARN] AST match count is lower than fallback count. Consider strengthening AST pattern/rules.'
}
Write-Host '[PASS] trigger AST gate completed'
