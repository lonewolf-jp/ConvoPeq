#============================================================================
# isr-run-runtime-evidence-live.ps1 - collect Live Runtime evidence
#
# Purpose:
#   Run the real Runtime (AudioEngine) via `--cli-run`, then collect and verify
#   the evidence that the run itself produced (evidence/live_runtime_evidence.json).
#
# Difference from isr-run-runtime-evidence.ps1 (`--verify-runtime-evidence`):
#   * existing: calls the Exporter only (no runtime), and the script fabricates
#     missing artifacts.
#   * this script: runs the runtime and writes NOTHING (verification only).
#
# Fail-closed conditions:
#   cannot clear the previous live evidence / run timeout / exit code != 0 /
#   no evidence / verification failure / evidence modified after process exit
#
# NOTE: this file is intentionally ASCII-only. PowerShell 5.1 reads BOM-less
#       files with the system ANSI code page, which can corrupt non-ASCII
#       comments and break parsing on non-Japanese runners.
#============================================================================

param(
    [int]$ExitMs = 4000,
    [int]$TimeoutSec = 0,
    [string]$ExePath = ''
)

$ErrorActionPreference = 'Stop'

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$evidenceRoot = Join-Path $repoRoot "evidence"

$runtimeRunId = [string]$env:CONVO_ISR_RUNTIME_RUN_ID
if ([string]::IsNullOrWhiteSpace($runtimeRunId)) {
    throw "CONVO_ISR_RUNTIME_RUN_ID is required for live runtime evidence mode."
}

if ($ExitMs -lt 1000) {
    throw "ExitMs must be >= 1000 (actual=$ExitMs). Short runs cannot prove runtime processing."
}

if ($TimeoutSec -le 0) {
    $TimeoutSec = [int][Math]::Ceiling($ExitMs / 1000.0) + 30
}

# -- 1) clear the previous live evidence (never reuse an earlier run) ----------
#    Existing artifacts (manifest etc.) are left untouched: live identity is
#    established by runId and file mtime, not by deleting the whole directory.
if (Test-Path -LiteralPath $evidenceRoot) {
    $livePath = Join-Path $evidenceRoot 'live_runtime_evidence.json'
    if (Test-Path -LiteralPath $livePath) {
        Remove-Item -LiteralPath $livePath -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $livePath) {
            throw "Failed to clear previous live evidence: $livePath"
        }
    }
}

New-Item -ItemType Directory -Path $evidenceRoot -Force | Out-Null

# -- 2) select the executable (prefer Debug, which emits full artifacts) ------
#    -ExePath overrides the search (a workflow may build into build-ci/).
if (-not [string]::IsNullOrWhiteSpace($ExePath)) {
    if (-not (Test-Path -LiteralPath $ExePath)) {
        throw "Executable not found (-ExePath): $ExePath"
    }
    $exe = (Resolve-Path -LiteralPath $ExePath).Path
    Write-Host "[INFO] Using the executable given by -ExePath: $exe"
} else {
    $buildRoot = Join-Path $repoRoot "build"
    if (-not (Test-Path -LiteralPath $buildRoot)) {
        throw "Build directory not found: $buildRoot (build ConvoPeq first, or pass -ExePath)."
    }

    $allCandidates = @(Get-ChildItem -Path $buildRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'ConvoPeq*.exe' })
    if ($allCandidates.Count -eq 0) {
        throw "No runnable ConvoPeq executable found under build/ (build ConvoPeq first, or pass -ExePath)."
    }

    $debugCandidates = @($allCandidates | Where-Object { $_.FullName -match '[\\/]Debug[\\/]' })
    if ($debugCandidates.Count -gt 0) {
        $exe = ($debugCandidates | Sort-Object LastWriteTimeUtc -Descending)[0].FullName
    } else {
        $exe = ($allCandidates | Sort-Object LastWriteTimeUtc -Descending)[0].FullName
        Write-Host "[WARN] No Debug build found. Using latest executable: $exe"
    }
}

# -- 2b) refuse to run while another instance holds the single-instance lock --
#    ConvoPeq allows one instance. A second process exits 0 before any of our
#    code runs, so no evidence is produced; report that instead of a vague
#    "missing evidence" failure.
$busy = @()
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    $busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        try { $_.Path -eq $exe } catch { $false }
    })
    if ($busy.Count -eq 0) { break }
    Start-Sleep -Milliseconds 500
}
if ($busy.Count -gt 0) {
    throw ("Another ConvoPeq instance is running (pid=" + (($busy | ForEach-Object { $_.Id }) -join ',') +
        "). The app is single-instance, so it would exit without evidence. Close it and retry.")
}

Write-Host "[INFO] Live runtime evidence: launching $exe (exitMs=$ExitMs timeoutSec=$TimeoutSec runId=$runtimeRunId)"

# -- 3) run the runtime ------------------------------------------------------
$startedUtc = [DateTime]::UtcNow
$process = Start-Process -FilePath $exe `
    -ArgumentList @("--cli-run", "--cli-exit-ms", "$ExitMs", "--verify-runtime-evidence-live") `
    -WorkingDirectory $repoRoot -PassThru -WindowStyle Hidden

$timedOut = $false
try {
    Wait-Process -Id $process.Id -Timeout $TimeoutSec -ErrorAction Stop
} catch {
    $timedOut = $true
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}

$exitedUtc = [DateTime]::UtcNow

try { $process.Refresh() } catch { }
$exitCode = $null
try { $exitCode = $process.ExitCode } catch { $exitCode = $null }

if ($timedOut) {
    throw "Live runtime run timed out after ${TimeoutSec}s (no evidence is accepted)."
}

if ($null -eq $exitCode) {
    throw "Live runtime run exit code is unavailable."
}

if ($exitCode -ne 0) {
    throw "Live runtime run failed: exit=$exitCode (expected 0). See the run log for the fail-closed reason."
}

# -- 4) verify (read only, no values written) --------------------------------
#    The checker runs in a child process so its `exit` cannot terminate the caller.
$checker = Join-Path $PSScriptRoot 'isr-check-evidence-live.ps1'
$checkerArgs = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $checker,
    '-StartedUtc', $startedUtc.ToString('o'),
    '-ExitedUtc', $exitedUtc.ToString('o')
)
$checkerOutput = & powershell.exe @checkerArgs 2>&1
$checkerExit = $LASTEXITCODE
foreach ($line in $checkerOutput) { Write-Host $line }
if ($checkerExit -ne 0) {
    throw "Live evidence verification failed (checker exit=$checkerExit)."
}

Write-Host "[PASS] Live runtime evidence collected and verified (runId=$runtimeRunId)"
