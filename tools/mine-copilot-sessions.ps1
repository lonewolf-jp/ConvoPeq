#!/usr/bin/env pwsh
# mine-copilot-sessions.ps1
# Mines VS Code Copilot chat session logs into mempalace for persistent memory.
# Run via Windows Task Scheduler for automatic daily recording.
#
# Safe to run even when VS Code is closed:
# - Debug-log directories persist after VS Code exits
# - mempalace mine skips already-filed files (idempotent)
# - Errors are logged to a timestamped file for review

param(
    [string]$WorkspaceStorageId = "6ad0ec77ccfd31b2de89aa570ef9a366",
    [string]$PalaceDir = "$env:USERPROFILE\mempalace_home",
    [string]$ProjectName = "ConvoPeq"
)

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$logFile = "$env:TEMP\mempalace-mine-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
$mempalaceExe = "$env:USERPROFILE\.local\bin\mempalace.exe"

function Write-Log {
    param([string]$Msg)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $Msg" | Out-File -FilePath $logFile -Append -Encoding utf8
    Write-Host "$timestamp $Msg"
}

Write-Log "=== MemPalace auto-mine started ==="
Write-Log "Palace dir: $PalaceDir"

# ---- Step 1: Mine VS Code Copilot session logs ----
$chatLogDir = "$env:APPDATA\Code\User\workspaceStorage\$WorkspaceStorageId\GitHub.copilot-chat\debug-logs"

if (-not (Test-Path $chatLogDir)) {
    Write-Log "WARNING: Copilot chat log directory not found: $chatLogDir (VS Code not yet run?)"
} else {
    Write-Log "Mining Copilot sessions from: $chatLogDir"
    try {
        $output = & $mempalaceExe mine $chatLogDir --mode convos --wing "vscode-copilot" 2>&1
        if ($LASTEXITCODE -eq 0) {
            $filesLine = ($output | Select-String "Files processed:" -SimpleMatch)
            $drawersLine = ($output | Select-String "Drawers filed:" -SimpleMatch)
            Write-Log "Copilot sessions: $filesLine $drawersLine"
        } else {
            Write-Log "ERROR: mempalace mine exited with code $LASTEXITCODE"
            Write-Log "Output: $output"
        }
    } catch {
        Write-Log "EXCEPTION: $_"
    }
}

# ---- Step 2: Mine project context (optional, only if in project dir) ----
$projectRoot = if (Test-Path ".\CMakeLists.txt") { (Resolve-Path ".").Path } else { $null }
if ($projectRoot) {
    Write-Log "Mining project context from: $projectRoot"
    try {
        $output = & $mempalaceExe mine $projectRoot --wing $ProjectName 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Log "Project context: OK"
        } else {
            Write-Log "WARNING: project mine exit code $LASTEXITCODE (non-critical)"
        }
    } catch {
        Write-Log "WARNING: project mine exception: $_ (non-critical)"
    }
} else {
    Write-Log "Skipping project mine (not in project directory)"
}

Write-Log "=== MemPalace auto-mine completed (log: $logFile) ==="
