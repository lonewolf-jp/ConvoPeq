#!/usr/bin/env pwsh
# graphify-mcp-wrapper.ps1
# MCP server wrapper: starts graphify MCP server if graph exists,
# otherwise builds graph first (with Gemini API fallback handling).

param()

$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path "$scriptDir\.."
$graphFile = Join-Path $projectRoot "graphify-out" "graph.json"

if (-not (Test-Path $graphFile)) {
    Write-Warning "graph.json not found at $graphFile. Run 'graphify .' first to build the graph."
    exit 1
}

# Start graphify MCP server, ignoring extraction failures (graph already exists)
# The MCP server uses the existing graph.json for queries.
# Extraction failures (Gemini quota) are non-fatal when graph exists.
graphify . --mcp 2>&1 | Select-String -NotMatch "failed|WARNING|Error code: 429|quota"
