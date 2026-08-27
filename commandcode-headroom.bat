@echo off
setlocal

REM === Environment Variables ===
set ANTHROPIC_BASE_URL=http://127.0.0.1:8787
set HEADROOM_SKIP_UPSTREAM_CHECK=1

REM === Headroom Proxy Auto-Start (single-instance, guard on LISTENING only) ===
netstat -ano 2>nul | findstr /r ":8787.*LISTENING" >nul
if %ERRORLEVEL% EQU 0 goto :proxy_up
echo [headroom] Starting proxy on port 8787...
start /B "" /D "%~dp0" "%~dp0.venv\Scripts\headroom.exe" proxy --port 8787 --host 127.0.0.1 --mode token --target-ratio 0.40 --memory --intercept-tool-results --rpm 200 --tpm 500000 --keepalive-expiry 30 --protect-tool-results Bash,WebFetch,Read --no-telemetry
timeout /t 3 /nobreak >nul
:proxy_up

REM === Launch commandcode (cmd) with proxy ===
echo [headroom] ANTHROPIC_BASE_URL=%ANTHROPIC_BASE_URL%
echo [headroom] Proxy: http://127.0.0.1:8787
echo [headroom] MCP servers (project scope .mcp.json): serena / context-mode / aidex
echo.

cmd %*
