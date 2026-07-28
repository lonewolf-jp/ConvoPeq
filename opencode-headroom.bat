@echo off
setlocal

REM === Headroom Proxy Auto-Start ===
REM Check if headroom proxy is already running on port 8787
netstat -ano 2>nul | findstr ":8787" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [headroom] Starting proxy on port 8787...
    start /B "" "%~dp0.venv\Scripts\headroom.exe" proxy --port 8787 --workers 2
    timeout /t 3 /nobreak >nul
)

REM === Environment Variables ===
set ANTHROPIC_BASE_URL=http://127.0.0.1:8787

REM === Launch opencode ===
echo [headroom] ANTHROPIC_BASE_URL=%ANTHROPIC_BASE_URL%
echo [headroom] Proxy: http://127.0.0.1:8787
echo [headroom] rtk(WSL): aliases configured in ~/.bashrc
echo [headroom] context-mode MCP: configured in opencode.json
echo.

opencode %*
