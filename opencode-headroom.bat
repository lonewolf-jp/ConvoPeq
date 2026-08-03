@echo off
setlocal

REM === Environment Variables ===
set ANTHROPIC_BASE_URL=http://127.0.0.1:8787
REM Silence: (2) /health probe HEAD 404 logs (proxy inherits this via set above)
set HEADROOM_SKIP_UPSTREAM_CHECK=1

REM === Headroom Proxy Auto-Start (single-instance, guard on LISTENING only) ===
netstat -ano 2>nul | findstr /r ":8787.*LISTENING" >nul
if %ERRORLEVEL% EQU 0 goto :proxy_up
echo [headroom] Starting proxy on port 8787...
start /B "" "%~dp0.venv\Scripts\headroom.exe" proxy --port 8787
timeout /t 3 /nobreak >nul
:proxy_up

REM === Launch opencode ===
echo [headroom] ANTHROPIC_BASE_URL=%ANTHROPIC_BASE_URL%
echo [headroom] Proxy: http://127.0.0.1:8787
echo [headroom] rtk(WSL): aliases configured in ~/.bashrc
echo [headroom] context-mode MCP: configured in opencode.json
echo.

opencode %*
