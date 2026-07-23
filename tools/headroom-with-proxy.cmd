@echo off
setlocal

set HEADROOM_EXE=%~dp0..\.venv\Scripts\headroom.exe
set PROXY_PORT=8787
set PROXY_LOG=%TEMP%\headroom-proxy.log

rem Clean up orphaned proxies
taskkill /FI "IMAGENAME eq headroom.exe" /F >nul 2>&1

rem Start proxy via PowerShell (detached process with PID tracking)
powershell -NoProfile -Command ^
    "$p = Start-Process -FilePath '%HEADROOM_EXE%' -ArgumentList 'proxy', '--port', '%PROXY_PORT%', '--host', '127.0.0.1' -WindowStyle Hidden -PassThru -RedirectStandardOutput '%PROXY_LOG%' -RedirectStandardError '%PROXY_LOG%.err'; $p.Id | Out-File '%TEMP%\headroom-proxy.pid' -Encoding ascii; Start-Sleep -Seconds 4"

rem Run MCP server in foreground (this communicates with VS Code via stdio)
"%HEADROOM_EXE%" mcp serve

rem ===== Cleanup =====
if exist "%TEMP%\headroom-proxy.pid" (
    for /f "usebackq" %%p in ("%TEMP%\headroom-proxy.pid") do (
        taskkill /PID %%p /F >nul 2>&1
    )
    del "%TEMP%\headroom-proxy.pid"
)
taskkill /FI "IMAGENAME eq headroom.exe" /F >nul 2>&1

endlocal
