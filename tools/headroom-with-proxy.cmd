@echo off
setlocal

set HEADROOM_EXE=%~dp0..\.venv\Scripts\headroom.exe
set PROXY_PORT=8787
set PROXY_LOG=%TEMP%\headroom-proxy.log
set PID_FILE=%TEMP%\headroom-proxy.pid

rem Clean up orphaned proxies
taskkill /FI "IMAGENAME eq headroom.exe" /F >nul 2>&1

rem Start proxy with optimal settings (detached)
powershell -NoProfile -Command ^
    "$hp = Start-Process -FilePath '%HEADROOM_EXE%' -ArgumentList 'proxy','--port','%PROXY_PORT%','--host','127.0.0.1','--mode','token','--target-ratio','0.40','--memory','--intercept-tool-results','--rpm','200','--tpm','500000','--keepalive-expiry','30','--protect-tool-results','Bash,WebFetch,Read' -WindowStyle Hidden -PassThru -RedirectStandardOutput '%PROXY_LOG%' -RedirectStandardError '%PROXY_LOG%.err'; $hp.Id | Out-File '%PID_FILE%' -Encoding ascii; Start-Sleep -Seconds 4"

rem Run MCP server in foreground
"%HEADROOM_EXE%" mcp serve

rem Cleanup
if exist "%PID_FILE%" (
    for /f "usebackq" %%p in ("%PID_FILE%") do (
        taskkill /PID %%p /F >nul 2>&1
    )
    del "%PID_FILE%"
)
taskkill /FI "IMAGENAME eq headroom.exe" /F >nul 2>&1

endlocal
