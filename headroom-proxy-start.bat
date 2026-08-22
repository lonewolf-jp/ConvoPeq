@echo off
REM Auto-start headroom proxy for token optimization
REM Single-instance guard: only one Launcher should hold the proxy.
REM Guard is based on LISTENING sockets only (netstat ":8787" without
REM LISTENING would false-positive on leftover TIME_WAIT connections and
REM make this start twice while the real server is already running).

netstat -ano 2>nul | findstr /r ":8787.*LISTENING" >nul
if %ERRORLEVEL% EQU 0 goto :eof

REM Single worker avoids the multi-worker CCR WARNING; skip-upstream-check
REM silences the recurring HEAD ... 404 log noise from the /health probe.
set HEADROOM_SKIP_UPSTREAM_CHECK=1
set HEADROOM_TELEMETRY=off
set HEADROOM_ROLLOUT_CHANNEL=canary
REM Pin cwd to the project dir so the .headroom memory DB is created here
REM (not in an inherited non-writable cwd like System32, which crashed with
REM  PermissionError: [WinError 5] ...\.headroom on logon auto-start).
start "headroom-proxy" /B /D "C:\VSC_Project\ConvoPeq" ^
  "C:\VSC_Project\ConvoPeq\.venv\Scripts\headroom.exe" ^
  proxy --port 8787 --host 127.0.0.1 --mode token --target-ratio 0.40 ^
  --memory --intercept-tool-results --rpm 200 --tpm 500000 ^
  --keepalive-expiry 30 --protect-tool-results Bash,WebFetch,Read --no-telemetry --code-aware
