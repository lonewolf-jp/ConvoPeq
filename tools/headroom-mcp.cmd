@echo off
setlocal

rem Use the latest installed version (Roaming profile) — v0.32.1
set HEADROOM_EXE=%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\headroom.exe

if not exist "%HEADROOM_EXE%" (
    rem Fallback to venv version
    set HEADROOM_EXE=%~dp0..\.venv\Scripts\headroom.exe
)

rem Run MCP server in foreground (no proxy — proxy is incompatible with VS Code Copilot)
"%HEADROOM_EXE%" mcp serve

endlocal
