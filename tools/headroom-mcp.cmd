@echo off
setlocal

rem Prefer the latest installed version (~/.local/bin, e.g. v0.37.0)
set HEADROOM_EXE=%USERPROFILE%\.local\bin\headroom.exe

if not exist "%HEADROOM_EXE%" (
    rem Fallback to Roaming profile version
    set HEADROOM_EXE=%USERPROFILE%\AppData\Roaming\Python\Python314\Scripts\headroom.exe
)

rem Run MCP server in foreground (no proxy — proxy is incompatible with VS Code Copilot)
"%HEADROOM_EXE%" mcp serve

endlocal
