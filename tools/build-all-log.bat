@echo off
setlocal EnableExtensions
REM build-all-log.bat - 全ターゲットの Debug ビルドをログに記録
set "LOGFILE=C:\VSC_Project\ConvoPeq\build\all-build-final.log"
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1
echo === FULL BUILD === > "%LOGFILE%"
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug >> "%LOGFILE%" 2>&1
echo === EXIT: %ERRORLEVEL% === >> "%LOGFILE%"
exit /b %ERRORLEVEL%
