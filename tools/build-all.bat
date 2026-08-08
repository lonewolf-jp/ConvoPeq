@echo off
setlocal EnableExtensions
REM build-all.bat - 全ターゲットの Debug ビルド
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug 2>&1 | findstr /i "error C[0-9] error LNK warning C[0-9]"
exit /b %ERRORLEVEL%
