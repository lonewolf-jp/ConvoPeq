@echo off
setlocal enabledelayedexpansion
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
cd /d C:\VSC_Project\ConvoPeq\build-icx
echo === Available Test Executables ===
dir *.exe 2>nul
echo.
echo === Running CTest ===
ctest -C Debug --output-on-failure
set EXIT=%errorlevel%
echo === CTest EXIT: %EXIT% ===
exit /b %EXIT%
