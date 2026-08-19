@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo VCVARSALL FAILED
    exit /b 1
)
echo VCVARSALL OK
cl.exe /? >nul 2>&1
if errorlevel 1 (
    echo CL.EXE NOT FOUND
    exit /b 1
)
echo CL.EXE OK
