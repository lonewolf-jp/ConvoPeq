@echo off
REM WORK102-PREV-01 build: UTF-8 codepage + vcvars64 + oneAPI + direct cmake --build (2-phase)
REM phase1: build ConvoPeq target first (generates JuceLibraryCode/JuceHeader.h early)
REM phase2: full build of all targets
chcp 65001 >nul
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 (
    echo [ERROR] vcvars64.bat failed.
    exit /b 1
)
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
if errorlevel 1 (
    echo [ERROR] oneAPI setvars.bat failed.
    exit /b 1
)
set "CFG=%1"
cmake --build build --config %CFG% --target ConvoPeq
if errorlevel 1 (
    echo [ERROR] ConvoPeq bootstrap failed.
    exit /b 1
)
cmake --build build --config %CFG%
exit /b %errorlevel%
