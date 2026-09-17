@echo off
REM WORK102-PREV-01 bootstrap: build ConvoPeq target first to generate JuceLibraryCode/JuceHeader.h
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
cmake --build build --config Release --target ConvoPeq
exit /b %errorlevel%
