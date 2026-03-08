@echo off
chcp 65001 >nul
REM ============================================================================
REM build.bat - build script for windows terminal (UTF-8)
REM
REM How to use:
REM   build.bat [Debug|Release] [clean]
REM
REM Build environment:
REM   - Visual Studio 2022 (17.11 or later)
REM   - CMake 3.22 or later
REM   - JUCE 8.0.12 (if you use other version, build will fail.)
REM   - Intel oneAPI
REM ============================================================================

echo ==========================================
echo ConvoPeq - Build Script
echo ==========================================
echo.

REM force to use 64-bit tool chain.
set PreferredToolArchitecture=x64

set BUILD_CONFIG=Release
if /i "%1"=="Debug" set BUILD_CONFIG=Debug

if /i "%2"=="clean" (
    echo [CLEAN] Removing build directory...
    if exist "build" rmdir /s /q "build"
)

REM Searching JUCE framework dicectory.
if not exist "JUCE" (
    echo [ERROR] JUCE directory not found!
    echo.
    echo Please place JUCE using one of the following methods:
    echo   1. Symbolic link: mklink /J JUCE C:\path\to\JUCE
    echo   2. Junction: mklink /J JUCE C:\path\to\JUCE
    echo   3. Copy: xcopy /E /I C:\path\to\JUCE JUCE
    echo.
    echo JUCE 8.0.12 Download:
    echo   https://github.com/juce-framework/JUCE/releases/tag/8.0.12
    echo.
    pause
    exit /b 1
)

echo [CHECK] JUCE Directory: OK
echo.

REM setting up Intel oneAPI environment. (if exist.)
if exist "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" (
    echo [INFO] Found Intel oneAPI setvars.bat. Executing...
    call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" intel64
) else (
    echo [ERROR] Intel oneAPI MKL not found!
    echo Please install Intel oneAPI Base Toolkit.
    pause
    exit /b 1
)

REM making build directory.
echo [1/4] Creating build directory...
if not exist "build" mkdir build
cd build

REM setting up CMake.
echo [2/4] Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64 -T host=x64
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    cd ..
    pause
    exit /b 1
)

REM building project.
echo [3/4] Building %BUILD_CONFIG% configuration...
echo   (This may take a while...)
cmake --build . --config %BUILD_CONFIG%
if errorlevel 1 (
    echo [ERROR] Build failed
    cd ..
    pause
    exit /b 1
)

echo [4/4] Checking build artifacts...
if exist "ConvoPeq_artefacts\%BUILD_CONFIG%\ConvoPeq.exe" (
    echo [SUCCESS] Executable created successfully
) else (
    echo [WARNING] Executable not found
)

cd ..

echo.
echo ==========================================
echo Build Complete!
echo ==========================================
echo.
echo Executable location:
echo   build\ConvoPeq_artefacts\%BUILD_CONFIG%\ConvoPeq.exe
echo.
echo To run:
echo   1. cd build\ConvoPeq_artefacts\%BUILD_CONFIG%
echo   2. ConvoPeq.exe
echo.
echo Or from VS Code:
echo   - Ctrl+Shift+B to Build
echo   - F5 to Debug
echo.
pause
