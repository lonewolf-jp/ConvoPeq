@echo off
chcp 65001 >nul
REM ============================================================================
REM build.bat - Windows用ビルドスクリプト (UTF-8)
REM
REM 使い方:
REM   build.bat [Debug|Release] [clean]
REM
REM 必要な環境:
REM   - Visual Studio 2022 (17.11以上推奨)
REM   - CMake 3.22以上
REM   - JUCE 8.0.12
REM ============================================================================

echo ==========================================
echo ConvoPeq - Build Script
echo ==========================================
echo.

set BUILD_CONFIG=Release
if /i "%1"=="Debug" set BUILD_CONFIG=Debug

if /i "%2"=="clean" (
    echo [CLEAN] Removing build directory...
    if exist "build" rmdir /s /q "build"
)

REM JUCEディレクトリの確認
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

REM r8brain-free-srcディレクトリの確認
if not exist "r8brain-free-src" (
    echo [ERROR] r8brain-free-src directory not found!
    echo.
    echo Please place r8brain-free-src using one of the following methods:
    echo   1. Symbolic link: mklink /J r8brain-free-src C:\path\to\r8brain-free-src
    echo   2. Junction: mklink /J r8brain-free-src C:\path\to\r8brain-free-src
    echo   3. Copy: xcopy /E /I C:\path\to\r8brain-free-src r8brain-free-src
    echo.
    echo r8brain-free-src Download:
    echo   https://github.com/avaneev/r8brain-free-src
    echo.
    pause
    exit /b 1
)

echo [CHECK] r8brain-free-src Directory: OK
echo.

REM Intel oneAPI 環境変数の設定 (存在する場合)
if exist "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat" (
    echo [INFO] Found Intel oneAPI setvars.bat. Executing...
    call "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\env\vars.bat"
)

REM ビルドディレクトリ作成
echo [1/4] Creating build directory...
if not exist "build" mkdir build
cd build

REM CMake設定
echo [2/4] Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo [ERROR] CMake configuration failed
    cd ..
    pause
    exit /b 1
)

REM ビルド実行
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
