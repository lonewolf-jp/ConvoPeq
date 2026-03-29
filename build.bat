@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul

REM ============================================================================
REM build.bat - build script for Windows terminal (UTF-8)
REM
REM Usage:
REM   build.bat [Debug|Release] [clean] [pgo-gen | pgo-use]
REM ============================================================================

echo ==========================================
echo ConvoPeq - Build Script
echo ==========================================
echo:

set PreferredToolArchitecture=x64

REM ------------------------------------------------------------
REM Parse arguments
set "BUILD_CONFIG=Release"
if /i "%~1"=="Debug" set "BUILD_CONFIG=Debug"
if /i "%~1"=="Release" set "BUILD_CONFIG=Release"

REM ------------------------------------------------------------
REM Parse PGO mode 3rd argument
set "PGO_MODE=normal"
if /i "%~2"=="pgo-gen" set "PGO_MODE=pgo-gen"
if /i "%~2"=="pgo-use" set "PGO_MODE=pgo-use"
if /i "%~3"=="pgo-gen" set "PGO_MODE=pgo-gen"
if /i "%~3"=="pgo-use" set "PGO_MODE=pgo-use"

REM PGO用CMakeフラグ 括弧ネスト最小化・パーサー干渉完全排除
set "CMAKE_PGO_FLAGS=-DCONVOPEQ_PGO_INSTRUMENT=OFF -DCONVOPEQ_PGO_USE=OFF"
echo [INFO] Initial PGO_FLAGS: %CMAKE_PGO_FLAGS%
if "!PGO_MODE!"=="pgo-gen" (
    set "CMAKE_PGO_FLAGS=-DCONVOPEQ_PGO_INSTRUMENT=ON -DCONVOPEQ_PGO_USE=OFF"
    echo [PGO] Instrumentation build mode selected /GENPROFILE
) else (
    if "!PGO_MODE!"=="pgo-use" (
        set "CMAKE_PGO_FLAGS=-DCONVOPEQ_PGO_INSTRUMENT=OFF -DCONVOPEQ_PGO_USE=ON"
        echo [PGO] Optimized build mode selected /USEPROFILE
    ) else (
        echo [PGO] Normal build   no PGO
    )
)

echo [INFO] Final PGO_FLAGS: %CMAKE_PGO_FLAGS%
echo [INFO] BUILD_CONFIG: %BUILD_CONFIG%
echo [INFO] PGO_MODE: %PGO_MODE%

set "DO_CLEAN=0"
if /i "%~2"=="clean" set "DO_CLEAN=1"
if /i "%~3"=="clean" set "DO_CLEAN=1"

REM Release は build\Release、Debug は build\Debug
set "BUILD_ROOT=build"
set "BUILD_DIR=%BUILD_ROOT%"

REM ------------------------------------------------------------
REM Check JUCE directory
if not exist "JUCE\CMakeLists.txt" (
    echo [ERROR] JUCE directory not found or invalid!
    echo Expected: "%~dp0JUCE\CMakeLists.txt"
    echo:
    echo Please place JUCE using one of the following methods:
    echo   1. Symbolic link: mklink /J JUCE C:\path\to\JUCE
    echo   2. Junction:     mklink /J JUCE C:\path\to\JUCE
    echo   3. Copy:         xcopy /E /I C:\path\to\JUCE JUCE
    echo:
    pause
    popd
    exit /b 1
)

echo [CHECK] JUCE Directory: OK
echo:

REM ------------------------------------------------------------
REM Clean build directory if requested
if "%DO_CLEAN%"=="1" (
    echo [CLEAN] Removing "%BUILD_DIR%"...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    echo:
)

REM ------------------------------------------------------------
REM Setup MSVC environment
set "VCVARS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if exist "%VCVARS_PATH%" (
    echo [INFO] Found vcvarsall.bat. Executing...
    call "%VCVARS_PATH%" x64
    @echo off
    if errorlevel 1 (
        echo [ERROR] Failed to initialize MSVC environment.
        pause
        exit /b 1
    )
    echo [INFO] MSVC environment initialized.
) else (
    echo [ERROR] vcvarsall.bat not found:
    echo   %VCVARS_PATH%
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Setup Intel oneAPI environment
set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if exist "%ONEAPI_SETVARS%" (
    echo [INFO] Found Intel oneAPI setvars.bat. Executing...
    call "%ONEAPI_SETVARS%" intel64
    @echo off
    if errorlevel 1 (
        echo [ERROR] Failed to initialize Intel oneAPI environment.
        pause
        exit /b 1
    )
    echo [INFO] Intel oneAPI environment initialized.
) else (
    echo [ERROR] Intel oneAPI MKL not found!
    echo Please install Intel oneAPI Base Toolkit.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Create build directory
echo [1/4] Creating build directory...
if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create build directory.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Configure CMake
echo [2/4] Configuring CMake...
cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl %CMAKE_PGO_FLAGS%
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Build project
echo [3/4] Building %BUILD_CONFIG% configuration...
cmake --build "%BUILD_DIR%" --config %BUILD_CONFIG%
if errorlevel 1 (
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Verify build configuration
echo [VERIFY] Checking CMakeCache.txt...
if exist "%BUILD_DIR%\CMakeCache.txt" (
    findstr /C:"CONVOPEQ_PGO_INSTRUMENT" "%BUILD_DIR%\CMakeCache.txt"
    findstr /C:"CONVOPEQ_PGO_USE" "%BUILD_DIR%\CMakeCache.txt"
    findstr /C:"CMAKE_BUILD_TYPE" "%BUILD_DIR%\CMakeCache.txt"
) else (
    echo [WARNING] CMakeCache.txt not found
)
echo:
echo [VERIFY] Build Configuration: %BUILD_CONFIG%
echo [VERIFY] PGO Mode: %PGO_MODE%
echo:

REM ------------------------------------------------------------
REM Check build artifacts
echo [4/4] Checking build artifacts...
set "EXE_PATH=%BUILD_DIR%\ConvoPeq_artefacts\%BUILD_CONFIG%\ConvoPeq.exe"
if exist "%EXE_PATH%" (
    echo [SUCCESS] Executable created successfully.
) else (
    echo [WARNING] Executable not found at:
    echo   %EXE_PATH%
)

echo.
echo ==========================================
echo Build Complete!
echo ==========================================
echo.
echo Build configuration:
echo   %BUILD_CONFIG%
echo Build directory:
echo   %BUILD_DIR%
echo Executable location:
echo   %EXE_PATH%
echo.
echo To run:
echo   "%EXE_PATH%"
echo:
pause
endlocal

