@echo off
setlocal EnableExtensions EnableDelayedExpansion

chcp 65001 >nul 2>&1

REM ============================================================================
REM build.bat - build script for Windows terminal (UTF-8)
REM
REM Usage:
REM   build.bat [Debug|Release] [clean] [nopause] [pgo-gen | pgo-use] [icx|icpx] [-DVAR]
REM
REM   -DVAR[=VALUE] : CMake definition (SHIFT解析方式対応、引用符不要).
REM           Examples:
REM             build.bat Release nopause -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
REM             build.bat Debug icx -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
REM             build.bat Release "-DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF"
REM   Note: CMAKE_EXTRA_FLAGS に ! を含めることは非推奨（DelayedExpansionとの競合）。
REM ============================================================================

echo ==========================================
echo ConvoPeq - Build Script
echo ==========================================
echo:

set PreferredToolArchitecture=x64

REM ------------------------------------------------------------
REM Parse arguments
set "BUILD_CONFIG=Release"
set "PGO_MODE=normal"
set "DO_CLEAN=0"
set "NO_PAUSE=0"
set "COMPILER_MODE=msvc"
set "CMAKE_EXTRA_FLAGS="

:argloop
if "%1"=="" goto :argend
set "arg=%1"
if /I "%arg:~0,2%"=="-D" (
    set "CMAKE_EXTRA_FLAGS=!CMAKE_EXTRA_FLAGS! %arg%"
    echo [INFO] Extra CMake define: %arg%
) else (
    if /i "%arg%"=="Debug" set "BUILD_CONFIG=Debug"
    if /i "%arg%"=="Release" set "BUILD_CONFIG=Release"
    if /i "%arg%"=="clean" set "DO_CLEAN=1"
    if /i "%arg%"=="nopause" set "NO_PAUSE=1"
    if /i "%arg%"=="pgo-gen" set "PGO_MODE=pgo-gen"
    if /i "%arg%"=="pgo-use" set "PGO_MODE=pgo-use"
    if /i "%arg%"=="icx"   set "COMPILER_MODE=icx"
    if /i "%arg%"=="icpx"  set "COMPILER_MODE=icpx"
)
shift
goto :argloop
:argend

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
echo [INFO] COMPILER_MODE: !COMPILER_MODE!
echo [INFO] Extra CMake flags: !CMAKE_EXTRA_FLAGS!
REM icx PGO exclusion: icx/icpx does not support PGO
if not "!COMPILER_MODE!"=="msvc" if not "!PGO_MODE!"=="normal" (
    echo [ERROR] PGO is only supported with MSVC compiler.
    echo [ERROR] icx/icpx PGO support is planned for a future phase.
    call :maybe_pause
    exit /b 1
)

REM Use a single build directory for Ninja Multi-Config.
REM Output binaries are placed under build\ConvoPeq_artefacts\[Config].
set "BUILD_ROOT=build"
REM コンパイラに応じてビルドディレクトリを分離（CMakeCache衝突防止）
if "!COMPILER_MODE!"=="icx" set "BUILD_ROOT=build-icx"
if "!COMPILER_MODE!"=="icpx" set "BUILD_ROOT=build-icx"
set "BUILD_DIR=!BUILD_ROOT!"

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
    call :maybe_pause
    exit /b 1
)

echo [CHECK] JUCE Directory: OK
echo:

REM ------------------------------------------------------------
REM Ensure stale juceaide sub-build cache does not cause generator mismatch.
REM JUCE's juceaide bootstrap (JUCE/extras/Build/juceaide/CMakeLists.txt:136)
REM invokes a sub-build at ${JUCE_BINARY_DIR}/tools/ with -G${CMAKE_GENERATOR}.
REM If the generator changes (e.g., VS2026 <-> Ninja) while the tools sub-cache
REM remains, CMake fails with "generator does not match the generator used previously".
REM We delete the tools sub-cache on every reconfigure to prevent this.
if exist "%BUILD_ROOT%\JUCE\tools\CMakeCache.txt" (
    echo [CLEAN] Removing stale juceaide sub-build cache...
    if exist "%BUILD_ROOT%\JUCE\tools\CMakeFiles" rmdir /s /q "%BUILD_ROOT%\JUCE\tools\CMakeFiles"
    del /q "%BUILD_ROOT%\JUCE\tools\CMakeCache.txt" 2>nul
    echo [CLEAN] Stale juceaide cache removed.
)

REM ------------------------------------------------------------
REM Clean build directory if requested
if "%DO_CLEAN%"=="1" (
    echo [CLEAN] Removing "%BUILD_ROOT%"...
    taskkill /F /IM cmcldeps.exe >nul 2>&1
    taskkill /F /IM ninja.exe >nul 2>&1
    taskkill /F /IM ConvoPeq.exe >nul 2>&1
    timeout /t 2 >nul
    if exist "%BUILD_ROOT%" rmdir /s /q "%BUILD_ROOT%"
    echo:
)

REM ------------------------------------------------------------
REM Setup environment for icx mode
set "ONEAPI_SETVARS=C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
set "ONEAPI_LIB=C:\Program Files (x86)\Intel\oneAPI"
if not "!COMPILER_MODE!"=="icx" goto setup_done

echo [INFO] icx mode: MSVC vcvarsall.bat skipped (icx auto-detects Windows SDK).
echo [INFO] Initializing Intel oneAPI environment...
call "%ONEAPI_SETVARS%" intel64
if errorlevel 1 (
    echo [ERROR] Failed to initialize Intel oneAPI environment.
    call :maybe_pause
    exit /b 1
)
set "ONEAPI_LIB=C:\Program Files (x86)\Intel\oneAPI"
set "MKLROOT=%ONEAPI_LIB%\mkl\latest"
set "IPPROOT=%ONEAPI_LIB%\ipp\latest"
set "ONEAPI_ROOT=%ONEAPI_LIB%"
if exist "!ONEAPI_LIB!\mkl\latest\lib" set "LIB=!ONEAPI_LIB!\mkl\latest\lib;!LIB!"
if exist "!ONEAPI_LIB!\ipp\latest\lib" set "LIB=!ONEAPI_LIB!\ipp\latest\lib;!LIB!"
if exist "!ONEAPI_LIB!\compiler\latest\lib" set "LIB=!ONEAPI_LIB!\compiler\latest\lib;!LIB!"
echo [INFO] Intel oneAPI environment initialized.

:setup_done

REM ------------------------------------------------------------
REM Create build directory
echo [1/4] Creating build directory...
if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if errorlevel 1 (
    echo [ERROR] Failed to create build directory.
    call :maybe_pause
    exit /b 1
)

REM ------------------------------------------------------------
REM Configure CMake
set "CMAKE_CONFIG_ATTEMPT=0"
:configure_cmake
set /a CMAKE_CONFIG_ATTEMPT+=1
echo [2/4] Configuring CMake... (attempt !CMAKE_CONFIG_ATTEMPT!)
if "!COMPILER_MODE!"=="msvc" (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_EXPORT_COMPILE_COMMANDS=ON %CMAKE_PGO_FLAGS% %CMAKE_EXTRA_FLAGS%
) else (
    cmake -S . -B "%BUILD_DIR%" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx -DCMAKE_EXPORT_COMPILE_COMMANDS=ON %CMAKE_EXTRA_FLAGS%
)
if not errorlevel 1 goto configure_cmake_ok

if !CMAKE_CONFIG_ATTEMPT! GEQ 3 (
    echo [ERROR] CMake configuration failed after !CMAKE_CONFIG_ATTEMPT! attempts.
    call :maybe_pause
    exit /b 1
)

echo [WARN] CMake configure failed. Cleaning transient Ninja state and retrying...
if exist "%BUILD_DIR%\.ninja_log" del /f /q "%BUILD_DIR%\.ninja_log" >nul 2>&1
if exist "%BUILD_DIR%\.ninja_deps" del /f /q "%BUILD_DIR%\.ninja_deps" >nul 2>&1
timeout /t 2 >nul
goto configure_cmake

:configure_cmake_ok

REM ------------------------------------------------------------
REM Copy compile_commands.json to project root for clangd/serena LSP
if exist "%BUILD_DIR%\compile_commands.json" (
    copy /y "%BUILD_DIR%\compile_commands.json" "%~dp0compile_commands.json" >nul
    echo [INFO] compile_commands.json copied to project root for clangd.
)
REM Convert icx/MSVC-specific flags to clangd-compatible ones
if exist "%~dp0compile_commands.json" if exist "%~dp0tools\fix_compile_commands_for_clangd.ps1" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\fix_compile_commands_for_clangd.ps1" -InputFile "%~dp0compile_commands.json"
    echo [INFO] compile_commands.json flags converted for clangd.
)
REM Ensure compile_commands.json has no UTF-8 BOM (clangd fails to parse it otherwise).
if exist "%~dp0compile_commands.json" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p='%~dp0compile_commands.json'; $t=[System.IO.File]::ReadAllText($p); if ($t.StartsWith([char]0xFEFF)) { $t=$t.Substring(1) }; $u=New-Object System.Text.UTF8Encoding($false); [System.IO.File]::WriteAllText($p,$t,$u); '[INFO] compile_commands.json BOM removed, if present.'"
)

REM ------------------------------------------------------------
REM Clean stale RC output: JUCE generates RC during CMake configure
REM which may conflict with Ninja build. Remove stale RC file.
if not "!COMPILER_MODE!"=="msvc" if exist "!BUILD_DIR!\CMakeFiles\ConvoPeq_rc_lib.dir\!BUILD_CONFIG!\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res" (
    echo [INFO] Removing stale RC resource file to avoid RC1109...
    del /f /q "!BUILD_DIR!\CMakeFiles\ConvoPeq_rc_lib.dir\!BUILD_CONFIG!\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res" >nul 2>&1
)

REM ------------------------------------------------------------
REM Build project (with automatic retry for RC1109 on first icx build)
echo [3/4] Building %BUILD_CONFIG% configuration...
set "BUILD_RETRY=0"
:build_retry
set "NINJA_FLAGS="
REM icx/icpx は /Qipo リンク等でメモリを大量消費するため並列度を -j 1 に抑える（LLVM out of memory 回避）
if /i "!COMPILER_MODE!"=="icx" set "NINJA_FLAGS=-- -j 1"
if /i "!COMPILER_MODE!"=="icpx" set "NINJA_FLAGS=-- -j 1"
cmake --build "%BUILD_DIR%" --config %BUILD_CONFIG% %NINJA_FLAGS%
if not errorlevel 1 goto build_ok

REM icx 初回ビルドで RC1109 発生時は一度だけリトライ
if not "!COMPILER_MODE!"=="msvc" if "!BUILD_RETRY!"=="0" (
    echo [WARN] Build failed. Checking for RC1109 (common on first icx build)...
    if exist "%BUILD_DIR%\CMakeFiles\ConvoPeq_rc_lib.dir\%BUILD_CONFIG%\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res" (
        echo [INFO] Removing stale RC resource and retrying build...
        del /f /q "%BUILD_DIR%\CMakeFiles\ConvoPeq_rc_lib.dir\%BUILD_CONFIG%\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res" >nul 2>&1
    )
    set "BUILD_RETRY=1"
    goto build_retry
)

echo [ERROR] Build failed.
call :maybe_pause
exit /b 1

:build_ok

if /i "!COMPILER_MODE!"=="icx" (
    echo [INFO] Building Phase 8 test targets...
    cmake --build "%BUILD_DIR%" --config %BUILD_CONFIG% --target GainStagingContractTests --target EQProcessorMaxGainTests %NINJA_FLAGS% >nul 2>&1
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
set "EXE_PATH=!BUILD_DIR!\ConvoPeq_artefacts\!BUILD_CONFIG!\ConvoPeq.exe"
if exist "!EXE_PATH!" (
    echo [SUCCESS] Executable created successfully.
) else (
    echo [WARNING] Executable not found at:
    echo   !EXE_PATH!
)

echo.
echo ==========================================
echo Build Complete!
echo.
echo Build configuration: %BUILD_CONFIG%
echo Build directory:
echo   %BUILD_DIR%
echo Executable location:
echo   !EXE_PATH!
echo.
echo To run:
echo   "!EXE_PATH!"
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
call :maybe_pause
endlocal

goto :eof

:maybe_pause
if "%NO_PAUSE%"=="1" goto :eof
pause
goto :eof

:setup_oneapi_done
for /f "usebackq tokens=1,* delims==" %%A in ("%ONEAPI_ENV_TMP%") do (
    if /i "%%A"=="PATH" set "PATH=%%B;!PATH!"
    if /i "%%A"=="INCLUDE" set "INCLUDE=%%B;!INCLUDE!"
    if /i "%%A"=="LIB" set "LIB=%%B;!LIB!"
    if /i "%%A"=="ICX_HOME" set "ICX_HOME=%%B"
    if /i "%%A"=="MKLROOT" set "MKLROOT=%%B"
    if /i "%%A"=="IPPROOT" set "IPPROOT=%%B"
    if /i "%%A"=="ONEAPI_ROOT" set "ONEAPI_ROOT=%%B"
)
del /q "%ONEAPI_ENV_TMP%" 2>nul
exit /b 0


