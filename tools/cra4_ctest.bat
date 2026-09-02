@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 ( echo [FATAL] vcvarsall failed & exit /b 1 )
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
set CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"

echo === CRA4: CTest Debug ===
ctest --test-dir C:\VSC_Project\ConvoPeq\build -C Debug --output-on-failure > C:\VSC_Project\ConvoPeq\evidence\cra4_ctest_debug.log 2>&1
echo CRA4_DBG_CTEST_EXIT=%errorlevel%
if %errorlevel% neq 0 ( echo [FATAL] Debug CTest failed & exit /b 1 )

echo === CRA4: CTest Release ===
ctest --test-dir C:\VSC_Project\ConvoPeq\build -C Release --output-on-failure > C:\VSC_Project\ConvoPeq\evidence\cra4_ctest_release.log 2>&1
echo CRA4_REL_CTEST_EXIT=%errorlevel%
if %errorlevel% neq 0 ( echo [FATAL] Release CTest failed & exit /b 1 )

echo CRA4_ALL_DONE
exit /b 0
