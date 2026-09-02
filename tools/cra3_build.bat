@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 ( echo [FATAL] vcvarsall failed & exit /b 1 )
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
set CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"

echo === CRA3: Configure ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build > C:\VSC_Project\ConvoPeq\evidence\cra3_configure.log 2>&1
echo CRA3_CONFIGURE_EXIT=%errorlevel%
if %errorlevel% neq 0 ( echo [FATAL] configure failed & exit /b 1 )

echo === CRA3: Build Debug ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug > C:\VSC_Project\ConvoPeq\evidence\cra3_build_debug.log 2>&1
echo CRA3_DBG_EXIT=%errorlevel%
if %errorlevel% neq 0 ( echo [FATAL] Debug build failed & exit /b 1 )

echo === CRA3: Build Release ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Release > C:\VSC_Project\ConvoPeq\evidence\cra3_build_release.log 2>&1
echo CRA3_REL_EXIT=%errorlevel%
if %errorlevel% neq 0 ( echo [FATAL] Release build failed & exit /b 1 )

echo CRA3_ALL_DONE
exit /b 0
