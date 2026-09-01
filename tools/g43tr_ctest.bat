@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 ( echo [FATAL] vcvarsall failed & exit /b 1 )
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
set CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
echo === G-4.3-T-R: Full build Debug ===
cmake --build "C:\VSC_Project\ConvoPeq\build" --config Debug
if %errorlevel% neq 0 ( echo [FATAL] Debug build failed & exit /b 1 )
echo === G-4.3-T-R: CTest Debug ===
ctest --test-dir "C:\VSC_Project\ConvoPeq\build" -C Debug --output-on-failure
echo DBG_CTEST_EXIT=%errorlevel%
echo === G-4.3-T-R: Full build Release ===
cmake --build "C:\VSC_Project\ConvoPeq\build" --config Release
if %errorlevel% neq 0 ( echo [FATAL] Release build failed & exit /b 1 )
echo === G-4.3-T-R: CTest Release ===
ctest --test-dir "C:\VSC_Project\ConvoPeq\build" -C Release --output-on-failure
echo REL_CTEST_EXIT=%errorlevel%
exit /b 0
