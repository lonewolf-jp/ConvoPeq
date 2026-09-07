@echo off
setlocal EnableExtensions
rem D164-A0 restore: build-diag cache lost CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON during
rem the I3-4-H5 reconfigure (09-06 00:01). Restore G4-equivalent config WITHOUT touching
rem production source/CMake. Compiler cache values are preserved (no -DCMAKE_CXX_COMPILER
rem re-spec) to avoid the short-form 'cl' cache issue documented in the H5 report.
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
set IPP_INC="C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
call %VCVARS% >nul 2>&1
if errorlevel 1 (
  echo [ERROR] vcvars64 failed
  exit /b 1
)
set "CL=/I %IPP_INC%"
cd /d C:\VSC_Project\ConvoPeq
cmake -B build-diag -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON
if errorlevel 1 (
  echo [ERROR] CMake reconfigure failed
  exit /b 1
)
findstr /C:"CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=ON" build-diag\CMakeCache.txt >nul
if errorlevel 1 (
  echo [ERROR] DIAGNOSTICS option not ON in cache
  exit /b 1
)
cmake --build build-diag --config RelWithDebInfo --target ConvoPeq
if errorlevel 1 (
  echo [ERROR] Build failed
  exit /b 1
)
echo [OK] D164 DIAG RWDI restore build complete
exit /b 0
