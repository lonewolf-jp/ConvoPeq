@echo off
setlocal
rem Build and run the decimate benchmark that produced the CSV files here.
rem Run it from any directory; it works in the folder that holds itself.
cd /d "%~dp0"
if errorlevel 1 (
  echo CD_FAIL
  exit /b 1
)

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo VSWHERE_NOT_FOUND
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
  echo VS_NOT_FOUND
  exit /b 1
)

call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo VCVARS_FAIL
  exit /b 1
)

del /q bench.obj bench_avx2.obj bench_scalar.obj bench_icx.obj bench_avx2.exe bench_scalar.exe bench_icx.exe 2>nul

cl /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_avx2.obj >build_avx2.log 2>&1
if errorlevel 1 (
  echo AVX2_BUILD_FAIL
  type build_avx2.log
  goto :fail
)
cl /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 bench.cpp /Fe:bench_avx2.exe >>build_avx2.log 2>&1
if errorlevel 1 (
  echo AVX2_LINK_FAIL
  type build_avx2.log
  goto :fail
)

cl /nologo /O2 /Ob2 /arch:SSE2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_scalar.obj >build_scalar.log 2>&1
if errorlevel 1 (
  echo SCALAR_BUILD_FAIL
  type build_scalar.log
  goto :fail
)
cl /nologo /O2 /Ob2 /arch:SSE2 /EHsc /std:c++17 bench.cpp /Fe:bench_scalar.exe >>build_scalar.log 2>&1
if errorlevel 1 (
  echo SCALAR_LINK_FAIL
  type build_scalar.log
  goto :fail
)

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
if exist "%ONEAPI_ROOT%\setvars.bat" (
  call "%ONEAPI_ROOT%\setvars.bat" >nul 2>&1
  set "PATH=%ONEAPI_ROOT%\compiler\2026.2\bin;%ONEAPI_ROOT%\compiler\latest\bin;%PATH%"
  set "LIB=%ONEAPI_ROOT%\compiler\2026.2\lib;%ONEAPI_ROOT%\compiler\2026.2\lib\clang\23\lib\windows;%LIB%"
  icx-cl /nologo /O2 /Ob2 /QxCORE-AVX2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_icx.obj >build_icx.log 2>&1
  if errorlevel 1 (
    echo ICX_OBJ_FAIL
    type build_icx.log
    goto :fail
  )
  icx-cl /nologo /O2 /Ob2 /QxCORE-AVX2 /EHsc /std:c++17 bench.cpp /Fe:bench_icx.exe >>build_icx.log 2>&1
  if errorlevel 1 (
    echo ICX_LINK_FAIL
    type build_icx.log
    goto :fail
  )
)

call :runbench bench_avx2.exe bench_avx2.csv msvc-avx2
if errorlevel 1 goto :fail
call :runbench bench_scalar.exe bench_scalar.csv msvc-scalar
if errorlevel 1 goto :fail
if exist bench_icx.exe (
  call :runbench bench_icx.exe bench_icx.csv icx-avx2
  if errorlevel 1 goto :fail
)

del /q bench.obj bench_avx2.obj bench_scalar.obj bench_icx.obj bench_avx2.exe bench_scalar.exe bench_icx.exe build_avx2.log build_scalar.log build_icx.log run_tmp.exe 2>nul
echo ALL_DONE
exit /b 0

:runbench
rem A freshly linked exe can be briefly locked by the virus scanner; retry.
rem ping is used instead of timeout because timeout needs a console.
set "EXE=%~1"
set "CSV=%~2"
set "VAR=%~3"
rem Some scanners keep a fresh exe open, so run a copy under a new name.
copy /b "%EXE%" run_tmp.exe >nul 2>&1
if errorlevel 1 (
  echo COPY_FAIL %EXE%
  exit /b 1
)
set /a TRIES=0
:runbench_retry
run_tmp.exe 1000000 25 "%CSV%" "%VAR%"
if not errorlevel 1 (
  del /q run_tmp.exe 2>nul
  exit /b 0
)
set /a TRIES+=1
if %TRIES% GEQ 10 (
  echo RUN_FAIL %EXE%
  del /q run_tmp.exe 2>nul
  exit /b 1
)
ping -n 3 127.0.0.1 >nul
goto :runbench_retry

:fail
del /q bench.obj bench_avx2.obj bench_scalar.obj bench_icx.obj bench_avx2.exe bench_scalar.exe bench_icx.exe 2>nul
echo FAILED
exit /b 1
