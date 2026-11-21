@echo off
setlocal
cd /d C:\Users\user\AppData\Local\Temp\opencode\bench231

call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
  echo VCVARS_FAIL
  exit /b 1
)

cl /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_avx2.obj >build_avx2.log 2>&1
cl /nologo /O2 /Ob2 /arch:AVX2 /EHsc /std:c++17 bench.cpp /Fe:bench_avx2.exe >>build_avx2.log 2>&1
echo AVX2_BUILD=%ERRORLEVEL%
cl /nologo /O2 /Ob2 /arch:SSE2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_scalar.obj >build_scalar.log 2>&1
cl /nologo /O2 /Ob2 /arch:SSE2 /EHsc /std:c++17 bench.cpp /Fe:bench_scalar.exe >>build_scalar.log 2>&1
echo SCALAR_BUILD=%ERRORLEVEL%

set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"
call "%ONEAPI_ROOT%\setvars.bat" >nul 2>&1
set "PATH=%ONEAPI_ROOT%\compiler\2026.2\bin;%ONEAPI_ROOT%\compiler\latest\bin;%PATH%"
set "LIB=%ONEAPI_ROOT%\compiler\2026.2\lib;%ONEAPI_ROOT%\compiler\2026.2\lib\clang\23\lib\windows;%LIB%"
icx-cl /nologo /O2 /Ob2 /QxCORE-AVX2 /EHsc /std:c++17 /c bench.cpp /Fo:bench_icx.obj >build_icx.log 2>&1
icx-cl /nologo /O2 /Ob2 /QxCORE-AVX2 /EHsc /std:c++17 bench.cpp /Fe:bench_icx.exe >>build_icx.log 2>&1
echo ICX_BUILD=%ERRORLEVEL%

bench_avx2.exe 1000000 25 bench_avx2.csv msvc-avx2
echo RUN_AVX2=%ERRORLEVEL%
bench_scalar.exe 1000000 25 bench_scalar.csv msvc-scalar
echo RUN_SCALAR=%ERRORLEVEL%
if exist bench_icx.exe (
  bench_icx.exe 1000000 25 bench_icx.csv icx-avx2
  echo RUN_ICX=%ERRORLEVEL%
) else (
  echo RUN_ICX=SKIPPED
)
echo ALL_DONE
