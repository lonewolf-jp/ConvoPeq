@echo off
setlocal enabledelayedexpansion
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
echo === Rebuild 2 test targets with MSVC ===
cd /d C:\VSC_Project\ConvoPeq\build
ninja BuildInputSemanticContractTests RuntimeWorldAuthorityProjectionTests
if %errorlevel% neq 0 (
    echo [WARN] MSVC rebuild failed, checking existing binaries
    if exist C:\VSC_Project\ConvoPeq\build\Debug\BuildInputSemanticContractTests.exe (
        echo Found existing binaries, running tests directly
        C:\VSC_Project\ConvoPeq\build\Debug\BuildInputSemanticContractTests.exe
        echo BISC exit=%errorlevel%
        C:\VSC_Project\ConvoPeq\build\Debug\RuntimeWorldAuthorityProjectionTests.exe
        echo RWAP exit=%errorlevel%
    )
)
echo === Trying icx build instead ===
cd /d C:\VSC_Project\ConvoPeq\build-icx
echo Retry linking with LIB path fix...
set "LIB=C:\Program Files (x86)\Intel\oneAPI\compiler\2026.0\lib;%LIB%"
ninja -j1 BuildInputSemanticContractTests RuntimeWorldAuthorityProjectionTests
echo icx build exit=%errorlevel%
exit /b %errorlevel%
