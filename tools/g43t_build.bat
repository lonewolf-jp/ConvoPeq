@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 ( echo [FATAL] vcvarsall failed & exit /b 1 )
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
set CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"
echo === G-4.3-T: Build ISRSemanticValidationTests (Debug) ===
cmake --build "C:\VSC_Project\ConvoPeq\build" --target ISRSemanticValidationTests --config Debug
if %errorlevel% neq 0 ( echo [FATAL] Debug build failed & exit /b 1 )
echo === G-4.3-T: Run ISRSemanticValidationTests (Debug) ===
"C:\VSC_Project\ConvoPeq\build\Debug\ISRSemanticValidationTests.exe"
echo DBG_EXIT=%errorlevel%
echo === G-4.3-T: Build ISRSemanticValidationTests (Release) ===
cmake --build "C:\VSC_Project\ConvoPeq\build" --target ISRSemanticValidationTests --config Release
if %errorlevel% neq 0 ( echo [FATAL] Release build failed & exit /b 1 )
echo === G-4.3-T: Run ISRSemanticValidationTests (Release) ===
"C:\VSC_Project\ConvoPeq\build\Release\ISRSemanticValidationTests.exe"
echo REL_EXIT=%errorlevel%
exit /b 0
