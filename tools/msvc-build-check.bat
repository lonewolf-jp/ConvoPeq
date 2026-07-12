@echo off
setlocal enabledelayedexpansion

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64

echo === Clean build ===
if exist "C:\VSC_Project\ConvoPeq\build" rmdir /s /q "C:\VSC_Project\ConvoPeq\build"

echo === CMake Configure (MSVC Debug) ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CXX_FLAGS_DEBUG="/bigobj /Zm200"
if %errorlevel% neq 0 exit /b 1

echo === Build tests only (Debug) ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target ConvoPeq_Standalone 2>&1

echo.
echo ===== FILTERED WARNINGS (exclude nodiscard, C1060) =====
grep -iE "warning" C:\VSC_Project\ConvoPeq\build-debug.log 2>nul | findstr /v "nodiscard\|C1060" 2>nul

exit /b %errorlevel%
