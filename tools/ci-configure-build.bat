@echo off
setlocal EnableExtensions
REM ci-configure-build.bat - CI フラグでのクリーン configure + ISR テストビルド検証
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1
set "BUILD_DIR=C:\VSC_Project\ConvoPeq\build-ci-check"
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
cmake -S C:\VSC_Project\ConvoPeq -B "%BUILD_DIR%" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCONVO_CI_BUILD=ON -DCONVOPEQ_REQUIRE_MKL=OFF > "%BUILD_DIR%-config.log" 2>&1
echo CONFIG_EXIT=%ERRORLEVEL%
cmake --build "%BUILD_DIR%" --config Debug --target ISRSemanticValidationTests >> "%BUILD_DIR%-config.log" 2>&1
echo BUILD_EXIT=%ERRORLEVEL%
if exist "%BUILD_DIR%\Debug\ISRSemanticValidationTests.exe" (
  echo EXE_LOCATION=%BUILD_DIR%\Debug\ISRSemanticValidationTests.exe
) else (
  echo EXE_LOCATION=MISSING
)
exit /b 0
