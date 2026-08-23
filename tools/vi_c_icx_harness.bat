@echo off
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build-icx --config Debug --target AudioEngineHarness > C:\VSC_Project\ConvoPeq\evidence\_vi_c_icx-harness.log 2>&1
echo ICX_HARNESS_EXIT=%ERRORLEVEL%
findstr /C:"FAILED" /C:"Linking" C:\VSC_Project\ConvoPeq\evidence\_vi_c_icx-harness.log
