@echo off
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64 >nul 2>&1
cd /d C:\VSC_Project\ConvoPeq\build-icx
ctest -C Release --output-on-failure
