@echo off
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build C:\VSC_Project\ConvoPeq\build --config Release
