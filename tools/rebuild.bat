@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
cmake -S "C:\VSC_Project\ConvoPeq" -B "C:\VSC_Project\ConvoPeq\build" -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CXX_FLAGS_DEBUG="/Zi /INCREMENTAL:NO"
if errorlevel 1 exit /b 1
cmake --build "C:\VSC_Project\ConvoPeq\build" --config Debug
