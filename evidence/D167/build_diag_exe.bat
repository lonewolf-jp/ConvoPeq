@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\VSC_Project\ConvoPeq\build-diag
cmake --build . --config RelWithDebInfo --target ConvoPeq
