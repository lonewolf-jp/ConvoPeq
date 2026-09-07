@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
set "CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include""
cmake --build C:\VSC_Project\ConvoPeq\build-diag --config RelWithDebInfo
