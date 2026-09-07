set "CL="
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
set "INCLUDE=C:\Progra~2\Intel\oneAPI\2026.1\include;%INCLUDE%"
cd /d C:\VSC_Project\ConvoPeq\build-msvc
cmake --build .
