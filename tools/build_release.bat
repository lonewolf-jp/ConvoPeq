@call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
@call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64
cmake --build C:\VSC_Project\ConvoPeq\build --config Release
