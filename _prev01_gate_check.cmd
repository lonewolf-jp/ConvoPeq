@echo off
REM WORK102-PREV-01 gate check wrapper
chcp 65001 >nul
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64
python src\tools\build_identity_gate.py --build-dir build --check --config %1
exit /b %errorlevel%
