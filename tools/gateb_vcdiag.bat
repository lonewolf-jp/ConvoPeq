@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d C:\VSC_Project\ConvoPeq
(
echo === vcvarsall start ===
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
echo VCVARS_RC=!errorlevel!
echo === where cl ===
where cl 2>&1
echo CL_RC=!errorlevel!
echo === cmake --version ===
cmake --version 2>&1
echo CMAKE_RC=!errorlevel!
) > "evidence\D135-8-9_GATE_B_VCVARSALL_diag.txt" 2>&1
echo DIAG_DONE
