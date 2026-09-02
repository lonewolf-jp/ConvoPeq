@echo off
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call %VCVARS% >nul 2>&1
cd /d C:\VSC_Project\ConvoPeq\evidence
"C:\Program Files (x86)\Windows Kits\10\Debuggers\x64\cdb.exe" -G -g -o -lines -y "C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\Debug" -c "sxe av; g; .ecxr; k 40; q" "C:\VSC_Project\ConvoPeq\build-diag\ConvoPeq_artefacts\Debug\ConvoPeq.exe" --cli-run --cli-log-file D162-2B_cdb.log --cli-ir D162-1P_active.wav --cli-ir-reload-count 6 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 6 --cli-intent-burst-interval-ms 6000 --cli-exit-ms 120000 > C:\VSC_Project\ConvoPeq\evidence\D162-2B_cdb_stack.log 2>&1
echo CDB_EXIT=%ERRORLEVEL%
