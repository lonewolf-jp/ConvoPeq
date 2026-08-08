@echo off
setlocal EnableExtensions
REM run-ctest-filter.bat - ctest をフィルタ付きで実行
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority" -R "ISR|OwnerChannel|Retire|Mpsc|Observe|Runtime"
exit /b %ERRORLEVEL%
