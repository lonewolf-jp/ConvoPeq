@echo off
setlocal EnableExtensions
REM run-ctest-full.bat - 全テスト実行（既知除外）
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"
exit /b %ERRORLEVEL%
