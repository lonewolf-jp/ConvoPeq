@echo off
rem ============================================================
rem D101-9 Step 5-VI-C helper: build + ctest for one config
rem Usage: vi_c_build_test.bat <Debug|Release> <tag>
rem Logs:  evidence\_vi_c_<tag>-build.log / _vi_c_<tag>-ctest.log
rem ============================================================
setlocal
set CFG=%1
set TAG=%2
if "%CFG%"=="" exit /b 1
if "%TAG%"=="" exit /b 1

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul
cd /d C:\VSC_Project\ConvoPeq\build-icx

echo === BUILD %CFG% ===
rem D101-9 5-VI-C: build all ctest test executables explicitly (full-build stops on
rem pre-existing ConvoPeq.exe icx link issue; test exes are independent targets)
set TESTTARGETS=PublicationAdmissionTests TerminalTelemetryContractTests ISRSoakTests ISRSemanticValidationTests invariant_INV3_INV5Tests RetireGraceSemanticsTests ShutdownRetireIntentDrainTests StuckReaderFallbackDrainTests PriorityIntegrationTests OwnerChannelTests DeferredDeletionQueueReclaimTests MpscBoundedRingTests SequenceArithmeticTests DSPHandleTableTests GainStagingContractTests EQProcessorMaxGainTests EQAnalysisUnitTests FFTBackendTests EQBoundExcessBenchmark ISRRuntimeIdentityTests RuntimePublicationCoordinatorTests NormalRetireDSPHandleCompareTests RuntimeSemanticSchemaValidationTests ObservePathSingleSourceTests OverlapAuthoritySingularTests ShadowCompareContractTests CrossfadeExecutorLocalContractTests RuntimeWorldAuthorityProjectionTests PartialPublicationRejectTests RebuildAdmissionRegressionTests BuildInputSemanticContractTests MTNUPCMeasurement AudioEngineHarness
cmake --build . --config %CFG% --target %TESTTARGETS% > C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log 2>&1
set BEXIT=%ERRORLEVEL%
rem ★ Workaround for pre-existing add_dependencies(<Test> ConvoPeq) chains: ConvoPeq.exe
rem   fails to link with icx Debug (_CrtDbgReport), skipping dependent test exes.
rem   Build those exe OUTPUT edges directly (bypasses phony dependency level).
if "%CFG%"=="Debug" set CFGDIR=Debug
if "%CFG%"=="Release" set CFGDIR=Release
if "%CFG%"=="RelWithDebInfo" set CFGDIR=RelWithDebInfo
set DIRECTEXES=%CFGDIR%\PublicationAdmissionTests.exe %CFGDIR%\TerminalTelemetryContractTests.exe %CFGDIR%\ISRSoakTests.exe %CFGDIR%\ISRSemanticValidationTests.exe %CFGDIR%\invariant_INV3_INV5Tests.exe %CFGDIR%\RetireGraceSemanticsTests.exe %CFGDIR%\ShutdownRetireIntentDrainTests.exe %CFGDIR%\StuckReaderFallbackDrainTests.exe %CFGDIR%\PriorityIntegrationTests.exe
ninja %DIRECTEXES% >> C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log 2>&1
set BEXIT2=%ERRORLEVEL%
echo BUILD_EXIT=%BEXIT% DIRECT_EXIT=%BEXIT2%
findstr /C:"FAILED:" C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log
findstr /C:"error:" C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log | findstr /C:"icx" | findstr /V /C:"warning"
echo === CTEST %CFG% ===
ctest -C %CFG% --output-on-failure > C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-ctest.log 2>&1
set CEXIT=%ERRORLEVEL%
echo CTEST_EXIT=%CEXIT%
findstr /C:"tests passed" /C:"tests failed" /C:"Subproject" /C:"tests timed out" C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-ctest.log
endlocal & exit /b 0
