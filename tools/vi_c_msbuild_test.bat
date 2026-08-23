@echo off
rem ============================================================
rem D101-9 Step 5-VI-C helper: build + ctest (MSVC build/ dir)
rem Usage: vi_c_msbuild_test.bat <Debug|Release> <tag>
rem   Uses C:\VSC_Project\ConvoPeq\build (Visual Studio generator).
rem   Rationale: build-icx (Ninja) has a PRE-EXISTING ConvoPeq.exe
rem   icx Debug link failure (_CrtDbgReport); 9 test targets carry
rem   add_dependencies(<Test> ConvoPeq) and get skipped there.
rem Logs: evidence\_vi_c_<tag>-build.log / _vi_c_<tag>-ctest.log
rem ============================================================
setlocal
set CFG=%1
set TAG=%2
if "%CFG%"=="" exit /b 1
if "%TAG%"=="" exit /b 1

call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul
call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64 >nul

echo === BUILD %CFG% (MSVC build/) ===
cmake --build C:\VSC_Project\ConvoPeq\build --config %CFG% --target ^
 PublicationAdmissionTests TerminalTelemetryContractTests ISRSoakTests ISRSemanticValidationTests invariant_INV3_INV5Tests RetireGraceSemanticsTests ShutdownRetireIntentDrainTests StuckReaderFallbackDrainTests PriorityIntegrationTests OwnerChannelTests DeferredDeletionQueueReclaimTests MpscBoundedRingTests SequenceArithmeticTests DSPHandleTableTests GainStagingContractTests EQProcessorMaxGainTests EQAnalysisUnitTests FFTBackendTests EQBoundExcessBenchmark ISRRuntimeIdentityTests RuntimePublicationCoordinatorTests NormalRetireDSPHandleCompareTests RuntimeSemanticSchemaValidationTests ObservePathSingleSourceTests OverlapAuthoritySingularTests ShadowCompareContractTests CrossfadeExecutorLocalContractTests RuntimeWorldAuthorityProjectionTests PartialPublicationRejectTests RebuildAdmissionRegressionTests BuildInputSemanticContractTests MTNUPCMeasurement AudioEngineHarness RuntimeHealthMonitorTierTests ^
 -- > C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log 2>&1
set BEXIT=%ERRORLEVEL%
echo BUILD_EXIT=%BEXIT%
findstr /C:": error" C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-build.log | findstr /V /C:"warning"

echo === CTEST %CFG% ===
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C %CFG% --output-on-failure > C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-ctest.log 2>&1
set CEXIT=%ERRORLEVEL%
echo CTEST_EXIT=%CEXIT%
findstr /C:"tests passed" /C:"tests failed" /C:"tests timed out" C:\VSC_Project\ConvoPeq\evidence\_vi_c_%TAG%-ctest.log
endlocal & exit /b 0
