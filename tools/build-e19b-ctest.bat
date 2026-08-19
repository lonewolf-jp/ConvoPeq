@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d C:\VSC_Project\ConvoPeq\build
ctest -C Debug -R "RetireGraceSemantics|DeferredDeletionQueueReclaim|ShutdownRetireIntentDrain|StuckReaderFallbackDrain|ISRRuntimeIdentity|ISRSemanticValidation|OwnerChannel|NormalRetireDSPHandleCompare|RuntimeSemanticSchema|ObservePathSingleSource|OverlapAuthoritySingular|ShadowCompareContract|CrossfadeExecutorLocalContract|RuntimeWorldAuthorityProjection|PartialPublicationReject|RebuildAdmissionRegression|MpscBoundedRing|SequenceArithmetic|DSPHandleTable|PriorityIntegration" --output-on-failure 2>&1
