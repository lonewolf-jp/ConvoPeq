@echo off
setlocal
call "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 exit /b 1
cd /d C:\VSC_Project\ConvoPeq
set OUT=evidence\st1\build
if not exist %OUT% mkdir %OUT%
set CLPATH=C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64
set DEFS=/nologo /TP /DJUCE_DSP_USE_INTEL_MKL=1 /DJUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1 /DJUCE_MODULE_AVAILABLE_juce_core=1 /DJUCE_MODULE_AVAILABLE_juce_data_structures=1 /DJUCE_MODULE_AVAILABLE_juce_events=1 /DJUCE_MODULE_AVAILABLE_juce_gui_basics=1 /DJUCE_MODULE_AVAILABLE_juce_gui_extra=1 /DNOMINMAX /DR8B_EXTFILTERS=0 /DR8B_FASTTIMING=1 /DUNICODE /D_CRT_SECURE_NO_WARNINGS /D_UNICODE /DWIN32 /D_WINDOWS
set INC=/IC:\VSC_Project\ConvoPeq /IC:\VSC_Project\ConvoPeq\src /IC:\VSC_Project\ConvoPeq\src\audioengine /IC:\VSC_Project\ConvoPeq\src\core /IC:\VSC_Project\ConvoPeq\src\convolver /IC:\VSC_Project\ConvoPeq\src\eqprocessor /IC:\VSC_Project\ConvoPeq\build\ConvoPeq_artefacts\JuceLibraryCode /IC:\VSC_Project\ConvoPeq\JUCE\modules /external:I"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include" /external:I"C:\Program Files (x86)\Intel\oneAPI\ipp\latest\include" /external:IC:\VSC_Project\ConvoPeq\r8brain-free-src /external:W0

echo [1/4] compile Debug obj
"%CLPATH%\cl.exe" %DEFS% /DDEBUG=1 /D_DEBUG=1 /DCMAKE_INTDIR=\"Debug\" %INC% /EHsc /D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /std:c++20 /MDd /FS /Fo%OUT%\st1_debug.obj /Fd%OUT%\ -c evidence\st1\D152R2_ST1_AVStress200.cpp
if errorlevel 1 exit /b 2

echo [2/4] compile Release obj
"%CLPATH%\cl.exe" %DEFS% /DNDEBUG /DCMAKE_INTDIR=\"Release\" %INC% /EHsc /Zm400 /bigobj /O2 /Ob2 /fp:fast /Gw /Gy /Zi /utf-8 /std:c++20 /MD /FS /Fo%OUT%\st1_release.obj /Fd%OUT%\ -c evidence\st1\D152R2_ST1_AVStress200.cpp
if errorlevel 1 exit /b 3

set BASE=CMakeFiles\ISRSemanticValidationTests.dir
set LIBS="C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\mkl_intel_lp64.lib" "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\mkl_sequential.lib" "C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\mkl_core.lib" kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib

echo [3/4] link Debug exe (D154 objs + driver obj)
"%CLPATH%\link.exe" /nologo /machine:x64 /debug /INCREMENTAL:NO /subsystem:console /OUT:%OUT%\st1_av_stress_debug.exe %OUT%\st1_debug.obj build\%BASE%\Debug\src\audioengine\ISRClosure.cpp.obj build\%BASE%\Debug\src\audioengine\ISRPayloadTier.cpp.obj build\%BASE%\Debug\src\audioengine\ISRRetireRouter.cpp.obj build\%BASE%\Debug\src\audioengine\ISRRetire.cpp.obj build\%BASE%\Debug\src\audioengine\ISRRetireRuntimeEx.cpp.obj build\%BASE%\Debug\src\audioengine\ISRRuntimePublicationCoordinator.cpp.obj build\%BASE%\Debug\src\audioengine\ISRDSPHandle.cpp.obj build\%BASE%\Debug\src\audioengine\ISRDSPQuarantine.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_core\juce_core.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_core\juce_core_CompilationTime.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_gui_extra\juce_gui_extra.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_gui_basics\juce_gui_basics.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_graphics\juce_graphics.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_graphics\juce_graphics_Harfbuzz.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_graphics\juce_graphics_Sheenbidi.c.obj build\%BASE%\Debug\JUCE\modules\juce_events\juce_events.cpp.obj build\%BASE%\Debug\JUCE\modules\juce_data_structures\juce_data_structures.cpp.obj %LIBS%
if errorlevel 1 exit /b 4

echo [4/4] link Release exe (D154 objs + driver obj)
"%CLPATH%\link.exe" /nologo /machine:x64 /INCREMENTAL:NO /subsystem:console /OUT:%OUT%\st1_av_stress_release.exe %OUT%\st1_release.obj build\%BASE%\Release\src\audioengine\ISRClosure.cpp.obj build\%BASE%\Release\src\audioengine\ISRPayloadTier.cpp.obj build\%BASE%\Release\src\audioengine\ISRRetireRouter.cpp.obj build\%BASE%\Release\src\audioengine\ISRRetire.cpp.obj build\%BASE%\Release\src\audioengine\ISRRetireRuntimeEx.cpp.obj build\%BASE%\Release\src\audioengine\ISRRuntimePublicationCoordinator.cpp.obj build\%BASE%\Release\src\audioengine\ISRDSPHandle.cpp.obj build\%BASE%\Release\src\audioengine\ISRDSPQuarantine.cpp.obj build\%BASE%\Release\JUCE\modules\juce_core\juce_core.cpp.obj build\%BASE%\Release\JUCE\modules\juce_core\juce_core_CompilationTime.cpp.obj build\%BASE%\Release\JUCE\modules\juce_gui_extra\juce_gui_extra.cpp.obj build\%BASE%\Release\JUCE\modules\juce_gui_basics\juce_gui_basics.cpp.obj build\%BASE%\Release\JUCE\modules\juce_graphics\juce_graphics.cpp.obj build\%BASE%\Release\JUCE\modules\juce_graphics\juce_graphics_Harfbuzz.cpp.obj build\%BASE%\Release\JUCE\modules\juce_graphics\juce_graphics_Sheenbidi.c.obj build\%BASE%\Release\JUCE\modules\juce_events\juce_events.cpp.obj build\%BASE%\Release\JUCE\modules\juce_data_structures\juce_data_structures.cpp.obj %LIBS%
if errorlevel 1 exit /b 5

echo BUILD OK
exit /b 0
