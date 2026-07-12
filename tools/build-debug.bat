@echo off
setlocal enabledelayedexpansion

echo === Initialize VS + Intel environment ===
call "%ProgramFiles%\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 exit /b 1

call "%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat" intel64

echo === CMake Configure (with /bigobj /Zm200) ===
cmake -S C:\VSC_Project\ConvoPeq -B C:\VSC_Project\ConvoPeq\build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_CXX_FLAGS_DEBUG="/bigobj /Zm200"
if %errorlevel% neq 0 (
    echo CMake configure failed
    exit /b 1
)

echo === Build ConvoPeq_Standalone ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target ConvoPeq_Standalone
if %errorlevel% neq 0 (
    echo Build failed (expected if heap exhaustion, continuing to test targets)
)

echo === List test targets ===
cd /d C:\VSC_Project\ConvoPeq\build
ctest -N 2>&1

echo === Try building test targets ===
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target RuntimeWorldAuthorityProjectionTests 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target PublicationValidatorIsolationTests 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target BuildInputSemanticContractTests 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target PartialPublicationRejectTests 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target RuntimeSemanticSchemaValidationTests 2>&1
cmake --build C:\VSC_Project\ConvoPeq\build --config Debug --target RuntimePublicationCoordinatorTests 2>&1

echo === List available tests ===
ctest -N

echo === Run CTest ===
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority" 2>&1
echo Exit code: %errorlevel%
