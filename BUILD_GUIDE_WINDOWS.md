# ConvoPeq Build Guide (Windows 11 x64)

This guide reflects the **current repository setup** — `build.bat`, `CMakeLists.txt` (v0.6.9), `CMakePresets.json`, and `.vscode/tasks.json` (22 tasks).

**Project**: ConvoPeq v0.6.9 — IR Convolution + 20-band Parametric EQ + Real-Time Analyzer
**Stack**: JUCE 8.0.12 · Intel oneMKL (sequential) · Intel IPP · AVX2 · C++20 · MSVC 19.44+ / icx 2026.0

---

## 1. Supported Environment

| Component | Specification |
|-----------|---------------|
| OS | Windows 11 x64 |
| Framework | JUCE 8.0.12 |
| Compiler (A) | MSVC 19.44+ (Visual Studio 2022 17.11+ or Visual Studio 2026) |
| Compiler (B) | Intel icx (oneAPI 2026.0) |
| Build System | CMake 3.22+ + Ninja Multi-Config |
| Language | C++20 |
| Math Backend | Intel oneMKL (sequential, static link) |
| SIMD | AVX2 (both MSVC and icx) |

ConvoPeq is a **Windows-only standalone application** (no plugin target).

---

## 2. Required Software

### 2.1 Compiler Toolchain (choose one)

**Option A — MSVC (Visual Studio)**
- Visual Studio 2022 (17.x) or Visual Studio 2026 (18.x)
- Workload: **Desktop development with C++**
- `vcvarsall.bat x64` initializes the environment

**Option B — Intel icx**
- Intel oneAPI 2026.0 (or later)
- `setvars.bat intel64` initializes the environment
- `icx.exe` handles both C and C++ (Intel recommendation for Windows)

### 2.2 Build Tools

| Tool | Version | Purpose |
|------|---------|---------|
| CMake | 3.22+ | Generator-agnostic build configuration |
| Ninja | any recent | Build system (used via `Ninja Multi-Config`) |
| Intel oneAPI Base Toolkit | 2026.0 | MKL + IPP libraries |

### 2.3 Optional Tools

| Tool | Purpose |
|------|---------|
| vswhere.exe | Auto-detects Visual Studio install path |
| clang-tidy | Static analysis (disabled by default) |
| AddressSanitizer (ASan) | Memory error detection (Debug only) |

---

## 3. Repository Layout

```
ConvoPeq/
├── build.bat                    # Primary build script
├── CMakeLists.txt               # v0.6.9, 1042 lines
├── CMakePresets.json            # 3 configure presets
├── ProjectMetadata.cmake        # APP_NAME, VERSION (v0.6.9), COMPANY, BUNDLE_ID
├── JUCE/                        # JUCE 8.0.12 (in-tree, required)
├── r8brain-free-src/            # IR resampler (optional, for 内蔵 FFT)
├── src/                         # 277 source files
├── config/                      # Authority manifests
├── tools/                       # CodeGraph, CodeQL scripts
├── .vscode/
│   ├── tasks.json               # 22 tasks
│   ├── launch.json              # 5 debug configs
│   └── c_cpp_properties.json
└── .github/scripts/             # CI/test scripts
```

`build.bat` validates `JUCE\CMakeLists.txt` before configuring.

---

## 4. build.bat (Recommended)

### 4.1 Usage

```cmd
build.bat Release              # MSVC Release (default)
build.bat Debug                # MSVC Debug
build.bat Release clean        # Clean + build
build.bat Release nopause      # No interactive pause

build.bat Release icx          # Intel icx Release
build.bat Debug   icx          # Intel icx Debug
build.bat Release clean icx    # Clean + icx

build.bat Release pgo-gen      # MSVC PGO instrumentation
build.bat Release pgo-use     # MSVC PGO optimization

build.bat Release icx pgo-gen  # ERROR: PGO not supported for icx
```

### 4.2 What build.bat Does

1. **Parses arguments**: BUILD_CONFIG (Debug/Release), PGO_MODE (normal/pgo-gen/pgo-use), DO_CLEAN, COMPILER_MODE (msvc/icx/icpx)
2. **Validates JUCE**: checks `JUCE\CMakeLists.txt` exists
3. **Initializes MSVC** (skipped for icx): uses vswhere to find latest VS, falls back to known VS17/VS18 paths
4. **Initializes oneAPI** (both MSVC and icx): calls `setvars.bat intel64`
5. **Cleans** (if `clean`): kills `cmcldeps.exe`, `ninja.exe`, `ConvoPeq.exe`; removes build dir
6. **Configures CMake**: `Ninja Multi-Config` generator, passes PGO flags as `-DCONVOPEQ_PGO_INSTRUMENT=/USE`
7. **Retries on RC1109**: for icx first-build, auto-removes stale RC resource and retries once
8. **Builds** selected config with `cmake --build`

### 4.3 Output Locations

| Compiler | Build Dir | Binary |
|----------|-----------|--------|
| MSVC | `build/` | `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe` |
| MSVC | `build/` | `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` |
| icx | `build-icx/` | `build-icx\ConvoPeq_artefacts\Debug\ConvoPeq.exe` |
| icx | `build-icx/` | `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe` |

MSVC and icx use **completely separate build directories** (`build` vs `build-icx`) to avoid CMakeCache conflicts.

---

## 5. CMakePresets.json

Three configure presets are available:

| Preset | Compiler | Binary Dir | Generator |
|--------|----------|-----------|----------|
| `vs2026-x64` | MSVC (cl) | `${sourceDir}/build` | Ninja Multi-Config |
| `icx-x64` | Intel icx | `${sourceDir}/build-icx` | Ninja Multi-Config |
| `カスタム構成のプリセット` | (custom) | `${sourceDir}/out/build/${presetName}` | Ninja |

Two build presets exist:

| Preset | Configure Preset | Config |
|--------|-----------------|--------|
| `debug` | vs2026-x64 | Debug |
| `release` | vs2026-x64 | Release |

Usage:

```cmd
cmake --preset vs2026-x64
cmake --build --preset release

cmake --preset icx-x64
cmake --build --preset release
```

---

## 6. Manual Build (Full Control)

### 6.1 MSVC

```cmd
call "C:\Program Files\Microsoft Visual Studio\[2022|2026]\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl

cmake --build build --config Debug
cmake --build build --config Release
```

Single-line PowerShell equivalent:

```powershell
cmd.exe /d /c "call \"%ProgramFiles%\Microsoft Visual Studio\[2022|2026]\VC\Auxiliary\Build\vcvarsall.bat\" x64 && call \"%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat\" intel64 && cmake -S . -B build -G \"Ninja Multi-Config\" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl && cmake --build build --config Debug"
```

### 6.2 Intel icx

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build-icx -G "Ninja Multi-Config" ^
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx

cmake --build build-icx --config Debug
cmake --build build-icx --config Release
```

**Note**: On Windows, `icx.exe` handles both C and C++ (unlike Linux which uses `icx` for C and `icpx` for C++). Both `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` are set to `icx`.

---

## 7. Compiler Flags Reference

### 7.1 MSVC (all configs)

| Flag | Value | Purpose |
|------|-------|---------|
| `/utf-8` | on | UTF-8 source + runtime |
| `/W4` | on | Warning level 4 |
| `/MP1` | on | Multi-processor compile (1 core, memory-minimal) |
| `/EHsc` | on | C++ exceptions only (no SEH) |
| `/Zm400` | on | 400% precompiled header heap |
| `/bigobj` | on | Large .obj file support |
| `/arch:AVX2` | on | AVX2 SIMD (all configs) |
| `/MT` (Release) | on | Static CRT link |
| `/MTd` (Debug) | on | Static CRT link |
| `/D_DEBUG` | Debug | Debug symbol |
| `/Od` | Debug | No optimization |
| `/Zi` | Debug | PDB debug info |
| `/RTC1` | Debug | Runtime checks |
| `/O2` | Release | Max speed |
| `/GL` | Release | Whole Program Optimization (LTO) |
| `/LTCG` | Release | Link-time code generation |
| `/OPT:REF /OPT:ICF /OPT:LBR` | Release | Linker optimization |

**Disabled warnings**: `4100` (unused param), `4189` (unused local, r8brain)

### 7.2 Intel icx (all configs)

| Flag | Value | Purpose |
|------|-------|---------|
| `-Wall -Wextra` | on | Warnings |
| `-Wno-unused-parameter` | on | Suppress unused param |
| `-Wno-unknown-argument` | on | Suppress unknown args |
| `-Wno-unused-command-line-argument` | on | Suppress unused cmd args |
| `-Wno-macro-redefined` | on | Suppress NOMINMAX redefinition |
| `/EHsc` | on | C++ exceptions |
| `/utf-8` | on | UTF-8 source |
| `/QxCORE-AVX2` | on | Haswell+ AVX2+FMA (all configs) |
| `/MT` | Release | Static CRT (icx default) |
| `/O3` | Release | Max optimization |
| `/fp:fast` | Release | Fast floating point (no denormal loss) |
| `/Gy` | Release | Function-level linking |
| `/Zi` | Release | PDB debug info |
| `/Qipo` | Release | Whole program optimization (=LTO) |

**Important**: `/fp:precise + /Qimf-arch-consistency:true` cause `LLVM ERROR: out of memory` in icx 2026.0 and are not used.

### 7.3 MSVC vs icx — Key Differences

| Aspect | MSVC | icx |
|--------|------|-----|
| AVX2 flag | `/arch:AVX2` | `/QxCORE-AVX2` |
| LTO | `/GL + /LTCG` | `/Qipo` |
| MKL linking | `find_package(MKL)` + `target_link_libraries` | `/Qmkl:sequential` compile option |
| IPO | `INTERPROCEDURAL_OPTIMIZATION_RELEASE=TRUE` | `/Qipo` target property |
| CRT | Static (`/MT` / `/MTd`) | Static (icx default) |

---

## 8. Intel oneMKL Configuration

### 8.1 Linking Strategy

**MSVC**:
```cmake
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
```

**icx**:
```cmake
target_compile_options(ConvoPeq PRIVATE /Qmkl:sequential)
# Compiler embeds MKL linking directives into .obj files
```

MKLROOT is detected from `$ENV{MKLROOT}` and added to CMAKE_PREFIX_PATH.

### 8.2 IPP Configuration

IPP is **optional** (quiet find). If found:
- `IPP::ippcore`, `IPP::ipps` are linked
- Used for supplementary DSP operations
- `R8B_IPP=1` is intentionally **not enabled** — r8brain uses its built-in FFT due to API incompatibility with IPP 2022.3+

---

## 9. AddressSanitizer (ASan)

ASan is **Debug-only** and requires **dynamic CRT** (`/MDd`).

```cmd
build.bat Debug asan=on
```

Or via CMake:

```cmd
cmake -S . -B build -DENABLE_ASAN=ON ...
```

| Compiler | ASan Effect |
|----------|------------|
| MSVC | `/fsanitize=address` + dynamic CRT override (`/MDd` for Debug) |
| icx | `-fsanitize=address` |

**Important**: Static CRT (`/MTd`) combined with MSVC ASan causes `LNK2038` mismatch error. The build system automatically switches to `/MDd` when ASan is enabled.

---

## 10. Profile-Guided Optimization (PGO) — MSVC Only

PGO is **not supported for icx**. It requires two separate builds on real workloads.

### 10.1 Step 1 — Instrumented Build

```cmd
build.bat Release pgo-gen
```

CMake flags: `-DCONVOPEQ_PGO_INSTRUMENT=ON -DCONVOPEQ_PGO_USE=OFF`
Result: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` + `*.pgc` files in same directory.

### 10.2 Step 2 — Exercise the Application

Run `ConvoPeq.exe` and use the application normally. CPU load will be ~200% during profiling.

### 10.3 Step 3 — Merge Profile Data

```cmd
cd build\ConvoPeq_artefacts\Release
pgomgr /merge *.pgc ConvoPeq.pgd
cd ..\..\..
```

Or use the full path to `pgomgr.exe`:
```
"C:\Program Files\Microsoft Visual Studio\[2022|2026]\VC\Tools\MSVC\<version>\bin\Hostx64\x64\pgomgr.exe" /merge *.pgc ConvoPeq.pgd
```

### 10.4 Step 4 — Optimized Build

```cmd
build.bat Release pgo-use
```

CMake flags: `-DCONVOPEQ_PGO_INSTRUMENT=OFF -DCONVOPEQ_PGO_USE=ON`
Uses: `/USEPROFILE:PGD=build\ConvoPeq_artefacts\Release\ConvoPeq.pgd`

Result: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` (PGO-optimized)

---

## 11. CTest Regression Suite

21 test executables are defined in CMakeLists.txt (enabled by default, `CONVOPEQ_ENABLE_ISR_TESTS=ON`).

### 11.1 Test List

| Test Name | Executable | Purpose |
|-----------|-----------|---------|
| `ISRRuntimeIdentityGenerators` | ISRRuntimeIdentityTests.exe | ISR identity generation |
| `RuntimePublicationCoordinatorRejects` | RuntimePublicationCoordinatorTests.exe | Publication coordinator rejection |
| `ISRSemanticValidationRejects` | ISRSemanticValidationTests.exe | Semantic validation |
| `RetireGraceSemantics` | RetireGraceSemanticsTests.exe | Retire grace semantics |
| `RuntimeSemanticSchemaValidation` | RuntimeSemanticSchemaValidationTests.exe | Schema validation |
| `ObservePathSingleSource` | ObservePathSingleSourceTests.exe | Observe path single source |
| `OverlapAuthoritySingular` | OverlapAuthoritySingularTests.exe | Overlap authority singular |
| `ShadowCompareContract` | ShadowCompareContractTests.exe | Shadow compare contract |
| `CrossfadeExecutorLocalContract` | CrossfadeExecutorLocalContractTests.exe | Crossfade executor local contract |
| `RuntimeWorldAuthorityProjectionContract` | RuntimeWorldAuthorityProjectionTests.exe | World authority projection |
| `PartialPublicationReject` | PartialPublicationRejectTests.exe | Partial publication rejection |
| `RebuildAdmissionRegression` | RebuildAdmissionRegressionTests.exe | Rebuild admission regression |
| `BuildInputSemanticContract` | BuildInputSemanticContractTests.exe | Build input semantic contract |
| `PriorityIntegration` | PriorityIntegrationTests.exe | Priority integration |
| `HeadlessAudioPathVerification` | cli-smoke-test.ps1 | Audio callback smoke test (skipped in CI) |

### 11.2 Running Tests

```cmd
cmake --build build --config Debug
cd build
ctest -C Debug --output-on-failure

# Exclude slow tests
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority"

# Skip audio test (CI environment)
# Set CONVO_CI_BUILD=1 to skip HeadlessAudioPathVerification
```

### 11.3 BuildInputSemanticContractTests Stack Size

This test reads large source files and may overflow the default stack. CMakeLists.txt applies:
- MSVC: `/GS-` (buffer security check off) + `/STACK:8388608` (8 MB stack)

---

## 12. Clang-Tidy Integration

clang-tidy is **disabled by default** (`CONVOPEQ_ENABLE_CLANG_TIDY=OFF`).

```cmd
cmake -S . -B build -DCONVOPEQ_ENABLE_CLANG_TIDY=ON ...
```

When enabled:
- Runs against `src/*.cpp` only (JUCE excluded via `CXX_CLANG_TIDY` empty global)
- MSVC driver mode: `--driver-mode=cl`
- Header filter: `.*/src/.*`
- JUCE modules treated as SYSTEM includes (clang-tidy warnings suppressed)

clang-tidy is invoked automatically during build if enabled and the binary is found.

---

## 13. VS Code Tasks (22 Total)

All tasks use `shell: cmd.exe`. Generator is `Ninja Multi-Config`.

### 13.1 Core Build Tasks

| Label | Description | Build Dir |
|-------|------------|-----------|
| `Debug` | MSVC Debug, default | `build/` |
| `Release` | MSVC Release, default | `build/` |
| `Debug (icx)` | Intel icx Debug | `build-icx/` |
| `Release (icx)` | Intel icx Release | `build-icx/` |

### 13.2 Utility Tasks

| Label | Description |
|-------|-------------|
| `Kill Previous Instance` | `taskkill /F /IM ConvoPeq.exe` |
| `Clean` | Remove `build/` directory |
| `CLI Smoke Test` | Run `cli-smoke-test.ps1 -KillExisting -RequireAudioCallbacks` (depends on Debug) |
| `Debug Build + Test` | Build Debug + run CTest (excludes BuildInputSemanticContract, RuntimeWorldAuthority) |

### 13.3 CodeGraph Tasks (Static Index)

| Label | Description |
|-------|-------------|
| `CodeGraph Full Index` | Run full CodeGraph indexing |
| `CodeGraph Incremental Index` | Run incremental CodeGraph indexing |
| `CodeGraph Stats` | Show CodeGraph stats |
| `CodeGraph Apply Local Patch` | Apply local CodeGraph patch |

### 13.4 CodeQL Tasks (Security Analysis)

| Label | Description |
|-------|-------------|
| `CodeQL Create DB (ConvoPeq Standard)` | Create CodeQL database |
| `CodeQL Create DB (ConvoPeq Standard DryRun)` | Dry-run the DB creation script |
| `CodeQL One-Step (ConvoPeq Standard)` | Run full CodeQL analysis |
| `CodeQL One-Step (ConvoPeq Standard DryRun)` | Dry-run the analysis |

### 13.5 PGO / Debug / Analysis Tasks

| Label | Description |
|-------|-------------|
| `Release Build With PDB` | MSVC Release with full PDB generation |
| `Debug Build (cmd env)` | Debug via `cmd.exe /d /c` chain |
| `Release Build (cmd env retry)` | Release via `cmd.exe /d /c` chain |
| `Debug Build (cmd env retry)` | Debug via `cmd.exe /d /c` chain |
| `Strict Atomic Dot-Call Scan` | PowerShell script to scan src/ for atomic dot-calls |
| `work21 EpochDomain CI Gate` | CI gate for work21 EpochDomain |

**Recommended workflow**:
- **Terminal → Run Task → Debug** (MSVC Debug)
- **Terminal → Run Task → Release** (MSVC Release)
- **Terminal → Run Task → Debug (icx)** / **Release (icx)**
- **Terminal → Run Task → Clean** (when switching toolchains or after cache conflicts)

---

## 14. VS Code Debug Configurations (launch.json)

Five configurations are available:

| Config | Program | Working Dir | Pre-Launch |
|--------|---------|------------|------------|
| `ConvoPeq Debug` | `build\...\Debug\ConvoPeq.exe` | `build\...\Debug` | Debug task |
| `ConvoPeq Debug (CLI)` | same | same | Debug task |
| `ConvoPeq Release` | `build\...\Release\ConvoPeq.exe` | `build\...\Release` | Release task |
| `ConvoPeq Release (Ninja Build)` | `build\Release\ConvoPeq.exe` | repo root | Release task |
| `ConvoPeq Crash調査 (例外停止)` | `build\Release\ConvoPeq.exe` | repo root | Release task |

The `PATH` environment in all configs includes:
```
C:\Program Files (x86)\Intel\oneAPI\mkl\latest\redist\intel64
```
plus the MSVC toolchain path for debugging.

---

## 15. Common Issues and Fixes

### A) `windows.h` or standard headers not found

**Cause**: MSVC/SDK environment not initialized in the same command chain.

**Fix**:
- Use `build.bat`, or
- Ensure both `vcvarsall.bat` and `setvars.bat` are called before CMake in the same shell session
- In PowerShell: `cmd.exe /d /c "... && ..."`

### B) `Release` task produces Debug artifacts

**Cause**: Single-config generator or cache mismatch.

**Fix**:
- Use `Ninja Multi-Config` (not `Ninja`)
- Build with `--config Release`
- Run `Clean` task and reconfigure

### C) `'C:\Program' is not recognized`

**Cause**: Broken quoting or shell mismatch.

**Fix**:
- Keep task shell as `cmd.exe`
- Use `%ProgramFiles%` / `%ProgramFiles(x86)%` environment variables
- Keep quoted paths exactly as in current `tasks.json`

### D) oneMKL not found

**Cause**: oneAPI environment not initialized.

**Fix**:
- Install Intel oneAPI Base Toolkit
- Confirm `C:\Program Files (x86)\Intel\oneAPI\setvars.bat` exists
- Run from a clean shell

### E) JUCE check fails in build.bat

**Cause**: Missing or invalid local `JUCE` folder.

**Fix**:
- Ensure `JUCE\CMakeLists.txt` exists under repository root
- Create via symbolic link: `mklink /J JUCE C:\path\to\JUCE`
- Or junction: `mklink /J JUCE C:\path\to\JUCE`
- Or copy: `xcopy /E /I C:\path\to\JUCE JUCE`

### F) ASan causes LNK2038 (runtime library mismatch)

**Cause**: ASan requires dynamic CRT (`/MDd`) but project defaults to static CRT (`/MTd`) on Debug.

**Fix**:
- Build system automatically overrides `MSVC_RUNTIME_LIBRARY` to `MultiThreadedDebugDLL` when `ENABLE_ASAN=ON`
- Use `build.bat Debug asan=on`

### G) icx first build fails with RC1109

**Cause**: Ninja + icx + cmcldeps creates stale RC resource file on first configure.

**Fix**:
- `build.bat` automatically detects RC1109 on first build, removes the stale `.res` file, and retries once
- If it persists: manually delete `build-icx\CMakeFiles\ConvoPeq_rc_lib.dir\<Config>\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res`

### H) PGO on icx fails

**Cause**: PGO is MSVC-only. icx does not support `/GENPROFILE` or `/USEPROFILE`.

**Fix**:
- Use MSVC for PGO: `build.bat Release pgo-gen` / `build.bat Release pgo-use`
- icx + PGO is planned for a future phase

---

## 16. CMake Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `CONVOPEQ_ENABLE_CLANG_TIDY` | OFF | Run clang-tidy during build |
| `CONVOPEQ_ENABLE_ISR_TESTS` | ON | Build CTest regression suite |
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | OFF | Enable runtime diagnostic logging |
| `CONVOPEQ_PGO_INSTRUMENT` | OFF | PGO instrumentation build |
| `CONVOPEQ_PGO_USE` | OFF | PGO optimized build |
| `ENABLE_ASAN` | OFF | AddressSanitizer (Debug only) |

---

## 17. Dependency Boundaries

Do **not** modify external dependency trees directly:

- `JUCE/` — JUCE framework (in-tree)
- `r8brain-free-src/` — IR resampler library

---

## 18. Architecture Notes

- ConvoPeq is a **standalone app** (not a plugin target)
- The **default daily workflow** is `build.bat` or VS Code tasks
- Use `Clean` when switching toolchains or after generator/cache conflicts
- **MSVC and icx modes are fully isolated** (separate build directories) and can coexist
- **icx binaries require Intel CPU** (AVX2 check enforced at runtime)
- **PGO is MSVC-only**; `.pgd` stored in `build\ConvoPeq_artefacts\Release\`
- **CI skips audio test** when `CONVO_CI_BUILD=1` is defined (no audio device in CI)
- **BuildInputSemanticContractTests** requires 8 MB stack on MSVC

---

## 19. Quick Reference

```cmd
# MSVC Debug
build.bat Debug

# MSVC Release
build.bat Release

# MSVC Release with PGO
build.bat Release pgo-gen
# (exercise app, then)
build.bat Release pgo-use

# Intel icx Debug
build.bat Debug icx

# Intel icx Release
build.bat Release icx

# Clean
build.bat Release clean

# ASan Debug
build.bat Debug asan=on

# Run tests
cmake --build build --config Debug
cd build && ctest -C Debug --output-on-failure
```

---

## Appendix A. Source File Map

```
ConvoPeq.exe sources (CMakeLists.txt target_sources):
  src/MainApplication.cpp
  src/MainWindow.cpp
  src/audioengine/
    AudioEngine.*.cpp  (15 files — Lifecycle, Timer, Commit, Rebuild, etc.)
    ISR*.cpp           (31 files — Closure, PayloadTier, HB, Retire, etc.)
    Processing.*.cpp   (11 files — AudioBlock, BlockDouble, DSPCore*, etc.)
  src/convolver/
    ConvolverProcessor.*.cpp  (8 files — Lifecycle, Rebuild, LoaderThread, etc.)
  src/eqprocessor/
    EQProcessor.*.cpp  (5 files — Core, Parameters, Coefficients, Processing, etc.)
  src/core/
    GlobalSnapshot.*, SnapshotCoordinator.*, EpochDomain.h, RCUReader.h, etc.
  src/CustomInputOversampler.cpp
  src/TruePeakDetector.cpp
  src/LoudnessMeter.cpp
  src/MKLNonUniformConvolver.cpp
  src/NoiseShaperLearner.cpp
  src/RuntimeBuilder.cpp
  src/AudioEngineProcessor.cpp
  + 21 test executables in src/tests/
```

## Appendix B. Build Directory Structure

```
build/                              # MSVC build root
├── CMakeCache.txt
├── CMakeFiles/
├── ConvoPeq_artefacts/
│   ├── Debug/ConvoPeq.exe
│   └── Release/ConvoPeq.exe
│       └── ConvoPeq.pgd            # (after PGO use phase)
└── ConvoPeq_artefacts/Debug/      # test executables also here

build-icx/                          # icx build root (fully isolated)
└── (same structure)

out/build/<presetName>/              # CMakePresets custom build dir
```

## Appendix C. MSVC Version Detection

`build.bat` auto-detects Visual Studio via vswhere, then falls back to known paths in order:

```
VS 18 Enterprise  → C:\Program Files\Microsoft Visual Studio\18\Enterprise\...
VS 18 Professional → ...
VS 18 Community   → ...
VS 17 Enterprise  → ...
VS 17 Professional → ...
VS 17 Community   → ...
```

CMakeLists.txt specifies `MSVC 19.44+` (VS2022 17.11+), so VS17 or VS18 both satisfy this requirement.
