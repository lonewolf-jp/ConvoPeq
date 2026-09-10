# ConvoPeq Build Guide (Windows 11 x64)

This guide reflects the **current repository setup** — `build.bat` (347 lines), `CMakeLists.txt` (v0.6.10, 2,030 lines), `CMakePresets.json`, and `.vscode/tasks.json` (77 task entries).

**Project**: ConvoPeq v0.6.10 — IR Convolution + 20-band Parametric EQ + Real-Time Analyzer
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
| SIMD | AVX2 (MSVC: all configs; icx: Release only) |

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
| Python 3 | 3.x | Build identity gate (`src/tools/build_identity_gate.py`) |

### 2.3 Optional Tools

| Tool | Purpose |
|------|---------|
| clang-tidy | Static analysis (disabled by default) |
| AddressSanitizer (ASan) | Memory error detection (`ENABLE_ASAN=ON`) |
| ThreadSanitizer (TSan) | Data-race detection (`ENABLE_TSAN=ON`, Clang only) |

---

## 3. Repository Layout

```
ConvoPeq/
├── build.bat                    # Primary build script (347 lines)
├── CMakeLists.txt               # v0.6.10, 2,030 lines
├── CMakePresets.json            # 3 configure presets
├── ProjectMetadata.cmake        # APP_NAME, VERSION (v0.6.10), COMPANY, BUNDLE_ID
├── JUCE/                        # JUCE 8.0.12 (in-tree, required)
├── r8brain-free-src/            # IR resampler library
├── src/                         # 336 source files
│   ├── [87 root files]          # DSP + UI + FFT abstraction
│   ├── audioengine/ (126)       # ISR Runtime Governance
│   ├── core/         (40)       # RCU Foundation
│   ├── convolver/    (10)       # Convolver Split
│   ├── eqprocessor/  (17)       # EQ Split + Analysis
│   ├── tests/        (55)       # CTest suite (36 executables)
│   ├── dsp/math/     ( 1)       # FastTanhApprox.h
│   └── tools/        ( 2)       # Build gate Python scripts
├── config/                      # Authority manifests (4 JSON)
├── tools/                       # 66 .py + 54 .bat build/verify scripts
├── .vscode/
│   ├── tasks.json               # 77 task entries (68 unique labels)
│   ├── launch.json              # 5 debug configs
│   └── c_cpp_properties.json
└── .github/scripts/             # CI/test scripts (184 files)
```

`build.bat` validates `JUCE\CMakeLists.txt` before configuring and runs a build identity gate after configure.

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
build.bat Release pgo-use      # MSVC PGO optimization

build.bat Release icx pgo-gen  # ERROR: PGO not supported for icx

# Pass extra CMake defines (-D prefix, SHIFT parsing, no quotes needed):
build.bat Release nopause -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
build.bat Debug icx -DCONVOPEQ_REQUIRE_MKL=OFF
```

> **Note**: Do not put `!` inside `-D` values — it conflicts with DelayedExpansion.

### 4.2 What build.bat Does

1. **Parses arguments**: `BUILD_CONFIG` (Debug/Release), `PGO_MODE` (normal/pgo-gen/pgo-use), `DO_CLEAN`, `NO_PAUSE`, `COMPILER_MODE` (msvc/icx/icpx), `CMAKE_EXTRA_FLAGS` (`-DVAR[=VALUE]`)
2. **Validates JUCE**: checks `JUCE\CMakeLists.txt` exists
3. **Cleans juceaide sub-build cache**: removes stale `JUCE/tools/CMakeCache.txt` to prevent generator-mismatch errors
4. **Environment setup**:
   - **icx mode**: calls `setvars.bat intel64`; sets `MKLROOT` / `IPPROOT` / `LIB`
   - **MSVC mode**: does **not** call `vcvarsall.bat` — the shell must already be initialized (Developer Command Prompt, VS Code task, or manual `vcvarsall.bat`)
5. **Cleans** (if `clean`): kills `cmcldeps.exe`, `ninja.exe`, `ConvoPeq.exe`; removes build dir
6. **Configures CMake**: `Ninja Multi-Config` generator, passes PGO flags + extra `-D` defines. Retries up to 3 times on configure failure (clears `.ninja_log` / `.ninja_deps`)
7. **Build identity gate**: runs `src/tools/build_identity_gate.py --check` (fail-closed on identity mismatch)
8. **Builds** selected config with `cmake --build` (icx uses `-j 1`)
9. **RC1109 auto-retry**: for icx first-build, removes stale RC resource and retries once
10. **Post-build verification**: checks CMakeCache PGO flags and artifact existence

> **ASan is not a build.bat argument.** Use CMake directly: `-DENABLE_ASAN=ON` (see §9).

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

### 7.1 MSVC (ConvoPeq target)

| Flag | Value | Purpose |
|------|-------|---------|
| `/utf-8` | on | UTF-8 source + runtime |
| `/W4` | on | Warning level 4 |
| `/wd4100` | on | Suppress unused param (JUCE) |
| `/wd4189` | on | Suppress unused local (r8brain) |
| `/MP1` | on | Multi-processor compile (1 core, memory-minimal) |
| `/EHsc` | on | C++ exceptions only (no SEH) |
| `/Zm400` | on | 400% precompiled header heap |
| `/bigobj` | on | Large .obj file support |
| `/arch:AVX2` | on | AVX2 SIMD (all configs) |
| `/MT` (Release) | on | Static CRT link |
| `/MTd` (Debug) | on | Static CRT link |

**Global flags** (CMAKE_CXX_FLAGS_*):

| Config | Flags |
|--------|-------|
| Release | `/Zm400 /bigobj /O2 /Ob2 /DNDEBUG /Gw /Gy /Zi /utf-8 /EHsc` |
| Debug | `/D_DEBUG /bigobj /Zm400 /Ob0 /Od /Zi /RTC1 /utf-8 /EHsc` |
| RelWithDebInfo | `/Zi /O2 /Ob1 /DNDEBUG /utf-8` |

**Linker (Release)**: `/DEBUG /LTCG /OPT:REF /OPT:ICF /OPT:LBR`

> **`/fp:fast` is intentionally NOT used for MSVC** (work92 C-8). MSVC uses the default `/fp:precise` to guarantee DSP numeric accuracy and standard AVX2 behavior on AMD CPUs.

### 7.2 Intel icx (ConvoPeq target)

| Flag | Value | Purpose |
|------|-------|---------|
| `-Wall -Wextra` | on | Warnings |
| `-Wno-unused-parameter` | on | Suppress unused param |
| `-Wno-unknown-argument` | on | Suppress unknown args |
| `-Wno-unused-command-line-argument` | on | Suppress unused cmd args |
| `-Wno-macro-redefined` | on | Suppress NOMINMAX redefinition |
| `/EHsc` | on | C++ exceptions |
| `/utf-8` | on | UTF-8 source |
| `-mvzeroupper` | CXX only | Auto-insert vzeroupper at AVX→SSE boundaries |
| `/QxCORE-AVX2` | **Release only** | Haswell+ AVX2+FMA (config-gated) |
| `/MT` | Release | Static CRT (when ASan off) |
| `/O2` | Release | Max optimization (`/O3` causes LLVM OOM on JUCE) |
| `/fp:fast` | Release | Fast floating point (icx performance choice) |
| `/Gy` | Release | Function-level linking |
| `/Zi` | Release | PDB debug info |
| `/Qipo` | Release | Whole program optimization (=LTO) |

**Important**:
- `/fp:precise + /Qimf-arch-consistency:true` cause `LLVM ERROR: out of memory` in icx 2026.0 and are not used.
- `/O3` causes `LLVM ERROR: out of memory` on large JUCE TUs; `/O2` is used instead.
- **icx binaries are officially supported on Intel CPUs only.** AMD execution is unsupported (AVX2 codegen verified within AVX2 subset but not guaranteed on all paths).

### 7.3 MSVC vs icx — Key Differences

| Aspect | MSVC | icx |
|--------|------|-----|
| AVX2 flag | `/arch:AVX2` (all configs) | `/QxCORE-AVX2` (**Release only**) |
| LTO | `/GL + /LTCG` | `/Qipo` |
| MKL linking | `find_package(MKL)` + `target_link_libraries` | `/Qmkl:sequential` compile option |
| IPO | `INTERPROCEDURAL_OPTIMIZATION_RELEASE=TRUE` | `/Qipo` target property |
| CRT | Static (`/MT` / `/MTd`) | Static (`/MT` Release, when ASan off) |
| FP mode | `/fp:precise` (default) | `/fp:fast` (Release) |

---

## 8. Intel oneMKL Configuration

### 8.1 Linking Strategy

**MSVC** (`CONVOPEQ_REQUIRE_MKL=ON`, default):
```cmake
set(MKL_LINK static)
set(MKL_THREADING sequential)
set(MKL_INTERFACE_FULL intel_lp64)
find_package(MKL REQUIRED CONFIG COMPONENTS intel_lp64 sequential)
target_link_libraries(ConvoPeq PRIVATE MKL::MKL)
```

**MSVC without MKL** (`CONVOPEQ_REQUIRE_MKL=OFF`):
- Falls back to system aligned allocator
- Useful for environments without oneAPI installed

**icx**:
```cmake
target_compile_options(ConvoPeq PRIVATE /Qmkl:sequential)
# Compiler embeds MKL linking directives into .obj files
```

MKLROOT is detected from `$ENV{MKLROOT}` and added to CMAKE_PREFIX_PATH.

### 8.2 IPP Configuration

IPP is **optional** (quiet find). If found:
- `IPP::ippcore`, `IPP::ipps` are linked
- Used for supplementary DSP operations and the FFTBackend abstraction
- `R8B_IPP=1` is intentionally **not enabled** — r8brain uses its built-in FFT due to API incompatibility with IPP 2022.3+

---

## 9. AddressSanitizer (ASan)

ASan requires **dynamic CRT** and a CRT-consistent ASan runtime DLL.

| CRT | Required ASan runtime DLL |
|-----|--------------------------|
| `/MDd` (Debug) | `clang_rt.asan_dbg_dynamic-*.dll` |
| `/MD` (RelWithDebInfo / Release) | `clang_rt.asan_dynamic-*.dll` |

Enable via CMake (not a build.bat argument):

```cmd
cmake -S . -B build -DENABLE_ASAN=ON ...
```

| Compiler | ASan Effect |
|----------|------------|
| MSVC | `/fsanitize=address` + dynamic CRT override (Debug → `/MDd`) + `/RTC1-` strip |
| icx | `-fsanitize=address` + dynamic CRT override |

**Important**:
- Static CRT (`/MTd`) combined with MSVC ASan causes `LNK2038` mismatch. The build system automatically switches to dynamic CRT when ASan is enabled.
- **ASan + PGO is mutually exclusive** — CMake errors if both are ON.
- **CRT must match the ASan runtime DLL.** `/MDd` requires `clang_rt.asan_dbg_dynamic-*.dll`; if only the release DLL is deployed, `/MDd` binaries hang or abort with `bad-free`. **Recommended flow: build test targets in `RelWithDebInfo`** (`/MD`):
  ```cmd
  cmake --build build_asan --config RelWithDebInfo --target <TestTarget>
  ```
  Copy `clang_rt.asan_dynamic-x86_64.dll` next to the resulting `.exe`. Run with `ASAN_OPTIONS=detect_leaks=0` to suppress LSan.
- `/RTC1` is incompatible with ASan. The build system strips it via `$<$<CONFIG:Debug>:/RTC1->`.

---

## 10. ThreadSanitizer (TSan) — Clang Only

TSan is available for Clang builds only (`ENABLE_TSAN=ON`). MSVC is not supported.

```cmd
cmake -S . -B build-clang -DENABLE_TSAN=ON ...
```

| Constraint | Detail |
|------------|--------|
| Compiler | Clang only (MSVC → CMake FATAL_ERROR) |
| Mutually exclusive | Cannot combine with `ENABLE_ASAN` |
| CRT | Requires dynamic CRT (static CRT incompatible) |
| LTO | Incompatible with LTCG/IPO |

---

## 11. Profile-Guided Optimization (PGO) — MSVC Only

PGO is **not supported for icx**. It requires two separate builds on real workloads.

### 11.1 Step 1 — Instrumented Build

```cmd
build.bat Release pgo-gen
```

CMake flags: `-DCONVOPEQ_PGO_INSTRUMENT=ON -DCONVOPEQ_PGO_USE=OFF`
Result: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` + `*.pgc` files in same directory.

### 11.2 Step 2 — Exercise the Application

Run `ConvoPeq.exe` and use the application normally. CPU load will be ~200% during profiling.

### 11.3 Step 3 — Merge Profile Data

```cmd
cd build\ConvoPeq_artefacts\Release
pgomgr /merge *.pgc ConvoPeq.pgd
cd ..\..\..
```

Or use the full path to `pgomgr.exe`:
```
"C:\Program Files\Microsoft Visual Studio\[2022|2026]\VC\Tools\MSVC\<version>\bin\Hostx64\x64\pgomgr.exe" /merge *.pgc ConvoPeq.pgd
```

### 11.4 Step 4 — Optimized Build

```cmd
build.bat Release pgo-use
```

CMake flags: `-DCONVOPEQ_PGO_INSTRUMENT=OFF -DCONVOPEQ_PGO_USE=ON`
Uses: `/USEPROFILE:PGD=build\ConvoPeq_artefacts\Release\ConvoPeq.pgd`

Result: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` (PGO-optimized)

---

## 12. CTest Regression Suite

**36 test executables** are defined in CMakeLists.txt, registering **40 `add_test()`** entries (enabled by default, `CONVOPEQ_ENABLE_ISR_TESTS=ON`).

### 12.1 Test List

| Test Name | Executable | Purpose |
|-----------|-----------|---------|
| `ISRRuntimeIdentityGenerators` | ISRRuntimeIdentityTests | ISR identity generation |
| `RuntimePublicationCoordinatorRejects` | RuntimePublicationCoordinatorTests | Publication coordinator rejection |
| `ISRSemanticValidationRejects` | ISRSemanticValidationTests | Semantic validation |
| `InvariantINV3INV5` | invariant_INV3_INV5Tests | INV-3 / INV-5 invariants |
| `AdmissionPackedState` | AdmissionPackedStateTests | Admission packed-state access |
| `RetireGraceSemantics` | RetireGraceSemanticsTests | Retire grace semantics |
| `ShutdownRetireIntentDrain` | ShutdownRetireIntentDrainTests | Shutdown retire-intent drain |
| `StuckReaderFallbackDrain` | StuckReaderFallbackDrainTests | Stuck-reader fallback drain |
| `NormalRetireDSPHandleCompare` | NormalRetireDSPHandleCompareTests | Normal-retire DSP handle comparison |
| `RuntimeSemanticSchemaValidation` | RuntimeSemanticSchemaValidationTests | Schema validation |
| `ObservePathSingleSource` | ObservePathSingleSourceTests | Observe path single source |
| `OverlapAuthoritySingular` | OverlapAuthoritySingularTests | Overlap authority singular |
| `ShadowCompareContract` | ShadowCompareContractTests | Shadow compare contract |
| `CrossfadeExecutorLocalContract` | CrossfadeExecutorLocalContractTests | Crossfade executor local contract |
| `RuntimeWorldAuthorityProjectionContract` | RuntimeWorldAuthorityProjectionTests | World authority projection |
| `PartialPublicationReject` | PartialPublicationRejectTests | Partial publication rejection |
| `RebuildAdmissionRegression` | RebuildAdmissionRegressionTests | Rebuild admission regression |
| `BuildInputSemanticContract` | BuildInputSemanticContractTests | Build input semantic contract |
| `BuildErrorClassificationTests` | BuildErrorClassificationTests | BuildError / FailureClassification contract |
| `RetrySchedulerTests` | RetrySchedulerTests | RetryScheduler schedule/dispatch |
| `PublicationAdmissionTests` | PublicationAdmissionTests | Publication admission evaluation |
| `D8_1_WrapperCacheTests` | D8_1_WrapperCacheTests | Wrapper cache (D8-1) |
| `D8_2_B_2_Tests` | D8_2_B_2_Tests | D8-2-B-2 contract |
| `TerminalTelemetryContract` | TerminalTelemetryContractTests | Terminal telemetry contract |
| `RuntimeHealthMonitorTierTests` | RuntimeHealthMonitorTierTests | Health monitor tier selection |
| `ISRSoakTests` | ISRSoakTests | ISR soak / stress |
| `OwnerChannel` | OwnerChannelTests | OwnerChannel SPSC transfer |
| `DeferredDeletionQueueReclaimTests` | DeferredDeletionQueueReclaimTests | Deferred deletion reclaim |
| `MpscBoundedRingTests` | MpscBoundedRingTests | MPSC bounded ring contract |
| `SequenceArithmeticTests` | SequenceArithmeticTests | Modular sequence arithmetic |
| `DSPHandleTableTests` | DSPHandleTableTests | DSPHandleTable O(1) map |
| `PriorityIntegration` | PriorityIntegrationTests | Priority integration |
| `GainStagingContractTests` | GainStagingContractTests | Auto gain staging contract |
| `EQProcessorMaxGainTests` | EQProcessorMaxGainTests | EQ max gain response math |
| `EQAnalysisUnitTests` | EQAnalysisUnitTests | EQ analysis unit tests |
| `FFTBackendTests` | FFTBackendTests | FFTBackend abstraction |
| `EQBoundExcessBenchmark` | EQBoundExcessBenchmark | boundExcessDb benchmark (--quick) |
| `MTNUPCMeasurement` | MTNUPCMeasurement | MT-NUPC measurement console |
| `AudioEngineHarness` | AudioEngineHarness | Audio engine integration harness |
| `HeadlessAudioPathVerification` | cli-smoke-test.ps1 | Audio callback smoke test (CI-gated) |

### 12.2 Running Tests

```cmd
cmake --build build --config Debug
cd build
ctest -C Debug --output-on-failure

# Exclude slow tests
ctest -C Debug --output-on-failure -E "BuildInputSemanticContract|RuntimeWorldAuthority|ISRSoak"

# Skip audio test (CI environment)
# Set CONVO_CI_BUILD=1 to skip HeadlessAudioPathVerification
```

### 12.3 BuildInputSemanticContractTests Stack Size

This test reads large source files and may overflow the default stack. CMakeLists.txt applies:
- MSVC: `/GS-` (buffer security check off) + `/STACK:8388608` (8 MB stack)

---

## 13. Clang-Tidy Integration

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

## 14. VS Code Tasks (77 Entries, 68 Unique Labels)

All tasks use `shell: cmd.exe`. Generator is `Ninja Multi-Config`.

> Many entries are one-off debugging tasks from past investigation sessions. The stable core tasks are listed below.

### 14.1 Core Build Tasks

| Label | Description | Build Dir |
|-------|------------|-----------|
| `Debug` | MSVC Debug, default | `build/` |
| `Release` | MSVC Release, default | `build/` |
| `Debug (icx)` | Intel icx Debug | `build-icx/` |
| `Release (icx)` | Intel icx Release | `build-icx/` |
| `Release Build With PDB` | MSVC Release with full PDB | `build/` |

### 14.2 Utility Tasks

| Label | Description |
|-------|-------------|
| `Kill Previous Instance` | `taskkill /F /IM ConvoPeq.exe` |
| `Clean` | Remove `build/` directory |
| `CLI Smoke Test` | Run `cli-smoke-test.ps1 -KillExisting -RequireAudioCallbacks` |
| `Debug Build + Test` | Build Debug + run CTest |
| `Check MKLROOT` | Verify MKLROOT environment |

### 14.3 CMake Reconfigure Tasks

| Label | Description |
|-------|-------------|
| `CMake Reconfigure` | Reconfigure MSVC build |
| `CMake Reconfigure vs2026` | Reconfigure with vs2026 preset |
| `CMake Reconfigure icx` | Reconfigure icx build |
| `CMake Reconfigure icx (build)` | Reconfigure + build icx |
| `CMake Reconfigure icx (full)` | Full icx reconfigure |

### 14.4 CodeGraph Tasks (Static Index)

| Label | Description |
|-------|-------------|
| `CodeGraph Full Index` | Run full CodeGraph indexing |
| `CodeGraph Incremental Index` | Run incremental CodeGraph indexing |
| `CodeGraph Stats` | Show CodeGraph stats |
| `CodeGraph Apply Local Patch` | Apply local CodeGraph patch |

### 14.5 CodeQL Tasks (Security Analysis)

| Label | Description |
|-------|-------------|
| `CodeQL Create DB (ConvoPeq Standard)` | Create CodeQL database |
| `CodeQL Create DB (ConvoPeq Standard DryRun)` | Dry-run the DB creation script |
| `CodeQL One-Step (ConvoPeq Standard)` | Run full CodeQL analysis |
| `CodeQL One-Step (ConvoPeq Standard DryRun)` | Dry-run the analysis |

### 14.6 Analysis / Verification Tasks

| Label | Description |
|-------|-------------|
| `Strict Atomic Dot-Call Scan` | Scan src/ for atomic dot-calls |
| `work21 EpochDomain CI Gate` | CI gate for EpochDomain |
| `Verify All Tools` | Verify build tools availability |
| `Headroom Proxy: Status` / `Stop` | Headroom proxy management |

**Recommended workflow**:
- **Terminal → Run Task → Debug** (MSVC Debug)
- **Terminal → Run Task → Release** (MSVC Release)
- **Terminal → Run Task → Debug (icx)** / **Release (icx)**
- **Terminal → Run Task → Clean** (when switching toolchains or after cache conflicts)

---

## 15. VS Code Debug Configurations (launch.json)

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

## 16. Common Issues and Fixes

### A) `windows.h` or standard headers not found

**Cause**: MSVC/SDK environment not initialized in the same command chain.

**Fix**:
- Use `build.bat` from a Developer Command Prompt, or
- Ensure both `vcvarsall.bat` and `setvars.bat` are called before CMake in the same shell session
- In PowerShell: `cmd.exe /d /c "... && ..."`

> **Note**: `build.bat` does **not** auto-detect or call `vcvarsall.bat` for MSVC mode. The environment must already be initialized.

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
- Or set `CONVOPEQ_REQUIRE_MKL=OFF` to use the system aligned allocator fallback

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
- Build system automatically overrides `MSVC_RUNTIME_LIBRARY` to dynamic when `ENABLE_ASAN=ON`
- Configure with `cmake -S . -B build -DENABLE_ASAN=ON`

### G) icx first build fails with RC1109

**Cause**: Ninja + icx + cmcldeps creates stale RC resource file on first configure.

**Fix**:
- `build.bat` automatically detects RC1109 on first build, removes the stale `.res` file, and retries once
- If it persists: manually delete `build-icx\CMakeFiles\ConvoPeq_rc_lib.dir\<Config>\ConvoPeq_artefacts\JuceLibraryCode\ConvoPeq_resources.rc.res`

### H) PGO on icx fails

**Cause**: PGO is MSVC-only. icx does not support `/GENPROFILE` or `/USEPROFILE`.

**Fix**:
- Use MSVC for PGO: `build.bat Release pgo-gen` / `build.bat Release pgo-use`

### I) Generator mismatch after switching toolchains

**Cause**: juceaide sub-build cache retains the old generator.

**Fix**:
- `build.bat` automatically removes `build\JUCE\tools\CMakeCache.txt` on every reconfigure
- Or run `Clean` task and reconfigure from scratch

---

## 17. CMake Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `CONVOPEQ_ENABLE_CLANG_TIDY` | OFF | Run clang-tidy during build |
| `CONVOPEQ_REQUIRE_MKL` | ON | Require Intel MKL for MSVC builds (OFF → system aligned allocator) |
| `CONVOPEQ_ENABLE_ISR_TESTS` | ON | Build CTest regression suite (36 executables) |
| `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` | OFF | Enable runtime diagnostic logging (XRUN/MEM/VERIFY/WORLD) |
| `CONVOPEQ_PGO_INSTRUMENT` | OFF | PGO instrumentation build |
| `CONVOPEQ_PGO_USE` | OFF | PGO optimized build |
| `ENABLE_ASAN` | OFF | AddressSanitizer (mutually exclusive with PGO and TSAN) |
| `ENABLE_TSAN` | OFF | ThreadSanitizer (Clang only; mutually exclusive with ASAN) |

### Convolver Split Feature Flags (all ON)

```
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LIFECYCLE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_REBUILD=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOADER_THREAD=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_MIXED_PHASE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RESAMPLE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_LOAD_PIPELINE=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_RUNTIME=1
CONVOPEQ_ENABLE_CONVOLVER_SPLIT_STATE_UI=1
```

---

## 18. Dependency Boundaries

Do **not** modify external dependency trees directly:

- `JUCE/` — JUCE framework (in-tree)
- `r8brain-free-src/` — IR resampler library

---

## 19. Architecture Notes

- ConvoPeq is a **standalone app** (not a plugin target)
- The **default daily workflow** is `build.bat` or VS Code tasks
- Use `Clean` when switching toolchains or after generator/cache conflicts
- **MSVC and icx modes are fully isolated** (separate build directories) and can coexist
- **icx binaries require Intel CPU** (AVX2 check enforced at runtime; AMD is unsupported)
- **PGO is MSVC-only**; `.pgd` stored in `build\ConvoPeq_artefacts\Release\`
- **CI skips audio test** when `CONVO_CI_BUILD=1` is defined (no audio device in CI)
- **BuildInputSemanticContractTests** requires 8 MB stack on MSVC
- **Build identity gate** runs after every configure (fail-closed)
- **MSVC uses `/fp:precise`** (not `/fp:fast`) for DSP numeric accuracy
- **icx uses `/O2`** (not `/O3`) to avoid LLVM OOM on large JUCE TUs

---

## 20. Quick Reference

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

# Extra CMake define
build.bat Release nopause -DCONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS

# ASan (via CMake, not build.bat)
cmake -S . -B build -DENABLE_ASAN=ON
cmake --build build --config Debug

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
  src/audioengine/                          (126 files total)
    AudioEngine.*.cpp         (31 files — Timer, Commit, Rebuild, Processing, etc.)
    ISR*.cpp                  (17 files — Closure, Retire, Shutdown, Publication, etc.)
    AudioEngineProcessor.cpp
    AutoGainPlanner.cpp
    CrossfadeAuthority.cpp
    DSPLifetimeManager.cpp
    FrozenRuntimeWorld.cpp
    PublicationAdmission.cpp
    PublicationExecutor.cpp
    RetryScheduler.cpp
    RuntimeBuilder.cpp
    RuntimeHealthMonitor.cpp
    RuntimePolicyEngine.cpp
    RuntimePublicationOrchestrator.cpp
    RuntimePublicationValidator.cpp
    TelemetryRecorder.cpp
    WorldLifecycleAudit.cpp
  src/convolver/
    ConvolverProcessor.*.cpp  (8 TUs — Lifecycle, Rebuild, LoaderThread, etc.)
  src/eqprocessor/
    EQProcessor.*.cpp         (5 TUs — Core, Parameters, Coefficients, Processing, etc.)
  src/core/
    GlobalSnapshot.cpp, SnapshotCoordinator.cpp, SnapshotFactory.cpp,
    SnapshotAssembler.cpp, DeletionQueue.cpp, WorkerThread.cpp
  src/CustomInputOversampler.cpp
  src/TruePeakDetector.cpp
  src/LoudnessMeter.cpp
  src/MKLNonUniformConvolver.cpp
  src/NoiseShaperLearner.cpp
  src/FFTBackend.cpp
  src/FFTExecutionContext.cpp
  src/OutputFilter.cpp
  src/ProgressiveUpgradeThread.cpp
  src/IRConverter.cpp
  src/IRAnalyzer.cpp
  src/IRDSP.cpp
  src/CacheManager.cpp
  src/MixedPhasePersistentCache.cpp
  src/PsychoacousticDither.cpp
  src/AllpassDesigner.cpp
  src/CmaEsOptimizerDynamic.cpp
  + 36 test executables in src/tests/
  + AudioEngineHarness in src/tests/AudioEngineHarness/
```

## Appendix B. Build Directory Structure

```
build/                              # MSVC build root
├── CMakeCache.txt
├── CMakeFiles/
│   └── .build_identity             # Build identity stamp (gate-checked)
├── ConvoPeq_artefacts/
│   ├── Debug/ConvoPeq.exe
│   └── Release/ConvoPeq.exe
│       └── ConvoPeq.pgd            # (after PGO use phase)
└── ConvoPeq_artefacts/Debug/      # test executables also here

build-icx/                          # icx build root (fully isolated)
└── (same structure)

out/build/<presetName>/              # CMakePresets custom build dir
```

## Appendix C. MSVC Environment Initialization

`build.bat` does **not** auto-detect Visual Studio via vswhere. For MSVC mode, the calling shell must already have the MSVC environment initialized:

- **VS Code tasks**: each task chain calls `vcvarsall.bat` before `cmake`
- **Developer Command Prompt**: environment is pre-initialized
- **Manual**: call `vcvarsall.bat x64` yourself before running `build.bat`

For icx mode, `build.bat` calls `setvars.bat intel64` automatically.
