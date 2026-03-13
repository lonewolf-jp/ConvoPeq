# ConvoPeq v0.4.4 Build Guide – Windows 11 x64

## Target Environment

- **OS**: Windows 11 x64
- **IDE**: Visual Studio Code (recommended)
- **Compiler**: MSVC 19.44.35222.0 (Visual Studio 2022 17.11+)
- **CMake**: 3.22+
- **JUCE**: 8.0.12 (required)
- **C++ Standard**: C++20
- **Intel oneAPI**: Base Toolkit (MKL required)

> This project is a **Windows-only standalone application**.

---

## Required Software

1. **Visual Studio 2022** (Desktop development with C++)
2. **CMake**
3. **Ninja** (or bundled Ninja via CMake/VS environment)
4. **Intel oneAPI Base Toolkit**
5. **JUCE 8.0.12** placed at:
   - `ConvoPeq/JUCE/...`

---

## Project Layout Requirement

The project root must contain:

- `build.bat`
- `CMakeLists.txt`
- `JUCE/` (JUCE 8.0.12 source tree)

---

## Quick Build (Recommended)

From project root:

```cmd
build.bat Release
build.bat Debug
build.bat Release clean
```

### Build Output Paths

- **Debug**:
  - `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- **Release**:
  - `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## Manual Build (Equivalent)

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Release
cmake --build build --config Debug
```

---

## VS Code Task Usage

Use:

- **Terminal → Run Task → Release**
- **Terminal → Run Task → Debug**
- **Terminal → Run Task → Clean**

`tasks.json` should use:

- `-G "Ninja Multi-Config"`
- common build directory: `-B "${workspaceFolder}\build"`
- build switch by `--config Release|Debug`

---

## Troubleshooting

### 1) `windows.h` / standard headers not found

Environment not initialized for the same shell session.
Run `vcvarsall.bat` and `setvars.bat` in the same command chain as `cmake`.

### 2) Release task builds Debug

Using single-config Ninja or stale CMake cache.
Use **Ninja Multi-Config** and `--config Release`.

### 3) `'C:\Program' is not recognized`

Quote escaping issue (PowerShell -> cmd nesting).
Force task shell to `cmd.exe` and keep command quoting simple.

### 4) `・ｿ@echo off` appears in `build.bat`

File saved with BOM/encoding issue.
Save `build.bat` as **UTF-8 (without BOM)**.

### 5) JUCE exists but check fails

Ensure script runs from project root or use `pushd "%~dp0"` in `build.bat`.
Validate path: `JUCE\CMakeLists.txt`.

---

## Notes

- Do **not** modify dependency sources directly:
  - `JUCE/`
  - `r8brain-free-src/`
- App target is standalone (not plugin build target).
- For daily development, VS Code task-based workflow is recommended.
