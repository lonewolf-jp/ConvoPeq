# ConvoPeq Build Guide (Windows 11 x64)

This guide reflects the **current repository setup** (`build.bat`, `tasks.json`, and CMake configuration).

---

## 1. Supported Environment

- **OS**: Windows 11 x64
- **Framework**: JUCE 8.0.12
- **Compiler**: MSVC (Visual Studio 2022)
- **Build System**: CMake 3.22+ + Ninja Multi-Config
- **Language Standard**: C++20
- **Math Backend**: Intel oneMKL (oneAPI Base Toolkit)

> ConvoPeq is a **Windows-only standalone application**.

---

## 2. Required Software

1. **Visual Studio 2022** with *Desktop development with C++*
2. **CMake** (3.22 or later)
3. **Ninja** (or Ninja available via your VS/CMake environment)
4. **Intel oneAPI Base Toolkit** (MKL)

---

## 3. Repository Layout Requirements

At minimum, the project root must contain:

- `build.bat`
- `CMakeLists.txt`
- `JUCE/`

`build.bat` validates `JUCE\CMakeLists.txt` before configuring.

---

## 4. Recommended Build Method (build.bat)

From the repository root:

```cmd
build.bat Release
build.bat Debug
build.bat Release clean
```

What the script does:

1. Validates local JUCE directory
2. Initializes MSVC environment via `vcvarsall.bat x64`
3. Initializes oneAPI via `setvars.bat intel64`
4. Configures with Ninja Multi-Config
5. Builds selected config

Output binaries:

- Debug: `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- Release: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## 5. Manual Build (Equivalent)

Use this when you want full manual control:

```cmd
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build --config Debug
cmake --build build --config Release
```

To run all initialization and build steps in a single line from PowerShell (ensuring all environment setup occurs in the same process):

```powershell
cmd.exe /d /c "call \"%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat\" x64 && call \"%ProgramFiles(x86)%\Intel\oneAPI\setvars.bat\" intel64 && cmake -S . -B build -G \"Ninja Multi-Config\" -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl && cmake --build build --config Debug"
```

---

## 6. VS Code Tasks (Current)

Current tasks are:

- `Kill Previous Instance`
- `Clean`
- `Debug`
- `Release`

Task characteristics:

- Shell is explicitly `cmd.exe`
- Generator is `Ninja Multi-Config`
- Shared build directory is `${workspaceFolder}\build`
- Config is selected by `--config Debug|Release`

Recommended usage:

- **Terminal -> Run Task -> Debug**
- **Terminal -> Run Task -> Release**
- **Terminal -> Run Task -> Clean** (when cache/build directory reset is needed)

---

## 7. Common Issues and Fixes

### A) `windows.h` or standard headers are not found

Cause: MSVC/SDK environment was not initialized in the same command chain.

Fix:

- Use `build.bat`, or
- Ensure both `vcvarsall.bat` and `setvars.bat` are called before CMake in the same shell command sequence.
- In PowerShell, use `cmd.exe /d /c "... && ..."` to execute all steps as a single command chain.

### B) `Release` task does not produce Release artifacts

Cause: Single-config generator/cache mismatch.

Fix:

- Use `Ninja Multi-Config`
- Build with `--config Release`
- If needed, run `Clean` and reconfigure.

### C) `'C:\Program' is not recognized`

Cause: Broken quoting or shell mismatch.

Fix:

- Keep task shell as `cmd.exe`
- Keep quoted paths exactly as in current `tasks.json`.
- For variable expansion, using `%ProgramFiles%` / `%ProgramFiles(x86)%` helps avoid quoting issues.

### D) oneMKL package not found

Cause: oneAPI environment not initialized.

Fix:

- Install Intel oneAPI Base Toolkit
- Confirm `C:\Program Files (x86)\Intel\oneAPI\setvars.bat` exists
- Re-run from a clean shell.

### E) JUCE check fails in `build.bat`

Cause: Missing or invalid local `JUCE` folder.

Fix:

- Ensure `JUCE\CMakeLists.txt` exists under repository root.

---

## 8. Dependency Boundaries

Do **not** modify external dependency trees directly:

- `JUCE/`
- `r8brain-free-src/`

---

## 9. Notes

- The app target is standalone (not a plugin target).
- The default daily workflow is `build.bat` or VS Code tasks.
- Use `Clean` when switching toolchain assumptions or after generator/cache conflicts.
