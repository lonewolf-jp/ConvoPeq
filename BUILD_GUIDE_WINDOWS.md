# ConvoPeq Build Guide (Windows 11 x64)

This guide reflects the **current repository setup** (`build.bat`, `tasks.json`, and CMake configuration).

---

## 1. Supported Environment

- **OS**: Windows 11 x64
- **Framework**: JUCE 8.0.12
- **Compiler**: MSVC (Visual Studio 2022/2026) または Intel icx (oneAPI 2026.0)
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
2. Detects compiler mode (MSVC/icx/icpx)
3. Initializes compiler environment (vcvarsall.bat for MSVC, skipped for icx)
4. Initializes oneAPI via `setvars.bat intel64` (required for both MSVC and icx)
5. Configures with Ninja Multi-Config
6. Builds selected config

Output binaries:

- MSVC Debug: `build\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- MSVC Release: `build\ConvoPeq_artefacts\Release\ConvoPeq.exe`
- icx Debug: `build-icx\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- icx Release: `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`

---

## 5. Intel icx コンパイラでのビルド

ConvoPeq は Intel icx (oneAPI DPC++/C++ Compiler) でのビルドに対応しています。
Windows では `icx.exe` 1つで C/C++ 両方をカバーするため、`CMAKE_C_COMPILER` と `CMAKE_CXX_COMPILER` の両方に `icx` を指定します（Intel推奨）。

### 5.1 build.bat 使用（推奨）

```cmd
REM Intel icx Releaseビルド（Haswell以降に極限最適化）
build.bat Release icx

REM Intel icx Debugビルド
build.bat Debug icx

REM clean + icx
build.bat Release clean icx
```

icx モードでは自動的に `build-icx` ディレクトリが使用され、MSVC のビルドキャッシュと完全分離されます。

出力バイナリ:

- Debug: `build-icx\ConvoPeq_artefacts\Debug\ConvoPeq.exe`
- Release: `build-icx\ConvoPeq_artefacts\Release\ConvoPeq.exe`

### 5.2 手動ビルド（icx 相当）

```cmd
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat" intel64

cmake -S . -B build-icx -G "Ninja Multi-Config" -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icx
cmake --build build-icx --config Release
```

> **注**: ConvoPeq はC++プロジェクトだが、**Windowsの `icx.exe` はC/C++両方に対応している**。
> Linuxのように `icx`（C用）と `icpx`（C++用）を区別する必要はなく、両方に `icx` を指定するのがIntel推奨。

### 5.3 icpx（実験的サポート）

Intel icpx (GNU-style driver) でのビルドも実験的にサポート:

```cmd
build.bat Release icpx
```

---

## 6. Manual Build (MSVC Equivalent)

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

## 7. VS Code Tasks (Current)

Current tasks are:

- `Kill Previous Instance`
- `Clean`
- `Debug` (MSVC)
- `Release` (MSVC)
- `Debug (icx)` — Intel icx Debug ビルド
- `Release (icx)` — Intel icx Release ビルド

Task characteristics:

- Shell is explicitly `cmd.exe`
- Generator is `Ninja Multi-Config`
- MSVC:  `${workspaceFolder}\build` ディレクトリ
- icx:  `${workspaceFolder}\build-icx` ディレクトリ（分離）
- Config is selected by `--config Debug|Release`

Recommended usage:

- **Terminal -> Run Task -> Debug**
- **Terminal -> Run Task -> Release**
- **Terminal -> Run Task -> Debug (icx)**
- **Terminal -> Run Task -> Release (icx)**
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

## 11. Notes

- The app target is standalone (not a plugin target).
- The default daily workflow is `build.bat` or VS Code tasks.
- Use `Clean` when switching toolchain assumptions or after generator/cache conflicts.
- icx モードと MSVC モードはビルドディレクトリが完全分離（`build-icx` vs `build`）されているため、同時に保持可能。
- icx でビルドしたバイナリは Intel CPU チェックを通過する必要がある（非Intel CPUでは実行不可）。
- PGO は MSVC モード限定。icx モードでは未対応。

---

## 10. Profile-Guided Optimization (PGO) Build

To build ConvoPeq with Profile-Guided Optimization (PGO) using MSVC, follow these steps:

1. **Generate Instrumented Build**

 ```cmd
 build.bat Release pgo-gen
 ```

1. **Navigate to Release Output Folder**

 ```cmd
 cd build\ConvoPeq_artefacts\Release
 ```

1. **Copy `pgort140.dll`**

- Copy `pgort140.dll` from your MSVC tools directory (e.g., `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<version>\bin\Hostx64\x64\pgort140.dll`) into this folder.

4. **Run the Instrumented Executable**

- Launch `ConvoPeq.exe` and exercise the application. (Note: CPU load will be about 200% of normal; audio dropouts are expected and not a problem.)

5. **Merge Profile Data**

 ```cmd
 "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\pgomgr.exe" /merge *.pgc ConvoPeq.pgd
 ```

1. **Return to Project Root**

 ```cmd
 cd ..\..\..
 ```

1. **Build with PGO Optimization**

 ```cmd
 build.bat Release pgo-use
 ```

1. **Result**

- The PGO-optimized binary will be generated in `build\ConvoPeq_artefacts\Release`.

---
