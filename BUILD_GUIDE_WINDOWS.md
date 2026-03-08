---

# ConvoPeq v0.3.5 Build Guide – Windows 11 x64  
*(Fully translated version — Japanese → English)*

## Target Environment

- **OS**: Windows 11 x64  
- **IDE**: Visual Studio Code  
  - **VS Code Extensions**: C/C++ Extension Pack, CMake Tools  
- **Compiler**: MSVC 19.44.35222.0 (Visual Studio 2022 17.11 or later)  
- **SDK**: Windows SDK 10.0.26100.0 (Target: Windows 10.0.26200)  
- **CMake**: 3.22 or later  
- **JUCE**: 8.0.12 (Strict)  
- **C++ Standard**: C++20  
- **Intel oneAPI**: Base Toolkit (Required, for MKL library)

**Important**: This application is a standalone application dedicated to Windows 11 x64. It cannot be built on macOS or Linux.

---

## Setup Instructions

### 1. Install Required Software

#### 1.3 Visual Studio Code

```powershell
# Install via winget
winget install Microsoft.VisualStudioCode

# Or from the official website
# https://code.visualstudio.com/
```

#### 1.4 VS Code Extensions

Install the following extensions in VS Code:

- C/C++ Extension Pack  
- CMake Tools  
- CMake syntax highlighting  

#### 1.5 Intel oneAPI Base Toolkit (Required)

Building this application **requires Intel oneMKL (Math Kernel Library)**, which is included in the Base Toolkit.

- The default installation path (`C:\Program Files (x86)\Intel\oneAPI`) is recommended because `build.bat` auto-detects it.

---

## 2. Preparing Dependencies

### 2.1 Download Libraries

JUCE is required. r8brain-free-src is already included.

### 2.2 Directory Structure

- `.vscode/` — VS Code settings  
- `JUCE/` — JUCE framework (download JUCE 8.0.12 and place here)  
- `r8brain-free-src/` — r8brain library  
- `CMakeLists.txt` — CMake configuration  
- `ProjectMetadata.cmake` — project metadata  
- `build.bat` — build script  

---

## Build Instructions

### Method 1: build.bat Script (recommended / easiest)

The provided `build.bat` automatically configures Intel MKL environment variables.

---

## Method 2: VS Code CMake Tools

### Step 1: Open the Project

Open the project folder in VS Code.

### Step 2: Configure CMake

Select a compiler kit such as “Visual Studio Community 2022 Release – amd64”.

### Step 3: Build

You can build from the status bar, keyboard shortcuts, or command palette.

### Step 4: Run

Click the “Run” button in the status bar.

---

## Method 3: VS Code Tasks

You can run the build task via `Ctrl+Shift+P` → “Tasks: Run Build Task”.

---

## Method 4: PowerShell / CMD

Open the Developer Command Prompt for VS 2022 and run CMake manually.

---

## Debugging

VS Code includes preconfigured debug settings.

- Set breakpoints by clicking next to line numbers.  
- Press **F5** to start debugging.  
- Use the debug sidebar for additional options.

---

## Troubleshooting

### Build Errors

#### Error: Could not find JUCE

Cause: The `JUCE` folder is missing or empty.  
Solution: Ensure the folder exists and re-clone if necessary.

#### Error: LNK1181: cannot open input file 'ole32.lib'

Cause: Windows SDK is not installed.  
Solution: Install Windows 11 SDK via Visual Studio Installer.

### Warning: C4819 (file cannot be displayed in current code page)

If the warning persists, add:

```cmake
add_compile_options(/source-charset:utf-8 /execution-charset:utf-8)
```

### Build is slow

Tips:  
- `/MP` is already enabled  
- Use SSD  
- Exclude build folder from antivirus  
- Use Ninja generator  

---

## VS Code Useful Features

- IntelliSense  
- Formatting shortcuts  
- Jump to errors  
- Quick task execution  

---

## Build Configuration Customization

This application **requires AVX2 and Intel MKL**.  
Removing `/arch:AVX2` may cause build or runtime errors.

---

## Recommended Workflow

The project is optimized for VS Code.

### Daily Development

- Edit code  
- Press **F5** to debug  
- Incremental build runs automatically  

### Release Build for Performance

- Press **Ctrl+Shift+B**  
- Run the optimized executable  

### Distribution Build

- Use `build.bat Release clean`

---

## FAQ

### Q: How long does the build take?

- First build: several minutes  
- Subsequent builds: seconds  
- Excluding the build folder from antivirus improves speed  

### Q: Audio drops or noise occurs

- Use Release build  
- Increase buffer size  
- Avoid heavy CPU operations  

### Q: ASIO device does not appear

Check `asio_blacklist.txt` in the executable folder.

### Q: Can it be built as VST3/AU?

Possible but requires modifying `CMakeLists.txt` and adding wrapper code.

### Q: Is Intel MKL required?

Yes, MKL is required for FFT and vector operations.

### Q: Reset all settings

Delete `%APPDATA%\ConvoPeq`.

---

## Collect Support Information

Run the provided PowerShell commands and share the output when reporting issues.

---