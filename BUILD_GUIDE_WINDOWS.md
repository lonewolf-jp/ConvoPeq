# ConvoPeq v0.3.5 Build Guide - Windows 11 x64

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

#### 1.1 Visual Studio 2022 (Build Tools only is OK)

##### Option 1: Visual Studio 2022 Community (Recommended)

1. Download from <https://visualstudio.microsoft.com/downloads/>
2. Launch installer
3. Check "Desktop development with C++"
4. Install

##### Option 2: Build Tools for Visual Studio 2022

1. <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>
2. Check "C++ Build Tools"
3. Install

#### 1.2 CMake

```powershell
# Install via winget (Recommended for Windows 11)
winget install Kitware.CMake

# Or download installer from official site
# https://cmake.org/download/
```

After installation, verify that CMake is available in PATH:

```powershell
cmake --version
# CMake version 3.22or later should be displayed.
```

#### 1.3 Visual Studio Code

```powershell
# wingetでインストール
winget install Microsoft.VisualStudioCode

# または公式サイトから
# https://code.visualstudio.com/
```

#### 1.4 VS Code拡張機能

Launch VS Code and install the following extensions:

1. **C/C++ Extension Pack** (ms-vscode.cpptools-extension-pack)
   - C/C++ IntelliSense, debugging, and code navigation

2. **CMake Tools** (ms-vscode.cmake-tools)
   - Build and run CMake projects

3. **CMake** (twxs.cmake)
   - Syntax highlighting for CMakeLists.txt

Alternatively, press `Ctrl+Shift+P` in VS Code and run `Extensions: Show Recommended Extensions` to install them all at once.

#### 1.5 Intel oneAPI Base Toolkit (必須)

本アプリケーションのビルドには **Intel oneMKL (Math Kernel Library)** が必須です。これは Base Toolkit に含まれています。

1. [Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html) Download it
2. Install it
3. デフォルトのインストールパス (`C:\Program Files (x86)\Intel\oneAPI`) is recommended (`build.bat` が自動検出します)
   - Note: The HPC Toolkit is not required because the MSVC compiler is used.

### 2. 依存ライブラリの準備

#### 2.1 ライブラリのダウンロード

```powershell
# Move to the project directory (example)
cd C:\path\to\ConvoPeq

# 1. JUCE 8.0.12 (必須)
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git

# 2. r8brain-free-src (Already bundled)
# It is already included in the project root, so no download is necessary.
```

#### 2.2 Verify directory structure

```text
ConvoPeq/
├── .vscode/           # VS Code設定
├── JUCE/              # JUCEフレームワーク（JUCE framework V8.0.12Download itし自分で作成）
├── r8brain-free-src/  # r8brainライブラリ
├── src/               # source code
├── resources/         # resource files (icons)
├── build/             # build output (created automatically)
├── CMakeLists.txt     # CMake設定
├── ProjectMetadata.cmake # プロジェクトメタデータ
├── build.bat          # ビルドスクリプト
├── README.md          # documentation
├── ARCHITECTURE.md    # architecture design document
└── BUILD_GUIDE_WINDOWS.md # build guide
```

---

## ビルド方法

### Method 1: build.batスクリプト（recommended / easiest）

プロジェクトルートに用意された`build.bat`Useします。Intel MKL環境変数の設定なども自動で行われます。

```powershell
# Run in the project directory
build.bat Release

# For a Debug build
build.bat Debug

# Clean build (rebuild)
build.bat Release clean
```

**Build artifact location**:

```text
build\ConvoPeq_artefacts\Release\ConvoPeq.exe
```

**How to run**:

```powershell
cd build\ConvoPeq_artefacts\Release
"ConvoPeq.exe"
```

### Method 2: VS Code CMake Tools（recommended development environment）

#### Step 1: プロジェクトを開く

```powershell
cd C:\path\to\ConvoPeq
code .
```

#### Step 2: CMake設定

1. VS Code下部のステータスバーで「CMake」をクリック
2. または `Ctrl+Shift+P` → `CMake: Configure`
3. コンパイラキットを選択: `Visual Studio Community 2022 Release - amd64` (または `amd64` を含むもの)

#### Step 3: ビルド

##### Method A: ステータスバー

- 下部ステータスバーの「Build」ボタンをクリック

##### Method B: キーボードショートカット

- `F7` または `Shift+F7`

##### Method C: コマンドパレット

- `Ctrl+Shift+P` → `CMake: Build`

#### Step 4: 実行

- 下部ステータスバーの「▶ Run」ボタンをクリック

### Method 3: VS Code タスクUse

1. `Ctrl+Shift+P` → `Tasks: Run Build Task`
2. 「CMake: Build (Release)」を選択
3. または `Ctrl+Shift+B`（デフォルトビルドタスクとして設定済み）

### Method 4: PowerShell/CMD（traditional method）

```powershell
# Developer Command Prompt for VS 2022 を開く

# CMake設定
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -T host=x64

# Releaseビルド
cmake --build . --config Release

# Debugビルド
cmake --build . --config Debug

# 実行
ConvoPeq_artefacts\Release\ConvoPeq.exe
```

---

## Debugging

VS Codeにはあらかじめデバッグ設定（`.vscode/launch.json`）が含まれており、すぐにデバッグを開始できます。

### 1. Setting breakpoints

1. VS Codeでソースファイル（例: `src/MainApplication.cpp`）を開きます。
2. 行番号の左側をクリックして、赤い丸（ブレークポイント）を表示させます。

### 2. Start debugging

#### Method A: キーボードショートカット (recommended)

- **F5** キーを押すと、自動的にDebugビルドが実行され、デバッガが起動します。

#### Method B: デバッグサイドバー

1. 左サイドバーの「実行とデバッグ」アイコン（`Ctrl+Shift+D`）をクリックします。
2. 上部のドロップダウンリストから **`(Windows) Launch (Debug)`** を選択します。
3. 緑色の「▶ (デバッグの開始)」ボタンをクリックします。

> **Note**: `(Windows) Launch (Release)` を選択すると、ReleaseビルドでRun debuggingできます（最適化されているため、変数値が見えない場合があります）。

### 3. Debug controls

- **F5**: Continue / run until next breakpoint
- **F10**: Step over (next line without entering function)
- **F11**: Step into (enter function)
- **Shift+F11**: Step out (exit current function)
- **Shift+F5**: Stop debugging

---

## Troubleshooting

### 🔨 ビルドエラー

#### エラー: CMake Error: Could not find JUCE

**Cause**: プロジェクトルートに `JUCE` フォルダが存在しないか、空です。

**Solution**:
`JUCE` フォルダが正しく配置されているかPlease check.。

```powershell
# Check the folder
dir JUCE

# If empty, clone again
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git
```

### エラー: Could not find JUCE

**Cause**: JUCEディレクトリが見つからない

**Solution**:

```powershell
# JUCEディレクトリの存在確認
dir JUCE

# Clone if missing
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git
```

### エラー: LNK1181: cannot open input file 'ole32.lib'

**Cause**: Windows SDKnot installed

**Solution**:

1. Visual Studio Installer起動
2. 「変更」→「個別のコンポーネント」
3. 「Windows 11 SDK (10.0.22621.0)」にチェック
4. インストール

### Warning: C4819 ファイルは現在のコードページで表示できない

**Solution**:
`CMakeLists.txt`に既に `/utf-8` オプションが設定されています。
それでもWarningが出る場合:

```cmake
# CMakeLists.txtにAdd
add_compile_options(/source-charset:utf-8 /execution-charset:utf-8)
```

### Build is slow

**Optimization tips**:

1. `/MP`Check option（CMakeLists.txtに既に設定済み）
2. SSDUse
3. ウイルス対策ソフトでbuildフォルダを除外
4. Ninja generatorUse:

   ```powershell
   cmake .. -G Ninja
   cmake --build .
   ```

---

## VS CodeUseful features

### IntelliSense（code completion）

- 自動で表示されます
- `Ctrl+Space`で手動起動
- `F12`でGo to definition
- `Shift+F12`でFind references

### フォーマット

- `Shift+Alt+F`: Format entire file
- `Ctrl+K Ctrl+F`: Format selection

### Jump to build errors

- `F8`: 次のエラー/Warning
- `Shift+F8`: 前のエラー/Warning

### Quick task execution

- `Ctrl+Shift+P` → `Tasks: Run Task`
- Common tasks:
  - CMake: Configure
  - CMake: Build (Release)
  - CMake: Clean
  - Run Application

---

## Build configuration customization

`CMakeLists.txt` をEditすることで、最適化レベルやターゲットアーキテクチャを変更できます。
ただし、本アプリケーションは **AVX2 命令セット** および **Intel MKL** を必須としています。
`/arch:AVX2` オプションを削除したり、SSE2 にダウングレードすると、ビルドエラーや実行時エラー（不正命令例外）のCauseとなります。

### Warningレベル変更

CMakeLists.txt:

```cmake
if(MSVC)
    add_compile_options(/W4)  # W3 → W4
endif()
```

### カスタムビルドタスクAdd

.vscode/tasks.json:

```json
{
    "label": "Build and Run",
    "type": "shell",
    "command": "cmake --build ${workspaceFolder}/build --config Release && ${workspaceFolder}/build/Release/ConvPeq.exe",
    "problemMatcher": ["$msCompile"]
}
```

---

## recommendedワークフロー

本プロジェクトは VS Code での開発に最適化されています。`.vscode` フォルダ内の設定により、ショートカットキーで効率的に開発できます。

### 1. Daily development cycle (Coding & Debugging)

**F5** キーを中心としたワークフローです。

1. **Edit**: source codeをEdit・保存します。
2. **Run debugging**: **`F5`** キーを押します。
   - 自動的に **Debug構成** でインクリメンタルビルドが行われます (`preLaunchTask` により自動実行)。
   - ビルド完了後、デバッガがアタッチされた状態でアプリが起動します。
   - ブレークポイントでの停止、変数の監視、ステップ実行が可能です。

### 2. Performance verification (Release Build)

最適化された状態での動作確認や、CPU負荷のチェックを行います。

1. **ビルド**: **`Ctrl + Shift + B`** を押します。
   - デフォルトビルドタスクとして **Release構成** が設定されています。
   - 最適化 (`/O2`, `/AVX2` 等) が適用されたバイナリが生成されます。
2. **実行**:
   - 生成された `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` を実行します。
   - または、コマンドパレットからタスク `Run Application` を実行します。

### 3. Distribution build (Distribution)

配布用の実行ファイルを作成する際は、クリーンビルドを行って確実なバイナリを生成します。

**Command line (recommended)**:

```powershell
# クリーンアップしてReleaseビルド
build.bat Release clean
```

---

## FAQ

### Q: ビルドに何分かかりますか？

**A**: Depends on the environment

- **First build**: JUCEフレームワーク全体のコンパイルが必要なため、several minutes（2〜10分）かかります。
- **2回目以降**: only changed filesのビルドとなるため、数秒〜数十秒で完了します。
- **Tips for speeding up**: ウイルス対策ソフトの除外設定に `build` フォルダをAddすると、リンク速度が向上します。

### Q: Audio drops or noise occurs

**A**: Check the following points:

1. **ReleaseビルドUseしていますか？** Debugビルドは最適化が無効化されており、リアルタイムオーディオ処理には不向きです。
2. **バッファサイズは適切ですか？** Audio Settingsでバッファサイズを大きく（例: 512 → 1024）してください。
3. **high CPU load operationsをしていませんか？** 長いIR（数秒以上）の使用や、192kHzなどの高サンプルレート設定はCPU負荷を高めます。

### Q: Specific ASIO device does not appear

**A**: It may be blacklisted as an unstable driver.
same folder as the executableにある `asio_blacklist.txt` をPlease check.。安定動作が確認できている場合は、リストから削除（またはコメントアウト）することで表示されるようになります。

### Q: VST3 / AU プラグインとしてビルドできますか？

**A**: The current version is standalone‑application only.
source codeはJUCEモジュールUseしているためプラグイン化は可能ですが、`CMakeLists.txt` の修正（`juce_add_plugin`への変更）とラッパーコードのAddが必要です。

### Q: Intel MKL は必須ですか？

**A**: はい、必須です。
本アプリケーションは Intel MKL の高度な FFT およびベクトル演算機能に依存しています。インストールされていない場合、ビルドは失敗します。

### Q: Reset all settings

**A**: Delete the following folder.
`%APPDATA%\ConvoPeq`
（エクスプローラーのアドレスバーに入力すると移動できます）
Device settings are stored here. (`device_settings.xml`) が保存されています。

### Q: "LNK1104: cannot open file 'mkl_...'" というエラーが出ます

**A**: Intel MKLのライブラリパスが見つかりません。
Intel oneAPIの環境変数設定スクリプト (`vars.bat`) を実行したターミナルから `code .` コマンドでVS Codeを起動してください。または、`build.bat` UseしてCommand lineからビルドしてください（`build.bat` は自動的に環境変数を設定します）。

---

## Collect support information

If the issue is not resolved、Collect and report the following information.

### 1. Basic environment information

PowerShellで以下のコマンドを実行し、出力を共有してください。

```powershell
Write-Output "=== OS Information ==="
Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, OSArchitecture

Write-Output "`n=== CPU Information ==="
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors

Write-Output "`n=== Tool Versions ==="
cmake --version
code --version
git -C JUCE describe --tags
```

この情報と共にエラーメッセージを報告してください。ただし、作者はバイブコーディングしかできないので対応できない可能性が高いです。
