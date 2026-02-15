# ビルドガイド - Windows 11 x64 + Visual Studio Code + MSVC

## 対象環境

- **OS**: Windows 11 x64
- **IDE**: Visual Studio Code
  - **VS Code拡張機能**: C/C++ Extension Pack, CMake Tools
- **コンパイラ**: MSVC 19.44.35222.0 (Visual Studio 2022 17.11以降)
- **SDK**: Windows SDK 10.0.26100.0 (Target: Windows 10.0.26200)
- **CMake**: 3.22以降
- **JUCE**: 8.0.12（厳密）
- **C++標準**: C++20
- **Intel oneAPI**: Base Toolkit、HPC Toolkit (任意, FFT最適化用)
  - **VS Code必要コンポーネント**: Extension Pack for Intel Software Developer Tools、Extension Pack for Intel Software Developer Tools、Analysis Configurator for Intel Software Developer Tools、Analysis Configurator for Intel Software Developer Tools

**重要**: 本アプリケーションはWindows 11 x64専用のスタンドアローンアプリケーションです。macOSやLinuxではビルドできません。

---

## セットアップ手順

### 1. 必須ソフトウェアのインストール

#### 1.1 Visual Studio 2022（ビルドツールのみでOK）

##### オプション1: Visual Studio 2022 Community（推奨）

1. <https://visualstudio.microsoft.com/ja/downloads/> からダウンロード
2. インストーラー起動
3. 「C++によるデスクトップ開発」にチェック
4. インストール

##### オプション2: Build Tools for Visual Studio 2022

1. <https://visualstudio.microsoft.com/ja/downloads/#build-tools-for-visual-studio-2022>
2. 「C++ Build Tools」にチェック
3. インストール

#### 1.2 CMake

```powershell
# wingetでインストール（Windows 11推奨）
winget install Kitware.CMake

# または公式サイトからインストーラーをダウンロード
# https://cmake.org/download/
```

インストール後、PATH確認:

```powershell
cmake --version
# CMake version 3.22以降が表示されればOK
```

#### 1.3 Visual Studio Code

```powershell
# wingetでインストール
winget install Microsoft.VisualStudioCode

# または公式サイトから
# https://code.visualstudio.com/
```

#### 1.4 VS Code拡張機能

VS Codeを起動して以下拡張機能をインストール:

1. **C/C++ Extension Pack** (ms-vscode.cpptools-extension-pack)
   - C/C++ IntelliSense, デバッグ, コード参照

2. **CMake Tools** (ms-vscode.cmake-tools)
   - CMakeプロジェクトのビルド・実行

3. **CMake** (twxs.cmake)
   - CMakeLists.txtのシンタックスハイライト

または、VS Code内で `Ctrl+Shift+P` → `Extensions: Show Recommended Extensions` で一括インストール可能

#### 1.5 Intel oneAPI Base Toolkit (任意)

FFT処理の高速化のためにIntel MKLを使用する場合:

1. Intel oneAPI Base Toolkit、Intel oneAPI HPC Toolkit からダウンロード
2. インストールする
3. デフォルトのインストールパス (`C:\Program Files (x86)\Intel\oneAPI`) を推奨 (`build.bat`が自動検出します)
4. VS Codeを起動し、拡張機能「Extension Pack for Intel Software Developer Tools」、「Extension Pack for Intel Software Developer Tools」、「Analysis Configurator for Intel Software Developer Tools」、「Analysis Configurator for Intel Software Developer Tools」をインストール

### 2. JUCEの準備

#### 2.1 JUCEをダウンロード

```powershell
# プロジェクトディレクトリに移動 (例)
cd C:\path\to\ConvoPeq

# Git経由でJUCE 8.0.12をクローン
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git

# または手動でダウンロード
# https://github.com/juce-framework/JUCE/releases/tag/8.0.12
# → ダウンロードしてプロジェクトルートに展開
```

#### 2.2 ディレクトリ構成確認

```text
ConvoPeq/
├── .vscode/           # VS Code設定
├── JUCE/              # JUCEフレームワーク（JUCE framework V8.0.12をダウンロードし自分で作成）
├── src/               # ソースコード
├── resources/         # リソースファイル (アイコン)
├── sample_setting/    # インポート用サンプル設定ファイル
├── build/             # ビルド出力（自動作成）
├── CMakeLists.txt     # CMake設定
├── ProjectMetadata.cmake # プロジェクトメタデータ
├── build.bat          # ビルドスクリプト
├── README.md          # 説明書
├── ARCHITECTURE.md    # アーキテクチャ設計書
└── BUILD_GUIDE_WINDOWS.md # ビルドガイド
```

---

## ビルド方法

### 方法1: build.batスクリプト（推奨・最も簡単）

プロジェクトルートに用意された`build.bat`を使用します。Intel MKL環境変数の設定なども自動で行われます。

```powershell
# プロジェクトディレクトリで実行
build.bat Release

# Debugビルドの場合
build.bat Debug

# クリーンビルド（リビルド）
build.bat Release clean
```

**ビルド成果物の場所**:

```text
build\ConvoPeq_artefacts\Release\ConvoPeq.exe
```

**実行方法**:

```powershell
cd build\ConvoPeq_artefacts\Release
"ConvoPeq.exe"
```

### 方法2: VS Code CMake Tools（推奨開発環境）

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

##### 方法A: ステータスバー

- 下部ステータスバーの「Build」ボタンをクリック

##### 方法B: キーボードショートカット

- `F7` または `Shift+F7`

##### 方法C: コマンドパレット

- `Ctrl+Shift+P` → `CMake: Build`

#### Step 4: 実行

- 下部ステータスバーの「▶ Run」ボタンをクリック

### 方法3: VS Code タスクを使用

1. `Ctrl+Shift+P` → `Tasks: Run Build Task`
2. 「CMake: Build (Release)」を選択
3. または `Ctrl+Shift+B`（デフォルトビルドタスクとして設定済み）

### 方法4: PowerShell/CMD（従来型）

```powershell
# Developer Command Prompt for VS 2022 を開く

# CMake設定
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64

# Releaseビルド
cmake --build . --config Release

# Debugビルド
cmake --build . --config Debug

# 実行
ConvoPeq_artefacts\Release\ConvoPeq.exe
```

---

## デバッグ方法

VS Codeにはあらかじめデバッグ設定（`.vscode/launch.json`）が含まれており、すぐにデバッグを開始できます。

### 1. ブレークポイント設定

1. VS Codeでソースファイル（例: `src/MainWindow.cpp`）を開きます。
2. 行番号の左側をクリックして、赤い丸（ブレークポイント）を表示させます。

### 2. デバッグ開始

#### 方法A: キーボードショートカット (推奨)

- **F5** キーを押すと、自動的にDebugビルドが実行され、デバッガが起動します。

#### 方法B: デバッグサイドバー

1. 左サイドバーの「実行とデバッグ」アイコン（`Ctrl+Shift+D`）をクリックします。
2. 上部のドロップダウンリストから **`(Windows) Launch (Debug)`** を選択します。
3. 緑色の「▶ (デバッグの開始)」ボタンをクリックします。

> **Note**: `(Windows) Launch (Release)` を選択すると、Releaseビルドでデバッグ実行できます（最適化されているため、変数値が見えない場合があります）。

### 3. デバッグ操作

- **F5**: 続行 / 次のブレークポイントまで実行
- **F10**: ステップオーバー (関数に入らず次の行へ)
- **F11**: ステップイン (関数の中へ入る)
- **Shift+F11**: ステップアウト (現在の関数を抜ける)
- **Shift+F5**: デバッグ停止

---

## トラブルシューティング

### 🔨 ビルドエラー

#### エラー: CMake Error: Could not find JUCE

**原因**: プロジェクトルートに `JUCE` フォルダが存在しないか、空です。

**解決方法**:
`JUCE` フォルダが正しく配置されているか確認してください。

```powershell
# フォルダの確認
dir JUCE

# 空の場合は再度クローン
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git
```

### エラー: Could not find JUCE

**原因**: JUCEディレクトリが見つからない

**解決方法**:

```powershell
# JUCEディレクトリの存在確認
dir JUCE

# なければクローン
git clone --branch 8.0.12 --depth 1 https://github.com/juce-framework/JUCE.git
```

### エラー: LNK1181: cannot open input file 'ole32.lib'

**原因**: Windows SDK未インストール

**解決方法**:

1. Visual Studio Installer起動
2. 「変更」→「個別のコンポーネント」
3. 「Windows 11 SDK (10.0.22621.0)」にチェック
4. インストール

### 警告: C4819 ファイルは現在のコードページで表示できない

**解決方法**:
`CMakeLists.txt`に既に `/utf-8` オプションが設定されています。
それでも警告が出る場合:

```cmake
# CMakeLists.txtに追加
add_compile_options(/source-charset:utf-8 /execution-charset:utf-8)
```

### ビルドが遅い

**最適化方法**:

1. `/MP`オプション確認（CMakeLists.txtに既に設定済み）
2. SSDを使用
3. ウイルス対策ソフトでbuildフォルダを除外
4. Ninja generatorを使用:

   ```powershell
   cmake .. -G Ninja
   cmake --build .
   ```

---

## VS Code便利機能

### IntelliSense（コード補完）

- 自動で表示されます
- `Ctrl+Space`で手動起動
- `F12`で定義へジャンプ
- `Shift+F12`で参照を検索

### フォーマット

- `Shift+Alt+F`: ファイル全体をフォーマット
- `Ctrl+K Ctrl+F`: 選択範囲をフォーマット

### ビルドエラーへのジャンプ

- `F8`: 次のエラー/警告
- `Shift+F8`: 前のエラー/警告

### タスククイック実行

- `Ctrl+Shift+P` → `Tasks: Run Task`
- よく使うタスク:
  - CMake: Configure
  - CMake: Build (Release)
  - CMake: Clean
  - Run Application

---

## ビルド設定のカスタマイズ

`CMakeLists.txt` を編集することで、最適化レベルやターゲットアーキテクチャを変更できます。

### 1. CPUアーキテクチャの変更 (AVX2 / SSE2)

デフォルトでは `/arch:AVX2` が指定されており、Haswell (2013年) 以降のCPU向けに最適化されています。古いCPUで動作させる必要がある場合は、`CMakeLists.txt` を修正します。

**CMakeLists.txt**:

```cmake
# 変更前 (デフォルト: 高速)
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /GL /arch:AVX2 /fp:fast /Gw /Gy")

# 変更後 (互換性重視: SSE2)
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /GL /arch:SSE2 /fp:fast /Gw /Gy")
```

### 警告レベル変更

CMakeLists.txt:

```cmake
if(MSVC)
    add_compile_options(/W4)  # W3 → W4
endif()
```

### カスタムビルドタスク追加

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

## 推奨ワークフロー

本プロジェクトは VS Code での開発に最適化されています。`.vscode` フォルダ内の設定により、ショートカットキーで効率的に開発できます。

### 1. 日常の開発サイクル (Coding & Debugging)

**F5** キーを中心としたワークフローです。

1. **編集**: ソースコードを編集・保存します。
2. **デバッグ実行**: **`F5`** キーを押します。
   - 自動的に **Debug構成** でインクリメンタルビルドが行われます (`preLaunchTask` により自動実行)。
   - ビルド完了後、デバッガがアタッチされた状態でアプリが起動します。
   - ブレークポイントでの停止、変数の監視、ステップ実行が可能です。

### 2. パフォーマンス確認 (Release Build)

最適化された状態での動作確認や、CPU負荷のチェックを行います。

1. **ビルド**: **`Ctrl + Shift + B`** を押します。
   - デフォルトビルドタスクとして **Release構成** が設定されています。
   - 最適化 (`/O2`, `/AVX2` 等) が適用されたバイナリが生成されます。
2. **実行**:
   - 生成された `build\ConvoPeq_artefacts\Release\ConvoPeq.exe` を実行します。
   - または、コマンドパレットからタスク `Run Application` を実行します。

### 3. 配布用ビルド (Distribution)

配布用の実行ファイルを作成する際は、クリーンビルドを行って確実なバイナリを生成します。

**コマンドライン (推奨)**:

```powershell
# クリーンアップしてReleaseビルド
build.bat Release clean
```

---

## よくある質問

### Q: ビルドに何分かかりますか？

**A**: 環境により異なります

- **初回ビルド**: JUCEフレームワーク全体のコンパイルが必要なため、数分（2〜10分）かかります。
- **2回目以降**: 変更差分のみのビルドとなるため、数秒〜数十秒で完了します。
- **高速化のコツ**: ウイルス対策ソフトの除外設定に `build` フォルダを追加すると、リンク速度が向上します。

### Q: 音が途切れる / ノイズが乗るのですが？

**A**: 以下の点を確認してください。

1. **Releaseビルドを使用していますか？** Debugビルドは最適化が無効化されており、リアルタイムオーディオ処理には不向きです。
2. **バッファサイズは適切ですか？** Audio Settingsでバッファサイズを大きく（例: 512 → 1024）してください。
3. **高負荷な処理をしていませんか？** 長いIR（数秒以上）の使用や、192kHzなどの高サンプルレート設定はCPU負荷を高めます。

### Q: 特定のASIOデバイスが表示されません

**A**: 不安定なドライバとしてブラックリストに登録されている可能性があります。
実行ファイルと同じフォルダにある `asio_blacklist.txt` を確認してください。安定動作が確認できている場合は、リストから削除（またはコメントアウト）することで表示されるようになります。

### Q: VST3 / AU プラグインとしてビルドできますか？

**A**: 現在のバージョンはスタンドアローンアプリケーション専用です。
ソースコードはJUCEモジュールを使用しているためプラグイン化は可能ですが、`CMakeLists.txt` の修正（`juce_add_plugin`への変更）とラッパーコードの追加が必要です。

### Q: Intel MKL は必須ですか？

**A**: 必須ではありません。
Intel MKLがインストールされていない場合、自動的にJUCE内蔵のFFTエンジン（低速ですが互換性が高い）が使用されます。MKLを使用すると、特にConvolverの処理負荷が軽減されます。

### Q: 設定を完全にリセットしたいです

**A**: 以下のフォルダを削除してください。
`%APPDATA%\ConvoPeq`
（エクスプローラーのアドレスバーに入力すると移動できます）
ここにはデバイス設定 (`device_settings.xml`) が保存されています。

### Q: "LNK1104: cannot open file 'mkl_...'" というエラーが出ます

**A**: Intel MKLのライブラリパスが見つかりません。
Intel oneAPIの環境変数設定スクリプト (`vars.bat`) を実行したターミナルから `code .` コマンドでVS Codeを起動してください。または、`build.bat` を使用してコマンドラインからビルドしてください（`build.bat` は自動的に環境変数を設定します）。

---

## サポート情報収集

問題が解決しない場合、以下の情報を収集して報告してください。

### 1. 基本環境情報

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
