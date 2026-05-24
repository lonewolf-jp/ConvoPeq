# ConvoPeq CodeQL 標準DB作成テンプレート（C++ / CMake）

ConvoPeq で毎回同じ条件の解析を行うための標準手順です。

- 対象: C++ (`cpp` extractor)
- ビルド: CMake + Ninja Multi-Config
- 解析DB保存先: `storage/codeql/databases`
- 標準ビルド構成: `Debug`

---

## 1. 追加した標準スクリプト

- `tools/codeql/create-convopeq-codeql-db.ps1`

このスクリプトは以下を一貫した条件で実行します。

1. MSVC / oneAPI 環境初期化
2. CMake configure + build
3. `codeql database create` 実行

---

## 2. 実行例

### 2-1. まずはドライラン（コマンド確認のみ）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\create-convopeq-codeql-db.ps1 -DryRun
```

### 2-2. 標準DBを作成（Debug）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\create-convopeq-codeql-db.ps1
```

### 2-3. Releaseで作成

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\create-convopeq-codeql-db.ps1 -Config Release
```

### 2-4. 完全作り直し

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\create-convopeq-codeql-db.ps1 -Clean
```

### 2-5. ワンステップ実行（DB作成 → 解析）

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\run-convopeq-codeql-onestep.ps1
```

### 2-6. ワンステップのドライラン

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\codeql\run-convopeq-codeql-onestep.ps1 -DryRun
```

---

## 3. 主なパラメータ

- `-DatabaseName`（既定: `convopeq-cpp-standard`）
- `-Config`（`Debug` / `Release`、既定: `Debug`）
- `-BuildParallel`（1〜64、既定: `1`）
- `-BuildTag`（既定: `default`。CodeQL用ビルドディレクトリ識別子）
- `-Clean`（build / DB を削除して再生成）
- `-DryRun`（実行せず予定コマンドを表示）

ワンステップスクリプト（`run-convopeq-codeql-onestep.ps1`）の追加パラメータ:

- `-QuerySuite`（既定: `codeql/cpp-queries:codeql-suites/cpp-security-and-quality.qls`）
- `-BuildParallel`（既定: `1`。C1060などメモリ圧迫時は `1` を推奨）
- `-BuildTag`（one-stepでは `run-<timestamp>` を自動使用）

---

## 4. 固定される条件（再現性のため）

- CodeQL言語: `cpp`
- source-root: リポジトリルート
- ビルドコマンド:
  - `vcvarsall.bat x64`
  - `setvars.bat intel64`
  - `cmake --build <repo>/build --config <Config> --parallel <BuildParallel>`
- DB出力先:
  - `storage/codeql/databases/<DatabaseName>`

---

## 5. 前提

- `CODEQL_PATH` が設定済み、または `C:\Users\user\tools\codeql\codeql.exe` が存在
- Visual Studio / oneAPI が既存タスクと同じ場所にインストール済み

---

## 6. 運用メモ

- 解析対象の除外方針（`JUCE`, `r8brain-free-src` など）は `mcp.json` の `CODEQL_MCP_SCAN_EXCLUDE_DIRS` で管理。
- DB作成条件を変えたい場合は、このスクリプトにだけ変更を集中させると差分管理しやすいです。
- ワンステップ実行の出力SARIFは `storage/codeql/query-runs/<DatabaseName>/<timestamp>/results.sarif` に保存されます。
- CodeQL DB作成は既存の `build` ディレクトリを再利用します（`cmake configure` は行いません）。
