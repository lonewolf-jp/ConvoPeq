# CR-α-3 — Debug / Release Build Verification（Work Report）

```text
CR-α-3 — Debug / Release Build Verification

Date: 2026-09-01
baseline: ConvoPeq.md Generated 2026-09-01 20:32:43（CR-α-3 開始時 --check 実測 FRESH / NEWER_SRC_COUNT=0 / CHECK_EXIT=0）
Production source: 0（本工程は検証のみ・変更なし）
Test source: 0（本工程は検証のみ・変更なし）
CMake: 0
Build: RUN（configure + Debug build 実施 / Release は Debug FAIL により未実施）
CTest: 0（指示どおり未実施）
stress: 0
```

## 総合判定

> ## **CR-α-3 = FAIL（Debug build compile FAIL）— この段階で STOP・source 修正は未実施**
>
> Debug build が `BuildErrorClassificationTests.cpp` のコンパイルエラー 18 件
> （C2653 ×9 + C2065 ×9、行 270/273/276/279/281/289/292/296/299）で失敗。
> 根本原因 = `runTestF()` の using 宣言ブロック（:236-239）に **`using convo::RetryDisposition;` が欠落**。
> Release build は指示どおり「Debug PASS 後のみ」のため**未実施**。CTest（CR-α-4）へは進行不可。

## 1. 実施前 snapshot / cleanliness（実測）

### git diff --stat（要旨）

```text
 .gitignore                                      |   1 +
 ConvoPeq.md                                     | 481 +++++++++++++++++++++++-
 output_sourcecode_markdown.py                   |  84 +++++
 src/audioengine/AudioEngine.RebuildDispatch.cpp |  84 ++++-
 src/audioengine/BuildErrorPolicy.h              |  81 ++++
 src/audioengine/RuntimeWorldAuthority.h         |  63 ++++
 src/tests/BuildErrorClassificationTests.cpp     |  89 ++++-
 src/tests/ISRSemanticValidationTests.cpp        | 162 +++++++++
 8 files changed, 1006 insertions(+), 39 deletions(-)
```

### git diff --name-only と帰属分類

| ファイル | 帰属 | 判定 |
|---|---|---|
| src/audioengine/BuildErrorPolicy.h | **CR-α-1**（期待対象） | ✓ |
| src/audioengine/AudioEngine.RebuildDispatch.cpp | **CR-α-1**（期待対象） | ✓ |
| src/tests/BuildErrorClassificationTests.cpp | **CR-α-1**（期待対象） | ✓ |
| src/audioengine/RuntimeWorldAuthority.h | ND-01〜04 CW-8（残留・ND-04 で Debug/Release full build + CTest 40/40 検証済み） | 期待外だが既検証 |
| src/tests/ISRSemanticValidationTests.cpp | ND-01〜04 CW-8（残留・同上） | 期待外だが既検証 |
| .gitignore / ConvoPeq.md / output_sourcecode_markdown.py | 運用ファイル（snapshot 再生成・ignore 設定） | 対象外 |

### forbidden-file verification

```text
RetrySchedulerTypes.h : diff = 0（diff --name-only 不在 → 実測 0）
AudioEngine.h         : diff = 0（diff --name-only 不在 → 実測 0）
```

## 2. Build 環境・手順（実証済みレシピ踏襲）

- `tools/cra3_build.bat`（新規 helper — source/test 以外の artifacts）:
  vcvarsall x64（VS2026 18.9.2）+ oneAPI setvars intel64 + `set CL=/I "C:\Program Files (x86)\Intel\oneAPI\2026.1\include"`
  （mkl.h C1083 回避の Gate B 実証済み quoted 形式）→ `cmake -S . -B build` → `cmake --build build --config Debug`
- Generator: Ninja Multi-Config（build/ 実測 `CMAKE_GENERATOR:INTERNAL=Ninja Multi-Config`）
- 実行: `powershell.exe -NoProfile -Command "& 'tools\cra3_build.bat'"`（Gate B で確立した cmd.exe /c マングリング回避）
- Logs: `evidence/cra3_configure.log` / `evidence/cra3_build_debug.log`

## 3. Debug build 結果

| 工程 | 結果 |
|---|---|
| configure/generate | **PASS**（CRA3_CONFIGURE_EXIT=0 — Intel IPP found / PGO Normal Release with LTCG / Build files written） |
| compile | **FAIL**（CRA3_DBG_EXIT=2 — test TU 1 件のみ FAILED、下記詳細） |
| link | 未到達（compile 失敗により中断） |
| ConvoPeq target | 未到達（Link 未実施） |
| BuildErrorClassificationTests target | **FAIL**（compile） |
| warnings | **0 件**（`warning C` 実測 0 — ただし ninja が [27/468] で中断のため全 TU 分ではない。中断までに新規 warning なし） |

### Compiler error 記録（指示 4 の必須項目）

```text
configuration      : Ninja Multi-Config / Debug / MSVC 14.51.36231 / -std:c++20（compile_commands 実測）
対象ファイル        : src/tests/BuildErrorClassificationTests.cpp
error code         : C2653 'RetryDisposition': 識別子がクラス名でも名前空間名でもありません ×9
                     C2065 'RetryImmediate'/'NoRetry'/'RetryBackoff': 定義されていない識別子です ×9
行番号             : 270, 273, 276, 279, 281, 289, 292, 296, 299（各 2 件 = 計 18 件）
linker error       : なし（link 未到達）
```

### 根本原因（確定）

`runTestF()` 冒頭の using 宣言ブロック（BuildErrorClassificationTests.cpp:236-239）:

```cpp
using convo::BuildError;
using convo::RetryBackoffPolicy;
using convo::WarmupRetryAction;
using Action = WarmupRetryAction::Value;
```

に **`using convo::RetryDisposition;` が欠落**している。エラー 9 行（270-299）はすべて
T-CRα-3/T-CRα-4 の `warmupRetryDecision(...)` 呼び出しで引数に**非修飾 `RetryDisposition::...`**
（`RetryImmediate` ×4 / `NoRetry` ×1 / `RetryBackoff` ×4）を使用しており、匿名 namespace 内の
非修飾名前検索が `convo::RetryDisposition`（BuildErrorPolicy.h:30, enum class）に解決できず失敗。
`RetryBackoffPolicy`（:243/253/268/287）と `Action`（:290 等）は using 済みのため正常にコンパイル
している（= 他の CR-α-1 追加コードの name resolution は健全）。

一方、ファイル冒頭（:45-52 / :62-71 / :200 等）は修飾名 `convo::RetryDisposition` を使用しており
コンパイル成功 — 修飾/非修飾の不整合が 1 点のみ。

### CR-α-2 で見逃された理由（再発防止）

CR-α-2 は read-only でビルドを実施していない。V16 は CHECK 実装数の source 照合であり
name resolution（using 漏れ）は検出不能だった。G-4.3-T で教訓化済みの
「read-only audit は早期 build 検証を含むべき」事例の再発 — **CR-α 系の read-only 検証にも
構文レベルの事前チェック（例: `cl /Zs` 単独 parse または targeted compile）を次回から付帯させる**。

### 実装者の残り領域への影響範囲（本 FAIL は test TU に局在）

| TU / 対象 | 本ビルドでの状態 | 根拠 |
|---|---|---|
| **BuildErrorPolicy.h の constexpr policy**（production/両側共通 header） | **コンパイル成功**（production TU 経由で実証） | RebuildDispatch.cpp.obj が 21:14 に新規生成（1,497,126 B・旧 19:09 から差し替わり）、error 0。ヘッダ側 ODR / C++20 constexpr / include dependency 問題は**出ていない** |
| **RebuildDispatch.cpp decision integration** | **コンパイル成功** | 同上。`milliseconds(decision.delayMs)`（:1294）を含む TU 全体が error 0 で obj 化 |
| **Scheduler 呼び出し**（RetryScheduler.cpp） | コンパイル成功 | [21/468] error 0（変更対象外の健全性確認） |
| **BuildErrorClassificationTests.cpp runTestF()** | **コンパイル失敗**（本 FAIL の唯一原因） | 上記 18 errors |
| Link（ConvoPeq / Harness / tests 全 target） | 未到達 | compile 中断のため |

※ BuildErrorClassificationTests の既存 obj は 19:06（Debug）/ 19:12（Release）の ND-04 期スタリール —
CR-α-1 追加分（runTestF）が初回ビルドで失敗したことをタイムスタンプが裏付け。

## 4. Release build

**未実施**（指示どおり Debug PASS 後のみ実施 — Debug FAIL により停止）。
`evidence/cra3_build_release.log` は生成されていない。

## 5. 禁止事項遵守（本工程の実績）

```text
Production source 変更: 0
Test source 変更: 0
CMake 変更: 0 / CTest: 0 / stress: 0
source 修正は FAIL 判定後も指示どおり実施せず（記録のみで STOP）
本工程で作成した artifacts: tools/cra3_build.bat（build helper）+ evidence/cra3_configure.log
  + evidence/cra3_build_debug.log（いずれも source/test/CMake 以外）
```

## 6. Overall

```text
Overall:
CR-α-3 = FAIL

Debug:   configure PASS / compile FAIL（C2653×9 + C2065×9 @ BuildErrorClassificationTests.cpp:270-299）
         link 未到達 / ConvoPeq target 未到達 / BuildErrorClassificationTests target FAIL / warning 0（中断まで）
Release: 未実施（Debug FAIL により）
diff/name-only: CR-α-1 期待 3 ファイル一致 / forbidden（RetrySchedulerTypes.h, AudioEngine.h）diff 0 確認
forbidden-file verification: PASS
```

## 7. Next

```text
CR-α-3 = FAIL → STOP（CR-α-4 へ進行不可）

最小修復候補（実施は次工程指示待ち・本工程では未適用）:
  src/tests/BuildErrorClassificationTests.cpp の using 宣言ブロック（:236-239）に
  `using convo::RetryDisposition;` を 1 行追加
  （実装意図は T-CRα-3/T-CRα-4 の非修飾呼び出しと整合・production source には無関係）

修復後は CR-α-3 をやり直し（Debug build → PASS 確認 → Release build → 報告書更新）。
```

## 遷移

```text
CR-α-1 Implementation        PASS
CR-α-2 Read-only verification PASS
CR-α-3 Debug/Release build   ← NOW = FAIL（test TU using 漏れ・1 行修復候補を記録）
CR-α-4 CTest 40/40 × 2       待機（CR-α-3 再 PASS まで不可）
CR-α-5 Retry-specific audit  待機
Closure                      待機
```
