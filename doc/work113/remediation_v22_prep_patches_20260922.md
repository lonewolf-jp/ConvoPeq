# ConvoPeq 改修計画 v2.2 — Phase 0 着手前 パッチ案（2026-09-22）

- **位置付け**: 中間保存 §4 の技術下準備。**production への適用は v2.2 承認 + ユーザー GO 後**
- **基準**: HEAD `8f127bfe` / v2.2 `remediation_plan_20260922_v2.2_revised.md`
- **原則**: 案 E = candidate hypothesis（確定表記禁止）/ D1 authoritative = 補正軸（R8-1）

---

## 1. CMake option（B-1 Phase 1 前提・未適用）

**対象**: `CMakeLists.txt`

```cmake
# --- 挿入位置 A: :40 近傍（CONVOPEQ_ENABLE_CLANG_TIDY 等と並置） ---
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
       "Correct polyphase gain convention (B-1 案E candidate)" OFF)

# --- 挿入位置 B: add_subdirectory(JUCE)（:1043）の後・juce_add_gui_app（:1062）の前 ---
# 例: r8brain INTERFACE 定義（:1048-1057）の直後・GUIアプリ定義（:1059）の手前
add_compile_definitions(
    CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
```

**注意事項（v2.2 継承）**:
- C++ は **`#if CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（`#ifdef` 禁止 — 値 0 でも defined）
- 既定 **OFF** / runtime flag 不採用
- 既存前例: `NUC_DEBUG_GUARDS`（:52-58 `add_compile_definitions`）
- 切替: `build.bat Release nopause -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON`
- ロールバック: OFF で rebuild（compile-time rollback）

**production C++ 挿入案（未適用・candidate）**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() :557 付近
        convValue *= 2.0;                 // 既存
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // 案 E candidate（両 polyphase 位相へ対称）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

---

## 2. F-2 案 A（test-only・呼出順差替・未適用）

**対象**: `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`

**現状（誤順）** — eq モード:

```cpp
// :1799-1803
configureProbeFlatEQ(e);                 // 内部で setEQAGCEnabled(false) :1081
// ...
e.setAutoGainStagingEnabled(false);      // ON→OFF 時のみ EQ AGC を !enabled=ON に戻す
```

`setAutoGainStagingEnabled`（`AudioEngine.h:1416-1425`）は **ON→OFF 遷移時のみ**
`getEQProcessor().setAGCEnabled(!enabled)` を呼ぶため、configure 後に staging を切ると
EQ AGC が再 ON される。

**修正案 A（eqdiag パターンへ揃える）** — 1 行差替相当:

```cpp
// :1799-1803 を次へ
e.setAutoGainStagingEnabled(false);      // 先に staging OFF（既定 true → OFF で EQ AGC OFF）
configureProbeFlatEQ(e);                 // その後に probe flat EQ（内部でも setEQAGCEnabled(false)）
// setEQFilterStructure / setEqBypassRequested は現状どおり
```

**参照（正順の既存実装）** — eqdiag / eqos パターン `:1833-1834`:

```cpp
e.setAutoGainStagingEnabled(false);
configureProbeFlatEQ(e);
```

**検証**:

```text
build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0
期待: staging=0 eqAGC=0
```

**適用条件**: Step 3（ユーザー GO 後）/ production 変更なし / 1 行 revert 可。

---

## 3. R-2 fail-closed 実装スケッチ（未適用）

**仕様（v2.2 §5 / R8-9）**:
- 数値 parse を try/catch + `[BUZZ] FAIL: invalid numeric` + 非ゼロ終了
- 既存 fail-closed 前例: `parseOnOff`（BassBuzz :1572-1577）/ B-2 `tryParseBuzzProcessingOrder`

### 3.1 BassBuzzMeasurement.cpp — 共通ヘルパ案

```cpp
// parseOnOff（:1572）の隣に追加する想定
auto parseIdxOrFail = [](const char* flag, const std::string& s, auto parseFn) -> int {
    try {
        return parseFn(s);
    } catch (...) {
        std::fprintf(stderr, "[BUZZ] FAIL: invalid numeric %s (got '%s')\n", flag, s.c_str());
        std::exit(2);
    }
};
```

### 3.2 対象箇所（実測行・R8-9 反映）

| 対象 | 行 | 現状 | 案 |
|------|----|------|----|
| parseHcIdx | :1562 | `return std::stoi(s);` try なし | 呼出側で parseIdxOrFail |
| parseLcIdx | :1567 | 同上 | 同上 |
| `--buzz-sr=` | :1582 | `std::stod` | parseIdxOrFail |
| `--buzz-block=` | :1583 | `std::stoi` | parseIdxOrFail |
| `--buzz-quiet=` | :1591 | `std::stoi` | parseIdxOrFail |
| `--buzz-dur=` | :1592 | `std::stod` | parseIdxOrFail |
| `--buzz-probe-level=` | :1590 | `std::stof` | parseIdxOrFail |
| `--buzz-flip-t=` | :1614 | `std::stod` | parseIdxOrFail |
| `--nuc6=` | :1615 | `std::stod` | parseIdxOrFail |
| lambda 呼出 | :1606-:1611 | parseHcIdx/parseLcIdx | 失敗時に exit |

### 3.3 PPIT — PublishPipelineIntegrationTests.cpp

| 対象 | 行 | フラグ |
|------|----|--------|
| stoi | :1136 | `--t1=` |
| stoi | :1146 | `--t2=` |
| stoi | **:1149** | `--duration-s=`（R8-9 で追加） |
| stoi | :1157 | `--t3=` |
| stoi | :1166 | `--t4=` |

```cpp
// 各 stoi を try/catch 包み（[BUZZ] ではなく PPIT 用メッセージでも可・仕様統一推奨）
auto stoiOrFail = [](const char* flag, const std::string& s) -> int {
    try { return std::stoi(s); }
    catch (...) {
        std::fprintf(stderr, "[BUZZ] FAIL: invalid numeric %s (got '%s')\n", flag, s.c_str());
        std::exit(2);
    }
};
```

### 3.4 regression test 案（要旨）

- 不正文字列（`"abc"`, `""`, `"12abc"`）で非ゼロ終了
- stderr に `invalid numeric` を含む
- 正常値の既存テストは変更しない

---

## 4. モデル D1 補正軸（R8-1・本セッション適用済み）

**ファイル**: `doc/work113/model_polyphase_20260920.py`

| 区分 | 定義 |
|------|------|
| **D1 AUTHORITATIVE（新）** | up 出力長 2N・tone bin = **f̂·N**・image bin = **N−f̂·N**・argmax 検証付き |
| **D1 OLD AXIS（破棄済参考）** | 旧定義 f̂ vs 0.5−f̂ を 2x レート bin で読む（v1.7 相当・解釈破棄済） |
| 期待値 | base **−9.5424 dB 構造定数** / cand **−84〜−116 dB**（v2.2 §2.7.3） |

結果ファイルには次のラベルを出力する:
- `D1 base/cand` = 補正軸（authoritative）
- `D1-ARGMAX` = tone/image bin と argmax 検証
- `D1 OLD AXIS base/cand` = 破棄済参考

**再実行**:

```bash
wsl -d Ubuntu-26.04 -- bash -lc \
  'cd /mnt/c/VSC_Project/ConvoPeq && python3 doc/work113/model_polyphase_20260920.py'
```

---

## 5. Shadow Reference ヘッダ（test-only 骨格・本セッション作成済み）

**ファイル**: `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`

- production 変更 **0**
- D1 bin 計算（補正軸）と expected DC（base 0.75^N / cand 1.0）を公開
- `CONVOPEQ_POLYPHASE_REF_CANDIDATE` 0/1 で shadow の candidate/base を切替
- **processUp/processDown 本体は Phase 0 実装時に production 契約へ合わせて追加**（骨格段階では API と計測定義のみ）

---

## 6. 判断待ち（本下準備では解消しない）

| 項目 | 推奨 |
|------|------|
| G-0 DESIGN-CONTRACT-A | ユーザー明示承認 |
| B-1 Phase 0 GO | 承認後に測定着手 |
| F-2 案 A | Step 3 で適用 |
| R-2 実装 | Step 7（最優先・極小） |
| O-1〜O-14 | ユーザー判断 |
| production C++ / CMake | **未適用のまま**（本書は案のみ） |

---

*本ファイルは pre-work パッチ案であり、承認済み改修計画の本体ではない。*
