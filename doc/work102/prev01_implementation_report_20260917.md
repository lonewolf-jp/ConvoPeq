# WORK102-PREV-01 — Implementation Report（Preview Loader Admission / Graceful Failure）

- **作成日**: 2026-09-17
- **種別**: Implementation Report（commit/push 未実施。POST-AUDIT → FINAL GATE → commit/push 待ち）
- **判定**: **IMPLEMENTED**（凍結契約 FC-1〜FC-5 全項目を実装・検証済み）
- **凍結契約の正本**: [doc/work102/prev01_contract_freeze_20260917.md](../doc/work102/prev01_contract_freeze_20260917.md)
- **検証台帳**: [evidence/WORK102_PREV01_IMPL_VERIFICATION.txt](../../evidence/WORK102_PREV01_IMPL_VERIFICATION.txt)

## 0. Baseline

```text
HEAD / origin/main = 859718e4（ahead/behind 0/0・WORK102 CLOSED）
ConvoPeq.md        = 再生成 2026-09-17 12:06:24 → --check NEWER_SRC_COUNT=0 / FRESH
production 変更    = 2 TU / tests = 2 TU / CMake = 0 / UI = 0 / commit = 0 / push = 0
```

## 1. 変更境界

| ファイル | 変更 | 内容 |
| --- | --- | --- |
| `src/convolver/ConvolverProcessor.ResampleAndFallback.cpp` | M | include 追加 + `loadImpulseResponsePreviewFile`（:271-382）本体のみ |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | M | include 追加 + `analyzeImpulseResponseFile`（:458-592）関数のみ |
| `src/tests/AudioEngineHarness/IRLoadAdmissionTests.cpp` | A | +228 行（include + PA〜PE 5 check + runner `runIRLoadPreviewAdmissionTests`） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | M | +7 行（宣言 + main 内呼出しのみ。新規 CTest target なし） |
| `ConvoPeq.md` | M | 再生成（467 行域） |

commit boundary 外を**混入させない**: 既存 staged items（`.gitignore` / `build.bat` / `doc/work68/*`）、
README.md・evidence/* の既存差分、他 untracked（doc/work101 等）は PREV-01 commit に入れない
（WORK102 Final Gate と同一の boundary 規律）。

## 2. 実装内容（凍結契約との対応）

### FC-1（P0）Admission ordering — narrowing 前に FC-FORM-4 → 3 → 2 → 1

実測（`loadImpulseResponsePreviewFile`）:

```text
:291 reader 生成
:296-297  fileLength (int64) / rawChannels (unsigned・narrowing 前)
:299  FC-FORM-4  admitChannelCountNonZero → diagnosticChannelLimit
:305  FC-FORM-4  admitFileLengthNonZero   → diagnosticLengthLimit
:311  FC-FORM-3  admitFileLengthRepresentable → diagnosticLengthLimit
:318  FC-FORM-2  admitChannelCount        → diagnosticChannelLimit（>8ch 決定論的拒否）
:325  FC-FORM-1  admitByteBudget          → diagnosticByteLimit（除算形）
:332  const int numChannels = static_cast<int>(rawChannels)   ← narrowing は FC-FORM-2/4 の後
```

旧実装の独自 constexpr（`maxFileLength = 2147483647`）・独自 `numChannels <= 0`・
narrowing 前置・byte bound 不在はすべて解消。main loader（LoaderThread.cpp FC-INV-9）と
同一述語・同一順序。

### FC-2（P0）全量 transient の廃止 → bounded / chunked read

```text
:336  constexpr int64 kStreamChunk = 256 * 1024;    ← 局所 constexpr（IRLoadAdmission.h 無変更）
:337  AudioBuffer<float> tempFloatBuffer(numChannels, static_cast<int>(kStreamChunk));
:338  makeAlignedArray<double>(kStreamChunk)（null 検査 → OOM graceful）
:345  loadedIR.setSize(numChannels, static_cast<int>(fileLength))   ← FC-FORM-1 の後の唯一の O(len) 確保
:346-366  チャンク読込ループ（jassert(offset+chunk ≤ INT32_MAX) の belt-and-braces 同梱）
```

preview ピーク = `loadedIR`（ch×len×8 ≤ 1 GiB）+ 256 KiB × 2。旧全量
`tempFloatBuffer(numChannels, fileLength)` / 全量 `tempAlignedBuffer` は消滅（S1）。

### FC-3（P0）no-throw failure boundary

`analyzeImpulseResponseFile` を唯一の例外境界にした（実測 :571 bad_alloc → :577 std::exception → :583 catch(...)）:

```text
bad_alloc       → "IR too large (Out of Memory)"   （LoaderThread.cpp:131 と同文言）
std::exception  → "Error analyzing IR: " + e.what()
catch(...)      → "Unknown error analyzing IR"
```

全 failure 経路で `IRLoadPreview` を値として返す → worker（**無変更**）→ callAsync →
`finishAsyncIRLoadPreview` → `setIRPreviewInProgress(false)` ＋ MessageBox。
JUCE ThreadPool による例外握り潰しによる **completion 喪失経路の消滅** を runtime で
確認（PE: missing / garbage → success=false ＋ 非空 errorMessage）。
`bad_alloc` 発火は決定論的に不可 → **S3 構造検証**として報告（executable PASS と誤表示しない）。

### FC-4（P1）FC-FORM-5（trim 後）

`:508-522` — trim 完了後・`resampleIR` 呼出し直前に `admitResampleOutput(trimmedLength,
loadedSampleRate, processingSampleRate)` ＋ `diagnosticResampleLimit`。raw 適用はしない
（R2-D4 凍結維持）。`resampleIR` 本体は無変更。

### FC-5（P1）diagnostic single source

admission rejection 4 経路（length / channel / byte / resample）を
`convo::irload::diagnostic*` に統一。not found / unsupported / read 失敗の 3 文言は
凍結文書どおり維持。

### 意図仕様の維持

hash は計算しない（FC-8 対象外）/ cancellation は supersession-only（`neverCancel` 維持・
新 API 導入なし）/ ThreadPool(1) 逐次 / engine publication 追加なし
（成功時の効果は `applyAutoDetectedIRLength` + `requestConvolverPreset` のみ — いずれも既存経路）。

## 2b. Runtime test results（実測）

| Check | 内容 | Debug | Release |
| --- | --- | --- | --- |
| PA | preview channel admission（1/2/8ch accept・9ch reject・diagnostic 文言） | PASS | PASS |
| PB | チャンク読込等価（300k samples・2 chunk 境界サンプル一致） | PASS | PASS |
| PC | FC-FORM-5 trim 後 accept（raw 50s > hardMax） | PASS | PASS |
| PD | FC-FORM-5 trim 後 reject（3s@44.1k → 768k） | PASS | PASS |
| PE | failure completion（missing / garbage → success=false + 非空 errorMessage） | PASS | PASS |
| WORK102 A–L | main loader 既存契約の回帰（I は 13 sizes oracle） | PASS | PASS |

CTest: **Release 40/40**（57.48s）・**Debug 40/40**（82.90s）。
Debug 第1回のみ T-I2-3（publish/registration 経路・PREV-01 変更面外）が 11.13s で FAIL →
再実行 40/40 PASS・単独再実行 PASS（60.83s）・直接実行 exit 0（全 PREV-01/WORK102 check PASS）。
環境/タイミング flake（clean 再構築直後の初回）と分類し、PREV-01 regression と
再分類しない。POST-AUDIT での再観察を推奨に記載。

## 3. Build / gates / static

```text
Build   : Release / Debug 全文ビルド BUILD_EXIT=0（FAILED 0 / compiler error 0 件）
identity gate   : GATE-OK（--check 再検証・stamp 一致）
dependency gate : GATE-OK（339 relevant .obj / production #deps 0 = 0 / test-only warn 10 件は許容）
clang-tidy      : ResampleAndFallback.cpp 0/0・IRLoadAdmissionTests.cpp 0/0。
                  StateAndUI.cpp 2 件（:445/:798 NewDeleteLeaks）は diff 外既存コード
                  （LatencySnapshot publish 経路）への pre-existing 指摘で、PREV-01 defect
                  として再分類しない。変更コードに帰属する指摘: 0。
cppcheck        : pre-existing unknownMacro 2 件（未変更 TU）のみ。変更コード新規指摘 0。
ConvoPeq        : 再生成 12:06:24 → NEWER_SRC_COUNT=0 / FRESH
Architecture    : atomic 追加 0 / RT 変更 0 / Publish・Retire・Epoch・RuntimeWorld・
                  LifetimeBudget・ISRShutdown・MAX_IR_LATENCY・DELAY_BUFFER_SIZE 変更 0 /
                  UI 制御構造変更 0（ConvolverControlPanel.cpp diff = 0）
```

## 4. 構造検証（S1–S5）

```text
S1  全量 tempFloatBuffer 形状の消滅            : 確認（チャンク形のみ）
S2  admission ordering 実測（行番号: 299→305→311→318→325→332→345）: PASS
S3  no-throw 境界の存在 + worker 無変更        : 確認
S4  convo::irload:: 横展開なし（3 TU 限定）    : 確認
S5  IRLoadAdmission.h 無変更（git diff = 0）   : 確認
```

S1–S5 は**構造検証**として明示し、executable PASS と誤表示しない（WORK102 の J/K 規律と同一）。

## 5. Build environment note

本 session の background shell は vcvars/oneAPI 環境が未初期化であり、またコンソール
コードページ 932 のままでは cl の日本語 `/showIncludes` 出力が ninja の UTF-8
`msvc_deps_prefix`（rules.ninja）と一致せず dependency gate が fail-closed になった。
対策として、未変更の build.bat の代わりに wrapper（`chcp 65001` + vcvars64 + oneAPI
setvars + `cmake --build` 2 フェーズ: ConvoPeq bootstrap → 全 target）でビルドし、
identity/dependency gate は `build_identity_gate.py --check` で再検証した。
build.bat / CMakeLists.txt / ソース以外の環境は一切変更していない（wrapper は
`_prev01_run_build.cmd` / `_prev01_gate_check.cmd`・作業用 untracked・commit 対象外）。

## 6. 完了条件チェック（凍結文書 §6）

```text
Build BUILD_EXIT=0 ×2          ✓   CTest 40/40 ×2            ✓（Debug は初回 flake 記録 + 再実行 PASS）
Runtime PA–PE PASS ×2構成 ✓        S1–S5 構造検証明示 ✓
static clean（変更コード 0）✓      ConvoPeq NEWER_SRC_COUNT=0 ✓
architecture drift 0 ✓            Parity matrix 消化（channel≥1/≤8/INT32/1GiB/admission先行/
                                   FC-FORM-5 trim後/hash N-A/bad_alloc graceful/completion/
                                   cancellation 維持/diagnostic 共有）✓
commit boundary 規律 ✓（混入 0）
```

## 7. 次工程

```text
POST-AUDIT（本台帳の再検証 + 独立確認）
      ↓
FINAL GATE（commit boundary / freshness / architecture 再監査）
      ↓
commit / push（WORK102 と同一規律）
```

本報告時点で **commit/push は実施していない**。
```
