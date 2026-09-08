# D162-2-I3-4-H5 — RWDI Configuration Coherence 修正実装報告

```text
Date:            2026-09-06
Type:            CMake configuration coherence 修正（H4 契約どおり最小実装）
Changes:         CMakeLists.txt  +10 行 / -0 行（RWDI /utf-8 block のみ）
                 src/** runtime: 0 / tests: 0 / build.bat: 0 / src/tools/**: 0
                 /EHsc: 本修正に含めない（H4/H5-3 契約 — 別途 provenance audit）
Baseline:        I3-4-H 監査 GO (H1-H4) / gate v2 (I3-4-G) 変更なし
判定:            **H5-H10 全 PASS → I3-4-H GO（RWDI /utf-8 configuration coherence 完了）**
備考:            作業中に PC フリーズが 2 回発生。中断由来の破損 obj/PDB を排除した上で
                 Release/Debug は clean-first rebuild で検証（H7-H8 の証拠性を保持）。
```

---

## 0. 要旨

> `CMakeLists.txt` に `set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /O2 /Ob1 /DNDEBUG /utf-8")`
> （CXX + C）を追加したのみで、RWDI の `/utf-8` 欠落を構成レベルで解消した。
> 生成 census は **RWDI 483/492 → 492/492**（Release/Debug は 492/492 維持）、
> RWDI clean rebuild が rc=0 で完走し、G6 で失敗した C1083/C2143/C4430/C2888/C2065/
> C2061/C1075 は**すべて消失**。gate v2 は無変更のまま 3 config で rc=0、
> negative regression 8/8 で fail-closed 意味論が完全保存された。

## H5 — 実装内容（`i3_4_h5_h10_change_scope.txt`）

`CMakeLists.txt` L1525 の後（Release/Debug set の直後・同一 MSVC ブロック内）に追加:

```cmake
    # RelWithDebInfo も Release/Debug と同じ semantic source から /utf-8 を得る。
    # （欠落時の症状・I3-4-H 監査参照のコメント 6 行）
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Zi /O2 /Ob1 /DNDEBUG /utf-8")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO   "/Zi /O2 /Ob1 /DNDEBUG /utf-8")
```

- `/EHsc` は **含めていない**（H5-3 契約: 根因の因果を純粋に保つ。別途 audit を要する）。
- `git diff`: **+10 行 / -0 行**。他の変更（削除・書き換え）は無い。

## H6 — generated Ninja census（`i3_4_h5_h6_census.txt`）

reconfigure（`Ninja Multi-Config` + cl）後の impl-*.ninja 全 CXX/C compile edge:

| Config | CXX edges | /utf-8 | C edges | /utf-8 |
| --- | ---: | ---: | ---: | ---: |
| Release | 492 | **492** | 16 | **16** |
| Debug | 492 | **492** | 16 | **16** |
| RelWithDebInfo | 492 | **492**（前回 483 → **+9**） | 16 | **16** |

旧 7 broken target（D8_1 / DeferredDeletion / EQAnalysisUnit / EQBoundExcess /
EQProcessorMaxGain / FFTBackend / GainStaging）の RWDI edge は **9/9 すべて /utf-8 付き**。

## H7 — RWDI clean rebuild

```text
RWDI objs 全削除（CMakeFiles/*.dir/RelWithDebInfo + artefacts）→ full rebuild:
  cmake --build --config RelWithDebInfo   rc=0（551 steps・全 link 含む）
  旧エラー（C1083/C2143/C4430/C2888/C2065/C2061/C1075/C4819）: build log 内 0 件 — すべて消失
  7 target の obj は 09-06 00:03 に再 compile を確認（例: DeferredDeletionQueueReclaimTests.cpp.obj）
gate --check RelWithDebInfo: rc=0（初回は stamp の COHERENCE-4 が 2 項目を正しく検出:
  (a) 作業用 reconfigure が短形式 'cl' を cache に書いた → 完全パスで再 configure して解消
  (b) source_revision 0aeb22c → 9cacee1f（ユーザー commit 反映）→ stamp 更新で解消）
  → 再実行: 337 objs checked / 10 WARN+ALLOW / production 0 / suspicious 0 / unresolved 0
```

## H8 — 3 configuration regression

| Config | build | gate --check | CTest |
| --- | --- | --- | --- |
| Release | clean-first rc=0 | rc=0（337 objs / 10 WARN） | **40/40** |
| Debug | clean-first rc=0（+ PDB 破損 LNK1285 修復後に resume rc=0） | rc=0（337 objs / 10 WARN） | **40/40**（注記あり） |
| RelWithDebInfo | clean rebuild rc=0（H7） | rc=0（337 objs / 10 WARN） | RWDI CTest は build.bat 対象外（H7 の gate+link で充足） |

注記（Debug CTest 初回 39/40）: 連続高負荷ビルド直後の CTest で AudioEngineHarness のみ
10.7 秒で Fail（通常 ~20 秒）。**直接実行 ×3 = exit 0x0、ctest 単体再実行 = PASS、
full CTest 再実行 = 40/40** — 機器不安定期間の環境フレークと判定（コード・データ起因ではない）。
PC フリーズ由来の破損（Release juce_audio_formats.obj: CVT1107/LNK1123、
juce_graphics_Harfbuzz.obj: LNK1143、Debug ISRSemanticValidationTests.pdb: LNK1285）は
すべて該当 artifact 削除 + clean-first/再 link で解消 — **修正内容とは無関係の作業環境事象**。

## H9 — negative regression（`i3_4_h5_h9_negative_regression.txt`）

G4 fixture 系譜 8 ケースを gate v2（無変更）で再実行: **8/8 MATCH**

```text
system-only            → WARN + ALLOW (rc=0)   ✓
project direct         → PASS (rc=0)           ✓
project transitive     → PASS (rc=0)           ✓
prefix mismatch        → FAIL (rc=3 SUSPICIOUS) ✓
unresolved             → FAIL (rc=3 UNRESOLVED) ✓
production + #deps 0   → FAIL (rc=3 PRODUCTION) ✓
test-only system-only  → WARN + ALLOW (rc=0)   ✓
test-only + project 混入 → FAIL (rc=3 SUSPICIOUS) ✓
```

→ **CMakeLists 修正によって gate の fail-closed semantics は一切変質していない**。

## H10 — change-scope audit（`i3_4_h5_h10_change_scope.txt`）

```text
CMakeLists.txt                    changed  (+10 / -0・RWDI /utf-8 block のみ)
src/** runtime                    unchanged（HEAD 9cacee1f に対し runtime diff なし）
tests/**                          unchanged（ObservePathSingleSourceTests.cpp は content clean）
build.bat                         unchanged（hash 一致: 67b291a5f0f16e90）
src/tools/build_identity_gate.py  unchanged（hash 一致: a6ebf9effa8c0d44 = G-final）
```

## GO 条件対合（H5-H10）

| Gate | 条件 | 判定 |
| --- | --- | --- |
| H5 | CMakeLists に RWDI `/utf-8` を最小修正 | **PASS**（+10/-0・/utf-8 のみ） |
| H6 | Release/Debug/RWDI = 492/492/492 | **PASS**（+C 16/16/16・7 target 修復） |
| H7 | RWDI clean rebuild + link 成功 | **PASS**（rc=0・旧エラー全消失） |
| H8 | 3 config build + gate + CTest 回帰 PASS | **PASS**（CTest 40/40 ×2・Debug 初回 1 フレークは再実行で解消） |
| H9 | fail-closed negative regression PASS | **PASS**（8/8・gate 無変更） |
| H10 | CMakeLists 以外の変更 0 | **PASS** |

**総合判定 = I3-4-H GO（H5-H10 全 PASS）。D162 build-coherence chain における
RWDI configuration coherence は閉じた。**

## 添付 evidence（evidence/D162-2I3/）

```text
I3_4_H5_RWDI_UTF8_FIX_IMPLEMENTATION_REPORT.md   本書
i3_4_h5_h10_change_scope.txt                     H10 変更範囲監査
i3_4_h5_h6_census.txt                            H6 census（492/492/492）
i3_4_h5_h9_negative_regression.txt               H9 fixture 8/8
i3_4_h5_configure.log / i3_4_h5_reconfigure2.log configure ログ
i3_4_h5_build_{Release,Debug,Debug2,RWDI}.log    build ログ（clean-first 含む）
i3_4_h5_check_{Release,Debug,RelWithDebInfo}.log gate --check ログ
i3_4_h5_ctest_{Release,Debug,Debug_retest,Debug2}.log
i3_4_h_h0_freeze/CMakeLists.txt                  修正前 CMakeLists（H0 保存）
```

## ツール使用記録

- 修正・検証主軸: Edit（CMakeLists.txt）+ python（census・diff audit・fixture harness）
- cppcheck / clang-tidy / Dr.Memory: 本修正は CMake flag 定義の追加であり C++ コード
  変更を含まないため適用対象なし（前回監査に続き根拠を明記）
```
