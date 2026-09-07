# D162-2-I3-4-I — `/EHsc` Configuration / Target Provenance Audit — 作業報告

```text
Date:      2026-09-06
Type:      read-only audit (I0-I10)
Changes:   production / test / CMake / build.bat / src / tools = 0 (fixture は %TEMP% のみ)
Baseline:  I3-4-H5 GO (commit 9cacee1f+dirty)
判定:      **I3-4-I GO — Case A (NO DEFECT)。/EHsc の修正実装 (I3-4-J) は不要。**
```

## 要旨

- **`/EHsc` は Release / Debug / RelWithDebInfo の全 CXX 492 edge に存在 (492/492/492・欠落 0)。**
- 供給源は base `CMAKE_CXX_FLAGS = /DWIN32 /D_WINDOWS /EHsc` — CMake 4.4.3 の MSVC platform default
  (`Windows-MSVC.cmake` L259/L538、CMP0117 NEW → `_GR=""`、CMP0092 NEW → `_W3=""`)。
  configuration-independent であり、4 build tree の cache 一致 + %TEMP% fresh configure fixture
  (Case E) + モジュールソース照合の 3 重裏取りで確定。
- H5 が書いた「/EHsc は RWDI にも欠落している」(CMakeLists L1532) は set() 文字列レベルでは正しいが
  **edge レベルでは誤り**。H 監査自身の census (i3_4_h_h1_flag_provenance.txt) が修正前 RWDI edge に
  /EHsc ありを記録済み。broken-build ログに C1189 が無いこととも整合。
- JUCE C1189 ガード (`juce_BasicNativeHeaders.h:92`, `#ifndef _CPPUNWIND → #error`) は fixture
  Case C1 で C1189 再現 (機構は real) するが、全 edge に /EHsc があるため現在は到達不能。
- C flags への /EHsc 追加は不要 (C に例外モデルなし・fixture Case D)。
- **I8 = NO-CHANGE 契約**: RWDI set() への /EHsc 追加は禁止 (重複増加のみ)。
  将来 base CMAKE_CXX_FLAGS を明示上書きする場合のみ /EHsc が失われる点を契約に明記。

## ゲート一覧

I1 全 47 occurrence 分類 (A3/B3/C33/E1/F5/G2) — PASS / I2 492×3 census 欠落 0 — PASS /
I3 target provenance (33 target-level + 7 base-only) — PASS / I4 **Case A** — PASS /
I5 JUCE C1189 実証 — PASS / I6 fixture A-E — PASS / I7 /utf-8 と /EHsc の分離 — PASS /
I8 NO-CHANGE 契約 — PASS / I9 regression contract N-A 明示 — PASS / I10 変更 0 — PASS

## 棚卸し (本監査では修正しない)

1. **icx Debug/RWDI の /utf-8 375/492** (117 edge 欠落・target-level 補完なし target 群) —
   /EHsc とは別問題。IntelLLVM block は Release のみ global /utf-8 を持つ。後続監査候補。
2. ConvoPeq.md stale (H5 編集前に生成) — 次回再生成が必要。
3. CMakeLists コメント訂正候補: L1531-1533 (H5 の /EHsc 記述) と L1996 (/utf-8 記述)。
   次回 CMakeLists 触及許可時に doc-hygiene として実施。
4. build/ tree に .build_identity stamp 現存なし (次回 build 時に再生成・census に影響なし)。

## 詳細

full report: `evidence/D162-2I3/I3_4_I_EHSC_CONFIG_TARGET_PROVENANCE_AUDIT.md`
evidence: `i3_4_i0_freeze/`, `i3_4_i1_occurrence_table.txt`, `i3_4_i2_raw_census.json`,
`i3_4_i2_edge_census.txt`, `i3_4_i6_fixture_log.txt`

## ツール使用記録

- census・ninja/cache 解析: ctx-mode (ctx_batch_execute / ctx_execute python) + grep/sed/find (Bash/WSL)
- serena: 初期指示読込 (監査は file-level のためシンボル操作なし)
- cppcheck / clang-tidy / Dr.Memory: C++ コード変更 0 の flag provenance 監査のため適用なし
  (H5 報告と同一理由)。ccc / graphify / semble: 完全列挙課題は grep 系で網羅性が保証されるため不使用
- fixture: vcvarsall x64 + cl.exe / cmake 4.4.3 / ninja — %TEMP% のみで完結
