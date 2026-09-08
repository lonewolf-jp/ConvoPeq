# D162-2-I3-4-K — IntelLLVM `/utf-8` Configuration Coherence Audit — 作業報告

```text
Date:      2026-09-06
Type:      read-only audit (K0-K10) — CMakeLists.txt 未変更・repository build tree 未変更
Baseline:  I3-4-I GO (Case A) / commit 9cacee1f+dirty
判定:      **I3-4-K GO — K7 = Case B (Configuration incoherence / no current build failure)。
           修正は機能的に不要。統一案 (K8) は次フェーズでユーザー判断。**
```

## 要旨

- **現行 CMakeLists での icx 現行生成** (%TEMP% scratch configure・repo は不 touch):
  Release CXX 492/492 `/utf-8` 付き、**Debug/RWDI は 375/492 (117 edge 欠落・26 target)**、
  C 16 edge 中 9 (7 target 欠落)。/EHsc は 3 config とも 492/492 (I3-4-I 維持)。
- 08-26 snapshot (build-icx) と現行生成の **edge 単位 diff = 0** (508 edge × 3 config)。
  H5 修正 (MSVC block) は icx に影響しないことが実証。
- **欠落の完封 (K3)**: ①ISR-family 19 target — /utf-8 の唯一の供給が L850-871 の
  `if(MSVC AND NOT ... IntelLLVM)` block 内で icx では不活性、②ISR test 7 target —
  target-level /utf-8 を一切持たず、cl では H5 の config-level set (L1519/L1524/L1534) で
  補完されるが IntelLLVM block には Debug/RWDI の set が存在しない。
  IntelLLVM block は Release のみ global `/utf-8` set (L1606/L1607) を持つ (現行生成物で再確認)。
- **compiler semantics (K4)**: icx は BOM 無し UTF-8 source を /utf-8 無しでも UTF-8 として
  解釈する (Case A/B の実行出力が完全一致・LEN=30・bytes 一致)。対比として cl は同一 source で
  C4819 + C2065 パース崩壊 (Case B2) — H5 の defect クラスは cl 固有。
- **JUCE interaction (K5)**: juce_recommended_config_flags は 0 参照・JUCEUtils に /utf-8 なし。
  icx Debug edge の /Od 508 件は CMake 既定 Debug flags 由来で JUCE interface ではない。

## ゲート一覧

K1 provenance PASS / K2 census PASS / K3 target provenance PASS / K4 semantics PASS /
K5 JUCE PASS / K6 fixture PASS / **K7 Case B** / K8 修正契約記録 PASS / K9 regression contract PASS / K10 変更 0 PASS

## K8 修正契約 (実装なし・記録のみ)

- 機能的必須項目なし (Case B)。
- 統一する場合の第一候補 = 案1: IntelLLVM block 内に cl block 対応の config-level set
  (CXX+C × Debug/RWDI) を追加。案2: L1606/L1607 から /utf-8 を外す (icx 不要論)。
  推奨基本線 = **現状維持 + 本報告を公式記録** (実害 0 のため)。実施は次フェーズでユーザー判断。
- 禁止: ad-hoc target options / BOM 追加 / test source 変更 / gate whitelist /
  compiler workaround / MSVC block への影響。

## K9 (将来 K-impl 実施時の基準)

icx: /utf-8 CXX 492/492/492・C 16/16/16 + 3 config build rc=0。
MSVC: /utf-8 492/492/492 (H5/H8)・/EHsc 492/492/492 (I) 不変条件。I3-4-G negative 8/8。change scope = IntelLLVM block のみ。

## 付記

1. repository の build-icx tree は再 configure していない (read-only 維持)。現行生成は
   `%TEMP%\icx_i34k\bld` で取得・保存済み (再検証可)。snapshot との edge diff 0 を確認済み。
2. icx configure には oneAPI setvars に加え Windows SDK LIB/INCLUDE の明示 probe が必要
   (build.bat L137-158 と同様) — 手順は configure log に記録。
3. 未実装新規 CR は BuildError retry backoff/count/telemetry と CW-8 PublishedWorldObservation
   atomic snapshot contract の 2 系統 — I3-4-K 後も勝手な実装着手はせず CR 判断を分ける。
   Phase-II recovery episode 系 (D159 freeze) も同様。Practical Stable ISR Bridge Runtime 原則
   (Build/Validate/Publish/Retire/Delete 責務分離・RT 観測限定) に照らし source-side
   workaround は不使用。

## 詳細

full report: `evidence/D162-2I3/I3_4_K_INTELLLVM_UTF8_CONFIG_COHERENCE_AUDIT.md`
evidence: `i3_4_k0_freeze/`, `i3_4_k1_provenance_chain.txt`, `i3_4_k2_current_configure.log`,
`i3_4_k2_census.json`, `i3_4_k3_target_provenance.json`, `i3_4_k4_k6_fixture_log.txt`,
`i3_4_k4_fixture_source.cpp`

## ツール使用記録

- census・condition chain・FLAGS 解析: ctx-mode (ctx_execute python) + grep/sed/find
- icx 現行生成: oneAPI setvars.bat intel64 + Windows SDK probe + cmake 4.4.3 (scratch %TEMP%)
- fixture: icx 2026.1.0 (Case A/B/C/D/E) + cl (Case B2 対比)
- serena: 初期指示読込。cppcheck / clang-tidy / Dr.Memory: C++ 変更 0 の flag provenance 監査のため適用なし
  (H5/I 報告と同一理由)。ccc / graphify / semble: 完全列挙課題は grep 系で網羅するため不使用。
