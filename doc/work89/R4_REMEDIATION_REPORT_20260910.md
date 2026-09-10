# work89 R-4 改修報告書（D-2 AGC atomic 整理 × D-3 デッドコード sync 系削除 統合）

- **実施日**: 2026-09-10
- **基準**: 設計書 `doc/work89/DESIGN_R4_D2D3_20260910.md` v2.3（監査承認: CONDITIONALLY APPROVED — IMPLEMENTATION GO）
- **authority**: `ConvoPeq.md Generated: 2026-09-10 21:11:33` / **NEWER_SRC_COUNT=0（FRESH）**
- **判定**: **R-4 CLOSED**（Phase 1・Phase 2 とも全 gate PASS）— 最終監査「**APPROVED — R-4 CLOSED / IMPLEMENTATION VERIFIED**」§9 参照
- **状態**: 変更は**未 commit**（§7 commit 手順 = 実行済みの hunk 分離表つき。残作業はコマンド実行のみ）

---

## 0. 総括

| Phase | 項目 | 状態 | 検証 |
|---|---|---|---|
| 1 | P-V1 verifier semantics 実証 | ✅ CLOSED | probe: present→exit1 / absent→exit0 |
| 1 | W-1 EQProcessor sync 3 関数削除 | ✅ CLOSED | G4-S 0 件・build 147/147 |
| 1 | W-2 ConvolverProcessor::syncStateFrom 削除 | ✅ CLOSED | G4-S 0 件 |
| 1 | W-4 verifier 意味論分離改名 | ✅ CLOSED | verifier PASS（3 dormant edges） |
| 2 | W-3 AGC atomic 3 個 + publish 9 行削除 | ✅ CLOSED | G4-B コード参照 0 件・fetchAdd 3 箇所無傷 |
| 全 | ビルド/CTest（Phase ×2 config） | ✅ **PASS ×4** | 40/40 全 4 ラン |
| 全 | cppcheck | ✅ 新規指摘 0 | JUCE マクロ限界注記のみ（既存） |
| 全 | ConvoPeq.md 再生成 | ✅ | 21:11:33 FRESH |

**成果**: work89 §15.4 の OPTIMIZATION / PROCESS DEBT として残置だった D-2・D-3 を閉じ、shadow 単一管理（AGC）と dead-code ゼロ（sync 系）を達成。再発明は verifier の FORBIDDEN_SYMBOLS ガードで検知可能。ACTIVE/OPEN BUG は従前どおり 0 を維持。

---

## 1. 承認経緯（監査履歴）

| 版 | 監査判定 | 解消条件 |
|---|---|---|
| v1 | CONDITIONALLY APPROVED — IMPLEMENTATION HOLD | D-2 全参照分類・verifier 意味論分離・G4 構造化の 3 必須修正 |
| v2 | CONDITIONALLY APPROVED — MINOR DESIGN TEXT HOLD | 修正 A（resetToDefaults は 3 経路・根拠は serial protocol）・修正 B（R-3a/R-3b/R-1 の用語分離） |
| v2.2 | **IMPLEMENTATION HOLD（戻し）** → 同 session 内で再照合 | P-D2-2（prepareToPlay reprepare 同値性の 4 項目証明）+ P-V1 実装前必須化 + authority 直接確認 |
| v2.3 | **IMPLEMENTATION GO**（Phase 1/2 分離維持・P-V1 を Phase 1 先頭 gate） | 表現修正 1 点（「正常完了する prepareToPlay では serial を前進」）適用済み。監査側で 15:48:13 版直接取得・再照合完了（HOLD 解除） |

---

## 2. Phase 1 実装詳細（D-3 + W-4）

### 2.1 P-V1（実装前必須 gate）

verifier 本体の semantics をコード挙動で実証（設計 §2.4 の.AC-V1）:
1. `src/__pv1_probe.h` に `syncStateFrom` 呼出構文を持つ probe を一時配置 → `dead_code_callers_verifier.py` **exit 1（FAIL 検出）**
2. probe 除去 → **exit 0（PASS・absent=PASS 確認）**

→ `FORBIDDEN_SYMBOLS = present なら FAIL / absent なら PASS`（存在必須ではない）を AC-V1 として記録。

### 2.2 W-1: EQProcessor sync 系 3 関数削除

| 対象 | 削除範囲 | 規模 |
|---|---|---|
| `EQProcessor::syncStateFrom` | EQProcessor.Core.cpp のセクション（`// 状態同期` ヘッダ〜関数終）| 定義計 3 関数 = **116 行**（R-1 で追加した休眠契約コメント含む） |
| `EQProcessor::syncBandNodeFrom` | 〃 | 〃 |
| `EQProcessor::syncGlobalStateFrom` | 〃 | 〃 |
| 宣言 3 個 | EQProcessor.h:378-382（コメント 2 行含む） | 5 行 |

削除スクリプトは行頭アンカー検証（`状態同期`/`単一バンド同期`/`グローバル状態同期`/次セクション`メモリ事前確保`）を必須化して実施。削除後 `grep -c syncStateFrom src/eqprocessor/EQProcessor.Core.cpp` = **0**。

### 2.3 W-2: ConvolverProcessor::syncStateFrom 削除

| 対象 | 削除範囲 | 規模 |
|---|---|---|
| 定義 | StateAndUI.cpp の `// [DEAD CODE]` 注記（4 行）+ 関数全体 | 42 行 |
| 宣言 | ConvolverProcessor.h の注記（3 行）+ 宣言 | 5 行 |

**維持**: AudioEngine.Parameters.cpp:645/:661 の「旧 syncStateFrom 方式は撤去済み（doc/work89/INTEGRATED-BUG-LIST.md §12 参照）」コメント（設計 §2.2）。

### 2.4 W-4: verifier 意味論分離（設計 §2.4）

- `WATCHED` → **`FORBIDDEN_SYMBOLS`**、`DORMANT_EDGES` → **`DORMANT_EDGE_GUARDS`**（**判定ロジックは無変更・改名のみ**）
- docstring に 3 層意味論表 + P-V1 実証記録 + R-4 で削除済み 4 関数のパターンを**再発明ガードとして残置**する方針を明記
- 実行結果: `[PASS] No call sites of dead-code functions detected (3 dormant edge(s) verified)`

---

## 3. Phase 2 実装詳細（D-2 + W-3）

| 対象 | 削除 | 根拠 |
|---|---|---|
| EQProcessor.h:569-571（宣言 3 行） | 削除→削除根拠コメントに移設 | P-D2-1: 全 18 site 分類完（宣言 3 + Write 9 + 休眠 feed 6。feed 6 は Phase 1 で関数ごと消滅済み）|
| Core.cpp resetToDefaults publish 3 行 | 削除→serial protocol 注記 | 修正 A: constructor / **UI Reset（EQControlPanel.cpp:433→AudioEngine.h:1318→EQEditProcessor.cpp:116）** / loadPreset の 3 経路とも `requestAgcReset()`→serial 前進→Audio Thread shadow 自己更新で同一 |
| Core.cpp reset() publish 3 行 | 削除→注記 | 同 fetchAdd が同関数内に既存（休眠コード） |
| Core.cpp prepareToPlay publish 3 行 | 削除→注記 | **P-D2-2**: 正常完了する prepareToPlay は必ず `fetchAddAtomic(agcResetSerial,1)` を実行（現行 :719 相当）→ shadow 自己更新が初期化を担う |

**削除禁止条項の実証（AC-W3）**: `fetchAddAtomic(agcResetSerial…` は EQProcessor.h:564（requestAgcReset）・Core.cpp reset / prepareToPlay の **3 箇所とも無傷**を grep で確認。`rtAgc*Shadow`（宣言 3 + Processing.cpp 12 site）、`agcAttack/Release/SmoothCoeff`（宣言+参照）も維持。

**残存名（判定対象外）**: EQProcessor.h の契約コメント 2 箇所（旧 protocol 歴史参照・strip 後判定のため G4 不合格にならない）。

---

## 4. 検証 gate 結果

### 4.1 Phase 1

| Gate | 結果 | 証跡 |
|---|---|---|
| G1 Debug build | ✅ PASS 147/147 | `evidence/work89_r4_p1_build_debug.log` |
| G1 Release build | ✅ PASS 239/239 | `evidence/work89_r4_p1_build_release.log` |
| G2 CTest Debug（UTF-8） | ✅ **40/40** | `evidence/work89_r4_p1_ctest_debug.log` |
| G2 CTest Release | ✅ **40/40** | `evidence/work89_r4_p1_ctest_release.log` |
| G3 verifier | ✅ PASS | 実行出力 |
| G4-S decl/def/callable | ✅ **0 件** | rg 構造化検査 |
| G5 cppcheck | ✅ 新規指摘 0 | 実行出力 |

### 4.2 Phase 2

| Gate | 結果 | 証跡 |
|---|---|---|
| G1 Debug build | ✅ PASS 147/147 | `evidence/work89_r4_p2_build_debug.log` |
| G1 Release build | ✅ PASS 239/239 | `evidence/work89_r4_p2_build_release.log` |
| G2 CTest Debug | ✅ **40/40** | `evidence/work89_r4_p2_ctest_debug.log` |
| G2 CTest Release | ✅ **40/40** | `evidence/work89_r4_p2_ctest_release.log` |
| G3 verifier | ✅ PASS（3 dormant edges・FORBIDDEN absent） | 実行出力 |
| G4-B AGC 完全一致 | ✅ **コード参照 0 件** | rg `\bagc…\b` コード行 0 |
| G5 cppcheck | ✅ 新規指摘 0 | 実行出力 |
| G6 ConvoPeq.md | ✅ `Generated: 2026-09-10 21:11:33` / NEWER_SRC_COUNT=0 FRESH | `output_sourcecode_markdown.py --check` |

### 4.3 受け入れ基準（AC）判定

| AC | 判定 |
|---|---|
| AC-W1 sync 系 3 宣言・定義消滅 | ✅（G4-S 0 件） |
| AC-W2 Convolver 側消滅 + Parameters §12 コメント維持 | ✅ |
| AC-W3 AGC 3 名 0 件 + rt*Shadow/coefficient/agcResetSerial + **fetchAdd 3 箇所**無傷 | ✅ |
| AC-W4 verifier 3 層意味論 + PASS | ✅ |
| AC-R3a 追補（3 契約分離）| ✅（Core.cpp reset/prepareToPlay の R-3a/R-3b/R-4 注記として実装済み） |
| AC-V1 P-V1 実証記録 | ✅（§2.1） |
| AC-G2 CTest 40/40 ×2 | ✅（Phase ごとに計 4 ラン） |

---

## 5. 実装過程の環境教訓（再発防止の実装）

1. **CR ゴミファイル事故の再発防止（標準手順化）**: 前回 R-1 実装時、bash の `while read`（`-r` なし）が Windows パスのバックスラッシュ + CR を誤処理し、src/ 直下に CR 付きゴミ .cpp/.h を生成 → `fs::recursive_directory_iterator` を使う RuntimeWorldAuthorityProjectionContract が namespace 解決不能で FAIL した。本次 R-4 では touch を **`IFS= read -r` + `tr -d '\r'` + `[ -f ]` 検証**で実施し、実行後に **`find src -name $'*\r*'` が 0 件**であることを毎回確認（Phase 1・Phase 2 両方で確認済）。
2. **ビルド並列度**: 前回の C1060（cl.exe ヒープ枯渇）を踏まえ、全ビルドを `ninja -j2`（vcvars64 + oneAPI include bat）で実施。
3. **CTest 実行環境**: `chcp 65001` 付き bat を使用（Unicode 出力テスト対策）。

---

## 6. 変更ファイル一覧（未 commit）

| ファイル | Phase | 変更種別 |
|---|---|---|
| `src/eqprocessor/EQProcessor.Core.cpp` | 1 + 2 | sync 3 関数削除（-116 行）・AGC publish 9 行→契約コメント化 |
| `src/eqprocessor/EQProcessor.h` | 1 + 2 | sync 3 宣言削除・AGC atomic 3 宣言削除→契約コメント移設 |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | 1 | syncStateFrom 削除（-42 行） |
| `src/ConvolverProcessor.h` | 1 | 宣言+注記削除（-5 行） |
| `tools/dead_code_callers_verifier.py` | 1 | FORBIDDEN_SYMBOLS / DORMANT_EDGE_GUARDS 改名 + 3 層意味論 docstring |
| `doc/work89/DESIGN_R4_D2D3_20260910.md` | 全 | v2→v2.3（承認経緯・実測証明・§6 実施完了記録・§7 承認記録） |
| `doc/work89/R4_REMEDIATION_REPORT_20260910.md` | 本報告書 | 新規 |
| `evidence/work89_r4_p1_*.log` / `work89_r4_p2_*.log` | 検証証跡 | ビルド/CTest ログ |

---

## 7. commit 手順（hunk 対応表で確定済み・実行可能）

`git add -p` の hunk 単位で Phase 分離が確定している（split `s` は不要・hunk 自体が Phase 単位で独立）。

**実測 hunk 対応表（HEAD 差分）**

| ファイル | hunk | 内容 | Phase |
|---|---|---|---|
| `src/eqprocessor/EQProcessor.Core.cpp` | `@@ -243,9` | resetToDefaults AGC publish 削除 | **2** |
| 〃 | `@@ -266,9` | reset() AGC publish 削除 | **2** |
| 〃 | `@@ -583,122` | sync 3 関数削除（W-1） | **1** |
| 〃 | `@@ -810,9` | prepareToPlay AGC publish 削除（W-3） | **2** |
| `src/eqprocessor/EQProcessor.h` | `@@ -375,12` | sync 3 宣言削除（W-1） | **1** |
| 〃 | `@@ -572,9` | AGC atomic 3 宣言削除→契約コメント化（W-3） | **2** |
| `src/ConvolverProcessor.h` | 全体 | 宣言+注記削除（W-2） | **1** |
| `src/convolver/ConvolverProcessor.StateAndUI.cpp` | 全体 | syncStateFrom 削除（W-2） | **1** |
| `tools/dead_code_callers_verifier.py` | 全体 | 意味論分離改名（W-4） | **1** |

**実行シーケンス**

```bash
# commit 1 — Phase 1 (D-3 + W-4)
git add src/ConvolverProcessor.h src/convolver/ConvolverProcessor.StateAndUI.cpp tools/dead_code_callers_verifier.py
git add -p src/eqprocessor/EQProcessor.Core.cpp   # @@ -583,122 のみ y、他 n
git add -p src/eqprocessor/EQProcessor.h          # @@ -375,12 のみ y、@@ -572,9 を n
git add doc/work89/DESIGN_R4_D2D3_20260910.md doc/work89/R4_REMEDIATION_REPORT_20260910.md
git commit -m "work89 R-4 Phase 1: D-3 dead-code sync functions removal (4) + verifier semantics split (FORBIDDEN_SYMBOLS/DORMANT_EDGE_GUARDS, P-V1 verified). Gates: build x2, CTest 40/40 x2, G4-S=0."

# commit 2 — Phase 2 (D-2)
git add src/eqprocessor/EQProcessor.Core.cpp src/eqprocessor/EQProcessor.h   # 残り hunk すべて
git commit -m "work89 R-4 Phase 2: D-2 AGC atomic (agcCurrentGain/EnvInput/EnvOutput) removal -> serial+RT-shadow single management. agcResetSerial fetchAdd x3 intact (AC-W3). Gates: build x2, CTest 40/40 x2, G4-B=0."
```

ConvoPeq.md・他文書変更（ARCHITECTURE.md 等）はスナップショット/文書同期の変更であり、上記 2 コミットとは独立にユーザー判断で分離する。

---

## 8. 参照

- `doc/work89/DESIGN_R4_D2D3_20260910.md`（v2.3 — 設計・証明・実施完了記録・承認記録）
- `doc/work89/REMEDIATION_PLAN_R123_20260910.md` §6/§7（前提: work89 R1/R2/R3 CLOSED・commit c0c0a789）
- `doc/work89/INTEGRATED-BUG-LIST.md` §10/§11/§15.4 D-2/D-3・§17.8
- `doc/work92/IMPLEMENTATION_REPORT_20260910.md` §B-7a/B-7b
- 証跡: `evidence/work89_r4_*`
- 本報告書（commit 時に §9 の判定を正式状態として記録）

---

## 9. 最終監査記録（2026-09-10・ユーザー再監査）

> **APPROVED — R-4 CLOSED / IMPLEMENTATION VERIFIED**（前回の authority mismatch HOLD は解除）

- 再監査者は**R-4 適用後の ConvoPeq.md（生成 2026-09-10 12:30:08 UTC = JST 21:30、本報告書 §0 の 21:11 版を包含する更新）**を直接取得し、`EQProcessor.h` の atomic 3 個が「work89 R-4 (D-2): … atomic を削除」コメントのみの実体不存在状態であることを確認。前回指摘の「旧 snapshot に対象 atomic が残存」は解消。
- 全 17 監査項目 PASS（D-2 / D-3 / P-V1 / W-4 / build / CTest / cppcheck / authority 再照合 / 設計逸脱なし）。
- 注意の分離: 「正常完了する prepareToPlay では serial を前進」（v2.3 表現）で確定済み — 「必ず前進」への一般化は不可。報告書・設計書とも該当表現に修正済みであり追加修正要求なし。
- **残作業は §7 の Git commit のみ**（実装監査上の不合格事由ではない）。

