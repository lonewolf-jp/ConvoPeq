# ConvoPeq 残件 改修計画書 — v3.1 → v3.2 改訂（2026-09-22）

- **版**: v3.2（v3.1 の部分改訂・**R18-1〜R18-3** を新規確定）
- **前版**: `doc/work113/remediation_plan_20260922_v3.1_revised.md`（R17-1〜R17-7）— **v3.1 本文は参照文書として継承し、R18 と矛盾する箇所では本書を優先する**
- **本改訂の起点**: v3.1 外部監査（総合判定「妥当。Phase 0 は所定の承認を取得すれば GO」）のうち、**着手可能範囲として GO とされた項目の実装を本セッションで実施**した。本書は (a) 実装前に確定した未確定事項、(b) レビュー §10 の C1 実装範囲管理指摘の反映、(c) 実装着手の記録、を定める。
- **監査指摘の反映（v3.1 レビュー）**: 全 15 項目 PASS・Shadow 実装 GO・Phase 0 characterization GO（G-0/T-1/T-2/T-3/C0 承認条件付き）・production 側は全 HOLD — を原則どおり維持。

---

## 0. v3.1 → v3.2 改訂サマリ

| ID | 区分 | 内容 |
|----|------|------|
| **R18-1** | **C1 実装範囲の契約明確化（レビュー §10 の反映）** | (a) `src/tools/build_identity_gate.py` の変更は **test-only C1 変更として扱う**（production source ではないが BUILD-ID 出力に影響するため）。(b) **実際の commit 時には staged 一覧を必ず人間が確認してから commit** する（本実装では C1 commit を行わず staged 一覧を提示した段階で停止 — §2）。(c) **CMakeLists.txt 不変条件の限定化**: v3.1 §1.3 の「CMakeLists.txt 変更不可」と §2.9 項目 4「PolyphaseGainFidelityTests.cpp（新設・add_executable 40 番目）」は**自己矛盾**だった。R15-2 の趣旨は「production flag token を Phase 1 まで CMake に定義しない」であるため、C1 の CMakeLists.txt 変更は **test-only ターゲット（add_executable 40 番目）の登録に限定**し、**flag token の定義行・production target への変更は禁止**とする。flag token が CMakeLists.txt に現れた（コメント含む）場合は commit 不変条件違反 |
| **R18-2** | R17-3「同一翻訳単位」の実装可能形への精緻化 | bitwise 一致の前提は「同一 compiler configuration / 同一 ISA path / 同一 target 内の **別翻訳単位**」とする。production `CustomInputOversampler.cpp` と Shadow（`PolyphaseGainCandidateRef.h` を include する test TU）は **同一 test executable の別 TU** として compile され、同一 `/arch:AVX2` 等 compile option を共有する。v3.1 §2.9 の「同一翻訳単位に compile」の表現は、production TU と Shadow TU の分割が必然であるため「**同一 target・同一 compile option 内の別 TU**」に読み替える |
| **R18-3** | **外部 working-tree 変更の実測記録（本セッションで発見・未解決 → ユーザー判断待ち）** | 本セッション実行中に、**C1 作業者が意図して変更したものではない外部 working-tree 変化として観測された**（本セッション開始時には実在を確認済み・因果の特定は本記録だけでは第三者検証不能なため表現をこの形に限定）: (i) tracked `output_sourcecode_markdown.py` が削除（` D`・O-18 の未 commit 修正 +11/−4 が消滅・**2026-09-22 にユーザー指示で GitHub origin/main 準拠 + 最小限パス安全化により復旧済み — §1.6**）、(ii) tracked `scripts/fix_lint.py` `fix_tables.py` `fix_tables_v2.py` が削除、(iii) untracked の root 解析スクリプト群（`analyze_mcp.py` `find_all_mcp.py` `find_mcp.py` `find_mcp2.py` `inspect_mcp.py` `inspect_mcp_logs.py`）が消失、(iv) **`build/CMakeCache.txt`・`build-Release.ninja`・`build.ninja` が削除され、その後 `build.ninja` 系が復活**（並行作業中の configure 競合・vcvars 正環境の re-configure で解消）。**O-18 の +11/−4 修正内容はディスクから失われた**（ConvoPeq.md には 0 hit・stash なし・GitHub 上にも存在しないため**推測での再構築は行わない**）。確定はユーザーの判断とする。**ConvoPeq.md 本体（16:16:35 生成・5,359,772 B）は変更されていない**ため、authoritative snapshot の ID 自体は無傷 |

---

## 1. C1 実装（R18-1 の GO 範囲・本セッションで実施）

レビュー §13 の GO 範囲のうち **C1（Shadow Reference 完全実装 + REF-FIDELITY 基盤）** と **Step 0.0 維持確認** に着手した。**commit は行わない**（レビュー §10: staged 一覧の人間確認が前提・C0 承認はユーザー判断）。

### 1.1 実装ファイル

| ファイル | 区分 | 内容 |
|----------|------|------|
| `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | **上書き・test-only** | 骨格（102 行）→ 完全実装版: `prepareStageModel`（R16-1 の 10 手順を演算順序ごと独立実装）・`dotProductAvx2Ref` / `loadStride2Ref` / `dotProductDecimateAvx2Ref`（production SIMD 帰還構造と同一順序の独立実装・同一 target 内で bitwise 一致の受皿）・`interpolateStageRef` / `decimateStageRef`（production :492-723 と同一の観測契約・履歴 copy/guard/denorm/isBadSample 位置同期）・`PolyphaseGainShadow`（processUp/processDown 両方向・reset 契約 atomic 3 / clearAllStages atomic 1 相当・hardFallback/corruption auto-clear 契約）・candidate は runtime 切替（macro 既定） |
| `src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp` | **新規・test-only** | R17-4 (A)〜(E) 5 検証 + REF-FIDELITY 本体（シード固定 PRNG 3 ブロック × 2 preset × ratio {2,4,8} の up/down bitwise + reset contract + scalar↔scalar 3-tap 経路）+ candidate 2 系統一致性 assert + `[BUILD]` 6 要素ブロック出力。FAIL = fail-closed（exit 1） |
| `CMakeLists.txt` | **test-only 変更・R18-1** | 40 番目 `add_executable(PolyphaseGainFidelityTests …)` + AudioEngineHarness と同一の compile option（/arch:AVX2 等・R17-3 同一 compiler configuration）+ add_test。**flag token は一切定義しない（R15-2）** |
| `src/tools/build_identity_gate.py` | **test-only 変更・R18-1** | `--emit-build-id [--shadow-candidate 0/1]` モードを追加: 6 要素 `[BUILD]` ブロック出力（snapshot identity・git HEAD・working tree・production_flag 未定義確認・shadow_candidate・build config） |

### 1.2 実装した検証仕様（PolyphaseGainFidelityTests の test 一覧）

```text
[定数パス契約]
 T1  kPiRef == juce::MathConstants<double>::pi（bitwise・R16-1 同一定数パス）
 T2  kDenormThresholdRef == numeric_policy::kDenormThresholdAudioState（bitwise）
 T3  R16-3 candidate macro 2 系統一致性: macro 既定 ↔ runtime {false,true} の DC 予測一致
     （base 0.75³ bitwise / cand 1.0 bitwise / centerPhaseGain 1.0↔2.0 bitwise）
[R17-4 追加 5 検証]
 T4  (A) coefficient invariant ×7 design（6 design + 3-tap synthetic）
     検査対象: taps / centerTap / centerParity / convParity / convCount /
     convCoeffs[]（reversed 一致含む）/ centerCoeff（bitwise 0.5）/ centerDelayInput /
     historyUpKeep / historyDownKeep / history sizes + FIRsum=1.0・convSum=0.5
     注: production Stage は private のため production 側係数一致は T9 本体 bitwise 比較で検証
 T5  (B) zero-input round-trip × 6 config（全出力 0 + prod↔shadow bitwise）
 T6  (C) constant-input DC: prod ↔ shadow(base) bitwise + base DC = 0.75^N ±1e-9
     + shadow cand DC = 1.0 ±1e-9（candidate は予測系・R12-8）
 T7  (D) impulse warm-start: S1(31/90) 単段 — h_rt[15] ≈ 0.25 ±1e-9・Σh ≈ 0.75 ±1e-9
     ・cand Σ ≈ 1.0 ±1e-9・prod↔shadow bitwise・warm-up δ@4096
 T8  〃 IIR3 r=8 多段 — Σh ≈ 0.421875 ±1e-9・peak ∈ [290,291]・prod↔shadow bitwise
 T9  (E) random block partition invariance × 6 config — partition {1,2,3,5,7,11,15,31,
     63,127,256,512,1024}+残・mixed {n/2,n/4,n/8,n/8}: one-shot == partitioned（up/dn 累積 bitwise）
[REF-FIDELITY 本体]
 T10 3 PRNG blocks × 2 preset × ratio {2,4,8} の up+dn bitwise + reset contract（atomic 3 相当
     の reset 後の同一刺激で再び bitwise 一致）
 T11 scalar↔scalar 経路 — 3-tap synthetic（convCount=2 → production/shadow 両側 scalar）
[HOLD 継続の明示]
     P0-I（SoftClip local OS）は Phase 0 0-5 で実施（G-0/C0 承認後・R17-1 契約）
     production centerValue *= 2.0 / CMake flag / default ON / calibration は HOLD 継続
```

### 1.3 ビルド・実行検証（**完了 — 本セッションで実測**）

```text
[configure]  vcvars64 (VS 18/Enterprise) → cmake -S . -B build -G "Ninja Multi-Config"
             -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl → Generating done
[build]      cmake --build build --config Release --target PolyphaseGainFidelityTests
             → build\Release\PolyphaseGainFidelityTests.exe 生成
[run]        build\Release\PolyphaseGainFidelityTests.exe → **PASS=31 FAIL=0・exit 0**
             （[BUILD] 6 要素ブロックは runner 出力と exe 出力の両方に自動出力）
[静的解析]   cppcheck 2.21.0: test TU で新規 error なし（constexpr aggregate の
             uninitMemberVarNoCtor 2 件は aggregate initializer 済みの誤検知）。
             clang-tidy 23.1.1: stages_[3] 境界の analyzer 警告に対し prepare() の
             clamp + processDown の fail-closed guard を追加 → **警告 0 件**
             （guard 加筆後の再実行で PASS=31 FAIL=0 を維持）。

[実装中に修正した自バグ（失敗→修正→再検証の記録）]
 (a) 初回実行で r=4/8 zero-input が "prod round-trip failed" → up バッファを
     2×block で確保していたため 4×/8× の up 出力で超過 → 8×block に修正。
 (b) S1 impulse の h_rt[15] 期待値を誤って 0.25 と記載 → R5-9 構造恒等式
     h_rt[15] = 2·(conv⋆conv)[15] + 0.25 ≈ 0.679897021291（δ 項のみが 0.25）に修正。
     修正後の実測: production 出力 = 期待値と bitwise 一致・Σ=0.75/cand Σ=1.0 厳密。
 (c) configure 再生成時に CMakeLists.txt:576 の
     "PriorityIntegrationTests → juce::juce_core not found" を 1 回観測したが、
     外部 working-tree/build-dir クリーンアップとの競合による過渡状態で、
     vcvars 正環境での re-configure で解消（再発なし・未確定扱いを解除）。
```

### 1.4 再実行コマンド（ユーザー環境で再現する場合）

```bash
build.bat Release                                        # 通常の vcvars 経由ビルド（gate 同梱）
cmake --build build --config Release --target PolyphaseGainFidelityTests
build\Release\PolyphaseGainFidelityTests.exe > log.txt
# 期待: [BUILD] 6 要素ブロック → 全 test [PASS] → summary PASS=31 FAIL=0・exit 0
# FAIL 時は REF-FIDELITY triage（R17-3 の 5 分類）で原因確定まで進めること。
```

### 1.5 検証 log

`tmp/c1_ref_fidelity_run_20260922.txt`（[BUILD] 6 要素 + 全 31 判定 + summary 保存済み）。

### 1.6 R18-3 復旧記録（2026-09-22・ユーザー指示で実施）

`output_sourcecode_markdown.py` を **GitHub（origin/main = `06794615`）から取得して復旧**した。

```text
取得元   : https://github.com/lonewolf-jp/ConvoPeq.git（git fetch origin main →
           FETCH_HEAD:output_sourcecode_markdown.py = 207 行）
取得版への変更（最小限・ Mimosa 安全審査の指摘対応）:
  - ensure_within_root(path, root_path): すべてのパス境界（CLI 引数・iterdir/rglob
    の列挙結果）でルート配下 containment を検証する単一関数を新設（逸脱は ValueError）
  - resolve_safe_output_path(root_path, output_file): 出力先は「相対パスかつ
    ルート直下」に限定して解決（絶対パス・上位ディレクトリ参照を拒否）
  - 列挙を os.walk/os.path.join から pathlib（iterdir/rglob）に置換
    （対象集合は is_subpath_of_target + IGNORE_EXTS で combine と iter_target_files
     の同一性を維持・--check の取りこぼしは発生しない）
検証:
  - SYNTAX OK（ast.parse）
  - python output_sourcecode_markdown.py --check → 正常動作を確認
    baseline Generated : 2026-09-21 16:16:35 / NEWER_SRC_COUNT=4 / STATUS: STALE
    （STALE の 4 件 = C1 変更ファイルそのもの — CMakeLists.txt / Ref.h /
      PolyphaseGainFidelityTests.cpp / build_identity_gate.py。
      これは O-12 の正しい手順: C1 commit 後に再生成して FRESH へ収束させる）
注意:
  - 消滅していた +11/−4 の未 commit 修正内容は復元できていない（GitHub 上にも存在しない）。
    本復旧は「GitHub の committed 最新版 + 最小限のパス安全化」であり、
    O-18 の内容確定はユーザー側の記録（別環境・エディタ履歴等）に依存する。
  - ConvoPeq.md 本体（16:16:35・5,359,772 B）は変更されていない（無傷）。

---

## 2. 実装後の commit 契約（R18-1 反映・本セッションでは実行しない）

```bash
# C1 の commit は人間の staged 確認を必須とする（レビュー §10・外部監査 §19）:
git add src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h
git add src/tests/AudioEngineHarness/PolyphaseGainFidelityTests.cpp
git add src/tools/build_identity_gate.py
git add CMakeLists.txt
git add output_sourcecode_markdown.py          # R18-3 復旧（GitHub 準拠 + パス安全化）
git add doc/work113/remediation_plan_20260922_v3.2_revised.md
git add doc/work113/implementation_report_20260922_c1_ref_fidelity.md

# 人間確認事項（R18-1 (b)・staged の実態は git diff --cached で確認する —
#   git status --short の ?? は untracked を示すため staged 証跡には使わない）:
#   1. git diff --cached --name-status → C1 の 7 ファイルのみ（production
#      src/*.cpp|h（tests 除く）を含まない）
#   2. git diff --cached -- CMakeLists.txt | grep -c CONVOPEQ_CORRECT_POLYPHASE_GAIN
#      → 0（flag token の定義・コメント・option・compile definition のいずれにも無い）
#      （コメントに token が現れる場合も commit 前に除去する）
#   3. 上記を確認してから commit -m "test(work113): C1 PolyphaseGainFidelityTests
#        REF-FIDELITY foundation (R15-2/R16-1/R17-1..4/R18-1)"

# C1-PRECOMMIT GATE（外部監査 §19・実測結果は implementation_report §5b）:
#   [1] git index / [2] production src = 0 / [3] CMake flag token = 0 /
#   [4] BUILD-ID 6 要素 / [5] PASS=31 FAIL=0 exit 0 — 全 PASS で commit 可
```

**本セッションでの停止点**: staged 一覧（実 index・7 ファイル）を確定し、C1-PRECOMMIT GATE 5/5 PASS を記録した段階で停止。commit はユーザーが `git diff --cached` の実態を確認した後に行う（レビュー §10 の指示どおり）。

---

## 3. 承認状態と HOLD の維持（v3.1 §11 継承 + 本版更新）

| 項目 | 状態 |
|------|------|
| G-0 / T-1 / T-2 / T-3 | **ユーザー承認待ち**（確定案提示済み） |
| C0（evidence/ 追跡外化） | **ユーザー承認待ち**（独立 commit・R17-2） |
| Phase 0 characterization（0-1 〜 0-6） | **G-0/C0 承認後に着手** |
| production `centerValue *= 2.0` | **HOLD**（本セッションで変更 0・本セッションで実装したものは test-only のみ） |
| CMake flag token / default ON / calibration | **HOLD**（CMakeLists.txt に flag 定義なし・本セッションで検証済み） |
| 案 E の最終採用 | **HOLD**（Phase 0 全 PASS + 条件D + ユーザー GO 後） |
| F-3 / F-4 との混在 | **禁止**（分離継続） |
| R18-3（外部 working-tree 変更） | **ユーザー判断待ち**（O-18 の +11/−4 修正内容の復元・確定） |

## 4. 判定

- **C1 のソースレベル実装: 完了**（R18-1〜R18-3 の契約のとおり・production 変更 0）。
- **REF-FIDELITY の実測: PASS（31/31・exit 0 — §1.3 のとおり・実ログ `tmp/c1_ref_fidelity_run_20260922.txt` で裏付け）**。ただし本 exe は **Shadow fidelity gate** であり production 正しさの gate ではない（R12-8）— 「production == shadow」から「production が数学的に正しい」は導かれず、T4〜T8 の独立した数学的制約が共通バグリスクを低減する位置づけ。P0-A〜P0-I の characterization は別物であり、C1 PASS ≠ B-1 PASS ≠ 案E採用。
- **次の着手境界**: C1-PRECOMMIT GATE（§2 の 5 項目・全 PASS）→ 人間の staged 確認 → C1 commit → G-0/C0 承認 → Phase 0 characterization（0-1 〜 0-6）。
