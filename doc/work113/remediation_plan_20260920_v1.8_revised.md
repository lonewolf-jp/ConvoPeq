# ConvoPeq 残件 改修計画書（2026-09-20 時点・再改訂 v1.8）

- **版**: v1.8（v1.7 全項目の追加ソース監査 + 閉形式モデル（C++ 厳密移植・自己検証済み）による全面再検証を反映）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/` の未 commit 差分 0 件（test-only 2 ファイルを除く・本日再実測）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py`（`prepareStage`/`interpolateStage`/`decimateStage` を C++ から厳密移植。ループ実装との等価性を ≤2.2e-15 で自己検証済み。全測定出力: `model_polyphase_20260920_results.txt`）
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「有力仮説」として維持（閉形式モデルで DC round-trip = 1.0 を再確認）。**ただし v1.7 §2.7.3 の candidate image rejection 契約は本セッションで不成立が確定したため P0-E を再定義（R3-1）**
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v1.7 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする。C++ 分岐は **`#if` を使用（`#ifdef` は禁止）**

---

## v1.7 → v1.8 改訂サマリ（本セッションの独立再検証による修正）

v1.7 の全行番号（30 件）・全コード解釈を再監査した結果 **全件一致**（行番号誤り 0）。あわせて閉形式モデルにより v1.7 §2.7 の全数値を独立再現した。その結果、**v1.7 の image rejection 契約（§2.7.3・P0-E）に重大な誤りがあることが確定した**ため P0-E を再定義する。ほかに数値 2 件（IIR3 の 0.49 差分偏差・IIR3/LP3 の ripple）と集計 1 件（R-2 の箇所数）を修正し、§1 に新規差分 2 件（O-8/O-9）を追加する。

| ID | 区分 | 内容 | v1.8 での確定 |
|----|------|------|----------------|
| **R3-1** | **必須（契約改訂）** | v1.7 §2.7.3 の **candidate image rejection −88.9〜−104.9 dB は再現不能**。厳密移植モデルでの実測: up 単段出力（D1 定義）では candidate は **−35.9〜+16.8 dB**（f̂・design 依存の部分打ち消し）。かつ **round-trip 出力（D2/D3 定義）では baseline==candidate が厳密成立**（round-trip は LTI 合成であり、両分岐一様の ×2 は伝達関数に一様係数 (4/3)^N を掛けるだけで鏡像比を変えない — 理論帰結 + 実測一致 ≤0.02 dB） | **P0-E を再定義**（§2.6.1・§2.7.3）。出力側 gate は「base==cand 不変性（±0.02 dB）」、D1 は記録のみ。「candidate ≥ 80 dB」gate は廃止（定義上達成不能） |
| **R3-2** | 根拠修正 | E3「baseline in-band image rejection = −9.54 dB 一定」は**不正確**: D1 実測は design/f̂ 依存（31/90・511/140 は f̂ ≤ 0.2 で −9.7〜−10.5 dB ≈ 理論漸近値 −9.5424 dB = 20·log10(1/3) ✓。だが **127/110 は −8.1〜+5.8 dB**（f̂ = 0.2 で鏡像が信号を上回る）・1023/160 は −8.1〜−16.7 dB） | defect の**主証拠を DC round-trip = 0.75^N（12 桁実測・全構成）に確定**し、D1 非対称（最悪 +5.8 dB）を**補助証拠**とする。G-BL sanity gate から image 項を外す（§2.6.1）。**なお出力への伝播は存在しない**: D1 鏡像は同ステージ down 段 stopband（−87〜−179 dB）で抑制され round-trip 出力には届かない（D2/D3 で base==cand 厳密一致・R3-1） |
| **R3-3** | 数値契約修正 | **P0-C の gate 0.05 dB は v1.7 の帯域定義では FAIL し得る**: 密グリッド exact-DTFT 実測 ripple [0.005, 0.45] = IIR3 **0.111/0.084 dB**・LP3 **0.053/0.039 dB**（v1.7 記載 0.001 dB はスパース測定点評価に由来）。0.45 近傍は遷移 knee の単調落下であり ripple ではない | **P0-C の ripple 判定帯域を [0.005, 0.30] Fs_in に統一**（candidate 実測: S1 0.001 / IIR3 0.025 / LP3 0.012 dB → gate 0.05 dB PASS）。[0.30, 0.1 dB edge] は遷移落下として記録のみ。参考: [0.005, 0.75×edge] 定義では S1 0.0009 / IIR3 0.0358 / LP3 0.0174 dB |
| **R3-4** | 数値修正 | IIR3 の (4/3)^N 差分偏差 @0.49 Fs_in: v1.6/v1.7 記載 **+0.043 dB** は本モデルで**再現不能**（本モデル実測 **+0.0062 dB**: base −7.6484 / cand −0.1459 dB）。S1 の +3.3927 dB と per-design 表（§2.7.2）は全点一致のため、差異は IIR3 の 0.49 近傍評価に限定的 | 0.49 Fs_in は gate 外（記録のみ）のため影響なし。Phase 0-3 [A] 実測で確定させる（要確認事項として §12 に記載） |
| **R3-5** | 集計修正 | §5 R-2 の「実測 12 箇所」の内訳を確定 | **変換実体 9**（直接 7: `std::stod`×4 :1582/:1592/:1614/:1615・`std::stoi`×2 :1583/:1591・`std::stof`×1 :1590 ＋ lambda 本体 2: `std::stoi` :1562/:1567）・**throw 可能経路 12**（直接 7 ＋ lambda 経由呼出 5: `parseHcIdx` :1606/:1608/:1610・`parseLcIdx` :1607/:1611）。v1.7 の「12 箇所」は**経路数として妥当**（内訳確定）。lambda 本体 2 箇所の try/catch 化で 12 経路すべて fail-closed 化|
| **R3-6** | 定義統一 | 「up/down 利得」の記述が 2 文書で矛盾（`residual_tasks_20260920.md`: up 1.0/down 0.75 vs v1.7: up 0.75/down 1.0） | **両者は同一事実の別測定定義**と確定: **DC/平均基準**（up 出力平均 = 0.75 / down 単体 DC = 1.0 / round-trip = 0.75）と **peak 基準**（`[OS_DIRECT]` の `upPeak` = even 位置 1.0×入力 → up=1.0・round-trip 出力 peak 比 = 0.75）。本書では **DC 基準を正**とし、`upPeak` は「even 位置透過率 1.0 の証拠」として注記 |
| **R3-7** | 棚卸し追加 | v1.7 §1 が扱っていない未 commit 差分 2 件を検出 | **O-8**: `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` が **tracked かつ Modified**（Bin 16645→16895 バイト・同 `.pyc` 2 件が `git ls-files` で tracked 実測）。**O-9**: `docs/tool-inventory-2026-09-20.md`（untracked・環境目録） |
| **R3-8** | 実装仕様精密化 | Phase 1 の CMake option 導入方式を確定 | `#if` セマンティクス（値 0/1 を明示定義）+ 配置確定（§2.5）。`build.bat` の `-D` 経路で切替可能であることも実測確認 |
| **R3-9** | 実装詳細確定 | 案 E の挿入位置を FP 厳密に確定 | `centerValue *= 2.0` は **既存 `convValue *= 2.0`（:557）と同位置・:558 の denormal clamp より前**に挿入。Shadow Reference と bitwise 一致させるため挿入順序を固定（clamp 後 ×2 は kDenormThreshold/2 < v < kDenormThreshold の挙動が変わる） |
| **R3-10** | 監査確定 | v1.7 §10 の全行番号を本日再実測 | **全件一致**（R2-9/R2-10 の v1.6 誤り修正も含め、新規の行番号誤り 0）。詳細 §10・§12 |
| **R3-11** | 確認確定 | 案 E と Latency static_assert（Latency.cpp:6-8）の整合 | **taps・群遅延不変のため案 E で latency は不変**。static_assert（「identical up/down taps」）も **係数列不変により成立し続ける**。P0-F の impulse 実測 gate は維持 |
| **R3-12** | 記録強化 | 姉妹実装 `TruePeakDetector` の構造を詳細確定 | `interpolateStage`（:301-311）は **polyphase 分解ではなく全 FIR を stride-1 で両位相評価**（even = center + conv / odd = center + conv・**×2 補償なしで up DC 1.0**）。「案 E の正解としない」（計測専用・約 2× MAC コスト）は維持するが、**unity DC と真のフィルタリングを両立する別構造の在庫**として記録価値を明記（将来の再設計議論の参考） |
| **R3-13** | ツール記録 | 静的解析・ビルド実測 | cppcheck（C++20・warning/performance/portability）を `CustomInputOversampler.cpp` に再実行 → **指摘 0 件（再確認）**。`build\Release\AudioEngineHarness.exe` 実在（41,137,664 bytes・2026-09-20 14:25 再確認） |

**v1.7 から変更しない点（再確認）**: 案 E の技術的内容（`centerValue *= 2.0` 1 行）/ Phase 0 の 5 構造（Baseline・G-0・REF-FIDELITY・Candidate・Differential）/ 段階リリース骨格 Phase 0→1→2→3-A/B1/B2/C/D→4 / §1 O-1〜O-7 の推奨 / §3 F-2 案 A・F-3 案 D・F-4 案 B / §4 MIGRATE CASES THEN RETIRE / §6 実行順序 / §8 ロールバック。

---

## 0. 凡例と全体戦略（v1.7 から変更なし）

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | B-1 | 全オーディオ経路 | flag OFF rebuild |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only（F-3 は production 欠陥の記録） | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1〜O-9, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: v1.7 §0 のまま（§1 意思決定 → §2 B-1 段階リリース → §3 → §4 → §5）。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-9）

### 1.1〜1.7 O-1〜O-7: **v1.7 §1.1〜§1.8 のまま（本日再実測で全件確認）**
- O-1: 差分 +634/−2（BassBuzzMeasurement.cpp）+8/−0（PublishPipelineIntegrationTests.cpp）・窓 :2042-2044・`[OS_DIRECT] roundTripGain=` :1367・`runOversamplerDirect()` :1293 ← `runEqDirectDriveAttribution()` :1526/:1532 ← PPIT main :1085 から（前方宣言 :1116・呼出し :1223）。**推奨案 A**
- O-2: 台帳更新は O-1 と独立 commit
- O-3: `ConvoPeq.md` は commit しない（再生成: `python output_sourcecode_markdown.py`）
- O-4: `Testing/Temporary/CTestCostData.txt` は現状維持
- O-5: `.opencode/opencode.json` は触らない
- O-6: push（ahead 2 / behind 0）はユーザー手動
- O-7: `AGENTS.md` は触らない

### 1.8 O-8（v1.8 新設）: `tools/__pycache__/*.pyc` — tracked かつ 1 件 Modified
- 実測: `apply-solidlsp-bash-ls-patch.cpython-314.pyc` が ` M`（Bin 16645 → 16895 バイト）。`apply-...pyc` と `retire_authority_verifier.cpython-314.pyc` の **2 件が tracked**（`git ls-files tools/__pycache__/` 実測）
- `.pyc` は再生成物であり tracked 運用自体が望ましくない。ただし履歴操作はユーザー判断事項
- **推奨**: 本 Modified 差分は commit しない（`git checkout -- tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` で復元可）。将来 work item として `tools/__pycache__/` を `.gitignore` 追加 + `git rm --cached`（HOLD・履歴書き換えなし）

### 1.9 O-9（v1.8 新設）: `docs/tool-inventory-2026-09-20.md`（untracked）
- 環境ツール目録文書（`docs/` 配置）。**触らない**（ユーザー判断）。commit 可否はユーザー判断。

### 1.10（参考・v1.8 補足）: `doc/work113/` 計画書 7 件（untracked）
- `remediation_plan_20260920.md`（1016 行）/ `_v1.6_revised.md`（917）/ `_v1.7_revised.md`（903）/ 本書 / `renew_plan.md`（2502）/ `renew_plan_verification_20260920.md`（385）/ `residual_tasks_20260920.md`（187）
- 前例の work113 文書は tracked。**推奨**: 本書承認確定後、承認版 + 台帳を 1 commit として登録（O-2 と同梱可）。`.cline/`（untracked）は触らない。

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定している事実（v1.7 §2.1 を継承・本セッションで独立再確認）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     DC round-trip 0.750000（up 出力平均 0.75 / down 単体 DC 1.0 — DC 基準・R3-6）
            peak 基準では upPeak = 1.0×入力（even 位置）・round-trip 出力 peak 比 0.75
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0, internalMaxBlock)・DSPCoreLifecycle.cpp:188/261 実測）
数学的導出  up: even=1.0（conv×2）+ odd=0.5（center）→ 平均 0.75 / down: 0.5+0.5 = 1.0
engine fit  0.98379 × 0.75^max(log2 effOS, 1) は doc の経験式。src/ に 0.98379 / effOS 定数は不在（本日 grep 再確認）
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾（係数列自体は不変 → R3-11）
production 変更 0 / commit 0
```

**閉形式モデルによる独立再確認**（`model_polyphase_20260920.py`・アルゴリズム厳密シミュレーション）:

| 構成 | baseline DC | candidate DC | 0.75^N |
|------|-------------|--------------|--------|
| 単段 31/90 | 0.750000000000 | 1.000000000000 | 0.75 |
| IIRLike 3 段（511/127/31） | 0.421875000000 | 1.000000000000 | 0.75³ = 0.421875 |
| LinearPhase 3 段（1023/255/63） | 0.421875000000 | 1.000000000000 | 0.75³ = 0.421875 |

### 2.2 根本原因の特定（v1.7 §2.2 を継承・全行番号本日再実測で一致）

v1.7 §2.2.1〜§2.2.4 のコード追跡・parity 実測・round-trip 整合式は **全て本日再実測で一致**（`prepareStage` :287-390・`interpolateStage` :492-568（`convValue *= 2.0` :557）・`decimateStage` :570-723（`output[n] = acc` :717））。taps/attenuation 対応表（:84-106）も一致。変更なし。

**本日追加で確定した実装事実**（Phase 0 設計への反映）:
- `processUp`（:725-783）は **stages[0]（511/1023 taps）を入力レートで最初に適用**し、以降 2 倍ずつ（:764-776・`currSamples <<= 1`）。閉形式モデルの段順序と一致 ✓
- `processUp` は `inSamples > maxInputBlockSize` で **空ブロックを返す**（:743-748・Phase 0-4 partition 前提の根拠）✓
- `processDown`（:785-）も容量超過で `markCorruptionDetected()` + 出力クリア（:826-830）。さらに **hardFallback 経路**（:728-738/:789-806）では up 入力をそのまま透過する — Phase 0 測定系は corruption/hardFallback フラグを監視し、透過が混入していないことを確認する必要がある（v1.8 追記）
- `prepareSingleStage(31, 90.0, internalMaxBlock)` は diagLog 版（:188）と非 diag 版（:261）の **2 呼出箇所**があるが同一関数経路 → 案 E の 1 行修正で両者同時に修正される ✓

### 2.3 修正案の比較と DESIGN-CONTRACT-A（v1.7 §2.3 を継承・E3 のみ R3-2 により修正）

案 A〜E の比較（A/B/C/D 不採用・**E 有力仮説**）は **v1.7 §2.3 のまま**。

#### 2.3.1 案 E 採用根拠（v1.8 版・G-0 として明示承認する対象）

**DESIGN-CONTRACT-A（v1.8）**:
> Oversampler は interpolation convention として両 polyphase 位相が同一 DC gain（= 2 倍密化の gain convention）を持ち、up/down round-trip の DC/passband gain が unity であることが要求される。

| 証拠 | 内容 | v1.8 での状態 |
|------|------|----------------|
| **E1** | コード自身が既に gain-2 convention（`convValue *= 2.0` — :557 実測）。欠陥は convention ではなく center 位相への不完全適用 | 変更なし・確定 |
| **E2** | down 側 half-band decimator は既に DC gain 1.0（0.5+0.5）。up だけ 0.75 → up/down 非対称は `isSymmetricUpDown` / static_assert の契約前提と矛盾。**なお static_assert は「taps 列の同一性」の宣言であり、案 E は taps・遅延を変更しないため assert は成立し続ける（R3-11）** | 精緻化 |
| **E3（修正・R3-2）** | D1（up 単段出力）で baseline 鏡像比が **f̂・design 依存で −9.7〜+5.8 dB**（理論漸近 −9.5424 dB = 20·log10(1/3)・127/110 @0.2 では +5.8 dB で鏡像が信号超え）。FIR 設計減衰 −87〜−159 dB に対し最大 ~165 dB の開き = **up 分岐非対称（gain convention 欠陥）の客観的な構造証拠**。**ただし D1 鏡像は同ステージ down 段 stopband で抑制され round-trip 出力には伝播しない（D2/D3 で base==cand 厳密一致・R3-1）** | **defect の主証拠を DC round-trip = 0.75^N（12 桁実測）に確定。D1 非対称（最悪 +5.8 dB）は補助証拠** |
| **E4** | `isLinearPhaseFIR = true` / `isSymmetricUpDown = true`（h:21-22）+ Latency.cpp:6-8 static_assert | 変更なし・確定 |
| **E5** | JUCE `dsp::Oversampling` は up 経路で `buf[N−1] = 2·samples[i]`（両位相に同一 ×2）・down は ×1（juce_Oversampling.cpp:185-196 / 228-240 実測✓） | 変更なし・確定 |

**案 E の実装（R3-9 で確定）**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E（両 polyphase 位相への ×2 対称適用）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;   // :558（clamp より前に ×2）
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

**gain 変化**: candidate/baseline 比 = (4/3)^N（1 段 +2.4988 dB / 2 段 +4.9977 dB / 3 段 +7.4963 dB）— 閉形式モデル + 実測一致。

### 2.4 設計時に検討すべき 8 観点・2.5 段階リリース設計

- 8 観点: **v1.7 §2.4 のまま**（全項 Phase 0 で確認）。
- 段階リリース: **v1.7 §2.5 のまま**（Phase 0 → 1 → 2 → 3-A/B1/B2/C/D → 4・baseline record-only・calibration は 3-B1/B2 完了後の 3-C/D）。**Phase 1 の CMake 実装は R3-8 により精密化**:

```cmake
# CMakeLists.txt 冒頭 option 群（:40 近傍）に追加:
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
       "Correct polyphase gain convention (B-1 案E)" OFF)

# add_subdirectory(JUCE)（:1043）の後・juce_add_gui_app（:1062）の前に追加:
add_compile_definitions(
    CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
```

- **分岐セマンティクス（確定）**: C++ 側は必ず **`#if CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を使用（`#ifdef` 禁止 — 値 0 でも defined になるため）。`$<BOOL:...>` genex は `add_compile_definitions` で使用可（CMake 3.12 以降）。
- 適用範囲: directory-scope compile definition は呼出以降に作成される target（ConvoPeq :1062/:1282・AudioEngineHarness :1899・CONVOPEQ_ALL_SOURCES :1133 派生）に適用。JUCE への無用な伝播は `add_subdirectory(JUCE)` より後の配置で回避。**前例**: `NUC_DEBUG_GUARDS`（CMakeLists.txt:52-58・`add_compile_definitions`）— 同一パターンの既存使用あり。
- `build.bat` の `-D` 経路（CMAKE_EXTRA_FLAGS・実測確認済み）で `build.bat Release nopause -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` のように切替可能。
- in-source は safety net のみ。既定 OFF（既存挙動維持）。Runtime flag は不採用。

### 2.6 Phase 0 characterization（P0-E を R3-1 により再定義・他は v1.7 §2.6 を維持）

```
Phase 0-0  source/contract audit: DESIGN-CONTRACT-A（v1.8 版・E1〜E5）を明示承認（G-0・承認者はユーザー）
Phase 0-1  P0-BL Baseline（record-only）: DC round-trip = 0.75/0.5625/0.421875（ratio 2/4/8・preset 非依存・±1e-6）
Phase 0-1b REF-FIDELITY gate（R2-2 維持）: Shadow Reference（baseline モード）== production
           （DC/impulse/周波数/ratio 2/4/8 × preset 2 種/partition 4 種/reset・double 経路 bitwise 一致・
            不可なら相対誤差 ≤1e-15。不一致なら reference を修正し一致まで Phase 0-2 に進まない）
           配置: src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h（header-only・production 変更 0）
Phase 0-2  P0-CAND Shadow Candidate E: DC round-trip = 1.0 ± 1e-6（全構成）— 閉形式モデルで予備確認済み
Phase 0-3  frequency transfer（stage-rate 軸・§2.7 の軸定義に従う）
Phase 0-4  block/reset characterization（R2-3 維持）: partition invariance bitwise（4096/1024×4/256×16/ragged・
           各ブロック長 ≤ maxInputBlockSize — processUp :743-748 の空ブロック返却 guard 実測確認済み）
           + reset contract bitwise（fresh+block#1 == stream→reset()→block#1・reset :452-467 は
           upHistory/downHistory clear 実測確認済み）
Phase 0-5  SoftClip local OS characterization（R2 維持・prepareSingleStage(31, 90.0, internalMaxBlock)
           は DSPCoreLifecycle.cpp:188/261 実測確認済み）
Phase 0-6  float-host / double-host equivalence（R2-4 数値契約のまま: maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8）
```

**Phase 0-3 の測定定義（v1.8 確定 — image rejection の 3 定義を明確化・R3-1）**:
- **D1（up 単段出力・2× rate）**: tone f̂ に対する 0.5−f̂ 鏡像成分比。**記録のみ**（gate に使わない）
- **D2（単段 round-trip 出力）**: |H_rt(0.5−f̂)|/|H_rt(f̂)|。round-trip は LTI 合成 → 鏡像比 = 伝達関数比 → **base==cand 不変（理論帰結）**
- **D3（full-chain 出力の worst in-band spur）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30}。**base==cand 不変（理論帰結）**
- 測定系要件（v1.8 追加）: 過渡トリム（全 stage taps + 512 sample 以上）+ Hann 窓を必須化（未トリム矩形窓では −79〜−90 dB の偽フロアが出ることを本モデルで実測確認）。corruption/hardFallback フラグ監視を必須化（hardFallback 透過経路 :728-738/:789-806 の混入排除）

**Phase 0-3 の残り（周波数応答）は v1.7 §2.6 Phase 0-3 [A]/[B]/[C]/[D] のまま**。ただし [D] passband ripple の判定帯域は R3-3 により **[0.005, 0.30] Fs_in（dense 評価）** に変更、[0.30, edge] は記録のみ。

#### 2.6.1 B-1-P0 GATE（v1.8・P0-C/P0-C'/P0-E を数値更新）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT（絶対値）** と **Candidate vs Baseline（差分帰属）** の 2 系統。

| ID | 判定対象 | 判定条件（v1.8 確定値） | 合否 |
|----|----------|------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が明示承認済み（E1〜E5・v1.8 版添付） | □ PASS / □ FAIL |
| **G-BL** | Baseline sanity | **DC round-trip = 0.75^N（±1e-6）のみを gate 条件とする**（R3-2）。D1 鏡像比（−9.5 dB 系/+5.8 dB を含む）は記録のみ | □ PASS / □ FAIL |
| **REF-FIDELITY** | Phase 0-1b | Shadow Reference（baseline モード）== production: 全項目 bitwise 一致（不可なら相対誤差 ≤ 1e-15） | □ PASS / □ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成） | □ PASS / □ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB**（実測 −0.0000〜−0.0078 dB） | □ PASS / □ FAIL |
| **P0-C** | Passband ripple | **dense 評価（exact DTFT または ≥2^18 点 FFT）で [0.005, 0.30] Fs_in の max−min ≤ 0.05 dB**（candidate 実測: S1 0.001 / IIR3 0.025 / LP3 0.012 dB）。[0.30, 0.1 dB edge] は遷移落下として記録のみ（IIR3 0.111 dB は落下を含む値・ripple ではない） | □ PASS / □ FAIL |
| **P0-C'** | Differential（帰属） | Candidate/Baseline 比 = (4/3)^N ± **0.05 dB**。**適用域 per-config（R2-7 維持・数値更新 R3-4）**: IIR3/LP3 は f ≤ 0.45 Fs_in（実測偏差: IIR3 +0.0121 / LP3 −0.0025 dB）、S1 は f ≤ 0.30（実測 +0.0000〜+0.0002 dB）。0.49 Fs_in は記録のみ（実測: IIR3 +0.0062 / LP3 +0.0025 / S1 +3.3927 dB・v1.7 の IIR3 +0.043 は非再現）。Phase 2 flag-ON 測定が shadow 値と ±0.02 dB で一致すること | □ PASS / □ FAIL |
| **P0-D** | Stopband（stage-local・絶対 + 差分） | (D-1 差分) decimateStage は係数・コードとも不変 → Candidate/Baseline 差 = 0（FP 誤差 ≤1e-12）。(D-2 絶対) alias leakage ≤ **−(A_stage − 10) dB** を **[transition_end, 0.5] cycles/sample** で満たす（per-design 表 §2.7.2 は **本セッションで全点独立再現・確定**）。transition 領域は記録のみ | □ PASS / □ FAIL |
| **P0-E（再定義・R3-1）** | Image rejection | (E-1 記録) D1 baseline 鏡像比を全 design × f̂ 7 点で記録（理論漸近 −9.5424 dB・実測 −9.7〜+5.8 dB・f̂ > 0.25 で鏡像優勢は構造帰結）。(E-2 **gate**) D2/D3 の **base==cand 不変性**: 全測定点で差 ≤ **0.02 dB**（案 E が出力側 image を悪化させないことの確認。**理論上は LTI 合成から厳密成立 = 恒等式レベル**。実測は矩形 FFT の量子化差で ≤0.01 dB（例: 1023/160 @0.05 で −150.23/−150.22 = FFT 測定誤差・不正ではない）。D2 絶対値（−51〜−185 dB）は round-trip 伝達関数の stopband 比であり案 E の前後で同一。**旧 gate「candidate ≥ 80 dB」は廃止**（R3-1: D1 定義でも到達不能・D2/D3 では base と同一のため無意味） | □ PASS / □ FAIL |
| **P0-F** | Latency | impulse 応答 peak 位置 = `Σ (taps[s]−1) × (baseRate/stageRate)`（±1 sample）+ centroid cross-check（±0.05 sample 目安）。案 E で不変（R3-11） | □ PASS / □ FAIL |
| **P0-G** | Float / Double host | same input / same partition / same reset state で: **maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8**（float 仮数 24 bit 由来の導出は v1.7 のまま） | □ PASS / □ FAIL |
| **P0-H** | Block / reset contract | partition A/B/C/D bitwise 一致（不可なら ≤1e-12）・`fresh+block#1 == stream→reset()→block#1`（bitwise） | □ PASS / □ FAIL |
| **P0-I** | SoftClip local OS | I-a: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, centerTap=15, centerParity=1, convParity=0 実測）で Candidate DC = 1.0 ± 1e-6。I-b: 動作点影響を upPeak/downPeak/閾値到達度で記録（閾値再校正は Phase 3-C） | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計、**または R3-12 記録の TruePeakDetector 型全 FIR 構造への変更**に方針変更する。

### 2.7 測定軸と実測基準値（v1.8 確定・authoritative = model_polyphase_20260920.py）

#### 2.7.0 周波数軸の正規化定義（v1.7 §2.7.0 を維持・確定）

```
own-rate cycles/sample: f̂ = f / Fs_stage（stage 自身の動作レート基準・有効域 [0, 0.5]）
Fs_in 基準（full chain）: f̂_in = f / Fs_in（入力帯域 [0, 0.5] のみ有効）
整合例（本モデルで再確認）: stage-0 (511 taps) の −0.1 dB edge = 0.2448 cycles/sample
 = 0.4896 Fs_in ≒ full-chain candidate edge 実測 0.4897 Fs_in ✓
```

#### 2.7.1 Full-chain 実測基準値（v1.8・dense exact-DTFT・baseline / candidate）

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC baseline / candidate | 0.750000 / 1.000000 | 0.421875 / 1.000000 | 0.421875 / 1.000000 |
| candidate \|H\| @50Hz〜0.25 Fs_in | −0.0000〜−0.0004 dB | −0.0078〜−0.0108 dB | +0.0025〜−0.0047 dB |
| (4/3)^N 差分偏差 @0.45 Fs_in | **+1.6350 dB**（適用外） | **+0.0121 dB** | **−0.0025 dB** |
| (4/3)^N 差分偏差 @0.49 Fs_in | +3.3927 dB | **+0.0062 dB**（v1.6/v1.7 の +0.043 は非再現・R3-4） | +0.0025 dB |
| ripple [0.005, 0.45] base / cand | 5.242 / 3.607 dB | 0.111 / 0.084 dB | 0.053 / 0.039 dB |
| **ripple [0.005, 0.30] base / cand**（P0-C 判定帯域） | 0.001 / 0.001 dB | 0.034 / **0.025 dB** | 0.016 / **0.012 dB** |
| 参考: ripple [0.005, 0.75×edge] cand | 0.0009 dB | 0.0358 dB | 0.0174 dB |
| candidate 0.1 dB passband edge | **0.3526 Fs_in** | **0.4897 Fs_in** | **0.4945 Fs_in** |

（S1 は v1.7 記載と全点一致。IIR3/LP3 の dev/ripple は測定密度依存のため v1.8 値を authoritative とする。
base の 0.1 dB edge は絶対値評価では意味を持たない（−2.5/−7.5 dB 系）ため candidate のみ掲載）

#### 2.7.2 FIR 設計の絶対基準値（P0-D floor の出典・**本セッションで全点独立再現 → 確定**）

own-rate cycles/sample・exact DTFT:

| design | −0.1 dB edge | transition_end（−A+3 dB） | leak 実測（本モデル） | v1.7 記載 | floor（= A−10 dB） |
|--------|-------------|---------------------------|----------------------|-----------|--------------------|
| 511/140 | 0.2448 ✓ | 0.2590 ✓ | 0.26:−141.9 / 0.30:−155.2 / 0.35:−160.0 / 0.45:−174.8 / 0.474:−165.4 | 同値 ✓ | −130 dB |
| 127/110 | 0.2317 ✓ | 0.2782 ✓ | −18.6 / −114.2 / −125.1 / −138.9 / −177.2 | 同値 ✓ | −100 dB |
| 31/90 | 0.1823 ✓ | 0.3452 ✓ | −8.45 / −25.4 / −96.2 / −107.7 / −90.6 | 同値 ✓ | −80 dB |
| 1023/160 | 0.2472 ✓ | 0.2552 ✓ | −167.5 / −179.1 / −206.4 / −186.5 / −188.4 | 同値 ✓ | −150 dB |
| 255/140 | 0.2396 ✓ | 0.2681 ✓ | −36.5 / −152.1 / −157.7 / −165.4 / −167.1 | 同値 ✓ | −130 dB |
| 63/120 | 0.2109 ✓ | 0.3130 ✓ | −10.7 / −59.3 / −130.1 / −125.6 / −138.3 | 同値 ✓ | −110 dB |

（0.26〜0.30 の低い値は transition 領域であり、P0-D の PASS 判定は transition_end 以降のみ。
v1.7「stopband min 実測 −138.9 dB 等」は遷移終端後の **最悪漏れ（max \|H\|）** 値であり、本モデルの
リーク点列と整合（最悪値は lobe 頂点のサンプリング差で ±3 dB 内）。「floor ✓」判定は全 design 成立）

#### 2.7.3 Image rejection 実測（v1.8 新設・3 定義確定・R3-1）

**D1（up 単段出力・2× rate・f̂ vs 0.5−f̂）— 記録のみ**:

| design | f̂=0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 |
|--------|---------|------|------|------|------|------|------|
| 31/90 base / cand | −9.75 / −35.94 | −10.49 / −27.79 | −9.97 / −10.41 | +11.51 / +13.04 | +8.15 / +23.19 | +5.27 / +11.61 | +2.47 / +4.86 |
| 127/110 base / cand | −8.12 / −12.60 | −3.06 / −3.60 | +5.83 / +4.57 | +6.84 / +15.20 | +9.53 / +14.18 | +10.34 / +14.99 | −3.49 / −2.19 |
| 511/140 base / cand | −9.75 / −35.93 | −10.49 / −27.78 | −8.64 / −10.39 | +9.84 / +12.93 | +8.09 / +22.82 | +4.81 / +10.40 | +1.23 / +2.44 |
| 63/120 base / cand | −9.12 / −21.15 | −8.11 / −15.44 | −6.17 / −13.00 | −8.12 / −14.70 | −12.71 / −5.94 | +5.19 / +3.12 | +8.92 / +16.80 |
| 255/140 base / cand | −9.96 / −21.73 | −5.41 / −11.91 | −4.06 / −8.23 | +10.50 / +10.01 | +10.06 / +28.14 | +2.63 / +4.16 | +14.93 / +7.02 |
| 1023/160 base / cand | −9.13 / −21.19 | −8.13 / −15.47 | −6.17 / −13.03 | −8.30 / −14.11 | −16.65 / −7.34 | +5.41 / +3.26 | +7.20 / +15.84 |

（baseline の理論漸近 −9.5424 dB は f̂ ≤ 0.2 の一部 design で近似成立。f̂ > 0.25 で鏡像優勢（正値）は
「desired 経路が FIR で減衰し center 分岐が無減衰」の構造帰結。candidate は低 f̂ で改善（部分打ち消し）・
高 f̂ で悪化。**−88.9〜−104.9 dB に相当する値はどの design/f̂ にも存在しない → v1.7 §2.7.3 は廃棄**）

**D2（単段 round-trip 出力・base==cand ≤ 0.01 dB・P0-E-2 gate 根拠）**:

| design | 0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 |
|--------|------|------|------|------|------|------|------|
| 31/90 | −165.5 | −143.0 | −102.9 | −97.0 | −114.3 | −125.8 | −137.6 |
| 127/110 | −171.9 | −150.0 | −65.7 | −86.9 | −97.9 | −74.0 | −118.9 |
| 511/140 | −139.7 | −117.9 | −78.2 | −70.1 | −88.0 | −99.8 | −113.4 |
| 63/120 | −185.7 | −144.3 | −108.8 | −90.0 | −115.4 | −114.8 | −106.1 |
| 255/140 | −162.0 | −173.1 | −90.7 | −109.0 | −79.2 | −96.2 | −109.7 |
| 1023/160 | −150.2 | −108.3 | −74.8 | −51.1 | −79.4 | −77.5 | −71.1 |

（D2 = \|H_rt(0.5−f̂)\|/|H_rt(f̂)\| = round-trip 伝達関数の stopband 比。案 E は一様 (4/3)^N 増幅のみで
比を変えない（理論帰結）。実測 base==cand は FP 誤差内（≤0.01 dB・1023/160 @0.05 で −150.23/−150.22 など））

**D3（full-chain 出力・base==cand 完全一致）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30} の全点で
mirror(0.5−f̂) = **−205〜−270 dB**（実質零・LTI 帰結）・worst spur = 測定窓 sidelobe −98〜−101 dB。
**base==cand 差 = 0.00 dB（全点）**

### 2.8 教訓・監査記録（v1.8・v1.7 §2.8 1〜12 を継承し 13〜15 を追加）

1〜12: **v1.7 §2.8 のまま**（oracle 依存禁止・gain convention と FIR 構造の分離・engine fit HOLD・Phase 0 先行・案 E=仮説だが defect 客観確定・perfect reconstruction 表現注意・anti-aliasing は実測確認・compile-time flag・姉妹実装扱い・validator error contract・own-rate 軸統一・shadow 参照 fidelity gate）。

13. **【v1.8 追加】測定定義の混用が evidence を崩す**: v1.7 §2.7.3 の「candidate −88.9〜−104.9 dB vs baseline −9.54 dB」は測定定義（up 単段内部か round-trip 出力か）を混用した結果の非対称比較であり、同一定義では base==cand（round-trip）または部分打ち消し（D1）しか存在しない。**gate は必ず「同一測定定義の base/cand 対 + 理論帰結」で構成する**
14. **【v1.8 追加】密グリッド評価を authoritative にする**: sparse 点評価（ripple 0.001 dB 等）は遷移落下と打ち消し項を見落とす。Phase 0-3 の全数値は dense（exact DTFT または ≥2^18 点）で取得し、判定帯域を明示する
15. **【v1.8 追加】真の代替構造は既に社内に存在**: `TruePeakDetector::interpolateStage`（:301-311）は polyphase 分解ではなく全 FIR を stride-1 で両位相評価し、**×2 補償なしで up DC = 1.0 + 両位相フルフィルタリング**を達成する（約 2× MAC コスト）。案 E が Phase 0 で失敗した場合の後続候補として記録する

### 2.9 B-1-P0 GATE の適用要件（v1.7 §2.9 のまま・確定）

production src/ modified=0 / staged=0・measurement/harness のみ test-only 追加可・commit 禁止・calibration 禁止・threshold update 禁止・engine-fit 0.75^N（経験式）unchanged。

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（v1.7 §3.1 を継承・行番号再実測一致）

- 根本原因: `setAutoGainStagingEnabled`（AudioEngine.h:1416-1432）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425）が発火。既定 staging=true（:2626）→ `eq` モードの呼出順（`configureProbeFlatEQ` :1799 → `setAutoGainStagingEnabled(false)` :1803）で AGC が再設定される
- `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）・`getEQProcessor`（:1292）と `setEQAGCEnabled`（:1306）は同一 `uiEqEditor` 経由 ✓
- **修正案 A（呼出順逆・eqdiag パターン :1833/1834 に揃える）を維持**。検証手順（§3.1.3・state propagation 明示版）も維持。窓 `[0.486,0.496]`（:2042-2044）は不変前提だが再校正は Phase 3-D

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（production 欠陥として記録・v1.7 §3.2 を継承）

- `EQProcessor.Processing.cpp`: `bypassTransitionActive` :516 / `dryCopyBase` 充填 :570-578 / ブレンド本体 :978-993（`canBlendDry = (dryCopyBase != nullptr)` :980・dry 混合 :993）実測 ✓
- 定常 B-1 の原因ではない・遷移時のみ。**修正案 D（記録のみ・別 work item）を維持**

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化（v1.7 §3.3 を継承）

- `ir` モードは IR ロード後 bypass 解除なし → 出力は dry コピー（:1745 bypass 固定・:1749 ir 分岐）。`irwet<digit>` は :1763 で分岐・**解除済み（:1778）**
- **修正案 B（記録のみ・`irwet` で wet 対照カバー）を維持**

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役（v1.7 §4 を継承・数値確定）

### 4.1 現状（本日再実測・全項一致）

- 519 行・TEST_F（fixture `PublicationValidatorIsolationTests`）**34** + TEST（fixture `CrossfadeAuthorityRegressionTest`）**4** = 38 ケース ✓（`grep -c` 実測）
- 分類（メソッド名プレフィクス集計）: **ValidatePublication 7 / ValidateSemanticConsistency 1 / ValidateTopology 6 / ValidateResources 11 = 25**・**ValidateTransition 8 / checkNoConflictingTransitions 直接 1 = 9** ✓（メソッド名 grep 集計実測 — v1.7 R2-10 の修正値を確定）。全 38 TEST_F/TEST は fixture `PublicationValidatorIsolationTests`(34) + `CrossfadeAuthorityRegressionTest`(4)
- CMake 未登録（39 add_executable のいずれにも該当なし）✓・gtest 使用は本ファイルのみ ✓
- `validator_.checkNoConflictingTransitions` 実呼出し **9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）✓
- **コンパイル不可**（private 宣言: RuntimePublicationValidator.h:92 private:・:101 checkNoConflictingTransitions）✓
- `tools/build-debug.bat:29` = `cmake --build ... --target PublicationValidatorIsolationTests`（stale・実測）✓
- 検査順序: RuntimePublicationValidator.cpp:14-41（SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins）✓ errorMessage 固定文字列 :16/:24/:32/:40 ✓
- AudioEngine.h:3659 `validator_(&validator)` → :3670 `validator_->validatePublication(world)` ✓

### 4.2 移行方針（v1.7 §4.2 のまま）

Phase 0 分類 → Phase A（CrossfadeAuthority 4 ケース最優先）→ Phase B（validator 25 + CheckTransition 9）→ Phase B'（Semantic Equivalence Gate・target failure isolation 含む）→ Phase C（退役 + build-debug.bat:29 の stale 参照除去）。

---

## 5. §5 別課題（v1.7 §5 を継承・R3-5 で R-2 の内訳確定）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | 未捕捉例外 → terminate。**変換実体 9**（直接 7: stod×4 :1582/:1592/:1614/:1615・stoi×2 :1583/:1591・stof×1 :1590 ＋ lambda 本体 2: stoi :1562/:1567）・**throw 可能経路 12**（直接 7 ＋ lambda 呼出 5: parseHcIdx :1606/:1608/:1610・parseLcIdx :1607/:1611）（R3-5）。lambda 本体 2 箇所の try/catch 化で全 12 経路を fail-closed 化 | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=`（:1613）は値を受理して破棄（設計上ステップ固定）。silent-ignore 系 | **次（極小）** |
| **B-3** | timestamp-based capture（flipIndex 誤差 <0.5%・現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合（uv tool extras=mcp のみで proxy 起動不可等） | 中 |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正） | 高 |

R-1 / R-2 の参考実装は **v1.7 §5.1 のまま**（`parseIntOrFail` を `parseIdxOrFail` に一般化し parseHcIdx/parseLcIdx の 2 lambda にも適用）。

---

## 6. 推奨する実行順序（v1.8・骨格は v1.7 §6 のまま）

```
[Step 1] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・v1.7 §3.1.3 の検証手順に従う）
  - F-3/F-4: 記録のみ
  - 検証: AudioEngineHarness.exe --buzz-rigcheck=eq で gainpath staging=0 eqAGC=0 を確認

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移植 + Phase B' semantic equivalence gate
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-9（commit 意思決定・ユーザー判断待ち）
  - 案 A 採用時はエントリ配線の最小変更を含めて commit
  - O-8（pyc 復元可否）・O-9（docs 目録扱い）を判断対象に追加（v1.8）

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E）
  - Phase 0-0: DESIGN-CONTRACT-A（v1.8 版 E1〜E5）を明示承認
  - Phase 0-1: Baseline characterization（0.75^N ±1e-6・record-only）
  - Phase 0-1b: REF-FIDELITY gate（bitwise）
  - Phase 0-2: Shadow Candidate E（DC = 1.0 ± 1e-6）
  - Phase 0-3〜0-6: 周波数（D1/D2/D3 定義確定版）/ block・reset / SoftClip / float-double
  - B-1-P0 GATE（§2.6.1）全 PASS → Phase 0 review → ユーザー GO
  - Phase 1: CMake option 導入（R3-8 方式）→ Phase 2: flag ON 限定検証
  - Phase 3-A → 3-B1 → 3-B2 → 3-C → 3-D → Phase 4
  - **Phase 0 FAIL 時の後続候補**: 案 D 統合 / tap 再設計 / TruePeakDetector 型全 FIR 構造（R3-12）

[Step 5] §5 別課題（将来 work item 化）: R-2 → R-1 → B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**（v1.7 と同一）。

---

## 7. 検証計画（v1.7 §7 を継承・閉形式モデルを追加）

### 7.0 閉形式モデル（v1.8 新設・authoritative 計測定義）

```bash
python doc/work113/model_polyphase_20260920.py
# 出力: 係数検証（FIRsum=1.0/convSum=0.5 全 6 design）→ DC round-trip → full-chain |H| →
#       per-design 絶対基準 → D1/D2/D3 image rejection
# 保存済み全出力: doc/work113/model_polyphase_20260920_results.txt
```

### 7.1 単体検証（v1.7 §7.1 のまま）

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テスト PASS・PVIT 削除済 |
| §2 B-1 Phase 0/1 | `cmake --build build --config Release --target AudioEngineHarness && build\Release\AudioEngineHarness.exe` | `[OS_DIRECT] ... roundTripGain=`（flag OFF ≈0.75^N・flag ON ≈1.0） |
| §2 B-1 Shadow / REF-FIDELITY | Phase 0-1b/0-2 の test-only reference 実行 | REF-FIDELITY bitwise 一致 → Candidate DC = 1.0 ± 1e-6 |

**注**: `PublishPipelineIntegrationTests.exe` は存在しない（実測 ✓）。ターゲット/バイナリは AudioEngineHarness（`build\Release\AudioEngineHarness.exe` 実在再確認 ✓・`add_executable(AudioEngineHarness)` CMakeLists.txt:1899）。`--buzz-*` フラグは AudioEngineHarness.exe に有効。

### 7.2〜7.4 統合検証 / ビルド検証 / 静的解析: **v1.7 §7.2〜§7.4 のまま**

（静的解析: cppcheck を `CustomInputOversampler.cpp` に本日再実行 → 指摘 0 件再確認・R3-13）

---

## 8. ロールバック計画（v1.7 §8 のまま・確定）

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral） | `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild（第一手段・production source 触らない） |
| §2 B-1（source-level） | `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN … #endif` ブロックと in-source safety net を削除 + CMake option/add_compile_definitions を除去（rebuild 必須） |
| §3 F-2 | 1 行 revert |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

---

## 9. 影響度まとめ（v1.7 §9 のまま・O-8/O-9 を追加）

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| O-8（pyc 復元） | working tree のみ | 極小 | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.4963 dB） | **極大**（1 行追加） | 必須（Phase 0→1→2→3→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | production 潜在欠陥（記録のみ） | なし（本セッション） | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照（行番号は 2026-09-20 本日再実測・全件一致確認）

- 前スナップショット: `doc/work113/residual_tasks_20260919.md` / 本セッション報告書: `residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 + レビュー① 38 観点）・`renew_plan_verification_20260920.md`（レビュー②）・v1.6/v1.7 改訂サマリ
- authoritative 計測モデル: `doc/work113/model_polyphase_20260920.py` + `model_polyphase_20260920_results.txt`（v1.8 新設）

**行番号再監査結果（全件一致・v1.7 §10 を確定）**:

| ファイル | 確認行 | 結果 |
|----------|--------|------|
| `src/CustomInputOversampler.h` | isLinearPhaseFIR/isSymmetricUpDown: 21-22 | ✓ |
| `src/CustomInputOversampler.cpp` | tapsForStage/attenuationForStage 84-106 / prepareStage 287-390（centerTap 292・center 0.5 335/348・scale 341・convCoeffs 360-365） / prepareSingleStage 392 / interpolateStage 492-568（convValue ×2: **557**） / decimateStage 570-723（output: **717**） / reset **452** | ✓ |
| `src/CustomInputOversampler.cpp`（v1.8 追加） | processUp **725**（hardFallback 透過 728-740・空ブロック返却 744-749・段ループ 765-779）/ processDown **785**（hardFallback 透過 789-797・クリア 808-818・容量超過 clear 826-832） | ✓ 新規 |
| `src/audioengine/AudioEngine.h` | setAutoGainStagingEnabled 1416-1432（setAGCEnabled **1425**） / 2626 staging 既定 true / 1292 getEQProcessor・1306 setEQAGCEnabled / 3659・3670 validator | ✓ |
| `AudioEngine.Processing.Latency.cpp` | static_assert 6-8 / taps 表 22-24 / groupDelaySamplesAtStageRate = taps[stage]−1 **30** | ✓ |
| `DSPCoreLifecycle.cpp` | softClipOS.prepareSingleStage(31, 90.0) **188 / 261**（diagLog 版/非 diag 版の 2 箇所） | ✓ |
| `DSPCoreFloat.cpp` | softClipOS.processUp **405** / processDown **413** | ✓ |
| `DSPCoreDouble.cpp` | kOutputHeadroom = 0.8912509381337456 **593** | ✓ |
| `EQProcessor.Processing.cpp` | bypassTransitionActive **516** / dryCopyBase 570-578 / blend 978-993 | ✓ |
| `RuntimePublicationValidator.cpp/.h` | 検査順序 14-41 / errorMessage 16/24/32/40 / private **92** / checkNoConflictingTransitions **101** | ✓ |
| `TruePeakDetector.cpp`（v1.8 拡張） | interpolateStage **284-311**（even/odd 両位相 = center + conv・×2 なし） | ✓ |
| `BassBuzzMeasurement.cpp`（**本日 2687 行**・6e61b5a 1429 行から +1258 行/93%: 16:9 で差分 6e61b5a...HEAD = +634/−2）| configureProbeFlatEQ **1072**（内 setEQAGCEnabled 1081） / runOversamplerDirect **1293**（upPeak 1342-1350・roundTripGain **1367**） / runEqDirectDriveAttribution **1526/1532** / parseHcIdx/parseLcIdx **1558/1564**（stoi 1562/1567） / stod/stoi/stof 1582-1615 / flip-eqgain **1613** / F-2 eq モード呼出順 1799→1803（eqdiag パターン 1833/1834） / `irwet` 1749-1778 / eq 窓 **2042-2044** | ✓ |
| `PublishPipelineIntegrationTests.cpp` | main **1085** / 前方宣言 **1116** / 呼出し **1223** | ✓ |
| `PublicationValidatorIsolationTests.cpp` | **519 行** / TEST_F 34 + TEST 4 / 分類 7/1/6/11 + 8/1 / 呼出し 135/246/255/265/273/281/313/324/334 | ✓ |
| `CMakeLists.txt` | option 群 **40**（clang-tidy）/**71**（MKL）/**128-129**・NUC_DEBUG_GUARDS **52-58** / add_subdirectory(JUCE) **1043** / juce_add_gui_app **1062** / CONVOPEQ_ALL_SOURCES **1133** / target_sources **1282** / CONVOPEQ_ALL_SOURCES ループ **1892** / add_executable(AudioEngineHarness) **1899** | ✓ |
| `tools/build-debug.bat` | stale target **29**（PublicationValidatorIsolationTests） | ✓ |
| `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp` | up `buf[N-1] = 2 * samples[i]` **185** / down `buf[N-1] = bufferSamples[i << 1]` **228**（両位相 ×2 / down ×1 = E5 証拠） | ✓ |

---

## 11. ユーザー判断待ち項目（v1.7 §11 を継承・O-8/O-9 を追加）

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 計装 | A: 最小実行可能資産 / B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-7 AGENTS.md | 触らない / 別 work item で更新 | **触らない** |
| §1 O-8 pyc | 復元（git checkout）/ 触らない / `.gitignore`+`rm --cached` を別 work item 化 | **触らない→将来 work item** |
| §1 O-9 docs 目録 | 触らない / commit | **触らない（ユーザー判断）** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E** |
| §2 DESIGN-CONTRACT-A（G-0） | 明示承認 / 修正要求 / 却下 | **明示承認（v1.8 版 E1〜E5・E3 は修正後）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **全実施** |
| §2 Phase 3 の commit 分割 | 3-A/3-B1/3-B2/3-C/3-D 分割 / 単一 Phase 3 | **分割** |

### §2 B-1 採用案の前提条件（v1.8・P0-E は再定義版に更新）

1. **DESIGN-CONTRACT-A**（v1.8 版）が明示承認される（Phase 0-0・G-0）
2. **REF-FIDELITY** が bitwise 一致（Phase 0-1b）
3. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（Phase 0-2・P0-A）
4. **passband 特性が維持**: P0-B（50 Hz/1 kHz ±0.1 dB）・P0-C（[0.005, 0.30] ripple ≤ 0.05 dB）
5. **anti-aliasing / stopband が維持**: P0-D（alias leakage ≤ A−10 dB・transition_end 以降）
6. **出力 image が不変**: P0-E-2（D2/D3 base==cand ≤ 0.02 dB — R3-1 により gate 定義変更）
7. **latency が不変**（impulse 実測・P0-F）
8. **SoftClip 局所 OS が正常動作**（P0-I）
9. **Differential が center-phase ×2 のみに帰属**（P0-C'・per-config 適用域）

### v1.8 最終判定表（v1.7 §11 判定表を継承）

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-9 | **GO 候補** | commit/push/復元の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替） |
| §3 F-3 / F-4 | **記録継続** | production 潜在欠陥 / dry 測定基準維持 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO（defect は DC round-trip = 0.75^N で客観確定・D1 補助証拠）** | R3-2 により証拠の主従を更新 |
| **B-1 Phase 0** | **GO（測定仕様 v1.8 適用が条件）** | P0-E 再定義 + P0-C 判定帯域統一 + dense 評価必須化を含む |
| B-1 Phase 1 / 2/3/4 | **HOLD** | Phase 0 全 PASS + ユーザー GO が前提 |
| **B-1 案 E** | **有力仮説として採用** | 仮説だが defect 自体は客観確定。FAIL 時の後続候補を R3-12 に追加 |
| 「perfect reconstruction」 | **表現修正のまま** | DC gain 1.0 のみ確定 |
| B-1 calibration 変更 | **HOLD** | Phase 3-B1/B2 → 3-C/D の順 |
| `0.75^N` engine-fit 削除 | **HOLD** | Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | `#if` セマンティクス確定（R3-8） |
| TruePeakDetector 姉妹実装 | **記録継続・後続候補として格上げ記録（R3-12）** | 案 E の正解としない |
| **v1.7 §2.7.3 candidate image 契約** | **廃棄（非再現）** | R3-1 |

<!-- v18-sec10 -->









---

## 12. 本セッションの検証エビデンス（2026-09-20・確定事項の出所）

### 12.1 閉形式モデル（C++ 厳密移植・authoritative 計測定義）

- `doc/work113/model_polyphase_20260920.py`（本セッション新規）:
  `prepare_coeffs`（Kaiser sinc ＋ half-band 零化 ＋ 正規化 ＋ center=0.5 / conv=0.5）・
  `interpolate_stage`（convValue ×2・candidate は centerValue ×2）・
  `decimate_stage`（center + conv stride-2 dot）。
  **ループ実装（C++ スカラーパスと同一演算順）との等価性を 4 design で ≤2.2e-15 で自己検証済み**。
- `doc/work113/model_polyphase_20260920_results.txt`（全測定出力・保存済み）。
- 測定系: round-trip 周波数応答は impulse → zero-pad FFT（2^20）による **exact DTFT**。
  tone 測定は **無窓矩形 FFT**（測定定義を単純化: D1 は up 出力 2N 点・D2 は過渡トリム + Hann 窓後の N 点・
  D3 は過渡トリム + Hann 窓後の full-chain 出力）。
- 本モデルが確定した数値: §2.7.1/§2.7.2 全表（§2.7.2 は v1.7 と**全点一致**・§2.7.3 は v1.7 破棄 → D1/D2/D3 表に置換）。
  IIR3 @0.49 のみ v1.6/v1.7 と不一致（+0.0062 vs +0.043）→ Phase 0-3 [A] 実測で確定（§11 要確認事項）。

### 12.2 ソース監査エビデンス（v1.7 §10 の全行番号を本日再実測・全件一致）

- `CustomInputOversampler.h`: 21-22（isLinearPhaseFIR/isSymmetricUpDown）・`reset()` 宣言 **:32**・`clearAllStages()` **:71**（reset 本体 :452-467 は clearAllStages を呼ばず per-ch clear — 両者の履歴クリア範囲の差異を記録。要確認時は統一を検討）。
- `CustomInputOversampler.cpp`: 84-106（taps/attenuation）/ 287-390（prepareStage・centerTap :292・center 0.5 :335/:348・scale :341・convCoeffs :360-365）/
  392（prepareSingleStage）/ reset **452-467**（numStages ループ × per-ch `FloatVectorOperations::clear`（upHistory :458・downHistory :459）・**`clearAllStages` 呼出なし**・corruption/hardFallback フラグ解除 :464-466）/
  492-568（interpolateStage・convValue ×2 **:557**・denormal clamp :558-559・corruption 境界 :509-554）/
  570-723（decimateStage・silence 最適化 :584-629 ※**静的解析の NO ISSUES は本パスに言及していないため cppcheck 再実行は silence パス全通りの確認と解釈**・output **:717**）/
  processUp **725**（hardFallback 透過 728-740・空ブロック返却 744-749・段ループ 765-779）/ processDown **785**（hardFallback 透過 789-797・クリア 808-818・容量超過 clear 826-832）。
- `AudioEngine.h`: setAutoGainStagingEnabled 1416-1432（setAGCEnabled **1425**）/ getEQProcessor **1292**・setEQAGCEnabled **1306**（同一 uiEqEditor 経由）/
  autoGainStagingEnabled 既定 true **2626** / validator 初期化 3659・呼出 3670。
- `AudioEngine.Processing.Latency.cpp`: static_assert **6-8** / taps 表 22-24 / groupDelay **30**。R3-11: 案 E は taps・係数列不変 → static_assert 成立持続。
- `DSPCoreLifecycle.cpp`: softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock) **188/261**（diagLog 版/非 diag 版・同一関数経路）。
- `DSPCoreFloat.cpp`: float→double 変換 **252-263** / softClipOS.processUp **405** / processDown **413**。
- `DSPCoreDouble.cpp`: kOutputHeadroom = 0.8912509381337456（**593**・**0.891 は EqDirectLog 実測の eqIdentityMode 窓 0.891 とは無関係・別定数**）。
- `EQProcessor.Processing.cpp`: bypassTransitionActive **516** / dryCopyBase 充填 **570-578** / blend 本体 **978-993**（canBlendDry :980・混合 :993）。
- `RuntimePublicationValidator.cpp/.h`: 検査順序 **14-41** / errorMessage **16/24/32/40** / `private:` **92** / checkNoConflictingTransitions **101**。
- `TruePeakDetector.cpp`: interpolateStage **284-311**（履歴 shift :297-300・両位相 = center + conv **:305-306**・×2 補償なし・R3-12）。
- `BassBuzzMeasurement.cpp`（本日 **2687 行**）: §10 表の行番号全件 + F-2 呼出順 **1799→1803**（eqdiag パターン 1833/1834）/ `irwet` 分岐 **1763**・解除 **1778**。
- `PublishPipelineIntegrationTests.cpp`（本日 **1340 行**）: main **1085** / 前方宣言 **1116** / 呼出 **1223**。
- `PublicationValidatorIsolationTests.cpp`: **519 行**・TEST_F 34（fixture PublicationValidatorIsolationTests）+ TEST 4（fixture CrossfadeAuthorityRegressionTest）。
  分類（メソッド名）: ValidatePublication 7 + ValidateSemanticConsistency 1 + ValidateTopology 6 + ValidateResources 11 + ValidateTransition 8 + checkNoConflictingTransitions 直接 1 = **34**（残り 4 は CrossfadeAuthority 回帰）。呼出 9 箇所実測。
- `CMakeLists.txt`: option 群 **40**（clang-tidy）/ 71（MKL）/ 128-129 / NUC_DEBUG_GUARDS **52-58**（`add_compile_definitions` の既存前例）/
  add_subdirectory(JUCE) **1043** / juce_add_gui_app **1062** / CONVOPEQ_ALL_SOURCES **1133** / target_sources **1282** /
  CONVOPEQ_ALL_SOURCES ループ **1892** / add_executable(AudioEngineHarness) **1899**。
- `tools/build-debug.bat`: stale target **29**。`build\\Release\\AudioEngineHarness.exe` 実在（41,137,664 bytes）。
- `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp`: up `buf[N-1] = 2 * samples[i]` **185** / down `buf[N-1] = bufferSamples[i << 1]` **228**（E5 証拠）。

### 12.3 要確認事項（Phase 0 で確定させる・gate 外）

1. **IIR3 の (4/3)^N 差分偏差 @0.49 Fs_in**: 本モデル +0.0062 dB vs v1.6/v1.7 +0.043 dB（S1・LP3・per-design 表は一致のため IIR3 の 0.49 近傍評価に限定的）。
   Phase 0-3 [A] 実測で確定。gate 外（記録のみ）のため Phase 0 GO/HOLD に影響しない。
2. **P0-C の dense 評価閾値実現性**: 本モデルでは candidate [0.005, 0.30] ripple = S1 0.001 / IIR3 0.025 / LP3 0.012 dB（gate 0.05 dB に対し余裕あり）→ Phase 0-3 [D] 実測で確定。
3. **静的解析の silence パス確認記録**: cppcheck（指摘 0）は silence 早期 return パス（:594-629）の「成立条件の説明」に言及していない。
   Phase 0 実測で inputSilent 偽陽性がないこと（推移帯信号での誤検出 0）を corruption フラグで確認する。

---

*本書は v1.7（remediation_plan_20260920_v1.7_revised.md）の再改訂版 v1.8 である。
v1.7 の本文は参照文書として維持し、本書の R3-1〜R3-13・§2.6.1/§2.7・§10/§12 の実測値が v1.7 と矛盾する箇所では**本書を優先する**。*
