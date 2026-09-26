# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v2.3）

- **版**: v2.3（v2.2 + Phase 0 前下準備完了を反映・**R9-1〜R9-6** 追加）
- **前版**: `doc/work113/remediation_plan_20260922_v2.2_revised.md`（R8-1〜R8-12）
- **中間保存**: `doc/work113/remediation_v22_intermediate_20260921.md`（§4 完了済みに更新）
- **下準備成果**: `doc/work113/remediation_v22_prep_patches_20260922.md`
- **Shadow 骨格**: `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（test-only・production 変更 0）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/` の未 commit 差分 0 件（test-only 除く）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `model_polyphase_20260920_results.txt`
  - **D1 は補正軸をモデルへ反映済み（R9-1）**: tone bin = f̂·N / image bin = N−f̂·N + argmax 検証
  - 旧軸は結果ファイルに `D1 OLD AXIS`（破棄済参考）として残置
  - 再実行環境: WSL python3 **3.14.4 / numpy 2.5.3 / scipy 1.18.1**
- **本書は read-only 監査と方針案の提示**。production / CMake への適用は本書承認後
- **案 E は有力仮説（candidate hypothesis）**。defect（DC = 0.75^N）は confirmed。Phase 0 完了まで「確定」と表記しない（R5-4）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）

---

## v2.2 → v2.3 改訂サマリ（2026-09-22・Phase 0 前下準備）

v2.2 の GO/HOLD・案 E 仮説位置付け・O-* 推奨は **変更しない**。本改訂は「技術的下準備が完了した」ことの確定と、モデル D1 補正軸の実測結果を記録する。新規確定は **R9-1〜R9-6**。

| ID | 区分 | 内容 | v2.3 での確定 |
|----|------|------|----------------|
| **R9-1** | **R8-1 の技術側完了** | モデル `d1_corrected`（tone=f̂·N / image=N−f̂·N / argmax）を実装し結果を再生成。**base −9.5424 dB が passband 24/24 一致**。旧軸は `D1 OLD AXIS` として残置 | R8-1 の「モデル未反映」は **解消**。authoritative は **モデル補正軸 + 本書 §2.7.3**（両者が一致） |
| **R9-2** | **argmax 挙動の確定** | base は tone/image とも expected bin と一致。cand は tone 一致・**image は近零域のため search 窓 ±8 bin で argmax が漂う** | Phase 0 測定仕様どおり。**argmax 採用 + bin 番号ログ必須**。image 漂移を FAIL と誤判定しないこと |
| **R9-3** | **Shadow 骨格** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` を作成（D1 bins / expected DC / candidate マクロ）。process 本体は未実装 | production 変更 **0**。Phase 0-1b REF-FIDELITY / 0-2 Candidate の受け皿 |
| **R9-4** | **パッチ案文書化** | CMake option / F-2 案 A / R-2 fail-closed（PPIT **:1149** 含む）を `prep_patches_20260922.md` に記載 | **未適用**。適用条件は v2.3 承認 + ユーザー GO |
| **R9-5** | **§2.7.3 表との差分** | モデル cand D1 は表と概ね一致。**63/120 @f̂=0.2 のみ ≈0.94 dB**（表 −103.3 / モデル −102.36）。base は全点一致 | GATE 影響なし（D1 は record-only）。Phase 0 C++ ハーネスで同一測定を再確定する |
| **R9-6** | **下準備完了** | intermediate §4 の 5 項目（モデル D1 / Shadow / CMake 案 / R-2 案 / F-2 案）がすべて完了 | **技術的未確定は引き続き 0**。残りはユーザー意思決定のみ（R8-12 継承） |

**v2.2 から変更しない点**: 案 E の技術的内容 / Phase 0 の 5 構造 / 段階リリース骨格 / P0-E = image invariance / P0-I 3 分割 / D2 gate [0.005, 0.40] / reset() 経由初期化必須 / compile-time flag 方式 / F-2 案 A / F-3 案 D / F-4 案 B / F-1 MIGRATE THEN RETIRE / R5〜R8 全件 / O-1〜O-14 推奨。

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高** | B-1 | 全オーディオ経路 | flag OFF rebuild（compile-time rollback） |
| **P2 中** | F-1〜F-4 | test-only（F-3 は production 潜在欠陥の記録） | ファイル revert |
| **P3 低** | O-1〜O-14, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: §6 の実行順序に従う（Step 0 現状固定 → Step 1 B-1 Phase 0 → Step 2 Phase 0 review → Step 3 F-2 → Step 4 F-1 → Step 5 O-* → Step 6 Phase 1+ → Step 7 別課題）。

**v2.3 での下準備位置**: Step 0 の技術側は完了済み。Step 1 着手前の残作業は **G-0 承認と Phase 0 GO（ユーザー判断）のみ**。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-14・R6-2/R8-3/R8-6/R9-3 反映）

- **O-1**: BassBuzzMeasurement.cpp **+634/−2** + PPIT **+8/−0**。窓 :2042-2044・`[OS_DIRECT]` / `runOversamplerDirect` :1293・`runEqDirectDriveAttribution` :1526/:1532 ← PPIT :1223。**推奨案 A（最小実行可能資産）**
- **O-2**: 台帳更新は独立 commit
- **O-3 / O-12**: `ConvoPeq.md` **+73/−2** — commit しない（再生成: `python output_sourcecode_markdown.py`）
- **O-4**: `Testing/Temporary/CTestCostData.txt`（D）現状維持
- **O-5**: `.opencode/opencode.json` 触らない
- **O-6**: push はユーザー手動（ahead 2）
- **O-7 / O-10**: `AGENTS.md` **+5/−3** — 環境記録として commit 可
- **O-8**: `tools/__pycache__/...pyc` — commit しない（将来: `.gitignore` + `git rm --cached`）
- **O-9**: `docs/tool-inventory-2026-09-20.md`（3102 bytes）— 触らない（ユーザー判断）
- **O-11**: `doc/work113/residual_tasks_20260919.md` **+71/−1** — 台帳として commit 可
- **O-13**: `doc/work113/*.md` untracked（計画書系列 + prep_patches 含む増加分）— 承認後、承認版+台帳を 1 commit。**件数は O-13 時点で再カウント**
- **O-14**: `.mcp.json` **+33/−23** — MCP/環境設定更新。**推奨: commit（環境記録・O-10 と同種）**
- **O-15（v2.3 記録）**: 下準備で追加された untracked は `PolyphaseGainCandidateRef.h`（test-only）と `remediation_v22_prep_patches_20260922.md`。**O-13 と同梱の承認 commit 対象に含めるかはユーザー判断**（production ではない）

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 欠陥

### 2.1 確定している事実

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     DC round-trip 0.750000（up 平均 0.75 / down 単体 DC 1.0）
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip    0.75（prepareSingleStage(31, 90.0, internalMaxBlock)・Lifecycle :188/:261）
            process: Float :405/:413 / Double :505/:513（R8-10）
数学的導出  up: even=1.0（conv×2）+ odd=0.5（center）→ 平均 0.75 / down: 0.5+0.5 = 1.0
コード根拠  convValue *= 2.0 のみ（:557）・centerValue *= 2.0 不在（AiDex/rg 0 hit）
            centerCoeff = 0.5（h:87）・decimate acc = centerCoeff * centerSample（×2 なし）
契約不整合  isSymmetricUpDown / Latency static_assert 前提と矛盾（taps は不変）
production  0 / commit 0
```

**閉形式モデル**: DC は全構成で base 0.75^N / cand 1.0 を 12 桁一致。FIRsum=1.0・center=0.5・convSum=0.5（全 6 design）。D1 補正軸でも base −9.5424 dB 構造定数が passband で成立（R9-1）。

### 2.2 根本原因（行番号）

- `prepareStage` :287-390（Kaiser β・center 0.5・convCoeffs）
- `interpolateStage` :492-568（centerValue :543-545・**convValue *= 2.0 :557**・denorm :558-559・出力 :563-564・**centerValue に ×2 なし**）
- `decimateStage` :570-723（silence :583-613・通常 :615-649・`acc = centerCoeff * centerSample` ×2 なし）
- `processUp` :725-783 / `processDown` :785-（hardFallback 透過・corruption クリア）
- `reset()` :452-467（アトミック 3）vs `clearAllStages()` :469-484（アトミック 1）→ **Phase 0 は reset() 経由必須（R4-3）**

### 2.3 修正案と DESIGN-CONTRACT-A

案 A〜D 不採用・**案 E 有力仮説**。E1〜E5 は v2.1/v2.2 のまま（E3 は R7-1 補正・E5 は JUCE 補助証拠）。

**案 E 実装案（candidate・未適用）**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E（両 polyphase 位相への ×2 対称適用・candidate）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

**構造恒等式（R5-9 継承）**: `h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`（c: base 0.25 / cand 0.5・残差 ≤1.4e-17）。passband では cand/base = (4/3)^N 厳密。

**gain 変化**: (4/3)^N（+2.4988 / +4.9977 / +7.4963 dB @1/2/3 段）。

### 2.4 CMake / flag（R3-8 方式・パッチ案は prep_patches §1）

```cmake
# option 群（:40 近傍・CONVOPEQ_ENABLE_CLANG_TIDY 等と並置）
option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
       "Correct polyphase gain convention (B-1 案E)" OFF)

# add_subdirectory(JUCE)（:1043）の後・juce_add_gui_app（:1062）の前
add_compile_definitions(
    CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
```

- 既存前例: `NUC_DEBUG_GUARDS`（:52-58 `add_compile_definitions`）
- C++ は **`#if CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（`#ifdef` 禁止）
- `build.bat` `CMAKE_EXTRA_FLAGS` 経路（:179/:181）で `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` 可
- 既定 OFF / runtime flag 不採用
- **現状: 未適用**（案のみ・R9-4）

### 2.5 Phase 0 characterization（測定仕様）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）をユーザーが明示承認
0-1  Baseline record-only: DC = 0.75^N ±1e-6
0-1b REF-FIDELITY: Shadow Reference == production（bitwise または ≤1e-15・reset() 経由）
0-2  Shadow Candidate E: DC = 1.0 ± 1e-6
0-3  周波数（D1/D2/D3）— D1 は補正軸（R8-1/R9-1）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）
0-5  SoftClip local OS（float + double・R8-10）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8）
```

**D1 定義（v2.3 確定・R9-1 反映済み）**:
- 補正軸: **D1 = |Y_up(image bin)| / |Y_up(tone bin)|**
  - up 出力長 = 2N・tone bin = **f̂·N**・image bin = **N−f̂·N**
- 期待値: base **−9.5424 dB 構造定数** / cand **−84〜−116 dB**
- **測定実装要件**: tone/image bin を **argmax で検証**し、ログに bin 番号を出力する（R9-2）
- cand の image argmax が ±8 bin 漂移しても **FAIL にしない**（近零域の数値フロア）
- results の `D1 OLD AXIS` 行は破棄済参考値
- **quality gate ではない**（E-1 記録）。final alias rejection は P0-D

**Shadow Reference（R9-3）**:
- 骨格: `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`
- `expectedDcRoundTrip(stages, candidate)` / `computeD1Bins(fhat, N)` / `CONVOPEQ_POLYPHASE_REF_CANDIDATE`
- processUp/processDown 本体は Phase 0 実装時に production 契約へ合わせて追加

**D2 / D3**: v2.2 のまま。D2 は [0.005, **0.40**] で base==cand ≤0.01 dB。0.45 は記録のみ（FP フロア）。

### 2.6 B-1-P0 GATE（判定表・v2.3）

| ID | 判定対象 | 条件 | 合否 |
|----|----------|------|------|
| G-0 | 契約 | DESIGN-CONTRACT-A 明示承認（E1〜E5） | □ |
| G-BL | Baseline sanity | DC = 0.75^N ±1e-6 のみ gate（D1 は記録） | □ |
| REF-FIDELITY | Shadow==production | bitwise または ≤1e-15・reset() 経由 | □ |
| P0-A | Candidate DC | 1.0 ± 1e-6（必要条件・十分条件ではない） | □ |
| P0-B | 低域 | 50 Hz / 1 kHz unity ±0.1 dB | □ |
| P0-C | passband ripple | [0.005,0.30] max−min ≤0.05 dB（dense） | □ |
| P0-C' | differential | (4/3)^N ±0.05 dB（IIR3/LP3 f≤0.45 / S1 f≤0.30） | □ |
| P0-D | stopband | alias ≤ −(A−10) dB @[t_end, 0.5] | □ |
| P0-E | **image invariance** | E-1 D1 補正軸記録（argmax ログ） / E-2 D2 [0.005,0.40] base==cand ≤0.01 dB / E-3 0.45 記録のみ | □ |
| P0-F | latency | peak = Σ(taps−1)/2^(s+1) ±1（up+down 合計・R7-4） | □ |
| P0-G | float/double | maxAbsErr ≤5e-7 / RMS ≤5e-8 | □ |
| P0-H | block/reset | partition bitwise・reset contract | □ |
| P0-I | SoftClip | I-a DC=1.0±1e-6 / I-b 安全性 gate / I-c 動作点 record | □ |

**全 PASS → Phase 1 eligibility**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

### 2.7 測定軸と実測基準値（authoritative）

#### 2.7.0 軸定義（確定）
```
f̂ = f / Fs_stage ∈ [0, 0.5]（own-rate）
f̂_in = f / Fs_in ∈ [0, 0.5]（full-chain）
例: stage0 (511) の −0.1 dB edge = 0.2448 own-rate = 0.4896 Fs_in
D1 bins（up・2x rate）: tone = f̂·N / image = N−f̂·N
```

#### 2.7.1 Full-chain

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC base / cand | 0.75 / 1.0 | 0.421875 / 1.0 | 0.421875 / 1.0 |
| cand \|H\| @50Hz〜0.25 | −0.0000〜−0.0004 dB | −0.0078〜−0.0108 dB | +0.0025〜−0.0047 dB |
| (4/3)^N dev @0.45 | **+1.6350 dB**（適用外） | +0.0121 dB | −0.0025 dB |
| (4/3)^N dev @0.49 | +3.3927 dB | **+0.0062 dB** | +0.0025 dB |
| ripple [0.005,0.30] base/cand | 0.001/0.001 | 0.034/**0.025** | 0.016/**0.012** |
| cand 0.1 dB edge | 0.3526 Fs_in | 0.4897 Fs_in | 0.4945 Fs_in |

#### 2.7.2 FIR 絶対基準（P0-D floor）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|----------------|---------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

#### 2.7.3 Image rejection（**D1 補正軸が authoritative・R9-1 でモデル反映済み**）

**D1 補正軸**（tone bin = f̂·N / image = N−f̂·N）:

| design | f̂=0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 |
|--------|---------|------|------|------|------|------|------|
| 31/90 base/cand | −9.54/−91.3 | −9.54/−104.4 | −9.54/−94.6 | −9.54/−90.8 | −9.66/−46.2 | −10.99/−25.0 | −24.18/−11.1 |
| 127/110 | −9.54/−94.6 | −9.54/−98.6 | −9.54/−93.0 | −9.54/−93.8 | −9.54/−92.4 | −9.54/−94.0 | −9.55/−70.4 |
| 511/140 | −9.54/−115.9 | −9.54/−111.5 | −9.54/−102.7 | −9.54/−97.1 | −9.54/−95.9 | −9.54/−92.0 | −9.54/−83.9 |
| 63/120 | −9.54/−97.7 | −9.54/−95.4 | −9.54/−103.3 | −9.54/−92.2 | −9.54/−91.8 | −9.57/−59.1 | −11.92/−21.2 |
| 255/140 | −9.54/−94.1 | −9.54/−102.4 | −9.54/−94.8 | −9.54/−91.4 | −9.54/−100.0 | −9.54/−89.5 | −9.54/−84.4 |
| 1023/160 | −9.54/−97.6 | −9.54/−95.6 | −9.54/−102.7 | −9.54/−92.2 | −9.54/−91.8 | −9.54/−92.0 | −9.54/−85.2 |

構造: base image 比 = (1−c_x)/(1+c_x) = 0.25/0.75 = **20·log10(1/3) = −9.5424 dB**（全 design・passband）/ cand = image 構造的零。

**モデル再実測（R9-1/R9-5）**:
- base: passband で **−9.5424 dB と 24/24 一致**
- cand: 本表と概ね一致。**63/120 @0.2 のみモデル −102.36 dB vs 表 −103.3 dB（Δ≈0.94 dB）** を記録。Phase 0 C++ 測定で再確定する
- `D1 OLD AXIS` 行（破棄済）: 例 31/90 base @0.30 = +11.51 dB 等。**解釈禁止**

**D2（gate 帯域 [0.005, 0.40] で base==cand ≤0.01 dB）**: 表維持。@0.45 は記録のみ。

**D3**: full-chain worst spur = 窓 sidelobe −98〜−101 dB・base==cand 差 = 0.00 dB。

### 2.8 教訓

1〜25: v1.8〜v2.1 のまま。

26. **【R8-1】authoritative 成果物と補正済み測定定義の乖離を放置しない**: モデル/結果ファイルが旧軸の D1 を出力し続けると、後続セッションが「モデルが全部 authoritative」と誤読する。補正した測定定義はモデル本体へ反映するか、結果ファイル側に「破棄済軸」ラベルを明記する。

27. **【R9-1】補正済み測定は「モデル反映 + 計画表」の両方を揃えてから完了とする**: v2.3 ではモデルに `d1_corrected` と `d1_old_axis_discarded` を実装し、結果に AUTHORITATIVE / OLD AXIS を併記した。片側だけの状態を「authoritative」呼ばわりしない。

28. **【R9-2】argmax 検証は bin 漂移を FAIL と誤判定しない**: cand の image は構造的近零のため、search 窓内で最大 bin が expected からずれることがある。Phase 0 では argmax 値を採用し、expected bin との差をログとして記録する（gate は D2 の base==cand）。

### 2.9 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止 / calibration 禁止 / engine-fit 0.75^N unchanged。

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC

- 根本原因: `setAutoGainStagingEnabled`（AudioEngine.h:1416-1432）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425）。既定 staging=true（:2626）
- eq モード誤順: **:1799 configureProbeFlatEQ → :1803 setAutoGainStagingEnabled(false)**（AGC が再 ON される）
- 正順（eqdiag 等）: **:1833 staging false → :1834 configureProbeFlatEQ**
- `configureProbeFlatEQ` :1072 内 `setEQAGCEnabled(false)` :1081
- **修正案 A（呼出順を eqdiag パターンへ揃える）**
- パッチ案: `remediation_v22_prep_patches_20260922.md` §2（**未適用**）
- 検証: `AudioEngineHarness.exe --buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（AGC 強制でも結果不変）

### 3.2 F-3: EQ dry/wet 混合（記録のみ・案 D）

- production 潜在欠陥として記録。遷移時のみ。定常 B-1 の原因ではない
- 修正は別 work item（scope discipline）

### 3.3 F-4: `ir` / `irwet` 用語（R5-8 継承）

- **`ir`**: IR ロード後も `setConvolverBypassRequested(true)`（:1745）維持 = **dry ベースライン測定モード**
- **`irwet<digit>`**: :1763 分岐・**:1778 `setConvolverBypassRequested(false)`** = wet 畳み込み対照
- 未知モード fail-closed :1708-1710
- **修正案 B（記録のみ）**

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状

- 経路: **`src/audioengine/RuntimePublicationValidator.h`（105 行）/ `.cpp`（211 行）**（`src/core/` 不在）
- 構造化判定: h:13 `ValidationFailureReason` / h:24 `failureReason` / h:64 `validatePublication`
- private: h:**92** / `checkNoConflictingTransitions` h:**101**（cpp:169 に本体）
- 検査順序 cpp:8-41: Semantic → Topology → Resources → ConflictingTransitions（first-fail）
- PVIT: **519 行 / TEST 38** / 呼出 **9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）
- CMake 未登録（add_executable **39**）/ gtest は本ファイルのみ / コンパイル不可
- `tools/build-debug.bat:29` stale target
- 分類 25/9 は手動分類の目安（R6-4）

### 4.2 移行方針

Phase 0 分類 → A（CrossfadeAuthority 4）→ B（validator 系 + transition）→ **B' Semantic Equivalence Gate** → C（退役 + build-debug.bat 修正）。

**B' 主判定**: 移植先が `result.failureReason`（enum）で旧 private 検査と一致。`errorMessage` は補助 assert のみ。**MIGRATE CASES THEN RETIRE**。

---

## 5. §5 別課題（R-2 は R8-9 + R9-4 パッチ案）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | 数値 parse 未捕捉例外 → fail-closed 化。対象: BassBuzz 直接 stod/stoi/stof（:1582/:1583/:1590/:1591/:1592/:1614 系）+ parseHcIdx/parseLcIdx 本体（:1562/:1567）+ 呼出（:1606-:1611）+ **PPIT stoi :1136/:1146/:1149/:1157/:1166**。仕様: try/catch + `[BUZZ] FAIL: invalid numeric` + 非ゼロ終了 + regression test。**実装スケッチ: prep_patches §3（未適用）** | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=` silent-ignore（設計上ステップ固定） | 次（極小） |
| **B-3** | timestamp-based capture（現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合 | 中 |
| **D-1** | build identity gate M1/M2 | 高 |

---

## 6. 推奨する実行順序（v2.3）

```
[Step 0] 現状固定 — 技術側完了
  - HEAD 8f127bfe / production src/ 未 commit 0
  - モデル D1 補正軸反映済み（R9-1）/ results に AUTHORITATIVE + OLD AXIS
  - Shadow 骨格 + パッチ案文書あり（production 未適用）
  - 残: ユーザー判断（O-* / G-0 / Phase 0 GO）

[Step 1] §2 B-1 Phase 0（production 変更 0）— G-0 承認が前提
  - Baseline → REF-FIDELITY（Shadow 骨格を実装）→ Candidate → 周波数/block/SoftClip/float-double
  - D1 は補正軸 + argmax ログ（R9-2）

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は image invariance として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §3 F-2（test-only・案 A・prep_patches §2）
  - :1799/:1803 を :1833/:1834 パターンへ
  - rigcheck=eq で staging=0 eqAGC=0 を確認
  - F-3/F-4 記録のみ

[Step 4] §4 F-1（A→B→B'→C・failureReason 主判定）

[Step 5] §1 O-1〜O-15（ユーザー判断）
  - O-14 .mcp.json / O-15 下準備成果（Shadow + prep_patches）の commit 方針
  - O-13 は件数再カウント後に承認版+台帳 commit

[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option 適用（prep_patches §1）→ flag ON 検証 → Phase 3-A…4

[Step 7] 別課題: R-2（prep_patches §3・PPIT :1149 含む）→ R-1 → B-3, D-2, D-1
```

---

## 7. 検証計画

### 7.0 閉形式モデル再現（D1 補正軸後）

```bash
wsl -d Ubuntu-26.04 -- bash -lc \
  'cd /mnt/c/VSC_Project/ConvoPeq && python3 doc/work113/model_polyphase_20260920.py'
# 期待:
#   DC base 0.75^N / cand 1.0（12 桁）
#   D1 base passband ≈ -9.5424 dB
#   D1 cand は §2.7.3 と概ね一致（63/120@0.2 は ≈0.94 dB 差を許容記録）
#   D1 OLD AXIS 行が残存（破棄済参考）
#   D2 フロアは ±0.01 dB まで許容
# 環境記録: python3 / numpy / scipy バージョン
```

### 7.1 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| F-2 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS・PVIT 削除 |
| B-1 Phase 0/1 | AudioEngineHarness `[OS_DIRECT] roundTripGain=` | OFF≈0.75^N / ON≈1.0 |
| Shadow/REF | Phase 0-1b/0-2 test-only | REF bitwise → Cand DC=1.0 |
| D1 Phase 0 | ハーネスログ | tone/image bin + argmax（R9-2） |
| 静的解析 | Windows cppcheck（指摘 0・exit=0） | 既知の portability のみ |

注: `PublishPipelineIntegrationTests.exe` は存在しない。ターゲットは **AudioEngineHarness**（41,137,664 bytes・CMakeLists :1899）。

---

## 8. ロールバック

| 対象 | 方法 |
|------|------|
| §1 commit | `git revert <sha>` |
| §2 B-1 behavioral | **compile-time rollback**: `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild（runtime 即時復旧ではない） |
| §2 B-1 source | `#if` ブロック + CMake option を削除 + rebuild |
| §3 F-2 | 1 行 revert |
| §4 F-1 | 移行先を残す場合のみファイル復元 |
| §5 | 実装しないため不要 |
| 下準備成果 | test-only / doc のため production ロールバック不要（commit する場合のみ revert） |

---

## 9. 影響度まとめ

| 区分 | 影響 | 影響度 | 段階リリース |
|------|------|--------|--------------|
| O-1〜O-15 | リポジトリ/環境記録 | 極小 | 任意 |
| B-1 案 E | 全オーディオ経路（最大 +7.4963 dB） | **極大** | 必須 |
| F-2 | test-only | 小 | 不要 |
| F-3/F-4 | 記録のみ | なし | 不要 |
| F-1 | test-only | 中 | 不要 |
| R-2 等 | CLI/harness | 極小 | 不要 |
| 下準備（モデル/Shadow/パッチ案） | doc + test-only | 極小 | 不要 |

---

## 10. 監査ログ・参照

### 10.1 v2.2 継承（主要ソース）

| ファイル | 確認 | 結果 |
|----------|------|------|
| CustomInputOversampler.h | :21-22 / :32 reset / :71 clearAll / :87 centerCoeff=0.5 | ✓ |
| CustomInputOversampler.cpp | :557 convValue×2・center ×2 無し / reset :452 / clearAll :469 | ✓ defect |
| AudioEngine.h | staging setter :1416-1425 / default true :2626 | ✓ |
| BassBuzzMeasurement.cpp | F-2 :1799/:1803 vs :1833/:1834 / parse :1558-1615 | ✓ |
| PublishPipelineIntegrationTests.cpp | stoi :1136/:1146/**:1149**/:1157/:1166 | ✓ R8-9 |
| CMakeLists.txt | option :40 / NUC :52-58 / JUCE :1043 / app :1062 / harness :1899 / 39 targets | ✓ |
| DSPCoreDouble.cpp | processUp :505 / processDown :513 | ✓ R8-10 |

### 10.2 v2.3 追加（下準備・R9）

| ファイル | 確認 | 結果 |
|----------|------|------|
| model_polyphase_20260920.py | `d1_corrected` / `d1_old_axis_discarded` / part4 ラベル | ✓ R9-1 |
| model_polyphase_20260920_results.txt | D1 補正軸 + ARGMAX + OLD AXIS / DC 12 桁 | ✓ R9-1/R9-5 |
| PolyphaseGainCandidateRef.h | test-only 骨格（D1 bins / expected DC / candidate マクロ） | ✓ R9-3 |
| remediation_v22_prep_patches_20260922.md | CMake / F-2 / R-2 パッチ案（未適用） | ✓ R9-4 |
| intermediate §4 | 5 項目すべて完了に更新 | ✓ R9-6 |
| git | HEAD 8f127bfe / production clean / 下準備は untracked or test-only | ✓ |

---

## 11. ユーザー判断待ち項目（v2.3）

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| O-1 計装 | A 最小 / B 全 / C 退役 | **A** |
| O-2 台帳 | 同梱 / 独立 | **独立** |
| O-6 push | 即時 / 別 / 手動 | **ユーザー手動** |
| O-8 pyc | 復元 / 触らない / 別 work item | **触らない→将来 work item** |
| O-9 docs | 触らない / commit | **触らない** |
| O-10 AGENTS.md | 触らない / commit | **commit（環境記録）** |
| O-11 台帳 | 触らない / commit | **commit** |
| O-12 ConvoPeq.md | 触らない | **触らない** |
| O-13 work113 md | 承認版+台帳 commit / 全 commit / 触らない | **承認版+台帳 1 commit（件数再カウント）** |
| O-14 .mcp.json | 触らない / commit（環境記録） | **commit（環境記録）** |
| **O-15 下準備成果** | Shadow + prep_patches を O-13 に同梱 / 別 commit / 触らない | **O-13 に同梱（test-only/doc）** |
| G-0 契約 | 承認 / 修正 / 却下 | **承認（v2.1 版 E1〜E5）** |
| B-1 修正案 | E（candidate） | **E（仮説）** |
| Phase 0 | GO / HOLD | **GO（測定仕様 v2.3 適用が条件）** |
| F-2 | A 呼出順逆 | **A** |
| F-1 | 全実施 | **全実施** |
| Phase 3 commit | 分割 / 単一 | **分割** |

### B-1 採用案の前提条件

1. DESIGN-CONTRACT-A が明示承認される（G-0）
2. REF-FIDELITY bitwise（reset() 経由）
3. Candidate DC = 1.0 ± 1e-6（必要条件）
4. P0-B/C passband 維持
5. P0-D stopband 維持
6. P0-E-2 image invariance（D2 [0.005,0.40] ≤0.01 dB）
7. P0-F latency 不変
8. P0-I SoftClip 安全性 gate
9. P0-C' differential が center-phase ×2 に帰属
10. **D1 測定は補正軸 + argmax ログ（R9-1/R9-2）**

### v2.3 最終判定表

| 項目 | 判定 |
|------|------|
| §1 O-1〜O-15 | **GO 候補（O-14/O-15 追加）** |
| §3 F-2 | **GO 候補（案 A・パッチ案済み・未適用）** |
| §3 F-3 / F-4 | **記録継続** |
| §4 F-1 | **GO 候補（failureReason 主判定）** |
| §5 R-2 | **仕様確定 + パッチ案済み・将来 work item（PPIT :1149 含む）** |
| B-1 原因分析 | **GO（defect 客観確定）** |
| B-1 Phase 0 | **GO（測定仕様 v2.3 適用・D1 補正軸 + argmax 必須）** |
| B-1 Phase 1+ | **HOLD** |
| B-1 案 E | **有力仮説（確定表記禁止）** |
| モデル D1 | **補正軸を反映済み（R9-1）・OLD AXIS は破棄済参考** |
| 下準備 | **完了（R9-6）・production 未適用** |
| 技術的未確定 | **0** |
| TruePeakDetector | **後続候補第三順位** |
| compile-time flag | **`CONVOPEQ_CORRECT_POLYPHASE_GAIN` + `#if`** |

---

## 12. エビデンス（v2.2 からの追加分）

### 12.1 モデル D1 補正軸（2026-09-22）

- 実装: `d1_corrected(y, fhat, N)` — tone=f̂·N / image=N−f̂·N / ±8 bin argmax
- 結果ラベル: `D1 base/cand`（authoritative）/ `D1-ARGMAX` / `D1 OLD AXIS base/cand`
- 実測: base passband **−9.5424 dB 24/24** / DC 不変 / cand は §2.7.3 と概ね一致
- 環境: WSL python3 3.14.4 / numpy 2.5.3 / scipy 1.18.1 / exit=0

### 12.2 未確定事項の棚卸し（R9-6）

| 種別 | 状態 |
|------|------|
| R8-1 モデル D1 未反映 | **技術側解消済（R9-1）** |
| intermediate §4 下準備 | **完了（R9-6）** |
| production 追加欠陥 | **検出なし** |
| ユーザー判断 | O-1〜O-15 / G-0 / Phase GO / F 方針 — **技術調査の対象外** |

---

## 13. v2.3 で追加した確定事項（R9）

- **R9-1** モデル D1 補正軸を反映し結果を再生成。base −9.5424 dB が passband 24/24 一致
- **R9-2** argmax 挙動: base 完全一致 / cand image は近零域で ±8 bin 漂移（Phase 0 仕様どおり）
- **R9-3** Shadow 骨格 `PolyphaseGainCandidateRef.h`（test-only）
- **R9-4** CMake / F-2 / R-2 のパッチ案を `remediation_v22_prep_patches_20260922.md` に文書化（未適用）
- **R9-5** §2.7.3 表との cand D1 差分を記録（63/120@0.2 ≈0.94 dB）。GATE 影響なし
- **R9-6** intermediate §4 完了。技術的未確定は 0 のまま。残りはユーザー判断のみ
- **§2.8.27 / §2.8.28** 教訓追加（両側反映・argmax 漂移の誤判定防止）

---

*本書は v2.2（remediation_plan_20260922_v2.2_revised.md）の再改訂版 v2.3 である。*
*v2.2 本文は参照文書として維持し、**R9-1〜R9-6 および本書が矛盾する箇所では本書を優先する**。*
*R5〜R8 は本書でも有効。中間エビデンスは `remediation_v22_intermediate_20260921.md`（§4 完了済み）。*
*下準備のパッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*

**v2.2 継承の監査・確定事項**: R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。
