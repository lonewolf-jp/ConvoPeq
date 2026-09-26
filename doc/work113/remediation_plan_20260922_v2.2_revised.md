# ConvoPeq 残件 改修計画書（2026-09-21 時点・再改訂 v2.2）

- **版**: v2.2（v2.1 全項目の独立再監査（2026-09-21 夜・本セッション）反映・**R8-1〜R8-12** 追加）
- **中間保存**: `doc/work113/remediation_v22_intermediate_20260921.md`（エビデンス詳細・引き継ぎ用）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/` の未 commit 差分 0 件（test-only 除く・本セッション再実測）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `model_polyphase_20260920_results.txt`
  - **注意（R8-1）**: 結果ファイルの **D1 セクションは R7-1 破棄済の旧軸**。補正 D1 の authoritative は本書 §2.7.3 表
  - 再実行（WSL python3 3.14.4 / numpy 2.5.3 / scipy 1.18.1）: CRLF 正規化後に **91/96 行一致**・残差 5 行はすべて D2 フロア ±0.01 dB（GATE 影響なし・R8-2）
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **案 E は有力仮説（candidate hypothesis）**。defect（DC = 0.75^N）は confirmed。Phase 0 完了まで「確定」と表記しない（R5-4）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.1 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**

---

## v2.1 → v2.2 改訂サマリ（2026-09-21 夜・独立再監査）

v2.1 の production 核心（B-1 gain 欠陥・F-2 AGC・F-1 validator・F-3/F-4・JUCE E5・CMake 39 targets・PVIT 38 TEST・R7-1 D1 補正）は**全件再確認**。GO/HOLD は変更しない。新規確定は **R8-1〜R8-12**。

| ID | 区分 | 内容 | v2.2 での確定 |
|----|------|------|----------------|
| **R8-1** | **authoritative モデルの D1 軸未反映** | `model_polyphase_20260920.py` / `*_results.txt` の D1 は **R7-1 破棄済の旧軸**（results:67「v1.7 §2.7.3 相当」・31/90 D1 base @0.30 = +11.51 dB 等が残存）。補正軸 D1（base **−9.5424 dB 構造定数** / cand **−84〜−116 dB**）は v2.1 §2.7.3 表と `tmp/d1_full_check.py` 系のみに存在 | **D1 authoritative = 本書 §2.7.3**。結果ファイル D1 行は「旧軸・解釈破棄済・参考」。Phase 0 測定ハーネスで **tone bin = f̂·N / image = N−f̂·N + argmax 検証** を必須実装。GATE 影響なし（record-only） |
| **R8-2** | **モデル再現の第 2 実測** | 本セッション再実行で **91/96**（v2.1 午前 92/96 から 1 行減）。差分 5 行はすべて D2 −100 dB 以下フロアの ±0.01 dB（511/140・255/140・1023/160 @0.05/@0.35 付近） | R6-1 の環境依存を追加実証。**GATE は差分（base==cand）判定なので不変** |
| **R8-3** | **O-14 新設** | `.mcp.json` が Modified **+33/−23**（MCP サーバパス/設定の更新）。v2.1 O リストに不在だった | **O-14**: 環境/MCP 設定記録として **commit 可**（O-10 AGENTS.md と同種・ユーザー判断） |
| **R8-4** | **行番号・構造の再監査** | B-1 CIO / F-2 呼出順 / F-4 ir・irwet / F-1 validator 経路と failureReason / SoftClip prepare / JUCE E5 / Latency.cpp:30 / CMake option と 39 add_executable / PVIT 519・TEST 38・呼出 9 / PPIT main:1085・宣言:1116・呼出:1223 を全件再実測 | **v2.1 と一致**。計画の根拠行番号は変更不要 |
| **R8-5** | **静的解析再確認** | Windows cppcheck（`C:\Program Files\Cppcheck\cppcheck.exe`）`CustomInputOversampler.cpp` へ `--enable=warning,performance,portability --std=c++20` → **exit=0・指摘 0** | R4-6/R6-7 維持 |
| **R8-6** | **O-13 件数更新** | `doc/work113/*.md` untracked = **30 件**（v2.1 では 11 件計上） | O-13 の実測件数を 30 に更新。推奨（承認版+台帳を 1 commit）は不変 |
| **R8-7** | **production クリーン確認** | `git diff --name-only -- src/` の non-test 差分 = **0** | Step 0 前提の維持 |
| **R8-8** | **HEAD・push 状態** | HEAD `8f127bfe` / ahead 2 / behind 0 | O-6 継続（push はユーザー手動） |
| **R8-9** | **R-2 経路の補強** | PPIT の `std::stoi` に **:1149 を追加検出**（v2.1 の :1136/:1146/:1157/:1166 + **:1149**）。BassBuzz は :1558-1567/:1582-1615 で v2.1 どおり | R-2 fail-closed 対象は **BassBuzz 直接 7 + lambda 2 + lambda 呼出 5 + PPIT 5** を仕様に含める |
| **R8-10** | **SoftClip 両ホスト** | `DSPCoreDouble.cpp` にも softClipOS **processUp :505 / processDown :513** が実在（Float :405/:413 と対） | Phase 0-I は **float/double 両経路**で SoftClip 局所 OS を測定する（既存 P0-G/P0-I と整合・表記を明確化） |
| **R8-11** | **TPD パラメータ再確認** | `kDefaultAttenuationDb=100.0` / stage1 = `max(15, taps/2)` / 2 段 4× / `decimateStage` 不在 / 正規化後 center=0.5 / interpolate に ×2 なし | R6-5/R7-4 維持（後続候補第三順位） |
| **R8-12** | **未確定事項の分離** | v2.1 §12.4 の技術要確認 4 件は既に確定済み。本セッションで追加検出された技術事項（R8-1 のモデル D1 未反映）も**方針確定済み** | **技術的未確定 = 0**。残る未確定は **ユーザー意思決定のみ**（O-1〜O-14 / G-0 / Phase GO / F 方針） |

**v2.1 から変更しない点**: 案 E の技術的内容 / Phase 0 の 5 構造 / 段階リリース骨格 / P0-E = image invariance / P0-I 3 分割 / D2 gate [0.005, 0.40] / reset() 経由初期化必須 / compile-time flag 方式 / F-2 案 A / F-3 案 D / F-4 案 B / F-1 MIGRATE THEN RETIRE / R5-1〜R5-12 / R6-1〜R6-7 / R7-1〜R7-4。

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高** | B-1 | 全オーディオ経路 | flag OFF rebuild（compile-time rollback） |
| **P2 中** | F-1〜F-4 | test-only（F-3 は production 潜在欠陥の記録） | ファイル revert |
| **P3 低** | O-1〜O-14, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: §6 の実行順序に従う（Step 0 現状固定 → Step 1 B-1 Phase 0 → Step 2 Phase 0 review → Step 3 F-2 → Step 4 F-1 → Step 5 O-* → Step 6 Phase 1+ → Step 7 別課題）。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-14・R6-2/R8-3/R8-6 反映）

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
- **O-13**: `doc/work113/*.md` untracked **30 件**（本セッション実測・R8-6）— 承認後、承認版+台帳を 1 commit
- **O-14（新・R8-3）**: `.mcp.json` **+33/−23** — MCP/環境設定更新。**推奨: commit（環境記録扱い・O-10 と同種）**

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 欠陥

### 2.1 確定している事実（本セッション再実測）

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

**閉形式モデル（2026-09-21 夜再実行・R8-2）**: DC は全構成で base 0.75^N / cand 1.0 を 12 桁一致。FIRsum=1.0・center=0.5・convSum=0.5（全 6 design）。

### 2.2 根本原因（行番号は本セッション再実測）

- `prepareStage` :287-390（Kaiser β・center 0.5・convCoeffs）
- `interpolateStage` :492-568（centerValue :543-545・**convValue *= 2.0 :557**・denorm :558-559・出力 :563-564・**centerValue に ×2 なし**）
- `decimateStage` :570-723（silence :583-613・通常 :615-649・`acc = centerCoeff * centerSample` ×2 なし）
- `processUp` :725-783 / `processDown` :785-（hardFallback 透過・corruption クリア）
- `reset()` :452-467（アトミック 3）vs `clearAllStages()` :469-484（アトミック 1）→ **Phase 0 は reset() 経由必須（R4-3）**

### 2.3 修正案と DESIGN-CONTRACT-A

案 A〜D 不採用・**案 E 有力仮説**。E1〜E5 は v2.1 のまま（E3 は R7-1 補正・E5 は JUCE 補助証拠）。

**案 E 実装案（candidate）**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E（両 polyphase 位相への ×2 対称適用・candidate）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

**構造恒等式（R5-9 継承）**: `h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`（c: base 0.25 / cand 0.5・残差 ≤1.4e-17）。passband では cand/base = (4/3)^N 厳密。遷移端近傍では center 係数差が顕在化し得る → base==cand の主張は「passband で厳密」と限定する。

**gain 変化**: (4/3)^N（+2.4988 / +4.9977 / +7.4963 dB @1/2/3 段）。

### 2.4 CMake / flag（R3-8 方式・本セッションで前例再確認）

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

### 2.5 Phase 0 characterization（測定仕様）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）をユーザーが明示承認
0-1  Baseline record-only: DC = 0.75^N ±1e-6
0-1b REF-FIDELITY: Shadow Reference == production（bitwise または ≤1e-15・reset() 経由）
0-2  Shadow Candidate E: DC = 1.0 ± 1e-6
0-3  周波数（D1/D2/D3）— D1 は補正軸（R8-1）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）
0-5  SoftClip local OS（float + double・R8-10）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8）
```

**D1 定義（v2.2 確定・R8-1 反映）**:
- 補正軸: **D1 = |Y_up(0.5−f̂/2)| / |Y_up(f̂/2)|**（input 正規化 f̂・tone bin = f̂·N・image bin = N−f̂·N）
- 期待値: base **−9.5424 dB 構造定数** / cand **−84〜−116 dB**
- **測定実装要件**: 測定 bin を argmax で検証し、ログに tone/image bin 番号を出力する
- results ファイルの旧 D1 行は解釈破棄済参考値
- **quality gate ではない**（E-1 記録）。final alias rejection は P0-D

**D2 / D3**: v2.1 のまま。D2 は [0.005, **0.40**] で base==cand ≤0.01 dB。0.45 は記録のみ（FP フロア・R5-9 (c)）。

### 2.6 B-1-P0 GATE（判定表・v2.2）

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
| P0-E | **image invariance** | E-1 D1 補正軸記録 / E-2 D2 [0.005,0.40] base==cand ≤0.01 dB / E-3 0.45 記録のみ | □ |
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
```

#### 2.7.1 Full-chain（モデル再現・本セッション DC 以外も整合）

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC base / cand | 0.75 / 1.0 | 0.421875 / 1.0 | 0.421875 / 1.0 |
| cand \|H\| @50Hz〜0.25 | −0.0000〜−0.0004 dB | −0.0078〜−0.0108 dB | +0.0025〜−0.0047 dB |
| (4/3)^N dev @0.45 | **+1.6350 dB**（適用外） | +0.0121 dB | −0.0025 dB |
| (4/3)^N dev @0.49 | +3.3927 dB | **+0.0062 dB** | +0.0025 dB |
| ripple [0.005,0.30] base/cand | 0.001/0.001 | 0.034/**0.025** | 0.016/**0.012** |
| cand 0.1 dB edge | 0.3526 Fs_in | 0.4897 Fs_in | 0.4945 Fs_in |

#### 2.7.2 FIR 絶対基準（P0-D floor・v2.1 表維持）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|----------------|---------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

#### 2.7.3 Image rejection（**D1 は補正軸が authoritative・R8-1**）

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

**D2（gate 帯域 [0.005, 0.40] で base==cand ≤0.01 dB）**: v2.1 表維持。@0.45 は記録のみ。

**D3**: full-chain worst spur = 窓 sidelobe −98〜−101 dB・base==cand 差 = 0.00 dB。

### 2.8 教訓（v2.2 で 26 を追加）

1〜25: v1.8〜v2.1 のまま（oracle 依存禁止 / 仮説と欠陥の分離 / reset() 必須 / FP フロア gate 外 / FFT bin は argmax 検証など）。

26. **【R8-1】authoritative 成果物と補正済み測定定義の乖離を放置しない**: モデル/結果ファイルが旧軸の D1 を出力し続けていると、後続セッションが「モデルが全部 authoritative」と誤読する。補正した測定定義はモデル本体へ反映するか、結果ファイル側に「破棄済軸」ラベルを明記する。**どちらか一方だけでは不十分**（本 v2.2 では双方を指示）

### 2.9 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止 / calibration 禁止 / engine-fit 0.75^N unchanged。

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（本セッション再確認）

- 根本原因: `setAutoGainStagingEnabled`（AudioEngine.h:1416-1432）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425）。既定 staging=true（:2626）
- eq モード誤順: **:1799 configureProbeFlatEQ → :1803 setAutoGainStagingEnabled(false)**（AGC が再 ON される）
- 正順（eqdiag 等）: **:1833 staging false → :1834 configureProbeFlatEQ**
- `configureProbeFlatEQ` :1072 内 `setEQAGCEnabled(false)` :1081
- **修正案 A（呼出順を eqdiag パターンへ揃える）を維持**
- 検証: `AudioEngineHarness.exe --buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（AGC 強制でも結果不変）

### 3.2 F-3: EQ dry/wet 混合（記録のみ・案 D）

- production 潜在欠陥として記録。遷移時のみ。定常 B-1 の原因ではない
- 修正は別 work item（scope discipline）

### 3.3 F-4: `ir` / `irwet` 用語（R5-8 継承・行番号再確認）

- **`ir`**: IR ロード後も `setConvolverBypassRequested(true)`（:1745）維持 = **dry ベースライン測定モード**
- **`irwet<digit>`**: :1763 分岐・**:1778 `setConvolverBypassRequested(false)`** = wet 畳み込み対照（OS 倍率は末尾 1 桁）
- 未知モード fail-closed :1708-1710（`[BUZZ] FAIL: unknown rigcheck mode`）
- **修正案 B（記録のみ）維持**

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状（本セッション再実測）

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

## 5. §5 別課題（R-2 を R8-9 で更新）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | 数値 parse 未捕捉例外 → fail-closed 化。対象: BassBuzz 直接 stod/stoi/stof（:1582/:1583/:1590/:1591/:1592/:1614 系）+ parseHcIdx/parseLcIdx 本体（:1562/:1567）+ 呼出（:1606-:1611）+ **PPIT stoi :1136/:1146/:1149/:1157/:1166（:1149 が R8-9 で追加）**。仕様: try/catch + `[BUZZ] FAIL: invalid numeric` + 非ゼロ終了 + regression test | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=` silent-ignore（設計上ステップ固定） | 次（極小） |
| **B-3** | timestamp-based capture（現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合 | 中 |
| **D-1** | build identity gate M1/M2 | 高 |

---

## 6. 推奨する実行順序（v2.2）

```
[Step 0] 現状固定
  - HEAD 8f127bfe / production src/ 未 commit 0 / numstat 再確認
  - モデル再現条件: tr -d '\r' + python/numpy/scipy 版記録
  - D1 は補正軸（R8-1）を使うことを測定系仕様に明記

[Step 1] §2 B-1 Phase 0（production 変更 0）
  - G-0 契約承認 → Baseline → REF-FIDELITY → Candidate → 周波数/block/SoftClip/float-double

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は image invariance として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §3 F-2（test-only・案 A）
  - :1799/:1803 を :1833/:1834 パターンへ
  - rigcheck=eq で staging=0 eqAGC=0 を確認
  - F-3/F-4 記録のみ

[Step 4] §4 F-1（A→B→B'→C・failureReason 主判定）

[Step 5] §1 O-1〜O-14（ユーザー判断）
  - O-14 .mcp.json を判断対象に追加
  - O-13 は 30 件ベースで承認版+台帳 commit

[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option → flag ON 検証 → Phase 3-A…4

[Step 7] 別課題: R-2（PPIT :1149 含む）→ R-1 → B-3, D-2, D-1
```

---

## 7. 検証計画

### 7.0 閉形式モデル再現

```bash
python3 doc/work113/model_polyphase_20260920.py > /tmp/rerun.txt
tr -d '\r' < doc/work113/model_polyphase_20260920_results.txt > /tmp/expected.txt
diff /tmp/expected.txt /tmp/rerun.txt
# 期待: DC/§2.7.1/§2.7.2/D3 一致・D2 フロアは ±0.01 dB まで許容（91〜92/96 級）
# 環境記録: python3 / numpy / scipy バージョン
# D1 行の差分は「旧軸の再現」であり補正軸 D1 の検証ではない（R8-1）
```

### 7.1 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| F-2 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS・PVIT 削除 |
| B-1 Phase 0/1 | AudioEngineHarness `[OS_DIRECT] roundTripGain=` | OFF≈0.75^N / ON≈1.0 |
| Shadow/REF | Phase 0-1b/0-2 test-only | REF bitwise → Cand DC=1.0 |
| 静的解析 | Windows cppcheck（指摘 0・exit=0）/ clang-tidy | 既知の portability のみ |

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

---

## 9. 影響度まとめ

| 区分 | 影響 | 影響度 | 段階リリース |
|------|------|--------|--------------|
| O-1〜O-14 | リポジトリ/環境記録 | 極小 | 任意 |
| B-1 案 E | 全オーディオ経路（最大 +7.4963 dB） | **極大** | 必須 |
| F-2 | test-only | 小 | 不要 |
| F-3/F-4 | 記録のみ | なし | 不要 |
| F-1 | test-only | 中 | 不要 |
| R-2 等 | CLI/harness | 極小 | 不要 |

---

## 10. 監査ログ・参照（本セッション再実測）

| ファイル | 確認 | 結果 |
|----------|------|------|
| CustomInputOversampler.h | :21-22 / :32 reset / :71 clearAll / :87 centerCoeff=0.5 | ✓ |
| CustomInputOversampler.cpp | prepareStage / interpolate :557 convValue×2・center ×2 無し / decimate / reset :452 / clearAll :469 / processUp :725 / processDown :785 | ✓ |
| AudioEngine.h | staging setter :1416-1425 / default true :2626 / getEQProcessor :1292 / setEQAGCEnabled :1306 | ✓ |
| AudioEngine.Processing.Latency.cpp | static_assert :6-7 / groupDelay :30 `// up + down` | ✓ |
| DSPCoreLifecycle.cpp | prepareSingleStage(31,90) :188/:261 | ✓ |
| DSPCoreFloat.cpp | processUp :405 / processDown :413 | ✓ |
| DSPCoreDouble.cpp | processUp :505 / processDown :513 / kOutputHeadroom :593 | ✓ R8-10 |
| RuntimePublicationValidator.h/.cpp | h105/cpp211 / reason :13 / failureReason :24 / private :92 / check :101 / cpp:169 | ✓ |
| BassBuzzMeasurement.cpp | F-2 :1799/:1803 vs :1833/:1834 / irwet :1763/:1778 / parse :1558-1615 / configureProbeFlatEQ :1072/:1081 | ✓ |
| PublishPipelineIntegrationTests.cpp | main :1085 / 宣言 :1116 / 呼出 :1223 / stoi :1136/:1146/:1149/:1157/:1166 | ✓ R8-9 |
| PublicationValidatorIsolationTests.cpp | 519 行 / TEST 38 / checkNoConflict 9 | ✓ |
| CMakeLists.txt | option :40 / NUC_DEBUG_GUARDS :52-58 / JUCE :1043 / juce_add_gui_app :1062 / ALL_SOURCES :1133 / harness :1899 / **add_executable 39** | ✓ |
| juce_Oversampling.cpp | up :185 `2 * samples[i]` / down :228 | ✓ |
| TruePeakDetector.cpp/.h | interpolate のみ / attenuation 100.0 / stage max(15,taps/2) / center=0.5 | ✓ R8-11 |
| model_polyphase results | 96 行・D1 旧軸ラベル :67・DC 12 桁一致・D2 フロア差 5 行 | ✓ R8-1/R8-2 |
| cppcheck | Windows CLI exit=0・指摘 0 | ✓ R8-5 |
| git | numstat 上記 / HEAD 8f127bfe / production clean | ✓ R8-3/R8-7/R8-8 |
| AudioEngineHarness.exe | 41,137,664 bytes | ✓ |

---

## 11. ユーザー判断待ち項目（v2.2）

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
| O-13 work113 md（30） | 承認版+台帳 commit / 全 commit / 触らない | **承認版+台帳 1 commit** |
| **O-14 .mcp.json（新）** | 触らない / commit（環境記録） | **commit（環境記録）** |
| G-0 契約 | 承認 / 修正 / 却下 | **承認（v2.1 版 E1〜E5）** |
| B-1 修正案 | E（candidate） | **E（仮説）** |
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

### v2.2 最終判定表

| 項目 | 判定 |
|------|------|
| §1 O-1〜O-14 | **GO 候補（O-14 追加）** |
| §3 F-2 | **GO 候補（案 A）** |
| §3 F-3 / F-4 | **記録継続** |
| §4 F-1 | **GO 候補（failureReason 主判定）** |
| §5 R-2 | **仕様確定・将来 work item（PPIT :1149 含む）** |
| B-1 原因分析 | **GO（defect 客観確定）** |
| B-1 Phase 0 | **GO（測定仕様 v2.2 適用・D1 補正軸 + argmax 必須）** |
| B-1 Phase 1+ | **HOLD** |
| B-1 案 E | **有力仮説（確定表記禁止）** |
| モデル D1 旧軸 | **破棄済参考・authoritative は本書 §2.7.3（R8-1）** |
| 技術的未確定 | **0（R8-12）** |
| TruePeakDetector | **後続候補第三順位** |
| compile-time flag | **`CONVOPEQ_CORRECT_POLYPHASE_GAIN` + `#if`** |

---

## 12. 本セッションの検証エビデンス（2026-09-21 夜）

### 12.1 モデル

- 再実行環境: WSL python3 **3.14.4 / numpy 2.5.3 / scipy 1.18.1**
- CRLF 正規化後 **91/96** 一致。差分 5 行はすべて D2 フロア ±0.01 dB
- DC / 係数検証 / §2.7.1 系 / §2.7.2 / D3 は一致
- **D1 は旧軸出力のまま（R8-1）** → 計画側 §2.7.3 が authoritative

### 12.2 ソース

- B-1 欠陥コード（`centerValue *= 2.0` 不在）を AiDex + rg で再確認
- F-2 / F-4 / F-1 / SoftClip / Latency / JUCE / CMake 39 を再実測し v2.1 と一致
- R-2 は PPIT **:1149** を追加確定
- SoftClip Double 経路 :505/:513 を追加確定

### 12.3 ツール

- rtk（WSL）/ context-mode / serena / AiDex / semble / cocoindex / headroom MCP / cppcheck を使用
- AiDex セッション開始・シグネチャ取得。serena overview で CIO メソッド列を確定

### 12.4 未確定事項の最終棚卸し（R8-12）

| 種別 | 状態 |
|------|------|
| v2.1 §12.4 技術要確認 4 件 | 既に確定済み（IIR3@0.49 +0.0062 / P0-C 余裕 / silence 1e-20 / 0.45 FP） |
| 本セッション追加の技術事項（モデル D1 軸） | **確定（R8-1 方針）** |
| production 追加欠陥 | **検出なし**（B-1 は既知・他は記録/将来 work item） |
| ユーザー判断 | O-1〜O-14 / G-0 / Phase GO / F 方針 — **技術調査の対象外** |

---

## 12.5 v2.2 で追加した監査・確定事項（R8）

- **R8-1** authoritative モデル D1 が旧軸のまま → 補正 D1 は本書 §2.7.3 を authoritative とし、モデル/結果への反映（または破棄ラベル）と Phase 0 の argmax 検証を義務化
- **R8-2** モデル再現 91/96（D2 フロア）— R6-1 の環境依存を追加実証
- **R8-3** O-14 `.mcp.json` +33/−23 を新設
- **R8-4** 主要行番号・構造の全件再監査（v2.1 一致）
- **R8-5** cppcheck Windows exit=0 再確認
- **R8-6** work113 untracked md 30 件
- **R8-7/R8-8** production clean / HEAD・push 状態
- **R8-9** R-2 に PPIT :1149 を追加
- **R8-10** SoftClip Double 経路 :505/:513
- **R8-11** TPD パラメータ再確認
- **R8-12** 技術的未確定 0 / 残りはユーザー判断のみ
- **§2.8.26** 教訓追加（authoritative 成果物と測定定義の乖離禁止）

---

*本書は v2.1（remediation_plan_20260921_v2.1_revised.md）の再改訂版 v2.2 である。*
*v2.1 本文は参照文書として維持し、**R8-1〜R8-12 および本書が矛盾する箇所では本書を優先する**。*
*R5〜R7 は本書でも有効。中間エビデンスは `remediation_v22_intermediate_20260921.md`。*

**v2.1 で新規追加した監査・確定事項（維持）**: R6-1〜R6-7 / R7-1〜R7-4（D1 軸補正・v1.7 再格付け・文献検証・P0-F 精密化）/ P0-E image invariance / P0-I 3 分割 / 実行順序 R5-3 / 案 E=仮説分離。
