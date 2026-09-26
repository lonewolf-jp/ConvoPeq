# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v2.7）

- **版**: v2.7（v2.6 の再改訂・**R13-1〜R13-9** を追加確定）
- **前版**: `doc/work113/remediation_plan_20260922_v2.6_revised.md`（R12-1〜R12-12）
- **本改訂監査実施日**: 2026-09-21（同一作業ツリー・CommandCode 環境）
- **基準ソース**: HEAD = `8f127bfe`（親 `c4a08171`）。production `src/*.cpp/h`（tests 除外）の未 commit 差分 **0 件を再実測で再確認**（`git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` → 出力なし）
- **★ source baseline 更新**: `ConvoPeq.md` は **2026-09-21 16:16:35 生成・5,359,772 B・`--check` → FRESH（NEWER_SRC_COUNT=0）** を本版の authoritative snapshot とする（v2.6 の 15:52:48 版から更新されている。以後の引用は本 snapshot 基準）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `_results.txt`（本版で**再実行し PASS を再確認**）
- **v2.7 の位置付け**: v2.6 の GO/HOLD・案 E（candidate hypothesis）・Phase 0 の 5 構造・R10/R11/R12 確定事項は**原則維持**。本改訂は (a) v2.6 内部の O-19 記載矛盾の解消（§2.1「要確認」vs §6「解決済み」→ **解決で確定**・実物確認済み）、(b) baseline タイムスタンプの更新、(c) 全 P0 数値の独立再計算による裏付け（R13-1〜R13-3）、(d) 二重補償・F-2/F-1/CMake の全ツール横断再検証（R13-4/R13-5）、(e) 外部文献による音響工学的裏付け（R13-7）、(f) 新規棚卸し O-20（isBadSample 閾値不整合・non-blocking）、を扱う
- **案 E は有力仮説（candidate hypothesis）のまま**。defect（DC = 0.75^N）は本版でも再実行で再現。是の設計判断は G-0 承認まで「確定」と表記しない（R5-4 継承）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**（production 側規約。test 側診断出力は §R12-7）
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）
- **R5〜R13 は本書でも有効（是正点を除く）。R13 と矛盾する v2.6 の箇所では本書を優先する**

---

## 0. v2.6 → v2.7 改訂サマリ

### 新規確定（R13-1 〜 R13-9）

| ID | 区分 | 内容 | v2.7 での確定 |
|----|------|------|----------------|
| **R13-1** | P0-F 数値の独立再計算 | `D = Σ_s (taps[s]−1)/2^(s+1)` を本セッションで再計算: S1 **15.00** / 511単段 **255.00** / IIR3 **290.25** / LP3 **582.25** — R11-2・R12-1 と全桁一致。peak gate [15,16]/[255,256]/[290,291]/[582,583] も成立 | P0-F 基線を三重裏付け（コード同型式・v26 スクリプト再実行・本版再計算）で確定 |
| **R13-2** | DC 利得の独立再計算 | base `0.75^N`・cand `1.0`・`(4/3)^N` dB を N=1..8 で再計算: N=1 **+2.498775** / N=2 **+4.997549** / N=3 **+7.496324** / N=4 **+9.995099** / N=8 **+19.990198** dB。authoritative モデル再実行も base=`0.75^n`・cand=`1.0` を 6 design 全一致で再現 | §3 基線を確定 |
| **R13-3** | E-1c の独立再検証 | D1 構造定数 `20·log10(1/3)` = **−9.542425 dB**・絶対 gate **−9.442 dB** を再計算。authoritative モデル経由の独立 30 点判定（6 design × f̂≤0.35）で **FAIL 0・最小 margin 36.7 dB** を再現（v26 スクリプトの 42 点判定とも一致） | E-1c（絶対形）を確定維持 |
| **R13-4** | 二重補償の再 census | production（tests 除外）の `0.75` 検索 → **5 件すべて無関係**（SpectrumAnalyzerComponent.h:116 コメント・AudioEngine.Retire.cpp:26 wet gain jlimit・EQControlPanel.cpp:493 alpha・PsychoacousticDither.h:219 係数・RuntimeHealthMonitor.h:86 閾値）。`compensate/makeup` 検索 → Output Makeup UI と IR scale のみで 0.75^N 補償なし。R12-5 を**別手段（二重検索）で再確認** | Phase 3-A 受入条件「補償不存在の再確認」を維持 |
| **R13-5** | F-2/F-1/CMake の再検証 | F-2: `:1799 configureProbeFlatEQ` → `:1803 staging false` の誤順と `:1833→:1834` の正順を再確認（`:1822` コメントも正順を文書化済み）。F-1: `checkNoConflictingTransitions` = h:101・cpp:38/:169・既存 test 使用を確認。CMake: `CONVOPEQ_CORRECT_POLYPHASE_GAIN` は **CMakeLists.txt に未存在**（未適用を再確認）。CIO は ConvoPeq target の `target_sources`（:1196/:1243/:1282）に含有 | F-2 案 A・F-1 分類・flag 未適用を確定維持 |
| **R13-6** | 欠陥局在の多手段再確認 | `:557 convValue *= 2.0` のみ（ast-grep 1 hit・`centerValue *= 2.0` は 0 hit）。`:563-564` 出力・`:658` decimate 累加（×2 なし＝DC 1.0 正規）・prepareStage（:291/:292/:335/:348/:350/:367/:368）・`reset()`/`clearAllStages()` を再照合。AiDex・serena・semble・cocoindex（ccc）・graphify・tgrep の全系で同一局在に収束 | B-1 局在を確定維持 |
| **R13-7** | 外部文献の裏付け | MathWorks FIR Halfband Interpolator 公式文書: **「halfband 補間フィルタの係数は出力パワー保存のため補間係数 2 でスケールされる」** — 案 E（両 polyphase 位相への ×2 対称適用）の直接の理論的裏付け。decimator との対比記述も decimate 側 ×2 なし（正規）と整合。AES Wang-Reiss（decimation filter 群遅延）・KVR（linear-phase FIR latency）が P0-F の群遅延定数性を支持 | 案 E の理論的妥当性を外部文献で裏付け（§5） |
| **R13-8** | isBadSample 意味論の厳密化 | `CustomInputOversampler.cpp:24` scalar 版: NaN/Inf **+ \|x\| > 2^53（≈9.007e15）** を不良判定。AVX2 版 `isBadSampleV`（:40-48）: NaN **+ \|x\| > 1e20**。両者は NaN/Inf では一致するが、**巨大振幅域（9e15, 1e20] で判定が divergence** する潜在不整合を発見。P0-I の NaN/Inf 契約には影響なし（両版とも NaN/Inf は確実に検出） | 新規棚卸し **O-20**（non-blocking・将来対応）に計上。P0-I 契約は変更なし |
| **R13-9** | O-table・baseline の刷新 | ConvoPeq.md 16:16:35 FRESH・O-19 解決確定・O-20 新規・diffstat 再実測（§1） | §1 を本版基準に更新 |

### v2.6 内部矛盾の是正（本版で解消）

- v2.6 §2.1 O-19 行は「D（HEAD からの削除・出所不明）・ユーザー確認のうえ復元推奨」としつつ、§6 Step 0 は「O-19 解決済み（2026-09-21）: `build_identity_gate.py` は `src/tools/` 復元・`check_layout_offsets.py` は `tools/` 移動のまま確定」と記載 — **同一文書内で矛盾**。本セッションで実物を両方確認（`src/tools/build_identity_gate.py` 存在・`tools/check_layout_offsets.py` 存在）したため、**「解決」で確定**し O-19 行を更新する（R13-9）。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-20・v2.7 時点の実測棚卸し）

### 1.1 実測 diffstat（2026-09-21 16:30 時点・`git diff HEAD --numstat` + `git ls-files --others --exclude-standard`）

| ID | パス | 実測 | v2.6 との差分 | v2.7 推奨 |
|----|------|------|----------------|-----------|
| **O-1** | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634/−2** | 一致 | **Phase 0 characterization 完了まで凍結（commit しない）→ 完了後に cleanup/分割して commit**（R12-10 維持） |
| **O-1b** | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8/−0** | 一致 | O-1 と同一運命（凍結） |
| **O-2** | 台帳更新 | 独立 commit | — | 独立 |
| **O-3 / O-12** | `ConvoPeq.md` | **16:16:35 再生成・5,359,772 B・FRESH（NEWER_SRC_COUNT=0）** | **更新（v2.6 は 15:52:48・5,361,988 B）** | **commit しない**（再生成資産・`--check` FRESH を維持） |
| **O-4** | `Testing/Temporary/CTestCostData.txt` | **D**（削除） | 一致 | 現状維持 |
| **O-5** | `.opencode/opencode.json` | untracked | 一致 | 触らない |
| **O-6** | push | ahead 2（`8f127bfe`・`c4a08171` は commit 済） | 一致 | **ユーザー手動** |
| **O-7 / O-10** | `AGENTS.md` | **+78/−3**（Freebuff 節追加を含む） | **更新（v2.6 は +71/−3）** | 環境記録として commit 可。R11-7 の Maxima 誤記修正（§9）を同時に適用 |
| **O-8** | `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` | M（追跡済みバイナリ） | 一致 | commit しない → 将来 `.gitignore` + `git rm --cached` |
| **O-9** | tool-inventory 2 件 | untracked | 一致 | 触らない（ユーザー判断） |
| **O-11** | `doc/work113/residual_tasks_20260919.md` | **+71/−1** | 一致 | 台帳として commit 可 |
| **O-13** | `doc/work113/*.md` + model py/results untracked | 一致（本版追加で v2.7 1 件） | 承認版+台帳+モデルを 1 commit |
| **O-14** | `.mcp.json` | **+33/−23** | 一致 | commit（環境記録） |
| **O-15** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | untracked（102 行） | 一致 | O-13 に併同可。**commit 前に §3.9 契約を反映した版へ** |
| **O-16** | `evidence/epoch_reclaim_audit.json` | **M・Bin** | 一致 | **C0 で追跡外化**（R11-8 確定・§1.3） |
| **O-17** | `doc/WorkBuddy_AI_Desktop_Toolchain.md` | untracked | 一致 | O-9 と同一方針 |
| **O-18** | `output_sourcecode_markdown.py` | **+11/−4** | 一致 | **commit 推奨**（R12-6・独立 commit）。本版 snapshot は修正版で生成済み |
| **O-19** | `src/tools/build_identity_gate.py` / `tools/check_layout_offsets.py` | **解決で確定**: `src/tools/build_identity_gate.py` 実在（`ls` 確認）・`tools/check_layout_offsets.py` 実在（`tools/*.py` 一覧で確認）。v2.6 §2.1 と §6 の矛盾は本版で解消 | **解決（R13-9）**。D-1（build identity gate）は R12-7 の BUILD-ID gate と統合のうえ `src/tools/build_identity_gate.py` を継続使用 |
| **O-20** | `src/CustomInputOversampler.cpp:24-48`（新規） | isBadSample scalar（2^53 閾値）vs AVX2 版（1e20 閾値）の不整合（R13-8）。production 変更を伴うため本版は棚卸しのみ | **将来対応（non-blocking）**。Phase 0 の P0-I gate には影響なし |

### 1.2 O-16 と Phase 0 の衝突（R11-8 確定方針の維持）

Phase 0 測定が tracked `evidence/` を書き換える構造的欠陥は未解決（追跡外化は未適用）。手順は §1.3 に固定済み。

### 1.3 evidence/ 追跡外化手順（R11-8 + R12-10 の commit 分離）

```bash
# C0: evidence policy 単独 commit（Phase 0 測定 commit と混ぜない）
echo "evidence/" >> .gitignore
git rm -r --cached evidence/
git add .gitignore
git commit -m "chore(evidence): untrack generated verification artifacts"
# 以後、測定出力は --buzz-out= で tmp/（.gitignore 済み）へ
# C1: Phase-0 shadow/reference 実装 commit（O-15 + §3.9 契約版）
# C2: characterization 結果 commit（測定 log・CSV は tmp/ 起点で添付判断）
```

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 利得欠陥

### 2.1 確定している事実（v2.6 §3.1 継承・本版で再実測）

```
主因        interpolateStage の center 位相に ×2 が無い（polyphase 利得規約の不対称）
欠陥局在    src/CustomInputOversampler.cpp :557（convValue *= 2.0 のみ・ast-grep 1 hit）
            :563-564（center に ×2 なし・centerValue *= 2.0 は 0 hit）
            decimateStage :658（centerCoeff * centerSample + conv tap 累加・×2 なし＝DC 利得 1.0 で正規）
round-trip  DC: base 0.75^N / cand 1.0（authoritative モデル再実行 + 本版再計算で再確認）
段ごとの和  conv 分岐: Σcoeffs 0.5 → ×2 → DC 1.0（正規）／center 分岐: 0.5 → ×2 なし → DC 0.5（不対称）
利益        (4/3)^N = +2.498775 / +4.997549 / +7.496324 dB（1/2/3 段・本版再計算）
画像        up 出力の像/トーン = 1/3 = −9.542425 dB（構造定数・R11-4 確定・本版再計算）
係数        FIRsum 1.0 / center 0.5 / convSum 0.5（全 6 design・モデル再実行で再現）
契約不整合  isLinearPhaseFIR / isSymmetricUpDown（h:21-22）・Latency.cpp:6-8 static_assert と矛盾
production  変更 0 / commit 0（本版で git diff 実測・tests 以外 0 件を再確認）
```

**peak / mean の帰属不能性（R10-3・E-8 確定表現の維持）**: up 出力は位相交互 `1.0, 0.5, …`。peak 系で「up 1.0/down 0.75」、mean 系で「up 0.75/down 1.0」— どちらも同じ階段信号の別記述で因果の根拠にならない。因果は分岐係数和の不対称（1.0 vs 0.5）で確定。

### 2.2 根本原因の行番号（v2.7 再実測・R13-6）

| 箇所 | 内容 | 実測手段 |
|------|------|----------|
| `prepareStage` :287-390 | `taps=jmax(3,taps\|1)` :291 / `centerTap=(taps-1)/2` :292 / halfband ゼロ化 :319-323 / `sum→1.0` :325-333 / `rawCoeffs[centerTap]=0.5` :335/:348 / 非 center を 0.5 へ :336-347 / `convCount` :350 / convCoeffs 構築 :360-365 / `centerCoeff` :367 / `centerDelayInput` :368 | Read + rg（AiDex query でも同一箇所） |
| `interpolateStage` :492-568 | **:557 `convValue *= 2.0`（ast-grep で 1 hit のみ）** / :558-559 denorm / **:563-564 出力（`centerValue *= 2.0` は ast-grep 0 hit＝不在確定）** | ast-grep + rg + tgrep（`centerValue *= 2.0` 不在を 3 手段で確認） |
| `decimateStage` :570-723 | silence fast path :583-613 / :658 `acc = centerCoeff * centerSample` + conv 累加（×2 なし＝DC 1.0 正規） | Read |
| `processUp` :725- / `processDown` :785- | hardFallback / corruption clear | 継承 |
| `reset()` :452-467（atomic 3）/ `clearAllStages()` :469-484（atomic 1） | Phase 0 は `reset()` 経由必須（R4-3） | 継承 |
| `prepareSingleStage` :392-450 | SoftClip 局所 OS（Lifecycle :188/:261 で `(31, 90.0, internalMaxBlock)`） | 継承 |
| `dotProductAvx2` :159-216 | conv 専用 SIMD・center 位相は非依存 | serena symbols + graphify |
| `isBadSample` :24 / `isBadSampleV` :40-48 | scalar は NaN/Inf + \|x\|>2^53 / AVX2 は NaN + \|x\|>1e20（**R13-8 の divergence**・O-20） | Read（新規精読） |

### 2.3 5 系統独立再現 + 数値計算裏付け（R10-1 + R11-3 + R13-1/R13-2）

production バイナリ / authoritative モデル / NumPy 逐語転写 / Octave 11.3.0 / Maxima 5.50.0 — 全系統が base 0.75^N・cand 1.0・Δ+2.498775 dB(N=1)・image/tone 1/3 に一致（v2.5 §2.3 表を継承）。本版は以下を追加実測した:

- `doc/work113/model_polyphase_20260920.py` **再実行**: 係数 6 design 全て FIRsum=1.0/center=0.5/convSum=0.5・DC round-trip base=`0.75^n`/cand=`1.0`（r=2/4/8・両 preset）・full-chain dev +0.0000 dB（passband）を再現
- `tmp/v26_p0f_gate_calc.py` **再実行**: latency 4 構成・peak gate・E-1c 42 点・DC N=1..8 の全 PASS を再現
- 本版独立再計算（R13-1/R13-2）: 上記 §2.1 の値と全桁一致

### 2.4 ライブ帰因テーブル（R10-2 継承・引数なし実行の制約）

`[OS_DIRECT]/[EQ_DIRECT]/[OF_DIRECT]` の全 row は v2.5 §2.4 を継承。**唯一の桁違い損失は Oversampler（0.75^N）**・EQ は ratio=1.000000・OutputFilter は −0.110 dB 一定。`rt = up × dn` 成立・preset 非依存・**P0 gate は rt のみ**。再現には引数なし実行が必須（`--buzz*` 引数があると `PublishPipelineIntegrationTests.cpp:1123` の派遣で :1223 に到達しない）。

### 2.5 修正案と DESIGN-CONTRACT-A（変更なし）

案 A〜D 不採用・**案 E 有力仮説**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内（:557 の直後）
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E（両 polyphase 位相への ×2 対称適用・candidate）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;   // 既存 :559
```

**構造的恒等式（R5-9 継承）**: `h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`（c: base 0.25 / cand 0.5）。passband で cand/base = (4/3)^N。

**外部文献の裏付け（R13-7）**: MathWorks `dsp.FIRHalfbandInterpolator` 公式文書は「halfband 補間フィルタの係数は**出力パワー保存のため補間係数 2 でスケールされる**」と明記し、decimator との対比で interpolator 側の ×2 を正規とする。これは案 E（center 位相にも ×2 を対称適用）と**同一の規約**であり、欠陥（center の ×2 不在）が規約違反であることを外部権威が支持する。decimate 側に ×2 が無いこと（:658 の正規性）とも整合する。

### 2.6 CMake / flag（R10-5 + E'-4 確定・R13-5 で未適用を再確認）

- `option(CONVOPEQ_CORRECT_POLYPHASE_GAIN "Correct polyphase gain convention (B-1 案E)" OFF)` を option 群（:40 近傍）に追加
- `add_compile_definitions(CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${...}>)` を `add_subdirectory(JUCE)`（:1043）後・`juce_add_gui_app(ConvoPeq`（:1062）前に配置
- 到達確認: CIO.cpp を compile するのは ConvoPeq（`target_sources` :1282・リスト内 :1196/:1243）と AudioEngineHarness の 2 論理ターゲット。**本版再検索で `CONVOPEQ_CORRECT_POLYPHASE_GAIN` は CMakeLists.txt に 0 hit＝現状未適用を再確認**
- C++ は **`#if`**（`#ifdef` 禁止）。既定 OFF / runtime flag 不採用 / **現状未適用**
- **R12-7**: 測定出力の `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` 行を Phase 1 の harness 変更で必須化（test-only・`add_compile_definitions` により macro は常に 0/1 で定義済み）

### 2.7 Phase 0 characterization（v2.4 改定 + E-1c 追加・変更なし）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）+ T-1/T-2/T-3 をユーザーが明示承認
0-1  Baseline record-only: round-trip DC = 0.75^N ±1e-6（invariant のみ gate・up/dn 分割は診断ログ）
0-1b REF-FIDELITY: Shadow == production（bitwise・reset() 経由・§2.9 の実装契約）
0-2  Shadow Candidate E: round-trip DC = 1.0 ±1e-6
0-3  周波数（D1/D2/D3）— D1 は base 側 gate + E-1c cand gate（§2.8）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）+ candidate マクロ 2 系統一致性 assert
0-5  SoftClip local OS（float + double・R12-4 の数値契約）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8 / 相対誤差記録のみ）
```

**D1 定義（R10-4 + E'-5 継承 + R12-2 + R13-3）**:
- 軸: `D1 = |Y_up(image bin)| / |Y_up(tone bin)|`、up 出力長 2N、tone bin = f̂·N、image bin = N−f̂·N（補正軸）
- **E-1（gate・base）**: `D1_base = −9.542425 ± 0.01 dB`（構造定数・手法不変 81 測定で max 差 0.0036 dB）
- **E-1c（gate・cand）**: `D1_cand ≤ −9.442 dB`（= −9.542425 + 0.1）を **f̂ ∈ [0.05, 0.35] で gate**。f̂ = 0.40 / 0.45 は**遷移帯のため記録のみ**。本版の独立 30 点判定でも FAIL 0・最小 margin 36.7 dB を再現（R13-3）
- E-1b（記録）: cand D1 を矩形窓・Hann+trim8 の 2 条件で併記
- tone/image bin は expected 主報告・argmax ±8 bin は検証ログ（R9-2 継承）
- final alias rejection は P0-D が担当（R12-3 の責務分離）

**Shadow Reference**: §2.9（T-1）。骨格のみでは gate 実行不能（R10-6/R11-9 確定の維持）。

**D2 / D3**: v2.2/v2.3 のまま。D2 gate [0.005, 0.40] base==cand ≤0.01 dB・0.45 記録のみ。D3 worst spur −98〜−101 dB（窓 sidelobe）・base==cand 差 0.00 dB。

### 2.8 B-1-P0 GATE（判定表・v2.7 維持 + R13 裏付け列）

| ID | 判定対象 | 責務（R12-3） | 条件 | v2.7 裏付け |
|----|----------|---------------|------|-------------|
| G-0 | 契約 | — | DESIGN-CONTRACT-A（E1〜E5）+ **T-1・T-2・T-3** 明示承認 | T-3 追加（v2.6） |
| BUILD-ID | binary identity | run-level | **全測定 log の先頭に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1`**。欠落 = 測定無効（R12-7・Phase 1 から必須・Phase 0 は record） | 維持 |
| G-BL | Baseline sanity | 全経路 | round-trip DC = 0.75^N ±1e-6 のみ gate。up/down 分割は診断ログ | R13-2 再計算 |
| REF-FIDELITY | Shadow==production | shadow fidelity | bitwise（tolerance 0）・§2.9 の 3 対象・`reset()` 経由。**注: production 正しさの gate ではない**（R12-8） | 維持 |
| P0-A | Candidate DC | 全経路 | round-trip 1.0 ±1e-6（必要条件） | モデル再実行 |
| P0-B | 低域 | 全経路 | 50 Hz / 1 kHz unity ±0.1 dB・4 経路 | モデル再実行（−2.4988/−0.0000 dB） |
| P0-C | passband ripple | 全経路 | **[0.005,0.30] max−min ≤0.05 dB（dense）**・周波数軸は **Fs_in（base レート）正規化**（R12-3） | モデル再実行（ripple[0.01,0.3] 0.001 dB） |
| P0-C' | differential | 全経路 | (4/3)^N ±0.05 dB（IIR3/LP3 f≤0.45・S1 f≤0.30）— attribution criterion | モデル再実行（dev +0.0000 dB） |
| **P0-D** | **decimator alias rejection** | decimator | stopband: alias ≤ −(A−10) dB @[t_end, 0.5]（per-design 絶対基準 §2.8.3） | 維持 |
| **P0-E** | **interpolator image rejection** | interpolator | **E-1: base D1 = −9.5424 ±0.01 dB（gate）** / **E-1c: cand D1 ≤ −9.442 dB @f̂≤0.35（gate・R13-3 で独立再検証）** / E-1b: cand D1 2 条件記録 / E-2: D2 [0.005,0.40] base==cand ≤0.01 dB / E-3: 0.45 記録のみ | **R13-3** |
| P0-F | latency | 全経路 | peak ∈ [floor(D), floor(D)+1]。**D = Σ_s (taps[s]−1) × (baseRate / stageRate(s))・stageRate(s) = baseRate × 2^(s+1)**（`Latency.cpp:29-31` 同型）。**作業例（IIR3）: 510/2 + 126/4 + 30/8 = 255 + 31.5 + 3.75 = 290.25**。基線: S1 15（dev 0）・511単段 255（dev 0）・IIR3 291（dev +0.75）・LP3 583（dev +0.75）・SoftClip 単段 15（`kSoftClipLatencyBaseRateSamples`・Latency.cpp:120）。base==cand も gate | **R13-1（三重裏付け）** |
| P0-G | float/double | 2 ペア | maxAbsErr ≤5e-7 / RMS ≤5e-8 + **max 相対誤差（\|ref\|>1e-6 のみ）record**（R12-12） | 維持 |
| P0-H | block/reset | 全経路 | partition bitwise・reset contract（atomic 3 / atomic 1）+ candidate マクロ 2 系統一致性 assert | 維持 |
| P0-I | SoftClip 安全性 | SoftClip 2 経路 | **I-a: DC = 1.0 ±1e-6** / **I-b（数値契約・R12-4）: 出力 NaN/Inf 1 件でも FAIL・baseline に無い飽和イベント FAIL** / **I-c（記録）: peak・RMS・THD・THD+N・saturation event 数・NaN/Inf 数・動作点**（`prepareSingleStage(31,90.0)`・Lifecycle :188/:261）。scalar/AVX2 の巨大振幅閾値差（R13-8）は NaN/Inf 検出に影響しないため本契約の対象外（O-20 に分離） | **R13-8 で意味論を厳密化** |

**全 PASS → Phase 1 eligibility**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

#### 2.8.1 passband ripple 基線（v2.4 実測・モデル再実行で整合）

| config | 帯域 [0.005,0.30] | base | cand |
|--------|------------------|------|------|
| S1 (31/90) N=1 | gate 帯 | 0.001 | 0.001 |
| IIR3 (511/127/31) | 〃 | 0.034 | 0.025 |
| LP3 (1023/255/63) | 〃 | 0.016 | 0.012 |
| 511/140 single | 〃 | 0.000 | 0.000 |

[0.005,0.45] の参考値（S1 5.242/3.607・IIR3 0.111/0.084・LP3 0.053/0.039）は v2.5 §2.8.1 を継承。**重要**: [0.005,0.30] と [0.01,0.30] は全 config で 3 桁一致。

#### 2.8.2 D1 手法不変性（v2.4 実測・継承）

base: 81 測定すべて [−9.546, −9.541]（gate 可）／cand: [−72.11, −164.50]（92.4 dB の手法依存 → 値での gate 不可・**E-1c の絶対形のみが window-independent な no-regression 契約**）。R9-5 の 0.94 dB 差は手法感度帯内のノイズ（R10-4 解消済み）。

#### 2.8.3 FIR 絶対基準（P0-D floor・継承）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|----------------|---------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

### 2.9 Shadow 実装契約（T-1 確定案の維持・R10-6 + R11-9 + R12-8）

`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（102 行）の実態: `prepare()` は stageCount 保存のみ・`reset()` は production 契約未模倣・processUp/processDown **存在せず** bitwise 比較対象なし。**Phase 0-1b/0-2 は実装なしには実行不能**（E-7 確定の維持）。

実装契約 5 項目（v2.5 §2.9 をそのまま継承）:
1. **係数生成の bitwise 一致**: prepareStage の全手順（:291-:368）を演算順序ごと複製。`juce::MathConstants<double>::pi` 経由の `std::sin` も同一経路
2. **履歴・分岐順序の一致**: keep・silence fast path（:583-613）・境界 guard（:620-649）・denorm クリア（:558-559）・isBadSample 位置
3. **比較対象（3 項目・bitwise）**: (a) 係数配列全要素 (b) シード固定疑似乱数 3 ブロック × 2 preset × ratio {2,4,8} の up/down 出力 (c) reset 前後状態差分（atomic 3 vs 1 の再現）
4. **candidate マクロ 2 系統の一致性 assert**: `PolyphaseGainFidelityTests.cpp`（新設・add_executable 40 番目）
5. **shadow cand は予測にすぎない**: 最終判定は flag ON ビルドの実測。**REF-FIDELITY は shadow fidelity の gate であり production 正しさの gate ではない**（R12-8）

### 2.10 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止（O-1 凍結・R12-10）/ calibration 禁止 / **0.75^N 補償の追加禁止（R12-5 + R13-4: production には補償が存在しないことを二重検索で確認済み・Phase 3-A でも再確認）** / evidence/ は C0 適用後の実行形態で。

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（R11-6 確定・R13-5 で再確認）

- 誤順: `BassBuzzMeasurement.cpp:1799` `configureProbeFlatEQ(e)` → `:1803` `setAutoGainStagingEnabled(false)`（AGC 再 ON・実測 `staging=0 eqAGC=1`）
- 正順: `:1833` staging false → `:1834` configureProbeFlatEQ（本版で再確認・`:1822` コメントも正順を文書化済み）
- **修正案 A（GO 候補）**: 3 行 swap（:1799 と :1803 の入替・コメント更新）。パッチは `remediation_v22_prep_patches_20260922.md §2`（未適用）
- 検証: `--buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（[EQ_DIRECT] ratio=1.000000 が独立証拠）

### 3.2 F-3 / F-4（記録のみ・継承）

F-3（EQ dry/wet 混合・`EQProcessor.Processing.cpp:982-996`）は別 work item。F-4（`ir`=dry baseline / `irwet`=wet 対照）は R5-8 の意味定義を維持。

### 3.3 測定 entry 派遣の罠（継承）

`--buzz*` 引数があると `PPIT:1123` が測定 entry へ派遣し `:1223` の無条件 attribution に到達しない。attribution 再現は引数なし実行。

---

## 4. 評価点マトリクス（R12-9・維持）

構成 = 8（RT float/double × {IIRLike 511/127/31, LinearPhase 1023/255/63, 511単段 140dB} + SoftClip float/double × {31/90}）。

| Gate \ 構成 | RT-f IIR3 | RT-f LP3 | RT-f 511 | RT-d IIR3 | RT-d LP3 | RT-d 511 | SC-f 31/90 | SC-d 31/90 |
|-------------|-----------|----------|----------|-----------|----------|----------|------------|------------|
| BUILD-ID | run-level（全測定 log 1 行） |
| G-BL / P0-A / P0-B / P0-C / P0-C' / P0-H | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-D（decimation alias） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-E（interpolation image・E-1/E-1c/E-2） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-F（latency） | ✓（D=290.25） | ✓（582.25） | ✓（255.00） | ✓ | ✓ | ✓ | ✓（15 固定） | ✓ |
| P0-G（float/double ペア） | ← RT ペア 1 点 | ← | ← | ← | ← | ← | ← SC ペア 1 点 | ← |
| P0-I（SoftClip 安全性） | — | — | — | — | — | — | ✓ | ✓ |

**実適用点数 = per-config gate 8 × 8 構成 = 64 + P0-G 2 + P0-I 2 = 68 点**（+BUILD-ID run-level）。歴史的呼称「4経路×3条件×9 gate=108」は **potential evaluation cells** として注記に残す。

---

## 5. 外部文献による裏付け（R13-7・新設）

| 論点 | 文献 | 裏付け内容 |
|------|------|------------|
| 案 E（interpolator の ×2 対称化） | MathWorks `dsp.FIRHalfbandInterpolator` 公式文書 | 「halfband 補間フィルタの係数は**出力パワー保存のため補間係数 2 でスケールされる**」。decimator との対比記述あり。center 位相の ×2 不在が規約違反であることの外部権威 |
| P0-F（群遅延定数性） | AES Convention Paper 8648（Wang-Reiss, decimation filters）・dsprelated/KVR の FIR latency 議論 | linear-phase FIR の群遅延一定性。`Latency.cpp:29-31` の `(taps-1)` 合成式の前提（対称 linear-phase）を支持 |
| Halfband polyphase 構造（center 0.5） | MathWorks polyphase 分解記述・wavewalkerdsp polyphase halfband 解説 | even polyphase 成分が単一遅延（center tap）である構造。`rawCoeffs[centerTap]=0.5` の構造的正当性と整合 |

---

## 6. 推奨する実行順序（v2.7）

```
[Step 0] 現状固定 — 技術側完了
  - source baseline 更新（R13-9・ConvoPeq.md 2026-09-21 16:16:35・FRESH）
  - P0-F 数値三重裏付け（R13-1）/ E-1c 独立再検証（R13-3）/ 二重補償二重検索（R13-4）
  - 文献裏付け取得（R13-7）/ O-20 棚卸し（R13-8）
  - O-19 解決確定（R13-9・v2.6 内部矛盾を解消）
  - 検証スクリプト実在: tmp/v26_p0f_gate_calc.py（再実行 PASS）・authoritative モデル（再実行 PASS）
  - 残: ユーザー判断（O-1〜O-20 / G-0 / T-1 / T-2 / T-3 / Phase 0 GO / O-16 C0）

[Step C0] evidence policy 単独 commit（O-16・R11-8 / R12-10）

[Step 1] §2 B-1 Phase 0（production 変更 0）— G-0 + T-1/T-2/T-3 + C0 が前提
  - 0-1 Baseline → 0-1b REF-FIDELITY（§2.9 契約の shadow 実装）→ 0-2 Candidate
  - 0-3 周波数（E-1 base gate + E-1c cand gate + E-1b 記録）→ 0-4 block/reset + macro assert
  - 0-5 SoftClip（R12-4 数値契約・O-20 は対象外）→ 0-6 float/double
  - harness は凍結（O-1）・BUILD-ID 行は Phase 0 から record

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は E-1（base）+ E-1c（cand no-regression）として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §3 F-2（test-only・案 A・3 行 swap）
[Step 4] §3 F-1（A→B→B'→C・failureReason 主判定・R12-11 の移行注意）
[Step 5] O-* commit（O-18 ジェネレータ修正を含む・O-1 は characterization 完了後に解凍）
[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option 適用 → flag ON 実測（BUILD-ID 行必須・R12-7）
  - 4 経路 × 8 構成で P0 再確認（§4 マトリクス）
  - Phase 3-A: **0.75^N 補償不存在の再確認（R13-4 の二重検索を再実行）** → flag default ON
  - Phase 3-B characterization → 3-C calibration → 3-D rigcheck threshold（commit 分離維持）
[Step 7] 別課題: R-2（parseOnOff 前例）→ R-1 → B-3, D-1（BUILD-ID 統合）, D-2 / O-20（isBadSample 閾値統一の検討）
```

---

## 7. 検証計画（v2.7）

### 7.1 検証スクリプト実在表（本版で実在確認＋再実行済みのもののみ列挙）

| ファイル | 目的 | 状態 |
|----------|------|------|
| `doc/work113/model_polyphase_20260920.py` + `_results.txt` | authoritative モデル | 実在・**本版再実行 PASS**（係数 6 design・DC・full-chain dev） |
| `tmp/v24_ripple_band.py` / `tmp/v24_d1_invariance.py` / `tmp/v24_latency_probe.py` | dev 全桁再現 / D1 81 測定 / latency peak | 実在 |
| `tmp/cio_literal.py` / `tmp/cio_octave.m` / `tmp/maxima_check.mac` | 逐語転写 / Octave / Maxima | 実在（v2.5 実績を継承・本版は Python 系で再検証） |
| **`tmp/v26_p0f_gate_calc.py` + `tmp/v26_p0f_gate_calc_results.txt`** | R12-1/R12-2/R11-3 | **本版再実行 PASS** |
| ~~`tmp/v25_*.py` 3 件~~ | v2.5 §7.1 が参照したが**実在しない**（保存漏れ） | **計画書から削除済み（v2.6 §1）** |

### 7.2 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| B-1 attribution | `build\Release\AudioEngineHarness.exe`（引数なし） | §2.4 全 row |
| F-2 | `--buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS |
| B-1 Phase 0/1 | 同上（flag ON/OFF rebuild） | OFF ≈ 0.75^N / ON ≈ 1.0・**log 先頭に BUILD-ID 行** |
| 静的解析 | cppcheck 2.21.0（`C:\Program Files\Cppcheck\cppcheck.exe`） | CIO.cpp 指摘 0（本版は version 確認まで実施・full run は Phase 0 実行時） |
| clang-tidy 23.1.1 | compile_commands 経由のみ（単独実行不可） | —（本版は version 確認まで実施） |
| ConvoPeq freshness | `python output_sourcecode_markdown.py --check` | FRESH（現状 FRESH・O-18 修正版） |

### 7.3 本版のツール census（調査に使用した全手段）

- **context-mode MCP**: `ctx_batch_execute`（並列 8・git/diffstat/rg 集約）・`ctx_search`・`ctx_execute`
- **rtk (WSL版)**: `wsl bash -lc '… ~/.local/bin/rtk rg …'`（出力圧縮・`'…'` 外側引用規約 준수）
- **AiDex MCP**: index 整備済み（487 files / 20,564 items / embeddings 有効）・`aidex_query(interpolateStage)`・`aidex_session`（外部変更 3 件を検出・自動 reindex）
- **serena MCP**: `get_symbols_overview(CustomInputOversampler.cpp)`（17 メソッド列挙）
- **semble**: MCP search + CLI `semble.exe search`（両系で同一局在に収束）
- **cocoindex (ccc.exe)**: `ccc search polyphase gain center tap`
- **graphify**: `graphify query 'CustomInputOversampler interpolateStage'`（143 nodes・isBadSample/decimateStage 連関を確認）
- **tgrep**: index FRESH（346 files・7h 前更新）・`tgrep search 'convValue'` で :557 の唯一性を cross-check
- **ast-grep (WSL)**: `convValue *= 2.0` 1 hit / `centerValue *= 2.0` 0 hit（不在の構造的証明）
- **rg (ripgrep/WSL)**: 0.75・compensate・F-2・F-1・CMake の全 census
- **cppcheck 2.21.0 / clang-tidy 23.1.1**: version 確認（full run は Phase 0 実行時に実施）
- **Brave Search**: R13-7 文献検索（MathWorks・AES・KVR）
- **NumPy/SciPy/Python 3.14.7**: authoritative モデル再実行・E-1c 独立判定・R13-1/R13-2 再計算
- **未使用の記録**: Dr.Memory（本監査は compile-time 数値照合であり動的メモリ計測の対象外のため N/A）・Octave/Maxima（v2.5 の 5 系統実績を継承・本版は Python 系統で再検証）・Obscura/Crawl4AI/DDGS/Trafilatura/context7/firecrawl/github/MSLearn（本版の調査対象に該当ページがなく未使用）

---

## 8. ロールバック（v2.5 §8 継承・変更なし）

§1 commit → `git revert` ／ B-1 behavioral → compile-time rollback（flag OFF rebuild）／ B-1 source → `#if` ブロック + option 削除 ／ Shadow → test-only revert ／ F-2 → 3 行 swap の再 swap ／ O-18 ジェネレータ → revert 後に ConvoPeq.md 再生成。

---

## 9. AGENTS.md 修正パッチ（R11-7 の維持・O-10 commit 用）

v2.5 §15 の内容をそのまま維持: `AGENTS.md` の「**Maxima は未インストール**」は誤り（`C:\maxima-5.50.0\bin\maxima.bat --very-quiet --batch-string` で稼働確認済）。O-10 commit 時に同時適用（差分統計は commit 時に再計測・現在の AGENTS.md は +78/−3 まで増加しているため）。

---

## 10. 監査ログ・参照（v2.7 追加分）

| ファイル / 手段 | 確認 | 結果 |
|----------------|------|------|
| `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | production 差分 | ✓ 出力なし（0 件を再確認） |
| `python output_sourcecode_markdown.py --check` | baseline | ✓ FRESH（2026-09-21 16:16:35・NEWER_SRC_COUNT=0） |
| `doc/work113/model_polyphase_20260920.py` 再実行 | authoritative | ✓ 全 PASS（係数・DC・full-chain） |
| `tmp/v26_p0f_gate_calc.py` 再実行 | R12-1/R12-2 | ✓ 全 PASS |
| 本版独立再計算（latency/DC/D1/E-1c 30 点） | R13-1/R13-2/R13-3 | ✓ 全一致・FAIL 0・margin 36.7 dB |
| ast-grep `convValue *= 2.0` / `centerValue *= 2.0` | 局在 | ✓ 1 hit / 0 hit（R13-6） |
| rg `0.75`（tests 除外）+ `compensate/makeup` | 二重補償 | ✓ 補償 0 件（R13-4） |
| `BassBuzzMeasurement.cpp` :1799/:1803/:1833/:1834 | F-2 | ✓ 誤順/正順を再確認（R13-5） |
| `RuntimePublicationValidator.h:101`・cpp:38/:169 | F-1 | ✓（R13-5） |
| CMakeLists.txt `CONVOPEQ_CORRECT_POLYPHASE_GAIN` 検索 | flag 未適用 | ✓ 0 hit（R13-5） |
| `src/tools/build_identity_gate.py` + `tools/check_layout_offsets.py` 実在 | O-19 | ✓ 解決確定（R13-9） |
| `CustomInputOversampler.cpp:24-48` 精読 | isBadSample | ✓ divergence 発見（R13-8・O-20） |
| Brave Search（MathWorks/AES/KVR） | 文献 | ✓ R13-7 |
| AiDex session（外部変更 3 件・自動 reindex） | index 鮮度 | ✓ |

---

## 11. 判断待ち項目（v2.7）

### 11.1 ユーザー意思決定

O-1（**凍結→完了後 commit**・R12-10）/ O-2〜O-17 は v2.5 §11.1 の推奨を維持 / **O-18（generator 修正）commit 推奨** / **O-19 解決確定**（対応不要）/ **O-20 将来対応**（対応不要・Phase 0 の対象外）/ G-0 / Phase 0 GO / F-2 案 A / F-1 全実施 / Phase 3 commit 分割。

### 11.2 残存技術判断（確定案提示済み・承認のみ）— 3 件（変更なし）

| ID | 論点 | v2.7 確定案 | 根拠 |
|----|------|-------------|------|
| **T-1** | REF-FIDELITY gate の実行条件 | §2.9 の 5 項目契約（骨格のままでは実行不能・R12-8 の性格明記を含む） | R10-6 / R11-9 / R12-8 |
| **T-2** | cand D1 の gate 可否 | **gate する（E-1c 絶対形）**。base E-1 と E-1c の 2 本立て。f̂ 0.40/0.45 は記録のみ | R12-2 / **R13-3（独立再検証）** |
| **T-3** | BUILD-ID gate の必須化 | Phase 1 の全測定 log に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` を必須化（欠落 = 測定無効） | R12-7 |

### 11.3 B-1 採用案の前提条件（v2.6 §11.3 から更新なし・R13 裏付け追加）

1. DESIGN-CONTRACT-A + **T-1/T-2/T-3** 明示承認（G-0）
2. REF-FIDELITY bitwise（§2.9・shadow fidelity gate）
3. round-trip DC = 1.0 ±1e-6
4. P0-B/C passband 維持（8 構成）
5. **P0-D stopband 維持（decimator 責務）**
6. **E-1 base D1 = −9.5424 ±0.01 dB + E-1c cand D1 ≤ −9.442 dB @f̂≤0.35（interpolator 責務・R13-3）**
7. P0-F latency 不変（コード同型式 §2.8 の基線・R13-1）
8. P0-I 数値契約（NaN/Inf FAIL・飽和 FAIL・指標記録・O-20 は対象外）
9. P0-C' differential 帰属 + P0-G（絶対誤差 gate・相対誤差記録）
10. D1 測定は補正軸 + expected bin 主報告（R9-1/R9-2/R10-4/E'-5）

### 11.4 v2.7 最終判定表

| 項目 | 判定 | v2.6 から |
|------|------|-----------|
| P0-F 数式 | **確定（三重裏付け R13-1）** | 強化 |
| E-1c 絶対形 | **確定（独立再検証 R13-3）** | 強化 |
| D/E 責務・P0-I 契約 | **確定維持** | 変更なし |
| source baseline | **更新（16:16:35・FRESH）** | 更新 |
| O-19 | **解決確定（v2.6 内部矛盾を解消）** | 是正 |
| O-20（新規） | **将来対応・non-blocking** | 新規 |
| 二重補償リスク | **不存在を二重検索で再確認（R13-4）** | 強化 |
| 案 E の理論的妥当性 | **外部文献で裏付け（R13-7）** | 新規 |
| §1 O-1〜O-20 | GO 候補（O-18 追加・O-19 解決・O-20 分離） | 拡充 |
| §3 F-2 / F-3 / F-4 | GO 候補 / 記録継承 / 記録継承 | 変更なし |
| §4 F-1 | GO 候補（R12-11 維持） | 変更なし |
| B-1 原因分析 | GO（5 系統 + 数値計算 + 文献） | 強化 |
| **B-1 Phase 0** | **GO（G-0 + T-1/T-2/T-3 + C0 承認が前提）** | 条件維持 |
| B-1 Phase 1+ / 案 E / Production patch / Calibration | **HOLD / 有力仮説 / HOLD / HOLD** | 変更なし |
| 技術的未確定 | **3 件（T-1/T-2/T-3）・確定案提示済み** | 変更なし |
| 計画書品質 | v2.6 内部矛盾（O-19）を是正・baseline 更新 | 是正 |

---

## 12. エビデンス（v2.7）

- **5 系統クロスチェック**: v2.5 §12.1 を継承 + 本版は authoritative モデル・v26 スクリプトを**再実行**し PASS を再確認（Python 3.14.7・NumPy/SciPy）
- **本版の独立数値計算**: latency 4 構成（15.00/255.00/290.25/582.25）・DC N=1..8・D1 構造定数（−9.542425）・E-1c 独立 30 点判定（FAIL 0・margin 36.7 dB）— R13-1/R13-2/R13-3
- **ツール実測**: §7.3 の census 表のとおり（context-mode MCP・rtk+rg/ast-grep・AiDex・serena・semble・ccc・graphify・tgrep・cppcheck/clang-tidy version・Brave Search・Python 数値計算）

## 13. v2.7 で追加した確定事項（R13 一覧）

- **R13-1** P0-F 数値の独立再計算（三重裏付け）
- **R13-2** DC 利得の独立再計算 + モデル再実行
- **R13-3** E-1c の独立再検証（30 点・margin 36.7 dB）
- **R13-4** 二重補償の二重検索 census
- **R13-5** F-2/F-1/CMake の再検証（flag 未適用を再確認）
- **R13-6** 欠陥局在の多手段再確認（ast-grep 不在証明を含む）
- **R13-7** 外部文献の裏付け（MathWorks ×2 規約・AES 群遅延）
- **R13-8** isBadSample 意味論の厳密化 → O-20 棚卸し
- **R13-9** O-table・baseline の刷新 + v2.6 O-19 内部矛盾の解消

---

*本書は v2.6（`remediation_plan_20260922_v2.6_revised.md`）の再改訂版 v2.7 である。*
*v2.6 本文は参照文書として維持し、**R13-1〜R13-9 が v2.6 と矛盾する箇所では本書を優先する**。*
*R5〜R13 は本書でも有効（上記是正点を除く）。中間エビデンスは `remediation_v22_intermediate_20260921.md`、パッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*
**継承の監査・確定事項**: R12-1〜R12-12 / R11-1〜R11-10 / R10-1〜R10-13 / R9-1〜R9-6 / R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。
