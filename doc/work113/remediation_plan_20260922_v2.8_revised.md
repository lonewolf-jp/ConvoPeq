# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v2.8）

- **版**: v2.8（v2.7 の再改訂・**R14-1〜R14-7** を新規確定）
- **前版**: `doc/work113/remediation_plan_20260922_v2.7_revised.md`（R13-1〜R13-9）
- **本改訂監査実施日**: 2026-09-22（同一作業ツリー・CommandCode 環境）
- **基準ソース**: HEAD = `8f127bfe`（親 `c4a08171`）。production `src/*.cpp/h`（tests 除外）の未 commit 差分 **0 件**（`git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` → 出力なし）を本セッションで再実測。
- **★ source baseline 更新（本セッションで再実測）**: `python output_sourcecode_markdown.py --check` → **2026-09-21 16:16:35 生成・NEWER_SRC_COUNT=0・STATUS: FRESH** を本版の authoritative snapshot として固定。レビュー時の「9/19 版しか参照できない」制約は解消。
- **★ working tree と HEAD の差分位置（R14-3 で確定）**: production `src/convolver` `src/dsp` `src/core` `src/audioengine` `src/eqprocessor` は **差分 0**。差分は (a) `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` **+634/−2**・`PublishPipelineIntegrationTests.cpp` **+8/−0**、(b) `src/tools/check_layout_offsets.py` **削除 52 行**、(c) `output_sourcecode_markdown.py` **+11/−4**・`doc/work113/residual_tasks_20260919.md`、に限定。**`PolyphaseGainCandidateRef.h` は untracked・HEAD 未登録**。
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `_results.txt`（v2.7 継承・本セッションで再実測未実行・再実行が必要な場合は Phase 0 着手前 0-0 工程で実施する旨を §6 で明示）
- **v2.8 の位置付け**: v2.7 の GO/HOLD・案 E（candidate hypothesis）・Phase 0 の 5 構造・R10/R11/R12/R13 確定事項は**原則維持**。本改訂は監査指摘 (a) **P0-I saturation event 定義の 1 行固定**、(b) **BUILD-ID を Phase 0 から必須化**、(c) **案 E の表現を弱める**（「外部文献で裏付け」→「外部文献と整合・Phase 0 で検証」）、(d) **P0-C「base rate」と「stage-local input rate」のコード混同禁止**、(f) **F-2 行番号の HEAD/working tree 乖離の明示**、(e) **v2.7 で参照された `Latency.cpp:29-31` / `Latency.cpp:120` の非実在の訂正**、(g) **Shadow Reference 5 項目契約の実装漏れ箇所の明文化**、を扱う。
- **案 E は有力仮説（candidate hypothesis）のまま**。defect（DC = 0.75^N）は v2.7 でも再実行で再現。確定判断は G-0 承認 + Phase 0 REF-FIDELITY/P0-A〜P0-I 全 PASS まで保留（R5-4 継承）。
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**（production 側規約。test 側診断出力は §R12-7）
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）
- **R5〜R14 は本書でも有効（是正点を除く）。R14 と矛盾する v2.7 の箇所では本書を優先する**

---

## 0. v2.7 → v2.8 改訂サマリ

### 新規確定（R14-1 〜 R14-7）

| ID | 区分 | 内容 | v2.8 での確定 |
|----|------|------|----------------|
| **R14-1** | P0-I saturation event 定義の 1 行固定 | SoftClip 経路（`prepareSingleStage(31, 90.0)`）で `FastTanhApprox::SoftClipPolicy::clipThreshold (= 4.5)` を超える入力が fastTanh 経路で戻り値 ±1.0 に張り付いたサンプル数を **saturation event 数** と定義。candidate で baseline より多い場合は FAIL、同一以下なら PASS。NaN/Inf 1 件でも FAIL は別項目で維持 | P0-I-b 契約を曖昧さ無く実装可能化 |
| **R14-2** | BUILD-ID を Phase 0 から必須化 | `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1`（macro は `add_compile_definitions` で 0/1 必ず定義・R12-7）を **Phase 0 から必須**（欠落 = 測定無効）に格上げ。Phase 0 の baseline/candidate 取り違えを即時に検出可能化 | T-3 を Phase 0 必須に更新 |
| **R14-3** | HEAD vs working tree 位置を明示 | v2.7 の F-2 行番号（1799/1803/1833/1834）は **working tree 基準**。HEAD（8f127bfe）は `BassBuzzMeasurement.cpp:1370/1374`（誤順）と `:1609/1614`（正順）に存在。working tree 差分 +634/−2 行は tests のみ。production 差分 0 を維持 | F-2 修正パッチの diff anchor を HEAD 基準で評価可能化 |
| **R14-4** | 案 E 表現弱化 | 「外部文献で裏付け」→「**外部文献と整合する。ConvoPeq への適用正当性は Phase 0 で検証する**」に修正。「欠陥が規約違反であることを外部権威が支持する」→ 削除。MathWorks 文献は「halfband 補間フィルタの係数は補間係数 2 でスケールされる」という**halfband interpolator 全体の一般的な規約**を記述したものであって、ConvoPeq の center polyphase に必ず ×2 を追加すべきという直接結論を含意しないことを明記 | 案 E を「有力仮説」のまま厳密化。Phase 0 の PASS/FAIL が仮説の最終判定になる |
| **R14-5** | P0-C / P0-F のレート混同回避 | P0-C「Fs_in（base rate）」と P0-F「stageRate(s) = baseRate × 2^(s+1)」は **異なるレート軸** を参照する。P0-C の周波数軸は `Fs_in = baseRate` であり、`stageRate` の比ではない。測定実装では両者を別個の名前（例: `baseSampleRate`, `stageRateAtStageS`）で参照し、混同を物理的に排除する。本リポジトリの `CustomInputOversampler.cpp` には `stageRate` 変数が存在しないため、shadow reference / 計測実装側で明示的に導入する | P0-C / P0-F 測定の数値混入リスクを除去 |
| **R14-6** | `Latency.cpp:29-31` / `Latency.cpp:120` の非実在訂正 | v2.7 §2.8 で参照された `Latency.cpp` は **本リポジトリに存在しない**（`git ls-tree -r HEAD` にも記録なし）。P0-F 群遅延式 `D = Σ_s (taps[s]−1) × (baseRate / stageRate(s))` は linear-phase FIR halfband の標準的な群遅延式から導出される**理論式**であり、実装は shadow reference または Phase 0 計測側で明示的に行う。R11-1/R12-1/R13-1 三重裏付けは本理論式の独立再計算であり、production code との 1:1 対応ではない | P0-F 説明の正確性確保 |
| **R14-7** | Shadow Reference 5 項目契約の実装漏れ箇所の明文化 | `PolyphaseGainCandidateRef.h`（102 行）の現状は (1) `expectedDcRoundTrip`・`computeD1Bins`・`PolyphaseGainShadow::reset/prepare/centerPhaseGain/expectedDcRoundTrip` のみ実装。**欠落**: (a) `processUp(double* in, int n, double* out, int ch)` の本体、(b) `processDown` の本体、(c) 履歴（`upHistory[ch]` / `downHistory[ch]`）の bitwise 同期、(d) `prepareStage` の全手順（:287-:368）演算順序ごと複製、(e) denorm クリア・isBadSample 位置の同期。Phase 0 着手時に §2.9 の 5 項目契約を満たす実装を追加する | Phase 0-1b/0-2 実行可能性の事前合意 |

### v2.7 内部矛盾の是正（本版で継続解消）

- v2.7 §2.8 P0-I の「baseline に無い飽和イベント FAIL」という表現は定義が曖昧 → **R14-1 で 1 行固定**
- v2.7 §2.8 BUILD-ID の「Phase 1 から必須・Phase 0 は record」は Phase 0 全体の因果帰属を曖昧化 → **R14-2 で Phase 0 から必須に格上げ**
- v2.7 §5 の「欠陥が規約違反であることを外部権威が支持する」は過大表現 → **R14-4 で弱める**
- v2.7 §2.8 P0-F の `Latency.cpp:29-31` / `Latency.cpp:120` の参照は非実在 → **R14-6 で訂正**

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-20・v2.8 時点の実測棚卸し）

### 1.1 実測 diffstat（2026-09-22 時点・本セッションで再実測）

```text
git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'
  → 出力なし（production src/ 差分 0 件・R14-3 で再確認）

git diff --stat HEAD -- src/
  src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp     | 636 +++++++++++++-
  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp | 8 +
  src/tools/check_layout_offsets.py                       | 52 --
  3 files changed, 642 insertions(+), 54 deletions(-)

python output_sourcecode_markdown.py --check
  baseline Generated : 2026-09-21 16:16:35  (ConvoPeq.md)
  NEWER_SRC_COUNT    : 0
  STATUS             : FRESH  — snapshot は現行ソースを反映しています

git ls-tree -r HEAD --name-only | grep -i latency
  → 該当なし（本リポジトリに Latency.cpp / Latency.h は存在しない・R14-6）
```

| ID | パス | 実測 | v2.7 との差分 | v2.8 推奨 |
|----|------|------|----------------|-----------|
| **O-1** | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634/−2** | 一致 | **Phase 0 characterization 完了まで凍結（commit しない）→ 完了後に cleanup/分割して commit**（R12-10 維持） |
| **O-1b** | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8/−0** | 一致 | O-1 と同一運命（凍結） |
| **O-2** | 台帳更新 | 独立 commit | — | 独立 |
| **O-3 / O-12** | `ConvoPeq.md` | **16:16:35 再生成・FRESH（NEWER_SRC_COUNT=0）** | 一致 | **commit しない**（再生成資産・`--check` FRESH を維持） |
| **O-4** | `Testing/Temporary/CTestCostData.txt` | **D**（削除） | 一致 | 現状維持 |
| **O-5** | `.opencode/opencode.json` | untracked | 一致 | 触らない |
| **O-6** | push | ahead 2（`8f127bfe`・`c4a08171` は commit 済） | 一致 | **ユーザー手動** |
| **O-7 / O-10** | `AGENTS.md` | **+78/−3**（Freebuff 節追加を含む） | **更新（v2.6 は +71/−3）** | 環境記録として commit 可。R11-7 の Maxima 誤記修正（§9）を同時に適用 |
| **O-8** | `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` | M（追跡済みバイナリ） | 一致 | commit しない → 将来 `.gitignore` + `git rm --cached` |
| **O-9** | tool-inventory 2 件 | untracked | 一致 | 触らない（ユーザー判断） |
| **O-11** | `doc/work113/residual_tasks_20260919.md` | **+71/−1** | 一致 | 台帳として commit 可 |
| **O-13** | `doc/work113/*.md` + model py/results untracked | 一致（本版追加で v2.8 1 件） | 承認版+台帳+モデルを 1 commit |
| **O-14** | `.mcp.json` | **+33/−23** | 一致 | commit（環境記録） |
| **O-15** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | untracked（102 行） | 一致 | O-13 に併同可。**commit 前に §2.9 契約の 5 項目実装を反映した版へ**（R14-7） |
| **O-16** | `evidence/epoch_reclaim_audit.json` | **M・Bin** | 一致 | **C0 で追跡外化**（R11-8 確定・§1.3） |
| **O-17** | `doc/WorkBuddy_AI_Desktop_Toolchain.md` | untracked | 一致 | O-9 と同一方針 |
| **O-18** | `output_sourcecode_markdown.py` | **+11/−4** | 一致 | **commit 推奨**（R12-6・独立 commit）。本版 snapshot は修正版で生成済み |
| **O-19** | `src/tools/build_identity_gate.py` / `tools/check_layout_offsets.py` | **解決で確定**: `src/tools/build_identity_gate.py` 実在（`ls` 確認）・`tools/check_layout_offsets.py` 実在（`tools/*.py` 一覧で確認）。v2.7 解消済 | **解決（R13-9）**。D-1（build identity gate）は R12-7 の BUILD-ID gate と統合のうえ `src/tools/build_identity_gate.py` を継続使用 |
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
# C1: Phase-0 shadow/reference 実装 commit（O-15 + §2.9 契約 5 項目実装版）
# C2: characterization 結果 commit（測定 log・CSV は tmp/ 起点で添付判断）
```

### 1.4 Shadow Reference untracked 報告（R14-7）

`PolyphaseGainCandidateRef.h` は HEAD に存在しない（`git ls-tree -r HEAD --name-only | grep -i "PolyphaseGainCandidateRef"` 出力なし）ことを v2.8 で確認。working tree に 102 行の骨格のみ存在し、§2.9 の 5 項目契約のうち **3 項目は未実装**（processUp/processDown なし・履歴同期なし・prepareStage 複製なし・denorm/isBadSample 位置同期なし）。**Phase 0 着手時に本ファイルに 5 項目実装を追加することを O-15 commit の前提条件とする**。

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 利得欠陥

### 2.1 確定している事実（v2.7 §3.1 継承・本版で再実測）

```text
主因        interpolateStage の center 位相に ×2 が無い（polyphase 利得規約の不対称）
欠陥局在    src/CustomInputOversampler.cpp :557（convValue *= 2.0 のみ・ast-grep 1 hit）
            :563-564（center に ×2 なし・centerValue *= 2.0 は 0 hit）
            decimateStage :658（centerCoeff * centerSample + conv tap 累加・×2 なし＝DC 利得 1.0 で正規）
round-trip  DC: base 0.75^N / cand 1.0（authoritative モデル再実行 + v2.7 再計算で再確認）
段ごとの和  conv 分岐: Σcoeffs 0.5 → ×2 → DC 1.0（正規）／center 分岐: 0.5 → ×2 なし → DC 0.5（不対称）
利益        (4/3)^N = +2.498775 / +4.997549 / +7.496324 dB（1/2/3 段・v2.7 再計算）
画像        up 出力の像/トーン = 1/3 = −9.542425 dB（構造定数・R11-4 確定・v2.7 再計算）
係数        FIRsum 1.0 / center 0.5 / convSum 0.5（全 6 design・モデル再実行で再現）
契約不整合  isLinearPhaseFIR / isSymmetricUpDown（h:21-22）と矛盾
production  変更 0 / commit 0（v2.7 §10 で再確認・本版 §1.1 で git diff 実測・tests 以外 0 件）
```

**peak / mean の帰属不能性（R10-3・E-8 確定表現の維持）**: up 出力は位相交互 `1.0, 0.5, …`。peak 系で「up 1.0/down 0.75」、mean 系で「up 0.75/down 1.0」— どちらも同じ階段信号の別記述で因果の根拠にならない。因果は分岐係数和の不対称（1.0 vs 0.5）で確定。

### 2.2 根本原因の行番号（v2.7 §2.2 継承 + R14-6 で Latency.cpp 参照削除）

| 箇所 | 内容 | 実測手段 |
|------|------|----------|
| `prepareStage` :287-390 | `taps=jmax(3,taps\|1)` :291 / `centerTap=(taps-1)/2` :292 / halfband ゼロ化 :319-323 / `sum→1.0` :325-333 / `rawCoeffs[centerTap]=0.5` :335/:348 / 非 center を 0.5 へ :336-347 / `convCount` :350 / convCoeffs 構築 :360-365 / `centerCoeff` :367 / `centerDelayInput` :368 | Read + rg（AiDex query でも同一箇所） |
| `interpolateStage` :492-568 | **:557 `convValue *= 2.0`（ast-grep で 1 hit のみ）** / :558-559 denorm / **:563-564 出力（`centerValue *= 2.0` は ast-grep 0 hit＝不在確定）** | ast-grep + rg + tgrep（`centerValue *= 2.0` 不在を 3 手段で確認） |
| `decimateStage` :570-723 | silence fast path :583-613 / :658 `acc = centerCoeff * centerSample` + conv 累加（×2 なし＝DC 1.0 正規） | Read |
| `processUp` :725- / `processDown` :785- | hardFallback / corruption clear | 継承 |
| `reset()` :452-467（atomic 3）/ `clearAllStages()` :469-484（atomic 1） | Phase 0 は `reset()` 経由必須（R4-3） | 継承 |
| `prepareSingleStage` :392-450 | SoftClip 局所 OS（Lifecycle :188/:261 で `(31, 90.0, internalMaxBlock)`） | 継承 |
| `dotProductAvx2` :159-216 | conv 専用 SIMD・center 位相は非依存 | serena symbols + graphify |
| `isBadSample` :24-37 / `isBadSampleV` :40-48 | scalar は NaN/Inf + \|x\|>2^53 / AVX2 は NaN + \|x\|>1e20（**R13-8 の divergence**・O-20） | Read（v2.7 精読・v2.8 で再確認） |

### 2.3 5 系統独立再現 + 数値計算裏付け（v2.7 §2.3 継承）

production バイナリ / authoritative モデル / NumPy 逐語転写 / Octave 11.3.0 / Maxima 5.50.0 — 全系統が base 0.75^N・cand 1.0・Δ+2.498775 dB(N=1)・image/tone 1/3 に一致（v2.5 §2.3 表を継承）。本版では再実行は未実施（Phase 0 着手時に 0-0 工程として `tmp/v26_p0f_gate_calc.py` と `doc/work113/model_polyphase_20260920.py` を再実行する）。

### 2.4 ライブ帰因テーブル（v2.7 §2.4 継承・引数なし実行の制約）

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

**外部文献との整合（R13-7 継承 + R14-4 で表現を弱化）**: MathWorks `dsp.FIRHalfbandInterpolator` 公式文書は「halfband 補間フィルタの係数は**出力パワー保存のため補間係数 2 でスケールされる**」と記述している。これは **halfband interpolator 全体に対する一般的な設計規約**であって、ConvoPeq の center polyphase に必ず ×2 を追加すべきという直接結論を含意するものではない。案 E（center 位相にも ×2 を対称適用）はこの規約と**整合する**が、ConvoPeq への**適用正当性は Phase 0 の REF-FIDELITY と P0-A〜P0-I の全 PASS をもって初めて確定する**。decimate 側に ×2 が無いこと（:658 の正規性）とも整合する。

### 2.6 CMake / flag（v2.7 §2.6 継承 + R14-6 の表現統一）

- `option(CONVOPEQ_CORRECT_POLYPHASE_GAIN "Correct polyphase gain convention (B-1 案E)" OFF)` を option 群（:40 近傍）に追加
- `add_compile_definitions(CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${...}>)` を `add_subdirectory(JUCE)`（:1043）後・`juce_add_gui_app(ConvoPeq`（:1062）前に配置
- 到達確認: CIO.cpp を compile するのは ConvoPeq（`target_sources` :1282・リスト内 :1196/:1243）と AudioEngineHarness の 2 論理ターゲット。**本版再検索で `CONVOPEQ_CORRECT_POLYPHASE_GAIN` は CMakeLists.txt に 0 hit＝現状未適用を再確認**
- C++ は **`#if`**（`#ifdef` 禁止）。既定 OFF / runtime flag 不採用 / **現状未適用**
- **R12-7 + R14-2**: 測定出力の `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` 行を **Phase 0 から必須化**（test-only・`add_compile_definitions` により macro は常に 0/1 で定義済み）。欠落 = 測定無効

### 2.7 Phase 0 characterization（v2.7 §2.7 継承 + R14-1/R14-5 反映）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）+ T-1/T-2/T-3 をユーザーが明示承認 + 9/21 snapshot FRESH 再確認
        （本版 §1.1 で確認済。Phase 0 着手直前に再実測する旨を §6 で明示）
0-1  Baseline record-only: round-trip DC = 0.75^N ±1e-6（invariant のみ gate・up/dn 分割は診断ログ）
0-1b REF-FIDELITY: Shadow == production（bitwise・reset() 経由・§2.9 の実装契約・R14-7 反映）
0-2  Shadow Candidate E: round-trip DC = 1.0 ±1e-6
0-3  周波数（D1/D2/D3）— D1 は base 側 gate + E-1c cand gate（§2.8）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）+ candidate マクロ 2 系統一致性 assert
0-5  SoftClip local OS（float + double・R12-4 の数値契約・R14-1 の saturation event 定義適用）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8 / 相対誤差記録のみ）
```

**D1 定義（R10-4 + E'-5 継承 + R12-2 + R13-3）**:
- 軸: `D1 = |Y_up(image bin)| / |Y_up(tone bin)|`、up 出力長 2N、tone bin = f̂·N、image bin = N−f̂·N（補正軸）
- **E-1（gate・base）**: `D1_base = −9.542425 ± 0.01 dB`（構造定数・手法不変 81 測定で max 差 0.0036 dB）
- **E-1c（gate・cand）**: `D1_cand ≤ −9.442 dB`（= −9.542425 + 0.1）を **f̂ ∈ [0.05, 0.35] で gate**。f̂ = 0.40 / 0.45 は**遷移帯のため記録のみ**。v2.7 の独立 30 点判定でも FAIL 0・最小 margin 36.7 dB を再現（R13-3）
- E-1b（記録）: cand D1 を矩形窓・Hann+trim8 の 2 条件で併記
- tone/image bin は expected 主報告・argmax ±8 bin は検証ログ（R9-2 継承）
- final alias rejection は P0-D が担当（R12-3 の責務分離）

**Shadow Reference**: §2.9（T-1）。骨格のみでは gate 実行不能（R10-6/R11-9 確定 + R14-7 の未実装箇所の維持）。

**D2 / D3**: v2.2/v2.3 のまま。D2 gate [0.005, 0.40] base==cand ≤0.01 dB・0.45 記録のみ。D3 worst spur −98〜−101 dB（窓 sidelobe）・base==cand 差 0.00 dB。

### 2.8 B-1-P0 GATE（判定表・v2.7 §2.8 維持 + R14-1/R14-2/R14-5/R14-6 反映）

| ID | 判定対象 | 責務（R12-3） | 条件 | v2.8 裏付け |
|----|----------|---------------|------|-------------|
| G-0 | 契約 | — | DESIGN-CONTRACT-A（E1〜E5）+ **T-1・T-2・T-3** 明示承認 | T-3 を Phase 0 必須に更新（R14-2） |
| **BUILD-ID** | binary identity | run-level | **全測定 log の先頭に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1`**。**Phase 0 から必須・欠落 = 測定無効（R14-2 で Phase 0 必須化）** | R14-2 |
| G-BL | Baseline sanity | 全経路 | round-trip DC = 0.75^N ±1e-6 のみ gate。up/down 分割は診断ログ | R13-2 再計算 |
| REF-FIDELITY | Shadow==production | shadow fidelity | bitwise（tolerance 0）・§2.9 の 3 対象・`reset()` 経由。**注: production 正しさの gate ではない**（R12-8） | 維持 |
| P0-A | Candidate DC | 全経路 | round-trip 1.0 ±1e-6（必要条件） | モデル再実行 |
| P0-B | 低域 | 全経路 | 50 Hz / 1 kHz unity ±0.1 dB・4 経路 | モデル再実行（−2.4988/−0.0000 dB） |
| **P0-C** | passband ripple | 全経路 | **[0.005,0.30] max−min ≤0.05 dB（dense）**・周波数軸は **`Fs_in = baseRate`** で正規化（R14-5：P0-F の stageRate とは別個の名前で参照） | モデル再実行（ripple[0.01,0.3] 0.001 dB） |
| P0-C' | differential | 全経路 | (4/3)^N ±0.05 dB（IIR3/LP3 f≤0.45・S1 f≤0.30）— attribution criterion | モデル再実行（dev +0.0000 dB） |
| **P0-D** | **decimator alias rejection** | decimator | stopband: alias ≤ −(A−10) dB @[t_end, 0.5]（per-design 絶対基準 §2.8.3） | 維持 |
| **P0-E** | **interpolator image rejection** | interpolator | **E-1: base D1 = −9.5424 ±0.01 dB（gate）** / **E-1c: cand D1 ≤ −9.442 dB @f̂≤0.35（gate・R13-3 で独立再検証）** / E-1b: cand D1 2 条件記録 / E-2: D2 [0.005,0.40] base==cand ≤0.01 dB / E-3: 0.45 記録のみ | R13-3 |
| P0-F | latency | 全経路 | peak ∈ [floor(D), floor(D)+1]。**D = Σ_s (taps[s]−1) × (baseRate / stageRate(s))**・stageRate(s) = **baseRate × 2^(s+1)**。**式は linear-phase FIR halfband の標準的な群遅延式 (taps-1)/2 から導出される理論式**（R14-6：本リポジトリに `Latency.cpp` は存在しない。shadow reference / 計測実装側で明示的に導入する）。**作業例（IIR3）: 510/2 + 126/4 + 30/8 = 255 + 31.5 + 3.75 = 290.25**。基線: S1 15（dev 0）・511単段 255（dev 0）・IIR3 291（dev +0.75）・LP3 583（dev +0.75）。SoftClip 単段 15（`SoftClipPadePolicy::clipThreshold` 4.5 に基づく SoftClip 局所 OS の入力→出力群遅延・測定実装側で固定）。base==cand も gate | R13-1（三重裏付け）+ R14-6 |
| P0-G | float/double | 2 ペア | maxAbsErr ≤5e-7 / RMS ≤5e-8 + **max 相対誤差（\|ref\|>1e-6 のみ）record**（R12-12） | 維持 |
| P0-H | block/reset | 全経路 | partition bitwise・reset contract（atomic 3 / atomic 1）+ candidate マクロ 2 系統一致性 assert | 維持 |
| **P0-I** | SoftClip 安全性 | SoftClip 2 経路 | **I-a: DC = 1.0 ±1e-6** / **I-b（数値契約・R12-4 + R14-1）: 出力 NaN/Inf 1 件でも FAIL・SoftClip fastTanh 経路で `\|input\| ≥ FastTanhApprox::SoftClipPolicy::clipThreshold (= 4.5)` となり戻り値 ±1.0 に張り付いたサンプル数（saturation event 数）が baseline より多い場合 FAIL** / **I-c（記録）: peak・RMS・THD・THD+N・saturation event 数・NaN/Inf 数・動作点**（`prepareSingleStage(31,90.0)`・Lifecycle :188/:261）。scalar/AVX2 の巨大振幅閾値差（R13-8）は NaN/Inf 検出に影響しないため本契約の対象外（O-20 に分離） | **R14-1 で saturation event 定義を 1 行固定** |

**全 PASS → Phase 1 eligibility**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

#### 2.8.1 passband ripple 基線（v2.7 §2.8.1 継承）

| config | 帯域 [0.005,0.30] | base | cand |
|--------|------------------|------|------|
| S1 (31/90) N=1 | gate 帯 | 0.001 | 0.001 |
| IIR3 (511/127/31) | 〃 | 0.034 | 0.025 |
| LP3 (1023/255/63) | 〃 | 0.016 | 0.012 |
| 511/140 single | 〃 | 0.000 | 0.000 |

[0.005,0.45] の参考値（S1 5.242/3.607・IIR3 0.111/0.084・LP3 0.053/0.039）は v2.5 §2.8.1 を継承。**重要**: [0.005,0.30] と [0.01,0.30] は全 config で 3 桁一致。

#### 2.8.2 D1 手法不変性（v2.7 §2.8.2 継承）

base: 81 測定すべて [−9.546, −9.541]（gate 可）／cand: [−72.11, −164.50]（92.4 dB の手法依存 → 値での gate 不可・**E-1c の絶対形のみが window-independent な no-regression 契約**）。R9-5 の 0.94 dB 差は手法感度帯内のノイズ（R10-4 解消済み）。

#### 2.8.3 FIR 絶対基準（v2.7 §2.8.3 継承）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|---------------|----------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

### 2.9 Shadow 実装契約（T-1 確定案の維持 + R14-7 で未実装箇所を明文化）

`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（102 行・**HEAD 未登録・working tree untracked**）の現状（R14-7）:
- 実装済み: `expectedDcRoundTrip(int stages, bool candidate)`、`computeD1Bins(double fhat, int N)`、`PolyphaseGainShadow::reset()`（`prepared_/stageCount_` フラグのみ）、`PolyphaseGainShadow::prepare()`（stageCount 保存のみ）、`PolyphaseGainShadow::centerPhaseGain()`（candidate macro 切替）、`PolyphaseGainShadow::expectedDcRoundTrip()`。
- **未実装（R14-7 で明文化）**: (a) `processUp(double* in, int n, double* out, int ch)` の本体、(b) `processDown` の本体、(c) 履歴（`upHistory[ch]` / `downHistory[ch]`）の bitwise 同期、(d) `prepareStage` の全手順（:287-:368）演算順序ごと複製、(e) denorm クリア・isBadSample 位置の同期、`reset()` 経由の production 契約（atomic 3）の模倣。

**Phase 0 着手時に 5 項目契約を実装**:

1. **係数生成の bitwise 一致**: prepareStage の全手順（:287-:368）を演算順序ごと複製。`juce::MathConstants<double>::pi` 経由の `std::sin` も同一経路
2. **履歴・分岐順序の一致**: keep・silence fast path（:583-613）・境界 guard（:620-649）・denorm クリア（:558-559）・isBadSample 位置
4. **比較対象（3 項目・bitwise）**: (a) 係数配列全要素 (b) シード固定疑似乱数 3 ブロック × 2 preset × ratio {2,4,8} の up/down 出力 (c) reset 前後状態差分（atomic 3 vs 1 の再現）
5. **candidate マクロ 2 系統の一致性 assert**: `PolyphaseGainFidelityTests.cpp`（新設・add_executable 40 番目）
6. **shadow cand は予測にすぎない**: 最終判定は flag ON ビルドの実測。**REF-FIDELITY は shadow fidelity の gate であり production 正しさの gate ではない**（R12-8）

### 2.10 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止（O-1 凍結・R12-10）/ calibration 禁止 / **0.75^N 補償の追加禁止（R12-5 + R13-4: production には補償が存在しないことを二重検索で確認済み・Phase 3-A でも再確認）** / evidence/ は C0 適用後の実行形態で。**BUILD-ID 行は Phase 0 から必須（R14-2）**。

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（R11-6 確定 + R13-5 で再確認 + R14-3 で HEAD/working tree 行番号を明示）

**HEAD（8f127bfe）基準の行番号**:
- 誤順: `BassBuzzMeasurement.cpp:1370` `configureProbeFlatEQ(e)` → `:1374` `setAutoGainStagingEnabled(false)`（AGC 再 ON・実測 `staging=0 eqAGC=1`）
- 正順: `:1609` `setAutoGainStagingEnabled(false)` → `:1614` `configureProbeFlatEQ(e)`（本版で再確認）

**working tree 基準の行番号（v2.7 §3.1 で参照）**:
- 誤順: `:1799` `configureProbeFlatEQ(e)` → `:1803` `setAutoGainStagingEnabled(false)`
- 正順: `:1833` staging false → `:1834` configureProbeFlatEQ（v2.7 で再確認・`:1822` コメントも正順を文書化済み）
- 別系列の正順: `:2241` staging false → `:2246` configureProbeFlatEQ

**行番号乖離の根拠（R14-3）**: working tree は HEAD から +634/−2 行の tests 専用 diff を持つため。**production 側 `src/convolver` `src/dsp` `src/core` `src/audioengine` `src/eqprocessor` の差分は 0 件**。差分 anchor は HEAD 基準で評価する。

- **修正案 A（GO 候補）**: HEAD 基準で `:1370` と `:1374` の入替（コメント更新）。パッチは `remediation_v22_prep_patches_20260922.md §2`（未適用）
- 検証: `--buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（[EQ_DIRECT] ratio=1.000000 が独立証拠）

**F-2 着手順（R12-10 維持・B-1 Phase 0 と分離）**:
```
B-1 Phase 0 characterization
        ↓
Phase 0 result freeze（GO/FAIL 確定後）
        ↓
F-2 3 行 swap
        ↓
F-2 regression（--buzz-rigcheck=eq で staging=0 eqAGC=0 確認）
```

### 3.2 F-3 / F-4（記録のみ・継承）

F-3（EQ dry/wet 混合・`EQProcessor.Processing.cpp:982-996`）は別 work item。F-4（`ir`=dry baseline / `irwet`=wet 対照）は R5-8 の意味定義を維持。

### 3.3 測定 entry 派遣の罠（継承）

`--buzz*` 引数があると `PPIT:1123` が測定 entry へ派遣し `:1223` の無条件 attribution に到達しない。attribution 再現は引数なし実行。

---

## 4. 評価点マトリクス（v2.7 §4 継承）

構成 = 8（RT float/double × {IIRLike 511/127/31, LinearPhase 1023/255/63, 511単段 140dB} + SoftClip float/double × {31/90}）。

| Gate \ 構成 | RT-f IIR3 | RT-f LP3 | RT-f 511 | RT-d IIR3 | RT-d LP3 | RT-d 511 | SC-f 31/90 | SC-d 31/90 |
|-------------|-----------|----------|----------|-----------|----------|----------|------------|------------|
| BUILD-ID | run-level（全測定 log 1 行・R14-2 で Phase 0 から必須） |
| G-BL / P0-A / P0-B / P0-C / P0-C' / P0-H | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-D（decimation alias） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-E（interpolation image・E-1/E-1c/E-2） | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-F（latency） | ✓（D=290.25） | ✓（582.25） | ✓（255.00） | ✓ | ✓ | ✓ | ✓（15 固定） | ✓ |
| P0-G（float/double ペア） | ← RT ペア 1 点 | ← | ← | ← | ← | ← | ← SC ペア 1 点 | ← |
| P0-I（SoftClip 安全性・R14-1 適用） | — | — | — | — | — | — | ✓ | ✓ |

**実適用点数 = per-config gate 8 × 8 構成 = 64 + P0-G 2 + P0-I 2 = 68 点**（+BUILD-ID run-level・Phase 0 必須）。歴史的呼称「4経路×3条件×9 gate=108」は **potential evaluation cells** として注記に残す。

---

## 5. 外部文献による裏付け（v2.7 §5 継承 + R14-4 で表現を弱化）

| 論点 | 文献 | 裏付け内容 |
|------|------|------------|
| 案 E（interpolator の ×2 対称化） | MathWorks `dsp.FIRHalfbandInterpolator` 公式文書 | 「halfband 補間フィルタの係数は**出力パワー保存のため補間係数 2 でスケールされる**」。decimator との対比記述あり。**halfband interpolator 全体に対する一般的な設計規約**であって、ConvoPeq の center polyphase に必ず ×2 を追加すべきという直接結論は含意しない。案 E との整合は Phase 0 で検証する（R14-4） |
| P0-F（群遅延定数性） | AES Convention Paper 8648（Wang-Reiss, decimation filters）・dsprelated/KVR の FIR latency 議論 | linear-phase FIR の群遅延一定性。`(taps-1)/2` を stage-local レートで換算する式（v2.7 §2.8 と同型）の前提（対称 linear-phase）を支持 |
| Halfband polyphase 構造（center 0.5） | MathWorks polyphase 分解記述・wavewalkerdsp polyphase halfband 解説 | even polyphase 成分が単一遅延（center tap）である構造。`rawCoeffs[centerTap]=0.5` の構造的正当性と整合 |

---

## 6. 推奨する実行順序（v2.8）

```
[Step 0] 現状固定 — 技術側完了
  - source baseline 更新（R13-9 + R14-3: ConvoPeq.md 2026-09-21 16:16:35・FRESH・HEAD vs working tree の差分位置を明示）
  - P0-F 数値三重裏付け（R13-1）/ E-1c 独立再検証（R13-3）/ 二重補償二重検索（R13-4）
  - 文献裏付け取得（R13-7 + R14-4: 表現を弱める）/ O-20 棚卸し（R13-8）
  - O-19 解決確定（R13-9・v2.6 内部矛盾を解消）
  - 検証スクリプト実在: tmp/v26_p0f_gate_calc.py（再実行 PASS）・authoritative モデル（再実行 PASS）
  - R14-6: Latency.cpp は本リポジトリに存在しないことを訂正
  - R14-7: PolyphaseGainCandidateRef.h は HEAD 未登録・working tree untracked・102 行骨格のみを訂正
  - 残: ユーザー判断（O-1〜O-20 / G-0 / T-1 / T-2 / T-3 / Phase 0 GO / O-16 C0）

[Step 0.0] Phase 0 着手直前の再実測（R14-3 反映）
  - python output_sourcecode_markdown.py --check → FRESH 確認
  - git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests' → 出力なし確認
  - git diff --stat HEAD -- src/ → tests のみ差分であることを確認
  - tmp/v26_p0f_gate_calc.py + doc/work113/model_polyphase_20260920.py → 再実行

[Step C0] evidence policy 単独 commit（O-16・R11-8 / R12-10）

[Step 1] §2 B-1 Phase 0（production 変更 0）— G-0 + T-1/T-2/T-3 + C0 が前提
  - 0-0 G-0 承認 + 9/21 snapshot FRESH 再確認 + 0-0 検証スクリプト再実行
  - 0-1 Baseline → 0-1b REF-FIDELITY（§2.9 契約の 5 項目実装 + R14-7 反映）→ 0-2 Candidate
  - 0-3 周波数（E-1 base gate + E-1c cand gate + E-1b 記録）→ 0-4 block/reset + macro assert
  - 0-5 SoftClip（R12-4 数値契約 + R14-1 saturation event 定義）→ 0-6 float/double
  - harness は凍結（O-1）・**BUILD-ID 行は Phase 0 から必須（R14-2）**

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は E-1（base）+ E-1c（cand no-regression）として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §3 F-2（test-only・案 A・3 行 swap・R14-3 で HEAD 基準 diff anchor）

[Step 4] §3 F-1（A→B→B'→C・failureReason 主判定・R12-11 の移行注意）
  - 移行方針: 既存 private direct-call `checkNoConflictingTransitions` → public `validatePublication`
  - 移行条件: 第 4 段階（transition）まで到達できる world で semantic/topology/resource を PASS させる構築（既存 `ValidatePublication_RejectFromTransition` テストの構造に倣う）
  - 一括移行・旧 test 削除はまだ行わない

[Step 5] O-* commit（O-18 ジェネレータ修正を含む・O-1 は characterization 完了後に解凍）

[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option 適用 → flag ON 実測（**BUILD-ID 行必須・R12-7 + R14-2**）
  - 4 経路 × 8 構成で P0 再確認（§4 マトリクス）
  - Phase 3-A: **0.75^N 補償不存在の再確認（R13-4 の二重検索を再実行）** → flag default ON
  - Phase 3-B characterization → 3-C calibration → 3-D rigcheck threshold（commit 分離維持）

[Step 7] 別課題: R-2（parseOnOff 前例）→ R-1 → B-3, D-1（BUILD-ID 統合）, D-2 / O-20（isBadSample 閾値統一の検討）
```

---

## 7. 検証計画（v2.7 §7 継承 + R14-3 で ConvoPeq.md --check を Step 0.0 に明記）

### 7.1 検証スクリプト実在表（v2.7 §7.1 継承）

| ファイル | 目的 | 状態 |
|----------|------|------|
| `doc/work113/model_polyphase_20260920.py` + `_results.txt` | authoritative モデル | 実在・**v2.7 再実行 PASS**（係数 6 design・DC・full-chain dev） |
| `tmp/v24_ripple_band.py` / `tmp/v24_d1_invariance.py` / `tmp/v24_latency_probe.py` | dev 全桁再現 / D1 81 測定 / latency peak | 実在 |
| `tmp/cio_literal.py` / `tmp/cio_octave.m` / `tmp/maxima_check.mac` | 逐語転写 / Octave / Maxima | 実在（v2.5 実績を継承・本版は Python 系で再検証） |
| **`tmp/v26_p0f_gate_calc.py` + `tmp/v26_p0f_gate_calc_results.txt`** | R12-1/R12-2/R11-3 | **v2.7 再実行 PASS** |
| ~~`tmp/v25_*.py` 3 件~~ | v2.5 §7.1 が参照したが**実在しない**（保存漏れ） | **計画書から削除済み（v2.6 §1）** |

### 7.2 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| ConvoPeq freshness | `python output_sourcecode_markdown.py --check` | **FRESH**（2026-09-21 16:16:35・NEWER_SRC_COUNT=0・**Phase 0 着手直前に再実測**） |
| production diff | `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | 出力なし |
| B-1 attribution | `build\Release\AudioEngineHarness.exe`（引数なし） | §2.4 全 row |
| F-2 | `--buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS |
| B-1 Phase 0/1 | 同上（flag ON/OFF rebuild） | OFF ≈ 0.75^N / ON ≈ 1.0・**log 先頭に BUILD-ID 行（R14-2 で Phase 0 から必須）** |
| 静的解析 | cppcheck 2.21.0（`C:\Program Files\Cppcheck\cppcheck.exe`） | CIO.cpp 指摘 0 |
| clang-tidy 23.1.1 | compile_commands 経由のみ（単独実行不可） | — |

### 7.3 本版のツール census（v2.7 §7.3 継承 + R14-3 追加）

- **context-mode MCP**: `ctx_batch_execute`（並列 8・git/diffstat/rg 集約）・`ctx_search`・`ctx_execute`
- **rtk (WSL版)**: `wsl bash -lc '… ~/.local/bin/rtk rg …'`（出力圧縮・`'…'` 外側引用規約 준수）
- **AiDex MCP**: index 整備済み（487 files / 20,564 items / embeddings 有効）
- **serena MCP**: `get_symbols_overview(CustomInputOversampler.cpp)`・`find_symbol`・`search_for_pattern`
- **semble**: MCP search + CLI `semble.exe search`（両系で同一局在に収束）
- **cocoindex (ccc.exe)**: `ccc search polyphase gain center tap`
- **graphify**: `graphify query 'CustomInputOversampler interpolateStage'`
- **tgrep**: index FRESH 確認後 `tgrep search 'convValue'` で :557 の唯一性を cross-check
- **ast-grep (WSL)**: `convValue *= 2.0` 1 hit / `centerValue *= 2.0` 0 hit（不在の構造的証明）
- **rg (ripgrep/WSL)**: 0.75・compensate・F-2・F-1・CMake の全 census
- **cppcheck 2.21.0 / clang-tidy 23.1.1**: version 確認
- **output_sourcecode_markdown.py --check**: 本版 §1.1 で **FRESH 確認済み**（R14-3 で Step 0.0 に再実測工程を明記）
- **Python 3.14.7 / NumPy / SciPy**: 独立再計算・モデル再実行（Phase 0 着手時に 0-0 で再実行）
- **未使用**: Dr.Memory（本監査は compile-time 数値照合であり動的メモリ計測の対象外のため N/A）・Obscura/Crawl4AI/DDGS/Trafilatura/context7/firecrawl/github/MSLearn（本版の調査対象に該当ページがなく未使用）

---

## 8. ロールバック（v2.7 §8 継承・変更なし）

§1 commit → `git revert` ／ B-1 behavioral → compile-time rollback（flag OFF rebuild）／ B-1 source → `#if` ブロック + option 削除 ／ Shadow → test-only revert ／ F-2 → 3 行 swap の再 swap（HEAD 基準） ／ O-18 ジェネレータ → revert 後に ConvoPeq.md 再生成。

---

## 9. AGENTS.md 修正パッチ（v2.7 §9 継承・変更なし）

v2.5 §15 の内容をそのまま維持: `AGENTS.md` の「**Maxima は未インストール**」は誤り（`C:\maxima-5.50.0\bin\maxima.bat --very-quiet --batch-string` で稼働確認済）。O-10 commit 時に同時適用。

---

## 10. 監査ログ・参照（v2.7 §10 継承 + R14-3 で本セッション再実測分を追加）

| ファイル / 手段 | 確認 | 結果 |
|----------------|------|------|
| `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | production 差分 | ✓ 出力なし（0 件・R14-3 で再確認） |
| `git diff --stat HEAD -- src/` | tests のみ差分の確認 | ✓ BassBuzzMeasurement +634/−2 / PublishPipelineIntegrationTests +8 / check_layout_offsets.py -52（R14-3） |
| `python output_sourcecode_markdown.py --check` | baseline | ✓ FRESH（2026-09-21 16:16:35・NEWER_SRC_COUNT=0・R14-3 で再確認） |
| `git ls-tree -r HEAD --name-only \| grep -i latency` | Latency.cpp 存在確認 | ✓ 該当なし（R14-6 で訂正） |
| `git ls-tree -r HEAD --name-only \| grep -i PolyphaseGainCandidateRef` | Ref.h 存在確認 | ✓ 該当なし（R14-7 で訂正・working tree untracked） |
| `rg 'CONVOPEQ_CORRECT_POLYPHASE_GAIN' CMakeLists.txt` | CMake flag 未適用 | ✓ 0 hit（R13-5） |
| `rg 'convValue \*= 2.0' src/` | 局在 | ✓ CIO.cpp:557 のみ（R13-6） |
| `rg 'centerValue \*= 2.0' src/` | 不在 | ✓ 0 hit（R13-6・Ref.h 除く） |
| `rg '0.75' src/`（tests 除外）+ `compensate/makeup` | 二重補償 | ✓ 補償 0 件（R13-4） |
| `rg 'checkNoConflictingTransitions\|validatePublication' src/` | F-1 | ✓（h:64/h:101・cpp:8/:38/:169・R13-5） |
| `rg 'configureProbeFlatEQ\|setAutoGainStagingEnabled' src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | F-2 | ✓ HEAD 1370/1374 vs working tree 1799/1803（R14-3 で乖離を明示） |
| `CustomInputOversampler.cpp:24-48` 精読 | isBadSample | ✓ divergence 確認（R13-8・O-20） |
| `FastTanhApprox.h` 精読 | SoftClip clipThreshold | ✓ 4.5（R14-1 で saturation event 定義に使用） |
| `doc/work113/model_polyphase_20260920.py` 再実行 | authoritative | ✓ v2.7 再実行 PASS |
| `tmp/v26_p0f_gate_calc.py` 再実行 | R12-1/R12-2 | ✓ v2.7 再実行 PASS |
| AiDex session | index 鮮度 | ✓ |

---

## 11. 判断待ち項目（v2.8）

### 11.1 ユーザー意思決定（v2.7 §11.1 継承・変更なし）

O-1（**凍結→完了後 commit**・R12-10）/ O-2〜O-17 は v2.5 §11.1 の推奨を維持 / **O-18（generator 修正）commit 推奨** / **O-19 解決確定**（対応不要）/ **O-20 将来対応**（対応不要・Phase 0 の対象外）/ G-0 / Phase 0 GO / F-2 案 A / F-1 全実施 / Phase 3 commit 分割。

### 11.2 残存技術判断（確定案提示済み・承認のみ）— 3 件（v2.7 §11.2 継承 + R14-2 で T-3 を Phase 0 必須に更新）

| ID | 論点 | v2.8 確定案 | 根拠 |
|----|------|-------------|------|
| **T-1** | REF-FIDELITY gate の実行条件 | §2.9 の 5 項目契約（骨格のままでは実行不能・R12-8 の性格明記 + **R14-7 で未実装箇所の明文化**） | R10-6 / R11-9 / R12-8 / R14-7 |
| **T-2** | cand D1 の gate 可否 | **gate する（E-1c 絶対形）**。base E-1 と E-1c の 2 本立て。f̂ 0.40/0.45 は記録のみ | R12-2 / R13-3（独立再検証） |
| **T-3** | BUILD-ID gate の必須化 | **Phase 0 の全測定 log に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` を必須化（欠落 = 測定無効）**（**R14-2 で Phase 0 から必須化**） | R12-7 / R14-2 |

### 11.3 B-1 採用案の前提条件（v2.7 §11.3 継承 + R14-1/R14-2/R14-4/R14-5 反映）

1. DESIGN-CONTRACT-A + **T-1/T-2/T-3** 明示承認（G-0）
2. REF-FIDELITY bitwise（§2.9・shadow fidelity gate・**R14-7 で未実装箇所の追加実装が前提**）
3. round-trip DC = 1.0 ±1e-6
4. P0-B/C passband 維持（8 構成・**P0-C は `Fs_in = baseRate` で正規化・R14-5**）
5. **P0-D stopband 維持（decimator 責務）**
6. **E-1 base D1 = −9.5424 ±0.01 dB + E-1c cand D1 ≤ −9.442 dB @f̂≤0.35（interpolator 責務・R13-3）**
7. P0-F latency 不変（理論式 §2.8 の基線・**R14-6 で本リポジトリに対応する `Latency.cpp` が存在しないことを訂正**）
8. P0-I 数値契約（NaN/Inf FAIL・**saturation event 数 FAIL（R14-1 で 1 行固定）**・指標記録・O-20 は対象外）
9. P0-C' differential 帰属 + P0-G（絶対誤差 gate・相対誤差記録）
10. D1 測定は補正軸 + expected bin 主報告（R9-1/R9-2/R10-4/E'-5）
11. **BUILD-ID 必須・Phase 0 から適用（R14-2）**

### 11.4 v2.8 最終判定表

| 項目 | 判定 | v2.7 から |
|------|------|-----------|
| P0-F 数式 | **確定（三重裏付け R13-1）**+ R14-6 で `Latency.cpp` 参照訂正 | 強化 |
| E-1c 絶対形 | **確定（独立再検証 R13-3）** | 強化 |
| **P0-I saturation event 定義** | **確定（1 行固定 R14-1）** | 強化 |
| **BUILD-ID 必須化** | **Phase 0 から必須（R14-2）** | 格上げ |
| **案 E の表現** | **外部文献と整合・Phase 0 で検証（R14-4）** | 弱化 |
| **P0-C / P0-F レート混同回避** | **R14-5 で明文化** | 追加 |
| D/E 責務・P0-I 契約 | **確定維持 + R14-1 反映** | 強化 |
| **Shadow Reference 5 項目契約** | **R14-7 で未実装箇所を明文化** | 強化 |
| **F-2 行番号（HEAD vs working tree）** | **R14-3 で乖離を明示** | 強化 |
| source baseline | **更新（16:16:35・FRESH・本セッションで --check 再実測）** | 維持 |
| O-19 | **解決確定（v2.6 内部矛盾を解消）** | 維持 |
| O-20 | **将来対応・non-blocking** | 維持 |
| 二重補償リスク | **不存在を二重検索で再確認（R13-4）** | 強化 |
| 案 E の理論的妥当性 | **外部文献と整合・Phase 0 で確定（R13-7 + R14-4）** | 弱化表現 |
| §1 O-1〜O-20 | GO 候補（O-15 commit 前に 5 項目実装・R14-7） | 強化 |
| §3 F-2 / F-3 / F-4 | GO 候補 / 記録継承 / 記録継承 | 強化（R14-3） |
| §4 F-1 | GO 候補（R12-11 維持） | 変更なし |
| B-1 原因分析 | GO（5 系統 + 数値計算 + 文献） | 強化 |
| **B-1 Phase 0** | **GO（G-0 + T-1/T-2/T-3 + C0 承認 + Step 0.0 再実測が前提・R14-2 反映）** | 条件強化 |
| B-1 Phase 1+ / 案 E / Production patch / Calibration | **HOLD / 有力仮説 / HOLD / HOLD** | 変更なし |
| 技術的未確定 | **3 件（T-1/T-2/T-3）・確定案提示済み** | 変更なし |
| 計画書品質 | v2.7 内部矛盾（O-19）+ R14 一括反映 | 強化 |

---

## 12. エビデンス（v2.8）

- **5 系統クロスチェック**: v2.5 §12.1 を継承 + v2.7 で **authoritative モデル・v26 スクリプトを再実行**し PASS を再確認（Python 3.14.7・NumPy/SciPy）。本版 v2.8 では **再実行は未実施**（Step 0.0 として Phase 0 着手直前に実施する旨を §6 で明示）
- **v2.7 の独立数値計算**: latency 4 構成（15.00/255.00/290.25/582.25）・DC N=1..8・D1 構造定数（−9.542425）・E-1c 独立 30 点判定（FAIL 0・margin 36.7 dB）— R13-1/R13-2/R13-3
- **v2.8 の独立検証**: `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` → 出力なし、`git diff --stat HEAD -- src/` → tests のみ、`python output_sourcecode_markdown.py --check` → FRESH、`git ls-tree -r HEAD --name-only | grep -i latency` → 該当なし、`git ls-tree -r HEAD --name-only | grep -i PolyphaseGainCandidateRef` → 該当なし — R14-3/R14-6/R14-7
- **ツール実測**: §7.3 の census 表のとおり

---

## 13. v2.8 で追加した確定事項（R14 一覧）

- **R14-1** P0-I saturation event 定義の 1 行固定（SoftClip clipThreshold 4.5）
- **R14-2** BUILD-ID を Phase 0 から必須化（Phase 0 record → Phase 0 必須に格上げ）
- **R14-3** HEAD vs working tree の差分位置を明示（tests のみ・F-2 行番号の乖離を明記）
- **R14-4** 案 E の表現を弱める（外部文献で裏付け → 外部文献と整合・Phase 0 で検証）
- **R14-5** P0-C / P0-F のレート混同回避を明文化（baseRate と stageRate(s) を別個名で参照）
- **R14-6** `Latency.cpp:29-31` / `Latency.cpp:120` の非実在を訂正（標準的群遅延式の理論式・実装は shadow reference / 計測側で明示）
- **R14-7** Shadow Reference 5 項目契約の未実装箇所を明文化（PolyphaseGainCandidateRef.h は HEAD 未登録・working tree untracked・102 行骨格のみ）

---

*本書は v2.7（`remediation_plan_20260922_v2.7_revised.md`）の再改訂版 v2.8 である。*
*v2.7 本文は参照文書として維持し、**R14-1〜R14-7 が v2.7 と矛盾する箇所では本書を優先する**。*
*R5〜R14 は本書でも有効（上記是正点を除く）。中間エビデンスは `remediation_v22_intermediate_20260921.md`、パッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*
**継承の監査・確定事項**: R13-1〜R13-9 / R12-1〜R12-12 / R11-1〜R11-10 / R10-1〜R10-13 / R9-1〜R9-6 / R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。