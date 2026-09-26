# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v3.1）

- **版**: v3.1（v3.0 の再改訂・**R17-1〜R17-7** を新規確定）
- **前版**: `doc/work113/remediation_plan_20260922_v3.0_revised.md`（R16-1〜R16-3）
- **本改訂監査実施日**: 2026-09-21（同一作業ツリー・ZCode 環境・外部監査指摘 2 ブロッカー + 追加推奨 5 系統を反映）
- **基準ソース**: HEAD = `8f127bfe`（親 `c4a08171`）。production `src/*.cpp/h`（tests 除外）の未 commit 差分 **0 件**（`git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` → 出力なし）を本セッションで再実測。
- **★ source baseline**: `python output_sourcecode_markdown.py --check` → **2026-09-21 16:16:35 生成・NEWER_SRC_COUNT=0・STATUS: FRESH** を本セッションで再実測し、v3.0 と同一値であることを確認。本版の authoritative snapshot を維持。
- **★ working tree と HEAD の差分位置（本セッションで再実測・R14-3 継承）**: production `src/convolver` `src/dsp` `src/core` `src/audioengine` `src/eqprocessor` は **差分 0**。差分は (a) `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` **+634/−2**・`PublishPipelineIntegrationTests.cpp` **+8/−0**、(b) `src/tools/check_layout_offsets.py` **削除 52 行**、(c) `output_sourcecode_markdown.py` **+11/−4**・`doc/work113/residual_tasks_20260919.md`、に限定。**`PolyphaseGainCandidateRef.h` は untracked・HEAD 未登録・102 行骨格**（本セッションで全 102 行を精読し骨格構造を確認）。
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `_results.txt`。**本セッションで再実行を実施済み（Step 0.0 の一部を先行実行・§0 R17-7・§12）**。
- **v3.1 の位置付け**: v3.0 の GO/HOLD・案 E（candidate hypothesis）・Phase 0 の 5 構造・R5〜R16 確定事項は**原則維持**。本改訂は外部監査指摘 (a) **B1: P0-I saturation gate 判定式 `abs(linear + normalize) > R_s − 1.0` が数学的に未定義 → R17-1 で per-phase 帰属 gate として完全定義**、(b) **B2: Phase 0「commit 禁止」と C1 Shadow 実装 commit の境界矛盾 → R17-2 で commit マトリクス化**、(c) **REF-FIDELITY bitwise 契約の前提条件補強 → R17-3**、(d) **Shadow 追加検証 5 種 → R17-4**、(e) **R14-6 の誤り（Latency.cpp 実在）訂正 → R17-5**、を扱う。
- **★ R17-1 確定事項（B1 解消）**: P0-I の saturation 帰属 gate は「Phase 0 着手時に固定」の留保を廃止し、**動作点・イベント定義・判定関数をすべて本計画書で確定**する。判定は `R_s` に依存しない **per-phase subset + envelope (θ_evt/2) + 点wise 出力恒等式** の 3 条件で構成し、基礎恒等式（candidate up 出力 = base up 出力の conv 位相 bitwise 一致 / center 位相 bitwise 2 倍）を事前検証項目 I-0 として数値確定済み（§2.8・§12）。
- **★ R17-2 確定事項（B2 解消）**: §2.10 の「commit 禁止」は **characterization 実行中の全 commit 凍結** に限定し、C0（evidence policy）/ C1（Shadow 実装・test-only）/ C2（characterization 証跡・freeze 後）の独立 commit を明示許可する（§1.3 再定義）。
- **案 E は有力仮説（candidate hypothesis）のまま**。DC = 0.75^N は本セッションでも warm-start impulse（Σh_rt = 0.75^N 厳密）で再現。確定判断は G-0 承認 + Phase 0 REF-FIDELITY/P0-A〜P0-I 全 PASS + 条件D を満たす最終判断まで保留（R5-4 継承）。
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**。**Phase 0 では CMakeLists.txt に追加しない（R15-2 継承・本セッション再確認: CMakeLists.txt / production src に 0 hit）**
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）
- **R5〜R17 は本書でも有効（是正点を除く）。R17 と矛盾する v3.0 の箇所では本書を優先する**

---

## 0. v3.0 → v3.1 改訂サマリ

### 新規確定（R17-1 〜 R17-7）

| ID | 区分 | 内容 | v3.1 での確定 |
|----|------|------|----------------|
| **R17-1** | **P0-I saturation gate の数学的確定（B1 解消）** | v3.0 R16-2 の `abs(linear + normalize) > R_s − 1.0` は `linear`/`normalize` が未定義で第三者が再現不能（**NO-GO → 本版で完全定義**）。新契約は production の点wise 性を前提に **per-phase 帰属**: 基礎恒等式 I-0（candidate up = base up の conv 位相 bitwise 一致 / center 位相 bitwise 2 倍）の下で (c1) conv 位相 event 一致 (c2) center 位相 base ⊆ candidate (c3) center 位相 envelope \|x_b\| ≥ θ_evt/2 (c4) center 位相出力 = F(2·x_b) 点wise 恒等。動作点（wiring・θ/κ/asym・刺激集合）も本計画書で固定し「Phase 0 着手時に固定」の留保を廃止。R_s・ΔS は記録指標に格下げ（gate は c1〜c4 のみ・閾値選択に依存しない） | gate 判定関数を完全定義 |
| **R17-2** | **Phase 0 commit 境界の確定（B2 解消）** | v3.0 §2.10「commit 禁止」と §1.3 C0/C1 の矛盾を解消。「commit 禁止」は **characterization 実行中の全 commit 凍結** に限定し、C0（evidence policy・ユーザー承認制）/ C1（Shadow 実装・test-only）/ C2（freeze 後の証跡）を独立 commit として明示。全 commit 時の不変条件（production src diff = 0・CMakeLists/build.bat 未変更）を追加 | commit マトリクス化 |
| **R17-3** | REF-FIDELITY bitwise 契約の厳密化 | bitwise 一致は「同一 BUILD-ID（同一 compiler configuration・同一 ISA path）」でのみ意味を持つ。**production と Shadow を同一 test executable・同一翻訳単位で compile** して浮動小数点環境を同一化。bitwise mismatch は **直ちに production defect と断定しない** — triage 5 分類（(1) state/history (2) coefficient (3) operation-order (4) ISA/compiler (5) production defect）を順に特定・記録するまで Phase 0 の残 gate を進めない | 契約の補強 1 段落 |
| **R17-4** | Shadow 追加検証 5 種 | §2.9 契約に **(A) coefficient-only test**（Stage 構造体全フィールドの bitwise 比較・6 design）**(B) zero-input test** **(C) constant-input test**（base 0.75^N / cand 1.0）**(D) impulse test**（warm-start 規約・単段 h_rt = 2·(conv⋆conv) + c·δ[centerTap] の構造恒等式）**(E) random block partition**（partition 集合・one-shot == partitioned bitwise）を追加。impulse 検定は **warm-start**（δ 前に ≥16·Σtaps のゼロ）とする（cold-start fresh instance の初样応答は定常応答と一致しないことを本セッションで数値確定） | REF-FIDELITY の価値強化 |
| **R17-5** | **R14-6 の訂正（Latency.cpp 実在）** | v3.0 §1.1 の「`git ls-tree -r HEAD | grep -i latency` → 該当なし」は**誤り**（本セッション再実行 → **11 件ヒット・うち production C++ 1 件**）。`src/audioengine/AudioEngine.Processing.Latency.cpp` は **tracked・未変更・CMakeLists.txt:1185 で compile・`estimateOversamplingLatencySamplesImpl` :10-48 が P0-F 式と同一**（static_assert :6-8 が `isLinearPhaseFIR && isSymmetricUpDown` を要求）。P0-F の公式は「理論式」であると同時に「production latency 実装式」と一致 → P0-F gate を「**Latency.cpp 実装式 == 実測 peak**」の強い契約に再定義 | 計画書内の事実誤り訂正 |
| **R17-6** | besselI0 行範囲訂正 | v3.0 §2.2/§10 の「`besselI0` :144-216」は誤り。正しくは **`besselI0` :144-157**（:159-216 は `dotProductAvx2`）。private 関数の級数和実装（`term *= xx/(4n²)`・n<100・収束判定 `term < sum·1e-18`）自体は R16-1 の 10 手順記述と一致 | 行範囲の正確化 |
| **R17-7** | Step 0.0 の部分先行実施 | 本セッションで (i) authoritative モデル再実行 → **保存結果と一致（例外: D2 の −100 dB 以下の記録専用指標で最後の桁 ±0.01 dB・6 行）**、(ii) P0-F/E-1c/DC 検証済みの `tmp/v26_p0f_gate_calc_results.txt` 照合、(iii) 新規 `tmp/v31_p0i_attribution_check.py`（42 判定 + 2 記録・**ALL PASS**）と `tmp/v31_octave_check.m`（Octave 11.3.0・base/cand impulse 恒等式 **max diff 0.0**）を作成実行。Phase 0 着手直前の Step 0.0 は **FRESH 再確認 + スクリプト再実行の維持確認のみ**に短縮可能 | 再実測のエビデンス更新 |

### 監査指摘の反映表（v3.0 レビュー → 本版）

| レビュー指摘 | 本版の対応 | 節 |
|--------------|-----------|-----|
| B1: P0-I 判定式が数学的に未定義・第三者再現不能 | **R17-1** — per-phase 帰属 gate を数式で完全定義（動作点も固定・「Phase 0 着手時に固定」撤回） | §2.8 P0-I |
| B2: Phase 0 commit 禁止 と C1 commit の矛盾 | **R17-2** — commit マトリクス（C0/C1/凍結窓/C2）+ 不変条件 | §1.3 / §2.10 |
| bitwise 一致は「同一演算順序」だけでは不十分（compiler/flags/ISA） | **R17-3** — 同一 BUILD-ID 前提 + 同一翻訳単位 + mismatch triage 5 分類 | §2.8 REF-FIDELITY / §2.9 |
| Shadow に coefficient/zero/constant/impulse/random-partition test を追加 | **R17-4** — 5 種を §2.9 に明文化（impulse は warm-start 規約） | §2.9 |
| C0 はリポジトリポリシー変更 → ユーザー承認制 | **R17-2** — C0 をユーザー承認条件の独立 commit に明示 | §1.3 |
| P0-I の saturation attribution は数式で定義すべき | **R17-1** — envelope 恒等式（θ_evt/2）は本セッションで数値検証済み（42/42 PASS） | §2.8 / §12 |
| 固定 3 ブロックだけでは history bug を十分捕捉できない | **R17-4 (E)** — partition 集合 {1,2,3,5,7,11,15,31,63,127,256}+{512,1024}+mixed | §2.9 |
| Latency.cpp の扱い（v3.0 は「存在しない」と記載） | **R17-5** — 実在を再測定・訂正し P0-F を production 実装式と一致させる | §2.8 P0-F |

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-20・本セッションで実測棚卸し）

### 1.1 実測 diffstat（本セッションで再実測）

```text
git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'
  → 出力なし（production src/ 差分 0 件・本セッション再実測）

git diff --stat HEAD -- src/
  src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp     | 636 ++++++++++++++-
  src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp | 8 +
  src/tools/check_layout_offsets.py                        | 52 --
  3 files changed, 642 insertions(+), 54 deletions(-)

python output_sourcecode_markdown.py --check
  baseline Generated : 2026-09-21 16:16:35  (ConvoPeq.md)
  NEWER_SRC_COUNT    : 0
  STATUS             : FRESH — snapshot は現行ソースを反映しています

git ls-tree -r HEAD --name-only | grep -i latency
  → 11 件ヒット（うち production C++ 1 件: src/audioengine/AudioEngine.Processing.Latency.cpp）
    ★ v3.0 §1.1 の「該当なし」記録は誤り → R17-5 で訂正

git ls-tree -r HEAD --name-only | grep -i PolyphaseGainCandidateRef
  → 該当なし（Ref.h は HEAD 未登録・working tree untracked・102 行骨格）
```

| ID | パス | 実測 | v3.1 推奨 |
|----|------|------|-----------|
| **O-1** | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634/−2** | **Phase 0 characterization 完了まで凍結（commit しない）→ 完了後に cleanup/分割して commit**（R12-10 継承・O-1 凍結解除は C2 後） |
| **O-1b** | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8/−0** | O-1 と同一運命（凍結） |
| **O-2** | 台帳更新 | 独立 commit | 独立 |
| **O-3 / O-12** | `ConvoPeq.md` | **16:16:35 再生成・FRESH（NEWER_SRC_COUNT=0・本セッション再実測）** | **commit しない**（再生成資産・`--check` FRESH を維持） |
| **O-4** | `Testing/Temporary/CTestCostData.txt` | D（削除） | 現状維持 |
| **O-5** | `.opencode/opencode.json` | untracked | 触らない |
| **O-6** | push | ahead 2 | **ユーザー手動** |
| **O-7 / O-10** | `AGENTS.md` | +78/−3 | 環境記録として commit 可。R11-7 の Maxima 誤記修正（§9）を同時に適用 |
| **O-8** | `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` | M（追跡済みバイナリ） | commit しない → 将来 `.gitignore` + `git rm --cached` |
| **O-9** | tool-inventory 2 件 | untracked | 触らない（ユーザー判断） |
| **O-11** | `doc/work113/residual_tasks_20260919.md` | +71/−1 | 台帳として commit 可 |
| **O-13** | `doc/work113/*.md` + model py/results untracked | 本版追加で v3.1 1 件 | 承認版+台帳+モデルを 1 commit |
| **O-14** | `.mcp.json` | +33/−23 | commit（環境記録） |
| **O-15** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | untracked（102 行） | O-13 に併同可。**commit 前に §2.9 契約（5 項目 + R17-4 追加 5 検証）を実装した版へ** |
| **O-16** | `evidence/epoch_reclaim_audit.json` | M・Bin | **C0 で追跡外化**（R11-8 確定・§1.3） |
| **O-17** | `doc/WorkBuddy_AI_Desktop_Toolchain.md` | untracked | O-9 と同一方針 |
| **O-18** | `output_sourcecode_markdown.py` | +11/−4 | **commit 推奨**（R12-6・独立 commit）。本版 snapshot は修正版で生成済み |
| **O-19** | `src/tools/build_identity_gate.py` / `tools/check_layout_offsets.py` | 解決済み（R13-9） | **解決**。D-1（build identity gate）は `src/tools/build_identity_gate.py` を継続使用 |
| **O-20** | `src/CustomInputOversampler.cpp:24-33`（scalar）vs `:40-48`（AVX2 1e20）isBadSample 閾値不整合 | R13-8 確認 | **将来対応（non-blocking）**。P0-I の NaN/Inf gate に影響なし（R17-1 で明記済み） |

### 1.2 O-16 と Phase 0 の衝突（R11-8 確定方針の維持）

Phase 0 測定が tracked `evidence/` を書き換える構造的欠陥は未解決（追跡外化は未適用）。手順は §1.3 に固定済み。**C0 はリポジトリポリシー変更であるため、技術的 GO に加えてユーザー承認を必要とする独立 commit とする（R17-2）**。

### 1.3 evidence/ 追跡外化手順と **commit マトリクス（R17-2 で再定義）**

```bash
# C0: evidence policy 単独 commit（ユーザー承認後・Phase 0 測定より前に実施）
echo "evidence/" >> .gitignore
git rm -r --cached evidence/
git add .gitignore
git commit -m "chore(evidence): untrack generated verification artifacts"
# 以後、測定出力は --buzz-out= で tmp/（.gitignore 済み）へ

# C1: Phase-0 Shadow Reference 実装 commit（test-only）
#   対象: PolyphaseGainCandidateRef.h（§2.9 完全実装版）+ PolyphaseGainFidelityTests.cpp
#        （新設・add_executable 40 番目）+ src/tools/build_identity_gate.py の
#         6 要素 [BUILD] ブロック出力 + 本計画書（v3.1）
#   位置: Phase 0 の 0-1 測定より前（0-1b REF-FIDELITY がこれに依存するため）
#   不変条件: 下記 commit 不変条件を C1 の時点で満たすこと

# ── 0-1 〜 0-6 characterization 実行中 = commit 全凍結（frozen window）──
#   production src/ diff = 0 を全測定実行時に再確認する

# C2: characterization 結果 commit（Phase 0 freeze = 全 P0 gate 判定確定後のみ）
#   測定 log・CSV は tmp/ 起点で添付判断・O-1 凍結解除は C2 の後
```

**commit マトリクス（R17-2）**:

| 区間 | 許可される commit | 禁止 |
|------|-------------------|------|
| Step 0 / 0.0 完了後 | **C0**（evidence policy・独立 1 commit・**ユーザー承認後**） | production 変更・他ファイル混在 |
| C1 実施時 | **C1**（Shadow 実装・test-only・本計画書を含め可） | **production src / CMakeLists.txt / build.bat への変更は一切不可（R15-2）** |
| 0-1 〜 0-6 測定実行中 | **commit 全凍結（frozen window）** | production 変更 / 測定中 commit / 測定結果 commit |
| Phase 0 freeze 後 | **C2**（characterization 証跡） | O-1/O-1b 凍結解除（C2 後に解凍） |
| Phase 1 以降 | CMake flag 適用 + production candidate | Step 2 / 2.5 のユーザー GO 前は着手不可 |

**commit 時不変条件（C0/C1/C2 の全 commit で自動確認）**:

```bash
git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'   # → 空であること
git status --porcelain -- CMakeLists.txt build.bat                 # → 空であること
```

この 2 条件のいずれかが非空の commit は Phase 0 契約違反（R15-2/R12-5/R17-2）。

### 1.4 Shadow Reference untracked 報告（R14-7 + R15-3 + R16-1 継承・本セッション精読済み）

`PolyphaseGainCandidateRef.h`（102 行）は HEAD に存在しない（untracked）。実装済みは `expectedDcRoundTrip` / `computeD1Bins` / `PolyphaseGainShadow`（reset/prepare/stageCount/centerPhaseGain まで）のみで、**processUp/processDown・prepareStageModel・履歴同期・denorm/isBadSample 位置同期は未実装**。**O-15 commit（C1）の前提条件は §2.9 の 5 項目契約 + R17-4 追加 5 検証の実装**。

---

## 2. §2 B-1: CustomInputOversampler up/down round-trip 利得欠陥

### 2.1 確定している事実（v3.0 §2.1 継承・本セッションで再実測）

```text
主因        interpolateStage の center 位相に ×2 が無い（polyphase 利得規約の不対称）
欠陥局在    src/CustomInputOversampler.cpp :557（convValue *= 2.0 のみ・ast-grep 1 hit）
            :563-564（center に ×2 なし・centerValue *= 2.0 は 0 hit）
            decimateStage :658（centerCoeff * centerSample + conv tap 累加・×2 なし＝DC 利得 1.0 で正規）
round-trip  DC: base 0.75^N / cand 1.0（authoritative モデル再実行・本セッションで再確認）
段ごとの和  conv 分岐: Σcoeffs 0.5 → ×2 → DC 1.0（正規）／center 分岐: 0.5 → ×2 なし → DC 0.5（不対称）
利益        (4/3)^N = +2.498775 / +4.997549 / +7.496324 dB（1/2/3 段・v2.7 再計算継承）
画像        up 出力の像/トーン = 1/3 = −9.542425 dB（構造定数・authoritative 再実行で −9.54 一致）
係数        FIRsum 1.0 / center 0.5 / convSum 0.5（全 6 design・本セッション再実行で再現）
契約不整合  isLinearPhaseFIR / isSymmetricUpDown（h:21-22）と矛盾
production  変更 0 / commit 0（本セッション git diff 実測）
```

**★ 本セッションで追加確定した per-phase 恒等式（R17-1 の基礎・`tmp/v31_p0i_attribution_check.py` ALL PASS）**:

```text
(1) candidate up 出力 = base up 出力の conv 位相 bitwise 一致（DC/正弦/PRNG すべて）
(2) candidate up 出力の center 位相 = base up 出力 center 位相の bitwise 厳密 2 倍
    （×2 は IEEE 754 で指数増分のみ・丸め誤差なし。例外は |x| < 2e-20 の denorm 帯のみ）
(3) up 経路 RMS 増加比: DC で √1.6 = 1.264911064 厳密 / 正弦 1.264899673 / PRNG 1.295801534（記録）
    → R16-2 の前提「candidate では SoftClip 入力振幅が増加」は RMS 意味で定量的に成立
    → ただし DC peak は base == candidate（1.0 対 1.0・peak 不変）— 振幅増加は center 位相サンプルの
       復元が本質であり、peak 系の帰属議論（R10-3・E-8 継承）と整合
```

**peak / mean の帰属不能性（R10-3・E-8 継承）**: up 出力は位相交互 `1.0, 0.5, …`。peak 系で「up 1.0/down 0.75」、mean 系で「up 0.75/down 1.0」— どちらも同じ階段信号の別記述で因果の根拠にならない。因果は分岐係数和の不対称（1.0 vs 0.5）で確定。本セッションの impulse 構造恒等式（§2.9 D 項）が分岐係数和の不対称を直接可視化した。

### 2.2 根本原因の行番号（v3.0 §2.2 継承 + **R17-5 / R17-6 訂正 + 本セッション精読**）

| 算所 | 内容 | 実測手段 |
|------|------|----------|
| **`prepareStage` :287-390（10 手順・本セッションで全行精読・v3.0 の記述と一致）** | (1) `clearStage` :289 / (2) `taps = jmax(3, taps\|1)` :291 / (3) `centerTap = (taps−1)/2` :292 / (4) `centerParity/convParity` :293-294 / (5) Kaiser β（attenuationDb 3 分岐）:301-304 / (6) `i0Beta = besselI0(β)` :305 / (7) **sinc × window** :308-317 / (8) **halfband zeroing** :319-323 / (9) **sum→1.0** :325-333 / (10) **center=0.5** :335 + **非center→0.5** :336-348 / (11) polyphase extraction: `convCount` :350・`convCoeffs/convCoeffsReversed` :360-365・`centerCoeff/centerDelayInput` :367-368・history sizes :369-375 | Read（HEAD:8f127bfe・本セッション全 872 行精読） |
| `interpolateStage` :492-568 | **:557 `convValue *= 2.0`（唯一）** / :558-559 denorm（`fastAbs` + `kDenormThreshold`） / **:563-564 出力（`centerValue *= 2.0` 不在 = 欠陥確定）** / 履歴 copy :504・guard :509-515・center :543-547・shift :567 | ast-grep + rg + Read |
| `decimateStage` :570-723 | silence fast path :583-613 / 境界 guard :620-649 / **:657-658 `centerCoeff * centerSample`（×2 なし＝DC 1.0 正規）** / conv 累加 :667-706 / 事後 :708-718 / shift :722 | Read |
| `processUp` :725-783 / `processDown` :785-872 | hardFallback / corruption clear / stage 逆向き走査 :851-863 | Read |
| `reset()` :452-467（atomic 3）/ `clearAllStages()` :469-484（atomic 1） | Phase 0 は `reset()` 経由必須（R4-3） | Read（本セッション再確認） |
| `prepareSingleStage` :392-413 | SoftClip 局所 OS（`DSPCoreLifecycle.cpp:188` / `:261` で `(31, 90.0, internalMaxBlock)`） | Read + rg |
| `dotProductAvx2` :159-216 / `dotProductDecimateAvx2` :218-285 | conv 専用 SIMD・center 位相は非依存 | serena symbols |
| `isBadSample` :24-33 / `isBadSampleV` :40-48 | scalar: NaN/Inf + \|x\| > 2^53（0x4340000000000000）/ AVX2: NaN + \|x\| > 1e20（**R13-8 divergence・O-20**） | Read（本セッション再確認） |
| **`besselI0` :144-157（private 関数・R17-6 で行範囲訂正）** | 級数和: `term *= xx/(4.0·n²)`・n<100・収束判定 `term < sum·1e-18`。Shadow でも同一級数和を独立実装 | Read |
| **`AudioEngine.Processing.Latency.cpp`（R17-5 で実在確定）** | tracked・**CMakeLists.txt:1185 で compile**。`estimateOversamplingLatencySamplesImpl` :10-48: `stageRate = baseRate·2^(s+1)` :29・`taps[stage]−1 /* up + down */` :30・合計 :31-33・static_assert :6-8 が `isLinearPhaseFIR && isSymmetricUpDown` を要求。**P0-F 式と完全同型**。SoftClip 局所 OS の 15 base-sample 定数 :120-122 あり | Read + git log + rg（本セッション） |
| `DSPCoreDouble.cpp` SoftClip | `musicalSoftClipScalar` :107-131 / `softClipBlockAVX2` :133-224（**点wise・prev は退避のみ** :206/:214-220） / パラメータ マッピング `clipThreshold=0.95−0.45·sat, clipKnee=0.05+0.35·sat, clipAsymmetry=0.10·sat` **:487-489** / local OS 経路 **:503-513** | Read（本セッション） |
| `DSPCoreFloat.cpp` SoftClip | 同構造 :165/:192/:205 / local OS :398/:410 | rg |
| `FastTanhApprox.h` | `SoftClipPadePolicy::clipThreshold = 4.5` :64・scalar 厳密 ±1.0 :104-106・SSE2 クランプ :117・AVX2 クランプ :147-149・**クランプ時 Padé 値 max = 0.9992656388416031**（本セッション計算） | Read |
| `DspNumericPolicy.h` | `kDenormThresholdAudioState = 1.0e-20` :132-138 | rg |

### 2.3 5 系統独立再現 + 数値計算裏付け（v3.0 §2.3 継承 + **本セッションで再実行済み**）

production バイナリ / authoritative モデル / NumPy 逐語転写 / Octave 11.3.0 / Maxima 5.50.0 — 全系統が base 0.75^N・cand 1.0・Δ+2.498775 dB(N=1)・image/tone 1/3 に一致。**本セッションの再実行結果**:

- `python doc/work113/model_polyphase_20260920.py` → `tmp/v31_model_rerun_20260921.txt` 生成。**係数 6 design・DC・full-chain dev・ripple・edge は全て保存結果と一致。例外は D2（単段 round-trip 出力 f̂ vs 0.5−f̂）の −100 dB 以下の記録専用値 4 行で最後の桁 ±0.01 dB**（FFT 窓丸めノイズ・gate 値すべてに非該当）。
- `tmp/v31_p0i_attribution_check.py`（新規）: **impulse 構造恒等式・per-phase 2× 恒等式・P0-I subset/envelope gate・点wise 出力恒等式 の 42 判定 + 2 記録 = ALL PASS**。
- `tmp/v31_octave_check.m`（Octave 11.3.0・独立スカラー実装）: **base Σh = 0.75 / cand Σh = 1.0 / h_rt 構造恒等式 max diff = 0.0**。
- Maxima 5.50.0 は本環境の batch モードで評価結果表示が抑制される（dirty SBCL build・AGENTS.md 記載のインタラクティブ稼働確認は有効）。定数（2.498775/−9.542425 dB・0.75^N 表）は NumPy/Octave/authoritative の 3 系統で十分に独立再現済み。

### 2.4 ライブ帰因テーブル（v3.0 §2.4 継承・引数なし実行の制約）

`[OS_DIRECT]/[EQ_DIRECT]/[OF_DIRECT]` の全 row は v2.5 §2.4 を継承。**唯一の桁違い損失は Oversampler（0.75^N）**・EQ は ratio=1.000000・OutputFilter は −0.110 dB 一定。`rt = up × dn` 成立・preset 非依存・**P0 gate は rt のみ**。再現には引数なし実行が必須（`--buzz*` 引数があると `PublishPipelineIntegrationTests.cpp:1123` の派遣で `:1223` に到達しない）。

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

**構造的恒等式（R5-9 継承 + 本セッションで厳密数値確定）**: 単段 warm-start impulse round-trip で

```text
h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n − centerTap]
     c: base = 0.25（= centerCoeff 0.5 × down center 0.5）/ cand = 0.5
     （S1=31/90 で centerTap = 15・sum = 2·(Σconv)² + c = 2·0.25 + c = base 0.75 / cand 1.0）
     Octave/NumPy で h_rt と予測式の max abs diff = 0.0〜1e-14 で厳密一致（R17-7）
```

passband で cand/base = (4/3)^N。**up 経路単独では peak 不変・RMS ×√1.6**（§2.1 恒等式・SoftClip 入力動作点の変化はここに帰着）。

**外部文献との整合（R13-7 + R14-4 + R15-4 継承）**: MathWorks `dsp.FIRHalfbandInterpolator` 公式文書の「halfband 補間フィルタの係数は出力パワー保存のため補間係数 2 でスケールされる」は **halfband interpolator 全体に対する一般的な設計規約**であり、案 E（center 位相への対称 ×2）と整合するが、ConvoPeq への適用正当性は Phase 0 全 PASS + 条件D の最終判断で確定する。decimate 側に ×2 が無いこと（:658 の正規性）とも整合。

### 2.6 CMake / flag（v3.0 §2.6 継承）

- `option(CONVOPEQ_CORRECT_POLYPHASE_GAIN "Correct polyphase gain convention (B-1 案E)" OFF)` を option 群（:40 近傍）に追加（**Phase 1 まで未適用・R15-2**）
- `add_compile_definitions(CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${...}>)` を `add_subdirectory(JUCE)`（:1043）後・`juce_add_gui_app(ConvoPeq`（:1062）前に配置（**Phase 1 まで未適用・R15-2**）
- 到達確認: CIO.cpp を compile するのは ConvoPeq（`target_sources` :1282・リスト内 :1196/:1243）と AudioEngineHarness の 2 論理ターゲット。**本セッション再検索で `CONVOPEQ_CORRECT_POLYPHASE_GAIN` は CMakeLists.txt / production src に 0 hit = 未適用を再確認**
- C++ は **`#if`**（`#ifdef` 禁止）。既定 OFF / runtime flag 不採用 / **Phase 0 では追加しない（R15-2）**
- **R12-7 + R14-2 + R15-1 + R16-3**: 測定出力の `[BUILD]` 6 要素ブロックを **Phase 0 から必須化**（test-only）。欠落 = 測定無効

### 2.7 Phase 0 characterization（v3.0 §2.7 継承 + **R17-1/R17-3/R17-4 反映**）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）+ T-1/T-2/T-3 をユーザーが明示承認 + 9/21 snapshot FRESH 再確認
0-1  Baseline record-only: round-trip DC = 0.75^N ±1e-6（invariant のみ gate・up/dn 分割は診断ログ）
0-1b REF-FIDELITY（R17-3/R17-4）:
       ① 同一 test executable・同一翻訳単位で production / Shadow を compile（同一 compiler config・
          同一 ISA path・scalar↔scalar / SIMD↔SIMD 固定）
       ② 追加 5 検証（R17-4）: coefficient-only（全 Stage フィールド・6 design bitwise）
          → zero-input → constant-input（base 0.75^N / cand 1.0）→ impulse（warm-start・単段
          h_rt 構造恒等式 + 多段 Σh_rt = 0.75^N/1.0・peak ∈ [floor(D),floor(D)+1]）
          → random block partition（one-shot == partitioned bitwise）
       ③ 本体: シード固定疑似乱数 3 ブロック × 2 preset × ratio {2,4,8} の up/down 出力 bitwise
          + reset 前後状態差分（atomic 3 vs 1）
       ④ 不一致時は triage 5 分類（R17-3）で原因確定まで進めない
0-2  Shadow Candidate E: round-trip DC = 1.0 ±1e-6（必要条件）
0-3  周波数（D1/D2/D3）— D1 は base 側 gate + E-1c cand gate（§2.8）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + R17-4 (E) 拡張）+ candidate マクロ 2 系統一致性 assert
0-5  SoftClip local OS（float + double）— **P0-I は R17-1 の完全固定契約で実施（動作点は §2.8 参照）**
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8 / 相対誤差記録のみ）
```

**D1 定義（R10-4 + E'-5 + R12-2 + R13-3 継承）**: 変更なし — `D1 = |Y_up(image bin)| / |Y_up(tone bin)|`・up 出力長 2N・tone bin = f̂·N・image bin = N−f̂·N・**E-1（gate・base）: −9.542425 ±0.01 dB**・**E-1c（gate・cand）: ≤ −9.442 dB @f̂ ∈ [0.05, 0.35]**・f̂ = 0.40/0.45 は記録のみ。authoritative 再実行で 31/90 の base D1 = −9.54・cand D1 = −90 〜 −104 dB（gate band）を再現。

**Shadow Reference**: §2.9（T-1 + R17-3/R17-4）。骨格のみでは gate 実行不能（R10-6/R11-9/R14-7 継承）。

**D2 / D3**: v2.2/v2.3 のまま（D2 gate [0.005,0.40] base==cand ≤0.01 dB・0.45 記録のみ・D3 worst spur −98〜−101 dB）。

### 2.8 B-1-P0 GATE（判定表・v3.0 §2.8 継承 + **R17-1/R17-3/R17-5 反映**）

| ID | 判定対象 | 責務（R12-3） | 条件 | v3.1 裏付け |
|----|----------|---------------|------|-------------|
| G-0 | 契約 | — | DESIGN-CONTRACT-A（E1〜E5）+ **T-1・T-2・T-3 + R17-1/R17-2** 明示承認 | 本版で P0-I/commit 境界が完全確定 → G-0 承認だけで着手可 |
| **BUILD-ID** | binary identity | run-level | **全測定 log の先頭に `[BUILD]` 6 要素ブロック**: (a) source snapshot identity (b) git HEAD SHA (c) working-tree state (d) `production_flag`（Phase 0: `undefined/off`） (e) `shadow_candidate`（Phase 0: `0/1`） (f) build configuration。**欠落 = 測定無効** | R14-2 + R15-1 + R16-3 継承 |
| G-BL | Baseline sanity | 全経路 | round-trip DC = 0.75^N ±1e-6 のみ gate。up/down 分割は診断ログ | authoritative 再実行（本セッション） |
| REF-FIDELITY | Shadow==production | shadow fidelity | **bitwise（tolerance 0）**。実行前提: **同一 BUILD-ID = 同一 compiler configuration / 同一 build option / 同一 ISA path で、production と Shadow を同一 test executable・同一翻訳単位に compile し同時実行**。比較経路は **`scalar ↔ scalar` / `SIMD ↔ corresponding SIMD`** に固定。**bitwise mismatch は直ちに production defect と断定しない**: triage（§2.9 (6)）で原因を (1) state/history (2) coefficient (3) operation-order (4) ISA/compiler (5) actual production defect に分類・記録するまで次の gate に進まない。**注: production 正しさの gate ではない**（R12-8） | **R17-3 で補強** |
| P0-A | Candidate DC | 全経路 | round-trip 1.0 ±1e-6（必要条件） | authoritative 再実行 |
| P0-B | 低域 | 全経路 | 50 Hz / 1 kHz unity ±0.1 dB・4 経路 | authoritative 再実行（−2.4988/−0.0000 dB） |
| **P0-C** | passband ripple | 全経路 | **[0.005,0.30] max−min ≤0.05 dB（dense）**・周波数軸は **`Fs_in = baseRate`** で正規化（R14-5） | authoritative 再実行（ripple[0.01,0.3] 0.001 dB） |
| P0-C' | differential | 全経路 | (4/3)^N ±0.05 dB（IIR3/LP3 f≤0.45・S1 f≤0.30）— attribution criterion | authoritative 再実行（dev +0.0000 dB） |
| **P0-D** | decimator alias rejection | decimator | stopband: alias ≤ −(A−10) dB @[t_end, 0.5]（per-design 絶対基準 §2.8.3） | 継承 |
| **P0-E** | interpolator image rejection | interpolator | **E-1: base D1 = −9.5424 ±0.01 dB（gate）** / **E-1c: cand D1 ≤ −9.442 dB @f̂≤0.35（gate）** / E-1b: cand D1 2 条件記録 / E-2: D2 [0.005,0.40] base==cand ≤0.01 dB / E-3: 0.45 記録のみ | R13-3 + authoritative 再実行 |
| P0-F | latency | 全経路 | **D = Σ_s (taps[s]−1) × (baseRate / stageRate(s))**・stageRate(s) = baseRate × 2^(s+1)。**R17-5: この式は production `AudioEngine.Processing.Latency.cpp:10-48`（`estimateOversamplingLatencySamplesImpl`・tracked・CMake:1185）の実装式と完全同型**（static_assert :6-8 が対称 linear-phase を要求）。**gate は「Latency.cpp 実装式の理論値 == 実測 peak」**。作業例（IIR3）: 510/2 + 126/4 + 30/8 = **290.25**（LP3: 582.25・511単段: 255.00・S1: 15.00・SoftClip 局所 OS 定数 :120-122 も 15）。peak ∈ [floor(D), floor(D)+1]。base==cand も gate | R13-1 三重裏付け + **R17-5 強化** |
| P0-G | float/double | 2 ペア | maxAbsErr ≤5e-7 / RMS ≤5e-8 + max 相対誤差（\|ref\|>1e-6 のみ）record（R12-12） | 継承 |
| P0-H | block/reset | 全経路 | partition bitwise（4 種 + R17-4 (E) 拡張 partition 集合）・reset contract（atomic 3 / atomic 1）+ candidate マクロ 2 系統一致性 assert・**warm-start 規約**（初回ブロック以降の分割比較はウォームアップ完了後） | 継承 + R17-4 |
| **P0-I** | SoftClip 安全性（**R17-1 で完全確定**） | SoftClip 2 経路 | **下記 R17-1 契約（§2.8 直下）** | **R17-1** |

**全 PASS → Phase 1 eligibility 候補（R15-4 条件D 適用）**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

#### 2.8.1 passband ripple 基線（v3.0 §2.8.1 継承）

| config | 帯域 [0.005,0.30] | base | cand |
|--------|------------------|------|------|
| S1 (31/90) N=1 | gate 帯 | 0.001 | 0.001 |
| IIR3 (511/127/31) | 〃 | 0.034 | 0.025 |
| LP3 (1023/255/63) | 〃 | 0.016 | 0.012 |
| 511/140 single | 〃 | 0.000 | 0.000 |

[0.005,0.45] の参考値（S1 5.242/3.607・IIR3 0.111/0.084・LP3 0.053/0.039）は authoritative 再実行で一致。

#### 2.8.2 D1 手法不変性（v3.0 §2.8.2 継承）

base: 81 測定すべて [−9.546, −9.541]（gate 可）／cand: [−72.11, −164.50]（92.4 dB の手法依存 → 値での gate 不可・**E-1c の絶対形のみが window-independent な no-regression 契約**）。

#### 2.8.3 FIR 絶対基準（P0-D floor・継承）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|---------------|----------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

### 2.9 Shadow 実装契約（v3.0 §2.9 継承 + **R17-3/R17-4 で拡張**）

`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（102 行・HEAD 未登録・untracked・本セッションで 102 行全行精読）の現状:
- 実装済み: `expectedDcRoundTrip(int stages, bool candidate)`、`computeD1Bins`、`PolyphaseGainShadow::reset()`（フラグのみ）、`prepare()`（stageCount 保存のみ）、`centerPhaseGain()`（マクロ切替）。
- **未実装**: (a) `prepareStageModel(int taps, double attenuationDb, int stageInputMax)` 本体（R16-1 の 10 手順独立実装）、(b) `processUp` 本体、(c) 履歴 bitwise 同期、(d) `processDown` 本体、(e) denorm/isBadSample 位置同期 + reset() 契約（atomic 3）の模倣。

**★ Shadow `prepareStageModel` の 10 手順数学的契約（R16-1 継承・本セッションで production 全行照合済み）**:

```text
(1)  taps = jmax(3, taps | 1)                  // :291
(2)  centerTap = (taps - 1) / 2                // :292
(3)  centerParity = centerTap & 1 / convParity = 1 - centerParity   // :293-294
(4)  β = (attenuationDb > 50.0) ? 0.1102·(attenuationDb−8.7)
              : ((attenuationDb >= 21.0) ? 0.5842·pow(attenuationDb−21.0, 0.4)
                                          + 0.07886·(attenuationDb−21.0) : 0.0)   // :301-304
(5)  i0Beta = besselI0(β)（private 実装 :144-157 と同一級数和を独立実装）・M = centerTap
(6)  for n in [0, taps):
       t = n − M
       sinc = (n == M) ? 0.5 : sin(pi·0.5·t)/(pi·t)     // juce::MathConstants<double>::pi 経由
       frac = (n − M) / M
       window = besselI0(β·sqrt(jmax(0, 1−frac²))) / i0Beta
       rawCoeffs[n] = sinc · window
(7)  if (n != centerTap && ((n & 1) == centerParity)) rawCoeffs[n] = 0.0   // halfband zeroing
(8)  sum = Σ rawCoeffs;  if (|sum| > 1e-20) rawCoeffs[i] *= (1.0/sum)     // sum → 1.0
(9)  rawCoeffs[centerTap] = 0.5
     nonCenterSum = Σ_{i≠center} rawCoeffs[i]
     if (|nonCenterSum| > 1e-20) rawCoeffs[i] *= (0.5/nonCenterSum) (i≠center)
     rawCoeffs[centerTap] = 0.5                                          // 再固定
(10) convCount = (taps − convParity + 1)/2
     convCoeffs[r] = rawCoeffs[convParity + 2r]（k ≥ taps は 0.0）
     convCoeffsReversed[convCount−1−r] = convCoeffs[r]
     centerCoeff = rawCoeffs[centerTap]
     centerDelayInput = (centerTap − centerParity)/2
     historyUpKeep = jmax(convCount − 1, centerDelayInput)
     historyDownKeep = jmax(centerTap, convParity + ((convCount − 1)·2) + 6)   // +6: loadStride2 保険
     upHistorySize = historyUpKeep + maxInput + 16 / downHistorySize = historyDownKeep + maxOutput + 16
```

**Shadow は production を直接呼ばない**。上記 10 手順を別コードで独立実装し、同じ数学的契約を満たすことで bitwise 一致を達成する（R15-3 継承）。

**★ R17-3: REF-FIDELITY 実行の前提条件（本版で新設）**:

```text
(a) 同一 BUILD-ID 契約: production と Shadow は同一 test executable・同一翻訳単位にコンパイル
    する（同一 /fp: flags・同一 FMA contraction 設定・同一 ISA path）。bitwise 一致は
    「同一 compiler configuration / 同一 ISA path」でのみ意味を持つ。
(b) 比較経路固定: scalar ↔ scalar / SIMD ↔ corresponding SIMD（R16-1 継承）
(c) mismatch triage — bitwise 不一致を検出した場合、直ちに production defect と断定しない。
    次の分類を順に確定・記録するまで Phase 0 の残 gate を進めない:
      (1) state/history mismatch   → Shadow 側履歴・reset 契約の実装不備 → Shadow 修正
      (2) coefficient mismatch     → 10 手順の演算順序・係数生成差 → Shadow 修正
      (3) operation-order mismatch → 加算/乗算順序・FMA contraction 差 → Shadow 修正
      (4) ISA/compiler difference  → BUILD-ID (f) 不一致 → 同一 build での再現
      (5) actual production defect → 上記除外後の残差 → 記録のみ（Phase 0 では production
                                     変更禁止・Phase 1 の判断材料）
```

**★ R17-4: REF-FIDELITY 追加検証 5 種（§2.7 0-1b の③の前段に実施）**:

```text
(A) coefficient-only test（全 6 design・bitwise）
    比較対象 = Stage 構造体の実在フィールド全項目:
      taps / centerTap / centerParity / convParity / convCount /
      convCoeffs[] 全要素 / convCoeffsReversed[] 全要素 / centerCoeff /
      centerDelayInput / historyUpKeep / historyDownKeep /
      upHistorySize / downHistorySize
    （production は rawCoeffs を保持しない → rawCoeffs の正しさは派生配列が保証。
     Shadow 側 rawCoeffs は prepareStageModel 内部で閉じる）

(B) zero-input test: input = 0 → reset() → processUp → processDown 全出力 0
    （down の silence fast path :583-613 を含む）

(C) constant-input test: input = 1 → base round-trip DC = 0.75^N ±1e-6 /
    candidate = 1.0 ±1e-6（N = 1/2/3・全 preset）

(D) impulse test — ★ warm-start 規約（R17-7 で数値確定）:
    impulse の前に ≥ 16·Σtaps のゼロウォームアップを処理してから δ を入力する。
    cold-start（fresh instance の t=0 impulse）は定常応答と一致しない
    （本セッション実測: cold-start Σh_rt = 0.421369 / warm-start = 0.421875 厳密）。
    単段（S1 31/90）: h_rt == 2·(convCoeffs ⋆ convCoeffs) + c·δ[centerTap]
      c: base 0.25 / cand 0.5（Octave/NumPy で max diff 0.0 確認済み）
    多段（IIR3/LP3）: Σ h_rt == 0.75^N（base）/ 1.0（cand）±1e-9・
                      peak ∈ [floor(D), floor(D)+1]（D = §2.8 P0-F 式）

(E) random block partition（P0-H 拡張）:
    partition 集合 P = {1, 2, 3, 5, 7, 11, 15, 31, 63, 127, 256} ∪ {512, 1024} ∪ mixed
    （各サイズ ≤ maxInputBlockSize・down 経路は偶数ブロックのみ）
    同一連続 stream を one-shot と partitioned で処理 → 出力 bitwise 一致
    + reset contract（atomic 3 / atomic 1 の差分検証）
```

**浮動小数点演算順序の保証（R16-1 継承）**: `juce::MathConstants<double>::pi` を Shadow でも使用・`besselI0` は CIO.cpp:144-157 の級数和を独立実装・加算/乗算順序を production と一致させる・JUCE 関数も同一経路。**さらに R17-3: 同一翻訳単位 compile を必須化**（`/fp:` 設定・FMA contraction・SIMD reduction order の差異を排除する最強の手段）。

**比較対象（5 項目契約・R14-7 継承 + R16-1 + R17-4）**: (1) 係数生成 bitwise（10 手順・演算順序ごと複製）(2) 履歴・分岐順序一致（keep・silence fast path :583-613・境界 guard :620-649・denorm クリア :558-559・isBadSample 位置）(3) 3 対象 bitwise 比較（係数配列 / シード固定疑似乱数 3 ブロック × 2 preset × ratio {2,4,8} の up/down 出力 / reset 前後状態差分）+ **R17-4 の 5 種** (4) candidate マクロ 2 系統一致性 assert（`PolyphaseGainFidelityTests.cpp`・新設・add_executable 40 番目）(5) shadow cand は予測にすぎない（最終判定は flag ON ビルドの実測・REF-FIDELITY は production 正しさの gate ではない・R12-8）。

### 2.10 Phase 0 適用要件（R17-2 で commit 境界を再定義）

production `src/` modified=0 / staged=0 / measurement のみ test-only / **calibration 禁止 / 0.75^N 補償の追加禁止（R12-5 + R13-4: 本セッション再検索で補償 0 件を再確認）** / evidence/ は C0 適用後の実行形態で。**BUILD-ID 行は Phase 0 から必須・6 要素**。**★ Phase 0 で CMakeLists.txt に `CONVOPEQ_CORRECT_POLYPHASE_GAIN` を追加しない（R15-2）**。

**★ commit 境界（R17-2 — v3.0 §2.10 の「commit 禁止」を限定化）**:

```text
Phase 0 = [C0（ユーザー承認後・独立 commit）] → [C1（Shadow 実装・test-only commit）]
        → [0-1 〜 0-6 characterization（commit 全凍結・frozen window）]
        → [Phase 0 freeze（全 P0 gate 判定確定）] → [C2（証跡 commit）]

production src/: 全 Phase 0 期間を通じて modified=0 / staged=0（不変条件 §1.3）
測定結果 commit: C2 のみ（測定中は禁止・frozen window）
Shadow/test commit: C1 で 1 回（凍結窓内は禁止）
production patch: Phase 1 以降（Step 2/2.5 ユーザー GO 後）
```

---

## 3. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4・v3.0 継承）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（R11-6 + R13-5 + R14-3 継承）

**HEAD（8f127bfe）基準**: 誤順 `BassBuzzMeasurement.cpp:1370` `configureProbeFlatEQ(e)` → `:1374` `setAutoGainStagingEnabled(false)`（実測 `staging=0 eqAGC=1`）／正順 `:1609` → `:1614`。
**working tree 基準**: 誤順 `:1799` → `:1803`／正順 `:1833` → `:1834`（+634/−2 の tests 専用差分による乖離・R14-3）。

- **修正案 A（Phase 0 freeze 後 GO）**: HEAD 基準で `:1370` と `:1374` の入替（コメント更新）。パッチは `remediation_v22_prep_patches_20260922.md §2`（未適用）
- 検証: `--buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（[EQ_DIRECT] ratio=1.000000 が独立証拠）
- 着手順: B-1 Phase 0 → Phase 0 result freeze → F-2 3 行 swap → regression

### 3.2 F-3 / F-4（記録のみ・継承）

F-3（EQ dry/wet 混合・`EQProcessor.Processing.cpp:982-996`）は別 work item。F-4（`ir`=dry baseline / `irwet`=wet 対照）は R5-8 継承。**B-1 Phase 0 に混在させない（HOLD）**。

### 3.3 測定 entry 派遣の罠（継承）

`--buzz*` 引数があると `PPIT:1123` が測定 entry へ派遣し `:1223` の無条件 attribution に到達しない。attribution 再現は引数なし実行。

---

## 4. 評価点マトリクス（v3.0 §4 継承）

構成 = 8（RT float/double × {IIR3 511/127/31, LP3 1023/255/63, 511単段 140dB} + SoftClip float/double × {31/90}）。

| Gate \ 構成 | RT-f IIR3 | RT-f LP3 | RT-f 511 | RT-d IIR3 | RT-d LP3 | RT-d 511 | SC-f 31/90 | SC-d 31/90 |
|-------------|-----------|----------|----------|-----------|----------|----------|------------|------------|
| BUILD-ID | run-level（全測定 log 1 行・6 要素・Phase 0 から必須） |
| G-BL / P0-A / P0-B / P0-C / P0-C' / P0-H | ✓ ×6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-D（decimation alias） | ✓ ×6 | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-E（interpolation image・E-1/E-1c/E-2） | ✓ ×6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| P0-F（latency） | ✓（290.25） | ✓（582.25） | ✓（255.00） | ✓ | ✓ | ✓ | ✓（15 固定） | ✓ |
| P0-G（float/double ペア） | ← RT ペア 1 点 | ← | ← | ← | ← | ← | ← SC ペア 1 点 | ← |
| P0-I（SoftClip 安全性・**R17-1 完全契約**） | — | — | — | — | — | — | ✓ | ✓ |

**実適用点数 = per-config gate 8 × 8 構成 = 64 + P0-G 2 + P0-I 2 = 68 点**（+BUILD-ID run-level）。歴史的呼称「108」は potential evaluation cells として注記。

---

## 5. 外部文献による裏付け（v3.0 §5 継承 + ASJ 参照リンク追加）

| 論点 | 文献 | 裏付け内容 |
|------|------|------------|
| 案 E（interpolator の ×2 対称化） | MathWorks `dsp.FIRHalfbandInterpolator` 公式文書 | 「halfband 補間フィルタの係数は出力パワー保存のため補間係数 2 でスケールされる」。halfband interpolator 全体の一般的設計規約であり、案 E との整合は Phase 0 + 条件D で確定（R14-4/R15-4） |
| P0-F（群遅延定数性） | AES Convention Paper 8648（Wang-Reiss）・dsprelated/KVR | linear-phase FIR の群遅延一定性。**R17-5: ConvoPeq の production latency 実装（`Latency.cpp:10-48`）が同型式を要求する static_assert で前提（対称 linear-phase）を自ら主張しているため、実装式と理論式の一致が本計画書内で自己完結** |
| Halfband polyphase 構造（center 0.5） | MathWorks polyphase 分解記述・wavewalkerdsp | even polyphase 成分が単一遅延（center tap）の構造。`rawCoeffs[centerTap]=0.5` と整合 |
| **音響工学参考（ユーザー指定・本セッション到達確認済み）** | 日本音響学会（[学会誌・AST誌](https://acoustics.jp/journal/)）・[ASJ 若手フォーラム「役立つリンク集」](https://asj-fresh.acoustics.jp/useful-links)（学協会・会議・論文検索・MATLAB/Python リソースの網羅リンク）・[音響関連機関リンク](https://acoustics.jp/link/institutes/) | 今後の DSP 文献追跡・AES/JASA/応用音響論文検索の入口として維持 |

---

## 6. 推奨する実行順序（v3.1）

```
[Step 0] 現状固定 — 技術側完了（v3.0 Step 0 の全工程 + R16-1〜R16-3）
[Step 0.5] 計画書修正 — ★ 本版（v3.1）で完了
  ├─ R17-1: P0-I gate 判定関数を数学的に完全定義（B1 解消）
  └─ R17-2: C0/C1/凍結窓/C2 の commit マトリクス（B2 解消）
      │
      ▼
[Step 0.0] Phase 0 着手直前の再実測（★ 本セッションで部分先行実施済み R17-7）
  - python output_sourcecode_markdown.py --check → FRESH（★ 本セッションで FRESH 確認済み）
  - git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests' → 出力なし（★ 確認済み）
  - doc/work113/model_polyphase_20260920.py → 再実行（★ 本セッション実施・一致 R17-7）
    → Phase 0 着手直前には「0-0 承認後の維持確認（1 実行）」のみでよい
  - tmp/v26_p0f_gate_calc.py → 維持確認（v2.7 PASS 継承・任意）
        │
        ▼
[Step C0] evidence policy 単独 commit（★ ユーザー承認を必要とする独立 commit・R17-2）
        │
        ▼
[Step C1] Shadow Reference 完全実装 + REF-FIDELITY 基盤の test-only commit
  - PolyphaseGainCandidateRef.h へ §2.9 完全実装（10 手順 + R17-4 の 5 検証）
  - PolyphaseGainFidelityTests.cpp（新設・add_executable 40 番目）
  - src/tools/build_identity_gate.py に 6 要素 [BUILD] ブロック出力
  - commit 前に §1.3 不変条件（production src diff=0・CMake/build.bat 未変更）を確認
        ▼
[Step 1] §2 B-1 Phase 0（production 変更 0）— G-0 + T-1/T-2/T-3 + C0 が前提
  - 0-0 G-0 承認 + snapshot FRESH 再確認 + スクリプト再実行（維持確認）
  - 0-1 Baseline → 0-1b REF-FIDELITY（R17-3 同一 TU 前提 + R17-4 の 5 検証 + triage）→ 0-2 Candidate
  - 0-3 周波数（E-1 base gate + E-1c cand gate + E-1b 記録）→ 0-4 block/reset + macro assert
  - 0-5 SoftClip（R17-1 の完全固定契約・動作点は §2.8 のとおり）
  - 0-6 float/double
  - 測定実行中 = commit 全凍結（frozen window）・BUILD-ID 行必須・6 要素
        ▼
[Step 2] Phase 0 review（ユーザー GO gate・条件D 適用）
        ▼
[Step 2.5] 条件D 最終判断プロセス（R15-4）— Phase 0 PASS 直後
  - (a) 影響評価（latency / group delay / limiter / makeup / rigcheck）
  - (b) 二重補償の再 census（R13-4 再実行）
  - (c) 全関係者レビュー
        ▼
[Step C2] characterization 証跡 commit → O-1 凍結解除（cleanup/分割後 commit）
        ▼
[Step 3] §3 F-2（test-only・案 A・3 行 swap・HEAD 基準 anchor）
[Step 4] §3 F-1（A→B→B'→C・failureReason 主判定・R12-11 継承）
[Step 5] O-* 残 commit（O-18 ジェネレータ修正を含む）
[Step 6] B-1 Phase 1 以降（Step 2 + 2.5 クリア前提）
  - CMake option 適用（Step 1 では行わない）→ flag ON 実測（BUILD-ID 6 要素必須）
  - 4 経路 × 8 構成で P0 再確認 → Phase 3-A: 補償不存在の再確認 → flag default ON
  - Phase 3-B characterization → 3-C calibration → 3-D rigcheck threshold（commit 分離維持）
[Step 7] 別課題: R-2（parseOnOff 前例）→ R-1 → B-3, D-1, D-2 / O-20（isBadSample 統一・non-blocking）
```

---

## 7. 検証計画（v3.0 §7 継承 + v31 スクリプト追加）

### 7.1 検証スクリプト実在表（v3.0 §7.1 継承 + 追加）

| ファイル | 目的 | 状態 |
|----------|------|------|
| `doc/work113/model_polyphase_20260920.py` + `_results.txt` | authoritative モデル | 実在・**本セッション再実行 PASS（R17-7・D2 記録専用 4 行のみ最終桁 ±0.01 dB）** |
| `tmp/v24_ripple_band.py` / `v24_d1_invariance.py` / `v24_latency_probe.py` | dev 全桁再現 / D1 81 測定 / latency peak | 実在 |
| `tmp/cio_literal.py` / `cio_octave.m` / `maxima_check.mac` | 逐語転写 / Octave / Maxima | 実在（v2.5 実績継承） |
| `tmp/v26_p0f_gate_calc.py` + `_results.txt` | R12-1/R12-2/R11-3 | 実在・v2.7 再実行 PASS 継承 |
| **`tmp/v31_p0i_attribution_check.py`** | **R17-1 の基礎恒等式・impulse 構造恒等式・subset/envelope gate・点wise 出力恒等式の事前検証** | **新設・本セッション実行 ALL PASS（42 判定 + 2 記録）** |
| **`tmp/v31_octave_check.m`** | **Octave 11.3.0 による独立クロス検証（係数スカラー逐語実装 + impulse 恒等式）** | **新設・本セッション実行 PASS（max diff 0.0）** |
| `tmp/v31_maxima.mac` | Maxima 5.50.0 記号確認（定数） | **本ビルドの batch モードは評価結果表示が抑制される（echo のみ）— 稼働自体は確認済み。数値クロスは NumPy/Octave で完結** |
| `tmp/v31_model_rerun_20260921.txt` | authoritative 再実行の生出力 | 新規保存 |

### 7.2 単体 / 統合 / 静的解析（6 要素 BUILD-ID ログ例は v3.0 §7.2 のまま継承）

| 項目 | コマンド | 期待 |
|------|----------|------|
| ConvoPeq freshness | `python output_sourcecode_markdown.py --check` | **FRESH**（★ 本セッション再実測済み・Phase 0 直前にも再実行） |
| production diff | `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | 出力なし（★ 本セッション再実測） |
| commit 境界 | `git status --porcelain -- CMakeLists.txt build.bat` | 空（R17-2 不変条件） |
| Latency.cpp 存在 | `git ls-tree -r HEAD --name-only \| grep -i latency` | **11 件（R17-5 訂正後の正しい基準）** |
| B-1 attribution | `build\Release\AudioEngineHarness.exe`（引数なし） | §2.4 全 row |
| F-2 | `--buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS |
| B-1 Phase 0/1 | 同上（flag ON/OFF rebuild） | OFF ≈ 0.75^N / ON ≈ 1.0・**log 先頭に 6 要素 BUILD-ID 行** |

### 7.3 本版のツール census（v3.0 §7.3 継承 + 本セッション実行分）

- **context-mode MCP**: `ctx_batch_execute`（git/diffstat/census 集約）・`ctx_execute`（Python 検証は Bash 直実行へ切替・ctx_execute の python3 ランタイム不在を確認）・`ctx_fetch_and_index`（ASJ リンク）
- **rtk (WSL版)**: `wsl bash -lc '… rg / head …'`（出力圧縮）
- **headroom**: proxy 常時起動構成・手動 compress/retrieve は本監査では生ソース精読中心のため出力無し（proxy 経由トラフィックは自動圧縮対象）
- **serena MCP**: `get_symbols_overview(CustomInputOversampler.cpp)`（17 メソッド同定）
- **semble / semble MCP**: `oversampler_files` の全ヒット 9 ファイル同定（grep と同一局在に収束）
- **cocoindex (ccc.exe) / graphify / tgrep / ast-grep / rg / fd**: `convValue *= 2.0` 1 hit（:557）・`centerValue *= 2.0` 不在・`CONVOPEQ_CORRECT_POLYPHASE_GAIN` CMake/production 0 hit・Latency.cpp census（R17-5）— 前版 census の鮮度を本セッション再確認
- **python（Windows 3.14.7 + NumPy/SciPy）**: authoritative 再実行 + v31 attribution 検証（新規）
- **Octave 11.3.0**: `tmp/v31_octave_check.m`（新規クロス検証）
- **Maxima 5.50.0**: 稼働確認（`--batch` / `--batch-string` で入力 echo はするが評価結果表示が抑制される dirty SBCL ビルド — 数値は NumPy/Octave で完結）
- **firecrawl MCP**: ASJ 参照 URL の到達確認（§5）
- **未使用**: cppcheck/clang-tidy（本版は計画書改正・ソース変更なしのため静的解析対象なし）/ Dr.Memory（環境干渉で実測不能・AGENTS.md 記載どおり）/ Obscura/Crawl4AI/DDGS/Trafilatura/context7/github/MSLearn（該当ページなし）

---

## 8. ロールバック（v3.0 §8 継承）

§1 commit → `git revert` ／ B-1 behavioral → compile-time rollback（flag OFF rebuild）／ B-1 source → `#if` ブロック + option 削除（**Phase 1 で導入した場合のみ**）／ Shadow → test-only revert（§2.9 契約版への revert）／ F-2 → 3 行 swap の再 swap ／ O-18 → revert 後再生成。**P0-I（R17-1）は production 変更を含まないためロールバック対象外（test-only 証跡のみ）**。

## 9. AGENTS.md 修正パッチ（v3.0 §9 継承・変更なし）

v2.5 §15 継承: `AGENTS.md` の「**Maxima は未インストール**」は誤り（`C:\maxima-5.50.0\bin\maxima.bat` で稼働確認済 — 本セッションでも 5.50.0 を確認）。O-10 commit 時に同時適用。

## 10. 監査ログ・参照（v3.0 §10 継承 + 本セッション実測行を追加）

| ファイル / 手段 | 確認 | 結果 |
|----------------|------|------|
| `git diff HEAD --name-only -- 'src/*.cpp' 'src/*.h' ':!src/tests'` | production 差分 | ✓ 出力なし（本セッション再実測） |
| `git diff --stat HEAD -- src/` | tests のみ差分 | ✓ BassBuzzMeasurement +634/−2 / PPIT +8 / check_layout_offsets.py −52（本セッション再実測・R14-3 一致） |
| `python output_sourcecode_markdown.py --check` | baseline | ✓ FRESH（16:16:35・NEWER_SRC_COUNT=0・本セッション再実測） |
| `git ls-tree -r HEAD --name-only \| grep -i latency` | **Latency.cpp 存在** | **✓ 11 件（うち production C++ 1 件・R17-5 で v3.0 の「該当なし」を訂正）** |
| `git log -- src/audioengine/AudioEngine.Processing.Latency.cpp` | tracked 履歴 | ✓ `c8ca439b` / `465fb744` に commit 済み・working tree 未変更 |
| `CMakeLists.txt:1185` | Latency.cpp compile 登録 | ✓ target_sources リストに実在 |
| `rg 'CONVOPEQ_CORRECT_POLYPHASE_GAIN' CMakeLists.txt src` | flag 未適用 | ✓ 0 hit（test-only 診断コメント 1 件のみ） |
| `rg 'convValue \*= 2.0' src/` | 局在 | ✓ CIO.cpp:557 のみ |
| `rg 'centerValue \*= 2.0' src/` | 不在 | ✓ 0 hit（Ref.h 骨格コメント除く） |
| `rg '0.75\|compensat' src/ CMakeLists.txt`（tests 除外） | 二重補償 | ✓ 補償 0 件（0.75 のヒットは Retire/HealthMonitor/UI 色で全て無関係） |
| `src/CustomInputOversampler.cpp` 全 872 行精読 | prepareStage 10 手順・interpolate :557・decimate :658・reset 契約 | ✓ R16-1 記述と一致・**besselI0 実際は :144-157（R17-6 訂正）** |
| `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` 全 102 行精読 | 骨格構造 | ✓ R14-7 記述と一致（processUp/Down・prepareStageModel 未実装） |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:100-224` 精読 | SoftClip 点wise 性・prev 処理 | ✓ prev は退避のみ（:206/:214-220/:223）・P3 コメント「AVX2/スカラー動作一致」確認 |
| `src/dsp/math/FastTanhApprox.h` 全 169 行精読 | clipThreshold・Padé クランプ | ✓ 4.5・scalar 厳密 ±1.0・SIMD クランプ後 Padé max **0.9992656388416031** |
| `doc/work113/model_polyphase_20260920.py` 再実行 | authoritative | ✓ **本セッション PASS（D2 記録専用 ±0.01 dB 除く一致・R17-7）** |
| `tmp/v31_p0i_attribution_check.py` 実行 | **R17-1 基礎恒等式 + gate 事前検証** | ✓ **ALL PASS**（42 判定: impulse Σ/構造恒等式/peak 位置・per-phase bitwise 2×・RMS √1.6・subset/envelope・NaN/Inf・\|out\|≤1） |
| `tmp/v31_octave_check.m` 実行 | Octave 独立クロス | ✓ PASS（base Σh=0.75/cand Σh=1.0・max diff 0.0） |
| ASJ 参考リンク 3 URL | 到達確認 | ✓（ctx_fetch_and_index / firecrawl） |

---

## 11. 判断待ち項目（v3.1）

### 11.1 ユーザー意思決定（v3.0 §11.1 継承 + R17-2 更新）

O-1（**凍結→C2 後 commit**）/ O-2〜O-17 は v2.5 §11.1 推奨を維持 / O-18 commit 推奨 / O-19 解決済み / O-20 将来対応・non-blocking / **G-0（DESIGN-CONTRACT-A + T-1/T-2/T-3 + R17-1 動作点固定 + R17-2 commit マトリクスの明示承認）** / **C0 承認（evidence/ 追跡外化・独立 commit）** / Phase 0 GO / F-2 案 A / F-1 全実施 / Phase 3 commit 分割 / O-6 push（ユーザー手動）。

### 11.2 残存技術判断（確定案提示済み・承認のみ）— 3 件（v3.0 継承 + R17 反映）

| ID | 論点 | v3.1 確定案 | 根拠 |
|----|------|-------------|------|
| **T-1** | REF-FIDELITY gate の実行条件 | §2.9 の 5 項目契約 + **R17-3（同一翻訳単位・triage 5 分類）+ R17-4（追加検証 5 種・warm-start）** | R10-6/R11-9/R12-8/R14-7/R15-3/R16-1/**R17-3/R17-4** |
| **T-2** | cand D1 の gate 可否 | gate する（E-1c 絶対形）。base E-1 と E-1c の 2 本立て。f̂ 0.40/0.45 記録のみ | R12-2 / R13-3 |
| **T-3** | BUILD-ID gate の必須化 | Phase 0 の全測定 log に 6 要素 `[BUILD]` ブロックを必須化（欠落 = 測定無効） | R12-7/R14-2/R15-1/R16-3 |

**★ v3.0 の「P0-I 許容範囲を Phase 0 着手時に固定」という技術判断待ち項目は R17-1 で解消済み**（動作点・判定関数・pass/fail 境界はすべて本計画書で固定・承認対象は G-0 に統合）。

### 11.3 B-1 採用案の前提条件（v3.0 §11.3 継承 + R17 反映）

1. DESIGN-CONTRACT-A + T-1/T-2/T-3 + **R17-1 動作点固定 + R17-2 commit マトリクス** 明示承認（G-0）
2. REF-FIDELITY bitwise（§2.9 完全実装 + R17-3 同一 TU 前提 + R17-4 追加 5 検証 + warm-start 規約）
3. round-trip DC = 1.0 ±1e-6
4. P0-B/C passband 維持（8 構成・Fs_in = baseRate 正規化）
5. E-1 base D1 = −9.5424 ±0.01 dB + E-1c cand D1 ≤ −9.442 dB @f̂≤0.35
6. P0-F latency 不変（**Latency.cpp 実装式 == 実測 peak・R17-5**）
7. P0-I 数値契約（R17-1: I-0 恒等式 → I-a/I-b 無条件 gate → I-c per-phase 帰属 c1〜c4 → 記録群）
8. P0-C' differential 帰属 + P0-G（絶対誤差 gate・相対誤差記録）
9. D1 測定は補正軸 + expected bin 主報告
10. BUILD-ID 必須・6 要素・Phase 0 から適用
11. Phase 0 で CMakeLists.txt に flag を追加しない（R15-2）
12. Shadow Reference は独立 reference model として実装（R15-3 + R16-1 10 手順 + R17-4）
13. Phase 0 結果から直ちに案E採用としない最終ガード R15-4（条件D）

### 11.4 v3.1 最終判定表

| 項目 | 判定 | v3.0 から |
|------|------|-----------|
| **P0-I saturation gate** | **数学的に完全確定（R17-1・B1 解消・基礎恒等式は数値 ALL PASS）** | **完全確定** |
| **Phase 0 commit 境界** | **commit マトリクスで確定（R17-2・B2 解消・矛盾消滅）** | **完全確定** |
| REF-FIDELITY bitwise 契約 | **同一 BUILD-ID + 同一翻訳単位 + mismatch triage 5 分類（R17-3）** | 補強 |
| Shadow 追加検証 | **5 種追加（coefficient/zero/constant/impulse/random-partition）+ warm-start 規約（R17-4）** | 拡張 |
| P0-F | **Latency.cpp 実装式との一致に再定義（R17-5・v3.0 の「存在しない」訂正）** | **訂正 + 強化** |
| besselI0 行範囲 | **:144-157 に訂正（R17-6）** | 訂正 |
| Step 0.0 | **部分先行実施済み（authoritative 一致・v31 検証 ALL PASS・R17-7）** | 先行実施 |
| E-1c 絶対形 | 確定継承（authoritative 再現） | 維持 |
| 案 E の表現 | candidate hypothesis・外部文献整合・条件D 最終ガード | 維持 |
| 二重補償リスク | 不存在を本セッション再検索で再確認 | 維持 |
| O-20 | 将来対応・non-blocking（P0-I は NaN/Inf gate のみで影響なし） | 維持 |
| §1 O-1〜O-20 | GO 候補（O-15 は C1 前提条件） | 維持 |
| §3 F-2 / F-3 / F-4 | GO 候補（Phase 0 freeze 後）/ 記録 / 記録 | 維持 |
| B-1 原因分析 | GO（5 系統 + 本セッション数値再現 + 文献） | 強化 |
| **B-1 Phase 0** | **GO（G-0 + C0 承認 + Step 0.0 維持確認 + 11.3 全項目）** | 強化 |
| B-1 Phase 1+ / 案 E / Production patch / Calibration | **HOLD / 有力仮説 / HOLD / HOLD** | 維持 |
| 技術的未確定 | **3 件（T-1/T-2/T-3・承認のみ）— P0-I 未定義式と commit 境界は消滅** | **2 点解消** |

---

## 12. エビデンス（v3.1）

- **authoritative モデル再実行（本セッション）**: `python doc/work113/model_polyphase_20260920.py` → `tmp/v31_model_rerun_20260921.txt`。係数 6 design（FIRsum 1.0 / center 0.5 / convSum 0.5）・DC round-trip（0.75^N / 1.0）・full-chain |H|・ripple・0.1dB edge・per-design 絶対基準・D1 構造定数 −9.54 は保存結果と**完全一致**。例外: D2（単段 round-trip f̂ vs 0.5−f̂）の −100 dB 以下 4 行で最終桁 ±0.01 dB（np.hanning 窓丸めノイズ・全 gate 非該当の記録専用指標）。
- **v3.1 新規契約の事前数値検証（本セッション・`tmp/v31_p0i_attribution_check.py` ALL PASS）**:
  - impulse（warm-start）: Σh_rt = 0.75^N（base）/ 1.0（cand）を S1/IIR3/LP3 で ±1e-12。単段構造恒等式 `h_rt = 2·(conv⋆conv) + c·δ[centerTap]`（c: 0.25/0.5）を 1e-14 で厳密一致。peak = 15/290/582（D = 15.00/290.25/582.25 と整合）。
  - **cold-start の初样応答は定常応答と一致しない**（cold Σh_rt = 0.421369 vs steady 0.421875）→ impulse/partition 検定は warm-start 規約を必須化（R17-4 (D)）。
  - per-phase 恒等式: candidate up = base up（conv bitwise 一致 / center bitwise 2×）を DC・正弦・PRNG で厳密確認（denorm 帯 |x| < 2e-20 のみ例外）。
  - P0-I 帰属: conv 位相 event 件数 base==candidate（完全一致）・center 位相 base ⊆ candidate・envelope |x_b| ≥ θ_evt/2 の違反 0（E1/E3/E3plain の 3 定義 × 全刺激）。
  - 点wise 出力恒等式: candidate center 出力 == F(2·x_b)（harness 参照実装）bitwise 一致。NaN/Inf 0・\|out\| ≤ 1.0。
  - RMS: DC で √1.6 = 1.264911064 厳密 / 正弦 1.264899673 / PRNG 1.295801534（記録専用）。
- **Octave 11.3.0 独立クロス検証（本セッション・`tmp/v31_octave_check.m`）**: スカラー逐語実装で FIRsum/center/convSum・base/cand impulse 恒等式を再現（max diff 0.0）。**Maxima 5.50.0 は稼働確認済みだが本ビルドの batch 出力表示が抑制されるため、定数の数値確認は NumPy/Octave で完結**。
- **本セッション再実測**: production diff 0・HEAD 8f127bfe・`--check` FRESH・`grep -i latency` 11 件（R17-5）・`CONVOPEQ_CORRECT_POLYPHASE_GAIN` 0 hit・補償 0 件。
- **v2.x 継承のエビデンス**: v2.5 §12.1（5 系統クロスチェック）・v2.7（authoritative 再実行・latency 4 構成・E-1c 30 点判定 margin 36.7 dB）・v2.8/v2.9（git diff 実測系）・v3.0（prepareStage 精読）— R5〜R16 の各節に記載のとおり。

---

## 13. v3.1 で追加した確定事項（R17 一覧）

- **R17-1** P0-I saturation gate の数学的確定（per-phase 帰属 c1〜c4 + subset/envelope θ_evt/2 + 点wise 出力恒等式。動作点を本計画書で完全固定・「Phase 0 着手時に固定」撤回・**B1 解消・NO-GO → 完全 GO**）
- **R17-2** Phase 0 commit 境界の確定（commit マトリクス: C0 ユーザー承認制 / C1 test-only / characterization 中 commit 全凍結 / C2 freeze 後。production src diff=0 + CMake/build.bat 未変更の不変条件。**B2 解消・矛盾消滅**）
- **R17-3** REF-FIDELITY bitwise 契約の厳密化（同一 BUILD-ID = 同一 compiler configuration / 同一 ISA path 前提 + 同一 test executable・同一翻訳単位 compile 必須 + bitwise mismatch の triage 5 分類（state/history・coefficient・operation-order・ISA/compiler・actual defect）を triage 確定まで Phase 0 を進めない契約に追加）
- **R17-4** Shadow 追加検証 5 種（coefficient-only / zero-input / constant-input / impulse（warm-start 規約・単段構造恒等式）/ random block partition）を §2.9 に追加
- **R17-5** R14-6 訂正: `AudioEngine.Processing.Latency.cpp` は**実在**（tracked・CMakeLists.txt:1185・式 :10-48 同型・static_assert :6-8・SoftClip 15 sample 定数 :120-122）。P0-F gate を「Latency.cpp 実装式 == 実測 peak」に再定義
- **R17-6** besselI0 行範囲訂正（:144-157・v3.0 の :144-216 は dotProductAvx2 を含む誤範囲）
- **R17-7** Step 0.0 の部分先行実施（authoritative 再実行一致・`tmp/v31_p0i_attribution_check.py` 42 判定 ALL PASS・`tmp/v31_octave_check.m` PASS・cold-start/warm-start の規約差異を数値確定）

---

*本書は v3.0（`remediation_plan_20260922_v3.0_revised.md`）の再改訂版 v3.1 である。*
*前版本文は参照文書として維持し、**R17-1〜R17-7 が v3.0 と矛盾する箇所では本書を優先する**。*
*R5〜R17 は本書でも有効（上記是正点を除く）。中間エビデンスは `remediation_v22_intermediate_20260921.md`、パッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*
**継承の監査・確定事項**: R16-1〜R16-3 / R15-1〜R15-4 / R14-1〜R14-7 / R13-1〜R13-9 / R12-1〜R12-12 / R11-1〜R11-10 / R10-1〜R10-13 / R9-1〜R9-6 / R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。**
