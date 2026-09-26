# ConvoPeq 残件 改修計画書（2026-09-22 時点・再改訂 v2.6）

- **版**: v2.6（v2.5 を外部レビューに付し、その 6 必須修正を裁定・**R12-1〜R12-12** を追加）
- **前版**: `doc/work113/remediation_plan_20260922_v2.5_revised.md`（R11-1〜R11-10 / E'-1〜E'-5）
- **外部レビュー**: 2026-09-21 受領（v2.5 監査・「P0-F 数式 NG / P0-E candidate image 未 gate / source-baseline 古い」を必須修正 3 点 + 条件付き GO と判定）
- **本改訂監査実施日**: 2026-09-21（ZCode 環境・v2.5 公開後の同一作業ツリー）
- **基準ソース**: HEAD = `8f127bfe`（親 `c4a08171`）。production `src/*.cpp/h`（tests 以外）の未 commit 差分 **0 件を実測再確認**
- **★ source freeze（本版で実施・レビュー必須修正 5 の解決）**: `ConvoPeq.md` を **HEAD 8f127bfe 作業ツリーから 2026-09-21 15:52:48 に再生成**。349 ファイル / 5,361,988 B / `python output_sourcecode_markdown.py --check` → **FRESH（NEWER_SRC_COUNT=0）**。以後の引用はこの snapshot を authoritative とする
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py` + `_results.txt`（本版で全文読了・転記値を実測突合）
- **v2.6 の位置付け**: 外部レビューの指摘を**全件ソース突合で裁定**した結果、**レビューの最重要指摘 P0-F「数式 NG」は誤り（式は正しい）**と確定。一方 P0-E candidate image gate の欠落指摘は正当で、**E-1c（絶対形 no-regression gate）を新設**。残る必須修正（D/E 責務分離・P0-I 数値契約・source freeze・binary identity）は全て本版で確定
- **案 E は有力仮説（candidate hypothesis）**。defect（DC = 0.75^N）は 5 系統独立で再現済み。是の設計判断は G-0 承認まで「確定」と表記しない（R5-4 継承）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v2.2 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** / C++ は **`#if`（`#ifdef` 禁止）**（production 側の規約。test 側の診断出力は §R12-7）
- **パッチ案は文書化済み・未適用**（`remediation_v22_prep_patches_20260922.md`）

---

## 0. 外部レビュー 6 必須修正への裁定（本版の最重要節）

外部レビューは「Phase 0 GO の前に修正すべき必須 6 点」を挙げた。v2.6 は全 6 点をソース実測で裁定した。

| # | レビューの必須修正 | v2.6 裁定 | 根拠（R12 番号） |
|---|------------------|-----------|------------------|
| 1 | **P0-F 数式が本文表と不整合（NG）** | **レビュー側の再計算が誤り（二重除算）→ v2.5 の式は正しい。ただし式を Latency.cpp 同型で再定義し作業例を付記して誤読を構造的に排除** | R12-1 |
| 2 | **P0-E: cand D1 を no-regression gate 化せよ（提案: `D1_cand ≤ D1_base+0.1dB`）** | **指摘は正当 → E-1c を新設。ただし提案の相対形は遷移帯で誤警報を出すため、絶対形（構造定数+0.1 dB = −9.442 dB・f̂≤0.35）を採用** | R12-2 |
| 3 | **P0-D/E の責務分離（D=decimation alias / E=interpolation image）** | **採用 → GATE 表に責務を明記。Fs_in 定義も固定** | R12-3 |
| 4 | **P0-I safety の数値契約（NaN/Inf=FAIL 等を明記）** | **採用 → I-a/I-b/I-c を数値契約として確定** | R12-4 |
| 5 | **最新 ConvoPeq.md 再生成・source freeze** | **本版で実施済み**（2026-09-21 15:52:48・HEAD・FRESH 確認）。付随してジェネレータのインデックスバイナリ混入バグを修正（O-18） | R12-6 |
| 6 | **Phase 2 binary identity（flag ON を machine-readable log へ）** | **採用 → BUILD-ID gate を新設（`[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` を全測定出力に必須化・test-only）** | R12-7 |

レビューのその他の指摘（REF-FIDELITY の性格明記＝R12-8 / 「108」の呼称＝R12-9 / Phase 3 二重補償禁止＝R12-5 / O-1 凍結＝R12-10 / F-1 移行注意＝R12-11 / P0-G 相対誤差＝R12-12）も全て採用し、該当節を更新した。

---

## 1. v2.5 → v2.6 改訂サマリ

v2.5 の GO/HOLD・案 E の位置付け・Phase 0 の 5 構造・R10/R11 確定事項は**原則維持**。本改訂は (a) 外部レビュー 6 必須修正の裁定、(b) E-1c gate の新設（相対形の誤警報を実証のうえ絶対形を採用）、(c) P0-F 式のコード同型再定義、(d) source freeze の実施とジェネレータ修正、(e) 評価点の適用マトリクス化、(f) O-18/O-19 の棚卸し追加、を扱う。

### 新規確定（R12-1 〜 R12-12）

| ID | 区分 | 内容 | v2.6 での確定 |
|----|------|------|----------------|
| **R12-1** | P0-F 数式の裁定 | `Latency.cpp:26-33` を実測: `stageRate = baseSampleRate * (1 << (stage+1))`・`groupDelaySamplesAtStageRate = taps[stage] - 1`（コメント `// up + down`）・`delayBaseSamples = groupDelay * (baseRate/stageRate)`。よって **D = Σ_s (taps[s]−1) × 2^(−(s+1))** はコードと代数的に同型。IIR3: 510/2 + 126/4 + 30/8 = **255 + 31.5 + 3.75 = 290.25** ✓。レビューの「255/2 + 126/4 + 30/8 = 162.75」は **(taps−1)/2 を先に適用した二重除算**（S1 で 7.5 となり実測 peak 15 と矛盾することでも証明される）。`tmp/v26_p0f_gate_calc.py` で 4 構成を独立計算し R11-2 全値（15.00/255.00/290.25/582.25）と peak gate（[15,16]/[255,256]/[290,291]/[582,583] vs 実測 15/255/291/583）を**全 PASS** で再現 | **P0-F の「NG」は取り下げ**。式は本文にコード同型 + 作業例で記載（§3 GATE 表） |
| **R12-2** | E-1c gate 設計 | レビュー提案の相対形 `D1_cand ≤ D1_base(f̂)+0.1 dB` を 42 点モデル grid（6 design × 7 f̂）で判定 → **f̂=0.45・31/90 で誤警報**（base が遷移帯 rolloff で −24.18 dB まで下がり、cand −11.14 dB が FAIL と誤判定。round-trip D2 では base==cand ±0.3 dB で実害なし）。よって**絶対形 `D1_cand ≤ −9.442 dB`（= 20·log10(1/3) + 0.1）を f̂≤0.35 で gate、f̂ 0.40/0.45 は記録のみ**とする。42 点中 gate 違反 **0 件・最小 margin 36.7 dB**。レビューの懸念（cand image が −5 dB に悪化しても検知できない）は絶対形が直接解決 | **E-1c 新設（絶対形）**。T-2 に「E-1c 採用」を追加 |
| **R12-3** | P0-D/E 責務分離 + 周波数軸定義 | **P0-D = decimator alias rejection**（stopband・per-design 絶対基準 §4.2）／**P0-E = interpolator image rejection**（up 単段出力の D1）。周波数軸は**全て base 入力レート正規化（Fs_in ≡ host/base rate）**。D1 の f̂ も Fs_in 正規化（up 出力は 2N 点 FFT で tone bin = f̂·N）。stage-local 正規化は per-design 絶対基準（own-rate cycles/sample）でのみ使用し、表にはその旨を明記 | GATE 表に責務列を追加 |
| **R12-4** | P0-I 数値契約 | I-b safety を数値契約化: **出力に NaN/Inf が 1 つでも → FAIL**（harness 側で検査。production `isBadSample` は `CustomInputOversampler.cpp:24` に存在するが SoftClip 経路 `softClipBlockAVX2` には NaN guard が無いことを実測 → harness 責務）／**baseline に無い飽和イベント（|out| ≥ rail）→ FAIL**／peak・RMS・THD・THD+N・saturation event 数・NaN/Inf 数を**全て記録**（I-c）。案 E は SoftClip 局所 OS の前段振幅を +2.4988 dB 動かすため、nonlinear 動作点の変化を「動いた」でなく数値で判定する | P0-I 行を契約化（§3） |
| **R12-5** | 二重補償リスク調査（レビュー §18） | 「既存 0.75^N engine-fit/compensation が Phase 3 で二重補償になる」懸念に対し、**production src 全体を rg 検索** → `0.75` の hit は 5 件すべて無関係（`AudioEngine.Retire.cpp:26` の wet gain jlimit(0.75,1.50)・`RuntimeHealthMonitor.h:86` の閾値定数・`SpectrumAnalyzerComponent.h:116` のコメント・`EQControlPanel.cpp:493` の alpha・`PsychoacousticDither.h:219` の係数値）。**production に 0.75^N 補償は存在しない** → Phase 3-A 受け入れ基準に「補償不存在の再確認（同一検索）」を明記し、リスクを構造的に封じる | Phase 3-A 条件に追加（§6） |
| **R12-6** | source freeze 実施 | `python output_sourcecode_markdown.py` を実行し ConvoPeq.md を HEAD で再生成（**2026-09-21 15:52:48・349 files・5,361,988 B**）。`--check` → FRESH。**ジェネレータの欠陥を発見・修正**: `IGNORE_EXTS` に `.bin`/`.dat` が無く、`src/.tgrep/` の trigram インデックス（index.bin 5.3MB 等・計約 6.6MB）が snapshot に混入していた → `.bin`/`.dat` 除外 + `SKIP_DIRS`（.tgrep/.aidex/.cocoindex_code/__pycache__/.git）を追加。修正版で再生成し tgrep 混入 0 を確認。**ジェネレータ修正は tracked 変更（+11/−4）として O-18 に計上** | source freeze 完了。以後の引用は本 snapshot 基準 |
| **R12-7** | BUILD-ID gate（レビュー必須修正 6） | flag ON 実測（Phase 1）の binary 取り違え防止として、**全測定出力の先頭行に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` を必須化**。実装は test-only（BassBuzzMeasurement の OS_DIRECT / rigcheck 出力ヘッダ・Phase 1 の harness commit に同梱）。CMake が `add_compile_definitions(...=$<BOOL:...>)` で両ターゲットに常時定義するため test コードは `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN` で 0/1 を確定印刷できる（`#ifdef` 不要・production 規約と矛盾しない）。測定ログの machine-readable 行と CSV メタ列に同じ値を出す。**D-1（build identity gate M1/M2）は本 gate と統合**（O-19 参照） | GATE 表に BUILD-ID 行を追加 |
| **R12-8** | REF-FIDELITY の性格明記（レビュー §4） | GATE 表 REF-FIDELITY 行に「**本 gate は shadow が production を正確に再現していることの gate であり、production が正しいことの gate ではない**（shadow が production のバグを複製しても bitwise 一致は成立する。production 正しさの gate は P0-A〜P0-I と Phase 1 実測が担う）」を追記 | §3 GATE 表 |
| **R12-9** | 評価点の再集計（レビュー §14） | v2.5 §9.2 の「9 gate × 12 構成 = 108」は**表の構成数（8）と整合しない potential 値**だった。v2.6 では構成を **8**（RT float/double × {IIR3, LP3, 511単段} + SoftClip float/double × {31/90}）に確定し、gate 適用マトリクスを提示: per-config gate（A/B/C/C'/D/E/F/H）× 8 = **64 適用点** + P0-G 2 ペア + P0-I 2 経路 + BUILD-ID run-level = **実適用 68 点**。歴史的呼称「108」は potential evaluation cells として注記に残す | §5 マトリクス新設 |
| **R12-10** | O-1 凍結 + evidence commit 分離（レビュー §22/23） | **O-1/O-1b（harness 計装）は Phase 0 characterization 完了まで commit しない（凍結）**。測定の再現経路が未 commit のまま凍結されることの監査性を優先（完了後に cleanup/分割して commit）。evidence/ 追跡外化は**単独 commit C0** とし、Phase 0 の shadow 実装 commit（C1）・characterization 結果 commit（C2）と**混ぜない** | §2 / §7 実行順序に反映 |
| **R12-11** | F-1 移行の追加注意（レビュー §25） | private `checkNoConflictingTransitions`（RuntimePublicationValidator.h:101・本体 cpp:169）を直接呼ぶ既存 test を公開 `validatePublication()` へ置換すると、first-fail-wins（Semantic → Topology → Resources → Transitions）で**別検査が先に落ち、旧 private 検査と等価にならない**ケースが存在する。A→B→B'→C 分類は維持し、**B' Semantic Equivalence Gate で failureReason が旧検査と一致しない case は公開 API で再現不能として「移行せず退役」に分類（無理に移植しない）** | §4.2 に追記 |
| **R12-12** | P0-G 相対誤差の扱い（レビュー §11） | `maxAbsErr ≤ 5e-7 / RMS ≤ 5e-8`（絶対誤差）は維持。絶対誤差は本系（unity DC の線形経路）で意味を持つため主契約とし、**`|ref| > 1e-6` のサンプルに限る max 相対誤差を記録のみ**追加（zero 近傍の相対誤差は発散するため gate にしない） | P0-G 行に追記 |

### 新規棚卸し（O-18 / O-19）

| ID | パス | 実測 | 推奨 |
|----|------|------|------|
| **O-18** | `output_sourcecode_markdown.py` | **+11/−4**（本セッションの R12-6 修正: `.bin`/`.dat` 除外 + `SKIP_DIRS` 追加） | **commit 推奨**（source freeze の品質を支えるツール修正。O-2/O-13 の台帳 commit とは別の独立 commit が望ましい） |
| **O-19** | `src/tools/build_identity_gate.py` / `check_layout_offsets.py` | **解決（2026-09-21 ユーザー確認）**: ユーザーが検証用ツールとして `tools/` へ移動していた。ただし `build_identity_gate.py` は `build.bat:204/206` の fail-closed pre-build gate（`src\tools\` パス参照・ゲート失敗でビルド拒否）に組み込まれているため **`src/tools/` に復元済み**（git checkout・HEAD と一致）。`check_layout_offsets.py` はビルドスクリプトから参照されない手動検証ツールのため **`tools/` 移動のまま確定**（`src/tools/check_layout_offsets.py` は D・`tools/` 側 untracked） | **解決**。D-1（build identity gate）は R12-7 の BUILD-ID gate と統合のうえ `src/tools/build_identity_gate.py` を継続使用 |

**O-7/O-10 の数値更新**: `AGENTS.md` は v2.5 時点の +38/−3 から **+71/−3** に増加（2026-09-21 の Freebuff Desktop 環境節追加を含む）。R11-7 の Maxima 誤記修正は未適用のまま（差分統計は commit 時に再計測）。

**v2.5 §7.1 の訂正**: v2.5 が「v2.5 追加」として参照した `tmp/v25_p0f_calc.py` / `tmp/v25_roundtrip_calc.py` / `tmp/v25_d1_struct.py` は**実在しない**（R11-2/3/4 はセッション内計算でスクリプト保存漏れ）。v2.6 では後継として **`tmp/v26_p0f_gate_calc.py` を実体として作成・実行済み**（`tmp/v26_p0f_gate_calc_results.txt` に結果保存・§6 参照）。計画書が参照する検証スクリプトは実在するもののみを列挙する規約に改める。

---

## 2. §1 未 commit / 未 push 対応（O-1〜O-19・v2.6 時点の実測棚卸し）

### 2.1 実測 diffstat（2026-09-21 15:55 時点・`git diff HEAD --numstat` + `git ls-files --others --exclude-standard`）

| ID | パス | 実測 | v2.5 との差分 | v2.6 推奨 |
|----|------|------|----------------|-----------|
| **O-1** | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` | **+634/−2** | 一致 | **Phase 0 characterization 完了まで凍結（commit しない）→ 完了後に cleanup/分割して commit**（R12-10・v2.5 の「案 A で commit」から変更） |
| **O-1b** | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | **+8/−0** | 一致 | O-1 と同一運命（凍結） |
| **O-2** | 台帳更新 | 独立 commit | — | 独立 |
| **O-3 / O-12** | `ConvoPeq.md` | **+824/−5**（source freeze 再生成を含む） | 再生成済み | **commit しない**（再生成資産・`--check` FRESH を維持） |
| **O-4** | `Testing/Temporary/CTestCostData.txt` | **D**（削除） | 一致 | 現状維持 |
| **O-5** | `.opencode/opencode.json` | untracked | 一致 | 触らない |
| **O-6** | push | ahead 2（`8f127bfe`・`c4a08171` は commit 済） | 一致 | **ユーザー手動** |
| **O-7 / O-10** | `AGENTS.md` | **+71/−3**（Freebuff 節追加を含む） | **更新（v2.5 は +38/−3）** | 環境記録として commit 可。R11-7 の Maxima 誤記修正（§9）を同時に適用 |
| **O-8** | `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` | M（追跡済みバイナリ） | 一致 | commit しない → 将来 `.gitignore` + `git rm --cached` |
| **O-9** | `doc/tool-inventory-2026-09-20.md` ほか tool-inventory 2 件 | untracked | 一致 | 触らない（ユーザー判断） |
| **O-11** | `doc/work113/residual_tasks_20260919.md` | **+71/−1** | 一致 | 台帳として commit 可 |
| **O-13** | `doc/work113/*.md` + model py/results untracked | 一致（本版追加で 16 件以上） | 承認版+台帳+モデルを 1 commit |
| **O-14** | `.mcp.json` | **+33/−23** | 一致 | commit（環境記録） |
| **O-15** | `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` | untracked（102 行） | 一致（実態は R11-9/R12-8） | O-13 に併同可。**commit 前に §3.9 契約を反映した版へ** |
| **O-16** | `evidence/epoch_reclaim_audit.json` | **M・Bin** | 一致 | **C0 で追跡外化**（R11-8 確定・§2.3） |
| **O-17** | `doc/WorkBuddy_AI_Desktop_Toolchain.md` | untracked | 一致 | O-9 と同一方針 |
| **O-18** | `output_sourcecode_markdown.py` | **+11/−4** | **新規** | **commit 推奨**（R12-6・独立 commit） |
| **O-19** | `src/tools/build_identity_gate.py` / `check_layout_offsets.py` | **D（HEAD からの削除・本日発生・出所不明）** | **新規** | **ユーザー確認のうえ復元推奨**。意図的なら D-1 スコープ再定義 |

### 2.2 O-16 と Phase 0 の衝突（R11-8 確定方針の維持）

 Phase 0 測定が tracked `evidence/` を書き換える構造的欠陥は未解決（追跡外化は未適用）。手順は §2.3 に固定済み。

### 2.3 evidence/ 追跡外化手順（R11-8 + R12-10 の commit 分離）

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

## 3. §2 B-1: CustomInputOversampler up/down round-trip 利得欠陥

### 3.1 確定している事実（v2.5 §2.1 継承・本版で行番号再確認済）

```
主因        interpolateStage の center 位相に ×2 が無い（polyphase 利得規約の不対称）
欠陥局在    src/CustomInputOversampler.cpp :557（convValue *= 2.0 のみ）/:563-564（center に ×2 なし）
            decimateStage :658（centerCoeff * centerSample + conv tap 累加・×2 なし＝DC 利得 1.0 で正規）
round-trip  DC: base 0.75^N / cand 1.0（R11-3 + 本版 §6 スクリプトで再確認）
段ごとの和  conv 分岐: Σcoeffs 0.5 → ×2 → DC 1.0（正規）／center 分岐: 0.5 → ×2 なし → DC 0.5（不対称）
利益        (4/3)^N = +2.498775 / +4.997549 / +7.496324 dB（1/2/3 段）
画像        up 出力の像/トーン = 1/3 = −9.542425 dB（構造定数・R11-4 確定）
係数        FIRsum 1.0 / center 0.5 / convSum 0.5（全 6 design・model_results.txt :2-7 と一致）
契約不整合  isLinearPhaseFIR / isSymmetricUpDown（h:21-22）・Latency.cpp:6-8 static_assert と矛盾
production  変更 0 / commit 0（本版で git diff 実測・tests 以外 0 件を再確認）
```

**peak / mean の帰属不能性（R10-3・E-8 確定表現の維持）**: up 出力は位相交互 `1.0, 0.5, …`。peak 系で「up 1.0/down 0.75」、mean 系で「up 0.75/down 1.0」— どちらも同じ階段信号の別記述で因果の根拠にならない。因果は分岐係数和の不対称（1.0 vs 0.5）で確定。

### 3.2 根本原因の行番号（v2.6 再実測・1 文字照合）

| 箇所 | 内容 | 実測 |
|------|------|------|
| `prepareStage` :287-390 | `taps=jmax(3,taps|1)` :291 / `centerTap=(taps-1)/2` :292 / parity :293-294 / Kaiser β 三分岐 :301-304 / sinc（center で 0.5）:308-317 / halfband ゼロ化 :319-323 / `sum→1.0` :325-333 / `rawCoeffs[centerTap]=0.5` :335（:348 再設定）/ 非 center を 0.5 へ :336-347 / `convCount` :350 / convCoeffs 構築 :360-365 / `centerCoeff` :367 / `centerDelayInput` :368 / keep :369/:372 | ✓ |
| `interpolateStage` :492-568 | **:557 `convValue *= 2.0`（conv のみ ×2）** / :558-559 denorm / **:563-564 出力（center に ×2 なし）** | ✓ defect 確認 |
| `decimateStage` :570-723 | silence fast path :583-613 / :658 `acc = centerCoeff * centerSample` + conv 累加（×2 なし＝DC 1.0 正規） | ✓ |
| `processUp` :725- / `processDown` :785- | hardFallback / corruption clear | ✓ |
| `reset()` :452-467（atomic 3）/ `clearAllStages()` :469-484（atomic 1） | Phase 0 は `reset()` 経由必須（R4-3） | ✓ |
| `prepareSingleStage` :392-450 | SoftClip 局所 OS（Lifecycle :188/:261 で `(31, 90.0, internalMaxBlock)`） | ✓ |
| `dotProductAvx2` :159-216 | conv 専用 SIMD・center 位相は非依存 | ✓ |

**×2 不在の 5 手段独立確認**（rtk rg / tgrep / ast-grep / AiDex / serena）は v2.5 §2.2 の記録どおり。

### 3.3 5 系統独立再現 + 数値計算裏付け（R10-1 + R11-3 + 本版再確認）

production バイナリ / authoritative モデル / NumPy 逐語転写 / Octave 11.3.0 / Maxima 5.50.0 — 全系統が base 0.75^N・cand 1.0・Δ+2.498775 dB(N=1)・image/tone 1/3 に一致（v2.5 §2.3 表を継承）。本版は `tmp/v26_p0f_gate_calc.py` §3 で N=1..8 の DC と (4/3)^N dB を再計算し全桁一致を確認。

### 3.4 ライブ帰因テーブル（R10-2 継承・引数なし実行の制約）

`[OS_DIRECT]/[EQ_DIRECT]/[OF_DIRECT]` の全 row は v2.5 §2.4 を継承。**唯一の桁違い損失は Oversampler（0.75^N）**・EQ は ratio=1.000000・OutputFilter は −0.110 dB 一定。`rt = up × dn` 成立・preset 非依存・**P0 gate は rt のみ**。再現には引数なし実行が必須（`--buzz*` 引数があると `PublishPipelineIntegrationTests.cpp:1123` の派遣で :1223 に到達しない）。

### 3.5 修正案と DESIGN-CONTRACT-A（変更なし）

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

### 3.6 CMake / flag（R10-5 + E'-4 確定・行番号本版再確認）

- `option(CONVOPEQ_CORRECT_POLYPHASE_GAIN "Correct polyphase gain convention (B-1 案E)" OFF)` を option 群（:40 近傍）に追加
- `add_compile_definitions(CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${...}>)` を `add_subdirectory(JUCE)`（:1043）後・`juce_add_gui_app(ConvoPeq`（:1062）前に配置
- 到達確認: CIO.cpp を compile するのは ConvoPeq（`target_sources` :1282・リスト内 :1196）と AudioEngineHarness（`add_executable` :1899・総数 39）の 2 論理ターゲット。両者とも :1043 以降に生成されるため 1 か所で到達
- C++ は **`#if`**（`#ifdef` 禁止）。既定 OFF / runtime flag 不採用 / **現状未適用**
- **R12-7**: 測定出力の `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` 行を Phase 1 の harness 変更で必須化（test-only・`add_compile_definitions` により macro は常に 0/1 で定義済み）

### 3.7 Phase 0 characterization（v2.4 改定 + E-1c 追加）

```
0-0  G-0: DESIGN-CONTRACT-A（E1〜E5）+ T-1/T-2/T-3 をユーザーが明示承認
0-1  Baseline record-only: round-trip DC = 0.75^N ±1e-6（invariant のみ gate・up/dn 分割は診断ログ）
0-1b REF-FIDELITY: Shadow == production（bitwise・reset() 経由・§3.9 の実装契約）
0-2  Shadow Candidate E: round-trip DC = 1.0 ± 1e-6
0-3  周波数（D1/D2/D3）— D1 は base 側 gate + E-1c cand gate（§3.8）・D2 gate [0.005,0.40]
0-4  block/reset bitwise（partition 4 種 + reset contract）+ candidate マクロ 2 系統一致性 assert
0-5  SoftClip local OS（float + double・R12-4 の数値契約）
0-6  float/double equivalence（maxAbsErr ≤5e-7 / RMS ≤5e-8 / 相対誤差記録のみ）
```

**D1 定義（R10-4 + E'-5 継承 + R12-2）**:
- 軸: `D1 = |Y_up(image bin)| / |Y_up(tone bin)|`、up 出力長 2N、tone bin = f̂·N、image bin = N−f̂·N（補正軸）
- **E-1（gate・base）**: `D1_base = −9.542425 ± 0.01 dB`（構造定数・手法不変 81 測定で max 差 0.0036 dB）
- **E-1c（gate・cand・新設）**: `D1_cand ≤ −9.442 dB`（= −9.542425 + 0.1）を **f̂ ∈ [0.05, 0.35] で gate**。f̂ = 0.40 / 0.45 は**遷移帯のため記録のみ**（両側が rolloff し image/tone 比が品質指標として不成立。相対形 `cand ≤ base+0.1` は f̂=0.45・31/90 で誤警報 — R12-2 実証）
- E-1b（記録）: cand D1 を矩形窓・Hann+trim8 の 2 条件で併記（手法感度 −72〜−164 dB の可視化）
- tone/image bin は expected 主報告・argmax ±8 bin は検証ログ（R9-2 継承）
- final alias rejection は P0-D が担当（R12-3 の責務分離）

**Shadow Reference**: §3.9（T-1）。骨格のみでは gate 実行不能（R10-6/R11-9 確定の維持）。

**D2 / D3**: v2.2/v2.3 のまま。D2 gate [0.005, 0.40] base==cand ≤0.01 dB・0.45 記録のみ。D3 worst spur −98〜−101 dB（窓 sidelobe）・base==cand 差 0.00 dB。

### 3.8 B-1-P0 GATE（判定表・v2.6 改定版）

| ID | 判定対象 | 責務（R12-3） | 条件 | v2.5 からの変更 | 合否 |
|----|----------|---------------|------|----------------|------|
| G-0 | 契約 | — | DESIGN-CONTRACT-A（E1〜E5）+ **T-1・T-2・T-3** 明示承認 | T-3 追加 | □ |
| BUILD-ID | binary identity | run-level | **全測定 log の先頭に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1`**。欠落 = 測定無効（R12-7・Phase 1 から必須・Phase 0 は record） | **新設** | □ |
| G-BL | Baseline sanity | 全経路 | round-trip DC = 0.75^N ±1e-6 のみ gate。up/down 分割は診断ログ | 維持 | □ |
| REF-FIDELITY | Shadow==production | shadow fidelity | bitwise（tolerance 0）・§3.9 の 3 対象・`reset()` 経由。**注: production 正しさの gate ではない**（R12-8） | 性格明記 | □ |
| P0-A | Candidate DC | 全経路 | round-trip 1.0 ±1e-6（必要条件） | — | □ |
| P0-B | 低域 | 全経路 | 50 Hz / 1 kHz unity ±0.1 dB・4 経路 | — | □ |
| P0-C | passband ripple | 全経路 | **[0.005,0.30] max−min ≤0.05 dB（dense）**・周波数軸は **Fs_in（base レート）正規化**（R12-3） | 軸定義固定 | □ |
| P0-C' | differential | 全経路 | (4/3)^N ±0.05 dB（IIR3/LP3 f≤0.45・S1 f≤0.30）— attribution criterion | — | □ |
| **P0-D** | **decimator alias rejection** | decimator | stopband: alias ≤ −(A−10) dB @[t_end, 0.5]（per-design 絶対基準 §3.8.3） | **責務を明記** | □ |
| **P0-E** | **interpolator image rejection** | interpolator | **E-1: base D1 = −9.5424 ±0.01 dB（gate）** / **E-1c: cand D1 ≤ −9.442 dB @f̂≤0.35（gate・新設 R12-2）** / E-1b: cand D1 2 条件記録 / E-2: D2 [0.005,0.40] base==cand ≤0.01 dB / E-3: 0.45 記録のみ | **E-1c 追加** | □ |
| P0-F | latency | 全経路 | peak ∈ [floor(D), floor(D)+1]。**D = Σ_s (taps[s]−1) × (baseRate / stageRate(s))・stageRate(s) = baseRate × 2^(s+1)**（`Latency.cpp:29-31` 同型）。**作業例（IIR3）: 510/2 + 126/4 + 30/8 = 255 + 31.5 + 3.75 = 290.25**。基線: S1 15（dev 0）・511単段 255（dev 0）・IIR3 291（dev +0.75）・LP3 583（dev +0.75）・SoftClip 単段 15（`kSoftClipLatencyBaseRateSamples`・Latency.cpp:120）。base==cand も gate | **式をコード同型で再定義（R12-1・数式の意味論は v2.5 から不変）** | □ |
| P0-G | float/double | 2 ペア | maxAbsErr ≤5e-7 / RMS ≤5e-8 + **max 相対誤差（\|ref\|>1e-6 のみ）record**（R12-12） | 追記 | □ |
| P0-H | block/reset | 全経路 | partition bitwise・reset contract（atomic 3 / atomic 1）+ candidate マクロ 2 系統一致性 assert | — | □ |
| P0-I | SoftClip 安全性 | SoftClip 2 経路 | **I-a: DC = 1.0 ±1e-6** / **I-b（数値契約・R12-4）: 出力 NaN/Inf 1 件でも FAIL・baseline に無い飽和イベント FAIL** / **I-c（記録）: peak・RMS・THD・THD+N・saturation event 数・NaN/Inf 数・動作点**（`prepareSingleStage(31,90.0)`・Lifecycle :188/:261） | **契約化** | □ |

**全 PASS → Phase 1 eligibility**。FAIL → ①案 D 統合 → ②tap 再設計 → ③TruePeakDetector 型（参照・第三順位）。

#### 3.8.1 passband ripple 基線（v2.4 実測・継承）

| config | 帯域 [0.005,0.30] | base | cand |
|--------|------------------|------|------|
| S1 (31/90) N=1 | gate 帯 | 0.001 | 0.001 |
| IIR3 (511/127/31) | 〃 | 0.034 | 0.025 |
| LP3 (1023/255/63) | 〃 | 0.016 | 0.012 |
| 511/140 single | 〃 | 0.000 | 0.000 |

[0.005,0.45] の参考値（S1 5.242/3.607・IIR3 0.111/0.084・LP3 0.053/0.039）は v2.5 §2.8.1 を継承。**重要**: [0.005,0.30] と [0.01,0.30] は全 config で 3 桁一致。

#### 3.8.2 D1 手法不変性（v2.4 実測・継承）

base: 81 測定すべて [−9.546, −9.541]（gate 可）／cand: [−72.11, −164.50]（92.4 dB の手法依存 → 値での gate 不可・**E-1c の絶対形のみが window-independent な no-regression 契約**）。R9-5 の 0.94 dB 差は手法感度帯内のノイズ（R10-4 解消済み）。

#### 3.8.3 FIR 絶対基準（P0-D floor・継承）

| design | −0.1 dB edge | transition_end | floor（A−10） |
|--------|--------------|----------------|---------------|
| 511/140 | 0.2448 | 0.2590 | −130 dB |
| 127/110 | 0.2317 | 0.2782 | −100 dB |
| 31/90 | 0.1823 | 0.3452 | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | −150 dB |
| 255/140 | 0.2396 | 0.2681 | −130 dB |
| 63/120 | 0.2109 | 0.3130 | −110 dB |

### 3.9 Shadow 実装契約（T-1 確定案の維持・R10-6 + R11-9 + R12-8）

`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（102 行）の実態: `prepare()` は stageCount 保存のみ・`reset()` は production 契約未模倣・processUp/processDown **存在せず** bitwise 比較対象なし。**Phase 0-1b/0-2 は実装なしには実行不能**（E-7 確定の維持）。

実装契約 5 項目（v2.5 §2.9 をそのまま継承）:
1. **係数生成の bitwise 一致**: prepareStage の全手順（:291-:368）を演算順序ごと複製。`juce::MathConstants<double>::pi` 経由の `std::sin` も同一経路
2. **履歴・分岐順序の一致**: keep・silence fast path（:583-613）・境界 guard（:620-649）・denorm クリア（:558-559）・isBadSample 位置
3. **比較対象（3 項目・bitwise）**: (a) 係数配列全要素 (b) シード固定疑似乱数 3 ブロック × 2 preset × ratio {2,4,8} の up/down 出力 (c) reset 前後状態差分（atomic 3 vs 1 の再現）
4. **candidate マクロ 2 系統の一致性 assert**: `PolyphaseGainFidelityTests.cpp`（新設・add_executable 40 番目）
5. **shadow cand は予測にすぎない**: 最終判定は flag ON ビルドの実測。**REF-FIDELITY は shadow fidelity の gate であり production 正しさの gate ではない**（R12-8）

### 3.10 Phase 0 適用要件

production src/ modified=0 / staged=0 / measurement のみ test-only / commit 禁止（O-1 凍結・R12-10）/ calibration 禁止 / **0.75^N 補償の追加禁止（R12-5: production には補償が存在しないことを確認済み・Phase 3-A でも再確認）** / evidence/ は C0 適用後の実行形態で。

---

## 4. §3 harness / production 潜在欠陥（F-2 / F-3 / F-4）

### 4.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（R11-6 確定の維持）

- 誤順: `BassBuzzMeasurement.cpp:1799` `configureProbeFlatEQ(e)` → `:1803` `setAutoGainStagingEnabled(false)`（AGC 再 ON・実測 `staging=0 eqAGC=1`）
- 正順: `:1833` staging false → `:1834` configureProbeFlatEQ（本版で行番号再確認済み）
- **修正案 A（GO 候補）**: 3 行 swap（:1799 と :1803 の入替・コメント更新）。パッチは `remediation_v22_prep_patches_20260922.md §2`（未適用）
- 検証: `--buzz-rigcheck=eq` で `staging=0 eqAGC=0`
- B-1 の原因ではない（[EQ_DIRECT] ratio=1.000000 が独立証拠）

### 4.2 F-3 / F-4（記録のみ・継承）

F-3（EQ dry/wet 混合・`EQProcessor.Processing.cpp:982-996`）は別 work item。F-4（`ir`=dry baseline / `irwet`=wet 対照）は R5-8 の意味定義を維持。

### 4.3 測定 entry 派遣の罠（継承）

`--buzz*` 引数があると `PPIT:1123` が測定 entry へ派遣し `:1223` の無条件 attribution に到達しない。attribution 再現は引数なし実行。

---

## 5. 評価点マトリクス（R12-9・新設）

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

**実適用点数 = per-config gate 8 × 8 構成 = 64 + P0-G 2 + P0-I 2 = 68 点**（+BUILD-ID run-level）。歴史的呼称「4経路×3条件×9 gate=108」は **potential evaluation cells**（全 gate が全経路に適用されないため実適用数ではない）として注記に残す。歴史的数値の 12 構成は v2.5 §9.2 の表（8 構成）と不整合だったため、本版で 8 構成に確定。

---

## 6. 推奨する実行順序（v2.6）

```
[Step 0] 現状固定 — 技術側完了
  - source freeze 実施済み（R12-6・ConvoPeq.md 2026-09-21 15:52:48・FRESH）
  - P0-F 数式確定（R12-1）/ E-1c 設計確定（R12-2）/ 二重補償不存在確認（R12-5）
  - 検証スクリプト実体化: tmp/v26_p0f_gate_calc.py → tmp/v26_p0f_gate_calc_results.txt（実行済み）
  - O-19 解決済み（R12-6 訂正後・2026-09-21）: `build_identity_gate.py` は build.bat gate のため `src/tools/` 復元・`check_layout_offsets.py` は `tools/` 移動のまま確定
  - 残: ユーザー判断（O-1〜O-19 / G-0 / T-1 / T-2 / T-3 / Phase 0 GO / O-16 C0）

[Step C0] evidence policy 単独 commit（O-16・R11-8 / R12-10）

[Step 1] §3 B-1 Phase 0（production 変更 0）— G-0 + T-1/T-2/T-3 + C0 が前提
  - 0-1 Baseline → 0-1b REF-FIDELITY（§3.9 契約の shadow 実装）→ 0-2 Candidate
  - 0-3 周波数（E-1 base gate + E-1c cand gate + E-1b 記録）→ 0-4 block/reset + macro assert
  - 0-5 SoftClip（R12-4 数値契約）→ 0-6 float/double
  - harness は凍結（O-1）・BUILD-ID 行は Phase 0 から record

[Step 2] Phase 0 review（ユーザー GO gate）
  - P0-E は E-1（base）+ E-1c（cand no-regression）として提示
  - GO → Phase 1 資格 / FAIL → 案D → tap再設計 → TPD型

[Step 3] §4 F-2（test-only・案 A・3 行 swap）
[Step 4] §4 F-1（A→B→B'→C・failureReason 主判定・R12-11 の移行注意）
[Step 5] O-* commit（O-18 ジェネレータ修正を含む・O-1 は characterization 完了後に解凍）
[Step 6] B-1 Phase 1 以降（Step 2 GO 前提）
  - CMake option 適用 → flag ON 実測（BUILD-ID 行必須・R12-7）
  - 4 経路 × 8 構成で P0 再確認（§5 マトリクス）
  - Phase 3-A: **0.75^N 補償不存在の再確認（R12-5 の検索を再実行）** → flag default ON
  - Phase 3-B characterization → 3-C calibration → 3-D rigcheck threshold（commit 分離維持）
[Step 7] 別課題: R-2（parseOnOff 前例）→ R-1 → B-3, D-1（BUILD-ID 統合）, D-2
```

---

## 7. 検証計画（v2.6）

### 7.1 検証スクリプト実在表（本版で実在確認済みのもののみ列挙）

| ファイル | 目的 | 状態 |
|----------|------|------|
| `doc/work113/model_polyphase_20260920.py` + `_results.txt` | authoritative モデル | 実在・本版全文読了 |
| `tmp/v24_ripple_band.py` / `tmp/v24_d1_invariance.py` / `tmp/v24_latency_probe.py` | dev 全桁再現 / D1 81 測定 / latency peak | 実在 |
| `tmp/cio_literal.py` / `tmp/cio_octave.m` / `tmp/maxima_check.mac` | 逐語転写 / Octave / Maxima | 実在 |
| **`tmp/v26_p0f_gate_calc.py` + `tmp/v26_p0f_gate_calc_results.txt`** | **R12-1（latency 式 4 構成 + peak gate）/ R12-2（E-1c 42 点判定）/ R11-3 再確認** | **本版で作成・実行済み（全 PASS）** |
| ~~`tmp/v25_*.py` 3 件~~ | v2.5 §7.1 が参照したが**実在しない**（保存漏れ） | **計画書から削除・§1 訂正** |

### 7.2 単体 / 統合 / 静的解析

| 項目 | コマンド | 期待 |
|------|----------|------|
| B-1 attribution | `build\Release\AudioEngineHarness.exe`（引数なし） | §3.4 全 row |
| F-2 | `--buzz-rigcheck=eq --buzz-dur=2.0` | `staging=0 eqAGC=0` |
| F-1 | `ctest --test-dir build --output-on-failure` | 移行先 PASS |
| B-1 Phase 0/1 | 同上（flag ON/OFF rebuild） | OFF ≈ 0.75^N / ON ≈ 1.0・**log 先頭に BUILD-ID 行** |
| 静的解析 | cppcheck 2.21.0（`C:\Program Files\Cppcheck\cppcheck.exe`） | CIO.cpp / TPD.cpp 指摘 0 |
| clang-tidy 23.1.1 | compile_commands 経由のみ（単独実行不可） | — |
| ConvoPeq freshness | `python output_sourcecode_markdown.py --check` | FRESH（現状 FRESH・O-18 修正版） |

---

## 8. ロールバック（v2.5 §8 継承・変更なし）

§1 commit → `git revert` ／ B-1 behavioral → compile-time rollback（flag OFF rebuild）／ B-1 source → `#if` ブロック + option 削除 ／ Shadow → test-only revert ／ F-2 → 3 行 swap の再 swap ／ O-18 ジェネレータ → revert 後に ConvoPeq.md 再生成。

---

## 9. AGENTS.md 修正パッチ（R11-7 の維持・O-10 commit 用）

v2.5 §15 の内容をそのまま維持: `AGENTS.md` の「**Maxima は未インストール**」は誤り（`C:\maxima-5.50.0\bin\maxima.bat --very-quiet --batch-string` で稼働確認済）。O-10 commit 時に同時適用（差分統計は commit 時に再計測・現在の AGENTS.md は +71/−3 まで増加しているため）。

---

## 10. 監査ログ・参照（v2.6 追加分）

| ファイル / 手段 | 確認 | 結果 |
|----------------|------|------|
| `src/audioengine/AudioEngine.Processing.Latency.cpp` 全文 154 行 | 式・コメント・定数 | ✓ R12-1（:6-8 / :22-23 / :29-32 / :117-122） |
| `tmp/v26_p0f_gate_calc.py` 実行 | latency 4 構成・peak gate・E-1c 42 点・DC N=1..8 | ✓ 全 PASS・results 保存 |
| `doc/work113/model_polyphase_20260920.py` 全文 + `_results.txt` 全文 | D1 42 点の転記元・D2/D3・argmax | ✓ 転記値一致 |
| `src/CustomInputOversampler.cpp` :287-390/:392-450/:452-484/:492-568/:570-723・`.h` 全文 | 行番号 1 文字照合 | ✓ |
| `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h` 全文 102 行 | 骨格の実態 | ✓（R11-9 継承） |
| `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp` :1280-1440/:1793-1840 | OS_DIRECT 出力項目・F-2 行番号 | ✓（R12-4 の受皿確認含む） |
| `CMakeLists.txt` :40 近傍/:1043/:1062/:1196/:1282/:1899 | option 配置・到達性 | ✓ |
| rg 検索（production 全体・0.75 / compensate / makeup） | 二重補償リスク | ✓ 補償 0 件（R12-5） |
| `python output_sourcecode_markdown.py`（修正版）+ `--check` | source freeze | ✓ FRESH・O-18 |
| `git status` / `git diff HEAD --numstat` | O-table 更新 | ✓ O-18/O-19 検出 |

---

## 11. 判断待ち項目（v2.6）

### 11.1 ユーザー意思決定

O-1（**凍結→完了後 commit**・R12-10）/ O-2〜O-17 は v2.5 §11.1 の推奨を維持 / **O-18（generator 修正）commit 推奨** / **O-19（src/tools 削除）出所確認のうえ復元推奨** / G-0 / Phase 0 GO / F-2 案 A / F-1 全実施 / Phase 3 commit 分割。

### 11.2 残存技術判断（確定案提示済み・承認のみ）— 3 件

| ID | 論点 | v2.6 確定案 | 根拠 |
|----|------|-------------|------|
| **T-1** | REF-FIDELITY gate の実行条件 | §3.9 の 5 項目契約（骨格のままでは実行不能・R12-8 の性格明記を含む） | R10-6 / R11-9 / R12-8 |
| **T-2** | cand D1 の gate 可否 | **gate する（E-1c 絶対形）**。base E-1 と E-1c の 2 本立て。f̂ 0.40/0.45 は記録のみ | R12-2（42 点実証） |
| **T-3** | BUILD-ID gate の必須化 | Phase 1 の全測定 log に `[BUILD] CONVOPEQ_CORRECT_POLYPHASE_GAIN=0/1` を必須化（欠落 = 測定無効） | R12-7 |

### 11.3 B-1 採用案の前提条件（v2.5 §11.3 から更新）

1. DESIGN-CONTRACT-A + **T-1/T-2/T-3** 明示承認（G-0）
2. REF-FIDELITY bitwise（§3.9・shadow fidelity gate）
3. round-trip DC = 1.0 ± 1e-6
4. P0-B/C passband 維持（8 構成）
5. **P0-D stopband 維持（decimator 責務）**
6. **E-1 base D1 = −9.5424 ±0.01 dB + E-1c cand D1 ≤ −9.442 dB @f̂≤0.35（interpolator 責務）**
7. P0-F latency 不変（コード同型式 §3.8 の基線）
8. P0-I 数値契約（NaN/Inf FAIL・飽和 FAIL・指標記録）
9. P0-C' differential 帰属 + P0-G（絶対誤差 gate・相対誤差記録）
10. D1 測定は補正軸 + expected bin 主報告（R9-1/R9-2/R10-4/E'-5）

### 11.4 v2.6 最終判定表

| 項目 | 判定 | v2.5 から |
|------|------|-----------|
| レビュー必須修正 1（P0-F） | **取り下げ（式は正しい）+ 式の再定義で再発防止** | 裁定 |
| レビュー必須修正 2（E-1c） | **採用（絶対形）** | **新設** |
| レビュー必須修正 3（D/E 責務） | **採用** | 明記 |
| レビュー必須修正 4（P0-I 契約） | **採用** | 契約化 |
| レビュー必須修正 5（source freeze） | **実施済み**（O-18 含む） | 完了 |
| レビュー必須修正 6（BUILD-ID） | **採用（設計確定・Phase 1 適用）** | 新設 |
| §1 O-1〜O-19 | GO 候補（O-18 追加・**O-19 解決済み**） | 拡充 |
| §3 F-2 / F-3 / F-4 | GO 候補 / 記録継承 / 記録継承 | 変更なし |
| §4 F-1 | GO 候補（R12-11 追加） | 強化 |
| §5 R-2 | 仕様確定（c4a08171 前例） | 維持 |
| B-1 原因分析 | GO（5 系統 + 数値計算） | 維持 |
| **B-1 Phase 0** | **GO（G-0 + T-1/T-2/T-3 + C0 承認が前提）** | 条件追加 |
| B-1 Phase 1+ / 案 E / Production patch / Calibration | **HOLD / 有力仮説 / HOLD / HOLD** | 変更なし |
| 二重補償リスク | **不存在を実測確認（R12-5）** | 解消 |
| source baseline | **freeze 完了（2026-09-21 15:52:48）** | 完了 |
| 技術的未確定 | **3 件（T-1/T-2/T-3）・確定案提示済み** | 1 件追加 |
| 計画書品質 | v2.5 §7.1 の実在しないスクリプト参照を訂正・108 を実適用 68 点に確定 | 是正 |

---

## 12. エビデンス（v2.6）

- **5 系統クロスチェック**: v2.5 §12.1 を継承（production バイナリ / モデル / NumPy 逐語 / Octave 11.3.0 / Maxima 5.50.0）
- **本版の独立数値計算**: `tmp/v26_p0f_gate_calc.py`（作成・実行）→ `tmp/v26_p0f_gate_calc_results.txt`
  - latency: S1 15.00 / 511単段 255.00 / IIR3 290.25 / LP3 582.25・peak gate 全 PASS・レビュー誤読形（7.50/127.50/145.12/291.12）との対比
  - E-1c: 相対形 FAIL 1 点（31/90 @f̂0.45 の誤警報）→ 絶対形 FAIL 0 点・最小 margin 36.7 dB
  - DC: N=1..8 全桁一致
- **ツール実測**: v2.5 §12.2 を継承 + 本版は headroom proxy / context-mode MCP（ctx_execute・数値計算）/ rtk+rg（WSL）/ Git Bash / Python 314 を使用（ZCode 環境: proxy 経由の 3 層パイプライン規約に従い実行）

## 13. v2.6 で追加した確定事項（R12 一覧）

- **R12-1** P0-F 数式の裁定（v2.5 式は正しい・レビュー再計算は二重除算の誤り・式をコード同型で再定義）
- **R12-2** E-1c candidate image no-regression gate（絶対形 −9.442 dB @f̂≤0.35・相対形の誤警報を実証）
- **R12-3** P0-D/E 責務分離 + 周波数軸（Fs_in base 正規化）の固定
- **R12-4** P0-I 数値契約（NaN/Inf FAIL・飽和 FAIL・指標記録）
- **R12-5** 0.75^N 補償の不存在確認（二重補償リスク解消・Phase 3-A 再確認条項）
- **R12-6** source freeze 実施 + ジェネレータ修正（O-18）
- **R12-7** BUILD-ID gate 設計（D-1 統合）
- **R12-8** REF-FIDELITY の性格明記
- **R12-9** 評価点マトリクス（8 構成・実適用 68 点・108 は potential）
- **R12-10** O-1 凍結 + evidence commit 分離（C0/C1/C2）
- **R12-11** F-1 移行の first-fail-wins 注意
- **R12-12** P0-G 相対誤差の扱い

---

*本書は v2.5（`remediation_plan_20260922_v2.5_revised.md`）の再改訂版 v2.6 である。*
*v2.5 本文は参照文書として維持し、**R12-1〜R12-12 が v2.5 と矛盾する箇所では本書を優先する**。*
*R5〜R11 は本書でも有効（上記是正点を除く）。中間エビデンスは `remediation_v22_intermediate_20260921.md`、パッチ案は `remediation_v22_prep_patches_20260922.md`（production 未適用）。*
**継承の監査・確定事項**: R11-1〜R11-10 / R10-1〜R10-13 / R9-1〜R9-6 / R8-1〜R8-12 / R7-1〜R7-4 / R6-1〜R6-7 / R5-1〜R5-12。
