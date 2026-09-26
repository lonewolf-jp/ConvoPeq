# ConvoPeq 残件 改修計画書（2026-09-21 時点・再改訂 v2.0）

- **版**: v2.0（v1.9 全項目の外部レビュー監査（2026-09-21 貼付レビュー）反映 + 本セッションの数値・ソース再検証を統合）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/` の未 commit 差分 0 件（test-only 2 ファイルを除く・2026-09-21 再実測）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py`（`prepareStage`/`interpolateStage`/`decimateStage` を C++ から厳密移植。ループ実装との等価性 ≤2.2e-15 自己検証済み）。**v2.0 で 2026-09-21 環境下に再実行し、保存済み全出力 `model_polyphase_20260920_results.txt` と diff 0 行で完全一致を独立確認**
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を **「有力仮説（candidate hypothesis）」として厳密に維持**（R5-4）。defect 自体（DC round-trip = 0.75^N）は confirmed。**Phase 0 完了まで案 E を「確定」と表記しない（文書全域で強制・R5-4）**
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v1.9 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする。C++ 分岐は **`#if` を使用（`#ifdef` は禁止）**

---

## v1.9 → v2.0 改訂サマリ（外部レビュー監査 + 本セッション再検証による修正）

v1.9 の全行番号（50+ 件）・全コード解釈を本セッションで独立再監査した結果 **全件一致**。さらに本セッションで閉形式モデルを再実行し、round-trip 応答の **厳密構造恒等式** を新規確定した（R5-9）。外部レビューが要求した 2 点の必須修正（P0-E の論理意味の明確化・P0-I 安全性 gate の追加）を R5-1/R5-2 として反映し、レビュー推奨の実行順序変更（B-1 Phase 0 先行・R5-3）を採用する。

| ID | 区分 | 内容 | v2.0 での確定 |
|----|------|------|----------------|
| **R5-1** | **P0-E 再定義** | P0-E の gate は「image rejection 品質」ではなく **「candidate 誘発 image 不変性（image invariance）」** であることを明文化。**名称を P0-E に残したまま語義を分離**: (E-1 記録) baseline D1 鏡像比を全 design × f̂ 7 点で記録（**baseline image quality は記録専用であり quality gate ではない**）。(E-2 gate) D2/D3 の base==cand 差 ≤0.01 dB。(E-3 新設) **baseline 鏡像比は「中間 upsampled 信号の品質評価（design 契約）」であり、最終 round-trip alias rejection の保証は P0-D が担う** | **必須修正①を反映**。旧 gate「candidate ≥ 80 dB」廃止は v1.9 R3-1 のまま維持 |
| **R5-2** | **P0-I gate 追加** | P0-I を **I-a（DC unity・gate）/ I-b（安全性・gate）/ I-c（動作点・record-only）の 3 分割** に再定義。**I-b gate（新設）**: NaN/Inf なし（`isBadSample` 0 件）・`corruptionDetected == false`（全ブロック）・`hardFallbackActive == false`・バウンド違反 0（`interpolateStage` :509-554 / `decimateStage` :626-649 の corruption 経路非発火）・**probe（−6dBFS）入力に対する予期しないハードクリッピングなし**（出力 peak が SoftClip clip threshold に接触しない）。I-c（旧 I-b の動作点記録）は record-only のまま | **必須修正②を反映**。SoftClip 経路は production path であるため「安全性・異常有無」を record-only に留めない |
| **R5-3** | **実行順序変更** | §6 実行順序をレビュー推奨版に変更: **Step 0 現状固定 → Step 1 B-1 Phase 0 → Step 2 Phase 0 review（ユーザー GO gate）→ Step 3 F-2 → Step 4 F-1 → Step 5 O-1〜O-12 → Step 6 B-1 Phase 1 → Step 7 別課題**。根拠: B-1 が最大リスクであり、Phase 0 の完了が「計画続行可否」を最速で確定する。F-2（eq rigcheck AGC 呼出順）・F-1（validator テスト移行）は B-1 の測定経路（`runOversamplerDirect`・`prepareSingleStage`）に触れないため順序逆は安全 | O-1（計測 commit 決定）を Phase 0 後に置くことで、Phase 0 の実測経験を commit 判断に反映可能 |
| **R5-4** | **仮説/欠陥の分離強制** | 文書全域で **`B-1 defect = confirmed（DC round-trip = 0.75^N 12 桁実測）` / `corrective E = candidate hypothesis` の 2 つを厳密分離**。v1.9 §2.3.1 の「案 E の実装（R3-9 で確定）」表記を「**案 E 実装案（candidate・Phase 0 全 PASS + ユーザー GO まで確定としない）**」に訂正。Phase 1 着手条件にも「案 E は仮説のまま Phase 0 を通過する」ことを明記 | 冒頭・§2.3・§6・§11 の表現を統一 |
| **R5-5** | **E5 の証拠格付け** | DESIGN-CONTRACT-A の E5（JUCE `dsp::Oversampling` の up 側 ×2）を **「有力な実装慣行の傍証（補助証拠）」** と明記。証拠の強さの序列: ① 現行コードからの 0.75 厳密導出 → ② 実測 0.75^N 再現 → ③ shadow model で candidate 1.0 導出 → ④ passband/stopband/latency/block/reset 検証 → ⑤ 実装慣行との整合（補助） | JUCE 一次確認済み（`juce_Oversampling.cpp` up `buf[N-1] = 2 * samples[i]` :185 / down `buf[N-1] = bufferSamples[i << 1]` :228） |
| **R5-6** | **rollback 命名** | §8 ロールバック計画の §2 behavioral 行を **「compile-time rollback（build artifact 切替を要する・runtime 即時復旧ではない）」** と明記。flag OFF rebuild が第一手段であることを維持 | レビュー §21 の指摘を反映 |
| **R5-7** | **F-1 判定基準** | §4 移行後の semantic equivalence gate を **構造化 `failureReason`（`ValidationFailureReason` enum・RuntimePublicationValidator.h:13/:24）を主判定**とし、`errorMessage` 文字列は補助 assert に限定 | レビュー §25 を反映。現行コードが既に構造化 reason を持つことを本日再確認 |
| **R5-8** | **F-4 用語定義** | §3.3 F-4 の rigcheck モード名を明文化: **`ir` = IR ロード後も convBypass 維持（dry ベースライン測定モード）・`irwet<digit>` = convBypass を解除し wet 畳み込みを実際に通す対照モード**（OS 倍率はモード末尾 1 桁: irwet1/2/4/8） | レビュー §24 を反映。`setConvolverBypassRequested(false)` :1778 実測 |
| **R5-9** | **数値メカニズム新規確定** | 本セッションで round-trip 応答の **厳密構造恒等式** を数値確定: **`h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`（base c=0.25 / candidate c=0.5・残差 1.4e-17 = FP フロア）**。帰結: (a) passband では conv 経路（案 E で不変）と center 項が同一位相因子を持つため **base==cand 比は理論上厳密に (4/3)^N**（§2.7.1 全点一致と整合）。(b) 遷移端近傍（31/90 @0.45）では近接打ち消しの残差として center 係数差が顕在化し得る（全 chain |H| 偏差 +1.635 dB＝§2.7.1 既載値と一致）。(c) D2 tone 測定の @0.45 差 0.27 dB は **FP 量子化フロア依存**（longdouble では −294.8/−296.2 dB まで低下・R4-5 の帰属を数値裏付け） | v1.9 R4-5/R4-12/R4-13 の帰属（FP 量子化拡大・gate 外）を数値で裏付け。gate 帯域 [0.005,0.40] は構造的に妥当 |
| **R5-10** | **数値契約再確認** | IIR3 (4/3)^N 偏差 @0.49 = +0.0062 dB・LP3 +0.0025 dB・S1 @0.49 +3.3927 dB を本日再実行し結果ファイルと全点一致 | v1.9 R4-8 のまま再確認 |
| **R5-11** | **P0-C' 適用域再確認** | P0-C' の per-config 適用域（IIR3/LP3 は f ≤ 0.45・S1 は f ≤ 0.30・0.49 は記録のみ）を v1.9 のまま確定 | レビュー §10 で妥当と判定済み |
| **R5-12** | **静的解析再確認** | cppcheck（`--enable=warning,performance,portability --std=c++20`・指摘 0 件）/ clang-tidy（portability-simd-intrinsics + bugprone-branch-clone のみ・実 bug 0 件）を v1.9 のまま維持 | R4-6/R4-7 継承 |

**v1.9 から変更しない点（再確認）**: 案 E の技術的内容（`centerValue *= 2.0` 1 行追加）/ Phase 0 の 5 構造（Baseline・G-0・REF-FIDELITY・Candidate・Differential）/ 段階リリース骨格 Phase 0→1→2→3-A/B1/B2/C/D→4 / §3 F-2 案 A・F-3 案 D・F-4 案 B / §4 MIGRATE CASES THEN RETIRE / §8 ロールバック骨格 / §2.7.1・§2.7.2 の candidate 絶対値基準値（本日全点再現）/ §2.7.3 の D1/D2/D3 実測表 / D2 gate 帯域 [0.005, 0.40]（R4-13）/ §11 の判定表骨格。

---

## 0. 凡例と全体戦略（v1.8/v1.9 から変更なし）

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | B-1 | 全オーディオ経路 | flag OFF rebuild（compile-time rollback） |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only（F-3 は production 欠陥の記録） | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1〜O-12, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: §1 意思決定 → §2 B-1 段階リリース → §3 → §4 → §5（v1.8 §0 のまま）。ただし §6 の実行順序は R5-3 によりレビュー推奨版に変更。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-12）— v1.9 §1.1〜§1.13 のまま（本日再実測で全件確認）

- O-1: 差分 +634/−2（BassBuzzMeasurement.cpp）+8/−0（PublishPipelineIntegrationTests.cpp）・窓 :2042-2044・`[OS_DIRECT] roundTripGain=` :1367・`runOversamplerDirect()` :1293 ← `runEqDirectDriveAttribution()` :1526/:1532 ← PPIT main :1085 から（前方宣言 :1116・呼出し :1223）。**推奨案 A**
- O-2: 台帳更新は O-1 と独立 commit
- O-3: `ConvoPeq.md` は commit しない（再生成: `python output_sourcecode_markdown.py`）
- O-4: `Testing/Temporary/CTestCostData.txt` は現状維持
- O-5: `.opencode/opencode.json` は触らない
- O-6: push（ahead 2 / behind 0）はユーザー手動
- O-7/O-10: `AGENTS.md` Modified（+8/−2・MIMO Desktop 追記）→ **記録更新扱いとして commit 可**（O-1 同梱 / 独立のいずれかユーザー判断）
- O-8: `tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` Modified → **commit しない**（将来 work item: `.gitignore` + `git rm --cached`・HOLD）
- O-9: `docs/tool-inventory-2026-09-20.md`（untracked・3102 bytes）→ 触らない（ユーザー判断）
- O-11: `doc/work113/residual_tasks_20260919.md` Modified（+72/−0・B-1 帰属詳細）→ **台帳更新として commit 可**（O-1/O-2 統合可）
- O-12: `ConvoPeq.md` Modified（+75/−0・生成物）→ O-3 方針継続（commit しない）
- O-13（本書追加・v1.9 §1.13 継承）: `doc/work113/` 計画書 9 件（v1.8/v1.6/v1.7/v1.8/v1.9 + 本書 v2.0）untracked。**推奨**: 本書承認確定後、承認版 + 台帳を 1 commit として登録（O-2/O-11 同梱可）。`.cline/`（untracked）は触らない

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定している事実（v1.9 §2.1 を継承・2026-09-21 再実測で独立再確認）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     DC round-trip 0.750000（up 出力平均 0.75 / down 単体 DC 1.0 — DC 基準・R3-6）
            peak 基準では upPeak = 1.0×入力（even 位置）・round-trip 出力 peak 比 0.75
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0, internalMaxBlock)・DSPCoreLifecycle.cpp:188/261 実測）
数学的導出  up: even=1.0（conv×2）+ odd=0.5（center）→ 平均 0.75 / down: 0.5+0.5 = 1.0
engine fit  0.98379 × 0.75^max(log2 effOS, 1) は doc の経験式。src/ に 0.98379 / effOS 定数は不在（本日 grep 再確認）
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾（係数列自体は不変 → R3-11）
production  0 / commit 0
```

**閉形式モデルによる独立再確認（2026-09-21 再実行・結果ファイルと diff 0 行で完全一致）**:

| 構成 | baseline DC | candidate DC | 0.75^N |
|------|-------------|--------------|--------|
| 単段 31/90 | 0.750000000000 | 1.000000000000 | 0.75 |
| IIR3 (511/127/31) | 0.421875000000 | 1.000000000000 | 0.75³ |
| LP3 (1023/255/63) | 0.421875000000 | 1.000000000000 | 0.75³ |

### 2.2 根本原因の特定（v1.9 §2.2 を継承・全行番号 2026-09-21 再実測で一致）

- `prepareStage` :287-390（centerTap :292・center 0.5 :335/:348・scale :341・convCoeffs :360-365・centerDelayInput :368）✓
- `interpolateStage` :492-568（`convValue *= 2.0` **:557**・denormal clamp :558-559・center 位相に ×2 なし :543-547/:564）
- `decimateStage` :570-723（silence パス **:583-613**（R4-4 訂正版）・通常パス :615-649・n ループ :652-・`acc = stage.centerCoeff * centerSample` :658・**down 分岐に ×2 補償なし**）
- `processUp` :725-783（hardFallback 透過 :728-738・容量超過で空ブロック返却 :744-748・段ループ :764-776・`currSamples <<= 1` :775・stages[0] を入力レートで最初に適用）
- `processDown` :785-（hardFallback 透過 :789-806・corruption クリア :808-818・`consecutive >= kHardFallbackAutoClearThreshold(=4)` で hardFallback 発火 :814-815・容量超過 clear :826-832）
- `reset()` :452-467（3 アトミック解除 :464-466）/ `clearAllStages()` :469-484（1 アトミック解除 :483）— **R4-3**: 履歴クリア範囲は完全同一、差分はアトミック解除数（`hardFallbackActive` 解除は `reset()` のみ）。**Phase 0 測定系は `reset()` 経由初期化を必須とする**

### 2.3 修正案の比較と DESIGN-CONTRACT-A（E3 のみ R5-5 で精緻化）

案 A〜E の比較（A/B/C/D 不採用・**E 有力仮説**）は **v1.8 §2.3 のまま**。

#### 2.3.1 案 E 採用根拠（v2.0 版・G-0 として明示承認する対象）

**DESIGN-CONTRACT-A（v2.0・v1.9 から変更なし）**:
> Oversampler は interpolation convention として両 polyphase 位相が同一 DC gain（= 2 倍密化の gain convention）を持ち、up/down round-trip の DC/passband gain が unity であることが要求される。

**証拠の強さの序列（R5-5）**: ① コード構造からの厳密導出（主）→ ② 実測再現（主）→ ③ shadow model による candidate 1.0（主）→ ④ Phase 0 特性検証（主）→ ⑤ 実装慣行との整合（**補助**）:

| 証拠 | 内容 | 区分 | v2.0 での状態 |
|------|------|------|----------------|
| **E1** | コード自身が既に gain-2 convention（`convValue *= 2.0` — :557 実測）。欠陥は convention ではなく center 位相への不完全適用 | **主** | 変更なし・確定 |
| **E2** | down 側 half-band decimator は既に DC gain 1.0（0.5+0.5）。up だけ 0.75 → up/down 非対称は `isSymmetricUpDown` / static_assert の契約前提と矛盾。**static_assert は「taps 列の同一性」の宣言であり、案 E は taps・遅延を変更しないため assert は成立し続ける（R3-11）** | **主** | 精緻化 |
| **E3** | D1（up 単段出力）で baseline 鏡像比が **f̂・design 依存で −16.65〜+14.93 dB**（理論漸近 −9.5424 dB = 20·log10(1/3)・255/140 @0.45 では +14.93 dB で鏡像が信号超え）。FIR 設計減衰 −87〜−159 dB に対し最大 ~165 dB の開き = up 分岐非対称（gain convention 欠陥）の構造証拠。**ただし D1 鏡像は同ステージ down 段 stopband で抑制され round-trip 出力には伝播しない（R3-1）**。**defect の主証拠を DC round-trip = 0.75^N（12 桁実測）に確定。D1 非対称（最悪 +14.93 dB）は補助証拠** | **主（DC）/ 補助（D1）** | **v1.9 の「−9.7〜+5.8 dB」表記を §2.7.3 D1 表実測値の全範囲（base −16.65〜+14.93 dB / cand −35.94〜+28.14 dB）に訂正** |
| **E4** | `isLinearPhaseFIR = true` / `isSymmetricUpDown = true`（h:21-22）+ Latency.cpp:6-7 static_assert | **主** | 変更なし・確定 |
| **E5** | JUCE `dsp::Oversampling` は up 経路で `buf[N−1] = 2·samples[i]`（両位相に同一 ×2）・down は ×1（juce_Oversampling.cpp:185 / 228 一次実測✓）。**ただし JUCE がそうしている → ConvoPeq もそうすべき、とは論理帰結しない（FIR 係数正規化方式・polyphase 分解・interpolation convention が異なり得る）** | **補助** | **R5-5 により「実装慣行の傍証」と明記** |

**案 E の実装案（candidate・R5-4 により「確定」表現を排除）**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内
        convValue *= 2.0;                 // 既存（:557）
#if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;               // ★ 案 E 実装案（両 polyphase 位相への ×2 対称適用・candidate）
#endif
        if (fastAbs(convValue) < kDenormThreshold) convValue = 0.0;   // :558（clamp より前に ×2）
        if (fastAbs(centerValue) < kDenormThreshold) centerValue = 0.0;
```

**実装上の注意（新規・v2.0）**: 案 E 適用時、center 位相の denormal clamp は ×2 後の値に対して行われる。`|centerValue| ∈ [kDenormThreshold/2, kDenormThreshold)` のサンプルは flag ON では 0 化されない（flag OFF では 0 化）。測定系の DC/Tone 測定への影響は FP フロア以下だが、Phase 0-2 で **clamp 前後の挙動差**を記録に含めること（R5-11 関連）。

**gain 変化**: candidate/baseline 比 = (4/3)^N（1 段 +2.4988 dB / 2 段 +4.9977 dB / 3 段 +7.4963 dB）— 閉形式モデル + 実測一致（本日再確認）。

**round-trip 応答の厳密構造恒等式（R5-9・本セッション数値確定）**:

```
単段 round-trip のインパルス応答は次の恒等式を FP フロア（残差 ≤1.4e-17）まで厳密に満たす:

  h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n − centerTap]   （c: base = 0.25 / candidate = 0.5）

帰結:
  (a) passband（|conv 経路| ≫ center 項）: 両項が同一位相因子を持つ → cand/base 比 = (4/3)^N 厳密不変
      （§2.7.1 の全点 ≤0.0002 dB dev と一致）
  (b) 遷移端近傍（31/90 @0.45・@0.49）: conv 経路応答の落下により同位相性が崩れ、
      center 係数差（0.25 vs 0.5）が干渉残差として顕在化（実測偏差 +1.635 dB @0.45 / +3.393 dB @0.49）
      → §2.7.1 の P0-C' 適用域を f ≤ 0.30（S1）とする根拠が構造的に確定
  (c) D2 stopband 測定（round-trip 出力の mirror 線）の @0.45 差 0.27 dB は FP 量子化フロア依存
      （longdouble 実測: base −294.77 dB / cand −296.19 dB — double 環境では −137 dB 級のフロア）
      → R4-5 の「FP 量子化ノイズ拡大」帰属を数値で裏付け、gate 帯域 [0.005, 0.40] を構造的に妥当化
```

### 2.4 設計時に検討すべき 8 観点・2.5 段階リリース設計

- 8 観点: **v1.8 §2.4 のまま**（全項 Phase 0 で確認）。
- 段階リリース: **v1.8 §2.5 のまま**（Phase 0 → 1 → 2 → 3-A/B1/B2/C/D → 4・baseline record-only・calibration は 3-B1/B2 完了後の 3-C/D）。**Phase 1 の CMake 実装は R3-8 により精密化**（v2.0 維持）:

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
- in-source は safety net のみ。既定 OFF（既存挙動維持）。Runtime flag は不採用（DSP algorithm convention は compile-time selection が適切・レビュー §20 確認済み）。

### 2.6 Phase 0 characterization（P0-E を R5-1・P0-I を R5-2 により再定義・他は v1.8 §2.6 を維持）

```
Phase 0-0  source/contract audit: DESIGN-CONTRACT-A（v2.0 版・E1〜E5）を明示承認（G-0・承認者はユーザー）
Phase 0-1  P0-BL Baseline（record-only）: DC round-trip = 0.75/0.5625/0.421875（ratio 2/4/8・preset 非依存・±1e-6）
Phase 0-1b REF-FIDELITY gate（R2-2 維持）: Shadow Reference（baseline モード）== production
           （DC/impulse/周波数/ratio 2/4/8 × preset 2 種/partition 4 種/reset・double 経路 bitwise 一致・
            不可なら相対誤差 ≤1e-15。不一致なら reference を修正し一致まで Phase 0-2 に進まない）
           配置: src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h（header-only・production 変更 0）
           **初期化前提条件（R4-3）**: 測定開始時に必ず `reset()` を呼出（hardFallbackActive 解除のため）。
           `clearAllStages()` のみでは hardFallback 透過（:728-738/:789-797）が混入し得る
Phase 0-2  P0-CAND Shadow Candidate E: DC round-trip = 1.0 ± 1e-6（全構成）— 閉形式モデルで予備確認済み
Phase 0-3  frequency transfer（stage-rate 軸・§2.7 の軸定義に従う・D1/D2/D3 定義確定版）
Phase 0-4  block/reset characterization（R2-3 維持）: partition invariance bitwise（4096/1024×4/256×16/ragged・
           各ブロック長 ≤ maxInputBlockSize — processUp :744-748 の空ブロック返却 guard 実測確認済み）
           + reset contract bitwise（fresh+block#1 == stream→reset()→block#1・reset :452-467 は
           upHistory/downHistory clear 実測確認済み・clearAllStages :469-484 との差異は R4-3 参照）
Phase 0-5  SoftClip local OS characterization（R2 維持・prepareSingleStage(31, 90.0, internalMaxBlock)
           は DSPCoreLifecycle.cpp:188/261 実測確認済み・harness 実装 :1376-1445 実測）
Phase 0-6  float-host / double-host equivalence（R2-4 数値契約のまま: maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8）
```

**Phase 0-3 の測定定義（v2.0 確定 — image rejection の 3 定義を明確化・R3-1 + R4-13 + R5-1 反映）**:
- **D1（up 単段出力・2× rate）**: tone f̂ に対する 0.5−f̂ 鏡像成分比。**記録のみ（gate に使わない）**。**意図の明文化（R5-1）**: D1 は「中間 upsampled 信号の品質評価（現行 design contract の記録）」であり、**final round-trip alias rejection は P0-D で保証される**。D1 の値が悪く見えることは案 E の FAIL を意味しない
- **D2（単段 round-trip 出力）**: |H_rt(0.5−f̂)|/|H_rt(f̂)|。**passband では base==cand が厳密成立（§2.3.1 構造恒等式 (a)）。stopband 側は近接打ち消しの感度領域** → gate 判定帯域を限定
- **D3（full-chain 出力の worst in-band spur）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30}。passband 域では base==cand 差 = 0.00 dB 厳密成立
- **D2 判定帯域（v2.0 維持・R4-13）**: [0.005, **0.40**] Fs_in に限定。0.45 は FP 量子化拡大領域（R5-9 (c) でメカニズム確定）のため **記録のみ**（典型点 ≤0.40 では全 design ≤0.01 dB 確認済・R4-12）
- 測定系要件（v1.8 追加・維持）: 過渡トリム（全 stage taps + 512 sample 以上）+ Hann 窓を必須化（未トリム矩形窓では −79〜−90 dB の偽フロアが出ることを本モデルで実測確認）。corruption/hardFallback フラグ監視を必須化（hardFallback 透過経路 :728-738/:789-806 の混入排除）

**Phase 0-3 の残り（周波数応答）は v1.8 §2.6 Phase 0-3 [A]/[B]/[C]/[D] のまま**。ただし [D] passband ripple の判定帯域は R3-3 により **[0.005, 0.30] Fs_in（dense 評価）** に変更、[0.30, edge] は記録のみ。

#### 2.6.1 B-1-P0 GATE（v2.0・P0-E を R5-1・P0-I を R5-2 で更新）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT（絶対値）** と **Candidate vs Baseline（差分帰属）** の 2 系統。

| ID | 判定対象 | 判定条件（v2.0 確定値） | 合否 |
|----|----------|------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が明示承認済み（E1〜E5・v2.0 版添付） | □ PASS / □ FAIL |
| **G-BL** | Baseline sanity | **DC round-trip = 0.75^N（±1e-6）のみを gate 条件とする**（R3-2）。D1 鏡像比（−16.65〜+14.93 dB を含む）は記録のみ | □ PASS / □ FAIL |
| **REF-FIDELITY** | Phase 0-1b | Shadow Reference（baseline モード）== production: 全項目 bitwise 一致（不可なら相対誤差 ≤ 1e-15）。**初期化は `reset()` 経由必須（R4-3）** | □ PASS / □ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成）。**※これは案 E が unity-DC 設計目標を満たすことの必要条件であり、案 E が「正解」であることの証明ではない（sufficient condition ではない・レビュー §8）** | □ PASS / □ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB**（実測 −0.0000〜−0.0078 dB） | □ PASS / □ FAIL |
| **P0-C** | Passband ripple | **dense 評価（exact DTFT または ≥2^18 点 FFT）で [0.005, 0.30] Fs_in の max−min ≤ 0.05 dB**（candidate 実測: S1 0.001 / IIR3 0.025 / LP3 0.012 dB）。[0.30, 0.1 dB edge] は遷移落下として記録のみ | □ PASS / □ FAIL |
| **P0-C'** | Differential（帰属） | Candidate/Baseline 比 = (4/3)^N ± **0.05 dB**。**適用域 per-config（R2-7 維持・数値更新 R3-4/R4-8）**: IIR3/LP3 は f ≤ 0.45 Fs_in（実測偏差: IIR3 +0.0121 / LP3 −0.0025 dB）、S1 は f ≤ 0.30（実測 +0.0000〜+0.0002 dB）。0.49 Fs_in は記録のみ（実測: IIR3 +0.0062 / LP3 +0.0025 / S1 +3.3927 dB・v1.7 の IIR3 +0.043 は非再現）。Phase 2 flag-ON 測定が shadow 値と ±0.02 dB で一致すること | □ PASS / □ FAIL |
| **P0-D** | Stopband（stage-local・絶対 + 差分） | (D-1 差分) decimateStage は係数・コードとも不変 → Candidate/Baseline 差 = 0（FP 誤差 ≤1e-12）。(D-2 絶対) alias leakage ≤ **−(A_stage − 10) dB** を **[transition_end, 0.5] cycles/sample** で満たす（per-design 表 §2.7.2 は **本日全点独立再現・確定**）。transition 領域は記録のみ | □ PASS / □ FAIL |
| **P0-E（R5-1・image invariance と明記）** | **Image invariance（not quality）** | (E-1 記録) D1 baseline 鏡像比を全 design × f̂ 7 点で記録。**baseline 鏡像比は中間 upsampled 信号の品質評価（design contract の記録）であり quality gate ではない。final round-trip alias rejection は P0-D が保証**。(E-2 **gate**) D2 の base==cand 不変性: **[0.005, 0.40] Fs_in** の全測定点で差 ≤ **0.01 dB**（passband では厳密成立・§2.3.1 恒等式 (a)）。D3 の [0.05, 0.30] × {S1/IIR3/LP3} は base==cand 差 = 0.00 dB 厳密成立。(E-3) 0.45 は FP 量子化拡大領域（R5-9 (c)）のため記録のみ。**旧 gate「candidate ≥ 80 dB」は廃止**（R3-1: D1 定義でも到達不能・D2/D3 では base と同一のため無意味） | □ PASS / □ FAIL |
| **P0-F** | Latency | impulse 応答 peak 位置 = `Σ (taps[s]−1) × (baseRate/stageRate)`（±1 sample）+ centroid cross-check（±0.05 sample 目安）。案 E で不変（R3-11） | □ PASS / □ FAIL |
| **P0-G** | Float / Double host | same input / same partition / same reset state で: **maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8**。**reset state 初期化は R4-3 経由で実行**。**レポート上では「float==double（数値精度契約）」と「candidate==baseline（意図した DSP 変更契約）」を混在させない（レビュー §15）** | □ PASS / □ FAIL |
| **P0-H** | Block / reset contract | partition A/B/C/D bitwise 一致（不可なら ≤1e-12）・`fresh+block#1 == stream→reset()→block#1`（bitwise）。**reset は `reset()` 経路を使用（R4-3）** | □ PASS / □ FAIL |
| **P0-I（R5-2・3 分割）** | SoftClip local OS | **I-a（gate）**: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, centerTap=15, centerParity=1, convParity=0 実測）で Candidate DC = 1.0 ± 1e-6。**I-b（gate・新設）**: 測定系全体で NaN/Inf なし（isBadSample 非検出）・`corruptionDetected == false`（全ブロック）・`hardFallbackActive == false`・バウンド違反 0（:509-554/:626-649 非発火）・**probe（−6dBFS）に対する unexpected ハードクリッピングなし**（出力 peak が clip threshold 非接触）。**I-c（record-only）**: 動作点影響を candidate upPeak/downPeak/閾値到達度で記録（数値再校正は Phase 3-C） | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計、**または R3-12 記録の TruePeakDetector 型全 FIR 構造への変更**に方針変更する。

### 2.7 測定軸と実測基準値（v2.0 確定・authoritative = model_polyphase_20260920.py）

#### 2.7.0 周波数軸の正規化定義（v1.8 §2.7.0 を維持・確定）

```
own-rate cycles/sample: f̂ = f / Fs_stage（stage 自身の動作レート基準・有効域 [0, 0.5]）
Fs_in 基準（full chain）: f̂_in = f / Fs_in（入力帯域 [0, 0.5] のみ有効）
整合例（本モデルで再確認）: stage-0 (511 taps) の −0.1 dB edge = 0.2448 cycles/sample
 = 0.4896 Fs_in ≒ full-chain candidate edge 実測 0.4897 Fs_in ✓
```

#### 2.7.1 Full-chain 実測基準値（v2.0・dense exact-DTFT・baseline / candidate・本日全点再現）

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC baseline / candidate | 0.750000 / 1.000000 | 0.421875 / 1.000000 | 0.421875 / 1.000000 |
| candidate \|H\| @50Hz〜0.25 Fs_in | −0.0000〜−0.0004 dB | −0.0078〜−0.0108 dB | +0.0025〜−0.0047 dB |
| (4/3)^N 差分偏差 @0.45 Fs_in | **+1.6350 dB**（適用外・R5-9 (b) 構造帰結） | **+0.0121 dB** | **−0.0025 dB** |
| (4/3)^N 差分偏差 @0.49 Fs_in | +3.3927 dB | **+0.0062 dB**（v1.6/v1.7 の +0.043 は非再現・R3-4/R4-8/R5-10） | +0.0025 dB |
| ripple [0.005, 0.45] base / cand | 5.242 / 3.607 dB | 0.111 / 0.084 dB | 0.053 / 0.039 dB |
| **ripple [0.005, 0.30] base / cand**（P0-C 判定帯域） | 0.001 / 0.001 dB | 0.034 / **0.025 dB** | 0.016 / **0.012 dB** |
| 参考: ripple [0.005, 0.75×edge] cand | 0.0009 dB | 0.0358 dB | 0.0174 dB |
| candidate 0.1 dB passband edge | **0.3526 Fs_in** | **0.4897 Fs_in** | **0.4945 Fs_in** |

（S1 は v1.8 記載と全点一致。IIR3/LP3 の dev/ripple は測定密度依存のため v2.0 値を authoritative とする。
base の 0.1 dB edge は絶対値評価では意味を持たないため candidate のみ掲載）

#### 2.7.2 FIR 設計の絶対基準値（P0-D floor の出典・**本日全点独立再現 → 確定**）

own-rate cycles/sample・exact DTFT:

| design | −0.1 dB edge | transition_end（−A+3 dB） | leak 実測（本モデル） | v1.8 記載 | floor（= A−10 dB） |
|--------|-------------|---------------------------|----------------------|-----------|--------------------|
| 511/140 | 0.2448 ✓ | 0.2590 ✓ | 0.26:−141.9 / 0.30:−155.2 / 0.35:−160.0 / 0.45:−174.8 / 0.474:−165.4 | 同値 ✓ | −130 dB |
| 127/110 | 0.2317 ✓ | 0.2782 ✓ | −18.6 / −114.2 / −125.1 / −138.9 / −177.2 | 同値 ✓ | −100 dB |
| 31/90 | 0.1823 ✓ | 0.3452 ✓ | −8.45 / −25.4 / −96.2 / −107.7 / −90.6 | 同値 ✓ | −80 dB |
| 1023/160 | 0.2472 ✓ | 0.2552 ✓ | −167.5 / −179.1 / −206.4 / −186.5 / −188.4 | 同値 ✓ | −150 dB |
| 255/140 | 0.2396 ✓ | 0.2681 ✓ | −36.5 / −152.1 / −157.7 / −165.4 / −167.1 | 同値 ✓ | −130 dB |
| 63/120 | 0.2109 ✓ | 0.3130 ✓ | −10.7 / −59.3 / −130.1 / −125.6 / −138.3 | 同値 ✓ | −110 dB |

（0.26〜0.30 の低い値は transition 領域であり、P0-D の PASS 判定は transition_end 以降のみ。
v1.8「stopband min 実測 −138.9 dB 等」は遷移終端後の **最悪漏れ（max \|H\|）** 値であり、本モデルの
リーク点列と整合（最悪値は lobe 頂点のサンプリング差で ±3 dB 内）。「floor ✓」判定は全 design 成立）

#### 2.7.3 Image rejection 実測（v2.0 確定・3 定義確定・R3-1 + R4-13 + R5-1/R5-9 反映）

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
高 f̂ で悪化。**−88.9〜−104.9 dB に相当する値はどの design/f̂ にも存在しない → v1.7 §2.7.3 は廃棄**。
**R5-1**: この表は quality gate ではなく「中間 upsampled 信号の現状品質の記録」であり、final round-trip
alias rejection の保証は P0-D が担う）

**D2（単段 round-trip 出力・base==cand ≤ 0.01 dB・[0.005, 0.40] で gate 確定）**:

| design | 0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | **0.45（記録のみ）** |
|--------|------|------|------|------|------|------|------|
| 31/90 | −165.5 | −143.0 | −102.9 | −97.0 | −114.3 | −125.8 | **−137.57 (base) / −137.84 (cand)・diff 0.27 dB（R4-5/R4-12）** |
| 127/110 | −171.9 | −150.0 | −65.7 | −86.9 | −97.9 | −74.0 | −118.9 |
| 511/140 | −139.7 | −117.9 | −78.2 | −70.1 | −88.0 | −99.8 | −113.4 |
| 63/120 | −185.7 | −144.3 | −108.8 | −90.0 | −115.4 | −114.8 | −106.1 |
| 255/140 | −162.0 | −173.1 | −90.7 | −109.0 | −79.2 | −96.2 | −109.7 |
| 1023/160 | −150.2 | −108.3 | −74.8 | −51.1 | −79.4 | −77.5 | −71.1 |

（D2 = \|H_rt(0.5−f̂)\|/|H_rt(f̂)| = round-trip 伝達関数の stopband 比。**§2.3.1 構造恒等式 (a) により
passband では cand/base 比は理論上厳密に (4/3)^N → 比不変**。実測 base==cand は [0.005, 0.40] で FP 誤差内
（≤0.01 dB・1023/160 @0.05 で −150.23/−150.22 など）。0.45 は 31/90 design の transition_end=0.3452 通過点
であり **FP 量子化フロア領域**（R5-9 (c): longdouble 実測 −294.8/−296.2 dB で double フロアの安定性を確認）・
R4-13 で gate 外）

**D3（full-chain 出力・base==cand 完全一致）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30} の全点で
mirror(0.5−f̂) = **−205〜−270 dB**（実質零・passband 構造恒等式帰結）・worst spur = 測定窓 sidelobe −98〜−101 dB。
**base==cand 差 = 0.00 dB（全点）**

### 2.8 教訓・監査記録（v2.0・v1.8 §2.8 1〜15 + v1.9 16〜18 を継承し 19〜21 を追加）

1〜15: **v1.8 §2.8 のまま**（oracle 依存禁止・gain convention と FIR 構造の分離・engine fit HOLD・Phase 0 先行・案 E=仮説だが defect 客観確定・perfect reconstruction 表現注意・anti-aliasing は実測確認・compile-time flag・姉妹実装扱い・validator error contract・own-rate 軸統一・shadow 参照 fidelity gate・測定定義混用・密グリッド評価・TruePeakDetector 後続候補）。

16. **【v1.9 追加・R4-3】reset() と clearAllStages() の動作差異**: 履歴クリア範囲は完全同一だがアトミックフラグ解除数が異なる（3 個 vs 1 個）。`hardFallbackActive` 解除は `reset()` のみ。Phase 0 測定系は `reset()` 経由の初期化を必須とする
17. **【v1.9 追加・R4-4】silence パスの精密な行範囲認識**: 「silence 早期 return パス」は :583-613（コメント :583 + 本体 :584-613・入口 :594・early return :606-612）であって、:614 以降は通常パス（バウンドチェック・境界違反時 early return :643-649）
18. **【v1.9 追加・R4-13】FP 量子化拡大領域の gate 除外**: 遷移端直後（31/90 transition_end=0.3452 通過後の 0.45 など）の FP 量子化ノイズ拡大は、対象 design の仕様上予期される数値であり gate 範囲外として記録のみとする。gate は [0.005, 0.40] Fs_in に統一（全 design で ≤0.01 dB 確認）
19. **【v2.0 追加・R5-9】round-trip の構造恒等式と不変性の限定**: 単段 round-trip は `h_rt = 2·(conv⋆conv) + c·δ[n−15]`（c: base 0.25 / cand 0.5）と厳密分解される。passband では conv 経路が支配し同一位相因子を持つため cand/base 比 = (4/3)^N 厳密。一方 stopband・遷移端近傍では同位相性が崩れ center 係数差が干渉残差として顕在化し得る → **「base==cand」を無条件の理論帰結として主張してはならず、「passband で厳密・遷移端近傍で感度領域」と限定して明文化する**
20. **【v2.0 追加・R5-2】安全性 gate の非委譲**: SoftClip 局所 OS は production path であるため、NaN/Inf・corruption・unexpected clipping の有無は record-only にせず gate（I-b）とする。数値再校正のみ Phase 3-C に留める
21. **【v2.0 追加・R5-4】仮説と欠陥の分離を文書全域で強制**: `B-1 defect = confirmed（0.75^N 12 桁実測）` と `corrective E = candidate hypothesis` を常に区別し、Phase 0 全 PASS + ユーザー GO まで案 E を「確定」と表記しない

### 2.9 B-1-P0 GATE の適用要件（v1.8 §2.9 のまま・確定）

production src/ modified=0 / staged=0・measurement/harness のみ test-only 追加可・commit 禁止・calibration 禁止・threshold update 禁止・engine-fit 0.75^N（経験式）unchanged。

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（v1.8 §3.1 を継承・行番号本日再実測一致）

- 根本原因: `setAutoGainStagingEnabled`（AudioEngine.h:1416-1432）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425・本日再実測）が発火。既定 staging=true（:2626）→ `eq` モードの呼出順（`configureProbeFlatEQ` :1799 → `setAutoGainStagingEnabled(false)` :1803）で AGC が再設定される
- `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）・`getEQProcessor` と `setEQAGCEnabled` は同一 `uiEqEditor` 経由 ✓
- **修正案 A（呼出順逆・eqdiag パターン :1833/1834 に揃える）を維持**。検証手順（§3.1.3・state propagation 明示版）も維持。窓 `[0.486,0.496]`（:2042-2044）は不変前提だが再校正は Phase 3-D
- 本日実測補足: eqdiag/eqos モードは既に `setAutoGainStagingEnabled(false)` → `configureProbeFlatEQ` の順（staging 先行）で構築済み → 案 A はそのパターンへの整合（R5-3 関連・順序変更と独立）

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（production 欠陥として記録・v1.8 §3.2 を継承）

- `EQProcessor.Processing.cpp`: `bypassTransitionActive` :516 / `dryCopyBase` 充填 :570-578 / ブレンド本体 :978-993（`canBlendDry = (dryCopyBase != nullptr)` :980・dry 混合 :993）実測 ✓
- 定常 B-1 の原因ではない・遷移時のみ。**修正案 D（記録のみ・別 work item）を維持**（レビュー §23 で scope discipline 妥当と判定済み）

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化（v1.8 §3.3 を継承・R5-8 で用語確定）

- **用語定義（R5-8・新規明文化）**:
  - **`ir`** = IR ロード後に convBypass を維持する **dry ベースライン測定モード**（`e.setEqBypassRequested(true)` + `e.setConvolverBypassRequested(true)` :1744-1745 → 出力は bypass blend の dry コピー・wet 経路を通らない）
  - **`irwet<digit>`** = convBypass を **明示解除**（`e.setConvolverBypassRequested(false)` :1778）して wet 畳み込みを実際に通す **対照モード**（OS 倍率はモード末尾 1 桁: irwet1/irwet2/irwet4/irwet8・IR 固有ゲインは OS=1 との比で相殺・絶対 unity を仮定しない）
- 未知モードは fail-closed（:1700-1710・`[BUZZ] FAIL: unknown rigcheck mode` で即終了 — commit `c4a08171` の fail-closed 方式）✓
- **修正案 B（記録のみ・`irwet` で wet 対照カバー）を維持**

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役（R5-7 で判定基準を精緻化）

### 4.1 現状（本日再実測・R4-9 数値確定）

- 519 行・TEST_F（fixture `PublicationValidatorIsolationTests`）**34** + TEST（fixture `CrossfadeAuthorityRegressionTest`）**4** = 38 ケース ✓（`grep -c` 実測）
- 分類（メソッド名プレフィクス集計）: **ValidatePublication 7 / ValidateSemanticConsistency 1 / ValidateTopology 6 / ValidateResources 11 = 25**・**ValidateTransition 8 / checkNoConflictingTransitions 直接 1 = 9** ✓。全 38 TEST は fixture `PublicationValidatorIsolationTests`(34) + `CrossfadeAuthorityRegressionTest`(4)
- **CMake 未登録（add_executable 39 件のいずれにも該当なし・R4-9 本日再実測）** ✓・gtest 使用は本ファイルのみ ✓
- `validator_.checkNoConflictingTransitions` 実呼出し **9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）✓
- **コンパイル不可**（private 宣言: RuntimePublicationValidator.h:92 private:・:101 checkNoConflictingTransitions）✓
- `tools/build-debug.bat:29` = `cmake --build ... --target PublicationValidatorIsolationTests`（stale・実測）✓
- 検査順序: RuntimePublicationValidator.cpp:14-41（SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins）✓ errorMessage 固定文字列 :16/:24/:32/:40 ✓
- AudioEngine.h:3659 `validator_(&validator)` → :3670 `validator_->validatePublication(world)` ✓
- **構造化判定の現状（R5-7 本日確認）**: 公開 API `validatePublication()` は `RuntimeValidationResult` を返し、**`failureReason` フィールド（`ValidationFailureReason` enum: None/InvalidTopology/InvalidResources/InvalidTransition/SemanticInconsistency・h:13-24）** を既に提供 → 移行後の主判定に使用可

### 4.2 移行方針（v1.8 §4.2 のまま・R5-7 判定基準を付記）

Phase 0 分類 → Phase A（CrossfadeAuthority 4 ケース最優先）→ Phase B（validator 25 + CheckTransition 9）→ Phase B'（Semantic Equivalence Gate・target failure isolation 含む）→ Phase C（退役 + build-debug.bat:29 の stale 参照除去）。

**Phase B' Semantic Equivalence Gate の判定基準（R5-7・新規明文化）**:
1. **主判定**: 移植テストの結果が **構造化 `result.failureReason`** で旧 private 検査と一致する
2. **補助**: `errorMessage` 文字列一致は補助 assert（semantic equivalence の唯一の根拠にしない）
3. 退役は移植完了 + Phase B' PASS 後のみ（MIGRATE CASES THEN RETIRE・v1.8 §4 継承）

---

## 5. §5 別課題（v1.8 §5 を継承・R3-5 で R-2 の内訳確定）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | 未捕捉例外 → terminate。**変換実体 9**（直接 7: stod×4 :1582/:1592/:1614/:1615・stoi×2 :1583/:1591・stof×1 :1590 ＋ lambda 本体 2: stoi :1562/:1567）・**throw 可能経路 12**（直接 7 ＋ lambda 呼出 5: parseHcIdx :1606/:1608/:1610・parseLcIdx :1607/:1611）（R3-5）。lambda 本体 2 箇所の try/catch 化で全 12 経路を fail-closed 化 | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=`（:1613）は値を受理して破棄（設計上ステップ固定）。silent-ignore 系 | **次（極小）** |
| **B-3** | timestamp-based capture（flipIndex 誤差 <0.5%・現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合（uv tool extras=mcp のみで proxy 起動不可等） | 中 |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正） | 高 |

R-1 / R-2 の参考実装は **v1.8 §5.1 のまま**（`parseIntOrFail` を `parseIdxOrFail` に一般化し parseHcIdx/parseLcIdx の 2 lambda にも適用）。

---

## 6. 推奨する実行順序（v2.0・R5-3 によりレビュー推奨版に変更）

```
[Step 0] 現状固定・working tree 確認
  - 基準ソース固定: HEAD 8f127bfe（+ 未 push c4a08171）・production src/ 未 commit 差分 = 0 を確認
  - authoritative モデル（model_polyphase_20260920.py）の実行環境固定（本日 diff 0 行で再現確認済み）

[Step 1] §2 B-1 Phase 0 characterization（最大リスクを最速で確定・production 変更 0）
  - Phase 0-0: DESIGN-CONTRACT-A（v2.0 版 E1〜E5・E5 は補助証拠）を明示承認（G-0・ユーザー）
  - Phase 0-1: Baseline characterization（0.75^N ±1e-6・record-only）
  - Phase 0-1b: REF-FIDELITY gate（bitwise・reset() 経由初期化必須・R4-3）
  - Phase 0-2: Shadow Candidate E（DC = 1.0 ± 1e-6）
  - Phase 0-3〜0-6: 周波数（D1/D2/D3 定義確定版・D2 gate [0.005, 0.40]・R4-13/R5-9）/ block・reset / SoftClip（I-a/I-b gate + I-c record）/ float-double
  - B-1-P0 GATE（§2.6.1 v2.0 版）全 PASS を確認

[Step 2] Phase 0 review（ユーザー GO gate）
  - 全 P0 PASS レポート + P0-E を「image invariance」としての意味とともに提示
  - ユーザー GO → Phase 1 資格確定 / FAIL → R3-12 後続候補へ方針変更

[Step 3] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・v1.8 §3.1.3 の検証手順に従う）
  - F-3/F-4: 記録のみ（R5-8 の ir/irwet 定義を記録に反映）
  - 検証: AudioEngineHarness.exe --buzz-rigcheck=eq で gainpath staging=0 eqAGC=0 を確認

[Step 4] §4 F-1（テスト移行・R5-7 判定基準適用）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移植 + Phase B' semantic equivalence gate（failureReason 主判定）
  - PublicationValidatorIsolationTests.cpp 退役

[Step 5] §1 O-1〜O-12（commit 意思決定・ユーザー判断待ち）
  - 案 A 採用時はエントリ配線の最小変更を含めて commit
  - O-8（pyc 復元可否）・O-9（docs 目録扱い）・O-10（AGENTS.md MIMO Desktop 追記）/O-11（残課題台帳 B-1 詳細追加）・O-12（ConvoPeq.md 生成物扱い）を判断対象に追加（v1.9 から継承）

[Step 6] §2 B-1 Phase 1 以降（Step 2 の GO が前提）
  - Phase 1: CMake option 導入（R3-8 方式）→ Phase 2: flag ON 限定検証
  - Phase 3-A → 3-B1 → 3-B2 → 3-C → 3-D → Phase 4
  - **Phase 0 FAIL 時の後続候補**: 案 D 統合 / tap 再設計 / TruePeakDetector 型全 FIR 構造（R3-12）

[Step 7] §5 別課題（将来 work item 化）: R-2 → R-1 → B-3, D-2, D-1
```

**Step 2（Phase 0 review）のユーザー GO が確定するまで、Phase 1 以降の着手は不可**（v1.8/v1.9 と同一の原則・R5-3 で位置を前倒し）。

---

## 7. 検証計画（v1.8 §7 を継承・閉形式モデルを追加）

### 7.0 閉形式モデル（v1.8 新設・v2.0 で本日再実行・diff 0 行）

```bash
python doc/work113/model_polyphase_20260920.py
# 出力: 係数検証（FIRsum=1.0/convSum=0.5 全 6 design）→ DC round-trip → full-chain |H| →
#       per-design 絶対基準 → D1/D2/D3 image rejection
# 保存済み全出力: doc/work113/model_polyphase_20260920_results.txt
# v2.0 で 2026-09-21 環境下に再実行し model_polyphase_20260920_results.txt と diff 0 行（完全一致）を独立確認
```

### 7.1 単体検証（v1.8 §7.1 のまま）

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テスト PASS・PVIT 削除済 |
| §2 B-1 Phase 0/1 | `cmake --build build --config Release --target AudioEngineHarness && build\Release\AudioEngineHarness.exe` | `[OS_DIRECT] ... roundTripGain=`（flag OFF ≈0.75^N・flag ON ≈1.0） |
| §2 B-1 Shadow / REF-FIDELITY | Phase 0-1b/0-2 の test-only reference 実行 | REF-FIDELITY bitwise 一致 → Candidate DC = 1.0 ± 1e-6 |

**注**: `PublishPipelineIntegrationTests.exe` は存在しない（実測 ✓）。ターゲット/バイナリは AudioEngineHarness（`build\Release\AudioEngineHarness.exe` 実在再確認 ✓・**41,137,664 bytes**・R4-11・`add_executable(AudioEngineHarness)` CMakeLists.txt:1899）。`--buzz-*` フラグは AudioEngineHarness.exe に有効。

### 7.2〜7.4 統合検証 / ビルド検証 / 静的解析: **v1.8 §7.2〜§7.4 のまま**

（静的解析: cppcheck を `CustomInputOversampler.cpp` に実行 → 指摘 0 件・R4-6/R5-11。clang-tidy では portability-simd-intrinsics / bugprone-branch-clone のみ検出・R4-7。両者 design intent として許容可）

---

## 8. ロールバック計画（R5-6 で命名精緻化）

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral） | **compile-time rollback**: `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild（第一手段・production source 触らない）。**build artifact の切替を要し runtime 即時復旧ではない** |
| §2 B-1（source-level） | `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN … #endif` ブロックと in-source safety net を削除 + CMake option/add_compile_definitions を除去（rebuild 必須） |
| §3 F-2 | 1 行 revert |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

---

## 9. 影響度まとめ（v1.8 §9 のまま・O-10/O-11/O-12 を追加）

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| O-8（pyc 復元） | working tree のみ | 極小 | 任意 |
| O-10（AGENTS.md） | 環境記録更新のみ | 極小 | 任意 |
| O-11（残課題台帳） | 台帳更新のみ | 極小 | 任意 |
| O-12（ConvoPeq.md） | 生成物のため commit しない方針 | なし | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.4963 dB） | **極大**（1 行追加） | 必須（Phase 0→1→2→3→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | production 潜在欠陥（記録のみ） | なし（本セッション） | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照（行番号は 2026-09-21 本日再実測・全件一致確認・R4-4 反映）

- 前スナップショット: `doc/work113/residual_tasks_20260919.md` / 本セッション報告書: `residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 + レビュー① 38 観点）・`renew_plan_verification_20260920.md`（レビュー②）・v1.6/v1.7/v1.8 改訂サマリ・v1.9 改訂サマリ・**2026-09-21 外部レビュー監査（Phase 0 GO / Phase 1 HOLD・必須修正 2 点）→ 本書 R5-1〜R5-11 に反映**
- authoritative 計測モデル: `doc/work113/model_polyphase_20260920.py` + `model_polyphase_20260920_results.txt`（v1.8 新設・v2.0 で再実行・diff 0 行確認）

**行番号再監査結果（全件一致・2026-09-21 再実測・v1.9 §10 の R4-3/R4-4 反映版を継承）**:

| ファイル | 確認行 | 結果 |
|----------|--------|------|
| `src/CustomInputOversampler.h` | isLinearPhaseFIR/isSymmetricUpDown: 21-22 / `reset()` 宣言: 32 / `clearAllStages()` 宣言: 71 / `Stage::centerCoeff = 0.5`: **87**（v1.9 記載 :87 と一致・本日確認） | ✓ |
| `src/CustomInputOversampler.cpp` | tapsForStage/attenuationForStage 84-106 / clearStage 108-119 / prepareStage 287-390（centerTap 292・center 0.5 335/348・scale 341・convCoeffs 360-365）/ prepareSingleStage 392 / **reset** 452-467（per-ch clear 459-460・3 アトミック 464-466・R4-3）/ **clearAllStages** 469-484（per-ch clear 477-479・1 アトミック 483・R4-3）/ markCorruptionDetected 486-490 / interpolateStage 492-568（convValue ×2: **557**・denormal clamp 558-559）/ decimateStage 570-723（silence パス **583-613**（R4-4 訂正）・通常パス 615-649・n ループ 652-・acc 658/716・output 717）/ reset 452 / processUp **725**（hardFallback 透過 728-740・空ブロック返却 744-748・段ループ 764-776・currSamples <<= 1 775）/ processDown **785**（hardFallback 透過 789-806・corruption クリア 808-818・容量超過 clear 826-832・hardFallback 発火 814-815） | ✓（R4-3/R4-4 反映） |
| `src/audioengine/AudioEngine.h` | setAutoGainStagingEnabled 1416-1432（setAGCEnabled **1425**）/ 2626 staging 既定 true / 1292 getEQProcessor・1306 setEQAGCEnabled / 3659・3670 validator | ✓ |
| `AudioEngine.Processing.Latency.cpp` | static_assert 6-7 / taps 表 22-24 / groupDelaySamplesAtStageRate = taps[stage]−1 **30** | ✓ |
| `DSPCoreLifecycle.cpp`（`AudioEngine.Processing.DSPCoreLifecycle.cpp`） | softClipOS.prepareSingleStage(31, 90.0) **188 / 261**（diagLog 版/非 diag 版の 2 箇所） | ✓ |
| `DSPCoreFloat.cpp` | softClipOS.processUp **405** / processDown **413** | ✓ |
| `DSPCoreDouble.cpp` | kOutputHeadroom = 0.8912509381337456 **593** | ✓ |
| `EQProcessor.Processing.cpp` | bypassTransitionActive **516** / dryCopyBase 570-578 / blend 978-993 | ✓ |
| `RuntimePublicationValidator.cpp/.h` | 検査順序 14-41 / errorMessage 16/24/32/40 / private **92** / checkNoConflictingTransitions **101** / ValidationFailureReason enum **h:13-19**・failureReason フィールド **h:24** | ✓ |
| `TruePeakDetector.cpp`（v1.8 拡張） | interpolateStage **284-311**（履歴 shift 297-300・両位相 = center + conv **305-306**・×2 補償なし・R3-12） | ✓ |
| `BassBuzzMeasurement.cpp`（**本日 2687 行**） | configureProbeFlatEQ **1072**（内 setEQAGCEnabled 1081）/ runOversamplerDirect **1293**（upPeak 1342-1350・roundTripGain **1367**）/ singleStage C2-Q **1376-1430**（prepareSingleStage(31, 90.0) 1384・upPeak/downGain 記録 1408-1416・roundTripGain 1424）/ runEqDirectDriveAttribution **1526/1532** / parseHcIdx/parseLcIdx **1558/1564**（stoi 1562/1567）/ stod/stoi/stof 1582-1615 / flip-eqgain **1613** / F-2 eq モード呼出順 1799→1803（eqdiag パターン 1833/1834）/ `irwet` 分岐 **1763**・convBypass 解除 **1778**・ir dry 固定 **1744-1745** / eq 窓 **2042-2044** | ✓ |
| `PublishPipelineIntegrationTests.cpp` | main **1085** / 前方宣言 **1116** / 呼出 **1223**（**1340 行**・本日確認） | ✓ |
| `PublicationValidatorIsolationTests.cpp` | **519 行** / TEST_F 34 + TEST 4 / 分類 7/1/6/11 + 8/1 / 呼出 135/246/255/265/273/281/313/324/334 | ✓ |
| `CMakeLists.txt` | option 群 **40**（clang-tidy）/ **71**（MKL）/ 128-129 / NUC_DEBUG_GUARDS **52-58**（add_compile_definitions の既存前例）/ add_subdirectory(JUCE) **1043** / juce_add_gui_app **1062** / CONVOPEQ_ALL_SOURCES **1133** / target_sources **1282** / CONVOPEQ_ALL_SOURCES ループ **1892** / add_executable(AudioEngineHarness) **1899** / **add_executable 計 39 件（R4-9）** | ✓ |
| `tools/build-debug.bat` | stale target **29**（PublicationValidatorIsolationTests） | ✓ |
| `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp` | up `buf[N-1] = 2 * samples[i]` **185** / down `buf[N-1] = bufferSamples[i << 1]` **228**（両位相 ×2 / down ×1 = E5 証拠・本日一次再確認） | ✓ |
| `build\Release\AudioEngineHarness.exe` | 実在 **41,137,664 bytes**（R4-11・2026-09-20 14:25 mtime） | ✓ |
| cppcheck (`CustomInputOversampler.cpp`) | `--enable=warning,performance,portability --std=c++20` 実行 → **exit=0・指摘 0 件**（R4-6） | ✓ |
| clang-tidy (`CustomInputOversampler.cpp`) | `bugprone-*,performance-*,portability-*,misc-*` 実行 → portability-simd-intrinsics（AVX2 由来）+ bugprone-branch-clone（parity 由来）のみ・実 bug 0 件（R4-7） | ✓ |

---

## 11. ユーザー判断待ち項目（v1.8 §11 を継承・O-10/O-11/O-12 追加）

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 計装 | A: 最小実行可能資産 / B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-8 pyc | 復元（git checkout）/ 触らない / `.gitignore`+`rm --cached` を別 work item 化 | **触らない→将来 work item** |
| §1 O-9 docs 目録 | 触らない / commit | **触らない（ユーザー判断）** |
| §1 O-10 AGENTS.md | 触らない / commit（環境記録更新扱い） | **commit（環境記録更新扱い）** |
| §1 O-11 残課題台帳 | 触らない / commit（台帳更新扱い・O-1/O-2 と同梱可） | **commit（台帳更新扱い）** |
| §1 O-12 ConvoPeq.md | 触らない（生成物・O-3 方針継続） | **触らない（O-3 方針継続）** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** — **candidate hypothesis（Phase 0 全 PASS + ユーザー GO まで確定としない・R5-4）** | **E（仮説として採用）** |
| §2 DESIGN-CONTRACT-A（G-0） | 明示承認 / 修正要求 / 却下 | **明示承認（v2.0 版 E1〜E5・E3 は D1 範囲訂正 + E5 補助区分化後）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **全実施** |
| §2 Phase 3 の commit 分割 | 3-A/3-B1/3-B2/3-C/3-D 分割 / 単一 Phase 3 | **分割** |

### §2 B-1 採用案の前提条件（v2.0・P0-E は image invariance 版・P0-I は安全性 gate 版・R5-1/R5-2 反映）

1. **DESIGN-CONTRACT-A**（v2.0 版）が明示承認される（Phase 0-0・G-0）
2. **REF-FIDELITY** が bitwise 一致（Phase 0-1b・`reset()` 経由初期化必須・**R4-3**）
3. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（Phase 0-2・P0-A）— 必要条件であり十分条件ではない（レビュー §8）
4. **passband 特性が維持**: P0-B（50 Hz/1 kHz ±0.1 dB）・P0-C（[0.005, 0.30] ripple ≤ 0.05 dB）
5. **anti-aliasing / stopband が維持**: P0-D（alias leakage ≤ A−10 dB・transition_end 以降）
6. **出力 image が不変（image invariance）**: P0-E-2（D2 [0.005, 0.40] で base==cand ≤ 0.01 dB — **R4-13** gate 帯域）— **quality gate ではなく invariance gate**（R5-1）
7. **latency が不変**（impulse 実測・P0-F）
8. **SoftClip 局所 OS が正常かつ安全に動作**（P0-I: I-a DC unity / **I-b NaN・corruption・clipping 安全性 gate** / I-c 動作点記録）
9. **Differential が center-phase ×2 のみに帰属**（P0-C'・per-config 適用域）

### v2.0 最終判定表（v1.9 §11 判定表を継承・R5 反映版）

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-12 | **GO 候補** | commit/push/復元の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替） |
| §3 F-3 / F-4 | **記録継続** | production 潜在欠陥 / dry 測定基準維持 + ir/irwet 用語定義確定（R5-8） |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate（failureReason 主判定・R5-7） |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO（defect は DC round-trip = 0.75^N で客観確定・D1 補助証拠・範囲 −16.65〜+14.93 dB に訂正）** | R3-2 + R5-9 |
| **B-1 Phase 0** | **GO（測定仕様 v2.0 適用が条件）** | P0-E 再定義（image invariance）+ P0-I 3 分割（I-b 安全性 gate）+ P0-C 判定帯域統一 + dense 評価必須化 + **P0-E-2 gate 帯域 [0.005, 0.40]（R4-13）** + **reset() 経由初期化必須（R4-3）** を含む |
| B-1 Phase 1 / 2/3/4 | **HOLD** | Phase 0 全 PASS + Phase 0 review でのユーザー GO が前提（R5-3 で review を独立ステップ化） |
| **B-1 案 E** | **有力仮説として採用（candidate hypothesis）** | 仮説だが defect 自体は客観確定。FAIL 時の後続候補を R3-12 に追加。**Phase 0 完了まで「確定」と表記しない（R5-4）** |
| 「perfect reconstruction」 | **表現修正のまま** | DC gain 1.0 のみ確定 |
| B-1 calibration 変更 | **HOLD** | Phase 3-B1/B2 → 3-C/D の順 |
| `0.75^N` engine-fit 削除 | **HOLD** | Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | `#if` セマンティクス確定（R3-8） |
| TruePeakDetector 姉妹実装 | **記録継続・後続候補として格上げ記録（R3-12）** | 案 E の正解としない |
| **v1.7 §2.7.3 candidate image 契約** | **廃棄（非再現）** | R3-1 |
| **P0-E gate 帯域 [0.005, 0.40]（R4-13）** | **確定** | D2 の [0.005, 0.40] で全 design ≤0.01 dB 確認済。0.45 は FP 量子化拡大領域のため gate 外（R5-9 (c) でメカニズム数値確定） |
| **P0-E の論理意味（R5-1）** | **確定** | 「image rejection quality」ではなく「**candidate-induced image invariance**」。baseline image quality は E-1 記録として保存し、round-trip alias rejection は P0-D が保証 |
| **P0-I 安全性 gate（R5-2）** | **確定** | I-a（DC unity）/ I-b（NaN・Inf・corruption・unexpected clipping — **gate**）/ I-c（動作点 record-only）の 3 分割 |
| **実行順序（R5-3）** | **確定** | B-1 Phase 0 を最優先（Step 1）→ Phase 0 review（Step 2・ユーザー GO gate）→ F-2 → F-1 → O-1〜O-12 → Phase 1 |
| **reset() vs clearAllStages() 差異（R4-3）** | **確定** | 履歴クリア同一・アトミック解除数 3 vs 1。Phase 0 測定系は reset() 経由初期化必須 |
| **silence パス :583-613（R4-4）** | **確定** | v1.8 §12.2 「:584-629」を訂正。:614-649 は通常パス |
| **O-10/O-11/O-12** | **GO 候補** | O-10（AGENTS.md 環境記録更新・commit 可）・O-11（残課題台帳 B-1 詳細追加・commit 可）・O-12（ConvoPeq.md 生成物・commit しない） |

---

## 12. 本セッションの検証エビデンス（2026-09-21・確定事項の出所・R5 反映）

### 12.1 閉形式モデル（C++ 厳密移植・authoritative 計測定義・v2.0 で再実行）

- `doc/work113/model_polyphase_20260920.py`（v1.8 セッション新規・v2.0 で再実行）:
  `prepare_coeffs`（Kaiser sinc ＋ half-band 零化 ＋ 正規化 ＋ center=0.5 / conv=0.5）・
  `interpolate_stage`（convValue ×2・candidate は centerValue ×2）・
  `decimate_stage`（center + conv stride-2 dot）。
  **ループ実装（C++ スカラーパスと同一演算順）との等価性を 4 design で ≤2.2e-15 で自己検証済み**。
- `doc/work113/model_polyphase_20260920_results.txt`（全測定出力・保存済み）。
- 測定系: round-trip 周波数応答は impulse → zero-pad FFT（2^20）による **exact DTFT**。
  tone 測定は **無窓矩形 FFT**（測定定義を単純化: D1 は up 出力 2N 点・D2 は過渡トリム + Hann 窓後の N 点・
  D3 は過渡トリム + Hann 窓後の full-chain 出力）。
- 本モデルが確定した数値: §2.7.1/§2.7.2 全表（§2.7.2 は v1.8 と**全点一致**・§2.7.3 は v1.7 破棄 → D1/D2/D3 表に置換）。
  IIR3 @0.49 = +0.0062 dB（v1.6/v1.7 の +0.043 は非再現・**R3-4/R4-8**）。
- **v2.0（2026-09-21）で再実行し、保存済み `model_polyphase_20260920_results.txt` と diff 0 行・完全一致を独立確認**（本セッション・Python/numpy 環境）。例外なく全表一致。R4-5 の 0.27 dB（31/90 D2 @0.45）も同条件で再現。

### 12.2 新規数値検証（R5-9・2026-09-21 本セッション追加）

- **round-trip 応答の厳密構造恒等式**: 単段構成で impulse 応答を直接計算し、次の恒等式を FP フロア（残差 ≤1.4e-17）で確定:
  **`h_rt = 2·(convCoeffs ⋆ convCoeffs) + c·δ[n−15]`**（c: base = 0.25 / candidate = 0.5）
  - この恒等式により、passband では cand/base 比 = (4/3)^N が位相整合の帰結として厳密成立することを裏付け（§2.7.1 全点 dev ≤0.0002 dB と一致）
- **遷移端近傍の構造残差の分離**: 31/90 @0.45 で (4/3)^N からの全 chain |H| 実測偏差 = **+1.6350 dB**（§2.7.1 記載値と完全一致）— これは conv 経路（案 E で不変）と center 項係数差（0.25 vs 0.5）の干渉による **構造帰結**であり、測定誤差ではない
- **D2 @0.45 の 0.27 dB は FP 量子化フロア依存**: longdouble での tone 測定実測 D2@0.45: base = **−294.77 dB** / cand = **−296.19 dB**（diff ≈ 1.42 dB）に対し、double 実測は −137.57/−137.84（diff 0.27 dB）→ **測定数値が FP 精度（double vs longdouble）で大幅に移動する = 測定対象は構造信号ではなく FP フロア** → R4-5/R4-13 の「0.45 を gate 外・記録のみ」判断を数値で裏付け
- 本検証で使用ツール: Python/numpy（閉形式モデル再実行）・rg/grep/sed/awk（WSL）・serena（シンボル・参照追跡）・semble（自然言語コード検索）

### 12.3 ソース監査エビデンス（v2.0 で全行番号再実測・R4-3/R4-4 反映版を継承）

- `CustomInputOversampler.h`: 21-22（isLinearPhaseFIR/isSymmetricUpDown）・`reset()` 宣言 **:32**・`clearAllStages()` 宣言 **:71**・`Stage::centerCoeff = 0.5` **:87**
- `CustomInputOversampler.cpp`:
  - 84-106（taps/attenuation）/ 108-119（clearStage）/ 287-390（prepareStage・centerTap :292・center 0.5 :335/:348・scale :341・convCoeffs :360-365）
  - 392（prepareSingleStage）
  - **reset** **:452-467**（numStages ループ × per-ch `FloatVectorOperations::clear`（upHistory :459・downHistory :460）・3 アトミック解除 :464-466 = corruptionDetected/consecutiveCorruptionAutoClearCount/hardFallbackActive・**R4-3**）
  - **clearAllStages** **:469-484**（numStages ループ × per-ch clear（upHistory :477・downHistory :479）・1 アトミック解除 :483 = corruptionDetected のみ・**R4-3**）
  - markCorruptionDetected :486-490
  - 492-568（interpolateStage・convValue ×2 **:557**・denormal clamp :558-559・corruption 境界 :509-554）
  - 570-723（decimateStage・**silence パス :583-613（コメント :583 + 本体 :584-613・入口 :594・early return :606-612・R4-4 訂正）**・通常パス :615-649 / :651-・n ループ :652-・output :717）
  - processUp **:725**（hardFallback 透過 :728-740・空ブロック返却 :744-748・段ループ :764-776・currSamples <<= 1 :775）
  - processDown **:785**（hardFallback 透過 :789-806・corruption クリア :808-818・hardFallback 発火 :814-815・容量超過 clear :826-832）
- `AudioEngine.h`: setAutoGainStagingEnabled 1416-1432（setAGCEnabled **:1425**）/ getEQProcessor **:1292**・setEQAGCEnabled **:1306**（同一 uiEqEditor 経由）/ autoGainStagingEnabled 既定 true **:2626** / validator 初期化 :3659・呼出 :3670
- `AudioEngine.Processing.Latency.cpp`: static_assert **:6-7** / taps 表 :22-24 / groupDelay **:30**。R3-11: 案 E は taps・係数列不変 → static_assert 成立持続
- `AudioEngine.Processing.DSPCoreLifecycle.cpp`: softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock) **:188/:261**（diagLog 版/非 diag 版・同一関数経路）
- `AudioEngine.Processing.DSPCoreFloat.cpp`: float→double 変換 :252-263 / softClipOS.processUp **:405** / processDown **:413**
- `AudioEngine.Processing.DSPCoreDouble.cpp`: kOutputHeadroom = 0.8912509381337456（**593**）
- `EQProcessor.Processing.cpp`: bypassTransitionActive **:516** / dryCopyBase 充填 **:570-578** / blend 本体 **:978-993**（canBlendDry :980・混合 :993）
- `RuntimePublicationValidator.cpp/.h`: 検査順序 **:14-41** / errorMessage **:16/:24/:32/:40** / `private:` **:92** / checkNoConflictingTransitions **:101** / **ValidationFailureReason enum h:13・failureReason フィールド h:24**
- `TruePeakDetector.cpp`: interpolateStage **:284-311**（履歴 shift :297-300・両位相 = center + conv **:305-306**・×2 補償なし・R3-12）
- `BassBuzzMeasurement.cpp`（本日 **2687 行**）: §10 表の行番号全件 + F-2 呼出順 **:1799→:1803**（eqdiag パターン :1833/:1834）/ `irwet` 分岐 **:1763**・convBypass 解除 **:1778** / ir dry 固定 **:1744-1745** / singleStage C2-Q **:1376-1430** / eq 窓 **:2042-2044**
- `PublishPipelineIntegrationTests.cpp`（本日 **1340 行**）: main **:1085** / 前方宣言 **:1116** / 呼出 **:1223**
- `PublicationValidatorIsolationTests.cpp`: **519 行**・TEST_F 34（fixture PublicationValidatorIsolationTests）+ TEST 4（fixture CrossfadeAuthorityRegressionTest）
  分類（メソッド名）: ValidatePublication 7 + ValidateSemanticConsistency 1 + ValidateTopology 6 + ValidateResources 11 + ValidateTransition 8 + checkNoConflictingTransitions 直接 1 = **34**（残り 4 は CrossfadeAuthority 回帰）。呼出 9 箇所実測
- `CMakeLists.txt`: option 群 **:40**（clang-tidy）/ **:71**（MKL）/ :128-129 / NUC_DEBUG_GUARDS **:52-58**（`add_compile_definitions` の既存前例）/ add_subdirectory(JUCE) **:1043** / juce_add_gui_app **:1062** / CONVOPEQ_ALL_SOURCES **:1133** / target_sources **:1282** / CONVOPEQ_ALL_SOURCES ループ **:1892** / add_executable(AudioEngineHarness) **:1899**。**add_executable 計 39 件（R4-9）**
- `tools/build-debug.bat`: stale target **:29**。`build\Release\AudioEngineHarness.exe` 実在（**41,137,664 bytes**・R4-11）
- `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp`: up `buf[N-1] = 2 * samples[i]` **:185** / down `buf[N-1] = bufferSamples[i << 1]` **:228**（E5 証拠・本日一次再確認）
- `tools/__pycache__/`: tracked 2 件（`apply-solidlsp-bash-ls-patch.cpython-314.pyc` Modified Bin 16645→16895 + `retire_authority_verifier.cpython-314.pyc`）— O-8（v1.8 §1.8・v2.0 継承）
- `docs/tool-inventory-2026-09-20.md`: untracked・**3102 bytes** — O-9（v1.8 §1.9・v2.0 継承）
- `AGENTS.md`: Modified（+8/−2・MIMO Desktop 追記+CRLF 警告）— O-10（v1.9 新設・R4-1）
- `doc/work113/residual_tasks_20260919.md`: Modified（+72/−0・B-1 詳細追加）— O-11（v1.9 新設・R4-2）
- `ConvoPeq.md`: Modified（+75/−0・生成物）— O-12（v1.9 新設）

### 12.4 要確認事項（Phase 0 で確定させる・gate 外）

1. **IIR3 の (4/3)^N 差分偏差 @0.49 Fs_in**: 本モデル +0.0062 dB vs v1.6/v1.7 +0.043 dB（S1・LP3・per-design 表は一致のため IIR3 の 0.49 近傍評価に限定的・R3-4/R4-8/R5-10）。
   Phase 0-3 [A] 実測で確定。gate 外（記録のみ）のため Phase 0 GO/HOLD に影響しない
2. **P0-C の dense 評価閾値実現性**: 本モデルでは candidate [0.005, 0.30] ripple = S1 0.001 / IIR3 0.025 / LP3 0.012 dB（gate 0.05 dB に対し余裕あり）→ Phase 0-3 [D] 実測で確定
3. **静的解析の silence パス確認記録**: cppcheck（指摘 0・R4-6）は silence 早期 return パス（**583-613**・R4-4 訂正）の「成立条件の説明」に言及していない。
   Phase 0 実測で inputSilent 偽陽性がないこと（推移帯信号での誤検出 0）を corruption フラグで確認する
4. **【v2.0 追記・R5-9】0.45 の FP 量子化拡大のメカニズム**: double で 31/90 D2 @0.45 の cand vs base 差 0.27 dB が、longdouble では base −294.77 / cand −296.19 dB（diff ≈ 1.42 dB）に移動 → 測定値が FP 精度フロア支配であることを確認。Phase 0 実測で同傾向を確認したら測定系の FP 精度依存として記録（R4-5 の判断を維持・gate 外）

---

*本書は v1.9（remediation_plan_20260920_v1.9_revised.md）の再改訂版 v2.0 である。
v1.9/v1.8 の本文は参照文書として維持し、本書の R5-1〜R5-11・§1.13・§2.3.1（恒等式追加・E3 範囲訂正）・§2.6/§2.6.1（P0-E image invariance・P0-I 安全性 gate）・§2.7.3（D2 gate 帯域のメカニズム補足）・§2.8（教訓 19〜21 追加）・§6（実行順序変更）・§8（compile-time rollback 命名）・§10/§12 の実測値が v1.8/v1.9 と矛盾する箇所では**本書を優先する**。*

**v2.0 で新規追加した監査・確定事項**:
- **R5-1** P0-E を image invariance として再定義（E-1 記録 / E-2 gate / E-3 P0-D 責務分離を明文化・レビュー必須修正①）
- **R5-2** P0-I を I-a（DC unity）/ I-b（NaN・Inf・corruption・unexpected clipping — **gate**）/ I-c（動作点 record-only）の 3 分割（レビュー必須修正②）
- **R5-3** 実行順序をレビュー推奨版に変更（B-1 Phase 0 → review → F-2 → F-1 → O-* → Phase 1）
- **R5-4** 案 E = candidate hypothesis の表記分離を文書全域で強制（「案 E の実装（R3-9 で確定）」表現を訂正）
- **R5-5** E5（JUCE）を「実装慣行の傍証（補助）」に区分化 + 一次再確認（juce_Oversampling.cpp:185/:228）
- **R5-6** ロールバック §2 behavioral 行を「compile-time rollback（runtime 即時復旧ではない）」と命名
- **R5-7** F-1 の Phase B' gate を構造化 `failureReason` 主判定に確定（errorMessage は補助）
- **R5-8** F-4 の `ir`/`irwet` 用語を正式定義（dry ベースライン / wet 対照モード）
- **R5-9** round-trip 厳密構造恒等式 `h_rt = 2·(conv⋆conv) + c·δ[n−15]` の新規数値確定（残差 ≤1.4e-17）+ 0.45 遷移端偏差 +1.635 dB の構造帰結 + D2@0.45 の FP フロア依存の longdouble 実証
- **R5-10/R5-11** IIR3 @0.49 = +0.0062 dB 再確認・P0-C' 適用域確定
- **§2.3.1 E3 範囲訂正** −9.7〜+5.8 dB → −16.65〜+14.93 dB（§2.7.3 D1 表実測の全範囲）
- **§2.8.19〜21** 教訓 3 件追加（不変性の構造的限定・安全性 gate・仮説/欠陥分離）
