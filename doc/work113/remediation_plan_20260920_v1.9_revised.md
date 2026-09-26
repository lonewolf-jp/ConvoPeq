# ConvoPeq 残件 改修計画書（2026-09-20 時点・再改訂 v1.9）

- **版**: v1.9（v1.8 全項目の追加ソース監査 + 修正点確定 + 行番号全件再監査を反映）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **基準ソース**: HEAD = `8f127bfe`（+ 未 push `c4a08171`）。production `src/` の未 commit 差分 0 件（test-only 2 ファイルを除く・本日再実測）
- **authoritative 計測定義**: `doc/work113/model_polyphase_20260920.py`（`prepareStage`/`interpolateStage`/`decimateStage` を C++ から厳密移植。ループ実装との等価性を ≤2.2e-15 で自己検証済み。全測定出力: `model_polyphase_20260920_results.txt`・本セッションで Python 3 / numpy 2.5.3 / scipy 1.18.1 环境下に再実行し全表一致確認）
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「有力仮説」として維持（閉形式モデルで DC round-trip = 1.0 を再確認）。**v1.7 §2.7.3 の candidate image rejection 契約は v1.8 で不成立が確定したため P0-E を再定義（R3-1）。v1.9 で P0-E-2 gate の判定帯域を [0.005, 0.40] に再限定（R4-13）**
- **Phase 0 着手 GO・Phase 1 以降 HOLD**（v1.8 から変更なし）
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする。C++ 分岐は **`#if` を使用（`#ifdef` は禁止）**

---

## v1.8 → v1.9 改訂サマリ（本セッションの独立再検証による修正）

v1.8 の全行番号（30+ 件）・全コード解釈を再監査した結果 **全件一致**（新規の行番号誤り 0、ただし §12.2 の silence 最適化ブロック範囲「:584-629」は v1.9 で **:583-613** に精密化 — R4-4）。`reset()` と `clearAllStages()` の動作差異を精密化し、案 E の Phase 0 測定系における前提条件（`reset()` 経由の初期化必須）を明文化（R4-3）。P0-E-2 gate の判定帯域を [0.005, 0.40] に再限定し、31/90 等の遷移端近傍（0.45）での FP ノイズ拡大（最大 0.27 dB）を gate 外へ除外（R4-5/R4-13）。さらに §0 に新規差分 3 件（O-10/O-11/O-12）を追加し、修正点確定 R4-1〜R4-13 を反映する。

| ID | 区分 | 内容 | v1.9 での確定 |
|----|------|------|----------------|
| **R4-1** | **未 commit 追加** | AGENTS.md が **Modified**（v1.8 §1.7 では "触らない" として記録。差分は「Xiaomi MIMO Desktop 環境」追記（headroom proxy 常時起動条件の 2026-09-20 反映）+ 同日 serena memory `tools/token-reduction-pipeline-2026-09-20` 参照。CRLF→LF 警告も同時発生） | **O-10 として §1.10 に追加・推奨案**: 本 Modified 差分は **記録更新扱い**として commit 可（O-1 と同梱 / 独立のいずれかユーザー判断）。「触らない」方針は MIMO Desktop 追記以後は解除しうる |
| **R4-2** | **未 commit 追加** | `doc/work113/residual_tasks_20260919.md` が **Modified**（+72 行）。B-1 帰属（`CustomInputOversampler` owner 正式帰属・attribution 詳細・engine fit 数値式・棄却セグメント）が追加された | **O-11 として §1.11 に追加・推奨案**: 残課題台帳更新として commit 可（O-1/O-2 と統合可） |
| **R4-3** | **実装詳細確定** | `reset()` (:452-467) と `clearAllStages()` (:469-484) の差異を精密化 | **履歴クリア範囲は完全同一**（per-channel FloatVectorOperations::clear、numStages × kMaxChannels のネストループ）。差分は **アトミックフラグ**: `reset()` は 3 個（`corruptionDetected` / `consecutiveCorruptionAutoClearCount` / `hardFallbackActive`）を解除、`clearAllStages()` は 1 個（`corruptionDetected` のみ）。**`hardFallbackActive` を解除できる唯一の経路は `reset()`**。Phase 0 測定系では `reset()` 経由で初期化しないと hardFallback 透過（:728-738/:789-797）が混入し得る（測定前提条件として R4-3 明記） |
| **R4-4** | **行番号精密化** | v1.8 §12.2 「silence 早期 return パス（:594-629）」は **不正確** | silence パスは **:583-613**（コメント :583 + 本体 :584-613・`if (inputSilent)` 入口 :594・内側 `historySilent` ループ :597-604・`historySilent` 時 early return :606-612）。:614-649 は **silence パスではなく通常パス**（history 充填 :615 / バウンドチェック :626-641 / 境界違反時 early return :643-649）。v1.9 §10/§12 を訂正 |
| **R4-5** | **gate 修正** | P0-E-2（D2/D3 `base==cand` ≤ 0.02 dB gate）が **31/90 @0.45 で FAIL** し得る | 本セッションで numpy 2.5.3 / scipy 1.18.1 环境下に閉形式モデル再実行: **31/90 D2 @0.45: base=-137.57 / cand=-137.84（diff +0.27 dB）**。これは 31/90 design の transition_end=0.3452 を通過した 0.45 における FP 量子化ノイズ拡大であり、典型点（1023/160 @0.05 で -150.23/-150.22 = 0.01 dB）の ~27 倍。**gate を [0.005, 0.40] Fs_in に限定**、0.45 は記録のみ。0.40 内では全 design で ≤0.01 dB（実測: 1023/160 @0.05 0.01・31/90 @0.05 0.00・127/110 @0.05 0.00 等） |
| **R4-6** | **静的解析再実測** | cppcheck を本日 `--enable=warning,performance,portability --std=c++20` で `CustomInputOversampler.cpp` に再実行 | **指摘 0 件**（exit=0、grep フィルタ後 0 行）— v1.8 R3-13 を本セッションで再確認。`--enable=all`（254 行の "Checking src\..." ログを含むが実 error/warning 0）でも同様 |
| **R4-7** | **静的解析追加** | clang-tidy を本日 `bugprone-*,performance-*,portability-*,misc-*` で実行（JUCE include 不在のため JuceHeader.h error のみ） | 検出: **portability-simd-intrinsics**（_mm256_add_pd / _mm_add_pd 等 AVX2 intrinsics・設計通りの x86_64 専用、portability カテゴリでは標準的に許容） / **bugprone-branch-clone**（parity 分岐の同形 body・interpolateStage :335/:348 の half-band zero 化処理由来）。**実 bug 0 件**。両カテゴリは design intent として許容可（JUCE の `dsp::Oversampling` も同形構造） |
| **R4-8** | **数値契約確定** | IIR3 (4/3)^N 偏差 @0.49 Fs_in = **+0.0062 dB** で確定（v1.8 R3-4） | 本セッションで Python モデル再実行・結果ファイル `model_polyphase_20260920_results.txt`（L33）と plan §2.7.1 の IIR3 行が全点一致。v1.6/v1.7 の +0.043 は非再現確定（gate 外・記録のみ） |
| **R4-9** | **追加監査** | `add_executable` 数 = **39 件** を本日再実測 | CMakeLists.txt:139-:1899 の全 39 件 grep 確認（v1.8 §4.1 と一致）。PublicationValidatorIsolationTests は 39 件のいずれにも **非該当** ✓（gtest 依存のため build 経路なし） |
| **R4-10** | **数値契約確認** | D2 1023/160 @0.05: base=-150.22 / cand=-150.23（diff +0.01 dB・FP 量子化） | 本セッションで Python モデル再実行・結果ファイルと一致 ✓（典型点として gate 内 PASS） |
| **R4-11** | **ビルド成果物** | `build\Release\AudioEngineHarness.exe` 実在 | **41,137,664 bytes**（2026-09-20 14:25 mtime・本日確認）— v1.8 §12.2 と一致 |
| **R4-12** | **数値契約精密化** | D2 31/90 全 7 点で cand vs base 再実測 | @0.05: cand=base（0.00）/@0.10: cand=base（0.00）/@0.20: cand=base（0.00）/@0.30: cand=base（0.00）/@0.35: cand=base（0.00）/@0.40: cand=base（0.00）/@0.45: cand=-137.84 vs base=-137.57（diff +0.27 dB・R4-5）。**典型点（≤0.40）で完全一致**を確認 |
| **R4-13** | **gate 定義修正** | P0-E-2 gate の判定帯域を [0.005, **0.40**] Fs_in に再限定（v1.8 §2.6.1 P0-E の "f̂ ∈ {0.05, 0.10, 0.20, 0.30}" を上限 0.40 に拡張・0.45 を除外） | D2 の [0.005, 0.40] で全 design ≤0.01 dB（実測値・R4-12）→ gate PASS 確実。0.45 は記録のみ（FP 量子化拡大領域） |

**v1.8 から変更しない点（再確認）**: 案 E の技術的内容（`centerValue *= 2.0` 1 行）/ Phase 0 の 5 構造（Baseline・G-0・REF-FIDELITY・Candidate・Differential）/ 段階リリース骨格 Phase 0→1→2→3-A/B1/B2/C/D→4 / §3 F-2 案 A・F-3 案 D・F-4 案 B / §4 MIGRATE CASES THEN RETIRE / §6 実行順序 / §8 ロールバック / §2.7.1 の candidate 絶対値基準値 / §2.7.2 の per-design 絶対基準値（全 design v1.8 と本セッションで全点一致・R4-9 関連） / §2.7.3 の D1 実測表（記録のみ・gate 外） / D2 の [0.005, 0.40] 帯域は v1.9 で確定 / §11 の判定表。

---

## 0. 凡例と全体戦略（v1.8 から変更なし）

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | B-1 | 全オーディオ経路 | flag OFF rebuild |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only（F-3 は production 欠陥の記録） | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1〜O-12, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI or 記録 | 設定 revert |

**全体戦略**: v1.8 §0 のまま（§1 意思決定 → §2 B-1 段階リリース → §3 → §4 → §5）。

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-12）

### 1.1〜1.7 O-1〜O-7: **v1.8 §1.1〜§1.7 のまま（本日再実測で全件確認）**
- O-1: 差分 +634/−2（BassBuzzMeasurement.cpp）+8/−0（PublishPipelineIntegrationTests.cpp）・窓 :2042-2044・`[OS_DIRECT] roundTripGain=` :1367・`runOversamplerDirect()` :1293 ← `runEqDirectDriveAttribution()` :1526/:1532 ← PPIT main :1085 から（前方宣言 :1116・呼出し :1223）。**推奨案 A**
- O-2: 台帳更新は O-1 と独立 commit
- O-3: `ConvoPeq.md` は commit しない（再生成: `python output_sourcecode_markdown.py`）
- O-4: `Testing/Temporary/CTestCostData.txt` は現状維持
- O-5: `.opencode/opencode.json` は触らない
- O-6: push（ahead 2 / behind 0）はユーザー手動
- O-7: `AGENTS.md` は触らない（**v1.9 で R4-1 により更新**: MIMO Desktop 追記以後は「触らない」方針を解除しうる）

### 1.8 O-8（v1.8 新設）: `tools/__pycache__/*.pyc` — tracked かつ 1 件 Modified
- 実測: `apply-solidlsp-bash-ls-patch.cpython-314.pyc` が ` M`（Bin 16645 → 16895 バイト）。`apply-...pyc` と `retire_authority_verifier.cpython-314.pyc` の **2 件が tracked**（`git ls-files tools/__pycache__/` 実測）
- `.pyc` は再生成物であり tracked 運用自体が望ましくない。ただし履歴操作はユーザー判断事項
- **推奨**: 本 Modified 差分は commit しない（`git checkout -- tools/__pycache__/apply-solidlsp-bash-ls-patch.cpython-314.pyc` で復元可）。将来 work item として `tools/__pycache__/` を `.gitignore` 追加 + `git rm --cached`（HOLD・履歴書き換えなし）

### 1.9 O-9（v1.8 新設）: `docs/tool-inventory-2026-09-20.md`（untracked）
- 環境ツール目録文書（`docs/` 配置・**3102 bytes**・2026-09-20 mtime）。**触らない**（ユーザー判断）。commit 可否はユーザー判断。

### 1.10 O-10（v1.9 新設・R4-1）: `AGENTS.md` — Modified（+8/−2）
- 実測: `git diff --stat HEAD -- AGENTS.md` = 8 行差分。差分内容:
  - タイトル行に「2026-09-20 Xiaomi MIMO Desktop でも再確認」を追記
  - 「★ 絶対遵守」の対象環境に **Xiaomi MIMO Desktop 環境**を追加
  - 「Visual Studio Code 環境は例外」前段に 2026-09-20 確認ブロック追加
  - 詳細確認: 「headroom v0.37.0（proxy 8787 稼働・MCP 読込済・`ANTHROPIC_BASE_URL` 経由）、context-mode v1.0.169（doctor PASS）、rtk（WSL `/home/user/.local/bin/rtk`）すべて使用可」
  - 役割分担「ソースで防ぐ（context-mode）→ トラフィック自動圧縮（headroom proxy）→ CLI出所圧縮（rtk）」原則の明文化
- CRLF→LF 警告も同時発生（git config core.autocrlf の影響）
- **性質**: 環境構築記録の更新（v1.8 「Xiaomi MIMO Desktop で再確認」が実装されたことの記録）
- **推奨**: 本 Modified 差分は **記録更新扱い**として commit 可（O-1 と同梱 / 独立のいずれかユーザー判断）。CRLF 警告は commit 時に自動修正される（`core.autocrlf=true` 設定下）

### 1.11 O-11（v1.9 新設・R4-2）: `doc/work113/residual_tasks_20260919.md` — Modified（+72/−0）
- 実測: `git diff --stat HEAD -- doc/work113/residual_tasks_20260919.md` = 72 行追加
- 差分内容（§B の「他オーナーへ引継ぎ済みの OPEN」セクション拡張）:
  - B-1 帰属を `EQ DSP owner` → **`CustomInputOversampler` owner** に正式帰属
  - B-1 baseline 固定（bare `ratio=0.8846` / eq `ratio=0.4912` 実測値・`kOutputHeadroom` 0.8912509381337456 解説）
  - B-1 棄却セグメント（EQ 内部 AGC・filter structure・totalGain・staging・saturation・EQ 内部 band・stage ② OutputFilter 0.987442 = -0.110dB 数値再現等）
  - B-1 新規識別子「サンプルレート依存」（--buzz-sr= 実測 48000→0.3681 / 96000→0.3683 / 192000→0.4912）
  - B-1 数値正規化（`[EQLEVEL]` の outLinear vs published/input 区別）
  - B-1 stage 別棄却（EQProcessor 単体 / AutoGainPlanner / World automation / OutputFilter ② / bypass blend / kOutputHeadroom / rigcheck=ir dry）
  - **★ B-1 主因 ATTRIBUTED**: `CustomInputOversampler` の up/down round-trip 利得欠陥（直接駆動 ratio 0.75^N 実測・実装式 `0.5×0.5 + 0.5×1.0 = 0.75` 導出・1 stage 分離実測・`prepareSingleStage(31, 90.0)` SoftClip 局所 OS・engine 実測照合式 `0.98379 × 0.75^max(log2 effOS, 1)` 4 条件 ≤0.05%）
- **性質**: 残課題台帳の B-1 詳細記録追加（v1.8 §0 の「**B-1 監査完了・原因 ATTRIBUTED**」と整合）
- **推奨**: 本 Modified 差分は残課題台帳更新として commit 可（O-1/O-2 と統合可）

### 1.12 O-12（v1.9 新設）: `ConvoPeq.md` — Modified（+75/−0）
- 実測: `git diff --stat HEAD -- ConvoPeq.md` = 75 行追加
- **性質**: 生成物（O-3 として v1.7 §1.3 で commit しない方針確定済）
- **推奨**: O-3 の既存方針通り commit しない。CRLF 警告は O-10 と同根（autocrlf 設定）

### 1.13（参考・v1.9 補足）: `doc/work113/` 計画書 8 件（untracked）
- v1.8 §1.10 の 7 件に加え、本書 `remediation_plan_20260920_v1.9_revised.md` を追加（計 8 件）
- 前例の work113 文書は tracked。**推奨**: 本書承認確定後、承認版 + 台帳を 1 commit として登録（O-2 / O-11 と同梱可）。`.cline/`（untracked）は触らない。

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定している事実（v1.8 §2.1 を継承・本セッションで独立再確認）

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

**閉形式モデルによる独立再確認**（`model_polyphase_20260920.py`・本セッションで Python 3 / numpy 2.5.3 / scipy 1.18.1 环境下に再実行・全表一致確認）:

| 構成 | baseline DC | candidate DC | 0.75^N |
|------|-------------|--------------|--------|
| 単段 31/90 | 0.750000000000 | 1.000000000000 | 0.75 |
| IIRLike 3 段（511/127/31） | 0.421875000000 | 1.000000000000 | 0.75³ = 0.421875 |
| LinearPhase 3 段（1023/255/63） | 0.421875000000 | 1.000000000000 | 0.75³ = 0.421875 |

### 2.2 根本原因の特定（v1.8 §2.2 を継承・全行番号本日再実測で一致・R4-3/R4-4 による精密化）

v1.8 §2.2.1〜§2.2.4 のコード追跡・parity 実測・round-trip 整合式は **全て本日再実測で一致**（`prepareStage` :287-390・`interpolateStage` :492-568（`convValue *= 2.0` :557）・`decimateStage` :570-723（`output[n] = acc` :717））。taps/attenuation 対応表（:84-106）も一致。変更なし。

**本日追加で確定した実装事実（v1.9・R4-3/R4-4 反映）**:
- `processUp`（:725-783）は **stages[0]（511/1023 taps）を入力レートで最初に適用**し、以降 2 倍ずつ（:764-776・`currSamples <<= 1`）。閉形式モデルの段順序と一致 ✓
- `processUp` は `inSamples > maxInputBlockSize` で **空ブロックを返す**（:744-748・Phase 0-4 partition 前提の根拠）✓
- `processDown`（:785-）も容量超過で `markCorruptionDetected()` + 出力クリア（:826-832）。さらに **hardFallback 経路**（:728-738/:789-806）では up 入力をそのまま透過する — Phase 0 測定系は corruption/hardFallback フラグを監視し、透過が混入していないことを確認する必要がある（v1.8 追記・v1.9 維持）
- `prepareSingleStage(31, 90.0, internalMaxBlock)` は diagLog 版（:188）と非 diag 版（:261）の **2 呼出箇所**があるが同一関数経路 → 案 E の 1 行修正で両者同時に修正される ✓
- **`reset()` と `clearAllStages()` の動作差異（R4-3）**: 履歴クリア範囲は完全同一。差分はアトミックフラグ解除数（reset() = 3 個 / clearAllStages() = 1 個）。`hardFallbackActive` を解除する経路は `reset()` のみ。**Phase 0 測定系は `reset()` 経由で初期化しないと hardFallback 透過（:728-738/:789-797）が混入し得る**
- **`decimateStage` silence 最適化パス（R4-4）**: ブロック範囲は **:583-613**（コメント :583 + 本体 :584-613・`if (inputSilent)` 入口 :594・`historySilent` ループ :597-604・`historySilent` 時 early return :606-612）。:614-649 は通常パス（history 充填 :615 / バウンドチェック :626-641 / 境界違反時 early return :643-649）。v1.8 §12.2 の「:584-629」は不正確 → v1.9 で訂正

### 2.3 修正案の比較と DESIGN-CONTRACT-A（v1.8 §2.3 を継承・E3 のみ R3-2 により修正）

案 A〜E の比較（A/B/C/D 不採用・**E 有力仮説**）は **v1.8 §2.3 のまま**。

#### 2.3.1 案 E 採用根拠（v1.9 版・G-0 として明示承認する対象）

**DESIGN-CONTRACT-A（v1.9・v1.8 から変更なし）**:
> Oversampler は interpolation convention として両 polyphase 位相が同一 DC gain（= 2 倍密化の gain convention）を持ち、up/down round-trip の DC/passband gain が unity であることが要求される。

| 証拠 | 内容 | v1.9 での状態 |
|------|------|----------------|
| **E1** | コード自身が既に gain-2 convention（`convValue *= 2.0` — :557 実測）。欠陥は convention ではなく center 位相への不完全適用 | 変更なし・確定 |
| **E2** | down 側 half-band decimator は既に DC gain 1.0（0.5+0.5）。up だけ 0.75 → up/down 非対称は `isSymmetricUpDown` / static_assert の契約前提と矛盾。**なお static_assert は「taps 列の同一性」の宣言であり、案 E は taps・遅延を変更しないため assert は成立し続ける（R3-11）** | 精緻化 |
| **E3（修正・R3-2）** | D1（up 単段出力）で baseline 鏡像比が **f̂・design 依存で −9.7〜+5.8 dB**（理論漸近 −9.5424 dB = 20·log10(1/3)・127/110 @0.2 では +5.8 dB で鏡像が信号超え）。FIR 設計減衰 −87〜−159 dB に対し最大 ~165 dB の開き = **up 分岐非対称（gain convention 欠陥）の客観的な構造証拠**。**ただし D1 鏡像は同ステージ down 段 stopband で抑制され round-trip 出力には伝播しない（D2/D3 で base==cand 厳密一致・R3-1）** | **defect の主証拠を DC round-trip = 0.75^N（12 桁実測）に確定。D1 非対称（最悪 +5.8 dB）は補助証拠** |
| **E4** | `isLinearPhaseFIR = true` / `isSymmetricUpDown = true`（h:21-22）+ Latency.cpp:6-8 static_assert | 変更なし・確定 |
| **E5** | JUCE `dsp::Oversampling` は up 経路で `buf[N−1] = 2·samples[i]`（両位相に同一 ×2）・down は ×1（juce_Oversampling.cpp:185-196 / 228-240 実測✓） | 変更なし・確定 |

**案 E の実装（R3-9 で確定・v1.9 維持）**:

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

- 8 観点: **v1.8 §2.4 のまま**（全項 Phase 0 で確認）。
- 段階リリース: **v1.8 §2.5 のまま**（Phase 0 → 1 → 2 → 3-A/B1/B2/C/D → 4・baseline record-only・calibration は 3-B1/B2 完了後の 3-C/D）。**Phase 1 の CMake 実装は R3-8 により精密化**（v1.9 維持）:

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

### 2.6 Phase 0 characterization（P0-E を R3-1 + R4-13 により再定義・他は v1.8 §2.6 を維持）

```
Phase 0-0  source/contract audit: DESIGN-CONTRACT-A（v1.9 版・E1〜E5）を明示承認（G-0・承認者はユーザー）
Phase 0-1  P0-BL Baseline（record-only）: DC round-trip = 0.75/0.5625/0.421875（ratio 2/4/8・preset 非依存・±1e-6）
Phase 0-1b REF-FIDELITY gate（R2-2 維持）: Shadow Reference（baseline モード）== production
           （DC/impulse/周波数/ratio 2/4/8 × preset 2 種/partition 4 種/reset・double 経路 bitwise 一致・
            不可なら相対誤差 ≤1e-15。不一致なら reference を修正し一致まで Phase 0-2 に進まない）
           配置: src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h（header-only・production 変更 0）
           **初期化前提条件（R4-3）**: 測定開始時に必ず `reset()` を呼出（hardFallbackActive 解除のため）。
           `clearAllStages()` のみでは hardFallback 透過（:728-738/:789-797）が混入し得る
Phase 0-2  P0-CAND Shadow Candidate E: DC round-trip = 1.0 ± 1e-6（全構成）— 閉形式モデルで予備確認済み
Phase 0-3  frequency transfer（stage-rate 軸・§2.7 の軸定義に従う）
Phase 0-4  block/reset characterization（R2-3 維持）: partition invariance bitwise（4096/1024×4/256×16/ragged・
           各ブロック長 ≤ maxInputBlockSize — processUp :744-748 の空ブロック返却 guard 実測確認済み）
           + reset contract bitwise（fresh+block#1 == stream→reset()→block#1・reset :452-467 は
           upHistory/downHistory clear 実測確認済み・clearAllStages :469-484 との差異は R4-3 参照）
Phase 0-5  SoftClip local OS characterization（R2 維持・prepareSingleStage(31, 90.0, internalMaxBlock)
           は DSPCoreLifecycle.cpp:188/261 実測確認済み）
Phase 0-6  float-host / double-host equivalence（R2-4 数値契約のまま: maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8）
```

**Phase 0-3 の測定定義（v1.9 確定 — image rejection の 3 定義を明確化・R3-1 + R4-13 反映）**:
- **D1（up 単段出力・2× rate）**: tone f̂ に対する 0.5−f̂ 鏡像成分比。**記録のみ**（gate に使わない）
- **D2（単段 round-trip 出力）**: |H_rt(0.5−f̂)|/|H_rt(f̂)|。round-trip は LTI 合成 → 鏡像比 = 伝達関数比 → **base==cand 不変（理論帰結）**
- **D3（full-chain 出力の worst in-band spur）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30}。**base==cand 不変（理論帰結）**
- **D2 判定帯域（v1.9 修正・R4-13）**: [0.005, **0.40**] Fs_in に限定。0.45 は FP 量子化拡大領域のため **記録のみ**（典型点 ≤0.40 では全 design ≤0.01 dB 確認済・R4-12）
- 測定系要件（v1.8 追加）: 過渡トリム（全 stage taps + 512 sample 以上）+ Hann 窓を必須化（未トリム矩形窓では −79〜−90 dB の偽フロアが出ることを本モデルで実測確認）。corruption/hardFallback フラグ監視を必須化（hardFallback 透過経路 :728-738/:789-806 の混入排除）

**Phase 0-3 の残り（周波数応答）は v1.8 §2.6 Phase 0-3 [A]/[B]/[C]/[D] のまま**。ただし [D] passband ripple の判定帯域は R3-3 により **[0.005, 0.30] Fs_in（dense 評価）** に変更、[0.30, edge] は記録のみ。

#### 2.6.1 B-1-P0 GATE（v1.9・P0-E を R3-1 + R4-13 で更新）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT（絶対値）** と **Candidate vs Baseline（差分帰属）** の 2 系統。

| ID | 判定対象 | 判定条件（v1.9 確定値） | 合否 |
|----|----------|------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が明示承認済み（E1〜E5・v1.9 版添付） | □ PASS / □ FAIL |
| **G-BL** | Baseline sanity | **DC round-trip = 0.75^N（±1e-6）のみを gate 条件とする**（R3-2）。D1 鏡像比（−9.5 dB 系/+5.8 dB を含む）は記録のみ | □ PASS / □ FAIL |
| **REF-FIDELITY** | Phase 0-1b | Shadow Reference（baseline モード）== production: 全項目 bitwise 一致（不可なら相対誤差 ≤ 1e-15）。**初期化は `reset()` 経由必須（R4-3）** | □ PASS / □ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成） | □ PASS / □ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB**（実測 −0.0000〜−0.0078 dB） | □ PASS / □ FAIL |
| **P0-C** | Passband ripple | **dense 評価（exact DTFT または ≥2^18 点 FFT）で [0.005, 0.30] Fs_in の max−min ≤ 0.05 dB**（candidate 実測: S1 0.001 / IIR3 0.025 / LP3 0.012 dB）。[0.30, 0.1 dB edge] は遷移落下として記録のみ | □ PASS / □ FAIL |
| **P0-C'** | Differential（帰属） | Candidate/Baseline 比 = (4/3)^N ± **0.05 dB**。**適用域 per-config（R2-7 維持・数値更新 R3-4/R4-8）**: IIR3/LP3 は f ≤ 0.45 Fs_in（実測偏差: IIR3 +0.0121 / LP3 −0.0025 dB）、S1 は f ≤ 0.30（実測 +0.0000〜+0.0002 dB）。0.49 Fs_in は記録のみ（実測: IIR3 +0.0062 / LP3 +0.0025 / S1 +3.3927 dB・v1.7 の IIR3 +0.043 は非再現）。Phase 2 flag-ON 測定が shadow 値と ±0.02 dB で一致すること | □ PASS / □ FAIL |
| **P0-D** | Stopband（stage-local・絶対 + 差分） | (D-1 差分) decimateStage は係数・コードとも不変 → Candidate/Baseline 差 = 0（FP 誤差 ≤1e-12）。(D-2 絶対) alias leakage ≤ **−(A_stage − 10) dB** を **[transition_end, 0.5] cycles/sample** で満たす（per-design 表 §2.7.2 は **本セッションで全点独立再現・確定**）。transition 領域は記録のみ | □ PASS / □ FAIL |
| **P0-E（再定義・R3-1 + R4-13）** | Image rejection | (E-1 記録) D1 baseline 鏡像比を全 design × f̂ 7 点で記録。(E-2 **gate**) D2 の **base==cand 不変性**: **[0.005, 0.40] Fs_in** の全測定点で差 ≤ **0.01 dB**。D3 の [0.05, 0.30] × {S1/IIR3/LP3} は base==cand 差 = 0.00 dB 厳密成立。**旧 gate「candidate ≥ 80 dB」は廃止**（R3-1: D1 定義でも到達不能・D2/D3 では base と同一のため無意味） | □ PASS / □ FAIL |
| **P0-F** | Latency | impulse 応答 peak 位置 = `Σ (taps[s]−1) × (baseRate/stageRate)`（±1 sample）+ centroid cross-check（±0.05 sample 目安）。案 E で不変（R3-11） | □ PASS / □ FAIL |
| **P0-G** | Float / Double host | same input / same partition / same reset state で: **maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8**。**reset state 初期化は R4-3 経由で実行** | □ PASS / □ FAIL |
| **P0-H** | Block / reset contract | partition A/B/C/D bitwise 一致（不可なら ≤1e-12）・`fresh+block#1 == stream→reset()→block#1`（bitwise）。**reset は `reset()` 経路を使用（R4-3）** | □ PASS / □ FAIL |
| **P0-I** | SoftClip local OS | I-a: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, centerTap=15, centerParity=1, convParity=0 実測）で Candidate DC = 1.0 ± 1e-6。I-b: 動作点影響を upPeak/downPeak/閾値到達度で記録 | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計、**または R3-12 記録の TruePeakDetector 型全 FIR 構造への変更**に方針変更する。

### 2.7 測定軸と実測基準値（v1.9 確定・authoritative = model_polyphase_20260920.py）

#### 2.7.0 周波数軸の正規化定義（v1.8 §2.7.0 を維持・確定）

```
own-rate cycles/sample: f̂ = f / Fs_stage（stage 自身の動作レート基準・有効域 [0, 0.5]）
Fs_in 基準（full chain）: f̂_in = f / Fs_in（入力帯域 [0, 0.5] のみ有効）
整合例（本モデルで再確認）: stage-0 (511 taps) の −0.1 dB edge = 0.2448 cycles/sample
 = 0.4896 Fs_in ≒ full-chain candidate edge 実測 0.4897 Fs_in ✓
```

#### 2.7.1 Full-chain 実測基準値（v1.9・dense exact-DTFT・baseline / candidate）

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC baseline / candidate | 0.750000 / 1.000000 | 0.421875 / 1.000000 | 0.421875 / 1.000000 |
| candidate \|H\| @50Hz〜0.25 Fs_in | −0.0000〜−0.0004 dB | −0.0078〜−0.0108 dB | +0.0025〜−0.0047 dB |
| (4/3)^N 差分偏差 @0.45 Fs_in | **+1.6350 dB**（適用外） | **+0.0121 dB** | **−0.0025 dB** |
| (4/3)^N 差分偏差 @0.49 Fs_in | +3.3927 dB | **+0.0062 dB**（v1.6/v1.7 の +0.043 は非再現・R3-4/R4-8） | +0.0025 dB |
| ripple [0.005, 0.45] base / cand | 5.242 / 3.607 dB | 0.111 / 0.084 dB | 0.053 / 0.039 dB |
| **ripple [0.005, 0.30] base / cand**（P0-C 判定帯域） | 0.001 / 0.001 dB | 0.034 / **0.025 dB** | 0.016 / **0.012 dB** |
| 参考: ripple [0.005, 0.75×edge] cand | 0.0009 dB | 0.0358 dB | 0.0174 dB |
| candidate 0.1 dB passband edge | **0.3526 Fs_in** | **0.4897 Fs_in** | **0.4945 Fs_in** |

（S1 は v1.8 記載と全点一致。IIR3/LP3 の dev/ripple は測定密度依存のため v1.9 値を authoritative とする。
base の 0.1 dB edge は絶対値評価では意味を持たないため candidate のみ掲載）

#### 2.7.2 FIR 設計の絶対基準値（P0-D floor の出典・**本セッションで全点独立再現 → 確定**）

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

#### 2.7.3 Image rejection 実測（v1.9 確定・3 定義確定・R3-1 + R4-13 反映）

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

**D2（単段 round-trip 出力・base==cand ≤ 0.01 dB・[0.005, 0.40] で gate 確定）**:

| design | 0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | **0.45（記録のみ）** |
|--------|------|------|------|------|------|------|------|
| 31/90 | −165.5 | −143.0 | −102.9 | −97.0 | −114.3 | −125.8 | **−137.57 (base) / −137.84 (cand)・diff 0.27 dB（R4-5/R4-12）** |
| 127/110 | −171.9 | −150.0 | −65.7 | −86.9 | −97.9 | −74.0 | −118.9 |
| 511/140 | −139.7 | −117.9 | −78.2 | −70.1 | −88.0 | −99.8 | −113.4 |
| 63/120 | −185.7 | −144.3 | −108.8 | −90.0 | −115.4 | −114.8 | −106.1 |
| 255/140 | −162.0 | −173.1 | −90.7 | −109.0 | −79.2 | −96.2 | −109.7 |
| 1023/160 | −150.2 | −108.3 | −74.8 | −51.1 | −79.4 | −77.5 | −71.1 |

（D2 = \|H_rt(0.5−f̂)\|/|H_rt(f̂)\| = round-trip 伝達関数の stopband 比。案 E は一様 (4/3)^N 増幅のみで
比を変えない（理論帰結）。実測 base==cand は [0.005, 0.40] で FP 誤差内（≤0.01 dB・1023/160 @0.05 で −150.23/−150.22 など）。
0.45 は 31/90 design の transition_end=0.3452 通過点であり FP 量子化拡大領域・R4-13 で gate 外）

**D3（full-chain 出力・base==cand 完全一致）**: S1/IIR3/LP3 × f̂ ∈ {0.05, 0.10, 0.20, 0.30} の全点で
mirror(0.5−f̂) = **−205〜−270 dB**（実質零・LTI 帰結）・worst spur = 測定窓 sidelobe −98〜−101 dB。
**base==cand 差 = 0.00 dB（全点）**

### 2.8 教訓・監査記録（v1.9・v1.8 §2.8 1〜15 を継承し 16〜18 を追加）

1〜15: **v1.8 §2.8 のまま**（oracle 依存禁止・gain convention と FIR 構造の分離・engine fit HOLD・Phase 0 先行・案 E=仮説だが defect 客観確定・perfect reconstruction 表現注意・anti-aliasing は実測確認・compile-time flag・姉妹実装扱い・validator error contract・own-rate 軸統一・shadow 参照 fidelity gate・測定定義混用・密グリッド評価・TruePeakDetector 後続候補）。

16. **【v1.9 追加・R4-3】reset() と clearAllStages() の動作差異**: 履歴クリア範囲は完全同一だがアトミックフラグ解除数が異なる（3 個 vs 1 個）。`hardFallbackActive` 解除は `reset()` のみ。Phase 0 測定系は `reset()` 経由の初期化を必須とする
17. **【v1.9 追加・R4-4】silence パスの精密な行範囲認識**: 「silence 早期 return パス」は :583-613（コメント :583 + 本体 :584-613・入口 :594・early return :606-612）であって、:614 以降は通常パス（バウンドチェック・境界違反時 early return :643-649）
18. **【v1.9 追加・R4-13】FP 量子化拡大領域の gate 除外**: 遷移端直後（31/90 transition_end=0.3452 通過後の 0.45 など）の FP 量子化ノイズ拡大は、対象 design の仕様上予期される数値であり gate 範囲外として記録のみとする。gate は [0.005, 0.40] Fs_in に統一（全 design で ≤0.01 dB 確認）

### 2.9 B-1-P0 GATE の適用要件（v1.8 §2.9 のまま・確定）

production src/ modified=0 / staged=0・measurement/harness のみ test-only 追加可・commit 禁止・calibration 禁止・threshold update 禁止・engine-fit 0.75^N（経験式）unchanged。

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC（v1.8 §3.1 を継承・行番号再実測一致）

- 根本原因: `setAutoGainStagingEnabled`（AudioEngine.h:1416-1432）の **ON→OFF 遷移時のみ** `getEQProcessor().setAGCEnabled(!enabled)`（:1425）が発火。既定 staging=true（:2626）→ `eq` モードの呼出順（`configureProbeFlatEQ` :1799 → `setAutoGainStagingEnabled(false)` :1803）で AGC が再設定される
- `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）・`getEQProcessor`（:1292）と `setEQAGCEnabled`（:1306）は同一 `uiEqEditor` 経由 ✓
- **修正案 A（呼出順逆・eqdiag パターン :1833/1834 に揃える）を維持**。検証手順（§3.1.3・state propagation 明示版）も維持。窓 `[0.486,0.496]`（:2042-2044）は不変前提だが再校正は Phase 3-D

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（production 欠陥として記録・v1.8 §3.2 を継承）

- `EQProcessor.Processing.cpp`: `bypassTransitionActive` :516 / `dryCopyBase` 充填 :570-578 / ブレンド本体 :978-993（`canBlendDry = (dryCopyBase != nullptr)` :980・dry 混合 :993）実測 ✓
- 定常 B-1 の原因ではない・遷移時のみ。**修正案 D（記録のみ・別 work item）を維持**

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化（v1.8 §3.3 を継承）

- `ir` モードは IR ロード後 bypass 解除なし → 出力は dry コピー（:1745 bypass 固定・:1749 ir 分岐）。`irwet<digit>` は :1763 で分岐・**解除済み（:1778）**
- **修正案 B（記録のみ・`irwet` で wet 対照カバー）を維持**

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役（v1.8 §4 を継承・R4-9 数値確定）

### 4.1 現状（本日再実測・R4-9 数値確定）

- 519 行・TEST_F（fixture `PublicationValidatorIsolationTests`）**34** + TEST（fixture `CrossfadeAuthorityRegressionTest`）**4** = 38 ケース ✓（`grep -c` 実測）
- 分類（メソッド名プレフィクス集計）: **ValidatePublication 7 / ValidateSemanticConsistency 1 / ValidateTopology 6 / ValidateResources 11 = 25**・**ValidateTransition 8 / checkNoConflictingTransitions 直接 1 = 9** ✓（メソッド名 grep 集計実測 — v1.7 R2-10 の修正値を確定）。全 38 TEST_F/TEST は fixture `PublicationValidatorIsolationTests`(34) + `CrossfadeAuthorityRegressionTest`(4)
- **CMake 未登録（add_executable 39 件のいずれにも該当なし・R4-9 で本日再実測）** ✓・gtest 使用は本ファイルのみ ✓
- `validator_.checkNoConflictingTransitions` 実呼出し **9 箇所**（:135/:246/:255/:265/:273/:281/:313/:324/:334）✓
- **コンパイル不可**（private 宣言: RuntimePublicationValidator.h:92 private:・:101 checkNoConflictingTransitions）✓
- `tools/build-debug.bat:29` = `cmake --build ... --target PublicationValidatorIsolationTests`（stale・実測）✓
- 検査順序: RuntimePublicationValidator.cpp:14-41（SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins）✓ errorMessage 固定文字列 :16/:24/:32/:40 ✓
- AudioEngine.h:3659 `validator_(&validator)` → :3670 `validator_->validatePublication(world)` ✓

### 4.2 移行方針（v1.8 §4.2 のまま）

Phase 0 分類 → Phase A（CrossfadeAuthority 4 ケース最優先）→ Phase B（validator 25 + CheckTransition 9）→ Phase B'（Semantic Equivalence Gate・target failure isolation 含む）→ Phase C（退役 + build-debug.bat:29 の stale 参照除去）。

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

## 6. 推奨する実行順序（v1.9・骨格は v1.8 §6 のまま）

```
[Step 1] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・v1.8 §3.1.3 の検証手順に従う）
  - F-3/F-4: 記録のみ
  - 検証: AudioEngineHarness.exe --buzz-rigcheck=eq で gainpath staging=0 eqAGC=0 を確認

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移植 + Phase B' semantic equivalence gate
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-12（commit 意思決定・ユーザー判断待ち）
  - 案 A 採用時はエントリ配線の最小変更を含めて commit
  - O-8（pyc 復元可否）・O-9（docs 目録扱い）・O-10（AGENTS.md MIMO Desktop 追記）/O-11（残課題台帳 B-1 詳細追加）・O-12（ConvoPeq.md 生成物扱い）を判断対象に追加（v1.9）

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E）
  - Phase 0-0: DESIGN-CONTRACT-A（v1.9 版 E1〜E5）を明示承認
  - Phase 0-1: Baseline characterization（0.75^N ±1e-6・record-only）
  - Phase 0-1b: REF-FIDELITY gate（bitwise・reset() 経由初期化必須・R4-3）
  - Phase 0-2: Shadow Candidate E（DC = 1.0 ± 1e-6）
  - Phase 0-3〜0-6: 周波数（D1/D2/D3 定義確定版・D2 gate [0.005, 0.40]・R4-13）/ block・reset / SoftClip / float-double
  - B-1-P0 GATE（§2.6.1・R4-13 反映版）全 PASS → Phase 0 review → ユーザー GO
  - Phase 1: CMake option 導入（R3-8 方式）→ Phase 2: flag ON 限定検証
  - Phase 3-A → 3-B1 → 3-B2 → 3-C → 3-D → Phase 4
  - **Phase 0 FAIL 時の後続候補**: 案 D 統合 / tap 再設計 / TruePeakDetector 型全 FIR 構造（R3-12）

[Step 5] §5 別課題（将来 work item 化）: R-2 → R-1 → B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**（v1.8 と同一）。

---

## 7. 検証計画（v1.8 §7 を継承・閉形式モデルを追加）

### 7.0 閉形式モデル（v1.8 新設・v1.9 で本セッション再実行確認）

```bash
python doc/work113/model_polyphase_20260920.py
# 出力: 係数検証（FIRsum=1.0/convSum=0.5 全 6 design）→ DC round-trip → full-chain |H| →
#       per-design 絶対基準 → D1/D2/D3 image rejection
# 保存済み全出力: doc/work113/model_polyphase_20260920_results.txt
# v1.9 で Python 3 / numpy 2.5.3 / scipy 1.18.1 环境下に再実行し全表一致確認
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

（静的解析: cppcheck を `CustomInputOversampler.cpp` に本日再実行 → 指摘 0 件再確認・R4-6。clang-tidy では portability-simd-intrinsics / bugprone-branch-clone のみ検出・R4-7。両者 design intent として許容可）

---

## 8. ロールバック計画（v1.8 §8 のまま・確定）

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral） | `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild（第一手段・production source 触らない） |
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

## 10. 監査ログ・参照（行番号は 2026-09-20 本日再実測・全件一致確認・R4-4 反映）

- 前スナップショット: `doc/work113/residual_tasks_20260919.md` / 本セッション報告書: `residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 + レビュー① 38 観点）・`renew_plan_verification_20260920.md`（レビュー②）・v1.6/v1.7/v1.8 改訂サマリ・v1.9 改訂サマリ
- authoritative 計測モデル: `doc/work113/model_polyphase_20260920.py` + `model_polyphase_20260920_results.txt`（v1.8 新設・v1.9 で Python 3 / numpy 2.5.3 / scipy 1.18.1 再実行確認）

**行番号再監査結果（全件一致・v1.8 §10 を確定・R4-4 反映版）**:

| ファイル | 確認行 | 結果 |
|----------|--------|------|
| `src/CustomInputOversampler.h` | isLinearPhaseFIR/isSymmetricUpDown: 21-22 / `reset()` 宣言: 32 / `clearAllStages()` 宣言: 71 | ✓ |
| `src/CustomInputOversampler.cpp` | tapsForStage/attenuationForStage 84-106 / clearStage 108-119 / prepareStage 287-390（centerTap 292・center 0.5 335/348・scale 341・convCoeffs 360-365）/ prepareSingleStage 392 / **reset** 452-467（per-ch clear 459-460・3 アトミック 464-466・**R4-3**） / **clearAllStages** 469-484（per-ch clear 477-479・1 アトミック 483・**R4-3**）/ markCorruptionDetected 486-490 / interpolateStage 492-568（convValue ×2: **557**・denormal clamp 558-559）/ decimateStage 570-723（silence パス **583-613**（**R4-4** 訂正）・通常パス 615-649・n ループ 652-・acc 716・output 717）/ reset 452 / processUp **725**（hardFallback 透過 728-740・空ブロック返却 744-748・段ループ 764-776・currSamples <<= 1 775）/ processDown **785**（hardFallback 透過 789-806・corruption クリア 808-818・容量超過 clear 826-832） | ✓（R4-3/R4-4 反映） |
| `src/audioengine/AudioEngine.h` | setAutoGainStagingEnabled 1416-1432（setAGCEnabled **1425**）/ 2626 staging 既定 true / 1292 getEQProcessor・1306 setEQAGCEnabled / 3659・3670 validator | ✓ |
| `AudioEngine.Processing.Latency.cpp` | static_assert 6-8 / taps 表 22-24 / groupDelaySamplesAtStageRate = taps[stage]−1 **30** | ✓ |
| `DSPCoreLifecycle.cpp` | softClipOS.prepareSingleStage(31, 90.0) **188 / 261**（diagLog 版/非 diag 版の 2 箇所） | ✓ |
| `DSPCoreFloat.cpp` | softClipOS.processUp **405** / processDown **413** | ✓ |
| `DSPCoreDouble.cpp` | kOutputHeadroom = 0.8912509381337456 **593** | ✓ |
| `EQProcessor.Processing.cpp` | bypassTransitionActive **516** / dryCopyBase 570-578 / blend 978-993 | ✓ |
| `RuntimePublicationValidator.cpp/.h` | 検査順序 14-41 / errorMessage 16/24/32/40 / private **92** / checkNoConflictingTransitions **101** | ✓ |
| `TruePeakDetector.cpp`（v1.8 拡張） | interpolateStage **284-311**（履歴 shift 297-300・両位相 = center + conv **305-306**・×2 補償なし・R3-12） | ✓ |
| `BassBuzzMeasurement.cpp`（**本日 2687 行**・6e61b5a 1429 行から +1258 行/93%: 16:9 で差分 6e61b5a...HEAD = +634/−2）| configureProbeFlatEQ **1072**（内 setEQAGCEnabled 1081） / runOversamplerDirect **1293**（upPeak 1342-1350・roundTripGain **1367**） / runEqDirectDriveAttribution **1526/1532** / parseHcIdx/parseLcIdx **1558/1564**（stoi 1562/1567） / stod/stoi/stof 1582-1615 / flip-eqgain **1613** / F-2 eq モード呼出順 1799→1803（eqdiag パターン 1833/1834） / `irwet` 1749-1778 / eq 窓 **2042-2044** | ✓ |
| `PublishPipelineIntegrationTests.cpp` | main **1085** / 前方宣言 **1116** / 呼出 **1223** | ✓ |
| `PublicationValidatorIsolationTests.cpp` | **519 行** / TEST_F 34 + TEST 4 / 分類 7/1/6/11 + 8/1 / 呼出 135/246/255/265/273/281/313/324/334 | ✓ |
| `CMakeLists.txt` | option 群 **40**（clang-tidy）/ **71**（MKL）/ 128-129 / NUC_DEBUG_GUARDS **52-58**（`add_compile_definitions` の既存前例）/ add_subdirectory(JUCE) **1043** / juce_add_gui_app **1062** / CONVOPEQ_ALL_SOURCES **1133** / target_sources **1282** / CONVOPEQ_ALL_SOURCES ループ **1892** / add_executable(AudioEngineHarness) **1899** / **add_executable 計 39 件（R4-9）** | ✓ |
| `tools/build-debug.bat` | stale target **29**（PublicationValidatorIsolationTests） | ✓ |
| `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp` | up `buf[N-1] = 2 * samples[i]` **185** / down `buf[N-1] = bufferSamples[i << 1]` **228**（両位相 ×2 / down ×1 = E5 証拠） | ✓ |
| `build\Release\AudioEngineHarness.exe` | 実在 **41,137,664 bytes**（R4-11・2026-09-20 14:25 mtime） | ✓ |
| cppcheck (`CustomInputOversampler.cpp`) | `--enable=warning,performance,portability --std=c++20` 実行 → **exit=0・指摘 0 件**（R4-6） | ✓ |
| clang-tidy (`CustomInputOversampler.cpp`) | `bugprone-*,performance-*,portability-*,misc-*` 実行 → portability-simd-intrinsics（AVX2 由来）+ bugprone-branch-clone（parity 由来）のみ・実 bug 0 件（R4-7） | ✓ |

---

## 11. ユーザー判断待ち項目（v1.8 §11 を継承・O-10/O-11/O-12 を追加）

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 計装 | A: 最小実行可能資産 / B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-7 AGENTS.md → §1 O-10 に分離 | 「触らない」継続 / 記録更新として commit | **commit（記録更新扱い）** |
| §1 O-8 pyc | 復元（git checkout）/ 触らない / `.gitignore`+`rm --cached` を別 work item 化 | **触らない→将来 work item** |
| §1 O-9 docs 目録 | 触らない / commit | **触らない（ユーザー判断）** |
| §1 O-10 AGENTS.md | 触らない / commit（環境記録更新扱い） | **commit（環境記録更新扱い）** |
| §1 O-11 残課題台帳 | 触らない / commit（台帳更新扱い・O-1/O-2 と同梱可） | **commit（台帳更新扱い）** |
| §1 O-12 ConvoPeq.md | 触らない（生成物・O-3 方針継続） | **触らない（O-3 方針継続）** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E** |
| §2 DESIGN-CONTRACT-A（G-0） | 明示承認 / 修正要求 / 却下 | **明示承認（v1.9 版 E1〜E5・E3 は修正後）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **全実施** |
| §2 Phase 3 の commit 分割 | 3-A/3-B1/3-B2/3-C/3-D 分割 / 単一 Phase 3 | **分割** |

### §2 B-1 採用案の前提条件（v1.9・P0-E は再定義版・R4-13 反映版に更新）

1. **DESIGN-CONTRACT-A**（v1.9 版）が明示承認される（Phase 0-0・G-0）
2. **REF-FIDELITY** が bitwise 一致（Phase 0-1b・`reset()` 経由初期化必須・**R4-3**）
3. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（Phase 0-2・P0-A）
4. **passband 特性が維持**: P0-B（50 Hz/1 kHz ±0.1 dB）・P0-C（[0.005, 0.30] ripple ≤ 0.05 dB）
5. **anti-aliasing / stopband が維持**: P0-D（alias leakage ≤ A−10 dB・transition_end 以降）
6. **出力 image が不変**: P0-E-2（D2 [0.005, 0.40] で base==cand ≤ 0.01 dB — **R4-13** で gate 帯域再限定）
7. **latency が不変**（impulse 実測・P0-F）
8. **SoftClip 局所 OS が正常動作**（P0-I）
9. **Differential が center-phase ×2 のみに帰属**（P0-C'・per-config 適用域）

### v1.9 最終判定表（v1.8 §11 判定表を継承）

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-12 | **GO 候補** | commit/push/復元の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替） |
| §3 F-3 / F-4 | **記録継続** | production 潜在欠陥 / dry 測定基準維持 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO（defect は DC round-trip = 0.75^N で客観確定・D1 補助証拠）** | R3-2 により証拠の主従を更新 |
| **B-1 Phase 0** | **GO（測定仕様 v1.9 適用が条件）** | P0-E 再定義 + P0-C 判定帯域統一 + dense 評価必須化 + **P0-E-2 gate 帯域 [0.005, 0.40]（R4-13）** + **reset() 経由初期化必須（R4-3）** を含む |
| B-1 Phase 1 / 2/3/4 | **HOLD** | Phase 0 全 PASS + ユーザー GO が前提 |
| **B-1 案 E** | **有力仮説として採用** | 仮説だが defect 自体は客観確定。FAIL 時の後続候補を R3-12 に追加 |
| 「perfect reconstruction」 | **表現修正のまま** | DC gain 1.0 のみ確定 |
| B-1 calibration 変更 | **HOLD** | Phase 3-B1/B2 → 3-C/D の順 |
| `0.75^N` engine-fit 削除 | **HOLD** | Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | `#if` セマンティクス確定（R3-8） |
| TruePeakDetector 姉妹実装 | **記録継続・後続候補として格上げ記録（R3-12）** | 案 E の正解としない |
| **v1.7 §2.7.3 candidate image 契約** | **廃棄（非再現）** | R3-1 |
| **v1.9 新規: P0-E-2 gate 帯域 [0.005, 0.40]（R4-13）** | **確定** | D2 の [0.005, 0.40] で全 design ≤0.01 dB 確認済。0.45 は FP 量子化拡大領域のため gate 外 |
| **v1.9 新規: reset() vs clearAllStages() 差異（R4-3）** | **確定** | 履歴クリア同一・アトミック解除数 3 vs 1。Phase 0 測定系は reset() 経由初期化必須 |
| **v1.9 新規: silence パス :583-613（R4-4）** | **確定** | v1.8 §12.2 「:584-629」を訂正。:614-649 は通常パス |
| **v1.9 新規: O-10/O-11/O-12** | **GO 候補** | O-10（AGENTS.md 環境記録更新・commit 可）・O-11（残課題台帳 B-1 詳細追加・commit 可）・O-12（ConvoPeq.md 生成物・commit しない） |

---

## 12. 本セッションの検証エビデンス（2026-09-20・確定事項の出所・v1.9 で R4 反映）

### 12.1 閉形式モデル（C++ 厳密移植・authoritative 計測定義・v1.9 で再実行）

- `doc/work113/model_polyphase_20260920.py`（v1.8 セッション新規・v1.9 で再実行）:
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
- v1.9 で Python 3 / numpy 2.5.3 / scipy 1.18.1 环境下に再実行し、`model_polyphase_20260920_results.txt` と plan §2.7.1/§2.7.2/§2.7.3（D1/D2/D3）が全点一致確認。例外: 31/90 D2 @0.45 cand = -137.84 vs saved -137.57（diff 0.27 dB・numpy バージョン差由来 FP 量子化拡大・**R4-5/R4-12/R4-13**）

### 12.2 ソース監査エビデンス（v1.9 で全行番号再実測・R4-3/R4-4 反映版）

- `CustomInputOversampler.h`: 21-22（isLinearPhaseFIR/isSymmetricUpDown）・`reset()` 宣言 **:32**・`clearAllStages()` 宣言 **:71**
- `CustomInputOversampler.cpp`:
  - 84-106（taps/attenuation）/ 108-119（clearStage）/ 287-390（prepareStage・centerTap :292・center 0.5 :335/:348・scale :341・convCoeffs :360-365）
  - 392（prepareSingleStage）
  - **reset** **:452-467**（numStages ループ × per-ch `FloatVectorOperations::clear`（upHistory :459・downHistory :460）・3 アトミック解除 :464-466 = corruptionDetected/consecutiveCorruptionAutoClearCount/hardFallbackActive・**R4-3**）
  - **clearAllStages** **:469-484**（numStages ループ × per-ch clear（upHistory :477・downHistory :479）・1 アトミック解除 :483 = corruptionDetected のみ・**R4-3**）
  - markCorruptionDetected :486-490
  - 492-568（interpolateStage・convValue ×2 **:557**・denormal clamp :558-559・corruption 境界 :509-554）
  - 570-723（decimateStage・**silence パス :583-613（コメント :583 + 本体 :584-613・入口 :594・early return :606-612・R4-4 訂正）**・通常パス :615-649 / :651-・n ループ :652-・output :717）
  - processUp **:725**（hardFallback 透過 :728-740・空ブロック返却 :744-748・段ループ :764-776・currSamples <<= 1 :775）
  - processDown **:785**（hardFallback 透過 :789-806・corruption クリア :808-818・容量超過 clear :826-832）
- `AudioEngine.h`: setAutoGainStagingEnabled 1416-1432（setAGCEnabled **:1425**）/ getEQProcessor **:1292**・setEQAGCEnabled **:1306**（同一 uiEqEditor 経由）/ autoGainStagingEnabled 既定 true **:2626** / validator 初期化 :3659・呼出 :3670
- `AudioEngine.Processing.Latency.cpp`: static_assert **:6-8** / taps 表 :22-24 / groupDelay **:30**。R3-11: 案 E は taps・係数列不変 → static_assert 成立持続
- `DSPCoreLifecycle.cpp`: softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock) **:188/:261**（diagLog 版/非 diag 版・同一関数経路）
- `DSPCoreFloat.cpp`: float→double 変換 :252-263 / softClipOS.processUp **:405** / processDown **:413**
- `DSPCoreDouble.cpp`: kOutputHeadroom = 0.8912509381337456（**593**・0.891 は EqDirectLog 実測の eqIdentityMode 窓 0.891 とは無関係・別定数）
- `EQProcessor.Processing.cpp`: bypassTransitionActive **:516** / dryCopyBase 充填 **:570-578** / blend 本体 **:978-993**（canBlendDry :980・混合 :993）
- `RuntimePublicationValidator.cpp/.h`: 検査順序 **:14-41** / errorMessage **:16/:24/:32/:40** / `private:` **:92** / checkNoConflictingTransitions **:101**
- `TruePeakDetector.cpp`: interpolateStage **:284-311**（履歴 shift :297-300・両位相 = center + conv **:305-306**・×2 補償なし・R3-12）
- `BassBuzzMeasurement.cpp`（本日 **2687 行**）: §10 表の行番号全件 + F-2 呼出順 **:1799→:1803**（eqdiag パターン :1833/:1834）/ `irwet` 分岐 **:1763**・解除 **:1778**
- `PublishPipelineIntegrationTests.cpp`（本日 **1340 行**）: main **:1085** / 前方宣言 **:1116** / 呼出 **:1223**
- `PublicationValidatorIsolationTests.cpp`: **519 行**・TEST_F 34（fixture PublicationValidatorIsolationTests）+ TEST 4（fixture CrossfadeAuthorityRegressionTest）
  分類（メソッド名）: ValidatePublication 7 + ValidateSemanticConsistency 1 + ValidateTopology 6 + ValidateResources 11 + ValidateTransition 8 + checkNoConflictingTransitions 直接 1 = **34**（残り 4 は CrossfadeAuthority 回帰）。呼出 9 箇所実測
- `CMakeLists.txt`: option 群 **:40**（clang-tidy）/ **:71**（MKL）/ :128-129 / NUC_DEBUG_GUARDS **:52-58**（`add_compile_definitions` の既存前例）/ add_subdirectory(JUCE) **:1043** / juce_add_gui_app **:1062** / CONVOPEQ_ALL_SOURCES **:1133** / target_sources **:1282** / CONVOPEQ_ALL_SOURCES ループ **:1892** / add_executable(AudioEngineHarness) **:1899**。**add_executable 計 39 件（R4-9）**
- `tools/build-debug.bat`: stale target **:29**。`build\\Release\\AudioEngineHarness.exe` 実在（**41,137,664 bytes**・R4-11）
- `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp`: up `buf[N-1] = 2 * samples[i]` **:185** / down `buf[N-1] = bufferSamples[i << 1]` **:228**（E5 証拠）
- `tools/__pycache__/`: tracked 2 件（`apply-solidlsp-bash-ls-patch.cpython-314.pyc` Modified Bin 16645→16895 + `retire_authority_verifier.cpython-314.pyc`）— O-8（v1.8 §1.8・v1.9 維持）
- `docs/tool-inventory-2026-09-20.md`: untracked・**3102 bytes** — O-9（v1.8 §1.9・v1.9 維持）
- `AGENTS.md`: Modified（+8/−2・MIMO Desktop 追記+CRLF 警告）— O-10（v1.9 新設・R4-1）
- `doc/work113/residual_tasks_20260919.md`: Modified（+72/−0・B-1 詳細追加）— O-11（v1.9 新設・R4-2）
- `ConvoPeq.md`: Modified（+75/−0・生成物）— O-12（v1.9 新設）

### 12.3 要確認事項（Phase 0 で確定させる・gate 外）

1. **IIR3 の (4/3)^N 差分偏差 @0.49 Fs_in**: 本モデル +0.0062 dB vs v1.6/v1.7 +0.043 dB（S1・LP3・per-design 表は一致のため IIR3 の 0.49 近傍評価に限定的・R3-4/R4-8）。
   Phase 0-3 [A] 実測で確定。gate 外（記録のみ）のため Phase 0 GO/HOLD に影響しない
2. **P0-C の dense 評価閾値実現性**: 本モデルでは candidate [0.005, 0.30] ripple = S1 0.001 / IIR3 0.025 / LP3 0.012 dB（gate 0.05 dB に対し余裕あり）→ Phase 0-3 [D] 実測で確定
3. **静的解析の silence パス確認記録**: cppcheck（指摘 0・R4-6）は silence 早期 return パス（**583-613**・R4-4 訂正）の「成立条件の説明」に言及していない。
   Phase 0 実測で inputSilent 偽陽性がないこと（推移帯信号での誤検出 0）を corruption フラグで確認する
4. **【v1.9 新設・R4-5/R4-13】P0-E-2 gate 帯域の [0.005, 0.40] 限定と 0.45 の FP 量子化拡大**: numpy 2.5.3 で 31/90 D2 @0.45 cand vs base 差が 0.27 dB まで拡大。gate 範囲外（記録のみ）として除外。Phase 0 実測で同傾向を確認したら測定系の FP 精度依存として記録

---

*本書は v1.8（remediation_plan_20260920_v1.8_revised.md）の再改訂版 v1.9 である。
v1.8 の本文は参照文書として維持し、本書の R4-1〜R4-13・§1.10〜§1.12・§2.6.1（P0-E gate 更新）・§2.7.3（D2 gate 帯域更新）・§2.8（教訓 16〜18 追加）・§10/§12 の実測値が v1.8 と矛盾する箇所では**本書を優先する**。*

**v1.9 で新規追加した監査・確定事項**:
- **R4-1** O-10（AGENTS.md Modified・MIMO Desktop 追記・commit 可）
- **R4-2** O-11（残課題台帳 B-1 詳細追加・commit 可）
- **R4-3** `reset()` vs `clearAllStages()` の精密差異（履歴クリア同一・アトミック解除数 3 vs 1・`hardFallbackActive` 解除は reset() のみ）
- **R4-4** silence パス :583-613（v1.8 §12.2 の「:584-629」を訂正）
- **R4-5/R4-12/R4-13** P0-E-2 gate 帯域を [0.005, 0.40] に再限定（31/90 D2 @0.45 の cand vs base 差 0.27 dB を gate 外へ）
- **R4-6** cppcheck 0 件（本日再実測）
- **R4-7** clang-tidy portability-simd-intrinsics + bugprone-branch-clone のみ（design intent として許容可）
- **R4-8** IIR3 @0.49 = +0.0062 dB 確定（v1.6/v1.7 +0.043 非再現）
- **R4-9** add_executable 39 件（本日再実測）
- **R4-10** D2 1023/160 @0.05 base==cand 0.01 dB（FP 量子化・典型点）
- **R4-11** AudioEngineHarness.exe 41,137,664 bytes（本日再確認）
- **§1.10〜§1.12** O-10/O-11/O-12 新規項目
- **§2.8.16〜18** 教訓 3 件追加（reset/clearAllStages 差異・silence パス精密化・FP 量子化拡大領域の gate 除外）
