# ConvoPeq 残件 改修計画書（2026-09-20 時点）

- **版**: v1.6（v1.5 の厳密再監査版・レビュー② として実施。要調査・未確定事項をソース実測で確定）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **方針**: 段階リリース重視・既存動作への影響を最小化・ロールバック容易性確保
- **前提**: production `src/` の未 commit 差分は 0 件、HEAD = `8f127bfe`（docs 1 件）+ `c4a08171`（B-2 test 1 件）が未 push
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **基準ソース**: HEAD の production source（`ConvoPeq.md` は working tree 版 Generated 2026-09-20 07:00:33 で参考扱い。HEAD 版は commit `06794615` 同梱）
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「**有力仮説**」として採用（oracle C 案はユーザー監査で撤回済み）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**: 案 E は Phase 0 の characterization（§2.6 P0 GATE）で全 PASS しなければ Phase 1 実装 GO とはしない
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする（`kCorrectPolyphaseGain` / `kUseV2Oversampler` は採用しない）

---

## v1.5 → v1.6 改訂サマリ（レビュー②: 全観点をソース実測で再検証）

v1.5 の技術的中核（案 E・原因特定・Phase 0 3 層構造・段階リリース骨格）は **再検証の結果すべて正しいことが確認された**。一方で以下の誤り・未確定事項を検出し、v1.6 で確定させた。

| ID | 区分 | v1.5 の問題 | v1.6 の確定内容 |
|----|------|-------------|------------------|
| **N1** | 誤り（実行不能） | §7.1 が `cmake --build ... --target PublishPipelineIntegrationTests` / `build\Release\PublishPipelineIntegrationTests.exe` を指示 | **そのターゲット・バイナリは存在しない**。正: `add_executable(AudioEngineHarness … PublishPipelineIntegrationTests.cpp BassBuzzMeasurement.cpp …)`（CMakeLists.txt:1898）→ **ターゲット/バイナリは `AudioEngineHarness`**（`build\Release\AudioEngineHarness.exe` は実在確認済み）。`--buzz-rigcheck=eq` も AudioEngineHarness.exe のフラグ。§7.1 全面修正 |
| **N2** | 誤り（数値） | v1.5 §2.2 冒頭の taps 表記・latency 表の `140/110/80`、`160/140/100` | **attenuation は IIRLike `140/110/90`・LinearPhase `160/140/120`**（`tapsForStage`/`attenuationForStage`（CustomInputOversampler.cpp:82-106）実測）。Latency.cpp:22-23 の taps 表 `{511,127,31}` `{1023,255,63}` は正しい。全 taps は奇数 centerTap → **centerParity=1 / convParity=0 が全構成で固定**（R16 の前提も確認） |
| **N3** | 誤り（行番号） | §10: `DSPCoreLifecycle.cpp:188/261` | 正: **187/260**（`softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock)` 本体行）。DSPCoreFloat は **404/412**（405/413 は誤差 1 行）、DSPCoreDouble は **505/513** ✓ |
| **N4** | 誤り（期待値） | P0-C'（Differential）の期待を「f ≤ 0.25 Fs_in で (4/3)^N ± 0.05 dB」 | NumPy 逐語移植で **DC / 0.10 / 0.25 / 0.45 Fs_in で差分 = (4/3)^N に ±0.001 dB 以内で一致**（理想は +2.4988 dB/stage）。**適用域を f ≤ 0.45 Fs_in に拡大**。0.49 Fs_in は +0.04 dB 偏差（帯域端 alias 混合・実測 7.5394 dB vs 理想 7.4963 dB @N=3）のため「記録のみ」。P0-C 判定閾値は変更なし（後述 N5） |
| **N5** | 要強化 | P0-C ripple ≤ 0.2 dB の根拠が未記載 | NumPy sweep（単段 31 taps/90 dB、[0.005, 0.45] Fs_in 60 点）: **baseline ripple ≈ 5.24 dB / candidate ≈ 3.61 dB**（端点含む）。つまり **candidate の ripple は baseline より小さいが、帯域端で単段でも ~3 dB 級の落ち込みは残る**（PR ではない）。ripple 判定は sweep 中央領域（[0.01, 0.30] Fs_in）で ≤ 0.2 dB、sweep 全域では max−min の絶対値を記録のみとする 2 段定義に変更（詳細 §2.6.1） |
| **N6** | 要確定 | Shadow Candidate E の配置場所が未定義 | **`src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（header-only test-only reference）** に確定。`prepareStage` 同等の係数生成 + up/down を複製し `centerValue *= 2.0` を適用。AudioEngineHarness からのみ include（production 0 変更の GATE を維持） |
| **N7** | 誤り（軽微） | §4.1「519 行」・§10「runEqDirectDriveAttribution 1529」 | 正: PublicationValidatorIsolationTests.cpp は **501 行**（TEST_F 34 + TEST 4 = 38 ケースは一致 ✓）。runOversamplerDirect=**1293** / runEqDirectDriveAttribution=**1526** / 呼出し側 PPIT main=**1223**（無条件）/ eqIdentityMode=**2042** / eq 窓=**2043-2044** |
| **N8** | 確定 | §4.2 の `checkNoConflictingTransitions` 外部呼出し「9 箇所」 | 実測では同メソッド名は **10 行ヒット（128/135/246/255/265/273/281/313/324/334）**。128 は TEST_F 名行のため**実呼出しは 9 箇所**で v1.5 と一致 ✓。`RuntimePublicationValidator.h:92 private:` → **:101** が private メソッド宣言 ✓、FRIEND_TEST 0 件 ✓、CMake `add_executable` 39 件中に該当なし ✓ |
| **N9** | 確定 | §4.2「error string は参考情報」 | validator は **4 段階それぞれに固定の errorMessage 文字列**を返す（RuntimePublicationValidator.cpp:16/24/32/40: "Semantic consistency check failed" / "Topology validation failed" / "Resource availability check failed" / "Conflicting transitions detected"）。**static const 文字列であり契約として使用可能**。Phase B' では enum + errorMessage 両方を比較する |
| **N10** | 確定 | AudioEngine.h:3666-3670「Bridge → validator_」 | 実測: 3659 が ctor 初期化 `validator_(&validator)`、**3670 が `validator_->validatePublication(world)`**（Bridge 本体の検証呼出し）。行番号は概ね一致 ✓ |
| **N11** | 要確定 | Phase 1 の CMake snippet が `target_compile_definitions(<対象ターゲット> …)` と曖昧 | **確定**: option を CMakeLists.txt:40 近傍（既存 `option(CONVOPEQ_ENABLE_CLANG_TIDY …)` 群と同じ冒頭ブロック）に追加し、`add_compile_definitions(CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)` を `juce_add_gui_app(ConvoPeq …)`（:1062）より前に配置 → ConvoPeq（:1062 作成・:1282 で ALL_SOURCES）と AudioEngineHarness（:1898 作成・:1890-1894 で ALL_SOURCES 派生）の双方に供給される。なお JUCE モジュール（add_subdirectory(JUCE) :1043）にもマクロが伝播するが CustomInputOversampler.cpp を含まないため無害 |
| **N12** | 記録 | v1.5 §12 の NumPy 再現は前セッションの記録 | 本セッションで **Python 3.14.7 + NumPy 2.5.2 で独立に再実装・再測定**（§12）。全確定値を再確認 |

**レビュー②で変更しない点（再確認済み）**:
- 案 E の技術的内容（`centerValue *= 2.0` 1 行追加）: NumPy 再現でも DC round-trip = 1.0（1 段・3 段両構成）を再確認 → 維持
- Phase 0 の Baseline / Candidate / Differential 3 層構造・P0 GATE 全体設計: 維持（N4/N5/N6/N11 で精密化）
- 段階リリース Phase 0→1→2→3-A/B/C/D→4 の骨格: 維持
- §1〜§5 の GO/HOLD 判定: 維持

---

## 0. 凡例と全体戦略

| 区分 | 着手優先度 | 影響範囲 | ロールバック |
|------|------------|----------|--------------|
| **P0 即時**（破壊的・不可逆） | なし | — | — |
| **P1 高**（影響大・段階リリース） | B-1 | 全オーディオ経路 | flag OFF rebuild |
| **P2 中**（harness / テスト cleanup） | F-1〜F-4 | test-only（F-3 は production 欠陥の記録） | ファイル revert |
| **P3 低**（運用 / 環境 / 既知制限） | O-1, B-3, D-1, D-2, R-1, R-2 | 環境 or CLI | 設定 revert |

**全体戦略**:
1. **§1（O-1〜O-7）の意思決定** → コミット方針確定（ユーザー判断待ち）
2. **§2 B-1** → 大規模 DSP 修正。Phase 0（characterization）→ 1（flag 導入）→ 2（flag ON 限定検証）→ 3-A/B/C/D（全段 ON + 分割 commit）→ 4（flag 削除）
3. **§3 F-2** → harness cleanup（test-only）。B-1 とは独立して先に着手可能。F-3/F-4 は記録のみ
4. **§4 F-1** → テスト移行（custom main ハーネスへ）。CrossfadeAuthority 4 件を最優先
5. **§5 別課題** → 既存記録のまま保留。必要時に別 work item 化

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-7）

### 1.1 O-1: B-1 計測用 test-only 計装（+642 行）

| 項目 | 内容 |
|------|------|
| ファイル | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（+634/−2）、`PublishPipelineIntegrationTests.cpp`（+8/−0）。git diff --stat HEAD 実測で一致 ✓ |
| 内容 | `eqdiag` / `eqdiagser` / `eqos<digit>` / `irwet<digit>` 診断モード、`gainpath` 行、`[EQ_DIRECT]` / `[OF_DIRECT]` / `[OS_DIRECT]` 直接駆動測定 |
| 影響 | test-only。production `src/` 変更なし |
| 判定窓 | 既存 `eq` 判定窓 `[0.486, 0.496]` は不変（`eqIdentityMode = (rigCheckMode == "eq")` — BassBuzzMeasurement.cpp:2042、窓本体 2043-2044 で実測確認 ✓） |
| 出力キー | `[OS_DIRECT] ... roundTripGain=`（`round-trip=` ではない） |
| エントリ | `runOversamplerDirect()`（BassBuzzMeasurement.cpp:1293）は `runEqDirectDriveAttribution()`（同:1526）から呼ばれ（同:1532）、`PublishPipelineIntegrationTests.cpp` main から**無条件呼出し**（:1116 前方宣言・:1223 呼出し。+8 diff と一致 ✓） |

**3 案**:

| 案 | 内容 | メリット | デメリット |
|----|------|----------|------------|
| **A（推奨）** | **「B-1 regression detection に必要な最小実行可能資産」を commit**（`[OS_DIRECT]` 本体 = `runOversamplerDirect()` + main からの直接呼出し配線） | 回帰検出を最小コストで残せる | `eqdiag` / `irwet` 診断モードは手元検証用に残る。2 ファイル双方に触れる |
| **B** | 全 +642 行を 1 commit | 将来の B-1 再発の即時検出資産として完全保存 | 行数大・レビュー負荷大 |
| **C** | すべて退役（破棄） | 後方互換問題なし | 再発時の解析遅延 |

**推奨案**: **A**。`[OS_DIRECT]` / `[EQ_DIRECT]` / `[OF_DIRECT]` は同一エントリ関数 `runEqDirectDriveAttribution()` に束ねられており `git add -p` での分離は不可能。案 A の実体:
1. `runOversamplerDirect()`（1293）とその呼出し必要分を commit
2. `PublishPipelineIntegrationTests.cpp` main 側で `runEqDirectDriveAttribution()` の代わりに（または追加で）`runOversamplerDirect()` を呼ぶ配線を commit

**CLI 注記**: `--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは存在しない。`[OS_DIRECT]` は **AudioEngineHarness.exe 既定実行（argc==1）**で出力される（N1）。

### 1.2 O-2: 台帳更新（`doc/work113/residual_tasks_20260919.md` 変更分）
記録操作として **O-1 とは独立に commit** する案を推奨（変更の性質が異なるため）。

### 1.3 O-3: `ConvoPeq.md`（生成物）
- tracked ファイル（`git ls-files` 確認済み）。working tree 版は `Generated: 2026-09-20 07:00:33` で **test-only 計装追加により stale**
- HEAD 版は commit `06794615`（2026-09-20 01:50:45）同梱
- **方針: commit しない**（監査用一時ファイル・ユーザー運用。既存方針どおり）。再生成は `python output_sourcecode_markdown.py`

### 1.4 O-4: `Testing/Temporary/CTestCostData.txt`（` D` 状態）
**現状維持**（ユーザー判断: 勝手に `git checkout` しない）。戻す場合は `git checkout -- Testing/Temporary/CTestCostData.txt`

### 1.5 O-5: `.opencode/opencode.json`（未追跡）
作成者・目的不明。**触らない**（削除も commit もしない）

### 1.6 O-6: push（`c4a08171` + `8f127bfe`、ahead 2 / behind 0）
**未実施のままユーザー承認待ち**

### 1.7 O-7: `AGENTS.md`（MIMO Desktop パイプライン運用メモ・未 commit）
` M` 状態。環境運用のみ・production DSP / テスト判定に影響なし。**触らない**（別 work item で更新するかはユーザー判断）

### 1.8 §1 全体の推奨手順
```
[Step 1.1] ユーザー判断: O-1（A/B/C 選択） + O-2（同梱 or 独立）
[Step 1.2] 該当ファイルを commit（例: "test: B-1 attribution diagnostic instrumentation ([OS_DIRECT])"）
[Step 1.3] O-3 は触らない。O-4/O-5/O-7 も触らない
[Step 1.4] O-6 push は独立運用操作としてユーザー承認後に実施
```

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定している事実（再掲・本セッションでソース＋数値の双方を再検証）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     round-trip 0.750000（up 平均 0.75 / down 1.0）★ NumPy 再現で再確認
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0) を production 引数で実測）
数学的導出  0.5×0.5 + 0.5×1.0 = 0.75
engine fit  0.98379 × 0.75^max(log2 effOS, 1) で 4 条件 ≤0.05%（doc の経験式・コード定数ではない）
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾
production 変更 0 / commit 0
```

数値再現（**本セッション・Python 3.14.7 + NumPy 2.5.2 で独立に逐語移植**）:
- 1 段（taps=31/90 dB）: baseline DC round-trip = **0.750000**、candidate（center ×2）= **0.9999999999999999**
- 3 段 IIRLike（511/140, 127/110, 31/90）: baseline = **0.4218749999999999 = 0.75³**、candidate = **0.9999999999999983**
- これらは v1.5 §12 の記録と厳密一致

### 2.2 根本原因の特定（コード追跡・行番号は 2026-09-20 実測）

#### 2.2.1 `prepareStage`（`src/CustomInputOversampler.cpp:287-390`）

```cpp
stage.centerTap = (stage.taps - 1) / 2;        // 292 行
stage.centerParity = stage.centerTap & 1;      // 293 行
stage.convParity = 1 - stage.centerParity;     // 294 行
// ...
rawCoeffs[stage.centerTap] = 0.5;              // center タップ重み = 0.5（335/348 行）
scale = 0.5 / nonCenterSum;                    // 非 center 合計を 0.5 に正規化（341 行）
// ...
stage.convCoeffs[r] = rawCoeffs[convParity + 2r];  // convParity タップ抜き出し（362-363 行）
```

**【v1.6 N2/R16 再確認】parity 実測**: 全 production taps は奇数 → centerTap = (taps−1)/2 は 15/255/63/511/127/31 → **全て奇数** → **centerParity=1 / convParity=0 が全構成で固定**。v1.5 R16 の parity 表記（up 出力: even 位置 = conv 位相 / odd 位置 = center 位相）はこの実測と一致 ✓

**FIR 合計**: center (0.5) + 非 center (0.5) = **1.0**（normalize 済み）
**convParity タップ合計**: **0.5**（half-band FIR の帰結。319-323 行で同 parity の非 center はゼロ化済みのため構造的に保証）

**【v1.6 N2 確定】taps / attenuation 対応表**（`tapsForStage` / `attenuationForStage`（CustomInputOversampler.cpp:82-106）実測）:

| stage | IIRLike taps | IIRLike atten (dB) | LinearPhase taps | LinearPhase atten (dB) |
|-------|-------------|--------------------|------------------|------------------------|
| 0 | 511 | **140** | 1023 | **160** |
| 1 | 127 | **110** | 255 | **140** |
| 2 | 31 | **90** | 63 | **120** |

`AudioEngine.Processing.Latency.cpp:22-23` の taps 表 `{511,127,31}` / `{1023,255,63}` はこれと一致 ✓（attenuation 列は Latency.cpp には存在しない）

#### 2.2.2 `interpolateStage`（`src/CustomInputOversampler.cpp:492-568`）

```cpp
double centerValue = stage.centerCoeff * history[idx - stage.centerDelayInput];  // 545 行
// ...
double convValue = Σ stage.convCoeffsReversed[r] * xWindow[r];  // = 0.5（524/531-540 行）
// ...
convValue *= 2.0;                                // ← ★ conv 位相のみ ×2 補正（557 行）
output[outBase + stage.convParity]   = convValue;    // even 位置 = 1.0 × input（convParity=0）
output[outBase + stage.centerParity] = centerValue;  // odd 位置  = 0.5 × input（centerParity=1）
```

**up の出力**: 入力 → 2 サンプル・全 production taps で parity 固定:
- even 位置（conv パス）: 0.5 × 2 = **1.0**
- odd 位置（center パス）: **0.5**
- 平均 = **0.75** / 合計 = 1.5

#### 2.2.3 `decimateStage`（`src/CustomInputOversampler.cpp:570-723`）

```cpp
double acc = stage.centerCoeff * centerSample;     // 0.5 × history[base - centerTap]（658 行）
// ...
acc += Σ coeffs[r] * history[base - convParity - 2r];  // stride-2 FIR = 0.5 × history[even]（672-676/686-688 行）
// ...
output[n] = acc;                                  // ← ×2 補正なし
```

**down の出力**: 入力 → 1 サンプル: center 対偶 0.5 + conv 対偶 0.5 = **1.0**

#### 2.2.4 round-trip 整合式

```
up:   even=1.0 (conv パス ×2) + odd=0.5 (center パス) → 平均 0.75
down: center=0.5（odd 側読み）+ conv=0.5（even 側読み）= 1.0
round-trip = 0.75 × 1.0 = 0.75
```

**多段**（ratio 8 = 3 stages）: round-trip_3 = 0.75³ = 0.421875 ≒ −7.5 dB
**SoftClip 局所 OS** も `prepareSingleStage(31, 90.0)` で同一経路を通るため 0.75

### 2.3 修正案の比較（確定）

| 案 | 内容 | 評価 |
|----|------|------|
| A | `decimateStage` に ×2 追加 | **不採用**: round-trip = 1.125 で過剰補正 |
| B | `interpolateStage` の ×2 削除 | **不採用**: round-trip = 0.375 で過小 |
| C | `prepareStage` を center=1.0 / non-center=0.0 に変更 | **棄却**（ユーザー監査で誤り確定）: `h[n] = δ[n-center]` の pure delay 化 → `convCoeffs = 0` → zero-stuffing 構造・anti-alias/anti-imaging 喪失 |
| D | A+B 統合 | **不採用**: offset 残り・`isSymmetricUpDown=true` の意味的整合に難 |
| **E** | **interpolateStage の両 polyphase に ×2**（centerValue *= 2.0 追加） | **有力仮説として採用**（詳細 §2.3.2） |

#### 2.3.2 案 E（有力仮説として採用）

**案 E: polyphase gain convention 対称化** — 既存 half-band FIR を維持し、`interpolateStage` の **両 polyphase に ×2 を適用**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内（557 行付近）
// 現行: center phase ×1, conv phase ×2
// 修正後: center phase ×2, conv phase ×2（対称化）
double convValue = ...           // = 0.5 × input（half-band FIR convParity タップ合計）
double centerValue = ...;        // = 0.5 × input（center タップ）
convValue *= 2.0;                // 既存（557 行）
centerValue *= 2.0;              // ★ 新規追加（1 行）
```

**案 E 採用根拠**（v1.5 から変更なし・本セッションで再検証）:
1. **設計意図との整合**: `isSymmetricUpDown = true`（CustomInputOversampler.h:22）と `AudioEngine.Processing.Latency.cpp:6-8` の static_assert（「symmetric linear-phase FIR with identical up/down taps」前提）は、up/down が同一の gain convention を持つことを暗に要求
2. **業界標準との整合**: JUCE `dsp::Oversampling` は polyphase buffer コピー時に ×2 相当（juce_Oversampling.cpp:185 付近）、fundsp 等でも両位相 ×2 が標準
3. **down 側は既に unity**: decimator は既に DC gain 1.0 で、up 側だけが 0.75 に非対称
4. **最小変更**: 1 行追加のみで既存 FIR 係数・構造・メモリ配置は不変

### 2.4 設計時に検討すべき 8 観点（確定・N2/N4 で数値を精密化）

| # | 観点 | 現状評価 | Phase 0 での確認 |
|---|------|----------|------------------|
| 1 | `decimateStage()` 側の補正 | 案 A として棄却（1.125 過剰） | — |
| 2 | up/down の gain convention 再設計 | 案 E で対称化 | Phase 0-2/0-3 |
| 3 | `isSymmetricUpDown` の意味 | **FIR 係数列は不変** → normalized shape 不変・絶対振幅は変化・phase/group delay は理論上不変 | Phase 0 で絶対振幅と shape を分離実測 |
| 4 | `AudioEngine.Processing.Latency.cpp` | **1 段あたり往復 `(taps−1)` サンプル（stage レート・up+down 合算）**。base レート換算は `×(baseRate/stageRate)`。コード実装どおり（`groupDelaySamplesAtStageRate = taps[stage] - 1`、:30） | Phase 0 判定基準 F で impulse 実測 |
| 5 | 既存 calibration / rigcheck（窓 `[0.486,0.496]`） | 再校正必要（後段・HOLD） | gain +2.5 dB/段・Phase 3-C/D で分割 commit |
| 6 | SoftClip 局所 OS（effOS==1）と主 OS 局所（effOS>1）の両方 | 同一修正で両者対応 | `prepareSingleStage(31, 90.0)` も同経路（Phase 0 判定基準 I） |
| 7 | Float host / Double host 経路 | **oversampler 自体は double 専用**。float host は float→double 変換後に同一 double OS を通す（DSPCoreFloat.cpp:252-263 で AudioBlock<double> 変換を実測確認 ✓） | float-host vs double-host の end-to-end equivalence として検証（Phase 0-6） |
| 8 | 既存 regression への波及 | 全 regression 期待値要再校正（HOLD） | work57 の NUC→OutputStage gain、`kOutputHeadroom` 等、Phase 3 まで保留 |

### 2.5 段階リリース設計（v1.6・authoritative source = CMake option・配置確定）

```
[Phase 0] Characterization（read-only・production src/ 変更 0）
  - 構造は §2.6（Phase 0-0 〜 0-6 + GATE）
  - Baseline（現行・record-only）・ Shadow Candidate E（test-only reference・N6 で配置確定）・ Differential
  - 全 PASS → Phase 1 eligibility

[Phase 1] Feature Flag 導入（compile-time・破壊的なし）
  - 【v1.6 N11 確定】CMakeLists.txt 冒頭（既存 option 群:40 近傍）に追加:
    ```cmake
    option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
           "Correct polyphase gain convention (B-1 案E)" OFF)
    add_compile_definitions(
        CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
    ```
    配置位置の根拠: `juce_add_gui_app(ConvoPeq …)`（:1062）より前 →
    ConvoPeq（:1062 作成・:1282 で CONVOPEQ_ALL_SOURCES・:1196 に CustomInputOversampler.cpp）
    と AudioEngineHarness（:1898 作成・:1890-1894 で ALL_SOURCES 派生）の双方に供給。
    JUCE モジュール（add_subdirectory(JUCE) :1043）にも伝播するが同 cpp を含まず無害
  - in-source は safety net のみ（authoritative source は CMake）
    ```cpp
    // src/CustomInputOversampler.cpp 冒頭（include 後）
    #ifndef CONVOPEQ_CORRECT_POLYPHASE_GAIN
    #define CONVOPEQ_CORRECT_POLYPHASE_GAIN 0   // safety net・通常は CMake option が供給
    #endif
    // interpolateStage() 内
    #if CONVOPEQ_CORRECT_POLYPHASE_GAIN
        centerValue *= 2.0;
    #endif
    ```
  - 既定 OFF（既存挙動維持）。Runtime flag は **不採用**（RT path の毎 sample/毎 block 分岐を避ける）
  - Phase 2 で `cmake -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` により rebuild（authoritative source が Phase を通じて移動しない）

[Phase 2] Flag ON で限定検証
  - `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` で rebuild
  - 再帰 dogfood: ratio 1/2/4/8 × preset IIRLike/LinearPhase 全段検証
  - Phase 0 の Shadow Candidate 測定値と**直接比較**（同一測定系・全周波数点で一致すること。帯域端の非一様差分を含め ±0.02 dB 以内を目安）
  - rigcheck 判定窓・経験式 0.75^N の再判定（必要なら Phase 3-C/D で対応）

[Phase 3] 全段 ON（破壊的変更・commit 分割）
  - **Phase 3-A**: CMake option の既定値を ON に変更（**calibration は一切変更しない** 1 commit）
  - **Phase 3-B**: 新 production gain を実測（characterization report のみ・commit なし）
  - **Phase 3-C**: calibration 値のみの commit（`kOutputHeadroom` 等が必要な場合のみ。oversampler 修正と混在させない）
  - **Phase 3-D**: rigcheck 判定窓 `[0.486, 0.496]` 更新のみの commit
  - 全テスト + 新 rigcheck 窓で PASS を確認

[Phase 4] flag 削除（次メジャーリリース）
  - `CONVOPEQ_CORRECT_POLYPHASE_GAIN` option / macro / `#if` 分岐を削除し修正行を恒久化
```

**重要（v1.6 でも維持）**:
- `0.98379 × 0.75^N` は `residual_tasks_20260919.md` の**経験式**であり、production コードに `0.98379` / `0.75^N` / `effOS` 定数は**存在しない**。`kOutputHeadroom = 0.8912509381337456`（−1.0 dBFS・`DSPCoreDouble.cpp:593` / `NoiseShaperLearner.cpp:18`・両所で実測確認 ✓）は別定数。Phase 3 の「0.75^N 削除」は **rigcheck 判定窓・監査記録・経験式記述の更新**と読み替える（HOLD）
- architectural boundary: 案 E は `interpolateStage()` 内の素朴な数値変更であり、Publish / RuntimeWorld / Retire / Crossfade / ISR bridge / lifetime には触れない（Practical Stable ISR Bridge Runtime の原則と整合）

### 2.6 Phase 0 characterization（v1.6・production src/ 変更 0・GATE 精密化）

Phase 0 を以下の 7 段構造に再構築する（v1.5 の Phase 0-0〜0-6 を維持・N4/N5/N6 で精密化）:

```
Phase 0-0  source/contract audit
    DESIGN-CONTRACT-A を確定する（測定ではなく文書化作業）
    ─────────────────────────────────────────────────────
    DESIGN-CONTRACT-A: Oversampler は interpolation convention として
    両 polyphase 位相が同一 DC gain（= 2 倍対称の gain convention）を持ち、
    up/down round-trip の DC/passband gain が unity であることが要求される。
    根拠:
      (1) interpolation-by-2 の標準 convention（zero-insert で振幅 2 倍 →
          polyphase 全体を ×2 convention に合わせる）
      (2) `CustomInputOversampler.h:21-22` の isLinearPhaseFIR / isSymmetricUpDown
          宣言と `AudioEngine.Processing.Latency.cpp:6-8` の static_assert が
          「対称」を前提とした契約を謳っている
      (3) down 側 half-band decimator は既に DC gain 1.0（up だけ 0.75 で非対称）
      (4) JUCE `dsp::Oversampling` の polyphase 実装パターンとの比較（文献調査）
    結論: baseline（0.75）が contract（1.0）に対する不整合 = defect の根拠資産となる

Phase 0-1  P0-BL: Baseline characterization（record-only・PASS/FAIL なし）
    production 無変更のまま現行実装を測定し記録する
    期待（sanity gate・一致しなければ測定系の誤り）:
      DC round-trip = 0.75 / 0.5625 / 0.421875（ratio 2/4/8・preset 非依存・±1e-6）

Phase 0-2  P0-CAND: Shadow Candidate E characterization
    【v1.6 N6 確定】production `src/` 変更 0 のまま、test-only reference implementation
    `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`（header-only）に
    prepareStage 同等の係数生成 + interpolate/decimate を複製し `centerValue *= 2.0` を適用。
    AudioEngineHarness からのみ include（production 0 変更の GATE を維持）
    期待: DC round-trip = 1.0 ± 1e-6（全構成）
    注: 本セッションの NumPy 再現で本値は既に予備確認済み（§12）。
        C++ test-only 実装で production と同一コード経路（juce::FloatVectorOperations 等）
        を通して正式化する

Phase 0-3  frequency transfer characterization（stage-rate 軸）
    ── 周波数軸の定義（v1.5 R4 維持）──
    [A] Full-chain round trip（入力帯域内・Fs_in 基準）
        50 Hz / 1 kHz / 10 kHz / 0.25 Fs_in / 0.45 Fs_in / 0.49 Fs_in
        ※ 入力は 0.5 Fs_in 超の tone を持てない。旧「0.55〜0.95 Fs_in input sweep」
          は不正（alias した成分としてしか観測されない）→ 廃止
    [B] Stage-local interpolation（各 stage・Fs_stage 基準）
        0.01 / 0.10 / 0.25 / 0.45 / 0.49 / 0.50 / 0.51 / 0.55 Fs_stage
        （ratio 8 なら stage 0/1/2 の 3 本を独立測定）
    [C] Stage-local decimation（高レート側から・Fs_stage 基準）
        0.50 〜 0.95 Fs_stage の sweep（ここで初めて alias rejection を
        意味のある形で測定できる）

    ── 【v1.6 N5】P0-C passband ripple の定義（2 段定義）──
    ripple は 2 点測定ではなく、passband sweep の |H(f)|dB の max−min とする:
      (a) 中央領域 [0.01, 0.30] Fs_in: candidate の ripple ≤ 0.2 dB（PASS 条件）
      (b) 全域 [0.005, 0.45] Fs_in: ripple の絶対値は記録のみ（PASS/FAIL なし）
    根拠（本セッション NumPy 実測）: 単段 31 taps/90 dB でも
      baseline ripple ≈ 5.24 dB / candidate ≈ 3.61 dB（全域・端点含む）
    candidate は baseline より小さいが、帯域端で単段でも ~3 dB 級の落ち込みは残る
    （PR ではない）。中央領域でのみ厳密判定とする
```
```
Phase 0-4  block/reset characterization（P0-H 具体化）
    同一 input stream を以下の block 分割で処理し output waveform を比較:
      A: 4096 × 1
      B: 1024 × 4
      C: 256 × 16
      D: 513 + 777 + 1000 + … の非一様分割
    加えて reset() 後の初回 block と continuous processing の差を確認。
    （CustomInputOversampler は upHistory/downHistory を保持し、reset()/
      clearAllStages() がこれらを clear する — 実測実装確認済み: 567/610 行）

Phase 0-5  SoftClip local OS characterization
    prepareSingleStage(31, 90.0) について以下を分離:
      I-a: round-trip DC gain / sine amplitude（candidate = 1.0 が gate）
      I-b: SoftClip 動作点: 案 E は SoftClip へ入る信号レベルを +2.5 dB 相当
           変え得るため、「oversampler が unity」と「SoftClip の動作点が意図位置」
           は別問題として扱う。upPeak / downPeak / 閾値への到達度を
           baseline / candidate 双方で記録（閾値再校正は Phase 3-C の役割）

Phase 0-6  float-host / double-host end-to-end equivalence
    （レビュー① §16 参照。oversampler 自体は double 専用）
      float host: float → double 変換 → double DSP / oversampler → float output
      double host: double → double DSP / oversampler → double output
    の end-to-end equivalence を測定（tolerance は GATE 表参照）
```

#### 2.6.1 B-1-P0 GATE（v1.6・絶対値 + 差分の 2 系統 gate・N4/N5 精密化）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT**（絶対値）と **Candidate vs Baseline の差分帰属**（Differential）の 2 系統で判定する:

| ID | 判定対象 | 判定条件（数値 tolerance 付き） | 合否 |
|----|----------|--------------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が文書化・承認済み | ✅ PASS / ❌ FAIL |
| **G-BL** | Baseline sanity | 現行 DC round-trip = 0.75^N（±1e-6・測定系 sanity。逸脱時は Phase 0 を中断し測定系を見直す） | ✅ PASS / ❌ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成） | ✅ PASS / ❌ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB** | ✅ PASS / ❌ FAIL |
| **P0-C** | Passband | Candidate: **[0.01, 0.30] Fs_in** sweep の **ripple ≤ 0.2 dB**（v1.6 N5 で中央領域に限定）+ [0.005, 0.45] 全域の ripple 絶対値を記録 | ✅ PASS / ❌ FAIL |
| **P0-C'** | Differential（帰属） | 【v1.6 N4】Candidate/Baseline 比が **f ≤ 0.45 Fs_in** で **(4/3)^N ± 0.05 dB** に一致（center-phase ×2 のみで説明可能）。NumPy 実測で DC/0.10/0.25/0.45 Fs_in は ±0.001 dB 以内で一致確認済み。0.49 Fs_in は差分値を記録（+0.04 dB 偏差・帯域端 alias 混合のため PASS 条件から除外）し、Phase 2 flag-ON 測定が shadow 値と **±0.02 dB** で一致すること | ✅ PASS / ❌ FAIL |
| **P0-D** | Stopband（stage-local） | Stage-local decimation 0.50〜0.95 Fs_stage sweep: Candidate の attenuation が Baseline と **同測定系で差 ≤ 0.1 dB** | ✅ PASS / ❌ FAIL |
| **P0-E** | Image / alias（stage-local） | Stage-local interpolation の image band（0.5 Fs_stage 近傍） rejection の Candidate/Baseline 差 ≤ **0.1 dB** | ✅ PASS / ❌ FAIL |
| **P0-F** | Latency | **測定方法を明示**: impulse 応答の peak 位置が `Σ (taps[s]−1) × (baseRate/stageRate)` と一致（許容 **±1 sample**）。group delay centroid も cross-check（±0.05 sample 目安）。static_assert は設計前提の宣言であり latency contract の PASS とは**別物**であることを明記 | ✅ PASS / ❌ FAIL |
| **P0-G** | Float / Double host | Float host 経路（float→double 変換後）と Double host 経路の end-to-end 出力が **float 量子化誤差内（目安 ≤ −120 dBFS 相当）** で振る舞い equivalent | ✅ PASS / ❌ FAIL |
| **P0-H** | Block / reset boundary | §2.6 Phase 0-4 の 4 分割すべてで出力が **double 経路で bitwise 同等（または相対誤差 ≤ 1e-12）**・reset() 後初回 block と continuous の差が規定内 | ✅ PASS / ❌ FAIL |
| **P0-I** | SoftClip local OS | I-a: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, centerTap=15, centerParity=1, convParity=0 確認済み）で Candidate DC = 1.0 ± 1e-6。I-b: SoftClip 動作点影響（+2.5 dB 相当）を upPeak/downPeak/閾値到達度で記録 | ✅ PASS / ❌ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計に方針変更。

### 2.7 Phase 0 測定対象（v1.5 維持・stage-rate 分離）

```
[A] Full-chain round trip（Fs_in 基準・入力帯域内のみ）
      50 Hz / 1 kHz / 10 kHz / 0.25 Fs_in / 0.45 Fs_in / 0.49 Fs_in

[B] Stage-local interpolation（Fs_stage 基準・stage ごと）
      0.01 / 0.10 / 0.25 / 0.45 / 0.49 / 0.50 / 0.51 / 0.55 Fs_stage
      （ratio 8 は stage 0/1/2 の 3 本を独立測定）

[C] Stage-local decimation（Fs_stage 基準・高レート側から）
      0.50 〜 0.95 Fs_stage の sweep
```

Input Nyquist（0.5 Fs_in）と stage rate（2^k × Fs_in）と image band の各軸を**分離**することで、案 E の「FIR 係数列不変 → normalized shape 不変」仮説を実験的に検証する。

### 2.8 教訓・監査記録（v1.6・N9 で補強）

1. **oracle の「教科書の答え」は現行実装と一致しない場合がある**: C 案は half-band FIR の数学的教科書条件としては正しいが、既に half-band FIR で構築された実装に適用すると pure delay に退化する
2. **「gain convention 不整合」と「FIR 構造不整合」を分離する**: 修正は polyphase 配列の対称化に限定し、FIR 構造は維持する
3. **engine fit の `0.75^N` は即時削除不可（HOLD）**: 経験式であり production 定数ではない。Phase 0/3 で再判定
4. **Phase 0 を必ず先行**: Baseline / Candidate / Contract / Differential の 4 構造で、「defect が意図された convention か」を確定する。**測定だけでは design intent は証明できない**（契約文書・既存テスト・calibration・基準実装との照合が必要）
5. **「案 E = 仮説」**: 「正しいことが証明された修正」ではなく「gain imbalance を最小変更で修正する有力仮説」
6. **「perfect reconstruction」の表現に注意**: DC gain 1.0 のみ現状確定。帯域端（0.45 Fs_in 近傍）の再構築挙動は変わる（NumPy 予備測定で確認）
7. **「anti-aliasing 維持」はコード変更のみから断定不可**: Phase 0-D/E の stage-local 実測で確認
8. **feature flag は compile-time gate（authoritative source = CMake option）**: RT path の runtime 分岐を避け、flag の真実の情報源をビルド設定に固定する
9. **姉妹実装の gain convention 差異（v1.5・コード確認済み・本セッション再確認 ✓）**: `src/TruePeakDetector.cpp:284-318` の `interpolateStage` は **even/odd とも `cCoeff × sample + conv dot` で ×2 補正なし**。TruePeakDetector は True Peak 計測専用経路（gain OS を経由しない）のため影響範囲外。**ただし TruePeakDetector の convention を案 E の正解とみなすことは避ける**（別設計目的）。Phase 0 の参考測定として round-trip DC gain を実測・記録するか、別 work item として残留
10. **【v1.6 N9】validator error contract**: `RuntimePublicationValidator` は 4 段階それぞれに固定の errorMessage 文字列を返す（RuntimePublicationValidator.cpp:16/24/32/40: "Semantic consistency check failed" / "Topology validation failed" / "Resource availability check failed" / "Conflicting transitions detected"）。**static const 文字列であり契約として使用可能**。F-1 移行時の semantic equivalence gate では enum（failureReason）+ errorMessage 文字列の両方を比較する

### 2.9 B-1-P0 GATE の適用要件（v1.5 維持・production 0 変更 gate）

```
B-1-P0 GATE
────────────────────────────────────
production src/:
    modified = 0
    staged   = 0

measurement/harness:
    test-only additions permitted
    （Phase 0-2 の test-only reference implementation を含む）

commit:
    forbidden

calibration:
    forbidden

threshold update:
    forbidden

engine-fit 0.75^N（経験式・監査記録）:
    unchanged
────────────────────────────────────
```

Phase 0 完了時の PASS/FAIL は §2.6.1 GATE 表（G-0 / G-BL / P0-A〜I）に記録する。

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない

#### 3.1.1 根本原因（2026-09-20 実測・本セッション再確認 ✓）

```cpp
// src/audioengine/AudioEngine.h:1416-1432（1425 行）
void setAutoGainStagingEnabled(bool enabled) noexcept
{
    const bool current = convo::consumeAtomic(autoGainStagingEnabled, ...);
    if (current == enabled) return;          // ← 既に同じ値なら AGC に触れない
    // ...
    getEQProcessor().setAGCEnabled(!enabled);  // ★ ON→OFF 遷移時のみ AGC が true に再設定される
    submitRebuildIntent(...);
}
```

- 既定値: `autoGainStagingEnabled { true }`（AudioEngine.h:2626 実測 ✓）
- `setEQAGCEnabled`（1306）と `getEQProcessor()`（1292）は同一オブジェクト `uiEqEditor`（EQEditProcessor）を経由（実測確認 ✓）→ `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）が staging toggle で上書きされる
- `eq` モードの呼出順（BassBuzzMeasurement.cpp:1799/1803）: `configureProbeFlatEQ` → `setAutoGainStagingEnabled(false)` → **AGC が true に再設定される**
- `eqdiag` 系（1833/1834）は逆順で最終 AGC=OFF — 修正案 A はこのパターンに揃える

#### 3.1.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A（推奨）** | 呼出順を逆に: `setAutoGainStagingEnabled(false)` → `configureProbeFlatEQ(e)` | 最小変更・eqdiag パターンと整合 |
| B | `setAutoGainStagingEnabled` から `setAGCEnabled(!enabled)` を削除 | 設計変更（Bug#4 の前提が崩れる） |
| C | `configureProbeFlatEQ` の最後に staging OFF を追加（順序固定） | コード複雑化 |

**推奨**: **A**。

#### 3.1.3 検証手順（v1.5・R12・state propagation を明示）

```
1. e.setAutoGainStagingEnabled(false)   // staging OFF（既定 true → 遷移が発生）
2. configureProbeFlatEQ(e)              // 最終的に AGC OFF を固定
3. e.setEQFilterStructure(EQProcessor::FilterStructure::Parallel)
4. e.setEqBypassRequested(false)
5. waitBacklogZero(e, 30000)            // ★ rebuild intent の完了待ち（必須）
6. sleepPump(2000)
7. gainpath 行で staging=0 eqAGC=0 を確認
8. rigcheck 測定 → ratio 再測定（窓 [0.486,0.496] 再確認）
```

- **窓 `[0.486,0.496]` は数学的に必ず不変とは言わない**: F-2 は EQ AGC の実効状態を変える修正。AGC ON 状態での既存窓整合は実測で再確認
- **B-1 の原因切り分けとは独立**: AGC を強制 OFF にした診断モードでも ratio 0.4912 が不変（監査記録済み）のため、F-2 修正は B-1 の根本原因に影響しない
- 窓境界が ±ε を超えて変動する場合は判定窓再校正を Phase 1 の前段で実施

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（v1.5・R14 再分類・本セッション再確認 ✓）

#### 3.2.1 確認事項（2026-09-20 実測）

`src/eqprocessor/EQProcessor.Processing.cpp`:
- `bypassTransitionActive` 判定: :516
- `dryCopyBase` 充填: :570-578（`bypassTransitionActive` 時のみ）
- ブレンド本体: :978-993（`canBlendDry = (dryCopyBase != nullptr)` :980、dry 混合 :993）

```cpp
if (bypassTransitionActive)
{
    const bool canBlendDry = (dryCopyBase != nullptr);
    for (sampleIndex...)
    {
        if (canBlendDry)
            wetPtr[n] = wetPtr[n] * wetGainState + dryValue * dryGain;  // ← 正常
        else
            wetPtr[n] = wetPtr[n] * wetGainState;  // ← dry 補償なし（wet-only 減衰）
    }
}
```

`dryCopyBase` は `bypassTransitionActive && dryBypassBuffer 容量十分` のときのみ充填される（570-578 行）。

#### 3.2.2 影響評価

- **定常状態の B-1（−5.11 dB）の原因ではない**（定常では `bypassTransitionActive=false`）
- 遷移時のみの潜在欠陥
- **切り分け上の注意（v1.5）**: 本件は **production behavior の欠陥**（`EQProcessor.Processing.cpp` 内）であり「test-only」ではない。本セッションでは修正しない

#### 3.2.3 修正案（将来 work item）

| 案 | 内容 | 評価 |
|----|------|------|
| A | dry バッファ確保失敗時は遷移を遅延（最大 N ms）してリトライ | 複雑・RT 影響 |
| B | dry バッファ確保失敗時は無音フェード（mute crossfade） | 聴感劣化小 |
| C | ドライコピー失敗時は事前コピー（process 開始前に確保済みに） | prepareToPlay で保証 |
| **D（推奨）** | **記録のみ**。B-1 と分離して別 work item | 既存動作変更なし・リスク最小 |

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化（本セッション再確認 ✓）

#### 3.3.1 根本原因（2026-09-20 実測）

`BassBuzzMeasurement.cpp:1744-1762`:
```cpp
e.setEqBypassRequested(true);
e.setConvolverBypassRequested(true);   // ← 1745: 一旦 ON
// ...
if (rigCheckMode == "ir")
{
    e.getConvolverProcessor().loadImpulseResponse(irFile, false);  // 1751
    // ← setConvolverBypassRequested(false) が無い → 出力は dry コピー
}
```

`irwet<digit>` モード（1763-1792）は 1778 行目で `setConvolverBypassRequested(false)` を実施済み（実測確認 ✓）。

#### 3.3.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| A | `ir` モードに `setConvolverBypassRequested(false)` を追加 | 既存窓 `[0.880,0.897]` の意味が変わる（dry → wet） |
| **B（推奨）** | `irwet<digit>` で wet 対照をカバー済みとし、`ir` は記録のみ（既存 dry 測定基準を維持） | 測定基準の意味変更なし |

**推奨**: **B**。`ir` モードの wet 有効化は将来別 work item。

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状（2026-09-20 実測・本セッション再確認 ✓）

- `src/tests/PublicationValidatorIsolationTests.cpp`: **501 行**（v1.5「519 行」は誤り・v1.6 N7 で修正）・TEST_F 34 + TEST 4 = **38 ケース** ✓
- **CMake 未登録**（ルート CMakeLists.txt `add_executable` 39 ターゲットのいずれにも該当なし。evidence 配下の 3 CMakeLists を含め全 4 ファイルに未登録）✓
- **gtest 使用は本ファイルのみ**（`#include <gtest/gtest.h>` は :1 のみ・FRIEND_TEST 0 件）✓
- `validator_.checkNoConflictingTransitions(...)` は **10 行ヒット（128/135/246/255/265/273/281/313/324/334）**、うち 128 は TEST_F 名行のため**実呼出しは 9 箇所**（v1.5「9 箇所」と一致 ✓）
- 現行 header（`RuntimePublicationValidator.h:92 private:` → **:101**）では `private` → **コンパイル不可** ✓
- `tools/build-debug.bat:29` の `--target PublicationValidatorIsolationTests` は stale（**2026-06-03 commit `85ac377c` で CMake から削除済み**・git -S で日付確認）✓
- validator の検査順序: `RuntimePublicationValidator.cpp:14-41` で **SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins** を実測確認 ✓
- errorMessage は 4 段階それぞれに固定文字列（:16/24/32/40）— **N9 で契約として使用可能と確定** ✓
- AudioEngine.h:3659 `validator_(&validator)`（ctor 初期化）→ :3670 `validator_->validatePublication(world)`（Bridge 本体の検証呼出し）✓

### 4.2 移行方針（v1.5・MIGRATE CASES THEN RETIRE・v1.6 で Phase B' 精密化）

#### Phase 0: Case Classification（機械的移植の前に実施）

38 ケースを 4 軸で分類:

| 分類軸 | 判定 |
|--------|------|
| **A. 公開 API coverage** | 公開 API（`validatePublication` / `validateSemanticConsistency` / `validateTopology` / `validateResources`）で再現可能か |
| **B. semantic equivalence** | 「同じ invalid state を作れば同じ failureReason に到達するか」（R13・レビュー① §27） |
| **C. error contract** | **error category（`ValidationFailureReason` enum）を第一優先契約**。errorMessage 文字列も static const で固定のため第二契約として併用可（v1.6 N9） |
| **D. 優先度** | `CrossfadeAuthority` 4 件（直接カバレッジ維持）＞ validator 25 件 / `CheckTransition_*` 系 9 件 |

**実測分類（v1.5 で 25/9 に修正・2026-09-20 再実測で一致 ✓）**:
- `ValidatePublication_*`（5）+ `ValidateSemanticConsistency_*`（1）+ `ValidateTopology_*`（7）+ `ValidateResources_*`（12）= **25 件**
- `CheckTransition_*`（8）+ `CheckNoConflictingTransitions_*`（1）= **9 件**
- `CrossfadeAuthorityRegressionTest`（TEST 4 件）= **4 件**

#### Phase A: `CrossfadeAuthorityRegressionTest` 4 ケース移植（最優先・API 不変）

| 移行先 | 内容 |
|--------|------|
| 既存カスタムハーネス（`AudioEngineHarness` 系・custom main・gtest 非依存） | `Decision{needsCrossfade, fadeTimeSec}` / `evaluate(old, new, policy)` の 4 ケース（DeterministicDecision / PolicyChangeChangesDecision / SameStructuralHashNoCrossfade / OversamplingChangeTriggersCrossfade） |

#### Phase B: validator 25 件 + CheckTransition 9 件の移植

- 公開 API 経由で再現可能なケースを custom main ハーネスへ移植
- `checkNoConflictingTransitions` を直接呼ぶ 9 ケースは、**間接経路**（`validatePublication` の第 4 段階として到達）で再現する

#### Phase B': Semantic Equivalence Gate（v1.5 R13・v1.6 N9 で精密化）

移行ケースごとに以下を検証:
1. **同一 invalid state を構成**したとき、**意図した failureReason に到達するか**（前段 check で先に fail しないか）
   - 例: topology 違反を意図したケースで semantic inconsistency が先に発火しないこと
2. **failureReason enum と errorMessage 文字列の両方**を比較（N9: errorMessage は static const で固定のため契約として使用可能）
3. 検査順序（Semantic → Topology → Resources → Transition・first-fail-wins）を踏まえた到達性を確認

#### Phase C: 退役

- 移行完了後に `PublicationValidatorIsolationTests.cpp` を削除
- `tools/build-debug.bat:29` の stale 参照を除去

---

## 5. §5 別課題（既存記録・本セッション対象外・v1.5 維持）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | `parseHcIdx` / `parseLcIdx`（BassBuzzMeasurement.cpp:1558/1564）と `stod`/`stoi`/`stof` は不正入力で未捕捉例外 → terminate（診断メッセージ付き fail-closed ではない）。実測で :1582-1615 の 12 箇所を確認 ✓ | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=`（:1613）は値を受理して破棄（設計上ステップ固定）。silent-ignore 系の別パターン。実測確認 ✓ | **次（極小）** |
| **B-3** | timestamp-based capture。平均実効レート binding（`out.size()/2.0s`）は非一様 callback rate を完全補正しない。現行 transition probe の目的には十分（flipIndex 誤差 <0.5% 実測） | 中 |
| **D-2** | headroom ランタイム統合（4 ランタイム混在）。uv tool `headroom-ai` 0.37.0 は extras=mcp のみで fastapi 無し → proxy 起動不可。Startup の .lnk / bat / vbs 3 起動体の整理 | 中 |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正）。M2: commit 毎に `source_revision` 変更 → COHERENCE-4 fail-closed（stamp 再作成で回避）。M1: ベアシェル/icx 起動時の cache 素字化 | 高 |

### 5.1 R-2 / R-1 の参考実装（v1.5 維持）

#### R-1: silent-ignore 修正

```cpp
else if (a.rfind("--buzz-flip-eqgain=", 0) == 0) {
    std::fprintf(stderr, "[BUZZ] INFO: --buzz-flip-eqgain=%s accepted but ignored (design-fixed -3dB step)\n",
                 a.substr(20).c_str());
    opt.flipKind = 4; opt.flipValue = 0;
}
```

#### R-2: stod/stoi 例外の try/catch

```cpp
inline int parseIntOrFail(const std::string& flag, const std::string& v) {
    try { return std::stoi(v); }
    catch (const std::exception&) {
        std::fprintf(stderr, "[BUZZ] FAIL: %s expects integer (got '%s')\n", flag.c_str(), v.c_str());
        std::exit(2);
    }
}
```

実装コスト: 極小。全 `std::stoi` / `std::stod` / `std::stof` 呼出し（実測 12 箇所）を統一

### 5.2 §5 の着手優先度（将来）

1. **R-2**（極小・fail-closed 化・即時効果）
2. **R-1**（極小・silent ignore 解消）
3. **B-3**（中・timestamp 化）
4. **D-2**（中・環境統一）
5. **D-1**（高・build identity gate 修正）

---

## 6. 推奨する実行順序（v1.6）

```
[Step 1] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・§3.1.3 の検証手順に従う）
  - F-3/F-4: 記録のみ
  - 検証: AudioEngineHarness.exe --buzz-rigcheck=eq で gainpath staging=0 eqAGC=0 を確認（N1 修正済み）

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移植 + Phase B' semantic equivalence gate（N9: enum + errorMessage 両方比較）
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-7（commit 意思決定）
  - ユーザー判断待ち
  - 案 A 採用時はエントリ配線の最小変更を含めて commit

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E）
  - Phase 0-0: source/contract audit（DESIGN-CONTRACT-A 確定）
  - Phase 0-1: Baseline characterization（record-only・期待 0.75^N）
  - Phase 0-2: Shadow Candidate E（test-only reference・N6 で配置確定）
  - Phase 0-3〜0-6: 周波数（stage-rate 軸）/ block/reset / SoftClip / float-double characterization
  - B-1-P0 GATE（§2.6.1）全 PASS → Phase 1 eligibility
  - Phase 1: CMake option 導入（compile-time・既定 OFF・既存テスト全 PASS・N11 で配置確定）
  - Phase 2: flag ON で限定検証（Phase 0 shadow 値と直接比較）
  - Phase 3-A/3-B/3-C/3-D: flag 常時 ON → 実測 → calibration-only commit → 窓-only commit
  - Phase 4: flag 削除（次メジャーリリース）

[Step 5] §5 別課題（将来 work item 化）
  - R-2（最優先・極小）→ R-1（次・極小）→ B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**。Phase 0 で契約・差分・周波数特性のいずれかが不成立の場合、案 E を見直し、案 D 統合や tap 再設計に方針変更する。

---

## 7. 検証計画（v1.6・N1 で全面修正）

### 7.1 単体検証（v1.6・実行可能なコマンドに修正）

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テストが PASS・PublicationValidatorIsolationTests 削除済 |
| §2 B-1 Phase 0/1 | `cmake --build build --config Release --target AudioEngineHarness && build\Release\AudioEngineHarness.exe` | `[OS_DIRECT] ... roundTripGain=`（flag OFF 時 ≈0.75 系・**flag ON 時 ≈ 1.0**）。DC/多周波の厳密判定は Phase 0 test-only reference implementation で実施 |
| §2 B-1 Shadow Candidate E | Phase 0-2 の test-only reference implementation 実行 | DC round-trip = 1.0 ± 1e-6（全構成） |

**注（v1.6 N1）**: `PublishPipelineIntegrationTests.exe` は **存在しない**。`PublishPipelineIntegrationTests.cpp` は `add_executable(AudioEngineHarness ...)`（CMakeLists.txt:1898）に含まれるソースであり、ターゲット/バイナリは **AudioEngineHarness**（`build\Release\AudioEngineHarness.exe` は実在確認済み）。`--buzz-rigcheck=*` / `--buzz-*` フラグは AudioEngineHarness.exe に有効。`--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは存在しない。`[OS_DIRECT]` は AudioEngineHarness.exe 既定実行（argc==1）で `PublishPipelineIntegrationTests.cpp` main → `runEqDirectDriveAttribution()` → `runOversamplerDirect()` 経由で出力される（PPIT:1116/1223, BassBuzz:1526/1532/1293）。

### 7.2 統合検証

| 検証項目 | 内容 | 期待 |
|----------|------|------|
| §2 B-1 Phase 3 | `eq` 判定窓の再校正 | 新窓での PASS（既存窓 0.4912 → 新範囲） |
| §4 F-1 Phase C | `tools/build-debug.bat` 実行 | `--target ...` エラーなし |

### 7.3 ビルド検証

```bash
# MSVC ビルド
build.bat
# icx ビルド
build-icx.bat
# AudioEngineHarness Release インクリメンタル
cmake --build build --config Release --target AudioEngineHarness
```

### 7.4 静的解析

```bash
# cppcheck（C++指定・機能欠陥0 を目標）
cppcheck --language=c++ --std=c++20 --enable=warning,performance,portability \
    src/CustomInputOversampler.cpp src/audioengine/AudioEngine.Processing.Latency.cpp

# clang-tidy（対象 TU）
clang-tidy -p compile_commands.json src/CustomInputOversampler.cpp
```

---

## 8. ロールバック計画（v1.5 維持）

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral rollback） | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild**（第一手段・production source は触らない） |
| §2 B-1（source-level rollback） | `interpolateStage` 内の `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN … #endif` ブロックと in-source safety net を削除（rebuild 必須） |
| §3 F-2 | 1 行 revert（呼出順を元に戻す） |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

※ flag 名は **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`**（CMake option）で全文統一（`kUseV2Oversampler` は採用しない）。behavioral rollback と source rollback を明確に分ける。

---

## 9. 影響度まとめ

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.5 dB 想定） | **極大**（1 行追加で済む） | 必須（Phase 0→1→2→3-A/B/C/D→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | **production behavior の潜在欠陥（記録のみ・本セッション修正なし）** | なし（本セッション） | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照

- 前スナップショット: `doc/work113/residual_tasks_20260919.md`
- 本セッション報告書: `doc/work113/residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 本文 + レビュー① 38 観点）
- 検証記録: `doc/work113/renew_plan_verification_20260920.md`（v1.3 検証確定）
- 関連監査記録:
  - `doc/work104/bass_buzz_measurement_20260917.md`
  - `doc/work105/ir_runtime_contract_and_remeasure_20260917.md`
  - `doc/audit/ConvoPeq_BugList_and_FixPlan_2026-09-13.md`
  - `doc/audit/ConvoPeq_Bug_Verification_2026-09-13.md`
  - `doc/audit/ConvoPeq_SampleRate_Reverification_2026-09-13.md`
- 関連ソース（行番号は 2026-09-20 実測・本セッションで全件再確認 ✓）:
  - `src/CustomInputOversampler.h`（isLinearPhaseFIR/isSymmetricUpDown: 21-22 ✓）
  - `src/CustomInputOversampler.cpp`（tapsForStage/attenuationForStage: 82-106 ✓ / prepareStage 287-390 ✓ / prepareSingleStage 392 ✓ / interpolateStage 492-568 ✓ / decimateStage 570-723 ✓ / convValue ×2 は 557 ✓）
  - `src/audioengine/AudioEngine.h:1416-1432`（F-2 根本 ✓）・2626（staging 既定 true ✓）・1292/1306（uiEqEditor 経由 ✓）・3659/3670（Bridge → validator_ ✓）
  - `src/audioengine/AudioEngine.Processing.Latency.cpp:6-8`（static_assert ✓）・22-24（taps 表 ✓）・30（groupDelaySamplesAtStageRate = taps[stage] - 1 ✓）
  - `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:187/260`（prepareSingleStage(31, 90.0) ✓）
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:252-263`（float→double 変換 ✓）・404/412（softClipOS up/down ✓）
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:505/513`（softClipOS up/down ✓）・593（kOutputHeadroom ✓）
  - `src/eqprocessor/EQProcessor.Processing.cpp:516/570-578/978-993`（F-3 ✓）
  - `src/audioengine/RuntimePublicationValidator.cpp:14-41`（検査順序・first-fail-wins ✓）・16/24/32/40（errorMessage 固定文字列 ✓）
  - `src/audioengine/RuntimePublicationValidator.h:92`（private ✓）・101（checkNoConflictingTransitions ✓）
  - `src/TruePeakDetector.cpp:284-318`（姉妹実装・両位相 ×2 なし ✓）
  - `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（診断モード 1744-1930 / runOversamplerDirect 1293 / runEqDirectDriveAttribution 1526 / 窓 2042-2044 ✓）
  - `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp`（main 1085 / runEqDirectDriveAttribution 前方宣言 1116 / 呼出し 1223 ✓）
  - `src/tests/PublicationValidatorIsolationTests.cpp`（F-1・501 行・38 ケース ✓）
  - `CMakeLists.txt`（option 群 :40 ✓ / CONVOPEQ_ALL_SOURCES :1133 ✓ / juce_add_gui_app(ConvoPeq) :1062 ✓ / target_sources(ConvoPeq) :1282 ✓ / add_executable(AudioEngineHarness) :1898 ✓ / CONVOPEQ_HARNESS_SOURCES :1890-1894 ✓）
  - `tools/build-debug.bat:29`（stale target ✓）

---

## 11. ユーザー判断待ち項目

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 計装 | A: 最小実行可能資産（[OS_DIRECT]+配線）/ B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-7 AGENTS.md | 触らない / 別 work item で更新 | **触らない** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E（ユーザー監査確定）** |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **Phase A→B→B'→C 全実施** |
| §2 Phase 3 の commit 分割 | 3-A〜3-D 分割 / 従来の単一 Phase 3 | **3-A〜3-D 分割（R11）** |

### §2 B-1 採用案の前提条件（v1.6）

**案 E（polyphase gain convention 対称化）は「有力仮説」であり、Phase 0 characterization（v1.6 GATE 表）が全 PASS しなければ Phase 1 実装 GO とはしない**:

1. **DESIGN-CONTRACT-A**（unity-gain requirement）が文書化・承認される（Phase 0-0）
2. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（test-only reference implementation・production 0 変更・N6 で配置確定）
3. **anti-aliasing / anti-imaging が維持**される（stage-local 実測・Phase 0-D/E）
4. **latency が不変**（impulse 実測・Phase 0-F）
5. **SoftClip 局所 OS（`prepareSingleStage(31, 90.0)`）が正常動作**（Phase 0-I）
6. **Differential（Candidate/Baseline 差分）が center-phase ×2 のみに帰属**（P0-C'・N4 で適用域を f ≤ 0.45 Fs_in に拡大）

Phase 0 で前提が崩れた場合は、案 E を見直し、案 D 統合（offset 補正）や tap 再設計を検討。

### v1.6 最終判定表

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-7 | **GO 候補** | commit/push の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替・検証手順 §3.1.3） |
| §3 F-3 | **記録継続** | production behavior の潜在欠陥（test-only ではない）・定常 B-1 の原因ではない |
| §3 F-4 | **記録継続** | 既存 dry 測定基準維持・`irwet<digit>` で wet 対照 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate（N9: enum + errorMessage 両方比較） |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO** | コード整合性は再確認済み |
| **B-1 Phase 0** | **GO（測定仕様 v1.6 適用が条件）** | Baseline/Candidate/Differential 3 層構造・N4/N5/N6/N11 で精密化 |
| B-1 Phase 1 | **HOLD** | Phase 0 全 PASS が前提 |
| B-1 Phase 2/3/4 | **HOLD** | Phase 0 → Phase 1 → Phase 2 の順次判定 |
| **B-1 案 E** | **仮説として採用** | 強い仮説だが Phase 0 で検証前は確定ではない |
| B-1 「perfect reconstruction」 | **表現修正** | DC gain 1.0 は言えるが PR 全帯域保証は未証明（N5: 帯域端で単段でも ~3 dB 級の落ち込み） |
| B-1 calibration 変更 | **HOLD** | Phase 3-C/D まで保留・分割 commit |
| `0.75^N` engine-fit 削除 | **HOLD** | 経験式（コード定数なし）・監査記録/窓の更新は Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | authoritative source = CMake・in-source `#ifndef` は safety net・N11 で配置確定 |
| TruePeakDetector 姉妹実装 | **記録継続** | 両位相 ×2 なし・案 E の正解としない・参考測定 or 別 work item |

---

## 12. 本計画書の検証エビデンス（2026-09-20 ソース監査・レビュー②）

本計画書 v1.6 は、以下の検証を 2026-09-20 に実施した結果に基づく。

1. **行番号監査（全件再実測）**: v1.5 §10 の参照一覧を production source と照合。不一致は以下の 3 件（v1.6 で修正）:
   - `DSPCoreLifecycle.cpp:188/261` → 正: **187/260**（N3）
   - `DSPCoreFloat.cpp:405/413` → 正: **404/412**（N3）
   - `PublicationValidatorIsolationTests.cpp:519 行` → 正: **501 行**（N7）
   - その他の全行番号（CustomInputOversampler.cpp:287-390/492-568/570-723/557, AudioEngine.h:1416-1432/2626/1292/1306/3659/3670, Latency.cpp:6-8/22-24/30, RuntimePublicationValidator.cpp:14-41/16/24/32/40, .h:92/101, TruePeakDetector.cpp:284-318, EQProcessor.Processing.cpp:516/570-578/978-993, BassBuzzMeasurement.cpp:1081/1293/1526/1532/1744-1762/1799/1803/1833/1834/2042-2044, PPIT:1085/1116/1223, CMakeLists.txt:40/1062/1133/1282/1890-1894/1898, build-debug.bat:29）は**実測と一致** ✓

2. **数値再現（本セッション独立実施）**: `prepareStage` → `interpolateStage` → `decimateStage` を Python 3.14.7 + NumPy 2.5.2 で逐語移植
   - 現行: 1 段 DC = **0.750000** / 3 段 = **0.4218749999999999**（=0.75³）— §2.1 の確定事項と厳密一致 ✓
   - 案 E shadow: 全構成 DC = **0.9999999999999999 / 0.9999999999999983** — Phase 0-2 の期待値の予備確認 ✓
   - Differential: DC / 0.10 / 0.25 / 0.45 Fs_in で candidate/baseline 比 = **(4/3)^N に ±0.001 dB 以内で一致**（N4 で適用域を f ≤ 0.45 Fs_in に拡大）✓
   - 帯域端 0.49 Fs_in で candidate/baseline 比が +0.04 dB 偏差（帯域端 alias 混合）— P0-C' から除外し記録のみ ✓
   - 単段 31 taps/90 dB の passband ripple: baseline ≈ **5.24 dB** / candidate ≈ **3.61 dB**（[0.005, 0.45] Fs_in 60 点 sweep）— N5 で P0-C を中央領域 [0.01, 0.30] Fs_in に限定する根拠 ✓

3. **静的解析**: cppcheck（--language=c++ --std=c++20 --enable=warning,performance,portability）を `CustomInputOversampler.cpp` に実行 → **指摘 0 件**（v1.5 §12 と同結果）

4. **git 実測**: ahead 2（`c4a08171` + `8f127bfe`）、numstat 全項目が計画書前提と一致 ✓。`ConvoPeq.md` は tracked（HEAD 版は `06794615` 同梱、working tree 版は 2026-09-20 07:00:33 で stale）✓

5. **ビルド実測**: `build\Release\AudioEngineHarness.exe` は実在 ✓、`PublishPipelineIntegrationTests.exe` は **存在しない**（N1 で §7.1 を全面修正）✓

6. **レビュー① との照合**: 38 観点のうち、7 必須項目をすべて採用（v1.5 R1〜R7）、追加指摘 R8〜R15 を採用、技術的中核（案 E・原因特定・段階リリース骨格）はレビュー①も妥当と評価 → 維持。レビュー②（本監査）で N1〜N12 を追加

---

**本書は方針案です（v1.6・レビュー② 反映版・Phase 0 測定仕様精密化済み）。実装着手は本書承認後**。
**§2 B-1 の DSP 修正方針：案 E（polyphase gain convention 対称化）を「有力仮説」として採用**。
**Phase 0 は「現行 baseline と Shadow Candidate E を同一測定系で比較し、設計契約・周波数特性・遅延・境界挙動のすべてを検証する characterization フェーズ」である（現行 production の PASS/FAIL で案 E を判定するフェーズではない）**。
**compile-time flag は CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`（authoritative source = CMake・N11 で配置確定）・段階リリース Phase 0 → 1 → 2 → 3-A/B/C/D → 4**。
**Phase 0 全 PASS → Phase 1 eligibility。Phase 1 以降は Phase 0 結果次第で GO 判定**。
**案 E は「正しいことが証明された修正」ではなく「現行実装の gain imbalance を最小変更で修正する有力仮説」として扱う**。
**「perfect reconstruction」表現は Phase 0 で PR が数値確認されるまで使用しない**。
**承認状態: §1/§3 F-2/§4 F-1 = GO候補 / §3 F-3・F-4/§5 = 保留継続 / §2 = Phase 0 着手承認（測定仕様 v1.6 適用が条件）・Phase 1 以降は HOLD**。
