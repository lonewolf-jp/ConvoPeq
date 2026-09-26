# ConvoPeq 残件 改修計画書（2026-09-20 時点）

- **版**: v1.7（レビュー② の 4 必須項目 + G-0 契約強化を反映・数値契約を厳密確定）
- **対象**: `doc/work113/residual_tasks_20260920.md` 全残件（§1〜§5）
- **方針**: 段階リリース重視・既存動作への影響を最小化・ロールバック容易性確保
- **前提**: production `src/` の未 commit 差分は 0 件、HEAD = `8f127bfe`（docs 1 件）+ `c4a08171`（B-2 test 1 件）が未 push
- **本書は read-only 監査と方針案の提示**。実装着手は本書承認後
- **基準ソース**: HEAD の production source（`ConvoPeq.md` は working tree 版 Generated 2026-09-20 07:00:33 で参考扱い）
- **§2 B-1 修正方針**: 案 E（polyphase gain convention 対称化・既存 half-band FIR 維持）を「**有力仮説**」として採用（oracle C 案は撤回済み）
- **Phase 0 着手 GO・Phase 1 以降 HOLD**: 案 E は Phase 0 の characterization（§2.6 P0 GATE）で全 PASS しなければ Phase 1 実装 GO とはしない
- **compile-time flag**: CMake option **`CONVOPEQ_CORRECT_POLYPHASE_GAIN`** を authoritative source とする

---

## v1.6 → v1.7 改訂サマリ（レビュー② の 4 必須項目 + G-0 強化 + v1.6 内の誤り修正）

レビュー②の判定「条件付き GO（Phase 0-0〜0-6 は着手可。P0-D/E 絶対値 gate・Shadow Reference Fidelity Gate・P0-H reset contract・P0-G error metric の 4 点と G-0 の明示承認を計画書に追記すれば承認可能）」を受け、4 点 + G-0 を本 v1.7 に**実装可能な仕様として**追記した。あわせてレビュー②自身が確認した v1.6 の数値が、本セッションの独立再実装（閉形式モデル）と 4 桁一致することを確認し、v1.6 が混入させた 2 件の行番号・行数誤りを修正した。

| ID | 区分 | 内容 | v1.7 での確定 |
|----|------|------|----------------|
| **R2-1** | 必須①（P1） | P0-D/E に**絶対値 floor** が無い（differential だけでは「Baseline 自体が十分良い」を確認できない） | §2.6.1 P0-D/E を **differential + absolute floor の二段**に変更。floor は本セッションで数値確定（FIR 設計減衰 −10 dB・transition 領域は記録のみ）。実測表を §2.7.2/2.7.3 に収録 |
| **R2-2** | 必須②（P1） | Shadow Reference の **baseline fidelity gate** が無い（reference 実装差を案 E の FAIL と誤認するリスク） | **Phase 0-1b: REF-FIDELITY gate を新設**（Shadow Reference baseline モード == production、DC/impulse/周波数/ratio/preset/partition/reset、double 経路 bitwise 一致）。Phase 0-2 の**前提条件**に設定 |
| **R2-3** | 必須③（P2） | P0-H の reset 判定「規定内」が未定義 | **reset contract を具体化**: `fresh instance + block#1` vs `stream → reset() → block#1` の**bitwise 一致**。partition invariance も分割 A/B/C/D の bitwise 一致に固定。処理容量（maxInputBlockSize）制約を明記 |
| **R2-4** | 必須④（P2） | P0-G の「≤ −120 dBFS 相当」が曖昧 | **サンプル値ベースの数値条件に変更**: maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8（float 仮数 24 bit からの導出を明記）。same input / same partition / same reset state を固定 |
| **R2-5** | G-0 強化 | `isSymmetricUpDown`・static_assert からの**推論ではなく**、プロジェクト DSP design contract として**明示承認**する | DESIGN-CONTRACT-A を「承認対象の決定事項」として明記。根拠を E1〜E5（下記 §2.3.1）に列挙。特に **E4: baseline の in-band image rejection = −9.54 dB 一定（本セッション実測）は、FIR 自身の設計減衰 −87〜−159 dB に対し ~80〜150 dB の開きがあり、defect の客観証拠** |
| **R2-6** | 根拠補強 | P0-C の「ripple ≤ 0.2 dB」の出典が不明確 | **理論 ripple ≈ 0.001 dB（本セッション実測）→ gate 0.05 dB（理論値×50）** に変更。per-config の 0.1 dB passband edge（S1 0.3527 / IIR3 0.4888 / LP3 0.4940 Fs_in・実測）を「passband の客観定義」として併記。[0.005, 0.45] 全域は記録のみ（transition 落ち込みを ripple と混同しない） |
| **R2-7** | 数値精密化 | P0-C' の適用域「f ≤ 0.45 Fs_in」は**全構成に一律適用できない** | 閉形式モデル実測: **multistage（IIR3/LP3）では 0.45 Fs_in まで偏差 ±0.0002 dB**（v1.6 の NumPy と一致・0.49 で +0.043 dB）だが、**単段 31 taps 構成では 0.45 で +1.64 dB・0.49 で +3.39 dB 偏差**。→ **P0-C' の適用域を per-config 化**（IIR3/LP3: f ≤ 0.45 / S1: f ≤ 0.30） |
| **R2-8** | 誤り修正 | v1.6 の stage-local decimation sweep「0.50〜0.95 Fs_stage」は **Nyquist（0.5 Fs_stage）超**の不正な軸（レビュー① §8 と同種の誤り） | **正しい軸に修正**: stage-local decimation の stopband = **[transition_end, 0.5] cycles/sample（= [2×transition_end…1.0] × Fs_stage/2 として表記可）**。測定は own-rate の cycles/sample に統一。per-design の transition_end を実測表で付与 |
| **R2-9** | 誤り修正 | v1.6 の N3「§10 正: 187/260・DSPCoreFloat 404/412」は**off-by-one の誤り**（diagLog 行を指していた） | 本セッション実測で復元: **DSPCoreLifecycle.cpp:188/261**（`softClipOS.prepareSingleStage(31, 90.0, internalMaxBlock)` 本体行）・**DSPCoreFloat.cpp:405/413**（processUp/processDown 本体行） |
| **R2-10** | 誤り修正 | v1.6 の N7「PublicationValidatorIsolationTests.cpp は 501 行」は誤り | 本セッション実測: **519 行**（`wc -l`）。TEST_F 34 + TEST 4 = 38 は一致 ✓。テスト分類も再実測: **ValidatePublication 7 / ValidateSemanticConsistency 1 / ValidateTopology 6 / ValidateResources 11 = 25、CheckTransition 8 / CheckNoConflictingTransitions 1 = 9**（v1.6 の 5/1/7/12 は誤り） |
| **R2-11** | 追加 | Phase 3-B「新 production gain を実測」が 1 ステップでは non-linear 動作点（SoftClip 閾値・limiter onset・NUC・output makeup）の変化を分離できない | **3-B1（production gain characterization）→ 3-B2（nonlinear operating-point characterization）に分割**。calibration 変更（3-C）は両方の完了後のみ |

**レビュー②で変更しない点（再確認済み）**:
- 案 E の技術的内容（`centerValue *= 2.0` 1 行追加）: 本セッションの独立再実装でも DC round-trip = 1.0 を確認 → 維持
- Phase 0 の Baseline / Candidate / Differential 3 層構造・段階リリース Phase 0→1→2→3→4 骨格: 維持（R2-1〜R2-8 で精密化）
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
2. **§2 B-1** → Phase 0（characterization）→ 1（flag 導入）→ 2（flag ON 限定検証）→ 3-A/B1/B2/C/D（全段 ON + 分割 commit）→ 4（flag 削除）
3. **§3 F-2** → harness cleanup（test-only）。B-1 とは独立して先に着手可能。F-3/F-4 は記録のみ
4. **§4 F-1** → テスト移行（custom main ハーネスへ）。`CrossfadeAuthority` 4 件を最優先
5. **§5 別課題** → 既存記録のまま保留。必要時に別 work item 化

---

## 1. §1 未 commit / 未 push 対応（O-1〜O-7）

### 1.1 O-1: B-1 計測用 test-only 計装（+642 行）

| 項目 | 内容 |
|------|------|
| ファイル | `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（+634/−2）、`PublishPipelineIntegrationTests.cpp`（+8/−0）。git diff --stat HEAD 実測で一致 ✓ |
| 内容 | `eqdiag` / `eqdiagser` / `eqos<digit>` / `irwet<digit>` 診断モード、`gainpath` 行、`[EQ_DIRECT]` / `[OF_DIRECT]` / `[OS_DIRECT]` 直接駆動測定 |
| 影響 | test-only。production `src/` 変更なし |
| 判定窓 | 既存 `eq` 判定窓 `[0.486, 0.496]` は不変（`eqIdentityMode = (rigCheckMode == "eq")` — BassBuzzMeasurement.cpp:2042、窓本体 2043-2044 実測 ✓） |
| 出力キー | `[OS_DIRECT] ... roundTripGain=`（`round-trip=` ではない） |
| エントリ | `runOversamplerDirect()`（BassBuzzMeasurement.cpp:1293）は `runEqDirectDriveAttribution()`（同:1526）から呼ばれ（同:1532）、`PublishPipelineIntegrationTests.cpp` main から**無条件呼出し**（前方宣言 :1116・呼出し :1223、+8 diff と一致 ✓） |

**3 案**:

| 案 | 内容 | メリット | デメリット |
|----|------|----------|------------|
| **A（推奨）** | **「B-1 regression detection に必要な最小実行可能資産」を commit**（`[OS_DIRECT]` 本体 = `runOversamplerDirect()` + main からの直接呼出し配線） | 回帰検出を最小コストで残せる | `eqdiag` / `irwet` 診断モードは手元検証用に残る。2 ファイル双方に触れる |
| **B** | 全 +642 行を 1 commit | 将来の B-1 再発の即時検出資産として完全保存 | 行数大・レビュー負荷大 |
| **C** | すべて退役（破棄） | 後方互換問題なし | 再発時の解析遅延 |

**推奨案**: **A**。`[OS_DIRECT]` / `[EQ_DIRECT]` / `[OF_DIRECT]` は同一エントリ関数 `runEqDirectDriveAttribution()` に束ねられており `git add -p` での分離は不可能。案 A の実体:
1. `runOversamplerDirect()`（1293）とその呼出し必要分を commit
2. `PublishPipelineIntegrationTests.cpp` main 側で `runEqDirectDriveAttribution()` の代わりに（または追加で）`runOversamplerDirect()` を呼ぶ配線を commit

**CLI 注記**: `--buzz-osdirect=2` 等の ConvoPeq.exe CLI フラグは存在しない。`[OS_DIRECT]` は **AudioEngineHarness.exe 既定実行（argc==1）**で出力される。

### 1.2 O-2: 台帳更新（`doc/work113/residual_tasks_20260919.md` 変更分）
記録操作として **O-1 とは独立に commit** する案を推奨（変更の性質が異なるため）。

### 1.3 O-3: `ConvoPeq.md`（生成物）
- tracked ファイル。working tree 版は `Generated: 2026-09-20 07:00:33` で **test-only 計装追加により stale**
- **方針: commit しない**（監査用一時ファイル・ユーザー運用）。再生成は `python output_sourcecode_markdown.py`

### 1.4 O-4: `Testing/Temporary/CTestCostData.txt`（` D` 状態）
**現状維持**。戻す場合は `git checkout -- Testing/Temporary/CTestCostData.txt`

### 1.5 O-5: `.opencode/opencode.json`（未追跡）
作成者・目的不明。**触らない**（削除も commit もしない）

### 1.6 O-6: push（`c4a08171` + `8f127bfe`、ahead 2 / behind 0）
**未実施のままユーザー承認待ち**

### 1.7 O-7: `AGENTS.md`（MIMO Desktop パイプライン運用メモ・未 commit）
` M` 状態（+5/−3 実測）。環境運用のみ。**触らない**（別 work item で更新するかはユーザー判断）

### 1.8 §1 全体の推奨手順
```
[Step 1.1] ユーザー判断: O-1（A/B/C 選択） + O-2（同梱 or 独立）
[Step 1.2] 該当ファイルを commit（例: "test: B-1 attribution diagnostic instrumentation ([OS_DIRECT])"）
[Step 1.3] O-3 は触らない。O-4/O-5/O-7 も触らない
[Step 1.4] O-6 push は独立運用操作としてユーザー承認後に実施
```

---

## 2. §2 B-1: CustomInputOversampler の up/down round-trip 欠陥（最大規模）

### 2.1 確定している事実（本セッションでソース＋閉形式モデルの双方を再検証）

```
主因        CustomInputOversampler の up/down round-trip 利得欠陥
1 stage     round-trip 0.750000（up 平均 0.75 / down 1.0）
多段        0.75^N（ratio 2/4/8 = 0.75 / 0.5625 / 0.421875、preset 非依存）
SoftClip局所OS 0.75（prepareSingleStage(31, 90.0) を production 引数で実測）
数学的導出  0.5×0.5 + 0.5×1.0 = 0.75
engine fit  0.98379 × 0.75^max(log2 effOS, 1) で 4 条件 ≤0.05%（doc の経験式・コード定数ではない）
契約不整合  isSymmetricUpDown / Latency の static_assert 前提と矛盾
production 変更 0 / commit 0
```

**本セッションの独立検証（閉形式モデル）**: `prepareStage` の係数生成 → up（`even_leg[n]=2·(conv⊛x)[n]`, `odd_leg[n]=c·x[n−cdi]`）→ down（`out[n]=0.5·y[2n−cT]+Σ cc[r]·y[2n−vP−2r]`）を閉形式に落とし、DC round-trip を厳密計算:

| 構成 | baseline DC | candidate DC | 0.75^N |
|------|-------------|--------------|--------|
| 単段 31/90 | **0.750000000000** | **1.000000000000** | 0.75 |
| IIRLike 3 段（511/127/31） | **0.421875000000** | **1.000000000000** | 0.75³ = 0.421875 |
| LinearPhase 3 段（1023/255/63） | **0.421875000000** | **1.000000000000** | 0.75³ = 0.421875 |

→ v1.6 §12 の NumPy 再現値と一致（独立実装で 12 桁一致）。**さらに本モデルは v1.6 の帯域数値も再現**: 単段 ripple 5.242/3.607 dB・IIR3 の 0.49 Fs_in 偏差 +0.0430 dB（v1.6 記載「+0.04 dB / 7.5394 dB vs 7.4963 dB」と一致）。

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

**parity 実測**: 全 production taps は奇数 → centerTap = (taps−1)/2 は 15/255/63/511/127/31 → **全て奇数** → **centerParity=1 / convParity=0 が全構成で固定**。

**FIR 合計**: center (0.5) + 非 center (0.5) = **1.0**（normalize 済み）
**convParity タップ合計**: **0.5**（half-band FIR の帰結。319-323 行で同 parity の非 center はゼロ化済みのため構造的に保証）

**taps / attenuation 対応表**（`tapsForStage`/`attenuationForStage`（CustomInputOversampler.cpp:84-106）実測）:

| stage | IIRLike taps | IIRLike atten (dB) | LinearPhase taps | LinearPhase atten (dB) |
|-------|-------------|--------------------|------------------|------------------------|
| 0 | 511 | 140 | 1023 | 160 |
| 1 | 127 | 110 | 255 | 140 |
| 2 | 31 | 90 | 63 | 120 |

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

**up の出力**: even 位置（conv パス）= 0.5 × 2 = **1.0** / odd 位置（center パス）= **0.5** → 平均 **0.75** / 合計 1.5。

#### 2.2.3 `decimateStage`（`src/CustomInputOversampler.cpp:570-723`）

```cpp
double acc = stage.centerCoeff * centerSample;     // 0.5 × history[base - centerTap]（658 行）
// ...
acc += Σ coeffs[r] * history[base - convParity - 2r];  // stride-2 FIR = 0.5 × history[even]（672-676/686-688 行）
// ...
output[n] = acc;                                  // ← ×2 補正なし（717 行）
```

**down の出力**: center 寄与 0.5（odd 側読み）+ conv 寄与 0.5（even 側読み）= **1.0**

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
| B | `interpolateStage` の ×2 削除 | **不採用**: round-trip = 0.375 で過少 |
| C | `prepareStage` を center=1.0 / non-center=0.0 に変更 | **棄却**: `h[n] = δ[n-center]` の pure delay 化 → `convCoeffs = 0` → anti-alias/anti-imaging 喪失 |
| D | A+B 統合 | **不採用**: offset 残り・`isSymmetricUpDown=true` の意味的整合に難 |
| **E** | **interpolateStage の両 polyphase に ×2**（`centerValue *= 2.0` 追加） | **有力仮説として採用** |

#### 2.3.1 案 E 採用根拠（v1.7・G-0 契約として明示承認する対象）

**DESIGN-CONTRACT-A（設計契約・推論ではなく明示承認する決定事項）**:
> Oversampler は interpolation convention として両 polyphase 位相が同一 DC gain（= 2 倍密化の gain convention）を持ち、up/down round-trip の DC/passband gain が unity であることが要求される。

| 証拠 | 内容 | 出所（本セッション検証） |
|------|------|--------------------------|
| **E1** | コード自身が既に gain-2 convention を採用している（`convValue *= 2.0` — 557 行）。欠陥は convention そのものではなく **center 位相への不完全適用** | CustomInputOversampler.cpp:557 実測 |
| **E2** | down 側 half-band decimator は既に DC gain 1.0（0.5+0.5）。up だけ 0.75 → up/down 非対称は `isSymmetricUpDown` / static_assert（「identical up/down taps」）の契約前提と矛盾 | 閉形式モデルで厳密確認 |
| **E3** | **baseline の in-band image rejection = −9.54 dB 一定（全 design・本セッション実測）**。これは FIR 自身の stopband 達成値（−87.0〜−159.2 dB・後述 §2.7.2）に対し **~80〜150 dB の開き**。すなわち現行 interpolation は自身の設計減衰（`attenuationForStage` 90〜160 dB）を実現できておらず、anti-imaging が破綻している**客観的 defect 証拠**（意図・非意図の論ではなく測定事実） | §2.7.3 実測表 |
| **E4** | 対称性宣言: `isLinearPhaseFIR = true` / `isSymmetricUpDown = true`（CustomInputOversampler.h:21-22）+ `AudioEngine.Processing.Latency.cpp:6-8` static_assert | ソース実測 |
| **E5** | **外部参照（プロジェクト同梱 JUCE 実装）**: JUCE `dsp::Oversampling` の 2 倍 FIR オーバーサンプラは up 経路で入力を `buf[N-1] = 2 * samples[i];`（juce_Oversampling.cpp:185）と **×2 して格納**し、even 出力（conv 畳み込み）と odd 出力（center タップ単独: `buf[Ndiv2+1] * fir[Ndiv2]`）の**両位相に同一の ×2 が乗る**。down 経路は ×1（同 :228 `buf[N-1] = bufferSamples[i << 1];`）。すなわち「両位相同 gain convention / down は unity」という本修正の構造は JUCE 参照実装と同一 | JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp:185-196（up）, 228-240（down）実測 |

**案 E の実装**:

```cpp
// src/CustomInputOversampler.cpp interpolateStage() 内（557 行付近）
double convValue = ...           // = 0.5 × input（convParity タップ合計）
double centerValue = ...;        // = 0.5 × input（center タップ）
convValue *= 2.0;                // 既存（557 行）
centerValue *= 2.0;              // ★ 新規追加（1 行・flag gate 内）

output[outBase + stage.convParity]   = convValue;    // even 位置 = 1.0 × input
output[outBase + stage.centerParity] = centerValue;  // odd 位置  = 1.0 × input
```

**gain 変化**: 1 stage +2.4988 dB / 2 stages +4.998 dB / 3 stages +7.4988 dB（閉形式モデルで確認: candidate/baseline 比 = (4/3)^N）。

### 2.4 設計時に検討すべき 8 観点（確定）

| # | 観点 | 現状評価 | Phase 0 での確認 |
|---|------|----------|------------------|
| 1 | `decimateStage()` 側の補正 | 案 A として棄却（1.125 過剰） | — |
| 2 | up/down の gain convention 再設計 | 案 E で対称化 | Phase 0-2/0-3 |
| 3 | `isSymmetricUpDown` の意味 | **FIR 係数列は不変** → normalized shape 不変・絶対振幅は変化・phase/group delay は理論上不変 | Phase 0 で絶対振幅と shape を分離実測 |
| 4 | `AudioEngine.Processing.Latency.cpp` | **1 段あたり往復 `(taps−1)` サンプル（stage レート・up+down 合算）**。base 換算 `×(baseRate/stageRate)`（:30 実測） | Phase 0 判定基準 P0-F で impulse 実測 |
| 5 | 既存 calibration / rigcheck（窓 `[0.486,0.496]`） | 再校正必要（後段・HOLD） | Phase 3-C/D で分割 commit |
| 6 | SoftClip 局所 OS（effOS==1）と主 OS 局所（effOS>1）の両方 | 同一修正で両者対応 | `prepareSingleStage(31, 90.0)` も同経路（P0-I） |
| 7 | Float host / Double host 経路 | **oversampler 自体は double 専用**。float host は float→double 変換後に同一 double OS を通す（DSPCoreFloat.cpp:252-263 実測 ✓） | float-host vs double-host end-to-end（P0-G・数値条件は §2.6.1） |
| 8 | 既存 regression への波及 | 全 regression 期待値要再校正（HOLD） | work57 の NUC→OutputStage gain、`kOutputHeadroom` 等、Phase 3 まで保留 |

### 2.5 段階リリース設計（v1.7・authoritative source = CMake option・配置確定）

```
[Phase 0] Characterization（read-only・production src/ 変更 0）
  - 構造は §2.6（Phase 0-0 / 0-1 / 0-1b / 0-2 〜 0-6 + GATE）
  - Baseline（record-only）・REF-FIDELITY（新設）・Shadow Candidate E・Differential
  - 全 PASS → Phase 1 eligibility

[Phase 1] Feature Flag 導入（compile-time・破壊的なし）
  - CMakeLists.txt 冒頭（既存 option 群:40 近傍）に追加:
    ```cmake
    option(CONVOPEQ_CORRECT_POLYPHASE_GAIN
           "Correct polyphase gain convention (B-1 案E)" OFF)
    add_compile_definitions(
        CONVOPEQ_CORRECT_POLYPHASE_GAIN=$<BOOL:${CONVOPEQ_CORRECT_POLYPHASE_GAIN}>)
    ```
    配置位置: `juce_add_gui_app(ConvoPeq …)`（:1062）より前 →
    ConvoPeq（:1062 作成・:1282 target_sources）と AudioEngineHarness
    （add_executable :1899・CONVOPEQ_ALL_SOURCES :1890-1894 派生）の双方に供給。
    JUCE モジュール（add_subdirectory(JUCE) :1043）にも伝播するが同 cpp を含まず無害
  - in-source は safety net のみ（authoritative source は CMake）
  - 既定 OFF（既存挙動維持）。Runtime flag は不採用（RT path 分岐回避）
  - Phase 2 で `cmake -DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` により rebuild

[Phase 2] Flag ON で限定検証
  - `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=ON` で rebuild
  - 再帰 dogfood: ratio 1/2/4/8 × preset IIRLike/LinearPhase 全段検証
  - Phase 0 の Shadow Candidate 測定値と**直接比較**（同一測定系・全周波数点一致 ±0.02 dB 目安）
  - rigcheck 判定窓・経験式 0.75^N の再判定（必要なら Phase 3-C/D で対応）

[Phase 3] 全段 ON（破壊的変更・commit 分割）
  - **Phase 3-A**: CMake option の既定値を ON に変更（calibration は一切変更しない 1 commit）
  - **Phase 3-B1**: production gain characterization（線形伝達の実測・commit なし）
  - **Phase 3-B2**: nonlinear operating-point characterization（SoftClip 閾値到達度・limiter onset・
    NUC・output makeup の動作点変化を実測・commit なし）
  - **Phase 3-C**: calibration 値のみの commit（3-B1/3-B2 の両完了後のみ。oversampler 修正と混在させない）
  - **Phase 3-D**: rigcheck 判定窓 `[0.486, 0.496]` 更新のみの commit
  - 全テスト + 新 rigcheck 窓で PASS を確認

[Phase 4] flag 削除（次メジャーリリース）
  - `CONVOPEQ_CORRECT_POLYPHASE_GAIN` option / macro / `#if` 分岐を削除し修正行を恒久化
```

**重要（v1.7 でも維持）**:
- `0.98379 × 0.75^N` は `residual_tasks_20260919.md` の**経験式**であり、production コードに `0.98379` / `0.75^N` / `effOS` 定数は**存在しない**。`kOutputHeadroom = 0.8912509381337456`（−1.0 dBFS・`DSPCoreDouble.cpp:593` / `NoiseShaperLearner.cpp:18`）は別定数。Phase 3 の「0.75^N 削除」は **rigcheck 判定窓・監査記録・経験式記述の更新**と読み替える（HOLD）
- architectural boundary: 案 E は `interpolateStage()` 内の数値変更のみで、Publish / RuntimeWorld / Retire / Crossfade / ISR bridge / lifetime には触れない

### 2.6 Phase 0 characterization（v1.7・production src/ 変更 0・GATE 精密化）

```
Phase 0-0  source/contract audit（文書化作業・測定ではない）
    DESIGN-CONTRACT-A を「推論」ではなく「承認対象の決定事項」として確定する。
    根拠証拠 E1〜E5 は §2.3.1。特に E3（baseline in-band image rejection = −9.54 dB 一定 vs
    FIR 設計減衰 −87〜−159 dB）は defect の客観証拠。承認者はユーザー。

Phase 0-1  P0-BL: Baseline characterization（record-only・PASS/FAIL なし）
    production 無変更のまま現行実装を測定し記録する
    期待（sanity gate・一致しなければ測定系の誤り）:
      DC round-trip = 0.75 / 0.5625 / 0.421875（ratio 2/4/8・preset 非依存・±1e-6）
    付帯記録: in-band image rejection = −9.54 dB 一定（E3 の再確認）

Phase 0-1b REF-FIDELITY: Shadow Reference fidelity gate（v1.7 新設・R2-2）
    【目的】Shadow Reference の実装差を案 E の FAIL と誤認しない
    【手順】Shadow Reference を baseline モード（center ×1）で production と比較:
      - DC（ratio 2/4/8 × preset IIRLike/LinearPhase）
      - impulse 応答（波形そのもの）
      - 周波数応答（§2.7 の全測定点）
      - block partition（4096 / 1024×4 / 256×16 / ragged）
      - reset() 境界
    【合否】double 経路で **bitwise 一致**（不可なら相対誤差 ≤ 1e-15）。
      不一致の場合は reference 実装を修正し、一致するまで Phase 0-2 へ進まない
    【配置】Shadow Reference は `src/tests/AudioEngineHarness/PolyphaseGainCandidateRef.h`
      （header-only・AudioEngineHarness からのみ include・production 変更 0）

Phase 0-2  P0-CAND: Shadow Candidate E characterization
    Shadow Reference の candidate モード（center ×2）で測定。
    期待: DC round-trip = 1.0 ± 1e-6（全構成）
    注: 本セッションの閉形式モデルで本値は予備確認済み（§2.1）。
        C++ test-only 実装で production と同一コード経路（juce::FloatVectorOperations 等）を
        通して正式化する

Phase 0-3  frequency transfer characterization（stage-rate 軸・§2.7 の軸定義に従う）
    [A] Full-chain round trip（Fs_in 基準・入力帯域内）: 50 Hz / 1 kHz / 10 kHz /
        0.25 Fs_in / 0.45 Fs_in / 0.49 Fs_in
    [B] Stage-local interpolation（own-rate cycles/sample）: 0.05 / 0.10 / 0.20 / 0.30 /
        0.35 / 0.40 / 0.45（+ image band 0.5 近傍）
    [C] Stage-local decimation（own-rate cycles/sample）: [transition_end, 0.5] の stopband
        sweep（transition 領域は記録のみ。per-design の transition_end は §2.7.2 実測表）
    [D] passband ripple: 0.1 dB edge（per-config・§2.6.1）までの sweep で max−min。
        全域 [0.005, 0.45] は記録のみ

Phase 0-4  block/reset characterization（v1.7・R2-3 で契約化）
    partition invariance（double 経路）:
      A: 4096 × 1 / B: 1024 × 4 / C: 256 × 16 / D: 513+777+1000+…（ragged）
      各分割の出力波形が **bitwise 一致**（不可なら相対誤差 ≤ 1e-12）
      ※ 実装制約: 各ブロック長 ≤ maxInputBlockSize（prepare 時の容量）。
        超過時 processUp は空ブロックを返すため、分割 D はこれを超えない範囲で構成する
    reset contract（double 経路）:
      A: fresh instance → input block #1 → Reference A
      B: 同一 instance → 任意 stream → reset() → 同一 input block #1 → Reference B
      **A == B（bitwise）**。reset() は upHistory/downHistory を clear するため
      （CustomInputOversampler.cpp:452-467 実測）、fresh と一致することが契約

Phase 0-5  SoftClip local OS characterization
    prepareSingleStage(31, 90.0) について以下を分離:
      I-a: round-trip DC gain / sine amplitude（candidate = 1.0 が gate）
      I-b: SoftClip 動作点: 案 E は SoftClip へ入る信号レベルを +2.5 dB 相当変え得るため、
           「oversampler が unity」と「SoftClip の動作点が意図位置」は別問題として扱う。
           upPeak / downPeak / 閾値到達度を baseline / candidate 双方で記録
           （閾値再校正は Phase 3-C の役割）

Phase 0-6  float-host / double-host end-to-end equivalence（v1.7・R2-4 で数値契約化）
    固定条件: same input / same block partition / same reset state
      float host: float → double 変換 → double DSP / oversampler → float output
      double host: double → double DSP / oversampler → double output
    指標（サンプル値ベース）: maxAbsErr・RMSerr（閾値は §2.6.1 P0-G）
```

#### 2.6.1 B-1-P0 GATE（v1.7・絶対値 + 差分の二重 gate・数値契約確定）

Baseline は record-only。PASS/FAIL は **Candidate vs CONTRACT（絶対値）** と **Candidate vs Baseline（差分帰属）** の 2 系統。

| ID | 判定対象 | 判定条件（v1.7 確定値） | 合否 |
|----|----------|------------------------|------|
| **G-0** | Phase 0-0 | DESIGN-CONTRACT-A が **承認対象の決定事項として明示承認**済み（E1〜E5 添付・推論ではない） | □ PASS / □ FAIL |
| **G-BL** | Baseline sanity | 現行 DC round-trip = 0.75^N（±1e-6）・in-band image rejection = −9.54 dB 一定（±0.05 dB）。逸脱時は Phase 0 中断 | □ PASS / □ FAIL |
| **REF-FIDELITY** | Phase 0-1b | Shadow Reference（baseline モード）== production: DC / impulse / 周波数 / ratio 2/4/8 / preset 2 種 / partition / reset の全項目で **bitwise 一致**（不可なら相対誤差 ≤ 1e-15） | □ PASS / □ FAIL |
| **P0-A** | Candidate DC | Shadow Candidate E の DC round-trip = **1.0 ± 1e-6**（ratio 2/4/8 × preset 全構成） | □ PASS / □ FAIL |
| **P0-B** | Low-freq passband（絶対） | Candidate: 50 Hz / 1 kHz で unity ± **0.1 dB**（理論値 0.0000 dB・実測 §2.7.1） | □ PASS / □ FAIL |
| **P0-C** | Passband ripple | **per-config の 0.1 dB passband edge**（S1: 0.3527 / IIR3: 0.4888 / LP3: 0.4940 Fs_in・実測）までの sweep で ripple ≤ **0.05 dB**（理論値 0.001 dB × 50）。[0.005, 0.45] 全域は記録のみ（transition 落ち込みを ripple と混同しない） | □ PASS / □ FAIL |
| **P0-C'** | Differential（帰属） | Candidate/Baseline 比 = (4/3)^N ± **0.05 dB**。**適用域は per-config（R2-7）**: IIR3/LP3 は f ≤ 0.45 Fs_in（実測偏差 ≤0.0002 dB）、S1 は f ≤ 0.30（実測偏差 0.001 dB）。0.49 Fs_in は記録のみ（IIR3: +0.043 dB / S1: +3.39 dB）。Phase 2 flag-ON 測定が shadow 値と ±0.02 dB で一致すること | □ PASS / □ FAIL |
| **P0-D** | Stopband（stage-local・絶対 + 差分） | (D-1 差分) decimateStage は係数・コードとも不変のため Candidate/Baseline 差 = 0（FP 誤差 ≤1e-12）。 (D-2 絶対) Candidate の alias leakage ≤ **−(A_stage − 10) dB**（A=140/110/90/160/120）を **[transition_end, 0.5] cycles/sample** で満たす（per-design transition_end と実測値は §2.7.2）。transition 領域は記録のみ | □ PASS / □ FAIL |
| **P0-E** | Image rejection（stage-local・絶対 + 差分） | (E-1 絶対) Candidate の image rejection ≥ **80 dB**（判定域 = f ≤ min(0.30, 当該 design の −0.1 dB edge) cycles/sample。実測: 全 design の判定域内で −88.9〜−104.9 dB）。 (E-2 差分/記録) Baseline = **−9.54 dB 一定**（defect 証拠・§2.7.3）を記録し、Candidate の改善幅 ≥ 70 dB を記録（PASS 条件は E-1） | □ PASS / □ FAIL |
| **P0-F** | Latency | impulse 応答 peak 位置 = `Σ (taps[s]−1) × (baseRate/stageRate)`（許容 **±1 sample**）+ centroid cross-check（±0.05 sample 目安）。static_assert は設計前提の宣言で latency PASS とは別物 | □ PASS / □ FAIL |
| **P0-G** | Float / Double host | same input / same partition / same reset state で: **maxAbsErr ≤ 5e-7・RMSerr ≤ 5e-8**。導出: float 仮数 24 bit → 入出力各 1 回の丸め ≤ 2^-24·(|x|+|y|) ≈ 1.8e-7（|x|≤1・|y|≤2）→ gate は約 2.8 倍マージン | □ PASS / □ FAIL |
| **P0-H** | Block / reset contract | Phase 0-4 の partition A/B/C/D が **bitwise 一致**（不可なら ≤1e-12）・`fresh+block#1 == stream→reset()→block#1`（bitwise） | □ PASS / □ FAIL |
| **P0-I** | SoftClip local OS | I-a: `prepareSingleStage(31, 90.0)` の round-trip（taps=31, centerTap=15, centerParity=1, convParity=0 確認済み）で Candidate DC = 1.0 ± 1e-6。I-b: 動作点影響を upPeak/downPeak/閾値到達度で記録 | □ PASS / □ FAIL |

**全 PASS → Phase 1 eligibility**。いずれか FAIL → 案 E を見直し、案 D 統合や tap 再設計に方針変更。

### 2.7 測定軸と実測基準値（v1.7 確定・本セッション閉形式モデル実測）

#### 2.7.0 周波数軸の正規化定義（v1.7・R2-8）

```
own-rate cycles/sample: f̂ = f / Fs_stage（stage 自身の動作レート基準）
  - 有効域 [0, 0.5]。v1.6 の「0.50〜0.95 Fs_stage」は Nyquist 超のため不正 → 廃止
  - stage filter の passband  = [0, edge]（§2.7.2 の −0.1 dB edge）
  - stage filter の stopband  = [transition_end, 0.5]（§2.7.2 実測表）
  - image band（up 段）      = [0.5·(入力レート), 出力レート] の領域。own-rate では
                              [0.5·edge … 0.5] に image が現れる（tone f̂ に対し 0.5−f̂）
Fs_in 基準（full chain）: f̂_in = f / Fs_in（入力帯域 [0, 0.5] のみ有効）
  整合例（モデル検証）: stage-0 (511 taps) の −0.1 dB edge = 0.2448 cycles/sample
   = 0.4896 Fs_in ≒ full-chain candidate edge 実測 0.4888 Fs_in ✓
```

#### 2.7.1 Full-chain 実測基準値（閉形式モデル・baseline / candidate）

| 項目 | S1 (31/90) | IIR3 (511/127/31) | LP3 (1023/255/63) |
|------|-----------|-------------------|-------------------|
| DC baseline / candidate | 0.750000 / 1.000000 | 0.421875 / 1.000000 | 0.421875 / 1.000000 |
| candidate |H| @50Hz〜0.25 Fs_in | −0.0000〜−0.0004 dB | −0.0000〜−0.0005 dB | −0.0000 dB |
| (4/3)^N 差分偏差 @0.45 Fs_in | **+1.635 dB**（適用外） | **−0.0000 dB** | **−0.0000 dB** |
| (4/3)^N 差分偏差 @0.49 Fs_in | +3.393 dB | **+0.043 dB**（v1.6 記載と一致） | −0.0000 dB |
| ripple [0.005, 0.45] base / cand | 5.242 / 3.607 dB | 0.001 / 0.001 dB | 0.000 / 0.000 dB |
| ripple [0.01, 0.30] base / cand | 0.001 / 0.001 dB | 0.001 / 0.001 dB | 0.000 / 0.000 dB |
| candidate 0.1 dB passband edge | **0.3527 Fs_in** | **0.4888 Fs_in** | **0.4940 Fs_in** |

（v1.6 §12 の NumPy 値と整合: ripple 5.24/3.61、0.49 偏差 +0.04 dB）

#### 2.7.2 FIR 設計の絶対基準値（P0-D floor の出典・own-rate cycles/sample）

| design | −0.1 dB edge | transition_end（−A+3 dB） | stopband min 実測 | floor（= A−10 dB） |
|--------|-------------|---------------------------|-------------------|--------------------|
| 511/140 | 0.2448 | 0.2590 | **−138.9 dB** | −130 dB |
| 127/110 | 0.2318 | 0.2782 | **−107.1 dB** | −100 dB |
| 31/90 | 0.1822 | 0.3452 | **−87.0 dB** | −80 dB |
| 1023/160 | 0.2472 | 0.2552 | **−159.2 dB** | −150 dB |
| 255/140 | 0.2396 | 0.2681 | **−138.5 dB** | −130 dB |
| 63/120 | 0.2109 | 0.3130 | **−117.1 dB** | −110 dB |

down-stage alias leakage 実測（own-rate・dB・高いほど悪い）:

| design | 0.26 | 0.30 | 0.35 | 0.45 | 0.474 | transition_end 以降の最悪 |
|--------|------|------|------|------|-------|---------------------------|
| 31/90 | −8.45 | −25.4 | −96.3 | −107.7 | −90.6 | ≈ −87 dB（floor −80 ✓） |
| 127/110 | −18.6 | −114.2 | −125.1 | −138.9 | −177.3 | ≈ −107 dB（floor −100 ✓） |
| 511/140 | −141.9 | −155.2 | −160.0 | −174.9 | −165.4 | ≈ −139 dB（floor −130 ✓） |
| 63/120 | −10.7 | −59.3 | −130.1 | −125.6 | −138.3 | ≈ −117 dB（floor −110 ✓） |
| 255/140 | −36.5 | −152.2 | −157.7 | −165.4 | −167.1 | ≈ −139 dB（floor −130 ✓） |
| 1023/160 | −167.5 | −179.1 | −206.3 | −186.6 | −188.4 | ≈ −159 dB（floor −150 ✓） |

（0.26〜0.30 の低い値は transition 領域であり、P0-D の PASS 判定は transition_end 以降のみ。transition 領域は記録）

#### 2.7.3 Stage-local interpolation image rejection 実測（P0-E の出典）

| design | 0.05 | 0.10 | 0.20 | 0.30 | 0.35 | 0.40 | 0.45 |
|--------|------|------|------|------|------|------|------|
| 31/90 baseline | −9.54 | −9.54 | −9.54 | −9.54 | −9.66 | −10.99 | −24.17 |
| 31/90 candidate | **−88.9** | **−94.2** | **−97.9** | **−99.6** | −46.1 | −25.0 | −11.1 |
| 127/110 baseline | −9.54 | −9.54 | −9.54 | −9.54 | −9.54 | −9.54 | −9.55 |
| 127/110 candidate | **−102.0** | **−97.2** | **−97.2** | **−104.8** | −95.3 | −90.6 | −68.6 |
| 511/140 baseline | −9.54 | −9.54 | −9.54 | −9.54 | −9.54 | −9.54 | −9.54 |
| 511/140 candidate | **−99.9** | **−96.2** | **−104.9** | **−94.6** | −92.4 | −92.6 | −87.7 |

（太字 = E-1 判定域（f ≤ min(0.30, −0.1 dB edge)）内の値。判定域外は transition 挙動として記録のみ
だが、全実測値が独立に −25 dB より深い点に注意。baseline の **−9.54 dB 一定** = 20·log10(1/3) は defect の客観証拠 E3。
判定域: 31/90 → f ≤ 0.1822 / 127/110 → f ≤ 0.2318 / 511/140 → f ≤ 0.2448）

### 2.8 教訓・監査記録（v1.7）

1. **oracle の「教科書の答え」は現行実装と一致しない場合がある**: C 案は half-band FIR の数学的教科書条件としては正しいが、既に half-band FIR で構築された実装に適用すると pure delay に退化する
2. **「gain convention 不整合」と「FIR 構造不整合」を分離する**: 修正は polyphase 配分の対称化に限定し、FIR 構造は維持する
3. **engine fit の `0.75^N` は即時削除不可（HOLD）**: 経験式であり production 定数ではない
4. **Phase 0 を必ず先行**: Baseline / Contract / REF-FIDELITY / Candidate / Differential の 5 構造。**測定だけでは design intent は証明できない** — 契約（G-0）・既存宣言・外部参照（JUCE）との照合が必要
5. **「案 E = 仮説」**: ただし defect そのものは E3（image rejection −9.54 dB vs 設計 −87〜−159 dB）で客観確定。案 E はその defect に対する最小修正仮説
6. **「perfect reconstruction」の表現に注意**: DC gain 1.0 のみ確定。帯域端挙動は変わる（§2.7.1）
7. **「anti-aliasing 維持」はコード変更のみから断定不可**: P0-D/E の実測で確認
8. **feature flag は compile-time gate（authoritative source = CMake option）**
9. **姉妹実装の gain convention 差異**: `src/TruePeakDetector.cpp:284-318` は even/odd とも ×2 補正なし（計測専用経路）。案 E の正解としない
10. **validator error contract**: `RuntimePublicationValidator` は 4 段階それぞれに固定の errorMessage を返す（RuntimePublicationValidator.cpp:16/24/32/40）。enum + errorMessage の両方を Phase B' で比較
11. **【v1.7 追加】測定軸は own-rate cycles/sample に統一する**: 「0.50〜0.95 Fs_stage」のような Nyquist 超表記を排除（レビュー① §8・R2-8 と同根の誤りが再発していた）
12. **【v1.7 追加】shadow 参照実装には fidelity gate が必須**: reference の写し間違いを案 E の FAIL と誤認しない（R2-2）

### 2.9 B-1-P0 GATE の適用要件（v1.7・production 0 変更 gate）

```
B-1-P0 GATE
────────────────────────────────────
production src/:
    modified = 0
    staged   = 0

measurement/harness:
    test-only additions permitted
    （Phase 0-1b/0-2 の test-only reference implementation を含む）

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

---

## 3. §3 harness / production 潜在欠陥（F-2/F-3/F-4）

### 3.1 F-2: `--buzz-rigcheck=eq` の実効 AGC が意図と一致しない

#### 3.1.1 根本原因（2026-09-20 実測）

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
- `setEQAGCEnabled`（1306）と `getEQProcessor()`（1292）は同一オブジェクト `uiEqEditor` を経由 → `configureProbeFlatEQ` 内 `setEQAGCEnabled(false)`（BassBuzzMeasurement.cpp:1081）が staging toggle で上書きされる
- `eq` モードの呼出順（1799/1803）: `configureProbeFlatEQ` → `setAutoGainStagingEnabled(false)` → **AGC が true に再設定される**
- `eqdiag` 系（1833/1834）は逆順で最終 AGC=OFF — 修正案 A はこのパターンに揃える

#### 3.1.2 修正案

| 案 | 内容 | 評価 |
|----|------|------|
| **A（推奨）** | 呼出順を逆に: `setAutoGainStagingEnabled(false)` → `configureProbeFlatEQ(e)` | 最小変更・eqdiag パターンと整合 |
| B | `setAutoGainStagingEnabled` から `setAGCEnabled(!enabled)` を削除 | 設計変更（Bug#4 の前提が崩れる） |
| C | `configureProbeFlatEQ` の最後に staging OFF を追加 | コード複雑化 |

#### 3.1.3 検証手順（state propagation を明示）

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

- **窓 `[0.486,0.496]` は数学的に必ず不変とは言わない**: AGC ON 状態での既存窓整合は実測で再確認
- **B-1 の原因切り分けとは独立**: AGC を強制 OFF にした診断モードでも ratio 0.4912 が不変（監査記録済み）
- 窓境界が ±ε を超えて変動する場合は判定窓再校正を Phase 1 の前段で実施

### 3.2 F-3: EQ dry/wet 混合の潜在欠陥（production 欠陥として記録）

#### 3.2.1 確認事項（2026-09-20 実測）

`src/eqprocessor/EQProcessor.Processing.cpp`:
- `bypassTransitionActive` 判定: :516
- `dryCopyBase` 充填: :570-578（`bypassTransitionActive` 時のみ）
- ブレンド本体: :978-993（`canBlendDry = (dryCopyBase != nullptr)` :980、dry 混合 :993）

```cpp
if (canBlendDry)
    wetPtr[n] = wetPtr[n] * wetGainState + dryValue * dryGain;  // ← 正常
else
    wetPtr[n] = wetPtr[n] * wetGainState;  // ← dry 補償なし（wet-only 減衰）
```

#### 3.2.2 影響評価

- **定常状態の B-1（−5.11 dB）の原因ではない**（定常では `bypassTransitionActive=false`）
- 遷移時のみの潜在欠陥。**production behavior の欠陥**（test-only ではない）・本セッションでは修正しない

#### 3.2.3 修正案（将来 work item）

| 案 | 内容 | 評価 |
|----|------|------|
| A | dry バッファ確保失敗時は遷移を遅延（最大 N ms）してリトライ | 複雑・RT 影響 |
| B | dry バッファ確保失敗時は無音フェード（mute crossfade） | 聴感劣化小 |
| C | ドライコピー失敗時は事前コピー（process 開始前に確保済みに） | prepareToPlay で保証 |
| **D（推奨）** | **記録のみ**。B-1 と分離して別 work item | 既存動作変更なし・リスク最小 |

### 3.3 F-4: `--buzz-rigcheck=ir` の convolver 未有効化

`BassBuzzMeasurement.cpp:1744-1762`: `setConvolverBypassRequested(true)`（1745）の後、`ir` モードは IR をロード（1751）するが bypass 解除がない → 出力は dry コピー。`irwet<digit>`（1763-1792）は 1778 行目で解除済み。

| 案 | 内容 | 評価 |
|----|------|------|
| A | `ir` モードに `setConvolverBypassRequested(false)` を追加 | 既存窓 `[0.880,0.897]` の意味が変わる（dry → wet） |
| **B（推奨）** | `irwet<digit>` で wet 対照をカバー済みとし、`ir` は記録のみ（既存 dry 測定基準を維持） | 測定基準の意味変更なし |

**推奨**: **B**。`ir` モードの wet 有効化は将来別 work item。

---

## 4. §4 F-1: PublicationValidatorIsolationTests 移行・退役

### 4.1 現状（2026-09-20 実測・v1.7 で再確定）

- `src/tests/PublicationValidatorIsolationTests.cpp`: **519 行**（v1.6 の「501 行」は誤り — R2-10）・TEST_F 34 + TEST 4 = **38 ケース** ✓
- **CMake 未登録**（ルート CMakeLists.txt `add_executable` 39 ターゲットのいずれにも該当なし）✓
- **gtest 使用は本ファイルのみ**（`#include <gtest/gtest.h>` のみ・FRIEND_TEST 0 件）✓
- `validator_.checkNoConflictingTransitions(...)` の実呼出しは **9 箇所**（135/246/255/265/273/281/313/324/334 行。128 行は TEST_F 名）✓
- **コンパイル不可**（private 宣言: RuntimePublicationValidator.h:101）✓
- `tools/build-debug.bat:29` の stale target（2026-06-03 `85ac377c` で CMake から削除済み）✓
- validator の検査順序: `RuntimePublicationValidator.cpp:14-41` で **SemanticConsistency → Topology → Resources → checkNoConflictingTransitions・first-fail-wins** ✓
- errorMessage は 4 段階それぞれに固定文字列（:16/24/32/40）— enum + errorMessage の両方を契約として使用 ✓
- AudioEngine.h:3659 `validator_(&validator)` → :3670 `validator_->validatePublication(world)` ✓

**テスト分類（v1.7 再実測・R2-10）**:
- `ValidatePublication_*`（**7**）+ `ValidateSemanticConsistency_*`（**1**）+ `ValidateTopology_*`（**6**）+ `ValidateResources_*`（**11**）= **25 件**
- `CheckTransition_*`（**8**）+ `CheckNoConflictingTransitions_*`（**1**）= **9 件**
- `CrossfadeAuthorityRegressionTest`（TEST 4 件）= **4 件**

### 4.2 移行方針（MIGRATE CASES THEN RETIRE）

#### Phase 0: Case Classification（機械的移植の前に実施）

| 分類軸 | 判定 |
|--------|------|
| **A. 公開 API coverage** | 公開 API（`validatePublication` / `validateSemanticConsistency` / `validateTopology` / `validateResources`）で再現可能か |
| **B. semantic equivalence** | 「同じ invalid state を作れば同じ failureReason に到達するか」 |
| **C. error contract** | `ValidationFailureReason` enum を第一契約・固定 errorMessage を第二契約 |
| **D. 優先度** | `CrossfadeAuthority` 4 件（直接カバレッジ維持）＞ validator 25 件 / `CheckTransition_*` 系 9 件 |

#### Phase A: `CrossfadeAuthorityRegressionTest` 4 ケース移植（最優先・API 不変）

`Decision{needsCrossfade, fadeTimeSec}` / `evaluate(old, new, policy)` の 4 ケース（DeterministicDecision / PolicyChangeChangesDecision / SameStructuralHashNoCrossfade / OversamplingChangeTriggersCrossfade）→ 既存カスタムハーネス（AudioEngineHarness 系・custom main）へ。

#### Phase B: validator 25 件 + CheckTransition 9 件の移植

- 公開 API 経由で再現可能なケースを custom main ハーネスへ移植
- `checkNoConflictingTransitions` を直接呼ぶ 9 ケースは、**間接経路**（`validatePublication` の第 4 段階として到達）で再現

#### Phase B': Semantic Equivalence Gate

移行ケースごとに:
1. **target failure isolation**: 意図した check に到達するよう、**前段 check をすべて PASS させる状態**を構成する（first-fail-wins のため、前段で fail しないこと）
2. **failureReason enum + errorMessage 文字列の両方**を比較
3. 検査順序（Semantic → Topology → Resources → Transition）を踏まえた到達性の記録

#### Phase C: 退役

- 移行完了後に `PublicationValidatorIsolationTests.cpp` を削除
- `tools/build-debug.bat:29` の stale 参照を除去

---

## 5. §5 別課題（既存記録・本セッション対象外）

| ID | 内容 | 優先度 |
|----|------|--------|
| **R-2** | `parseHcIdx` / `parseLcIdx`（BassBuzzMeasurement.cpp:1558/1564）と `stod`/`stoi`/`stof` は不正入力で未捕捉例外 → terminate。実測 12 箇所 | **最優先（極小）** |
| **R-1** | `--buzz-flip-eqgain=`（:1613）は値を受理して破棄（設計上ステップ固定）。silent-ignore 系 | **次（極小）** |
| **B-3** | timestamp-based capture。平均実効レート binding は非一様 callback rate を完全補正しない（flipIndex 誤差 <0.5% 実測・現行目的には十分） | 中 |
| **D-2** | headroom ランタイム統合（4 ランタイム混在）。uv tool `headroom-ai` 0.37.0 は extras=mcp のみで fastapi 無し → proxy 起動不可。Startup の 3 起動体整理 | 中 |
| **D-1** | build identity gate の M1/M2 欠陥（E-G3-3 既知・未修正） | 高 |

### 5.1 R-2 / R-1 の参考実装

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

### 5.2 §5 の着手優先度（将来）

1. **R-2**（極小・fail-closed 化・即時効果）
2. **R-1**（極小・silent ignore 解消）
3. **B-3**（中・timestamp 化）
4. **D-2**（中・環境統一）
5. **D-1**（高・build identity gate 修正）

---

## 6. 推奨する実行順序（v1.7）

```
[Step 1] §3 F-2（harness cleanup・test-only）
  - F-2: 呼出順修正（1 行差替・§3.1.3 の検証手順に従う）
  - F-3/F-4: 記録のみ
  - 検証: AudioEngineHarness.exe --buzz-rigcheck=eq で gainpath staging=0 eqAGC=0 を確認

[Step 2] §4 F-1（テスト移行）
  - CrossfadeAuthority 4 ケース → カスタム main ハーネス
  - Phase 0 分類 → Phase B 移植 + Phase B' semantic equivalence gate（target failure isolation 含む）
  - PublicationValidatorIsolationTests.cpp 退役

[Step 3] §1 O-1〜O-7（commit 意思決定）
  - ユーザー判断待ち
  - 案 A 採用時はエントリ配線の最小変更を含めて commit

[Step 4] §2 B-1（DSP 修正・破壊的変更・案 E）
  - Phase 0-0: source/contract audit（DESIGN-CONTRACT-A を明示承認・E1〜E5 添付）
  - Phase 0-1: Baseline characterization（record-only・期待 0.75^N・image rejection −9.54 dB 記録）
  - Phase 0-1b: REF-FIDELITY gate（Shadow Reference == production・bitwise）
  - Phase 0-2: Shadow Candidate E（test-only reference・上記 gate 通過後）
  - Phase 0-3〜0-6: 周波数 / block・reset / SoftClip / float-double characterization
  - B-1-P0 GATE（§2.6.1）全 PASS → Phase 0 review → ユーザー GO
  - Phase 1: CMake option 導入（compile-time・既定 OFF・既存テスト全 PASS）
  - Phase 2: flag ON で限定検証（Phase 0 shadow 値と直接比較）
  - Phase 3-A → 3-B1 → 3-B2 → 3-C → 3-D: flag 常時 ON → 線形実測 → 非線形動作点実測 →
    calibration-only commit → 窓-only commit
  - Phase 4: flag 削除（次メジャーリリース）

[Step 5] §5 別課題（将来 work item 化）
  - R-2（最優先・極小）→ R-1（次・極小）→ B-3, D-2, D-1
```

**Step 4 の Phase 0 が完了するまで、Phase 1 以降の着手は不可**。Phase 0 で契約・差分・絶対値・周波数特性のいずれかが不成立の場合、案 E を見直し、案 D 統合や tap 再設計に方針変更する。

---

## 7. 検証計画

### 7.1 単体検証（実行可能なコマンド・v1.6 N1 修正済み）

| 検証項目 | コマンド | 期待 |
|----------|----------|------|
| §3 F-2 修正 | `build\Release\AudioEngineHarness.exe --buzz-rigcheck=eq --buzz-dur=2.0` | `[BUZZ] RIGCHECK(eq) gainpath: staging=0 eqAGC=0 ...` |
| §4 F-1 移行 | `ctest --test-dir build --output-on-failure` | 移行先テストが PASS・PublicationValidatorIsolationTests 削除済 |
| §2 B-1 Phase 0/1 | `cmake --build build --config Release --target AudioEngineHarness && build\Release\AudioEngineHarness.exe` | `[OS_DIRECT] ... roundTripGain=`（flag OFF 時 ≈0.75 系・flag ON 時 ≈ 1.0） |
| §2 B-1 Shadow / REF-FIDELITY | Phase 0-1b/0-2 の test-only reference implementation 実行 | REF-FIDELITY bitwise 一致 → Candidate DC = 1.0 ± 1e-6 |

**注**: `PublishPipelineIntegrationTests.exe` は存在しない。ターゲット/バイナリは **AudioEngineHarness**（`build\Release\AudioEngineHarness.exe` 実在確認済み。`add_executable(AudioEngineHarness ...)` は CMakeLists.txt:1899）。`--buzz-*` フラグは AudioEngineHarness.exe に有効。

### 7.2 統合検証

| 検証項目 | 内容 | 期待 |
|----------|------|------|
| §2 B-1 Phase 3 | `eq` 判定窓の再校正 | 新窓での PASS（既存窓 0.4912 → 新範囲） |
| §4 F-1 Phase C | `tools/build-debug.bat` 実行 | `--target ...` エラーなし |

### 7.3 ビルド検証

```bash
build.bat                 # MSVC ビルド
build-icx.bat             # icx ビルド
cmake --build build --config Release --target AudioEngineHarness   # インクリメンタル
```

### 7.4 静的解析

```bash
cppcheck --language=c++ --std=c++20 --enable=warning,performance,portability \
    src/CustomInputOversampler.cpp src/audioengine/AudioEngine.Processing.Latency.cpp
clang-tidy -p compile_commands.json src/CustomInputOversampler.cpp
```

---

## 8. ロールバック計画

| Step | ロールバック方法 |
|------|------------------|
| §1 commit | `git revert <sha>` |
| §2 B-1（behavioral rollback） | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` で rebuild**（第一手段・production source は触らない） |
| §2 B-1（source-level rollback） | `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN … #endif` ブロックと in-source safety net を削除（rebuild 必須） |
| §3 F-2 | 1 行 revert（呼出順を元に戻す） |
| §4 F-1 | 移行先テストが残る場合、ファイルを git から復元 |
| §5 別課題 | 実装しないため不要 |

---

## 9. 影響度まとめ

| 区分 | 影響範囲 | 影響度 | 段階リリース |
|------|----------|--------|--------------|
| §1 commit | リポジトリ履歴のみ | 極小 | 任意 |
| §2 B-1（案 E） | **全オーディオ経路**（最大 +7.5 dB 想定） | **極大**（1 行追加で済む） | 必須（Phase 0→1→2→3-A/B1/B2/C/D→4） |
| §3 F-2 | test-only | 小 | 不要 |
| §3 F-3 | **production behavior の潜在欠陥（記録のみ）** | なし（本セッション） | 不要 |
| §3 F-4 | test-only（記録のみ） | なし | 不要 |
| §4 F-1 | test-only | 中 | 不要 |
| §5 別課題 | 環境 or CLI | 極小〜中 | 不要 |

---

## 10. 監査ログ・参照

- 前スナップショット: `doc/work113/residual_tasks_20260919.md`
- 本セッション報告書: `doc/work113/residual_tasks_20260920.md`
- レビュー記録: `doc/work113/renew_plan.md`（v1.4 + レビュー① 38 観点）・本書 v1.6→v1.7 サマリ（レビュー② 反映）
- 関連ソース（行番号は 2026-09-20 実測・v1.7 で再確定）:
  - `src/CustomInputOversampler.h`（isLinearPhaseFIR/isSymmetricUpDown: 21-22）
  - `src/CustomInputOversampler.cpp`（tapsForStage/attenuationForStage: 84-106 / prepareStage 287-390 / prepareSingleStage 392 / interpolateStage 492-568（convValue ×2 は 557）/ decimateStage 570-723（output 717）/ reset 452-467）
  - `src/audioengine/AudioEngine.h:1416-1432`（F-2 根本・1425）・2626（staging 既定 true）・1292/1306/1314（uiEqEditor）・3659/3670（Bridge → validator_）
  - `src/audioengine/AudioEngine.Processing.Latency.cpp:6-8`（static_assert）・22-24（taps 表）・30（groupDelaySamplesAtStageRate = taps[stage] - 1）
  - `src/audioengine/AudioEngine.Processing.DSPCoreLifecycle.cpp:188/261`（prepareSingleStage(31, 90.0) 本体行）
  - `src/audioengine/AudioEngine.Processing.DSPCoreFloat.cpp:252-263`（float→double 変換）・405/413（softClipOS up/down 本体行）
  - `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:505/513`・593（kOutputHeadroom）
  - `src/eqprocessor/EQProcessor.Processing.cpp:516/570-578/978-993`（F-3）
  - `src/audioengine/RuntimePublicationValidator.cpp:14-41`（検査順序）・16/24/32/40（errorMessage 固定文字列）
  - `src/audioengine/RuntimePublicationValidator.h:92/101`（private）
  - `src/TruePeakDetector.cpp:284-318`（姉妹実装・両位相 ×2 なし）
  - `JUCE/modules/juce_dsp/processors/juce_Oversampling.cpp:185-196`（up: `2 * samples[i]` → 両位相同一 convention）・`:228-240`（down: ×1）— **E5 証拠**
  - `src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`（runOversamplerDirect 1293 / runEqDirectDriveAttribution 1526/1532 / rigcheck 1744-1930 / 窓 2042-2044）
  - `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp`（main 1085 / 前方宣言 1116 / 呼出し 1223）
  - `src/tests/PublicationValidatorIsolationTests.cpp`（**519 行**・38 ケース・呼出し 135/246/255/265/273/281/313/324/334）
  - `CMakeLists.txt`（option 群 :40 / juce_add_gui_app :1062 / CONVOPEQ_ALL_SOURCES :1133 / target_sources :1282 / add_executable(AudioEngineHarness) :1899）
  - `tools/build-debug.bat:29`（stale target）

---

## 11. ユーザー判断待ち項目

| 項目 | 選択肢 | 推奨 |
|------|--------|------|
| §1 O-1 計装 | A: 最小実行可能資産（[OS_DIRECT]+配線）/ B: 全 +642 行 / C: 退役 | **A** |
| §1 O-2 commit 方針 | O-1 と同梱 / 独立 | **独立** |
| §1 O-6 push | 即時 / 別タイミング / ユーザー手動 | **ユーザー手動** |
| §1 O-7 AGENTS.md | 触らない / 別 work item で更新 | **触らない** |
| §2 B-1 修正案 | **E: interpolateStage に `centerValue *= 2.0` 追加（既存 half-band FIR 維持）** | **E（ユーザー監査確定）** |
| §2 DESIGN-CONTRACT-A（G-0） | 明示承認 / 修正要求 / 却下 | **明示承認**（E1〜E5 添付・§2.3.1） |
| §3 F-2 修正 | A: 呼出順逆 / B: setAGCEnabled 削除 / C: 順序固定 | **A** |
| §4 F-1 移行 | Phase A→B→B'→C 全実施 / Phase A のみ / 記録のみ | **全実施** |
| §2 Phase 3 の commit 分割 | 3-A/3-B1/3-B2/3-C/3-D 分割 / 従来の単一 Phase 3 | **分割** |

### §2 B-1 採用案の前提条件（v1.7）

1. **DESIGN-CONTRACT-A** が明示承認される（Phase 0-0・G-0）
2. **REF-FIDELITY** が bitwise 一致（Phase 0-1b）
3. **Shadow Candidate E の DC gain round-trip = 1.0 ± 1e-6**（Phase 0-2・P0-A）
4. **anti-imaging が成立**: candidate image rejection ≥ 80 dB（passband 内・P0-E、実測 −88.9〜−104.9 dB）
5. **anti-aliasing / stopband が維持**: alias leakage ≤ A−10 dB（transition_end 以降・P0-D 実測 −87〜−159 dB）
6. **latency が不変**（impulse 実測・P0-F）
7. **SoftClip 局所 OS が正常動作**（P0-I）
8. **Differential が center-phase ×2 のみに帰属**（P0-C'・per-config 適用域）

### v1.7 最終判定表

| 項目 | 判定 | 備考 |
|------|------|------|
| §1 O-1〜O-7 | **GO 候補** | commit/push の意思決定として分離 |
| §3 F-2 | **GO 候補** | 呼出順修正（1 行差替・§3.1.3） |
| §3 F-3 | **記録継続** | production behavior の潜在欠陥・定常 B-1 の原因ではない |
| §3 F-4 | **記録継続** | 既存 dry 測定基準維持・`irwet<digit>` で wet 対照 |
| §4 F-1 | **GO 候補** | case classification 先行 → 公開 API 経由移植 + Phase B' gate |
| §5 | **保留継続** | 将来 work item 化 |
| **B-1 原因分析** | **GO（defect は E3 で客観確定）** | baseline image rejection −9.54 dB vs 設計 −87〜−159 dB |
| **B-1 Phase 0** | **GO（測定仕様 v1.7 適用が条件）** | REF-FIDELITY + 絶対値 floor + 数値契約を追加済み |
| B-1 Phase 1 | **HOLD** | Phase 0 全 PASS + ユーザー GO が前提 |
| B-1 Phase 2/3/4 | **HOLD** | 順次判定 |
| **B-1 案 E** | **有力仮説として採用** | 仮説だが defect 自体は客観確定 |
| 「perfect reconstruction」 | **表現修正** | DC gain 1.0 のみ確定（帯域端は transition 挙動） |
| B-1 calibration 変更 | **HOLD** | Phase 3-B1/B2 → 3-C/D の順 |
| `0.75^N` engine-fit 削除 | **HOLD** | 経験式・監査記録/窓の更新は Phase 3-C/D |
| compile-time flag | **CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`** | authoritative source = CMake |
| TruePeakDetector 姉妹実装 | **記録継続** | 案 E の正解としない |

---

## 12. 本計画書の検証エビデンス（2026-09-20 ソース監査・レビュー② 反映）

1. **行番号監査（全件再実測・v1.6 の誤りを修正）**:
   - 修正: `DSPCoreLifecycle.cpp` **188/261**（v1.6 の 187/260 は off-by-one）・`DSPCoreFloat.cpp` **405/413**（同 404/412 は off-by-one）・`PublicationValidatorIsolationTests.cpp` **519 行**（v1.6 の 501 は誤り）・テスト分類 **7/1/6/11 + 8/1**（v1.6 の 5/1/7/12 は誤り）・`add_executable(AudioEngineHarness)` **:1899**（v1.6 の 1898 は off-by-one）
   - 一致確認: CustomInputOversampler.cpp 系・AudioEngine.h 系・Latency.cpp 系・Validator 系・BassBuzzMeasurement 系・PPIT 系・CMake 系（その他）・build-debug.bat:29

2. **数値再現（本セッション・独立の閉形式モデル実装）**:
   - DC: baseline 0.750000 / 0.421875 / 0.421875・candidate 1.000000（全構成）— 12 桁一致
   - 差分 (4/3)^N: IIR3/LP3 で f ≤ 0.45 Fs_in ±0.0002 dB・IIR3 の 0.49 で +0.043 dB（v1.6 記載と一致）
   - ripple: S1 全域 5.242/3.607 dB（v1.6 記載 5.24/3.61 と一致）・mid 0.001 dB
   - passband edge（0.1 dB）: S1 0.3527 / IIR3 0.4888 / LP3 0.4940 Fs_in
   - FIR 設計停止帯: −87.0〜−159.2 dB（design A − 3 dB 相当）
   - down alias leakage: transition 領域（例 31/90: −8.45 dB @0.26）と stopband（−87〜−179 dB）を分離測定
   - **baseline in-band image rejection = −9.54 dB 一定（全 design）— defect の客観証拠（E3）**
   - candidate image rejection: passband 内 −88.9〜−104.9 dB

3. **外部参照（JUCE 同梱実装・E5）**: `juce_Oversampling.cpp:185` の `buf[N - 1] = 2 * samples[i];` → up 経路で入力 ×2 を両位相に適用、down は ×1。本修正の構造と同一の convention

4. **静的解析**: cppcheck（C++20・warning/performance/portability）を `CustomInputOversampler.cpp` に実行 → 指摘 0 件

5. **ビルド実測**: `build\Release\AudioEngineHarness.exe` 実在 ✓・`PublishPipelineIntegrationTests.exe` は存在しない ✓

6. **レビュー①/② との照合**: レビュー① 7 必須 + R8〜R15 採用（v1.5）。レビュー② 4 必須（R2-1〜R2-4）+ G-0 強化（R2-5）を本 v1.7 で採用し、あわせて v1.6 の誤り 2 件（N3/N7）を修正。「4 点を入れれば Phase 0 着手計画として承認可能」の状態に到達

---

**本書は方針案です（v1.7・レビュー② 4 必須 + G-0 強化反映・数値契約確定版）。実装着手は本書承認後**。
**§2 B-1：案 E（polyphase gain convention 対称化）を「有力仮説」として採用し、defect 自体は E3（baseline image rejection −9.54 dB vs FIR 設計減衰 −87〜−159 dB）で客観確定**。
**Phase 0 は Baseline / Contract(G-0) / REF-FIDELITY / Candidate / Differential の 5 構造 characterization。全 PASS + ユーザー GO 後に Phase 1 eligibility**。
**compile-time flag は CMake option `CONVOPEQ_CORRECT_POLYPHASE_GAIN`・段階リリース Phase 0 → 1 → 2 → 3-A/B1/B2/C/D → 4**。
**承認状態: §1/§3 F-2/§4 F-1 = GO候補 / §3 F-3・F-4/§5 = 保留継続 / §2 = Phase 0 着手承認（測定仕様 v1.7 適用が条件）・Phase 1 以降は HOLD**。
