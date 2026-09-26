# renew_plan.md (v1.3) 妥当性検証 — 確定報告

- **検証日**: 2026-09-20
- **対象**: `doc/work113/renew_plan.md`（v1.3・Phase 0 着手承認版）
- **基準ソース**: HEAD `8f127bfe` の production source + working tree の test-only 差分
- **ツール**: serena MCP / AiDex / context-mode / rtk(WSL) / headroom MCP / cppcheck / NumPy / 文献 (JUCE, fundsp, MathWorks, WaveWalker DSP, Brave LLM context)
- **総合判定**: **技術的中核は正しい。案 E は「DC gain 対称化」として業界実装と整合。ただし計画書本文に実務影響のある記述誤りが複数あり、v1.4 修正後に Phase 0 着手を推奨。Phase 1 以降は HOLD 維持。**

---

## 0. 結論サマリ

| 区分 | 判定 | 確定事項 |
|------|------|----------|
| §1 O-1〜O-6 前提 | **GO** | git 状態・行数・HEAD が計画書と一致（AGENTS.md 未 commit のみ未記載） |
| §2 B-1 原因特定 | **確定（defect）** | `interpolateStage` 557 行で conv のみ ×2、center に ×2 なし。DC round-trip = 0.75 |
| §2 案 E | **有力仮説として妥当** | DC gain 1.0 は数値確定。JUCE/fundsp が両位相へ gain-2 を掛ける同一 convention。全帯域 PR は未証明のまま Phase 0 必須 |
| §2 Phase 0 先行 | **GO** | 帯域端では案 E でも round-trip ≠ 1.0（0.45Fs_in で ≈0.66）。多帯域実測が必要 |
| §3 F-2/F-3/F-4 | **GO（F-2）/ 記録（F-3/F-4）** | コード根拠一致。F-2 呼出順修正は妥当。窓の再校正は別途 |
| §4 F-1 | **GO（分類数のみ要修正）** | 519 行・38 ケース・CMake 未登録・private 直叩き・stale bat を確認 |
| §5 保留 | **妥当** | R-1/R-2 は記述どおり。B-3/D-1/D-2 は保留継続で可 |
| 計画書そのもの | **v1.4 修正後に承認推奨** | 実務影響あり 3 件 + 文言 4 件（§9 参照） |

---

## 1. 環境・前提の確定

### 1.1 Git 実測（§1 の前提）

| 項目 | 計画書 | 実測 | 判定 |
|------|--------|------|------|
| HEAD | `8f127bfe` | `8f127bfeea17831e049ef6ce569f3c570cb96a62` | ✅ |
| 未 push | `c4a08171` + `8f127bfe` | ahead 2 / behind 0 | ✅ |
| production `src/` 未 commit | 0 | test ファイル以外の `src/` 変更なし | ✅ |
| BassBuzzMeasurement.cpp | +634/−2 | git numstat 634 / 2（working tree 2687 行） | ✅ |
| PublishPipelineIntegrationTests.cpp | +8/0 | +8/0（1340 行） | ✅ |
| ConvoPeq.md | Generated 2026-09-20 07:00:33 | 同一 | ✅ |
| CTestCostData.txt | ` D` | ` D` | ✅ |
| `.opencode/opencode.json` | 未追跡・触らない | `??` のまま | ✅ |
| **AGENTS.md** | **計画書に記載なし** | **` M`（+5/−3・MIMO Desktop パイプライン運用メモ）** | ⚠️ **O リストへ追加推奨** |

### 1.2 ツール稼働

| ツール | 状態 |
|--------|------|
| headroom MCP | 読込済（本セッション compress 未使用・proxy は環境側） |
| context-mode MCP | v1.0.169 / doctor PASS |
| rtk (WSL) | `/home/user/.local/bin/rtk` 経由で git/ls/rg |
| serena MCP | ConvoPeq プロジェクト活性・cpp/python_basedpyright/bash LS |
| AiDex | インデックス有効・query 正常 |
| ccc (cocoindex-code) | インデックス 304992 chunks / 2580 files（使用可） |
| graphify | `graphify.exe` / `graphify-mcp.exe` 確認（パスは Scripts 配下） |
| cppcheck | CustomInputOversampler **0 指摘**（warning/performance/portability） |

---

## 2. §2 B-1 — コード照合と数値確定

### 2.1 prepareStage（`src/CustomInputOversampler.cpp:287-390`）

計画書 §2.2.1 と**手順・行番号とも一致**。

1. `centerTap = (taps-1)/2`、`centerParity`、`convParity = 1-centerParity`（292-294）
2. sinc×Kaiser 生成（308-317）※center の sinc 値は `0.5`（312）
3. center と同 parity の非 center タップを 0 に（319-323）— half-band parity 間引き
4. sum 正規化（325-333）
5. `rawCoeffs[centerTap] = 0.5` 強制（335）
6. `nonCenterSum` を `scale = 0.5 / nonCenterSum` で正規化（336-347）
7. center を再度 0.5 に（348）→ **FIR 合計 = 1.0**
8. `convCoeffs` は `convParity` タップのみ抽出（350-365）→ **合計 = 0.5**
9. `centerCoeff = rawCoeffs[centerTap] = 0.5`（367）

taps テーブルも一致：

| Preset | taps | （備考） |
|--------|------|----------|
| LinearPhase | 1023, 255, 63 | 88 行 |
| IIRLike | 511, 127, 31 | 92 行 |

全 taps で centerTap が奇数 → `centerParity=1` / `convParity=0`。

### 2.2 interpolateStage（492-568）— **欠陥の所在**

```text
545: centerValue = stage.centerCoeff * history[...]     // = 0.5 × x
539: convValue  += convCoeffsReversed[r] * x            // 合計 0.5 × x
557: convValue *= 2.0;                                  // ★ conv のみ
563: output[outBase + convParity]   = convValue;        // 1.0 × x
564: output[outBase + centerParity] = centerValue;      // 0.5 × x
```

**center 側に ×2 は存在しない**（558-559 は denormal ゼロ化のみ）。

### 2.3 decimateStage（570-723）

```text
658: acc = centerCoeff * centerSample                   // 0.5 × s_center
672-705: acc += Σ convCoeffs[r] * history[...]          // 0.5 × s_conv
717: output[n] = acc;                                   // ×2 補正なし
```

定数高レート入力 `v` に対して down DC gain = **1.0**。
up 直後（even=0.5 / odd=1.0）を down に通した往復は **0.75**。

### 2.4 数値検証（本セッション NumPy）

| 条件 | 現行 | 案 E (`centerValue *= 2`) | Δ |
|------|------|---------------------------|---|
| 1 stage DC round-trip | 0.750000 | 1.000000 | +2.4988 dB |
| 2 stages (ratio 4) | 0.562500 | 1.000000 | +4.9975 dB |
| 3 stages (ratio 8) | 0.421875 | 1.000000 | +7.4963 dB |
| SoftClip taps=31/90 DC | 0.750000 | 1.000000 | 同一経路 |

half-band 係数生成（taps=31, atten=90）の独立再現：

- `centerCoeff = 0.5`
- `conv sum = 0.5000000000`
- `full FIR sum = 1.0000000000`
- E0/E1 polyphase DC 各 0.5

**周波数依存（案 E でも DC 以外は 1.0 にならない）**:

| f / Fs_in | 現行 round-trip | 案 E round-trip |
|-----------|-----------------|-----------------|
| 0.00 | 0.750 | 1.000 |
| 0.01 | 0.750 | 1.000 |
| 0.125 | 0.699 | 0.924 |
| 0.25 | 0.559 | 0.707 |
| 0.45 | 0.109 | **0.351〜0.660**（実装依存・帯域端） |
| 0.50 | 0.250 | 0.500 |

→ 計画書の「**perfect reconstruction は未証明**」「**Phase 0 判定基準 C/D/E で帯域実測**」は**数値的にも必須**。

### 2.5 ソフトクリップ経路

- `prepareSingleStage(31, 90.0, internalMaxBlock)` — `DSPCoreLifecycle.cpp:188 / :261` で production 実使用
- softClipOS 呼出し — `DSPCoreFloat.cpp:405/413`（Double も同一関数群）
- taps=31 → centerTap=15, centerParity=1, convParity=0（計画書 §2.5.1 I と一致）
- `oversamplingFactor > 1` のとき主 OS、`== 1` のとき softClipOS が局所 2× を供給する分岐は Float/Double に実在

### 2.6 契約とレイテンシ

| 項目 | 実測 | 計画書 | 判定 |
|------|------|--------|------|
| `isSymmetricUpDown` | `CustomInputOversampler.h:22` = `true` | あり | ✅ |
| `isLinearPhaseFIR` | 同 21 行 = `true` | あり | ✅ |
| static_assert | `AudioEngine.Processing.Latency.cpp:6-8` | 6-9 と記載 | 微差（実害なし） |
| **latency 計算式** | **`groupDelaySamplesAtStageRate = taps[stage] - 1`**（up+down 合算・stage レート） | **`(taps-1) × 2 per stage`** | ❌ **要修正** |

正しい表現: **「1 段あたり往復 group delay = (taps−1) サンプル（その段の stage レート）」**。base レート換算は `× (baseRate / stageRate)`。

IIRLike OS=8 の合計 base レート遅延（コード実装どおり）: **290.25 サンプル**。
LinearPhase OS=8: **582.25 サンプル**。

SoftClip の理論値コメント（31 tap → 15 base rate samples）は `(taps-1)/2 per pass × 2 passes` であり、こちらは実装と整合。

### 2.7 engine fit 0.75^N

- `0.98379 × 0.75^max(log2 effOS, 1)` は **`residual_tasks_20260919.md:79` の経験式**
- production コードに `0.98379` / `0.75^N` / `effOS` 定数は **存在しない**（全域 grep）
- `kOutputHeadroom = 0.8912509381337456`（−1.0 dBFS）は別定数（DSPCoreDouble/IO/NoiseShaperLearner）
- Phase 3 の「0.75^N 削除」は **コード削除ではなく、rigcheck 判定窓 `[0.486,0.496]` と監査記録・経験式記述の更新** と読み替えるのが正確

---

## 3. 案 E の妥当性 — 文献・実装参照

### 3.1 業界実装との対応

| 実装 | gain convention | 案 E との関係 |
|------|-----------------|---------------|
| **JUCE `dsp::Oversampling` FIR half-band** (`juce_Oversampling.cpp:185`) | `buf[N-1] = 2 * samples[i]` — **FIR 入力を両位相にわたって ×2**。even は polyphase 畳み込み、odd は center tap × 既に ×2 済み履歴 | **Case E 相当**（両位相 DC gain が揃う） |
| **fundsp** (`oversample.rs`) | `tick_even` / `tick_odd` とも `output * 2.0` | **Case E と同一** |
| **JUCE IIR polyphase** (`:425`) | down 側で `(delay + directOut) * 0.5` | down で 0.5 スケールを明示。up 側 allpass と対になる convention |
| **MathWorks FIR Halfband Interpolator** | polyphase + 一方の枝が pure delay。フィルタ設計側でゲインを規定 | half-band 構造の標準的説明と一致 |
| **DSPRelated** (cascaded half-band) | 「each upsample-by-2 function includes a gain of 2」 | **interpolate 側に gain 2** を置く慣行 |
| **KVR / 実装メモ** | 4x で `effect.process(a * 4.0)` と外部ゲイン補償する例も有り | 「どこに gain 2^N を置くか」は実装選択。位相非対称は非標準 |

### 3.2 案 E として確定する事項 / しない事項

**確定してよいこと**:

1. 現行の `conv のみ ×2` は polyphase 両位相を対称に扱う標準 convention と一致しない
2. `centerValue *= 2.0` 追加で **constant / DC gain round-trip = 1.0**
3. FIR 係数形状（sinc×Kaiser×parity 間引き、center=0.5 / 他合計=0.5）は **変更不要**
4. `decimateStage` は **変更不要**
5. 修正は事実上 **1 行**（compile-time flag 付き）
6. レイテンシ契約（taps と group delay）は **FIR 不変なら不変**
7. `static_assert(isSymmetricUpDown && isLinearPhaseFIR)` は **保持される見込み**（宣言は tap 対称性を指す）

**確定してはいけないこと（計画書の留保を支持）**:

1. 全帯域 perfect reconstruction
2. stopband / image rejection の相対不変（帯域端の再構成は変わる）
3. 既存 calibration 窓 `[0.486,0.496]` の自動成立
4. SoftClip 閾値まわりの実機挙動
5. 「案 E だけで PR が言える」という主張

**追加観点（計画書に無いが Phase 0 で見てよい）**:

- `TruePeakDetector::interpolateStage`（同リポジトリ）は **両位相とも ×2 なし**。`CustomInputOversampler` と別の gain convention。True Peak 計測が主 OS 経路の 0.75 を吸収していないか、Phase 0 の参考測定に加えられる
- OS_DIRECT の `upGain`/`downGain` 分離は **ピーク比**（`upPeakMax/inPeak`, `outPeak/upPeakMax`）。台帳の「down 側に局在」はこの測定定義による表記であり、数学的欠陥の所在は **up の位相非対称**

---

## 4. §3 harness — コード照合

### 4.1 F-2 — 確定（GO 候補）

| 観点 | 実測 |
|------|------|
| 反転の本体 | `AudioEngine.h:1425` `getEQProcessor().setAGCEnabled(!enabled)` |
| 既定値 | `AudioEngine.h:2625` `std::atomic<bool> autoGainStagingEnabled { true }` |
| early-return | `current == enabled` なら AGC を触らない（1418-1419） |
| `eq` 呼出順 | 1799 `configureProbeFlatEQ` → 1803 `setAutoGainStagingEnabled(false)` |
| `configureProbeFlatEQ` | 1080-1081 で `setEQTotalGain(0)` / `setEQAGCEnabled(false)` |
| `eqdiag` 呼出順 | 1833 staging(false) → 1834 `configureProbeFlatEQ`（正しい順序） |
| 観測 API | `isEQAGCEnabled()`（1313）/ `isAutoGainStagingEnabled()`（1433）— `gainpath` 行で可観測 |

**動作分解**:

1. 既定 staging=true の状態で `eq` ブロックに入る
2. `configureProbeFlatEQ` が AGC を false にする
3. `setAutoGainStagingEnabled(false)` は true→false 遷移なので early-return せず、`setAGCEnabled(!false)=true` に上書き
4. 結果: staging=0 / **eqAGC=1**（意図と逆）

案 A（呼出順入替）で eqdiag と同型になり、`staging=0 eqAGC=0` が期待できる。

**窓について**: 既存 `eq` 窓 `[0.486,0.496]` は「AGC 0」を主張したコメントの下で、**実際は F-2 バグにより AGC=1 のまま測られた可能性**がある。F-2 修正後は必ず再測定。計画書 §3.1.3 の「数学的に必ず不変とは言わない」は正しい。

なお residual の「AGC 強制 OFF 診断でも 0.4912 不変」は **eqdiag 系（正しい呼出順）での観察**であり、B-1 主因が CustomInputOversampler であることと整合する。

### 4.2 F-3 — 確定（記録継続で妥当）

- `dryCopyBase` 充填: `EQProcessor.Processing.cpp:570-582`（`bypassTransitionActive` かつ容量十分時のみ）
- ブレンド: 978-1015
- `canBlendDry==false` 時: 1002-1003 `wetPtr = wetPtr * wetGainState` のみ（dry 補償なし）
- 定常時は `bypassTransitionActive=false` でブレンド自体が走らない → **定常 B-1 の原因ではない**
- 案 D（記録のみ）妥当

### 4.3 F-4 — 確定（案 B 妥当）

- 1744-1745: 両 bypass ON
- `ir` 分岐 1749-1762: IR ロードのみ・`setConvolverBypassRequested(false)` なし
- コード内コメント（1765-1767）も「出力は bypass blend の dry コピー」と明記
- `irwet<digit>` 1763-1792: 1778 で明示的に conv bypass 解除
- 既存窓 `[0.880,0.897]` を dry 基準として維持する案 B が妥当

---

## 5. §4 F-1 — コード照合

| 計画書の主張 | 実測 | 判定 |
|--------------|------|------|
| 520 行 | **519 行** | 微差 |
| TEST_F 34 + TEST 4 = 38 | TEST_F 34 + CrossfadeAuthority TEST 4 = **38** | ✅ |
| CMake 未登録 | `add_executable` に `PublicationValidatorIsolationTests` なし | ✅ |
| gtest は本ファイルのみ | 当該ファイルで gtest 参照あり・リポジトリ依存 0 | ✅ |
| FRIEND_TEST 0 | 0 | ✅ |
| `checkNoConflictingTransitions` private | `RuntimePublicationValidator.h:101`（private 配下） | ✅ |
| 呼出し 9 箇所 | テスト側で 9 | ✅ |
| build-debug.bat:29 stale | 29 行に `--target PublicationValidatorIsolationTests` | ✅ |
| コンパイル不可 | private 直叩き + CMake 未登録 → **コンパイル不能** | ✅ |
| `ValidationFailureReason` | `RuntimePublicationValidator.h` に enum あり | ✅ |
| CrossfadeAuthority API | `Decision{needsCrossfade, fadeTimeSec}` / `evaluate(...)` | ✅ |
| **分類 28 / 6** | ValidatePublication 系など **25** + CheckTransition/CheckNoConflicting **9** | ❌ **要修正 25/9** |

**error category 契約**（string 非契約）は `ValidationFailureReason` の存在により実装可能。Phase A→B→C 方針は妥当。

---

## 6. §5 保留項目 — 確定

| ID | コード確認 | 判定 |
|----|------------|------|
| R-1 | 1613 行 `--buzz-flip-eqgain=` → `flipKind=4; flipValue=0`（値破棄） | 記述どおり。修正コスト極小 |
| R-2 | `parseHcIdx` 等 + `stod/stoi/stof` に try/catch なし | 記述どおり。fail-closed 化は別 item |
| B-3 | capture の実効レート補完は未完全 | 保留妥当（flipIndex 誤差 <0.5%） |
| D-1 | build identity gate M1/M2 | 保留妥当 |
| D-2 | headroom 4 ランタイム | 保留妥当。MIMO 環境では v0.37.0 proxy は稼働確認済み |

---

## 7. O-1 計装の実体（commit 判断材料）

未 commit の test-only 計装は **実在**し、計画書が列挙する診断モードを含む。

| 記号 | 所在 | 内容 |
|------|------|------|
| `[OS_DIRECT]` | BassBuzzMeasurement.cpp:1361 / 1431 | `runOversamplerDirect()` — ratio 1/2/4/8 × preset + SoftClip singleStage。出力キーは **`roundTripGain=`**（`round-trip=` ではない） |
| `[EQ_DIRECT]` | 1189 ほか | 実 EQProcessor 直接駆動 |
| `[OF_DIRECT]` | 1507 | 実 OutputFilter 直接駆動 |
| `gainpath` | 1852-1860 | staging / eqAGC / totalGain / hdr / makeup / struct |
| `eqdiag` / `eqdiagser` / `eqos<digit>` / `irwet<digit>` | 1705-1708, 1817-1840 | 受理・実装あり |
| エントリ | `runEqDirectDriveAttribution()` 経由（PublishPipelineIntegrationTests 側） | **`--buzz-osdirect` フラグは存在しない** |

**O-1 案 A（[OS_DIRECT] のみ commit）の実務上の制約**:

- OS_DIRECT / EQ_DIRECT / OF_DIRECT は同一エントリに束なっている
- `git add -p` だけでは分離できない
- 案 A を採るなら **main からの配線最小変更**（例: `runOversamplerDirect()` を直接呼ぶ）を含める必要がある
- 期待する検証出力は `PublishPipelineIntegrationTests` 実行時の `[OS_DIRECT] ... roundTripGain=` であり、`ConvoPeq.exe --buzz-osdirect=2` ではない

---

## 8. 文献・参考

| ソース | 用途 |
|--------|------|
| JUCE `juce::dsp::Oversampling` ドキュメント + 同梱ソース `juce_Oversampling.cpp` | FIR half-band の入力 ×2、latency 式 |
| fundsp `oversample.rs` | 両 polyphase `* 2.0` |
| Wave Walker DSP「Polyphase Half Band Filter for Decimation by 2」 | half-band polyphase の標準構造 |
| MathWorks `dsp.FIRHalfbandInterpolator` / `firhalfbandinterpolator` | polyphase + 一方が delay |
| DSPRelated half-band cascaded decimators | upsample-by-2 の gain of 2 |
| Brave LLM context（KVR, Reddit FPGA 等） | 実装慣行の補助 |
| 日本音響学会 https://acoustics.jp/journal/ | 学会誌ポータル（今回の DSP convention 判定には直接論文ヒットなし。多速度系の理論は DSP 教科書・JUCE/MathWorks 実装が一次根拠） |

---

## 9. 計画書の不整合 — v1.4 修正確定リスト

| 優先 | 箇所 | 問題 | **確定した修正文** |
|------|------|------|---------------------|
| **P0** | §7.1 / §2 検証コマンド | `--buzz-osdirect=2` は**存在しない**。出力キーも `round-trip=` ではない | 「`PublishPipelineIntegrationTests` 実行 → `[OS_DIRECT] ... roundTripGain=1.000000`（flag OFF 時 `0.750000`）を確認」に書き換え。専用 CLI を作るなら別途 |
| **P0** | §1.1 案 A | OS_DIRECT と EQ/OF_DIRECT が同一エントリ | 「案 A にはエントリ配線の最小変更を含む」を追記 |
| **P0** | フラグ名の表記ゆれ | 冒頭は `CONVOPEQ_CORRECT_POLYPHASE_GAIN` 採用・`kCorrectPolyphaseGain` 不採用だが、本文 §2.5/§2.6.4/§6 に `kCorrectPolyphaseGain` が残る | **全文 `CONVOPEQ_CORRECT_POLYPHASE_GAIN` に統一**。`kCorrectPolyphaseGain` / `kUseV2Oversampler` は削除 |
| **P1** | §2.4 / §2.5.1 F | latency を `(taps-1) × 2 per stage` と記載 | 「1 段あたり往復 **(taps−1)** サンプル（**stage レート**）。base レート換算は ×(baseRate/stageRate)」 |
| **P1** | §4.2 Phase B | テスト分類 28/6 | **25 / 9**（計 34 は一致） |
| **P2** | §2.5「engine fit の 0.75^N 削除」 | production に該当定数なし | 「rigcheck 判定窓・監査記録側の 0.75^N 依存記述の更新」 |
| **P2** | 行番号微差 | prepareStage 287-390、テスト 519 行、F-3 は 978 起点が blend 本体、static_assert 6-8 | 範囲表記を補正 |
| **P3** | §1 O リスト | `AGENTS.md` 未 commit が未記載 | 「触らない / 別 commit でドキュメント更新」のどちらかを明記 |
| **P3** | TruePeakDetector | 姉妹実装が別 gain convention | Phase 0 の参考測定 or 別 work item として記録 |

---

## 10. 未確定事項 → 確定結果

| 元の未確定事項 | 確定 |
|----------------|------|
| B-1「0.75 は defect か意図 convention か」 | **defect（コード的）**。conv のみ ×2 は標準と不一致。ただし**帯域端挙動を変える修正**なので Phase 0 実測が GO 条件 |
| 案 E の DC gain | **1.0 確定**（数値・文献・JUCE 入力×2 で裏付け） |
| 案 E で PR か | **未証明**。帯域端で round-trip ≠ 1.0。Phase 0 判定基準 A〜I |
| isSymmetricUpDown と latency | **FIR 不変なら保持**。ただし計画書の latency 数式表現は誤り（(taps-1)×2 → (taps-1)@stage rate） |
| F-2 修正後の観測 | **`gainpath: staging=0 eqAGC=0` が実装可能かつ可観測**。窓は再測定必須 |
| F-2 が B-1 原因か | **否**。eqdiag（正しい順序）でも ratio 0.4912 不変の記録と整合 |
| F-3 が定常 B-1 原因か | **否** |
| F-4 の `ir` は wet か | **否（dry）**。`irwet` で wet |
| F-1 コンパイル可否 | **不可**（private + CMake 未登録） |
| engine fit 0.75^N の所在 | **監査記録の経験式のみ**。production コードに無し |
| O-1 〜O-6 前提 | **ほぼ全件一致**（AGENTS.md のみ O リスト外） |
| 検証用 CLI `--buzz-osdirect` | **存在しない**。計装は PublishPipelineIntegrationTests 側 |
| flag 名衝突 | 既存コードに `CONVOPEQ_*` / `kCorrectPolyphaseGain` / `kUseV2Oversampler` は**未使用**（新規導入可） |
| 静的解析 | CustomInputOversampler.cpp は cppcheck **0 指摘** |

---

## 11. 推奨アクション（確定）

1. **計画書 v1.4** に §9 の P0〜P2 を反映してから「確定版」として承認する
2. **Phase 0 着手は GO**（production 変更 0・B-1-P0 GATE のまま）
   - 判定基準 A〜I の測定を、計画書 §2.5.2 の帯域分離どおり実施
   - latency 判定は **(taps−1) @ stage rate** を期待値にする
   - OS_DIRECT は PublishPipeline 経由で `roundTripGain=` を読む
3. **Phase 1 以降は HOLD**（Phase 0 全 PASS まで）
4. **O-1** は案 A を採る場合、エントリ配線を含む最小差分として commit 文言を決める
5. **F-2** は Phase 0 と独立に test-only として着手可能（呼出順 1 行差替）
6. **F-1** は分類 25/9 で Phase 0 分類表を作り直してから移植
7. `0.75^N` はコードではなく **窓と文書** の再校正対象として Phase 3 まで HOLD
8. TruePeakDetector の gain convention は別 work item として台帳へ

---

## 12. 最終判定

```text
計画書 v1.3
├── 技術的中核（B-1 原因 / 案 E / F-2 / F-4 / F-1 事実認定） : 正確
├── 段階リリース・Phase 0 先行・compile-time flag 方針        : 妥当
├── 記述上の実務不整合（CLI / 配線 / latency 式 / 分類数 / flag 名） : 修正必須
└── 総合                                                   : v1.4 修正後に承認
                                                              Phase 0 = GO
                                                              Phase 1+ = HOLD
```

本書は監査結果であり、production `src/` は一切変更していない。
