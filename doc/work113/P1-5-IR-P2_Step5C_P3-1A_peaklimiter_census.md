# P1-5-IR-P2 — Step 5-C / P3-1-A: `peakLimiter` lifetime / reset census（read-only）

- **作成**: 2026-09-23 / work113 P1-5-IR Phase 2（P3-1-A）
- **性質**: **完全 read-only source audit**。source / CMake / production / settle / sleep = **0 変更**。
  実測・ビルド・追加 run = **0**。
- **指示どおり P3-1-A のみを実施し、本報告で一旦停止する**（targeted experiment の実装可否判定まで）。

---

## 0. State Freeze（変更前・PASS）

```text
HEAD = 1e9e63e3 / kP15FullMatrix = true / binary = b2c39a9a0e3b1fe3 / flag = OFF
production source diff = 0 / CMake diff = 0 / JUCE diff = 0
default/calibration = 0 / settle/sleep = 0 / measurement semantics = 0

P3-0 = PASS / A3 = REPRODUCED / A4 = PASS
1.4759 dB = REPRODUCED MEASUREMENT ANOMALY / mechanism = UNKNOWN
H5 = SUPPORTED / H6 = SUPPORTED / H7 = SUPPORTED / H9 = SUPPORTED
```

---

## 1. A-1: `peakLimiter` インスタンス寿命 census

### 1-1. 質問への回答

| # | 質問 | 回答 | 根拠 |
| --- | --- | --- | --- |
| 1 | `peakLimiter` が DSPCore のメンバとして何個存在するか | **1 個** | `src/audioengine/AudioEngine.h:975` `::SimplePeakLimiter peakLimiter;`（DSPCore クラス本体のメンバ。`oversampling` / `softClipOS` / `outputFilter` / `dither` と同スコープ） |
| 2 | `runCase()` の sc0 → sc1 の間で同一 DSPCore が使われるか | **UNKNOWN** | `configureChain` は毎回 Structural rebuild intent を投入し、rebuild 成功時は **新 DSPCore が生成される**（`RuntimeBuilder.cpp:469` `aligned_make_unique<DSPCore>()`）。完了すれば別 core（limiter は default 構築で `envelope=1.0`）、未完なら同一 core。**完了タイミングに依存し、source だけでは確定できない** |
| 3 | `AudioEngine` 再構築なしで `runCase()` が継続するか | **「再構築なし」とは言えない** | 各 `runCase` は最低 2 本の Structural intent を投入する（`setSoftClipEnabled` 1 本 + `endBulkParameterRestore(true)` 1 本）。OS 変更時は `setOversamplingFactor` からさらに 1 本 |
| 4 | `prepare()` が `peakLimiter.reset()` を間接的にも呼んでいないか | **呼ばない** | `SimplePeakLimiter::prepare` は `releaseCoeff` のみ設定（`SimplePeakLimiter.h:19-24`）。`reset()` を呼ぶ経路なし |
| 5 | `reset()` 以外に `SimplePeakLimiter::reset()` を呼ぶ箇所がないか | **`reset()` 自体がどこからも呼ばれていない**（全ソースで 0 件） | `git grep 'peakLimiter.reset\|peakLimiter->reset'` = **NONE** |

### 1-2. `peakLimiter` 全呼び出し census（ライフサイクル別）

| ライフサイクル | 呼び出し箇所 | 内容 |
| --- | --- | --- |
| **constructor** | （明示なし） | メンバ default 構築。`envelope = 1.0`・`releaseCoeff = 0.0`（`SimplePeakLimiter.h:88`） |
| **prepare** | `DSPCoreLifecycle.cpp:228`（`:227-229` の診断ブロック内） | `peakLimiter.prepare(newSampleRate, 100.0)` — **`releaseCoeff` のみ**設定 |
| **prepare** | `DSPCoreLifecycle.cpp:292`（`:291-293`） | 同上（別 prepare 経路） |
| **reset** | **なし** | **`SimplePeakLimiter::reset()` は全ソースで 0 回** |
| **processBlock** | `DSPCoreDouble.cpp:722`（`DSPCore::processOutputDouble` 内） | `peakLimiter.processBlock(dataL, dataR, numSamples, kPLThreshold, kPLKnee)` |
| **processBlock** | `DSPCoreIO.cpp:529` | 同上（別 IO 経路） |
| **destructor** | （明示なし） | DSPCore 破棄に伴う default 破棄 |
| **accessor** | `SimplePeakLimiter.h:85` `getCurrentEnvelope()` | **呼び出し 0 件**（診断に未使用） |

- `SimplePeakLimiter` 型の使用箇所は include（`AudioEngine.h:92`）・メンバ定義（`:975`）・クラス定義のみで、
  **他に instance は存在しない**。

### 1-3. 測定経路における limiter の位置（source 確定）

```text
AudioEngineHarness::audioLoop                         AudioEngineHarness.cpp:97-113
  ├─ tapCopy(buffer, true)          ← 入力 tap（信号生成）      :105
  ├─ engine_->getNextAudioBlock(info)                            :107
  │     └─ dsp->process(...)        ← active DSPCore            AudioBlock.cpp
  │           └─ processOutputDouble(...)                       DSPCoreDouble.cpp:589
  │                 ├─ truePeakDetector.processBlock            :710
  │                 ├─ loudnessMeter.processBlock（読み取り専用）  :713
  │                 ├─ peakLimiter.processBlock                 :722  ★
  │                 └─ hard clamp（kOutputHeadroom で jlimit）    :724-739
  └─ tapCopy(buffer, false)         ← 出力 tap（capture 元）      :109
```

- **出力 tap は `getNextAudioBlock` の後**に呼ばれるため、**capture される信号は limiter の後段**である。
  → H6 の機構（limiter の envelope が測定出力に影響しうる）は **source 上 reachable**。
- limiter は chain の**最終段**（hard clamp の直前）に位置する。
- `DSPCore::reset()`（`DSPCoreLifecycle.cpp:381-406`）の reset 対象は
  `convolverState` / `eqState` / `dcBlockers` / `dither` / `fixedNoiseShaper` / `adaptiveNoiseShaper` /
  `oversampling` / `outputFilter` / `activeAdaptiveCoeff*` / バッファ / `ramps()` / `histories()` のみで、
  **`peakLimiter` は含まれない**（P3-0 で既出・本 census で再確認）。

---

## 2. A-2: limiter の level-response を source から確定

### 2-1. 3 領域の実 gain（`SimplePeakLimiter::processBlock` :41-82）

```cpp
clipStart = thresholdLinear - kneeLinear * 0.5
          = 0.8413951287507587 - 0.108748 * 0.5
          = 0.7870210643753794        (約 -2.08 dBFS)
```

| 領域 | `peak` 範囲 | `desiredGain` | 実際に掛かる gain |
| --- | --- | --- | --- |
| **1** | `peak <= clipStart`（≤ 0.787021） | `1.0`（`if` に入らない） | `envelope = 1.0 + (envelope-1)*releaseCoeff` の**回復途中の値**。**envelope が 1.0 未満なら gain < 1.0 が掛かる** |
| **2** | `clipStart < peak <= threshold`（0.787021〜0.841395） | smoothstep knee: `1 - (1 - threshold/peak) * t²(3-2t)`, `t=(peak-clipStart)/kneeLinear` | `envelope = min(desiredGain, 前回 envelope)`（attack 即時） |
| **3** | `peak > threshold`（> 0.841395） | `threshold / peak` | 同上 |

```cpp
if (desiredGain < envelope) envelope = desiredGain;              // attack 即時（0 ms）
else envelope = 1.0 + (envelope - 1.0) * releaseCoeff;           // release 時定数 100 ms
dataL[i] *= envelope;  dataR[i] *= envelope;                     // 無条件適用
```

- **重要な帰結（source-level）**: **領域1 でも gain は 1.0 に即戻らない。**
  一度でも gain reduction が起きれば、`releaseCoeff`（100 ms 時定数）で回復するまでの間、
  **閾値未満のサンプルにも残留 gain が乗算され続ける**。
  これが「ケース／ブロックを跨ぐ残留」の具体的な機構であり、`envelope` が DSPCore の寿命中持続する
  （§1-2 のとおり reset されない）ことと組み合わさって、**level 依存の履歴効果**を生みうる。
- `kPLThreshold` / `kPLKnee` の導出は source コメントで明示されている（`DSPCoreDouble.cpp:715-721`）:
  `kOutputHeadroom = -1.0 dBFS = 0.8912509381337456`、`threshold = kOutputHeadroom - 0.5 dB = 0.8413951287507587`、
  `knee = 1.0 dB width = 0.108748`。

### 2-2. -20 / -6 dBFS の chain peak が knee に到達するかを source の数値だけで判断できるか

**判定: 判断不能 → `UNKNOWN`**（指示どおり）。

判断不能の理由（3 点）:

1. **`p15ir` 行は出力ピークを出さない**。`outMax` を出力するのは `kind=level` / `kind=softclip` 行のみ
   （`P1PolyphaseGainCharacterization.cpp:440` の `out.outMax` は `kind=level`/`kind=softclip` の emit にのみ含まれる）。
   limiter 入力（= hard clamp 直前の最終出力）のピークは `p15ir` 行からは得られない。
2. **limiter 状態の観測手段が存在しない**。`getCurrentEnvelope()`（`SimplePeakLimiter.h:85`）は定義されているが
   **呼び出し 0 件**（§1-2）。したがって `envelope` の実測値はログに存在しない。
3. **`gainDb` から導出できるのは単一ビン成分のみ**。`dftDb = gainDb + ampDb` より
   - am=−20: `-14.5035 + (-20) = -34.5035 dB` → 1 kHz 成分 0.018845
   - am=−6 : `-13.0276 + (-6)  = -19.0276 dB` → 1 kHz 成分 0.111775

   これらは `clipStart = 0.787021` を大きく下回るが、**peak ≠ 単一ビン成分**である
   （IR は delta・chain にフィルタ／リンギングがあり out-of-band 成分を含む）。よって peak の
   上限・下限を source から確定できず、**knee 到達の有無は判定不能**。

```text
A-2 = UNKNOWN（limiter 入力 peak の knee 到達可否は source から確定できない）
```

---

## 3. P3-1-D 必須項目の取得可能性評価（feasibility）

`--p1-char` vehicle が出すログ prefix の完全 census（A3 実測ログ）:

```text
[L0_WRITE] 3240 / [IR_CHAIN] 1874 / [IR_TAIL_ENV] 460 /
[IR_RATE_GEN] 313 / [IR_TAIL_GEOM] 313 / [IR_TAIL_ENERGY] 313 / [P1CHAR] 286
```

**出ていない**もの: `[PUBLISH]` / `[CONV_STATUS]` / `[REBUILD_TELEMETRY]` / `[GEOM]` / `[CONV_IR]` / `[DIAG]` / `[DSPCORE_PREPARE]` / `[FAULT]`

**原因（source 確定）**: `[PUBLISH]` は `juce::Logger::writeToLog` 経由（`PublicationExecutor.cpp:91`）であるが、
`--p1-char` は stderr logger を設置しない。JUCE の `Logger::writeToLog` は
`if (currentLogger != nullptr) currentLogger->logMessage(...)`（`juce_Logger.cpp:54-55`）であるため、
**logger 未設置時はメッセージが破棄される**。一方 `[IR_CHAIN]` / `[L0_WRITE]` は
`std::fprintf(stderr, ...)` 直接出力（`ConvolverProcessor.LoaderThread.cpp:531` 等）のため stderr に現れる。

| P3-1-D 必須項目 | 取得可能性 | 根拠 |
| --- | --- | --- |
| `gainDb_sc0` | **可** | `[P1CHAR] p15ir` 行 |
| `ampDb` | **可** | 同上 |
| `limitingEngaged_sc0` | **可** | 同上 |
| `hardClamp_sc0` | **可** | 同上 |
| `clipEngagement` | **可** | 同上 |
| `clipEngMax` | **可** | 同上 |
| peak / limiter diagnostic | **不可** | `p15ir` 行に `outMax` なし・`getCurrentEnvelope()` 未使用 |
| OS | **可** | 行の `os=` / `n=`（`n = log2(os)`） |
| IR geometry | **可** | `[IR_TAIL_GEOM]` / `[L0_WRITE] geom` |
| F_scale | **可** | `[IR_CHAIN] F_scale` |
| **publication sequence** | **不可** | `--p1-char` は `[PUBLISH]` を出さない（writeToLog 破棄） |
| **generation** | **△ 部分可** | IR generation は `[IR_RATE_GEN] gen=` / `[IR_TAIL_GEOM] gen=`（ただし全行 `gen=0` 固定）。**world generation は不可** |

- **不可 3 項目**（peak/limiter diagnostic・publication sequence・world generation）を取得するには、
  いずれかの追加が必要:
  - **(a)** `--p1-char` に stderr logger を設置（`--buzz` / T1〜T4 と同じ既存機構
    `juce::Logger::setCurrentLogger(&logger)`）→ `writeToLog` 経由の**既存**ログ行が stderr に現れる
    （**新規ログ文の追加ではない**が、vehicle への計装追加ではある）
  - **(b)** limiter の `envelope` をログ出力 → `getCurrentEnvelope()` の呼び出し追加 =
    **新しい accessor 使用／新規ログ**に該当

---

## 4. P3-1-F 停止条件との照合（実装可否判定）

| # | 停止条件 | 該当 | 備考 |
| --- | --- | --- | --- |
| 1 | targeted vehicle に production source が必要 | **非該当** | 変更は `P1PolyphaseGainCharacterization.cpp`（test TU）に閉じる |
| 2 | `runCase` の settle 変更が必要 | **非該当** | topology 維持のため不要 |
| 3 | `sleepPump` 変更が必要 | **非該当** | 同上 |
| 4 | `waitWorldPublished` 変更が必要 | **非該当** | 同上 |
| 5 | `gainDb` / `dftMag` 変更が必要 | **非該当** | 同一解析をそのまま使用 |
| 6 | 新しい CLI flag が必要 | **非該当** | targeted vehicle はハードコードで実現可能 |
| 7 | limiter reset の production API が必要 | **非該当** | reset は行わない（介入禁止のため） |
| 8 | targeted vehicle が historical topology を維持できない | **非該当（維持可能）** | `p15eq os=8 → setOversamplingFactor(1) → g0/os1 → sc0(-6) → sc0(-20)` は既存 helper の組み合わせで表現可能。IR reload（`ensureTestIr`）と 2 パス構造も維持可能 |
| 9 | OS8→OS1 transition を省略する必要 | **非該当** | 直前ケースとして p15eq os=8 を残せる |
| 10 | IR reload topology を省略する必要 | **非該当** | `ensureTestIr` を維持 |

- **停止条件 1〜10 はいずれも該当しない** → reverse-order experiment 自体は
  **test-only / topology 維持 / 新 CLI flag なし / settle 変更なし** で実装可能。
- **ただし独立の阻害要因が 1 件**: P3-1-D の必須 3 項目
  （**peak/limiter diagnostic・publication sequence・world generation**）が現行 vehicle では取得不能。
  取得には上記 (a) または (b) の計装追加が必要であり、特に (b) は「新しい logging/accessor」に該当し
  **P1 §9-3 の停止条件**に触れる。

---

## 5. P3-1-A の結論

```text
P3-1-A = 完了（read-only census 完了・source 変更 0）
```

**確定事項**:

1. `peakLimiter` は **DSPCore に 1 個**。`SimplePeakLimiter::reset()` は**どこからも呼ばれない**。
   `prepare()` は `releaseCoeff` のみ設定。`DSPCore::reset()` にも含まれない。
   → **`envelope` は DSPCore インスタンスの寿命中、reset されない**。
2. ただし **DSPCore は structural rebuild ごとに新規生成される**（`RuntimeBuilder.cpp:469`）ため、
   rebuild が完了した場合の `envelope` は default（1.0）に戻る。**ケース間持続の有無は
   「rebuild が wait 窓内に完了するか」に依存し、source だけでは UNKNOWN**。
3. limiter の 3 領域応答を確定。**領域1（閾値未満）でも release 中の残留 gain が掛かる**。
4. limiter は **chain 最終段（hard clamp 直前）**にあり、**harness の出力 tap はその下流**。
   → H6 の機構は測定出力に reachable。
5. **-20 / -6 dBFS の chain peak が knee に到達するかは source から判定不能 → UNKNOWN**。
6. P3-1-D の必須項目のうち **3 項目（peak/limiter diagnostic・publication sequence・world generation）が
   現行 vehicle で取得不能**。取得には計装追加が必要（(b) は P1 §9-3 に抵触）。

**実装可否判定**:

```text
reverse-order experiment 本体 : 実装可能（停止条件 1〜10 非該当）
P3-1-D の観測要件            : 3 項目が未充足（計装追加が必要）
```

→ **判定は「条件付き可」**。次段階の設計判断は以下から選択が必要:

| 選択肢 | 内容 | 影響 |
| --- | --- | --- |
| **X-1** | 3 項目を **UNKNOWN のまま**として受け入れ、reverse-order experiment を実施（取得可能な項目のみで Pattern 1/2/3 を判定） | 計装追加 0。ただし publication sequence / generation / limiter 状態を根拠にできないため、Pattern 判定の根拠が `gainDb` 系列と `clipEngagement`/`clipEngMax` に限定される |
| **X-2** | stderr logger の設置（(a)）のみ追加し、既存 `[PUBLISH]` / `[CONV_STATUS]` / `[REBUILD_TELEMETRY]` を stderr に出す | test TU への計装追加（既存機構の再利用・新規ログ文なし）。publication sequence / world generation が取得可能になる。limiter 診断は依然不可 |
| **X-3** | limiter の `envelope` ログ（(b)）も追加 | **P1 §9-3（新しい logging/accessor）に抵触**。実施するなら明示的な承認が必要 |

- **本 P3-1-A ではいずれも実施していない**（read-only）。指示どおり**ここで一旦停止**する。
- `kP15FullMatrix = true`・binary `b2c39a9a0e3b1fe3`・production diff 0 を維持。
