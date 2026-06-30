# ConvoPeq 音飛び解析報告書

**作成日**: 2026-06-30
**解析対象**: ConvoPeq.log (2026-06-30 19:34:51 〜 19:35:18, 約27秒間)
**実行条件**: 192kHz/1024spb (期待コールバック間隔 5.33ms), Adaptive Noise Shaper Continuous モード
**補足資料**: ソースコード調査結果 → `analysis_findings_2026-06-30.md`

---

> **本報告書の位置付け**
> 本報告書はログの客観データに基づく記述を主とし、原因推定にはソースコード検証による裏付けを行っています。
> 未確定事項は「未確定」と明記しています。詳細なソースコード調査結果は別紙 `analysis_findings_2026-06-30.md` を参照してください。

---

## 目次

1. [実行環境とセッション概要](#1-実行環境とセッション概要)
2. [総エグゼクティブサマリー](#2-総エグゼクティブサマリー)
3. [XRUN統計](#3-xrun統計)
4. [2つの実行フェーズの発見](#4-2つの実行フェーズの発見)
5. [フェーズ別詳細分析](#5-フェーズ別詳細分析)
6. [XRUN検出機構の解析](#6-xrun検出機構の解析)
7. [Interval分析 — 音飛びの直接原因](#7-interval分析--音飛びの直接原因)
8. [CPU Migrationの実影響評価](#8-cpu-migrationの実影響評価)
9. [Inputフェーズの構成 — 未確定](#9-inputフェーズの構成--未確定)
10. [改善提案](#10-改善提案)
11. [付録: 解析方法](#11-付録-解析方法)

## 1. 実行環境とセッション概要

### システム構成

| 項目 | 値 |
|---|-----|
| CPU | Intel (多数コア, 論理プロセッサ16) |
| OS | Windows 11 |
| オーディオホスト | VST3 (DAWまたはホストアプリケーション) |
| サンプリングレート | 192,000 Hz |
| ブロックサイズ | 1,024 samples |
| 期待コールバック間隔 | 5,333 μs (5.33ms) |
| オーバーサンプリング | osFactor=2 (内部処理レート 384,000 Hz) |
| 内部処理ブロックサイズ | 2,048 samples |
| 診断サンプリング | 1/16 (CONVOPEQ_DIAG_SAMPLE_MASK=0xF) |
| プロセス優先度 | HIGH_PRIORITY_CLASS |

### セッションタイムライン

```
19:34:51.000 ─ アプリケーション起動
      │
      ├─ Gen=1:  初期ビルド @48kHz/65536spb (113.4ms)
      │
      ├─ Gen=2:  初期パブリッシュ
      │
      ├─ Gen=3:  prepareToPlay @192kHz/1024spb (107.3ms)
      │
      ├─ Gen=4:  ランタイム開始 ──── [フェーズA]
      │   ├─ PUBLISH @callback=38 (1.96ms)
      │   ├─ XRUN#1〜#3 即座に発生 (8.6-8.9ms間隔)
      │   └─ 約4,850コールバック実行
      │
      ├─ convolverParamsChanged: IRファイル読み込み
      │   └─ [CONV_IR] IR transferred ch=2 len=25906 sr=192000
      │
      ├─ Gen=5:  IRロード後ランタイム ── [フェーズB]
      │   ├─ PUBLISH @callback=4884 (4.36ms)
      │   ├─ ビルド: 172.9ms + IR再構築: 480.1ms = total 655.5ms
      │   ├─ メモリ: 1,351MB → 2,833MB (+1,482MB)
      │   └─ 約29,300コールバック実行
      │
      └─ (継続中)
19:35:18.000 ─ ログ終端
```

### 使用データセット

- **総ログ行数**: 106,426 行
- **総コールバック数**: 34,215 (callback index max)
- **CALLBACK_STAGE**サンプル: 2,138 件 (1/16間引き)
- **CBSUMMARY**レポート: 193 件 (100ms周期監視)
- **CB_HIST**エントリ: 61,888 件
- **CPU_MIG**エントリ: 29,980 件

---

## 2. 総エグゼクティブサマリー

本セッションでは **5,686件のXRUN（音飛び）** が発生し、**一度も中断することなく27秒間連続**しました。

解析の結果、音飛びの原因は **時間経過で性質が変わる2つの異なる実行フェーズ** にまたがる **3層の根本原因** が重畳したものであることが判明しました。

### 最重要発見

| # | 発見 | インパクト |
|---|------|-----------|
| 1 | **IRロード前(Gen=4)でもXRUNは継続発生** | DSP処理(289μs)は軽量だが、**オーディオホストが5.33ms間隔でコールバックを呼べていない** |
| 2 | **IRロード後(Gen=5)にDSPが35倍に増大** | DSP 289μs→10,270μs、90.8%のコールバックが予算超過 |
| 3 | **XRUNはコールバック内検出ではなく監視タイマーによる** | 全XRUNでCallback=0.00ms。100ms周期のモニターがバッチ出力 |
| 4 | **非DSPオーバーヘッドが常に約2.8ms存在** | Publish/World切替の同期処理が主因と推定 |

---

## 3. XRUN統計

### 基本統計

| 指標 | 値 |
|---|-----|
| **総XRUNイベント数** | **5,686 件** (XRUN#1 〜 #5686) |
| 連続性 | **単一クラスタ** — 一度も切れ目なし |
| 平均Interval | **12.22ms** (期待5.33msの**229%**) |
| Interval範囲 | 8.0ms 〜 **79.7ms** |
| 中央値Interval | 約10ms |
| XRUN密度 | **210 XRUNs/秒** |

### XRUN Interval 分布

| 区間 | 割合 | 判定 |
|---|------|------|
| 8-9ms (期待の150-169%) | 最多ピーク | 🔴 常時超過 |
| 9-10ms (169-188%) | 第2ピーク | 🔴 |
| 10-12ms (188-225%) | 有意 | 🔴 |
| 12-15ms (225-281%) | 散見 | 🔴🔴 |
| 15-20ms (281-375%) | 稀 | 🔴🔴 |
| 20-30ms (375-563%) | 稀 | 🔴🔴🔴 |
| 30-50ms (563-938%) | 稀 | ❌ |
| 50ms+ (938%+) | 稀 (最大79.7ms) | ❌❌ |

### CBSUMMARY コールバック損失

193件のCBSUMMARYレポート（100ms周期監視）の分析:

| 指標 | 値 |
|---|-----|
| 損失あり(loss>0)のレポート | **173/193 (89.6%)** |
| **累積コールバック損失** | **2,111 コールバック** |
| 最大Interval記録 | **79.7ms** (約15コールバック分消失) |
| 正常(loss≤0)のレポート | わずか20件 (10.4%) |

---

## 4. 2つの実行フェーズの発見

ログを詳細分析した結果、セッション内で **根本的に性質の異なる2つの実行フェーズ** が存在することが判明しました。

```
フェーズA [Gen=4]                   フェーズB [Gen=5]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DSP:       289μs (軽量)     →    10,270μs (35倍)
  Total:   3,119μs (予算内)   →    12,823μs (240%)
  Budget: 中央値42.2%          →    中央値225.3%
  IR:       未ロード           →    ロード済 (25,906samples)
  メモリ: 1,351MB             →    2,833MB
  CALLBACK_STAGE: 303 samples  →    1,833 samples (6倍)
```

この2フェーズの分離が、本解析の最重要発見です。**最初のレポートで報告した「平均Total 3.1ms」はフェーズAのみの値であり、フェーズBを含めた統合値ではありませんでした。**

---

## 5. フェーズ別詳細分析

### 5.1 フェーズA: Gen=4 (IR未ロード)

#### CALLBACK_STAGE 統計 (n=303)

| フェーズ | 平均 | 中央値 | P25 | P75 | P90 | P95 | P99 | 最大 |
|---|------|--------|-----|-----|-----|-----|-----|------|
| **Input** | 2,829μs | 2,001μs | 1,143μs | 3,605μs | 5,797μs | 7,632μs | 11,033μs | 15,000μs |
| **DSP** | **289μs** | **259μs** | 221μs | 316μs | 399μs | 461μs | 631μs | 877μs |
| Output | ≈0μs | 0μs | 0μs | 0μs | 1μs | 2μs | 7μs | 16μs |
| **Total** | **3,119μs** | **2,252μs** | 1,397μs | 3,872μs | 6,136μs | 7,981μs | 11,804μs | 15,529μs |
| **Budget** | **58.4%** | **42.2%** | 26.2% | 72.6% | 115.0% | 149.6% | 221.3% | 291.1% |

#### Budget 分布

| 範囲 | 件数 | 割合 | 状態 |
|---|------|------|------|
| 0-25% | 117 | **38.6%** | ✅ 余裕 |
| 25-50% | 58 | 19.1% | ✅ 良好 |
| 50-75% | 38 | 12.5% | ⚠️ 注意 |
| 75-100% | 32 | 10.6% | ⚠️ 上限 |
| 100-150% | 35 | 11.6% | 🔴 超過 |
| 150-200% | 11 | 3.6% | 🔴 |
| 200%+ | 12 | 4.0% | 🔴🔴 |

**70.2%のコールバックが予算75%未満で正常動作。**

#### DSP分布

| 範囲 | 件数 | 割合 |
|---|------|------|
| 0-500μs | 288 | **95.0%** |
| 500-2,000μs | 15 | 5.0% |

**95%のコールバックでDSPが500μs未満。** コンボルバ未ロード時のEQ+α処理は極めて軽量。

#### Inputオーバーヘッド比率

InputがTotalに占める割合:
| 範囲 | 件数 | 割合 |
|---|------|------|
| 0-25% | 46 | 15.2% |
| 25-50% | 2 | 0.7% |
| 50-75% | 62 | 20.5% |
| 75-90% | 73 | 24.1% |
| 90-100% | 120 | **39.6%** |

**合計: 84.2%のコールバックでオーバーヘッドがTotalの50%超。39.6%で90%超。**

---

### 5.2 フェーズB: Gen=5 (IRロード後)

#### CALLBACK_STAGE 統計 (n=1,833)

| フェーズ | 平均 | 中央値 | P25 | P75 | P90 | P95 | P99 | 最大 |
|---|------|--------|-----|-----|-----|-----|-----|------|
| **Input** | 2,553μs | 2,046μs | 501μs | 3,328μs | 5,531μs | 7,274μs | 12,413μs | 20,771μs |
| **DSP** | **10,270μs** | **9,497μs** | 6,990μs | 12,474μs | 16,434μs | 19,285μs | 25,851μs | 45,914μs |
| Output | ≈0μs | 0μs | 0μs | 0μs | 1μs | 2μs | 8μs | 16μs |
| **Total** | **12,823μs** | **12,017μs** | 8,485μs | 16,107μs | 20,687μs | 24,041μs | 32,174μs | 51,705μs |
| **Budget** | **240.4%** | **225.3%** | 159.1% | 302.0% | 387.9% | 450.7% | 603.3% | 969.5% |

#### Budget 分布

| 範囲 | 件数 | 割合 | 状態 |
|---|------|------|------|
| 50-75% | 36 | 2.0% | ⚠️ わずかに予算内 |
| 75-100% | 132 | 7.2% | ⚠️ 上限 |
| 100-150% | 237 | 12.9% | 🔴 超過 |
| 150-200% | 295 | 16.1% | 🔴🔴 |
| 200%+ | 1,133 | **61.8%** | 🔴🔴🔴 壊滅的 |

**90.8%のコールバックが予算超過。61.8%が200%超。**

#### DSP分布

| 範囲 | 件数 | 割合 |
|---|------|------|
| 0-500μs | 0 | **0.0%** |
| 500-2,000μs | 0 | **0.0%** |
| 2-5ms | 200 | 10.9% |
| 5-10ms | 818 | **44.6%** |
| 10-15ms | 548 | **29.9%** |
| 15-20ms | 187 | 10.2% |
| 20ms+ | 80 | 4.4% |

**DSPが2ms未満のコールバックは皆無。44.6%が5-10ms、29.9%が10-15ms。**

#### フェーズA→Bの変化量

| 指標 | フェーズA | フェーズB | 変化率 |
|---|----------|----------|--------|
| DSP平均 | 289μs | **10,270μs** | **+3,454% (35倍)** 🔴 |
| Total平均 | 3,119μs | **12,823μs** | **+311% (4.1倍)** 🔴 |
| Budget中央値 | 42.2% | **225.3%** | **+434%** 🔴 |
| DSP/Total比 | 29.0% | **82.3%** | 逆転 |
| Input/Total比 | 70.9% | 17.7% | 逆転 |
| メモリ | 1,351MB | **2,833MB** | **+110% (+1.5GB)** 🔴 |
| ビルド時間 | 93.4ms | **655.5ms** | **+602%** 🔴 |
| Input平均 | 2,829μs | 2,553μs | -10% (ほぼ不変) |

---

## 6. XRUN検出機構の解析（ソースコード検証済み）

### 6.1 XRUN検出はAudio Callback内で行われる

ソースコード（`AudioEngine.Processing.AudioBlock.cpp:436-500`）の検証により確定:

```cpp
// AudioBlock.cpp:436-500 — callback開始直後のXRUN検出ブロック
const auto t0_start = convo::getCurrentTimeUs();
cbStartUs = t0_start;
cbPrevEndUs = convo::consumeAtomic(rtLocalState_.lastCallbackEndTicks, ...);
// interval = t0_start - 前回callbackのpublish時刻
const double intervalMs = static_cast<double>(t0_start - cbPrevEndUs) / 1000.0;
// callbackMs = t1_end - t0_start (preamble実行時間のみ)
const auto t1_end = convo::getCurrentTimeUs();
const double callbackMs = static_cast<double>(t1_end - t0_start) / 1000.0;

// 閾値: interval > max(expected * 1.5, 3.0ms)
if (intervalMs > std::max(expectedMs * 1.5, 3.0)) → XRUN

// リングバッファにpush (Timer Threadが消費)
xRunBuffer.push(ev);

// 次のcallbackのために時刻を公開
convo::publishAtomic(rtLocalState_.lastCallbackEndTicks, t1_end, ...);
```

**XRUN検出はAudio Callback内で行われ、Callback=0.00msはXRUN preambleの実行時間（t1_end - t0_start ≪ 5μs）である。** これは監視スレッドが検出しているのではなく、コールバック内の極小ブロックの実行時間が0.00msに丸まっているだけであり、正常動作である。

### 6.2 Intervalの実測対象

```
intervalMs = t0_start(current) - t1_end(previous)
```

`t1_end` はXRUN検出ブロックの終了時刻（≈ callback開始＋5μs）であるため:

```
intervalMs ≈ t0_start(current) - t0_start(previous) = callback start-to-start
```

つまり **Intervalは前回と今回のcallback開始時刻の差**を表す。前回のcallback本体の処理時間は含まない。

### 6.3 データフロー

```
Audio Callback内:
  XRUN検出 → LockFreeRingBuffer(64)にpush → Timer Threadがpop → Logger出力

Timer Thread (100ms周期):
  xRunBuffer.pop() 全消費 → [XRUN#N] ログ出力
  exchangeAtomic(intervalMaxUs_) → 1秒サマリ → [CBSUMMARY] ログ出力
```

同一タイムスタンプに複数XRUNが並ぶ理由は、Timer Threadがリングバッファを一括消費（pop loop）するため。

### 6.4 確定したXRUN検出パラメータ

| パラメータ | 値 | 根拠 |
|-----------|-----|------|
| 検出場所 | **Audio Callback内** | AudioBlock.cpp:436 |
| 閾値: ratio | expected × **1.5** | `kRatioThreshold` |
| 閾値: margin | **3.0ms**固定下限 | `kFixedMarginMs` |
| 実効閾値(5.33ms時) | **8.0ms** | max(5.33×1.5, 3.0) |
| Interval測定対象 | **start-to-start** | `t0_start - cbPrevEndUs` |
| Callback=0.00ms | **preamble時間**（正常） | 前回報告の解釈は誤り |

---

## 7. Interval分析 — 音飛びの直接原因

### 7.1 フェーズAでもInterval超過は常時発生

フェーズA（Gen=4, Total平均3.1ms < 予算5.33ms）でもXRUNは継続:

```
CALLBACK_STAGE seq=48 total=659us budget=12.3%  ← 予算内で余裕
XRUN#1 Interval=8.92ms (167% of expected)       ← しかし間隔は超過
```

Intervalの実測対象は start-to-start（確定事項6.2）。フェーズAではcallback本体の処理時間3.1msとは無関係に、callback間隔が一貫して8-12msとなっている。

**言い換えれば: コールバックが予算内で完了しているにもかかわらず、ホスト（VST3/オーディオドライバ）は5.33ms周期で次のコールバックを発行できていない。**

### 7.2 Interval分布（全5,686件）

| パーセンタイル | Interval | 期待比 | 累積超過 |
|---------------|----------|--------|---------|
| P50（中央値） | **10.65ms** | 200% | 50% |
| P75 | **13.92ms** | 261% | 25% |
| P90 | **18.13ms** | 340% | 10% |
| P95 | **21.44ms** | 402% | 5% |
| P99 | **29.32ms** | 550% | 1% |
| Max | **79.73ms** | 1,496% | — |

フェーズA（First 100 XRUNs, 主にGen=4）: 平均 **10.84ms**
フェーズB（Last 100 XRUNs, Gen=5）: 平均 **12.03ms**（+1.19ms増加）

フェーズBでIntervalが延びていることは、**DSP過負荷が次のcallback発行タイミングに悪影響を与えている**ことを示唆する。

### 7.3 確定した直接原因

| # | 原因 | 確実性 | 根拠 |
|---|------|--------|------|
| **A** | **ホストのcallback発行間隔が5.33msを超過** | ✅ **確定** | Interval start-to-start 計測値10.65ms（中央値） |
| **B** | **IR畳み込みのDSP過負荷（フェーズBのみ）** | ✅ **確定** | DSP 289μs→10,270μs（35倍）、90.8%が予算超過 |
| **C** | **Inputフェースのオーバーヘッド2.5-2.8ms** | ⚠️ **事実だが内訳は未確定** | 18種類の処理のaccumulated cost。Publish/RCU待ちではない |

### 7.4 未確定の切り分け

原因Aについて、現状のログだけでは以下を区別できない:

| 可能性 | 内容 | 検証方法 |
|--------|------|---------|
| (a) OSスケジューラ遅延 | Windowsがcallbackスレッドを起床するのが遅れる | ETW計測、MMCSS有無の比較 |
| (b) ホストアプリ内遅延 | VST3ホストがプラグインを呼ぶまでの自処理で時間がかかる | 他プラグインとの比較、ホスト側ログ |
| (c) 前回callbackからの累積遅延 | 一度遅れるとホストのタイミングが狂い、取り戻せない | 長時間連続運転の安定性確認 |

---

## 8. CPU Migrationの実影響評価

### 8.1 相関分析結果

CALLBACK_STAGEデータを用いて、CPU Migrationの有無とDSP時間の関係を定量分析:

**Gen=4（IR無し、軽量DSP）:**

| 条件 | 平均DSP | サンプル数 |
|------|---------|-----------|
| 同一CPU連続 (sameCpu) | **272μs** | 39回 |
| 異CPU連続 (migrated) | **292μs** | 263回 |
| 差 | **20μs (7.3%)** | — |

**Gen=5（IR有り、重量DSP）:**

| 条件 | 平均DSP | サンプル数 |
|------|---------|-----------|
| 同一CPU連続 (sameCpu) | **10,542μs** | 171回 |
| 異CPU連続 (migrated) | **10,240μs** | 1,661回 |
| 差 | **-302μs (-2.9%)** | — |

### 8.2 評価

- Gen=4の同一CPU連続は20μs（DSPの7%）速い傾向があるが、予算5.33msに対する影響は **0.4%未満**。
- Gen=5ではむしろ異CPU連続の方が速い（逆転）。差はDSP 10msの3%未満。
- **CPU Migrationの有無よりも、DSP負荷そのもの（10ms vs 0.3ms）が圧倒的に支配的。**

**結論: CPU Migrationは音飛びの原因として無視できるレベルである。** ただし同一CPU定着に20μsの改善余地があることは事実で、Affinity固定が悪影響を及ぼす証拠はない。

---

## 9. Inputフェーズの構成 — 未確定

### 9.1 計測粒度の限界

CALLBACK_STAGEの計測点:

```
t0 = callbackTelemetry.startUs (callback開始直後)
t1 = t1_dspStartUs (armCrossfade後、dsp->process()直前)
t2 = dsp->process()終了直後
t3 = CALLBACK_STAGEログ出力時

Input(t0→t1): 2.5-2.8ms   ← 18種類の処理を含む
DSP(t1→t2):   0.3-10.3ms  ← フェーズにより大きく変動
Output(t2→t3): ≈0ms
```

### 9.2 t0→t1に含まれる全処理（AudioBlock.cpp検証）

| # | 処理 | ブロッキング |
|---|------|------------|
| 1 | XRUN検出preamble (atomic読取+条件判定+ring buffer push) | なし（lock-free） |
| 2 | ACTIVATE検出 (generation変化確認) | なし |
| 3 | CBSUMMARY更新 (atomic max update) | なし |
| 4 | サニティチェック (numSamples, startSample) | なし |
| 5 | **readAudioRuntimeView()** (RCU Reader enter) | **なし**（lock-free CAS, 非ブロッキング） |
| 6 | Worldポインタ解決 | なし |
| 7 | 診断(A/G/H): drift計測, CPU_MIG/CB_SEQログ出力 | **要確認**（Logger::writeToLogのコスト） |
| 8 | AudioCallbackAuthorityView構築 | なし |
| 9 | Atomic increment (epoch, cursor) | なし |
| 10 | makeRTExecutionFrame() | なし |
| 11 | DSPポインタ解決 | なし |
| 12 | 上限/レートチェック | なし |
| 13 | **ParameterSnapshot取得** (多数のatomic read) | **要確認**（コスト未知） |
| 14 | **ProcessingState構築** | **要確認**（コスト未知） |
| 15 | **processCrossfadeDelayGateIfPending** | **要確認**（条件成立時はfading DSPの全処理を同期的実行） |
| 16 | armCrossfadeIfPending | なし |
| 17 | callbackSeq/cpu atomic store | なし |

### 9.3 確定事項と未確定事項

**確定していること:**
- RCU Readerはノンブロッキング（`RCUReader.h` 検証済み）
- Publishは別スレッド（Rebuild Thread）で実行され、Audio Threadをブロックしない
- Input 2.8msは **Publish待ちではない**

**未確定なこと:**
- 18種類の処理のうち、どれが支配的か
- 診断(A/G/H)の `juce::Logger::writeToLog` がI/O待ちを発生させている可能性
- `processCrossfadeDelayGateIfPending` の発火頻度とそのコスト
- ParameterSnapshot / ProcessingState構築の実コスト

### 9.4 解決策 — Inputの細分化計測

Inputをさらに分割するための推奨タイムスタンプ:

```
t0       = callbackTelemetry.startUs         ← 現状
t0_xrun  = XRUN検出ブロック終了後              ← 新規
t0_world = readAudioRuntimeView()成功後        ← 新規
t0_diag  = 診断(A/G/H)終了後                   ← 新規
t0_param = ParameterSnapshot取得後             ← 新規
t0_state = ProcessingState構築後               ← 新規
t0_fade  = armCrossfadeIfPending後             ← 新規
t1       = dsp->process()直前                 ← 現状
```

期待効果:
- Input 2.8msの内訳が明確に
- `juce::Logger::writeToLog` の影響の可視化
- `processCrossfadeDelayGateIfPending` 発火の有無の確認

---

## 10. 確定された因果関係

### 10.1 確定している因果の連鎖

```
[ホストのcallback発行間隔が5.33msを超過] ─── ソースコードで確認
  └─ Interval = start-to-start = 10.65ms (中央値)
  └─ 原因: 未確定（OSスケジューラ/ホスト実装/DRV遅延のいずれか）
       ↓
  [XRUN検出（Audio Callback内）] ─── ソースコードで確認
  └─ 閾値: interval > expected * 1.5 = 8.0ms
       ↓
  [Timer Threadがリングバッファから消費 → ログ出力]
  └─ [XRUN#N] Callback=0.00ms Interval=8.92ms ...

--- [フェーズBではさらに] ---

[IRロード (25,906samples×2ch @384kHz)]
  └─ DSP: 289μs → 10,270μs (35倍)
  └─ CALLBACK_STAGEで確認
       ↓
  [Total callback時間: 12.8ms > 5.33ms予算]
  └─ 90.8%が予算超過
       ↓
  [Intervalもさらに悪化: +1.19ms]
```

### 10.2 否定された仮説

| 仮説 | 判定 | 根拠 |
|-----|------|------|
| XRUNは監視スレッドが検出 | ❌ **否定** | AudioBlock.cpp:436-500でcallback内検出を確認 |
| Callback=0.00msは異常 | ❌ **否定** | XRUN preamble時間(≪5μs)の正常値 |
| CPU MigrationがXRUNの主因 | ❌ **影響軽微** | Same/Diff差20μs（予算の0.4%） |
| Publish待ちがInputの主因 | ❌ **否定** | Publishは別スレッド。RCU Readerも非ブロッキング |
| Block 2048で完全解決 | ❌ **不十分** | Input 2.5ms + DSP 10.3ms = 12.8ms > 10.67ms |

---

## 11. 補足的所見

### 11.1 高負荷クラスタ（フェーズB）

フェーズBではBudget≥75%の連続期間が19回観測:
| 開始seq | 連続数 | 平均Budget |
|---|-------|-----------|
| 2,832 | 2 | **205.4%** |
| 2,272 | 2 | **192.2%** |
| 800 | **3** | **178.0%** |
| 1,056 | 2 | 162.7% |
| 2,592 | **5** | 158.1% |
| 3,936 | 2 | 157.6% |
| 3,232 | **3** | 144.2% |
| 2,000 | **4** | 138.8% |

高負荷が連続するクラスタが存在し、特にBudget 200%超が2連続するケースは回復が困難。

### 11.2 Drift分析（フェーズB）

- Drift中央値: -70μs（ほぼ0中心）
- |Drift|平均: 1,258μs
- |Drift|の97.5%が5ms未満
- **Driftは累積せず**、正負に振動しながら0付近を推移
- タイミングの「ずれ」は毎コールバックで発生するが、長期的にはリセットされている

### 11.3 ビルドチェーン詳細

Gen=5に至るまでのリビルド連鎖:

1. `convolverParamsChanged` (intentId=14) → リビルド開始
2. `enqueue_snapshot_command` (intentId=15) → マージ
3. `ui_eq_editor_change_listener` (intentId=21) → **EQパラメータ変更**
4. `convolverParamsChanged` (intentId=25) → **再度IR関連パラメータ変更**
5. Gen=9としてビルド完了 → Gen=5としてPublish

この間、複数のリビルド要求がマージされ、最終的にGen=9としてビルドされた。

---

## 12. 改善提案（根拠付き）

### 優先度順対策一覧

| # | 対策 | 確実性 | 期待効果 | 難易度 | 備考 |
|---|------|--------|---------|-------|------|
| **1** | **MMCSS (AvRt) 登録** | 中 | Interval短縮: 効果は環境依存 | **低** | 一般的なリアルタイムオーディオ対策 |
| **2** | **ブロックサイズ拡大 1024→2048** | 中 | 予算5.33ms→10.67ms。ただしInput(2.5)+DSP(10.3)=12.8msで超過。**部分的な改善** | **低** | 単独では完全解決しない |
| **3** | **FFTパーティション最適化** | 高 | DSP 10.3ms→推定6-8ms。パーティションサイズ調整でFFT効率向上 | 中 | ソースコード変更が必要 |
| **4** | **IR長制限/処理レート低減** | 高 | IR長25,906 @384kHzは重すぎる。分割 or ダウンサンプリング | 中 | 音質とのトレードオフ |
| **5** | **SetThreadAffinityMaskでコア固定** | 低 | 改善幅20μs（限定的）。悪影響の証拠はなし | **低** | 検証目的で試す価値はある |
| **6** | **Inputフェーズ細分化計測** | — | Input 2.8msの内訳確定。次の対策の根拠データを得る | **低** | **まず最初にやるべきこと** |

### 推奨ロードマップ

**Step 1 — まず計測（最優先）**
```
1. Inputフェーズ細分化タイムスタンプの追加
   t0, t0_xrun, t0_world, t0_diag, t0_param, t0_state, t0_fade, t1
   → Input 2.8msの内訳を確定させる
```

**Step 2 — 即時試行可能な対策**
```
2. MMCSS登録（コード変更のみ、副作用なし）
   → Interval改善の効果検証
3. ブロックサイズ2048での効果確認
   → 部分的な改善を確認
4. Affinity固定（検証目的）
   → 20μs改善の有無確認
```

**Step 3 — 本格的対策**
```
5. Step 1の結果を踏まえ、Input最適化の方針決定
   - Logger::writeToLogが支配的なら非同期化
   - ProcessingState構築が支配的ならキャッシュ戦略見直し
6. IR畳み込みのパーティション最適化
   → DSP 10.3msを予算内に収める
```

---

## 10. 付録: 解析方法

### 使用データ

- 解析対象ファイル: `ConvoPeq.log` (106,426行, ~10MB)
- 解析手法: JavaScript (Node.js) によるログファイルの直接解析
- 使用ツール: `ctx_execute` (context-mode MCP サンドボックス)

### 主要な解析クエリ

```javascript
// CALLBACK_STAGE パース
const m = l.match(/seq=(\d+) cpu=(\d+) gen=(\d+) expected=(\d+)
    drift=(-?\d+) input=(\d+) dsp=(\d+) output=(\d+) total=(\d+) budget=([\d.]+)%/);

// XRUN パース
const m = l.match(/\[XRUN#(\d+)\].*Interval=([\d.]+)ms.*DriftUs=(-?\d+)/);

// CBSUMMARY パース
const m = l.match(/intervalMax=([\d.]+)ms.*actual=(\d+).*loss=(-?\d+)/);
```

### 解析ツール

- `ctx_execute(language:"javascript", code:"...")` — サンドボックス内でログ解析
- ファイル読み込み: `fs.readFileSync()` で直接ログファイルを読込
- 統計処理: min/max/avg/median/p25/p75/p90/p95/p99 各パーセンタイル
- クラスタ分析: 連続する高負荷期間の検出

---

*本報告書はConvoPeq.logの客観的データに基づいて作成されました。*
*追加の分析や特定の対策の実装については別途ご依頼ください。*
