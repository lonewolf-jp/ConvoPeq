# ConvoPeq 測定追加改修計画

> 作成日: 2026-06-30
> 根拠: xrun_analysis_report.md v7 の分析結果に基づく
> 目標: 「callback開始以前（OS/Driver）の遅延（drift）」と「callback内部のINPUT/DSP/OUTPUT各区間」の配分を定量し、真因を特定する
> 注意: OS/Driver区間そのものは直接計測不可。既存の `drift` 値がこれを表す。新規計測はcallback内部の3区間（INPUT/DSP/OUTPUT）に限定。

---

## 現状の診断能力と不足点

### 既存の診断タグ

| タグ | 計測対象 | 現状の閾値 | 発火状況 |
|------|---------|-----------|---------|
| `[CB_HIST]` | Callback全体の処理時間/proc, drift, cpu, budget% | なし（常時） | ✅ 正常 |
| `[CPU_MIG]` | コア間Migration | callbackごと | ✅ 正常 |
| `[CB_SEQ]` | PublicationSequence変化 | 変化時のみ | ✅ 正常 |
| `[CONV_TIME]` | Convolver全体処理時間 | max(300, expected×0.2) | ❌ 未発火 |
| `[STCONV_TIME]` | StereoConvolver個別処理時間 | max(200, expected×0.15) | ❌ 未発火 |
| `[EQ_TIME]` | EQ処理時間 | max(100, expected×0.1) | ❌ 未発火 |
| `[ANS_SWITCH]` | ANS係数切替時間 | なし（常時） | 要確認 |
| `[XRUN#]` | XRUNイベント | Interval>5.33ms | ✅ 正常 |

### 不足している計測

| # | 不足 | なぜ必要か | 現状の代替手段 |
|---|------|-----------|--------------|
| 1 | **INPUT_STAGE（callback開始→DSP直前）** | 「DSP以外の前処理」の定量 | なし |
| 2 | **OUTPUT_STAGE（DSP直後→callback終了）** | 「DSP以外の後処理」の定量 | なし |
| 3 | **Convolver/EQの実測値**（閾値未満でも出力） | DSP負荷の定量把握 | 閾値からの推定のみ |
| 4 | **Estimated Scheduling Headroom（drift + expectedInterval - totalProcTime から推定されるスケジューリング余裕度）** | DSP負荷 vs スケジューリング遅延の分離 | なし |

**既存の `drift` 値でカバー済みの領域:**
- OS/Driver区間の遅延 → `[CB_HIST] drift` で取得可能
- 新規計測では **callback内部の3区間（INPUT/DSP/OUTPUT）** のみ追加

### 既存データからの追加分析結果

**Drift × Proc時間の関係（CB_HIST 58,137ユニークcallbackから）:**

| グループ | 定義 | 件数 | P50 | P90 | P95 |
|---------|------|------|-----|-----|-----|
| Early | drift ≤ -1,000μs | 13,842 | 2,592 μs | 6,772 μs | 8,547 μs |
| OnTime | -1,000 < drift < 1,000 | 32,143 | 3,083 μs | 6,788 μs | 8,559 μs |
| Late | drift ≥ +1,000μs | 12,152 | **4,040 μs** | **8,135 μs** | **10,218 μs** |

**発見:**
- **遅延起動(Late)のcallbackは処理時間も31%長い**（P50: 4,040μs vs 3,083μs）
- 早期起動(Early)のcallbackは処理時間が最も短い（P50: 2,592μs）
- これは「システム負荷が高いとdriftとproc時間が同時に増加」を示唆
- **driftとproc時間は独立した要因ではなく、共通の原因（システム負荷・割り込み等）で同時に変動している可能性が高い**

**確認事項:**
- `convo::getCurrentTimeUs()` は `std::chrono::steady_clock::now()` ベース（≒ QueryPerformanceCounter）。1回の呼び出しは約10-100nsで、既にcallback内で5-7回呼ばれており、追加コストは無視できる。
- `CallbackTelemetryScope` の `startUs` はcallback開始直後に取得されるため、INPUT_STAGE計測の起点として利用可能。

---

## Phase 0: 閾値調整（コード変更最小）

### 変更内容

既存の診断タグの出力閾値を大幅に引き下げる。これにより、**コード構造を変えずに**詳細データを得る。

#### 対象ファイルと変更箇所

**① EQ_TIME: `DSPCoreFloat.cpp` / `DSPCoreDouble.cpp`**

```cpp
// 現状:
const double thr = std::max(100.0, expectedUs * 0.1);
// 変更後（収集は毎回、出力はcallbackSeqベースでサンプリング）:
// ★ elapsedUs は毎回計算（閾値の概念は不要: 予算超過判定は分析時に行う）
// ★ callbackSeq は最初に一度だけ取得した値を使い回す
// ★ CONVOPEQ_DIAG_SAMPLE_MASK は CMakeLists.txt の target_compile_definitions で指定する。
//    #ifndef/#define は不要（コンパイル定義として与える）。
//    例: Debug   → -DCONVOPEQ_DIAG_SAMPLE_MASK=0x3（1/4）
//        Release → -DCONVOPEQ_DIAG_SAMPLE_MASK=0xF（1/16）
//        HeavyAnalyze → -DCONVOPEQ_DIAG_SAMPLE_MASK=0x1（1/2）
//
// ★ seq が唯一の結合キー。gen は補助情報（publish切替/XRUN/crossfadeとの相関用）であり、結合には使わない。
// ★ diagLog は (callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK)==0 のときのみ
if ((callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
{
    diagLog("[EQ_TIME]"
        + " seq=" + juce::String(static_cast<int64_t>(callbackSeq))
        + " cpu=" + juce::String(static_cast<int>(currentCpu))
        + " gen=" + juce::String(static_cast<int64_t>(gen))
        + " us=" + juce::String(static_cast<int64_t>(eqElapsedUs))
        + " budget=" + juce::String(budget, 1) + "%");
}
// 注意: elapsedUs 計算は毎回行うが10-100nsで無視可能。
//       文字列生成とdiagLogのみサンプリングする。
//       予算超過（>thr）の概念は撤廃。全データはサンプリングで取得し、分析時に評価する。
```

**サンプリング方針の変更点:**
- thrの概念を撤廃。全データはサンプリングで取得し、分析時に予算超過を評価する
- static counterではなく `callbackSeq` を使用 → CB_HIST/INPUT/DSP/OUTPUTと同一callback番号で突合可能
- 予算超過時の常時出力は行わない。1/16サンプリングで十分な統計が取れる

**② STCONV_TIME: `ConvolverProcessor.Runtime.cpp`**

```cpp
// 現状:
const double thr = std::max(200.0, expectedUs * 0.15);
// 変更後（EQ_TIMEと同様、callbackSeqベースのサンプリング、thr概念撤廃）:
if ((callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    diagLog("[STCONV_TIME]"
        + " seq=" + juce::String(static_cast<int64_t>(callbackSeq))
        + " cpu=" + juce::String(static_cast<int>(currentCpu))
        + " gen=" + juce::String(static_cast<int64_t>(gen))
        + " us=" + ...);
```

**③ CONV_TIME: `ConvolverProcessor.Runtime.cpp`**

```cpp
// 現状:
const double thr = std::max(300.0, expectedUs * 0.2);
// 変更後（STCONV_TIMEと同様）:
if ((callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    diagLog("[CONV_TIME]"
        + " seq=" + juce::String(static_cast<int64_t>(callbackSeq))
        + " cpu=" + juce::String(static_cast<int>(currentCpu))
        + " gen=" + juce::String(static_cast<int64_t>(gen))
        + " us=" + ...);
```

#### リスクと対策

| リスク | 確率 | 対策 |
|--------|------|------|
| ログが大量になりファイルサイズ爆発 | 中 | 1/16サンプリングで抑制。1回の観測は30秒で十分 |
| ログI/Oによるパフォーマンス影響 | 中 | **`diagLog()` は非同期でもformat/string生成/heap確保/queue投入のコストがある。** 数十万回出力で十分影響しうるため、サンプリングは必須。 |
| 既存のCB_HISTと重複 | 低 | 分析時にタグでフィルタ可能 |
| ConvolverProcessorにcallbackSeq/cpuを渡す方法 | 低 | **ConvolverProcessor::process() は AudioBlock.cpp から dsp->process() 経由で呼ばれる。引数追加以外の方法として、ConvolverProcessor側の `process()` 内部で `currentCallbackSeq(rtLocalState_)` を呼ぶ方法がある。ただし rtLocalState_ への参照が必要。現状の `currentSampleRate` / `currentBufferSize`（ConvolverProcessor.h:1074, atomic<int>）と同様のパターンで実現可能。** |

### ソースコード調査による確定事項

以下の実装詳細はソースコード調査（serena/grep）により確認済み:

**① `RTLocalState` に追加するフィールド**
```cpp
// AudioEngine.h 既存 RTLocalState struct 内（line 1260-1296）
uint64_t expectedCallbackIntervalUs { 0 };  // prepareToPlayで設定、callback内で読取専用
```
- `std::atomic` 不要（prepareToPlay + device restart 時のみ更新）

**② `currentCallbackSeq()` helper の配置**
- `AudioEngine.h` または共通インクルードヘッダに static inline 関数として定義
- AudioBlock.cpp: line 374 の `thisCallbackIndex` を流用（二度読みしない）
- BlockDouble.cpp: line 187 (`fetchAddAtomic` の後) に `consumeAtomic` を追加

**③ `logEqTime()` シグネチャ変更（DSPCoreFloat.cpp:25 / DSPCoreDouble.cpp:40）**
```cpp
// Before:
inline void logEqTime(uint64_t eqStartUs, int numSamples, int numChannels,
                      const convo::EQParameters* eqParams,
                      convo::ProcessingOrder order, double sampleRate)
// After:
inline void logEqTime(uint64_t eqStartUs, int numSamples, int numChannels,
                      const convo::EQParameters* eqParams,
                      convo::ProcessingOrder order, double sampleRate,
                      uint64_t callbackSeq, uint32_t cpu, uint64_t gen)
```

**④ CONV_TIME/STCONV_TIME（ConvolverProcessor.Runtime.cpp:653/715）**
- 現状: `thr` ベースの閾値判定。`seq`/`cpu` なし
- 変更: サンプリング＋`seq`/`cpu`/`gen` を追加。セッション内の `currentSampleRate` (atomic) や `currentBufferSize` (atomic) と同様のパターンで値を取得

**⑤ prepareToPlay での expectedCallbackIntervalUs 設定**
- `AudioEngine.Processing.PrepareToPlay.cpp:15`: `samplesPerBlockExpected` + `sampleRate` から計算
```cpp
rtLocalState_.expectedCallbackIntervalUs =
    (sampleRate > 0.0 && samplesPerBlockExpected > 0)
    ? static_cast<uint64_t>(static_cast<double>(samplesPerBlockExpected) / sampleRate * 1e6)
    : 0;
```
- 更新タイミング: prepareToPlay, device restart, setupAudioProcess

**⑥ CPU番号の取得と受け渡し**
- 既存: A-block（AudioBlock.cpp:148）で `::GetCurrentProcessorNumber()` 使用済み
- 方針: callback先頭で1回取得し、引数または `RTLocalState` の `lastCallbackProcessor` (既存atomic) 経由で全タグへ受け渡し

---

#### 期待される効果

- ConvolverとEQの実測処理時間がサンプリング（1/16）で取得可能に
- P90=6,920μsの内訳が判明（CONV何μs、EQ何μs、DSP_STAGEの残り何μs）
- 「DSP以外の処理」の定量が初めて可能に

---

## Phase 1: Callback内部3区間計測 + Scheduling Headroom解析（コード変更あり）

### 設計方針

RAIIスコープオブジェクトは使用しない。**4つのタイムスタンプ（t0〜t3）を単純なuint64_t変数で取得**し、差分計算する。これにより:
- デストラクタ/コンストラクタのオーバーヘッド排除
- 明示的デストラクタ呼び出し（二重破棄の危険）の回避
- インライン展開の阻害要因除去

### 1-1: 計測区間の定義

```
callbackSeq = audioCallbackEpochCounter  ← ★ 最初に一度だけ取得し使い回す
     │
t0 = startUs（CallbackTelemetryScope既存値） ← ★ 新規QPC呼び出し不要
     │
     ▼
[INPUT_STAGE] = t1 - t0  ← callback entry ～ dsp->process() 直前
     │
t1 = now()  ← DSP開始直前
     │
     ▼
[DSP_STAGE] = t2 - t1  ← dsp->process() 内部（CONV+EQ+OTHER_DSP）
     │  ├─ CONV_TIME（既存、内訳）
     │  ├─ EQ_TIME（既存、内訳）
     │  └─ OTHER_DSP = DSP_STAGE - (CONV_TIME + EQ_TIME)
     │
t2 = now()  ← DSP終了直後
     │
     ▼
[OUTPUT_STAGE] = t3 - t2  ← dsp->process() 直後 ～ callback exit
     │
t3 = now()  ← callback終了
     │
     ▼
Estimated Scheduling Headroom = drift + expectedInterval - totalProcTime
    （callback終了から次callback開始までの推定余裕度（headroom）。
     直接観測は不可能なため、drift + expected - proc から推定する合成指標）
```

### 1-2: 実装（4タイムスタンプ方式）

```cpp
// AudioBlock.cpp / BlockDouble.cpp

// ★ diagCallbackSeq:
//   AudioBlock.cpp: 既存の thisCallbackIndex（consumeAtomic直後）をそのまま流用する。
//                   二度目の atomic load は行わない（incrementとの競合を完全に排除）。
//   BlockDouble.cpp / BlockFloat.cpp: fetchAddAtomic後に consumeAtomic を1回追加。
//   統一 helper（BlockDouble/BlockFloat用）:
//   static inline uint64_t currentCallbackSeq(const RTLocalState& rls) noexcept {
//       return convo::consumeAtomic(rls.audioCallbackEpochCounter, std::memory_order_relaxed);
//   }
//   gen: RuntimeWorld generation。publish切替/XRUN/crossfadeとの相関用補助情報（join keyではない）。
uint64_t diagCallbackSeq = 0;
uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    // t0 は CallbackTelemetryScope::startUs を再利用（新規QPC呼び出し不要）
    t0 = callbackTelemetry.startUs;
    // callbackSeq: AudioBlock.cpp では既存の thisCallbackIndex を流用（二度読みしない）。
    // BlockDouble.cpp では currentCallbackSeq(rtLocalState_) helper を使用。
#endif

    // ... authority取得, parameterSnapshot, crossfade準備 ...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    t1 = convo::getCurrentTimeUs();  // INPUT_STAGE終了 / DSP_STAGE開始
#endif

    dsp->process(...);

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    t2 = convo::getCurrentTimeUs();  // DSP_STAGE終了 / OUTPUT_STAGE開始
#endif

    // ... 後処理（crossfade mix, buffer出力, analyzer）...

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    t3 = convo::getCurrentTimeUs();  // OUTPUT_STAGE終了

    // ★ gen / currentCpu: callback開始時点の値を1回だけ取得し、全診断タグで共通利用する。
    //   （publishが途中で発生しても、このcallbackの所属generationは開始時点の値。）
    //   CALLBACK_STAGE / EQ_TIME / CONV_TIME すべてに同一の gen / cpu を渡す。
    const uint64_t gen = (runtimeWorld != nullptr)
        ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
    const uint32_t currentCpu = static_cast<uint32_t>(::GetCurrentProcessorNumber());

    // CONVOPEQ_DIAG_SAMPLE_MASK はコンパイル定義（全タグ共通、CMakeLists.txtで指定）
    if ((diagCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    {
        const uint64_t inputUs = t1 - t0;
        const uint64_t dspUs = t2 - t1;  // DSP_STAGE（CONV+EQ+OTHER_DSPの実測合計）
        const uint64_t outputUs = t3 - t2;
        const uint64_t totalUs = t3 - t0;
        // ★ CALLBACK_STAGE は input/dsp/output/total のみ保持。
        //   CONV/EQは別タグ（CONV_TIME/EQ_TIME）でseqにより結合。
        //   gen は結合に使わない（補助情報: publish切替/XRUN/crossfadeとの相関用）。
        //   OTHER_DSPは解析スクリプトで dsp - conv - eq として算出。
        //   これによりRTコードのファイル間結合を回避する。
        // expectedUs: prepareToPlay 時に事前計算し、RTLocalState のプレーンな uint64_t に保持。
        // 更新タイミング:
        //   - prepareToPlay(int samplesPerBlockExpected, double sampleRate)
        //   - setupAudioProcess / device restart（sampleRate または bufferSize が変わった場合）
        // prepareToPlay 以外では変更されないため、atomic である必要はない。
        // 毎callbackの除算（numSamples/sr*1e6）を避ける目的。
        // ※ RuntimeWorld（RuntimeState）は numSamples を知らないため RTLocalState に持たせる。
        const uint64_t expectedUs = rtLocalState_.expectedCallbackIntervalUs;
        // currentCpu, gen: 上記の callback 開始時点で1回だけ取得済みの値を使用。
        // 各診断タグごとに再取得すると途中 migration で不一致が生じる。
        // drift: 既存の A-block で取得済みの lastCallbackDriftUs を流用。
        // CALLBACK_STAGE に含めることで、別タグ CB_HIST を結合せずに
        // 1タグで input/dsp/output/total/drift 全てを参照可能。
        const int64_t driftUs = convo::consumeAtomic(
            rtLocalState_.lastCallbackDriftUs, std::memory_order_relaxed);
        // budget: total/expected*100。CB_HIST と同様の permille 整数演算でフォーマット。
        // 除算1回のコストは無視可能。解析スクリプトが即座に budget% を比較可能になる。
        // （double budget = 100.0 * totalUs / expectedUs でも同等。全タグで方式を統一する。）
        const uint32_t budgetPermille = (expectedUs > 0)
            ? static_cast<uint32_t>(totalUs * 1000 / expectedUs) : 0;
        diagLog("[CALLBACK_STAGE]"
            + " seq=" + juce::String(static_cast<int64_t>(diagCallbackSeq))
            + " cpu=" + juce::String(static_cast<int>(currentCpu))
            + " gen=" + juce::String(static_cast<int64_t>(gen))
            + " expected=" + juce::String(static_cast<int64_t>(expectedUs))
            + " drift=" + juce::String(static_cast<int64_t>(driftUs))
            + " input=" + juce::String(static_cast<int64_t>(inputUs))
            + " dsp=" + juce::String(static_cast<int64_t>(dspUs))
            + " output=" + juce::String(static_cast<int64_t>(outputUs))
            + " total=" + juce::String(static_cast<int64_t>(totalUs))
            + " budget=" + juce::String(static_cast<int>(budgetPermille / 10)) + "."
            + juce::String(static_cast<int>(budgetPermille % 10)) + "%");
    }
#endif
```

**実装上の注意:**
- `t0 = callbackTelemetry.startUs`: `CallbackTelemetryScope` はローカル構造体で `startUs` はパブリックメンバ。同一関数内から参照可能。
- `diagCallbackSeq`: AudioBlock.cpp では既存の `thisCallbackIndex` をそのまま流用する（二度の atomic load による increment との競合を避ける）。BlockDouble.cpp では `currentCallbackSeq(rtLocalState_)` helper を使用。
- 3回の `getCurrentTimeUs()` 呼び出し追加コスト（t1/t2/t3。t0は既存startUsを流用）: 最大300ns/callback = 約0.006%の予算。無視可能。

```text
Estimated Scheduling Headroom = drift + expectedInterval - totalProcTime
```

これは「次callback開始までの余裕時間」を直接表すものではない（drift は callback 開始時点の位相誤差であり、callback終了→次callback開始の区間は直接観測不可能）。
あくまで drift/expected/proc から逆算した**スケジューリング余裕の推定指標（Scheduling headroom indicator）**であり、以下を全て含む合成指標:
- ドライバ起床遅延
- MMCSSスケジューリング遅延
- 割り込み/DPC遅延
- スレッド切り替えコスト

#### 分析: OTHER_DSP の算出（解析時）

`DSP_STAGE`（CALLBACK_STAGE dsp=）と `CONV_TIME` / `EQ_TIME`（別タグ、seqで結合）から、解析スクリプトで以下の内訳を算出する:

```text
OTHER_DSP = DSP_STAGE - CONV_TIME - EQ_TIME
```

**RTコードでは OTHER_DSP を出力しない。** 解析時に `seq` をキーにCALLBACK_STAGEとCONV_TIME/EQ_TIMEを結合し、外部スクリプトで計算する。これにより:
- RTコードのファイル間結合（AudioBlock.cpp → ConvolverProcessor.Runtime.cpp / DSPCore*.cpp）を回避
- 後から計算式を変更可能（例: STCONV_TIMEも含める等）
- RTオーバーヘッド最小化

OTHER_DSPが大きい場合、gain/limiter/crossover/analyzer/AGC/oversamplingなどの
DSP内部のその他処理が支配的であることを示す。その場合は DSP_STAGE のさらなる分解を検討する。

#### 分析: Estimated Scheduling Headroom

| 状態 | Estimated Scheduling Headroom | 示唆 |
|------|--------------------------|------|
| Margin大 + Drift大 | OS/Driver区間が支配的 | MMCSS昇格を検討 |
| Margin小 + Proc大 | callback内部処理が支配的 | DSP最適化を検討 |
| Margin負（proc > expectedInterval） | 予算超過＝XRUN確定 | どちらかの区間を短縮 |

### 1-4: リスクと注意点

- `~StageTelemetryScope()` の明示的呼び出しは**二重破棄の原因となるため禁止**。代わりに単純なt0〜t3変数と `convo::getCurrentTimeUs()` を使用。
- RAII導入によるオーバーヘッド（デストラクタ/インライン展開阻害/static counter）を排除。
- 3回の `getCurrentTimeUs()` 呼び出し追加コスト（t1/t2/t3。t0は既存startUsを流用）: 最大300ns/callback = 約0.006%の予算。無視可能。

### 1-5: 将来の拡張（必要に応じて）

INPUT_STAGEが支配的と判明した場合、さらに以下に分解可能:

```
INPUT_STAGE
 ├─ RCU取得（makeRuntimeReadHandle / getRuntimeWorldFromReadHandle）
 ├─ ParameterSnapshot取得
 └─ その他（authority作成, crossfade準備）
```

この分解はPhase1の結果を見てから判断する。Practical Stable ISR Runtimeの原則「計測も段階的」に従う。

---

## Phase 2: 計測頻度制御（統合）

サンプリング戦略は全タグで `callbackSeq` ベースに統一する。
Phase0の各タグおよびPhase1のSTAGEログで以下の方針を適用:

### サンプリング条件（全タグ統一）

```cpp
// callbackSeq ベースのサンプリング（全タグで同一条件）
// static counter は使用しない
// CONVOPEQ_DIAG_SAMPLE_MASK は CMakeLists.txt の target_compile_definitions で指定。
// ヘッダ内の #ifndef/#define は不要。
if ((diagCallbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK) == 0)
    diagLog(...);
```

- `CONVOPEQ_DIAG_SAMPLE_MASK` は CMakeLists.txt の target_compile_definitions で指定。ビルドコンフィギュレーションごとに変更可能。
  - Debug: `-DCONVOPEQ_DIAG_SAMPLE_MASK=0x3`（1/4）
  - Release: 未指定（デフォルト 0xF = 1/16）
  - HeavyAnalyze: `-DCONVOPEQ_DIAG_SAMPLE_MASK=0x1`（1/2）
- static counterを使わないことで、全タグの出力callback番号が一致する
- 予算超過時の常時出力は行わない。1/16サンプリングで十分な統計が取れる
- 分析時にサンプリングデータから予算超過率を推計可能

---

## Phase 3: 推奨実行手順

### Step 1: Phase 0 閾値調整（15分）

1. `DSPCoreFloat.cpp` + `DSPCoreDouble.cpp` の `logEqTime` 出力条件を `(callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK)==0` に変更（thr概念撤廃）
2. `ConvolverProcessor.Runtime.cpp` の `STCONV_TIME` + `CONV_TIME` 出力条件を `(callbackSeq & CONVOPEQ_DIAG_SAMPLE_MASK)==0` に変更（thr概念撤廃）
3. Releaseビルド
4. 30秒間起動 → ログ取得
5. 閾値を元に戻す

### Step 2: ログ分析（30分）

6. CONV_TIME / STCONV_TIME / EQ_TIME の実測値分布を解析
7. P50/P90/P95/P99 を計算
8. 6,920μsの内訳を確定（Convolver/EQ/その他）

### Step 3: Phase0の結果評価 — Phase1へ進むかの判定

**注意: DSP_STAGE（t2-t1）はPhase1で初めて取得される。Phase0の時点では存在しない。**

Phase0で得られたCONV_TIME / STCONV_TIME / EQ_TIMEの実測値分布から、以下の判定でPhase1の要否を決定する:

```text
IF (CONV_TIME + EQ_TIME) / expectedUs < 0.1（P95基準: budget 10%未満）
   AND totalProc P95 < expectedUs × 0.3（DSP全体が30%未満）
   AND Drift P95 > expectedUs × 0.4（driftだけが増加傾向）THEN
    -- CONV+EQが予算10%未満 → DSP負荷は明らかに小さい
    -- Phase1（INPUT/DSP/OUTPUT計測）は省略可能
    -- 次の対策（MMCSS昇格 or DirectSound試験）へ直接進む
ELSE
    -- Phase0だけではCONV+EQ以外のDSP（Limiter/Meter/Oversampling/NoiseShaper/
    --   Crossfade/Analyzer/Gain）が隠れている可能性を排除できない
    -- Phase1へ進みINPUT/DSP/OUTPUTの詳細を取得
    -- ★ 不明確な場合は必ずPhase1を実施する（Phase1の追加コストはQPC3回のみ）
END
```

**判定方針:** Phase0では `CONV+EQ` しか計測できず、`OTHER_DSP`（Limiter/Meter/Oversampling/NoiseShaper/Crossfade/Analyzer/Gain等）は不可視である。
Phase1の追加オーバーヘッドはQPC3回（最大300ns/callback）と極小であるため、「迷うケースでは実施」が基本方針。
Phase1省略は**CONV+EQが予算10%未満かつDSP全体30%未満というかなり厳しい条件**でのみ許容する。

| 結果パターン | 判定 | 次のアクション |
|------------|------|--------------|
| CONV+EQ<10% かつ totalProc<30% かつ Drift増加 | DSP負荷小 → OS/Driver要因濃厚 | **Phase1不要。** MMCSS昇格 or DirectSound試験へ |
| いずれかの条件に違反 | OTHER_DSPが隠れている可能性 | **Phase1へ進む。** INPUT/DSP/OUTPUTの内訳を取得 |
| 判定が不明確 | CONV+EQだけではDSP全体の姿が見えない | **Phase1へ進む。** 診断を一度で終わらせる |
| proc > expectedInterval | 予算超過＝XRUN確定 | **Phase1不要。** spb=2048緩和試験へ |

### Step 4: 必要に応じてPhase 1実装（2-4時間）

9. INPUT/DSP/OUTPUTの4タイムスタンプ実装（t0〜t3変数方式、t0はstartUs流用）
10. Estimated Scheduling Headroom解析（drift + expectedInterval - procTime で事後計算）
11. ビルド・検証
12. 再度30秒観測

---

## 見積もり

| Phase | 作業 | 工数 | 難易度 |
|-------|------|------|--------|
| 0 | 閾値調整（3ファイル、定数変更のみ） | 15分 | 低 |
| 1 | 前後処理時間計測追加（2ファイル） | 2-4時間 | 中 |
| 2 | 計測頻度制御 | 30分 | 低 |
| 3 | データ解析 | 1-2時間 | 低 |

**合計**: 最小構成（Phase 0のみ）で **15分**、最大構成（Phase 1-3）で **4-6時間**

---

## 成功基準

1. ✅ CONV_TIME/STCONV_TIME/EQ_TIME が常時出力される
2. ✅ P90=6,920μsの内訳が「Convolver」「EQ」「その他」に分解できている
3. ✅ 「Callback開始前」と「Callback内部」のどちらが支配的か判断できる
4. ✅ 次の対策（MMCSS昇格 / DirectSound / Affinity / spb変更）の優先順位が確定する
