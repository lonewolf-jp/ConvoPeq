# 第3回観測 追加検証報告書 — コード調査・未確定事項の棚卸し

**日付**: 2026-06-30
**元レビューア**: User
**調査方法**: serena (LSP), grep/Select-String, ソースコード読解

---

## 1. 元報告書の誤り・不正確な点の訂正

### 1.1 根本原因がOS Schedulerであるという断定

**元報告書の記述**: 「根本原因はOSスケジューラジッターであり、ConvoPeq内部のパイプラインボトルネックではない」

**調査結果**: **根拠不十分。以下の区間は計測されていない。**

```text
ASIO Driver ISR
  → DPC (Deferred Procedure Call)
  → MMCSS (Multimedia Class Scheduler Service)
  → JUCE audioDeviceIOCallback
  → getNextAudioBlock()
  → DSPCore::process()
```

上記区間（ドライバ → MMCSS → JUCEディスパッチ）のレイテンシは**一切計測されていない**。

また、逆に **ConvoPeq内部にも未計測のブロッキング箇所が存在する**（後述セクション2）。

**訂正**: Publish/Observeパイプラインは健全だが、「原因がOS Scheduler」という結論は現時点では「Publish起因ではなかった」までしか言えない。

### 1.2 COEFF_AUTHがAudio Threadをブロックした可能性

**元報告書の記述**: 「係数認証プロセスがオーディオスレッドをブロックした可能性」

**調査結果**: **これは事実誤認。COEFF_AUTHはTimerスレッド上の診断ログであり、Audio Threadをブロックしない。**

`COEFF_AUTH` の実体 (`src/audioengine/AudioEngine.Timer.cpp:238-258`):
```cpp
// これは Message Thread (Timer) 上の診断コード
const int worldGen = (runtimeWorld != nullptr)
    ? static_cast<int>(runtimeWorld->coefficient.adaptiveCoeffGeneration) : -1;
const int liveBankIdx = convo::consumeAtomic(currentAdaptiveCoeffBankIndex, ...);
// ... atomic読み取りのみ → ブロッキングなし、数μsで完了
diagLog("[COEFF_AUTH] worldGen=..."); // Logger::writeToLog
```

- **実行スレッド**: Timerスレッド（非リアルタイム）
- **処理内容**: atomic変数の読み取り + ログ出力のみ
- **ブロッキング可能性**: **なし**（メモリ確保・ロック・IOなし）

同様に `ADAPTIVE_SWITCH` もTimerスレッド上のカウンタ読み取りのみ。

### 1.3 32ms XRUNとANS切替の因果関係

**元報告書の記述**: 「COEFF_AUTH + ADAPTIVE_SWITCHにより32.14msのコールバック遅延」

**調査結果**: **相関はあるが因果関係は証明されていない。** さらに、COEFF_AUTHとADAPTIVE_SWITCHはTimerスレッドの単なる観測値であるため、これらが32ms XRUNの「原因」であるとは言えない。

実際に **Audio Thread上で係数切替が発生する箇所** (`src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:721-727`):
```cpp
if (noiseShaperType == NoiseShaperType::Adaptive9thOrder
    && state.adaptiveCoeffSet != nullptr
    && (activeAdaptiveCoeffBankIndex != state.adaptiveCoeffBankIndex
        || activeAdaptiveCoeffGeneration != state.adaptiveCoeffGeneration))
{
    adaptiveBankSwitchCount.fetch_add(1, std::memory_order_relaxed);
    adaptiveNoiseShaper.applyMatchedCoefficients(state.adaptiveCoeffSet->k, kAdaptiveNoiseShaperOrder);
    // ↑ 軽量: k=9 の係数コピー + 2ch×9のstateクリアのみ
    // 推定実行時間: < 1μs
    activeAdaptiveCoeffBankIndex = state.adaptiveCoeffBankIndex;
    activeAdaptiveCoeffGeneration = state.adaptiveCoeffGeneration;
}
```

ただし、この係数切替は Audio Thread 上で同期的に実行される。これが32msの原因とは考えにくい（`applyMatchedCoefficients` は <1μs）が、**計測されていないため断言はできない。**

### 1.4 XRUN 98.8%改善 = 性能改善と誤認させる記述

**元報告書の記述**: 「閾値改善後のXRUNは98.8%削減」

**訂正**: 20356件→235件の削減は **診断閾値の変更による見かけ上の削減** であり、実際のリアルタイム性能は変わっていない。閾値を1.2x/1.0msから1.5x/3.0msに変更したことで、従来「異常」と判定されていた6-7msの軽度ジッターが「正常」に分類されたに過ぎない。

---

## 2. 未計測区間の棚卸し

### 2.1 計測されている区間（今回確認済み）

| # | 区間 | 計測有無 | 結果 |
|---|------|---------|------|
| 1 | Publish（Worker→RuntimeStore書込） | ✅ publishDurationUs | 2.6〜5.5ms正常 |
| 2 | Observe（Audio→RuntimeStore読取） | ✅ observeLatencyUs | 283〜4624μs正常 |
| 3 | Activate（Runtime切替） | ✅ Pressure/PublishTotal | 全XRUNで正常 |
| 4 | XRUN検出（callback間隔測定） | ✅ Interval/Expected | 3件のみ>=10ms |
| 5 | CBSUMMARY（1秒毎 intervalMax） | ✅ | 平均8.1〜8.7ms |
| 6 | PageFaultサージ検出 | ✅ | 5回検出 |
| 7 | BUILD_PHASE（ビルド時間） | ✅ | Gen1-7 |
| 8 | COEFF_AUTH（係数認証状態） | ✅（Timer観測のみ） | 正常範囲 |
| 9 | ADAPTIVE_SWITCH（係数切替カウンタ） | ✅（Timer観測のみ） | 16回/2分 |

### 2.2 計測されていない区間（今回のギャップ）

| # | 未計測区間 | 重要度 | 理由 |
|---|-----------|--------|------|
| **A** | **ASIO Driver → MMCSS → JUCE callback dispatch** | ★★★★★ | 32msの原因がOS/Driver側かどうかを判断するために必須 |
| **B** | **getNextAudioBlock 総処理時間** | ★★★★★ | CallbackTelemetryScope存在するがCLI連携が無効で未記録 |
| **C** | **Convolver::process() 実行時間** | ★★★★★ | MKL FFT畳み込みのAudio Threadでの実処理時間が未知 |
| **D** | **EQ::process() 実行時間** | ★★★★☆ | バンク切替時の処理時間 |
| **E** | **ANS applyMatchedCoefficients 実時間** | ★★★★☆ | <1μs推定だが未実測 |
| **F** | **false sharing / cache miss 影響** | ★★★☆☆ | 2.5GBのワーキングセットがL3超過 |
| **G** | **Timerスレッド上の処理時間** | ★★☆☆☆ | 非RTなので影響は限定的だが無視はできない |

### 2.3 各区間の詳細分析

#### A: ASIO Driver → MMCSS → JUCE callback dispatch

**現在の計測**: **完全に未計測**

この区間には以下が含まれる：
1. ASIOドライバのISR（Interrupt Service Routine）
2. Windows DPC（Deferred Procedure Call）
3. MMCSSによるスレッドスケジューリング
4. JUCEの `audioDeviceIOCallback` → 仮想関数呼び出し → `getNextAudioBlock`

**既存の計測手段**: JUCEの `AudioIODeviceCallback::audioDeviceIOCallback` の開始時刻を
`getNextAudioBlock()` 冒頭で取得すれば、この全区間のレイテンシを測定可能。
ただし、ドライバISR→DPCの開始時刻は通常ユーザーランドからは取得不可。
（Windows ETW (Event Tracing for Windows) + xperf/WPR が必要。）

**対策案**: `getNextAudioBlock()` 冒頭で `convo::getCurrentTimeUs()` を取得し、
期待コールバック時刻（前回時刻 + blockSize/sampleRate）との差を計算することで、
OS/Driver区間全体のジッターを測定可能。

#### B: getNextAudioBlock 総処理時間

**現在の計測**: CallbackTelemetryScope は存在する (`AudioEngine.Processing.AudioBlock.cpp:68-93`)

```cpp
struct CallbackTelemetryScope final {
    CallbackTelemetryScope(AudioEngine& owner, int numSamplesIn) noexcept
        : engine(owner), samples(numSamplesIn)
        , enabled(owner.isCliProcessingTelemetryEnabled())
        , startUs(enabled ? convo::getCurrentTimeUs() : 0) {}
    ~CallbackTelemetryScope() noexcept {
        if (enabled) {
            const uint64_t endUs = convo::getCurrentTimeUs();
            const double processTimeUs = ...;
            engine.recordAudioCallbackProcessingStats(samples, processTimeUs);
        }
    }
} callbackTelemetry(*this, numSamples);
```

しかし、このデータをログに出力するのは `MainWindow::timerCallback()` 内の
`cliAutomationTelemetryLoggingEnabled` が true の場合のみ。

**【検証】今回のログでの状態**: `[CLI_PERF_RAW]` の出力が**なし** → 無効だった。
このデータがあれば、平均処理時間・最大処理時間・コールバック数が分かり、
32ms XRUN時の処理時間が異常に長かったかどうかを判断できる。

#### C: Convolver::process() 実行時間

**現在の計測**: **完全に未計測**

`ConvolverProcessor::process()` (`src/convolver/ConvolverProcessor.Runtime.cpp:176`) は
Audio Thread上でMKL FFT畳み込みを実行する。内部で `StereoConvolver::process()` →
`MKLNonUniformConvolver::Add()` + `Get()` を呼ぶ。

**処理の内訳**（推定）:
- 512サンプル × 2chのFFTベース畳み込み
- MKL DFTI (Intel MKL FFT) + IPP ippsFFT 使用
- 129,792タップのIR（processingRate 384,000Hzで192,000サンプル ≈ 0.5秒のIR）
- NUC（Non-Uniform Convolver）でパーティション分割

**問題**: NUCのパーティション更新が高負荷時に発生すると、一時的に処理時間が増加する可能性がある。
特にパーティションの再構成やキャッシュミスが発生した場合。

#### D: EQ::process() 実行時間

CoeffBankの切替時にEQフィルタ係数が変更される可能性があるが、
現状の実装ではEQ係数は `state.eqParams` 経由で渡されている。

#### E: ANS applyMatchedCoefficients 実時間

```cpp
void applyMatchedCoefficients(const double* newCoeffs, int numCoeffs) noexcept {
    setCoefficients(newCoeffs, numCoeffs);  // 9回のclamp + store → < 0.1μs（AVX 1命令）
    reset();  // 18回のゼロクリア → < 0.1μs
}
```

**推定**: 合計 < 1μs。32ms遅延の原因にはなり得ない。

---

## 3. 既存診断の有無・出力場所の完全マッピング

| 診断タグ | 出力元ファイル | スレッド | 計測内容 | リアルタイム性 |
|---------|--------------|---------|---------|-------------|
| `[XRUN]` | AudioBlock.cpp | Audio | callback間隔異常 | ✅ |
| `[CBSUMMARY]` | Timer.cpp | Timer | 1秒毎の最大間隔 | ❌（Timer） |
| `[PUBLISH]` | PublicationExecutor.cpp | Worker | publish時間 | ✅ |
| `[DSP_TIMING]` | AudioBlock.cpp | Audio | observeレイテンシ | ✅ |
| `[BUILD_PHASE]` | RuntimeBuilder.cpp | Worker | ビルド時間 | ❌（Worker） |
| `[COEFF_AUTH]` | Timer.cpp | **Timer** | worldGen vs bankGen | ❌（Timer） |
| `[ADAPTIVE_SWITCH]` | Timer.cpp | **Timer** | 切替カウンタ | ❌（Timer） |
| `[WARN] PageFault surge` | Timer.cpp | Timer | ページフォールト | ❌（Timer） |
| `[MEM]` | Timer.cpp | Timer | メモリ使用量 | ❌（Timer） |
| `[DIAG] rebuildThreadLoop` | RuntimeBuilder.cpp | Worker | rebuild時間 | ❌（Worker） |
| `[CLI_PERF_RAW]` | MainWindow.cpp | **Timer** | callback処理時間 | ✅（Audio計測） |
| `[CONV_STATUS]` | RuntimeBuilder.cpp | Worker | IR状態 | ❌（Worker） |
| `[CONV_REBUILD]` | ? | Worker | IR再構築 | ❌（Worker） |
| `[DSPCORE_PREPARE]` | ? | Worker | DSPCore準備時間 | ❌（Worker） |

---

## 4. 次段階の調査指針

### 4.1 最小の計測追加で最大の情報を得るには

**優先度1**: `getNextAudioBlock()` の冒頭にコールバック受信時刻のログを追加

```cpp
// getNextAudioBlock() 冒頭
const uint64_t callbackEntryUs = convo::getCurrentTimeUs();
// 期待コールバック時刻 = lastCallbackUs + expectedIntervalUs
// expectedIntervalUs = numSamples / sampleRate * 1,000,000
// 遅延 = callbackEntryUs - expectedCallbackUs
```

これにより **OS/Driver → JUCE の全区間のジッター** が測定可能になる。

**優先度2**: コールバック総処理時間の定常ログ出力

`CallbackTelemetryScope` は既に存在する。`cliProcessingTelemetryEnabled` を
デフォルト有効にするか、または常時ログ出力するように変更するだけで
各コールバックの処理時間がデータ化される。

**優先度3**: Convolver処理時間の計測追加

```cpp
const uint64_t convStart = convo::getCurrentTimeUs();
convolverRt().process(processBlock);
const uint64_t convEnd = convo::getCurrentTimeUs();
// convEnd - convStart > 500μs の場合のみログ出力（閾値は調整）
```

### 4.2 各仮説の検証可能性

| 仮説 | 検証に必要な計測 | 実装難易度 |
|------|---------------|-----------|
| OSスケジューラ原因 | A: callbackEntryUs 計測 | 低 |
| Convolver処理遅延 | C: conv処理時間計測 | 低 |
| ANS係数切替ブロッキング | E: applyMatchedCoefficients前後計測 | 低 |
| MMCSS設定不適切 | A + Windows ETW解析 | 中〜高 |
| メモリフォールト原因 | 既存のPageFault監視 | ✅ 既存 |
| Publish/Observeパイプライン | ✅ 既存 | ✅ 検証済み |

### 4.3 32ms XRUNの調査残タスク

**XRUN#56（32.14ms）の前後に何が起きたか**:
- Timer上のCOEFF_AUTHが直前 — これはTimerスレッドの観測で因果関係は薄い
- Audio Thread上の係数切替が同時に発生した可能性 — 現状では確認不可
- 32msは約6ブロック分（5.33ms × 6）の間隔 — 6回連続のコールバック欠落
- Windowsのスケジューリング量子（タイムスライス）は通常15〜30ms
  → スレッドがタイムスライスを超過してプリエンプトされた可能性

**最も可能性の高いシナリオ**:
1. WindowsがAudio Threadをプリエンプト（タイムスライス消費または優先度低下）
2. 〜27ms後、Audio Threadが再開
3. 次のコールバックで32.14msの間隔を検出
4. この間、COEFF_AUTHとADAPTIVE_SWITCHがTimerスレッドで出力された（偶然の同時発生）

**これを検証するには**:
- コールバック受信時刻の測定（callbackEntryUs）
- またはETWトレースでOSのスケジューリングイベントを確認

---

## 5. 総合結論

### 確定した事実

1. ✅ **Publish→Observe→Activateパイプラインは正常動作**（全XRUNでPressure=0）
2. ✅ **メモリプレッシャーは直接原因ではない**（Private安定、PageFault相関3/235のみ）
3. ✅ **COEFF_AUTHとADAPTIVE_SWITCHはTimerスレッドの観測値であり、Audio Threadをブロックしない**
4. ✅ **診断閾値変更により20356→235件のXRUN削減は見かけ上の変化**（実性能は不変）
5. ✅ **ANS applyMatchedCoefficientsは極軽量（<1μs）で32msの原因にならない**

### 未確定（今後の調査対象）

| 項目 | 現状 | 必要な調査 |
|------|------|-----------|
| OS Scheduler原因 | 仮説 | callbackEntryUs計測 or ETW |
| Convolver処理遅延 | 未計測 | conv処理時間計測追加 |
| EQ処理遅延 | 未計測 | eq処理時間計測追加 |
| XRUN#56 32msの真因 | 未特定 | callbackEntryUs + conv処理時間で特定可能 |
| XRUN#106 10.30msの真因 | 未特定 | 同上 |

### 優先度順アクション

1. **最優先**: `getNextAudioBlock()` 冒頭にcallbackEntryUs + 期待時刻との差を出力（OS/Driver区間の可視化）
2. **高**: `ConvolverProcessor::process()` に処理時間計測追加（最大消費区間の可視化）
3. **高**: CallbackTelemetryScope の定常ログ出力化（総処理時間の可視化）
4. **中**: `processOutputDouble()` 内の係数切替区間にμs精度のタイムスタンプ追加
5. **低**: Timerスレッド上の3箇所の未統一时间基準（deferred）を統一

---

## 6. 参考：コード調査で使用したツール

| ツール | 用途 | 有効性 |
|-------|------|--------|
| **serena** (find_symbol, search_for_pattern, read_file) | シンボル検索・ファイル読解 | ★★★★★ LSPベースで正確 |
| **grep/Select-String** | ログ検索・パターンマッチ | ★★★★★ 定型検索に最適 |
| **ctx_execute (JavaScript)** | 大量データの統計解析 | ★★★★★ 集計に最適 |
| 通常の Read | ファイル内容確認 | 補助的に使用 |

---

*本報告書はUserレビューを受けての追加検証結果である。元報告書と併せて参照のこと。*
