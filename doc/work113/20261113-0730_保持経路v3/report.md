# 保持経路の設計 v3（Timer 全数・独立ウィンドウ・Logger 所有・EQEditProcessor）

- 作業日時: 2026-11-13 07:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。
- 置き換え範囲: `doc/work113/20261113-0630_保持経路v2/report.md` の §2（Timer）・§3（Logger）・§8（実装単位）。
- 前稿の誤り: 「UI 部分木を破棄すれば Timer はゼロになる」は成立しない。**4 件の見落とし**があった（§2）。

---

## 0. 判定

```
sticky gate            = v2 のまま（最上段・terminal フラグ消費より前）
Timer                  = 全 11 クラスを棚卸し。AudioEngine::uiEqEditor と独立 DialogWindow 3 種を追加
Logger                 = CLI / 通常で所有経路を分けて明記
DeviceSettings         = 破棄で解除（v2 のまま）
実装 GO / Gate 1 / U-3 = 付与しない
```

---

## 1. 前稿の見落とし（Owner 指摘の確認）

| # | 見落とし | 現行ソースでの確認 |
| --- | --- | --- |
| L-1 | `EQEditProcessor` は UI ではなく `AudioEngine` のメンバ | `AudioEngine.h:2377` に `EQEditProcessor uiEqEditor;`。`EQEditProcessor.cpp:31` で `startTimer(kDebounceMs)`、`:34-39` の `timerCallback` が `audioEngine.submitRebuildIntent(...)` を呼ぶ |
| L-2 | `ConvolverSettingsComponent` は独立 `DialogWindow` | `ConvolverControlPanel.cpp:1102-1112` の `launchAsync()`。パネルが持つのは `SafePointer<DialogWindow> convolverSettingsWindow`（`ConvolverControlPanel.h:140`） |
| L-3 | `NoiseShaperLearningComponent` は独立 `DialogWindow` | `DeviceSettings.cpp:629-640` の `launchAsync()`。`SafePointer` で保持 |
| L-4 | `PeriodicSaver` が学習中に autosave を要求する | `NoiseShaperLearningComponent.h:65-77`（`juce::Timer` 派生のメンバ） |

`ConvolverControlPanel` の 3 ウィンドウは所有形態が異なる。

| ウィンドウ | メンバ | 型 | パネル破棄時の挙動 |
| --- | --- | --- | --- |
| 最適化進捗 | `optimizationProgressWindow` | `unique_ptr<MixedPhaseOptimizationWindow>`（`.h:42`） | 破棄される（Timer も消える） |
| IR 詳細 | `irAdvancedWindow` | `SafePointer<DialogWindow>`（`.h:139`） | **残る** |
| Convolver 設定 | `convolverSettingsWindow` | `SafePointer<DialogWindow>`（`.h:140`） | **残る**（5 Hz Timer が動き続ける） |

---

## 2. Timer の全数棚卸し（11 クラス）

`src/` 全体で `juce::Timer` を継承するクラスを列挙した結果（tests を除く）。

| # | クラス | 実体・所有 | 保持グラフ | 停止方法 |
| --- | --- | --- | --- | --- |
| T1 | `MainWindow` | 自身 | 保持 | `stopTimer()`（既存・自クラス） |
| T2 | `AudioEngine` | 自身 | 保持 | **新 API** `stopTimersForRetention()` |
| T3 | `ConvolverProcessor`（`convolver`） | AudioEngine の値メンバ | 保持 | **新 API**（`stopTimerForRetention()`） |
| T4 | `ConvolverProcessor`（`uiConvolverProcessor`） | AudioEngine の値メンバ | 保持 | 同上 |
| T5 | `EQEditProcessor`（`uiEqEditor`） | **AudioEngine の値メンバ**（L-1） | 保持 | **新 API に含める**（`uiEqEditor.stopTimer()` を内部で呼ぶ） |
| T6 | `SpectrumAnalyzerComponent` | `MainWindow::specAnalyzer` | 破棄 | デストラクタ |
| T7 | `ConvolverControlPanel` | `MainWindow::convolverPanel` | 破棄 | デストラクタ |
| T8 | `MixedPhaseOptimizationComponent` | `optimizationProgressWindow`（unique_ptr） | 破棄 | ウィンドウ破棄 |
| T9 | `ConvolverSettingsComponent` | **独立 DialogWindow**（L-2） | 保持されてしまう | **明示的に閉じる**（§2.2） |
| T10 | `NoiseShaperLearningComponent` | **独立 DialogWindow**（L-3） | 保持されてしまう | **明示的に閉じる** |
| T11 | `PeriodicSaver`（T10 のメンバ） | T10 と同じウィンドウ | T10 と一緒 | T10 を閉じれば止まる |
| T12 | `DeviceSettings` | `MainWindow::deviceSettings` | 破棄 | デストラクタ（`DeviceSettings.cpp:502-504` で timer + 2 listener 解除） |

（T12 まで数えて 12 行だが、T10 と T11 は同一ウィンドウなのでクラス数は 11。）

### 2.1 音声側の停止 API（新設 1 つ＋内部 1 つ）

```
// AudioEngine（新規・Message Thread 専用）
void stopTimersForRetention() noexcept
{
    stopTimer();                       // T2
    uiEqEditor.stopTimer();            // T5 ← 前稿の漏れ
    convolver.stopTimerForRetention();          // T3
    uiConvolverProcessor.stopTimerForRetention(); // T4
}

// ConvolverProcessor（新規・private 継承のため wrapper が要る）
void stopTimerForRetention() noexcept { stopTimer(); }
```

T5 を落とすと、保持中に debounce Timer が発火して `submitRebuildIntent()` を投入し得る。
これが前稿の最大の欠陥であり、本稿で閉じる。

### 2.2 独立 DialogWindow の扱い

保持経路では、パネルを破棄する**前に**独立ウィンドウを明示的に閉じる。
`SafePointer` は閉じると自動で nullptr になるため、null 検査だけで足りる。

```
// ConvolverControlPanel（新規・Message Thread）
void closeDetachedWindowsForRetention()
{
    if (convolverSettingsWindow != nullptr) { convolverSettingsWindow->closeButtonPressed(); convolverSettingsWindow = nullptr; }
    if (irAdvancedWindow      != nullptr) { irAdvancedWindow->closeButtonPressed();      irAdvancedWindow      = nullptr; }
    // optimizationProgressWindow は unique_ptr なので破棄で消える（触らない）
}

// DeviceSettings（新規）
void closeDetachedWindowsForRetention()
{
    if (adaptiveLearningWindow != nullptr) { adaptiveLearningWindow->closeButtonPressed(); adaptiveLearningWindow = nullptr; }
}
```

`DialogWindow::closeButtonPressed()` は既定で自分を削除する（JUCE の `DialogWindow` 実装）。
`SafePointer` は削除時に nullptr になるため、直後の代入は二重防御である。

### 2.3 学習 Worker の副作用（T11 の先）

T10 を閉じれば `PeriodicSaver` は止まる。しかし **学習 Worker 自体は `AudioEngine` の中に残る**。
保持経路では `releaseResources()` が最上段で return するため `stopLearning()` が呼ばれず、
学習が続けばファイル書き込みが継続する。Owner 指摘の「単なる UI 更新にとどまらない副作用」である。

対処: 保持経路で **明示的に学習を止める**。

```
// AudioEngine（新規）
void stopLearningForRetention() noexcept;   // noiseShaperLearner->stopLearning() を呼ぶだけ
```

- 実測: 学習 ON の `stopLearning` は 2.4〜10.2 ms（計測報告 C3、n=4）。
- これは「破棄」ではなく「停止」であり、保持の前提（破棄しない）と矛盾しない。

### 2.4 保持経路で残る Timer

| 種別 | 数 |
| --- | ---: |
| 破棄または明示停止する Timer | 12（T1〜T12） |
| **保持経路で動き続ける Timer** | **0** |

---

## 3. Logger の所有（CLI / 通常の 2 経路）

| モード | current logger | 所有者 | 保持経路での扱い |
| --- | --- | --- | --- |
| CLI（`--cli-log-file`） | `MainWindow::cliLogFileLogger`（`MainWindow.cpp:450-454`） | `MainWindow`（`unique_ptr`） | ウィンドウを保持するので logger も保持。current のまま |
| 通常（`--log-file` 等） | `MainApplication::fileLogger` | `MainApplication`（`unique_ptr`） | `StuckProcessHolder` へ **`release()` で移す**。`reset()` しない |

`MainApplication::shutdown()` の末尾（`:244-247`）を、保持の有無で分岐する。

```
if (StuckProcessHolder::retainedWindowCount() == 0)
{
    juce::Logger::setCurrentLogger (nullptr);
    fileLogger.reset();
}
else
{
    if (fileLogger != nullptr)
        StuckProcessHolder::adoptLogger (fileLogger.release());   // 通常モード
    // CLI モードでは fileLogger が null。MainWindow 側の logger が current のまま残る
    juce::Logger::writeToLog ("[D123] LOGGER_RETAINED (stuck path)");
}
```

- CLI モードでは `MainApplication::fileLogger` は空なので、何も移さなくてよい。
- 通常モードでは移さないと `~MainApplication` で破棄され、current logger が dangling になる。
- どちらのモードでも `setCurrentLogger(nullptr)` を**呼ばない**ことが要点である。

---

## 4. keep/skip 表（v3）

| step | 処理 | 保持経路 |
| --- | --- | --- |
| 0 | `cliLogFileLogger` を Logger から外す | 実行しない |
| 0b | `cliAutomationCallbacksEnabled = false` | 実行する |
| 0c | `setLookAndFeel(nullptr)` | 実行しない |
| 1 | `audioEngine.removeChangeListener(this)` | 実行する |
| 2 | `setAdaptiveAutosaveCallback({})` | 実行する |
| 3 | `setCliProcessingTelemetryEnabled(false)` | 実行する |
| 4 | `setProcessor(nullptr)` | 実行する（gate で no-op） |
| 4b | `audioEngine.stopLearningForRetention()` | **新規**（§2.3） |
| 5 | `stopTimer()`（MainWindow） | 実行する |
| 5b | `audioEngine.stopTimersForRetention()` | **新規**（§2.1。`uiEqEditor` を含む） |
| 5c | `convolverPanel->closeDetachedWindowsForRetention()` | **新規**（§2.2） |
| 5d | `deviceSettings->closeDetachedWindowsForRetention()` | **新規**（§2.2） |
| 6 | `DeviceSettings::saveSettings` | 実行しない |
| 7 | `removeAudioCallback` | 実行する |
| 8 | `closeAudioDevice` | 実行する（無界） |
| 9 | `audioEngineProcessor.reset()` | 実行しない |
| 10 | UI の reset | **実行する** |

順序の根拠: 5b は 5 の直後（UI 破棄前に音声側 Timer を止める）、5c/5d は**パネル破棄の前**でなければならない
（破棄後は `SafePointer` を保持しているオブジェクトが消えるため、ウィンドウを閉じる手段が無くなる）。

---

## 5. 不変条件（更新）

| # | 不変条件 |
| --- | --- |
| I-U5-1〜I-U5-7 | v2 のまま |
| I-U5-8 | 保持経路で動き続ける Timer は 0。T1〜T12 を §2 の方法で停止または破棄する。 |
| I-U5-9 | 保持時は current logger を外さず、所有権を holder へ移す（CLI / 通常で経路を分ける）。 |
| I-U5-10 | UI を破棄した後、`MainWindow` のタイマと ChangeListener は動かない。 |
| I-U5-11 | 独立 DialogWindow（`convolverSettingsWindow` / `irAdvancedWindow` / `adaptiveLearningWindow`）は、パネル破棄の前に閉じる。 |
| I-U5-12 | 保持経路で学習を停止する（`stopLearningForRetention()`）。学習 Worker を残さない。 |

---

## 6. 残余リスク

| # | 内容 | 扱い |
| --- | --- | --- |
| S-1 | Debug のリーク検出器が `MainWindow` を報告 | 意図的保持 |
| S-2 | `closeAudioDevice()` は無界 | 既存挙動（U-13） |
| S-3 | 保持時のログはプロセス終了まで続く | 意図的（§3） |
| S-4 | UI を破棄するので画面が空になる | 過渡状態として許容 |
| S-5 | `stopLearningForRetention()` の join（2.4〜10.2 ms） | 有界。保持判断の後に行う |
| S-6 | 独立ウィンドウを閉じる際の JUCE 内部削除（`closeButtonPressed`） | `SafePointer` で nullptr 化を確認 |
| S-7 | `irAdvancedWindow` は Timer を持たないが UI を残さないため閉じる | 副作用なし |

---

## 7. 実装単位（P8 の更新）

| # | 内容 | 対象 |
| --- | --- | --- |
| P8-a | stuck gate を最上段へ | `AudioEngine.Processing.ReleaseResources.cpp` |
| P8-b | `stopTimersForRetention()`（T2〜T5） | `AudioEngine.h/.cpp`、`ConvolverProcessor.h/.cpp` |
| P8-c | `stopLearningForRetention()` | `AudioEngine.h/.cpp` |
| P8-d | `closeDetachedWindowsForRetention()`（2 クラス） | `ConvolverControlPanel.h/.cpp`、`DeviceSettings.h/.cpp` |
| P8-e | `shutdownAudioOnlyKeepingState()`（§4 の順序） | `MainWindow.h/.cpp` |
| P8-f | logger handoff（CLI / 通常） | `MainApplication.h/.cpp` |
| P8-g | 保持分岐と `markWorkersStuck()` | `MainApplication.cpp` |

---

## 8. 証跡

| 対象 | 位置 |
| --- | --- |
| EQEditProcessor の所有と Timer | `AudioEngine.h:2377`、`EQEditProcessor.h:28-29`、`EQEditProcessor.cpp:31, 34-39` |
| ConvolverSettingsComponent の独立生成 | `ConvolverControlPanel.cpp:1102-1112`、`ConvolverControlPanel.h:140` |
| IR 詳細ウィンドウ | `ConvolverControlPanel.cpp:1069-1079`、`.h:139` |
| 最適化進捗ウィンドウ（unique_ptr） | `ConvolverControlPanel.h:42`、`MixedPhaseOptimizationComponent.h:39` |
| 学習ウィンドウの独立生成 | `DeviceSettings.cpp:629-640` |
| PeriodicSaver | `NoiseShaperLearningComponent.h:65-77` |
| Timer 全数 | `ConvolverControlPanel.h:22`、`ConvolverProcessor.h:66`、`ConvolverSettingsComponent.h:8`、`DeviceSettings.h:32`、`EQEditProcessor.h:29`、`MainWindow.h:21`、`MixedPhaseOptimizationComponent.h:13`、`NoiseShaperLearningComponent.h:11,65`、`SpectrumAnalyzerComponent.h:43`、`AudioEngine.h:605` |
| DeviceSettings の解除 | `DeviceSettings.cpp:420, 502-504` |
| Logger の終了順 | `MainApplication.cpp:229-247` |
| CLI logger の設定 | `MainWindow.cpp:450-454` |
| `~MainWindow` の step | `MainWindow.cpp:1169-1220` |

---

## 9. 停止条件の確認

- production source / test / CMake を本稿のために変更していない。
- build / test / debugger を本稿のために実行していない（計測は別報告）。
- git の変更操作を行っていない。HEAD は `173e6995…`。
- U-3 には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。
