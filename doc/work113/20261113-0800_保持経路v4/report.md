# 保持経路の設計 v4（Timer 全数訂正・学習停止の非有界性）

- 作業日時: 2026-11-13 08:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。
- 置き換え範囲: `doc/work113/20261113-0730_保持経路v3/report.md` の §2（Timer）・§2.3（学習停止）・§6（残余リスク）。
- 前稿の誤り 2 点:
  1. Timer の全数が「11 クラス / 12 インスタンス」だった。正しくは **12 クラス / 13 インスタンス**（`IRAdvancedSettingsComponent` の 8 Hz が漏れ）。
  2. `stopLearningForRetention()` を「有界」と書いた。実装は `workerThread.join()` で**無界**である。

---

## 0. 判定

```
Timer 全数        = 12 クラス / 13 インスタンスに訂正（IRAdvancedSettingsComponent を追加）
学習停止          = join をやめ、request-only（join しない）方式へ変更 → 保持経路を妨げない
DialogWindow 閉鎖 = 方針維持。JUCE 8.0.12 での実挙動確認を実装条件に追加
Logger            = 方針維持。検証条件 3 点を明記
実装 GO / Gate 1 / U-3 = 付与しない
```

---

## 1. 修正 1 — Timer の全数（12 クラス / 13 インスタンス）

### 1.1 前稿の漏れ

`ConvolverControlPanel.cpp:71-75` に `IRAdvancedSettingsComponent : private juce::Timer` があり、
コンストラクタで `startTimerHz(8)` を実行する（`:185`）。前稿の一覧に入れていなかった。

停止自体は `irAdvancedWindow` を閉じる設計でカバーされるが、**全数棚卸しとしては漏れ**である。

### 1.2 訂正後の全数

| # | クラス（インスタンス数） | 実体・所有者 | 保持時の停止方法 | 停止/破棄のタイミング |
| --- | --- | --- | --- | --- |
| T1 | `MainWindow`（1） | 自身 | `stopTimer()` | `shutdownAudioOnlyKeepingState()` の step 5 |
| T2 | `AudioEngine`（1） | 自身 | `stopTimersForRetention()` 内の `stopTimer()` | 同 step 5b |
| T3 | `ConvolverProcessor`（2: `convolver` / `uiConvolverProcessor`） | `AudioEngine` の値メンバ | `stopTimerForRetention()` ×2 | 同 step 5b |
| T4 | `EQEditProcessor`（1: `uiEqEditor`） | `AudioEngine` の値メンバ（`AudioEngine.h:2377`） | `uiEqEditor.stopTimer()` | 同 step 5b |
| T5 | `SpectrumAnalyzerComponent`（1） | `MainWindow::specAnalyzer` | 破棄（デストラクタが `stopTimer`） | 同 step 10 |
| T6 | `ConvolverControlPanel`（1） | `MainWindow::convolverPanel` | 破棄（同上） | 同 step 10 |
| T7 | `MixedPhaseOptimizationComponent`（1） | `optimizationProgressWindow`（`unique_ptr`） | ウィンドウ破棄 | 同 step 10 |
| T8 | `ConvolverSettingsComponent`（1） | 独立 `DialogWindow`（`SafePointer`） | 明示 close → ウィンドウ削除 | 同 step 5c（パネル破棄より前） |
| T9 | `IRAdvancedSettingsComponent`（1） | 独立 `DialogWindow`（`SafePointer`） | 明示 close → ウィンドウ削除 | 同 step 5c |
| T10 | `NoiseShaperLearningComponent`（1） | 独立 `DialogWindow`（`SafePointer`） | 明示 close → ウィンドウ削除 | 同 step 5d |
| T11 | `PeriodicSaver`（1） | T10 のメンバ | T10 の破棄に伴い停止 | 同 step 5d |
| T12 | `DeviceSettings`（1） | `MainWindow::deviceSettings` | 破棄（デストラクタが `stopTimer` + 2 listener 解除） | 同 step 10 |

合計: **クラス 12 / インスタンス 13**。保持経路で動き続ける Timer は **0**。

### 1.3 停止タイミングの順序制約

`T8`・`T9`・`T10` は「パネルを破棄する前に閉じる」必要がある。破棄後は `SafePointer` を持つオブジェクトが
消えるため、ウィンドウを閉じる手段が無くなる。したがって順序は必ず

```
step 5b（音声側 Timer 停止） → step 5c/5d（独立ウィンドウを閉じる） → step 10（UI 破棄）
```

とする。T11 は T10 の破棄で止まるが、`PeriodicSaver` が学習中に `requestAdaptiveAutosave()` を
呼び得るため、**§2 の学習停止を step 5d より先に済ませる**（step 4b）。

---

## 2. 修正 2 — 学習停止は join しない（非有界性の解消）

### 2.1 前稿の誤り

前稿は `stopLearningForRetention()` を「有界（実測 2.4〜10.2 ms）」と書いた。実装は次のとおりで、
`join()` にタイムアウトが無い。

```
NoiseShaperLearner::shutdownWorkerThread()      // NoiseShaperLearner.cpp:210-224
  stopRequested = true
  workerThread.request_stop()
  evaluationDispatchCv.notify_all(); intervalCv_.notify_all()
  workerThread.join();                          // ★ 無界
```

したがって「実測 2.4〜10.2 ms」は観測値であって上限ではない。学習 Worker が停止しない場合、
保持経路へ移る前で止まってしまう。

### 2.2 修正方針 — request-only（join しない）

`std::jthread` の停止トークンを使い、**要求だけして待たない** API にする。

```
// NoiseShaperLearner（新規）
void requestStopOnly() noexcept           // join しない
{
    if (! workerThread.joinable()) return;
    publishAtomic (stopRequested, true, release);
    workerThread.request_stop();
    evaluationDispatchCv.notify_all();
    intervalCv_.notify_all();
}

// AudioEngine（新規・Message Thread）
void requestLearningStopForRetention() noexcept
{ if (noiseShaperLearner) noiseShaperLearner->requestStopOnly(); }
```

- 所要は O(1)（atomic publish と 2 回の notify のみ）。**join しないので保持経路を妨げない。**
- 学習 Worker は停止要求を観測して自力で抜ける。抜けた後も `std::jthread` は joinable のまま残るが、
  **Engine を破棄しないので `std::jthread` のデストラクタ（request_stop + join）も走らない**（保持の H-2）。
- `stopLearning()`（join する既存 API）は保持経路で**呼ばない**。通常経路では従来どおり使う。

### 2.3 残余リスク（明示）

| # | 内容 | 影響 |
| --- | --- | --- |
| S-L1 | 停止要求を出しても学習 Worker が直ちに抜けない可能性 | 保持中も学習処理が続く。ファイル書き込みが継続し得る |
| S-L2 | 学習 Worker が抜けた後の `join` は誰もしない | スレッド資源はプロセス終了まで残る（保持の前提） |
| S-L3 | `PeriodicSaver` は T10 の破棄で止まるが、学習 Worker 側の書き込みは S-L1 の影響下 | 書き込み頻度は学習モード依存。未計測 |

S-L1 は「有界でない」ことを受け入れるのではなく、**待たないことで保持経路を守る**設計である。
停止の完了は保証しない。この点は不変条件 I-U5-12 に「要求のみ。完了は保証しない」と明記する。

---

## 3. 独立 DialogWindow の閉鎖（実装時の確認条件）

方針は前稿のまま（`closeButtonPressed()` で閉じる）だが、Owner 指摘のとおり
**SafePointer が null になったことは停止の証明にならない**。実装時に次を確認する。

| # | 確認項目 | 方法 |
| --- | --- | --- |
| D-1 | JUCE 8.0.12 の `DialogWindow::closeButtonPressed()` が `setVisible(false)` を行い、modal 終了で削除されること | 使用バージョンのソース（`juce_DialogWindow.cpp`）を実読 |
| D-2 | `launchAsync()` の `enterModalState(true, nullptr, true)`（deleteWhenDismissed）で削除されること | 同上 |
| D-3 | 閉鎖後に各コンポーネントのデストラクタが実行されること | `ConvolverSettingsComponent` / `IRAdvancedSettingsComponent` / `NoiseShaperLearningComponent` のデストラクタに一時ログを入れ、閉鎖後の出力を確認（計測時のみ） |
| D-4 | デストラクタで `stopTimer()` が呼ばれていること | 各クラスのデストラクタ実装を確認（`NoiseShaperLearningComponent` は `stopTimer` の明示が必要か要確認） |
| D-5 | 閉鎖後、`SafePointer` が null であること | `jassert(window == nullptr)` を実装時に追加 |

D-3/D-4 は**設計だけでは閉じない**。実装時のテスト項目として登録する（本稿では未検証と明記）。

---

## 4. Logger の検証条件（実装時）

方針（CLI は `MainWindow::cliLogFileLogger`、通常は `MainApplication::fileLogger` を holder へ移す）は維持。
実装時に次を確認する。

| # | 確認項目 |
| --- | --- |
| G-1 | holder が Logger をプロセス終了まで所有し続ける（`release()` 後に `reset()` しない）。 |
| G-2 | `Logger::setCurrentLogger` が破棄済みオブジェクトを指さない（`~MainApplication` の後に current が null でも dangling でもない）。 |
| G-3 | 保持経路では `~MainWindow` が走らないため、`cliLogFileLogger` の解除・破棄が起きない。 |

---

## 5. keep/skip 表（v4）

| step | 処理 | 保持経路 |
| --- | --- | --- |
| 0 | `cliLogFileLogger` を Logger から外す | 実行しない |
| 0b | `cliAutomationCallbacksEnabled = false` | 実行する |
| 1 | `audioEngine.removeChangeListener(this)` | 実行する |
| 2 | `setAdaptiveAutosaveCallback({})` | 実行する |
| 3 | `setCliProcessingTelemetryEnabled(false)` | 実行する |
| 4 | `setProcessor(nullptr)` | 実行する（stuck gate で no-op） |
| 4b | `audioEngine.requestLearningStopForRetention()` | **新規**（join しない。§2.2） |
| 5 | `stopTimer()`（MainWindow） | 実行する |
| 5b | `audioEngine.stopTimersForRetention()`（T2〜T4、T12 相当を含む） | **新規** |
| 5c | `convolverPanel->closeDetachedWindowsForRetention()`（T8・T9） | **新規** |
| 5d | `deviceSettings->closeDetachedWindowsForRetention()`（T10・T11） | **新規** |
| 6 | `DeviceSettings::saveSettings` | 実行しない |
| 7 | `removeAudioCallback` | 実行する |
| 8 | `closeAudioDevice` | 実行する（無界） |
| 9 | `audioEngineProcessor.reset()` | 実行しない |
| 10 | UI の reset（T5〜T7、T12） | 実行する |

---

## 6. 不変条件（更新）

| # | 不変条件 |
| --- | --- |
| I-U5-8 | 保持経路で動き続ける Timer は 0。T1〜T12（13 インスタンス）を §1.2 の方法で停止・破棄する。 |
| I-U5-11 | 独立 DialogWindow（`convolverSettingsWindow` / `irAdvancedWindow` / `adaptiveLearningWindow`）は、パネル破棄の前に閉じる。 |
| I-U5-12 | 保持経路の学習停止は **要求のみ**。`join` しない。停止完了は保証しない。 |
| I-U5-13 | 保持経路で `stopLearning()`（join する既存 API）を呼ばない。 |
| I-U5-14 | 独立ウィンドウの閉鎖は「SafePointer が null」ではなく「デストラクタ実行と Timer 停止」で確認する（実装時テスト）。 |

---

## 7. 残余リスク（更新）

| # | 内容 | 扱い |
| --- | --- | --- |
| S-1 | Debug のリーク検出器が `MainWindow` を報告 | 意図的保持 |
| S-2 | `closeAudioDevice()` は無界 | 既存挙動 |
| S-3 | 保持時のログはプロセス終了まで続く | 意図的 |
| S-4 | UI を破棄するので画面が空になる | 過渡状態 |
| S-L1〜S-L3 | 学習停止は要求のみ。停止完了は保証しない（§2.3） | **非有界性の受け入れではなく、待たない設計** |
| S-D1 | DialogWindow の閉鎖挙動は JUCE 8.0.12 で実装時に確認 | 未検証（§3） |
| S-G1 | Logger の所有権移管は実装時に確認 | 未検証（§4） |

---

## 8. 実装単位（更新）

| # | 内容 | 対象 |
| --- | --- | --- |
| P8-a | stuck gate を最上段へ | `AudioEngine.Processing.ReleaseResources.cpp` |
| P8-b | `stopTimersForRetention()`（T2〜T4） | `AudioEngine.h/.cpp`、`ConvolverProcessor.h/.cpp` |
| P8-c | `requestStopOnly()` / `requestLearningStopForRetention()`（join しない） | `NoiseShaperLearner.h/.cpp`、`AudioEngine.h/.cpp` |
| P8-d | `closeDetachedWindowsForRetention()`（T8・T9・T10） | `ConvolverControlPanel.h/.cpp`、`DeviceSettings.h/.cpp` |
| P8-e | `shutdownAudioOnlyKeepingState()`（§5 の順序） | `MainWindow.h/.cpp` |
| P8-f | logger handoff（CLI / 通常） | `MainApplication.h/.cpp` |
| P8-g | 保持分岐と `markWorkersStuck()` | `MainApplication.cpp` |

---

## 9. 証跡

| 対象 | 位置 |
| --- | --- |
| `IRAdvancedSettingsComponent` | `ConvolverControlPanel.cpp:71-75`（`private juce::Timer`）、`:185`（`startTimerHz(8)`） |
| 独立ウィンドウ（Convolver） | `ConvolverControlPanel.cpp:1069-1079`（IR 詳細）、`:1102-1112`（設定）、`.h:139-140`（`SafePointer`） |
| 独立ウィンドウ（学習） | `DeviceSettings.cpp:629-640` |
| `PeriodicSaver` | `NoiseShaperLearningComponent.h:65-77` |
| 学習停止（join） | `NoiseShaperLearner.cpp:210-224`（`workerThread.join()`） |
| `std::jthread` | `NoiseShaperLearner.h:274` |
| `uiEqEditor` | `AudioEngine.h:2377`、`EQEditProcessor.h:28-29`、`EQEditProcessor.cpp:31, 34-39` |
| Timer 全数 | §1.2 の 12 クラス（各ヘッダの基底宣言） |
| Logger | `MainApplication.cpp:229-247`、`MainWindow.cpp:450-454` |

---

## 10. 停止条件の確認

- production source / test / CMake を本稿のために変更していない（計測専用変更は別報告）。
- build / test / debugger を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e6995…`。
- U-3 には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。
