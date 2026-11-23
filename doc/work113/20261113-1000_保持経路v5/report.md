# 保持経路の設計 v5（Timer 停止タイミング・request-only の副作用）

- 作業日時: 2026-11-13 10:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。
- 置き換え範囲: `doc/work113/20261113-0800_保持経路v4/report.md` の §1.2（停止タイミング）・§2（学習停止）・§5（keep/skip）・§6（不変条件）。
- 前稿の未解決 4 点を閉じる:
  1. T12（`DeviceSettings`）の停止タイミングが §1.2（step 10）と §5（step 5b）で矛盾していた。
  2. `NoiseShaperLearningComponent` のデストラクタが `= default`。Timer 停止の担保が未確定だった。
  3. `requestStopOnly()` を「O(1)」と書いた。厳密な時間上限ではない。
  4. 停止要求後の非同期副作用（保存ジョブ・係数公開・autosave）が未定義だった。

---

## 0. 判定

```
Timer 停止タイミング = T12 は step 10（破棄）。step 5b は T2〜T4 のみ
Timer 停止の担保     = juce::Timer のデストラクタ挙動を JUCE 8.0.12 で確認（D-4）
学習停止             = request-only。「join しない・同期的に待たない」。時間上限は主張しない
停止後の副作用       = 進行中の世代公開を許容し、Engine と Logger を終了まで保持する（案 i）
実装 GO / Gate 1 / U-3 = 付与しない
```

---

## 1. Timer 停止タイミングの確定（修正 1）

### 1.1 前稿の矛盾

v4 は §1.2 で T12（`DeviceSettings`）を「step 10 の破棄で停止」とし、§5 の step 5b に
「`stopTimersForRetention()`（T2〜T4、T12 相当を含む）」と書いた。**どちらが正か不明確だった。**

### 1.2 確定

`DeviceSettings` は `AudioEngine` の所有物ではなく `MainWindow::deviceSettings`（`MainWindow.h:73`）である。
自身の 5 Hz Timer を `startTimerHz(5)` で開始し（`DeviceSettings.cpp:497`）、デストラクタで
`stopTimer()` と 2 つの ChangeListener 解除を行う（`:502-504`）。
したがって **step 5b では触らず、step 10 の破棄で停止する。**

| step | 対象 Timer | 方法 |
| --- | --- | --- |
| 5（MainWindow） | T1 | `stopTimer()` |
| 5b（音声側） | T2 `AudioEngine` / T3 `ConvolverProcessor` ×2 / T4 `EQEditProcessor` | `AudioEngine::stopTimersForRetention()` |
| 5c（Convolver の独立窓） | T8 `ConvolverSettingsComponent` / T9 `IRAdvancedSettingsComponent` | 明示 close → 窓削除 |
| 5d（学習の独立窓） | T10 `NoiseShaperLearningComponent` / T11 `PeriodicSaver` | 明示 close → 窓削除 |
| 10（UI 破棄） | T5 `SpectrumAnalyzer` / T6 `ConvolverControlPanel` / T7 `MixedPhaseOptimization` / **T12 `DeviceSettings`** | 破棄（各デストラクタが停止） |

`stopTimersForRetention()` の対象は **T2・T3・T4 の 4 インスタンスのみ**とする（T12 を含めない）。

---

## 2. `NoiseShaperLearningComponent` の Timer 停止（修正 2・D-4）

現行の `NoiseShaperLearningComponent` はデストラクタが `= default`（`NoiseShaperLearningComponent.h:10-11`）。
停止は `juce::Timer` 基底のデストラクタに依存する。

**確認済み（本稿で実ソースを読んだ）。** リポジトリ同梱の JUCE 実装 `juce_Timer.cpp:359-369` で、
`Timer::~Timer()` は `stopTimer();` を呼ぶ。したがって `= default` のデストラクタでも Timer は停止する。

条件が 1 つある。同じ箇所の `jassert` は「Message Thread 以外で Timer を破棄するなら、
事前に `stopTimer()` を呼べ」と要求する。デストラクタは

```
jassert (! isTimerRunning()
         || MessageManager::getInstanceWithoutCreating() == nullptr
         || MessageManager::getInstanceWithoutCreating()->currentThreadHasLockedMessageManager());
```

を評価するため、**破棄は Message Thread 上で行う**必要がある。保持経路の破棄は
`shutdownAudioOnlyKeepingState()`（MT）から実行するので条件を満たす。DialogWindow の削除も
modal 終了に伴い MT で起きる。

対象は T10 `NoiseShaperLearningComponent`（`NoiseShaperLearningComponent.h:10-11`）と
そのメンバ T11 `PeriodicSaver`（`:65-77`）、T8 `ConvolverSettingsComponent`（`ConvolverSettingsComponent.h:8`）、
T9 `IRAdvancedSettingsComponent`（`ConvolverControlPanel.cpp:71-75`）である。
いずれも基底デストラクタで停止する。**明示的な `stopTimer()` の追加は不要**（D-4 完了）。

---

## 3. `requestStopOnly()` の意味（修正 3）

### 3.1 主張の修正

前稿の「所要は O(1)」は撤回する。正しい主張は次のとおり。

> `requestStopOnly()` は停止**要求**を出すだけで、**join しない**。
> したがって Message Thread が同期的に停止を待つことはない。
> 停止完了までの時間は保証しない（Worker の実行状態に依存する）。

### 3.2 評価用 Worker の join は Worker 側で完結する

`NoiseShaperLearner` はメインの `std::jthread` とは別に評価用 Worker を持つ。
現行実装では、メイン Worker の終了処理の末尾で `stopEvaluationWorkers()` が呼ばれ、
その中で評価用スレッドの `join()` が行われる（`NoiseShaperLearner.cpp:1128` → `:566-591`、join は `:580-581`）。

`workerThreadMain` の末尾で呼ばれるため、**join は Worker スレッド上で完結し、Message Thread は待たない。**
これは実装時に「`stopEvaluationWorkers()` が MT から呼ばれていないこと」を確認する（D-5）。

---

## 4. 停止要求後の非同期副作用（修正 4）

### 4.1 現状（裏づけ）

`publishGenerationResult()`（`NoiseShaperLearner.cpp:1602-1657`）は 4 つの副作用を持つ。

| # | 副作用 | 実装 | 停止要求後の挙動 |
| --- | --- | --- | --- |
| E-1 | スナップショット公開 | `stateAuthority_.publishSnapshot()`（`:1607`） | atomic。安全 |
| E-2 | 学習状態の非同期保存 | `g_saveThreadPool.addJob(...)`（`:1626-1631`）。`weakSelf` ガード付き | 保持中は learner が生きているので実行され得る。`learned_state.xml` を書く。プロセス終了が先なら job は破棄される |
| E-3 | 係数の公開 | `MessageManager::callAsync` → `storeLearnedCoeffs` / `setAdaptiveNoiseShaperState` / `requestAdaptiveAutosave`（`:1641-1658`） | MessageManager が生きていれば MT で実行。終了後は `callAsync` が false を返し実行されない |
| E-4 | autosave 要求 | 上記 `requestAdaptiveAutosave()` | step 2 の `setAdaptiveAutosaveCallback({})` により **no-op** |

### 4.2 採用する扱い（案 i）

**進行中の世代公開・保存を許容し、Engine と Logger をプロセス終了まで保持する。**

理由:

- 保持モデルでは `NoiseShaperLearner` も `AudioEngine` も破棄しないため、E-1〜E-3 が走っても
  Use-after-free にはならない。
- 案 ii（停止要求後の世代公開を抑止する commit point を設ける）は、`NoiseShaperLearner` の
  公開経路そのものに条件分岐を入れる変更であり、保持経路の設計範囲を超える。
- E-4 は step 2 で既に無効化される。E-3 は MessageManager 終了後に自然消滅する。
  残るのは E-2 のファイル書き込みだけで、これは「保持中も学習が動き得る」ことの一部である。

### 4.3 残余リスク（具体化）

| # | 内容 | 影響 |
| --- | --- | --- |
| S-L1 | 停止要求後も、進行中の世代で E-2 の保存 job が走り `learned_state.xml` を書く | 終了間際の書き込み。途中終了ならファイルは前回内容のまま |
| S-L2 | 停止要求後も E-3 の係数公開が MT で走る（MessageManager が生きている間） | 保持中の bank が更新される。音声デバイスは閉じているため音には出ない |
| S-L3 | 停止しない Worker が残る | プロセス終了まで残る（保持の前提） |
| S-L4 | `g_saveThreadPool` はファイルスコープのグローバルで、Engine 所有ではない | プロセス終了時の静的破棄で job が破棄され得る（保存が失われる） |

**「停止要求で学習の副作用が止まる」とは主張しない。** 止まるのは「新しい学習の反復」であり、
進行中の 1 世代の公開・保存は残る。

---

## 5. keep/skip 表（v5・確定版）

| step | 処理 | 保持経路 |
| --- | --- | --- |
| 0 | `cliLogFileLogger` を Logger から外す | 実行しない |
| 0b | `cliAutomationCallbacksEnabled = false` | 実行する |
| 1 | `audioEngine.removeChangeListener(this)` | 実行する |
| 2 | `setAdaptiveAutosaveCallback({})` | 実行する（E-4 を無効化） |
| 3 | `setCliProcessingTelemetryEnabled(false)` | 実行する |
| 4 | `setProcessor(nullptr)` | 実行する（stuck gate で no-op） |
| 4b | `requestLearningStopForRetention()`（join しない） | 実行する |
| 5 | `stopTimer()`（T1） | 実行する |
| 5b | `stopTimersForRetention()`（**T2・T3・T4 のみ**） | 実行する |
| 5c | `convolverPanel->closeDetachedWindowsForRetention()`（T8・T9） | 実行する |
| 5d | `deviceSettings->closeDetachedWindowsForRetention()`（T10・T11） | 実行する |
| 6 | `DeviceSettings::saveSettings` | 実行しない |
| 7 | `removeAudioCallback` | 実行する |
| 8 | `closeAudioDevice` | 実行する（無界） |
| 9 | `audioEngineProcessor.reset()` | 実行しない |
| 10 | UI の reset（T5・T6・T7・**T12**） | 実行する |

---

## 6. 不変条件（更新）

| # | 不変条件 |
| --- | --- |
| I-U5-8 | 保持経路で動き続ける Timer は 0。T1〜T12 を §1.2・§5 の方法で停止・破棄する。 |
| I-U5-9 | 保持時は current logger を外さず、所有権を holder へ移す（CLI / 通常で経路を分ける）。 |
| I-U5-10 | UI 破棄後、`MainWindow` のタイマと ChangeListener は動かない。 |
| I-U5-11 | 独立 DialogWindow はパネル破棄の前に閉じる。 |
| I-U5-12 | 学習停止は要求のみ。`join` しない。停止完了は保証しない。 |
| I-U5-13 | 保持経路で `stopLearning()`（join する既存 API）を呼ばない。 |
| I-U5-14 | 独立ウィンドウの閉鎖はデストラクタ実行と Timer 停止で確認する。 |
| I-U5-15 | `stopTimersForRetention()` の対象は T2・T3・T4 のみ。T12 は step 10 の破棄で停止する。 |
| I-U5-16 | 停止要求後の世代公開・保存を抑止しない。保持により安全を担保する（案 i）。 |

---

## 7. 実装時の検証項目（未検証）

| # | 項目 | 方法 |
| --- | --- | --- |
| D-4 | `juce::Timer` のデストラクタが `stopTimer()` を呼ぶこと | **確認済み**（`juce_Timer.cpp:359-369`）。破棄は MT で行う |
| D-5 | `stopEvaluationWorkers()` が MT から呼ばれないこと | 呼出し元を全走査（現状は `workerThreadMain` 末尾のみ） |
| D-1〜D-3 | DialogWindow の閉鎖・削除・デストラクタ実行 | JUCE 実装確認 + 一時ログでの証跡 |
| G-1〜G-3 | Logger の所有権移管 | 実装後に保持経路でログ出力が続くことを確認 |

---

## 8. 証跡

| 対象 | 位置 |
| --- | --- |
| `DeviceSettings` の所有と Timer | `MainWindow.h:73`、`DeviceSettings.cpp:497, 500-504` |
| `NoiseShaperLearningComponent` の dtor | `NoiseShaperLearningComponent.h:10-11`、`PeriodicSaver` は `:65-77` |
| 評価用 Worker の join | `NoiseShaperLearner.cpp:566-591`（join は `:580-581`）、呼出しは `:1128`（`workerThreadMain` 末尾） |
| 停止経路 | `NoiseShaperLearner.cpp:210-224`（`workerThread.join()`） |
| 世代公開の副作用 | `NoiseShaperLearner.cpp:1602-1657`（E-1〜E-4） |
| Timer 全数 | v4 §1.2（12 クラス / 13 インスタンス） |
| keep/skip の元 | `MainWindow.cpp:1169-1220` |
| Logger | `MainApplication.cpp:229-247`、`MainWindow.cpp:450-454` |

---

## 9. 停止条件の確認

- production source / test / CMake を本稿のために変更していない（計測専用変更は別報告）。
- build / test / debugger を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e6995…`。
- U-3 には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。
