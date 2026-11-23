# 保持経路の設計 v2（U-5 修正・READ-ONLY）

- 作業日時: 2026-11-13 06:30 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（HEAD = origin/main、作業ツリー）
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。
- 入力: Owner 判定（2026-11-13・4 回目）。U-5 は条件付き受け入れ。sticky gate の位置・Timer 安全性・
  Logger・DeviceSettings の 4 点が未解決。
- 置き換え範囲: `doc/work113/20261113-0500_保持経路と予算/report.md` の §3（U-5）と §3.8（副作用）。

---

## 0. 判定

```
U-5 の keep/skip 分類 = 維持（ただし step 10 を「実行する」に変更）
sticky gate          = releaseResources() の最上段へ移動（terminal フラグの消費より前）
Timer                = UI 部分木を破棄して解消。engine/processor は公開停止 API を追加
Logger               = S-4 を撤回し、保持時の logger handoff を設計
DeviceSettings       = 破棄により timer と ChangeListener を解除
実装 GO / Gate 1 / U-3 = いずれも付与しない
```

---

## 1. sticky gate の位置（修正 1）

### 1.1 前稿の誤り

前稿は gate を `terminalReleaseRequested_` の exchange（`ReleaseResources.cpp:50-52`）の**後**、
`lifecycleState` の CAS（`:61-90`）の**前**に置いていた。Owner 指摘のとおり、これでは
terminal フラグを消費してから return するため、後続の pass が「terminal か reconfigure か」を誤る。

### 1.2 修正後

```
void AudioEngine::releaseResources()
{
    ASSERT_NON_RT_THREAD();

    // ★ 最上段で判定する。terminal フラグの消費より前。
    if (convo::consumeAtomic(workersStuck_, std::memory_order_acquire))
    {
        diagLog("[DIAG] releaseResources: suppressed (workersStuck)");
        return;                      // 何も消費しない・何も変更しない
    }

    const bool terminalRelease = convo::exchangeAtomic(terminalReleaseRequested_, false, ...);
    ...
}
```

- 位置は `ASSERT_NON_RT_THREAD()` の直後、`:50` の exchange より前。
- これにより、stuck の間は次のいずれも実行されない。
  `terminalReleaseRequested_` の消費 / `lifecycleState` の `Releasing` 遷移 / `lifecycleRuntime_.enterRelease()` /
  `closeAdmission()` / `shutdownRuntime_.transitionTo` / learner 停止 / slot clear / join / drain。
- `releaseResourcesForReconfigure()` へは到達しない（上段で return するため）。reconfigure 用の別 gate は不要。
- terminal フラグは消費されないので、stuck の間に何度 `releaseResources()` が呼ばれても同じ結果になる（冪等）。
- `workersStuck_` は sticky（解除 API を作らない）ため、この gate はプロセス終了まで効き続ける。

### 1.3 副作用として明記

stuck 経路では `releaseResources()` が完全な no-op になるため、`isPrepared` は false にならず、
`lifecycleState` は `Prepared` のまま残る。これは保持の前提（破棄しない）と一致する。
「shutdown 中かどうか」の判定は、前稿どおり `isStopAcceptingWork()`（S1 で真）で行う。

---

## 2. Timer の棚卸しと方針（修正 2）

### 2.1 保持対象になり得る Timer の全数

`juce::Timer` を継承するクラスを `src/` 全体から列挙し、保持グラフに入るものを特定した。

| クラス | 継承 | Timer の用途 | 保持グラフ | 停止手段 |
| --- | --- | --- | --- | --- |
| `MainWindow` | private | 500 ms（CPU 表示） | 保持 | 既存 `stopTimer()`（自クラス内） |
| `AudioEngine` | private | 100 ms（reclaim / 診断） | **保持** | 公開 API が無い → **新設**（§2.4） |
| `ConvolverProcessor`（2 個） | private | load 完了処理 / reclaim 補助 | **保持** | 公開 API が無い → **新設**（§2.4） |
| `SpectrumAnalyzerComponent` | private | スペクトラム更新 | 破棄する | 破棄で解除 |
| `ConvolverControlPanel` | private | UI 更新 | 破棄する | 破棄で解除 |
| `NoiseShaperLearningComponent` | private | `periodicSaver`（300 s） | 破棄する | 破棄で解除 |
| `DeviceSettings` | private | 5 Hz（状態同期） | 破棄する | 破棄で解除（＋ChangeListener 解除） |
| `EQEditProcessor` | private | デバウンス | 破棄する（EQControlPanel 配下） | 破棄で解除 |

`MainWindow` 以外はすべて `private juce::Timer` なので、外側から `stopTimer()` を呼べない（`AudioEngine.h:605`、
`ConvolverProcessor.h:66` で確認）。したがって「保持したまま止める」には公開 API の追加が要る。

### 2.2 方針

1. **UI 部分木は破棄する**（前稿の「UI を保持する」を変更）。対象は `~MainWindow` の step 10 と同じ
   `settingsWindow` / `deviceSettings` / `specAnalyzer` / `eqPanel` / `convolverPanel`。
2. **保持するのは音声側だけ**: `MainWindow` のシェル、`AudioEngine`、`audioEngineProcessor`、
   `audioDeviceManager`、`audioProcessorPlayer`、`cliLogFileLogger`。
3. 音声側の 3 つの Timer（`AudioEngine`、`ConvolverProcessor` ×2）は**公開停止 API で止める**。

### 2.3 UI を破棄してよい根拠

| 確認 | 内容 |
| --- | --- |
| 破棄順序 | `~MainWindow` でも step 10 として最後に破棄している。同じ順序を保持経路で先に実行するだけ |
| nullptr ガード | `MainWindow::changeListenerCallback` は `eqPanel` / `convolverPanel` を nullptr 検査してから触る（`MainWindow.cpp:1242-1245`） |
| Timer 解除 | 各コンポーネントのデストラクタが `stopTimer()` を呼ぶ（`SpectrumAnalyzerComponent` は `:121/:238`、`ConvolverControlPanel` は `:704/:1322`） |
| Listener 解除 | `DeviceSettings` のデストラクタ相当処理が `stopTimer()` + `removeChangeListener(this)` を行う（`DeviceSettings.cpp:502-504`） |
| 唯一性 | `audioDeviceManager` に `addChangeListener` するのは `DeviceSettings` だけ（`DeviceSettings.cpp:420`、全走査で 1 件） |
| 値メンバ | `orderModeBox` などの値メンバは `MainWindow` に残るが、`timerCallback` を止めるので参照されない |

UI を破棄すると `MainWindow` は「空のウィンドウ + 音声状態」になる。保持の目的は診断と破棄の回避であり、
UI の継続表示ではないため、この変更は目的に反しない。

### 2.4 追加する公開 API（2 つ）

```
// AudioEngine（新規）
void stopTimersForRetention() noexcept;   // stopTimer(); convolver.stopTimerForRetention(); uiConvolverProcessor.stopTimerForRetention();

// ConvolverProcessor（新規）
void stopTimerForRetention() noexcept;    // stopTimer();（private 継承のため wrapper が要る）
```

いずれも Message Thread 専用。state を変更せず、タイマ周期だけを止める。停止後は
`ConvolverProcessor` の `timerCallback`（`Lifecycle.cpp:206` の rebuild 拾い）と
`AudioEngine::timerCallback`（`Timer.cpp:584` 以降）が動かないため、保持中の engine 内部活動は
「stuck な Worker 自身」だけになる。

---

## 3. Logger（修正 3）

### 3.1 前稿 S-4 の撤回

前稿は「`cliLogFileLogger` を外さないので保持中もログが残る」としたが、**誤り**である。
`MainApplication::shutdown()` の末尾は次の順で動く（`MainApplication.cpp:229-247`）。

```
mainWindow.reset();                          // 保持経路では release() に置き換わる
...
juce::Logger::setCurrentLogger(nullptr);     // ★ ここで current logger が外れる
fileLogger.reset();                          // ★ MainApplication 側の logger も破棄
```

したがって `cliLogFileLogger` オブジェクトを保持しても、`shutdown()` が終わった時点で
current logger ではなくなり、**保持中の書き込みは捨てられる**（JUCE の Logger は current が null なら何もしない）。

### 3.2 修正後の設計（保持時のみ logger を引き継ぐ）

`shutdown()` の末尾を、保持の有無で分岐する。

```
if (StuckProcessHolder::retainedWindowCount() == 0)
{
    juce::Logger::setCurrentLogger (nullptr);
    fileLogger.reset();
}
else
{
    // 保持した窓の logger を current のまま維持する。
    // MainApplication 側の logger も破棄しない（~MainApplication で消えると dangling になる）。
    StuckProcessHolder::adoptLogger (std::move (fileLogger));
    juce::Logger::writeToLog ("[D123] LOGGER_RETAINED (stuck path)");
}
```

- `cliLogFileLogger` は `MainWindow` と一緒に保持される（`MainWindow.h:100`）。
- `MainApplication::fileLogger` は `unique_ptr` なので、`reset()` せずに holder へ移す。
  移さないと `~MainApplication` で破棄され、current logger が dangling になる。
- 結果、保持経路ではプロセス終了までログが書き込める。stuck の診断に必要。
- 非保持（通常経路）では従来どおり即時に外す。

### 3.3 副作用

保持時のログは「プロセスが強制終了されるまで」書き続けられる。ファイルハンドルも開いたままになる。
これは保持そのものの副作用（§8 S-4 改訂）として明記する。

---

## 4. DeviceSettings の再入・通知（修正 4）

### 4.1 前稿 RE-4 の撤回

前稿は「デバイス変更イベントは状態通知のみ」としたが、`removeAudioCallback()` は
`AudioProcessorPlayer` のオーディオコールバックを外すだけで、`DeviceSettings` の ChangeListener は外れない。
`DeviceSettings::changeListenerCallback`（`DeviceSettings.cpp:581-593`）はデバイス変更時に
`ensureUsableChannelSelection()` を呼ぶため、保持中に動くと「通知のみ」では済まない。

### 4.2 修正後

§2.2 のとおり `DeviceSettings` を破棄する。これで次が同時に解除される。

| 解除対象 | 手段 |
| --- | --- |
| 5 Hz の Timer | デストラクタの `stopTimer()`（`DeviceSettings.cpp:502`） |
| `audioDeviceManager` の ChangeListener | `removeChangeListener(this)`（`:503`） |
| タブバーの ChangeListener | `filterTypeTabs.getTabbedButtonBar().removeChangeListener(this)`（`:504`） |

結果、保持中にデバイス変更通知が届いても、`MainWindow` も `DeviceSettings` も購読していないため
コールバックは走らない（`audioDeviceManager` の ChangeListener は `DeviceSettings` の 1 件だけであることを全走査で確認）。

### 4.3 再入経路の更新

| # | 経路 | 挙動 |
| --- | --- | --- |
| RE-1 | `shutdown()` から 1 回 | `audioOnlyShutdownDone_` で固定 |
| RE-2 | `~MainApplication` 経由 | 保持済みなら `mainWindow == nullptr` で到達しない |
| RE-3 | `closeAudioDevice()` → `audioDeviceStopped` → `releaseResources()` | stuck gate（最上段）で no-op |
| RE-4 | デバイス変更通知 | UI 破棄により購読者ゼロ。`audioDeviceManager` の状態だけが変わる |
| RE-5 | 保持後の二重 `reset()` | 起こさない（呼出し側は `release()` 済み） |
| RE-6 | UI 破棄に伴う `changeListenerCallback` | `MainWindow` は step 1 で解除済み。`nullptr` ガードもある |

---

## 5. keep/skip 表（更新）

`~MainWindow`（`MainWindow.cpp:1169-1220`）に対する最終形。

| step | 処理 | 保持経路 |
| --- | --- | --- |
| 0 | `cliLogFileLogger` を Logger から外す | **実行しない**（§3 で保持） |
| 0b | `cliAutomationCallbacksEnabled = false` | 実行する |
| 0c | `setLookAndFeel(nullptr)` | 実行しない |
| 1 | `audioEngine.removeChangeListener(this)` | 実行する |
| 2 | `setAdaptiveAutosaveCallback({})` | 実行する |
| 3 | `setCliProcessingTelemetryEnabled(false)` | 実行する |
| 4 | `setProcessor(nullptr)` | 実行する（releaseResources は最上段 gate で no-op） |
| 5 | `stopTimer()`（MainWindow） | 実行する |
| 5b | `audioEngine.stopTimersForRetention()` | **新規**（§2.4） |
| 6 | `DeviceSettings::saveSettings` | **実行しない** |
| 7 | `removeAudioCallback` | 実行する |
| 8 | `closeAudioDevice` | 実行する（無界） |
| 9 | `audioEngineProcessor.reset()` | **実行しない**（保持） |
| 10 | UI の reset（settingsWindow / deviceSettings / specAnalyzer / eqPanel / convolverPanel） | **実行する**（変更点。§2.2） |

---

## 6. 不変条件（更新）

| # | 不変条件 |
| --- | --- |
| I-U5-1 | 保持経路で `DeviceSettings::saveSettings` を呼ばない。 |
| I-U5-2 | 保持経路で `audioEngineProcessor.reset()` を呼ばない（音声側は保持）。 |
| I-U5-3 | 保持経路で `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` を実行しない（H-1〜H-4）。 |
| I-U5-4 | `shutdownAudioOnlyKeepingState()` は `audioOnlyShutdownDone_` で 1 回に固定する。 |
| I-U5-5 | 保持経路で `terminalStopState_` / `workersStuck_` を変更しない。 |
| I-U5-6 | 保持経路で `lifecycleState` を `Releasing` にしない。 |
| I-U5-7 | **`AudioEngine::releaseResources()` の stuck gate は最上段（terminal フラグの消費より前）に置く。** |
| I-U5-8 | 保持経路で残す Timer はゼロにする（UI は破棄、音声側は `stopTimersForRetention()`）。 |
| I-U5-9 | 保持時は logger を holder へ引き継ぎ、current logger を維持する。 |
| I-U5-10 | UI を破棄した後、`MainWindow` のタイマと ChangeListener は動かない状態を保つ。 |

---

## 7. 残余リスク（改訂）

| # | 内容 | 影響 | 扱い |
| --- | --- | --- | --- |
| S-1 | Debug のリーク検出器が `MainWindow` を報告 | Debug 終了時の警告 | 意図的保持として許容 |
| S-2 | `closeAudioDevice()` は無界 | 保持経路でも停止しない可能性 | 既存挙動（U-13） |
| S-3 | 保持時のログはプロセス強制終了まで続く | ファイルハンドルが残る | 意図的（§3） |
| S-4 | UI を破棄するため、保持中は画面が空になる | 見た目の変化 | プロセス終了までの過渡状態として許容 |
| S-5 | 保持後に MessageManager が終了すると Worker の `callAsync` は false | 既存分岐で処理 | 追加対応なし |
| S-6 | `eqPanel` / `convolverPanel` を破棄した後の `changeListenerCallback` | step 1 で解除済み + nullptr ガード | 二重防御 |

---

## 8. 実装単位（P8 の内訳）

| # | 内容 | 対象ファイル |
| --- | --- | --- |
| P8-a | stuck gate を `releaseResources()` 最上段へ | `AudioEngine.Processing.ReleaseResources.cpp` |
| P8-b | `stopTimersForRetention()`（AudioEngine / ConvolverProcessor） | `AudioEngine.h`、`AudioEngine.Threading.cpp`、`ConvolverProcessor.h`、`ConvolverProcessor.Lifecycle.cpp` |
| P8-c | `shutdownAudioOnlyKeepingState()`（UI 破棄を含む） | `MainWindow.h/.cpp` |
| P8-d | logger handoff と `StuckProcessHolder` | `MainApplication.h/.cpp` |
| P8-e | 保持経路の分岐と `markWorkersStuck()` | `MainApplication.cpp` |

---

## 9. 証跡

| 対象 | 位置 | 確認内容 |
| --- | --- | --- |
| releaseResources の入口 | `ReleaseResources.cpp:34-58` | `ASSERT_NON_RT_THREAD` → terminal フラグ exchange の順 |
| reconfigure の learner 停止 | 同 `:818-822` | pass 側にも `stopLearning()` がある |
| Timer 継承 | `AudioEngine.h:601-606`、`ConvolverProcessor.h:65-66`、`DeviceSettings.h:30-32`、`SpectrumAnalyzerComponent.h:42-43`、`ConvolverControlPanel.h:19-22`、`NoiseShaperLearningComponent.h:10-11`、`EQEditProcessor.h:28-29` | すべて private 継承 |
| DeviceSettings の解除 | `DeviceSettings.cpp:420, 502-504` | addChangeListener / stopTimer / removeChangeListener |
| ChangeListener の唯一性 | 全走査 | `audioDeviceManager.addChangeListener` は 1 件のみ |
| `~MainWindow` の step | `MainWindow.cpp:1169-1220` | step 0〜10 |
| Logger の終了順 | `MainApplication.cpp:229-247` | `mainWindow.reset()` → `setCurrentLogger(nullptr)` → `fileLogger.reset()` |
| `MainWindow` の nullptr ガード | `MainWindow.cpp:1242-1245` | `eqPanel` / `convolverPanel` |

---

## 10. 停止条件の確認

- production source / test / CMake を本稿のために変更していない。
- build / test / debugger を本稿のために実行していない（計測は別報告）。
- git の変更操作を行っていない。HEAD は `173e69957b046290f7077797782f031ed8633fd0`。
- U-3 には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認は付与しない。
