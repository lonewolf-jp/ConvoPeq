# 保持経路の設計（U-5）と停止予算の整理（READ-ONLY）

- 作業日時: 2026-11-13 05:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（HEAD = origin/main、作業ツリー）
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。build / test / debugger も実行していない。
- 入力:
  - Owner 判定（2026-11-13・3 回目）: A 条件付き受け入れ、C 修正方針受け入れ、E-5 構造受け入れ（cache 分類の整理が必要）、
    W1 受け入れ、v2-a 全体は保留、実装 GO なし、U-3 未承認。
    次は **U-5 の具体化**、その後 **U-1・U-2 を含む停止予算の整理**。
  - 前稿: `doc/work113/20261113-0400_設計v2a再修正/report.md`

---

## 0. 本稿の範囲

本稿は 3 つを扱う。

1. 前回指摘の 2 点の修正（C の判定順の統一、commit point の 3 分類）。
2. **U-5**: `shutdownAudioOnlyKeepingState()` の設計（副作用・停止対象・保持対象・再入経路）。
3. **U-1 / U-2**: 停止予算の構造を整理し、どこが有界でどこが無界かを確定する。

```
実装 GO / Gate 1 / U-3 = いずれも付与しない
```

---

## 1. 前回指摘の反映

### 1.1 C の判定順の統一（線形化点を state の読み取りに置く）

前稿はコードが epoch → state の順、説明が「state → epoch」で食い違っていた。**統一する。**

```
// commit 直前（MT）。RAII ラッパーは生成済み。
if (convo::consumeAtomic(shutdownEpoch_, std::memory_order_acquire) != capturedEpoch)
    return;                                  // ① 世代が変わった（enqueue 後に閉鎖が完了した）

if (isStopAcceptingWork())
    return;                                  // ② ここが線形化点。以降の commit は許される
```

**線形化点は ② の `isStopAcceptingWork()` の読み取り**である。その意味は次のとおり。

- ② が false を返した時点で、`terminalStopState_` は `Idle`、かつ `shutdownPhase < StopAcceptingWork` である。
- ② の後に Intent が publish されても、commit は MT 上で完走する。teardown も MT なので割り込めない（E-1/§4.5）。
  「② の後に立った Intent」は、この commit が終わってから閉鎖処理に入る。
- ① を先に置くのは、閉鎖後の世代を早期に弾くためである。① は安全性の荷重を負わない（前稿 §2.3 の整理を維持）。

前稿の説明文（「state → epoch」）は削除し、本節の順序に統一する。

### 1.2 C の適用先に E5-9 を明記

C の判定（① ②）は **C2〜C7 の callback と E5-9（`executePendingCommit` 入口）の両方**に適用する。
E5-9 は callback 経路（`LoadPipeline.cpp:851`）だけでなく同期経路（`:834`、`applyNewState` 経由）も通るため、
callback 側の判定だけでは同期経路を守れない。前稿 §3.2 の choke point の記述を、C の章からも参照する形にした。

---

## 2. commit point の 3 分類

### 2.1 分類の定義

前稿は「外部から観測可能な状態を変更する直前」を commit point と定義したため、cache 書き込みを
commit point に含めながら「Intent 後も完走させる」と記述して矛盾していた。3 つに分ける。

| 分類 | 定義 | Intent 後の扱い |
| --- | --- | --- |
| **RC: Runtime commit** | engine の実体・IR metadata・publication・`irFinalized` を変更する | **開始しない**。開始済みなら完走させる |
| **SE: 許容する副作用** | 変換・cache 入出力・ログ・atomic フラグの後始末。runtime の公開状態を変えない | **完走させる**（途中中断の仕組みが無い） |
| **WS: Worker 起動** | 新しいスレッドを生成・起動する | **拒否する**。起動済みなら `requestWorkerStop()` で回収する |

### 2.2 9 箇所の再分類

| # | 位置 | 分類 | gate |
| --- | --- | --- | --- |
| E5-1 | `LoadPipeline.cpp:284` cache direct の `applyComputedIR()` | RC | 呼出し前 + E5-7 |
| E5-2 | `LoadPipeline.cpp:312` cached low の `applyComputedIR()` | RC | 呼出し前 + E5-7 |
| E5-3 | `LoadPipeline.cpp:362` 変換結果の `applyComputedIR()` | RC | 呼出し前 + E5-7 |
| E5-4 | `LoadPipeline.cpp:371` `startProgressiveUpgrade()` | **WS**（前稿は RC と誤分類） | 呼出し前 + 関数入口 `:120` |
| E5-5 | `LoadPipeline.cpp:85/93` → `:99` `activeLoader` → `startThread()` | **WS** | 関数入口 + `:99` 直前 |
| E5-6 | `LoadPipeline.cpp:129` → `:142` `upgradeThread` → `startThread()` | **WS** | 関数入口 + `:142` 直前 |
| E5-7 | `LoadPipeline.cpp:460` `applyComputedIR()` 入口 | RC（choke point） | 1 回（`:470` の直前） |
| E5-8 | `LoaderThread.cpp:271-342` / `:229-268` worker 側 finalize | RC | C3（epoch + state）+ worker 停止 |
| E5-9 | `LoadPipeline.cpp:892` `executePendingCommit()` 入口 | RC（choke point） | 1 回（`:894` の直後） |

**cache 書き込みは RC から外す。** `cacheManager->save()`（`LoadPipeline.cpp:350`）と
`irConverter->convertFile()`（`:328`）は SE である。runtime の公開状態を変えないため、
Intent 後も完走させてよい。前稿 §3.4 の「計算は完走、適用は拒否」はこの分類で読み替える。

### 2.3 4 つの規則（分類に対応）

| # | 規則 |
| --- | --- |
| R-RC | RC は開始前に ① ② を評価する。開始後は完走させる（中断すると IRState と engine が半端になる） |
| R-SE | SE は Intent 後も完走させる。所要は実測値（変換 16〜20 ms、cache save 31〜61 ms）を使う |
| R-WS | WS は Intent 後に開始しない。開始済みなら `requestWorkerStop()` の対象にする（前稿 §1.3） |
| R-LC | 後始末（`isLoading` / `isRebuilding` / `latencyChangePending` の解除）は抑止しない |

---

## 3. U-5 — `shutdownAudioOnlyKeepingState()` の設計

### 3.1 現行 `~MainWindow` の 10 ステップ（実測）

`MainWindow.cpp:1169-1220` の実体。保持経路はこのうち**どこまで実行するか**を決める問題である。

| step | 処理 | 行 | 保持経路での扱い |
| --- | --- | --- | --- |
| 0 | `cliLogFileLogger` を Logger から外す | 1172-1173 | **実行しない**（保持中もログを残す） |
| 0b | `cliAutomationCallbacksEnabled = false` | 1175 | **実行する** |
| 0c | `orderModeBox.setLookAndFeel(nullptr)` | 1177 | 実行しない（UI を残す） |
| 1 | `audioEngine.removeChangeListener(this)` | 1188 | **実行する** |
| 2 | `audioEngine.setAdaptiveAutosaveCallback({})` | 1190 | **実行する** |
| 3 | `audioEngine.setCliProcessingTelemetryEnabled(false)` | 1192-1193 | **実行する** |
| 4 | `audioProcessorPlayer.setProcessor(nullptr)` | 1196 | **実行する**（`releaseResources` は stuck で非破壊） |
| 5 | `stopTimer()`（MainWindow の 500 ms タイマ） | 1198 | **実行する** |
| 6 | `DeviceSettings::saveSettings(...)` | 1201 | **実行しない**（§3.5） |
| 7 | `audioDeviceManager.removeAudioCallback(&audioProcessorPlayer)` | 1205 | **実行する** |
| 8 | `audioDeviceManager.closeAudioDevice()` | 1209 | **実行する**（無界・§3.7） |
| 9 | `audioEngineProcessor.reset()` | 1211 | **実行しない**（保持対象） |
| 10 | UI の reset（settingsWindow / deviceSettings / specAnalyzer / eqPanel / convolverPanel） | 1214-1218 | **実行しない**（保持対象） |

### 3.2 停止するもの／保持するもの

| 区分 | 対象 | 理由 |
| --- | --- | --- |
| **停止する** | オーディオコールバック、デバイス、MainWindow のタイマ、engine への ChangeListener、自動保存コールバック、CLI telemetry | 保持中に外部から状態を動かさない |
| **保持する** | `AudioEngine`（値メンバ）、`audioEngineProcessor`、`audioDeviceManager`、`audioProcessorPlayer`、UI パネル、`settingsWindow`、`cliLogFileLogger`、stuck な Worker | 破棄すると Worker の依存状態が壊れる |

### 3.3 変更してよい状態／いけない状態

| 状態 | 変更 | 根拠 |
| --- | --- | --- |
| `audioDeviceManager` のデバイス | 閉じてよい | 停止が目的 |
| `audioProcessorPlayer` の processor | `nullptr` にしてよい | コールバック経路を切る |
| `audioEngine` の lifecycle 状態 | **変えない**（`releaseResources` が stuck で早期 return するため、`lifecycleState` は `Prepared` のまま） | 破棄を実行していないので「解放済み」と記録しない |
| `terminalStopState_` | `Closed` のまま | 一方向 |
| `workersStuck_` | `true` のまま | sticky |
| `IRState` / router / epoch domain / DSP | **触らない** | 保持対象 |
| `DeviceSettings`（永続ファイル） | **書かない** | §3.5 |
| `isPrepared` | 変えない（`releaseResources` が return するので false にならない） | 同上 |

### 3.4 実装形（設計）

```
void MainWindow::shutdownAudioOnlyKeepingState()   // MT 専用・冪等
{
    if (convo::exchangeAtomic(audioOnlyShutdownDone_, true, std::memory_order_acq_rel))
        return;                                     // 2 回目以降は何もしない

    convo::publishAtomic(cliAutomationCallbacksEnabled, false, std::memory_order_release);
    audioEngine.removeChangeListener (this);
    audioEngine.setAdaptiveAutosaveCallback ({});
    audioEngine.setCliProcessingTelemetryEnabled (false);

    audioProcessorPlayer.setProcessor (nullptr);     // → releaseResources(terminal) は stuck で return
    stopTimer();                                     // MainWindow の 500 ms タイマ

    // ★ DeviceSettings::saveSettings は呼ばない（§3.5）

    audioDeviceManager.removeAudioCallback (&audioProcessorPlayer);
    audioDeviceManager.closeAudioDevice();           // 無界（§3.7）

    // audioEngineProcessor / UI / logger は破棄しない
}
```

`audioOnlyShutdownDone_` は `MainWindow` の `std::atomic<bool>`（新規）。保持経路の再入を 1 回に固定する。

### 3.5 `DeviceSettings::saveSettings` を呼ばない理由（副作用）

1. 保持経路では engine が「解放済み」ではない中途状態にある。この状態から設定を永続化すると、
   次回起動時に不整合な設定（例: 閉じたデバイス構成）を読み込む可能性がある。
2. このビルドの `saveSettings` は学習ノイズシェイパの状態ファイル
   （`%APPDATA%\ConvoPeq\noise_shaper_learn.xml`、278 KB / 180 要素）を扱い、既存の実測で **+10.2 秒**を要する
   （`doc/work113/20261110-1313_終了遅延調査/report.md`、AGENTS.md 記載）。
   保持経路は「停止に失敗したときの最終手段」であり、ここで 10 秒を追加する理由が無い。
3. 設定の保存は通常経路（S2）で従来どおり行われる。失われるのは stuck 時だけであり、
   その場合でも次回起動は既定値／前回の保存値で立ち上がる。

**副作用として明記**: stuck 時のデバイス設定・IR 設定は永続化されない。

### 3.6 再入経路

| # | 経路 | 挙動 |
| --- | --- | --- |
| RE-1 | `MainApplication::shutdown()` から 1 回呼ぶ | 通常。`audioOnlyShutdownDone_` で 1 回に固定 |
| RE-2 | `MainApplication` のデストラクタ経由（`shutdown()` を経ない異常系） | 同じ関数を呼ぶ。`mainWindow` が保持済みなら `nullptr` なので到達しない |
| RE-3 | `closeAudioDevice()` → `audioDeviceStopped` → `releaseResources()` | `setProcessor(nullptr)` 済みなので processor は `nullptr`。呼ばれても stuck で return |
| RE-4 | デバイス変更イベント（保持中に OS が発火） | `removeAudioCallback` 済みなので player には届かない。`audioDeviceManager` の状態通知のみ |
| RE-5 | 保持後に `mainWindow.reset()` が再度呼ばれる | **起こしてはならない**。呼出し側は `release()` 後に `unique_ptr` を空にする |

### 3.7 保持の成立条件（重要）

保持経路が成立するには、次の 4 つが同時に満たされる必要がある。

| # | 条件 | 理由 |
| --- | --- | --- |
| H-1 | `MainWindow` は**ヒープ上**に残し、デストラクタを実行しない | `~MainWindow` が `releaseResources` / `closeAudioDevice` を再実行し、破棄を進めてしまう |
| H-2 | `AudioEngine`（`MainWindow` の値メンバ）も破棄しない | `~AudioEngine` は W1〜W4 の `join` を再度実行する（`CtorDtor.cpp:173-177`）。stuck なら永久に待つ |
| H-3 | joinable な `std::thread` のデストラクタを実行しない | `std::thread` は joinable のまま破棄されると `std::terminate` する。H-1/H-2 がこれを保証する |
| H-4 | 保持ポインタは静的破棄されない | `static auto* v = new std::vector<MainWindow*>()`（前稿 §2.6）。`unique_ptr` の静的保持だと終了時に `~MainWindow` が走る |

H-3 は見落としやすい。**「保持する」は「メモリを残す」ではなく「デストラクタを実行しない」**である。

### 3.8 保持経路の副作用・残余リスク（明示）

| # | 内容 | 影響 | 扱い |
| --- | --- | --- | --- |
| S-1 | `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR`（`MainWindow.h:102`）が Debug でリークを報告する | Debug ビルドの終了時に `MainWindow` 1 件のリーク警告 | **意図的な保持**として許容。リリースビルドでは無効 |
| S-2 | `AudioEngine` の 100 ms タイマは止めない（`juce::Timer` を private 継承しており、`stopTimer()` は外部から呼べない） | 保持中も `driveRetireReclaimNonRt` 等が動く。タイマ経路の publication は admission Closed のため `tryAdmit` で拒否される（`ISRRuntimePublicationCoordinator.h:785` の入口 CAS） | 保持対象内の活動なので許容。止める必要が出たら public な停止 API を追加（U-11） |
| S-3 | `closeAudioDevice()` は無界（ASIO ドライバの close が返らない可能性） | 保持経路でも停止しない可能性 | 既存挙動と同じ。U-13 として記録 |
| S-4 | `cliLogFileLogger` を外さないのでログファイルは開いたまま | プロセス終了まで書き込み可能 | 望ましい（stuck の診断ログが残る） |
| S-5 | 保持後、MessageManager が終了すると Worker の `callAsync` は false を返す | C1/C3 の分岐で処理済み | 追加対応なし |
| S-6 | 保持した `MainWindow` は UI として応答しない（タイマ停止・コールバック解除） | 画面は最後の状態で残る | プロセス終了までなので許容 |

---

## 4. 停止予算の整理（U-1 / U-2）

### 4.1 shutdown の時間構造

```
[同期区間の残り]  →  [A] 閉鎖  →  [B] preflight  →  [C] 通常 teardown  または  [D] 保持
   102.6〜144.0 ms      O(1)        B_pre             drain + join + close      close
   334〜638 ms
```

| フェーズ | 内容 | 有界性 | 根拠 |
| --- | --- | --- | --- |
| 同期区間 | `loadIR` / `rebuildAllIRsSynchronous` | **無界**（IR 長・位相モード依存） | 計測 102.6〜144.0 ms / 334〜638 ms。`bound` を適用しない（E-2） |
| A 閉鎖 | `beginTerminalStopAcceptingWork()` | 有界（待たない） | 設計 |
| B preflight | W1・W2・W5・W6 の停止要求と待機 | **待機は `B_pre` で打ち切る**（有界）。ただし W1 のスレッド終了自体は保証しない（C の join は無界） | U-1（未計測） |
| C 通常 teardown | `mainWindow.reset()` → `releaseResources` | **無界**（下記） | 既存実装 |
| D 保持 | `shutdownAudioOnlyKeepingState()` | **無界**（close のみ） | §3.7/S-3 |

### 4.2 C の中身（無界の内訳）

| 処理 | 位置 | 有界性 |
| --- | --- | --- |
| `stopLearning()`（learner の join） | `ReleaseResources.cpp:146-150` | **未計測**（U-2） |
| `shutdownCoordinatorLoop()`（`stopThread(2000)`） | `:237` | 2000 ms で有界（超過時は強制終了） |
| `stopRebuildThread()`（無期限 `join`） | `:238` | **無界**（U-8） |
| graceful drain（`kGracefulDrainMaxMs`） | `:281` | 5000 ms で有界 |
| `uiConvolverProcessor.releaseResources()`（W5/W6 の `stopThread`） | `:558` | 500/2000 ms で有界（preflight 済みなら即時） |
| `closeAudioDevice()` | `MainWindow.cpp:1209` | **無界**（U-13） |

### 4.3 `kGracefulDrainMaxMs = 5000` の位置づけ

- これは **drain の待機上限**であり、shutdown 全体の上限ではない。
- v2-a の `B_pre` に 5000 を暫定採用するのは「既存値と同じ桁」という理由だけであり、
  **実測された Worker 停止時間ではない**。5 秒を「停止保証」として報告・文書化してはならない。
- 総 shutdown 時間は **有界ではない**（4.2 の 2 つの無界項）。設計として保証できるのは
  「A と B が有界」「C と D の無界項は既存挙動」までである。

### 4.4 U-2（`stopLearning()`）の位置づけ

`stopLearning()` は 2 箇所で呼ばれる。

| 経路 | 位置 | 保持経路での扱い |
| --- | --- | --- |
| terminal pass | `ReleaseResources.cpp:146-150` | stuck で入口 return するため**呼ばれない** |
| reconfigure pass | `:818-822` | stuck で入口 return するため**呼ばれない** |

つまり U-2 は **通常経路（C）だけ**の項である。learner の join が長引くと C が伸びる。
所要は未計測であり、`B_pre` とは独立に「C の無界項」として扱う。

### 4.5 計測計画（U-1・U-2 を閉じるために何を測るか）

| 対象 | 方法 | 必要な承認 |
| --- | --- | --- |
| W2 の停止時間 | 既存 `[DIAG] releaseResources` の前後に時刻を付けて差を取る（時刻は現在のログに無い） | なし（ログ追加のみ） |
| W5・W6 の停止時間 | 走行中の Worker を再現する必要がある（CLI スイッチ or 設定の一時切替） | **計測専用の変更承認**（前回未承認） |
| `stopLearning()` の join | 同じく releaseResources 経路のスパン計測 | 上と同じ |
| 同期区間 | 前回の計測で取得済み（102.6〜144.0 / 334〜638 ms） | 不要 |

### 4.6 予算が「決まった」と判定する条件

次の 3 つが揃ったとき、`B_pre` を確定できる。

1. W2・W5・W6 の停止時間の分布（p50 / p95 / max）が取れている。
2. `stopLearning()` の join 時間が取れている（C の見積りに必要）。
3. `B_pre` の値を「worker 停止の上限」ではなく「待つ時間の上限」として文書化している
   （超過時は保持に落ちるので、上限は安全側に倒れる）。

現時点で 1 と 2 が未取得のため、`B_pre = 5000` は**暫定**である。

---

## 5. 不変条件（追加分）

| # | 不変条件 |
| --- | --- |
| I-U5-1 | 保持経路で `DeviceSettings::saveSettings` を呼ばない。 |
| I-U5-2 | 保持経路で `audioEngineProcessor.reset()` と UI の reset を呼ばない。 |
| I-U5-3 | 保持経路で `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` を実行しない（H-1〜H-4）。 |
| I-U5-4 | `shutdownAudioOnlyKeepingState()` は `audioOnlyShutdownDone_` で 1 回に固定する。 |
| I-U5-5 | 保持経路で `terminalStopState_` / `workersStuck_` を変更しない（`Closed` / `true` のまま）。 |
| I-U5-6 | 保持経路で `lifecycleState` を `Releasing` にしない（`releaseResources` は入口で return する）。 |
| I-B3 | shutdown の総時間を有界と主張しない。有界なのは A と B だけである。 |
| I-B4 | `B_pre` を「Worker 停止の保証値」として文書化しない。「待機の上限（超過時は保持）」として扱う。 |

---

## 6. 未解決の更新

| # | 項目 | 状態 |
| --- | --- | --- |
| U-1 | `B_pre` の根拠 | **未計測**（§4.5 の計画）。暫定 5000 は待機上限としてのみ扱う |
| U-2 | `stopLearning()` の所要 | **未計測**。位置は C に確定（§4.4） |
| U-3 | W3・W4・W7 の停止要求 API | **未承認のまま**（本稿でも扱わない） |
| U-5 | `shutdownAudioOnlyKeepingState()` | **本稿 §3 で設計**（副作用・停止/保持対象・再入・成立条件） |
| U-8 | W1 の join が戻らない場合 | 受容（既存挙動）。待ち時間ログのみ追加 |
| U-9 | W1 の bounded 化（reaper） | 別設計・未着手 |
| U-11 | `AudioEngine` のタイマ停止 API | 新規。保持中も動く（S-2）。必要になった場合のみ |
| U-12 | Debug のリーク検出器 | 新規。保持により `MainWindow` 1 件が報告される（S-1）。意図的として許容 |
| U-13 | `closeAudioDevice()` の無界性 | 新規。C/D 共通の無界項。既存挙動 |
| U-10 | commit point の網羅性 | レビュー条件（`applyComputedIR` と `executePendingCommit` の 2 choke point が最終防衛線） |

---

## 7. 証跡（本稿で確認したソース）

| 対象 | 位置 | 確認内容 |
| --- | --- | --- |
| `~MainWindow` の処理順 | `MainWindow.cpp:1169-1220` | step 0〜10 の実体（`saveSettings` 1201、`setProcessor(nullptr)` 1196、`stopTimer` 1198、`removeAudioCallback` 1205、`closeAudioDevice` 1209、`audioEngineProcessor.reset` 1211） |
| MainWindow のメンバ | `MainWindow.h:64-101` | `AudioEngine` は値メンバ（65）、`audioEngineProcessor`（66）、`audioDeviceManager`（67）、`audioProcessorPlayer`（68）、UI（70-73）、`cliLogFileLogger`（100） |
| リーク検出器 | `MainWindow.h:102` | `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR` |
| `AudioEngine` の基底 | `AudioEngine.h:601-606` | `juce::Timer` を private 継承（外部から `stopTimer()` 不可） |
| `~AudioEngine` の join | `AudioEngine.CtorDtor.cpp:173-177` | RetryScheduler → CoordinatorLoop → RebuildThread の停止・join |
| `AudioEngineProcessor` | `AudioEngineProcessor.cpp:55-66` | `releaseResources` は未 prepare なら何もせず（`:63`）、prepare 済みなら `audioEngine.releaseResources()`（`:66`） |
| terminal / reconfigure | `ReleaseResources.cpp:50-58, 146-150, 818-822` | terminal フラグ消費と learner 停止の 2 箇所 |
| drain 上限 | `ReleaseResources.cpp:281` | `kGracefulDrainMaxMs = 5000` |
| 無期限 join | `RebuildDispatch.cpp:825-826` | `join()` にタイムアウト無し |
| C の適用先 | `LoadPipeline.cpp:851` / `:834` | callback 経路と同期経路の両方が `executePendingCommit` を通る |
| E5 分類 | `LoadPipeline.cpp:284, 312, 362, 371, 460, 85, 93, 99, 129, 142, 892` | 9 箇所の位置 |
| SE の実測 | 計測報告 §5.1 | 変換 16〜20 ms、cache save 31〜61 ms、適用 54〜59 ms |

---

## 8. 停止条件の確認

- production source / test / CMake を本稿のために変更していない。
- build / test / debugger / replay を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e69957b046290f7077797782f031ed8633fd0`。
- AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。
- U-3 の実装設計には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認・U-3 着手許可はいずれも付与しない。
