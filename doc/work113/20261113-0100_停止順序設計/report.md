# shutdown 順序の再設計（READ-ONLY）

作業日時: 2026-11-13 01:00〜
基準 commit: `173e6995`（HEAD = origin/main・main ツリー）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 判定

```
所有権の方向（MainWindow 保持）  = 維持（Owner 判断を支持）
前稿の shutdown ゲート           = 未完成（順序が逆。§1 で確認）
今回の提示                       = 2 段階プロトコル（preflight → gate）で A〜D を閉じる設計
実装 GO                           = 未承認
```

Owner の指摘 4 点はすべてソースで裏づけられた。とくに **`releaseResources` はゲート通過後に走る**（`shutdown()` → `mainWindow.reset()` → `~MainWindow` → `setProcessor(nullptr)` → `releaseResources`）ため、前稿の「releaseResources でタイムアウトを記録し、ゲートで見る」は成立しない。順序を逆にする必要がある。

---

## 1. 事実確認（問題 A〜D の裏づけ）

### 1.1 実行順序（問題 A）

| 段階 | 処理 | 根拠 |
|---|---|---|
| 1 | `shutdown()` が `requestTerminalRelease()` を呼ぶ | `MainApplication.cpp:217-222` |
| 2 | `mainWindow.reset()` | `MainApplication.cpp:229` |
| 3 | `~MainWindow`: `setProcessor(nullptr)` | `MainWindow.cpp:1196` |
| 4 | `~MainWindow`: `removeAudioCallback` → `closeAudioDevice` | `MainWindow.cpp:1205, 1209` |
| 5 | メンバ破棄（`audioEngine` → `ConvolverProcessor`） | `MainWindow.h:65` |

JUCE の `AudioProcessorPlayer::setProcessor` は `oldOne->releaseResources()` と `processor->releaseResources()` を呼び（`juce_AudioProcessorPlayer.cpp:191, 204`）、`audioDeviceStopped` も `processor->releaseResources()` を呼ぶ（`:380`）。つまり **Worker 停止と内部状態破棄を担う `releaseResources` は、ゲート（2）より後に走る**。

### 1.2 releaseResources が壊すもの（問題 B）

`ConvolverProcessor::releaseResources()`（`Lifecycle.cpp:468-515`）は、`stopUpgradeThread()`（`LoadPipeline.cpp:101-109`）→ `forceCleanup()`（`StateAndUI.cpp:1035-1053`）→ `activeLoader.reset()`（`LoaderThread.cpp:42-48` の `stopThread(500)`）→ バッファ解放 → FFT handle 解放 → `currentIRState` の即時 free → `deferredFreeThread` 停止、を実行する。**MainWindow を保持しても、この内部状態破棄は止まらない。**

### 1.3 MainWindow を保持すると止まるもの（問題 C）

`~MainWindow` が行う `setProcessor(nullptr)` / `removeAudioCallback` / `closeAudioDevice` は、保持すると実行されない。**オーディオコールバックとデバイスの停止が保証されない。**

### 1.4 静的ホルダ（問題 D）

`static std::vector<std::unique_ptr<MainWindow>>` はプロセス終了時に破棄され、`~MainWindow` 以下の dtors が走る。保持の意味が消える。

---

## 2. 再設計: 2 段階プロトコル

### 2.1 形

```
MainApplication::shutdownApp()                 // shutdown() と ~MainApplication の共通ルーチン（冪等）
  if (mainWindow == nullptr) return;
  auto* engine = mainWindow->getAudioEngine();

  // ── Phase 0: preflight（ゲートの前）──
  engine->requestTerminalRelease();            // 既存
  const bool allStopped = engine->preflightStopWorkers(kBoundMs);   // 新規: 協調停止 + 有限待ち

  // ── Phase 1: 分岐 ──
  if (allStopped)
      mainWindow.reset();                      // 通常経路（従来と同じ順序で device close → releaseResources → 即時 free）
  else
  {
      engine->markStuckWorkers();              // sticky（atomic）
      mainWindow->shutdownAudioOnlyKeepingState();   // 新規: コールバック/デバイスだけ止める
      StuckProcessHolder::keep(mainWindow.release()); // ヒープ確保の静的ポインタ（破棄しない）
  }
```

### 2.2 Phase 0: preflight

| 要素 | 内容 |
|---|---|
| `ConvolverProcessor::requestWorkerStop()` | `activeLoader` / `loaderTrashBin` に `signalThreadShouldExit()`、`upgradeThread` に `cancel()` + `signalThreadShouldExit()`。**`stopThread` は呼ばない（殺さない）** |
| `ConvolverProcessor::waitForWorkersToExit(bound)` | 各 Worker の `waitForThreadToExit(bound)`。全員 true なら true |
| `AudioEngine::preflightStopWorkers(bound)` | `convolver` と `uiConvolverProcessor` の AND |
| 実行スレッド | message thread（`shutdown()` の呼出し元）。Worker は非 RT |

preflight は `closeAudioDevice()` より前に走るので、audio callback はまだ動いている。Worker は非 RT であり、`owner` / router は生存しているため、この時点の協調停止は安全である。

### 2.3 Phase 1a: 全停止（通常経路）

`mainWindow.reset()` をそのまま実行する。`~MainWindow` の `setProcessor(nullptr)` → `closeAudioDevice` が `releaseResources` を呼ぶが、**Worker は既に停止済み**なので、既存の破棄処理（`stopUpgradeThread` / `forceCleanup` / `activeLoader.reset` / IRState 即時 free）はそのまま安全に走る。`stopThread` は「停止済みスレッド」に対して即座に true を返すため、強制終了経路にも入らない。

### 2.4 Phase 1b: タイムアウト（保持経路）

1. `engine->markStuckWorkers()` で sticky フラグを立てる（`TimedOut` も記録）。
2. `mainWindow->shutdownAudioOnlyKeepingState()`:
   - `audioProcessorPlayer.setProcessor(nullptr)`（→ `releaseResources` が呼ばれるが、stuck フラグにより**非破壊の早期 return**）
   - `audioDeviceManager.removeAudioCallback(&audioProcessorPlayer)`
   - `audioDeviceManager.closeAudioDevice()`
   - `stopTimer()`
   - UI・`audioEngine`・router/domain は触らない
3. `StuckProcessHolder::keep(mainWindow.release())` で保持。

これで「コールバックとデバイスは止まる（問題 C）」「Worker の依存状態は残る（問題 B）」が両立する。

### 2.5 `releaseResources` の非破壊モード

```
void ConvolverProcessor::releaseResources()
{
    if (convo::consumeAtomic(workersStuck_, std::memory_order_acquire))
    {
        audioStopped_ = true;   // 状態だけ記録し、Worker / IRState / バッファを破棄しない
        return;
    }
    ... 既存の破棄 ...
}
```

`workersStuck_` は preflight の失敗で立ち、**Worker が終了した時点でクリアする**（Worker 側の終了処理で `fetch_sub` するか、`hasUnstoppedWorker()` が false になったら次の機会にクリアする）。クリア後は通常の `releaseResources` が使える。

### 2.6 holder（問題 D）

```
struct StuckProcessHolder {
    static std::vector<MainWindow*>& windows() {
        static auto* v = new std::vector<MainWindow*>();   // ヒープ。静的破棄の対象外
        return *v;
    }
    static void keep(MainWindow* w) { if (w != nullptr) windows().push_back(w); }
};
```

`unique_ptr` を `release()` して生ポインタで保持する。ベクタ自体は `new` されたままなので、プロセス終了時に破棄されない（ポインタの静的変数は trivially destructible）。**`~MainWindow` は走らない。**

### 2.7 異常終了経路

`MainApplication` にデストラクタを追加し、`shutdownApp()` を呼ぶ（`shutdown()` 後は `mainWindow == nullptr` なので冪等）。これで「`shutdown()` を経由しないが C++ デストラクタが走る」経路にも同じ preflight とゲートが適用される。OS による強制終了・クラッシュは対象外（通常の dtor 後始末の外）。

---

## 3. 順序図

```
通常:
  shutdown()
    requestTerminalRelease()
    preflightStopWorkers(bound) ── 全停止 ──▶ true
    mainWindow.reset()
      ~MainWindow: setProcessor(nullptr) → removeAudioCallback → closeAudioDevice
        → AudioEngineProcessor::releaseResources → AudioEngine::releaseResources
          → uiConvolverProcessor.releaseResources（stuck=false: 破棄してよい。Worker は停止済み）
      ~AudioEngine（dtor body: coordinator/rebuild join → drain）
      メンバ破棄（coordinator → router → epochDomain → … → uiConvolverProcessor → convolver）

タイムアウト:
  shutdown()
    requestTerminalRelease()
    preflightStopWorkers(bound) ── 一部未停止 ──▶ false
    markStuckWorkers()（sticky, TimedOut 記録）
    shutdownAudioOnlyKeepingState()
      setProcessor(nullptr) → releaseResources（stuck=true: 早期 return、何も破棄しない）
      removeAudioCallback → closeAudioDevice → stopTimer
    StuckProcessHolder::keep(mainWindow.release())
    （プロセス終了まで MainWindow / AudioEngine / router / IRState / Worker は生存）
```

---

## 4. 問題への対応

| 問題 | 対応 | 節 |
|---|---|---|
| A: ゲートと停止待ちの順序 | preflight を `mainWindow.reset()` の前に置く | §2.2 |
| B: releaseResources が Worker の依存状態を壊す | `workersStuck_` による非破壊モード | §2.5 |
| C: デバイス／コールバックが止まらない | `shutdownAudioOnlyKeepingState()` で明示的に停止 | §2.4 |
| D: 静的ホルダの破棄 | ヒープ確保の静的ポインタ（破棄しない） | §2.6 |
| 異常終了 | `~MainApplication` から同じ共通ルーチン | §2.7 |
| デストラクタ条件 | 「stuck=false のときだけ dtor に入る」を不変条件にする | §5 |

---

## 5. 不変条件

| # | 不変条件 |
|---|---|
| I1 | `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` に入る時点で、未停止 Worker が存在しない（通常経路）。 |
| I2 | `workersStuck_` が立っている間、`releaseResources` は Worker・IRState・バッファ・FFT・deferredFreeThread を破棄しない。 |
| I3 | デバイスとコールバックは、保持経路でも必ず停止する（`shutdownAudioOnlyKeepingState`）。 |
| I4 | 保持した `MainWindow` は二度と破棄しない（holder は静的破棄されない）。 |
| I5 | `workersStuck_` が立っている間は、新しい IR ロードを開始しない（`prepareToPlay` / `loadImpulseResponse` を拒否）。 |
| I6 | `~ConvolverProcessor` から retire provider を呼ばない（router/domain は破棄済み）。 |
| I7 | デストラクタ内の `return` は破棄を中止しないため、Worker 動作中に dtor へ入らないことを構造で保証する（I1）。 |
| I8 | Worker の終了確認（I1）と IRState の reclaim 可否（epoch-gated drain）は別々に確認し、両方成立したときだけ即時 free する。 |

---

## 6. 残る設計項目

| # | 項目 | 内容 | 状態 |
|---|---|---|---|
| S-1 | `bound` の決定 | 非割込み区間の計測結果から決める（計測が先行ゲート） | 未（計測待ち） |
| S-2 | stuck のクリア | Worker 終了時にクリアする経路（atomic の fetch_sub / フラグ再評価） | 要設計 |
| S-3 | `prepareToPlay` の拒否 | I5 の実装（stuck 中の再開禁止） | 要設計 |
| S-4 | 保持経路のテレメトリ | 保持件数、`TimedOut` 理由、生存 Worker の ID | 要設計 |
| S-5 | `shutdownAudioOnlyKeepingState` の副作用 | `DeviceSettings::saveSettings` を呼ばない、ChangeListener 解除の順序 | 要設計 |
| S-6 | IRState の guard 9 箇所 | 前稿の範囲 | 未 |
| S-7 | (b) パック + ペア getter | 前稿 §5 | 未 |
| S-8 | snapshot 2 経路 | 独立実装単位 | 承認済み・未実装 |

---

## 7. 証跡

`doc/work113/20261113-0100_停止順序設計/evidence/`

| ファイル | 内容 |
|---|---|
| `shutdown_order_excerpts.md` | shutdown の実行順序と破棄内容の逐語抜粋（7 ファイル / 299 行 / 10,215 文字、SHA-256 付き、HEAD blob 基準。JUCE の `AudioProcessorPlayer` を含む） |

---

## 8. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** 前稿のゲートを 2 段階プロトコルに置き換える設計を提示したが、実装 GO・Gate 1 承認は付与しない。
