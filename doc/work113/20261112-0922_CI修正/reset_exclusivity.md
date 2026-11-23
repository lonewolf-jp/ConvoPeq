# `RetireLatencyRecorder::reset()` の排他性

対象: `src/audioengine/ISRRetireLatency.h` の `reset()`（HEAD では 803〜892 行）
目的: `reset()` 内の `store` が release でも成立する根拠（＝呼び出し中に記録・参照が並行しないこと）を示す。

## 1. 呼び出し箇所（全 10 箇所）

`RetireLatencyRecorder::instance().reset()` と `rec.reset()` をツリー全体で検索した結果は次のとおりで、これ以外に呼び出しは無い。

| # | 位置 | 呼び出し元の関数 | ライフサイクル上の位置 |
| --- | --- | --- | --- |
| 1 | `src/audioengine/AudioEngine.CtorDtor.cpp:52` | `AudioEngine::AudioEngine()` | コンストラクタ本体の先頭文。`initialize()` より前 |
| 2 | `src/tests/D8_2_B_2_Tests.cpp:1002` | `test_T11_CompleteFadeRecords` | テスト本体の初期化（T11 の 1 回目） |
| 3 | `src/tests/D8_2_B_2_Tests.cpp:1076` | `test_T11_CompleteFadeRecords` | テスト本体の初期化（T11 の 2 回目） |
| 4 | `src/tests/D8_2_B_2_Tests.cpp:1140` | `test_T12_FinalizeRetiresTargetAndCurrent` | テスト本体の初期化 |
| 5-9 | `src/tests/D8_2_B_2_Tests.cpp:1233, 1275, 1317, 1345, 1380` | `test_T13_C2NonRtOnly` | 各サブケースの初期化 |
| 10 | `src/tests/DomainTagTests.cpp:126` | `testRecorderInvariants` | テスト本体の初期化 |

本番コードの呼び出しは 1 箇所だけ（#1）。ほかはテストの初期化である。

## 2. 本番（#1）が静止状態である根拠

レコーダへ書き込むスレッドは、`AudioEngine` の生成完了後、`initialize()` 以降にしか存在しない。

- ワーカースレッド: `AudioEngine::initWorkerThread()` → `m_workerThread.start()`。`initialize()` の末尾から呼ばれる（`AudioEngine.Init.cpp:189-192`）。
- メッセージスレッドのタイマー: `initialize()` 内の `startTimer(100)`（`AudioEngine.Init.cpp:132`）。
- コーディネータループ: `initialize()` 内の `startCoordinatorLoop()`（`AudioEngine.Init.cpp:133`）→ `CoordinatorLoop::startLoop()` → `startThread()`（`ISRCoordinatorLoop.cpp:21`）。
- オーディオコールバック: `MainWindow` の `audioDeviceManager.addAudioCallback(&audioProcessorPlayer)`（`MainWindow.cpp:289`）。`audioEngine.initialize()`（279 行）とプロセッサ設定（281-283 行）の後。
- `AudioEngine` のインスタンスは `MainWindow` のメンバ 1 個（`MainWindow.h:32`）。アプリ内で 2 個目を作らない。
- `WorkerThread` のコンストラクタは空で、`start()` を呼ぶまでスレッドを作らない（`WorkerThread.cpp:21-30`）。

したがって `AudioEngine::AudioEngine()` の先頭で `reset()` が走る時点では、このプロセスに「レコーダへ書くスレッド」が 1 本も存在しない。

## 3. テスト（#2〜#10）が静止状態である根拠

- `main()` はテスト関数を順に呼ぶだけで、テスト間に並行実行は無い（`D8_2_B_2_Tests.cpp:1485-1500`）。
- `D8_2_B_2_Tests` で生成するスレッドは `test_T13_C2NonRtOnly` の 1 本だけ（1396 行）。`rec.reset()` は 1380 行で、スレッド生成より前。スレッドは 1401 行で `join()` してから残りの検証が続く。
- 他の 7 箇所（1002, 1076, 1140, 1233, 1275, 1317, 1345）はスレッド生成を含まない区間にある。
- `DomainTagTests` の `testRecorderInvariants` はスレッドを作らない。

## 4. 将来の通常運用で維持されること

- `reset()` は通常運用の経路（タイマー、RT コールバック、パブリケーション、シャットダウン）から呼ばれない。呼び出しは上表の 10 箇所だけで、本番はコンストラクタ 1 箇所。
- 契約をコード側に明記した（`reset()` の直前にコメントを追加）。「静止状態からのみ呼ぶ」「本番は ctor のみ」「稼働中に呼ぶとオーダーの強弱以前に競合になる」を書いている。
- 実行時の検出機構は無い。新しい呼び出し箇所を足す場合は、この契約（他スレッドが記録・参照していないこと）をレビューで確認する必要がある。

## 5. この根拠が崩れる条件

次のいずれかが起きた場合、`reset()` 内の release は根拠を失う（seq_cst に戻すか、呼び出し側の設計を見直す必要がある）。

1. `AudioEngine` 以外の本番経路（タイマー、RT、ワーカー、コーディネータ）から `reset()` を呼ぶ。
2. 1 プロセスで `AudioEngine` を 2 個同時に生かし、片方の稼働中にもう片方を生成する。
3. テストで `reset()` の後にスレッドを残したまま次の `reset()` を呼ぶ。

現状はいずれも該当しない。
