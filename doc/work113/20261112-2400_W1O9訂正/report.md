# W1-O9 の訂正: JUCE の停止意味論と releaseResources の契約

作業日時: 2026-11-12 24:00〜
基準 commit: `173e6995`（HEAD = origin/main）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 訂正と結果

```
W1-O9a の記述 = 訂正（タイムアウトでスレッドは「生き残る」のではなく「強制終了される」）
W1-O9a の方針 = 協調停止 + waitForThreadToExit（強制終了を通常経路で使わない）
W1-O9b         = JUCE 保証は「再生停止後に呼ばれる」まで。message thread 契約は現行コードにも無い
(b)            = 単一 atomic<uint64_t> パックを採用（条件を §4 に明記）
#7             = guard 対象（維持）
W1-O8          = CLOSED を維持
実装 GO         = 未承認
```

前稿の「タイムアウト時は upgrade / loader が生き残り、IRState を読み続け得る」は**誤り**だった。同梱 JUCE の `stopThread` は、タイムアウト時に `killThread()` を呼び、`false` を返す。危険の性質は「読み手が残る（UAF）」ではなく「**強制終了がロック・イベント・RAII を不整合に残す**」に変わる。したがって、IRState の解放を延期するだけでは足りない。

---

## 1. 同梱 JUCE の実装確認（Owner 依頼）

`JUCE/modules/juce_core/threads/juce_Thread.cpp`

| 関数 | 実装 | 行 |
|---|---|---|
| `stopThread(timeout)` | `signalThreadShouldExit()` → `notify()` → `waitForThreadToExit(timeout)` → **まだ running なら** `jassertfalse` + `Logger::writeToLog("!! killing thread by force !!")` + `killThread()` + `threadHandle = nullptr` + `threadId = {}` + **`return false`** | `245-277` |
| `waitForThreadToExit(timeout)` | `isThreadRunning()` を 2 ms 間隔でポーリング。タイムアウトで `false` を返す。**強制終了しない** | `227-243` |
| `~Thread()` | `jassert(!isThreadRunning())` → `stopThread(-1)`（無期限待ち） | `43-58` |
| `killThread()`（Windows） | `TerminateThread(threadHandle, 0)` | `juce_Threads_windows.cpp:135-144` |

`stopThread` の強制終了直前のコメント（`263-266`）:

> very bad karma if this point is reached, as there are bound to be
> locks and events left in silly states when a thread is killed by force

つまり **`false` は「まだ動いている」ではなく「殺して止めた」**を意味する。Owner の訂正のとおりである。

この事実が W1-O9a に与える影響は 2 つ。

1. **UAF の議論は解消する。** 殺されたスレッドは IRState を読まない。
2. **別の危険が残る。** `TerminateThread` はスレッドのスタック上の RAII（`juce::ScopedLock`、`ScopedNoDenormals`、`std::lock_guard` など）を解放しない。ロックが保持されたまま残れば、以後のロック取得は永久に待つ。ヒープの中間状態も残る。これは IRState の寿命問題ではなく、**プロセス状態の整合性問題**である。

---

## 2. W1-O9a の設計方向

Owner 推奨の方針を採用する。すなわち、**強制終了を通常の停止経路で使わない**。

```
1. signalThreadShouldExit()（＋ Convolver 側は cancel() で協調フラグも立てる）
2. Worker が区切りで終了要求を確認して戻る
3. waitForThreadToExit(timeout)（殺さない）
4. true（終了確認）の後に、Worker と IRState を破棄する
```

### 2.1 協調停止が成立するかの確認（本稿で実測）

| Worker | 終了要求の確認点 | 判定 |
|---|---|---|
| `LoaderThread` | `run()` のループに `checkCancellation(shouldStop, …)` が複数（`LoaderThread.cpp:578, 700, 717, 734, 842`）。`shouldStop` は `threadShouldExit()` を見る（`:574-578, 838-842`）。`:128` にも早期 return あり | **協調停止できる** |
| `ProgressiveUpgradeThread` | `threadShouldExit()` は見ないが、`cancel()` が `cancelled` を立て、ループ（`:91, :135`）と `convertToHighRes` のラムダ（`:120-135`）がそれを見る。`stopUpgradeThread` は `cancel()` → `stopThread(2000)` の順（`LoadPipeline.cpp:105-106`） | **協調停止できる**（`cancelled` 経由） |

したがって `stopThread(500/2000)` を `signalThreadShouldExit` + `waitForThreadToExit` に置き換えても、停止は成立する見込みがある。

### 2.2 タイムアウト時に何を守るか

Owner の指摘どおり、**IRState だけを延期しても不十分**である。Worker は `ConvolverProcessor&` を保持し（`loader` は `owner` 参照、`upgrade` は `owner` の weak 参照）、`~LoaderThread` は `owner.retireStereoConvolver(...)` も呼ぶ（`LoaderThread.cpp:46-47`）。したがって、タイムアウト時に維持すべきは「Worker → owner → IRState」の寿命依存関係全体である。

設計上の選択肢は 2 つ。

| 案 | 内容 | 評価 |
|---|---|---|
| (i) 無期限 join | `waitForThreadToExit(-1)` で終了まで待つ | ホストのコールバック上で待つとホストが止まる。`releaseResources` では避ける（Owner の指摘どおり） |
| (ii) 有限待ち + 失敗時は停止を保留 | `waitForThreadToExit(timeout)` が false なら、Worker と owner と IRState を**破棄しない**（保留リストへ移し、後続の停止機会で join） | 安全側。ただし「保留中は新しいロードを受け付けない」等の状態設計が要る |

推奨は (ii)。**強制終了は異常系の最終手段**とし、通常経路では使わない。

### 2.3 無期限 join を採る場合の前提

Owner の条件をそのまま義務として残す。

* Worker がキャンセル要求に応答できること（§2.1 で確認済み。ただし FFT 1 回などの**非割込み区間の長さは未計測**）。
* ホストのコールバック上で無期限に待たないこと（`releaseResources` はホストスレッドの可能性があるため）。

---

## 3. W1-O9b: releaseResources の契約

### 3.1 JUCE が保証していること／していないこと

* 同梱 JUCE の宣言コメントは次のとおり（`JUCE/modules/juce_audio_processors_headless/processors/juce_AudioProcessor.h:141-143`）。
  > Called after playback has stopped, to let the object free up any resources it no longer needs.
* **どのスレッドから呼ばれるかは書かれていない。** つまり JUCE の保証は「再生停止後」までで、スレッドは保証されない（Owner の指摘どおり）。

### 3.2 現行コードが前提にしていること

| 箇所 | 実際の制約 | 内容 |
|---|---|---|
| `AudioEngine::releaseResources()` | `ASSERT_NON_RT_THREAD()`（`ReleaseResources.cpp:36`） | **audio thread 以外**のみ。message thread を要求していない |
| `AudioEngineProcessor::releaseResources()` | 無し（JUCE override、`:55-67`） | ホストが呼ぶ。スレッドはホスト依存 |
| `uiConvolverProcessor.releaseResources()` | `:558`（terminal pass のみ） | reconfigure pass では呼ばれない（IRState の即時 free は terminal のみ） |

つまり、**「message thread 契約」は JUCE にも現行コードにも存在しない**。前稿で「message thread 契約の invariant 化」と書いたのは誤りで、Owner の指摘どおり「JUCE の保証」と「ConvoPeq 独自の契約」を分ける必要がある。

### 3.3 何が危ないか（具体）

`ConvolverProcessor::releaseResources` は、停止（upgrade / loader）の後に `currentIRState` を交換して**即時 free** する（`Lifecycle.cpp:506-511`）。このとき、message thread が `loadImpulseResponse` や `applyComputedIR` の中で IRState を読んでいると、free が読取りと競合する。

`releaseResources` が message thread で呼ばれるなら、message の読み手は呼出し元自身なので競合しない。しかし**現行の制約は「non-RT」だけ**であり、ホストが非 RT の別スレッド（ワーカー等）から呼ぶと、message の読み手と競合し得る。

### 3.4 選択肢

| 案 | 内容 | 評価 |
|---|---|---|
| (i) 全呼出し元を message thread に marshal | ホストから呼ばれた時点で message へ投げ、完了を待つ | ホストスレッドを止める危険。JUCE の保証（再生停止後のみ）を超える要求になる |
| (ii) releaseResources をスレッド非依存にする | IRState の free を即時に行わず、engine ドメインの deferred delete に移送する。読み手は §4 の guard で保護済みにする | **推奨**。既存の epoch 機構と整合し、ホストスレッドを止めない |
| (iii) 現状維持 | 「non-RT で、message の読み手と重ならない」ことを運用で保証 | 証明できない。不可 |

推奨は (ii)。IRState の guard 追加（前稿の 9 箇所）と組み合わせると、free が epoch-gated になり、message の読み手も守れる。ただし `releaseResources` 自体が「読み手を止める」責務を持つ場面（エンジン終了）では、別途 quiescence が必要である。

---

## 4. (b) 単一 atomic<uint64_t> パックの実装条件

採用を前提に、実装時に明示すべき条件を列挙する。

| # | 条件 | 内容 |
|---|---|---|
| 1 | サイズ前提 | `static_assert(sizeof(float) == 4)`。`uint64_t` に 2 個詰める |
| 2 | ビット変換 | `std::bit_cast<std::uint32_t>(float)` / `std::bit_cast<float>(std::uint32_t)`（C++20）。数値変換ではなくビット保存 |
| 3 | 配置 | 例: 上位 32 bit = `additionalAttenuationDb`、下位 32 bit = `irFreqPeakGainDb`。順序をコメントで固定 |
| 4 | 初期値 | `0.0f` 同士を 0 として表現。未設定を区別する必要があるなら別途 sentinel（現行 `IRState` の既定も `0.0f`） |
| 5 | 更新 | `updateIRState` の中で 1 回の `store(release)`。2 つの独立 store を残さない |
| 6 | 読み | 1 回の `load(acquire)` で 2 値を復元。`getIrAdditionalAttenuationDb` / `getIrFreqPeakGainDb` はこの語から返す |
| 7 | 非有限値 | `IRState` のメタデータに NaN/Inf が入らないことを確認（入るならビット比較の意味を定義） |
| 8 | 戻り値型 | 既存 API の `float` を維持（呼出し側の変更を最小化） |

---

## 5. 残る義務（更新）

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| W1-O9a | 協調停止への切替 | `stopThread` → `signalThreadShouldExit` + `waitForThreadToExit`。タイムアウト時は Worker + owner + IRState を保持（案 ii） | 未（設計） |
| W1-O9a-2 | 非割込み区間の確認 | LoaderThread の FFT／リサンプル 1 回の所要時間（キャンセル粒度） | 未計測 |
| W1-O9b | releaseResources のスレッド非依存化 | IRState の free を deferred delete へ移送（案 ii）＋ 読み手 guard | 未（設計） |
| W1-O9b-2 | 「message thread」表現の撤回 | JUCE 保証ではない。文書からも独自 invariant としても書かない | **本稿で撤回** |
| W1-O8a | IRState の所有バッファ invariant | `ir` は常に `irOwner` | 明文化待ち |
| (b)-1 | 単一語パック | §4 の条件で実装 | 未 |
| #7 | guard 対象 | 前稿の 9 箇所に含める | 未 |
| W1-O6 | snapshot 2 経路 | 独立実装単位 | 承認済み・未実装 |
| A-O1〜A-O5 / N6 / N7 / N9 | 前回のまま | — | 未 |

---

## 6. 証跡

`doc/work113/20261112-2400_W1O9訂正/evidence/`

| ファイル | 内容 |
|---|---|
| `w1o9_juce_excerpts.md` | 同梱 JUCE の Thread 実装（`stopThread` / `waitForThreadToExit` / `~Thread` / win32 `killThread`）と、ConvoPeq 側の呼出し元（9 ファイル / 474 行、SHA-256 付き、HEAD blob 基準） |

`find-nise-kanji` が証跡の 1 行（`AudioEngine.Processing.ReleaseResources.cpp:39`）を検出する。これは HEAD のソースに元からある表記（正字でない漢字 1 字）で、逐語抜粋のため原文のまま残した（READ-ONLY のため修正していない）。本報告書自身は検出 0 件である。

---

## 7. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** W1-O9a は判断材料を提示し、W1-O9b は「JUCE 保証ではない」ことを撤回・整理した。実装 GO・Gate 1 承認は付与しない。
