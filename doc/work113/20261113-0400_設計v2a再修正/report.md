# shutdown 設計 v2-a 再修正（READ-ONLY）

- 作業日時: 2026-11-13 04:00 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（HEAD = origin/main、作業ツリー）
- 性質: **設計のみ**。production source / test / CMake / git を変更していない。build / test / debugger も実行していない。
- 置き換え範囲: 前稿 v2-a（`doc/work113/20261113-0300_設計v2修正/report.md`）の
  **§1.4（A の producer 契約）・§2.5（W1 の保証）・§3（C の判定）・§4（E-5 の commit point）**。
  A の 4 状態 CAS 状態機械そのもの、B、D は変更しない。
- 行番号は**現行ツリー**（計測用 probe の挿入を含む）のもの。

---

## 0. 判定

```
A の 4 状態 CAS       = 維持（Owner 受容済み）。producer 起動との契約だけ追加
C の判定              = epoch 単独 → epoch AND isStopAcceptingWork() に修正
E-5                   = commit point を 9 箇所列挙し、choke point を確定
W1 の保証             = LoopExited は「join 完了」を意味しない。bounded の主張を撤回
実装 GO / Gate 1 / U-3 = いずれも付与しない
```

---

## 1. A — producer 起動との競合契約

### 1.1 競合の実体（コードで確認）

`isStopAcceptingWork()` を入口で 1 回検査するだけでは、検査と Worker 起動の間に Intent が立つ可能性が残る。
現行コードの順序は次のとおり。

| 関数 | 入口検査を置く位置 | Worker の生成 | `startThread()` |
| --- | --- | --- | --- |
| `loadImpulseResponse()` | 関数入口（`:22` 付近。`isLoading` publish の `:47` より前） | `activeLoader = make_unique<LoaderThread>`（`:85` rebuild / `:93` file） | `:99` |
| `startProgressiveUpgrade()` | 関数入口（`:120` の `isProgressiveUpgradeEnabled` より前） | `upgradeThread = make_unique<ProgressiveUpgradeThread>`（`:129`） | `:142` |

どちらも **Message Thread 上で実行される**。`loadImpulseResponse()` には `JUCE_ASSERT_MESSAGE_THREAD` が無いが、
呼出し元は `Lifecycle.cpp:220`（`timerCallback`）と `Rebuild.cpp:17`（`rebuildAllIRs()`、その呼出し元は
`LoadPipeline.cpp:955` の `callAsync` とモード変更 = いずれも MT）に限られる。

Intent は任意スレッドから publish できるので、**検査（T1）と `startThread()`（T3）の間**に Intent が入り得る。

### 1.2 3 案の比較

| 案 | 内容 | 単独で閉じるか | 変更量 | 判定 |
| --- | --- | --- | --- | --- |
| 1 | Intent 前に開始した処理として許容し、`requestWorkerStop()` の停止対象に必ず含める | **閉じる**（§1.3） | 小（契約の明文化 + 既存の生成順の維持） | **採用** |
| 2 | `startThread()` 直前に再検査し、Intent 後なら起動しない | 単独では閉じない（再検査と `startThread` の間に同じ窓がある） | 小 | **併用（最適化）** |
| 3 | Intent と Worker 起動を admission の CAS で直列化する | 閉じるが、Worker 起動は現在 admission の予約対象外であり、新しい線形化点を導入することになる | 大 | 不採用（v2-a の範囲外） |

**採用は 1（正しさの根拠）+ 2（無駄な起動の抑止）**。2 を入れても窓は 0 にならないため、
**正しさを 2 に依存させない**。

### 1.3 案 1 が閉じる理由（MT 直列性）

`beginTerminalStopAcceptingWork()`（S1）と `requestWorkerStop()`（T4）は MT 上で走る。
`loadImpulseResponse` / `startProgressiveUpgrade` の入口検査（T1）と `startThread()`（T3）も MT 上で走る。
したがって **T1 < T3 < T4** が同一スレッド系列で保証される（T4 は S1 の後に必ず 1 回呼ばれる）。

| Intent の位置 | 結果 |
| --- | --- |
| Intent < T1 | 入口検査で拒否。Worker は生成も起動もされない |
| T1 < Intent < T3 | Worker は起動する。しかし T3 < T4 なので、`requestWorkerStop()` が `activeLoader` / `upgradeThread` を見つけて停止要求を出す |
| T3 < Intent < T4 | 同上（起動済みの Worker が停止対象になる） |
| Intent > T4 | **起こらない**。T4 は S1 の後、S1 以降は入口検査が拒否するため、T1 を通る新しいロードは存在しない |

この証明は **I-A8（`loadImpulseResponse` が MT 専用であること）** に依存する。MT 以外から呼ばれる経路が
将来増えると T1 < T3 < T4 が崩れるため、`JUCE_ASSERT_MESSAGE_THREAD` を追加して前提を固定する。

必要条件は 2 つだけである。

1. Worker を **stop target に登録してから** `startThread()` する（現行コードはこれを満たす:
   `activeLoader` 代入 `:85/:93` → `startThread` `:99`、`upgradeThread` 代入 `:129` → `startThread` `:142`）。
2. `requestWorkerStop()` が `activeLoader` / `loaderTrashBin` / `upgradeThread` の**全部**を列挙する。

### 1.4 案 2 を併用する理由（正しさとは独立）

案 2 の再検査は、Intent 後に生成された Worker を「起動してすぐ停止要求」する無駄を避ける。
計測では IR 変換が 16〜20 ms、NUC 構築が 14 ms 程度なので、無駄の絶対量は小さい。
それでも入れるのは、**停止要求の到着が遅い場合に Worker が余分な作業を進めるのを防ぐ**ためである。

### 1.5 追加する不変条件

| # | 不変条件 |
| --- | --- |
| I-A5 | Worker は stop target へ登録してから `startThread()` する（登録 → 起動の順を崩さない）。 |
| I-A6 | `requestWorkerStop()` は MT で、S1 の後に 1 回以上呼ばれる。 |
| I-A7 | S1 以降は新しい Worker を起動しない（入口拒否 + 起動直前の再検査）。 |
| I-A8 | `loadImpulseResponse()` に `JUCE_ASSERT_MESSAGE_THREAD` を追加し、MT 前提をコードで固定する（P4 に含める）。 |

---

## 2. C — 判定を epoch AND StopAccepting にする

### 2.1 前稿の穴（Owner 指摘の確認）

前稿 §3.2 の判定は epoch 単独だった。epoch は `Closing` の処理中にしか進まないため、次の実行が成立する。

| 順序 | 実行 | 状態 |
| --- | --- | --- |
| 1 | C6 の callback が MT に登録される（epoch = E） | `Idle` |
| 2 | 非 MT が `requestTerminalShutdownIntent()` | `Intent` |
| 3 | MT はまだ `beginTerminalStopAcceptingWork()` を呼んでいない | `Intent` |
| 4 | C6 の callback が実行される | `Intent` |
| 5 | epoch は E のまま（進んでいない）ので一致判定を通る | `Intent` |
| 6 | `executePendingCommit()` が IR metadata 更新・Engine swap・`irFinalized` publish を行う | `Intent` |

`executePendingCommit()`（`LoadPipeline.cpp:944-948` で `irFinalized` を publish し、engine を差し替える）は
軽微な通知ではない。**Intent 後・epoch 更新前の callback を抑止できていない。** Owner の指摘は正しい。

### 2.2 修正後の判定

```
// callback 本体（MT）。RAII ラッパーを生成した後、状態変更の前に判定する。
const uint64_t nowEpoch = convo::consumeAtomic(shutdownEpoch_, std::memory_order_acquire);
if (nowEpoch != capturedEpoch || isStopAcceptingWork())
{
    // ここで return。所有権は生成済みの RAII が解放する。
    return;
}
```

- 判定は **OR**（`||`）。どちらか一方でも真なら本体を実行しない。
- 判定順は **state → epoch**。`terminalStopState_` は単調で、`isStopAcceptingWork()` は `Intent` で真になる。
  epoch の読み取りは 1 回で足りる。
- `isStopAcceptingWork()` は `terminalStopState_ >= Intent || shutdownPhase >= StopAcceptingWork`（前稿 §1.4 のまま）。

### 2.3 各判定の役割（過大評価しない）

| 窓 | state で閉じる | epoch で閉じる |
| --- | --- | --- |
| Intent 〜 Closing（Owner 指摘の窓） | **○** | × |
| Closing 以降に enqueue された callback | ○ | ○ |
| S3 保持中に enqueue された callback | ○ | ○ |
| 判定直後に Intent が立つ（callback 実行中） | ×（割り込めないので不要） | ×（同上） |

**安全性を担うのは state 側**である。epoch は「enqueue 時点の世代」を保持するので、
(a) 抑止理由を「シャットダウン開始前に積まれた」と「開始後に積まれた」で区別でき、
(b) 将来 state をリセットする設計変更が入った場合の防御になる。両方を持つ理由はここにある。

### 2.4 対象 callback ごとの判定（前稿 §3.3 の更新）

抑止対象（C2〜C7）はすべて §2.2 の判定を先頭に置く。所有権の扱いは前稿 §3.4 から変更しない。

| # | 箇所 | 判定 | 備考 |
| --- | --- | --- | --- |
| C1 | `LoaderThread.cpp:87` | 判定しない（許容） | `isLoading` / `isRebuilding` の解除。止めると固着 |
| C2 | `LoaderThread.cpp:120` | 抑止 | 文字列のみ。所有権なし |
| C3 | `LoaderThread.cpp:288` | 抑止 | RAII 4 本を生成した後に判定 |
| C4 | `LoaderThread.cpp:331` | 抑止 | 文字列のみ |
| C5 | `ProgressiveUpgradeThread.cpp:218` | 抑止 | `unique_ptr` を `:220` で生成後に判定 |
| C6 | `LoadPipeline.cpp:843` | 抑止 | `unique_ptr` を `:846` で生成後に判定。加えて E5-9（`executePendingCommit` 入口）が同期経路も守る |
| C7 | `LoadPipeline.cpp:952` | 抑止 | 所有権なし。§3 の入口拒否と二重防御 |
| C8 | `Rebuild.cpp:36` | 判定しない（許容） | `sendChangeMessage` のみ。状態を変えない |
| C9 | `StateAndUI.cpp:817` | 判定しない（許容） | `latencyChangePending` の解除。止めると固着 |

---

## 3. E-5 — `loadIR()` の commit point を個別に列挙する

### 3.1 commit point の定義

「外部から観測可能な状態を変更する直前」を commit point とする。対象状態は
IR metadata（`irLength` / `currentSampleRate` / `irFinalized`）、engine（`exchangeActiveEngine`）、
publication（`enqueuePublicationIntent…`）、`isLoading` / `isRebuilding`、cache 書き込み、Worker 起動である。

### 3.2 列挙（現行コードの行番号）

| # | 位置 | 何をするか | gate |
| --- | --- | --- | --- |
| E5-1 | `LoadPipeline.cpp:284` | cache direct の `applyComputedIR()` | 呼出し前 + `applyComputedIR` 入口 |
| E5-2 | `LoadPipeline.cpp:312` | cached low の `applyComputedIR()` | 同上 |
| E5-3 | `LoadPipeline.cpp:362` | 変換結果の `applyComputedIR()` | 同上 |
| E5-4 | `LoadPipeline.cpp:371` | `startProgressiveUpgrade()` | 呼出し前 + 関数入口（`:120`） |
| E5-5 | `LoadPipeline.cpp:85/93` → `:99` | `activeLoader` 生成 → `startThread()` | 関数入口 + `:99` の直前 |
| E5-6 | `LoadPipeline.cpp:129` → `:142` | `upgradeThread` 生成 → `startThread()` | 関数入口 + `:142` の直前 |
| E5-7 | `LoadPipeline.cpp:460` | `applyComputedIR()` 本体（scale / validate / engine 差し替え） | 入口で 1 回（`:470` の generation 検査の直前） |
| E5-8 | `LoaderThread.cpp:271-342` / `:229-268` | finalize（`callAsync`）と同期 NUC 構築 | C3（epoch + state）と worker 停止 |
| E5-9 | `LoadPipeline.cpp:892` の `executePendingCommit()` 入口（`:894` の直後） | IR metadata 更新（`:913`）・engine swap・`irFinalized` publish | **入口で 1 回（choke point）**。`:834` の同期経路と `:851` の callback 経路の両方を守る |

choke point は 2 つ置く。**E5-7（`applyComputedIR` 入口）** と **E5-9（`executePendingCommit` 入口）** である。
理由は次の 3 点。

- E5-1〜E5-3（`applyComputedIR` の 3 経路）を 1 箇所で守れる。
- E5-9 は C6 の callback だけでなく **`:834` の同期経路（`applyNewState` 経由）** も守る。
  同期経路は `rebuildAllIRsSynchronous()` から呼ばれ、commit の実行主体が callback ではないため、
  C6 の判定だけでは守れない。
- どちらも payload / engine が `unique_ptr` なので、拒否時の解放が自動である
  （`applyComputedIR` は引数、`executePendingCommit` は `PendingCommit` のデストラクタが engine を retire する）。

抑止時に `isLoading` / `isRebuilding` を false に戻すのは、既存の stale generation 経路（`:904-905`）と同じ扱いである
（軽微な後始末として許容。C1 と同じ原則）。一方、`applyComputedIR` の入口で拒否した場合は
`isLoading` を触らない（設定は呼出し元の `loadImpulseResponse` `:47`、解除は C1 の `FlagResetter` が担う）。

### 3.3 4 つの規則

| # | 状況 | 規則 |
| --- | --- | --- |
| R-A | commit 開始**前**に Intent を観測 | commit しない。何も変更せず、payload は破棄する |
| R-B | commit 開始**後**に Intent が立つ | 開始済みの適用は**完了させる**（中断しない） |
| R-C | Worker 起動**前**に Intent を観測 | 起動しない（生成済みの Worker は破棄） |
| R-D | Worker 起動**後**に Intent が立つ | `requestWorkerStop()` の停止対象として必ず回収する（§1.3） |

R-B で中断しないのは、`applyComputedIR()` が IRState と engine を差し替える途中で止まると、
半端な状態（metadata だけ新しく engine が旧）を作るためである。MT 直列性により、開始済みの適用は
そのまま完了できる（他要因に中断されない）。所要は実測 54〜59 ms（計測報告 §5.1）。

### 3.4 どこまで完了させるか

| 処理 | Intent 後の扱い | 根拠 |
| --- | --- | --- |
| `irConverter->convertFile()`（同期変換） | 完走させる（16〜20 ms） | 途中中断の仕組みが無い。完走後に結果を捨てる方が単純で安全 |
| `cacheManager->save()`（31〜61 ms） | 完走させる | 内容アドレスのキャッシュ。中断は未対応で、途中破棄は部分書き込みを残し得る |
| `applyComputedIR()` | R-A / R-B に従う | 上記 |
| `startProgressiveUpgrade()` | 起動しない（R-C） | 停止要求と競合するだけ |
| `loadImpulseResponse()` の Worker | R-C / R-D | §1 |

つまり「**計算は完走、適用は拒否**」を原則とする。計算を打ち切る最適化は v2-a の範囲外である。

### 3.5 `loadIR()` の終端チェックだけでは足りない理由

`loadIR()` には `applyComputedIR()` が 3 箇所（`:284` / `:312` / `:362`）あり、いずれも関数終端より前にある。
関数終端だけで検査すると、その前の適用を止められない。したがって検査は
**各 commit point（E5-1〜E5-4）と choke point（E5-7）**に置く。関数終端のチェックは
「以降の upgrade 起動を止める」ためだけに残す（E5-4 の二重化）。

---

## 4. W1 — `LoopExited` と join 完了の区別

### 4.1 前稿の記述の撤回

前稿 §2.5 の「`Exited` を観測できた場合、その後の `join()` は直ちに戻る」は**誤り**である。撤回する。
`rebuildThreadLoopExited = true` の publish 後、スレッドが OS にプリエンプトされれば、
`join()` は再スケジュールまで戻らない。制御フローの短さは時間的な保証ではない。

### 4.2 保証できること／できないこと

| 命題 | 成立 | 理由 |
| --- | --- | --- |
| `LoopExited == true` → そのスレッドは以後 engine の状態に触れない | **○** | フラグは関数の最終行で publish され、以降は return のみ。engine のメンバには触れない |
| `LoopExited == true` → `join()` が期限内に戻る | **×** | OS のスケジューリングに依存する |
| `join()` 完了 → スレッドは実行していない | ○ | `join` の定義 |
| `LoopExited == false` → 停止していない | ○ | 保守側の判定（偽陽性なし） |

### 4.3 採用する扱い（Owner 案 1）

`LoopExited` は「ループ関数が終了直前まで到達した」状態として扱い、**W1 について bounded shutdown を主張しない**。

- preflight が `LoopExited` を観測した場合、その意味は「W1 はもう engine 状態を変更しない」である。
  「停止した」ではない。
- 通常経路（S2）では従来どおり `stopRebuildThread()` の `join()`（`RebuildDispatch.cpp:825-826`）を実行する。
  この join は**無期限**である。
- 保持経路（S3）では join を呼ばない（`I-B2`）。したがって W1 が動き続けても teardown は起きない。

**これは既存挙動と同じであり、v2-a が新たに導入するリスクではない。** 現在のシャットダウンも
`stopRebuildThread()` で無期限に join する。v2-a が変えるのは「bounded だと主張しない」ことだけである。

### 4.4 shutdown の保証範囲（再定義）

| Worker | preflight でできること | 保証 |
| --- | --- | --- |
| W1 `rebuildThread` | 退出要求 + `LoopExited` の期限付き待機 | **状態変更の停止**まで。join の bounded は保証しない |
| W2 `coordinatorLoop_` | `signalThreadShouldExit()` + `waitForThreadToExit(bound)` | 期限内の終了を観測できる（bounded）。超えたら保持 |
| W5 `LoaderThread` | `signalThreadShouldExit()` + `waitForThreadToExit(bound)` | 同上 |
| W6 `ProgressiveUpgradeThread` | `cancel()` + `waitForThreadToExit(bound)` | 同上 |
| W3 / W4 / W7 | 対象外（要求 API が無い） | 保証しない（保持で守る） |

つまり「bounded shutdown」が成立するのは W2・W5・W6 についてだけで、W1 は「作業停止の確認」までである。
この非対称を設計の前提として明記する。

### 4.5 bounded が必要になった場合（Owner 案 2）

W1 の join を bounded にするには、所有権を Engine から切り離して reaper に委ねる設計が要る
（Engine を破棄せず、join 完了を待って reaper が後始末する）。これは所有権設計の変更であり、
v2-a には含めない（§7 U-9）。Owner 案 2 のとおり**別設計**として扱う。

---

## 5. 不変条件（更新）

| # | 不変条件 |
| --- | --- |
| I-A1〜I-A4 | 前稿のまま（状態機械・閉鎖の単一実行・Intent での拒否・要求のみ） |
| I-A5 | Worker は stop target へ登録してから `startThread()` する |
| I-A6 | `requestWorkerStop()` は MT で、S1 の後に 1 回以上呼ばれる |
| I-A7 | S1 以降は新しい Worker を起動しない（入口拒否 + 起動直前の再検査） |
| I-A8 | `loadImpulseResponse()` は MT 前提を assert で固定する |
| I-B1・I-B2 | 前稿のまま |
| I-C1 | 抑止対象（C2〜C7）は **epoch 不一致 OR StopAccepting** で本体を実行しない |
| I-C2 | 抑止しても移転済みの所有権は必ず解放する（RAII を判定より先に生成） |
| I-C3 | callback 本体はメッセージループを回さない |
| I-D1 | `workersStuck_` を false に戻す API を作らない |
| I-E1・I-E2・I-E3・I-E4 | 前稿のまま |
| I-E5 | commit point は §3.2 の 9 箇所。commit 開始前は拒否、開始後は完走させる（R-A〜R-D） |
| I-F1 | W1 の停止確認に `rebuildThreadIsRunning == false` を使わない。`LoopExited` または `!joinable()` のみ |
| I-F2 | `rebuildThreadLoopExited` はループだけが true にし、生成側だけが false にする |
| I-F3 | **`LoopExited` を join 完了の証明として扱わない。W1 の bounded を主張しない** |
| I-F4 | 保持経路では `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` を実行しない |
| I-G1 | `~ConvolverProcessor` から retire provider を呼ばない |

---

## 6. 実装単位の更新（前稿 §7 に対する差分）

| 単位 | 追加・変更 |
| --- | --- |
| P1 | 変更なし（状態機械） |
| P2 | 変更なし（`rebuildThreadLoopExited`） |
| P3 | `requestWorkerStop()` の列挙対象を **`activeLoader` / `loaderTrashBin` / `upgradeThread`** に固定（I-A5/I-A6 の実装） |
| P4 | `loadImpulseResponse()` 入口の gate は `isLoading` publish（`:47`）より前。`JUCE_ASSERT_MESSAGE_THREAD` を追加（I-A8）。`startThread()` 直前の再検査（案 2） |
| P6 | 6 箇所の判定を **epoch OR StopAccepting** に統一（I-C1） |
| P7 | commit point 9 箇所（§3.2）。choke point は 2 つ: `applyComputedIR` 入口（`:470` の直前）と `executePendingCommit` 入口（`:894` の直後） |
| P8 | 変更なし |

W1 に関する追加実装は無い（`LoopExited` は P2 で導入済み）。**bounded を主張しない**のは設計記述の変更である。

---

## 7. 未解決（更新）

| # | 項目 | 内容 | 状態 |
| --- | --- | --- | --- |
| U-1 | `bound` | 5 秒は既存値の踏襲。W2/W5/W6 は未計測、W1 は bounded の対象外 | 未検証の暫定値 |
| U-2 | `stopLearning()` の所要 | 通常経路の reconfigure で join する | 未計測 |
| U-4 | `Closing` のまま停止した場合 | 再入時は `Closing` を実行中として return | 設計済み |
| U-5 | `shutdownAudioOnlyKeepingState()` の副作用 | P8 で設計 | 要設計 |
| U-6 | stuck テレメトリ | `terminalStopState` / `syncWorkActive` / 生存 Worker を出す | 要設計 |
| U-8 | W1 の join が戻らない場合 | S2 で無期限に待つ。既存挙動と同じ。観測（待ち時間ログ）だけ追加する | 受容（計測で監視） |
| U-9 | W1 の bounded 化 | reaper による所有権の切り離し（Owner 案 2）。所有権設計の変更 | **別設計**（未着手） |
| U-10 | commit point の網羅性 | commit point の一覧が将来の追加で漏れないようにする（`applyComputedIR` の choke point が最終防衛線） | レビュー条件に含める |
| U-3 | W3・W4・W7 の停止要求 API | 未承認のまま | 未承認 |

---

## 8. 証跡（本稿で追加確認したソース）

| 対象 | 位置 | 確認内容 |
| --- | --- | --- |
| Worker 生成順 | `LoadPipeline.cpp:85, 93, 99, 129, 142` | 代入 → `startThread()` の順（stop target 登録が先） |
| `loadIR` の commit point | `LoadPipeline.cpp:284, 312, 362, 371` | `applyComputedIR` 3 箇所 + `startProgressiveUpgrade` |
| `applyComputedIR` | `LoadPipeline.cpp:460-492` | 入口（`:470` の generation 検査）と `JUCE_ASSERT_MESSAGE_THREAD`（`:492`） |
| `executePendingCommit` | `LoadPipeline.cpp:892-948` | 入口 `:894`、metadata 更新 `:913`、`irFinalized` publish `:944`。同期 `:834` / callback `:851` の両経路 |
| W1 末尾 | `RebuildDispatch.cpp:1449-1453` | `IsRunning=false` → token=0 が関数の最終処理 |
| W1 join | `RebuildDispatch.cpp:815-832` | 要求 publish → notify → **無期限 join** |
| 呼出し元 | `Lifecycle.cpp:220` / `Rebuild.cpp:17` / `LoadPipeline.cpp:955` | `loadImpulseResponse` / `rebuildAllIRs` は MT 経由 |

---

## 9. 停止条件の確認

- production source / test / CMake を本稿のために変更していない。
- build / test / debugger / replay を本稿のために実行していない。
- git の変更操作を行っていない。HEAD は `173e69957b046290f7077797782f031ed8633fd0`。
- AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。
- U-3 の実装設計には着手していない。

**本稿の提出をもって停止する。** 実装 GO・Gate 1 承認・U-3 着手許可はいずれも付与しない。
