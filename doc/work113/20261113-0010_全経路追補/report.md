# 全経路追補: terminal shutdown の保留 Worker と RetireClosed 後の IRState retire

作業日時: 2026-11-13 00:10〜
基準 commit: `173e6995`（HEAD = origin/main・main ツリー）
性質: **読み取り専用**（production source / test / CMake / git は未変更）。build / test / debugger / replay も未実行。

---

## 0. 結果

```
main 版ソースの識別物   = 追加（ConvoPeq.md + split 4 分割 + MANIFEST。結合一致を実測）
terminal の保留 Worker  = 3 案を比較し、(C) 有限待ち → 失敗時はプロセス終了まで保持 を推奨
RetireClosed 後の retire = 登録は可能（shutdownReclaim → terminalReclaim）
~ConvolverProcessor      = **deferred retire を呼んではならない**（router/domain が先に破棄される）
非割込み区間の計測       = 計画を作成（実施は実装 GO 後）
(b) の追加条件           = ペア getter・lock-free 確認を追加
実装 GO                  = 未承認
```

Owner が指摘した 2 点は、どちらも「設計を変えないと壊れる」ことが確認できた。特に **`~ConvolverProcessor` の時点で retire router / epoch domain は既に破棄されている**（宣言順で convolver が最後）。したがって「IRState の解放を deferred delete に置き換える」は、`releaseResources` では成立するが、デストラクタでは成立しない。ここを分けないと、終了時に破棄済みオブジェクトへ enqueue することになる。

---

## 1. main 版ソースの識別物（Owner の再照合のため）

Dropbox の検索で main 版が見つからなかったという指摘に対して、**main ツリーの抽出物そのもの**を成果物として添付した。

`doc/work113/20261113-0010_全経路追補/main_src/`

| ファイル | 内容 |
|---|---|
| `ConvoPeq.md` | main ツリーの抽出物（7,011,745 B / SHA-256 `1801bbbacf4d65484418a1cf978acf0c09fc728028b3c8c8105ddce95f25e14f` / stamp `2026-11-12 00:48:16` / タイトル `Project Extract & Source Code: ConvoPeq`） |
| `split/ConvoPeq_part01..04.md` | 2 MB 未満に分割（各 1,949,951 / 1,949,940 / 1,950,000 / 1,161,854 B） |
| `MANIFEST.md` | 上記の識別情報と、**4 分割の結合が元ファイルとバイト一致すること**の記録 |

ci-fix 版との区別は 2 点で機械的に行える。

* タイトル行が `ConvoPeq`（ci-fix は `ConvoPeq-ci-fix`）
* サイズが 7,011,745 B（ci-fix は 7,035,321 B）

分割は行境界で切り、結合して SHA-256 が一致することを実測した（`join == original: True`）。

---

## 2. terminal shutdown での保留 Worker の最終処理

### 2.1 現行の動作

`stopThread(timeout)` はタイムアウトすると `killThread()`（Windows は `TerminateThread`）で強制終了し、`false` を返す（`juce_Thread.cpp:245-277`、`juce_Threads_windows.cpp:135-144`）。強制終了はロック・RAII を残すため、通常経路で使えない。新方針では「終了確認が取れるまで Worker と寿命依存全体を保持する」ことになるが、**後続の停止機会が二度と来ない場合**（terminal shutdown）の扱いを決める必要がある。

### 2.2 3 案の比較

| 案 | 内容 | 利点 | 欠点 |
|---|---|---|---|
| (A) terminal で無期限 join | `waitForThreadToExit(-1)` | 確実に停止してから破棄できる | ホストのコールバック上で無期限に待つ。プラグインではホストの watchdog に殺される危険。`releaseResources` はホストスレッドの可能性がある（§3.2） |
| (B) 保持したままプロセス終了 | 保留ホルダを process-lifetime にし、破棄しない（意図的リーク） | UAF も破壊も起きない。ホストを止めない | メモリが解放されない（終了時のみ）。`~Thread` の `stopThread(-1)` を踏まない設計が必要 |
| (C) 有限待ち → 失敗時は (B) | 既存の shutdown 予算（5 s / 2 s / 500 ms）で待ち、失敗なら保持して `TimedOut` を記録 | 通常は停止して回収、異常時のみリーク | 保留ホルダと状態遷移の実装が必要 |

**推奨は (C)。** 既存の shutdown はすでに予算付きである（`kGracefulDrainMaxMs = 5000`：`ReleaseResources.cpp:281`、`waitForDrain(2000, 2)`：`:651`、`kEmergencyDrainMaxMs = 500`：`:388`、`markTimedOut`：`:668`）。同じ予算で待ち、超えたら保持して `TimedOut` を残すのが一貫している。

### 2.3 (C) の実装条件（Owner の 6 条件 + 追加）

1. `waitForThreadToExit()` が `false` の間、Worker を破棄しない。
2. Worker だけでなく、参照先の owner と IRState を含む寿命依存全体を保持する。
3. 保留対象はローカル変数に留めず、後続の停止機会まで存続する管理領域（process-lifetime のホルダ）へ移す。
4. 保留中は、その Worker が参照するリソースと競合する新規処理を開始しない（`isLoading` / `isRebuilding` 相当の状態を保留中として扱う）。
5. 後続の停止機会（次の `prepareToPlay` / `releaseResources` / デストラクタ前の drain）で再待機し、終了確認後に破棄する。
6. **`~Thread` の `stopThread(-1)` を踏まない**こと。ホルダは破棄しない（プロセス終了時に OS が回収する）。`deleteOnThreadEnd` は使わない（owner 側の寿命と結合できない）。
7. terminal で「無条件に owner を破棄する」設計にしない。停止できなければリークを選ぶ。
8. `TimedOut` の記録と、テレメトリ（保留件数）を残す。

---

## 3. RetireClosed 後の IRState retire 経路

### 3.1 RetireClosed の意味

`shutdownRuntime_.transitionTo(RetireClosed)` は `ReleaseResources.cpp:264` で、`ForceEpochAdvance`（`:262`）の直後に置かれる。`isShutdownInProgress()` は「phase が Running でも terminal でもない」間 true なので（`ISRShutdown.cpp:154-158`）、**RetireClosed は shutdown 中**である。

### 3.2 RetireClosed 以降に retire 登録はできるか

**できる。** `AudioEngine::enqueueDeferredDeleteNonRtWithResult` は、shutdown 中なら `markRetireEpoch()` で epoch を取り、`m_retireRouter->shutdownReclaim(...)` に移送する（`AudioEngine.h:4687-4710`）。`shutdownReclaim` は `terminalReclaim` に委譲し（`ISRRetireRouter.cpp:866-875`）、`TerminalReclaimAuthority` は growable store なので**必ず受領する**（`ISRRetireRouter.cpp:609-612` の Stage 5）。epoch safe かつ Non-RT なら即時破棄、そうでなければ保持して後続の drain で解放する。

つまり「RetireClosed だから登録できない」ということはない。制約は **AudioEngine とその router が生存していること**だけである。

### 3.3 破棄順序の問題（重要）

`AudioEngine.h` の宣言順は次のとおり。

| 宣言位置 | メンバ | 破棄順（逆順） |
|---|---|---|
| 916 | `ConvolverProcessor convolver;` | 最後 |
| 2376 | `ConvolverProcessor uiConvolverProcessor;` | 最後の 2 番目 |
| 5456 | `convo::EpochDomain m_epochDomain` | 早い |
| 5461 | `std::unique_ptr<ISRRetireRouter> m_retireRouter` | 早い |
| 5468 | `convo::SnapshotCoordinator m_coordinator` | 最も早い |

したがって `~ConvolverProcessor` が走る時点で、**router と epoch domain（と terminal authority）は既に破棄されている**。ここから `enqueueDeferredDeleteNonRt` を呼ぶと破棄済みオブジェクトへのアクセスになる。現行コードが `~ConvolverProcessor` で provider を使わず即時 free しているのは、この順序を踏まえた設計である（`Lifecycle.cpp:118-131` のコメント「Destructor phase must not depend on rcuProvider/AudioEngine lifetime」）。

### 3.4 設計の分岐

| 経路 | 現状 | 新設計 |
|---|---|---|
| `releaseResources`（router 生存） | 即時 free（`Lifecycle.cpp:506-511`） | **deferred retire へ置換**（`enqueueDeferredDeleteNonRt` → shutdownReclaim → terminal） |
| `~ConvolverProcessor`（router 破棄済み） | 即時 free（`:126-131`） | 次のいずれかが必要: (i) engine dtor body で convolver の IRState を先に retire する、(ii) 即時 free を維持して Worker 停止（quiescence）を証明する、(iii) 意図的リーク |

**推奨は (i) + (ii)。** engine のデストラクタ本体（メンバ破棄前）に「convolver の IRState を retire して drain する」ステップを置き、デストラクタ側の即時 free は「停止済みの Worker しか IRState に触れない」ことを前提に残す。停止できなかった場合は §2 の保留（リーク）に合流させ、即時 free しない。

これで Owner の 4 条件に答える。

| # | 条件 | 回答 |
|---|---|---|
| 1 | RetireClosed 以降に登録できるか | **できる**（`shutdownReclaim` → `terminalReclaim`、growable） |
| 2 | 登録できない場合の terminal 解放 | **router 破棄後（~ConvolverProcessor）は登録できない**。engine dtor body での事前 retire、または即時 free + quiescence、またはリーク |
| 3 | provider / domain が IRState より先に破棄されないか | **現状は先に破棄される**（§3.3）。破棄経路から provider を使わない制約が必要 |
| 4 | 他の `releaseResources` 対象資源の競合 | EQ は自身の domain を持ち `~EQProcessor` で drain する。convolver の engine は `retireStereoConvolver(nullptr)` で即時 free。いずれも「Worker 停止済み」が前提（同じ quiescence 条件） |

---

## 4. 非割込み区間の計測計画（実装前必須・実施は実装 GO 後）

| 対象 | 計測する量 | 方法（実装フェーズ） |
|---|---|---|
| LoaderThread の FFT / パーティション生成 1 回 | 経過時間の p99 / max | 既存の `RtMeasureRing` / `RTTraceRelay` に区間を追加、または局所的に `steady_clock` で計測 |
| リサンプル 1 回 | 同上 | 同上 |
| `convertToHighRes` | 開始から完了までの最大区間 | 同上 |
| キャンセル確認点の間隔 | 最長の「確認なし区間」 | `checkCancellation` 呼出し点の前後で計測 |
| ProgressiveUpgradeThread のステップ | 同上 | `cancelled` 確認点の前後 |

合格条件の案: 有限待ち時間 `T`（例: 2000 ms）を決め、各非割込み区間の max が `T` を十分下回ること。下回らない場合は `T` を区間 max + 余裕に設定し、それを超えたら §2 の保留へ入る（保留管理が機能することをテストで確認する）。平均値は使わない。

---

## 5. (b) の追加条件

Owner 追加分を設計条件に加える。

1. **ペア getter** を用意する（1 回の load で 2 値）。既存の float getter は互換維持のため残してよいが、**両者を別々に呼ぶと世代混在が残る**ことをコメントで明記する。
2. `std::atomic<std::uint64_t>::is_always_lock_free` を `static_assert` する（RT 経路で使う場合に必須）。
3. 既出の条件（`sizeof(float)==4`、`bit_cast`、配置順、初期値、1 store / 1 load、NaN 前提、戻り値型）を維持する。

---

## 6. 残る義務（更新）

| # | 義務 | 内容 | 状態 |
|---|---|---|---|
| W1-O9a | 協調停止 + 保留管理 | §2 の (C)。process-lifetime ホルダ、`TimedOut` 記録、無条件破棄の禁止 | 未（設計） |
| W1-O9a-2 | 非割込み区間の計測 | §4 の計画を実装フェーズで実施 | 未（計画済み） |
| W1-O9b | releaseResources の deferred 化 | §3.4 の (i)+(ii) | 未（設計） |
| W1-O9b-2 | 破棄順序の制約 | `~ConvolverProcessor` から provider を使わない（現状維持）＋ engine dtor body での事前 retire | 未 |
| W1-O8a | 所有バッファ invariant | 維持 | 明文化待ち |
| (b)-1 | 単一語パック + ペア getter + lock-free 確認 | §5 | 未 |
| #7 | guard 対象 | 維持 | 未 |
| W1-O6 | snapshot 2 経路 | 独立実装単位 | 承認済み・未実装 |
| A-O1〜A-O5 / N6 / N7 / N9 | 前回のまま | — | 未 |

---

## 7. 証跡

`doc/work113/20261113-0010_全経路追補/`

| ファイル | 内容 |
|---|---|
| `main_src/MANIFEST.md` | main 版 ConvoPeq.md の識別情報、split 4 分割の SHA-256、結合一致の記録 |
| `main_src/ConvoPeq.md` | main ツリーの抽出物（7,011,745 B） |
| `main_src/split/ConvoPeq_part01..04.md` | 2 MB 未満の分割（結合で元とバイト一致） |

---

## 8. 停止条件の確認

* production source / test / CMake を変更していない（`doc/` 配下の成果物のみ）。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** terminal の保留 Worker と RetireClosed 後の retire 経路について設計材料を揃えたが、実装 GO・Gate 1 承認は付与しない。
