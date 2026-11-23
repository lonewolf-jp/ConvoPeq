# 保留 Worker の所有権構造（READ-ONLY 設計）

作業日時: 2026-11-13 00:30〜
基準 commit: `173e6995`（HEAD = origin/main・main ツリー）
性質: **読み取り専用**。production source / test / CMake / git を変更していない。build / test / debugger / replay も実行していない。

---

## 0. 結果

```
推奨する所有権構造 = MainApplication::shutdown() のゲート + process-lifetime ホルダ（MainWindow を保持）
理由             = Worker の owner は ConvolverProcessor であり、それは AudioEngine（MainWindow の値メンバ）の中にあるため
代替案           = Worker の参照先を ConvolverWorkerContext へ抽出（owner API 20 点の付け替え。将来案）
実装 GO           = 未承認
```

Owner が求めた「holder が Worker の参照する owner より長く存続すること」は、**Worker と owner を別々に保持しても成立しない**。owner が `AudioEngine` の値メンバだからである。成立させる最小の構造は、**`AudioEngine` を内包する `MainWindow` をプロセス終了まで保持する**ことであり、その入口は `MainApplication::shutdown()` の 1 箇所しかない。

---

## 1. 所有連鎖（コード実測）

```
MainApplication                      MainApplication.h:40   std::unique_ptr<MainWindow> mainWindow;
  └ MainWindow                       MainWindow.h:65        AudioEngine audioEngine;            ← 値メンバ
      │                              MainWindow.h:66        std::unique_ptr<AudioEngineProcessor> audioEngineProcessor;
      └ AudioEngine                  AudioEngineProcessor.h:42   AudioEngine& audioEngine;       ← 参照
          ├ convolver                AudioEngine.h:916      ConvolverProcessor convolver;
          ├ uiConvolverProcessor     AudioEngine.h:2376     ConvolverProcessor uiConvolverProcessor;
          ├ m_epochDomain            AudioEngine.h:5456     convo::EpochDomain
          ├ m_retireRouter           AudioEngine.h:5461     std::unique_ptr<ISRRetireRouter>
          └ m_coordinator            AudioEngine.h:5468     convo::SnapshotCoordinator
```

* `ConvolverProcessor` は `activeLoader` / `upgradeThread` / `loaderTrashBin` と `currentIRState` を所有する。
* `LoaderThread` は `ConvolverProcessor& owner` を保持し、終了時に `owner.retireStereoConvolver(...)` を呼ぶ（`LoaderThread.cpp:46-47`）。
* 破棄順は宣言の逆順なので、**`~ConvolverProcessor` より先に `m_coordinator` / `m_retireRouter` / `m_epochDomain` が破棄される**（前稿 §3.3 の再掲）。

したがって「Worker の owner を生かす」＝「`ConvolverProcessor` を生かす」＝「`AudioEngine` を生かす」＝「`MainWindow` を生かす」である。

### 1.1 破棄経路は 1 つ

通常の破棄経路は `MainApplication::shutdown()` の `mainWindow.reset()`（`MainApplication.cpp:229`）の 1 箇所だけで、`~AudioEngine` / `~ConvolverProcessor` はその内側で走る。**通常経路でゲートを置ける場所はここしかない。**

異常終了時は `~MainApplication` が `mainWindow` を破棄する経路が残る（`shutdown()` を経ない）。クラッシュ時は問題にならないが、それ以外の異常終了ではゲートを通らない。この残留は §2.3 の不変条件 1 に「通常経路」と明記して扱う。

---

## 2. 推奨する所有権構造

### 2.1 形

```
MainApplication::shutdown()
  ├ if (mainWindow && mainWindow->hasUnstoppedWorkers())
  │     StuckProcessHolder::instance().keep(std::move(mainWindow));   // static・破棄しない
  │     log "[SHUTDOWN] stuck worker: keeping MainWindow until process exit"
  └ else
        mainWindow.reset();                                          // 従来どおり
```

* `StuckProcessHolder` は**プロセス終了まで破棄しない**静的な保持領域（`std::vector<std::unique_ptr<MainWindow>>` 相当）。
* 保持された `MainWindow` は `~MainWindow` / `~AudioEngine` / `~ConvolverProcessor` / `~Thread` のいずれも走らせない。したがって `~Thread` の `stopThread(-1)`（無限待ち）も踏まない。
* Worker は `owner` 参照を保ったまま継続し、終了時に `owner.retireStereoConvolver(...)` を呼ぶ。owner（と router / domain）は保持されているので有効である。
* retire された object は drain されない（engine の timer / coordinator は停止済み）。プロセス終了で OS が回収する。**回収不能を許容する**という Owner 判断に一致する。

### 2.2 判定 API

| 層 | API | 内容 |
|---|---|---|
| `ConvolverProcessor` | `bool hasUnstoppedWorker() const noexcept` | upgrade / loader のいずれかが「停止要求後も終了未確認」であることを返す |
| `AudioEngine` | `bool hasUnstoppedWorkers() const noexcept` | 2 つの ConvolverProcessor の OR |
| `MainWindow` | `bool hasUnstoppedWorkers() const noexcept` | engine へ委譲 |

「未確認」は、`releaseResources` の停止処理で `waitForThreadToExit()` が `false` を返した時点でフラグを立てる（同時に `TimedOut` を記録する）。停止できた場合はフラグを立てない。

### 2.3 不変条件

1. **通常経路では、`~AudioEngine` に入る時点で未停止 Worker が存在しない。** ゲート（§2.1）がこれを保証する。異常終了（`shutdown()` を経ない破棄）は対象外であり、残留リスクとして記録する。新しい破棄経路を追加する場合は、同じゲートを通すこと。
2. `~ConvolverProcessor` から retire provider を呼ばない（router/domain は破棄済み）。
3. 保持された `MainWindow` は二度と破棄しない（`~Thread` の無限待ちを避ける）。
4. 保持中は新しいロードを開始しない（UI は終了済みだが、Worker 自身が `applyComputedIR` を呼ぶ可能性は残る。これは owner が生存しているので安全）。

### 2.4 デストラクタ内の停止をどう変えるか

現行の `stopThread(500)`（`LoaderThread.cpp:44`、`StateAndUI.cpp:1051`）、`stopThread(2000)`（`LoadPipeline.cpp:106`）は、タイムアウトで強制終了する。通常経路から強制終了を排除するため、次の形にする。

```
signalThreadShouldExit();                       // LoaderThread
cancel();                                       // ProgressiveUpgradeThread（協調フラグ）
if (!waitForThreadToExit(bound)) {              // 殺さない
    stuck = true;                               // → ゲートで保持
    return;                                     // 破棄しない
}
```

`~AudioEngine` の内側では「`stuck == false`」が不変条件なので、ここでの `return` は実際には到達しない（防御）。到達した場合はログを残し、**保持経路に移れない以上、その場では何も破棄しない**（リークしてプロセス終了に委ねる）。

---

## 3. 代替案: Worker の参照先を owner から切り離す

| 案 | 内容 | 変更規模 |
|---|---|---|
| (A) 推奨案 | `MainWindow` を保持（§2） | `MainApplication::shutdown()` + 判定 API 3 つ + 停止処理の書き換え |
| (B) owner-detach | `ConvolverWorkerContext`（IRState / scale / warnings / generation / retire sink / 警告 API）を抽出し、Worker は context のみ参照。holder が context を所有 | LoaderThread の owner 経由 17 メンバ、ProgressiveUpgradeThread の 3 メンバの付け替え |

`LoaderThread` が owner に要求する API は 17 個ある（`convolverStateGeneration`、`retireStereoConvolver`、`setLastPhaseWarning`、`applyNewState`、`finalizeNUCEngineOnMessageThread`、`computeTargetIRLength`、`acquireIRState` ほか）。`applyNewState` や `finalizeNUCEngineOnMessageThread` は processor の内部状態に強く結合しているため、(B) は「抽出」ではなく「再設計」に近い。

**今回の目的（状態破壊を避ける）には (A) で足りる。** (B) はリークを避けたい場合の将来案として残す。

---

## 4. Owner 条件への対応

| # | 条件 | 対応 |
|---|---|---|
| 1 | `waitForThreadToExit()` が false の間、Worker を破棄しない | §2.1（ゲートで `MainWindow` ごと保持） |
| 2 | Worker・owner・IRState の寿命依存全体を保持 | `MainWindow` を保持するので全部生き残る |
| 3 | 保留対象をローカルに留めず管理領域へ | `StuckProcessHolder`（static・破棄しない） |
| 4 | 競合する新規処理を開始しない | ゲート後は UI / device が停止済み。Worker 自身の `applyComputedIR` は owner 生存下で安全 |
| 5 | 後続の停止機会で再待機 | 保持中は再待機しない（プロセス終了まで保持）。再待機が必要な設計にする場合は、releaseResources の再入でフラグを再評価する |
| 6 | `~Thread` の `stopThread(-1)` を踏まない | ホルダを破棄しないので `~Thread` は走らない |
| 7 | terminal で無条件に owner を破棄しない | ゲートが `mainWindow.reset()` を回避する |
| 8 | `TimedOut` の記録 | 停止失敗時に記録し、保持件数をテレメトリに残す |
| **追加** | **holder が owner より長く存続** | **holder が `MainWindow`（= owner の祖先）を所有するため、構造的に成立** |

### 4.1 デストラクタ側の 2 つの確認（Owner 指摘）

engine dtor body で IRState を事前 retire しても、「Worker が IRState を参照していない」証明にはならない。次の 2 つを**別々**に確認する。

| 確認 | 手段 | 失敗時 |
|---|---|---|
| (a) Worker の終了確認 | `hasUnstoppedWorker() == false`（ゲート済み） | ゲートで保持（dtor に入らない） |
| (b) IRState の reclaim 可否 | 事前 retire 後の epoch-gated drain（`minReaderEpoch` と retire epoch の比較） | 即時 free しない（保持へ合流） |

両方が成立したときだけ、`~ConvolverProcessor` の即時 free を実行する。

---

## 5. 残る設計項目

| # | 項目 | 内容 | 状態 |
|---|---|---|---|
| D-1 | ゲートの実装設計 | `StuckProcessHolder` の型・配置・テレメトリ、`hasUnstoppedWorkers()` の 3 層 API | 本稿で形を提示（実装は未承認） |
| D-2 | 停止処理の書き換え | `stopThread` → `signal` + `cancel` + `waitForThreadToExit` | 同上 |
| D-3 | 非割込み区間の計測 | 計測のみの変更を先行（Owner の段階 1） | 未（実装 GO 待ち） |
| D-4 | engine dtor body の事前 retire | メンバ破棄前に IRState を retire して drain | 未 |
| D-5 | IRState の guard 9 箇所 | 前稿の範囲 | 未 |
| D-6 | (b) の単一語パック + ペア getter | 前稿 §5 | 未 |
| D-7 | snapshot 2 経路 | 独立実装単位 | 承認済み・未実装 |
| D-8 | N6 / N7 / N9 | 前回のまま | 未 |

---

## 6. 証跡

`doc/work113/20261113-0030_所有権設計/evidence/`

| ファイル | 内容 |
|---|---|
| `ownership_excerpts.md` | 所有連鎖の逐語抜粋（`MainApplication` の `mainWindow.reset()`、`MainWindow` の値メンバ、`AudioEngineProcessor` の参照、`AudioEngine` のメンバ宣言順、停止 API）（7 ファイル / 261 行 / 9,137 文字、SHA-256 付き、HEAD blob 基準） |

---

## 7. 停止条件の確認

* production source / test / CMake を変更していない。
* build / test / debugger / replay を実行していない。
* git の変更操作を行っていない。HEAD は `173e6995`、`origin/main` と一致。
* AGENTS.md の既存未コミット変更に触れていない。設計差分 v2.9 も不変。

**本稿の提出をもって停止する。** 所有権構造の形を提示したが、実装 GO・Gate 1 承認は付与しない。
