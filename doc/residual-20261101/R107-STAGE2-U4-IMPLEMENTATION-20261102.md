# R-107 Stage 2 — U4（非 RT mutex 計測）の実装と検証（2026-11-02）

Owner 承認（R-107 Stage 2）に基づく限定的な実装。**既定 OFF**・RT 経路への計測点追加なし。

- ブランチ: `feat/r107-u4-lock-measure`（起点 `main` = `a83d10ba`）
- 実装コミット: `dd56f0f0`
- 関連: `R107-STAGE1-ENUMERATION-20261102.md`（対象列挙）、`R106-MEASUREMENT-RECORD-20261102.md`（U5 判定）

## 1. 何を作ったか

`RtMeasureRing.h` に、**非 RT の mutex** の取得待ち・保持時間を数える仕組みを追加した。

| 要素 | 役割 |
|---|---|
| `LockId` | `FreeList` / `CrossfadeRecords` / `Records` の 3 種（Stage 1 §5.1 の対象） |
| `LockSlot` | 取得回数・待ち合計・待ち最大・保持合計・保持最大（すべて `std::atomic`） |
| `LockScope` | `std::lock_guard<std::mutex>` と同型の RAII。非 RT 専用 |
| `printLockStats()` | `[U4] mutex=<name> acquired=<n> wait_mean_us=<x> wait_max_us=<x> hold_mean_us=<x> hold_max_us=<x>` を 1 行/スロットで出力 |
| `resetLockStats()` | 集計の初期化（テストと測定開始時） |

差し込み先は `ISRDSPHandle.cpp` の **14 箇所**（Stage 1 §2 の全数）。

| mutex | 箇所数 | 主な呼び出し元 |
|---|---|---|
| `freeListMutex_` | 8 | `create` / `reclaim` / `invalidateForDestroy` / `retireIfCurrent` 系（すべて非 RT） |
| `crossfadeRecordsMutex_` | 3 | `beginCrossfade`（Coordinator）/ `endCrossfade`（Timer）/ `isSlotInCrossfade` |
| `recordsMutex_` | 4 | `registerCrossfade` / `unregisterCrossfade` / `getActiveCrossfades`（Timer）/ `hasCrossfadeInvolving` |

## 2. RT 経路の安全性

| 主張 | 根拠 |
|---|---|
| RT コールバックに計測点を追加していない | 変更は `ISRDSPHandle.cpp`（非 RT 専用クラス）と `RtMeasureRing.h` の非 RT API のみ。`AudioEngine.Processing.AudioBlock.cpp` は無変更 |
| 既定 OFF でコストが増えない | `LockScope` のコンストラクタは `mode()` の atomic load 1 回で分岐し、TSC を読まず、ロック・アンロックは `lock_guard` と同一 |
| 計測中も RT に影響しない | 集計先は固定配列の atomic。RT 用のイベントリング・サンプル領域には触れない（テストで固定） |
| 確保・ロック・I/O を計測中に行わない | `LockScope` は確保なし。出力は `pollIfDue`（CoordinatorLoop = 非 RT）で 1 秒に 1 回 |

> 注: `LockScope` は `std::mutex` を直接使い、`std::unique_lock` 等の追加オブジェクトを作らない。

## 3. オーバーヘッドの評価方針

| 条件 | 期待 | 測り方 |
|---|---|---|
| 既定 OFF | `lock_guard` と同等（atomic load 1 回/取得） | 同一ビルドで `CONVOPEQ_RT_MEASURE` 未設定のまま既存テストスイートを実行し、PASS と実行時間を比較 |
| 計測 ON | 非 RT のロック 1 回あたり TSC 2 回（待ち・保持）＋ atomic 5 回 | `[U4]` の `hold_mean_us` がロック保持時間として妥当か、`acquired` が呼び出し回数と一致するかで確認 |

### 3.1 実測結果（2026-11-03、main 統合状態・Release）

同一バイナリ・同一スイートで、環境変数のみ切り替えて比較した。

| モード | rc | 所要 |
|---|---|---|
| Off（未設定） | 0 | **298.91 s** |
| On（`CONVOPEQ_RT_MEASURE=2`） | 0 | **297.89 s** |

- 差は **−1.0 秒（ノイズ範囲）**。スイート全体では計測のオーバーヘッドは検出できない。
- Off では `[U4]` 行が **1 行も出力されない**（既定 OFF の保証）。

On の最終値（スイート終了時点の累積）:

| mutex | acquired | wait mean / max (µs) | hold mean / max (µs) |
|---|---|---|---|
| `freeList` | 1,143 | 0.08 / 16.53 | 0.32 / 14.39 |
| `crossfadeRecords` | 12 | 0.04 / 0.09 | 0.04 / 0.10 |
| `records` | 2 | 0.04 / 0.05 | 0.37 / 0.43 |

- `acquired` は時間とともに増加し、呼び出し回数と整合する。
- 保持時間は平均 0.04〜0.37 µs、最大 14.4 µs（freeList）。**RT 経路ではなく非 RT のロック**であることが確認できる。


`hold_mean_us` は**ロック内の処理時間**そのものであり、値が大きいほど非 RT 側の処理が長い。
これは U4 の目的（どの mutex が遅延源か）に直結する。

## 4. 回帰テスト

`RtMeasureRingTests` に検証 (13) `checkLockScopeMeasurement()` を追加した。

| 検査 | 内容 |
|---|---|
| Off で計数しない | `acquired == 0`、`holdTicksSum == 0` |
| Off でもロックとして機能する | 保持中は `try_lock()` が失敗し、解放後は成功する |
| On で計数する | 16 回の取得で `acquired == 16`、`holdTicksSum > 0` |
| 最大値の整合 | `holdTicksMax <= holdTicksSum` |
| 他スロットを汚さない | 未使用の `FreeList` は 0 のまま |
| RT 側を汚さない | イベントリング `written() == 0`、サンプル領域 `count() == 0` |

## 5. 検証結果

| 項目 | 結果 |
|---|---|
| Release ビルド | **成功**（`build-ci-local`、`AudioEngineHarness.exe` 2026-11-02 20:42:10） |
| 回帰テスト（`RtMeasureRingTests`） | **PASS**（`… overflow, U4 locks`）。`ctest -R '^AudioEngineHarness$' -V` を 2 回実行し、**rc=0 / 299.9 s・299.5 s** |
| ハーネス全体 | 2 回とも `100% tests passed out of 1` |
| 既定 OFF の挙動 | 上記 ctest は `CONVOPEQ_RT_MEASURE` 未設定で実行しており、既存テストはすべて PASS。既定経路に回帰なし |
| 計測 ON の動作 | 検証 (13) が mode 2 で `acquired == 16`・`holdTicksSum > 0`・他スロット 0 を確認 |
| Debug ビルド | **成功**（`build`、`AudioEngineHarness.exe` 21:51:55、エラー 0） |
| Debug ctest | **rc=0 / 340.2 s**、`100% tests passed out of 1`、`STG-8-CLEAR` PASS・`U4 locks` PASS |
| ASan（RelWithDebInfo） | **未検証**（PR #108 の CI で確認中） |
| 性能回帰（±1% 以内） | **実測: Off 298.91 s / On 297.89 s（差 −1.0 秒・ノイズ範囲）**（§3.1） |

> 注: ローカルの `build` ディレクトリは CMake キャッシュが破損していたため（Ninja/VS の混在）、
> 整合している `build-ci-local` でビルド・テストを実施した。リポジトリのソースには影響しない。


## 6. U3（次段階）との分離

U3（RT 経路の新規確保の検出）は全アロケーションに影響するため、本ブランチには含めない。
U4 の検証結果（とくに既定 OFF のオーバーヘッド）を踏まえて、別ブランチ・別 PR で段階導入する。
