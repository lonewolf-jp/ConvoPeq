# R-107 Stage 1 — ロック取得点・アロケーション点の列挙と変更計画（2026-11-02）

Owner 承認（R-107 Stage 1）に基づく対象列挙。**実装は未着手**（Stage 2 以降）。

- 基準: `main` = `42232148`
- 手法: `git grep` による全数走査 ＋ 関数帰属の特定 ＋ RT 入口からの到達性判定

## 1. RT 経路の入口と直接の呼び出し

| 項目 | 実測 |
|---|---|
| RT 入口 | `AudioEngine::getNextAudioBlock`（`AudioEngine.Processing.AudioBlock.cpp:29`） |
| 呼び出し | `process` / `processCrossfadeDelayGateIfPending` / `processToBuffer` / `publishAtomic` / `updateAtomicMaximum` |
| **AudioBlock.cpp 内のロック取得** | **0 件** |
| **AudioBlock.cpp 内のアロケーション** | **0 件** |
| DSPCore 処理（Float / Double / IO） | **ロック 0 件・アロケーション 0 件** |
| `processCrossfadeDelayGateIfPending`（`AudioEngine.h:4446`） | カウンタ減算と関数呼び出しのみ。ロック・確保なし |

→ **現行の RT コールバック経路は、走査した範囲では lock-free・非確保**である。

## 2. ロック取得点の全数（RT 関連ファイル）

`src/audioengine` の RT 関連 22 ファイルを走査した結果、ロック取得は次の 16 箇所。

| # | ファイル:行 | 取得する mutex | 所属関数 | スレッド分類 |
|---|---|---|---|---|
| 1 | `ISRDSPHandle.cpp:53` | `freeListMutex_` | `DSPHandleRuntime::create` | 非 RT（生成） |
| 2 | `ISRDSPHandle.cpp:233` | `crossfadeRecordsMutex_` | `beginCrossfade` | 非 RT（コミット） |
| 3 | `ISRDSPHandle.cpp:256` | `crossfadeRecordsMutex_` | `endCrossfade` | 非 RT（Timer） |
| 4 | `ISRDSPHandle.cpp:304` | `freeListMutex_` | `reclaim` | 非 RT（Commit / 解放処理） |
| 5 | `ISRDSPHandle.cpp:361` | `freeListMutex_` | `invalidateForDestroy` | 非 RT |
| 6 | `ISRDSPHandle.cpp:384` | `freeListMutex_` | `invalidateForDestroy`（内部） | 非 RT |
| 7 | `ISRDSPHandle.cpp:423` | `freeListMutex_` | `invalidateForDestroyBySlot` | 非 RT |
| 8 | `ISRDSPHandle.cpp:484` | `freeListMutex_` | `retireIfCurrent` 系 | 非 RT |
| 9 | `ISRDSPHandle.cpp:504` | `crossfadeRecordsMutex_` | `isSlotInCrossfade` | 非 RT（`endCrossfade` から） |
| 10 | `ISRDSPHandle.cpp:590` | `freeListMutex_` | `reclaimLocked` 系 | 非 RT |
| 11 | `ISRDSPHandle.cpp:728` | `recordsMutex_` | `CrossfadeAuthorityRuntime::registerCrossfade` | 非 RT |
| 12 | `ISRDSPHandle.cpp:736` | `recordsMutex_` | `unregisterCrossfade` | 非 RT（Timer） |
| 13 | `ISRDSPHandle.cpp:748` | `recordsMutex_` | `getActiveCrossfades` | 非 RT（Timer） |
| 14 | `ISRDSPHandle.cpp:761` | `recordsMutex_` | `hasCrossfadeInvolving` | 非 RT |
| 15 | `AudioEngine.Processing.PrepareToPlay.cpp:98` | `rebuildMutex` | `prepareToPlay` | 非 RT（開始前） |
| 16 | `AudioEngine.Processing.ReleaseResources.cpp:170` | `rebuildMutex` | `releaseResources` | 非 RT（終了後） |

- `reclaim()` の呼び出し元は `AudioEngine.Commit.cpp`（632/639/684）と
  `ReleaseResources.cpp`（479/863/870）で、**RT コールバックからは呼ばれていない**。
- `getActiveCrossfades()` の呼び出し元は `AudioEngine.Timer.cpp`（1116/1902）で **Timer スレッド**。

## 3. アロケーション点の全数（RT 関連ファイル）

| # | ファイル:行 | 操作 | 所属関数 | スレッド分類 |
|---|---|---|---|---|
| 1 | `ISRDSPHandle.cpp:234` | `crossfadeRecords_.push_back` | `beginCrossfade` | 非 RT |
| 2 | `ISRDSPHandle.cpp:729` | `records_.push_back` | `registerCrossfade` | 非 RT |
| 3 | `ISRDSPHandle.cpp:752` | `result.push_back`（戻り値ベクタ） | `getActiveCrossfades` | 非 RT（Timer） |
| 4 | `DSPCoreLifecycle.cpp:58-62` | `new ...RuntimeState()` ×5 | `DSPCore` 構築 | 非 RT（構築時） |
| 5 | `PrepareToPlay.cpp:268` | `aligned_make_unique<DSPCore>` | `prepareToPlay` | 非 RT |

- RT コールバック経路に確保は無い（`AudioBlock.cpp`・DSPCore 処理とも 0 件）。
- `juce::String` の使用は `diagLog` など**診断ログ**（`#if` で Release 無効）に限られる。

## 4. RT 経路への影響評価

| 経路 | 影響 | 根拠 |
|---|---|---|
| 直接のロック | **なし** | `AudioBlock.cpp` にロック 0 件。呼び出し先も atomics のみ |
| 直接の確保 | **なし** | 同上。DSPCore 処理も 0 件 |
| 間接（共有 mutex の保持） | **可能性あり** | 非 RT が `freeListMutex_` / `crossfadeRecordsMutex_` を保持している間に RT が同じ状態へ atomic で触れる設計。ロック待ちにはならないが、状態遷移の遅延として観測され得る |

## 5. U4 の変更計画（Stage 2・未承認実行）

### 5.1 計測対象（限定）

| 対象 | 計測内容 | 理由 |
|---|---|---|
| `freeListMutex_`（`ISRDSPHandle`） | 取得回数・保持時間（最大／平均） | 解放・再クレーム経路の遅延源 |
| `crossfadeRecordsMutex_` | 同上 | クロスフェード登録・終了の遅延源 |
| `recordsMutex_` | 同上 | Timer 集計の遅延源 |

※ RT コールバックは計測しない（RT に計測コードを入れない）。**非 RT 側のみ**を計測する。

### 5.2 実装方針

- `#if CONVOPEQ_RT_MEASURE` の既定 OFF とし、ON のときだけ `ScopedLockTimer` を差し込む
- 計測値は既存の RT 計測リング（別スレッド集計）へ流す。新規の RT 経路確保は行わない
- 出力は `[U4] mutex=<name> acquired=<n> max_hold_us=<x> mean_hold_us=<y>` 形式

### 5.3 検証計画

| 段階 | 内容 | 合格条件 |
|---|---|---|
| V-1 | 既定 OFF のビルド・全テスト | Release / Debug / ASan PASS |
| V-2 | 既定 ON の限定測定 | RT 経路の新規確保 0・ロック取得 0 を維持 |
| V-3 | 保持時間の分布 | 各 mutex の最大保持時間を記録（閾値判定はしない） |
| V-4 | 性能回帰（OFF 時） | スループット ±1% 以内 |

### 5.4 想定リスク

| リスク | 緩和 |
|---|---|
| 計測コードが非 RT 側のタイミングを変える | 計測は ON 時のみ。OFF 時は完全に除去（`#if`） |
| 計測値の読み取りで競合 | 既存の計測リング（ロックフリー）を使用 |
| 対象の取り違え（RT に計測を入れる） | RT 経路に計測点を置かないことを設計で固定 |

## 6. U3 の実装計画（Stage 3・別承認）

| 項目 | 内容 |
|---|---|
| 目的 | RT 経路の新規確保を**検出**する（置換は既定 OFF・RT 判定時のみ記録） |
| 方式 | `operator new` / `operator delete` のグローバル置換。スレッドローカルなフラグで RT スレッドのみ記録 |
| 影響範囲 | **全モジュール**（リンク順・ODR に注意） |
| 前提 | U4 の検証完了後。U4 と一括変更しない |
| 合格条件 | RT 経路で新規確保 0 を確認。既定 OFF で Release/Debug/ASan PASS・性能回帰 ±1% 以内 |
| リスク | 置換が全アロケーションに及ぶため、導入時のリンクエラー・初期化順の破壊が起こり得る。段階導入（テスト専用バイナリで先行）を推奨 |

## 7. まとめ

- **RT 経路は現状 lock-free・非確保**であり、U4 の計測対象は「RT を直接ロックする箇所」ではなく
  「RT と状態を共有する非 RT 側の mutex 3 種」に限定するのが妥当。
- U3 は全アロケーションに影響するため、U4 完了後に別ブランチで段階導入する。
