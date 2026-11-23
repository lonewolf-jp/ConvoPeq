# EQ ドメイン reclaim ドライバ設計（実装未承認・設計のみ）

Owner 判断（2026-11-11）を受け、`EQProcessor` の専用 `EpochDomain` に対する reclaim ドライバの設計を具体化する。**タイマー接続・製品コード変更は行っていない。**

## §0 前提（現在のソースで確認した事実）

| 項目 | 事実 | 根拠 |
|---|---|---|
| エンジンの EQ 実体 | `EQEditProcessor final : public EQProcessor, private juce::Timer` | `src/EQEditProcessor.h:28-29` |
| エンジンからの参照 | `AudioEngine::getEQProcessor()` が `uiEqEditor` を返す | `src/audioengine/AudioEngine.h:1338` |
| 実行中の周期処理 | `AudioEngine::timerCallback()`（Message Thread）、`startTimer(100)` | `AudioEngine.Timer.cpp:584`、`AudioEngine.Init.cpp:131` |
| タイマー停止 | `~AudioEngine` の**本体**で `stopTimer()` | `AudioEngine.CtorDtor.cpp:167` |
| EQ ドメインの reclaim | `flushPendingEpochAdvance()` が pending のときだけ `publishEpoch()` → `m_ownedRetireRouter.tryReclaim()` | `EQProcessor.Core.cpp:83-94` |
| 退役側 | `retireEQStateDeferred()` は epoch を進めない（`m_epochAdvancePending` を立てるだけ） | `EQProcessor.Core.cpp:271`、`EQProcessor.h:588` |

**重要な帰結**: 退役は epoch を進めない。reclaim は `isOlder(entry.epoch, minReaderEpoch)` を要求する。reader がいないとき `minReaderEpoch = currentEpoch` なので、`entry.epoch == currentEpoch` のエントリは**publish されるまで永久に reclaim できない**。したがって「`tryReclaim()` だけを定期実行する」案では不十分であり、**publish と reclaim を一体で行う必要がある**（Owner 指摘のとおり）。

## §1 設計案（推奨: 案 A）

### 1.1 呼ぶ関数

`AudioEngine::timerCallback()`（Message Thread）から、**`flushPendingEpochAdvance()` 相当を 1 回呼ぶ**。

```cpp
// 設計案（未実装）: AudioEngine::timerCallback() の末尾付近
if (! isShutdownInProgress())
{
    jassert (! convo::numeric_policy::isAudioThread());   // Message Thread であることの表明
    uiEqEditor.driveRetireReclaimNonRt();                  // 新設の public ラッパ
}
```

`EQProcessor::flushPendingEpochAdvance()` は現在 private のため、`EQEditProcessor` に 1 行の public ラッパを置く（`void driveRetireReclaimNonRt() noexcept { flushPendingEpochAdvance(); }`）。

### 1.2 epoch を進める条件と頻度

| 状況 | 動作 |
|---|---|
| `m_epochAdvancePending == true`（退役があった） | `publishEpoch()` を 1 回 → `m_ownedRetireRouter.tryReclaim()` |
| pending が無く、滞留もある | `tryReclaim()` のみ（publish しない） |
| pending が無く、滞留も無い | 何もしない |

- 頻度の上限は**タイマー周期 100 ms**。退役が無ければ publish しない。
- 1 回のタイマーで publish は最大 1 回（`exchange(false)` の勝者のみ）。
- これにより、通常運転での待ち時間の上限は「退役から最大 100 ms + 次回 reclaim の処理時間」になる。

### 1.3 重複 publish と不要な epoch 前進の防止

- `m_epochAdvancePending` の `exchange(false)` により、同じ退役バッチで publish は 1 回だけ。既存の `prepareToPlay` / `releaseResources` とタイマーが同時に呼んでも、publish は勝者の 1 回だけ。
- 退役が無い期間は publish されない（epoch は無駄に進まない）。
- 既存の `flushPendingEpochAdvance()` の意味を変えないため、`prepareToPlay` / `releaseResources` の挙動も不変。

### 1.4 スレッド・寿命・破棄順序

| 論点 | 設計 |
|---|---|
| 実行スレッド | Message Thread（`timerCallback`）。`jassert(!isAudioThread())` を置く |
| 寿命 | `~AudioEngine` の本体で `stopTimer()` が実行され、その後でメンバ（`uiEqEditor`）が破棄される。**タイマーが止まってから EQ が壊れる**順序であり、破棄後の呼出しは起きない |
| 停止中の防御 | `isShutdownInProgress()` を先頭で確認し、shutdown 中は呼ばない |
| タイマー再開 | 停止後に `startTimer` を呼ぶ経路が無いことを実装時に確認する |
| EQProcessor の破棄 | `~EQProcessor` が `tryReclaim` / `drainAll` を実行する既存経路はそのまま |

### 1.5 D→Q→E→T 経路との整合

`m_ownedRetireRouter.tryReclaim()` は EQ ドメインの D + Q + E + T を既存の epoch-gated 経路で drain する。新しい authority や store は追加しない。engine ドメインの経路とは独立（epoch provenance は EQ ドメインのまま）。

### 1.6 同時実行・再入

- `timerCallback` は Message Thread の 1 本のみ。`prepareToPlay` / `releaseResources` も同じスレッドで呼ばれる想定であり、**同一スレッド内の再入は起こらない**（タイマーコールバック中に prepareToPlay は走らない）。
- 将来別スレッドから呼ばれた場合も、`m_epochAdvancePending` の exchange で publish は 1 回、`tryReclaim` はキュー内部の CAS（`dequeuePos`）でエントリ単位に直列化される。**外部 mutex は追加しない**。
- 再入防止のため、`driveRetireReclaimNonRt()` には `std::atomic_flag` による再入ガードを置く案を併記する（必須ではない。キューが壊れないため）。

### 1.7 案の比較

| 案 | 内容 | 評価 |
|---|---|---|
| **A（推奨）** | タイマーから `flushPendingEpochAdvance()`（publish + reclaim） | epoch 前進と reclaim が一体。既存関数を再利用。publish はバッチごとに 1 回 |
| B | タイマーから `tryReclaim()` のみ | **不可**。退役は epoch を進めないため、`epoch == currentEpoch` のエントリは永久に reclaim されない |
| C | 専用スレッドを追加 | 不要（スレッド追加はコストと寿命管理が増える） |
| D | `EQProcessor` のタイマー（`EQEditProcessor` は `juce::Timer` を private 継承）を使う | 既に Timer を持っているが、コールバックの用途が別（UI）。混在させると責務が不明瞭 |

### 1.8 実装時の変更点（承認後）

| ファイル | 変更 |
|---|---|
| `src/EQEditProcessor.h` | `driveRetireReclaimNonRt()`（1 行ラッパ） |
| `src/audioengine/AudioEngine.Timer.cpp` | `timerCallback()` から呼ぶ（shutdown ガード付き） |

## §2 検証計画（実装時）

| # | 検証 | 期待 |
|---|---|---|
| 1 | `retire_delay` の最大待ち | 100 ms 程度以下（タイマー周期 + 処理時間） |
| 2 | EQ ドメインの `publishCount` | 退役バッチごとに 1 回（同一バッチで複数回にならない） |
| 3 | 通常運転での `pendingRetireCount` | タイマー周期内に 0 に戻る |
| 4 | `evaluateRetireDelay` | バイト一致・閾値 50 ms・N_req 59 は不変 |
| 5 | 既存 60 テスト | 合格 |
| 6 | shutdown 順序 | `stopTimer` 後に呼ばれないこと（ログで確認） |

## §3 未確定事項（Owner 判断）

1. タイマー周期（100 ms）のままでよいか。50 ms 閾値に対しては 100 ms で不足するため、閾値内に収めるには周期の短縮か、retire 時の即時 flush が必要になる。**周期を短縮すると publish 頻度が上がる**ため、閾値との関係を先に決める必要がある。
2. `retire 時に即時 flush する`案（`retireEQStateDeferred` の直後に publish+reclaim）は、retire が NonRT（パラメータ変更・prepare）である場合に限り可能。RT から retire する経路があるかは未確認。
3. 再入ガード（`atomic_flag`）を置くか。

## §4 付記: 保留事項への回答

### 4.1 `ScopedEpochReader` の 512 件問題（Owner 指摘 A）

そのとおりである。`ScopedEpochReader` は共通レジストリではなく独立した 512 スロット表を使っており、今回のタグ別件数の合計はちょうど 512 で**飽和の可能性**がある。共通レジストリの `recordFailures=0` は、この独立表の完全性を保証しない。

対応案（未実装）: `ScopedEpochReader` の計測を共通レジストリ（128 スロット＋再利用）へ統合する。統合すれば `recordFailures` / `aggMisses` が同じ意味で読める。今回の「最長 8.1 ms」は飽和前の記録に基づく下限値として扱う。

### 4.2 `publishCount=1` の同一実行証拠（Owner 指摘 B）

今回の D1 JSON には EQ ドメインの publish 回数が無い。次の計測で 1 つの実行から次を同時に取得する（実装は計測のみ）。

| 証拠 | 取得方法（設計） |
|---|---|
| EQ ドメインの `globalEpoch` 推移と `publishCount` | `EpochDomain` に計数（前回の計測と同じ）＋ final / postTeardown のダンプ |
| `reclaim` 呼出し回数 | `DeferredDeletionQueue::reclaim` の呼出し回数（atomic） |
| enqueue 時刻と reclaim 実行時刻 | レコーダの `enqueueTsNs` / `safeTsNs` ＋ 新設の `firstScanTsNs`（分類設計 §1） |
| reader slot の active 状態と epoch | 既存の `[M]` 相当のダンプ（NonRT・1 秒 1 回） |

## §5 Owner 推奨判断の反映（2026-11-11）

| 論点 | 決定（推奨） |
|---|---|
| タイマー周期 | **100 ms を維持**（短縮しない） |
| 50 ms の意味 | **reclaim ドライバ待ちの性能目標・診断閾値**として扱う。50 ms 以内の保証はしない |
| `evaluateRetireDelay` / N_req / `retireLatencyD1` | 不変 |
| 実装の段階 | ① reclaim ドライバ → ② 競合・寿命テスト → ③ `firstScanTs` と 3 分類 → ④ D1 JSON の後方互換性検証 → ⑤ 同一実行の統合計測 → ⑥ Reader 共通レジストリ統合（次段階） |
| 実装上の必須条件 | pending 時のみ publish／`exchange(false)` の勝者のみ／publish と reclaim の一体構造維持／Message Thread 限定／shutdown 中は呼ばない／破棄前にタイマー停止／既存 `m_ownedRetireRouter` の再利用（新 authority なし） |
| 追加テスト | `driveRetireReclaimNonRt()` の公開ラッパと、他の publish 経路との競合 |

**今回のステータス**: 設計提出まで。実装は未着手（Owner 指示「設計 2 点の提出完了と、実装未着手というステータスを維持」）。

### 5.1 実装着手前の確認事項（Owner 指摘）

参照可能な結合ソースが 2026-10-07 版であり HEAD `c94d3767` と対応していないため、**HEAD 対応の結合ソースを生成して提出した**（`doc/work113/20261111-0400_更新ソース/ConvoPeq.md`、2 MB 未満に 4 分割した `split/` を含む）。実装承認時はこの版で差分を再確認する。
