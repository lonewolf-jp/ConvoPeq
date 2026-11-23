# reclaim ドライバ改訂設計 v2（実装未承認・設計のみ）

Owner 指摘（2026-11-11）の 2 点を解消した改訂版である。**製品コードの変更は行っていない（実装未着手）。**

## §0 解消する 2 つの不整合

| # | 指摘 | 本改訂での解消 |
|---|---|---|
| 1 | `flushPendingEpochAdvance()` は pending が無いと reclaim もしない。設計 v1 §1.2 の「pending 無しでも滞留があれば `tryReclaim`」を、既存関数を呼ぶだけでは実現できない | **no-pending の reclaim を新設メソッド側に置く**。既存関数は一切変更しない |
| 2 | 既存 `flushPendingEpochAdvance()` は `EQProcessor` の **private**（`EQProcessor.h:553` 以降）であり、派生の `EQEditProcessor` からは呼べない。v1 のラッパ案はコンパイルできない | **新設メソッドを `EQProcessor` の public メンバにする**。`EQEditProcessor` は継承するだけ（ラッパ不要） |

## §1 改訂した設計

### 1.1 追加する API（`EQProcessor` の public）

```cpp
// ★ 追加（public）: NonRT 専用の reclaim ドライバ。既存の flush の意味は変えない。
void driveRetireReclaimNonRt() noexcept
{
    // (1) 退役があった場合: 既存 flush と同じ「publish + reclaim」
    if (convo::exchangeAtomic (m_epochAdvancePending, false, std::memory_order_acq_rel))
    {
        m_epochDomain->publishEpoch();
        m_ownedRetireRouter.tryReclaim();
        return;
    }

    // (2) 退役は無いが滞留が残っている場合: publish せず reclaim のみ
    if (m_ownedRetireRouter.pendingRetireCount() > 0
        || m_ownedRetireRouter.quarantineResidentCount() > 0
        || m_ownedRetireRouter.terminalReclaimResidentCount() > 0)
    {
        m_ownedRetireRouter.tryReclaim();
    }
}
```

- **既存の `flushPendingEpochAdvance()` は 1 文字も変更しない。** したがって `prepareToPlay` / `releaseResources` の挙動も不変。
- 5 行の重複（exchange + publish + reclaim）は意図的である。既存関数をヘルパ化すると「意味は同じでも実装が変わる」ため、v1 のリスクを避ける。

### 1.2 pending が無い場合の滞留判定

| 判定 | 意味 | コスト |
|---|---|---|
| `pendingRetireCount() > 0` | EQ ドメインの D キューに滞留 | atomic 1 回 |
| `quarantineResidentCount() > 0` | Q または E に滞留 | atomic 1 回 |
| `terminalReclaimResidentCount() > 0` | T に滞留 | atomic 1 回 |

いずれかが真のときだけ `tryReclaim()` を呼ぶ。3 つとも 0 なら何もしない（100 ms ごとに atomic 3 回のみ）。

### 1.3 呼出し側（`AudioEngine::timerCallback`）

```cpp
if (! isShutdownInProgress())
{
    jassert (! convo::numeric_policy::isAudioThread());
    uiEqEditor.driveRetireReclaimNonRt();   // EQProcessor の public メンバを継承
}
```

`EQEditProcessor` 側の追加は不要（`public` 継承）。v1 の「`EQEditProcessor` に 1 行ラッパ」は取り下げる。

### 1.4 §1.1〜§1.3 以外（タイマー周期・寿命・RT・再入・D→Q→E→T・案の比較・検証計画）

v1（`reclaim_driver_design.md`）の §1.3〜§3 を維持する。変更点は上記 2 点のみ。

### 1.5 実装上の必須条件（Owner 指定）への対応

| 条件 | 本設計 |
|---|---|
| pending のときだけ epoch を進める | (1) の exchange が真のときだけ `publishEpoch()` |
| `exchange(false)` の成功者だけが publish | 同上 |
| publish と reclaim の一体構造を維持 | (1) は既存 flush と同一順序・同一呼出し |
| Message Thread 以外から呼べない | `timerCallback` からのみ。`jassert(!isAudioThread())` |
| shutdown 中は呼ばない | `isShutdownInProgress()` ガード |
| 破棄前にタイマー停止 | `~AudioEngine` 本体の `stopTimer()`（`CtorDtor.cpp:167`）→ その後メンバ破棄 |
| 新しい reclaim authority を作らない | `m_ownedRetireRouter` の再利用のみ |

## §2 同一実行計測の最終仕様（実装後のゲート）

Owner 指定の 9 項目を、1 回の実行で取得する。

| # | 項目 | 取得元（設計） |
|---|---|---|
| 1 | EQ ドメインの開始・終了 `globalEpoch` と推移 | `EpochDomain` に `publishCount` と `firstPublishUs` / `lastPublishUs`、`final` / `postTeardown` ダンプ |
| 2 | `publishEpoch()` 呼出し回数（呼出し元・理由つき） | EQ ドメインの publish は**すべて** `flushPendingEpochAdvance` 系を通るため、`driveRetireReclaimNonRt` 内の 2 経路（pending 由来 / 新規 publish なし）と、既存 flush（prepare/release 由来）を**別カウンタ**にする |
| 3 | `tryReclaim` 呼出し回数と、走査・解放件数 | `reclaimCalls`（atomic）と、`DeferredDeletionQueue::reclaim` の戻り値 `reclaimed` の累計 |
| 4 | enqueue 時刻・対象 entry・退役時 epoch | 既存のレコーダ（`ts` / `epoch`）＋ entry の `firstScanTs` |
| 5 | `firstScanTs`、未走査のまま終了した件数 | entry に追加。`drainAll` で解放されたものは「未走査」として別計数 |
| 6 | reader の active 状態・epoch・登録/終了の変化 | `final` / `postTeardown` のレジストリダンプ（既存） |
| 7 | 4 分類の件数・sum・max | `retireLatencyClassified`（新設） |
| 8 | 分類不能件数・`consistencyErrors`・時刻順序違反 | 同上（理由別: `noScan` / `noSafe` / `inconsistent`） |
| 9 | 既存判定の不変 | `retireLatencyD1` / `evaluateRetireDelay` / 50 ms / N_req=59 のハッシュと verdict を記録 |

### 2.1 母数の分離（Owner 指摘）

分類対象と除外対象を**別の母数**として記録する。全体の分解式から黙って落とさない。

```json
"retireLatencyClassified": {
  "population": {
    "totalRequests": 1892,
    "classified": 1849,
    "excluded": { "noScan": 0, "noSafe": 43, "inconsistent": 0 }
  },
  "paths": [ ... ],
  "consistencyErrors": 0,
  "note": "excluded は従来の unobservedSafe 等に対応し、分類の母数からは外れるが総数には含む"
}
```

- `noSafe` は現行の `unobservedSafe`（`drainAll` の強制解放など）と同一定義。
- `noScan` は「一度も reclaim 走査されずに解放された」件数（新設）。
- `totalRequests = classified + excluded(全理由)` を常に満たすことを検証項目にする。

## §3 実装順序（Owner 推奨）

| 段階 | 作業 | ゲート |
|---|---|---|
| ① | reclaim ドライバ（§1） | ビルド成功＋競合・寿命テスト |
| ② | 競合・寿命テスト | タイマー中の破棄・shutdown 中の停止・他 publish 経路との競合 |
| ③ | `firstScanTs` と 3 分類 | 分解式の一致（classified + excluded = total） |
| ④ | D1 JSON 後方互換性 | 既存キーの不変・`evaluateRetireDelay` のバイト一致 |
| ⑤ | 同一実行の統合計測（§2） | 9 項目の同時取得 |
| ⑥ | Reader 共通レジストリ統合 | **今回の実装範囲に含めない**（次段階） |

## §4 状態

- 本改訂は**設計のみ**。製品コードは未変更（HEAD `c94d3767`、作業ツリーは設計文書のみ追加）。
- 実装着手の承認は未取得。段階①の承認後に着手する。
