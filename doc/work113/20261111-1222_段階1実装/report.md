# 段階① reclaim ドライバの実装（設計 v3 準拠）

Owner 承認（2026-11-11「Gate ② 合格。段階①の実装着手を承認します」）に基づく実装報告である。対象は**設計 v3 に限定**し、段階③（`firstScanTs` と 3 分類）・Reader レジストリ統合には着手していない。commit／merge／release も未実施（HEAD `c94d3767`）。

## §0 実装したもの

| 項目 | 内容 |
|---|---|
| 新設メソッド | `EQProcessor::driveRetireReclaimNonRt()`（public・Non-RT・Message Thread 専用） |
| 呼出し元 | `AudioEngine::timerCallback()`（既存 `startTimer(100)`。新規 timer / thread は追加しない） |
| pending 時 | `m_epochAdvancePending` を exchange で消費 → `publishEpoch()` → `tryReclaim()`（publish 理由 = `timerPending`） |
| pending なし・滞留あり | `tryReclaim()` のみ（**epoch は前進させない**） |
| pending なし・滞留なし | 何もしない（`idleTicks` を数えるだけ） |
| 滞留判定 | `pendingRetireCount()`（D キュー）と `residentCountAtomic()`（Q+E+T）の atomic のみ。mutex を取る `residentCount` 系は不使用 |
| スレッド制約 | `jassert(isThisTheMessageThread())` ＋ **Message Thread 以外から呼ばれたら何もせず拒否**（`offThreadRejects`） |
| Shutdown | タイマー側で `! isShutdownInProgress()` を確認してから呼ぶ |
| publish 理由 | `timerPending` / `prepare` / `release` / `other` の 4 区分。未分類は `other` にのみ加算（既存区分へ暗黙加算しない） |
| 既存経路 | `prepareToPlay()` / `releaseResources()` の `flushPendingEpochAdvance()` は**挙動不変**（理由タグを立てるだけ） |

## §1 実装の詳細

### 1.1 ドライバ本体（`EQProcessor.Core.cpp`）

```cpp
void EQProcessor::driveRetireReclaimNonRt() noexcept
{
    auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
    const bool onMessageThread = (messageManager != nullptr)
                                 && messageManager->isThisTheMessageThread();
    jassert(onMessageThread);
    if (! onMessageThread) { /* offThreadRejects++ */ return; }

    if (convo::exchangeAtomic(m_epochAdvancePending, false, std::memory_order_acq_rel))
    {
        m_epochDomain->publishEpoch();
        countPublishReason(PublishReason::TimerPending);
        m_ownedRetireRouter.tryReclaim();
        return;
    }

    if (m_ownedRetireRouter.pendingRetireCount() > 0
        || m_ownedRetireRouter.residentCountAtomic() > 0)
    {
        /* reclaimOnlyRuns++ */
        m_ownedRetireRouter.tryReclaim();
        return;
    }

    /* idleTicks++ */
}
```

- `m_epochAdvancePending` の exchange は既存 `flushPendingEpochAdvance()` と同一の意味（acq_rel）であり、**同時に走っても pending を取り合って二重 publish にはならない**。
- 拒否経路は fail-safe である。Message Thread 以外から呼ばれても何も壊さず、滞留は次の正規の drain（タイマー／prepare／release）まで残るだけである。

### 1.2 タイマー配線（`AudioEngine.Timer.cpp`）

`timerCallback()` の冒頭（runtime world 解決直後）に追加した。

```cpp
if (! isShutdownInProgress())
{
    if (auto* eqDriverDsp = resolveActiveRuntimeDSPFromRuntimeWorldOnly(runtimeReadHandle))
        eqDriverDsp->eqRt().driveRetireReclaimNonRt();
}
```

- 周期は既存の `startTimer(100)`（`AudioEngine.Init.cpp:131`）をそのまま使う。**新しい timer も thread も作っていない**。
- 対象は published RuntimeWorld の current DSP。legacy `activeRuntimeDSPSlot`（placeholder 専用の非所有 mirror）は dereference しない（既存コードの禁止事項に従う）。
- `isShutdownInProgress()` で teardown 中の drain 契約に触れない。

### 1.3 publish 理由の計測

- `m_publishReasonTag`（atomic）を `prepareToPlay` / `releaseResources` が flush 直前に立てる。
- `flushPendingEpochAdvance()` は**必ずタグを消費**して `Other` に戻し、publish が成立したときだけ理由別カウンタを 1 つ進める。タグ無しの flush は `other` になる。
- ドライバ経由の publish は `TimerPending` として直接数える。

`retireDriverStats()`（診断・読み取り専用）で次を取得できる。

| フィールド | 意味 |
|---|---|
| `publishTimerPending` / `publishPrepare` / `publishRelease` / `publishOther` | 理由別 publish 回数（合計 = EQ ドメインの publish 総数） |
| `reclaimOnlyRuns` | pending なし・滞留ありで reclaim のみ実行した回数 |
| `idleTicks` | pending なし・滞留なし（何もしなかった回数） |
| `offThreadRejects` | Message Thread 以外から呼ばれて拒否した回数 |
| `currentEpoch` / `pendingRetire` / `resident` | 現在値（epoch・D 滞留・Q+E+T 滞留） |

## §2 検証（Owner 指定の 7 項目）

`src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` に `checkRetireDriverContract()` を追加した（既存 standalone target 内。CMake 変更なし）。

| # | 検証項目 | 方法 | 結果 |
|---|---|---|---|
| 1 | pending あり | setter → ドライバ | epoch が **+1**、`publishTimerPending` +1、D 滞留 0、`reclaimOnlyRuns` は増えない → **PASS** |
| 2 | pending なし・滞留あり | 現在 epoch より 1 古い epoch で検証用 entry を直接 enqueue → ドライバ | **epoch は前進せず**、`reclaimOnlyRuns` +1、deleter が実際に 1 回呼ばれ、D 滞留 0 → **PASS** |
| 3 | pending なし・滞留なし | ドライバ | `idleTicks` +1、`reclaimOnlyRuns` 不変、epoch 不変、publish なし → **PASS** |
| 4 | 100 ms 周期 | 既存 `startTimer(100)` を変更していないこと（差分）＋ アプリ実測でドライバ経由 publish が継続すること | 周期変更なし・新規 timer/thread なし → **PASS**（§3 の実測で裏付け） |
| 5 | publish 理由 4 区分 | prepare / release / タグ無し flush / ドライバ を個別に実行 | `prepare` +1、`release` +1、`other` +1、`timerPending` +1（実測 `timer=402 prepare=2 release=1 other=1`）→ **PASS** |
| 6 | 競合・寿命 | 別スレッドが setter＋prepare/release を 200 回、Message Thread がドライバを 68,702 回、同時進行 | drop 0、最終的に D/Q/E/T 滞留 0、epoch 407、非有限値なし・クラッシュなし → **PASS** |
| 7 | RT 制約 | 別スレッドからドライバを呼ぶ | 拒否（`offThreadRejects` +1）・epoch 不変・publish なし → **PASS**。オーディオ経路（`process()`）は段階①で**未変更** |

テスト実行結果（`build/Release/STG11EQRetireTests.exe`、終了コード 0）

```
DRV6: ticks=68702 iterations=200 epoch=407
DRV: timer=402 prepare=2 release=1 other=1 reclaimOnly=998 idle=67307 offThread=1
STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b/Gate1/Stage1)
```

- ctest: **60/60 合格（360.19 秒）**。
- 既存の Gate 1 検証（`checkGate1ProcessIsolation()`）も同時に PASS のまま。

## §3 アプリ実測（3 シナリオ・同一条件で前後比較）

`run_cases.ps1`（`--cli-run --verify-runtime-evidence-live`、12 秒）を段階①の前後で実行した。

| シナリオ | 指標 | 段階①前 | 段階①後 |
|---|---|---|---|
| burst40 | p90 | 327 ms | **131 ms** |
| | p99 | 4,421 ms | **136 ms** |
| | 最大 | 4,421 ms | 2,163 ms |
| | 200 ms 超の件数 | 120 | **1** |
| burstfast | p90 | 2,763 ms | **55 ms** |
| | p99 | 11,655 ms | **62 ms** |
| | 最大 | 11,655 ms | 2,166 ms |
| | 200 ms 超の件数 | 42 | **2** |
| eq | p90 | 13,989 ms | **913 ms** |
| | p99 | 13,989 ms | 11,815 ms |
| | 最大 | 13,989 ms | 11,815 ms |
| | 200 ms 超の件数 | 71 | **31** |

- D1 は全シナリオで `unmatched=0`・`pending=0`（不変）。`world: commits == destroys`、`overflow=0`。
- `retire_delay` の判定は **fail のまま**（`latency_exceeds_threshold`）。ただし分布は大きく改善した（p90/p99 が 1〜2 桁改善、200 ms 超がほぼ消滅）。

### 3.1 残る最大値の正体（設計上の帰結）

残っている最大値（2,163 / 2,166 / 11,815 ms）は**末尾の 1 件（eq は数件）**である。証跡の `observations` を見ると、いずれも「その時点の最新 epoch で retire され、**次に pending が立つまで epoch が進まない** entry」である。

- burst40: `epoch=90` の 1 件が 2,163 ms（次の setter が来るまで待ち）、その後は 199 ms 以下。
- burstfast: `epoch=14` の 1 件が 2,166 ms、その後は 204 ms 以下。
- eq: `epoch=1` の数件が 11.8 s（12 秒間に追加の setter が無いため、teardown の `releaseResources` まで publish されない）。

Owner 条件「滞留しているという理由だけで epoch を前進させない」を守っているため、**この末尾待ちは設計どおり**である。50 ms 閾値は性能目標であり保証ではない（Owner 決定）ため、`retire_delay` が fail のままであることをもって実装の失敗とはしない。

> 留意: D1 の `observations` は複数ドメイン（EQ ドメインとエンジン側 ISR ドメイン）の entry を同じリングに記録する。
> 実測では `epoch` が 1 のまま完了する entry と、`epoch` が 15 以上まで進んでいる entry が並存していた。
> 後者は EQ ドメインの publish（ドライバによる前進）を受けていることの証拠であり、
> 前者はエンジン側 ISR ドメインのように**本ドライバの対象外**の entry である公算が高い。
> どちらのドメインに属するかは `observations` にドメイン ID がないため本実験では断定できない。
> 帰属を定めるには `doc/work113/20261111-0600_reclaim設計v3/` の次段階（診断拡張）で domain タグを追加する必要がある。

一方で「pending が来れば 100 ms 以内に reclaim される」ことは、200 ms 超の件数が 120→1 / 42→2 に落ちたことで裏付けられる。

## §4 変更ファイルと差分

| ファイル | 変更 |
|---|---|
| `src/eqprocessor/EQProcessor.h` | `driveRetireReclaimNonRt()`・`RetireDriverStats` / `retireDriverStats()`・publish 理由 enum とタグ・理由別カウンタ 7 個 |
| `src/eqprocessor/EQProcessor.Core.cpp` | ドライバ本体・`countPublishReason()`・`retireDriverStats()`・`flushPendingEpochAdvance()` の理由タグ消費・prepare/release のタグ設定 |
| `src/audioengine/AudioEngine.Timer.cpp` | `timerCallback()` からドライバを呼ぶ（`! isShutdownInProgress()` ガード付き） |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | `checkRetireDriverContract()`（7 項目）＋テスト用アクセス拡張 |

差分は 2 系統を用意した。

- `patches/stage1_only/`—**段階①単独の差分**（段階①着手前のツリーを復元して比較）
  - `01_eqprocessor_h.patch`（+52 行）
  - `02_tests.patch`（+198 行）
  - `03_driver_core.patch`（3 ハンク：flush の理由タグ化と新設 3 関数）
  - `04_timer_wiring.patch`（`timerCallback` からの呼出し 13 行）
- `patches/`—HEAD `c94d3767` 基準の全体差分（Gate 1 と他の未コミット作業を含む）。段階①の影響を見るには `stage1_only/` を使うこと。

## §5 不変条件の確認（Owner 指定）

| 制約 | 確認方法 | 結果 |
|---|---|---|
| 100 ms 周期の維持 | `AudioEngine.Init.cpp` の `startTimer(100)` を変更していない（差分） | 維持 |
| pending 時のみ epoch 前進 | テスト #1〜#3 と実装の exchange 1 か所 | 維持 |
| lock-free 滞留判定 | `pendingRetireCount()` / `residentCountAtomic()` のみ使用（`grep` で `residentCount()` 不使用を確認） | 維持 |
| Audio Thread で実行しない | `jassert(isThisTheMessageThread())` ＋ 拒否経路（テスト #7） | 維持 |
| Shutdown 中は呼ばない | `! isShutdownInProgress()` ガード（Timer.cpp） | 維持 |
| 既存 flush の挙動不変 | タグ消費を追加しただけ（publish 条件・順序・tryReclaim は同一） | 維持 |
| D→Q→E→T の reclaim 順序 | `tryReclaim()` をそのまま呼ぶ（ルータ内部は未変更） | 維持 |
| Reader の epoch 保護 | `publishEpoch()` の呼出し位置のみ追加。reader 側は未変更 | 維持 |
| Shutdown 時の drain 契約 | `releaseResources` / 破棄経路は未変更 | 維持 |
| RT 経路の制約 | `process()` は段階①で未変更（Gate 1 の変更のみ） | 維持 |

## §6 未着手・制約

- 段階③（`firstScanTs` と 3 分類）・Reader レジストリ統合は未承認・未着手。
- ASan ビルドでの実行は未実施（段階①でも同様）。
- `retire_delay` は fail のまま。末尾待ちは設計上の帰結であり、閾値 50 ms を満たすには「pending 以外でも epoch を進める」等の設計変更が必要になる（今回は Owner 条件により不可）。
- エンジン側 ISR ドメインの末尾 entry（`epoch` がそのドメインで進まないもの）は本ドライバの対象外である。

## §7 成果物

- 本報告: `doc/work113/20261111-1222_段階1実装/report.md`
- 差分: `patches/`（HEAD 基準）と `patches/stage1_only/`（段階①単独）
- 証跡: `evidence/`（driver 契約テストのログ、アプリ 3 シナリオの JSON と前後比較表、ctest ログ）
- Dropbox: `/ConvoPeq-stage1-20261111/`
