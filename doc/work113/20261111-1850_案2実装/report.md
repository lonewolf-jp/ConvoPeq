# PublishReason 案2 の実装（引数化・デフォルト引数なし）

Owner 承認（2026-11-11「PublishReason 案2：実装着手を承認」「独立した実装単位として進行可能」）に基づく実装報告である。**domain tag には着手していない**（設計 v2.3 のレビュー待ち）。commit／merge／release も未実施。

## §0 要旨

- `flushPendingEpochAdvance()` を **`flushPendingEpochAdvance(PublishReason)`** に変更した（**デフォルト引数なし**）。
- 理由タグ `m_publishReasonTag`（メンバ）と `setPublishReasonForNextFlush()` を**削除**した（書き込みと消費が別操作である構造をなくした。残骸のメンバも残さない）。
- publish 条件・順序・reclaim 条件・戻り値の意味は**一切変更していない**。カウンタの増加は publish が成立したときだけ。
- ドライバ（タイマー）は従来どおり `TimerPending` を直接計数し、`flushPendingEpochAdvance()` を**呼ばない**（二重計数なし）。

## §1 実装

### 1.1 enum（公開・4 値）

```cpp
enum class PublishReason : std::uint8_t
{
    Prepare      = 0,   // prepareToPlay による呼出し
    Release      = 1,   // releaseResources による呼出し（終了処理を意味しない）
    TimerPending = 2,   // タイマードライバが pending を消費（flush は呼ばない）
    Explicit     = 3    // 上記以外の明示的な直接呼出し
};
```

- `PublishReason` は **public** に置いた（`RetireDriverStats` と同じ診断語彙であり、テストが `Explicit` を指定するため）。

### 1.2 呼出し箇所（本番 2 か所）

| 呼出し元 | 変更前 | 変更後 |
|---|---|---|
| `EQProcessor::prepareToPlay` | `setPublishReasonForNextFlush(Prepare); flushPendingEpochAdvance();` | `flushPendingEpochAdvance(PublishReason::Prepare);` |
| `EQProcessor::releaseResources` | 同上（Release） | `flushPendingEpochAdvance(PublishReason::Release);` |
| `EQProcessor::driveRetireReclaimNonRt` | `countPublishReason(TimerPending)`（flush を呼ばない） | **変更なし** |

- `flushPendingEpochAdvance()` の他の呼出しは存在しない（ソース全体で 2 か所のみ。テストは friend 経由）。

### 1.3 `flushPendingEpochAdvance` の本体

```cpp
void EQProcessor::flushPendingEpochAdvance(PublishReason reason) noexcept
{
    if (convo::exchangeAtomic(m_epochAdvancePending, false, std::memory_order_acq_rel))
    {
        m_epochDomain->publishEpoch();
        countPublishReason(reason);        // ← publish 成立時のみ
        m_ownedRetireRouter.tryReclaim();
    }
}
```

- 変更前は先頭でタグを `exchange` して理由を読んでいた（タグ書き込みと消費が別操作）。現在は**引数**であり、同時実行でも理由が入れ替わらない。

### 1.4 カウンタ

| フィールド | 旧 | 新 |
|---|---|---|
| タイマー | `publishTimerPending` | 同じ |
| prepare | `publishPrepare` | 同じ |
| release | `publishRelease` | 同じ |
| 未分類 | `publishOther` | **`publishExplicit`**（`m_publishOtherCount` → `m_publishExplicitCount`） |

- `retireDriverStats()` は C++ の診断 API であり、**JSON には出力していない**ため外部互換への影響はない。

## §2 Owner 実装条件への対応

| 条件 | 対応 | 証拠 |
|---|---|---|
| 本番の Prepare / Release 呼出しをすべて更新 | 2 か所とも引数化 | `EQProcessor.Core.cpp` の 2 行（`PublishReason::Prepare` / `::Release`） |
| テスト用ラッパーと直接 flush 呼出しも更新 | friend を `flushPendingEpochAdvance(eq, reason)` に変更し、4 か所で `Explicit` を明示 | `STG11EQRetireTests.cpp` |
| `Explicit` を指定した直接 flush のテスト | Stage-1 契約テストの項目 5 を「Explicit」に変更 | 実行結果 `explicit=1` |
| TimerPending の二重計数が無いこと | 新規チェック 5b: ドライバ 1 回で `publishTimerPending` だけが +1、`Explicit`/`Prepare`/`Release` は不変 | PASS |
| publish 成功時だけカウンタ加算 | `countPublishReason` は exchange 成功の内側にのみ存在 | コード上の位置 |
| 別スレッド Prepare/Release 相当を含む競合テスト維持 | Stage-1 の並行テスト（別スレッドから setter＋prepare/release、68,000 回超のドライバ）をそのまま実行 | `DRV6: ticks=… iterations=200` PASS |

## §3 検証

### 3.1 テスト（STG11EQRetireTests）

```
DRV6: ticks=67260 iterations=200 epoch=407
DRV: timer=402 prepare=2 release=1 explicit=1 reclaimOnly=720 idle=66144 offThread=1
CLS6: observations_with_firstScan=235 classified=235 compositionErrors=0
STG11EQRetireTests: PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b/Gate1/Stage1/Stage3)
```

- 4 区分すべてが実際の入口と一致（`prepare=2` は `prepareToPlay` 2 回、`release=1` は `releaseResources` 1 回、`explicit=1` はテストの直接 flush 1 回）。

### 3.2 ctest

```
```
100% tests passed out of 60
Total Test time (real) = 359.02 sec
ctest_exit=0
```

- 最終ツリー（タグメンバの残骸を削除した後）での実行。
  途中経過は 339.04 秒（タグメンバ削除前）・333.02 秒（コメント整形後）で、いずれも 60/60。
```

### 3.3 アプリ実測（3 シナリオ・12 秒）

| シナリオ | req | comp | unmatched | retire_delay | maxMs |
|---|---|---|---|---|---|
| burst40 | 1849 | 1849 | 0 | fail（閾値超過） | 2,165 |
| burstfast | 504 | 504 | 0 | fail | 2,161 |
| eq | 228 | 228 | 0 | fail | 12,098 |

- 段階③の実測（2,178 / 2,164 / 11,915 ms）と同水準であり、**挙動の変化はない**（差は実行ごとのばらつき）。
- `unmatched=0`・`pending=0`・`world: commits == destroys` も不変。

## §4 変更ファイル

| ファイル | 変更 |
|---|---|
| `src/eqprocessor/EQProcessor.h` | `PublishReason` を 4 値・`std::uint8_t` に変更し public へ移動／`flushPendingEpochAdvance(PublishReason)`（デフォルトなし）／タグと setter を削除／`publishExplicit` へ改名 |
| `src/eqprocessor/EQProcessor.Core.cpp` | flush 本体のタグ交換を削除し引数化／呼出し 2 か所を引数化／switch と stats の改名 |
| `src/tests/AudioEngineHarness/STG11EQRetireTests.cpp` | friend を理由付きに変更／4 か所で `Explicit` 明示／項目 5 を Explicit に／項目 5b（二重計数なし）を追加 |

差分: `patches/reason2_only/`（本単位のみ）と `patches/`（HEAD 基準の全体）。

## §5 未着手・注意

- domain tag は未着手（設計 v2.3 のレビュー待ち）。
- `PublishReason` を public にした理由は「テストが `Explicit` を指定する」ため。製品 API の意味は「publish 理由の語彙」であり、公開しても呼出し側の挙動は変わらない。
- `releaseResources` の `Release` は**終了処理を意味しない**（再構成でも同じ値）。この意味は enum のコメントに明記した。
- commit／merge／release は未実施。

## §6 成果物

- 本報告: `doc/work113/20261111-1850_案2実装/report.md`
- 差分: `patches/reason2_only/`（`01_eqprocessor_h.patch` / `02_driver_core.patch` / `03_tests.patch`）
- 証跡: `evidence/`（テストログ、3 シナリオの JSON、ctest ログ）
- 結合ソース: `ConvoPeq.md` と 2 MB 未満の分割
- Dropbox: `/ConvoPeq-reason2-20261111/`、`/ConvoPeq-source-20261111-reason2/`
