# C3-D 本体（`waitForDrain` 内の epoch 前進）の実装と計測

- 作業日時: 2026-11-13 22:51 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋未コミットの診断ログ
- 種別: **Owner 承認済み範囲（epoch 前進本体）のみ**。C9-A と改名は別報告。
- 実装条件（Owner 指示 1〜6）との対応を §1 に示す。

---

## 1. 実装（承認条件との対応）

| # | 承認条件 | 実装 |
| --- | --- | --- |
| 1 | `waitForDrain` の待機ループ内で epoch を前進 | `AudioEngine.Threading.cpp:326-358` に前進ブロックを追加 |
| 2 | 前進の後に `drainDeferredRetireQueues(true)` | 同 `:361`（前進ブロックの直後） |
| 3 | 順序は必ず「前進 → drain」 | ループ本体の先頭が前進、その後に drain。逆順の経路は無い |
| 4 | `drainDeferredRetireQueues` の内部には置かない | 同関数（`AudioEngine.Retire.cpp:58`）は無変更 |
| 5 | shutdown 待機経路に限定 | `waitForDrain` の呼出し元は `ReleaseResources.cpp:255`（100ms）と `:673`（2000ms）の 2 箇所のみ。通常動作からの呼出しは無い |
| 6 | 既存のタイムアウト・終了条件・エラー処理を維持 | タイムアウト判定（`elapsedMs >= boundedTimeoutMs`）と `return false`、`jassert`、ポーリング間隔は無変更。2,000 ms も不変 |

### 1.1 前進条件（設計 §2.3 / §2.4）

```cpp
while (!isFullyDrained())
{
    // (1) epoch 前進（同一 identity につき 1 回だけ）
    const auto head = m_epochDomain.inspectRetireHead();
    if (head.valid && !head.epochOlderThanMinReader)
    {
        const bool sameIdentity = advancedOnce
            && head.headSlotIndex == advancedSlot
            && head.headSequence == advancedSequence
            && head.headEpoch == advancedEpoch;
        if (!sameIdentity)
        {
            (void) advanceRetireEpoch();
            advancedSlot = head.headSlotIndex;
            advancedSequence = head.headSequence;
            advancedEpoch = head.headEpoch;
            advancedOnce = true;
        }
    }
    // (2) drain（必ず前進の後）
    drainDeferredRetireQueues(true);
    ...
}
```

- identity は `(headSlotIndex, headSequence, headEpoch)` の 3 要素（C3-D API 拡張で取得）。
- 同一 identity への前進は最大 1 回。前進後は `epochOlderThanMinReader` が真になるため、
  同じ identity では条件自体が成立しない（二重の歯止め）。
- `advanceRetireEpoch()` は `m_retireRouter->publishEpoch()` → `m_epochDomain.publishEpoch()`。
  `minReaderEpoch` は変えないため、Reader の安全条件（`retireEpoch < minReaderEpoch`）は不変
  （INV-C3-3 / INV-C3-7）。
- 先頭が空（`valid=false`）のときは前進しない。

## 2. 計測（C3-D 単独・3 回）

C3-D だけを入れてビルドし、`--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` を 3 回実行した。

| 実行 | exit | 総経過 | `failed` | `c3_retireDepth` | `head` | `minReaderEpoch` / `currentEpoch` | `sh3_pendingRetire` | `reclaimSuccess` |
| --- | ---: | ---: | --- | ---: | --- | --- | ---: | ---: |
| 1 | 0 | 6.56 s | `c9` | **0** | unavailable | 13 / 13 | **1** | 7 |
| 2 | 0 | 6.28 s | `c9` | **0** | unavailable | 13 / 13 | **1** | 7 |
| 3 | 0 | 6.23 s | `c9` | **0** | unavailable | 13 / 13 | **1** | 7 |

読み取れること:

- **c3 が解消した。** 変更前は `c3_retireDepth=1` で `head=present`（epoch 12 のまま 2 秒停滞）だったが、
  変更後は `c3=0` / `head=unavailable`（先頭が回収済み）。
- **epoch が前進した。** `currentEpoch` が 12 → 13 になり、`minReaderEpoch` も 13。
- **退役キューが進んだ。** `reclaimSuccess` が 6 → 7 に増え、`sh3_pendingRetire` が 2 → 1。
- **タイムアウトは残る。** `failed=c9`（`retireBacklogCount_` = 164）が残るため、`isFullyDrained()` は
  依然として偽で、2 秒タイムアウトは 3/3 で発生した。**C3-D 単独では 2 秒は解消しない**（設計どおり）。

## 3. 検証

| # | 検証 | 結果 |
| --- | --- | --- |
| 1 | 処理順序（前進 → drain） | コードで確認（ループ先頭が前進、`:361` が drain） |
| 2 | epoch 前進による drain の進行 | `c3=0`・`head=unavailable`・`reclaimSuccess` 6→7・`sh3` 2→1 |
| 3 | 既存 drain 判定・shutdown 経路の回帰 | `ctest` 61/61（§4） |
| 4 | ビルド | C3-D 単独はアプリ 3 ターゲット、C9-A 込みの全体ビルドは 17 ターゲット。いずれもエラー 0 |
| 5 | 2 秒タイムアウトの解消 | **未解消**（C9-A 実装後に再測定。別報告） |

## 4. 回帰

`ctest --test-dir build -C Release` → 61/61 成功。詳細は C9-A の報告に併記（同一ビルドで実施）。

## 5. 残留リスク

- 前進は「先頭が epoch 条件で止まっている」ときだけ 1 回。先頭が変われば新しい identity で 1 回。
  待機中に enqueue が続く場合は identity の数だけ前進し得るが、実測では 1 回で収束した。
- `inspectRetireHead()` が `getMinReaderEpoch()`（128 slot 走査）を 2 ms ごとに 1 回追加する。
  drain 側でも同じ走査を行うため、待機中の走査回数は約 2 倍になる。実測の総経過に悪化は無い
  （6.23〜6.56 s、変更前 6.23〜6.81 s と同じ範囲）。
- epoch 前進は `epochGeneration_` も進める。本番の quiescence proof 生成は 2 秒待ちより前の 2 箇所だけなので
  identity は壊れない（設計 INV-C3-6）。将来 proof を待機後に作る変更を入れる場合は再検討が要る。

## 6. ロールバック

`AudioEngine.Threading.cpp` の前進ブロック（`:326-358`）を削除し、ループを
`drainDeferredRetireQueues(true)` だけに戻す。`drainDeferredRetireQueues` と `isFullyDrained` は
無変更なので、他への波及はない。

## 7. 判定

- 承認範囲（epoch 前進本体）は**実装完了・検証完了**。
- 2 秒タイムアウトは C3-D 単独では解消しない（c9 が残るため）。C9-A との統合結果は別報告。
