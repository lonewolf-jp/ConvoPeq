# c3 / c9 原因修正の設計検討（追加調査の結果を反映）

- 作業日時: 2026-11-13 20:31 起票 / 21:47 最終改訂
- 改訂内容: 追加調査、C3-D の十分性判断、指摘の反映（identity の取得要件、期待値の適用範囲、
  Faulted 側のテスト、別課題への分離）、前進処理の配置確定
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋
  未コミットの診断ログ（暫定保持）
- 種別: 設計検討と実装状況の記録。設計判断は 2026-11-13 21:47 時点。API 拡張の実装は
  別報告（`doc/work113/20261113-2215_C3D実装/`）に記載し、本書の設計作業自体では
  ソース変更・ビルド・テストを行っていない。
- 状態（2026-11-13 22:15 更新）:
  - 設計: 完了。
  - C3-D のうち **head identity 取得 API 拡張: 実装・検証完了**（報告は
    `doc/work113/20261113-2215_C3D実装/`）。
  - **C3-D の本体（`waitForDrain` 内の epoch 前進）: 未実装**（実装 GO 待ち）。
  - C9-A: 未実装（実装 GO 待ち）。pressure カウンタ改名は別課題。
  - 2 秒 drain タイムアウト: **未解消**（本体の実装後に再検証する）。
- 参照ソース: `doc/ConvoPeqMD/20261113221637/ConvoPeq.md`（生成 22:16:37、API 拡張の実装後。
  `--verify-snapshot` で `content == current: OK`）。2026-10-10 版とは別物。この検証は本書作成者が
  実行したもので、独立再実行ではない。
- 配置: `waitForDrain` のループ内で前進し、その後に `drainDeferredRetireQueues(true)` を実行する（§2.7）。
- 前提: A/C 計測（`doc/work113/20261113-2021_AC計測結果/`）で c3 と c9 は独立した原因と確定した。
- 判断: C3-D と C9-A を採用。実装 GO は未発行。
- 固定事項: drain timeout は **2,000 ms のまま**（`waitForDrain(2000, 2)`）。診断ログは**暫定保持**。

---

## 0. 追加調査の結論（先に要点）

Owner が求めた 4 点は、**ソース変更なしで既存ログとコードから確定できた**。

| 問い | 結論 |
| --- | --- |
| `drainPendingRetireIntentsForShutdown()` 由来か | **違う**。この関数は `ReleaseResources.cpp:680` で、診断ログ（`:728`）より**前**に走る。診断時点で既に c3=1 であり、+1 はその後 |
| OverflowRing 再注入由来か | **違う**。再注入は 5 秒 graceful drain のループ（`:293-303`）とそのタイムアウト分岐（`:327-335`）だけ。今回のログに `graceful drain timeout` が無く、ループは**即座に break** している（＝再注入は 0 回） |
| enqueue 時の epoch | **前進後の新しい epoch**（12 → 13） |
| enqueue は前進より前か後か | **後**。`publishEpoch()` の直後に enqueue する |

**+1 の正体**: `m_coordinator.finalizeShutdown(timedOut)`（`ReleaseResources.cpp:812`）→
`SnapshotCoordinator::retireCurrentAndTarget()`（`SnapshotCoordinator.h:179-203`）が
`publishEpoch()` で epoch を 1 進めてから、`GlobalSnapshot` を 1 件 `enqueueWithRetry` する。
これが `sh3_pendingRetire=2` の 2 件目である。
つまり enqueue は epoch 前進の**後**に行われ、entry は前進後の epoch を持つ。

補強材料:

- OwnerChannel 残存の drain（`:826-842`）は `drainedResidual=0` だった
  （`[AUDIT] drainAllNonRt residual:` の行が 3 回とも無い）。
- `drainDeferredRetireQueues(true)`（`:808`）の内部は `tryReclaim` / `reclaim` /
  `retryDeferredDestroys` / `retryDeferredQuarantineDestroy` / 保留 reclaim 再試行のみで、
  いずれも enqueue しない（`reclaimNormal` は state 遷移と reclaim だけ）。
- `m_coordinator.reclaim()` は `SnapshotCoordinator::reclaim`（ヘッダ `:116-118`）で、
  `tryReclaim()` を呼ぶだけ。

## 1. 実測の再掲（設計の前提）

| 観測 | 値 | 意味 |
| --- | --- | --- |
| `currentEpoch` | 12（2 秒間不変） | 2 秒窓では epoch が前進しない |
| 先頭 entry の `epoch` | 12 | 積まれたときの epoch のまま |
| `minReaderEpoch` | 12 | 有効 Reader 0 なので現在 epoch と一致 |
| `retireEpochLtMinReader` | 0 | `isOlder(12, 12)` が偽 → `reclaim()` が先頭で break |
| `headScanAgeMs` | 2001 | 最初の走査から一度も進んでいない |
| `retireBacklogCount_` | 143（3 回同値） | EQ 退役の回数。減算されない |
| c9 の他 13 条件 | すべて 0 | キュー・residency・recovery は空 |

## 2. c3: epoch 前進の設計

### 2.1 現状の時系列（追加調査で確定）

```
StopAcceptingWork → StopAudio → StopWorkers → ObserverDrained
  ForceEpochAdvance : advanceRetireEpoch() 1 回 → epoch 12
  5 秒 graceful drain（:287-313）
      → キューが空だったため即 break（epoch 前進 0 回。timeout ログ無し）
  closeReaderRegistration（:276）
  worldClear（:579-604）→ World を epoch 12 で enqueue
  2 秒 waitForDrain（:673）: drainDeferredRetireQueues のみ。epoch は 12 のまま
      → 先頭 entry（epoch 12）が isOlder(12,12)=偽 で回収不能。タイムアウト
  finalizeShutdown（:812）→ publishEpoch() で 13 → snapshot を epoch 13 で enqueue（+1）
  （以降）~AudioEngine の 5 秒 drain が publishEpoch + tryReclaim を 10 ms ごとに実行
```

**重要な発見**: 5 秒 graceful drain のループ（`:307-308`）は既に
`publishEpoch()` + `tryReclaim()` を 10 ms ごとに実行している。今回の実行ではキューが空だったため
一度も回らなかった。つまり「前進させながら reclaim する」パターンは同じ関数内に既にあり、
**2 秒待ちのループにだけ無い**。

また `~AudioEngine` の drain（`AudioEngine.CtorDtor.cpp:278`, `:299`）も同じパターンである。
3 箇所のうち 1 箇所（2 秒待ち）だけ前進しない、というのが c3 の実体である。

### 2.2 設計案（C3-D 採用済み）

| 案 | 内容 | 前進回数 | 長所 | 短所 |
| --- | --- | --- | --- | --- |
| C3-A | `waitForDrain` の直前に 1 回追加 | 1 回 | 変更が最小 | 待機中に積まれる entry は回収不能 |
| C3-B | 待機ループで N 回ごとに前進（上限 K） | 最大 K | 待機中の enqueue にも追随 | K の根拠が経験則 |
| C3-C | 待機ループで毎回前進 | 最大 1000 | 5 秒ループと同じ挙動 | epoch が 2 秒で 1000 進む |
| **C3-D** | **先頭 entry の identity 単位で、同一 identity につき最大 1 回前進** | 頭詰まり 1 件につき 1 回 | 有界。詰まりの実体にだけ反応する | identity 判定が必要 |
| C3-E | 前進せず 5 秒 drain に委ねる（現状） | 0 | 変更なし | 2 秒待ちが常に無駄 |

### 2.3 C3-D の判定条件（確定案）

**identity の定義**: 先頭 entry の「epoch + type + domainId」の組では同一性を保証できない
（同じ epoch で同じ型の entry が複数あり得る）。一方、`peekHead` は slot 位置と sequence を
観測できる。したがって identity は次の組で定義する。

```
identity = (headSlotIndex, headSequence, headEpoch)
```

- `headSlotIndex = dequeuePos & (kQueueSize - 1)`
- `headSequence` = コピー前に読んだ `sequences[slot]`（`pos + 1`）
- `headEpoch` = entry の epoch

同じ slot が再利用されると sequence が `pos + 1 + kQueueSize` に変わるため、
slot と sequence の組で「同じ物理 entry か」を判定できる。epoch を加えるのは、
同一 entry の epoch が書き換わらないことの確認（診断上の安全弁）である。

**identity の取得要件（実装条件・Owner 指摘）**: 現行の `RetireHeadSnapshot` は `valid` /
`headEpoch` / `minReaderEpoch` / `currentEpoch` / `type` / `domainId` / `firstScanAgeMs` /
`scanned` / `epochOlderThanMinReader` のみを返し、**slot と sequence を外部に出していない**。
実装では `peekHead` を次の形に拡張し、identity を明示的に取得できるようにする。

```cpp
// DeferredDeletionQueue（呼出し元は inspectRetireHead の 1 箇所のみ）
[[nodiscard]] bool peekHead(DeletionEntry& out,
                            std::uint32_t& slotIndex,
                            std::uint32_t& sequence) const noexcept;

// EpochDomain::RetireHeadSnapshot に追加
std::uint32_t headSlotIndex{0};
std::uint32_t headSequence{0};
```

- **identity の比較を epoch / type / domain の比較で代用してはいけない**（同じ epoch に同型の
  entry が複数あり得るため）。これは Owner 条件として固定する。
- 前進の記録は `(headSlotIndex, headSequence, headEpoch)` の 3 つを保持し、3 つすべてが一致した
  ときだけ「同一 identity」と判定する。
- `peekHead` の既存の整合性確認（コピー前後の sequence と dequeuePos）はそのまま維持する。

**前進条件**:

1. 先頭 entry が存在する（`peekHead` 成功）。
2. `isOlder(headEpoch, minReaderEpoch)` が偽（＝epoch 条件で止まっている）。
3. その identity に対してまだ前進していない（直前回の identity と異なる、または未前進）。
4. `closeReaderRegistration()` 済み（2 秒待ちの時点で成立済み）。
5. 前進は 1 回だけ行い、その identity を記録する。前進後も同じ identity が残る場合は
   **再度前進しない**（INV-C3-3/4）。

**前進後の確認**: 前進しても同じ identity が残る場合、epoch 以外の理由で止まっている可能性がある
（例: `minReaderEpoch` が上がらない）。その場合は前進をやめ、`pendingRetireCount()` の減少を待つ。

### 2.4 不変条件（Owner 提示 6 項の具体化）

| # | 不変条件 | 実装上の担保 |
| --- | --- | --- |
| INV-C3-1 | 前進判定は先頭 entry の一意な identity に結び付ける | `(slot, sequence, epoch)` を記録して比較 |
| INV-C3-2 | 同一 identity に対する前進は最大 1 回 | 記録した identity と一致したら前進しない |
| INV-C3-3 | 前進後も reclaim 条件は `retireEpoch < minReaderEpoch` を維持 | 前進は `minReaderEpoch` を上げない。判定式は不変 |
| INV-C3-4 | 前進しても先頭が残る場合、同一 identity に繰り返し前進しない | INV-C3-2 と同じ記録で担保 |
| INV-C3-5 | 新しい先頭に変わった場合は、その identity について改めて判定する | identity 不一致でリセット |
| INV-C3-6 | `epochGeneration` と quiescence proof の順序制約を維持 | 本番の proof 生成は `releaseResources:536` / `:550` の 2 箇所のみで、いずれも 2 秒待ちより前。待機中の前進は proof を壊さない。将来 proof を待機後に作る場合は前進を禁止する |
| **INV-C3-7** | **epoch 前進は Reader の quiescence proof を代替しない** | 前進は `minReaderEpoch` を変えない。Reader が居れば entry は回収されない。proof の成立条件（reader 0・登録封鎖・epoch settled）はそのまま必要 |

### 2.5 追加調査を踏まえた実装条件の確定

- 2 秒待ちのループに前進を入れる。位置は **`waitForDrain` のループ内、`drainDeferredRetireQueues(true)`
  の直前**とする（Owner 指定 2026-11-13。理由は §2.7）。
- 5 秒 graceful drain のループ（`:307-308`）は毎回前進（C3-C 相当）であり、C3-D は
  「同一 identity につき 1 回」なので、5 秒ループの実装をそのままコピーしない。
- `finalizeShutdown` の +1（epoch 13 の snapshot）は**2 秒待ちの外**で起きるため、C3-D の対象外。
  この entry は `~AudioEngine` の 5 秒 drain が回収する。**C3-D で sh3_pendingRetire が 0 になることは
  期待しない**（1 にはなる想定）。検証の期待値は「2 秒待ちがタイムアウトしないこと」と
  「`retireEpochLtMinReader` が 1 になること」に置く。

### 2.6 C3-D の前進条件が十分かの最終判断

**判断: 十分。** 根拠は 3 点。

1. 2 秒待ちの窓で先頭 entry は 1 件のまま動かなかった。`headScanAgeMs=2001` は「最初の走査から
   2,001 ms 経過」を意味し、`firstScanTsNs` は enqueue で 0 に戻り（`DeferredDeletionQueue.h:135`）、
   consumer が最初に評価した 1 回だけ書かれる（同 `:170-171`）。先頭が入れ替われば新しい entry は
   0 から再記録されるため、2001 ms という値は**同一 identity が維持された**ことの証拠になる。
2. c3（`pendingRetireCount()`）は 2 秒間 1 のままだった。窓の中で新しい enqueue は無い。
3. したがって前進は 1 回で足りる。仮に待機中に新しい entry が積まれても、INV-C3-5 により
   新しい identity で 1 回だけ前進するため、上限は「窓内に現れた identity の数」で有界になる。

再判断が必要になる条件:

- 待機ループの中で retire を発行する変更を入れる場合（新しい identity が窓内で増える）。
- 実機の高負荷時に先頭が複数回入れ替わる場合（今回の CLI 実行は 1 件で安定。負荷条件は未確認）。

**`sh3_pendingRetire=1` の適用範囲（Owner 指摘）**: この 1 は**今回の再現条件に対する期待値**であり、
一般的不変条件ではない。`retireCurrentAndTarget()` は current と target の両方を処理するため、
どちらが有効かで enqueue 数は変わる。実装後の検証では「1 であること」ではなく
「2 秒待ちの成功」と「その後の Snapshot 退役の内訳」を分けて記録する。

**`retireEpochLtMinReader=1` の評価タイミング（Owner 指摘）**: この値は C3-D の前進・回収を評価する
時点（2 秒待ちの窓）で確認する。`finalizeShutdown()` 後の新しい epoch の entry まで同じ条件で
評価してはならない（別の entry・別の epoch である）。

### 2.7 前進処理の配置（Owner 指定）

**指定: `waitForDrain` のループ内で epoch を前進させ、その後に `drainDeferredRetireQueues(true)` を
実行する。**

| 観点 | `waitForDrain` 内 | `drainDeferredRetireQueues` 内 |
| --- | --- | --- |
| 通常動作への影響 | 避けられる | shutdown 条件の分岐が必要 |
| shutdown 専用性 | 関数の責務で明確 | `allowDuringShutdown` に依存 |
| 既存 drain 呼出しへの影響 | 待機ループに限定 | shutdown 中の他の呼出しにも波及 |
| 責務の分離 | 待機・前進・再試行を集約 | drain と epoch 前進が結合 |

根拠として、`drainDeferredRetireQueues(true)` の本番呼出しは 3 箇所ある。

| 呼出し元 | 用途 |
| --- | --- |
| `AudioEngine.Threading.cpp:327` | `waitForDrain` のループ（本件の対象） |
| `AudioEngine.Processing.ReleaseResources.cpp:377` | `ReclaimComplete` 前の最終回収 |
| `AudioEngine.CtorDtor.cpp:320` | `~AudioEngine` の world clear 後 |

drain 関数の中に前進を入れると、この 3 経路すべての意味が変わる。`waitForDrain` に閉じ込めれば
待機サイクルだけに限定できる。通常動作側の `processDeferredReleases()` は
`drainDeferredRetireQueues(false)` を呼ぶため、影響を受けない。

## 3. c9: retire backlog 計数の設計

### 3.1 現状の計数責任

| 段階 | 実装 | 計数 |
| --- | --- | --- |
| enqueue 受付 | `RuntimeIntentCoordinator::enqueueRetire`（`:149-182`）→ `onRetireAccepted()`（`:198`） | **+1** |
| 消費 | `onRetireConsumed()`（`:206`） | **呼出し元 0**（`src` 全体を検索） |
| 絶対値設定 | `setRetireBacklogCount()`（`:274`） | TEST-ONLY |

呼出し元は `EQProcessor.Core.cpp:60`（`m_retireCoordinator->enqueueRetire`）。EQ の
`m_retireCoordinator` は engine の `runtimePublicationBridge_` そのもの
（`AudioEngine.CtorDtor.cpp:73`、`DSPCoreLifecycle.cpp:113`）。

### 3.2 影響範囲（drain 述語だけではない）

| 利用箇所 | 影響 |
| --- | --- |
| `AudioEngine::isFullyDrained()`（c9） | 恒久的に偽 → `waitForDrain(2000, 2)` が常にタイムアウト |
| `ShutdownScheduler::markShutdownComplete()`（`:602-614`） | `isFullyDrained()` が偽なので**毎回 Faulted へ遷移** |
| `noteRetireBacklogChanged()`（`:242-271`） | pressure の slope 検出に使用 |

**実測で確認できた副作用**: 3 回とも `~AudioEngine` が
`[FAULT] ~AudioEngine: coordinator in Faulted state after markShutdownComplete — residual intents may remain in System 1 queues`
を出している。C9-A の検証では、この行が消えることを確認項目にする。

### 3.3 C9-A の変更範囲（確定案）

`ShutdownScheduler::isFullyDrained()` から `retireBacklogCount_ == 0` の 1 条件だけを外す。
残す 13 条件は次のとおり。

| # | 条件（式そのまま） |
| --- | --- |
| 1 | `consumeAtomic(swapPending_) != 0` → 即 false |
| 2 | `intentQueue_.sizeApprox() == 0` |
| 3 | `observeDeferredRing_.size() == 0` |
| 4 | `quarantineFallbackQueue_.sizeApprox() == 0` |
| 5 | `recoveryIntentQueue_.size() == 0` |
| 6 | `publicationBacklogCount_ == 0` |
| 7 | `publicationIntentResidencyCount_ == 0` |
| 8 | `pendingIntentCount_ == 0` |
| 9 | `reclaimInFlightCount_ == 0` |
| 10 | `quarantineIntentResidencyCount_ == 0` |
| 11 | `quarantineRingResidencyCount_ == 0` |
| 12 | `recoveryAdmissionPending_ == false` |
| 13 | `liveLogicalRecoveryObligationCount() == 0` |

変更しないもの:

- `AudioEngine::isFullyDrained()` の `pendingRetireCount() == 0`（Layer 1 の実測判定）。
- `retireBacklogCount_` の増減処理（`onRetireAccepted` / `onRetireConsumed` の呼出し関係）。
- `markShutdownComplete()` の実装（遷移そのものは変えず、結果だけ検証する）。

**意味の確認**: この変更は「retire 資源が残っていても Engine 側が drain 完了と判定する」ことを
意味しない。Engine 側は `pendingRetireCount()` を実測しており、retire entry が残れば
`AudioEngine::isFullyDrained()` は依然として偽になる。Coordinator 側の重複判定を外すだけである。

### 3.4 `markShutdownComplete()` の状態遷移（検証対象）

| 遷移前 | `isFullyDrained()` | 遷移後 | 現状 | C9-A 後（期待） |
| --- | --- | --- | --- | --- |
| `ShuttingDown` | true | `Bootstrapping` | 到達しない | 到達する |
| `ShuttingDown` | false | `Faulted` | 毎回こちら | 他 13 条件が 0 なら到達しない |

検証は次の 3 点で行う。

1. `~AudioEngine` の `[FAULT] ... coordinator in Faulted state` が出ないこと。
2. `retireBacklogCount_` が 143 のままでも `isFullyDrained()` が真になること
   （他 13 条件が 0 のとき）。
3. **Faulted 側もテストで固定する**（Owner 指摘）。13 条件のいずれか 1 つを非 0 にして
   `ShuttingDown → Faulted` になることを確認する。正常系だけを固定すると、述語の変更で
   異常系が通らなくなっても検出できない。

### 3.5 不変条件（c9）

- **INV-C9-1**: drain 述語に含める計数は、必ず対になる減算経路を持つこと。
- **INV-C9-2**: 減算を導入する場合、`accepted - consumed == 生存中の retire 義務数` を
  全経路（D/Q/E/T）で満たすこと。満たせないなら減算を導入しない。
- **INV-C9-3**: 減算は underflow しない（既存ガード `:208` を維持）。
- **INV-C9-4**: `markShutdownComplete()` の状態遷移をテストで固定すること
  （`Bootstrapping` と `Faulted` の両方）。

## 4. `retireBacklogCount_` の意味（文書化）

Owner 判断により、pressure 用途は当面保持し、drain 判定には使わない。

| 用途 | 扱い | 理由 |
| --- | --- | --- |
| drain 完了判定 | **使用しない**（C9-A で除去） | 減算経路が無く、実 backlog を表さない |
| 実 retire backlog の表示 | **使用しない** | 同上。実 backlog は `pendingRetireCount()`（Layer 1） |
| pressure slope 検出 | 暫定保持 | 増加量（slope）だけを見る用途は成立する |
| 累積 accepted 数 | 保持可能 | 単調増加は事実として正しい |

`retireBacklogCount_` という名前は残存件数を連想させるが、実体は累積 accepted 数である。
改名（例: `retireAcceptedCount_`）は今回の変更に混在させず、別課題として扱う。
当面はコード内コメントで意味を明示する。

## 5. c3 と c9 を混ぜない理由

| 観点 | c3 | c9 |
| --- | --- | --- |
| 問題の種類 | 回収の**タイミング**（epoch が進まない） | 計数の**整合性**（減算が無い） |
| 触る場所 | 2 秒待ちのループ | Coordinator の drain 述語 |
| 失敗の現れ方 | entry が残る（メモリは安全側） | 述語が恒久的に偽（状態遷移が Faulted） |
| 検証方法 | `retireEpochLtMinReader=1`、タイムアウト行が消える | `c9` が `failed` から消え、`[FAULT]` が消える |

## 6. 実装後の受け入れ条件（Owner 提示の 3 群）

### 6.1 C3-D

| # | 検証 | 期待 |
| --- | --- | --- |
| 1 | 2 秒 drain timeout | 発生しない（`drain timeout reached` の行が消える） |
| 2 | 対象 entry の epoch 条件 | `retireEpochLtMinReader=1`。**2 秒待ちの窓で評価する**（`finalizeShutdown` 後の entry とは別） |
| 3 | 同一 identity への前進回数 | 最大 1 回（`(slot, sequence, epoch)` の記録で固定。テストまたはログで確認） |

### 6.2 C9-A

| # | 検証 | 期待 |
| --- | --- | --- |
| 4 | `retireBacklogCount_` が非 0 でも他 13 条件が成立 | Coordinator が drained と判定する |
| 5 | `ShuttingDown → Bootstrapping` | 遷移する |
| 6 | 未 drained 条件がある場合 | `ShuttingDown → Faulted`（**Faulted 側もテストで固定**） |
| 7 | `[FAULT] ~AudioEngine: coordinator in Faulted state` | 消える |

### 6.3 両方の統合

| # | 検証 | 期待 |
| --- | --- | --- |
| 8 | `sh3_pendingRetire` | 今回の再現条件で 1（一般不変条件ではない）。**テストで確定値を記録し、必要なら診断ログで補足する**（ログだけを一次証拠にしない） |
| 9 | 2 秒待ちの成功と Snapshot 退役 | 区別して記録する（同じ `pendingRetireCount` でまとめない） |
| 10 | Reader 安全 | `reader_registry_final.json` の `activeAtStage: []` を維持 |
| 11 | 回帰 | `ctest` 61 件 + 新規テスト（C3-D の identity、C9-A の 13 条件と両分岐） |

## 7. 残る未確定

設計上の未確定は無い。前進の配置は §2.7 で `waitForDrain` のループ内と確定した（Owner 指定）。

記録方法も確定した（Owner 推奨 2026-11-13）。`sh3_pendingRetire` と前進回数は**テストで確定値を
記録**し、診断ログは補足に留める。ログは実行タイミングや採取漏れで再現性が落ちるため、
一次証拠にしない。

## 8. 別課題（今回の修正に混在させない）

pressure 用カウンタの意味と更新経路の整理。Owner 指摘のとおり、改名だけでなく
**production の更新経路**も明確にする。

| # | 内容 | 現状 |
| --- | --- | --- |
| 1 | 改名（`retireBacklogCount_` → `retireAcceptedCount_` 候補） | 未実施 |
| 2 | production の更新経路 | `EQProcessor.Core.cpp:60` → `RuntimeIntentCoordinator::enqueueRetire` → `onRetireAccepted()` で +1。`onRetireConsumed()` は呼出し元 0 |
| 3 | コード内コメントの誤り | `ISRRuntimePublicationCoordinator.cpp:177`「本メソッドは production で未使用」は事実と違う（EQ が本番で呼ぶ）。要修正 |
| 4 | pressure 検出の実入力 | `noteRetireBacklogChanged(newCount)` の slope（増分）のみ。絶対値は使わない |
| 5 | `onRetireConsumed()` の扱い | 削除するか、減算契機を定義するか（INV-C9-2 の全経路会計が前提） |

## 9. 実装状況（2026-11-13 22:15 時点）

| 対象 | 状態 | 根拠 |
| --- | --- | --- |
| head identity 取得 API 拡張 | **実装・検証完了** | `peekHead` の out 引数、`headSlotIndex` / `headSequence`、`headSlot` / `headSeq` のログ、AC2（6 ケース）、ミューテーション試験、`ctest` 61/61、CLI 3 回で `slot 6 / seq 7 / epoch 12` が一致 |
| C3-D 本体（`waitForDrain` 内の epoch 前進） | 未実装 | 今回の承認は API 拡張に限定 |
| C9-A（Coordinator の drain 述語） | 未実装 | 実装 GO 待ち |
| `retireBacklogCount_` 改名 | 未着手 | 別課題（§8） |
| 2 秒 drain タイムアウト | **未解消** | API 拡張は診断のみで、判定・回収の挙動を変えない |

**判定の切り分け**: API 拡張の完了をもって C3-D 全体の完了とはしない。drain 完了性の改善は、
本体（epoch 前進）の実装後に測る。

## 10. 停止条件の確認

- 本書の設計作業では、ソース変更・ビルド・テストを行っていない。
- 診断ログは暫定保持のまま（API 拡張で追加した `headSlot` / `headSeq` を除き変更なし）。
- 本書は設計記録であり、未実装部分（epoch 前進本体・C9-A）の実装承認を意味しない。
