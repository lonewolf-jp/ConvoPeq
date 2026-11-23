# A/C 計測の変更計画（編集前の提示）

- 作業日時: 2026-11-13 19:43 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋未コミットの診断ログ（暫定保持）
- 種別: **計画の提示のみ。ソースは変更していない。**
- 対象: Owner が条件付き承認した A（c9 の内訳出力）と C（`minReaderEpoch` と先頭 entry の epoch 比較）。B（先頭 entry の属性取得）は保留のため含めない。

---

## 0. 前回報告の訂正（条件数）

前回「c9 の内訳は 11 個の積」と書いたが、**実際は 14 条件**だった。数え直した結果を示す。

`ISRRuntimePublicationCoordinator.cpp:521-572` の `ShutdownScheduler::isFullyDrained()` は、先頭の
`swapPending_` による早期 return と、続く 13 個の条件（`&&` 演算子は 12 個）で構成される。

| # | 条件（式そのまま） |
| --- | --- |
| 1 | `consumeAtomic(swapPending_) != 0` → 即 false |
| 2 | `intentQueue_.sizeApprox() == 0` |
| 3 | `observeDeferredRing_.size() == 0` |
| 4 | `quarantineFallbackQueue_.sizeApprox() == 0` |
| 5 | `recoveryIntentQueue_.size() == 0` |
| 6 | `retireBacklogCount_ == 0` |
| 7 | `publicationBacklogCount_ == 0` |
| 8 | `publicationIntentResidencyCount_ == 0` |
| 9 | `pendingIntentCount_ == 0` |
| 10 | `reclaimInFlightCount_ == 0` |
| 11 | `quarantineIntentResidencyCount_ == 0` |
| 12 | `quarantineRingResidencyCount_ == 0` |
| 13 | `recoveryAdmissionPending_ == false` |
| 14 | `liveLogicalRecoveryObligationCount() == 0` |

前回の報告書 §5.2 と §13 の「11」はこの 14 に読み替える。報告書は本計画の提出時に訂正する。

## 1. 方針（A と C を同じ 1 行で採る）

A と C はどちらも**既存のタイムアウト診断行**（`ReleaseResources.cpp:728` の `[DRAIN_DIAG] timedOut=1`）に
項目を足す形にし、同じ時点で採取する。**ログ行は増やさない。**

## 2. A: c9 の内訳

### 2.1 設計（単一の真実源にする）

14 条件を 1 つの構造体として返す `DrainBreakdown` を追加し、`isFullyDrained()` 自身もこの構造体から
判定する。ログと判定式が同じ関数を共有するため、片方だけ直して食い違うことが構造的に起きない。

追加する型と API（`src/audioengine/ISRRuntimePublicationCoordinator.h`、`isFullyDrained()` 宣言の直前 `:221` 付近）:

```cpp
struct DrainBreakdown {
    bool     swapPending{false};
    uint32_t intentQueue{0};              // sizeApprox()
    uint32_t observeDeferredRing{0};      // size()
    uint32_t quarantineFallback{0};       // sizeApprox()
    uint32_t recoveryIntentQueue{0};      // size()
    uint64_t retireBacklog{0};
    uint64_t publicationBacklog{0};
    uint64_t publicationIntentResidency{0};
    uint64_t pendingIntent{0};
    uint64_t reclaimInFlight{0};
    uint64_t quarantineIntentResidency{0};
    uint64_t quarantineRingResidency{0};
    bool     recoveryAdmissionPending{false};
    uint64_t liveRecoveryObligation{0};

    // 14 条件の AND（isFullyDrained と同じ式）。ヘッダ内でインライン定義する
    [[nodiscard]] bool allDrained() const noexcept;
};

[[nodiscard]] DrainBreakdown collectDrainBreakdown() const noexcept;
```

`ShutdownScheduler::isFullyDrained()`（`ISRRuntimePublicationCoordinator.cpp:521-572`）は次の 1 行に置き換える。

```cpp
bool RuntimeIntentCoordinator::ShutdownScheduler::isFullyDrained() const noexcept {
    return coordinator_.collectDrainBreakdown().allDrained();
}
```

### 2.2 取得元と判定式の一致

| # | 構造体のフィールド | 取得式（現行と同一） | 既存 getter |
| --- | --- | --- | --- |
| 1 | `swapPending` | `consumeAtomic(swapPending_, acquire) != 0` | `isSwapPending()` あり |
| 2 | `intentQueue` | `intentQueue_.sizeApprox()` | **なし（新規）** |
| 3 | `observeDeferredRing` | `observeDeferredRing_.size()` | **なし（新規）** |
| 4 | `quarantineFallback` | `quarantineFallbackQueue_.sizeApprox()` | **なし（新規）** |
| 5 | `recoveryIntentQueue` | `recoveryIntentQueue_.size()` | **なし（新規）** |
| 6 | `retireBacklog` | `consumeAtomic(retireBacklogCount_, acquire)` | `getRetireBacklogCount()` あり |
| 7 | `publicationBacklog` | `consumeAtomic(publicationBacklogCount_, acquire)` | `getPublicationBacklogCount()` あり |
| 8 | `publicationIntentResidency` | `consumeAtomic(publicationIntentResidencyCount_, acquire)` | `getPublicationIntentResidencyCount()` あり |
| 9 | `pendingIntent` | `consumeAtomic(pendingIntentCount_, acquire)` | `getPendingIntentCount()` あり |
| 10 | `reclaimInFlight` | `consumeAtomic(reclaimInFlightCount_, acquire)` | `getReclaimInFlightCount()` あり |
| 11 | `quarantineIntentResidency` | `consumeAtomic(quarantineIntentResidencyCount_, acquire)` | `getQuarantineIntentResidencyCount()` あり |
| 12 | `quarantineRingResidency` | `consumeAtomic(quarantineRingResidencyCount_, acquire)` | `getQuarantineRingResidencyCount()` あり |
| 13 | `recoveryAdmissionPending` | `consumeAtomic(recoveryAdmissionPending_, acquire)` | **なし（新規）** |
| 14 | `liveRecoveryObligation` | `liveLogicalRecoveryObligationCount()` | あり |

注意した点:

- **13 は既存の `hasPendingRecoveryAdmission()` を使わない。** あの関数は
  `recoveryAdmissionPending_ && state != NoAdmission` を返す別の述語で、`isFullyDrained()` が見ている
  `recoveryAdmissionPending_` 単独と一致しない（`ISRRuntimePublicationCoordinator.cpp:1340-1345`）。
- 2〜5 は 4 本のキューの `size` / `sizeApprox`。いずれも `const noexcept` で、内部は atomic の読み取りのみ
  （`MpscBoundedRing.h:134`、`LockFreeRingBuffer.h:76`）。
- 既存 getter があるものも、`collectDrainBreakdown()` の中で**同じ式**を書く（getter 経由に統一すると
  `hasPendingRecoveryAdmission` のような別述語を掴む事故が起きるため、式を揃える方を優先する）。

### 2.3 短絡評価が変わる点（明示）

現行は `&&` の短絡で、`swapPending_` が真なら残り 13 個は読まれない。新設計は**毎回 14 個すべてを読む**。

- 増えるのは atomic ロード 16 回相当（直接 10 回 ＋ リングの size 呼出し 4 回で 6 回）。
  ロック・CAS・書き込みは無い。
- `isFullyDrained()` の呼出し頻度は 2 ms ポーリングと CoordinatorLoop の tick。1 回あたり数十 ns の増。
- 判定結果（真偽）は変わらない。順序依存も無い（全条件が純粋な読み取り）。

この変更を避けたい場合の代替（A2）: `isFullyDrained()` は現行のまま残し、ログ用に同じ 14 式を
`collectDrainBreakdown()` へ複写する。判定式の一致は保たれるが、将来 `isFullyDrained()` を直したときに
ログ側が追随しない危険が残る。**推奨は A1（本節の設計）。**

## 3. C: `minReaderEpoch` と先頭 entry の epoch

### 3.1 既存手段の確認結果

| 値 | 既存手段 | 判定 |
| --- | --- | --- |
| `minReaderEpoch` | `EpochDomain::getMinReaderEpoch()` が **public**（`src/core/EpochDomain.h:372`、`public:` は `:35`、`private:` は `:706`） | 新規 API **不要** |
| `currentEpoch` | `EpochDomain::currentEpoch()` が public（`:344`） | 新規 API **不要** |
| 先頭 entry の epoch | **既存手段なし**。`DeferredDeletionQueue` は `reclaim()` の内部で先頭を読むだけで、覗く API を持たない（`dequeue`/`peek`/`front` いずれも無し） | 新規 API **必要** |
| 先頭 entry の型・ドメイン | 同上 | 新規 API に同梱 |

`deferredDeletionQueue` は `EpochDomain` の `private:` 配下（`:747`）なので、`AudioEngine` から
直接は触れない。したがって公開は `EpochDomain` 経由の 1 本にまとめる。

### 3.2 追加する API（2 箇所）

**(1) `src/DeferredDeletionQueue.h`（`sizeApprox()` の直後 `:276` 付近）— 読み取り専用の先頭参照**

```cpp
// ★ AC_DIAG (2026-11-13): 先頭 entry の読み取り専用スナップショット。
//   書き込み・CAS・dequeue は行わない。seq が pos+1 でなければ「空」として false。
[[nodiscard]] bool peekHead(DeletionEntry& out) const noexcept
{
    const uint32_t pos = convo::consumeAtomic(dequeuePos, std::memory_order_acquire);
    const uint32_t seq = convo::consumeAtomic(sequences[pos & kMask], std::memory_order_acquire);
    if (static_cast<int32_t>(static_cast<uint32_t>(seq - (pos + 1))) != 0)
        return false;                      // 空（reclaim() の Empty 判定と同じ式）
    out = ringBuffer[pos & kMask];         // POD のコピーのみ
    return true;
}
```

**(2) `src/core/EpochDomain.h`（`tryReclaim()` の直後 `:557` 付近）— 3 値を 1 回で採る**

```cpp
struct RetireHeadSnapshot {
    bool     valid{false};
    uint64_t headEpoch{0};
    uint64_t minReaderEpoch{0};
    uint64_t currentEpoch{0};
    DeletionEntryType type{DeletionEntryType::Generic};
    convo::RetireDomainId domainId{convo::RetireDomainId::Unknown};
    uint64_t firstScanTsNs{0};
    bool     olderThanMinReader{false};   // isOlder(headEpoch, minReaderEpoch)
};

[[nodiscard]] RetireHeadSnapshot inspectRetireHead() const noexcept
{
    RetireHeadSnapshot s;
    s.minReaderEpoch = getMinReaderEpoch();   // reclaim() に渡すのと同じ値
    s.currentEpoch   = currentEpoch();
    DeletionEntry e;
    if (deferredDeletionQueue.peekHead(e)) {
        s.valid = true;
        s.headEpoch = e.epoch;
        s.type = e.type;
        s.domainId = e.domainId;
        s.firstScanTsNs = e.firstScanTsNs;
        s.olderThanMinReader = isOlder(e.epoch, s.minReaderEpoch);  // EpochDomain::isOlder
    }
    return s;
}
```

`olderThanMinReader` は `reclaim()` が削除可否に使う述語そのもの（`DeferredDeletionQueue.h:174` の
`isOlder(entry.epoch, minReaderEpoch)`、実装は `static_cast<int64_t>(a - b) < 0` で同一）。
したがって「先頭が回収されない理由が epoch 条件か」を直接判定できる。

### 3.3 同期への影響

- `peekHead` は atomic ロード 2 回と `DeletionEntry`（診断 OFF で 56 B）のコピー 1 回。**ロックなし・書き込みなし・
  CAS なし**。`reclaim()` の 1 周目とまったく同じ読み方をする。
- 前提: 呼出しは**タイムアウト時点**で、(a) 音声スレッドは停止済み、(b) rebuild スレッドは join 済み
  （`ReleaseResources.cpp:671` の `jassert(!rebuildThreadIsRunning)`）、(c) consumer はこのスレッドのみ。
  つまり producer / consumer とも同時書き込みが無い。`reclaim()` の 2 ms ループと同じスレッドで走る。
- 万一 producer が同時に居ても、読むのは POD のフィールドだけでポインタを辿らないため、参照外しの危険は無い。
  値が 1 世代ずれる可能性はあり、その場合は `valid=false`（seq 不一致）か、直前の entry を読む。
- `getMinReaderEpoch()` は 128 スロットを acquire で走査する。`reclaim()` が 2 ms ごとに既に実行している
  同じ処理で、今回は終了時に 1 回だけ追加。
- `firstScanTsNs` は「最初に走査された時刻」。0 なら未走査。`firstScanAgeMs` として出す。

## 4. ログ（同じ 1 行に追加）

`ReleaseResources.cpp:728` の既存行の末尾に次を足す。**新しい行は作らない。**

```
[DRAIN_DIAG] timedOut=1 domain=EngineISR failed=... （既存の 9 条件）
  c9_swapPending=0 c9_intentQueue=0 c9_observeRing=0 c9_quarantineFallback=0
  c9_recoveryQueue=0 c9_retireBacklog=0 c9_pubBacklog=0 c9_pubIntentRes=0
  c9_pendingIntent=0 c9_reclaimInFlight=0 c9_quarantineIntentRes=0
  c9_quarantineRingRes=0 c9_recoveryAdmission=0 c9_liveObligation=0
  headValid=1 headEpoch=12345 headType=1 headDomain=2 headFirstScanAgeMs=2001
  minReaderEpoch=12345 currentEpoch=12345 epochOlder=0
```

- `c9_*` は 14 個。`failed=c3,c9` の c9 がどの内訳で落ちているかが 1 行で読める。
- `headType` は `DeletionEntryType` の整数値（0=Generic / 1=World）、`headDomain` は `RetireDomainId`
  （2=EngineISR / 3=EQ）。
- `epochOlder=0` かつ `headEpoch == minReaderEpoch` なら「同じ epoch のため `isOlder` が成立しない」と確定できる。
- 先頭が空なら `headValid=0` とし、他は出さない（c3=0 と整合するはず）。

**1 shutdown あたりのログ出力回数: 2 行**（`worldClear` 1 行 + `timedOut` 1 行）。前回と同じで増えない。

## 5. 変更ファイルと追加箇所（一覧）

| # | ファイル | 箇所 | 内容 | 追加行（概算） |
| --- | --- | --- | --- | ---: |
| 1 | `src/audioengine/ISRRuntimePublicationCoordinator.h` | `:221` の直前 | `DrainBreakdown` 定義 + `collectDrainBreakdown()` 宣言 | 30 |
| 2 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp` | `:522-571` | `isFullyDrained()` の本体を `collectDrainBreakdown().allDrained()` の 1 行に置換 | −50 / +1 |
| 3 | 同 | `:477` の直後 | `collectDrainBreakdown()` の実装（`allDrained()` はヘッダでインライン） | 35 |
| 4 | `src/DeferredDeletionQueue.h` | `:276` 付近 | `peekHead()` | 12 |
| 5 | `src/core/EpochDomain.h` | `:557` 付近 | `RetireHeadSnapshot` + `inspectRetireHead()` | 28 |
| 6 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | `:728` の既存行 | ログ項目の追加 | 18 |

## 6. 計測と検証

| # | 検証 | 方法 |
| --- | --- | --- |
| 1 | ビルド | `cmake --build build --config Release`（既知の Octave PATH 問題を回避して構成） |
| 2 | 回帰 | `ctest -C Release` 61 件 |
| 3 | 再現 | `--cli-run --cli-exit-ms 2000 --verify-runtime-evidence-live` を 3 回、作業ディレクトリは一時フォルダ |
| 4 | 行数 | `[DRAIN_DIAG]` が 1 shutdown あたり 2 行のままであること |
| 5 | 判定の一致 | 変更前後で `failed=` が同じ（`c3,c9`）であること。`isFullyDrained()` の真偽が変わらないこと |
| 6 | 終了への影響 | 総経過と exit 0 を前回（5.99〜6.69 s）と比較 |
| 7 | 設定復元 | `%APPDATA%\ConvoPeq` の 3 ファイルを実行前ハッシュへ戻す |

## 7. 計測後の撤去方法

1. `ReleaseResources.cpp` のログ項目を削除する（行は前回の 2 行に戻る）。
2. `EpochDomain::RetireHeadSnapshot` / `inspectRetireHead()` を削除する。
3. `DeferredDeletionQueue::peekHead()` を削除する。
4. `DrainBreakdown` / `collectDrainBreakdown()` を削除し、`ShutdownScheduler::isFullyDrained()` を
   現行の 13 条件（12 個の `&&`）へ戻す（`git show HEAD:src/audioengine/ISRRuntimePublicationCoordinator.cpp` の
   該当範囲と同一）。
5. 5 ファイルを `git checkout` すれば前回の状態（診断ログのみ）に戻る。**データ構造・ファイル形式の
   変更は無い**ので、移行や後始末は不要。

## 8. リスクと未確定

| # | 内容 | 扱い |
| --- | --- | --- |
| 1 | `isFullyDrained()` の短絡評価が無くなる | §2.3 のとおり。真偽は不変。A2（複写）で回避可能 |
| 2 | `peekHead` が読む値は「その瞬間」のもの | 2 ms ループの直後なので、直前に見た値と 1 entry ずれる可能性がある。時刻ではなく値のスナップショットとして扱う |
| 3 | c9 の内訳がすべて 0 になる可能性 | その場合 `isFullyDrained()` の偽は別要因（`allDrained()` の実装ミスか、読んだ瞬間の差）。そのときは `isFullyDrained()` の戻り値も同時に記録して突き合わせる（ログに `c9_allDrained=` を 1 項目追加） |
| 4 | 先頭 entry の `type`/`domain` まで出す | B（保留）の一部を先取りする形になるが、`peekHead` が返す構造体に最初から入っているため追加コストが無い。不要なら 2 項目削る |
| 5 | `ConvoPeq.md` の再生成 | Owner が「診断変更後の最新統合ソースを確認できていない」と述べた点。今回は範囲外のため未実施。必要なら別途生成する |

## 9. 停止条件の確認

- ソース変更・ビルド・テストは行っていない。
- 本書は計画であり、A/C の実装承認を意味しない。
