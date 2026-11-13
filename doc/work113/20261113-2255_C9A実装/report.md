# C9-A の実装と検証（Coordinator の drain 述語）

- 作業日時: 2026-11-13 22:55 JST
- 基準 commit: `173e69957b046290f7077797782f031ed8633fd0`（`main` / `origin/main`）＋未コミットの診断ログ
- 種別: **Owner 承認済み範囲（C9-A）のみ**。C3-D 本体と改名は別報告。
- 対象: `ShuttingDown → Bootstrapping`（正常系）と `ShuttingDown → Faulted`（異常系）の両方。

---

## 1. 実装（承認条件との対応）

| # | 承認条件 | 実装 |
| --- | --- | --- |
| 1 | `ShuttingDown → Faulted` の遷移を対象 | Faulted 側のテストを追加（§3.2）。遷移ロジック自体は無変更 |
| 2 | 確定済み設計の 13 条件を基準 | 述語から `retireBacklogCount_ == 0` のみを削除。残り 13 条件は不変（§1.1） |
| 3 | Faulted 側は対象条件を非 0 にして検証 | `setPendingIntentCount(1)` で 13 条件の 1 つを非 0 にし、Faulted を確認 |
| 4 | 正常系の完了条件・既存の drain 判定を変更しない | 13 条件と Layer 1（`pendingRetireCount`）は不変。変えたのは retireBacklog の 1 条件だけ |
| 5 | 既存の状態遷移契約・不変条件を維持 | `markShutdownComplete` の実装は無変更。遷移先の判定材料だけが変わる |
| 6 | 結果と Faulted を成立させた条件を記録 | §3.2 に記載 |

### 1.1 変更内容

`ISRRuntimePublicationCoordinator.cpp:545` の `ShutdownScheduler::isFullyDrained()` から
次の 1 条件を削除した（`:561-570`）。

```cpp
// 削除前
&& convo::consumeAtomic(coordinator_.retireBacklogCount_, std::memory_order_acquire) == 0
```

残る 13 条件:

| # | 条件 |
| --- | --- |
| 1 | `swapPending_` の早期 return |
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

- `AudioEngine::isFullyDrained()` の `pendingRetireCount() == 0`（Layer 1）。
- `retireBacklogCount_` の増減処理（`onRetireAccepted` / `onRetireConsumed`）。
- `markShutdownComplete()` の実装。

**意味の確認**: この変更は retire 資源が残っていても Engine 側が drain 完了と判定することを
意味しない。Layer 1 は `pendingRetireCount()` を実測しており、retire entry が残れば
`AudioEngine::isFullyDrained()` は依然として偽になる。

## 2. テスト

### 2.1 既存テストの更新（1 件）

`testShutdownCompleteFailsWhenNotDrained` は `setRetireBacklogCount(1)` で drain 条件を破っていた。
C9-A 後はこの操作が条件を破らないため、**残る 13 条件の 1 つ**である `setPendingIntentCount(1)` に
置き換えた（`ISRSemanticValidationTests.cpp:368-373`）。テストの目的（未 drain なら Faulted）は不変。

### 2.2 追加テスト（2 件）

| テスト | 内容 | 期待 |
| --- | --- | --- |
| `testC9A_retireBacklogDoesNotBlockDrain`（`:2738`） | `setRetireBacklogCount(5)` のみ設定 → `isFullyDrained()` → `requestShutdown()` → `markShutdownComplete()` | drained と判定され、**Bootstrapping** へ遷移 |
| `testC9A_faultedWhenOtherConditionSet`（`:2752`） | `setRetireBacklogCount(5)` ＋ `setPendingIntentCount(1)` → 同上 | **Faulted** へ遷移（除外した条件ではなく、残した条件で成立） |

登録は `main()` の `:3035-3038`。

## 3. 検証結果

### 3.1 テスト実行

`build\Release\ISRSemanticValidationTests.exe` → **PASS（exit 0）**。
Bootstrapping 側・Faulted 側の両方が通った。

### 3.2 Faulted 遷移を成立させた条件

`setPendingIntentCount(1)`（13 条件の 8 番目）。`retireBacklogCount_` は 5 のまま非 0 だが、
これは判定に使われないため Faulted の理由ではない。**除外した条件が原因ではないこと**を
このテストで固定した。

### 3.3 実機 3 回（C3-D 本体との統合）

| 実行 | exit | 総経過 | `DRAIN_DIAG` 行数 | タイムアウト行 | `sh3_pendingRetire` | `reclaimSuccess` | `[FAULT] ... Faulted state` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0 | 4.57 s | 1 | **0** | **0** | 8 | なし |
| 2 | 0 | 4.28 s | 1 | **0** | **0** | 8 | なし |
| 3 | 0 | 4.30 s | 1 | **0** | **0** | 8 | なし |

- **2 秒 drain タイムアウトが解消した。** `drain timeout reached` と
  `[ISR][Shutdown] Drain incomplete` の行が 3 回とも出ていない。
- **`[FAULT] ~AudioEngine: coordinator in Faulted state` が消えた。** 変更前は 3/3 で出ていた。
  Coordinator が `Bootstrapping` に遷移したことの実機証跡である。
- 総経過が 6.23〜6.81 s → 4.28〜4.57 s に短縮（約 2 秒 = 待ち時間の消滅）。
- `sh3_pendingRetire=0`（変更前は 2）。キューは終了時点で空。

### 3.4 `sh3_pendingRetire=0` の理由（コード上の説明）

設計では「2 秒待ちの後に `finalizeShutdown` が snapshot を 1 件 enqueue するため 1 が残る」と
見積もったが、実測は 0 だった。理由は次のとおり。

- `EpochDomain::publishEpoch()` は `fetchAddAtomic` の戻り値、すなわち**前進前の epoch** を返す
  （`EpochDomain.h:357`）。
- `SnapshotCoordinator::retireCurrentAndTarget()` はその戻り値（例: 13）を retire epoch として
  enqueue するが、その時点の `currentEpoch` は既に 14。
- `finalizeShutdown(timedOut=false)` は retire の直後に `tryReclaim()` を呼ぶ
  （`SnapshotCoordinator.h:69-70`）。`isOlder(13, 14)` が真なので**その場で回収**される。
- タイムアウト時（`timedOut=true`）はこの `tryReclaim()` がスキップされるため、以前は 1 件残っていた。

つまり 0 は「snapshot が退役されなかった」のではなく「退役されたうえで即回収された」結果である。

## 4. 回帰

`ctest --test-dir build -C Release` → **61/61 成功**（153.89 s）。C9-A で更新した既存テストと
追加 2 件を含む。

## 5. 残留リスク

- `retireBacklogCount_` は依然として単調増加する（本変更では触れていない）。drain 判定から外れたため
  実害は無いが、pressure の slope 以外の用途に使うと誤解を生む。改名課題（別途）で整理する。
- Faulted 側のテストは `pendingIntentCount_` の 1 条件のみを代表として固定している。13 条件すべてを
  個別に Faulted 検証しているわけではない（述語が空でないことの確認としては十分）。
- Layer 1（`pendingRetireCount`）の実測判定は不変。retire entry が残れば drain は完了しない。

## 6. ロールバック

`ISRRuntimePublicationCoordinator.cpp` の削除した 1 条件を戻し、テストを旧契約
（`setRetireBacklogCount(1)`）へ戻す。追加 2 テストを削除する。

## 7. 判定

- 承認範囲（C9-A）は**実装完了・検証完了**。
- 正常系（Bootstrapping）と異常系（Faulted）の両方をテストで固定した。
- C3-D 本体との統合で **2 秒タイムアウトが解消**し、`[FAULT]` も消えた（3/3）。
