# STG-9-D1 Repair Gate（2026-09-28・RC-1 実装）

> **Verdict: STOP — FAIL（RC-1 は不完全。Owner 再承認が必要）**
> **Commit = NOT AUTHORIZED / Push = NOT AUTHORIZED**
> 本書は §12 stop rule により、**修正を追加せず** 失敗内容・最小再現・該当 source location を報告する。

---

## 1. Authority

`ConvoPeq.md` / `ConvoPeq(10).md` を**同一の統一 authority**として扱った。main repo ルートに実在する
ファイルは `ConvoPeq.md` 1 件（`ConvoPeq(10).md` という別ファイルは存在しない）。
本セッションで authority を**再生成していない**（SHA / mtime とも開始時と同一）。

| 項目 | 値 |
| --- | --- |
| HEAD | `b0b4694161817e705b21534777a2b2cf8085f0f0`（実装前後で不変） |
| branch | `main` |
| authority SHA-256 | `615940DC90D73CED680081A8FBD0B491DBDC376788C0469B0A59AF7D2F6F2E3B` |
| size | 5,612,326 B |
| Generated | `2026-09-28 16:17:11` |
| NEWER_SRC_COUNT | 0（実装前）/ **1**（実装後・`AudioEngine.h` / `AudioEngine.Retire.cpp` が authority より新しい） |
| FRESH（実装前） | **YES** |

**source authority と production source の一致（内容レベル）**: 実装前に、authority から 351 セクションを
抽出し RC-1 対象を含む 12 ファイル（`AudioEngine.h` / `AudioEngine.Retire.cpp` /
`ISRRuntimePublicationCoordinator.{cpp,h}` / `AudioEngine.Threading.cpp` / `ISRLifetimeProof.h` /
`ISRDSPHandle.cpp` / `DSPLifetimeManager.cpp` / `invariant_INV3_INV5.cpp` /
`ISRSemanticValidationTests.cpp` / `CMakeLists.txt` / `build.bat`）と on-disk を**バイト比較**。
**12/12 MATCH**。したがって実装開始時点で authority は production source と一致していた（停止条件に該当せず着手可）。

---

## 2. Implementation delta（RC-1 のみ）

| ファイル | 変更 | 種別 |
| --- | --- | --- |
| `src/audioengine/AudioEngine.h` | `requestReclaimHandle`: caller-side epoch pre-check（`retireEpoch < minReaderEpoch` 判定）と else 分岐を削除。`if (!requestReclaim(...)) { push_back }` の単一形へ統一。doc comment を RC-1 後の意味論へ更新。`pendingReclaimHandles_` 宣言コメントに 1:1 ownership invariant を追記 | **production** |
| `src/audioengine/AudioEngine.Retire.cpp` | `drainDeferredRetireQueues` の retry ループ: caller-side epoch pre-check と else 分岐を削除。`!isRetired` の drop 分岐に `onReclaimEnd()` を追加（RC-1 の中核）。コメント更新 | **production** |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 既存 friend 経由の test-only accessor 3 件追加（`pendingReclaimCount` / `reclaimInFlightCount` / `handleRuntime`）。**production ヘッダ・可視性・ロジック無変更** | test |
| `src/tests/AudioEngineHarness/STG9ReclaimAccountingTests.cpp` | **新規**。T-01〜T-04 | test |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | `runSTG9ReclaimAccountingTests()` の前方宣言 + 呼び出し（STG-8 直後） | test |
| `CMakeLists.txt` | harness sources に 1 行追加 | build |

`git diff --stat -- src CMakeLists.txt`:
```
 CMakeLists.txt                                     |  1 +
 src/audioengine/AudioEngine.Retire.cpp             | 39 ++++++++++--------
 src/audioengine/AudioEngine.h                      | 47 +++++++++++++++++-------
 .../DeferredPublicationTestAccess.h                | 25 ++++++++++++++++
 .../PublishPipelineIntegrationTests.cpp            |  6 +++
 5 files changed, 79 insertions(+), 39 deletions(-)
```

### 無変更（Owner §3 の禁止リスト）— すべて実測で確認

`ISRRuntimePublicationCoordinator.cpp` / `.h` / `AudioEngine.Threading.cpp` / `ISRLifetimeProof.h` /
`ShutdownScheduler::isFullyDrained()` / `AudioEngine::isFullyDrained()` / `reclaimNormal()` の +1・−1 /
`reclaimShutdownQuiescent()` / `QuarantineService` / crossfade authority / `EpochDomain` /
`RuntimeWorld`・`RuntimePublishWorld` / RT path — **すべて無変更**。

---

## 3. RC-1 acceptance: **FAIL（不完全）**

**RC-1 の terminal drop 修正は正しく、しかし retry（再登録）経路の account 欠落により
`reclaimInFlightCount_` が entry 数を超えて際限なく増加する。T-04 の oracle がこれを検出した。**

### 3.1 根本原因（exact source location）

`src/audioengine/AudioEngine.Retire.cpp:112-123`（RC-1 実装後の形）:

```cpp
112  if (dspHandleRuntime_.isRetired(handle))
113  {
114      const auto retireEpoch = m_retireRouter->currentEpoch();
115      // deferred 1 通知 = pending entry 1 件 = reclaimInFlightCount_ +1 1 回（1:1）。
116      if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter))
117      {
118          std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
119          // ★ dash2 §2.2 (G14): ReclaimIdentity として再登録。
120          pendingReclaimHandles_.push_back(
121              convo::isr::ReclaimIdentity{ handle, retireEpoch });
122          // ★ ここに onReclaimEnd() が無い  ← accounts 欠落点
123      }
124  }
```

**_account の流れ（1 回の retry pass）**:
1. `Retire.cpp:96` `pending.swap(pendingReclaimHandles_)` により、既存 entry E_old が member list から**除去される**（entry 数 −1）。
2. `Retire.cpp:116` `requestReclaim` が deferred を返す → `Coordinator.cpp:705` `onReclaimBegin()` により **+1**。
3. `Retire.cpp:120` `push_back` により E_new が登録される（entry 数 +1 に戻る）。
4. 結果: **entry 数は不変、counter は +1。** E_old が持つ前の `+1` は**永久に解放されない**。

停止済み engine（active reader 0）では `EpochDomain::getMinReaderEpoch() == currentEpoch()` となるため
`reclaimNormal` は**必ず** deferred を返す。よって **drain pass ごとに +1**、そして **entry 数は 1 のまま**。

### 3.2 修正前の（pre-RC-1）との差分 — RC-1 が流出を加速した

pre-RC-1 の drain ループは caller-side pre-check を持ち、停止済み engine では
`retireEpoch == minReaderEpoch` により pre-check が false となり **`requestReclaim` を呼ばなかった**
（= `+1` が発生しない）。したがって:

| | pre-RC-1 | post-RC-1 |
| --- | --- | --- |
| 停止済み engine の counter | 0 | **822〜844**（3 連続実行で実測） |
| 停止済み engine の pending | 1 | 1 |
| INV-1（`pending 空 ⟹ counter == 0`） | 恒真（pending=1 なので空ではない） | **違反**（pending=1, counter=830 相当） |

**`:115` のコメント「deferred 1 通知 = pending entry 1 件 = +1 1 回（1:1）」は誤り。**
deferred 通知は「既存 entry の**置換**」であり「新規 obligation の追加」ではないため、
置換される entry の `+1` を対で解放する必要がある。

### 3.3 Contract Audit 側の誤り（自認）

`P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-AUDIT_20260928.md` §4.2 の不変条件 I-2 は
「entry の消滅は 3 通り（**再 push** / 成功 consume / drop）で、再 push は counter 不変」と書いた。
**この「再 push は counter 不変」が誤り**だった。再 push は常に `reclaimNormal` の **新しい `+1`** と
同時に発生し、置換された entry の `+1` を解放しない。I-2 の証明は**この遷移を数えていなかった**ため
不完全であった。T-04 の oracle（「retry = 新しい +1 としてはいけない」）がこれを検出した。

### 3.4 最小再現

```
1. AudioEngineHarness h;  h.start(48000.0, 512);  h.stop();
   （CoordinatorLoop join 済み・message pump なし = 決定的）
2. AudioEngine& e = h.engine();
3. observe: pendingReclaimCount(e) == 1, reclaimInFlightCount(e) == 822〜844
4. → INV-1 違反 / isFullyDrained() == false
```
実際の観測値（3 連続実行）:

| run | T-01 baseline | T-02 baseline | T-04 row0 step0 |
| --- | --- | --- | --- |
| 1 | pending=1, counter=**834** | pending=1, counter=**842** | pending=2, counter=**825** |
| 2 | pending=1, counter=**844** | pending=1, counter=**839** | pending=2, counter=**825** |
| 3 | pending=1, counter=**822** | pending=1, counter=**829** | pending=2, counter=**836** |

### 3.5 Owner 再承認時に必要な追加（**本 STG では未実装**）

`AudioEngine.Retire.cpp:121` の直後に 1 行追加すれば I-2 が完全になる（`reclaimNormal` は無改変のまま）:

```cpp
if (!runtimePublicationBridge_.requestReclaim(handle, dspHandleRuntime_, *m_retireRouter))
{
    std::lock_guard<std::mutex> lock(pendingReclaimHandlesMutex_);
    pendingReclaimHandles_.push_back(
        convo::isr::ReclaimIdentity{ handle, retireEpoch });
    // ★ 置換された entry（旧 E_old）の +1 を対で解放する
    runtimePublicationBridge_.onReclaimEnd();
}
```

これで「entry 1 件 = counter +1 1 回」が全遷移（新規登録 / 再登録 / 成功 consume / terminal drop）で
成立する。**production 変更範囲は `AudioEngine.Retire.cpp` 内の 1 行追加に収まり、RC-1 の境界を
拡大しない。** ただし §12 により本 STG では追加していない。

### 3.6 T-03 の oracle 設計の誤り（別問題）

T-03 の baseline 前提「停止済み engine は完全 drain 済み = `isFullyDrained()==true`」は**不正**。
実測で停止済み engine には `pendingReclaimHandles_` に **1 件が正当に残留**している
（epoch がMLE を越えておらず reclaim が deferred 継続中）。
pre-RC-1 でも同じで、この 1 件は `pendingReclaimEmpty == false` として `isFullyDrained()` を false にする。
したがって T-03 の oracle は「engine 全体が drained」ではなく
「**本シーケンスが導入した reclaim conjunct が baseline へ戻る**」へ設定し直す必要がある。

---

## 4. T-01 / T-02 / T-03 / T-04

| test | 結果 | 観測 |
| --- | --- | --- |
| **T-01** quarantine drop releases counter | **FAIL** | `T-01 baseline expected (pending=0, counter=0) actual (pending=1, counter=834)`。baseline が既に INV-1 違反のため到達不能。**drop 経路自体の `+1` 解放は実装済み**（T-01 の狙いは正しいが、baseline 前提が崩れている） |
| **T-02** shutdown-quiescent drop releases counter | **FAIL** | `T-02 baseline expected (pending=0, counter=0) actual (pending=1, counter=842)`。`reclaimShutdownQuiescent` には一切手を入れておらず（Owner §3 準拠）、drop 集約も機能している。baseline 前提の問題 |
| **T-03** drain completion oracle | **FAIL** | `T-03 baseline isFullyDrained()==false`。§3.6 のとおり baseline 前提が不正。T-03 の真の oracle（`pending==0 && counter==0 && isFullyDrained()==true`）には到達していない |
| **T-04** INV-1 property matrix | **FAIL** | `T-04 row 0 step 0 1:1 broken (pending=2 counter=825)`。**row 0 は `{defer, drain, drain, drain}` = retry のみのパスで、§3.1 のaccounts 欠落を初手から検出する。Owner §5「retry = 新しい +1 としてはいけない」の直接的な oracle として機能した** |

**T-01〜T-04 の 4 件すべてが baseline 不成立により失敗。T-04 のみが RC-1 本体の欠陥を直接指している。**

---

## 5. Existing regression

| 対象 | 結果 |
| --- | --- |
| CTest（Debug） | **41 / 42 PASS**、1 FAIL（`AudioEngineHarness`） |
| 失敗テスト | `41 - AudioEngineHarness (Failed)` — **原因は STG-9 subtest のみ** |
| `AudioEngineHarness` 内 STG-1 / STG-2 / STG-2-A/B/C / STG-2-R / STG-4-1 / STG-6-D1 / STG-7-D1 | **PASS**（回帰なし） |
| `AudioEngineHarness` 内 STG-8-D1 / D2 / D2b / D3 | **PASS**（3 連続実行すべて） |
| `ConvolverStateRoundTripTests` | **PASS** |
| `invariant_INV3_INV5Tests`（`testInv3_1` / `testInv3_2` / `testInvX3_4` の 4 oracle を含む） | **PASS** |
| `ISRSemanticValidationTests` | **PASS** |
| `ISRSoakTests` / `ShutdownRetireIntentDrainTests` / `StuckReaderFallbackDrainTests` | **PASS** |
| CTest total time | 185.38 sec |

**既存 oracle の改変: 0 件。既存回帰の破壊: 0 件**（Owner §6 準拠）。

### 5.1 観測された間欠的失敗（要 Owner 判定）

1 回目のみ `FAIL: checkSTG82OverwriteTerminalizesEvicted`（STG-8-D2）が 1 回だけ発生した。
後続 3 連続実行では STG-8-D1/D2/D2b/D3 すべて PASS で再現しなかった。
**STG-8-D2 の pre-existing フレークか、RC-1 の counter 膨張によるタイミング変化かは
本 STG の範囲内では未確定。**Owner の指示（既存回帰維持）と 관련하므로明示して報告する。

---

## 6. Debug

| 項目 | 結果 |
| --- | --- |
| build | **PASS**（`BUILD_EXIT=0`、548 targets、`AudioEngineHarness.exe` 生成） |
| 環境 | MSVC 14.51.36231 + Windows SDK 10.0.26100 + Ninja（VS18 同梱）+ oneAPI `setvars.bat intel64` + `MKLROOT` を 2026.0 に固定 |
| 前提 | `build/CMakeCache.txt` の stale `MKL_VERSION_H`（`mkl/latest` = 2026.1 を指す残留エントリ）を除去。バックアップ `build/CMakeCache.txt.bak-stg9`。**source への変更ではない** |

---

## 7. Release

**NOT RUN** — §12 stop rule（`T-01〜T-04 の oracle が成立しない`）により停止。
Release の結果を捏造せず、未実施であることを記録する。

---

## 8. CTest

| config | 結果 |
| --- | --- |
| Debug | **41 / 42 PASS**（185.38 sec）、1 FAIL = `AudioEngineHarness`（STG-9 subtest のみ） |
| Release | **NOT RUN**（§7 と同じ理由） |

baseline 42/42 に対し **41/42**。差分は STG-9 subtest のみであり、既存 41 テストは全て PASS。

---

## 9. Static validation

WSL `rg`（内蔵 ripgrep と独立エンジン）による 2 エンジン交差検証。

| 検査項目 | 期待 | 実測 | 判定 |
| --- | --- | --- | --- |
| `onReclaimBegin()` production call sites | 3（変更なし） | `AudioEngine.Retire.cpp:62`（B1）、`:363`（B2）、`Coordinator.cpp:211`（定義） | **PASS** |
| `onReclaimEnd()` 既存 balanced caller | 2 組 変更なし | `:66`（E1）、`:366`（E2） | **PASS** |
| `onReclaimEnd()` 新規 terminal-drop call | **exactly 1** | `:138` | **PASS** |
| B1/E1・B2/E2 balance | 維持 | begin/end 同一スコープ・return 無し・`noexcept` | **PASS** |
| B3/E3（`reclaimNormal` +1/−1） | 変更なし | `Coordinator.cpp:705/716` 無 diff | **PASS** |
| `requestReclaim()` production callers | 2 | `AudioEngine.h:4614`、`AudioEngine.Retire.cpp:116` | **PASS** |
| caller-side epoch pre-check（`requestReclaimHandle`） | 0 | `minReaderEpoch` 読取 0・caller 判定 0（`retireEpoch` は `retireSequence` 捕捉のみ残置） | **PASS** |
| caller-side epoch pre-check（`drainDeferredRetireQueues`） | 0 | 同上（`isRetired` ガード・`requestReclaim`・`onReclaimEnd` のみ） | **PASS** |
| `retireEpoch < minReaderEpoch` の残存 | `Coordinator.cpp:700` のみ | `Coordinator.cpp:700` のみ（他はコメント） | **PASS** |
| `reclaimInFlightCount_` mutator | 2（+1/−1 のみ） | `Coordinator.cpp:212` / `:227` のみ。reset / clamp なし | **PASS** |
| 新規 raw `std::atomic` | 0 | 0 | **PASS** |
| RT path 変更 | 0 | 0 | **PASS** |
| Coordinator authority 変更 | 0 | 0（`git diff --name-only` に Coordinator ファイル無し） | **PASS** |
| 変更 src ファイル | RC-1 + test のみ | `AudioEngine.Retire.cpp`, `AudioEngine.h`, `DeferredPublicationTestAccess.h`, `PublishPipelineIntegrationTests.cpp`（+ 新規 `STG9ReclaimAccountingTests.cpp`） | **PASS** |

**静的検証は全項目 PASS。破綻は runtime accounting のみ。**

---

## 10. ISR / authority audit

| 検査項目 | 結果 |
| --- | --- |
| RT no-wait | **遵守**。RC-1 の変更は NonRT のみ（`requestReclaimHandle` は AudioEngine.h の AC-ISR-1 注記どおり Audio Thread から呼ばれない／`drainDeferredRetireQueues` は Timer・CoordinatorLoop・RebuildThread・MessageThread からのみ） |
| RT no-lock | **遵守**。`pendingReclaimHandlesMutex_` の使用範囲を**拡大していない**。追加 lock 0 |
| RT no-alloc | **遵守**。新規 allocation 0（`ReclaimIdentity` は stack POD） |
| RT no-delete | **遵守**。delete 0 |
| RT no-decision | **遵守**。判断を AudioEngine → Coordinator（`reclaimNormal`）へ**移設**した。RT 経路の判断を追加していない |
| Coordinator sole authority for reclaim | **強化**。epoch 判断が caller から Coordinator へ一本化（既存 AC-2 方針と同一） |
| Retire through Epoch | **不変**。`retireEpoch < minReaderEpoch` の意味論は `reclaimNormal` に保持。`ReclaimIdentity.retireSequence`（INV-FIFO-1 secondary）不変 |
| RuntimeWorld immutable | **遵守**。`RuntimePublishWorld` に触れない |
| Overflow ≠ silent loss | **遵守**。drop 時に `-1` する（loss ではなく deferred obligation の終端記録） |
| Shutdown = complete drain | **未達成**。§3 の accounts 欠落により counter が identity を上回り、`isFullyDrained()` が false のまま残る。**これが本 gate の FAIL 原因** |

**ISR 構造違反: 0 件。Authority 違反: 0 件。Shutdown 終端契約: 未達成（RC-1 不完全）。**

---

## 11. ConvoPeq.md regeneration

**NOT DONE** — §10 により「Debug/Release/CTest と STG-9-D1 gate が PASS した後にのみ再生成」。
本 gate は **FAIL** のため再生成していない。

| 項目 | 値（実装前後で不変） |
| --- | --- |
| SHA-256 | `615940DC90D73CED680081A8FBD0B491DBDC376788C0469B0A59AF7D2F6F2E3B` |
| size | 5,612,326 B |
| Generated | `2026-09-28 16:17:11` |
| HEAD | `b0b4694161817e705b21534777a2b2cf8085f0f0` |

`git status` の ` M ConvoPeq.md` は**実装前から存在した** `Generated:` 行 1 行の差分（HEAD blob は
15:58:53 / 5,489,292 B、改行コード正規化差を含む）であり、本 STG の変更ではない。

---

## 12. Git status

### 12.1 本 STG で変更したファイル（main repo / branch `main`）

```
 M CMakeLists.txt                                              (+1)
 M src/audioengine/AudioEngine.Retire.cpp                      (production, RC-1)
 M src/audioengine/AudioEngine.h                               (production, RC-1)
 M src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h  (test)
 M src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp (test)
?? src/tests/AudioEngineHarness/STG9ReclaimAccountingTests.cpp  (test, 新規)
?? doc/work113/P1-5-IR-P2_STG-9_DISCOVERY_20260928.md            (先行文書)
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-AUDIT_20260928.md (先行文書)
?? doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE_20260928.md      (本書)
```

`5 files changed, 79 insertions(+), 39 deletions(-)`（src + CMakeLists）。

### 12.2 本 STG が**触っていない**既存差分（commit 対象外・要非混入）

```
 M AGENTS.md
 M doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md
 M headroom-proxy-start.ps1
 M ConvoPeq.md                     ← 実装前から存在する Generated 行 1 行差
?? .memories/  ?? scripts/gen_clangdb.py  ?? vc140.pdb  ?? doc/ConvoPeqMD/  他 60+ 件の doc/work113 未追跡文書
```

これらはすべて STG-8 gate が既に「FAIL 条件に非該当」と判定した既存差分であり、**本 STG では
一切変更していない**。worktree 側の `output_sourcecode_markdown.py` 変更も main repo とは無関係。

---

## 13. Commit / Push

- **Commit = NOT AUTHORIZED**（未実施）
- **Push = NOT AUTHORIZED**（未実施）

---

## 14. Stop rule 判定

| stop rule | 該当 | 根拠 |
| --- | --- | --- |
| T-01〜T-04 の oracle が成立しない | **YES** | 4/4 FAIL（§4） |
| counter が pending より先に減る | NO | counter は pending より**多く**なる方向 |
| duplicate +1 が発生する | **YES** | retry ごとに `+1`、置換 entry の `+1` 未解放（§3.1） |
| pending が empty なのに counter > 0 | NO（現状は pending=1 / counter≈830） | |
| B1/E1 または B2/E2 が崩れる | NO | 静的検証 PASS |
| `reclaimShutdownQuiescent` への変更が必要になる | NO | 変更不要。drop 集約で吸収可能 |
| Coordinator 変更が必要になる | NO | 変更不要 |
| RT path 変更が必要になる | NO | 変更不要 |
| production 変更範囲が RC-1 から拡大する | NO | 追加分は `AudioEngine.Retire.cpp` 内の 1 行に収まる（§3.5） |
| authority が stale | NO | 実装前に内容レベル一致を確認済 |

**§12 に従い、修正を追加せず停止した。**

---

## 15. Owner への依頼事項

1. **RC-1 契約の訂正承認**: Contract Audit §4.2 I-2 の「再 push は counter 不変」が誤りであった。
   正しい契約は「**entry 1 件 = `reclaimInFlightCount_` +1 1 回**。再 push は『置換』なので
   置換される entry の `+1` を `onReclaimEnd()` で対で解放する」。
2. **`AudioEngine.Retire.cpp:121` への 1 行追加の GO**（§3.5）。production 変更範囲は拡大しない。
3. **T-03 oracle の再定義 GO**（§3.6）。baseline を「engine 全体が drained」ではなく
   「本シーケンスが導入した reclaim conjunct が baseline へ戻る」へ変更する。
4. **STG-8-D2 の間欠的失敗（§5.1）の扱い** — pre-existing フレークの切り分けを別 STG とするか、
   本 STG の RC-1 修正後に再評価するか。
5. Release ビルド / Release CTest は、上記 2・3 の GO 後に実施する。

**Commit / Push は引き続き NOT AUTHORIZED。**
