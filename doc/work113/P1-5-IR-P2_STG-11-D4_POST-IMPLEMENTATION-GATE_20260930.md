# STG-11-D4 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D4_POST-IMPLEMENTATION-GATE_20260930.md`
- Work item: **STG-11-D4 / OBS-1** — Diagnostics ON 時の `applyMmcssPriority()` success path の RT-unsafe logging 除去
- Predecessor: `P1-5-IR-P2_STG-11-D4_REPAIR-CONTRACT-AUDIT_20260930.md`（GO） /
  `P1-5-IR-P2_STG-11-D4_REPAIR-IMPLEMENTATION_20260930.md`
- Date: 2026-09-30
- HEAD: `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- Branch: `main...origin/main [ahead 9]`, staged = 0

---

## 判定

```
READY FOR COMMIT
```

**commit / push は未実施。** Owner の commit GO を待つ。

---

## 1. Authority 再確認（§9）

### 1.1 着手時

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md`（Owner の統一ファイル規定による） |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `F441875191E30FA864B06C0538AA4E5A2B775470E0FA49990A59F65E726CD4C0` |
| size | 5,761,584 B |
| Generated | `2026-09-30 01:42:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH`（exit 0） |

Owner 指示の authority 名 `ConvoPeq(20260930-111230).md` はリポジトリ内に存在しない
（`doc/ConvoPeqMD/` の実在 10 スナップショットに該当なし）。
D3 第 2 版の OBS-4 と同一のため、同様にルート `ConvoPeq.md` を authority とした。
古い snapshot は authority にしていない。不整合なしのため proceed。

### 1.2 完了後（再生成後）

| 項目 | 値 |
| --- | --- |
| SHA-256 | `678B5BA15658A7AEDCED97D0E6A59EC2D3AF4C9BBE107350EF4710CC9B3DF57C` |
| size | 5,805,842 B（着手時比 +44,258 B） |
| Generated | `2026-09-30 21:09:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH`（exit 0） |

`ConvoPeq.md` に D4 の production 2 ファイル・test TU・test access がすべて反映されていることを確認
（`recordSuccessObserved` / `reportSuccessIfRecorded` / `successObserved_` 等の出現）。
D1（`m_ownedRetireRouter { m_epochDomain }`）/ D2（`quarantineRetireSink`）/
D3（`fetchAddAtomic(affinityFailureCount_, ...)`）も引き続き反映済み。
`ConvoPeq.md` は編集していない（生成物として再生成しただけ）。

---

## 2. Core repair の判定

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| RT → lock-free atomic observation のみ | **PASS** | `recordSuccessObserved` は `convo::publishAtomic` / `convo::fetchAddAtomic` のみ |
| RT から `diagLog()` 除去（S1/S2/S3） | **PASS** | 旧文言 3 種が RT 関数内に存在しない（D4-T1）。`if (nativeRtOk)` block と S2/S3 周辺に backend token なし |
| RT から `std::mutex` 除去 | **PASS** | 同上 ＋ recorder body に mutex / lock 不在 |
| RT から `juce::String` 構築除去 | **PASS** | 同上 |
| RT から heap alloc / free 除去 | **PASS** | 同上 |
| RT から logging backend 除去 | **PASS** | 同上（`asyncSink` / `s_logMutex` / `Logger::` / `DBG(` / `flushLogBuffer`） |
| RT から blocking 除去 | **PASS** | 上記が RT 到達面全体 |
| 成功ログの意味・内容の維持 | **PASS** | 3 文言と同一（+ `count=` suffix）。`toHexString` 整形は NonRT 側に移動 |
| success policy 不変 | **PASS** | `priorityApplied` に影響しない（informational のまま） |
| `priorityApplied` / caller handling / `t_mmcssTried` 不変 | **PASS** | 変更なし |
| Observation identity（3 種の区別） | **PASS** | kind 1/2/3。D4 機能 test が 5 ラウンドで対応を一意に検証 |
| payload（必要情報のみ） | **PASS** | kind 1: prio/procClass/savedClass、kind 2: audioMask/prevMask、kind 3: なし（Contract §3.3 どおり） |
| 新規 thread / timer / queue / authority なし | **PASS** | authority audit 0 件。report は既存 `timerCallback` の 1 行追加のみ |

---

## 3. Atomic rule の判定

| 項目 | 結果 |
| --- | --- |
| D3+D4 追加行の raw `std::atomic` API | **0 件** |
| D4 の atomic 操作 | **全件** `convo::publishAtomic` / `convo::fetchAddAtomic` / `convo::consumeAtomic` |
| memory-order | 既存 convention（`release` / `acquire` / count は `acq_rel`） |
| 新設 synchronization 構造 | **0 件** |

D4 atomic member は **7 件**（Contract §4 の表どおり。D3 の 6 件とは別物）。

---

## 4. 不変条件の判定

| ID | 判定 | 実測 |
| --- | --- | --- |
| INV-D4-1 mutex 取得なし | **PASS** | `S1/S2/S3 RT-safe (backend-free, record-only)` |
| INV-D4-2 alloc / free なし | **PASS** | 該当 0 |
| INV-D4-3 OFF が前提でない | **PASS** | (a) success 分岐は guard に関係なく検査、(b) record 呼び出し 3 件が guard 外、(c) D4-T4（残存 backend token 全件 guard 内） |
| INV-D4-4 新 authority なし / ownership なし | **PASS** | 該当 0。輸送物は整数のみ |
| INV-D4-5 lossy-coalescing（payload → count → observed） | **PASS** | `recorder = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)` |
| INV-D4-5 read order（count acquire → snapshot） | **PASS** | `report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)` |
| INV-D4-5 古い snapshot の誤確定なし | **PASS** | 5 ラウンド全 PASS。`reportedCount` は毎回ちょうど +1 |
| D1 / D2 invariant | **PASS** | TD1 / TD2 全 PASS（値も不変） |
| D3 invariant（failure transport・ordering・policy） | **PASS** | D3-T1/T1b/T2/T3/T5 全 PASS ＋ D4-T6（read-only 境界検査）PASS |

---

## 5. Diagnostics ON / OFF の判定

| 象限 | 判定 | 根拠 |
| --- | --- | --- |
| **OFF + success/failure** | **RT-safe** | D4-T4: body 内の残存 backend token 全件 guard 内。関数全体が mutex / sink を直接触らない |
| **ON + success** | **RT-safe + 既存文言で診断** | D4-T1（RT 側 backend 不在）＋ D4-T2/T3（NonRT report が原文言＋guard 内） |
| **ON + failure** | **RT-safe（D3 のまま）** | D3-T1/T4 PASS（不変） |

### Negative control（oracle の実効性）

S2 を旧 `diagLog` へ**一時的に復元**し再 build:

```
D4-T1: success wording '[AFFINITY] AudioThread pinned mask=0x' still inside RT function
        (ON success path reaches logging backend)
FAIL: D4-T1 success sites backend-free
harness rc = 1
```

→ oracle は欠陥を正確に捕捉する。復元後の再 build・再実行で全 PASS に戻ることを確認。
実ファイルで `recordSuccessObserved(1/2/3)` の存在を再確認した。

---

## 6. Regression の判定（§8）

### 6.1 D4 結果（Debug / Release、harness 直接実行、rc=0 両方）

```
D4-T1 PASS (success sites backend-free, ON/OFF independent)
D4-T2/D4-T3 PASS (original wording kept, guarded, NonRT)
D4-T4 PASS (Diagnostics OFF semantics kept)
D4 functional PASS (record + once-per-count NonRT diagnosis)
D4-T6 PASS (D3 failure transport untouched)
PASS (D4-T1/D4-T2/D4-T3/D4-T4/D4-T5/D4-T6)
```

### 6.2 D3 regression（ordering correction 込み）

```
D3-T1/D3-T4 PASS / D3-T1b PASS / D3-T2/D3-T4 PASS / D3-T3 PASS / D3-T5 PASS
PASS (D3-T1/D3-T2/D3-T3/D3-T4/D3-T5)
```

**D3-T3 を含む全件 PASS。** success 文言は report 関数内（guard 内）に移動したため、
D3-T3 の存在＋guard 検査は引き続き成立する。D3 のファイル・関数・member・test・順序は
一切変更していない（D4-T6 が read-only で保証）。

### 6.3 D1 regression（`STG11EQRetireTests`, Debug）

```
TD1-1 PASS (Q=304) / TD1-2 PASS (T=280) / TD1-3 / TD1-4a / TD1-4b PASS   rc=0
```

D/Q/E/T = 4096 / 512 / 512 / 280、total 5400（**不変**）。

### 6.4 D2 regression（`STG11D2SnapshotRetireTests`, Debug）

```
TD2-1 PASS (E=3) / TD2-2 PASS (T=3) / TD2-3 PASS (全 0)   rc=0
```

### 6.5 full Debug CTest

```
100% tests passed out of 45
```

### 6.6 full Release CTest

```
100% tests passed out of 45
```

### 6.7 raw std::atomic audit

D3+D4 追加行に raw API **0 件**。

### 6.8 RT lock/allocation/backend audit

`recordSuccessObserved` 本体 **0 件**。

### 6.9 authority audit

新規 authority / 新規 synchronization 構造 **0 件**。

### 6.10 静的解析

| ツール | 対象 | 結果 |
| --- | --- | --- |
| clang-tidy | `AudioEngine.Timer.cpp` | D4 hunk 内 finding **0**。残存 2+6 件はいずれも hunk 外の既存事象。`easily-swappable` は根拠付き NOLINT |
| cppcheck 2.22 | project DB | project 全体 135 件は既存ノイズ。**D4 production 2 ファイル 0 件** |

### 6.11 ASAN

環境 block（`0xC0000139`）。従来どおり別問題として記録。
代替 evidence: 構造的 oracle（D4-T1/T5a/T5b）＋ negative control ＋ Debug/Release 45/45 ＋
exact-count oracle（D1/D2 不変）＋ deterministic 可視性契約（D4 5 ラウンド）＋静的解析。

---

## 7. Diff scope の判定

### 7.1 変更ファイル一覧

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Timer.cpp` | 191 | 23 | **D4**（D3 分を含む累積） |
| `src/audioengine/AudioEngine.h` | 34 | 0 | **D4** 19 行 + D3 15 行 |
| `src/tests/AudioEngineHarness/STG11D4SuccessObservationTests.cpp` | 新規（untracked） | — | **D4** |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 103 | 0 | D1 + D3 + **D4**（53 行） |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 21 | 0 | D1 + D2 + D3 + **D4**（5 行） |
| `CMakeLists.txt` | 134 | 0 | D1 + D2 + D3 + **D4**（1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存・保持） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存・保持） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存・保持） |
| `ConvoPeq.md` | 3634 | 234 | 再生成 |

### 7.2 禁止領域の変更なし（`git status --porcelain` が空であることを確認）

`ISRRetireRouter.h` / `ISRRetireRouter.cpp` / `EpochDomain.h` / `EpochDomain.cpp` /
`ISRRuntimePublicationCoordinator.h` / `ISRRuntimePublicationCoordinator.cpp` /
`ISRCoordinatorLoop.cpp` / `PublicationAdmission.h` / `RuntimeStore.h` /
`AudioEngine.Mmcss.cpp` — **すべて未変更**。

`Crossfade` / `Publish` / `Retire` / `Recovery` 関連ファイルにも差分なし。

### 7.3 D1 / D2 / D3 の保持

D1（`EQProcessor.*`）、D2（`SnapshotCoordinator.cpp`）、D3（failure transport・member・test・
ordering・policy）の変更はそのまま保持されている。D3 境界は D4-T6 が read-only で保証する。

### 7.4 pre-existing worktree 変更

`AGENTS.md` / `headroom-proxy-start.ps1` / STG-8 gate 文書には**触れていない**。

---

## 8. 観測事項（未修正・次工程の投入候補）

### OBS-D4-1: `AudioEngine.Mmcss.cpp` の MMCSS 登録ログ（RT 到達のまま）

`[MMCSS-*] registered` / `already registered` / `FAILED` / `reverted` は同一 first-call
RT 経路から file-local `diagLog` を呼ぶ。Contract §6 の限定により scope 外。
次回 Discovery 候補。

### OBS-D4-2: `revertMmcssPriorityOnAudioThread()` / `finalizeMmcssShutdown()` の guarded `diagLog`

同じく scope 外。次回 Discovery 候補。

### OBS-D4-3: 既存 test の intermittent（R8）

復元直後の直接実行で `0xC0000005` を 1 件観測（28 行で I2T 内に停止。D4/D3 より遥か前）。
再実行で rc=0・全 PASS。D3 Gate の OBS-5 と同一 phenomenology。
**証拠から D4 への帰属は立証できない。** 未解決の既知 flake として記録し、
R8 の調査対象に含めることを推奨する。

---

## 9. Gate 判定

| Gate 項目 | 判定 |
| --- | --- |
| Authority 再確認（着手時・再生成後） | PASS |
| Core repair（RT-safe success observation のみ） | PASS |
| 成功ログの意味・内容の維持 | PASS |
| success policy / priorityApplied / caller handling 不変 | PASS |
| Observation identity（3 種の区別） | PASS |
| payload（必要情報のみ） | PASS |
| Atomic rule（raw `std::atomic` 追加 0） | PASS |
| NonRT side（既存 execution point・新規 thread なし） | PASS |
| Diagnostics ON/OFF 双方 | PASS（negative control で実証） |
| INV-D4-1 .. INV-D4-5 | PASS |
| D1 / D2 / D3 invariant | PASS |
| D4-T1 / T2 / T3 / T4 / T5 / T6 | PASS（Debug / Release 両方） |
| Negative control | PASS |
| D1 / D2 / D3 regression | PASS |
| full Debug CTest | PASS 45/45 |
| full Release CTest | PASS 45/45 |
| raw std::atomic audit | PASS |
| RT lock/allocation/backend audit | PASS |
| authority audit | PASS |
| 静的解析 | PASS |
| ConvoPeq regeneration + freshness | PASS（FRESH, NEWER_SRC_COUNT=0） |
| Diff scope（禁止領域 0 変更） | PASS |
| D1 / D2 / D3 の保持 | PASS |
| commit / push | **未実施** |

```
READY FOR COMMIT
```

---

## 10. commit 時に Owner が確定すべき境界（D3 Gate §10 を継承・更新）

現在の作業ツリーには **D1 + D2 + D3 + D4** の未 commit 変更が同時に乗っている。

| 選択肢 | 内容 |
| --- | --- |
| A | D1 + D2 + D3 + D4 を 1 つの commit にまとめる（同一 STG の連続修正であり、CMakeLists / test wiring を共有するため分割コストが高い） |
| B | 複数コミットに分割する（D1 → D2 → D3 → D4）。ただし `DeferredPublicationTestAccess.h` / `PublishPipelineIntegrationTests.cpp` / `CMakeLists.txt` が 4 件にまたがるため、各コミットが中間状態で build 不能になる |
| C | D1 + D2 を先に commit し、D3 + D4 を別 commit にする（`ConvoPeq.md` の再生成をどちらに含めるかで diff が分かれる） |

Owner が指示を出すまでは、追加 commit を行わない。

**push は引き続き一切禁止。**
