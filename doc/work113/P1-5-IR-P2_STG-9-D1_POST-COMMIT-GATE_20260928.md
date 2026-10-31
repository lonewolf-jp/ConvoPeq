# STG-9-D1 Post-Commit Gate（2026-09-28）

> **Verdict: STG-9-D1 = CLOSED**
> **Push = NOT AUTHORIZED（未実施）**
> 本書は Owner 指示 §4（commit 後 post-commit gate）の実施記録である。

---

## 0. Authority 前提の確定

Owner 指示: **`ConvoPeq(10).md` は `ConvoPeq.md` と同一の統一 authority として扱う。**

本 gate の全検証をこの前提で実施した。`ConvoPeq.md` を唯一の最新ソース基準として扱い、
authority の生成・鮮度判定・source 一致確認のすべてをこれに対して行った。

---

## 1. Commit 実施

### 1.1 Commit 前の最終確認（Owner §2）

| # | 確認項目 | 結果 |
| --- | --- | --- |
| 1 | working tree / staged diff の確認 | 実施。staged は空 |
| 2 | STG-9-D1 対象以外が stage されていないこと | **OK**（`git add` は対象 12 パスを個別指定。禁止 7 項目は絶対指定せず） |
| 3 | `ConvoPeq.md` が FRESH | **YES**（`--check` FRESH / NEWER_SRC_COUNT 0） |
| 4 | production diff が報告済み内容から変化していないこと | **OK**（`git diff --stat` が Gate-R2 §3 と一致） |
| 5 | R2 の 1 行修正の順序が `push_back` → `onReclaimEnd` | **OK**（`L126: push_back` → `L129: onReclaimEnd`） |

### 1.2 Commit

```text
[main 13a80d66] fix(runtime): repair reclaim accounting across deferred retry
 12 files changed, 3299 insertions(+), 79 deletions(-)
```

commit 対象（Owner 指定どおり 12 ファイル）:

```text
CMakeLists.txt
ConvoPeq.md
doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-AUDIT_20260928.md
doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-CONTRACT-CORRECTION_20260928.md
doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE-R2_20260928.md
doc/work113/P1-5-IR-P2_STG-9-D1_REPAIR-GATE_20260928.md
doc/work113/P1-5-IR-P2_STG-9_DISCOVERY_20260928.md
src/audioengine/AudioEngine.Retire.cpp
src/audioengine/AudioEngine.h
src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h
src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp
src/tests/AudioEngineHarness/STG9ReclaimAccountingTests.cpp
```

### 1.3 非混入の確認

commit 済みファイル集合に対し `AGENTS.md` / `headroom-proxy-start.ps1` / `.memories/` /
`scripts/gen_clangdb.py` / `vc140.pdb` / `doc/ConvoPeqMD/` / `STG-8-D1-D3_REPAIR-GATE` を
正規表現で照合。**該当なし**。禁止項目および STG-9-D1 と無関係な未追跡ファイルは全て除外された。

---

## 2. A. Git

| 項目 | 結果 |
| --- | --- |
| commit 成功 | **YES**（`13a80d66e8e58fbcb66e71eb162f8e95fe134eeb`） |
| HEAD が新 commit を指す | **YES**（`13a80d66`） |
| staged changes = empty | **YES**（0 files） |
| STG-9-D1 対象外の混入 | **なし** |
| Push | **未実施**（NOT AUTHORIZED） |

`git status -sb` = `## main...origin/main [ahead 8]`。Push は行っていない。

---

## 3. B. Authority（commit 後再生成）

`ConvoPeq.md` を最新 production source から再生成し、`--check` で検証した。

| 項目 | 値 |
| --- | --- |
| SHA-256 | `B15640ADCE15A38801DA8192652876EE4CA8EC6113247EEF9EB93E9EA754D7F3` |
| size | 5,642,888 B |
| Generated timestamp | `2026-09-28 22:19:21` |
| HEAD | `13a80d66e8e58fbcb66e71eb162f8e95fe134eeb` |
| NEWER_SRC_COUNT | **0** |
| FRESH | **YES**（`STATUS: FRESH` / `check exit=0`） |

**FRESH を確認。停止条件に該当しない。**

`ConvoPeq.md` の HEAD に対する差分は再生成による `Generated:` 行 1 行のみ
（`1 file changed, 1 insertion(+), 1 deletion(-)`）。

---

## 4. C. Post-commit regression

### 4.1 Debug CTest

| 項目 | 値 |
| --- | --- |
| 結果 | **42/42 PASS**（`100% tests passed out of 42`） |
| Failed lines | **0** |
| Total time | 210.90 sec |

### 4.2 Release CTest

| 項目 | 値 |
| --- | --- |
| 結果 | **42/42 PASS**（`100% tests passed out of 42`） |
| Failed lines | **0** |
| Total time | 150.48 sec |

### 4.3 STG-9-D1 T-01〜T-04（Debug / Release 両方）

```text
STG9ReclaimAccountingTests: T-01 PASS (quarantine drop releases counter)
STG9ReclaimAccountingTests: T-02 PASS (destroy drop releases counter)
STG9ReclaimAccountingTests: T-03 PASS (baseline return oracle)
STG9ReclaimAccountingTests: T-04 PASS (INV-1/INV-2 property matrix)
STG9ReclaimAccountingTests: PASS (T-01/T-02/T-03/T-04)
```

Debug harness `exit=0` / Release harness `exit=0`。両 configuration で全 4 件 PASS。

### 4.4 STG-1 / STG-2 / STG-4-1 / STG-6-D1 / STG-7-D1（Debug / Release 両方）

```text
ConvolverStateRoundTripTests: PASS (STG-1 bypass staging preservation 6/6)
ConvolverStateRoundTripTests: PASS (STG-2 StateIO round-trip 3/3)
ConvolverStateRoundTripTests: PASS (STG-2-A order transition ceiling holds)
ConvolverStateRoundTripTests: PASS (STG-2-B bypass transition ceiling holds)
ConvolverStateRoundTripTests: PASS (STG-2-C save/load idempotent)
ConvolverStateRoundTripTests: PASS (STG-2-R R4/R5/R6 preservation+stability)
ConvolverStateRoundTripTests: PASS (STG-4-1 no stale overwrite)
ConvolverStateRoundTripTests: PASS (STG-6-D1 failure reported, state preserved)
ConvolverStateRoundTripTests: PASS (STG-7-D1 autoGain flag round-trip)
```

### 4.5 STG-8-D1 / D2 / D2b / D3（Debug / Release 両方）

```text
STG8RecoveryObligationTests: PASS (STG-8-D1 discard terminalizes recovery)
STG8RecoveryObligationTests: PASS (STG-8-D2 overwrite terminalizes evicted)
STG8RecoveryObligationTests: PASS (STG-8-D2b same obligation retained)
STG8RecoveryObligationTests: PASS (STG-8-D3 pressure signal redrives)
```

**既存 test の oracle は一切変更していない**（Gate-R2 で確認済みのとおり、
本 commit の test 差分は新規 `STG9ReclaimAccountingTests.cpp` と
既存 friend への test-only accessor 追加・呼び出し配線のみ。既存 assertion の期待値改変は 0 件）。

---

## 5. D. Intermittent observation の再発判定

Owner 指示 §4-D に従い、以下の 4 項目について post-commit regression での再発を記録する。

| 観測項目 | pre-commit での発生 | **post-commit での再発** |
| --- | --- | --- |
| `STG-8-D2`（`checkSTG82OverwriteTerminalizesEvicted`）一時 FAIL | R1 期間に 1 回 | **再発なし**（Debug CTest / Release CTest / Debug harness / Release harness の全 4 実行で PASS） |
| `STG-8-D1`（`checkSTG81DiscardTerminalizesRecovery`）一時 FAIL | R2 Release 初回に 1 回（`discard leaked (hasDeferred=0 L=1)` / `L resurrected to 1`） | **再発なし**（同上 4 実行で PASS） |
| `EpochDomain.h:229` assert 連発 + stack overflow (`0xC00000FD`) | R2 Debug 期間に 1 回（STG-6-D1 領域） | **再発なし**（同上 4 実行で PASS） |
| `testRebuildPublishCompletes`（`audio thread did not run`）一時 FAIL | R2 Release 前 Debug に 1 回 | **再発なし**（同上 4 実行で PASS） |

**いずれも再発していない。** Owner 指示 §4-D の「再発した場合のみ記録して STOP」に該当しないため、
これらは **STG-9-D1 の修正と切り離された独立事項**（`STG-8-D2 intermittent observation` /
`STG-8-D1 intermittent observation` / `EpochDomain assert intermittent observation` /
`testRebuildPublishCompletes intermittent observation`）として記録のみ行う。
**本 gate では何も修正していない。**

code-path 交差の記録（Owner §4-D-5 に対応）:

| 観測 | STG-9-D1 変更との交差 |
| --- | --- |
| STG-8-D1/D2/D2b/D3 | `RuntimePublicationOrchestrator` の deferred slot と `RecoveryAdmissionTable` の obligation。STG-9-D1 が変更するのは `pendingReclaimHandles_` の retry 会計のみであり、両者は別 domain。**交差なし** |
| `EpochDomain.h:229` assert | `getMinReaderEpoch()` の reader quarantine flag 検証。STG-9-D1 は epoch 判定そのものを変更せず（Coordinator 側へ一本化）、reader 状態を触らない。**交差なし** |
| `testRebuildPublishCompletes` | audio thread スケジューリング。STG-9-D1 は NonRT のみ。**交差なし** |

---

## 6. E. STG-9-D1 終了判定

| 条件 | 結果 |
| --- | --- |
| commit 済み | **YES**（`13a80d66`） |
| post-commit authority FRESH | **YES**（NEWER_SRC_COUNT 0 / FRESH） |
| Debug 42/42 | **YES** |
| Release 42/42 | **YES** |
| T-01〜T-04 PASS | **YES**（Debug / Release 両方） |
| regression PASS | **YES**（STG-1 / STG-2 / STG-2-A/B/C / STG-2-R / STG-4-1 / STG-6-D1 / STG-7-D1 / STG-8-D1 / D2 / D2b / D3 全 PASS） |
| STG-9-D1 production diff が想定外に変化していない | **YES**（commit 内 production diff = `AudioEngine.h` + `AudioEngine.Retire.cpp` のみ。R2 の 1 行は `push_back` → `onReclaimEnd` の順序。working tree と HEAD は同一） |
| Push 未実施 | **YES** |

# **STG-9-D1 = CLOSED**

**STG-9-D1 の作業を終了する。**

---

## 7. 次のバグ修正作業

Owner 指示 §5 に従い、**次の production 修正は開始していない。**
次期 STG 開始時は、統一 authority `ConvoPeq.md`（SHA `B15640AD...4D7F3` /
HEAD `13a80d66`）と `doc/work113/` の最新 bug ledger を確認し、
既存の gate 方式（read-only discovery → concrete defect → Repair Contract Audit → Owner GO →
implementation → regression → authority regeneration → commit）を維持する。

**Push は引き続き NOT AUTHORIZED。**
