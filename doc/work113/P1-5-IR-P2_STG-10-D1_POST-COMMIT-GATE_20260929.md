# STG-10-D1 Post-Commit Gate（2026-09-29）

> **Verdict: STG-10-D1 COMMIT = PASS**
> **Push = NOT EXECUTED（NOT AUTHORIZED）**
> 本書は Owner GO に基づく Post-Implementation Gate（G10-P1〜P5）と commit、
> および Post-Commit Gate の実施記録である。

---

## 1. Pre-commit authority

Owner 指定: `ConvoPeq(20260928-152951).md` は `ConvoPeq.md` と統一 authority。
実 source 検証は repo の `ConvoPeq.md` を基準に実施した。

```text
HEAD            = 13a80d66e8e58fbcb66e71eb162f8e95fe134eeb
ConvoPeq SHA    = E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
ConvoPeq size   = 5,662,249 B
Generated       = 2026-09-29 00:19:43
NEWER_SRC_COUNT = 0
FRESH           = yes   (output_sourcecode_markdown.py --check exit 0,
                       "STATUS : FRESH — snapshot は現行ソースを反映しています")
```

Owner 提示 baseline との一致:

| 項目 | Owner baseline | 実測 | 一致 |
| --- | --- | --- | --- |
| SHA-256 | `E9FBD8F9…F29D09` | `E9FBD8F9…F29D09` | **YES** |
| size | 5,662,249 B | 5,662,249 B | **YES** |
| Generated | 2026-09-29 00:19:43 | 2026-09-29 00:19:43 | **YES** |
| NEWER_SRC_COUNT | 0 | 0 | **YES** |
| FRESH | yes | yes | **YES** |

---

## 2. G10-P1 — Authority

**PASS**。§1 のとおり 5 項目すべて一致。STOP 条件に該当しない。

---

## 3. G10-P2 — Worktree / diff containment

**PASS**。

### 3.1 tracked modifications（commit 時点、`git diff --name-only`）

```text
AGENTS.md                                                    ← pre-existing / 除外
CMakeLists.txt                                               ← STG-10-D1 対象
ConvoPeq.md                                                  ← STG-10-D1 対象（authority 再生成）
doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md   ← pre-existing / 除外
headroom-proxy-start.ps1                                     ← pre-existing / 除外
src/core/EpochDomain.h                                       ← STG-10-D1 対象
```

### 3.2 untracked（計 77 件）

| 分類 | 件数 | 扱い |
| --- | --- | --- |
| `src/tests/STG10ReaderQuarantineTests.cpp` | 1 | **STG-10-D1 対象**（commit 対象） |
| `.memories/` / `scripts/gen_clangdb.py` / `vc140.pdb` / `doc/ConvoPeqMD/` | 4 | **除外**（Owner 禁止項目） |
| `doc/work113/` の STG-10 文書 3 件 | 3 | 保持（Owner 指示により本次 commit 対象外） |
| `doc/work113/` のその他未追跡文書 | 64 | 除外（無関係） |
| その他 | 5 | 除外（無関係） |

### 3.3 containment 検証結果

| 検査 | 結果 |
| --- | --- |
| staging 前の staged 件数 | **0** |
| forbidden 項目の staging | **0 件**（10 項目を個別照合） |
| 意図しない production/test/build/document 差分の混入 | **なし** |

**`git add` は対象 4 パスを個別指定して実施し、禁止パスは絶対指定していない。**

---

## 4. G10-P3 — Production diff containment

**PASS**。

### 4.1 production source 変更ファイル数

```text
src/core/EpochDomain.h
count = 1   (1 = PASS)
```

### 4.2 関数単位のバイト一致検証（HEAD `13a80d66` との比較）

| 関数 | 結果 |
| --- | --- |
| `enterReader` | identical |
| `exitReader` | identical |
| `registerReaderThread` | identical |
| `reserveReaderThread` | identical |
| `quarantineReader` | identical |
| `unquarantineAllReaders` | identical |
| `publishEpoch` | identical |
| `currentEpoch` | identical |
| `tryReclaim` | identical |
| `drainAll` | identical |
| `enqueueRetire` | identical |
| `enqueueRetireTyped` | identical |
| `activeReaderCount` | identical |
| `quarantinedReaderCount` | identical |
| `getReaderSlotDetail` | identical |
| `detectStuckReaders` | identical |
| `verifyReaderInvariants` | identical |
| `readers`（member 宣言） | identical |
| **`getMinReaderEpoch`** | **DIFFERS（意図どおり RC-1 のみ）** |

`all non-RC-1 functions identical = True`

### 4.3 禁止ファイル（差分 0 を確認）

| ファイル | diff |
| --- | --- |
| `src/core/RCUReader.h` | 0 |
| `src/DeferredDeletionQueue.h` | 0 |
| `src/audioengine/ISRRetireRouter.h` / `.cpp` | 0 / 0 |
| `src/audioengine/ISRRuntimePublicationCoordinator.h` / `.cpp` | 0 / 0 |
| `src/audioengine/AudioEngine.h` | 0（STG-9-D1 保護） |
| `src/audioengine/AudioEngine.Retire.cpp` | 0（STG-9-D1 保護） |
| `src/audioengine/ISRShutdown.h` / `.cpp` | 0 / 0（Shutdown authority 保護） |
| `src/audioengine/AudioEngine.Threading.cpp` | 0（R-4 diagnostics 保護） |
| `src/audioengine/AudioEngine.CtorDtor.cpp` | 0（R-4 diagnostics 保護） |
| `src/core/IReaderEpochProvider.h` / `IEpochProvider.h` / `IRetireProvider.h` | 0 / 0 / 0（API / virtual interface 保護） |

### 4.4 RuntimeWorld / ownership surface

`runtimeworld` / `ownership` / `lifetime` / `worldauthor` を含むファイルの差分 = **0**。

### 4.5 production hunk 全量（コメント行を除いた実コード差分）

```diff
+            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
-            if ((flags & ReaderSlot::kQuarantinedFlag) != 0)
-            {
-                assert(convo::consumeAtomic(slot.depth, std::memory_order_acquire) == 0
-                    && "quarantined reader must have depth==0");
+            if ((flags & ReaderSlot::kQuarantinedFlag) != 0 && depth == 0)
-            }
-            const uint32_t depth = convo::consumeAtomic(slot.depth, std::memory_order_acquire);
```

| 検査 | 結果 |
| --- | --- |
| raw `std::atomic` method call の追加 | **0 件** |
| 追加行の `memory_order_*` | `memory_order_acquire` **1 件のみ**（既存パターンと同一）。変更された ordering 0 |
| RC-1 の契約 exceeded | **なし** |

---

## 5. G10-P4 — Test / CMake containment

**PASS**。

| 検査 | 結果 |
| --- | --- |
| 変更された既存 test ファイル | **0 件**（既存 oracle の改変なし） |
| 新規 test | `src/tests/STG10ReaderQuarantineTests.cpp`（454 行）のみ |
| 新規 test 関数 | 4 件（`testT10_1` / `testT10_2` / `testT10_3` / `testT10_4`） |
| CMakeLists.txt の削除行 | **0 行**（純追加） |
| CMake wiring | `add_executable` / `add_test(NAME STG10ReaderQuarantine)` / IPO-OFF / ASAN リスト登録のみ |
| production への test hook / friend 追加 | **0 件** |

---

## 6. G10-P5 — Regression evidence

**PASS**。取得済み証跡を gate として確認（Owner §2-P5 により再試験は不要）。

### 6.1 CTest

| | 結果 | 証跡 |
| --- | --- | --- |
| Debug CTest | **43/43 PASS**（196.50 s） | `100% tests passed out of 43` / Failed 行 0 |
| Release CTest | **43/43 PASS**（154.93 s） | `100% tests passed out of 43` / Failed 行 0 |

### 6.2 T10-1〜T10-4（gate 時点で Debug / Release 両バイナリを再実行）

```text
STG10ReaderQuarantineTests: T10-1 PASS (quarantined live reader contributes to minReaderEpoch)
STG10ReaderQuarantineTests: T10-2 PASS (slot reuse allowed, live occupancy protected)
STG10ReaderQuarantineTests: T10-3 PASS (pending->quarantined unblocks reclaim)
STG10ReaderQuarantineTests: T10-4 PASS (shutdown unquarantine + drainAll)
STG10ReaderQuarantineTests: PASS (T10-1/T10-2/T10-3/T10-4)
Debug   exit=0
Release exit=0
```

### 6.3 既存 STG regression（Debug / Release とも）

| STG | Debug | Release |
| --- | --- | --- |
| STG-1 | PASS | PASS |
| STG-2 | PASS | PASS |
| STG-2-A | PASS | PASS |
| STG-2-B | PASS | PASS |
| STG-2-C | PASS | PASS |
| STG-2-R | PASS | PASS |
| STG-4-1 | PASS | PASS |
| STG-6-D1 | PASS | PASS |
| STG-7-D1 | PASS | PASS |
| STG-8-D1 | PASS | PASS |
| STG-8-D2 | PASS | PASS |
| STG-8-D2b | PASS | PASS |
| STG-8-D3 | PASS | PASS |
| STG-9-D1（T-01 / T-02 / T-03 / T-04） | PASS | PASS |

両 harness とも exit 0、FAIL 行なし。

---

## 7. Commit

### 7.1 commit 前提条件

```text
staged               = intended 4 files only          →  YES
unrelated changes    = unstaged / untouched           →  YES
ConvoPeq.md          = FRESH                          →  YES
production diff      = RC-1 only                      →  YES
tests                = PASS                           →  YES
```

### 7.2 commit 結果

```text
hash    = f95f524cf6df0b3e1529425e690c02767912614e
short   = f95f524c
parent  = 13a80d66e8e58fbcb66e71eb162f8e95fe134eeb
subject = fix(epoch): prevent quarantined live readers from being excluded
author  = lonewolf-jp <lonewolf-jp@users.noreply.github.com>
date    = Tue Sep 29 00:53:02 2026 +0900
stats   = 4 files changed, 988 insertions(+), 25 deletions(-)
```

### 7.3 committed files（厳密に 4 件）

```text
CMakeLists.txt                                    |  22 ++
ConvoPeq.md                                       | 511 ++++++++++++++++++++++++++++++-
src/core/EpochDomain.h                            |  26 +-
src/tests/STG10ReaderQuarantineTests.cpp          | 454 +++++++++++++++++++++++++++
```

新規ファイルとして `src/tests/STG10ReaderQuarantineTests.cpp` が `create mode 100644` で追加。

### 7.4 commit に含まれなかったもの（禁止項目）

| 項目 | in-commit |
| --- | --- |
| `AGENTS.md` | **no（OK）** |
| `headroom-proxy-start.ps1` | **no（OK）** |
| `doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md` | **no（OK）** |
| `.memories/` | **no（OK）** |
| `scripts/gen_clangdb.py` | **no（OK）** |
| `vc140.pdb` | **no（OK）** |
| `doc/ConvoPeqMD/` | **no（OK）** |
| `doc/work113/P1-5-IR-P2_STG-10_DISCOVERY_20260928.md` | **no（保持・未追跡）** |
| `doc/work113/P1-5-IR-P2_STG-10_REPAIR-CONTRACT-AUDIT_20260928.md` | **no（保持・未追跡）** |
| `doc/work113/P1-5-IR-P2_STG-10_REPAIR-IMPLEMENTATION_20260928.md` | **no（保持・未追跡）** |

> **Owner への明示**: STG-10-D1 の 3 文書（Discovery / Repair Contract Audit /
> Repair Implementation）は Owner §2-G10-P2 の「4 ファイルだけに厳密に限定」指示に
>従い、**本 commit には含めず未追跡のまま保持した**。
> STG-9-D1 では同種の文書が commit に含まれていたが、今回の指示は明示的に 4 ファイルを
> 限定しているため指示を優先した。**これら 3 文書の追加 commit が必要であれば、
> Owner の指示を請うこと**（本 cycle では推測で追加しない）。

---

## 8. Post-Commit Gate

**PASS**。

| 検査 | 結果 |
| --- | --- |
| HEAD が新規 STG-10-D1 commit を指す | **YES**（`f95f524cf6df0b3e1529425e690c02767912614e` / parent `13a80d66`） |
| working tree に STG-10-D1 の意図しない差分が残っていない | **YES**。残る tracked 差分は §9 の 3 件のみ（すべて pre-existing） |
| STG-10-D1 commit に unrelated files が含まれていない | **YES**（§7.4） |
| `ConvoPeq.md` が commit 後の source と一致 | **YES**。`git diff HEAD -- ConvoPeq.md` = **empty**（commit 済み authority と完全同一） |
| `NEWER_SRC_COUNT` | **0** |
| `FRESH` | **yes**（`--check` exit 0） |

**post-commit authority（commit により source は変化していないため SHA / size /
Generated は commit 前と同一）**:

```text
SHA-256         = E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
size            = 5,662,249 B
Generated       = 2026-09-29 00:19:43
NEWER_SRC_COUNT = 0
FRESH           = yes
```

**Owner §4「commit 後に source が変化した場合は freshness を再確認」に対する回答:
commit は source content を変更していないため、再生成は不要であり authority は
commit 前と同一のまま FRESH を維持している。**

---

## 9. Working tree（commit 後）

```text
tracked modifications remaining (pre-existing / 無関係):
   M AGENTS.md
   M doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md
   M headroom-proxy-start.ps1

untracked = 76
   (STG-10 文書 3 件 + その他 doc/work113 64 件 + .memories/ + scripts/gen_clangdb.py
    + vc140.pdb + doc/ConvoPeqMD/ + その他)
   ※ STG10ReaderQuarantineTests.cpp は commit 済み（追跡済みになった）

staged  = 0
branch  = ## main...origin/main [ahead 9]
```

---

## 10. Push

```text
git push = NOT EXECUTED   (Owner §5 で明示的に禁止)
```

---

## 11. Residual（commit 後も未解決のまま保持）

STG-10-D1 の repair implementation 文書 §10 に記録した R1〜R8 をそのまま保持する。
**本 commit では R1〜R8 のいずれも修正していない。**

| # | 項目 | 状態 |
| --- | --- | --- |
| R1 | R-4 diagnostics（`collectDrainAudit` の reclaim 系 field 欠落、`[FAULT]` ログ文言） | 未着手。独立 P2/P3 |
| R2 | RC-2（allocation 時の flag reset） | 不採用（liveness 課題） |
| R3 | Path A（`quarantineReader` の TOCTOU） | Owner 指示どおり未修正。RC-1 により S6 は無害 |
| R4 | Candidate C（atomic state word 統合） | 不採用。将来課題 |
| R5 | `quarantinedReaderCount()` の production 消費者 0 件 | 未着手（observability） |
| R6 | `verifyReaderInvariants()` の呼出し元 0 件（死コード） | 未着手。RC-1 により S6 は無害化済み |
| R7 | `RCUReader::enter()` 失敗時の `rootEnterSucceeded()` 非検査 | 既存の別問題。scope 外 |
| R8 | STG-8-D1 / STG-8-D2 / `testRebuildPublishCompletes` の intermittent observation 3 件 | 再発なし。記録継続 |

**この 3 件の intermittent observation について**: STG-9-D1 Post-Commit Gate で記録した
4 件のうち、`EpochDomain.h:229` assert 連発 + stack overflow の 1 件は
**STG-10-D1 で根因を修正済み**（Audit §6 / Implementation §2）。残る 3 件は
STG-10-D1 と非交差であり、本 commit でも再発していない。

---

## 12. 最終報告

```text
STG-10-D1 Commit Gate

Pre-commit authority:
HEAD:            13a80d66e8e58fbcb66e71eb162f8e95fe134eeb
ConvoPeq SHA:    E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
ConvoPeq size:   5,662,249 B
Generated:       2026-09-29 00:19:43
NEWER_SRC_COUNT: 0
FRESH:           yes

G10-P1 Authority:             PASS
G10-P2 Diff containment:     PASS   (staged before = 0; 10 forbidden paths all unstaged)
G10-P3 Production containment: PASS  (1 file / 1 function; 18 functions byte-identical;
                                         15 forbidden files diff=0; raw atomic +0;
                                         memory_order change 0)
G10-P4 Test/CMake containment: PASS  (existing test oracle changes = 0; new TU 454 lines;
                                         CMake pure addition, 0 removed lines;
                                         production test hooks = 0)
G10-P5 Regression evidence:   PASS  (Debug 43/43, Release 43/43, T10-1..4 PASS,
                                         STG-1..9 all PASS in both configs)

Commit:
f95f524cf6df0b3e1529425e690c02767912614e
(parent 13a80d66e8e58fbcb66e71eb162f8e95fe134eeb)

Commit message:
fix(epoch): prevent quarantined live readers from being excluded

Committed files:
  CMakeLists.txt
  ConvoPeq.md
  src/core/EpochDomain.h
  src/tests/STG10ReaderQuarantineTests.cpp
(4 files changed, 988 insertions(+), 25 deletions(-))

Post-commit authority:
SHA:            E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
size:           5,662,249 B
Generated:      2026-09-29 00:19:43
NEWER_SRC_COUNT: 0
FRESH:          yes   (git diff HEAD -- ConvoPeq.md = empty)

Working tree:
  tracked:  M AGENTS.md
            M doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md
            M headroom-proxy-start.ps1
  untracked: 76 items (all pre-existing / unrelated; STG-10 docs 3 kept untracked)
  staged: 0
  branch: ## main...origin/main [ahead 9]

Push:
NOT EXECUTED

Verdict:
STG-10-D1 COMMIT = PASS
```

**`STG-10-D1 = CLOSED`。commit 完了。push は未実施。**
