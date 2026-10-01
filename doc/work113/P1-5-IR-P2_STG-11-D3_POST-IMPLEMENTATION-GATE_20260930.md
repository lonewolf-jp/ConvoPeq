# STG-11-D3 Post-Implementation Gate

- Document: `doc/work113/P1-5-IR-P2_STG-11-D3_POST-IMPLEMENTATION-GATE_20260930.md`
- Work item: **STG-11-D3 / Candidate A** — MMCSS / CPU-affinity failure path の RT-unsafe logging 除去
- Predecessor: `P1-5-IR-P2_STG-11-D3_REPAIR-CONTRACT-AUDIT_20260929.md` / `P1-5-IR-P2_STG-11-D3_REPAIR-IMPLEMENTATION_20260930.md`
- Date: 2026-09-30
- Revision: **第 2 版（publication ordering correction 反映後）** — 2026-09-30 01:45
- HEAD: `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- Branch: `main...origin/main [ahead 9]`, staged = 0

---

## 判定

```
READY FOR COMMIT
```

**commit / push は未実施。** Owner の commit GO を待つ。

---

## 0. 第 2 版での変更点（Owner 指摘への対応）

Owner は「`recordAffinityFailure()` が `observed` を payload より先に publish しており、
`observed` の release/acquire を payload 全体の publication synchronization として
使うには順序が逆」と指摘した。**指摘は正しい。** release store が publish するのは
その store より sequenced-before された書き込みだけであり、旧順序では payload の
store が marker より後にあったため acquire にhappens-before edge が含まれなかった。

第 2 版で以下を実施した:

1. **recorder の publication 順序を修正** — payload 3 種 → count（`fetchAddAtomic` / acq_rel）→ observed
2. **D3-T5a / T5b / T5c を追加** — structural な順序 oracle と deterministic な可視性契約
3. **Negative control #2 を追加** — 旧順序に一時戻すと D3-T5 が FAIL することを実証
4. **clang-tidy の指摘に対応** — `bugprone-easily-swappable-parameters` に根拠付き NOLINT（OBS-3）
5. **記録の訂正** — D3 atomic member を **6 件** と明記（Owner §6）
6. **authority を再取得・再生成**（§1）

第 1 版から不変の項目: production 変更ファイル、failure policy、`priorityApplied`、
caller handling、D1 / D2 の変更、禁止領域の無変更。

---

## 1. Authority 再確認（§1 / §9）

### 1.1 初回 implementation 開始時（第 1 版）

| 項目 | 値 |
| --- | --- |
| authority file | `ConvoPeq(20260929-124301).md` == `ConvoPeq.md` |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE` |
| size | 5,710,250 B |
| Generated | `2026-09-29 21:16:18` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH — snapshot は現行ソースを反映しています` |

→ production source と `ConvoPeq.md` の不整合なし。停止条件に該当しない。

### 1.2 ordering correction 着手時（第 2 版、Owner 提示の authority 名）

Owner §1 が提示した authority 名 `ConvoPeq(20260929-153558).md` は
リポジトリ内に**存在しない**（`doc/ConvoPeqMD/` の実在 9 スナップショット =
`20260928104322` / `20260928143345` / `20260928155853` / `20260928161811` /
`20260928221921` / `20260929001943` / `20260929193552` / `20260929211618` /
`20260930001956`）。OBS-4 参照。

Owner は同時に「`ConvoPeq(20260929-153558).md` はプロジェクトルートの
`ConvoPeq.md` と統一ファイルである」と規定しているため、
リポジトリルート `ConvoPeq.md` を authority として再取得した。

| 項目 | 値 |
| --- | --- |
| authority file | リポジトリルート `ConvoPeq.md`（Owner §1 の統一ファイル規定による） |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `1F94D3B40B0AE6537BF72D24A2713ADE779B1F87AEFDAFD32BDB7D55A1F3E1F8` |
| size | 5,742,476 B |
| Generated | `2026-09-30 00:19:56` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH — snapshot は現行ソースを反映しています` |

この authority の内容は Owner §2 の描述（`recordAffinityFailure` が `observed` を
先に publish する順序）と**完全に一致**していた。指摘された defect は authority 上に
実在した。停止条件に該当せず proceed。

### 1.3 ordering correction 完了後（再生成後、§9）

| 項目 | 値 |
| --- | --- |
| SHA-256 | `F441875191E30FA864B06C0538AA4E5A2B775470E0FA49990A59F65E726CD4C0` |
| size | 5,761,584 B（correction 前比 +19,108 B） |
| Generated | `2026-09-30 01:42:22` |
| NEWER_SRC_COUNT | 0 |
| `--check` | `STATUS : FRESH — snapshot は現行ソースを反映しています`（exit 0） |

`ConvoPeq.md` に D3 の production 2 ファイル・test TU・test access がすべて反映され、
**corrected 順序の `convo::fetchAddAtomic(affinityFailureCount_, ...)` が
line 21888 に存在**することを確認（marker が payload の後にあることの副証）。
D1（`m_ownedRetireRouter { m_epochDomain }`）/ D2（`SnapshotCoordinator::quarantineRetireSink`）
も引き続き反映済み。`ConvoPeq.md` は編集していない（生成物として再生成しただけ）。

---

## 2. Core repair の判定（§3 / §6 / §7）

| 項目 | 判定 | 根拠 |
| --- | --- | --- |
| RT → lock-free atomic record のみ | **PASS** | `recordAffinityFailure` は `convo::publishAtomic` / `convo::fetchAddAtomic` のみ。OS 呼び出し・分岐・循環なし |
| RT から `diagLog()` 除去 | **PASS** | F1/F2/F4 の failure 分岐に `diagLog` 不在（brace 抽出） |
| RT から `std::mutex` 除去 | **PASS** | 同上 ＋ recorder body に mutex / lock_guard 不在 |
| RT から `juce::String` 構築除去 | **PASS** | 同上 |
| RT から heap alloc / free 除去 | **PASS** | 同上（`new ` / `make_unique` / `make_shared` / `malloc` / `calloc` / `realloc` / `free(`） |
| RT から logging backend 除去 | **PASS** | 同上 ＋ `asyncSink` / `s_logMutex` / `Logger::` / `DBG(` / `flushLogBuffer` / `OutputDebugString` |
| RT から blocking 除去 | **PASS** | 上記が RT 到達面全体 |
| failure policy 不変 | **PASS** | `if (!nativeRtOk) { priorityApplied = false; }` 保持。affinity failure は `priorityApplied` に影響しない（従来どおり informational） |
| `priorityApplied` の意味不変 | **PASS** | 変更なし |
| caller の return value handling 不変 | **PASS** | `AudioEngine.Mmcss.cpp` 等 caller 未変更（diff に現れない） |
| `t_mmcssTried` semantics 不変 | **PASS** | 未変更 |
| fallback policy 不変 | **PASS** | 未変更 |
| MMCSS policy の再設計をしない | **PASS** | `AudioEngine.Mmcss.cpp` 未変更。`applyMmcssPriority` の分岐構造も不変 |

---

## 3. Atomic rule の判定（§4）

| 項目 | 結果 |
| --- | --- |
| D3 追加行に含まれる raw `std::atomic` API（`.load/.store/.fetch_add/.fetch_sub/.fetch_or/.fetch_xor/.exchange/.compare_exchange`） | **0 件** |
| D3 が実行する atomic 操作 | 12 箇所、**全件** `convo::publishAtomic` / `convo::fetchAddAtomic` / `convo::consumeAtomic` |
| memory-order | 既存 convention に一致（`release` で publish、`acquire` で load、count は `acq_rel`） |
| 新設した同期機構 | **0**。sequence counter / mutex / queue / CAS loop のいずれも新設していない（Owner §4） |

D3 atomic member は **6 件**（Owner §6 の訂正を反映）:

| # | member | 役割 | writer | reader |
| --- | --- | --- | --- | --- |
| 1 | `affinityFailureObserved_` | 単調 bool flag（fast-out 専用） | RT | NonRT |
| 2 | `affinityFailureCount_` | 単調 count ＋ **per-record publication marker** | RT | NonRT |
| 3 | `affinityFailureLastMask_` | payload（last-wins） | RT | NonRT |
| 4 | `affinityFailureLastError_` | payload（last-wins） | RT | NonRT |
| 5 | `affinityFailureLastKind_` | payload（last-wins） | RT | NonRT |
| 6 | `affinityFailureReportedCount_` | NonRT 専用 once-only 記録 | NonRT のみ | NonRT |

※ `AudioEngine.Timer.cpp` / `AudioEngine.h` には**既存の** raw atomic 使用が存在するが、
D3 が**新規に追加していない**ことは diff スコープで証明済み（§5）。

---

## 4. 不変条件の判定（§9）

| ID | 判定 | 検証手段 | 実測 |
| --- | --- | --- | --- |
| INV-D3-1 mutex 取得なし | **PASS** | F1/F2/F4 の brace 抽出で `std::mutex` / `lock_guard` / `asyncSink` / `s_logMutex` を検査 ＋ recorder body に同検査 | 3 分岐すべて `RT-safe (backend-free, record-only)` |
| INV-D3-2 alloc / free なし | **PASS** | 同 brace 抽出で `juce::String` / `new ` / `make_unique` / `make_shared` / `malloc` / `calloc` / `realloc` / `free(` を検査 | 該当 0 |
| INV-D3-3 Diagnostics OFF が前提でない | **PASS** | (a) failure 分岐は **guard に関係なく**禁止 token を検査、(b) `recordAffinityFailure` 呼び出しが diagnostics guard 内でないことを検査、(c) success 側の全 `diagLog` / `juce::String` が guard 内であることを検査 | (a)(b)(c) とも充足。`recordAffinityFailure call is inside a diagnostics guard` は発生せず |
| INV-D3-4 新 authority を作らない | **PASS** | recorder body に `std::thread` / `std::jthread` / `juce::Timer` / `LockFreeRingBuffer` / `enqueueDeferredDelete` / `enqueueRetire` / `DeferredDeletionQueue` / `RuntimePublication` / `Recovery` が無いこと | 該当 0。呼び出しは既存 `timerCallback()` 1 箇所のみ |
| INV-D3-5 明示的 lossy-coalescing / silent ownership loss 禁止 | **PASS** | 輸送物は `count`（単調増加）+ last-wins snapshot（mask / error / kind）のみ。`RuntimeWorld` / `Snapshot` / `Retire object` / `Publication object` / `Recovery obligation` を輸送しないため coalescing でも ownership を失わない。`reportedCount` により coalescing が診断上可視化される | `D3-T2` で 3 連続 record → 3 回とも `reportedCount` 進展（1 → 2 → 3）、重複診断なし |
| **INV-D3-5 副条件: publication order（Owner 指摘）** | **PASS（第 1 版は FAIL）** | D3-T5a が recorder の atomic access 順序を構造検証。payload → count → observed であること、`count` が `fetchAddAtomic` であること、`observed` が最後の access であること | `recorder order = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)` |
| **INV-D3-5 副条件: read order** | **PASS** | D3-T5b が report の read 順序を構造検証。snapshot の read が `count` の acquire より後であること、全 read が `convo::consumeAtomic` + `memory_order_acquire` であること | `report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)` |
| **INV-D3-5 副条件: 古い snapshot を新しい count として確定しない** | **PASS** | D3-T5c が 5 ラウンドで `count == k` かつ snapshot == record #k の値を検証 | 5 ラウンド全 PASS。`reportedCount` は毎回ちょうど +1 進展 |
| D1 / D2 invariant 維持 | **PASS** | §7 の D1 / D2 回帰 | TD1-1 / TD1-2 / TD1-3 / TD1-4a / TD1-4b、TD2-1 / TD2-2 / TD2-3 すべて PASS（値も不変） |

---

## 5. Diagnostics ON / OFF の判定（§8）

| 象限 | 判定 | 根拠 |
| --- | --- | --- |
| **OFF + success** | **RT-safe** | D3-T1b: `applyMmcssPriority` 内の全 `diagLog(` / `juce::String` が guard 内。OFF でコンパイルされる文に backend 到達なし。関数全体が mutex / SPSC sink を直接触らない |
| **OFF + failure** | **RT-safe** | D3-T1: F1/F2/F4 が backend / lock / allocation を含まない。record 呼び出しは guard 外なので OFF でも記録される |
| **ON + success** | **existing behavior** | D3-T3: 既存 guarded 成功ログ 3 種すべて保持、guard 位置も不変 |
| **ON + failure** | **RT-safe** | D3-T1 は guard の有無に依存せず failure 分岐の文を検査するため、ON でも OFF と同じ判定が成立 |

**「Diagnostics OFF だから安全」ではない**ことを構造的に保証:
判定の単位は guard ではなく「failure 分岐に含まれる文」であるため、
guard を外しても（ON でも）同じ禁止 token 不在が成立することを意味する。

### Negative control #1 — RT-unsafe logging の再導入

F4 を原始欠陥（無条件 `diagLog`）へ**一時的に復元**し再 build:

```
D3-T1: F4 SetThreadAffinityMask failure reaches RT-forbidden backend 'diagLog('
FAIL: D3-T1/D3-T4 RT failure paths backend-free
D3-T1b: 'diagLog(' reachable with Diagnostics OFF
FAIL: D3-T1b diagnostics-OFF success path
harness rc = 1
```

### Negative control #2 — publication 逆順の再導入（Owner 指摘の defect そのもの）

recorder を旧順序（`observed` 先頭、`payload` 末尾）へ**一時的に復元**し再 build:

```
D3-T5: payload affinityFailureLastKind_ is published AFTER the count increment
        (index 2 > 1): the release/acquire pair would not cover it
FAIL: D3-T5 recorder publication order
harness rc = 1
```

→ 順序 oracle は Owner が指摘した欠陥を**正確に**捕捉する。両 negative control の
復元後の再 build・再実行で D3-T1 / T1b / T2 / T3 / T5a / T5b / T5c は全 PASS に
戻ることを確認。実ファイルで recorder の順序が
`LastKind_ → LastMask_ → LastError_ → Count_(fetchAdd/acq_rel) → Observed_`
であることを再確認した。

### 順序 oracle の信頼性

順序判定は `stripComments()`（quote-aware な `//` / `/* */` 除去）を先に適用してから
`convo::` wrapper 呼び出しを source 順に列挙して行うため、**コメント文字列に
依存せず実 code のみ**を見る。Owner 指示の「古い snapshot を新しい count の
snapshot として誤って確定しない」がコメント経由で誤判定されることはない。

---

## 6. Regression の判定（§11）

### 6.1 D1 回帰（`STG11EQRetireTests`, Debug）

```
TD1-1 PASS (D->Q retained across returns, Q=304 drained)
  [fill 2500] pending=4096 Q+E=904 E=392 T=0 drop=0
TD1-2 PASS (E/T retained across returns, T=280 drained)
TD1-3 PASS (release leaves no residue)
TD1-4a PASS (router bound to private domain)
TD1-4b PASS (reclaim gated only on bound provider)
PASS (TD1-1/TD1-2/TD1-3/TD1-4a/TD1-4b)   rc=0
```

D/Q/E/T = 4096 / 512 / 512 / 280、total 5400（**不変**）。

### 6.2 D2 回帰（`STG11D2SnapshotRetireTests`, Debug）

```
TD2-1 PASS (Q-full escalates to E, E=3 drained)
TD2-2 PASS (E-full escalates to T, T=3 drained)
TD2-3 PASS (shutdown drain completes)
PASS (TD2-1/TD2-2/TD2-3)   rc=0
```

### 6.3 D3 結果

Debug（`AudioEngineHarness.exe` 直接実行, rc=0）:

```
D3-T1/D3-T4 PASS (RT failure paths backend-free, ON/OFF independent)
D3-T1b PASS (Diagnostics OFF success path has no backend)
D3-T2/D3-T4 PASS (record + once-per-count NonRT diagnosis)
D3-T3 PASS (existing guarded success logs preserved)
D3-T5 PASS (payload published before count, count monotonic, snapshot last-wins)
PASS (D3-T1/D3-T2/D3-T3/D3-T4/D3-T5)
```

Release（`AudioEngineHarness.exe` 直接実行, rc=0）: **同 5 行すべて PASS**。
branch 別の詳細:

```
D3-T1: F1 SetPriorityClass failure            RT-safe (backend-free, record-only)
D3-T1: F2 SetThreadPriority failure           RT-safe (backend-free, record-only)
D3-T1: F4 SetThreadAffinityMask failure       RT-safe (backend-free, record-only)
D3-T1: recordAffinityFailure = convo:: wrappers only, no backend/allocation
D3-T1: NonRT report uses existing diagLog + existing timerCallback (no new authority)
D3-T1: lossy-coalescing semantics explicit (monotonic count + last-wins snapshot)
D3-T5: recorder order = payload -> count(fetchAdd/acq_rel) -> observed(publish/release)
D3-T5: report order = observed(fast-out) -> count(acquire) -> snapshot(acquire)
```

### 6.4 full Debug CTest

```
100% tests passed out of 45
```

### 6.5 full Release CTest

```
100% tests passed out of 45
```

### 6.6 raw std::atomic audit

D3 追加行に対する grep: raw API **0 件**。atomic 操作 12 箇所すべて `convo::` wrapper。

### 6.7 RT allocation / lock audit

recorder 本体（`AudioEngine.Timer.cpp` の `recordAffinityFailure` 全体）に対する grep
（`std::mutex` / `lock_guard` / `unique_lock` / `shared_lock` / `condition_variable` /
`new` / `make_unique` / `make_shared` / `malloc` / `calloc` / `realloc` / `free(` /
`juce::String` / `asyncSink` / `s_logMutex` / `Logger::` / `diagLog` / `DBG(`）:
**0 件**。D3 追加行全体では、非RT `reportAffinityFailureIfRecorded()` の
`diagLog(juce::String...)` とコメントのみが該当。

### 6.8 authority audit

新規 publication / retire / recovery / thread / timer / worker / queue authority **0 件**。
新規 synchronization 構造（sequence counter / mutex / queue / CAS loop）も **0 件**。

### 6.9 静的解析

| ツール | 対象 | 結果 |
| --- | --- | --- |
| clang-tidy | `AudioEngine.Timer.cpp`（JSON-array compile DB） | D3 追加行の finding **0**。残存 `bugprone-branch-clone` ×2（1260, 1291）と `bugprone-unchecked-optional-access` ×6（2046-2073）はいずれも **D3 の diff hunk 外**（D3 は 252-394 と 1011-1015）の既存事象。`bugprone-easily-swappable-parameters` は OBS-3 のとおり根拠付き NOLINT で対応済み |
| cppcheck 2.22 | project DB、`-i` で D3 TU 限定 | project 全体 135 件は**すべて既存ノイズ**（`uninitMemberVarNoCtor` / `throwInEntryPoint` が無関係ヘッダ由来）。**D3 production 2 ファイルは 0 件** |

### 6.10 ASAN

環境 block（`0xC0000139`、pre-main の `ntdll!LdrGetProcedureAddressForCaller` forwarder 解決失敗）。
**ASAN failure と implementation failure を混同していない。** 従来どおり別問題として記録する。
代替 evidence:

1. 構造的 oracle（brace 抽出による禁止 token 不在）— Debug / Release 両方で PASS
2. 構造的 ordering oracle（D3-T5a / T5b）— Debug / Release 両方で PASS
3. negative control ×2（#1 RT-unsafe logging、#2 publication 逆順。いずれも FAIL を実証）
4. Debug / Release 2 構成で 45/45 PASS
5. exact-count oracle（D1: Q=304 / D/Q/E/T=4096/512/512/280、D2: E=3 / T=3 / 0）が不変
6. deterministic 可視性契約（D3-T5c の 5 ラウンド）
7. 静的解析（clang-tidy D3 追加行 0 / cppcheck D3 2 ファイル 0）
8. 完全 drain の確認（TD1-3 / TD2-3 が residual 0 を assert）

---

## 7. Diff scope の判定（§13）

### 7.1 変更ファイル一覧

| ファイル | + | − | 帰属 |
| --- | --- | --- | --- |
| `src/audioengine/AudioEngine.Timer.cpp` | 99 | 11 | **D3**（ordering correction 込み） |
| `src/audioengine/AudioEngine.h` | 15 | 0 | **D3** |
| `src/tests/AudioEngineHarness/STG11D3AffinityFailureTests.cpp` | 新規（untracked） | — | **D3** |
| `src/tests/AudioEngineHarness/DeferredPublicationTestAccess.h` | 50 | 0 | D1 + **D3** |
| `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 16 | 0 | D1 + D2 + **D3** |
| `CMakeLists.txt` | 133 | 0 | D1 + D2 + **D3**（`STG11D3AffinityFailureTests.cpp` 1 行） |
| `src/core/SnapshotCoordinator.cpp` | 21 | 3 | D2（既存・保持） |
| `src/eqprocessor/EQProcessor.Core.cpp` | 37 | 10 | D1（既存・保持） |
| `src/eqprocessor/EQProcessor.h` | 39 | 0 | D1（既存・保持） |
| `ConvoPeq.md` | 2299 | 25 | 再生成 |

D3 の diff hunk は `AudioEngine.Timer.cpp` の `@@ -252,270`（`applyMmcssPriority`
の failure 分岐 3 箇所）と recorder/report 定義の追加、および `@@ -927,1011`
（`timerCallback` の NonRT 診断呼び出し 1 箇所）のみ。

### 7.2 禁止領域の変更なし（`git status --porcelain` が空であることを確認）

`ISRRetireRouter.h` / `ISRRetireRouter.cpp` / `EpochDomain.h` / `EpochDomain.cpp` /
`ISRRuntimePublicationCoordinator.h` / `ISRRuntimePublicationCoordinator.cpp` /
`ISRCoordinatorLoop.cpp` / `PublicationAdmission.h` / `RuntimeStore.h` — **すべて未変更**。

`Crossfade` / `Publish` / `Retire` / `Recovery` 関連ファイルにも差分なし。

### 7.3 D1 / D2 の保持

D1（`EQProcessor.*`）と D2（`SnapshotCoordinator.cpp`）の変更はそのまま保持されている。
D3 の baseline 測定のために一時的に D3 だけを除去した実験を行ったが、
除去範囲は D3 の 6 ファイルに限定し、除去後に D1 / D2 の diff が `git diff --stat` 上
`SnapshotCoordinator.cpp` / `EQProcessor.Core.cpp` / `EQProcessor.h` として
**同一内容で存在することを確認**した上でバックアップから完全復元した。

### 7.4 pre-existing worktree 変更

`AGENTS.md` / `headroom-proxy-start.ps1` /
`doc/work113/P1-5-IR-P2_STG-8-D1-D3_REPAIR-GATE_20260928.md` には
**触れていない**（整理・stash・reset・削除なし）。

---

## 8. 観測事項（未修正・次工程の投入候補）

### OBS-1: Diagnostics ON 時の success path は依然 RT から logging backend を呼ぶ

`applyMmcssPriority()` の guard 内 success ログ 3 経路
（`[NATIVE_RT] applied` / `[AFFINITY] AudioThread pinned` / `[AFFINITY] P/E cores`）は
Diagnostics ON で RT → `diagLog` → `asyncSink` → mutex → heap に到達する。
Owner §8 が「Diagnostics ON / success → existing behavior」と規定し §7 は failure policy のみを
不変対象とするため、**D3 の修復範囲外**。guard 構造は一切動かしていない。
次回の Discovery 候補（STG-11-D5 相当）として記録する。

### OBS-3: `recordAffinityFailure` の 3 引数が交換可能（clang-tidy が指摘）

clang-tidy が `bugprone-easily-swappable-parameters` で
`recordAffinityFailure(kind, audioMask, error)` を指摘した。3 引数すべて
`std::uint32_t` / `DWORD_PTR` / `DWORD` で互いに暗黙変換可能であり、入れ違えても
コンパイルが通る。

**対応**: 根拠コメント付き `// NOLINT(bugprone-easily-swappable-parameters)` を付与。
**引数順序は変更していない**（Owner の publication 順序指示と独立した判断であり、
順序変更は diff を拡大するため行わない）。

安全性は呼び出し側で担保されている: production の呼び出しは failure 分岐 3 箇所のみで、
いずれも第 1 引数にリテラル（`0` / `1` / `2`）を渡すため、交換しても `kind` 値が
0/1/2 以外になりえない。加えて D3-T5c は 5 ラウンドの各回で
`(kind, mask, error)` の対応を一意に検証している。

### OBS-4: Owner 提示の authority スナップショット名 `ConvoPeq(20260929-153558).md` は存在しない

Owner 指示 §1 の authority 名 `ConvoPeq(20260929-153558).md` はリポジトリ内に
**存在しない**。`doc/ConvoPeqMD/` の実在スナップショットは
`20260928104322` / `20260928143345` / `20260928155853` / `20260928161811` /
`20260928221921` / `20260929001943` / `20260929193552` / `20260929211618` /
`20260930001956` の **9 件**である。

Owner は同時に「`ConvoPeq(20260929-153558).md` はプロジェクトルートの `ConvoPeq.md`
と統一ファイルである」と規定しているため、**リポジトリルート `ConvoPeq.md` を唯一の
source authority として作業した**（§1.2）。その内容は Owner が §2 で描述した実装
（`recordAffinityFailure` が `observed` を先に publish する順序）と完全に一致していた。
**事実の記録のみ。追加の採取は行っていない。**

### OBS-5: 既存 `ConvolverStateRoundTripTests` / `STG8RecoveryObligationTests` の flake

**観測事実（時系列）**

| 条件 | 結果 |
| --- | --- |
| D3 構築 / CTest run 1 | `AudioEngineHarness` **SEGFAULT**（`[STG-6-D1] probe:` の直後、`ConvolverStateRoundTripTests` 内） |
| D3 構築 / CTest run 2 | D3 test helper の自己バグ（D3-T1 / T1b 誤検出）。**修正済み** |
| D3 構築 / CTest run 3 | `[STG-8-D2b] FAIL: over-terminalized (L=1 hasDeferred=0)` |
| **D3 除去（baseline）/ CTest run 1-5** | **5/5 PASS** |
| D3 構築 / CTest run 4 | `AudioEngineHarness` **SEGFAULT**（run 1 と同署名） |
| D3 構築 / CTest run 5 | 45/45 PASS |
| D3 構築 / CTest run 6-10 | **5/5 PASS** |
| D3 構築 / harness 直接実行 | 4/4 PASS |
| D3 構築 / procdump 付き直接実行 | 8/8 PASS |
| D3 構築 / 他 43 test 実行後の直接実行 | 4/4 PASS |
| procdump による crash dump 取得 | 12 回試行して **未再現**（初回のみ `0x406D1388` = JUCE `setCurrentThreadName` の benign first-chance 例外を捕獲） |
| **ordering correction 構築 / CTest ×2（Debug + Release）** | **45/45 PASS 両方** |
| **ordering correction 構築 / harness 直接実行 ×2** | **rc=0 両方**（Debug / Release） |

**判定**

- 失敗は **D3 sub-test より前**の test（`ConvolverStateRoundTripTests` の STG-6、
  `STG8RecoveryObligationTests` の STG-8-D2b）で発生している。D3 sub-test は `main()` 上で
  `runSTG8RecoveryObligationTests()` / `runSTG9ReclaimAccountingTests()` の**後**に呼ばれるため、
  **D3 test コードでは説明できない**。
- D3 の production 変更が RT に影響し得る唯一の面は `timerCallback()` に追加した
  `reportAffinityFailureIfRecorded()` 1 呼び出しで、これは
  「未記録なら atomic を 1 回 load して return」という**副作用ゼロ**の経路である
  （store / lock / allocation / 分岐なし）。
- D3 除去ベースラインは 5/5 PASS であったが、D3 構築でも後続 5/5 PASS + 直接実行 12/12 PASS、
  さらに ordering correction 構築でも Debug / Release 両方で 45/45 であり、初期窓（run 1〜5）に集中した。
- crash dump が 12 回試行で取得できず、faulting frame の特定できていない。

**結論: 証拠から D3 への帰属は立証できない。** ただし「無関係」と断定することもできないため、
本 Gate では **D3 を失敗扱いにしないが、未解決の既知 flake として明示的に記録**し、
R8（intermittent 3 件）の調査対象に含めることを推奨する。
再現時の crash dump が取れた時点で本 OBS を再判定する。
Owner 指示により D4 / N4 / OBS-1 の修正には**進んでいない**。

---

## 9. Gate 判定

| Gate 項目 | 判定 |
| --- | --- |
| Authority 再確認（初回・correction 着手時・再生成後） | PASS |
| Core repair（RT-safe observation のみ） | PASS |
| failure policy / priorityApplied / caller handling 不変 | PASS |
| Atomic rule（raw `std::atomic` 追加 0） | PASS |
| Diagnostic state semantics（queue/retire ownership を持たない） | PASS |
| NonRT side（既存 execution point・新規 thread なし） | PASS |
| Diagnostics ON/OFF 双方 | PASS（negative control #1 で実証） |
| INV-D3-1 .. INV-D3-5 | PASS |
| **publication order: payload → count → observed** | **PASS（Owner 指摘の defect を修正。negative control #2 で実証）** |
| **read order: count acquire → snapshot acquire** | **PASS** |
| **count 単調 / snapshot last-wins / 古い snapshot の誤確定なし** | **PASS**（D3-T5c 5 ラウンド） |
| 新設 synchronization 構造（sequence counter / mutex / queue / CAS loop） | 0 件 |
| D1 invariant | PASS |
| D2 invariant | PASS |
| D3-T1 / T2 / T3 / T4（既存 4 象限、**全件維持**） | PASS（Debug / Release 両方） |
| D3-T5a / T5b / T5c（ordering、追加分） | PASS（Debug / Release 両方） |
| Negative control ×2 | PASS（#1 RT-unsafe logging、#2 publication 逆順。両方 FAIL を実証） |
| D1 regression | PASS |
| D2 regression | PASS |
| full Debug CTest | PASS 45/45 |
| full Release CTest | PASS 45/45 |
| raw std::atomic audit | PASS |
| RT allocation / lock audit | PASS |
| authority audit | PASS |
| 静的解析 | PASS（clang-tidy D3 追加行 0、cppcheck D3 2 ファイル 0） |
| ConvoPeq regeneration + freshness | PASS（FRESH, NEWER_SRC_COUNT=0） |
| Diff scope（禁止領域 0 変更） | PASS |
| D1 / D2 の保持 | PASS |
| commit / push | **未実施** |

```
READY FOR COMMIT
```

---

## 10. commit 時に Owner が確定すべき境界

現在の作業ツリーには **D1 + D2 + D3** の未 commit 変更が同時に乗っている。

| 選択肢 | 内容 |
| --- | --- |
| A | D1 + D2 + D3 を 1 つの commit にまとめる（`fix(isr): ...` 系。3 件が同一 STG の連続修正であり、CMakeLists / test wiring を共有するため分割コストが高い） |
| B | 3 コミットに分割する（D1 → D2 → D3）。ただし `DeferredPublicationTestAccess.h` / `PublishPipelineIntegrationTests.cpp` / `CMakeLists.txt` が 3 件にまたがるため、各コミットが中間状態で build 不能になる |
| C | D1 + D2 を先に commit し、D3 を別 commit にする（`ConvoPeq.md` の再生成をどちらに含めるかで diff が分かれる） |

Owner が「4 ファイル厳格限定」の指示を出すまでは、追加 commit を行わない。

**push は引き続き一切禁止。**
