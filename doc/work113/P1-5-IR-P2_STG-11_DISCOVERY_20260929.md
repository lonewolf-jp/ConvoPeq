# STG-11 Concrete Defect Discovery（2026-09-29）

> **Verdict: read-only Discovery COMPLETE。Owner の次段階 GO なしに実装へ進まない。**
> **production / test / CMake / ConvoPeq.md 変更 = 0。commit = 0。push = 0。**

---

## 1. Authority

Owner 指定: `ConvoPeq(20260928-152951).md` は `ConvoPeq.md` と統一ファイル。
実 source 検証は repo の現行 `ConvoPeq.md` を基準に実施した。

| 項目 | Owner baseline（STG-10-D1 post-commit） | 実測 | 一致 |
| --- | --- | --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` | `f95f524cf6df0b3e1529425e690c02767912614e` | **YES** |
| SHA-256 | `E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09` | 同左 | **YES** |
| size | 5,662,249 B | 5,662,249 B | **YES** |
| Generated | 2026-09-29 00:19:43 | 2026-09-29 00:19:43 | **YES** |
| NEWER_SRC_COUNT | 0 | 0 | **YES** |
| STATUS | FRESH | FRESH（`--check` exit 0） | **YES** |

**完全一致。STOP 条件 1・2 に該当せず。**

---

## 2. Scope

```text
対象  : C:\VSC_Project\ConvoPeq（main, HEAD f95f524c）
authority : ConvoPeq.md（統一 authority）
方式  : read-only 構造探索 + source 実読による到達性・寿命の証明
変更  : production 0 / test 0 / CMake 0 / ConvoPeq.md 0 / commit 0 / push 0
```

使用: `ConvoPeq.md`（唯一 authority）、`src/**` 実読、`rg`（WSL）、
`git`（HEAD 比較）、subagent 2 本（breadth 用。**全ての主張を自分で再検証済み**）。

---

## 3. STG-1〜STG-10 CLOSED boundary = **PASS**

| STG | commit | 本 STG での非交差確認 |
| --- | --- | --- |
| STG-1 / 2 / 2-A/B/C/R | — | `ConvolverProcessor` staging / StateIO / order ceiling。**本 STG の D1〜D3 と非交差** |
| STG-4-1 | — | stale overwrite。`RuntimePublicationOrchestrator` 領域。**非交差** |
| STG-6-D1 | — | SilentIR 失敗報告。IR load 領域。**非交差** |
| STG-7-D1 | — | autoGain flag。**非交差** |
| STG-8-D1/D2/D2b/D3 | — | `RuntimePublicationOrchestrator` deferred slot ＋ `RecoveryAdmissionTable` obligation。**D1〜D3 と別 domain** |
| STG-9-D1 | `13a80d66` | `AudioEngine.h` / `AudioEngine.Retire.cpp` の `pendingReclaimHandles_` 会計。**D1〜D3 は当該 2 ファイルを触らない** |
| STG-10-D1 | `f95f524c` | `EpochDomain::getMinReaderEpoch()` の除外条件。**D1〜D3 は同関数を触らない** |

**reopen した項目: 0 件。**（§5 の D2 のみ `SnapshotCoordinator` という既存ファイル，涉及するが
§10 で「既知 defect の再発見ではない」ことを source で証明する。）

---

## 4. R1〜R8 reassessment

**この表自体を bug の存在証明として扱わない。** 各項目に source evidence を付して分類する。

| ID | 分類 | source evidence |
| --- | --- | --- |
| **R1** R-4 diagnostics | **Observability Gap**（＋ *contract-vs-usage の緊張*。Concrete Defect **ではない**） | `RuntimeDrainAudit.h:12` に「`isAllZero()` は監査ログ出力専用。shutdown 完了判定の authority にはしない」と明記。一方 `ReleaseResources.cpp:703` は `if (!drainedWithinBudget \|\| !audit.isAllZero())` と分岐している。**決定的な分岐本体は本 Discovery で全文読了していない**ため、shutdown 完了判定を誤らせる瑕疵は**証明できていない**。Owner §6 の 7 問に対する回答: (1) runtime correctness への影響 = なし（純粋な観測構造体）、(2) shutdown 完了判定の誤り = **未証明**、(3) fault classification の誤り = なし、(4) ownership/reclaim 状態の誤認 = `isAllZero()` が診断ログ専用なら誤認しない、(5) observable contract 違反 = **未証明**、(6) diagnostics 改善余地 = **これ**(最確度)、(7) 既存 invariant 違反 = **未証明**。**したがって repair target にしない。** |
| **R2** RC-2（allocation 時 flag reset） | **Already Safe / Mitigated** | STG-10-D1 の RC-1 適用後、`quarantined && depth>0` は通常 active reader として minEpoch に寄与する（`EpochDomain.h:234`）。flag の stale 残存は **安全性に影響しない**。liveness のみが対象。**bug ではない** |
| **R3** Path A（TOCTOU） | **Already Safe / Mitigated** | STG-10 Audit §10.3 の補題により、S6 は**生成経路によらず無害**。`quarantineReader:286` の read と `:294/:302` の CAS の間に enter が入っても、minEpoch を大きくし得ない。**bug ではない** |
| **R4** Candidate C（atomic state word 統合） | **Design Debt** | STG-10 Audit §9.3 で RT 経路の大規模書き換えと memory ordering 再設計が必要と判定。**不改変でも安全**。P3 |
| **R5** `quarantinedReaderCount()` の production 消費者 0 | **Observability Gap** | 定義 `EpochDomain.h:339`、委譲 `ISRRetireRouter.h:276-278` のみ。**production 呼び出し元 0 件**。`quarantine()` が既存 reader を誤隔離しても検知できない。P2 |
| **R6** `verifyReaderInvariants()` 呼出し元 0 | **Observability Gap / Design Debt** | 定義 `EpochDomain.h:353` のみ、**呼出し元 0 件**（死コード）。RC-1 により S6 は無害。P3 |
| **R7** `RCUReader::enter()` failure handling | **Potential Defect（要検証）** | §6 D4 参照。**source で 2 つの事実が確定**（(a) close 後に既存 reader が slot を再取得できない／(b) RT 2 経路が `rootEnterSucceeded()` を確認しない）。残る「その窓で audio callback が実際に走るか」は**未証明**。**最も優先すべき Audit 項目** |
| **R8** intermittent observation 3 件 | **Not Reproducible** | STG-10-D1 post-commit 4 実行で再発なし。`EpochDomain.h:229` の 1 件は STG-10-D1 で**根因修正済み**。残り 3 件は非交差・記録継続 |

---

## 5. New findings — 概要

| ID | 分類 | Severity |
| --- | --- | --- |
| **D1** | `EQProcessor` の stack-local `ISRRetireRouter` による Q/E/T ownership loss | **P1（確定）** |
| **D2** | `SnapshotCoordinator::quarantineRetireSink` の Q-full が assert のみ（終端解決なし） | **P1（確定）** |
| **D3** | `applyMmcssPriority` 失敗分岐が RT で mutex + heap allocation | **P1（確定）** |
| **D4** | `closeReaderRegistration()` が RT reader の slot 再取得を壊す（governing comment と矛盾） | **Potential（要検証）** |

**最高確定 severity: P1。**（P0 ではない — 各項目の consequence を §6 で source evidence と共に示す。）

---

## 6. Concrete defect proof

### 6.1 D1 — `EQProcessor` stack-local router による Q/E/T ownership loss（**P1 / 確定**）

#### Invariant violated

`ISRRetireRouter.cpp:315-319` が宣言する ownership 契約:

> ★ P-4: Ownership chain: D → Q → EmergencyQ → TerminalReclaimAuthority
> Ownership invariant: ptr を手放す前に、必ず次の authority に ownership が移る。
> ★ P-4: TerminalReclaimAuthority は growable store のため常に ownership を受領する。

→ **「次の authority」が関数寿命に束縛されている場合、この契約は構造的に成立しない。**

#### Exact source locations

| 要素 | file:line |
| --- | --- |
| stack-local router 構築 | `src/eqprocessor/EQProcessor.Core.cpp:49` |
| coordinator への router 受け渡し | `src/eqprocessor/EQProcessor.Core.cpp:52-55` |
| coordinator の完全委譲 | `src/audioengine/ISRRuntimePublicationCoordinator.cpp:147-172`（`const auto result = router.enqueueWithRetry(...)`） |
| 2 回目の retry | `src/eqprocessor/EQProcessor.Core.cpp:61` |
| 「成功」と返す判定 | `src/eqprocessor/EQProcessor.Core.cpp:65-67` |
| Q 昇格先（router の member） | `src/audioengine/ISRRetireRouter.cpp:342` → `ISRRetireRouter.h:401` |
| E 昇格先（router の member） | `src/audioengine/ISRRetireRouter.cpp:352` → `ISRRetireRouter.h:403` |
| T 昇格先（router の member） | `src/audioengine/ISRRetireRouter.cpp:365` → `ISRRetireRouter.h:405` |
| EQProcessor の private EBR domain | `src/eqprocessor/EQProcessor.h:488` `convo::EpochDomain m_epochDomain;` |
| 注入済みだが未使用の router | `src/eqprocessor/EQProcessor.h:461-463` 代入 / `:494` 宣言（**読み取り 0 件**） |

#### Caller / Callee

```text
caller : EQProcessor::retireEQStateDeferred   EQProcessor.Core.cpp:98-105
         EQProcessor::retireBandNodeDeferred   EQProcessor.Core.cpp:107-114
         ← EQProcessor.Parameters.cpp:29,49,69,93,118,139,173,196,222,257
         ← EQProcessor.Coefficients.cpp:75
         ← EQProcessor.Core.cpp:142,148,242,739
         ← ~EQProcessor               EQProcessor.Core.cpp:142,148
callee: EQProcessor::enqueueDeferredDeleteWithFallback  EQProcessor.Core.cpp:26-68
         └─ RuntimeIntentCoordinator::enqueueRetire      ISRRuntimePublicationCoordinator.cpp:147-172
              └─ ISRRetireRouter::enqueueWithRetry       ISRRetireRouter.cpp:303-384
```

#### State before → transition → state after

```text
before : state != nullptr, 所有者 = EQProcessor::retireEQStateDeferred の呼び出し元
         D queue (private EpochDomain) が満杯（4096）かつ tryReclaim x2 が空けない
transition:
         :49  ISRRetireRouter stackRouter(m_epochDomain)   ← スタック上
         :52  coordinator → stackRouter.enqueueWithRetry
         :342 stackRouter.m_retireQuarantine.quarantine(...)   → true（Q 512 に空きあり）
         :378 signalDrainWakeup()   ← 消費側（engine の singleton router）を起こすが見ない
         :65  return true
after  : :68 関数 return → stackRouter 破棄
         → Q に格納されていた ptr/deleter/epoch が**永久に失われる**
         → T まで到達した場合は std::vector の heap ブロックごと leak
         → 呼び出し元は true を受け取り「ownership 移転成立」と誤認
```

#### Reachability proof

| 条件 | 確証 |
| --- | --- |
| `m_retireCoordinator != nullptr` | **YES** — `AudioEngine.Processing.DSPCoreLifecycle.cpp:90` / `:95`、`AudioEngine.CtorDtor.cpp:50` |
| `enqueueDeferredDeleteWithFallback` が production から呼ばれる | **YES** — 上記 14 箇所の setter と `~EQProcessor` |
| D queue 満杯 → Q 昇格 | **YES（設計上の昇格トリガそのもの）** — `enqueueWithRetry` の Stage 3 が D 満杯を前提にしており、Q/E/T は**この scenario のために存在している** |
| Q が 512 に満杯でないこと | 前提（満杯なら E/T へ進み、同じく stack member なので結果は同じ） |
| shutdown drain が stack router に届くか | **NO** — `ReleaseResources.cpp:435-555` は engine の `m_retireRouter`（`AudioEngine` の unique_ptr）を使う。`~EQProcessor`（`EQProcessor.Core.cpp:158-160`）は private D queue のみを drain。**stack router の Q/E/T には到達しない** |

#### Lifetime / ownership proof

`RetireQuarantineStore` の storage は **`entries_`（inline、`kMaxQuarantinedEntries = 512`、
ヘッダコメント「allocation-free: std::array + index 配置」）**。
`TerminalReclaimAuthority` の storage は **`std::vector<Entry> entries_`（growable）**。

- Q / E: `stackRouter` オブジェクト内に inline 存在 → 関数 return で**記憶ごと消滅**（heap leak なし）
- T: heap ブロックごと leak + entry 未 drain
- どちらの場合も **deleter は一度も呼ばれない**

#### Observable consequence

```text
1. EQState / BandNode の恒久的メモリリーク（無制限）
2. m_retireDropCount が増えない（EQProcessor.Core.cpp:142/148/242/739 は
   retire*Deferred が false を返した場合のみ increment するが、本件は true を返す）
   → 既存 telemetry が「retire drop 0」を報告し続け、障害を隠蔽する
3. terminalStoreCount / reclaimSuccessCount にも増分が出ない
4. signalDrainWakeup() が起こす consumer は engine の router を drain するため無意味
```

#### Why existing tests do not catch it

- `D8_2_B_2_Tests.cpp:200-768` は `enqueueWithRetry` の 4 段昇格を**engine の router** で検証する。
  `stackRouter` 経路を検証しない。
- `src/tests` に `EQProcessor::enqueueDeferredDeleteWithFallback` の**容量満杯 scenarios** は存在しない。
- `EQProcessor` の private `EpochDomain` を 4096 entry で満たす test は存在しない。
- **既存 oracle の改変 0  mondial** → 既存 test は構造的に検出できない。

#### Why it is not already CLOSED

`doc/work113/P1-5-IR-P2_Step5AW_P3-5-FPM-RCA-2_DQueue_provenance_terminal_state_exhaustive_audit.md:110` に

> `EQProcessor` owns a separate `EpochDomain m_epochDomain`. `EQProcessor::enqueueDeferredDeleteWithFallback()` creates a stack `ISRRetireRouter` over that local domain. Its D queue is not the `AudioEngine::m_epochDomain` queue.

という **provenance 注記**がある。しかし同文書は
「engine router の D queue に残る 2 entry の説明」を目的とした会計照合であり
（`:108`「must not be counted as routerPending=2」）、
**stack router 破棄による entry 喪失の consequence は記載されていない**。
同文書の verdict は `PARTIALLY-LOCALIZED` / `STOP / SOURCE-UNRESOLVED`（`:5-6`）で、
本件は**判定に至っていない**。

→ **同一ファイルへの既知の注記はあるが、同一 defect の記録は無い。
本件は STG-8 / STG-9 とも無関係（`EQProcessor` / `ISRRetireRouter` の domain であり、
`pendingReclaimHandles_` でも `RecoveryAdmissionTable` でもない）。**

#### Candidate repairs（未採用・Owner 判定待ち）

1. **`m_retireRouter`  member を実際に使う**（`EQProcessor.h:461-463` で注入済み）。
   engine の singleton router を渡す。但し `setRetireRouter` の production 呼出が **0 件**のため、
   注入配線自体の追加が必要。
2. `EQProcessor` を engine の `m_retireRouter` を受け取る形に組み替え、Q/E/T を共有。
3. `stackRouter` を member 化する（engine 生命周期に束縛）。ただし `EQProcessor` は
   `aligned_malloc` で構築され得るため member 増加の cache 影響を確認が必要。

#### Rejected alternatives

- **`stackRouter` の Q/E/T に shutdown drain を追加**: 到達不能（stack オブジェクトは
  関数 return 済み）。**不可**
- **`enqueueWithRetry` が返す前に stack router を drain**: 同一関数内で Q/E/T を空にできても
  epoch-safe でない entry は残る。**ownership 解決にならず**
- **戻り値を `false` に変更**: 呼び出し元は「caller retains ownership」として
  `m_retireDropCount` を増やすだけで、**オブジェクトは依然 leak**。**部分的**（observability は改善するが
  ownership は解決しない）

---

### 6.2 D2 — `SnapshotCoordinator` の Q-full が assert のみ（**P1 / 確定**）

#### Invariant violated

`ISRRetireRouter.cpp:315-319` と同じ「必ず次の authority へ ownership を移す」契約。
ただし D2 では**昇格先が engine の singleton router であり、shutdown drain は届く**。
欠落しているのは「Q 自体が満杯」の末端における**終端解決の不在**。

#### Exact source locations

| 要素 | file:line |
| --- | --- |
| D のみの retry（Q/E/T 昇格なし） | `src/core/SnapshotCoordinator.h:151-163` |
| Q 直接移送 sink | `src/core/SnapshotCoordinator.cpp:21-30` |
| Q-full 時の assert のみ | `src/core/SnapshotCoordinator.cpp:28-29` |
| Q の capacity と false 返却 | `src/audioengine/RetireQuarantineStore.h:78-84` |

#### State before → transition → state after

```text
before : GlobalSnapshot* を retire したい。D（4096）満杯 → tryReclaim 後 D 満杯
transition:
         SnapshotCoordinator.h:154 provider.enqueueRetire() → false
         :158 provider.tryReclaim()
         :159 provider.enqueueRetire() → false
         :162 return false
         SnapshotCoordinator.cpp:26 m_retireSink->quarantineRetire(...)
         RetireQuarantineStore.h:78 size_ >= 512 → :83 ++overflowCount_ → :84 return false
         SnapshotCoordinator.cpp:28 if (!stored) assert(false && "...")
after  : Release（NDEBUG）で assert は消滅
         → GlobalSnapshot* は**どこにも格納されず永久に失われる**
```

#### Caller / Callee

```text
caller : SnapshotCoordinator::switchImmediate   SnapshotCoordinator.h:97, 108
         SnapshotCoordinator::startFade        SnapshotCoordinator.cpp:60
         SnapshotCoordinator::completeFade     SnapshotCoordinator.cpp:117
         SnapshotCoordinator::retireCurrentAndTarget  SnapshotCoordinator.h:177, 182
         ← production: AudioEngine.Snapshot.cpp:158,162,171
                   AudioEngine.Processing.ReleaseResources.cpp:664 付近
callee : SnapshotCoordinator::quarantineRetireSink → ISRRetireRouter::quarantineRetire
         → RetireQuarantineStore::quarantine（engine の singleton Q）
setRetireSink: AudioEngine.CtorDtor.cpp:42   m_coordinator.setRetireSink(m_retireRouter.get())
```

#### Reachability proof

| 条件 | 確証 |
| --- | --- |
| `m_retireSink != nullptr` | **YES** — `AudioEngine.CtorDtor.cpp:42` |
| `switchImmediate` / `startFade` が production から呼ばれる | **YES** — `AudioEngine.Snapshot.cpp:158` / `:162` / `:171` |
| D が満杯 | EBR 滞留 scenario（Q/E/T 存在的理由そのもの） |
| Q も満杯 | 512 entry 滞留。**二重 exceptional** |

**Q に**格納された** entry は shutdown drain で回収される**
（`ReleaseResources.cpp:444` / `:555`）ため、**遺失するのは Q-full の 1 entry のみ**。

#### Observable consequence

- `GlobalSnapshot` 1 個が**恒久的に leak**（`SnapshotFactory::destroy` が呼ばれない）
- `RetireQuarantineStore::overflowCount_` は**増加する**（`:83`）ため、
  D1 と異なり**観測可能性はある**
- `ReleaseResources.cpp:421-428` の `audit.isAllZero()` は `pendingRetire` 等を見るため、
  この 1 個を検出できない（audit に D/Q の個別残存 field がない。→ R1 と相互関連）

#### Why existing tests do not catch it

`D8_2_B_2_Tests.cpp:274-268`（T2）は「D 満杯 → Q へ」の**成功**経路を検証するが、
**Q 満番 scenarios を検証しない**。`RetireQuarantineStore` の容量を実際に 512 まで
埋める test は存在しない。

#### Why it is not already CLOSED

`SnapshotCoordinator.cpp:16-20` のコメントは
「capacity exhaustion は health escalation で先行検知｜ここでは jassert で異常検出」と
**前提を表明している**。しかし health escalation は**検知**であって
**失われた entry の終端解決ではない**。この前提が裏付けられているかを示す
source evidence は repo 内に存在しない。
STG-8 系（`RuntimePublicationOrchestrator` deferred slot / recovery obligation）、
STG-9-D1（`pendingReclaimHandles_`）とは**別 domain**。

#### Candidate repairs

1. `quarantineRetireSink` の Q-full 時に **E → T へ昇格**（engine router の既存 API を利用）。
2. `quarantineRetireSink` が `bool` を返し、呼び出し側が engine の
   `enqueueWithRetry` の T 経路へ委譲する。
3. `SnapshotCoordinator` の free `enqueueWithRetry`（`:151-163`）に
   Q/E/T 昇格を追加し、`quarantineRetireSink` を廃止。

#### Rejected alternatives

- **Q-full 時に `deleter(ptr)` を直接実行**: `SnapshotCoordinator.cpp:18` が明示的に禁止
  （「directDelete は禁止｜T 巻中の UAF 排除」）。RT 参照中の UAF を生む。**不可**
- **`assert` を `jassert` / 恒久 assert に変更**: Release で消滅する問題は変わらない。**不可**

---

### 6.3 D3 — `applyMmcssPriority` 失敗分岐の RT mutex + heap allocation（**P1 / 確定**）

#### Invariant violated

> **RT は待たない**（Practical Stable ISR Bridge Runtime）

#### Exact source locations

| 要素 | file:line | 内容 |
| --- | --- | --- |
| RT 入口（double 経路） | `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:66` | `tryApplyMmcssForSelfManagedThread()` |
| RT 入口（float 経路） | `src/audioengine/AudioEngine.Processing.AudioBlock.cpp:62` | 同上 |
| 初回のみゲート | `src/audioengine/AudioEngine.Mmcss.cpp:74-77` | `if (t_mmcssTried) return ...; t_mmcssTried = true;` |
| 失敗分岐の diagLog | `src/audioengine/AudioEngine.Timer.cpp:288-293` | `diagLog("[AFFINITY] FAILED: mask=0x" + juce::String::toHexString(...) + ...)` |
| diagnostics guard の開始位置 | `src/audioengine/AudioEngine.Timer.cpp:294` | `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` |
| diagnostics の既定値 | `CMakeLists.txt:129` | `option(... RUNTIME_DIAGNOSTICS "..." OFF)` |
| diagLog 実装 | `src/audioengine/AudioEngine.Timer.cpp:136-140` | `DBG(message); asyncSink(message);`（**guard 外で定義**） |
| asyncSink 実装 | `src/audioengine/AudioEngine.Timer.cpp:84-96` | `:87 std::lock_guard<std::mutex> lock(s_logMutex);` |
| mutex の contention 前提 | `src/audioengine/AudioEngine.Timer.cpp:82` | 「asyncSink は 2 スレッド（Message+Rebuild）Producer のため」 |

**決定的な非対称**: 成功時のログ（`:296-298`）は `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の
**内側**、**失敗時のログ（`:290-292`）は外側**。既定が OFF なので、
**通常ビルド／Release で実行されるのは失敗分岐側**である。

#### Reachability proof

| 条件 | 確証 |
| --- | --- |
| audio callback から呼ばれる | **YES** — `BlockDouble.cpp:66` / `AudioBlock.cpp:62` |
| 初回のみ | **YES** — `t_mmcssTried`（`Mmcss.cpp:74-77`） |
| `!hasHeterogeneousCores_` | **YES（既定で live）** — `AudioEngine.h:2939` `= false`、代入は `initialize` 時のみ |
| `audioMask != 0` | 対称コア環境で通常 true |
| `::SetThreadAffinityMask(...) == 0`（失敗） | **劣化環境でのみ**。これが唯一の追加条件 |

#### Observable consequence

初回 audio callback においてのみ、`juce::String` の**ヒープ確保 2 回**と
**`std::mutex` 取得**が audio thread 上で実行される。
`s_logMutex` は Message / Rebuild thread が保持し得るため、
**audio callback がブロックされ得る**（xrun / dropout）。
メモリ破壊・データ loss ではない。

**Severity 判定の明示**: Owner の定義は「RT safety breach」を P0 に属す。
本件は RT safety breach に**該当する**が、consequence が
「初回 callback 限定 × affinity 失敗時限定」で**有界**であるため **P1** と判定する。
**より強い consequence の証拠（無制限の停止等）が提出されれば P0 へ昇格しうる。**

#### Why existing tests do not catch it

`src/tests` に `isAudioThread` / `RTAllocatorFirewall` / `Mmcss` / `SetThreadAffinityMask` を
検査する test は**存在しない**（実測 0 件）。
`RTCapabilityFirewall` / `RTAllocatorFirewall` の hook も**呼出し 0 件**で、
さらに `JUCE_DEBUG || CONVO_CI_BUILD` guard 内の `assert` であり Release では消える。
**RT/NonRT 所属を検証する test 系が存在しない**（§10 参照）。

#### Why it is not already CLOSED

STG-1〜STG-10 のいずれ.Platform RT / MMCSS / affinity / スレッド優先度
を扱っていない。`AudioEngine.Mmcss.cpp:71` のコメント
「RT impact: first call only (~50-200μs for LPC call to MMCSS service)」は
**LPC コストのみ**を自認しており、**失敗分岐の mutex** は記録されていない。

#### Candidate repairs

1. `:288-293` の失敗時 diagLog を `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` の**内側**に移す
   （成功時との対称性を回復）。
2. affinity 設定失敗の記録を `t_mmcssTried` 相当の「非 RT へ回す」経路に queue する。
3. `applyMmcssPriority` 全体を非 RT（`prepareToPlay`）へ移設する。

#### Rejected alternatives

- **`s_logMutex` を lock-free 化**: 影響範囲が `asyncSink` 全体に及び、MP→SPSC ring の
  設計（`Timer.cpp:82`）に波及する。**局所修正ではない**
- **`diagLog` 全体を RT 無効化**: 全診断が失われる。**過剰**

---

### 6.4 D4 — `closeReaderRegistration()` が RT reader の slot 再取得を壊す（**Potential / 要検証**）

> **本項目は Concrete Defect として確定していない。** Repair Contract Audit の
> 優先検証項目として引き継ぐ。

#### Source で確定した事実

| # | 事実 | file:line |
| --- | --- | --- |
| a | `reserveReaderThread` は **`registrationClosed_` が true なら slot が既存でも `false`** | `EpochDomain.h:92-93` |
| b | `registerReaderThread` は `registrationClosed_` が true なら `-1` | `EpochDomain.h:50-51` |
| c | `RCUReader::acquireThreadSlot()` は `activeThreadId < 0`（block 間）かつ `reserveReaderThread` 失敗時に `registerReaderThread()` へ落ち、`-1` を返す | `RCUReader.h:162-181` |
| d | `RCUReader::enter()` は `tid < 0` なら `enterReader` を**呼ばない**が、guard は生成され呼び出し元は状態を読む | `RCUReader.h:62-68` |
| e | `ConvolverProcessor::process()` と `EQProcessor::process()` は `rootEnterSucceeded()` を**検査しない** | `ConvolverProcessor.Runtime.cpp:240` / `EQProcessor.Processing.cpp:488` |
| f | `closeReaderRegistration()` は `releaseResources()` 終端パス（`ReleaseResources.cpp:274`）と `~AudioEngine`（`CtorDtor.cpp:236`）で呼ばれる | 同左 |
| g | **governing comment が (a) と矛盾する**: 「登録済み slot の enter/exit は継続可能｜既存 Reader の epoch 安全性は維持」 | `ReleaseResources.cpp:274-275`、`CtorDtor.cpp:234-235` |

**(a)〜(d)(g) はすべて source で確定した。** comment が保証すると主張している
「既存 Reader の epoch 安全性維持」は、reader が**毎 block slot を再取得する**
（`RCUReader.h:167-172`）以上、close 後は成立しない。

#### 未証明の要素

**「その窓で audio callback が実際に実行されるか」**。
- 終端パス（`releaseResources`）では JUCE 規約上 device が先に停止する可能性が高い。
- しかし `~AudioEngine` パス（`CtorDtor.cpp:236`）のコメントは
  「**releaseResources 未実行な異常系 shutdown でも**」と明記しており、
  この経路では device 停止順序が確立されていない。
- graceful drain の終了条件は `pendingRetireCount()==0 && activeReaderCount()==0`
  （`ReleaseResources.cpp:287-288`）であり、reader が保護を失うと
  `activeReaderCount()` が 0 になり**条件を満たしうる**。

→ **到達性_orders の証明が必要。Repair Contract Audit の最優先項目。**

#### Severity（暫定）

到達性が証明された場合 = **P0**（epoch 保護 없는 RT reader と reclaim 並行 = UAF）。
未証明 = **P1（invariant 矛盾）／要検証**。

---

## 7. Severity 総括

| ID | Severity | 根拠（consequence） |
| --- | --- | --- |
| D1 | **P1** | ownership loss / 恒久的リーク / 既存 telemetry による隠蔽。データ loss は無いが lifecycle strand |
| D2 | **P1** | ownership loss（1 object）/ `overflowCount_` は増加するため観測可能 |
| D3 | **P1** | RT safety breach（mutex + heap）。初回 callback 限定・失敗時限定のため有界 |
| D4 | **要検証** | 証明されれば P0。現状は invariant 矛盾（P1） |

**最高確定 severity = P1。P0 は確定していない。**

---

## 8. Existing test coverage

| 領域 | カバレッジ |
| --- | --- |
| `ISRRetireRouter::enqueueWithRetry` の 4 段昇格 | `D8_2_B_2_Tests.cpp:200-768` にあるが、**engine の singleton router のみ**。`stackRouter` 経路は未検証 |
| `RetireQuarantineStore` capacity 満杯 | **未検証**（512 まで埋める test なし） |
| `SnapshotCoordinator` の Q-full 末端 | **未検証**（`D8_2_B_2_Tests.cpp:274-307` は Q 成功経路のみ） |
| `EQProcessor::enqueueDeferredDeleteWithFallback` の容量満番 | **未検証**（test 呼出 0 件） |
| RT / NonRT  スレッド所属 | **test 0 件**（`isAudioThread` / `RTAllocatorFirewall` / `Mmcss` / `SetThreadAffinityMask` を検索して 0 件） |
| RT runtime guard | `RTAllocatorFirewall` / `RTCapabilityFirewall` の hook は**呼出し 0 件**。さらに `JUCE_DEBUG \|\| CONVO_CI_BUILD` guard 内の `assert` で Release では消える |
| CI verifier | `tools/retire_authority_verifier.py` は宣言パターンの一致のみ。RT/alloc/mutex の規則は 0 件 |
| STG-10-D1 の T10-1〜T10-4 | 追加済み（`src/tests/STG10ReaderQuarantineTests.cpp`） |

---

## 9. Candidate repair direction（総合）

| 優先 | ID | 方向 | 想定 scope |
| --- | --- | --- | --- |
| 1 | **D1** | `EQProcessor` の retire を engine の共有 router へ（`m_retireRouter` member が既に存在） | `EQProcessor.Core.cpp` ＋ 注入配線 ＋ `EQProcessor.h` |
| 2 | **D3** | 失敗分岐 diagLog を diagnostics guard 内へ（非対称の解消） | `AudioEngine.Timer.cpp:288-293` 1 箇所 |
| 3 | **D2** | Q-full 時に E → T へ昇格 | `SnapshotCoordinator.cpp:21-30` |
| 4 | **D4** | closeReaderRegistration と reader slot 再取得の関係を再設計 | `EpochDomain` / `RCUReader` / shutdown ordering。**最大** |

**いずれも Owner の Repair Contract Audit なしには着手しない。**
D1 は D4 とテーマ（authority の寿命）が近いが、**別 defect であり混ぜない**。

---

## 10. Rejected / non-defect findings

以下はConcrete Defect と**判定しない**。理由付きで記録する。

| # | 対象 | 判定 | 理由 |
| --- | --- | --- | --- |
| N1 | `DeletionQueue::enqueue` の容量超過時の無条件 `deleter(ptr)`（`core/DeletionQueue.cpp:12-19`、コメントは「安全側に倒れる前に deleter を呼び出す」と明記） | **未到達 → Design Debt (P3)** | 唯一の実体は `SnapshotRetireManager::m_queue`（`SnapshotRetireManager.h:49`）。`SnapshotRetireManager` は**どこでも構築されない**（`rg` で定義のみ）。production から到達不能 |
| N2 | `TerminalReclaimAuthority::drainAll` の無条件 `deleter`（`ISRRetireRouter.cpp:104-117`） | **設計どおり** | shutdown drain。ヘッダ契約（`ISRRetireRouter.h:40-45`）が「audio thread stopped」で呼ぶと明記 |
| N3 | `ConvolverProcessor.Lifecycle.cpp:61` の即時 delete | **正しい fallback** | `getRcuProvider() == nullptr` 時の分岐。RCU が無いので deferred する意味がない |
| N4 | `EQProcessor::m_retireRouter` が write-only | **D1 の**修復方向の一部**として扱う**（単独 defect ではない） | 宣言 `EQProcessor.h:494`・代入 `:463`・**読み取り 0 件**。未使用 member 自体は無害 |
| N5 | `SafeStateSwapper::swap` の overflow leak（`SafeStateSwapper.h:123-126`、コメント「ADD-1: overflow → quarantine」に対し実際は `fallbackOverflowCount_++` のみで leak） | **Design Debt (P3) / 要注記** | 「quarantine」するコードが存在せずコメントと実装が乖離。production 到達可（`ConvolverProcessor.StateAndUI.cpp:1083`）だが overflow 条件が極端。**Ownership 喪失ではあるが §6 の 3 条件（durable / retry / terminal）をいずれも満たさない-design 上の明示的 leak** |
| N6 | `RuntimeIntentCoordinator::submitObserve` の全層溢れ drop（`ISRRuntimePublicationCoordinator.cpp:657-658`） | **設計どおり** | Observe は coalesce 可と明記。`recoveryAdmissions_` table に Live 保持される類と異なり、Observe は値 coalesce |
| N7 | `RuntimeIntentCoordinator::submitQuarantine` の全層溢れ（`:1528-1529`） | **Potential（要検証）** | `delivery==None` の obligation が Live 保持される。terminal resolution は table 終端に存在。**STG-8-D3「pressure signal redrives」と同一系統の可能性があるため、STG-8 と非交差であることを Audit で先に確認する必要がある** |
| N8 | `resetFadeStateAndRetireTarget` の戻り値破棄（`SnapshotCoordinator.cpp:92`） | **D2 の一部として要確認** | 素の `enqueueRetire` を bool 破棄で呼ぶ。production 呼出元的確認は本 Discovery では完了していない |
| N9 | `stuckReaderInfo.readerIndex` 既定 `-1` | **健全** | `IEpochProvider.h:29`。両 emit 経路（`RuntimeHealthMonitor.cpp:502` / `:527`）が `isStuck` 下で `readerIndex` を設定。`quarantineReader(-1)` も範囲検査で false。**欠陥なし** |

---

## 11. STOP conditions の判定

| # | STOP 条件 | 判定 |
| --- | --- | --- |
| 1 | authority が FRESH でない | 該当せず（baseline 完全一致） |
| 2 | source authority と repo source が一致しない | 該当せず（NEWER_SRC_COUNT 0） |
| 3 | CLOSED defect の reopen が必要になる | 該当せず（§3 / §6 の各「not already CLOSED」で source 証拠提示） |
| 4 | production source modification が必要になる | 該当せず（read-only 完遂） |
| 5 | test modification が必要になる | 該当せず |
| 6 | Concrete Defect と Design Debt の区別ができない | 該当せず（§6 に P1 確定 3 件、§10 に Design Debt 3 件を分離） |
| 7 | lifetime / ownership proof が成立しない | **該当せず**（D1/D2 は lifetime/ownership を source で証明。**D4 のみ未証明だが「Potential Defect」と明示的に分離し、確定扱いにしていない**） |
| 8 | repair candidate が一意に絞れない | 該当せず（§9 に優先順位付きの単一方向を提示） |
| 9 | R1〜R8 の既知情報だけでは判断できず追加実装が必要になる | 該当せず（read-only で分類完了） |
| 10 | 外部情報が必要なのに source evidence だけで結論を出そうとしている | 該当せず（結論は全て source evidence に基づく。D4 は結論を出さず要検証と明記） |
| 11 | unrelated worktree changes に触れる必要が生じる | 該当せず（**pre-existing 差分へ一切触れていない**） |

**STOP 条件に該当しない。Owner へ差し戻す。**

### 変更禁止の遵守

```text
production source change = 0
test source change       = 0
CMake change             = 0
ConvoPeq.md change       = 0（再生成も実施せず）
commit                   = 0
push                     = 0
worktree の整理・stash・reset・削除 = 0（pre-existing 差分に未接触）
```

---

## 12. Verdict

```text
STG-11 Discovery
================

Authority:
HEAD = f95f524cf6df0b3e1529425e690c02767912614e
ConvoPeq SHA = E9FBD8F9E47DACA786D1BC6DB9596180A8659961BE34973CE4C4CC738CF29D09
size = 5,662,249 B
Generated = 2026-09-29 00:19:43
NEWER_SRC_COUNT = 0
FRESH = yes

STG-1〜10 CLOSED boundary = PASS   (reopen 0 件)

Residual reassessment:
R1 = Observability Gap（Concrete Defect ではない。Owner §6 の 7 問で判定）
R2 = Already Safe / Mitigated（STG-10-D1 RC-1 により安全性に影響しない）
R3 = Already Safe / Mitigated（STG-10 Audit §10.3 補題により S6 は無害）
R4 = Design Debt（P3）
R5 = Observability Gap（P2）
R6 = Observability Gap / Design Debt（P3）
R7 = Potential Defect（D4 へ引継ぎ。最優先 Audit 項目）
R8 = Not Reproducible（再発なし）

New Concrete Defects:
D1 = EQProcessor stack-local ISRRetireRouter → Q/E/T ownership loss / 恒久的リーク
     / 既存 telemetry による隠蔽                              [P1 / 確定]
D2 = SnapshotCoordinator::quarantineRetireSink の Q-full が assert のみ
     → 終端解決なしの object loss                             [P1 / 確定]
D3 = applyMmcssPriority 失敗分岐が RT で mutex + heap allocation
     → RT safety breach（初回 callback 限定）                  [P1 / 確定]
D4 = closeReaderRegistration が RT reader の slot 再取得を壊す
     （governing comment と矛盾）                              [Potential / 要検証]

Highest confirmed severity: P1

Production changes = 0
Test changes = 0
CMake changes = 0
ConvoPeq changes = 0
Commit = 0
Push = 0

Verdict:
STG-11 Discovery = COMPLETE
Owner decision required for next stage
```

**推奨 next stage**: `STG-11-D1 Repair Contract Audit`（D1 を主対象）。
ただし Owner は **D4 の到達性検証のみ**を優先し得る。D4 が P0 に達しうるため、
**Audit の順序は Owner の判断事項**。

**本 stage では implementation へ進んでいない。**
