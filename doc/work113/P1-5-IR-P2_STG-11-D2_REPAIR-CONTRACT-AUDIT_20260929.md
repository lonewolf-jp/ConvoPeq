# STG-11-D2 Repair Contract Audit（2026-09-29）

> **Verdict: D2 = CONFIRMED / Repair Contract = PROVEN（候補 R-A を推奨）**
> **本書は read-only 監査である。production = 0 / test = 0 / CMake = 0 / ConvoPeq.md = 0 / commit = 0 / push = 0。**
> **D1 の変更に触れていない。D1 commit なし。D2 implementation には進まない。**
> **D2 / D3 / D4 の実装開始ではなく、Owner の implementation GO を待つ。**

---

## 1. Authority

Owner 指定: `ConvoPeq(20260929-105350).md == ConvoPeq.md`（統一 authority）。
実 source 検証は repo の現行 `ConvoPeq.md` を基準に実施した。

| 項目 | 値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `CDAFA18BBCA815595ACB0B2C3D3F4C1FCB25D01DE150FEA6772837F65E5C38B1` |
| size | 5,690,145 B |
| Generated | 2026-09-29 19:35:52 |
| NEWER_SRC_COUNT | 0 |
| STATUS | FRESH（`--check` exit 0） |

D1 の production/test/CMake 変更は作業ツリーに保持されたまま
（`EQProcessor.*` 2 件＋harness 配線＋TU＋CMake＋authority）。
**巻き戻し・commit ともに行っていない。**

---

## 2. 現行 defect の正確な再現条件

### 2.1 Defect（STG-11 Discovery §6.2 の再構築）

```text
SnapshotCoordinator::quarantineRetireSink() は Q（RetireQuarantineStore）へ
直接移送するだけで、E（EmergencyQ）/ T（TerminalReclaimAuthority）へ昇格しない。
Q が満杯（512）の場合、Release ビルドでは assert が消滅して no-op となり、
GlobalSnapshot* 1 個がどこにも格納されず永久に失われる。
```

### 2.2 正確な再現条件（source 実読）

| # | 条件 | file:line | 備考 |
| --- | --- | --- | --- |
| C1 | `m_retireSink != nullptr` | `SnapshotCoordinator.cpp:24` | production は `AudioEngine.CtorDtor.cpp:42` で設定済み。未設定時は return による leak（別 boundary、§9） |
| C2 | D（DeferredDeletionQueue 4096）が満杯 | `SnapshotCoordinator.h:154` が false | free `enqueueWithRetry`（D のみ＋tryReclaim 1 回）の両方が失敗 |
| C3 | Q（RetireQuarantineStore 512）が満杯 | `RetireQuarantineStore.h:78` `size_ >= 512` | `quarantine()` が `false`＋`overflowCount_++`（`:83-84`） |
| C4 | Release ビルド（NDEBUG） | `SnapshotCoordinator.cpp:28-29` | `assert(false && ...)` が消滅し no-op |

**C1 ∧ C2 ∧ C3 ∧ C4 の同時成立が必要**（二重 exceptional: D+Q に 4608 entry 滞留）。
C2 単独では Q が受け、C3 単独では D が受ける。**末端条件である。**

### 2.3 Observable consequence

- `GlobalSnapshot` 1 個（条件成立の都度）が恒久的に leak（`SnapshotFactory::destroy` 未実行）。
- UAF・crash・データ loss はない（失われるのは retire 済みの旧 snapshot のみ）。
- 観測可能性はある: `RetireQuarantineStore::overflowCount_` が増加する（`:83`）。
  ただし `collectDrainAudit()` には D/Q の個別残存 field がなく、
  shutdown 完了判定では検出できない（R1 と相互関連。STG-11 §10 N 表参照）。

---

## 3. Ownership state machine

### 3.1 状態（実コード上の entity のみ）

| # | 状態 | holder | 判定材料 |
| --- | --- | --- | --- |
| O0 | caller 所有 | `switchImmediate` 等のローカル（`oldTarget` / `oldSnap` / `snap`） | slot exchange 直後 |
| O1 | D 所有 | `DeferredDeletionQueue`（engine `m_epochDomain` 内蔵） | `enqueueRetire`/`enqueueRetireTyped` が true |
| O2 | Q 所有 | `ISRRetireRouter::m_retireQuarantine`（engine singleton router の member） | `quarantineRetire` が true |
| O3 | E 所有 | `ISRRetireRouter::m_emergencyQuarantine` | **D2 経路では到達不能** |
| O4 | T 所有 | `ISRRetireRouter::m_terminalReclaim` | **D2 経路では到達不能** |
| O5 | 解放済み | —（deleter 実行後） | epoch-safe 到達後の drain |
| **OX** | **喪失（leak）** | **なし** | **C1∧C2∧C3∧C4 時に到達** |

### 3.2 全 enqueue 経路の ownership state transition

凡例: `→On` = 状態遷移、`[ret]` = 戻り値、`caller owns?` = 遷移後の caller 所有。

#### P1 — `switchImmediate` oldTarget（`SnapshotCoordinator.h:91-99`）

```text
O0 --[D enqueue ok]--> O1, caller: 失う, [true]
O0 --[D full]--> tryReclaim --> [D ok] --> O1
O0 --[D full ×2]--> quarantineRetireSink
     --[Q ok]--> O2, caller: 失う
     --[Q full]--> OX（Release）, caller: 失う（と誤認。実際は誰も所有しない）
```

#### P2 — `switchImmediate` oldSnap（`h:103-109`）

P1 と同一（epoch のみ `publishEpoch()` 値が異なる）。

#### P3 — `startFade` oldTarget（`cpp:53-62`）

P1 と同一（NonRT Timer 起点。§7 でスレッド確定）。

#### P4 — `completeFade` old（`cpp:100-119`）

P1 と同一（NonRT Timer 起点 `Timer.cpp:928`。§7 で確定）。

#### P5 — `retireCurrentAndTarget` current/target（`h:167-184`）

P1 と同一 ×2（shutdown / dtor 起点。NonRT）。

#### P6 — `resetFadeStateAndRetireTarget`（`cpp:81-96`）

```text
O0 --[D enqueue (bool 破棄)]--> O1（成功時）/ OX（D full 時、無通知）
```

**production 呼び出し元 0 件**（dead code。`rg` 実測。§10 N 表に分離）。
到達不能のため defect としては数えないが、将来の呼び出し追加時の罠として記録する。

#### P7 — dtor safety net（`h:50-58`）

P5（`retireCurrentAndTarget`）＋ `tryReclaim()`。NonRT（owner thread）。

### 3.3 全 overflow / retry 経路

| 経路 | retry 内容 | 二重所有の有無 |
| --- | --- | --- |
| free `enqueueWithRetry`（`h:151-163`） | D enqueue → tryReclaim → D enqueue（最大 2 試行）。**格納成功時は即 return** | **なし**（格納前に return しない。D1 の二重投入パターンとは異なる） |
| `quarantineRetireSink`（`cpp:21-30`） | **retry なし**。Q へ 1 回のみ | **なし**（再 enqueue しない） |
| P1〜P5 の sink フォールバック | `enqueueWithRetry==false` の場合のみ 1 回 | **なし**（D 格納済みなら sink に到達しない） |

**D1 で発見された「coordinator 委譲＋直接再実行」の二重投入パターンは
SnapshotCoordinator に存在しない。** 全 6 箇所（P1〜P5＋P7）は
「D 成功 → 終了／D 失敗 → Q へ 1 回」の逐次フォールバックであり、
同一 ptr が 2 container に格納される経路はない。

### 3.4 `RetireEnqueueResult` 5 値ごとの ownership 所在

| 結果 | 発生箇所（D2 経路） | ownership |
| --- | --- | --- |
| Success | free `enqueueWithRetry` が true（D 格納） | **O1（D）**。caller は失う |
| QueuePressure | 発生しない（free helper は bool。Q 移送は sink が別途行う） | — |
| TerminalReclaim | 発生しない（D2 経路は T に到達しない） | — |
| Shutdown | 発生しない（engine router の通常経路に Shutdown 判定なし） | — |
| QueueFull（bool false） | D 満杯 ×2 | caller が保持 → sink へ移送 → **O2（Q）または OX（Q 満杯時）** |

`sink` の `quarantineRetire` 戻り値: `true` = O2、**`false` = OX**（Q 満杯時）。
`m_retireSink == nullptr` 時も OX（早期 return。production では設定済み）。

---

## 4. Authority boundary

```text
caller（SnapshotCoordinator の各 retire site）
  │ O0 → D attempt（free enqueueWithRetry）
  ▼
coordinator（IEpochProvider abstract: enqueueRetire / tryReclaim のみ）
  │ D full ×2 → false
  ▼
router（ISRRetireRouter::quarantineRetire — Q への直接移送 API）
  │ Q ok → true
  ▼
quarantine（RetireQuarantineStore::quarantine — Q store）
  │ Q full → false（E/T への昇格なし ← 欠陥点）
  ▼
OX（Release）/ assert（Debug）
```

| boundary | 所有するもの | 所有しないもの |
| --- | --- | --- |
| caller | slot exchange までの snapshot ポインタ | retire 後の lifetime |
| coordinator（IEpochProvider） | D queue（engine domain 内蔵） | Q/E/T（router member。provider から不可視） |
| router（sink 先） | Q/E/T（members）。ただし D2 経路は Q のみ使用 | D（provider 内蔵） |
| Q store | 格納済み entry（epoch-gated drain まで） | E/T への転送判断 |
| E/T store | —（D2 経路では未使用） | — |

**Epoch / Retire lifetime**: 全て engine domain
（`m_coordinator(m_epochDomain)` — `CtorDtor.cpp:26`、
retireEpoch は engine `currentEpoch()` / `publishEpoch()`、
sink は engine `m_retireRouter` — `CtorDtor.cpp:39,42`）。
**単一 domain のため epoch mismatch は存在しない**
（D1 の EQ private/engine 問題とは対照的）。

**Sink lifetime**: `m_coordinator`（`AudioEngine.h:5027`）は
`m_retireRouter`（`:5020`）より後に宣言されるため、
破棄は逆順（coordinator → router）で sink が先に死なない。
`m_epochDomain`（`:5015`）は最後に破棄される。
`~SnapshotCoordinator`（`:50-58`）実行時は router・domain とも生存。✓

---

## 5. RT reachability

**D2 の retire 経路は RT から到達不能。**

| retire site | 呼び出し元 | スレッド | 根拠 |
| --- | --- | --- | --- |
| switchImmediate | `AudioEngine.Snapshot.cpp:162,171`（`messageThreadRcuReader` 文脈 `:133`） | NonRT（Message） | snapshot apply 経路 |
| startFade | `AudioEngine.Snapshot.cpp:158` | NonRT（Message） | 同上 |
| completeFade | `Timer.cpp:928` `tryCompleteFade()` | NonRT（Timer） | — |
| retireCurrentAndTarget | `finalizeShutdown`（`ReleaseResources.cpp:664`）＋ dtor | NonRT | shutdown / owner thread |
| advanceFade | `AudioBlock.cpp:483` | **RT** | **ただし body は `m_fade.advance` カウンタ減算のみ**（`cpp:67-70`）。retire / lock / alloc なし。コメント `:481-482` が RT/NonRT 分離を明記 |

したがって Q/E/T の mutex・allocation（`entries_` array 配置は alloc-free だが
`std::lock_guard` 取得を伴う）・deleter 実行・`terminalReclaim` の同期破棄は
いずれも RT から到達しない。**RT safety breach ではない。**

---

## 6. Atomic audit

`SnapshotCoordinator.h` / `.cpp` の raw `std::atomic` method 呼び出し（`.load/.store/
fetch_add/...`）= **0 件**（`rg` 実測）。全て `convo::consumeAtomic` /
`convo::publishAtomic` wrapper 経由。**PASS**。

---

## 7. Shutdown / destruction

| 経路 | 内容 | Q/E/T 到達 |
| --- | --- | --- |
| `finalizeShutdown(timedOut)`（`h:62-72`） | `retireCurrentAndTarget()` → Q sink → `tryReclaim()`（timedOut でなければ）。`ReleaseResources.cpp:664` から呼ばれ、producer/consumer 停止済み（`:667-668`） | Q のみ |
| `~SnapshotCoordinator`（`h:50-58`） | finalize 済みなら no-op。未済みなら retire＋tryReclaim（異常系の安全網） | Q のみ |
| engine `drainAllQuarantineStore()`（`CtorDtor.cpp:291` / `ReleaseResources.cpp:444,555`） | Q＋E＋T の force drain（audio 停止後） | **Q 到達。E/T は D2 経路で未使用のため対象外** |
| `m_epochDomain.drainAll()` | D の force drain | D |

R-A 適用後は E/T も engine 既存 drain に乗る（新規 drain 機構不要）。

---

## 8. D1 との相互作用

**衝突なし。**

| 観点 | D1（EQ） | D2（Snapshot） | 交差 |
| --- | --- | --- | --- |
| EpochDomain | EQ private `m_epochDomain` | engine `m_epochDomain` | なし（別オブジェクト） |
| Router | EQ-owned member router | engine singleton router | なし（別オブジェクト） |
| Retire 対象 | EQState / BandNode | GlobalSnapshot | なし（disjoint） |
| RT reader | EQ private rcuReader | engine audioThread/messageThread readers＋coordinator observe | なし |
| D2 候補の変更先 | — | `SnapshotCoordinator.cpp` のみ（予定） | EQ ファイルに触れない |

**D2 修正後に INV-D1-1〜INV-D1-6 を破壊する可能性: なし。**
理由: D2 候補（§9）は `SnapshotCoordinator.cpp` の sink 関数内か
その呼び出し関係に閉じ、EQ の private domain / member router /
retireEpoch / reclaim boundary のいずれにも触れない。
TD1 suite（standalone＋harness）は D2 実装後の非回帰として再実行する
（acceptance criteria §11）。

---

## 9. Repair candidates

### 9.1 定義

| 候補 | 内容 |
| --- | --- |
| **R-A** | `quarantineRetireSink` 内で Q-full 時に E → T へ昇格（既存 public API `emergencyQuarantine` / `terminalReclaim` を使用） |
| **R-B** | 4 箇所（P1〜P5相当）を router `enqueueWithRetry`（完全 D→Q→E→T）に置換し、sink を不要化 |
| **R-C** | sink が `bool` を返し、呼び出し側で terminal へ委譲 |
| **R-D** | `quarantineRetire` router API 自体に E/T fallback を追加（共有 API の意味変更） |

### 9.2 比較

| 評価軸 | R-A（推奨） | R-B | R-C | R-D |
| --- | --- | --- | --- | --- |
| ownership 終端解決 | **Q-full → E → T で解決**（T は growable で常時受領） | 同左（完全 chain） | 同左 | 同左＋他経路にも波及 |
| 変更範囲 | **`quarantineRetireSink` 1 関数のみ** | 4 呼出箇所＋sink 削除検討 | sink＋4-5 呼出箇所 | 共有 API＋全利用者 |
| 既存 API 変更 | **なし**（public E/T API をそのまま使用） | なし | sink signature 変更 | **あり**（`quarantineRetire` の契約変更） |
| D-only fast path の維持 | 維持（D 成功時は従来どおり） | 変更（常に完全 chain。retry＋wakeup が snapshot path に追加） | 維持 | 変更（全利用者に波及） |
| authority singularization | 維持（sink が単一移送点のまま） | 向上するが変更大 | 後退（escalation が分散） | 向上するが blast radius 大 |
| RT 影響 | なし（全呼出 NonRT） | なし（同左。ただし jassert ガードが追加の前提になる） | なし | DSPLifetimeManager 経路の再検証が必要 |
| D1 非干渉 | **完全**（EQ に触れない） | 完全 | 完全 | 要検証（共有 API のため） |
| epoch provenance | engine domain のまま | 同左 | 同左 | 同左 |

### 9.3 判定

| 候補 | 判定 | 理由 |
| --- | --- | --- |
| **R-A** | **ADOPT（推奨）** | 最小・局所・API 不変・D1 非干渉。欠陥点（sink の終端不在）を直接塞ぐ |
| R-B | ALTERNATIVE | chain 均一化の利点はあるが、D-only fast path の変更＋4 箇所編集＋wakeup 追加で R-A より変更大。R-A に構造的障害が見つかった場合の代替 |
| R-C | REJECT | escalation 分散で singularization 後退。R-A で同等効果を局所的に達成できる |
| R-D | REJECT（本次 scope） | 共有 API のため DSPLifetimeManager 経路の再検証が必要。将来の統一路線として記録 |

### 9.4 R-A の最小 repair scope（予定）

```text
files:
  src/core/SnapshotCoordinator.cpp（quarantineRetireSink 1 関数のみ）
functions:
  SnapshotCoordinator::quarantineRetireSink（Q-full 時に E→T へ昇格）
NOT touched:
  SnapshotCoordinator.h（signature 不変）
  ISRRetireRouter.*（API 不変）
  EpochDomain / RCUReader / RT path / Coordinator / authority（不変）
  EQ 系（D1 領域、完全非干渉）
```

---

## 10. Rejected / non-defect findings

| # | 対象 | 判定 | 理由 |
| --- | --- | --- | --- |
| N1 | `resetFadeStateAndRetireTarget` の raw `enqueueRetire`＋bool 破棄（`cpp:92`） | **到達不能（dead code）** | production 呼び出し元 0 件。欠陥として数えない。将来の呼び出し追加時は R-A 後の sink 経路を使うべき旨を記録 |
| N2 | `m_retireSink == nullptr` 時の早期 return（`cpp:24-25`） | **boundary 条件** | production は `CtorDtor.cpp:42` で設定済み。standalone 使用時は leak のみ・UAF なし（コメント `:187` の設計どおり） |
| N3 | `advanceFade` の RT 呼び出し | **設計どおり** | カウンタ減算のみ。retire/lock/alloc なし |
| N4 | `submitObserve`/`submitQuarantine` の全層溢れ drop | **scope 外** | STG-8 系統の可能性あり。D2 と混ぜず、STG-8 との非交差を D2 実装前に確認する（STG-11 Discovery §10 N7 に記録済み） |

---

## 11. 必要な regression tests／acceptance criteria

### 11.1 Required tests（実装段階）

| ID | 内容 | oracle |
| --- | --- | --- |
| TD2-1 | Q-full → E 昇格（D+Q 満杯後に E residency > 0、単一所有の正確な counts） | D=4096/Q=512/E>0、合計一致、drop 0 |
| TD2-2 | E-full → T 昇格（D+Q+E 満杯後に T residency > 0） | D/Q/E/T の正確な counts、合計一致 |
| TD2-3 | drain / shutdown 後の完全解決（leak 0、全 counts 0） | release＋drain 後に全 0・drop 0・alive 0 |
| TD2-4 | 既存非回帰（`D8_2_B_2_Tests` T2 の D→Q 成功経路、`StuckReaderFallbackDrainTests`、harness STG 全件、TD1 suite） | 改変 0・全 PASS |

### 11.2 Acceptance criteria（実装前）

```text
1. R-A 適用後、C1∧C2∧C3（D+Q 満杯）で E に格納され、C1∧C2∧C3∧E満杯で T に格納されること
2. 全オブジェクトが正確に 1 箇所に所有されること（TD1 と同等の合計一致 oracle）
3. Release（NDEBUG）でも終端解決すること（assert 非依存の value oracle）
4. RT path の変更 0（advanceFade はカウンタのみのまま）
5. ISRRetireRouter.* の変更 0
6. EQ 系の変更 0 かつ TD1 suite（standalone＋harness）が全 PASS のまま
7. 既存 test oracle の改変 0
8. ASAN は環境 block のため必須としない。代わりに §11.1 の exact-count oracle＋
   negative control（guard 除去で Q-full 喪失を検出）で代替する
   （D1 Gate §6 の方針を踏襲。D1 で ASAN INCONCLUSIVE を記録済み）
```

### 11.3 Negative control（実装段階で実施）

R-A の昇格を除去した状態で TD2-1 が Q-full 喪失（Release 相当）を検出すること。
D1 と同様、production の一時的 revert は検証後に完全復元する。

---

## 12. Remaining risks

| # | 項目 |
| --- | --- |
| 1 | C1∧C2∧C3 の同時成立は二重 exceptional（4608 entry 滞留）のため、実運用での発火頻度は極めて低い。ただし EBR 破綻 scenario（assert 文言が指すもの）では到達しうる |
| 2 | `quarantineRetireSink` の E/T 昇格後も、engine epoch が前進しなければ entries は滞留する（timeliness のみ。既存 Q と同一条件） |
| 3 | ASAN 実行は D1 と同一の環境 block が想定される。§11.2-8 の代替 evidence で対応する |
| 4 | N4（submit 系 overflow）は STG-8 との非交差確認が別途必要。本 Audit の scope 外 |

---

## 13. 最終報告

```text
STG-11-D2 = AUDIT COMPLETE

production source changes = 0
D1 commit = 0
push = 0

D2 status:
  CONFIRMED

ownership proof:
  PASS（§3: 全 7 経路の state transition を証明。D1 型の二重投入なし）

overflow proof:
  PASS（§2: C1∧C2∧C3∧C4 の同時成立で OX に到達。Q-full は overflowCount_++ のみ）

terminal reachability:
  PASS（現状は到達不能＝欠陥。R-A で E→T へ到達可能になる）

shutdown reachability:
  PASS（§7: finalizeShutdown＋dtor＋engine drains。R-A 後は E/T も既存 drain に乗る）

RT contract:
  PASS（§5: retire 経路の RT 到達なし。advanceFade はカウンタのみ）

repair candidate:
  R-A（quarantineRetireSink 内で Q-full 時に E→T へ昇格。既存 public API 使用）
  R-B は ALTERNATIVE。R-C／R-D は REJECT（§9.3）。

minimum repair scope:
  src/core/SnapshotCoordinator.cpp（quarantineRetireSink 1 関数のみ）
  header／Router／Epoch／RT／Coordinator／authority／EQ 系は不変

D1 interaction:
  なし（§8: domain／router／object の全てが disjoint。
  D2 候補は EQ に触れない。INV-D1-1〜6 の破壊可能性なし。
  TD1 suite を非回帰として再実行する）

Verdict = GO（R-A。§11 の tests＋acceptance criteria を実装段階で実施すること）

Recommended Repair Contract = R-A（§9.4）

Required Tests = TD2-1／TD2-2／TD2-3／TD2-4（§11.1）

Remaining Risks = §12（発火頻度／timeliness／ASAN 環境／N4）
```

**`Repair Contract = PROVEN`（候補 R-A）に収束した。ただし implementation には進まない。**
Owner の implementation GO を待つ。
