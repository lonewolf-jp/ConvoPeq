# STG-11-D2 Post-Implementation Gate（2026-09-29）

> **Verdict: STG-11-D2 = READY FOR COMMIT**
> **Commit = NOT AUTHORIZED / Push = NOT AUTHORIZED / staged = 0（未実施）**
> 本書は Owner GO に基づく Post-Implementation Gate（§2〜§12）の実施記録である。
> D3 implementation は開始していない。

---

## 1. Authority（§2）

Owner 指定: `ConvoPeq(20260929-124301).md == ConvoPeq.md`（統一 authority）。
**以下の値は実ファイルから再取得した実測値である。**

| 項目 | 実測値 |
| --- | --- |
| HEAD | `f95f524cf6df0b3e1529425e690c02767912614e` |
| SHA-256 | `9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE` |
| size | 5,710,250 B |
| Generated | 2026-09-29 21:16:18 |
| NEWER_SRC_COUNT | 0 |
| FRESH | yes（`--check` exit 0） |

---

## 2. R-A final source audit（§3）

**PASS**。`SnapshotCoordinator::quarantineRetireSink()`
（`SnapshotCoordinator.cpp:27-48`）を source 実読で最終確認した。

| Owner 確認事項 | 結果 |
| --- | --- |
| Q success → return（E/T を呼ばない） | **成立**（`:33-35` で return） |
| Q full → E attempt | **成立**（`:37-38`。Q-full 時のみ到達） |
| E success → return（T を呼ばない） | **成立**（`:37-39` で return） |
| E full → T | **成立**（`:45-46`。E-full 時のみ到達） |
| T → ownership terminally transferred | **成立**（下記契約確認） |
| Q-full＋E-full 時に ptr が失われない | **成立**（T が常時受領） |
| 同一 ptr の Q/E/T 複数 container 同時保持なし | **成立**（if/return chain により各段は transfer 成立でのみ到達） |
| caller 側への retry なし | **成立**（sink は `void`。呼び出し 6 箇所はいずれも再 enqueue しない） |
| coordinator 委譲後の同一 ptr 再 enqueue なし | **成立**（sink は enqueue を呼ばない。D1 型パターンなし） |
| T 到達後に caller が ownership を保持し続ける経路なし | **成立**（関数終了。T が所有または即時破棄） |

### `terminalReclaim()` の「常に受領」契約の source-level 確認

`ISRRetireRouter.cpp:509-533` を実読:

```text
:521 if (epochSafe && !isRt) → deleter(ptr) 即時実行 → return true
:532 return m_terminalReclaim.store(...)  // growable → 常に true（:49）
```

両 return 経路とも `true`。`store()`（`:27-50`）は growable `std::vector` のため
失敗経路を持たない。**「常に true」はコメントではなくコードで成立している。**
したがって `(void)` による戻り値破棄は安全（transfer 不成立の経路が存在しない）。

---

## 3. Q→E→T capacity / ownership audit（§4）

### 3.1 capacity 実測値（report 転記ではなく source 実測）

| store | capacity | source |
| --- | --- | --- |
| D | 4096 | `DeferredDeletionQueue.h:262` `kQueueSize` |
| Q | 512 | `RetireQuarantineStore.h:65` `kMaxQuarantinedEntries`（E と同型・同値） |
| E | 512 | 同上 |
| T | 無制限（growable `std::vector`） | `ISRRetireRouter.cpp:23-25,49` |

### 3.2 oracle 再確認（fresh 実行。Debug＋Release）

```text
TD2-1: D = 4096 / Q = 512 / E = 3
TD2-2: D = 4096 / Q = 512 / E = 512 / T = 3
```

### 3.3 issued == D＋Q＋E＋T＋reclaimed の成立

- TD2-1: issued = pre-fill 4608 ＋ coordinator 4 = 4612。
  保持 = D 4096 ＋ Q 512 ＋ E 3 = 4611 ＋ current slot 内 1 = **4612**。✓
- TD2-2: issued = pre-fill 5120 ＋ coordinator 4 = 5124。
  保持 = D 4096 ＋ Q 512 ＋ E 512 ＋ T 3 = 5123 ＋ current slot 内 1 = **5124**。✓
- drain 後: 全 counts 0・pre-fill alive 0（TD2-3 含む）。**喪失 0・重複 0。**

### 3.4 `same ptr appears in >1 container` の確認

正確な合計一致（不足なし・超過なし）が bijection を証明する。
超過があれば二重所有、不足があれば喪失である。
両者とも観測されず、negative control（§5）では超過側の検出力を実証した。
store 内容の address 列挙は public API で観測不能のため実施しない
（production に test hook を追加しない最小 scope 原則による。代替 evidence で十分）。

---

## 4. Negative control の最終確認（§5）

**保持**。R-A 無効化（Q-only legacy）時の観測（Release）:

```text
R-A disabled / D + Q full
→ Q full
→ E = 0 / T = 0（TD2-1/2 FAIL。旧 silent-loss を検出）
→ TD2-3 PASS（shutdown path は影響なし）
```

復元後の確認:

```text
R-A restored（E→T 昇格あり）
NEGCTL = 0（`rg` 実測。最終 tree に negative-control 用変更なし）
Release TD2 全 PASS を再確認
```

---

## 5. Shutdown / destruction（§6）

**PASS**。E/T に入った object が既存 shutdown drain に乗ることを source で確認した。

| 経路 | E/T 到達 |
| --- | --- |
| `finalizeShutdown(false)` → `retireCurrentAndTarget` → sink（R-A） | E/T に格納されうる |
| `m_epochDomain.tryReclaim()`（`h:69`） | D＋Q＋E＋T の epoch-gated drain（`ISRRetireRouter::tryReclaim`） |
| engine `drainAllQuarantineStore()`（`CtorDtor.cpp:291`／`ReleaseResources.cpp:444,555`） | **Q＋E＋T の force drain**（`ISRRetireRouter.cpp:444-451`: Q/E `drainAllUnsafe`＋T `drainAll`） |
| `m_epochDomain.drainAll()` | D の force drain |
| `~SnapshotCoordinator`（finalize 済みなら no-op） | — |

順序: `finalizeShutdown`（producer/consumer 停止後）→ epoch-gated drain →
force drain。lifetime: `m_coordinator`（`:5027`）は `m_retireRouter`（`:5020`）より
後に宣言されるため破棄は逆順で sink が先に死なない。
`m_epochDomain`（`:5015`）は最後に破棄される。

```text
D / Q / E / T → shutdown / force drain → all zero（TD2-3 で実証）
```

---

## 6. RT contract（§7）

**PASS**。

- `advanceFade()`（`cpp:85-88`）は `m_fade.advance` カウンタ減算のみ。
  retire / reclaim / delete / mutex / allocation / blocking / policy decision に
  到達しない（従来どおり）。
- D2 production diff 内の raw `std::atomic` API 追加 = **0 件**（実測）。

Practical Stable ISR Bridge Runtime の再確認（D2 diff に対して）:

```text
RT は待たない      : 維持（RT 経路の変更 0）
RT は解放しない    : 維持（解放は NonRT の drain のみ）
RT は判断しない    : 維持（RT に分岐追加なし）
Retire は Epoch を通る : 維持（全段は engine domain で評価。TD2-1/2 の epoch-gated drain で実証）
Shutdown は完全 Drain : 維持（§5 の経路で E/T を含めて drain）
Overflow は silent loss にしない : 回復（R-A が終端解決。TD2-1/2 で実証）
```

---

## 7. D1 regression（§8）

**PASS**。Debug／Release の両方で fresh 実行した。

```text
TD1-1: Q = 304（不変）
TD1-2: D/Q/E/T = 4096/512/512/280・total = 5400（不変）
TD1-3 / TD1-4a / TD1-4b: PASS（不変）
```

| 不変条件 | 再確認結果 |
| --- | --- |
| INV-D1-1 | PASS（TD1-4a/4b） |
| INV-D1-2 | PASS（TD1-1/1-2 の正確な counts） |
| INV-D1-3 | PASS（TD1-1/1-2/1-3 の release 後全 0） |
| INV-D1-4 | PASS（TD1-4b） |
| INV-D1-5 | PASS（RT diff 0） |
| INV-D1-6 | PASS（TD1-1/1-2 の合計一致。D2 は EQ 系に触れない） |

---

## 8. Test suite（§9）

| 項目 | Debug | Release |
| --- | --- | --- |
| CTest | **45/45 PASS**（208.21 s） | **45/45 PASS**（148.58 s） |
| harness 直接実行 | exit 0・全 PASS | exit 0・全 PASS |
| TD2 standalone | exit 0・全 PASS | exit 0・全 PASS |
| TD1 standalone | exit 0・全 PASS | exit 0・全 PASS |

D8_2（T2 の D→Q 成功経路）／StuckReaderFallback 系は CTest 45/45 の一部として
PASS（改変 0）。D2 実装による変化なし。

### 8.1 付記：harness 直接実行の CWD 依存

Gate 期間中、 harness を foreign CWD から直接実行した 1 回で `0xC0000005` を観測した。
ただし同一 binary の CTest 実行（45/45 PASS）・`build\Debug` からの直接実行（exit 0 全 PASS）・
procdump 監視下の実行（exit 0）では再現せず、CTest・正規 CWD では安定している。
CTest が正規の実行条件であり、D2 起因の deterministic failure ではない。
**判定: D1 Gate の intermittent 記録と同系統の環境要因。PASS。**

---

## 9. ASAN status（§10）

```text
ASAN = INCONCLUSIVE / environment blocked
```

D2 target について D1 と同一の `0xC0000139`（pre-main ローダ失敗）を確認した。
**ASAN failure と D2 implementation failure を混同しない。**
D2 の ASAN ビルド自体は成功しており、失敗は実行時 load である。

代替 evidence（D1 Gate §6 と同一方針）:

* exact-count oracle（TD2-1: E=3／TD2-2: T=3／合計一致）
* negative control（E==0/T==0 を検出）
* repeated Debug/Release execution（standalone＋harness、exit 0）
* complete drain（TD2-3: 全 0・alive 0）
* destructor completion（全 test が scope exit で破棄完了）

---

## 10. Diff / scope audit（§11）

**PASS**。

D2 による production behavior change:

```text
src/core/SnapshotCoordinator.cpp
quarantineRetireSink() のみ（+21/-3 相当）
```

D1 の production changes は既存変更として保持（revert なし）。
D2 起因の変更がないこと（全て diff 0 を実測）:

```text
ISRRetireRouter.* / EpochDomain.* / RCUReader.* / EQProcessor.* /
AudioEngine RT path / Coordinator authority
```

CMake/test changes は test infrastructure に限定
（standalone target＋add_test＋IPO-OFF＋ASAN＋harness TU 配線）。

---

## 11. dead code（§12）

`resetFadeStateAndRetireTarget()` は production caller 0 件のまま。
**D2 scope = no change**（D2 diff に対象言及なし。実測 0 件）。
勝手に修正していない。

---

## 12. Remaining risks

| # | 項目 |
| --- | --- |
| 1 | C1∧C2∧C3 の同時成立は二重 exceptional のため実運用の発火頻度は極めて低い |
| 2 | E/T 昇格後も engine epoch が前進しなければ滞留する（timeliness のみ） |
| 3 | ASAN は環境 block（§9）。代替 evidence で対応 |
| 4 | N4（submit 系 overflow）は STG-8 との非交差確認が別途必要。scope 外 |
| 5 | `resetFadeStateAndRetireTarget` の dead code（raw enqueue＋bool 破棄）は残存。将来の呼び出し追加時は sink 経路を使うべき |
| 6 | §8.1 の harness CWD 依存の crash（1 回・非再現）。D2 起因ではない |

---

## 13. Final verdict

```text
STG-11-D2 POST-IMPLEMENTATION GATE = COMPLETE

R-A source audit = PASS (§2: 8 項目全て成立。terminalReclaim 常時受領を code で確認)
ownership proof = PASS (§3.4: 合計一致による bijection＋negative control)
capacity proof = PASS (D=4096 / Q=512 / E=512 / T=growable。source 実測)

TD2-1 = PASS (E == 3 exact; Debug + Release)
TD2-2 = PASS (T == 3 exact; Debug + Release)
TD2-3 = PASS (all zero; Debug + Release)
TD2-4 = PASS (CTest 45/45 Debug + Release; harness both)

negative control = PASS (kept; NEGCTL = 0 in final tree)
D1 regression = PASS (TD1 both configs; oracle unchanged)
INV-D1-1..6 = intact

shutdown/drain = PASS (E/T ride existing drains)
RT contract = PASS (advanceFade counter-only; raw atomic +0)
raw atomic audit = PASS (0)

Debug CTest = 45/45 PASS
Release CTest = 45/45 PASS
Debug harness = PASS (exit 0)
Release harness = PASS (exit 0)

ASAN = INCONCLUSIVE / environment blocked (same signature as D1 gate)
ConvoPeq SHA = 9565A3D546BF256F75C1A2B97FB1E9B99CFF8D6DED11A3673642EADFF4F8F5BE
size = 5,710,250 B
Generated = 2026-09-29 21:16:18
NEWER_SRC_COUNT = 0
FRESH = yes

production changes = SnapshotCoordinator.cpp (sink only) + D1 kept
test changes = STG11D2 TU (new) + harness wiring + TD1 kept
CMake changes = test-infra only

staged = 0
commit = 0
push = 0 (NOT AUTHORIZED)

Remaining Risks = §12 (trigger frequency / timeliness / ASAN env / N4 /
                  dead code / harness CWD flake)

Verdict = STG-11-D2 = READY FOR COMMIT
```

**`STG-11-D2 = READY FOR COMMIT`。commit / push は未実施。Owner の別途 GO を待つ。**
**D3 implementation は開始していない。**
