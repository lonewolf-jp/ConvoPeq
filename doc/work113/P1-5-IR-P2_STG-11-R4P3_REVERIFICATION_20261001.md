# R4/P3 — Double Representation Re-verification Record

- Document: `doc/work113/P1-5-IR-P2_STG-11-R4P3_REVERIFICATION_20261001.md`
- Work item: **R4/P3 Double Representation** — Owner 指示による現行 source での再検証
- Date: 2026-10-01
- Authority: リポジトリルート `ConvoPeq.md`（`ConvoPeq(20261001-054723).md` と同一。
  FRESH を作業開始時に確認。過去資料・記憶・旧行番号は未使用）
- Commit / push: **未実施**

---

## 0. Verdict

```
ALREADY REPAIRED — no production change made
```

Owner 指示 §1 の P3-2 counterexample sequence を現行 source で再検証した結果、
記述された欠陥形状（same-O durable holder に対する transport push）は存在しない。
`tryAttachDurableRecovery`＋`redriveDeferredRecovery`＋`casDelivery` が
Owner §2／§3 の repair contract を完全に満たし、Owner §6 の T-P3-1〜T-P3-6 と
同名・同期待の回帰テストが存在して PASS することを実実行で確認した
（`ISRSemanticValidationTests.exe` rc=0）。

存在しない欠陥への production 変更は行わない（並行 state machine への
speculative 変更は malpractice であり、「最小修正」の対象が存在しないため）。
Implementation／Gate は作成しない。D1〜D11 の未 commit 状態はそのまま保持する。

---

## 1. Owner sequence の再検証（§1）

Owner の counterexample を現行 code に写像する：

| sequence step | 現行 code の対応 | 結果 |
| --- | --- | --- |
| DurablePending(O) | `pendingRecoveryAdmission_.state == DurablePending`, holder O | — |
| Builder takes O → Building | `takePendingRecoveryAdmission` CAS（D146） | — |
| build failure → settle(true) | `settlePendingRecoveryAdmission(true)` → DurablePending（payload 保持） | slot は O を保持 |
| markTransientFailure → delivery=None | 現行 production は T1（`postRecoveryFailureSignal`）＋T2（`adjudicateRecoveryFailureSignals` の P-B repair）経由。`markTransientFailure` 自体は TEST-ONLY（production caller 0 件を grep 実証） | delivery=None |
| next redrive sees delivery None | `redriveDeferredRecoveryObligations` → `redriveDeferredRecovery(O)` | — |
| durable occupied by same O | `tryAttachDurableRecovery`：observed ≠ NoAdmission、holder == O → **`AlreadyRepresented`** | **transport push なし** |
| delivery = Durable | `casDelivery(idx, O, Durable)`（full-word CAS、失敗時は無書き込み） | 二重表現なし |

Owner のいう「state != NoAdmission → transport push → delivery = Transport」は
現行 code では発生しない。`OccupiedByOther`（transport へ）は holder ≠ O の
場合のみであり、Case C の既存 fallback 契約どおりである。

## 2. Repair contract の照合（§2／§3）

| Owner 要求 | 現行 code | file:line |
| --- | --- | --- |
| durable busy 時に holder == current oblId を明示判定 | `holder == oblId → AlreadyRepresented / else OccupiedByOther` | `ISRRuntimePublicationCoordinator.cpp:1373-1376` |
| same-O で transport push 禁止 | `if (attach != OccupiedByOther)` 分岐が transport を含まない | 同 `:1254-1259` |
| `s.delivery = Durable`＋wake | `casDelivery(..., Durable)`＋`redriveWakePending_ = true` | 同 `:1255-1256` |
| Case A（free → Durable） | `Attached` 経路 | 同 `:1254`＋`:1390` |
| Case B（same O → Durable、追加投入禁止） | `AlreadyRepresented` 経路 | 同 `:1254`＋`:1395` |
| Case C（different O → fallback） | `OccupiedByOther` → transport push（C13 不変） | 同 `:1262-1268` |
| container＋identity＋delivery の三者一致 | full-word CAS（`casDelivery`）。失敗時は無書き込み | 同 `:1417-1440` |

## 3. Ownership invariant（§4）

`one obligation → one delivery representation` は以下で成立する：

- durable attach は単一 primitive（`tryAttachDurableRecovery`）に収束（D146）。
  NoAdmission からの退出権限は CL 経路自身のみ。
- delivery 変更は full-word CAS（`casDelivery`）のみ。terminal／reuse には書かない。
- `markTransientFailure` の production caller 0 件を grep 実証（定義＋test のみ 39 件中 38 件が test）。
- T-P3-3 が `¬(Transport(O) ∧ Durable(O))` を直接 assertion する。

## 4. RT / ISR 制約（§5）

変更を行わないため不変。redrive／attach／adjudicate はいずれも
CoordinatorLoop／Builder（NonRT）のみ。RT path の追加なし。

## 5. Regression tests（§6 対応）

Owner §6 の T-P3-1〜T-P3-6 は同名・同期待で `ISRSemanticValidationTests.cpp`
（`:1628-1740`）に存在する：

| Owner 要求 | 既存 test | 内容一致 |
| --- | --- | --- |
| T-P3-1 same holder | `testP3_T1_sameHolderRepairsNoTransport` | transport なし＋Durable＋wake＋holder 維持 |
| T-P3-2 different holder | `testP3_T2_differentHolderFallsBackToTransport` | O1 維持＋O2 のみ transport |
| T-P3-3 free slot | `testP3_T6_freeSlotAttachAndP2Regression` | free→Durable（＋P2 契約） |
| T-P3-4 repeated | `testP3_T4_repeatedRedriveNoGrowth` | 1 回目のみ wake、増加なし |
| T-P3-5 failure→redrive→retry | `testP3_T5_failureRedriveCycleXor` | XOR＋K=4 終端 |
| T-P3-6 negative control | `testP3_T2`＋`testP3_T6`（fallback 契約維持） | C13 不変 |

実実行結果：`ISRSemanticValidationTests.exe` **rc=0**（全件 PASS。
失敗時は throw するため silent PASS はあり得ない）。
D1〜D11 変更込みの build で実行した（incremental rebuild 済み）。

## 6. なぜ production 変更をしないか

- 欠陥が存在しないことの証明：上記 §1 の写像＋§5 のテスト実証。
- 並行 state machine（D105/D139/D144/D146/D152 の多層監査済み）への
  無根拠な変更は、動作中の不変条件を破壊する risk のみを生む。
- Owner §2 の「最小修正」は「契約が破れている場合」の指示であり、
  契約が成立している現状では適用対象がない。
- D1〜D11 の設計不変条件・authority boundary は untouched（変更なしのため）。

## 7. 推奨

- 本件は CLOSED（already repaired＋verified）として記録することを推奨する。
- R4 label の残余（Candidate C の atomic state word 等）は別途 P3 扱いとすること。
- D6 NO-GO・R8 flake・dead code 一括整理は引き続き未解決として残る。

## 8. 停止状態

- HEAD = `f95f524cf6df0b3e1529425e690c02767912614e`（**未 commit**）
- branch = `main...origin/main [ahead 9]`、staged = 0
- **commit / push は未実施**
- D1〜D11 の未 commit 変更・D12〜D17 監査成果物はすべて保持
- Implementation／Gate は作成しない（変更がないため）
