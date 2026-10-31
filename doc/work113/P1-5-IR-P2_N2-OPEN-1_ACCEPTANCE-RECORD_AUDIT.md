# N2-OPEN-1 Audit — acceptance record for the D105-R5-8 replacement

```text
record type          read-only audit
authorised           N2-OPEN-1 read-only audit
question             does an acceptance record for the D105-R5-8 replacement behaviour
                     exist, or does an existing record already serve that role
classification       C2   acceptance record does not exist
subjects modified    0
gate entered         none
occurrence numbered  none.  no new occurrence number is assigned by this audit.
```

**This audit does not judge the replacement behaviour.** It establishes what the corpus
does and does not contain. "No acceptance record exists" and "the replacement behaviour is
unacceptable" are different claims, and only the first is in scope.

---

## 1. Corpus and method

```text
corpus     doc/work113, 246 markdown records, 4,017,359 B
method     read-only.  identifier and phrase probes, then per-record subject / gate /
           conclusion reading.  String-hit counts were treated as leads, not as findings.
git        not used.  no revision, no blob, no diff was needed for this audit.
```

The decisive distinction the Owner asked for is between a record that *mentions* a term and
a record whose *subject* is the acceptance of a behaviour. Every probe below was therefore
followed by reading the record's title, headings and terminal state block.

---

## 2. Search targets and what each returned

| probe | hits | records |
| --- | --- | --- |
| `D105-R5-8` | 7 | 2 |
| `D105-R5-9` | 4 | **1** |
| `D105` (any) | 49 | 9 |
| `onPublishCommitted` | 14 | 7 |
| `CAS single-winner` | 3 | **1** |
| `Route A` / `Route B` | 11 / 16 | 6 / 6 |
| `recoveryObligationId` | 57 | 7 |
| `obligation` + transport | 4 | 4 |

The two bolded rows are the finding. In both cases the **single record is this workstream's
own N2 audit**, which quoted the source comment. Before that record existed:

```text
D105-R5-9         pre-existing record coverage : 0
CAS single-winner pre-existing record coverage : 0
```

The entire justification for the replacement behaviour — the claim that the synchronous
Route-A completion becomes a no-op because the CAS has a single winner — exists **only as a
C++ source comment**. It has never appeared in a work113 record.

`D105-R5-8` appears in exactly one pre-existing record:

```text
P1-5-IR-P2_Step5AG_...R25...  L93
  intent.payload.publish.recoveryObligationId = 0;   // ★ D105-R5-8: Route B (non-recovery) ⇒ no obligation
```

and it appears there as a **quotation of the pre-change source**, i.e. as a statement of the
pre-state. Not as an acceptance of anything.

---

## 3. The seven evidence classes, separated

The Owner's §5B asked that these not be conflated. They are not.

| class | present? | where |
| --- | --- | --- |
| pre-state verification | **YES** | Step5AI six-point check, item 6: "旧: Route-B intent は id=0 のため async completion は recovery を resolve しない → 一致" |
| design record | **YES** | Step5AH (R26) origin disambiguation; Step5AK (R29) D105 dynamic verification design |
| implementation record | **YES** | Step5AI (R27) origin plumbing + build gate |
| source comment | **YES** | `AudioEngine.h` — declares D105-R5-8 void, names D105-R5-9 |
| build gate | **YES** | Step5AL (R30); Step5AN (R32) build gate PASS |
| **behavioural acceptance** | **NO** | — see §5 |
| runtime verification of the decisive half | **NO** | — see §5 |

The pre-state was verified. The implementation was gated. **Neither is acceptance of the
replacement.** The gap is specific and is not a gap in the pre-state work.

---

## 4. What the runtime records did and did not establish

The D105 chain runs R27 → R34. Its terminal states, read from the records:

```text
R28  Step5AJ   判定 R28-B (Observability insufficient)
               §6 "D105 dynamic verification":
                 D105 dynamic winner = UNOBSERVABLE
               §10 L2:
                 D105 dynamic winner は観測不能（§6）。
                 R27 静的 PASS は動的 PASS に置換しない。

R29  Step5AK   R29-B ADOPTED (Path B).  §11:
                 D105 Route A / Route B の dynamic 検証方法を定義
                 （exactly-once は drain で間接・winner は UNOBSERVABLE）
                 D105 winner identity は別途 observability gate（R28-B 残件）として分離

R30  Step5AL   runtime, 3/3 episodes:
                 [OK] origin observation   Recovery publish で coordΔ == 0 (3/3)
                 [OK] Main counter         Recovery によって coord が増加しない (3/3)
                 R30-C 非該当（contract 矛盾・Recovery id の 0 化・coord 混入なし）
                 [NG] drain                waitForDrain() 成功 → 不成立 (STOP-E)
               R30-B attributes STOP-E to the observer design, explicitly NOT to the
               recovery path and NOT to a D105 contract defect.

R31  Step5AM   R31-B ADOPTED (test-only vehicle ordering fix).
               STOP-4（winner identity なしでは exactly-once を証明できない）：該当
               R31-C： per-episode T1 と E2 に限り該当。新 observability が必要・未実装

R32  Step5AN   Build PASS / Control PASS / Recovery Episode 3/3 PASS /
               Shutdown nominal PASS.   C1–C4 constraints recorded.
               mentions of D105, D105-R5-9, single-winner, onPublishCommitted,
               recoveryObligationId : ZERO

R33-A Step5AO  R33-A-PASS (限定付き).  drain timeout cause isolated as pre-existing
               and not Recovery-caused.   D105 / obligation mentions : ZERO

R33-B Step5AP  R33-B-PASS, contract clarification, production/test/CMake 0, STOP

R34  Step5AQ   判定 C — P3-5 terminal contract CLOSED
               判定 A（R31-C を次に採る）: 不採用
               判定 B（R33-C を次に採る）: 不採用
               R31-C は「P3-5 closure の先行条件ではない」「独立 track として残置」
               R34 の禁止リスト: × R33-C 実装 × R31-C 実装 × Recovery 変更 × ...
```

### 4.1 The split, stated plainly

```text
OBSERVED at runtime, 3/3, by R30 and re-confirmed by R32
    a Recovery-origin publish keeps recoveryObligationId != 0
    and therefore does not increment the Main-only coordinator counter
    i.e. the origin PLUMBING works and the counter discriminates Main from Recovery

NOT observed, and recorded as unobservable
    which completion path wins the CAS for a given obligation
    i.e. D105 dynamic winner identity
    i.e. whether onPublishCommitted resolved asynchronously, and whether the
        synchronous Route-A completion at Orchestrator.cpp:321 was a no-op
    i.e. the D105-R5-9 claim itself
```

The replacement behaviour named in the source comment has two halves. The first half has
runtime evidence. The second half — the half that D105-R5-9 and CAS single-winner exist to
justify — has none, and the corpus says so in its own words, twice, in two different records.

---

## 5. Classification

```text
C1   an acceptance record already exists               NO
C2   no acceptance record exists                       YES   <- this classification
C3   an existing record could be read as acceptance    not reached; see 5.2
```

### 5.1 Why not C1

A record whose subject is the acceptance of the R27 replacement behaviour does not exist.
The candidates were each examined and each fails on subject, not on wording:

| candidate | subject | why it is not acceptance |
| --- | --- | --- |
| Step5AI (R27) | origin plumbing implementation + build gate | its own six-point check verifies the **pre**-state. Subject is implementation. |
| Step5AL (R30) | recovery-origin vehicle implementation + build gate | subject is the **vehicle**. Its `[OK] origin observation` is runtime evidence for the plumbing half, not an acceptance of the replacement. |
| Step5AN (R32) | terminal observation, build + run | zero mentions of D105, obligation, or winner. Subject is shutdown terminalisation. |
| Step5AO / AP (R33-A/B) | drain cause isolation / contract clarification | subject is shutdown drain. Zero obligation mentions. |
| Step5AQ (R34) | terminal contract closure | "P3-5 terminal contract CLOSED" is the **shutdown terminal** contract, not the publish-origin contract. Its exclusion list names `× Recovery 変更`. |
| Step5AG (R25) | coordinator take counter | mentions D105-R5-8 only as quoted pre-change source. |

A corpus-wide probe for the words *acceptance* / *受容* co-occurring with R27, origin, D105,
obligation or Route returned **no record**.

### 5.2 Why not C3

C3 would apply if an existing record could plausibly be *read* as acceptance and the Owner
should decide. That question does not have to be reached here, because the decisive half has
no evidence of any kind, so no reading of the existing records can supply it.

The residual judgement the Owner may still wish to make is narrower, and is recorded here
without being decided:

```text
R30 and R32 supply runtime evidence for the plumbing half of the replacement.
Whether that constitutes acceptance of that half is the Owner's to say.
It does not affect the classification of the item as a whole, because the
winner-identity half is separately recorded as UNOBSERVABLE and deferred.
```

### 5.3 The finding is stronger than absence

This is not merely a missing document. The corpus contains an affirmative, repeated
statement that the decisive property cannot be established with the available observability,
and an explicit instruction not to substitute the static result for the dynamic one:

```text
Step5AJ §6    D105 dynamic winner = UNOBSERVABLE
Step5AJ §10   R27 静的 PASS は動的 PASS に置換しない
Step5AK §11   winner は UNOBSERVABLE／別 observability gate として分離
Step5AM       STOP-4：winner identity なしでは exactly-once を証明できない：該当
Step5AQ §6    R31-C は P3-5 closure の先行条件ではない／独立 track として残置
```

So the record set is internally consistent about its own limit. What is missing is not a
verdict on the replacement behaviour; it is any evidence capable of supporting one, and an
identified, still-open route to obtaining that evidence (R28-B residual, carried as R31-C,
explicitly not selected at R34).

---

## 6. D105-R5-9, stage by stage

| stage | present? | evidence |
| --- | --- | --- |
| design | **partial, at D105 level only** | Step5AK is titled "D105 Dynamic Verification Design Gate" and defines the verification method, including that the winner is unobservable. **No record treats D105-R5-9 as a subject.** |
| implementation | **NO, as a subject** | D105-R5-9 names no code change. The change is R27's, under Step5AI. |
| build | **NO, as a subject** | Step5AL / Step5AN build gates cover R27 / R30 / R32. Neither names R5-9. |
| runtime | **NO** | R30 and R32 ran; neither observed a winner. Step5AJ §6 and Step5AK §11 state it cannot be observed. |
| acceptance | **NO** | §5. |

Runtime correctness was not inferred from the corpus here, and could not have been: the
corpus states the observation is unavailable rather than reporting an observation.

---

## 7. N1 — unchanged

```text
N1   authority provenance   OPEN / NOT PROVEN

adjudicated authority, held unchanged
    G0/G1
    5,535,334 B
    E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
    content identity   ACCEPTED
    provenance         OPEN

current working-tree snapshot, not an authority, not promoted
    5,550,764 B
    C587352201F0BE3D8F39D30396137A663E35A7412EAA749FBE591D661E8A7DFE
```

This audit did not touch N1, did not use the working-tree snapshot as an authority, and
resolved nothing about authority provenance.

---

## 8. N2-OPEN-1 status

```text
N2-OPEN-1   the D105-R5-8 invalidation has no recorded acceptance

finding     C2.  No acceptance record for the replacement behaviour exists.
            The decisive half of the replacement — D105-R5-9 / CAS single-winner /
            async resolve — has zero pre-existing record coverage and is affirmatively
            recorded as UNOBSERVABLE, with the residual deferred to R28-B, carried as
            R31-C, and explicitly not selected at R34.

status      REMAINS OPEN.  Not closed by this audit.

NOT concluded by this audit
    that the replacement behaviour is correct
    that it is incorrect
    that it is safe or unsafe
    that D105-R5-9 holds or fails
    that CAS single-winner does or does not hold in practice
    that D105-R5-8 may be reinstated
```

---

## 9. Prohibitions observed

```text
production source change   0    build                        0
test execution             0    runtime execution            0
CDB                        0    debuggee                     0
g / go / gc                0    R1                           0
RR-1..RR-18 re-evaluation  0    CR-3                         0
Phase 2-B                  0    Contract Amendment           0
Amendment 1 modification   0    ConvoPeq.md regeneration     0
ConvoPeq.md re-pin          0    authority replacement        0
git write operation        0    commit / checkout / reset / restore   0
968377BE reconstruction    0
git queries of any kind    0    not needed; the corpus was sufficient
```

The only file created by this audit is this record. No occurrence number was assigned.

---

## 10. Result

```text
corpus searched          246 records
acceptance record        C2 — does not exist
decisive-half coverage   0 pre-existing records for D105-R5-9 and CAS single-winner
what the corpus says     the property is UNOBSERVABLE, and static PASS does not
                         substitute for dynamic PASS
residual route           R28-B, carried as R31-C, not selected at R34, still open
N1                       OPEN / NOT PROVEN, unchanged
subjects modified        0
gate entered             none
```

### 10.1 Returned to the Owner

```text
N1                  OPEN / NOT PROVEN
N2                  DOCUMENTED
N2-OPEN-1           read-only audit complete, classification C2, item REMAINS OPEN
all execution gates STOP
```

Nothing was closed. No step was taken from this record, and none is proposed.
