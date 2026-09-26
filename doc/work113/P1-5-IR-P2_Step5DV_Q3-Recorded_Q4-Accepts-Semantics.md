# Step5DV — Q3 Ruling Recorded, and the Q4-1..Q4-4 Read-Only Acceptance Gate

```text
gate                = Owner Decision Gate, Q3 ruling recorded + Q4 acceptance semantics
mode                = READ-ONLY.  No CDB run, no retry, no .cdb edit, no source / test / CMake /
                      build / harness change.  No predicate selected.  No candidate enumerated.
execution this gate = 0
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

## 1. Q3 = a, RECORDED

```text
Q3 = (a)  SPECIFIC RING POSITION          CONFIRMED

target   one ring-slot acquisition of slot 4 in DeferredDeletionQueue, bracketed

           T0E   enqueuePos == 4
           T0P   slot 4 free, pre-CAS
           T0Q   CAS executed
           T0S   slot 4 published, enqueuePos -> 5

         the selector is a trigger / capture selector for a ring acquisition episode.
         it is NOT a retirement-outcome predicate, and slot 4 is chosen as the
         observation target, not as a production retirement semantic.

epoch    @r9 == 9 is NOT part of the target.  It is phase-wide, 4 of 4 sites, and redundant
literal  against the positional bracket.  Under Q3 = a, deleting it is a CONSEQUENCE of the
         target, not an independent decision.

scope    this ruling fixes the SEMANTIC TARGET only.  It does NOT authorise
           source modification / test / CMake / .cdb / build / runtime
           deletion of the epoch literal as an implementation
           adoption of any RDG-2 candidate
```

The Owner's framing is adopted verbatim and is load-bearing for section 4:

> slot 4 is not chosen as a production retirement semantic; it is chosen as the concrete ring
> acquisition episode this CDB / T0 capture should observe.

## 2. Two corrections to my own earlier work, made before deriving anything

### 2.1 A parser defect that produced a fictitious result

My first pass attributed the 4 `enqueuePos == 4` hits to queues and reported Baseline-1 as having
**1 T0E hit**. Both numbers were artifacts of my code, not properties of the runs.

```text
cause 1   Baseline-1 has no C3_DG_T0E_EP0 and no C3_DG_T0E_WIN_END.  Both count 0.  The window
          dump was introduced in Step5DN, not Step5DI.  My block scan used WIN_END as its
          terminator, found none, ran to the end of the log in one iteration, and returned 1 hit.
cause 2   the register dump is three registers per line, e.g.
              rax=0000000000000000 rbx=00000044c04ee3e0 rcx=000001bbdab58580
          so rcx= is never at the start of a line and my ^rcx= anchor never matched.
```

Redone with a terminator that exists in both logs (the next `C3_DG_T0E_HIT`), with a search-based
register anchor, and with explicit self-checks that the segment count contains no segment missing
its EQ marker. The corrected figures are in section 3.

### 2.2 A structural count that does not reconcile

```text
source declares  3  production EpochDomain members, each owning one DeferredDeletionQueue
                   src/audioengine/AudioEngine.h:5016        convo::EpochDomain m_epochDomain;
                   src/eqprocessor/EQProcessor.h:487        convo::EpochDomain m_epochDomain;
                   src/ConvolverProcessor.h:1417            convo::EpochDomain m_epochDomain;

                   a full scan of src/ finds no fourth production declaration.  The remaining
                   EpochDomain objects are stack locals in test files
                   (src/tests/invariant_INV3_INV5.cpp x2, StuckReaderFallbackDrainTests.cpp x6)
                   and could not account for heap-range queue bases.

observed         4  distinct queue bases at T0E in Baseline-2

NOT RECONCILED.  This is recorded as an open item, not explained away, and it is the sole reason
Q4-2 cannot be closed.  See section 5.
```

## 3. The measured evidence, corrected and self-checked

Whole-line marker equality, both `--measurement=normal` runs:

| marker | Baseline-1 (Step5DI) | Baseline-2 (Step5DN) |
|---|---|---|
| `C3_DG_T0E_HIT` | 116 | 116 |
| `C3_DG_T0E_R9_PASS` | 0 | 0 |
| `C3_DG_T0E_R9_FAIL` | 116 | 116 |
| `C3_DG_T0E_EQ_PASS` | **4** | **4** |
| `C3_DG_T0E_EQ_FAIL` | 112 | 112 |
| `C3_DG_T0P_HIT` | 116 | 116 |
| `C3_DG_T0Q_HIT` | 116 | 116 |
| `C3_DG_T0S_HIT` | 116 | 116 |
| `C3_T0_ENTRY` | 0 | 0 |
| `C3_T0_SEQUENCE_PUBLISHED` | 0 | 0 |
| register dump present | **no** | yes |
| per-queue attribution | **unavailable** | available |

```text
the COUNT 4 is reproducible across two independent runs.  CONFIRMED.

the DECOMPOSITION rests on Baseline-2 alone, because Baseline-1 carries no register dump:
  4 distinct queue bases, and enqueuePos == 4 held exactly once on each,
  no queue selected twice, no queue never selected.

  0x1BBD21ED680   21 T0E hits   1 selected
  0x1BBD326DB80   13 T0E hits   1 selected
  0x1BBDAB58580   41 T0E hits   1 selected
  0x1BBE165C580   41 T0E hits   1 selected
```

Also observed, and relevant to C5: **every T0E hit is followed by a T0P, T0Q and T0S hit**, 116
each. The four breakpoints are exercised on all 116 enqueues. Only the 4 with `enqueuePos == 4`
satisfy the T0E conjunct, so the phase has never advanced past T0E for the other 112.

## 4. Q4-1, Q4-3 and Q4-4 are DERIVABLE. Q4-2 is not.

### 4.0 The structural fact all three rest on

From `src/DeferredDeletionQueue.h` enqueue:

```text
L77   uint32_t pos = convo::consumeAtomic(enqueuePos, std::memory_order_acquire);
L79   auto& seq_atom = sequences[pos & kMask];
L82   int32_t diff = static_cast<int32_t>(static_cast<uint32_t>(seq - pos));
L85   compareExchangeAtomic(enqueuePos, ...)
```

`enqueuePos` is a per-queue monotonic ring index, `kMask` is 4095, and it advances by one per
successful CAS. Therefore:

```text
enqueuePos takes the value 4 EXACTLY ONCE per 4096 successful enqueues on a given queue.

T0E fires once per enqueue call, and dwo(@rcx+0x34000) == 4 is true at T0E exactly when that
call is the one that will claim position 4.

  => ONE slot-4 acquisition per queue instance per 4096-enqueue cycle.
  => the number of acquisitions in a run = the number of queue instances that REACH position 4.
```

This is the conversion the Owner required. The specification is **one acquisition per instance**.
The observed 4 is what 4 instances produced. It is not the specification, and a run with 3
instances would make 4 a failure.

### 4.1 Q4-1 expected capture count — DERIVED, closed

```text
Q4-1  =  one slot-4 acquisition per queue instance that reaches enqueuePos == 4

        NOT "4".  NOT the 4/116 observation.  The 4 is an outcome, not a specification.

        a queue with fewer than 4 enqueues never reaches position 4 and contributes 0.
```

### 4.2 Q4-2 acceptable range — NOT DERIVABLE, blocked

Two independent obstacles, both real:

```text
obstacle 1   the instance count is unreconciled.  source declares 3, the run exhibited 4.
             until that is settled, the expected total has two candidate values.

obstacle 2   the expectation must NOT be read out of the log it validates.
             "captures == distinct rcx values observed in the same run" is exactly the
             self-validating check this workstream has recorded as a recurring defect:
             a total assembled from one population compared against a count taken from
             another, or a property identified by the very check meant to test it.
             it can never fail, so it is not a test.

             an INDEPENDENT specification is required.  Only two acceptable forms:
               (i)  a constant established by a source-level inventory of what the
                    --measurement=normal path instantiates, which requires obstacle 1 resolved
               (ii) an Owner-stated tolerance
```

This is an Owner decision. It is not resolved here, and no default is assumed.

### 4.3 Q4-3 zero-capture disposition — DERIVED, closed

```text
Q4-3  =  CONDITIONAL on target existence, which is measurable and is NOT assumed

        the Owner's §7 states the correct procedure: confirm whether the target acquisition
        existed, then judge.  both branches are decidable from the run itself.

        the existence test is the per-queue T0E hit count:
          a queue with >= 4 T0E hits MUST have produced exactly one slot-4 acquisition.
          therefore if max(per-queue T0E hits) >= 4, target existence is PROVEN.

        Baseline-2:  per-queue counts 21, 13, 41, 41.  all >= 4.  existence PROVEN.
                     so a capture count of 0 would be FAIL, not a normal outcome.

        Baseline-1:  NO register dump, so per-queue counts are UNAVAILABLE and the existence
                     test CANNOT be performed on that run.  a zero there would be INDETERMINATE
                     rather than PASS or FAIL.

        the asymmetry is a property of the two scripts, not of the runs.  Baseline-2 can decide
        Q4-3; Baseline-1 structurally cannot.
```

### 4.4 Q4-4 multiple-capture disposition — DERIVED, closed

```text
Q4-4  =  capture count MUST EQUAL the number of instances that reached position 4

        == expected   PASS
        <  expected   FAIL, an acquisition was missed
        >  expected   FAIL, and it is also structurally impossible under monotonicity:
                      a given queue's enqueuePos equals 4 exactly once, so no instance can
                      contribute two acquisitions.  an excess means the selector is not the
                      one that was specified.

        this RESOLVES the Owner's §7 concern directly.  T0E does not write $t0, so its entry
        branch can fire repeatedly, and that is correct behaviour, not a latch defect.
        the multiplicity is the multiplicity of QUEUE INSTANCES, each contributing one.
        "capture > 1" is therefore the expected shape whenever two or more instances reach
        position 4, and its correct value is the instance count, never 1.
```

## 5. What is left open, precisely

```text
OPEN-1   instance count: 3 declared in source, 4 observed at T0E.  unreconciled.
         blocks Q4-2 only.  Q4-1, Q4-3 and Q4-4 are unaffected, because each is stated in
         terms of the instance count rather than as a constant.

OPEN-2   Baseline-1 has no register dump, so its per-queue decomposition and its Q4-3
         existence test are unavailable.  single-run evidence only, permanently, for that run.

OPEN-3   the positional conjuncts at T0P / T0Q / T0S have still never been evaluated, because
         the gate has never opened.  sequences[4] == 4 and == 5 remain UNMEASURED at every site.
         C5 in RDG-2 will need this.

OPEN-4   C7, the false-positive / false-negative definition.  Q3 = a supplies the target, so
         C7 is now answerable in RDG-2 rather than blocking.  It moves from blocker to work.
```

## 6. Gate result

```text
Q1        CONFIRMED    A, trigger / capture selector
Q2        CONFIRMED    the literal 9 is not established as a baseline specification
Q3        CONFIRMED    (a) specific ring position, one acquisition of slot 4
Q4-5      CONFIRMED    T0E is not a one-shot latch; T0S is the latch
Q4-1      DERIVED      one acquisition per instance that reaches position 4
Q4-2      BLOCKED      instance count unreconciled (3 vs 4), and the expectation must not be
                       read from the log it validates.  OWNER DECISION REQUIRED
Q4-3      DERIVED      conditional on a measurable existence test; decidable on Baseline-2,
                       indeterminate on Baseline-1
Q4-4      DERIVED      count must equal the instance count; excess is also impossible
Q4        NOT CLOSED   one item open

RDG-2     NOT STARTED
RDG-3     NOT STARTED
RDG-4     NOT STARTED
IMPLEMENTATION FORBIDDEN      RUNTIME NOT AUTHORIZED
```

## 7. Next

```text
Q4-2  requires one of
        (i)  a source-level inventory of what the --measurement=normal path instantiates,
             reconciling 3 declared against 4 observed, yielding an independent constant
        (ii) an Owner-stated tolerance
        (iii) an Owner ruling that Q4-2 is deliberately left open and inherited into RDG-2
              as a stated risk

on Q4 closing, RDG-2 compares candidates read-only under C1 to C7.  C7 becomes answerable
now that the target is fixed.  C2 is partially answered: enqueuePos == 4 held 4/116 in two
independent runs, and the per-instance decomposition is single-run.
```

```text
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no predicate selected    no candidate enumerated    no repair proposed
```
