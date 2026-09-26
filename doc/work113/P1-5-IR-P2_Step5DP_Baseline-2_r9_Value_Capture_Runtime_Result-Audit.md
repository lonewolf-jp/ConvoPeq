# Step5DP — Baseline-2 Runtime Result Audit, r9 Value Capture

```text
gate                = single authorized runtime execution
script              = P1-5-IR-P2_Step5DN_…_T0-Guard-Repair-Baseline-2.cdb
script SHA-256      = BD5129A3655FA04FB1EAF8539BA596BC47179FABD8817B85F3742216F083B763
script bytes        = 14,968
log                 = P1-5-IR-P2_Step5DN_…_Baseline-2_Runtime_CDB.log
                     193,507 B, 3,493 lines
invocation          = cdb.exe -cf <target> -logo <log> AudioEngineHarness.exe --measurement=normal
execution_count     = 1        Retry = 0
exit_code           = 0
elapsed             = 11 s
R1                  = PASS, 6 of 6
r9 raw value        = OBTAINED, 13 distinct values, never 9
window slot +0x28   = IDENTIFIED as the live global epoch
enqueueEpoch delta  = {0, +1} only, over 116 of 116 hits
```

## 1. R1, continuation

| item | value | required | result |
|---|---|---|---|
| `C3_T1_REDESIGN2_SCRIPT_BEGIN` | 1 | 1 | PASS |
| `C3_T1_REDESIGN2_SCRIPT_END` | 1 | 1 | PASS |
| `ZwTerminateProcess` | 1 | 1 | PASS |
| `quit` | 1 | 1 | PASS |
| exit code | 0 | 0 | PASS |
| debugger errors | 0 | 0 | PASS |

Error census, every pattern zero: `Syntax error`, `Extra character`, `Illegal`, `Invalid`,
`Undefined`, `^Error`, `Cannot`, `Memory access error`, `Unable to read memory`, `Couldn't`,
`Evaluate expression:`. `Unable` appears 4 times, classified benign, excluded from the census
rather than subtracted from it.

Both admitted new constructs executed at the T0E payload position without stranding the debuggee:
`r` printed its register block and the body continued, and `dq @rcx-0x1450 L11` printed its 11
quads and the body continued into the frozen guard. The two position risks carried by the
authorization are discharged.

## 2. R2, T0E hit count

```text
C3_DG_T0E_HIT           116
C3_DG_T0E_R9_PASS         0
C3_DG_T0E_R9_FAIL       116
C3_DG_T0E_EQ_PASS         4
C3_DG_T0E_EQ_FAIL       112
C3_DG_T0E_EP0            116
C3_DG_T0E_WIN_END        116
C3_T0_ENTRY               0
```

Both extension witnesses fired on every hit, so the value capture and the window were emitted
116 of 116 times. `C3_T0_ENTRY` remains 0, unchanged from the Baseline-1 run.

## 3. R3, `r9` raw distribution

These are **enqueue epoch** values, the 3rd declared parameter of
`DeferredDeletionQueue::enqueue`, stored by the callee to `entry.epoch`. They are **not**
`enqueuePos`.

```text
distinct values = 13

   1   x 41
   2   x 64
   5   x  1
   6   x  1
  16   x  1
  19   x  1
  22   x  1
  25   x  1
  28   x  1
  31   x  1
  34   x  1
  37   x  1
  48   x  1
```

```text
r9 == 9 ?   TRUE 0 of 116     never observed in this run
```

This settles the question the gate was chartered to settle, as a measurement: the enqueue epoch is a
small advancing counter, and the literal `9` was not among its 116 values.

The three prior observations stay separate and are **not** merged into an invariant:

```text
Redesign-5        1        one sample, run did not complete
Step5BB Retry-1   9        one sample, gate opened, --fpm-m0
Step5DK           not 9    116 of 116 hits, --measurement=normal
Step5DP           13 values, never 9, 116 of 116 hits, --measurement=normal
```

## 4. R4, attribution to the hit

```text
T0E hits                                116
complete blocks recovered                116
structural anomalies                     none
every hit yielded r9 + EP0 + WIN_END + window   True
window quads per block, distinct        [16]
```

The per-hit correspondence is one to one, established by ordinal position inside each block rather
than by totals. Each block was recovered in the authorised order: `C3_DG_T0E_HIT`, the R9 verdict,
the EQ verdict, the `r` output, `C3_DG_T0E_EP0`, the 11-quad window, `C3_DG_T0E_WIN_END`.

## 5. R5, extension, and the slot identification

### 5.1 The three relations, per hit

| relation | TRUE | FALSE |
|---|---|---|
| `r9 == window+0x18` (`rcx-0x1438`) | 0 | 116 |
| `r9 == window+0x28` (`rcx-0x1428`) | **105** | 11 |
| `r9 == 9` | **0** | 116 |

### 5.2 The 11 exceptions are a bounded lag, not noise

```text
delta = (window+0x28) - r9, over all 116 hits

    0   x 105
   +1   x  11
```

**The delta is only ever 0 or +1.** Never ahead, never behind by more than one, in 116 of 116 hits.
The eleven exceptions are all `+1`:

```text
r9 =  1 -> window 2      r9 = 16 -> window 17
r9 =  2 -> window 3      r9 = 19 -> window 20
r9 =  5 -> window 6      r9 = 22 -> window 23
                          r9 = 25 -> window 26
                          r9 = 28 -> window 29
                          r9 = 31 -> window 32
                          r9 = 34 -> window 35
                          r9 = 37 -> window 38
```

### 5.3 The identification criterion is now MET

The design gate set two conditions before a tracking slot could be called `globalEpoch`. Both hold.

```text
condition 1  the slot tracks r9 across hits
             MET.  window+0x28 equals r9 at 105 of 116, and the remaining 11 differ by exactly
             +1, never by anything else.  The two distributions have the same shape.

condition 2  the slot advances with the world epoch
             MET.  The observed progression, first 40 hits:
               window+0x28   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
               r9            1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
             The two advance together and the global value moves 1 -> 2 -> 3 while r9 moves 1 -> 2.
             Over the run window+0x28 reaches 48.
```

```text
IDENTIFIED   window+0x28, that is [rcx - 0x1428], is the live global epoch.
```

### 5.4 Three independent cross-checks fix the layout

The recorded offset algebra is self-contradictory by `0x10`, which is why the extension read a
window rather than a single address. The window resolves it, using source constants rather than the
recorded anchors.

```text
window+0x20  rcx-0x1430   0x7FF77F4A2A20 at 116 of 116 hits, a single constant code address
                         -> a vtable pointer -> the first word of a polymorphic object
                         -> EpochDomain + 0x00, since EpochDomain : public IEpochProvider

window+0x28  rcx-0x1428   the advancing value, identified above
                         -> EpochDomain + 0x08, and source declares globalEpoch as the
                            FIRST data member, i.e. immediately after the vptr

window+0x30  rcx-0x1420   kInactiveEpoch 0xFFFFFFFFFFFFFFFF at 108 of 116
                         kReservedEpoch 0xFFFFFFFFFFFFFFFE at  2 of 116
                         and the real epochs 1, 5, 29, 32 at 2 each
                         -> exactly the value set of readers[i].epoch
                         -> EpochDomain + 0x10, matching source where readers is the member
                            after globalEpoch and ReaderSlot.epoch is its first field
```

The three are mutually consistent and independently anchored, so the `0x10` contradiction in the
recorded anchors is resolved **in favour of the design gate's address**. The two values remain
distinct facts:

```text
source-confirmed   globalEpoch is EpochDomain + 0x08
empirically found  EpochDomain sits at rcx - 0x1430, hence globalEpoch at rcx - 0x1428
recorded and wrong one of the three legacy anchors, still uncorrected in the lineage
```

## 6. R6, T0 sequence

```text
T0E / T0P / T0Q / T0S = 116 / 116 / 116 / 116
T0 markers = 464 ; complete 4-cycles = 116 ; ordering violations = 0
```

Verified from this log. Reproducibility of the equal counts is now n=2 for Baseline-1 and n=1 for
Baseline-2, and equal counts remain a measurement rather than a guarantee, since the exit path at
`0x1f9c5e0` can skip T0P.

## 7. The relation this run measured, stated without a verdict

```text
MEASURED   enqueueEpoch equals the live global epoch, or is exactly one behind it.
           116 of 116 hits, delta in {0, +1} and nothing else.

MEASURED   enqueueEpoch never equalled the literal 9 in this run, 0 of 116.

CONSEQUENCE the guard's conjunct 2, '.if (@r9 == 9)', compares the enqueue epoch against a
           fixed literal, while the quantity that actually tracks it is the live global epoch.
```

**This is a relation, not a verdict.** The design gate and the authorisation both reserved this
determination for the Intent / Semantic Audit, and that reservation is honoured here. Nothing in
this section authorises changing the predicate.

```text
NOT concluded  that 9 is the wrong constant.  The run measures relations, not intent.
NOT concluded  what the predicate was meant to be.  That is the next gate's question.
NOT concluded  that the gate should compare against globalEpoch.  That is a candidate repair,
               and it needs the intent audit first.
```

## 8. Tooling defect in this gate

One, in the audit rather than in the measurement.

```text
The first block-extraction pass rejected all 116 blocks with 'EP0 not after r output'.
Cause: it assumed EP0 sat one line after the 'AudioEngineHarness+0x1f9c550:' echo.  Two lines
sit between them, the echo and the instruction line, so EP0 is at m+2, not m+1.  A fixed
positional assumption where a scan was required.
Fix: scan forward for the EP0 and WIN_END witnesses instead of assuming an offset.
Result: 116 of 116 blocks recovered, anomalies none.

R2 and R6 were unaffected throughout.  They are whole-line counts with no positional assumption,
and they reported 116 / 116 / 116 / 116 and 0 ordering violations on the first pass.
```

The artifact did not change. The run is the only one, and it was not repeated.

## 9. State

```text
r9 raw value              OBTAINED, 13 distinct values, never 9
r9 meaning                enqueue epoch, 3rd parameter of DeferredDeletionQueue::enqueue
window+0x28               IDENTIFIED as the live global epoch
enqueueEpoch vs globalEpoch   delta in {0, +1}, 116 of 116
C3_T0_ENTRY               0
DG-T0-Repair-3b           CLOSED, at Step5DK
predicate '@r9 == 9'      UNCHANGED, and its validity is UNRESOLVED
T1 repair                 NOT AUTHORIZED, 0x1f9ce04 and 0x1f9cfb0 still malformed and unreached
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D   UNRESOLVED
CDB execution             1, consumed, Retry 0
build / source / test / CMake / harness   untouched
IMPLEMENTATION            FORBIDDEN
```

## 10. The next gate, and the order that is not compressed

```text
Capture Result, this document
      |
      v
Intent / Semantic Audit
      what was conjunct 2 meant to assert?  A literal snapshot, or a relation?
      |
      v
predicate validity determined
      |
      v
Repair Design Gate
      |
      v
separate Implementation Authorization
```

The Intent / Semantic Audit is the gate that must now answer the question this programme was
chartered to answer, and the answer cannot be reached by further measurement of `@r9`. It requires
the intent of the original capture: whether conjunct 2 was written to compare against a fixed epoch
snapshot or against the live epoch. Two things are already on the table for it:

```text
the literal 9 coincided with the live global epoch at the only observation where the gate
opened, Step5BB Retry-1, where r9 and EpochDomain+0x18 were both 9

and the enqueue epoch tracks the live global epoch, or trails it by exactly one, in 116 of 116
```

Whether those two facts make `9` a mis-transcription or a deliberate snapshot is a question about
intent, and it is not answerable from this log.
