# Step5DK — Runtime Result Audit, T0-Guard-Repair-Baseline-1

```text
gate                = single authorized runtime execution
script              = P1-5-IR-P2_Step5DI_…_T0-Guard-Repair-Baseline-1.cdb
script SHA-256      = 74E8C64CD346BA981A1ABD2E1A941B2EDB31E1932A7D586C7D683C1E6EBF9142
script bytes        = 14,895
log                 = P1-5-IR-P2_Step5DI_…_T0-Guard-Repair-Baseline-1_Runtime_CDB.log
                     61,500 B, 1,056 lines
invocation          = cdb.exe -cf <target> -logo <log> AudioEngineHarness.exe --measurement=normal
execution_count     = 1        Retry-1 / 2 / 3 = NOT used
exit_code           = 0
elapsed             = 10.5 s
R1                  = PASS, 6 of 6
DG-T0-Repair-3b     = CLOSED
T0 gate cause       = conjunct 2, in this run
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

Error census, every pattern zero:

```text
Syntax error 0   Extra character 0   Illegal 0   Invalid 0   Undefined 0
^Error 0         Cannot 0            Memory access error 0
Unable to read memory 0              Couldn't 0            Evaluate expression: 0
'Unable' any 4, classified benign: 3 extension DLL loads and 1 checksum line
```

The repair holds continuation across all four T0 sites. This is the first clean run in the
lineage since Redesign-6, and the first in which the T0 conjunct 3 was reachable at all.

## 2. R2, dwo / T0E evidence

```text
C3_DG_T0E_HIT             116
C3_DG_T0E_R9_PASS           0
C3_DG_T0E_R9_FAIL         116
C3_DG_T0E_EQ_PASS           4
C3_DG_T0E_EQ_FAIL         112
C3_T0_ENTRY                 0
C3_T0_SEQUENCE_PUBLISHED    0
C3_T0_CAS_PRE / _POST / _WRITE_DONE   0 / 0 / 0
```

Every one of the 116 T0E hits emitted both verdicts. The payload completed at every hit.

### 2.1 DG-T0-Repair-3b is CLOSED

```text
Syntax error count         = 0     no parse-stage rejection
Memory access error count  = 0     no evaluation-stage failure
EQ_PASS emissions         = 4     the comparison was SATISFIED
```

3b asked whether `dwo` can successfully read a valid address. It can, and this is stronger than the
absence of an error: `dwo(@rcx+0x34000) == 4` evaluated **true at 4 of 116 hits**, which requires
that `dwo` returned the actual counter value, not merely that it parsed. `enqueuePos` was 4 at those
4 hits.

```text
DG-T0-Repair-3a   dwo is a syntactically accepted .if operand            PROVEN  (Step5BK)
DG-T0-Repair-3b   dwo returns a correct value from a valid address        NOW PROVEN
```

The chain is closed: the operator is valid, it parses, it reads, and its value satisfies the
comparison that the frozen guard uses.

## 3. R3, T0E to T0P to T0Q to T0S traversal

```text
T0E  C3_DG_T0E_HIT = 116
T0P  C3_DG_T0P_HIT = 116
T0Q  C3_DG_T0Q_HIT = 116
T0S  C3_DG_T0S_HIT = 116
all four equal = True
```

116 of 116. Checked, not assumed. This is the second independent run to produce equal counts, the
first being Redesign-6, so reproducibility moves from n=1 to n=2 for the count, though still n=1 for
this particular baseline.

## 4. R4, per-hit pairing

```text
T0E hits, whole-line count          116
complete verdict triples recovered  116
triples with both verdicts adjacent 116
T0E hits left unpaired              0
```

Pairing is by ordinal position within the same breakpoint body, so a verdict cannot be attributed to
the wrong hit. Every hit produced exactly one `R9_*` and exactly one `EQ_*`, and they were adjacent
in the log at every hit.

## 5. R5, four-way discrimination

| conjunct 2 `@r9 == 9` | conjunct 3 `enqueuePos == 4` | hits | meaning |
|---|---|---|---|
| FAIL | FAIL | 112 | both false |
| FAIL | PASS | 4 | r9 conjunct false, enqueuePos conjunct held |
| PASS | FAIL | 0 | not observed |
| PASS | PASS | 0 | not observed |

```text
conjunct 1  @$t0 == 0      TRUE, frozen from Redesign-6
conjunct 2  @r9 == 9       FALSE at 116 of 116 hits
conjunct 3  enqueuePos == 4 TRUE at 4 of 116, FALSE at 112
gate opened C3_T0_ENTRY    0
```

### 5.1 What this does and does not establish

```text
ESTABLISHED   conjunct 3 is reachable and evaluable, and it is satisfiable.
              It was TRUE at 4 hits. It is not a permanent blocker.

ESTABLISHED   conjunct 2 is the blocker in this run. It was FALSE at every one of 116 hits,
              including the 4 hits where conjunct 3 held. So at those 4 hits the gate was
              blocked by conjunct 2 alone, with conjunct 3 satisfied.

NOT ESTABLISHED  that '@r9 == 9' is the wrong constant.
                  r9 was 1 in Redesign-5, 9 at the T0 entry in Step5BB Retry-1, and not 9 at
                  all 116 hits here. The value is context dependent across runs. Whether 9 is
                  the correct expected value is a separate question this run does not answer,
                  and the standing constraint forbids asserting either way.

NOT ESTABLISHED  that enqueuePos is 4 in general. 4 held at 4 of 116 hits. The other 112
                  hits returned something else, and the raw values were not captured, so the
                  distribution is unknown. Redesign-7's single 221 sample is unaffected and
                  still n=1.

NOT ESTABLISHED  that the 4 conjunct-3-true hits are a distinct class of work. They were not
                  distinguished by any captured operand, and no raw value was taken.
```

### 5.2 Correction to the Step5DF conclusion

Step5DF recorded conjunct 3 as **"never evaluable"**, which was correct at that time and remains
correct as a statement about the `dd(` form. With the `dwo` repair it is evaluable and sometimes
true. The T0 gate cause is therefore **more precisely** stated as:

```text
under the dd( form    conjunct 3 was a permanent structural blocker; the gate could not open
under the dwo form    conjunct 3 is satisfiable; the blocker in this run is conjunct 2
```

## 6. R6, strict interleaving

```text
T0 markers in the stream          464
complete 4-cycles                 116
aligned at a multiple of 4        True
cycles violating T0E < T0P < T0Q < T0S   0
```

Ordering verified from this log, not inherited from the previous run.

## 7. T1-phase reachability, observation only

| marker | count |
|---|---|
| `C3_T1A_CANDIDATE` | 0 |
| `C3_T1_SELECTION` | 0 |
| `C3_T1_CAS_PRE` | 0 |
| `C3_T1_CAS_SUCCEEDED` | 0 |
| `C3_T1_EPOCH_BLOCKED` | 0 |
| `C3_CR1_STOP_T1_RETURN_POSITION_CONTRADICTION` | 0 |

```text
0x1f9cfb0 reached = False
0x1f9ce04 reached = False
```

The residual T1-phase risk recorded in `DG-T0-Repair` section 5.2 is **retired for this run**. Neither
malformed site was reached, so neither could abort. This is the observation that choosing scope A'
was designed to produce, and it was obtained without repairing anything on the T1 side.

This also explains why R1 passed cleanly: the repair covered the whole enqueue CAS bracket, and the
bracket is the only phase the run entered.

## 8. Tooling defects in this gate

Two, both mine, both in the audit rather than in the measurement. Neither changed a whole-line count,
and every substantive number below is label-free and independently recomputed.

| # | defect | consequence | correction |
|---|---|---|---|
| 1 | The R1 verdict line printed `FAIL` on the first pass. The error census was accumulated over the error-pattern list, all of which were zero, and then the benign `Unable` count of 4 was **subtracted** from that total, producing `error-census total = -4` and a negative count, which is impossible | R1 was reported as FAIL when all six substantive criteria were met | the census is now the sum over the error list alone; `Unable` is classified separately and not subtracted. R1 re-evaluated as PASS 6 of 6 |
| 2 | The R4 pairing scan reported "39 of 116" pairs. The loop was bounded by `len(marker_lines)` = 464 while indexing the full log line list, so it stopped early | an alarming undercount of the pairing | rebuilt over the full log. All 116 hits pair, all verdicts adjacent, 0 unpaired |
| 3, minor | The rebuilt scan printed a long list of `C3_DG_T0P/Q/S_HIT` entries as "structural anomalies" | misleading output; these markers legitimately follow each consumed verdict triple and are the normal cycle continuation | the correct reading is 0 anomalies. The label was wrong, not the data |

The recurring family is unchanged: a total assembled from one population and compared against a
count taken from another, and a loop bounded by the wrong collection. Both are proxies standing in
for the property.

## 9. State

```text
T0 gate cause            conjunct 2, @r9 == 9, FALSE at 116 of 116 hits in this run
conjunct 1 @$t0 == 0     TRUE, frozen
conjunct 2 @r9 == 9      FALSE 116/116
conjunct 3 enqueuePos    TRUE 4/116, FALSE 112/116
C3_T0_ENTRY              0
DG-T0-Repair-3b          CLOSED
residual T1 risk         RETIRED for this run; 0x1f9cfb0 and 0x1f9ce04 not reached
enqueuePos raw value     NOT OBTAINED, booleans only
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D   UNRESOLVED
CDB execution            1
.cdb artifacts           14, none modified
build / source / test / CMake   untouched
IMPLEMENTATION           FORBIDDEN
RUNTIME                  consumed
```

## 10. What the next gate has to decide

The T0 path is now clean end to end and its blocker is identified, so the T1 phase is reachable in
principle. Three questions are open and none is answered by this run:

```text
1. Is 9 the correct expected value for r9 at 0x1f9c550, or is the constant itself wrong?
   r9 read 1 in one run, 9 in another, and not 9 in 116 hits here. This needs a value, not a
   boolean, and the boolean route has now been exhausted.

2. Does the T1 phase need the same dd( -> dwo( repair before it can be entered?
   0x1f9ce04 has 2 occurrences inside .if conditions and 0x1f9cfb0 has 2 inside display-command
   address expressions. Both are still malformed. Neither was reached, so this run says nothing
   about whether they would abort.

3. Which of the two should be repaired first, given that repairing 0x1f9ce04 without 0x1f9cfb0,
   or the reverse, may or may not be coherent for the same reason that scope A' had to cover all
   four T0 sites together.
```

Recommendation for the next Design Gate: question 1 first. Questions 2 and 3 are T1 repair scope,
and the A' precedent, that a partial repair of one phase strands the run one site later, applies
directly to them.
