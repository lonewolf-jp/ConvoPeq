# C3 Entry-Gate Diagnostic — Redesign-5 Single Execution / Result Audit

## 1. Gate result

```text
Gate   = P3-5-FPM-C3-T1-Capture-Redesign-5-Single-Execution-Result-Audit
CDB executions = 1, the single authorized measurement
re-runs        = 0
harness        = launched, then terminated while stopped at a breakpoint

R1  CDB parse / execution              PARTIAL, see 3
R2  T0E  0x1f9c550 reached             YES, 1 hit
R3  T0P  0x1f9c57d reached             NOT DETERMINED
R4  T0Q  0x1f9c59a reached             NOT DETERMINED
R5  T0S  0x1f9c5da reached             NOT DETERMINED
R6  operand values                     CAPTURED for T0E only, 1 partial sample
R7  conjunct PASS/FAIL markers         NOT EMITTED, the verdict block never executed
R8  R9CHANGE                          0
R9  T0 entry condition                did not become true at the one sample obtained
R10 T1 reachable                      NO, and not evaluable

BRANCH = B.  site reached, guard conjuncts false.
          A is excluded for T0E.  C and D are excluded.

CAUSE OF THE PARTIAL MEASUREMENT = a CDB command-syntax defect in the diagnostic
                                   preamble written at this gate. Not an invocation error,
                                   not a breakpoint-definition parse error, not a target defect.
```

## 2. Execution record

```text
script  doc/work113/P1-5-IR-P2_Step5CT_…_Redesign-5_EntryGate-Diagnostic.cdb
SHA     247B6A71752AC1986600A52A5D5F4EFC84DDACD1DC8A27203EDB78657C45B47F   unchanged
bytes   17,427
target  build/Release/AudioEngineHarness.exe
SHA     E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
args    --measurement=normal
cdb     tmp/cdb.exe 10.0.29617.1000, SHA 5F54ABAF…FBEE67
form    cdb.exe -cf <script> -logo <log> <exe> --measurement=normal

started 2026-09-26 14:18:57   ended 14:18:57   elapsed 0.3 s   exit 0
log     doc/work113/P1-5-IR-P2_Step5CU_…_Redesign-5_Single-Execution_CDB.log
        43,114 bytes, 189 lines
```

The `-o` misuse of Step5CO did not recur. `-logo` was used and the log was produced.

## 3. R1, partial, and precisely why

```text
SCRIPT_BEGIN                        1
SCRIPT_END                          1
breakpoints set, bl rows            13 of 13, all resolved
'quit:'                             1
'Extra character error'             1
ZwTerminateProcess reached          0
elapsed                             0.3 s   (Redesign-4 run: 8.7 s)
```

What succeeded:

```text
all 13 breakpoint definitions were accepted, with the preamble text embedded and visible
in the bl listing, so the bp command lines themselves parsed
the debuggee ran, and the T0E breakpoint was reached and its preamble began executing
```

What failed, in the order it happened, taken from log lines 175 to 189:

```text
175  C3_DG_T0E_HIT                              .echo executed
176  Evaluate expression: 0                      ? @$t0
177  Evaluate expression: 1                      ? @r9
178  Evaluate expression: 774250685712            ? @r10
179  Evaluate expression: 774250685712            ? @rbx
180  Evaluate expression: 2595993372032           ? @rcx
181  Evaluate expression: 140710351988503         ? @r11
182  Evaluate expression: 221                     ? dd(@rcx+0x34000)
183  ^ Extra character error in ' r $t16 = @r9 ; .echo C3_DG_T0E_HIT'
184  AudioEngineHarness+0x1f9c550:                current frame, debuggee still stopped
185  00007ff7`7f1ec550  mov qword ptr [rsp+8],rbx
186  0:000> .echo C3_T1_REDESIGN2_SCRIPT_END
188  0:000> q
189  quit:
```

The preamble declares eight `?` commands; seven produced output. The eighth,
`? dd(@rcx+0x30000)`, did not, and neither did the whole per-conjunct verdict chain that
follows it.

The consequence is the important part. Because the body aborted, the terminal `gc` inside the
existing guard never ran. The debuggee stayed stopped at `0x1f9c550`. CDB then consumed the two
remaining script lines, `SCRIPT_END` and `q`, and quit, which terminated the harness while it was
stopped. Hence `ZwTerminateProcess` never appeared, elapsed time collapsed to 0.3 s, and the
harness telemetry stops mid `prepareToPlay`. The measurement is therefore truncated by our own
defect, not by the target.

### 3.1 What is not determined

The reported span and the observed execution order do not fit a simple "parsing stopped here"
reading: `r $t16 = @r9` is named in the error, yet the `.echo` in the same named span demonstrably
executed, and seven later commands executed after it. This audit therefore does **not** claim a
mechanism for the error, and does not claim which single token CDB objected to. Establishing that
would require a probe, which is not authorized and was not performed.

## 4. R2, the primary question, answered for T0E

```text
C3_DG_T0E_HIT   1 occurrence, whole trimmed line equality
```

**The instruction at `0x1f9c550` was executed.** This is the discrimination the whole gate was
built to obtain, and for this site it is unambiguous:

```text
breakpoint NOT reached          ruled out for 0x1f9c550
breakpoint reached, guard false confirmed, see 6
```

The two prior candidate explanations are separated for T0E. The previous run could not tell them
apart; this one can.

## 5. R3, R4, R5, reachability of the other three sites

```text
C3_DG_T0P_HIT   0
C3_DG_T0Q_HIT   0
C3_DG_T0S_HIT   0
C3_T0_ENTRY / C3_T0_CAS_PRE / C3_T0_CAS_POST / C3_T0_SEQUENCE_PUBLISHED   all 0
```

These zeros are **NOT** evidence that `0x1f9c57d`, `0x1f9c59a` or `0x1f9c5da` were not reached.
The debuggee was killed while stopped at the *first* T0E hit, before control could reach any later
site. Their reachability is **unknown**, and the correct entry for R3, R4 and R5 is NOT DETERMINED,
not "no".

## 6. R6 and R7, the measured values

Seven raw values were captured. The per-conjunct PASS/FAIL markers were not emitted, because the
verdict block never executed. The evaluation below is therefore **derived by hand from the captured
raw values**, and is labelled as such. It is not a marker-measured verdict.

T0E sample 1, at `0x1f9c550`:

| operand | captured value | guard conjunct at this site | holds? |
|---|---|---|---|
| `$t0` | `0` | `@$t0 == 0` | **YES** |
| `@r9` | `1` | `@r9 == 9` | **NO** |
| `@r10` | `0x000000b4`44efe510` | not a conjunct here | — |
| `@rbx` | `0x000000b4`44efe510` | not a conjunct here | — |
| `@rcx` | `0x0000025c`6d434580` | base for the DQueue read | — |
| `@r11` | `0x00007ff9`ae8be717` | not a conjunct here | — |
| `dd(@rcx+0x34000)` | `221` (`0xDD`) | `== 4` | **NO** |
| `dd(@rcx+0x30000)` | not captured | not a conjunct here | — |

So at this one sample the T0 entry condition failed on **two** conjuncts: `@r9` held 1 rather than
9, and the enqueue position held 221 rather than 4. The `$t0` conjunct held.

### 6.1 The Design section 4.2 correction is corroborated at runtime

This is a real confirmation, and it is the reason the per-gate operand selection was made as it was.

```text
exe image base    0x00007ff7`7d250000
captured @r11     0x00007ff9`ae8be717      a different module range, not the exe
Design claim      at 0x1f9c550, `movq %rcx, %r11` at 0x1f9c55c has NOT yet executed,
                  so @r11 is the caller's value, not the queue base
```

The captured `@r11` is indeed not an address in the executable, exactly as the corrected table
predicted. Had the preamble used `@r11`-relative reads at this site, as the other three sites do,
it would have read through a wild pointer. It used `@rcx`, and `dd(@rcx+0x34000)` returned a
plausible enqueue position of 221, so `@rcx` does behave as a queue base at this site. The
correction and the per-gate base choice are both validated by measurement.

### 6.2 What one sample does not license

The preamble was designed to collect up to eight samples per site and to track every distinct
`@r9` value. The abort yielded **one** sample, of which the eighth read and all verdicts are
missing. Therefore:

```text
NOT claimed  that @r9 is never 9
NOT claimed  that the enqueue position never equals 4
NOT claimed  that 9 is the wrong constant
NOT claimed  that 4 is the wrong constant
NOT claimed  that the entry window was missed rather than mis-specified
```

A single sample at the first hit of a site says what the state was at that instant and nothing
about the states that follow.

## 7. R8, R9, R10

```text
R8   C3_DG_*_R9CHANGE occurrences = 0
     consistent with a single sample and no transition, and it proves nothing further,
     because the transition test sits after the abort point in the body

R9   C3_T0_ENTRY = 0.  The entry condition did not become true at the one sample obtained.

R10  no C3_T1 marker, 0.  T1 was not reached.  Not evaluable, and not a statement about T1:
     $t0 never became 1, so every T1 gate was correctly closed by its own precondition.
```

## 8. Branch

```text
A  all four sites no HIT                 EXCLUDED for T0E, which was hit
B  HIT present, guard conjunct false     THIS BRANCH
C  T0E all conjuncts PASS                EXCLUDED, two conjuncts were false
D  T1 evidence reached                   EXCLUDED
```

Branch B, with the measured values in section 6 as the basis. Per the owner's ordering, the next
step is to fix the cause of the entry condition using those measured values. This audit does not
perform that fix and does not propose specific new constants, because one sample cannot
distinguish a wrong constant from a window that simply had not arrived yet.

## 9. What is now settled, and what is not

Settled by this run:

```text
the T0 entry region IS executed during the measurement window        PROVEN
0x1f9c550 is reached at least once, before teardown                   PROVEN
the existing T0 guard is therefore never the reason nothing was seen  at this site the
                                                                       reason is conjunct
                                                                       falsity, and the entry
                                                                       condition was false
@r11 is not the queue base at 0x1f9c550                               PROVEN, runtime
                                                                       corroboration of the
                                                                       Design 4.2 correction
@c3 behaves as a queue base at 0x1f9c550                              PROVEN
T1 was correctly inert because $t0 never became 1                     PROVEN, trivially
```

Not settled:

```text
reachability of 0x1f9c57d, 0x1f9c59a, 0x1f9c5da                     unknown
whether @r9 ever equals 9 at any of the four sites                    unknown
whether enqueuePos ever equals 4, or 5, at the CAS sites             unknown
sequences[slot] at any site                                          not captured
the mechanism of the Extra character error                            not determined
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D           unchanged, unevaluated
```

## 10. The defect this run exposed in our own work

The preamble written at Step5CT does not execute to completion on this CDB build. The specific
failure mode is a CDB command-syntax error inside a breakpoint body, and it has a consequence
worse than a wrong reading: because the body aborts before its terminal `gc`, the debuggee is left
stopped and the run dies.

Two properties of the existing, working script are relevant and were not applied to the preamble:

```text
the existing bodies write registers in the compact form
    r $t13=@$t13+1 ; r $t11=@$tid ; r $t12=1
with no spaces around the assignment operator, and these are proven at runtime,
since C3_T1A_CANDIDATE and later markers have executed in earlier work

the preamble instead used the spaced form
    r $t16 = @r9 ; r $t17 = @$t17 + 1
which is the form the top-level script lines use, where each command owns its own line
```

Whether the spacing is the cause is **not** claimed here, for the reason in 3.1. What is claimed
is that the preamble diverges from a form already proven to work in the same file, on the same
build, and that this is where the divergence should be looked first.

A second, independent design weakness is visible in the same log and should be corrected at the
same time:

```text
any future body that can abort before its terminal gc will kill the run in this way
a body whose diagnostic prefix can fail must not be placed in front of a required gc
```

## 11. Constraints honored

```text
.cdb modified            0, SHA unchanged
ConvoPeq.md / harness / production source / test / CMake / build   untouched
$t0..$t19 design         unchanged
breakpoint topology      unchanged, 13
T1 correlation           unchanged
ReaderSlot windows       unchanged
A2 'dd @$t1 L1'          not re-edited
canary                  not back-added
additional executions    0
Design canary            not added, as recorded at Step5CT
```

## 12. Final state

```text
Redesign-5 single execution + Result Audit
= CLOSED / R1 PARTIAL / R2 YES / R3-R5 NOT DETERMINED / BRANCH B

log    doc/work113/P1-5-IR-P2_Step5CU_…_Redesign-5_Single-Execution_CDB.log
       43,114 bytes, 189 lines, exit 0, elapsed 0.3 s

NEW PROVEN      0x1f9c550 is executed; the T0 entry region runs
NEW PROVEN      the entry condition was false at that sample: @r9 = 1, enqueuePos = 221
NEW PROVEN      @r11 is not the queue base at 0x1f9c550, runtime corroboration of Design 4.2
NEW DEFECT      the diagnostic preamble aborts with a CDB Extra character error, and the
                abort strands the debuggee because the body never reaches its terminal gc

A1 separator    = RUNTIME PROVEN, unaffected
A2 globalEpoch  = still installed, still execution count 0
T0 gate reason  = PARTIALLY RESOLVED.  Reached, conjunct false, values measured.
                   Reachability of the three later sites and the behaviour over time
                   remain open.
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```

## 13. What the next gate must decide, not assumed here

The preamble must be made to execute to completion before any further question about the entry
condition can be answered, because the current failure destroys the measurement. That is a script
change and needs its own design and authorization.

Only after a complete run exists can the entry condition itself be adjudicated on evidence, since
one sample cannot distinguish a mis-specified constant from a window not yet reached. No new
constant is proposed here, and `@r9 == 9` remains unmodified and unjudged.
