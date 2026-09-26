# Redesign-7 Runtime Result Audit

## 1. Execution

```text
script   doc/work113/P1-5-IR-P2_Step5DA_…_Redesign-7_Operand-Capture.cdb
SHA-256  5AA98A592DD5B8745317A439AEC5BBC8B564A207D1447BC1E819504015E4C5F9   unchanged after the run
target   build/Release/AudioEngineHarness.exe
SHA-256  E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
form     cdb.exe -cf <script> -logo <log> <exe> --measurement=normal

execution_count = 1
exit_code       = 0
elapsed         = 0.3 s
log bytes       = 37,424
log lines       = 183
log             doc/work113/P1-5-IR-P2_Step5DB_…_Redesign-7_Runtime_CDB.log
```

`-logo` was used. No re-run. No Retry-1, Retry-2 or Retry-3.

## 2. R1

```text
SCRIPT_BEGIN       = 1
SCRIPT_END         = 1
ZwTerminateProcess = 0
quit               = 1
exit code          = 0
elapsed            = 0.3 s      against 10.9 s for the same harness under Redesign-6
R1 VERDICT         = FAIL
```

## 3. R2, R3, R4, witness census

```text
R2  C3_DG_T0E_HIT  = 1
R3  C3_DG_T0E_EQ1  = 0
R4  C3_DG_T0E_EQ2  = 0
    'Evaluate expression:' lines = 1
```

## 4. R5, R6, operand values

```text
R5  per-hit enqueuePos = ONE sample, 221 (0x000000DD)
R6  per-hit r9         = NO samples, zero
```

The single captured line, verbatim from the log:

```text
Evaluate expression: 221 = 00000000`000000dd
```

## 5. R7, witness correspondence and attribution

> **CORRECTION, issued at the Redesign-8 gate.** The attribution in the first version of this section
> was wrong. It read "the command between HIT and EQ1 is the first `?`, therefore the second `?`
> failed". The command between HIT and EQ1 **completed**, so it cannot be the failing one. The
> corrected attribution is below and it changes the constraint materially. Sections 10, 12 and 13 of
> this document are affected where they state the "one `?` per body" bound.

The failure site, verbatim, log lines 174 to 180:

```text
[174] [EQ_PREPARE] agc tables allocated
[175] C3_DG_T0E_HIT
[176] Evaluate expression: 221 = 00000000`000000dd
[177]                                  ^ Extra character error in '.echo C3_DG_T0E_HIT'
[178] AudioEngineHarness+0x1f9c550:
[179] 00007ff7`7f1ec550 48895c2408      mov     qword ptr [rsp+8],rbx
[180] 0:000> .echo C3_T1_REDESIGN2_SCRIPT_END
```

The payload, command by command:

| # | command | observed |
|---|---|---|
| 1 | `.echo C3_DG_T0E_HIT` | **emitted**, line 175 |
| 2 | `? dd(@rcx+0x34000)` | **completed**, line 176 emitted the evaluation |
| 3 | `.echo C3_DG_T0E_EQ1` | **not emitted**, absent from the entire log |
| 4 | `? @r9` | never reached |
| 5 | `.echo C3_DG_T0E_EQ2` | never reached |
| 6 | frozen guard, ending in `gc` | never reached |

The witness bracket is `HIT | [ ? dd ] | EQ1`. The bracketed `?` demonstrably completed and the
closing witness is absent. Therefore **the command that did not complete is command 3, the first
command following a completed `?`.**

Confirmation that `EQ1` was never emitted, rather than emitted in another form: the only whole-line
`C3_DG_*` occurrence in the log is `C3_DG_T0E_HIT`. The two lines that merely *contain* the strings
`EQ1` and `EQ2` are the `bp` command echo and the `bl` listing, which is exactly the exclusion the
whole-line marker rule exists to make.

**The constraint is therefore not "at most one `?` per body". It is:**

```text
NO COMMAND MAY FOLLOW A COMPLETED '?' IN A BREAKPOINT BODY.
```

The command that failed was an `.echo`, the construct with the strongest evidence in this file, not a
second `?`. The failure is a property of the **position after a `?`**, not of the `?` construct.

### 5.0 Consequence for the option set

```text
'?' can only ever be the LAST command of a body.
A body whose last command is '?' has no guard and therefore no terminal gc, so the debuggee
is stranded.  Therefore '?' is unusable in any body that must continue.

Option A  '.echo ; ? ; guard'   the guard follows the '?'   -> ELIMINATED
Option B  same shape in each run                           -> ELIMINATED
Option C  no '?' at all, .if and .echo only                -> the only surviving option
```

This is a stronger statement than "A carries an unproven position". A and B are not merely riskier
than C; they are structurally excluded by the corrected constraint.

### 5.1 The debugger's own report is unreliable, and this is now confirmed twice

The error names `.echo C3_DG_T0E_HIT`, which is on line 175 and demonstrably executed. The failure is
one and two commands later.

```text
Redesign-5   error named ' r $t16 = @r9 ; .echo C3_DG_T0E_HIT', and the .echo in that span had run
Redesign-7   error named '.echo C3_DG_T0E_HIT', and that .echo had run
```

Two independent runs, the same pattern: the reported span does not localise the failure, and in both
cases it names a command that completed. The caret position and the quoted fragment are therefore
not usable as an attribution channel on this build.

**The witness markers are.** They are produced by the proven `.echo` class and they bracket each new
command, and in both runs they are the only channel that located the failure precisely. That is the
whole value of the Design Gate section 5.3 design, and it is now paid for.

## 6. Consequence of the abort

```text
the body's terminal gc, inside the frozen guard, was never reached
the debuggee was left stopped at 0x1f9c550
CDB consumed the remaining script lines, SCRIPT_END and q, and quit
the harness was terminated while stopped
ZwTerminateProcess never appeared
harness telemetry stops mid prepareToPlay
elapsed collapsed to 0.3 s
unexpected breakpoint stop = 1
```

Identical to the Redesign-5 failure mode. The diagnostic reduction from twelve unproven commands to
two did not prevent it; it made it locatable.

## 7. Error census

```text
Extra character error     = 1
Syntax error              = 0
Illegal / Invalid         = 0
Undefined                 = 0
^Error                    = 0
Unable to read memory     = 0
Cannot                    = 0
Unable (any)              = 4    3 extension DLLs plus the image-checksum line, benign, unchanged
WARNING                   = 1    image checksum, unsigned, unchanged
unexpected breakpoint stop = 1
```

## 8. T0 markers and downstream, not interpreted

```text
C3_T0_ENTRY               = 0
C3_T0_CAS_PRE             = 0
C3_T0_CAS_POST            = 0
C3_T0_SEQUENCE_PUBLISHED  = 0
C3_CR1_STOP_*             = 0
C3_*_SLOTS_BEGIN          = 0
T1 capture markers        = 0
```

A raw pattern match for `^C3_T1` returns 2, and both are the script lifecycle markers
`C3_T1_REDESIGN2_SCRIPT_BEGIN` and `C3_T1_REDESIGN2_SCRIPT_END`. The actual T1 capture markers,
`C3_T1A`, `C3_T1B`, `C3_T1C`, `C3_T1_SELECTION` and `C3_T1_CAS`, number zero. This is recorded so the
count of 2 is not misread later as T1 evidence.

None of the above is interpreted. R1 failed, and the run died at the first T0E hit.

## 9. Result

```text
Runtime Result   = FAIL
T0 gate cause    = UNRESOLVED
Implementation   = FORBIDDEN
```

## 10. What this run establishes, and what it does not

Established by measurement:

```text
PROVEN   '?' as a body command executes and produces a value. 'Evaluate expression: 221'
         at Redesign-7 line 176.
PROVEN   NO COMMAND AFTER A COMPLETED '?' EXECUTES.  The command that failed was
         '.echo C3_DG_T0E_EQ1', a construct with the strongest evidence in this file.
         The constraint is positional, not a property of '?'.
CORRECTED  the earlier reading of this section, that '?' is unsafe when repeated, is withdrawn.
         See section 5 for the corrected derivation and its consequences.
PROVEN   the witness-marker design localises a failure to a command bracket.  In this run it
         correctly bracketed the failure between the completed '?' and the absent EQ1, which is
         precisely what identified the real cause.
PROVEN   the enqueuePos observation, '? dd(@rcx+0x34000)', is well formed and readable.
         It returned 221 at the first T0E hit.
```

Not established:

```text
NOT  the value of @r9 at any hit. Zero samples.
NOT  the range of enqueuePos over the window. One sample.
NOT  whether enqueuePos ever equals 4.
NOT  whether @r9 ever equals 9.
NOT  which conjunct of the T0 entry guard is false.
```

The single enqueuePos value of 221 reproduces the Redesign-5 sample exactly. That is corroboration of
one sample by an independent run, and nothing more. It remains one sample, taken at the first hit,
and it does not bound the counter's range over the 116-cycle window. The hypothesis that the pinned
value 4 is unreachable is neither confirmed nor excluded by this run.

## 11. Frozen items, unchanged

```text
S7_READER      = UNRESOLVED
S7_READER_SLOT = UNRESOLVED
minReaderEpoch = NOT CAPTURED
Case A / B / C / D = NOT PROVEN
T0 gate cause  = UNRESOLVED
A1 separator   = RUNTIME PROVEN
A2             = installed, accepted, execution count 0, globalEpoch value UNPROVEN
D6-10 / D6-11  = PROVEN under the Redesign-6 single-command payload.
                 NOT re-proven under the enlarged payload; this run failed before the guard.
IMPLEMENTATION = FORBIDDEN
```

Per the instruction, this gate stops here. No re-run, no script edit, no guard change, no constant
change, and no production source work.

## 12. Final state

```text
Redesign-7 Runtime Result Audit
= CLOSED / R1 FAIL / attribution complete

execution_count = 1   exit_code = 0   elapsed = 0.3 s   log 37,424 bytes / 183 lines
R1  SCRIPT_BEGIN 1, SCRIPT_END 1, ZwTerminateProcess 0, quit 1, debugger_error 1  -> FAIL
R2  C3_DG_T0E_HIT = 1
R3  C3_DG_T0E_EQ1 = 0
R4  C3_DG_T0E_EQ2 = 0
R5  enqueuePos = one sample, 221
R6  r9         = no samples
R7  attribution: no command after a completed '?' executes.  The failing command was
    '.echo C3_DG_T0E_EQ1', not a second '?'.  '?' is therefore unusable in a body that
    must reach its terminal gc.
T0_ENTRY = 0    T0_SEQUENCE_PUBLISHED = 0
T1 capture markers = 0
S7_READER / S7_READER_SLOT / minReaderEpoch / Case A-D = UNRESOLVED
error census: Extra character 1, all other error classes 0, benign Unable 4, WARNING 1

Runtime Result = FAIL
T0 gate cause  = UNRESOLVED
Implementation = FORBIDDEN
```

## 13. What the next gate must decide, not assumed here

The design question this run raises is narrow and it is a design decision, not an execution plan. One
`?` per body is proven safe. The objective requires two values per hit, and two `?` in one body is
proven unsafe. Those two facts are in tension, and resolving them is a design choice about how to
obtain two values per hit without placing two `?` in one body. Options exist, for example obtaining
the second value at a different site, or accepting one value per hit across two authorized runs. None
of them is selected here, none is assumed correct, and no script is edited on the strength of this
run.
