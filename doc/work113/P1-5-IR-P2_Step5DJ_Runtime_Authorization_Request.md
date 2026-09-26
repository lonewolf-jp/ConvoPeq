# Runtime Authorization — Request

```text
gate                = Runtime Authorization REQUEST.  Not granted.  Not executed.
target              = P1-5-IR-P2_Step5DI_…_T0-Guard-Repair-Baseline-1.cdb
target SHA-256      = 74E8C64CD346BA981A1ABD2E1A941B2EDB31E1932A7D586C7D683C1E6EBF9142
target bytes        = 14,895      41 lines, ASCII, no BOM, CRLF, trailing CRLF
preconditions       = Step5DH authorization, Step5DI implementation, Static Validation 17/17 PASS
status              = AWAITING OWNER AUTHORIZATION
CDB execution to date for this artifact = 0
```

## 1. Invocation

```text
cdb.exe -cf <target> -logo <log> build\Release\AudioEngineHarness.exe --measurement=normal
```

`-logo`, never `-o`. The log path will be resolved from the target's directory at execution time
rather than transcribed.

```text
execution_count   1
Retry-1/2/3       FORBIDDEN
.cdb edit         FORBIDDEN
source / test / CMake / build   FORBIDDEN
```

## 2. R1, judged first and alone

```text
SCRIPT_BEGIN       = 1
SCRIPT_END         = 1
ZwTerminateProcess = 1
quit               = 1
exit code          = 0
debugger errors    = 0
```

Error census must be empty of `Syntax error`, `Extra character`, `Illegal`, `Invalid`, `Undefined`,
`^Error`, `Cannot`, and `Memory access error`. `Unable` lines from extension DLL loading are benign
and are counted separately.

```text
R1 FAIL  ->  STOP.  Attribute the failure from the log using witness markers, close the Result
             Audit, do not interpret R2 to R6, do not retry.
R1 PASS  ->  and only then evaluate R2 to R6.
```

## 3. R2 to R6, conditional on R1 PASS

```text
R2   dwo / T0E evidence
R3   T0E -> T0P -> T0Q -> T0S traversal
R4   per-hit pairing
R5   four-way discrimination
R6   strict interleaving
```

## 4. What each outcome would establish

```text
gate OPENS, C3_T0_ENTRY emitted
    conjuncts 1, 2 and 3 all held at that hit
    DG-T0-Repair-3b CLOSES for the address 0x1f9c550: dwo performed a successful read of a
    valid address and returned a value that satisfied the comparison
    the enqueue CAS bracket becomes observable end to end
    the same run also observes whether 0x1f9cfb0 and 0x1f9ce04 are reached, which retires or
    confirms the residual T1-phase risk recorded in DG-T0-Repair section 5.2

gate does NOT open, but R1 PASSES
    every body reached its terminal gc, so the construct defect is repaired and continuation
    holds across all four T0 sites
    the diagnostic verdicts localise whether conjunct 2 or conjunct 3 is the false one
    3b does NOT close, because no successful dwo comparison was observed

R1 FAILS
    a construct or continuation problem exists outside the scope repaired here
    attribute, close, do not retry
```

## 5. Interpretation constraints carried forward

```text
C3_DG_T0E_EQ_PASS is NOT independent evidence.  After the repair it is the same expression as
the guard's conjunct 3, so given conjuncts 1 and 2 it is equivalent to "the gate opened".
It is a redundant witness for attribution and must not be counted as a second confirmation.

C3_DG_T0E_R9_FAIL / _PASS IS an independent measurement of conjunct 2.

"enqueuePos is always 221" must NOT be concluded.  221 is a single sample from Redesign-7 and
this gate produces booleans, not raw values.

"@r9 != 9" must NOT be concluded from a failure.  Redesign-9 measured FALSE at n=1;
Step5BB Retry-1 measured TRUE at n=1.

r9 is invariant across the four T0 sites within a cycle, proven from disassembly: no
instruction in 0x1f9c4e0..0x1f9c5f0 has r9 as a destination operand.  It is NOT invariant
between cycles.

T0 site counts were 116 of 116 in one run only.  The exit path at 0x1f9c5e0 can skip T0P, so
equal counts are measured, not guaranteed, and must be checked rather than assumed.
```

## 6. Out of scope for this run

```text
0x1f9ce04 repair          still malformed, deliberately
0x1f9cfb0 repair          still malformed, deliberately
ReaderSlot                not started
minReaderEpoch            not started
Case A / B / C / D        not started
T1 capture                not started
```

Whether `0x1f9cfb0` and `0x1f9ce04` are reached is an **observation** of this run, not a repair.
That is the entire point of choosing scope A'.

## 7. Preconditions, all verified before the request

| item | value | verified |
|---|---|---|
| target SHA-256 | `74E8C64C…F9142` | S-T0-12 |
| Static Validation | 17 of 17 PASS | Step5DI |
| guard byte-integrity | structure identical, 2 bytes per site | S-T0-07 |
| excluded `dd(` intact | `0x1f9ce04` 2, `0x1f9cfb0` 2 | S-T0-05, S-T0-06 |
| `C3_T0_ENTRY` intact | 1 occurrence | S-T0-08 |
| `gc` intact | 22 across the 4 sites | S-T0-09 |
| non-target lines | 37 byte-identical | S-T0-10 |
| `poi` not introduced | 7 = 7 | S-T0-15 |
| topology | 13 RVAs, identical order | S-T0-17 |
| harness | `E5C7AFB9…34C75` | S-T0-11 |
| `ConvoPeq.md` | `E5E74200…F3609` | S-T0-11 |
| `cdb.exe` | `5F54ABAF…FBEE67`, 10.0.29617.1000 | S-T0-11 |
| git `src/` `CMakeLists.txt` `build.bat` | 12, all pre-existing | S-T0-11 |

## 8. Binding precondition on the operand offsets

Carried from `DG-T0-Repair` section 7.2 and not recorded anywhere earlier in the lineage:

```text
The offsets in the guard are valid only while sizeof(DeletionEntry) == 0x30, which requires
CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS off.  Verified in build/CMakeCache.txt as OFF, and the
target harness digest matches.  With diagnostics on, sizeof becomes 0x38 and sequences moves
to +0x38000, invalidating every offset in every guard.  Any rebuild invalidates the baseline
and the offsets must be re-derived before use.
```

## 9. State at the time of the request

```text
IMPLEMENTATION      COMPLETE
STATIC VALIDATION   PASS 17/17
BASELINE            T0-Guard-Repair-Baseline-1 CONFIRMED
RUNTIME             NOT AUTHORIZED, requested here
CDB execution       0
Retry               none
source/test/CMake/build   untouched
T1                  untouched
```

Awaiting authorization. `cdb.exe` will not be launched until it is granted.
