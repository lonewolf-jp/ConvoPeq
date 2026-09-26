# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-CDB-Expression-Semantics-Result-Audit-1

## 1. Gate result

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-CDB-Expression-Semantics-Result-Audit-1
Mode                                  = read-only log audit
CDB execution                         = 0
ping execution                        = 0
AudioEngineHarness                    = 0
Retry-3                               = 0
Probe rerun                           = 0
Production capture                    = 0
Script modification                   = 0
Source/test/CMake/build/Dr.Memory/M1/M2 = 0
Prior probe authorization             = CONSUMED
AUDIT VERDICT                         = FAILURE PRIMITIVE IDENTIFIED / EXPRESSION SEMANTICS STILL UNPROVEN
```

This audit explains the consumed run only. It does not repair the script and does not authorize any execution.

## 2. Audited evidence

```text
Log    = doc/work113/P1-5-IR-P2_Step5BL_P3-5-FPM-C3-T1-Capture-PostRetry2_Benign-Probe-Execution-1_CDB.log
SHA-256 = BCFE1398DB6896C3C1FBB381C65C43EBD48ECAC5ACAE3D75F213F595B1C5DD2B
Bytes  = 20,447
CDB    = 10.0.29617.1000
Script = 5C12E1515CC2206430F34ADC7B8FE39D84A04BF4B33F2D7628970C395EC7699B
```

Only this log was used. No new debugger, target, or capture activity was performed.

## 3. Item A: runtime state of $t1..$t5

The log contains 29 `r $tN = ...` command echoes. Of these:

```text
literal numeric assignments accepted  = 24
module-qualified assignments rejected = 5
```

Accepted literal writes are `$t0..$t19 = 0`, then `$t0 = 9`, `$t7 = 0`, `$t8 = 1`, `$t10 = 4`.

Rejected writes:

```text
r $t1 = ping!+0x3c     -> Syntax error at 'ping!+0x3c'
r $t2 = ping!+0x4FD0   -> Syntax error at 'ping!+0x4FD0'
r $t3 = ping!+0x2600   -> Syntax error at 'ping!+0x2600'
r $t4 = ping!+0x5000   -> Syntax error at 'ping!+0x5000'
r $t5 = ping!+0x40     -> Syntax error at 'ping!+0x40'
```

Critically, the log contains **no register value dump** for any pseudo-register. There is no `r $t1` read, no `r $t1..$t19` display, and no `r` output line of the form `$tN = <value>`. Every line matching a pseudo-register value pattern is a command echo (`0:000> r $tN = ...`) or text embedded inside a breakpoint command body.

Therefore:

```text
$t1 runtime value after failure = NOT ESTABLISHED BY LOG
$t2 runtime value after failure = NOT ESTABLISHED BY LOG
$t3 runtime value after failure = NOT ESTABLISHED BY LOG
$t4 runtime value after failure = NOT ESTABLISHED BY LOG
$t5 runtime value after failure = NOT ESTABLISHED BY LOG
```

The value `0` is **not** asserted here. The only thing the log proves is that these five writes did not complete with their intended image-relative addresses. The actual post-failure contents are unknown and must not be inferred as either the previous value, zero, or another value.

## 4. Item B: classification of the `ping!+RVA` failure

Candidate explanations were separated and tested against the log.

### 4.1 MASM evaluator not active — RULED OUT

The log records:

```text
0:000> .expr /s masm
Current expression evaluator: MASM - Microsoft Assembler expressions
```

MASM was selected before the assignments. This candidate is excluded.

### 4.2 Pseudo-register write mechanism unsupported — RULED OUT

The same `r $tN = <value>` command form succeeded 24 times in the same session, including consecutive commands immediately before and after the failing five. Excluded.

### 4.3 Register-command expression parsing of module-qualified symbols — SUPPORTED BY LOG

The failure is confined to the right-hand side token reported by CDB. Each error line quotes the entire `ping!+0x...` token:

```text
Syntax error at 'ping!+0x3c'
```

Within the same session, `bp ping+0x1184`, `bp ping+0x387d`, and `bp ping+0x39d9` all resolved to concrete addresses, and `bl` showed retained breakpoints at `00007ff7'deea1184`, `00007ff7'deea387d`, and `00007ff7'deea39d9`. CDB also resolved the module-qualified address in the disassembly line `[ping+0x5308 ...]`.

So the log proves an asymmetry in this run: the module-plus-offset form was accepted by the `bp` address argument and by the disassembler, while the same form was rejected as the right-hand side of an `r` assignment. This is the classification the evidence supports. It does not establish a general CDB rule.

### 4.4 Module symbol absence as the cause — NOT ESTABLISHED

The log contains:

```text
0:000> .reload /f ping.exe
************* Symbol Loading Error Summary **************
ping                   The system cannot find the file specified
```

This shows no PDB was found for `ping.exe`. It is explicitly **not** treated here as the cause of the `ping!+RVA` rejection. Reasons:

```text
bp ping+0x1184 and similar resolved without any ping symbols
ping!+0x5308 resolved in the disassembly display
no log line links the symbol-load failure to the syntax errors
```

Symbol unavailability and the rejection of `ping!` in that specific position are therefore recorded as correlated but causally unlinked. Determining whether an alternative anchor form works requires a future authorized execution, not this audit.

### 4.5 Failure primitive

```text
failure primitive = the five module-qualified $t1..$t5 anchor assignments
                    were rejected at parse/evaluation time
```

This is the primitive that stopped the probe. It is upstream of every Family A-G expression.

## 5. Item C: `dwo` must not be blamed for the STOP

The log's first Family A condition produced:

```text
Memory access error at ') == 0x100) { .echo PASS_A_POS; ... }'
```

The error text is a fragment of the breakpoint body, and no `PASS_A_POS` or `FAIL_A_POS` marker was emitted. The first dereference in the body is `dwo(@$t1)`, and `$t1` was never established to its intended PE-header address.

Two distinct questions must stay separate:

```text
is dwo() syntactically accepted in a breakpoint body?   = NOT PROVEN
does dwo() evaluate correctly against a valid address? = NOT PROVEN
```

A `Memory access error` at the first dereference of an unset anchor is consistent with an invalid address, and says nothing about whether `dwo` itself is well-formed. Concluding that `dwo` caused the STOP would be an unsupported leap. It is explicitly rejected here.

## 6. Evidence classification table

| item | from this execution |
|---|---|
| breakpoint registration | PROVEN |
| breakpoint retention in `bl` | PROVEN |
| actual breakpoint hit | PROVEN (`ping+0x39D9`) |
| hit-time module-qualified address resolution | PROVEN (`ping+0x5308` in disassembly) |
| MASM evaluator selected | PROVEN |
| `ping!+RVA` assignment | FAILED (5 of 5 rejected) |
| `$t1..$t5` intended anchors | NOT ESTABLISHED |
| `$t1..$t5` post-failure values | NOT ESTABLISHED BY LOG |
| `dwo(@$t1)` with valid anchor | NOT TESTED |
| `dwo` syntax itself | NOT PROVEN |
| `poi` | NOT TESTED |
| low/high `&` / `>>` | NOT TESTED |
| complex ring arithmetic | NOT TESTED |
| MASM `and` in compound condition | NOT TESTED |
| `dd`/`dq` address expression | NOT TESTED |
| positive/negative branch execution | NOT TESTED |
| PASS marker execution | NOT TESTED |
| `PASS_TERMINATION_Q` | NOT TESTED |
| termination contract via probe logic | NOT TESTED (debugger `q` observed) |
| clean debugger exit | PROVEN (`quit:`) |
| process residue 0 | PROVEN |

`Bad register`, `Couldn't resolve`, and `Operand error` were not observed in this run. Their absence is recorded as not observed, not as proof of correct usage.

## 7. Confirmed causal chain

```text
five $t1..$t5 anchor assignments use module-qualified ping!+RVA
        |
        v
all five rejected with Syntax error
        |
        v
$t1..$t5 never hold their intended target anchors
        |
        v
actual hit at ping+0x39D9
        |
        v
first Family A condition dereferences @$t1
        |
        v
Memory access error before any marker
        |
        v
Families A-G, branch logic, and PASS_TERMINATION_Q never executed
        |
        v
script tail q executed -> quit: -> residue 0
```

## 8. What this audit does not change

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The benign probe added no evidence about ConvoPeq reader, epoch, or reclaim state.

## 9. Remaining unknowns and their resolution path

| open question | why it is open | required resolution |
|---|---|---|
| post-failure `$t1..$t5` values | log has no register dump | new authorized probe that dumps state before use |
| why `ping!+RVA` was rejected in `r` | log shows only correlation with symbol absence | new authorized probe using a different anchor form |
| `dwo` syntax and valid-address evaluation | never reached with a valid anchor | new authorized probe |
| `poi`, `&`/`>>`, ring arithmetic, MASM `and` | never reached | new authorized probe |
| `dd`/`dq` address expression | never reached | new authorized probe |
| branch and termination contract | never reached | new authorized probe |

All of these require a newly designed and newly authorized probe. None of them can be derived from the consumed authorization.

## 10. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-CDB-Expression-Semantics-Result-Audit-1
= CLOSED / FAILURE PRIMITIVE IDENTIFIED / EXPRESSION SEMANTICS UNPROVEN

failure primitive               = $t1..$t5 module-qualified anchor assignment rejected
dwo blamed for STOP             = NO / REJECTED
$t1..$t5 values asserted       = NO / NOT ESTABLISHED
dwo syntax                      = NOT PROVEN
poi / arithmetic / and / dd/dq = NOT TESTED
branch and termination contract = NOT TESTED
prior authorization             = CONSUMED / NOT REUSABLE
new probe design                = separate gate
new runtime authorization       = separate gate
Retry-3                         = FORBIDDEN
Production capture              = FORBIDDEN
AudioEngineHarness              = FORBIDDEN
Implementation                  = FORBIDDEN
```
