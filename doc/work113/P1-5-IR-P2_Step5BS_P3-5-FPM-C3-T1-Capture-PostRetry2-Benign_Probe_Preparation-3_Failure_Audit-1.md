# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Failure-Audit-1

## 1. Gate result

```text
Gate                       = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Failure-Audit-1
Mode                       = read-only failure analysis
CDB execution              = 0
ping execution             = 0
new CDB script created     = 0
script modification        = 0
@$pc corrected             = NO (not performed, not permitted here)
Preparation-4              = 0 (not started)
AudioEngineHarness         = 0
Retry-3                    = 0
production capture         = 0
M1/M2/build/Dr.Memory      = 0
rerun                      = 0
PRIMARY FAILURE            = Bad register error on @$pc in CDB 10.0.29617.1000
SECONDARY SCRIPT DEFECT   = ';' is not a comment introducer in a CDB -cf command file
```

This gate analyses one consumed execution. It creates nothing, repairs nothing, and re-runs nothing.

## 2. Frozen evidence

```text
log      SHA-256 = 4E2BB488253B7FAE83579576DBAE9416777EC2BC6326D4113DDB278488E4BBB4
script   SHA-256 = 671F9D453FF814DD2AE01217A6CAE7F69E0D033A7A0AA70BD42D21E797D63771 (unchanged)
ConvoPeq.md SHA   = E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609 (unchanged)
CDB                = 10.0.29617.1000
process residue    = 0
```

All three identities re-verified identical at audit time. The production source authority was not re-read at source-line granularity, and no new source-level claim is made in this document.

## 3. Primary failure primitive

```text
Candidate A  -> ping+offset -> $t1..$t5 established correctly -> PROVEN
Candidate B  -> @$pc        -> Bad register error            -> FAILED
                              -> $t11 never established
                              -> T1..T5 never evaluated
                              -> Preparation-3 FAILED
```

Exact evidence, log line 132:

```text
Bad register error at '@$pc-0x39d9; r $t12=@$t11+0x3c; ... '
```

The failure occurs at the very first state assignment of the breakpoint body, immediately after `r $t16=1` and `r $t17=@$t17+1`. Nothing after that point executed, which is why `C3BP3_BASE_DERIVED` is absent (count 0) and no `PASS_T1..T5`, `PASS_SENTINEL_CLEARED`, or `PASS_TERMINATION_Q` was produced.

The zero `FAIL_*` count does not indicate success. The body aborted before reaching any conditional, so no failure branch could execute.

Corroboration from Microsoft Learn, *MASM Numbers and Operators* and *Pseudo-Register Syntax*:

```text
$ip   "the instruction pointer register; x64: same as rip"
.     "current instruction pointer ... same meaning as the $ip pseudo-register"
@rip  architecture register
$pc   NOT PRESENT in the documented pseudo-register list
```

This matches the R1 risk declared in the authorization review, which predicted exactly this outcome and correctly stated the impact would be fail-closed.

## 4. Scope of impact

```text
item                              verdict
r $tN = ping+offset  RHS          PROVEN
pseudo-register readback          PROVEN
sentinel cleared                  PROVEN
breakpoint registration           PROVEN
actual breakpoint hit             PROVEN
@$pc                              FAILED
Candidate B                       FAILED
A == B                            NOT EVALUATED
dwo                               NOT PROVEN
poi                               NOT PROVEN
& / >>                            NOT PROVEN
ring arithmetic                   NOT PROVEN
and                               NOT PROVEN
dd / dq                           NOT PROVEN
```

The confirmed statement is narrow: in CDB 10.0.29617.1000, `@$pc` is rejected as a register.

It is expressly **not** inferred that:

```text
dwo is unusable
poi is unusable
MASM arithmetic is unusable
CDB expressions are broadly unusable
```

No such inference follows from one rejected token, and none is made.

## 5. Secondary script defect, evaluated separately

This must not be merged into the `@$pc` failure.

Observed error classes, all traceable to `;` comment lines:

```text
Couldn't resolve error               = 5
Syntax error (from comment text)     = 4
pass count must be preceeded by ...  = 6
total comment-origin errors          = 15
```

Every error line pairs with a preceding `0:000> ; ...` command echo. The reported tokens reveal the mechanism:

```text
source line : ; Anchor-establishment-only minimal probe. DESIGN ARTIFACT - NOT EXECUTED.
reported    : 'nchor-establishment-only minimal probe. DESIGN ARTIFACT - NOT EXECUTED.'
source line : ; Scope: establish $t1..$t5 as ping.exe image-relative anchors ...
reported    : 'cope: establish $t1..$t5 as ping.exe image-relative anchors ...'
source line : ; Candidate B - derive the image base from @pc at hit time ...
reported    : 'andidate B - derive the image base from @pc at hit time ...'
```

In every case the reported token is the original comment text with its **leading two characters** removed. The first word begins with an uppercase letter, so CDB attempted to resolve it as a symbol. `; ` is therefore consumed as if the text were an expression starting at the second character.

```text
CONCLUSION = ';' does not introduce a comment in a CDB -cf command file
             in this build. No comment line is silently ignored; every one
             produces an error.
```

Microsoft Learn documents the `bp` *CommandString* rule that semicolons separate commands, and states that semicolons inside second-level quotation marks are treated as literal text. This is consistent with `;` having no comment role in the command file.

### Reproduction sites

| log line | source line (abbreviated) | error class |
|---|---|---|
| 49 | `; ====...` | Syntax error |
| 51 | `; P3-5-FPM-C3-...` | pass count error |
| 53 | `; Anchor-establishment-only ...` | Couldn't resolve |
| 55 | `; Scope: establish ...` | Couldn't resolve |
| 57 | `; Explicitly EXCLUDED: ...` | Couldn't resolve |
| 59 | `; No memory is read ...` | Syntax error |
| 76 | `; Phase 1: distinct sentinels ...` | pass count error |
| 91 | `; Phase 2: Candidate A - ...` | pass count error |
| 93 | `; This is the documented ...` | pass count error |
| 95 | `; It is the exact form that failed ...` | Syntax error |
| 106 | `; Phase 3: immediate unconditional ...` | pass count error |
| 108 | `; "$t1..$t5 post-failure value ...` | Syntax error |
| 116 | `; Phase 4: single benign breakpoint ...` | pass count error |
| 118 | `; Candidate B - derive the image base ...` | Couldn't resolve |
| 120 | `; arithmetically, with no module-qualified ...` | Couldn't resolve |

### Execution continuity

The comment errors did not stop the script. Functional milestones in log order:

```text
44   C3BP3_SCRIPT_BEGIN
49   first comment error
99   C3BP3_ASSIGN_DONE
110  C3BP3_READBACK_DONE
121  breakpoint registered
132  Bad register error (@$pc)
133  actual breakpoint hit
137  quit:
```

Every functional stage between the first comment error and the fatal `@$pc` error completed. Confirmed non-causal for the STOP:

```text
comment defect -> NOT the cause of Preparation-3 failure
```

It is nonetheless a real defect and a candidate correction target for any future probe, because it pollutes every log with 15 spurious error lines and could mask a genuine error of a similar shape.

## 6. Candidate A result carried forward

Candidate A is **PROVEN** and may be carried into the next gate unchanged:

```text
observed image base = 0x7FF71CD00000

$t1 = 0x7FF71CD0003C   = base + 0x3C
$t2 = 0x7FF71CD04FD0   = base + 0x4FD0
$t3 = 0x7FF71CD02600   = base + 0x2600
$t4 = 0x7FF71CD05000   = base + 0x5000
$t5 = 0x7FF71CD00040   = base + 0x40
sentinel cleared = YES
```

The `r $tN = ping+offset` form works as an assignment right-hand side. This is the durable positive result of the run and must not be discarded because the gate as a whole failed.

## 7. Candidate B replacement options

No script is written here. These are design inputs only.

### Option B1 — breakpoint address pseudo-register (recommended)

```text
r $t11=@$bp0-0x39d9
```

Microsoft Learn, *Breakpoint Syntax*:

> If you want to refer to a breakpoint address in an expression, you can use a pseudo-register with the **$bp**Number syntax, where Number is the breakpoint ID.

```text
advantages = resolves the breakpoint's own address; no module symbol; no architecture
             register name; already exercised in this session as breakpoint id 0
risk       = $bp0 is NOT yet runtime-proven in this build; must be validated, not assumed
```

### Option B2 — instruction pointer pseudo-register

```text
r $t11=@$ip-0x39d9
```

Documented: `$ip` is the instruction pointer register, same as `rip` on x64.

```text
advantages = directly documented as the instruction pointer
risk       = not runtime-proven here; the bare period '.' is documented as equivalent
             but is explicitly disallowed as the first parameter of r
```

### Option B3 — architecture register

```text
r $t11=@rip-0x39d9
```

Documented as the x64 instruction pointer. Lowerest preference: it hard-codes an architecture, and the MASM guidance notes that omitting `@` on uncommon registers makes the debugger fall back through hex, symbol, then register parsing.

### Option B4 — module name plus offset (already proven)

```text
r $t1 = ping+0x3c
```

```text
status = PROVEN in this run
role   = if B1/B2/B3 all fail, the A==B cross-check may be abandoned rather than
         forced, because Candidate A is already established independently
```

### Design conclusion

The A==B cross-check existed to independently confirm anchor construction. Candidate A is now proven directly against the observed image base, so the cross-check is a redundancy rather than the only evidence. Any of B1, B2, or B3 may be validated in a replacement preparation; failing all three would not invalidate Candidate A.

## 8. Comment defect correction options

For any future probe:

```text
remove all comment text from the command file, or
use only .echo lines as in-band annotations, or
verify an alternative comment introducer in this build before relying on it
```

No comment line may be assumed safe in this build.

## 9. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 10. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Preparation-3-Failure-Audit-1
= CLOSED / PRIMARY AND SECONDARY CAUSES SEPARATED

PRIMARY  = @$pc -> Bad register error in CDB 10.0.29617.1000
SECONDARY= ';' is not a comment introducer in -cf files (15 error lines, non-fatal)
Candidate A        = PROVEN (carried forward)
Candidate B        = FAILED
A == B             = NOT EVALUATED
dwo / poi / & / >> / ring / and / dd / dq = NOT PROVEN
comment defect caused the STOP = NO

next gate  = Candidate-B replacement preparation (NOT Preparation-4)
then       = static validation
then       = runtime authorization
then       = one benign execution
then       = result audit

Preparation-4      = NOT STARTED / BLOCKED
Retry-3            = FORBIDDEN
production C3      = FORBIDDEN
AudioEngineHarness = FORBIDDEN
M1 / M2 / build / Dr.Memory = FORBIDDEN
implementation     = FORBIDDEN
rerun              = FORBIDDEN
```
