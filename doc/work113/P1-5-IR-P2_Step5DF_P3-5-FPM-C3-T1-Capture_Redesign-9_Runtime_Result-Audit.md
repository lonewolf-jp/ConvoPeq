# Redesign-9 Runtime Result Audit

```text
gate                 = P3-5-FPM-C3-T1-Capture, Redesign-9 single execution
script               = doc/work113/…_Redesign-9_Conjunct-Verdicts.cdb
script SHA-256       = D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14
authorized SHA       = D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14   match
log                  = doc/work113/…_Redesign-9_Runtime_CDB.log      37,789 B, 183 lines
invocation           = tmp\cdb.exe -cf <script> -logo <log> build\Release\AudioEngineHarness.exe --measurement=normal
execution_count      = 1        Retry-1 / Retry-2 / Retry-3 = NOT used
exit_code            = 0
elapsed              = 0.3 s
Runtime Result       = FAIL  (R1 not met)
T0 gate cause        = RESOLVED, and not for the reason this programme assumed
```

## 1. R1, continuation

| item | value | required |
|---|---|---|
| `C3_T1_REDESIGN2_SCRIPT_BEGIN` | 1 | 1 |
| `C3_T1_REDESIGN2_SCRIPT_END` | 1 | 1 |
| `ZwTerminateProcess` | **0** | 1 |
| `quit` | 1 | 1 |
| exit code | 0 | 0 |
| debugger errors | **1** | 0 |

**R1 = FAIL.** `ZwTerminateProcess` was not reached and one debugger error was emitted. Per the
standing stop condition, no downstream criterion is interpreted as a result.

### 1.1 Error census

| pattern | count |
|---|---|
| `Syntax error` | **1** |
| `Extra character` | 0 |
| `Illegal` / `Invalid` / `Undefined` / `^Error` / `Cannot` | 0 |
| `Evaluate expression:` | 0 |
| `Unable to read memory` | 0 |
| `Unable` (any) | 4 — 3 benign extension-DLL loads, 1 benign checksum |
| `WARNING` | 1 |

## 2. Failure site, verbatim

```text
[174] [EQ_PREPARE] agc tables allocated
[175] C3_DG_T0E_HIT
[176] C3_DG_T0E_R9_FAIL
[177] Syntax error at '(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo C3_DG_T0E_EQ_FAIL } ;
      .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1;
      ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }'
[178] AudioEngineHarness+0x1f9c550:
[179] 00007ff7`7f1ec550 48895c2408      mov     qword ptr [rsp+8],rbx
[180] 0:000> .echo C3_T1_REDESIGN2_SCRIPT_END
```

| # | command | observed |
|---|---|---|
| 1 | `.echo C3_DG_T0E_HIT` | emitted, line 175 |
| 2 | `.if (@r9 == 9) { .echo …R9_PASS }` `;` `.else { .echo …R9_FAIL }` | executed, `R9_FAIL` emitted, line 176 |
| 3 | `.if (dd(@rcx+0x34000) == 4) { .echo …EQ_PASS }` … | **Syntax error**, line 177 |
| 4 | frozen guard | never reached |
| 5 | terminal `gc` | never reached |

The reported span begins at the `(` that immediately follows `dd`. The parser consumed `.if (dd`
and rejected the call parenthesis.

## 3. The cause: `dd` is a display command, not an expression function

Microsoft documents the `.if` *Condition* as follows:

> *Condition* must be an expression, **not a debugger command**. It will be evaluated by the default
> expression evaluator (MASM or C++).

`dd` is the display-memory command. It is not callable inside an expression. The documented MASM
unary operators for reading memory are `by`, `wo`, `dwo`, `qwo` and `poi`; `dwo` is "Double-word from
the specified address", `poi` is "Pointer-sized data from the specified address". `dd` is not among
them.

This is consistent with the measurement in both directions within a single body:

```text
.if (@r9 == 9)                 -> executed, R9_FAIL emitted      register operand parses
.if (dd(@rcx+0x34000) == 4)    -> Syntax error at '(' after 'dd'  display command does not parse
```

It also explains a fact that was recorded at the time and never reconciled: `? dd(@rcx+0x34000)` in
Redesign-7 **did** evaluate, returning 221. `?` uses a different evaluation path from a `.if`
condition, which is why the same token is accepted there and rejected here.

## 4. The regression, and the evidence that closes it

The malformed conjunct is **not** an artifact of this gate. It is inherited, and its history is
directly observable in the archived logs.

### 4.1 The gate DID open once, with a different memory-read form

`Step5BB-Retry-1` is the only run in the entire corpus that emitted `C3_T0_ENTRY`. Its own `bp`
echo, recovered from its log, shows the T0E guard at RVA `0x1F9C550` as:

```text
.if (@$t5 == 0) { .if (@$t0 <= 1) { .if (@r9 == 9) { .if (poi(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; …
```

Same binary, same site, same 64-bit harness, `--fpm-m0`. The emitted register dump shows
`r9=0000000000000009`, so conjunct 2 held, and the gate opened through conjunct 3 using **`poi`**.
Module base `00007ff7`57af0000` with `rip=00007ff759a8c550` gives RVA `0x1F9C550`, confirming the site.

### 4.2 The next run failed on the guard's own conjunct 3

`Step5BG-Retry-2` contains the same construct failing, this time inside the **guard itself**, not a
diagnostic:

```text
Syntax error at '(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } .else { gc } '
```

The fragment is the guard's own innermost conjunct, identifiable by `.echo C3_T0_ENTRY` inside it.
This was measured at Retry-2 and not diagnosed at the time. It is not an inference about the guard;
it is a recorded execution of the guard's own malformed conjunct failing.

### 4.3 The lineage carried it forward under the frozen-guard rule

| script | memory reads inside `.if` conditions | `C3_T0_ENTRY` |
|---|---|---|
| Step5BB Retry-1 | `poi(` ×2 | **1, gate opened** |
| Step5BB Redesign-2 onward | `dd(` ×10 | 0 |
| Step5DD Redesign-9 | `dd(` ×10, plus 1 restated by this gate's diagnostic | 0 |

```text
REGRESSION POINT   Retry-1 -> Retry-2
CAUSE             the conjunct-3 memory read was changed from  poi(addr)  to  dd(addr)
EFFECT            the innermost conjunct of the T0 gate became a parse error, permanently
```

## 5. Why it stayed latent for the whole lineage

The malformed conjunct is the **innermost** of three. Every evaluation reaches conjunct 1
(`@$t0 == 0`, true, established at Redesign-6) and conjunct 2 (`@r9 == 9`), then stops. The
malformed conjunct is only parsed if conjunct 2 holds. It never held in any run, so the parse error
was never re-triggered after Retry-2, and the gate presented as a silent non-event rather than as an
error.

That is why eight redesigns were spent on operand values. The gate could not have opened on any
operand, because the condition guarding entry was not a well-formed expression.

## 6. Full extent of the defect in the executed script

Paren-aware classification of every `(dd(` occurrence in Redesign-9:

| location | RVA | context | count |
|---|---|---|---|
| line 25 | `0x1f9c550` | inside `.if` conditions, diagnostic + guard | 2 |
| line 26 | `0x1f9c57d` | inside `.if` conditions | 2 |
| line 27 | `0x1f9c59a` | inside `.if` conditions | 2 |
| line 28 | `0x1f9c5da` | inside `.if` conditions | 2 |
| line 36 | `0x1f9ce04` | inside `.if` conditions | 2 |
| line 30 | `0x1f9cfb0` | **outside** `.if`, inside a display-command address expression | 2 |

```text
TOTAL (dd( in file            = 12
  inside .if conditions       = 10   at 5 sites
  outside .if conditions      =  2   both at 0x1f9cfb0
```

The two outside cases are the same construct class, used to obtain a value inside an address
computation:

```text
dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
dq (@$t2+0x00+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
```

They are flagged as suspect on the same grammar grounds but were **not measured** in this run, and
are recorded as unverified.

## 7. R2 to R6

Not assessed. The stop condition for R1 failure forbids interpreting downstream criteria, and R1
failed. The raw counts are recorded without interpretation:

| marker | count |
|---|---|
| `C3_DG_T0E_HIT` | 1 |
| `C3_DG_T0E_R9_PASS` | 0 |
| `C3_DG_T0E_R9_FAIL` | 1 |
| `C3_DG_T0E_EQ_PASS` | 0 |
| `C3_DG_T0E_EQ_FAIL` | 0 |
| `C3_DG_T0P_HIT` / `T0Q` / `T0S` | 0 / 0 / 0 |
| `C3_T0_ENTRY` | 0 |
| `C3_T0_SEQUENCE_PUBLISHED` | 0 |
| `C3_CR1_STOP_*` | 0 |
| T1 capture markers | 0 |

The four-way discrimination, per-hit pairing and T0P/T0Q/T0S interleaving are **not derived**. One
T0E hit occurred, and it aborted at command 3, so no correspondence evidence exists.

## 8. What is now established, and what is not

```text
PROVEN   '.if (dd(...) == n)' is a parse error on this build.  The error span begins at the
         call parenthesis following 'dd'.  Independently corroborated by the Microsoft
         definition of a .if Condition.

PROVEN   the malformed conjunct is pre-existing and frozen, not introduced by this gate.
         10 occurrences inside .if conditions at 5 sites, all in the Redesign-4-or-earlier region.

PROVEN   the guard's own conjunct 3 has failed exactly this way before, at Step5BG Retry-2.

PROVEN   the same site opened successfully at Step5BB Retry-1 using poi(@rcx+0x34000),
         with r9 = 9, on this same 64-bit binary.

PROVEN   Option C's own design goal was met for conjunct 2: the verdict R9_FAIL was emitted,
         so '.echo' plus a two-branch '.if'/'.else' carrying '.echo' is tolerated at
         payload positions 1 and 2 at 0x1f9c550.

NOT PROVEN  '@r9 != 9' generally.  One hit, one run.  Retry-1 observed r9 = 9 at the same site.
            r9 is invariant across the four T0 sites within a cycle (no instruction in
            0x1f9c4e0..0x1f9c5f0 writes r9), but is not invariant between cycles.

NOT PROVEN  anything about enqueuePos.  Conjunct 3 was never evaluated in this run.

NOT PROVEN  that poi() is a correct substitute at every one of the 5 sites.  It is proven at
            0x1f9c550 and 0x1f9c57d only, and only for the 0x34000 offset.

NOT PROVEN  that the 2 out-of-condition occurrences at 0x1f9cfb0 are defective.  Same construct
            class, not exercised.

NOT KNOWN  the mechanism by which a completed '?' suppresses the following command.  The
            corrected constraint from the Step5DB audit stands as a bound, not a cause.
```

## 9. Answer to the question this programme was chartered to answer

The question was: which conjunct of the T0 entry guard is false?

```text
conjunct 1  .if (@$t0 == 0)              TRUE, established at Redesign-6, sole writer fired 0 times
conjunct 2  .if (@r9 == 9)               FALSE at the single hit measured here; n = 1, not generalisable
conjunct 3  .if (dd(@rcx+0x34000) == 4)  NOT A CONDITION.  Malformed. Never evaluable.
```

The gate cause is conjunct 3, and it is not a wrong constant. It is a malformed condition. Because
conjunct 3 could never be evaluated, no operand value could ever have opened the gate, and the
question of which operand was wrong was unaskable for the whole Redesign-2 through Redesign-9 span.

Conjunct 2 is separately false at the one hit observed, and that remains open. It is secondary: even
if it held, conjunct 3 would still have blocked entry.

## 10. The repair candidate, for the next design gate, not applied here

```text
proven working at this site   poi(@rcx+0x34000) == 4      Step5BB Retry-1
current, malformed            dd(@rcx+0x34000)  == 4
documented, unproven here    dwo(@rcx+0x34000) == 4      exact dword width
```

`dwo` is the exact-width operator for a 32-bit field. `poi` is pointer-sized, 8 bytes on x64, and
compares the full 64-bit result; it worked at Retry-1 because the dword above `enqueuePos` was zero.
Choosing between them is a design decision for the next gate, together with the width semantics. No
change is made in this gate.

The `bp` bodies at `0x1f9c550`, `0x1f9c57d`, `0x1f9c59a`, `0x1f9c5da` and `0x1f9ce04`, and the two
address expressions at `0x1f9cfb0`, are all in scope for that repair. Changing any of them voids the
"byte-identical to frozen Redesign-4" property that every static check in this lineage has asserted,
so the next gate must decide explicitly whether a corrected guard becomes a new frozen baseline.

## 11. Tooling defects encountered in this gate

Recorded because each one produced a wrong intermediate statement that had to be withdrawn.

| # | defect | consequence | correction |
|---|---|---|---|
| 1 | Recorded artifact path as `…Capture_Redesign-4_Corrected.cdb`; the real name uses a hyphen, `…Capture-Redesign-4_Corrected.cdb` | spurious `FileNotFoundError`, momentarily suggesting the frozen artifact was missing | confirmed by directory listing and SHA prefix `a935369d711535c9`; the file was never missing |
| 2 | A PowerShell `&` at the start of a line was parsed as the background-job operator | the `cd` was swallowed into a job, the working directory was wrong, and the same `FileNotFoundError` appeared a second time with a different cause | used the shell tool's `workdir` parameter and absolute paths in the helper |
| 3 | Read `Step5BK` with a CRLF-only split | the file uses LF, so the split returned one element and the script printed nothing, briefly suggesting it had no `bp` bodies | re-read with a robust split; it has 41 lines and 3 `bp` bodies |
| 4 | First `.if`-condition scan used a regex that required a second `(` after `.if (` | reported 5 sites and omitted `0x1f9cfb0`, which carries `dd(` in an address expression | replaced with a paren-aware classifier; the correct split is 10 inside conditions at 5 sites, plus 2 outside at 1 further site |
| 5 | A `conds()` census reported 11 occurrences inside `.if` conditions | the paren-aware classifier reports 10 | the paren-aware figure is the one recorded; a convenience extractor disagreed with the paren-aware scan and the convenience extractor was wrong |
| 6 | Initial hypothesis from the payload failure was that the guard's `dd(` was a *latent* defect | would have been an inference only | superseded: the guard's own conjunct 3 is recorded failing at Step5BG Retry-2, so the claim is measured, not inferred |

## 12. State

```text
T0 gate cause            = RESOLVED.  Conjunct 3 is malformed: dd() used as an expression
                           function inside a .if condition.  Regression Retry-1 -> Retry-2.
conjunct 1 @$t0 == 0     = TRUE, frozen
conjunct 2 @r9 == 9      = FALSE at n=1, open
conjunct 3 enqueuePos    = NEVER EVALUABLE
enqueuePos raw value     = NOT OBTAINED.  Only 221 at one hit, Redesign-7, n=1
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D  = UNRESOLVED
A1 separator = RUNTIME PROVEN      A2 = installed, accepted, execution count 0
CDB execution this gate = 1
harness / build / source / test / CMake = untouched
IMPLEMENTATION = FORBIDDEN
RUNTIME AUTHORIZATION for any further run = NOT GRANTED
```

## 13. Stop

R1 failed, the failure is attributed to a command, the cause is identified, and the run is closed.
No retry was performed. No `.cdb` was edited. No guard, source, test, CMake or build input was
touched. The repair is named but not applied, and applying it requires a new design gate, a new
implementation authorization, and a new static validation.
