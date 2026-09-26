# CDB Repair Implementation Authorization

```text
gate                = CDB Repair Implementation Authorization
supersedes          = the T0E guard of Redesign-4 (retired as a measurement baseline)
adjudicated by      = DG-T0-Repair (Step5DG), scope A', operator dwo
mode                = AUTHORIZATION ONLY.  No .cdb created.  No CDB executed.
parent artifact     = P1-5-IR-P2_Step5DD_…_Redesign-9_Conjunct-Verdicts.cdb
parent SHA-256      = D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14
parent bytes        = 14,887      lines = 41
execution this gate = 0
status              = GRANTED, scoped.  Implementation and Static Validation not yet started.
```

## 1. Authorized conditions

| item | authorized content |
|---|---|
| purpose | repair the malformed `dd()` in the T0 CAS bracket |
| target | 4 sites, `0x1f9c550` / `0x1f9c57d` / `0x1f9c59a` / `0x1f9c5da` |
| operator | `dd` → `dwo` |
| `poi` | **FORBIDDEN.** Not to be introduced anywhere |
| `0x1f9ce04` | **FORBIDDEN this round.** Unchanged |
| `0x1f9cfb0` | **FORBIDDEN this round.** Unchanged |
| guard structure | conjuncts, `.echo`, `.else`, `gc`, nesting, order all **FORBIDDEN to change** |
| source | **FORBIDDEN** |
| test | **FORBIDDEN** |
| CMake | **FORBIDDEN** |
| build | **FORBIDDEN** |
| runtime execution | **FORBIDDEN at this stage.** Requested separately after Static Validation |
| baseline | Redesign-4 retained on disk unmodified as history; the corrected guard becomes `T0-Guard-Repair-Baseline-1` |

## 2. The repair scope, stated exactly

Not "replace `dd` with `dwo`". The scope is enumerated occurrence by occurrence, because a
file-wide substitution is precisely what is forbidden.

```text
file total (dd(                        12
  inside .if conditions                10
    at the 4 target sites               8   <- THE REPAIR SCOPE
    at 0x1f9ce04                        2   <- EXCLUDED, stays dd(
  outside .if conditions                2
    at 0x1f9cfb0 (address expressions)  2   <- EXCLUDED, stays dd(
```

| line | RVA | site | `dd(` in `.if` | disposition |
|---|---|---|---|---|
| 25 | `0x1f9c550` | T0E | 2 | **REPAIR** |
| 26 | `0x1f9c57d` | T0P | 2 | **REPAIR** |
| 27 | `0x1f9c59a` | T0Q | 2 | **REPAIR** |
| 28 | `0x1f9c5da` | T0S | 2 | **REPAIR** |
| 30 | `0x1f9cfb0` | T1 candidate | 0 in `.if`, 2 in address expressions | **UNCHANGED** |
| 36 | `0x1f9ce04` | T1 reclaim return | 2 | **UNCHANGED** |

### 2.1 The three distinct string forms, and only these

```text
(dd(@rcx+0x34000)   ->  (dwo(@rcx+0x34000)     2 occurrences, both at 0x1f9c550
(dd(@r11+0x34000)   ->  (dwo(@r11+0x34000)     2 occurrences, 0x1f9c57d and 0x1f9c59a,
                                                  plus 2 more at 0x1f9c5da
(dd(@r11+0x30010)   ->  (dwo(@r11+0x30010)     likewise
```

Concretely, the only permitted textual edit is the three-character sequence `dd(` becoming `dwo(`,
applied at 8 positions, and at no other position in the file.

```text
byte delta      8 occurrences x 1 byte    =  +8
predicted size  14,887 + 8                 =  14,895
```

> **CORRECTION, issued at the Step5DI implementation gate.** The two lines above originally read
> `x 2 bytes` and `14,903`. That arithmetic was wrong. `dd(` is 3 characters and `dwo(` is 4, so
> each occurrence adds **one** byte, not two. The correct prediction is **+8, total 14,895 B**,
> which is what the product actually is. The size assertion in the generation script used 1 byte
> and was right; the prose prediction here and the script's print statement were wrong. S-T0-16 is
> evaluated against 14,895.

### 2.2 Why `0x1f9cfb0` is not touched, restated so it is not re-litigated

```text
dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
dq (@$t2+0x00+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
```

These are not `.if` conditions. They are display commands whose **address argument** is computed
from an inline read. The documented rule that a condition must be an expression rather than a
command does not directly govern an address argument, so the two situations are governed by different
grammar rules and are not interchangeable. Whether `dwo` is accepted as a value-producing call
inside a display-command address is a separate, still-unverified question. Repairing it here would
mix the T0 and T1 phases and destroy the attribution this scope exists to protect.

## 3. One decision the instruction left implicit, fixed here and open to veto

The instruction describes the repair as acting on the **guard**. The parent artifact, Redesign-9,
carries a diagnostic payload at `0x1f9c550` that restates the same predicate, so that line contains
**two** `dd(` occurrences, not one:

```text
payload  .echo C3_DG_T0E_HIT ; .if (@r9 == 9) { … } ; .else { … } ;
         .if (dd(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo …EQ_FAIL }
guard    .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; …
```

Leaving the payload's occurrence unrepaired is not available. It is the third command of the body,
so it aborts the body at the first hit exactly as it did in Redesign-9, and the run fails before the
guard is ever entered.

```text
DECISION   repair all 8, payload included.  Parent stays Redesign-9.
REASON     it is the smaller delta.  Dropping the payload would delete 194 bytes of proven
           diagnostic and would not be "dd( -> dwo() and nothing else".  Keeping it preserves
           S-T0-01's property that the only change at 0x1f9c550 is the operator token.
VETO       if the owner intends a guard-only repair, the parent must instead be a payload-free
           artifact and the delta becomes "add a diagnostic back", which is a different
           authorization.  Say so before implementation.
```

### 3.1 An analytical consequence that must not be over-read later

After the repair, the payload's conjunct-3 verdict and the guard's conjunct 3 are the **same
expression**:

```text
.if (dwo(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS }   payload
.if (dwo(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; … }     guard
```

So `C3_DG_T0E_EQ_PASS` is no longer independent evidence of anything. Given conjuncts 1 and 2, it is
equivalent to "the gate opened". It is a redundant witness, useful for attribution, and must not be
counted as a second confirmation.

The r9 verdict is unaffected and remains an independent measurement of conjunct 2.

## 4. CDB generation constraints

```text
FORBIDDEN   a file-wide 'dd(' -> 'dwo(' substitution.  It would silently repair 0x1f9ce04 and
            0x1f9cfb0, which are excluded by this authorization.
FORBIDDEN   any substitution not inside a '.if' condition at the 4 target sites.
FORBIDDEN   touching 0x1f9ce04 or 0x1f9cfb0 in any way, including whitespace.
FORBIDDEN   introducing 'poi'.  It is excluded on the record: poi reads [0x34000,0x34008) and its
            upper 32 bits are 60 bytes of tail padding whose value no code writes and no language
            rule fixes.  A baseline must not rest on an indeterminate value.
FORBIDDEN   changing any conjunct, constant, operand base, offset, register, or pseudo-register.
FORBIDDEN   changing nesting, ordering, separator style, or adding or removing any command.
FORBIDDEN   changing the '.echo' markers, including C3_T0_ENTRY, which occurs once in the 4 sites.
FORBIDDEN   changing any 'gc'.  Continuation depends on all of them.
REQUIRED    the edit be expressed as a site-scoped, condition-scoped, counted replacement, and
            that the count be asserted equal to 8 before the file is written.
```

### 4.1 The structure that must survive, as a reversibility anchor

These counts are taken from the parent and must be identical in the product.

| RVA | `.echo` | `gc` | `.if` | `.else` | body bytes |
|---|---|---|---|---|---|
| `0x1f9c550` | 6 | 4 | 5 | 5 | 363 |
| `0x1f9c57d` | 2 | 5 | 4 | 4 | 251 |
| `0x1f9c59a` | 2 | 5 | 4 | 4 | 252 |
| `0x1f9c5da` | 4 | 8 | 7 | 7 | 2229 |

```text
REVERSIBILITY TEST   substituting 'dwo(' back to 'dd(' at exactly the 8 repaired positions must
                     reproduce the parent byte for byte.  This is a stronger check than any
                     per-line comparison, because it fails if anything else moved.
```

## 5. Static Validation, mandatory items

Read-only. No debugger is launched.

| ID | check |
|---|---|
| S-T0-01 | `0x1f9c550`: the only change is `dd(` → `dwo(`. Nothing else differs |
| S-T0-02 | `0x1f9c57d`: the only change is `dd(` → `dwo(`. Nothing else differs |
| S-T0-03 | `0x1f9c59a`: the only change is `dd(` → `dwo(`. Nothing else differs |
| S-T0-04 | `0x1f9c5da`: the only change is `dd(` → `dwo(`. Nothing else differs |
| S-T0-05 | `0x1f9ce04` `dd(` unchanged, count still 2 |
| S-T0-06 | `0x1f9cfb0` `dd(` unchanged, count still 2, still outside `.if` |
| S-T0-07 | guard text integrity at T0E / T0P / T0Q / T0S, per the section 4.1 anchor |
| S-T0-08 | `.echo C3_T0_ENTRY` unchanged, 1 occurrence in the 4 target sites |
| S-T0-09 | terminal `gc` unchanged at every site |
| S-T0-10 | every line outside the 4 target spans byte-identical to the parent |
| S-T0-11 | source, test, CMake, build inputs unchanged, by SHA comparison and git status |
| S-T0-12 | new `.cdb` SHA-256 recorded |
| S-T0-13 | **reversibility.** `dwo(` → `dd(` at exactly 8 positions reproduces the parent byte for byte |
| S-T0-14 | `dwo(` present inside a `.if` condition at all 4 target sites, 2 each |
| S-T0-15 | `poi(` count in the product equals the parent count, i.e. `poi` was not introduced |
| S-T0-16 | file size exactly **14,895 B**; ASCII, no BOM, 41 CRLF, trailing CRLF |
| S-T0-17 | breakpoint topology unchanged, 13 RVAs in identical order |

S-T0-13 and S-T0-14 are additions to the required twelve. S-T0-13 is the load-bearing one: it
converts "the diff looks right" into "the diff is provably only this", and it is the check that
would have caught a file-wide substitution.

### 5.1 What Static Validation must NOT assert

```text
MUST NOT claim  that dwo returns the correct value.
MUST NOT claim  that dwo is runtime-valid.
MUST NOT claim  that the T0 gate will open.
MUST NOT claim  anything about enqueuePos beyond the source-confirmed declaration and the
               width analysis.
```

`DG-T0-Repair-3b` is OPEN and stays OPEN through Static Validation. `dwo` is proven only to be a
**syntactically accepted** `.if` condition operand, on the evidence that a `Memory access error` is
an evaluation-stage failure whereas `dd` produced a parse-stage `Syntax error`. Whether `dwo`
successfully reads a valid address is unproven, and this stage is not permitted to close it.

## 6. The new frozen baseline

```text
NAME              T0-Guard-Repair-Baseline-1
SUPERSEDES        the T0E guard of Redesign-4
OLD BASELINE      RETIRED as a measurement baseline.  Retained byte-identical on disk as history.
                  Not deleted, not edited.
                  SHA A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C, 14,624 B
```

The three T0E conjuncts, as they will be fixed:

```text
conjunct 1   .if (@$t0 == 0)
conjunct 2   .if (@r9 == 9)
conjunct 3   .if (dwo(@rcx+0x34000) == 4)
```

Operand bases are per site and must be recorded per site, because conflating them is a recorded
defect of this lineage. At `0x1f9c550` only `@rcx` is a proven queue base; `@r10`, `@r11` and `@rbx`
are still caller values there. At `0x1f9c57d`, `0x1f9c59a` and `0x1f9c5da` the base is `@r11`.

Offsets, now source-confirmed as well as measured:

```text
+0x30000  sequences      +0x34000  enqueuePos   +0x30010  sequences[4]   +0x34040  dequeuePos
```

```text
BINDING PRECONDITION, not recorded anywhere in the current lineage:
the offsets above hold only while sizeof(DeletionEntry) == 0x30, which requires
CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS off.  Verified in build/CMakeCache.txt as OFF, and
the target harness is SHA E5C7AFB9C4EAA48C1ACB031689910F38…  With diagnostics on,
sizeof becomes 0x38 and sequences moves to +0x38000, invalidating every offset in every
guard.  Any rebuild changes this baseline and must re-derive the offsets before use.
```

## 7. Runtime Authorization, to be requested separately after Static Validation

```text
cdb.exe -cf <repaired script> -logo <log> build\Release\AudioEngineHarness.exe --measurement=normal
```

`-logo`, never `-o`. One execution. No Retry-1 / 2 / 3.

### 7.1 R1, judged first and alone

```text
SCRIPT_BEGIN = 1
SCRIPT_END   = 1
ZwTerminateProcess = 1
quit         = 1
exit code    = 0
debugger errors = 0
```

```text
R1 FAIL  ->  STOP.  Attribute the failure from the log, close the Result Audit, do not interpret
             R2 to R6, do not retry.
R1 PASS  ->  and only then evaluate R2 to R6.
```

### 7.2 R2 to R6, conditional on R1 PASS

```text
R2   dwo / T0E evidence
R3   T0E -> T0P -> T0Q -> T0S traversal
R4   per-hit pairing
R5   four-way discrimination
R6   strict interleaving
```

### 7.3 The decision tree this run is expected to produce

```text
gate OPENS, C3_T0_ENTRY emitted
    -> conjuncts 1, 2 and 3 all held at that hit
    -> DG-T0-Repair-3b closes for the address 0x1f9c550
    -> the enqueue CAS bracket becomes observable end to end
    -> also observe whether 0x1f9cfb0 and 0x1f9ce04 are reached, which retires or confirms
       the residual T1-phase risk recorded in DG-T0-Repair 5.2

gate does NOT open, but R1 PASSES
    -> every body reached its terminal gc, so the construct defect is repaired
    -> the diagnostic verdicts localise which of conjunct 2 or 3 is false
    -> 3b still does not close, because no successful dwo read was observed

R1 FAILS
    -> a further construct or continuation problem exists outside the scope repaired here
    -> attribute, close, do not retry
```

### 7.4 What remains open regardless of the outcome

```text
conjunct 2 @r9 == 9      n=1 FALSE at Redesign-9, n=1 TRUE at Step5BB Retry-1.  Not generalised
enqueuePos raw value     still NOT OBTAINED.  221 at one hit, n=1, Redesign-7
0x1f9ce04, 0x1f9cfb0     still malformed, deliberately unrepaired
DG-T0-Repair-3b          closes only on a successful dwo read
```

## 8. T1 is not touched, and the ordering this forces

```text
repair T0
   -> observe the T0 path to its end
   -> observe whether 0x1f9cfb0 and 0x1f9ce04 are reached
   -> only then decide the T1 repair
```

Excluded from this CDB by this authorization, and to remain excluded:

```text
0x1f9ce04 repair
0x1f9cfb0 repair
ReaderSlot measurement
minReaderEpoch
Case A / B / C / D
T1 capture
```

This ordering is what keeps the effect of the T0 repair distinguishable from the pre-existing T1
defects, instead of confounding them in a single run.

## 9. State

```text
DG-T0-Repair             PASS / ADJUDICATED
repair scope             A'  = 4 T0 sites, 8 occurrences
operator                 dwo
parent artifact          Redesign-9, SHA D48665B8…AD14, 14,887 B
predicted product        14,903 B
new baseline             T0-Guard-Repair-Baseline-1, defined, not yet instantiated
CDB creation             NOT YET AUTHORIZED
static validation        NOT YET RUN
runtime                  NOT AUTHORIZED
source / tests / CMake / build   untouched
T1                       untouched
IMPLEMENTATION           authorized, scope as above
```

## 10. Next step

Implementation of the 8-occurrence repair, then Static Validation S-T0-01 to S-T0-17, then stop and
request Runtime Authorization. Runtime is a separate gate and is not covered by this document.

## 11. Tooling defects in this gate

Both are mine, both produced a wrong intermediate statement, neither touched an artifact. They are
recorded because the first is the second occurrence of a pattern already logged once.

| # | defect | consequence | correction |
|---|---|---|---|
| 1 | Artifact path hand-transcribed as `…C3-T1-Capture_Redesign-7_Operand-Capture.cdb`; the real name uses a hyphen, `…C3-T1-Capture-Redesign-7_Operand-Capture.cdb` | `Get-FileHash` failed, printing a bare `False` beside the filename and momentarily implying the frozen Redesign-7 artifact had changed | the file was never altered. Re-verified by resolving the name from a directory listing: SHA `5AA98A59…C5F9`, 14,785 B, match |
| 2 | A frozen-SHA check keyed artifacts by a regex tag `Step5[A-Z]{2}` extracted from the filename | two distinct files share the `Step5BB` tag, `…C3-RP_retry_not_executed.cdb` (SHA `AFCC0017…`) and `…C3-T1-Capture-Redesign-2.cdb` (SHA `FBE32851…`). One expectation was applied to both, producing a false MISMATCH on the former | keyed on the full filename instead. All 5 frozen baselines match |

Defect 1 is a **repeat**. `Capture_Redesign-4` versus `Capture-Redesign-4` caused the same false
`FileNotFoundError` in the Step5DE gate. The recurring cause is transcribing a long artifact
filename by hand into a shell command, in a corpus whose names mix `_` and `-` at exactly the
segment boundary being typed.

```text
COUNTERMEASURE, binding on the implementation and validation steps of this gate:
  never hand-transcribe an artifact filename into a command.
  resolve it from a directory listing, or have the generating script write the path it used.
  A missing-file error in this corpus is a suspect path string until proven otherwise.
```

Neither defect affected the authorization content, which was derived from an in-process paren-aware
scan that read the files by resolved path and asserted the exact occurrence counts (8 repair, 2
excluded at `0x1f9ce04`, 2 excluded at `0x1f9cfb0`) before the authorization was written.

### 11.1 Post-gate integrity, verified

```text
.cdb artifacts            13, none created, none modified
frozen baselines           Step5BB Redesign-2, Step5CP Redesign-4, Step5CX Redesign-6,
                           Step5DA Redesign-7, Step5DD Redesign-9   all match
ConvoPeq.md                match
AudioEngineHarness.exe     match
tmp\cdb.exe                match
git src/ CMakeLists build.bat delta   12, all pre-existing
CDB execution              0
residue processes          0
build invoked              no
```
