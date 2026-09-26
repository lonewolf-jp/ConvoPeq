# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Preparation-1

## 1. Gate result

```text
Gate                       = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Preparation-1
Mode                       = design and candidate comparison only
CDB execution              = 0
ping execution             = 0
new .cdb created           = 0
existing .cdb modified     = 0
@$pc replaced in a script  = 0
dwo / poi validated        = 0
Preparation-4              = 0 (still BLOCKED)
AudioEngineHarness         = 0
Retry-3                    = 0
production capture         = 0
M1/M2/build/Dr.Memory      = 0
ConvoPeq source change     = 0
rerun                      = 0
VERDICT                    = DESIGN COMPLETE / B1 SELECTED / NO SCRIPT WRITTEN
```

This gate compares replacement candidates and fixes the next design boundary. It writes no command file.

## 2. Frozen baseline re-verified

| artifact | SHA-256 | result |
|---|---|---|
| Preparation-3 execution log | `4E2BB488253B7FAE83579576DBAE9416777EC2BC6326D4113DDB278488E4BBB4` | identical |
| Preparation-3 script | `671F9D453FF814DD2AE01217A6CAE7F69E0D033A7A0AA70BD42D21E797D63771` | identical |
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | identical |

```text
.cdb files in repository = 4 (all pre-existing and frozen)
process residue          = 0
```

The four existing command files are the C3 retry, the Redesign-2 capture, the consumed benign authorization, and the frozen Preparation-3 probe. None was created or altered in this gate.

## 3. Inherited state

| item | state |
|---|---|
| Candidate A `ping+offset` | **PROVEN** |
| `@$pc` | **FAILED** |
| Candidate B | **FAILED** |
| A == B | **NOT EVALUATED** |
| `dwo` | NOT PROVEN |
| `poi` | NOT PROVEN |
| MASM arithmetic | NOT PROVEN |
| Reader identity | UNRESOLVED |
| `minReaderEpoch` | NOT CAPTURED |
| DQueue epoch-gate equality | **PROVEN** |
| Preparation-4 | **BLOCKED** |
| Retry-3 | FORBIDDEN |

## 4. Structural change: A becomes a base fact

Because Candidate A was proven directly against the observed image base, the A==B cross-check is demoted from a mandatory gate to supporting evidence.

```text
BASE FACT (independent, already proven)
    ping+offset  ->  image-relative anchor
    Candidate A = PROVEN

AUXILIARY (independent re-verification, optional)
    breakpoint-hit address  ->  image base derivation  ->  compare with Candidate A
    failure of this path does NOT invalidate Candidate A
```

Consequence for the next probe: a run in which no candidate B expression works is still a **valid** outcome, as long as Candidate A is reproduced and dumped. The A==B comparison must not be expressed as a condition that can manufacture a FAIL for Candidate A.

## 5. Candidate comparison

### B1 — breakpoint address pseudo-register

```text
r $t11 = @$bp0 - 0x39d9
```

Microsoft Learn, *Breakpoint Syntax*, "Breakpoint pseudo-registers":

> If you want to refer to a breakpoint address in an expression, you can use a pseudo-register with the **$bp***Number* syntax, where *Number* is the breakpoint ID.

Microsoft Learn, *Pseudo-Register Syntax*:

> **$bp***Number* — The address of the corresponding breakpoint. For example, **$bp3** (or **$bp03**) refers to the breakpoint whose breakpoint ID is 3. ... If no breakpoint has an ID of *Number*, **$bp***Number* evaluates to zero.

```text
basis          = documented, and explicitly intended for referring to a breakpoint address
inertness     = independent of architecture, module symbol, and symbol load state
in-session fit = breakpoint id 0 already exists in the observed session
risk           = NOT runtime-proven in this build
failure mode   = evaluates to 0 if the ID is absent, which would surface as an
                 obviously wrong derived base rather than a subtle error
```

### B2 — instruction pointer pseudo-register

```text
r $t11 = @$ip - 0x39d9
```

Microsoft Learn, *Pseudo-Register Syntax*:

> **$ip** — The instruction pointer register. x64-based processors: the same as **rip**.

Microsoft Learn, *MASM Numbers and Operators*:

> You can also use a period (.) to indicate the current instruction pointer. ... This period has the same meaning as the **$ip** pseudo-register.

```text
basis          = documented as the instruction pointer
in-session fit = the probe stops exactly at the breakpoint, so $ip equals the hit address
risk           = NOT runtime-proven in this build
caveat         = the bare period '.' is documented as equivalent but is explicitly
                 disallowed as the first parameter of the r command; the @$ip form
                 avoids that restriction
```

### B3 — architecture register

```text
r $t11 = @rip - 0x39d9
```

```text
basis          = documented as the x64 instruction pointer
risk           = NOT runtime-proven; hard-codes an architecture
caveat         = MASM guidance notes that omitting @ on uncommon registers makes the
                 debugger fall back through hex, symbol, then register parsing,
                 which makes the @ form preferable
preference     = lowest of the three
```

### B4 — Candidate A itself, used as its own cross-check

```text
r $t1 = ping+0x3c
```

```text
status = PROVEN in the consumed run
role   = if B1, B2 and B3 all fail, the A==B cross-check is abandoned rather than forced,
         because Candidate A already stands on its own proof
```

## 6. Selection

```text
PRIMARY   = B1  @$bp0
FALLBACK  = B2  @$ip
EXCLUDED  = B3  @rip  (architecture-bound, weakest documentation fit)
RETAINED  = B4  ping+offset as Candidate A base fact
```

Rationale for B1 as primary:

```text
1. Its documented purpose is exactly "refer to a breakpoint address in an expression",
   which is the precise need: derive a base from a known hit address.
2. It does not depend on architecture, so it survives any future target change.
3. It does not depend on symbol availability, which is the class of failure that
   already cost one run (the ping!+offset form).
4. Its failure mode is loud: an undefined ID yields 0, which cannot be mistaken
   for a plausible image base.
```

B2 is retained as fallback because it is the semantically most direct expression of
"the address I am stopped at", and its `@` form avoids the documented restriction on `.`
as an `r` parameter.

Neither is asserted to work. The next execution exists precisely to find out.

## 7. What the next benign probe must prove

```text
P1  Candidate A still reproduces        -> r $tN = ping+offset, dumped
P2  B1 yields a usable base             -> @$bp0 evaluates to the breakpoint address
P3  derived anchors match Candidate A   -> base+RVA == $t1..$t5   (auxiliary only)
P4  no comment-origin error             -> zero error lines attributable to non-command text
P5  clean termination                   -> final q, process residue 0
```

Explicitly not in scope: `dwo`, `poi`, `dd`, `dq`, dereference of any kind, ring arithmetic, `and`, ReaderSlot reads, epoch capture, reclaim capture.

The probe must perform **no memory dereference**. Its entire purpose is address arithmetic and value observation.

## 8. Comment-defect elimination rule

The Failure Audit established that `;` does not introduce a comment in a `-cf` command file in this build; every comment line produced an error, with the reported token being the source text minus its first two characters.

```text
RULE = the next command file contains no comment lines at all
```

Accepted in-band annotation mechanisms:

```text
.echo <text>    used freely for step narration
```

Rejected:

```text
;  ...          proven to generate errors in this build
*  ...          proven to generate errors in this build
```

Consequence: documentation of intent moves into the design document, not into the command file. The command file carries commands and `.echo` lines only.

## 9. Gate separation

```text
Candidate-B Replacement Preparation-1   (this gate, design only)
        |
        v
static validation                       (structure, no comment lines, no memory reads)
        |
        v
runtime authorization                    (one attempt, frozen script SHA, frozen CDB)
        |
        v
one benign execution                    (ping.exe only)
        |
        v
result audit
        |
        +-- PASS -> next probe in the expression-family sequence
        |
        +-- FAIL -> failure audit; no repair, no rerun in the same gate
```

Preparation-4 remains blocked until this sequence completes with a proven base-derivation path.

## 10. What remains unchanged

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT_CAPTURED
Case A/B/C/D       = NOT_PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The RT-side observation boundary and the NonRT lifetime isolation line are unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 11. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B-Replacement-Preparation-1
= CLOSED / DESIGN COMPLETE / NO SCRIPT WRITTEN

Candidate A (ping+offset)   = PROVEN, retained as base fact
A == B                      = demoted to auxiliary evidence, not a mandatory gate
B1 @$bp0                    = SELECTED PRIMARY, documented, not runtime-proven
B2 @$ip                     = FALLBACK, documented, not runtime-proven
B3 @rip                     = EXCLUDED (architecture-bound)
';' comments                = FORBIDDEN in future command files
memory dereference          = 0 in the next probe
dwo / poi / dd / dq         = NOT PROVEN, out of scope

CDB execution               = 0
new .cdb                    = 0
script modification         = 0
Preparation-4               = BLOCKED / NOT STARTED
Retry-3                     = FORBIDDEN
production C3               = FORBIDDEN
AudioEngineHarness          = FORBIDDEN
M1 / M2 / build / Dr.Memory = FORBIDDEN
implementation              = FORBIDDEN
```
