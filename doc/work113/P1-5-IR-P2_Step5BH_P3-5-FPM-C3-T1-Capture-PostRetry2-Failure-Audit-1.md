# P3-5-FPM-C3-T1-Capture-PostRetry2-Failure-Audit-1

## 1. Gate result

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Failure-Audit-1
Mode                                  = read-only failure / static expression audit
Failure primitive                     = IDENTIFIED
Affected execution expressions        = 12 exact dd(address) command-as-expression sites
RVA/source identity                   = RECONCILED
Replacement syntax family             = MASM dwo(address), plus command-body cleanup
Same measurement design preservable   = YES
Hit-based benign parser probe needed  = YES
New runtime probe required            = YES, in a separately authorized gate
Retry authorization                   = NOT GRANTED
Retry-2 authorization reuse           = FORBIDDEN / CONSUMED
S7_READER                             = UNRESOLVED
Case A/B/C/D                          = NOT_PROVEN
IMPLEMENTATION                        = FORBIDDEN
```

This gate did not launch CDB or a debuggee, create a corrected runtime script, modify Redesign-2, run the Harness, build, or change source/test/CMake.

## 2. Frozen identity reconciliation

| artifact | SHA-256 | result |
|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | PASS |
| Redesign-2 CDB | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | PASS |
| Retry-2 CDB log | `3C38B24CE1690820D1816C9F5F7F6CF54D3483CE5A0F74A74CB379E2E440DCEC` | PASS |

`ConvoPeq.md` is the production source authority. The separately attached older `ConvoPeq(3).md` was not substituted.

## 3. Failure primitive

Retry-2 loaded and accepted all 13 breakpoint definitions. The first breakpoint at `AudioEngineHarness+0x1f9c550` was actually hit. At command execution, CDB emitted:

```text
Syntax error at '(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } .else { gc } '
```

The offending frozen predicate was:

```text
.if (dd(@rcx+0x34000) == 4)
```

The failure primitive is not a pseudo-register failure, RVA drift, or breakpoint registration failure. It is use of the debugger display command `dd` where `.if` requires a MASM expression.

Microsoft documents that a `.if` condition:

```text
must be an expression, not a debugger command
```

MASM memory dereference is provided by numeric operators such as:

```text
dwo(address)  = 32-bit value at address
poi(address)  = pointer-sized value at address
```

`dd` remains a display command (`dd address L1`), not the documented MASM expression operator.

## 4. Source and RVA reconciliation

Frozen source in `ConvoPeq.md` contains `DeferredDeletionQueue::enqueue`:

```text
load enqueuePos
load sequence[pos & mask]
compare seq - pos
CAS enqueuePos: pos -> pos+1
write entry
release sequence = pos+1
```

The frozen EXE disassembly shows:

```text
0x1f9c550  mov [rsp+8],rbx
0x1f9c555  mov r10d,[rcx+0x34000]       // enqueuePos
0x1f9c569  mov eax,[r11+rbx*4+0x30000]  // sequence
0x1f9c57d  lock cmpxchg [r11+0x34000]   // enqueuePos CAS
0x1f9c5d2  mov [r11+rbx*4+0x30000],r10d // sequence publication
0x1f9c5da  mov rbx,[rsp+8]
```

The file offset for RVA `0x1f9c550` is `0x1f9b950`; its first bytes are `48 89 5c 24 08`, exactly the instruction logged by Retry-2. The RVA is therefore correctly mapped and was not the failure cause.

## 5. Full breakpoint payload inventory

| RVA | role | `.if` count | `dd(` token count | `poi(` | `dq` display | `db` literal reads | MASM `and` | execution-risk disposition |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `0x1f9c550` | T0 entry | 3 | 1 | 1 | 1 | 0 | 0 | one confirmed invalid `dd()` predicate; `poi` was runtime-proven in Retry-1 |
| `0x1f9c57d` | T0 CAS pre | 4 | 2 | 0 | 0 | 0 | 0 | two invalid `dd()` predicates |
| `0x1f9c59a` | T0 CAS post | 4 | 2 | 0 | 0 | 0 | 0 | two invalid `dd()` predicates |
| `0x1f9c5da` | T0 publication / arming | 7 | 2 | 0 | 2 | 64 | 0 | two invalid `dd()` predicates |
| `0x1fa80b3` | S7 terminal anchor | 4 | 0 | 0 | 2 | 64 | 0 | no confirmed `dd()` family defect; hit-based probe still required |
| `0x1f9cfb0` | T1-A candidate | 8 | 2 | 0 | 3 | 0 | 0 | two `dd()` tokens embedded in display-command address arguments |
| `0x1f9cfe2` | T1-B pre | 5 | 0 | 0 | 0 | 64 | 0 | predicates documented; no prior actual T1 hit |
| `0x1f9cfe5` | T1-B return | 5 | 0 | 0 | 0 | 0 | 0 | predicates documented; no prior actual T1 hit |
| `0x1f9cd00` | T1-C entry | 7 | 0 | 0 | 0 | 0 | 0 | documented register/pseudo/shift equality forms; no prior actual T1 hit |
| `0x1f9cd59` | target epoch gate / selection | 14 | 1 | 6 | 1 | 0 | 7 | combined expression contains one invalid `dd()`; `poi` and `and` documented but not hit-proven |
| `0x1f9cd72` | CAS pre | 12 | 0 | 0 | 1 | 0 | 0 | documented address/equality forms; no prior actual selected T1 hit |
| `0x1f9ce04` | reclaim return | 8 | 2 | 0 | 0 | 0 | 0 | two invalid `dd()` predicates |
| `0x1fa80b2` | T2 anchor | 5 | 0 | 0 | 2 | 64 | 0 | no confirmed `dd()` family defect; Retry-1 proves this breakpoint can hit |

Whole-script command-family totals:

```text
.if predicates / command blocks = 86 occurrences
unique predicate texts          = 73
dd(address) tokens             = 12
poi(address) tokens            = 7
dq display commands            = 12
db literal ReaderSlot reads    = 256
MASM and tokens                 = 7
custom pseudo-registers         = 0
```

## 6. Exact affected-expression enumeration

### 6.1 Confirmed command-as-expression in `.if`

| # | RVA | exact expression fragment | documented defect |
|---:|---|---|---|
| 1 | `0x1f9c550` | `dd(@rcx+0x34000) == 4` | `.if` condition contains display command `dd` |
| 2 | `0x1f9c57d` | `dd(@r11+0x34000) == 4` | same |
| 3 | `0x1f9c57d` | `dd(@r11+0x30010) == 4` | same |
| 4 | `0x1f9c59a` | `dd(@r11+0x34000) == 5` | same |
| 5 | `0x1f9c59a` | `dd(@r11+0x30010) == 4` | same |
| 6 | `0x1f9c5da` | `dd(@r11+0x34000) == 5` | same |
| 7 | `0x1f9c5da` | `dd(@r11+0x30010) == 5` | same |
| 8 | `0x1f9cd59` | combined target-gate predicate containing `dd(@r15)==(@$t10+1)` | same; additional `poi`/`and` forms are documented but not execution-proven by this audit |
| 9 | `0x1f9ce04` | `dd(@rdi+0x34040) == (@$t10+1)` | same |
| 10 | `0x1f9ce04` | `dd(@rdi+0x34040) == @$t10` | same |

### 6.2 Command-token misuse in display arguments

At `0x1f9cfb0`, two occurrences are embedded in display-command address expressions:

```text
dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
dq (@$t2+0xc0+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
```

The outer `dd`/`dq` commands are valid display commands. Their address arguments are MASM expressions, so the inner `dd(...)` tokens are still invalid as expression syntax. These two sites are likely to fail when the positive T1-A branch executes.

This gives:

```text
10 invalid dd(...) inside .if conditions
2 invalid dd(...) inside display-command address expressions
12 affected sites total
```

## 7. Other expression families

| family | inventory | evidence status |
|---|---|---|
| `@register` comparisons | hardware registers such as `@r9`, `@r11`, `@rsi` | documented; actual T0/T2 hits in Retry-1 prove basic use |
| `@$tN` comparisons | `$t0..$t19` | documented; prior CDB pseudo-register probe and Retry-1 assignments/hits prove basic use |
| `@$tid` | candidate/S7 thread correlation | documented automatic pseudo-register; no actual T1-S7 correlated hit exists |
| `poi(address)` | 7 tokens; six at T1-C, one as `ln poi(@rsp)` at T0 | documented; `poi(@rcx+0x34000)` was used and succeeded in actual Retry-1 T0 entry |
| `dwo(address)` | absent in Redesign-2 | documented replacement family for 32-bit values |
| arithmetic `+`, `*`, `&` | present | documented MASM operators; simple register arithmetic was hit-proven, complex ring-index arithmetic was not |
| shifts `>> 32` | present in low/high pointer joins | documented MASM operators; not yet proven by a selected T1 hit |
| equality / `!=` | present throughout | documented MASM operators; basic forms hit-proven |
| MASM `and` | 7 tokens in T1-C combined predicate | documented bitwise AND; not yet proven by a selected T1 hit |
| `dq` / `db` / `dd` display commands | present | valid outside expression positions; actual Retry-1 payloads executed many `dq`, `db`, and `dd` display commands |

## 8. Prior runtime comparison

Retry-1 used `poi(@rcx+0x34000) == 4` at the same T0 entry. It produced `C3_T0_ENTRY` with no syntax error. It also produced `C3_T0_CAS_PRE` and `C3_T0_CAS_POST` and later `C3_T2_WAIT_RETURN` with no syntax error.

This proves:

```text
@register/pseudo-register conditions     = execution-time accepted for the prior forms
poi(address) 32-bit comparison           = execution-time accepted
display commands db/dq/dd as commands    = execution-time accepted
```

It does not prove the redesigned T1 candidate/getMin/reclaim command programs, because Retry-2 failed before target arming and Retry-1 did not execute T1 under the identity-correlated state.

## 9. Replacement syntax family

The direct documented replacement for a 32-bit memory comparison is:

```text
dd(address)  == N   ->  dwo(address) == N
```

Examples for the identified sites:

```text
dwo(@rcx+0x34000) == 4
dwo(@r11+0x34000) == 4
dwo(@r11+0x30010) == 4
dwo(@r15) == (@$t10+1)
dwo(@rdi+0x34040) == (@$t10+1)
```

For the T1-A display address arguments, the same expression family should be used:

```text
dd @$t2+0x30000+((dwo(@$t2+0x34040)&0xfff)*4) L1
dq (@$t2+0xc0+((dwo(@$t2+0x34040)&0xfff)*0x30)) L6
```

This is a syntax-family identification only. No corrected script is created in this gate.

## 10. Same measurement design preservability

The intended causal design can be preserved without changing production or test code:

```text
same T0 publication identity
same target DQueue/EpochDomain
same candidate thread/token
same getMin RAX capture
same R13 target gate
same ticket/sequence/ptr/deleter/epoch join
same CAS/blocked classification
same 4 x 64 literal ReaderSlot captures
same T2 anchor
```

Only debugger expression spelling and command-body validation need to change. No production instrumentation, telemetry, getter, counter, source layout, or runtime measurement point is required.

Disposition:

```text
same measurement design preservable = YES
```

## 11. Required next probe

A `bu`/`bl` registration probe is insufficient. The next independent gate must test each revised command body at a breakpoint that is actually hit, while using a benign non-Harness target.

Minimum probe matrix:

| probe group | must hit | forms to evaluate |
|---|---|---|
| A | yes | `dwo(@reg+offset) == constant` and false branch |
| B | yes | `@reg`, `@$tN`, `!=`, `==` |
| C | yes | 64-bit low/high split using `&0xffffffff` and `>>32` |
| D | yes | `poi(address)` and pointer high/low split |
| E | yes | complex ring address `base+0xc0+((ticket&0xfff)*0x30)` |
| F | yes | combined equality with MASM `and` |
| G | yes | display command with `dwo` embedded in an address argument |
| H | yes | positive branch marker, negative branch marker, and `gc` |
| I | yes | hard-stop `q` path after a marker |

The probe must verify both:

```text
command registration
breakpoint-hit command execution
```

An unresolved `bu` plus `bl` can no longer be the sole parser acceptance evidence.

## 12. Probe design requirements

The future benign probe must:

```text
use a harmless executable that reliably hits selected addresses
use the same CDB 10.0.29617.1000 engine
use the same MASM evaluator
initialize any pseudo-registers explicitly
hit every revised expression family
emit unique PASS/FAIL markers
fail on any Syntax error, Bad register, Couldn't resolve, or silent branch failure
terminate and leave no CDB/debuggee residue
never load AudioEngineHarness
```

A synthetic software breakpoint target or a dedicated benign helper executable may provide deterministic hits. The helper must not be a new production/test source change. If no existing deterministic benign hit path is available, the next gate must explicitly authorize and place a session-scoped helper outside the project tree.

## 13. Gate STOP conditions for a future redesign

Any future correction/preparation gate stops if:

```text
any dd(...) remains in a MASM expression position
any display command is used as an .if condition
hit-based probe does not execute every revised command family
any probe command emits an error or silently misses an expected branch
same design requires source/test/CMake changes
RVA/source/EXE/PDB/script identity changes
```

## 14. Scope compliance

| prohibited action | result |
|---|---|
| Retry-3 | NOT PERFORMED |
| Redesign-2 direct edit | NOT PERFORMED |
| new runtime CDB script | NOT CREATED |
| CDB process | NOT STARTED |
| AudioEngineHarness | NOT STARTED |
| M1/M2 | NOT PERFORMED |
| build | NOT PERFORMED |
| Dr.Memory | NOT PERFORMED |
| production source edit | NOT PERFORMED |
| test source edit | NOT PERFORMED |
| CMake edit | NOT PERFORMED |
| implementation | FORBIDDEN |

## 15. Final disposition

```text
failure primitive                  = IDENTIFIED
all affected execution expressions = ENUMERATED (12 sites)
RVA/source identity                = RECONCILED
replacement syntax family          = IDENTIFIED: MASM dwo(address)
same measurement design preservable = YES
new runtime probe required         = YES
Retry authorization                 = NOT GRANTED
Retry-2 authorization reuse        = FORBIDDEN / CONSUMED
S7_READER                          = UNRESOLVED
Case A/B/C/D                       = NOT_PROVEN
IMPLEMENTATION                     = FORBIDDEN
```

The next permissible work is a separate read-only design/preparation gate that defines a hit-based benign probe and, only after that probe is separately authorized and passes, a new runtime authorization review. This audit does not authorize any of those executions.

## 16. Authoritative references

- Microsoft Learn, `.if` command: a condition must be an expression, not a debugger command.
- Microsoft Learn, MASM Numbers and Operators: `dwo`, `poi`, registers/pseudo-registers, arithmetic, shifts, equality, and `and`.
- Microsoft Learn, Pseudo-Register Syntax: `$tN` and `@$tN` usage.
- Project Retry-1 log: actual hit-time evidence for `poi`, hardware/pseudo-register predicates, and display commands.
- Project Retry-2 log: actual hit-time `Syntax error` at `dd(@rcx+0x34000)`.
