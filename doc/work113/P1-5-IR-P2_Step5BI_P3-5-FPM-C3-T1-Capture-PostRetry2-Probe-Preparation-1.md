# P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-1

## 1. Gate result

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-1
Mode                                  = read-only probe design / static preparation
Probe design                          = READY / STATIC VALIDATED
Target executable                     = C:\Windows\System32\ping.exe
Target SHA-256                        = E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468
Target arguments                      = 127.0.0.1 -n 32
CDB execution in this gate            = 0
AudioEngineHarness in this gate       = 0
New probe script created              = 0
Retry-3                               = 0
M1/M2                                 = 0
Build                                 = 0
Dr.Memory                             = 0
Production/test/CMake changes         = 0
Runtime authorization                 = NOT GRANTED
```

This gate defines only the future probe contract. It does not create a probe CDB script, start CDB, or run any target.

## 2. Frozen audit inputs

| artifact | SHA-256 | result |
|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | PASS |
| Redesign-2 CDB | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | PASS |
| Retry-2 log | `3C38B24CE1690820D1816C9F5F7F6CF54D3483CE5A0F74A74CB379E2E440DCEC` | PASS |
| PostRetry2 audit | `02CCE6B7A5024FDE6AA9CF8ED6F2B1458263E0E635B4E449655E081143989619` | PASS |

No process residue was present at the start of this gate.

## 3. Failure primitive carried forward

Retry-2 hit `AudioEngineHarness+0x1f9c550`, then CDB rejected the command-body expression:

```text
.if (dd(@rcx+0x34000) == 4)
```

PostRetry2 audit established:

```text
dd(address) is a display command, not a MASM expression operator
.if requires an expression, not a debugger command
MASM dwo(address) is the documented 32-bit memory-value operator
```

The redesign specification therefore requires only expression-family spelling changes, not a change to the measurement design.

## 4. Twelve affected `dd(...)` conversion specifications

No file is edited in this gate. The following is the frozen in-memory replacement specification for a future separate preparation gate.

### 4.1 T0 predicates

| # | frozen affected form | future form |
|---:|---|---|
| 1 | `dd(@rcx+0x34000) == 4` | `dwo(@rcx+0x34000) == 4` |
| 2 | `dd(@r11+0x34000) == 4` | `dwo(@r11+0x34000) == 4` |
| 3 | `dd(@r11+0x30010) == 4` | `dwo(@r11+0x30010) == 4` |
| 4 | `dd(@r11+0x34000) == 5` | `dwo(@r11+0x34000) == 5` |
| 5 | `dd(@r11+0x30010) == 4` | `dwo(@r11+0x30010) == 4` |
| 6 | `dd(@r11+0x34000) == 5` | `dwo(@r11+0x34000) == 5` |
| 7 | `dd(@r11+0x30010) == 5` | `dwo(@r11+0x30010) == 5` |

### 4.2 T1 target-gate / return predicates

| # | frozen affected form | future form |
|---:|---|---|
| 8 | `dd(@r15)==(@$t10+1)` | `dwo(@r15)==(@$t10+1)` |
| 9 | `dd(@rdi+0x34040) == (@$t10+1)` | `dwo(@rdi+0x34040) == (@$t10+1)` |
| 10 | `dd(@rdi+0x34040) == @$t10` | `dwo(@rdi+0x34040) == @$t10` |

### 4.3 T1-A display-command address arguments

The two remaining occurrences are inside display-command address expressions, not `.if` conditions. They must still use `dwo` because the address is parsed as a MASM expression:

```text
dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
→
dd @$t2+0x30000+((dwo(@$t2+0x34040)&0xfff)*4) L1

dq (@$t2+0xc0+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
→
dq (@$t2+0xc0+((dwo(@$t2+0x34040)&0xfff)*0x30)) L6
```

Total:

```text
dd(address) tokens in Redesign-2  = 12
future dwo(address) replacements  = 12
remaining dd(address) tokens      = 0 in the in-memory replacement specification
source/script modified            = 0
```

## 5. Benign target selection

### 5.1 Target

```text
C:\Windows\System32\ping.exe 127.0.0.1 -n 32
```

The target is existing platform software. It is not ConvoPeq production source, ConvoPeq test source, or a project helper.

### 5.2 Static identity and architecture

```text
SHA-256 = E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468
bytes   = 45,056
machine = x86-64
PE magic = 0x20B
image base = 0x140000000
entry RVA = 0x1300
```

The entry point is a guaranteed code hit under CDB. The `.rdata` IAT entry for `Sleep` is statically present at RVA `0x5308`, imported from `api-ms-win-core-synch-l1-2-0.dll`.

Three target-local indirect call sites reference the `Sleep` IAT slot:

| target RVA | file offset | instruction bytes | use |
|---:|---:|---|---|
| `0x1184` | `0x1184` | `ff 15 7e 41 00 00` | primary repeated hit site |
| `0x387d` | `0x387d` | `ff 15 85 1a 00 00` | fallback/secondary hit site |
| `0x39d9` | `0x39d9` | `ff 15 29 19 00 00` | fallback/secondary hit site |

The future probe must register all three target-local addresses or dynamically resolve the imported `Sleep` call path and verify it resolves to these same locations. It must not rely on an unresolved synthetic address.

The loopback argument and `-n 32` provide enough repeated execution to exercise positive and negative branches without an external network destination.

## 6. Seven expression-family contracts

The future probe is one script, but each family has a unique marker and must be evaluated at an actual breakpoint hit. `registration` and `hit execution` are separate results.

### Family A — `dwo` memory constant

```text
r $t0 = ping!+0x3c
.if (dwo(@$t0) == 0x100) { .echo PASS_A_POS; gc } .else { .echo FAIL_A_POS; q }
.if (dwo(@$t0) != 0x100) { .echo FAIL_A_NEG; q } .else { .echo PASS_A_NEG; gc }
```

`ping.exe+0x3c` is the PE header field containing the statically verified value `0x100`. The future probe must resolve `ping` only after the module is loaded, then assign its absolute address to `$t0`. This keeps the probe in valid target image memory and does not reinterpret `RCX`, which is only the `Sleep` duration at a `ping` call site.

### Family B — hardware/pseudo register and equality

```text
r $t0 = 0
r $t1 = 1
.if (@$t0 == 0) { .if (@$t1 == 1) { .echo PASS_B_POS; gc } .else { .echo FAIL_B_POS; q } } .else { q }
.if (@$t0 != 0) { .echo FAIL_B_NEG; q } .else { .echo PASS_B_NEG; gc }
.if (@rsp != 0) { .echo PASS_B_HW_POS; gc } .else { .echo FAIL_B_HW_POS; q }
.if (@rsp == 0) { .echo FAIL_B_HW_NEG; q } .else { .echo PASS_B_HW_NEG; gc }
```

The future script must prove both a live hardware-register operand (`@rsp`) and explicit `@$tN` operands. It must not use reader values or classify a reader Case.

### Family C — 64-bit low/high split

```text
r $t14 = @r13&0xffffffff
r $t15 = @r13>>32
.if ((@r13&0xffffffff) == @$t14) { .if ((@r13>>32) == @$t15) { .echo PASS_C_POS; gc } .else { .echo FAIL_C_POS; q } } .else { q }
.if ((@r13&0xffffffff) == (@$t14+1)) { .echo PASS_C_NEG; gc } .else { .echo FAIL_C_NEG; q }
```

The split values are captured from the same live 64-bit register and immediately rejoined to those halves. The negative branch changes only the expected low half. No `minReaderEpoch` value is assumed and no reader attribution is derived.

### Family D — `poi(address)`

```text
r $t1 = ping!+0x3c
.if (poi(@$t1) == 0x100) { .echo PASS_D_POS; gc } .else { .echo FAIL_D_POS; q }
.if (poi(@$t1) != 0x100) { .echo FAIL_D_NEG; q } .else { .echo PASS_D_NEG; gc }
```

The address is a statically verified readable location in the loaded `ping.exe` image. A memory read failure is a probe failure, not a false PASS.

### Family E — complex ring address

```text
r $t10 = 4
r $t2 = ping!+0x40
r $t3 = @$t2+0xc0+((@$t10&0xfff)*0x30)
.if (@$t3 == ping!+0x1c0) { .echo PASS_E_POS; gc } .else { .echo FAIL_E_POS; q }
.if (@$t3 != ping!+0x1c0) { .echo FAIL_E_NEG; q } .else { .echo PASS_E_NEG; gc }
```

Both operands and result remain inside the `ping.exe` image. The probe validates the full `base + 0xC0 + ((ticket & 0xFFF) * 0x30)` form without reading an uninitialized ConvoPeq queue.

### Family F — MASM `and`

```text
r $t0 = 9
r $t10 = 4
.if ((@$t0 == 9) and (@$t10 == 4)) { .echo PASS_F_POS; gc } .else { .echo FAIL_F_POS; q }
.if ((@$t0 == 9) and (@$t10 == 5)) { .echo FAIL_F_NEG; q } .else { .echo PASS_F_NEG; gc }
```

The negative form holds the first operand true and changes only the second operand. `&&` and `||` are prohibited; MASM `and`/`or` are required by this family.

### Family G — display address with `dwo`

```text
r $t1 = ping!+0x3c
r $t4 = ping!+0
.if (dwo(@$t1) == 0x100) {
  dd @$t4+0x30000+((dwo(@$t1)&0xfff)*4) L1
  dq (@$t4+0xc0+((dwo(@$t1)&0xfff)*0x30)) L6
  .echo PASS_G_POS
  gc
} .else { .echo FAIL_G_POS; q }
.if (dwo(@$t1) != 0x100) { .echo FAIL_G_NEG; q } .else { .echo PASS_G_NEG; gc }
```

`dwo(@$t1)` reads the verified `0x100` field at `ping!+0x3c`. The `dq` result remains inside `ping.exe`; the `dd` address is deliberately allocated as a `ping`-image-relative display address in the future script and must be proven readable by the next gate before execution. The family must emit both display commands without `Bad register`, `Couldn't resolve`, or parser error.

## 7. Branch and hit-state machine

The future script must use a monotonically increasing state pseudo-register and explicit pseudo-register initialization. It must not use the old chronology gating (`$t0 == 2/3`).

Required states:

```text
$probeState = 0       initial
$probeHit  = 0       number of observed target hits
$probePass  = 0      number of PASS markers
$probeFail  = 0      number of FAIL markers
```

The exact state layout may use additional `$tN` values only within `$t0..$t19`. The future script must emit one unique marker per family and branch:

```text
PASS_A_POS / FAIL_A_POS
PASS_A_NEG / FAIL_A_NEG
PASS_B_POS / FAIL_B_POS
PASS_B_NEG / FAIL_B_NEG
PASS_B_HW_POS / FAIL_B_HW_POS
PASS_B_HW_NEG / FAIL_B_HW_NEG
PASS_C_POS / FAIL_C_POS
PASS_C_NEG / FAIL_C_NEG
PASS_D_POS / FAIL_D_POS
PASS_D_NEG / FAIL_D_NEG
PASS_E_POS / FAIL_E_POS
PASS_E_NEG / FAIL_E_NEG
PASS_F_POS / FAIL_F_POS
PASS_F_NEG / FAIL_F_NEG
PASS_G_POS / FAIL_G_POS
PASS_G_NEG / FAIL_G_NEG
PASS_TERMINATION_Q / FAIL_TERMINATION_Q
```

A `PASS` marker must be emitted only after the expected positive branch executes. A `FAIL` marker must immediately set the probe anomaly latch and issue `q`.

## 8. Registration versus execution gate

For every family, the future log must contain six independent results:

| result | evidence required |
|---|---|
| registration | exact `bp` command appears in log and `bl` entry exists |
| breakpoint hit | target address appears in log at least once |
| expression evaluation | no syntax/register/resolve error after the hit |
| expected branch | exact positive or negative marker is emitted |
| marker emission | marker appears as standalone output line, not command echo |
| termination | expected `gc`/final `q` path and clean process exit |

The invariant is explicit:

```text
registration PASS != execution PASS
```

A family with registration PASS and no hit/evaluation/marker evidence is FAIL.

## 9. Hard-stop and clean termination

The production `C3_CR1_STOP_*` predicates are not copied into this probe. The probe uses unique PASS/FAIL markers.

The termination probe is separate:

```text
marker PASS_TERMINATION_Q
q
quit:
```

A clean termination requires:

```text
no FAIL marker
PASS_TERMINATION_Q emitted
`quit:` emitted
no CDB/ping process residue
no AudioEngineHarness process
```

## 10. Target and probe execution boundary

The next gate may execute this probe only after separately authorizing:

```text
target = ping.exe 127.0.0.1 -n 32
CDB = exact frozen engine
```

The next gate must not start `AudioEngineHarness` and must not use the consumed Retry-2 authorization. A failed probe requires a new design review; it does not authorize a runtime capture.

## 11. STOP conditions

Stop the future probe gate if any of the following occurs:

```text
ping identity/architecture/import mismatch
any target-local call site cannot be resolved statically
registration PASS without actual hit evidence
any `dd(...)` remains in an expression position
any missing dwo/poi/operator family marker
any CDB Syntax error, Bad register, Couldn't resolve, Operand error
positive/negative branch mismatch
marker is command echo rather than standalone output
gc/q termination does not produce clean process exit
source/test/CMake change becomes necessary
```

## 12. Static validation performed in this gate

```text
ConvoPeq.md identity                         PASS
Redesign-2 identity                          PASS
Retry-2 log identity                         PASS
PostRetry2 audit identity                    PASS
ping.exe identity                            PASS
ping.exe x64 PE/import inspection             PASS
Sleep IAT RVA 0x5308                          PASS
Sleep call sites 0x1184/0x387d/0x39d9         PASS
12 dd() replacement specification             PASS
remaining dd() in replacement specification  0
pseudo-register range                         $t0..$t19
expression family contracts                   7
registration/execution separation             DEFINED
helper creation                               NOT REQUIRED NOW
CDB execution                                 0
target execution                              0
runtime authorization                         NOT GRANTED
```

## 13. Final disposition

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-1
= PROBE DESIGN READY / STATIC VALIDATED

Probe target: C:\Windows\System32\ping.exe 127.0.0.1 -n 32
Probe CDB: not created
Probe execution: 0
AudioEngineHarness: 0
Retry-3: 0
Retry authorization: NOT GRANTED
S7_READER: UNRESOLVED
Case A/B/C/D: NOT_PROVEN
IMPLEMENTATION: FORBIDDEN
```

A separate gate must authorize execution of this benign hit-based probe before any corrected production capture is considered.

## 14. Authoritative references

- Microsoft Learn, `.if` command: condition must be an expression, not a debugger command.
- Microsoft Learn, MASM Numbers and Operators: `dwo`, `poi`, arithmetic, equality, `and`, and register syntax.
- Microsoft Learn, Pseudo-Register Syntax: `$tN` and `@$tN` forms.
- Microsoft Learn, Conditional Breakpoints: registration and hit-time command execution are separate operations.
- Static `ping.exe` PE import evidence: `Sleep` IAT RVA `0x5308`, with three local indirect call sites.
