# P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-2

## 1. Gate result

```text
Gate                                  = P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-2
Mode                                  = read-only / static correction
Preparation-1 result                  = SUPERSEDED
Family G readable target              = PROVEN
State/counter register allocation     = PROVEN
Seven expression families             = RECHECKED / STATIC VALIDATED
New probe script                      = NOT CREATED
CDB execution                         = 0
ping execution                        = 0
AudioEngineHarness execution          = 0
Retry-3                               = 0
M1/M2                                 = 0
Build                                 = 0
Dr.Memory                             = 0
Source/test/CMake changes             = 0
Probe runtime authorization           = NOT GRANTED
DIRECT_DQUEUE_CAUSE                   = epoch gate equality PROVEN
S7_READER                             = UNRESOLVED
S7_READER_SLOT                        = UNRESOLVED
minReaderEpoch at target reclaim      = NOT_CAPTURED
Case A/B/C/D                          = NOT_PROVEN
IMPLEMENTATION                        = FORBIDDEN
```

This gate corrects the static probe design only. It does not start CDB, ping, or AudioEngineHarness and does not reuse Retry-2 authorization.

## 2. Frozen identities

| artifact | SHA-256 | result |
|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | PASS |
| Redesign-2 CDB | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | PASS |
| Retry-2 log | `3C38B24CE1690820D1816C9F5F7F6CF54D3483CE5A0F74A74CB379E2E440DCEC` | PASS |
| PostRetry2 failure audit | `02CCE6B7A5024FDE6AA9CF8ED6F2B1458263E0E635B4E449655E081143989619` | PASS |
| Preparation-1 | `A6753D38D214DDDD66E791C9E12C5F979B197F2CECE8B438F585D2B13C953B24` | superseded for corrected addresses/register map |
| `C:\Windows\System32\ping.exe` | `E4224D18C3C96826E6F4240893FAC02C8A1FF335CE92614FC8744369594DD468` | PASS |

## 3. Preparation-1 defect disposition

Preparation-1 correctly identified seven expression families and the 12 `dd(address)` replacement sites, but it was not closed as final because:

```text
Family G old computed address = ping!+0x30400
ping SizeOfImage              = 0xC000
old address inside image      = false
symbolic pseudo names         = $probeState, $probeHit, $probePass, $probeFail
allowed pseudo-register set   = $t0..$t19
```

This gate replaces the invalid Family G arithmetic and freezes the symbolic state names to real CDB pseudo-registers.

## 4. `ping.exe` PE memory map

The frozen target has:

```text
file bytes       = 45,056 = 0xB000
SizeOfImage      = 0xC000
SizeOfHeaders    = 0x1000
SectionAlignment = 0x1000
FileAlignment    = 0x1000
```

Relevant mapped ranges:

| range | section/region | mapped | readable | raw-backed |
|---|---|---:|---:|---:|
| `0x0000..0x0FFF` | PE headers | yes | yes | yes |
| `0x1000..0x3FFF` | `.text` mapped/raw range | yes | yes | yes |
| `0x4000..0x4FFF` | `fothk` | yes | yes | yes |
| `0x5000..0x6FFF` | `.rdata` | yes | yes | yes |
| `0x7000..0x7FFF` | raw-backed `.data` portion | yes | yes | yes |
| `0x8000..0x8A3F` | zero-filled `.data` virtual tail | yes | yes | no raw backing needed for readability |
| `0x9000..0x91AF` | `.pdata` | yes | yes | yes |
| `0xA000..0xA81F` | `.rsrc` | yes | yes | yes |
| `0xB000..0xB087` | `.reloc` | yes | yes | yes |

The removed `ping!+0x30400` is outside `SizeOfImage=0xC000` and outside every section. It is not a valid static display target.

## 5. Family A — `dwo` PE-header constant

```text
r $t0 = ping!+0x3c
.if (dwo(@$t0) == 0x100) { .echo PASS_A_POS; gc } .else { .echo FAIL_A_POS; q }
.if (dwo(@$t0) != 0x100) { .echo FAIL_A_NEG; q } .else { .echo PASS_A_NEG; gc }
```

Proof:

```text
ping RVA 0x3c is inside SizeOfHeaders 0x1000
file offset 0x3c
raw dword    = 0x00000100
dwo result   = 0x100
```

Both positive and negative predicates use readable target-image memory.

## 6. Family B — hardware/pseudo register and equality

```text
r $t0 = 0
r $t1 = 1
.if (@$t0 == 0) { .if (@$t1 == 1) { .echo PASS_B_POS; gc } .else { .echo FAIL_B_POS; q } } .else { q }
.if (@$t0 != 0) { .echo FAIL_B_NEG; q } .else { .echo PASS_B_NEG; gc }
.if (@rsp != 0) { .echo PASS_B_HW_POS; gc } .else { .echo FAIL_B_HW_POS; q }
.if (@rsp == 0) { .echo FAIL_B_HW_NEG; q } .else { .echo PASS_B_HW_NEG; gc }
```

The design uses only `$t0`, `$t1`, `@rsp`, `==`, and `!=`. It performs no reader attribution.

## 7. Family C — 64-bit low/high split

```text
r $t14 = @r13&0xffffffff
r $t15 = @r13>>32
.if ((@r13&0xffffffff) == @$t14) { .if ((@r13>>32) == @$t15) { .echo PASS_C_POS; gc } .else { .echo FAIL_C_POS; q } } .else { q }
.if ((@r13&0xffffffff) == (@$t14+1)) { .echo PASS_C_NEG; gc } .else { .echo FAIL_C_NEG; q }
```

The test captures both halves of one live 64-bit register and rejoins them to the same value. It does not assign a reader meaning to the resulting number.

## 8. Family D — `poi` at readable raw-backed qword

The prior `poi(ping!+0x3c) == 0x100` design mixed a pointer-sized 8-byte read with a 4-byte dword result. It is corrected to a non-zero raw-backed qword:

```text
r $t1 = ping!+0x5000
.if (poi(@$t1) != 0) { .echo PASS_D_POS; gc } .else { .echo FAIL_D_POS; q }
.if (poi(@$t1) == 0) { .echo FAIL_D_NEG; q } .else { .echo PASS_D_NEG; gc }
```

Proof:

```text
RVA 0x5000
section .rdata (0x5000..0x6FFF)
IMAGE_SCN_MEM_READ = set
raw-backed          = yes
file offset         = 0x5000
8 raw bytes         = 70 71 00 40 01 00 00 00
qword               = 0x0000000140007170 != 0
```

## 9. Family E — complex ring-address expression

```text
r $t10 = 4
r $t2 = ping!+0x40
r $t3 = @$t2+0xc0+((@$t10&0xfff)*0x30)
.if (@$t3 == ping!+0x1c0) { .echo PASS_E_POS; gc } .else { .echo FAIL_E_POS; q }
.if (@$t3 != ping!+0x1c0) { .echo FAIL_E_NEG; q } .else { .echo PASS_E_NEG; gc }
```

Arithmetic:

```text
0x40 + 0xC0 + (4 & 0xFFF) * 0x30
= 0x40 + 0xC0 + 0xC0
= 0x1C0
```

Proof:

```text
RVA 0x1C0 is inside SizeOfHeaders 0x1000
file offset 0x1C0
raw byte is available
```

The corrected result is `0x1C0`, not the inconsistent `0x160` value in an intermediate audit calculation.

## 10. Family F — MASM `and`

```text
r $t0 = 9
r $t10 = 4
.if ((@$t0 == 9) and (@$t10 == 4)) { .echo PASS_F_POS; gc } .else { .echo FAIL_F_POS; q }
.if ((@$t0 == 9) and (@$t10 == 5)) { .echo FAIL_F_NEG; q } .else { .echo PASS_F_NEG; gc }
```

The negative form holds the first operand true and changes only the second. `&&` and `||` remain prohibited.

## 11. Family G — corrected readable display targets

Family G uses two separate expression bases. Each display address still evaluates `dwo(@$t1)` inside its MASM address expression, while both final targets remain inside the readable, raw-backed `.rdata` section.

```text
$t1 = ping!+0x3c
$t4 = ping!+0x4FD0
$t5 = ping!+0x2600

.if (dwo(@$t1) == 0x100) {
  dd @$t4+((dwo(@$t1)&0xfff)*4) L1
  dq @$t5+((dwo(@$t1)&0xfff)*0x30) L6
  .echo PASS_G_POS
  gc
} .else { .echo FAIL_G_POS; q }

.if (dwo(@$t1) != 0x100) { .echo FAIL_G_NEG; q } .else { .echo PASS_G_NEG; gc }
```

Exact arithmetic:

```text
dwo(@$t1) = 0x100

dd RVA = 0x4FD0 + (0x100 * 4)
       = 0x4FD0 + 0x400
       = 0x53D0
dd end = 0x53D4

dq RVA = 0x2600 + (0x100 * 0x30)
       = 0x2600 + 0x3000
       = 0x5600
dq end = 0x5630
```

Static PE-section and file-backing proof:

| display | RVA range | region | mapped | `IMAGE_SCN_MEM_READ` | fully raw-backed | file range |
|---|---|---|---:|---:|---:|---|
| `dd L1` | `0x53D0..0x53D3` | `.rdata` | yes | yes | yes | `0x53D0..0x53D3` |
| `dq L6` | `0x5600..0x562F` | `.rdata` | yes | yes | yes | `0x5600..0x562F` |

The `.rdata` section covers RVA `0x5000..0x6FFF`, has raw backing through `0x6FFF`, and carries `IMAGE_SCN_MEM_READ`. Both complete display ranges are inside that initialized/readable interval.

The superseded `ping!+0x30400` target is outside `SizeOfImage=0xC000`. It is not used. The superseded `0x4FD0` arithmetic is also not used.

This satisfies the corrected requirement:

```text
valid address expression
inside PE image range
inside readable section/header
fully backed by initialized file data
```

## 12. Explicit state and counter allocation

No symbolic names such as `$probeState` are valid CDB pseudo-registers. The future script must use the real allocated registers:

| role | actual CDB register | initialization | writers |
|---|---|---|---|
| probe state | `$t16` | `0` | initial state / family transition only |
| hit counter | `$t17` | `0` | actual breakpoint-hit path only |
| pass counter | `$t18` | `0` | successful marker-emission path only |
| fail counter | `$t19` | `0` | failed-marker or error path only |

`$t18` is used as the probe failure counter, not the production S7 anomaly latch. The production capture's `$t18` meaning does not carry into this standalone benign probe.

Only `$t0..$t19` may be used. No `$probeState`, `$probeHit`, `$probePass`, or `$probeFail` names are allowed.

## 13. Seven-family executable status

| family | static status | remaining runtime evidence |
|---|---|---|
| A `dwo` | READY | actual hit/evaluation/marker |
| B register/pseudo-register | READY | actual hit/evaluation/marker |
| C low/high split | READY | actual hit/evaluation/marker |
| D `poi` | READY after raw-backed qword correction | actual hit/evaluation/marker |
| E complex ring address | READY after arithmetic/readability correction | actual hit/evaluation/marker |
| F MASM `and` | READY | actual hit/evaluation/marker |
| G display address with `dwo` | READY after mapped/readable/raw-backed correction | actual hit/evaluation/marker |

Static preparation does not claim runtime parser acceptance. That remains the purpose of the separately authorized benign ping execution.

## 14. Registration/execution separation

The following remain independent invariants:

```text
registration PASS
!= breakpoint-hit PASS
!= expression-evaluation PASS
!= expected-branch PASS
!= marker-emission PASS
!= clean-termination PASS
```

No `bu`/`bl` registration result can substitute for a hit-time command execution result.

## 15. Expected marker set

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

A future script must initialize and increment `$t16..$t19` only in the paths specified by this gate.

## 16. STOP conditions

A future benign-probe preparation/execution gate stops on:

```text
ping identity/import/call-site drift
any proposed address outside SizeOfImage
any display range crossing an unmapped boundary
any section lacking IMAGE_SCN_MEM_READ
any display range not fully raw-backed
any symbolic pseudo-register other than $t0..$t19
Family D qword width/value mismatch
Family E arithmetic/result mismatch
Family G dd/dq target mismatch
registration-only evidence presented as execution PASS
missing standalone marker
CDB syntax/register/resolve/operand error
dirty termination or process residue
```

## 17. Scope compliance

| prohibited action | result |
|---|---|
| CDB execution | 0 |
| ping execution | 0 |
| AudioEngineHarness execution | 0 |
| Retry-3 | 0 |
| M1/M2 | 0 |
| build | 0 |
| Dr.Memory | 0 |
| production source modification | 0 |
| test source modification | 0 |
| CMake modification | 0 |
| new probe CDB script creation | 0 |
| implementation | FORBIDDEN |

## 18. Final disposition

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Probe-Preparation-2
= PROBE DESIGN READY / STATIC VALIDATED

Family G readable addresses    = PROVEN
State/counter allocation       = $t16/$t17/$t18/$t19
Seven expression families      = READY / STATIC VALIDATED
New probe script               = NOT CREATED
CDB/ping/Harness execution     = 0
Probe runtime authorization    = NOT GRANTED
DIRECT_DQUEUE_CAUSE            = epoch gate equality PROVEN
S7_READER                      = UNRESOLVED
S7_READER_SLOT                 = UNRESOLVED
minReaderEpoch                 = NOT_CAPTURED
Case A/B/C/D                   = NOT_PROVEN
IMPLEMENTATION                 = FORBIDDEN
```

The next permissible gate is a separate benign `ping.exe` probe runtime authorization review. It must not create a production capture authorization implicitly.
