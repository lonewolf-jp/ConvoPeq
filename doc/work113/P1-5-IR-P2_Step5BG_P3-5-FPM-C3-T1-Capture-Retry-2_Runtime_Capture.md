# P3-5-FPM-C3-T1-Capture-Retry-2 — Exact One Runtime Capture

## 1. Final result

```text
Gate                  = P3-5-FPM-C3-T1-Capture-Retry-2
Mode                  = runtime debugger capture
Stimulus              = AudioEngineHarness.exe --fpm-m0
Authorized attempts   = 1
Attempts used         = 1
Authorization state   = CONSUMED
CDB process exit      = completed at scripted q
Harness process       = terminated with CDB
CDB/Harness residue   = 0
T0                    = NOT CAPTURED
S7                    = NOT CAPTURED
T1 candidate          = NOT CAPTURED
T1-B pre/return       = NOT CAPTURED
minReaderEpoch RAX    = NOT CAPTURED
T1-C                  = NOT CAPTURED
T1_SELECTION          = NOT CAPTURED
CAS / epoch blocked   = NOT CAPTURED
T2                    = NOT CAPTURED
ReaderSlot[0..63]     = NOT CAPTURED
Hard-stop marker      = NONE
Failure class         = CDB_EXECUTION_TIME_MASM_SYNTAX_ERROR
S7_READER             = UNRESOLVED
Case A/B/C/D          = NOT_PROVEN
IMPLEMENTATION        = FORBIDDEN
Gate verdict          = STOP / RETRY-2 CONSUMED
```

No second Retry-2 execution, script correction, M1/M2, build, source/test/CMake edit, or Dr.Memory run was performed.

## 2. Frozen preflight

| artifact | SHA-256 | result |
|---|---|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | PASS |
| Redesign-2 CDB | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | PASS |
| CDB engine | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | PASS |

Preflight also confirmed:

```text
CDB version             = 10.0.29617.1000
breakpoint count        = 13
ReaderSlot literal rows = 256
pseudo-register set     = $t0..$t19
old state 2/3 gating    = 0
pre-run process residue = 0
```

All required identities matched immediately before the invocation.

## 3. Exact invocation

The only authorized runtime invocation was:

```text
C:\VSC_Project\ConvoPeq\tmp\cdb.exe
  -logo C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BG_P3-5-FPM-C3-T1-Capture-Retry-2_CDB.log
  -cf C:\VSC_Project\ConvoPeq\doc\work113\P1-5-IR-P2_Step5BB_P3-5-FPM-C3-T1-Capture-Redesign-2.cdb
  C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe
  --fpm-m0
```

CDB recorded:

```text
CommandLine = "C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe" "--fpm-m0"
Process     = 6a80
Module base = 00007ff7`57af0000
PDB GUID identity was not printed by CDB; the preflight and post-run file hashes establish the frozen PDB
```

The script was used byte-for-byte as SHA `FBE32851...CFC39E`. It was not edited.

## 4. CDB log evidence

| property | value |
|---|---|
| log | `doc/work113/P1-5-IR-P2_Step5BG_P3-5-FPM-C3-T1-Capture-Retry-2_CDB.log` |
| SHA-256 | `3C38B24CE1690820D1816C9F5F7F6CF54D3483CE5A0F74A74CB379E2E440DCEC` |
| bytes | 44,852 |
| script begin line | 77 |
| first failure line | 304 |
| terminal | lines 307-310, `C3_T1_REDESIGN2_SCRIPT_END` then `quit:` |
| process residue after run | 0 |

The initial `*** WARNING: Unable to verify checksum ...` is a symbol checksum warning after the external file hash had already passed. It is not the terminal failure.

## 5. Runtime marker matrix

Only standalone marker output lines were counted; marker text embedded in `bp`/`bl` command listings is excluded.

| stage | marker | count | disposition |
|---|---|---:|---|
| T0 entry | `C3_T0_ENTRY` | 0 | NOT CAPTURED |
| T0 CAS pre | `C3_T0_CAS_PRE` | 0 | NOT CAPTURED |
| T0 CAS post | `C3_T0_CAS_POST` | 0 | NOT CAPTURED |
| T0 publication | `C3_T0_SEQUENCE_PUBLISHED` | 0 | NOT CAPTURED |
| S7 | `C3_TERMINAL_S7_ANCHOR` | 0 | NOT CAPTURED |
| T1-A | `C3_T1A_CANDIDATE` | 0 | NOT CAPTURED |
| T1-B pre | `C3_T1B_GETMIN_CALL_PRE` | 0 | NOT CAPTURED |
| T1-B return | `C3_T1B_GETMIN_RETURN` | 0 | NOT CAPTURED |
| T1-C | `C3_T1C_DQUEUE_ENTRY` | 0 | NOT CAPTURED |
| T1 selection | `C3_T1_SELECTION` | 0 | NOT CAPTURED |
| CAS pre | `C3_T1_CAS_PRE` | 0 | NOT CAPTURED |
| CAS success | `C3_T1_CAS_SUCCEEDED` | 0 | NOT CAPTURED |
| Epoch blocked | `C3_T1_EPOCH_BLOCKED` | 0 | NOT CAPTURED |
| Candidate no target | `C3_CANDIDATE_NO_TARGET` | 0 | NOT CAPTURED |
| T2 | `C3_T2_WAIT_RETURN` | 0 | NOT CAPTURED |

No `C3_CR1_STOP_*` marker was emitted. The failure occurred in command parsing at the first breakpoint before its positive branch executed.

## 6. Exact failure

At `AudioEngineHarness+0x1f9c550`, CDB emitted:

```text
Syntax error at '(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } .else { gc } '
AudioEngineHarness+0x1f9c550:
00007ff7`59a8c550 48895c2408      mov     qword ptr [rsp+8],rbx
```

The failing condition belongs to the frozen T0 entry command:

```text
.if (dd(@rcx+0x34000) == 4)
```

CDB's MASM expression parser rejected the command program at execution time. The `bl` registration-time check had retained the command string, but it did not evaluate the breakpoint's internal expression program. This run therefore demonstrates that Redesign-3's benign `bu` retention check was insufficient to prove breakpoint-command execution-time expression acceptance. That is an audit finding, not a reason to alter or rerun this gate.

After the parser error, CDB consumed the remaining script lines, emitted the script-end marker, and executed `q`. The debuggee and debugger terminated.

## 7. T0 result

```text
T0 process identity  = capture did not reach T0 payload
T0 DQueue            = NOT CAPTURED
T0 EpochDomain       = NOT CAPTURED
T0 ptr/deleter/epoch = NOT CAPTURED
T0 ticket/sequence   = NOT CAPTURED
targetArmed          = NOT SET BY T0 PAYLOAD
```

The T0 breakpoint was physically hit, but the command program failed before `.echo C3_T0_ENTRY`. Therefore the hit is not accepted as target-publication evidence.

## 8. T1 result

```text
candidate thread       = NOT CAPTURED
invocation token       = NOT CAPTURED
globalEpoch at getMin  = NOT CAPTURED
T1-B ReaderSlot[0..63] = NOT CAPTURED
getMin RAX             = NOT CAPTURED
target R13             = NOT CAPTURED
T1-C identity          = NOT CAPTURED
T1_SELECTION           = NOT CAPTURED
CAS result             = NOT CAPTURED
epoch-blocked result   = NOT CAPTURED
```

`minReaderEpoch` remains `NOT_CAPTURED` for this execution. It must not be reconstructed from global/head epoch equality, historical logs, or inactive T0/S7/T2 snapshots.

## 9. ReaderSlot result

No ReaderSlot block began:

| block | BEGIN | END | captured slot count |
|---|---:|---:|---:|
| T0 | 0 | 0 | 0 |
| S7 | 0 | 0 | 0 |
| T1-B pre-call | 0 | 0 | 0 |
| T2 | 0 | 0 | 0 |

Accordingly, the required 64-slot T1-B state is absent. This is a hard capture failure.

## 10. Chronology and attribution

```text
T0 publication  = NOT CAPTURED
T1_SELECTION    = NOT CAPTURED
T2              = NOT CAPTURED
```

The required ordering `T0 < T1_SELECTION < T2` is not established.

No Case A/B/C/D classification is valid for this run. The evidence contains neither a selected target reclaim nor a same-invocation `getMinReaderEpoch` return.

```text
S7_READER    = UNRESOLVED
Case A/B/C/D = NOT_PROVEN
```

## 11. Hard-stop disposition

No `C3_CR1_STOP_*` hard-stop marker was emitted. The terminal condition was an unhandled CDB execution-time syntax error.

Disposition:

```text
Retry permitted after this gate = NO
Script correction in this gate  = FORBIDDEN
Immediate Retry-2 rerun        = FORBIDDEN
Retry-2 state                  = CONSUMED
```

Any future debugger-script change or another runtime capture requires a new independently authorized gate.

## 12. Scope verification

| prohibited action | result |
|---|---|
| script edit during Retry-2 | NOT PERFORMED |
| Retry-2 rerun | NOT PERFORMED |
| M1/M2 | NOT PERFORMED |
| build | NOT PERFORMED |
| Dr.Memory | NOT PERFORMED |
| production source edit | NOT PERFORMED |
| test source edit | NOT PERFORMED |
| CMake edit | NOT PERFORMED |
| implementation | FORBIDDEN |

Unrelated working-tree source/test modifications existed before this gate. This gate did not modify or overwrite them.

## 13. Final disposition

```text
P3-5-FPM-C3-T1-Capture-Retry-2 = STOP / RETRY-2 CONSUMED
CDB log SHA-256                 = 3C38B24CE1690820D1816C9F5F7F6CF54D3483CE5A0F74A74CB379E2E440DCEC
T0                              = NOT CAPTURED
T1                              = NOT CAPTURED
minReaderEpoch                  = NOT CAPTURED
ReaderSlot[0..63]               = NOT CAPTURED
S7_READER                       = UNRESOLVED
Case A/B/C/D                    = NOT_PROVEN
IMPLEMENTATION                  = FORBIDDEN
```
