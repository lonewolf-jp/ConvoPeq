# P3-5-FPM-C3-T1-Capture-Redesign-3 — Runtime Authorization Review

## 1. Final disposition

```text
Gate                         = P3-5-FPM-C3-T1-Capture-Redesign-3
Mode                         = runtime authorization review
Source authority             = ConvoPeq.md
Identity gate                = PASS
Redesign-2 static gate       = PASS, 27/27
Independent static recheck   = PASS
CDB parser probe             = PASS, 13/13 exact payloads accepted and retained
Parser errors                = 0
Harness execution in gate    = 0
Retry-2 execution in gate    = 0
RETRY-2                      = AUTHORIZED / EXACTLY ONE FUTURE EXECUTION
M1/M2                        = 0
Dr.Memory                     = 0
Build                        = 0
Production source changes    = 0
Test source changes          = 0
CMake changes                = 0
S7_READER                    = UNRESOLVED
Case A/B/C/D                 = NOT_PROVEN
minReaderEpoch               = NOT_CAPTURED
IMPLEMENTATION               = FORBIDDEN
```

This gate authorizes one future Retry-2 execution using the exact frozen Redesign-2 script. It does not execute Retry-2 and does not authorize a second attempt, M1/M2, build, source change, or implementation.

## 2. Frozen identities

| artifact | SHA-256 | bytes | result |
|---|---|---:|---|
| `ConvoPeq.md` | `E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609` | 5,535,334 | PASS |
| `AudioEngineHarness.exe` | `E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75` | 41,216,512 | PASS |
| `AudioEngineHarness.pdb` | `A793738F12F9BA3E0C7A19981B35212659107A356C6904C2BF809933CE1F4391` | 59,355,136 | PASS |
| Redesign-2 CDB | `FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E` | 14,377 | PASS |
| CDB engine | `5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67` | 178,016 | PASS |

CDB version:

```text
10.0.29617.1000
```

`ConvoPeq.md` is the sole production source authority. No older attachment was substituted.

## 3. Redesign-2 static revalidation

The frozen script was rechecked independently of its prior report.

| invariant | result |
|---|---|
| breakpoint count | 13 |
| breakpoint RVA set | exact match |
| user pseudo-registers | `$t0` through `$t19` only |
| custom `$s7*` register | 0 |
| ReaderSlot literal reads | 256 = 4 x 64 |
| `.for` / `.while` | 0 |
| old `$t0 == 2/3` chronology gating | 0 |
| C++ `&&` / `||` in expressions | 0 |
| brace / quote balance | PASS |
| minReaderEpoch source | getMin return RAX only |
| T2 dependency on candidate/T1 | none |
| T1 selection uniqueness | enforced |
| target ptr/deleter/epoch/ticket/sequence/DQueue/domain join | enforced |
| hard-stop markers | 37 duplicated terminal paths / 11 stop classes |

The frozen Redesign-2 preparation report records the full 27/27 result. The independent authorization precheck reproduced the relevant invariants without editing the CDB.

## 4. CDB parser compatibility

### 4.1 Benign target

Parser validation used only:

```text
cdb.exe
cmd.exe /d /c exit
```

The session was stopped at the initial break, registered 13 `bu` commands on unique unresolved synthetic addresses, listed them with `bl`, then used `qd`. It did not execute any breakpoint payload and did not load or start `AudioEngineHarness`.

### 4.2 Exact payload equivalence

The 13 frozen `bp AudioEngineHarness+...` command bodies were extracted without modification and transformed only by:

```text
bp <module+RVA> <payload>
→
bu <unique-unresolved-address> <payload>
```

An exact case-sensitive comparison found:

```text
frozen bp payload bodies = 13
benign bu payload bodies = 13
exact payload match      = true
```

Therefore CDB parser validation exercised the same command programs that Retry-2 will use.

### 4.3 CDB acceptance and retention

CDB `bl` showed unresolved breakpoint entries `0..12`, each retaining its full command body.

```text
payloads accepted = 13/13
bl retained       = 13
parser errors     = 0
Bad register      = 0
Couldn't resolve  = 0
Syntax error      = 0
Unexpected token  = 0
Ambiguous         = 0
```

The completion marker was emitted before `qd`.

Preserved parser evidence:

| artifact | SHA-256 | bytes |
|---|---|---:|
| `doc/work113/P1-5-IR-P2_Step5BF_P3-5-FPM-C3-T1-Capture-Redesign-3_CDB_Parser_Probe.log` | `A72CF3EA68143CC57051C118F86D6814D186F9F06A4156754A024B7FB533BCCA` | 30,886 |

The log contains no `AudioEngineHarness.exe` load or invocation. CDB and Harness process residue after the probe was zero.

This closes the parser-acceptance portion of `STOP-R2-2` without performing Retry-2.

## 5. Object-provenance join

The target DQueue and S7/T1 domain are joined in one execution:

```text
T0 target DQueue = $t2
T0 epochBase     = $t2 - 0x1440
T0 EpochDomain   = $t3 = $t2 - 0x1430

S7 requires      engineThis + 0x10b76c0 == $t3
T1-A requires    RCX == $t3
```

Frozen production source constructs the Engine coordinator/router from the same `m_epochDomain` used for Engine reader registration and reclaim. Therefore the T0-derived DQueue, S7-derived Engine EpochDomain, and T1 candidate domain are identity-correlated without relying on reader values or temporal proximity.

The authorized capture must still record these values and fail closed if this relation does not hold at runtime.

## 6. Pseudo-register authorization

| register | authorized role | overwrite rule |
|---|---|---|
| `$t0` | targetArmed | initialization/reset + T0 publication only |
| `$t1` | epochBase | initialization/reset + T0 publication only |
| `$t2` | target DQueue | initialization/reset + T0 publication only |
| `$t3` | target EpochDomain | initialization/reset + T0 publication only |
| `$t4/$t5` | target ptr | initialization/reset + T0 publication only |
| `$t6/$t7` | target deleter | initialization/reset + T0 publication only |
| `$t8/$t9` | target epoch | initialization/reset + T0 publication only |
| `$t10` | ticket | initialization/reset + T0 publication only |
| `$t11` | candidate thread | reset + candidate admission only |
| `$t12` | candidate active | reset/admission/return completion only |
| `$t13` | invocation token | reset/monotonic candidate increment only |
| `$t14/$t15` | same-invocation minReaderEpoch | reset + getMin return RAX only |
| `$t16` | T1 selected | reset + unique T1_SELECTION only |
| `$t17` | T2 captured | reset + T2 only |
| `$t18` | anomaly | reset + hard-stop path only |
| `$t19` | S7 terminal thread | reset + S7 only |

The CDB parser accepted assignment, readback, and conditional reference forms for this allocation. `$t14/$t15` may not be synthesized from T0/S7/T2 or any historical runtime value.

## 7. Authorized capture sequence

The single future execution may capture:

```text
T0 entry
T0 CAS pre/post
T0 sequence publication
T0 64-slot snapshot
S7 terminal anchor
S7 64-slot snapshot
first same-domain tryReclaim candidate
T1-B getMin pre-call 64-slot snapshot
T1-B getMin return RAX
T1-C DQueue entry
T1_SELECTION at target entry epoch gate
T1 CAS pre
T1 CAS succeeded or epoch blocked
T2 wait return
T2 64-slot final state
```

A provisional candidate that never reaches ticket 4 may terminate as `C3_CANDIDATE_NO_TARGET`; it is not a Case A/B/C/D result.

## 8. Mandatory runtime STOP rules

The future execution controller must classify and terminate the one authorized attempt if any of the following occurs.

### 8.1 Script-side immediate hard stops

All 11 frozen stop classes set `$t18=1`, dump registers, and execute `q`. In CDB, `q` closes the debuggee and exits the debugger.

```text
NESTED_OR_CONCURRENT_CANDIDATE
T1B_THREAD_TOKEN_JOIN
TARGET_DOMAIN_MISMATCH
CANDIDATE_THREAD_MISMATCH
T1C_DQUEUE_ENTRY_JOIN
T1C_THREAD_TOKEN_R13_JOIN
MULTIPLE_TARGET_SELECTION
TARGET_IDENTITY_CONFLICT
T1_CAS_CORRELATION
T1_RETURN_POSITION_CONTRADICTION
T1_RETURN_THREAD_DOMAIN
```

### 8.2 Post-event controller stops

These conditions are adjudicated from the log and process state because they are cross-event requirements rather than single-breakpoint predicates:

```text
T0 sequence publication absent
S7 does not join the T0 target DQueue/EpochDomain
CDB expression/register/parser error
T2 appears before T1_SELECTION when the required target selection is expected
T1_SELECTION lacks same-invocation getMin pre/return
T1_SELECTION lacks a T1-C or target CAS/return classification
T1-B BEGIN/END does not contain all 64 literal ReaderSlot reads
process/CDB cannot terminate and be preserved cleanly
source/EXE/PDB/CDB/script identity drift
```

The future execution controller must stop the retry at the first such event. It must not edit the CDB and rerun within the same authorization.

## 9. T0/S7/T1/T2 evidence requirements

The authorized run is valid only if it records, without reconstruction:

```text
same process PID
T0 DQueue + target ptr/deleter/epoch/ticket/sequence publication
S7 engineThis and targetEpochDomain identity
candidate thread and invocation token
T1-B globalEpoch + ReaderSlot[0..63]
T1-B return RAX as minReaderEpoch
T1-C target ptr/deleter/epoch/ticket/expectedSequence
same-invocation R13 minReaderEpoch
CAS or epoch-blocked dequeue result
T2 final DQueue/domain/globalEpoch + ReaderSlot[0..63]
marker order T0 publication < T1_SELECTION < T2
```

If a required item is absent, the result is STOP/UNRESOLVED even if CDB exits with code 0.

## 10. Single-execution authorization

Authorization conditions for the future run:

```text
RETRY-2 attempt count   = 1 maximum
CDB script              = exact SHA-256 FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E
stimulus                = AudioEngineHarness.exe --fpm-m0
script edit during run  = forbidden
immediate rerun         = forbidden
M1/M2 fallback          = forbidden
source/test/CMake edit  = forbidden
build                   = forbidden
Dr.Memory               = forbidden
```

The authorization is consumed by the first future invocation. If the run is attempted, no second invocation is authorized by this report regardless of the outcome.

## 11. Evidence-state boundary after execution

Before Retry-2:

```text
S7 DQueue epoch-gate equality = PROVEN
S7 reader identity            = UNRESOLVED
minReaderEpoch at relevant T1 = NOT_CAPTURED
Case A/B/C/D                  = NOT_PROVEN
```

These states may change only from direct Retry-2 evidence. Historical `minReaderEpoch=9`, T0/S7/T2 inactive slots, or global/head epoch equality may not be used to classify the relevant reclaim invocation.

Even after a valid Retry-2:

```text
S7_READER may remain UNRESOLVED if the identity/join gates fail
IMPLEMENTATION remains FORBIDDEN until an attribution verdict and separate authorization close the causal chain
```

## 12. Gate result

```text
P3-5-FPM-C3-T1-Capture-Redesign-3 = CLOSED
Identity                          = PASS
Static invariants                 = PASS
CDB parser compatibility          = PASS
Object provenance                 = PASS
Runtime stop contract             = PASS
RETRY-2                           = AUTHORIZED / EXACTLY ONE FUTURE EXECUTION
Retry-2 executed in this gate     = NO
S7_READER                         = UNRESOLVED
Case A/B/C/D                      = NOT_PROVEN
IMPLEMENTATION                    = FORBIDDEN
```
