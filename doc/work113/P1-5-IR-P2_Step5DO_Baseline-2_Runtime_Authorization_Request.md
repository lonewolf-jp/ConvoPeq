# Baseline-2 Runtime Authorization — Request

```text
gate                 = Runtime Authorization REQUEST
status               = NOT GRANTED.  Awaiting an explicit owner GRANT.
requesting           = the agent
grant must come from = the owner
CDB execution        = 0 for Baseline-2
artifact             = Baseline-2, 14,968 B
```

## 0. Why this document does not grant anything

The agent does not issue its own execution authorisation. In this lineage the requesting party and
the granting party have been kept distinct on purpose: the agent designs, implements and validates,
and the owner authorises the single execution. An agent that granted itself permission to run the
debuggee would make every `NOT AUTHORIZED` state in the record decorative.

```text
this document   = the REQUEST, fixing the exact scope that would be granted
the grant       = must be an explicit owner declaration
```

To proceed, the owner sends:

```text
Baseline-2 Runtime exactly 1 run / Retry=0 を認可する
```

Nothing else is required. No further analysis, no further design.

## 1. Fixed scope that would be granted

```text
artifact
  Baseline-2
  SHA-256  BD5129A3655FA04FB1EAF8539BA596BC47179FABD8817B85F3742216F083B763
  size     14,968 bytes

parent, frozen and unmodified
  Baseline-1
  SHA-256  74E8C64CD346BA981A1ABD2E1A941B2EDB31E1932A7D586C7D683C1E6EBF9142
  size     14,895 bytes

runtime
  execution_count  exactly 1
  Retry            0
```

```text
invocation   cdb.exe -cf <target resolved by digest> -logo <log derived from the target>
             build\Release\AudioEngineHarness.exe --measurement=normal
  -logo only, never -o
  no filename hand-typed; both paths resolved from a directory listing
```

## 2. R1, judged alone and first

```text
SCRIPT_BEGIN       = 1
SCRIPT_END         = 1
ZwTerminateProcess = 1
quit               = 1
exit code          = 0
debugger errors    = 0
```

Error census required empty of `Syntax error`, `Extra character`, `Illegal`, `Invalid`,
`Undefined`, `^Error`, `Cannot`, `Memory access error`, `Unable to read memory`, `Couldn't`,
`Evaluate expression:`. `Unable` from extension DLL loading is classified benign and excluded from
the census total, not subtracted from it.

```text
R1 FAIL -> attribution from the log
       -> Result Audit closed
       -> R2 to R6 NOT evaluated
       -> Retry NOT performed
       -> Baseline-2 NOT edited
```

## 3. R2 to R6, conditional on R1 PASS

### R2, T0E hit count
`C3_DG_T0E_HIT` whole-line occurrences.

### R3, `r9` raw distribution
Extract `r9=` from the `r` output. These are **enqueue epoch** values. They are **not** `enqueuePos`.

### R4, attribution to the hit
Each `r9` reading must be attributed to its own T0E block by ordinal position:

```text
C3_DG_T0E_HIT
  -> R9 verdict
  -> EQ verdict
  -> r output            (about 8 lines)
  -> C3_DG_T0E_EP0
  -> the 11-qword window
  -> C3_DG_T0E_WIN_END
  -> frozen guard
```

A simple total is not sufficient. The count of `r` output blocks must correspond one to one with the
T0E hit count, and each block must contain its own EP0 and WIN_END witnesses.

### R5, extension, per hit
At minimum these four window slots, recorded per hit:

```text
window + 0x18   =  [rcx - 0x1438]
window + 0x20   =  [rcx - 0x1430]
window + 0x28   =  [rcx - 0x1428]
window + 0x30   =  [rcx - 0x1420]
```

and, evaluated separately:

```text
r9 == window + 0x18 ?
r9 == window + 0x28 ?
r9 == 9 ?
```

plus, across hits, which window slot tracks `r9`.

```text
DISCIPLINE, fixed in advance
  a slot that tracks r9 is a CANDIDATE, not an identification.
  calling it 'globalEpoch' requires the identification criterion to be met, namely that the
  slot tracks r9 across hits AND advances with the world epoch observed in the harness output.
  the recorded offset algebra is self-contradictory by 0x10, so the identification is
  deliberately left to this run's data.  See Step5DM sections 1.1 to 1.3.
```

### R6, T0 sequence
`T0E -> T0P -> T0Q -> T0S` ordering, verified from this log rather than assumed.

## 4. After the run, the following remain forbidden

```text
changing '@r9 == 9' on the strength of the capture alone
any T1 repair
  0x1f9ce04 repair
  0x1f9cfb0 repair
source / test / CMake / build / harness changes
Dr.Memory
M1 / M2
any retry
editing Baseline-1 or Baseline-2
```

The order is fixed in advance and is not compressed:

```text
Baseline-2 Runtime
      |
      v
Capture Result
      |
      v
Intent / Semantic Audit
      |
      v
predicate validity
      |
      v
Repair Design Gate
      |
      v
separate Implementation Authorization
```

## 5. What the capture may not conclude, whatever it shows

```text
NOT  that 9 is wrong.  The run measures relations, not intent.
NOT  that a tracking slot IS globalEpoch, unless the R5 identification criterion is met.
NOT  any raw enqueuePos.  This gate captures epoch.
NOT  that the captured r9 distribution is invariant.  One configuration, one run.
NOT  anything about T1.  0x1f9ce04 and 0x1f9cfb0 remain malformed and were not reached.
```

The three prior `r9` observations stay separate and are not merged into an invariant:

```text
Redesign-5        1          one sample, run did not complete
Step5BB Retry-1   9          one sample, gate opened, --fpm-m0
Step5DK           not 9      116 of 116 hits, --measurement=normal
```

## 6. Pre-flight, read-only, verified at the time of this request

| item | value | result |
|---|---|---|
| Baseline-2 resolved by digest | unique, 14,968 B | match |
| Baseline-1 resolved by digest | unique, 14,895 B | match, unmodified |
| log path, derived from the target | `…_Baseline-2_Runtime_CDB.log` | **does not exist** |
| `AudioEngineHarness.exe` | `E5C7AFB9…34C75` | match |
| `ConvoPeq.md` | `E5E74200…F3609` | match |
| `cdb.exe` | `5F54ABAF…FBEE67` | match |
| git `src/` `CMakeLists.txt` `build.bat` | 12 | all pre-existing |
| residue processes | 0 | clean |
| `.cdb` corpus | 15 | Baseline-2 is the newest |

The log not existing is what makes "exactly 1 execution" auditable after the fact. If a log with
this name is ever present before a run, the run is not unambiguous and must not proceed.

## 7. State

```text
Baseline-1      FROZEN, unmodified
Baseline-2      implemented, Static Validation 12 of 12 PASS
Implementation  COMPLETE
Static          PASS 12/12
Runtime         NOT AUTHORIZED, requested here
CDB execution   0 for Baseline-2
T0 repair       FROZEN, not revisited
T1 repair       NOT AUTHORIZED
Retry           FORBIDDEN
source / test / CMake / build / harness   untouched
```

Awaiting the owner's explicit grant.
