# P3-5-FPM-C3-T1-Capture — A1 + A2 Applied / Static Validation

## 1. Gate result

```text
Gate     = P3-5-FPM-C3-T1-Capture-A1-A2-Applied-Static-Validation
A1 separator fix  = applied, 85 of 85
A2 O1-new capture = applied, 7 of 7
7-site correction = retained, 7 of 7

V01..V20 = 20 PASS / 0 FAIL
VERDICT  = STATIC VALIDATION PASS

CDB execution         = 0
AudioEngineHarness    = 0
build                 = 0
Runtime Authorization = 0
source/test/CMake     = untouched
```

Execution stops here, as instructed. Runtime Authorization is **requested** as the next gate and has
**not** been granted.

## 2. Artifacts

```text
frozen source  doc/work113/P1-5-IR-P2_Step5BB_P3-5-FPM-C3-T1-Capture-Redesign-2.cdb
               FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E   unchanged
               NOT an execution candidate

superseded     doc/work113/P1-5-IR-P2_Step5CO_P3-5-FPM-C3-T1-Capture-Redesign-3_Corrected.cdb
               5E977694DE91273AA8126DF6D4A714E1CA103D0447E55EF4323DC936834E52A7
               NOT an execution candidate, per instruction

this gate      doc/work113/P1-5-IR-P2_Step5CP_P3-5-FPM-C3-T1-Capture-Redesign-4_Corrected.cdb
               A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C
               14,624 bytes   ASCII, no BOM, 41 CRLF, 41 lines, trailing LF
               the only execution candidate
```

## 3. Transforms, all three

Built from the frozen original, so the total change surface is provable from one provenance.

| step | transform | count |
|---|---|---|
| 1 | `@$t2+0xc0` → `@$t2+0x00` | 7, retained from the prior authorization |
| 2 | A1, `}` + whitespace + `.else`/`.if`/`.echo` → `}` + whitespace + `; ` + command | 85 |
| 3 | A2, insert `dd @$t1 L1;` before the first `dd @` of each T1-chain body | 7 |

A1 added no register, removed no `.else`, and altered no `gc` control flow. It changed one character
position per site, inserting a command separator.

A2 placement, one per T1-chain breakpoint, each verified to sit on a `;` or space boundary:

| line | RVA | marker |
|---|---|---|
| 30 | `0x1f9cfb0` | `C3_T1A_CANDIDATE` |
| 31 | `0x1f9cfe2` | `C3_T1B_GETMIN_CALL_PRE` |
| 32 | `0x1f9cfe5` | `C3_T1B_GETMIN_RETURN` |
| 33 | `0x1f9cd00` | `C3_T1C_DQUEUE_ENTRY` |
| 34 | `0x1f9cd59` | `C3_T1_SELECTION` |
| 35 | `0x1f9cd72` | `C3_T1_CAS_PRE` |
| 36 | `0x1f9ce04` | `C3_T1_CAS_SUCCEEDED` |

`T1B_GETMIN_CALL_PRE` and `T1B_GETMIN_RETURN` are the pair that brackets the `getMinReaderEpoch`
call. `globalEpoch` is now read in the same invocation that produces `minReaderEpoch`, which is what
the S7 Case C contract requires, and it costs no register budget.

## 4. The defect this gate caught in its own A2 execution

**V12 failed on first pass.** The A2 transform was written as

```text
'dd @' + [char]36 + '$t1 L1;'
```

which concatenates to `dd @` + `$` + `$t1 L1;`, producing the malformed token

```text
dd @$$t1 L1        7 occurrences
```

instead of the authorized

```text
dd @$t1 L1
```

`@$$t1` is not a valid WinDbg dereference. Left uncorrected, all seven T1 bodies would have failed to
parse and the single authorized execution would have been spent on a predictable null result, the
same failure mode as the previous gate.

Corrected at 7 sites, `@$$t1 L1;` → `@$t1 L1;`, and re-validated. This completes the already
authorized A2 change; it is not a scope expansion. No other transform was touched while fixing it.

### 4.1 Why the first V12 check did not catch it

Three defects compounded, and all three are recorded because they are the kind that silently
manufactures a false PASS.

```text
1  the transform concatenated 'dd @' + $ + '$t1 L1;'  -> produced @$$
2  the V12 and V12b checks built their SEARCH string the same way, so they
   searched for '@$$t1 L1', matched the defect, and reported 7 -> the check
   validated the bug against itself
3  the earlier V17 line reported 'poi = 0' because the search string had been
   concatenated into '@$pc dwo poi', a literal that occurs nowhere
```

Every affected check was re-run with independent single-quoted literals. `$` is already literal in
PowerShell single-quoted strings, so no character-code concatenation is needed at all.

A fourth defect, in the same family, appeared during the re-run:

```text
4  a backtick inside a PowerShell SINGLE-quoted string is a literal character,
   not an escape. '`$t1+0x20' searched for a backtick and returned 0 for
   windows that do exist. The window series then appeared to be missing
   entirely. Removed the backticks; 256 confirmed.
```

Defect 4 is why the window census is reported here from a clean run rather than from the run that
first appeared to show zero windows.

## 5. V01–V20

| # | check | observed | result |
|---|---|---|---|
| V01 | corrected script SHA and bytes frozen | `A935369D…022C`, 14,624 | PASS |
| V02 | `ConvoPeq.md` SHA and bytes | `E5E74200…F3609`, 5,535,334 | PASS |
| V03 | `AudioEngineHarness.exe` SHA | `E5C7AFB9…34C75` | PASS |
| V04 | `cdb.exe` SHA and version | `5F54ABAF…FBEE67`, `10.0.29617.1000` | PASS |
| V05 | breakpoint RVA set, ordered | 13 = 13, identical | PASS |
| V06 | marker set, ordered | 64 = 64, identical | PASS |
| V07 | `0xC0` remaining in any `@$t2` expression | 0, and `$t2+0x00` = 7 | PASS |
| V08 | ReaderSlot window terms, anchor `$t1 + 0x20` | 256 = 64 × 4, unchanged | PASS, see 5.1 |
| V09 | corrected entry form at both guards | present at `0x1f9cd59` and `0x1f9cd72`, 2 total | PASS |
| V10 | writer sequence, ordered | 71 = 71, identical | PASS |
| V11 | register set | `$t0..$t19`, max 19, new registers 0 | PASS |
| V12 | O1-new `dd @$t1 L1` | 7 correct, 0 malformed, 0 bad boundaries | PASS after 4 |
| V13 | `0x34000` never used as `globalEpoch` | 0 | PASS |
| V14 | brace balance per body | 0 unbalanced | PASS |
| V15 | inter-block separator | 0 space-separated, 85 `} ;` | PASS |
| V16 | no resume inside a non-resuming body | no such body | PASS, rescoped, see 5.2 |
| V17 | `@$pc`, `dwo`, `poi` | 0, 0, 7 read-only | PASS, rescoped, see 5.3 |
| V18 | existing STOP markers retained | 11 `C3_CR1_STOP_*` plus 3 named | PASS |
| V19 | source, test, CMake, build untouched | SHA unchanged, 12 pre-existing entries | PASS |
| V20 | 64 windows anchored at `$t1 + 0x20` | 256 total across 4 sites | PASS |

### 5.1 V08, the corrected condition, and an independent corroboration

The owner directed that the previous gate's literal `0xC0 = 64` condition not be reinstated. It is not
reinstated. The condition enforced here is the real one:

```text
'@$t1+0x20' terms                 = 256    (64 slots x 4 dump sites, unchanged)
'db @$t1+0x20+' window terms      = 256
distinct window offsets           = 64
base offset                       = 0x0
last offset                       = 0x13b0  = 5040 = 63 x 80
strides                           = 0, 80, 160, ... 5040   single stride 0x50
0xC0 (= 192) among the offsets    = absent, and necessarily so, 192 not a multiple of 80
distribution                      = 64 at each of 0x1f9c5da, 0x1fa80b3, 0x1f9cfe2, 0x1fa80b2
```

The window series spans `0x20` through `0x20 + 0x1400 = 0x1420` relative to `$t1`. U3, at runtime and
independently, measured `readers = this + 0x20 .. this + 0x1420`. The static geometry of the script
and the runtime measurement of the object now agree, from two unrelated methods. This is a
corroboration, not a proof of either, and it is recorded as such.

### 5.2 V16 rescoped

The drafted criterion "no `.else`" is wrong for this script. `.else` and `gc`, 85 and 61 occurrences,
are the intended control flow of a conditional breakpoint, which must resume on a non-matching hit;
earlier probes in this workstream failed precisely for want of that resume path. V16 is rescoped to
"no resume command inside a body that must not resume", and there is no such body. The separator
defect that `.else` participates in is carried by V15, which now passes.

### 5.3 V17 rescoped, with the corrected count

```text
'@$pc' = 0     'dwo' = 0     as intended
'poi'  = 7     not 0, and the earlier 0 was defect 3 in section 4.1
```

All 7 are pre-existing read-only dereferences inside `.if` predicates, none introduced by this gate:

```text
line 25  poi(@rsp)                 return address
line 34  poi(@rsi) x2              T1_SELECTION reader-identity predicate
line 34  poi(@rsi+0x8)  x2
line 34  poi(@rsi+0x10) x2
```

`poi()` used as a write destination = 0. A bare `poi()` is a read-only dereference and does not
perturb the measurement, so it is permitted and the criterion is rescoped accordingly. `@$pc` and
`dwo`, the two constructs that could displace the program counter or break stepping, remain at 0.

## 6. Arithmetic reconciliation

Every count change accounted for, original → corrected:

```text
'0xc0'                       7 ->  0    7 sites corrected, no residue
'$t2+0x00'                   0 ->  7    the 7 corrections
'dd @$t1 L1'                 0 ->  7    7 A2 insertions, 0 malformed
'dd @$$t1 L1'                0 ->  0    7 produced then 7 corrected, see section 4
'dd'                        44 -> 51    44 pre-existing + 7 A2 insertions
'dq'                        12 -> 12    unchanged
'$t1+0x20'                 256 -> 256   unchanged
'} ;' family '}ws;'          1 -> 86    1 pre-existing '};' on line 30 + 85 A1
'} ;' spaced form            0 -> 85    the 85 A1 separators
'} + space + command'       85 ->  0    all 85 eliminated
'@$t2+0x34000'               6 ->  6    unchanged
lines                       41 -> 41    unchanged
CRLF                        41 -> 41    unchanged
bytes                   14,377 -> 14,624
```

The pre-existing `};` on line 30 sits inside `.if (@$t13 == 0) { r $t13=1 };`, a single-statement
block whose closing brace is already followed by a semicolon. The A1 regex requires whitespace
followed by `.else`, `.if` or `.echo`, so it did not match and did not touch it. The two families are
disjoint, which is why the spaced form is 85 and the family total is 86. This was checked rather than
assumed, because 85 + 1 does not equal the observed spaced count of 85 by accident alone.

## 7. Byte-level confirmation

```text
                   original    corrected
bytes                  14,377         14,624    +247
BOM                       none           none
non-ASCII                   0              0
CRLF pairs                 41             41
bare LF                     0              0
lines                      41             41
trailing LF               yes            yes
```

## 8. Constraints honored

```text
changes outside the 3 authorized transforms = 0
                                          verified by V05, V06, V10, V11 and section 6
$t1 modified                              = 0
$t1 + 0x20 modified                       = 0, 256 terms unchanged, V08 and V20
$t2 modified                              = 0
+0x30000 modified                         = 0
+0x34000 modified                         = 0
+0x34040 modified                         = 0
ReaderSlot windows modified               = 0
breakpoint RVAs modified                  = 0, V05
markers modified                          = 0, V06
$t0..$t19 writer set modified             = 0, V10
pseudo-register added                     = 0, V11
$7, $7e, $7q, $7rbx introduced           = 0
C3-Redesign-2.cdb modified                = 0, SHA unchanged
CDB execution                             = 0
AudioEngineHarness execution              = 0
build                                     = 0
Runtime Authorization                     = 0
source / test / CMakeLists / build.bat    = untouched, 12 pre-existing entries
```

## 9. Base state, unchanged

```text
U3 = SUCCESS    S-14-A = PROVEN    S-14-B = PROVEN
Candidate A = PROVEN    B1 = PROVEN    @$bp0 = PROVEN    $t11 = PROVEN
STOP-C3-T1-11 = CLEARED    STOP-C3-T1-12 = CLEARED

S7_READER          = UNRESOLVED, frozen for an execution result
S7_READER_SLOT     = UNRESOLVED, frozen for an execution result
minReaderEpoch(T1) = NOT CAPTURED, frozen for an execution result
Case A / B / C / D = NOT PROVEN, frozen for an execution result
IMPLEMENTATION     = FORBIDDEN
```

None of these advanced. Creating a script and validating it statically cannot advance them.

## 10. Final state

```text
A1 + A2 applied, Static Validation re-run
= CLOSED / STATIC VALIDATION PASS / 20 of 20

execution candidate
  doc/work113/P1-5-IR-P2_Step5CP_P3-5-FPM-C3-T1-Capture-Redesign-4_Corrected.cdb
  A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C
  14,624 bytes

V12 failed once on a defect in this gate's own A2 transform, a doubled `$`
producing `dd @$$t1 L1`, was corrected at 7 sites, and was re-validated.
Four harness defects in the validation tooling itself are recorded in 4.1.

CDB execution / harness execution / build = 0
Runtime Authorization = REQUESTED, NOT GRANTED
Execution of the C3 capture = BLOCKED pending authorization
```

## 11. Next gate requested

```text
Runtime Authorization
        |
        v
single execution of Redesign-4 against build/Release/AudioEngineHarness.exe --measurement=normal
        |
        v
Result Audit for T1, S7_READER, S7_READER_SLOT, minReaderEpoch, Case A / B / C / D
```

Static validation establishes that the script is internally consistent, that its address algebra
matches the U3 measurement, and that it no longer carries the parser-breaking construct. It does not
establish that the script runs. That remains unproven until one authorized execution says so, and
this gate performed none.
