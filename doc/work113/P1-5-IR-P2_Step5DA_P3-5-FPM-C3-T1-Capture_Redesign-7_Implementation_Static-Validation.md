# Redesign-7 Implementation + Static Validation S7-01..S7-19

## 1. Gate result

```text
Gate   = P3-5-FPM-C3-T1-Capture-Redesign-7-IMPLEMENTATION-STATIC-VALIDATION
.cdb created          = 1, Redesign-7
lines changed         = 1, line 25 only
payload               = 92 bytes, 5 command tokens, inserted verbatim from the Design Gate
guard                 = 166 bytes, taken byte-wise from the frozen Redesign-4
S7-01 .. S7-19        = 19 PASS / 0 FAIL
VERDICT = STATIC VALIDATION PASS

CDB execution = 0    harness execution = 0    build = 0
production source = 0    test = 0    CMake = 0
IMPLEMENTATION = CLOSED
RUNTIME AUTHORIZATION = NOT GRANTED
```

## 2. Artifact

```text
created   doc/work113/P1-5-IR-P2_Step5DA_…_Redesign-7_Operand-Capture.cdb
SHA-256   5AA98A592DD5B8745317A439AEC5BBC8B564A207D1447BC1E819504015E4C5F9
bytes     14,785      delta +73 from Redesign-6
encoding  ASCII, no BOM, 41 CRLF, 41 lines, trailing CRLF

parent    Redesign-6   8D64862031F24A5285EEAFC10DBCD00C4A94C4CFB635CD2E891D227AB5CC23A5   unchanged
frozen    Redesign-4   A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C   unchanged
frozen    Redesign-5   247B6A71752AC1986600A52A5D5F4EFC84DDACD1DC8A27203EDB78657C45B47F   unchanged
frozen    C3-Redesign-2 FBE32851F795AF900E6C46D6E38F25138E01217C2B52C0BDDE1F015029CFC39E unchanged
```

## 3. The produced line 25, in full

```text
bp AudioEngineHarness+0x1f9c550 ".echo C3_DG_T0E_HIT ; ? dd(@rcx+0x34000) ; .echo C3_DG_T0E_EQ1 ;
  ? @r9 ; .echo C3_DG_T0E_EQ2 ; .if (@$t0 == 0) { .if (@r9 == 9) { .if (dd(@rcx+0x34000) == 4) {
  .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }"
```

Decomposition, obtained by the unique-split method, not by inspection:

```text
payload  92 bytes, 5 command tokens
   '.echo C3_DG_T0E_HIT'
   '? dd(@rcx+0x34000)'
   '.echo C3_DG_T0E_EQ1'
   '? @r9'
   '.echo C3_DG_T0E_EQ2'
guard   166 bytes, equal to the frozen Redesign-4 guard
```

The payload was not shortened, reordered or extended. It is the Design Gate section 3 string
verbatim, and the five tokens above are the Design Gate section 3 command inventory, unchanged.

## 4. S7-01 to S7-19

| ID | result | check | observed |
|---|---|---|---|
| S7-01 | PASS | only line 25 differs from Redesign-6 | changed = [25] |
| S7-02 | PASS | guard byte-identical to the frozen Redesign-4 guard | 166 B, unique split found, whole-line exact reconstruction true |
| S7-03 | PASS | guard conjunct `@r9 == 9` unchanged | R4 = 4, R7 = 4 |
| S7-04 | PASS | guard conjunct `dd(@rcx+0x34000) == 4` unchanged | R4 = 1, R7 = 1 |
| S7-05 | PASS | guard conjunct `@$t0 == 0` unchanged | R4 = 2, R7 = 2 |
| S7-06 | PASS | breakpoint topology unchanged | 13 RVAs ordered identical |
| S7-07 | PASS | `$t0..$t19` untouched | payload register writes 0, guard writes 71 = 71, registers 0..19 |
| S7-08 | PASS | exactly two `?` commands | payload `?` commands = 2, file-wide = 2 |
| S7-09 | PASS | no forbidden construct introduced | every payload command is `.echo MARKER` or `? EXPR`, offending = none |
| S7-10 | PASS | only new tokens are `.echo` and the three markers | `C3_DG_T0E_HIT`, `C3_DG_T0E_EQ1`, `C3_DG_T0E_EQ2` |
| S7-11 | PASS | T0P / T0Q / T0S byte-identical to Redesign-6 | lines 26, 27, 28 compared |
| S7-12 | PASS | all other lines byte-identical to Redesign-6 | 40 lines compared |
| S7-13 | PASS | ReaderSlot windows | 256 terms, 64 distinct offsets |
| S7-14 | PASS | A2 `dd @$t1 L1` | 7 occurrences |
| S7-15 | PASS | `gc` count unchanged | 61 = 61 |
| S7-16 | PASS | `q` count unchanged | 1 = 1 |
| S7-17 | PASS | payload and frozen guard separated, then operand comparison | payload operands = `['@rcx+0x34000']`, new addresses = none, new constants = none |
| S7-18 | PASS | source, harness, build, CMake, `ConvoPeq.md` unchanged | seven SHA-256 comparisons, all equal; git delta 12, pre-existing |
| S7-19 | PASS | execution count 0 | no debugger process, no CDB log for this gate |

### 4.1 S7-17, done by separation rather than by string search

The owner required this not to rest on "it is in the guard so it is fine". It does not. The payload
and the frozen guard were separated first, by the unique-split method, and the operand sets were then
compared as sets:

```text
payload memory operands, extracted from the payload alone = ['@rcx+0x34000']    exactly 1
frozen guard memory operands                               = 3 distinct forms
payload memory operands NOT present in the guard          = none                 0
payload numeric literals NOT present in the guard         = none                 0
```

So the single memory operand the payload introduces is drawn from the guard's own vocabulary, and
the payload introduces no address expression and no numeric literal of its own. That is a property of
sets, not of a substring being present somewhere in the file.

### 4.2 S7-09, done per command token rather than by substring

A naive check for `dd` in the payload would fail, because the payload legitimately contains
`dd(@rcx+0x34000)` as the **operand** of a `?` command, not as a `dd` command. The check is therefore
performed on the five command tokens individually:

```text
each token is matched against  ^\.echo C3_DG_T0E_[A-Z0-9_]+$   or   ^\? 
and no token begins with a bare  dd / db / dq
```

All five tokens conform. This scoping distinction is recorded because it is the same class of error as
the C1 semicolon mis-specification at Step5CX, where a check was written against a convenient proxy
rather than against the property.

## 5. What this gate does not certify

```text
does NOT certify  that the '?' construct is safe in a breakpoint body
```

The Design Gate's position is preserved verbatim: Redesign-5 recorded seven consecutive `?` commands
executing correctly and the body then stopping, with an error report that named a command which had
demonstrably run. The mechanism was never determined. Two `?` instances sit inside the region observed
to work, but that is an empirical bound and not a guarantee.

Therefore a static PASS here says nothing about whether the run will complete. That remains the sole
purpose of the next authorized execution.

```text
D6-10 / D6-11 were PROVEN under the single-command payload of Redesign-6.
They are NOT re-proven here. The next execution re-tests them under the enlarged payload, which is
exactly what the Design Gate listed as the primary runtime criterion.
```

## 6. Constraints honored

```text
changed beyond line 25               = 0, S7-01 and S7-11 and S7-12
T0P / T0Q / T0S                       = byte-identical
non-T0E lines                         = byte-identical, 40 lines
guard edits                           = 0, byte-identical to the frozen Redesign-4
@r9 == 9 / enqueuePos == 4 / @$t0 == 0 = unchanged
$t0..$t19                             = untouched, zero register writes in the payload
ReaderSlot windows / A2               = untouched
new construct class                   = 1, '?'
new constant / new address / new inference = 0
topology 13 / gc 61 / q 1 / markers    = unchanged
canary                                = not added
CDB execution / harness execution / build = 0
source / test / CMake / ConvoPeq.md / Harness / cdb.exe = untouched
```

## 7. Frozen items, unchanged

```text
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A / B / C / D = UNRESOLVED
A1 separator = RUNTIME PROVEN
A2           = installed, accepted, execution count 0, globalEpoch value UNPROVEN
T0 gate cause = UNRESOLVED, bounded to 0 of 116 cycles, the failing conjunct not yet measured
IMPLEMENTATION = FORBIDDEN
```

## 8. Final state

```text
Redesign-7 Implementation + Static Validation
= CLOSED / STATIC VALIDATION PASS / 19 of 19

execution candidate
  doc/work113/P1-5-IR-P2_Step5DA_…_Redesign-7_Operand-Capture.cdb
  5AA98A592DD5B8745317A439AEC5BBC8B564A207D1447BC1E819504015E4C5F9
  14,785 bytes

CDB execution / harness execution / build = 0
RUNTIME AUTHORIZATION = NOT GRANTED
```

## 9. Runtime Authorization requested

One execution, using the form established at Step5CQ:

```text
tmp\cdb.exe -cf <Redesign-7> -logo <log> build\Release\AudioEngineHarness.exe --measurement=normal
```

Primary criterion, in the owner's order:

```text
R1  SCRIPT_BEGIN = 1, SCRIPT_END = 1, ZwTerminateProcess = 1, exit = 0, debugger errors = 0
    i.e. does continuation survive the addition of two '?' commands.
    This re-tests D6-10 and D6-11 under the enlarged payload.
```

Then the witness relationship, which is what makes a failure attributable:

```text
C3_DG_T0E_HIT   present, C3_DG_T0E_EQ1 absent  -> the first  '?' failed
C3_DG_T0E_EQ1   present, C3_DG_T0E_EQ2 absent  -> the second '?' failed
both witnesses present                        -> both '?' completed, the class is tolerance-proven
                                               at this position
no HIT at all                                 -> reachability changed
```

Then, only if both witnesses are present, the operand values for the whole hit window, collected with
their correspondence intact:

```text
? dd(@rcx+0x34000)   -> enqueuePos, conjunct 3
? @r9                -> conjunct 2
conjunct 1 is already established, not re-measured
```

Discrimination is then read per hit from those two values, not inferred. No outcome is pre-judged,
and no constant is proposed before the measurement.
