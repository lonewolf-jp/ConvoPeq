# C3 Entry-Gate Diagnostic — Implementation + Static Validation

## 1. Gate result

```text
Gate    = P3-5-FPM-C3-T1-Capture-C3-Entry-Gate-Diagnostic-IMPLEMENTATION-STATIC-VALIDATION
.cdb created            = 1, Redesign-5
preambles inserted      = 4 of 4
existing T0 guards      = byte-identical, proven by unique-split-point reconstruction
S00..S16 + V01..V20     = 32 PASS / 0 FAIL
VERDICT = STATIC VALIDATION PASS

CDB execution          = 0
harness execution      = 0
build                  = 0
Runtime Authorization  = 0, NOT GRANTED
```

## 2. Artifacts

```text
source (frozen)  doc/work113/P1-5-IR-P2_Step5CP_P3-5-FPM-C3-T1-Capture-Redesign-4_Corrected.cdb
                 A935369D711535C9837DFD25761B7886F6CB49D4157534BA34446DBB15F1022C   unchanged
                 14,624 bytes

created          doc/work113/P1-5-IR-P2_Step5CT_P3-5-FPM-C3-T1-Capture-Redesign-5_EntryGate-Diagnostic.cdb
                 247B6A71752AC1986600A52A5D5F4EFC84DDACD1DC8A27203EDB78657C45B47F
                 17,427 bytes, delta +2,803
                 ASCII, no BOM, 41 CRLF, 41 lines, trailing CRLF
```

## 3. Change surface

Exactly four lines differ from Redesign-4, proven by whole-file line comparison:

```text
changed line indices (0-based) = [24, 25, 26, 27]     i.e. lines 25, 26, 27, 28
every other line               = byte-identical
```

| line | RVA | marker | guard bytes | preamble bytes | split point |
|---|---|---|---|---|---|
| 25 | `0x1f9c550` | `C3_T0_ENTRY` | 166 | 540 | 540 |
| 26 | `0x1f9c57d` | `C3_T0_CAS_PRE` | 229 | 661 | 661 |
| 27 | `0x1f9c59a` | `C3_T0_CAS_POST` | 230 | 661 | 661 |
| 28 | `0x1f9c5da` | `C3_T0_SEQUENCE_PUBLISHED` | 2,207 | 929 | 929 |

Insertion form, per line:

```text
bp AudioEngineHarness+0x<rva> "<preamble> ; <existing guard, byte-identical>"
```

## 4. Guard identity, proved structurally and not by search

The Design made byte-level identity a central safety condition and forbade relying on string search.
It is proved as follows.

The preamble itself contains `" ; "`, so the first occurrence is not the boundary. For each of the
four lines the boundary is established as the **unique** index `j` such that

```text
body[j:j+3] == " ; "     and     body[j+3:] == original guard, byte for byte
```

Uniqueness is required, and was obtained for all four. The preamble is `body[:j]`. Two independent
equalities are then checked:

```text
S03   the guard slice in the new line equals the original guard exactly
S03b  new_line == original_prefix + preamble + " ; " + original_guard + original_suffix
```

`S03b` is an exact reconstruction of the entire line, so it subsumes prefix and suffix identity as
well. A substring search would have passed even if the preamble had corrupted the guard's interior;
this cannot, because the whole line must reassemble.

Independent corroboration from the apply step, which computed the guard slice by **length-anchored
offset** rather than by pattern: `guardIdentical = True` and `reconExact = True` for all four.

## 5. Preamble form

Registers used: `$t16`, `$t17` only. `$t0` untouched. No register added.

```text
r $t17 = @$t17 + 1 ; .if (@$t17 <= 8) { r $t16 = @r9 ; .echo C3_DG_<tag>_HIT ; <operand dumps> ; <per-conjunct verdict echoes> } ; .if (@r9 != @$t16) { r $t16 = @r9 ; .echo C3_DG_<tag>_R9CHANGE ; ? @r9 ; ? dd(<base>+0x34000) }
```

Per-gate tags and operand bases, following the Design section 4.2 correction:

| tag | RVA | memory base used | why |
|---|---|---|---|
| `C3_DG_T0E` | `0x1f9c550` | `@rcx` only | at this site `@r10`, `@r11`, `@rbx` are still caller values, so no address may be formed from them |
| `C3_DG_T0P` | `0x1f9c57d` | `@r11` | queue base established at `0x1f9c55c` |
| `C3_DG_T0Q` | `0x1f9c59a` | `@r11` | queue base |
| `C3_DG_T0S` | `0x1f9c5da` | `@r11` | queue base; sequences indexed by `@rbx`, not by `@r10`, because `incl` already ran |

## 6. Marker inventory, 44 additions, S08

```text
T0E   8   HIT  R9CHANGE  V0_T0  V1_R9  V2_DQ            each PASS and FAIL
T0P  10   HIT  R9CHANGE  V0_R9  V1_ENQ V2_DQ  V3_SEQ
T0Q  10   HIT  R9CHANGE  V0_R9  V1_ENQ V2_DQ  V3_SEQ
T0S  16   HIT  R9CHANGE  V0_T0  V1_R9  V2_ENQ V3_SLOT V4_DQ V5_SEQ V6_DQNZ
     ---
     44   expected 44, matched
```

Pre-existing markers: 64 before, 64 after excluding `C3_DG_*`, sequence identical.

## 7. The `@r9 == 9` prohibition

The Design recorded that the disassembly supplies no basis for the value 9, and the authorization
forbade changing, relaxing or removing it. Neither was done.

```text
'@r9 != 9'  anywhere in the file = 0        nothing inverts, negates or bypasses the predicate
value 9                            = untouched
```

Occurrence counts of the guard predicates rise, and the rise must not be misread:

```text
'@r9 == 9'   4 -> 8
'@r10 == 4'  1 -> 2
'@r10 == 5'  2 -> 4
'@rbx == 4'  1 -> 2
```

Each addition is a **read-only verdict echo** inside the preamble,

```text
.if (@r9 == 9) { .echo C3_DG_T0E_V1_R9_PASS } ; .else { .echo C3_DG_T0E_V1_R9_FAIL }
```

The guard occurrences themselves are byte-identical, proved in section 4. The doubling is the
diagnostic reporting the same predicate, which is its entire purpose. No guard was relaxed.

## 8. Full validation table

| ID | result | check | observed |
|---|---|---|---|
| S00 | PASS | only the 4 T0 lines differ from Redesign-4 | indices [24,25,26,27] |
| S01 | PASS | preamble sentinel present exactly 4 times | 4 |
| S02 | PASS | each RVA carries its own preamble and tag | T0E=8 T0P=10 T0Q=10 T0S=16 |
| S03 | PASS | each T0 guard byte-identical, unique split proven | 166/229/230/2207 B |
| S03b | PASS | each T0 line an exact prefix+preamble+guard+suffix reconstruction | 4 of 4 |
| S04 | PASS | `$t0..$t19` set unchanged | 0..19, no addition |
| S05 | PASS | `$t16`/`$t17` writes added only inside preambles | old 6, new 18, delta 12, all 12 inside |
| S06 | PASS | 256 ReaderSlot window terms unchanged | 256, 64 distinct, base `0x0`, last `0x13b0` |
| S07 | PASS | pre-existing marker sequence unchanged | 64 = 64, ordered |
| S08 | PASS | `C3_DG_*` count matches design | 44 = 44 |
| S09 | PASS | `q` count unchanged | 1 = 1 |
| S10 | PASS | `gc` count unchanged | 61 = 61 |
| S11 | PASS | breakpoint RVA topology unchanged | 13 = 13 |
| S12 | PASS | T1/T2 chain bodies byte-identical | lines 29..37 |
| S13 | PASS | `$t2+0xc0` residue | 0 |
| S14 | PASS | `dd @$t1 L1` | 7 |
| S15 | PASS | A1 separator rule, no space-separated close | space-separated 0; `} ;` 86 -> 126 |
| S16 | PASS | source, test, CMake, build untouched | see section 9 |
| V01 | PASS | Redesign-5 SHA and bytes frozen | `247B6A71…B47F`, 17,427 |
| V05 | PASS | breakpoint RVA set ordered identical | 13 |
| V07 | PASS | `0xC0` residue | 0 |
| V08 | PASS | windows 256 / 64 slots at `$t1+0x20` | 256, 64 |
| V09 | PASS | corrected entry form at both guards | 2 |
| V10 | PASS | writer sequence with preambles excised equals original | 91 = 91, delta 0, preamble writes 12 |
| V11 | PASS | no new pseudo-register | `$t0..$t19` |
| V12 | PASS | O1-new `dd @$t1 L1` | 7 |
| V13 | PASS | `0x34000` never used as `globalEpoch` | 0 |
| V14 | PASS | brace balance per body | 0 unbalanced |
| V16 | PASS | rescoped, no resume inside a non-resuming body | by design |
| V17 | PASS | `@$pc` 0, `dwo` 0, `poi` read-only | 0 / 0 / 7, write dest 0 |
| V18 | PASS | STOP markers retained | 11 + 3 |
| V20 | PASS | windows anchored at `$t1+0x20` | 256 |
| E01 | PASS | ASCII, no BOM, CRLF 41, trailing CRLF | preserved |

**V10 is the register-budget proof.** With each preamble excised, the writer sequence is identical
to the original, 91 writes, delta 0. The 12 added writes are all inside preambles, 3 per gate:
one `$t17` increment and two `$t16` assignments. Since `T0_SEQUENCE_PUBLISHED` assigns `$t16` and
`$t17` itself, both diagnostic registers are wiped the moment the T0 chain opens.

## 9. Two tooling defects during this gate, recorded

Neither changed the artifact; both produced false results that had to be corrected.

```text
DEFECT 1  the first apply attempt used a PowerShell hashtable that was treated as a String,
          so only 1 of 4 lines was processed and the output file was truncated to 1,154 bytes.
          The bad file was deleted, Redesign-4 verified intact, and the transform was redone
          in Python, which has no PowerShell interpolation or quoting hazard.
          Caught because the byte count was sanity-checked before proceeding.

DEFECT 2  the first validation run reported S02, S03, S03b, S05 and V10 as FAIL. All five had a
          single cause, in the validator not the script: the preamble and guard boundary was taken
          as the FIRST occurrence of " ; ", but the preamble itself contains " ; ".
          Fixed by deriving the boundary as the unique index j where
          body[j:j+3] == " ; " and body[j+3:] equals the original guard, and requiring uniqueness.
          After the fix all 32 pass.
          The apply step had already proved guard identity independently by length-anchored
          offset, so the FAIL verdicts were contradicted by existing evidence before the
          validator was touched.
```

This is the same family as the earlier `$d` concatenation defect, the self-validating search
string, and the backtick-in-single-quotes defect. The pattern is consistent: harness string
handling, not the artifact under test. The guard against it is the same each time, byte counts and
an independently derived cross-check before trusting a PASS or a FAIL.

## 10. Constraints honored

```text
preamble added to 4 T0 bodies only          = 4, proven by S00
pseudo-registers used                       = $t16, $t17 only
$t0                                         = untouched
$t0..$t15, $t18, $t19                       = untouched, S04 and S10
existing T0 guard predicate                 = byte-identical, S03 and S03b
existing gc                                 = unchanged count 61, S10
q                                           = unchanged count 1, S09
T1 correlation C1..C13                      = untouched, S12
ReaderSlot windows / @$t1+0x20              = untouched, S06 and V20
DQueue layout 0x30000 / 0x34000 / 0x34040   = untouched
breakpoint RVA topology                     = unchanged, 13, S11
marker topology                             = 64 pre-existing unchanged, S07
@r9 == 9                                    = value untouched, not relaxed, not removed
ConvoPeq.md / production source / test / CMake / build / Harness = untouched
Design section 5.7 canary breakpoint         = NOT added; it would add a breakpoint and a marker
                                              outside the authorized scope. Omitted deliberately.
```

The canary is the one Design element not implemented. The authorization permitted only preamble
insertion into the four T0 bodies and only the four `C3_DG_*` marker classes, while a canary would
require a fifteenth breakpoint and a marker outside those classes. Per-site reachability does not
depend on it, because each site's own `..._HIT` marker is the first statement of its body.

## 11. Frozen items, unchanged

```text
S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A / B / C / D = UNRESOLVED
A1 separator  = RUNTIME PROVEN
A2            = installed, CDB accepted, execution count 0, globalEpoch value UNPROVEN
T0 gate reason = UNRESOLVED
IMPLEMENTATION = FORBIDDEN
```

Creating a script and validating it statically cannot advance any of these.

## 12. Final state

```text
C3 Entry-Gate Diagnostic, Implementation + Static Validation
= CLOSED / STATIC VALIDATION PASS / 32 of 32

execution candidate
  doc/work113/P1-5-IR-P2_Step5CT_…_Redesign-5_EntryGate-Diagnostic.cdb
  247B6A71752AC1986600A52A5D5F4EFC84DDACD1DC8A27203EDB78657C45B47F
  17,427 bytes

CDB execution / harness execution / build = 0
Runtime Authorization = NOT GRANTED
```

## 13. Runtime Authorization requested

One execution of Redesign-5 against `build/Release/AudioEngineHarness.exe --measurement=normal`,
using the invocation form established at Step5CQ:

```text
tmp\cdb.exe -cf <script> -logo <log> <exe> --measurement=normal
```

The question this run is authorized to answer is narrower than the previous one. It is not
"capture the T1 correlation". It is, in the owner's order:

```text
were the four T0 sites reached at all
which operand or conjunct was false where they were reached
was the T0 entry condition ever actually observed true
```

Only after that is known can the reason Redesign-4 produced no T1 evidence be adjudicated. Nothing
in this gate predicts the answer, and no value is assumed: `@r9 == 9` is reported, never asserted.
