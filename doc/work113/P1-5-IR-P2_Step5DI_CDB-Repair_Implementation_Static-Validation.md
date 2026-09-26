# Step5DI — CDB Repair Implementation and Static Validation

```text
gate                 = CDB repair, scope A' (4 T0 sites), operator dwo
authorization        = Step5DH, scoped
parent               = P1-5-IR-P2_Step5DD_…_Redesign-9_Conjunct-Verdicts.cdb
parent SHA-256       = D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14
parent bytes         = 14,887
product              = P1-5-IR-P2_Step5DI_P3-5-FPM-C3-T1-Capture_T0-Guard-Repair-Baseline-1.cdb
product SHA-256      = 74E8C64CD346BA981A1ABD2E1A941B2EDB31E1932A7D586C7D683C1E6EBF9142
product bytes        = 14,895      delta +8
CDB execution        = 0
Static Validation    = 17 of 17 PASS
baseline             = T0-Guard-Repair-Baseline-1, CONFIRMED
DG-T0-Repair-3b      = STILL OPEN
```

## 1. Countermeasure applied

The parent path was **not** hand-transcribed. The generation script enumerated every `.cdb` in
`doc/work113`, hashed each one, and selected the file whose digest equals the authorized value. It
stops unless exactly one file matches. It then prints the name it resolved.

```text
resolved  P1-5-IR-P2_Step5DD_P3-5-FPM-C3-T1-Capture-Redesign-9_Conjunct-Verdicts.cdb
digest    D48665B8E8B4DF780B6B174640EA869FFF6E2FF0F951D1799C1589809F5DAD14   match
bytes     14,887   match
```

The validation script resolves both artifacts the same way, and identifies the product **by name
from the listing**, never by content.

### 1.1 Two deviations from the countermeasure, both disclosed

**First.** To delete the malformed first product I hand-typed its filename. That is the practice the
countermeasure forbids. The risk was low, the file having been created seconds earlier by my own
script, but the commitment was absolute and I broke it once.

**Second, and worse.** The closing integrity check of this gate built a `FROZEN` dict whose *keys
were hand-typed artifact filenames*, with the same underscore-for-hyphen error. All five baselines
printed as unverified. The digests were visibly correct in the output, but eyeballing a truncated
SHA prefix is not verification, and this was the very check meant to close the gate. It is a **third
occurrence** of the same defect, after Step5DE and Step5DH.

```text
re-verified by pure content lookup, with no filename used as a key:
  for each expected digest, assert exactly one file in the directory carries it.
  6 of 6 match, including the new baseline.  14 files, 14 distinct digests, no duplicates.
```

The lesson is now stated three times and the third statement is the operative one:

```text
Never key a check on a hand-typed artifact filename.  Key it on content.
A long artifact name in this corpus mixes '_' and '-' at exactly the segment a human
retypes, and it has now produced three false failures in a row, each of which looked
like a corrupted frozen baseline.
```

## 2. The repair as executed

Scope was measured on the parent before anything was written, and asserted to be exactly 8
occurrences, 2 per site, at exactly the 4 target sites.

```text
line 25  RVA 1f9c550   REPAIR     in .if = 2   outside .if = 0
line 26  RVA 1f9c57d   REPAIR     in .if = 2   outside .if = 0
line 27  RVA 1f9c59a   REPAIR     in .if = 2   outside .if = 0
line 28  RVA 1f9c5da   REPAIR     in .if = 2   outside .if = 0
line 30  RVA 1f9cfb0   EXCLUDED   in .if = 0   outside .if = 2
line 36  RVA 1f9ce04   EXCLUDED   in .if = 2   outside .if = 0
```

The eight repaired conditions, read back from the product:

```text
0x1f9c550   .if (dwo(@rcx+0x34000) == 4)      payload, diagnostic
0x1f9c550   .if (dwo(@rcx+0x34000) == 4)      guard, conjunct 3
0x1f9c57d   .if (dwo(@r11+0x34000) == 4)
0x1f9c57d   .if (dwo(@r11+0x30010) == 4)
0x1f9c59a   .if (dwo(@r11+0x34000) == 5)
0x1f9c59a   .if (dwo(@r11+0x30010) == 4)
0x1f9c5da   .if (dwo(@r11+0x34000) == 5)
0x1f9c5da   .if (dwo(@r11+0x30010) == 5)
```

Exclusions confirmed untouched, `dwo(` count 0 at both:

```text
0x1f9ce04   dd( x2, unchanged
0x1f9cfb0   dd( x2, unchanged, still address expressions
            dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
            dq (@$t2+0x00+((dd(@$t2+0x34040)&0xfff)*0x30)) L6
```

The T0E body in full, structure visibly preserved:

```text
.echo C3_DG_T0E_HIT ; .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ; .else { .echo C3_DG_T0E_R9_FAIL } ;
.if (dwo(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo C3_DG_T0E_EQ_FAIL } ;
.if (@$t0 == 0) { .if (@r9 == 9) { .if (dwo(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1;
ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```

## 3. Three defects, all found before any validation result was accepted

None of these touched the final product's bytes. The product SHA is the same before and after the
validator corrections. Disclosed in full because two of them produced wrong intermediate statements.

### 3.1 Generation defect: the call parenthesis was dropped

The first generation attempt produced `(dwo@rcx+0x34000)`. Root cause: `b[p]` is `(` and
`b[p+1:p+4]` is `dd(`, and I replaced those 3 characters with the 3 characters `dwo` instead of the
4 characters `dwo(`. The `(` of the call was consumed and never re-emitted.

```text
caught by   the predicted-size assert.  Predicted +16, actual +0.  8 sites x (3-3) = 0.
            A dedicated assert now also checks b"dwo@" is absent, and that the
            replacement grew the body by exactly 1 byte.
disposition the malformed file was deleted rather than left in a 13-file corpus.
            It was the product of this same step, referenced by no gate record, and
            syntactically broken.  8 x '(dwo@', 0 x '(dwo('.
            This happened before Static Validation and before any execution.  No
            validation result was suppressed.
```

### 3.2 The size prediction in the authorization was wrong

`dd(` is 3 characters, `dwo(` is 4. The delta is **1 byte per occurrence, +8 in total, 14,895 B**,
not the `+16 / 14,903` the authorization stated. The instruction for this gate repeated `14,903`
because it came from my own document.

```text
the machine check was right    the generation assert used 1 byte and passed
the prose was wrong            the authorization text and the script's print said 2 bytes
corrected in                   Step5DH section 2.1 and the S-T0-16 row
```

S-T0-16 is evaluated against the corrected **14,895**, not against the figure the instruction named.

### 3.3 Validator defect: label decoupled from the data under test

The first validation run reported S-T0-05, S-T0-06 and S-T0-07 as FAIL. All three were **validator**
defects and the artifact was correct.

```text
S-T0-05 / S-T0-06   EXCLUDED was defined as ("1f9ce04", "1f9cfb0").  The checks indexed
                    EXCLUDED[1] and EXCLUDED[0] to select the data, but hardcoded the
                    label text.  So S-T0-05 was labelled 0x1f9ce04 while testing 0x1f9cfb0's
                    data, and S-T0-06 the reverse.  Both compared correct data against the
                    wrong site's expectation and failed.
                    FIX  the label is now derived from the same constant that selects the
                    data: "0x%s ... " % s.

S-T0-07             expected a body length delta of +1 per site.  Each site carries 2
                    occurrences at 1 byte each, so the correct delta is +2.  Same
                    arithmetic error as 3.2.  The structural counts it also checks,
                    .echo / gc / .if / .else, were identical from the first run.
```

How the artifact's correctness was established independently of the buggy checks:

```text
1. the generation script's own paren-aware scan, which reported 2 inside .if at 0x1f9ce04
   and 0 inside .if at 0x1f9cfb0
2. a context dump of all 12 (dd( occurrences with 46 characters of leading context, which
   shows unambiguously that 0x1f9cfb0's two sit in display-command address arguments
   ("dd @$t2+0x30000+((dd(@$t2+0x34040)&...)") and 0x1f9ce04's two sit in .if conditions
   (".if (@$t16 == 1) { .if (dd(@rdi+0x34040) == (")
3. a third, separately written classifier that recomputed the .if condition spans and
   reported the same classification as the generation script
4. S-T0-13, whole-file reversibility, which is content-based and carries no site label
```

### 3.4 A near-miss worth recording: the self-validating check

My first product-identification rule was "the `.cdb` that is not the parent", which matched 13
files. The obvious fallback was to identify the product by the property S-T0-13 tests, namely that
reversing `dwo(` to `dd(` reproduces the parent. That would have made S-T0-13 **tautological**: it
would have passed by construction.

```text
the product is identified by NAME from the directory listing instead, so the reversibility
check remains an independent test of the bytes rather than a restatement of how the file
was selected.
```

This is the same family as every other defect in this lineage: a check written against a convenient
proxy instead of against the property.

## 4. Static Validation S-T0-01 to S-T0-17

| ID | result | check | observed |
|---|---|---|---|
| S-T0-01 | PASS | `0x1f9c550`: only change is `dd(` → `dwo(` | 363 B → 365 B, site reversal identical |
| S-T0-02 | PASS | `0x1f9c57d`: only change is `dd(` → `dwo(` | 251 B → 253 B, site reversal identical |
| S-T0-03 | PASS | `0x1f9c59a`: only change is `dd(` → `dwo(` | 252 B → 254 B, site reversal identical |
| S-T0-04 | PASS | `0x1f9c5da`: only change is `dd(` → `dwo(` | 2229 B → 2231 B, site reversal identical |
| S-T0-05 | PASS | `0x1f9ce04` `dd(` unchanged, 2 inside `.if` | parent (2, 0) = product (2, 0) |
| S-T0-06 | PASS | `0x1f9cfb0` `dd(` unchanged, 2, still outside `.if` | parent (0, 2) = product (0, 2) |
| S-T0-07 | PASS | guard text integrity, body length +2 per site | structure differs at none; delta 2/2/2/2 |
| S-T0-08 | PASS | `.echo C3_T0_ENTRY` unchanged | 1 = 1 |
| S-T0-09 | PASS | terminal `gc` unchanged | 22 = 22 |
| S-T0-10 | PASS | all 37 lines outside the 4 target spans byte-identical | none differ |
| S-T0-11 | PASS | source / test / CMake / build inputs unchanged | 0 SHA mismatches, git delta 12 pre-existing |
| S-T0-12 | PASS | new `.cdb` SHA-256 recorded | `74E8C64C…F9142` |
| S-T0-13 | PASS | **reversibility, whole file** | 8 × `dwo(` → `dd(` gives 14,887 B, identical to parent |
| S-T0-14 | PASS | `dwo(` inside `.if` at all 4 target sites, 2 each | (2,0) × 4 |
| S-T0-15 | PASS | `poi(` count identical | 7 = 7 |
| S-T0-16 | PASS | size exactly **14,895 B**, ASCII, no BOM, 41 CRLF, trailing CRLF | all confirmed |
| S-T0-17 | PASS | topology unchanged, 13 RVAs identical order | 13 |

**17 of 17 PASS.** `T0-Guard-Repair-Baseline-1` is confirmed.

## 5. What this gate does not certify

```text
NOT certified   that dwo returns the correct value.  DG-T0-Repair-3b remains OPEN.
                dwo is proven only to be a syntactically accepted .if condition operand,
                from the error-class distinction: a Memory access error is an
                evaluation-stage failure, whereas dd produced a parse-stage Syntax error.
NOT certified   that the T0 gate will open.
NOT certified   anything about enqueuePos beyond the source-confirmed declaration
                std::atomic<uint32_t> with alignas(64), and the width analysis.
NOT certified   that 0x1f9ce04 and 0x1f9cfb0 are correct.  They are deliberately still
                malformed.
```

## 6. State

```text
.cdb artifacts      14, one created this gate, none modified
frozen baselines     Step5BB-Redesign-2, Step5CP-Redesign-4, Step5CX-Redesign-6,
                     Step5DA-Redesign-7, Step5DD-Redesign-9   all unchanged
new baseline         T0-Guard-Repair-Baseline-1   74E8C64C…F9142   14,895 B
ConvoPeq.md          unchanged
AudioEngineHarness.exe   unchanged
tmp\cdb.exe          unchanged
git src/ CMakeLists build.bat   12, all pre-existing
CDB execution        0
residue processes    0
build invoked        no
IMPLEMENTATION       COMPLETE
STATIC VALIDATION    PASS 17/17
RUNTIME              NOT AUTHORIZED
```

## 7. Next

The Runtime Authorization request document. `cdb.exe` is not launched by this gate.
