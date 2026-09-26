# P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Execution-1-Result-Audit-1

## 1. Audit result

```text
Gate     = P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Execution-1-Result-Audit-1
Mode     = read-only classification of the consumed execution
CDB execution / ping execution / rerun = 0
repair / new .cdb / build                = 0

B1              = PROVEN
Candidate A     = PROVEN (unchanged, independently established)
```

## 2. Frozen identity

All values recomputed at this gate, not inherited.

| # | item | fixed value | observed | result |
|---|---|---|---|---|
| R1 | `ConvoPeq.md` SHA-256 | `E5E74200…F3609` | identical | PASS |
| R2 | `ConvoPeq.md` bytes | `5,535,334` | identical | PASS |
| R3 | Semicolon Separator script SHA-256 | `EA87DE2E…0E2F55` | identical | PASS |
| R4 | Semicolon Separator script bytes | `1,184` | identical | PASS |
| R5 | `cdb.exe` SHA-256 | `5F54ABAF…FBEE67` | identical | PASS |
| R6 | CDB version | `10.0.29617.1000` | identical | PASS |
| R7 | Execution-1 log SHA-256 | `2797F5F4…28193F` | identical | PASS |
| R8 | Execution-1 log bytes | `6,184` | identical | PASS |

```text
frozen identity drift = 0
```

Source authority is the current `ConvoPeq.md`. The superseded `ConvoPeq(3).md` was not used and is not
present in the working tree; exactly one `ConvoPeq*.md` exists, so no substitution was available even
in principle.

## 3. B1 success conditions, machine cross-checked

Markers were counted under the rule fixed by the authorization: a marker counts as reached only where
a whole log line, after trimming, is exactly the marker. Occurrences inside the `bp` command string,
the `bl` listing, or any error message are not arrival evidence.

| # | condition | evidence | observed | result |
|---|---|---|---|---|
| C1 | breakpoint reached | `C3B1C_BREAKPOINT_HIT` | 1, line 91 | PASS |
| C2 | B1 derivation reached | `C3B1C_B1_DERIVED` | 1, line 92 | PASS |
| C3 | `$t11` is the image base | `$t11` vs `ModLoad` | identical | PASS |
| C4 | T1 agreement | `XCHK_T1_AGREE` | 1, line 100 | PASS |
| C5 | T2 agreement | `XCHK_T2_AGREE` | 1, line 101 | PASS |
| C6 | T3 agreement | `XCHK_T3_AGREE` | 1, line 102 | PASS |
| C7 | T4 agreement | `XCHK_T4_AGREE` | 1, line 103 | PASS |
| C8 | T5 agreement | `XCHK_T5_AGREE` | 1, line 104 | PASS |
| C9 | crosscheck complete | `C3B1C_CROSSCHECK_END` | 1, line 105 | PASS |
| C10 | sentinel complete | `C3B1C_SENTINEL_CLEARED` | 1, line 106 | PASS |
| C11 | tally complete | `C3B1C_TALLY` | 1, line 107 | PASS |
| C12 | `$t18` equals 6 | counter readback | `0000000000000006` | PASS |
| C13 | parser / runtime error | error scan | 0 | PASS |
| C14 | process residue | process list | 0 | PASS |

```text
conditions met = 14 / 14
```

## 4. Image base verification, six independent forms

The claim that `$t11` is the `ping.exe` image base was checked six ways rather than once, since the
whole conclusion rests on it.

```text
image range from ModLoad  = 00007ff601b80000 .. 00007ff601b8c000
breakpoint address (bl)   = 00007ff601b839d9   symbolic form ping+0x39d9
$t11 readback              = 00007ff601b80000
```

| # | check | result |
|---|---|---|
| C3 | `$t11` equals the `ModLoad` image base | PASS |
| C3b | `$t11 + 0x39d9` equals the breakpoint address | PASS |
| C3c | image base `+ 0x39d9` equals the breakpoint address | PASS |
| C3d | the breakpoint lies inside `[base, end)`, RVA `0x39d9` | PASS |
| C3e | `@$bp0 - 0x39d9` reproduces the image base | PASS |
| C3f | `bl` also carries the symbolic form `ping+0x39d9` | PASS |

## 5. Counter internal consistency

The verdict does not rest on the bare number 6. The counter was reconciled against the script and
against the marker set.

| # | check | observed | result |
|---|---|---|---|
| S1 | `r $t18 = 0` initialiser present exactly once | 1 | PASS |
| S2 | increment sites in the frozen script | 6 | PASS |
| S3 | no other mutator of `$t18` | only the initialiser and the 6 increments | PASS |
| S4 | `$t16`, `$t17`, `$t19` never incremented | 0 increment-form occurrences | PASS |
| S5 | `$t18` readback in the log | `0000000000000006` | PASS |
| S6 | `0` initialiser `+ 6` reached increments `= 6` readback | consistent | PASS |
| S7 | 5 `AGREE` markers `+ 1` `SENTINEL_CLEARED` `= 6` increments `= $t18` | consistent | PASS |

```text
T1..T5 reached = 1, 1, 1, 1, 1
SENTINEL_CLEARED reached = 1
total increments = 6
$t18 readback   = 6
```

The decomposition is exact. The log's `$t16 = 0`, `$t17 = 0`, `$t19 = 0` corroborate that no other
counter moved, so the tally cannot have been reached by any path other than the six intended
increments.

## 6. Candidate A and B1 independence

Re-confirmed rather than assumed. The interpretation that Candidate A had been regenerated from
`$t11` is explicitly rejected.

### 6.1 Structural, from the frozen script

| # | check | observed | result |
|---|---|---|---|
| I1 | Candidate A assignments derived from the `ping` symbol | 5 | PASS |
| I2 | Candidate A assignments derived from `$t11` | 0 | PASS |
| I3 | B1 primitive `r $t11=@$bp0-0x39d9` | 1 | PASS |
| I3 | B-side anchors derived from `@$t11` | 5 | PASS |

```text
Candidate A side, script level
  r $t1 = ping+0x3c
  r $t2 = ping+0x4FD0
  r $t3 = ping+0x2600
  r $t4 = ping+0x5000
  r $t5 = ping+0x40

B1 side, breakpoint body
  r $t11=@$bp0-0x39d9
  r $t12=@$t11+0x3c    r $t13=@$t11+0x4FD0    r $t14=@$t11+0x2600
  r $t15=@$t11+0x5000   r $t6=@$t11+0x40
```

### 6.2 Arithmetic, from the log

Every A-side anchor was recomputed as image base `+ RVA`, and every B-side anchor as `$t11 + RVA`.

| | A reg | A observed | expected `base+RVA` | B reg | `\$t11+RVA` | agree |
|---|---|---|---|---|---|---|
| T1 | `$t1` | `00007ff601b8003c` | `00007ff601b8003c` | `$t12` | `00007ff601b8003c` | yes |
| T2 | `$t2` | `00007ff601b84fd0` | `00007ff601b84fd0` | `$t13` | `00007ff601b84fd0` | yes |
| T3 | `$t3` | `00007ff601b82600` | `00007ff601b82600` | `$t14` | `00007ff601b82600` | yes |
| T4 | `$t4` | `00007ff601b85000` | `00007ff601b85000` | `$t15` | `00007ff601b85000` | yes |
| T5 | `$t5` | `00007ff601b80040` | `00007ff601b80040` | `$t6` | `00007ff601b80040` | yes |

```text
I4-I8  all five anchors agree on both derivations   PASS
I9     both paths yield the same base               PASS
```

### 6.3 The two paths are genuinely independent

```text
path A   input = the "ping" module symbol, resolved by the debugger from the loaded module
path B1  input = "@$bp0", the runtime address of the breakpoint just hit
```

Neither input is computed from the other. Path B1 never consults the `ping` symbol; it starts from a
live address the processor actually reached and subtracts a known offset. The agreement of the two
bases is therefore evidence, not tautology.

Had Candidate A been written as `$t11 + RVA`, each comparison would have compared a value with
itself and the independence under test would have been destroyed. It was not, and check I2 confirms
this structurally.

## 7. Why PROVEN rather than FAILED or INCONCLUSIVE

```text
B1 = FAILED      requires CROSSCHECK_END reached AND an explicitly shown disagreement.
                 CROSSCHECK_END was reached, and no disagreement exists. The body has
                 no DISAGREE branch, so a disagreement would surface as a missing
                 AGREE marker after a completed crosscheck. All five AGREE markers
                 are present and the counter confirms six increments.
                 -> excluded

B1 = INCONCLUSIVE requires the breakpoint unreached, or CROSSCHECK_END unreached, or
                 TALLY unreached, or a parser or runtime error.
                 None applies. All were reached, and the error scan is empty.
                 -> excluded

B1 = PROVEN      all conditions met.  -> adopted
```

The rule that a `T2..T5_AGREE` marker missing before `CROSSCHECK_END` must not be read as a
disagreement is not needed here, because `CROSSCHECK_END` was in fact reached. The crosscheck ran to
its end marker, so the five agreements are positive evidence rather than an absence.

## 8. What is now established

```text
B1 = PROVEN
  the anchor-address derivation primitive
      image base = @$bp0 - (breakpoint RVA)
  is valid, and it agrees with the module-symbol derivation across all five anchors

CDB command-body fact, also established
  on this build, a block-closing "}" followed by a space does not act as a command
  boundary, while "}" followed by ";" does
```

The second point closes the investigation opened by the two earlier executions. It is a statement
about this CDB build's command-body parsing, established by three executions whose bodies differed in
length and in the construct following the brace, and identical only in the separator.

## 9. What B1 = PROVEN does not mean

```text
B1 = PROVEN  is NOT  the production C3 T1 capture
B1 = PROVEN  is NOT  identification of S7_READER
B1 = PROVEN  is NOT  a capture of minReaderEpoch
B1 = PROVEN  is NOT  a reclaim observation
```

The evidence was produced entirely on `C:\Windows\System32\ping.exe`, a benign target. The anchor
primitive was demonstrated on `ping`, whose image base came from `ping+0x39d9`. Nothing in this
audit touched ConvoPeq reader, epoch, or reclaim state, and no such conclusion may be drawn from it.

## 10. Disposition

```text
DIRECT_DQUEUE_CAUSE = epoch gate equality PROVEN
Candidate A         = PROVEN
@$bp0 primitive     = PROVEN
$t11 derivation     = PROVEN
T1 agreement        = PROVEN
B1 full runtime     = PROVEN          <- changed by this gate

S7_READER          = UNRESOLVED
S7_READER_SLOT     = UNRESOLVED
minReaderEpoch     = NOT CAPTURED
T1                 = NOT CAPTURED
Case A/B/C/D       = NOT PROVEN
IMPLEMENTATION     = FORBIDDEN
```

The reclaim condition remains `retireEpoch < minReaderEpoch`, with reader and epoch observation kept
separate from reclaim. The RT-side observation boundary and the NonRT lifetime isolation line are
unchanged. No ConvoPeq reader, epoch, or reclaim evidence was produced or altered.

## 11. Not performed in this gate

```text
CDB execution / ping execution / rerun / script modification / new .cdb = 0
build / Dr.Memory / AudioEngineHarness / production C3 / T1 capture
ReaderSlot / EpochDomain / getMinReaderEpoch / DQueue / reclaim / M1 / M2
Retry-3 / Preparation-4 / production, test, CMake, source modification = 0
```

Read-only confirmation:

```text
process residue                = 0
execution logs on disk         = 5, unchanged
newest log last write          = 09/26/2026 10:04:43, that is, the Gate 4 execution
audit performed at             = 09/26/2026 10:09:36
.cdb inventory                 = 7, unchanged
frozen script SHA              = EA87DE2E…0E2F55, unchanged
execution log SHA              = 2797F5F4…28193F, unchanged
ConvoPeq.md                    = E5E74200…F3609, unchanged
src, test, CMake, build.bat    = 12 entries, unchanged from before this gate
```

The log's last write time precedes this audit, which confirms the audit did not alter the evidence it
read.

## 12. Audit harness integrity

Several checks first reported FAIL for reasons in the checking harness, not in the evidence. Each was
diagnosed, corrected, and re-run. They are recorded so the PASS results are not credited to a
permissive check.

| # | harness defect | correction |
|---|---|---|
| 1 | PowerShell consumed the backtick in an address pattern | built the pattern from `[char]96` |
| 2 | `ModLoad` pattern matched one address pair, but the line has two | matched start and end explicitly |
| 3 | the breakpoint field was captured with the rest of the `bl` line | captured the 17-character address field, then stripped the backtick |
| 4 | `$t16`/`$t17`/`$t19` check matched the `= 0` initialisers | tested for the increment form `@$t1[679]` only |
| 5 | nested quoting produced a parser error in the arithmetic block | precomputed booleans into variables |

None of these touched the log or the script. The figures in sections 3 to 6 come from the corrected
checks.

## 13. Next step

A separate **design-only** gate may now address connecting the proven anchor derivation to the C3 T1
capture, that is:

```text
C3 runtime
    |
    v
T1 getMinReaderEpoch
    |
    v
minReaderEpoch
    |
    v
ReaderSlot identity correlation
```

That gate is design only. It must not create the production script, must not execute anything, and
requires its own preparation, static validation, and authorization before any run.

Proceeding directly to `AudioEngineHarness`, production C3, T1 capture, ReaderSlot, EpochDomain,
`getMinReaderEpoch`, `minReaderEpoch` capture, DQueue or reclaim, M1, M2, Retry-3, Preparation-4, or
any production source modification remains forbidden. The `ping` plus breakpoint anchor proof must not
be read as a successful ConvoPeq T1 capture.

## 14. Final state

```text
P3-5-FPM-C3-T1-Capture-PostRetry2-Benign-Probe-Candidate-B1-Semicolon-Separator-Probe-Execution-1-Result-Audit-1
= CLOSED / B1 = PROVEN / CANDIDATE A = PROVEN

frozen identity            = 8 / 8 PASS, drift 0
B1 success conditions      = 14 / 14 PASS
image base forms           = 6 / 6 PASS
counter consistency        = 7 / 7 PASS
independence               = 9 / 9 PASS
harness defects found      = 5, corrected and re-run

BREAKPOINT_HIT = 1    B1_DERIVED      = 1
T1..T5_AGREE   = 5/5  CROSSCHECK_END  = 1
SENTINEL_CLEARED = 1  TALLY           = 1
$t18          = 6    parser error    = 0
process residue = 0

B1 full runtime    = PROVEN
CDB command-body separator fact = PROVEN
S7_READER / S7_READER_SLOT = UNRESOLVED
minReaderEpoch / T1 = NOT CAPTURED
Case A/B/C/D = NOT PROVEN
IMPLEMENTATION = FORBIDDEN
```
