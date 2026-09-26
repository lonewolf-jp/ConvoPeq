# Step5EA-0 — T0→T1 Handoff Audit

```text
gate                = prerequisite audit for RDG-4-1, at the Owner's direction
mode                = READ-ONLY.  No CDB run, no .cdb edit, no source / test / CMake / build /
                      harness change.  No runtime measurement.
execution this gate = 0
inputs              = the frozen Baseline-2 .cdb, whole corpus, balanced-paren guard extraction
purpose             = answer the five questions the Owner raised before RDG-4-1 is posed:
                      1 does T1 consume $t0 == 1 only once
                      2 how do T1's later state transitions treat $t0
                      3 how does $t17 prevent T1 re-entry
                      4 are the $t0 == 1 reads at 0x1fa80b2 / 0x1fa80b3 part of the same handoff
                      5 whether multi-handoff is meaningful
```

## 1. Q1 — `$t0` is monotonic and write-once

```text
site            RVA        reads $t0     writes $t0
--------------  ---------  ------------  ------------
T0E             0x1f9c550  == 0          none
T0P             0x1f9c57d  (none)        none
T0Q             0x1f9c59a  (none)        none
T0S             0x1f9c5da  == 0          = 1
T1A_CANDIDATE   0x1f9cfb0  == 1          none
SITE_b2         0x1fa80b2  == 1          none
SITE_b3         0x1fa80b3  == 1          none
T1C_DQUEUE      0x1f9cd00  (none)        none
T1_CAS_SUCCEED  0x1f9ce04  (none)        none
four post sites (0x1f9cd59, cd72, cfe2, cfe5)   (none)   none

every $t0 write in the entire corpus : ['1']
writes to 0                          : 0
```

**Answer to Q1: no, T1 does not consume `$t0 == 1` once.** `$t0` is never reset. It is
monotonic 0 → 1 and write-once, so after the first T0S the `$t0 == 1` condition is permanently
satisfied and **cannot itself prevent any re-entry**. The latch that actually prevents re-entry is
`$t17`, per Q3.

## 2. Q3 — `$t17` is the real one-shot latch, and it governs the S7 / T2 correlation

```text
site                RVA        reads $t17    writes $t17
------------------  ---------  ------------  -------------
T0S                 0x1f9c5da  (none)        = 0        ARMS it
T1A_CANDIDATE       0x1f9cfb0  == 0          none
SITE_b2  T2_RETURN  0x1fa80b2  == 0          = 1        SEALS it
SITE_b3  S7_ANCHOR  0x1fa80b3  == 0          none
```

The two sites that read `$t0 == 1` and are not T1 are, verbatim:

```text
0x1fa80b3  .if (@$t0 == 1) { .if (@$t17 == 0) { .if (@$t19 == 0) { .if (@rbx+0x10b76c0 == @$t3) {
              .echo C3_TERMINAL_S7_ANCHOR; r $t19=@$tid; ...

0x1fa80b2  .if (@$t0 == 1) { .if (@$t17 == 0) { .if (@$t19 != 0) { .if (@$tid == @$t19) {
              .if (@r15+0x10b76c0 == @$t3) { .echo C3_T2_WAIT_RETURN; r $t17=1; ...
```

**Answer to Q3, and to Q4.** The protocol is a three-way race on `$t17 == 0`:

```text
  S7_ANCHOR   0x1fa80b3   $t19 == 0                 ->  r $t19 = @tid    CLAIM the anchor
  T1A_CAND    0x1f9cfb0   $t17 == 0, $t12 == 0 ...  ->  r $t12 = 1       claim the T1 candidate
  T2_RETURN   0x1fa80b2   $t19 != 0, @tid == $t19    ->  r $t17 = 1       SEAL, only the anchor
                                                                           thread may do this

  $t19 records which thread won the anchor.  $t17 seals the pair so no second site can win.
```

**Answer to Q4: yes, both are part of the same handoff protocol**, and it is the protocol that
implements the workstream's own unresolved items. `$t19` is the S7 anchor thread identity and
`$t17` is the S7-to-T2 seal. This is the mechanism behind the open items `S7_READER` and
`S7_READER_SLOT`; they are gated by `$t17` and `$t19`, not by `$t0`.

## 3. Q2 — a second T0S firing destroys the entire accumulated T1 and S7 state

T0S's payload initialises, in this order:

```text
  $t0 = 1
  $t1 = @r11-0x1440     $t2 = @r11        $t3 = @$t1+0x10
  $t4,$t5 = rdx halves  $t6,$t7 = r8 halves   $t8,$t9 = r9 halves
  $t10 = @rbx
  $t11 = 0   $t12 = 0   $t13 = 0   $t14 = 0   $t15 = 0
  $t16 = 0   $t17 = 0   $t18 = 0   $t19 = 0
```

Cross-referencing which downstream sites depend on the zeroed registers:

```text
site                tests on registers that a SECOND T0S would zero
------------------  --------------------------------------------------------
T1A_CANDIDATE       $t12 == 1
T1C_DQUEUE_ENTRY    $t12 == 1
T1_CAS_SUCCEEDED    $t12 == 1, $t16 == 1
0x1f9cd59           $t12 == 1
0x1f9cd72           $t12 == 1, $t16 == 1
0x1f9cfe2           $t12 == 1
0x1f9cfe5           $t12 == 1
```

```text
  $t12 is the T1 phase-active flag, set by T1A_CANDIDATE.
  $t13 is the T1 observation count.
  $t14, $t15 are the captured pointer halves.
  $t16 is the CAS-succeeded flag.
  $t17 is the S7 seal.      $t19 is the S7 anchor thread.
  $t11 is the thread identity.

  A second T0S firing zeroes all of them.
```

**Answer to Q2: T1's later state transitions do not treat `$t0` at all — none of the seven
downstream sites reads it. They depend entirely on `$t11` to `$t19`, which T0S owns and
re-initialises.**

## 4. Q5 — multi-handoff is not merely undesigned. It is destructive.

```text
M-naive   remove the $t0 = 1 write
          => $t0 stays 0, T1A_CANDIDATE's @$t0 == 1 never holds, T1 never fires.
          EXCLUDED, as the Owner already ruled.

M         allow T0S to fire N times, keeping the write
          => firing 2 re-arms $t17 = 0 and $t19 = 0, so the S7 anchor race RE-RUNS,
             and it zeroes $t12, so the T1 phase flag is cleared and the T1 chain restarts
             from the beginning mid-flight.
          with N up to 4 the T1 and S7 state is erased up to three times after being built.
          this is not a missing design. it is a design that destroys what it depends on.
```

```text
S-EXT     a shape that was considered and is also defective

          drop T0S's entry guard so it fires N times, but keep the $t1..$t19 initialisation
          inside a one-shot guard so it runs only on the first firing.

          rejected on inspection: T0S's payload addresses its dumps through the STORED
          pseudo-registers, for example  dq @$t1+0x18 L1  and  dd @$t2+0x34000 L1,
          not through live r11.  on a second firing without re-initialisation, those dumps would
          describe the FIRST instance's EpochDomain and queue, not the current one.

          so S-EXT would emit N publication markers whose payloads all report instance 1.
          that is worse than a missing marker, because it is a wrong one.
```

**Answer to Q5: multi-handoff has no meaning under the current protocol.** Every variant is
either non-functional or state-destroying, and the only variant that would preserve state would
report the wrong instance.

## 5. Conclusion carried to RDG-4-1

```text
S   multi-capture / single-handoff
    T0Q captures = N,  T0S publication marker = 1,  $t0 = 1 written once,
    $t11..$t19 initialised once, S7 anchor race runs once, T1 fires once.

    THIS IS THE ONLY NON-DESTRUCTIVE DESIGN.  It is also the current structure, so adopting it
    requires no protocol change at all.

M   multi-handoff        ruled out, section 4
M-naive                  ruled out, section 4 and by the Owner
S-EXT                    ruled out, section 4
```

```text
CONSEQUENCE FOR RDG-3-4, WHICH MUST BE CARRIED INTO RDG-4-1

  RDG-3-4 fixed  BC = every captured T0Q has a corresponding T0S.
  Under S, T0S fires once, so for N > 1 exactly one capture has a corresponding T0S and the
  other N-1 do not.

  BC therefore CANNOT be "every capture" under the only surviving design.  it must be restated,
  and that restatement is part of the RDG-4-1 ruling, not a consequence the Owner can inherit
  silently.  for N = 1 the two formulations coincide, and N = 1 is not excluded: Q4-2 bounds
  N at 0..4 and does not fix it.
```

## 6. State

```text
RDG-4-2  Candidate E                    CLOSED   Owner ruled
RDG-4-3  live enqueuePos diagnostic     CLOSED   Owner ruled, KEEP as diagnostic only
RDG-4-1  single / multi handoff         NOT RULED.  audit complete, ruling now possible

RDG-4    NOT CLOSED   one item open, RDG-4-1

no CDB run    no .cdb edit    no source / test / CMake / build / harness change
no runtime measurement    no predicate implemented    no candidate implemented
OPEN-3  T0P / T0Q / T0S conjunct hit rates remain UNMEASURED and are deliberately not pursued
        here.  this audit used the frozen script and source/disassembly only.
```
