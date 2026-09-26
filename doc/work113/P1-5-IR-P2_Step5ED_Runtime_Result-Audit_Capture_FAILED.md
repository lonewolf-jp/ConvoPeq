# Step5ED — Runtime Result. The capture FAILED, and the design that authorised it is falsified.

```text
gate                = Runtime Result Audit, under the Runtime Authorization
mode                = measurement only.  No .cdb edit, no source / test / CMake / harness edit,
                      no build.  ONE execution, no retry.
execution           = CDB 1, harness 1, build 0
target              = Step5EC Baseline-3, SHA-256 re-verified before the run
                      EE82A517698C3257B5612D0EE9957B0AB7E0E027997AE0B4BE93C69947035E58  14,819 B
                      1 match from a directory listing.  OK to proceed.
inputs unchanged    Baseline-1  74E8C64C... unchanged
                      Baseline-2  BD5129A3... unchanged
                      child       EE82A517... unchanged by the run
                      harness     E5C7AFB9C4EAA48C
                      cdb.exe     5F54ABAFCA3AE563
log                 P1-5-IR-P2_Step5EC_...-Baseline-3_Runtime_CDB.log   2,603 lines, 170,197 B
VERDICT             FAIL.  capture count 0, FN 2.
```

## 1. Withdrawal — my first analysis returned a false PASS

My acceptance script tested `all(...)` over the list of accepted captures. That list was **empty**,
so every `all()` returned True and the script printed PASS. A check over an empty population is not
a pass; it is the absence of evidence, reported as evidence of absence.

```text
WITHDRAWN   "every accepted T0Q satisfies Candidate E  ->  PASS"
            "no non-slot-4 capture accepted              ->  PASS"

CORRECT     both are NOT APPLICABLE.  there were zero accepted captures to test.
            reported as no-data, not as pass.
```

## 2. Marker census, whole-line equality

```text
C3_DG_T0E_HIT                   65
C3_DG_T0E_R9_PASS                0
C3_DG_T0E_R9_FAIL               65
C3_DG_T0E_EQ_PASS                2
C3_DG_T0E_EQ_FAIL               63
C3_T0_ENTRY                      1
C3_DG_T0P_HIT                   65
C3_T0_CAS_PRE                    2
C3_DG_T0Q_HIT                   65
C3_T0_CAS_POST                   0      <-- THE CAPTURE NEVER FIRED
C3_DG_T0S_HIT                   65
C3_T0_SEQUENCE_PUBLISHED         1
C3_T1A_CANDIDATE                 1
C3_TERMINAL_S7_ANCHOR            0
C3_T2_WAIT_RETURN                0
C3_T1C_DQUEUE_ENTRY              0
C3_T1_CAS_SUCCEEDED              0
```

Emission order of the non-zero markers:

```text
C3_T0_ENTRY
C3_T0_CAS_PRE
C3_T0_SEQUENCE_PUBLISHED
C3_T0_CAS_PRE
C3_T1A_CANDIDATE
```

## 3. What DID happen, and it is a real result

```text
the T0 phase completed for the first time in this workstream.

  T0S fired, wrote $t0 = 1, and the T1 entry C3_T1A_CANDIDATE fired at 0x1f9cfb0.

this had never happened before.  in the prior two runs the gate never opened, so $t0 stayed 0
and the T1 chain was never entered.  removing the epoch literal is what made this possible, and
the handoff design of RDG-4-1 and Step5EA-0 is confirmed to work as designed.

but the CAPTURE marker did not fire, and it sits upstream of the publication.
```

## 4. Root cause. The conjunct `@r10 == 5` is UNSATISFIABLE at T0Q.

```text
0x141f9c555  movl 0x34000(%rcx), %r10d    r10d = pos
0x141f9c57d  lock cmpxchgl 0x34000(%r11)  CAS pos -> pos + 1        <-- T0P
0x141f9c591  movl 0x34000(%r11), %r10d   reload, diff > 0 path ONLY
0x141f9c59a  movzbl 0x28(%rsp), %eax     <-- T0Q    r10d is STILL pos
0x141f9c5cd  incl %r10d                  r10d = pos + 1
0x141f9c5d2  movl %r10d, 0x30000(...)   publish
0x141f9c5da  movq 0x8(%rsp), %rbx        <-- T0S    r10d = pos + 1

ORDER  0x59a  <  0x5cd  <  0x5da
```

```text
  at T0Q, r10d == pos == 4
  at T0S, r10d == pos + 1 == 5

  so  @r10 == 5  cannot hold at T0Q for a slot-4 acquisition.
  the correct producer-local conjunct at T0Q is  @r10 == 4.

confirmed by the run: r10 observed at C3_DG_T0Q_HIT took the values 0x4 and 0x5.  the 0x4 cases
are the slot-4 winners that then failed @r10 == 5.  the 0x5 cases are calls claiming position 5,
which correctly fail @rbx == 4 as well.
```

## 5. FN = 2. Two genuine acquisitions, zero captured.

```text
C3_T0_CAS_PRE    = 2    calls that reached the free-slot test at position 4, that is,
                         two genuine slot-4 acquisition attempts
C3_T0_CAS_POST   = 0    captures taken
difference       = 2    BOTH were missed

acceptance criterion 3, FN == 0, is therefore FAILED on direct evidence.
```

## 6. This falsifies my own design work, and the chain of propagation

```text
MY ERROR, in Step5EA-0 section 3

  I wrote:  "at T0Q and T0S, r10d = pos + 1, so @r10 == 5 is satisfiable and means
             this call advanced enqueuePos from 4 to 5"

  the increment  incl %r10d  is at 0x141f9c5cd, which is AFTER T0Q at 0x141f9c59a.
  I attributed the post-increment value to a breakpoint that precedes the increment.

  I had the instruction order in front of me and still misread it.  the earlier statement in
  the same document, that 0x141f9c591 reloads enqueuePos "only on the diff > 0 retry path",
  was correct, and it should have been the thing that blocked the wrong conclusion.
```

```text
WHERE IT PROPAGATED, all of it mine, none of it the Owner's

  Step5EA-0 §3      claimed @r10 == 5 was producer-local, satisfiable, and FN-immune
  RDG-3-2           the Owner adopted exactly that claim as the ruling
  RDG-4-2           Candidate E was defined with @r10 == 5
  Step5EB §3.3      the byte-exact contract specified @r10 == 5
  Step5EC           the implementation applied @r10 == 5 faithfully, and I-2 PASSED it

  the implementation is FAITHFUL.  the contract is WRONG.
```

```text
THE CONJUNCT WAS PRE-EXISTING, NOT INTRODUCED HERE

  parent 0x1f9c59a  already carried '@r10 == 5'
  parent 0x1f9c5da  already carried '@r10 == 5'

  so T0Q was never fireable in ANY run, including the original Retry-1.  the epoch literal was
  masking an unsatisfiable conjunct.  removing the literal exposed it.

  this is the same shape as the original finding: one conjunct was never satisfiable, and the
  workstream spent eight Redesigns on the wrong conjunct.  the second unsatisfiable conjunct
  was sitting immediately behind the first.
```

## 7. Criterion-by-criterion, honestly stated

```text
1  every accepted T0Q satisfies Candidate E
   NOT APPLICABLE.  zero accepted captures.  NOT a pass.

2  no non-slot-4 capture accepted
   NOT APPLICABLE.  zero accepted captures.  NOT a pass.
   note: this is not a vacuous truth being read as success.  it is untested.

3  FN = 0
   FAIL.  2 genuine acquisitions, 0 captured.

4  live enqueuePos as diagnostic
   NOT EXERCISED.  no capture was accepted, so the diagnostic was never observed at a capture.
   the enqueuePos observation that the Owner asked for cannot be made until the predicate fires.

5  cardinality
   N is a symbol, 0..4, and N == 4 is not a criterion.  no criterion was violated here.
   for the record:  T0E EQ_PASS 2, CAS_PRE 2, CAS_POST 0, SEQUENCE_PUBLISHED 1.
   the Owner ruled that SEQUENCE_PUBLISHED < CAS_POST is the Step5EB §4 T0S residual and not a
   capture defect.  here the INVERSE holds, SEQUENCE_PUBLISHED 1 > CAS_POST 0, which is a
   capture defect and not that residual.

6  T0E / T0P / T0Q / T0S cardinality observed.  DONE, values above.
```

## 8. Two further observations, not defects in this implementation

```text
PRE-EXISTING SYNTAX ERROR, one occurrence

  Syntax error at '(@$t2+0x34040)&0xfff)*4) '

  site   0x1f9cfb0, the T1A_CANDIDATE site
  text   dd @$t2+0x30000+((dd(@$t2+0x34040)&0xfff)*4) L1
  cause  dd( as an EXPRESSION FUNCTION inside an address expression, the Step5DF defect family
  parent and child are byte-identical here, and I-8 certified them

  so T1A_CANDIDATE's own payload is partially malformed.  it still emitted its marker, because
  the marker precedes the malformed command.  this is one of the two T1 sites already known to
  be malformed and deliberately left out of scope.
```

```text
THE RUN TERMINATED EARLY, and that is a behavioural change

  g returned at 0x1f9cfb0, C3_T1A_CANDIDATE, then the script ran to SCRIPT_END and q.

  in the prior two runs the gate never opened, so g ran to ZwTerminateProcess with 116 T0E
  hits.  here the gate opened, T1A fired, and the single g in the script ended the session
  after 65 T0E hits.

  consequence: the measurement window is shorter than the prior runs, so this run is NOT
  volume-comparable with the 116-hit runs.  and any future runtime gate must handle the early
  T1A hit, for example by looping on g rather than issuing it once.
```

## 9. State

```text
Implementation      executed, I-1..I-13 ALL PASS, contract faithfully applied
Runtime             ONE run executed, no retry
Capture             FAILED.  0 captures.
FN                  2
Handoff             SUCCEEDED.  T0S fired, $t0 = 1 written, T1 entry reached.
baselines           7/7 intact      .cdb corpus 16      child unchanged by the run
ConvoPeq.md MATCH   harness MATCH    git delta 12, all pre-existing    residue 0
build               NOT INVOKED
```

## 10. What the next gate must do, and what must not be inferred

```text
the defect is in the DESIGN, so the next step is a design correction, not a re-run.
re-running this baseline would reproduce the same result and is not authorised.

the correction is narrow and is stated as a finding, NOT as an implementation:

  at T0Q, the producer-local position conjunct must be  @r10 == 4, not @r10 == 5.
  @rbx == 4 and dwo(@r11+0x30010) == 4 are unaffected and remain correct.

this requires, in order:
  a correction to the Step5EA-0 §3 register analysis, with the instruction order as evidence
  a re-opening of RDG-3-2 and RDG-4-2, because Candidate E as ruled is not implementable
  a new Implementation Authorization
  a new Runtime Authorization

nothing above is authorised, and no edit may be made on the strength of this document.

ALSO UNCHANGED, still out of scope
  the two malformed T1 sites, 0x1f9ce04 and 0x1f9cfb0.  the 0x1f9cfb0 malformation is now
  observed at runtime, which is new evidence but not authorisation.
  OPEN-3 remains partly open.  CAS_PRE 2 and CAS_POST 0 are now measured, which closes the
  T0P and T0Q conjunct rates for THIS run only, at 65 T0E hits rather than 116.
```
