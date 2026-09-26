# Step5EB — Implementation Authorization Request

```text
gate                = Implementation Authorization Request.  NOT AN AUTHORIZATION.
mode                = READ-ONLY.  Nothing is edited, nothing is executed, nothing is built.
                      No .cdb created.  No source, test, CMake or harness change.
execution           = 0
parent artifact     = Baseline-2, resolved by SHA-256, NOT by a transcribed filename
                      SHA-256 BD5129A3655FA04FB1EAF8539BA596BC47179FABD8817B85F3742216F083B763
                      14,968 B
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
preconditions       = Q3 CLOSED, Q4 CLOSED, RDG-2 CLOSED, RDG-3 CLOSED, RDG-4 CLOSED,
                      Step5EA-0 CLOSED
supersedes          the two earlier drafts of this request.  the first scoped epoch-literal
                    removal OUT, which the Owner rejected as implementation scope.  the second
                    left the footprint open, which this draft closes.
status              = REQUEST ONLY.  scope RESOLVED.  all confirmation points CLOSED.
                      authorization NOT YET ISSUED.  no edit performed.
```

## 1. Footprint ruling, recorded

```text
F-C  the obsolete epoch literal conjunct  @r9 == 9  is removed from ALL FOUR T0 protocol
     sites:  T0E  T0P  T0Q  T0S

     T0Q  additionally receives the Candidate E change
             REMOVE  dwo(@r11+0x34000) == 5
             ADD     @rbx == 4
             RETAIN  @r10 == 5
             RETAIN  dwo(@r11+0x30010) == 4

     T0P and T0S have NO other conjunct changed.

     enqueuePos   predicate conjunct at T0Q  REMOVED
                  diagnostic payload at T0Q  KEPT

     the rationale the Owner gave, and which this contract implements:
       the epoch literal is not a constituent of the capture identity adopted at RDG-4-2, and
       leaving it at only some sites would create a non-uniform dead gate WITHIN one
       acquisition sequence.
```

## 2. One reading that must be confirmed at authorization

```text
THE CENSUS, verified in the frozen file

  @r9 == 9 occurrences, whole file          5
    T0E  0x1f9c550   2 occurrences
    T0P  0x1f9c57d   1 occurrence
    T0Q  0x1f9c59a   1 occurrence
    T0S  0x1f9c5da   1 occurrence

AT T0E THE TWO OCCURRENCES HAVE DIFFERENT ROLES

  occurrence 1, in the unconditional diagnostic prefix
      .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ; .else { .echo C3_DG_T0E_R9_FAIL }
      this is the MEASUREMENT that produced 0 of 116.  it is not a conjunct.

  occurrence 2, inside the guard
      .if (@$t0 == 0) { .if (@r9 == 9) { .if (dwo(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY ...
      this is the CONJUNCT the ruling removes.
```

```text
READING APPLIED IN THIS REQUEST

  the ruling says "the obsolete epoch literal CONJUNCT", so at T0E the GUARD occurrence is
  removed and the DIAGNOSTIC occurrence is RETAINED.

  it is retained because deleting it would destroy the per-run evidence that r9 is still never
  9, which is the premise the whole repair rests on.  without it, no future run can falsify that
  premise.  the Owner also identified this measurement as the traceability cost of F-C, which
  indicates awareness that it is there.

  RESULT  the whole-file count goes 5 -> 1, and the surviving occurrence is the T0E diagnostic.

  >>> OWNER CONFIRMED, section 2 CLOSED.  The Owner ruled YES: the T0E diagnostic-only
  >>> @r9 == 9 is RETAINED, and only the T0E guard occurrence is REMOVED.
  >>>
  >>>   T0E  @r9 == 9   diagnostic prefix   ->  RETAIN
  >>>   T0E  @r9 == 9   guard conjunct      ->  REMOVE
  >>>   T0P  @r9 == 9   guard conjunct      ->  REMOVE
  >>>   T0Q  @r9 == 9   guard conjunct      ->  REMOVE
  >>>   T0S  @r9 == 9   guard conjunct      ->  REMOVE
  >>>
  >>>   whole-file census  before 5  ->  after 1
  >>>   surviving occurrence  T0E diagnostic only, NOT a guard conjunct at any site
  >>>
  >>> The Owner's stated reason: the diagnostic is the measurement that reproduces or falsifies
  >>> the 0-of-116 result, the ruling targets the obsolete guard conjunct, and the contract as
  >>> presented contained no ground for deleting the diagnostic as well.
  >>>
  >>> CONSEQUENCE  I-9 is now FIXED at 1 and no longer carries a restatement option.  the
  >>> alternative reading is closed, not deferred.  the C-a correction in section 3.6 stands and
  >>> I-9 verifies the census independently of any conjunct count.
```

## 3. The change contract, byte-exact, per line

Current lengths and ladders, verified in the frozen file:

```text
site  RVA        length  .if  .else  ladder balanced
T0E   0x1f9c550    438 B    5     5   yes
T0P   0x1f9c57d    253 B    4     4   yes
T0Q   0x1f9c59a    254 B    4     4   yes
T0S   0x1f9c5da   2231 B    7     7   yes
```

### 3.1 T0E, 0x1f9c550 — guard conjunct removed, diagnostics untouched

BEFORE
```text
.echo C3_DG_T0E_HIT ; .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ; .else { .echo C3_DG_T0E_R9_FAIL } ; .if (dwo(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo C3_DG_T0E_EQ_FAIL } ; r ; .echo C3_DG_T0E_EP0 ; dq @rcx-0x1450 L11 ; .echo C3_DG_T0E_WIN_END ; .if (@$t0 == 0) { .if (@r9 == 9) { .if (dwo(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```
AFTER
```text
.echo C3_DG_T0E_HIT ; .if (@r9 == 9) { .echo C3_DG_T0E_R9_PASS } ; .else { .echo C3_DG_T0E_R9_FAIL } ; .if (dwo(@rcx+0x34000) == 4) { .echo C3_DG_T0E_EQ_PASS } ; .else { .echo C3_DG_T0E_EQ_FAIL } ; r ; .echo C3_DG_T0E_EP0 ; dq @rcx-0x1450 L11 ; .echo C3_DG_T0E_WIN_END ; .if (@$t0 == 0) { .if (dwo(@rcx+0x34000) == 4) { .echo C3_T0_ENTRY; r; dq @rsp L1; ln poi(@rsp); gc } ; .else { gc } } ; .else { gc }
```
`.if` 5 → 4.  The diagnostic prefix, the register dump, the window dump and the payload are byte-identical.

### 3.2 T0P, 0x1f9c57d — outer conjunct removed, nothing else touched

BEFORE
```text
.echo C3_DG_T0P_HIT ; .if (@r9 == 9) { .if (@r10 == 4) { .if (dwo(@r11+0x34000) == 4) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_PRE; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc } } ; .else { gc }
```
AFTER
```text
.echo C3_DG_T0P_HIT ; .if (@r10 == 4) { .if (dwo(@r11+0x34000) == 4) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_PRE; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```
`.if` 4 → 3.  `@r10 == 4`, `dwo(@r11+0x34000) == 4`, `dwo(@r11+0x30010) == 4` all retained.

### 3.3 T0Q, 0x1f9c59a — Candidate E, the formal capture predicate

BEFORE
```text
.echo C3_DG_T0Q_HIT ; .if (@r9 == 9) { .if (@r10 == 5) { .if (dwo(@r11+0x34000) == 5) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_POST; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc } } ; .else { gc }
```
AFTER
```text
.echo C3_DG_T0Q_HIT ; .if (@r10 == 5) { .if (@rbx == 4) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_POST; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```
`.if` 4 → 3.  The result is exactly Candidate E, the RDG-3-1 / RDG-4-2 capture identity.

### 3.4 T0S, 0x1f9c5da — one conjunct removed, payload and latch byte-identical

BEFORE, guard head
```text
.echo C3_DG_T0S_HIT ; .if (@$t0 == 0) { .if (@r9 == 9) { .if (@r10 == 5) { .if (@rbx == 4) { .if (dwo(@r11+0x34000) == 5) { .if (dwo(@r11+0x30010) == 5) { .if (@r11 != 0) { .echo C3_T0_SEQUENCE_PUBLISHED;r $t0=1; ...
```
AFTER, guard head
```text
.echo C3_DG_T0S_HIT ; .if (@$t0 == 0) { .if (@r10 == 5) { .if (@rbx == 4) { .if (dwo(@r11+0x34000) == 5) { .if (dwo(@r11+0x30010) == 5) { .if (@r11 != 0) { .echo C3_T0_SEQUENCE_PUBLISHED;r $t0=1; ...
```
`.if` 7 → 6.  Everything from `.echo C3_T0_SEQUENCE_PUBLISHED` to the end of the line, that is
`r $t0=1`, the whole `$t1 .. $t19` initialisation, every `dd` and `dq` and the 64-slot reader
sweep, is byte-identical.

### 3.5 Summary

```text
              T0E   T0P   T0Q   T0S
epoch @r9==9  -1    -1    -1    -1        guard conjuncts only
T0Q enqueuePos conjunct             -1
T0Q @rbx == 4                      +1
everything else                     unchanged, byte for byte

.if count    5->4  4->3  4->3  7->6
@r9 == 9 in the whole file   5 -> 1, the survivor being the T0E diagnostic
```

### 3.6 Three independent checks required on the edit

```text
C-a  conjunct extraction per line, NOT a regex.  an earlier regex attempt in this workstream
     silently dropped dwo(...) conjuncts and produced a fictitious count, so the extractor
     itself is part of the contract.

     PREDICATES ARE PARTITIONED.  guard conjuncts and diagnostic-only predicates are counted
     separately and never summed into one figure.  the partition is MECHANICAL, not by
     convention:

       T0E   everything before  .echo C3_DG_T0E_WIN_END   is the DIAGNOSTIC prefix
             everything from    .if (@$t0 == 0)            onward is the GUARD
       T0P   no diagnostic prefix, every .if is a GUARD conjunct
       T0Q   no diagnostic prefix, every .if is a GUARD conjunct
       T0S   no diagnostic prefix, every .if is a GUARD conjunct

     EXPECTED, per line, after the edit

       T0E   guard conjuncts = 2
               @$t0 == 0
               dwo(@rcx+0x34000) == 4
             diagnostic-only predicates = 2, both unchanged from the parent
               @r9 == 9                  <-- the subject of the census, retained exactly once
               dwo(@rcx+0x34000) == 4    <-- the EQ_PASS / EQ_FAIL probe, also retained
             total .if = 4

       T0P   guard conjuncts = 3
               @r10 == 4
               dwo(@r11+0x34000) == 4
               dwo(@r11+0x30010) == 4
             diagnostic-only predicates = 0
             total .if = 3

       T0Q   guard conjuncts = 3
               @r10 == 5
               @rbx == 4
               dwo(@r11+0x30010) == 4
             diagnostic-only predicates = 0
             total .if = 3

       T0S   guard conjuncts = 6
               @$t0 == 0
               @r10 == 5
               @rbx == 4
               dwo(@r11+0x34000) == 5
               dwo(@r11+0x30010) == 5
               @r11 != 0
             diagnostic-only predicates = 0
             total .if = 6
```

```text
C-a SEPARATE CHECK, the whole-file census, stated independently of any conjunct count

  @r9 == 9 occurrences, whole file
    before   5
    after    1
  the surviving occurrence
    is the T0E DIAGNOSTIC-ONLY predicate
    is NOT a guard conjunct, at any site

  this is the same quantity I-9 verifies, and it is deliberately NOT expressed as a conjunct
  count, because conflating the two is what this correction removes.
```

```text
C-b  brace and ladder balance per line, against the totals in section 3.5

C-c  byte-for-byte preservation of every payload, checked by substring:
       T0Q   dd @r11+0x34000 L1 ; dd @r11+0x30010 L1
       T0S   r $t0=1  and  the $t11 = 0 .. $t19 = 0 initialisation and the 64-slot sweep
```

## 4. Residual risk, stated and accepted

```text
T0P and T0S RETAIN  dwo(@r11+0x34000) == 4  and  == 5  respectively.

RDG-3-2's prohibition binds CAPTURE ACCEPTANCE PREDICATES.  T0P and T0S are not the capture
marker, so the prohibition does not reach them, and the Owner ruled their other conjuncts
unchanged.

the consequence is nevertheless real and is recorded rather than minimised:

  T0P can fail to fire if another producer advances enqueuePos between the free test and the
  CAS, and T0S can fail for the same reason roughly fifteen instructions later.  either failure
  leaves the bracket without its CAS_PRE or without its publication.

  this does NOT affect the capture identity, which is producer-local and owner-exclusive and
  therefore FN = 0 regardless.  it affects bracket completeness only, and bracket completeness
  is not an acceptance criterion under the BC as restated in Step5EA section 4.

  if a future run shows SEQUENCE_PUBLISHED absent while CAS_POST is present N times, that is
  this effect and not a capture defect.  the Owner should know the signature in advance.
```

## 5. Out of scope, explicitly

```text
T0S payload, $t0 write, $t11..$t19 semantics     PROHIBITED
$t0 reset or reuse                               PROHIBITED
T1 site repair                                   PROHIBITED.  0x1f9ce04 and 0x1f9cfb0 keep
                                                 2 x dd( each
S7 reader semantics, $t17, $t19                  PROHIBITED
S7_READER / S7_READER_SLOT                      OUT OF SCOPE, separate work item
minReaderEpoch(T1) / Case A-D                    OUT OF SCOPE, separate work item
any epoch semantics other than removing the      PROHIBITED.  the epoch VALUE is not redefined,
obsolete conjunct                                 no new epoch predicate is added, and r9 is
                                                 not re-read or re-interpreted
entry.type, publicationSequenceId, generation   OUT OF SCOPE.  PROVEN available at
                                                 [rsp+0x28], [rsp+0x30], [rsp+0x38], but
                                                 not required by Candidate E
OPEN-3 conjunct hit rates                       NOT MEASURED.  no runtime execution here.
source / test / CMake / harness                  PROHIBITED
build                                            PROHIBITED
```

## 6. Artifact handling discipline

```text
the parent MUST be resolved by SHA-256 from a directory listing, never by a transcribed filename.
the expected value is in this request's header and MUST be re-verified at edit time.  if the
listing yields an unexpected count, the edit does not proceed and the discrepancy is reported.

the edit produces a NEW file, a Baseline-3 descendant.  Baseline-1 and Baseline-2 are NOT edited
and MUST remain byte-identical.  I-11 requires this.

the child filename is assigned at edit time and recorded.  no filename is invented in advance.
```

## 7. Implementation acceptance criteria

```text
I-1   the four capture-relevant markers are unchanged in name and count of definition sites:
      C3_T0_ENTRY, C3_T0_CAS_PRE, C3_T0_CAS_POST, C3_T0_SEQUENCE_PUBLISHED
I-2   the T0Q capture predicate is exactly Candidate E, 3 conjuncts, per check C-a
I-3   dwo(@r11+0x34000) == 5 does NOT appear in the T0Q guard
I-4   dd @r11+0x34000 L1 IS present in the T0Q diagnostic payload, byte-identical to the parent
I-5   per-line .if and .else counts match section 3.5, and every ladder is balanced
I-6   the T0S payload is byte-identical to the parent, including r $t0=1, the $t11..$t19
      initialisation, every dd and dq, and the 64-slot reader sweep
I-7   exactly one site writes $t0 = 1, and it is T0S
I-8   every T1 and S7 site line is byte-identical to the parent, that is
      0x1f9cfb0, 0x1f9cd00, 0x1f9ce04, 0x1fa80b2, 0x1fa80b3
I-9   the whole-file count of @r9 == 9 is exactly 1, and that occurrence is the T0E
      DIAGNOSTIC-ONLY predicate, not a guard conjunct at any site.  verified as a census, NOT as
      a conjunct count, per the C-a separate check.  FIXED at 1; the alternative reading was
      closed by Owner confirmation in section 2, not deferred.
I-10  source, test, CMake and harness are unchanged; git status on those paths shows the same
      12 pre-existing entries and no new ones
I-11  all 7 frozen baselines verify by SHA-256, and the .cdb corpus grows by exactly 1
I-12  the child's SHA-256 and byte length are recorded, and the parent's are re-verified
      unchanged after the edit
I-13  a read-only diff audit precedes any execution and must show changes ONLY at the four T0
      RVAs 0x1f9c550, 0x1f9c57d, 0x1f9c59a, 0x1f9c5da, with no other line differing in any byte
I-14  no execution occurs.  runtime requires a separate Runtime Authorization
```

## 8. Order after authorization, unchanged

```text
RDG-4 CLOSED
   ->  Implementation Authorization
   ->  confirm the section 2 reading
   ->  .cdb edit, four T0 lines, new file
   ->  static validation, I-1 .. I-13
   ->  read-only diff audit
   ->  Runtime Authorization                SEPARATE
   ->  measurement
```

## 9. First runtime measurement, and what it will and will not assert

```text
after a separate Runtime Authorization, and NOT before:

  assert  every accepted T0Q satisfies Candidate E
  assert  no non-slot-4 T0Q is accepted
  assert  FN == 0, that is no genuine slot-4 acquisition is missed
  observe the marker cardinalities of T0E, T0P, T0Q and T0S
  observe @r10, @rbx, sequences[4] and live enqueuePos at each accepted T0Q
  observe whether live enqueuePos ever reads other than 5 at an accepted T0Q, which is the only
  evidence that can falsify FN == 0 after the fact, and which is retained for that purpose

  if SEQUENCE_PUBLISHED is absent while CAS_POST is present N times, attribute it to the
  section 4 residual, not to a capture defect

  N  =  number of queue instances reaching position 4,  0 <= N <= 4
  N is a SYMBOL.  the acceptance is NOT  observed N == 4.  4/116 is an observation and is not
  used as a specification value anywhere.
```

## 10. Current state

```text
scope                             RESOLVED, F-C
section 2 reading                 CONFIRMED by the Owner, CLOSED
C-a                               corrected, guard and diagnostic counts separated
authorization                     NOT YET ISSUED
CDB execution 0                  runtime NOT AUTHORIZED    build NOT AUTHORIZED
.cdb created 0                   .cdb corpus 15 unchanged
baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no source / test / CMake / harness edit    no predicate implemented    no repair proposed
```

```text
THE ONLY GATE REMAINING BEFORE ANY EDIT IS THE OWNER'S EXPLICIT ISSUANCE OF THE
IMPLEMENTATION AUTHORIZATION.  the request is complete and every confirmation point in it is
closed.  no .cdb edit, no execution, no build is performed until that issuance.
```
