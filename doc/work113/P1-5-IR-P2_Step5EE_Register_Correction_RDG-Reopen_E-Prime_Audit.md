# Step5EE — Step5EA-0 §3 CORRECTED, RDG-3-2 and RDG-4-2 RE-OPENED, E' audited read-only

```text
gate                = design correction and re-opening.  NOT an implementation.
mode                = READ-ONLY.  No .cdb edit.  No source / test / CMake / harness edit.  No build.
                      NO RUNTIME.  Baseline-3 is NOT re-run.
execution           = 0
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
evidence base       the Step5ED runtime log, and the disassembly of
                      build/Release/AudioEngineHarness.exe
                      SHA-256 E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
status              STEP5ED FAIL stands.  Step5ED's FN = 2 is CORRECTED here.
```

## 1. Correction to Step5ED — FN = 2 was an over-claim

The Owner is right that `CAS_PRE = genuine acquisition` overstates the evidence. T0P sits **before**
the CAS. Re-derived from the log, read-only:

```text
  C3_T0_CAS_PRE           2    attempts that reached the free-slot test at position 4
                               both dumped r10 = 4 and rbx = 4
  C3_T0_CAS_POST          0    captures taken
  C3_T0_SEQUENCE_PUBLISHED 1   post-publish proof, so at least 1 genuine acquisition

  wins  is bounded below by the publication proof and above by the attempt count
       1  <=  wins  <=  2
  FN   =  wins - captures  =  wins - 0

  =>  FN >= 1  PROVEN
  =>  FN  =  2  NOT ESTABLISHED
```

```text
CORRECTED   Step5ED recorded  FN = 2.        WITHDRAWN.
            the correct statement is  FN is in {1, 2}, with FN >= 1 proven.

            the acceptance criterion FN == 0 FAILS either way, so the verdict is unchanged.
            only the magnitude was overstated, and the over-claim is withdrawn here.
```

### 1.1 Why the exact count could not be obtained, and it is a diagnosability defect

```text
  the direct method would be  count T0Q hits with r10 == 4,  because T0Q is reachable only
  through the je on the cmpxchg success branch, so a T0Q hit IS a CAS win at its position.

  that method FAILED, because the T0Q site has no unconditional register dump:

      .echo C3_DG_T0Q_HIT ; <guard> { .echo C3_T0_CAS_POST; r; dd ... ; gc }

  the  r  is INSIDE the acceptance branch.  on a non-accepting hit T0Q emits nothing but the
  marker, so no register evidence survives at that site.  r10 was None at all 65 T0Q hits.

  contrast T0E, which does dump unconditionally:
      .echo C3_DG_T0E_HIT ; <two diagnostic if/else> ; r ; .echo EP0 ; dq window ;
      .echo WIN_END ; <guard>

FINDING, and it becomes a REQUIREMENT for any successor predicate

  a T0Q capture that FAILS leaves no evidence at the T0Q site.
  any predicate whose runtime validation depends on T0Q register values is therefore
  UNVALIDATABLE with the current script shape.

  this is not the cause of the capture failure.  it is the reason the failure could not be
  characterised precisely from the log alone.
```

## 2. Step5EA-0 §3 — CORRECTED register analysis

The original text read:

```text
  WITHDRAWN  "on the success path r10d is NOT reloaded.  0x141f9c5cd incl %r10d makes it
              pos + 1.  therefore at T0Q and T0S, r10d = pos + 1, and @r10 == 5 is
              satisfiable and means exactly this call advanced enqueuePos from 4 to 5."
```

### 2.1 The instruction order, which is what I misread

```text
  0x141f9c555  movl 0x34000(%rcx), %r10d    r10d = pos
  0x141f9c560  movl %r10d, %ebx             ebx  = pos
  0x141f9c563  andl $0xfff, %ebx             ebx  = pos & 0xFFF, the slot index
  0x141f9c569  movl 0x30000(%r11,%rbx,4),%eax  eax = sequences[slot]
  0x141f9c571  subl %r10d, %eax              eax  = seq - pos = diff
  0x141f9c574  jne 0x141f9c58d               free-slot test
  0x141f9c576  leal 0x1(%r10), %ecx          ecx  = pos + 1
  0x141f9c57d  lock                          <-- T0P
  0x141f9c57e  cmpxchgl %ecx, 0x34000(%r11)  CAS enqueuePos : pos -> pos+1
  0x141f9c586  je 0x141f9c59a                 SUCCESS
  0x141f9c588  movl %eax, %r10d              FAILURE path, r10d = pos, retry
  0x141f9c591  movl 0x34000(%r11), %r10d     diff > 0 path, reload
  0x141f9c59a  movzbl 0x28(%rsp), %eax       <-- T0Q   r10d is STILL pos
  0x141f9c59f  leaq (%rbx,%rbx,2), %rcx     entry address arithmetic
  0x141f9c5a6  movb %al, 0x18(%r11,%rcx,8)  entry.type
  0x141f9c5c8  movq %r9, 0x10(%r11,%rcx,8)  entry.epoch
  0x141f9c5cd  incl %r10d                    r10d = pos + 1
  0x141f9c5d2  movl %r10d, 0x30000(%r11,%rbx,4)  publish
  0x141f9c5da  movq 0x8(%rsp), %rbx          <-- T0S   r10d = pos + 1
```

```text
ORDER   0x59a (T0Q)   <   0x5cd (incl)   <   0x5da (T0S)

  at T0Q   r10d == pos      == 4
  at T0S   r10d == pos + 1  == 5
```

### 2.2 The corrected statement

```text
  CORRECT   at T0Q, r10d == pos.
            at T0S, r10d == pos + 1.

  THEREFORE  @r10 == 5  is UNSATISFIABLE at T0Q for a slot-4 acquisition.
            @r10 == 4  is the producer-local position conjunct at T0Q.
            @r10 == 5  remains correct at T0S.

  the original claim was wrong in one place only.  its companion statement, that 0x141f9c591
  reloads enqueuePos only on the diff > 0 retry path, was CORRECT, and it is what should have
  blocked the wrong conclusion.  I had the ordering in front of me and misread it anyway.
```

### 2.3 Runtime corroboration

```text
  r10 observed at C3_DG_T0E_HIT   0x1        consistent: T0E precedes the load at 0x1f9c555
  r10 observed at C3_DG_T0P_HIT   0x4  x2    the two CAS_PRE acceptances, pos = 4
  r10 observed at C3_DG_T0S_HIT   0x5  x1    the publication, pos + 1 = 5
  r10 at C3_DG_T0Q_HIT            not observable, no unconditional dump.  see 1.1
```

## 3. RDG-3-2 — RE-OPENED

```text
status   RE-OPENED.  the previous ruling is NOT maintained.

what it assumed
    T0Q capture identity =  @r10 == 5 , @rbx == 4 , sequences[4] == 4
    and certified @r10 == 5 as producer-local, satisfiable and immune to contention.

what falsified it
    @r10 == 5 is unsatisfiable at T0Q.  the certification rested on the register analysis
    corrected in section 2, which was wrong.

the part that survives
    the RDG-3-2 PRINCIPLE is untouched and was not what failed:
        producer-local or owner-exclusive state  ->  capture identity
        shared live state                        ->  diagnostic observable only
    @r10 and @rbx ARE producer-local.  sequences[4] IS owner-exclusive.
    live enqueuePos remains correctly excluded.

    the defect is the VALUE, not the classification.
```

## 4. RDG-4-2 — RE-OPENED

```text
status   RE-OPENED.  Candidate E is INVALID, not merely superseded.

  Candidate E as ruled
      @r10 == 5  AND  @rbx == 4  AND  dwo(@r11+0x30010) == 4
      INVALID, because its first conjunct cannot hold.

  Candidate E'
      @r10 == 4  AND  @rbx == 4  AND  dwo(@r11+0x30010) == 4
      CANDIDATE ONLY.  NOT a ruling, NOT adopted, NOT to be treated as a settled design.

  Step5EB  the implementation contract is SUPERSEDED.  it specified @r10 == 5.
  Step5EC  Baseline-3 is a FAILED DESIGN.  it stays in the corpus, unchanged, as the evidence.
```

## 5. E' read-only semantic audit. A, B, C. NO RULING.

### A. `@r10 == 4`

```text
  claim under test   @r10 == 4 identifies a slot-4 acquisition at T0Q

  derivation         0x141f9c555  r10d = pos, loaded once on entry
                     0x141f9c57e  the CAS writes pos+1 to enqueuePos, a SHARED location
                     0x141f9c586  je to T0Q on success
                     0x141f9c59a  T0Q, r10d untouched since the load, therefore r10d == pos
                     0x141f9c591  the only reload of r10d is on the diff > 0 path, which is a
                                   different control-flow path that does not reach T0Q

  producer-local     YES.  r10d is a register.  no other thread can alter it.  no code path
                     reachable at T0Q writes r10 between the load and the trap.

  discriminating     it pins pos == 4, hence the acquisition is of position 4.

  runtime testable   NOT with the current T0Q shape.  see section 1.1.  validating this needs
                     an UNCONDITIONAL register dump at T0Q.

  VERDICT            sound as producer-local, and consistent with T0P's observed r10 = 4.
                     it is a CANDIDATE conjunct, not a ruling.
```

### B. `@rbx == 4`

```text
  derivation         0x141f9c560  ebx = r10d = pos
                     0x141f9c563  andl $0xfff, %ebx    ebx = pos & 0xFFF
                     0x141f9c569  sequences indexed by ebx
                     0x141f9c5a6  entry indexed by ebx, stride 48
                     0x141f9c5d2  publication indexed by ebx

  restore            0x141f9c5da  movq 0x8(%rsp), %rbx  restores the CALLER's rbx.  x86 traps
                     before the instruction executes, so at the T0Q trap rbx is still
                     pos & 0xFFF.  this is unaffected by the correction in section 2.

  producer-local     YES.  derived from r10d, held in a register.

  note               for pos in 0..4095, pos & 0xFFF == pos, so rbx == 4 iff pos == 4.  the
                     conjunct is therefore REDUNDANT with @r10 == 4, not independent.  it is
                     not wrong, and T0S already relies on it, but it adds no discrimination.
                     recorded so the redundancy is a decision, not an accident.

  VERDICT            sound, producer-local, and redundant with A.
```

### C. `sequences[4] == 4`

```text
  derivation         0x141f9c569  eax = sequences[pos & 0xFFF]
                     0x141f9c571  eax = seq - pos = diff
                     0x141f9c574  jne leaves the free-slot branch, so diff == 0 on the path
                     0x141f9c5d2  sequences[slot] = pos + 1, the publication

  for pos == 4       sequences[4] == 4 is the diff == 0 condition, i.e. the free-slot identity,
                     and it is still 4 at T0Q because the publication at 0x141f9c5d2 has not
                     executed.

  owner-exclusive    YES.  only the owner of slot 4 writes sequences[4], and the owner is the
                     thread that won the CAS, which is this thread.  a competing producer
                     claiming position 5 writes sequences[5].

  contention         IMMUNE.  a foreign producer cannot alter this word before this thread
                     publishes it.

  runtime testable   the payload already contains  dd @r11+0x30010 L1, but it is inside the
                     acceptance branch, so it is not observed on failure.  see 1.1.

  VERDICT            sound, owner-exclusive, contention-immune, and it is the conjunct that
                     carries the free-slot semantics rather than merely the position.
```

### D. what the audit adds beyond substituting the value

```text
  REQUIRED, and not previously identified

  T0Q must gain an UNCONDITIONAL diagnostic payload, in the shape T0E already has:

      .echo C3_DG_T0Q_HIT
      <unconditional register dump or equivalent>
      <unconditional probe of the predicate operands>
      .echo C3_DG_T0Q_WIN_END
      <the guard>

  without it, a failing T0Q leaves no evidence, and neither @r10 == 4 nor sequences[4] == 4 can
  be validated at runtime.  this is a REQUIREMENT on any successor, and it was not part of the
  Step5EB contract.

  this is a structural change to the T0Q line, larger than a one-conjunct substitution, and it
  is therefore a design decision for the re-opened gate, not something to assume.
```

### E. explicitly NOT decided here

```text
  @r10 == 4 is NOT ruled.  it is audited and found sound as a candidate conjunct.
  E' is NOT adopted.  live enqueuePos is NOT returned to the predicate.
  the unconditional-diagnostic requirement is NOT adopted; it is raised as a requirement.
  no conjunct count, no marker set and no contract wording is fixed.
```

## 6. Separations the Owner required

```text
6.1  "CAS_PRE = genuine acquisition" is WITHDRAWN as a phrasing.
     T0P is a pre-CAS site.  CAS_PRE means an ATTEMPT was observed.  the attempt-to-win
     conversion is not available from the markers, and section 1 keeps the bound rather than
     collapsing it.

6.2  T0S is NOT brought into the capture failure.
      CAS_POST = 0 and SEQUENCE_PUBLISHED = 1 is the INVERSE of the Step5EB section 4 case,
      which was SEQUENCE_PUBLISHED < CAS_POST.  T0S behaved as an independent, pre-existing
      handoff path and its residual is NOT added to the repair scope.

6.3  the T1A early exit is recorded as a SEPARATE issue.
      the run ended after 65 T0E hits because T0A fired and the script's single g then quit.
      65 must not be compared with 116.  this is unrelated to the predicate correction.

6.4  the 0x1f9cfb0 malformed dd( is NOT promoted to a cause of the capture failure.
      it is the runtime manifestation of the already-recorded, deliberately out-of-scope
      malformed T1 site.

6.5  the vacuous all() PASS is not reused as acceptance evidence anywhere.
```

## 7. State

```text
Step5EA-0 original register ruling   INVALIDATED, corrected in section 2
Step5EA-0 surviving statements       the diff>0-only reload, rbx = pos & 0xFFF, the
                                      owner-exclusive property of sequences[slot]
RDG-3-2                              RE-OPENED.  the principle survives, the value does not.
RDG-4-2                              RE-OPENED.  Candidate E INVALID.
Candidate E                          INVALID
Candidate E'                         CANDIDATE ONLY, audited A / B / C / D, NOT ruled
Step5EB contract                     SUPERSEDED
Step5EC Baseline-3                   FAILED DESIGN, retained unchanged as evidence
Step5ED runtime                      FAIL.  FN >= 1 proven, FN = 2 WITHDRAWN, FN in {1,2}
Implementation Authorization          exhausted
Runtime Authorization                 exhausted, no retry
.baseline-3 re-run                    NOT DONE and NOT authorised
cdb edit                             NONE
runtime                              NONE in this gate
build                                NOT INVOKED
```

## 8. What the re-opened gate must decide, in order

```text
1  whether T0Q's producer-local position conjunct is @r10 == 4.  audited sound, not ruled.
2  whether @rbx == 4 is retained given that it is redundant with (1).  retaining it is a
   decision, not a default.
3  whether the unconditional diagnostic payload required by section 5.D is adopted.  it is a
   structural change to the T0Q line and it is the price of runtime validability.
4  only then, a new contract, a new Implementation Authorization, and a new Runtime
   Authorization.

OPEN-3 status
  T0P and T0Q conjunct rates are now measured FOR THIS RUN ONLY, at 65 T0E hits.
  T0P  2 of 65 satisfied its conjuncts
  T0Q  0 of 65 produced an accepted capture
  the 116-hit runs are not comparable with this 65-hit run.
```
