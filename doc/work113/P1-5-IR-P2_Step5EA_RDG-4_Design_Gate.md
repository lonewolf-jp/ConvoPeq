# Step5EA — RDG-4 Design Gate

```text
gate                = Repair Design Gate, stage 4 of 4.  DESIGN.
mode                = READ-ONLY / OWNER DECISION.  No CDB run, no .cdb edit, no source / test /
                      CMake / build / harness change.  No runtime measurement.
                      No predicate implemented.
execution this gate = 0
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
disassembly         = build/Release/AudioEngineHarness.exe
                      SHA-256 E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
prerequisite        = Step5EA-0 T0→T1 Handoff Audit, CLOSED
```

## 1. RDG-4-1 — handoff design. OWNER RULED.

```text
RDG-4-1  CLOSED

  $t0 is monotonic and write-once; the only write in the whole corpus is $t0 = 1.
  $t0 == 1 is therefore NOT a one-shot consumption mechanism and cannot prevent re-entry.
  the actual re-entry prevention is $t17, which seals the S7-anchor / T2-return race.

  re-firing T0S would zero $t11..$t19, and seven downstream sites test $t12 == 1 and
  $t16 == 1, so it would destroy the accumulated T1 and S7 state.

  RULING
    T0S remains single-shot handoff.
    multi-shot capture is NOT to be realised by repeating T0S.
    multi-shot capture must terminate at T0Q.

  design space as ruled
    T0S multi-shot                        NO-GO
    $t0 reuse for multi-handoff           NO-GO
    T0S fires per instance, state restored elsewhere
                                        NO-GO, changes the handoff semantics
    keep state, report a different instance
                                        NO-GO, falsifies instance identity
    keep the current one-shot handoff     GO
```

## 2. RDG-4-2 — formal capture predicate. OWNER RULED.

```text
RDG-4-2  CLOSED

  formal capture predicate = Candidate E

    T0Q capture identity
      @r10 == 5                    producer-local,   r10d = pos + 1
      @rbx == 4                    producer-local,   rbx = pos & 0xFFF
      dwo(@r11 + 0x30010) == 4     owner-exclusive,  sequences[4]

    NOT part of the predicate
      dwo(@r11 + 0x34000) == 5     contention-sensitive live shared state

  Candidate C is recorded as a semantically equivalent representation and is NOT separately
  selected.  under the mandated conjunct @rbx == 4,  (4 & 0xfff) * 4 == 0x10, so C reduces to E
  identically.  the difference is traceability only, and E names the audited quantity directly.
```

## 3. RDG-4-3 — live enqueuePos. OWNER RULED.

```text
RDG-4-3  CLOSED

  KEEP as diagnostic payload.  NOT a capture conjunct.

  T0Q capture predicate   @r10 == 5  AND  @rbx == 4  AND  dwo(@r11 + 0x30010) == 4
  T0Q diagnostic payload   dwo(@r11 + 0x34000)

  the RDG-3 principle is preserved verbatim:
      producer-local or owner-exclusive state  ->  capture identity
      shared live state                        ->  diagnostic observation

  the retained diagnostic has a concrete payoff.  if a future run shows a T0Q capture at which
  live enqueuePos read something other than 5, that is the contention condition, and it is the
  only evidence that can falsify FN == 0 after the fact.
```

## 4. RDG-3-4 versus RDG-4-1 — RESOLVED. RDG-3-4's BC is WITHDRAWN.

This was flagged in Step5EA-0 section 5 as something the Owner must rule on rather than inherit.
It was concrete, not hypothetical. The Owner has ruled.

```text
THE CONFLICT, as it stood

  RDG-3-4 as ruled
    BC  =  every captured T0Q has a corresponding T0S for the same slot-4 acquisition
    i.e.  forall capture, exists corresponding T0S

  RDG-4-1 as ruled
    T0S is single-shot, so T0S fires at most once
    T0Q has no $t0 guard, so T0Q fires once per instance, that is N times

  and N = 4 in BOTH measured runs, so the old BC was false for 3 of 4 captures on every run
  ever measured.
```

```text
OWNER RULING

  RDG-4-1 = S is CORRECT and is maintained.
  the old RDG-3-4 BC is INVALID and is WITHDRAWN.

  the error in RDG-3-4 was that it identified the completion condition of a T0Q CAPTURE with
  the T0S HANDOFF.  these are two different lifecycles and must not be unified.
```

### 4.1 The two lifecycles, separated

```text
ACQUISITION LIFECYCLE
  T0E   acquisition attempt observed
  T0Q   CAS success, slot ownership acquired      <-- this IS the capture event
  ...   entry write
  T0S   entry publication complete                 <-- belongs to the handoff, not the capture

HANDOFF LIFECYCLE
  T0S
  $t0 = 1
  T1
  S7
```

### 4.2 New BC

```text
  OLD BC   WITHDRAWN.  forall captured T0Q, exists corresponding T0S

  NEW BC   T0S must NOT be used as the completion witness for every T0Q capture.

  T0Q cardinality   N,  0 <= N <= 4      the acquisition count
  T0S cardinality   H,  H <= 1            the handoff count
```

### 4.3 The corrected classification. Captures 2 to N are NOT incomplete.

```text
  T0Q #1  ->  entry write  ->  T0S  ->  $t0 = 1  ->  T1 / S7
  T0Q #2  ->  entry write  ->  no T0S
  T0Q #3  ->  entry write  ->  no T0S
  T0Q #4  ->  entry write  ->  no T0S

  T0Q -> entry write -> T0S  is the internal sequence of ONE handoff episode.

  captures #2 to #N are NOT incomplete captures.  T0Q is itself the acquisition event, so each
  capture is complete on its own terms.  what they lack is the handoff, and the handoff is
  single-shot by RDG-4-1, not by any defect in the capture.

  therefore  N = 4, H = 1  is consistent, and the measured N = 4 is NOT a violation.
  it is instead the evidence that the old RDG-3-4 BC demanded a cardinality coupling that the
  protocol never provided.
```

### 4.4 Corrected observation table, superseding RDG-3-4's

| observation | verdict |
|---|---|
| T0E only | attempt observed, no capture |
| T0Q only | complete capture, no handoff, handoff not consumed |
| T0Q → T0S | complete capture WITH the handoff episode |
| T0S only | cannot occur, since T0S is reached only inside a T0Q's own instruction sequence |

## 6. RDG-4 status

```text
RDG-4-1  handoff design                 CLOSED   Owner ruled, S variant
RDG-4-2  formal capture predicate       CLOSED   Owner ruled, Candidate E
RDG-4-3  live enqueuePos diagnostic     CLOSED   Owner ruled, KEEP
         BC restatement                 CLOSED   Owner ruled, section 4.  RDG-3-4's old BC
                                                WITHDRAWN, lifecycles separated

RDG-4    CLOSED
```

```text
CONSEQUENTIAL RULING CHANGE, recorded here and flagged in Step5DZ

  RDG-3-4's BC  =  every captured T0Q has a corresponding T0S     WITHDRAWN, INVALID
  RDG-3-4's observation table                                       SUPERSEDED by section 4.4

  RDG-3-1, RDG-3-2 and RDG-3-3 are UNAFFECTED.  the capture marker, the FN/FP criteria and N
  all survive the restatement unchanged.  only the bracket-completeness criterion was wrong,
  because it had coupled capture cardinality to handoff cardinality.
```

## 7. What is settled and what is still outside

```text
SETTLED, and carried into the implementation request that has NOT been made
  capture marker            T0Q / C3_T0_CAS_POST
  capture identity          @r10 == 5  AND  @rbx == 4  AND  dwo(@r11 + 0x30010) == 4
  live enqueuePos           diagnostic only, retained
  handoff                   T0S single-shot, $t0 = 1 written once, $t11..$t19 initialised once
  FN                        0, structurally
  FP                        0, structurally
  N                         symbol, 0 <= N <= 4
  terminology               ownership acquired / entry publication complete
  epoch literal             excluded from the target by Q3 = a; its DELETION is not authorised

STILL OUTSIDE, each needing its own authorisation
  implementation of any change        Implementation Authorization, NOT REQUESTED
  any .cdb edit                       Implementation Authorization, NOT REQUESTED
  runtime measurement                 Runtime Authorization, NOT REQUESTED
  T0P / T0Q / T0S conjunct hit rates  OPEN-3, UNMEASURED, deliberately not pursued
  T1 site repairs, 0x1f9ce04 and 0x1f9cfb0
                                     still malformed with 2 x dd( each, still unverified, and
                                     deliberately kept out of scope
```

```text
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no predicate implemented    no .cdb edited    no repair proposed    no build invoked
```
