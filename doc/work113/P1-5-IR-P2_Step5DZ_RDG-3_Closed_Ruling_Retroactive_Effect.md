# Step5DZ — RDG-3 CLOSED, and the ruling's retroactive effect on RDG-2

```text
gate                = Repair Design Gate, stage 3 of 4.  CLOSURE RECORD.
mode                = READ-ONLY / OWNER DECISION.  No CDB run, no .cdb edit, no source / test /
                      CMake / build / harness change.  No predicate implemented.
execution this gate = 0
```

## 1. RDG-3-2 — OWNER RULED

```text
Genuine acquisition     CAS pos = 4 -> 5 succeeded
Capture FN              a genuine acquisition exists and the T0Q capture marker does not hold
FN acceptance           0 REQUIRED
contention-induced FN   NOT PERMITTED

@rbx == 4               usable as acquisition identity      producer-local, 0x141f9c563
@r10 == 5               usable as acquisition identity      producer-local, 0x141f9c5cd
live enqueuePos == 5    EXCLUDED from the required conjuncts
live enqueuePos         observable as diagnostic information only
```

The ruling states a principle rather than a patch:

```text
  producer-local state expressing CAS success      ->  capture identity
  shared live state that can change after the CAS   ->  diagnostic observable
                                                      NEVER a capture acceptance predicate
```

This is accepted as the governing rule for the T0 phase. It is a semantic criterion, not an
implementation convenience, and it is the right distinction: `enqueuePos` is a shared cache line
that any producer may advance, so a predicate on it describes the world at trap time rather than
what this thread did.

## 2. FN = 0 is achievable under the ruled principle. Verified, not assumed.

The capture predicate the principle admits is:

```text
@r10 == 5                     producer-local   r10d = pos + 1, 0x141f9c5cd
@rbx == 4                     producer-local   rbx = pos & 0xFFF, 0x141f9c563
dwo(@r11+0x30010) == 4        owner-exclusive  sequences[4], written only by slot 4's owner,
                                             which is this thread
```

```text
interleaving analysis, one foreign producer winning position 5 at the worst moment

  @r10 == 5                  UNAFFECTED.  r10d is a register.  no other thread can alter it.
  @rbx == 4                  UNAFFECTED.  same, and it is derived from r10d.
  sequences[4] == 4          UNAFFECTED.  only the owner of slot 4 writes that word, and the
                                          owner is this thread.  a competing producer claims
                                          position 5, which is a different slot.

  => all three conjuncts are immune.  FN = 0 STRUCTURALLY, not statistically.
```

So the ruling is self-consistent: it demands FN = 0, and the predicate shape it admits delivers
FN = 0 by construction rather than by tolerance.

## 3. RETROACTIVE EFFECT — the RDG-2 candidate set is partly invalidated

This is the most important consequence and it is not a formality.

```text
RDG-2 compared A / B / C / D BEFORE the FN principle existed.  Applying the principle now:

CANDIDATE A   capture marker C3_T0_ENTRY at T0E.
              T0E's only positional operand is  dwo(@rcx+0x34000) == 4,
              which IS the live shared enqueuePos.
              => NON-COMPLIANT.  a candidate whose capture predicate is built on
                 contention-sensitive live shared state is now prohibited.

CANDIDATE B   capture marker C3_T0_CAS_POST at T0Q.
              T0Q's conjuncts as frozen include  dwo(@r11+0x34000) == 5,
              which IS the live shared enqueuePos.
              => NON-COMPLIANT AS SPECIFIED.  B satisfies everything else, and becomes
                 compliant only if that one conjunct is removed and @rbx == 4 is added.

CANDIDATE C   a generalisation of the sequences index.  orthogonal to the principle.
              => neither compliant nor non-compliant on this axis.  it was already
                 dominated on C7, and the ruling does not change that.

CANDIDATE D   already excluded on C1, C2 and C3.
```

```text
THEREFORE

  the RDG-2 ordering  B > A > C > D  is SUPERSEDED on compliance grounds.

  A is no longer a permissible fallback.  it was the only candidate that avoided the
  downstream conjuncts entirely, and that avoidance is exactly what disqualifies it.

  B is not discarded, but B-as-specified in RDG-2 is not the design the ruling admits.
  the admitted shape is a bracket whose T0Q conjunct set is
      @r10 == 5  AND  @rbx == 4  AND  dwo(@r11+0x30010) == 4
  with the live enqueuePos conjunct relocated to the diagnostic payload.

  that shape is NOT among A, B, C, D.  it is a candidate the ruling implies and RDG-2 never
  enumerated, because the principle that forces it did not exist when RDG-2 ran.
```

## 4. Consequences for the other sites, checked against the same principle

```text
T0E   dwo(@rcx+0x34000) == 4  is live shared state.
      RDG-3-4 already classes T0E as  acquisition attempt observed,  i.e. a DIAGNOSTIC stage.
      so the principle is satisfied by construction: T0E is not the capture marker, and its
      operand is diagnostic.  no change required, and none is implied.

      note: T0E sits at 0x1f9c550, BEFORE 0x1f9c555 which loads enqueuePos, so this thread has
      no local copy of pos at T0E.  a producer-local predicate is not available there even in
      principle.  this is why the capture marker cannot be T0E, independently of the ruling.

T0S   dwo(@r11+0x34000) == 5  is live shared state, and T0S sits about fifteen instructions
      later than T0Q, so its exposure to the same FN class is materially larger.
      T0S is the PUBLICATION marker under RDG-3-4, not the capture marker, so the principle
      does not bind it as a capture predicate.  it is additionally corroborated by @rbx == 4,
      which is producer-local.

      this is a second independent reason the Owner's RDG-3-1 choice is sound, and a further
      reason SEQUENCE_PUBLISHED would have been the worst available capture marker.
```

## 5. RDG-3 status

```text
RDG-3-1  capture marker         CLOSED   C3_T0_CAS_POST / T0Q
RDG-3-2  TP / FP / FN and N     CLOSED   FN = 0 required; contention-sensitive live enqueuePos
                                        prohibited as a required conjunct
RDG-3-3  N                      CLOSED   symbol, 0 <= N <= 4, not a constant
RDG-3-4  bracket completeness   CLOSED   ownership-to-publication bracket, BC = T0Q -> T0S

RDG-3    CLOSED
```

## 6. What RDG-4 must now decide, and what it must not assume

```text
RDG-4 INPUT, carried as established
  capture marker            T0Q / C3_T0_CAS_POST
  capture identity rule     producer-local or owner-exclusive state ONLY
  FN                        0, structurally, verified in section 2
  FP                        0, structurally, single-entry argument
  N                         symbol, 0 <= N <= 4
  terminology               ownership acquired / entry publication complete

RDG-4 MUST DECIDE
  1  single-shot or multi-shot.  T0S writes r $t0 = 1, which disables T0E and T0S for every
     later instance.  T0Q has no $t0 guard, so captures already scale with N.  the question is
     whether publication stays latched at 1, and whether the two cardinalities are permitted to
     differ.  RDG-3 established that they do differ; RDG-4 states whether that is intended.
  2  the candidate to adopt.  A and B are non-compliant as specified.  the admitted shape is
     enumerated in section 3 and must be compared, not assumed.
  3  whether live enqueuePos is retained in the diagnostic payload at T0Q.  the ruling permits
     it as diagnostic; it does not require it.

STILL OUTSIDE, unchanged
  deletion of the epoch literal          needs a separate Implementation Authorization
  any .cdb edit                          needs a separate Implementation Authorization
  runtime measurement                    needs a separate Runtime Authorization
  T0P / T0Q / T0S conjunct hit rates     OPEN-3, UNMEASURED
  candidate B's bracket conjuncts        OPEN-3, UNMEASURED, and now also superseded in form
```

```text
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no predicate implemented    no .cdb edited    no candidate adopted    no repair proposed
```
