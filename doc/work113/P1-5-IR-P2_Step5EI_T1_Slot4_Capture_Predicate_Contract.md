# Step5EI — T-1 Slot-4 Capture Predicate Contract

```text
gate                = New Capture Predicate Contract.  A CONTRACT, not an authorization.
mode                = READ-ONLY.  No .cdb edit.  No runtime.  No build.  No source modification.
execution           = 0
authority           = ConvoPeq.md  5,535,334 B
                      SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                      verified by direct read.  ConvoPeq(4).md does not exist, not used.
inputs              = Step5EH D2-C closure checklist, complete
status              = DRAFTED.  Implementation Authorization NOT requested.
```

## 1. Target and capture identity

```text
  TARGET            slot-4 acquisition of DeferredDeletionQueue

  CAPTURE IDENTITY  @rbx == 4
                    AND
                    dwo(@r11 + 0x30010) == @r10

  @r10              diagnostic observable only.  NOT a position conjunct.
                    at T0Q, r10d == pos, per the Step5EE correction.

  @rbx              slot identity, pos & 0xFFF
                    producer-local, derived at 0x141f9c560 and 0x141f9c563
                    CAS-success conditional, see section 4

  sequences[slot]  stores ABSOLUTE position values, from the authority at lines 62015, 62078,
                   62127.  the capture-time free condition is therefore

                       sequences[4] == @r10

                   and NOT the literal  sequences[4] == 4.  the literal is correct only in the
                   first 4096 cycle, and T-1 was selected precisely because slot 4 RECURS.

  live enqueuePos   diagnostic observable only, retained in the acceptance payload as
                    dwo(@r11 + 0x34000) L1.  never a capture conjunct.

  T0S              UNCHANGED.  byte-identical.  its conjuncts, its  r $t0 = 1  write and its
                    $t1..$t19 initialisation are not touched by this contract.  its live
                    enqueuePos conjunct is the accepted Step5EB section 4 residual, out of scope.

  T1 / S7          UNCHANGED.  no T1 site and no S7 site is a target of this contract.  the two
                    malformed T1 sites 0x1f9ce04 and 0x1f9cfb0 keep their existing content.
                    this contract adds no T1 predicate, no S7 anchor and no reader-side change.
```

## 2. The byte-exact T0Q form this contract fixes

Parent for the next baseline is Baseline-3, resolved by SHA-256 at edit time.

```text
  BEFORE, Baseline-3
  .echo C3_DG_T0Q_HIT ; .if (@r10 == 5) { .if (@rbx == 4) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_POST; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc }

  AFTER, fixed by this contract
  .echo C3_DG_T0Q_HIT ; r ; .echo C3_DG_T0Q_EP0 ; .echo C3_DG_T0Q_WIN_END ; .if (@rbx == 4) { .if (dwo(@r11+0x30010) == @r10) { .echo C3_T0_CAS_POST; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc }
```

```text
  the four edits, enumerated
    1  insert   r ; .echo C3_DG_T0Q_EP0 ; .echo C3_DG_T0Q_WIN_END ;
              immediately after the C3_DG_T0Q_HIT marker, that is OUTSIDE the guard
    2  remove   the .if (@r10 == 5) conjunct and one ladder level.  @r10 is withdrawn as a
              position conjunct at RDG-REOPEN-D1, so it is REMOVED, not relocated.
    3  change   dwo(@r11+0x30010) == 4   ->   dwo(@r11+0x30010) == @r10
    4  remove   the  r;  from inside the acceptance branch, since the dump is now unconditional

  guard .if count   3 -> 2
  ladder levels     3 -> 2
  the address +0x30010 is UNCHANGED.  only the comparison operand changes.
  with @rbx == 4 already a conjunct, (@rbx & 0xfff) * 4 == 0x10, so the general form
  dwo(@r11 + 0x30000 + ((@rbx & 0xfff) * 4)) == @r10  reduces to this one.
```

## 3. Q4 cardinality, in the general form

```text
  E_i        number of successful enqueues on queue instance i, within the measurement window
  K_i        max(0, floor((E_i - 4) / 4096) + 1)

             a slot-4 acquisition on instance i can only occur at a position
                 p = 4 + 4096k,  k = 0, 1, 2, ...
             and the instance reaches p only if enqueuePos advanced to p.

  captures_i  <=  K_i
  captures_i  =  K_i   if and only if the free test passed and the CAS succeeded at every
                        one of those positions

  GENERAL ACCEPTANCE
      0  <=  captures  <=  SUM over instances i of K_i

  SPECIAL CASE, and 0..4 appears ONLY here
      if E_i <= 4100 for every i, that is no instance completes two full cycles, then K_i <= 1
      for every i, and therefore

          0  <=  captures  <=  (number of instances that reached slot 4)  <=  4

  THE GENERAL SPECIFICATION IS 0 <= captures <= SUM K_i.
  0..4 IS NOT THE GENERAL SPECIFICATION AND MUST NOT BE WRITTEN AS ONE.
```

### 3.1 E_i is observable with no new instrumentation

```text
  T0E emits an UNCONDITIONAL register dump, which includes rcx, and T0E fires once per enqueue.
  counting T0E hits per distinct rcx yields E_i directly.  that is the method that produced the
  21 / 13 / 41 / 41 per-queue figures in Step5DP.

  the resulting check is NOT self-validating.  the prediction uses the source-derived recurrence
  K, and E_i is an input measured independently of the capture, so the prediction can fail in
  both directions:

      captures < SUM K_i    the predicate is missing acquisitions
      captures > SUM K_i    the predicate is over-firing
```

## 4. FN criterion, structural

```text
  T0Q is reachable ONLY through  je 0x141f9c586  on the cmpxchg success branch.  a CAS loser
  falls to 0x141f9c58b, jumps to 0x141f9c560, and re-enters the loop.  it never reaches T0Q.

  therefore, at T0Q,   @rbx == 4   entails   the CAS for the position whose slot index is 4
                                      was WON.

  and every CAS win falls through to T0Q, at which moment rbx == pos & 0xFFF.

  FN = 0 follows STRUCTURALLY.  no run is needed to establish it.  a run can only add contrary
  evidence.

  corollary   captures < K_i is not an FN.  it is evidence that the free test or the CAS
              declined an occurrence, and section 5 is what makes that class distinguishable.
```

## 5. capture and publication are separate quantities

```text
  capture cardinality        = SUM captures_i,  a function of K_i.  may exceed 1.

  T0S handoff / publication  <=  1 per run,  single-shot at $t0.

  capture #1       may carry the T0Q -> T0S handoff
  capture #2..#N   are complete T0Q captures and do NOT require a handoff

  this is the Step5EA section 4 BC restatement, preserved unchanged.  this contract does NOT
  extend to firing T0S more than once, and does NOT alter the $t0 latch.
```

## 6. Implementation scope, per site

```text
  T0Q   CHANGED.  the four edits of section 2.  this is the only site this contract modifies.

  T0E   NOT a target of this contract.  no change.  its unconditional dump is retained as the
        provenance of E_i, which section 3.1 depends on.

  T0P   NOT a target of this contract.  its sequence conjunct  dwo(@r11+0x30010) == 4  is NOT
        generalised to @r10 in this contract.  see section 7.

  T0S   BYTE-IDENTICAL.  its conjuncts, its  r $t0 = 1  write, and its $t1..$t19 initialisation
        are all unchanged.  its live enqueuePos conjunct is the accepted Step5EB section 4
        residual and stays out of scope.
```

## 7. Explicitly OUT of this contract

```text
  T0P sequence conjunct generalisation        OPEN, diagnostic-consistency sub-decision.
                                               NOT part of this contract.  its
                                               dwo(@r11+0x30010) == 4 stays as it is, which
                                               makes C3_T0_CAS_PRE a first-cycle-only
                                               diagnostic.  accepted for now.

  T0S live enqueuePos conjunct                OPEN, accepted residual.  NOT part of this contract.

  the two malformed T1 sites                  OUT OF SCOPE.  0x1f9ce04 and 0x1f9cfb0 retain
                                               their dd( misuse and their malformed address
                                               expression.

  65 versus 116 hits                          NON-COMPARABLE.  the Step5ED run ended at the
                                               T1A hit.

  S7_READER / S7_READER_SLOT / minReaderEpoch(T1) / Case A-D    OUT OF SCOPE.

  entry.type as a predicate operand           OUT OF SCOPE.  it becomes OBSERVABLE as a
                                               by-product of the unconditional dump, because
                                               0x141f9c59a reads [rsp+0x28], but it is not
                                               used by this predicate.
```

## 8. A SYNTAX RISK this contract carries, which CR-3 must not wave through

```text
  the contract requires  dwo(@r11+0x30010) == @r10,  a display-function result compared
  against a pseudo-register.

  PRECEDENT SURVEY over the whole frozen .cdb corpus, 16 files

      PROVEN at the four T0 sites
          dwo(...) == <literal>      T0E 2   T0P 2   T0Q 1   T0S 2
          @reg    == <literal>      T0E 1   T0P 1   T0Q 2   T0S 2
          .if (dwo( ... ))  is used at all four T0 sites

      NOT PRESENT at the four T0 sites
          dwo(...) == @reg          0 occurrences

      PRESENT elsewhere in the corpus
          dd(...) == (@$t10+1)      0x1f9cd59  x10
                                    0x1f9ce04  x10

      and 0x1f9ce04 is one of the two KNOWN-MALFORMED T1 sites.

  THEREFORE   the shape has precedent, but no VERIFIED precedent.  CR-3 cannot be satisfied by
  analogy and must be satisfied by a static syntax validation step, or recorded as an
  accepted unverified risk.

  MITIGATION, and it is a real one
      a syntax error in a bp command means cdb reports it and DOES NOT set the breakpoint.
      the run would then show ZERO C3_DG_T0Q_HIT markers.

      that is DISTINGUISHABLE from a predicate failure, which shows C3_DG_T0Q_HIT present with
      C3_T0_CAS_POST absent.

      so the risk is not silent.  it costs one run to discover, which is why a static check is
      preferred, and why the unconditional dump and the WIN_END marker matter: they give a parse
      boundary so a missing payload is visible.
```

## 9. A limitation of this contract, recorded rather than expanded

```text
  the contract moves the REGISTER dump unconditional, as specified.  it does NOT move the
  dd operand probes out of the acceptance branch.

  consequence   on a NON-ACCEPTING T0Q hit, the log will carry r10, rbx and r11, but NOT the
                memory value of sequences[4].

                the predicate's three operands are therefore not all observable on failure.  a
                failure attributable to the sequence conjunct cannot be distinguished from a
                failure of the slot conjunct using this dump alone.

  this is recorded as a limitation of the contract as specified.  widening the payload to include
  the unconditional dd probes would close it, but that is a scope change and is NOT taken here
  without a decision.
```

## 10. Contract Review checklist, CR-1 to CR-14

To be executed read-only against this contract, before any Implementation Authorization.

```text
  CR-1    the capture predicate does NOT contain  @r10 == 4
  CR-2    the capture predicate does NOT contain a literal  sequence == 4
  CR-3    the sequence comparison uses  @r10,  and the shape's syntax is VALIDATED STATICALLY
          rather than by analogy.  see section 8.
  CR-4    @rbx == 4 is retained
  CR-5    the T0Q register dump is unconditional, outside the guard
  CR-6    the acceptance branch's former  r  is removed by the contract
  CR-7    the FN = 0 justification is the CAS-success topology, je 0x141f9c586
  CR-8    K_i is stated as the general recurrence  max(0, floor((E_i-4)/4096)+1)
  CR-9    0..4 appears ONLY as the E_i <= 4100 special case, never as the general specification
  CR-10   T0S is byte-identical
  CR-11   T0P generalisation is recorded as OPEN and out of scope
  CR-12   T1 and S7 are unchanged
  CR-13   the malformed T1 sites are out of scope
  CR-14   the contract carries no .cdb edit, no runtime authorization and no build authorization
```

## 11. State

```text
Step5EI        DRAFTED.  contract fixed, review not yet executed.
CR-1 .. CR-14  NOT EXECUTED
Implementation  FORBIDDEN
cdb edit        0
runtime         0
build           NOT INVOKED
source          unchanged
Baseline-3      unchanged, FAILED DESIGN, retained as evidence
baselines 7/7 + Baseline-3 = 8/8 intact    .cdb 16    git delta 12, all pre-existing
residue 0
```

```text
next, in the Owner's fixed order
    Contract Review, read-only, CR-1 through CR-14
    Implementation Authorization, only if all fourteen PASS
    ... and so on.

OPEN items this contract deliberately does not absorb
    T0P sequence conjunct generalisation, diagnostic consistency
    the sequence operand not being observable on a failing hit, section 9
    the two malformed T1 sites
    the 65 versus 116 hit non-comparability
```
