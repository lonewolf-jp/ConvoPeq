# Step5EK-R1 — Contract Amendment Authorization Request / **DF-2 and DF-3 only**

```text
gate                = an authorization REQUEST.  NOT an authorization.  NOT an amendment.
mode                = NOT EXECUTED.  the R1 contract has NOT been modified.
subject             = P1-5-IR-P2_Step5EK-R1_Pre-Execution_Contract.md
                      19,268 B   SHA-256 9690F7C5C29BC283084FB242FD4970F5CA919730CFB42A5CA12812718F621C9F

precondition        = Step5EK-R1 Contract Review CLOSED, RR 18/18 PASS, 3 findings
                      DF-1 ACCEPTED / CLOSED by the Owner, no contract change
                      DF-2 and DF-3 adjudicated as CONTRACT AMENDMENT REQUIRED
                      Amendment 1 established and NOT to be touched

status              = AWAITING OWNER EXPLICIT AUTHORIZATION
```

## 0. DF-1 — ACCEPTED / CLOSED, recorded here and not acted upon

```text
  the Owner has adjudicated

      DF-1  ACCEPT / CLOSED.  documentation-only.
            the authority state after the Phase 3-B adjudication is already fixed by the
            subsequent records, and DF-1 is not a substantive defect that would make the R1
            execution judgement ambiguous.

      section 15 of the R1 contract is NOT changed on account of DF-1
      DF-1 does NOT block R1 execution

  this request therefore amends nothing for DF-1, and no DF-1 text appears in the amendment
  scope below.
```

## 1. Scope — two named regions, and nothing else

The contract is 19,268 B. The amendment touches **two** regions and leaves every other byte
alone. Both are located here so that the amendment is mechanical rather than interpretive.

```text
  REGION 1  DF-2   section 8, G conditions                    L236..L261
           an INSERTION is made.  G-1 through G-7 keep their existing text verbatim.
           nothing in G-1..G-7 is reworded, renumbered, reordered or removed.

  REGION 2  DF-3   section 9, AB-6                            L281..L282
           the two lines are REPLACED.  the condition id AB-6, its position in the AB-1..AB-8
           sequence, and its role as an immediate abort are unchanged.

  NOT TOUCHED, byte-identical
      section 0   the three-authorizations separation
      section 1   what R1 is and is not
      section 2   debuggee identity, the x64 requirement, the resolution rule, the excluded
                  AudioEngineHarness locations
      section 3   SHAPE A fixed, SHAPE B rejected
      section 4   cdb pinned to the production parser
      section 5   the invocation, the executable-last rule, the executable-never-runs clause
      section 6   the probe script, its order, the markers
      section 7   C1..C4, the substring trap, C2 and C3 both meaning parser-accepted
      section 8   G-1..G-7 text itself, as distinct from the insertion beside it
      section 9   AB-1..AB-5, AB-7, AB-8, and the ON ANY ABORT clause
      section 10  the no-retry rule and VOID-is-terminal
      section 11  artifacts and resolution discipline
      section 12  the prohibitions
      section 13  what a PASS would and would not establish
      section 14  the RR-1..RR-18 checklist
      section 15  the State block
```

## 2. DF-2 — what the insertion must say

The Owner's requirement is not a priority ordering. It is **gate dependency and precedence**,
and the five properties below are the acceptance criteria for the inserted text.

```text
  THE AMENDMENT MUST MAKE EXPLICIT, IN THE CONTRACT TEXT

  P1   G-1 through G-4 are PREREQUISITE GATES.
       they are evaluated first and independently of G-5, G-6 and G-7.

  P2   if any of G-1 through G-4 does not hold, the execution is VOID.

  P3   the outcome of G-5 or G-6 must NOT validate, cancel, offset or override a
       prerequisite failure.

  P4   G-7 must NOT convert a prerequisite failure into a valid execution.
       VOID is terminal and is neither a PASS nor a FAIL.

  P5   the FINAL-VERDICT PROPERTY MUST BE INDEPENDENT OF THE EXECUTOR'S EVALUATION ORDER.
       stated as a property of the contract, not as a procedure.  any order of evaluation
       that yields the same class per probe must yield the same final verdict.

  WHY P5 IS THE ONE THAT MATTERS, and it is why this is not a wording change
      the defect is that G-5 and G-6 read as terminal.  an executor who evaluates only
      G-5 and G-6 can report PASS while G-1 has already failed.  the current text prevents
      that only if the executor happens to read G-7.  an order-independent statement of the
      verdict removes the dependence on that.

  FORM OF THE INSERTION
      an ORDER-INDEPENDENT formulation is required, not merely a sequence.  a formulation of
      the form "evaluate G-1..G-4 first, then G-5/G-6, then G-7" would satisfy P1 to P4 and
      still fail P5, because it makes the verdict depend on the order.  the amendment must
      state the verdict as a function of the per-probe classes alone.
```

## 3. DF-3 — what the AB-6 replacement must define

The current text is the defect: **"taking effect" names no observable.**

```text
  CURRENT, L281..L282, to be REPLACED

      AB-6  the log shows g, go or gc taking effect
            immediate abort.  the target was allowed to run, which the contract forbids.

  THE AMENDMENT MUST DEFINE THREE DISTINCT LEVELS, and keep them distinct

    LEVEL 1  COMMAND ACCEPTED
             the debugger accepted the command as syntactically well formed and did not
             reject it.  this is a property of the PARSER.

    LEVEL 2  EVALUATION SUCCEEDED
             the command was carried out against a live target and did not fail for a reason
             of its own.  this is a property of EXECUTION.

    LEVEL 3  EFFECT OBSERVABLE
             the target's state changed in a way that is visible in the log.  this is a
             property of the TARGET.

    LEVEL 4  EFFECT ATTRIBUTABLE
             the observed change can be attributed to that command rather than to anything
             else.  without attribution there is no evidence that the command did anything.

  THE INVARIANT THE AMENDMENT MUST STATE

      syntax accepted  !=  evaluation succeeded  !=  observable effect  !=  attributable effect

      and, specifically as the Owner requires

      the mere PRESENCE OF A STRING in stdout or stderr is NOT evidence that the command
      took effect.  a string is evidence of LEVEL 1 at most.

  AB-6's TRIGGER, once defined
      the abort must fire on LEVEL 3 or LEVEL 4 attributable to g, go or gc.
      LEVEL 1 alone, and LEVEL 2 without an observable attributable effect, must NOT trigger
      it.  otherwise the abort would fire on a command that was accepted and evaluated but
      changed nothing, which is not a breach of "the target was allowed to run".

  THE OBSERVABLES THE AMENDMENT MAY NAME, offered as candidates and not as a fixed list
      the target's instruction pointer leaving the loader breakpoint
      any new module load, thread, or handle in the log
      any transition in the target's reported phase or state
      the count of log lines attributable to the post-breakpoint region
      a final instruction-pointer check, immediately before quit, against the loader
      breakpoint recorded at the initial stop

  the last of these is the strongest, because it is a direct state comparison rather than an
  inference from log text.  the amendment should state whether LEVEL 4 is established by
  the instruction-pointer check alone, or by that check in combination with log evidence.
```

## 4. FORBID

```text
  in this amendment, and in the Contract Amendment Review that follows it

  x  any change to the meaning of existing R1 contract text, other than the AB-6 replacement
  x  any change to G-1, G-2, G-3, G-4, G-5, G-6 or G-7 as written
  x  any change to the probe script, its order, its markers, or the classification C1..C4
  x  any change to the debuggee identity, the SHAPE choice, the cdb pin, or the invocation
  x  any change to the artifacts, the resolution discipline, the prohibitions, or no-retry
  x  any DF-1 amendment.  DF-1 is ACCEPTED / CLOSED and section 15 is not changed.
  x  Step5EK-R1 execution
  x  unblocking CR-3
  x  14/14, which remains NOT MET
  x  Implementation Authorization, which remains FORBIDDEN
  x  Phase 2-B
  x  any change to Amendment 1, which is established
  x  any change to the authoritative ConvoPeq.md, or any regeneration or re-pin of it
  x  source modification
  x  build
  x  CDB execution
  x  git operations
  x  any .cdb edit
```

## 5. What the amendment will do to the contract's identity, stated in advance

```text
  the contract's SHA-256 WILL CHANGE.  9690F7C5C29BC283084FB242FD4970F5CA919730CFB42A5CA12812718F621C9F
  becomes historical the moment the amendment is written.

  consequences, each of which is a separate later step and none of which this request performs

    - the Contract Review's RR-1..RR-18 verdicts were rendered against the 9690F7C5 text.
      the Contract Amendment Review must re-render them against the amended text.
    - the Review's own findings DF-2 and DF-3 are closed by the amendment, and the Review
      record that carries them is superseded on those two fields only.
    - the 9690F7C5 hash must not be cited as the current contract by any later record.

  none of this is an authorisation.  the Contract Amendment Review is a separate gate.
```

## 6. Verification the amendment must carry

```text
  V-1   the two target regions are the ONLY regions that differ from 9690F7C5
  V-2   a diff of amended against 9690F7C5 shows insertions in section 8 and a replacement of
        exactly L281..L282 in section 9, and nothing else
  V-3   G-1..G-7 appear in the amended text byte-identically to the 9690F7C5 text
  V-4   AB-1..AB-5, AB-7, AB-8 and ON ANY ABORT appear byte-identically
  V-5   the inserted DF-2 text satisfies P1..P5, each checked separately
  V-6   the replaced AB-6 text defines LEVEL 1..LEVEL 4, states the four-way inequality, and
        states that a string in stdout or stderr is not evidence of effect
  V-7   the AB-6 trigger is LEVEL 3 or LEVEL 4 attributable, not LEVEL 1
  V-8   the old "taking effect" phrase no longer appears
  V-9   DF-1 text, section 15, Amendment 1 and the authority state are untouched
  V-10  the whole of doc/work113 other than the R1 contract is byte-unchanged

  ANY V-1..V-10 FAILURE  ->  the amendment is not established.  report and stop.
  the amended contract is not itself a re-run.  a failed amendment verification is repaired
  by a further amendment under a new authorization, never by a silent second edit.
```

## 7. Sequence, unchanged and not compressed

```text
  DF-1  ACCEPT / CLOSED                                     DONE, no contract change
      |
      v
  DF-2 / DF-3 Contract Amendment                           THIS REQUEST.  not authorized.
      |
      v
  Contract Amendment Review                                separate gate
      |
      v
  R1 execution authorization                               Owner's judgement, separate gate
      |
      v
  Step5EK-R1 execution                                     the first CDB invocation
```

## 8. State

```text
Step5EI      DRAFTED, unchanged
Step5EJ      CLOSED  13 PASS / 0 FAIL / 1 PENDING
Step5EK      CLOSED / INCONCLUSIVE
Phase 3-A    CLOSED / PREPARED
Phase 3-B    CLOSED
Phase 3-C    D1 SELECTED, executed
Amendment 1  ESTABLISHED, unchanged, 9,546 B  SHA-256 B163DF9600B4A3B245C76F2889B95985767D17E05F960D27F67B1308508FA8F2

Step5EK-R1   CONTRACT  19,268 B  SHA-256 9690F7C5…  UNCHANGED
             Contract Review CLOSED  RR 18/18 PASS
               DF-1  ACCEPTED / CLOSED
               DF-2  CONTRACT AMENDMENT REQUIRED   this request
               DF-3  CONTRACT AMENDMENT REQUIRED   this request
             CONTRACT AMENDMENT NOT AUTHORIZED, NOT WRITTEN
             EXECUTION NOT AUTHORIZED

authority   ADJUDICATED = G0/G1 content, premise INTACT
CR-3        PENDING
14/14       NOT MET
Implementation Authorization   FORBIDDEN
Phase 2-B   NOT STARTED, NOT AUTHORIZED

cdb 0   build 0   source 0   commits 0   checkouts 0   re-pins 0
ConvoPeq.md regeneration 0   .cdb edits 0   git operations 0
R1 contract modifications 0   Amendment 1 modifications 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
Release AudioEngineHarness.exe  E5C7AFB9...  UNCHANGED

HOLD.  the only outstanding action is the Owner's explicit authorization of this request.
```

```text
held, untouched
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
