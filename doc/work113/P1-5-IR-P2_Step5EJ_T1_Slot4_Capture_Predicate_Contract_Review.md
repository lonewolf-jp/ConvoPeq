# Step5EJ — T-1 Slot-4 Capture Predicate Contract Review

```text
gate                = READ-ONLY Contract Review of Step5EI.  A REVIEW, not an authorization.
mode                = READ-ONLY.  No .cdb edit.  No CDB launch.  No runtime.  No build.
cdb launched        = 0
authority           = ConvoPeq.md  5,535,334 B
                      SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
subject             = P1-5-IR-P2_Step5EI_T1_Slot4_Capture_Predicate_Contract.md
                      13,126 B
                      SHA-256 3E513B3973936CAEE4BFE472FC8D503C5A25D7051EC5B88D3C03BF1928A6ADA5
result              = CR-1,2,4..14  PASS  (13)
                      CR-3          PENDING STATIC SYNTAX VALIDATION
outcome             = 14/14 NOT met.  Implementation Authorization must NOT be issued.  STOP.
```

## 0. Region discipline

Each CR was tested in the region where the property lives, not document-wide. A
document-wide substring test is a proxy, and for this contract it is a *falsely failing*
proxy, because the contract necessarily names the strings it forbids.

```text
  PRE     74 chars   unconditional region
                   = '.echo C3_DG_T0Q_HIT ; r ; .echo C3_DG_T0Q_EP0 ; .echo C3_DG_T0Q_WIN_END ; '
  GUARD  150 chars   capture-predicate region
                   = '.if (@rbx == 4) { .if (dwo(@r11+0x30010) == @r10) { .echo C3...'
  EDIT              the four declared edits of section 2
  S3 / S4 / S6 / S7 / S10   the sections the documentation CRs are tested in

  BEFORE/AFTER pairs specified in S2 : 1 / 1
  -> a single pair means a single site is specified, so no T0E, T0P, T0S or T1 edit
     can exist in this contract.  confinement is a structural consequence, not a promise.
```

## 1. Verdict table

```text
  CR-1   PASS    @r10 == 4 absent from GUARD and from AFTER
  CR-2   PASS    the sequence literal dwo(@r11+0x30010) == 4 absent from GUARD and AFTER
  CR-3   PENDING STATIC SYNTAX VALIDATION      identity CONFIRMED, syntax NOT assessed
  CR-4   PASS    @rbx == 4 retained, at offset 5 of GUARD
  CR-5   PASS    '; r ;' at 20 precedes '.if (' at 74, so the dump is unconditional
  CR-6   PASS    no 'r' between CAS_POST and gc, and it WAS in BEFORE, so the edit is real
  CR-7   PASS    FN=0 rests on the CAS-success topology, not on any run
  CR-8   PASS    K_i = max(0, floor((E_i-4)/4096)+1), general acceptance bound SUM K_i
  CR-9   PASS    the only upper-bound-in-4 line is inside the E_i <= 4100 special case
  CR-10  PASS    T0S BYTE-IDENTICAL in S6, T0S UNCHANGED in S1, absent from the edit list
  CR-11  PASS    T0P generalisation recorded OPEN, and no T0P line is an edit target
  CR-12  PASS    T1 / S7 UNCHANGED in S1, no T1/S7 site is a target
  CR-13  PASS    both malformed T1 RVAs named as retaining content, neither in the edit section
  CR-14  PASS    no authorisation-granting phrase anywhere; header disclaims authorisation twice

  PASS 13 / 13 executable      FAIL 0      PENDING 1  (CR-3)
```

## 2. CR-1 and CR-2, and why a document-wide test would have been wrong

```text
  CR-1  the string '@r10 == 4' occurs document-wide in exactly ONE region: S10, the CR
        checklist line.  that is the prohibition text.  testing document-wide would have
        reported a FAIL against the contract's own description of what it forbids.

  CR-2  a digit-specific test, '== 4 must be absent', would have reported a FALSE FAIL,
        because '@rbx == 4' is present in GUARD and is CORRECT.  that is the slot
        conjunct required by CR-4, not the sequence conjunct.

        the test is therefore sequence-specific:
            dwo(@r11+0x30010) == 4   must be ABSENT
            @rbx == 4                must be PRESENT

        document-wide the sequence literal occurs in S2, S6, S7.  all three are legitimate:
            S2  the BEFORE line, and edit #3's LEFT-HAND SIDE, which is the T0Q form
                being changed
            S6  the T0P conjunct quoted as NOT being generalised
            S7  the T0P conjunct quoted as staying as it is
        none of them is a T0P edit.
```

## 3. CR-3 — PENDING, and the two things that must not be confused

```text
  IDENTITY, CONFIRMED
      dwo(@r11+0x30010) == @r10   is present in GUARD
      it occurs exactly ONCE in AFTER, so there is no ambiguity between two forms
      the retained slot conjunct precedes it:  '.if (@rbx == 4) {'

  SYNTAX, NOT ASSESSED
      the shape  dwo(...) == @reg   occurs 0 times across the whole frozen 16-file corpus.

      therefore 'a similar form exists elsewhere' is NOT available as a PASS:
          dwo(...) == <literal>   is proven at all four T0 sites
          @reg    == <literal>   is proven at all four T0 sites
          dwo(...) == @reg        has no occurrence, verified or otherwise

  corpus precedent is NOT parser validation.  this review did not run a parser.
  cdb was not launched.  debuggee none, production runtime none, AudioEngineHarness none,
  build none, .cdb corpus unmodified.

  CR-3 = PENDING STATIC SYNTAX VALIDATION.  it is neither PASS nor FAIL.
```

## 4. CR-5 and CR-6 read together, because either alone would be weak

```text
  CR-5 alone would pass on a contract that dumped registers twice.
  CR-6 alone would pass on a contract that deleted the dump entirely.
  together they establish the intended shape: the dump exists, it is outside the guard,
  and the copy inside the acceptance branch is gone.
```

## 5. CR-9, checked by enumeration rather than by presence

```text
  every line in the contract that asserts an upper bound ending in 4:

      L99   0  <=  captures  <=  (number of instances that reached slot 4)  <=  4

  that is the ONLY such line, it sits inside the special case, and the special case is
  introduced by 'if E_i <= 4100 for every i'.  UNCONDITIONAL bound-in-4 lines: none.

  the general specification is stated as  0 <= captures <= SUM K_i
  and the contract explicitly disclaims 0..4 as a general specification.
```

## 6. Site confinement established by fingerprint, not by prose

```text
  the contract specifies one BEFORE/AFTER pair.  the BEFORE is byte-identical to the real
  Baseline-3 line at RVA 0x1f9c59a, which is T0Q:

      contract BEFORE == real 0x1f9c59a line : True

  Baseline-3 itself is unmodified:
      sha256 EE82A517698C3257B5612D0EE9957B0AB7E0E027997AE0B4BE93C69947035E58 : MATCH

  so CR-10, CR-11 and CR-12 rest on a verified anchor, not on the contract's assurance
  that it touches nothing else.
```

## 7. Two defects found in the REVIEWER, recorded rather than quietly fixed

Both were caught during this step and both are the same family the workstream already
names: a check that does not test the property in its own region, or a conversion that
silently changes a verdict.

```text
  D-R1  revision 1 reported CR-9, CR-10 and CR-12 as FAIL.  all three were TEST defects,
        not contract defects.
          CR-10, CR-12  the haystack was whitespace-normalised but the needle was not, so a
                        needle containing runs of spaces could never match.
          CR-9           the disclaimer was searched only inside the lines matching
                        'THE GENERAL SPECIFICATION IS', but it is the following line.
        fixed by routing every haystack and every needle through one normaliser, and by
        searching CR-9's own region.

  D-R2  revision 1 printed 'CR-3  PASS'.  the cause was  res[k] = bool(ok)  in the recorder:
        bool("PENDING") is True, so a PENDING verdict was promoted to a PASS by a type
        coercion, in the exact place where the Owner required it not to be a PASS.
        fixed by removing the coercion.  the verdict is stored as given.

        this is the most serious of the three, because it is the one that could have
        produced a false 14/14.
```

## 8. Why the review stops here

```text
  the Owner's gate condition is  CR-1 .. CR-14 all PASS  before Implementation
  Authorization may be issued.

  current state is  13 PASS, 0 FAIL, 1 PENDING.  14/14 is not met.

  therefore   Implementation Authorization   NOT issued
              Step5EK Implementation         NOT started
              .cdb edit                      FORBIDDEN
              runtime                        FORBIDDEN
              build                          FORBIDDEN
              source edit                    FORBIDDEN

  Case A applies.  CDB parser validation was not performed and was not authorised.

  the pending item is a PERMISSION question, not an analysis question.  resolving it
  requires launching cdb, even with no debuggee, which the current state forbids.  the
  agent has not taken that permission for itself.
```

## 9. What would close CR-3, if the Owner authorises it

Recorded as a request, not as a plan to be executed.

```text
  a separate gate, parser-only static validation, with

      CDB debuggee                          none
      production runtime                    none
      AudioEngineHarness launched           none
      build invoked                         none
      .cdb production corpus modified       none
      artifact written                      a NEW log under doc/work113, resolved by SHA-256
                                           at creation time, not a new .cdb

  the question it answers, and ONLY this question
      does the CDB expression parser accept  dwo(@r11+0x30010) == @r10  inside .if ?

  a PASS there promotes CR-3 to PASS and makes 14/14 reachable.
  a FAIL there means the predicate must be re-expressed, and the contract returns to draft.

  it is worth noting what CR-3 does NOT need: it needs no debuggee, no registers and no
  process state, because the question is whether the expression parses, not what it
  evaluates to.  that is what makes a parser-only gate possible at all.
```

## 10. State

```text
Step5EI      DRAFTED, unchanged by this review
             13,126 B  sha256 3E513B3973936CAEE4BFE472FC8D503C5A25D7051EC5B88D3C03BF1928A6ADA5

Step5EJ      CLOSED as a REVIEW.  result 13 PASS / 0 FAIL / 1 PENDING.
             Implementation Authorization NOT issued.  STOP.

cdb launched  0
cdb edit      0
runtime       0
build         not invoked
source        unchanged

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16    ConvoPeq.md MATCH
git delta 12, all pre-existing    residue 0

OPEN carry, unchanged by this review
    CR-3 static syntax validation of dwo(@r11+0x30010) == @r10     PENDING, needs authorisation
    T0P sequence conjunct generalisation, diagnostic consistency
    the sequence operand not being observable on a failing hit, Step5EI section 9
    the two malformed T1 sites
    the 65 versus 116 hit non-comparability
```
