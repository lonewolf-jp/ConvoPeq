# Step5EK — CR-3 Parser-Only Static Validation / AUTHORIZATION REQUEST — PENDING

```text
gate                = an authorization REQUEST.  NOT an authorization.  NOT a validation result.
mode                = READ-ONLY.  cdb NOT launched.  no .cdb edit.  no runtime.  no build.
status              = AWAITING OWNER EXPLICIT PERMISSION
authority           = ConvoPeq.md  5,535,334 B
                      SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
                      re-verified this step.  state maintained as instructed.
precondition        = Step5EJ CLOSED  13 PASS / 0 FAIL / 1 PENDING (CR-3)
```

## 1. A finding that changes the gate design, obtained without launching cdb

The Owner's proposed gate asks whether the CDB parser accepts

```text
dwo(@r11+0x30010) == @r10
```

inside `.if (...)`. Before designing the check, I asked whether a *`bp`-based*
check could answer that at all. It cannot, and the workstream's own runtime log proves it.

```text
  EVIDENCE, Step5ED runtime log, 2603 lines, single run

      L2577   C3_T1A_CANDIDATE                      <- .echo inside the SAME command string ran
      L2578   r14=0000000000000004 r15=...          <- the register dump in that string ran
      ...    AudioEngineHarness+0x1f9cfb0:  push rbx
      L2596   Syntax error at '(@$t2+0x34040)&0xfff)*4) '
      L2600   C3_T1_REDESIGN2_SCRIPT_END

  READING   the breakpoint was SET SUCCESSFULLY while its command string contained a malformed
            expression.  execution of that string proceeded through .echo and r, and only
            raised the error when it REACHED the bad expression.  the failure is therefore
            DEFERRED TO HIT TIME.

  CONSEQUENCE   a gate that sets a bp and looks for an error is NON-DISCRIMINATING.  it would
            report "no syntax error" for a malformed expression and PASS VACUOUSLY.  that is
            the same defect family the workstream already names: an all() over an empty list
            reported as PASS.

  DOCUMENTATION, consistent
      "You can include a command in a breakpoint that is automatically executed WHEN THE
       BREAKPOINT IS HIT."   Methods of Controlling Breakpoints, learn.microsoft.com

  THEREFORE   the gate must NOT use bp.  it must use a command that parses the expression
            EAGERLY, in the command loop, with no process attached.
```

## 2. The command that does parse eagerly, and the documentation that shows its output

```text
  COMMAND   ? (Evaluate Expression)

  DOCUMENTED BEHAVIOUR, verbatim from learn.microsoft.com, ? (Evaluate Expression)

      0:001> ? $spat( "c:\dir\", "*filename*" )
      Syntax error at '( "c:\dir\", "*filename*" )

  THIS IS THE KEY PROPERTY.  ? reports  Syntax error at '...'  AT COMMAND-ISSUE TIME, in the
  command loop, with no breakpoint and no hit involved.  and the message class is the SAME
  'Syntax error at' that the log shows at hit time, so it is the same parser.

  a well-formed expression instead yields  Evaluate expression: <value>

  ∴ the discriminator exists and is binary:
        Syntax error present   ->  the parser REJECTED the form
        Syntax error absent    ->  the parser ACCEPTED the form
```

## 3. The gate, as it would be run IF permission is granted

```text
  INVOCATION   cdb with NO debuggee, NO image, NO AudioEngineHarness, one -logo log,
               -c carrying exactly the four probes below.  no .cdb is passed to cdb at all.
               -o is not used; -logo only.

  PROBES, in this order.  the order matters and is not cosmetic.

  P1  NEGATIVE CONTROL, primary
          ? (@$t2+0x34040)&0xfff)*4)
      this is the expression that ACTUALLY produced a syntax error in the Step5ED run.
      it is used because its failure is empirically established, not predicted.

  P2  NEGATIVE CONTROL, secondary
          ? (dwo(@r11+0x30010)==@r10
      unbalanced parenthesis, the minimal malformation of the target form itself.

  P3  POSITIVE CONTROL, the documented-working shape
          ? dwo(@r11+0x30010) == 4
      display function compared against a literal.  proven working at all four T0 sites.

  P4  TARGET
          ? dwo(@r11+0x30010) == @r10
```

## 4. Gate validity conditions — the part that makes the result mean something

```text
  G-1  P1 and P2 MUST each produce  Syntax error at ...
       if either produces no syntax error, the harness is NON-DISCRIMINATING and the gate
       result is VOID.  a void gate does not promote CR-3, regardless of P4's outcome.

  G-2  P3 MUST produce no syntax error.
       if P3 errors, the harness is over-reporting and the gate is VOID.

  G-3  P4 produces no syntax error  ->  CR-3 = PASS
       P4 produces Syntax error     ->  CR-3 = FAIL, and Step5EI returns to draft for
                                       re-expression of the predicate.

  G-4  the log is written under doc/work113 as a NEW file, named by SHA-256 at creation.
       it is NOT a .cdb.  the .cdb corpus is not touched.

  THE NEGATIVE CONTROLS ARE WHAT SEPARATE A REAL PASS FROM A VACUOUS ONE.  without them,
  "no syntax error in P4" carries no information, because a harness that reports nothing
  would produce the same observation.  P1 and P2 run FIRST precisely so that a harness
  which cannot discriminate is discovered before any conclusion is drawn about the target.
```

## 5. Scope of what a PASS would and would not establish

Stated narrowly, so the result cannot later be stretched.

```text
  A PASS WOULD ESTABLISH
      the MASM expression grammar accepts a display-function result compared against a
      pseudo-register.  that is exactly the one element of the contract with no verified
      precedent, since dwo(...) == @reg has 0 occurrences across the 16-file frozen corpus.

  A PASS WOULD NOT ESTABLISH
      that the full bp command string is well formed.  a bp string additionally involves
      ';' splitting, brace matching and the .if control wrapper.  those are separately
      proven already -- .if (dwo(...)) appears at all four T0 sites and @r10 == <literal>
      appears inside .if at all four -- so the uncovered element is precisely the operand
      pairing, which P4 does cover.

      anything about behaviour.  the gate asks whether the expression parses, not what it
      evaluates to.  with no process attached, @r10 has no value and dwo(...) cannot read
      memory, so P4's VALUE is meaningless and must not be read as a result.

  DOCUMENTATION-BASED PRIOR, recorded but NOT a substitute
      the MASM operator table lists dwo as "Double-word from the specified address" under
      Numeric Operators, so its result is a number, and numbers are operands of ==.
      on the documented grammar the target form looks legal.
      this RAISES the prior.  it does not replace parser validation, and it is not offered
      as a reason to skip P4.  the workstream's standing rule is that one kind of evidence
      does not stand in for another.
```

## 6. Residual risk stated in advance

```text
  UNKNOWN, and it is an executability question, not a logic question
      whether cdb, given no image at all, enters its command loop and processes -c probes,
      or refuses to start.  this cannot be determined without launching cdb, which is
      precisely what is being withheld.

  consequence if cdb refuses
      P1 is the first probe, so the failure is immediate, cheap, and produces no conclusion
      about the target.  the gate would report INCONCLUSIVE and CR-3 would stay PENDING.

  this is why P1 runs first.  the gate is designed to fail cheaply and honestly rather than
  to guess.
```

## 7. What is explicitly NOT being asked for, and NOT being done

```text
  NOT requested, NOT taken:
      production runtime          debuggee                 build
      AudioEngineHarness          .cdb edit                .cdb corpus modification
      source edit                 predicate behaviour verification

  the four held OPEN items are untouched and remain held.  in particular T0P's
  dwo(@r11+0x30010) == 4 is NOT generalised to @r10 here.  that is a separate gate, and
  widening Step5EI's implementation scope is exactly what must not happen.

      1  T0P sequence conjunct generalisation
      2  the sequence operand not being observable on a non-accepting T0Q hit
      3  the two malformed T1 sites
      4  the 65 versus 116 hit non-comparability
```

## 8. Decision paths, for the Owner

```text
  GRANT as specified in section 3
      the gate runs, CR-3 becomes PASS or FAIL on evidence, and 14/14 becomes reachable
      only in the PASS case.  a separate 14/14 authorization gate is still required before
      any implementation, per the Owner's own sequencing.

  GRANT with the section 4 conditions relaxed
      NOT recommended.  removing the negative controls reduces the gate to a rubber stamp
      whose PASS is indistinguishable from silence.

  WITHHOLD
      CR-3 stays PENDING, 14/14 stays unreachable, and the workstream holds where Step5EJ
      left it.  nothing is lost by waiting; the gate is not time-sensitive.

  The agent will not launch cdb without an explicit grant.  It has not done so.
```

## 9. State

```text
Step5EI   DRAFTED, unchanged    13,126 B  sha256 3E513B3973936CAEE4BFE472FC8D503C5A25D7051EC5B88D3C03BF1928A6ADA5
Step5EJ   CLOSED  13 PASS / 0 FAIL / 1 PENDING
          10,695 B  sha256 8DF5FE802AC2964C70213467...
Step5EK   AUTHORIZATION REQUEST, PENDING OWNER

cdb launched   0
cdb edit       0
runtime        0
build          not invoked
source         unchanged

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16    ConvoPeq.md MATCH
git delta 12, all pre-existing    residue 0

HOLD.  the single next action is the Owner's decision on section 8.
```
