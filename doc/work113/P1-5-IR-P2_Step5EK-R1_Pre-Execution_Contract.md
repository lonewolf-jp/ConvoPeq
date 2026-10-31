# Step5EK-R1 — Pre-Execution Contract

```text
gate                = a CONTRACT.  NOT an authorization to execute.
mode                = READ-ONLY drafting.  cdb NOT launched.  no debuggee.  no image executed.
owner decision      = OPTION 1 APPROVED
authorized scope    = CREATION OF THIS CONTRACT ONLY
NOT authorized      = CDB execution, Step5EK-R1 execution, .cdb modification, source modification,
                      build, production runtime, Harness execution, Step5EI predicate modification,
                      T0P generalization, T0Q dd probe modification, malformed T1 repair
purpose             = remove the structural blocker that made Step5EK INCONCLUSIVE, so that the
                      target expression can be submitted to a REAL user-mode parser
status              = DRAFTED.  awaiting Contract Review.  NOT approved, NOT executed.
```

## 0. Three authorizations, kept separate

```text
  OPTION 1 approval              !=  R1 execution approval
  R1 execution approval          !=  Implementation Authorization

  this document is the FIRST of those three things.  it is a contract for a gate that
  has not been approved to run.  approving this contract does not approve running it.
```

## 1. What R1 is, and what it is not

```text
  IS
      a parser question, asked against the PRODUCTION cdb binary, inside a real user-mode
      command loop, with per-probe diagnostic classification.

  IS NOT
      a re-run of Step5EK.  Step5EK remains CLOSED / INCONCLUSIVE and its log is retained
      as evidence.  R1 is a different gate with a different invocation, a different
      precondition, and a different purpose.

  IS NOT
      a behavioural test.  nothing is measured.  no capture is attempted.  no predicate
      semantics are touched.
```

## 2. Debuggee identity — FIXED

```text
  SHAPE A.  cdb launches the image itself and stops at the loader breakpoint.

  debuggee          C:\Windows\System32\hostname.exe
  size              40,960 B
  SHA-256           F8B207720D8673FEB5839DF1E52360F72C2AD2CE9917D5F3DA4F30368DC747C9
  PE                MZ / PE, machine 0x8664, subsystem CONSOLE

  bitness is a REQUIREMENT, not a preference
      AudioEngineHarness.exe is machine 0x8664, x64.  the predicate names @r10 and @r11,
      which are x64 pseudo-registers.  a 32-bit debuggee would be debugged through a
      32-bit register view and the pseudo-register names would not mean the same thing.
      the debuggee is therefore REQUIRED to be x64.  hostname.exe is x64.

  RESOLUTION RULE
      the path is resolved by SHA-256 at execution time, by scanning the directory and
      matching the hash.  it is NEVER hand-transcribed into the command line.

  candidates verified and NOT selected, recorded so the choice is auditable
      C:\Windows\System32\find.exe    40,960 B  CF9CC5A552DE280750B36743E80C33381E59D5EC8AA65A2DCFDD6A34198799BC  x64
      C:\Windows\System32\where.exe   65,536 B  F51FAB8041E5023D7290B540A2106F051EF0BD2BC3443C9DAF318628B560FA29  x64
      C:\Windows\System32\mkdir.exe   absent, ruled out
      hostname.exe is chosen because it is the smallest x64 console candidate, has no
      arguments, spawns nothing, and its entire behaviour is irrelevant because the
      process is never allowed to execute past the loader breakpoint.

  EXCLUDED, and located so that it cannot be selected by accident
      C:\VSC_Project\ConvoPeq\build\Release\AudioEngineHarness.exe              41,216,512 B
      C:\VSC_Project\ConvoPeq\build-asan\Debug\AudioEngineHarness.exe           98,243,072 B
      C:\VSC_Project\ConvoPeq\build-asan\RelWithDebInfo\AudioEngineHarness.exe  73,467,392 B
```

## 3. SHAPE B — considered and rejected, with the reason

```text
  SHAPE B was  cdb -pv -p <pid>  attaching non-invasively to a process already running.

  REJECTED for three reasons, each independent

    1  DETERMINISM.  the subject would be one of 423 running processes, chosen at
       execution time.  that makes the gate's subject a function of machine state rather
       than a fixed input, which is the wrong property for a gate.

    2  BLAST RADIUS.  even a non-invasive attach is an attach to a live system process.
       SHAPE A touches nothing that was running before the gate.

    3  THE SUBJECT IS IRRELEVANT ANYWAY.  the gate asks whether the parser accepts a
       grammar form.  it does not ask what any register or memory contains.  a fresh
       process is therefore not a compromise, it is sufficient.

  SHAPE A is FIXED.  SHAPE B is not used.
```

## 4. Debugger identity — FIXED, and it must be the production parser

```text
  cdb              tmp\cdb.exe
  size             178,016 B
  SHA-256          5F54ABAFCA3AE5638BBF807D402FABB350A64575C1DFA9FBFC7F5732DF5BEE67

  THIS IS A REQUIREMENT, NOT A CONVENIENCE

      the question CR-3 asks is whether the parser of the cdb that will run the production
      script accepts dwo(@r11+0x30010) == @r10.

      a PASS obtained from any other cdb build would not transfer.  a different build is
      a different parser, and the whole point of the gate is to test THIS parser.

      therefore the binary is pinned by hash, and a hash mismatch is an abort, not a
      substitution.
```

## 5. Invocation — byte-exact

```text
  tmp\cdb.exe  -cf  <script>  -logo  <log>  <debuggee>

  constraints taken from the CDB command-line documentation, not assumed
      -cf "filename"   specifies a script file, executed as soon as the debugger starts.
                       -cfr is the same plus re-execution on target restart; -cfr is NOT
                       used because the target is never restarted, and the simpler option
                       has one fewer behaviour to reason about.
      "If an executable is specified, it must appear LAST on the command line; any text
       after the executable name is passed to the executable program as its own
       command-line parameters."
                       therefore the debuggee is placed LAST and nothing may follow it.
      -logo is used.  -o is NOT used.
      -y is NOT used.  no symbols are needed, because the probes reference registers, not
            symbols, and omitting -y avoids any symbol-path side effect.

  the executable never runs
      cdb stops at the loader breakpoint.  g, go and gc are forbidden, so the debuggee's
      first instruction is never executed.  its behaviour is therefore irrelevant and its
      termination on quit is the only consequence of its existence.
```

## 6. Probe script — byte-exact, order FIXED

```text
  .echo ====R1_P3_POSITIVE_CONTROL====
  ? dwo(@r11+0x30010) == 4
  .echo ====R1_P1_NEGATIVE_CONTROL_PROVEN====
  ? (@$t2+0x34040)&0xfff)*4)
  .echo ====R1_P2_NEGATIVE_CONTROL_BALANCED====
  ? dwo(@r11+0x30010) == == @r10
  .echo ====R1_P4_TARGET====
  ? dwo(@r11+0x30010) == @r10
  .echo ====R1_PROBES_END====
  q

  ORDER, and why it is not the order in the authorization request

      P3  positive control, FIRST.  parenthesis-balanced, so it carries no risk of
          swallowing a following line.  it establishes that the script runs and that the
          harness produces probe output at all.  if P3 yields nothing, everything after it
          is already known to be unusable.

      P1  negative control, the one whose failure is PROVEN.  Step5ED recorded
          "Syntax error at '(@$t2+0x34040)&0xfff)*4) '" in a real run, so this probe is
          expected to produce a syntax error with high confidence.  a prediction is used
          here precisely because a proven one exists.

      P2  negative control, the minimal malformation OF THE TARGET FORM ITSELF, and
          parenthesis-balanced so that it cannot swallow the target probe.  the duplicated
          operator is the smallest edit that turns the target into a non-expression.

      P4  target, LAST.  both negatives precede it, so the guarantee that a
          non-discriminating harness is discovered before any conclusion is drawn about
          the target is preserved.

  the ordering risk is CONTAINED rather than removed
      P1's expression contains an unmatched ')'.  if that swallowed a following line, the
      swallowed probe would produce NO diagnostic, which is outcome class C4, which forces
      VOID.  the failure mode of this risk is therefore a void gate, never a false PASS and
      never a false FAIL.

  MARKERS
      every probe is preceded by its own echo marker and followed by the next.  the
      classifier attributes each outcome to the region between markers, so a diagnostic can
      never be attributed to the wrong probe.  this is the attribution discipline that
      Step5EE had to introduce mid-investigation; it is built in from the start here.
```

## 7. Outcome classification — the syntax / memory separation, made explicit

This section is the reason R1 can produce a trustworthy verdict rather than a manufactured one.

```text
  each probe is assigned EXACTLY ONE class, from the lines between its marker and the next

    C1  PARSE_REJECT
        a line of the form   Syntax error at '<token>'   is present.
        the parser REFUSED the form.  this is the only class that says anything about
        grammar.

    C2  EVAL_FAILED_NON_SYNTAX
        no 'Syntax error' line, and no value, but a diagnostic indicating the expression
        could not be completed.  memory read failure, unreadable address, unusable
        register context.  THE PARSER ACCEPTED THE FORM.  the evaluation failed.

    C3  VALUE
        a line of the form   Evaluate expression: <v> = <hex>   is present.
        the parser accepted the form and the evaluation completed.

    C4  NO_OUTPUT
        nothing at all between this probe's marker and the next.
        the probe did not run, or its output was swallowed.  THIS IS NOT A PASS.

  WHY C2 AND C3 BOTH MEAN "PARSER ACCEPTED", AND WHY THAT MATTERS

      under SHAPE A, ? evaluates against a real but arbitrary process.  @r10 takes that
      process's r10, and dwo(@r11+0x30010) attempts to read that process's memory at
      r11+0x30010.  that address is almost certainly unmapped in a fresh hostname.exe
      image, so P4 will most likely land in C2, not C3.

      C2 is a SUCCESS for the gate's question.  the grammar was accepted; only the memory
      access failed.  a contract that did not separate C2 from C1 would report a false FAIL
      and send the predicate back for redesign on the strength of an unmapped address.

      THE TRAP, stated so it cannot be walked into
          if cdb's memory-error text happened to contain the words "Syntax error", a
          classifier keyed on that substring would report C1 for a form the parser accepted.
          the classifier is therefore keyed on the FULL diagnostic form
          'Syntax error at '<token>'' and on nothing shorter.

  CLASSES ARE MUTUALLY EXCLUSIVE AND COVER THE ABSENT CASE
      C4 exists precisely so that "nothing happened" has a name.  a gate without a name for
      absence will eventually read absence as success.  that is the all()-over-an-empty-list
      defect, and it is excluded here by construction.
```

## 8. G conditions — how the classes become a CR-3 verdict

```text
  G-1  P1 is C1  AND  P2 is C1
       the harness demonstrably distinguishes a malformed expression from a well-formed one.
       if this fails, the harness is non-discriminating and NO verdict may be drawn.

  G-2  P3 is C2 or C3   (that is: NOT C1 and NOT C4)
       the harness does not over-report, and the probe demonstrably ran.

  G-3  R1_PROBES_END marker is present
       the script ran to completion rather than stopping partway.

  G-4  no probe is C4
       every probe produced an attributable outcome.

  G-5  P4 is C1   ->  CR-3 = FAIL
       the parser refused dwo(@r11+0x30010) == @r10.  Step5EI returns to draft and the
       predicate must be re-expressed.

  G-6  P4 is C2 or C3   ->  CR-3 = PASS
       the parser accepted the form.

  G-7  P4 is C4, or any of G-1..G-4 fails   ->  VOID
       and VOID is NOT a pass, NOT a fail, and NOT a licence to retry.

  DF-2 INSERTION.  GATE DEPENDENCY AND PRECEDENCE.  G-1 through G-7 above are unchanged.

    DEPENDENCY
      G-1, G-2, G-3 and G-4 are PREREQUISITE GATES.  each is evaluated on its own terms,
      independently of G-5, G-6 and G-7.

    A PREREQUISITE FAILURE IS ABSOLUTE
      if any of G-1, G-2, G-3, G-4 does not hold, the verdict is VOID.
      VOID is terminal and is neither CR-3 = PASS nor CR-3 = FAIL.

    NO OFFSET, NO RESCUE
      the outcome of G-5 or G-6 must not validate, cancel, offset, override or in any way
      excuse a prerequisite failure.  a CR-3 = PASS produced by G-6 while any prerequisite has
      failed is not a PASS; it is VOID.  G-7 does not convert a prerequisite failure into a
      valid execution.

    THE VERDICT IS A FUNCTION OF THE OBSERVATIONS ALONE, NOT OF ANY EVALUATION ORDER

      let c1 c2 c3 c4 be the outcome classes of P1, P2, P3 and P4, each in {C1, C2, C3, C4},
      and let M be the presence of the R1_PROBES_END marker.

      prerequisites, as predicates on (c1, c2, c3, c4, M)

        pre1   c1 == C1  and  c2 == C1
        pre2   c3 in {C2, C3}
        pre3   M is true
        pre4   no c_i equals C4

      verdict

        if pre1 and pre2 and pre3 and pre4
            c4 == C1              ->  CR-3 = FAIL
            c4 in {C2, C3}        ->  CR-3 = PASS
            c4 == C4              ->  VOID
        otherwise
            VOID

      this is a definition over the observations.  it names no sequence and it is not a
      procedure, so nothing in it depends on the order in which an executor happens to
      evaluate anything.

    ORDER INDEPENDENCE, stated as a falsifiable property rather than as a promise

      permuting the order in which G-1 through G-7 are evaluated does not change the verdict.

      an executor that evaluates only G-5 and G-6 does not produce CR-3 = PASS when a
      prerequisite has failed.  it produces VOID.

      an executor that reaches a CR-3 verdict without first establishing pre1 to pre4 has
      not applied this definition, whatever it concludes.
```

## 9. Abort conditions

```text
  AB-1  cdb.exe hash is not 5F54ABAF...5EE67
        the parser under test is not the production parser.  abort.  do not substitute.

  AB-2  the debuggee hash is not F8B20772...47C9
        abort.  do not substitute another image.

  AB-3  the first marker R1_P3 does not appear
        the script did not run.  VOID.

  AB-4  R1_PROBES_END does not appear
        the script did not complete.  VOID.

  AB-5  an AudioEngineHarness process is present at any point
        immediate abort.  the contract is already breached.

  AB-6  an OBSERVABLE AND ATTRIBUTABLE EFFECT of g, go or gc is present
        immediate abort.  the target was allowed to run, which the contract forbids.

        THE FOUR LEVELS.  they are distinct and must not be conflated.

          LEVEL 1   COMMAND ACCEPTED
                     the debugger accepted the command as syntactically well formed and did
                     not reject it.  a property of the PARSER.

          LEVEL 2   EVALUATION SUCCEEDED
                     the command was carried out against a live target and did not fail for a
                     reason of its own.  a property of EXECUTION.

          LEVEL 3   EFFECT OBSERVABLE
                     the target's state changed in a way that is visible in the log.
                     a property of the TARGET.

          LEVEL 4   EFFECT ATTRIBUTABLE
                     the observed change can be attributed to that command rather than to
                     anything else.  without attribution there is no evidence that the command
                     did anything at all.

        syntax accepted  !=  evaluation succeeded  !=  observable effect  !=  attributable effect

        A STRING APPEARING IN stdout OR stderr IS NOT EVIDENCE OF EFFECT.  it is evidence of
        LEVEL 1 at most.  a log line that merely names g, go or gc does NOT trigger AB-6.

        AB-6 TRIGGERS ON LEVEL 3, or on LEVEL 4 attributable to g, go or gc.
        it does NOT trigger on LEVEL 1 alone, nor on LEVEL 2 without an observable and
        attributable effect.  a command that was accepted and evaluated but changed nothing
        is not a breach of "the target was allowed to run".

        OBSERVABLES.  offered as candidates, not as a closed list.
          the instruction pointer having left the loader breakpoint
          a new module load, thread or handle appearing in the log
          a transition in the target's reported phase or state
          a FINAL INSTRUCTION-POINTER CHECK, taken immediately before quit, compared against
          the loader breakpoint recorded at the initial stop

        the final check is a direct state comparison rather than an inference from log text.
        this contract REQUIRES the final check.  LEVEL 4 is established by that check alone,
        or by that check in combination with log evidence.

  AB-7  cdb is still resident after the script completes
        immediate abort.

  AB-8  any G-1..G-4 condition fails
        VOID.

  ON ANY ABORT
      report, do not retry.  see section 10.
```

## 10. No-retry rule

```text
  ONE invocation.  that is the whole budget.

  VOID is TERMINAL for this gate.  it is reported as VOID with the evidence, and the gate
  ends there.

  changing the probe set, the probe order, the shape, the image or the cdb binary in order
  to obtain a non-VOID result requires a NEW authorization and an AMENDED contract.  it is
  not permitted as a repair of a VOID run.

  REASON, and it is the same reason Step5EK's no-image invocation was not retried
      re-running to obtain a better answer is the re-run pattern this workstream exists to
      prevent.  a gate that can be retried until it passes is not a gate.

  and specifically for AB-3 and AB-4
      those are the failures most likely to be misread as "the tool did not work, try
      again".  they are the ones this rule exists to stop.  their correct handling is a
      report of VOID plus an amended contract, not a second invocation.
```

## 11. Artifacts and resolution discipline

```text
  ONE new log
      doc/work113/P1-5-IR-P2_Step5EK-R1_CR-3_User-Mode_Parser_Validation_CDB.log
      created fresh.  must not pre-exist.  -logo only.

  ONE new result record, written after execution
      doc/work113/P1-5-IR-P2_Step5EK-R1_CR-3_User-Mode_Parser_Validation_Result.md
      states the class of every probe, the G evaluation, the CR-3 verdict or VOID, and the
      SHA-256 of the log.

  probe script
      tmp/, ephemeral.  its SHA-256 is recorded in the result record, then it is deleted,
      so that the invocation is reproducible without leaving residue.

  NOT created, NOT modified
      no .cdb of any kind.  the production .cdb corpus stays at 16 files, unaltered.
      no source file.  no build output.  no existing log.

  RESOLUTION
      the debuggee and the cdb binary are located by scanning for the recorded SHA-256.
      no artifact filename is ever hand-transcribed into a command.
```

## 12. Prohibitions, carried forward

```text
  x  CDB execution under this contract.  this contract does not authorize it.
  x  g, go, gc
  x  bp or bu at any address.  the gate sets NO breakpoints.
  x  .reload /f, -y, and any symbol-path manipulation
  x  -o
  x  AudioEngineHarness, in any of its three build locations
  x  any production runtime
  x  build
  x  source modification
  x  any .cdb edit, and any .cdb corpus change
  x  Step5EI predicate modification
  x  T0P generalization
  x  T0Q dd probe modification
  x  malformed T1 repair
  x  folding the 65 versus 116 question into this gate
  x  a second invocation of any kind
```

## 13. What a PASS would and would not establish

Stated before the run, so the result cannot be stretched afterwards.

```text
  A PASS WOULD ESTABLISH
      the MASM expression grammar of the PRODUCTION cdb accepts a display-function result
      compared against a pseudo-register.  this is the single element with no verified
      precedent, dwo(...) == @reg having 0 occurrences across the 16-file frozen corpus.

  A PASS WOULD NOT ESTABLISH
      that the full bp command string is well formed.  a bp string additionally involves
      ';' splitting, brace matching and the .if control wrapper.  those are separately
      proven: .if (dwo(...)) appears at all four T0 sites, and @r10 == <literal> appears
      inside .if at all four.  the uncovered element is precisely the operand pairing, and
      that is what P4 covers.

      and it is worth being explicit about WHY a bp cannot be used to cover the rest: the
      Step5ED log shows a malformed expression escaping bp-set-time validation entirely and
      surfacing only at hit time.  set-time validation of the .if expression does not
      happen, so there is no bp-based route to broader coverage.  ? is the only route.

      anything about behaviour.  no capture is attempted.  P4's VALUE is meaningless and
      must not be read as a result.  C2 is a pass, and its value column is noise.

      anything transferable to another cdb build.  the verdict is bound to the pinned hash.
```

## 14. Contract Review checklist

To be executed read-only against THIS contract, before any execution authorization.

```text
  RR-1   the debuggee is fixed by SHA-256, not merely by path
  RR-2   the debuggee is x64, and the bitness requirement is stated
  RR-3   the three AudioEngineHarness locations are explicitly excluded
  RR-4   SHAPE A is fixed and SHAPE B is rejected with reasons
  RR-5   cdb is pinned to the PRODUCTION parser by SHA-256
  RR-6   the executable is placed LAST and nothing follows it
  RR-7   g / go / gc is forbidden, with an abort condition
  RR-8   no bp or bu is used
  RR-9   the probe order is fixed and BOTH negatives precede the target
  RR-10  every probe has a marker, so outcomes are attributed by region
  RR-11  classes C1..C4 are mutually exclusive and C4 covers absence
  RR-12  syntax error is separated from memory / register error, and the substring trap
         is named
  RR-13  C2 and C3 both count as parser-accepted, with the reason
  RR-14  no G condition can yield PASS from an absent or non-discriminating result
  RR-15  abort conditions and the no-retry rule are fixed
  RR-16  artifacts are named and hash-resolved, and no .cdb is created or modified
  RR-17  the contract carries NO execution authorization
  RR-18  the carried-forward prohibitions are present and unchanged
```

## 15. State

```text
Step5EI      DRAFTED, unchanged
Step5EJ      CLOSED  13 PASS / 0 FAIL / 1 PENDING
Step5EK      CLOSED / INCONCLUSIVE.  log retained as evidence.
Step5EK-R1   PRE-EXECUTION CONTRACT  DRAFTED.  Contract Review NOT executed.
             EXECUTION NOT AUTHORIZED.

cdb launches 0   debuggees 0   images executed 0   harness 0
production runs 0   builds 0   .cdb edits 0   source changes 0

CR-3  PENDING.  14/14  NOT MET.  Implementation Authorization  FORBIDDEN.

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
ConvoPeq.md  5,535,334 B  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

```text
held, untouched, and NOT to be folded into R1
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
