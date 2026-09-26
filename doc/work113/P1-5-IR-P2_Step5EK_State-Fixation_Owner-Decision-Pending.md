# Step5EK — STATE FIXATION / Owner decision PENDING

```text
gate                = state fixation.  NO execution.  NO cdb launch.  NO command run against any gate.
Step5EK             = CLOSED / INCONCLUSIVE
CR-3                = PENDING          (NOT promoted, NOT failed)
CR tally            = 13 PASS / 0 FAIL / 1 PENDING
14/14               = NOT MET
Implementation Authorization = NOT issuable.  remains FORBIDDEN.
next                = OWNER DECISION among OPTION 1 / OPTION 2 / OPTION 3
authority           = ConvoPeq.md  5,535,334 B
                      SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

## 1. Fixed state

```text
  Step5EI   DRAFTED, frozen
            13,126 B   sha256 3E513B3973936CAEE4BFE472FC8D503C5A25D7051EC5B88D3C03BF1928A6ADA5

  Step5EJ   CLOSED as a review
            13 PASS / 0 FAIL / 1 PENDING
            10,695 B   sha256 8DF5FE802AC2964C70213467...

  Step5EK   CLOSED / INCONCLUSIVE
            the structural blocker  "no image -> no user-mode command loop"  is PROVEN.
            the target expression  dwo(@r11+0x30010) == @r10  remains UNTESTED.
            record  9,717 B   sha256 5BF23944803EB1F82D74...
            log     1,275 B   sha256 A08F829D5CF4806A6BD4...
            the log is RETAINED as the INCONCLUSIVE evidence.  it is not invalidated.

  CR-3      PENDING
```

## 2. Standing prohibitions, in force until the Owner changes them

Recorded so that a later step cannot drift into any of these by inertia.

```text
  PROHIBITED, explicitly, by Owner instruction
      x  re-running the same no-image CDB invocation
         reason  the structural blocker is already proven.  a retry spends a second
                 invocation to obtain the same null result.  this is the re-run pattern the
                 workstream exists to prevent.
      x  starting T0P generalisation
      x  making the T0Q dd probes unconditional
      x  repairing the malformed T1 sites
      x  folding the 65 versus 116 question into any current gate
      x  changing the Step5EI predicate
      x  issuing Implementation Authorization
      x  editing any production .cdb
      x  changing source
      x  building
```

## 3. The three options, as fixed

```text
  OPTION 1   a NEW authorization, then a SEPARATE gate
             Step5EK-R1  CR-3 User-Mode Parser Validation
             it is NOT a re-run of Step5EK.  it is a new gate whose purpose is to remove the
             structural blocker, not to repeat the failed attempt.

             proposed scope, for the Owner to fix or amend
                 debuggee            1, and NOT AudioEngineHarness
                 AudioEngineHarness   forbidden
                 production runtime   forbidden
                 build                forbidden
                 source edit          forbidden
                 .cdb edit            forbidden
                 .cdb corpus change   forbidden
                 g  (continue)        forbidden
                 probes               P1 / P2 / P3 / P4 only

             flow   debuggee attach -> P1/P2/P3/P4 -> G-1 / G-2 / G-3 -> CR-3 PASS or FAIL

  OPTION 2   record CR-3 formally as
                 PENDING  +  accepted unverified risk
             and resolve it by observation at the next real runtime gate, using the four-way
             A / B / C / D discrimination.

             THIS DOES NOT MAKE CR-3 PASS.  stated plainly:
                 13 PASS + 1 PENDING  is  not  14/14
                 so Implementation Authorization remains unavailable under this option too.

             the A/B/C/D table shows the syntax problem WOULD BE IDENTIFIABLE at runtime.
             that is a statement about identifiability.  it is NOT evidence that the syntax
             has been validated.  the two must not be conflated.

  OPTION 3   continue to HOLD.  most conservative.  no new authority, no execution.
```

## 4. Facts prepared for the Step5EK-R1 contract, gathered read-only

Gathered so that the contract is written against verified paths rather than guesses. This
is input, NOT a decision and NOT the contract itself. The contract is written only after a
grant, per the Owner's sequence.

```text
  the structural blocker, restated
      cdb with no image falls through to KERNEL debugger initialisation, fails on \\.\com1,
      and exits 0x80070002.  a user-mode command loop requires an image or a live process.
      one of the two is unavoidable, which is why OPTION 1 needs a new grant.

  candidate non-harness images, existence VERIFIED this step
      C:\Windows\System32\hostname.exe    40,960 B
      C:\Windows\System32\find.exe        40,960 B
      C:\Windows\System32\where.exe       65,536 B
      C:\Windows\System32\whoami.exe      98,304 B
      C:\Windows\System32\verifier.exe   214,440 B
      C:\Windows\System32\mkdir.exe       absent, ruled out

  the forbidden image, for contrast, located so that it cannot be selected by accident
      build\Release\AudioEngineHarness.exe              41,216,512 B
      build-asan\Debug\AudioEngineHarness.exe           98,243,072 B
      build-asan\RelWithDebInfo\AudioEngineHarness.exe  73,467,392 B

  two shapes the R1 contract will have to choose between, neither chosen here

    SHAPE A  launch a trivial image under cdb
             cdb -logo <log> -cf <probes> C:\Windows\System32\hostname.exe
             cdb starts the process and stops at the initial breakpoint.  a user-mode command
             loop exists, registers exist, no live system process is disturbed, and the
             process dies on quit.
             advantages  deterministic fresh state, nothing pre-existing is touched
             caveat      it is still an image and still a debuggee, so it needs the grant

    SHAPE B  attach non-invasively to a process that is already running
             cdb -pv -p <pid>
             -pv is the non-invasive attach, which is the property that makes an attach to a
             live system process defensible.
             advantages  no new process is created
             caveats     a live system process is the subject, and the choice among 423
                         running processes is a real decision the contract must make rather
                         than inherit

  neither shape uses g.  neither shape touches AudioEngineHarness.  neither builds anything.

  one technical caveat the R1 contract must record
      under EITHER shape, ? evaluates against the debuggee.  @r10 takes that process's r10,
      and dwo(@r11+0x30010) attempts to read that process's memory.
      so P4 may produce a memory or register error rather than a value.
      that is acceptable ONLY because the discriminator is binary and syntactic:
          'Syntax error at ...'   ->  the parser REJECTED the form
          no 'Syntax error'       ->  the parser ACCEPTED the form
      G-1 and G-2 exist precisely to prove the harness can see a Syntax error at all before
      P4 is read.  a P4 that errors for a memory reason is NOT a parse failure, and the
      contract must say so explicitly, or the gate will manufacture a false FAIL.
```

## 5. Sequence, unchanged, for whichever option is chosen

```text
  OPTION 1
      Owner explicit GRANT for Step5EK-R1
          -> Step5EK-R1 PRE-EXECUTION CONTRACT, fixing
                 debuggee identity, attach method, probe script, abort conditions,
                 artifacts, prohibitions
          -> execute
          -> CR-3 PASS or FAIL on evidence
          -> only on PASS does 14/14 become reachable, and that is still a SEPARATE
             authorization gate before any implementation

  OPTION 2
      record CR-3 as PENDING + accepted unverified risk
          -> 14/14 stays unreachable
          -> the syntax question moves to the runtime gate, carrying the A/B/C/D table and
             the knowledge that a failure there consumes the single authorized run

  OPTION 3
      hold.  nothing is spent.
```

## 6. State

```text
no command was run against any gate this step.  cdb was not launched.
debuggee 0   image 0   harness 0   production run 0   build 0   .cdb edit 0   source 0

frozen 7 + Baseline-3 = 8/8 intact    .cdb corpus 16, unmodified
ConvoPeq.md  5,535,334 B  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
git delta 12, all pre-existing    proc residue 0    tmp residue 0

Step5EK  CLOSED / INCONCLUSIVE
CR-3     PENDING
14/14    NOT MET
Implementation Authorization  NOT issuable

HOLD.  the only outstanding action is the Owner's choice among section 3.
```

```text
held, untouched, and not to be folded into any option above
    1  T0P sequence conjunct generalisation
    2  the sequence operand not being observable on a non-accepting T0Q hit
    3  the two malformed T1 sites
    4  the 65 versus 116 hit non-comparability
```
