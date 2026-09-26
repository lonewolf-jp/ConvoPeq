# RDG-1 — Semantic Target of the T0 Capture Selector

```text
gate                = Repair Design Gate, stage 1 of 4
mode                = READ-ONLY DESIGN.  No CDB run, no retry, no .cdb edit, no source/test/
                      CMake/build/harness change.  No repair selected, none proposed.
purpose             = define WHAT the T0 selector must identify, before any candidate is compared
inputs              = Step5DQ Intent / Semantic Audit, Step5DP Capture Result, the measured
                      operand inventory at T0E, the archived gate records
Q1 ruling           = A. trigger / capture selector.  CONFIRMED
Q2 ruling           = no record establishes the 9 as a baseline specification.
                      The 9 is an observation-derived snapshot from --fpm-m0.  CONFIRMED
execution this gate = 0
status              = RDG-1 CLOSED.  RDG-2 candidate comparison NOT started.
```

## 1. What Q1 and Q2 settle, and what they do not

```text
SETTLED   the T0 guard is a capture selector, not a production-semantic assertion
SETTLED   the literal 9 is an observation snapshot, not a specified contract
NOT SETTLED   what the selector should select.  That is this document's job.
NOT SETTLED   whether the selector should keep an epoch-based condition at all.
```

The Q1 ruling has a consequence that must be applied rather than merely noted:

```text
"repair @r9 == 9 into the correct form of a production semantic"
is REMOVED from the candidate set, because under Q1 the guard never was a
production-semantic assertion and therefore has no "correct form" of that kind.
```

## 2. The measured constraint that reshapes the candidate space

This is the central finding of RDG-1, and it is measurement rather than opinion.

Every quantity the selector can test at T0E was measured across two `--measurement=normal` runs,
232 T0E hits in total.

| condition | Step5DK | Step5DP | rate | discriminating? |
|---|---|---|---|---|
| `enqueuePos == 4` | 4 / 116 | 4 / 116 | 3.4 %, same count twice | **yes** |
| `r9 == 9` | 0 / 116 | 0 / 116 | 0 of 232 | **no** |
| `r9 == liveGlobalEpoch` | — | 105 / 116 | 91 % | **no** |
| `r9 == liveGlobalEpoch - 1` | — | 11 / 116 | 9.5 % | yes |
| `sequences[4] == 4` | unmeasured | unmeasured | — | unknown |

```text
CRITICAL FINDING

  the intuitive repair, comparing r9 against the live global epoch, would hold at 91 percent
  of hits.  As a TRIGGER it is barely better than the literal it replaces.

  Q1 ruled that the guard is a trigger.  Under Q1, therefore, an epoch-versus-globalEpoch
  condition does not solve the problem the gate was opened to solve.  It would make conjunct 2
  almost always true, which shifts the selector's whole burden onto conjunct 3.

  this is stated as a constraint on the candidate space.  it is NOT a proposal to adopt any
  particular condition.
```

The reason is visible in section 5.4: the quantity `entry.epoch` **is** the current epoch, by
construction, so any condition equating it with the current epoch is close to a tautology at this
site.

## 3. What a selector can test at T0E, bounded by the disassembly

The breakpoint is at `0x1f9c550`, the **first** instruction of the callee. The disassembly of
`src/core/DeferredDeletionQueue.h`'s enqueue path, established at Step5DL:

```text
0x1f9c550  movq %rbx, 0x8(%rsp)           <- the breakpoint fires here
0x1f9c555  movl 0x34000(%rcx), %r10d      <- r10d assigned AFTER the breakpoint
0x1f9c55c  movq %rcx, %r11                <- r11   assigned AFTER the breakpoint
```

```text
usable at T0E
  @r9                              the epoch argument, live, no address needed
  @rcx                             the queue base
  dwo(@rcx+0x34000)                enqueuePos
  dwo(@rcx+0x34040)                dequeuePos
  dwo(@rcx+0x30000 + idx*4)        sequences[idx], idx derivable from enqueuePos
  dwo(@rcx-0x1428)                 live global epoch, but a NEW address, needs its own authorisation

NOT usable at T0E
  @r10   carries an indeterminate leftover caller value.  Step5DP observed 13 distinct values
         ranging from 0x1 to 0x7FF9AE8A0000.
  @r11   likewise.  4 distinct values ranging from 0x1 to 0x7FF9AE8BE717.
         Neither is the queue base nor enqueuePos, because the callee has not assigned either yet.
         This is why T0P, T0Q and T0S use @r11 while T0E must use @rcx.
```

The operand inventory is therefore closed. Any candidate selector is a Boolean combination of the
six usable quantities, and a candidate requiring `@r10` or `@r11` at T0E is not implementable.

## 4. Constraints any candidate must satisfy

These are the conditions a candidate is judged against, fixed before any candidate is enumerated, so
that the comparison cannot be tailored to a favoured answer.

```text
C1  it must DISCRIMINATE under --measurement=normal.  A condition that holds at 91 percent of hits
    is not a selector.  This follows directly from Q1.

C2  its discriminative power must be REPRODUCIBLE.  enqueuePos == 4 gave 4/116 in two independent
    runs, which is the only reproducibility evidence any candidate currently has.

C3  it must not depend on a value observed under a different run mode.  Q2 removed the authority
    of the --fpm-m0 observation, and with it the authority of any literal derived from it.

C4  it must be expressible with the section 3 operand inventory, or declare the new address it
    needs and seek authorisation for it.

C5  it must not break the $t0 sequencer.  conjunct 1 is @$t0 == 0 and the capture is a state
    machine 0 -> 1 -> 2 -> 3 -> 4.  A selector that fires at a rate the sequencer cannot absorb
    is a defect, not a feature.

C6  it must be independent of T1.  0x1f9ce04 and 0x1f9cfb0 remain malformed and unreached, and the
    selector must not depend on state that only exists after T0 opens.

C7  false positives and false negatives must be DEFINED, not assumed.  For a trigger, a false
    positive is a capture taken at an uninteresting episode, and a false negative is a relevant
    episode not captured.  Neither is currently defined in any record.
```

C7 is stated as a requirement rather than met. No record in the corpus defines what a "relevant"
retirement episode is, which is the substance of RDG-1's open question.

## 5. The open question this stage cannot answer for the owner

### 5.1 What is still unknown about the purpose

The workstream's unresolved items are `S7_READER`, `S7_READER_SLOT`, `minReaderEpoch(T1)` and
`Case A / B / C / D`. The C3 capture was titled, in its own gate record, *"S7 Producer / Reader
Temporal Correlation Runtime Capture"*.

```text
What the capture is FOR is recorded as a title and as a list of unresolved items.  It is not
recorded as a specification of which retirement episode matters, or why.

Without that, "which selector is correct" has no criterion.  A selector can be shown to
discriminate, but not to select the RIGHT thing, because the right thing is undefined.
```

### 5.2 The two questions the owner must settle

```text
Q3  Which retirement episodes is the T0 capture supposed to catch?

    The candidates that the measurements make visible, WITHOUT any of them being proposed here:

    a  a specific ring position.  enqueuePos == 4 held 4 of 116 in both runs and is the only
       condition with reproducibility evidence.  It means "the fourth enqueue of the ring", which
       is positional rather than semantic.

    b  a specific temporal relationship.  The 11 of 116 hits where the enqueue epoch trails the
       live global epoch by exactly one.  This is the condition under which a reclaim would be
       premature, i.e. the isOlder test is false.  It is the only measured condition that
       corresponds to a production-meaningful state.

    c  a specific entry class.  The DeletionEntry type at +0x18, and the fact that the reference
       observation had type 0 while the harness output reports worldReclaimCount reaching 10.
       Unmeasured at T0E.

       >>> CORRECTED IN Step5DT.  Two errors in the paragraph above.
       >>>   1  "the harness output reports worldReclaimCount reaching 10" is UNSUPPORTED.  No log
       >>>      in doc/work113 reports worldReclaimCount.  The counter exists in source and is
       >>>      printed by WorldRetirementMeasurementTests / OdenomCampaignTests, but NOT by the
       >>>      --measurement=normal run that produced the 116 T0E hits.  The value 10 is
       >>>      withdrawn.
       >>>   2  "+0x18" is the struct field offset, valid only once the entry is in its slot.  At
       >>>      T0E the slot holds the previous occupant and the entry is still in registers, so
       >>>      type is not read from +0x18 there.  Step5DT traces the propagation chain and shows
       >>>      both enum values do reach T0E, and that the candidate operand is db @rsp+0x28.

    d  a specific reclamation outcome.  Not expressible at T0E, because it depends on T1 state
       which does not exist yet, and C6 forbids depending on it.

Q4  How many captures per run are wanted, and does the $t0 sequencer absorb that rate?
```

Q3 option **b** is the only one of the four whose measured frequency corresponds to a
production-meaningful state rather than an arbitrary position, and it is also the only one that
would make conjunct 2 carry meaning. That is an observation about the candidate space, recorded so
the owner can rule on it. **It is not a proposal**, and no candidate is adopted here.

## 6. What RDG-1 deliberately does not do

```text
does NOT select a candidate
does NOT propose '@r9 == globalEpoch', or any replacement for conjunct 2
does NOT propose dropping conjunct 2
does NOT change conjunct 1 or conjunct 3
does NOT define the false-positive and false-negative criteria, which is Q4's subject
does NOT authorise the new address rcx-0x1428
does NOT re-run, retry, or edit any artifact
```

```text
read-only inputs touched
  Step5DQ, Step5DP, Step5BB Retry-1, Step5CR, Step5CS, Step5BC   records read
  src/core/DeferredDeletionQueue.h, src/core/EpochDomain.h        read
  CDB execution 0    residue 0    baselines 7/7 intact    .cdb 15
```

## 7. One correction to my own reasoning during this gate

While assembling the section 3 inventory I asserted that `r10` and `r11` are "always a stack-range
value" and "always a code-range value" at T0E. The measurement refutes the wording: each carries a
*mixture* of small integers and addresses, 13 and 4 distinct values respectively.

```text
the conclusion is unchanged and rests on the disassembly, which is decisive on its own:
r10d and r11 are assigned at 0x1f9c555 and 0x1f9c55c, both AFTER the breakpoint at 0x1f9c550,
so at T0E they hold indeterminate leftover caller values.

the runtime observation does not support the stronger claim I made, and the claim is withdrawn.
it does support the weaker and sufficient one: neither register carries a meaningful value at
T0E.
```

## 8. State and next

```text
Q1  T0 guard purpose                     A, trigger / capture selector        CONFIRMED
Q2  literal 9 as baseline specification no record establishes it                 CONFIRMED
RDG-1  semantic target                    NOT YET DEFINED, blocked on Q3
RDG-2  candidate comparison               NOT STARTED
RDG-3  acceptance criteria               NOT STARTED
RDG-4  selected design                    NOT STARTED
IMPLEMENTATION                           FORBIDDEN
RUNTIME                                  NOT AUTHORIZED
```

```text
RDG-1 CLOSED as a stage, with the open question named.

next   the owner settles Q3 and Q4.
       Q3 decides what the selector must select, and it is the question no amount of
       measurement can answer, because no record defines a relevant episode.
       RDG-2 cannot begin before Q3, since C1 and C7 have no criterion without it.
```

Nothing in this document should be read as a selected design or as a proposed repair.
