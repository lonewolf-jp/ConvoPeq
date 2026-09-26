# RDG-1.2 — Q3 Ruling Brief

```text
gate                = Owner Decision Gate, Q3 ruling brief
mode                = READ-ONLY.  No CDB run, no retry, no .cdb edit, no source / test / CMake /
                      build / harness change.  No predicate selected.  Q3 NOT ruled by the agent.
execution this gate = 0
purpose             = move Q3 from "un-rulable on current information" to "rulable", by
                      determining read-only which of a / b / c / d are decidable, what each
                      commits to, and what remains unknown
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

## 1. Headline

Q3 was previously blocked because option c had "insufficient measurement" and no record defined a
relevant episode. Read-only source work has changed that:

```text
option a   rulable NOW.  its predicate evidence already exists.
option b   rulable NOW.  the quantity is measured at 11/116.
option c   was un-rulable.  is now PARTIALLY rulable: the enum, the propagation chain and the
           observability point are all established.  only the hit rate is missing, and it is
           NOT recoverable from any existing log.
option d   CLOSED as inadmissible, by constraint C6, not by preference.
```

And one structural fact emerged that changes the cost ratio between a and b, in section 3.

## 2. Correction to my own RDG-1 §5.2 before anything else

RDG-1's description of option c asserted:

```text
"the harness output reports worldReclaimCount reaching 10"
```

**That is withdrawn. It has no traceable basis in this corpus.**

```text
checked   all 4 runtime logs in doc/work113, whole-line equality
          Step5DB  184 lines   0 world/reclaim hits
          Step5DE  184 lines   0
          Step5DI 1056 lines   0
          Step5DN 3493 lines   0

          the token 'worldReclaim' appears in 6 .md records of this workstream and in NO log

what is true
          DeferredDeletionQueue.h:270 declares worldReclaimCount_{0}
          DeferredDeletionQueue.h:150,201 increment it when type == World is reclaimed
          it is printed by tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp and
          OdenomCampaignTests.cpp

what is false
          the --measurement=normal run that produced the 116 T0E hits does NOT print it.
          the value "10" has no identifiable source and is not used anywhere in this lineage.
```

This is an instance of the defect family this workstream keeps recording: a quantity carried
forward from an unidentifiable provenance. It is corrected here rather than left standing, and no
conclusion below depends on it.

## 3. FINDING 1 — the epoch literal is gate-wide, and redundant over a complete positional bracket

Extracted from Baseline-2 with a balanced-paren scan, all four T0 sites:

| site | RVA | script state | positional conjuncts | epoch literal | other |
|---|---|---|---|---|---|
| T0E | `0x1f9c550` | `@$t0 == 0` | `dwo(@rcx+0x34000) == 4` | `@r9 == 9` | — |
| T0P | `0x1f9c57d` | — | `dwo(@r11+0x34000) == 4`, `@r10 == 4`, `dwo(@r11+0x30010) == 4` | `@r9 == 9` | — |
| T0Q | `0x1f9c59a` | — | `dwo(@r11+0x34000) == 5`, `@r10 == 5`, `dwo(@r11+0x30010) == 4` | `@r9 == 9` | — |
| T0S | `0x1f9c5da` | `@$t0 == 0` | `dwo(@r11+0x34000) == 5`, `@r10 == 5`, `@rbx == 4`, `dwo(@r11+0x30010) == 5` | `@r9 == 9` | `@r11 != 0`, `r $t0 = 1` |

```text
4 of 4 T0 sites carry @r9 == 9.  It is a phase-wide filter, not a T0E-local condition.

read as one chain, the positional conjuncts form a complete and self-consistent bracket:

  T0E  enqueuePos == 4                      slot 4 is the current position
  T0P  r10 == 4, enqueuePos == 4,
       sequences[4] == 4                    slot 4 is free, CAS is about to claim it
  T0Q  r10 == 5, enqueuePos == 5,
       sequences[4] == 4                    the CAS has executed
  T0S  r10 == 5, rbx == 4, enqueuePos == 5,
       sequences[4] == 5                    slot 4 is published, position advanced to 5

  this bracket fully characterises ONE ring acquisition of ONE slot, from free to published.
  it requires no epoch condition to be complete.
```

```text
CONSEQUENCE

  the epoch literal is ORTHOGONAL to what the phase already measures, and it is the one conjunct
  that never fires: 0 of 232 across two runs.

  therefore Q3 = a and "delete the epoch conjunct" are the SAME ruling, not two decisions.
  under Q3 = a the literal is redundant at all four sites.

  under Q3 = b the literal must instead be REDEFINED at all four sites, and its interaction with
  the positional filter is unmeasured, because the gate has never opened.

  the two live options are not symmetric in cost.  that asymmetry is the substance of the ruling.
```

## 4. FINDING 2 — measurement and gating are already separated at T0E

The T0E line is not one guard. It is two independent parts, and this is why Step5DP's numbers are
trustworthy and why a repair need not re-measure the conjuncts:

```text
chars   0-270   DIAGNOSTICS, unconditional, runs on every hit
                  .echo C3_DG_T0E_HIT
                  .if (@r9 == 9)                    → C3_DG_T0E_R9_PASS / _R9_FAIL
                  .if (dwo(@rcx+0x34000) == 4)      → C3_DG_T0E_EQ_PASS / _EQ_FAIL
                  r
                  dq @rcx-0x1450 L11
                  .echo C3_DG_T0E_WIN_END

chars 271-437   THE GATE
                  .if (@$t0 == 0) { .if (@r9 == 9) { .if (dwo(@rcx+0x34000) == 4) {
                  .echo C3_T0_ENTRY ; r ; dq @rsp L1 ; ln poi(@rsp) ; gc } ... } ... }
```

Verified by whole-line marker counts in the Step5DN log, 3,493 lines:

```text
C3_DG_T0E_HIT           116
C3_DG_T0E_R9_PASS         0
C3_DG_T0E_R9_FAIL       116
C3_DG_T0E_EQ_PASS        4
C3_DG_T0E_EQ_FAIL      112
C3_T0_ENTRY               0
C3_T0_CAS_PRE             0
C3_T0_CAS_POST            0
C3_T0_SEQUENCE_PUBLISHED  0
C3_T0_S_ARRIVED           0
```

The two conjuncts were measured **independently of the gate**, which is why `0/232` and `4/116` are
attributable to the conjuncts rather than to the conjunction. Any RDG-2 candidate may reuse this
diagnostics payload unchanged.

## 5. FINDING 3 — correction to RDG-1's operand inventory for option c

RDG-1 described option c as "`DeletionEntry` type at `+0x18`". That offset is the **struct field**
offset, valid only once the entry has been written into its slot. It is **not** where the type is
read at T0E.

```text
the entry does not exist in memory at T0E.  T0E is the callee's FIRST instruction
(movq %rbx, 0x8(%rsp)); the slot still holds the PREVIOUS occupant, and the incoming entry is
carried in registers.

T0S materialises the incoming entry from registers, which identifies the register parameters:
  r $t4 = @rdx & 0xffffffff ; r $t5 = @rdx >> 32      → ptr
  r $t7 = @r8  & 0xffffffff ; r $t8 = @r8  >> 32      → deleter
  r $t9 = @r9  & 0xffffffff ; r $t10 = @r9 >> 32      → epoch

type is NOT among them.
```

`type` is the 4th parameter of the 6-argument `enqueue`, so by the x64 ABI it arrives on the stack.
At T0E the breakpoint has just written the shadow slot at `rsp+0x8` and has not touched the
incoming-argument area, so the candidate operand is:

```text
db @rsp+0x28      width-exact for a uint8_t
                  (5th arg publicationSequenceId → rsp+0x30, 6th generation → rsp+0x38)

STATUS: ABI INFERENCE, NOT MEASURED, NOT PROVEN.  It is a C4 candidate, not an established one.
```

## 6. Option-by-option ruling readiness

### Q3 = a — a specific ring position — **RULABLE NOW**

```text
target        one ring acquisition of slot 4, bracketed 4 → 5
predicate     already exists and is already measured: enqueuePos == 4, 4/116 in BOTH runs
discriminating yes, and reproducible across two independent runs — the only such evidence held
residual      none for the target.  RDG-2 would compare variants under C1-C7.
what it commits to
              treating the epoch literal as redundant and deleting it at all four sites.  This is
              not a separate decision; it follows from the target.
```

### Q3 = b — a specific temporal relationship — **RULABLE NOW**

```text
target        episodes whose enqueue epoch trails the live global epoch by exactly one, i.e. the
              state in which isOlder(entry.epoch, minReaderEpoch) is false and a reclaim would be
              premature
predicate     not chosen.  the quantity is measured: 11/116, from the r9 and window deltas
discriminating yes at 9.5 %
residual      the epoch conjunct must be redefined at all FOUR sites, not only T0E, and its
              interaction with each site's positional conjuncts is unmeasured because the gate has
              never opened.  the new address rcx-0x1428 would need its own authorisation.
what it commits to
              a phase-wide change with an unmeasured joint hit rate
```

### Q3 = c — a specific entry class — **PARTIALLY RULABLE, was not before**

Established read-only in this gate:

```text
enum          exactly two values.  src/DeferredDeletionQueue.h:20-23
                Generic = 0
                World   = 1

World DOES reach the T0E site.  proven chain, every hop in source:
  AudioEngine.h:3786        retirePublishedRuntimeWorldNonRt, the sole production World producer
  AudioEngine.h:4452        enqueueDeferredDeleteNonRt(world, deleter, DeletionEntryType::World)
  AudioEngine.h:4460-4481   enqueueDeferredDeleteNonRtWithResult → m_retireRouter->enqueueWithRetry
                            (the CALLER'S type is forwarded, not replaced)
  ISRRetireRouter.cpp:302   enqueueWithRetry
  ISRRetireRouter.cpp:321   enqueueRetire(ptr, deleter, epoch, type)
  ISRRetireRouter.cpp:248   provider_->enqueueRetireTyped(ptr, deleter, epoch, type)
  EpochDomain.h:405-408     enqueueRetireTyped → deferredDeletionQueue.enqueue(ptr, deleter, epoch, type)
  0x1f9c550                 the T0E site

Generic also reaches it, by a different route:
  ISRRetireRouter.cpp:281-287  retireRT, the RT lock-free path
  EpochDomain.h:398-400        enqueueRetire → 3-arg enqueue
  DeferredDeletionQueue.h:66   the 3-arg overload forces DeletionEntryType::Generic

  so BOTH enum values occur at T0E.  the class is not degenerate.
```

```text
observability  db @rsp+0x28, ABI-inferred, unverified          (section 5)

hit rate      UNMEASURED, and NOT recoverable from existing data:
              the T0E diagnostics capture registers and the window dq @rcx-0x1450 L11, which
              covers EpochDomain, not the incoming-argument area.  the type distribution was
              never dumped, and no log in the corpus reports it.

source caveat  the codebase states three times that type is not authoritative for lifetime:
                DeferredDeletionQueue.h:22  "telemetry 専用・lifetime authority ではない"
                EpochDomain.h:404          "telemetry metadata のみ・lifetime authority にしない"
                IRetireProvider.h:33       "非交渉条件 1・lifetime authority にしない"
              selecting on type means selecting on a field the source explicitly disclaims as
              authoritative.  this is a caveat, not an exclusion.

what it commits to
              a selector whose firing rate is unknown, on a field declared non-authoritative
```

### Q3 = d — a specific reclaim outcome — **CLOSED, INADMISSIBLE**

```text
a reclaim outcome is a property of T1 and of reader state.  neither exists before T0 opens.
constraint C6 forbids the T0 selector from depending on them.  choosing d would require the Owner
first to dissolve C6, which is a larger ruling than Q3 and is not proposed here.
```

## 7. What remains un-rulable read-only

```text
1  the hit rate of any entry-class selector.  not measurable without a new run, and the existing
   logs do not contain it.  ruling Q3 = c would commit to an unknown rate.

2  whether the positional conjuncts at T0P / T0Q / T0S hold.  the gate has never opened, so they
   have never been evaluated.  sequences[4] == 4 and == 5 are UNMEASURED at every site.

3  whether @rsp+0x28 actually carries type at T0E.  ABI inference only.

4  C7, the false-positive / false-negative definition.  no record defines a relevant retirement
   episode, and that is the substance of Q3 itself.  RDG-2 cannot satisfy C7 without Q3.

none of these four blocks Q3.  items 1 to 3 are RDG-2 inputs, correctly deferred.  item 4 is
resolved by Q3.
```

## 8. The a / b asymmetry, stated plainly

```text
Q3 = a   the phase already measures what the target names.  the epoch literal is redundant and is
         deleted at 4 sites.  the gate's hit rate is bounded by the already-measured 4/116.

Q3 = b   the target names something the phase does not measure.  a new conjunct must be defined at
         4 sites, a new address must be authorised, and the joint hit rate of the new conjunct
         with each site's existing positional conjuncts is unmeasured.

Q3 = c   the target names a class that does reach the site, but the class rate is unmeasured and
         the observability point is inferred rather than proven.

Q3 = d   inadmissible under C6.
```

This is offered as the factual cost comparison, not as a recommendation. The Owner rules.

## 9. State

```text
Q1        CONFIRMED    A, trigger / capture selector
Q2        CONFIRMED    the literal 9 is not established as a baseline specification
Q4-5      CONFIRMED    T0E is not a one-shot latch; T0S is the latch
Q3        UNDECIDED BY DESIGN.  now rulable on a / b, partially on c, d closed.
Q4-1..4   BLOCKED ON Q3
RDG-2     NOT STARTED
RDG-3     NOT STARTED
RDG-4     NOT STARTED
IMPLEMENTATION FORBIDDEN      RUNTIME NOT AUTHORIZED

CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness / cdb.exe unchanged    git delta 12, all pre-existing
no predicate selected    no candidate enumerated    no repair proposed
```

## 10. Next

```text
Q3 = a   |   Q3 = b   |   Q3 = c  (and if c, whether to accept an unmeasured rate)
```

On receipt, Q4-1 to Q4-4 are derived from the target, then RDG-2 compares candidates read-only
under C1 to C7.
