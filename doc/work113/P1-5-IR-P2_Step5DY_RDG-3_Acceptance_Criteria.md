# Step5DY — RDG-3 Acceptance Criteria, items 1 to 4

```text
gate                = Repair Design Gate, stage 3 of 4.  ACCEPTANCE CRITERIA.
mode                = READ-ONLY / OWNER DECISION.  No CDB run, no retry, no .cdb edit, no
                      source / test / CMake / build / harness change.
execution this gate = 0
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
disassembly         = build/Release/AudioEngineHarness.exe
                      SHA-256 E5C7AFB9C4EAA48C1ACB031689910F381B38E937A29F16292B0017E959E34C75
                      ImageBase 0x140000000, read with llvm-objdump, no execution
```

## 1. RDG-3-1 — capture marker. OWNER RULED.

```text
RDG-3-1  capture marker  =  C3_T0_CAS_POST  ( T0Q, bp 0x1f9c59a )

  capture event        CAS succeeded and producer ownership of the position-4 slot was obtained
  position identity    @r10 == 5  <=>  pos == 4
  false positive       0, STRUCTURALLY.  0x141f9c59a is reachable only through
                       0x141f9c586 `je`, the sole entry of the cmpxchg success branch.  the
                       diff != 0 paths at 0x141f9c58d (full) and 0x141f9c591 (retry) never
                       reach it.
  cardinality          N
  entry write          AFTER T0Q, at 0x141f9c5a6, 0x141f9c5bf, 0x141f9c5c8
  publication          AFTER T0S, at 0x141f9c5d2
  T0Q -> T0S           the producer-owned slot window
```

The Owner's correction to the earlier wording is adopted: T0Q is **not** "CAS success plus entry
written". It is **ownership acquired, entry write not yet begun**.

## 2. RDG-3-4 — bracket completeness terminology. OWNER RULED.

```text
T0E   acquisition attempt observed
T0Q   slot ownership acquired
T0S   entry publication complete

bracket  =  ownership-to-publication bracket  ( T0Q -> T0S )

"acquisition complete = T0Q" is ABOLISHED.  the term is  ownership acquired = T0Q.

  observation pattern        verdict
  ------------------------   --------------------------------------------
  T0Q -> T0S                 complete bracket
  T0Q only                   incomplete bracket
  T0E only                   attempt only
  T0S only                   orphan publication marker, corresponds to no capture
  T0Q -> T0S, run ends       incomplete

BC  =  every captured T0Q has a corresponding T0S for the same slot-4 acquisition
```

> ### CORRECTED IN Step5EA §4 — the BC and the table above are WITHDRAWN
>
> ```text
> WITHDRAWN   BC = every captured T0Q has a corresponding T0S
>             the table row "T0Q only -> incomplete bracket"
>             the table row "T0S only -> orphan publication marker"
>
>             the BC coupled CAPTURE cardinality to HANDOFF cardinality.  Step5EA-0 established
>             that the handoff is single-shot, so at most one of N captures can have a
>             corresponding T0S.  with N = 4 measured in BOTH runs, this BC was false for 3 of 4
>             captures on every run ever measured.  the Owner ruled it INVALID.
>
>             "T0Q only" is NOT an incomplete capture.  T0Q is itself the acquisition event, so
>             each capture is complete on its own terms; what it lacks is the handoff, and the
>             handoff is single-shot by design, not by any capture defect.
>
> STILL VALID from this section
>   T0E  acquisition attempt observed
>   T0Q  slot ownership acquired
>   T0S  entry publication complete
>   "acquisition complete = T0Q" remains ABOLISHED;  the term is  ownership acquired = T0Q
>   T0Q -> T0S is the internal sequence of ONE handoff episode
> ```
>
> Replacement BC and corrected table: `P1-5-IR-P2_Step5EA_RDG-4_Design_Gate.md` section 4.

## 3. Two RDG-2 uncertainties WITHDRAWN, resolved by disassembly

```text
WITHDRAWN   "r10's meaning is not established, so @r10 == 5 may be unsatisfiable"

  RESOLVED.  on the success path r10d is NOT reloaded.  0x141f9c5cd `incl %r10d` makes it
  pos + 1.  therefore at T0Q and T0S, r10d = pos + 1, and @r10 == 5 is satisfiable and means
  exactly "this call advanced enqueuePos from 4 to 5".  the concern is withdrawn.

  0x141f9c591 `movl 0x34000(%r11), %r10d` reloads enqueuePos only on the diff > 0 retry path,
  which is not the path T0Q or T0S sits on.
```

```text
WITHDRAWN   "rbx's meaning is not established"

  RESOLVED.  0x141f9c563 `andl $0xfff, %ebx` makes rbx = pos & 0xFFF, the slot index.  the
  restoring instruction 0x141f9c5da `movq 0x8(%rsp), %rbx` sits AT the T0S breakpoint address
  and x86 traps before execution, so at the trap rbx still holds pos & 0xFFF = 4.
  @rbx == 4 is the slot index and is correct.
```

## 4. Incidental confirmations from the same disassembly

```text
CONFIRMED  sizeof(DeletionEntry) == 0x30
  0x141f9c59f leaq (%rbx,%rbx,2), %rcx   -> 3 * slot
  0x141f9c5a3 addq %rcx, %rcx             -> 6 * slot
  0x141f9c5a6 movb %al, 0x18(%r11,%rcx,8) -> 48 * slot + 0x18
  48 = 0x30, and type sits at struct offset 0x18.
  this independently confirms CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=OFF, on which every recorded
  offset in this lineage depends.

CONFIRMED  the proven address algebra
  ringBuffer at r11 + 0, sequences at r11 + 0x30000, enqueuePos at r11 + 0x34000.
  0x141f9c569 and 0x141f9c5d2 index sequences with the MASKED slot, rbx & 0xFFF.

UPGRADED  entry `type` is observability  ABI inference -> PROVEN
  0x141f9c59a `movzbl 0x28(%rsp), %eax` reads the 4th argument and 0x141f9c5a6 stores it at
  entry + 0x18.  so `db @rsp+0x28` reads type.  valid at T0Q and also at T0E, since the callee
  writes only [rsp+8] before T0Q.  publicationSequenceId is at [rsp+0x30], generation at
  [rsp+0x38].
```

## 5. RDG-3-2 — TP / FN / FP and N. NOT YET RULED. One finding changes it.

### 5.1 N — already fixed, restated

```text
N  =  the number of queue instances that reach position 4
0 <= N <= 4          upper bound from the source-defined instance cardinality of Step5DW

N is a SYMBOL.  it is not 4.  neither run's outcome becomes a specification, and the
self-validating form "captures == distinct rcx observed in the same log" remains forbidden.
```

### 5.2 FP — zero, structurally

```text
FP  =  a T0Q firing that is not a slot-4 acquisition
     =  0, structurally, by the single-entry argument of section 1.
```

### 5.3 FN — NOT structurally zero. A contention class exists.

This is the new finding. Reading the instruction order at T0Q:

```text
0x141f9c57e   cmpxchgl %ecx, 0x34000(%r11)    enqueuePos : 4 -> 5
0x141f9c586   je 0x141f9c59a                    success
0x141f9c59a   <T0Q trap>                        guard is evaluated HERE
```

Between the cmpxchg retiring and the trap, **another producer can win the CAS for position 5**,
making `enqueuePos == 6`. T0Q's three conjuncts then behave as follows:

```text
conjunct                        under an interleaved producer at position 5
------------------------------   --------------------------------------------
@r10 == 5                        STILL TRUE.  r10d is a register holding pos+1, local to
                                 this thread.  no other thread can change it.
dwo(@r11+0x30000..) == 4         STILL TRUE.  sequences[4] is written only by slot 4's owner,
   that is sequences[4] == 4     which is this thread.  no other thread can change it.
dwo(@r11+0x34000) == 5           FALSE.  enqueuePos is shared.  the competing producer
   that is enqueuePos             advanced it to 6.

=> a genuine slot-4 acquisition is MISSED.  that is a FALSE NEGATIVE.
```

```text
THE STRUCTURAL POINT

  exactly one of T0Q's three conjuncts is contention-sensitive, and it is the shared one.
  the other two are thread-local or owner-exclusive and are immune to any interleaving.

  so the FN class is real, narrow, and UNMEASURED.  the window is the cmpxchg-to-trap distance,
  a few instructions, but it is non-zero on a multi-core producer set.

  the same conjunct at T0S sits roughly fifteen instructions later, so T0S is materially MORE
  exposed to this class than T0Q.  this is a second, independent reason the Owner's RDG-3-1
  choice is the right one, and it further indicts SEQUENCE_PUBLISHED as a capture marker.
```

### 5.4 Why this is an Owner decision and not mine

```text
dropping  dwo(@r11+0x34000) == 5  from the T0Q guard

  closes the FN class completely, because the two surviving conjuncts are both immune.

  but it is not redundant.  it asserts something stronger: not "this call claimed position 4"
  but "enqueuePos is STILL 5", i.e. no other producer has advanced past it.  removing it weakens
  the assertion, even as it strengthens the capture.

  so the choice is between a stricter but loss-prone predicate and a complete but weaker one.
  that is a semantic trade-off, not an implementation detail, and it changes the acceptance
  criterion itself: whether FN == 0 is REQUIRED or merely BOUNDED.
```

## 6. RDG-3 status

```text
RDG-3-1  capture marker          CLOSED   C3_T0_CAS_POST / T0Q
RDG-3-2  TP / FP / FN and N      CLOSED   OWNER RULED in Step5DZ.  FN = 0 required;
                                        contention-sensitive live enqueuePos prohibited as a
                                        required conjunct
RDG-3-3  N                       CLOSED   symbol, 0 <= N <= 4, not a constant
RDG-3-4  bracket completeness    CLOSED   ownership-to-publication bracket, BC = T0Q -> T0S

RDG-3    CLOSED   see Step5DZ for the ruling and for its retroactive effect on the RDG-2
                 candidate set, which supersedes the ordering B > A > C > D on compliance
                 grounds.
```

```text
still outside RDG-3 and NOT decided here
  single-shot vs multi-shot              RDG-4
  candidate B vs C selection             RDG-4
  deletion of the epoch literal          requires a separate Implementation Authorization
  runtime measurement                    requires a separate Runtime Authorization
  T0P / T0Q / T0S conjunct hit rates     OPEN-3, still UNMEASURED
```

```text
CDB execution 0    .cdb 15 unchanged    baselines 7/7 intact
ConvoPeq.md / harness unchanged    git delta 12, all pre-existing
no predicate adopted    no candidate implemented    no repair proposed
```
