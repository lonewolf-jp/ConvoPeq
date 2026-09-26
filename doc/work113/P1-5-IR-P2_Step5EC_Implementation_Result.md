# Step5EC — Implementation Executed, I-1..I-13 PASS, STOPPED before Runtime

```text
gate                = Implementation, under the Step5EB authorization, scope F-C
mode                = .cdb edit only.  No source / test / CMake / harness edit.  No build.
                      NO RUNTIME EXECUTION.
execution           = CDB 0, harness 0, build 0
authorization       = "IMPLEMENTATION AUTHORIZATION: GO", Owner-issued
authority           = ConvoPeq.md  SHA-256 E5E74200F12784FDF37BE24864F5E4B1EA486255DAD7432B9B9D728B06EF3609
```

## 1. Artifact lineage, resolved by SHA-256 and never by transcribed filename

```text
parent   P1-5-IR-P2_Step5DN_...-Baseline-2.cdb
         14,968 B   SHA-256 BD5129A3655FA04FB1EAF8539BA596BC47179FABD8817B85F3742216F083B763
         UNCHANGED.  re-verified after the edit.

child    P1-5-IR-P2_Step5EC_...-Baseline-3.cdb
         14,819 B   SHA-256 EE82A517698C3257B5612D0EE9957B0AB7E0E027997AE0B4BE93C69947035E58

net      -149 B  =  -34  T0E  -34  T0P  -47  T0Q  -34  T0S
```

## 2. NO-GO EVENT — the first attempt was defective and was discarded

This must be on the record. It was not a scope violation and the parent was never at risk, but a
corrupt artifact was produced and then removed.

```text
WHAT HAPPENED
  the first edit produced a child with only 12 bp lines instead of 13.
  T0E at 0x1f9c550 was GONE.  its line had lost the wrapper and read as a bare body
  beginning with  .echo C3_DG_T0E_HIT  and ending in a stray closing quote.

ROOT CAUSE
  a variable-shadowing bug in my own edit script.  the variable holding the LINE prefix
  ( bp ... " ) was rebound inside the T0E branch to a HEAD slice, so the reassembly
  step  pre + after_body + post  used the head slice as the line prefix.

HOW IT WAS CAUGHT
  the I-13 read-only diff audit.  the differing-line RVA list came back as
  [None, '1f9c57d', '1f9c59a', '1f9c5da'] instead of the four expected T0 RVAs, because
  line 25 no longer parsed as a bp line.  I-13 is the check the Owner designated as the
  decisive gate, and it did its job.

ACTION TAKEN
  parent verified byte-identical BEFORE any cleanup
  defective child deleted, corpus returned to 15
  the shadowing fixed, and an assertion added so the wrapper cannot be lost again:
      assert after.startswith(before[:q1+1])
      assert after.count('"') == before.count('"')
  regenerated, re-validated.  34 checks, 0 failures.
```

```text
ALSO CAUGHT BEFORE ANY WRITE, by assertions in the edit script
  1  the ladder matcher assumed the wrong unit shape,  ; .else { gc }  instead of
     ; .else { gc } } .  it failed twice before I stopped guessing and read the bytes.
  2  the ladder was located with rindex("gc"), but "gc" is a SUBSTRING of "{ gc }", so it
     matched inside the else-ladder rather than at the payload terminator.  replaced with an
     end-anchored pattern.
  3  the transform was applied to the whole .cdb LINE, whose trailing closing quote broke the
     $ anchor.  the bp body is now transformed and the wrapper reassembled byte for byte.
  4  I had set T0Q's ladder drop to 2, but the enqueuePos-to-@rbx change is an IN-PLACE SWAP,
     not a removal, so the drop is 1.  the guard-depth assertion caught my error before any
     write.
```

## 3. The change, per line

```text
site  RVA        line  guard .if   diagnostic .if   total .if   body bytes
T0E   0x1f9c550    25   3 -> 2      2  (kept)        5 -> 4      438 -> 404
T0P   0x1f9c57d    26   4 -> 3      0               4 -> 3      253 -> 219
T0Q   0x1f9c59a    27   4 -> 3      0               4 -> 3      254 -> 207
T0S   0x1f9c5da    28   7 -> 6      0               7 -> 6     2231 -> 2197
```

### 3.1 T0Q, the formal capture predicate, after the edit

```text
.echo C3_DG_T0Q_HIT ; .if (@r10 == 5) { .if (@rbx == 4) { .if (dwo(@r11+0x30010) == 4) { .echo C3_T0_CAS_POST; r; dd @r11+0x34000 L1; dd @r11+0x30010 L1; gc } ; .else { gc } } ; .else { gc } } ; .else { gc }
```

This is exactly Candidate E, and the `dd @r11+0x34000 L1` diagnostic is preserved inside the
payload, so RDG-4-3's KEEP is met by preservation.

## 4. I-criteria, all PASS

```text
I-1   the four markers are each present exactly once in both parent and child
        C3_T0_ENTRY  C3_T0_CAS_PRE  C3_T0_CAS_POST  C3_T0_SEQUENCE_PUBLISHED

I-2   T0Q guard conjuncts, by balanced-paren extraction
        ['@r10 == 5', '@rbx == 4', 'dwo(@r11+0x30010) == 4']      PASS

I-3   'dwo(@r11+0x34000) == 5' absent from the T0Q guard          PASS

I-4   T0Q payload byte-identical to the parent, 67 B, and contains
        'dd @r11+0x34000 L1'                                      PASS

I-5   per-line .if == .else and the ladder is the tail, for all four sites
        T0E 4/4   T0P 3/3   T0Q 3/3   T0S 6/6                      PASS

I-6   T0S payload byte-identical, 1,939 B, all spot-checks present:
        r $t0=1, r $t11=0, r $t12=0, r $t17=0, r $t19=0,
        C3_T0_SLOTS_BEGIN, C3_T0_SLOTS_END                         PASS

I-7   exactly one site writes $t0 = 1, and it is 0x1f9c5da, same as the parent   PASS

I-8   every T1 / S7 line byte-identical to the parent
        0x1f9cfb0 721 B   0x1f9cd00 672 B   0x1f9ce04 770 B
        0x1fa80b2 1880 B  0x1fa80b3 1848 B                        PASS

I-9   whole-file @r9 == 9 census = 1, at 0x1f9c550, of which
        diagnostic-only = 1, guard = 0                             PASS

I-10  source / test / CMake / build.bat git delta = 12, all pre-existing, no new entries
        ConvoPeq.md MATCH   harness MATCH                          PASS

I-11  frozen baselines 7/7 intact; .cdb corpus 15 -> 16, exactly one new file   PASS

I-12  child 14,819 B, SHA-256 EE82A517...35E58
        parent 14,968 B, SHA-256 BD5129A3...83B763, unchanged     PASS

I-13  READ-ONLY DIFF AUDIT, the Owner's decisive gate                PASS
        line count        41 / 41
        differing indices 24, 25, 26, 27  (1-based 25, 26, 27, 28)
        their RVAs        1f9c550, 1f9c57d, 1f9c59a, 1f9c5da
        expected          1f9c550, 1f9c57d, 1f9c59a, 1f9c5da
        identical lines   37 of 41
        any other line differing   NO

C-a   guard and diagnostic conjuncts counted SEPARATELY, by balanced-paren extraction
        T0E  guard 2 = [@$t0 == 0, dwo(@rcx+0x34000) == 4]
             diagnostic-only 2 = [@r9 == 9, dwo(@rcx+0x34000) == 4]   both retained
             total 4
        T0P  guard 3   diagnostic-only 0   total 3
        T0Q  guard 3   diagnostic-only 0   total 3
        T0S  guard 6   diagnostic-only 0   total 6
```

```text
checks run 34      failures 0      RESULT ALL PASS
```

## 5. Deliberately NOT done

```text
no CDB execution        no harness execution      no measurement
no build                no source / test / CMake / harness edit
no OPEN-3 conjunct hit-rate measurement
no T1 site repair.  0x1f9ce04 and 0x1f9cfb0 keep their 2 x dd( each
no S7 reader semantics.  $t17 and $t19 untouched
no entry.type / publicationSequenceId / generation observation
residue processes 0
```

## 6. State

```text
Implementation        EXECUTED, I-1..I-13 ALL PASS
Runtime               NOT AUTHORIZED
Build                 NOT INVOKED
baselines 7/7 intact  .cdb corpus 16
ConvoPeq.md MATCH     harness MATCH    git delta 12, all pre-existing
STOPPED at step 7 of the authorized sequence, as instructed
```

## 7. What the next gate must decide

```text
Runtime Authorization is required before anything is executed.  when granted, the first
measurement asserts only:

  every accepted T0Q satisfies Candidate E
  no non-slot-4 T0Q is accepted
  FN == 0
  the marker cardinalities of T0E, T0P, T0Q and T0S
  @r10, @rbx, sequences[4] and live enqueuePos at each accepted T0Q
  whether live enqueuePos ever reads other than 5, which is the only evidence that can
  falsify FN == 0 after the fact

  N = number of queue instances reaching position 4, 0 <= N <= 4
  N is a SYMBOL.  the acceptance is NOT observed N == 4.

  and if SEQUENCE_PUBLISHED is absent while CAS_POST is present N times, that is the Step5EB
  section 4 residual at T0S, not a capture defect.
```

OPEN-3, the conjunct hit rates, is what the runtime gate exists to close. It remains unmeasured.
