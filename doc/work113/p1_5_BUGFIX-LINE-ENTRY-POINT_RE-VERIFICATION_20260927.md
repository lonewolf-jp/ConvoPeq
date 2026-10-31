# Bug-Fix Line — Entry Point RE-VERIFICATION

```text
record type      read-only document and history re-examination
trigger          Owner request, on the basis of INCIDENT-3 (VSCode log forensics)
question         re-verify WHICH DOCUMENT bug-fix work should resume from
answer changed   YES — materially.  See §1.
subjects modified 0
repository state unchanged; no build, no run, no measurement
```

---

## 1. The correction I have to make first

My previous answer (`p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md`) asserted:

> the bug-fix line's authoritative terminal document is `p1_5_adoption_characterization_20260922.md`;
> `P1-5-IR-P0` begins the separate frozen C3 series.

**Both halves of that are wrong.** The line is not two lines, and `p1_5` is not terminal.

The actual chain is **one continuous line of 196 records**, and it runs straight through:

```text
p1_5_adoption_characterization_20260922.md      09-22 11:41   P1-5 = HOLD
  └─ its §5 "IR measurement invalid" is the only item anyone picked up
P1-5-IR-P0_PROTOCOL_RECONCILIATION.md          09-22 17:38   PASS   (record 1/196)
P1-5-IR-P1_MEASUREMENT_PROTOCOL.md             09-22 17:50   PASS   (record 2/196)
P1-5-IR-P2_MEASUREMENT_EVIDENCE.md             09-22 19:02   BLOCKED, 15 deviations
   ├─ P2E SIGSEGV → P2F → P2F3 CDB → P2G heap census
   ├─ P2H ASAN → P2I teardown lifetime → P2J post-fix
P1-5-IR-P2_MEASUREMENT_EVIDENCE_v2.md          09-23 07:32   Step 4-A..4-F COMPLETE
P1-5-IR-P2_Step5_A1_vehicle_recheck.md         09-23 07:44   ← P2 v2's own "A-1"
P1-5-IR-P2_Step5A2_vehicle_source_audit.md     09-23 07:53   ← P2 v2's own "A-2"
P1-5-IR-P2_Step5A3_vehicle_restore_single_run.md   09-23 09:11
P1-5-IR-P2_Step5A4_provenance_vehicle_audit.md     09-23 09:16
P1-5-IR-P2_Step5B_P3-0_attribution_protocol_freeze.md  09-23 09:24   ← P3 ENTERED
P1-5-IR-P2_Step5C..5AO  (P3-1A..R34)           09-23 09:29 .. 09-24 00:25
P1-5-IR-P2_Step5AP..5EI  (FPM / C3-T0 / C3-T1) 09-24 00:40 .. 09-26 22:50
P1-5-IR-P2_PHASE-3A/B/C, Step5EK, INCIDENT-1/2  09-27 00:38 .. 01:56
P1-5-IR-P2_RZ2C / R3 / R3-1 / RY2 / N1 / N2    09-27 10:20 .. 14:50
P1-5-IR-P2_INCIDENT-3_VSCODE-LOG-FORENSICS     09-27 15:32   (this session)
```

**The frozen C3 series IS the bug-fix line.** It is not a separate workstream that
"took over"; it is `p1_5`'s own §5 item, taken through protocol freeze, measurement,
A-1, A-2, and into P3 root-cause attribution, where it has been for four days.

**Why I got it wrong.** I classified records by filename prefix. My prefix set included
`P1-5-IR-P1`, so I filed P0 and P1 under "the frozen series" and excluded them from the
bug-fix corpus. P1 is in fact a **Measurement Protocol Freeze** — the successor to
`p1_5` §13.2's ad-hoc "add a settling wait" proposal — and it explicitly names P2 as its
executor. Excluding P0/P1 from the corpus removed the two documents that connect `p1_5`
to the rest. A classification filter, applied without checking whether the excluded class
was actually the same line, produced a false lineage.

---

## 2. What the C3 line consumed, and what it left untouched

`p1_5` §13 lists **six** unresolved items. Mapping each against all 196 records of the
line that followed:

| §13 | item | fate |
| --- | --- | --- |
| 1 | ON build blocked by `fatal error C1060` | **RESOLVED** — but see §5 |
| 2 | IR + OS measurement invalid (§5) | **SUPERSEDED, then ADVANCED** — P0 → P1 → P2 → A-1 → A-2 → **P3** |
| 3 | measurement rate ~37 s / pair | **RESOLVED by redesign** — P1 froze a 6-run matrix in place of the 34-pair one |
| 4 | **EQ × OS frequency mapping** (§4) | **NEVER PICKED UP** |
| 5 | **Convolver input trim not applied on bypass** (§3 P4) | **NEVER PICKED UP** |
| 6 | **listening material + evaluation** (§9) | **NEVER PICKED UP** |

Items 4–6 were not deferred, not blocked, not carried forward. They were never entered.
The evidence, over the whole corpus:

```text
pattern                                              files
"13.4"                                               0
"13.5"                                               0
calcBiquadCoeffs | calcPeakingBiquad | svfToDisplayBiquad    0
EQProcessor.Coefficients | BandHelper                 0
listening | 聴感                                     1   (2 incidental hits)
trim                                                 19  (ALL false positives — §4)
C1060                                                 0
O-18 | R18-3                                          0
```

**The 19 `trim` hits are false positives, and each was checked individually** rather than
counted as coverage:

```text
checkResampleBoundUsesTrimmedLength                  test function names
checkPreviewResampleBoundUsesTrimmedLength           test function names
state.convolverInputTrimGain / setConvolverInputTrimDb(0)   a gain parameter held at 0 dB
"trimmed content" / "trimmed log lines"              the ordinary sense of the word
headroom/makeup/trim=0.0f, convBypass=true           a DSP matrix row, not a study of trim under bypass
```

Not one addresses `p1_5` §13.5, which is about the **bypass guard**, not the gain value.

---

## 3. The open set is larger than §13 admits

`p1_5`'s own section headings show three further incomplete areas that §13 does not list:

```text
L40  ## 4.  EQ + Oversampling（OFF 側・全 18 条件取得済み）    OFF done · ON side 未取得
L60  ## 6.  Staging × Nonlinear（OFF 側・全 12 条件取得済み）  OFF done · ON side 未取得
L67  ## 7.  Saturation Sweep（OFF 側・部分取得）              PARTIAL
L71  ## 8.  Limiter Engagement Sweep（未実施）                NOT PERFORMED
L76  ## 9.  Listening Evaluation（AB/ABX 素材: 生成済みだが無効）  generated but VOID
```

`P1-5-IR-P1` is titled **"Test environment / Binary identity"** and covers the IR axis
only. So the whole of §4, §6, §7, §8, §9 stands **exactly as `p1_5` left it on
2026-09-22**, and the ON side of the characterization was never obtained at all.

**`§13` therefore under-reports the open set.** It lists the IR item and two secondary
observations; it does not list §7 partial, §8 not performed, §9 void, or the absent ON side.

---

## 4. The answer

**Resume from `doc/work113/p1_5_adoption_characterization_20260922.md`,**
not from any `P1-5-IR-P2_*` record, and not from `p1_4`.

```text
document    doc/work113/p1_5_adoption_characterization_20260922.md
read with   §2  Fixed State / Source Identity     — the fixed state any work must hold
            §4  EQ + Oversampling                 — L40-51, primary evidence for §13.4
            §3  Preset Replay, Case P4 + note     — L30, L36, primary evidence for §13.5
            §7  Saturation Sweep（部分取得）        — L67
            §8  Limiter Engagement Sweep（未実施）   — L71
            §9  Listening Evaluation（素材無効）     — L76
            §13 Unresolved Items                  — L98-106, the register
            §14 PASS / HOLD                       — L107-120, the state that must not be disturbed
```

**Concrete entry points, in the order the evidence supports:**

```text
E-1   §13.4  EQ × OS frequency mapping                    ← the deepest open item
E-2   §13.5  Convolver input trim under bypass            ← a product decision, not a defect
E-3   §8     Limiter Engagement Sweep（未実施）
E-4   §7     Saturation Sweep（部分取得）
E-5   §9     listening material regeneration + evaluation
E-6   ON side of §3 / §4 / §6                             ← needs a build strategy decision first
```

**E-1 is the substantive bug.** `p1_5` §4 records the measurement:

| condition | measured gainDb (OFF) | `p1_5`'s own reading |
| --- | --- | --- |
| os=1, 1 kHz band +3 dB, −20 dBFS | **+1.9930** | 3 − 1.0 = +2.0 ✓ |
| os=1, +6 / +9 dB | +4.9930 / +7.9930 | boost steps exactly +3.000 dB ✓ (decisive control) |
| os=4, +3 dB | **−5.6273** | expected −3.0 → **2.63 dB low** |
| os=8, +3 dB | **−8.4083** | expected −5.5 → **2.91 dB low** |

The os=1 rows are a clean control: the boost lands exactly, so the display/DSP conversion
is right at unity. The deficit appears only once `oversamplingFactor` ≠ 1, and it grows
with it. `p1_5` classified it **OBSERVATION, outside this adoption, flag-independent**.

Note carefully what that is and is not. Earlier in this investigation the os=8 peak
displacement was measured at 8× while a previously-fixed in-source defect was up to 16× —
same shape, different magnitude. **That is an observation about shape, not a root cause,
and it is not offered as one here.**

**E-2 is not a bug.** §3's note records the trim application sitting inside a
`!convBypassed` guard in `DSPCore`. Whether trim *should* apply when the convolver is
bypassed is a **product decision for the Owner**, not a defect to be fixed.

---

## 5. Two things the corpus does not record, which matter

```text
the C1060 remedy is undocumented in this line
    §13 item 1 said ON build fails with fatal error C1060 and prescribed
    `-- -j 2`.  "C1060" appears in 0 of the 196 records.  Yet an ON build was in fact
    obtained: P2 v1 ran 14 runs, P2 v2 ran 6 runs on binary SHA-256[:16]
    2054d970da4ec91c.  The remedy was applied and never written down here.
    → Anyone repeating §13.1 from the record alone will not find the fix that worked.

the v3.2 R18-3 lost fix is still lost
    O-18's uncommitted +11/−4 to output_sourcecode_markdown.py: 0 of 196 records.
    Confirms the earlier finding; not recovered by this re-verification.
```

---

## 6. What this re-verification does not do

```text
no build, no run, no measurement          — this is document examination only
no frozen C3 subject modified             — N1 OPEN/NOT PROVEN, N2 DOCUMENTED,
                                           N2-OPEN-1 C2/OPEN, all gates at STOP
no production / test / CMake / harness change
no attempt to open either line            — §4's entry points are offered, not started
no correctness or safety judgement on any code — classification and description only
no intent or fault finding about e565019d — INCIDENT-3 already scoped itself out of that
```

The prior record `p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md` is **left in place and not
edited**. Its §1 lineage claim is wrong and this record supersedes it; deleting it would
destroy the evidence of the error.

---

## 7. State

```text
lineage                ONE line, 196 P1-5-IR records, 09-22 17:38 .. 09-27 15:32
                       C3 is not separate; it is this line's own §5 item, advanced to P3
resume document        doc/work113/p1_5_adoption_characterization_20260922.md
                       (read with its own §2/§3/§4/§7/§8/§9/§13/§14)
open entry points      E-1 §13.4 · E-2 §13.5 · E-3 §8 · E-4 §7 · E-5 §9 · E-6 ON side
never started          all six.  No fix attempted, none authorized.
frozen C3 state        unchanged
subjects modified      0
gate entered           none
```
