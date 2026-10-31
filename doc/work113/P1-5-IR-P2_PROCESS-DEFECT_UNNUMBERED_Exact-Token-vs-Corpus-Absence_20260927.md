# Process Defect — exact-token absence used as corpus absence

```text
record type      process-defect record.  NOT a ledger.  NOT numbered.
occurrence       NONE ASSIGNED.  no number is granted or implied by this file.
ledger           P1-5-IR-P2_RY2_PROCESS-DEFECT_Gate-Checker-Normaliser.md
                 NOT appended to.  occurrences #1..#23 untouched.
scope            audit method only.  asserts nothing new about any product defect,
                 and changes no classification.
subjects modified  2 existing records read only, neither modified
```

---

## 1. The failure that occurred

A corpus search was run to establish whether a historical frequency/rate defect
existed in `doc/work113`. The search returned a non-empty result set. **The result
set was then converted into an absence conclusion without opening the records it
contained.**

The absence conclusion was reported to the Owner as established fact, and was used
to propose retracting an existing, cited record.

---

## 2. The pattern / notation mismatch

The search was expressed over **token forms**, not over meaning.

```text
patterns used in the failing search
    16\s*[×x倍]        "16x" / "16倍"
    最大\s*16            "最大 16"
```

The substance existed in the corpus in a **different notation**: the phrase
`最大16倍`, a word-plus-number construction rather than a bare `16×` token. A
search keyed on the bare token does not reliably bound the set of places where the
same claim is written down.

```text
generalisable form:
  a search keyed on one surface form of a term does not bound
  the set of locations where that term's meaning is recorded
```

---

## 3. The record that was in the result set and was not opened

The failing search's result set, **as measured at the time of that search**, was:

```text
4 files, 5 hits
    P1-5-IR-P2_Step5CY_…Redesign-6_Result-Audit.md            1 hit
    P1-5-IR-P2_Step5DP_…Baseline-2_r9…Result-Audit.md         1 hit
    p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md                  2 hits
    p1_5_BUGFIX-LINE-ENTRY-POINT_RE-VERIFICATION_20260927.md  1 hit
```

The claim-bearing record, `p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md`, **was in this
set and was not opened.** Its relevant region is §3.2, lines 170–193:

```text
L170  ### 3.2 The most useful single fact found
L193  …point and its first discriminating question.
```

A later co-occurrence probe also returned it:

```text
probe   16× … (修正|fix|repaired|既正|already)   ->  1 file, and it was this one
```

That result was on screen. It was read as "the fix-language co-occurrence exists
only in my own records, therefore there is no external source."

**That inference is invalid, and it is the actual defect.** Authorship of the
*mentioning* record is not evidence about the *sourcing* of the claim inside it. A
record can quote a primary artefact verbatim and cite it by file and line, and
still be a record I wrote. Authorship was used as a proxy for provenance, and the
record that would have settled the question was never opened.

### 3.1 Measurement boundary — the counts above are not reproducible now

This file quotes the search patterns in §2 and §4. **A record that quotes a search
pattern changes the result of that pattern.** Re-running the same sweep after this
file exists returns more files and more hits than the figures above, because this
file is now among them.

```text
before this record existed   4 files,  5 hits
after  this record exists    5 files, 10 hits   (this file contributes 5 and 2)
```

The counts in §3 and §4 are therefore **historical measurements with a stated
boundary**, not reproducible current-state facts. Any future re-check of this
record's own arithmetic must exclude this file, or must expect the inflation.

This is itself an instance of the defect class: a measurement whose instrument
list was altered by the act of writing the measurement down.

---

## 4. The absence conclusion was wrong

The report stated that no corpus evidence existed and that the claim was an
unsourced assertion to be withdrawn. On re-examination the claim **is** recorded
with a citation: the record names a file and line range and reproduces a source
comment verbatim in a fenced block, together with the mechanism, the magnitude and
a user report.

The pattern that surfaces it is a **word-plus-number** form:

```text
最大\s*16   ->  p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md, line 179
16\s*倍     ->  the same file, the same line
```

The information was present and was reachable by search. It was lost at the
interpretation step, not at the retrieval step.

---

## 5. Consequence

```text
a sourced, cited, correctly hedged historical record was
temporarily designated for withdrawal on the strength of a
false absence conclusion, and the withdrawal was put to the
Owner as a recommended action.
```

The Owner had approved the withdrawal. Executing it would have retracted a
documented historical defect on the basis of a search artefact.

**The withdrawal did not happen. No withdrawal record was created.**

---

## 6. How the error was detected

During the subsequent problem-definition audit of `p1_5 §13.4`, the claim was
re-examined because it was load-bearing for the anomaly narrative. An eight-pattern
sweep was run over the corpus — bare tokens, word-plus-number, ratio, and reversed
forms — and the word-plus-number pattern returned the record and line the earlier
search had left unopened.

Detection came from **widening the pattern set**, not from re-reading the earlier
output with more care. The earlier output already contained the filename.

---

## 7. Files that were not modified

```text
p1_5_BUGFIX-LINE-ENTRY-POINT_20260927.md
  43C8BFDAC9B9C52FF9ABB0360A9D2DC66F5E14FAA9251737A94E25B8B8EA5994   unchanged

p1_5_BUGFIX-LINE-ENTRY-POINT_RE-VERIFICATION_20260927.md
  E925D821FEC754A76261328126D507ED24944B3B5FB1CCD378A8A0260206C392   unchanged
```

Both hedges stand as written. Their exact wording, which differs between the two
records and is reproduced here so it is not paraphrased:

```text
first record, L186-L189
  "That is the same shape and a similar magnitude. It is a reason to begin the
   investigation at the SVF coefficient path and at `processingRate` / `zCache`
   coherence, and not at `calcBiquadCoeffs`"

first record, L191-L192
  "Whether §13.4 is a recurrence of that defect, a distinct one, or something
   else, is not determined here."

re-verification record, L167-L169
  "same shape, different magnitude. That is an observation about shape, not a
   root cause, and it is not offered as one here."
```

No hedge was removed and none requires correction. An earlier draft of this record
attributed the phrase "same shape, different magnitude" to the first record; that
attribution was wrong, and the two phrasings are distinguished above.

---

## 8. Rule going forward

```text
an exact-token absence is not evidence of corpus absence.

before reporting that something is absent from the corpus:
  1  the pattern set must cover more than one surface form of the term
     — bare token, word-plus-number, ratio, reversed, and
     hyphenated/parenthesised variants;
  2  every hit the pattern set returns must be opened and read,
     including hits in records authored in the same session;
  3  authorship of the mentioning record must not be used as
     evidence about the sourcing of the claim inside it;
  4  a hit count of zero from a single pattern family is reported
     as "zero for these patterns", never as "absent";
  5  if the claim's own record is among the hits, the claim is
     provisionally sourced-until-checked, not unsourced;
  6  a presence check must tolerate line wrapping — a literal
     substring probe will report FALSE for a phrase that is present
     but broken across lines, which is a false negative of the
     same class;
  7  a presence check must not be a bare literal string when the
     expected text may contain a newline; use a whitespace-tolerant
     pattern;
  8  any count quoted in a record that also quotes the pattern is
     historical. State the measurement boundary, or exclude the
     record from its own arithmetic.
```

Rule 3 is the inference that was violated. Rules 1, 2 and 4 would have prevented
it. Rules 6, 7 and 8 were added after this record's own verification pass
reproduced the same failure mode in miniature — see §9.

---

## 9. Defects found in this record's own first draft

The first draft of this file was self-checked. The check found four problems, three
in the draft and one in the checker. All are recorded rather than quietly fixed.

```text
D-1  §3 and §4 quoted hit counts as if they were reproducible current-state
     facts. They are not: this file quotes the patterns, so re-running the sweep
     inflates the counts. Corrected by adding §3.1 and a measurement boundary,
     and by adding rule 8.

D-2  §7 attributed the phrase "same shape, different magnitude" to the first
     record. That phrase is in the re-verification record; the first record says
     "the same shape and a similar magnitude". Corrected in §7, which now quotes
     both verbatim.

D-3  the first verification pass reported the first record's hedge as absent.
     It was present. The probe was a bare literal substring, and the phrase is
     broken across a line break in the source record. The checker produced a
     FALSE NEGATIVE of exactly the class this document is about, while verifying
     this document about that class. Corrected by using a whitespace-tolerant
     pattern, and captured as rules 6 and 7.

D-4  the same pass reported the RY2 ledger as "modified". The check tested
     `git status --porcelain` output, which is non-empty for an UNTRACKED file.
     The ledger was untracked, not modified; its digest is unchanged. The check's
     logic was wrong, not the file.
```

D-3 is the one that matters most. The instrument used to confirm the absence of a
defect was itself too narrow to find the text it was looking for, and it did so
silently.

---

## 10. What this record does not do

```text
it does not assign an occurrence number
it does not append to the RY2 process-defect ledger
it does not renumber, reuse, or alter occurrences #1..#23
it does not restate, re-claim, or add technical content about the
    historical defect in question — it records only that the audit
    of its existence was performed incorrectly
it does not change the classification of p1_5 §13.4, which remains
    MEASUREMENT OBSERVATION / UNRESOLVED
it does not assert a source location, a cause, or a defect finding
it does not authorise any source audit, build, measurement, or repair
```

---

## 11. State

```text
record type        process defect, unnumbered, standalone
ledger modified    none
occurrences        #1..#23 unchanged; none added
files modified     0
files created      1  (this record)
classification     p1_5 §13.4 = MEASUREMENT OBSERVATION / UNRESOLVED  (unchanged)
gate entered       none
```
