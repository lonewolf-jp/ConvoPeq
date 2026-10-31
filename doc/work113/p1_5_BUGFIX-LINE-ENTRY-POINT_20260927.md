# Bug-Fix Line — Verified Entry Point

```text
record type        read-only investigation
question           which document, and where in it, should bug-fix work resume from
lineage            the pre-C3 Option-E / polyphase-gain adoption line
method             git history (624 commits) + all 52 non-C3 records in doc/work113,
                  read in chronological order, then the terminal documents read in full
subjects modified  0
fixes performed    0
```

**No bug was fixed and no fix was authorised here.** This record establishes where the line
stopped and what each candidate entry point actually requires.

---

## 1. The lineage, in order

The pre-C3 bug-fix record corpus is 52 documents. Its final sequence is unambiguous:

```text
2026-09-22 00:18  implementation_report_20260922_c1_ref_fidelity.md
2026-09-22 06:15  remediation_plan_20260922_v3.2_revised.md          C1785D8D4BB9EE24…
2026-09-22 06:57  step25_condition_d_evidence_20260922.md
2026-09-22 06:59  phase0_characterization_report_20260922.md         BB7785FC4E0A1A8F…
2026-09-22 07:19  phase1_preimplementation_audit_20260922.md       CD9070342745D9E5…
2026-09-22 09:58  implementation_report_20260922_p1_1_polyphase_gain.md
2026-09-22 10:05  p1_2_double_compensation_review_20260922.md
2026-09-22 10:32  p1_3_pre_adoption_impact_review_20260922.md
2026-09-22 10:39  p1_4_adoption_decision_package_20260922.md         1B0BEE54DD0E7085…
2026-09-22 11:41  p1_5_adoption_characterization_20260922.md        1ADE32E86F04C220…  ← TERMINAL
2026-09-22 17:38  P1-5-IR-P0_PROTOCOL_RECONCILIATION.md             1520A656DAA91F74…  ← frozen series opens
```

Gates passed on the way, all recorded in the documents themselves:

```text
v3.2 §4      C1 committed 85aa13b9 · C0 committed 88c6f000 · PRECOMMIT GATE 5/5 PASS
phase0       characterization 全 gate PASS 54/54 exit 0
phase1       P1-0 = PASS  (Phase 1 pre-implementation audit, read-only)
P1-4 §10.1   D1 Option E adoption = DECISION-READY
             D2 Default ON        = HOLD
             D3 Flag policy F1/F2/F3 = DECISION-READY
             D4 Calibration       = HOLD
             D5 Release handling  = DECISION-READY
P1-5 §14     P1-5 = HOLD   (D2 / D4 HOLD maintained; D1 / D3 / D5 unchanged)
```

**The authoritative terminal document of the bug-fix line is therefore
`p1_5_adoption_characterization_20260922.md`.** The line was **interrupted, not concluded**:
5 h 57 min after P1-5, `P1-5-IR-P0` opened the C3 series that is now frozen.

### 1.1 Nothing supersedes it

```text
"P1-5 = HOLD"                  appears in 1 record only   (p1_5 itself)
the os4/os8 measured values     appear in 1 record only   (p1_5 itself)
   5.6273 / 8.4083 / 1.9930
"convolver trim under bypass"   2 records  (filter_application_implementation_plan 2026-09-18, p1_5)
"EQ x OS frequency mapping"     6 records  (remediation_plan v1.8 / v1.9 / v2.0, p1_5,
                                               and one C3-series vehicle record)
```

No later document updates, closes or supersedes P1-5's status. The remediation-plan chain
also terminates at v3.2 (2026-09-22 06:15) — nineteen versions exist, v3.2 is the last, and it
is the smallest, being a partial revision that inherits v3.1 by reference.

---

## 2. Two entry points, and the distinction is load-bearing

P1-5 §13 lists six unresolved items. They are not one queue. They split by what they need.

### 2.1 Entry point A — the adoption decision line

```text
document    doc/work113/p1_5_adoption_characterization_20260922.md
section     §14「次工程（提案）」step ①
gate        §13.1
```

P1-5 states the blocker and its own diagnosis:

```text
§13.1   ON build が in-session で確保できず
        AudioEngineHarness のコンパイルで fatal error C1060（コンパイラのヒープ枯渇）。
        初回は OFF build 直後の並行実行、再試行でも同エラーが再発。
        コードの誤りではない。
```

The remedy is already written into the record:

```text
cmake --build build --config Release --target … -- -j 2      (standalone / low parallel)
```

Then, in P1-5's own order:

```text
②  narrowed matrix OFF/ON   ~34 pair + 11 run ≈ 25 min per build
③  IR measurement retry      after the settling fix
④  regenerate listening material with the ON build
```

**What it is:** an environment and sequencing task, not a code change. `C1060` is MSVC
compiler heap exhaustion.

**What it unblocks:** D2 (default ON) and D4 (calibration), both HOLD only for want of ON-side data.

**Separate and already actionable:** D1, D3 and D5 are `DECISION-READY` and await only the
user, per `p1_4_adoption_decision_package_20260922.md` §10.1. They do not need the ON build.

### 2.2 Entry point B — two flag-independent behaviour defects

```text
document    doc/work113/p1_5_adoption_characterization_20260922.md
section     §13.4  and  §13.5
```

P1-5 explicitly fences both outside the adoption decision:

```text
§13.4   本 adoption の範囲外だが、製品影響の観点で別途調査が必要
        本 P1-5 の範囲外・判断しない
§13.5   本 adoption とは無関係の既存挙動として記録
```

**These are not blocked by the ON build, and not gated on D1–D5.** They are the only
unblocked code-level work in the whole bug-fix corpus.

| | §13.4 | §13.5 |
| --- | --- | --- |
| symptom | a `+3 dB @ 1 kHz` request realises as **+1.9930 dB** at os1, **−5.6273 dB** at os4, **−8.4083 dB** at os8 | `convolverInputTrim +6 dB` produces no output change when the convolver is bypassed |
| flag-dependent | no | no |
| P1-5's own hypothesis | the band is processed on the OS-rate side, so the specified frequency is interpreted against the OS rate — equivalent to the peak moving to 250 Hz (os4) and 125 Hz (os8) | trim application sits inside a `!convBypassed` guard |
| status | OPEN, unclosed | OPEN, unclosed |

---

## 3. Source-level localisation of §13.4 — a starting point, not a root cause

The Owner asked where to resume. I looked far enough to say where **not** to start, which is
itself useful. **This is not a root cause and must not be reported as one.**

### 3.1 What the source shows

```text
src/eqprocessor/EQProcessor.Coefficients.cpp:135
    calcBiquadCoeffs(type, freq, gainDb, q, sr)   — comment:「Biquad係数計算 (UI Thread用)」
    ⇒ display / UI path, not the audio path

src/eqprocessor/EQProcessor.Coefficients.cpp:202-206
    calcPeakingBiquad:  w0 = 2*pi*freq / sr
    ⇒ the same 2*pi*f/sr normalisation, on the display path

src/eqprocessor/BandHelper.cpp:29
    svfToDisplayBiquad(svf)   ⇒ the audio path holds EQCoeffsSVF, converted for display

src/eqprocessor/EQProcessor.Coefficients.cpp:354 (comment)
    「processingRate = sr * resolvedOsFactor（呼出し側の責務）」
```

So: **audio uses SVF coefficients; biquads are a display artefact.** A search of
`src/eqprocessor` for the audio SVF design functions locates `calcLowShelfSVF`,
`calcPeakingSVF`, `calcHighShelfSVF`, `calcLowPassSVF` and the high-pass counterpart in
`EQProcessor.Coefficients.cpp` (≈ L172, 206, 238, 271, 302, 336). **Their frequency
normalisation has not been read.**

### 3.2 The most useful single fact found

`EQProcessor.Coefficients.cpp:138-143` already carries a documented, previously-fixed
instance of this defect *shape*:

```text
[FIX] 上限チェックを削除: 以前の sr > 384000.0 は誤り。
オーバーサンプリング使用時 (例: 96kHz × 8 = 768kHz) に sr=48000 へフォールバックし、
zCache (処理srベース) と biquad係数 (48kHzベース) が不整合になり、
表示曲線のピーク周波数が最大16倍ずれる不具合を引き起こしていた。
(例: 100Hz設定 → 1600Hz表示、ユーザー報告: 25-100Hzブーストが800Hz付近に表示)
```

i.e. an oversampling-rate versus base-rate incoherence, previously producing a frequency
error of up to **16×**, fixed by removing the `sr > 384000` fallback.

§13.4's os8 peak displacement is a factor of **8**. That is the same shape and a similar
magnitude. It is a **reason to begin the investigation at the SVF coefficient path and at
`processingRate` / `zCache` coherence, and not at `calcBiquadCoeffs`** — which is display
code that has already been repaired for exactly this class.

Whether §13.4 is a recurrence of that defect, a distinct one, or something else, is not
determined here. P1-5 asked for a separate investigation; this record supplies its starting
point and its first discriminating question.

### 3.3 §13.5 may not be a defect at all

The current source shows the bypass behaviour is **deliberate at the UI level**, not an
oversight in one place:

```text
src/ConvolverControlPanel.cpp:938
    const bool shouldShowTrim = !eqBypassed && !convBypassed && isEqThenConv;
src/DeviceSettings.cpp:669-684     ordering / bypass state machine
src/MainWindow.cpp:1195-1201       same, at the top level
```

The trim control is *hidden* when the convolver is bypassed. Whether the gain should still be
applied to the audio path while the control is hidden is a **product decision, not an
obvious bug**, and P1-5 recorded it only as an observation. It should not be picked up as a
fix on the strength of the measurement alone.

---

## 4. The two entry points, ranked by what they actually require

| | entry point | document + section | needs | blocked by |
| --- | --- | --- | --- | --- |
| A1 | obtain the ON build | p1_5 §14 ① / §13.1 | `-j 2` build, ~25 min/build measurement | nothing — a resource problem |
| A2 | decide D1 / D3 / D5 | p1_4 §10.1 | only the user | nothing — already DECISION-READY |
| B1 | investigate §13.4 | p1_5 §13.4 | source reading in the SVF path; no build | nothing |
| B2 | settle §13.5 | p1_5 §13.5 | a product decision | nothing, but it is a decision not a fix |

**The least-blocked, highest-value opening is B1**, because it is the only item in the corpus
that is a genuine uncharacterised product defect, is not blocked by anything, and is not a
decision. A1 is the correct opening for the adoption line, and it is cheap.

---

## 5. One self-correction

My first pass reported `git log -1 -- doc/work113/p1_4_adoption` as "(no commit)". That was
**my own argument error** — I passed truncated paths, and `git log -- <path>` returns nothing
for a path that does not exist. The documents are tracked; the bug-fix corpus was committed
in `e565019d` (2026-09-27), together with the C3 corpus. Re-checked with full filenames.

This is recorded rather than quietly corrected, because it is the same class of error this
project keeps recording: a conclusion drawn from an instrument that returned an empty set.

---

## 6. Not concluded

```text
that §13.4 has a particular root cause            not determined; SVF path not read
that §13.4 is a recurrence of the 2026 fix        not determined
that §13.5 is a defect                            not determined; may be intended behaviour
that the ON build will succeed with -j 2          not attempted
that D1 / D3 / D5 should be decided a particular way   not my decision
that any fix is authorised                       no authorisation exists
```

## 7. Also open in this lineage, noted not resolved

```text
v3.2 R18-3   the O-18 uncommitted +11/-4 fix to output_sourcecode_markdown.py is lost
             (not on disk, no stash, not on GitHub) and is recorded as ユーザー判断待ち.
             v3.2 §1.6 restored the file from origin/main plus a minimal path-safety
             change.  The original edit content is unrecoverable.  Structurally the same
             failure shape as occurrence #23, in an earlier series.
```

---

## 8. State

```text
authoritative terminal document   p1_5_adoption_characterization_20260922.md
                                  13263 B  1ADE32E86F04C220ACDF402307556702…
lineage status                    INTERRUPTED at P1-5 = HOLD, not concluded
superseded by                     nothing
entry points identified           4   (A1, A2, B1, B2 — see §4)
subjects modified                 0
fixes performed                   0
gate entered                      none
```

The frozen C3 state is untouched and remains as the Owner set it: `N1 OPEN / NOT PROVEN`,
`N2 DOCUMENTED`, `N2-OPEN-1 C2 / REMAINS OPEN`, and all execution gates at STOP.
