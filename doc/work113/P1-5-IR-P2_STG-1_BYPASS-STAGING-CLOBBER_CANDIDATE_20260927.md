# STG-1 — Bypass Toggle Destroys User Staging Gains (Candidate)

```text
record type      defect candidate.  NOT an implementation.  NOT an authorization.
status           CANDIDATE — Owner GO required before any change.
authority        ConvoPeq.md  5,551,216 B
                 6F1ED4815A4398FA3A5D34C6D72A914B63D9FDFC38E0BB2A6F2B47E743C90FB3
                 Generated 2026-09-27 00:36:31 → regenerated 2026-09-27 20:33:59
                 HEAD e5e2f668
subjects modified  0
```

---

## 1. Defect ID

```text
STG-1 (proposed; Owner confirmation required — no collision check against
future ledger numbering is implied by this file)
```

## 2. Current authority location

```text
trigger sites
  src/audioengine/AudioEngine.Parameters.cpp
    L12568  void AudioEngine::setEqBypassRequested(bool)
    L12575      applyDefaultsForCurrentMode();          // <-- unconditional
    L12579  void AudioEngine::setConvolverBypassRequested(bool)
    L12586      applyDefaultsForCurrentMode();          // <-- unconditional

overwrite site
  src/audioengine/AudioEngine.Parameters.cpp
    L12711  void AudioEngine::applyDefaultsForCurrentMode()
    L12713  if (m_isRestoringState) return;
    L12716  if (autoGainStagingEnabled) return;         // Bug#1 guard
    L12748  publishAtomic(inputHeadroomDb, newInputHeadroomDb, release);
    L12749  publishAtomic(outputMakeupDb, newOutputMakeupDb, release);
    L12750  publishAtomic(convolverInputTrimDb, newConvTrimDb, release);
    L12751  publishAtomic(inputHeadroomGain, ...decibelsToGain...);
    L12752  publishAtomic(outputMakeupGain, ...);
    L12753  publishAtomic(convolverInputTrimGain, ...);
    L12755  publishAtomic(m_currentInputHeadroomDb, ...);   // persisted mirror
    L12756  publishAtomic(m_currentOutputMakeupDb, ...);
    L12757  publishAtomic(m_currentConvInputTrimDb, ...);

user setters (destroyed values originate here)
    L12639  setInputHeadroomDb / L12664 setOutputMakeupDb / L12692 setConvolverInputTrimDb
persistence (loss becomes permanent on save)
    StateIO write L20870-20873 / restore L20749-20756
```

Only two callers of `applyDefaultsForCurrentMode` exist in the authority
(L12575, L12586). No dirty-flag / user-modified mechanism exists anywhere.

## 3. Trigger / precondition

```text
1. AutoGain OFF (default) and not restoring state.
2. User sets staging via setters, e.g. setOutputMakeupDb(+3.0).
3. User toggles either bypass
   (setEqBypassRequested / setConvolverBypassRequested, any direction).
```

## 4. Actual behavior

The toggle publishes the bypass change AND unconditionally overwrites all three
staging gains (+linear gains + persisted mirrors) with mode-branch defaults:

```text
convBypassed && !eqBypassed          → { 0.0,   0.0,  0.0 }
!convBypassed && EQThenConv && !eqBy  → { 0.0, +10.0, -6.0 }
else                                  → {-6.0, +12.0,  0.0 }
```

Concrete trace (PEQ-only → convolver enabled, order EQThenConvolver):

```text
setOutputMakeupDb(+3.0)  → makeup=+3.0, getter +3.0           (L12664-12675)
setConvolverBypassRequested(false)
  → convBypass=false published                              (L12582-12583)
  → applyDefaultsForCurrentMode()                           (L12586)
      guards pass → branch 2 → makeup=+10.0 published       (L12736/L12749)
  → getter now +10.0 (was +3.0)
```

## 5. Expected behavior

A bypass toggle changes bypass routing only. User-set staging (+3.0) is
preserved; getters unchanged; a subsequent session save round-trips +3.0.

## 6. Concrete consequence

```text
- Audible level jump beyond the bypass change itself
  (example above: +7 dB of makeup appearing from nowhere).
- Getter values change without any setter call.
- Save after toggle persists the overwritten values: the user's +3.0 is
  permanently lost (StateIO writes m_current* mirrors).
```

## 7. Defect成立 (why incorrect, not product decision)

```text
(a) work77 Bug#1 establishes the overwrite as conflict-class: the same
    function destroying AutoGain-computed values was reported AS A BUG and
    guarded. The bypass-toggle path performs the identical destruction on
    user-set values with no guard.
(b) work72 describes the defaults as "Auto OFF時のフォールバック"
    (fallback), i.e. values for when nothing is set — not a clobber-everything
    mechanism. No record documents toggle-revoices-staging as intended.
(c) The setters enforce state-dependent validation (e.g. headroom maxDb from
    order/bypass, L12644-12648); the overwrite path bypasses that layer with
    direct writes.
(d) The two existing guards prove the project treats this overwrite as
    destructive; the toggle path is the unguarded third instance of the
    same shape.
```

## 8. Reproduction / counterexample

Deterministic by construction (straight-line NonRT code, no concurrency):

```text
pre:  AutoGain OFF, order EQThenConvolver, convBypassed=true, eqBypassed=false
1.    setOutputMakeupDb(+3.0);  assert getOutputMakeupDb()==+3.0
2.    setConvolverBypassRequested(false);
3.    assert getOutputMakeupDb()==+3.0   // FAILS: actual +10.0
```

(No new code was written to execute this; the path is fully determined by the
cited source. Execution awaits Owner GO under the test gate if required.)

## 9. Why existing tests/contracts do not already close it

A whole-authority search for tests combining bypass toggle with staging values
returns 4 lines, none asserting preservation (measurement-matrix rows with fixed
combos; UI visibility logic). No test, no contract, no comment addresses
toggle-vs-staging interaction. Not covered by work92 / I-5-R1 / D105 / frozen
tracks. Not the §13.5 trim product decision (that concerns trim UNDER bypass;
this concerns destruction OF trim/headroom/makeup BY the toggle).

## 10. Minimal repair direction (proposal only, NOT applied)

```text
Option 1 (minimal): remove the applyDefaultsForCurrentMode() calls at
  L12575 and L12586. The function then has no callers; staging is owned
  solely by the validated setters. Fresh-session values remain the member
  initializers (-6.0 / +12.0 / 0.0).
Option 2 (conservative): gate the calls on "user never set staging"
  (requires new dirty flags — larger change, NOT recommended as minimal).
```

## 11. Risk / ISR impact

```text
RT/ISR boundary : none (all NonRT; ASSERT_NON_RT_THREAD on setters).
Ownership        : none (same atoms, same publishers).
Authority        : none.
Behavioral change: YES — staging no longer resets on bypass toggle.
  Users relying on re-voicing would notice. Flagged, not judged.
Regression test  : toggle-preservation assertions (set → toggle → get),
  plus save→toggle→save→load round-trip.
Risk             : LOW (2 call-site deletions; no logic change).
```

## 12. Required Owner GO

```text
(a) Bug ID confirmation (STG-1 proposed).
(b) Repair option selection (Option 1 recommended).
(c) Implementation + build/test authorization.
```

---

## State

```text
candidate discovery  COMPLETE (1 candidate)
implementation       NOT STARTED
production source    0 changes / test 0 / build 0 / commit 0 / push 0
files created 1 (this record) / files modified 0
```
