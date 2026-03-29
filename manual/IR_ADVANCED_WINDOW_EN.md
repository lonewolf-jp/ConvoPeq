# IR Advanced Settings – Detailed Explanation

In this window, you can configure detailed processing settings after loading an impulse response (IR).  
Below, the purpose, default values, range, and recommended usage for each control are explained.

---

## 1. IR Length
- **Purpose** : Specifies the effective length (in seconds) of the IR used for convolution. Overrides the automatically detected length, allowing you to manually limit the length.
- **Default** : Automatically detected length (Auto) or 1.0 second (manual)
- **Range** : 0.5 seconds to the maximum allowed value (depends on sample rate)
- **Recommendations** :  
  - For room correction IRs, 0.5–1.0 seconds is appropriate.  
  - For reverberation (reverb) purposes, 1.0–3.0 seconds is typical.  
  - Shorter lengths reduce low‑frequency phase rotation and pre‑ringing but also decrease the sense of reverberation.

---

## 2. Rebuild Debounce
- **Purpose** : Sets the delay (in milliseconds) before an IR rebuild is triggered. Groups consecutive parameter changes to reduce the frequency of resource‑intensive rebuilds.
- **Default** : 400 ms
- **Range** : 50 – 3000 ms
- **Recommendations** :  
  - The default value works well for most cases.  
  - For faster responsiveness during adjustments, set a shorter value (100–200 ms). To minimize CPU load, set a longer value (≥1000 ms).

---

## 3. Mixed Phase Parameters

Mixed Phase is a process that transitions the phase characteristics of the impulse response (IR) from being close to linear phase in the low frequencies to being close to minimum phase in the high frequencies.

- **Linear Phase (As‑Is)** : Delay is constant across frequency, resulting in minimal waveform distortion, but pre‑ringing (ringing that occurs before the sound begins) is more likely.
- **Minimum Phase** : Energy is concentrated toward the beginning of the impulse, suppressing pre‑ringing, but the delay varies with frequency, causing non‑linear phase behavior.

Mixed Phase balances the benefits of both: **low frequencies retain linear phase (less phase shift, better localisation), while high frequencies transition to minimum phase (reduced pre‑ringing)**.  
The frequencies at which this transition starts and ends are specified by `f1` (Mix Start f) and `f2` (Mix End f) below.

### 3.1 Mix Start f (f1)
- **Purpose** : Sets the frequency (in Hz) where the mixed‑phase transition begins. Below this frequency, the phase remains linear (the phase before minimum‑phase conversion).
- **Default** : 200 Hz
- **Range** : 100 – 400 Hz
- **Recommendations** :  
  - Keep the default to preserve low‑frequency phase characteristics.  
  - Lower values (100–150 Hz) increase the amount of phase correction.

### 3.2 Mix End f (f2)
- **Purpose** : Sets the frequency (in Hz) where the mixed‑phase transition ends. Above this frequency, the phase becomes minimum phase.
- **Default** : 1000 Hz
- **Range** : 700 – 1300 Hz
- **Recommendations** :  
  - The default provides a natural phase balance.  
  - Lower values (700–900 Hz) apply minimum‑phase characteristics to a wider high‑frequency range.

### 3.3 Mix tau (τ)
- **Purpose** : Specifies the strength of pre‑ringing suppression during the mixed‑phase conversion. Larger values attenuate early reflections more strongly, reducing pre‑ringing.
- **Default** : 32 samples
- **Range** : 4 – 256 samples
- **Recommendations** :  
  - The default gives a moderate suppression effect.  
  - Increase (64–128) if pre‑ringing is noticeable; decrease (8–16) if you prefer a more natural decay.

---

## 4. Tail Processing – New Feature

Adjusts the high‑frequency characteristics of the latter part (tail) of the IR. You can simulate physical air absorption or preserve the clarity of the direct sound and early reflections while processing only the tail.

### 4.1 Tail Mode
- **Purpose** : Selects the method of tail processing.
- **Options** :
  - **Air Absorption** : Simulates physical air absorption, applying high‑frequency attenuation across the entire IR. Attenuation starts above the roll‑off start frequency and increases continuously with strength.
  - **Layer Tail Contouring (L1/L2)** : Leaves the L0 part (the beginning, corresponding to direct sound and early reflections) untouched, and applies attenuation only to the L1 and L2 layers (the later reverberation tail). Preserves the clarity of the direct sound while smoothing out tail roughness.
- **Default** : Air Absorption
- **Recommendations** (by IR length) :

| IR Length | Recommended Mode | Rolloff Start (Hz) | Strength | Partition Strength (※1) | Notes |
|-----------|------------------|--------------------|----------|--------------------------|-------|
| **0.5 s**  | **Layer Tail Contouring** | 2000 | 0.5 | 1.0 | Preserves direct sound/early reflections completely; most faithful to original. |
|            | Air Absorption | 3500 | 0.3 | — | Gentle overall attenuation while maintaining clarity. |
| **0.6 s**  | Layer Tail Contouring | 2000 | 0.5 | 1.0 | Same as above. |
|            | Air Absorption | 3000 | 0.4 | — | Overall attenuation still keeps clarity. |
| **0.7 s**  | Layer Tail Contouring | 2000 | 0.5 | 1.0 | Same as above. |
|            | Air Absorption | 2500 | 0.5 | — | Default settings give a natural result. |
| **0.8 s**  | Layer Tail Contouring | 2000 | 0.5 | 1.0 | Same as above. |
|            | Air Absorption | 2000 | 0.5 | — | Default settings are sufficiently natural. |
| **0.9 s**  | Layer Tail Contouring | 2000 | 0.5 | 1.0 | Same as above. |
|            | Air Absorption | 2000 | 0.5 | — | Same as above. |
| **≥1.0 s** | Air Absorption | 2000 | 0.5–0.7 | — | Long tail benefits from natural attenuation; adjust strength to taste. |
|            | Layer Tail Contouring | 2000 | 0.5–0.7 | 1.0 | When clarity is a higher priority. |

※1: Partition Strength is effective only in Layer Tail Contouring mode.

### 4.2 Rolloff Start
- **Purpose** : Sets the frequency (in Hz) above which attenuation begins. Frequencies below this point are not attenuated.
- **Default** : 3500 Hz (Air Absorption), 2000 Hz (Layer Tail Contouring)
- **Range** : 20 – 20000 Hz
- **Recommendations** :  
  - When using Air Absorption with a 0.5‑second IR, set to 3500 Hz or higher to maintain clarity.  
  - For longer IRs (≥1.0 s), 2000 Hz gives a natural roll‑off.

### 4.3 Strength
- **Purpose** : Determines the amount of attenuation for frequencies above the roll‑off start. Larger values produce stronger high‑frequency attenuation.
- **Default** : 0.3 (Air Absorption), 0.5 (Layer Tail Contouring)
- **Range** : 0.0 – 2.0
- **Recommendations** :  
  - For Air Absorption with a 0.5‑second IR, keep strength around 0.3 to preserve clarity.  
  - For longer IRs (≥1.0 s), 0.5–0.7 is natural.  
  - Setting to 0.0 disables attenuation (for backward compatibility with old presets).

### 4.4 Partition Strength – Layer Tail Contouring Mode Only
- **Purpose** : Multiplier for the attenuation strength applied to the L1 and L2 layers (the tail). The effective strength becomes `Strength × Partition Strength`.
- **Default** : 1.0
- **Range** : 0.0 – 2.0
- **Recommendations** :  
  - The default works well.  
  - Increase to 1.2–1.5 for stronger tail attenuation; decrease to 0.5–0.8 for milder effect.

---

## 5. Relationship with HCF / LCF / EQ LPF
- **HCF (High‑Cut Filter)** , **LCF (Low‑Cut Filter)** , and **EQ LPF (EQ Low‑Pass Filter)** are real‑time filters applied to the output signal (in the Audio Thread).  
- Tail processing (Air Absorption / Layer Tail Contouring) is “baked into” the IR data and therefore operates independently from these real‑time filters.  
- When adjusting overall high‑frequency attenuation, start by setting the tail processing to achieve the desired character, then use HCF or EQ LPF for fine‑tuning if needed.