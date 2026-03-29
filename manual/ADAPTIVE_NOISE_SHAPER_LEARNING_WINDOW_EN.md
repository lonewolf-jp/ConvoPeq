# Adaptive Noise Shaper Learning Window Manual

The Adaptive Noise Shaper Learning window is a dedicated screen for automatically optimising (learning) the coefficients of the **Adaptive 9th‑order** noise shaper based on the actual playback signal.

> **How to open**: In the Audio Settings screen, set the Noise Shaper type to “Adaptive 9th‑order” and click the “Adaptive learning…” button.

---

## 1. Learning Mode

Select the speed and thoroughness of the learning process.

| Mode | Description |
|------|-------------|
| **Shortest** | Shortest (approx. 10 seconds). Use when you need a quick, coarse coefficient set. |
| **Short** | Short (approx. 30 seconds). Recommended as a standard starting point. |
| **Middle** | Medium (approx. 60 seconds). Expect more stable results. |
| **Long** | Long (approx. 120 seconds). Performs high‑precision adjustment. |
| **Ultra** | Very long (approx. 300 seconds). For the highest quality. |
| **Continuous** | Continuous. Continues learning until you press Stop Learning or exit the application, constantly trying to improve. |

---

## 2. Learning Control Buttons

| Button | Action |
|--------|--------|
| **Start learning** | Starts learning in the selected mode. Discards previous learning results and begins a fresh optimisation. |
| **Stop learning** | Interrupts learning. The current best coefficients are kept; you can resume later with “Resume learning”. |
| **Resume learning** | Resumes a previously interrupted learning session, inheriting the past best coefficients and CMA‑ES state. |

---

## 3. Status Display Area

| Item | Description |
|------|-------------|
| **Status** | Shows the current learning state (Idle / Waiting for audio / Running / Completed / Error). |
| **Format** | Shows the current processing sample rate (Hz) and dither bit depth. |
| **Elapsed audio** | Cumulative playback time (seconds) used for learning. More time means evaluation on more data. |
| **Phase** | Learning phase (1 / 2 / 3). Automatically changes according to mode and elapsed time, affecting the intensity of the search. |
| **Generation** | CMA‑ES generation count (total generations also shown except in Continuous mode). |
| **Process count** | Total number of candidate coefficient sets evaluated so far. |
| **Training segments** | Number of audio segments currently used for training. More segments mean evaluation on more diverse signals. |
| **Best score** | Best evaluation score found so far (lower is better). |
| **Latest score** | Score of the most recent candidate coefficient set. |

---

## 4. Score History Graph (recent score)

- Horizontal axis: learning step (generation or evaluation count)
- Vertical axis: **logarithmic scale** of the best score (lower is better)
- The graph displays the history of best scores, allowing you to visually check the convergence of the learning process.

---

## 5. Advanced Settings (Learning Parameters)

| Item | Description |
|------|-------------|
| **CMA-ES Restarts** | Number of initial exploration runs with different random seeds at the start of learning (1–10). Higher values increase the chance of finding a good initial set but prolong the start‑up phase. |
| **Coeff Safety Margin** | Upper limit for the absolute value of the learned reflection coefficients (parcor), range 0.3–0.95, default 0.85. Lower values improve noise shaper stability but reduce the amount of sound quality improvement. |
| **Enable Stability Check** | When enabled, heavily penalises unstable coefficients (reflection coefficients close to ±1) and excludes them from learning. Recommended to keep always on. |

---

## 6. Learning Flow

1. **Waiting for audio**  
   Stereo audio playback is required. When a sufficiently long and diverse signal (e.g., music) is played, training segments are automatically collected.

2. **Running**  
   The CMA-ES algorithm generates candidate coefficient sets, evaluates them on the actual signal, and iteratively improves them. Scores and the graph update on screen.

3. **Completed / Stopped**  
   When learning finishes automatically because no further improvement is expected, or when you stop it manually, the best coefficients are automatically applied to the **Adaptive 9th‑order** noise shaper and used for all subsequent audio processing. Learning results are saved per sample rate, bit depth, and mode, and persist across application restarts.

---

## 7. How Adaptive 9th‑order Learning Works

The Adaptive 9th‑order noise shaper has a **lattice structure** and operates with 9 parameters called **reflection coefficients (PARCOR)**. The learning function automatically searches for optimal reflection coefficients as follows:

1. **Signal collection**  
   From the stereo playback signal, multiple audio segments with different loudness levels (-40 dBFS to -10 dBFS) and characteristics (transient, tonal, broadband) are extracted.

2. **CMA‑ES optimisation**  
   The Covariance Matrix Adaptation Evolution Strategy (an evolutionary algorithm) generates, evaluates, and updates candidate reflection coefficients. Each generation tests 18 candidates, and the 6 highest‑scoring ones become the parents for the next generation.

3. **Scoring (evaluation)**  
   Each candidate coefficient set is applied to the actual noise shaper, and the quantisation error is analysed. The evaluation takes into account:
   - Frequency‑domain weighting based on the auditory masking curve
   - Time‑domain RMS error
   - Spectral flatness penalty
   - Excess high‑frequency energy penalty
   - Tonal component (peak) penalty

4. **Learning phases**  
   Depending on the elapsed time, the process switches through three phases: “exploration (wide range)” → “convergence (narrow range)” → “fine‑tuning”, leading efficiently to the optimum.

5. **Saving and applying results**  
   The best reflection coefficients are immediately applied to the noise shaper and simultaneously saved to disk. The next time you start the application with the same sample rate, bit depth, and learning mode, they are automatically reloaded.

This learning enables high‑quality, signal‑adaptive noise shaping that fixed‑coefficient noise shapers cannot achieve.

---

## 8. Notes

- If you switch to another noise shaper type while learning is in progress, learning will be interrupted (it can be resumed later).
- Learning requires a **stereo** playback signal. Silence or simple test tones will not produce effective learning.
- Normal audio processing continues during learning, but CPU load increases slightly.
- Learning results are automatically saved to `%APPDATA%\ConvoPeq\learned_state.xml`. You can also manually back up this file.

---

This concludes the description of the Adaptive Noise Shaper Learning window. By using the learning feature, you can achieve optimal noise shaping tailored to your environment and music genre.