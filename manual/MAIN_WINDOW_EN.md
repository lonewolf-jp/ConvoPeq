# ConvoPeq Main Window User Guide

The ConvoPeq main window consists of the following main sections:

1. **Top Control Bar** – Saving/loading presets, switching processing modes, soft clipping, saturation, system status display.
2. **Convolver Control Panel** – Loading room correction FIR file, mix adjustment, phase mode, waveform display, output filters.
3. **EQ Control Panel** – 20-band parametric EQ settings, total gain, AGC, filter structure.
4. **Spectrum Analyzer** – Real‑time frequency analysis, EQ response curve overlay, input/output level meters.

Detailed explanations of each section follow.

---

## 1. Top Control Bar

| Item | Description |
|------|-------------|
| **Save / Load** | Saves/loads all current settings (EQ, convolver, device settings, tail processing parameters, etc.) to/from an XML file. |
| **Processing Mode** | Drop‑down to select one of four signal paths:<br>- **Conv** : Convolver only (EQ bypassed)<br>- **Peq** : EQ only (convolver bypassed)<br>- **Conv->Peq** : Convolver → EQ<br>- **Peq->Conv** : EQ → Convolver<br>※ Changing the order affects timbre and latency. |
| **Soft Clip** | Enables/disables the output soft clipper. When enabled, peaks exceeding 0 dBFS are smoothly rounded to prevent digital clipping. |
| **Sat:** | Sets the amount of asymmetric distortion (saturation) added to the soft clipper (0.0–1.0). Higher values increase tube‑like warmth. |
| **CPU** | Shows the utilisation (%) of the CPU core on which the audio processing thread runs. Audio processing is pinned to a single core; GUI and learning tasks run on other cores. Thus the displayed value reflects only the audio‑dedicated core. |
| **Lat** | Shows the total system latency (milliseconds) and sample count. Includes oversampling and convolver delay. Unlike the convolver‑only delays (Lat A / Lat T), this includes all processing stages (oversampling, EQ, output filters). |
| **Audio Settings** | Opens the audio device settings (input/output, sample rate, buffer size, dither settings, etc.). |
| **?** | Shows version information. |

---

## 2. Convolver Control Panel

### 2.1 Loading and Displaying the room correction FIR file and so on
- **Load IR…** button: Select an room correction FIR file and so on in WAV, AIFF, or FLAC format. Both stereo (2‑channel) and mono (1‑channel) IRs are supported; mono IRs are automatically expanded to both channels. After loading, the waveform is displayed and the IR length is automatically detected.
- **IR Advanced…** button: Opens the detailed settings window (see below).

#### IR Waveform Graph
Displays the time‑domain waveform of the loaded IR (peak absolute amplitude). The horizontal axis is time (ms or s), the vertical axis is normalised amplitude. Allows visual inspection of the decay behaviour.

#### IR Information Label
- **IR name**: The filename (without extension).
- **Auto**: Shows the automatically detected IR length (seconds). If you manually set “IR Length”, `[Manual]` appears and the auto‑detected value is shown in parentheses as a reference.
- **Lat A**: Fixed algorithmic latency of the convolver (milliseconds, sample‑rate converted). Mainly depends on the partition size.
- **Lat T**: Total latency including Lat A plus additional delay caused by the IR peak position (influenced by phase processing). An indication of how much the output signal is delayed.
- **Difference from the top “Lat”**: The above values are convolver‑only latencies. The top “Lat” displays the **system‑wide** latency (converted to the base sample rate) including oversampling, EQ, output filters, etc. Use the top “Lat” when you need the effective end‑to‑end latency.

### 2.2 Basic Controls
- **Mix** slider: Adjusts the balance between dry (original) and wet (convolved) signal (0% = dry only, 100% = wet only).
- **Smoothing** slider: Sets the time (ms) over which mix value changes are smoothed. Prevents clicks or abrupt volume changes. Larger values produce slower transitions.
- **Phase Mode** drop‑down:
  - **As‑Is**: Preserves the original IR phase.
  - **Mixed**: Combines linear phase (good localisation, low phase distortion) in the low frequencies with minimum phase (reduced pre‑ringing) in the high frequencies. This balanced mode works well for a wide range of material (music, movies) and is the recommended default. Transition frequencies can be adjusted in the advanced settings.
  - **Minimum**: Converts the IR to minimum phase. Reduces pre‑ringing but changes phase characteristics.
- **Exp Direct Head** (experimental): When enabled, the early part of the IR (direct sound) is added to the output with zero latency, bypassing the convolution engine’s delay. **The normal wet/dry mix still works**, but the temporal balance changes because the direct sound arrives earlier. Consider this feature when you need the lowest possible latency (e.g., live monitoring). Note that it may affect overall frequency and phase response.

### 2.3 Output Frequency Filters (active only when the convolver is the last stage)
- **HCF** (High‑Cut Filter):  
  - **Sharp**: Steep roll‑off (Butterworth 4th order)  
  - **Natural**: Linkwitz‑Riley 4th order (good phase response, default)  
  - **Soft**: Gentle roll‑off (2nd order, Q=0.5)  
  Cut‑off frequency is set automatically based on sample rate (19 kHz for ≤48 kHz, 22 kHz for higher rates).
- **LCF** (Low‑Cut Filter):  
  - **Natural**: 18 Hz Butterworth 2nd order (default)  
  - **Soft**: 15 Hz 2nd order HPF with Q=0.5  
  Both remove DC offset and very low‑frequency noise.

### 2.4 Advanced Settings Window (IR Advanced…)
- **IR Length**: Manually set the effective IR length (0.5 s to upper limit). The auto‑detected length (Auto) is also shown.
- **Rebuild Debounce**: Waiting time (ms) before an IR rebuild is triggered after parameter changes.
- **Mixed Phase Parameters**:
  - **Mix Start f (f1)**: Frequency where the transition from linear to mixed phase begins.
  - **Mix End f (f2)**: Frequency where the transition from mixed to minimum phase ends.
  - **Mix tau (τ)**: Strength of pre‑ringing suppression (samples).
- **Tail Processing** (new feature):
  - **Tail Mode**:
    - **Air Absorption**: Applies high‑frequency roll‑off to the entire IR, simulating physical air absorption.
    - **Layer Tail Contouring (L1/L2)**: Leaves the beginning (L0) untouched, applies attenuation only to the tail (L1/L2). Preserves the clarity of direct sound and early reflections.
  - **Rolloff Start**: Frequency (Hz) above which roll‑off begins.
  - **Strength**: Roll‑off strength (0.0 = disabled).
  - **Partition Strength**: Multiplier for the L1/L2 strength (Layer Tail Contouring mode only).

---

## 3. EQ Control Panel

### 3.1 Per‑Band Controls
- 20‑band parametric EQ. For each band you can:
  - **ON/OFF**: Enable/disable the band.
  - **Type** (Low Shelf / Peaking / High Shelf / Low Pass / High Pass)
  - **Channel** (Stereo / Left / Right): Select which channel(s) the filter applies to.
  - **Gain (dB)** – Enter a numeric value.
  - **Frequency (Hz)** – Enter a numeric value.
  - **Q** – Enter a numeric value.

### 3.2 Global Controls (top‑right)
- **Total Gain**: Overall output gain of the EQ (dB). Disabled when AGC is on.
- **AGC**: Automatic Gain Control. Compensates for input/output level differences to keep the overall loudness stable.
- **Sat:** Non‑linear saturation amount for the SVF (State Variable Filter). Adds tube‑like harmonics by introducing non‑linearity into the filter’s internal states.
- **Structure**:
  - **Serial**: Bands connected in series (simple, low CPU load).
  - **Parallel**: Bands connected in parallel. Changes phase behaviour; may produce a more natural reverberation feel. (Higher CPU load)
- **Reset**: Resets all bands to default settings (gain 0 dB, Q=0.707; band 1 and band 20 are shelves, the others are peaking).

### 3.3 EQ Output Frequency Filters (active only when EQ is the last stage)
- **HCF** (High‑Cut Filter):  
  - **Sharp**: Steep roll‑off (Butterworth 4th order)  
  - **Natural**: Linkwitz‑Riley 4th order (good phase response, default)  
  - **Soft**: Gentle roll‑off (2nd order, Q=0.5)  
  Cut‑off frequency is set automatically based on sample rate (19 kHz for ≤48 kHz, 22 kHz for higher rates). Additionally, a fixed 20 Hz Butterworth 2nd order high‑pass filter is always inserted in series for speaker protection and subsonic noise removal.

---

## 4. Spectrum Analyzer

### 4.1 Display Area
- **Frequency axis**: Logarithmic scale from 20 Hz to 20 kHz, with a special mapping that slightly expands the low end for better readability.
- **Level axis**: Range –80 dB to +20 dB. Grid lines every 20 dB; the 0 dB line is highlighted in red.
- **Spectrum**: Real‑time FFT (4096 points, 75% overlap) analysis. Bar colours change from blue → cyan → yellow → red as level increases.
- **Peak hold**: A white horizontal line holds the peak level for 1 second, then decays slowly.
- **EQ response curve**: The overall EQ frequency response is overlaid as a white (left channel) and red (right channel) line. Individual band responses are shown in faint colours.
- **Frequency points**: Circles mark the current frequency of each band, giving an immediate overview of EQ settings.

### 4.2 Right‑side Level Meters
- **IN / OUT**: Vertical bars show instantaneous input and output levels (dB). The 0 dB reference line is red; peak values are held as white “needles”.
- **Buttons**: Top‑left “Analyzer: Input / Output” switches the spectrum analysis source between input and output signals.  
  The adjacent “Analyzer: ON/OFF” turns the analyzer display on/off (reduces CPU load when off).

---

## 5. Saving and Loading Presets
- **Save**: Saves all current settings (EQ, convolver, device settings, tail processing parameters, etc.) to an XML file.
- **Load**: Restores the state from a saved XML file. Additionally, the following text files can be loaded (EQ settings only are restored):
  - **Equalizer APO configuration files** (.txt): Basic parametric EQ settings can be imported, though complex configurations (e.g., multiple band combinations) are not fully supported.
  - **AutoEq headphone setting files**: Supports files exported from AutoEq in “Custom Parametric Eq” format. Makes it easy to apply headphone correction curves.

---

## 6. Hints and Cautions
- **IR length**: Cannot be set below 0.5 seconds. Room correction IRs are typically 0.5–1.0 s.
- **Tail processing**: For short IRs (around 0.5 s), the **Layer Tail Contouring** mode is best at preserving direct‑sound clarity.
- **AGC**: When enabled, Total Gain is automatically adjusted and the manual Total Gain control becomes inactive.
- **Processing order**: Placing the convolver first (Conv→Peq) allows the EQ to further shape the reverberation. Placing the EQ first (Peq→Conv) is suitable for adding spatial characteristics after tonal shaping.
- **Latency**: Convolver latency depends on IR length and oversampling settings. The total system latency is displayed in the top bar.

---

This concludes the overview of the ConvoPeq main window. For detailed explanations of individual functions or troubleshooting, please refer to the separate help pages.