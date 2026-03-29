# Audio Settings Screen Manual

The Audio Settings screen allows you to configure the audio device, signal processing precision (dither / noise shaper), oversampling, input/output level adjustments, and more.

> **Note**: This application does **not** support MIDI functionality. MIDI device input/output, MIDI clock, MIDI learn, and similar features are not available.

---

## 1. Device Settings

| Item | Description |
|------|-------------|
| **Audio Device** | Select the audio driver/interface to use (ASIO, WASAPI, etc.). |
| **Input / Output Channels** | Set the number of active input and output channels (1–2). |
| **Sample Rate** | Specify the processing reference frequency (e.g., 44.1 kHz, 48 kHz). Higher rates improve high‑frequency reproduction but increase CPU load. |
| **Buffer Size** | The size of the audio buffer (in samples). Smaller values reduce latency but raise the risk of dropouts. Increase this value if you experience instability. |

---

## 2. Oversampling Filter Type (IIR / FIR Tabs)

The tabs at the top of the screen let you choose the filter characteristics used during oversampling. The selected type is reflected in the “Type” field of the Oversampling section.

| Tab | Name | Description |
|-----|------|-------------|
| **IIR** | IIR (Low Latency) | Low‑latency type. Slightly affects phase, but has a lower computational load than FIR and minimises latency. |
| **FIR** | Linear Phase (FIR) | Linear‑phase type. Introduces very little phase distortion and preserves waveform integrity, but has higher latency than IIR. |

---

## 3. Dither / Noise Shaper

| Item | Description |
|------|-------------|
| **Bit Depth** | Sets the target output bit depth (16 / 24 / 32 / Off). Higher values preserve more dynamic range. Normally match the OS output setting (often 24‑bit or 32‑bit). “Off” may cause quantisation distortion with low‑level signals. |
| **Noise Shaper Type** | Selects the quantisation noise‑shaping algorithm.<br>- **4th‑order**: 4‑tap error‑feedback type, lightweight.<br>- **12th‑order**: 12‑tap, high‑quality shaping considering auditory masking.<br>- **15th‑order**: 15‑tap, even stronger shaping.<br>- **Adaptive 9th‑order**: 9‑th order lattice filter + CMA‑ES real‑time learning. |
| **Adaptive learning… button** | Enabled when “Adaptive 9th‑order” is selected. Opens a dedicated learning window that automatically finds optimal coefficients based on the current playback signal. |

---

## 4. Oversampling

| Item | Description |
|------|-------------|
| **Type** | Displays the filter type selected in the Oversampling Filter Type tabs (IIR or Linear Phase). |
| **Factor** | Choose from Auto / 1x / 2x / 4x / 8x. **Auto automatically selects the highest factor allowed for the current sample rate** (8x up to 96 kHz, 4x up to 192 kHz, 2x up to 384 kHz, 1x above 384 kHz).<br>**Visible factor limit**: The combo box shows only factors that are allowed for the current sample rate:<br>- ≤96 kHz: up to 8x<br>- ≤192 kHz: up to 4x<br>- ≤384 kHz: up to 2x<br>- >384 kHz: only 1x<br>Factors beyond the limit are not shown and cannot be selected. |

---

## 5. Gain Staging

| Item | Description |
|------|-------------|
| **Input Headroom** | -12.0 to 0.0 dB. Attenuates the input level before processing to prevent internal overflow (clipping). The upper limit is automatically adjusted according to the current processing mode. |
| **Output Makeup** | -6.0 to 12.0 dB. Amplifies the final output to compensate for level loss caused by processing. The range also changes depending on the mode. |

> **Note**: Loaded IR (impulse response) files are internally **automatically attenuated by -6 dB** to prevent overflow. Therefore, even if you set Input Headroom and Output Makeup to 0 dB, the IR’s peak level is already adjusted to -6 dB. This automatic attenuation is part of the IR energy normalisation and does not require user intervention.

---

## 6. Usage Tips

- **Latency vs. Stability**  
  If you experience dropouts or noise, increase the **Buffer Size**. For real‑time monitoring, decrease the buffer size as far as your CPU allows.

- **Filter Type (IIR / FIR)**  
  **IIR** is suitable for live performance or any scenario where low latency is critical. **FIR** is better for mixing and mastering when phase accuracy and sound quality are top priorities.

- **Oversampling**  
  “Auto” selects the highest factor, giving the strongest anti‑aliasing effect. Manually selecting a lower factor reduces CPU load and latency, but also reduces high‑frequency quality. Higher factors are especially beneficial for non‑linear processing (e.g., saturation).

- **Dithering**  
  Match the bit depth to your output device. 24‑bit or 32‑bit is generally safe. “Off” is not recommended except for debugging or measurement.

- **Automatic Gain Staging Adjustment**  
  The allowable ranges for Input Headroom and Output Makeup automatically change according to the selected processing mode (Conv / Peq / Conv→Peq / Peq→Conv). The current mode is displayed in the UI; adjust accordingly.

- **Adaptive 9th‑order Learning**  
  If you switch to another noise shaper type while learning is in progress, learning will be stopped. The learned results are saved and can be resumed the next time you select “Adaptive 9th‑order”.

---

This concludes the main features of the Audio Settings screen.