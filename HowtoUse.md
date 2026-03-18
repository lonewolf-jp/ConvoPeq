# How to Use ConvoPeq

This is actually the very reason the author built this software.

## Setup Steps

1. Measure your room's IR (Impulse Response) using [REW (Room EQ Wizard)](https://www.roomeqwizard.com/).
2. Install a music subscription app such as Amazon Music.
3. Install [VB-AUDIO Voicemeeter Banana](https://vb-audio.com/Voicemeeter/).

## Usage Scenarios

### 4-1. Listening Through Speakers

Connect your audio chain as follows:

```text
Music App → Voicemeeter Banana → DAC → Amp → Speakers
                    ↕
        ConvoPeq (connected via Voicemeeter Virtual ASIO)
```

Load your IR file, apply convolver processing, and shape the sound to your preference using the parametric equalizer.

### 4-2. Listening Through Headphones

Connect your audio chain as follows:

```text
Music App → Voicemeeter Banana → DAC → Headphone Amp → Headphones
                    ↕
        ConvoPeq (connected via Voicemeeter Virtual ASIO)
```

Disable the convolver and use the parametric equalizer to apply headphone correction.

Headphone correction profiles can be obtained from [AutoEq](https://autoeq.app/).
In AutoEq, select **Custom Parametric EQ**, export the settings as a text file, and load it directly into ConvoPeq.

## 5. Noise Shaper A/B Check (Current vs New)

Use this quick procedure after setup to compare output behavior.

1. Start playback with stable program material (pink noise or a sustained music section).
2. Open Device Settings and set **Dither Bit Depth** to **24 bit**.
3. Switch **Noise Shaper** between **Current** and **New (Fixed 4-tap)** while playing.
4. Confirm there is no crash, no mute, and no large level jump during switching.
5. Set **Dither Bit Depth** to **Off**, then back to **16/24/32 bit** and confirm audible change in quantization character/noise texture.
6. Restart the app and confirm both **Noise Shaper** and **Dither Bit Depth** are restored.

Expected behavior:

- Switching Noise Shaper works in real time without dropouts.
- Off bypasses final quantization/noise-shaping stage.
- 16/24/32 bit selections are available in Device Settings (plus Off).
