# How to Use ConvoPeq

This is actually the very reason the author built this software.

## Setup Steps

1. Measure your room environment using [REW (Room EQ Wizard)](https://www.roomeqwizard.com/).Then output measurement file and correction PEQ settings.
2. Import measuremnt file and correction PEQ settings to [rePahse](https://rephase.org/).Then output FIR filter file.
3. Install a music subscription app such as Amazon Music.
4. Install [VB-AUDIO Voicemeeter Banana](https://vb-audio.com/Voicemeeter/).

## Usage Scenarios

### 4-1. Listening Through Speakers

Connect your audio chain as follows:

```text
Music App → Voicemeeter Banana (WASAPI input / ASIO output) → DAC → Amp → Speakers
                    ↕
        ConvoPeq (connected via Voicemeeter Virtual ASIO)
```

Load your FIR filter file, apply convolver processing, and shape the sound to your preference using the parametric equalizer.

### 4-2. Listening Through Headphones

Connect your audio chain as follows:

```text
Music App → Voicemeeter Banana (WASAPI input / ASIO output) → DAC → Headphone Amp → Headphones
                    ↕
        ConvoPeq (connected via Voicemeeter Virtual ASIO)
```

Disable the convolver and use the parametric equalizer to apply headphone correction.

Headphone correction profiles can be obtained from [AutoEq](https://autoeq.app/).
In AutoEq, select **Custom Parametric EQ**, export the settings as a text file, and load it directly into ConvoPeq.
