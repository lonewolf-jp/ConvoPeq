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
