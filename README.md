# ConvoPeq

ConvoPeq is a high-performance, standalone audio application that combines convolution reverb and a parametric equalizer. It is built using the JUCE framework and designed for real-time audio processing.This project is programmed by vibe-coding.

## Features

* **Convolution Reverb:** Utilizes FFT-based convolution for realistic reverb effects. Supports loading impulse responses (IR) in WAV, AIFF, and FLAC formats.
* **Parametric Equalizer:** A 20-band parametric EQ for detailed audio shaping.
* **Real-time Audio Processing:** Designed with a focus on low-latency, real-time performance.
* **User Interface:** A modern and intuitive user interface built with JUCE.
* **Multi-format IR Support:** Supports WAV, AIFF, and FLAC impulse response files.
* **ASIO Blacklisting**: Allows excluding problematic ASIO drivers.
* **Preset Management:** Save and load EQ and Convolver settings.
* **Spectrum Analyzer**: Visual feedback of audio output.
* **Adjustable Oversampling:** Allows users to select different oversampling factors for quality/performance trade-offs.
* **Dither**: Includes psychoacoustic dither for high-quality output, especially at lower bit depths.

## Dependencies

* [JUCE Framework](https://github.com/juce-framework/JUCE) (V8.0.12)
* [FFTConvolver](https://github.com/ouroboros-audio/FFTConvolver)
* [r8brain-free-src](https://github.com/avaneev/r8brain-free-src)

## Building ConvoPeq

These instructions will guide you through building ConvoPeq from source on Windows using Visual Studio Code and CMake.

### Prerequisites

* [Visual Studio Code](https://code.visualstudio.com/)
* [CMake](https://cmake.org/) (3.22 or higher)
* [JUCE Framework](https://github.com/juce-framework/JUCE) (V8.0.12)
* [r8brain-free-src](https://github.com/avaneev/r8brain-free-src)

### Build Steps

1. Clone the repository.
2. Ensure that the JUCE framework and r8brain-free-src are placed either as submodules or as sibling directories to the project.
3. Open the project in Visual Studio Code.
4. Configure the project using CMake: `cmake .. -G "Visual Studio 17 2022" -A x64`
5. Build the project using CMake: `cmake --build . --config Release` (or Debug)

### Build Artifacts

The executable will be located in `build\ConvoPeq_artefacts\[Configuration]\ConvoPeq.exe`, where `[Configuration]` is either `Debug` or `Release`.

## Architecture

The application is structured as follows:

* **AudioEngine**: Manages the audio processing graph, including the convolver and EQ. Uses a lock-free RCU pattern for thread-safe parameter updates.
* **ConvolverProcessor**: Implements the convolution reverb.
* **EQProcessor**: Implements the parametric equalizer.
* **SpectrumAnalyzerComponent**: Provides a visual representation of the audio spectrum and EQ curve.
* **DeviceSettings**: Handles audio device selection and settings.

## License

This project is licensed under the AGPLv3 License.
