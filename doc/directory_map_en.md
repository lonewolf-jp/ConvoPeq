```text
ConvoPeq/
|-- src/                      # Main C++ source (DSP, engine, UI)
|   |-- core/                 # Snapshot/RCU foundation and thread-safe state handoff
|   |-- AudioEngine.*         # Audio processing core
|   |-- EQProcessor.*         # 20-band parametric EQ
|   |-- ConvolverProcessor.*  # IR convolution processing
|   `-- MainApplication.*     # App entry/runtime wiring
|-- manual/                   # User manuals (EN/JP)
|-- resources/                # App resources (icons, assets)
|-- sampledata/               # Sample IR/EQ files
|-- JUCE/                     # JUCE framework source (external dependency)
|-- r8brain-free-src/         # r8brain source (external dependency)
|-- build/                    # Generated build outputs (CMake/Ninja)
|-- README.md
|-- ARCHITECTURE.md
|-- SOUND_PROCESSING.md
|-- BUILD_GUIDE_WINDOWS.md
`-- HOW_TO_USE.md
```