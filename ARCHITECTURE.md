# ConvoPeq v0.3.5 - アーキテクチャ設計書

## プロジェクト概要

**名称**: ConvoPeq (Convolution + Parametric EQ)
**バージョン**: v0.3.5
**種類**: Windows 11 x64 専用スタンドアローン・オーディオアプリケーション
**目的**:

- 高精度な20バンド・パラメトリックイコライザー (TPT SVF) とゼロレイテンシー・コンボルバー (MKL NUC) を統合した、マスタリンググレードのオーディオ処理環境を提供すること。
- システムオーディオやDAW外でのオーディオ信号に対するリアルタイム補正（ルームアコースティック補正、ヘッドホン補正など）。

**技術スタック**:

- **言語**: C++20
- **フレームワーク**: JUCE 8.0.12
- **ビルドシステム**: CMake 3.22+
- **コンパイラ**: MSVC 19.44+ (Visual Studio 2022)
- **ターゲットOS**: Windows 11 x64 (AVX2必須)
- **外部ライブラリ**: Intel oneAPI Math Kernel Library (oneMKL) - FFT, Vector Math, RNG

## 設計の核心原則

### 1. 厳格なリアルタイム制約とスレッド安全性 (Strict Real-time Safety)

オーディオ処理スレッド（Audio Thread）における処理落ち（グリッチ）を完全に排除するため、以下のルールを徹底しています。

- **Wait-free / Lock-free**: Audio Thread内での `Mutex`、`CriticalSection`、`std::promise` などのブロッキング同期プリミティブの使用を禁止。
- **動的メモリ確保の禁止**: `malloc`, `new`, `std::vector::resize` などのヒープ割り当てを禁止。すべてのバッファは `DSPCore` の構築時（Message Thread / Worker Thread）に `ScopedAlignedPtr` (MKLアロケータ) を用いて事前確保します。
- **システムコールの禁止**: ファイルI/O、コンソール出力、スレッド生成などを禁止。
- **MKL設定**: `mkl_set_num_threads(1)` および `mkl_set_dynamic(0)` により、MKL内部のスレッド生成を抑制し、予測不可能なレイテンシを防ぎます。

### 2. RCU (Read-Copy-Update) パターンによる状態管理

UIスレッド/WorkerスレッドとAudio Thread間のパラメータ共有には、ロックフリーな **RCU パターン** を採用しています。

- **更新 (Writer)**: 新しい状態オブジェクト（`DSPCore`, `EQState`, `StereoConvolver`）をヒープ上に作成し、セットアップ完了後に `std::atomic<T*>` を介してアトミックにポインタを差し替えます（`std::memory_order_release`）。
- **読み取り (Reader/Audio Thread)**: `std::atomic::load` で生ポインタを取得します。Audio Threadの処理サイクル中はポインタが有効であることを、Message Thread側の遅延解放メカニズム（Trash Bin）によって保証します。
- **遅延解放 (Garbage Collection)**: 参照されなくなった古いオブジェクトは `trashBin` リスト（Message Thread管理）に送られ、タイムスタンプに基づいて一定時間（例: 2000ms）経過後に安全に破棄されます。

### 3. 数値安定性とDSP品質 (Numerical Stability)

- **TPT SVF (Topology-Preserving Transform State Variable Filter)**: EQフィルタには、従来のBiquadで発生する高域の歪みや、高速なパラメータ変調時の不安定さを解消するTPT SVFアルゴリズムを採用しています。
- **Denormal対策**: `juce::ScopedNoDenormals` の使用に加え、IIRフィルタの状態変数に対して極小値をゼロにフラッシュする処理を手動で実装し、CPU負荷のスパイクを防ぎます。また、MKL VMLモードを `VML_FTZDAZ_ON` に設定しています。
- **NaN/Inf 保護**: 外部入力や発振による不正な浮動小数点数（NaN/Inf）がDSPチェーン全体に伝播しないよう、検出とクランプ処理、および `UltraHighRateDCBlocker` (1次IIR) による保護を実装しています。

### 4. 堅牢なデバイス管理 (Robust Device Management)

- **ASIOブラックリスト**: シングルクライアント専用や動作が不安定なASIOドライバ（BRAVO-HD, ASIO4ALL等）を検出し、自動的に除外します。
- **Windows最適化**: `timeBeginPeriod(1)` によるタイマー精度向上、`SetPriorityClass(HIGH_PRIORITY_CLASS)`、およびWindows 11の効率モード（EcoQoS）無効化を行い、音切れを防ぎます。

## コンポーネント設計

### 全体構成図

```text
MainApplication
  │
  └─ MainWindow
       │
       ├─ AudioDeviceManager
       │    └─ AudioProcessorPlayer
       │         └─ AudioEngineProcessor (AudioProcessor)
       │              └─ AudioEngine (AudioSource)
       │         │
       │         ├─ DSPCore (Audio Thread用処理コンテナ: RCU管理)
       │         │    ├─ CustomInputOversampler (Polyphase IIR/FIR)
       │         │    ├─ UltraHighRateDCBlocker (DC除去)
       │         │    ├─ ConvolverProcessor (MKL NUC)
       │         │    ├─ EQProcessor (TPT SVF)
       │         │    ├─ SoftClipper (AVX2)
       │         │    └─ PsychoacousticDither (MKL VSL)
       │         │
       │         ├─ Rebuild Thread (Worker)
       │         │    └─ DSPCore構築・IRリサンプリング
       │         │
       │         └─ UI State Instances (Message Thread用)
       │
       └─ UI Components (ConvolverControlPanel, EQControlPanel, SpectrumAnalyzer)
```

### 信号フロー詳細

```text
Input Device
    ↓ (float) [IoCallback]
[IoCallback - Audio Thread]
    ↓ (double変換 + Headroom(-0.1dB) + Sanitize + Input DCBlocker: 3Hz)
AlignedBuffer (DSPCore内)
    ↓
入力レベル測定 → inputLevelLinear (atomic)
    ↓
オーバーサンプリング (Up: 1x, 2x, 4x, 8x)
    ↓
OS後段 DCBlocker (1Hz)
    ↓
Analyzer Input Tap (Pre-DSP) → Lock-free FIFO
    ↓
┌──────────────────────────────────────┐
│  処理順序可変 (ProcessingOrder)       │
│                                      │
│  Option 1: Conv → EQ                 │
│    ├─ ConvolverProcessor::process()  │
│    │    └─ MKLNonUniformConvolver    │ (Dry/Wet Mix, Latency Comp)
│    │                                  │
│    └─ EQProcessor::process()         │
│         ├─ 20バンド TPT SVF処理      │
│         │   (AVX2 Stereo Optimized)  │
│         └─ Total Gain / AGC          │
│                                      │
│  Option 2: EQ → Conv                 │
│    (順序逆転)                         │
└──────────────────────────────────────┘
    ↓
Soft Clipper (AVX2 Optimized: Tanh + Poly)
    ↓
Analyzer Output Tap (Post-DSP) → Lock-free FIFO
    ↓
オーバーサンプリング (Down)
    ↓
出力レベル測定 → outputLevelLinear (atomic)
    ↓
Output DCBlocker (3Hz)
    ↓
Headroom(-0.1dB) + Psychoacoustic Dither (Noise Shaping)
    ↓
Float変換 + クランプ
    ↓
Output Device
```

### AudioEngine & DSPCore

**ファイル**: `src/AudioEngine.cpp`, `src/AudioEngine.h`

#### DSPCore

Audio Threadで実行される処理のコンテナ。RCUパターンにより、設定変更時は新しい `DSPCore` インスタンスがバックグラウンドで構築され、アトミックに差し替えられます。

- **メモリ管理**: `ScopedAlignedPtr` を使用し、MKL/AVX2に最適な64バイトアライメントでメモリを確保します。
- **バッファサイズ**: `SAFE_MAX_BLOCK_SIZE` (65536) * 8 (最大オーバーサンプリング倍率) を確保し、実行時の再確保を排除します。

#### Rebuild Thread

サンプルレート変更、バッファサイズ変更、オーバーサンプリング設定変更、IRロードなどの重い処理は、専用の `rebuildThreadLoop` で実行されます。

1. Message Thread から `requestRebuild` を発行。
2. Worker Thread で新しい `DSPCore` を構築、メモリ確保、IRリサンプリング、FFT計画作成を実行。
3. 完了後、Message Thread 経由で `commitNewDSP` を呼び出し、ポインタを更新。

### ConvolverProcessor

**ファイル**: `src/ConvolverProcessor.cpp`, `src/ConvolverProcessor.h`
**エンジン**: `MKLNonUniformConvolver` (Custom MKL Implementation)

#### ConvolverProcessorの機能

1. **インパルス応答読み込み**
   - `LoaderThread` による非同期読み込み。
   - **前処理**:
     - Float -> Double 変換 (High Quality)
     - Auto Makeup Gain (エネルギー正規化)
     - 無音カット (末尾トリミング)
     - リサンプリング (r8brain-free-src)
     - DC除去 (1Hz HighPass)
     - Asymmetric Tukey Window (ピーク基準の窓関数)
     - Minimum Phase 変換 (MKL FFT + Cepstrum法, オプション)

2. **MKL Non-Uniform Partitioned Convolution (NUC)**
   - Intel MKL DFTI を使用した独自の畳み込みエンジン。
   - **構成**: 現在は安定性重視のため、単一レイヤー（Uniform Partitioned）構成で動作。
   - **最適化**: AVX2 FMAを使用した複素数積和演算。

3. **レイテンシー補正**
   - リングバッファによるDry信号遅延。
   - ドップラー効果を防ぐためのクロスフェード付き遅延時間変更。

### EQProcessor

**ファイル**: `src/EQProcessor.cpp`, `src/EQProcessor.h`
**フィルタ方式**: TPT SVF (Topology-Preserving Transform State Variable Filter)

#### EQProcessorの機能

1. **20バンドパラメトリックEQ**
   - LowShelf, Peaking, HighShelf, LowPass, HighPass。
   - 各バンド独立調整: 周波数、ゲイン、Q値。

2. **AVX2最適化**
   - `processBandStereo`: L/Rチャンネルの係数と状態変数をSIMDレジスタにパックし、同時処理を行います。

3. **係数計算**
   - **Audio Thread**: TPT SVF係数 (`EQCoeffsSVF`) を使用。時間変化に強く、オートメーション時のノイズが少ない。
   - **UI Thread**: 表示用にBiquad係数 (`EQCoeffsBiquad`) を使用。

4. **AGC (Auto Gain Control)**
   - 入出力のRMSレベルを追跡し、ラウドネスを維持するようにトータルゲインを自動調整します。

### CustomInputOversampler

**ファイル**: `src/CustomInputOversampler.cpp`

- **倍率**: 1x, 2x, 4x, 8x (サンプルレートに応じて自動制限)。
- **方式**: Polyphase IIR (低遅延) または Linear Phase (位相直線) を選択可能。
- **実装**: AVX2を使用した高速な畳み込み処理。

### PsychoacousticDither

**ファイル**: `src/PsychoacousticDither.h`

- **RNG**: Intel MKL VSL (`VSL_BRNG_SFMT19937`) + `SplitMix64` シード生成。
- **Noise Shaping**: 5次 Error Feedback Topology (Lipshitz / Wannamaker系係数)。
- **処理**: 可聴域の量子化ノイズを低減し、超音波域へシフトさせます。

### SpectrumAnalyzerComponent

**ファイル**: `src/SpectrumAnalyzerComponent.cpp`

- **FFT**: Intel MKL DFTI (4096 points, Single Precision).
- **表示**: 60fps タイマー駆動。Lock-free FIFO経由でデータを取得。
- **機能**:
  - 入力/出力ソース切り替え。
  - ピークホールド、スムージング。
  - EQ応答曲線のオーバーレイ表示。

## データ構造とメモリ

### ScopedAlignedPtr

`src/AlignedAllocation.h` で定義されるスマートポインタ。

- `mkl_malloc` / `mkl_free` をラップ。
- 64バイトアライメントを保証（AVX-512/AVX2対応）。
- RAIIによる自動解放。

### DeviceSettings

**ファイル**: `src/DeviceSettings.cpp`
**保存先**: `%APPDATA%\ConvoPeq\device_settings.xml`

保存される設定:

- デバイスタイプ/ID
- サンプルレート/バッファサイズ
- ディザビット深度
- オーバーサンプリング設定
