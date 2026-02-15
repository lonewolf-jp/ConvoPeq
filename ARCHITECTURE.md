# ConvoPeq v0.1 - アーキテクチャ設計書

## プロジェクト概要

**名称**: ConvoPeq (Convolution + Parametric EQ)
**バージョン**: v0.1.0
**種類**: Windows 11 x64 専用スタンドアローン・オーディオアプリケーション
**目的**:

- 高精度な20バンド・パラメトリックイコライザー (TPT SVF) とゼロレイテンシー・コンボルバー (FFT Convolution) を統合した、マスタリンググレードのオーディオ処理環境を提供すること。
- システムオーディオやDAW外でのオーディオ信号に対するリアルタイム補正（ルームアコースティック補正、ヘッドホン補正など）。

**技術スタック**:

- **言語**: C++20 (ISO/IEC 14882:2020)
- **フレームワーク**: JUCE 8.0.12 (Modules: core, events, graphics, data_structures, gui_basics, audio_basics, audio_devices, audio_formats, audio_processors, audio_utils, dsp)
- **ビルドシステム**: CMake 3.22以上
- **コンパイラ**: MSVC 19.44.35222.0 (Visual Studio 2022 v17.11以降)
- **ターゲットOS**: Windows 11 x64 (Windows SDK 10.0.26100.0 / Target 10.0.26200)
- **ハードウェア要件**: AVX2命令セット対応CPU (Intel Haswell / AMD Zen以降)
- **外部ライブラリ (任意)**: Intel oneAPI Math Kernel Library (oneMKL) - FFT処理の高速化に使用

## 設計の核心原則

### 1. 厳格なリアルタイム制約とスレッド安全性 (Strict Real-time Safety)

オーディオ処理スレッド（Audio Thread）における処理落ち（グリッチ）を完全に排除するため、以下のルールを徹底しています。

- **Wait-free / Lock-free**: Audio Thread内での `Mutex`、`CriticalSection`、`std::promise` などのブロッキング同期プリミティブの使用を禁止。
- **動的メモリ確保の禁止**: `malloc`, `new`, `std::vector::resize`, `AudioBuffer::setSize` などのヒープ割り当てを禁止。すべてのバッファは `prepareToPlay` またはコンストラクタで最大サイズ（`SAFE_MAX_BLOCK_SIZE`）にて事前確保します。
- **システムコールの禁止**: ファイルI/O、コンソール出力（`printf`, `std::cout`）、スレッド生成などを禁止。

### 2. RCU (Read-Copy-Update) パターンによる状態管理

UIスレッドとAudio Thread間のパラメータ共有には、ロックフリーな **RCU パターン** を採用しています。

- **更新 (Writer/UI Thread)**: 新しい状態オブジェクト（`DSPCore`, `EQState`, `Convolution`）をヒープ上に作成し、セットアップ完了後に `std::atomic<std::shared_ptr<T>>` を介してアトミックにポインタを差し替えます。
- **読み取り (Reader/Audio Thread)**: `std::atomic::load` でポインタを取得し、処理中はローカルの `std::shared_ptr` で参照カウントを保持します。これにより、ロックなしで常に整合性の取れた最新の状態を参照できます。
- **遅延解放 (Garbage Collection)**: 参照されなくなった古いオブジェクトは `trashBin` リスト（Message Thread管理）に送られ、Audio Threadが参照していないことが確定したタイミングで安全に破棄されます。

### 3. 数値安定性とDSP品質 (Numerical Stability)

デジタル信号処理における数値的な破綻を防ぎ、高音質を維持します。

- **TPT SVF (Topology-Preserving Transform State Variable Filter)**: EQフィルタには、従来のBiquadで発生する高域の歪みや、高速なパラメータ変調時の不安定さを解消するTPT SVFアルゴリズムを採用しています。
- **Denormal対策**: `juce::ScopedNoDenormals` の使用に加え、IIRフィルタの状態変数に対して極小値をゼロにフラッシュする処理を手動で実装し、CPU負荷のスパイクを防ぎます。
- **NaN/Inf 保護**: 外部入力や発振による不正な浮動小数点数（NaN/Inf）がDSPチェーン全体に伝播しないよう、検出とクランプ処理を実装しています。

### 4. 堅牢なデバイス管理 (Robust Device Management)

Windows環境特有の多様なオーディオドライバに対応するための防御策を講じています。

- **ASIOブラックリスト**: シングルクライアント専用や動作が不安定なASIOドライバ（BRAVO-HD, ASIO4ALL等）を検出し、自動的に除外することでアプリケーションのクラッシュや排他制御エラーを防ぎます。
- **安全なフォールバック**: デバイスの初期化失敗やサンプルレートの不整合を検知した場合、クラッシュせずにデフォルト設定や安全な値へ自動的にフォールバックします。

## コンポーネント設計

### 全体構成図

```text
MainApplication (JUCEエントリポイント)
  │
  └─ MainWindow (DocumentWindow)
       │
       ├─ AudioDeviceManager (デバイス管理)
       │    └─ AudioSourcePlayer (コールバック仲介)
       │         └─ AudioEngine (AudioSource実装)
       │              │
       │              ├─ DSPCore (Audio Thread用処理コンテナ: RCU管理)
       │              │    ├─ ConvolverProcessor (DSP)
       │              │    ├─ EQProcessor (DSP: TPT SVF)
       │              │    ├─ DCBlocker (DC除去)
       │              │    └─ TPDFDither (ディザリング)
       │              │
       │              ├─ UI State Instances (Message Thread用)
       │              │    ├─ ConvolverProcessor (UI)
       │              │    └─ EQProcessor (UI)
       │              │
       │              └─ Lock-free FIFO (Audio -> UI データ転送)
       │
       ├─ UI Components
       │    ├─ ConvolverControlPanel (IR読み込み、Mix制御)
       │    ├─ EQControlPanel
       │    ├─ SpectrumAnalyzerComponent
       │    └─ DeviceSettings
       │
       └─ Settings Management
            ├─ DeviceSettings (XML永続化)
            └─ AsioBlacklist
```

### MainWindow（中心的な役割）

**ファイル**: `src/MainWindow.cpp`, `src/MainWindow.h`
**継承**: `juce::DocumentWindow`, `juce::ChangeListener`, `juce::Timer`

#### MainWindowの責務

1. **オーディオデバイスとドライバの管理**
   - **所有権**: アプリケーション内で唯一の `AudioDeviceManager` インスタンスを保持します。
   - **ブラックリスト適用**: 初期化前に `AsioBlacklist` を読み込み、不安定なASIOドライバを除外します。
   - **設定の永続化**: `DeviceSettings` を使用して、デバイス設定（サンプルレート、バッファサイズ等）の保存と復元を行います。
   - **設定UI**: ポップアップウィンドウとしてオーディオ設定画面を表示・管理します。

2. **オーディオエンジンの統合と制御**
   - **接続**: `AudioSourcePlayer` を介して、デバイスからのコールバックを `AudioEngine` にルーティングします。
   - **ライフサイクル**: エンジンの初期化 (`initialize`) と終了処理を管理します。
   - **状態同期**: `ChangeListener` としてエンジンからの通知（プリセットロード完了など）を受け取り、各UIコンポーネントに更新を促します。

3. **UIコンポーネントの構築とレイアウト**
   - **主要パネル**: `ConvolverControlPanel`, `EQControlPanel`, `SpectrumAnalyzerComponent` を生成し、ウィンドウ内に配置します。
   - **グローバル制御**: バイパス（EQ/Conv）、処理順序（Order）、プリセット保存/読込ボタンなどのイベントハンドリングを行います。
   - **レイアウト**: ウィンドウリサイズ時に各コンポーネントのサイズと位置を動的に調整します。

4. **アプリケーションロジック**
   - **プリセット管理**: ファイルダイアログを表示し、XML形式（全体設定）やテキスト形式（Equalizer APO互換）の読み書きを調停します。
   - **モニタリング**: タイマー (`juce::Timer`) を使用してCPU使用率を定期的に取得し、表示を更新します。
   - **終了処理**: ウィンドウの閉じるボタンをフックし、安全なシャットダウンシーケンスを開始します。

#### 初期化シーケンス

```text
起動 (MainApplication::initialise)
  ↓
MainWindow コンストラクタ
  ↓
1. ASIO Blacklist 準備
   - `asio_blacklist.txt` の確認・作成（存在しない場合）
   - ファイル読み込みと `DeviceSettings::applyAsioBlacklist` によるドライバ除外
   ↓
2. DeviceSettings::loadSettings()
   - `deviceManager.closeAudioDevice()` (初期化前のクリーンアップ)
   - 保存された設定 (XML) で `AudioDeviceManager` を初期化
   - 失敗時はデフォルトデバイスへフォールバック
   ↓
3. AudioEngine 初期化 & 接続
   - `audioEngine.initialize()`: デフォルトDSPチェーン構築
   - `audioSourcePlayer.setSource(&audioEngine)`: プレイヤーにエンジンを接続
   ↓
4. コールバック・リスナー登録
   - `audioEngine.addChangeListener(this)`: エンジンからの通知受信開始
   - `audioDeviceManager.addAudioCallback(&audioSourcePlayer)`: Audio Thread 開始
   ↓
5. UI コンポーネント生成 (createUIComponents)
   - `ConvolverControlPanel`, `EQControlPanel`, `SpectrumAnalyzerComponent` 生成
   - 各コンポーネントが `AudioEngine` のリスナーとして登録
   - `DeviceSettings` (設定画面) の準備
   ↓
6. タイマー開始 & 表示
   - `startTimer(5000)`: CPU使用率更新用
   - `setVisible(true)`: ウィンドウ表示
   ↓
完了
```

#### デバイス変更シーケンス

```text
ユーザーがデバイスを変更
  ↓
AudioDeviceSelectorComponent が AudioDeviceManager に通知
  ↓
AudioDeviceManager が内部処理:
  1) 現在の IoCallback を停止
  2) 現在のデバイスを閉じる
  3) 新しいデバイスを開く
  4) IoCallback を再開
  ↓
AudioDeviceManager::sendChangeMessage()
  ↓
MainWindow::changeListenerCallback()
  - デバイス情報をログ出力
  - AudioEngine のサンプルレート更新（必要に応じて）
  ↓
完了
```

#### シャットダウンシーケンス

```text
アプリケーション終了要求
  ↓
MainWindow::~MainWindow()
  ↓
1. DeviceSettings::saveSettings()
   - 現在のデバイス設定を保存
   ↓
2. audioEngine.removeChangeListener(this)
   - 変更通知の受信停止
   ↓
3. audioDeviceManager.removeAudioCallback(&audioSourcePlayer)
   - IoCallback 解除
   - Audio Thread 停止
   ↓
4. audioDeviceManager.closeAudioDevice()
   - デバイスを明示的に閉じる（ASIO対策）
   ↓
5. UI コンポーネント破棄 (reset)
   - settingsWindow, deviceSettings, specAnalyzer, eqPanel, convolverPanel
   ↓
6. AudioEngine 破棄
   - MainWindowのメンバ変数として自動的に破棄
   ↓
完了
```

### AudioEngine

**ファイル**: `src/AudioEngine.cpp`, `src/AudioEngine.h`
**継承**: `juce::AudioSource`, `juce::ChangeBroadcaster`, `juce::ChangeListener`
**DSP精度**: `double` (内部処理)

#### AudioEngineの責務

1. **オーディオ処理のコア**
   - `prepareToPlay()`: バッファ事前確保、プロセッサ初期化
   - `getNextAudioBlock()`: リアルタイムオーディオ処理（Audio Thread）
   - `releaseResources()`: リソース解放

2. **処理チェーン管理**
   - ConvolverProcessor + EQProcessor
   - 処理順序切り替え: Conv→EQ / EQ→Conv（ProcessingOrder enum）
   - 個別バイパス制御（atomic bool）

3. **レベルメータリング**
   - 入力レベル・出力レベル計算（dBFS）
   - atomic floatで安全にUI Threadへ通知

4. **FFTデータ提供**
   - Lock-free FIFO（AbstractFifo）でAudio Thread→UI Thread転送
   - UI ThreadでFFT計算（4096ポイント、Hanning窓）
   - 50%オーバーラップ処理

5. **EQ応答曲線計算**
   - UI Thread用ヘルパー関数
   - 現在のEQ設定から周波数応答を計算
   - L/Rチャンネル別対応

#### Audio Thread 処理フロー（getNextAudioBlock）

```cpp
void AudioEngine::getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)
{
    // (1) Denormal対策
    juce::ScopedNoDenormals noDenormals;

    // (2) DSPコアの取得 (Atomic Load - RCU Pattern)
    auto dsp = currentDSP.load(std::memory_order_acquire);

    if (dsp)
    {
        // (3) パラメータのスナップショット取得
        const bool eqBypassed = eqBypassRequested.load(std::memory_order_relaxed);
        // ...

        // (4) 処理委譲
        dsp->process(bufferToFill, ...);
    }
    else
    {
        bufferToFill.clearActiveBufferRegion();
    }
}
```

**重要な制約**:

- `processBuffer`: `prepareToPlay()`で`maxSamplesPerBlock * 2`のサイズで確保済み
- `audioFifoBuffer`: コンストラクタでFIFO_SIZE（16384）確保済み
- `fftWorkBuffer`: スタック配列（NUM_FFT_POINTS * 2 = 8192要素）
- すべてのatomic変数は`std::memory_order_relaxed`で読み書き

#### UI Thread 処理フロー（getLatestFFTData）

```cpp
bool AudioEngine::getLatestFFTData(float* outData, int& outBinCount)
{
    // (1) FIFOから十分なデータがあるか確認
    if (audioFifo.getNumReady() < OVERLAP_SAMPLES) return false;

    // (2) 既存の時間領域バッファを左シフト（古いデータ破棄）
    std::memmove(...);

    // (3) FIFOから新しいデータを読み出し
    audioFifo.prepareToRead(...);
    std::memcpy(...);
    audioFifo.finishedRead(...);

    // (4) Hanning窓適用
    window.multiplyWithWindowingTable(...);

    // (5) FFT実行（周波数領域へ変換）
    fft->performFrequencyOnlyForwardTransform(fftWorkBuffer);

    // (6) dB変換して出力
    for (int i = 0; i < numBins; ++i)
        outData[i] = Decibels::gainToDecibels(fftWorkBuffer[i] * scale);

    return true;
}
```

**FFT設定**:

- ポイント数: 4096（NUM_FFT_POINTS）
- ビン数: 2049（NUM_FFT_BINS = 4096/2 + 1）
- オーバーラップ: 50%（OVERLAP_SAMPLES = 2048）
- 窓関数: Hanning（juce::dsp::WindowingFunction）
- スケール補正: 4.0 / NUM_FFT_POINTS
  - 2.0: 正の周波数成分のみ使用（エネルギー2倍）
  - 2.0: Hanning窓のコヒーレントゲイン補正（1/0.5）

#### データフロー全体図

```text
Input Device
    ↓
[IoCallback - Audio Thread]
    ↓
processBuffer (メンバ変数、事前確保済み)
    ↓ double精度
入力レベル測定 → inputLevelDb (atomic)
    ↓
┌──────────────────────────────────────┐
│  処理順序可変 (ProcessingOrder)       │
│                                      │
│  Option 1: Conv → EQ                 │
│    ├─ ConvolverProcessor::process()  │
│    │    ├─ Dry信号保存               │
│    │    ├─ juce::dsp::Convolution    │
│    │    │    (FFTベース高速畳み込み) │
│    │    └─ Dry/Wet Mix               │
│    │                                  │
│    └─ EQProcessor::process()         │
│         ├─ 係数更新チェック           │
│         ├─ 20バンドBiquad処理        │
│         │   (Direct Form II)         │
│         ├─ トータルゲイン適用        │
│         └─ AGC (Optional)            │
│                                      │
│  Option 2: EQ → Conv                 │
│    (順序逆転)                         │
└──────────────────────────────────────┘
    ↓
出力レベル測定 → outputLevelDb (atomic)
    ↓
FFT用モノラルミックス → Lock-free FIFO
    ↓
Output Device

[UI Thread - 60fps Timer]
    ↓
getLatestFFTData()
  ├─ FIFO読み出し
  ├─ Hanning窓適用
  ├─ FFT実行
  └─ dB変換
    ↓
SpectrumAnalyzerComponent::paint()
  ├─ FFTスペクトラム描画
  ├─ EQ応答曲線オーバーレイ
  └─ レベルメーター表示
```

### ConvolverProcessor

**ファイル**: `src/ConvolverProcessor.cpp`, `src/ConvolverProcessor.h`
**継承**: `juce::ChangeBroadcaster`
**エンジン**: `juce::dsp::Convolution` (JUCE DSP)

#### ConvolverProcessorの機能

1. **インパルス応答読み込み（Message Thread）**
   - 対応形式: WAV, AIFF, FLAC
   - `juce::AudioFormatManager`でファイル読み込み
   - ピーク正規化（最大値を1.0に）
   - ステレオ判定（2ch以上でTrue Stereo）
   - 波形スナップショット作成（512ポイント、UI表示用）

2. **高サンプルレート最適化**
   - 176.4kHz以上: 2段階コンボリューション自動適用
     - **Head**: 先頭2048サンプル（低レイテンシー）
     - **Tail**: 残り全体（パーティション処理）
   - 176.4kHz未満: 単一コンボリューション

3. **レイテンシー補正**
   - `juce::dsp::DelayLine`でDry信号を遅延
   - Wet信号とタイミング合わせ
   - レイテンシー値は`getLatency()`で取得

4. **非同期ロード**
   - Audio Thread実行中にIRをロード可能
   - ダブルバッファリング（nextConvolutionHead/Tail）
   - `newConvolutionReady` atomicフラグでスワップ制御
   - ロード中は古いIRで処理継続（音切れ防止）

5. **Dry/Wet Mix**
   - 0.0 = 完全Dry（IR無効）
   - 1.0 = 完全Wet（IR 100%）
   - `juce::SmoothedValue`で滑らかに変化（50ms）

#### ConvolverProcessorの処理（process）

```cpp
void ConvolverProcessor::process(AudioBuffer<double>& buffer, int numSamples)
{
    // (1) 新しいIRが準備できていればスワップ
    if (newConvolutionReady.load(std::memory_order_acquire)) {
        convolutionHead.swap(nextConvolutionHead);
        convolutionTail.swap(nextConvolutionTail);
        currentLatency.store(newLatency.load());
        newConvolutionReady.store(false, std::memory_order_release);
    }

    // (2) Dry信号をバッファに保存
    dryBuffer.copyFrom(buffer);

    // (3) Wet信号処理
    //     double → float 変換（juce::dsp::Convolutionはfloat）
    convolutionBuffer.copyFrom(buffer);

    // (4) コンボリューション実行
    AudioBlock<float> block(convolutionBuffer);
    convolutionHead->process(ProcessContextReplacing(block));

    if (convolutionTail) {
        tailBuffer.clear();
        AudioBlock<float> tailBlock(tailBuffer);
        convolutionTail->process(ProcessContextReplacing(tailBlock));
        // Head + Tail 合成
    }

    // (5) Dry/Wet Mix
    float mixValue = mixSmoother.getNextValue();
    buffer = dry * (1.0 - mix) + wet * mix;
}
```

### EQProcessor

**ファイル**: `src/EQProcessor.cpp`, `src/EQProcessor.h`
**継承**: `juce::ChangeBroadcaster`
**フィルタ方式**: Biquad (Direct Form II Transposed)

#### EQProcessorの機能

1. **20バンドパラメトリックEQ**
   - バンド0: LowShelf (25Hz)
   - バンド1-18: Peaking (40Hz ~ 18kHz)
   - バンド19: HighShelf (19.5kHz)
   - 各バンド独立調整: 周波数、ゲイン、Q値
   - フィルタタイプ変更可能: LowPass / HighPass

2. **チャンネルモード**
   - Stereo: 両チャンネルに適用
   - Left: 左チャンネルのみ
   - Right: 右チャンネルのみ

3. **トータルゲイン & AGC**
   - トータルゲイン: -24dB ~ +24dB
   - AGC（自動ゲイン補正）:
     - 入出力のRMSレベルを追跡
     - ゲイン変化を滑らかに適用
     - 最小/最大ゲイン制限

4. **係数計算**
   - Audio EQ Cookbook (RBJ) 方式
   - UI Threadでパラメータ変更 → `coeffsDirty` フラグセット
   - Audio Thread内で係数再計算（次回フレーム）
   - `juce::SmoothedValue`でパラメータ補間（50ms）

5. **処理順序最適化**
   - High Q → Low Q の順で処理
   - 数値安定性向上

#### EQProcessorの処理（process）

```cpp
void EQProcessor::process(AudioBuffer<double>& buffer, int numSamples)
{
    // (1) 係数再計算（必要な場合のみ）
    if (coeffsDirty.load(std::memory_order_relaxed))
        recalculateCoeffs();

    // (2) 各バンドのキャッシュ更新（atomicアクセス削減）
    for (int band = 0; band < NUM_BANDS; ++band) {
        bandCache[band].active = bandParams[band].enabled.load();
        bandCache[band].mode = bandChannelModes[band].load();
    }

    // (3) バンド処理（最適化された順序）
    for (int bandIndex : processingOrder) {
        if (!bandCache[bandIndex].active) continue;

        const auto& coeffs = coeffs[bandIndex];
        auto mode = bandCache[bandIndex].mode;

        // Direct Form II Transposed Biquad
        for (int ch = 0; ch < numChannels; ++ch) {
            if (mode == Stereo || (mode == Left && ch == 0) || (mode == Right && ch == 1)) {
                double* samples = buffer.getWritePointer(ch);
                double& z1 = filterState[ch][bandIndex][0];
                double& z2 = filterState[ch][bandIndex][1];

                for (int i = 0; i < numSamples; ++i) {
                    double in = samples[i];
                    double out = coeffs.b0 * in + z1;
                    z1 = coeffs.b1 * in - coeffs.a1 * out + z2;
                    z2 = coeffs.b2 * in - coeffs.a2 * out;
                    samples[i] = out;
                }
            }
        }
    }

    // (4) トータルゲイン適用
    if (!agcEnabled.load()) {
        float gain = smoothTotalGain.getNextValue();
        buffer.applyGain(gain);
    } else {
        // AGC処理
        applyAGC(buffer, numChannels, numSamples, inputRMS);
    }
}
```

#### 係数計算（RBJ方式）

```cpp
BiQuadCoeffs EQProcessor::calcPeaking(float freq, float gainDb, float q, int sr)
{
    double A = std::pow(10.0, gainDb / 40.0); // sqrt(linear gain)
    double w0 = 2.0 * M_PI * freq / sr;
    double cosw0 = std::cos(w0);
    double sinw0 = std::sin(w0);
    double alpha = sinw0 / (2.0 * q);

    BiQuadCoeffs c;
    c.b0 = 1.0 + alpha * A;
    c.b1 = -2.0 * cosw0;
    c.b2 = 1.0 - alpha * A;
    c.a0 = 1.0 + alpha / A;
    c.a1 = -2.0 * cosw0;
    c.a2 = 1.0 - alpha / A;

    // a0で正規化
    c.b0 /= c.a0; c.b1 /= c.a0; c.b2 /= c.a0;
    c.a1 /= c.a0; c.a2 /= c.a0;
    c.a0 = 1.0;

    return c;
}
```

## オーディオバックエンド詳細

### Windows専用バックエンド

本アプリケーションはWindows 11 x64専用です。以下のオーディオバックエンドに対応しています。

### AsioBlacklist（不安定ドライバー除外機構）

**ファイル**: `src/AsioBlacklist.h`
**設定ファイル**: `asio_blacklist.txt`（実行ファイルと同じディレクトリ）

#### 目的

一部のASIOドライバーは以下の問題を抱えています：

- **シングルクライアント制限**: 同時に複数のアプリケーションから使用不可
- **動作不安定**: クラッシュ、音切れ、デバイス切り替え失敗
- **JUCE互換性問題**: JUCE AudioDeviceManagerとの相性問題

#### デフォルトブラックリスト

```text
BRAVO-HD          # SAVITECH USB オーディオチップ専用、シングルクライアント
ASIO4ALL          # シングルクライアント、動作不安定
FlexASIO          # マルチクライアント対応だが動作不安定
```

#### 動作

1. 起動時に`asio_blacklist.txt`を読み込み
2. `AudioDeviceManager::getAvailableDeviceTypes()`の結果からフィルタリング
3. UIの選択肢から除外（ユーザーは選択不可）
4. 既に保存されていた設定がブラックリスト対象の場合、自動的に別のデバイスへフォールバック

#### カスタマイズ

`asio_blacklist.txt`を編集してドライバーを追加/削除可能：

```text
# ASIO Driver Blacklist
# Add partial driver names to exclude them from the list.

# 例: Voicemeeterは安定しているので除外しない
# BRAVO-HD
# ASIO4ALL
FlexASIO
```

### DeviceSettings（デバイス設定永続化）

**ファイル**: `src/DeviceSettings.cpp`, `src/DeviceSettings.h`
**設定ファイル**: `%APPDATA%\ConvoPeq\device_settings.xml`

#### 保存される設定

- デバイスタイプ（WASAPI/ASIO/DirectSound）
- 入力デバイス名
- 出力デバイス名
- サンプルレート
- バッファサイズ
- 有効な入力チャンネル
- 有効な出力チャンネル

#### 保存・読み込みタイミング

- **保存**: `MainWindow::~MainWindow()`（アプリケーション終了時）
- **読み込み**: `MainWindow::MainWindow()`（起動時）

#### フォールバック機構

1. 設定ファイルが存在しない → デフォルトデバイスで初期化
2. 設定ファイルが壊れている → デフォルトデバイスで初期化
3. 保存されたデバイスが利用不可 → 警告ダイアログ表示、他のデバイスにフォールバック

#### XML形式例

```xml
<?xml version="1.0" encoding="UTF-8"?>
<DEVICESETUP deviceType="Windows Audio"
             audioOutputDeviceName="Speakers (Realtek High Definition Audio)"
             audioInputDeviceName=""
             audioDeviceRate="48000"
             audioDeviceBufferSize="512"
             audioDeviceInChans="00"
             audioDeviceOutChans="11"/>
```

## Windows オーディオバックエンド比較

| 項目 | WASAPI Exclusive | WASAPI Shared | DirectSound | ASIO |
| :--- | :--- | :--- | :--- | :--- |
| **レイテンシー** | 非常に低い (5-20ms) | 低～中 (10-50ms) | 高い (50-200ms) | 最低 (2-10ms) |
| **ビットパーフェクト** | ✅ Yes | ❌ No (Windowsミキサー経由) | ❌ No | ✅ Yes |
| **サンプルレート変換** | ❌ なし | ✅ あり（強制） | ✅ あり | ❌ なし |
| **マルチクライアント** | ❌ 排他占有 | ✅ 可能 | ✅ 可能 | ❌ 通常は不可* |
| **対応デバイス** | すべて | すべて | すべて | 専用ドライバー必要 |
| **互換性** | 高い (Windows Vista以降) | 最高 | 最高 (XP以降) | 中～低（ドライバー依存） |
| **CPU使用率** | 低い | 中 | 中～高 | 最低 |
| **推奨用途** | 音楽制作、クリティカルリスニング | 一般用途 | レガシーサポート | プロオーディオ |

*VB-AUDIO Voicemeeterなど一部のASIOドライバーはマルチクライアント対応

### WASAPI (Windows Audio Session API)

#### Exclusive Mode（推奨）

**利点**:

- 最低レイテンシー（5-20ms、設定次第）
- ビットパーフェクト出力（サンプルレート変換なし）
- Windowsミキサーをバイパス
- 最高の音質

**欠点**:

- デバイスを排他占有（他のアプリが使用不可）
- デバイス切り替え時に一時的に音が途切れる

**推奨設定**:

- サンプルレート: 48000 Hz（デバイスのネイティブレート）
- バッファサイズ: 512 samples（約10.7ms @ 48kHz）

#### Shared Mode

**利点**:

- 他のアプリケーションと同時使用可能
- デバイス切り替えが安定
- 常に利用可能

**欠点**:

- Windowsミキサー経由（サンプルレート変換）
- レイテンシーが若干高い（10-50ms）
- ビットパーフェクトではない

**推奨設定**:

- サンプルレート: 48000 Hz（Windowsのデフォルトと合わせる）
- バッファサイズ: 1024 samples（安定性重視）

### DirectSound

**概要**: Windows XP時代のレガシーAPI

**利点**:

- 最高の互換性
- すべてのWindowsデバイスで動作
- 複数アプリケーション同時使用可能

**欠点**:

- 高レイテンシー（50-200ms）
- 古いAPI（新機能なし）
- 音質が劣る（サンプルレート変換、ビット深度変換）

**推奨用途**: 互換性が最優先の場合のみ

### ASIO (Audio Stream Input/Output)

**概要**: Steinberg社が開発したプロオーディオ向け低レイテンシーAPI

**利点**:

- 最低レイテンシー（2-10ms）
- ビットパーフェクト
- マルチチャンネル対応（最大64ch）
- CPU効率が最高

**欠点**:

- 専用ドライバー必要（すべてのデバイスが対応しているわけではない）
- 多くのドライバーはシングルクライアント（同時に1つのアプリのみ）
- ドライバーの品質にばらつき
- 一部のドライバーはJUCEとの相性が悪い

**推奨ASIOドライバー**:

- **VB-AUDIO Voicemeeter ASIO**: マルチクライアント対応、安定性高い
- **専用オーディオインターフェースのASIO**: RME、Focusrite、UADなど

**非推奨（ブラックリスト対象）**:

- **BRAVO-HD**: SAVITECH USB Audio専用、シングルクライアント、不安定
- **ASIO4ALL**: 汎用ASIOドライバー、シングルクライアント、クラッシュしやすい
- **FlexASIO**: マルチクライアント対応だがJUCEとの相性が悪い

## エラーハンドリング戦略

### デバイスオープン失敗

```cpp
juce::String error = deviceManager.initialiseWithDefaultDevices(2, 2);
if (error.isNotEmpty())
{
    // ログ出力
    juce::Logger::writeToLog("[Init] WARNING: " + error);

    // ユーザーに通知（警告ダイアログ）
    juce::AlertWindow::showMessageBoxAsync(
        juce::AlertWindow::WarningIcon,
        "Audio Device Warning",
        "Could not open default audio device:\n\n" + error +
        "\n\nYou can select a different device from Audio Settings.",
        "OK"
    );

    // アプリケーションは継続（デバイスなし状態）
    // ユーザーが後でデバイスを選択可能
}
```

### サンプルレート不一致

```cpp
// AudioEngine::ioCallback() 内で検出
if (sampleRate != currentSampleRate.load())
{
    // atomic 更新（Audio Thread 安全）
    currentSampleRate.store(static_cast<int>(sampleRate));

    // EQProcessor に通知（次のフレームから新しいサンプルレートを使用）
    eqProcessor.prepareToPlay(static_cast<int>(sampleRate), numSamplesPerBuffer);
}
```

### バッファサイズ変更

```cpp
// ensureProcessBuffer() で安全にサイズ変更
void AudioEngine::ensureProcessBuffer(int numChannels, int numSamples)
{
    if (processBuffer.getNumChannels() != numChannels ||
        processBuffer.getNumSamples() < numSamples)
    {
        // setSize は必要に応じて再確保（通常は何もしない）
        processBuffer.setSize(numChannels, numSamples, false, false, true);
    }
}
```

## パフォーマンス最適化

### コンパイラ最適化フラグ（CMakeLists.txt）

#### MSVC Release設定

```cmake
# /O2: 速度優先の最適化
# /Ob3: 積極的なインライン展開（MSVC 19.29以降）
# /GL: プログラム全体の最適化（リンク時コード生成）
# /arch:AVX2: AVX2命令セット使用
# /fp:fast: 高速浮動小数点演算（IEEE 754厳密性を緩和）
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /GL /arch:AVX2 /fp:fast")

# リンカー最適化
# /LTCG: リンク時コード生成
# /OPT:REF: 未使用関数削除
# /OPT:ICF: 同一関数の統合
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /OPT:REF /OPT:ICF")
```

#### 並列ビルド

```cmake
# /MP: 並列コンパイル（全CPUコア使用）
add_compile_options(/MP)
```

### Audio Thread での最適化

#### 1. 動的メモリ確保の完全排除

**原則**: Audio Thread内では`new`/`malloc`/`std::vector::resize`等を**絶対に**使用しない

**実装例**:

```cpp
// ❌ NG: Audio Threadで動的確保
void getNextAudioBlock(...)
{
    std::vector<float> tempBuffer(numSamples); // NG!
    AudioBuffer<float> temp(2, numSamples);    // NG!
    fft->performFrequencyOnlyForwardTransform(data); // 内部確保あり、NG!
}

// ✅ OK: prepareToPlayで事前確保
void prepareToPlay(int samplesPerBlock, double sampleRate)
{
    maxSamplesPerBlock = samplesPerBlock * 2; // 余裕を持たせる
    processBuffer.setSize(2, maxSamplesPerBlock, false, true, false);

    // std::vectorも事前確保
    audioFifoBuffer.resize(FIFO_SIZE, 0.0f);
    fftTimeDomainBuffer.resize(NUM_FFT_POINTS, 0.0f);
}

void getNextAudioBlock(...)
{
    // サイズチェックのみ、確保はしない
    if (numSamples > maxSamplesPerBlock) {
        bufferToFill.clearActiveBufferRegion();
        return;
    }

    // 既存バッファを使用（サイズ変更なし）
    // processBuffer, audioFifoBufferなど
}
```

**重要な注意点**:

- `AudioBuffer::setSize()`: 内部でメモリ確保する可能性あり → Audio Thread内では使用禁止
- `AudioBuffer::copyFrom()`: サイズ不一致時に確保する可能性あり → 事前サイズチェック必須
- `std::vector::push_back/resize`: 常に確保する可能性あり → 生ポインタまたはAudioBufferをwrap

#### 2. キャッシュ効率の最適化

**データ構造を連続メモリに配置**:

```cpp
// ❌ NG: ポインタの配列（キャッシュミス多発）
struct EQBand {
    float* frequency;
    float* gain;
    float* q;
};
EQBand bands[20];

// ✅ OK: 構造体の配列（キャッシュヒット率向上）
struct EQBandParams {
    std::atomic<float> frequency;
    std::atomic<float> gain;
    std::atomic<float> q;
};
EQBandParams bandParams[20]; // 連続メモリ
```

**SIMD命令の活用**:

```cmake
# AVX2対応（8個のfloat/4個のdoubleを同時処理）
add_compile_options(/arch:AVX2)
```

JUCE DSPモジュールは自動的にSIMD命令を使用します（juce::FloatVectorOperationsなど）。

#### 3. 分岐予測の最適化

**ホットパスでの条件分岐を最小化**:

```cpp
// ❌ NG: 毎サンプルで分岐
for (int i = 0; i < numSamples; ++i) {
    if (std::abs(gain - 0.0f) < 0.01f) continue; // 毎回チェック
    samples[i] *= gain;
}

// ✅ OK: 事前チェック
if (std::abs(gain - 0.0f) >= 0.01f) { // 1回のみチェック
    for (int i = 0; i < numSamples; ++i)
        samples[i] *= gain;
}
```

**バンド処理の早期スキップ**:

```cpp
void EQProcessor::process(...)
{
    // バンドキャッシュ更新（atomicアクセス削減）
    for (int band = 0; band < NUM_BANDS; ++band) {
        bandCache[band].active = bandParams[band].enabled.load();
        // ...
    }

    // 処理ループ
    for (int bandIndex : processingOrder) {
        if (!bandCache[bandIndex].active) continue; // 早期スキップ
        // フィルタ処理...
    }
}
```

#### 4. Denormal対策

**ScopedNoDenormals使用**:

```cpp
void getNextAudioBlock(...)
{
    juce::ScopedNoDenormals noDenormals; // 非正規化数を0にフラッシュ
    // ...
}
```

非正規化数（denormal）は通常の浮動小数点数の1000倍以上遅い処理になる場合があります。

### UI Thread での最適化

#### 1. フレームレート制御

```cpp
// 60fps固定
class SpectrumAnalyzerComponent : public Component, private Timer
{
    SpectrumAnalyzerComponent() {
        startTimerHz(60); // 16.6ms周期
    }

    void timerCallback() override {
        repaint(); // 描画要求
    }
};
```

#### 2. FFTデータのスムージング

```cpp
// 視覚的な安定性のため、FFT結果をスムージング
for (int i = 0; i < numBins; ++i) {
    smoothedFFT[i] = smoothedFFT[i] * 0.8f + rawFFT[i] * 0.2f;
}
```

#### 3. ピーク保持

```cpp
// ピーク値を2秒間保持
if (currentValue > peakValue[i]) {
    peakValue[i] = currentValue;
    peakHoldTime[i] = getCurrentTime();
} else if (getCurrentTime() - peakHoldTime[i] > 2.0) {
    peakValue[i] *= 0.95f; // 緩やかに減衰
}
```

### メモリ使用量最適化

#### 固定サイズバッファの事前確保

```cpp
class AudioEngine
{
    // コンストラクタで固定サイズ確保
    AudioEngine() {
        audioFifoBuffer.resize(FIFO_SIZE, 0.0f);        // 16384要素
        fftTimeDomainBuffer.resize(NUM_FFT_POINTS, 0.0f); // 4096要素
    }

    // prepareToPlayで可変サイズ確保
    void prepareToPlay(int samplesPerBlock, double sampleRate) {
        maxSamplesPerBlock = samplesPerBlock * 2;
        processBuffer.setSize(2, maxSamplesPerBlock, false, true, false);
    }
};
```

#### メモリリーク防止

- `std::unique_ptr`の活用（自動管理）
- `JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR`マクロ使用
- デストラクタで明示的にリソース解放

### CPU使用率目標値

| 条件 | 目標CPU使用率 | 実測値 |
| :--- | :--- | :--- |
| **アイドル時** | < 1% | 0.5% |
| **EQ処理のみ（20バンド）** | < 5% | 3-4% |
| **Convolver処理のみ** | < 8% | 5-7% |
| **EQ + Convolver（48kHz, 512buf）** | < 12% | 10-12% |
| **EQ + Convolver（96kHz, 512buf）** | < 20% | 15-18% |

測定環境: Intel Core i7-10700K @ 3.8GHz, Windows 11

## デバッグとログ出力

### ログレベル

```cpp
// 初期化プロセス
juce::Logger::writeToLog("=== MainWindow Initialization Start ===");
juce::Logger::writeToLog("[Init] Initializing audio device...");

// デバイス情報
juce::Logger::writeToLog("[DeviceInfo] Current Audio Device:");
juce::Logger::writeToLog("  Name: " + device->getName());

// エラー
juce::Logger::writeToLog("[Init] WARNING: " + error);

// デバイス変更
juce::Logger::writeToLog("[DeviceChange] Audio device configuration changed");

// シャットダウン
juce::Logger::writeToLog("=== MainWindow Shutdown Start ===");
```

### ログ出力先

- Windows: デバッグ出力ウィンドウ

## テストシナリオ

### 基本動作テスト

1. デフォルトデバイスで起動
2. EQ調整動作確認
3. スペクトラムアナライザー表示確認
4. 正常終了

### デバイス切り替えテスト

1. WASAPI Shared で起動
2. WASAPI Exclusive に切り替え
3. 音声が途切れないことを確認
4. ASIO に切り替え
5. DirectSound に切り替え

### エラーハンドリングテスト

1. デバイス未接続で起動
2. 警告ダイアログが表示されることを確認
3. Audio Settings からデバイスを選択
4. 正常に音声が出力されることを確認

### ストレステスト

1. 高速にデバイスを切り替え
2. サンプルレートを変更
3. バッファサイズを変更
4. メモリリークがないことを確認

## 今後の拡張可能性

### プリセット管理

- EQ設定の保存・読み込み
- XMLまたはJSON形式

### MIDI制御

- MIDIコントローラーでEQパラメータを操作
- juce_audio_utils が既にMIDIデバイス管理機能を提供

### VST3プラグイン化

- juce_audio_plugin_client を使用
- DAWで使用可能なプラグインとして配布

### マルチバンドコンプレッサー追加

- 各バンドに独立したコンプレッサー
- スレッショルド・レシオ・アタック・リリース調整可能
