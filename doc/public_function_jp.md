以下は、ConvoPeq の主要パブリック API に関する Doxygen 形式のドキュメントです。各関数には引数、戻り値、スレッド安全性に関する注釈が含まれています。

---

# ConvoPeq Public API Documentation

## 1. AudioEngine

`AudioEngine` はアプリケーション全体のオーディオ処理を統括する中心クラスです。

### 1.1 オーディオ I/O ライフサイクル

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void prepareToPlay(int samplesPerBlockExpected, double sampleRate)` | オーディオ再生の準備を行います。内部 DSP グラフの再構築を要求します。 | **Message Thread** からのみ呼び出し可能。 |
| `void releaseResources()` | オーディオデバイス停止時にリソースを解放します。 | **Message Thread** からのみ呼び出し可能。 |
| `void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill)` | オーディオコールバック。DSP 処理を実行します。 | **Audio Thread** からのみ呼び出し可能。リアルタイム制約あり。 |

### 1.2 パラメータ制御

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void setInputHeadroomDb(float db)` | 入力ヘッドルームゲイン (dB) を設定します。 | **Message Thread**。内部で atomic 変数を更新します。 |
| `float getInputHeadroomDb() const` | 現在の入力ヘッドルームゲイン (dB) を取得します。 | スレッドセーフ (atomic load)。 |
| `void setOutputMakeupDb(float db)` | 出力メイクアップゲイン (dB) を設定します。 | **Message Thread**。 |
| `float getOutputMakeupDb() const` | 現在の出力メイクアップゲイン (dB) を取得します。 | スレッドセーフ (atomic load)。 |
| `void setProcessingOrder(ProcessingOrder order)` | 処理順序 (`ConvolverThenEQ` または `EQThenConvolver`) を設定します。 | **Message Thread**。 |
| `ProcessingOrder getProcessingOrder() const` | 現在の処理順序を取得します。 | スレッドセーフ (atomic load)。 |
| `void setEqBypassRequested(bool shouldBypass)` | EQ バイパス状態を要求します。 | **Message Thread**。 |
| `bool isEqBypassRequested() const` | EQ バイパス要求状態を取得します。 | スレッドセーフ (atomic load)。 |
| `void setConvolverBypassRequested(bool shouldBypass)` | コンボルバーバイパス状態を要求します。 | **Message Thread**。 |
| `bool isConvolverBypassRequested() const` | コンボルバーバイパス要求状態を取得します。 | スレッドセーフ (atomic load)。 |

### 1.3 コンボルバー / EQ アクセサ

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `ConvolverProcessor& getConvolverProcessor()` | UI 用 `ConvolverProcessor` インスタンスへの参照を返します。 | **Message Thread** のみ。 |
| `EQEditProcessor& getEQProcessor()` | UI 用 `EQEditProcessor` インスタンスへの参照を返します。 | **Message Thread** のみ。 |

### 1.4 ノイズシェーパー学習

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume)` | ノイズシェーパー学習を開始します。 | **Message Thread**。非同期タスクとしてワーカースレッドを起動します。 |
| `void stopNoiseShaperLearning()` | ノイズシェーパー学習を停止します。 | **Message Thread**。 |
| `bool isNoiseShaperLearning() const` | 学習が実行中かどうかを返します。 | スレッドセーフ (atomic load)。 |
| `const NoiseShaperLearner::Progress& getNoiseShaperLearningProgress() const` | 学習の進捗状況を返します。 | スレッドセーフ。戻り値の参照はアトミック変数を含みます。 |

---

## 2. ConvolverProcessor

`ConvolverProcessor` は IR ファイルの読み込み、管理、および畳み込み処理を担当します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void prepareToPlay(double sampleRate, int samplesPerBlock)` | コンボルバーエンジンを準備します。 | **Message Thread**。 |
| `void releaseResources()` | エンジンリソースを解放します。 | **Message Thread**。 |
| `void process(juce::dsp::AudioBlock<double>& block)` | 入力ブロックに対して畳み込みを実行します。 | **Audio Thread** のみ。リアルタイム制約あり。 |
| `bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false)` | 指定された IR ファイルの非同期読み込みを開始します。 | **Message Thread**。 |
| `void setMix(float mixAmount)` | Dry/Wet ミックス比率 (0.0～1.0) を設定します。 | **Message Thread**。 |
| `float getMix() const` | 現在のミックス比率を取得します。 | スレッドセーフ (atomic load)。 |
| `void setPhaseMode(PhaseMode mode)` | 位相モード (AsIs, Mixed, Minimum) を設定します。 | **Message Thread**。 |
| `PhaseMode getPhaseMode() const` | 現在の位相モードを取得します。 | スレッドセーフ (atomic load)。 |
| `void setSmoothingTime(float timeSec)` | ミックススムーシング時間 (秒) を設定します。 | **Message Thread**。 |
| `float getSmoothingTime() const` | 現在のスムーシング時間を取得します。 | スレッドセーフ (atomic load)。 |
| `bool isIRLoaded() const` | IR がロード済みかどうかを返します。 | スレッドセーフ (RCU)。 |
| `juce::String getIRName() const` | 現在ロードされている IR のファイル名を返します。 | **Message Thread**。 |
| `int getLatencySamples() const` | 現在のアルゴリズム遅延 (サンプル数) を返します。 | スレッドセーフ (キャッシュされた値を返す)。 |

---

## 3. EQProcessor

`EQProcessor` は 20 バンドパラメトリック EQ の DSP 処理を提供します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void prepareToPlay(double sampleRate, int maxBlockSize)` | サンプルレートと最大ブロックサイズを設定し、内部状態をリセットします。 | **Message Thread**。 |
| `void process(juce::dsp::AudioBlock<double>& block)` | 入力ブロックに対して EQ 処理を実行します。 | **Audio Thread** のみ。 |
| `void setBandFrequency(int band, float freq)` | 指定バンドの周波数 (Hz) を設定します。 | **Message Thread**。 |
| `void setBandGain(int band, float gainDb)` | 指定バンドのゲイン (dB) を設定します。 | **Message Thread**。 |
| `void setBandQ(int band, float q)` | 指定バンドの Q 値を設定します。 | **Message Thread**。 |
| `void setBandEnabled(int band, bool enabled)` | 指定バンドの有効/無効を切り替えます。 | **Message Thread**。 |
| `EQBandParams getBandParams(int band) const` | 指定バンドの現在のパラメータを取得します。 | **Message Thread**。 |
| `void setTotalGain(float gainDb)` | トータルゲイン (dB) を設定します。 | **Message Thread**。 |
| `float getTotalGain() const` | 現在のトータルゲインを取得します。 | **Message Thread**。 |
| `void setAGCEnabled(bool enabled)` | 自動ゲインコントロール (AGC) の有効/無効を切り替えます。 | **Message Thread**。 |
| `bool getAGCEnabled() const` | AGC の有効状態を取得します。 | スレッドセーフ (atomic load)。 |

---

## 4. MKLNonUniformConvolver

`MKLNonUniformConvolver` は Intel IPP ベースの非均一パーティション畳み込みエンジンです。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `bool SetImpulse(const double* impulse, int irLen, int blockSize, double scale, bool enableDirectHead, const FilterSpec* filterSpec)` | IR データを設定し、内部 FFT 構造を構築します。 | **Message Thread** のみ。 |
| `void Add(const double* input, int numSamples)` | 入力サンプルをエンジンに供給します。 | **Audio Thread** のみ。 |
| `int Get(double* output, int numSamples)` | 畳み込み結果を出力バッファに書き出します。 | **Audio Thread** のみ。 |
| `void Reset()` | 内部バッファと状態をゼロクリアします。 | **Message Thread** または `releaseResources()` から呼び出し可能。 |
| `bool isReady() const` | エンジンが使用可能な状態かどうかを返します。 | スレッドセーフ (atomic load)。 |
| `int getLatency() const` | エンジンの先頭レイテンシー (サンプル数) を返します。 | スレッドセーフ。 |

---

## 5. SnapshotCoordinator

`SnapshotCoordinator` は DSP パラメータのスナップショットをロックフリーで管理します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `const GlobalSnapshot* getCurrent() const` | 現在アクティブなスナップショットへのポインタを返します。 | **Audio Thread** 安全 (atomic load)。 |
| `void switchImmediate(const GlobalSnapshot* newSnap)` | スナップショットを即座に切り替えます。 | **Message Thread** / **Worker Thread**。 |
| `void startFade(const GlobalSnapshot* target, int fadeSamples)` | 指定サンプル数でクロスフェードを開始します。 | **Message Thread** / **Worker Thread**。 |
| `bool updateFade(float& outAlpha, const GlobalSnapshot*& outCurrent, const GlobalSnapshot*& outTarget)` | フェード状態を更新し、現在のアルファ値とスナップショットを返します。 | **Audio Thread** のみ。 |
| `void advanceFade(int numSamples)` | フェードの進行状況を進めます。 | **Audio Thread** のみ。 |
| `bool tryCompleteFade()` | フェード完了を試み、成功したら状態を更新します。 | **Message Thread** (タイマーコールバック)。 |

---

## 6. WorkerThread

`WorkerThread` はパラメータ変更コマンドをデバウンスし、スナップショット生成を要求する専用スレッドです。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void start()` | ワーカースレッドを開始します。 | **Message Thread**。 |
| `void stop()` | ワーカースレッドを停止します。 | **Message Thread**。 |
| `void flush()` | 保留中のコマンドを即時処理するよう要求します。 | **Message Thread**。 |
| `void setSnapshotCreator(SnapshotCreatorCallback callback, void* userData)` | スナップショット生成コールバックを設定します。 | **Message Thread**。 |

---

## 7. SafeStateSwapper (RCU)

`SafeStateSwapper` はエポックベースの RCU (Read-Copy-Update) 機構を提供します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void swap(ConvolverState* newState)` | 新しい状態を公開し、古い状態を retired キューに積みます。 | **Message Thread** のみ。 |
| `void enterReader(int readerIndex)` | リーダーがクリティカルセクションに入ることを宣言します。 | **Audio Thread** のみ。 |
| `void exitReader(int readerIndex)` | リーダーがクリティカルセクションから出ることを宣言します。 | **Audio Thread** のみ。 |
| `ConvolverState* getState() const` | 現在アクティブな状態へのポインタを返します。 | **Audio Thread** のみ (`enterReader()` / `exitReader()` 間で呼び出し)。 |
| `ConvolverState* tryReclaim(uint64_t minReaderEpoch)` | 解放可能な古い状態を 1 件取得します。 | **DeferredFree Thread** のみ。 |
| `uint64_t getMinReaderEpoch() const` | アクティブな全リーダーの最小エポックを返します。 | **DeferredFree Thread** のみ。 |

---

## 8. NoiseShaperLearner

`NoiseShaperLearner` は CMA-ES を使用して格子型ノイズシェーパー係数を最適化します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `void startLearning(bool resume = false)` | 学習を開始 (または再開) します。 | **Message Thread**。 |
| `void stopLearning()` | 学習を停止します。 | **Message Thread**。 |
| `bool isRunning() const` | 学習が実行中かどうかを返します。 | スレッドセーフ。 |
| `void setLearningMode(LearningMode mode)` | 学習モード (Short, Middle, Long, Ultra, Continuous) を設定します。 | **Message Thread**。 |
| `const Progress& getProgress() const` | 学習の進捗状況を返します。 | スレッドセーフ。 |
| `int copyBestScoreHistory(double* outScores, int maxPoints) const` | ベストスコアの履歴をコピーします。 | **Message Thread**。 |

---

## 9. EQCoeffCache

`EQCoeffCache` は EQ 係数と Parallel モード用作業バッファを保持する不変キャッシュです。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `static uint64_t computeParamsHash(const convo::EQParameters& params)` | EQ パラメータのハッシュ値を計算します。 | スレッドセーフ (純粋関数)。 |
| `static EQCoeffCache* createCoeffCache(const convo::EQParameters& eqParams, double sampleRate, int maxBlockSize, uint64_t generation)` | 新しいキャッシュインスタンスを生成します。 | **Message Thread** / **Worker Thread**。 |

---

## 10. AllpassDesigner

`AllpassDesigner` は目標群遅延を近似する全通過フィルタを設計します。

| 関数シグネチャ | 説明 | スレッド安全性 |
| :--- | :--- | :--- |
| `bool design(double sampleRate, const std::vector<double>& freq_hz, const std::vector<double>& target_group_delay_samples, const Config& config, std::vector<SecondOrderAllpass>& sections, const std::function<bool()>& shouldExit, std::function<void(float)> progressCallback)` | 全通過フィルタを設計します (Greedy+AdaGrad または CMA-ES)。 | **Loader Thread** から呼び出し可能。 |
| `static juce::AudioBuffer<double> applyAllpassToIR(const juce::AudioBuffer<double>& linearIR, const std::vector<SecondOrderAllpass>& sections, double sampleRate, const std::vector<double>& freq_hz, int fftSize, const std::function<bool()>& shouldExit, std::function<void(float)> progressCallback)` | 設計済み全通過フィルタを IR に適用します。 | **Loader Thread** から呼び出し可能。 |