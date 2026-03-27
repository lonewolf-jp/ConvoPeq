# 分類ラベル一覧

- [ATT] : Audio Thread（オーディオ処理リアルタイムスレッド）で呼ばれる関数
- [MGT] : Message Thread / GUI Thread（JUCEのUI・メッセージスレッド）で呼ばれる関数
- [WLT] : Worker Thread / Learning Thread（学習・非同期ワーカースレッド）で呼ばれる関数
- [MTT] : Main Thread（非GUI）で呼ばれる関数
- [CTT] : Callback Thread（コールバック専用スレッド）で呼ばれる関数
- [HLP] : Helper/Utility（補助的なヘルパー・ユーティリティ関数）
- [STC] : Static/Constexpr（static関数・constexpr関数・クラス外の静的関数）
- [PRV] : Private/Internal（クラスや名前空間のprivate/protectedメンバ関数・内部専用関数）
- [TST] : Test/Debug（テスト・デバッグ・診断・ロギング専用関数）
- [CST] : Constructor/Destructor/Operator（コンストラクタ・デストラクタ・演算子オーバーロード）
- [TMP] : Template/Generic（テンプレート関数・ジェネリック型依存関数）
- [OTH] : Other/Uncategorized（その他・特殊用途・一時的なもの）

---

## EQProcessor.cpp

// EQProcessorクラスの実装部。引数・戻り値型はEQProcessor.hと同一。

[MGT] EQProcessor::EQProcessor()
[MGT] EQProcessor::~EQProcessor()
[MGT] void EQProcessor::releaseResources()
[MGT] void EQProcessor::resetToDefaults()
[MGT] void EQProcessor::reset()
[MGT] void EQProcessor::loadPreset(int index)
[MGT] bool EQProcessor::loadFromTextFile(const juce::File& file)
[MGT] juce::ValueTree EQProcessor::getState() const
[MGT] void EQProcessor::setState(const juce::ValueTree& v)
[MGT] void EQProcessor::syncStateFrom(const EQProcessor& other)
[MGT] void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
[MGT] void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
[ATT] void EQProcessor::prepareToPlay(double sampleRate, int newMaxInternalBlockSize)
[MGT] void EQProcessor::setBandFrequency(int band, float freq)
[MGT] void EQProcessor::setBandGain(int band, float gainDb)
[MGT] void EQProcessor::setBandQ(int band, float q)
[MGT] void EQProcessor::setBandEnabled(int band, bool enabled)
[MGT] void EQProcessor::setTotalGain(float gainDb)
[MGT] void EQProcessor::setAGCEnabled(bool enabled)
[MGT] bool EQProcessor::getAGCEnabled() const
[MGT] void EQProcessor::setBandType(int band, EQBandType type)
[MGT] void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
[MGT] EQBandParams EQProcessor::getBandParams(int band) const
[MGT] EQProcessor::EQState*EQProcessor::getEQState() const
[MGT] float EQProcessor::getTotalGain() const
[MGT] EQBandType EQProcessor::getBandType(int band) const
[MGT] EQChannelMode EQProcessor::getBandChannelMode(int band) const
[MGT] double EQProcessor::calculateAGCGain(double inputEnv, double outputEnv) const noexcept
[MGT] void EQProcessor::processAGC(juce::dsp::AudioBlock<double>& block)
[MGT] bool EQProcessor::isBufferSilent(const juce::AudioBuffer<double>& buffer, int numSamples) const noexcept
[ATT] void EQProcessor::process(juce::dsp::AudioBlock<double>& block)
[NON] static inline double calculateRMS(const double* data, int numSamples) noexcept
[NON] namespace { ... }
// processBand, processBandStereo, applyGainRamp_AVX2 など

---

## Fixed15TapNoiseShaper.h

### クラス: convo::Fixed15TapNoiseShaper

[MGT] bool setCoefficients(const std::array<double, ORDER>& newCoeffs) noexcept
[MGT] void setDiagnosticsWindowSamples(uint32_t samples) noexcept
[MGT] uint32_t getDiagnosticsWindowSamples() const noexcept
[MGT] Diagnostics getDiagnostics() const noexcept
[MGT] void prepare(double sampleRate, int bitDepth) noexcept
[MGT] void reset() noexcept
[ATT] void processStereoBlock(double*dataL, double* dataR, int numSamples, double headroom) noexcept
[ATT] inline double processSample(double x, int channel, double& outError) noexcept
[NON] inline void publishDiagnostics(double sumSqLBlock, double sumSqRBlock, double peakAbsBlock, uint32_t sampleCountBlock, bool hasRightChannel) noexcept
[TST] inline void resetDiagnostics() noexcept
[HLP] inline double absNoLibm(double x) const noexcept
[HLP] inline double get(const std::array<double, ORDER>& buffer, int idx, int k) const noexcept
[STC] static inline uint64_t rotl(const uint64_t x, int k) noexcept
[STC] static inline uint64_t xoshiro256plusplus(Xoshiro256State& state) noexcept
[HLP] inline double uniform(Xoshiro256State& state) const noexcept
[HLP] inline double quantize(double v, Xoshiro256State& rng) const noexcept
[STC] static void selectPresetWithInterpolation(double sampleRate, int& idxLow, int& idxHigh, double& t) noexcept

---

## FixedNoiseShaper.h

### クラス: convo::FixedNoiseShaper

[MGT] bool setCoefficients(const std::array<double, ORDER>& newCoeffs) noexcept
[MGT] void setDiagnosticsWindowSamples(uint32_t samples) noexcept
[MGT] uint32_t getDiagnosticsWindowSamples() const noexcept
[MGT] Diagnostics getDiagnostics() const noexcept
[MGT] void prepare(double sampleRate, int bitDepth) noexcept
[MGT] void reset() noexcept
[ATT] void processStereoBlock(double*dataL, double* dataR, int numSamples, double headroom) noexcept
[ATT] inline double processSample(double x, int channel, double& outError) noexcept
[NON] inline void publishDiagnostics(double sumSqLBlock, double sumSqRBlock, double peakAbsBlock, uint32_t sampleCountBlock, bool hasRightChannel) noexcept
[TST] inline void resetDiagnostics() noexcept
[HLP] inline double absNoLibm(double x) const noexcept
[HLP] inline double get(const std::array<double, ORDER>& buffer, int idx, int k) const noexcept
[STC] static inline uint64_t rotl(const uint64_t x, int k) noexcept
[STC] static inline uint64_t xoshiro256plusplus(Xoshiro256State& state) noexcept
[HLP] inline double uniform(Xoshiro256State& state) const noexcept
[HLP] inline double quantize(double v, Xoshiro256State& rng) const noexcept
[STC] static void selectPresetWithInterpolation(double sampleRate, int& idxLow, int& idxHigh, double& t) noexcept

---

## ConvolverControlPanel.h

[MGT] ConvolverControlPanel(AudioEngine& audioEngine)
[MGT] ~ConvolverControlPanel() override
[MGT] void paint(juce::Graphics& g) override
[MGT] void resized() override
[MGT] void updateIRInfo()
[MGT] void updateFilterModeButtons()
[MGT] void buttonClicked(juce::Button*button) override
[MGT] void sliderValueChanged(juce::Slider* slider) override
[MGT] void mouseDown(const juce::MouseEvent& event) override
[MGT] void timerCallback() override
[MGT] void updateWaveformPath()
[MGT] void updateMixedPhaseControlsEnabled()
[WLT] void startAsyncIRLoadPreview(const juce::File& irFile)
[WLT] void finishAsyncIRLoadPreview(const juce::File& irFile, const ConvolverProcessor::IRLoadPreview& preview, int requestId)
[MGT] void setIRPreviewInProgress(bool isInProgress)
[MGT] void showIRAdvancedWindow()
[MGT] void updateTrimSlider()
[MGT] void markConvolverParameterDirty()
[MGT] void applyPendingConvolverParameters()
[MGT] bool hasPendingConvolverParameters() const noexcept

---

## ConvolverControlPanel.cpp

[MGT] ConvolverControlPanel::ConvolverControlPanel(AudioEngine& audioEngine)
[MGT] ConvolverControlPanel::~ConvolverControlPanel()
[MGT] void ConvolverControlPanel::paint(juce::Graphics& g)
[MGT] void ConvolverControlPanel::resized()
[MGT] void ConvolverControlPanel::updateFilterModeButtons()
[MGT] void ConvolverControlPanel::updateTrimSlider()
[MGT] void ConvolverControlPanel::updateMixedPhaseControlsEnabled()
[MGT] void ConvolverControlPanel::buttonClicked(juce::Button*button)
[MGT] void ConvolverControlPanel::showIRAdvancedWindow()
[WLT] void ConvolverControlPanel::startAsyncIRLoadPreview(const juce::File& irFile)
[WLT] void ConvolverControlPanel::finishAsyncIRLoadPreview(const juce::File& irFile, const ConvolverProcessor::IRLoadPreview& preview, int requestId)
[MGT] void ConvolverControlPanel::setIRPreviewInProgress(bool isInProgress)
[MGT] void ConvolverControlPanel::sliderValueChanged(juce::Slider* slider)
[MGT] void ConvolverControlPanel::timerCallback()
[MGT] void ConvolverControlPanel::markConvolverParameterDirty()
[MGT] bool ConvolverControlPanel::hasPendingConvolverParameters() const noexcept
[MGT] void ConvolverControlPanel::applyPendingConvolverParameters()
[MGT] void ConvolverControlPanel::mouseDown(const juce::MouseEvent& event)
[MGT] void ConvolverControlPanel::updateIRInfo()
[MGT] void ConvolverControlPanel::updateWaveformPath()

---

## ConvolverProcessor.h

[MGT] ConvolverProcessor()
[MGT] ~ConvolverProcessor()
[ATT] void prepareToPlay(double sampleRate, int samplesPerBlock)
[ATT] void releaseResources()
[WLT] bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false)
[ATT] void process(juce::dsp::AudioBlock<double>& block)
[MGT] void setBypass(bool shouldBypass)
[MGT] bool isBypassed() const
[MGT] void setMix(float mixAmount)
[MGT] float getMix() const
[MGT] void setPhaseMode(PhaseMode mode)
[MGT] PhaseMode getPhaseMode() const
[MGT] void setUseMinPhase(bool useMinPhase)
[MGT] bool getUseMinPhase() const
[MGT] void setNUCFilterModes(convo::HCMode hcMode, convo::LCMode lcMode)
[MGT] void setExperimentalDirectHeadEnabled(bool enabled)
[MGT] bool getExperimentalDirectHeadEnabled() const
[MGT] void setSmoothingTime(float timeSec)
[MGT] float getSmoothingTime() const
[MGT] void setMixedTransitionStartHz(float hz)
[MGT] float getMixedTransitionStartHz() const
[MGT] void setMixedTransitionEndHz(float hz)
[MGT] float getMixedTransitionEndHz() const
[MGT] void setMixedPreRingTau(float tau)
[MGT] float getMixedPreRingTau() const
[MGT] void setRebuildDebounceMs(int ms)
[MGT] int getRebuildDebounceMs() const
[MGT] void setTargetIRLength(float timeSec)
[MGT] float getTargetIRLength() const
[MGT] void applyAutoDetectedIRLength(float timeSec)
[MGT] void setIRLengthManualOverride(bool isManual)
[MGT] bool hasManualIRLengthOverride() const
[MGT] float getAutoDetectedIRLength() const
[NON] static float getMaximumAllowedIRLengthSecForSampleRate(double sampleRate)
[STC] static IRLoadPreview analyzeImpulseResponseFile(const juce::File& irFile, double processingSampleRate)
[MGT] void reset()
[MGT] bool isIRLoaded() const
[MGT] juce::String getIRName() const
[MGT] int getIRLength() const
[MGT] juce::String getLastError() const
[MGT] float getLoadProgress() const
[MGT] int getCurrentBufferSize() const
[MGT] LatencyBreakdown getLatencyBreakdown() const
[MGT] int getLatencySamples() const
[MGT] int getTotalLatencySamples() const
[MGT] std::vector<float> getIRWaveform() const
[MGT] std::vector<float> getIRMagnitudeSpectrum() const
[MGT] double getIRSpectrumSampleRate() const
[MGT] juce::ValueTree getState() const
[MGT] void setState(const juce::ValueTree& state)

## EQControlPanel.h

### クラス: EQControlPanel

- EQControlPanel(AudioEngine& engine)
- void handleLoadError(const juce::String& error)

---

## EQControlPanel.cpp

[MGT] void EQControlPanel::updateBandValues(int band)
[MGT] void EQControlPanel::updateAllControls()
[MGT] void EQControlPanel::updateLPFModeButtons()
[MGT] void EQControlPanel::labelTextChanged(juce::Label*label)
[MGT] void EQControlPanel::editorShown(juce::Label* label, juce::TextEditor& editor)
[MGT] void EQControlPanel::buttonClicked(juce::Button*button)
[MGT] void EQControlPanel::comboBoxChanged(juce::ComboBox* comboBox)
[MGT] const EQControlPanel::ControlID*EQControlPanel::findControlId(const juce::Component* control) const
[MGT] void EQControlPanel::paint(juce::Graphics& g)
[MGT] void EQControlPanel::resized()

---

## EQProcessor.h

### クラス: EQProcessor

- EQProcessor()
- void releaseResources() override
- bool isBusesLayoutSupported(const BusesLayout& layouts) const override
- bool supportsDoublePrecisionProcessing() const override
- void processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages) override
- void processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer& midiMessages) override
- bool hasEditor() const override
- void getStateInformation(juce::MemoryBlock& destData) override
- void setStateInformation(const void* data, int sizeInBytes) override

---

## AudioEngineProcessor.cpp

[MGT] AudioEngineProcessor::AudioEngineProcessor(AudioEngine& engineRef)
[MGT] juce::String AudioEngineProcessor::getName() const
[MGT] bool AudioEngineProcessor::acceptsMidi() const
[MGT] bool AudioEngineProcessor::producesMidi() const
[MGT] bool AudioEngineProcessor::isMidiEffect() const
[MGT] double AudioEngineProcessor::getTailLengthSeconds() const
[MGT] int AudioEngineProcessor::getNumPrograms()
[MGT] int AudioEngineProcessor::getCurrentProgram()
[MGT] void AudioEngineProcessor::setCurrentProgram(int)
[MGT] juce::String AudioEngineProcessor::getProgramName(int)
[MGT] void AudioEngineProcessor::changeProgramName(int, const juce::String&)
[ATT] void AudioEngineProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
[ATT] void AudioEngineProcessor::releaseResources()
[NON] bool AudioEngineProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
[ATT] void AudioEngineProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
[ATT] void AudioEngineProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)
[MGT] bool AudioEngineProcessor::hasEditor() const
[MGT] juce::AudioProcessorEditor*AudioEngineProcessor::createEditor()
[MGT] void AudioEngineProcessor::getStateInformation(juce::MemoryBlock&)
[MGT] void AudioEngineProcessor::setStateInformation(const void*, int)

# AudioEngine.h / AudioEngine.cpp / AsioBlacklist.h 関数リスト（完全版）

---

## AudioEngine.h / AudioEngine.cpp

### パブリックメンバ関数

[MGT] AudioEngine();
[MGT] ~AudioEngine() override;
[MGT] void initialize();
[ATT] void prepareToPlay(int samplesPerBlockExpected, double sampleRate) override;
[ATT] void releaseResources() override;
[ATT] void getNextAudioBlock(const juce::AudioSourceChannelInfo& bufferToFill) override;
[ATT] void processBlockDouble(juce::AudioBuffer<double>& buffer);
[MGT] void changeListenerCallback(juce::ChangeBroadcaster*source) override;
[MGT] void eqBandChanged(EQProcessor* processor, int bandIndex) override;
[MGT] void eqGlobalChanged(EQProcessor*processor) override;
[MGT] void convolverParamsChanged(ConvolverProcessor* processor) override;
[MGT] void timerCallback() override;
[MGT] ConvolverProcessor& getConvolverProcessor();
[MGT] EQProcessor& getEQProcessor();
[MGT] double getSampleRate() const;
[MGT] double getProcessingSampleRate() const;
[MGT] LatencyBreakdown getCurrentLatencyBreakdown() const;
[MGT] int getCurrentLatencySamples() const;
[MGT] double getCurrentLatencyMs() const;
[MGT] float getInputLevel() const;
[MGT] float getOutputLevel() const;
[MGT] int getFifoNumReady() const;
[MGT] void readFromFifo(float*dest, int numSamples);
[MGT] void skipFifo(int numSamples);
[MGT] void calcEQResponseCurve(float*outMagnitudesL, float*outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);
[MGT] void setEqBypassRequested(bool shouldBypass);
[MGT] void setConvolverBypassRequested(bool shouldBypass);
[MGT] bool isEqBypassRequested() const noexcept;
[MGT] bool isConvolverBypassRequested() const noexcept;
[MGT] void setConvolverPhaseMode(ConvolverProcessor::PhaseMode mode);
[MGT] ConvolverProcessor::PhaseMode getConvolverPhaseMode() const;
[MGT] void setConvolverUseMinPhase(bool useMinPhase);
[MGT] bool getConvolverUseMinPhase() const;
[MGT] void requestEqPreset(int presetIndex);
[MGT] void requestEqPresetFromText(const juce::File& file);
[MGT] void requestConvolverPreset(const juce::File& irFile);
[MGT] void requestLoadState(const juce::ValueTree& state);
[MGT] juce::ValueTree getCurrentState() const;
[MGT] void setProcessingOrder(ProcessingOrder order);
[MGT] ProcessingOrder getProcessingOrder() const;
[MGT] void setAnalyzerSource(AnalyzerSource source);
[MGT] AnalyzerSource getAnalyzerSource() const;
[MGT] void setAnalyzerEnabled(bool enabled) noexcept;
[MGT] bool isAnalyzerEnabled() const noexcept;
[MGT] void setInputHeadroomDb(float db);
[MGT] float getInputHeadroomDb() const;
[MGT] void setOutputMakeupDb(float db);
[MGT] float getOutputMakeupDb() const;
[MGT] void setConvolverInputTrimDb(float db);
[MGT] float getConvolverInputTrimDb() const;
[MGT] void setDitherBitDepth(int bitDepth);
[MGT] int getDitherBitDepth() const;
[MGT] void setNoiseShaperType(NoiseShaperType type);
[MGT] NoiseShaperType getNoiseShaperType() const;
[MGT] void setFixedNoiseLogIntervalMs(int intervalMs) noexcept;
[MGT] int getFixedNoiseLogIntervalMs() const noexcept;
[MGT] void setFixedNoiseWindowSamples(int windowSamples) noexcept;
[MGT] int getFixedNoiseWindowSamples() const noexcept;
[MGT] void setSoftClipEnabled(bool enabled);
[MGT] bool isSoftClipEnabled() const;
[MGT] void setSaturationAmount(float amount);
[MGT] float getSaturationAmount() const;
[MGT] void setOversamplingFactor(int factor);
[MGT] int getOversamplingFactor() const;
[MGT] void setOversamplingType(OversamplingType type);
[MGT] OversamplingType getOversamplingType() const;
[MGT] void setConvHCFilterMode(convo::HCMode mode) noexcept;
[MGT] convo::HCMode getConvHCFilterMode() const noexcept;
[MGT] void setConvLCFilterMode(convo::LCMode mode) noexcept;
[MGT] convo::LCMode getConvLCFilterMode() const noexcept;
[MGT] void setEqLPFFilterMode(convo::HCMode mode) noexcept;
[MGT] convo::HCMode getEqLPFFilterMode() const noexcept;
[WLT] void startNoiseShaperLearning(NoiseShaperLearner::LearningMode mode, bool resume = false);
[WLT] void stopNoiseShaperLearning();
[WLT] void setNoiseShaperLearningMode(NoiseShaperLearner::LearningMode mode);
[WLT] NoiseShaperLearner::LearningMode getNoiseShaperLearningMode() const;
[WLT] bool isNoiseShaperLearning() const;
[WLT] const NoiseShaperLearner::Progress& getNoiseShaperLearningProgress() const;
[WLT] int copyNoiseShaperLearningHistory(double*outScores, int maxPoints) const noexcept;
[WLT] const char* getNoiseShaperLearningError() const noexcept;
[WLT] static int getAdaptiveSampleRateBankCount() noexcept;
[WLT] static double getAdaptiveSampleRateBankHz(int bankIndex) noexcept;
[WLT] void getCurrentAdaptiveCoefficients(double*outCoeffs, int maxCoefficients) const noexcept;
[WLT] void setCurrentAdaptiveCoefficients(const double* coeffs, int numCoefficients);
[WLT] void getAdaptiveCoefficientsForSampleRate(double sampleRate, double*outCoeffs, int maxCoefficients) const noexcept;
[WLT] void setAdaptiveCoefficientsForSampleRate(double sampleRate, const double* coeffs, int numCoefficients);
[WLT] void getAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, double*outCoeffs, int maxCoefficients) const noexcept;
[WLT] void setAdaptiveCoefficientsForSampleRateAndBitDepth(double sampleRate, int bitDepth, const double* coeffs, int numCoefficients);
[WLT] void setAdaptiveAutosaveCallback(std::function<void()> callback);
[WLT] void requestAdaptiveAutosave();
[WLT] void publishCoeffs(const double*coeffs);
[WLT] static int getAdaptiveCoeffBankIndex(double sampleRate, int bitDepth, NoiseShaperLearner::LearningMode mode) noexcept;
[WLT] bool getAdaptiveNoiseShaperState(int bankIndex, NoiseShaperLearner::State& outState) const noexcept;
[WLT] void setAdaptiveNoiseShaperState(int bankIndex, const NoiseShaperLearner::State& inState) noexcept;
[WLT] void requestRebuild(double sampleRate, int samplesPerBlock);
[WLT] void commitNewDSP(DSPCore* newDSP, int generation);
[WLT] bool isRebuildObsolete(int generation) const;
[WLT] void rebuildThreadLoop();

### 静的・インライン・プライベート・ヘルパー関数

[NON] static inline const CoeffSet*getActiveCoeffSet(const AdaptiveCoeffBankSlot& slot) noexcept;
[NON] static inline bool reserveInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept;
[NON] static inline CoeffSet* getReservedInactiveCoeffSet(AdaptiveCoeffBankSlot& slot) noexcept;
[OTH] ...（DSPCore, CoeffSetWriteLockGuard, その他の内部クラス・RAIIガードのメンバ関数も含む）
[STC] AVX2/数値処理系のstatic/inline関数（absNoLibm, isFiniteNoLibm, fastTanh, musicalSoftClipScalar, softClipBlockAVX2, scaleBlockFallback, applyGainRamp など）

---

## AsioBlacklist.h

[MGT] bool loadFromFile(const juce::File& file) noexcept
[MGT] bool isBlacklisted(const juce::String& deviceName) const noexcept

---

※ AudioEngine.h/cppは、クラス内のprivate/protected/publicすべての関数・static関数・インライン関数・RAIIガード・内部クラスのメンバ関数も含めて抽出しています。

---

## InputBitDepthTransform.h

### 名前空間: convo::input_transform

[ATT] inline bool isFiniteAndAboveThresholdMask(double value, double threshold) noexcept
[ATT] inline void sanitizeAndLimit(double*data, int numSamples) noexcept
[ATT] inline void applyHighQuality64BitTransform(double* data, int numSamples, double gain = 1.0) noexcept
[ATT] inline void convertFloatToDoubleHighQuality(const float*src, double* dst, int numSamples, double gain = 1.0) noexcept
[ATT] inline void convertDoubleToDoubleHighQuality(const double*src, double* dst, int numSamples, double gain = 1.0) noexcept

---

## LatticeNoiseShaper.h

### クラス: LatticeNoiseShaper

[MGT] void prepare(int bitDepth) noexcept
[MGT] void reset() noexcept
[MGT] void setCoefficients(const double*newCoeffs, int numCoeffs) noexcept
[MGT] void startCoefficientRamp(const double* newCoeffs) noexcept
[MGT] void applyMatchedCoefficients(const double*newCoeffs, int numCoeffs) noexcept
[MGT] const double* getCoefficients() const noexcept
[ATT] void processStereoBlock(double*dataL, double* dataR, int numSamples, double headroom) noexcept
[NON] inline void clampStateSIMD(double* state) noexcept
[NON] static inline double clampCoeff(double value) noexcept
[NON] static inline double absNoLibm(double x) noexcept
[NON] static inline double killDenormal(double x) noexcept
[NON] static inline uint64_t rotl(const uint64_t x, int k) noexcept
[NON] static inline uint64_t xoshiro256plusplus(Xoshiro256State& state) noexcept
[NON] inline double uniform(Xoshiro256State& state) const noexcept
[NON] inline double quantize(double value, Xoshiro256State& rng) const noexcept
[NON] inline double computeFeedback(const std::array<double, kOrder>& channelState) const noexcept
[NON] inline void advanceState(std::array<double, kOrder>& channelState, double error) const noexcept
[NON] inline double processSample(int channel, double inputSample, std::array<double, kOrder>& channelState, double headroom) noexcept

---

## LockFreeRingBuffer.h

### テンプレートクラス: LockFreeRingBuffer<T, Capacity>

[ATT] bool push(const T& item) noexcept
[ATT] bool pop(T& item) noexcept
[ATT] size_t size() const noexcept
[ATT] void clear() noexcept
