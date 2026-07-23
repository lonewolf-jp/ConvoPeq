//============================================================================
#include "AudioEngineProcessor.h"
#include <cmath>

AudioEngineProcessor::AudioEngineProcessor(AudioEngine& engineRef)
    : juce::AudioProcessor(BusesProperties()
                           .withInput("Input", juce::AudioChannelSet::stereo(), true)
                           .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      audioEngine(engineRef)
{
}

const juce::String AudioEngineProcessor::getName() const
{
    return "ConvoPeqEngine";
}

bool AudioEngineProcessor::acceptsMidi() const { return false; }
bool AudioEngineProcessor::producesMidi() const { return false; }
bool AudioEngineProcessor::isMidiEffect() const { return false; }
// D5: IR長のみの概算値。oversampling やフィルターによるテール延長は未反映。
// C5: cachedTailLength を返す（Runtime Publish 時に更新。ValueTree 依存を断つ）
// 現在の ConvoPeq 実装の Runtime Publish シーケンスでは同一スレッドで実行されるため double（非 atomic）で十分
double AudioEngineProcessor::getTailLengthSeconds() const
{
    return cachedTailLength;
}

int AudioEngineProcessor::getNumPrograms() { return 1; }
int AudioEngineProcessor::getCurrentProgram() { return 0; }
void AudioEngineProcessor::setCurrentProgram(int) {}
const juce::String AudioEngineProcessor::getProgramName(int) { return {}; }
void AudioEngineProcessor::changeProgramName(int, const juce::String&) {}

void AudioEngineProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    audioEngine.prepareToPlay(samplesPerBlock, sampleRate);
    setLatencySamples(audioEngine.getTotalLatencySamples());

    // C5: cachedTailLength を更新（Runtime Publish 時を Authority にする）
    // prepareToPlay は IR 変更時に呼ばれるため、ここでキャッシュを更新
    const auto convState = audioEngine.getConvolverStateTree();
    if (convState.isValid())
    {
        const double irLengthSec = static_cast<double>(convState.getProperty("irLength", 0.0));
        cachedTailLength = (std::isfinite(irLengthSec) && irLengthSec > 0.0) ? irLengthSec : 0.0;
    }
    else
    {
        cachedTailLength = 0.0;
    }
}

void AudioEngineProcessor::releaseResources()
{
    // ★ 案B: Engine が Prepared 状態でなければ早期リターン
    //   JUCE が audio device 列挙時に releaseResources() を複数回呼ぶことがある。
    //   Engine 側の CAS ガードが二重解放を防止しているが、ここで事前チェックすることで
    //   "duplicate release ignored" のログノイズを抑制し、無駄な処理を回避する。
    if (!audioEngine.isEnginePrepared())
    {
        DBG("[DIAG] AudioEngineProcessor::releaseResources: skipped (engine not prepared)");
        return;
    }
    audioEngine.releaseResources();
}

bool AudioEngineProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto inSet = layouts.getMainInputChannelSet();
    const auto outSet = layouts.getMainOutputChannelSet();

    if (outSet.isDisabled())
        return false;
    if (outSet.size() < 1 || outSet.size() > 2)
        return false;
    if (!inSet.isDisabled() && (inSet.size() < 1 || inSet.size() > 2))
        return false;

    return true;
}

#ifndef CONVOPEQ_STANDALONE_ONLY
void AudioEngineProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
    audioEngine.getNextAudioBlock(info);
}
#else
void AudioEngineProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    // Standalone-only: Float route is unused; setDoublePrecisionProcessing(true)
    // ensures processBlock(double&) is always called. This stub satisfies the
    // pure virtual interface requirement of juce::AudioProcessor.
    buffer.clear();
}
#endif

void AudioEngineProcessor::processBlock(juce::AudioBuffer<double>& buffer, juce::MidiBuffer&)
{
    audioEngine.processBlockDouble(buffer);
}

bool AudioEngineProcessor::hasEditor() const { return false; }
juce::AudioProcessorEditor* AudioEngineProcessor::createEditor() { return nullptr; }

void AudioEngineProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    destData.reset();
    const auto state = audioEngine.getCurrentState();
    if (!state.isValid())
        return;

    if (auto xml = state.createXml())
        copyXmlToBinary(*xml, destData);
}

void AudioEngineProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0)
        return;

    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));
    if (xmlState == nullptr)
        return;

    const auto state = juce::ValueTree::fromXml(*xmlState);
    if (!state.isValid())
        return;

    auto* messageManager = juce::MessageManager::getInstanceWithoutCreating();
    if (messageManager != nullptr && !messageManager->isThisTheMessageThread())
    {
        const juce::WeakReference<AudioEngine> weakEngine(&audioEngine);
        juce::MessageManager::callAsync([weakEngine, state]()
        {
            if (auto* engine = weakEngine.get())
                engine->requestLoadState(state);
        });
        return;
    }

    audioEngine.requestLoadState(state);
}
