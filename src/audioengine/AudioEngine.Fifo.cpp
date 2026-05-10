#include <JuceHeader.h>
#include "AudioEngine.h"

#if defined(CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_FIFO_UI)

//--------------------------------------------------------------
// FIFOからデータ読み出し (UI Thread)
//--------------------------------------------------------------
void AudioEngine::readFromFifo(float* dest, int numSamples)
{
    const int actualRead = analyzerFifo.popMixToMono(dest, numSamples);
    if (actualRead < numSamples)
        juce::FloatVectorOperations::clear(dest + actualRead, numSamples - actualRead);
}

//--------------------------------------------------------------
// FIFOからデータをスキップ (Latency対策)
//--------------------------------------------------------------
void AudioEngine::skipFifo(int numSamples)
{
    analyzerFifo.skip(numSamples);
}

#endif // CONVOPEQ_ENABLE_AUDIOENGINE_SPLIT_FIFO_UI
