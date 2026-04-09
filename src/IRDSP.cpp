#include "IRDSP.h"
#include <algorithm>
#include <future>
#include <cstring>
#include <vector>

namespace IRDSP {

juce::AudioBuffer<double> resampleIR(
    const juce::AudioBuffer<double>& inputIR,
    double inputSR,
    double targetSR,
    const std::function<bool()>& shouldExit,
    const ResampleConfig& cfg)
{
    if (inputSR <= 0.0 || targetSR <= 0.0 || std::abs(inputSR - targetSR) <= 1e-6)
        return inputIR;

    const double ratio = targetSR / inputSR;
    const int inLength = inputIR.getNumSamples();
    const double expectedLen = static_cast<double>(inLength) * ratio + 2.0;
    if (expectedLen > static_cast<double>(std::numeric_limits<int>::max()))
        return {};

    const int maxOutLen = static_cast<int>(expectedLen);
    juce::AudioBuffer<double> resampled(inputIR.getNumChannels(), maxOutLen);
    resampled.clear();

    const int numCh = inputIR.getNumChannels();
    const int chunkSize = std::clamp(cfg.chunkSizeBase, 1024, 8192);

    // チャンネル並列処理（Loader Thread のみ）
    std::vector<std::future<void>> futures;

    for (int ch = 0; ch < numCh; ++ch) {
        futures.emplace_back(std::async(std::launch::async, [&, ch]() {
            if (shouldExit && shouldExit()) return;

            auto resampler = std::make_unique<r8b::CDSPResampler>(
                inputSR, targetSR, inLength,
                cfg.transBand, cfg.stopBandAtten, cfg.phase);

            const double* inPtr = inputIR.getReadPointer(ch);
            double* outPtr = resampled.getWritePointer(ch);

            int inputProcessed = 0;
            int done = 0;
            int iterations = 0;
            constexpr int maxIterations = 1000000;

            while (inputProcessed < inLength && done < maxOutLen && ++iterations < maxIterations) {
                if (shouldExit && shouldExit()) return;

                const int chunk = std::min(chunkSize, inLength - inputProcessed);
                // const_cast を排除：一時バッファにコピー
                std::vector<double> tempIn(chunk);
                std::memcpy(tempIn.data(), inPtr + inputProcessed, chunk * sizeof(double));

                double* r8bOutput = nullptr;
                const int generated = resampler->process(tempIn.data(), chunk, r8bOutput);
                inputProcessed += chunk;

                if (generated > 0) {
                    const int toCopy = std::min(generated, maxOutLen - done);
                    std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
                    done += toCopy;
                }
            }

            while (done < maxOutLen && ++iterations < maxIterations) {
                if (shouldExit && shouldExit()) return;
                double* r8bOutput = nullptr;
                const int generated = resampler->process(nullptr, 0, r8bOutput);
                if (generated <= 0) break;
                const int toCopy = std::min(generated, maxOutLen - done);
                std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
                done += toCopy;
            }
        }));
    }

    for (auto& f : futures) f.wait();

    resampled.setSize(numCh, maxOutLen, true, true, true);

    return resampled;
}

} // namespace IRDSP
