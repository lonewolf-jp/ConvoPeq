#include "IRDSP.h"
#include <algorithm>
#include <future>
#include <cstring>
#include <vector>
#include <atomic>

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

    const int inLength = inputIR.getNumSamples();

    // r8brain の内部フィルタレイテンシを考慮した出力バッファサイズ。
    // CDSPResampler::getLatency() は常に 0 を返す（内部レイテンシ自動除去）ため、
    // 単純な inLength * ratio + margin ではフィルタタップ長が考慮されず、
    // 高品質設定（140dB/2%）ではフラッシュ出力が切り捨てられるリスクがある。
    // getMaxOutLen() は理論上の最大出力長を返すため、これをバッファサイズとして使用する。
    r8b::CDSPResampler tempResampler(inputSR, targetSR, inLength,
                                      cfg.transBand, cfg.stopBandAtten, cfg.phase);
    const int maxOutLen = tempResampler.getMaxOutLen(inLength);
    if (maxOutLen <= 0)
        return {};

    juce::AudioBuffer<double> resampled(inputIR.getNumChannels(), maxOutLen);
    resampled.clear();

    const int numCh = inputIR.getNumChannels();
    const int chunkSize = std::clamp(cfg.chunkSizeBase, 1024, 8192);

    // チャンネル並列処理（Loader Thread のみ）
    std::vector<std::future<void>> futures;
    std::vector<int> channelDone(numCh, -1);  // -1初期化: 例外・未完了を識別
    std::atomic<bool> anyChannelCancelled{false};

    for (int ch = 0; ch < numCh; ++ch) {
        futures.emplace_back(std::async(std::launch::async, [&, ch]() {
            try {
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
                    if (shouldExit && shouldExit()) {
                        anyChannelCancelled.store(true, std::memory_order_relaxed);
                        return;
                    }

                    const int chunk = std::min(chunkSize, inLength - inputProcessed);
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
                    if (shouldExit && shouldExit()) {
                        anyChannelCancelled.store(true, std::memory_order_relaxed);
                        return;
                    }
                    double* r8bOutput = nullptr;
                    const int generated = resampler->process(nullptr, 0, r8bOutput);
                    if (generated <= 0) break;
                    const int toCopy = std::min(generated, maxOutLen - done);
                    std::memcpy(outPtr + done, r8bOutput, toCopy * sizeof(double));
                    done += toCopy;
                }

                channelDone[ch] = done;
            } catch (...) {
                anyChannelCancelled.store(true, std::memory_order_relaxed);
                throw;  // get() で再送出
            }
        }));
    }

    for (auto& f : futures) f.get();  // get(): 例外を確実に伝播

    if (anyChannelCancelled.load(std::memory_order_relaxed))
        return {};

    // 理論上は全チャンネル同一長となるが、安全のため maxDone を採用
    const int maxDone = *std::max_element(channelDone.begin(), channelDone.end());
    if (maxDone < 0)
        return {};
    if (maxDone < maxOutLen)
        resampled.setSize(numCh, maxDone, true, true, true);
    // maxDone == maxOutLen の場合はバッファサイズを維持

    return resampled;
}

} // namespace IRDSP
