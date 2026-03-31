#pragma once

#include <juce_core/juce_core.h>
#include <memory>
#include <atomic>
#include <cstdint>
#include "StereoConvolver.h"

/**
 * @struct ConvolverState
 * @brief コンボルバーの全状態（IRデータ、NUCエンジン、メタデータ）をカプセル化する。
 * 
 * SafeStateSwapper によって管理され、Audio Thread から RCU 方式で参照される。
 */
struct ConvolverState {
    uint64_t generationId = 0;
    convo::StereoConvolver* convolver = nullptr;
    convo::StereoConvolver* fadingConvolver = nullptr;

    // メタデータ (表示やリビルド用)
    std::shared_ptr<juce::AudioBuffer<double>> loadedIR;
    std::shared_ptr<juce::AudioBuffer<double>> displayIR;
    double loadedSR = 0.0;
    int targetLength = 0;
    juce::File irFile;
    double scaleFactor = 1.0;
    bool isRebuild = false;

    ConvolverState() = default;
    
    ~ConvolverState() {
        if (convolver) {
            convolver->release();
            convolver = nullptr;
        }
        if (fadingConvolver) {
            fadingConvolver->release();
            fadingConvolver = nullptr;
        }
    }

    // コピー禁止 (SafeStateSwapper は unique_ptr を使用)
    ConvolverState(const ConvolverState&) = delete;
    ConvolverState& operator=(const ConvolverState&) = delete;

    // ムーブは必要に応じて定義可能だが、基本は unique_ptr で扱う
    ConvolverState(ConvolverState&& other) noexcept 
        : generationId(other.generationId),
          convolver(other.convolver),
          fadingConvolver(other.fadingConvolver),
          loadedIR(std::move(other.loadedIR)),
          displayIR(std::move(other.displayIR)),
          loadedSR(other.loadedSR),
          targetLength(other.targetLength),
          irFile(std::move(other.irFile)),
          scaleFactor(other.scaleFactor),
          isRebuild(other.isRebuild)
    {
        other.convolver = nullptr;
        other.fadingConvolver = nullptr;
    }

    ConvolverState& operator=(ConvolverState&& other) noexcept {
        if (this != &other) {
            if (convolver) convolver->release();
            if (fadingConvolver) fadingConvolver->release();
            generationId = other.generationId;
            convolver = other.convolver;
            fadingConvolver = other.fadingConvolver;
            loadedIR = std::move(other.loadedIR);
            displayIR = std::move(other.displayIR);
            loadedSR = other.loadedSR;
            targetLength = other.targetLength;
            irFile = std::move(other.irFile);
            scaleFactor = other.scaleFactor;
            isRebuild = other.isRebuild;
            other.convolver = nullptr;
            other.fadingConvolver = nullptr;
        }
        return *this;
    }
};
