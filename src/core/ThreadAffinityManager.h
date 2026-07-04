#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <cstdint>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>

#include "audioengine/AtomicAccess.h"
#endif

enum class ThreadType
{
    Worker,
    LearnerMain,
    LearnerEval,
    HeavyBackground,
    LightBackground,
    UI
};

struct ThreadAffinityMasks
{
#ifdef _WIN32
    DWORD_PTR worker = 0;
    DWORD_PTR learnerMain = 0;
    DWORD_PTR learnerEvalBase = 0;
    DWORD_PTR heavyBackground = 0;
    DWORD_PTR lightBackground = 0;
    DWORD_PTR ui = 0;
#else
    std::uint64_t worker = 0;
    std::uint64_t learnerMain = 0;
    std::uint64_t learnerEvalBase = 0;
    std::uint64_t heavyBackground = 0;
    std::uint64_t lightBackground = 0;
    std::uint64_t ui = 0;
#endif
};

class ThreadAffinityManager
{
public:
    ThreadAffinityManager() = default;

    // ★ [work62] ThreadAffinityManager 初期化 — affinity mask 設定 + initialized_ 有効化
    void initialize(const ThreadAffinityMasks& masks) noexcept
    {
        masks_ = masks;
        convo::publishAtomic(initialized_, true, std::memory_order_release);
    }

    [[nodiscard]] bool isInitialized() const noexcept
    {
        return convo::consumeAtomic(initialized_, std::memory_order_acquire);
    }

    void applyMessageThreadPolicy() const noexcept
    {
#ifdef _WIN32
        if (!convo::consumeAtomic(initialized_, std::memory_order_acquire))
            return;

        if (masks_.ui != 0)
        {
            ::SetThreadAffinityMask(::GetCurrentThread(), masks_.ui);
            ::SetThreadPriority(::GetCurrentThread(), THREAD_PRIORITY_NORMAL);
        }
#endif
    }

    void applyCurrentThreadPolicy(ThreadType type, int evalWorkerIndex = 0) const noexcept
    {
#ifdef _WIN32
        if (!convo::consumeAtomic(initialized_, std::memory_order_acquire))
            return;

        DWORD_PTR mask = 0;
        int priority = THREAD_PRIORITY_NORMAL;

        switch (type)
        {
            case ThreadType::Worker:
                mask = masks_.worker;
                priority = THREAD_PRIORITY_ABOVE_NORMAL;
                break;
            case ThreadType::LearnerMain:
                mask = masks_.learnerMain;
                priority = THREAD_PRIORITY_NORMAL;
                break;
            case ThreadType::LearnerEval:
                mask = getEvalWorkerMask(evalWorkerIndex);
                priority = THREAD_PRIORITY_BELOW_NORMAL;
                break;
            case ThreadType::HeavyBackground:
                mask = masks_.heavyBackground;
                priority = THREAD_PRIORITY_BELOW_NORMAL;
                break;
            case ThreadType::LightBackground:
                mask = masks_.lightBackground;
                priority = THREAD_PRIORITY_LOWEST;
                break;
            case ThreadType::UI:
                mask = masks_.ui;
                priority = THREAD_PRIORITY_NORMAL;
                break;
        }

        if (mask != 0)
            ::SetThreadAffinityMask(::GetCurrentThread(), mask);

        ::SetThreadPriority(::GetCurrentThread(), priority);
#else
        juce::ignoreUnused(type, evalWorkerIndex);
#endif
    }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ThreadAffinityManager)

private:
#ifdef _WIN32
    DWORD_PTR getEvalWorkerMask(int workerIndex) const noexcept
    {
        const DWORD_PTR base = masks_.learnerEvalBase;
        if (base == 0)
            return 0;

        std::array<DWORD_PTR, 64> bits{};
        size_t count = 0;

        for (DWORD_PTR bit = 1; bit != 0; bit <<= 1)
        {
            if ((base & bit) != 0)
            {
                bits[count++] = bit;
                if (count >= bits.size())
                    break;
            }
        }

        if (count == 0)
            return 0;

        const int normalized = workerIndex < 0 ? 0 : workerIndex;
        return bits[static_cast<size_t>(normalized) % count];
    }
#endif

    ThreadAffinityMasks masks_;
    std::atomic<bool> initialized_{false};
};
