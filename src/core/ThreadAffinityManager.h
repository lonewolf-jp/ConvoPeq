#pragma once

#include <JuceHeader.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
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

    bool isInitialized() const noexcept
    {
#ifdef _WIN32
        return (worker | learnerMain | learnerEvalBase | heavyBackground | lightBackground | ui) != 0;
#else
        return false;
#endif
    }
};

class ThreadAffinityManager
{
public:
    ThreadAffinityManager() = default;

    bool initialize() noexcept
    {
#ifdef _WIN32
        if (initialized_.load(std::memory_order_acquire))
            return true;

        std::vector<PhysicalCoreInfo> cores;
        if (!getPhysicalCoreInfo(cores))
            return initializeFallback();

        std::vector<DWORD_PTR> pCoreMasks;
        std::vector<DWORD_PTR> eCoreMasks;
        pCoreMasks.reserve(cores.size());
        eCoreMasks.reserve(cores.size());

        for (const auto& core : cores)
        {
            if (core.mask == 0)
                continue;

            if (core.efficiencyClass >= 2)
                pCoreMasks.push_back(core.mask);
            else if (core.efficiencyClass == 1)
                eCoreMasks.push_back(core.mask);
            else
                pCoreMasks.push_back(core.mask);
        }

        if (pCoreMasks.empty() && !eCoreMasks.empty())
            pCoreMasks = eCoreMasks;

        if (pCoreMasks.empty() && eCoreMasks.empty())
            return initializeFallback();

        assignMasks(pCoreMasks, eCoreMasks);
        initialized_.store(masks_.isInitialized(), std::memory_order_release);
        return initialized_.load(std::memory_order_acquire);
#else
        return false;
#endif
    }

    bool isInitialized() const noexcept
    {
        return initialized_.load(std::memory_order_acquire);
    }

    const ThreadAffinityMasks& getMasksForDebug() const noexcept
    {
        return masks_;
    }

    void applyMessageThreadPolicy() const noexcept
    {
#ifdef _WIN32
        if (!isInitialized())
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
        if (!isInitialized())
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
                priority = THREAD_PRIORITY_NORMAL;
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
    struct PhysicalCoreInfo
    {
        DWORD_PTR mask = 0;
        BYTE efficiencyClass = 0;
    };

    static bool getPhysicalCoreInfo(std::vector<PhysicalCoreInfo>& outCores) noexcept
    {
        DWORD bufferSize = 0;
        ::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufferSize);
        if (bufferSize == 0)
            return false;

        std::vector<BYTE> buffer(bufferSize);
        auto* base = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (!::GetLogicalProcessorInformationEx(RelationProcessorCore, base, &bufferSize))
            return false;

        BYTE* ptr = buffer.data();
        BYTE* end = ptr + bufferSize;

        while (ptr < end)
        {
            auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
            if (current->Relationship == RelationProcessorCore)
            {
                const auto& proc = current->Processor;
                const WORD groupCount = proc.GroupCount;
                const auto groupMask = proc.GroupMask;

                if (groupCount > 0)
                {
                    // v2.1: single processor-group support only.
                    DWORD_PTR mask = groupMask[0].Mask;
                    if (mask != 0)
                        outCores.push_back({ mask, proc.EfficiencyClass });
                }
            }

            if (current->Size == 0)
                break;

            ptr += current->Size;
        }

        return !outCores.empty();
    }

    void assignMasks(const std::vector<DWORD_PTR>& pCoreMasks,
                     const std::vector<DWORD_PTR>& eCoreMasks) noexcept
    {
        std::vector<DWORD_PTR> p = pCoreMasks;
        std::vector<DWORD_PTR> e = eCoreMasks;

        auto choose = [](const std::vector<DWORD_PTR>& v, size_t idx) -> DWORD_PTR {
            return idx < v.size() ? v[idx] : 0;
        };

        masks_.worker = choose(p, 0);
        masks_.ui = choose(p, 1);
        masks_.learnerMain = choose(p, 2);

        if (!e.empty())
        {
            for (auto m : e)
                masks_.learnerEvalBase |= m;
        }
        else
        {
            for (size_t i = 3; i < p.size(); ++i)
                masks_.learnerEvalBase |= p[i];
        }

        masks_.heavyBackground = masks_.learnerEvalBase != 0 ? masks_.learnerEvalBase : masks_.worker;

        DWORD_PTR processMask = 0;
        DWORD_PTR systemMask = 0;
        if (::GetProcessAffinityMask(::GetCurrentProcess(), &processMask, &systemMask))
            masks_.lightBackground = processMask;
        else
            masks_.lightBackground = masks_.heavyBackground;

        if (masks_.worker == 0)
            masks_.worker = masks_.lightBackground;
        if (masks_.ui == 0)
            masks_.ui = masks_.worker;
        if (masks_.learnerMain == 0)
            masks_.learnerMain = masks_.worker;
        if (masks_.learnerEvalBase == 0)
            masks_.learnerEvalBase = masks_.lightBackground;
        if (masks_.heavyBackground == 0)
            masks_.heavyBackground = masks_.lightBackground;
    }

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

    bool initializeFallback() noexcept
    {
        DWORD_PTR processMask = 0;
        DWORD_PTR systemMask = 0;
        if (!::GetProcessAffinityMask(::GetCurrentProcess(), &processMask, &systemMask) || processMask == 0)
            return false;

        masks_.worker = processMask;
        masks_.learnerMain = processMask;
        masks_.learnerEvalBase = processMask;
        masks_.heavyBackground = processMask;
        masks_.lightBackground = processMask;
        masks_.ui = processMask;

        initialized_.store(true, std::memory_order_release);
        return true;
    }
#endif

    ThreadAffinityMasks masks_;
    std::atomic<bool> initialized_{false};
};
