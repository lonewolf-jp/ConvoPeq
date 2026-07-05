#pragma once

#include <JuceHeader.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <vector>

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
    UI,
    AudioRealtime  // ★ [work64] 将来の拡張性のため。現在 AudioThread の affinity は
                   //   applyMmcssPriority() 末尾で直接 SetThreadAffinityMask しているため、
                   //   本 enum を applyCurrentThreadPolicy() で使うと二重適用になる。
                   //   二重適用自体は無害（同一マスクの再設定）だが、責務は
                   //   applyMmcssPriority() (Timer.cpp) 側にあり、
                   //   applyCurrentThreadPolicy は将来のリファクタリング用に用意。
                   // ★ v19: convo::numeric_policy::ThreadRole::AudioRealtime (DspNumericPolicy.h)
                   //   とは別概念。そちらはランタイムスレッド検出（assertion用）であり、
                   //   CPUアフィニティ管理とは無関係。名前空間が異なるため衝突なし。
};

struct ThreadAffinityMasks
{
#ifdef _WIN32
    DWORD_PTR worker = 0;
    DWORD_PTR learnerMain = 0;
    DWORD_PTR learnerEvalBase = 0;
    DWORD_PTR heavyBackground = 0;
    DWORD_PTR audioRealtime = 0;  // ★ [work64] Audioスレッド専用コアマスク
    DWORD_PTR lightBackground = 0;
    DWORD_PTR ui = 0;
#else
    std::uint64_t worker = 0;
    std::uint64_t learnerMain = 0;
    std::uint64_t learnerEvalBase = 0;
    std::uint64_t heavyBackground = 0;
    std::uint64_t audioRealtime = 0;
    std::uint64_t lightBackground = 0;
    std::uint64_t ui = 0;
#endif
};

// ★ [work64 v15] 物理コア情報: mask (SMT兄弟含む論理プロセッサ集合) + efficiencyClass (P/E判定用)
struct PhysicalCoreInfo {
    DWORD_PTR mask;
    BYTE efficiencyClass;
};

// ★ [work64 v15] コアトポロジ情報 (detectCoreTopology の戻り値)
struct CoreTopology {
    int physicalCoreCount = 0;
    bool hasHeterogeneousArchitecture = false;
    std::vector<PhysicalCoreInfo> cores;  // mask 最下位ビット順にソート済み
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
            case ThreadType::AudioRealtime:
                // ★ [work64] MMCSS が優先度管理済みのため SetThreadPriority を呼ばず return
                mask = masks_.audioRealtime;
                if (mask != 0)
                    ::SetThreadAffinityMask(::GetCurrentThread(), mask);
                return;
        }

        // ★ v17: prevMask を取得（診断ログ有効時に追跡可能にするため）
        if (mask != 0) {
            const DWORD_PTR prevMask = ::SetThreadAffinityMask(::GetCurrentThread(), mask);
            juce::ignoreUnused(prevMask);
        }

        ::SetThreadPriority(::GetCurrentThread(), priority);
#else
        juce::ignoreUnused(type, evalWorkerIndex);
#endif
    }

    // ★ [work64] AudioRealtime マスクアクセサ
    [[nodiscard]] DWORD_PTR getAudioRealtimeMask() const noexcept {
        return masks_.audioRealtime;
    }

    // ★ [work64] コアトポロジ検出（static メソッド / 起動時に1度だけ呼ばれる）
    static CoreTopology detectCoreTopology() noexcept
    {
        // ★ v20: noexcept を維持する。起動時初期化であり、
        //   メモリ確保失敗は回復不能と判断している。
        CoreTopology topo;
#ifdef _WIN32
        // 1. バッファサイズ取得
        DWORD bufLen = 0;
        if (!::GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bufLen)
            && ::GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            // API 未対応 → 空の topo（呼び出し側が physicalCoreCount==0 で検出）
            return topo;
        }

        // 2. バッファ確保・取得
        std::vector<BYTE> buf(bufLen);
        if (!::GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
            &bufLen)) {
            return topo;
        }

        // 3. 可変長レコードを走査
        DWORD offset = 0;
        while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bufLen) {
            auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                buf.data() + offset);
            if (info->Size == 0) break;
            if (offset + info->Size > bufLen) break;

            if (info->Relationship == RelationProcessorCore) {
                const auto& proc = info->Processor;
                // ★ v21: Mask!=0 の防御チェック
                if (proc.GroupCount == 1 && proc.GroupMask[0].Mask != 0) {
                    topo.cores.push_back({proc.GroupMask[0].Mask, proc.EfficiencyClass});
                }
            }
            offset += info->Size;
        }

        topo.physicalCoreCount = static_cast<int>(topo.cores.size());

        // 4. 最下位ビット（論理CPU番号）順にソート
        std::sort(topo.cores.begin(), topo.cores.end(),
            [](const PhysicalCoreInfo& a, const PhysicalCoreInfo& b) noexcept {
                auto lowestBit = [](DWORD_PTR mask) noexcept -> int {
                    // ★ v20: C++20 std::countr_zero（<bit> ヘッダ）
                    return mask == 0 ? 64 : static_cast<int>(std::countr_zero(mask));
                };
                return lowestBit(a.mask) < lowestBit(b.mask);
            });

        // 5. P/E混在判定: 全コアの EfficiencyClass が同一か検査
        if (topo.physicalCoreCount > 1) {
            const BYTE first = topo.cores.empty() ? 0 : topo.cores[0].efficiencyClass;
            for (const auto& core : topo.cores) {
                if (core.efficiencyClass != first) {
                    topo.hasHeterogeneousArchitecture = true;
                    break;
                }
            }
        }
#endif
        return topo;
    }

    // ★ [work64] 対称コア環境のマスク計算（static メソッド）
    static ThreadAffinityMasks computeSymmetricMasks(const CoreTopology& topo) noexcept
    {
        ThreadAffinityMasks m{};
        // ★ v20: cores.size() を計算基準に使用
        const size_t N = topo.cores.size();
        if (N < 2)
            return m;

        // ★ v17: 整合性チェック（万が一の防御）
        if (static_cast<size_t>(topo.physicalCoreCount) != N)
            return m;

        DWORD_PTR nonAudioMask = 0;
        for (size_t i = 0; i < N - 1; ++i)
            nonAudioMask |= topo.cores[i].mask;

        m.audioRealtime   = topo.cores[N - 1].mask;
        m.worker          = topo.cores[0].mask;
        m.learnerMain     = topo.cores[std::min(size_t{1}, N - 2)].mask;
        m.learnerEvalBase = nonAudioMask;
        m.heavyBackground = nonAudioMask;
        m.lightBackground = nonAudioMask;
        m.ui              = nonAudioMask;
        return m;
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
