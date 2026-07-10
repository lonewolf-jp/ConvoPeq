#!/usr/bin/env python3
"""Insert MEM_SNAP log after publishWorld in AudioEngine.Timer.cpp."""

with open('src/audioengine/AudioEngine.Timer.cpp', 'r', newline='') as f:
    content = f.read()

old = '            coordinator.publishWorld(std::move(worldOwner));\n        }\n\n        sendChangeMessage();'
new_code = '''            coordinator.publishWorld(std::move(worldOwner));
        }

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
        // ★ work70: [MEM_SNAP] Publish snapshot
        {
            const uint64_t gen = (runtimeWorld != nullptr) ? static_cast<uint64_t>(runtimeWorld->generation) : 0;
            const uint64_t nucBytes = convo::diag::allocatedBytes();
            const uint64_t nucPeak  = convo::diag::peakBytes();
            const uint64_t nucTotalA = convo::diag::totalAllocBytes();
            const uint64_t nucTotalF = convo::diag::totalFreedBytes();
            const uint32_t lostFree = convo::diag::lostFreeCount();
            const uint32_t curZeroAlloc = convo::diag::zeroAllocSizeCount();
            static uint32_t lastZeroAlloc = 0;
            const int32_t deltaZero = static_cast<int32_t>(curZeroAlloc) - static_cast<int32_t>(lastZeroAlloc);
            lastZeroAlloc = curZeroAlloc;
            auto osMem = getProcessMemoryInfo();
            const uint64_t retireBytes = m_retireRouter ? m_retireRouter->pendingRetireBytes() : 0;
            const uint32_t pendingCount = m_retireRouter ? m_retireRouter->pendingRetireCount() : 0;
            const double trackedRatio = m_retireRouter ? m_retireRouter->trackedRatio() : 0.0;
            const uint64_t overflow = m_retireRouter ? m_retireRouter->overflowCount() : 0;
            const uint64_t reclaim = m_retireRouter ? m_retireRouter->reclaimAttemptCount() : 0;
            const uint32_t nucLive = (uint32_t)convo::MKLNonUniformConvolver::liveCount.load(std::memory_order_relaxed);
            const uint64_t otherPrivate = convo::diag::computeOtherPrivate(osMem.privateUsageMB, nucBytes, retireBytes);
            juce::Logger::writeToLog(juce::String::formatted(
                "[MEM_SNAP] PUBLISH gen=%llu | NUC: live=%u alloc=%.0fMB peak=%.0fMB tA=%.0fGB tF=%.0fGB lost=%u zero=%u(d=%+d) | Ret: pend=%u trBytes=%.1fMB tr=%u/%u(%.0f%%) ovf=%llu rec=%llu | Priv=%lluMB WS=%lluMB | Other=%.0fMB",
                (unsigned long long)gen, (unsigned)nucLive,
                nucBytes/(1024.0*1024.0), nucPeak/(1024.0*1024.0),
                nucTotalA/(1024.0*1024.0*1024.0), nucTotalF/(1024.0*1024.0*1024.0),
                (unsigned)lostFree, (unsigned)curZeroAlloc, (int)deltaZero,
                (unsigned)pendingCount, retireBytes/(1024.0*1024.0),
                0u, (unsigned)pendingCount, trackedRatio*100.0,
                (unsigned long long)overflow, (unsigned long long)reclaim,
                (unsigned long long)osMem.privateUsageMB, (unsigned long long)osMem.workingSetMB,
                otherPrivate/(1024.0*1024.0)));
        }
#endif

        sendChangeMessage();'''

assert old in content, 'ERROR: old string not found!'
content = content.replace(old, new_code, 1)

with open('src/audioengine/AudioEngine.Timer.cpp', 'w', newline='') as f:
    f.write(content)
print('OK - MEM_SNAP added successfully')
