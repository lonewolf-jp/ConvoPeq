#pragma once

// ★ WORK111: Test/measurement-only hook.
//   production では誰も true にしない（既定 false）。RT 経路からは一切参照しない。
//   NonRT の LoaderThread::doTrimStep() のみが参照する。
//   IRRuntimeContract / engine / limiter / gain には関与しない。
#include <atomic>

namespace convo
{
namespace trimtest
{
inline std::atomic<bool>& disableTailFadeForMeasurement() noexcept
{
    static std::atomic<bool> flag { false };
    return flag;
}
} // namespace trimtest
} // namespace convo
