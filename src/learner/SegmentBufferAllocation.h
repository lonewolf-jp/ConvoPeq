//============================================================================
#pragma once
// SegmentBufferAllocation.h ── AudioSegmentBuffer::create() の例外を外へ出さない境界
//
//   create() は確保に失敗すると std::bad_alloc を送出する（B-1 の契約）。
//   NoiseShaperLearner のコンストラクタは「確保に失敗したら Status::Error へ落ちて
//   戻る」設計なので、例外はここで受けて nullptr に変える（#265）。
//
//   nothrow 版（makeAlignedArray_nothrow）と意味論を揃えるための境界である。
//   呼出側は nullptr だけを見ればよい。
//============================================================================

#include "AudioSegmentBuffer.h"

#include <memory>
#include <new>

namespace convo::learner
{

// 確保関数を注入できる形。テストは std::bad_alloc を投げる関数を渡す。
//   契約: 確保関数は失敗時に std::bad_alloc を送出する（nullptr を返さない）。
template <typename Factory>
inline std::unique_ptr<AudioSegmentBuffer> createSegmentBufferNoThrow (Factory&& factory) noexcept
{
    try
    {
        return factory();
    }
    catch (const std::bad_alloc&)
    {
        return nullptr;
    }
}

inline std::unique_ptr<AudioSegmentBuffer> createSegmentBufferNoThrow() noexcept
{
    return createSegmentBufferNoThrow ([] { return AudioSegmentBuffer::create(); });
}

} // namespace convo::learner
