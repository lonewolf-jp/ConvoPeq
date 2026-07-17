#pragma once

#include <cstdint>

//==============================================================================
// IRetireRouter.h — 最小限の Retire 抽象 Interface
//
// ★ R-1: RefCountedDeferred の Retire Router 統合（抽象 Interface 経由）
//   「Retire してください」だけを表現する。リトライ戦略・epoch・削除分類は
//   すべて Router 内部に隠蔽され、呼び出し元（RefCountedDeferred）はそれらを知らない。
//   QueuePressure が発生した場合、Router が RuntimeHealthMonitor へ通知する責務を持つ。
//
// この Interface は src/core/ に配置され、src/audioengine/ISRRetireRouter が実装する。
// これにより Core Utility 層が AudioEngine 層に依存する Layering 違反を防止する。
//==============================================================================

namespace convo {

class IRetireRouter {
public:
    virtual ~IRetireRouter() = default;

    // RT-safe: 単発 enqueue、リトライなし。戻り値で成否を呼び出し元に伝える。
    // QueueFull 時は呼び出し元（AudioEngine）が後続処理を判断する。
    // NonRT の retire() と異なり、RT はリトライ不可のため bool を返す。
    virtual bool retireRT(
        void* ptr, void (*deleter)(void*)) noexcept = 0;

    // NonRT: リトライ込みの retire。Router 内部で tryReclaim + 再試行を行い、
    // 最終失敗時に QueuePressure を RuntimeHealthMonitor へ通知する。
    // 呼び出し元はリトライ回数・方針を一切知らない。戻り値不要のため void。
    virtual void retire(
        void* ptr, void (*deleter)(void*)) noexcept = 0;
};

} // namespace convo
