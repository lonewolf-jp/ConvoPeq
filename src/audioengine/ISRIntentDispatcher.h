#pragma once
#include <cstddef>
#include <type_traits>
#include "ISRRuntimePublicationCoordinator.h"  // Intent/IntentType/kIntentTypeCount (public nested types of RuntimePublicationCoordinator)

class AudioEngine;
class DSPLifetimeManager;

namespace convo::isr {

class DSPQuarantineManager; // ★ A3: pulled in Step 4 (Quarantine handler)
class DSPTransition;              // ★ A3 Step 5-3: stateless publish-completion facade (ADR-D2)

// ★ A3 Step 1: aliases for the Intent-type family (public nested members of RuntimePublicationCoordinator).
using Intent     = RuntimePublicationCoordinator::Intent;
using IntentType = RuntimePublicationCoordinator::IntentType;

// ★ A3 Step 1: sole execution context for intent handlers (HANDLER-1).
// Coordinator hands this to the DispatchTable; handlers hold NO decision/policy
// and perform NO world-write — they delegate to existing domain executors only.
struct IntentHandlerContext {
    AudioEngine& engine;
    DSPLifetimeManager& lifetimeMgr;
    QuarantineService& quarantine;  // ★ A3 Step 4: QSVC-2 execution boundary (handlers never bypass)
    DSPTransition& transition;      // ★ A3 Step 5-3: stateless publish-completion facade (ADR-D2) / Completion layer
};

// ★ A3 Step 1: IntentHandler interface — one concrete handler per IntentType.
// HANDLER-1: handlers are stateless singletons (g_*IntentHandler) dispatched through a
// const IntentHandler* table (kDispatchTable); therefore handle() is const — handlers mutate
// domain state only via IntentHandlerContext, never via 'this'.
struct IntentHandler {
    virtual ~IntentHandler() = default;
    virtual void handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept = 0;
};

struct ObserveIntentHandler final : IntentHandler {
    void handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept override; // A3 Step 3
};
struct PublishIntentHandler final : IntentHandler {
    void handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept override; // A3 Step 5-2: → PublishExecutor
};
struct RecoveryIntentHandler final : IntentHandler {
    void handle(const Intent&, IntentHandlerContext&) const noexcept override {} // A3 Step 5: → Recovery path
};
struct QuarantineIntentHandler final : IntentHandler {
    void handle(const Intent& intent, IntentHandlerContext& ctx) const noexcept override; // A3 Step 4: → QuarantineService
};

constexpr ObserveIntentHandler    g_observeIntentHandler{};
constexpr PublishIntentHandler    g_publishIntentHandler{};
constexpr RecoveryIntentHandler   g_recoveryIntentHandler{};
constexpr QuarantineIntentHandler g_quarantineIntentHandler{};

// ★ A3 Step 1: constexpr 1:1 total mapping IntentType -> IntentHandler (DISPATCH-1).
// Indexing by static_cast<std::size_t>(intent.type); the static_assert guarantees
// a new IntentType cannot be silently dropped from the table.
constexpr const IntentHandler* kDispatchTable[RuntimePublicationCoordinator::kIntentTypeCount] = {
    &g_observeIntentHandler,   // IntentType::Observe   (0)
    &g_publishIntentHandler,   // IntentType::Publish   (1)
    &g_recoveryIntentHandler,  // IntentType::Recovery  (2)
    &g_quarantineIntentHandler // IntentType::Quarantine(3)
};
static_assert(std::size(kDispatchTable) == RuntimePublicationCoordinator::kIntentTypeCount,
    "QUEUE-22/DISPATCH-1: kDispatchTable must be a 1:1 total mapping over IntentType "
    "(pure routing; Dispatcher has no decision)");

// ★ BUG-FIX: DispatcherHasNoDecision — HANDLER-1 契約のコンパイル時強制。
//   既存 static_assert は dispatch table の 1:1 サイズのみを検証しており、
//   「ハンドラ自身が decision/policy 状態を持たない」ことは未検証だった。
//   handlers は polymorphic singleton（vptr を持つ）ため std::is_empty_v と
//   std::is_trivially_default_constructible_v は使えない（vptr が非静的データメンバとして
//   扱われ、trivial 性が失われる）。そこで「sizeof(T) == sizeof(void*)（vptr のみ）＋
//   default-constructible」を以て「非静的データメンバなし＝状態なし」と判定する。
//   ハンドラに decision/policy フィールドを追加すると sizeof が増え、この assert が失敗する。
template <typename T>
struct HandlerIsStateless : std::bool_constant<
    std::is_polymorphic_v<T>
    && sizeof(T) == sizeof(void*)
    && std::is_default_constructible_v<T>> {};

inline constexpr bool DispatcherHasNoDecision =
    HandlerIsStateless<ObserveIntentHandler>::value
    && HandlerIsStateless<PublishIntentHandler>::value
    && HandlerIsStateless<RecoveryIntentHandler>::value
    && HandlerIsStateless<QuarantineIntentHandler>::value;
static_assert(DispatcherHasNoDecision,
    "HANDLER-1: intent handlers must be stateless singletons — adding decision/policy "
    "state violates the Dispatcher-has-no-decision contract");

} // namespace convo::isr
