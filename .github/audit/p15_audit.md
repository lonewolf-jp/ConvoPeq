# P15 RuntimeStore Mutation Authority Audit Report

AUDIT_RESULT: PASS
AUDIT_DATE: 2026-06-05
AUDITOR: GitHub Copilot (AI Assistant)
CHECKED_SYMBOLS: RuntimePublicationCoordinator::publishWorld, RuntimeStore::publishAndSwap, RuntimeStore::acquireWriteAccess, RuntimeStore::WriteAccess, convo::RuntimePublicationCoordinator (template), convo::isr::RuntimePublicationCoordinator (ISR), RetireRuntime::enqueueRetire, EpochDomain::enqueueRetire
FINDINGS: RuntimeStore Mutation Authority Audit（capability 発行主体ベース）。RuntimeStore（template coordinator 版）の mutation は WriteAccess を通じてのみ行われる。WriteAccess は RuntimePublicationCoordinator::create() 内で Store::acquireWriteAccess() からのみ発行される。Coordinator 以外が独自に WriteAccess を生成したり、Coordinator の関与なしに mutation を実行する経路は存在しない。ISR 版（convo::isr::RuntimePublicationCoordinator）の mutation も commit()/retire() メソッド経由でのみ行われ、外部から直接 atomic を書き換える経路は存在しない（setPublicationBacklogCount/setPendingIntentCount 等は監視用カウンタであり、RuntimeStore の mutation ではない）。RetireRuntime::enqueueRetire は coordinator.enqueueRetire() からのみ呼ばれ、deprecated 警告付きで EpochDomain::enqueueRetire に委譲するが、これは coordinator 経由の委譲パターンに該当する。ただし新規コードでは coordinator.enqueueRetire() 経由を推奨（P3 参照）。
SEARCH_COMMANDS: serena query "callers of RuntimeStore::acquireWriteAccess", serena query "callers of WriteAccess constructor", grep -r "WriteAccess" src/core/RuntimeStore.h, grep -r "acquireWriteAccess" src/**, grep -r "publishAndSwap" src/**, grep -r "enqueueRetire" src/**, codegraph analysis for WriteAccess capability flow
